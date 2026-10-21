#!/usr/bin/env python3
"""Combine edge-triggered PC-counter samples with regcmd task metadata.

Input files are produced by the rknpu ioctl hook in blockwatch edge mode:
  EPERF_RKNPU_BLOCKWATCH=1
  EPERF_RKNPU_BLOCKWATCH_EDGE=1
  EPERF_RKNPU_BLOCKWATCH_SELECT=5
  EPERF_RKNPU_BLOCKWATCH_OP_IDX=6
  EPERF_RKNPU_BLOCKWATCH_REQUIRE_ALL_TASKS_SAME_OP=1

The hook samples PC_TASK_STATUS + CNA/CORE/DPU/DPU_RDMA S_STATUS once whenever
the low 12-bit PC task counter changes.  Thus:
  * task index is exact (low 12 bits == op6 task index);
  * task start time is the sample timestamp of the counter transition;
  * module state is an instantaneous S_STATUS snapshot at that transition.
The four S_STATUS registers are read sequentially by the hook, not
simultaneously, so a boundary row is not a synchronous four-module sample.

The measurable task sequence ends at the post-task counter-zero sample.
The driver's hw_elapse_time is retained separately; it is not assigned to the
last task.  All busy fractions and concurrency times are zero-order-hold
estimates between task boundaries, not continuous occupancy integrals.

S_STATUS is decoded from the RK3588 TRM:
  status0 = S_STATUS[1:0], status1 = S_STATUS[17:16]
  2'd0 idle, 2'd1 operating, 2'd2 operating/waiting, 2'd3 reserved.
A module is marked busy when status0 or status1 is 1 or 2.  Bits [3:2] are
reserved in the TRM and must not be treated as busy.

The output intentionally keeps three states:
  busy    all successful edge runs say operating
  idle    all successful edge runs say idle
  mixed   the runs disagree (fast toggle / boundary race), rendered hatched
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from sdpa_operation_map import PHASE_CONFIDENCE, classify_operation

MODULES = [
    ("CNA", "c0_cna", "n_cna"),
    ("CORE", "c0_core", "n_core"),
    ("DPU", "c0_dpu", "n_dpu"),
    ("DPU_RDMA", "c0_rdma", "n_dpu_rdma"),
]


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def busy_value(value: str) -> bool:
    reg = int(value, 16)
    status0 = reg & 0x3
    status1 = (reg >> 16) & 0x3
    return status0 in (1, 2) or status1 in (1, 2)


def load_edge_run(path: Path) -> dict:
    rows = load_csv(path)
    if not rows:
        raise SystemExit(f"empty blockwatch file: {path}")
    t0 = int(rows[0]["t_ns"])
    first_us: dict[int, float] = {}
    samples: dict[int, list[dict]] = collections.defaultdict(list)
    parsed: list[tuple[int, int]] = []
    for row in rows:
        task_idx = int(row["c0_ts"], 16) & 0xFFF
        t_ns = int(row["t_ns"])
        t_us = (t_ns - t0) / 1000.0
        first_us.setdefault(task_idx, t_us)
        samples[task_idx].append(row)
        parsed.append((t_ns, task_idx))

    # The op6 task sequence ends when the PC task counter returns to zero
    # after the last observed task.  The driver's hw_elapse_time can be longer
    # (interrupt/teardown overhead); it must not be assigned to the last task.
    last_task = max(task_idx for _t_ns, task_idx in parsed)
    last_task_ns = max(t_ns for t_ns, task_idx in parsed if task_idx == last_task)
    end_row = next(
        ((t_ns, t_us)
         for t_ns, task_idx in parsed
         if t_ns > last_task_ns and task_idx == 0),
        None,
    )
    if end_row is not None:
        sequence_end_us = (end_row[0] - t0) / 1000.0
        sequence_end_source = f"counter_zero_after_task_{last_task}"
    else:
        sequence_end_us = max((t_ns - t0) / 1000.0 for t_ns, _ in parsed)
        sequence_end_source = "last_sample_fallback"

    return {
        "path": str(path),
        "t0_ns": t0,
        "first_us": first_us,
        "samples": samples,
        "last_task_index": last_task,
        "sequence_end_us": sequence_end_us,
        "sequence_end_source": sequence_end_source,
    }


def parse_hw_elapse(meta_path: Path | None) -> int | None:
    if meta_path is None or not meta_path.exists():
        return None
    text = meta_path.read_text(encoding="utf-8", errors="replace")
    values = []
    for line in text.splitlines():
        if "BLOCKWATCH_META" in line and "result=0" in line:
            for part in line.split():
                if part.startswith("hw_elapse_ns="):
                    values.append(int(part.split("=", 1)[1]))
    return int(statistics.median(values)) if values else None


def interpolate_missing(first_us: dict[int, float]) -> tuple[dict[int, float], set[int]]:
    out = dict(first_us)
    estimated: set[int] = set()
    for task_idx in range(1, 401):
        if task_idx in out:
            continue
        lo = max(k for k in range(1, task_idx) if k in out)
        hi = min(k for k in range(task_idx + 1, 401) if k in out)
        out[task_idx] = out[lo] + (out[hi] - out[lo]) * (task_idx - lo) / (hi - lo)
        estimated.add(task_idx)
    return out, estimated


def classify_status(votes: list[bool]) -> str:
    if not votes:
        return "unknown"
    if all(votes):
        return "busy"
    if not any(votes):
        return "idle"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--edge", required=True, type=Path, nargs="+")
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    task_rows = [r for r in load_csv(args.tasks) if r.get("op_idx") == "6"]
    task_rows.sort(key=lambda r: int(r["local_task_index"]))
    if len(task_rows) != 401:
        raise SystemExit(f"expected 401 op6 tasks, got {len(task_rows)}")

    runs = [load_edge_run(p) for p in args.edge]
    if not runs:
        raise SystemExit("no edge runs")

    # Timing comes from the first run.  It normally sees 400/401 tasks; the
    # one short task missed by a counter transition is linearly interpolated.
    # The measurable task sequence ends at the post-task counter-zero sample,
    # not at the driver's hw_elapse_time (which includes unattributed tail).
    first_us, estimated = interpolate_missing(runs[0]["first_us"])
    first_us[0] = 0.0
    total_us = float(runs[0]["sequence_end_us"])
    if total_us <= 0:
        raise SystemExit("first edge run has no positive sequence end time")
    first_us[401] = total_us
    hw_ns = parse_hw_elapse(args.meta)
    hw_elapse_us = hw_ns / 1000.0 if hw_ns is not None else None
    unattributed_tail_us = (
        round(hw_elapse_us - total_us, 3) if hw_elapse_us is not None else None
    )

    # One vote per successful run for each (task, module).  Task 0 is mostly
    # idle-prefix samples and inter-task zero glitches; leave it unknown.
    state: dict[int, dict[str, str]] = {}
    for task_idx in range(401):
        state[task_idx] = {}
        for module, column, _ in MODULES:
            if task_idx == 0:
                state[task_idx][module] = "unknown"
                continue
            votes: list[bool] = []
            for run in runs:
                samples = run["samples"].get(task_idx, [])
                if samples:
                    votes.append(busy_value(samples[0][column]))
            state[task_idx][module] = classify_status(votes)

    out_rows: list[dict] = []
    for task_idx, task in enumerate(task_rows):
        start = first_us.get(task_idx, 0.0)
        end = first_us.get(task_idx + 1, total_us)
        duration = max(end - start, 0.0)
        modules_touched = [m for m, _, ncol in MODULES if int(task.get(ncol) or 0) > 0]
        phase = classify_operation(task)
        start_estimated = task_idx in estimated
        end_estimated = (task_idx + 1) in estimated
        start_time_source = "interpolated" if start_estimated else "measured"
        end_time_source = "interpolated" if end_estimated else "measured"
        if task_idx == 400:
            end_time_source = "measured_counter_zero"
        if start_estimated and end_estimated:
            time_source = "start_end_interpolated"
        elif start_estimated:
            time_source = "start_interpolated"
        elif end_estimated:
            time_source = "end_interpolated"
        elif task_idx == 400:
            time_source = "measured_end_counter_zero"
        else:
            time_source = "measured"
        row = {
            "task_index": task_idx,
            "head": task.get("head_group", ""),
            "head_task_order": task.get("head_task_order", ""),
            "stage": task.get("stage", ""),
            "task_class": task.get("task_class", ""),
            "math_phase": phase,
            "phase_confidence": PHASE_CONFIDENCE.get(phase, "conservative"),
            "start_us": round(start, 3),
            "end_us": round(end, 3),
            "duration_us": round(duration, 3),
            "time_source": time_source,
            "start_time_source": start_time_source,
            "end_time_source": end_time_source,
            "duration_depends_on_interpolation": int(start_estimated or end_estimated),
            "n_cna": task.get("n_cna", "0"),
            "n_core": task.get("n_core", "0"),
            "n_dpu": task.get("n_dpu", "0"),
            "n_dpu_rdma": task.get("n_dpu_rdma", "0"),
            "modules_touched": ",".join(modules_touched),
        }
        for module, column, _ in MODULES:
            s = state[task_idx][module]
            row[f"{module.lower()}_state"] = s
            row[f"{module.lower()}_busy"] = "1" if s == "busy" else ("0.5" if s == "mixed" else "0")
        states = [state[task_idx][module] for module, _, _ in MODULES]
        has_unknown = any(value == "unknown" for value in states)
        row["busy_blocks_observation"] = "unknown" if has_unknown else "known"
        row["busy_blocks"] = "" if has_unknown else (
            sum(1 for value in states if value == "busy")
            + 0.5 * sum(1 for value in states if value == "mixed")
        )
        out_rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Summary statistics.
    duration_by_task = {int(r["task_index"]): float(r["duration_us"]) for r in out_rows}
    module_summary = {}
    for module, _, ncol in MODULES:
        busy_us = sum(duration_by_task[k] for k, r in enumerate(out_rows) if r[f"{module.lower()}_state"] == "busy")
        mixed_us = sum(duration_by_task[k] for k, r in enumerate(out_rows) if r[f"{module.lower()}_state"] == "mixed")
        idle_us = sum(duration_by_task[k] for k, r in enumerate(out_rows) if r[f"{module.lower()}_state"] == "idle")
        unknown_us = total_us - busy_us - mixed_us - idle_us
        expected = [k for k, t in enumerate(task_rows) if int(t.get(ncol) or 0) > 0]
        active = [k for k, r in enumerate(out_rows) if r[f"{module.lower()}_state"] in ("busy", "mixed")]
        module_summary[module] = {
            "boundary_hold_busy_us_estimate": round(busy_us, 3),
            "boundary_hold_mixed_us_estimate": round(mixed_us, 3),
            "boundary_hold_idle_us_estimate": round(idle_us, 3),
            "boundary_hold_unknown_us_estimate": round(unknown_us, 3),
            "boundary_hold_busy_fraction_estimate": round(busy_us / total_us, 6),
            "boundary_hold_busy_plus_half_mixed_fraction_estimate": round(
                (busy_us + 0.5 * mixed_us) / total_us, 6
            ),
            "tasks_with_this_module_in_regcmd": len(expected),
            "tasks_measured_active": len(active),
            "measured_active_but_no_regcmd_touch": sorted(set(active) - set(expected)),
            "regcmd_touch_but_boundary_sample_idle": sorted(set(expected) - set(active)),
            "edge_vote_disagreements": sum(1 for r in out_rows if r[f"{module.lower()}_state"] == "mixed"),
            "estimate_semantics": "task-boundary zero-order hold, not continuous occupancy",
        }

    head_summary = []
    for head_idx in range(16):
        start_task = 1 + 25 * head_idx
        end_task = min(26 + 25 * head_idx, 401)
        start_source = "interpolated" if start_task in estimated else "measured"
        if end_task == 401:
            end_source = "measured_counter_zero"
        else:
            end_source = "interpolated" if end_task in estimated else "measured"
        head_summary.append({
            "head": head_idx,
            "start_us": round(first_us[start_task], 3),
            "end_us": round(first_us[end_task], 3),
            "duration_us": round(first_us[end_task] - first_us[start_task], 3),
            "start_time_source": start_source,
            "end_time_source": end_source,
            "duration_depends_on_interpolation": int(
                start_task in estimated or end_task in estimated
            ),
        })

    concurrency: dict[str, dict] = {}
    for r in out_rows:
        if r["busy_blocks_observation"] == "unknown":
            key = "unknown"
        else:
            key = f"{float(r['busy_blocks']):.1f}"
        concurrency.setdefault(key, {"tasks": 0, "time_us": 0.0})
        concurrency[key]["tasks"] += 1
        concurrency[key]["time_us"] = round(concurrency[key]["time_us"] + float(r["duration_us"]), 3)

    phase_task_counts: dict[str, int] = {}
    phase_time_us: dict[str, float] = {}
    confidence_counts: dict[str, int] = {}
    for r in out_rows:
        phase = r["math_phase"]
        phase_task_counts[phase] = phase_task_counts.get(phase, 0) + 1
        phase_time_us[phase] = round(
            phase_time_us.get(phase, 0.0) + float(r["duration_us"]), 3
        )
        conf = r["phase_confidence"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    summary = {
        "source": "blockwatch edge mode, S_STATUS sampled at PC task-counter changes",
        "model": "sdpa_qwen224.rknn",
        "op": "op6 exSDPAttention",
        "edge_runs": [run["path"] for run in runs],
        "measured_counter_sequence_us": round(total_us, 3),
        "driver_hw_elapse_us": round(hw_elapse_us, 3) if hw_elapse_us is not None else None,
        "unattributed_driver_tail_us": unattributed_tail_us,
        "timing_end_source": runs[0]["sequence_end_source"],
        "task_count": len(task_rows),
        "timing_run": runs[0]["path"],
        "interpolated_task_indices": sorted(estimated),
        "interpolation_affected_intervals": (
            [
                "task50.end_us",
                "task51.start_us",
                "head1.end_us",
                "head2.start_us",
            ]
            if 51 in estimated else []
        ),
        "interpolation_affected_math_phases": (
            ["pv_matmul (task50)", "staging (task51)"]
            if 51 in estimated else []
        ),
        "heads": head_summary,
        "modules": module_summary,
        "math_phases": {
            "task_counts": phase_task_counts,
            "task_time_us": phase_time_us,
            "confidence_counts": confidence_counts,
        },
        "busy_blocks_boundary_hold_estimate_histogram": dict(
            sorted(
                concurrency.items(),
                key=lambda kv: (1, 0.0) if kv[0] == "unknown" else (0, float(kv[0])),
            )
        ),
        "busy_blocks_histogram_semantics": {
            "unknown": "at least one module had no usable boundary state (task0); not idle",
            "0.0": "all four modules had known idle boundary state",
            "numeric": "boundary-hold estimate; mixed votes count as 0.5",
        },
        "sampling_semantics": {
            "module_state": (
                "one instantaneous S_STATUS read per module, sequentially after each "
                "PC counter transition; not simultaneous and not continuous"
            ),
            "duration": (
                "PC counter transition intervals; module fractions are zero-order-hold "
                "estimates between boundaries, not integrated occupancy"
            ),
            "concurrency_histogram": (
                "boundary-hold estimate per task interval; not proof that the same "
                "module combination existed continuously inside the interval"
            ),
            "end_of_sequence": (
                "ends at the post-task counter-zero sample; driver hw_elapse_time is "
                "retained separately and not assigned to the final task"
            ),
            "interpolation": (
                "task51 start is linearly interpolated from task50/task52; this also "
                "sets task50 end and the H1/H2 boundary"
            ),
        },
        "status_encoding": {
            "status0_bits": "S_STATUS[1:0]",
            "status1_bits": "S_STATUS[17:16]",
            "idle": 0,
            "operating": [1, 2],
            "busy_rule": "status0 in {1,2} or status1 in {1,2}",
        },
        "limitations": [
            "S_STATUS is read sequentially once per module after a PC counter transition, not synchronously or continuously",
            "module fractions are boundary-hold estimates; they do not establish continuous occupancy between samples",
            "the histogram records no observed 4-module-busy boundary sample; it does not prove 4 modules were never simultaneously busy between samples",
            "task boundaries are PC task-counter transitions; they represent issue/execution phase, not a full per-engine cycle trace",
            "mixed means the two runs disagree at a boundary; it is rendered hatched, not silently resolved",
            "task51 start is interpolated and therefore task50 duration, H1 duration, two-/three-module bucket times, and H2 start all depend on interpolation",
            f"driver hw_elapse_time is {unattributed_tail_us} us longer than the measured counter sequence on this run; this tail is not attributed to task400",
            "softmax_exp_lut is validated by FP16 workspace dumps of the subtracted score and its exponential (H1 tasks 38/39)",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    print(
        f"measured_counter_sequence={total_us:.3f} us  "
        f"driver_hw_elapse={hw_elapse_us:.3f} us  "
        f"unattributed_tail={unattributed_tail_us:.3f} us"
        if hw_elapse_us is not None and unattributed_tail_us is not None
        else f"measured_counter_sequence={total_us:.3f} us"
    )
    for module, data in module_summary.items():
        print(
            f"{module:9s} boundary-hold busy-estimate="
            f"{data['boundary_hold_busy_fraction_estimate']*100:5.1f}% "
            f"(+half-mixed="
            f"{data['boundary_hold_busy_plus_half_mixed_fraction_estimate']*100:5.1f}%) "
            f"active_tasks={data['tasks_measured_active']:3d} "
            f"regcmd_tasks={data['tasks_with_this_module_in_regcmd']:3d} "
            f"mixed={data['edge_vote_disagreements']:2d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
