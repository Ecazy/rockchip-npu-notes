#!/usr/bin/env python3
"""Free-running S_STATUS analysis: module idle / overlap inside each submit.

Inputs (all produced on the board)
  fw_*.csv     one free-running blockwatch file per submit
               (EPERF_RKNPU_BLOCKWATCH=1, no EDGE: fixed-interval sampling)
  stamp.err    SUBMIT <enter_ns> <leave_ns> <ret> <tasks> lines from the minimal
               submit-only interposer, one inference = the first 18 lines

For every submit this reports
  * sample interval (p50/p95) and the per-sample read bracket (t_end - t)
    and the four-register group bracket (ge - gb): the S_STATUS reads are
    sequential, so a sample is a short window, not an instant;
  * busy share per module;
  * the four-way coverage split the figure is about:
        compute-only   CNA/CORE busy, DPU/RDMA idle
        dma-only       DPU/RDMA busy, CNA/CORE idle
        both           both groups busy
        all-idle       submit still running, all four report not-operating
    with the sampled time spent in each;
  * all-idle runs with their start offset and duration (the gaps worth
    explaining later with task/regcmd structure).

Output: gdn_module_gaps.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def busy(value: str) -> bool:
    v = int(value, 16)
    return (v & 0x3) in (1, 2) or ((v >> 16) & 0x3) in (1, 2)


def classify(row: dict) -> str:
    compute = busy(row["c0_cna"]) or busy(row["c0_core"])
    dma = busy(row["c0_dpu"]) or busy(row["c0_rdma"])
    if compute and dma:
        return "both"
    if compute:
        return "compute-only"
    if dma:
        return "dma-only"
    return "all-idle"


def runs_of(samples: list[dict], state: str) -> list[dict]:
    out: list[dict] = []
    current = None
    for smp in samples:
        if smp["class"] == state:
            if current is None:
                current = {"start_ms": smp["t_ms"], "end_ms": smp["t_ms"], "n": 1}
            else:
                current["end_ms"] = smp["t_ms"]
                current["n"] += 1
        elif current is not None:
            out.append(current)
            current = None
    if current is not None:
        out.append(current)
    for run in out:
        run["dur_ms"] = round(run["end_ms"] - run["start_ms"], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--stamp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen35_gdn_layer0 (monolithic RKNN graph)")
    args = ap.parse_args()

    stamps = []
    for line in args.stamp.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "SUBMIT":
            stamps.append({"enter_ns": int(parts[1]), "leave_ns": int(parts[2]),
                           "ret": int(parts[3]), "tasks": int(parts[4])})
    n_sub = 18
    stamps = stamps[:n_sub]
    t0 = stamps[0]["enter_ns"]

    submits = []
    for seq in range(1, n_sub + 1):
        path = args.samples_dir / f"fw_{seq}.csv"
        rows = list(csv.DictReader(path.open())) if path.exists() else []
        samples = []
        for row in rows:
            t_ns = int(row["t_ns"])
            samples.append(
                {
                    "t_ns": t_ns,
                    "t_ms": 0.0,
                    "dt_us": (int(row["t_end_ns"]) - t_ns) / 1000.0,
                    "grp_us": (int(row["c0_ge_ns"]) - int(row["c0_gb_ns"])) / 1000.0,
                    "class": classify(row),
                    "modules": {
                        "CNA": busy(row["c0_cna"]),
                        "CORE": busy(row["c0_core"]),
                        "DPU": busy(row["c0_dpu"]),
                        "RDMA": busy(row["c0_rdma"]),
                    },
                }
            )
        enter = int(rows[0]["submit_enter_ns"]) if rows else stamps[seq - 1]["enter_ns"]
        for smp in samples:
            smp["t_ms"] = (smp["t_ns"] - enter) / 1e6
        intervals = [(samples[i + 1]["t_ns"] - samples[i]["t_ns"]) / 1000.0 for i in range(len(samples) - 1)]
        durations = [s["dt_us"] for s in samples]
        groups = [s["grp_us"] for s in samples]
        classes = {k: sum(1 for s in samples if s["class"] == k)
                   for k in ("compute-only", "dma-only", "both", "all-idle")}
        span = samples[-1]["t_ms"] if samples else 0.0
        per_class_ms = {k: round(v / max(len(samples), 1) * span, 3) for k, v in classes.items()}
        submits.append(
            {
                "submit_seq": seq,
                "tasks": stamps[seq - 1]["tasks"],
                "submit_start_ms": round((stamps[seq - 1]["enter_ns"] - t0) / 1e6, 3),
                "submit_wall_ms": round((stamps[seq - 1]["leave_ns"] - stamps[seq - 1]["enter_ns"]) / 1e6, 3),
                "samples": len(samples),
                "sampled_span_ms": round(span, 3),
                "interval_p50_us": round(statistics.median(intervals), 2) if intervals else 0.0,
                "interval_p95_us": round(sorted(intervals)[int(0.95 * len(intervals))], 2) if len(intervals) > 2 else 0.0,
                "read_bracket_p50_us": round(statistics.median(durations), 2) if durations else 0.0,
                "read_bracket_max_us": round(max(durations), 2) if durations else 0.0,
                "group_bracket_p50_us": round(statistics.median(groups), 2) if groups else 0.0,
                "busy_share": {
                    m: round(sum(1 for s in samples if s["modules"][m]) / max(len(samples), 1), 4)
                    for m in ("CNA", "CORE", "DPU", "RDMA")
                },
                "class_share": {k: round(v / max(len(samples), 1), 4) for k, v in classes.items()},
                "class_ms": per_class_ms,
                "all_idle_runs": runs_of(samples, "all-idle"),
                "timeline": [
                    {"t_ms": round(s["t_ms"], 4), "class": s["class"], "modules": s["modules"]}
                    for s in samples
                ],
            }
        )

    total_wall = sum(s["submit_wall_ms"] for s in submits)
    gaps = []
    for i in range(len(submits) - 1):
        a = stamps[i]
        b = stamps[i + 1]
        gaps.append(
            {
                "after_submit": i + 1,
                "gap_ms": round((b["enter_ns"] - a["leave_ns"]) / 1e6, 3),
                "at_ms": round((a["leave_ns"] - t0) / 1e6, 3),
            }
        )
    summary = {
        "model": args.model,
        "submits": len(submits),
        "inference_wall_ms": round((stamps[-1]["leave_ns"] - t0) / 1e6, 3),
        "submit_wall_sum_ms": round(total_wall, 3),
        "inter_submit_gap_sum_ms": round(sum(g["gap_ms"] for g in gaps), 3),
        "inter_submit_gaps": gaps,
        "submit_detail": submits,
        "notes": {
            "sampling": "fixed-interval free-running blockwatch: EPERF_RKNPU_BLOCKWATCH=1 SELECT=5 NSLEEP=50000",
            "read_model": "t_ns/t_end_ns bracket one sample; c0_gb_ns/c0_ge_ns bracket the four sequential S_STATUS reads",
            "resolution": "gaps shorter than the sample interval cannot be resolved",
            "submit_axis": "submit start/leave and inter-submit gaps come from the submit-only interposer (no MMIO, no regcmd capture)",
        },
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("submit_detail", "inter_submit_gaps")}, indent=2))
    for s in submits:
        print(f"S{s['submit_seq']:2d} tasks={s['tasks']:6d} wall={s['submit_wall_ms']:7.2f}ms samples={s['samples']:4d} "
              f"dt={s['interval_p50_us']:6.1f}us comp={s['class_share']['compute-only']:.2f} "
              f"dma={s['class_share']['dma-only']:.2f} both={s['class_share']['both']:.2f} idle={s['class_share']['all-idle']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
