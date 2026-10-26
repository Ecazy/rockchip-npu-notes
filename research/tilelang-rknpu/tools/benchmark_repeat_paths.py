#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare the three ways to execute one static-repeat group of a plan.

For a QK/PV-style node the manifest declares N independent instances of one
immutable plan.  They can be executed as

  separate : N calls, N submits, N guard gates, N uploads/downloads rounds
  group    : one native call with N repeats (B1): N submits, one guard gate
  chain    : one expanded plan with N tasks and one chained submit (B2)

This tool measures all three on the real device with the real runtime, including
uploads, downloads and the final guard, and also reports what each variant costs
to *create* (workspace + BOs) and how much device/Host memory it declares.  It is
deliberately independent of the hybrid manifests: no whole-model wiring is
needed to decide whether the chain is worth it.

Usage (board):
  RKNPU_CORE_MASK=1 taskset -c 6 python3 tools/benchmark_repeat_paths.py \
      --plans-dir models/compiled --lib <build>/librknpu_plan.so \
      --plan tiny_vit_b8_qk --repeat 32 --runs 30 --out results/<stamp>-b2bench/bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def _device_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_bytes().rstrip(b"\x00").decode(
            "utf-8", "replace")
    except OSError:
        return ""


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _inputs(plan, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        row = []
        for binding in plan.bindings:
            if binding.role == "output":
                continue
            row.append(np.ascontiguousarray(
                rng.standard_normal(binding.shape).astype(np.float16)))
        out.append(row)
    return out


def _outputs(plan, n):
    (out_binding,) = [b for b in plan.bindings if b.role == "output"]
    return [[np.zeros(out_binding.shape, dtype=np.float16)] for _ in range(n)]


def _counts(report) -> dict:
    return {
        "task_count": int(report.task_count),
        "submit_count": int(report.submit_count),
        "upload_calls": int(report.upload_calls),
        "upload_bytes": int(report.upload_bytes),
        "download_calls": int(report.download_calls),
        "download_bytes": int(report.download_bytes),
    }


def _attribution(report) -> dict:
    fields = ("validate_us", "pack_us", "emit_us", "command_us",
              "submit_alloc_us", "submit_wait_us", "output_sync_us",
              "download_us", "guard_us", "publish_us", "alloc_us",
              "total_us", "clock_us")
    return {f: float(getattr(report, f, 0.0)) for f in fields}


ATTRIBUTION_FIELDS = ("validate_us", "pack_us", "emit_us", "command_us",
                       "submit_alloc_us", "submit_wait_us", "output_sync_us",
                       "download_us", "guard_us", "publish_us", "alloc_us",
                       "total_us", "clock_us")


def make_caller(*, name: str, call_for, group_call, plan, group_plans: dict,
                nrepeat: int, inputs: list, outputs: list):
    """Return a callable executing one whole group and returning its aggregate.

    ``separate`` sums the N per-call reports, so every variant's attribution
    describes the *whole group* and the numbers are comparable.  The aggregate is
    recorded per round (the tool alternates variants round by round), and the
    reported figures are medians over those per-round aggregates -- not one
    sampled run.
    """
    def call() -> dict:
        agg = {f: 0.0 for f in ATTRIBUTION_FIELDS}
        counts = {"task_count": 0, "submit_count": 0, "upload_calls": 0,
                  "upload_bytes": 0, "download_calls": 0, "download_bytes": 0}
        if name == "separate":
            for i in range(nrepeat):
                report, _ = call_for(inputs[i], outputs[i])
                for f in ATTRIBUTION_FIELDS:
                    agg[f] += float(getattr(report, f, 0.0))
                for k in counts:
                    counts[k] += int(getattr(report, k, 0))
        elif name == "group":
            report, _ = group_call(inputs, outputs)
            for f in ATTRIBUTION_FIELDS:
                agg[f] += float(getattr(report, f, 0.0))
            for k in counts:
                counts[k] += int(getattr(report, k, 0))
        else:
            report, _ = call_for([a for row in inputs for a in row],
                                 [o for row in outputs for o in row])
            for f in ATTRIBUTION_FIELDS:
                agg[f] += float(getattr(report, f, 0.0))
            for k in counts:
                counts[k] += int(getattr(report, k, 0))
        return {"attribution": agg, "counts": counts}

    return call


def median_dict(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        out[key] = statistics.median(values) if values else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="tiny_vit_b8_qk")
    ap.add_argument("--plans-dir", type=Path, default=ROOT / "models" / "compiled")
    ap.add_argument("--lib", default="")
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fd", type=int, default=0,
                    help="already-open device fd; 0 opens its own")
    ap.add_argument("--mock", action="store_true",
                    help="host mock library: use fd 0 and do not open a device")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from rknpu_backend import load_plan
    from rknpu_backend.passes.repeat_expand import expand_static_repeat
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

    lib_path = args.lib or str(ROOT.parent / "custom-ops" / "build-host-mvp" /
                               "librknpu_plan.so")
    lib = LibRKPlan(lib_path)
    configure_rocket_abi(lib.cdll)
    fd = args.fd
    opened = False
    if args.mock:
        fd = 0
    elif not fd:
        fd = lib.cdll.rocket_open()
        if fd < 0:
            raise SystemExit("rocket_open failed")
        opened = True
    try:
        plan = load_plan(str(args.plans_dir / f"{args.plan}.rkplan.json"))
        group_plans = {
            "chain": expand_static_repeat(plan, args.repeat),
            "chain_shared": expand_static_repeat(plan, args.repeat, shared_bo=True),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "repeat-path-benchmark",
            "plan": args.plan,
            "repeat": args.repeat,
            "rounds": args.runs,
            "interleaved": True,
            "attribution_scope": "per-round aggregate over the whole group "
                                 "(separate sums its N calls)",
            "library": lib_path,
            "host": {"machine": platform.machine(),
                     "device_model": _device_model(),
                     "kernel": platform.release(),
                     "cpu_affinity": sorted(__import__("os").sched_getaffinity(0)),
                     "npu_freq": _read("/sys/kernel/debug/rknpu/freq"),
                     "npu_load": _read("/sys/kernel/debug/rknpu/load")},
            "plans": {
                "single": {"hash": plan.plan_hash,
                           "accounting": plan.accounting.to_dict()},
                "chain": {"hash": group_plans["chain"].plan_hash,
                          "accounting": group_plans["chain"].accounting.to_dict()},
                "chain_shared": {
                    "hash": group_plans["chain_shared"].plan_hash,
                    "accounting": group_plans["chain_shared"].accounting.to_dict()},
            },
            "variants": [],
        }

        from rknpu_backend.models import PlanCall
        variants = ("separate", "group", "chain", "chain_shared")
        inputs = _inputs(plan, args.repeat, 20261026 + args.repeat)
        callers = {}
        creation = {}
        calls = {}
        try:
            for name in variants:
                t0 = time.perf_counter()
                if name in ("chain", "chain_shared"):
                    target = group_plans[name]
                else:
                    target = plan
                calls[name] = PlanCall(lib, target, {}, fd=fd)
                creation[name] = (time.perf_counter() - t0) * 1e6
                if name in ("separate", "chain", "chain_shared"):
                    callers[name] = make_caller(
                        name=name, call_for=calls[name].run_into,
                        group_call=None, plan=plan, group_plans=group_plans,
                        nrepeat=args.repeat, inputs=inputs,
                        outputs=_outputs(plan, args.repeat))
                else:
                    callers[name] = make_caller(
                        name=name, call_for=None,
                        group_call=calls[name].run_group_into, plan=plan,
                        group_plans=group_plans, nrepeat=args.repeat,
                        inputs=inputs, outputs=_outputs(plan, args.repeat))
            for _ in range(args.warmup):
                for name in variants:
                    callers[name]()
            samples = {name: [] for name in variants}
            aggregates = {name: [] for name in variants}
            # Interleaved rounds: every variant sees the same machine state.
            for _ in range(args.runs):
                for name in variants:
                    t0 = time.perf_counter()
                    aggregate = callers[name]()
                    samples[name].append((time.perf_counter() - t0) * 1e6)
                    aggregates[name].append(aggregate)
        finally:
            for call in calls.values():
                call.close()
        for name in variants:
            med = statistics.median(samples[name])
            agg_med = median_dict([a["attribution"] for a in aggregates[name]])
            counts = median_dict([{k: float(v) for k, v in a["counts"].items()}
                                  for a in aggregates[name]])
            result["variants"].append({
                "variant": name,
                "rounds_measured": len(samples[name]),
                "wall_us": {"median": med, "min": min(samples[name]),
                            "max": max(samples[name]),
                            "samples": samples[name]},
                "per_task_us": med / args.repeat,
                "attribution_median": agg_med,
                "counts": {k: int(v) for k, v in counts.items()},
                "workspace_create_us": creation[name],
            })

        # Correctness: the three paths must agree bit for bit.
        from rknpu_backend.models import PlanCall
        inputs = _inputs(plan, args.repeat, 20261026 + args.repeat)
        outs = {}
        for name in ("separate", "group", "chain", "chain_shared"):
            target = group_plans[name] if name in ("chain", "chain_shared") else plan
            call = PlanCall(lib, target, {}, fd=fd)
            try:
                produced = _outputs(plan, args.repeat)
                if name == "separate":
                    for i in range(args.repeat):
                        call.run_into(inputs[i], produced[i])
                elif name == "group":
                    call.run_group_into(inputs, produced)
                else:
                    call.run_into([a for row in inputs for a in row],
                                  [o for row in produced for o in row])
                outs[name] = produced
            finally:
                call.close()
        same = {}
        for name in ("separate", "group", "chain", "chain_shared"):
            same[name] = all(
                np.array_equal(outs["separate"][i][0], outs[name][i][0])
                for i in range(args.repeat))
        result["bitwise_equal_to_separate"] = same
        result["all_paths_equal"] = all(same.values())
    finally:
        if opened:
            lib.cdll.rocket_close(fd)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(f"plan={args.plan} repeat={args.repeat} runs={args.runs}")
    for v in result["variants"]:
        c = v["counts"]
        print(f"  {v['variant']:<13} median {v['wall_us']['median'] / 1e3:8.3f} ms  "
              f"per task {v['per_task_us']:7.3f} us  "
              f"tasks {c.get('task_count', '?')} submits {c.get('submit_count', '?')}  "
              f"up {c.get('upload_calls', '?')} down {c.get('download_calls', '?')}  "
              f"create {v['workspace_create_us'] / 1e3:.3f} ms")
        a = v["attribution_median"]
        print(f"            submit_wait {a.get('submit_wait_us', 0) / 1e3:7.3f} ms  "
              f"pack {a.get('pack_us', 0) / 1e3:6.3f}  "
              f"download {a.get('download_us', 0) / 1e3:6.3f}  "
              f"guard {a.get('guard_us', 0) / 1e3:6.3f}  "
              f"total {a.get('total_us', 0) / 1e3:7.3f}")
    for label, acct in result["plans"].items():
        print(f"  {label:<7} device {acct['accounting']['device_request_bytes']} B  "
              f"host {acct['accounting']['host_bytes']} B  "
              f"BOs {acct['accounting']['data_bo_count']}  "
              f"tasks {acct['accounting']['task_count']}  "
              f"submits {acct['accounting']['submit_count']}")
    print(f"  bitwise equal to separate: {result['bitwise_equal_to_separate']}")
    if args.out:
        print(f"wrote {args.out}")
    return 0 if result["all_paths_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
