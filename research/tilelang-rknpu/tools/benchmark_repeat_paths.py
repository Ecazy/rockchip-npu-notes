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


def run_variant(*, name: str, lib, plan, group_plan, nrepeat: int,
                runs: int, warmup: int, fd: int) -> dict:
    from rknpu_backend.models import PlanCall

    inputs = _inputs(plan, nrepeat, 20261026 + nrepeat)
    created = {}
    t0 = time.perf_counter()
    if name == "chain":
        call = PlanCall(lib, group_plan, {}, fd=fd)
    else:
        call = PlanCall(lib, plan, {}, fd=fd)
    create_us = (time.perf_counter() - t0) * 1e6
    created["workspace_create_us"] = create_us
    samples = []
    report = None
    try:
        def once():
            """One full group; returns the last native report it saw."""
            outs = _outputs(plan, nrepeat)
            last = None
            if name == "separate":
                for i in range(nrepeat):
                    last, _ = call.run_into(inputs[i], outs[i])
            elif name == "group":
                last, _ = call.run_group_into(inputs, outs)
            else:
                last, _ = call.run_into([a for row in inputs for a in row],
                                        [o for row in outs for o in row])
            return last

        for _ in range(warmup):
            once()
        for _ in range(runs):
            t = time.perf_counter()
            report = once()
            samples.append((time.perf_counter() - t) * 1e6)
    finally:
        call.close()
    med = statistics.median(samples)
    return {
        "variant": name,
        "runs": runs,
        "wall_us": {"median": med, "min": min(samples), "max": max(samples),
                    "samples": samples},
        "per_task_us": med / nrepeat,
        # ``counts``/``attribution`` are the last native report: one call for
        # "separate" (per-call cost) and the whole group for "group"/"chain".
        "counts": _counts(report) if report is not None else {},
        "attribution": _attribution(report) if report is not None else {},
        "attribution_scope": "one_call" if name == "separate" else "whole_group",
        **created,
    }


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
        group_plan = expand_static_repeat(plan, args.repeat)
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "repeat-path-benchmark",
            "plan": args.plan,
            "repeat": args.repeat,
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
                "group": {"hash": group_plan.plan_hash,
                          "accounting": group_plan.accounting.to_dict()},
            },
            "variants": [],
        }
        for name in ("separate", "group", "chain"):
            result["variants"].append(run_variant(
                name=name, lib=lib, plan=plan, group_plan=group_plan,
                nrepeat=args.repeat, runs=args.runs, warmup=args.warmup, fd=fd))
        # Correctness: the three paths must agree bit for bit.
        from rknpu_backend.models import PlanCall
        inputs = _inputs(plan, args.repeat, 20261026 + args.repeat)
        outs = {}
        for name in ("separate", "group", "chain"):
            target = group_plan if name == "chain" else plan
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
        for name in ("separate", "group", "chain"):
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
        print(f"  {v['variant']:<9} median {v['wall_us']['median'] / 1e3:8.3f} ms  "
              f"per task {v['per_task_us']:7.3f} us  "
              f"tasks {c.get('task_count', '?')} submits {c.get('submit_count', '?')}  "
              f"up {c.get('upload_calls', '?')} down {c.get('download_calls', '?')}  "
              f"create {v['workspace_create_us'] / 1e3:.3f} ms")
        a = v["attribution"]
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
