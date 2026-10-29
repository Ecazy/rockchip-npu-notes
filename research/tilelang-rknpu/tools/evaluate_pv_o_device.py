#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal real device PV->O dataflow probe for Tiny ViT b1.

Four PV head matmuls feed four O-weight slice matmuls and three FP16 EW adds in
one plan.  The Host does not merge head outputs.  Inputs are Host-packed for
this experiment; QKV->PV residency is a separate unsolved boundary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def f16(a):
    return np.ascontiguousarray(a.astype(np.float16))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                               pv_o_device_module, save_plan)
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

    m, heads, hd, out_dim = 16, 4, 16, 64
    rng = np.random.default_rng(909)
    probs = f16(rng.normal(size=(heads, m, hd)))
    v = f16(rng.normal(size=(heads, m, hd)))
    w = f16(rng.normal(size=(out_dim, heads * hd)))
    # Module input order is P0..P3, Vt0..Vt3.
    inputs = [np.ascontiguousarray(probs[h]) for h in range(heads)]
    inputs += [np.ascontiguousarray(v[h].T) for h in range(heads)]
    constants = {f"W{h}": np.ascontiguousarray(w[:, h * hd:(h + 1) * hd])
                 for h in range(heads)}
    constants["__identity_64"] = np.eye(64, dtype=np.float16)

    def host_four():
        pv = [f16(probs[h].astype(np.float32) @ v[h].astype(np.float32))
              for h in range(heads)]
        parts = [f16(pv[h].astype(np.float32) @
                     w[:, h * hd:(h + 1) * hd].astype(np.float32).T)
                 for h in range(heads)]
        y = parts[0]
        for p in parts[1:]:
            y = f16(y.astype(np.float32) + p.astype(np.float32))
        return y

    def host_merged():
        merged = f16(np.concatenate(
            [probs[h].astype(np.float32) @ v[h].astype(np.float32)
             for h in range(heads)], axis=1))
        return f16(merged.astype(np.float32) @ w.astype(np.float32).T)

    plan = compile_rkir(
        pv_o_device_module(m=m, heads=heads, head_dim=hd, out_dim=out_dim),
        profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               fuse_task_chain=True)).plan
    lib = LibRKPlan(args.lib) if args.lib else LibRKPlan()
    fd = 0
    owns_fd = False
    if args.backend == "board":
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
        owns_fd = True
    dev_times, host_times, merged_times = [], [], []
    try:
        with PlanCall(lib, plan, constants, fd=fd) as call:
            out = np.empty((m, out_dim), dtype=np.float16)
            for _ in range(args.warmup):
                call.run_into(inputs, [out])
            for _ in range(args.calls):
                t0 = time.perf_counter()
                stats, _ = call.run_into(inputs, [out])
                dev_times.append(time.perf_counter() - t0)
            stats, _ = call.run_into(inputs, [out])
        for _ in range(args.warmup):
            host_four()
            host_merged()
        for _ in range(args.calls):
            t0 = time.perf_counter(); host_four()
            host_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); host_merged()
            merged_times.append(time.perf_counter() - t0)
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)
    hf = host_four()
    hm = host_merged()
    df = np.abs(out.astype(np.float32) - hf.astype(np.float32))
    dm = np.abs(out.astype(np.float32) - hm.astype(np.float32))
    merged_atol = 0.0625
    merged_rmse_limit = 0.02
    problems = []
    if not np.isfinite(out).all():
        problems.append("nonfinite")
    if not np.array_equal(out, hf):
        problems.append("four_head_bitwise")
    if float(dm.max()) > merged_atol:
        problems.append("merged_max_abs")
    if float(np.sqrt(np.mean(dm * dm))) > merged_rmse_limit:
        problems.append("merged_rmse")
    if stats.task_count != 11:
        problems.append("task_count")
    if stats.submit_count != 1:
        problems.append("submit_count")
    if stats.upload_calls != 8 or stats.download_calls != 1:
        problems.append("transfer_count")
    report = {
        "schema_version": 1,
        "kind": "pv-o-device-dataflow-probe",
        "plan_hash": plan.plan_hash,
        "profile_hash": plan.profile_hash,
        "m": m, "heads": heads, "head_dim": hd, "out_dim": out_dim,
        "task_count": stats.task_count,
        "submit_count": stats.submit_count,
        "upload_calls": stats.upload_calls,
        "upload_bytes": stats.upload_bytes,
        "download_calls": stats.download_calls,
        "download_bytes": stats.download_bytes,
        "bitwise_equal_host_four_head_fp16": bool(np.array_equal(out, hf)),
        "max_abs_diff_vs_host_four_head": float(df.max()),
        "rmse_vs_host_four_head": float(np.sqrt(np.mean(df * df))),
        "max_abs_diff_vs_host_merged": float(dm.max()),
        "rmse_vs_host_merged": float(np.sqrt(np.mean(dm * dm))),
        "device_median_us": statistics.median(dev_times) * 1e6,
        "host_four_head_median_us": statistics.median(host_times) * 1e6,
        "host_merged_median_us": statistics.median(merged_times) * 1e6,
        "calls": args.calls,
        "host_head_merge_eliminated": True,
        "input_residency": "probs/V are Host inputs in this probe; QKV->PV residency not claimed",
        "numeric_policy": "four independent PV/O MM stages plus FP16 EW accumulate; not bitwise to one K=64 O MM",
        "merged_atol": merged_atol,
        "merged_rmse_limit": merged_rmse_limit,
        "finite": bool(np.isfinite(out).all()),
        "problems": problems,
        "pass": not problems,
    }
    out_path = Path(args.out).resolve(); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    save_plan(plan, str(out_path.with_suffix(".rkplan.json")))
    print(json.dumps(report, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
