#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tiny ViT b8 PV→O multi-Submit device dataflow probe.

Eight independent 11-task PV→O chains share one workspace.  The 64-task
single-chain limit is not used here; each batch is a legal 11-task Submit and
the intermediates stay in the same device workspace between Submit groups.
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
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                               pv_o_device_batches_module, save_plan)
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    batches, m, heads, hd, out_dim = 8, 16, 4, 16, 64
    rng = np.random.default_rng(910)
    probs = f16(rng.normal(size=(batches, heads, m, hd)))
    v = f16(rng.normal(size=(batches, heads, m, hd)))
    w = f16(rng.normal(size=(out_dim, heads * hd)))
    inputs = [np.ascontiguousarray(probs[b, h])
              for b in range(batches) for h in range(heads)]
    inputs += [np.ascontiguousarray(v[b, h].T)
               for b in range(batches) for h in range(heads)]
    constants = {f"W{h}": np.ascontiguousarray(w[:, h * hd:(h + 1) * hd])
                 for h in range(heads)}
    constants["__identity_64"] = np.eye(64, dtype=np.float16)

    def host_four(b):
        pv = [f16(probs[b, h].astype(np.float32) @ v[b, h].astype(np.float32))
              for h in range(heads)]
        parts = [f16(pv[h].astype(np.float32) @
                     w[:, h * hd:(h + 1) * hd].astype(np.float32).T)
                 for h in range(heads)]
        y = parts[0]
        for part in parts[1:]:
            y = f16(y.astype(np.float32) + part.astype(np.float32))
        return y

    def host_merged(b):
        merged = f16(np.concatenate(
            [probs[b, h].astype(np.float32) @ v[b, h].astype(np.float32)
             for h in range(heads)], axis=1))
        return f16(merged.astype(np.float32) @ w.astype(np.float32).T)

    plan = compile_rkir(
        pv_o_device_batches_module(batches=batches, m=m, heads=heads,
                                   head_dim=hd, out_dim=out_dim),
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
            outs = [np.empty((m, out_dim), dtype=np.float16)
                    for _ in range(batches)]
            for _ in range(args.warmup):
                call.run_into(inputs, outs)
            for _ in range(args.calls):
                t0 = time.perf_counter()
                stats, _ = call.run_into(inputs, outs)
                dev_times.append(time.perf_counter() - t0)
            stats, _ = call.run_into(inputs, outs)
        for _ in range(args.warmup):
            for b in range(batches):
                host_four(b)
                host_merged(b)
        for _ in range(args.calls):
            t0 = time.perf_counter()
            for b in range(batches):
                host_four(b)
            host_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for b in range(batches):
                host_merged(b)
            merged_times.append(time.perf_counter() - t0)
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)
    problems = []
    max_four = 0.0
    rmse_four = 0.0
    max_merged = 0.0
    rmse_merged = 0.0
    exact = True
    for b in range(batches):
        hf = host_four(b)
        hm = host_merged(b)
        if not np.array_equal(outs[b], hf):
            exact = False
        df = np.abs(outs[b].astype(np.float32) - hf.astype(np.float32))
        dm = np.abs(outs[b].astype(np.float32) - hm.astype(np.float32))
        max_four = max(max_four, float(df.max()))
        rmse_four = max(rmse_four, float(np.sqrt(np.mean(df * df))))
        max_merged = max(max_merged, float(dm.max()))
        rmse_merged = max(rmse_merged, float(np.sqrt(np.mean(dm * dm))))
        if not np.isfinite(outs[b]).all():
            problems.append(f"nonfinite_b{b}")
    # The multi-Submit path is an independent FP16 policy: require a bounded
    # difference from the per-batch four-head reference, not bitwise identity
    # to the single merged O GEMM.
    if max_four > 0.03125:
        problems.append("four_head_max_abs")
    if rmse_four > 0.01:
        problems.append("four_head_rmse")
    if max_merged > 0.125:
        problems.append("merged_max_abs")
    if rmse_merged > 0.02:
        problems.append("merged_rmse")
    if stats.task_count != 88:
        problems.append("task_count")
    if stats.submit_count != 8:
        problems.append("submit_count")
    if stats.upload_calls != 64 or stats.download_calls != 8:
        problems.append("transfer_count")
    report = {
        "schema_version": 1,
        "kind": "pv-o-b8-multi-submit-device-probe",
        "plan_hash": plan.plan_hash,
        "batches": batches, "m": m, "heads": heads,
        "head_dim": hd, "out_dim": out_dim,
        "task_count": stats.task_count,
        "submit_count": stats.submit_count,
        "upload_calls": stats.upload_calls,
        "upload_bytes": stats.upload_bytes,
        "download_calls": stats.download_calls,
        "download_bytes": stats.download_bytes,
        "bitwise_equal_host_four_head_all_batches": exact,
        "max_abs_diff_vs_host_four_head": max_four,
        "rmse_vs_host_four_head": rmse_four,
        "max_abs_diff_vs_host_merged": max_merged,
        "rmse_vs_host_merged": rmse_merged,
        "four_head_atol": 0.03125,
        "merged_atol": 0.125,
        "merged_rmse_limit": 0.02,
        "device_median_us": statistics.median(dev_times) * 1e6,
        "host_four_head_median_us": statistics.median(host_times) * 1e6,
        "host_merged_median_us": statistics.median(merged_times) * 1e6,
        "problems": problems,
        "pass": not problems,
        "residency": "eight independent 11-task Submit groups in one workspace; no Host head merge",
        "numeric_policy": "four independent PV/O stages plus FP16 EW add per batch; not bitwise to merged O",
    }
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    save_plan(plan, str(out.with_suffix(".rkplan.json")))
    print(json.dumps(report, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
