#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal device RoPE reorder probe.

The graph is X @ P (signed block-diagonal permutation), then three FP16 EW
tasks.  Cos/Sin are precomputed inputs, so this is a *reorder* experiment, not
yet a claim of coefficient residency or full Smol integration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def coefficients(m, width, head_dim, rotary_dim, style):
    half = rotary_dim // 2
    cos = np.ones((m, width), dtype=np.float16)
    sin = np.zeros((m, width), dtype=np.float16)
    perm = np.zeros((width, width), dtype=np.float16)
    for h in range(width // head_dim):
        base = h * head_dim
        for d in range(rotary_dim):
            if style == "interleaved":
                pair = d ^ 1
                sign = -1.0 if d % 2 == 0 else 1.0
                freq_index = d // 2
            else:
                pair = d + half if d < half else d - half
                sign = -1.0 if d < half else 1.0
                freq_index = d % half
            perm[base + d, base + pair] = sign
            freq = 10000.0 ** (-2.0 * freq_index / rotary_dim)
            for pos in range(m):
                angle = pos * freq
                cos[pos, base + d] = np.float16(np.cos(angle))
                sin[pos, base + d] = np.float16(np.sin(angle))
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin), perm


def host_rope(x, cos, sin, width, head_dim, rotary_dim, style):
    half = rotary_dim // 2
    rot = np.empty_like(x)
    for h in range(width // head_dim):
        base = h * head_dim
        for d in range(rotary_dim):
            pair = (d ^ 1) if style == "interleaved" else (
                d + half if d < half else d - half)
            rot[:, base + d] = np.float16(
                (-1.0 if (d % 2 == 0 and style == "interleaved") or
                 (d < half and style != "interleaved") else 1.0) *
                x[:, base + pair].astype(np.float32))
    rot[:, :] = np.where(
        (np.arange(width) % head_dim)[None, :] < rotary_dim, rot, 0)
    t1 = np.float16(x.astype(np.float32) * cos.astype(np.float32))
    t2 = np.float16(rot.astype(np.float32) * sin.astype(np.float32))
    return np.float16(t1.astype(np.float32) + t2.astype(np.float32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--style", choices=("interleaved", "split_half"),
                    default="interleaved")
    ap.add_argument("--rotary-dim", type=int, default=64)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir,
                               default_profile, load_plan, save_plan,
                               rope_device_module)
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

    m, width, head_dim = 16, 64, 64
    cos, sin, perm = coefficients(m, width, head_dim, args.rotary_dim,
                                  args.style)
    rng = np.random.default_rng(777)
    x = np.ascontiguousarray(rng.normal(size=(m, width)).astype(np.float16))
    want = host_rope(x, cos, sin, width, head_dim, args.rotary_dim, args.style)
    plan = compile_rkir(
        rope_device_module(m=m, width=width,
                           name=f"rope_{args.style}_{args.rotary_dim}"),
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
    device_times, host_times = [], []
    try:
        with PlanCall(lib, plan, {"P": perm,
                                  "__identity_64": np.eye(64,
                                                          dtype=np.float16)},
                      fd=fd) as call:
            got = np.empty_like(x)
            for _ in range(args.warmup):
                call.run_into([x, cos, sin], [got])
            for _ in range(args.calls):
                t0 = time.perf_counter()
                stats, _ = call.run_into([x, cos, sin], [got])
                device_times.append(time.perf_counter() - t0)
            # Use a fresh output to measure one clean statistics record.
            stats, _ = call.run_into([x, cos, sin], [got])
        for _ in range(args.warmup):
            host_rope(x, cos, sin, width, head_dim, args.rotary_dim, args.style)
        for _ in range(args.calls):
            t0 = time.perf_counter()
            host_rope(x, cos, sin, width, head_dim, args.rotary_dim, args.style)
            host_times.append(time.perf_counter() - t0)
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)
    exact = bool(np.array_equal(got, want))
    diff = np.abs(got.astype(np.float32) - want.astype(np.float32))
    problems = []
    if not np.isfinite(got).all():
        problems.append("nonfinite")
    if not exact or float(diff.max()) > 0.0:
        problems.append("numeric_mismatch")
    if stats.task_count != 4:
        problems.append("task_count")
    if stats.submit_count != 1:
        problems.append("submit_count")
    if stats.upload_calls != 3 or stats.download_calls != 1:
        problems.append("transfer_count")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "rope-device-reorder-probe",
        "plan_hash": plan.plan_hash,
        "profile_hash": plan.profile_hash,
        "style": args.style,
        "rotary_dim": args.rotary_dim,
        "m": m, "width": width, "head_dim": head_dim,
        "task_count": stats.task_count,
        "submit_count": stats.submit_count,
        "upload_calls": stats.upload_calls,
        "upload_bytes": stats.upload_bytes,
        "download_calls": stats.download_calls,
        "download_bytes": stats.download_bytes,
        "bitwise_equal_host_fp16": exact,
        "max_abs_diff": float(diff.max()),
        "device_median_us": statistics.median(device_times) * 1e6,
        "device_mean_us": statistics.mean(device_times) * 1e6,
        "host_median_us": statistics.median(host_times) * 1e6,
        "host_mean_us": statistics.mean(host_times) * 1e6,
        "calls": args.calls,
        "host_activation_reads": 0,
        "device_reorder": "gate MM with signed block-diagonal P",
        "coefficient_residency": "precomputed Cos/Sin uploaded per call; not yet resident",
        "finite": bool(np.isfinite(got).all()),
        "problems": problems,
        "pass": not problems,
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    save_plan(plan, str(out.with_suffix(".rkplan.json")))
    print(json.dumps(report, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
