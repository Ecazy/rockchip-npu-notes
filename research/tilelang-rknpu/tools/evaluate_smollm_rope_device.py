#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual SmolLM2-shape device RoPE vs the production Host RoPE action.

Compares the production `smollm.rope_half.v1` implementation (coefficient
selection + FP32 expression + FP16 writeback) with the device reorder plan for
q (9 heads) and k (3 heads), in prefill and decode shapes.  Coefficients are
runtime inputs in this probe; coefficient residency is reported separately.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / ".data" / "models" / "SmolLM2-135M-Instruct" / "config.json"


def cos_sin(tokens, heads, head_dim, theta, positions):
    inv = np.power(np.float32(theta),
                   -np.arange(0, head_dim, 2, dtype=np.float32) /
                   np.float32(head_dim))
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return (
        np.ascontiguousarray(
            np.broadcast_to(cos[:, None, :], (tokens, heads, head_dim)
                            ).reshape(tokens, heads * head_dim).astype(
                                np.float16)),
        np.ascontiguousarray(
            np.broadcast_to(sin[:, None, :], (tokens, heads, head_dim)
                            ).reshape(tokens, heads * head_dim).astype(
                                np.float16)),
    )


def perm(width, head_dim):
    p = np.zeros((width, width), dtype=np.float16)
    half = head_dim // 2
    for base in range(0, width, head_dim):
        for d in range(head_dim):
            pair = d + half if d < half else d - half
            p[base + d, base + pair] = -1.0 if d < half else 1.0
    return np.ascontiguousarray(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                               rope_device_module, save_plan)
    from rknpu_backend.hybrid import HOST_ACTIONS
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    cfg = json.loads(CFG.read_text())
    heads, kv_heads, hd = (int(cfg["num_attention_heads"]),
                           int(cfg["num_key_value_heads"]),
                           int(cfg["hidden_size"] // cfg["num_attention_heads"]))
    theta = float(cfg["rope_theta"])
    host = HOST_ACTIONS["smollm.rope_half.v1"].func
    lib = LibRKPlan(args.lib) if args.lib else LibRKPlan()
    fd = 0
    owns_fd = False
    if args.backend == "board":
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
        owns_fd = True
    plans = {}
    configs = [
        ("prefill_q", 256, heads, np.arange(256, dtype=np.int32)),
        ("prefill_k", 256, kv_heads, np.arange(256, dtype=np.int32)),
        ("decode_q", 1, heads, np.asarray([42], dtype=np.int32)),
        ("decode_k", 1, kv_heads, np.asarray([42], dtype=np.int32)),
    ]
    results = {}
    try:
        for label, tokens, nheads, positions in configs:
            width = nheads * hd
            x = np.ascontiguousarray(
                np.random.default_rng(1000 + width).normal(
                    size=(tokens, width)).astype(np.float16))
            cos, sin = cos_sin(tokens, nheads, hd, theta, positions)
            hx = x.reshape(tokens, nheads, hd)
            ref = host({"x": hx, "positions": positions},
                       {"theta": theta, "interleaved": False})["y"]
            plan = compile_rkir(
                rope_device_module(m=tokens, width=width,
                                   name=f"smollm_{label}"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            plans[label] = plan
            consts = {}
            for spec in plan.constants:
                if spec.name == "P":
                    consts["P"] = perm(width, hd)
                elif spec.generator == "identity":
                    n = int(spec.shape[0])
                    consts[spec.name] = np.eye(n, dtype=np.float16)
                else:
                    raise SystemExit(f"unexpected constant {spec.name}")
            dev_times, host_times = [], []
            with PlanCall(lib, plan, consts, fd=fd) as call:
                out = np.empty((tokens, width), dtype=np.float16)
                for _ in range(args.warmup):
                    call.run_into([x, cos, sin], [out])
                for _ in range(args.calls):
                    t0 = time.perf_counter()
                    stats, _ = call.run_into([x, cos, sin], [out])
                    dev_times.append(time.perf_counter() - t0)
                stats, _ = call.run_into([x, cos, sin], [out])
            for _ in range(args.warmup):
                host({"x": hx, "positions": positions},
                     {"theta": theta, "interleaved": False})
            for _ in range(args.calls):
                t0 = time.perf_counter()
                host({"x": hx, "positions": positions},
                     {"theta": theta, "interleaved": False})
                host_times.append(time.perf_counter() - t0)
            got = out.reshape(tokens, nheads, hd)
            diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
            results[label] = {
                "tokens": tokens, "heads": nheads, "head_dim": hd,
                "plan_hash": plan.plan_hash,
                "task_count": stats.task_count,
                "submit_count": stats.submit_count,
                "upload_calls": stats.upload_calls,
                "upload_bytes": stats.upload_bytes,
                "download_calls": stats.download_calls,
                "download_bytes": stats.download_bytes,
                "device_median_us": statistics.median(dev_times) * 1e6,
                "production_host_median_us": statistics.median(host_times) * 1e6,
                "bitwise_equal": bool(np.array_equal(got, ref)),
                "max_abs_diff": float(diff.max()),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "finite": bool(np.isfinite(got).all()),
                "coefficient_residency": "runtime Cos/Sin inputs; not resident",
            }
        problems = []
        for label, r in results.items():
            if not r["finite"]:
                problems.append(f"{label}:nonfinite")
            if r["max_abs_diff"] > 0.03125:
                problems.append(f"{label}:max_abs")
            if r["rmse"] > 0.01:
                problems.append(f"{label}:rmse")
        report = {
            "schema_version": 1,
            "kind": "smollm-rope-device-probe",
            "model_config": {"hidden_size": cfg["hidden_size"],
                             "q_heads": heads, "kv_heads": kv_heads,
                             "head_dim": hd, "rope_theta": theta},
            "results": results,
            "problems": problems,
            "pass": not problems,
            "coefficient_residency": "not achieved; runtime uploads measured",
            "host_comparison": "production smollm.rope_half.v1 Host action",
        }
        out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        for label, plan in plans.items():
            save_plan(plan, str(out.with_suffix(f".{label}.rkplan.json")))
        print(json.dumps(report, indent=1))
        return 0 if not problems else 1
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
