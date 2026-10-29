#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""SmolLM2 RoPE residency experiment.

Prefill: Cos/Sin are plan constants for the fixed 256-token bucket, uploaded once
and reused across 30 simulated layers.  Decode: coefficients are runtime inputs
because the current runtime has no dynamic position-indexed device view; the
per-token transfer is measured explicitly.
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


def cos_sin(tokens, heads, hd, theta, positions):
    inv = np.power(np.float32(theta),
                   -np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd))
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return (
        np.ascontiguousarray(np.broadcast_to(
            cos[:, None, :], (tokens, heads, hd)).reshape(
                tokens, heads * hd).astype(np.float16)),
        np.ascontiguousarray(np.broadcast_to(
            sin[:, None, :], (tokens, heads, hd)).reshape(
                tokens, heads * hd).astype(np.float16)),
    )


def perm(width, hd):
    p = np.zeros((width, width), dtype=np.float16)
    half = hd // 2
    for base in range(0, width, hd):
        for d in range(hd):
            pair = d + half if d < half else d - half
            p[base + d, base + pair] = -1.0 if d < half else 1.0
    return np.ascontiguousarray(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                               rope_device_constant_module,
                               rope_device_module, save_plan)
    from rknpu_backend.hybrid import HOST_ACTIONS
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    cfg = json.loads(CFG.read_text())
    q_heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    hd = int(cfg["hidden_size"] // cfg["num_attention_heads"])
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
    report = {"schema_version": 1, "kind": "smollm-rope-residency-probe",
              "model_config": {"q_heads": q_heads, "kv_heads": kv_heads,
                               "head_dim": hd, "rope_theta": theta},
              "layers": args.layers, "prefill": {}, "decode": {},
              "problems": []}
    plans = {}
    try:
        # Prefill: fixed bucket, coefficients are constants and reused.
        positions = np.arange(256, dtype=np.int32)
        for label, nheads in (("q", q_heads), ("k", kv_heads)):
            width = nheads * hd
            cos, sin = cos_sin(256, nheads, hd, theta, positions)
            plan = compile_rkir(
                rope_device_constant_module(m=256, width=width,
                                            name=f"smollm_prefill_{label}_const"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            plans[f"prefill_{label}"] = plan
            consts = {"Cos": cos, "Sin": sin, "P": perm(width, hd)}
            for spec in plan.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    consts[spec.name] = np.eye(n, dtype=np.float16)
            t0 = time.perf_counter()
            call = PlanCall(lib, plan, consts, fd=fd, share_constants=True)
            setup_us = (time.perf_counter() - t0) * 1e6
            xs, refs, times = [], [], []
            for layer in range(args.layers):
                x = np.ascontiguousarray(np.random.default_rng(
                    5000 + layer).normal(size=(256, width)).astype(np.float16))
                ref = host({"x": x.reshape(256, nheads, hd),
                            "positions": positions},
                           {"theta": theta, "interleaved": False})["y"]
                xs.append(x); refs.append(ref)
            out = np.empty((256, width), dtype=np.float16)
            for x in xs[:2]:
                call.run_into([x], [out])
            for x in xs:
                t0 = time.perf_counter()
                stats, _ = call.run_into([x], [out])
                times.append(time.perf_counter() - t0)
            call.close()
            got = out.reshape(256, nheads, hd)
            diff = np.abs(got.astype(np.float32) - refs[-1].astype(np.float32))
            max_diff = float(diff.max())
            if not np.isfinite(got).all() or max_diff > 0.03125:
                report["problems"].append(f"prefill_{label}_numeric")
            report["prefill"][label] = {
                "plan_hash": plan.plan_hash,
                "task_count": stats.task_count,
                "submit_count": stats.submit_count,
                "per_layer_upload_calls": stats.upload_calls,
                "per_layer_upload_bytes": stats.upload_bytes,
                "per_layer_download_bytes": stats.download_bytes,
                "setup_constant_upload_us": setup_us,
                "constant_bytes": int(sum(c.nbytes for c in plan.constants)),
                "device_layers_median_us": statistics.median(times) * 1e6,
                "device_layers_total_us": sum(times) * 1e6,
                "max_abs_diff_last_layer": max_diff,
                "finite": bool(np.isfinite(got).all()),
                "coefficient_residency": "Cos/Sin/P uploaded once at PlanCall creation; reused for all layers",
            }
            # Production Host action on the same 30 inputs.
            ht = []
            for x in xs:
                t0 = time.perf_counter()
                host({"x": x.reshape(256, nheads, hd), "positions": positions},
                     {"theta": theta, "interleaved": False})
                ht.append(time.perf_counter() - t0)
            report["prefill"][label]["production_host_layers_total_us"] = sum(ht) * 1e6
        # Decode: runtime inputs; current runtime lacks dynamic device row views.
        for label, nheads in (("q", q_heads), ("k", kv_heads)):
            width = nheads * hd
            pos = np.asarray([42], dtype=np.int32)
            cos, sin = cos_sin(1, nheads, hd, theta, pos)
            plan = compile_rkir(
                rope_device_module(m=1, width=width,
                                   name=f"smollm_decode_{label}_runtime"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            plans[f"decode_{label}"] = plan
            consts = {"P": perm(width, hd)}
            for spec in plan.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    consts[spec.name] = np.eye(n, dtype=np.float16)
            x = np.ascontiguousarray(np.random.default_rng(
                6000 + width).normal(size=(1, width)).astype(np.float16))
            ref = host({"x": x.reshape(1, nheads, hd), "positions": pos},
                       {"theta": theta, "interleaved": False})["y"]
            with PlanCall(lib, plan, consts, fd=fd, share_constants=True) as call:
                out = np.empty((1, width), dtype=np.float16)
                for _ in range(2):
                    call.run_into([x, cos, sin], [out])
                times = []
                for _ in range(args.calls):
                    t0 = time.perf_counter()
                    stats, _ = call.run_into([x, cos, sin], [out])
                    times.append(time.perf_counter() - t0)
            diff = np.abs(out.reshape(1, nheads, hd).astype(np.float32) -
                          ref.astype(np.float32))
            max_diff = float(diff.max())
            if not np.isfinite(out).all() or max_diff > 0.03125:
                report["problems"].append(f"decode_{label}_numeric")
            ht = []
            for _ in range(args.calls):
                t0 = time.perf_counter()
                host({"x": x.reshape(1, nheads, hd), "positions": pos},
                     {"theta": theta, "interleaved": False})
                ht.append(time.perf_counter() - t0)
            report["decode"][label] = {
                "plan_hash": plan.plan_hash,
                "task_count": stats.task_count,
                "submit_count": stats.submit_count,
                "per_token_upload_bytes": stats.upload_bytes,
                "per_token_download_bytes": stats.download_bytes,
                "device_median_us": statistics.median(times) * 1e6,
                "production_host_median_us": statistics.median(ht) * 1e6,
                "max_abs_diff": max_diff,
                "finite": bool(np.isfinite(out).all()),
                "coefficient_residency": "runtime inputs; no dynamic position-indexed device view",
            }
        report["pass"] = not report["problems"]
        out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        for label, plan in plans.items():
            save_plan(plan, str(out.with_suffix(f".{label}.rkplan.json")))
        print(json.dumps(report, indent=1))
        return 0 if not report["problems"] else 1
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
