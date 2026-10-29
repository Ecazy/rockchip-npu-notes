#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode RoPE position-indexed gather probe on actual SmolLM2 shapes.

The Cos/Sin tables and P are resident plan constants.  Each token uploads only
X and a one-hot position vector; the device MM gathers the coefficient row.
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


def coeffs(tokens, heads, hd, theta, positions):
    inv = np.power(np.float32(theta),
                   -np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd))
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return (np.ascontiguousarray(np.broadcast_to(
        cos[:, None, :], (tokens, heads, hd)).reshape(
            tokens, heads * hd).astype(np.float16)),
        np.ascontiguousarray(np.broadcast_to(
            sin[:, None, :], (tokens, heads, hd)).reshape(
                tokens, heads * hd).astype(np.float16)))


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
    ap.add_argument("--calls", type=int, default=100)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                               rope_decode_indexed_module, save_plan)
    from rknpu_backend.hybrid import HOST_ACTIONS
    from rknpu_backend.models import PlanCall
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    cfg = json.loads(CFG.read_text())
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    hd = int(cfg["hidden_size"] // heads)
    theta = float(cfg["rope_theta"])
    cap = 256
    host = HOST_ACTIONS["smollm.rope_half.v1"].func
    lib = LibRKPlan(args.lib) if args.lib else LibRKPlan()
    fd = 0; owns_fd = False
    if args.backend == "board":
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0: raise SystemExit(f"rocket_open failed: {fd}")
        owns_fd = True
    results = {}
    try:
        for label, nh in (("q", heads), ("k", kv_heads)):
            width = nh * hd
            positions = np.arange(cap, dtype=np.int32)
            cos, sin = coeffs(cap, nh, hd, theta, positions)
            # Resident tables are laid out in the MM B convention [N,K].
            cosw = np.ascontiguousarray(cos.T)
            sinw = np.ascontiguousarray(sin.T)
            plan = compile_rkir(
                rope_decode_indexed_module(width=width, capacity=cap,
                                           name=f"smollm_decode_indexed_{label}"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            consts = {"CosW": cosw, "SinW": sinw, "P": perm(width, hd)}
            for spec in plan.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    consts[spec.name] = np.eye(n, dtype=np.float16)
            def onehot_for(pos):
                if not isinstance(pos, int) or pos < 0 or pos >= cap:
                    raise ValueError(f"position {pos} outside [0,{cap - 1}]")
                v = np.zeros((1, cap), dtype=np.float16)
                v[0, pos] = 1.0
                return v

            x = np.ascontiguousarray(np.random.default_rng(
                7000 + width).normal(size=(1, width)).astype(np.float16))
            position_cases = [0, 1, 15, 16, 127, 254, 255]
            case_max = 0.0
            aba_bitwise = True
            with PlanCall(lib, plan, consts, fd=fd, share_constants=True) as call:
                out = np.empty((1, width), dtype=np.float16)
                first_a = None
                for idx, pos in enumerate(position_cases + [0]):
                    oh = onehot_for(pos)
                    call.run_into([x, oh], [out])
                    ref = host({"x": x.reshape(1, nh, hd),
                                "positions": np.asarray([pos], dtype=np.int32)},
                               {"theta": theta, "interleaved": False})["y"]
                    d = np.abs(out.reshape(1, nh, hd).astype(np.float32) -
                               ref.astype(np.float32))
                    case_max = max(case_max, float(d.max()))
                    if pos == 0:
                        snap = out.copy()
                        if first_a is None:
                            first_a = snap
                        else:
                            aba_bitwise = aba_bitwise and bool(
                                np.array_equal(snap, first_a))
                # Timing at the last valid position.
                oh = onehot_for(255)
                for _ in range(10):
                    call.run_into([x, oh], [out])
                times = []
                for _ in range(args.calls):
                    t0 = time.perf_counter()
                    stats, _ = call.run_into([x, oh], [out])
                    times.append(time.perf_counter() - t0)
            out_of_range_rejected = False
            try:
                onehot_for(256)
            except ValueError:
                out_of_range_rejected = True
            results[label] = {
                "tokens": 1, "heads": nh, "head_dim": hd,
                "plan_hash": plan.plan_hash,
                "task_count": stats.task_count,
                "submit_count": stats.submit_count,
                "per_token_upload_bytes": stats.upload_bytes,
                "per_token_download_bytes": stats.download_bytes,
                "device_median_us": statistics.median(times) * 1e6,
                "max_abs_diff_vs_production_host": case_max,
                "position_cases": position_cases,
                "aba_repeat_bitwise": aba_bitwise,
                "out_of_range_rejected": out_of_range_rejected,
                "finite": bool(np.isfinite(out).all()),
                "coefficient_residency": "CosW/SinW/P are plan constants; one-hot position vector is the only per-token coefficient metadata",
            }
        problems = []
        for label, r in results.items():
            if not r["finite"]:
                problems.append(f"{label}:nonfinite")
            if r["max_abs_diff_vs_production_host"] > 0.03125:
                problems.append(f"{label}:max_abs")
            if not r["aba_repeat_bitwise"]:
                problems.append(f"{label}:aba")
            if not r["out_of_range_rejected"]:
                problems.append(f"{label}:range")
        report = {"schema_version": 1, "kind": "smollm-rope-indexed-decode-probe",
                  "model_config": {"q_heads": heads, "kv_heads": kv_heads,
                                   "head_dim": hd, "rope_theta": theta,
                                   "capacity": cap},
                  "results": results, "problems": problems,
                  "pass": not problems}
        out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        for label, r in results.items():
            pass
        # Save deterministic plan files for replay.
        for label, nh in (("q", heads), ("k", kv_heads)):
            width = nh * hd
            p = compile_rkir(
                rope_decode_indexed_module(width=width, capacity=cap,
                                           name=f"smollm_decode_indexed_{label}"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            save_plan(p, str(out.with_suffix(f".{label}.rkplan.json")))
        print(json.dumps(report, indent=1))
        return 0 if not problems else 1
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
