#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full Tiny ViT b8 A/B test of the experimental program-cache cache.

Both runners are identical compiled graphs.  The current runner is created with
the cache disabled; the experimental runner is created after enabling
RK_PLAN_PROGRAM_CACHE, so the environment is snapshotted at each workspace
creation.  No pointer arrays are rebuilt by a wrapper and no node proxies are
installed in the timed path.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
from rknpu_backend.models import load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

FLAG = "RK_PLAN_PROGRAM_CACHE"


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">III", f.read(12))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def summary(values):
    vals = sorted(float(x) for x in values)
    if not vals:
        return {}
    return {"n": len(vals), "median_us": float(statistics.median(vals)),
            "p10_us": float(vals[(len(vals) - 1) // 10]),
            "min_us": float(vals[0]), "max_us": float(vals[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.limit % args.batch:
        raise SystemExit("limit must be divisible by batch")
    images = read_idx(args.data_dir / "t10k-images-idx3-ubyte.gz")[:args.limit]
    program = load_hybrid_program(
        args.plans_dir / f"tiny_vit_b{args.batch}.rkhybrid.json",
        plans_dir=args.plans_dir)
    weights = load_weight_npz(args.weights_dir / "tiny_vit.npz")
    lib = LibRKPlan(args.lib)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise RuntimeError(f"rocket_open failed: {fd}")
    current = experimental = None
    old = os.environ.get(FLAG)
    try:
        os.environ.pop(FLAG, None)
        t0 = time.perf_counter()
        current = HybridRunner(lib, program, weights,
                               weights_path=args.weights_dir / "tiny_vit.npz",
                               fd=fd, execution_mode="graph")
        current_setup_us = (time.perf_counter() - t0) * 1e6
        os.environ[FLAG] = "1"
        t0 = time.perf_counter()
        experimental = HybridRunner(lib, program, weights,
                                    weights_path=args.weights_dir / "tiny_vit.npz",
                                    fd=fd, execution_mode="graph")
        experimental_setup_us = (time.perf_counter() - t0) * 1e6
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old
        current.run(images[:args.batch], stats=None)
        experimental.run(images[:args.batch], stats=None)
        calls = args.limit // args.batch
        samples = {"current": [], "experimental": []}
        round_rows = []
        all_logits = {
            "current": np.empty((args.limit, 10), dtype=np.float32),
            "experimental": np.empty((args.limit, 10), dtype=np.float32),
        }
        bitwise = True
        for round_index in range(args.rounds):
            call_times = {"current": [], "experimental": []}
            for call_index in range(calls):
                start = call_index * args.batch
                stop = start + args.batch
                rotate = (call_index + round_index) % 2
                order = (("current", "experimental") if rotate == 0 else
                         ("experimental", "current"))
                outputs = {}
                times = {}
                for name in order:
                    runner = current if name == "current" else experimental
                    t0 = time.perf_counter()
                    result = runner.run(images[start:stop], stats=None)
                    times[name] = (time.perf_counter() - t0) * 1e6
                    outputs[name] = np.array(result.outputs["logits"], copy=True)
                if not np.array_equal(outputs["current"],
                                      outputs["experimental"]):
                    bitwise = False
                    raise RuntimeError("full-model outputs differ bitwise")
                for name in ("current", "experimental"):
                    samples[name].append(times[name])
                    call_times[name].append(times[name])
                    all_logits[name][start:stop] = outputs[name]
            med = {name: float(statistics.median(values))
                   for name, values in call_times.items()}
            round_rows.append({
                "round": round_index + 1,
                "current_median_us": med["current"],
                "experimental_median_us": med["experimental"],
                "ratio_current_over_experimental":
                    med["current"] / med["experimental"],
            })
        per_round_ratios = [x["ratio_current_over_experimental"]
                            for x in round_rows]
        call_ratios = [a / b for a, b in zip(samples["current"],
                                             samples["experimental"])]
        for name, value in all_logits.items():
            np.save(args.out.parent / f"{name}_logits.npy", value)
        result = {
            "schema_version": 1,
            "kind": "full-model-program-cache-ab",
            "flag": FLAG,
            "limit": args.limit, "batch": args.batch,
            "rounds": args.rounds,
            "setup_us": {"current": current_setup_us,
                         "experimental": experimental_setup_us},
            "bitwise": bitwise,
            "trusted_timing": {
                "current": summary(samples["current"]),
                "experimental": summary(samples["experimental"]),
            },
            # Primary causal estimate: per-round paired medians.
            "rounds": round_rows,
            "round_ratio_median": float(statistics.median(per_round_ratios)),
            "round_ratio_p10": float(sorted(per_round_ratios)[
                (len(per_round_ratios) - 1) // 10]),
            "call_ratio_summary": summary(call_ratios),
            "output_files": {name: f"{name}_logits.npy"
                             for name in all_logits},
            "note": "Both arms are the same compiled graph. Only the "
                    "workspace-snapshotted program-cache flag differs; "
                    "finite scans, guard, sync, pack, emitter, command and "
                    "publication are otherwise identical.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old
        if current is not None:
            current.close()
        if experimental is not None:
            experimental.close()
        lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
