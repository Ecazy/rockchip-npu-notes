#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-model A/B/C for creation-time command-BO packing reuse.

All runners share one compiled graph, one weight object and one fd.  Only the
workspace-snapshotted command-cache flags differ:

  control      RK_PLAN_COMMAND_CACHE=0
  cache_sync   RK_PLAN_COMMAND_CACHE=1, RK_PLAN_COMMAND_CACHE_SYNC=1
  cache_nosync RK_PLAN_COMMAND_CACHE=1, RK_PLAN_COMMAND_CACHE_SYNC=0

The sync arm keeps the historical command-BO prep/fini pair every submit, so
the difference between cache_sync and cache_nosync is exactly the sync decision.
No node proxies or stats are installed in the timed path.
"""
from __future__ import annotations

import argparse
import gzip
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

CACHE_FLAG = "RK_PLAN_COMMAND_CACHE"
SYNC_FLAG = "RK_PLAN_COMMAND_CACHE_SYNC"
ARMS = ("control", "cache_sync", "cache_nosync")


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">III", f.read(12))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def summary(values):
    vals = sorted(float(x) for x in values)
    if not vals:
        return {}
    return {
        "n": len(vals),
        "median_us": float(statistics.median(vals)),
        "p10_us": float(vals[(len(vals) - 1) // 10]),
        "min_us": float(vals[0]),
        "max_us": float(vals[-1]),
    }


def make_runner(lib, program, weights, weights_dir, fd, arm):
    old_cache = os.environ.get(CACHE_FLAG)
    old_sync = os.environ.get(SYNC_FLAG)
    try:
        if arm == "control":
            os.environ[CACHE_FLAG] = "0"
            os.environ.pop(SYNC_FLAG, None)
        elif arm == "cache_sync":
            os.environ[CACHE_FLAG] = "1"
            os.environ[SYNC_FLAG] = "1"
        else:
            os.environ[CACHE_FLAG] = "1"
            os.environ[SYNC_FLAG] = "0"
        started = time.perf_counter()
        runner = HybridRunner(
            lib, program, weights,
            weights_path=weights_dir / "tiny_vit.npz",
            fd=fd, execution_mode="graph")
        return runner, (time.perf_counter() - started) * 1e6
    finally:
        if old_cache is None:
            os.environ.pop(CACHE_FLAG, None)
        else:
            os.environ[CACHE_FLAG] = old_cache
        if old_sync is None:
            os.environ.pop(SYNC_FLAG, None)
        else:
            os.environ[SYNC_FLAG] = old_sync


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
    runners = {}
    setup_us = {}
    try:
        for arm in ARMS:
            runners[arm], setup_us[arm] = make_runner(
                lib, program, weights, args.weights_dir, fd, arm)
        for arm in ARMS:
            runners[arm].run(images[:args.batch], stats=None)
        calls = args.limit // args.batch
        samples = {arm: [] for arm in ARMS}
        round_rows = []
        all_logits = {arm: np.empty((args.limit, 10), dtype=np.float32)
                      for arm in ARMS}
        bitwise = True
        for round_index in range(args.rounds):
            call_times = {arm: [] for arm in ARMS}
            for call_index in range(calls):
                start = call_index * args.batch
                stop = start + args.batch
                shift = (call_index + round_index) % len(ARMS)
                order = ARMS[shift:] + ARMS[:shift]
                outputs = {}
                times = {}
                for arm in order:
                    started = time.perf_counter()
                    result = runners[arm].run(images[start:stop], stats=None)
                    times[arm] = (time.perf_counter() - started) * 1e6
                    outputs[arm] = np.array(result.outputs["logits"], copy=True)
                first = outputs[ARMS[0]]
                if any(not np.array_equal(first, outputs[arm])
                       for arm in ARMS[1:]):
                    bitwise = False
                    raise RuntimeError("full-model outputs differ bitwise")
                for arm in ARMS:
                    samples[arm].append(times[arm])
                    call_times[arm].append(times[arm])
                    all_logits[arm][start:stop] = outputs[arm]
            med = {arm: float(statistics.median(values))
                   for arm, values in call_times.items()}
            round_rows.append({
                "round": round_index + 1,
                "median_us": med,
                "ratio_control_over_cache_sync":
                    med["control"] / med["cache_sync"],
                "ratio_control_over_cache_nosync":
                    med["control"] / med["cache_nosync"],
                "ratio_cache_sync_over_cache_nosync":
                    med["cache_sync"] / med["cache_nosync"],
            })
        for arm, value in all_logits.items():
            np.save(args.out.parent / f"{arm}_logits.npy", value)
        ratios = {key: [row[key] for row in round_rows] for key in (
            "ratio_control_over_cache_sync",
            "ratio_control_over_cache_nosync",
            "ratio_cache_sync_over_cache_nosync")}
        result = {
            "schema_version": 1,
            "kind": "full-model-command-cache-ab",
            "flags": {"cache": CACHE_FLAG, "sync": SYNC_FLAG},
            "limit": args.limit,
            "batch": args.batch,
            "rounds": args.rounds,
            "setup_us": setup_us,
            "bitwise": bitwise,
            "timing": {arm: summary(samples[arm]) for arm in ARMS},
            "rounds_detail": round_rows,
            "round_ratio_median": {
                key: float(statistics.median(values))
                for key, values in ratios.items()},
            "round_ratio_p10": {
                key: float(sorted(values)[(len(values) - 1) // 10])
                for key, values in ratios.items()},
            "output_files": {arm: f"{arm}_logits.npy" for arm in ARMS},
            "note": "Same compiled graph, weights, fd and timed interval. "
                    "Only the two creation-time command-cache flags differ; "
                    "program cache, pack, guard, download, publication and "
                    "output sync are otherwise identical.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        for runner in runners.values():
            runner.close()
        lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
