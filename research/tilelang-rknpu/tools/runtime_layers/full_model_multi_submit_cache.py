#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-runner interleaved five-profile A/B for multi-submit static preparation.

The control arm sets RK_PLAN_MULTI_SUBMIT_CACHE=0 (the historical single-submit
strategy); the prepared arm leaves it at the adopted default (1).  Both runners
share one fd and one compiled graph/weights object.  Each call alternates arm
order so process drift cancels.  Statistics, metric calculation and serialization
happen outside the timed interval.
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

from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.models import MLPRunner, TinyViTRunner, load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

FLAG = "RK_PLAN_MULTI_SUBMIT_CACHE"
PROFILES = (
    ("mlp_b32", "mnist_mlp", 32),
    ("mlp_b8", "mnist_mlp", 8),
    ("mlp_b1", "mnist_mlp", 1),
    ("vit_b8", "tiny_vit", 8),
    ("vit_b1", "tiny_vit", 1),
)


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def make_runner(name, model, lib, plans_dir, weights, weights_dir, fd, arm):
    manifest_name = (f"mnist_mlp_b{name.split('b')[-1]}" if model == "mnist_mlp"
                     else f"tiny_vit_b{name.split('b')[-1]}")
    old = os.environ.get(FLAG)
    try:
        os.environ[FLAG] = "0" if arm == "control" else "1"
        started = time.perf_counter()
        if model == "mnist_mlp":
            runner = MLPRunner(
                lib, plans_dir / f"{manifest_name}.rkhybrid.json", weights,
                int(name.split("b")[-1]), fd=fd,
                weights_path=weights_dir / "mnist_mlp.npz")
        else:
            runner = TinyViTRunner(
                lib, plans_dir / f"{manifest_name}.rkhybrid.json", weights,
                int(name.split("b")[-1]), fd=fd,
                weights_path=weights_dir / "tiny_vit.npz")
        return runner, (time.perf_counter() - started) * 1e6
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.limit <= 0 or args.rounds <= 0:
        raise SystemExit("limit and rounds must be positive")
    images = read_idx(args.data_dir / "t10k-images-idx3-ubyte.gz", 3)[:args.limit]
    labels = read_idx(args.data_dir / "t10k-labels-idx1-ubyte.gz", 1).astype(
        np.int64)[:args.limit]
    lib = LibRKPlan(args.lib)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise RuntimeError(f"rocket_open failed: {fd}")
    weights = {
        "mnist_mlp": load_weight_npz(args.weights_dir / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(args.weights_dir / "tiny_vit.npz"),
    }
    result = {
        "schema_version": 1,
        "kind": "full-model-multi-submit-cache-ab",
        "flag": FLAG,
        "limit": args.limit,
        "rounds": args.rounds,
        "profiles": {},
        "samples": [],
        "note": "Two runners in one process and one fd; call order alternates "
                "per call and round. Control=0, prepared=default 1. "
                "stats=None; no node proxies.",
    }
    try:
        for name, model, batch in PROFILES:
            if args.limit % batch:
                raise SystemExit(f"limit must be divisible by {batch} for {name}")
            manifest_name = (f"mnist_mlp_b{batch}" if model == "mnist_mlp"
                             else f"tiny_vit_b{batch}")
            program = load_hybrid_program(
                args.plans_dir / f"{manifest_name}.rkhybrid.json",
                plans_dir=args.plans_dir)
            control, control_setup = make_runner(
                name, model, lib, args.plans_dir, weights[model],
                args.weights_dir, fd, "control")
            prepared, prepared_setup = make_runner(
                name, model, lib, args.plans_dir, weights[model],
                args.weights_dir, fd, "prepared")
            runners = {"control": control, "prepared": prepared}
            try:
                control.run(images[:batch], stats=None)
                prepared.run(images[:batch], stats=None)
                calls = args.limit // batch
                per_round = []
                call_times = {"control": [], "prepared": []}
                control_logits = np.empty((args.limit, 10), dtype=np.float32)
                prepared_logits = np.empty((args.limit, 10), dtype=np.float32)
                bitwise = True
                for round_index in range(args.rounds):
                    round_times = {"control": [], "prepared": []}
                    for call_index in range(calls):
                        start = call_index * batch
                        stop = start + batch
                        order = (("control", "prepared") if
                                 (call_index + round_index) % 2 == 0 else
                                 ("prepared", "control"))
                        outputs = {}
                        timings = {}
                        for arm in order:
                            started = time.perf_counter()
                            outputs[arm] = np.array(
                                runners[arm].run(images[start:stop], stats=None),
                                copy=True)
                            timings[arm] = (time.perf_counter() - started) * 1e6
                        if not np.array_equal(outputs["control"],
                                              outputs["prepared"]):
                            bitwise = False
                            raise RuntimeError(f"{name} outputs differ bitwise")
                        control_logits[start:stop] = outputs["control"]
                        prepared_logits[start:stop] = outputs["prepared"]
                        for arm in ("control", "prepared"):
                            round_times[arm].append(timings[arm])
                            call_times[arm].append(timings[arm])
                        result["samples"].append({
                            "profile": name,
                            "round": round_index + 1,
                            "call_index": call_index,
                            "offset": start,
                            "size": batch,
                            "order": list(order),
                            "seconds_by_arm": timings,
                        })
                    med = {arm: float(statistics.median(round_times[arm]))
                           for arm in ("control", "prepared")}
                    total_us = {arm: sum(round_times[arm]) for arm in
                                ("control", "prepared")}
                    per_round.append({
                        "round": round_index + 1,
                        "median_us": med,
                        "throughput_images_s": {
                            arm: args.limit / max(1e-12, total_us[arm] / 1e6)
                            for arm in ("control", "prepared")},
                        "ratio_control_over_prepared":
                            med["control"] / med["prepared"],
                    })
                ratios = [row["ratio_control_over_prepared"]
                          for row in per_round]
                accuracy = {
                    "control": float((control_logits.argmax(axis=1) ==
                                      labels).mean()),
                    "prepared": float((prepared_logits.argmax(axis=1) ==
                                       labels).mean()),
                }
                result["profiles"][name] = {
                    "model": model,
                    "batch": batch,
                    "setup_us": {"control": control_setup,
                                 "prepared": prepared_setup},
                    "bitwise": bitwise,
                    "accuracy": accuracy,
                    "logits_sha256": {
                        "control": digest(control_logits),
                        "prepared": digest(prepared_logits)},
                    "timing": {arm: summary(call_times[arm])
                               for arm in ("control", "prepared")},
                    "rounds": per_round,
                    "round_ratio_median": float(statistics.median(ratios)),
                    "round_ratio_p10": float(sorted(ratios)[
                        (len(ratios) - 1) // 10]),
                }
                np.save(args.out.parent / f"{name}.control_logits.npy",
                        control_logits)
                np.save(args.out.parent / f"{name}.prepared_logits.npy",
                        prepared_logits)
            finally:
                control.close()
                prepared.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: {kk: v[kk] for kk in
                              ("timing", "round_ratio_median",
                               "round_ratio_p10", "bitwise", "accuracy",
                               "setup_us")}
                          for k, v in result["profiles"].items()},
                         indent=2, sort_keys=True))
    finally:
        lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
