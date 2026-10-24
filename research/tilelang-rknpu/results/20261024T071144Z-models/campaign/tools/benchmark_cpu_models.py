#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU FP32 / declared-path baselines using the model's static batch schedule.

The former benchmark fed the entire 10,000-image split to one NumPy matmul while
the NPU used batch 32/8.  Schema v2 uses the exact compiled batch grouping and
records the actual CPU affinity, process thread count, environment and BLAS pool
sizes so the comparison basis is explicit.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hostinfo import thread_config
from rknpu_backend.models import load_weight_npz
from rknpu_backend.model_reference import (mlp_logits, mlp_logits_declared,
                                           tiny_vit_logits,
                                           tiny_vit_logits_declared)

ROOT = Path(__file__).resolve().parents[1]
TIMING_SCHEMA_VERSION = 2


def read_idx(path: Path, rank: int):
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def make_batch_schedule(n: int, primary: int,
                        tail_profiles: tuple[int, ...] = ()) -> list[dict]:
    schedule = [{"offset": i * primary, "size": primary, "profile": primary,
                 "role": "primary"} for i in range(n // primary)]
    rem, offset = n % primary, (n // primary) * primary
    candidates = tuple(sorted((b for b in tail_profiles if 0 < b < primary),
                              reverse=True))
    while rem:
        size = next((b for b in candidates if b <= rem), None)
        if size is None:
            raise ValueError(f"no tail profile covers remainder {rem}")
        schedule.append({"offset": offset, "size": size, "profile": size,
                         "role": "tail"})
        offset += size
        rem -= size
    return schedule


def schedule_summary(schedule: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for row in schedule:
        key = str(row["size"])
        hist[key] = hist.get(key, 0) + 1
    return {"calls": len(schedule),
            "batch_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "tail_calls": sum(row["role"] == "tail" for row in schedule),
            "no_dynamic_shapes": True}


def _schedule_logits(fn, weights, x, schedule):
    out = np.empty((len(x), 10), dtype=np.float32)
    for row in schedule:
        start, size = row["offset"], row["size"]
        out[start:start + size] = fn(weights, x[start:start + size])
    return out


def _measure(fn, weights, x, schedule, repeats: int):
    # Warm every code path (including the tail) once.
    logits = _schedule_logits(fn, weights, x, schedule)
    core_times: list[float] = []
    argmax_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        logits = _schedule_logits(fn, weights, x, schedule)
        core_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = logits.argmax(axis=1)
        argmax_s = time.perf_counter() - t0
        core_times.append(core_s)
        argmax_times.append(argmax_s)
    core_s = float(np.median(core_times))
    argmax_s = float(np.median(argmax_times))
    return logits, {
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "reference_timing": "not_applicable_cpu_baseline",
        "execution_seconds": core_s,
        "postprocess_seconds": argmax_s,
        "end_to_end_seconds": core_s + argmax_s,
        "throughput_images_s": len(x) / max(1e-12, core_s + argmax_s),
        "throughput_basis": "same static batch schedule as NPU; includes "
                            "preprocess+forward+argmax, references excluded",
        "repeats": repeats,
        "samples_seconds": core_times,
        "argmax_samples_seconds": argmax_times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "evidence" /
                                         "cpu_benchmark_v2.json"))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(Path(args.data_dir) / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    mlp = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
    vit = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
    schedules = {
        "mnist_mlp": make_batch_schedule(len(x), 32, tail_profiles=(8, 1)),
        "tiny_vit": make_batch_schedule(len(x), 8, tail_profiles=()),
    }
    config_before = thread_config()
    logits_mlp, time_mlp = _measure(
        mlp_logits, mlp, x, schedules["mnist_mlp"], args.repeats)
    logits_vit, time_vit = _measure(
        tiny_vit_logits, vit, x, schedules["tiny_vit"], args.repeats)
    decl_mlp, time_decl_mlp = _measure(
        mlp_logits_declared, mlp, x, schedules["mnist_mlp"], args.repeats)
    decl_vit, time_decl_vit = _measure(
        tiny_vit_logits_declared, vit, x, schedules["tiny_vit"], args.repeats)
    config_after = thread_config()
    report = {
        "schema_version": 2,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "host_label": "benchmark-host",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "test_images": int(len(x)),
        "thread_config_before": config_before,
        "thread_config_after": config_after,
        "cpu_thread_config": config_before,
        "batch_policy": {
            name: schedule_summary(schedule) for name, schedule in schedules.items()
        },
        "models": {
            "mnist_mlp": {
                **time_mlp,
                "fp32_accuracy": float((logits_mlp.argmax(1) == y).mean()),
                "declared": {
                    **time_decl_mlp,
                    "accuracy": float((decl_mlp.argmax(1) == y).mean()),
                },
            },
            "tiny_vit": {
                **time_vit,
                "fp32_accuracy": float((logits_vit.argmax(1) == y).mean()),
                "declared": {
                    **time_decl_vit,
                    "accuracy": float((decl_vit.argmax(1) == y).mean()),
                },
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
