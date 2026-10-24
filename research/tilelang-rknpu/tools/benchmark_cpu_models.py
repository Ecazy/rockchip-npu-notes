#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU FP32 baseline timings for the two frozen models, same data/preprocessing."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.models import load_weight_npz
from rknpu_backend.model_reference import (mlp_logits, mlp_logits_declared,
                                           tiny_vit_logits,
                                           tiny_vit_logits_declared)

ROOT = Path(__file__).resolve().parents[1]


def read_idx(path: Path, rank: int):
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


def cpu_mhz():
    values = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("cpu MHz"):
                values.append(float(line.split(":", 1)[1]))
    except OSError:
        pass
    if not values:
        # ARM kernels commonly omit cpu MHz /proc entries; use the cpufreq
        # interface instead and convert kHz to MHz.
        for p in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
            try:
                values.append(float(p.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
    return {"min_mhz": min(values) if values else None,
            "max_mhz": max(values) if values else None,
            "mean_mhz": sum(values) / len(values) if values else None}


def _run(fn, *args, repeats: int):
    # one warmup, then median; all inputs are immutable and Host-only.
    fn(*args)
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        values.append(time.perf_counter() - t0)
    return out, float(np.median(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "evidence" / "cpu_benchmark.json"))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(Path(args.data_dir) / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    mlp = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
    vit = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
    logits_mlp, sec_mlp = _run(mlp_logits, mlp, x, repeats=args.repeats)
    logits_vit, sec_vit = _run(tiny_vit_logits, vit, x, repeats=args.repeats)
    decl_mlp, sec_decl_mlp = _run(mlp_logits_declared, mlp, x, repeats=args.repeats)
    decl_vit, sec_decl_vit = _run(tiny_vit_logits_declared, vit, x, repeats=args.repeats)
    report = {
        "host_label": "benchmark-host",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS")},
        "cpuinfo": cpu_mhz(),
        "test_images": int(len(x)),
        "models": {
            "mnist_mlp": {
                "fp32_seconds": sec_mlp,
                "fp32_images_s": len(x) / sec_mlp,
                "fp32_accuracy": float((logits_mlp.argmax(1) == y).mean()),
                "declared_seconds": sec_decl_mlp,
                "declared_accuracy": float((decl_mlp.argmax(1) == y).mean()),
            },
            "tiny_vit": {
                "fp32_seconds": sec_vit,
                "fp32_images_s": len(x) / sec_vit,
                "fp32_accuracy": float((logits_vit.argmax(1) == y).mean()),
                "declared_seconds": sec_decl_vit,
                "declared_accuracy": float((decl_vit.argmax(1) == y).mean()),
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
