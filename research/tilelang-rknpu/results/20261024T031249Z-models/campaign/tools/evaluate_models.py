#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate the complete frozen MNIST MLP and hybrid Tiny Transformer."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend import load_plan
from rknpu_backend.models import (MLPRunner, RunStats, TinyViTRunner,
                                  load_weight_npz)
from rknpu_backend.model_reference import (mlp_logits, mlp_logits_declared,
                                           tiny_vit_logits,
                                           tiny_vit_logits_declared)
from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                      default_library)

ROOT = Path(__file__).resolve().parents[1]


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_test(data_dir: Path):
    x = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    return x, y


def metrics(logits: np.ndarray, reference: np.ndarray, labels: np.ndarray,
            declared: np.ndarray | None = None) -> dict:
    pred = logits.argmax(axis=1)
    ref_pred = reference.argmax(axis=1)
    diff = np.abs(logits.astype(np.float32) - reference.astype(np.float32))
    out = {
        "n": int(labels.size),
        "accuracy": float((pred == labels).mean()),
        "cpu_accuracy": float((ref_pred == labels).mean()),
        "accuracy_delta": float((pred == labels).mean() - (ref_pred == labels).mean()),
        "top1_agreement": float((pred == ref_pred).mean()),
        "mean_abs_logit_error": float(diff.mean()),
        "max_abs_logit_error": float(diff.max()),
    }
    if declared is not None:
        dref = declared.astype(np.float32)
        ddiff = np.abs(logits.astype(np.float32) - dref)
        out.update({
            "declared_accuracy": float((declared.argmax(axis=1) == labels).mean()),
            "declared_top1_agreement": float((pred == declared.argmax(axis=1)).mean()),
            "declared_mean_abs_logit_error": float(ddiff.mean()),
            "declared_max_abs_logit_error": float(ddiff.max()),
        })
    return out


def _load_model_plans(manifest: dict, plans_dir: Path, prefix: str) -> dict:
    return {name: load_plan(str(plans_dir / info["path"]))
            for name, info in manifest["plans"].items()
            if name.startswith(prefix)}


def _run_mlp_profile(lib, plans_dir, manifest, weights, x, y, batch, fd, stats):
    prefix = f"mnist_mlp_b{batch}"
    info = manifest["plans"][prefix]
    runner = MLPRunner(lib, load_plan(str(plans_dir / info["path"])), weights,
                       batch, fd=fd)
    logits = np.empty((len(x), 10), dtype=np.float32)
    t0 = time.perf_counter()
    try:
        for i in range(0, len(x) - (len(x) % batch), batch):
            logits[i:i + batch] = runner.run(x[i:i + batch], stats=stats)
        rem = len(x) % batch
        if rem:
            # Tail uses the compiled smaller profile, never a dynamic shape.
            tail_batch = 8 if rem % 8 == 0 and f"mnist_mlp_b8" in manifest["plans"] else 1
            tail = MLPRunner(lib, load_plan(str(plans_dir / manifest["plans"][f"mnist_mlp_b{tail_batch}"]["path"])),
                             weights, tail_batch, fd=fd)
            for i in range(len(x) - rem, len(x), tail_batch):
                logits[i:i + tail_batch] = tail.run(x[i:i + tail_batch], stats=stats)
            tail.close()
    finally:
        runner.close()
    t_ref = time.perf_counter()
    ref = mlp_logits(weights, x)
    cpu_reference_seconds = time.perf_counter() - t_ref
    declared = mlp_logits_declared(weights, x)
    result = metrics(logits, ref, y, declared)
    result.update({"batch": batch, "wall_seconds": time.perf_counter() - t0,
                   "cpu_reference_seconds": cpu_reference_seconds,
                   "throughput_images_s": float(len(y) / max(1e-9, time.perf_counter() - t0)),
                   "stats": stats.to_dict()})
    return result, logits, ref


def _run_vit_profile(lib, plans_dir, manifest, weights, x, y, batch, fd, stats):
    prefix = f"tiny_vit_b{batch}_"
    plans = {name[len(prefix):]: load_plan(str(plans_dir / info["path"]))
             for name, info in manifest["plans"].items() if name.startswith(prefix)}
    runner = TinyViTRunner(lib, plans, weights, batch, fd=fd)
    logits = np.empty((len(x) - len(x) % batch, 10), dtype=np.float32)
    t0 = time.perf_counter()
    try:
        for i in range(0, len(logits), batch):
            logits[i:i + batch] = runner.run(x[i:i + batch], stats=stats)
    finally:
        runner.close()
    xe = x if len(x) == len(logits) else x[:len(logits)]
    labels = y if len(y) == len(logits) else y[:len(logits)]
    t_ref = time.perf_counter()
    ref = tiny_vit_logits(weights, xe)
    cpu_reference_seconds = time.perf_counter() - t_ref
    declared = tiny_vit_logits_declared(weights, xe)
    result = metrics(logits, ref, labels, declared)
    result.update({"batch": batch, "wall_seconds": time.perf_counter() - t0,
                   "cpu_reference_seconds": cpu_reference_seconds,
                   "throughput_images_s": float(len(labels) / max(1e-9, time.perf_counter() - t0)),
                   "stats": stats.to_dict()})
    return result, logits, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default=str(ROOT / "models" / "compiled"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--lib", default=os.environ.get("RKNPU_PLAN_LIB", ""))
    ap.add_argument("--fd", type=int, default=None,
                    help="use this already-open device fd; otherwise mock fd=0")
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--models", choices=("mlp", "vit", "both"), default="both")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "model_eval.json"))
    ap.add_argument("--profile-samples", type=int, default=512,
                    help="sample count for non-full batch profiles")
    ap.add_argument("--global-limit", type=int, default=0,
                    help="limit the loaded test split; 0 means the full split")
    args = ap.parse_args()
    plans_dir = Path(args.plans_dir).resolve()
    manifest = json.loads((plans_dir / "manifest.json").read_text())
    x, y = load_test(Path(args.data_dir))
    if args.global_limit:
        x, y = x[:args.global_limit], y[:args.global_limit]
    if args.backend == "mock":
        lib_path = args.lib or str(ROOT / ".." / "custom-ops" / "build-host-mvp" /
                                   "librknpu_plan_mock.so")
        fd = 0
        lib = LibRKPlan(lib_path)
    else:
        lib = default_library() if not args.lib else LibRKPlan(args.lib)
        configure_rocket_abi(lib.cdll)
        fd = args.fd
        if fd is None:
            fd = int(lib.cdll.rocket_open())
            if fd < 0:
                raise RuntimeError(f"rocket_open failed: {fd}")
    report = {"backend": args.backend, "manifest_profile_hash": manifest["profile_hash"],
              "test_images": int(len(x)),
              "cpu_threads": os.cpu_count(),
              "cpu_env_threads": {k: os.environ.get(k)
                                  for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS")},
              "models": []}
    try:
        if args.models in {"mlp", "both"}:
            w = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
            for batch in (32, 8, 1):
                full = batch == 32
                xe = x if full else x[:args.profile_samples]
                ye = y if full else y[:args.profile_samples]
                stats = RunStats()
                result, logits, ref = _run_mlp_profile(
                    lib, plans_dir, manifest, w, xe, ye, batch, fd, stats)
                result["full_test_split"] = full
                result["model"] = "mnist_mlp"
                report["models"].append(result)
                np.save(str(Path(args.out).with_suffix(f".mlp_b{batch}.logits.npy")), logits)
                np.save(str(Path(args.out).with_suffix(f".mlp_b{batch}.ref.npy")), ref)
                print("MLP", batch, result["accuracy"], result["top1_agreement"], flush=True)
        if args.models in {"vit", "both"}:
            w = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
            for batch in (8, 1):
                full = batch == 8
                xe = x if full else x[:args.profile_samples]
                ye = y if full else y[:args.profile_samples]
                stats = RunStats()
                result, logits, ref = _run_vit_profile(
                    lib, plans_dir, manifest, w, xe, ye, batch, fd, stats)
                result["full_test_split"] = full
                result["model"] = "tiny_vit"
                report["models"].append(result)
                np.save(str(Path(args.out).with_suffix(f".vit_b{batch}.logits.npy")), logits)
                np.save(str(Path(args.out).with_suffix(f".vit_b{batch}.ref.npy")), ref)
                print("ViT", batch, result["accuracy"], result["top1_agreement"], flush=True)
    finally:
        if args.backend == "board" and args.fd is None and fd >= 0:
            lib.cdll.rocket_close(fd)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
