#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-side RKNN Lite2 and CPU comparison for the frozen MNIST models.

The RKNN graphs are exported from the same weights and FP32 math by
``export_vendor_models.py`` and compiled without quantization by
``build_vendor_rknn.py``.  This runner measures full application calls:
raw uint8 batch -> float input preparation -> RKNN inference -> output retrieval
and argmax.  CPU arms use the same static batch schedule and the same raw inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from rknpu_backend.models import load_weight_npz  # noqa: E402
from rknpu_backend.model_reference import (  # noqa: E402
    mlp_logits, mlp_logits_declared, tiny_vit_logits, tiny_vit_logits_declared)


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule(n: int, batch: int) -> list[dict]:
    if batch == 1:
        return [{"offset": i, "size": 1, "profile": 1, "role": "primary"}
                for i in range(n)]
    rows = [{"offset": i * batch, "size": batch, "profile": batch,
             "role": "primary"} for i in range(n // batch)]
    rem = n % batch
    offset = (n // batch) * batch
    # Match the frozen evaluator schedules.
    tails = (1,) if batch == 8 else (8, 1) if batch == 32 else ()
    while rem:
        size = next((v for v in sorted(tails, reverse=True) if v <= rem), None)
        if size is None:
            raise RuntimeError(f"no tail profile for {batch}, remainder {rem}")
        rows.append({"offset": offset, "size": size, "profile": size,
                     "role": "tail"})
        offset += size
        rem -= size
    return rows


def schedule_summary(rows: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for row in rows:
        key = str(row["size"])
        hist[key] = hist.get(key, 0) + 1
    return {"calls": len(rows), "batch_histogram": dict(
        sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "tail_calls": sum(row["role"] == "tail" for row in rows),
        "no_dynamic_shapes": True}


def metrics(logits: np.ndarray, cpu: np.ndarray, declared: np.ndarray,
            labels: np.ndarray) -> dict:
    pred = logits.argmax(1)
    cpu_pred = cpu.argmax(1)
    dec_pred = declared.argmax(1)
    diff = np.abs(logits.astype(np.float64) - cpu.astype(np.float64))
    ddiff = np.abs(logits.astype(np.float64) - declared.astype(np.float64))
    return {
        "n": int(labels.size),
        "accuracy": float((pred == labels).mean()),
        "cpu_accuracy": float((cpu_pred == labels).mean()),
        "accuracy_delta": float((pred == labels).mean() -
                                (cpu_pred == labels).mean()),
        "top1_agreement": float((pred == cpu_pred).mean()),
        "declared_accuracy": float((dec_pred == labels).mean()),
        "declared_top1_agreement": float((pred == dec_pred).mean()),
        "mean_abs_logit_error": float(diff.mean()),
        "max_abs_logit_error": float(diff.max(initial=0.0)),
        "declared_mean_abs_logit_error": float(ddiff.mean()),
        "declared_max_abs_logit_error": float(ddiff.max(initial=0.0)),
    }


def run_vendor(args) -> dict:
    from rknnlite.api import RKNNLite
    rknn_dir = Path(args.rknn_dir).resolve()
    manifest = json.loads((rknn_dir / "rknn-manifest.json").read_text())
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(Path(args.data_dir) / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    weights = {"mnist_mlp": load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz"),
               "tiny_vit": load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")}
    profiles = [("mnist_mlp", b) for b in (32, 8, 1)] + \
               [("tiny_vit", b) for b in (8, 1)]
    result = {"schema_version": 1, "kind": "vendor-rknn-runtime-eval",
              "backend": "rknn-toolkit-lite2", "profiles": []}
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    for model, batch in profiles:
        profile_x = x if args.limit == 0 else x[:args.limit]
        profile_y = y if args.limit == 0 else y[:args.limit]
        rows = schedule(len(profile_x), batch)
        needed = sorted({row["size"] for row in rows}, reverse=True)
        contexts = {}
        t0 = time.perf_counter()
        for b in needed:
            key = f"{model}_b{b}"
            entry = manifest["profiles"][key]
            ctx = RKNNLite(verbose=False)
            rc = ctx.load_rknn(str(rknn_dir / entry["rknn"]))
            if rc:
                raise RuntimeError(f"load_rknn failed {key}: {rc}")
            rc = ctx.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_0)
            if rc:
                raise RuntimeError(f"init_runtime failed {key}: {rc}")
            contexts[b] = ctx
        setup_seconds = time.perf_counter() - t0
        # Warm each distinct batch profile once, outside the timed interval.
        seen = set()
        t0 = time.perf_counter()
        for row in rows:
            if row["size"] in seen:
                continue
            seen.add(row["size"])
            inp = profile_x[row["offset"]:row["offset"] + row["size"]].reshape(
                row["size"], 784).astype(np.float32)
            contexts[row["size"]].inference(inputs=[inp], data_type="float32")
        warmup_seconds = time.perf_counter() - t0
        logits = np.empty((len(profile_x), 10), dtype=np.float32)
        t0 = time.perf_counter()
        for row in rows:
            start, size = row["offset"], row["size"]
            inp = profile_x[start:start + size].reshape(size, 784).astype(np.float32)
            out = contexts[size].inference(inputs=[inp], data_type="float32")
            if out is None or len(out) != 1 or out[0].shape != (size, 10):
                raise RuntimeError(f"unexpected RKNN output for {model} b{size}")
            logits[start:start + size] = np.asarray(out[0], dtype=np.float32)
        model_call_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = logits.argmax(1)
        argmax_seconds = time.perf_counter() - t0
        cpu = _refs_for_profile(model, weights, profile_x, rows, declared=False)
        declared = _refs_for_profile(model, weights, profile_x, rows, declared=True)
        item = {
            "model": model, "batch": batch, "n": int(len(profile_x)),
            "batch_schedule": schedule_summary(rows),
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "model_call_seconds": model_call_seconds,
            "argmax_seconds": argmax_seconds,
            "end_to_end_seconds": model_call_seconds + argmax_seconds,
            "throughput_images_s": len(profile_x) / max(
                1e-12, model_call_seconds + argmax_seconds),
            "throughput_basis": "app-side input preparation+RKNN inference+argmax",
            "metrics": metrics(logits, cpu, declared, profile_y),
            "context_batches": needed,
        }
        result["profiles"].append(item)
        np.save(Path(args.out).with_name(
            f"vendor.{model}_b{batch}.logits.npy"), logits.astype(np.float32))
        for ctx in contexts.values():
            ctx.release()
    return result


def _refs_for_profile(model, weights, x, rows, *, declared: bool):
    fn = {"mnist_mlp": mlp_logits_declared if declared else mlp_logits,
          "tiny_vit": tiny_vit_logits_declared if declared else tiny_vit_logits}[model]
    out = np.empty((len(x), 10), dtype=np.float32)
    for row in rows:
        start, size = row["offset"], row["size"]
        out[start:start + size] = fn(weights[model], x[start:start + size])
    return out


def run_cpu(args) -> dict:
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(Path(args.data_dir) / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    mlp = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
    vit = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
    weights = {"mnist_mlp": mlp, "tiny_vit": vit}
    profiles = [("mnist_mlp", b) for b in (32, 8, 1)] + \
               [("tiny_vit", b) for b in (8, 1)]
    result = {"schema_version": 1, "kind": "cpu-reference-eval",
              "backend": "numpy-fp32", "profiles": []}
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    for model, batch in profiles:
        profile_x = x if args.limit == 0 else x[:args.limit]
        profile_y = y if args.limit == 0 else y[:args.limit]
        rows = schedule(len(profile_x), batch)
        # CPU reference timing includes the same raw->FP32 preprocessing in
        # each reference call.  No candidate arithmetic is replaced.
        t0 = time.perf_counter()
        logits = _refs_for_profile(model, weights, profile_x, rows, declared=False)
        call_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = logits.argmax(1)
        argmax_seconds = time.perf_counter() - t0
        declared = _refs_for_profile(model, weights, profile_x, rows, declared=True)
        result["profiles"].append({
            "model": model, "batch": batch, "n": int(len(profile_x)),
            "batch_schedule": schedule_summary(rows),
            "setup_seconds": 0.0, "warmup_seconds": 0.0,
            "model_call_seconds": call_seconds,
            "argmax_seconds": argmax_seconds,
            "end_to_end_seconds": call_seconds + argmax_seconds,
            "throughput_images_s": len(profile_x) / max(
                1e-12, call_seconds + argmax_seconds),
            "throughput_basis": "CPU FP32 raw->preprocess->forward+argmax",
            "metrics": metrics(logits, logits, declared, profile_y),
            "context_batches": [],
        })
        np.save(Path(args.out).with_name(
            f"cpu.{model}_b{batch}.logits.npy"), logits.astype(np.float32))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("vendor", "cpu"), required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--rknn-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means full 10,000-image test split")
    args = ap.parse_args()
    result = run_vendor(args) if args.mode == "vendor" else run_cpu(args)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
