#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate the frozen MNIST MLP and hybrid Tiny Transformer.

Timing schema v2 deliberately separates setup, warm-up, preprocessing,
inference/execution, postprocessing and reference calculation.  Reference
seconds never contribute to model latency or throughput.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rknpu_backend import load_plan
from rknpu_backend.hostinfo import thread_config
from rknpu_backend.models import (MLPRunner, RunStats, TinyViTRunner,
                                  load_weight_npz)
from rknpu_backend.model_reference import (mlp_logits, mlp_logits_declared,
                                           tiny_vit_logits,
                                           tiny_vit_logits_declared)
from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                      default_library)

ROOT = Path(__file__).resolve().parents[1]
TIMING_SCHEMA_VERSION = 2


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
    logits = np.asarray(logits)
    reference = np.asarray(reference)
    pred = logits.argmax(axis=1)
    ref_pred = reference.argmax(axis=1)
    diff = np.abs(logits.astype(np.float32) - reference.astype(np.float32))
    out = {
        "n": int(labels.size),
        "accuracy": float((pred == labels).mean()),
        "cpu_accuracy": float((ref_pred == labels).mean()),
        "accuracy_delta": float((pred == labels).mean() -
                                (ref_pred == labels).mean()),
        "top1_agreement": float((pred == ref_pred).mean()),
        "mean_abs_logit_error": float(diff.mean()),
        "max_abs_logit_error": float(diff.max()),
    }
    if declared is not None:
        dref = np.asarray(declared, dtype=np.float32)
        ddiff = np.abs(logits.astype(np.float32) - dref)
        out.update({
            "declared_accuracy": float((declared.argmax(axis=1) == labels).mean()),
            "declared_top1_agreement": float(
                (pred == declared.argmax(axis=1)).mean()),
            "declared_mean_abs_logit_error": float(ddiff.mean()),
            "declared_max_abs_logit_error": float(ddiff.max()),
        })
    return out


def make_batch_schedule(n: int, primary: int,
                        tail_profiles: tuple[int, ...] = (8, 1)) -> list[dict]:
    """Static batch schedule; every call uses a separately compiled profile."""
    if n < 0 or primary <= 0:
        raise ValueError("invalid batch schedule")
    schedule: list[dict] = []
    full = n // primary
    for index in range(full):
        schedule.append({"offset": index * primary, "size": primary,
                         "profile": primary, "role": "primary"})
    rem = n % primary
    offset = full * primary
    candidates = tuple(sorted((b for b in tail_profiles if b < primary and b > 0),
                              reverse=True))
    while rem:
        size = next((b for b in candidates if b <= rem), None)
        if size is None:
            raise ValueError(f"no compiled tail profile covers remainder {rem}")
        schedule.append({"offset": offset, "size": size,
                         "profile": size, "role": "tail"})
        offset += size
        rem -= size
    return schedule


def schedule_summary(schedule: list[dict]) -> dict:
    hist = Counter(str(row["size"]) for row in schedule)
    return {
        "calls": len(schedule),
        "batch_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "tail_calls": sum(row["role"] == "tail" for row in schedule),
        "no_dynamic_shapes": True,
    }


def _reference_with_schedule(fn, weights, x, schedule):
    out = np.empty((len(x), 10), dtype=np.float32)
    for row in schedule:
        start, size = row["offset"], row["size"]
        out[start:start + size] = fn(weights, x[start:start + size])
    return out


def _model_plan(manifest: dict, plans_dir: Path, name: str):
    info = manifest["plans"][name]
    return load_plan(str(plans_dir / info["path"]))


def _build_mlp_runners(lib, plans_dir, manifest, weights, schedule, fd):
    t0 = time.perf_counter()
    profiles = sorted({row["profile"] for row in schedule}, reverse=True)
    runners = {}
    try:
        for batch in profiles:
            runners[batch] = MLPRunner(
                lib, _model_plan(manifest, plans_dir, f"mnist_mlp_b{batch}"),
                weights, batch, fd=fd)
    except BaseException:
        for runner in reversed(list(runners.values())):
            try:
                runner.close()
            except BaseException:
                pass
        raise
    return runners, time.perf_counter() - t0


def _run_mlp_profile(lib, plans_dir, manifest, weights, x, y, batch, fd,
                     *, weight_load_seconds: float, primary_full: bool):
    """Evaluate one static MLP profile; reference time is outside the interval."""
    schedule = make_batch_schedule(len(x), batch, tail_profiles=(8, 1))
    runners, setup_seconds = _build_mlp_runners(
        lib, plans_dir, manifest, weights, schedule, fd)
    stats = RunStats()
    warm_stats = RunStats()
    logits = np.empty((len(x), 10), dtype=np.float32)
    try:
        # Warm each distinct compiled profile once.  This is recorded separately
        # and never included in the timed interval.
        t0 = time.perf_counter()
        seen = set()
        for row in schedule:
            batch_id = row["profile"]
            if batch_id in seen:
                continue
            seen.add(batch_id)
            start, size = row["offset"], row["size"]
            runners[batch_id].run(x[start:start + size], stats=warm_stats)
        warmup_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        for row in schedule:
            start, size = row["offset"], row["size"]
            logits[start:start + size] = runners[row["profile"]].run(
                x[start:start + size], stats=stats)
        model_call_seconds = time.perf_counter() - t0

        t_argmax = time.perf_counter()
        _ = logits.argmax(axis=1)
        argmax_seconds = time.perf_counter() - t_argmax

        t_ref = time.perf_counter()
        ref = _reference_with_schedule(mlp_logits, weights, x, schedule)
        cpu_reference_seconds = time.perf_counter() - t_ref
        t_ref = time.perf_counter()
        declared = _reference_with_schedule(
            mlp_logits_declared, weights, x, schedule)
        declared_reference_seconds = time.perf_counter() - t_ref

        result = metrics(logits, ref, y, declared)
        timing = {
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "reference_timing": "excluded_from_model_interval",
            "weight_load_seconds": weight_load_seconds,
            "runner_setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "preprocess_seconds": stats.preprocess_us / 1e6,
            "execution_seconds": stats.execution_us / 1e6,
            "postprocess_seconds": stats.postprocess_us / 1e6 + argmax_seconds,
            "argmax_seconds": argmax_seconds,
            "model_call_seconds": model_call_seconds,
            "end_to_end_seconds": model_call_seconds + argmax_seconds,
            "plan_call_seconds": stats.plan_call_us / 1e6,
            "host_action_seconds": stats.host_action_us / 1e6,
            "unattributed_seconds": max(
                0.0, stats.end_to_end_us / 1e6 - (
                    stats.preprocess_us + stats.execution_us +
                    stats.postprocess_us) / 1e6),
            "reference_seconds": {
                "cpu_fp32": cpu_reference_seconds,
                "declared_fp16_path": declared_reference_seconds,
            },
        }
        throughput = len(y) / max(1e-12, timing["end_to_end_seconds"])
        result.update({
            "batch": batch,
            "evaluation_set": ("full_test_split" if primary_full
                               else "admission_subsample"),
            "full_test_split": bool(primary_full),
            "batch_schedule": schedule_summary(schedule),
            "timing": timing,
            "throughput_images_s": throughput,
            "throughput_basis": "end_to_end_seconds"
                                 " (preprocess+execution+postprocess,"
                                 " references excluded)",
            "stats": stats.to_dict(),
        })
        return result, logits, ref, declared
    finally:
        for runner in reversed(list(runners.values())):
            runner.close()


def _build_vit_runners(lib, plans_dir, manifest, weights, batch, fd):
    t0 = time.perf_counter()
    prefix = f"tiny_vit_b{batch}_"
    plans = {name[len(prefix):]:
             load_plan(str(plans_dir / info["path"]))
             for name, info in manifest["plans"].items()
             if name.startswith(prefix)}
    try:
        runner = TinyViTRunner(lib, plans, weights, batch, fd=fd)
    except BaseException:
        raise
    return runner, time.perf_counter() - t0


def _run_vit_profile(lib, plans_dir, manifest, weights, x, y, batch, fd,
                     *, weight_load_seconds: float, primary_full: bool):
    schedule = make_batch_schedule(len(x), batch, tail_profiles=())
    runner, setup_seconds = _build_vit_runners(
        lib, plans_dir, manifest, weights, batch, fd)
    stats = RunStats()
    warm_stats = RunStats()
    logits = np.empty((len(x), 10), dtype=np.float32)
    try:
        first = schedule[0]
        t0 = time.perf_counter()
        runner.run(x[first["offset"]:first["offset"] + first["size"]],
                   stats=warm_stats)
        warmup_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        for row in schedule:
            start, size = row["offset"], row["size"]
            logits[start:start + size] = runner.run(
                x[start:start + size], stats=stats)
        model_call_seconds = time.perf_counter() - t0

        t_argmax = time.perf_counter()
        _ = logits.argmax(axis=1)
        argmax_seconds = time.perf_counter() - t_argmax

        t_ref = time.perf_counter()
        ref = _reference_with_schedule(tiny_vit_logits, weights, x, schedule)
        cpu_reference_seconds = time.perf_counter() - t_ref
        t_ref = time.perf_counter()
        declared = _reference_with_schedule(
            tiny_vit_logits_declared, weights, x, schedule)
        declared_reference_seconds = time.perf_counter() - t_ref

        result = metrics(logits, ref, y, declared)
        timing = {
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "reference_timing": "excluded_from_model_interval",
            "weight_load_seconds": weight_load_seconds,
            "runner_setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "preprocess_seconds": stats.preprocess_us / 1e6,
            "execution_seconds": stats.execution_us / 1e6,
            "postprocess_seconds": stats.postprocess_us / 1e6 + argmax_seconds,
            "argmax_seconds": argmax_seconds,
            "model_call_seconds": model_call_seconds,
            "end_to_end_seconds": model_call_seconds + argmax_seconds,
            "plan_call_seconds": stats.plan_call_us / 1e6,
            "host_action_seconds": stats.host_action_us / 1e6,
            "unattributed_seconds": max(
                0.0, stats.end_to_end_us / 1e6 - (
                    stats.preprocess_us + stats.execution_us +
                    stats.postprocess_us) / 1e6),
            "reference_seconds": {
                "cpu_fp32": cpu_reference_seconds,
                "declared_fp16_path": declared_reference_seconds,
            },
        }
        throughput = len(y) / max(1e-12, timing["end_to_end_seconds"])
        result.update({
            "batch": batch,
            "evaluation_set": ("full_test_split" if primary_full
                               else "admission_subsample"),
            "full_test_split": bool(primary_full),
            "batch_schedule": schedule_summary(schedule),
            "timing": timing,
            "throughput_images_s": throughput,
            "throughput_basis": "end_to_end_seconds"
                                 " (preprocess+execution+postprocess,"
                                 " references excluded)",
            "stats": stats.to_dict(),
        })
        return result, logits, ref, declared
    finally:
        runner.close()


def _sidecar(out: Path, model: str, batch: int, kind: str) -> Path:
    return out.with_name(f"{out.stem}.{model}_b{batch}.{kind}.npy")


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
    ap.add_argument("--out", default=str(ROOT / "evidence" /
                                         "model_eval_mock_v2.json"))
    ap.add_argument("--profile-samples", type=int, default=512,
                    help="sample count for non-primary batch profiles")
    ap.add_argument("--global-limit", type=int, default=0,
                    help="limit the loaded split; 0 means the full 10000 test set")
    args = ap.parse_args()
    plans_dir = Path(args.plans_dir).resolve()
    manifest = json.loads((plans_dir / "manifest.json").read_text())
    x, y = load_test(Path(args.data_dir))
    if args.global_limit:
        x, y = x[:args.global_limit], y[:args.global_limit]
    full_requested = args.global_limit == 0
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
    report = {
        "schema_version": 2,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "backend": args.backend,
        "manifest_profile_hash": manifest["profile_hash"],
        "evaluation_set": ("full_test_split" if full_requested
                           else "admission_subsample"),
        "test_images": int(len(x)),
        "full_test_split": bool(full_requested),
        "cpu_thread_config": thread_config(),
        "models": [],
    }
    try:
        if args.models in {"mlp", "both"}:
            t_weight = time.perf_counter()
            w = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
            weight_load_seconds = time.perf_counter() - t_weight
            for batch in (32, 8, 1):
                primary = batch == 32
                xe = x if primary and full_requested else x[:args.profile_samples]
                ye = y if primary and full_requested else y[:args.profile_samples]
                result, logits, ref, declared = _run_mlp_profile(
                    lib, plans_dir, manifest, w, xe, ye, batch, fd,
                    weight_load_seconds=weight_load_seconds,
                    primary_full=(primary and full_requested))
                result["model"] = "mnist_mlp"
                report["models"].append(result)
                np.save(_sidecar(Path(args.out), "mlp", batch, "logits"), logits)
                np.save(_sidecar(Path(args.out), "mlp", batch, "ref"), ref)
                np.save(_sidecar(Path(args.out), "mlp", batch, "declared"), declared)
                print("MLP", batch, result["accuracy"],
                      result["top1_agreement"], flush=True)
        if args.models in {"vit", "both"}:
            t_weight = time.perf_counter()
            w = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
            weight_load_seconds = time.perf_counter() - t_weight
            for batch in (8, 1):
                primary = batch == 8
                xe = x if primary and full_requested else x[:args.profile_samples]
                ye = y if primary and full_requested else y[:args.profile_samples]
                result, logits, ref, declared = _run_vit_profile(
                    lib, plans_dir, manifest, w, xe, ye, batch, fd,
                    weight_load_seconds=weight_load_seconds,
                    primary_full=(primary and full_requested))
                result["model"] = "tiny_vit"
                report["models"].append(result)
                np.save(_sidecar(Path(args.out), "vit", batch, "logits"), logits)
                np.save(_sidecar(Path(args.out), "vit", batch, "ref"), ref)
                np.save(_sidecar(Path(args.out), "vit", batch, "declared"),
                        declared)
                print("ViT", batch, result["accuracy"],
                      result["top1_agreement"], flush=True)
    finally:
        if args.backend == "board" and args.fd is None and fd >= 0:
            lib.cdll.rocket_close(fd)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
