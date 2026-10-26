#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Attribute the RKPlan/Host envelope of a frozen hybrid model profile.

This is the measurement instrument for the first optimisation package in
``TRANSFORMER-OPTIMIZATION-PLAN.md``:

* it runs one frozen ``rkhybrid`` profile (default: Tiny Transformer batch 8)
  over a fixed, deterministic input set;
* it reports the model-level envelope (preprocess/execution/postprocess,
  RKPlan dwell, C workspace dwell, Host actions, executor overhead);
* it reports the C-side workspace attribution (validate / pack / emit / command
  / submit / output-sync / download / guard / publish / alloc) and the
  *residual*, i.e. the part of the workspace dwell that is still unexplained;
* it counts tasks, submits, uploads and downloads so a candidate can be judged
  on structural changes, not only on wall time.

Mock and board runs share the same code path; only the library differs.  The
output is JSON so a campaign can diff two builds field by field.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import statistics
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hostinfo import thread_config
from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
from rknpu_backend import models as _models
from rknpu_backend.models import RunStats, load_weight_npz

# A baseline library/Python pair predates the envelope tail; the profiler must be
# able to measure it, so an absent attribution field reads as zero.
ENVELOPE_FIELDS = getattr(_models, "ENVELOPE_FIELDS", ())
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_SCHEMA_VERSION = 1

MODEL_SPECS = {
    "tiny_vit": {"weights": "tiny_vit.npz", "manifest": "tiny_vit_b{batch}"},
    "mnist_mlp": {"weights": "mnist_mlp.npz", "manifest": "mnist_mlp_b{batch}"},
}


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_images(data_dir: Path, count: int) -> np.ndarray:
    x = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    return x[:count]


def _device_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_bytes().rstrip(b"\x00").decode(
            "utf-8", "replace")
    except OSError:
        return ""


def _ms(us: float) -> float:
    return float(us) / 1e6


def _stat(stats: RunStats, name: str) -> float:
    return float(getattr(stats, name, 0.0))


def envelope_row(stats: RunStats) -> dict:
    """Summarise one run's RKPlan envelope.

    ``plan_workspace_us`` is the caller-observed dwell around each
    ``rk_workspace_run`` call; ``plan_total_us`` is the same interval measured
    inside C.  The residual is computed against the C-side total when the
    library reports one, because the attribution intervals are measured there
    too and only that pairing is self-consistent.
    """
    attributed = _stat(stats, "plan_envelope_attributed_us")
    c_total = _stat(stats, "plan_total_us")
    c_clock = _stat(stats, "plan_clock_us")
    base = c_total if c_total > 0.0 else float(stats.plan_workspace_us)
    residual = base - attributed
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "plan_call_us": stats.plan_call_us,
        "plan_workspace_us": stats.plan_workspace_us,
        "plan_total_us": c_total,
        "plan_clock_us": c_clock,
        "plan_envelope_attributed_us": attributed,
        "plan_envelope_residual_us": residual,
        "plan_envelope_residual_fraction": (
            residual / base if base else 0.0),
        "plan_envelope_clock_fraction": (c_clock / base if base else 0.0),
        **{field: _stat(stats, f"plan_{field}")
           for field in ENVELOPE_FIELDS},
        "python_plan_node_us": max(
            0.0, float(stats.plan_call_us) - float(stats.plan_workspace_us)),
        "host_action_us": stats.host_action_us,
        "executor_overhead_us": stats.executor_overhead_us,
        "task_count": stats.task_count,
        "submit_count": stats.submit_count,
        "upload_calls": stats.upload_calls,
        "upload_bytes": stats.upload_bytes,
        "download_calls": stats.download_calls,
        "download_bytes": stats.download_bytes,
    }


def resolve_library(args) -> str:
    """Pick the library explicitly: the mock profile never falls back to a device
    build through ``RKNPU_PLAN_LIB``, so a mock run cannot silently acquire the
    production target check."""
    if args.lib:
        return str(args.lib)
    build = ROOT.parent / "custom-ops" / "build-host-mvp"
    if args.backend == "mock":
        return str(build / "librknpu_plan_mock.so")
    return os.environ.get("RKNPU_PLAN_LIB") or str(build / "librknpu_plan.so")


def run_profile(args) -> dict:
    lib_path = resolve_library(args)
    lib = LibRKPlan(lib_path)
    configure_rocket_abi(lib.cdll)

    spec = MODEL_SPECS[args.model]
    manifest = (args.plans_dir /
                (spec["manifest"].format(batch=args.batch) + ".rkhybrid.json"))
    program = load_hybrid_program(manifest)
    weights = load_weight_npz(args.weights_dir / spec["weights"])
    images = load_images(args.data_dir, max(args.runs + args.warmup, 1) * args.batch)
    needed = (args.runs + args.warmup) * args.batch
    if images.shape[0] < needed:
        raise SystemExit(f"need {needed} images, only {images.shape[0]} available")

    fd = args.fd
    if args.backend == "board" and not fd:
        import ctypes
        fd = lib.cdll.rocket_open()
        if fd < 0:
            raise SystemExit("rocket_open failed")

    runner = HybridRunner(lib, program, weights, fd=fd)
    per_run = []
    try:
        for i in range(args.warmup):
            start = i * args.batch
            runner.run(images[start:start + args.batch])
        for i in range(args.runs):
            start = (args.warmup + i) * args.batch
            stats = RunStats()
            t0 = time.perf_counter()
            runner.run(images[start:start + args.batch], stats=stats)
            wall_us = (time.perf_counter() - t0) * 1e6
            per_run.append((stats, wall_us))
    finally:
        runner.close()
        if args.backend == "board" and not args.fd:
            lib.cdll.rocket_close(fd)

    walls = [w for _, w in per_run]
    median_wall = statistics.median(walls)
    # One representative run at the median keeps the per-plan rows readable.
    representative = per_run[min(range(len(walls)),
                                 key=lambda i: abs(walls[i] - median_wall))][0]
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "kind": "bridge-envelope-profile",
        "backend": args.backend,
        "library": lib_path,
        "model": args.model,
        "batch": args.batch,
        "runs": args.runs,
        "warmup": args.warmup,
        "host": {
            "machine": platform.machine(),
            # Recorded from the device-tree model, not the hostname: the
            # hostname is not evidence and can collide with credential text.
            "device_model": _device_model(),
            "kernel": platform.release(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "thread_config": thread_config(),
        },
        "wall_us": {
            "median": median_wall,
            "min": min(walls),
            "max": max(walls),
            "mean": float(statistics.fmean(walls)),
            "samples": walls,
        },
        "wall_images_per_s": (
            float(args.batch) / (median_wall / 1e6) if median_wall else 0.0),
        "envelope": envelope_row(representative),
        "plans": representative.plans,
        "host_actions": representative.host_actions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=tuple(MODEL_SPECS), default="tiny_vit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--lib", default="")
    ap.add_argument("--fd", type=int, default=0,
                    help="already-open device fd (board); 0 opens its own")
    ap.add_argument("--plans-dir", type=Path,
                    default=ROOT / "models" / "compiled")
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--data-dir", type=Path, default=ROOT / ".data" / "mnist")
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")

    result = run_profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    env = result["envelope"]
    print(f"model={result['model']} batch={result['batch']} "
          f"backend={result['backend']} runs={result['runs']}")
    print(f"  median wall      {env_sec(result['wall_us']['median'])}"
          f"  {result['wall_images_per_s']:.1f} img/s")
    print(f"  python dwell     {env_sec(env['plan_workspace_us'])}")
    print(f"  C total          {env_sec(env['plan_total_us'])}")
    print(f"  attributed       {env_sec(env['plan_envelope_attributed_us'])}")
    print(f"  residual         {env_sec(env['plan_envelope_residual_us'])}"
          f"  ({env['plan_envelope_residual_fraction'] * 100:.1f}% of C total)")
    print(f"  instrumentation  {env_sec(env['plan_clock_us'])}"
          f"  ({env['plan_envelope_clock_fraction'] * 100:.1f}%)")
    base = env["plan_total_us"] or env["plan_workspace_us"]
    for field in ENVELOPE_FIELDS:
        share = (env[field] / base * 100.0) if base else 0.0
        print(f"    {field:<16} {env_sec(env[field])}  {share:5.1f}%")
    print(f"  tasks/submits    {env['task_count']} / {env['submit_count']}")
    if args.out:
        print(f"wrote {args.out}")
    return 0


def env_sec(us: float) -> str:
    return f"{float(us) / 1e6:9.6f} s"


if __name__ == "__main__":
    raise SystemExit(main())
