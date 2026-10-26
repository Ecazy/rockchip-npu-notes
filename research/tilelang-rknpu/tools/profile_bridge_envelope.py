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
PACK_BREAKDOWN_FIELDS = getattr(_models, "PACK_BREAKDOWN_FIELDS", ())
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_SCHEMA_VERSION = 1

def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "p10": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p50": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1,
                           int(0.9 * (len(ordered) - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


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


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _device_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_bytes().rstrip(b"\x00").decode(
            "utf-8", "replace")
    except OSError:
        return ""


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
    # The C-side total only exists when the library was built with probes and the
    # attributed entry was used.  Without it the residual is undefined, and
    # reporting a number anyway would be the exact mistake this package is
    # supposed to prevent.
    available = c_total > 0.0
    residual = (c_total - attributed) if available else 0.0
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "envelope_available": available,
        "plan_call_us": stats.plan_call_us,
        "plan_workspace_us": stats.plan_workspace_us,
        "plan_total_us": c_total,
        "plan_clock_us": c_clock,
        "plan_envelope_attributed_us": attributed,
        "plan_envelope_residual_us": residual,
        "plan_envelope_residual_fraction": (
            residual / c_total if available and c_total else 0.0),
        "plan_envelope_clock_fraction": (
            c_clock / c_total if available and c_total else 0.0),
        **{field: _stat(stats, f"plan_{field}")
           for field in ENVELOPE_FIELDS},
        **{field: _stat(stats, f"plan_{field}")
           for field in PACK_BREAKDOWN_FIELDS},
        "upload_coalesce": int(getattr(stats, "plan_upload_coalesce", 0)),
        "upload_fast_pack": int(getattr(stats, "plan_upload_fast_pack", 0)),
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
    key = getattr(args, "manifest_key", "") or spec["manifest"].format(
        batch=args.batch)
    manifest = args.plans_dir / (key + ".rkhybrid.json")
    t_manifest = time.perf_counter()
    program = load_hybrid_program(manifest)
    manifest_load_us = (time.perf_counter() - t_manifest) * 1e6
    weights = load_weight_npz(args.weights_dir / spec["weights"])
    images = load_images(args.data_dir, max(args.runs + args.warmup, 1) * args.batch)
    needed = (args.runs + args.warmup) * args.batch
    if images.shape[0] < needed:
        raise SystemExit(f"need {needed} images, only {images.shape[0]} available")

    fd = args.fd
    if args.backend == "board" and not fd:
        fd = lib.cdll.rocket_open()
        if fd < 0:
            raise SystemExit("rocket_open failed")

    entry = getattr(args, "plan_entry", "attributed")
    if entry == "legacy":
        # The legacy entry has no attribution, and a static-repeat node always
        # goes through the attributed group entry.  Mixing the two would produce
        # an envelope that neither configuration alone has, so it is refused
        # instead of being reported.
        repeated = [n.id for n in program.nodes
                    if getattr(n, "repeat", ()) and len(n.repeat) > 0]
        if repeated:
            raise SystemExit(
                "plan-entry=legacy cannot profile a graph with static-repeat "
                f"nodes ({', '.join(repeated)}); use the probe-free library "
                "build (rknpu_plan_noprobe) for the instrumentation control")
    # The entry selector only exists in the extended Python package; an older
    # arm simply gets the default (attributed) entry, which is what the campaign
    # compares anyway.  Only forward the argument when it is actually non-default
    # so a baseline package is never asked for a keyword it does not have.
    extra = {} if entry == "attributed" else {"plan_entry": entry}
    repeat_submit = getattr(args, "repeat_submit", "manifest")
    if repeat_submit != "manifest":
        extra["repeat_submit"] = repeat_submit
    t_setup = time.perf_counter()
    reuse_bindings = getattr(args, "reuse_bindings", "on") != "off"
    runner = HybridRunner(lib, program, weights, fd=fd,
                          reuse_bindings=reuse_bindings, **extra)
    runner_setup_us = (time.perf_counter() - t_setup) * 1e6
    plan_call_count = len(runner.calls)
    per_run = []
    host_samples: dict[str, dict[str, list[float]]] = {}
    plan_samples: dict[str, dict[str, list[float]]] = {}
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
            for row in stats.host_actions:
                bucket = host_samples.setdefault(row["name"], {})
                bucket.setdefault("elapsed_us", []).append(row["elapsed_us"])
                bucket.setdefault("host_bytes", []).append(row["host_bytes"])
            for row in stats.plans:
                bucket = plan_samples.setdefault(row["name"], {})
                for field in ("workspace_us", "validate_us", "pack_us",
                              "emit_us", "command_us", "submit_wait_us",
                              "output_sync_us", "download_us", "guard_us",
                              "publish_us", *PACK_BREAKDOWN_FIELDS):
                    if field in row:
                        bucket.setdefault(field, []).append(float(row[field]))
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
        "label": getattr(args, "label", ""),
        "plan_entry": getattr(args, "plan_entry", "attributed"),
        "repeat_submit": getattr(args, "repeat_submit", "manifest"),
        "reuse_bindings": reuse_bindings,
        "creation": {
            "manifest_load_us": manifest_load_us,
            "runner_setup_us": runner_setup_us,
            "plan_call_count": plan_call_count,
        },
        "host": {
            "machine": platform.machine(),
            # The affinity mask actually in force (taskset / sched_setaffinity),
            # so a reader can tell whether the arms shared a CPU set.
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "npu_freq": _read_text("/sys/kernel/debug/rknpu/freq"),
            "npu_load_before": _read_text("/sys/kernel/debug/rknpu/load"),
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
        "host_actions_pooled": {
            name: {field: _summary(values) for field, values in fields.items()}
            for name, fields in sorted(host_samples.items())
        },
        "plans_pooled": {
            name: {field: _summary(values) for field, values in fields.items()}
            for name, fields in sorted(plan_samples.items())
        },
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
    ap.add_argument("--manifest-key", default="",
                    help="explicit rkhybrid manifest stem; default is the "
                         "model/batch profile")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="",
                    help="free-form arm/round label recorded in the JSON")
    ap.add_argument("--reuse-bindings", choices=("on", "off"), default="on",
                    help="reuse PlanCall ctypes binding descriptors across runs")
    ap.add_argument("--repeat-submit",
                    choices=("manifest", "separate", "b1", "shared",
                             "chain_shared"),
                    default="manifest",
                    help="repeat path for static-repeat nodes: the manifest's "
                         "declared policy, or an explicit control")
    ap.add_argument("--plan-entry", choices=("attributed", "legacy"),
                    default="attributed",
                    help="attributed uses rk_workspace_run_ex; legacy uses the "
                         "frozen 88-byte rk_workspace_run report and is only "
                         "valid for graphs without static-repeat nodes")
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
    if not env["envelope_available"]:
        print("  attribution      unavailable (probes compiled out)")
    else:
        print(f"  C total          {env_sec(env['plan_total_us'])}")
        print(f"  attributed       {env_sec(env['plan_envelope_attributed_us'])}")
        print(f"  residual         {env_sec(env['plan_envelope_residual_us'])}"
              f"  ({env['plan_envelope_residual_fraction'] * 100:.1f}% of C total)")
        print(f"  instrumentation  {env_sec(env['plan_clock_us'])}"
              f"  ({env['plan_envelope_clock_fraction'] * 100:.1f}%)")
        base = env["plan_total_us"]
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
