#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure selected frozen-plan GEMM/attention templates on the board.

The timed application interval is ``PlanCall.run_into`` with the legacy
report-only C entry and a reused workspace.  Constant materialization and C
workspace creation are reported separately.  A second attributed pass records
the C envelope components; its clock probes are not part of the application
median.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import load_hybrid_program, _materialize_constant
from rknpu_backend.models import PlanCall, load_weight_npz
from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                      default_library)

CASES = (
    ("mlp_b1", "mnist_mlp_b1", "mlp", "mlp"),
    ("mlp_b8", "mnist_mlp_b8", "mlp", "mlp"),
    ("mlp_b32", "mnist_mlp_b32", "mlp", "mlp"),
    ("qkv_b1", "tiny_vit_b1", "qkv", "gemm"),
    ("qkv_b8", "tiny_vit_b8", "qkv", "gemm"),
    ("gate_up_b1", "tiny_vit_b1", "gate_up", "gemm"),
    ("gate_up_b8", "tiny_vit_b8", "gate_up", "gemm"),
    ("down_b1", "tiny_vit_b1", "down", "gemm"),
    ("down_b8", "tiny_vit_b8", "down", "gemm"),
    ("classifier_b1", "tiny_vit_b1", "classifier", "gemm"),
    ("classifier_b8", "tiny_vit_b8", "classifier", "gemm"),
    ("qk_b1", "tiny_vit_b1", "attn_qk_qkv_group", "group"),
    ("qk_b8", "tiny_vit_b8", "attn_qk_qkv_group", "group"),
    ("pv_b1", "tiny_vit_b1", "attn_pv_qkv_group", "group"),
    ("pv_b8", "tiny_vit_b8", "attn_pv_qkv_group", "group"),
)
ENVELOPE = ("validate_us", "pack_us", "emit_us", "command_us",
            "submit_alloc_us", "submit_wait_us", "output_sync_us",
            "download_us", "guard_us", "publish_us", "alloc_us",
            "total_us", "clock_us", "pack_prep_us", "pack_clear_us",
            "pack_layout_us", "pack_fini_us")


def _stats(values):
    return {"median": float(statistics.median(values)),
            "mean": float(statistics.fmean(values)),
            "p10": float(statistics.quantiles(values, n=10)[0]),
            "min": float(min(values)), "max": float(max(values))}


def _materialize(program, plan, plan_key, weights):
    if not plan.constants:
        return {}
    node = next((n for n in program.nodes
                 if getattr(n, "plan", None) == plan_key), None)
    if node is None:
        raise RuntimeError(f"constant-carrying plan {plan_key!r} has no node")
    lookup = {spec.name: spec for spec in plan.constants}
    return {name: _materialize_constant(desc, lookup[name], weights,
                                        program.weights)
            for name, desc in node.constants.items()}


def _inputs(plan, rng):
    inputs, outputs = [], []
    for b in plan.bindings:
        if b.role == "input":
            inputs.append((rng.standard_normal(b.shape) * 0.1).astype(
                np.float16))
        elif b.role == "output":
            outputs.append(np.empty(b.shape, dtype=np.float16))
    return inputs, outputs


def _run_case(args, lib, fd, manifest_key, plan_key, kind, weights, rng):
    plans_dir = args.plans_dir.resolve()
    program = load_hybrid_program(plans_dir / f"{manifest_key}.rkhybrid.json",
                                  plans_dir=plans_dir)
    plan = program.plans[plan_key].plan
    t0 = time.perf_counter()
    constants = _materialize(program, plan, plan_key, weights)
    constant_materialize_us = (time.perf_counter() - t0) * 1e6
    inputs, outputs = _inputs(plan, rng)
    # Setup is measured separately from the steady-state call and includes C
    # plan creation, BO allocation, and constant upload/packing.
    setup_values = []
    for _ in range(3):
        t0 = time.perf_counter()
        call = PlanCall(lib, plan, constants, fd=fd, entry="legacy")
        setup_values.append((time.perf_counter() - t0) * 1e6)
        call.close()
    # Production selected paths prebuild their binding descriptors once.  The
    # single-plan path uses PlanCall.run_prepared; the expanded QK/PV plan uses
    # the group entry, matching HybridRunner's shared-group lowering.
    t0 = time.perf_counter()
    app = PlanCall(lib, plan, constants, fd=fd, entry="attributed")
    app_setup_us = (time.perf_counter() - t0) * 1e6
    t0 = time.perf_counter()
    if kind == "group":
        app.bind_group_prepared_into([inputs], [outputs],
                                     binding_key="template-bench")
    else:
        app.bind_prepared_into(inputs, outputs, binding_key="template-bench")
    bind_prepared_us = (time.perf_counter() - t0) * 1e6
    def invoke(call):
        if kind == "group":
            return call.run_group_prepared(binding_key="template-bench")
        return call.run_prepared(binding_key="template-bench")
    try:
        for _ in range(args.warmup):
            invoke(app)
        app_us = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            invoke(app)
            app_us.append((time.perf_counter() - t0) * 1e6)
        # Fresh row-major output allocation is measured through the generic
        # (non-prepared) entry and is reported separately; it is not the
        # production C envelope.
        fresh = PlanCall(lib, plan, constants, fd=fd, entry="legacy")
        fresh_us = []
        fresh_runs = min(args.runs, 100)
        try:
            for _ in range(fresh_runs):
                fresh_outputs = [np.empty_like(x) for x in outputs]
                t0 = time.perf_counter()
                if kind == "group":
                    fresh.run_group_into([inputs], [fresh_outputs])
                else:
                    fresh.run_into(inputs, fresh_outputs)
                fresh_us.append((time.perf_counter() - t0) * 1e6)
        finally:
            fresh.close()
        # CPU reference is only meaningful for the simple one-input/one-constant
        # GEMM plans; it uses FP32 and is labelled a different numeric policy.
        cpu = None
        if kind == "gemm" and len(inputs) == 1 and len(outputs) == 1:
            bname = next(iter(constants))
            a32 = inputs[0].astype(np.float32)
            b32 = constants[bname].astype(np.float32)
            for _ in range(5):
                ref = a32 @ b32
            cpu_times = []
            for _ in range(20):
                t0 = time.perf_counter()
                ref = a32 @ b32
                cpu_times.append((time.perf_counter() - t0) * 1e3)
            delta = np.abs(outputs[0].astype(np.float32) - ref)
            cpu = {"median_ms": float(statistics.median(cpu_times)),
                   "min_ms": float(min(cpu_times)),
                   "max_ms": float(max(cpu_times)),
                   "max_abs_error": float(delta.max()),
                   "mean_abs_error": float(delta.mean()),
                   "tolerance_applied": False}
        # Attribution comes from the same prepared-path call; its clock probes
        # are counted explicitly and can be subtracted for kernel/envelope
        # interpretation.
        sums = {name: 0.0 for name in ENVELOPE}
        calls = max(1, min(args.runs, 100))
        upload_calls = download_calls = upload_bytes = download_bytes = 0
        task_count = submit_count = 0
        for _ in range(calls):
            report, _ = invoke(app)
            for name in ENVELOPE:
                sums[name] += float(getattr(report, name, 0.0))
            upload_calls += int(report.upload_calls)
            download_calls += int(report.download_calls)
            upload_bytes += int(report.upload_bytes)
            download_bytes += int(report.download_bytes)
            task_count += int(report.task_count)
            submit_count += int(report.submit_count)
        attributed = {
            "calls": calls,
            "per_call": {name: sums[name] / calls for name in ENVELOPE},
            "upload_calls": upload_calls / calls,
            "download_calls": download_calls / calls,
            "upload_bytes": upload_bytes / calls,
            "download_bytes": download_bytes / calls,
            "task_count": task_count / calls,
            "submit_count": submit_count / calls,
        }
    finally:
        app.close()
    setup_us = float(statistics.median(setup_values))
    steady = _stats(app_us)
    return {
        "case": f"{plan_key}_{manifest_key}",
        "kind": kind,
        "manifest": manifest_key,
        "plan": plan_key,
        "plan_hash": plan.plan_hash,
        "bindings": [{"name": b.name, "role": b.role,
                      "shape": list(b.shape), "nbytes": b.nbytes}
                     for b in plan.bindings],
        "constant_bytes": sum(x.nbytes for x in constants.values()),
        "constant_materialize_us": constant_materialize_us,
        "setup_us": setup_us,
        "setup_samples_us": setup_values,
        "workspace_setup_us": app_setup_us,
        "bind_prepared_us": bind_prepared_us,
        "application_reuse_us": steady,
        "application_fresh_output_us": _stats(fresh_us),
        "cold_once_us": (constant_materialize_us + app_setup_us +
                         bind_prepared_us + steady["median"]),
        "amortized_1000_us": (constant_materialize_us + app_setup_us +
                              bind_prepared_us) / 1000.0 + steady["median"],
        "output_sha256": hashlib.sha256(outputs[0].tobytes()).hexdigest(),
        "cpu_fp32_reference": cpu,
        "attributed": attributed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--lib", default="")
    ap.add_argument("--backend", choices=("mock", "board"), default="board")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reverse", action="store_true",
                    help="run the frozen case list in reverse order as an "
                         "order/thermal-drift control")
    args = ap.parse_args()
    if args.backend == "mock":
        lib = LibRKPlan(args.lib or str(
            Path(__file__).resolve().parents[1].parent /
            "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"))
        fd = 0
    else:
        lib = default_library() if not args.lib else LibRKPlan(args.lib)
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise RuntimeError(f"rocket_open failed: {fd}")
    weights_by_model = {
        "mnist_mlp": load_weight_npz(args.weights_dir / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(args.weights_dir / "tiny_vit.npz"),
    }
    result = {
        "schema_version": 1,
        "kind": "gemm-template-benchmark",
        "backend": args.backend,
        "runs": args.runs,
        "warmup": args.warmup,
        "application_interval": "PlanCall.run_into with entry=legacy and "
                                "reused workspace/output",
        "instrumentation": "separate attributed pass; excluded from app median",
        "cpu_reference": "NumPy FP32; different numeric policy, not ranked",
        "environment": {
            "machine": platform.machine(),
            "kernel": platform.release(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_config": {name: os.environ.get(name)
                              for name in ("OMP_NUM_THREADS",
                                           "OPENBLAS_NUM_THREADS",
                                           "MKL_NUM_THREADS")},
        },
        "cases": [],
    }
    try:
        rng = np.random.default_rng(20261027)
        cases = tuple(reversed(CASES)) if args.reverse else CASES
        for _label, manifest, plan_key, kind in cases:
            model = "mnist_mlp" if manifest.startswith("mnist_mlp") else "tiny_vit"
            result["cases"].append(_run_case(
                args, lib, fd, manifest, plan_key, kind,
                weights_by_model[model], rng))
    finally:
        if args.backend == "board":
            lib.cdll.rocket_close(fd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "out": str(args.out),
                      "cases": len(result["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
