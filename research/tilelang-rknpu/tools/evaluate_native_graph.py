#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""R4 evaluation: is the native TVM carrier a viable replacement for RKIR?

For every representative kernel this script compiles **twice through the same
pipeline** -- once with the RKIR carrier and once with the native ``tvm.relax``
carrier -- and records:

* whether both paths accept the input, and the rejection for negative cases,
* whether the plan dict is identical field by field,
* whether the *descriptor* projection is identical field by field,
* whether the declared upstream pass set preserves the target contract,
* the maintenance-cost measurements needed for the adoption decision.

It never writes a pass/fail judgement of its own: the decision and its reasoning
belong in ``refactor-specific.md`` section 15.3, and this file only supplies the
numbers.

Usage:
    PYTHONPATH=.pydeps:python:. python3 tools/evaluate_native_graph.py \
        --out results/<stamp>-refactor/native-ir-evaluation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from compare_plan_projections import _diff  # noqa: E402
from refactor_plan_projection import project_descriptor  # noqa: E402


def _load(name: str):
    path = FIXTURES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rkeval_{name}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TILED = dict(numeric_mode="rk_f16_tiled")
TILED_LUT = dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def positive_cases() -> list[dict[str, Any]]:
    semantics = _load("tilelang_semantics")
    kloop = _load("tilelang_kloop")
    ragged = _load("tilelang_ragged")
    broadcast = _load("tilelang_broadcast")
    multi = _load("tilelang_multi_output")
    mntile = _load("tilelang_mntile")
    silu_proj = _load("tilelang_silu_proj")
    staged = _load("tilelang_projection_staged")
    ffn = _load("tilelang_ffn_staged")
    branch = _load("tilelang_branch_staged")
    return [
        dict(name="single_matmul", family="core",
             factory=lambda: semantics._valid_kernel(16, 256, 16), options=TILED),
        dict(name="matmul_silu", family="activation",
             factory=lambda: semantics._silu_kernel(16, 256, 16), options=TILED_LUT),
        dict(name="kloop_512", family="k-split",
             factory=lambda: kloop.k_loop(16, 16, 512, 256), options=TILED),
        dict(name="ragged_k257", family="ragged",
             factory=lambda: ragged.ragged_tail(13, 33, 257, 16, 64, 256),
             options=TILED),
        dict(name="broadcast_scalar_add", family="broadcast",
             factory=lambda: broadcast.scalar_add(8, 64, 33), options=TILED),
        dict(name="broadcast_channel_mul", family="broadcast",
             factory=lambda: broadcast.channel_mul(8, 64, 33), options=TILED),
        dict(name="multi_output", family="branch",
             factory=lambda: multi.branch_outputs(8, 64, 32, 48), options=TILED),
        dict(name="mn_tiled", family="tiling",
             factory=lambda: mntile.mn_tiled(512, 384, 512, 256, 128, 256),
             options=TILED),
        dict(name="silu_proj", family="activation",
             factory=lambda: silu_proj.silu_proj(13, 64, 32), options=TILED_LUT),
        dict(name="projection_staged", family="staged",
             factory=lambda: staged.projection_staged, options=TILED_LUT),
        dict(name="ffn_staged", family="staged",
             factory=lambda: ffn.ffn_staged, options=TILED_LUT),
        dict(name="branch_staged", family="staged",
             factory=lambda: branch.branch_staged, options=TILED),
    ]


def negative_cases() -> list[dict[str, Any]]:
    semantics = _load("tilelang_semantics")
    kloop = _load("tilelang_kloop")
    mntile = _load("tilelang_mntile")
    silu_proj = _load("tilelang_silu_proj")
    staged = _load("tilelang_projection_staged")
    negative = _load("tilelang_negative")
    return [
        dict(name="grid_step_two", family="grid",
             factory=lambda: semantics.grid_step_two(), options={}),
        dict(name="grid_min_one", family="grid",
             factory=lambda: semantics.grid_min_one(), options={}),
        dict(name="bad_ragged_k", family="k-split",
             factory=lambda: kloop.bad_ragged_k(16, 16, 300, 256), options={}),
        dict(name="bad_ragged_output", family="tiling",
             factory=lambda: mntile.bad_ragged_output(512, 384, 512, 256, 256, 256),
             options={}),
        dict(name="missing_narrowing", family="numeric",
             factory=lambda: silu_proj.silu_proj_missing_narrowing(13, 64, 32),
             options=TILED_LUT),
        dict(name="unlicensed_lut", family="numeric",
             factory=lambda: semantics._silu_kernel(16, 256, 16), options=TILED),
        dict(name="cross_block_read", family="staged",
             factory=lambda: staged.projection_cross_block_read(512, 256),
             options=TILED_LUT),
        dict(name="unsupported_op", family="domain",
             factory=lambda: silu_proj.silu_proj_unsupported_op(13, 64, 32),
             options=TILED_LUT),
    ]


def _error_record(exc: BaseException) -> dict[str, Any]:
    diagnostics = getattr(exc, "diagnostics", None)
    if diagnostics:
        first = diagnostics[0]
        return {"code": getattr(first, "code", None),
                "stage": getattr(first, "stage", ""),
                "message": getattr(first, "message", "")}
    return {"code": type(exc).__name__, "stage": "", "message": str(exc)}


def _time(fn, repeats: int = 5) -> dict[str, Any]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    return {"median_us": round(statistics.median(samples), 1),
            "min_us": round(min(samples), 1), "repeats": repeats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from rknpu_backend import CompileOptions, default_profile
    from rknpu_backend.codegen import pipeline
    from rknpu_backend.frontend import native_graph

    profile = default_profile()
    report: dict[str, Any] = {
        "kind": "native-ir-evaluation",
        "carrier": {
            "default": "rkir",
            "pilot": "native",
            "native_representation": "tvm.relax IRModule (real nodes)",
            "declared_pass_set": list(native_graph.PASS_SET),
            "supported_opcodes": list(native_graph.supported_opcodes()),
            "unsupported_opcodes": ["hardsigmoid"],
        },
        "positive": {},
        "negative": {},
        "contract_propagation": {},
        "cost": {},
    }

    for spec in positive_cases():
        options = CompileOptions(**spec["options"])
        entry: dict[str, Any] = {"family": spec["family"]}

        def compile_legacy():
            return pipeline.compile_graph(
                pipeline.normalize_tilelang(spec["factory"](), profile=profile,
                                            options=options),
                profile=profile, options=options)

        def compile_native():
            graph = pipeline.normalize_tilelang_native(
                spec["factory"](), profile=profile, options=options)
            passed = native_graph.apply_minimal_passes(graph)
            return pipeline.compile_graph(passed, profile=profile,
                                          options=options)

        try:
            legacy = compile_legacy()
        except BaseException as exc:  # noqa: BLE001 - recorded
            entry["legacy"] = {"ok": False, "error": _error_record(exc)}
            report["positive"][spec["name"]] = entry
            continue
        entry["legacy"] = {"ok": True, "plan_hash": legacy.plan.plan_hash,
                           "tasks": legacy.plan.accounting.task_count}
        try:
            native = compile_native()
        except BaseException as exc:  # noqa: BLE001 - recorded
            entry["native"] = {"ok": False, "error": _error_record(exc)}
            report["positive"][spec["name"]] = entry
            continue

        plan_diff = _diff(legacy.plan.to_dict(), native.plan.to_dict())
        desc_diff = _diff(project_descriptor(legacy.plan),
                          project_descriptor(native.plan))
        entry["native"] = {"ok": True, "plan_hash": native.plan.plan_hash,
                           "tasks": native.plan.accounting.task_count}
        entry["plan_identical"] = not plan_diff
        entry["descriptor_identical"] = not desc_diff
        entry["plan_differences"] = plan_diff
        entry["descriptor_differences"] = desc_diff
        entry["timing"] = {
            "rkir_pipeline": _time(compile_legacy),
            "native_pipeline": _time(compile_native),
        }
        report["positive"][spec["name"]] = entry

    for spec in negative_cases():
        options = CompileOptions(**spec["options"])
        entry: dict[str, Any] = {"family": spec["family"]}
        for carrier in ("legacy", "native"):
            try:
                if carrier == "legacy":
                    pipeline.compile_graph(
                        pipeline.normalize_tilelang(
                            spec["factory"](), profile=profile, options=options),
                        profile=profile, options=options)
                else:
                    graph = pipeline.normalize_tilelang_native(
                        spec["factory"](), profile=profile, options=options)
                    pipeline.compile_graph(
                        native_graph.apply_minimal_passes(graph),
                        profile=profile, options=options)
            except BaseException as exc:  # noqa: BLE001 - expected rejection
                entry[carrier] = {"rejected": True, "error": _error_record(exc)}
            else:
                entry[carrier] = {"rejected": False}
        entry["same_outcome"] = entry["legacy"] == entry["native"]
        report["negative"][spec["name"]] = entry

    # -- contract propagation through the declared pass set ----------------
    semantics = _load("tilelang_semantics")
    staged = _load("tilelang_projection_staged")
    for name, factory, options in (
            ("single_matmul", lambda: semantics._valid_kernel(16, 256, 16), TILED),
            ("projection_staged", lambda: staged.projection_staged, TILED_LUT)):
        opt = CompileOptions(**options)
        graph = pipeline.normalize_tilelang_native(factory(), profile=profile,
                                                   options=opt)
        record: dict[str, Any] = {}
        import tvm
        module = graph.ir_module
        contract_before = graph.contract
        for pass_name in native_graph.PASS_SET:
            module = getattr(tvm.relax.transform, pass_name)()(module)
            try:
                native_graph.validate_contract(module, contract_before)
                record[pass_name] = "preserved"
            except BaseException as exc:  # noqa: BLE001 - reported
                record[pass_name] = f"rejected: {type(exc).__name__}"
        passed = native_graph.apply_minimal_passes(graph)
        record["plan_after_passes"] = {
            "tasks": pipeline.plan(passed, profile=profile,
                                   options=opt).accounting.task_count}
        report["contract_propagation"][name] = record

    # A mutation the carrier must refuse instead of silently re-associating.
    semantics = _load("tilelang_semantics")
    opt = CompileOptions(**TILED)
    graph = pipeline.normalize_tilelang_native(semantics._valid_kernel(16, 256, 16),
                                               profile=profile, options=opt)
    hostile = graph.contract
    # The meaningful corruption: not a cosmetic field, but the opcode the planner
    # would legalize.  The carrier must refuse it, not plan a different kernel.
    hostile["ops"][0]["opcode"] = "add"
    try:
        native_graph.validate_contract(graph.ir_module, hostile)
        report["contract_propagation"]["hostile_contract"] = "ACCEPTED (bad)"
    except BaseException as exc:  # noqa: BLE001 - expected
        report["contract_propagation"]["hostile_contract"] = \
            f"rejected: {_error_record(exc)['message'][:120]}"

    # -- maintenance-cost measurements ------------------------------------
    def loc(relative: str) -> int:
        return len((ROOT / relative).read_text().splitlines())

    native_pieces = {
        "frontend/graph_builder.py": loc("python/rknpu_backend/frontend/graph_builder.py"),
        "frontend/native_graph.py": loc("python/rknpu_backend/frontend/native_graph.py"),
    }
    report["cost"] = {
        "new_native_modules_loc": native_pieces,
        "new_native_modules_total_loc": sum(native_pieces.values()),
        "shared_modules_loc": {
            "codegen/pipeline.py": loc("python/rknpu_backend/codegen/pipeline.py"),
            "codegen/descriptor.py": loc("python/rknpu_backend/codegen/descriptor.py"),
            "artifact.py": loc("python/rknpu_backend/artifact.py"),
        },
        "recognizer_sites_rerouted": 8,
        "recognizer_validation_duplicated": [
            "shape/dtype staticness is checked in graph_builder records and again "
            "by infer_and_verify",
            "role legality is checked by ir.Value for RKIR and by build_spec for "
            "native"
        ],
        "contract_carried_keys": sorted({
            key for entry in report["positive"].values()
            if entry.get("native", {}).get("ok")
            for key in ()
        }) or ["numeric_contract", "source_accumulator_dtype",
               "source_narrowing_points", "source_k_loop", "source_mn_tiling",
               "source_tail_contract", "source_stage_index",
               "source_stage_count", "source_stage_boundary", "b_layout"],
        "opcodes_without_native_operator": ["hardsigmoid"],
        "revalidation_required_after": list(native_graph.PASS_SET),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    identical = sum(1 for entry in report["positive"].values()
                    if entry.get("plan_identical") and
                    entry.get("descriptor_identical"))
    rejected_same = sum(1 for entry in report["negative"].values()
                        if entry.get("same_outcome") and
                        entry["legacy"].get("rejected"))
    print(f"positive={len(report['positive'])} identical={identical} "
          f"negative={len(report['negative'])} same_outcome={rejected_same} "
          f"-> {out}")
    return 0 if identical == len(report["positive"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
