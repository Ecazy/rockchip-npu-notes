#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projection oracle for the codegen refactor: plan JSON + descriptor contents.

This script is deliberately written against API spellings that exist both before
and after the refactor (``compile_rkir``/``compile_tilelang`` and
``rknpu_backend.runtime.ffi.build_plan_descriptor``), so the *same file* can be run
against a baseline worktree and the working tree.  Comparing the two JSON outputs
is the plan/descriptor identity check required by the refactor spec.

The descriptor projection decodes every field and every live array element and
skips nothing that carries execution meaning.  Host pointer *addresses* and
padding bytes are not copied because they are not semantics; the field values,
counts, capacities, shapes and orderings are.

Usage:
    PYTHONPATH=<tree>/python python3 tools/refactor_plan_projection.py --out out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    path = FIXTURES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rkfix_{name}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- projections

def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", "replace")
    return "" if value is None else str(value)


def _ints(pointer, count: int) -> list[int]:
    if not count:
        return []
    return [int(pointer[i]) for i in range(int(count))]


def project_descriptor(plan) -> dict[str, Any]:
    """Decode one ``RKPlanDesc`` into pure JSON, field by field."""
    from rknpu_backend.runtime.ffi import build_plan_descriptor

    desc, _holder = build_plan_descriptor(plan)
    values = []
    for i in range(int(desc.nvalues)):
        item = desc.values[i]
        rank = int(item.rank)
        values.append({
            "abi_version": int(item.abi_version),
            "struct_size": int(item.struct_size),
            "id": int(item.id), "role": int(item.role), "dtype": int(item.dtype),
            "rank": rank, "shape": [int(item.shape[k]) for k in range(rank)],
            "nbytes": int(item.nbytes), "name": _text(item.name),
            "name_len": int(item.name_len), "reserved": int(item.reserved),
        })
    buffers = []
    for i in range(int(desc.nbuffers)):
        item = desc.buffers[i]
        buffers.append({
            "id": int(item.id), "role": int(item.role), "slot_id": int(item.slot_id),
            "alignment": int(item.alignment), "layout_id": int(item.layout_id),
            "padding_state": int(item.padding_state),
            "lifetime": [int(item.lifetime_first), int(item.lifetime_last)],
            "offset": int(item.offset), "capacity": int(item.capacity),
        })
    views = []
    for i in range(int(desc.nviews)):
        item = desc.views[i]
        tensor = item.tensor
        views.append({
            "id": int(item.id), "buffer_id": int(item.buffer_id),
            "offset_bytes": int(item.offset_bytes),
            "valid_region": _ints(item.valid_region, 4),
            "tensor": {
                "dtype": int(tensor.dtype), "layout": int(tensor.layout),
                "rows": int(tensor.rows), "cols": int(tensor.cols),
                "physical_rows": int(tensor.physical_rows),
                "physical_cols": int(tensor.physical_cols),
                "atom": int(tensor.atom), "bytes": int(tensor.bytes),
            },
        })
    actions = []
    for i in range(int(desc.nactions)):
        item = desc.actions[i]
        actions.append({
            "id": int(item.id), "kind": int(item.kind),
            "template_id": int(item.template_id),
            "inputs": _ints(item.inputs, int(item.ninputs)),
            "outputs": _ints(item.outputs, int(item.noutputs)),
            "deps": _ints(item.deps, int(item.ndeps)),
            "source_binding": int(item.source_binding),
            "source_row_offset": int(item.source_row_offset),
            "source_col_offset": int(item.source_col_offset),
            "source_row_stride": int(item.source_row_stride),
            "flags": int(item.flags), "act_kind": int(item.act_kind),
            "origin": _text(item.origin),
        })
    submits = []
    for i in range(int(desc.nsubmits)):
        item = desc.submits[i]
        submits.append({
            "id": int(item.id), "mode": int(item.mode),
            "core_policy": int(item.core_policy),
            "task_ids": _ints(item.task_ids, int(item.ntask_ids)),
            "deps": _ints(item.deps, int(item.ndeps)),
        })
    publish = []
    for i in range(int(desc.npublish)):
        item = desc.publish[i]
        publish.append({
            "binding_id": int(item.binding_id), "view_id": int(item.view_id),
            "dst_row_offset": int(item.dst_row_offset),
            "dst_col_offset": int(item.dst_col_offset),
            "rows": int(item.rows), "cols": int(item.cols),
        })
    return {
        "header": {
            "abi_version": int(desc.abi_version),
            "struct_size": int(desc.struct_size),
            "schema_version": int(desc.schema_version),
            "flags": int(desc.flags),
            "profile_hash": int(desc.profile_hash),
            "source_hash": int(desc.source_hash),
            "runtime_abi_version": int(desc.runtime_abi_version),
        },
        "counts": {
            "values": int(desc.nvalues), "buffers": int(desc.nbuffers),
            "views": int(desc.nviews), "actions": int(desc.nactions),
            "submits": int(desc.nsubmits), "publish": int(desc.npublish),
        },
        "budgets": {
            "device_budget_bytes": int(desc.device_budget_bytes),
            "host_budget_bytes": int(desc.host_budget_bytes),
            "claimed_device_bytes": int(desc.claimed_device_bytes),
            "claimed_host_bytes": int(desc.claimed_host_bytes),
        },
        "values": values, "buffers": buffers, "views": views,
        "actions": actions, "submits": submits, "publish": publish,
    }


# ``str(Span)`` embeds the address of the TVM ``SourceName`` object, so two
# processes compiling the same PrimFunc produce different ``source_hash`` strings
# (this is true of the pre-refactor tree as well; it is recorded as an identity
# observation, not masked away).  The pointer is the only volatile part, so the
# projection normalises exactly that substring and recomputes the canonical plan
# hash.  Everything else -- bindings, layout, actions, deps, slots, submits,
# publish, budgets -- is compared unmasked.
_VOLATILE = re.compile(r"SourceName\((.*?), 0x[0-9a-fA-F]+\)")


def normalize_source_hash(value: str) -> str:
    return _VOLATILE.sub(lambda m: f"SourceName({m.group(1)})", value or "")


def normalized_plan(plan):
    """Return ``(plan, masked)`` with the volatile SourceName address removed."""
    masked_source = normalize_source_hash(plan.source_hash)
    if masked_source == plan.source_hash:
        return plan, False
    from dataclasses import replace

    from rknpu_backend.plan import compute_plan_hash

    candidate = replace(plan, source_hash=masked_source, plan_hash="")
    return replace(candidate, plan_hash=compute_plan_hash(candidate)), True


def project_plan(plan) -> dict[str, Any]:
    return {"plan": plan.to_dict(), "descriptor": project_descriptor(plan)}


# -------------------------------------------------------------------- cases

def cases() -> list[dict[str, Any]]:
    from rknpu_backend import (CompileOptions, activation_module,
                               branch_outputs_module, matmul_module,
                               projection_module, swiglu_module)
    from rknpu_backend.frontend.tilelang_adapter import tilelang_available

    tiled = dict(numeric_mode="rk_f16_tiled")
    tiled_lut = dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)
    out: list[dict[str, Any]] = [
        dict(name="rkir_matmul_13x64x33", kind="rkir",
             factory=lambda: matmul_module(m=13, k=64, n=33),
             options=tiled),
        dict(name="rkir_matmul_silu_17x64x33", kind="rkir",
             factory=lambda: matmul_module(m=17, k=64, n=33, activation="silu"),
             options=tiled_lut),
        dict(name="rkir_projection", kind="rkir",
             factory=lambda: projection_module(m=64, d=128, h=256),
             options=tiled_lut),
        dict(name="rkir_swiglu", kind="rkir",
             factory=lambda: swiglu_module(m=32, d=128, h=256),
             options=tiled_lut),
        dict(name="rkir_branch_outputs", kind="rkir",
             factory=lambda: branch_outputs_module(m=16, k=64, n1=32, n2=48),
             options=tiled),
        dict(name="rkir_activation_relu", kind="rkir",
             factory=lambda: activation_module(m=16, n=16, activation="relu"),
             options=tiled),
        dict(name="rkir_matmul_chain_separate", kind="rkir",
             factory=lambda: matmul_module(m=13, k=64, n=33, activation="silu"),
             options=dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True,
                          submit_mode="separate")),
    ]
    if not tilelang_available():
        return out

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

    out += [
        dict(name="tilelang_single_matmul", kind="tilelang",
             factory=lambda: semantics._valid_kernel(16, 256, 16), options=tiled),
        dict(name="tilelang_matmul_silu", kind="tilelang",
             factory=lambda: semantics._silu_kernel(16, 256, 16), options=tiled_lut),
        dict(name="tilelang_kloop_512", kind="tilelang",
             factory=lambda: kloop.k_loop(16, 16, 512, 256), options=tiled),
        dict(name="tilelang_ragged_k257", kind="tilelang",
             factory=lambda: ragged.ragged_tail(13, 33, 257, 16, 64, 256),
             options=tiled),
        dict(name="tilelang_broadcast_scalar_add", kind="tilelang",
             factory=lambda: broadcast.scalar_add(8, 64, 33), options=tiled),
        dict(name="tilelang_broadcast_channel_mul", kind="tilelang",
             factory=lambda: broadcast.channel_mul(8, 64, 33), options=tiled),
        dict(name="tilelang_multi_output", kind="tilelang",
             factory=lambda: multi.branch_outputs(8, 64, 32, 48), options=tiled),
        dict(name="tilelang_mn_tiled", kind="tilelang",
             factory=lambda: mntile.mn_tiled(512, 384, 512, 256, 128, 256),
             options=tiled),
        dict(name="tilelang_silu_proj", kind="tilelang",
             factory=lambda: silu_proj.silu_proj(13, 64, 32), options=tiled_lut),
        dict(name="tilelang_projection_staged", kind="tilelang",
             factory=lambda: staged.projection_staged, options=tiled_lut),
        dict(name="tilelang_ffn_staged", kind="tilelang",
             factory=lambda: ffn.ffn_staged, options=tiled_lut),
        dict(name="tilelang_branch_staged", kind="tilelang",
             factory=lambda: branch.branch_staged, options=tiled),
    ]
    return out


def negatives() -> list[dict[str, Any]]:
    """Cases that must be rejected with the same code, stage and message."""
    from rknpu_backend import CompileOptions, matmul_module
    from rknpu_backend.frontend.tilelang_adapter import tilelang_available

    out = [
        dict(name="strict_fp32_numeric_reject", kind="rkir",
             factory=lambda: matmul_module(m=13, k=64, n=33),
             options=dict(numeric_mode="strict")),
    ]
    if not tilelang_available():
        return out
    semantics = _load("tilelang_semantics")
    kloop = _load("tilelang_kloop")
    mntile = _load("tilelang_mntile")
    silu_proj = _load("tilelang_silu_proj")
    staged = _load("tilelang_projection_staged")
    out += [
        dict(name="negative_grid_step_two", kind="tilelang",
             factory=lambda: semantics.grid_step_two(), options={}),
        dict(name="negative_bad_ragged_k", kind="tilelang",
             factory=lambda: kloop.bad_ragged_k(16, 16, 300, 256), options={}),
        dict(name="negative_bad_ragged_output", kind="tilelang",
             factory=lambda: mntile.bad_ragged_output(512, 384, 512, 256, 256, 256),
             options={}),
        dict(name="negative_missing_narrowing", kind="tilelang",
             factory=lambda: silu_proj.silu_proj_missing_narrowing(13, 64, 32),
             options=dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)),
        dict(name="negative_unlicensed_lut", kind="tilelang",
             factory=lambda: semantics._silu_kernel(16, 256, 16),
             options=dict(numeric_mode="rk_f16_tiled")),
        dict(name="negative_cross_block_read", kind="tilelang",
             factory=lambda: staged.projection_cross_block_read(512, 256),
             options=dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)),
    ]
    return out


def _error_record(exc) -> dict[str, Any]:
    diagnostics = getattr(exc, "diagnostics", None)
    if diagnostics:
        first = diagnostics[0]
        return {"code": _code(first), "stage": getattr(first, "stage", ""),
                "message": getattr(first, "message", "")}
    return {"code": type(exc).__name__, "stage": "", "message": str(exc)}


def _code(diag) -> Any:
    return getattr(diag, "code", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None,
                        help="comma separated case names")
    args = parser.parse_args(argv)

    from rknpu_backend import CompileOptions, compile_rkir, compile_tilelang, \
        default_profile
    import rknpu_backend

    profile = default_profile()
    only = set(args.only.split(",")) if args.only else None
    result: dict[str, Any] = {
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "backend_module": rknpu_backend.__file__,
            "profile_hash": profile.profile_hash,
        },
        "cases": {},
        "negatives": {},
    }
    for spec in cases():
        if only and spec["name"] not in only:
            continue
        options = CompileOptions(**spec["options"])
        factory = spec["factory"]
        try:
            if spec["kind"] == "rkir":
                compiled = compile_rkir(factory(), profile=profile,
                                        options=options)
            else:
                compiled = compile_tilelang(factory(), profile=profile,
                                            options=options)
        except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
            result["cases"][spec["name"]] = {"ok": False,
                                             "error": _error_record(exc)}
            continue
        normalized, masked = normalized_plan(compiled.plan)
        result["cases"][spec["name"]] = {
            "ok": True,
            "plan_hash": normalized.plan_hash,
            "projection": project_plan(normalized),
            "identity": {
                "source_hash_volatile": masked,
                "source_hash_raw": compiled.plan.source_hash,
                "plan_hash_as_compiled": compiled.plan.plan_hash,
                "profile_hash": compiled.plan.profile_hash,
            },
        }
    for spec in negatives():
        if only and spec["name"] not in only:
            continue
        options = CompileOptions(**spec["options"])
        try:
            if spec["kind"] == "rkir":
                compile_rkir(spec["factory"](), profile=profile, options=options)
            else:
                compile_tilelang(spec["factory"](), profile=profile,
                                 options=options)
        except BaseException as exc:  # noqa: BLE001 - expected rejection
            result["negatives"][spec["name"]] = {"rejected": True,
                                                 "error": _error_record(exc)}
            continue
        result["negatives"][spec["name"]] = {"rejected": False}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    ok = sum(1 for c in result["cases"].values() if c["ok"])
    rejected = sum(1 for c in result["negatives"].values() if c["rejected"])
    print(f"cases={len(result['cases'])} compiled={ok} "
          f"negatives={len(result['negatives'])} rejected={rejected} "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
