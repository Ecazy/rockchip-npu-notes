# SPDX-License-Identifier: GPL-3.0-or-later
"""The one compile pipeline: normalize -> plan -> verify -> encode.

Every entry point in the package funnels through :func:`compile_graph`:

* ``api.compile_tilelang`` / ``api.compile_rkir`` / ``artifact.compile_artifact``
* the CLI ``compile-*`` subcommands
* the model and hybrid campaign tools

The stages and their permitted side effects are fixed:

======  ====================================  =========================
stage   input -> output                      side effects
======  ====================================  =========================
normalize  source or graph -> compute graph  none
plan       compute graph + profile -> Plan   none
verify     Plan -> Plan                      none
encode     Plan -> descriptor (codegen.descriptor)  Host memory only
======  ====================================  =========================

Nothing here opens a device, imports the runtime library, or serializes to JSON.
Planning and encoding are also not allowed to consult a device: the codegen side
has no way to obtain a BO or an fd.

The *compute representation* is a duck-typed graph access protocol (``values``,
``ops``, ``value(id)``, ``inputs``, ``outputs``, ``name``, ``source_hash``,
``profile_hash``) so that the same planner runs over more than one carrier.  The
two implemented carriers are the internal RKIR ``Module`` and the experimental
native TVM graph view in :mod:`rknpu_backend.frontend.native_graph`.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from ..capabilities import TargetProfile
from ..errors import RK_E_UNSUPPORTED, reject
from ..graph import is_compute_graph, missing_graph_attributes
from ..options import CompileOptions
from ..passes.dag import build_draft
from ..passes.infer import infer_and_verify
from ..passes.legalize import legalize
from ..passes.schedule import build_submits
from ..passes.storage import assign_storage
from ..passes.verify import verify_plan
from ..plan import Plan


def require_compute_graph(obj: Any, *, stage: str = "codegen") -> Any:
    """Guard the planner's graph access protocol; see :mod:`rknpu_backend.graph`."""
    if not is_compute_graph(obj):
        reject(RK_E_UNSUPPORTED,
               "compile requires a compute graph (module or native view)",
               stage=stage, actual=type(obj).__name__,
               required=missing_graph_attributes(obj))
    return obj


@dataclass(frozen=True)
class CompileSnapshot:
    """Immutable compile-time capture of the target profile and options.

    The artifact keeps this instead of the caller's mutable objects so later
    option edits cannot change an already compiled plan's meaning.  ``profile``
    and ``options`` are themselves frozen dataclasses; ``target_dict`` is the
    canonical hash-bearing projection used for identity comparison.
    """

    profile: TargetProfile
    options: CompileOptions

    def to_dict(self) -> dict[str, Any]:
        return {"profile_hash": self.profile.profile_hash,
                "chip_id": self.profile.chip_id,
                "runtime_abi_id": self.profile.runtime_abi_id,
                "driver_abi_id": self.profile.driver_abi_id,
                "target": self.profile.to_dict(),
                "options": options_to_dict(self.options)}

    def __post_init__(self) -> None:
        if not isinstance(self.profile, TargetProfile):
            reject(RK_E_UNSUPPORTED, "compile requires a TargetProfile",
                   stage="codegen")
        if not isinstance(self.options, CompileOptions):
            reject(RK_E_UNSUPPORTED, "compile requires CompileOptions",
                   stage="codegen")


def options_to_dict(options: CompileOptions) -> dict[str, Any]:
    return {
        "numeric_mode": options.numeric_mode,
        "allow_approx_lut": options.allow_approx_lut,
        "fallback": options.fallback,
        "submit_mode": options.submit_mode,
        "tile_limit": options.tile_limit,
        "device_budget_bytes": options.device_budget_bytes,
        "host_budget_bytes": options.host_budget_bytes,
        "fuse_epilogue": options.fuse_epilogue,
        "fuse_add_relu": options.fuse_add_relu,
        "allow_experimental": options.allow_experimental,
        "max_actions": options.max_actions,
        "max_ir_nodes": options.max_ir_nodes,
    }


def source_identity(graph: Any, *, options: CompileOptions) -> dict[str, Any]:
    """Describe where a plan came from.  Diagnostics only; never execution."""
    matmul = next((op for op in graph.ops if op.opcode == "matmul"), None)
    narrowing: list[str] = []
    for op in graph.ops:
        declared = op.attrs.get("source_narrowing_points")
        if isinstance(declared, (list, tuple)):
            for point in declared:
                if isinstance(point, str) and point not in narrowing:
                    narrowing.append(point)
    attrs = getattr(matmul, "attrs", {}) or {}
    return {
        "module": graph.name,
        "source_hash": graph.source_hash,
        "numeric_mode": options.numeric_mode,
        "submit_mode": options.submit_mode,
        "approx_lut": options.allow_approx_lut,
        "narrowing_points": narrowing,
        "source_k_loop": attrs.get("source_k_loop"),
        "source_mn_tiling": attrs.get("source_mn_tiling"),
        "source_schedule": attrs.get("source_schedule"),
        "source_k_rounding": attrs.get("source_k_rounding"),
        "source_tail_contract": attrs.get("source_tail_contract"),
    }


def normalize(graph: Any, *, stage: str = "frontend") -> Any:
    """Reject anything the planner is not allowed to reason about."""
    require_compute_graph(graph, stage=stage)
    return infer_and_verify(graph)


def plan_verified(verified: Any, *, profile: TargetProfile,
                  options: CompileOptions) -> Plan:
    """Plan an already-normalized graph.  No device access."""
    legalized = legalize(verified, profile=profile, options=options)
    draft = build_draft(verified, legalized, profile=profile, options=options)
    submits = build_submits(draft, options=options)
    result = assign_storage(draft, profile=profile, options=options,
                            submits=submits)
    return verify_plan(result)


def plan(graph: Any, *, profile: TargetProfile,
         options: CompileOptions) -> Plan:
    """Normalize and plan one graph in a single step.  No device access."""
    return plan_verified(normalize(graph), profile=profile, options=options)


@dataclass(frozen=True)
class Compilation:
    """One verified plan plus the diagnostics that describe it."""

    plan: Plan
    snapshot: CompileSnapshot
    report: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "diagnostics": [dict(item) for item in self.diagnostics],
                "report": copy.deepcopy(self.report)}


def compile_graph(graph: Any, *, profile: TargetProfile,
                  options: CompileOptions,
                  stage: str = "frontend") -> Compilation:
    """Normalize, plan, verify and explain one compute graph."""
    snapshot = CompileSnapshot(profile=profile, options=options)
    profile.validate_environment()
    verified = normalize(graph, stage=stage)
    result = plan_verified(verified, profile=profile, options=options)
    report = explain_plan(result)
    report["source"] = source_identity(verified, options=options)
    return Compilation(plan=result, snapshot=snapshot, report=report,
                       diagnostics=())


def normalize_tilelang(prim_func: Any, *, profile: TargetProfile,
                       options: CompileOptions,
                       b_is_input: bool = False) -> Any:
    """Normalize a real TileLang PrimFunc into the compute representation.

    TileLang stays an optional import: this is the only place the adapter is
    entered, and the caller has already checked availability.
    """
    from ..frontend import tilelang_adapter

    return tilelang_adapter.extract(prim_func, options=options,
                                    profile_hash=profile.profile_hash,
                                    b_is_input=b_is_input)


def normalize_tilelang_native(prim_func: Any, *, profile: TargetProfile,
                              options: CompileOptions,
                              b_is_input: bool = False) -> Any:
    """Normalize a real TileLang PrimFunc into a native TVM graph view.

    This is the R4 pilot carrier.  It never instantiates RKIR values or modules;
    the recognition records go straight to native ``tvm.relax`` nodes and the
    result is a read-only view over them.
    """
    from ..frontend import native_graph

    return native_graph.build(prim_func, options=options,
                              profile_hash=profile.profile_hash,
                              b_is_input=b_is_input)


def explain_plan(plan: Plan) -> dict[str, Any]:
    """Describe a plan without touching a device.

    The returned structure is fully detached: every nested ``attrs`` mapping is
    deep-copied, so a caller that edits an explain report cannot change the
    authoritative plan or a later instantiation.
    """
    if not isinstance(plan, Plan):
        reject(RK_E_UNSUPPORTED, "explain_plan expects a Plan", stage="api")
    slot_payload: dict[int, int] = {}
    slot_buffers: dict[int, list[int]] = {}
    for buffer in plan.buffers:
        slot_payload[buffer.slot_id] = max(slot_payload.get(buffer.slot_id, 0),
                                           buffer.offset + buffer.capacity)
        slot_buffers.setdefault(buffer.slot_id, []).append(buffer.id)
    return {
        "schema_version": plan.schema_version,
        "kind": plan.kind,
        "name": plan.name,
        "plan_hash": plan.plan_hash,
        "profile_hash": plan.profile_hash,
        "source_hash": plan.source_hash,
        "bindings": [b.to_dict() for b in plan.bindings],
        "constants": [c.to_dict() for c in plan.constants],
        "counts": {
            "bindings": len(plan.bindings),
            "buffers": len(plan.buffers),
            "views": len(plan.views),
            "actions": len(plan.actions),
            "tasks": plan.accounting.task_count,
            "submits": plan.accounting.submit_count,
            "data_bos": plan.accounting.data_bo_count,
            "broadcast_expand_actions": sum(
                1 for action in plan.actions
                if action.kind == "upload" and
                action.attrs.get("broadcast") is not None),
        },
        "broadcast": {
            "expanded_bytes": plan.accounting.broadcast_expand_bytes,
            "kinds": sorted({str(action.attrs.get("broadcast"))
                             for action in plan.actions
                             if action.kind == "upload" and
                             action.attrs.get("broadcast") is not None}),
        },
        "slots": [
            {"slot_id": slot_id, "payload_bytes": payload,
             "buffer_ids": sorted(slot_buffers[slot_id])}
            for slot_id, payload in sorted(slot_payload.items())
        ],
        "actions": [a.to_dict() for a in plan.actions],
        "submits": [s.to_dict() for s in plan.submits],
        "publish": [p.to_dict() for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
        "schedule": {
            "applied": sorted({
                json.dumps(action.attrs.get("applied_schedule"), sort_keys=True)
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("applied_schedule") is not None
            }),
            "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback",
        },
        "policy": {
            "numeric_mode": "rk_f16_tiled",
            "fallback": "error",
            "device_access": "none (compile/explain)",
        },
    }
