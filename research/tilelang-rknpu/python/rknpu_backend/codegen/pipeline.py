# SPDX-License-Identifier: GPL-3.0-or-later
"""The one compile pipeline: normalize -> plan -> verify -> encode.

Every frontend entry point in the package funnels through ``compile_graph``.
compile_from_tasks is the pass-chain sibling: it takes the ordered instantiated
target operations plus a module that supplies binding identity and the numeric
contract, skips lazy normalisation and multi-node dispatch, and shares submits,
records, storage and the verifier:

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
``profile_hash``).  The remaining adapter forms use the internal RKIR
``Module``; migrated full-logical and guarded-tail forms use task records.
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
from ..passes.dag import attach_plan_records, build_draft
from ..passes.infer import infer_and_verify
from ..passes.legalize import legalize
from ..passes.schedule import build_submits
from ..passes.storage import assign_storage
from ..passes.verify import verify_plan
from ..plan import Plan


def _logical_ops_view(plan: Plan) -> list[dict[str, Any]]:
    """Each recorded logical op, plus what its tasks really read and write.

    The plan stores the read/write region once, on the host actions, so this
    joins the two rather than keeping a second copy that could drift: a
    converted plan says the same thing here that its uploads actually pack.
    """
    bindings = {binding.id: binding.name for binding in plan.bindings}
    upload_of_view: dict[int, Any] = {}
    downloads_of_view: dict[int, list[Any]] = {}
    for action in plan.actions:
        if action.kind == "upload":
            for view_id in action.outputs:
                upload_of_view[view_id] = action
        elif action.kind == "download":
            for view_id in action.inputs:
                downloads_of_view.setdefault(view_id, []).append(action)
    actions = {action.id: action for action in plan.actions}
    out: list[dict[str, Any]] = []
    for entry in plan.logical_ops:
        record = entry.to_dict()
        reads: list[dict[str, Any]] = []
        writes: list[dict[str, Any]] = []
        for task in entry.tasks:
            action = actions[task]
            for view_id in action.inputs:
                producer = upload_of_view.get(view_id)
                if producer is None:
                    continue
                source = producer.attrs.get("source_binding")
                if not isinstance(source, int) or isinstance(source, bool):
                    continue
                reads.append({
                    "task": task,
                    "binding": bindings.get(source),
                    "row_offset": producer.attrs.get("src_row_offset"),
                    "col_offset": producer.attrs.get("src_col_offset"),
                    "row_stride": producer.attrs.get("src_row_stride"),
                    "transpose": producer.attrs.get("transpose_source"),
                    "cache": producer.attrs.get("cache"),
                })
            for view_id in action.outputs:
                for download in downloads_of_view.get(view_id, ()):
                    target = download.attrs.get("binding_id")
                    if not isinstance(target, int) or isinstance(target, bool):
                        continue
                    writes.append({"task": task,
                                   "binding": bindings.get(target)})
        record["reads"] = reads
        record["writes"] = writes
        out.append(record)
    return out


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
        "linear_task_order": options.linear_task_order,
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
        # Where the M/N block decision came from, and which grid the plan will
        # use because of it: an annotation is a requirement, a proven K-blocked
        # source is honoured source geometry, and everything else takes the
        # backend default grid.
        # "annotated" / "derived" come from the frontend; "caller" means the
        # attribute was written by the caller (direct RKIR), never a frontend
        # derivation, so explain never mislabels it.
        "source_schedule_origin": (
            attrs.get("source_schedule_origin")
            or ("caller" if attrs.get("source_schedule") is not None else None)),
        "mn_grid_source": (
            "annotation" if attrs.get("source_schedule_origin") == "annotated"
            else ("source_geometry"
                  if attrs.get("source_schedule_origin") == "derived"
                  else ("caller" if attrs.get("source_schedule") is not None
                        else "default_grid"))),
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
    # After submit scheduling: host uploads may have been hoisted, so action ids
    # are only final here.
    draft = attach_plan_records(verified, draft, options)
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

    The adapter extractor is retired; every supported source form has a formal
    task-chain entry above this function.  Unknown forms are rejected instead of
    being routed through a hidden graph fallback.
    """
    reject(RK_E_UNSUPPORTED,
           "the generic TileLang adapter extractor is retired; use a formal "
           "source-operation entry", stage="normalize")


def _effective_geometry(plan: Plan) -> dict[str, Any]:
    """Declared nominal block versus the rectangles actually published."""
    rectangles = sorted({(publish.rows, publish.cols) for publish in plan.publish})
    nominals = {
        json.dumps(action.attrs.get("applied_schedule"), sort_keys=True)
        for action in plan.actions
        if action.kind == "npu_task"
        and action.attrs.get("applied_schedule") is not None
    }
    clipped = False
    for rendered in nominals:
        nominal = json.loads(rendered)
        if any(rows < nominal["tile_m"] or cols < nominal["tile_n"]
               for rows, cols in rectangles):
            clipped = True
    return {
        "publish_rects": [[rows, cols] for rows, cols in rectangles],
        "clipped_from_declared": clipped,
    }


def _execution_value(plan: Plan, key: str) -> Any:
    """The numeric execution record the plan itself carries, when it has one."""
    for requirement in getattr(plan, "numeric_requirements", ()) or ():
        execution = requirement.to_dict().get("execution") or {}
        if key in execution:
            return execution[key]
    return None


def explain_plan(plan: Plan) -> dict[str, Any]:
    """Describe a plan without touching a device.

    The returned structure is fully detached: every nested ``attrs`` mapping is
    deep-copied, so a caller that edits an explain report cannot change the
    authoritative plan or a later instantiation.
    """
    if not isinstance(plan, Plan):
        reject(RK_E_UNSUPPORTED, "explain_plan expects a Plan", stage="api")
    declared_schedules = {
        json.dumps(action.attrs.get("source_schedule"), sort_keys=True)
        for action in plan.actions
        if action.kind == "npu_task" and
        action.attrs.get("source_schedule") is not None}
    applied_schedules = {
        json.dumps(action.attrs.get("applied_schedule"), sort_keys=True)
        for action in plan.actions
        if action.kind == "npu_task" and
        action.attrs.get("applied_schedule") is not None}
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
        # The source graph the plan restates, so a requirement can be read
        # against the op it names instead of on its own.
        "logical": {
            "stored": bool(plan.logical_ops),
            "source": "plan" if plan.logical_ops else "not stored",
            "ops": _logical_ops_view(plan),
            "without_tasks": [entry.index for entry in plan.logical_ops
                              if not entry.tasks],
        },
        # What the caller asked of each logical op, which tasks implement it, and
        # what is deliberately NOT claimed: the plan stores requirements, not
        # evidence that the device met them, and it defines no whole-graph
        # tolerance.
        "numeric": {
            "stored": bool(plan.numeric_requirements),
            "source": ("plan" if plan.numeric_requirements else "not stored"),
            "operators": [dict(requirement.to_dict(),
                             tolerance_source=f"plan:op[{requirement.op_index}]")
                          for requirement in plan.numeric_requirements],
            "end_to_end_criterion": "undefined (per-operator requirements only)",
            "device_satisfaction": "not asserted by the plan",
        },
        "accounting": plan.accounting.to_dict(),
        "schedule": {
            # Explicit requirement (what the source asked for), source
            # geometry, the M/N strategy actually adopted, and the linear task
            # policy are reported separately: an equivalence rewrite must be
            # readable, not inferred from a hash.
            "declared": sorted({
                json.dumps(action.attrs.get("source_schedule"), sort_keys=True)
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("source_schedule") is not None
            }),
            "source_geometry": sorted({
                json.dumps(action.attrs.get("source_mn_tiling"),
                           sort_keys=True)
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("source_mn_tiling") is not None
            }),
            "applied": sorted({
                json.dumps(action.attrs.get("applied_schedule"), sort_keys=True)
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("applied_schedule") is not None
            }),
            "source_hints": sorted({
                json.dumps(action.attrs.get("source_hints"), sort_keys=True)
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("source_hints") is not None
            }),
            "linear_task_policy": sorted({
                str(action.attrs.get("linear_task_policy"))
                for action in plan.actions
                if action.kind == "npu_task" and
                action.attrs.get("linear_task_policy") is not None
            }),
            # The declared block is nominal: a tail tile can be smaller.  Both
            # numbers are reported so a clipped declaration is never silent.
            "effective": _effective_geometry(plan),
            # A rewrite is only acceptable when it is readable: was the adopted
            # M/N strategy the declared one, and if not, what changed.
            "reblocked": bool(declared_schedules) and
                         declared_schedules != applied_schedules,
            "k_tile": _execution_value(plan, "k_tile"),
            "accumulation": _execution_value(plan, "k_chain"),
        },
        "policy": {
            "numeric_mode": "rk_f16_tiled",
            "fallback": "error",
            "device_access": "none (compile/explain)",
        },
    }


def compile_from_tasks(module: Any, tasks: list[dict[str, Any]],
                       facts: dict[str, Any], *,
                       profile: TargetProfile,
                       options: CompileOptions) -> Compilation:
    """Compile a plan from the instantiated target operation sequence.

    ``module`` supplies binding identity and the NumericContract only.  The
    ordered ``rk_*`` task records are the authority for operand, region, tile,
    K-slice, accumulation order and publication.  Normalize() (the TileLang
    frontend) and the multi-node dispatch of build_draft are not re-entered;
    submits, records, storage and verification remain the same planner stages.
    """
    from ..passes.dag import build_from_tasks

    snapshot = CompileSnapshot(profile=profile, options=options)
    profile.validate_environment()
    legalized = legalize(module, profile=profile, options=options)
    draft = build_from_tasks(module, tasks=tasks, facts=facts,
                             legalized=legalized,
                             profile=profile, options=options)
    submits = build_submits(draft, options=options)
    draft = attach_plan_records(module, draft, options)
    result = assign_storage(draft, profile=profile, options=options,
                            submits=submits)
    plan = verify_plan(result)
    report = explain_plan(plan)
    return Compilation(plan=plan, snapshot=snapshot, report=report,
                       diagnostics=())
