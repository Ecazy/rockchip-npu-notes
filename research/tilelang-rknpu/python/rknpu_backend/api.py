# SPDX-License-Identifier: GPL-3.0-or-later
"""Public compile/load/save/explain API.  No device is opened here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import TargetProfile
from .errors import RK_E_UNSUPPORTED, reject
from .ir import Module
from .options import CompileOptions
from .passes.dag import build_draft
from .passes.infer import infer_and_verify
from .passes.legalize import legalize
from .passes.schedule import build_submits
from .passes.storage import assign_storage
from .passes.verify import verify_plan
from .plan import Plan
from .serialize import load_plan as _load_plan
from .serialize import save_plan as _save_plan


@dataclass
class CompileResult:
    plan: Plan
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "diagnostics": list(self.diagnostics),
                "report": dict(self.report)}


def _compile_module(module: Module, *, profile: TargetProfile,
                    options: CompileOptions) -> CompileResult:
    if not isinstance(profile, TargetProfile):
        reject(RK_E_UNSUPPORTED, "compile requires a TargetProfile", stage="api")
    if not isinstance(options, CompileOptions):
        reject(RK_E_UNSUPPORTED, "compile requires CompileOptions", stage="api")
    profile.validate_environment()
    verified = infer_and_verify(module)
    legalized = legalize(verified, profile=profile, options=options)
    draft = build_draft(verified, legalized, profile=profile, options=options)
    submits = build_submits(draft, options=options)
    plan = assign_storage(draft, profile=profile, options=options,
                          submits=submits)
    plan = verify_plan(plan)
    report = explain_plan(plan)
    matmul = next((op for op in module.ops if op.opcode == "matmul"), None)
    narrowing: list[str] = []
    for op in module.ops:
        declared = op.attrs.get("source_narrowing_points")
        if isinstance(declared, (list, tuple)):
            for point in declared:
                if isinstance(point, str) and point not in narrowing:
                    narrowing.append(point)
    report["source"] = {"module": module.name,
                        "source_hash": module.source_hash,
                        "numeric_mode": options.numeric_mode,
                        "submit_mode": options.submit_mode,
                        "approx_lut": options.allow_approx_lut,
                        "narrowing_points": narrowing,
                        "source_k_loop": (matmul.attrs.get("source_k_loop")
                                          if matmul is not None else None),
                        "source_mn_tiling": (matmul.attrs.get("source_mn_tiling")
                                             if matmul is not None else None),
                        "source_tail_contract": (
                            matmul.attrs.get("source_tail_contract")
                            if matmul is not None else None)}
    return CompileResult(plan=plan, diagnostics=[], report=report)


def compile_rkir(module: Module, *, profile: TargetProfile,
                 options: CompileOptions) -> CompileResult:
    """Compile already-normalized RKIR to an immutable RKPlan offline."""
    if not isinstance(module, Module):
        reject(RK_E_UNSUPPORTED, "compile_rkir expects an ir.Module", stage="api")
    return _compile_module(module, profile=profile, options=options)


def compile_tilelang(prim_func: Any, *, profile: TargetProfile,
                     options: CompileOptions) -> CompileResult:
    """Extract a real TileLang PrimFunc and compile it to RKPlan.

    TileLang is an optional import.  Without the pinned dependency this function
    rejects immediately; it never falls back to a string parser or a fake IR.
    """
    from .frontend import tilelang_adapter

    if not tilelang_adapter.tilelang_available():
        reject(RK_E_UNSUPPORTED,
               "TileLang is not installed in this Python environment",
               stage="frontend",
               alternative="install the pinned optional TileLang dependency from "
                           "provenance.json")
    # JIT wrappers are accepted only by unwrapping a real PrimFunc, never by
    # executing a GPU pipeline or guessing from names.
    if not hasattr(prim_func, "buffer_map"):
        for attr in ("prim_func", "func", "ir_module"):
            candidate = getattr(prim_func, attr, None)
            if candidate is not None and hasattr(candidate, "buffer_map"):
                prim_func = candidate
                break
    module = tilelang_adapter.extract(prim_func, options=options,
                                      profile_hash=profile.profile_hash)
    return _compile_module(module, profile=profile, options=options)


def load_plan(path: str) -> Plan:
    return _load_plan(path)


def save_plan(plan: Plan, path: str) -> None:
    if not isinstance(plan, Plan):
        reject(RK_E_UNSUPPORTED, "save_plan expects a Plan", stage="api")
    _save_plan(plan, path)


def explain_plan(plan: Plan) -> dict[str, Any]:
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
        "policy": {
            "numeric_mode": "rk_f16_tiled",
            "fallback": "error",
            "device_access": "none (compile/explain)",
        },
    }
