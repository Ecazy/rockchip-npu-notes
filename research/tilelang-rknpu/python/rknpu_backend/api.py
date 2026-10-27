# SPDX-License-Identifier: GPL-3.0-or-later
"""Legacy compile/load/save/explain API.  No device is opened here.

These entry points are compatibility spellings.  They all call the one pipeline in
:mod:`rknpu_backend.codegen.pipeline`; nothing here legalizes, plans or encodes a
second time.  New code should prefer
:func:`rknpu_backend.artifact.compile_artifact`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import TargetProfile
from .codegen import pipeline as codegen
from .codegen.pipeline import Compilation, explain_plan
from .errors import RK_E_UNSUPPORTED, reject
from .ir import Module
from .options import CompileOptions
from .plan import Plan
from .serialize import load_plan as _load_plan
from .serialize import save_plan as _save_plan

__all__ = ["CompileResult", "compile_rkir", "compile_tilelang", "explain_plan",
           "load_plan", "save_plan"]


@dataclass
class CompileResult:
    """Compatibility result: the plan, its diagnostics and an explain report.

    ``artifact`` exposes the same compilation through the artifact facade; it is
    derived from the stored :class:`Compilation`, never re-planned.
    """

    plan: Plan
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    _compilation: Compilation | None = field(default=None, repr=False,
                                             compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "diagnostics": [dict(item) for item in self.diagnostics],
                "report": dict(self.report)}

    @property
    def artifact(self):
        """The same compilation as a :class:`~rknpu_backend.artifact.CompiledArtifact`."""
        from .artifact import CompiledArtifact

        if self._compilation is None:
            reject(RK_E_UNSUPPORTED,
                   "this CompileResult was not produced by the compile pipeline",
                   stage="api")
        return CompiledArtifact.from_compilation(self._compilation)


def _result(compilation: Compilation) -> CompileResult:
    return CompileResult(plan=compilation.plan,
                         diagnostics=[dict(item) for item in compilation.diagnostics],
                         report=dict(compilation.report),
                         _compilation=compilation)


def compile_rkir(module: Module, *, profile: TargetProfile,
                 options: CompileOptions) -> CompileResult:
    """Compile already-normalized RKIR to an immutable RKPlan offline."""
    if not isinstance(module, Module):
        reject(RK_E_UNSUPPORTED, "compile_rkir expects an ir.Module", stage="api")
    return _result(codegen.compile_graph(module, profile=profile,
                                        options=options, stage="api"))


def compile_tilelang(prim_func: Any, *, profile: TargetProfile,
                     options: CompileOptions, b_is_input: bool = False) -> CompileResult:
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
    graph = codegen.normalize_tilelang(prim_func, profile=profile,
                                       options=options, b_is_input=b_is_input)
    return _result(codegen.compile_graph(graph, profile=profile, options=options,
                                         stage="frontend"))


def load_plan(path: str) -> Plan:
    return _load_plan(path)


def save_plan(plan: Plan, path: str) -> None:
    if not isinstance(plan, Plan):
        reject(RK_E_UNSUPPORTED, "save_plan expects a Plan", stage="api")
    _save_plan(plan, path)
