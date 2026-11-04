# SPDX-License-Identifier: GPL-3.0-or-later
"""The compile artifact: one verified plan, instantiable without a JSON round-trip.

The intended lifecycle is::

    artifact = compile_artifact(prim_func, profile=profile, options=options)
    artifact.explain()                            # host only, no device
    workspace = artifact.instantiate(fd=fd, constants=consts)
    result = workspace.run(inputs)

``export_plan`` is a *projection* for offline deployment and replay.  It is never
required before ``instantiate``, and no in-memory call path goes through it.

An artifact owns no device resource:

* it never stores an fd, a BO, a C plan handle or a native descriptor,
* it never stores the caller's input arrays or the source PrimFunc,
* each :meth:`CompiledArtifact.instantiate` allocates independent resources
  through the same production workspace path used by ``Workspace.create``,

so releasing the original PrimFunc (or the artifact itself) after instantiation
cannot invalidate a live workspace.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .capabilities import TargetProfile
from .codegen import pipeline as codegen
from .codegen.pipeline import Compilation, CompileSnapshot
from .errors import RK_E_INVALID, RK_E_UNSUPPORTED, reject
from .ir import Module
from .options import CompileOptions
from .plan import Plan
from .runtime.bindings import Workspace
from .serialize import save_plan as _save_plan

__all__ = ["CompiledArtifact", "compile_artifact"]


@dataclass(frozen=True)
class CompiledArtifact:
    """An immutable, device-free compilation product.

    Fields are private by convention: the artifact exposes ``plan``,
    ``snapshot``, ``explain()``, ``instantiate()`` and ``export_plan()``.  There is
    no mutable cache of descriptors: every instantiation re-encodes the plan, so
    a workspace can never observe another workspace's pointer changes.
    """

    _plan: Plan
    _snapshot: CompileSnapshot
    _source: dict[str, Any]
    _diagnostics: tuple[dict[str, Any], ...] = ()
    _retained_graph: Any = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_compilation(cls, compilation: Compilation) -> "CompiledArtifact":
        if not isinstance(compilation, Compilation):
            reject(RK_E_UNSUPPORTED, "from_compilation expects a Compilation",
                   stage="artifact")
        source = copy.deepcopy(compilation.report.get("source", {}))
        return cls(_plan=compilation.plan, _snapshot=compilation.snapshot,
                   _source=source,
                   _diagnostics=tuple(dict(item) for item in compilation.diagnostics))

    # -- read-only views --------------------------------------------------
    @property
    def plan(self) -> Plan:
        """The authoritative execution plan.  Treat as immutable."""
        return self._plan

    @property
    def snapshot(self) -> CompileSnapshot:
        return self._snapshot

    @property
    def profile(self) -> TargetProfile:
        return self._snapshot.profile

    @property
    def options(self) -> CompileOptions:
        return self._snapshot.options

    @property
    def name(self) -> str:
        return self._plan.name

    @property
    def plan_hash(self) -> str:
        return self._plan.plan_hash

    @property
    def profile_hash(self) -> str:
        return self._plan.profile_hash

    @property
    def source_hash(self) -> str:
        return self._plan.source_hash

    @property
    def source(self) -> dict[str, Any]:
        return copy.deepcopy(self._source)

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._diagnostics)

    @property
    def has_retained_graph(self) -> bool:
        return self._retained_graph is not None

    def with_retained_graph(self, graph: Any) -> "CompiledArtifact":
        """Attach an optional, releasable diagnostic graph.

        The retained object is never consulted by ``instantiate`` or
        ``export_plan``; ``release_diagnostics()`` drops it.
        """
        return CompiledArtifact(_plan=self._plan, _snapshot=self._snapshot,
                                _source=self._source,
                                _diagnostics=self._diagnostics,
                                _retained_graph=graph)

    def release_diagnostics(self) -> "CompiledArtifact":
        return CompiledArtifact(_plan=self._plan, _snapshot=self._snapshot,
                                _source=self._source,
                                _diagnostics=self._diagnostics)

    # -- host-only reporting ---------------------------------------------
    def explain(self) -> dict[str, Any]:
        """Describe the compiled artifact.  Never opens or inspects a device.

        The report is rebuilt from the artifact's own plan and compile snapshot on
        every call and is fully detached (nested ``attrs`` are deep-copied), so a
        caller cannot use it to change execution.
        """
        report = codegen.explain_plan(self._plan)
        report["source"] = copy.deepcopy(self._source)
        report["compile"] = {
            "options": codegen.options_to_dict(self._snapshot.options),
            "profile": {
                "profile_hash": self._snapshot.profile.profile_hash,
                "chip_id": self._snapshot.profile.chip_id,
                "driver_abi_id": self._snapshot.profile.driver_abi_id,
                "runtime_abi_id": self._snapshot.profile.runtime_abi_id,
                "numeric_policy": self._snapshot.profile.numeric_policy,
                "fallback_policy": self._snapshot.profile.fallback_policy,
                "execution_policy": list(self._snapshot.profile.execution_policy),
            },
            "diagnostics": [copy.deepcopy(item) for item in self._diagnostics],
            "retained_graph": type(self._retained_graph).__name__
            if self._retained_graph is not None else None,
        }
        # Report the values this artifact was actually compiled with instead of a
        # constant: a plan can only exist under rk_f16_tiled, and the fallback
        # policy is whatever the snapshot holds.
        report["policy"] = {
            "numeric_mode": self._snapshot.options.numeric_mode,
            "fallback": self._snapshot.options.fallback,
            "approx_lut": self._snapshot.options.allow_approx_lut,
            "submit_mode": self._snapshot.options.submit_mode,
            "linear_task_order": self._snapshot.options.linear_task_order,
            "device_access": "none (compile/explain)",
        }
        return report

    # -- runtime ---------------------------------------------------------
    def instantiate(self, *, fd: int,
                    constants: dict[str, np.ndarray]) -> Workspace:
        """Create one independent workspace for a caller-owned ``fd``.

        Shares the production resource path (`Workspace.create`): same constant
        validation, same C plan deep-copy, same ownership/BUSY/poisoned rules.
        """
        return Workspace.create(self._plan, fd=fd, constants=constants)

    # -- projections -----------------------------------------------------
    def export_plan(self, path: str) -> None:
        """Write the existing RKPlan JSON format.  No new artifact container."""
        if not isinstance(path, (str, bytes)) and not hasattr(path, "__fspath__"):
            reject(RK_E_INVALID, "export_plan requires a filesystem path",
                   stage="artifact")
        _save_plan(self._plan, str(path))


def _unwrap_prim_func(prim_func: Any) -> Any:
    """Accept a real PrimFunc or a JIT wrapper that directly exposes one."""
    if hasattr(prim_func, "buffer_map"):
        return prim_func
    for attr in ("prim_func", "func", "ir_module"):
        candidate = getattr(prim_func, attr, None)
        if candidate is not None and hasattr(candidate, "buffer_map"):
            return candidate
    return prim_func


CARRIERS = ("rkir",)
RETIRED_CARRIERS = {"native": "the native Relax carrier is retired; use carrier=\"rkir\" with the task-chain lowering"}


def compile_artifact(prim_func: Any, *, profile: TargetProfile,
                     options: CompileOptions,
                     b_is_input: bool = False,
                     retain_graph: bool = False,
                     carrier: str = "rkir") -> CompiledArtifact:
    """Compile a real TileLang PrimFunc (or normalized RKIR) to an artifact.

    No GPU pipeline is executed and no source text is parsed.  ``retain_graph``
    keeps the intermediate compute graph only as a releasable diagnostic object;
    it is never needed by ``instantiate`` or ``export_plan``.

    ``carrier`` selects the compute-representation carrier behind the one
    pipeline.  ``"rkir"`` is the production default.  ``"native"`` is the R4
    pilot carrier, which builds a real ``tvm.relax`` graph instead of RKIR; its
    adoption status is recorded in ``ARCHITECTURE.md`` section 3.2.  Both
    carriers run the same legalizer, planner, storage assignment and encoder, so
    this is not a second backend.
    """
    if carrier in RETIRED_CARRIERS:
        reject(RK_E_UNSUPPORTED, RETIRED_CARRIERS[carrier],
               stage="artifact", actual=carrier, required=list(CARRIERS))
    if carrier not in CARRIERS:
        reject(RK_E_UNSUPPORTED, f"unknown compute carrier {carrier!r}",
               stage="artifact", actual=carrier, required=list(CARRIERS))
    if isinstance(prim_func, Module):
        compilation = codegen.compile_graph(prim_func, profile=profile,
                                           options=options, stage="api")
        return CompiledArtifact.from_compilation(compilation)

    target = _unwrap_prim_func(prim_func)
    from .frontend import tilelang_lowering

    if carrier == "rkir" and \
            tilelang_lowering.is_logical_gemm_candidate(target):
        lowering = tilelang_lowering.compile_logical_gemm_compilation(
            target, profile=profile, options=options,
            b_is_input=b_is_input, name="rknpu-logical-task")
        artifact = CompiledArtifact.from_compilation(lowering.compilation)
        if retain_graph:
            artifact = artifact.with_retained_graph({
                "kind": "instantiated-target-tasks",
                "tasks": list(lowering.tasks),
                "facts": lowering.facts.to_dict(),
            })
        return artifact

    if tilelang_lowering.is_staged_candidate(target):
        compilation = tilelang_lowering.compile_staged_compilation(
            target, profile=profile, options=options,
            name="rknpu-staged")
        artifact = CompiledArtifact.from_compilation(compilation)
        if retain_graph:
            artifact = artifact.with_retained_graph({
                "kind": "staged-launch-boundaries",
                "source": compilation.report.get("source", {}),
            })
        return artifact

    if tilelang_lowering.is_ffn_candidate(target):
        compilation = tilelang_lowering.compile_ffn_compilation(
            target, profile=profile, options=options,
            name="rknpu-ffn-dataflow")
        artifact = CompiledArtifact.from_compilation(compilation)
        if retain_graph:
            artifact = artifact.with_retained_graph({
                "kind": "ffn-source-dataflow",
                "source": compilation.report.get("source", {}),
            })
        return artifact

    if tilelang_lowering.is_k_loop_candidate(target):
        compilation = tilelang_lowering.compile_k_loop_compilation(
            target, profile=profile, options=options, b_is_input=b_is_input,
            name="rknpu-kloop-task")
        artifact = CompiledArtifact.from_compilation(compilation)
        if retain_graph:
            artifact = artifact.with_retained_graph({
                "kind": "k-loop-source-rewrite",
                "source": compilation.report.get("source", {}),
            })
        return artifact

    if tilelang_lowering.is_guarded_tail_candidate(
            target, options=options):
        compilation = tilelang_lowering.compile_guarded_tail_compilation(
            target, profile=profile, options=options,
            name="rknpu-guarded-task", b_is_input=b_is_input)
        artifact = CompiledArtifact.from_compilation(compilation)
        if retain_graph:
            artifact = artifact.with_retained_graph({
                "kind": "guarded-tail-source",
                "source": compilation.report.get("source", {}),
            })
        return artifact

    reject(RK_E_UNSUPPORTED,
           "no migrated TileLang lowering owns this source; the adapter "
           "fallback is retired", stage="frontend")