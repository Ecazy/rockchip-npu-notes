# SPDX-License-Identifier: GPL-3.0-or-later
"""Task-chain entries for closed full-logical GEMMs.

Both the no-activation QKV form and the supported SiLU accumulator update are
dispatched to :mod:`rknpu_backend.frontend.full_logical_rewrite`, which rewrites
the real TileLang fill/copy/GEMM/store/output-copy nodes and then lowers the
scheduled operations item-by-item.  The result reuses the guarded-tail task
instantiation and task-consumer planner; the synthetic RKIR module is only a
binding/NumericContract carrier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..capabilities import default_profile
from ..codegen.pipeline import Compilation, compile_from_tasks
from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED,
                      reject)
from ..options import CompileOptions
from .full_logical_rewrite import (rewrite_ffn_tir,
                                   rewrite_full_logical_tir,
                                   rewrite_k_loop_tir)
from .rknpu_pass import (LocalFacts, Stage, _const, _digest, _minimal_module,
                         _op_name, _walk, _tir)


@dataclass(frozen=True)
class LogicalTaskLowering:
    """One task-chain lowering product plus its diagnostic stages."""

    compilation: Compilation
    stages: tuple[Any, ...]
    facts: Any
    tasks: tuple[dict[str, Any], ...]

    @property
    def plan(self) -> Any:
        return self.compilation.plan


def _predicate_true(node: Any) -> bool:
    if node is None or node is True:
        return True
    if type(node).__name__ != "IntImm":
        return False
    try:
        return str(node.dtype) == "bool" and bool(node.value)
    except Exception:
        return False


def is_logical_gemm_candidate(prim_func: Any) -> bool:
    """Structural input-range classifier; never edits or executes the source.

    This is deliberately a classification, not a "try the new path, then fall
    back".  The admitted range is the closed full-logical form with a 1x1 launch
    grid and no conditional/loop structure around tile ops.  Other forms keep
    their existing adapter path; if a structurally admitted form is malformed,
    the task-chain proof rejects it rather than falling back.
    """
    try:
        tir = _tir()
        if not isinstance(prim_func, tir.PrimFunc):
            return False
        params = list((getattr(prim_func, "buffer_map", None) or {}).values())
        if len(params) != 3:
            return False
        for buffer in params:
            shape = tuple(int(value) for value in buffer.shape)
            if len(shape) != 2 or str(buffer.dtype) != "float16":
                return False
        calls: list[Any] = []
        stores = 0
        thread_loops: list[str] = []
        for _, node in _walk(prim_func.body):
            kind = type(node).__name__
            if kind == "IfThenElse":
                return False
            if kind == "SBlockRealize" and (
                    not _predicate_true(node.predicate) or
                    list(getattr(node, "iter_values", ()) or ())):
                return False
            if kind == "SBlock" and list(getattr(node, "iter_vars", ()) or ()):
                return False
            if kind == "BufferStore":
                stores += 1
                if stores > 1:
                    return False
            if kind == "For":
                binding = getattr(node, "thread_binding", None)
                if binding is not None:
                    extent = _const(node.extent)
                    if extent is None or extent < 1:
                        return False
                    thread_loops.append(str(getattr(binding, "thread_tag", "")))
                elif any(_op_name(inner.value).startswith("tl.tileop.")
                         for _, inner in _walk(node)
                         if type(inner).__name__ == "Evaluate"):
                    return False
            if kind == "Evaluate" and \
                    _op_name(node.value).startswith("tl.tileop."):
                calls.append(node.value)
        if not (1 <= len(thread_loops) <= 2) or \
                len(set(thread_loops)) != len(thread_loops) or \
                any(tag not in {"blockIdx.x", "blockIdx.y"}
                    for tag in thread_loops):
            return False
        gemms = [call for call in calls if _op_name(call) == "tl.tileop.gemm"]
        fills = [call for call in calls if _op_name(call) == "tl.tileop.fill"]
        copies = [call for call in calls if _op_name(call) == "tl.tileop.copy"]
        if len(gemms) != 1 or len(fills) != 1 or len(copies) != 3:
            return False
        return len(gemms[0].args) >= 13
    except Exception:
        return False


def _finish_task_chain(*, facts: LocalFacts, tasks: list[dict[str, Any]],
                       stages: list[Stage], profile: Any, options: Any,
                       name: str, schedule_declared: bool,
                       lowering: str,
                       extra_facts: dict[str, Any] | None = None,
                       source_extra: dict[str, Any] | None = None
                       ) -> LogicalTaskLowering:
    facts_dict = {"m": facts.logical_m, "k": facts.logical_k,
                  "n": facts.logical_n, "tile_m": facts.tile_m,
                  "tile_n": facts.tile_n, "order": facts.order,
                  "rounding": facts.rounding, "hints": dict(facts.hints),
                  "schedule_declared": schedule_declared}
    facts_dict.update(extra_facts or {})
    module = _minimal_module(facts, name=name)
    compilation = compile_from_tasks(module, tasks, facts_dict,
                                     profile=profile, options=options)
    compilation.report["source"] = {
        "module": name, "source_hash": facts.source_hash,
        "numeric_mode": options.numeric_mode,
        "source_k_rounding": facts.rounding,
        "source_schedule": {"kind": "mn_tiles_v1",
                            "tile_m": facts.tile_m,
                            "tile_n": facts.tile_n,
                            "order": facts.order,
                            "k_tile": facts.block_k},
        "lowering": lowering,
    }
    compilation.report["source"].update(source_extra or {})
    stages.append(Stage("plan", "analysis",
                        _digest(compilation.plan.plan_hash),
                        {"plan_hash": compilation.plan.plan_hash,
                         "tasks": len(tasks)}))
    return LogicalTaskLowering(compilation=compilation,
                               stages=tuple(stages), facts=facts,
                               tasks=tuple(tasks))


def lower_logical_gemm_tasks(prim_func: Any, *, profile: Any = None,
                             options: Any = None, b_is_input: bool = False,
                             simplify: bool = False,
                             name: str = "rknpu-logical-task") -> LogicalTaskLowering:
    """Compile a full-logical GEMM through target TIR and task consumption."""
    if simplify:
        reject(RK_E_UNSUPPORTED,
               "the full-logical task scheduler proves its source form and does "
               "not run upstream Simplify", stage="frontend")
    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=False)
    if options.numeric_mode != "rk_f16_tiled":
        reject(RK_E_NUMERIC_CONTRACT,
               "the task scheduler only proves the rk_f16_tiled policy",
               stage="frontend", actual=options.numeric_mode)

    rewritten = rewrite_full_logical_tir(
        prim_func, options=options, b_is_input=b_is_input,
        max_tasks=options.max_actions)
    schedule_declared = any(
        str(key).startswith("rknpu.")
        for key in dict(getattr(prim_func, "attrs", None) or {}))
    return _finish_task_chain(
        facts=rewritten.facts, tasks=list(rewritten.tasks),
        stages=list(rewritten.stages), profile=profile, options=options,
        name=name, schedule_declared=schedule_declared,
        lowering="source-op rewrite -> target TIR -> task consumer")


def _annotate_provenance(lowering: Any, result: Any) -> None:
    rounding = result.facts.rounding
    source = lowering.compilation.report.setdefault("source", {})
    source["source_k_rounding"] = rounding
    first_task = next(action for action in lowering.compilation.plan.actions
                      if action.kind == "npu_task")
    source["source_k_loop"] = dict(
        first_task.attrs.get("source_k_loop") or {})
    source["source_mn_tiling"] = dict(
        first_task.attrs.get("source_mn_tiling") or {})
    if rounding == "single_k_tile_fp16":
        source["source_schedule_origin"] = None
        source["mn_grid_source"] = "default_grid"
    elif result.schedule_declared:
        source["source_schedule_origin"] = "annotated"
        source["mn_grid_source"] = "annotation"
    else:
        source["source_schedule_origin"] = "derived"
        source["mn_grid_source"] = "source_geometry"


def compile_ffn_tasks(prim_func: Any, *, profile: Any = None,
                     options: Any = None, b_is_input: bool = False,
                     simplify: bool = False,
                     name: str = "rknpu-ffn-dataflow") -> tuple[Any, ...]:
    if b_is_input:
        reject(RK_E_UNSUPPORTED,
               "local dataflow does not support b_is_input",
               stage="frontend")
    """Compile the local SwiGLU dataflow from ordered target tasks."""
    if simplify:
        reject(RK_E_UNSUPPORTED,
               "the FFN dataflow rewrite does not run upstream Simplify",
               stage="frontend")
    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True)
    data = rewrite_ffn_tir(prim_func, profile_hash=profile.profile_hash,
                           name=name)
    compilation = compile_from_tasks(
        data.module, data.tasks, dict(data.facts), profile=profile,
        options=options)
    compilation.report["source"] = {
        "module": name, "source_hash": data.module.source_hash,
        "numeric_mode": options.numeric_mode,
        "lowering": "source dataflow rewrite -> target tasks -> task consumer",
        "dataflow_nodes": len(data.module.ops),
        "narrowing_points": [
            point for op in data.module.ops
            for point in op.attrs.get("numeric_contract", {}).get(
                "narrowing_points", ()) if point != "k_tile_boundary"],
    }
    return (compilation.plan, list(data.stages), dict(data.facts),
            list(data.tasks))


def compile_ffn_compilation(prim_func: Any, *, profile: Any = None,
                            options: Any = None, b_is_input: bool = False,
                            simplify: bool = False,
                            name: str = "rknpu-ffn-dataflow") -> Any:
    if b_is_input:
        reject(RK_E_UNSUPPORTED,
               "local dataflow does not support b_is_input",
               stage="frontend")
    """Return the task-chain Compilation for the local SwiGLU dataflow."""
    if simplify:
        reject(RK_E_UNSUPPORTED,
               "the FFN dataflow rewrite does not run upstream Simplify",
               stage="frontend")
    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True)
    data = rewrite_ffn_tir(prim_func, profile_hash=profile.profile_hash,
                           name=name)
    compilation = compile_from_tasks(
        data.module, data.tasks, dict(data.facts), profile=profile,
        options=options)
    compilation.report["source"] = {
        "module": name, "source_hash": data.module.source_hash,
        "numeric_mode": options.numeric_mode,
        "lowering": "source dataflow rewrite -> target tasks -> task consumer",
        "dataflow_nodes": len(data.module.ops),
        "narrowing_points": [
            point for op in data.module.ops
            for point in op.attrs.get("numeric_contract", {}).get(
                "narrowing_points", ()) if point != "k_tile_boundary"],
    }
    return compilation


def compile_k_loop_tasks(prim_func: Any, *, profile: Any = None,
                         options: Any = None, b_is_input: bool = False,
                         simplify: bool = False,
                         name: str = "rknpu-kloop-task") -> tuple[Any, ...]:
    """Lower one migrated explicit K-loop through the ordered task chain."""
    if simplify:
        reject(RK_E_UNSUPPORTED,
               "the K-loop lowering does not run upstream Simplify",
               stage="frontend")
    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=False)
    result = rewrite_k_loop_tir(
        prim_func, options=options, b_is_input=b_is_input,
        max_tasks=options.max_actions)
    lowering = _finish_task_chain(
        facts=result.facts, tasks=list(result.tasks),
        stages=list(result.stages), profile=profile, options=options,
        name=name, schedule_declared=result.schedule_declared,
        lowering="K-loop source rewrite -> target TIR -> task consumer",
        extra_facts={"linear_k_loop": True,
                     "block_k": result.facts.block_k},
        source_extra={"source_k_loop": dict(result.source_k_loop)})
    _annotate_provenance(lowering, result)
    return lowering.plan, lowering.stages, lowering.facts, lowering.tasks


def compile_k_loop_compilation(prim_func: Any, *, profile: Any = None,
                               options: Any = None, b_is_input: bool = False,
                               simplify: bool = False,
                               name: str = "rknpu-kloop-task") -> Any:
    """Return the full task-chain Compilation for a migrated K-loop."""
    if simplify:
        reject(RK_E_UNSUPPORTED,
               "the K-loop lowering does not run upstream Simplify",
               stage="frontend")
    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=False)
    result = rewrite_k_loop_tir(
        prim_func, options=options, b_is_input=b_is_input,
        max_tasks=options.max_actions)
    lowering = _finish_task_chain(
        facts=result.facts, tasks=list(result.tasks),
        stages=list(result.stages), profile=profile, options=options,
        name=name, schedule_declared=result.schedule_declared,
        lowering="K-loop source rewrite -> target TIR -> task consumer",
        extra_facts={"linear_k_loop": True,
                     "block_k": result.facts.block_k},
        source_extra={"source_k_loop": dict(result.source_k_loop)})
    _annotate_provenance(lowering, result)
    return lowering.compilation


def compile_logical_gemm_tasks(prim_func: Any, *, profile: Any = None,
                               options: Any = None, b_is_input: bool = False,
                               simplify: bool = False,
                               name: str = "rknpu-logical-task") -> tuple[Any, ...]:
    """Compatibility tuple wrapper around :func:`lower_logical_gemm_tasks`."""
    result = lower_logical_gemm_tasks(
        prim_func, profile=profile, options=options, b_is_input=b_is_input,
        simplify=simplify, name=name)
    return result.plan, result.stages, result.facts, result.tasks
