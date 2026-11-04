# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared TileLang-TIR dispatcher for the RKNPU lowering paths.

The official ``compile_tilelang`` / ``compile_artifact(rkir)`` entries and the
registered backend share this dispatch:

* closed full-logical QKV/SiLU goes through the source-operation rewrite chain;
* guarded-tail GEMMs on the migrated baseline/scheduled_baseline policies go
  through the existing localise/instantiate/task-consumer chain;
* unguarded K-loop GEMMs in the migrated final-only and explicit-roundtrip
  forms use the source-operation rewrite and ordered task consumer;
* guarded K-loop sources, including guarded single-trip and final-only forms,
  use the guarded source-operation lowering; unrelated source forms remain on
  the adapter paths that still explicitly own them.

Classification is structural and policy-based only.  Once selected, the
lowering performs the complete semantic checks and its failure is returned
directly; there is no adapter fallback.
"""
from __future__ import annotations

from typing import Any

from . import rknpu_pass


def is_logical_gemm_candidate(prim_func: Any) -> bool:
    """Whether the official entries must use the task-chain logical path."""
    from . import logical_schedule

    return logical_schedule.is_logical_gemm_candidate(prim_func)


def compile_logical_gemm_compilation(prim_func: Any, *, profile: Any = None,
                                    options: Any = None,
                                    b_is_input: bool = False,
                                    simplify: bool = False,
                                    name: str = "rknpu-logical-task") -> Any:
    """Return the full task-chain Compilation for a logical GEMM candidate."""
    from . import logical_schedule

    return logical_schedule.lower_logical_gemm_tasks(
        prim_func, profile=profile, options=options, b_is_input=b_is_input,
        simplify=simplify, name=name)


def _has_serial_tile_loop(prim_func: Any) -> bool:
    """Whether a non-thread loop contains a tile op (direct/dispatcher use)."""
    tir = rknpu_pass._tir()
    for _, node in rknpu_pass._walk(prim_func.body):
        if not isinstance(node, tir.For) or node.thread_binding is not None:
            continue
        if any(rknpu_pass._is_tile_op_call(inner)
               for _, inner in rknpu_pass._walk(node)):
            return True
    return False


def _has_tile_gemm(prim_func: Any) -> bool:
    """Whether the program contains any tile GEMM at all."""
    for _, node in rknpu_pass._walk(prim_func.body):
        if rknpu_pass._is_tile_op_call(node) and \
                rknpu_pass._op_name(node.value) == "tl.tileop.gemm":
            return True
    return False


def selects_guarded_tail(prim_func: Any) -> bool:
    """The one classification used by the dispatcher and diagnostics.

    A serial tile loop is the guarded-tail form.  Programs without a tile GEMM
    keep the previous guarded-tail rejection path, so this migration does not
    change their diagnostics or accept new shapes.
    """
    return _has_serial_tile_loop(prim_func) or not _has_tile_gemm(prim_func)


def is_guarded_tail_candidate(prim_func: Any, *, options: Any) -> bool:
    """Structural/policy gate for the formal guarded-tail compile entries.

    The admitted shape is: one tile GEMM, a two-axis block grid, one explicit
    non-thread tile/K loop and at least one guarded statement, under a migrated
    source-order policy.  No semantic proof runs here.  A selected source that
    later fails the lowering returns that diagnostic directly; it never falls
    back to the adapter.
    """
    if options is None or not _has_tile_gemm(prim_func):
        return False
    tir = rknpu_pass._tir()
    block_loops = []
    tile_loops = []
    has_guard = False
    for _, node in rknpu_pass._walk(prim_func.body):
        if type(node).__name__ == "IfThenElse":
            has_guard = True
        if not isinstance(node, tir.For):
            continue
        if node.thread_binding is not None:
            block_loops.append(node)
            continue
        if any(rknpu_pass._is_tile_op_call(inner)
               for _, inner in rknpu_pass._walk(node)):
            tile_loops.append(node)
    if len(block_loops) != 2 or len(tile_loops) != 1 or not has_guard:
        return False
    allocations = []
    for _, node in rknpu_pass._walk(prim_func.body):
        if type(node).__name__ == "SBlock":
            allocations.extend(list(getattr(node, "alloc_buffers", ()) or ()))
    shared = [buffer for buffer in allocations
              if rknpu_pass._scope_of(buffer).startswith("shared")]
    fragments = [buffer for buffer in allocations
                 if rknpu_pass._scope_of(buffer).startswith("local")]
    fp16 = [buffer for buffer in fragments if str(buffer.dtype) == "float16"]
    fp32 = [buffer for buffer in fragments if str(buffer.dtype) == "float32"]
    if len(shared) != 2 or len(fp16) not in (0, 1) or len(fp32) != 1:
        return False
    tags = [str(getattr(loop.thread_binding, "thread_tag", ""))
            for loop in block_loops]
    return len(set(tags)) == 2 and set(tags) <= {"blockIdx.x", "blockIdx.y"}


def is_k_loop_candidate(prim_func: Any) -> bool:
    """Structural gate for the migrated explicit unguarded K-loop forms."""
    from . import full_logical_rewrite
    return full_logical_rewrite.is_k_loop_candidate(prim_func)


def is_ffn_candidate(prim_func: Any) -> bool:
    """Structural gate for the two admitted local SwiGLU dataflow chains."""
    from . import full_logical_rewrite
    return full_logical_rewrite.is_ffn_candidate(prim_func)


def compile_ffn_compilation(prim_func: Any, *, profile: Any = None,
                            options: Any = None, b_is_input: bool = False,
                            simplify: bool = False,
                            name: str = "rknpu-ffn-dataflow") -> Any:
    from . import logical_schedule
    return logical_schedule.compile_ffn_compilation(
        prim_func, profile=profile, options=options,
        b_is_input=b_is_input, simplify=simplify, name=name)


def compile_ffn_tasks(prim_func: Any, *, profile: Any = None,
                      options: Any = None, b_is_input: bool = False,
                      simplify: bool = False,
                      name: str = "rknpu-ffn-dataflow") -> tuple[Any, ...]:
    from . import logical_schedule
    return logical_schedule.compile_ffn_tasks(
        prim_func, profile=profile, options=options,
        b_is_input=b_is_input, simplify=simplify, name=name)


def compile_k_loop_compilation(prim_func: Any, *, profile: Any = None,
                               options: Any = None, b_is_input: bool = False,
                               simplify: bool = False,
                               name: str = "rknpu-kloop-task") -> Any:
    """Formal K-loop entry through source-operation lowering."""
    from . import logical_schedule
    return logical_schedule.compile_k_loop_compilation(
        prim_func, profile=profile, options=options, b_is_input=b_is_input,
        simplify=simplify, name=name)


def compile_k_loop_tasks(prim_func: Any, *, profile: Any = None,
                         options: Any = None, b_is_input: bool = False,
                         simplify: bool = False,
                         name: str = "rknpu-kloop-task") -> tuple[Any, ...]:
    from . import logical_schedule
    return logical_schedule.compile_k_loop_tasks(
        prim_func, profile=profile, options=options, b_is_input=b_is_input,
        simplify=simplify, name=name)


def compile_guarded_tail_compilation(prim_func: Any, *, profile: Any = None,
                                     options: Any = None,
                                     simplify: bool = False,
                                     name: str = "rknpu-guarded-task",
                                     b_is_input: bool = False) -> Any:
    """Formal compile-entry lowering while preserving source order policy."""
    return rknpu_pass.compile_guarded_tail_compilation(
        prim_func, profile=profile, options=options, simplify=simplify,
        name=name, b_is_input=b_is_input)


def is_staged_candidate(prim_func: Any) -> bool:
    """Whether actual top-level launch scopes form a staged program."""
    from . import staged_launch
    return staged_launch.is_staged_candidate(prim_func)


def compile_staged_compilation(prim_func: Any, *, profile: Any = None,
                               options: Any = None, b_is_input: bool = False,
                               simplify: bool = False,
                               name: str = "rknpu-staged") -> Any:
    """Compile a launch-segmented staged source through the common pipeline."""
    from . import staged_launch
    from ..codegen import pipeline as codegen
    if b_is_input:
        reject(RK_E_UNSUPPORTED,
               "staged task consumption does not support b_is_input",
               stage="frontend")
    profile_hash = profile.profile_hash if profile is not None else ""
    groups = staged_launch.stage_groups(prim_func)
    attrs = dict(getattr(prim_func, "attrs", {}) or {})
    symbol = str(attrs.get("global_symbol", ""))
    use_task_consumer = len(groups) > 1
    if use_task_consumer:
        module, facts = staged_launch.two_stage_plan_inputs(
            prim_func, options=options, profile_hash=profile_hash)
        compilation = codegen.compile_from_tasks(
            module, [], facts, profile=profile, options=options)
        compilation.report["source"] = {
            "module": module.name,
            "source_hash": module.source_hash,
            "lowering": "staged launches -> ordered tasks -> task consumer",
        }
        return compilation
    reject(RK_E_UNSUPPORTED,
           "staged source is not owned by the task consumer",
           stage="frontend")


def compile_staged_tasks(prim_func: Any, *, profile: Any = None,
                         options: Any = None, b_is_input: bool = False,
                         simplify: bool = False,
                         name: str = "rknpu-staged") -> tuple[Any, ...]:
    compilation = compile_staged_compilation(
        prim_func, profile=profile, options=options,
        b_is_input=b_is_input, simplify=simplify, name=name)
    return (compilation.plan, (), {"kind": "staged-launch-boundaries"},
            ())


def entry_for(prim_func: Any):
    """The conversion function that owns this source form."""
    if is_staged_candidate(prim_func):
        return compile_staged_tasks
    if is_ffn_candidate(prim_func):
        return compile_ffn_tasks
    if is_k_loop_candidate(prim_func):
        return compile_k_loop_tasks
    if selects_guarded_tail(prim_func):
        return rknpu_pass.compile_guarded_tail
    from . import logical_schedule

    return logical_schedule.compile_logical_gemm_tasks


def compile_tilelang_ir(prim_func: Any, *, profile: Any = None,
                        options: Any = None, b_is_input: bool = False,
                        simplify: bool = False,
                        name: str = "rknpu-lowering") -> tuple[Any, ...]:
    """Lower one supported TileLang PrimFunc without invoking the adapter.

    Returns ``(plan, stages, facts, tasks)``.  Both the direct caller and the
    registered TileLang backend call this function, so the conversion choice is
    not duplicated at the two entries.
    """
    if is_staged_candidate(prim_func):
        return compile_staged_tasks(
            prim_func, profile=profile, options=options,
            b_is_input=b_is_input, simplify=simplify, name=name)
    if is_ffn_candidate(prim_func):
        return compile_ffn_tasks(
            prim_func, profile=profile, options=options,
            b_is_input=b_is_input, simplify=simplify, name=name)
    if is_k_loop_candidate(prim_func):
        return compile_k_loop_tasks(
            prim_func, profile=profile, options=options,
            b_is_input=b_is_input, simplify=simplify, name=name)
    if selects_guarded_tail(prim_func):
        return rknpu_pass.compile_guarded_tail(
            prim_func, profile=profile, options=options, simplify=simplify,
            name=name, b_is_input=b_is_input)
    from . import logical_schedule

    return logical_schedule.compile_logical_gemm_tasks(
        prim_func, profile=profile, options=options, b_is_input=b_is_input,
        simplify=simplify, name=name)
