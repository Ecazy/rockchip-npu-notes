# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal regressions for the four confirmed source/task semantic losses."""
import pytest
import tilelang  # Sets up the bundled TVM runtime.
from tvm import tirx as tir
from tvm.tirx import stmt_functor

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend import staged_launch
from tests.fixtures import (tilelang_branch_staged, tilelang_projection_staged,
                            tilelang_silu_proj)

OPTIONS = CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def transform(prim, post):
    return prim.with_body(stmt_functor.ir_transform(prim.body, None, post))


def compile_source(prim):
    return compile_tilelang(prim, profile=default_profile(), options=OPTIONS)


@pytest.mark.parametrize("kind", ["Mul", "Add", "Call"])
def test_nested_local_expression_is_rejected(kind):
    prim = tilelang_silu_proj.silu_proj(13, 64, 32)
    changed = False

    def post(node):
        nonlocal changed
        if (not changed and type(node).__name__ == "BufferStore" and
                type(node.value).__name__ == kind):
            value = node.value
            if kind == "Call":
                value = tir.Call(value.dtype, value.op,
                                 [tir.Mul(tir.const(2, "float16"), value.args[0])])
            else:
                # The confirmed bug: (2*z16)*g16 became z16*g16.
                value = type(value)(tir.Mul(tir.const(2, "float16"), value.a), value.b)
            changed = True
            return tir.BufferStore(node.buffer, value, node.indices)
        return None

    prim = transform(prim, post)
    assert changed
    with pytest.raises(BackendError, match="unsupported nested elementwise"):
        compile_source(prim)


@pytest.mark.parametrize("replacement", [1, None])
def test_staged_fill_value_and_missing_initialization_are_distinct(replacement):
    changed = False

    def post(node):
        nonlocal changed
        if (not changed and type(node).__name__ == "Evaluate" and
                getattr(getattr(node.value, "op", None), "name", "") == "tl.tileop.fill"):
            changed = True
            if replacement is None:
                return tir.Evaluate(0)
            call = node.value
            args = list(call.args)
            args[1] = tir.const(replacement, str(args[1].dtype))
            return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return None

    prim = transform(tilelang_branch_staged.branch_staged, post)
    assert changed
    message = "actual zero initial value" if replacement else "neither a value nor a proven zero"
    with pytest.raises(BackendError, match=message):
        compile_source(prim)


@pytest.mark.parametrize("inside", [False, True])
def test_unconsumed_external_effect_is_rejected(inside):
    prim = tilelang_branch_staged.branch_staged
    effect = tir.Evaluate(tir.call_extern("int32", "external_side_effect"))
    if inside:
        changed = False

        def post(node):
            nonlocal changed
            if not changed and type(node).__name__ == "Evaluate":
                changed = True
                return tir.SeqStmt([effect, node])
            return None

        prim = transform(prim, post)
    else:
        prim = prim.with_body(tir.SeqStmt([prim.body, effect]))
    with pytest.raises(BackendError):
        staged_launch.stage_groups(prim)
    with pytest.raises(BackendError):
        compile_source(prim)


def test_elementwise_copy_after_store_is_not_moved_before_it():
    changed = False

    def post(node):
        nonlocal changed
        if not changed and type(node).__name__ == "SeqStmt":
            seq = list(node.seq)
            # The actual sigmoid launch is copy -> tile loop -> output copy.
            if (len(seq) == 3 and type(seq[0]).__name__ == "Evaluate" and
                    type(seq[1]).__name__ == "For" and
                    type(seq[2]).__name__ == "Evaluate"):
                changed = True
                return tir.SeqStmt([seq[1], seq[0], seq[2]])
        return None

    prim = transform(tilelang_projection_staged.projection_staged, post)
    assert changed
    with pytest.raises(BackendError, match="unestablished value"):
        compile_source(prim)


def test_elementwise_tasks_preserve_actual_copy_and_store_order():
    prim = tilelang_projection_staged.projection_staged
    chain = staged_launch.two_stage_chain(prim, options=OPTIONS)
    tasks = chain["stages"][2]["tasks"]
    assert [t["kind"] for t in tasks[:4]] == ["copy", "copy", "mul", "store"]
    mul = tasks[2]
    assert set(mul["inputs"]) == {tasks[0]["destination"], tasks[1]["destination"]}
    assert mul["output"] == tasks[3]["source"]
    assert mul["input_regions"] == [t["destination_region"] for t in tasks[:2]]


def test_staged_copy_checks_both_actual_regions():
    changed = False

    def post(node):
        nonlocal changed
        if (not changed and type(node).__name__ == "Evaluate" and
                getattr(getattr(node.value, "op", None), "name", "") == "tl.tileop.copy"):
            call = node.value
            args = list(call.args)
            destination = args[1]
            region_args = list(destination.args)
            region_args[2] = region_args[2] - 1
            args[1] = tir.Call(destination.dtype, destination.op, region_args)
            changed = True
            return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return None

    prim = transform(tilelang_branch_staged.branch_staged, post)
    assert changed
    with pytest.raises(BackendError, match="copy region extents differ"):
        compile_source(prim)
