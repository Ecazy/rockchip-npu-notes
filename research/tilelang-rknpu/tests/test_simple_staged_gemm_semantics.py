# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded source-effect, GEMM-region and operand-use seam regressions."""
import pytest
import tilelang  # Loads the bundled TVM.
from tvm import tirx as tir
from tvm.tirx import stmt_functor

from rknpu_backend import BackendError, CompileOptions, compile_tilelang, default_profile
from rknpu_backend.frontend import staged_launch
from tests.fixtures.tilelang_branch_staged import branch_staged

OPTIONS = CompileOptions(numeric_mode="rk_f16_tiled")


def transform(post):
    return branch_staged.with_body(stmt_functor.ir_transform(branch_staged.body, None, post))


def compile_source(prim):
    return compile_tilelang(prim, options=OPTIONS, profile=default_profile()).plan


def tile_call(node, name):
    return (type(node).__name__ == "Evaluate" and
            getattr(getattr(node.value, "op", None), "name", "") == "tl.tileop." + name)


def test_simple_launch_does_not_filter_buffer_store():
    changed = False

    def post(node):
        nonlocal changed
        if not changed and tile_call(node, "copy"):
            changed = True
            buffer = node.value.args[1].args[0].buffer
            effect = tir.BufferStore(buffer, tir.const(1, str(buffer.dtype)), [0, 0])
            return tir.SeqStmt([node, effect])
        return None

    prim = transform(post)
    assert changed
    with pytest.raises(BackendError, match="unconsumed source effect"):
        compile_source(prim)


@pytest.mark.parametrize("name", ["fill", "copy", "gemm"])
def test_simple_launch_rejects_guarded_effects(name):
    changed = False
    parameter = next(iter(branch_staged.buffer_map.values()))
    condition = tir.BufferLoad(parameter, [0, 0]) > tir.const(0, "float16")

    def post(node):
        nonlocal changed
        if not changed and tile_call(node, name):
            changed = True
            return tir.IfThenElse(condition, node, None)
        return None

    prim = transform(post)
    assert changed
    with pytest.raises(BackendError, match="conditional or looped effects"):
        compile_source(prim)


@pytest.mark.parametrize("index", [3, 5, 8, 9, 10, 11])
def test_simple_gemm_consumes_semantic_fields(index):
    changed = False

    def post(node):
        nonlocal changed
        if not changed and tile_call(node, "gemm"):
            changed = True
            call = node.value
            args = list(call.args)
            args[index] = (args[index] - 1 if index == 5 else
                           tir.const(1, str(args[index].dtype)))
            return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return None

    prim = transform(post)
    assert changed
    with pytest.raises(BackendError):
        compile_source(prim)


@pytest.mark.parametrize("operand", [0, 1, 2])
def test_simple_gemm_does_not_assume_local_region_starts_at_zero(operand):
    changed = False

    def post(node):
        nonlocal changed
        if not changed and tile_call(node, "gemm"):
            changed = True
            call = node.value
            args = list(call.args)
            reg = args[operand]
            fields = list(reg.args)
            load = fields[0]
            fields[0] = tir.BufferLoad(load.buffer, [1, load.indices[1]])
            fields[2] = fields[2] - 1
            args[operand] = tir.Call(reg.dtype, reg.op, fields)
            return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return None

    prim = transform(post)
    assert changed
    with pytest.raises(BackendError, match="not the whole buffer"):
        compile_source(prim)


def test_simple_output_store_cannot_shrink_the_gemm():
    changed = False
    parameters = list(branch_staged.buffer_map.values())

    def post(node):
        nonlocal changed
        if not changed and tile_call(node, "copy"):
            call = node.value
            destination = call.args[1].args[0].buffer
            if any(destination.same_as(p) for p in parameters):
                changed = True
                args = []
                for region in call.args:
                    fields = list(region.args)
                    fields[2] = fields[2] // 2
                    args.append(tir.Call(region.dtype, region.op, fields))
                return tir.Evaluate(tir.Call(call.dtype, call.op, args))
        return None

    prim = transform(post)
    assert changed
    with pytest.raises(BackendError, match="not the whole buffer"):
        compile_source(prim)


def test_copy_order_does_not_assign_a_b_or_pack():
    changed = False

    def post(node):
        nonlocal changed
        if not changed and type(node).__name__ == "SeqStmt":
            seq = list(node.seq)
            if len(seq) >= 2 and all(tile_call(n, "copy") for n in seq[:2]):
                changed = True
                return tir.SeqStmt([seq[1], seq[0], *seq[2:]])
        return None

    prim = transform(post)
    assert changed
    stage = staged_launch.two_stage_chain(prim, options=OPTIONS)["stages"][0]
    assert stage["op_inputs"] == ["X", "W1"]
    gemm = next(t for t in stage["tasks"] if t["kind"] == "gemm")
    a_copy, b_copy = [stage["tasks"][i] for i in gemm["operand_producers"]]
    assert a_copy["source"] == "X" and a_copy["operand_uses"] == ["A"]
    assert b_copy["source"] == "W1" and b_copy["operand_uses"] == ["B"]
    assert a_copy["ordinal"] > b_copy["ordinal"]
    assert gemm["gemm_fields"]["shape"] == (256, 64, 256)
    plan = compile_source(prim)
    first = next(a for a in plan.actions if a.kind == "npu_task")
    uploads = {view: action for action in plan.actions if action.kind == "upload"
               for view in action.outputs}
    for view, name, weight in zip(first.inputs, ("X", "W1"), (False, True)):
        upload = uploads[view]
        assert plan.bindings[upload.attrs["source_binding"]].name == name
        assert upload.attrs["transpose_source"] is weight
        assert plan.views[view].tensor.layout == ("co.weight.f16.v1" if weight else "co.cube.f16.v1")
