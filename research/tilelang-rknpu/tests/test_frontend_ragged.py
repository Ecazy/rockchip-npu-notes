"""Explicit guarded rectangular-tail GEMM extraction.

The accepted source contract is: reset both physical input tiles each K
iteration, conditionally load only logical elements, and guard the output store.
`ceildiv` alone is never treated as an access-safety proof.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                           RK_E_UNSUPPORTED, compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import (extract, tilelang_available,
                                                     _op_name, _region)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
tirx = None


def load_fixture():
    global tirx
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_ragged_frontend", str(FIXTURES / "tilelang_ragged.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # TileLang initialises the pinned TVM package path; import it only after the
    # fixture has done so.
    from tvm import tirx as tvm_tirx
    tirx = tvm_tirx
    return module


def _rebuild(node, mutate):
    replacement = mutate(node)
    if replacement is not None:
        return replacement
    if isinstance(node, tirx.SBlockRealize):
        return tirx.SBlockRealize(list(node.iter_values), node.predicate,
                                  _rebuild(node.block, mutate))
    if isinstance(node, tirx.SBlock):
        return tirx.SBlock(
            iter_vars=list(node.iter_vars), reads=list(node.reads),
            writes=list(node.writes), name_hint=node.name_hint,
            body=_rebuild(node.body, mutate),
            init=_rebuild(node.init, mutate) if node.init is not None else None,
            alloc_buffers=list(node.alloc_buffers),
            match_buffers=list(node.match_buffers),
            annotations=dict(node.annotations))
    if isinstance(node, tirx.SeqStmt):
        return tirx.SeqStmt([_rebuild(x, mutate) for x in node.seq])
    if isinstance(node, tirx.For):
        return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                        _rebuild(node.body, mutate), node.thread_binding,
                        dict(node.annotations))
    if isinstance(node, tirx.Evaluate):
        return tirx.Evaluate(_rebuild(node.value, mutate))
    if isinstance(node, tirx.BufferStore):
        return tirx.BufferStore(node.buffer, node.value, list(node.indices),
                                node.predicate)
    if isinstance(node, tirx.IfThenElse):
        return tirx.IfThenElse(
            node.condition,
            _rebuild(node.then_case, mutate),
            _rebuild(node.else_case, mutate) if node.else_case is not None else None)
    return node


def _replace_body(prim_func, mutate):
    return tirx.PrimFunc(prim_func.params, _rebuild(prim_func.body, mutate),
                         prim_func.ret_type, prim_func.buffer_map,
                         prim_func.attrs, prim_func.span)


def _ints_in_lt(expr):
    out = []
    if type(expr).__name__ == "And":
        out.extend(_ints_in_lt(expr.a))
        out.extend(_ints_in_lt(expr.b))
    elif type(expr).__name__ == "LT":
        value = getattr(expr.b, "value", None)
        if isinstance(value, int):
            out.append(value)
    return out


def missing_tile_zero_fill(prim_func, *, bm, bk):
    def mutate(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) \
                and _op_name(node.value) == "tl.tileop.fill":
            reg = _region(node.value.args[0], tirx, "fill", expect_mask=2)
            shape = tuple(int(x) for x in reg.buffer.shape)
            if shape == (bm, bk):
                return tirx.Evaluate(tirx.const(0, "int32"))
        return None
    return _replace_body(prim_func, mutate)


def off_by_one_guard(prim_func, *, bound):
    seen = {"done": False}

    def fix(expr):
        if type(expr).__name__ == "And":
            return tirx.And(fix(expr.a), fix(expr.b))
        if type(expr).__name__ == "LT":
            rhs = getattr(expr.b, "value", None)
            if rhs == bound and not seen["done"]:
                seen["done"] = True
                return tirx.LT(expr.a, tirx.const(bound + 1, "int32"))
        return expr

    def mutate(node):
        if isinstance(node, tirx.IfThenElse) and not seen["done"]:
            return tirx.IfThenElse(
                fix(node.condition),
                _rebuild(node.then_case, mutate),
                _rebuild(node.else_case, mutate)
                if node.else_case is not None else None)
        return None
    return _replace_body(prim_func, mutate)


def wrong_axis_guard(prim_func, *, old_bound, new_bound):
    seen = {"done": False}

    def fix(expr):
        if type(expr).__name__ == "And":
            return tirx.And(fix(expr.a), fix(expr.b))
        if type(expr).__name__ == "LT":
            rhs = getattr(expr.b, "value", None)
            if rhs == old_bound and not seen["done"]:
                seen["done"] = True
                return tirx.LT(expr.a, tirx.const(new_bound, "int32"))
        return expr

    def mutate(node):
        if isinstance(node, tirx.IfThenElse) and not seen["done"]:
            return tirx.IfThenElse(
                fix(node.condition),
                _rebuild(node.then_case, mutate),
                _rebuild(node.else_case, mutate)
                if node.else_case is not None else None)
        return None
    return _replace_body(prim_func, mutate)


def unguarded_output_store(prim_func, *, bounds):
    target = set(bounds)

    def mutate(node):
        if isinstance(node, tirx.IfThenElse) and \
                set(_ints_in_lt(node.condition)) == target:
            return _rebuild(node.then_case, mutate)
        return None
    return _replace_body(prim_func, mutate)


def with_rounding_annotation(prim_func, value):
    from tvm import ir
    attrs = dict(prim_func.attrs)
    attrs["rknpu.source_k_rounding"] = value
    return tirx.PrimFunc(prim_func.params, prim_func.body,
                         prim_func.ret_type, prim_func.buffer_map,
                         ir.DictAttrs(attrs), prim_func.span)


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class RaggedTailFrontendTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixture = load_fixture()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled")

    def assert_rejected(self, prim_func, code, needle):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(prim_func, profile=self.profile,
                             options=self.options)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, code, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    def test_combined_m_n_k_tails_extract_with_explicit_contract(self):
        kernel = self.fixture.ragged_tail(m=13, n=33, k=257, bm=16, bn=64)
        module = extract(kernel, options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops], ["matmul"])
        self.assertEqual([v.type.shape for v in module.values],
                         [(13, 257), (257, 33), (13, 33)])
        attrs = module.ops[0].attrs
        self.assertEqual(attrs["source_k_loop"]["logical_k"], 257)
        self.assertEqual(attrs["source_k_loop"]["source_k_trips"], 2)
        self.assertEqual(attrs["source_k_loop"]["tail_contract"]["tails"],
                         {"M": 13, "N": 33, "K": 1})
        self.assertEqual(attrs["source_mn_tiling"]["coverage"], "ragged_tail")

        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options)
        self.assertEqual(result.plan.accounting.task_count, 2)
        self.assertEqual(len(result.plan.publish), 1)
        self.assertEqual(result.report["source"]["source_tail_contract"],
                         attrs["source_tail_contract"])

    def test_multi_grid_tail_plan_uses_existing_tail_tiling(self):
        result = compile_tilelang(
            self.fixture.ragged_tail(m=257, n=257, k=300, bm=128, bn=128),
            profile=self.profile, options=self.options)
        # The legacy final-narrowing source has no explicit schedule and is
        # therefore lowered by the historical default 256x256 path.
        self.assertEqual(result.plan.accounting.task_count, 8)
        self.assertEqual(len(result.plan.publish), 4)
        self.assertEqual(sorted((p.rows, p.cols) for p in result.plan.publish),
                         [(1, 1), (1, 256), (256, 1), (256, 256)])

    def test_missing_tail_zero_fill_is_rejected(self):
        base = self.fixture.ragged_tail(m=13, n=33, k=257, bm=16, bn=64)
        broken = missing_tile_zero_fill(base, bm=16, bk=256)
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT, "zero fill")

    def test_off_by_one_guard_is_rejected(self):
        base = self.fixture.ragged_tail(m=13, n=33, k=257, bm=16, bn=64)
        broken = off_by_one_guard(base, bound=13)
        self.assert_rejected(broken, RK_E_SHAPE, "boundary")

    def test_wrong_axis_guard_is_rejected(self):
        base = self.fixture.ragged_tail(m=13, n=33, k=257, bm=16, bn=64)
        broken = wrong_axis_guard(base, old_bound=13, new_bound=33)
        self.assert_rejected(broken, RK_E_SHAPE, "guard")

    def test_unprotected_output_write_is_rejected(self):
        base = self.fixture.ragged_tail(m=13, n=33, k=257, bm=16, bn=64)
        broken = unguarded_output_store(base, bounds=(13, 33))
        self.assert_rejected(broken, RK_E_UNSUPPORTED, "output stores")

    # ---- explicit per-K FP16 tail source --------------------------------
    def test_explicit_k576_tail_extracts_three_ordered_blocks(self):
        kernel = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                              bm=16, bn=64)
        module = extract(kernel, options=self.options,
                         profile_hash=self.profile.profile_hash)
        attrs = module.ops[0].attrs
        self.assertEqual(attrs["source_k_rounding"], "explicit_per_k_fp16")
        self.assertEqual(attrs["source_accumulator_dtype"], "float16")
        loop = attrs["source_k_loop"]
        self.assertEqual(loop["logical_k"], 576)
        self.assertEqual(loop["k_blocks"], [
            {"index": 0, "k0": 0, "extent": 256},
            {"index": 1, "k0": 256, "extent": 256},
            {"index": 2, "k0": 512, "extent": 64},
        ])
        self.assertEqual(loop["tail_contract"]["tails"],
                         {"M": 13, "N": 33, "K": 64})
        self.assertEqual(loop["rounding_points"], "after_each_k_block")
        self.assertEqual(attrs["source_schedule"]["tile_m"], 16)
        self.assertEqual(attrs["source_schedule"]["tile_n"], 64)

        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options)
        tasks = [a for a in result.plan.actions if a.kind == "npu_task"]
        self.assertEqual([(a.attrs["k0"], a.attrs["kt"]) for a in tasks],
                         [(0, 256), (256, 256), (512, 64)])
        self.assertEqual([a.template_id for a in tasks],
                         ["co.mm.f16.v1", "co.mm_acc.f16.v1",
                          "co.mm_acc.f16.v1"])

    def test_explicit_k_partition_covers_representative_extents(self):
        expected = {
            1: [1], 255: [255], 256: [256], 257: [256, 1],
            511: [256, 255], 512: [256, 256], 576: [256, 256, 64],
            1536: [256] * 6,
        }
        for k, extents in expected.items():
            with self.subTest(k=k):
                kernel = self.fixture.ragged_tail_f16(
                    m=13, n=33, k=k, bm=16, bn=64)
                module = extract(kernel, options=self.options,
                                 profile_hash=self.profile.profile_hash)
                blocks = module.ops[0].attrs["source_k_loop"]["k_blocks"]
                self.assertEqual([b["extent"] for b in blocks], extents)
                self.assertEqual(sum(extents), k)

    def test_explicit_k257_tail_layout_has_zeroed_physical_padding(self):
        kernel = self.fixture.ragged_tail_f16(m=13, n=33, k=257,
                                              bm=16, bn=64)
        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options)
        tasks = [a for a in result.plan.actions if a.kind == "npu_task"]
        self.assertEqual((tasks[1].attrs["k0"], tasks[1].attrs["kt"],
                          tasks[1].attrs["k_physical_extent"],
                          tasks[1].attrs["k_padding"]),
                         (256, 1, 32, 31))
        tail_views = [v for v in result.plan.views if v.tensor.cols == 1]
        self.assertEqual(len(tail_views), 2)
        for view in tail_views:
            self.assertEqual(view.tensor.physical_cols, 32)
            self.assertEqual(
                result.plan.buffers[view.buffer_id].padding_state, "zero")

    def test_explicit_tail_can_treat_b_as_input(self):
        kernel = self.fixture.ragged_tail_f16(m=3, n=256, k=64,
                                              bm=256, bn=64)
        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options, b_is_input=True)
        roles = [(b.name, b.role) for b in result.plan.bindings]
        self.assertIn(("B", "input"), roles)
        self.assertEqual(result.plan.constants, ())

    def test_explicit_k1536_has_six_full_blocks(self):
        kernel = self.fixture.ragged_tail_f16(m=13, n=33, k=1536,
                                              bm=16, bn=64)
        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options)
        tasks = [a for a in result.plan.actions if a.kind == "npu_task"]
        self.assertEqual([(a.attrs["k0"], a.attrs["kt"]) for a in tasks],
                         [(i * 256, 256) for i in range(6)])
        first = tasks[0]
        self.assertEqual(first.attrs["source_k_rounding"],
                         "explicit_per_k_fp16")
        self.assertEqual(first.attrs["source_k_loop"]["logical_k"], 1536)

    def test_explicit_tail_swapped_grid_is_role_inferred(self):
        kernel = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                              bm=16, bn=64, swapped=True)
        module = extract(kernel, options=self.options,
                         profile_hash=self.profile.profile_hash)
        attrs = module.ops[0].attrs
        self.assertEqual(attrs["source_k_loop"]["logical_k"], 576)
        self.assertEqual(attrs["source_mn_tiling"]["coverage"], "ragged_tail")
        result = compile_tilelang(kernel, profile=self.profile,
                                  options=self.options)
        self.assertEqual(result.plan.accounting.task_count, 3)
        self.assertEqual(len(result.plan.publish), 1)

    def test_explicit_missing_tail_zero_fill_is_rejected(self):
        base = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                            bm=16, bn=64)
        broken = missing_tile_zero_fill(base, bm=16, bk=256)
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT, "zero fill")

    def test_explicit_bad_tail_index_is_rejected(self):
        broken = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                              bm=16, bn=64, bad_index=True)
        self.assert_rejected(broken, RK_E_SHAPE,
                             "canonical guarded tail indexing")

    def test_explicit_unprotected_output_is_rejected(self):
        base = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                            bm=16, bn=64)
        broken = unguarded_output_store(base, bounds=(13, 33))
        self.assert_rejected(broken, RK_E_UNSUPPORTED, "output stores")

    def test_scheduled_final_narrowing_is_rejected(self):
        broken = self.fixture.ragged_tail_legacy_scheduled(m=13, n=33, k=576,
                                                           bm=16, bn=64)
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT,
                             "different K-rounding contract")

    def test_rounding_annotation_mismatch_is_rejected(self):
        base = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                            bm=16, bn=64)
        broken = with_rounding_annotation(base, "fp32_final_narrowing")
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT,
                             "annotation does not match")

    def test_explicit_source_block_k_other_than_256_is_rejected(self):
        broken = self.fixture.ragged_tail_f16(m=13, n=33, k=576,
                                              bm=16, bn=64, bk=128)
        self.assert_rejected(broken, RK_E_UNSUPPORTED, "BK == 256")


if __name__ == "__main__":
    unittest.main()
