import importlib.util
import os
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_UNSUPPORTED, compile_tilelang,
                           default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]


def load_fixtures():
    return load_module(ROOT / "tests" / "fixtures" / "tilelang_negative.py",
                       "tl_negative_fixtures")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class TileLangFrontendTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()

    def test_real_matmul_primfunc_extracts(self):
        module = load_module(ROOT / "examples" / "matmul.py", "tl_matmul_fixture")
        result = compile_tilelang(
            module.kernel, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(result.plan.accounting.task_count, 1)
        self.assertEqual(result.plan.bindings[0].name, "A")
        self.assertEqual(result.plan.bindings[1].role, "constant")
        self.assertTrue(result.report["source"]["source_hash"])

    def test_real_matmul_silu_extracts_structural_expression(self):
        module = load_module(ROOT / "examples" / "matmul_silu.py",
                             "tl_matmul_silu_fixture")
        result = compile_tilelang(
            module.kernel, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True))
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates, ["co.mm.f16.v1",
                                     "co.lut.sigmoid.f16.v1",
                                     "co.ew_mul.f16.v1"])

    def test_strict_fp32_accumulator_is_rejected(self):
        module = load_module(ROOT / "examples" / "matmul.py", "tl_strict_fixture")
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(module.kernel, profile=self.profile,
                             options=CompileOptions(numeric_mode="strict"))
        self.assertEqual(ctx.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_square_shapes_resolve_roles_from_copy_flow(self):
        fixtures = load_fixtures()
        result = compile_tilelang(
            fixtures.square_matmul(), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(result.plan.bindings[0].shape, (16, 16))
        self.assertEqual(result.plan.bindings[2].shape, (16, 16))
        self.assertEqual(result.plan.accounting.task_count, 1)

    def test_nonzero_accumulator_fill_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.nonzero_fill(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_extra_copy_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.extra_copy(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_transposed_activation_index_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.wrong_activation_index(),
                             profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                    allow_approx_lut=True))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_other_activation_expression_is_not_matched_as_silu(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.incorrect_silu_exponent(),
                             profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                    allow_approx_lut=True))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_looped_gemm_is_rejected(self):
        # A serial loop around the GEMM means repeated accumulation; emitting one
        # task would silently drop the loop semantics.
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.looped_gemm(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("loop", ctx.exception.diagnostics[0].message)

    def test_false_block_predicate_is_rejected(self):
        # A guarded body may never run; emitting tasks would execute work the
        # source program does not perform.
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.false_predicate(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("predicate", ctx.exception.diagnostics[0].message)

    def test_variable_predicate_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.variable_predicate(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("predicate", ctx.exception.diagnostics[0].message)

    def test_unfolded_compound_predicate_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.compound_predicate(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("predicate", ctx.exception.diagnostics[0].message)

    def test_nested_guarded_block_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.nested_guarded_block(),
                             profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("predicate", ctx.exception.diagnostics[0].message)

    def test_activation_loop_must_start_at_zero(self):
        fixtures = load_fixtures()
        for factory, expected in (
                (fixtures.shifted_outer_activation_loop, "[1, 0]"),
                (fixtures.shifted_inner_activation_loop, "[0, 1]"),
                (fixtures.negative_activation_loop, "[-1, 0]")):
            with self.assertRaises(BackendError) as ctx:
                compile_tilelang(
                    factory(), profile=self.profile,
                    options=CompileOptions(numeric_mode="rk_f16_tiled",
                                           allow_approx_lut=True))
            self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
            message = ctx.exception.diagnostics[0].message
            self.assertIn("lower bounds", message)
            self.assertIn(expected, message)

    def test_multi_block_grid_without_k_loop_is_rejected(self):
        # A wider grid is only lowered together with a K loop; a bare
        # multi-block launch has no proven output coverage.
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.multi_block_launch(),
                             profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("block grid", ctx.exception.diagnostics[0].message)

    def test_clear_accum_is_rejected(self):
        # The zero fill is the only allowed clear; clear_accum=True would change
        # the accumulation contract.
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.clear_accum_matmul(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_NUMERIC_CONTRACT)
        self.assertIn("clear_accum", ctx.exception.diagnostics[0].message)

    def test_transposed_gemm_is_rejected(self):
        fixtures = load_fixtures()
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(fixtures.transposed_gemm(), profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_swiglu_multigemm_is_explicitly_rejected(self):
        module = load_module(ROOT / "examples" / "swiglu.py", "tl_swiglu_fixture")
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(module.kernel, profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                    allow_approx_lut=True))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
