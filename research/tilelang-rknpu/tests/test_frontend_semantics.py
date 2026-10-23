"""Semantic guards: predicated loads/stores, access masks, grid/K loop geometry.

Every fixture is a real TileLang PrimFunc mutated in exactly one structural way, so
each test proves the guard is reachable and not shadowed by an earlier check.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_UNSUPPORTED, compile_tilelang,
                           default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load(name):
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        f"tl_sem_{name}", str(FIXTURES / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class SemanticGuardTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixtures = load("tilelang_semantics")

    def compile(self, prim_func):
        return compile_tilelang(prim_func, profile=self.profile,
                                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                       allow_approx_lut=True))

    def assert_rejected(self, prim_func, needle):
        with self.assertRaises(BackendError) as ctx:
            self.compile(prim_func)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    def test_accumulator_region_must_be_load_and_store(self):
        # Reachable independently of clear_accum: the region access mask alone is
        # enough to prove the accumulator is not written blind.
        self.assert_rejected(self.fixtures.gemm_accum_store_only(),
                             "pattern requires a load+store")

    def test_predicated_region_load_is_rejected(self):
        self.assert_rejected(self.fixtures.predicated_gemm_load(),
                             "region load carries a predicate")

    def test_block_grid_must_start_at_zero(self):
        self.assert_rejected(self.fixtures.grid_min_one(),
                             "block grid loops must start at zero")

    def test_block_grid_must_step_by_one(self):
        self.assert_rejected(self.fixtures.grid_step_two(),
                             "block grid loops must step by one")

    def test_k_loop_must_start_at_zero(self):
        self.assert_rejected(self.fixtures.k_loop_min_one(),
                             "K loop must start at zero")

    def test_unhandled_call_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self.compile(self.fixtures.unhandled_call_inside_kernel())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("call", ctx.exception.diagnostics[0].message)

    def test_silu_inner_load_predicate_is_rejected(self):
        self.assert_rejected(self.fixtures.silu_inner_load_predicated(),
                             "acc * sigmoid(acc)")

    def test_activation_loop_must_step_by_one(self):
        self.assert_rejected(self.fixtures.activation_loop_step_two(),
                             "must step by one")

    def test_k_loop_must_step_by_one(self):
        self.assert_rejected(self.fixtures.k_loop_step_two(),
                             "K loop must step by one")


if __name__ == "__main__":
    unittest.main()
