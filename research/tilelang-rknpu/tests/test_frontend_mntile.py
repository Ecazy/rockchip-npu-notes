"""P1 batch 2: real TileLang M/N-tiled GEMM extraction.

The extractor may re-arrange the M/N tiling (specific.md 12.4) only when it proves
each output element is covered exactly once and no communication is involved.
These tests pin the coverage proof, the index roles of the block variables and the
adjacent negatives.
"""
import importlib.util
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_SHAPE, RK_E_UNSUPPORTED,
                           compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]


def load_fixtures():
    spec = importlib.util.spec_from_file_location(
        "tl_mntile_fixtures",
        str(ROOT / "tests" / "fixtures" / "tilelang_mntile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class MNTileTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixtures = load_fixtures()

    def compile(self, prim_func, **kwargs):
        return compile_tilelang(
            prim_func, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled", **kwargs))

    def assert_rejected(self, prim_func, code, needle):
        with self.assertRaises(BackendError) as ctx:
            self.compile(prim_func)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, code, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    def test_two_dimensional_grid_extracts_whole_problem(self):
        result = self.compile(self.fixtures.mn_tiled())
        self.assertEqual(result.plan.bindings[0].shape, (512, 512))
        self.assertEqual(result.plan.bindings[1].shape, (512, 384))
        self.assertEqual(result.plan.bindings[2].shape, (512, 384))
        # M/N re-tiled at 256: 2 M tiles * 2 N tiles * 2 K tiles.
        self.assertEqual(result.plan.accounting.task_count, 8)
        self.assertEqual(len(result.plan.publish), 4)

    def test_mn_tiling_and_coverage_are_recorded(self):
        result = self.compile(self.fixtures.mn_tiled())
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        mn = first.attrs["source_mn_tiling"]
        self.assertEqual(mn["block_m"], 256)
        self.assertEqual(mn["block_n"], 128)
        self.assertEqual(mn["grid_x"], 3)
        self.assertEqual(mn["grid_y"], 2)
        self.assertEqual(mn["coverage"], "exact_once")
        # The source tile is re-tiled to the 256-wide lowering tile, so the
        # report must show both.
        self.assertEqual(result.report["source"]["source_mn_tiling"], mn)
        self.assertEqual(result.report["source"]["source_k_loop"]["source_k_block"],
                         256)

    def test_one_dimensional_grid_is_accepted(self):
        result = self.compile(self.fixtures.mn_tiled_n_only())
        self.assertEqual(result.plan.bindings[0].shape, (256, 512))
        self.assertEqual(result.plan.bindings[2].shape, (256, 512))
        self.assertEqual(result.plan.accounting.task_count, 4)

    def test_m_only_grid_is_accepted(self):
        # A 1-D grid can cover the M axis: the extractor infers the axis from how
        # the block variable indexes the output tile.
        result = self.compile(self.fixtures.mn_tiled_m_only())
        self.assertEqual(result.plan.bindings[0].shape, (512, 512))
        self.assertEqual(result.plan.bindings[2].shape, (512, 256))
        mn = next(a for a in result.plan.actions
                  if a.template_id == "co.mm.f16.v1").attrs["source_mn_tiling"]
        self.assertEqual(mn["grid_y"], 2)
        self.assertEqual(mn["grid_x"], 1)

    def test_ambiguous_single_axis_grid_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_single_grid_axis_ambiguous(),
                             RK_E_UNSUPPORTED,
                             "cannot prove which output axis")

    def test_ragged_output_tiling_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_ragged_output(), RK_E_SHAPE,
                             "does not cover the output exactly once")

    def test_block_variable_must_match_the_operand_axis(self):
        self.assert_rejected(self.fixtures.bad_block_var_on_wrong_axis(),
                             RK_E_UNSUPPORTED,
                             "does not match the supported tiling")

    def test_output_tile_must_be_placed_by_the_grid_indices(self):
        self.assert_rejected(self.fixtures.bad_output_tile_swapped(),
                             RK_E_UNSUPPORTED,
                             "does not match the supported tiling")

    def test_three_dimensional_grid_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_three_dim_grid(),
                             RK_E_UNSUPPORTED, "at most a 2-D block grid")


if __name__ == "__main__":
    unittest.main()
