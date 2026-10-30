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
        # The source schedule is now applied: 2 M tiles * 3 N tiles * 2 K
        # tiles.  The old planner silently re-tiled this to 256x256.
        self.assertEqual(result.plan.accounting.task_count, 12)
        self.assertEqual(len(result.plan.publish), 6)

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

    def test_mn_tile_size_changes_task_buffer_and_plan_identity(self):
        small = self.compile(self.fixtures.mn_tiled(m=512, n=384, k=512,
                                                    bm=128, bn=128)).plan
        large = self.compile(self.fixtures.mn_tiled(m=512, n=384, k=512,
                                                    bm=256, bn=128)).plan
        self.assertEqual(small.accounting.task_count, 24)
        self.assertEqual(large.accounting.task_count, 12)
        self.assertNotEqual(len(small.buffers), len(large.buffers))
        self.assertNotEqual(small.plan_hash, large.plan_hash)
        small_sched = next(a for a in small.actions
                           if a.template_id == "co.mm.f16.v1").attrs[
                               "source_schedule"]
        large_sched = next(a for a in large.actions
                           if a.template_id == "co.mm.f16.v1").attrs[
                               "source_schedule"]
        self.assertEqual((small_sched["tile_m"], small_sched["tile_n"]),
                         (128, 128))
        self.assertEqual((large_sched["tile_m"], large_sched["tile_n"]),
                         (256, 128))

    def test_traversal_order_changes_serialized_plan(self):
        row = self.compile(self.fixtures.mn_tiled(order="m_major")).plan
        col = self.compile(self.fixtures.mn_tiled(order="n_major")).plan
        self.assertEqual(row.accounting.task_count, col.accounting.task_count)
        self.assertEqual(row.accounting.data_bo_count, col.accounting.data_bo_count)
        self.assertNotEqual(row.plan_hash, col.plan_hash)
        self.assertNotEqual([(p.dst_row, p.dst_col) for p in row.publish],
                            [(p.dst_row, p.dst_col) for p in col.publish])
        sched = next(a for a in col.actions
                     if a.template_id == "co.mm.f16.v1").attrs[
                         "source_schedule"]
        self.assertEqual(sched["order"], "n_major")

    def test_unsupported_traversal_order_is_rejected(self):
        self.assert_rejected(self.fixtures.mn_tiled(order="row_major"),
                             RK_E_UNSUPPORTED, "traversal order")

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
