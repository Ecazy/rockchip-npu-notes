"""P1 batch 1: real TileLang K-loop extraction.

Positive: the canonical blocked GEMM must become one RKIR matmul with logical
K = trips * block_k, and the planner must split it into 256-wide K tiles.
Negative: every way of breaking the reduction (ragged K, cleared accumulator,
duplicated GEMM, loop variable leaking into another axis, work inside the loop,
parallel loop, tile-to-tile copies, output written inside the loop) must be
rejected with a diagnostic that names the violated property.
"""
import importlib.util
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                           RK_E_UNSUPPORTED, compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]


def load_fixtures():
    spec = importlib.util.spec_from_file_location(
        "tl_kloop_fixtures", str(ROOT / "tests" / "fixtures" / "tilelang_kloop.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class KLoopTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixtures = load_fixtures()

    def compile(self, prim_func, **kwargs):
        options = CompileOptions(numeric_mode=kwargs.pop("numeric_mode",
                                                         "rk_f16_tiled"),
                                 allow_approx_lut=True, **kwargs)
        return compile_tilelang(prim_func, profile=self.profile, options=options)

    def assert_rejected(self, prim_func, code, needle):
        with self.assertRaises(BackendError) as ctx:
            self.compile(prim_func)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, code, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    # ---- positives ---------------------------------------------------------
    def test_canonical_k_loop_extracts_logical_k(self):
        result = self.compile(self.fixtures.k_loop(k=512, bk=256))
        self.assertEqual(result.plan.bindings[0].shape, (16, 512))
        self.assertEqual(result.plan.bindings[1].shape, (512, 16))
        # 512 / 256 primitive K tiles
        self.assertEqual(result.plan.accounting.task_count, 2)
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates, ["co.mm.f16.v1", "co.mm_acc.f16.v1"])

    def test_plain_serial_loop_is_accepted(self):
        result = self.compile(self.fixtures.k_loop(k=512, bk=256, serial=True))
        self.assertEqual(result.plan.accounting.task_count, 2)

    def test_single_iteration_loop_matches_single_tile(self):
        looped = self.compile(self.fixtures.k_loop(k=256, bk=256)).plan
        self.assertEqual(looped.accounting.task_count, 1)
        self.assertEqual(looped.bindings[0].shape, (16, 256))

    def test_k_4096_covers_the_whole_primitive_range(self):
        result = self.compile(self.fixtures.k_loop(m=8, n=8, k=4096, bk=256))
        self.assertEqual(result.plan.accounting.task_count, 16)

    def test_source_loop_is_recorded_in_the_plan_and_report(self):
        result = self.compile(self.fixtures.k_loop(k=512, bk=256))
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        loop = first.attrs["source_k_loop"]
        # How the source loop was tiled, and that the pipelining hint was
        # dropped (a scheduling hint) rather than silently obeyed.  The source
        # K block must equal the MVP semantic tile, so no re-tiling happens.
        self.assertEqual(loop["source_k_block"], 256)
        self.assertEqual(loop["source_k_trips"], 2)
        self.assertEqual(loop["lowering_k_tile"], 256)
        self.assertTrue(loop["pipeline_hint_dropped"])
        self.assertEqual(loop["source_loop_annotations"], {"num_stages": 2})
        report = result.report["source"]
        self.assertEqual(report["numeric_mode"], "rk_f16_tiled")
        self.assertEqual(report["source_k_loop"]["source_k_trips"], 2)

    def test_serial_loop_records_no_dropped_pipeline_hint(self):
        result = self.compile(self.fixtures.k_loop(k=512, bk=256, serial=True))
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertFalse(first.attrs["source_k_loop"]["pipeline_hint_dropped"])

    def test_strict_numeric_mode_still_rejects_the_tiled_path(self):
        with self.assertRaises(BackendError) as ctx:
            self.compile(self.fixtures.k_loop(k=512, bk=256),
                         numeric_mode="strict")
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)

    # ---- negatives ---------------------------------------------------------
    def test_gemm_before_copies_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_gemm_before_copies(),
                             RK_E_UNSUPPORTED, "must precede the GEMM")

    def test_gemm_between_copies_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_gemm_between_copies(),
                             RK_E_UNSUPPORTED, "must precede the GEMM")

    def test_ragged_k_is_rejected(self):
        # 300 = 1 * 256 + 44: the last K slice is genuinely partial, which the
        # looped extractor does not expand yet.
        self.assert_rejected(self.fixtures.bad_ragged_k(), RK_E_SHAPE,
                             "does not tile the parameter shapes exactly")

    def test_source_block_k_must_be_256(self):
        # specific.md: the MVP semantic K block is 256; any other source block
        # changes the reduction/narrowing order and must be rejected, not merged.
        diagnostic = self.assert_rejected(self.fixtures.bad_block_k_128(),
                                          RK_E_UNSUPPORTED, "fixed at 256")
        self.assertEqual(diagnostic.actual, 128)

    def test_unknown_tileop_is_rejected_before_the_k_path(self):
        # Regression: the K-loop branch used to return before the registered
        # tileop check ran, so an unknown instruction could slip through.
        self.assert_rejected(self.fixtures.bad_unknown_tileop_in_loop(),
                             RK_E_UNSUPPORTED, "unhandled call")

    def test_extra_gemm_outside_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_extra_gemm_outside_loop(),
                             RK_E_UNSUPPORTED,
                             "unexpected tile operations outside the K loop")

    def test_extra_copy_outside_loop_is_rejected(self):
        # Rejected by the "exactly one accumulator-to-output copy" rule; either
        # diagnostic is acceptable as long as the extra node is not dropped.
        self.assert_rejected(self.fixtures.bad_extra_copy_outside_loop(),
                             RK_E_UNSUPPORTED,
                             "exactly once outside the loop")

    def test_loop_variable_must_step_the_k_axis(self):
        self.assert_rejected(self.fixtures.bad_both_k_axes_shifted(),
                             RK_E_UNSUPPORTED, "must advance by the block size")

    def test_clear_inside_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_clear_inside_loop(),
                             RK_E_NUMERIC_CONTRACT, "outside the loop")

    def test_clear_accum_in_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_clear_accum_in_loop(),
                             RK_E_NUMERIC_CONTRACT, "clear_accum")

    def test_two_gemms_in_loop_are_rejected(self):
        self.assert_rejected(self.fixtures.bad_two_gemms(), RK_E_UNSUPPORTED,
                             "exactly one GEMM")

    def test_activation_inside_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_activation_in_loop(),
                             RK_E_UNSUPPORTED,
                             "activation inside a K-loop program")

    def test_parallel_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_parallel_loop(), RK_E_UNSUPPORTED,
                             "only a serial/pipelined K loop")

    def test_tile_to_tile_copy_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_copy_from_shared(),
                             RK_E_UNSUPPORTED, "slice copies")

    def test_output_written_inside_loop_is_rejected(self):
        self.assert_rejected(self.fixtures.bad_output_copy_inside_loop(),
                             RK_E_UNSUPPORTED, "not inside it")


if __name__ == "__main__":
    unittest.main()
