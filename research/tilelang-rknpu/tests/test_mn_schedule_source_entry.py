# SPDX-License-Identifier: GPL-3.0-or-later
"""Why the production gate/up source cannot declare M/N tiling.

The production primitive is a single launch with one final narrowing.  The tiled
schedule writes FP16 back at every 256-wide K block, so the request can only be
honoured by a source that states that writeback (a K loop).  These tests pin the
rejection, its reason, and the working alternative.
"""
from __future__ import annotations

import unittest

import tilelang.language as T

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError
from models.tilelang_matmul import make_matmul_prim
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
DECODE = (1, 576, 3072)
PREFILL = (256, 576, 3072)


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def compile_plan(prim_func):
    return compile_tilelang(prim_func, profile=PROFILE,
                            options=options()).plan


def production_with_grid(m, k, n, bm=256, bn=256, order="m_major",
                         rounding="fp32_final_narrowing"):
    """The production computation, plus the M/N annotation, without a K loop."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               Y: T.Tensor((m, n), "float16")):
        attrs = {"rknpu.schedule": "mn_tiles_v1", "rknpu.tile_m": bm,
                 "rknpu.tile_n": bn, "rknpu.mn_order": order}
        if rounding is not None:
            attrs["rknpu.source_k_rounding"] = rounding
        T.func_attr(attrs)
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=128) as (bx, by):
            As = T.alloc_shared((bm, k), "float16")
            Bs = T.alloc_shared((k, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            T.copy(A[by * bm, 0], As)
            T.copy(B[0, bx * bn], Bs)
            T.clear(Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Y[by * bm, bx * bn])

    return kernel


def k_loop_source(m, k, n, *, bm=256, bn=256, order="m_major", rounding=None):
    """The scheduled K-loop body with the rounding annotation optional."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               Y: T.Tensor((m, n), "float16")):
        attrs = {"rknpu.schedule": "mn_tiles_v1", "rknpu.tile_m": bm,
                 "rknpu.tile_n": bn, "rknpu.mn_order": order}
        if rounding is not None:
            attrs["rknpu.source_k_rounding"] = rounding
        T.func_attr(attrs)
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bx, by):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(k, 256), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if by * bm + i < m and ko * 256 + j < k:
                        As[i, j] = A[by * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, bn):
                    if ko * 256 + i < k and bx * bn + j < n:
                        Bs[i, j] = B[ko * 256 + i, bx * bn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < m and bx * bn + j < n:
                    Y[by * bm + i, bx * bn + j] = Acc[i, j]

    return kernel


class ProductionSourceTests(unittest.TestCase):
    def test_the_unannotated_source_is_accepted_with_planner_defaults(self):
        for shape in (DECODE, PREFILL):
            with self.subTest(shape=shape):
                plan = compile_plan(make_matmul_prim(*shape))
                tasks = [a for a in plan.actions if a.kind == "npu_task"]
                self.assertEqual(len(tasks), 36)
                self.assertEqual(len(plan.submits), 12)
                # The planner's default already narrows at every K block.
                contract = plan.numeric_requirements[0].to_dict()["contract"]
                self.assertEqual(contract["narrowing_points"], ["k_tile_boundary"])
                self.assertEqual(tasks[0].attrs.get("source_k_rounding"),
                                 "fp32_final_narrowing")

    def test_a_grid_without_a_k_loop_is_a_numeric_contract_refusal(self):
        # TILELANG-SCHEDULE.md states this case as a numeric-contract
        # rejection ("a scheduled source that only narrows once"), and the
        # scheduler itself refuses the same rounding later; the frontend now
        # reports the same code so the two layers agree.
        for shape in (DECODE, PREFILL):
            with self.subTest(shape=shape):
                with self.assertRaises(BackendError) as caught:
                    compile_plan(production_with_grid(*shape))
                diagnostic = caught.exception.diagnostics[0]
                self.assertEqual(diagnostic.to_dict().get("code"),
                                 -4)  # RK_E_NUMERIC_CONTRACT
                self.assertIn("narrows only after the full K reduction",
                              diagnostic.message)

    def test_a_k_loop_source_needs_no_rounding_annotation(self):
        # The adapter derives the rounding from the source copies, so asking for
        # the schedule does not require the author to repeat it.
        plan = compile_plan(k_loop_source(*PREFILL, rounding=None))
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        self.assertEqual(len(tasks), 36)
        self.assertEqual(tasks[0].attrs.get("source_k_rounding"),
                         "explicit_per_k_fp16")

    def test_a_final_only_rounding_is_also_refused_by_the_schedule(self):
        # Even with a K loop the schedule refuses a final-only declaration: the
        # two gates say the same thing, the frontend just says it earlier.
        def source():
            import tests.fixtures.tilelang_linear_scheduled as fixture

            func = fixture.make_linear(256, 576, 3072, bm=256, bn=256)
            return func.with_attr("rknpu.source_k_rounding",
                                  "fp32_final_narrowing")

        with self.assertRaises(BackendError) as caught:
            compile_plan(source())
        # The formal lowering owns the semantic check for the selected source.
        self.assertIn("schedule declaration disagrees with the proven geometry",
                      caught.exception.diagnostics[0].message)


class WorkingAlternativeTests(unittest.TestCase):
    def test_the_k_loop_source_tiles_and_orders(self):
        big = compile_plan(make_linear(*PREFILL, bm=256, bn=256, order="m_major"))
        small = compile_plan(make_linear(*PREFILL, bm=128, bn=256, order="m_major"))
        # Two M tiles are needed before the traversal order can differ.
        wide = (512, 576, 512)
        row2 = compile_plan(make_linear(*wide, bm=256, bn=256, order="m_major"))
        col = compile_plan(make_linear(*wide, bm=256, bn=256, order="n_major"))

        def tiles(plan):
            sequence = []
            for action in plan.actions:
                if action.kind != "npu_task":
                    continue
                group = action.attrs.get("submit_group")
                if group and (not sequence or sequence[-1] != group):
                    sequence.append(group)
            return sequence

        def tasks(plan):
            return sum(1 for a in plan.actions if a.kind == "npu_task")

        # tile_m 256 -> one M tile, 128 -> two: the task grid really changes.
        self.assertEqual(tasks(big), 36)
        self.assertEqual(tasks(small), 72)
        # Same grid, different traversal: the emitted tile order changes.
        self.assertEqual(tiles(row2), ["mm:0:0", "mm:0:1", "mm:1:0", "mm:1:1"])
        self.assertEqual(tiles(col), ["mm:0:0", "mm:1:0", "mm:0:1", "mm:1:1"])

    def test_a_single_m_tile_cannot_distinguish_order(self):
        # Decode has M=1: one row block, so m_major and n_major emit the same
        # task sequence.  Only the recorded schedule differs, which is why this
        # shape cannot be used as evidence that traversal changed execution.
        def tiles(plan):
            return [a.attrs.get("submit_group") for a in plan.actions
                    if a.kind == "npu_task" and a.attrs.get("k_index") == 0]

        row = compile_plan(make_linear(*DECODE, bm=256, bn=256, order="m_major"))
        col = compile_plan(make_linear(*DECODE, bm=256, bn=256, order="n_major"))
        self.assertEqual(tiles(row), tiles(col))
        row_order = next(a.attrs["applied_schedule"]["order"]
                         for a in row.actions if a.attrs.get("applied_schedule"))
        col_order = next(a.attrs["applied_schedule"]["order"]
                         for a in col.actions if a.attrs.get("applied_schedule"))
        self.assertEqual((row_order, col_order), ("m_major", "n_major"))


if __name__ == "__main__":
    unittest.main()
