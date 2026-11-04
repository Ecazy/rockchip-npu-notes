# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-stage source-operation rewrite for the real QKV PrimFunc."""
from __future__ import annotations

from collections import Counter
import unittest

import tilelang

from models.tilelang_matmul import make_matmul_prim
from rknpu_backend import CompileOptions
from tests.fixtures import tilelang_semantics as sem
from rknpu_backend.frontend.full_logical_rewrite import (
    locate_full_logical_source, rewrite_full_logical_tir)


class SourceOperationRewriteTests(unittest.TestCase):
    def test_qkv_operations_are_scheduled_then_lowered_individually(self):
        source = make_matmul_prim(1, 576, 960)
        located = locate_full_logical_source(source)
        self.assertEqual(len(located.statements), 5)
        self.assertEqual(located.gemm_index, 3)
        self.assertEqual(tuple(region.name for region in located.writes[0]), ("As",))
        self.assertEqual(tuple(region.name for region in located.writes[1]), ("Bs",))
        self.assertEqual(tuple(region.name for region in located.writes[2]), ("Cl",))
        self.assertIn("Cl", tuple(region.name for region in located.reads[3]))
        self.assertIn("Cl", tuple(region.name for region in located.writes[3]))
        self.assertEqual(tuple(region.name for region in located.reads[4]), ("Cl",))
        self.assertEqual(tuple(region.name for region in located.writes[4]), ("Y",))
        result = rewrite_full_logical_tir(
            source,
            options=CompileOptions(numeric_mode="rk_f16_tiled"),
            max_tasks=1000)

        scheduled = result.scheduled.script()
        target = result.target.script()
        instantiated = result.instantiated.script()

        # Stage 1 keeps the real TileLang operations visible, but their regions
        # now come from the actual source regions plus the tile/K coordinates.
        self.assertIn(
            "T.fill(T.region(Cl[0, 0], 2, "
            "T.min(256, 1 - mi * 256), T.min(256, 960 - ni * 256)), 0)",
            scheduled)
        self.assertIn(
            "T.copy(T.region(A[mi * 256, ko * 256], 1, "
            "T.min(256, 1 - mi * 256), T.min(256, 576 - ko * 256)), "
            "T.region(As[0, 0], 2, T.min(256, 1 - mi * 256), "
            "T.min(256, 576 - ko * 256)))",
            scheduled)
        self.assertIn(
            "T.copy(T.region(B[ko * 256, ni * 256], 1, "
            "T.min(256, 576 - ko * 256), T.min(256, 960 - ni * 256)), "
            "T.region(Bs[0, 0], 2, T.min(256, 576 - ko * 256), "
            "T.min(256, 960 - ni * 256)))",
            scheduled)
        self.assertIn("T.gemm(", scheduled)
        self.assertIn(
            "T.copy(T.region(Cl[0, 0], 1, T.min(256, 1 - mi * 256), "
            "T.min(256, 960 - ni * 256)), "
            "T.region(Y[mi * 256, ni * 256], 2, "
            "T.min(256, 1 - mi * 256), T.min(256, 960 - ni * 256)))",
            scheduled)
        self.assertNotIn('"rk_', scheduled)

        # Stage 2 visits those same scheduled operations and replaces each one
        # with target operations while retaining the loops around them.
        self.assertIn(
            'Acc16 = T.sblock_alloc_buffer((256, 256), "float16"', target)
        self.assertIn("for mi, ni in T.grid(1, 4)", target)
        self.assertIn("for ko in range(3)", target)
        self.assertIn('"rk_fill", "As"', target)
        self.assertIn('"rk_fill", "Bs"', target)
        self.assertIn('"rk_copy", "A", "As", mi * 256, ko * 256', target)
        self.assertIn('"rk_copy", "B", "Bs", ko * 256, ni * 256', target)
        self.assertIn('"rk_copy", "Acc16", "Cl"', target)
        self.assertIn('"rk_gemm", "A", "B", "Y"', target)
        self.assertIn('"rk_copy", "Cl", "Acc16"', target)
        self.assertIn('"rk_store", "Acc16", "Y", mi * 256, ni * 256', target)
        self.assertNotIn("tl.tileop.", target)

        # Existing task instantiation is still the next stage, not a new
        # recognizer or a template-built replacement kernel.
        self.assertNotIn("T.grid(", instantiated)
        self.assertNotIn("range(3)", instantiated)
        self.assertEqual(Counter(task["kind"] for task in result.tasks),
                         {"fill": 28, "copy": 48, "gemm": 12, "store": 4})
        self.assertEqual([stage.name for stage in result.stages],
                         ["schedule-source-ops", "lower-scheduled-ops",
                          "instantiate"])
        self.assertEqual(result.stages[0].notes["original_statements"], 5)
        self.assertEqual(result.stages[0].notes["moved_to_k"], [0, 1])
        self.assertEqual(result.stages[0].notes["local_range"], [0, 4])


    def test_real_silu_store_is_retiled_and_lowered_as_one_operation(self):
        source = sem._silu_kernel(16, 256, 16)
        located = locate_full_logical_source(source)
        self.assertEqual(located.store_statement_index, 4)
        self.assertEqual(tuple(loop.extent for loop in located.store_loops),
                         (16, 16))
        self.assertIn("T.sigmoid", str(located.store_expression))
        self.assertIn("C_l", tuple(region.name for region in located.reads[4]))
        self.assertIn("C_l", tuple(region.name for region in located.writes[4]))

        result = rewrite_full_logical_tir(
            source,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True),
            max_tasks=1000)
        self.assertEqual(result.facts.activation, "silu")

        scheduled = result.scheduled.script()
        target = result.target.script()
        self.assertIn(
            "for i in T.parallel(T.min(256, 16 - mi * 256)):",
            scheduled)
        self.assertIn(
            "for j in T.parallel(T.min(256, 16 - ni * 256)):",
            scheduled)
        self.assertIn(
            "C_l[i, j] = C_l[i, j] * T.sigmoid(C_l[i, j])",
            scheduled)
        self.assertIn(
            'T.call_extern("void", "rk_silu", "Acc16", '
            'T.min(256, 16 - mi * 256), T.min(256, 16 - ni * 256))',
            target)
        self.assertNotIn("tl.tileop.", target)


if __name__ == "__main__":
    unittest.main()
