"""Real TileLang full-logical GEMM sources are lowered to 256-wide tiles."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.frontend.tilelang_adapter import tilelang_available


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class LargeGemmSourceTests(unittest.TestCase):
    def compile(self, m, k, n, *, b_input=False):
        import models.tilelang_matmul as tmm
        return compile_tilelang(
            tmm.make_matmul_prim(m, k, n), profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False),
            b_is_input=b_input).plan

    def test_non_aligned_full_logical_gemm_is_tiled(self):
        plan = self.compile(3, 257, 300)
        self.assertEqual(len(plan.constants), 1)
        self.assertEqual(tuple(plan.constants[0].shape), (257, 300))
        templates = [a.template_id for a in plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates, ["co.mm.f16.v1", "co.mm_acc.f16.v1",
                                     "co.mm.f16.v1", "co.mm_acc.f16.v1"])
        self.assertEqual(plan.accounting.task_count, 4)
        self.assertEqual([b.shape for b in plan.bindings if b.role == "input"],
                         [(3, 257)])
        self.assertEqual([b.shape for b in plan.bindings if b.role == "output"],
                         [(3, 300)])

    def test_dynamic_weight_full_logical_gemm_has_no_constant(self):
        plan = self.compile(9, 64, 256, b_input=True)
        self.assertEqual(plan.constants, ())
        self.assertEqual([b.name for b in plan.bindings if b.role == "input"],
                         ["A", "B"])
        self.assertTrue(all(a.template_id.startswith("co.mm")
                            for a in plan.actions if a.kind == "npu_task"))


if __name__ == "__main__":
    unittest.main()
