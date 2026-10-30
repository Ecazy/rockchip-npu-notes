"""Tests that the source rounding contract is distinguishable from final narrowing."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_schedule_experiment import reference_mm_tiled


class RoundingSemanticsTests(unittest.TestCase):
    def test_input_distinguishes_per_k_and_final_fp16_rounding(self):
        rng = np.random.default_rng(0)
        a = np.asarray(rng.normal(0, 1, (4, 512)), dtype=np.float16)
        b = np.asarray(rng.normal(0, 1, (512, 4)), dtype=np.float16)
        per_k = reference_mm_tiled(a, b)
        final = np.asarray(a.astype(np.float32) @ b.astype(np.float32),
                           dtype=np.float16)
        self.assertFalse(np.array_equal(per_k, final))
        self.assertGreater(float(np.max(np.abs(
            per_k.astype(np.float32) - final.astype(np.float32)))), 0.0)

    def test_explicit_source_rounding_metadata_is_visible(self):
        from rknpu_backend import (CompileOptions, compile_tilelang,
                                   default_profile)
        from models.tilelang_matmul import make_matmul_tiled_prim
        result = compile_tilelang(
            make_matmul_tiled_prim(128, 512, 128, bm=64, bn=64),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled"))
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertEqual(first.attrs["source_k_loop"]["source_rounding"],
                         "explicit_per_k_fp16")
        self.assertEqual(first.attrs["source_mn_tiling"]["block_m"], 64)


if __name__ == "__main__":
    unittest.main()
