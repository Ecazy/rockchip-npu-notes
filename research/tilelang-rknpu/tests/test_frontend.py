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

    def test_swiglu_multigemm_is_explicitly_rejected(self):
        module = load_module(ROOT / "examples" / "swiglu.py", "tl_swiglu_fixture")
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(module.kernel, profile=self.profile,
                             options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                    allow_approx_lut=True))
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
