"""Real TileLang SwiGLU FFN frontend: explicit FP16 positive, implicit FP32 negative.

Only the staged-FP16 variant that materialises the projection in FP16 *before*
the residual add maps onto the current hardware numeric contract.  The original
FP32-accumulator residual is kept as a negative control: accepting it would
silently move a rounding point.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_UNSUPPORTED, compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import (FFN_NARROWING_NAMES,
                                                     extract,
                                                     tilelang_available)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_ffn():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_ffn_fixtures", str(FIXTURES / "tilelang_ffn.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class FFNExtractionTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixtures = load_ffn()

    def test_fixture_is_a_real_three_gemm_ffn(self):
        text = self.fixtures.swiglu_ffn().script()
        self.assertEqual(text.count("T.gemm("), 3)
        self.assertIn("T.sigmoid", text)
        self.assertIn("Y", text)

    def test_explicit_f16_variant_extracts_and_compiles(self):
        kernel = self.fixtures.swiglu_ffn_f16(m=8, d=64, h=32, bk=64)
        text = kernel.script()
        self.assertEqual(text.count("T.gemm("), 3)
        for name in ("gate16", "up16", "silu16", "mul16", "proj16"):
            self.assertIn(name, text)

        options = CompileOptions(numeric_mode="rk_f16_tiled",
                                 allow_approx_lut=True)
        module = extract(kernel, options=options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "matmul", "silu", "mul", "matmul", "add"])
        self.assertEqual([(v.name, v.type.shape) for v in module.values],
                         [("X", (8, 64)), ("Wg", (64, 32)),
                          ("Wu", (64, 32)), ("Wd", (32, 64)),
                          ("Zg", (8, 32)), ("Zu", (8, 32)),
                          ("S", (8, 32)), ("A", (8, 32)),
                          ("H", (8, 64)), ("Y", (8, 64))])
        self.assertEqual(module.ops[2].inputs, (module.ops[0].outputs[0],))
        self.assertEqual(module.ops[3].inputs,
                         (module.ops[2].outputs[0], module.ops[1].outputs[0]))
        self.assertEqual(module.ops[4].inputs[0], module.ops[3].outputs[0])
        self.assertEqual(module.ops[5].inputs,
                         (module.ops[4].outputs[0], module.inputs[0]))
        declared = module.ops[0].attrs["source_narrowing_points"]
        self.assertEqual(declared, list(FFN_NARROWING_NAMES))

        result = compile_tilelang(kernel, profile=self.profile,
                                  options=options)
        self.assertEqual([a.template_id for a in result.plan.actions
                          if a.kind == "npu_task"],
                         ["co.mm.f16.v1", "co.mm.f16.v1",
                          "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1",
                          "co.ew_mul.f16.v1", "co.mm.f16.v1",
                          "co.ew_add.f16.v1"])
        self.assertEqual(len(result.plan.submits), 5)
        self.assertEqual({s.recipe_id for s in result.plan.submits},
                         {"linear_k_acc_v1", "gate_ew_v1",
                          "linear_epilogue_v1"})
        self.assertEqual(sum(a.kind == "upload" for a in result.plan.actions), 6)
        self.assertEqual(sum(a.kind == "download" for a in result.plan.actions), 1)
        self.assertEqual(result.report["source"]["narrowing_points"],
                         list(FFN_NARROWING_NAMES))

    def test_experimental_opt_in_does_not_block_generic_frontend_lowering(self):
        # The TileLang frontend emits natural [K,N] weights (`b_layout="kn"`),
        # which the `nk` whole-graph specialisation does not match.  The opt-in
        # must fall through to the generic per-node path, not reject the graph.
        kernel = self.fixtures.swiglu_ffn_f16(m=8, d=64, h=32, bk=64)
        result = compile_tilelang(
            kernel, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True,
                                   allow_experimental=True))
        self.assertEqual({s.recipe_id for s in result.plan.submits},
                         {"linear_k_acc_v1", "gate_ew_v1",
                          "linear_epilogue_v1"})
        self.assertEqual(len(result.plan.submits), 5)

    def test_fp32_residual_variant_is_rejected_even_with_all_opt_ins(self):
        for factory in (self.fixtures.swiglu_ffn_fp32_residual,
                        self.fixtures.swiglu_ffn):
            kernel = factory(m=8, d=64, h=32, bk=64)
            for mode in ("chain", "separate"):
                with self.subTest(factory=factory.__name__, mode=mode):
                    with self.assertRaises(BackendError) as ctx:
                        compile_tilelang(
                            kernel, profile=self.profile,
                            options=CompileOptions(
                                numeric_mode="rk_f16_tiled",
                                allow_approx_lut=True,
                                allow_experimental=True,
                                submit_mode=mode))
                    diagnostic = ctx.exception.diagnostics[0]
                    self.assertEqual(diagnostic.code, RK_E_NUMERIC_CONTRACT,
                                     diagnostic.message)
                    self.assertIn("residual", diagnostic.message)
                    self.assertIn("FP32", diagnostic.message)
                    self.assertIn("final T.copy", diagnostic.message)


if __name__ == "__main__":
    unittest.main()
