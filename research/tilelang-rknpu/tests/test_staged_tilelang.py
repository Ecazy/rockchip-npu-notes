"""Real multi-block TileLang projection/FFN/branch -> tiled RKIR closure."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_SHAPE, RK_E_UNSUPPORTED, compile_tilelang,
                           default_profile)
from rknpu_backend.frontend.tilelang_adapter import extract, tilelang_available
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def f16(x):
    return np.asarray(x, dtype=np.float16)


def kchain(x, w):
    """Natural [K,N] weights; explicit FP16 writeback at each 256 K tile."""
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    acc = f16(a[:, :256] @ b[:256, :])
    for k0 in range(256, a.shape[1], 256):
        acc = f16(acc.astype(np.float64) +
                  a[:, k0:k0 + 256] @ b[k0:k0 + 256, :])
    return acc


def projection_reference(x, w1, w2):
    z = kchain(x, w1)
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    silu = f16(z.astype(np.float32) * sig.astype(np.float32))
    p = kchain(silu, w2)
    return f16(p.astype(np.float32) + x.astype(np.float32))


def ffn_reference(x, wg, wu, wd):
    gate = kchain(x, wg)
    up = kchain(x, wu)
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))
    act = f16(silu.astype(np.float32) * up.astype(np.float32))
    p = kchain(act, wd)
    return f16(p.astype(np.float32) + x.astype(np.float32))


def load_fixture(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name,
                                                  FIXTURES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class StagedTileLangTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cls.profile = default_profile()
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.projection = load_fixture("staged_projection",
                                      "tilelang_projection_staged.py")
        cls.ffn = load_fixture("staged_ffn", "tilelang_ffn_staged.py")
        cls.branch = load_fixture("staged_branch", "tilelang_branch_staged.py")

    def setUp(self):
        self._arrays = []
        bos = self.lib.mock_control("rk_mock_live_bos")
        res = self.lib.mock_control("rk_mock_resident")
        self.addCleanup(self.assert_no_leak, bos, res)

    def assert_no_leak(self, bos, res):
        self.assertEqual(self.lib.mock_control("rk_mock_untracked"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_tag_overflow"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), bos)
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), res)
        self._arrays.clear()

    def _binding(self, value_id, array):
        self._arrays.append(array)
        return make_binding(value_id, array)

    def _compile(self, prim_func):
        return compile_tilelang(
            prim_func, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True)).plan

    def _run(self, plan, constants, inputs):
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        constant_bindings = []
        for spec in plan.constants:
            if spec.name.startswith("__identity_"):
                array = f16(np.eye(spec.shape[0], dtype=np.float32))
            else:
                array = constants[spec.name]
            constant_bindings.append(
                self._binding(spec.binding_id, array))
        ws = self.lib.create_workspace(cplan, constant_bindings, fd=0)
        self.addCleanup(ws.destroy)
        outputs = {}
        out_bindings = []
        for binding in plan.bindings:
            if binding.role != "output":
                continue
            array = np.zeros(binding.shape, dtype=np.float16)
            self._arrays.append(array)
            outputs[binding.name] = array
            out_bindings.append(self._binding(binding.id, array))
        ws.run([self._binding(b.id, inputs[b.name])
                for b in plan.bindings if b.role == "input"], out_bindings)
        return outputs

    def test_projection_staged_extracts_ssa_and_executes(self):
        module = extract(self.projection.projection_staged,
                         options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                allow_approx_lut=True),
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "sigmoid", "mul", "matmul", "add"])
        self.assertEqual(module.inputs, (0,))
        self.assertEqual(module.outputs, (7,))
        self.assertEqual([v.role for v in module.values],
                         ["input", "constant", "constant", "temporary",
                          "temporary", "temporary", "temporary", "output"])
        plan = self._compile(self.projection.projection_staged)
        self.assertEqual(plan.accounting.task_count, 22)
        self.assertEqual(len(plan.publish), 2)
        mm2 = [a for a in plan.actions
               if a.template_id == "co.mm_acc.f16.v1"]
        self.assertTrue(mm2)
        self.assertEqual(mm2[0].attrs["source_k_loop"]["source_k_trips"], 2)
        self.assertIn("explicit f16 writeback",
                      mm2[0].attrs["source_k_loop"]["narrowing_base"])
        rng = np.random.default_rng(51)
        x = f16(rng.normal(0, 0.15, (512, 256)))
        w1 = f16(rng.normal(0, 0.08, (256, 512)))
        w2 = f16(rng.normal(0, 0.08, (512, 256)))
        out = self._run(plan, {"W1": w1, "W2": w2}, {"X": x})
        np.testing.assert_allclose(out["Y"].astype(np.float64),
                                   projection_reference(x, w1, w2).astype(
                                       np.float64),
                                   atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_ffn_staged_extracts_branch_and_executes(self):
        module = extract(self.ffn.ffn_staged,
                         options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                allow_approx_lut=True),
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "matmul", "sigmoid", "mul", "mul",
                          "matmul", "add"])
        plan = self._compile(self.ffn.ffn_staged)
        self.assertEqual(plan.accounting.task_count, 30)
        self.assertEqual(len(plan.publish), 2)
        rng = np.random.default_rng(52)
        x = f16(rng.normal(0, 0.15, (512, 256)))
        wg = f16(rng.normal(0, 0.08, (256, 512)))
        wu = f16(rng.normal(0, 0.08, (256, 512)))
        wd = f16(rng.normal(0, 0.08, (512, 256)))
        out = self._run(plan, {"Wg": wg, "Wu": wu, "Wd": wd}, {"X": x})
        np.testing.assert_allclose(
            out["Y"].astype(np.float64),
            ffn_reference(x, wg, wu, wd).astype(np.float64),
            atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_branch_staged_keeps_two_outputs(self):
        module = extract(self.branch.branch_staged,
                         options=CompileOptions(numeric_mode="rk_f16_tiled"),
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "matmul"])
        self.assertEqual(module.outputs, (3, 4))
        plan = self._compile(self.branch.branch_staged)
        self.assertEqual(plan.accounting.task_count, 6)
        self.assertEqual(len(plan.publish), 6)
        rng = np.random.default_rng(53)
        x = f16(rng.normal(0, 0.15, (512, 256)))
        w1 = f16(rng.normal(0, 0.08, (256, 512)))
        w2 = f16(rng.normal(0, 0.08, (256, 256)))
        out = self._run(plan, {"W1": w1, "W2": w2}, {"X": x})
        np.testing.assert_array_equal(out["Y1"], kchain(x, w1))
        np.testing.assert_array_equal(out["Y2"], kchain(x, w2))

    def test_same_launch_cross_block_dependency_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(self.projection.projection_cross_block_read)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("cross-block", ctx.exception.diagnostics[0].message)

    def test_fp32_accumulator_without_k_tile_writeback_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(self.projection.projection_fp32_accum_only)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)
        self.assertIn("FP16 accumulator",
                      ctx.exception.diagnostics[0].message)

    def test_incomplete_block_coverage_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(self.projection.projection_bad_coverage)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE)
        self.assertIn("cover", ctx.exception.diagnostics[0].message)


if __name__ == "__main__":
    unittest.main()
