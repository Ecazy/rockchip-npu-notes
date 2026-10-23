"""True numeric closure for the extracted MatMul -> SiLU -> projection kernel.

The PrimFunc goes through ``compile_tilelang`` -> RKIR -> RKPlan -> the C ABI and
the test-only NPU seam.  The numpy reference spells out all five FP16 rounding
points independently; it does not call the mock's C helpers.
"""
import ctypes
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.runtime.ffi import LibRKPlan, RKBinding

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def load_mock():
    if not Path(MOCK_LIB).is_file():
        raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
    return LibRKPlan(MOCK_LIB)


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_silu_proj_exec", str(FIXTURES / "tilelang_silu_proj.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f16(x):
    return np.asarray(x, dtype=np.float16)


def reference(x, w1, w2):
    """Five explicit FP16 narrowing points, in source order.

    1 GEMM1 accumulator -> Z16
    2 sigmoid(Z16)      -> G16
    3 Z16 * G16         -> S16
    4 GEMM2 accumulator -> P16
    5 P16 + X           -> Y16
    """
    z16 = f16(np.matmul(x.astype(np.float32), w1.astype(np.float32)))     # 1
    g16 = f16(1.0 / (1.0 + np.exp(-z16.astype(np.float32))))              # 2
    s16 = f16(z16.astype(np.float32) * g16.astype(np.float32))            # 3
    p16 = f16(np.matmul(s16.astype(np.float32), w2.astype(np.float32)))   # 4
    return f16(p16.astype(np.float32) + x.astype(np.float32))             # 5


def make_binding(value_id, array, holder):
    # Keep the numpy backing store alive for the whole native call sequence.
    holder.append(array)
    binding = RKBinding()
    binding.abi_version = 1
    binding.struct_size = ctypes.sizeof(RKBinding)
    binding.id = value_id
    binding.dtype = 0                 # RK_DTYPE_F16
    binding.rank = 2
    binding.shape = (array.shape[0], array.shape[1])
    binding.data = array.ctypes.data_as(ctypes.c_void_p)
    binding.nbytes = array.nbytes
    return binding


class SiluProjExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_mock()
        cls.fixture = load_fixture()
        cls.profile = default_profile()

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

    def build(self, mode="chain"):
        result = compile_tilelang(
            self.fixture.silu_proj(), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True,
                                   submit_mode=mode))
        return result.plan

    def own_plan(self, plan):
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        return cplan

    def constant_bindings(self, plan, tensors):
        out = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                size = binding.shape[0]
                array = f16(np.eye(size, dtype=np.float32))
            else:
                array = tensors[binding.name]
            out.append(make_binding(binding.id, array, self._arrays))
        return out

    def make_workspace(self, plan, tensors):
        cplan = self.own_plan(plan)
        ws = self.lib.create_workspace(
            cplan, self.constant_bindings(plan, tensors))
        self.addCleanup(ws.destroy)
        return cplan, ws

    def make_io(self, plan, x, y):
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        inputs = [make_binding(in_id, x, self._arrays)]
        outputs = [make_binding(out_id, y, self._arrays)]
        return inputs, outputs

    def test_two_input_rounds_match_independent_reference(self):
        plan = self.build()
        rng = np.random.default_rng(7)
        m, d, h = 13, 64, 32
        w1 = f16(rng.normal(0, 0.3, (d, h)))
        w2 = f16(rng.normal(0, 0.3, (h, d)))
        _cplan, ws = self.make_workspace(plan, {"W1": w1, "W2": w2})
        x1 = f16(rng.normal(0, 0.5, (m, d)))
        y = np.zeros((m, d), dtype=np.float16)
        inputs, outputs = self.make_io(plan, x1, y)

        report = ws.run(inputs, outputs)
        expected1 = reference(x1, w1, w2)
        np.testing.assert_array_equal(y, expected1)
        self.assertEqual(report.task_count, 5)
        # The report counts caller-input uploads per run; constant/identity
        # uploads happen once at workspace creation.  The plan-level test covers
        # all five upload actions.
        self.assertEqual(report.upload_calls, 1)
        self.assertEqual(report.download_calls, 1)

        # Same workspace/BOs, new caller-owned input.  The device-side residual
        # must be read from the current X upload, not a stale first-round value.
        x2 = f16(np.asarray(x1, dtype=np.float32) * 0.5 + 0.125)
        inputs, outputs = self.make_io(plan, x2, y)
        ws.run(inputs, outputs)
        expected2 = reference(x2, w1, w2)
        np.testing.assert_array_equal(y, expected2)
        self.assertFalse(np.array_equal(expected1, expected2))
        np.testing.assert_array_equal(x2, f16(np.asarray(x1, dtype=np.float32)
                                              * 0.5 + 0.125))

    def test_separate_submit_mode_matches_reference(self):
        plan = self.build(mode="separate")
        self.assertEqual(len(plan.submits), 5)
        rng = np.random.default_rng(31)
        m, d, h = 13, 64, 32
        w1 = f16(rng.normal(0, 0.3, (d, h)))
        w2 = f16(rng.normal(0, 0.3, (h, d)))
        _cplan, ws = self.make_workspace(plan, {"W1": w1, "W2": w2})
        x = f16(rng.normal(0, 0.5, (m, d)))
        y = np.zeros((m, d), dtype=np.float16)
        inputs, outputs = self.make_io(plan, x, y)
        ws.run(inputs, outputs)
        np.testing.assert_array_equal(y, reference(x, w1, w2))

    def test_zero_weights_leave_the_residual_input_intact(self):
        """With zero weights, Y must equal X after both GEMMs and SiLU."""
        plan = self.build()
        m, d, h = 13, 64, 32
        w1 = f16(np.zeros((d, h), dtype=np.float32))
        w2 = f16(np.zeros((h, d), dtype=np.float32))
        _cplan, ws = self.make_workspace(plan, {"W1": w1, "W2": w2})
        rng = np.random.default_rng(19)
        for _ in range(2):
            x = f16(rng.normal(0, 0.75, (m, d)))
            y = np.zeros((m, d), dtype=np.float16)
            inputs, outputs = self.make_io(plan, x, y)
            ws.run(inputs, outputs)
            # Broadcast the independently computed zero-weight path, including
            # the explicit five narrowing points.
            expected = reference(x, w1, w2)
            np.testing.assert_array_equal(y, expected)
            np.testing.assert_array_equal(y, x)


if __name__ == "__main__":
    unittest.main()
