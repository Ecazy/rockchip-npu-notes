"""Mock numeric closure for compile-time scalar / per-channel broadcast."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError, RK_E_NUMERIC_CONTRACT
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

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
        "tl_broadcast_exec", str(FIXTURES / "tilelang_broadcast.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f16(x):
    return np.asarray(x, dtype=np.float16)


def reference(x, w, bias, op):
    """MatMul in the mock's double accumulator, then two explicit FP16 stores."""
    z16 = f16(np.matmul(x.astype(np.float64), w.astype(np.float64)))
    if op == "add":
        return f16(z16.astype(np.float32) + bias.astype(np.float32))
    return f16(z16.astype(np.float32) * bias.astype(np.float32))


class BroadcastExecutionTests(unittest.TestCase):
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

    def build(self, factory):
        return compile_tilelang(
            factory(m=8, k=64, n=33), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan

    def constant_map(self, plan, w, bias):
        out = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                array = f16(np.eye(binding.shape[0], dtype=np.float32))
            elif binding.name == "W":
                array = w
            else:
                array = bias
            self._arrays.append(array)  # keep ctypes data pointers alive
            out.append(make_binding(binding.id, array))
        return out

    def test_scalar_and_channel_tiles_survive_two_input_rounds(self):
        cases = (
            (self.fixture.scalar_add, "add", f16(np.asarray(0.25))),
            (self.fixture.channel_mul, "mul",
             f16(np.linspace(-0.5, 0.5, 33, dtype=np.float32))),
        )
        for factory, op, bias in cases:
            with self.subTest(op=op):
                plan = self.build(factory)
                self.assertGreater(plan.accounting.broadcast_expand_bytes, 0)
                cplan = self.lib.create_plan(plan)
                self.addCleanup(cplan.destroy)
                rng = np.random.default_rng(61)
                w = f16(rng.normal(0, 0.2, (64, 33)))
                constants = self.constant_map(plan, w, bias)
                ws = self.lib.create_workspace(cplan, constants)
                self.addCleanup(ws.destroy)
                in_id = next(b.id for b in plan.bindings if b.role == "input")
                out_id = next(b.id for b in plan.bindings if b.role == "output")
                previous = None
                for _round in range(2):
                    x = f16(rng.normal(0, 0.3, (8, 64)))
                    y = np.zeros((8, 33), dtype=np.float16)
                    report = ws.run([make_binding(in_id, x)],
                                    [make_binding(out_id, y)])
                    expected = reference(x, w, bias, op)
                    np.testing.assert_array_equal(y, expected)
                    # Constants, including the expanded broadcast tile, are
                    # uploaded once at workspace creation; only X is per-run.
                    self.assertEqual(report.upload_calls, 1)
                    self.assertEqual(report.download_calls, 1)
                    if previous is not None:
                        self.assertFalse(np.array_equal(previous, y))
                    previous = y.copy()

    def test_non_finite_broadcast_constant_is_rejected_at_create(self):
        plan = self.build(self.fixture.channel_mul)
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        rng = np.random.default_rng(67)
        w = f16(rng.normal(0, 0.2, (64, 33)))
        bias = f16(np.zeros((33,), dtype=np.float32))
        bias[7] = np.float16(np.nan)
        with self.assertRaises(BackendError) as ctx:
            ws = self.lib.create_workspace(
                cplan, self.constant_map(plan, w, bias))
            self.addCleanup(ws.destroy)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)


if __name__ == "__main__":
    unittest.main()
