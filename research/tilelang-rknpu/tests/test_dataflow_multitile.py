"""Numeric closure for per-node dataflow with a tile set per RKIR value."""
import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, Module, Op,
                           RK_E_SHAPE, TensorType, Value, activation_module,
                           branch_outputs_module, compile_rkir, default_profile,
                           projection_module, swiglu_module)
from rknpu_backend.plan import with_plan_hash
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def f16(x):
    return np.asarray(x, dtype=np.float16)


def kchain(x, w):
    """Weight is RKIR [N,K]; return the ordered 256-wide FP16 K-chain result."""
    a = x.astype(np.float64)
    b = w.T.astype(np.float64)
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
    proj = kchain(act, wd)
    return f16(proj.astype(np.float32) + x.astype(np.float32))


class MultiTileDataflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cls.lib = LibRKPlan(MOCK_LIB)
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

    def _binding(self, value_id, array):
        self._arrays.append(array)
        return make_binding(value_id, array)

    def _run(self, module, constants, inputs, mode="chain"):
        plan = compile_rkir(
            module, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True,
                                   submit_mode=mode)).plan
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        constant_bindings = []
        for spec in plan.constants:
            if spec.name.startswith("__identity_"):
                size = spec.shape[0]
                array = f16(np.eye(size, dtype=np.float32))
            else:
                array = constants[spec.name]
            constant_bindings.append(
                self._binding(spec.binding_id, array))
        ws = self.lib.create_workspace(cplan, constant_bindings, fd=0)
        self.addCleanup(ws.destroy)
        output_arrays = {}
        outputs = []
        for binding in plan.bindings:
            if binding.role != "output":
                continue
            array = np.zeros(binding.shape, dtype=np.float16)
            self._arrays.append(array)
            output_arrays[binding.name] = array
            outputs.append(self._binding(binding.id, array))
        inputs_c = [self._binding(b.id, inputs[b.name])
                    for b in plan.bindings if b.role == "input"]
        ws.run(inputs_c, outputs)
        return plan, output_arrays

    def test_projection_two_gemms_multi_tile(self):
        m = d = h = 300
        rng = np.random.default_rng(21)
        x = f16(rng.normal(0, 0.2, (m, d)))
        w1 = f16(rng.normal(0, 0.1, (h, d)))
        w2 = f16(rng.normal(0, 0.1, (d, h)))
        plan, out = self._run(
            projection_module(m=m, d=d, h=h), {"W1": w1, "W2": w2}, {"X": x})
        self.assertEqual(len(plan.publish), 4)  # 2x2 logical output tiles
        np.testing.assert_allclose(
            out["Y"].astype(np.float64),
            projection_reference(x, w1, w2).astype(np.float64),
            atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_ffn_multi_tile_keeps_branches_and_residual_live(self):
        m = d = h = 300
        rng = np.random.default_rng(22)
        x = f16(rng.normal(0, 0.2, (m, d)))
        wg = f16(rng.normal(0, 0.1, (h, d)))
        wu = f16(rng.normal(0, 0.1, (h, d)))
        wd = f16(rng.normal(0, 0.1, (d, h)))
        plan, out = self._run(
            swiglu_module(m=m, d=d, h=h),
            {"Wg": wg, "Wu": wu, "Wd": wd}, {"X": x})
        self.assertGreaterEqual(len(plan.publish), 4)
        np.testing.assert_allclose(
            out["Y"].astype(np.float64),
            ffn_reference(x, wg, wu, wd).astype(np.float64),
            atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_branch_outputs_multi_tile(self):
        m = d = 300
        n1, n2 = 300, 200
        rng = np.random.default_rng(23)
        x = f16(rng.normal(0, 0.2, (m, d)))
        w1 = f16(rng.normal(0, 0.1, (n1, d)))
        w2 = f16(rng.normal(0, 0.1, (n2, d)))
        plan, out = self._run(
            branch_outputs_module(m=m, k=d, n1=n1, n2=n2),
            {"W1": w1, "W2": w2}, {"X": x})
        self.assertEqual(set(out), {"Y1", "Y2"})
        np.testing.assert_array_equal(out["Y1"], kchain(x, w1))
        np.testing.assert_array_equal(out["Y2"], kchain(x, w2))
        self.assertEqual(sum(1 for b in plan.bindings if b.role == "output"), 2)

    def test_channel_broadcast_multi_tile_uses_channel_window(self):
        for opcode in ("add", "mul"):
            with self.subTest(opcode=opcode):
                m = k = n = 300
                rng = np.random.default_rng(24 + (opcode == "mul"))
                x = f16(rng.normal(0, 0.2, (m, k)))
                w = f16(rng.normal(0, 0.1, (n, k)))
                b = f16(np.linspace(-0.5, 0.5, n, dtype=np.float32))
                module = Module(
                    name=f"broadcast-{opcode}",
                    values=(
                        Value(0, "X", TensorType("float16", (m, k)), None,
                              "input"),
                        Value(1, "W", TensorType("float16", (n, k)), None,
                              "constant"),
                        Value(2, "B", TensorType("float16", (n,),
                                                 axes=("col",)), None,
                              "constant"),
                        Value(3, "Z", TensorType("float16", (m, n)), 0,
                              "temporary"),
                        Value(4, "Y", TensorType("float16", (m, n)), 1,
                              "output")),
                    ops=(Op(0, "matmul", (0, 1), (3,),
                            {"b_layout": "nk",
                             "numeric_contract": __import__(
                                 "rknpu_backend").NumericContract(
                                     accumulator_dtype="float16",
                                     accumulation="ordered_k_tiles_with_f16_writeback",
                                     narrowing_points=("k_tile_boundary",),
                                     approximate=False,
                                     tolerance={"atol": 8e-3, "rtol": 8e-3}
                                 ).to_dict()}),
                         Op(1, opcode, (3, 2), (4,), {})),
                    inputs=(0,), outputs=(4,), source_hash="test")
                _, out = self._run(module, {"W": w, "B": b}, {"X": x})
                z = kchain(x, w)
                expected = (f16(z.astype(np.float32) + b.astype(np.float32))
                            if opcode == "add" else
                            f16(z.astype(np.float32) * b.astype(np.float32)))
                np.testing.assert_array_equal(out["Y"], expected)

    def test_channel_broadcast_window_outside_source_is_rejected(self):
        module = Module(
            name="bad-broadcast-window",
            values=(
                Value(0, "X", TensorType("float16", (300, 300)), None,
                      "input"),
                Value(1, "W", TensorType("float16", (300, 300)), None,
                      "constant"),
                Value(2, "B", TensorType("float16", (300,), axes=("col",)),
                      None, "constant"),
                Value(3, "Z", TensorType("float16", (300, 300)), 0,
                      "temporary"),
                Value(4, "Y", TensorType("float16", (300, 300)), 1,
                      "output")),
            ops=(Op(0, "matmul", (0, 1), (3,),
                    {"b_layout": "nk",
                     "numeric_contract": __import__(
                         "rknpu_backend").NumericContract(
                             accumulator_dtype="float16",
                             accumulation="ordered_k_tiles_with_f16_writeback",
                             narrowing_points=("k_tile_boundary",),
                             approximate=False,
                             tolerance={"atol": 8e-3, "rtol": 8e-3}
                         ).to_dict()}),
                 Op(1, "add", (3, 2), (4,), {})),
            inputs=(0,), outputs=(4,), source_hash="test")
        plan = compile_rkir(
            module, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        actions = list(plan.actions)
        idx = next(i for i, a in enumerate(actions)
                   if a.kind == "upload" and
                   a.attrs.get("broadcast") == "channel" and
                   a.attrs.get("src_col_offset", 0) > 0)
        actions[idx] = replace(actions[idx],
                               attrs={**actions[idx].attrs,
                                      "src_col_offset": 300})
        broken = with_plan_hash(replace(plan, actions=tuple(actions),
                                        plan_hash=""))
        with self.assertRaises(BackendError) as ctx:
            self.lib.create_plan(broken)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_activation_only_multi_tile(self):
        m = n = 300
        rng = np.random.default_rng(26)
        x = f16(rng.normal(0, 0.4, (m, n)))
        _, out = self._run(activation_module(m=m, n=n), {}, {"X": x})
        sig = f16(1.0 / (1.0 + np.exp(-x.astype(np.float32))))
        expected = f16(x.astype(np.float32) * sig.astype(np.float32))
        np.testing.assert_allclose(out["Y"].astype(np.float64),
                                   expected.astype(np.float64),
                                   atol=8e-3, rtol=8e-3, equal_nan=False)


if __name__ == "__main__":
    unittest.main()
