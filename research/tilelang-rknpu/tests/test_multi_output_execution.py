"""Multiple exported SSA values through the planner and mock runtime.

Two independent ``X @ W_i^T`` nodes share one input upload and export two output
bindings.  This exercises per-output download/publish mapping and verifies that
neither output view is reclaimed before its publish.
"""
import ctypes
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, compile_tilelang, default_profile,
                           matmul_module)
from rknpu_backend.frontend.tilelang_adapter import tilelang_available
from rknpu_backend.runtime.ffi import LibRKPlan, RKBinding

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def load_mock():
    if not Path(MOCK_LIB).is_file():
        raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
    return LibRKPlan(MOCK_LIB)


def f16(x):
    return np.asarray(x, dtype=np.float16)


def branch_module(*, m=13, k=63, n1=33, n2=47):
    contract = matmul_module(m=m, k=k, n=n1, b_layout="nk").ops[0].attrs[
        "numeric_contract"]
    attrs = {"b_layout": "nk", "numeric_contract": contract}
    values = (
        Value(0, "X", TensorType("float16", (m, k)), None, "input"),
        Value(1, "W1", TensorType("float16", (n1, k)), None, "constant"),
        Value(2, "W2", TensorType("float16", (n2, k)), None, "constant"),
        Value(3, "Y1", TensorType("float16", (m, n1)), 0, "output"),
        Value(4, "Y2", TensorType("float16", (m, n2)), 1, "output"),
    )
    ops = (Op(0, "matmul", (0, 1), (3,), dict(attrs)),
           Op(1, "matmul", (0, 2), (4,), dict(attrs)))
    return Module(name="branch-output-exec", values=values, ops=ops,
                  inputs=(0,), outputs=(3, 4), source_hash="test")


def projection_two_outputs_module(*, m=11, d=64, h=31):
    """Export the pre-activation Z *and* the final projection P.

    Z is consumed by the SiLU node after it is produced, so this pins the rule
    that an exported intermediate stays live until both its last device consumer
    and its own publish.
    """
    contract = matmul_module(m=m, k=d, n=h, b_layout="nk").ops[0].attrs[
        "numeric_contract"]
    attrs = {"b_layout": "nk", "numeric_contract": contract}
    values = (
        Value(0, "X", TensorType("float16", (m, d)), None, "input"),
        Value(1, "W1", TensorType("float16", (h, d)), None, "constant"),
        Value(2, "Z", TensorType("float16", (m, h)), 0, "output"),
        Value(3, "S", TensorType("float16", (m, h)), 1, "temporary"),
        Value(4, "W2", TensorType("float16", (d, h)), None, "constant"),
        Value(5, "P", TensorType("float16", (m, d)), 2, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 1), (2,), dict(attrs)),
        Op(1, "silu", (2,), (3,)),
        Op(2, "matmul", (3, 4), (5,), dict(attrs)),
    )
    return Module(name="projection-two-output-exec", values=values, ops=ops,
                  inputs=(0,), outputs=(2, 5), source_hash="test")


def load_branch_fixture():
    fixtures = ROOT / "tests" / "fixtures"
    if str(fixtures) not in sys.path:
        sys.path.insert(0, str(fixtures))
    spec = importlib.util.spec_from_file_location(
        "tl_multi_output_exec", str(fixtures / "tilelang_multi_output.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_binding(value_id, array, holder):
    holder.append(array)
    binding = RKBinding()
    binding.abi_version = 1
    binding.struct_size = ctypes.sizeof(RKBinding)
    binding.id = value_id
    binding.dtype = 0
    binding.rank = 2
    binding.shape = (array.shape[0], array.shape[1])
    binding.data = array.ctypes.data_as(ctypes.c_void_p)
    binding.nbytes = array.nbytes
    return binding


class MultiOutputExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_mock()
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

    def test_two_outputs_match_reference_across_input_rounds(self):
        module = branch_module()
        plan = compile_rkir(
            module, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        m, k, n1, n2 = 13, 63, 33, 47
        rng = np.random.default_rng(23)
        w1 = f16(rng.normal(0, 0.25, (n1, k)))
        w2 = f16(rng.normal(0, 0.25, (n2, k)))
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        constants = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            array = {"W1": w1, "W2": w2}[binding.name]
            constants.append(make_binding(binding.id, array, self._arrays))
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)

        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_ids = [b.id for b in plan.bindings if b.role == "output"]
        self.assertEqual(len(out_ids), 2)
        for _round in range(2):
            x = f16(rng.normal(0, 0.5, (m, k)))
            y1 = np.zeros((m, n1), dtype=np.float16)
            y2 = np.zeros((m, n2), dtype=np.float16)
            inputs = [make_binding(in_id, x, self._arrays)]
            outputs = [make_binding(out_ids[0], y1, self._arrays),
                       make_binding(out_ids[1], y2, self._arrays)]
            report = ws.run(inputs, outputs)
            self.assertEqual(report.task_count, 2)
            np.testing.assert_array_equal(
                y1, f16(np.matmul(x.astype(np.float64),
                                  w1.astype(np.float64).T)))
            np.testing.assert_array_equal(
                y2, f16(np.matmul(x.astype(np.float64),
                                  w2.astype(np.float64).T)))

    def test_exported_intermediate_stays_live_through_later_consumer(self):
        module = projection_two_outputs_module()
        plan = compile_rkir(
            module, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True)).plan
        m, d, h = 11, 64, 31
        rng = np.random.default_rng(29)
        w1 = f16(rng.normal(0, 0.25, (h, d)))
        w2 = f16(rng.normal(0, 0.25, (d, h)))
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        constants = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                array = f16(np.eye(binding.shape[0], dtype=np.float32))
            else:
                array = {"W1": w1, "W2": w2}[binding.name]
            constants.append(make_binding(binding.id, array, self._arrays))
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)

        x = f16(rng.normal(0, 0.5, (m, d)))
        z = np.zeros((m, h), dtype=np.float16)
        p = np.zeros((m, d), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_ids = [b.id for b in plan.bindings if b.role == "output"]
        outputs = [make_binding(out_ids[0], z, self._arrays),
                   make_binding(out_ids[1], p, self._arrays)]
        ws.run([make_binding(in_id, x, self._arrays)], outputs)

        z_ref = f16(np.matmul(x.astype(np.float64), w1.astype(np.float64).T))
        g = f16(1.0 / (1.0 + np.exp(-z_ref.astype(np.float32))))
        s = f16(z_ref.astype(np.float32) * g.astype(np.float32))
        p_ref = f16(np.matmul(s.astype(np.float64), w2.astype(np.float64).T))
        np.testing.assert_array_equal(z, z_ref)
        np.testing.assert_array_equal(p, p_ref)

        z_binding_id = next(b.id for b in plan.bindings
                            if b.role == "output" and b.name == "Z")
        z_view = next(a.inputs[0] for a in plan.actions
                      if a.kind == "download" and
                      a.attrs.get("binding_id") == z_binding_id)
        z_buffer = plan.buffers[plan.views[z_view].buffer_id]
        silu_tasks = [a for a in plan.actions if a.kind == "npu_task" and
                      a.template_id == "co.ew_mul.f16.v1"]
        self.assertGreaterEqual(z_buffer.lifetime[1], silu_tasks[0].id)

    @unittest.skipUnless(tilelang_available(),
                         "pinned TileLang optional dependency absent")
    def test_real_tilelang_two_output_frontend_to_mock(self):
        fixture = load_branch_fixture()
        m, k, n1, n2 = 8, 64, 33, 47
        plan = compile_tilelang(
            fixture.branch_outputs(m=m, k=k, n1=n1, n2=n2),
            profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        self.assertEqual(sum(a.kind == "download" for a in plan.actions), 2)
        self.assertEqual(len(plan.publish), 2)

        rng = np.random.default_rng(53)
        x = f16(rng.normal(0, 0.4, (m, k)))
        w1 = f16(rng.normal(0, 0.2, (k, n1)))
        w2 = f16(rng.normal(0, 0.2, (k, n2)))
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        constants = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            array = {"W1": w1, "W2": w2}[binding.name]
            constants.append(make_binding(binding.id, array, self._arrays))
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)

        y1 = np.zeros((m, n1), dtype=np.float16)
        y2 = np.zeros((m, n2), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_ids = [b.id for b in plan.bindings if b.role == "output"]
        outputs = [make_binding(out_ids[0], y1, self._arrays),
                   make_binding(out_ids[1], y2, self._arrays)]
        ws.run([make_binding(in_id, x, self._arrays)], outputs)
        np.testing.assert_array_equal(
            y1, f16(np.matmul(x.astype(np.float64), w1.astype(np.float64))))
        np.testing.assert_array_equal(
            y2, f16(np.matmul(x.astype(np.float64), w2.astype(np.float64))))


if __name__ == "__main__":
    unittest.main()
