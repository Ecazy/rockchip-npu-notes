"""Scheduled, device-resident SmolLM-shaped SwiGLU FFN tests."""
import importlib.util
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                           RK_E_UNSUPPORTED, compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import extract, tilelang_available
from rknpu_backend.plan import with_plan_hash
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
M, D, H = 256, 576, 1536


def f16(x):
    return np.asarray(x, dtype=np.float16)


def kchain(x, w):
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    acc = f16(a[:, :256] @ b[:256, :])
    for k0 in range(256, a.shape[1], 256):
        k1 = min(a.shape[1], k0 + 256)
        acc = f16(acc.astype(np.float64) + a[:, k0:k1] @ b[k0:k1, :])
    return acc


def declared_ffn(x, wgu, wd):
    z = kchain(x, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
    silu = f16(gate * sig)
    act = f16(silu * up)
    down = kchain(act, wd)
    return f16(down + x), z, sig, silu, act, down


def with_stage_rounding(prim_func, index, value):
    from tvm import ir, tirx
    attrs = dict(prim_func.attrs)
    attrs[f"rknpu.stage{index}.source_k_rounding"] = value
    return tirx.PrimFunc(prim_func.params, prim_func.body,
                         prim_func.ret_type, prim_func.buffer_map,
                         ir.DictAttrs(attrs), prim_func.span)


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_ffn_scheduled", str(FIXTURES / "tilelang_ffn_scheduled.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class FFNScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()
        cls.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                     allow_approx_lut=True)
        cls.fixture = load_fixture()
        cls.lib = LibRKPlan(MOCK_LIB) if Path(MOCK_LIB).is_file() else None

    def setUp(self):
        self._arrays = []
        if self.lib is not None:
            bos = self.lib.mock_control("rk_mock_live_bos")
            res = self.lib.mock_control("rk_mock_resident")
            self.addCleanup(self.assert_no_leak, bos, res)

    def assert_no_leak(self, bos, res):
        self.assertEqual(self.lib.mock_control("rk_mock_untracked"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_tag_overflow"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), bos)
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), res)
        self._arrays.clear()

    def compile(self, **kwargs):
        return compile_tilelang(self.fixture.make_ffn(**kwargs),
                                profile=self.profile, options=self.options)

    def assert_rejected(self, prim_func, code, needle):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(prim_func, profile=self.profile,
                             options=self.options)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, code, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    def test_extracts_packed_gate_up_views_and_per_stage_schedules(self):
        module = extract(self.fixture.make_ffn(m=M, tn=192),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "view", "view", "sigmoid", "mul", "mul",
                          "matmul", "add"])
        self.assertEqual(module.ops[1].attrs["col_offset"], 0)
        self.assertEqual(module.ops[2].attrs["col_offset"], H)
        self.assertEqual(module.ops[1].attrs["cols"], H)
        self.assertEqual(module.ops[0].attrs["source_k_loop"]["logical_k"], 576)
        self.assertEqual(module.ops[6].attrs["source_k_loop"]["logical_k"], H)
        self.assertEqual(module.ops[3].attrs["source_rounding"],
                         "fp16_after_sigmoid")
        self.assertEqual(module.ops[5].attrs["source_rounding"],
                         "fp16_after_gate_mul")

    def test_plan_keeps_all_intermediates_device_resident(self):
        plan = self.compile(m=M, tn=192).plan
        self.assertEqual([b.name for b in plan.bindings
                          if not b.name.startswith("__identity_")],
                         ["X", "Wgu", "Wd", "Y"])
        self.assertEqual(plan.accounting.task_count, 81)
        self.assertEqual(plan.accounting.submit_count, 36)
        self.assertEqual(len(plan.publish), 3)
        stage1 = [a for a in plan.actions
                  if a.kind == "npu_task" and
                  a.attrs.get("source_stage_index") == 1]
        stage3 = [a for a in plan.actions
                  if a.kind == "npu_task" and
                  a.attrs.get("source_stage_index") == 3]
        stage4 = [a for a in plan.actions
                  if a.kind == "npu_task" and
                  a.attrs.get("source_stage_index") == 4]
        self.assertTrue(stage1 and stage3 and stage4)
        for task in stage1:
            self.assertIn("gate_source", task.attrs)
            gate = task.attrs["gate_source"]
            self.assertEqual(gate["half"], "gate")
        for task in stage3:
            self.assertIn("up_source", task.attrs)
            self.assertEqual(task.attrs["up_source"]["half"], "up")
        for task in stage4:
            sources = task.attrs.get("activation_sources")
            self.assertTrue(sources)
            for source in sources:
                self.assertIn(source["action"], task.deps)
        # Every intermediate buffer stays live through its last consumer.
        for action in plan.actions:
            if action.kind == "npu_task" and "activation_sources" in action.attrs:
                for source in action.attrs["activation_sources"]:
                    if source["view"] not in action.inputs:
                        continue
                    buf = plan.views[source["view"]].buffer_id
                    self.assertGreaterEqual(plan.buffers[buf].lifetime[1],
                                            action.id)

    def test_m1_decode_plan(self):
        plan = self.compile(m=1, bm=1, tn=192).plan
        self.assertEqual(plan.accounting.task_count, 75)
        self.assertEqual(plan.accounting.submit_count, 36)
        self.assertEqual(len(plan.publish), 3)

    def test_bad_gate_up_pairing_is_rejected(self):
        self.assert_rejected(self.fixture.bad_gate_up_swap(),
                             RK_E_UNSUPPORTED,
                             "up half at column H")

    def test_unaligned_down_tile_is_rejected(self):
        self.assert_rejected(self.fixture.make_ffn(m=M, tn=128),
                             RK_E_SHAPE, "does not cover")

    def test_stage4_rounding_mismatch_is_rejected(self):
        base = self.fixture.make_ffn(m=M, tn=192)
        broken = with_stage_rounding(base, 4, "fp32_final_narrowing")
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT,
                             "rounding declaration does not match")

    def test_c_verifier_rejects_missing_activation_dependency(self):
        if self.lib is None:
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        plan = self.compile(m=M, tn=192).plan
        task = next(a for a in plan.actions
                    if a.kind == "npu_task" and
                    a.attrs.get("source_stage_index") == 4)
        source = task.attrs["activation_sources"][0]
        broken_task = replace(task, deps=tuple(
            dep for dep in task.deps if dep != source["action"]))
        broken = with_plan_hash(replace(plan, actions=tuple(
            broken_task if a.id == task.id else a for a in plan.actions)))
        with self.assertRaises(BackendError) as ctx:
            self.lib.create_plan(broken)
        self.assertIn("dependency", ctx.exception.diagnostics[0].message)

    def _run(self, plan, x, wgu, wd, rounds=1):
        if self.lib is None:
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        prepared = []
        bindings = []
        for spec in plan.constants:
            if spec.name == "Wgu":
                arr = wgu
            elif spec.name == "Wd":
                arr = wd
            elif spec.generator == "identity":
                arr = np.eye(spec.shape[0], dtype=np.float16)
            else:
                raise AssertionError(spec.name)
            self._arrays.append(arr)
            prepared.append(arr)
            bindings.append(make_binding(spec.binding_id, arr))
        ws = self.lib.create_workspace(cplan, bindings, fd=0)
        self.addCleanup(ws.destroy)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        previous = None
        for round_index in range(rounds):
            current = x if round_index == 0 else f16(
                x.astype(np.float32) * 0.75 + 0.125)
            y = np.zeros((current.shape[0], D), dtype=np.float16)
            self._arrays.append(y)
            report = ws.run([make_binding(in_id, current)],
                            [make_binding(out_id, y)])
            expected = declared_ffn(current, wgu, wd)[0]
            np.testing.assert_array_equal(y, expected)
            if previous is not None:
                self.assertFalse(np.array_equal(previous, y))
            previous = y.copy()
        return report

    def test_mock_execution_matches_declared_reference(self):
        plan = self.compile(m=M, tn=192).plan
        rng = np.random.default_rng(101)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = f16(rng.normal(0, 0.02, (H, D)))
        x = f16(rng.normal(0, 0.2, (M, D)))
        report = self._run(plan, x, wgu, wd, rounds=2)
        self.assertEqual(report.task_count, 81)

    def test_mock_execution_m1(self):
        plan = self.compile(m=1, bm=1, tn=192).plan
        rng = np.random.default_rng(102)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = f16(rng.normal(0, 0.02, (H, D)))
        x = f16(rng.normal(0, 0.2, (1, D)))
        report = self._run(plan, x, wgu, wd, rounds=2)
        self.assertEqual(report.task_count, 75)


if __name__ == "__main__":
    unittest.main()
