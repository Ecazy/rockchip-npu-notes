"""Scheduled, device-resident SmolLM-shaped SwiGLU FFN tests."""
import importlib.util
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                           RK_E_UNSUPPORTED, compile_rkir, compile_tilelang,
                           default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.ir import TensorType
from rknpu_backend.frontend.tir_utils import tilelang_available
from rknpu_backend.frontend import staged_launch
from rknpu_backend.plan import with_plan_hash
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]


def carrier_module(prim, *, options, profile_hash):
    return staged_launch.two_stage_plan_inputs(
        prim, options=options, profile_hash=profile_hash)[0]
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


def r0_host(h, r, wgu, wd):
    z = kchain(h, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = np.float32(1.0) / (np.float32(1.0) +
                              np.exp(-gate.astype(np.float32)))
    act = f16(gate.astype(np.float32) * sig * up.astype(np.float32))
    return f16(kchain(act, wd) + r)


def r1_fp16_points(h, r, wgu, wd):
    z = kchain(h, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
    silu = f16(gate * sig)
    act = f16(silu * up)
    return f16(kchain(act, wd) + r)


def lut_sigmoid(values):
    step = 32.0 / 2596.0
    grid = np.arange(-512, 513, dtype=np.float64) * step
    q = np.clip(np.rint(1.0 / (1.0 + np.exp(-grid)) * 32768.0), 1.0,
                32767.0)
    v = values.astype(np.float64)
    qi = np.interp(v, grid, q)
    edge = 512.0 * step
    qi = np.where(v > edge, q[-1] + 0.447 * (v - edge), qi)
    qi = np.where(v < -edge, q[0] - 0.447 * (-v - edge), qi)
    return f16((np.clip(qi, 1.0, 32767.0) + 1.0) / 32768.0)


def r2_declared_lut(h, r, wgu, wd):
    z = kchain(h, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    silu = f16(gate * lut_sigmoid(gate))
    act = f16(silu * up)
    return f16(kchain(act, wd) + r)


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
        module = carrier_module(self.fixture.make_ffn(m=M, tn=192),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops
                          if op.opcode != "view"],
                         ["matmul", "sigmoid", "mul", "mul",
                          "matmul", "add"])
        self.assertEqual(module.ops[1].attrs["col_offset"], 0)
        up_views = [op for op in module.ops
                    if op.opcode == "view" and op.attrs.get("col_offset") == H]
        self.assertTrue(up_views)
        self.assertEqual(module.ops[1].attrs["cols"], H)
        self.assertEqual(module.ops[0].attrs["source_k_loop"]["logical_k"], 576)
        self.assertTrue(any(op.opcode == "matmul" and op.attrs.get(
            "source_k_loop", {}).get("logical_k") == H
            for op in module.ops))
        non_view = [op for op in module.ops if op.opcode != "view"]
        self.assertTrue(all("source_stage_index" in op.attrs
                            for op in non_view))

    def test_plan_keeps_all_intermediates_device_resident(self):
        plan = self.compile(m=M, tn=192).plan
        self.assertEqual([b.name for b in plan.bindings
                          if not b.name.startswith("__identity_")],
                         ["H", "R", "Wgu", "Wd", "Y"])
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

    def _run(self, plan, h, r, wgu, wd, rounds=1):
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
        inputs = {b.name: b.id for b in plan.bindings if b.role == "input"}
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        previous = None
        for round_index in range(rounds):
            current_h = h if round_index == 0 else f16(
                h.astype(np.float32) * 0.75 + 0.125)
            current_r = r if round_index == 0 else f16(
                r.astype(np.float32) * 0.75 + 0.125)
            y = np.zeros((h.shape[0], D), dtype=np.float16)
            self._arrays.append(y)
            report = ws.run([
                make_binding(inputs["H"], current_h),
                make_binding(inputs["R"], current_r),
            ], [make_binding(out_id, y)])
            expected = r1_fp16_points(current_h, current_r, wgu, wd)
            np.testing.assert_array_equal(y, expected)
            if previous is not None:
                self.assertFalse(np.array_equal(previous, y))
            previous = y.copy()
        return report

    def test_h_and_r_are_independent_inputs_and_residual_is_not_h(self):
        plan = self.compile(m=M, tn=192).plan
        inputs = {b.name: b for b in plan.bindings if b.role == "input"}
        self.assertEqual(set(inputs), {"H", "R"})
        self.assertNotEqual(inputs["H"].id, inputs["R"].id)
        rng = np.random.default_rng(111)
        h = f16(rng.normal(0, 0.2, (M, D)))
        r = f16(rng.normal(0, 0.2, (M, D)))
        self.assertGreater(float(np.max(np.abs(h.astype(np.float32) -
                                            r.astype(np.float32)))), 0.1)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd0 = np.zeros((H, D), dtype=np.float16)
        report = self._run(plan, h, r, wgu, wd0, rounds=1)
        self.assertEqual(report.task_count, 81)

    def test_fixed_h_changing_r_only_changes_final_add(self):
        plan = self.compile(m=M, tn=192).plan
        rng = np.random.default_rng(112)
        h = f16(rng.normal(0, 0.2, (M, D)))
        r1 = f16(rng.normal(0, 0.2, (M, D)))
        r2 = f16(r1.astype(np.float32) + 0.5)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = f16(rng.normal(0, 0.02, (H, D)))
        z = kchain(h, wgu)
        gate = z[:, :H]; up = z[:, H:]
        sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
        act = f16(f16(gate * sig) * up)
        down = kchain(act, wd)
        y1 = f16(down + r1)
        y2 = f16(down + r2)
        self.assertTrue(np.array_equal(y1, r1_fp16_points(h, r1, wgu, wd)))
        self.assertTrue(np.array_equal(y2, r1_fp16_points(h, r2, wgu, wd)))
        self.assertFalse(np.array_equal(r1, r2))
        self.assertFalse(np.array_equal(y1, y2))

    def test_duplicate_h_r_input_value_is_rejected(self):
        module = carrier_module(self.fixture.make_ffn(m=M, tn=192),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        res = module.ops[-1]
        h_id = module.ops[0].inputs[0]
        bad_op = replace(res, inputs=(res.inputs[0], h_id))
        bad = replace(module, ops=module.ops[:-1] + (bad_op,))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(bad, profile=self.profile, options=self.options)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)

    def test_gate_up_views_are_distinct_halves_without_host_repack(self):
        module = carrier_module(self.fixture.make_ffn(m=M, tn=192),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual(module.ops[1].attrs["col_offset"], 0)
        self.assertTrue(any(op.opcode == "view" and
                            op.attrs.get("col_offset") == H
                            for op in module.ops))
        self.assertTrue(any(op.opcode == "view" and
                            op.attrs.get("col_offset") == 0
                            for op in module.ops))
        self.assertTrue(any(op.opcode == "view" and
                            op.attrs.get("col_offset") == H
                            for op in module.ops))
        plan = self.compile(m=M, tn=192).plan
        for task in plan.actions:
            if task.kind == "npu_task" and task.attrs.get("source_stage_index") in (1, 3):
                for key in ("gate_source", "up_source"):
                    if key in task.attrs:
                        view = task.attrs[key]["view"]
                        self.assertNotIn(view, [p.view_id for p in plan.publish])
                        self.assertFalse(any(a.kind == "upload" and view in a.outputs
                                             for a in plan.actions))

    def test_every_ffn_stage_names_its_producer_dependencies(self):
        plan = self.compile(m=M, tn=192).plan
        by_id = {a.id: a for a in plan.actions}

        def ancestors(aid):
            seen = set()
            stack = list(by_id[aid].deps)
            while stack:
                dep = stack.pop()
                if dep in seen:
                    continue
                seen.add(dep)
                stack.extend(by_id[dep].deps)
            return seen

        for task in plan.actions:
            if task.kind != "npu_task":
                continue
            for key in ("gate_source", "sig_source", "silu_source",
                        "up_source", "down_source", "residual_source"):
                source = task.attrs.get(key)
                if source is not None:
                    self.assertIn(source["action"], ancestors(task.id),
                                  (task.id, key))

    def test_r_dtype_and_shape_are_rejected(self):
        module = carrier_module(self.fixture.make_ffn(m=M, tn=192),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        r = module.value(1)
        bad_shape = replace(r, type=TensorType("float16", (M, D - 32)))
        bad_vals = tuple(bad_shape if v.id == r.id else v
                         for v in module.values)
        with self.assertRaises(BackendError):
            compile_rkir(replace(module, values=bad_vals),
                         profile=self.profile, options=self.options)

    def test_stage0_rounding_mismatch_is_rejected(self):
        base = self.fixture.make_ffn(m=M, tn=192)
        broken = with_stage_rounding(base, 0, "fp32_final_narrowing")
        self.assert_rejected(broken, RK_E_NUMERIC_CONTRACT,
                             "rounding declaration does not match")

    def test_mock_execution_matches_declared_reference(self):
        plan = self.compile(m=M, tn=192).plan
        rng = np.random.default_rng(101)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = f16(rng.normal(0, 0.02, (H, D)))
        h = f16(rng.normal(0, 0.2, (M, D)))
        r = f16(rng.normal(0, 0.2, (M, D)))
        report = self._run(plan, h, r, wgu, wd, rounds=2)
        self.assertEqual(report.task_count, 81)

    def test_zero_wd_returns_r_bitwise(self):
        plan = self.compile(m=M, tn=192).plan
        rng = np.random.default_rng(113)
        h = f16(rng.normal(0, 0.2, (M, D)))
        r = f16(rng.normal(0, 0.2, (M, D)))
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = np.zeros((H, D), dtype=np.float16)
        self._run(plan, h, r, wgu, wd, rounds=1)

    def test_mock_execution_m1(self):
        plan = self.compile(m=1, bm=1, tn=192).plan
        rng = np.random.default_rng(102)
        wgu = f16(rng.normal(0, 0.02, (D, 2 * H)))
        wd = f16(rng.normal(0, 0.02, (H, D)))
        h = f16(rng.normal(0, 0.2, (1, D)))
        r = f16(rng.normal(0, 0.2, (1, D)))
        report = self._run(plan, h, r, wgu, wd, rounds=2)
        self.assertEqual(report.task_count, 75)


if __name__ == "__main__":
    unittest.main()
