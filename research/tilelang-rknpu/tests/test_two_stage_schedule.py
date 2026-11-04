"""Restricted two-stage, schedule-aware device-resident execution tests."""
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


def with_global_schedule_attr(prim_func):
    from tvm import ir, tirx
    attrs = dict(prim_func.attrs)
    attrs["rknpu.tile_m"] = 64
    return tirx.PrimFunc(prim_func.params, prim_func.body,
                         prim_func.ret_type, prim_func.buffer_map,
                         ir.DictAttrs(attrs), prim_func.span)


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_two_stage", str(FIXTURES / "tilelang_two_stage.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class TwoStageScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()
        cls.options = CompileOptions(numeric_mode="rk_f16_tiled")
        cls.fixture = load_fixture()
        cls.lib = (LibRKPlan(MOCK_LIB) if Path(MOCK_LIB).is_file() else None)

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
        return compile_tilelang(
            self.fixture.make_two_stage(**kwargs),
            profile=self.profile, options=self.options)

    def assert_rejected(self, prim_func, code, needle):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(prim_func, profile=self.profile,
                             options=self.options)
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, code, diagnostic.message)
        self.assertIn(needle, diagnostic.message)
        return diagnostic

    def test_extracts_two_launch_stages_with_independent_schedules(self):
        module = carrier_module(self.fixture.make_two_stage(bm0=65, bm1=65),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops], ["matmul", "matmul"])
        self.assertEqual([v.role for v in module.values],
                         ["input", "constant", "constant", "output",
                          "temporary"])
        s0, s1 = module.ops[0].attrs["source_schedule"],             module.ops[1].attrs["source_schedule"]
        self.assertNotEqual(s0["tile_n"], s1["tile_n"])
        self.assertNotEqual(s0["order"], s1["order"])
        self.assertEqual(module.ops[0].attrs["source_k_loop"]["logical_k"], 576)
        self.assertEqual(module.ops[1].attrs["source_k_loop"]["logical_k"], 576)
        self.assertEqual(module.ops[0].attrs["source_stage_boundary"],
                         "kernel_launch")

    def test_plan_keeps_z_device_resident_and_orders_stages(self):
        plan = self.compile(bm0=65, bm1=65).plan
        self.assertEqual([b.name for b in plan.bindings],
                         ["X", "W1", "W2", "Y"])
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        stages = [a.attrs["source_stage_index"] for a in tasks]
        self.assertEqual(stages, sorted(stages))
        self.assertIn(0, stages)
        self.assertIn(1, stages)
        self.assertEqual(plan.accounting.task_count, 15)
        self.assertEqual(plan.accounting.submit_count, 5)
        self.assertEqual(len(plan.publish), 2)

        z_actions = [a for a in tasks if a.attrs.get("z_source")]
        self.assertTrue(z_actions)
        z_buffers = {plan.views[a.attrs["z_source"]["producer_view"]].buffer_id
                     for a in z_actions}
        self.assertTrue(z_buffers)
        for action in plan.actions:
            if action.kind == "upload":
                for view_id in action.outputs:
                    self.assertNotIn(plan.views[view_id].buffer_id, z_buffers)
            if action.kind == "download":
                for view_id in action.inputs:
                    self.assertNotIn(plan.views[view_id].buffer_id, z_buffers)
        for action in z_actions:
            self.assertIn(action.attrs["z_source"]["producer_action"],
                          action.deps)
            buffer = plan.buffers[plan.views[
                action.attrs["z_source"]["producer_view"]].buffer_id]
            self.assertGreaterEqual(buffer.lifetime[1], action.id)
        start = next(a for a in tasks if a.attrs["source_stage_index"] == 1)
        self.assertIn("z_source", start.attrs)
        self.assertEqual(start.attrs["z_source"]["logical_k"], 256)

    def test_different_source_m_tiles_are_retiled_for_z_view(self):
        module = carrier_module(self.fixture.make_two_stage(
            m=256, src_bm0=128, bm0=64, src_bm1=64, bm1=64),
            options=self.options, profile_hash=self.profile.profile_hash)
        self.assertTrue(any(op.opcode == "matmul" for op in module.ops))
        plan = self.compile(m=256, src_bm0=128, bm0=64,
                            src_bm1=64, bm1=64).plan
        self.assertEqual(plan.accounting.task_count, 60)
        z_sources = [a.attrs["z_source"] for a in plan.actions
                     if a.kind == "npu_task" and a.attrs.get("z_source")]
        self.assertTrue(z_sources)
        self.assertTrue(all(z["row_offset"] == 0 for z in z_sources))

    def test_swapped_stage_grids_still_map_exactly(self):
        canonical = self.compile(bm0=65, bm1=65).plan
        swapped = self.compile(bm0=65, bm1=65, swapped0=True,
                               swapped1=True).plan
        self.assertEqual(canonical.accounting.task_count,
                         swapped.accounting.task_count)
        self.assertEqual([(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in canonical.publish],
                         [(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in swapped.publish])

    def test_unsupported_h257_tail_is_rejected(self):
        self.assert_rejected(self.fixture.make_two_stage(
            m=65, k=576, h=257, n=97, bm0=65, bm1=65),
            RK_E_UNSUPPORTED, "32-element aligned")

    def test_stage0_n_tile_must_be_256(self):
        self.assert_rejected(self.fixture.make_two_stage(
            bm0=65, bm1=65, bn0=128), RK_E_UNSUPPORTED, "tile_n must be 256")

    def test_stage1_m_tile_cannot_cross_producer_tile(self):
        self.assert_rejected(self.fixture.make_two_stage(
            m=65, bm0=128, bm1=64), RK_E_UNSUPPORTED,
            "applied stage0/stage1 M tile must match")

    def test_applied_row_subview_is_rejected_without_repack(self):
        self.assert_rejected(self.fixture.make_two_stage(
            m=256, bm0=128, bm1=64), RK_E_UNSUPPORTED,
            "applied stage0/stage1 M tile must match")

    def test_stage1_schedule_rounding_mismatch_is_rejected(self):
        self.assert_rejected(self.fixture.make_two_stage(
            bad_rounding1=True), RK_E_NUMERIC_CONTRACT,
            "rounding declaration does not match")

    def test_global_schedule_attribute_cannot_be_copied_to_all_stages(self):
        base = self.fixture.make_two_stage(bm0=65, bm1=65)
        self.assert_rejected(with_global_schedule_attr(base), RK_E_UNSUPPORTED,
                             "single global schedule attribute")

    def test_equal_m_and_n_grids_still_use_the_output_m_axis(self):
        plan = self.compile(m=256, bm0=128, bm1=128).plan
        self.assertEqual(plan.accounting.task_count, 30)

    def test_reorder_groups_are_closed_and_launch_local(self):
        from rknpu_backend.frontend import staged_launch
        from rknpu_backend import CompileOptions

        prim = self.fixture.make_two_stage(m=512, bm0=128, bm1=128,
                                           order0="n_major")
        chain = staged_launch.two_stage_chain(
            prim, options=CompileOptions(numeric_mode="rk_f16_tiled"),
            schedules=staged_launch.schedules_from_attrs(prim, 2))
        proof = chain["stages"][0]["reorder_proof"]
        self.assertEqual(proof["groups"], 12)
        self.assertTrue(proof["store_regions_disjoint"])
        self.assertTrue(proof["local_values_group_local"])
        self.assertTrue(proof["launch_local"])
        self.assertTrue({"As", "Bs", "Cl"}.issubset(proof["compute_reads"]))
        self.assertIn("Cl", proof["compute_writes"])

    def test_declared_mn_order_changes_task_traversal(self):
        from rknpu_backend.frontend import staged_launch
        from rknpu_backend import CompileOptions

        def traversal(order):
            prim = self.fixture.make_two_stage(m=512, bm0=128, bm1=128,
                                               order0=order)
            chain = staged_launch.two_stage_chain(
                prim, options=CompileOptions(numeric_mode="rk_f16_tiled"),
                schedules=staged_launch.schedules_from_attrs(prim, 2))
            return [(task["m0"], task["n0"])
                    for task in chain["stages"][0]["tasks"]
                    if task.get("kind") == "gemm"][:12]

        self.assertNotEqual(traversal("m_major"), traversal("n_major"))
        self.assert_rejected(
            self.fixture.make_two_stage(m=512, bm0=128, bm1=128,
                                        order0="bogus"),
            RK_E_UNSUPPORTED, "unknown staged mn_order")

    def test_stage_outside_source_is_rejected(self):
        self.assert_rejected(self.fixture.make_two_stage(
            extra_stage_attr=True), RK_E_UNSUPPORTED,
            "outside the source")

    def test_missing_intermediate_is_rejected(self):
        self.assert_rejected(self.fixture.two_stage_bad_no_intermediate(),
                             RK_E_UNSUPPORTED, "one distinct intermediate")

    def _run(self, plan, x, w1, w2, rounds=1):
        if self.lib is None:
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        values = {"W1": w1, "W2": w2}
        constants = []
        for spec in plan.constants:
            self._arrays.append(values[spec.name])
            constants.append(make_binding(spec.binding_id, values[spec.name]))
        ws = self.lib.create_workspace(cplan, constants, fd=0)
        self.addCleanup(ws.destroy)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        previous = None
        for round_index in range(rounds):
            current = x if round_index == 0 else f16(
                x.astype(np.float32) * 0.75 + 0.125)
            y = np.zeros((x.shape[0], w2.shape[1]), dtype=np.float16)
            self._arrays.append(y)
            report = ws.run([make_binding(in_id, current)],
                            [make_binding(out_id, y)])
            expected = kchain(kchain(current, w1), w2)
            np.testing.assert_array_equal(y, expected)
            if previous is not None:
                self.assertFalse(np.array_equal(previous, y))
            previous = y.copy()
        return report

    def test_c_verifier_rejects_missing_z_producer_dependency(self):
        if self.lib is None:
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        plan = self.compile(bm0=65, bm1=65).plan
        task = next(a for a in plan.actions
                    if a.kind == "npu_task" and a.attrs.get("z_source"))
        producer = task.attrs["z_source"]["producer_action"]
        broken_task = replace(task, deps=tuple(
            dep for dep in task.deps if dep != producer))
        broken = with_plan_hash(replace(plan, actions=tuple(
            broken_task if a.id == task.id else a for a in plan.actions)))
        with self.assertRaises(BackendError) as ctx:
            self.lib.create_plan(broken)
        self.assertIn("dependency", ctx.exception.diagnostics[0].message)

    def test_mock_execution_matches_independent_two_stage_oracle(self):
        plan = self.compile(bm0=65, bm1=65).plan
        rng = np.random.default_rng(71)
        w1 = f16(rng.normal(0, 0.1, (576, 576)))
        w2 = f16(rng.normal(0, 0.1, (576, 97)))
        x = f16(rng.normal(0, 0.2, (65, 576)))
        report = self._run(plan, x, w1, w2, rounds=2)
        self.assertEqual(report.task_count, 15)
        self.assertEqual(report.submit_count, 5)

    def test_mock_execution_with_different_source_m_tiles(self):
        plan = self.compile(m=256, src_bm0=128, bm0=64,
                            src_bm1=64, bm1=64).plan
        rng = np.random.default_rng(72)
        w1 = f16(rng.normal(0, 0.1, (576, 576)))
        w2 = f16(rng.normal(0, 0.1, (576, 97)))
        x = f16(rng.normal(0, 0.2, (256, 576)))
        self._run(plan, x, w1, w2, rounds=2)


if __name__ == "__main__":
    unittest.main()
