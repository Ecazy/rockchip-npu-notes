# SPDX-License-Identifier: GPL-3.0-or-later
"""Weight-stationary order and the CBUF weight-reuse bit.

The reuse bit means "the previous task on this core already loaded this operand",
so it may only be set where the previous task of the same submit uses the same
weight view at the same K index.  The planner can now emit such an order; both
layers check the rule, and the emitted command word is where the bit lands.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile, load_plan, save_plan
from rknpu_backend.errors import BackendError
from rknpu_backend.models import PlanCall
from rknpu_backend.plan import _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
MOCK = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
REAL = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
SHAPE = (256, 576, 512)


def options(**extra):
    base = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    base.update(extra)
    return CompileOptions(**base)


def build(task_order, geometry=None):
    geometry = geometry or {"bm": 128, "bn": 256}
    return compile_tilelang(make_linear(*SHAPE, **geometry), profile=PROFILE,
                            options=options(linear_task_order=task_order)).plan


def tasks_of(plan):
    return [a for a in plan.actions if a.kind == "npu_task"]


class WeightStationaryOrderTests(unittest.TestCase):
    def test_the_order_groups_m_tiles_for_one_weight_tile(self):
        plan = build("weight_stationary")
        tasks = tasks_of(plan)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len(plan.submits), 1)
        # (k0, n0), (k0, n0) -- the two M tiles -- then (k0, n256), ...
        self.assertEqual([t.attrs["k_index"] for t in tasks[:4]], [0, 0, 0, 0])
        self.assertEqual(sum(1 for t in tasks if t.attrs.get("weight_reuse")), 0)

    def test_the_reuse_variant_flags_exactly_the_second_m_tile(self):
        plan = build("weight_stationary_reuse")
        tasks = tasks_of(plan)
        previous = None
        expected = 0
        for task in tasks:
            if previous is not None and previous.inputs[1] == task.inputs[1] \
                    and previous.attrs.get("k_index") == task.attrs.get("k_index"):
                expected += 1
                self.assertTrue(task.attrs.get("weight_reuse"))
            else:
                self.assertFalse(task.attrs.get("weight_reuse"))
            previous = task
        self.assertEqual(expected, 6)
        self.assertEqual(sum(1 for t in tasks if t.attrs.get("weight_reuse")), 6)
        self.assertEqual(sum(1 for t in tasks if t.attrs.get("data_reuse")), 0)

    def test_merged_keeps_k_innermost_and_no_reuse(self):
        plan = build("merged")
        tasks = tasks_of(plan)
        self.assertEqual([t.attrs["k_index"] for t in tasks[:3]], [0, 1, 2])
        self.assertEqual(sum(1 for t in tasks if t.attrs.get("weight_reuse")), 0)

    def test_flags_survive_save_and_load(self):
        import tempfile

        plan = build("weight_stationary_reuse")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(sum(1 for t in tasks_of(loaded)
                             if t.attrs.get("weight_reuse")), 6)


@unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
class WeightReuseVerifierTests(unittest.TestCase):
    def _rebuild(self, plan, actions):
        """Attach new actions without running the Python validator again."""
        tampered = replace(plan, plan_hash="")
        object.__setattr__(tampered, "actions", tuple(actions))
        object.__setattr__(tampered, "plan_hash",
                           sha256_json(tampered.to_dict(include_hash=False)))
        return tampered

    def _tamper(self, plan, change):
        actions = list(plan.actions)
        index = next(i for i, a in enumerate(actions)
                     if a.kind == "npu_task" and a.attrs.get("weight_reuse"))
        actions[index] = change(actions[index])
        return self._rebuild(plan, actions)

    def test_both_layers_accept_the_planner_output(self):
        plan = build("weight_stationary_reuse")
        _validate_plan(plan)
        lib = LibRKPlan(REAL)
        cplan = lib.create_plan(plan)
        cplan.destroy()

    def test_first_task_reuse_is_refused(self):
        plan = build("weight_stationary_reuse")
        actions = list(plan.actions)
        anchor = next(a for a in actions
                      if a.kind == "npu_task" and a.attrs.get("weight_reuse"))
        first = next(a for a in actions if a.kind == "npu_task")
        actions[first.id] = replace(
            first, attrs={**dict(first.attrs), "weight_reuse": True})
        tampered = self._rebuild(plan, actions)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(tampered)
        self.assertIn("first task of a submit cannot reuse",
                      caught.exception.diagnostics[0].message)
        lib = LibRKPlan(REAL)
        with self.assertRaises(BackendError) as caught:
            lib.create_plan(tampered)
        self.assertIn("reuse", caught.exception.diagnostics[0].message)

    def test_a_different_weight_view_is_refused(self):
        plan = build("weight_stationary_reuse")

        def change(task):
            other = next(t for t in tasks_of(plan)
                         if t.id != task.id and t.inputs[1] != task.inputs[1])
            return replace(task, inputs=(task.inputs[0], other.inputs[1],
                                         *task.inputs[2:]))

        tampered = self._tamper(plan, change)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(tampered)
        self.assertIn("same weight view", caught.exception.diagnostics[0].message)
        lib = LibRKPlan(REAL)
        with self.assertRaises(BackendError) as caught:
            lib.create_plan(tampered)
        self.assertIn("weight_reuse", caught.exception.diagnostics[0].message)


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class WeightReuseNumericTests(unittest.TestCase):
    def _run(self, plan):
        rng = np.random.default_rng(20261103)
        m, k, n = SHAPE
        a = np.ascontiguousarray(rng.standard_normal((m, k)).astype(np.float16))
        b = np.ascontiguousarray(rng.standard_normal((k, n)).astype(np.float16))
        constants = {"B": b}
        for spec in plan.constants:
            if spec.name in constants:
                continue
            constants[spec.name] = (np.eye(spec.shape[0], dtype=np.float16)
                                    if spec.generator == "identity"
                                    else np.zeros(tuple(spec.shape), dtype=np.float16))
        lib = LibRKPlan(MOCK)
        out = [np.zeros((m, n), dtype=np.float16)]
        call = PlanCall(lib, plan, constants, fd=0)
        try:
            call.run_into([a], out)
        finally:
            call.close()
        return out[0]

    def test_the_bit_does_not_change_the_result(self):
        plain = self._run(build("weight_stationary"))
        reuse = self._run(build("weight_stationary_reuse"))
        baseline = self._run(build("merged"))
        self.assertTrue(np.array_equal(plain, reuse))
        self.assertTrue(np.array_equal(plain, baseline))
        self.assertTrue(np.isfinite(plain).all())


if __name__ == "__main__":
    unittest.main()
