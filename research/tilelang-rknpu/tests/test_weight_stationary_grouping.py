# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded N-tile grouping for the weight-stationary order on the full gate/up.

Whole N tiles are packed into one submit while the group stays inside the
64-task bound, so an output tile's K chain never crosses a submit and every
submit starts with reuse off.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import CompileOptions, default_profile, compile_tilelang
from rknpu_backend.errors import BackendError
from rknpu_backend.plan import _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
REAL = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
FULL = (256, 576, 3072)


def build(shape, bm, order, bn=256):
    return compile_tilelang(make_linear(*shape, bm=bm, bn=bn), profile=PROFILE,
                            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                   allow_approx_lut=False,
                                                   linear_task_order=order)).plan


def tasks_of(plan):
    return [a for a in plan.actions if a.kind == "npu_task"]


class FullShapeGroupingTests(unittest.TestCase):
    def test_full_gate_up_uses_two_legal_submits(self):
        plan = build(FULL, 128, "weight_stationary_reuse")
        sizes = [len(s.task_ids) for s in plan.submits]
        self.assertEqual(len(tasks_of(plan)), 72)
        self.assertEqual(sizes, [60, 12])
        self.assertTrue(all(size <= 64 for size in sizes))

    def test_每个_submit_的首任务都不复用(self):
        plan = build(FULL, 128, "weight_stationary_reuse")
        for submit in plan.submits:
            first = next(a for a in plan.actions if a.id == submit.task_ids[0])
            self.assertFalse(first.attrs.get("weight_reuse"))
            self.assertFalse(first.attrs.get("data_reuse"))
        self.assertEqual(sum(1 for a in tasks_of(plan)
                             if a.attrs.get("weight_reuse")), 36)

    def test_每条_K_链都完整落在同一个_submit_内(self):
        plan = build(FULL, 128, "weight_stationary")
        submit_of = {tid: submit.id for submit in plan.submits
                     for tid in submit.task_ids}
        by_id = {t.id: t for t in tasks_of(plan)}
        continuations = 0
        for task in tasks_of(plan):
            k_index = task.attrs.get("k_index")
            if not k_index:
                continue
            prior = [by_id[dep] for dep in task.deps if dep in by_id]
            prior = [p for p in prior if p.attrs.get("k_index") == k_index - 1]
            self.assertTrue(prior, "task has no K predecessor")
            for p in prior:
                self.assertEqual(submit_of[p.id], submit_of[task.id],
                                 "a K chain crosses a submit boundary")
            continuations += 1
        self.assertEqual(continuations, 48)

    def test_no_shape_exceeds_the_submit_bound(self):
        for shape, bm in (((256, 576, 3072), 128), ((384, 576, 1024), 128),
                          ((256, 576, 512), 128), ((256, 512, 768), 256),
                          ((512, 1024, 512), 256)):
            with self.subTest(shape=shape, bm=bm):
                plan = build(shape, bm, "weight_stationary_reuse")
                self.assertTrue(all(len(s.task_ids) <= 64
                                    for s in plan.submits))
                _validate_plan(plan)

    def test_the_existing_policies_still_refuse_what_does_not_fit(self):
        with self.assertRaises(BackendError) as caught:
            build(FULL, 128, "merged")
        self.assertIn("64-task bound", caught.exception.diagnostics[0].message)


@unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
class GroupBoundarySafetyTests(unittest.TestCase):
    def test_c_accepts_the_two_submit_plan(self):
        plan = build(FULL, 128, "weight_stationary_reuse")
        lib = LibRKPlan(REAL)
        lib.create_plan(plan).destroy()

    def test_reuse_on_a_group_first_task_is_refused(self):
        plan = build(FULL, 128, "weight_stationary_reuse")
        actions = list(plan.actions)
        second_submit = plan.submits[1]
        first = next(a for a in actions if a.id == second_submit.task_ids[0])
        actions[first.id] = replace(
            first, attrs={**dict(first.attrs), "weight_reuse": True})
        tampered = replace(plan, plan_hash="")
        object.__setattr__(tampered, "actions", tuple(actions))
        object.__setattr__(tampered, "plan_hash",
                           sha256_json(tampered.to_dict(include_hash=False)))
        with self.assertRaises(BackendError) as caught:
            _validate_plan(tampered)
        self.assertIn("first task of a submit", caught.exception.diagnostics[0].message)
        lib = LibRKPlan(REAL)
        with self.assertRaises(BackendError) as caught:
            lib.create_plan(tampered)
        self.assertIn("reuse", caught.exception.diagnostics[0].message)

    def test_reuse_without_a_matching_predecessor_is_refused(self):
        # The first task of a later K block does not follow the same weight
        # operand at the same K index, so the bit may not be set.
        plan = build(FULL, 128, "weight_stationary_reuse")
        actions = list(plan.actions)
        submit = plan.submits[0]
        by_id = {a.id: a for a in plan.actions}
        target = next(by_id[tid] for tid in submit.task_ids
                      if by_id[tid].attrs.get("k_index") == 1)
        actions[target.id] = replace(
            target, attrs={**dict(target.attrs), "weight_reuse": True})
        tampered = replace(plan, plan_hash="")
        object.__setattr__(tampered, "actions", tuple(actions))
        object.__setattr__(tampered, "plan_hash",
                           sha256_json(tampered.to_dict(include_hash=False)))
        lib = LibRKPlan(REAL)
        with self.assertRaises(BackendError):
            lib.create_plan(tampered)


if __name__ == "__main__":
    unittest.main()
