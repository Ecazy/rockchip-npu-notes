# SPDX-License-Identifier: GPL-3.0-or-later
"""A submit owns a contiguous run of actions, and both layers must say so.

The runtime walks actions in ascending id order and flushes a submit when it
reaches that submit's last listed task.  A plan that puts the first and third
task in one submit and the middle task in another therefore executes a
different order than it declares -- and the flushed batch does not even contain
the task it skipped over.  Python and the C verifier both have to refuse it on
their own; the C cases here are fed straight to the library with the Python plan
validator bypassed, so a C acceptance can never be excused by "Python would have
caught it".
"""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, default_profile, load_plan,
                           matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.storage import CO_PROGRAM_BYTES
from rknpu_backend.plan import _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT.parent / "custom-ops" / "build-host-mvp"
REAL = str(BUILD / "librknpu_plan.so")
MOCK = str(BUILD / "librknpu_plan_mock.so")


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def _chain_module(rows=16, cols=16):
    shape = TensorType("float16", (rows, cols))
    values = (Value(0, "X", shape, None, "input"),
              Value(1, "A", shape, 0, "temporary"),
              Value(2, "B", shape, 1, "temporary"),
              Value(3, "Y", shape, 2, "output"))
    ops = (Op(0, "sigmoid", (0,), (1,), {}),
           Op(1, "sigmoid", (1,), (2,), {}),
           Op(2, "sigmoid", (2,), (3,), {}))
    return Module(name="sigmoid-chain", values=values, ops=ops,
                  inputs=(0,), outputs=(3,), source_hash="sigmoid-chain",
                  profile_hash=PROFILE.profile_hash)


def chain_plan(rows=16, cols=16):
    return compile_rkir(_chain_module(rows, cols), profile=PROFILE,
                        options=options()).plan


def tampered(plan, submits, *, max_tasks, actions=None):
    """Apply new submits with a consistent accounting, bypassing validation."""
    command = max_tasks * 2048 * 8
    accounting = replace(plan.accounting, submit_count=len(submits),
                         command_workspace_bytes=command,
                         device_request_bytes=(plan.accounting.reserved_bytes
                                               + command),
                         host_bytes=(plan.accounting.host_bytes
                                     + (max_tasks - 1) * CO_PROGRAM_BYTES))
    tampered_plan = replace(plan, plan_hash="")
    if actions is not None:
        object.__setattr__(tampered_plan, "actions", tuple(actions))
    object.__setattr__(tampered_plan, "submits", submits)
    object.__setattr__(tampered_plan, "accounting", accounting)
    object.__setattr__(tampered_plan, "plan_hash",
                       sha256_json(tampered_plan.to_dict(include_hash=False)))
    return tampered_plan


def submissions(plan):
    return [a.id for a in plan.actions if a.kind == "npu_task"]


class SubmitRunShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = chain_plan()
        cls.tasks = submissions(cls.plan)
        base = cls.plan.submits[0]
        cls.base_submit = base

    def _submit(self, index, task_ids, deps=()):
        return replace(self.base_submit, id=index, task_ids=tuple(task_ids),
                       deps=tuple(deps))

    def test_every_npu_task_belongs_to_exactly_one_submit(self):
        first = self.tasks[0]
        duplicate = tampered(self.plan,
                             (self._submit(0, (first, first)),), max_tasks=2)
        with self.assertRaises(BackendError):
            _validate_plan(duplicate)
        missing = tampered(self.plan,
                           (self._submit(0, (first,)),), max_tasks=1)
        with self.assertRaises(BackendError):
            _validate_plan(missing)

    def test_task_lists_must_be_in_emission_order(self):
        first, middle, third = self.tasks
        descending = tampered(
            self.plan,
            (self._submit(0, (third, first)), self._submit(1, (middle,), (0,))),
            max_tasks=2)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(descending)
        self.assertIn("ascending emission order",
                      caught.exception.diagnostics[0].message)

    def test_another_submits_task_cannot_sit_inside_a_run(self):
        first, middle, third = self.tasks
        interleaved = tampered(
            self.plan,
            (self._submit(0, (first, third)), self._submit(1, (middle,), (0,))),
            max_tasks=2)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(interleaved)
        self.assertIn("contiguous run of tasks",
                      caught.exception.diagnostics[0].message)

    def test_a_host_action_cannot_be_hidden_inside_a_run(self):
        first, middle, third = self.tasks
        # The middle action becomes a second host upload; the run from `first`
        # to `third` now spans it, which a chained submit cannot honour.  A real
        # upload is used because a finished plan may not carry a transient
        # commit step any more (see test_action_kind_contract).
        upload = next(a for a in self.plan.actions if a.kind == "upload")
        actions = [replace(a, kind="upload", template_id=None, inputs=(),
                           outputs=upload.outputs, deps=upload.deps,
                           attrs=upload.attrs) if a.id == middle else a
                   for a in self.plan.actions]
        hidden = tampered(self.plan, (self._submit(0, (first, third)),),
                          max_tasks=2, actions=actions)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(hidden)
        self.assertIn("host action is interleaved",
                      caught.exception.diagnostics[0].message)

    def test_runs_must_follow_the_declared_submit_order(self):
        first, middle, third = self.tasks
        swapped = tampered(
            self.plan,
            (self._submit(0, (middle,)), self._submit(1, (first,), (0,)),
             self._submit(2, (third,), (1,))),
            max_tasks=1)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(swapped)
        self.assertIn("must appear in submit order",
                      caught.exception.diagnostics[0].message)

    def test_normal_chained_plan_is_unchanged(self):
        self.assertEqual(len(self.tasks), 3)
        self.assertEqual([s.task_ids for s in self.plan.submits],
                         [(task,) for task in self.tasks])
        _validate_plan(self.plan)


@unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
class SubmitRunShapeCTests(unittest.TestCase):
    """The C verifier must refuse these on its own (Python validator bypassed)."""

    @classmethod
    def setUpClass(cls):
        cls.plan = chain_plan()
        cls.tasks = submissions(cls.plan)
        cls.base_submit = cls.plan.submits[0]
        cls.lib = LibRKPlan(REAL)

    def _submit(self, index, task_ids, deps=()):
        return replace(self.base_submit, id=index, task_ids=tuple(task_ids),
                       deps=tuple(deps))

    def _c_reject(self, plan, needle):
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(plan)
        diagnostic = caught.exception.diagnostics[0]
        self.assertIn(needle, diagnostic.message)

    def test_c_accepts_the_normal_chain(self):
        cplan = self.lib.create_plan(self.plan)
        cplan.destroy()

    def test_c_refuses_an_interleaved_run(self):
        first, middle, third = self.tasks
        interleaved = tampered(
            self.plan,
            (self._submit(0, (first, third)), self._submit(1, (middle,), (0,))),
            max_tasks=2)
        self._c_reject(interleaved, "contiguous run of tasks")

    def test_c_refuses_descending_task_lists(self):
        first, middle, third = self.tasks
        descending = tampered(
            self.plan,
            (self._submit(0, (third, first)), self._submit(1, (middle,), (0,))),
            max_tasks=2)
        self._c_reject(descending, "ascending emission order")

    def test_c_refuses_runs_out_of_submit_order(self):
        first, middle, third = self.tasks
        swapped = tampered(
            self.plan,
            (self._submit(0, (middle,)), self._submit(1, (first,), (0,)),
             self._submit(2, (third,), (1,))),
            max_tasks=1)
        self._c_reject(swapped, "must appear in submit order")

    # A host action hidden inside a run is refused by the C verifier already
    # (the span must be all NPU tasks, checked in validate_execution_order);
    # the planner never emits such a plan and this round did not change that
    # rule, so the Python side carries the new test for it.


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class RejectedPlansNeverRunTests(unittest.TestCase):
    def test_a_rejected_plan_cannot_start_executing(self):
        from rknpu_backend.models import PlanCall

        plan = chain_plan()
        first, middle, third = submissions(plan)
        base = plan.submits[0]
        interleaved = tampered(
            plan,
            (replace(base, id=0, task_ids=(first, third), deps=()),
             replace(base, id=1, task_ids=(middle,), deps=(0,))),
            max_tasks=2)
        lib = LibRKPlan(MOCK)
        with self.assertRaises(BackendError):
            PlanCall(lib, interleaved, {}, fd=0)


class NormalPlanCompatibilityTests(unittest.TestCase):
    """Every shape the planner emits today keeps its own run structure."""

    def _submits_are_contiguous(self, plan):
        owner = {task: submit.id for submit in plan.submits
                 for task in submit.task_ids}
        previous = -1
        for submit in plan.submits:
            self.assertEqual(list(submit.task_ids), sorted(submit.task_ids))
            self.assertGreater(submit.task_ids[0], previous)
            for action_id in range(submit.task_ids[0], submit.task_ids[-1] + 1):
                self.assertEqual(owner.get(action_id), submit.id)
            previous = submit.task_ids[-1]

    def test_shapes_the_planner_produces_stay_contiguous(self):
        cases = {
            "activation-chain": (lambda: _chain_module(), options()),
            "merged-multi-k": (lambda: matmul_module(m=128, k=576, n=3072),
                               options(linear_task_order="merged")),
            "baseline-multi-k": (lambda: matmul_module(m=128, k=576, n=3072),
                                 options(linear_task_order="baseline")),
            "separate-mode": (lambda: matmul_module(m=13, k=64, n=33),
                              options(submit_mode="separate")),
            "multi-output": (lambda: matmul_module(m=13, k=64, n=33),
                             options(submit_mode="chain")),
        }
        for name, (build, setting) in cases.items():
            with self.subTest(case=name):
                plan = compile_rkir(build(), profile=PROFILE,
                                    options=setting).plan
                _validate_plan(plan)
                self._submits_are_contiguous(plan)
                if Path(REAL).is_file():
                    lib = LibRKPlan(REAL)
                    cplan = lib.create_plan(plan)
                    cplan.destroy()

    def test_repeat_group_stays_contiguous(self):
        from rknpu_backend.passes.repeat_expand import expand_static_repeat

        contract = matmul_module(m=16, k=16, n=16, b_layout="nk").ops[0].attrs[
            "numeric_contract"]
        module = Module(
            name="qk",
            values=(Value(0, "Q", TensorType("float16", (16, 16)), None, "input"),
                    Value(1, "K", TensorType("float16", (16, 16)), None, "input"),
                    Value(2, "S", TensorType("float16", (16, 16)), 0, "output")),
            ops=(Op(0, "matmul", (0, 1), (2,),
                    {"b_layout": "kn", "numeric_contract": contract}),),
            inputs=(0, 1), outputs=(2,), source_hash="qk")
        parent = compile_rkir(module, profile=PROFILE, options=options()).plan
        group = expand_static_repeat(parent, 4, shared_bo=True)
        _validate_plan(group)
        self._submits_are_contiguous(group)
        if Path(REAL).is_file():
            lib = LibRKPlan(REAL)
            cplan = lib.create_plan(group)
            cplan.destroy()


if __name__ == "__main__":
    unittest.main()
