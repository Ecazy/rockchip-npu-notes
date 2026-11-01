# SPDX-License-Identifier: GPL-3.0-or-later
"""A finished plan may only contain action kinds the runtime can execute.

The plan schema still knows the transient host steps (host_pack/host_expand/
host_check/host_compute/commit) so a draft can express them, but a Plan that
leaves plan validation is the artifact handed to the C verifier and the runtime.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                           load_plan, matmul_module, save_plan, swiglu_module)
from rknpu_backend.errors import BackendError
from rknpu_backend.plan import (ACTION_KINDS, EXECUTABLE_ACTION_KINDS, Action,
                                _validate_plan)
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
REAL = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
MOCK = str(ROOT.parent / "custom-ops" / "build-host-mvp" /
           "librknpu_plan_mock.so")
TRANSIENT = sorted(ACTION_KINDS - EXECUTABLE_ACTION_KINDS)


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def plan_for(builder=matmul_module, **kwargs):
    return compile_rkir(builder(**kwargs) if kwargs else builder(m=16, k=16,
                                                                     n=8),
                        profile=PROFILE, options=options()).plan


def rehash(plan):
    object.__setattr__(plan, "plan_hash",
                       sha256_json(plan.to_dict(include_hash=False)))
    return plan


def with_extra_action(plan, kind, *, action_id=None):
    extra = Action(action_id if action_id is not None else len(plan.actions),
                   kind, None, (), (), (0,), "repro:" + kind, {})
    tampered = replace(plan, plan_hash="")
    object.__setattr__(tampered, "actions", plan.actions + (extra,))
    return rehash(tampered)


class ActionKindSetsTests(unittest.TestCase):
    def test_the_schema_keeps_the_transient_steps(self):
        self.assertEqual(TRANSIENT,
                         ["commit", "host_check", "host_compute",
                          "host_expand", "host_pack"])
        self.assertEqual(sorted(EXECUTABLE_ACTION_KINDS),
                         ["download", "npu_task", "upload"])
        self.assertTrue(EXECUTABLE_ACTION_KINDS <= ACTION_KINDS)

    def test_a_transient_action_can_still_be_built(self):
        # Draft code is allowed to express the step; only a finished plan is not.
        action = Action(0, "commit", None, (), (), (), "draft", {})
        self.assertEqual(action.kind, "commit")


class PlanKindTests(unittest.TestCase):
    def test_planner_output_only_contains_executable_kinds(self):
        for label, plan in (("matmul", plan_for()),
                            ("swiglu", plan_for(swiglu_module, m=64, d=256,
                                               h=128))):
            with self.subTest(case=label):
                kinds = {a.kind for a in plan.actions}
                self.assertTrue(kinds <= EXECUTABLE_ACTION_KINDS, kinds)

    def test_a_transient_step_in_a_plan_is_refused(self):
        plan = plan_for()
        for kind in TRANSIENT:
            with self.subTest(kind=kind):
                tampered = with_extra_action(plan, kind)
                with self.assertRaises(BackendError) as caught:
                    _validate_plan(tampered)
                message = caught.exception.diagnostics[0].message
                if kind == "host_compute":
                    self.assertIn("hidden host arithmetic", message)
                else:
                    self.assertIn("transient planning step", message)
                    self.assertIn(kind, message)

    def test_the_error_names_the_step_and_the_allowed_kinds(self):
        plan = with_extra_action(plan_for(), "commit")
        with self.assertRaises(BackendError) as caught:
            _validate_plan(plan)
        diagnostic = caught.exception.diagnostics[0]
        self.assertEqual(diagnostic.to_dict().get("actual"), "commit")
        self.assertEqual(diagnostic.to_dict().get("required"),
                         ["download", "npu_task", "upload"])

    def test_loading_a_plan_with_a_transient_step_is_refused(self):
        plan = plan_for()
        payload = json.loads(json.dumps(plan.to_dict()))
        payload["actions"].append({
            "id": len(payload["actions"]), "kind": "commit", "template_id": None,
            "inputs": [], "outputs": [], "deps": [0], "origin": "test",
            "attrs": {}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.rkplan.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(BackendError) as caught:
                load_plan(str(path))
        self.assertIn("transient planning step",
                      caught.exception.diagnostics[0].message)

    def test_normal_plans_keep_their_identity(self):
        plan = plan_for()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.to_dict(), plan.to_dict())


@unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
class CKindTests(unittest.TestCase):
    """The C verifier refuses the same plans on its own."""

    def test_c_accepts_planner_output(self):
        lib = LibRKPlan(REAL)
        for plan in (plan_for(), plan_for(swiglu_module, m=64, d=256, h=128)):
            cplan = lib.create_plan(plan)
            cplan.destroy()

    def test_c_refuses_every_transient_kind(self):
        lib = LibRKPlan(REAL)
        plan = plan_for()
        for kind in TRANSIENT:
            with self.subTest(kind=kind):
                tampered = with_extra_action(plan, kind)
                with self.assertRaises(BackendError) as caught:
                    lib.create_plan(tampered)
                message = caught.exception.diagnostics[0].message
                self.assertTrue("unsupported action kind" in message or
                                "hidden host compute" in message, message)


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class ExecutionRefusalTests(unittest.TestCase):
    def test_a_plan_with_a_transient_step_never_starts(self):
        from rknpu_backend.models import PlanCall

        plan = with_extra_action(plan_for(), "commit")
        lib = LibRKPlan(MOCK)
        with self.assertRaises(BackendError):
            PlanCall(lib, plan, {"B": np.zeros((16, 8), dtype=np.float16)},
                     fd=0)


if __name__ == "__main__":
    unittest.main()
