# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal entry for the local projection and branch dataflow inputs."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.frontend import tilelang_lowering
from rknpu_backend.plan import Plan
from tests.fixtures import tilelang_multi_output, tilelang_silu_proj

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def frozen(name):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def execution(plan, *, projection=False):
    return {
        "bindings": [(b.name, b.role, tuple(b.shape)) for b in plan.bindings],
        "actions": [(a.kind, a.template_id, tuple(a.inputs), tuple(a.outputs),
                     () if projection and
                     a.template_id == "co.ew_mul.f16.v1" else tuple(a.deps))
                    for a in plan.actions],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode, s.deps)
                    for s in plan.submits],
        "publish": [(p.name, p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
    }


class LocalDataflowFormalEntryTests(unittest.TestCase):
    def compile(self, prim, name):
        if True:
            result = compile_tilelang(prim, profile=PROFILE, options=options())
            artifact = compile_artifact(prim, profile=PROFILE,
                                        options=options())
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        projection = name == "silu_proj_static_dataflow"
        self.assertEqual(execution(result.plan, projection=projection),
                         execution(frozen(name), projection=projection))
        if projection:
            new_mul = next(a for a in result.plan.actions if
                           a.template_id == "co.ew_mul.f16.v1")
            old_mul = next(a for a in frozen(name).actions if
                           a.template_id == "co.ew_mul.f16.v1")
            self.assertTrue(set(old_mul.deps).issubset(new_mul.deps))
            self.assertEqual(set(new_mul.deps) - set(old_mul.deps), {5})
        return result

    def test_projection_chain_uses_formal_entry(self):
        result = self.compile(tilelang_silu_proj.silu_proj(13, 64, 32),
                              "silu_proj_static_dataflow")
        self.assertEqual(result.plan.accounting.task_count, 5)
        self.assertEqual(len(result.plan.submits), 3)
        self.assertEqual([p.name for p in result.plan.publish], ["Y"])

    def test_explicit_sigmoid_and_mul_keep_separate_attribution(self):
        result = compile_tilelang(tilelang_silu_proj.silu_proj(13, 64, 32),
                                  profile=PROFILE, options=options())
        requirements = list(result.plan.numeric_requirements)
        by_opcode = {}
        for requirement in requirements:
            by_opcode.setdefault(requirement.opcode, []).append(requirement)
        self.assertEqual(by_opcode["sigmoid"][0].tasks, (6,))
        self.assertEqual(by_opcode["mul"][0].tasks, (7,))
        self.assertEqual([item.tasks for item in by_opcode["matmul"]],
                         [(5,), (8,)])
        sigmoid = next(a for a in result.plan.actions
                       if a.template_id == "co.lut.sigmoid.f16.v1")
        mul = next(a for a in result.plan.actions
                   if a.template_id == "co.ew_mul.f16.v1")
        self.assertEqual(sigmoid.attrs["submit_group"], "act:dataflow:1")
        self.assertEqual(mul.attrs["submit_group"], "act:dataflow:1")

    def test_branch_outputs_use_two_real_store_tasks(self):
        result = self.compile(
            tilelang_multi_output.branch_outputs(8, 64, 32, 48),
            "branch_outputs_multi")
        self.assertEqual(result.plan.accounting.task_count, 2)
        self.assertEqual([p.name for p in result.plan.publish], ["Y1", "Y2"])
        self.assertEqual([(p.rows, p.cols) for p in result.plan.publish],
                         [(8, 32), (8, 48)])

    def test_direct_dispatch_uses_shared_local_dataflow_entry(self):
        prim = tilelang_multi_output.branch_outputs(8, 64, 32, 48)
        self.assertEqual(tilelang_lowering.entry_for(prim).__name__,
                         "compile_ffn_tasks")
        plan, _, _, _ = tilelang_lowering.compile_tilelang_ir(
            prim, profile=PROFILE, options=options())
        self.assertEqual(plan.accounting.task_count, 2)


class LocalDataflowMockTests(unittest.TestCase):
    def run_pair(self, prim, name, inputs_n, outputs_n):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        old = frozen(name)
        new = compile_tilelang(prim, profile=PROFILE,
                               options=options()).plan
        rng = np.random.default_rng(20261107)
        consts = {}
        for binding in old.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                consts[binding.name] = np.eye(binding.shape[0],
                                              dtype=np.float16)
            else:
                consts[binding.name] = rng.standard_normal(
                    binding.shape).astype(np.float16)
        inputs = [
            rng.standard_normal(binding.shape).astype(np.float16)
            for binding in old.bindings if binding.role == "input"
        ]
        outputs_old = [
            np.zeros(binding.shape, dtype=np.float16)
            for binding in old.bindings if binding.role == "output"
        ]
        outputs_new = [item.copy() for item in outputs_old]
        lib = LibRKPlan(str(LIB))
        old_call = PlanCall(lib, old, consts, fd=0)
        new_call = PlanCall(lib, new, consts, fd=0)
        try:
            old_call.run_into(inputs, outputs_old)
            new_call.run_into(inputs, outputs_new)
        finally:
            old_call.close()
            new_call.close()
        self.assertEqual(len(outputs_new), outputs_n)
        for before, after in zip(outputs_old, outputs_new):
            self.assertTrue(np.array_equal(before, after))

    def test_projection_mock(self):
        self.run_pair(tilelang_silu_proj.silu_proj(13, 64, 32),
                      "silu_proj_static_dataflow", 1, 1)

    def test_branch_mock(self):
        self.run_pair(
            tilelang_multi_output.branch_outputs(8, 64, 32, 48),
            "branch_outputs_multi", 1, 2)


if __name__ == "__main__":
    unittest.main()
