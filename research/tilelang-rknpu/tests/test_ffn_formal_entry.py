# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal entry for the two admitted local SwiGLU dataflow chains."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from models.tilelang_matmul import make_swiglu_ffn_f16
from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.frontend import tilelang_lowering
from rknpu_backend.plan import Plan
from tests.fixtures import tilelang_ffn

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def frozen(name):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def execution(plan):
    return {
        "bindings": [(b.name, b.role, tuple(b.shape)) for b in plan.bindings],
        "actions": [(a.kind, a.template_id, tuple(a.inputs), tuple(a.outputs),
                     tuple(a.deps))
                    for a in plan.actions],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode, s.deps)
                    for s in plan.submits],
        "publish": [(p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
    }


class FFNFormalEntryTests(unittest.TestCase):
    def compile(self, prim, name):
        if True:
            result = compile_tilelang(prim, profile=PROFILE, options=options())
            artifact = compile_artifact(prim, profile=PROFILE,
                                        options=options())
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        self.assertEqual(result.report["source"]["lowering"],
                         "source dataflow rewrite -> target tasks -> task consumer")
        self.assertEqual(execution(result.plan), execution(frozen(name)))
        return result

    def test_model_ffn_uses_formal_task_entry(self):
        result = self.compile(make_swiglu_ffn_f16(64, 256, 128),
                              "ffn_model_f16")
        self.assertEqual(result.plan.accounting.task_count, 7)
        self.assertEqual(len(result.plan.submits), 5)

    def test_fixture_ffn_uses_formal_task_entry(self):
        result = self.compile(tilelang_ffn.swiglu_ffn_f16(8, 64, 32, bk=64),
                              "ffn_fixture_f16")
        self.assertEqual(result.plan.accounting.task_count, 7)

    def test_value_ids_disambiguate_projection_and_residual(self):
        result = compile_tilelang(
            make_swiglu_ffn_f16(64, 256, 128), profile=PROFILE,
            options=options())
        requirements = {r.op_index: r for r in result.plan.numeric_requirements}
        self.assertEqual(requirements[4].inputs, ("mul16", "Wd"))
        self.assertEqual(requirements[4].outputs, ("proj32",))
        self.assertEqual(requirements[5].inputs, ("proj32", "H"))
        self.assertEqual(requirements[5].outputs, ("Y",))

    def test_direct_dispatch_uses_the_shared_ffn_entry(self):
        prim = tilelang_ffn.swiglu_ffn_f16(8, 64, 32, bk=64)
        self.assertEqual(tilelang_lowering.entry_for(prim).__name__,
                         "compile_ffn_tasks")
        plan, _, _, _ = tilelang_lowering.compile_tilelang_ir(
            prim, profile=PROFILE, options=options())
        self.assertEqual(plan.accounting.task_count, 7)


class FFNMockTests(unittest.TestCase):
    def run_pair(self, prim, name):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        old = frozen(name)
        new = compile_tilelang(prim, profile=PROFILE,
                               options=options()).plan
        rng = np.random.default_rng(20261106)
        consts = {}
        for binding in old.bindings:
            if binding.role == "constant":
                if binding.name.startswith("__identity_"):
                    consts[binding.name] = np.eye(
                        binding.shape[0], dtype=np.float16)
                else:
                    consts[binding.name] = rng.standard_normal(
                        binding.shape).astype(np.float16)
        arrays = {}
        for binding in old.bindings:
            if binding.role == "input":
                arrays[binding.name] = rng.standard_normal(
                    binding.shape).astype(np.float16)
        out_name = next(b.name for b in old.bindings if b.role == "output")
        old_out = np.zeros(old.bindings[next(
            b.id for b in old.bindings if b.name == out_name)].shape,
            dtype=np.float16)
        new_out = old_out.copy()
        lib = LibRKPlan(str(LIB))
        old_call = PlanCall(lib, old, consts, fd=0)
        new_call = PlanCall(lib, new, consts, fd=0)
        try:
            old_call.run_into(list(arrays.values()), [old_out])
            new_call.run_into(list(arrays.values()), [new_out])
        finally:
            old_call.close()
            new_call.close()
        self.assertTrue(np.array_equal(new_out, old_out))

    def test_model_ffn_mock(self):
        self.run_pair(make_swiglu_ffn_f16(64, 256, 128), "ffn_model_f16")

    def test_fixture_ffn_mock(self):
        self.run_pair(tilelang_ffn.swiglu_ffn_f16(8, 64, 32, bk=64),
                      "ffn_fixture_f16")


if __name__ == "__main__":
    unittest.main()
