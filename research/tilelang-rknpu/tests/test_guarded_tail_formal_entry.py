# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal guarded-tail entries for the source-order policies."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import tilelang

from models.tilelang_matmul import make_matmul_tiled_tail_prim
from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.frontend import rknpu_pass as R
from rknpu_backend.frontend import tilelang_lowering
from rknpu_backend.plan import Plan
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
LIB = (Path(__file__).resolve().parents[1].parent / "custom-ops"
       / "build-host-mvp" / "librknpu_plan_mock.so")


def options(policy: str) -> CompileOptions:
    return CompileOptions(numeric_mode="rk_f16_tiled",
                          allow_approx_lut=False,
                          linear_task_order=policy)


def frozen(case: str) -> Plan:
    path = Path(__file__).resolve().parent / "fixtures" / "frozen_plans" / \
        f"{case}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def tail_prim():
    return make_matmul_tiled_tail_prim(1, 576, 960, bm=1, bn=256)


def execution(plan: Plan):
    return {
        "bindings": [(b.name, b.role, b.dtype, tuple(b.shape))
                     for b in plan.bindings],
        "constants": [(c.name, c.dtype, tuple(c.shape), c.pack_key)
                      for c in plan.constants],
        "tasks": [(a.template_id, a.attrs.get("k0"), a.attrs.get("kt"))
                  for a in plan.actions if a.kind == "npu_task"],
        "uploads": [(a.attrs.get("source_binding"),
                     a.attrs.get("src_row_offset"),
                     a.attrs.get("src_col_offset"),
                     a.attrs.get("src_row_stride"),
                     a.attrs.get("transpose_source"),
                     a.attrs.get("cache"))
                    for a in plan.actions if a.kind == "upload"],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode)
                    for s in plan.submits],
        "publish": [(p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
        "numeric": [requirement.to_dict()["contract"]
                    for requirement in plan.numeric_requirements],
    }


class FormalEntryTests(unittest.TestCase):
    CASES = (
        ("baseline", "guarded_tail_baseline"),
        ("scheduled_baseline", "guarded_tail_scheduled_baseline"),
    )

    def test_tail_policies_use_lowering_without_adapter(self):
        for policy, case in self.CASES:
            with self.subTest(policy=policy):
                prim = tail_prim()
                if True:
                    result = compile_tilelang(
                        prim, profile=PROFILE, options=options(policy))
                    artifact = compile_artifact(
                        prim, profile=PROFILE, options=options(policy))
                self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
                self.assertEqual(
                    result.report["source"]["lowering"],
                    "guarded-tail localise -> target TIR -> task consumer")
                self.assertEqual(execution(result.plan), execution(frozen(case)))
                self.assertEqual(result.plan.accounting.data_bo_count,
                                 frozen(case).accounting.data_bo_count)

    def test_linear_baseline_uses_lowering_without_adapter(self):
        prim = make_linear(128, 576, 3072, bm=256, bn=256)
        if True:
            result = compile_tilelang(
                prim, profile=PROFILE, options=options("baseline"))
        self.assertEqual(execution(result.plan),
                         execution(frozen("guarded_linear_baseline")))
        self.assertEqual(result.plan.accounting.data_bo_count,
                         frozen("guarded_linear_baseline").accounting.data_bo_count)

    def test_non_default_policy_uses_formal_guarded_lowering(self):
        prim = make_linear(128, 576, 3072, bm=256, bn=256)
        if True:
            result = compile_tilelang(
                prim, profile=PROFILE, options=options("merged"))
        self.assertGreater(result.plan.accounting.task_count, 0)
        tasks = [a for a in result.plan.actions if a.kind == "npu_task"]
        self.assertEqual({task.attrs["linear_task_policy"] for task in tasks},
                         {"merged"})


class MockParityTests(unittest.TestCase):
    def test_tail_policies_match_frozen_mock(self):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        lib = LibRKPlan(str(LIB))
        rng = np.random.default_rng(20261103)
        a = rng.standard_normal((1, 576)).astype(np.float16)
        b = rng.standard_normal((576, 960)).astype(np.float16)
        for policy, case in FormalEntryTests.CASES:
            with self.subTest(policy=policy):
                old = frozen(case)
                new = compile_tilelang(
                    tail_prim(), profile=PROFILE,
                    options=options(policy)).plan
                old_out = np.zeros((1, 960), dtype=np.float16)
                new_out = np.zeros((1, 960), dtype=np.float16)
                old_call = PlanCall(lib, old, {"B": b}, fd=0)
                new_call = PlanCall(lib, new, {"B": b}, fd=0)
                try:
                    old_call.run_into([a], [old_out])
                    new_call.run_into([a], [new_out])
                finally:
                    old_call.close()
                    new_call.close()
                self.assertTrue(np.array_equal(new_out, old_out))


if __name__ == "__main__":
    unittest.main()
