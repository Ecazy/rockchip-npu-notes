# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal entries for the guarded single-trip, final-only and policy forms."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from models.tilelang_matmul import make_matmul_tiled_prim
from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.plan import Plan
from tests.fixtures.tilelang_linear_scheduled import make_linear
from tests.fixtures import tilelang_ragged as ragged

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"


def options(policy="baseline", group=0):
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                          linear_task_order=policy,
                          linear_group_n_tiles=group)


def frozen(name):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def execution(plan):
    return {
        "bindings": [(b.name, b.role, tuple(b.shape)) for b in plan.bindings],
        "tasks": [(a.template_id, a.attrs.get("k0"), a.attrs.get("kt"),
                   bool(a.attrs.get("weight_reuse")),
                   bool(a.attrs.get("data_reuse")),
                   a.attrs.get("submit_group"))
                  for a in plan.actions if a.kind == "npu_task"],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode)
                    for s in plan.submits],
        "publish": sorted((p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in plan.publish),
        "accounting": plan.accounting.to_dict(),
        "numeric": [r.to_dict()["contract"]
                    for r in plan.numeric_requirements],
    }


class RaggedFormalEntryTests(unittest.TestCase):
    def compile(self, prim, *, name, policy="baseline", group=0,
                b_is_input=False):
        if True:
            result = compile_tilelang(
                prim, profile=PROFILE, options=options(policy, group),
                b_is_input=b_is_input)
            artifact = compile_artifact(
                prim, profile=PROFILE, options=options(policy, group),
                b_is_input=b_is_input)
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        self.assertEqual(result.report["source"]["lowering"],
                         "guarded-tail localise -> target TIR -> task consumer")
        if name is not None:
            self.assertEqual(execution(result.plan), execution(frozen(name)))
        return result

    def test_guarded_single_trip_uses_formal_entry(self):
        result = self.compile(
            make_linear(128, 256, 256, bm=256, bn=256),
            name="ragged_single_trip_baseline")
        self.assertEqual(result.plan.accounting.task_count, 1)

    def test_guarded_single_trip_b_is_input_uses_formal_entry(self):
        self.compile(ragged.ragged_tail_f16(3, 256, 64, bm=256, bn=64),
                     name="ragged_single_trip_b_input", b_is_input=True)

    def test_three_buffer_final_only_uses_formal_entry(self):
        self.compile(ragged.ragged_tail(13, 33, 257, 16, 64),
                     name="ragged_final_only_k257")

    def test_weight_stationary_policy_keeps_behavior(self):
        self.compile(make_linear(256, 576, 3072, bm=128, bn=256),
                     name="ragged_policy_weight_stationary",
                     policy="weight_stationary")

    def test_weight_stationary_reuse_policy_keeps_behavior(self):
        self.compile(make_linear(256, 576, 3072, bm=128, bn=256),
                     name="ragged_policy_weight_stationary_reuse",
                     policy="weight_stationary_reuse")

    def test_merged_policy_keeps_behavior(self):
        self.compile(make_linear(128, 576, 3072, bm=256, bn=256),
                     name="ragged_policy_merged", policy="merged")

    def test_grouped_and_group_count_policies_keep_behavior(self):
        self.compile(make_linear(256, 576, 3072, bm=128, bn=256),
                     name="ragged_policy_grouped_linear",
                     policy="grouped_linear")
        self.compile(make_linear(256, 576, 3072, bm=128, bn=256),
                     name="ragged_policy_weight_stationary_group5",
                     policy="weight_stationary", group=5)
        self.compile(make_linear(256, 576, 3072, bm=128, bn=256),
                     name="ragged_policy_reuse_group5",
                     policy="weight_stationary_reuse", group=5)

    def test_unannotated_multi_grid_final_only_is_source_faithful(self):
        result = compile_tilelang(
            ragged.ragged_tail(257, 257, 300, 128, 128),
            profile=PROFILE, options=options())
        # The old adapter used the historical 256x256 default for this form.
        # The task chain keeps the actual 128x128 source grid.
        self.assertEqual(result.plan.accounting.task_count, 18)
        self.assertEqual(len(result.plan.publish), 9)


class RaggedMockTests(unittest.TestCase):
    def run_pair(self, prim, name, *, policy="baseline", group=0):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        old = frozen(name)
        new = compile_tilelang(
            prim, profile=PROFILE, options=options(policy, group)).plan
        m, k = old.bindings[0].shape
        n = old.bindings[1].shape[1]
        rng = np.random.default_rng(20261105)
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        old_out = np.zeros((m, n), dtype=np.float16)
        new_out = np.zeros((m, n), dtype=np.float16)
        lib = LibRKPlan(str(LIB))
        old_call = PlanCall(lib, old, {"B": b}, fd=0)
        new_call = PlanCall(lib, new, {"B": b}, fd=0)
        try:
            old_call.run_into([a], [old_out])
            new_call.run_into([a], [new_out])
        finally:
            old_call.close()
            new_call.close()
        self.assertTrue(np.array_equal(new_out, old_out))

    def test_single_trip_mock(self):
        self.run_pair(make_linear(128, 256, 256, bm=256, bn=256),
                      "ragged_single_trip_baseline")

    def test_final_only_mock(self):
        self.run_pair(ragged.ragged_tail(13, 33, 257, 16, 64),
                      "ragged_final_only_k257")

    def test_weight_stationary_mock(self):
        self.run_pair(make_linear(256, 576, 3072, bm=128, bn=256),
                      "ragged_policy_weight_stationary",
                      policy="weight_stationary")


if __name__ == "__main__":
    unittest.main()
