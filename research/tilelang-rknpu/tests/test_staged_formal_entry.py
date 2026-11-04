# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal entry for launch-segmented staged sources."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.frontend import tilelang_lowering, staged_launch
from rknpu_backend.plan import Plan
from tests.fixtures import (tilelang_branch_staged, tilelang_ffn_scheduled,
                            tilelang_ffn_staged, tilelang_projection_staged,
                            tilelang_two_stage)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = default_profile()
TILED = CompileOptions(numeric_mode="rk_f16_tiled")
LUT = CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)
CASES = (
    ("staged_projection", tilelang_projection_staged.projection_staged, LUT),
    ("staged_ffn", tilelang_ffn_staged.ffn_staged, LUT),
    ("staged_branch", tilelang_branch_staged.branch_staged, TILED),
    ("two_stage_scheduled", tilelang_two_stage.make_two_stage(bm0=65, bm1=65),
     TILED),
    ("two_stage_scheduled_m1", tilelang_two_stage.make_two_stage(
        m=1, bm0=1, bm1=1), TILED),
    ("ffn_scheduled", tilelang_ffn_scheduled.make_ffn(m=256, tn=192), LUT),
    ("ffn_scheduled_m1", tilelang_ffn_scheduled.make_ffn(
        m=1, bm=1, tn=192), LUT),
)


def frozen(name):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def two_stage_summary(plan):
    return {
        "tasks": [(a.kind, a.template_id,
                   a.attrs.get("source_stage_index"),
                   a.attrs.get("k0"), a.attrs.get("kt"),
                   len(a.deps))
                  for a in plan.actions if a.kind == "npu_task"],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode)
                    for s in plan.submits],
        "publish": [(p.name, p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
    }


def execution(plan):
    return {
        "bindings": [(b.name, b.role, tuple(b.shape)) for b in plan.bindings],
        "actions": [(a.kind, a.template_id, tuple(a.inputs), tuple(a.outputs),
                     tuple(a.deps), tuple(sorted(a.attrs.items())))
                    for a in plan.actions],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode, s.deps)
                    for s in plan.submits],
        "publish": [(p.name, p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
    }


class StagedFormalEntryTests(unittest.TestCase):
    def compile(self, prim, name, options):
        if True:
            result = compile_tilelang(prim, profile=PROFILE, options=options)
            artifact = compile_artifact(prim, profile=PROFILE,
                                        options=options)
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        reference = frozen(name)
        if name.startswith("two_stage"):
            self.assertEqual(two_stage_summary(result.plan),
                             two_stage_summary(reference))
        else:
            self.assertEqual(sorted({p.name for p in result.plan.publish}),
                             sorted({p.name for p in reference.publish}))
            self.assertTrue(result.plan.actions)
        return result

    def test_real_staged_plans_use_formal_entry(self):
        for name, prim, options in CASES:
            with self.subTest(name=name):
                self.compile(prim, name, options)

    def test_launch_split_matches_actual_kernel_scopes(self):
        expected = {
            "staged_projection": 5,
            "staged_ffn": 7,
            "staged_branch": 2,
            "two_stage_scheduled": 2,
            "ffn_scheduled": 6,
        }
        for name, count in expected.items():
            prim = dict((item[0], item[1]) for item in CASES)[name]
            with self.subTest(name=name):
                self.assertEqual(len(staged_launch.stage_groups(prim)), count)
                self.assertTrue(staged_launch.is_staged_candidate(prim))

    def test_scheduled_stage_contract_and_submit_boundaries(self):
        for name in ("two_stage_scheduled", "ffn_scheduled"):
            prim = dict((item[0], item[1]) for item in CASES)[name]
            options = dict((item[0], item[2]) for item in CASES)[name]
            module, _ = staged_launch.two_stage_plan_inputs(
                prim, options=options, profile_hash=PROFILE.profile_hash)
            stages = [op.attrs.get("source_stage_index")
                      for op in module.ops
                      if op.attrs.get("source_stage_index") is not None]
            self.assertTrue(stages)
            for op in module.ops:
                if op.attrs.get("source_stage_index") is None:
                    continue
                self.assertEqual(op.attrs["source_stage_boundary"],
                                 "kernel_launch")
                self.assertEqual(op.attrs["source_stage_schedule_contract"],
                                 "staged_mn_tiles_v1")
            plan = compile_tilelang(prim, profile=PROFILE,
                                    options=options).plan
            for submit in plan.submits:
                stages_in_submit = {
                    plan.actions[task].attrs.get("source_stage_index")
                    for task in submit.task_ids}
                self.assertEqual(len(stages_in_submit), 1)

    def test_two_stage_actual_launch_chain_and_global_relation(self):
        prim = tilelang_two_stage.make_two_stage(bm0=65, bm1=65)
        chain = staged_launch.two_stage_chain(prim, options=TILED)
        self.assertEqual(chain["launch_count"], 2)
        self.assertEqual(chain["stages"][0]["operand_a"], "X")
        self.assertEqual(chain["stages"][0]["operand_b"], "W1")
        self.assertEqual(chain["stages"][0]["output_name"], "Z")
        self.assertEqual(chain["stages"][1]["operand_a"], "Z")
        self.assertEqual(chain["stages"][1]["operand_b"], "W2")
        self.assertEqual(chain["stages"][1]["output_name"], "Y")
        relations = [row for row in chain["global_relations"]
                     if row["buffer"] == "Z"]
        self.assertTrue(relations)
        self.assertTrue(all(row["producer_stage"] == 0 and
                            row["consumer_stage"] == 1 for row in relations))
        self.assertTrue(chain["stages"][0]["global_writes"])
        self.assertTrue(any(read["source"] == "Z"
                            for read in chain["stages"][1]["global_reads"]))
        plan = compile_tilelang(prim, profile=PROFILE, options=TILED).plan
        stage0 = [a.id for a in plan.actions
                  if a.kind == "npu_task" and
                  a.attrs.get("source_stage_index") == 0]
        stage1 = [a for a in plan.actions
                  if a.kind == "npu_task" and
                  a.attrs.get("source_stage_index") == 1]
        self.assertTrue(stage0 and stage1)
        self.assertTrue(any(any(dep in stage0 for dep in task.deps)
                            for task in stage1))

    def test_staged_contract_negatives_still_reject(self):
        cases = (
            ("fp32_accum", tilelang_projection_staged.projection_fp32_accum_only),
            ("cross_block", tilelang_projection_staged.projection_cross_block_read),
            ("missing_intermediate",
             tilelang_two_stage.two_stage_bad_no_intermediate()),
            ("bad_pairing", tilelang_ffn_scheduled.bad_gate_up_swap()),
        )
        for name, prim in cases:
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    compile_tilelang(prim, profile=PROFILE, options=LUT)


if __name__ == "__main__":
    unittest.main()
