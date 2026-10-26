"""Static-repeat plan expansion: one plan, N independent tasks, one submit.

The B2 capability has to be admitted from the manifest's repeat semantics, not
from a model name, and the expanded plan has to be *proven* equivalent to the N
separate runs it replaces.  These tests use the real compiled QK plan and the
real C verifier/runtime through the mock seam:

* the expanded plan passes the C validator (``rk_plan_create``) and the workspace
  budget check (``rk_workspace_create``);
* it declares one submit carrying N tasks -- the entry merge is not mistaken for
  fewer device submits *within* the group, it is exactly the point of the group;
* running it once produces bit-identical outputs to N runs of the parent plan;
* the admission rules refuse shapes without evidence (K chaining, tiling,
  too many tasks, constants) instead of guessing.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
PLANS = ROOT / "models" / "compiled"
REPEATS = 32


class RepeatExpandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import load_plan
        from rknpu_backend.runtime.ffi import LibRKPlan
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.parent = load_plan(str(PLANS / "tiny_vit_b8_qk.rkplan.json"))

    def _inputs(self, n=REPEATS, seed=20261026):
        rng = np.random.default_rng(seed)
        return [[np.ascontiguousarray(
            rng.standard_normal((16, 16)).astype(np.float16)) for _ in range(2)]
            for _ in range(n)]

    def test_expanded_plan_is_admitted_by_the_c_validator(self):
        from rknpu_backend.passes.repeat_expand import expand_static_repeat
        group = expand_static_repeat(self.parent, REPEATS)
        self.assertEqual(len(group.submits), 1)
        self.assertEqual(len(group.submits[0].task_ids), REPEATS)
        self.assertEqual(group.submits[0].mode, "chain")
        self.assertEqual(group.submits[0].recipe_id, "independent_mm_v1")
        self.assertEqual(group.accounting.task_count, REPEATS)
        self.assertEqual(group.accounting.submit_count, 1)
        # First round: one BO per task (3N), the shape the board probe
        # validated.  Each task's read/write stays in distinct slots.
        self.assertEqual(group.accounting.data_bo_count, 3 * REPEATS)
        # The C validator is the authority: it proves structural independence.
        cplan = self.lib.create_plan(group)
        try:
            self.assertTrue(cplan._ptr)
        finally:
            cplan.destroy()

    def test_expanded_group_matches_n_separate_runs_bitwise(self):
        from rknpu_backend.models import PlanCall
        from rknpu_backend.passes.repeat_expand import expand_static_repeat
        inputs = self._inputs()
        single = PlanCall(self.lib, self.parent, {}, fd=0)
        separate = [[np.zeros((16, 16), dtype=np.float16)] for _ in range(REPEATS)]
        try:
            for i in range(REPEATS):
                single.run_into(inputs[i], separate[i])
        finally:
            single.close()

        group = expand_static_repeat(self.parent, REPEATS)
        grouped = PlanCall(self.lib, group, {}, fd=0)
        produced = [[np.zeros((16, 16), dtype=np.float16)] for _ in range(REPEATS)]
        try:
            report, _ = grouped.run_into(
                [a for repeat in inputs for a in repeat],
                [out for repeat in produced for out in repeat])
        finally:
            grouped.close()
        self.assertEqual(int(report.task_count), REPEATS)
        self.assertEqual(int(report.submit_count), 1)
        self.assertEqual(int(report.upload_calls), 2 * REPEATS)
        self.assertEqual(int(report.download_calls), REPEATS)
        for i in range(REPEATS):
            self.assertTrue(np.array_equal(separate[i][0], produced[i][0]),
                            f"task {i} differs from the separate run")

    def test_expansion_refuses_shapes_without_evidence(self):
        from rknpu_backend import load_plan
        from rknpu_backend.errors import BackendError
        from rknpu_backend.passes.repeat_expand import (MAX_GROUP_TASKS,
                                                        expand_static_repeat)
        with self.assertRaises(BackendError):
            expand_static_repeat(self.parent, MAX_GROUP_TASKS + 1)
        with self.assertRaises(BackendError):
            expand_static_repeat(self.parent, 0)
        # A K-chained / tiled plan is a different recipe: refuse it rather than
        # reusing the independent-group admission.
        for name in ("tiny_vit_b8_gate", "tiny_vit_b8_qkv", "mnist_mlp_b32"):
            path = PLANS / f"{name}.rkplan.json"
            if not path.is_file():
                continue
            plan = load_plan(str(path))
            with self.assertRaises(BackendError, msg=name):
                expand_static_repeat(plan, 2)

    def test_single_repeat_returns_the_parent_unchanged(self):
        from rknpu_backend.passes.repeat_expand import expand_static_repeat
        self.assertIs(expand_static_repeat(self.parent, 1), self.parent)


if __name__ == "__main__":
    unittest.main()
