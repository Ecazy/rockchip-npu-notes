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

    def _bypass_and_expand(self, nrepeat):
        """Expand while skipping the Python admission limits.

        The C verifier is the authority: a caller that does not run this module
        (or patches its limits) must still be refused.  Raising the module
        constants is exactly that caller.
        """
        from rknpu_backend.passes import repeat_expand
        saved = (repeat_expand.MAX_GROUP_TASKS,)
        repeat_expand.MAX_GROUP_TASKS = max(nrepeat, repeat_expand.MAX_GROUP_TASKS)
        try:
            return repeat_expand.expand_static_repeat(self.parent, nrepeat)
        finally:
            (repeat_expand.MAX_GROUP_TASKS,) = saved

    def test_c_verifier_refuses_a_task_count_outside_the_domain(self):
        """64 independent 16x16x16 tasks: structurally fine, still refused."""
        from rknpu_backend.errors import BackendError
        group = self._bypass_and_expand(64)
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(group)
        self.assertIn("task count", str(caught.exception))

    def test_c_verifier_refuses_geometry_outside_the_domain(self):
        """8x8x8 / 32x32x32 are structurally independent but unmeasured."""
        import dataclasses
        from rknpu_backend.errors import BackendError
        from rknpu_backend.plan import Buffer, View
        from rknpu_backend.passes.repeat_expand import expand_static_repeat

        base = expand_static_repeat(self.parent, 4)

        def reshape(rows, cols, phys_rows, phys_cols, atom, w_rows, w_phys_rows):
            # Rewrite every view/buffer consistently so the *only* thing wrong is
            # the geometry the C verifier gates on.
            views = []
            buffers = []
            for v in base.views:
                tensor = dataclasses.replace(
                    v.tensor,
                    rows=w_rows if v.tensor.layout == "co.weight.f16.v1" else rows,
                    cols=w_rows if v.tensor.layout == "co.weight.f16.v1" else cols,
                    physical_rows=(w_phys_rows
                                   if v.tensor.layout == "co.weight.f16.v1"
                                   else phys_rows),
                    physical_cols=phys_cols, atom=atom,
                    bytes=None)
                views.append(dataclasses.replace(v, tensor=tensor))
            return views

        # A canonical (but unmeasured) 8x8x8 geometry: cube physical rows are
        # round_up(max(rows,12),4), weight physical rows round_up(rows,32).
        views = []
        for v in base.views:
            weight = v.tensor.layout == "co.weight.f16.v1"
            rows = 8
            cols = 8
            phys_rows = 32 if weight else 12
            views.append(dataclasses.replace(
                v, tensor=dataclasses.replace(
                    v.tensor, rows=rows, cols=cols, physical_rows=phys_rows,
                    physical_cols=32, bytes=phys_rows * 32 * 2),
                valid_region=(0, rows, 0, cols)))
        # The publish tiles were sized for the 16x16 geometry; keep the plan
        # internally consistent so the *only* remaining objection is the geometry.
        publish = tuple(dataclasses.replace(p, rows=8, cols=8)
                        for p in base.publish)
        bindings = tuple(dataclasses.replace(b, shape=(8, 8))
                         for b in base.bindings)
        actions = tuple(
            dataclasses.replace(a, attrs=dict(a.attrs, src_row_stride=8))
            if a.kind == "upload" else a for a in base.actions)
        tampered = dataclasses.replace(base, views=tuple(views),
                                       publish=publish, bindings=bindings,
                                       actions=actions)
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(tampered)
        self.assertIn("geometry", str(caught.exception))

    def test_single_repeat_returns_the_parent_unchanged(self):
        from rknpu_backend.passes.repeat_expand import expand_static_repeat
        self.assertIs(expand_static_repeat(self.parent, 1), self.parent)


if __name__ == "__main__":
    unittest.main()
