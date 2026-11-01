# SPDX-License-Identifier: GPL-3.0-or-later
"""Group partition, intra-group order, and the explicit submit recipe field."""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import CompileOptions, compile_tilelang, default_profile, load_plan, save_plan
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.schedule import build_submits
from rknpu_backend.plan import _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
REAL = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
FULL = (256, 576, 3072)


def build(order, bm=128, group_n=0, shape=FULL):
    return compile_tilelang(
        make_linear(*shape, bm=bm, bn=256), profile=PROFILE,
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=False,
                               linear_task_order=order,
                               linear_group_n_tiles=group_n)).plan


def draft_from(plan, actions):
    """Rebuild a DraftPlan with new actions, keeping every other field."""
    import dataclasses

    from rknpu_backend.passes.dag import DraftPlan

    names = {f.name for f in dataclasses.fields(DraftPlan)} - {"actions"}
    values = {name: getattr(plan, name) for name in names
              if hasattr(plan, name)}
    return DraftPlan(actions=tuple(actions), **values)


def tasks_of(plan):
    return [a for a in plan.actions if a.kind == "npu_task"]


class OrderControlTests(unittest.TestCase):
    def test_grouped_and_weight_stationary_share_geometry_and_partition(self):
        grouped = build("grouped_linear")
        wstat = build("weight_stationary")
        self.assertEqual(len(tasks_of(grouped)), len(tasks_of(wstat)))
        self.assertEqual([len(s.task_ids) for s in grouped.submits],
                         [len(s.task_ids) for s in wstat.submits])
        self.assertEqual(len(grouped.submits), 2)
        # only the intra-group order differs
        self.assertEqual([t.attrs["k_index"] for t in tasks_of(grouped)[:3]], [0, 1, 2])
        self.assertEqual([t.attrs["k_index"] for t in tasks_of(wstat)[:3]], [0, 0, 0])
        for plan in (grouped, wstat):
            self.assertEqual(sum(1 for t in tasks_of(plan)
                                 if t.attrs.get("weight_reuse")), 0)

    def test_group_partition_knob_changes_only_the_partition(self):
        auto = build("weight_stationary")
        five = build("weight_stationary", group_n=5)
        self.assertEqual([len(s.task_ids) for s in auto.submits], [60, 12])
        self.assertEqual([len(s.task_ids) for s in five.submits], [30, 30, 12])
        self.assertEqual(len(tasks_of(auto)), len(tasks_of(five)))
        # same intra-group order and same per-tile K chains in both
        self.assertEqual([t.attrs["k_index"] for t in tasks_of(auto)[:6]],
                         [t.attrs["k_index"] for t in tasks_of(five)[:6]])

    def test_a_partition_that_would_overflow_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            build("weight_stationary", group_n=11)
        self.assertIn("64-task submit bound",
                      caught.exception.diagnostics[0].message)


@unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
class RecipeFieldTests(unittest.TestCase):
    def test_the_planner_states_the_recipe_on_the_task(self):
        # The grouped policies state the recipe on the task; the legacy mm:
        # prefixed path keeps the prefix fallback so old plans stay valid.
        for order in ("grouped_linear", "weight_stationary"):
            with self.subTest(order=order):
                plan = build(order)
                declared = {t.attrs.get("submit_recipe") for t in tasks_of(plan)}
                self.assertEqual(declared, {"multi_k_acc_v1"})
                self.assertEqual({s.recipe_id for s in plan.submits},
                                 {"multi_k_acc_v1"})
        legacy = build("baseline", bm=256)
        self.assertEqual({s.recipe_id for s in legacy.submits},
                         {"linear_k_acc_v1"})
        self.assertEqual({t.attrs.get("submit_recipe") for t in tasks_of(legacy)},
                         {None})

    def test_renaming_the_group_does_not_change_the_recipe(self):
        # Same partition, different label: the recipe must still come from the
        # declared field, not from the group name.
        plan = build("weight_stationary")
        actions = list(plan.actions)
        tasks = tasks_of(plan)
        labels = {}
        for action in tasks:
            labels.setdefault(action.attrs["submit_group"], len(labels))
        for action in tasks:
            actions[action.id] = replace(
                action, attrs={**dict(action.attrs),
                               "submit_group": "mm:%d" % labels[
                                   action.attrs["submit_group"]]})
        draft = draft_from(plan, actions)
        submits = build_submits(draft, options=CompileOptions(
            numeric_mode="rk_f16_tiled", submit_mode="chain"))
        self.assertEqual({s.recipe_id for s in submits}, {"multi_k_acc_v1"})
        self.assertEqual([len(s.task_ids) for s in submits], [60, 12])

    def test_a_declared_recipe_that_does_not_match_the_chain_is_refused(self):
        # A 60-task mm/mm_acc group is not a linear_k_acc_v1 chain, while it is
        # a legitimate multi_k_acc_v1 chain; the plan-level matcher decides.
        from rknpu_backend.plan import _recipe_matches

        plan = build("weight_stationary")
        templates = [t.template_id for t in tasks_of(plan)[:60]]
        self.assertFalse(_recipe_matches("linear_k_acc_v1", templates))
        self.assertTrue(_recipe_matches("multi_k_acc_v1", templates))

    def test_recipe_and_flags_survive_save_and_load(self):
        import tempfile

        plan = build("weight_stationary_reuse", group_n=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual({s.recipe_id for s in loaded.submits}, {"multi_k_acc_v1"})
        self.assertEqual(sum(1 for t in tasks_of(loaded)
                             if t.attrs.get("weight_reuse")), 36)
        _validate_plan(loaded)


if __name__ == "__main__":
    unittest.main()
