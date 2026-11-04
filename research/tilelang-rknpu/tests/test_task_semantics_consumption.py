# SPDX-License-Identifier: GPL-3.0-or-later
"""Planner consumes the instantiated target task sequence directly.

The guarded-tail pass emits ordered ``rk_fill``/``rk_copy``/``rk_gemm``/
``rk_store`` operations.  These tests prove that the plan is assembled from
those operations: their operands, regions, order, K chain and store rectangles
are observable in the plan, and task-level corruption is rejected instead of
being silently repaired from shapes.
"""
from __future__ import annotations

import copy
import unittest

import tilelang

import models.tilelang_matmul as tmm
from rknpu_backend import CompileOptions, default_profile
from rknpu_backend.codegen.pipeline import compile_from_tasks
from rknpu_backend.errors import BackendError, RK_E_SHAPE, RK_E_UNSUPPORTED
from rknpu_backend.frontend import rknpu_pass as R

M, K, N = 1, 576, 960
PROFILE = default_profile()
# Canonical plan produced by the guarded-tail source before the task-consumer
# seam was changed.  It freezes the complete execution description.
FROZEN_GUARDED_TASK_PLAN_HASH = (
    "7e544f4c55cca409b1620d05f6446b448cafc8d7be8669dbf39a098f9fc4c892"
)


def options(**extra):
    base = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    base.update(extra)
    return CompileOptions(**base)


def _tasks_and_facts():
    plan, stages, facts, tasks = R.compile_guarded_tail(
        tmm.make_matmul_tiled_tail_prim(M, K, N, bm=1, bn=256),
        profile=PROFILE, options=options())
    return plan, stages, facts, tasks


def _compile_tasks(tasks, facts):
    module = R._minimal_module(facts, name="rknpu-pass")
    facts_dict = {
        "m": facts.logical_m, "k": facts.logical_k, "n": facts.logical_n,
        "tile_m": facts.tile_m, "tile_n": facts.tile_n,
        "order": facts.order, "rounding": facts.rounding,
        "hints": dict(facts.hints),
    }
    return compile_from_tasks(module, tasks, facts_dict, profile=PROFILE,
                              options=options()).plan


def _gemm_tasks(tasks):
    return [task for task in tasks if task["kind"] == "gemm"]


def _copy_before(tasks, gemm_index, destination):
    for index in range(gemm_index - 1, -1, -1):
        task = tasks[index]
        if task.get("kind") == "gemm":
            break
        if task.get("kind") == "copy" and task.get("destination") == destination:
            return task
    raise AssertionError(f"no copy before gemm {gemm_index} to {destination}")


def _swap_first_two_k_groups(tasks):
    """Keep all associated copies with their GEMM but reverse K order."""
    changed = copy.deepcopy(tasks)
    first_gemm = next(i for i, task in enumerate(changed)
                      if task["kind"] == "gemm")
    first_load = None
    for index in range(first_gemm - 1, -1, -1):
        if changed[index]["kind"] == "fill" and \
                changed[index]["destination"] == "Acc16":
            first_load = index
            break
    if first_load is None:
        raise AssertionError("initial accumulator fill not found")
    group0_start = first_load + 1
    group0_gemm = first_gemm
    group0_end = None
    for index in range(group0_gemm + 1, len(changed)):
        task = changed[index]
        if task["kind"] == "copy" and task["source"] == "Cl":
            group0_end = index
            break
    if group0_end is None:
        raise AssertionError("first writeback not found")
    group1_start = group0_end + 1
    group1_gemm = next(i for i in range(group1_start, len(changed))
                       if changed[i]["kind"] == "gemm")
    group1_end = None
    for index in range(group1_gemm + 1, len(changed)):
        task = changed[index]
        if task["kind"] == "copy" and task["source"] == "Cl":
            group1_end = index
            break
    if group1_end is None:
        raise AssertionError("second writeback not found")
    return (changed[:group0_start]
            + changed[group1_start:group1_end + 1]
            + changed[group0_start:group0_end + 1]
            + changed[group1_end + 1:])


class TaskConsumptionTests(unittest.TestCase):
    def test_task_geometry_order_and_region_drive_the_plan(self):
        plan, _, facts, tasks = _tasks_and_facts()
        gemms = _gemm_tasks(tasks)
        npu = [action for action in plan.actions if action.kind == "npu_task"]
        self.assertEqual(len(npu), len(gemms))
        self.assertEqual([(action.template_id, action.attrs["k0"],
                           action.attrs["kt"]) for action in npu],
                         [(("co.mm.f16.v1" if index % 3 == 0
                            else "co.mm_acc.f16.v1"), task["k0"],
                           min(task["kt"], facts.logical_k - task["k0"]))
                          for index, task in enumerate(gemms)])
        # K-chain dependencies are the target sequence's accumulation chain.
        for index, action in enumerate(npu):
            if index % 3 == 0:
                continue
            self.assertIn(npu[index - 1].id, action.deps)
            self.assertEqual(action.inputs[2], npu[index - 1].outputs[0])

        # A/B upload offsets come from the rk_copy source regions, not from
        # the logical output shape.
        upload_by_view = {}
        for action in plan.actions:
            if action.kind == "upload":
                for view in action.outputs:
                    upload_by_view[view] = action
        for index, task in enumerate(gemms):
            task_actions = [a for a in plan.actions if a.kind == "npu_task"]
            action = task_actions[index]
            a_view, b_view = action.inputs[:2]
            a_upload = upload_by_view[a_view]
            b_upload = upload_by_view[b_view]
            a_copy = _copy_before(tasks, tasks.index(task), task["tiles"][0])
            b_copy = _copy_before(tasks, tasks.index(task), task["tiles"][1])
            self.assertEqual(a_upload.attrs["src_row_offset"],
                             a_copy["source_region"]["row"])
            self.assertEqual(a_upload.attrs["src_col_offset"],
                             a_copy["source_region"]["col"])
            self.assertEqual(b_upload.attrs["src_row_offset"],
                             b_copy["source_region"]["row"])
            self.assertEqual(b_upload.attrs["src_col_offset"],
                             b_copy["source_region"]["col"])

        stores = [task for task in tasks if task["kind"] == "store"]
        self.assertEqual([(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in plan.publish],
                         [(task["m0"], task["n0"], task["rows"], task["cols"])
                          for task in stores])

    def test_the_previous_execution_signature_is_preserved(self):
        plan, _, _, _ = _tasks_and_facts()
        self.assertEqual(plan.plan_hash, FROZEN_GUARDED_TASK_PLAN_HASH)
        self.assertEqual(plan.accounting.task_count, 12)
        self.assertEqual(plan.accounting.submit_count, 4)
        self.assertEqual(plan.accounting.data_bo_count, 20)
        self.assertEqual([(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in plan.publish],
                         [(0, 0, 1, 256), (0, 256, 1, 256),
                          (0, 512, 1, 256), (0, 768, 1, 192)])
        self.assertEqual(
            [r.to_dict()["contract"]["accumulation"]
             for r in plan.numeric_requirements],
            ["ordered_k_tiles_with_f16_writeback"])


class TaskMutationTests(unittest.TestCase):
    def setUp(self):
        self.plan, _, self.facts, self.tasks = _tasks_and_facts()

    def test_a_duplicate_gemm_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        first = next(i for i, task in enumerate(bad) if task["kind"] == "gemm")
        duplicate = copy.deepcopy(bad[first])
        duplicate["ordinal"] = len(bad)
        bad.insert(first + 1, duplicate)
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("GEMM consumes a Cl tile with no accumulator load",
                      caught.exception.diagnostics[0].message)

    def test_a_missing_gemm_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        first = next(i for i, task in enumerate(bad) if task["kind"] == "gemm")
        del bad[first]
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("unfinished accumulator chain",
                      caught.exception.diagnostics[0].message)

    def test_a_wrong_gemm_operand_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        gemm = next(task for task in bad if task["kind"] == "gemm")
        gemm["operands"] = ["A", "NOT_B", "Y"]
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("names operands the binding carrier does not bind",
                      caught.exception.diagnostics[0].message)

    def test_a_wrong_staging_operand_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        gemm_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "gemm")
        a_copy = _copy_before(bad, gemm_index, "As")
        a_copy["source"] = "NOT_A"
        a_copy["operands"] = ["NOT_A", "As"]
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("does not copy the bound input into the GEMM tile",
                      caught.exception.diagnostics[0].message)

    def test_a_wrong_staging_region_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        gemm_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "gemm")
        a_copy = _copy_before(bad, gemm_index, "As")
        a_copy["source_region"]["col"] += 256
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("does not copy the bound input into the GEMM A tile",
                      caught.exception.diagnostics[0].message)

    def test_an_initial_clear_after_the_accumulator_load_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        fill_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "fill"
                          and task["destination"] == "Acc16")
        load_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "copy"
                          and task["destination"] == "Cl")
        gemm_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "gemm")
        self.assertLess(fill_index, load_index)
        initial = bad.pop(fill_index)
        if gemm_index > fill_index:
            gemm_index -= 1
        bad.insert(gemm_index, initial)
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("accumulator load has no producer state",
                      caught.exception.diagnostics[0].message)

    def test_a_staging_copy_followed_by_a_full_zero_fill_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        gemm_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "gemm")
        copy_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "copy"
                          and task["destination"] == "As")
        fill = next(task for task in bad
                    if task["kind"] == "fill"
                    and task["destination"] == "As")
        # A later full zero fill erases the staged value.  The region is fully
        # covered and zero-filled again, but the GEMM's logical data no longer
        # comes from the staging copy.
        bad.insert(copy_index + 1, copy.deepcopy(fill))
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("does not have a logical value over a zero-filled physical tile",
                      caught.exception.diagnostics[0].message)

    def test_a_tail_fill_covering_only_the_copy_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        tail_gemm = next(i for i, task in enumerate(bad)
                         if task["kind"] == "gemm" and task["k0"] == 512)
        tail_copy = bad.index(_copy_before(bad, tail_gemm, "As"))
        fill_index = next(i for i in range(tail_copy - 1, -1, -1)
                          if bad[i]["kind"] == "fill"
                          and bad[i]["destination"] == "As")
        # The source has a full 1x256 zero fill and a 1x64 tail copy.  Making
        # the fill exactly the copy region leaves stale physical K bytes for
        # the GEMM's full 1x256 read and must be rejected.
        bad[fill_index]["region"] = copy.deepcopy(
            bad[tail_copy]["destination_region"])
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("zero-filled physical tile",
                      caught.exception.diagnostics[0].message)

    def test_a_partial_initial_clear_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        initial = next(task for task in bad
                       if task["kind"] == "fill"
                       and task["destination"] == "Acc16")
        initial["region"] = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("must read the complete zero-filled running state",
                      caught.exception.diagnostics[0].message)

    def test_a_missing_operand_fill_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        index = next((i for i, task in enumerate(bad)
                      if task["kind"] == "fill" and task["destination"] == "As"),
                     None)
        self.assertIsNotNone(index)
        del bad[index]
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("zero-filled write region",
                      caught.exception.diagnostics[0].message)

    def test_a_missing_store_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        index = next((i for i, task in enumerate(bad)
                      if task["kind"] == "store"), None)
        self.assertIsNotNone(index)
        del bad[index]
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("needs exactly one store",
                      caught.exception.diagnostics[0].message)

    def test_an_operand_fill_after_its_staging_copy_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        gemm_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "gemm")
        fill_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "fill"
                          and task["destination"] == "As")
        copy_index = next(i for i, task in enumerate(bad)
                          if task["kind"] == "copy"
                          and task["destination"] == "As")
        self.assertLess(fill_index, copy_index)
        fill = bad.pop(fill_index)
        if copy_index > fill_index:
            copy_index -= 1
        bad.insert(copy_index + 1, fill)
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("zero-filled write region",
                      caught.exception.diagnostics[0].message)

    def test_a_store_before_the_final_writeback_is_rejected(self):
        bad = copy.deepcopy(self.tasks)
        store_index = next(i for i, task in enumerate(bad)
                           if task["kind"] == "store")
        store = bad.pop(store_index)
        last_gemm = max(i for i, task in enumerate(bad)
                        if task["kind"] == "gemm")
        bad.insert(last_gemm + 1, store)
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("store lacks its FP16 accumulator writeback",
                      caught.exception.diagnostics[0].message)

    def test_an_illegal_accumulation_order_is_rejected(self):
        bad = _swap_first_two_k_groups(self.tasks)
        with self.assertRaises(BackendError) as caught:
            _compile_tasks(bad, self.facts)
        self.assertIn("do not form an ordered partition",
                      caught.exception.diagnostics[0].message)


if __name__ == "__main__":
    unittest.main()
