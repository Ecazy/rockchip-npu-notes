"""One admission contract for explicit M/N declarations x linear task policy."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                           RK_E_UNSUPPORTED, compile_rkir, compile_tilelang,
                           default_profile, load_plan, matmul_module, save_plan)
from rknpu_backend.passes.verify import verify_plan
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tir_utils import tilelang_available

PROFILE = default_profile()
M, K, N = 256, 576, 3072
LINEAR = ("scheduled_baseline", "merged", "interleaved",
          "interleaved_reuse")
DECLARED_KEYS = ("source_schedule", "source_k_rounding",
                 "source_mn_tiling", "applied_schedule")


def schedule(tile_m=128, tile_n=128, order="m_major"):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": order, "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def options(order, submit_mode="chain"):
    return CompileOptions(numeric_mode="rk_f16_tiled",
                          allow_approx_lut=False,
                          linear_task_order=order,
                          submit_mode=submit_mode)


def module(declared=True, rounding="explicit_per_k_fp16", tile_m=128,
           tile_n=128, order="m_major"):
    mod = matmul_module(m=M, k=K, n=N)
    if not declared:
        return mod
    attrs = dict(mod.ops[0].attrs)
    attrs["source_schedule"] = schedule(tile_m, tile_n, order)
    if rounding is not None:
        attrs["source_k_rounding"] = rounding
    return replace(mod, ops=(replace(mod.ops[0], attrs=attrs),))


def compile_(policy, **kwargs):
    return compile_rkir(module(**kwargs), profile=PROFILE,
                        options=options(policy))


def npu_tasks(plan):
    return [action for action in plan.actions if action.kind == "npu_task"]


def publication(plan):
    return sorted((p.dst_row, p.dst_col, p.rows, p.cols) for p in plan.publish)


class LinearScheduleContractTests(unittest.TestCase):
    def test_undeclared_sources_stay_on_the_historical_path(self):
        # Byte compatibility: a source without an explicit declaration must
        # not gain schedule attrs, otherwise every frozen plan identity (the
        # decode merged evidence in particular) would move.
        for order in ("baseline",) + LINEAR:
            with self.subTest(policy=order):
                plan = compile_(order, declared=False).plan
                # The classic builder always records applied_schedule; the
                # linear builder must not gain it without a declaration.
                for task in npu_tasks(plan):
                    if order == "baseline":
                        # The classic builder writes the keys with None when
                        # nothing was declared; only absence of a value counts.
                        self.assertIsNone(task.attrs.get("source_schedule"))
                        self.assertIsNone(task.attrs.get("source_k_rounding"))
                    else:
                        for key in DECLARED_KEYS:
                            self.assertNotIn(key, task.attrs)
                self.assertEqual(plan.accounting.task_count, 36)
                self.assertEqual(plan.accounting.submit_count,
                                 1 if order in ("merged", "interleaved",
                                                "interleaved_reuse") else 12)

    def test_declared_schedule_is_recorded_not_dropped(self):
        for order in LINEAR:
            with self.subTest(policy=order):
                bare = compile_(order, declared=False).plan
                read = compile_(order, declared=True, tile_m=256,
                                tile_n=256).plan
                self.assertNotEqual(bare.plan_hash, read.plan_hash)
                tasks = npu_tasks(read)
                self.assertTrue(all(t.attrs["applied_schedule"]
                                    == schedule(256, 256) for t in tasks))
                self.assertEqual(read.accounting.task_count, 36)

    def test_nondefault_mn_and_linear_policy_both_take_effect(self):
        plan = compile_("scheduled_baseline", declared=True, tile_m=128,
                        tile_n=128).plan
        self.assertEqual(plan.accounting.task_count, 144)
        self.assertEqual(plan.accounting.submit_count, 48)
        self.assertEqual(publication(plan),
                         sorted((m0, n0, 128, 128)
                                for m0 in range(0, M, 128)
                                for n0 in range(0, N, 128)))
        for task in npu_tasks(plan):
            self.assertEqual(task.attrs["applied_schedule"],
                             schedule(128, 128))
            self.assertEqual(task.attrs["linear_task_policy"],
                             "scheduled_baseline")
            self.assertEqual(task.attrs["k_count"], 3)
        # The same declaration on the baseline path keeps the same geometry,
        # so the policy - not the declaration - is what the origin records.
        classic = compile_("baseline", declared=True, tile_m=128,
                           tile_n=128).plan
        self.assertEqual(classic.accounting.task_count, 144)
        self.assertTrue(all("linear_task_policy" not in task.attrs
                            for task in npu_tasks(classic)))

    def test_declared_traversal_order_survives_the_linear_policy(self):
        # The declaration owns the output-tile traversal; the linear policy owns
        # emission order and submit grouping.  Both must be visible.
        row = compile_("scheduled_baseline", declared=True, tile_m=128,
                       tile_n=128, order="m_major").plan
        col = compile_("scheduled_baseline", declared=True, tile_m=128,
                       tile_n=128, order="n_major").plan
        self.assertEqual(row.accounting.task_count, col.accounting.task_count)
        self.assertEqual(row.accounting.submit_count, col.accounting.submit_count)
        self.assertNotEqual([(p.dst_row, p.dst_col) for p in row.publish],
                            [(p.dst_row, p.dst_col) for p in col.publish])
        self.assertEqual(len({json.dumps(t.attrs["applied_schedule"],
                                         sort_keys=True)
                              for t in npu_tasks(col)}), 1)
        self.assertIn("n_major",
                      json.dumps(npu_tasks(col)[0].attrs["applied_schedule"]))

    def test_linear_policy_order_is_visible_in_task_order(self):
        merged = compile_("merged", declared=True, tile_m=256,
                          tile_n=256).plan
        inter = compile_("interleaved", declared=True, tile_m=256,
                         tile_n=256).plan
        merged_k = [t.attrs["k0"] for t in npu_tasks(merged)[:3]]
        inter_k = [t.attrs["k0"] for t in npu_tasks(inter)[:3]]
        self.assertEqual(merged_k, [0, 256, 512])
        self.assertEqual(inter_k, [0, 0, 0])
        self.assertEqual(merged.accounting.submit_count, 1)
        self.assertEqual(inter.accounting.submit_count, 1)
        self.assertEqual(merged.accounting.task_count,
                         inter.accounting.task_count)

    def test_conflicting_numeric_declaration_is_rejected_on_every_branch(self):
        # F1 regression: the policy must not move the numeric gate.
        for order in ("baseline",) + LINEAR:
            with self.subTest(policy=order):
                with self.assertRaises(BackendError) as caught:
                    compile_(order, declared=True,
                             rounding="fp32_final_narrowing")
                self.assertEqual(caught.exception.code,
                                 RK_E_NUMERIC_CONTRACT)

    def test_missing_rounding_is_rejected_on_every_branch(self):
        for order in ("baseline",) + LINEAR:
            with self.subTest(policy=order):
                with self.assertRaises(BackendError) as caught:
                    compile_(order, declared=True, rounding=None)
                self.assertEqual(caught.exception.code,
                                 RK_E_NUMERIC_CONTRACT)

    def test_unknown_field_and_bad_order_are_rejected_on_every_branch(self):
        broken = dict(schedule(256, 256))
        broken["made_up"] = 1234
        for order in LINEAR:
            with self.subTest(policy=order, case="unknown field"):
                mod = module(declared=True, tile_m=256, tile_n=256)
                attrs = dict(mod.ops[0].attrs)
                attrs["source_schedule"] = broken
                mod = replace(mod, ops=(replace(mod.ops[0], attrs=attrs),))
                with self.assertRaises(BackendError) as caught:
                    compile_rkir(mod, profile=PROFILE, options=options(order))
                self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)
            with self.subTest(policy=order, case="bad order"):
                with self.assertRaises(BackendError) as caught:
                    compile_(order, declared=True, order="k_major")
                self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)

    def test_combination_beyond_one_submit_is_rejected_with_the_boundary(self):
        for order in ("merged", "interleaved", "interleaved_reuse"):
            with self.subTest(policy=order):
                with self.assertRaises(BackendError) as caught:
                    compile_(order, declared=True, tile_m=128, tile_n=128)
                error = caught.exception
                diagnostic = error.diagnostics[0]
                self.assertEqual(error.code, RK_E_SHAPE)
                self.assertIn("64-task", diagnostic.message)
                self.assertEqual(diagnostic.actual["tasks_per_submit"], 144)
                self.assertEqual(diagnostic.actual["output_tiles"], 48)
                self.assertEqual(diagnostic.required["max_tasks_per_submit"], 64)
                self.assertIn("scheduled_baseline",
                              diagnostic.required["policies_that_fit"])

    def test_nominal_tile_wider_than_extent_is_clipped_not_rejected(self):
        # M = 1: the declaration names a 256x256 block, the only row is a tail.
        # The baseline path already clipped it; every policy must agree.
        for policy in ("baseline", "scheduled_baseline", "merged",
                       "interleaved", "interleaved_reuse"):
            with self.subTest(policy=policy):
                module_one_row = matmul_module(m=1, k=K, n=N)
                attrs = dict(module_one_row.ops[0].attrs)
                attrs["source_schedule"] = schedule(256, 256)
                attrs["source_k_rounding"] = "explicit_per_k_fp16"
                module_one_row = replace(
                    module_one_row,
                    ops=(replace(module_one_row.ops[0], attrs=attrs),))
                result = compile_rkir(module_one_row, profile=PROFILE,
                                      options=options(policy))
                plan = result.plan
                self.assertEqual(plan.accounting.task_count, 36)
                self.assertEqual(publication(plan),
                                 [(0, n0, 1, 256)
                                  for n0 in range(0, N, 256)])
                # the declaration is kept as declared, not rewritten
                for task in npu_tasks(plan):
                    self.assertEqual(task.attrs["applied_schedule"],
                                     schedule(256, 256))
                effective = result.report["schedule"]["effective"]
                self.assertEqual(effective["publish_rects"], [[1, 256]])
                self.assertTrue(effective["clipped_from_declared"])
                self.assertEqual(plan.accounting.submit_count,
                                 12 if policy != "merged"
                                 and policy != "interleaved"
                                 and policy != "interleaved_reuse" else 1)

    def test_256x192_declaration_composes_with_linear_policies(self):
        for policy, submits in (("scheduled_baseline", 16), ("merged", 1),
                                ("interleaved", 1), ("interleaved_reuse", 1)):
            with self.subTest(policy=policy):
                plan = compile_(policy, declared=True, tile_m=256,
                                tile_n=192).plan
                self.assertEqual(plan.accounting.task_count, 48)
                self.assertEqual(plan.accounting.submit_count, submits)
                self.assertEqual(publication(plan),
                                 [(0, n0, 256, 192)
                                  for n0 in range(0, N, 192)])
                for task in npu_tasks(plan):
                    self.assertEqual(task.attrs["applied_schedule"],
                                     schedule(256, 192))

    def test_unsupported_declarations_are_still_rejected(self):
        # Beyond the 1..256 block limit.
        with self.assertRaises(BackendError) as caught:
            compile_("merged", declared=True, tile_m=512, tile_n=256)
        self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)
        # Legal blocks whose submit grouping cannot fit one submit.
        with self.assertRaises(BackendError) as caught:
            compile_("merged", declared=True, tile_m=256, tile_n=64)
        self.assertEqual(caught.exception.code, RK_E_SHAPE)
        self.assertEqual(caught.exception.diagnostics[0].actual
                         ["tasks_per_submit"], 144)

    def test_save_load_reverify_keeps_the_combination(self):
        import tempfile
        from pathlib import Path

        plan = compile_("scheduled_baseline", declared=True, tile_m=128,
                        tile_n=128).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combo.rkplan.json"
            save_plan(plan, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.accounting.task_count,
                         plan.accounting.task_count)
        self.assertEqual(publication(loaded), publication(plan))
        for before, after in zip(npu_tasks(plan), npu_tasks(loaded)):
            self.assertEqual(after.attrs["applied_schedule"],
                             before.attrs["applied_schedule"])
            self.assertEqual(after.attrs["linear_task_policy"],
                             before.attrs["linear_task_policy"])
        verify_plan(loaded)

    def test_explain_separates_requirement_strategy_and_policy(self):
        report = compile_("scheduled_baseline", declared=True, tile_m=128,
                          tile_n=128).report
        section = report["schedule"]
        self.assertEqual(section["declared"],
                         [__import__("json").dumps(schedule(128, 128),
                                                   sort_keys=True)])
        self.assertEqual(section["applied"], section["declared"])
        self.assertEqual(section["linear_task_policy"],
                         ["scheduled_baseline"])
        self.assertEqual(report["source"]["source_schedule"],
                         schedule(128, 128))
        self.assertEqual(report["counts"]["tasks"], 144)
        self.assertEqual(report["counts"]["submits"], 48)


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class TileLangLinearScheduleContractTests(unittest.TestCase):
    def _prim(self, bm, bn, order="m_major"):
        from models.tilelang_matmul import make_matmul_tiled_tail_prim
        return make_matmul_tiled_tail_prim(M, K, N, bm=bm, bn=bn,
                                           order=order)

    def test_real_tilelang_declaration_composes_with_a_linear_policy(self):
        result = compile_tilelang(self._prim(128, 128), profile=PROFILE,
                                  options=options("scheduled_baseline"))
        plan = result.plan
        self.assertEqual(plan.accounting.task_count, 144)
        self.assertEqual(plan.accounting.submit_count, 48)
        self.assertEqual(publication(plan),
                         sorted((m0, n0, 128, 128)
                                for m0 in range(0, M, 128)
                                for n0 in range(0, N, 128)))
        section = result.report["schedule"]
        self.assertTrue(section["declared"])
        self.assertTrue(section["source_geometry"])
        self.assertEqual(section["applied"], section["declared"])
        self.assertEqual(section["linear_task_policy"],
                         ["scheduled_baseline"])

    def test_real_tilelang_declaration_beyond_the_bound_is_rejected(self):
        for order in ("merged", "interleaved"):
            with self.subTest(policy=order):
                with self.assertRaises(BackendError) as caught:
                    compile_tilelang(self._prim(128, 128), profile=PROFILE,
                                     options=options(order))
                self.assertEqual(caught.exception.code, RK_E_SHAPE)
                self.assertIn("64-task",
                              caught.exception.diagnostics[0].message)

    def test_real_tilelang_undeclared_source_keeps_its_history(self):
        from models.tilelang_matmul import make_matmul_prim
        plan = compile_tilelang(make_matmul_prim(1, K, N), profile=PROFILE,
                                options=options("merged")).plan
        self.assertEqual(plan.accounting.task_count, 36)
        self.assertEqual(plan.accounting.submit_count, 1)
        for task in npu_tasks(plan):
            for key in DECLARED_KEYS:
                self.assertNotIn(key, task.attrs)


if __name__ == "__main__":
    unittest.main()