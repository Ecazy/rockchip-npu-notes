# SPDX-License-Identifier: GPL-3.0-or-later
"""Admission for the linear task-order policies, and M=128 in particular.

The policies used to be gated on a hard-coded shape list (M in {1, 256},
K=576, N=3072).  That list recorded where the first board evidence came from;
it was never a condition of the lowering.  What actually has to hold is checked
here: one matmul in B=[K,N] kn form, a real K chain for the grouped policies,
every task of the policy inside one submit, and the primitive tile bound.  These
tests pin the conditions, not the old shape list.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                           explain_plan, load_plan, matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tir_utils import tilelang_available

PROFILE = default_profile()
GROUPED = ("merged", "interleaved", "interleaved_reuse")


def options(policy, **extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False,
                "linear_task_order": policy}
    settings.update(extra)
    return CompileOptions(**settings)


def schedule(tile_m=256, tile_n=256, order="m_major"):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": order, "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def declared(m=128, k=576, n=3072, tile_m=256, tile_n=256,
             rounding="explicit_per_k_fp16"):
    module = matmul_module(m=m, k=k, n=n)
    attrs = dict(module.ops[0].attrs)
    attrs["source_schedule"] = schedule(tile_m, tile_n)
    if rounding is not None:
        attrs["source_k_rounding"] = rounding
    return replace(module, ops=(replace(module.ops[0], attrs=attrs),))


def tasks(plan):
    return [a for a in plan.actions if a.kind == "npu_task"]


class ShapeAdmissionTests(unittest.TestCase):
    def test_target_shape_is_admitted_and_merge_really_happens(self):
        plan = compile_rkir(matmul_module(m=128, k=576, n=3072),
                            profile=PROFILE,
                            options=options("merged")).plan
        emitted = tasks(plan)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(len(plan.submits), 1)
        self.assertEqual(plan.submits[0].recipe_id, "multi_k_acc_v1")
        self.assertEqual([a.attrs["k0"] for a in emitted[:3]], [0, 256, 512])
        self.assertEqual({a.attrs["k_count"] for a in emitted}, {3})
        self.assertEqual(sorted({a.template_id for a in emitted}),
                         ["co.mm.f16.v1", "co.mm_acc.f16.v1"])
        self.assertEqual(sorted({(plan.views[a.outputs[0]].tensor.rows,
                                  plan.views[a.outputs[0]].tensor.cols)
                                 for a in emitted}), [(128, 256)])
        self.assertEqual(len(plan.logical_ops), 1)
        self.assertEqual(len(plan.numeric_requirements), 1)

    def test_neighbouring_m_follow_the_same_conditions(self):
        # Not a patch for 128: the whole band that fits one submit is admitted,
        # and the boundary is the 64-task submit bound, not a shape list.
        for m in (1, 64, 128, 192, 256):
            with self.subTest(m=m):
                for policy in GROUPED:
                    plan = compile_rkir(matmul_module(m=m, k=576, n=3072),
                                        profile=PROFILE,
                                        options=options(policy)).plan
                    self.assertEqual(plan.accounting.task_count, 36)
                    self.assertEqual(len(plan.submits), 1)
                    self.assertEqual(plan.submits[0].recipe_id,
                                     "multi_k_acc_v1")

    def test_baseline_policies_keep_their_own_grouping(self):
        for policy in ("baseline", "scheduled_baseline"):
            with self.subTest(policy=policy):
                plan = compile_rkir(matmul_module(m=128, k=576, n=3072),
                                    profile=PROFILE,
                                    options=options(policy)).plan
                self.assertEqual(plan.accounting.task_count, 36)
                self.assertEqual(len(plan.submits), 12)
                self.assertEqual({s.recipe_id for s in plan.submits},
                                 {"linear_k_acc_v1"})

    def test_grouped_policy_needs_a_real_k_chain(self):
        # One 256-wide slice has no accumulation step, so the grouped recipe
        # cannot describe it; baseline plans it as a single plain matmul.
        for policy in GROUPED:
            with self.subTest(policy=policy):
                with self.assertRaises(BackendError) as caught:
                    compile_rkir(matmul_module(m=128, k=256, n=3072),
                                 profile=PROFILE, options=options(policy))
                diagnostic = caught.exception.diagnostics[0]
                self.assertIn("K-accumulation chain", diagnostic.message)
                self.assertEqual(diagnostic.required["k_greater_than"], 256)
        for policy in ("baseline", "scheduled_baseline"):
            with self.subTest(policy=policy, note="single slice is legal"):
                plan = compile_rkir(matmul_module(m=128, k=256, n=3072),
                                    profile=PROFILE,
                                    options=options(policy)).plan
                # One K slice, twelve output tiles: twelve plain matmuls, no
                # accumulation step anywhere.
                self.assertEqual(plan.accounting.task_count, 12)
                self.assertEqual({a.template_id for a in tasks(plan)},
                                 {"co.mm.f16.v1"})

    def test_submit_bound_still_rejects_what_cannot_fit(self):
        with self.assertRaises(BackendError) as caught:
            compile_rkir(matmul_module(m=300, k=576, n=3072), profile=PROFILE,
                         options=options("merged"))
        diagnostic = caught.exception.diagnostics[0]
        self.assertIn("64-task", diagnostic.message)
        self.assertEqual(diagnostic.actual["tasks_per_submit"], 72)
        # The same shape on the per-tile path still compiles.
        plan = compile_rkir(matmul_module(m=300, k=576, n=3072),
                            profile=PROFILE,
                            options=options("baseline")).plan
        self.assertEqual(plan.accounting.task_count, 72)


class DeclaredScheduleAtM128Tests(unittest.TestCase):
    def test_declared_tile_is_applied_not_dropped(self):
        for tile_m, tile_n in ((256, 256), (128, 256)):
            with self.subTest(tile=(tile_m, tile_n)):
                for policy in ("baseline", "merged"):
                    plan = compile_rkir(declared(tile_m=tile_m,
                                                 tile_n=tile_n),
                                        profile=PROFILE,
                                        options=options(policy)).plan
                    applied = {(a.attrs["applied_schedule"]["tile_m"],
                                a.attrs["applied_schedule"]["tile_n"])
                               for a in tasks(plan) if "applied_schedule" in a.attrs}
                    self.assertEqual(applied, {(tile_m, tile_n)})
                    self.assertEqual({a.attrs["source_k_rounding"]
                                      for a in tasks(plan)},
                                     {"explicit_per_k_fp16"})
                    self.assertFalse(explain_plan(plan)["schedule"]["reblocked"])
                    self.assertEqual(len(plan.submits),
                                     1 if policy == "merged" else 12)

    def test_numeric_conflict_is_rejected_at_m128(self):
        for policy in ("baseline", "merged"):
            with self.subTest(policy=policy):
                with self.assertRaises(BackendError) as caught:
                    compile_rkir(declared(rounding="fp32_final_narrowing"),
                                 profile=PROFILE, options=options(policy))
                self.assertIn("K-rounding contract",
                              caught.exception.diagnostics[0].message)


class IdentityTests(unittest.TestCase):
    def test_save_load_keeps_the_plan_and_its_records(self):
        import tempfile

        plan = compile_rkir(matmul_module(m=128, k=576, n=3072),
                            profile=PROFILE, options=options("merged")).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m128-merged.rkplan.json"
            save_plan(plan, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.actions, plan.actions)
        self.assertEqual(loaded.submits, plan.submits)
        self.assertEqual(loaded.logical_ops, plan.logical_ops)
        self.assertEqual(loaded.numeric_requirements,
                         plan.numeric_requirements)


@unittest.skipUnless(tilelang_available(), "TileLang absent")
class TileLangSourceTests(unittest.TestCase):
    def test_production_kernel_at_m128_merges(self):
        from rknpu_backend import compile_tilelang
        from models.tilelang_matmul import make_matmul_prim
        plan = compile_tilelang(make_matmul_prim(128, 576, 3072),
                                profile=PROFILE,
                                options=options("merged")).plan
        self.assertEqual(plan.accounting.task_count, 36)
        self.assertEqual(len(plan.submits), 1)

    def test_scheduled_source_at_m128_keeps_its_declaration(self):
        import sys

        from rknpu_backend import compile_tilelang
        fixtures = Path(__file__).resolve().parent / "fixtures"
        if str(fixtures) not in sys.path:
            sys.path.insert(0, str(fixtures))
        import tilelang_linear_scheduled as fixture
        for bm, bn in ((256, 256), (128, 256)):
            with self.subTest(tile=(bm, bn)):
                plan = compile_tilelang(
                    fixture.make_linear(128, 576, 3072, bm=bm, bn=bn),
                    profile=PROFILE, options=options("merged")).plan
                applied = {(a.attrs["applied_schedule"]["tile_m"],
                            a.attrs["applied_schedule"]["tile_n"])
                           for a in tasks(plan) if "applied_schedule" in a.attrs}
                self.assertEqual(applied, {(bm, bn)})
                self.assertEqual(len(plan.submits), 1)
                self.assertEqual(plan.accounting.task_count, 36)


@unittest.skipUnless(Path("../custom-ops/build-host-mvp/"
                    "librknpu_plan_mock.so").resolve().is_file(),
                    "mock library not built")
class MockExecutionTests(unittest.TestCase):
    def test_merged_matches_baseline_bitwise(self):
        import os

        import numpy as np

        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan

        lib = LibRKPlan(str(Path("../custom-ops/build-host-mvp/"
                              "librknpu_plan_mock.so").resolve()))
        baseline = compile_rkir(matmul_module(m=128, k=576, n=3072),
                                profile=PROFILE,
                                options=options("baseline")).plan
        merged = compile_rkir(matmul_module(m=128, k=576, n=3072),
                              profile=PROFILE,
                              options=options("merged")).plan
        rng = np.random.default_rng(4242)
        a = np.ascontiguousarray(
            rng.standard_normal((128, 576)).astype(np.float16))
        w = np.ascontiguousarray(
            rng.standard_normal((576, 3072)).astype(np.float16))

        def run(plan):
            out = [np.zeros((128, 3072), dtype=np.float16)]
            call = PlanCall(lib, plan, {"B": w}, fd=0)
            try:
                report, _ = call.run_into([a], out)
            finally:
                call.close()
            return report, out[0]

        base_report, base_out = run(baseline)
        merged_report, merged_out = run(merged)
        self.assertEqual(int(base_report.task_count), 36)
        self.assertEqual(int(base_report.submit_count), 12)
        self.assertEqual(int(merged_report.task_count), 36)
        self.assertEqual(int(merged_report.submit_count), 1)
        self.assertTrue(np.array_equal(base_out, merged_out))
        acc = np.zeros((128, 3072), dtype=np.float16)
        for k0 in (0, 256, 512):
            k1 = min(576, k0 + 256)
            acc = (acc.astype(np.float64) +
                   a[:, k0:k1].astype(np.float64) @
                   w[k0:k1].astype(np.float64)).astype(np.float16)
        self.assertTrue(np.array_equal(merged_out, acc))


if __name__ == "__main__":
    unittest.main()
