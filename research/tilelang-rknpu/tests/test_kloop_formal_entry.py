# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal entry for the migrated unguarded final-only and round-trip K loops."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from models.tilelang_matmul import make_matmul_tiled_prim
from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.errors import BackendError, RK_E_NUMERIC_CONTRACT, \
    RK_E_UNSUPPORTED
from rknpu_backend.frontend import tilelang_lowering
from rknpu_backend.plan import Plan

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tilelang_kloop.py"
LIB = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"


def load_fixture():
    spec = importlib.util.spec_from_file_location("tl_kloop_formal", str(FIXTURE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(policy: str = "baseline"):
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                          linear_task_order=policy)


def frozen(name: str):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


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
    def assert_no_adapter(self, prim, *, name, policy="baseline",
                          b_is_input=False):
        if True:
            result = compile_tilelang(
                prim, profile=PROFILE, options=options(policy),
                b_is_input=b_is_input)
            artifact = compile_artifact(
                prim, profile=PROFILE, options=options(policy),
                b_is_input=b_is_input)
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        self.assertEqual(result.report["source"]["lowering"],
                         "K-loop source rewrite -> target TIR -> task consumer")
        self.assertEqual(execution(result.plan), execution(frozen(name)))
        return result

    def test_final_only_multi_trip_uses_source_rewrite(self):
        result = self.assert_no_adapter(
            load_fixture().k_loop(m=16, n=16, k=512, bk=256),
            name="kloop_multi_trip_baseline")
        self.assertEqual(result.plan.accounting.task_count, 2)
        self.assertEqual(result.report["source"]["source_k_loop"]
                         ["source_k_trips"], 2)

    def test_final_only_multi_trip_scheduled_baseline(self):
        self.assert_no_adapter(
            load_fixture().k_loop(m=16, n=16, k=512, bk=256),
            name="kloop_multi_trip_scheduled_baseline",
            policy="scheduled_baseline")

    def test_final_only_single_trip_uses_the_same_entry(self):
        self.assert_no_adapter(
            load_fixture().k_loop(m=16, n=16, k=256, bk=256),
            name="single_trip_kloop_baseline")

    def test_roundtrip_multi_trip_uses_source_rewrite(self):
        result = self.assert_no_adapter(
            make_matmul_tiled_prim(128, 512, 128, bm=64, bn=64),
            name="tiled_roundtrip_multi_baseline")
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertEqual(first.attrs["source_k_loop"]["source_rounding"],
                         "explicit_per_k_fp16")
        self.assertEqual(first.attrs["source_schedule"]["tile_m"], 64)

    def test_roundtrip_single_trip_uses_source_rewrite(self):
        self.assert_no_adapter(
            make_matmul_tiled_prim(16, 256, 16, bm=16, bn=16),
            name="tiled_roundtrip_single_baseline")

    def test_b_is_input_is_preserved(self):
        prim = make_matmul_tiled_prim(16, 256, 16, bm=16, bn=16)
        if True:
            result = compile_tilelang(
                prim, profile=PROFILE, options=options(), b_is_input=True)
        reference = frozen("tiled_roundtrip_single_baseline")
        self.assertEqual([(b.name, b.role) for b in result.plan.bindings],
                         [("A", "input"), ("B", "input"), ("Y", "output")])
        self.assertEqual(result.plan.accounting.to_dict(),
                         reference.accounting.to_dict())
        self.assertEqual(execution(result.plan)["tasks"],
                         execution(reference)["tasks"])

    def test_unsupported_policy_and_grouping_do_not_fall_back(self):
        prim = load_fixture().k_loop(m=16, n=16, k=512, bk=256)
        if True:
            with self.assertRaises(BackendError) as policy:
                compile_tilelang(prim, profile=PROFILE,
                                 options=options("weight_stationary"))
            grouped = CompileOptions(
                numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                linear_group_n_tiles=2)
            with self.assertRaises(BackendError) as grouping:
                compile_tilelang(prim, profile=PROFILE, options=grouped)
        self.assertEqual(policy.exception.diagnostics[0].code,
                         RK_E_UNSUPPORTED)
        self.assertEqual(grouping.exception.diagnostics[0].code,
                         RK_E_UNSUPPORTED)

    def test_submit_mode_is_passed_through(self):
        prim = load_fixture().k_loop(m=16, n=16, k=512, bk=256)
        result = compile_tilelang(
            prim, profile=PROFILE,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False,
                                   submit_mode="separate"))
        self.assertEqual(result.plan.accounting.submit_count, 2)
        self.assertEqual([submit.mode for submit in result.plan.submits],
                         ["separate", "separate"])

    def test_direct_dispatch_uses_the_shared_k_loop_entry(self):
        prim = load_fixture().k_loop(m=16, n=16, k=512, bk=256)
        self.assertEqual(tilelang_lowering.entry_for(prim).__name__,
                         "compile_k_loop_tasks")
        plan, _, _, _ = tilelang_lowering.compile_tilelang_ir(
            prim, profile=PROFILE, options=options())
        self.assertEqual(plan.accounting.task_count, 2)

    def test_strict_mode_keeps_the_numeric_rejection(self):
        prim = load_fixture().k_loop(m=16, n=16, k=512, bk=256)
        if True:
            with self.assertRaises(BackendError) as caught:
                compile_tilelang(
                    prim, profile=PROFILE,
                    options=CompileOptions(numeric_mode="strict",
                                           allow_approx_lut=False))
        self.assertEqual(caught.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)


class MockTests(unittest.TestCase):
    def assert_mock_equal(self, prim, name):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        old = frozen(name)
        new = compile_tilelang(prim, profile=PROFILE,
                               options=options()).plan
        rng = np.random.default_rng(20261104)
        m = old.bindings[0].shape[0]
        k = old.bindings[0].shape[1]
        n = old.bindings[1].shape[1]
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

    def test_final_only_plan_matches_frozen_mock(self):
        self.assert_mock_equal(
            load_fixture().k_loop(m=16, n=16, k=512, bk=256),
            "kloop_multi_trip_baseline")

    def test_roundtrip_plan_matches_frozen_mock(self):
        self.assert_mock_equal(
            make_matmul_tiled_prim(128, 512, 128, bm=64, bn=64),
            "tiled_roundtrip_multi_baseline")


if __name__ == "__main__":
    unittest.main()
