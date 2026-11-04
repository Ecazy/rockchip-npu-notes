# SPDX-License-Identifier: GPL-3.0-or-later
"""Real full-logical GEMM -> target TIR -> existing task consumer."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import tilelang
from tvm import tirx

import models.tilelang_matmul as tmm
from tests.fixtures import tilelang_negative as negative
from tests.fixtures import tilelang_semantics as sem
from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend import logical_schedule as S
from rknpu_backend.frontend.rknpu_pass import _op_name
from rknpu_backend.plan import Plan


M, K, N = 1, 576, 960
PROFILE = default_profile()
LEGACY_LOGICAL_PLAN_HASH = (
    "3792f3b2ef7d583eb063a8ca66ff398c8462cdc6d5554f79084c748ecb42f221"
)


def _frozen_plan(case: str) -> Plan:
    path = Path(__file__).resolve().parent / "fixtures" / "frozen_plans" / \
        f"{case}.json"
    document = json.loads(path.read_text())
    assert document["kind"] == "rknpu-frozen-plan-reference"
    assert document["profile_hash"] == PROFILE.profile_hash
    plan = Plan.from_dict(document["plan"])
    if case == "qkv_decode":
        assert plan.plan_hash == LEGACY_LOGICAL_PLAN_HASH
    return plan


def options(**extra):
    base = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    base.update(extra)
    return CompileOptions(**base)


def _execution_description(plan):
    """Fields that describe execution, excluding provenance-only metadata."""
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
        "submits": [(len(submit.task_ids), submit.recipe_id, submit.mode)
                    for submit in plan.submits],
        "publish": [(item.dst_row, item.dst_col, item.rows, item.cols)
                    for item in plan.publish],
        "accounting": plan.accounting.to_dict(),
        "numeric": [requirement.to_dict()["contract"]
                    for requirement in plan.numeric_requirements],
    }


class TargetTirTests(unittest.TestCase):
    def test_real_qkv_source_uses_target_tir_without_adapter_or_graph_planner(self):
        prim = tmm.make_matmul_prim(M, K, N)
        expected = _frozen_plan("qkv_decode")
        self.assertFalse(hasattr(S, "_prove_full_logical"))
        self.assertFalse(hasattr(S, "_target_tir"))
        with mock.patch(
                "rknpu_backend.codegen.pipeline.compile_graph",
                side_effect=AssertionError("graph planner was called")):
            plan, stages, facts, tasks = S.compile_logical_gemm_tasks(
                prim, profile=PROFILE, options=options())

        self.assertEqual([stage.name for stage in stages],
                         ["schedule-source-ops", "lower-scheduled-ops", "instantiate", "plan"])
        counts = Counter(task["kind"] for task in tasks)
        self.assertEqual(counts, {"fill": 28, "copy": 48,
                                  "gemm": 12, "store": 4})
        self.assertEqual(len(tasks), 92)
        self.assertEqual(plan.accounting.task_count, 12)
        self.assertEqual(plan.accounting.submit_count, 4)
        self.assertEqual(plan.accounting.data_bo_count, 17)
        self.assertEqual(facts.order, "m_major")
        self.assertEqual(facts.b_role, "constant")

        # Task-level provenance proves the default logical policy, not the
        # guarded-tail `scheduled_baseline` pin.
        npu = [action for action in plan.actions if action.kind == "npu_task"]
        self.assertTrue(all(action.attrs.get("linear_task_policy") == "baseline"
                            for action in npu))
        self.assertEqual(_execution_description(plan),
                         _execution_description(expected))


class OfficialEntryTests(unittest.TestCase):
    def test_compile_tilelang_and_artifact_use_the_task_chain(self):
        prim = tmm.make_matmul_prim(M, K, N)
        self.assertFalse(hasattr(S, "_prove_full_logical"))
        self.assertFalse(hasattr(S, "_target_tir"))
        with mock.patch(
                "rknpu_backend.codegen.pipeline.compile_graph",
                side_effect=AssertionError("graph planner was called")):
            result = compile_tilelang(prim, profile=PROFILE, options=options())
            artifact = compile_artifact(prim, profile=PROFILE,
                                        options=options())

        self.assertEqual(result.report["source"]["lowering"],
                         "source-op rewrite -> target TIR -> task consumer")
        self.assertEqual(artifact.source["lowering"],
                         "source-op rewrite -> target TIR -> task consumer")
        self.assertEqual(result.plan.to_dict(), artifact.plan.to_dict())
        self.assertEqual(result.plan.accounting.task_count, 12)
        self.assertEqual(result.plan.accounting.submit_count, 4)
        self.assertEqual(result.plan.accounting.data_bo_count, 17)

    def test_the_dispatch_range_is_structural_not_the_qkv_shape(self):
        prim = tmm.make_matmul_prim(2, 64, 128)
        with mock.patch(
                "rknpu_backend.codegen.pipeline.compile_graph",
                side_effect=AssertionError("graph planner was called")):
            result = compile_tilelang(prim, profile=PROFILE, options=options())
        self.assertEqual(result.report["source"]["lowering"],
                         "source-op rewrite -> target TIR -> task consumer")
        self.assertGreater(result.plan.accounting.task_count, 0)


class SourceEffectBoundaryTests(unittest.TestCase):
    def test_a_non_true_block_predicate_is_rejected(self):
        prim = negative.false_predicate()
        with self.assertRaises(BackendError) as caught:
            S.compile_logical_gemm_tasks(
                prim, profile=PROFILE, options=options())
        self.assertIn("block predicate is not provably true",
                      caught.exception.diagnostics[0].message)

    def test_an_unconsumed_side_effect_call_is_rejected(self):
        prim = tmm.make_matmul_prim(M, K, N)

        def change(node):
            if type(node).__name__ == "Evaluate" and \
                    _op_name(node.value) == "tl.tileop.copy":
                return tirx.SeqStmt([
                    node,
                    tirx.Evaluate(tirx.call_extern("void", "side_effect")),
                ])
            return None

        bad = prim.with_body(tirx.stmt_functor.ir_transform(
            prim.body, None, change))
        with self.assertRaises(BackendError) as caught:
            S.compile_logical_gemm_tasks(
                bad, profile=PROFILE, options=options())
        self.assertIn("call", caught.exception.diagnostics[0].message.lower())


def _silu_prim():
    return sem._silu_kernel(16, 256, 16)


class SiluTaskChainTests(unittest.TestCase):
    def test_real_silu_uses_target_tir_and_task_chain(self):
        prim = _silu_prim()
        expected = _frozen_plan("matmul_silu")
        with mock.patch(
                "rknpu_backend.codegen.pipeline.compile_graph",
                side_effect=AssertionError("graph planner was called")):
            result = compile_tilelang(
                prim, profile=PROFILE,
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True))
            artifact = compile_artifact(
                prim, profile=PROFILE,
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True))
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        self.assertEqual([action.template_id for action in result.plan.actions
                          if action.kind == "npu_task"],
                         ["co.mm.f16.v1", "co.lut.sigmoid.f16.v1",
                          "co.ew_mul.f16.v1"])
        self.assertEqual(_execution_description(result.plan),
                         _execution_description(expected))

    def test_silu_mock_execution_matches_the_frozen_plan(self):
        if not NumericParityTests.LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan

        prim = _silu_prim()
        old = _frozen_plan("matmul_silu")
        new = S.compile_logical_gemm_tasks(
            prim, profile=PROFILE,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True))[0]
        rng = np.random.default_rng(20261103)
        a = rng.standard_normal((16, 256)).astype(np.float16)
        b = rng.standard_normal((256, 16)).astype(np.float16)
        old_out = np.zeros((16, 16), dtype=np.float16)
        new_out = np.zeros((16, 16), dtype=np.float16)
        lib = LibRKPlan(str(NumericParityTests.LIB))
        identity = np.eye(32, dtype=np.float16)
        constants = {"B": b, "__identity_32": identity}
        old_call = PlanCall(lib, old, constants, fd=0)
        new_call = PlanCall(lib, new, constants, fd=0)
        try:
            old_call.run_into([a], [old_out])
            new_call.run_into([a], [new_out])
        finally:
            old_call.close()
            new_call.close()
        self.assertTrue(np.array_equal(new_out, old_out))


class NumericParityTests(unittest.TestCase):
    LIB = (Path(__file__).resolve().parents[1].parent / "custom-ops"
           / "build-host-mvp" / "librknpu_plan_mock.so")

    def test_mock_execution_matches_the_adapter_default_plan(self):
        if not self.LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan

        prim = tmm.make_matmul_prim(M, K, N)
        old = _frozen_plan("qkv_decode")
        new = S.compile_logical_gemm_tasks(
            prim, profile=PROFILE, options=options())[0]
        rng = np.random.default_rng(20261103)
        a = rng.standard_normal((M, K)).astype(np.float16)
        b = rng.standard_normal((K, N)).astype(np.float16)
        old_out = np.zeros((M, N), dtype=np.float16)
        new_out = np.zeros((M, N), dtype=np.float16)
        lib = LibRKPlan(str(self.LIB))
        identity = np.eye(32, dtype=np.float16)
        constants = {"B": b, "__identity_32": identity}
        old_call = PlanCall(lib, old, constants, fd=0)
        new_call = PlanCall(lib, new, constants, fd=0)
        try:
            old_call.run_into([a], [old_out])
            new_call.run_into([a], [new_out])
        finally:
            old_call.close()
            new_call.close()
        self.assertTrue(np.array_equal(new_out, old_out))


if __name__ == "__main__":
    unittest.main()
