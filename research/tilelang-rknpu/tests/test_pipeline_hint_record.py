# SPDX-License-Identifier: GPL-3.0-or-later
"""The source pipeline request is recorded, never claimed as applied."""
from __future__ import annotations

import unittest

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.frontend.tilelang_adapter import (
    PIPELINE_NOT_APPLIED_REASON, _pipeline_record)
from models.tilelang_matmul import make_matmul_prim
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def k_loop_record(plan):
    for action in plan.actions:
        if action.kind == "npu_task" and "source_k_loop" in action.attrs:
            return action.attrs["source_k_loop"]
    return {}


class PipelineRecordUnitTests(unittest.TestCase):
    def test_absent_request_records_nothing_dropped(self):
        record = _pipeline_record({})
        self.assertEqual(record["source_pipeline_stages"], 0)
        self.assertFalse(record["pipeline_hint_dropped"])
        self.assertIsNone(record["pipeline_not_applied_reason"])

    def test_zero_stages_is_not_a_request(self):
        record = _pipeline_record({"num_stages": 0})
        self.assertEqual(record["source_pipeline_stages"], 0)
        self.assertFalse(record["pipeline_hint_dropped"])
        self.assertIsNone(record["pipeline_not_applied_reason"])

    def test_a_positive_request_is_recorded_and_marked_unapplied(self):
        record = _pipeline_record({"num_stages": 2})
        self.assertEqual(record["source_pipeline_stages"], 2)
        self.assertTrue(record["pipeline_hint_dropped"])
        self.assertEqual(record["pipeline_not_applied_reason"],
                         PIPELINE_NOT_APPLIED_REASON)
        self.assertEqual(record["source_loop_annotations"], {"num_stages": 2})

    def test_manual_pipeline_lists_count_as_a_request(self):
        record = _pipeline_record({"software_pipeline_order": [0, 1]})
        self.assertEqual(record["source_pipeline_stages"], 1)
        self.assertTrue(record["pipeline_hint_dropped"])

    def test_values_are_json_scalars(self):
        class FakeInt:
            def __int__(self):
                return 3

            def __str__(self):
                return "3"

        record = _pipeline_record({"num_stages": FakeInt(), "other": FakeInt()})
        self.assertEqual(record["source_pipeline_stages"], 3)
        self.assertEqual(record["source_loop_annotations"],
                         {"num_stages": 3, "other": 3})


class PipelineRecordPlanTests(unittest.TestCase):
    def test_a_pipelined_k_loop_plan_records_the_request(self):
        plan = compile_tilelang(make_linear(256, 576, 3072, bm=256, bn=256),
                                profile=PROFILE, options=options()).plan
        record = k_loop_record(plan)
        self.assertEqual(record.get("source_pipeline_stages"), 2)
        self.assertTrue(record.get("pipeline_hint_dropped"))
        self.assertIn("no software pipeline", record.get(
            "pipeline_not_applied_reason", ""))
        self.assertEqual(record.get("source_loop_annotations"), {"num_stages": 2})

    def test_a_source_without_a_loop_carries_no_request(self):
        plan = compile_tilelang(make_matmul_prim(256, 576, 3072),
                                profile=PROFILE, options=options()).plan
        self.assertFalse(k_loop_record(plan))
        self.assertEqual(plan.plan_hash[:12], "11f43cc45ee9")

    def test_the_record_survives_save_and_load(self):
        import tempfile
        from pathlib import Path

        from rknpu_backend import load_plan, save_plan

        plan = compile_tilelang(make_linear(256, 576, 3072, bm=256, bn=256),
                                profile=PROFILE, options=options()).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        record = k_loop_record(loaded)
        self.assertEqual(record.get("source_pipeline_stages"), 2)
        self.assertTrue(record.get("pipeline_hint_dropped"))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)


if __name__ == "__main__":
    unittest.main()
