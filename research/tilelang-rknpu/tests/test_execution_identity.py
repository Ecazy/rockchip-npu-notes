"""Negative tests for effective execution identity and policy validation."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verify_smollm2 import check_execution_identity
import tune_schedule


def expected_identity():
    return {
        "kind": "smollm2-effective-execution-identity",
        "base_package": {"manifest_sha256": "manifest",
                         "weights_sha256": "weights"},
        "profile_hash": "profile", "numeric_mode": "rk_f16_tiled",
        "plan_library_sha256": "lib",
        "schedule_policy_sha256": "policy",
        "overrides": {
            "attn.qk.prefill": {
                "replacement_kind": "single_matmul_schedule_v1",
                "base_plan_hash": "base-plan", "plan_hash": "override-plan",
                "plan_sha256": "override-file", "task_count": 6,
                "cache_key": "cache",
            },
        },
    }


def board_record():
    return {
        "schedule_policy": {
            "sha256": "policy", "override_keys": ["attn.qk.prefill"],
            "overrides": {
                "attn.qk.prefill": {
                    "replacement_kind": "single_matmul_schedule_v1",
                    "base_plan_hash": "base-plan", "plan_hash": "override-plan",
                    "plan_sha256": "override-file", "task_count": 6,
                    "cache_key": "cache",
                },
            },
            "loaded_plan_hashes": {"attn.qk.prefill": "override-plan"},
        },
    }


ENV = {
    "package_manifest_sha256": "manifest",
    "package_weights_sha256": "weights",
    "profile_hash": "profile",
    "numeric_mode": "rk_f16_tiled",
    "plan_lib_sha256": "lib",
}


class EffectiveIdentityTests(unittest.TestCase):
    def test_matching_identity_passes(self):
        self.assertEqual(check_execution_identity(
            expected_identity(), ENV, board_record()), [])

    def test_policy_hash_change_fails(self):
        board = board_record()
        board["schedule_policy"]["sha256"] = "other"
        self.assertIn("execution_identity_policy_hash",
                      check_execution_identity(expected_identity(), ENV, board))

    def test_override_key_change_fails(self):
        board = board_record()
        board["schedule_policy"]["override_keys"] = ["other.plan"]
        board["schedule_policy"]["overrides"] = {}
        self.assertTrue(check_execution_identity(expected_identity(), ENV, board))

    def test_plan_hash_and_loaded_hash_change_fail(self):
        board = board_record()
        board["schedule_policy"]["overrides"]["attn.qk.prefill"]["plan_hash"] = "other"
        board["schedule_policy"]["loaded_plan_hashes"]["attn.qk.prefill"] = "other"
        failures = check_execution_identity(expected_identity(), ENV, board)
        self.assertIn("execution_identity_override_attn.qk.prefill_plan_hash", failures)
        self.assertIn("execution_identity_loaded_plan_hashes", failures)

    def test_package_library_or_numeric_change_fails(self):
        for key, value in (("package_manifest_sha256", "x"),
                           ("package_weights_sha256", "x"),
                           ("plan_lib_sha256", "x"),
                           ("numeric_mode", "strict")):
            env = dict(ENV)
            env[key] = value
            self.assertTrue(check_execution_identity(
                expected_identity(), env, board_record()), key)

    def test_unexpected_override_when_none_expected_fails(self):
        expected = expected_identity()
        expected["overrides"] = {}
        expected["schedule_policy_sha256"] = "policy"
        self.assertIn("execution_identity_override_keys",
                      check_execution_identity(expected, ENV, board_record()))

    def test_empty_and_duplicate_policy_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            plans = Path(td)
            for payload in ({"kind": "smollm2-schedule-policy", "cases": []},
                            {"kind": "smollm2-schedule-policy",
                             "cases": [{"id": "x"}, {"id": "x"}]}):
                (plans / "policy.json").write_text(json.dumps(payload))
                with self.assertRaises(SystemExit):
                    tune_schedule.replay(SimpleNamespace(
                        plans=str(plans), package=str(plans), out=str(plans/"out"),
                        repeats=1, mock=True))

    def test_output_validation_separates_bitwise_numeric_and_validity(self):
        ref = np.asarray([[1.0, 2.0]], dtype=np.float16)
        exact = tune_schedule.check_output(ref.copy(), ref)
        self.assertTrue(exact["bitwise_equal"])
        self.assertTrue(exact["numeric_pass"])
        near = ref.copy()
        near[0, 0] = np.float16(1.001)
        check = tune_schedule.check_output(near, ref)
        self.assertFalse(check["bitwise_equal"])
        self.assertTrue(check["numeric_pass"])
        bad_shape = tune_schedule.check_output(np.zeros((2, 2), dtype=np.float16), ref)
        self.assertFalse(bad_shape["shape_ok"])
        self.assertFalse(bad_shape["numeric_pass"])


if __name__ == "__main__":
    unittest.main()
