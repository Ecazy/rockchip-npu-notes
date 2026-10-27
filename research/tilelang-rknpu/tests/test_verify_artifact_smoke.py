"""The board-smoke verifier must reject bad device output.

The verifier is evidence-producing code: if it silently accepted a mismatched or
missing output, the board run would prove nothing.  These cases build a synthetic
campaign/board pair with ``.npy`` payloads and assert the verifier's exit status and
classification, independent of any device.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = ROOT / "tools" / "verify_artifact_smoke.py"
    spec = importlib.util.spec_from_file_location("_verify_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()
STEPS = ["compile_artifact", "explain", "instantiate_run_a_b_a",
         "two_independent_instantiations", "export_load_run"]
TOLERANCE = {"atol": 0.008, "rtol": 0.008}


class SmokeVerifierTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.campaign = base / "campaign"
        self.board = base / "board"
        self.out = base / "verify.json"
        self._build()

    # -- fixture ---------------------------------------------------------
    def _build(self):
        case = self.campaign / "cases" / "demo"
        rng = np.random.default_rng(7)
        reference_a = rng.normal(0, 0.3, (4, 5)).astype(np.float16)
        reference_b = rng.normal(0, 0.3, (4, 5)).astype(np.float16)
        (case / "reference").mkdir(parents=True, exist_ok=True)
        (case / "inputs").mkdir(parents=True, exist_ok=True)
        np.save(case / "reference" / "a_Y.npy", reference_a)
        np.save(case / "reference" / "b_Y.npy", reference_b)
        np.save(case / "inputs" / "a_X.npy", np.zeros((4, 5), np.float16))
        np.save(case / "inputs" / "b_X.npy", np.ones((4, 5), np.float16))
        meta = {
            "name": "demo",
            "tolerance": TOLERANCE,
            "options": {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False,
                        "submit_mode": "chain"},
            "plan_hash": "a" * 64,
            "bindings": [{"name": "X", "role": "input", "shape": [4, 5]},
                         {"name": "Y", "role": "output", "shape": [4, 5]}],
            "inputs": ["X"],
            "constants": [],
            "outputs": ["Y"],
            "tasks": 1, "submits": 1, "data_bos": 2,
            "explain_counts": {"tasks": 1, "submits": 1, "data_bos": 2},
        }
        (case / "meta.json").write_text(json.dumps(meta))
        (self.campaign / "manifest.json").write_text(json.dumps({
            "kind": "artifact-smoke-campaign",
            "cases": [{"name": "demo", "dir": "cases/demo",
                       "plan_hash": meta["plan_hash"], "tolerance": TOLERANCE,
                       "tasks": 1, "inputs": ["X"], "constants": [],
                       "outputs": ["Y"]}],
        }))
        self.reference_a = reference_a
        self.reference_b = reference_b

        case_board = self.board / "demo"
        result = {
            "case": "demo", "status": "ok",
            "steps": list(STEPS),
            "plan_hash_board": "a" * 64, "plan_hash_match": True,
            "plan_hash_reloaded": "a" * 64, "plan_hash_reloaded_match": True,
            "explain_counts": {"tasks": 1, "submits": 1, "data_bos": 2},
            "explain_matches_plan": True,
            "report_single": {"task_count": 1, "submit_count": 1},
        }
        (case_board / "result.json").parent.mkdir(parents=True, exist_ok=True)
        (case_board / "result.json").write_text(json.dumps(result))
        for stage in ("a1", "a2", "independent_1", "independent_2", "replay"):
            path = case_board / "board" / stage
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "Y.npy", reference_a)
        (case_board / "board" / "b").mkdir(parents=True, exist_ok=True)
        np.save(case_board / "board" / "b" / "Y.npy", reference_b)
        (self.board / "summary.json").write_text(json.dumps({
            "kind": "artifact-smoke-board",
            "environment": {"board_model": "synthetic", "chip": "rk3588"},
            "cases": [{"case": "demo", "status": "ok"}],
        }))

    def run_verifier(self):
        status = VERIFIER.main(["--campaign", str(self.campaign),
                                "--board", str(self.board),
                                "--out", str(self.out)])
        return status, json.loads(self.out.read_text())

    def _stage(self, stage: str) -> Path:
        return self.board / "demo" / "board" / stage / "Y.npy"

    # -- cases -----------------------------------------------------------
    def test_consistent_evidence_passes(self):
        status, report = self.run_verifier()
        self.assertEqual(status, 0, report["failures"])
        self.assertTrue(report["passed"])

    def test_output_beyond_the_frozen_tolerance_fails(self):
        tampered = self.reference_a.copy()
        tampered[0, 0] = np.float16(float(tampered[0, 0]) + 0.05)  # > atol
        np.save(self._stage("a1"), tampered)
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("numeric A mismatch" in failure
                            for failure in report["failures"]))

    def test_missing_board_output_fails(self):
        self._stage("replay").unlink()
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("missing board output" in failure
                            for failure in report["failures"]))

    def test_input_update_not_applied_fails(self):
        # B returns the A result: exactly the failure a stale-input run would show.
        np.save(self._stage("b"), self.reference_a)
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("input update not applied" in failure
                            for failure in report["failures"]))

    def test_replay_difference_fails(self):
        np.save(self._stage("replay"),
                (self.reference_a + np.float16(0.01)).astype(np.float16))
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("replay differs" in failure
                            for failure in report["failures"]))

    def test_plan_hash_mismatch_fails(self):
        result_path = self.board / "demo" / "result.json"
        result = json.loads(result_path.read_text())
        result["plan_hash_board"] = "b" * 64
        result_path.write_text(json.dumps(result))
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("plan_hash_board_matches_host" in failure
                            for failure in report["failures"]))

    def test_missing_board_summary_fails(self):
        (self.board / "summary.json").unlink()
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        self.assertIn("board summary.json is missing", report["failures"])

    def test_non_finite_device_output_fails(self):
        tampered = self.reference_a.copy()
        tampered[1, 1] = np.float16("nan")
        np.save(self._stage("a1"), tampered)
        status, report = self.run_verifier()
        self.assertNotEqual(status, 0)
        numeric = report["cases"]["demo"]["outputs"]["Y"]["numeric_a"]
        self.assertFalse(numeric["ok"])


if __name__ == "__main__":
    unittest.main()
