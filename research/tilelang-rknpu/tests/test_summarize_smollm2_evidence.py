"""Summarizer must reject failed, numeric-only or incomplete verifier output."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import summarize_smollm2_evidence as summary


class SummarizerFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "verify").mkdir(parents=True)
        (self.root / "package").mkdir()
        (self.root / "package/manifest.json").write_text("{}\n")
        (self.root / "spec.json").write_text("{}\n")
        self.verify = self.root / "verify/verify-summary.json"

    def tearDown(self):
        self.tmp.cleanup()

    def args(self):
        return [
            "--results", str(self.root),
            "--raw", str(self.root / "raw"),
            "--package", str(self.root / "package"),
            "--spec", str(self.root / "spec.json"),
            "--calibration-bundle", str(self.root / "cal.json"),
            "--acceptance-bundle", str(self.root / "acc.json"),
        ]

    def write_verify(self, **updates):
        value = {
            "mode": "complete",
            "complete_acceptance": True,
            "quality_pass": True,
            "pass": True,
            "gate_failures": [],
            "failures": [],
            "calibration_bundle": {"pass": True},
            "acceptance_bundle": {"pass": True},
        }
        value.update(updates)
        self.verify.write_text(json.dumps(value) + "\n")

    def test_rejects_failed_verifier(self):
        self.write_verify(**{"pass": False})
        with self.assertRaises(SystemExit):
            summary.main(self.args())

    def test_rejects_numeric_only_result(self):
        self.write_verify(mode="numeric-only", complete_acceptance=False)
        with self.assertRaises(SystemExit):
            summary.main(self.args())

    def test_rejects_quality_failure(self):
        self.write_verify(quality_pass=False)
        with self.assertRaises(SystemExit):
            summary.main(self.args())

    def test_rejects_gate_failure_hidden_in_other_fields(self):
        self.write_verify(gate_failures=["logits.full"])
        with self.assertRaises(SystemExit):
            summary.main(self.args())

    def test_rejects_missing_verifier(self):
        with self.assertRaises(SystemExit):
            summary.main(self.args())


if __name__ == "__main__":
    unittest.main()
