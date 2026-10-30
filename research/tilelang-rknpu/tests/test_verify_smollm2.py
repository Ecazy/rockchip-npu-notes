"""Adversarial tests for the SmolLM2 acceptance verifier.

These tests build complete synthetic observation directories and invoke the CLI
entry point, so they exercise the JSON result, overall pass flag and exit code,
not only the low-level gate helper.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_smollm2 as verifier


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


class SyntheticAcceptance:
    def __init__(self, root: Path):
        self.root = root
        self.board = root / "board"
        for sub in ("run-00", "run-01", "run-02"):
            (self.board / sub).mkdir(parents=True, exist_ok=True)
        self.ref = root / "reference"
        self.reftf = root / "reference_tf"
        self.official = root / "official"
        self.ppl_board = root / "ppl_board"
        self.ppl_official = root / "ppl_official"
        self.spec_path = root / "spec.json"
        self.cal_manifest = root / "calibration.json"
        self.acc_manifest = root / "acceptance.json"
        self.accept_ids = [1, 2, 3, 4]
        self.calib_ids = [9, 10, 11]
        self.tf_ids = self.accept_ids + [5, 6]
        self.vocab = 16
        self.package_manifest = "a" * 64
        self.package_weights = "b" * 64
        self.model_weights = "c" * 64
        self.model_config = "d" * 64
        self.make()

    def make(self) -> None:
        for path in (self.board, self.ref, self.reftf, self.official,
                     self.ppl_board, self.ppl_official):
            path.mkdir(parents=True, exist_ok=True)
        # Captured tensor and full logits.
        self.x_ref = np.arange(6, dtype=np.float16).reshape(2, 3)
        self.x_val = self.x_ref.copy()
        (self.board / "run-00").mkdir(parents=True, exist_ok=True)
        (self.ref).mkdir(parents=True, exist_ok=True)
        np.savez(self.board / "run-00" / "captured-00.npz", x=self.x_val)
        np.savez(self.ref / "captured.npz", x=self.x_ref)
        self.logits_ref = np.zeros((4, self.vocab), dtype=np.float16)
        self.logits_ref[:, 3] = 1.0
        self.logits_val = self.logits_ref.copy()
        self.official_logits = self.logits_ref.copy()
        np.save(self.board / "run-00" / "all-logits-00.npy", self.logits_val)
        np.save(self.ref / "logits.npy", self.logits_ref)
        np.save(self.official / "logits.npy", self.official_logits)
        # Teacher forcing / incremental.
        self.tf_ref = np.zeros((6, self.vocab), dtype=np.float16)
        self.tf_ref[:, 2] = 1.0
        self.tf_val = self.tf_ref.copy()
        self.inc = self.tf_ref[3:6].copy()
        np.save(self.board / "run-01" / "logits-00.npy", self.tf_val)
        np.save(self.reftf / "logits.npy", self.tf_ref)
        np.save(self.board / "run-02" / "incremental-logits-00.npy",
                self.inc)
        write_json(self.ref / "meta.json", {
            "tokens": 4, "token_ids": self.accept_ids,
            "package_manifest_sha256": self.package_manifest,
            "weights_file_sha256": self.package_weights,
        })
        write_json(self.reftf / "meta.json", {
            "tokens": 6, "token_ids": self.tf_ids,
            "package_manifest_sha256": self.package_manifest,
            "weights_file_sha256": self.package_weights,
        })
        write_json(self.official / "result.json", {
            "token_ids": self.accept_ids,
            "weights_sha256": self.model_weights,
            "config_sha256": self.model_config,
        })
        write_json(self.board / "run.json", {
            "kind": "smollm2-board-run",
            "request_sha256": "e" * 64,
            "environment": {
                "package_manifest_sha256": self.package_manifest,
                "package_weights_sha256": self.package_weights,
            },
            "requests": [
                {"mode": "capture", "token_count": 4,
                 "token_ids": self.accept_ids,
                 "runs": [{"valid_len": 4}]},
                {"mode": "teacher_forcing", "token_count": 6,
                 "token_ids": self.tf_ids,
                 "runs": [{"valid_len": 6}]},
                {"mode": "incremental", "token_count": 4,
                 "token_ids": self.accept_ids,
                 "runs": [{"continuation_ids": self.tf_ids[4:],
                           "valid_len": 6}]},
            ],
        })
        write_json(self.board / "stats.json", {})
        for i, mode in enumerate(("capture", "teacher_forcing",
                                  "incremental")):
            write_json(self.board / f"run-{i:02d}" / "result.json", {
                "index": i, "mode": mode,
                "token_ids": self.accept_ids,
                "token_count": 4, "runs": [{}]})
        # PPL evidence, one small window with two scored tokens.
        self.window_ids = [1, 2, 3]
        labels = [2, 3]
        nll = np.asarray([1.0, 1.5], dtype=np.float64)
        (self.ppl_board / "run-00").mkdir(parents=True, exist_ok=True)
        np.save(self.ppl_board / "run-00" / "nll-00.npy", nll)
        write_json(self.ppl_board / "run-00" / "result.json", {
            "index": 0, "mode": "score", "repeat": 1,
            "token_ids": self.window_ids,
            "runs": [{
                "token_ids": self.window_ids,
                "scored_label_ids": labels,
                "scored_tokens": 2,
                "sum_nll": float(nll.sum()),
                "mean_nll": float(nll.mean()),
                "ppl": float(np.exp(nll.mean())),
                "top1_accuracy": 0.5,
                "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
            }],
        })
        write_json(self.ppl_board / "stats.json", {})
        write_json(self.ppl_board / "run.json", {})
        np.save(self.ppl_official / "nll.npy", nll)
        write_json(self.ppl_official / "result.json", {
            "kind": "smollm2-official-reference",
            "windows": [{
                "window": 0,
                "token_ids": self.window_ids,
                "scored_label_ids": labels,
                "scored_tokens": 2,
                "sum_nll": float(nll.sum()),
                "mean_nll": float(nll.mean()),
                "ppl": float(np.exp(nll.mean())),
                "top1_accuracy": 0.5,
            }],
            "scored_tokens": 2,
            "sum_nll": float(nll.sum()),
            "mean_nll": float(nll.mean()),
            "ppl": float(np.exp(nll.mean())),
            "top1_accuracy": 0.5,
            "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
        })
        self.windows_path = self.root / "windows.json"
        write_json(self.windows_path, {
            "kind": "smollm2-nll-windows",
            "windows": [self.window_ids],
            "window_meta": [{
                "window": 0, "token_ids": self.window_ids,
                "scored_label_ids": labels, "scored_tokens": 2,
            }],
        })
        self.write_spec()

    def write_spec(self, **overrides) -> None:
        gates = {
            "x": {"atol": 0.01, "rmse_limit": 0.01,
                  "finite_required": True},
            "logits.full": {"atol": 0.01, "rmse_limit": 0.01,
                            "finite_required": True,
                            "top1_rate_min": 1.0,
                            "top5_overlap_mean_min": 4.0},
            "logits.shard00": {"atol": 0.01, "rmse_limit": 0.01,
                               "finite_required": True},
            "teacher_forcing.logits": {"atol": 0.01, "rmse_limit": 0.01,
                                       "finite_required": True},
            "incremental_vs_full_prefix.logits": {
                "atol": 0.0, "rmse_limit": 0.0, "finite_required": True,
                "top1_rate_min": 1.0, "top5_overlap_mean_min": 4.0},
        }
        gates.update(overrides.pop("gates", {}))
        calibration = {
            "token_ids": self.calib_ids,
            "prompt_tokens": 3,
            "prompt_text": "calibration prompt",
            "package_manifest_sha256": self.package_manifest,
            "package_weights_sha256": self.package_weights,
            "board_run_sha256": sha(self.board / "run.json"),
            "board_stats_sha256": sha(self.board / "stats.json"),
            "board_captured_sha256": sha(
                self.board / "run-00" / "captured-00.npz"),
            "board_logits_sha256": sha(
                self.board / "run-00" / "all-logits-00.npy"),
            "board_teacher_forcing_sha256": sha(
                self.board / "run-01" / "logits-00.npy"),
            "board_incremental_sha256": sha(
                self.board / "run-02" / "incremental-logits-00.npy"),
            "reference_meta_sha256": sha(self.ref / "meta.json"),
            "reference_logits_sha256": sha(self.ref / "logits.npy"),
            "reference_tf_logits_sha256": sha(self.reftf / "logits.npy"),
            "official_logits_sha256": sha(self.official / "logits.npy"),
        }
        quality = {
            "limits": {
                "scored_tokens_min": 1,
                "mean_nll_delta_max": 0.1,
                "ppl_ratio_max": 1.1,
                "top1_accuracy_delta_min": -0.1,
            },
            "dataset": {"source_sha256": "f" * 64},
        }
        spec = {
            "schema_version": 1,
            "kind": "smollm2-numeric-acceptance",
            "policy_revision": 3,
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            "calibration": calibration,
            "official_reference": {
                "weights_sha256": self.model_weights,
                "config_sha256": self.model_config,
            },
            "expected_gates": list(gates),
            "gates": gates,
            "quality": quality,
        }
        spec.update(overrides)
        body = dict(spec)
        spec["spec_sha256"] = verifier.canonical_json_sha(body)
        write_json(self.spec_path, spec)

    def seal(self, role: str, out: Path, spec: Path | None = None) -> None:
        sys.path.insert(0, str(TOOLS))
        import seal_smollm2_observation as seal_mod
        argv = [
            "--role", role,
            "--board", str(self.board),
            "--reference", str(self.ref),
            "--reference-tf", str(self.reftf),
            "--official-logits", str(self.official / "logits.npy"),
            "--ppl-board", str(self.ppl_board),
            "--ppl-official", str(self.ppl_official / "result.json"),
            "--spec", str(spec or self.spec_path),
            "--out", str(out),
        ]
        if seal_mod.main(argv) != 0:
            raise RuntimeError("seal failed")

    def args(self, *, numeric_only: bool = True, complete: bool = False):
        args = [
            "--spec", str(self.spec_path),
            "--board", str(self.board),
            "--reference", str(self.ref),
            "--reference-tf", str(self.reftf),
            "--official-logits", str(self.official / "logits.npy"),
            "--out", str(self.root / "verify.json"),
        ]
        if numeric_only:
            args.append("--numeric-only")
        if complete:
            args += [
                "--ppl-board", str(self.ppl_board),
                "--ppl-official", str(self.ppl_official / "result.json"),
                "--ppl-windows", str(self.windows_path),
                "--calibration-manifest", str(self.cal_manifest),
                "--acceptance-manifest", str(self.acc_manifest),
                "--calibration-board", str(self.board),
                "--calibration-reference", str(self.ref),
                "--calibration-reference-tf", str(self.reftf),
                "--calibration-official-logits",
                str(self.official / "logits.npy"),
                "--calibration-ppl-board", str(self.ppl_board),
                "--calibration-ppl-official",
                str(self.ppl_official / "result.json"),
            ]
        return args

    def run(self, *, numeric_only: bool = True, complete: bool = False):
        rc = verifier.main(self.args(numeric_only=numeric_only,
                                     complete=complete))
        result = json.loads((self.root / "verify.json").read_text())
        return rc, result


class VerifierNegativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.case = SyntheticAcceptance(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_baseline_numeric_passes_and_is_explicitly_incomplete(self):
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 0)
        self.assertTrue(result["pass"])
        self.assertFalse(result["complete_acceptance"])
        self.assertEqual(result["mode"], "numeric-only")

    def test_gate_failure_cannot_be_hidden_by_failures_list(self):
        self.case.write_spec(gates={
            "logits.full": {"atol": 0.0, "rmse_limit": 0.0,
                            "finite_required": True}})
        self.case.logits_val[0, 0] = 0.5
        np.save(self.case.board / "run-00" / "all-logits-00.npy",
                self.case.logits_val)
        self.case.write_spec(gates={
            "logits.full": {"atol": 0.0, "rmse_limit": 0.0,
                            "finite_required": True}})
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertFalse(result["pass"])
        self.assertIn("logits.full", result["gate_failures"])
        self.assertIn("numeric_gate_failures", result["failures"])

    def test_rank_gate_failure_with_small_numeric_error(self):
        self.case.logits_val[0, 3] = 0.0
        self.case.logits_val[0, 2] = 0.01
        np.save(self.case.board / "run-00" / "all-logits-00.npy",
                self.case.logits_val)
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("logits.full", result["gate_failures"])
        self.assertIn("top1_rate", result["gates"]["logits.full"]["failures"])

    def test_missing_capture_gate_fails(self):
        self.case.write_spec(gates={"missing.tensor": {
            "atol": 0.1, "rmse_limit": 0.1, "finite_required": True}})
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("missing.tensor", result["gate_failures"])
        self.assertIn("capture_tensor_set", result["failures"])

    def test_broadcastable_shape_mismatch_is_rejected(self):
        # Replace the reference tensor with broadcastable (2,1) shape.
        np.savez(self.case.ref / "captured.npz",
                 x=np.zeros((2, 1), dtype=np.float16))
        np.savez(self.case.board / "run-00" / "captured-00.npz",
                 x=np.zeros((2, 3), dtype=np.float16))
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("shape_mismatch", result["gates"]["x"]["failures"])

    def test_nonfinite_observation_and_reference_fail(self):
        bad = self.case.x_val.copy()
        bad[0, 0] = np.nan
        np.savez(self.case.board / "run-00" / "captured-00.npz", x=bad)
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("nonfinite_observation_or_reference",
                      result["gates"]["x"]["failures"])
        np.savez(self.case.board / "run-00" / "captured-00.npz",
                 x=self.case.x_val)
        np.savez(self.case.ref / "captured.npz", x=bad)
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("nonfinite_observation_or_reference",
                      result["gates"]["x"]["failures"])

    def test_invalid_threshold_fails_closed(self):
        self.case.write_spec(gates={
            "x": {"atol": float("nan"), "rmse_limit": 0.1,
                  "finite_required": True}})
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("atol_invalid", result["gates"]["x"]["failures"])

    def test_package_identity_mismatch_fails(self):
        run = json.loads((self.case.board / "run.json").read_text())
        run["environment"]["package_manifest_sha256"] = "0" * 64
        write_json(self.case.board / "run.json", run)
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("package_manifest_identity", result["failures"])

    def test_input_token_mismatch_fails(self):
        run = json.loads((self.case.board / "run.json").read_text())
        run["requests"][0]["token_ids"] = [7, 8]
        write_json(self.case.board / "run.json", run)
        rc, result = self.case.run(numeric_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("capture_token_ids_vs_reference", result["failures"])

    def test_complete_requires_quality_evidence(self):
        with self.assertRaises(SystemExit) as ctx:
            verifier.main(self.case.args(numeric_only=False, complete=False))
        self.assertEqual(ctx.exception.code, 2)

    def test_complete_passes_with_matching_manifests(self):
        self.case.seal("calibration", self.case.cal_manifest)
        self.case.seal("acceptance", self.case.acc_manifest)
        rc, result = self.case.run(numeric_only=False, complete=True)
        self.assertEqual(rc, 0)
        self.assertTrue(result["pass"])
        self.assertTrue(result["complete_acceptance"])
        self.assertTrue(result["quality_pass"])


    def test_quality_window_token_mismatch_fails(self):
        result_path = self.case.ppl_board / "run-00" / "result.json"
        row = json.loads(result_path.read_text())
        row["token_ids"] = [7, 8, 9]
        row["runs"][0]["token_ids"] = [7, 8, 9]
        write_json(result_path, row)
        self.case.seal("calibration", self.case.cal_manifest)
        self.case.seal("acceptance", self.case.acc_manifest)
        rc, result = self.case.run(numeric_only=False, complete=True)
        self.assertEqual(rc, 1)
        self.assertFalse(result["quality_pass"])
        self.assertIn("quality_gate_failures", result["failures"])

    def test_quality_scored_token_count_mismatch_fails(self):
        result_path = self.case.ppl_official / "result.json"
        row = json.loads(result_path.read_text())
        row["windows"][0]["scored_tokens"] = 1
        write_json(result_path, row)
        self.case.seal("calibration", self.case.cal_manifest)
        self.case.seal("acceptance", self.case.acc_manifest)
        rc, result = self.case.run(numeric_only=False, complete=True)
        self.assertEqual(rc, 1)
        self.assertFalse(result["quality_pass"])
        self.assertIn("quality_gate_failures", result["failures"])

    def test_nonfinite_quality_nll_fails(self):
        bad = np.asarray([np.nan, 1.0], dtype=np.float64)
        np.save(self.case.ppl_board / "run-00" / "nll-00.npy", bad)
        self.case.seal("calibration", self.case.cal_manifest)
        self.case.seal("acceptance", self.case.acc_manifest)
        rc, result = self.case.run(numeric_only=False, complete=True)
        self.assertEqual(rc, 1)
        self.assertFalse(result["quality_pass"])
        self.assertIn("quality_gate_failures", result["failures"])


if __name__ == "__main__":
    unittest.main()
