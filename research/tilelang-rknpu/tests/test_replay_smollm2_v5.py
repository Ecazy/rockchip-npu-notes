"""Tests for the executable SmolLM2 replay entry."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import replay_smollm2_v5 as replay


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


class ReplayIdentityTests(unittest.TestCase):
    def test_independent_record_controls_expected_spec_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = root / "spec.json"
            body = {"kind": "test", "value": 1}
            spec_body = dict(body)
            spec_body["canonical_content_sha256"] = replay.canonical_json_sha(body)
            write_json(spec, spec_body)
            record = root / "hashes.json"
            write_json(record, {"spec.json": {
                "file_sha256": replay.sha(spec),
                "canonical_content_sha256":
                    spec_body["canonical_content_sha256"],
            }})
            got = replay.check_spec_identity(spec, record)
            self.assertEqual(got["file_sha256"], replay.sha(spec))
            write_json(spec, {**spec_body, "value": 2})
            with self.assertRaises(SystemExit):
                replay.check_spec_identity(spec, record)

    def test_build_request_set_preserves_required_order_and_identities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cal_prompt = {"id": "cal-1", "token_ids": [1, 2, 3]}
            acc_prompt = {"id": "acc-2", "token_ids": [1, 4, 5]}
            windows_c = {"kind": "w", "windows": [[1, 2], [3, 4]],
                         "window_meta": []}
            windows_a = {"kind": "w", "windows": [[5, 6]],
                         "window_meta": []}
            inp = {
                "calibration_prompts": [cal_prompt],
                "acceptance_prompts": [acc_prompt],
                "numeric_acceptance_prompt_id": "acc-2",
                "calibration_nll_windows": windows_c,
                "acceptance_nll_windows": windows_a,
                "boundary_cases": {
                    "short_1": [1], "short_2": [1, 4093],
                    "non_aligned_prompt": "hello",
                    "capacity_prompt_tokens": 8,
                    "capacity_token_id": 9690,
                },
                "frozen_sha256": "old",
            }
            inputs = root / "inputs.json"
            write_json(inputs, inp)
            cal = root / "cal"
            write_json(cal / "cal-1/result.json", {
                "prompt_ids": cal_prompt["token_ids"],
                "generated_ids": list(range(20, 40)),
            })
            num = root / "num.json"
            write_json(num, {"prompt_ids": acc_prompt["token_ids"],
                             "generated_ids": list(range(40, 80))})
            fake_tok = mock.Mock()
            fake_tok.apply_chat_template.return_value = [1, 7, 8]
            with mock.patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=fake_tok):
                extended = replay.build_request_set(
                    inputs, root / "requests", root / "model", cal, num)
            manifest = replay.read_json(extended)
            self.assertNotEqual(manifest["frozen_sha256"], "old")
            self.assertEqual(manifest["calibration_request"]["sha256"],
                             replay.sha(root / "requests/calibration.json"))
            cal_req = replay.read_json(root / "requests/calibration.json")
            self.assertEqual([r["mode"] for r in cal_req["runs"]],
                             ["capture", "teacher_forcing", "incremental"])
            acc_req = replay.read_json(root / "requests/acceptance.json")
            self.assertEqual(acc_req["runs"][0]["mode"], "capture")
            self.assertEqual(acc_req["runs"][1]["mode"], "teacher_forcing")
            self.assertEqual(acc_req["runs"][2]["mode"], "incremental")
            self.assertEqual(acc_req["runs"][1]["token_ids"][-1], 71)
            self.assertEqual(acc_req["runs"][2]["continuation_ids"],
                             list(range(40, 56)))
            self.assertEqual(acc_req["runs"][-1]["stop_on_eos"], True)
            self.assertEqual(len(replay.read_json(
                root / "requests/ppl-acceptance.json")["runs"]), 1)

    def test_verifier_command_uses_external_expected_hash_and_numeric_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = {"numeric_acceptance_prompt_id": "acc-2",
                   "calibration_prompts": [{"id": "cal-1"}]}
            write_json(root / "inputs.json", inp)
            cmd = replay.verifier_command(
                spec=root / "spec.json",
                expected_spec_file_sha="expected-hash",
                acceptance_root=root / "acc",
                calibration_root=root / "cal",
                calibration_bundle=root / "cal-bundle.json",
                acceptance_bundle=root / "acc-bundle.json",
                requests=root / "requests",
                inputs_manifest=root / "inputs.json",
                ppl_windows=root / "windows.json",
                acceptance_prompt="prompt", out=root / "out.json")
            text = " ".join(cmd)
            self.assertIn("--expected-spec-file-sha256 expected-hash", text)
            self.assertIn("acc-2/logits.npy", text)
            self.assertIn("cal-1/logits.npy", text)


if __name__ == "__main__":
    unittest.main()
