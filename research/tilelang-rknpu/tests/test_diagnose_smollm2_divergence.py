"""Tests for the finite instruction-prompt divergence diagnostic."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import diagnose_smollm2_divergence as diag


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


class DivergenceDiagnosticTests(unittest.TestCase):
    def test_first_divergence_and_probe_prefix_are_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = [{"id": f"acc-{i}", "text": str(i),
                        "token_ids": [1, i]}
                       for i in range(1, 9)]
            board = {"requests": []}
            for i, prompt in enumerate(prompts, 1):
                bg = ([10 + i, 20 + i, 30 + i] if i != 5 else
                      [10, 20, 30, 40, 50, 60, 70, 80, 90])
                og = ([10 + i, 20 + i, 31 + i] if i != 5 else
                      [10, 20, 30, 40, 50, 60, 70, 80, 91])
                board["requests"].append({
                    "mode": "generate", "token_ids": prompt["token_ids"],
                    "runs": [{"generated_ids": bg, "output_tokens": len(bg)}]})
                write_json(root / "official" / f"acc-{i}" / "result.json", {
                    "prompt_ids": prompt["token_ids"],
                    "generated_ids": og,
                    "ignore_eos": True,
                    "max_new_tokens": len(og),
                })
            write_json(root / "board.json", board)
            inputs = {"acceptance_prompts": prompts}
            write_json(root / "inputs.json", inputs)
            manifest = diag.dataset_manifest(
                root / "board.json", root / "official",
                root / "inputs.json")
            row = manifest["rows"][0]
            self.assertEqual(row["first_divergence"], 2)
            self.assertEqual(row["probe_prefix_token_ids"], [1, 1, 11, 21])
            row5 = manifest["rows"][4]
            self.assertEqual(row5["first_divergence"], 8)
            self.assertEqual(row5["probe_prefix_len"], 10)

    def test_top_candidates_and_classification_keep_measured_margin(self):
        class Tok:
            def decode(self, ids):
                return f"t{ids[0]}"

        row = np.asarray([1.0, 3.0, 2.0, -1.0], dtype=np.float16)
        top1, margin, cands = diag.top_candidates(row, Tok(), k=3)
        self.assertEqual(top1, 1)
        self.assertAlmostEqual(margin, 1.0)
        self.assertEqual([x["token_id"] for x in cands], [1, 2, 0])
        self.assertEqual(diag.classify(1, 1, 1),
                         "three_way_top1_agreement")
        self.assertEqual(diag.classify(1, 1, 2),
                         "declared_reference_matches_backend_not_official")
        self.assertEqual(diag.classify(1, 2, 2),
                         "backend_differs_declared_and_official_agree")
        self.assertEqual(diag.classify(1, 2, 3),
                         "three_way_top1_difference")


if __name__ == "__main__":
    unittest.main()
