"""Frozen SmolLM2 package identity and matrix-coverage checks (local package)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / ".data" / "models" / "smollm2_135m_instruct_package"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipUnless((PACKAGE / "manifest.json").is_file(),
                     "local SmolLM2 package not compiled")
class SmolLM2PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PACKAGE / "manifest.json").read_text())

    def test_fixed_architecture(self):
        c = self.manifest["config"]
        self.assertEqual(self.manifest["revision"],
                         "12fd25f77366fa6b3b4b768ec3050bf629380bac")
        self.assertEqual(c["num_hidden_layers"], 30)
        self.assertEqual(c["hidden_size"], 576)
        self.assertEqual(c["intermediate_size"], 1536)
        self.assertEqual(c["vocab_size"], 49152)
        self.assertEqual(c["num_attention_heads"], 9)
        self.assertEqual(c["num_key_value_heads"], 3)
        self.assertEqual(c["head_dim"], 64)
        self.assertEqual(c["hidden_act"], "silu")
        self.assertTrue(c["tie_word_embeddings"])
        self.assertEqual(self.manifest["capacity"], 256)

    def test_plan_count_and_matrix_only_tasks(self):
        plans = self.manifest["plans"]
        self.assertEqual(len(plans), 268)
        for name, info in plans.items():
            self.assertTrue(info["templates"], name)
            self.assertTrue(all(t.startswith("co.mm") for t in
                                info["templates"]), name)
            self.assertIn("models/tilelang_matmul.py", info["source_hash"],
                          name)
        self.assertEqual(self.manifest["resources"]["task_count"], 5660)
        self.assertEqual(self.manifest["resources"]["submit_count"], 1712)

    def test_lm_head_shards_cover_the_full_vocabulary(self):
        width = self.manifest["lm_head_shard_width"]
        self.assertEqual(width, 4096)
        self.assertEqual(width * self.manifest["lm_head_shards"], 49152)
        for profile in ("decode", "prefill"):
            for j in range(12):
                info = self.manifest["plans"][f"lm_head.{profile}.{j:02d}"]
                self.assertEqual(info["constants"][0]["shape"], [576, 4096])
                self.assertEqual(info["constants"][0]["nbytes"], 576 * 4096 * 2)

    def test_packed_a_slices_are_shared_across_output_tiles(self):
        def uploads(name):
            info = self.manifest["plans"][name]
            plan = json.loads((PACKAGE / info["path"]).read_text())
            by_binding = {}
            for action in plan["actions"]:
                if action["kind"] != "upload":
                    continue
                name = plan["bindings"][action["attrs"]["source_binding"]]["name"]
                by_binding.setdefault(name, []).append(action)
            return by_binding

        # QKV/gate_up/lm_head have several N tiles but only three distinct
        # K-slice A regions; down has six K slices.
        for name in ("layer.0.qkv.decode", "layer.0.gate_up.decode",
                     "lm_head.decode.00"):
            self.assertEqual(len(uploads(name)["A"]), 3, name)
        self.assertEqual(len(uploads("layer.0.down.decode")["A"]), 6)
        # The M-tiled prefill QK plan reuses the same dynamic B tile for every
        # row tile, so it appears once rather than once per M tile.
        self.assertEqual(len(uploads("attn.qk.prefill")["B"]), 1)

    def test_attention_plans_use_runtime_device_operands(self):
        for name in ("attn.qk.decode", "attn.pv.decode",
                     "attn.qk.prefill", "attn.pv.prefill"):
            info = self.manifest["plans"][name]
            self.assertTrue(info["b_is_input"])
            self.assertEqual(info["constants"], [])


if __name__ == "__main__":
    unittest.main()
