"""Generation stop/count/timing semantics without a device."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend.models import RunStats
from rknpu_backend.smollm import SmolLM2Runner


class FakeRunner(SmolLM2Runner):
    def __init__(self, tokens, *, capacity=10):
        self.stats = RunStats()
        self.capacity = capacity
        self.valid_len = 0
        self.config = SimpleNamespace(vocab_size=10, eos_token_id=2)
        self.layer_indices = ()
        self._tokens = list(tokens)
        self._sample_index = 0
        self._decode_calls = 0

    def reset(self):
        self.valid_len = 0

    def prefill(self, ids, *, return_all_logits=False):
        self.valid_len = len(ids)
        time.sleep(0.001)
        return {"logits": np.zeros((1, 10), dtype=np.float16),
                "all_logits": None, "valid_len": self.valid_len,
                "elapsed_us": 2000.0}

    def _act(self, action_id, inputs, attrs, *, nbytes=None):
        if action_id != "smollm.greedy_sample.v1":
            raise AssertionError(action_id)
        token = self._tokens[self._sample_index]
        self._sample_index += 1
        self.stats.host_action_us += 1000.0
        return {"token_ids": np.asarray([token], dtype=np.int32)}

    def decode_one(self, token_id):
        self._decode_calls += 1
        self.valid_len += 1
        self.stats.upload_bytes += 128
        self.stats.download_bytes += 256
        self.stats.plan_pack_us += 10.0
        self.stats.plan_submit_wait_us += 20.0
        self.stats.task_count += 3
        self.stats.submit_count += 1
        time.sleep(0.001)
        return {"logits": np.zeros((1, 10), dtype=np.float16),
                "valid_len": self.valid_len, "elapsed_us": 3000.0}


class GenerationTimingTests(unittest.TestCase):
    def test_max_new_tokens_has_no_unused_final_decode_forward(self):
        runner = FakeRunner([5, 6, 7])
        result = runner.generate([1], max_new_tokens=3, stop_on_eos=False)
        self.assertEqual(result.output_tokens, 3)
        self.assertEqual(result.decode_forwards, 2)
        self.assertEqual(runner._decode_calls, 2)
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(result.stages["decode"]["forward_count"], 2)
        self.assertEqual(result.stages["sampling"]["calls"], 3)
        self.assertGreater(result.ttft_us, 0.0)
        self.assertAlmostEqual(
            result.ttft_us,
            result.reset_us + result.prefill_us +
            result.first_sample_us, places=6)
        self.assertGreater(result.generate_wall_us, result.decode_us)

    def test_eos_stops_without_decoding_the_eos_token(self):
        runner = FakeRunner([5, 2, 7])
        result = runner.generate([1], max_new_tokens=5, stop_on_eos=True)
        self.assertEqual(result.generated_ids, [5, 2])
        self.assertTrue(result.stopped_by_eos)
        self.assertEqual(result.stop_reason, "eos")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.decode_forwards, 1)
        self.assertEqual(runner._decode_calls, 1)

    def test_capacity_stop_is_explicit_and_has_no_final_decode(self):
        runner = FakeRunner([5], capacity=4)
        result = runner.generate([1, 2, 3], max_new_tokens=1,
                                 stop_on_eos=False)
        self.assertEqual(result.output_tokens, 1)
        self.assertEqual(result.decode_forwards, 0)
        self.assertEqual(result.stop_reason, "capacity")
        self.assertEqual(runner.valid_len, 3)


if __name__ == "__main__":
    unittest.main()
