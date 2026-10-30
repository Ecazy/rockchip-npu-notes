"""Targeted tests for immutable Host-weight caching and profile accounting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend.smollm import SmolLM2Runner


class _CountingWeights:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.arrays[key]


def _runner(*, cache: bool, weights):
    runner = SmolLM2Runner.__new__(SmolLM2Runner)
    runner.cache_host_weights = cache
    runner.weights = weights
    runner._cached_embed = None
    runner._cached_norm_gamma = {}
    runner.config = SimpleNamespace(hidden_size=4, rms_norm_eps=1e-5)
    runner.seen_tables = []

    def act(action, inputs, attrs, *, nbytes=None):
        runner.seen_tables.append(np.asarray(inputs.get("table")) )
        return {"x": np.asarray(inputs["table"])}

    runner._act = act
    return runner


class HostWeightCacheTests(unittest.TestCase):
    def test_cached_embedding_reads_npz_once_and_returns_same_array(self):
        table = np.arange(20, dtype=np.float16).reshape(5, 4)
        weights = _CountingWeights({"embed": table})
        runner = _runner(cache=True, weights=weights)
        runner._cached_embed = np.ascontiguousarray(table)
        runner._embed(np.asarray([1, 3], dtype=np.int32), bucket=2)
        runner._embed(np.asarray([4], dtype=np.int32), bucket=1)
        self.assertEqual(weights.reads, [])
        self.assertIs(runner.seen_tables[0], runner.seen_tables[1])
        np.testing.assert_array_equal(runner.seen_tables[0], table)

    def test_uncached_embedding_reads_npz_every_call(self):
        table = np.arange(20, dtype=np.float16).reshape(5, 4)
        weights = _CountingWeights({"embed": table})
        runner = _runner(cache=False, weights=weights)
        runner._embed(np.asarray([1], dtype=np.int32), bucket=1)
        runner._embed(np.asarray([2], dtype=np.int32), bucket=1)
        self.assertEqual(weights.reads, ["embed", "embed"])

    def test_cached_norm_gamma_reads_each_key_once(self):
        gamma0 = np.ones(4, dtype=np.float16)
        gamma1 = np.arange(4, dtype=np.float16)
        weights = _CountingWeights({"layer.0.input_norm": gamma0,
                                    "layer.0.post_norm": gamma1})
        runner = _runner(cache=True, weights=weights)

        def act(action, inputs, attrs, *, nbytes=None):
            runner.seen_tables.append(np.asarray(inputs["gamma"]))
            return {"y": np.asarray(inputs["x"])}

        runner._act = act
        x = np.zeros((1, 4), dtype=np.float16)
        runner._norm(x, "layer.0.input_norm")
        runner._norm(x, "layer.0.input_norm")
        runner._norm(x, "layer.0.post_norm")
        self.assertEqual(weights.reads, ["layer.0.input_norm",
                                         "layer.0.post_norm"])
        self.assertIs(runner.seen_tables[0], runner.seen_tables[1])

    def test_profile_report_keeps_levels_explicit_and_residuals_nonnegative(self):
        runner = SmolLM2Runner.__new__(SmolLM2Runner)
        runner.profile = True
        runner._profile_us = {
            "top.embed": 10.0,
            "top.layer_total": 100.0,
            "top.final_norm": 2.0,
            "top.lm_head_total": 20.0,
            "top.concat": 1.0,
            "layer.norm_input": 10.0,
            "layer.qkv_plan": 20.0,
            "layer.attention_total": 30.0,
            "attn.qk_layout": 2.0,
            "attn.qk_plan": 5.0,
            "attn.softmax": 3.0,
            "attn.pv_plan": 6.0,
            "attn.merge": 1.0,
            "plan.node": 200.0,
            "plan.alloc": 1.0,
            "plan.call": 190.0,
            "plan.finite": 2.0,
            "plan.stats": 3.0,
            "plan.residual": 4.0,
            "plan_call.lock_wait": 1.0,
            "plan_call.input_prepare": 2.0,
            "plan_call.output_prepare": 3.0,
            "plan_call.binding": 4.0,
            "plan_call.c_runtime": 170.0,
            "plan_call.inner_unattributed": 5.0,
            "plan_call.wrapper_unattributed": 5.0,
            "c_runtime.total": 170.0,
            "c_runtime.submit_wait": 120.0,
            "c_runtime.pack": 10.0,
            "c_runtime.guard": 20.0,
            "c_runtime.download": 5.0,
            "c_runtime.output_sync": 4.0,
            "host_action.us": 50.0,
        }
        runner._profile_count = {"plan.node": 10, "host_action": 20}
        p = runner._profile_report(decode_wall_us=150.0, decode_forwards=2)
        self.assertAlmostEqual(p["top_level_sum_us"], 133.0)
        self.assertAlmostEqual(p["top_level_residual_us"], 17.0)
        self.assertAlmostEqual(p["layer_ops_sum_us"], 60.0)
        self.assertAlmostEqual(p["layer_residual_us"], 40.0)
        self.assertAlmostEqual(p["attention_ops_sum_us"], 17.0)
        self.assertAlmostEqual(p["attention_residual_us"], 13.0)
        self.assertAlmostEqual(p["plan_call_sum_us"], 190.0)
        self.assertAlmostEqual(p["c_runtime_children_sum_us"], 159.0)
        self.assertAlmostEqual(p["c_runtime_residual_us"], 11.0)
        self.assertGreaterEqual(p["plan_node_residual_us"], 0.0)
        self.assertEqual(p["counters"]["decode_forwards"], 2)


if __name__ == "__main__":
    unittest.main()
