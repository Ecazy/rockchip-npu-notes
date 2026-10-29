"""Focused tests for the SmolLM2 Host actions and request boundaries."""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend.errors import (RK_E_INVALID, RK_E_SHAPE,
                                  RK_E_UNSUPPORTED, BackendError)
from rknpu_backend.hybrid import HOST_ACTIONS
from rknpu_backend.smollm import SmolLM2Runner


def act(name, inputs, attrs):
    return HOST_ACTIONS[name].func(inputs, attrs)


class SmolLM2HostActionTests(unittest.TestCase):
    def test_embedding_bucket_and_bounds(self):
        table = np.arange(20, dtype=np.float16).reshape(5, 4)
        out = act("smollm.embed_padded.v1",
                  {"token_ids": np.asarray([1, 3], dtype=np.int32),
                   "table": table},
                  {"bucket": 4, "hidden": 4})["x"]
        self.assertEqual(out.shape, (4, 4))
        np.testing.assert_array_equal(out[:2], table[[1, 3]])
        np.testing.assert_array_equal(out[2:], np.zeros((2, 4), np.float16))
        with self.assertRaises(BackendError) as ctx:
            act("smollm.embed_padded.v1",
                {"token_ids": np.asarray([5], dtype=np.int32), "table": table},
                {"bucket": 2, "hidden": 4})
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_rope_position_zero_is_identity_and_half_split(self):
        x = np.arange(3 * 2 * 4, dtype=np.float16).reshape(3, 2, 4) / 7
        y = act("smollm.rope_half.v1",
                {"x": x, "positions": np.asarray([0, 1, 7],
                                                 dtype=np.int32)},
                {"theta": 100000.0, "interleaved": False})["y"]
        np.testing.assert_array_equal(y[0], x[0])
        self.assertEqual(y.dtype, np.float16)
        with self.assertRaises(BackendError) as ctx:
            act("smollm.rope_half.v1",
                {"x": x, "positions": np.asarray([0, 1, 2],
                                                 dtype=np.int32)},
                {"theta": 100000.0, "interleaved": True})
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_gqa_layout_merge_roundtrip(self):
        rng = np.random.default_rng(5)
        q = rng.standard_normal((4, 9, 64)).astype(np.float16)
        qh = act("smollm.qk_layout_gqa.v1", {"q": q},
                 {"q_heads": 9, "kv_heads": 3, "head_dim": 64})["a"]
        self.assertEqual(qh.shape, (3, 3 * 4, 64))
        contexts = np.zeros((3, 12, 64), dtype=np.float16)
        merged = act("smollm.merge_gqa_heads.v1", {"contexts": contexts},
                     {"tokens": 4, "q_heads": 9, "kv_heads": 3,
                      "head_dim": 64})["merged"]
        self.assertEqual(merged.shape, (4, 9 * 64))
        # qk_layout orders rows as q_local * tokens + token.
        for g in range(3):
            for h in range(3):
                np.testing.assert_array_equal(qh[g, h * 4:(h + 1) * 4],
                                              q[:, g * 3 + h])

    def test_kv_append_is_in_place_and_rejects_overflow(self):
        ck = np.zeros((3, 8, 4), dtype=np.float16)
        cv = np.zeros_like(ck)
        k = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
        v = k + 100
        out = act("smollm.kv_append_inplace.v1",
                  {"cache_k": ck, "cache_v": cv, "k": k, "v": v,
                   "start": np.asarray([6], dtype=np.int32)},
                  {"capacity": 8})
        self.assertIs(out["cache_k"], ck)
        np.testing.assert_array_equal(ck[:, 6:8], k.transpose(1, 0, 2))
        with self.assertRaises(BackendError) as ctx:
            act("smollm.kv_append_inplace.v1",
                {"cache_k": ck, "cache_v": cv, "k": k, "v": v,
                 "start": np.asarray([7], dtype=np.int32)},
                {"capacity": 8})
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_swiglu_act_only_action_matches_legacy_bit_exactly(self):
        rng = np.random.default_rng(31)
        packed = np.ascontiguousarray(
            rng.normal(size=(42, 3072)).astype(np.float16))
        legacy = act("swiglu.packed_gate_up.v1", {"gate_up": packed},
                     {})["act"]
        fused = act("swiglu.packed_gate_up_act.v1", {"gate_up": packed},
                    {})["act"]
        np.testing.assert_array_equal(legacy, fused)

    def test_causal_softmax_masks_padded_rows_and_columns(self):
        scores = np.zeros((3, 5), dtype=np.float16)
        scores[0, :2] = [1.0, 1.0]
        scores[1, :1] = [7.0]
        probs = act("smollm.causal_softmax.v1",
                    {"scores": scores,
                     "valid_lengths": np.asarray([2, 1, 0], dtype=np.int32)},
                    {"scale": 1.0 / math.sqrt(64)})["probs"]
        np.testing.assert_allclose(probs[0, :2], [0.5, 0.5], atol=2e-3)
        self.assertEqual(float(probs[1, 0]), 1.0)
        np.testing.assert_array_equal(probs[1, 1:], 0)
        np.testing.assert_array_equal(probs[2], 0)
        with self.assertRaises(BackendError) as ctx:
            act("smollm.causal_softmax.v1",
                {"scores": scores,
                 "valid_lengths": np.asarray([1, 2], dtype=np.int32)},
                {"scale": 1.0})
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_logits_concat_and_greedy(self):
        shards = []
        for i in range(12):
            shard = np.full((2, 4), -100 + i, dtype=np.float16)
            shard[:, i % 4] = i * 10
            shards.append(shard)
        inputs = {f"shard{i}": x for i, x in enumerate(shards)}
        logits = act("smollm.concat_logits.v1", inputs,
                     {"rows": 2, "vocab_size": 48,
                      "shards": 12})["logits"]
        self.assertEqual(logits.shape, (2, 48))
        np.testing.assert_array_equal(logits[:, :4], shards[0])
        np.testing.assert_array_equal(logits[:, 44:48], shards[11])
        tokens = act("smollm.greedy_sample.v1",
                     {"logits": logits}, {"vocab_size": 48})["token_ids"]
        self.assertEqual(tokens.dtype, np.int32)
        self.assertEqual(tokens.tolist(), [47, 47])

    @unittest.skipUnless(
        os.environ.get("RKNPU_SMOLLM_SLOW_TESTS") == "1",
        "set RKNPU_SMOLLM_SLOW_TESTS=1 for the package-sized mock replay")
    def test_persistent_kv_reset_replays_identical(self):
        import hashlib
        import json
        package = ROOT / ".data" / "models" / "smollm2_135m_instruct_package"
        if not (package / "manifest.json").is_file():
            self.skipTest("local SmolLM2 package not compiled")
        mock = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
            ROOT.parent / "custom-ops" / "build-host-mvp" /
            "librknpu_plan_mock.so")
        if not Path(mock).is_file():
            self.skipTest("mock library not built")
        from rknpu_backend.runtime.ffi import LibRKPlan
        from rknpu_backend.models import RunStats
        lib = LibRKPlan(mock)
        stats = RunStats()
        runner = SmolLM2Runner(lib, package, fd=0,
                               stats=stats, layer_indices=(0,))
        try:
            def one_round():
                runner.reset()
                runner.prefill([1])
                first = runner.decode_one(9690)
                second = runner.decode_one(198)
                return (np.asarray(first["logits"]).copy(),
                        np.asarray(second["logits"]).copy())

            a0, a1 = one_round()
            b0, b1 = one_round()
            np.testing.assert_array_equal(a0, b0)
            np.testing.assert_array_equal(a1, b1)
            self.assertGreater(stats.kv_state_init_calls, 0)
            self.assertGreater(stats.kv_state_append_calls, 0)
            self.assertGreater(stats.kv_state_clear_calls, 0)
            # Incremental state updates are tiny compared with a full
            # per-token K/V upload of all 30 layers.
            self.assertLess(stats.kv_state_append_bytes,
                            stats.kv_state_init_bytes)
        finally:
            runner.close()

    def test_unknown_action_and_capacity_fail_closed(self):
        with self.assertRaises(BackendError) as ctx:
            SmolLM2Runner._act(object(), "smollm.not_registered.v1", {}, {})
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)
        runner = SmolLM2Runner.__new__(SmolLM2Runner)
        runner.capacity = 8
        runner._ensure_capacity(8)
        with self.assertRaises(BackendError) as ctx:
            runner._ensure_capacity(9)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)


if __name__ == "__main__":
    unittest.main()
