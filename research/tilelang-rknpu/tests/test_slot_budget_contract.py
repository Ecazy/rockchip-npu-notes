"""One slot must not carry more buffers than the C verifier admits.

F2: a planner-produced tile_m=1 plan passed Python verification and was then
refused by rk_plan_create() with 'too many buffers share one slot'.  The bound
is a verifier bookkeeping limit, so the planner now keeps every slot below it
and plan verification re-checks the same rule.
"""
from __future__ import annotations

import re
import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_SCHEMA, compile_rkir,
                           default_profile, matmul_module)
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.verify import verify_plan
from rknpu_backend.plan import MAX_BUFFERS_PER_SLOT

PROFILE = default_profile()
RESEARCH = Path(__file__).resolve().parents[2]
HEADER = RESEARCH / "custom-ops" / "include" / "rk_plan.h"


def options(**extra):
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                          **extra)


def schedule(tile_m, tile_n):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": "m_major", "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def declared(module, tile_m, tile_n):
    attrs = dict(module.ops[0].attrs)
    attrs["source_schedule"] = schedule(tile_m, tile_n)
    attrs["source_k_rounding"] = "explicit_per_k_fp16"
    return replace(module, ops=(replace(module.ops[0], attrs=attrs),))


def peak_per_slot(plan):
    counts = {}
    for buffer in plan.buffers:
        counts[buffer.slot_id] = counts.get(buffer.slot_id, 0) + 1
    return max(counts.values())


class SlotBudgetContractTests(unittest.TestCase):
    def test_python_bound_matches_the_c_header(self):
        text = HEADER.read_text()
        match = re.search(r"#define RK_MAX_BUFFERS_PER_SLOT (\d+)u", text)
        self.assertIsNotNone(match, "RK_MAX_BUFFERS_PER_SLOT missing")
        self.assertEqual(int(match.group(1)), MAX_BUFFERS_PER_SLOT)

    def test_tile_m_one_plan_stays_inside_one_slot(self):
        # Original F2 case: 64x512x384 with an explicit 1x256 tile.
        plan = compile_rkir(declared(matmul_module(m=64, k=512, n=384), 1, 256),
                            profile=PROFILE, options=options()).plan
        self.assertLessEqual(peak_per_slot(plan), MAX_BUFFERS_PER_SLOT)
        self.assertEqual(plan.accounting.task_count, 256)
        # The upper-layer requirement is untouched: every task still records
        # the declared 1x256 block and the emitted rows are the tail of 1.
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        self.assertTrue(all(task.attrs["applied_schedule"]
                            == schedule(1, 256) for task in tasks))
        # n = 384 with a 256-wide nominal block: one full block and one tail.
        self.assertEqual({(p.rows, p.cols) for p in plan.publish},
                         {(1, 256), (1, 128)})

    def test_splitting_a_slot_does_not_touch_alias_safety(self):
        plan = compile_rkir(declared(matmul_module(m=64, k=512, n=384), 1, 256),
                            profile=PROFILE, options=options()).plan
        # Buffers sharing a slot must still be time-disjoint.
        by_slot = {}
        for buffer in plan.buffers:
            by_slot.setdefault(buffer.slot_id, []).append(buffer)
        for group in by_slot.values():
            ordered = sorted(group, key=lambda b: b.lifetime[0])
            for left, right in zip(ordered, ordered[1:]):
                self.assertLess(left.lifetime[1], right.lifetime[0])

    def test_verification_refuses_a_slot_over_the_bound(self):
        # The 1x256 plan has a long serial chain, so plenty of buffers are
        # pairwise time-disjoint and can be packed into one slot.
        plan = compile_rkir(declared(matmul_module(m=64, k=512, n=384), 1, 256),
                            profile=PROFILE, options=options()).plan
        # Pick buffers that are pairwise time-disjoint, so the only broken rule
        # is the count bound itself (alias overlap is checked separately).
        crowded_ids = []
        for buffer in plan.buffers:
            if all(buffer.lifetime[1] < other.lifetime[0] or
                   other.lifetime[1] < buffer.lifetime[0]
                   for other in crowded_ids):
                crowded_ids.append(buffer)
            if len(crowded_ids) == MAX_BUFFERS_PER_SLOT + 1:
                break
        self.assertEqual(len(crowded_ids), MAX_BUFFERS_PER_SLOT + 1)
        remaining = iter(range(1000, 1000 + len(plan.buffers)))
        crowded = replace(plan, buffers=tuple(
            replace(buffer, slot_id=0) if buffer in crowded_ids
            else replace(buffer, slot_id=next(remaining))
            for buffer in plan.buffers))
        with self.assertRaises(BackendError) as caught:
            verify_plan(crowded)
        diagnostic = caught.exception.diagnostics[0]
        self.assertEqual(caught.exception.code, RK_E_SCHEMA)
        self.assertEqual(diagnostic.actual["buffers_in_slot"],
                         MAX_BUFFERS_PER_SLOT + 1)
        self.assertEqual(diagnostic.required["max_buffers_per_slot"],
                         MAX_BUFFERS_PER_SLOT)

    def test_narrow_tiles_keep_working_and_stay_inside_the_bound(self):
        for tile_m, expected_tasks in ((2, 128), (4, 64), (16, 16)):
            with self.subTest(tile_m=tile_m):
                plan = compile_rkir(
                    declared(matmul_module(m=64, k=512, n=384), tile_m, 256),
                    profile=PROFILE, options=options()).plan
                self.assertEqual(plan.accounting.task_count, expected_tasks)
                self.assertLessEqual(peak_per_slot(plan),
                                     MAX_BUFFERS_PER_SLOT)


if __name__ == "__main__":
    unittest.main()