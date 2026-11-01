# SPDX-License-Identifier: GPL-3.0-or-later
"""View.valid_region bounds (Python) and the C budget accumulation.

The View cases are refused while the view is built.  The budget cases are
descriptor-level only: they never allocate device memory and are never executed;
they prove the C verifier detects an answer that wrapped.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                           load_plan, matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.storage import CO_GUARD, CO_SLACK
from rknpu_backend.plan import View, _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
REAL = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
UINT64_MAX = 2 ** 64 - 1
RESERVE = CO_SLACK + CO_GUARD


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def base_plan():
    return compile_rkir(matmul_module(m=16, k=16, n=8), profile=PROFILE,
                        options=options()).plan


def rehash(plan):
    object.__setattr__(plan, "plan_hash",
                       sha256_json(plan.to_dict(include_hash=False)))
    return plan


class ViewRegionTests(unittest.TestCase):
    def setUp(self):
        self.plan = base_plan()
        self.view = next(v for v in self.plan.views
                         if v.tensor.rows == 16 and v.tensor.cols == 16)

    def _with_region(self, region, *, bypass_constructor=False):
        """Set the region on one view; optionally skip the Python view check so
        the C verifier is exercised on its own."""
        views = []
        for view in self.plan.views:
            if view.id != self.view.id:
                views.append(view)
            elif bypass_constructor:
                patched = replace(view)
                object.__setattr__(patched, "valid_region", region)
                views.append(patched)
            else:
                views.append(replace(view, valid_region=region))
        tampered = replace(self.plan, plan_hash="")
        object.__setattr__(tampered, "views", tuple(views))
        return rehash(tampered)

    def test_a_region_that_runs_past_the_end_is_refused(self):
        cases = (((10, 10, 0, 16), "is outside the logical tensor"),
                 ((10, 7, 0, 16), "is outside the logical tensor"),
                 ((0, 16, 10, 8), "is outside the logical tensor"),
                 ((17, 1, 0, 8), "starts outside the logical tensor"),
                 ((0, 8, 17, 1), "starts outside the logical tensor"))
        for region, needle in cases:
            with self.subTest(region=region):
                with self.assertRaises(BackendError) as caught:
                    View(self.view.id, self.view.buffer_id,
                         self.view.offset_bytes, self.view.tensor, region)
                self.assertIn(needle,
                              caught.exception.diagnostics[0].message)

    def test_negative_entries_are_refused(self):
        for region in ((-1, 4, 0, 4), (0, -1, 0, 4), (0, 4, -1, 4),
                       (0, 4, 0, -1)):
            with self.subTest(region=region):
                with self.assertRaises(BackendError) as caught:
                    View(self.view.id, self.view.buffer_id,
                         self.view.offset_bytes, self.view.tensor, region)
                self.assertIn("non-negative",
                              caught.exception.diagnostics[0].message)

    def test_boundary_and_interior_regions_stay_legal(self):
        for region in ((6, 10, 0, 16), (2, 4, 3, 5), (0, 16, 0, 16),
                       (16, 0, 0, 0), (0, 0, 0, 0)):
            with self.subTest(region=region):
                view = View(self.view.id, self.view.buffer_id,
                            self.view.offset_bytes, self.view.tensor, region)
                self.assertEqual(view.valid_region, region)

    def test_legal_region_survives_save_and_load(self):
        plan = self._with_region((2, 4, 3, 5))
        _validate_plan(plan)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.to_dict(), plan.to_dict())

    def test_the_buffer_byte_bound_is_still_a_separate_rule(self):
        # The region is legal, but the view no longer fits the buffer.
        buffers = list(self.plan.buffers)
        buffers[self.view.buffer_id] = replace(
            buffers[self.view.buffer_id], capacity=16)
        tampered = replace(self.plan, plan_hash="")
        object.__setattr__(tampered, "buffers", tuple(buffers))
        rehash(tampered)
        with self.assertRaises(BackendError) as caught:
            _validate_plan(tampered)
        self.assertIn("exceeds its logical buffer",
                      caught.exception.diagnostics[0].message)

    @unittest.skipUnless(Path(REAL).is_file(), "host plan library not built")
    def test_c_still_refuses_the_region_the_python_layer_refuses(self):
        lib = LibRKPlan(REAL)
        plan = self._with_region((2, 4, 3, 5))
        cplan = lib.create_plan(plan)
        cplan.destroy()
        for region in ((10, 10, 0, 16), (0, 16, 10, 8), (17, 1, 0, 8)):
            with self.subTest(region=region):
                tampered = self._with_region(region,
                                             bypass_constructor=True)
                with self.assertRaises(BackendError) as caught:
                    lib.create_plan(tampered)
                self.assertIn("valid_region",
                              caught.exception.diagnostics[0].message)

    def test_a_serialised_bad_region_is_refused_on_load(self):
        import json

        plan = self._with_region((2, 4, 3, 5))
        payload = json.loads(json.dumps(plan.to_dict()))
        for view in payload["views"]:
            if view["id"] == self.view.id:
                view["valid_region"] = [10, 10, 0, 16]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.rkplan.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(BackendError) as caught:
                load_plan(str(path))
        self.assertIn("is outside the logical tensor",
                      caught.exception.diagnostics[0].message)


class BudgetOverflowTests(unittest.TestCase):
    """Descriptor-level only: no workspace, no allocation, no device."""

    def setUp(self):
        if not Path(REAL).is_file():
            self.skipTest("host plan library not built")
        self.lib = LibRKPlan(REAL)
        self.plan = base_plan()

    def _capacity_plan(self, capacities):
        buffers = list(self.plan.buffers)
        for index, capacity in enumerate(capacities):
            buffers[index] = replace(buffers[index], slot_id=index,
                                     capacity=capacity, offset=0)
        for index in range(len(capacities), len(buffers)):
            buffers[index] = replace(buffers[index], slot_id=0, offset=0)
        slot_caps = {}
        for buffer in buffers:
            slot_caps[buffer.slot_id] = max(slot_caps.get(buffer.slot_id, 0),
                                            buffer.offset + buffer.capacity)
        wrapped = 0
        for cap in slot_caps.values():
            wrapped = (wrapped + cap + RESERVE) % (2 ** 64)
        command = max(len(s.submit_ids) if hasattr(s, "submit_ids")
                          else len(s.task_ids)
                      for s in self.plan.submits) * 2048 * 8
        accounting = replace(self.plan.accounting,
                             reserved_bytes=wrapped,
                             device_request_bytes=wrapped + command,
                             command_workspace_bytes=command,
                             data_bo_count=len(slot_caps),
                             host_bytes=self.plan.accounting.host_bytes + 4096)
        tampered = replace(self.plan, buffers=tuple(buffers),
                           accounting=accounting, plan_hash="")
        return rehash(tampered), wrapped

    def test_normal_capacities_are_accepted(self):
        plan, _ = self._capacity_plan([4096, 8192, 16384])
        cplan = self.lib.create_plan(plan)
        cplan.destroy()

    def test_a_slot_amount_that_overflows_alone_is_refused(self):
        plan, _ = self._capacity_plan([UINT64_MAX - RESERVE + 1, 8192, 16384])
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(plan)
        self.assertIn("slot capacity overflow",
                      caught.exception.diagnostics[0].message)

    def test_a_wrapped_running_total_is_refused(self):
        # Two huge slots push the total to 2**64 - RESERVE//2; the small third
        # slot then wraps to ~20 KiB, which is *larger* than that slot capacity,
        # so a "sum < last slot capacity" test could not see it.
        plan, wrapped = self._capacity_plan(
            [2 ** 63, 2 ** 63 - RESERVE // 2 - 2 * RESERVE, 4096])
        self.assertLess(wrapped, 4096 + RESERVE)
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(plan)
        self.assertIn("reserved capacity overflow",
                      caught.exception.diagnostics[0].message)

    def test_a_total_that_fills_the_ceiling_is_also_refused(self):
        plan, _ = self._capacity_plan([UINT64_MAX - RESERVE, 8192, 16384])
        with self.assertRaises(BackendError) as caught:
            self.lib.create_plan(plan)
        self.assertIn("capacity overflow",
                      caught.exception.diagnostics[0].message)

    def test_the_python_layer_rejects_a_huge_request_even_earlier(self):
        # A plan built through the public API never reaches the descriptor: the
        # storage stage compares the exact (non-wrapping) total with the profile
        # budget.  This is why the C arithmetic has to be right on its own: a
        # descriptor can arrive from elsewhere.
        with self.assertRaises(BackendError) as caught:
            compile_rkir(matmul_module(m=16, k=16, n=8), profile=PROFILE,
                         options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                device_budget_bytes=4096))
        self.assertIn("device budget exceeded",
                      caught.exception.diagnostics[0].message)


if __name__ == "__main__":
    unittest.main()
