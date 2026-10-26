"""Static-repeat group entry: identical results, unchanged submits, no partial publish.

The group entry exists for one reason: a manifest may declare N independent
instances of the same immutable plan (the Tiny Transformer's per-(batch,head)
QK/PV work).  Executing that repeat set through one native call must not change
what the device does, so these tests pin the three properties the optimisation
claims:

* the numbers are bit-identical to N independent calls;
* the number of device submits and tasks does not change;
* a failing repeat publishes nothing, and the failure is reported rather than
  partially committed.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
PLANS = ROOT / "models" / "compiled"

REPEATS = 32


class RepeatGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import load_plan
        from rknpu_backend.runtime.ffi import LibRKPlan
        from rknpu_backend.models import PlanCall
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.plan = load_plan(str(PLANS / "tiny_vit_b8_qk.rkplan.json"))
        cls.call = PlanCall(cls.lib, cls.plan, {}, fd=0)
        rng = np.random.default_rng(20261026)
        cls.inputs = [[np.ascontiguousarray(
            rng.standard_normal((16, 16)).astype(np.float16)) for _ in range(2)]
            for _ in range(REPEATS)]

    @classmethod
    def tearDownClass(cls):
        cls.call.close()

    def _fresh_outputs(self, fill=0.0):
        return [[np.full((16, 16), fill, dtype=np.float16)] for _ in range(REPEATS)]

    def test_group_entry_is_bit_identical_to_independent_calls(self):
        single = self._fresh_outputs()
        for repeat in range(REPEATS):
            self.call.run_into(self.inputs[repeat], single[repeat])
        grouped = self._fresh_outputs()
        report, _ = self.call.run_group_into(self.inputs, grouped)
        self.assertEqual(int(report.repeat_count), REPEATS)
        self.assertEqual(int(report.repeat_completed), REPEATS)
        self.assertEqual(int(report.fail_repeat), 0xFFFFFFFF)
        # Unchanged device work: one task and one submit per repeat.
        self.assertEqual(int(report.task_count), REPEATS)
        self.assertEqual(int(report.submit_count), REPEATS)
        self.assertEqual(int(report.upload_calls), 2 * REPEATS)
        self.assertEqual(int(report.download_calls), REPEATS)
        for repeat in range(REPEATS):
            self.assertTrue(np.array_equal(single[repeat][0], grouped[repeat][0]),
                            f"repeat {repeat} differs from the independent call")

    def test_group_entry_validates_every_repeat(self):
        """A bad binding in the middle of a repeat set is rejected, not skipped."""
        bad = [list(repeat) for repeat in self.inputs]
        poisoned = np.full((16, 16), np.nan, dtype=np.float16)
        bad[REPEATS // 2][1] = poisoned
        from rknpu_backend.errors import BackendError, RK_E_NUMERIC_CONTRACT
        with self.assertRaises(BackendError) as caught:
            self.call.run_group_into(bad, self._fresh_outputs())
        self.assertEqual(caught.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_failing_repeat_publishes_nothing(self):
        """No repeat may be committed when a later repeat fails."""
        bad = [list(repeat) for repeat in self.inputs]
        bad[REPEATS - 1][0] = np.full((16, 16), np.nan, dtype=np.float16)
        sentinel = -12345.0
        outputs = self._fresh_outputs(fill=sentinel)
        from rknpu_backend.errors import BackendError
        with self.assertRaises(BackendError):
            self.call.run_group_into(bad, outputs)
        for repeat, row in enumerate(outputs):
            self.assertTrue(np.all(row[0] == np.float16(sentinel)),
                            f"repeat {repeat} was published despite the failure")

    def test_repeat_lists_must_agree(self):
        from rknpu_backend.errors import BackendError
        with self.assertRaises(BackendError):
            self.call.run_group_into(self.inputs, self._fresh_outputs()[:-1])
        with self.assertRaises(BackendError):
            self.call.run_group_into([], [])

    def test_workspace_recovers_after_a_rejected_group(self):
        """A rejected input is a validation failure, not a poisoned device."""
        bad = [list(repeat) for repeat in self.inputs]
        bad[0][0] = np.full((16, 16), np.inf, dtype=np.float16)
        from rknpu_backend.errors import BackendError
        with self.assertRaises(BackendError):
            self.call.run_group_into(bad, self._fresh_outputs())
        outputs = self._fresh_outputs()
        report, _ = self.call.run_group_into(self.inputs, outputs)
        self.assertEqual(int(report.repeat_completed), REPEATS)
        self.assertTrue(np.isfinite(outputs[0][0]).all())


if __name__ == "__main__":
    unittest.main()
