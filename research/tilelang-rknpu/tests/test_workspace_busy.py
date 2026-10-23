"""A concurrent run on one workspace must return RK_E_BUSY, not enter twice."""
import os
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, RK_E_BUSY, compile_rkir,
                           default_profile, matmul_module)
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


class WorkspaceBusyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cls.lib = LibRKPlan(MOCK_LIB)

    def test_concurrent_run_returns_busy_and_does_not_publish_second(self):
        baseline_bos = self.lib.mock_control("rk_mock_live_bos")
        baseline_res = self.lib.mock_control("rk_mock_resident")
        plan = compile_rkir(
            matmul_module(m=13, k=64, n=33), profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        cplan = self.lib.create_plan(plan)
        rng = np.random.default_rng(101)
        a = np.asarray(rng.normal(0, 0.3, (13, 64)), dtype=np.float16)
        b = np.asarray(rng.normal(0, 0.2, (64, 33)), dtype=np.float16)
        constants = [make_binding(spec.binding_id, b)
                     for spec in plan.constants if spec.name == "B"]
        ws = self.lib.create_workspace(cplan, constants, fd=0)
        in_id = next(x.id for x in plan.bindings if x.role == "input")
        out_id = next(x.id for x in plan.bindings if x.role == "output")
        y1 = np.zeros((13, 33), dtype=np.float16)
        y2 = np.full((13, 33), np.float16(3.25))
        second_snapshot = y2.tobytes()
        outcome = {}

        def first_run():
            try:
                ws.run([make_binding(in_id, a)], [make_binding(out_id, y1)])
                outcome["first"] = "PASS"
            except BaseException as exc:  # surfaced by the assertion below
                outcome["first"] = exc

        self.lib.mock_control("rk_mock_delay_us", 500_000)
        worker = threading.Thread(target=first_run)
        worker.start()
        time.sleep(0.05)
        try:
            with self.assertRaises(BackendError) as ctx:
                ws.run([make_binding(in_id, a)],
                       [make_binding(out_id, y2)])
            self.assertEqual(ctx.exception.code, RK_E_BUSY)
        finally:
            worker.join(timeout=5)
            self.lib.mock_control("rk_mock_delay_us", 0)
        self.assertEqual(outcome.get("first"), "PASS", outcome.get("first"))
        self.assertEqual(y2.tobytes(), second_snapshot)
        expected = np.asarray(a.astype(np.float64) @ b.astype(np.float64),
                              dtype=np.float16)
        np.testing.assert_array_equal(y1, expected)
        ws.destroy()
        cplan.destroy()
        self.assertEqual(self.lib.mock_control("rk_mock_untracked"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_tag_overflow"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), baseline_bos)
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), baseline_res)


if __name__ == "__main__":
    unittest.main()
