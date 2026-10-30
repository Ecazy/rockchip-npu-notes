"""Mock execution of explicit guarded M/N/K tail GEMMs."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def load_mock():
    if not Path(MOCK_LIB).is_file():
        raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
    return LibRKPlan(MOCK_LIB)


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_ragged_exec", str(FIXTURES / "tilelang_ragged.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f16(x):
    return np.asarray(x, dtype=np.float16)


def reference(x, w, k):
    """Emulate the planner's ordered 256-wide K chain with FP16 writeback."""
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    first = min(256, k)
    acc = f16(a[:, :first] @ b[:first, :])
    for k0 in range(256, k, 256):
        k1 = min(k, k0 + 256)
        partial = a[:, k0:k1] @ b[k0:k1, :]
        acc = f16(acc.astype(np.float64) + partial)
    return acc


class RaggedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_mock()
        cls.fixture = load_fixture()
        cls.profile = default_profile()

    def setUp(self):
        self._arrays = []
        bos = self.lib.mock_control("rk_mock_live_bos")
        res = self.lib.mock_control("rk_mock_resident")
        self.addCleanup(self.assert_no_leak, bos, res)

    def assert_no_leak(self, bos, res):
        self.assertEqual(self.lib.mock_control("rk_mock_untracked"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_tag_overflow"), 0)
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), bos)
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), res)
        self._arrays.clear()

    def run_case(self, m, n, k, bm, bn, *, rounds=1):
        plan = compile_tilelang(
            self.fixture.ragged_tail(m=m, n=n, k=k, bm=bm, bn=bn),
            profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        rng = np.random.default_rng(73)
        w = f16(rng.normal(0, 0.1, (k, n)))
        self._arrays.append(w)
        constants = [make_binding(b.id, w)
                     for b in plan.bindings if b.role == "constant"]
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        constant_ids = {b.id for b in plan.bindings if b.role == "constant"}
        expected_upload_calls = sum(
            1 for action in plan.actions
            if action.kind == "upload" and
            action.attrs.get("source_binding") not in constant_ids)
        previous = None
        for _round in range(rounds):
            x = f16(rng.normal(0, 0.2, (m, k)))
            self._arrays.append(x)
            y = np.zeros((m, n), dtype=np.float16)
            self._arrays.append(y)
            report = ws.run([make_binding(in_id, x)],
                            [make_binding(out_id, y)])
            np.testing.assert_array_equal(y, reference(x, w, k))
            self.assertEqual(report.upload_calls, expected_upload_calls)
            self.assertEqual(report.download_calls, len(plan.publish))
            if previous is not None:
                self.assertFalse(np.array_equal(previous, y))
            previous = y.copy()
        return plan

    def test_combined_m_n_k_tails_match_reference_over_two_rounds(self):
        plan = self.run_case(13, 33, 257, 16, 64, rounds=2)
        self.assertEqual(plan.accounting.task_count, 2)
        self.assertEqual(len(plan.publish), 1)
        self.assertEqual((plan.publish[0].rows, plan.publish[0].cols), (13, 33))

    def test_multi_grid_ragged_tail_assembly_matches_reference(self):
        plan = self.run_case(257, 257, 300, 128, 128)
        self.assertEqual(plan.accounting.task_count, 18)
        self.assertEqual(len(plan.publish), 9)
        self.assertEqual(sorted((p.rows, p.cols) for p in plan.publish),
                         [(1, 1), (1, 128), (1, 128), (128, 1), (128, 1),
                          (128, 128), (128, 128), (128, 128), (128, 128)])


if __name__ == "__main__":
    unittest.main()
