"""Submit failure marks the workspace poisoned (specific.md contract).

Contract under test, through `rk_workspace_run` only:
  * first unclassified submit failure -> RK_E_DEVICE, output not published,
    workspace poisoned;
  * a later run -> RK_E_POISONED and no further submission;
  * destroy -> RK_E_POISONED and the handle plus every still-reachable resource
    is retained (never freed here);
  * a classified safe failure is NOT covered here yet (no such case exists in
    this file); only the unclassified path is exercised.

The mock library is used as the device; every assertion is about the workspace
policy, not about mock internals.
"""
import ctypes
import json
import os
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_rkir, default_profile, matmul_module
from rknpu_backend.errors import (BackendError, RK_E_DEVICE,
                                  RK_E_NUMERIC_CONTRACT, RK_E_POISONED,
                                  RK_E_SHAPE)
from rknpu_backend.runtime.ffi import RKBinding

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def f16(x):
    return np.asarray(x, dtype=np.float16)


def _child_scenario():
    """Run the poisoned scenario in a throw-away process.

    The retained BOs are deliberately NOT freed, so the resources must die with
    this process instead of polluting the pytest process.
    """
    from rknpu_backend.runtime.ffi import LibRKPlan
    lib = LibRKPlan(MOCK_LIB)
    lib.mock_control("rk_mock_reset")
    plan = compile_rkir(matmul_module(m=8, k=256, n=32), profile=default_profile(),
                        options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    cplan = lib.create_plan(plan)
    keep = []

    def bind(value_id, array):
        keep.append(array)
        b = RKBinding()
        b.abi_version, b.struct_size = 1, ctypes.sizeof(RKBinding)
        b.id, b.dtype, b.rank = value_id, 0, 2
        b.shape = (array.shape[0], array.shape[1])
        b.data = array.ctypes.data_as(ctypes.c_void_p)
        b.nbytes = array.nbytes
        return b

    rng = np.random.default_rng(5)
    w = f16(rng.normal(0, 0.3, (256, 32)))
    ws = lib.create_workspace(
        cplan, [bind(b.id, w) for b in plan.bindings if b.role == "constant"])
    x = f16(rng.normal(0, 0.5, (8, 256)))
    y = np.zeros((8, 32), dtype=np.float16)
    inputs = [bind(next(b.id for b in plan.bindings if b.role == "input"), x)]
    outputs = [bind(next(b.id for b in plan.bindings if b.role == "output"), y)]

    failures = []

    def check(cond, what):
        if not cond:
            failures.append(what)

    ws.run(inputs, outputs)                       # healthy baseline
    lib.mock_control("rk_mock_fail_at", lib.mock_control("rk_mock_execs") + 1)
    sentinel = np.full(y.view(np.uint8).shape, 0x5A, dtype=np.uint8)
    y.view(np.uint8)[:] = sentinel
    try:
        ws.run(inputs, outputs)
        failures.append("failure run did not raise")
    except BackendError as exc:
        check(exc.diagnostics[0].code == RK_E_DEVICE, "expected RK_E_DEVICE")
    check(np.array_equal(y.view(np.uint8), sentinel), "output was published")
    execs_after = lib.mock_control("rk_mock_execs")
    bos = lib.mock_control("rk_mock_live_bos")
    resident = lib.mock_control("rk_mock_resident")
    try:
        ws.run(inputs, outputs)
        failures.append("poisoned run did not raise")
    except BackendError as exc:
        check(exc.diagnostics[0].code == RK_E_POISONED, "expected RK_E_POISONED")
    check(lib.mock_control("rk_mock_execs") == execs_after, "resubmitted after poison")
    try:
        ws.destroy()
        failures.append("destroy did not raise")
    except BackendError as exc:
        check(exc.diagnostics[0].code == RK_E_POISONED, "destroy expected RK_E_POISONED")
    check(bool(ws.handle), "destroy dropped the handle")
    check(lib.mock_control("rk_mock_live_bos") == bos, "BO set changed")
    check(lib.mock_control("rk_mock_resident") == resident, "resident changed")
    # NOTE: ws is intentionally NOT destroyed: the contract retains it, and this
    # process is about to exit.  The resources are "retained per contract" and the
    # child "exited"; that is NOT a normal destruction nor a device recovery.
    result = {"scenario": "unclassified_submit_failure_poisons",
              "status": "ok" if not failures else "fail",
              "live_bos_retained": bos, "resident_retained": resident,
              "execs_at_failure": execs_after,
              "handle_retained": bool(ws.handle),
              "failures": failures}
    print("POISON-RESULT " + json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if os.environ.get("RKNPU_POISON_CHILD") == "1":
    import sys
    sys.exit(_child_scenario())


class WorkspacePoisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rknpu_backend.runtime.ffi import LibRKPlan
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.ffi = __import__("rknpu_backend.runtime.ffi", fromlist=["x"])

    def setUp(self):
        self._keep = []
        self.lib.mock_control("rk_mock_reset")
        self._baseline = (self.lib.mock_control("rk_mock_live_bos"),
                          self.lib.mock_control("rk_mock_resident"))

    def bind(self, value_id, array):
        self._keep.append(array)
        b = RKBinding()
        b.abi_version, b.struct_size = 1, ctypes.sizeof(RKBinding)
        b.id, b.dtype, b.rank = value_id, 0, 2
        b.shape = (array.shape[0], array.shape[1])
        b.data = array.ctypes.data_as(ctypes.c_void_p)
        b.nbytes = array.nbytes
        return b

    def own_workspace(self, cplan, constants):
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)
        return ws

    def assert_baseline(self, bos, res):
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), bos, "BO leak")
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), res, "resident leak")
        self.lib.mock_control("rk_mock_reset")

    def constants_for(self, plan):
        """Constant bindings kept alive with the test case."""
        cached = getattr(self, "_consts", None)
        if cached is not None and getattr(self, "_consts_plan", None) is plan:
            return cached
        rng = np.random.default_rng(5)
        out = []
        for b in plan.bindings:
            if b.role != "constant":
                continue
            arr = (f16(np.eye(b.shape[0])) if b.name.startswith("__identity_")
                   else f16(rng.normal(0, 0.3, b.shape)))
            out.append(self.bind(b.id, arr))
        self._consts, self._consts_plan = out, plan
        return out

    def build(self):
        plan = compile_rkir(matmul_module(m=8, k=256, n=32), profile=default_profile(),
                            options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
        cplan = self.lib.create_plan(plan)
        self._plan_module = plan
        rng = np.random.default_rng(5)
        w = f16(rng.normal(0, 0.3, (256, 32)))
        consts = [self.bind(b.id, w) for b in plan.bindings if b.role == "constant"]
        self._consts, self._consts_plan = consts, plan
        ws = self.lib.create_workspace(cplan, consts)
        x = f16(rng.normal(0, 0.5, (8, 256)))
        y = np.zeros((8, 32), dtype=np.float16)
        inputs = [self.bind(next(b.id for b in plan.bindings if b.role == "input"), x)]
        outputs = [self.bind(next(b.id for b in plan.bindings if b.role == "output"), y)]
        return cplan, ws, inputs, outputs, y

    def test_binding_error_before_submit_is_safe(self):
        """A binding/validation failure happens before any emitter call, so the
        workspace stays usable: the executor is never entered, nothing is
        published, and the workspace can still run and be destroyed normally."""
        cplan, ws, inputs, outputs, y = self.build()
        ws.run(inputs, outputs)                       # healthy baseline
        execs_before = self.lib.mock_control("rk_mock_execs")
        sentinel = np.full(y.view(np.uint8).shape, 0x5A, dtype=np.uint8)
        y.view(np.uint8)[:] = sentinel
        # wrong shape for the input binding: rejected before any submission
        bad = np.zeros((8, 128), dtype=np.float16)
        bad_inputs = [self.bind(inputs[0].id, bad)]
        with self.assertRaises(BackendError) as ctx:
            ws.run(bad_inputs, outputs)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE,
                         ctx.exception.diagnostics[0].message)
        self.assertEqual(self.lib.mock_control("rk_mock_execs"), execs_before,
                         "no submit may happen on a pre-submit failure")
        np.testing.assert_array_equal(y.view(np.uint8), sentinel)
        # the same workspace is still healthy
        ws.run(inputs, outputs)
        self.assertNotEqual(y.view(np.uint8).tobytes(), sentinel.tobytes())
        ws.destroy()
        self.assertFalse(ws.handle, "a healthy destroy releases the handle")
        # rebuild contract: a NEW workspace (not just a plan) must run and destroy
        ws2 = self.own_workspace(cplan, self.constants_for(self._plan_module))
        ws2.run(inputs, outputs)
        ws2.destroy()
        self.assertFalse(ws2.handle)
        cplan.destroy()
        self.assert_baseline(*self._baseline)

    def test_upload_prep_failure_is_safe_presubmit_and_frees_all_bos(self):
        """An upload BO-prep failure occurs before any NPU submit.  It must not
        poison or reuse the command/task BOs and destroy must release the full
        data+command BO set."""
        cplan, ws, inputs, outputs, y = self.build()
        execs_before = self.lib.mock_control("rk_mock_execs")
        prep_before = self.lib.mock_control("rk_mock_prep_calls")
        sentinel = np.full(y.view(np.uint8).shape, 0x4C, dtype=np.uint8)
        y.view(np.uint8)[:] = sentinel
        self.lib.mock_control("rk_mock_fail_prep_at", prep_before + 1)
        with self.assertRaises(BackendError) as ctx:
            ws.run(inputs, outputs)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_DEVICE)
        self.lib.mock_control("rk_mock_fail_prep_at", 0)
        self.assertEqual(self.lib.mock_control("rk_mock_execs"), execs_before)
        np.testing.assert_array_equal(y.view(np.uint8), sentinel)
        ws.run(inputs, outputs)
        ws.destroy()
        self.assertFalse(ws.handle)
        cplan.destroy()
        self.assert_baseline(*self._baseline)

    def test_nonfinite_input_is_a_safe_presubmit_failure(self):
        """Input validation fails before upload/submit; destroy must still free
        both data and command BOs and the workspace must stay reusable."""
        cplan, ws, inputs, outputs, y = self.build()
        execs_before = self.lib.mock_control("rk_mock_execs")
        sentinel = np.full(y.view(np.uint8).shape, 0x6D, dtype=np.uint8)
        y.view(np.uint8)[:] = sentinel
        bad = np.full(inputs[0].shape, np.nan, dtype=np.float16)
        with self.assertRaises(BackendError) as ctx:
            ws.run([self.bind(inputs[0].id, bad)], outputs)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)
        self.assertEqual(self.lib.mock_control("rk_mock_execs"), execs_before)
        np.testing.assert_array_equal(y.view(np.uint8), sentinel)
        ws.run(inputs, outputs)  # no poisoning: the same workspace is reusable
        ws.destroy()
        self.assertFalse(ws.handle)
        cplan.destroy()
        self.assert_baseline(*self._baseline)

    def test_unclassified_submit_failure_poisons_and_retains(self):
        """The scenario runs in a child process: the retained BOs die with it."""
        import subprocess
        import sys
        before_bos = self.lib.mock_control("rk_mock_live_bos")
        before_res = self.lib.mock_control("rk_mock_resident")
        env = dict(os.environ, RKNPU_POISON_CHILD="1", RKNPU_PLAN_MOCK_LIB=MOCK_LIB)
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                              capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(proc.returncode, 0,
                         f"child failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [l for l in proc.stdout.splitlines() if l.startswith("POISON-RESULT ")]
        self.assertEqual(len(lines), 1, "the worker must emit exactly one result")
        result = json.loads(lines[0][len("POISON-RESULT "):])
        for key in ("scenario", "status", "live_bos_retained", "resident_retained",
                    "execs_at_failure", "handle_retained", "failures"):
            self.assertIn(key, result, f"result is missing {key}")
        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertTrue(result["handle_retained"])
        # "retained per contract", not destroyed and not recovered
        self.assertGreater(result["live_bos_retained"], 0)
        self.assertGreater(result["resident_retained"], 0)
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), before_bos,
                         "the parent process must not inherit retained resources")
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), before_res)

if __name__ == "__main__":
    unittest.main()
