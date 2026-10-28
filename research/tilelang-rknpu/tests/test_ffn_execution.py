"""Full FFN numeric evidence through planner -> FFI -> rk_workspace_run.

The plan comes from the real planner (`swiglu_module` -> `compile_rkir`) and is
executed by the test-only `librknpu_plan_mock.so`, which shares the production
runtime objects and C ABI but binds the vendor calls to the mock seam.
The reference is computed independently with numpy and copies the declared FP16
narrowing points; it never reuses the mock's helpers.
"""
import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, compile_rkir, compile_tilelang,
                           default_profile, swiglu_module)
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def load_mock():
    from rknpu_backend.runtime.ffi import LibRKPlan
    if not Path(MOCK_LIB).is_file():
        raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
    return LibRKPlan(MOCK_LIB)


def f16(x):
    return np.asarray(x, dtype=np.float16)


def reference(x, wg, wu, wd):
    """Y = (SiLU_f16(X@Wg) * (X@Wu)) @ Wd + X, with FP16 rounding per stage."""
    # RKIR weights are [N, K] (custom-ops convention), so the reference must
    # transpose them: gate = X[d] @ Wg[h, d]^T.
    gate = f16(np.matmul(x.astype(np.float32), wg.astype(np.float32).T))  # narrow 1
    up = f16(np.matmul(x.astype(np.float32), wu.astype(np.float32).T))    # narrow 2
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))            # narrow 3
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))         # narrow 4
    act = f16(silu.astype(np.float32) * up.astype(np.float32))           # narrow 5
    proj = f16(np.matmul(act.astype(np.float32), wd.astype(np.float32).T))  # narrow 6
    return f16(proj.astype(np.float32) + x.astype(np.float32))           # narrow 7


def reordered_swiglu_module(*, m, d, h):
    """Equivalent SwiGLU graph with the two branch nodes emitted up-first.

    The whole-graph ``_match_swiglu`` optimisation expects gate-first, so this
    module exercises the generic per-node dataflow lowering of the same branch,
    shared-input and residual structure without changing its semantics.
    """
    module = swiglu_module(m=m, d=d, h=h)
    values = list(module.values)
    # New op order: up(id 0), gate(id 1), silu, mul, projection, add.
    values[4] = replace(values[4], producer=1)  # Zg produced by gate op id 1
    values[5] = replace(values[5], producer=0)  # Zu produced by up op id 0
    ops = (replace(module.ops[1], id=0),
           replace(module.ops[0], id=1),
           module.ops[2], module.ops[3], module.ops[4], module.ops[5])
    return replace(module, values=tuple(values), ops=ops)


def load_ffn_fixture():
    import importlib.util
    import sys
    fixtures = ROOT / "tests" / "fixtures"
    if str(fixtures) not in sys.path:
        sys.path.insert(0, str(fixtures))
    spec = importlib.util.spec_from_file_location(
        "tl_ffn_execution", str(fixtures / "tilelang_ffn.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_weight_reference(x, wg, wu, wd):
    """Reference for the TileLang fixture's natural ``[K,N]`` weights.

    Six explicit FP16 roundings: gate, up, sigmoid, SiLU, gated product,
    projection, then the residual add result.
    """
    gate = f16(np.matmul(x.astype(np.float32), wg.astype(np.float32)))
    up = f16(np.matmul(x.astype(np.float32), wu.astype(np.float32)))
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))
    act = f16(silu.astype(np.float32) * up.astype(np.float32))
    proj = f16(np.matmul(act.astype(np.float32), wd.astype(np.float32)))
    return f16(proj.astype(np.float32) + x.astype(np.float32))


def make_binding(cls, value_id, array, holder):
    """Build a ctypes binding and keep the numpy backing array alive.

    A dangling pointer here would surface as a bogus "non-finite input" error, so
    the array is retained for the whole native call sequence.
    """
    holder.append(array)   # released when the case's holder goes out of scope
    binding = cls()
    binding.abi_version = 1
    binding.struct_size = __import__("ctypes").sizeof(cls)
    binding.id = value_id
    binding.dtype = 0
    binding.rank = 2
    binding.shape = (array.shape[0], array.shape[1])
    binding.data = array.ctypes.data_as(__import__("ctypes").c_void_p)
    binding.nbytes = array.nbytes
    return binding


# Collected by default on purpose: while the harness is broken it must SHOW UP as
# a failure, never hide behind a skip, so "all green" can never be read as
# "FFN execution verified".
# No WIP switch: this suite runs for real.
class FFNExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_mock()
        cls.profile = default_profile()
        from rknpu_backend.runtime import ffi
        cls.ffi = ffi

    def setUp(self):
        self._arrays = []          # backing store for this test case only
        bos = self.lib.mock_control("rk_mock_live_bos")
        res = self.lib.mock_control("rk_mock_resident")
        # Registered first, so it runs LAST: after every plan/workspace cleanup.
        self.addCleanup(self.assert_no_leak, bos, res)

    def assert_no_leak(self, bos, res):
        # The seam must have recognised every emitted program (no guessing) and
        # never overflowed its table.
        self.assertEqual(self.lib.mock_control("rk_mock_untracked"), 0,
                         "an emitted program was untracked")
        self.assertEqual(self.lib.mock_control("rk_mock_tag_overflow"), 0,
                         "mock tag table overflowed")
        self.assertEqual(self.lib.mock_control("rk_mock_live_bos"), bos,
                         "BO leak")
        self.assertEqual(self.lib.mock_control("rk_mock_resident"), res,
                         "resident bytes leak")
        self._arrays.clear()       # numpy backing arrays released with the case

    def own_plan(self, plan):
        cplan = self.lib.create_plan(plan)
        self.addCleanup(cplan.destroy)
        return cplan

    def own_workspace(self, cplan, constants):
        ws = self.lib.create_workspace(cplan, constants)
        self.addCleanup(ws.destroy)
        return ws

    def constant_bindings(self, plan, tensors):
        """Every constant binding the plan declares, in plan order.

        The planner also emits identity weights for the element-wise stages; they
        must be real identity matrices, otherwise the EW pass would be wrong.
        """
        out = []
        for binding in plan.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                size = binding.shape[0]
                array = f16(np.eye(size, dtype=np.float32))
            else:
                array = tensors[binding.name]
            out.append(make_binding(self.ffi.RKBinding, binding.id, array, self._arrays))
        return out

    def build(self, m, d, h, mode="chain"):
        module = swiglu_module(m=m, d=d, h=h)
        result = compile_rkir(module, profile=self.profile,
                              options=CompileOptions(numeric_mode="rk_f16_tiled",
                                                     allow_approx_lut=True,
                                                     allow_experimental=True,
                                                     submit_mode=mode))
        return result.plan

    def run_generic_module(self, module, m, d, h, seed=41, mode="chain"):
        """Compile without the experimental opt-in and execute per-node.

        The canonical and reordered branch graphs must select the same generic
        path and produce the same numerical result.
        """
        plan = compile_rkir(
            module, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True,
                                   submit_mode=mode)).plan
        self.assertNotIn("swiglu_ffn_v1",
                         {s.recipe_id for s in plan.submits})
        # chain mode fuses the final Linear + residual add; separate mode keeps
        # one Submit per task.
        self.assertEqual(len(plan.submits), 5 if mode == "chain" else 7)
        self.assertEqual(
            [a.template_id for a in plan.actions if a.kind == "npu_task"],
            ["co.mm.f16.v1", "co.mm.f16.v1",
             "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1",
             "co.ew_mul.f16.v1", "co.mm.f16.v1",
             "co.ew_add.f16.v1"])

        cplan = self.own_plan(plan)
        rng = np.random.default_rng(seed)
        x = f16(rng.normal(0, 0.5, (m, d)))
        wg = f16(rng.normal(0, 0.3, (h, d)))
        wu = f16(rng.normal(0, 0.3, (h, d)))
        wd = f16(rng.normal(0, 0.3, (d, h)))
        constants = self.constant_bindings(
            plan, {"Wg": wg, "Wu": wu, "Wd": wd})
        ws = self.own_workspace(cplan, constants)
        y = np.zeros((m, d), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        inputs = [make_binding(self.ffi.RKBinding, in_id, x, self._arrays)]
        outputs = [make_binding(self.ffi.RKBinding, out_id, y, self._arrays)]
        report = ws.run(inputs, outputs)
        np.testing.assert_array_equal(y, reference(x, wg, wu, wd))
        self.assertEqual(report.task_count, 7)
        return plan

    def run_case(self, m, d, h, mode="chain"):
        plan = self.build(m, d, h, mode)
        cplan = self.own_plan(plan)
        rng = np.random.default_rng(7)
        x = f16(rng.normal(0, 0.5, (m, d)))
        wg = f16(rng.normal(0, 0.3, (h, d)))
        wu = f16(rng.normal(0, 0.3, (h, d)))
        wd = f16(rng.normal(0, 0.3, (d, h)))
        constants = self.constant_bindings(plan, {"Wg": wg, "Wu": wu, "Wd": wd})
        ws = self.own_workspace(cplan, constants)
        out = np.full((m, d), 0x23, dtype=np.uint8).view(np.uint8)
        y = np.zeros((m, d), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        inputs = [make_binding(self.ffi.RKBinding, in_id, x, self._arrays)]
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        outputs = [make_binding(self.ffi.RKBinding, out_id, y, self._arrays)]
        report = ws.run(inputs, outputs)
        expected = reference(x, wg, wu, wd)
        np.testing.assert_array_equal(y, expected)
        self.assertGreater(report.task_count, 0)
        return ws, cplan, inputs, outputs, y, report, (x, wg, wu, wd), expected

    def test_full_ffn_matches_reference_chain(self):
        self.run_case(8, 64, 32)

    def test_full_ffn_matches_reference_separate(self):
        self.run_case(8, 64, 32, mode="separate")

    def test_non_square_and_input_update(self):
        ws, _cplan, inputs, outputs, y, _report, tensors, expected = \
            self.run_case(6, 96, 48)
        self.assertEqual(y.shape, (6, 96))
        x, wg, wu, wd = tensors
        # second run with different inputs through the same workspace
        x2 = f16(np.asarray(x, dtype=np.float32) * 0.5 + 0.125)
        inputs[0] = make_binding(self.ffi.RKBinding, inputs[0].id, x2,
                                 self._arrays)
        ws.run(inputs, outputs)
        np.testing.assert_array_equal(y, reference(x2, wg, wu, wd))
        self.assertFalse(np.array_equal(y, expected))

    def test_generic_branch_dataflow_matches_reference(self):
        self.run_generic_module(
            reordered_swiglu_module(m=8, d=64, h=32), 8, 64, 32)

    def test_canonical_swiglu_defaults_to_generic_per_node_path(self):
        self.run_generic_module(
            swiglu_module(m=8, d=64, h=32), 8, 64, 32, seed=43)

    def test_canonical_generic_path_matches_reference_separate(self):
        self.run_generic_module(
            swiglu_module(m=8, d=64, h=32), 8, 64, 32, seed=45,
            mode="separate")

    @unittest.skipUnless(tilelang_available(),
                         "pinned TileLang optional dependency absent")
    def test_real_tilelang_ffn_frontend_to_mock_numeric_closure(self):
        fixture = load_ffn_fixture()
        m, d, h = 8, 64, 32
        kernel = fixture.swiglu_ffn_f16(m=m, d=d, h=h, bk=d)
        plan = compile_tilelang(
            kernel, profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True)).plan
        self.assertEqual(len(plan.submits), 5)
        self.assertNotIn("swiglu_ffn_v1",
                         {s.recipe_id for s in plan.submits})

        cplan = self.own_plan(plan)
        rng = np.random.default_rng(47)
        x = f16(rng.normal(0, 0.2, (m, d)))
        wg = f16(rng.normal(0, 0.1, (d, h)))
        wu = f16(rng.normal(0, 0.1, (d, h)))
        wd = f16(rng.normal(0, 0.1, (h, d)))
        constants = self.constant_bindings(
            plan, {"Wg": wg, "Wu": wu, "Wd": wd})
        ws = self.own_workspace(cplan, constants)
        y = np.zeros((m, d), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        inputs = [make_binding(self.ffi.RKBinding, in_id, x, self._arrays)]
        outputs = [make_binding(self.ffi.RKBinding, out_id, y, self._arrays)]
        report = ws.run(inputs, outputs)
        np.testing.assert_array_equal(y, natural_weight_reference(x, wg, wu, wd))
        self.assertEqual(report.task_count, 7)

    def test_failure_does_not_publish_and_resources_are_released(self):
        plan = self.build(8, 64, 32)
        cplan = self.own_plan(plan)
        rng = np.random.default_rng(11)
        x = f16(rng.normal(0, 0.5, (8, 64)))
        wg = f16(rng.normal(0, 0.3, (32, 64)))
        wu = f16(rng.normal(0, 0.3, (32, 64)))
        wd = f16(rng.normal(0, 0.3, (64, 32)))
        constants = self.constant_bindings(plan, {"Wg": wg, "Wu": wu, "Wd": wd})
        ws = self.own_workspace(cplan, constants)
        live_before = self.lib.mock_control("rk_mock_live_bos")
        y = np.zeros((8, 64), dtype=np.float16)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        inputs = [make_binding(self.ffi.RKBinding, in_id, x, self._arrays)]
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        outputs = [make_binding(self.ffi.RKBinding, out_id, y, self._arrays)]
        ws.run(inputs, outputs)
        from rknpu_backend.errors import BackendError, RK_E_NUMERIC_CONTRACT
        raw = y.view(np.uint8)                      # (8, 64*2) bytes
        snapshot = np.full(raw.shape, 0x5A, dtype=np.uint8)
        raw[:] = snapshot
        self.lib.mock_control("rk_mock_inject_nan", 1)
        try:
            with self.assertRaises(BackendError) as ctx:
                ws.run(inputs, outputs)
            self.assertEqual(ctx.exception.diagnostics[0].code,
                             RK_E_NUMERIC_CONTRACT)
            # the published buffer must be byte-identical to the pre-call image
            np.testing.assert_array_equal(y.view(np.uint8), snapshot)
        finally:
            self.lib.mock_control("rk_mock_inject_nan", 0)
        self.assertGreater(live_before, 0)


if __name__ == "__main__":
    unittest.main()
