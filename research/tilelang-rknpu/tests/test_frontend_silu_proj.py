"""End-to-end frontend target: MatMul -> SiLU -> projection + residual.

The fixture is a real TileLang PrimFunc with two GEMMs and five explicit FP16
narrowing points.  The extractor must return an SSA RKIR graph, not a single
matmul with silently dropped neighbours.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_UNSUPPORTED, compile_tilelang, default_profile)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import (F16_NARROWING_NAMES,
                                                     extract,
                                                     tilelang_available)
from rknpu_backend.ir import NumericContract

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_silu_proj", str(FIXTURES / "tilelang_silu_proj.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(**kwargs):
    kwargs.setdefault("numeric_mode", "rk_f16_tiled")
    kwargs.setdefault("allow_approx_lut", True)
    return CompileOptions(**kwargs)


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class SiluProjKernelTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixtures = load()

    def test_fixture_has_the_declared_narrowing_points(self):
        """The FP16 narrowing points are checked in the TIR, not by name."""
        text = self.fixtures.silu_proj().script()
        self.assertEqual(text.count("T.gemm("), 2)
        # five explicit fp16 intermediates, matching the documented points
        for name in ("z16", "g16", "s16", "p16"):
            self.assertIn(name, text)
        self.assertIn("float16", text)

    def test_multi_gemm_program_extracts_to_ssa_rkir(self):
        module = extract(self.fixtures.silu_proj(), options=options(),
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "silu", "matmul", "add"])
        self.assertEqual([(v.name, v.type.shape) for v in module.values],
                         [("X", (13, 64)), ("W1", (64, 32)),
                          ("W2", (32, 64)), ("Z", (13, 32)),
                          ("S", (13, 32)), ("P", (13, 64)),
                          ("Y", (13, 64))])
        # The second GEMM consumes the first matmul's activation value, not a
        # branch and not a host-visible copy.
        self.assertEqual(module.ops[1].inputs, (module.ops[0].outputs[0],))
        self.assertEqual(module.ops[2].inputs[0], module.ops[1].outputs[0])
        self.assertEqual(module.ops[3].inputs,
                         (module.ops[2].outputs[0], module.inputs[0]))
        self.assertEqual(module.outputs, (module.ops[3].outputs[0],))

        points: set[str] = set()
        for op in module.ops:
            contract = NumericContract.from_dict(op.attrs["numeric_contract"])
            points.update(contract.narrowing_points)
        points.discard("k_tile_boundary")
        self.assertEqual(points, set(F16_NARROWING_NAMES))
        self.assertEqual(len(points), 5)

    def test_compile_plan_lowers_each_node_and_keeps_the_middle_on_device(self):
        result = compile_tilelang(self.fixtures.silu_proj(),
                                  profile=self.profile, options=options())
        plan = result.plan
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        self.assertEqual([a.template_id for a in tasks],
                         ["co.mm.f16.v1", "co.lut.sigmoid.f16.v1",
                          "co.ew_mul.f16.v1", "co.mm.f16.v1",
                          "co.ew_add.f16.v1"])
        self.assertEqual(len(plan.submits), 4)
        # Four external/weight uploads plus two identity weights; exactly one
        # final download.  The two GEMM intermediates never round-trip.
        uploads = [a for a in plan.actions if a.kind == "upload"]
        self.assertEqual(len(uploads), 5)
        self.assertEqual(sum(a.kind == "download" for a in plan.actions), 1)
        # The C runtime stages all Host uploads before any NPU submit.  Keeping
        # every upload action ahead of the first task makes action-id liveness
        # and that Host-first phase agree (identity weights included).
        self.assertLess(max(a.id for a in uploads), min(a.id for a in tasks))
        self.assertEqual([a.id for a in tasks],
                         list(range(tasks[0].id, tasks[-1].id + 1)))
        self.assertEqual(result.report["source"]["narrowing_points"],
                         list(F16_NARROWING_NAMES))

        mm1, lut, act_mul, mm2, add = tasks
        self.assertEqual(plan.views[lut.inputs[0]].buffer_id,
                         plan.views[mm1.outputs[0]].buffer_id)
        self.assertEqual(plan.views[mm2.inputs[0]].buffer_id,
                         plan.views[act_mul.outputs[0]].buffer_id)
        self.assertEqual(plan.views[mm2.outputs[0]].buffer_id,
                         plan.views[add.inputs[0]].buffer_id)

        # Residual liveness: X's device buffer must stay alive until the add.
        x_binding = next(b.id for b in plan.bindings if b.name == "X")
        x_upload = next(a for a in plan.actions
                        if a.kind == "upload" and
                        a.attrs.get("source_binding") == x_binding)
        x_buffer = plan.buffers[plan.views[x_upload.outputs[0]].buffer_id]
        self.assertGreaterEqual(x_buffer.lifetime[1], add.id)

    def test_wrong_branch_is_rejected_by_def_use(self):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(self.fixtures.silu_proj_wrong_branch(),
                             profile=self.profile, options=options())
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
        self.assertIn("wrong branch", diagnostic.message)

    def test_missing_fp16_narrowing_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(self.fixtures.silu_proj_missing_narrowing(),
                             profile=self.profile, options=options())
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_NUMERIC_CONTRACT)
        self.assertIn("narrowing", diagnostic.message)

    def test_unsupported_elementwise_op_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(self.fixtures.silu_proj_unsupported_op(),
                             profile=self.profile, options=options())
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
        self.assertIn("tirx.tanh", diagnostic.message)


if __name__ == "__main__":
    unittest.main()
