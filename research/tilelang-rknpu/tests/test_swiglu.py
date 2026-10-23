"""SwiGLU FFN dataflow: pattern matching, plan shape and C validation.

The graph is ``Y = (SiLU(X@Wg^T) * (X@Wu^T)) @ Wd^T + X`` with weights already in
the custom-ops ``[N,K]`` convention.  Every stage is its own submit; nothing is
claimed to be fused.
"""
import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT, RK_E_SCHEMA,
                           RK_E_SHAPE, RK_E_UNSUPPORTED, compile_rkir,
                           default_profile, swiglu_module)
from rknpu_backend.errors import BackendError

PROFILE = default_profile()


def options(**kwargs):
    # These tests select the experimental whole-graph SwiGLU specialisation.
    # The generic per-node path is covered separately without this opt-in.
    kwargs.setdefault("allow_experimental", True)
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True,
                          **kwargs)


class SwiGLUTests(unittest.TestCase):
    def test_plan_contains_every_stage(self):
        result = compile_rkir(swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                              options=options())
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates,
                         ["co.mm.f16.v1", "co.mm.f16.v1",
                          "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1",
                          "co.ew_mul.f16.v1", "co.mm.f16.v1",
                          "co.ew_add.f16.v1"])
        # one submit per stage: gate, up, silu, mul, projection, residual
        self.assertEqual(len(result.plan.submits), 6)
        self.assertEqual({s.recipe_id for s in result.plan.submits},
                         {"swiglu_ffn_v1"})

    def test_bindings_and_output_shape(self):
        result = compile_rkir(swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                              options=options())
        bindings = {b.name: b for b in result.plan.bindings}
        self.assertEqual(bindings["X"].shape, (8, 64))
        self.assertEqual(bindings["Wg"].shape, (32, 64))
        self.assertEqual(bindings["Wd"].shape, (64, 32))
        self.assertEqual(bindings["Y"].shape, (8, 64))
        self.assertEqual(len(result.plan.publish), 1)

    def test_intermediate_tensors_stay_on_device(self):
        result = compile_rkir(swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                              options=options())
        uploads = [a for a in result.plan.actions if a.kind == "upload"]
        downloads = [a for a in result.plan.actions if a.kind == "download"]
        # 4 bindings + 2 identity weights; no intermediate round trip
        self.assertEqual(len(uploads), 6)
        self.assertEqual(len(downloads), 1)
        self.assertEqual(result.plan.accounting.transfers, 7)

    def test_canonical_graph_uses_per_node_path_by_default(self):
        result = compile_rkir(
            swiglu_module(m=8, d=64, h=32), profile=PROFILE,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True))
        recipes = {submit.recipe_id for submit in result.plan.submits}
        self.assertNotIn("swiglu_ffn_v1", recipes)
        self.assertEqual(recipes, {"linear_k_acc_v1", "gate_ew_v1"})
        self.assertEqual(len(result.plan.submits), 6)
        self.assertEqual([a.template_id for a in result.plan.actions
                          if a.kind == "npu_task"],
                         ["co.mm.f16.v1", "co.mm.f16.v1",
                          "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1",
                          "co.ew_mul.f16.v1", "co.mm.f16.v1",
                          "co.ew_add.f16.v1"])

    def test_experimental_opt_in_still_selects_the_specialisation(self):
        result = compile_rkir(swiglu_module(m=8, d=64, h=32),
                              profile=PROFILE, options=options())
        self.assertEqual({s.recipe_id for s in result.plan.submits},
                         {"swiglu_ffn_v1"})

    def test_strict_numeric_mode_is_rejected_on_both_paths(self):
        for allow_experimental in (False, True):
            with self.assertRaises(BackendError) as ctx:
                compile_rkir(
                    swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                    options=CompileOptions(numeric_mode="strict",
                                           allow_approx_lut=True,
                                           allow_experimental=allow_experimental))
            self.assertEqual(ctx.exception.diagnostics[0].code,
                             RK_E_NUMERIC_CONTRACT)

    def test_lut_approval_is_required_on_both_paths(self):
        for allow_experimental in (False, True):
            with self.assertRaises(BackendError) as ctx:
                compile_rkir(
                    swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                    options=CompileOptions(numeric_mode="rk_f16_tiled",
                                           allow_approx_lut=False,
                                           allow_experimental=allow_experimental))
            self.assertEqual(ctx.exception.diagnostics[0].code,
                             RK_E_NUMERIC_CONTRACT)

    def test_hidden_width_above_one_tile_uses_generic_tiled_path(self):
        for allow_experimental in (False, True):
            result = compile_rkir(
                swiglu_module(m=8, d=64, h=300), profile=PROFILE,
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True,
                                       allow_experimental=allow_experimental))
            self.assertNotIn("swiglu_ffn_v1",
                             {s.recipe_id for s in result.plan.submits})
            self.assertGreaterEqual(
                sum(1 for a in result.plan.actions
                    if a.template_id == "co.mm.f16.v1"), 5)

    def test_broken_branch_connection_is_rejected(self):
        """Both matmuls must read X; routing one from another value is refused."""
        module = swiglu_module(m=8, d=64, h=32)
        ops = list(module.ops)
        ops[1] = replace(ops[1], inputs=(4, 2))  # up branch reads Zg instead of X
        # The IR itself already refuses: routing the up branch from Zg gives the
        # matmul an operand shape that does not match Wu.
        with self.assertRaises(BackendError) as ctx:
            replace(module, ops=tuple(ops))
        self.assertIn(ctx.exception.diagnostics[0].code,
                      (RK_E_UNSUPPORTED, RK_E_SCHEMA, RK_E_SHAPE))

    def test_specialised_capability_remains_experimental(self):
        self.assertEqual(
            PROFILE.experimental("co.swiglu_ffn.f16.v1").status,
            "experimental")
        with self.assertRaises(BackendError) as ctx:
            PROFILE.enabled("co.swiglu_ffn.f16.v1")
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)

    def test_c_verifier_accepts_the_swiglu_plan(self):
        try:
            from rknpu_backend.runtime.ffi import default_library
            lib = default_library()
        except Exception as exc:
            self.skipTest(f"rknpu_plan shared library not built: {exc}")
        plan = compile_rkir(swiglu_module(m=8, d=64, h=32), profile=PROFILE,
                            options=options()).plan
        cplan = lib.create_plan(plan)
        cplan.destroy()


if __name__ == "__main__":
    unittest.main()
