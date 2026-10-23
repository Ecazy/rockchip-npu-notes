"""Data-flow correctness of the restricted lowering.

The MVP lowers one *connected* chain: ``matmul -> (activation?) -> (binary?)``.
These tests pin that the lowering follows the real graph edges (Value IDs) instead
of assuming operand positions, and that a graph which does not match the chain is
rejected rather than silently rewritten.
"""
import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, Module, Op, RK_E_NUMERIC_CONTRACT,
                           RK_E_SCHEMA, RK_E_SHAPE, RK_E_UNSUPPORTED,
                           TensorType, Value, compile_rkir, default_profile,
                           matmul_module)
from rknpu_backend.errors import BackendError

PROFILE = default_profile()


def options(**kwargs):
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True,
                          **kwargs)


def with_last_op_inputs(module, inputs):
    ops = list(module.ops)
    ops[-1] = replace(ops[-1], inputs=tuple(inputs))
    return replace(module, ops=tuple(ops))


def chain_modules():
    """Value IDs of the reference program ``Z=X@B; S=SiLU(Z); Y=S*R``.

    values: 0=A (input), 1=B (constant), 2=Z (matmul), 3=S (activation),
            4=R (input), 5=Y (mul)
    """
    silu = matmul_module(m=13, k=64, n=33, activation="silu", residual="mul")
    plain = matmul_module(m=13, k=64, n=33, activation=None, residual="mul")
    return silu, plain


def projection_module(m=13, d=64, h=32):
    """SSA graph ``Z=X@W1; S=SiLU(Z); P=S@W2; Y=P+X``.

    W1/W2 are in the frontend's natural ``[K,N]`` orientation, so b_layout is
    ``kn``.  The two matmul contracts are copied from a real single-GEMM module so
    this helper exercises the same numeric policy as the frontend.
    """
    contract = matmul_module(m=m, k=d, n=h).ops[0].attrs["numeric_contract"]
    attrs = {"b_layout": "kn", "numeric_contract": contract}
    values = (
        Value(0, "X", TensorType("float16", (m, d)), None, "input"),
        Value(1, "W1", TensorType("float16", (d, h)), None, "constant"),
        Value(2, "Z", TensorType("float16", (m, h)), 0, "temporary"),
        Value(3, "S", TensorType("float16", (m, h)), 1, "temporary"),
        Value(4, "W2", TensorType("float16", (h, d)), None, "constant"),
        Value(5, "P", TensorType("float16", (m, d)), 2, "temporary"),
        Value(6, "Y", TensorType("float16", (m, d)), 3, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 1), (2,), dict(attrs)),
        Op(1, "silu", (2,), (3,)),
        Op(2, "matmul", (3, 4), (5,), dict(attrs)),
        Op(3, "add", (5, 0), (6,)),
    )
    return Module(name="projection-test", values=values, ops=ops,
                  inputs=(0,), outputs=(6,), source_hash="test")


def branch_outputs_module(*, m=13, k=64, n1=32, n2=48):
    """Two independent matmuls with two exported results.

    Both outputs share the X upload.  Their views must stay live until their
    respective downloads, and the planner must publish each output exactly once.
    """
    contract = matmul_module(m=m, k=k, n=n1, b_layout="nk").ops[0].attrs[
        "numeric_contract"]
    attrs = {"b_layout": "nk", "numeric_contract": contract}
    values = (
        Value(0, "X", TensorType("float16", (m, k)), None, "input"),
        Value(1, "W1", TensorType("float16", (n1, k)), None, "constant"),
        Value(2, "W2", TensorType("float16", (n2, k)), None, "constant"),
        Value(3, "Y1", TensorType("float16", (m, n1)), 0, "output"),
        Value(4, "Y2", TensorType("float16", (m, n2)), 1, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 1), (3,), dict(attrs)),
        Op(1, "matmul", (0, 2), (4,), dict(attrs)),
    )
    return Module(name="branch-output-test", values=values, ops=ops,
                  inputs=(0,), outputs=(3, 4), source_hash="test")


class MultiNodeDataflowTests(unittest.TestCase):
    def test_projection_graph_is_accepted_and_lowered_once(self):
        result = compile_rkir(projection_module(), profile=PROFILE,
                              options=options())
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates,
                         ["co.mm.f16.v1", "co.lut.sigmoid.f16.v1",
                          "co.ew_mul.f16.v1", "co.mm.f16.v1",
                          "co.ew_add.f16.v1"])

    def test_dead_second_branch_is_not_handed_to_the_planner(self):
        module = projection_module()
        ops = list(module.ops)
        # Same-shaped Z replaces S, leaving the SiLU node disconnected.
        ops[2] = replace(ops[2], inputs=(2, 4))
        broken = replace(module, ops=tuple(ops))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("does not contribute",
                      ctx.exception.diagnostics[0].message)

    def test_each_matmul_needs_the_f16_tiled_contract(self):
        module = projection_module()
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=PROFILE,
                         options=CompileOptions(numeric_mode="strict",
                                                allow_approx_lut=True))
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)

    def test_consumed_external_value_must_have_a_binding_role(self):
        module = projection_module()
        values = list(module.values)
        values[1] = replace(values[1], role="temporary")
        broken = replace(module, values=tuple(values))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("device producer",
                      ctx.exception.diagnostics[0].message)

    def test_output_role_outside_module_outputs_is_rejected(self):
        module = projection_module()
        values = list(module.values)
        values[3] = replace(values[3], role="output")  # S is not exported
        broken = replace(module, values=tuple(values))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("module.outputs", ctx.exception.diagnostics[0].message)

    def test_two_output_branch_plans_two_downloads_and_publishes(self):
        result = compile_rkir(branch_outputs_module(), profile=PROFILE,
                              options=options())
        plan = result.plan
        self.assertEqual([a.template_id for a in plan.actions
                          if a.kind == "npu_task"],
                         ["co.mm.f16.v1", "co.mm.f16.v1"])
        self.assertEqual(sum(a.kind == "download" for a in plan.actions), 2)
        self.assertEqual(len(plan.publish), 2)
        self.assertEqual([p.rows for p in plan.publish], [13, 13])
        self.assertEqual(sorted(p.cols for p in plan.publish), [32, 48])
        output_bindings = [b for b in plan.bindings if b.role == "output"]
        self.assertEqual([b.name for b in output_bindings], ["Y1", "Y2"])


class DataflowTests(unittest.TestCase):
    def test_connected_chain_is_accepted(self):
        silu, _ = chain_modules()
        result = compile_rkir(silu, profile=PROFILE, options=options())
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        # K=64 fits in one primitive tile, so there is no mm_acc step.
        self.assertEqual(templates, ["co.mm.f16.v1", "co.lut.sigmoid.f16.v1",
                                     "co.ew_mul.f16.v1", "co.ew_mul.f16.v1"])

    def test_commutative_operand_order_is_equivalent(self):
        silu, _ = chain_modules()
        forward = compile_rkir(silu, profile=PROFILE, options=options())
        reversed_inputs = with_last_op_inputs(silu, (4, 3))
        backward = compile_rkir(reversed_inputs, profile=PROFILE,
                                options=options())
        # Same graph up to commutative operand order => same plan.
        self.assertEqual(forward.plan.plan_hash, backward.plan.plan_hash)

    def test_binary_must_consume_the_produced_value(self):
        silu, _ = chain_modules()
        # Y = Z * R with a SiLU in between: the graph asks for Z*R, not SiLU(Z)*R.
        broken = with_last_op_inputs(silu, (2, 4))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        diagnostic = ctx.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
        self.assertIn("does not consume", diagnostic.message)

    def test_broken_chain_is_not_silently_reinterpreted(self):
        """The previous bug produced the *same* plan hash for both graphs."""
        silu, _ = chain_modules()
        good = compile_rkir(silu, profile=PROFILE, options=options()).plan
        broken = with_last_op_inputs(silu, (2, 4))
        try:
            bad = compile_rkir(broken, profile=PROFILE, options=options()).plan
        except BackendError:
            return  # rejected: the silent rewrite cannot happen
        self.assertNotEqual(good.plan_hash, bad.plan_hash,
                            "a different graph must not share a plan hash")

    def test_activation_must_consume_the_matmul_result(self):
        silu, _ = chain_modules()
        ops = list(silu.ops)
        # Feed SiLU from R (value 4, same shape) instead of Z (value 2).
        ops[1] = replace(ops[1], inputs=(4,))
        broken = replace(silu, ops=tuple(ops))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)

    def test_output_root_must_be_the_produced_value(self):
        silu, _ = chain_modules()
        # Export the matmul result (2) while the chain ends at Y (5).
        values = list(silu.values)
        values[2] = replace(values[2], role="output")
        values[5] = replace(values[5], role="temporary")
        broken = replace(silu, values=tuple(values), outputs=(2,))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(broken, profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)

    def test_distinct_valid_graphs_have_distinct_plan_hashes(self):
        silu, plain = chain_modules()
        hashes = {
            "silu_mul": compile_rkir(silu, profile=PROFILE,
                                     options=options()).plan.plan_hash,
            "plain_mul": compile_rkir(plain, profile=PROFILE,
                                      options=options()).plan.plan_hash,
        }
        add = matmul_module(m=13, k=64, n=33, activation="silu", residual="add")
        hashes["silu_add"] = compile_rkir(add, profile=PROFILE,
                                          options=options()).plan.plan_hash
        self.assertEqual(len(set(hashes.values())), len(hashes), hashes)

    def test_operand_may_not_be_the_chain_result(self):
        silu, _ = chain_modules()
        # Y = Y * R would make the mul read its own result; the IR already
        # rejects a read whose producer is not earlier in the graph.
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(with_last_op_inputs(silu, (5, 4)), profile=PROFILE,
                         options=options())
        self.assertIn(ctx.exception.code, (RK_E_SHAPE, RK_E_UNSUPPORTED))


if __name__ == "__main__":
    unittest.main()
