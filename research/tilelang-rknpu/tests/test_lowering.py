import unittest

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_SHAPE, RK_E_UNSUPPORTED, compile_rkir,
                           default_profile, matmul_module)
from rknpu_backend.errors import BackendError


class LoweringTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)

    def _compile(self, m=13, k=257, n=33, activation=None, residual=None,
                 **kwargs):
        return compile_rkir(
            matmul_module(m=m, k=k, n=n, activation=activation,
                          residual=residual),
                            profile=self.profile,
                            options=CompileOptions(
                                numeric_mode=kwargs.pop("numeric_mode",
                                                        "rk_f16_tiled"),
                                allow_approx_lut=kwargs.pop("allow_approx_lut",
                                                            True),
                                **kwargs))

    def test_k_chain_task_counts(self):
        expected = {256: 1, 257: 2, 512: 2, 513: 3, 4096: 16}
        for k, tasks in expected.items():
            result = self._compile(k=k, activation=None)
            mm_tasks = [a for a in result.plan.actions
                        if a.template_id in {"co.mm.f16.v1", "co.mm_acc.f16.v1"}]
            self.assertEqual(len(mm_tasks), tasks, k)
            self.assertEqual(mm_tasks[0].template_id, "co.mm.f16.v1")
            for action in mm_tasks[1:]:
                self.assertEqual(action.template_id, "co.mm_acc.f16.v1")

    def test_strict_accumulator_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(numeric_mode="strict")
        self.assertEqual(ctx.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_activation_requires_explicit_approx(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(activation="silu", allow_approx_lut=False)
        self.assertEqual(ctx.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_silu_four_task_two_submit(self):
        result = self._compile(activation="silu")
        self.assertEqual(result.plan.accounting.task_count, 4)
        self.assertEqual(len(result.plan.submits), 2)
        self.assertEqual(set(s.mode for s in result.plan.submits), {"chain"})

    def test_256x256_silu_lut_has_two_chunks(self):
        result = self._compile(m=256, k=256, n=256, activation="silu")
        luts = [a for a in result.plan.actions
                if a.template_id == "co.lut.sigmoid.f16.v1"]
        self.assertEqual(len(luts), 2)
        self.assertEqual([a.attrs["chunk_index"] for a in luts], [0, 1])
        # Activation itself is two LUTs plus one EW multiply; the MatMul is extra.
        self.assertEqual(result.plan.accounting.task_count, 4)

    def test_k_over_4096_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            self._compile(k=4097)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_same_shape_add_after_matmul(self):
        result = self._compile(activation=None, residual="add")
        tasks = [a for a in result.plan.actions if a.kind == "npu_task"]
        self.assertEqual([a.template_id for a in tasks],
                         ["co.mm.f16.v1", "co.mm_acc.f16.v1",
                          "co.ew_add.f16.v1"])
        self.assertEqual(len(result.plan.submits), 2)
        # The residual is a second input binding, packed per output tile.
        names = [b.name for b in result.plan.bindings if b.role == "input"]
        self.assertEqual(names, ["A", "R"])
        uploads = [a for a in result.plan.actions
                   if a.kind == "upload" and
                   result.plan.bindings[a.attrs["source_binding"]].name == "R"]
        self.assertEqual(len(uploads), 1)

    def test_gated_multiply_with_runtime_operand(self):
        result = self._compile(activation="silu", residual="mul")
        templates = [a.template_id for a in result.plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates, ["co.mm.f16.v1", "co.mm_acc.f16.v1",
                                     "co.lut.sigmoid.f16.v1",
                                     "co.ew_mul.f16.v1", "co.ew_mul.f16.v1"])
        self.assertEqual(len(result.plan.submits), 3)

    def test_binary_operand_must_be_same_shape_input(self):
        module = matmul_module(m=13, k=257, n=33, residual="add")
        obj = module.to_dict()
        # Shrink the residual binding: broadcast shapes are not in this path.
        for value in obj["values"]:
            if value["name"] == "R":
                value["type"]["shape"] = [1, 33]
        from rknpu_backend.ir import Module
        # The IR itself rejects a broadcast-shaped binary operand, before the
        # planner is reached.
        with self.assertRaises(BackendError) as ctx:
            Module.from_dict(obj)
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_separate_mode_changes_submit_count_and_hash(self):
        chain = self._compile(activation="silu", submit_mode="chain")
        separate = self._compile(activation="silu", submit_mode="separate")
        self.assertEqual(chain.plan.accounting.task_count,
                         separate.plan.accounting.task_count)
        self.assertEqual(len(chain.plan.submits), 2)
        self.assertEqual(len(separate.plan.submits), 4)
        self.assertNotEqual(chain.plan.plan_hash, separate.plan.plan_hash)


if __name__ == "__main__":
    unittest.main()
