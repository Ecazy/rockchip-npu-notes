import json
import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, RK_E_ALIAS, RK_E_SCHEMA,
                           RK_E_SHAPE, compile_rkir, default_profile,
                           matmul_module)
from rknpu_backend.errors import BackendError
from rknpu_backend import RK_E_SCHEMA
from rknpu_backend.plan import Plan, Submit, with_plan_hash


def make_plan():
    return compile_rkir(
        matmul_module(m=13, k=257, n=33, activation="silu"),
        profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True)).plan


class PlanVerifyTests(unittest.TestCase):
    def test_roundtrip_and_hash_tamper(self):
        plan = make_plan()
        payload = plan.to_dict()
        clone = Plan.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(clone.plan_hash, plan.plan_hash)
        payload["actions"][0]["origin"] = "tampered"
        with self.assertRaises(BackendError) as ctx:
            Plan.from_dict(payload)
        self.assertEqual(ctx.exception.code, RK_E_SCHEMA)

    def test_mm_acc_alias_is_rejected(self):
        plan = make_plan()
        actions = list(plan.actions)
        idx = next(i for i, a in enumerate(actions)
                   if a.template_id == "co.mm_acc.f16.v1")
        action = actions[idx]
        actions[idx] = replace(action, outputs=(action.inputs[2],))
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash,
                 plan.bindings, plan.constants, plan.buffers,
                 plan.views, tuple(actions), plan.submits,
                 plan.publish, plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_ALIAS)

    def test_unknown_padding_is_rejected(self):
        plan = make_plan()
        buffers = list(plan.buffers)
        with self.assertRaises(BackendError) as ctx:
            buffers[0] = replace(buffers[0], padding_state="magic")
        self.assertEqual(ctx.exception.code, RK_E_SCHEMA)

    def test_plan_schema_accepts_real_plan(self):
        import jsonschema
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schema" / "rkplan.v2.json").read_text())
        jsonschema.validate(make_plan().to_dict(), schema)

    def test_overlapping_publish_tiles_are_rejected(self):
        plan = make_plan()
        publish = list(plan.publish)
        publish[0] = replace(publish[0], cols=publish[0].cols + 1) \
            if publish[0].cols < 33 else replace(publish[0], rows=publish[0].rows + 1)
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash,
                 plan.bindings, plan.constants, plan.buffers, plan.views,
                 plan.actions, plan.submits, tuple(publish), plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_incomplete_publish_coverage_is_rejected(self):
        plan = make_plan()
        publish = list(plan.publish)
        publish[0] = replace(publish[0], rows=publish[0].rows - 1)
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash,
                 plan.bindings, plan.constants, plan.buffers, plan.views,
                 plan.actions, plan.submits, tuple(publish), plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_noncanonical_physical_geometry_is_rejected(self):
        from rknpu_backend.plan import TensorLayout
        with self.assertRaises(BackendError) as ctx:
            TensorLayout("float16", "co.cube.f16.v1", 13, 33, 12, 64, 8,
                         12 * 64 * 2)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)
        with self.assertRaises(BackendError) as ctx:
            TensorLayout("float16", "co.weight.f16.v1", 33, 64, 16, 64, 32,
                         16 * 64 * 2)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_recipe_must_match_the_task_sequence(self):
        plan = make_plan()
        submits = list(plan.submits)
        submits[0] = replace(submits[0], recipe_id="gate_ew_v1")
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash, plan.bindings,
                 plan.constants, plan.buffers, plan.views, plan.actions,
                 tuple(submits), plan.publish, plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_SCHEMA)
        self.assertIn("recipe", ctx.exception.diagnostics[0].message)

    def test_unknown_recipe_is_rejected(self):
        plan = make_plan()
        submits = list(plan.submits)
        submits[0] = replace(submits[0], recipe_id="made_up_v9")
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash, plan.bindings,
                 plan.constants, plan.buffers, plan.views, plan.actions,
                 tuple(submits), plan.publish, plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_SCHEMA)

    def test_65_task_submit_is_rejected(self):
        plan = make_plan()
        submit = Submit(0, tuple([0] * 65), "linear_k_acc_v1", "chain",
                        "single_core_per_submit", ())
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash,
                 plan.bindings, plan.constants, plan.buffers, plan.views,
                 plan.actions, (submit,), plan.publish, plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)

    def test_deleting_producer_rejected(self):
        module = matmul_module(m=2, k=3, n=4)
        obj = module.to_dict()
        obj["ops"] = []
        obj["values"][2]["producer"] = None
        with self.assertRaises(BackendError):
            from rknpu_backend.ir import Module
            Module.from_dict(obj)


if __name__ == "__main__":
    unittest.main()
