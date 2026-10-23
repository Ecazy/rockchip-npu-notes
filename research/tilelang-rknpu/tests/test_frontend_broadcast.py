"""Compile-time scalar / per-channel constant broadcast through the frontend."""
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, Module, RK_E_SCHEMA, RK_E_SHAPE,
                           TensorType, Value, compile_rkir, compile_tilelang,
                           default_profile, load_plan, save_plan)
from rknpu_backend.errors import BackendError
import jsonschema

from rknpu_backend.frontend.tilelang_adapter import extract, tilelang_available

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_broadcast_frontend", str(FIXTURES / "tilelang_broadcast.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class BroadcastFrontendTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixture = load_fixture()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled")

    def test_scalar_add_extracts_rank0_constant(self):
        module = extract(self.fixture.scalar_add(m=8, k=64, n=33),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops], ["matmul", "add"])
        self.assertEqual([(v.name, v.type.shape, v.role)
                          for v in module.values],
                         [("X", (8, 64), "input"),
                          ("W", (64, 33), "constant"),
                          ("Z", (8, 33), "temporary"),
                          ("B", (), "constant"),
                          ("Y", (8, 33), "output")])
        self.assertEqual(module.ops[1].inputs, (2, 3))
        schema = json.loads(
            (ROOT / "schema" / "rkir.v1.json").read_text())
        jsonschema.validate(module.to_dict(), schema)

    def test_channel_mul_extracts_rank1_constant_and_unaligned_tail(self):
        module = extract(self.fixture.channel_mul(m=8, k=64, n=33),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops], ["matmul", "mul"])
        self.assertEqual(module.value(3).type.shape, (33,))
        self.assertEqual(module.ops[1].inputs, (2, 3))

    def test_plan_expands_constants_once_and_accounts_for_the_tile(self):
        cases = ((self.fixture.scalar_add, "scalar", ()),
                 (self.fixture.channel_mul, "channel", (33,)))
        for factory, kind, source_shape in cases:
            with self.subTest(kind=kind):
                result = compile_tilelang(
                    factory(m=8, k=64, n=33), profile=self.profile,
                    options=self.options)
                plan = result.plan
                self.assertEqual(result.report["counts"][
                                     "broadcast_expand_actions"], 1)
                self.assertEqual(result.report["broadcast"]["kinds"], [kind])
                bindings = {b.name: b for b in plan.bindings}
                self.assertEqual(bindings["B"].shape, source_shape)
                uploads = [a for a in plan.actions if a.kind == "upload"]
                broadcast = [a for a in uploads
                             if a.attrs.get("broadcast") == kind]
                self.assertEqual(len(broadcast), 1)
                self.assertEqual(broadcast[0].attrs["broadcast_target"], [8, 33])
                # Expanded cube is 12 x 64 FP16 elements; source binding stays
                # rank-0/1 and the expansion is charged on the Host side.
                self.assertEqual(plan.accounting.broadcast_expand_bytes, 12 * 64 * 2)
                self.assertGreaterEqual(plan.accounting.host_bytes,
                                        plan.accounting.broadcast_expand_bytes)
                tasks = [a for a in plan.actions if a.kind == "npu_task"]
                self.assertEqual([a.template_id for a in tasks],
                                 ["co.mm.f16.v1",
                                  "co.ew_add.f16.v1" if kind == "scalar"
                                  else "co.ew_mul.f16.v1"])
                self.assertLess(max(a.id for a in uploads),
                                min(a.id for a in tasks))

    def test_broadcast_plan_json_roundtrip(self):
        plan = compile_tilelang(
            self.fixture.channel_mul(m=8, k=64, n=33),
            profile=self.profile, options=self.options).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broadcast.rkplan.json"
            save_plan(plan, path)
            again = load_plan(path)
            obj = json.loads(path.read_text())
        self.assertEqual(again.plan_hash, plan.plan_hash)
        self.assertEqual(next(b for b in again.bindings if b.name == "B").shape,
                         (33,))
        self.assertEqual(again.accounting.broadcast_expand_bytes,
                         12 * 64 * 2)
        # The serialized JSON is the versioned plan form, not a Python pickle.
        self.assertEqual(next(b for b in obj["bindings"]
                              if b["name"] == "B")["shape"], [33])
        schema = json.loads(
            (ROOT / "schema" / "rkplan.v2.json").read_text())
        jsonschema.validate(obj, schema)

    def test_wrong_channel_length_is_rejected(self):
        module = extract(self.fixture.channel_mul(m=8, k=64, n=33),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        values = list(module.values)
        values[3] = replace(
            values[3],
            type=TensorType("float16", (34,), axes=("col",)))
        with self.assertRaises(BackendError) as ctx:
            # The frozen IR rejects the inconsistent broadcast shape while the
            # replacement Module is being constructed.
            replace(module, values=tuple(values))
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE)
        self.assertIn("column count", ctx.exception.diagnostics[0].message)

    def test_runtime_scalar_input_cannot_pretend_to_be_a_constant(self):
        with self.assertRaises(BackendError) as ctx:
            Value(0, "bias", TensorType("float16", (), axes=()), None, "input")
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE)
        self.assertIn("only constants", ctx.exception.diagnostics[0].message)


if __name__ == "__main__":
    unittest.main()
