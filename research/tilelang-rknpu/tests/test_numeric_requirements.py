"""NumericContract must survive into the plan, tied to the tasks it produced."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, Module, NumericContract, Op,
                           RK_E_NUMERIC_CONTRACT, TensorType, Value,
                           compile_rkir, default_profile, explain_plan,
                           load_plan, matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    settings.update(extra)
    return CompileOptions(**settings)


def schedule(tile_m=128, tile_n=256):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": "m_major", "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def with_schedule(module, tile_m=128, tile_n=256):
    attrs = dict(module.ops[0].attrs)
    attrs["source_schedule"] = schedule(tile_m, tile_n)
    attrs["source_k_rounding"] = "explicit_per_k_fp16"
    return replace(module, ops=(replace(module.ops[0], attrs=attrs),))


class RequirementRecordingTests(unittest.TestCase):
    def test_single_matmul_multi_k_records_contract_and_tasks(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        self.assertEqual(len(plan.numeric_requirements), 1)
        record = plan.numeric_requirements[0]
        self.assertEqual(record.op_index, 0)
        self.assertEqual(record.opcode, "matmul")
        self.assertEqual(record.inputs, ("A", "B"))
        self.assertEqual(record.outputs, ("Y",))
        npu = {a.id for a in plan.actions if a.kind == "npu_task"}
        self.assertEqual(set(record.tasks), npu)
        self.assertEqual(record.execution["k_chain"], [0, 256])
        self.assertEqual(record.execution["k_tile"], 256)
        self.assertEqual(record.execution["output_dtypes"], ["float16"])
        self.assertTrue(record.execution["attributed"])
        self.assertEqual(record.contract["tolerance"],
                         {"atol": 8e-3, "rtol": 8e-3})

    def test_two_stage_keeps_two_separate_contracts(self):
        values = (Value(0, "X", TensorType("float16", (256, 256)), None, "input"),
                  Value(1, "W1", TensorType("float16", (256, 256)), None, "constant"),
                  Value(2, "Z", TensorType("float16", (256, 256)), 0, "temporary"),
                  Value(3, "W2", TensorType("float16", (256, 256)), None, "constant"),
                  Value(4, "Y", TensorType("float16", (256, 256)), 1, "output"))
        base = dict(matmul_module(m=256, k=256, n=256).ops[0].attrs)
        first = dict(base, source_schedule=schedule(),
                     source_k_rounding="explicit_per_k_fp16")
        second = dict(first)
        second["numeric_contract"] = dict(base["numeric_contract"],
                                          tolerance={"atol": 4e-3,
                                                     "rtol": 4e-3})
        module = Module(name="chain", values=values,
                        ops=(Op(0, "matmul", (0, 1), (2,), first),
                             Op(1, "matmul", (2, 3), (4,), second)),
                        inputs=(0,), outputs=(4,), source_hash="chain",
                        profile_hash=PROFILE.profile_hash)
        plan = compile_rkir(module, profile=PROFILE, options=options()).plan
        self.assertEqual([r.op_index for r in plan.numeric_requirements], [0, 1])
        self.assertEqual(plan.numeric_requirements[0].inputs, ("X", "W1"))
        self.assertEqual(plan.numeric_requirements[1].inputs, ("Z", "W2"))
        self.assertEqual(plan.numeric_requirements[0].contract["tolerance"],
                         {"atol": 8e-3, "rtol": 8e-3})
        self.assertEqual(plan.numeric_requirements[1].contract["tolerance"],
                         {"atol": 4e-3, "rtol": 4e-3})
        left = set(plan.numeric_requirements[0].tasks)
        right = set(plan.numeric_requirements[1].tasks)
        self.assertTrue(left and right and not (left & right))

    def test_save_load_preserves_requirements_and_hash(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.rkplan.json"
            save_plan(plan, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.numeric_requirements,
                         plan.numeric_requirements)

    def test_explain_reports_source_and_absent_end_to_end_criterion(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        numeric = explain_plan(plan)["numeric"]
        self.assertTrue(numeric["stored"])
        self.assertEqual(numeric["source"], "plan")
        self.assertEqual(numeric["operators"][0]["tolerance_source"],
                         "plan:op[0]")
        self.assertIn("undefined", numeric["end_to_end_criterion"])


class OldPlanCompatibilityTests(unittest.TestCase):
    def test_frozen_plan_without_requirements_reports_not_stored(self):
        plan = load_plan(str(ROOT / "models" / "compiled" /
                             "tiny_vit_b1_attn_out.rkplan.json"))
        self.assertEqual(plan.numeric_requirements, ())
        numeric = explain_plan(plan)["numeric"]
        self.assertFalse(numeric["stored"])
        self.assertEqual(numeric["source"], "not stored")

    def test_resaving_an_old_plan_keeps_its_identity_and_omits_the_field(self):
        plan = load_plan(str(ROOT / "models" / "compiled" /
                             "tiny_vit_b1_attn_out.rkplan.json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.rkplan.json"
            save_plan(plan, str(path))
            payload = json.loads(path.read_text())
        self.assertNotIn("numeric_requirements", payload)
        self.assertEqual(payload["plan_hash"], plan.plan_hash)


class TamperTests(unittest.TestCase):
    def _payload(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        return plan, json.loads(json.dumps(plan.to_dict()))

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.rkplan.json"
            path.write_text(json.dumps(payload))
            return load_plan(str(path))

    def test_changing_the_contract_is_detected(self):
        plan, payload = self._payload()
        payload["numeric_requirements"][0]["contract"]["tolerance"] = {
            "atol": 1.0, "rtol": 1.0}
        with self.assertRaises(BackendError):
            self._load(payload)

    def test_pointing_at_a_non_npu_action_is_detected(self):
        plan, payload = self._payload()
        upload = next(a.id for a in plan.actions if a.kind == "upload")
        payload["numeric_requirements"][0]["tasks"] = [upload]
        with self.assertRaises(BackendError):
            self._load(payload)

    def test_op_index_mismatch_is_detected(self):
        plan, payload = self._payload()
        payload["numeric_requirements"][0]["op_index"] = 7
        with self.assertRaises(BackendError):
            self._load(payload)


@unittest.skipUnless(tilelang_available(), "TileLang absent")
class TileLangEntryTests(unittest.TestCase):
    def test_real_tilelang_source_records_its_contract(self):
        from tests.test_schedule_provenance import TileLangProvenanceTests
        prim = TileLangProvenanceTests()._kernel(acc16=True)
        from rknpu_backend import compile_tilelang
        plan = compile_tilelang(prim, profile=PROFILE,
                                options=options()).plan
        self.assertEqual(len(plan.numeric_requirements), 1)
        record = plan.numeric_requirements[0]
        self.assertEqual(record.contract["accumulation"],
                         "ordered_k_tiles_with_f16_writeback")
        self.assertEqual(record.execution["k_chain"], [0])
        self.assertTrue(record.execution["attributed"])


@unittest.skipUnless(tilelang_available(), "TileLang absent")
class LutGraphTests(unittest.TestCase):
    def test_lut_graph_keeps_the_matmul_contract_only(self):
        from rknpu_backend import compile_tilelang
        from tests.fixtures import tilelang_ffn
        plan = compile_tilelang(tilelang_ffn.swiglu_ffn_f16(64, 256, 128),
                                profile=PROFILE,
                                options=options(allow_approx_lut=True)).plan
        records = plan.numeric_requirements
        self.assertTrue(records)
        templates = {a.id: (a.template_id or "")
                     for a in plan.actions if a.kind == "npu_task"}
        for record in records:
            # Attribution must be explicit either way: a record never claims
            # tasks it does not have.
            self.assertEqual(record.execution["attributed"],
                             bool(record.tasks))
            for task in record.tasks:
                if templates[task].startswith("co.lut."):
                    # A lookup table is an approximation.  The contract keeps
                    # stating what the source asked for, so the admission is
                    # recorded in execution -- and the plan rejects the op if
                    # it is neither recorded nor approximate.
                    self.assertTrue(
                        record.contract["approximate"] or
                        record.execution.get("approximated"),
                        "a LUT task must be recorded as an admitted "
                        "approximation")
        self.assertTrue(any(r.opcode == "matmul" for r in records))


class ToleranceFinitenessTests(unittest.TestCase):
    """NaN and infinities are contract errors, not JSON quirks."""

    BASE = dict(accumulator_dtype="float16",
                accumulation="ordered_k_tiles_with_f16_writeback",
                narrowing_points=("k_tile_boundary",), approximate=False)

    def test_finite_values_are_accepted(self):
        contract = NumericContract(tolerance={"atol": 8e-3, "rtol": 0.0},
                                   **self.BASE)
        self.assertEqual(contract.tolerance, {"atol": 8e-3, "rtol": 0.0})

    def test_non_finite_and_invalid_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), -1.0,
                      True, "0.1"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(BackendError) as caught:
                    NumericContract(tolerance={"atol": value}, **self.BASE)
                self.assertEqual(caught.exception.code,
                                 RK_E_NUMERIC_CONTRACT)

    def test_from_dict_rejects_non_finite(self):
        payload = {
            "input_dtype": "float16", "output_dtype": "float16",
            "accumulator_dtype": "float16",
            "accumulation": "ordered_k_tiles_with_f16_writeback",
            "narrowing_points": ["k_tile_boundary"], "approximate": False,
            "tolerance": {"atol": float("inf")},
        }
        with self.assertRaises(BackendError) as caught:
            NumericContract.from_dict(payload)
        self.assertEqual(caught.exception.code, RK_E_NUMERIC_CONTRACT)


if __name__ == "__main__":
    unittest.main()