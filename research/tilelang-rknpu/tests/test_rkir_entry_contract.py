# SPDX-License-Identifier: GPL-3.0-or-later
"""RKIR entry contract: attributes, axes, producer/role and the signature.

Each negative is refused while the module is built or compiled, so nothing
reaches a plan, a binding list or the device.  The positive cases keep the
plans the planner really emits byte-identical.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, default_profile, load_plan,
                           matmul_module, save_plan, swiglu_module)
from rknpu_backend.errors import BackendError
from rknpu_backend.ir import MATMUL_ATTRS, MATMUL_SOURCE_META

PROFILE = default_profile()
CONTRACT = dict(matmul_module(m=16, k=16, n=8).ops[0].attrs["numeric_contract"])


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def matmul_values(axes_a=("row", "col"), axes_b=("row", "col"),
                  extra_value=None):
    values = [Value(0, "A", TensorType("float16", (16, 16), axes=axes_a),
                    None, "input"),
              Value(1, "B", TensorType("float16", (16, 8), axes=axes_b),
                    None, "constant"),
              Value(2, "Y", TensorType("float16", (16, 8)), 0, "output")]
    if extra_value is not None:
        values.append(extra_value)
    return tuple(values)


def matmul_op(extra_attrs=None, inputs=(0, 1)):
    attrs = {"b_layout": "kn", "numeric_contract": CONTRACT}
    attrs.update(extra_attrs or {})
    return Op(0, "matmul", inputs, (2,), attrs, "matmul")


def build(values=None, ops=None, inputs=(0,), outputs=(2,)):
    return Module(name="entry", values=values or matmul_values(),
                  ops=ops or (matmul_op(),), inputs=inputs, outputs=outputs,
                  source_hash="entry", profile_hash=PROFILE.profile_hash)


class RkirCase(unittest.TestCase):
    def assert_rejected(self, build_fn, needle):
        with self.assertRaises(BackendError) as caught:
            compile_rkir(build_fn(), profile=PROFILE, options=options())
        diagnostic = caught.exception.diagnostics[0]
        self.assertIn(needle, diagnostic.message)
        return diagnostic


class MatMulAttributeTests(RkirCase):
    def test_transposition_requests_are_refused(self):
        for key in ("transpose_a", "transpose_b", "a_layout", "transposed_a"):
            with self.subTest(attr=key):
                self.assert_rejected(
                    lambda key=key: build(ops=(matmul_op({key: True}),)),
                    "does not transpose an operand")

    def test_unknown_attributes_are_refused_not_ignored(self):
        for key, value in (("k_pad", 1024), ("fused_magic", 1),
                           ("b_layout_typo", "kn")):
            with self.subTest(attr=key):
                self.assert_rejected(
                    lambda key=key, value=value: build(
                        ops=(matmul_op({key: value}),)),
                    "is not read by the lowering")

    def test_every_supported_attribute_is_accepted(self):
        # The exact metadata the TileLang matmul adapter stamps today, plus the
        # stage attribution keys the multiplexed sources add.
        legal = {
            "source_accumulator_dtype": "float32",
            "source_narrowing_points": ["gemm_accumulator"],
            "source_k_rounding": "fp32_final_narrowing",
            "source_k_loop": {"kind": "k_tiles"},
            "source_mn_tiling": {"block_m": 16, "block_n": 8},
            "source_tail_contract": {"m_tail": 0, "n_tail": 0},
            # ``source_schedule`` carries its own value schema (checked by the
            # schedule tests); here only the key has to be recognized.
            "source_schedule_origin": "annotated",
            "source_stage_index": 0,
            "source_stage_count": 1,
            "source_stage_boundary": "begin",
            "source_stage_schedule_contract": {"kind": "stage_v1"},
            "tile_m": 16, "tile_n": 8, "tile_k": 16,
        }
        self.assertTrue(set(legal) <= MATMUL_ATTRS)
        self.assertIn("source_schedule", MATMUL_ATTRS)
        plan = compile_rkir(build(ops=(matmul_op(legal),)), profile=PROFILE,
                            options=options()).plan
        self.assertIn("co.mm.f16.v1",
                      [a.template_id for a in plan.actions])

    def test_b_layout_and_contract_still_work(self):
        for layout, b_shape in (("kn", (16, 8)), ("nk", (8, 16))):
            with self.subTest(b_layout=layout):
                values = (Value(0, "A", TensorType("float16", (16, 16)), None,
                                "input"),
                          Value(1, "B", TensorType("float16", b_shape), None,
                                "constant"),
                          Value(2, "Y", TensorType("float16", (16, 8)), 0,
                                "output"))
                ops = (Op(0, "matmul", (0, 1), (2,),
                          {"b_layout": layout, "numeric_contract": CONTRACT}),)
                plan = compile_rkir(build(values, ops), profile=PROFILE,
                                    options=options()).plan
                self.assertIn("co.mm.f16.v1",
                              [a.template_id for a in plan.actions])

    def test_missing_contract_is_still_refused(self):
        self.assert_rejected(
            lambda: build(ops=(Op(0, "matmul", (0, 1), (2,), {"b_layout": "kn"},
                                "matmul"),)),
            "NumericContract")


class AxisLabelTests(RkirCase):
    def _compile(self, axes_a, axes_b) -> None:
        compile_rkir(build(values=matmul_values(axes_a, axes_b)),
                     profile=PROFILE, options=options())

    def test_canonical_labels_are_accepted(self):
        self._compile(("row", "col"), ("row", "col"))

    def test_custom_labels_are_just_labels(self):
        self._compile(("token", "depth"), ("depth", "out"))

    def test_default_and_custom_labels_mix(self):
        # A custom label on one side is a rename, not an order claim, so mixing
        # it with a default name on the other side is legal.
        for axes_a, axes_b in ((("row", "feature"), ("token", "col")),
                               (("token", "col"), ("row", "feature")),
                               (("row", "feature"), ("row", "col")),
                               (("token", "col"), ("depth", "out"))):
            with self.subTest(axes_a=axes_a, axes_b=axes_b):
                self._compile(axes_a, axes_b)

    def test_a_default_name_in_the_wrong_position_is_refused(self):
        for axes_a, axes_b, needle in (
                (("col", "row"), ("row", "col"), "col label describes"),
                (("row", "col"), ("col", "row"), "col label describes"),
                (("col", "feature"), ("row", "col"), "col label describes"),
                (("row", "col"), ("feature", "row"), "row label describes")):
            with self.subTest(axes_a=axes_a, axes_b=axes_b):
                with self.assertRaises(BackendError) as caught:
                    self._compile(axes_a, axes_b)
                self.assertIn(needle,
                              caught.exception.diagnostics[0].message)

    def test_duplicate_labels_are_refused(self):
        for axes_a in (("row", "row"), ("feature", "feature")):
            with self.subTest(axes_a=axes_a):
                with self.assertRaises(BackendError) as caught:
                    self._compile(axes_a, ("row", "col"))
                self.assertIn("unique",
                              caught.exception.diagnostics[0].message)

    def test_label_count_and_emptiness_are_still_checked(self):
        for axes_a in (("row",), ("row", "col", "third"), ("row", ""), ()):
            with self.subTest(axes_a=axes_a):
                with self.assertRaises(BackendError) as caught:
                    self._compile(axes_a, ("row", "col"))
                self.assertIn("one non-empty string per tensor rank",
                              caught.exception.diagnostics[0].message)

    def test_rank0_needs_an_empty_axis_list(self):
        with self.assertRaises(BackendError):
            TensorType("float16", (), axes=("scalar",))
        TensorType("float16", (), axes=())

    def test_rank1_broadcast_axis_and_rank0(self):
        bias = Value(1, "bias", TensorType("float16", (8,), axes=("col",)),
                     None, "constant")
        values = (Value(0, "X", TensorType("float16", (16, 8)), None, "input"),
                  bias,
                  Value(2, "Y", TensorType("float16", (16, 8)), 0, "output"))
        ops = (Op(0, "add", (0, 1), (2,), {"numeric_contract": CONTRACT}),)
        compile_rkir(build(values, ops), profile=PROFILE, options=options())

        with self.assertRaises(BackendError) as caught:
            # The label is refused when the tensor type is constructed.
            wrong = Value(1, "bias",
                          TensorType("float16", (8,), axes=("row",)),
                          None, "constant")
            compile_rkir(build((values[0], wrong, values[2]), ops),
                         profile=PROFILE, options=options())
        self.assertIn("row label describes rank-2 position 0 only",
                      caught.exception.diagnostics[0].message)

        # A rank-1 broadcast source may also carry a custom label next to the
        # rank-2 operands, which are allowed to use the default names.
        custom = Value(1, "bias",
                       TensorType("float16", (8,), axes=("channel",)),
                       None, "constant")
        compile_rkir(build((values[0], custom, values[2]), ops),
                     profile=PROFILE, options=options())

        scalar = Value(1, "s", TensorType("float16", (), axes=()), None,
                       "constant")
        ops = (Op(0, "mul", (0, 1), (2,), {"numeric_contract": CONTRACT}),)
        compile_rkir(build((values[0], scalar, values[2]), ops),
                     profile=PROFILE, options=options())


class LabelInvarianceTests(unittest.TestCase):
    """Labels are names: the same math has to lower to the same plan."""

    def _plan(self, axes_a, axes_b):
        return compile_rkir(build(values=matmul_values(axes_a, axes_b)),
                            profile=PROFILE, options=options()).plan

    def test_label_choice_does_not_change_the_plan(self):
        baseline = self._plan(("row", "col"), ("row", "col"))
        for axes_a, axes_b in ((("token", "depth"), ("depth", "out")),
                               (("row", "feature"), ("token", "col"))):
            with self.subTest(axes_a=axes_a, axes_b=axes_b):
                other = self._plan(axes_a, axes_b)
                self.assertEqual(other.plan_hash, baseline.plan_hash)
                self.assertEqual(other.to_dict(), baseline.to_dict())
                self.assertEqual([b.to_dict() for b in other.bindings],
                                 [b.to_dict() for b in baseline.bindings])
                self.assertEqual([a.to_dict() for a in other.actions],
                                 [a.to_dict() for a in baseline.actions])
                self.assertEqual([s.to_dict() for s in other.submits],
                                 [s.to_dict() for s in baseline.submits])

    def test_rank1_broadcast_labels_do_not_change_the_plan(self):
        def plan(label):
            bias = Value(1, "bias", TensorType("float16", (8,), axes=(label,)),
                         None, "constant")
            values = (Value(0, "X", TensorType("float16", (16, 8),
                                              axes=("row", "feature")),
                            None, "input"),
                      bias,
                      Value(2, "Y", TensorType("float16", (16, 8),
                                               axes=("row", "feature")), 0,
                            "output"))
            ops = (Op(0, "add", (0, 1), (2,),
                      {"numeric_contract": CONTRACT}),)
            return compile_rkir(build(values, ops), profile=PROFILE,
                                options=options()).plan

        default = plan("col")
        custom = plan("channel")
        self.assertEqual(default.plan_hash, custom.plan_hash)
        self.assertEqual(default.to_dict(), custom.to_dict())


class ProducerRoleTests(RkirCase):
    def test_input_cannot_claim_a_producer(self):
        values = (replace_value(0, producer=0),) + matmul_values()[1:]
        self.assert_rejected(lambda: build(values),
                             "must not claim an op as its producer")

    def test_constant_cannot_claim_a_producer(self):
        values = list(matmul_values())
        values[1] = replace_value(1, producer=0)
        self.assert_rejected(lambda: build(tuple(values)),
                             "must not claim an op as its producer")

    def test_temporary_and_output_need_a_producer(self):
        values = list(matmul_values())
        values[2] = replace_value(2, producer=None)
        self.assert_rejected(lambda: build(tuple(values)),
                             "has no producer operation")

    def test_producer_must_list_the_value(self):
        values = (Value(0, "A", TensorType("float16", (16, 16)), None, "input"),
                  Value(1, "B", TensorType("float16", (16, 8)), None,
                        "constant"),
                  Value(2, "T", TensorType("float16", (16, 8)), 0,
                        "temporary"),
                  Value(3, "Y", TensorType("float16", (16, 8)), 1, "output"))
        ops = (Op(0, "relu", (0,), (3,), {"numeric_contract": CONTRACT}),)
        self.assert_rejected(
            lambda: build(values, ops, outputs=(3,)),
            "does not list it as a result")

    def test_phantom_producer_id_is_refused(self):
        values = list(matmul_values())
        values[2] = replace_value(2, producer=7)
        self.assert_rejected(lambda: build(tuple(values)),
                             "which does not exist")

    def test_legal_modules_are_unchanged(self):
        for builder in (lambda: matmul_module(m=13, k=64, n=33),
                        lambda: swiglu_module(m=64, d=256, h=128)):
            plan = compile_rkir(builder(), profile=PROFILE,
                                options=options()).plan
            self.assertTrue(plan.actions)
            self.assertTrue(plan.logical_ops)


class SignatureTests(RkirCase):
    def test_undeclared_input_is_refused(self):
        extra = Value(3, "EXTRA", TensorType("float16", (16, 16)), None, "input")
        values = matmul_values() + (extra,)
        self.assert_rejected(lambda: build(values),
                             "is not declared in Module.inputs")

    def test_declared_but_unused_input_is_refused_at_the_plan(self):
        extra = Value(3, "EXTRA", TensorType("float16", (16, 16)), None, "input")
        values = matmul_values() + (extra,)
        with self.assertRaises(BackendError) as caught:
            compile_rkir(build(values, inputs=(0, 3)), profile=PROFILE,
                         options=options())
        self.assertIn("has to be consumed by at least one upload",
                      caught.exception.diagnostics[0].message)

    def test_declared_and_used_input_is_accepted(self):
        # add(X, EXTRA) -> T, then matmul(T, B) -> Y: both declared inputs are
        # consumed, so both become plan parameters.
        extra = Value(3, "EXTRA", TensorType("float16", (16, 16)), None, "input")
        y = Value(2, "Y", TensorType("float16", (16, 8)), 1, "output")
        values = (matmul_values()[0], matmul_values()[1], y, extra,
                  Value(4, "T", TensorType("float16", (16, 16)), 0, "temporary"))
        ops = (Op(0, "add", (0, 3), (4,), {"numeric_contract": CONTRACT}),
               Op(1, "matmul", (4, 1), (2,),
                  {"b_layout": "kn", "numeric_contract": CONTRACT}, "matmul"))
        plan = compile_rkir(build(values, ops, inputs=(0, 3)), profile=PROFILE,
                            options=options()).plan
        names = [b.name for b in plan.bindings if b.role == "input"]
        self.assertEqual(sorted(names), ["A", "EXTRA"])


def replace_value(value_id, **changes):
    base = {0: ("A", (16, 16), "input"), 1: ("B", (16, 8), "constant"),
            2: ("Y", (16, 8), "output")}[value_id]
    name, shape, role = base
    return Value(value_id, name, TensorType("float16", shape),
                 data_producer(value_id, changes), changes.get("role", role))


def data_producer(value_id, changes):
    if "producer" in changes:
        return changes["producer"]
    return None if value_id == 0 else (0 if value_id == 2 else None)


class ExecutionContractTests(unittest.TestCase):
    """A legal plan still computes exactly A@B; the transpose never ran."""

    def test_only_a_times_b_is_computable(self):
        from pathlib import Path
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan

        mock = str(Path(__file__).resolve().parents[1].parent / "custom-ops" /
                   "build-host-mvp" / "librknpu_plan_mock.so")
        if not Path(mock).is_file():
            self.skipTest("mock library not built")
        plan = compile_rkir(build(), profile=PROFILE,
                            options=options()).plan
        rng = np.random.default_rng(11)
        A = np.ascontiguousarray(rng.standard_normal((16, 16)).astype(np.float16))
        B = np.ascontiguousarray(rng.standard_normal((16, 8)).astype(np.float16))
        lib = LibRKPlan(mock)
        out = [np.zeros((16, 8), dtype=np.float16)]
        call = PlanCall(lib, plan, {"B": B}, fd=0)
        try:
            call.run_into([A], out)
        finally:
            call.close()
        plain = A.astype(np.float64) @ B.astype(np.float64)
        transposed = A.astype(np.float64).T @ B.astype(np.float64)
        self.assertFalse(np.allclose(plain, transposed, atol=8e-3))
        self.assertTrue(np.allclose(out[0].astype(np.float64), plain,
                                    atol=8e-3))
        self.assertFalse(np.allclose(out[0].astype(np.float64), transposed,
                                     atol=8e-3))


class RoundTripTests(unittest.TestCase):
    def test_plan_saves_loads_and_revalidates(self):
        import tempfile
        from pathlib import Path

        plan = compile_rkir(matmul_module(m=13, k=64, n=33), profile=PROFILE,
                            options=options()).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        declared = {b.name for b in plan.bindings if b.role == "input"}
        module_inputs = {v.name for v in matmul_module(m=13, k=64, n=33).values
                         if v.id in matmul_module(m=13, k=64, n=33).inputs}
        self.assertEqual(declared, module_inputs)


if __name__ == "__main__":
    unittest.main()
