# SPDX-License-Identifier: GPL-3.0-or-later
"""Axis labels are names: they must not decide whether a computation is legal.

Each case builds the *same* computation twice, once with the default labels and
once with relabelled operands or results, and compares the whole plan plus the
mock output.  The negative cases show that a real difference (dtype, shape,
stride, index map, finite, broadcast shape, numeric contract) is still refused.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, default_profile, load_plan,
                           matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.ir import same_compute_type

PROFILE = default_profile()
CONTRACT = dict(matmul_module(m=16, k=16, n=8).ops[0].attrs["numeric_contract"])
ROW = ("row", "col")
CUSTOM = ("token", "feature")
MIXED = ("row", "feature")
MOCK = str(Path(__file__).resolve().parents[1].parent / "custom-ops" /
           "build-host-mvp" / "librknpu_plan_mock.so")
REAL = str(Path(__file__).resolve().parents[1].parent / "custom-ops" /
           "build-host-mvp" / "librknpu_plan.so")


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def binary(opcode, a_axes=ROW, b_axes=ROW, out_axes=ROW, dtype="float16",
           a_shape=(16, 8), b_shape=(16, 8), out_shape=(16, 8),
           a_stride=None, contract=None):
    values = (Value(0, "A", TensorType(dtype, a_shape, axes=a_axes,
                                     stride=a_stride), None, "input"),
              Value(1, "B", TensorType(dtype, b_shape, axes=b_axes), None,
                    "input"),
              Value(2, "Y", TensorType(dtype, out_shape, axes=out_axes), 0,
                    "output"))
    ops = (Op(0, opcode, (0, 1), (2,),
              {"numeric_contract": contract or CONTRACT}),)
    return Module(name=opcode, values=values, ops=ops, inputs=(0, 1),
                  outputs=(2,), source_hash=opcode,
                  profile_hash=PROFILE.profile_hash)


def unary(opcode, x_axes=ROW, out_axes=ROW):
    shape = (16, 8)
    values = (Value(0, "X", TensorType("float16", shape, axes=x_axes), None,
                    "input"),
              Value(1, "Y", TensorType("float16", shape, axes=out_axes), 0,
                    "output"))
    ops = (Op(0, opcode, (0,), (1,), {"numeric_contract": CONTRACT}),)
    return Module(name=opcode, values=values, ops=ops, inputs=(0,),
                  outputs=(1,), source_hash=opcode,
                  profile_hash=PROFILE.profile_hash)


def broadcast(kind, tensor_axes=ROW, out_axes=ROW, const_axes=None,
              const_len=8):
    shape = (16, 8)
    const_shape = () if kind == "scalar" else (const_len,)
    if const_axes is None:
        const_axes = () if kind == "scalar" else ("col",)
    values = (Value(0, "X", TensorType("float16", shape, axes=tensor_axes),
                    None, "input"),
              Value(1, "C", TensorType("float16", const_shape,
                                      axes=const_axes), None, "constant"),
              Value(2, "Y", TensorType("float16", shape, axes=out_axes), 0,
                    "output"))
    ops = (Op(0, "add", (0, 1), (2,), {"numeric_contract": CONTRACT}),)
    return Module(name=f"bcast-{kind}", values=values, ops=ops, inputs=(0,),
                  outputs=(2,), source_hash=f"bcast-{kind}",
                  profile_hash=PROFILE.profile_hash)


def compile_plan(module):
    return compile_rkir(module, profile=PROFILE, options=options()).plan


class LabelNeutralCompileTests(unittest.TestCase):
    def assert_same_plan(self, baseline_module, relabelled_module, label):
        baseline = compile_plan(baseline_module)
        other = compile_plan(relabelled_module)
        self.assertEqual(other.plan_hash, baseline.plan_hash, label)
        self.assertEqual(other.to_dict(), baseline.to_dict(), label)
        return baseline

    def test_binary_add_and_mul_ignore_labels(self):
        for opcode in ("add", "mul"):
            for label, axes in (("one-input", (CUSTOM, ROW, ROW)),
                                ("other-input", (ROW, CUSTOM, ROW)),
                                ("output", (ROW, ROW, CUSTOM)),
                                ("mixed", (MIXED, MIXED, MIXED))):
                with self.subTest(opcode=opcode, case=label):
                    a_axes, b_axes, out_axes = axes
                    self.assert_same_plan(binary(opcode),
                                          binary(opcode, a_axes, b_axes,
                                                 out_axes),
                                          f"{opcode}-{label}")

    def test_unary_ignores_labels(self):
        for opcode in ("relu", "sigmoid"):
            for label, axes in (("output", (ROW, CUSTOM)),
                                ("input", (CUSTOM, ROW)),
                                ("both", (CUSTOM, CUSTOM))):
                with self.subTest(opcode=opcode, case=label):
                    x_axes, out_axes = axes
                    self.assert_same_plan(unary(opcode),
                                          unary(opcode, x_axes, out_axes),
                                          f"{opcode}-{label}")

    def test_broadcast_ignores_labels(self):
        cases = (("channel-tensor", ("channel", CUSTOM, ROW, ("col",))),
                 ("channel-output", ("channel", ROW, CUSTOM, ("col",))),
                 ("channel-const", ("channel", ROW, ROW, ("bias",))),
                 ("scalar", ("scalar", CUSTOM, CUSTOM, ())),
                 ("scalar-mixed", ("scalar", MIXED, ROW, ())))
        for label, (kind, tensor_axes, out_axes, const_axes) in cases:
            with self.subTest(case=label):
                self.assert_same_plan(
                    broadcast(kind),
                    broadcast(kind, tensor_axes, out_axes, const_axes),
                    f"bcast-{label}")

    def test_relabelled_plan_roundtrips(self):
        plan = compile_plan(binary("add", CUSTOM, ROW, CUSTOM))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.to_dict(), plan.to_dict())
        labels = {(v.name, v.type.axes) for v in loaded.ir_values} \
            if hasattr(loaded, "ir_values") else None
        self.assertIsNone(labels)


class _Plain:
    """Duck-typed stand-in so the helper's unrepresentable terms stay covered."""

    def __init__(self, base_type, **changes):
        for name in ("dtype", "shape", "stride", "index_map", "finite"):
            setattr(self, name, changes.get(name, getattr(base_type, name)))


class RealDifferenceTests(unittest.TestCase):
    def assert_rejected(self, build, needle):
        """The module is validated while it is built, so build inside the guard."""
        with self.assertRaises(BackendError) as caught:
            compile_plan(build())
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_shape_mismatch_is_still_refused(self):
        self.assert_rejected(lambda: binary("add", b_shape=(16, 4)),
                             "same-shape operands")

    def test_finite_requirement_is_still_compared(self):
        def module():
            built = binary("add")
            values = list(built.values)
            values[1] = replace(values[1],
                                type=replace(values[1].type, finite=False))
            return replace(built, values=tuple(values))

        self.assert_rejected(module, "same-shape operands")

    def test_helper_still_compares_every_compute_field(self):
        base_type = TensorType("float16", (16, 8), axes=ROW)
        self.assertTrue(same_compute_type(base_type,
                                          replace(base_type,
                                                  axes=CUSTOM)))
        self.assertTrue(same_compute_type(base_type,
                                          TensorType("float16", (16, 8),
                                                     axes=MIXED,
                                                     stride=(8, 1))),
                        "an omitted stride means the canonical dense one")
        self.assertFalse(same_compute_type(base_type,
                                           TensorType("float16", (16, 4))))
        # dtype, index map and a non-dense stride cannot be expressed through
        # TensorType today, so the remaining terms are covered by feeding the
        # helper duck-typed descriptors with the same field names.
        for change in ({"dtype": "float32"}, {"index_map": "permuted"},
                       {"stride": (4, 1)}):
            with self.subTest(change=change):
                plain = _Plain(base_type)
                self.assertTrue(same_compute_type(plain, _Plain(base_type)))
                self.assertFalse(same_compute_type(
                    plain, _Plain(base_type, **change)), change)

    def test_illegal_broadcast_is_still_refused(self):
        self.assert_rejected(lambda: broadcast("channel", const_len=4),
                             "per-channel constant length must equal")

    def test_label_rules_are_unchanged(self):
        for axes, needle in (((("col", "row")), "col label describes"),
                             ((("feature", "row")), "row label describes"),
                             ((("row", "row")), "unique")):
            with self.subTest(axes=axes):
                with self.assertRaises(BackendError) as caught:
                    TensorType("float16", (16, 8), axes=axes)
                self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_numeric_contract_conflicts_are_still_refused(self):
        # The contract is part of the contract, not a label: an accumulation
        # policy the MVP does not implement stays refused, whatever the labels.
        broken = dict(CONTRACT)
        broken["accumulation"] = "unordered_fast"
        self.assert_rejected(lambda: binary("add", contract=broken),
                             "unknown accumulation policy")
        # A negative tolerance is refused as well.
        broken = dict(CONTRACT)
        broken["tolerance"] = {"atol": -1.0}
        self.assert_rejected(lambda: binary("add", contract=broken),
                             "non-negative finite numbers")


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class LabelNeutralExecutionTests(unittest.TestCase):
    """Same inputs, relabelled plans: identical device-side behaviour."""

    def _run(self, module, lib_path, values, constants):
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan

        plan = compile_plan(module)
        # A relabelled plan must need exactly the same generated plumbing
        # constants (identity / zero) as the default one.
        full = dict(constants)
        for spec in plan.constants:
            if spec.name in full:
                continue
            if spec.generator == "identity":
                full[spec.name] = np.eye(spec.shape[0], dtype=np.float16)
            elif spec.generator == "zero":
                full[spec.name] = np.zeros(tuple(spec.shape), dtype=np.float16)
            else:
                raise AssertionError("missing constant " + spec.name)
        inputs = [values[b.name] for b in plan.bindings if b.role == "input"]
        constants = full
        lib = LibRKPlan(lib_path)
        out = [np.zeros(tuple(b.shape), dtype=np.float16)
               for b in plan.bindings if b.role == "output"]
        call = PlanCall(lib, plan, constants, fd=0)
        try:
            report, _ = call.run_into(list(inputs), out)
        finally:
            call.close()
        return {"plan": plan,
                "tasks": int(report.task_count),
                "submits": int(report.submit_count),
                "sha": [hashlib_sha(a) for a in out],
                "out": out}

    def test_relabelled_plans_execute_identically(self):
        rng = np.random.default_rng(7)
        A = np.ascontiguousarray(rng.standard_normal((16, 8)).astype(np.float16))
        B = np.ascontiguousarray(rng.standard_normal((16, 8)).astype(np.float16))
        C = np.ascontiguousarray(
            rng.standard_normal((8,)).astype(np.float16))
        values = {"A": A, "B": B, "X": A, "C": C}
        cases = (("add-mixed", binary("add"), binary("add", MIXED, CUSTOM,
                                                CUSTOM)),
                 ("mul-output", binary("mul"), binary("mul", ROW, ROW,
                                                    CUSTOM)),
                 ("relu", unary("relu"), unary("relu", ROW, CUSTOM)),
                 ("bcast", broadcast("channel"),
                  broadcast("channel", CUSTOM, ROW, ("col",))))
        for label, baseline_module, relabelled in cases:
            constants = {"C": C} if label == "bcast" else {}
            with self.subTest(case=label):
                baseline = self._run(baseline_module, MOCK, values, constants)
                other = self._run(relabelled, MOCK, values, constants)
                self.assertEqual(other["plan"].to_dict(),
                                 baseline["plan"].to_dict())
                self.assertEqual(other["sha"], baseline["sha"])
                self.assertEqual(other["tasks"], baseline["tasks"])
                self.assertEqual(other["submits"], baseline["submits"])
                self.assertTrue(np.allclose(other["out"][0],
                                            baseline["out"][0],
                                            atol=0.0, rtol=0.0))

    def test_real_c_verifier_accepts_relabelled_plans(self):
        from rknpu_backend.runtime.ffi import LibRKPlan

        if not Path(REAL).is_file():
            self.skipTest("host plan library not built")
        lib = LibRKPlan(REAL)
        for module in (binary("add", CUSTOM, ROW, CUSTOM),
                       unary("relu", ROW, CUSTOM),
                       broadcast("channel", CUSTOM, ROW, ("bias",))):
            with self.subTest(module=module.name):
                cplan = lib.create_plan(compile_plan(module))
                cplan.destroy()


def hashlib_sha(array) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
