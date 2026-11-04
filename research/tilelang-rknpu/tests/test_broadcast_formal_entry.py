# SPDX-License-Identifier: GPL-3.0-or-later
"""Formal local-dataflow entry for scalar and column broadcast tails."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

from rknpu_backend import (CompileOptions, RK_E_SHAPE, compile_tilelang,
                           default_profile)
from rknpu_backend.artifact import compile_artifact
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.full_logical_rewrite import rewrite_ffn_tir
from rknpu_backend.plan import Plan
from tests.fixtures import tilelang_broadcast

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"
CASES = (
    ("scalar_add", tilelang_broadcast.scalar_add, "scalar", (), (8, 33)),
    ("scalar_mul", tilelang_broadcast.scalar_mul, "scalar", (), (8, 33)),
    ("channel_add", tilelang_broadcast.channel_add, "channel", (33,), (8, 33)),
    ("channel_mul", tilelang_broadcast.channel_mul, "channel", (33,), (8, 33)),
)


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled")


def frozen(name):
    path = ROOT / "tests" / "fixtures" / "frozen_plans" / f"{name}.json"
    return Plan.from_dict(json.loads(path.read_text())["plan"])


def execution(plan):
    return {
        "bindings": [(b.name, b.role, tuple(b.shape)) for b in plan.bindings],
        "actions": [(a.kind, a.template_id, tuple(a.inputs), tuple(a.outputs),
                     tuple(a.deps), tuple(sorted(a.attrs.items())))
                    for a in plan.actions],
        "submits": [(len(s.task_ids), s.recipe_id, s.mode, s.deps)
                    for s in plan.submits],
        "publish": [(p.name, p.dst_row, p.dst_col, p.rows, p.cols)
                    for p in plan.publish],
        "accounting": plan.accounting.to_dict(),
    }


class BroadcastFormalEntryTests(unittest.TestCase):
    def compile(self, prim, name):
        if True:
            result = compile_tilelang(prim, profile=PROFILE, options=options())
            artifact = compile_artifact(prim, profile=PROFILE,
                                        options=options())
        self.assertEqual(result.plan.plan_hash, artifact.plan_hash)
        self.assertEqual(execution(result.plan), execution(frozen(name)))
        return result

    def test_four_real_inputs_use_formal_entry(self):
        frozen_names = ("broadcast_scalar_add", "broadcast_scalar_mul",
                        "broadcast_channel_add", "broadcast_channel_mul")
        for (name, factory, kind, shape, target), frozen_name in zip(
                CASES, frozen_names):
            with self.subTest(name=name):
                result = self.compile(factory(m=8, k=64, n=33), frozen_name)
                plan = result.plan
                self.assertEqual(plan.accounting.task_count, 2)
                self.assertEqual([x.name for x in plan.publish], ["Y"])
                bindings = {b.name: b for b in plan.bindings}
                self.assertEqual(bindings["B"].shape, shape)
                broadcast = [a for a in plan.actions
                             if a.kind == "upload" and
                             a.attrs.get("broadcast") == kind]
                self.assertEqual(len(broadcast), 1)
                self.assertEqual(broadcast[0].attrs["broadcast_target"],
                                 list(target))
                self.assertEqual(plan.accounting.broadcast_expand_bytes,
                                 12 * 64 * 2)

    def test_actual_broadcast_mapping_is_in_the_task(self):
        for name, factory, kind, shape, target in CASES:
            with self.subTest(name=name):
                data = rewrite_ffn_tir(factory(m=8, k=64, n=33))
                binary = next(task for task in data.tasks
                              if task["kind"] in {"add", "mul"})
                self.assertEqual(binary["input_broadcasts"],
                                 ["none", kind])
                bias_value = data.module.value(binary["input_values"][1])
                self.assertEqual(bias_value.type.shape, shape)
                self.assertEqual(bias_value.role, "constant")
                self.assertEqual(binary["region"]["rows"], 8)
                self.assertEqual(binary["region"]["cols"], 33)

    def test_wrong_channel_length_is_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(
                tilelang_broadcast.channel_mul(m=8, k=64, n=33, bias_n=32),
                profile=PROFILE, options=options())
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE)


class BroadcastMockTests(unittest.TestCase):
    def run_pair(self, factory, name):
        if not LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.runtime.ffi import LibRKPlan, make_binding
        old = frozen(name)
        new = compile_tilelang(factory(m=8, k=64, n=33), profile=PROFILE,
                               options=options()).plan
        rng = np.random.default_rng(20261108)
        w = rng.normal(0, 0.2, (64, 33)).astype(np.float16)
        bias_shape = next(b.shape for b in old.bindings if b.name == "B")
        bias = (np.asarray(0.25, dtype=np.float16) if bias_shape == () else
                rng.normal(0, 0.5, bias_shape).astype(np.float16))
        constants = []
        for binding in old.bindings:
            if binding.role != "constant":
                continue
            if binding.name.startswith("__identity_"):
                array = np.eye(binding.shape[0], dtype=np.float16)
            elif binding.name == "W":
                array = w
            else:
                array = bias
            constants.append(make_binding(binding.id, array))
        x = rng.normal(0, 0.3, (8, 64)).astype(np.float16)
        lib = LibRKPlan(str(LIB))
        values = []
        for plan in (old, new):
            cplan = lib.create_plan(plan)
            workspace = lib.create_workspace(cplan, constants)
            in_id = next(b.id for b in plan.bindings if b.role == "input")
            out_id = next(b.id for b in plan.bindings if b.role == "output")
            y = np.zeros((8, 33), dtype=np.float16)
            try:
                workspace.run([make_binding(in_id, x)],
                              [make_binding(out_id, y)])
            finally:
                workspace.destroy()
                cplan.destroy()
            values.append(y)
        np.testing.assert_array_equal(values[0], values[1])

    def test_four_broadcast_mock_outputs(self):
        frozen_names = ("broadcast_scalar_add", "broadcast_scalar_mul",
                        "broadcast_channel_add", "broadcast_channel_mul")
        for (name, factory, _kind, _shape, _target), frozen_name in zip(
                CASES, frozen_names):
            with self.subTest(name=name):
                self.run_pair(factory, frozen_name)


if __name__ == "__main__":
    unittest.main()
