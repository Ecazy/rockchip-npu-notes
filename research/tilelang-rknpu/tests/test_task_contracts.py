# SPDX-License-Identifier: GPL-3.0-or-later
"""mm_acc / EW / LUT task contracts: Python and C must both refuse what the
emitter cannot run.

Each negative starts from a plan the planner really emits and changes exactly
the property the emitter rejects, keeping bytes, coverage and budgets
consistent, so the refusal cannot come from an unrelated construction error.
The same artifact is fed to the Python plan validator and to the C library with
that validator bypassed, so the C evidence is independent.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, compile_tilelang, default_profile,
                           matmul_module)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys_path = __import__("sys").path
if str(ROOT_DIR) not in sys_path:
    sys_path.insert(0, str(ROOT_DIR))
from models.tilelang_mlp import make_mlp_prim  # noqa: E402
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.storage import CO_GUARD, CO_SLACK
from rknpu_backend.passes.tile import K_TILE, cube_layout, weight_layout
from rknpu_backend.plan import K_TILE_LIMIT, TensorLayout, _validate_plan
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import sha256_json

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT.parent / "custom-ops" / "build-host-mvp"
REAL = str(BUILD / "librknpu_plan.so")
MOCK = str(BUILD / "librknpu_plan_mock.so")


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def cube_tensor(rows, cols):
    pr = _round_up(max(rows, 12), 4)
    pc = _round_up(cols, 32)
    return TensorLayout("float16", "co.cube.f16.v1", rows, cols, pr, pc, 8,
                        pr * pc * 2)


def weight_tensor(rows, cols):
    pr = _round_up(rows, 32)
    pc = _round_up(cols, 32)
    return TensorLayout("float16", "co.weight.f16.v1", rows, cols, pr, pc, 32,
                        pr * pc * 2)


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def rebuild(plan, *, views=None, buffers=None, bindings=None, publish=None,
            host_extra=0):
    """Rehash and rebuild without running the Python validator."""
    views = tuple(views) if views is not None else plan.views
    buffers = tuple(buffers) if buffers is not None else plan.buffers
    slot_caps: dict[int, int] = {}
    for buffer in buffers:
        slot_caps[buffer.slot_id] = max(slot_caps.get(buffer.slot_id, 0),
                                        buffer.capacity)
    reserved = sum(capacity + CO_SLACK + CO_GUARD
                   for capacity in slot_caps.values())
    command = max(len(s.task_ids) for s in plan.submits) * 2048 * 8
    accounting = replace(plan.accounting,
                         peak_payload_bytes=sum(slot_caps.values()),
                         reserved_bytes=reserved,
                         data_bo_count=len(slot_caps),
                         device_request_bytes=reserved + command,
                         command_workspace_bytes=command,
                         host_bytes=plan.accounting.host_bytes + host_extra,
                         task_count=sum(1 for a in plan.actions
                                        if a.kind == "npu_task"),
                         submit_count=len(plan.submits))
    edited = replace(plan, plan_hash="")
    object.__setattr__(edited, "views", views)
    object.__setattr__(edited, "buffers", buffers)
    if bindings is not None:
        object.__setattr__(edited, "bindings", tuple(bindings))
    if publish is not None:
        object.__setattr__(edited, "publish", tuple(publish))
    object.__setattr__(edited, "accounting", accounting)
    object.__setattr__(edited, "plan_hash",
                       sha256_json(edited.to_dict(include_hash=False)))
    return edited


def _sigmoid_plan(rows=13, cols=33):
    shape = TensorType("float16", (rows, cols))
    module = Module(name="lut", values=(Value(0, "X", shape, None, "input"),
                                        Value(1, "Y", shape, 0, "output")),
                    ops=(Op(0, "sigmoid", (0,), (1,), {}),),
                    inputs=(0,), outputs=(1,), source_hash="lut",
                    profile_hash=PROFILE.profile_hash)
    return compile_rkir(module, profile=PROFILE, options=options()).plan


def _bare_ew_plan(opcode="add", rows=16, cols=16):
    shape = TensorType("float16", (rows, cols))
    module = Module(name=f"bare-{opcode}",
                    values=(Value(0, "X", shape, None, "input"),
                            Value(1, "R", shape, None, "input"),
                            Value(2, "Y", shape, 0, "output")),
                    ops=(Op(0, opcode, (0, 1), (2,), {}),),
                    inputs=(0, 1), outputs=(2,), source_hash=f"bare-{opcode}",
                    profile_hash=PROFILE.profile_hash)
    return compile_rkir(module, profile=PROFILE, options=options()).plan


class TaskContractCase(unittest.TestCase):
    """Shared helpers: Python verdict plus the same artifact in the C library."""

    def verdicts(self, plan):
        try:
            _validate_plan(plan)
            python = "ACCEPT"
        except BackendError as exc:
            python = exc.diagnostics[0].message
        c_verdict = "ACCEPT"
        if Path(REAL).is_file():
            lib = LibRKPlan(REAL)
            try:
                cplan = lib.create_plan(plan)
                cplan.destroy()
            except BackendError as exc:
                c_verdict = exc.diagnostics[0].message
        return python, c_verdict

    def assert_rejected_on_both_layers(self, plan, needle):
        python, c_verdict = self.verdicts(plan)
        self.assertIn(needle, python)
        self.assertIn(needle, c_verdict)


class MmAccContractTests(TaskContractCase):
    def _plan(self, k=576):
        return compile_rkir(matmul_module(m=128, k=k, n=256), profile=PROFILE,
                            options=options(linear_task_order="merged")).plan

    def test_k_tile_and_tail_stay_accepted(self):
        # 576 -> 256 + 256 + 64, 320 -> 256 + 64: the tail stays legal.
        for k in (576, 320):
            with self.subTest(k=k):
                plan = self._plan(k)
                self.assertEqual(self.verdicts(plan), ("ACCEPT", "ACCEPT"))
        single = compile_rkir(matmul_module(m=128, k=256, n=256),
                              profile=PROFILE,
                              options=options(linear_task_order="baseline")).plan
        self.assertEqual(self.verdicts(single), ("ACCEPT", "ACCEPT"))

    def test_k_wider_than_one_tile_is_rejected(self):
        plan = self._plan()
        task = next(a for a in plan.actions
                    if a.template_id == "co.mm_acc.f16.v1"
                    and a.attrs.get("k0") == 256)
        x_view = plan.views[task.inputs[0]]
        w_view = plan.views[task.inputs[1]]
        wider = cube_tensor(x_view.tensor.rows, 512)
        wide_weight = weight_tensor(w_view.tensor.rows, 512)
        views = list(plan.views)
        views[x_view.id] = replace(x_view, tensor=wider,
                                   valid_region=(0, wider.rows, 0, wider.cols))
        views[w_view.id] = replace(w_view, tensor=wide_weight,
                                   valid_region=(0, wide_weight.rows, 0,
                                                 wide_weight.cols))
        buffers = list(plan.buffers)
        buffers[x_view.buffer_id] = replace(buffers[x_view.buffer_id],
                                            capacity=wider.bytes)
        buffers[w_view.buffer_id] = replace(buffers[w_view.buffer_id],
                                            capacity=wide_weight.bytes)
        # The feeding uploads have to describe the wider destination too, so the
        # only remaining defect is the K extent itself.
        actions = []
        for action in plan.actions:
            if action.kind == "upload" and action.outputs and \
                    action.outputs[0] in (x_view.id, w_view.id):
                attrs = dict(action.attrs)
                if action.outputs[0] == x_view.id:
                    attrs["src_col_offset"] = 0
                    attrs["src_row_stride"] = 576
                else:
                    attrs["src_row_offset"] = 0
                actions.append(replace(action, attrs=attrs))
            else:
                actions.append(action)
        tampered = replace(plan, actions=tuple(actions), plan_hash="")
        object.__setattr__(tampered, "plan_hash",
                           sha256_json(tampered.to_dict(include_hash=False)))
        tampered = rebuild(tampered, views=views, buffers=buffers)
        self.assert_rejected_on_both_layers(tampered, "K extent")

    def test_prior_and_output_must_share_geometry(self):
        plan = self._plan()
        task = next(a for a in plan.actions
                    if a.template_id == "co.mm_acc.f16.v1")
        out_view = plan.views[task.outputs[0]]
        # Same physical bytes (64x512 == 128x256 elements), different logical and
        # physical geometry: the accumulation reads the prior tile back with the
        # descriptor it writes.
        elements = (out_view.tensor.physical_rows *
                    out_view.tensor.physical_cols)
        other = cube_tensor(elements // 512, 512)
        self.assertEqual(other.bytes, out_view.tensor.bytes)
        views = list(plan.views)
        views[out_view.id] = replace(out_view, tensor=other,
                                     valid_region=(0, other.rows, 0,
                                                   other.cols))
        tampered = rebuild(plan, views=views)
        self.assert_rejected_on_both_layers(tampered, "prior/output tensors")


class EwContractTests(TaskContractCase):
    def test_add_mul_max_stay_accepted(self):
        for opcode in ("add", "mul"):
            with self.subTest(opcode=opcode):
                self.assertEqual(self.verdicts(_bare_ew_plan(opcode)),
                                 ("ACCEPT", "ACCEPT"))
        # co.ew_max comes from the relu epilogue of a real matmul plan.
        relu = compile_rkir(matmul_module(m=13, k=64, n=33,
                                          activation="relu"),
                            profile=PROFILE, options=options()).plan
        self.assertIn("co.ew_max.f16.v1",
                      [a.template_id for a in relu.actions])
        self.assertEqual(self.verdicts(relu), ("ACCEPT", "ACCEPT"))
        # co.ew_add_relu comes from the fused epilogue path.
        mnist = compile_tilelang(
            make_mlp_prim(1), profile=PROFILE,
            options=options(fuse_epilogue=True, fuse_add_relu=True)).plan
        self.assertIn("co.ew_add_relu.f16.v1",
                      [a.template_id for a in mnist.actions])
        self.assertEqual(self.verdicts(mnist), ("ACCEPT", "ACCEPT"))

    def test_non_cube_operands_are_rejected(self):
        plan = _bare_ew_plan()
        task = next(a for a in plan.actions if a.kind == "npu_task")
        x_view = plan.views[task.inputs[0]]
        y_view = plan.views[task.inputs[1]]
        o_view = plan.views[task.outputs[0]]
        id_view = plan.views[task.inputs[2]]
        tensor = weight_tensor(x_view.tensor.rows, x_view.tensor.cols)
        identity = weight_tensor(tensor.physical_cols, tensor.physical_cols)
        views = list(plan.views)
        for view in (x_view, y_view, o_view):
            views[view.id] = replace(view, tensor=tensor,
                                     valid_region=(0, tensor.rows, 0,
                                                   tensor.cols))
        views[id_view.id] = replace(id_view, tensor=identity,
                                    valid_region=(0, identity.rows, 0,
                                                  identity.cols))
        buffers = list(plan.buffers)
        for view in (x_view, y_view, o_view):
            buffers[view.buffer_id] = replace(buffers[view.buffer_id],
                                              capacity=tensor.bytes)
        buffers[id_view.buffer_id] = replace(buffers[id_view.buffer_id],
                                             capacity=identity.bytes)
        tampered = rebuild(plan, views=views, buffers=buffers)
        self.assert_rejected_on_both_layers(tampered, "CUBE tiles")

    def test_identity_must_match_the_operand_width(self):
        plan = _bare_ew_plan()
        task = next(a for a in plan.actions if a.kind == "npu_task")
        id_view = plan.views[task.inputs[2]]
        smaller = weight_tensor(16, 16)
        views = list(plan.views)
        views[id_view.id] = replace(id_view, tensor=smaller,
                                    valid_region=(0, smaller.rows, 0,
                                                  smaller.cols))
        tampered = rebuild(plan, views=views)
        self.assert_rejected_on_both_layers(tampered, "identity weight")


class LutContractTests(TaskContractCase):
    def test_linear_chunked_and_cube_luts_stay_accepted(self):
        for shape in ((13, 33), (16, 16), (64, 128)):
            with self.subTest(shape=shape):
                plan = _sigmoid_plan(*shape)
                self.assertEqual(self.verdicts(plan), ("ACCEPT", "ACCEPT"))

    def test_same_bytes_different_geometry_is_rejected(self):
        plan = _sigmoid_plan()
        task = next(a for a in plan.actions if a.kind == "npu_task")
        x_view = plan.views[task.inputs[0]]
        o_view = plan.views[task.outputs[0]]
        download_view = plan.views[1]
        in_tensor = cube_tensor(13, 33)
        out_tensor = cube_tensor(32, 17)
        self.assertEqual(in_tensor.bytes, out_tensor.bytes)
        views = list(plan.views)
        for view, tensor in ((x_view, in_tensor), (o_view, out_tensor),
                             (download_view, out_tensor)):
            views[view.id] = replace(view, tensor=tensor,
                                     valid_region=(0, tensor.rows, 0,
                                                   tensor.cols))
        bindings = [replace(b, shape=(out_tensor.rows, out_tensor.cols))
                    if b.role == "output" else b for b in plan.bindings]
        publish = [replace(item, rows=out_tensor.rows, cols=out_tensor.cols)
                   for item in plan.publish]
        tampered = rebuild(plan, views=views, bindings=bindings,
                           publish=publish, host_extra=8192)
        self.assert_rejected_on_both_layers(tampered, "same tile")

    def test_element_limit_is_unchanged(self):
        self.assertEqual(K_TILE, K_TILE_LIMIT)
        self.assertEqual(K_TILE_LIMIT, 256)


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class RejectedTasksNeverRunTests(unittest.TestCase):
    def test_a_rejected_task_stops_before_execution(self):
        from rknpu_backend.models import PlanCall

        plan = _bare_ew_plan()
        task = next(a for a in plan.actions if a.kind == "npu_task")
        x_view = plan.views[task.inputs[0]]
        y_view = plan.views[task.inputs[1]]
        o_view = plan.views[task.outputs[0]]
        id_view = plan.views[task.inputs[2]]
        tensor = weight_tensor(x_view.tensor.rows, x_view.tensor.cols)
        identity = weight_tensor(tensor.physical_cols, tensor.physical_cols)
        views = list(plan.views)
        for view in (x_view, y_view, o_view):
            views[view.id] = replace(view, tensor=tensor,
                                     valid_region=(0, tensor.rows, 0,
                                                   tensor.cols))
        views[id_view.id] = replace(id_view, tensor=identity,
                                    valid_region=(0, identity.rows, 0,
                                                  identity.cols))
        tampered = rebuild(plan, views=views)
        constants = {}
        for spec in tampered.constants:
            constants[spec.name] = (
                np.eye(spec.shape[0], dtype=np.float16)
                if spec.generator == "identity"
                else np.zeros(tuple(spec.shape), dtype=np.float16))
        lib = LibRKPlan(MOCK)
        with self.assertRaises(BackendError):
            PlanCall(lib, tampered, constants, fd=0)


if __name__ == "__main__":
    unittest.main()
