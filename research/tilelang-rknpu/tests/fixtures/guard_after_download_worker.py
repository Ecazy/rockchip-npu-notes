# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess regression: a later submit after a download corrupts a BO guard.

The first output is downloaded/staged before the second submit.  The second
submit corrupts its output BO guard.  Final publication must still be refused and
neither caller output may be modified.
"""
import os
from dataclasses import replace
import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, RK_E_DEVICE,
                           TensorType, Value, compile_rkir, default_profile,
                           matmul_module)
from rknpu_backend.errors import BackendError
from rknpu_backend.plan import with_plan_hash
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding


def branch_module(m=8, k=64, n1=32, n2=48):
    contract = matmul_module(m=m, k=k, n=n1, b_layout="nk").ops[0].attrs[
        "numeric_contract"]
    attrs = {"b_layout": "nk", "numeric_contract": contract}
    return Module(
        name="guard-after-download",
        values=(Value(0, "X", TensorType("float16", (m, k)), None, "input"),
                Value(1, "W1", TensorType("float16", (n1, k)), None, "constant"),
                Value(2, "W2", TensorType("float16", (n2, k)), None, "constant"),
                Value(3, "Y1", TensorType("float16", (m, n1)), 0, "output"),
                Value(4, "Y2", TensorType("float16", (m, n2)), 1, "output")),
        ops=(Op(0, "matmul", (0, 1), (3,), dict(attrs)),
             Op(1, "matmul", (0, 2), (4,), dict(attrs))),
        inputs=(0,), outputs=(3, 4), source_hash="test")


def interleaved_plan():
    plan = compile_rkir(
        branch_module(), profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    old = plan.actions
    # task A -> download A -> task B -> download B.  A host download between
    # submits is a legal hard boundary.
    actions = (
        old[0], old[1], old[2],
        replace(old[3], id=3, deps=(0, 1)),
        replace(old[5], id=4, deps=(3,)),
        replace(old[4], id=5, deps=(0, 2)),
        replace(old[6], id=6, deps=(5,)),
    )
    submits = (
        replace(plan.submits[0], id=0, task_ids=(3,), deps=()),
        replace(plan.submits[1], id=1, task_ids=(5,), deps=(0,)),
    )
    buffers = (
        replace(plan.buffers[0], lifetime=(0, 5)),
        replace(plan.buffers[1], lifetime=(1, 4)),
        replace(plan.buffers[2], lifetime=(2, 6)),
        replace(plan.buffers[3], lifetime=(3, 4)),
        replace(plan.buffers[4], lifetime=(5, 6)),
    )
    return with_plan_hash(replace(
        plan, actions=actions, submits=submits, buffers=buffers, plan_hash=""))


def main() -> int:
    plan = interleaved_plan()
    lib = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"])
    cplan = lib.create_plan(plan)
    rng = np.random.default_rng(83)
    x = np.asarray(rng.normal(0, 0.3, (8, 64)), dtype=np.float16)
    w1 = np.asarray(rng.normal(0, 0.2, (32, 64)), dtype=np.float16)
    w2 = np.asarray(rng.normal(0, 0.2, (48, 64)), dtype=np.float16)
    constants = []
    for spec in plan.constants:
        array = w1 if spec.name == "W1" else w2
        constants.append(make_binding(spec.binding_id, array))
    ws = lib.create_workspace(cplan, constants)
    in_id = next(b.id for b in plan.bindings if b.role == "input")
    out_ids = [b.id for b in plan.bindings if b.role == "output"]
    y1 = np.full((8, 32), np.float16(1.25))
    y2 = np.full((8, 48), np.float16(-0.75))
    snapshots = (y1.tobytes(), y2.tobytes())
    lib.mock_control("rk_mock_corrupt_guard", 1)
    lib.mock_control("rk_mock_corrupt_guard_at", 2)
    try:
        ws.run([make_binding(in_id, x)],
               [make_binding(out_ids[0], y1), make_binding(out_ids[1], y2)])
    except BackendError as exc:
        diag = exc.diagnostics[0]
    else:
        raise AssertionError("post-download guard corruption was not rejected")
    if diag.code != RK_E_DEVICE or "guard" not in diag.message.lower():
        raise AssertionError(f"unexpected diagnostic {diag.code}: {diag.message}")
    if (y1.tobytes(), y2.tobytes()) != snapshots:
        raise AssertionError("a caller output was modified before the guard gate")
    print("PASS post-download task guard corruption refused publication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
