# SPDX-License-Identifier: GPL-3.0-or-later
"""Child: one provider-context fault blocks all workspaces sharing that fd."""
import json
import os

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, RK_E_DEVICE,
                           RK_E_POISONED, compile_rkir, default_profile,
                           matmul_module)
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding


def main() -> int:
    lib = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"])
    plan = compile_rkir(
        matmul_module(m=8, k=256, n=32), profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    cplan = lib.create_plan(plan)
    rng = np.random.default_rng(121)
    a = np.asarray(rng.normal(0, 0.4, (8, 256)), dtype=np.float16)
    b = np.asarray(rng.normal(0, 0.2, (256, 32)), dtype=np.float16)
    consts = [make_binding(spec.binding_id, b)
              for spec in plan.constants if spec.name == "B"]
    ws1 = lib.create_workspace(cplan, consts, fd=0)
    ws2 = lib.create_workspace(cplan, consts, fd=0)
    in_id = next(x.id for x in plan.bindings if x.role == "input")
    out_id = next(x.id for x in plan.bindings if x.role == "output")
    y1 = np.full((8, 32), np.float16(1.25))
    y2 = np.full((8, 32), np.float16(-1.25))
    s1, s2 = y1.tobytes(), y2.tobytes()
    before_execs = lib.mock_control("rk_mock_execs")
    before_bos = lib.mock_control("rk_mock_live_bos")
    lib.mock_control("rk_mock_fail_at", before_execs + 1)
    out = {}

    try:
        ws1.run([make_binding(in_id, a)], [make_binding(out_id, y1)])
        out["ws1"] = "NO_ERROR"
    except BackendError as exc:
        out["ws1"] = exc.diagnostics[0].code
    out["context_faulted"] = lib.mock_control("rk_mock_context_faulted", 0)
    out["execs_after_failure"] = lib.mock_control("rk_mock_execs")

    try:
        ws2.run([make_binding(in_id, a)], [make_binding(out_id, y2)])
        out["ws2"] = "NO_ERROR"
    except BackendError as exc:
        out["ws2"] = exc.diagnostics[0].code
    out["execs_after_ws2"] = lib.mock_control("rk_mock_execs")
    out["outputs_unchanged"] = (
        y1.tobytes() == s1 and y2.tobytes() == s2)

    try:
        ws3 = lib.create_workspace(cplan, consts, fd=0)
        out["new_workspace"] = "CREATED"
        # This must never run; keep the handle for process-exit cleanup.
        out["ws3_handle"] = bool(ws3.handle)
    except BackendError as exc:
        out["new_workspace"] = exc.diagnostics[0].code

    out["destroy1"] = "NO_ERROR"
    try:
        ws1.destroy()
        out["destroy1"] = "NO_ERROR"
    except BackendError as exc:
        out["destroy1"] = exc.diagnostics[0].code
    try:
        ws2.destroy()
        out["destroy2"] = "NO_ERROR"
    except BackendError as exc:
        out["destroy2"] = exc.diagnostics[0].code
    out["handles_retained"] = bool(ws1.handle) and bool(ws2.handle)
    out["bos_retained"] = lib.mock_control("rk_mock_live_bos") == before_bos
    ok = (
        out["ws1"] == RK_E_DEVICE and
        out["context_faulted"] == 1 and
        out["execs_after_failure"] == before_execs + 1 and
        out["ws2"] == RK_E_POISONED and
        out["execs_after_ws2"] == before_execs + 1 and
        out["outputs_unchanged"] and
        out["new_workspace"] == RK_E_POISONED and
        out["destroy1"] == RK_E_POISONED and
        out["destroy2"] == RK_E_POISONED and
        out["handles_retained"] and out["bos_retained"])
    print(json.dumps({"status": "PASS" if ok else "FAIL", **out},
                     sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
