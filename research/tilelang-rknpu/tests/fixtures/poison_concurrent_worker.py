# SPDX-License-Identifier: GPL-3.0-or-later
"""Child process: failing run poisons while another run competes for the gate."""
import json
import os
import threading
import time

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, RK_E_BUSY,
                           RK_E_DEVICE, RK_E_POISONED, compile_rkir,
                           default_profile, matmul_module)
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding


def main() -> int:
    lib = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"])
    plan = compile_rkir(
        matmul_module(m=8, k=256, n=32), profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    cplan = lib.create_plan(plan)
    rng = np.random.default_rng(111)
    a = np.asarray(rng.normal(0, 0.4, (8, 256)), dtype=np.float16)
    b = np.asarray(rng.normal(0, 0.2, (256, 32)), dtype=np.float16)
    consts = [make_binding(spec.binding_id, b)
              for spec in plan.constants if spec.name == "B"]
    ws = lib.create_workspace(cplan, consts, fd=0)
    in_id = next(x.id for x in plan.bindings if x.role == "input")
    out_id = next(x.id for x in plan.bindings if x.role == "output")
    y1 = np.full((8, 32), np.float16(1.5))
    y2 = np.full((8, 32), np.float16(-1.5))
    sentinel1, sentinel2 = y1.tobytes(), y2.tobytes()
    before_execs = lib.mock_control("rk_mock_execs")
    before_bos = lib.mock_control("rk_mock_live_bos")
    lib.mock_control("rk_mock_fail_at", before_execs + 1)
    lib.mock_control("rk_mock_delay_us", 500_000)
    out = {}

    def failing_run():
        try:
            ws.run([make_binding(in_id, a)], [make_binding(out_id, y1)])
            out["first"] = "NO_ERROR"
        except BackendError as exc:
            out["first"] = exc.diagnostics[0].code

    thread = threading.Thread(target=failing_run)
    thread.start()
    time.sleep(0.05)
    try:
        ws.run([make_binding(in_id, a)], [make_binding(out_id, y2)])
        out["second"] = "NO_ERROR"
    except BackendError as exc:
        out["second"] = exc.diagnostics[0].code
    out["execs_after_second"] = lib.mock_control("rk_mock_execs")
    thread.join(timeout=5)
    lib.mock_control("rk_mock_delay_us", 0)
    out["execs_after_join"] = lib.mock_control("rk_mock_execs")
    try:
        ws.run([make_binding(in_id, a)], [make_binding(out_id, y2)])
        out["third"] = "NO_ERROR"
    except BackendError as exc:
        out["third"] = exc.diagnostics[0].code
    out["execs_after_third"] = lib.mock_control("rk_mock_execs")
    out["outputs_unchanged"] = (
        y1.tobytes() == sentinel1 and y2.tobytes() == sentinel2)
    try:
        ws.destroy()
        out["destroy"] = "NO_ERROR"
    except BackendError as exc:
        out["destroy"] = exc.diagnostics[0].code
    out["handle_retained"] = bool(ws.handle)
    out["bos_retained"] = lib.mock_control("rk_mock_live_bos") == before_bos
    ok = (
        out["first"] == RK_E_DEVICE and
        out["second"] == RK_E_BUSY and
        out["execs_after_second"] == before_execs and
        out["execs_after_join"] == before_execs + 1 and
        out["third"] == RK_E_POISONED and
        out["execs_after_third"] == before_execs + 1 and
        out["outputs_unchanged"] and
        out["destroy"] == RK_E_POISONED and
        out["handle_retained"] and out["bos_retained"])
    print(json.dumps({"status": "PASS" if ok else "FAIL", **out},
                     sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
