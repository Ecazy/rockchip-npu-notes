# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess worker: a guard failure poisons the workspace for process lifetime."""
import json
import os
import sys

import numpy as np

from rknpu_backend import (CompileOptions, RK_E_DEVICE, RK_E_POISONED,
                           compile_rkir, default_profile, matmul_module)
from rknpu_backend.errors import BackendError
from rknpu_backend.runtime.ffi import LibRKPlan, make_binding


def main() -> int:
    plan = compile_rkir(
        matmul_module(m=8, k=64, n=33), profile=default_profile(),
        options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    lib = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"])
    cplan = lib.create_plan(plan)
    rng = np.random.default_rng(79)
    a = np.asarray(rng.normal(0, 0.3, (8, 64)), dtype=np.float16)
    b = np.asarray(rng.normal(0, 0.2, (64, 33)), dtype=np.float16)
    constants = []
    for spec in plan.constants:
        array = b if spec.name == "B" else np.asarray(
            np.eye(spec.shape[0]), dtype=np.float16)
        constants.append(make_binding(spec.binding_id, array))
    ws = lib.create_workspace(cplan, constants)
    in_id = next(binding.id for binding in plan.bindings
                 if binding.role == "input")
    out_id = next(binding.id for binding in plan.bindings
                  if binding.role == "output")
    y = np.full((8, 33), np.float16(1.25))
    snapshot = y.tobytes()
    lib.mock_control("rk_mock_corrupt_guard", 1)
    try:
        ws.run([make_binding(in_id, a)], [make_binding(out_id, y)])
    except BackendError as exc:
        code = exc.diagnostics[0].code
        message = exc.diagnostics[0].message
    else:
        raise AssertionError("guard-corrupted run was not rejected")
    if code != RK_E_DEVICE:
        raise AssertionError(f"guard failure code {code} != RK_E_DEVICE")
    if "guard" not in message.lower():
        raise AssertionError(f"guard diagnostic lacks guard context: {message}")
    if y.tobytes() != snapshot:
        raise AssertionError("failed guard run modified the output")
    lib.mock_control("rk_mock_corrupt_guard", 0)
    try:
        ws.run([make_binding(in_id, a)], [make_binding(out_id, y)])
    except BackendError as exc:
        poison_code = exc.diagnostics[0].code
    else:
        raise AssertionError("poisoned workspace accepted a second run")
    if poison_code != RK_E_POISONED:
        raise AssertionError(f"second-run code {poison_code} != RK_E_POISONED")
    print(json.dumps({"status": "PASS", "guard_code": code,
                      "poison_code": poison_code,
                      "output_unchanged": True}, sort_keys=True))
    # Deliberately do not destroy the poisoned workspace: the production contract
    # requires process-level recovery.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
