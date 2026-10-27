#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative/positive checks for the experimental prepared-validation cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.models import PlanCall
from rknpu_backend.runtime.ffi import LibRKPlan


def load_case(case: Path, plans_dir: Path):
    meta = json.loads((case / "case.json").read_text())
    program = load_hybrid_program(
        plans_dir / f"{meta['manifest_key']}.rkhybrid.json",
        plans_dir=plans_dir)
    plan = program.plans[meta["plan_key"]].plan
    frames_raw = (case / "frames.bin").read_bytes()
    in_meta = meta["inputs"]
    stride = sum(int(x["nbytes"]) for x in in_meta)
    frame = 0
    inputs = []
    base = frame * stride
    for x in in_meta:
        n = int(x["nbytes"])
        inputs.append(np.frombuffer(frames_raw[base:base + n], dtype="<f2")
                      .reshape(x["shape"]).copy())
        base += n
    outputs = [np.zeros(x["shape"], dtype=np.float16)
               for x in meta["outputs"]]
    return program, plan, inputs, outputs


def run(call, in_bind, out_bind):
    return call.lib.workspace_run(call.workspace, in_bind, out_bind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--board", action="store_true")
    args = ap.parse_args()
    program, plan, inputs, outputs = load_case(args.case, args.plans_dir)
    old = os.environ.get("RK_PLAN_PREPARED_VALIDATION")
    os.environ["RK_PLAN_PREPARED_VALIDATION"] = "1"
    lib = LibRKPlan(args.lib)
    fd = 0
    if args.board:
        from rknpu_backend.runtime.ffi import configure_rocket_abi
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise RuntimeError(f"rocket_open failed: {fd}")
    call = None
    try:
        call = PlanCall(lib, plan, {}, fd=fd, entry="legacy")
        call.bind_prepared_into(inputs, outputs, binding_key="static")
        in_arrays, out_arrays, in_bind, out_bind = \
            call._prepared_single["static"]
        run(call, in_bind, out_bind)
        baseline = [x.copy() for x in outputs]

        # Wrong static shape: exact cache miss, full validation must reject.
        in_bind[0].shape[0] = int(in_bind[0].shape[0]) + 1
        rejected_shape = False
        try:
            run(call, in_bind, out_bind)
        except Exception:
            rejected_shape = True
        in_bind[0].shape[0] = int(in_bind[0].shape[0]) - 1
        shape_unchanged = all(np.array_equal(a, b)
                              for a, b in zip(outputs, baseline))

        # Same cached pointer, non-finite input: per-call finite scan must
        # reject and must not publish partial output.
        saved = inputs[0].copy()
        inputs[0].flat[0] = np.nan
        rejected_finite = False
        try:
            run(call, in_bind, out_bind)
        except Exception:
            rejected_finite = True
        inputs[0][...] = saved
        finite_unchanged = all(np.array_equal(a, b)
                               for a, b in zip(outputs, baseline))

        # Address change must force full validation and refresh the cache.
        replacement = inputs[0].copy()
        old_ptr = int(in_bind[0].data)
        in_bind[0].data = int(replacement.ctypes.data)
        run(call, in_bind, out_bind)
        addr_revalidated = int(in_bind[0].data) == int(replacement.ctypes.data)
        in_bind[0].data = old_ptr
        run(call, in_bind, out_bind)
        final_unchanged = all(np.array_equal(a, b)
                              for a, b in zip(outputs, baseline))

        result = {
            "status": "PASS" if (rejected_shape and rejected_finite and
                                 shape_unchanged and finite_unchanged and
                                 addr_revalidated and final_unchanged) else "FAIL",
            "rejected_shape": rejected_shape,
            "rejected_finite": rejected_finite,
            "shape_output_unchanged": shape_unchanged,
            "finite_output_unchanged": finite_unchanged,
            "address_revalidated": addr_revalidated,
            "restored_output_unchanged": final_unchanged,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    finally:
        if call is not None:
            call.close()
        if old is None:
            os.environ.pop("RK_PLAN_PREPARED_VALIDATION", None)
        else:
            os.environ["RK_PLAN_PREPARED_VALIDATION"] = old
        if args.board and fd >= 0:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
