#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-workspace isolation check for the creation-time co_program cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.models import PlanCall
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi

FLAG = "RK_PLAN_PROGRAM_CACHE"


def load_case(case: Path, plans_dir: Path, frame_index: int):
    meta = json.loads((case / "case.json").read_text())
    program = load_hybrid_program(
        plans_dir / f"{meta['manifest_key']}.rkhybrid.json",
        plans_dir=plans_dir)
    plan = program.plans[meta["plan_key"]].plan
    raw = (case / "frames.bin").read_bytes()
    sizes = [int(x["nbytes"]) for x in meta["inputs"]]
    stride = sum(sizes)
    base = frame_index * stride
    inputs = []
    for x in meta["inputs"]:
        n = int(x["nbytes"])
        inputs.append(np.frombuffer(raw[base:base + n], dtype="<f2")
                      .reshape(x["shape"]).copy())
        base += n
    outputs = [np.zeros(x["shape"], dtype=np.float16)
               for x in meta["outputs"]]
    return plan, inputs, outputs


def make_call(lib, plan, fd, frame_dir, frame_index, cache):
    old = os.environ.get(FLAG)
    os.environ[FLAG] = "1" if cache else "0"
    try:
        plan_obj, inputs, outputs = load_case(frame_dir, PLANS, frame_index)
        call = PlanCall(lib, plan_obj, {}, fd=fd, entry="legacy")
        call.bind_prepared_into(inputs, outputs, binding_key="iso")
        return call, inputs, outputs
    finally:
        if old is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old


def run(call):
    _, _, in_bind, out_bind = call._prepared_single["iso"]
    return call.lib.workspace_run(call.workspace, in_bind, out_bind)


def main():
    global PLANS
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    PLANS = args.plans_dir
    lib = LibRKPlan(args.lib)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise RuntimeError(f"rocket_open failed: {fd}")
    calls = []
    try:
        # Two cache-enabled workspaces with different input frames, and a
        # cache-disabled reference workspace.
        a, a_in, a_out = make_call(lib, None, fd, args.case, 0, True)
        calls.append(a)
        b, b_in, b_out = make_call(lib, None, fd, args.case, 1, True)
        calls.append(b)
        c, c_in, c_out = make_call(lib, None, fd, args.case, 0, False)
        calls.append(c)
        run(a); a0 = [x.copy() for x in a_out]
        run(b); b1 = [x.copy() for x in b_out]
        run(a); a0_again = [x.copy() for x in a_out]
        run(c); c0 = [x.copy() for x in c_out]
        # Feed B's frame through the cache-off reference too.
        for dst, src in zip(c_in, b_in):
            np.copyto(dst, src)
        run(c); c1 = [x.copy() for x in c_out]
        result = {
            "status": "PASS" if (
                all(np.array_equal(x, y) for x, y in zip(a0, c0)) and
                all(np.array_equal(x, y) for x, y in zip(b1, c1)) and
                all(np.array_equal(x, y) for x, y in zip(a0, a0_again))
            ) else "FAIL",
            "frame0_cache_matches_reference": all(
                np.array_equal(x, y) for x, y in zip(a0, c0)),
            "frame1_other_workspace_matches_reference": all(
                np.array_equal(x, y) for x, y in zip(b1, c1)),
            "repeat_same_workspace_unchanged": all(
                np.array_equal(x, y) for x, y in zip(a0, a0_again)),
            "workspace_count": 3,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    finally:
        for call in calls:
            call.close()
        lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
