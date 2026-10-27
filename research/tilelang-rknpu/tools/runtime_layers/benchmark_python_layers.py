#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python A arms for the runtime-layering experiment.

A0 = ordinary PlanCall.run_into/run_group_into (Python validation + ctypes + C)
A1 = prepared binding array, direct legacy rk_workspace_run/group call from
     Python ctypes (no per-call Python shape/finite scan; same C runtime).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.models import PlanCall
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi


def now_us():
    return time.perf_counter() * 1e6


def read_frames(case, inputs):
    raw = (case / "frames.bin").read_bytes()
    sizes = [int(item["nbytes"]) for item in inputs]
    stride = sum(sizes)
    count = int(json.loads((case / "case.json").read_text())["frames"])
    if len(raw) != stride * count:
        raise SystemExit("frame file size mismatch")
    out = []
    for f in range(count):
        base = f * stride
        frame = []
        for item, n in zip(inputs, sizes):
            frame.append(np.frombuffer(
                raw[base:base + n], dtype="<f2").reshape(item["shape"]).copy())
            base += n
        out.append(frame)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--mode", choices=("a0", "a1"), required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--plans-dir", type=Path)
    ap.add_argument("--weights-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    case = args.case
    meta = json.loads((case / "case.json").read_text())
    plans_dir = args.plans_dir or Path(meta["plans_dir"])
    weights_dir = args.weights_dir or Path(meta["weights_dir"])
    program = load_hybrid_program(
        plans_dir / f"{meta['manifest_key']}.rkhybrid.json",
        plans_dir=plans_dir)
    plan = program.plans[meta["plan_key"]].plan
    constants = {}
    for item in meta["constants"]:
        constants[item["name"]] = np.frombuffer(
            (case / f"const_{item['id']}.bin").read_bytes(),
            dtype="<f2").reshape(item["shape"]).copy()
    inputs_meta = meta["inputs"]
    outputs_meta = meta["outputs"]
    frames = read_frames(case, inputs_meta)
    inputs = [np.empty(item["shape"], dtype=np.float16)
              for item in inputs_meta]
    outputs = [np.zeros(item["shape"], dtype=np.float16)
               for item in outputs_meta]
    lib = LibRKPlan(args.lib)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise SystemExit(f"rocket_open failed: {fd}")
    try:
        t0 = now_us()
        call = PlanCall(lib, plan, constants, fd=fd, entry="legacy")
        setup_us = now_us() - t0
        bind_us = 0.0
        if args.mode == "a1":
            t0 = now_us()
            if meta["kind"] == "group":
                call.bind_group_prepared_into(
                    [inputs], [outputs], binding_key="runtime-layer")
            else:
                call.bind_prepared_into(
                    inputs, outputs, binding_key="runtime-layer")
            bind_us = now_us() - t0
        if meta["kind"] == "group":
            def run():
                if args.mode == "a1":
                    flat, nrepeat, _, _ = call._prepared_group["runtime-layer"]
                    return call.lib.workspace_run_group(
                        call.workspace, flat, nrepeat)
                return call.run_group_into([inputs], [outputs])
        else:
            def run():
                if args.mode == "a1":
                    _, _, in_bind, out_bind = \
                        call._prepared_single["runtime-layer"]
                    return call.lib.workspace_run(
                        call.workspace, in_bind, out_bind)
                return call.run_into(inputs, outputs)

        def load_frame(frame_index):
            for dst, src in zip(inputs, frames[frame_index]):
                np.copyto(dst, src)

        load_frame(0)
        t0 = now_us()
        run()
        first_us = now_us() - t0
        warm = []
        for _ in range(args.rounds):
            for f in range(1, len(frames)):
                load_frame(f)
                t0 = now_us()
                run()
                warm.append(now_us() - t0)
        with args.out.open("wb") as f:
            for out in outputs:
                f.write(out.tobytes(order="C"))
        print(f"SETUP_US {setup_us:.6f}")
        print(f"BIND_US {bind_us:.6f}")
        print(f"FIRST_US {first_us:.6f}")
        print("WARM_US " + " ".join(f"{v:.6f}" for v in warm))
        call.close()
    finally:
        lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
