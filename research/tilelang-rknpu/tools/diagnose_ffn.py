#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic capture for the real H/R SmolLM FFN.

The production plan remains publish-only-Y.  This tool captures the actual
device gate/up and down projections through the Group-A plans, and records the
host sigmoid/SiLU/activation derived from the actual device gate/up.  It is not
used for timing and is not a candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from run_ffn_experiment import (D, H, f16, lut_sigmoid, r0_host,
                                r2_declared_lut)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--layers", default="0,15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    root = Path(args.plans).resolve()
    man = json.loads((root / "manifest.json").read_text())
    data = np.load(root / man["inputs"], allow_pickle=False)
    m = int(man["m"])
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())

    def constants(plan, vals):
        prepared, binds = [], []
        for c in plan.constants:
            arr = np.eye(c.shape[0], dtype=np.float16) if c.generator == "identity" else vals[c.name]
            prepared.append(arr)
            binds.append(make_binding(c.binding_id, arr))
        return prepared, binds

    def run_case(name, h, r, wgu, wd):
        p0 = load_plan(str(root / man["A"]["gate_up"]["plan"]))
        p1 = load_plan(str(root / man["A"]["down"]["plan"]))
        pb = load_plan(str(root / "B_fixed_192.rkplan.json"))
        c0 = lib.create_plan(p0); c1 = lib.create_plan(p1); cb = lib.create_plan(pb)
        a0, b0 = constants(p0, {"B": wgu})
        a1, b1 = constants(p1, {"B": wd})
        ab, bb = constants(pb, {"Wgu": wgu, "Wd": wd})
        w0 = lib.create_workspace(c0, b0, fd=fd)
        w1 = lib.create_workspace(c1, b1, fd=fd)
        wb = lib.create_workspace(cb, bb, fd=fd)
        try:
            i0 = next(b.id for b in p0.bindings if b.role == "input")
            o0 = next(b.id for b in p0.bindings if b.role == "output")
            i1 = next(b.id for b in p1.bindings if b.role == "input")
            o1 = next(b.id for b in p1.bindings if b.role == "output")
            ins = {b.name: b.id for b in pb.bindings if b.role == "input"}
            out = next(b.id for b in pb.bindings if b.role == "output")
            z = np.zeros((m, 2 * H), dtype=np.float16)
            p = np.zeros((m, D), dtype=np.float16)
            y = np.zeros((m, D), dtype=np.float16)
            w0.run([make_binding(i0, h)], [make_binding(o0, z)])
            gate = z[:, :H]; up = z[:, H:]
            sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
            silu = f16(gate * sig)
            act = f16(silu * up)
            w1.run([make_binding(i1, act)], [make_binding(o1, p)])
            wb.run([make_binding(ins["H"], h),
                    make_binding(ins["R"], r)], [make_binding(out, y)])
            r0 = r0_host(h, r, wgu, wd)[0]
            r2 = r2_declared_lut(h, r, wgu, wd)[0]
            return {
                f"{name}_gate_hw": gate, f"{name}_up_hw": up,
                f"{name}_sigmoid_host": sig, f"{name}_silu_host": silu,
                f"{name}_activation_host": act, f"{name}_down_hw": p,
                f"{name}_Y_hw": y, f"{name}_Y_r0": r0, f"{name}_Y_r2": r2,
                f"{name}_H": h, f"{name}_R": r,
            }
        finally:
            w0.destroy(); w1.destroy(); wb.destroy()
            c0.destroy(); c1.destroy(); cb.destroy()

    arrays = {}
    try:
        for layer in [int(x) for x in args.layers.split(",") if x]:
            h = data[f"h{layer}"]; r = data[f"r{layer}"]
            arrays.update(run_case(f"layer{layer}", h, r,
                                   data[f"wgu{layer}"], data[f"wd{layer}"]))
        # Synthetic sparse case: gate and up are driven directly so LUT edge
        # and saturation values reach the real device final add.
        gates = np.array([0.0, 0.0015, -0.0015, 0.003, -0.003,
                          6.3112, -6.3112, 6.32, -6.32, 50.0, -9.79,
                          0.5, -0.5], dtype=np.float16)
        if m < gates.size:
            gates = gates[:1]
        h = np.zeros((m, D), dtype=np.float16)
        r = np.zeros((m, D), dtype=np.float16)
        h[:gates.size, 0] = gates
        h[:gates.size, 1] = np.float16(1.0)
        wgu = np.zeros((D, 2 * H), dtype=np.float16)
        wgu[0, 0] = np.float16(1.0)
        wgu[1, H] = np.float16(1.0)
        wd = np.zeros((H, D), dtype=np.float16)
        wd[0, 0] = np.float16(1.0)
        arrays.update(run_case("boundary", h, r, wgu, wd))
        arrays["boundary_gate_target"] = gates
    finally:
        try: lib.cdll.rocket_close(fd)
        except Exception: pass
    np.savez_compressed(args.out, **arrays)
    print(json.dumps({"out": args.out, "arrays": sorted(arrays)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
