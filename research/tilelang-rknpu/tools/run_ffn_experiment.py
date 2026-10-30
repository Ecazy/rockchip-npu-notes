#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduled SmolLM-shaped SwiGLU FFN experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

D, H = 576, 1536
LAYERS = (0, 15, 29)
STEP = 32.0 / 2596.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f16(x):
    return np.asarray(x, dtype=np.float16)


def kchain(x, w):
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    acc = f16(a[:, :256] @ b[:256, :])
    for k0 in range(256, a.shape[1], 256):
        k1 = min(a.shape[1], k0 + 256)
        acc = f16(acc.astype(np.float64) + a[:, k0:k1] @ b[k0:k1, :])
    return acc


_GRID = np.arange(-512, 513, dtype=np.float64) * STEP
_Q = np.clip(np.rint(1.0 / (1.0 + np.exp(-_GRID)) * 32768.0), 1.0,
             32767.0) / 32768.0


def lut_sigmoid(x):
    return f16(np.interp(x.astype(np.float64), _GRID, _Q))


def ffn_oracle(x, wgu, wd, sigmoid):
    z = kchain(x, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = f16(sigmoid(gate))
    silu = f16(gate * sig)
    act = f16(silu * up)
    down = kchain(act, wd)
    return f16(down + x), z, sig, silu, act, down


def load_fixture():
    path = ROOT / "tests" / "fixtures" / "tilelang_ffn_scheduled.py"
    spec = importlib.util.spec_from_file_location("ffn_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_ffn


def candidate_specs():
    base = {"name": "B_fixed_192", "stage0": "m_major",
            "ew": "m_major", "down_order": "m_major", "tn": 192}
    specs = [base]
    for name, s0, ew, down, tn in (
            ("C_s0n_n192", "n_major", "m_major", "m_major", 192),
            ("C_ewn_n192", "m_major", "n_major", "n_major", 192),
            ("C_s0n_ewn_n192", "n_major", "n_major", "n_major", 192),
            ("C_tn64_m", "m_major", "m_major", "m_major", 64),
            ("C_tn64_n", "m_major", "m_major", "n_major", 64),
            ("C_s0n_tn64_n", "n_major", "m_major", "n_major", 64)):
        specs.append({"name": name, "stage0": s0, "ew": ew,
                      "down_order": down, "tn": tn})
    return specs


def compile_plans(m: int):
    from rknpu_backend import CompileOptions, compile_tilelang, default_profile, save_plan
    from models.tilelang_matmul import make_matmul_tiled_tail_prim
    make_ffn = load_fixture()
    prof = default_profile()
    opt = CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=True)
    plans = {}
    for spec in candidate_specs():
        prim = make_ffn(
            m=m, d=D, h=H, bm=m, bn=256, tn=spec["tn"],
            order0=spec["stage0"], order1=spec["ew"],
            order2=spec["ew"], order3=spec["ew"],
            order4=spec["down_order"], order5=spec["down_order"])
        res = compile_tilelang(prim, profile=prof, options=opt)
        plans[spec["name"]] = res.plan
    a0 = compile_tilelang(
        make_matmul_tiled_tail_prim(m, D, 2 * H, bm=m, bn=256),
        profile=prof, options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    a1 = compile_tilelang(
        make_matmul_tiled_tail_prim(m, H, D, bm=m, bn=192),
        profile=prof, options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
    return plans, a0, a1, prof, opt


def prepare(args) -> int:
    from rknpu_backend import save_plan
    package = Path(args.package).resolve()
    capture = Path(args.capture).resolve()
    out = Path(args.out).resolve()
    plans_dir = out / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    pm = json.loads((package / "manifest.json").read_text())
    weights = np.load(package / pm["weights"]["path"], allow_pickle=False)
    with np.load(capture, allow_pickle=False) as cap:
        arrays = {}
        for layer in LAYERS:
            act = np.asarray(cap[f"layer.{layer}.post_norm"],
                             dtype=np.float16)
            if args.m > act.shape[0]:
                padded = np.zeros((args.m, D), dtype=np.float16)
                padded[:act.shape[0]] = act
                x = padded
            elif args.m == 1:
                x = np.ascontiguousarray(act[:1])
            else:
                x = np.ascontiguousarray(act[:args.m])
            arrays[f"x{layer}"] = x
            arrays[f"wgu{layer}"] = np.ascontiguousarray(
                np.asarray(weights[f"layer.{layer}.gate_up"]).T,
                dtype=np.float16)
            arrays[f"wd{layer}"] = np.ascontiguousarray(
                np.asarray(weights[f"layer.{layer}.down"]).T,
                dtype=np.float16)
    np.savez(plans_dir / "inputs.npz", **arrays)
    plans, a0, a1, prof, opt = compile_plans(args.m)
    save_plan(a0, str(plans_dir / "A_gate_up.rkplan.json"))
    save_plan(a1, str(plans_dir / "A_down.rkplan.json"))
    manifest = {
        "schema_version": 1, "kind": "smollm-ffn-experiment",
        "m": args.m, "d": D, "h": H, "layers": list(LAYERS),
        "capture": str(capture), "capture_sha256": sha(capture),
        "package_manifest_sha256": sha(package / "manifest.json"),
        "package_weights_sha256": pm["weights"]["file_sha256"],
        "inputs": "inputs.npz", "inputs_sha256": sha(plans_dir / "inputs.npz"),
        "A": {
            "gate_up": {"plan": "A_gate_up.rkplan.json",
                        "plan_sha256": sha(plans_dir / "A_gate_up.rkplan.json"),
                        "plan_hash": a0.plan_hash},
            "down": {"plan": "A_down.rkplan.json",
                     "plan_sha256": sha(plans_dir / "A_down.rkplan.json"),
                     "plan_hash": a1.plan_hash},
            "note": "NPU gate/up -> Host SwiGLU -> NPU down -> Host residual",
        },
        "candidates": [],
    }
    from rknpu_backend.runtime.ffi import LibRKPlan
    validator = LibRKPlan(os.environ["RKNPU_PLAN_MOCK_LIB"]) \
        if os.environ.get("RKNPU_PLAN_MOCK_LIB") else None
    spec_by = {s["name"]: s for s in candidate_specs()}
    for name, plan in plans.items():
        spec = spec_by[name]
        path = plans_dir / f"{name}.rkplan.json"
        save_plan(plan, str(path))
        c_ok, c_err = True, None
        if validator is not None:
            cp = ws = None
            arrs = []
            try:
                cp = validator.create_plan(plan)
                bindings = []
                for c in plan.constants:
                    arr = np.eye(c.shape[0], dtype=np.float16) \
                        if c.generator == "identity" else np.zeros(c.shape,
                                                                  np.float16)
                    arrs.append(arr)
                    bindings.append(__import__("rknpu_backend").runtime.ffi.make_binding(c.binding_id, arr))
                ws = validator.create_workspace(cp, bindings, fd=0)
            except Exception as exc:
                c_ok, c_err = False, f"{type(exc).__name__}: {exc}"
            finally:
                if ws is not None:
                    ws.destroy()
                if cp is not None:
                    cp.destroy()
        manifest["candidates"].append({
            "name": spec["name"], "requested": spec,
            "plan": path.name, "plan_sha256": sha(path),
            "plan_hash": plan.plan_hash,
            "task_count": plan.accounting.task_count,
            "submit_count": plan.accounting.submit_count,
            "data_bo_count": plan.accounting.data_bo_count,
            "publish": [[p.dst_row, p.dst_col, p.rows, p.cols]
                        for p in plan.publish],
            "c_verifier_pass": c_ok, "c_verifier_error": c_err,
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1,
                                                  sort_keys=True) + "\n")
    print(json.dumps({"candidates": len(plans),
                      "manifest": str(out / "manifest.json")}, indent=1))
    return 0


def _check(out, ref):
    shape_ok = tuple(out.shape) == tuple(ref.shape)
    fin = bool(np.isfinite(out).all()) if shape_ok else False
    eq = bool(np.array_equal(out, ref)) if shape_ok else False
    ok = bool(np.allclose(out, ref, atol=5e-1, rtol=1e-3)) \
        if shape_ok and fin else False
    mx = (float(np.max(np.abs(out.astype(np.float32) -
                            ref.astype(np.float32)))) if shape_ok else None)
    return {"shape_ok": shape_ok, "finite": fin, "bitwise_equal": eq,
            "numeric_pass": ok, "max_abs": mx}


def run(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans_dir = Path(args.plans).resolve()
    man = json.loads((plans_dir / "manifest.json").read_text())
    m = int(man["m"])
    data = np.load(plans_dir / man["inputs"], allow_pickle=False)
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit("rocket_open failed")

    def consts(plan, vals):
        prepared, bindings = [], []
        for c in plan.constants:
            if c.generator == "identity":
                arr = np.eye(c.shape[0], dtype=np.float16)
            else:
                arr = vals[c.name]
            prepared.append(arr)
            bindings.append(make_binding(c.binding_id, arr))
        return prepared, bindings

    rows = []
    try:
        for layer in man["layers"]:
            x = data[f"x{layer}"]
            wgu = data[f"wgu{layer}"]
            wd = data[f"wd{layer}"]
            exact, _, _, _, _, _ = ffn_oracle(x, wgu, wd,
                                              lambda z: 1.0 / (1.0 + np.exp(-z.astype(np.float64))))
            declared, _, _, _, _, _ = ffn_oracle(x, wgu, wd, lut_sigmoid)
            # A: two independent NPU plans plus Host SwiGLU and residual.
            p0 = load_plan(str(plans_dir / man["A"]["gate_up"]["plan"]))
            p1 = load_plan(str(plans_dir / man["A"]["down"]["plan"]))
            c0 = lib.create_plan(p0); c1 = lib.create_plan(p1)
            arrs0, b0 = consts(p0, {"B": wgu})
            arrs1, b1 = consts(p1, {"B": wd})
            ws0 = lib.create_workspace(c0, b0, fd=fd)
            ws1 = lib.create_workspace(c1, b1, fd=fd)
            try:
                in0 = next(b.id for b in p0.bindings if b.role == "input")
                out0 = next(b.id for b in p0.bindings if b.role == "output")
                in1 = next(b.id for b in p1.bindings if b.role == "input")
                out1 = next(b.id for b in p1.bindings if b.role == "output")
                ta = []
                for _ in range(args.warmup + args.repeats):
                    z = np.zeros((m, 2 * H), dtype=np.float16)
                    p = np.zeros((m, D), dtype=np.float16)
                    y = np.zeros((m, D), dtype=np.float16)
                    t0 = time.perf_counter()
                    ws0.run([make_binding(in0, x)], [make_binding(out0, z)])
                    gate = z[:, :H]; up = z[:, H:]
                    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
                    act = f16(f16(gate * sig) * up)
                    ws1.run([make_binding(in1, act)], [make_binding(out1, p)])
                    y[...] = f16(p + x)
                    ta.append((time.perf_counter() - t0) * 1e6)
                a_check = _check(y, exact)
            finally:
                ws0.destroy(); ws1.destroy(); c0.destroy(); c1.destroy()
            a_row = {"name": "A_host_swiglu", "layer": layer,
                     "median_us": statistics.median(ta),
                     "times_us": ta, "check": a_check,
                     "candidate_valid": a_check["numeric_pass"]}
            rows.append(a_row)
            for cand in man["candidates"]:
                plan = load_plan(str(plans_dir / cand["plan"]))
                cp = lib.create_plan(plan)
                arrs, binds = consts(plan, {"Wgu": wgu, "Wd": wd})
                ws = lib.create_workspace(cp, binds, fd=fd)
                try:
                    in_id = next(b.id for b in plan.bindings
                                 if b.role == "input")
                    out_id = next(b.id for b in plan.bindings
                                  if b.role == "output")
                    times = []
                    checks = []
                    for _ in range(args.warmup + args.repeats):
                        y = np.zeros((m, D), dtype=np.float16)
                        t0 = time.perf_counter()
                        rep = ws.run([make_binding(in_id, x)],
                                     [make_binding(out_id, y)])
                        times.append((time.perf_counter() - t0) * 1e6)
                        checks.append(_check(y, declared))
                    rows.append({
                        "name": cand["name"], "layer": layer,
                        "median_us": statistics.median(times),
                        "times_us": times,
                        "check_vs_declared": checks[-1],
                        "check_vs_exact": _check(y, exact),
                        "candidate_valid": bool(
                            all(c["numeric_pass"] for c in checks)),
                        "task_count": rep.task_count,
                        "submit_count": rep.submit_count,
                        "plan_hash": plan.plan_hash,
                        "A_C_max_abs": _check(y, exact)["max_abs"],
                    })
                finally:
                    ws.destroy(); cp.destroy()
    finally:
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass
    out = {
        "schema_version": 1, "kind": "smollm-ffn-run",
        "lib_sha256": sha(Path(os.environ["RKNPU_PLAN_LIB"])),
        "manifest_sha256": sha(plans_dir / "manifest.json"),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows)}, indent=1))
    return 0



def _runtime_handles(lib, plan, vals, fd):
    prepared, bindings = [], []
    for c in plan.constants:
        arr = np.eye(c.shape[0], dtype=np.float16) if c.generator == "identity" else vals[c.name]
        prepared.append(arr)
        bindings.append(__import__("rknpu_backend").runtime.ffi.make_binding(c.binding_id, arr))
    cp = lib.create_plan(plan)
    ws = lib.create_workspace(cp, bindings, fd=fd)
    return prepared, cp, ws


def confirm(args):
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    root = Path(args.plans).resolve()
    man = json.loads((root / "manifest.json").read_text())
    data = np.load(root / man["inputs"], allow_pickle=False)
    by = {c["name"]: c for c in man["candidates"]}
    a, b = by[args.baseline], by[args.candidate]
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
    rounds, ta, tb = [], [], []
    try:
        for layer in man["layers"]:
            x = data[f"x{layer}"]; wgu = data[f"wgu{layer}"]; wd = data[f"wd{layer}"]
            declared = ffn_oracle(x, wgu, wd, lut_sigmoid)[0]
            handles = {}
            for spec in (a, b):
                plan = load_plan(str(root / spec["plan"]))
                arrs, cp, ws = _runtime_handles(lib, plan, {"Wgu": wgu, "Wd": wd}, fd)
                handles[spec["name"]] = (plan, cp, ws, arrs)
            try:
                for r in range(args.rounds):
                    order = (a, b) if r % 2 == 0 else (b, a)
                    row = {"layer": layer, "round": r,
                           "order": [s["name"] for s in order]}
                    for spec in order:
                        plan, cp, ws, arrs = handles[spec["name"]]
                        inp = next(v.id for v in plan.bindings if v.role == "input")
                        out = next(v.id for v in plan.bindings if v.role == "output")
                        y = np.zeros((man["m"], D), dtype=np.float16)
                        t0 = time.perf_counter()
                        ws.run([__import__("rknpu_backend").runtime.ffi.make_binding(inp, x)], [__import__("rknpu_backend").runtime.ffi.make_binding(out, y)])
                        dt = (time.perf_counter() - t0) * 1e6
                        row[spec["name"]] = {"us": dt, "check": _check(y, declared)}
                        (ta if spec is a else tb).append(dt)
                    rounds.append(row)
            finally:
                for _, cp, ws, _ in handles.values():
                    ws.destroy(); cp.destroy()
    finally:
        try: lib.cdll.rocket_close(fd)
        except Exception: pass
    result = {"schema_version": 1, "kind": "ffn-confirmation",
              "baseline": args.baseline, "candidate": args.candidate,
              "rounds": rounds,
              "baseline_median_us": statistics.median(ta),
              "candidate_median_us": statistics.median(tb),
              "speedup": statistics.median(ta) / statistics.median(tb),
              "threshold": args.threshold,
              "confirmed": bool(statistics.median(ta) / statistics.median(tb) >= args.threshold)}
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"speedup": result["speedup"], "confirmed": result["confirmed"]}, indent=1))
    return 0 if result["confirmed"] else 1


def replay(args):
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    root = Path(args.plans).resolve(); man = json.loads((root / "manifest.json").read_text())
    data = np.load(root / man["inputs"], allow_pickle=False)
    cand = next(c for c in man["candidates"] if c["name"] == args.candidate)
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll); fd = int(lib.cdll.rocket_open())
    rows = []
    try:
        for layer in man["layers"]:
            x = data[f"x{layer}"]; wgu = data[f"wgu{layer}"]; wd = data[f"wd{layer}"]
            declared = ffn_oracle(x, wgu, wd, lut_sigmoid)[0]
            plan = load_plan(str(root / cand["plan"]))
            arrs, cp, ws = _runtime_handles(lib, plan, {"Wgu": wgu, "Wd": wd}, fd)
            try:
                inp = next(v.id for v in plan.bindings if v.role == "input")
                out = next(v.id for v in plan.bindings if v.role == "output")
                y = np.zeros((man["m"], D), dtype=np.float16)
                ws.run([__import__("rknpu_backend").runtime.ffi.make_binding(inp, x)], [__import__("rknpu_backend").runtime.ffi.make_binding(out, y)])
                rows.append({"layer": layer, "check": _check(y, declared)})
            finally:
                ws.destroy(); cp.destroy()
    finally:
        try: lib.cdll.rocket_close(fd)
        except Exception: pass
    result = {"schema_version": 1, "kind": "ffn-replay", "candidate": args.candidate,
              "plan_hash": cand["plan_hash"], "plan_sha256": cand["plan_sha256"],
              "rows": rows, "pass": all(r["check"]["numeric_pass"] for r in rows)}
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"]}, indent=1))
    return 0 if result["pass"] else 1

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--package", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--plans", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run)
    p = sub.add_parser("confirm")
    p.add_argument("--plans", required=True)
    p.add_argument("--baseline", default="B_fixed_192")
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--threshold", type=float, default=1.02)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=confirm)
    p = sub.add_parser("replay")
    p.add_argument("--plans", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=replay)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
