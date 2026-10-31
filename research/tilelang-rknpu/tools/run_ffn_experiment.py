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
             32767.0)
_EDGE = 512.0 * STEP
_TAIL_SLOPE = 0.447


def lut_sigmoid(x):
    v = x.astype(np.float64)
    q = np.interp(v, _GRID, _Q)
    pos = v > _EDGE
    neg = v < -_EDGE
    q = np.where(pos, _Q[-1] + _TAIL_SLOPE * (v - _EDGE), q)
    q = np.where(neg, _Q[0] - _TAIL_SLOPE * (-v - _EDGE), q)
    q = np.clip(q, 1.0, 32767.0)
    return f16((q + 1.0) / 32768.0)


def r0_host(h, r, wgu, wd):
    z = kchain(h, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = np.float32(1.0) / (np.float32(1.0) +
                              np.exp(-gate.astype(np.float32)))
    act = f16(gate.astype(np.float32) * sig * up.astype(np.float32))
    down = kchain(act, wd)
    return f16(down + r), z, None, None, act, down


def r1_fp16_points(h, r, wgu, wd):
    return ffn_oracle(h, r, wgu, wd,
                      lambda z: 1.0 / (1.0 + np.exp(-z.astype(np.float64))))


def r2_declared_lut(h, r, wgu, wd):
    return ffn_oracle(h, r, wgu, wd, lut_sigmoid)


def ffn_oracle(h, r, wgu, wd, sigmoid):
    z = kchain(h, wgu)
    gate = z[:, :H]
    up = z[:, H:]
    sig = f16(sigmoid(gate))
    silu = f16(gate * sig)
    act = f16(silu * up)
    down = kchain(act, wd)
    return f16(down + r), z, sig, silu, act, down


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
        for layer in args.layers_list:
            h = np.asarray(cap[f"layer.{layer}.post_norm"],
                           dtype=np.float16)
            r = np.asarray(cap[f"layer.{layer}.resid_attn"],
                           dtype=np.float16)
            if h.shape != r.shape:
                raise SystemExit(f"layer {layer}: H/R capture shape mismatch")
            if args.m > h.shape[0]:
                hp = np.zeros((args.m, D), dtype=np.float16)
                rp = np.zeros((args.m, D), dtype=np.float16)
                hp[:h.shape[0]] = h
                rp[:r.shape[0]] = r
                h, r = hp, rp
            elif args.m == 1:
                h = np.ascontiguousarray(h[:1])
                r = np.ascontiguousarray(r[:1])
            else:
                h = np.ascontiguousarray(h[:args.m])
                r = np.ascontiguousarray(r[:args.m])
            arrays[f"h{layer}"] = h
            arrays[f"r{layer}"] = r
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
        "m": args.m, "d": D, "h": H, "layers": list(args.layers_list),
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
    policy_path = out.parent / "module-acceptance-policy.json"
    if policy_path.is_file():
        manifest["policy"] = {"path": policy_path.name,
                              "sha256": sha(policy_path)}
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


K_ULP = 2.0
NRMSE_MAX = 1e-3


def _check(out, ref):
    shape_ok = tuple(out.shape) == tuple(ref.shape)
    fin = bool(np.isfinite(out).all()) if shape_ok else False
    if not shape_ok:
        return {"shape_ok": False, "finite": fin, "bitwise_equal": False,
                "numeric_pass": False, "max_abs": None, "rmse": None,
                "nrmse": None, "ref_absmax": None,
                "f16_spacing": None, "threshold_abs": None}
    ref32 = ref.astype(np.float32)
    out32 = out.astype(np.float32)
    diff = out32 - ref32
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_absmax = float(np.max(np.abs(ref32))) if ref32.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref32 * ref32))) if ref32.size else 0.0
    nrmse = rmse / ref_rms if ref_rms > 0.0 else (0.0 if rmse == 0 else float("inf"))
    spacing = float(np.spacing(np.float16(ref_absmax))) if ref_absmax > 0 else 0.0
    threshold = K_ULP * spacing
    ok = bool(fin and max_abs <= threshold and nrmse <= NRMSE_MAX)
    return {"shape_ok": True, "finite": fin,
            "bitwise_equal": bool(np.array_equal(out, ref)),
            "numeric_pass": ok, "max_abs": max_abs, "rmse": rmse,
            "nrmse": nrmse, "ref_absmax": ref_absmax,
            "f16_spacing": spacing, "threshold_abs": threshold}


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
            h = data[f"h{layer}"]
            r = data[f"r{layer}"]
            wgu = data[f"wgu{layer}"]
            wd = data[f"wd{layer}"]
            r0, r0_z, _, _, r0_a, r0_p = r0_host(h, r, wgu, wd)
            r1 = r1_fp16_points(h, r, wgu, wd)[0]
            r2 = r2_declared_lut(h, r, wgu, wd)[0]
            # Group A: production Host semantics at two NPU projection
            # boundaries. It is not a scheduling baseline.
            p0 = load_plan(str(plans_dir / man["A"]["gate_up"]["plan"]))
            p1 = load_plan(str(plans_dir / man["A"]["down"]["plan"]))
            c0 = lib.create_plan(p0); c1 = lib.create_plan(p1)
            arrs0, b0 = consts(p0, {"B": wgu})
            arrs1, b1 = consts(p1, {"B": wd})
            ws0 = lib.create_workspace(c0, b0, fd=fd)
            ws1 = lib.create_workspace(c1, b1, fd=fd)
            try:
                in0s = [b for b in p0.bindings if b.role == "input"]
                out0s = [b for b in p0.bindings if b.role == "output"]
                in1s = [b for b in p1.bindings if b.role == "input"]
                out1s = [b for b in p1.bindings if b.role == "output"]
                if len(in0s) != 1 or len(out0s) != 1 or                         len(in1s) != 1 or len(out1s) != 1:
                    raise SystemExit("A plan must expose exactly A and Y")
                in0, out0 = in0s[0].id, out0s[0].id
                in1, out1 = in1s[0].id, out1s[0].id
                a_times, a_checks = [], []
                for index in range(args.warmup + args.repeats):
                    z = np.zeros((m, 2 * H), dtype=np.float16)
                    p = np.zeros((m, D), dtype=np.float16)
                    y = np.zeros((m, D), dtype=np.float16)
                    t0 = time.perf_counter()
                    ws0.run([make_binding(in0, h)], [make_binding(out0, z)])
                    gate = z[:, :H]; up = z[:, H:]
                    sig32 = 1.0 / (1.0 + np.exp(-gate.astype(np.float64)))
                    act = f16(gate.astype(np.float64) * sig32 *
                              up.astype(np.float64))
                    ws1.run([make_binding(in1, act)], [make_binding(out1, p)])
                    y[...] = f16(p + r)
                    elapsed = (time.perf_counter() - t0) * 1e6
                    check = {
                        "y": _check(y, r0),
                        "z": _check(z, r0_z),
                        "down": _check(p, r0_p),
                        "activation": _check(act, r0_a),
                    }
                    a_checks.append(check)
                    if index >= args.warmup:
                        a_times.append(elapsed)
                a_valid = all(all(v["numeric_pass"] for v in c.values())
                              for c in a_checks)
            finally:
                ws0.destroy(); ws1.destroy(); c0.destroy(); c1.destroy()
            rows.append({
                "name": "A_host_swiglu", "layer": layer,
                "median_us": statistics.median(a_times),
                "timed_us": a_times,
                "warmup_checks": a_checks[:args.warmup],
                "timed_checks": a_checks[args.warmup:],
                "check_r0": a_checks[-1]["y"],
                "candidate_valid": bool(a_valid),
            })
            for cand in man["candidates"]:
                if args.candidates and cand["name"] not in args.candidates:
                    continue
                plan = load_plan(str(plans_dir / cand["plan"]))
                cp = lib.create_plan(plan)
                arrs, binds = consts(plan, {"Wgu": wgu, "Wd": wd})
                ws = lib.create_workspace(cp, binds, fd=fd)
                try:
                    inputs = {b.name: b.id for b in plan.bindings
                              if b.role == "input"}
                    outputs = [b.id for b in plan.bindings
                               if b.role == "output"]
                    if set(inputs) != {"H", "R"} or len(outputs) != 1:
                        raise SystemExit("FFN plan bindings must be H/R -> Y")
                    out_id = outputs[0]
                    times, checks = [], []
                    for index in range(args.warmup + args.repeats):
                        y = np.zeros((m, D), dtype=np.float16)
                        t0 = time.perf_counter()
                        rep = ws.run([
                            make_binding(inputs["H"], h),
                            make_binding(inputs["R"], r),
                        ], [make_binding(out_id, y)])
                        elapsed = (time.perf_counter() - t0) * 1e6
                        checks.append({
                            "y_vs_r2": _check(y, r2),
                            "y_vs_r1": _check(y, r1),
                            "y_vs_r0": _check(y, r0),
                        })
                        if index >= args.warmup:
                            times.append(elapsed)
                    valid = all(c["y_vs_r2"]["numeric_pass"] for c in checks)
                    rows.append({
                        "name": cand["name"], "layer": layer,
                        "median_us": statistics.median(times),
                        "timed_us": times,
                        "warmup_checks": checks[:args.warmup],
                        "timed_checks": checks[args.warmup:],
                        "check_vs_r2": checks[-1]["y_vs_r2"],
                        "check_vs_r1": checks[-1]["y_vs_r1"],
                        "check_vs_r0": checks[-1]["y_vs_r0"],
                        "candidate_valid": bool(valid),
                        "task_count": rep.task_count,
                        "submit_count": rep.submit_count,
                        "plan_hash": plan.plan_hash,
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
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi, make_binding
    root = Path(args.plans).resolve()
    man = json.loads((root / "manifest.json").read_text())
    data = np.load(root / man["inputs"], allow_pickle=False)
    by = {c["name"]: c for c in man["candidates"]}
    specs = {"base": by[args.baseline], "cand": by[args.candidate]}
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
    layers = []
    try:
        for layer in man["layers"]:
            h = data[f"h{layer}"]
            r = data[f"r{layer}"]
            wgu = data[f"wgu{layer}"]
            wd = data[f"wd{layer}"]
            declared = r2_declared_lut(h, r, wgu, wd)[0]
            handles = {}
            for key, spec in specs.items():
                plan = load_plan(str(root / spec["plan"]))
                arrs, cp, ws = _runtime_handles(lib, plan, {"Wgu": wgu, "Wd": wd}, fd)
                ins = {v.name: v.id for v in plan.bindings if v.role == "input"}
                outs = [v.id for v in plan.bindings if v.role == "output"]
                if set(ins) != {"H", "R"} or len(outs) != 1:
                    raise SystemExit("FFN plan bindings must be H/R -> Y")
                handles[key] = (plan, cp, ws, arrs, ins, outs[0])
            try:
                def invoke(key):
                    plan, cp, ws, arrs, ins, out = handles[key]
                    y = np.zeros((man["m"], D), dtype=np.float16)
                    t0 = time.perf_counter()
                    rep = ws.run([
                        make_binding(ins["H"], h),
                        make_binding(ins["R"], r),
                    ], [make_binding(out, y)])
                    return (time.perf_counter() - t0) * 1e6, _check(y, declared), rep
                warm = {"base": [], "cand": []}
                for _ in range(args.warmup):
                    for key in ("base", "cand"):
                        _, check, _ = invoke(key)
                        warm[key].append(check)
                rounds, base_us, cand_us, ratios = [], [], [], []
                for index in range(args.rounds):
                    order = ("base", "cand") if index % 2 == 0 else ("cand", "base")
                    row = {"layer": layer, "round": index, "order": list(order)}
                    measured = {}
                    for key in order:
                        us, check, rep = invoke(key)
                        measured[key] = us
                        row[key] = {"us": us, "check": check,
                                    "task_count": rep.task_count,
                                    "submit_count": rep.submit_count}
                    rounds.append(row)
                    base_us.append(measured["base"])
                    cand_us.append(measured["cand"])
                    ratios.append(measured["base"] / measured["cand"])
                layers.append({
                    "layer": layer,
                    "warmup_checks": warm,
                    "rounds": rounds,
                    "baseline_median_us": statistics.median(base_us),
                    "candidate_median_us": statistics.median(cand_us),
                    "paired_median_ratio": statistics.median(ratios),
                    "ratios": ratios,
                    "warmup_valid": all(
                        c["numeric_pass"] for k in warm for c in warm[k]),
                    "timed_valid": all(
                        row[k]["check"]["numeric_pass"]
                        for row in rounds for k in ("base", "cand")),
                })
            finally:
                for _, cp, ws, _, _, _ in handles.values():
                    ws.destroy(); cp.destroy()
    finally:
        try: lib.cdll.rocket_close(fd)
        except Exception: pass
    per_layer = [x["paired_median_ratio"] for x in layers]
    confirmed = bool(all(x["warmup_valid"] and x["timed_valid"] for x in layers)
                     and all(r >= args.threshold for r in per_layer))
    result = {"schema_version": 1, "kind": "ffn-confirmation",
              "baseline": args.baseline, "candidate": args.candidate,
              "layers": layers,
              "per_layer_paired_median_ratio": per_layer,
              "threshold": args.threshold, "confirmed": confirmed,
              "warmup": args.warmup, "rounds": args.rounds}
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"per_layer": per_layer, "confirmed": confirmed}, indent=1))
    return 0 if confirmed else 1


def replay(args):
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi, make_binding
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
            h = data[f"h{layer}"]
            r = data[f"r{layer}"]
            wgu = data[f"wgu{layer}"]
            wd = data[f"wd{layer}"]
            declared = r2_declared_lut(h, r, wgu, wd)[0]
            plan = load_plan(str(root / cand["plan"]))
            arrs, cp, ws = _runtime_handles(lib, plan, {"Wgu": wgu, "Wd": wd}, fd)
            try:
                ins = {v.name: v.id for v in plan.bindings if v.role == "input"}
                outs = [v.id for v in plan.bindings if v.role == "output"]
                if set(ins) != {"H", "R"} or len(outs) != 1:
                    raise SystemExit("FFN plan bindings must be H/R -> Y")
                y = np.zeros((man["m"], D), dtype=np.float16)
                ws.run([
                    make_binding(ins["H"], h),
                    make_binding(ins["R"], r),
                ], [make_binding(outs[0], y)])
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
    p.add_argument("--layers", default="0,15,29")
    p.add_argument("--out", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--plans", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--candidates", default="")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run)
    p = sub.add_parser("confirm")
    p.add_argument("--plans", required=True)
    p.add_argument("--baseline", default="B_fixed_192")
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
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
    if hasattr(args, "layers"):
        args.layers_list = [int(x) for x in args.layers.split(",") if x]
    if getattr(args, "candidates", ""):
        args.candidates = {x.strip() for x in args.candidates.split(",") if x.strip()}
    else:
        args.candidates = set()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
