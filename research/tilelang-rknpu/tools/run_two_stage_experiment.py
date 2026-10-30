#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restricted two-stage schedule experiment: X->W1->Z->W2->Y."""
from __future__ import annotations

import argparse
import hashlib
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

M, K, H, N = 65, 576, 576, 97


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
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


def oracle(x, w1, w2):
    z = kchain(x, w1)
    y = kchain(z, w2)
    return z, y


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def candidate_specs():
    specs = [
        {"name": "A_fixed_256_m", "group": "A",
         "stage0": [256, 256, "m_major"],
         "stage1": [256, 256, "m_major"]},
        {"name": "B_manual_256_m_s1_256x128_n", "group": "B",
         "stage0": [256, 256, "m_major"],
         "stage1": [256, 128, "n_major"]},
    ]
    for tm in (256, 128, 64):
        for tn in (256, 128, 64, 192):
            for order in ("m_major", "n_major"):
                specs.append({
                    "name": f"C_m{tm}_s1_n{tn}_{order[0]}",
                    "group": "C",
                    "stage0": [tm, 256, "m_major"],
                    "stage1": [tm, tn, order],
                })
    return specs


def load_two_stage_fixture():
    import importlib.util
    path = ROOT / "tests" / "fixtures" / "tilelang_two_stage.py"
    spec = importlib.util.spec_from_file_location("two_stage_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_two_stage


def compile_candidate(spec):
    from rknpu_backend import CompileOptions, compile_tilelang, default_profile
    profile = default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled")
    tm0, tn0, order0 = spec["stage0"]
    tm1, tn1, order1 = spec["stage1"]
    prim = load_two_stage_fixture()(
        m=M, k=K, h=H, n=N,
        bm0=tm0, bn0=tn0, bm1=tm1, bn1=tn1,
        order0=order0, order1=order1)
    result = compile_tilelang(prim, profile=profile, options=options)
    return result, profile


def prepare(args) -> int:
    from rknpu_backend import save_plan
    from rknpu_backend.runtime.ffi import LibRKPlan, make_binding

    out = Path(args.out).resolve()
    plans = out / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261030)
    x = f16(rng.normal(0, 0.2, (M, K)))
    w1 = f16(rng.normal(0, 0.1, (K, H)))
    w2 = f16(rng.normal(0, 0.1, (H, N)))
    np.savez(plans / "inputs.npz", x=x, w1=w1, w2=w2)
    validator = None
    mock_lib = os.environ.get("RKNPU_PLAN_MOCK_LIB", "")
    if mock_lib:
        validator = LibRKPlan(mock_lib)

    manifest = {
        "schema_version": 1,
        "kind": "two-stage-schedule-manifest",
        "geometry": {"m": M, "k": K, "h": H, "n": N},
        "inputs": "inputs.npz",
        "inputs_sha256": sha(plans / "inputs.npz"),
        "candidates": [],
    }
    for spec in candidate_specs():
        try:
            result, _ = compile_candidate(spec)
            plan = result.plan
            path = plans / f"{spec['name']}.rkplan.json"
            save_plan(plan, str(path))
            first0 = next(a for a in plan.actions
                          if a.kind == "npu_task" and
                          a.attrs.get("source_stage_index") == 0)
            first1 = next(a for a in plan.actions
                          if a.kind == "npu_task" and
                          a.attrs.get("source_stage_index") == 1)
            c_ok, c_error = True, None
            cplan = ws = None
            if validator is not None:
                try:
                    cplan = validator.create_plan(plan)
                    zero_arrays = [np.zeros(tuple(c.shape), dtype=np.float16)
                                   for c in plan.constants]
                    constants = [
                        make_binding(c.binding_id, array)
                        for c, array in zip(plan.constants, zero_arrays)]
                    ws = validator.create_workspace(cplan, constants, fd=0)
                except Exception as exc:
                    c_ok, c_error = False, f"{type(exc).__name__}: {exc}"
                finally:
                    if ws is not None:
                        ws.destroy()
                    if cplan is not None:
                        cplan.destroy()
            manifest["candidates"].append({
                "name": spec["name"], "group": spec["group"],
                "requested": spec,
                "plan": path.name, "plan_sha256": sha(path),
                "plan_hash": plan.plan_hash,
                "task_count": plan.accounting.task_count,
                "submit_count": plan.accounting.submit_count,
                "data_bo_count": plan.accounting.data_bo_count,
                "publish": [[p.dst_row, p.dst_col, p.rows, p.cols]
                            for p in plan.publish],
                "stage0_source_schedule": first0.attrs.get("source_schedule"),
                "stage0_applied_schedule": first0.attrs.get("applied_schedule"),
                "stage1_source_schedule": first1.attrs.get("source_schedule"),
                "stage1_applied_schedule": first1.attrs.get("applied_schedule"),
                "source_k_rounding": first0.attrs.get("source_k_rounding"),
                "c_verifier_pass": c_ok,
                "c_verifier_error": c_error,
            })
        except Exception as exc:
            manifest["candidates"].append({
                "name": spec["name"], "group": spec["group"],
                "requested": spec,
                "rejected_reason": f"{type(exc).__name__}: {exc}",
                "c_verifier_pass": False,
            })

    # D: two independent single-GEMM plans.  These are only an overhead control;
    # the board runner performs the Host Z transfer explicitly between calls.
    from models.tilelang_matmul import make_matmul_tiled_tail_prim
    from rknpu_backend import compile_tilelang, default_profile
    profile = default_profile()
    d0 = compile_tilelang(
        make_matmul_tiled_tail_prim(M, K, H, bm=256, bn=256),
        profile=profile, options=__import__("rknpu_backend").CompileOptions(
            numeric_mode="rk_f16_tiled")).plan
    d1 = compile_tilelang(
        make_matmul_tiled_tail_prim(M, H, N, bm=256, bn=256),
        profile=profile, options=__import__("rknpu_backend").CompileOptions(
            numeric_mode="rk_f16_tiled")).plan
    (plans / "D_stage0.rkplan.json").write_text(
        __import__("json").dumps(d0.to_dict(), indent=1, sort_keys=True) + "\n")
    (plans / "D_stage1.rkplan.json").write_text(
        __import__("json").dumps(d1.to_dict(), indent=1, sort_keys=True) + "\n")
    manifest["independent"] = {
        "stage0": {"plan": "D_stage0.rkplan.json",
                   "plan_sha256": sha(plans / "D_stage0.rkplan.json"),
                   "plan_hash": d0.plan_hash,
                   "task_count": d0.accounting.task_count},
        "stage1": {"plan": "D_stage1.rkplan.json",
                   "plan_sha256": sha(plans / "D_stage1.rkplan.json"),
                   "plan_hash": d1.plan_hash,
                   "task_count": d1.accounting.task_count},
        "note": "Host downloads Z, then uploads it as the second plan's A input.",
    }
    manifest["valid_candidates"] = sum(
        1 for c in manifest["candidates"] if c.get("c_verifier_pass"))
    write_json(out / "manifest.json", manifest)
    print(json.dumps({
        "candidates": len(manifest["candidates"]),
        "valid": manifest["valid_candidates"],
        "manifest": str(out / "manifest.json"),
    }, indent=1))
    return 0


def _load_inputs(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return f16(z["x"]), f16(z["w1"]), f16(z["w2"])


def _check(out: np.ndarray, ref: np.ndarray) -> dict:
    shape_ok = tuple(out.shape) == tuple(ref.shape)
    finite = bool(np.isfinite(out).all()) if shape_ok else False
    bitwise = bool(np.array_equal(out, ref)) if shape_ok else False
    numeric = bool(np.allclose(out, ref, atol=8e-3, rtol=8e-3)) \
        if shape_ok and finite else False
    max_abs = (float(np.max(np.abs(out.astype(np.float32) -
                            ref.astype(np.float32))))
               if shape_ok else None)
    return {"shape_ok": shape_ok, "finite": finite,
            "bitwise_equal": bitwise, "numeric_pass": numeric,
            "max_abs": max_abs}


def run(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans = Path(args.plans).resolve()
    manifest = json.loads((plans / "manifest.json").read_text())
    x, w1, w2 = _load_inputs(plans / manifest["inputs"])
    z_ref, y_ref = oracle(x, w1, w2)
    lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
    if not lib_path:
        raise SystemExit("RKNPU_PLAN_LIB is required")
    lib = LibRKPlan(lib_path)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise SystemExit(f"rocket_open failed: {fd}")
    rows = []
    try:
        for cand in manifest["candidates"]:
            if not cand.get("c_verifier_pass"):
                continue
            plan = load_plan(str(plans / cand["plan"]))
            if plan.plan_hash != cand["plan_hash"]:
                raise SystemExit(f"{cand['name']}: plan hash changed")
            cplan = lib.create_plan(plan)
            try:
                constants = []
                cmap = {"W1": w1, "W2": w2}
                for spec in plan.constants:
                    constants.append(make_binding(
                        spec.binding_id, cmap[spec.name]))
                ws = lib.create_workspace(cplan, constants, fd=fd)
                try:
                    in_id = next(b.id for b in plan.bindings
                                 if b.role == "input")
                    out_id = next(b.id for b in plan.bindings
                                  if b.role == "output")
                    out = np.zeros((M, N), dtype=np.float16)
                    times = []
                    checks = []
                    for repeat in range(args.warmup + args.repeats):
                        out.fill(0)
                        t0 = time.perf_counter()
                        report = ws.run([make_binding(in_id, x)],
                                        [make_binding(out_id, out)])
                        elapsed = (time.perf_counter() - t0) * 1e6
                        if repeat >= args.warmup:
                            times.append(elapsed)
                            checks.append(_check(out, y_ref))
                    history = np.zeros((M, N), dtype=np.float16)
                    x2 = f16(x.astype(np.float32) * 0.75 + 0.125)
                    report2 = ws.run([make_binding(in_id, x2)],
                                     [make_binding(out_id, history)])
                    _, y_ref2 = oracle(x2, w1, w2)
                    history_check = _check(history, y_ref2)
                    changed = not np.array_equal(out, history)
                    valid = bool(all(c["numeric_pass"] for c in checks) and
                                 history_check["numeric_pass"] and changed)
                    rows.append({
                        "name": cand["name"], "group": cand["group"],
                        "plan_hash": plan.plan_hash,
                        "task_count": report.task_count,
                        "submit_count": report.submit_count,
                        "times_us": times,
                        "median_us": float(statistics.median(times)),
                        "min_us": float(min(times)),
                        "checks": checks,
                        "history_check": history_check,
                        "history_input_changed_output": changed,
                        "candidate_valid": valid,
                        "stage0_applied_schedule":
                            cand.get("stage0_applied_schedule"),
                        "stage1_applied_schedule":
                            cand.get("stage1_applied_schedule"),
                    })
                finally:
                    ws.destroy()
            finally:
                cplan.destroy()
        # D independent plan calls, including the Host Z round trip.
        d0 = load_plan(str(plans / manifest["independent"]["stage0"]["plan"]))
        d1 = load_plan(str(plans / manifest["independent"]["stage1"]["plan"]))
        cp0 = lib.create_plan(d0)
        cp1 = lib.create_plan(d1)
        ws0 = ws1 = None
        try:
            ws0 = lib.create_workspace(
                cp0, [make_binding(c.binding_id, {"B": w1}[c.name])
                      for c in d0.constants], fd=fd)
            ws1 = lib.create_workspace(
                cp1, [make_binding(c.binding_id, {"B": w2}[c.name])
                      for c in d1.constants], fd=fd)
            in0 = next(b.id for b in d0.bindings if b.role == "input")
            out0 = next(b.id for b in d0.bindings if b.role == "output")
            in1 = next(b.id for b in d1.bindings if b.role == "input")
            out1 = next(b.id for b in d1.bindings if b.role == "output")
            times = []
            checks = []
            for _ in range(args.repeats):
                z_mid = np.zeros((M, H), dtype=np.float16)
                y = np.zeros((M, N), dtype=np.float16)
                t0 = time.perf_counter()
                r0 = ws0.run([make_binding(in0, x)],
                             [make_binding(out0, z_mid)])
                r1 = ws1.run([make_binding(in1, z_mid)],
                             [make_binding(out1, y)])
                times.append((time.perf_counter() - t0) * 1e6)
                checks.append({"z": _check(z_mid, z_ref),
                               "y": _check(y, y_ref)})
            rows.append({
                "name": "D_two_independent_plans", "group": "D",
                "note": "two ws.run calls; Z crosses the Host boundary between them",
                "task_count": r0.task_count + r1.task_count,
                "submit_count": r0.submit_count + r1.submit_count,
                "times_us": times,
                "median_us": float(statistics.median(times)),
                "min_us": float(min(times)),
                "checks": checks,
                "candidate_valid": bool(all(
                    c["z"]["numeric_pass"] and c["y"]["numeric_pass"]
                    for c in checks)),
            })
        finally:
            if ws0 is not None:
                ws0.destroy()
            if ws1 is not None:
                ws1.destroy()
            cp0.destroy()
            cp1.destroy()
    finally:
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass
    result = {
        "schema_version": 1,
        "kind": "two-stage-schedule-run",
        "lib_sha256": sha(Path(lib_path)),
        "manifest_sha256": sha(plans / "manifest.json"),
        "rows": rows,
        "pass": bool(rows) and all(r["candidate_valid"] for r in rows),
    }
    write_json(Path(args.out), result)
    print(json.dumps({"pass": result["pass"], "rows": len(rows)}, indent=1))
    return 0 if result["pass"] else 1


def confirm(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans = Path(args.plans).resolve()
    manifest = json.loads((plans / "manifest.json").read_text())
    by_name = {c["name"]: c for c in manifest["candidates"]}
    baseline = by_name.get(args.baseline)
    winner = by_name.get(args.candidate)
    if baseline is None or winner is None:
        raise SystemExit("confirmation candidate is missing")
    x, w1, w2 = _load_inputs(plans / manifest["inputs"])
    _, y_ref = oracle(x, w1, w2)
    lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
    lib = LibRKPlan(lib_path)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise SystemExit(f"rocket_open failed: {fd}")

    def make(cand):
        plan = load_plan(str(plans / cand["plan"]))
        if plan.plan_hash != cand["plan_hash"]:
            raise SystemExit(f"{cand['name']}: plan hash changed")
        cplan = lib.create_plan(plan)
        ws = lib.create_workspace(
            cplan, [make_binding(c.binding_id, {"W1": w1, "W2": w2}[c.name])
                    for c in plan.constants], fd=fd)
        in_id = next(b.id for b in plan.bindings if b.role == "input")
        out_id = next(b.id for b in plan.bindings if b.role == "output")
        return plan, cplan, ws, in_id, out_id

    handles = []
    try:
        base = make(baseline)
        win = make(winner)
        handles = [base, win]
        base_times, win_times, rounds = [], [], []
        all_valid = True
        for round_index in range(args.rounds):
            order = ((baseline, base), (winner, win)) if round_index % 2 == 0 \
                else ((winner, win), (baseline, base))
            row = {"round": round_index, "order": [x[0]["name"] for x in order]}
            for cand, handle in order:
                _, _, ws, in_id, out_id = handle
                out = np.zeros((M, N), dtype=np.float16)
                t0 = time.perf_counter()
                report = ws.run([make_binding(in_id, x)],
                                [make_binding(out_id, out)])
                elapsed = (time.perf_counter() - t0) * 1e6
                check = _check(out, y_ref)
                all_valid = all_valid and check["numeric_pass"]
                row[cand["name"]] = {"us": elapsed, "check": check,
                                      "task_count": report.task_count}
                if cand is baseline:
                    base_times.append(elapsed)
                else:
                    win_times.append(elapsed)
            rounds.append(row)
        speedup = statistics.median(base_times) / statistics.median(win_times)
        result = {
            "schema_version": 1,
            "kind": "two-stage-schedule-confirmation",
            "baseline": args.baseline,
            "candidate": args.candidate,
            "rounds": rounds,
            "baseline_median_us": statistics.median(base_times),
            "winner_median_us": statistics.median(win_times),
            "speedup": speedup,
            "threshold_speedup": args.threshold,
            "all_valid": all_valid,
            "confirmed": bool(all_valid and speedup >= args.threshold),
        }
        write_json(Path(args.out), result)
        print(json.dumps({"speedup": speedup,
                          "confirmed": result["confirmed"]}, indent=1))
        return 0 if result["confirmed"] else 1
    finally:
        for _, cplan, ws, _, _ in handles:
            ws.destroy()
            cplan.destroy()
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass


def replay(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans = Path(args.plans).resolve()
    manifest = json.loads((plans / "manifest.json").read_text())
    cand = next((c for c in manifest["candidates"]
                 if c["name"] == args.candidate), None)
    if cand is None:
        raise SystemExit(f"unknown candidate {args.candidate!r}")
    x, w1, w2 = _load_inputs(plans / manifest["inputs"])
    _, y_ref = oracle(x, w1, w2)
    lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
    lib = LibRKPlan(lib_path)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise SystemExit(f"rocket_open failed: {fd}")
    try:
        plan = load_plan(str(plans / cand["plan"]))
        if plan.plan_hash != cand["plan_hash"]:
            raise SystemExit("replay plan hash changed")
        cplan = lib.create_plan(plan)
        try:
            ws = lib.create_workspace(
                cplan, [make_binding(c.binding_id, {"W1": w1, "W2": w2}[c.name])
                        for c in plan.constants], fd=fd)
            try:
                in_id = next(b.id for b in plan.bindings if b.role == "input")
                out_id = next(b.id for b in plan.bindings if b.role == "output")
                y = np.zeros((M, N), dtype=np.float16)
                report = ws.run([make_binding(in_id, x)],
                                [make_binding(out_id, y)])
                check = _check(y, y_ref)
            finally:
                ws.destroy()
        finally:
            cplan.destroy()
    finally:
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass
    result = {
        "schema_version": 1,
        "kind": "two-stage-schedule-replay",
        "candidate": args.candidate,
        "plan_hash": plan.plan_hash,
        "plan_sha256": cand["plan_sha256"],
        "check": check,
        "pass": bool(check["numeric_pass"]),
        "task_count": report.task_count,
        "submit_count": report.submit_count,
    }
    write_json(Path(args.out), result)
    print(json.dumps({"pass": result["pass"]}, indent=1))
    return 0 if result["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--plans", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run)
    p = sub.add_parser("confirm")
    p.add_argument("--plans", required=True)
    p.add_argument("--baseline", default="A_fixed_256_m")
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--threshold", type=float, default=1.02)
    p.add_argument("--out", required=True)
    p.set_defaults(func=confirm)
    p = sub.add_parser("replay")
    p.add_argument("--plans", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=replay)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
