#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""A/B/C/D task-order and CBUF-reuse experiment for a real SmolLM gate/up GEMM."""
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

D, H = 576, 3072


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
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


def _check(out, ref):
    if tuple(out.shape) != tuple(ref.shape):
        return {"ok": False, "bitwise": False, "max_abs": None,
                "nrmse": None}
    d = out.astype(np.float32) - ref.astype(np.float32)
    rms = float(np.sqrt(np.mean(ref.astype(np.float32) ** 2)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return {"ok": bool(np.isfinite(out).all()),
            "bitwise": bool(np.array_equal(out, ref)),
            "max_abs": float(np.max(np.abs(d))),
            "nrmse": rmse / rms if rms else 0.0,
            "ref_absmax": float(np.max(np.abs(ref.astype(np.float32))))}


def prepare(args):
    from rknpu_backend import CompileOptions, compile_tilelang, default_profile, load_plan, save_plan
    from models.tilelang_matmul import make_matmul_prim
    pkg = Path(args.package).resolve()
    cap_path = Path(args.capture).resolve()
    out = Path(args.out).resolve()
    plans = out / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    pm = json.loads((pkg / "manifest.json").read_text())
    weights = np.load(pkg / pm["weights"]["path"], allow_pickle=False)
    with np.load(cap_path, allow_pickle=False) as cap:
        arrays = {}
        for layer in [int(x) for x in args.layers.split(",") if x]:
            h = np.asarray(cap[f"layer.{layer}.post_norm"], dtype=np.float16)
            if args.m > h.shape[0]:
                hp = np.zeros((args.m, D), dtype=np.float16)
                hp[:h.shape[0]] = h
                h = hp
            elif args.m == 1:
                h = np.ascontiguousarray(h[:1])
            else:
                h = np.ascontiguousarray(h[:args.m])
            arrays[f"h{layer}"] = h
            arrays[f"w{layer}"] = np.ascontiguousarray(
                np.asarray(weights[f"layer.{layer}.gate_up"]).T,
                dtype=np.float16)
    np.savez(plans / "inputs.npz", **arrays)
    prof = default_profile()
    manifest = {
        "schema_version": 1, "kind": "smollm-gateup-task-order",
        "m": args.m, "k": D, "n": H, "layers": args.layers_list,
        "capture": str(cap_path), "capture_sha256": sha(cap_path),
        "package_manifest_sha256": sha(pkg / "manifest.json"),
        "weights_sha256": pm["weights"]["file_sha256"],
        "inputs": "inputs.npz", "inputs_sha256": sha(plans / "inputs.npz"),
        "plans": {},
    }
    for name, policy in (("P", "production_package"), ("B", "merged"),
                         ("C", "interleaved"), ("D", "interleaved_reuse")):
        if name == "P":
            key = ("layer.0.gate_up.prefill" if args.m == 256
                   else "layer.0.gate_up.decode")
            src = pkg / pm["plans"][key]["path"]
            plan = load_plan(str(src))
        else:
            opt = CompileOptions(numeric_mode="rk_f16_tiled",
                                 linear_task_order=policy)
            plan = compile_tilelang(
                make_matmul_prim(args.m, D, H),
                profile=prof, options=opt).plan
        path = plans / f"{name}.rkplan.json"
        save_plan(plan, str(path))
        manifest["plans"][name] = {
            "policy": policy, "plan": path.name, "sha256": sha(path),
            "plan_hash": plan.plan_hash,
            "task_count": plan.accounting.task_count,
            "submit_count": plan.accounting.submit_count,
            "data_bo_count": plan.accounting.data_bo_count,
            "data_reuse_tasks": sum(
                1 for a in plan.actions
                if a.kind == "npu_task" and a.attrs.get("data_reuse")),
            "reuse_tasks": [a.id for a in plan.actions
                            if a.kind == "npu_task" and
                            a.attrs.get("data_reuse")],
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"plans": manifest["plans"]}, indent=1))
    return 0


def _runtime(lib, plan, w, fd):
    from rknpu_backend.runtime.ffi import make_binding
    arrs, binds = [], []
    for c in plan.constants:
        arr = np.eye(c.shape[0], dtype=np.float16) if c.generator == "identity" else w
        arrs.append(arr)
        binds.append(make_binding(c.binding_id, arr))
    cp = lib.create_plan(plan)
    ws = lib.create_workspace(cp, binds, fd=fd)
    return arrs, cp, ws


def run(args):
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi, make_binding
    root = Path(args.plans).resolve()
    man = json.loads((root / "manifest.json").read_text())
    data = np.load(root / man["inputs"], allow_pickle=False)
    if not args.production_cache:
        # Mechanism-isolation control: A is 12 submits, B/C/D are one submit.
        # The creation-time program/command caches only apply to single-submit
        # plans, so disable all three caches for every arm.
        for key in ("RK_PLAN_PROGRAM_CACHE", "RK_PLAN_COMMAND_CACHE",
                    "RK_PLAN_MULTI_SUBMIT_CACHE"):
            os.environ[key] = "0"
    lib = LibRKPlan(os.environ["RKNPU_PLAN_LIB"])
    if args.mock:
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit("rocket_open failed")
    policies = ["P", "B", "C", "D"]
    if args.only:
        policies = [x.strip() for x in args.only.split(",") if x.strip()]
    rows = []
    try:
        for layer in man["layers"]:
            h = data[f"h{layer}"]
            w = data[f"w{layer}"]
            ref = kchain(h, w)
            handles = {}
            for name in policies:
                spec = man["plans"][name]
                plan = load_plan(str(root / spec["plan"]))
                arrs, cp, ws = _runtime(lib, plan, w, fd)
                ins = {b.name: b.id for b in plan.bindings if b.role == "input"}
                outs = [b.id for b in plan.bindings if b.role == "output"]
                if set(ins) != {"A"} or len(outs) != 1:
                    raise SystemExit("GEMM plan must expose A -> Y")
                cache = lib.cache_info(ws)
                cbuf = []
                reuse_tasks = 0
                for action in plan.actions:
                    if action.kind != "npu_task":
                        continue
                    info, values = lib.task_cbuf(ws, action.id)
                    reuse_tasks += info.data_reuse_tasks
                    cbuf.append({"action_id": action.id,
                                 "task_count": info.task_count,
                                 "cbuf_writes": info.cbuf_writes,
                                 "data_reuse_tasks": info.data_reuse_tasks,
                                 "weight_reuse_tasks": info.weight_reuse_tasks,
                                 "first_reuse_task": info.first_reuse_task,
                                 "first_cbuf_con0": info.first_cbuf_con0,
                                 "values": values[:4]})
                handles[name] = (plan, cp, ws, arrs, ins["A"], outs[0],
                                 cache, reuse_tasks, cbuf)
            try:
                def invoke(name):
                    plan, cp, ws, arrs, inp, out, _, _, _ = handles[name]
                    y = np.zeros((man["m"], man["n"]), dtype=np.float16)
                    t0 = time.perf_counter()
                    rep = ws.run([make_binding(inp, h)],
                                 [make_binding(out, y)])
                    return (time.perf_counter() - t0) * 1e6, _check(y, ref), rep, y
                warm = {name: [] for name in policies}
                for _ in range(args.warmup):
                    for name in policies:
                        _, chk, _, _ = invoke(name)
                        warm[name].append(chk)
                a_reference = None
                times = {name: [] for name in policies}
                checks = {name: [] for name in policies}
                round_rows = []
                order = policies
                for r in range(args.rounds):
                    if r % 2:
                        order = list(reversed(policies))
                    order = order[r % len(policies):] + order[:r % len(policies)]
                    row = {"layer": layer, "round": r, "order": order}
                    for name in order:
                        us, chk, rep, y = invoke(name)
                        times[name].append(us)
                        checks[name].append(chk)
                        if name == policies[0] and a_reference is None:
                            a_reference = y.copy()
                        row[name] = {"us": us, "check": chk,
                                     "bitwise_vs_A": bool(np.array_equal(y, a_reference))
                                     if a_reference is not None else None,
                                     "task_count": rep.task_count,
                                     "submit_count": rep.submit_count,
                                     "pack_us": rep.pack_us,
                                     "emit_us": rep.emit_us,
                                     "submit_wait_us": rep.submit_wait_us,
                                     "output_sync_us": rep.output_sync_us,
                                     "upload_calls": rep.upload_calls,
                                     "upload_bytes": rep.upload_bytes,
                                     "download_calls": rep.download_calls,
                                     "download_bytes": rep.download_bytes}
                    round_rows.append(row)
                for name in policies:
                    spec = man["plans"][name]
                    _, _, _, _, _, _, cache, reuse_tasks, cbuf = handles[name]
                    rows.append({
                        "name": name, "layer": layer,
                        "policy": spec["policy"],
                        "median_us": statistics.median(times[name]),
                        "timed_us": times[name],
                        "warmup": warm[name],
                        "checks": checks[name],
                        "warmup_valid": all(c["ok"] for c in warm[name]),
                        "candidate_valid": all(c["ok"] for c in checks[name]),
                        "bitwise_vs_reference": all(c["bitwise"] for c in checks[name]),
                        "bitwise_vs_A": all(r.get(name, {}).get("bitwise_vs_A") is True
                                            for r in round_rows),
                        "task_count": spec["task_count"],
                        "submit_count": spec["submit_count"],
                        "data_reuse_tasks": spec["data_reuse_tasks"],
                        "register_data_reuse_tasks": reuse_tasks,
                        "cache": {k: int(getattr(cache, k)) for k in
                                  ("nsubmits", "program_cache_requested",
                                   "program_cache_ready", "command_cache_requested",
                                   "command_cache_ready", "multi_submit_requested",
                                   "multi_submit_ready", "multi_submit_cached",
                                   "program_cache_submit", "program_cache_tasks",
                                   "command_plan_present", "submit_cache_present",
                                   "host_bytes", "device_bytes",
                                   "skipped_budget", "skipped_static")},
                        "cbuf_decoded": cbuf,
                        "plan_hash": spec["plan_hash"],
                    })
                rows.append({"name": "rotation", "layer": layer,
                             "rounds": round_rows})
            finally:
                for _, cp, ws, _, _, _, _, _, _ in handles.values():
                    ws.destroy(); cp.destroy()
    finally:
        try: lib.cdll.rocket_close(fd)
        except Exception: pass
    result = {"schema_version": 1, "kind": "smollm-gateup-task-order-run",
              "lib_sha256": sha(Path(os.environ["RKNPU_PLAN_LIB"])),
              "manifest_sha256": sha(root / "manifest.json"),
              "cache_control": {
                  "program_cache": os.environ.get("RK_PLAN_PROGRAM_CACHE", "default"),
                  "command_cache": os.environ.get("RK_PLAN_COMMAND_CACHE", "default"),
                  "multi_submit_cache": os.environ.get("RK_PLAN_MULTI_SUBMIT_CACHE", "default"),
                  "production_cache": bool(args.production_cache)},
              "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows)}, indent=1))
    return 0


def main(argv=None):
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
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--only", default="")
    p.add_argument("--production-cache", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run)
    args = ap.parse_args(argv)
    if hasattr(args, "layers"):
        args.layers_list = [int(x) for x in args.layers.split(",") if x]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
