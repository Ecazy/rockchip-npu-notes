#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile and execute minimal M/N-scheduled GEMM variants.

This is intentionally a schedule experiment, not a scheduler: K blocking and
accumulation remain the existing 256-wide ordered FP16 contract.
"""
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


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def parse_schedule(text: str) -> dict:
    parts = text.split(",")
    if len(parts) != 4:
        raise SystemExit("schedule must be name,tile_m,tile_n,order")
    name, tm, tn, order = parts
    return {"name": name, "tile_m": int(tm), "tile_n": int(tn),
            "order": order}


def compile_schedules(args) -> int:
    from rknpu_backend import (CompileOptions, compile_rkir, compile_tilelang,
                               default_profile, matmul_module, save_plan)
    from models.tilelang_matmul import (make_matmul_prim,
                                         make_matmul_prim_scheduled,
                                         make_matmul_tiled_prim)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    options = CompileOptions(numeric_mode="rk_f16_tiled")
    profile = default_profile()
    rows = []
    for spec in [parse_schedule(x) for x in args.schedule]:
        if args.source == "tiled":
            prim = make_matmul_tiled_prim(
                args.m, args.k, args.n, bm=spec["tile_m"],
                bn=spec["tile_n"], order=spec["order"])
        elif spec["name"] == "default":
            prim = make_matmul_prim(args.m, args.k, args.n)
        else:
            prim = make_matmul_prim_scheduled(
                args.m, args.k, args.n, tile_m=spec["tile_m"],
                tile_n=spec["tile_n"], order=spec["order"])
        result = compile_tilelang(prim, profile=profile, options=options,
                                  b_is_input=bool(args.b_is_input))
        plan = result.plan
        plan_path = out / f"{spec['name']}.rkplan.json"
        save_plan(plan, str(plan_path))
        first = next(a for a in plan.actions
                     if a.template_id == "co.mm.f16.v1")
        schedule = first.attrs.get("source_schedule")
        applied = first.attrs.get("applied_schedule")
        rows.append({
            "name": spec["name"],
            "requested": (None if spec["name"] == "default" else spec),
            "plan": plan_path.name,
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "plan_hash": plan.plan_hash,
            "task_count": plan.accounting.task_count,
            "submit_count": plan.accounting.submit_count,
            "buffer_count": len(plan.buffers),
            "data_bo_count": plan.accounting.data_bo_count,
            "publish_count": len(plan.publish),
            "source_schedule": schedule,
            "applied_schedule": applied,
            "source_mn_tiling": first.attrs.get("source_mn_tiling"),
            "source_k_loop": first.attrs.get("source_k_loop"),
            "publish_order": [[p.dst_row, p.dst_col, p.rows, p.cols]
                              for p in plan.publish],
        })
        print(json.dumps({k: rows[-1][k] for k in
                          ("name", "plan_hash", "task_count", "submit_count",
                           "buffer_count", "data_bo_count", "publish_count",
                           "source_schedule")}, sort_keys=True))
        if getattr(args, "direct_rkir", False) and not args.b_is_input:
            module = matmul_module(m=args.m, k=args.k, n=args.n)
            if spec["name"] != "default":
                attrs = dict(module.ops[0].attrs)
                attrs["source_schedule"] = {
                    "kind": "mn_tiles_v1",
                    "tile_m": spec["tile_m"], "tile_n": spec["tile_n"],
                    "order": spec["order"], "k_tile": 256,
                    "accumulation": "ordered_ascending_f16_writeback",
                }
                module = module.__class__(
                    name=module.name, values=module.values,
                    ops=(module.ops[0].__class__(
                        module.ops[0].id, module.ops[0].opcode,
                        module.ops[0].inputs, module.ops[0].outputs, attrs,
                        module.ops[0].source, module.ops[0].side_effects),),
                    inputs=module.inputs, outputs=module.outputs,
                    source_hash=module.source_hash,
                    state_signature=module.state_signature)
            direct = compile_rkir(module, profile=profile, options=options)
            direct_plan = direct.plan
            direct_path = out / f"direct_{spec['name']}.rkplan.json"
            save_plan(direct_plan, str(direct_path))
            first_direct = next(a for a in direct_plan.actions
                                if a.template_id == "co.mm.f16.v1")
            rows.append({
                "name": "direct_" + spec["name"],
                "requested": (None if spec["name"] == "default" else spec),
                "path_kind": "direct_rkir",
                "plan": direct_path.name,
                "plan_sha256": hashlib.sha256(
                    direct_path.read_bytes()).hexdigest(),
                "plan_hash": direct_plan.plan_hash,
                "task_count": direct_plan.accounting.task_count,
                "submit_count": direct_plan.accounting.submit_count,
                "buffer_count": len(direct_plan.buffers),
                "data_bo_count": direct_plan.accounting.data_bo_count,
                "publish_count": len(direct_plan.publish),
                "source_schedule": first_direct.attrs.get("source_schedule"),
                "applied_schedule": first_direct.attrs.get("applied_schedule"),
                "source_mn_tiling": first_direct.attrs.get("source_mn_tiling"),
                "source_k_loop": first_direct.attrs.get("source_k_loop"),
                "publish_order": [[p.dst_row, p.dst_col, p.rows, p.cols]
                                  for p in direct_plan.publish],
            })
    write_json(out / "manifest.json", {
        "schema_version": 1,
        "kind": "smollm2-schedule-experiment",
        "geometry": {"m": args.m, "k": args.k, "n": args.n},
        "source": args.source,
        "b_is_input": bool(args.b_is_input),
        "k_contract": {"k_tile": 256,
                       "accumulation": "ordered_ascending_f16_writeback"},
        "runs": rows,
    })
    return 0


def reference_mm_tiled(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    f16 = np.float16
    a = np.asarray(a, dtype=f16)
    b = np.asarray(b, dtype=f16)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=f16)
    for k0 in range(0, a.shape[1], 256):
        k1 = min(a.shape[1], k0 + 256)
        part = a[:, k0:k1].astype(np.float32) @ b[k0:k1, :].astype(np.float32)
        out = np.asarray(out.astype(np.float32) + part, dtype=f16)
    return out


def run_plans(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans_dir = Path(args.plans).resolve()
    manifest = json.loads((plans_dir / "manifest.json").read_text())
    backend = args.backend
    if backend == "board":
        lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
        if not lib_path:
            raise SystemExit("RKNPU_PLAN_LIB is required for board mode")
        lib = LibRKPlan(lib_path)
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
    else:
        lib_path = os.environ.get("RKNPU_PLAN_MOCK_LIB", "")
        if not lib_path:
            raise SystemExit("RKNPU_PLAN_MOCK_LIB is required for mock mode")
        lib = LibRKPlan(lib_path)
        fd = 0
    rng = np.random.default_rng(12345)
    geometry = manifest["geometry"]
    m, k, n = geometry["m"], geometry["k"], geometry["n"]
    b_is_input = bool(manifest.get("b_is_input", False))
    a = np.asarray(rng.normal(0, 0.2, (m, k)), dtype=np.float16)
    b = np.asarray(rng.normal(0, 0.05, (k, n)), dtype=np.float16)
    expected = reference_mm_tiled(a, b)
    rows = []
    baseline_output = None
    try:
        for item in manifest["runs"]:
            plan = load_plan(str(plans_dir / item["plan"]))
            cplan = lib.create_plan(plan)
            try:
                const_specs = list(plan.constants)
                constants = []
                if const_specs:
                    if len(const_specs) != 1 or const_specs[0].name != "B":
                        raise SystemExit(
                            "schedule experiment expects at most one B constant")
                    if tuple(const_specs[0].shape) != (k, n):
                        raise SystemExit("plan B constant shape differs from geometry")
                    constants = [make_binding(const_specs[0].binding_id, b)]
                input_by_name = {x.name: x for x in plan.bindings
                                 if x.role == "input"}
                if b_is_input:
                    if "B" not in input_by_name or "A" not in input_by_name:
                        raise SystemExit("b-is-input plan must expose A and B inputs")
                elif set(input_by_name) != {"A"}:
                    raise SystemExit("constant-B plan must expose exactly A input")
                ws = lib.create_workspace(cplan, constants, fd=fd)
                try:
                    out_spec = next(x for x in plan.bindings
                                    if x.role == "output")
                    out = np.zeros(tuple(out_spec.shape), dtype=np.float16)
                    input_arrays = []
                    for name, spec in input_by_name.items():
                        if name == "A":
                            input_arrays.append(make_binding(spec.id, a))
                        elif name == "B":
                            input_arrays.append(make_binding(spec.id, b))
                        else:
                            raise SystemExit(f"unexpected input binding {name!r}")
                    times = []
                    for _ in range(args.warmup):
                        out.fill(0)
                        ws.run(input_arrays, [make_binding(out_spec.id, out)])
                    first = out.copy()
                    for _ in range(args.repeats):
                        out.fill(0)
                        t0 = time.perf_counter()
                        report = ws.run(input_arrays,
                                        [make_binding(out_spec.id, out)])
                        times.append((time.perf_counter() - t0) * 1e6)
                    reference_equal = bool(np.array_equal(out, expected))
                    reference_max_abs = float(np.max(np.abs(
                        out.astype(np.float32) - expected.astype(np.float32))))
                    reference_rmse = float(np.sqrt(np.mean(np.square(
                        out.astype(np.float32) - expected.astype(np.float32)))))
                    reference_pass = bool(np.allclose(
                        out, expected, atol=8e-3, rtol=8e-3))
                    if baseline_output is None:
                        baseline_output = out.copy()
                    schedule_equal = bool(np.array_equal(out, baseline_output))
                    rows.append({
                        "name": item["name"],
                        "plan_hash": plan.plan_hash,
                        "task_count": plan.accounting.task_count,
                        "submit_count": plan.accounting.submit_count,
                        "buffer_count": len(plan.buffers),
                        "data_bo_count": plan.accounting.data_bo_count,
                        "publish_count": len(plan.publish),
                        "source_schedule": item["source_schedule"],
                        "applied_schedule": item["applied_schedule"],
                        "publish_order": item["publish_order"],
                        "output_sha256": sha_array(out),
                        "expected_sha256": sha_array(expected),
                        "schedule_equal_to_baseline": schedule_equal,
                        "reference_equal": reference_equal,
                        "reference_pass": reference_pass,
                        "reference_max_abs": reference_max_abs,
                        "reference_rmse": reference_rmse,
                        "first_run_sha256": sha_array(first),
                        "times_us": times,
                        "median_us": float(statistics.median(times)),
                        "min_us": float(min(times)),
                        "mean_us": float(statistics.mean(times)),
                        "upload_calls": int(report.upload_calls),
                        "download_calls": int(report.download_calls),
                        "task_report_count": int(report.task_count),
                        "submit_report_count": int(report.submit_count),
                    })
                finally:
                    ws.destroy()
            finally:
                cplan.destroy()
    finally:
        if backend == "board":
            try:
                lib.cdll.rocket_close(fd)
            except Exception:
                pass
    passed = all(r["schedule_equal_to_baseline"] and r["reference_pass"]
                 for r in rows)
    result = {
        "schema_version": 1,
        "kind": "smollm2-schedule-execution",
        "backend": backend,
        "geometry": geometry,
        "lib_path": lib_path,
        "lib_sha256": hashlib.sha256(Path(lib_path).read_bytes()).hexdigest(),
        "environment": {
            "machine": os.uname().machine,
            "kernel": os.uname().release,
            "core_mask": os.environ.get("RKNPU_CORE_MASK", "unset"),
        },
        "schedule_equal_all": all(
            r["schedule_equal_to_baseline"] for r in rows),
        "reference_pass_all": all(r["reference_pass"] for r in rows),
        "pass": passed,
        "runs": rows,
    }
    write_json(Path(args.out), result)
    print(json.dumps({k: result[k] for k in
                      ("backend", "geometry", "schedule_equal_all",
                       "reference_pass_all", "pass")},
                     indent=1, sort_keys=True))
    return 0 if passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compile")
    p.add_argument("--out", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--b-is-input", action="store_true")
    p.add_argument("--direct-rkir", action="store_true",
                   help="also compile equivalent direct-RKIR plans")
    p.add_argument("--source", choices=("full", "tiled"), default="full",
                   help="full-source schedule annotation or a real TileLang "
                        "block-grid source")
    p.add_argument("--schedule", action="append", required=True,
                   help="name,tile_m,tile_n,order; use name=default for "
                        "the old planner default")
    p.set_defaults(func=compile_schedules)
    p = sub.add_parser("run")
    p.add_argument("--plans", required=True)
    p.add_argument("--backend", choices=("board", "mock"), required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_plans)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
