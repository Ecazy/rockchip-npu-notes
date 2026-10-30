#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile, board-measure, select and replay M/N TileLang schedules.

The search space is intentionally finite and tied to the implemented schedule
contract: only M/N tile size and traversal order are candidates.  K stays at the
existing 256-wide ordered FP16-writeback contract.  Every candidate is a real
TileLang PrimFunc generated from ``models.tilelang_matmul``.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from run_schedule_experiment import reference_mm_tiled, sha_array  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def parse_candidate(text: str) -> dict:
    parts = text.split(",")
    if len(parts) != 4:
        raise SystemExit("candidate must be name,tile_m,tile_n,order")
    return {"name": parts[0], "tile_m": int(parts[1]),
            "tile_n": int(parts[2]), "order": parts[3]}


def base_candidates(m: int, kind: str) -> list[dict]:
    if kind == "decode":
        return [
            {"name": "default", "tile_m": 256, "tile_n": 256,
             "order": "m_major", "default": True},
            {"name": "m256_n128_m", "tile_m": 256, "tile_n": 128,
             "order": "m_major"},
            {"name": "m256_n128_n", "tile_m": 256, "tile_n": 128,
             "order": "n_major"},
            {"name": "m256_n64_m", "tile_m": 256, "tile_n": 64,
             "order": "m_major"},
            {"name": "m1_n256_m", "tile_m": 1, "tile_n": 256,
             "order": "m_major"},
            {"name": "m1_n128_n", "tile_m": 1, "tile_n": 128,
             "order": "n_major"},
        ]
    return [
        {"name": "default", "tile_m": 256, "tile_n": 256,
         "order": "m_major", "default": True},
        {"name": "m16_n128_m", "tile_m": 16, "tile_n": 128,
         "order": "m_major"},
        {"name": "m16_n128_n", "tile_m": 16, "tile_n": 128,
         "order": "n_major"},
        {"name": "m32_n256_m", "tile_m": 32, "tile_n": 256,
         "order": "m_major"},
        {"name": "m32_n128_n", "tile_m": 32, "tile_n": 128,
         "order": "n_major"},
        {"name": "m64_n128_m", "tile_m": 64, "tile_n": 128,
         "order": "m_major"},
        {"name": "m128_n128_m", "tile_m": 128, "tile_n": 128,
         "order": "m_major"},
        {"name": "m128_n128_n", "tile_m": 128, "tile_n": 128,
         "order": "n_major"},
        {"name": "m256_n128_n", "tile_m": 256, "tile_n": 128,
         "order": "n_major"},
        {"name": "m256_n64_m", "tile_m": 256, "tile_n": 64,
         "order": "m_major"},
        {"name": "m44_n192_m", "tile_m": 44, "tile_n": 192,
         "order": "m_major"},
        {"name": "m44_n192_n", "tile_m": 44, "tile_n": 192,
         "order": "n_major"},
        {"name": "m44_n96_m", "tile_m": 44, "tile_n": 96,
         "order": "m_major"},
        {"name": "m4_n192_m", "tile_m": 4, "tile_n": 192,
         "order": "m_major"},
        {"name": "m4_n192_n", "tile_m": 4, "tile_n": 192,
         "order": "n_major"},
        {"name": "m11_n192_m", "tile_m": 11, "tile_n": 192,
         "order": "m_major"},
    ]


def case_specs(capture: Path) -> list[dict]:
    """Cases use captured backend activations and fixed model weight keys."""
    return [
        {"id": "prefill_qkv", "kind": "prefill",
         "activation_key": "layer.0.input_norm",
         "weight_key": "layer.0.qkv", "k": 576, "n": 960},
        {"id": "prefill_gate_up", "kind": "prefill",
         "activation_key": "layer.0.post_norm",
         "weight_key": "layer.0.gate_up", "k": 576, "n": 3072},
        {"id": "prefill_down", "kind": "prefill",
         "activation_key": "layer.0.swiglu",
         "weight_key": "layer.0.down", "k": 1536, "n": 576},
        {"id": "decode_qkv", "kind": "decode",
         "activation_key": "layer.0.input_norm",
         "activation_row": 0,
         "weight_key": "layer.0.qkv", "k": 576, "n": 960},
        {"id": "lm_head_shard0", "kind": "prefill",
         "activation_key": "final_norm",
         "weight_key": "embed", "weight_shard": [0, 12],
         "k": 576, "n": 4096},
        {"id": "attention_qk_prefill", "kind": "prefill",
         "qk_construction": "prefill", "group": 0,
         "capacity": 256, "heads_per_group": 3,
         "k": 64, "n": 256, "b_is_input": True},
        {"id": "attention_qk_decode", "kind": "decode",
         "qk_construction": "decode", "group": 0,
         "capacity": 256, "heads_per_group": 3,
         "k": 64, "n": 256, "b_is_input": True},
    ]


def prepare(args) -> int:
    from rknpu_backend import (CompileOptions, compile_tilelang, default_profile,
                               save_plan)
    from models.tilelang_matmul import (make_matmul_prim,
                                         make_matmul_prim_scheduled,
                                         make_matmul_tiled_prim)
    out = Path(args.out).resolve()
    plans_root = out / "plans"
    plans_root.mkdir(parents=True, exist_ok=True)
    capture_path = Path(args.capture).resolve()
    with np.load(capture_path, allow_pickle=False) as cap:
        available = set(cap.files)
        cases = case_specs(capture_path)
        activations = {}
        for case in cases:
            if case.get("qk_construction") is not None:
                if "layer.0.q_rope" not in available or \
                        "layer.0.k_rope" not in available:
                    raise SystemExit("capture lacks q_rope/k_rope")
                q = np.asarray(cap["layer.0.q_rope"], dtype=np.float16)
                k = np.asarray(cap["layer.0.k_rope"], dtype=np.float16)
                group = int(case["group"])
                per = int(case["heads_per_group"])
                cap_n = int(case["capacity"])
                if case["qk_construction"] == "prefill":
                    a = np.zeros((per * cap_n, case["k"]), dtype=np.float16)
                    for h in range(per):
                        a[h * cap_n:h * cap_n + q.shape[0]] = \
                            q[:, group * per + h, :]
                    b = np.zeros((cap_n, case["k"]), dtype=np.float16)
                    b[:k.shape[0]] = k[:, group, :]
                else:
                    a = np.ascontiguousarray(
                        q[0, group * per:(group + 1) * per, :])
                    b = np.zeros((cap_n, case["k"]), dtype=np.float16)
                    b[:k.shape[0]] = k[:, group, :]
                case["m"] = int(a.shape[0])
                activations[f"{case['id']}.A"] = np.ascontiguousarray(a)
                activations[f"{case['id']}.B"] = np.ascontiguousarray(b.T)
                continue
            if case["activation_key"] not in available:
                raise SystemExit(f"capture lacks {case['activation_key']!r}")
            a = np.asarray(cap[case["activation_key"]], dtype=np.float16)
            if case.get("activation_row") is not None:
                row = int(case["activation_row"])
                a = np.ascontiguousarray(a[row:row + 1])
            if a.ndim != 2 or a.shape[1] != case["k"]:
                raise SystemExit(
                    f"{case['id']}: activation shape {a.shape} is not [M,{case['k']}]")
            case["m"] = int(a.shape[0])
            activations[f"{case['id']}.A"] = a
    np.savez(plans_root / "activations.npz", **activations)
    profile = default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled")
    validator_lib_path = os.environ.get("RKNPU_PLAN_MOCK_LIB", "")
    validator = None
    if validator_lib_path:
        from rknpu_backend.runtime.ffi import LibRKPlan, make_binding
        validator = LibRKPlan(validator_lib_path)

    def c_plan_preflight(plan):
        if validator is None:
            return True, None
        cplan = None
        ws = None
        try:
            cplan = validator.create_plan(plan)
            constants = []
            for spec in plan.constants:
                arr = np.zeros(tuple(spec.shape), dtype=np.float16)
                constants.append(make_binding(spec.binding_id, arr))
            ws = validator.create_workspace(cplan, constants, fd=0)
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            if ws is not None:
                ws.destroy()
            if cplan is not None:
                cplan.destroy()

    manifest = {
        "schema_version": 1,
        "kind": "smollm2-schedule-tuning-manifest",
        "capture": str(capture_path),
        "capture_sha256": sha(capture_path),
        "activation_file": "activations.npz",
        "activation_file_sha256": sha(plans_root / "activations.npz"),
        "k_contract": {"k_tile": 256,
                       "accumulation": "ordered_ascending_f16_writeback"},
        "cases": [],
    }
    for case in cases:
        case_dir = plans_root / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        candidates = base_candidates(case["m"], case["kind"])
        prepared = []
        for spec in candidates:
            name = spec["name"]
            source_kind = "default"
            try:
                if spec.get("default"):
                    prim = make_matmul_prim(case["m"], case["k"], case["n"])
                elif (case["k"] % 256 == 0 and
                      case["m"] % spec["tile_m"] == 0 and
                      case["n"] % spec["tile_n"] == 0):
                    prim = make_matmul_tiled_prim(
                        case["m"], case["k"], case["n"],
                        bm=spec["tile_m"], bn=spec["tile_n"],
                        order=spec["order"])
                    source_kind = "tiled_explicit_rounding"
                else:
                    prim = make_matmul_prim_scheduled(
                        case["m"], case["k"], case["n"],
                        tile_m=spec["tile_m"], tile_n=spec["tile_n"],
                        order=spec["order"])
                    source_kind = "single_tile_annotation"
                result = compile_tilelang(
                    prim, profile=profile, options=options,
                    b_is_input=bool(case.get("b_is_input", False)))
                plan = result.plan
                plan_path = case_dir / f"{name}.rkplan.json"
                save_plan(plan, str(plan_path))
                first = next(a for a in plan.actions
                             if a.template_id == "co.mm.f16.v1")
                c_ok, c_error = c_plan_preflight(plan)
                prepared.append({
                    "name": name,
                    "requested": spec,
                    "source": source_kind,
                    "plan": f"{case['id']}/{plan_path.name}",
                    "plan_sha256": sha(plan_path),
                    "plan_hash": plan.plan_hash,
                    "source_schedule": first.attrs.get("source_schedule"),
                    "applied_schedule": first.attrs.get("applied_schedule"),
                    "task_count": plan.accounting.task_count,
                    "submit_count": plan.accounting.submit_count,
                    "buffer_count": len(plan.buffers),
                    "data_bo_count": plan.accounting.data_bo_count,
                    "publish_count": len(plan.publish),
                    "publish_order": [[p.dst_row, p.dst_col, p.rows, p.cols]
                                      for p in plan.publish],
                    "controller_us": float(plan.accounting.compile_us)
                    if hasattr(plan.accounting, "compile_us") else None,
                    "c_verifier_pass": c_ok,
                    "c_verifier_error": c_error,
                })
            except Exception as exc:
                prepared.append({
                    "name": name,
                    "requested": spec,
                    "source": source_kind,
                    "rejected_reason": f"{type(exc).__name__}: {exc}",
                    "c_verifier_pass": False,
                    "c_verifier_error": None,
                })
        manifest["cases"].append({**case, "candidates": prepared})
        print(f"case {case['id']:20s} m={case['m']:3d} k={case['k']:4d} "
              f"n={case['n']:4d} candidates={len(prepared)}", flush=True)
    write_json(plans_root / "manifest.json", manifest)
    write_json(out / "prepare-summary.json", {
        "pass": True,
        "cases": len(cases),
        "candidates": sum(len(c["candidates"]) for c in manifest["cases"]),
        "manifest": "plans/manifest.json",
        "manifest_sha256": sha(plans_root / "manifest.json"),
    })
    print(json.dumps({"pass": True, "cases": len(cases),
                      "candidates": sum(len(c["candidates"])
                                        for c in manifest["cases"])},
                     indent=1))
    return 0


def package_arrays(package: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest = json.loads((package / "manifest.json").read_text())
    weights = np.load(package / manifest["weights"]["path"],
                      allow_pickle=False)
    return manifest, weights


def case_arrays(case: dict, activations: np.lib.npyio.NpzFile,
                weights: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    a = np.ascontiguousarray(activations[f"{case['id']}.A"], dtype=np.float16)
    if case.get("b_is_input"):
        b = np.ascontiguousarray(activations[f"{case['id']}.B"], dtype=np.float16)
        if tuple(b.shape) != (case["k"], case["n"]):
            raise SystemExit(
                f"{case['id']}: dynamic B shape {b.shape} != "
                f"{(case['k'], case['n'])}")
        return a, b
    key = case["weight_key"]
    w = np.asarray(weights[key], dtype=np.float16)
    if "weight_shard" in case:
        index, shards = case["weight_shard"]
        width = (w.shape[0] + shards - 1) // shards
        w = w[index * width:min(w.shape[0], (index + 1) * width)]
    if w.ndim != 2:
        raise SystemExit(f"{case['id']}: weight must be rank 2")
    b = np.ascontiguousarray(w.T)
    if tuple(b.shape) != (case["k"], case["n"]):
        raise SystemExit(
            f"{case['id']}: inferred weight shape {b.shape} != "
            f"{(case['k'], case['n'])}")
    return a, b


def cache_identity(case: dict, schedule: dict, plan_sha: str,
                   package_manifest: dict, lib_sha: str,
                   profile_hash: str) -> dict:
    return {
        "schema_version": 1,
        "case": case["id"],
        "geometry": {"m": case["m"], "k": case["k"], "n": case["n"]},
        "dtype": "float16",
        "b_is_input": bool(case.get("b_is_input", False)),
        "numeric": {"k_tile": 256,
                    "accumulation": "ordered_ascending_f16_writeback"},
        "schedule": schedule,
        "plan_sha256": plan_sha,
        "package_manifest_sha256": package_manifest["_sha256"],
        "package_weights_sha256": package_manifest["weights"]["file_sha256"],
        "plan_lib_sha256": lib_sha,
        "profile_hash": profile_hash,
        "layout_ids": ["co.cube.f16.v1", "co.weight.f16.v1"],
    }


def cache_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_output(out: np.ndarray, ref: np.ndarray) -> dict:
    """Validate shape/finite/tolerance and keep bitwise as a separate fact."""
    shape_ok = tuple(out.shape) == tuple(ref.shape)
    finite = bool(np.isfinite(out).all()) if shape_ok else False
    bitwise = bool(np.array_equal(out, ref)) if shape_ok else False
    numeric = bool(np.allclose(out, ref, atol=8e-3, rtol=8e-3))         if shape_ok and finite else False
    max_abs = (float(np.max(np.abs(out.astype(np.float32) -
                            ref.astype(np.float32))))
               if shape_ok else None)
    return {"shape_ok": shape_ok, "finite": finite,
            "bitwise_equal": bitwise, "numeric_pass": numeric,
            "max_abs": max_abs}


def run_case_candidates(args) -> tuple[dict, dict]:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans_root = Path(args.plans).resolve()
    manifest = json.loads((plans_root / "manifest.json").read_text())
    package = Path(args.package).resolve()
    package_manifest, weights = package_arrays(package)
    package_manifest["_sha256"] = sha(package / "manifest.json")
    with np.load(plans_root / manifest["activation_file"],
                 allow_pickle=False) as activations:
        if getattr(args, "mock", False):
            lib_path = os.environ.get("RKNPU_PLAN_MOCK_LIB", "")
            if not lib_path:
                raise SystemExit("RKNPU_PLAN_MOCK_LIB is required for mock")
            lib = LibRKPlan(lib_path)
            fd = 0
        else:
            lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
            if not lib_path:
                raise SystemExit("RKNPU_PLAN_LIB is required")
            lib = LibRKPlan(lib_path)
            configure_rocket_abi(lib.cdll)
            fd = int(lib.cdll.rocket_open())
            if fd < 0:
                raise SystemExit(f"rocket_open failed: {fd}")
        lib_sha = sha(Path(lib_path))
        candidate_rows = []
        policy_cases = []
        fatal = None
        try:
            for case in manifest["cases"]:
                a, b = case_arrays(case, activations, weights)
                expected = reference_mm_tiled(a, b)
                rows = []
                for candidate in case["candidates"]:
                    if candidate.get("c_verifier_pass") is False:
                        continue
                    plan_path = plans_root / candidate["plan"]
                    rec = {
                        "name": candidate["name"],
                        "source": candidate["source"],
                        "requested": candidate["requested"],
                        "plan": candidate["plan"],
                        "plan_sha256": candidate["plan_sha256"],
                        "plan_hash": candidate["plan_hash"],
                        "task_count": candidate["task_count"],
                        "submit_count": candidate["submit_count"],
                        "buffer_count": candidate["buffer_count"],
                        "data_bo_count": candidate["data_bo_count"],
                        "publish_count": candidate["publish_count"],
                        "publish_order": candidate["publish_order"],
                        "source_schedule": candidate["source_schedule"],
                        "applied_schedule": candidate["applied_schedule"],
                    }
                    cplan = None
                    ws = None
                    try:
                        plan = load_plan(str(plan_path))
                        if plan.plan_hash != candidate["plan_hash"]:
                            raise RuntimeError("loaded plan hash differs from manifest")
                        t0 = time.perf_counter()
                        cplan = lib.create_plan(plan)
                        consts = list(plan.constants)
                        if case.get("b_is_input"):
                            if consts:
                                raise RuntimeError(
                                    "dynamic-B case unexpectedly has constants")
                            ws = lib.create_workspace(cplan, [], fd=fd)
                        else:
                            if len(consts) != 1 or consts[0].name != "B":
                                raise RuntimeError(
                                    "tuning case expects one constant B")
                            b_binding = make_binding(consts[0].binding_id, b)
                            ws = lib.create_workspace(cplan, [b_binding], fd=fd)
                        rec["workspace_init_us"] = (
                            time.perf_counter() - t0) * 1e6
                        in_specs = [x for x in plan.bindings if x.role == "input"]
                        out_spec = next(x for x in plan.bindings
                                        if x.role == "output")
                        out = np.zeros(tuple(out_spec.shape), dtype=np.float16)
                        input_binds = []
                        for spec in in_specs:
                            if spec.name == "A":
                                input_binds.append(make_binding(spec.id, a))
                            elif spec.name == "B" and case.get("b_is_input"):
                                input_binds.append(make_binding(spec.id, b))
                            else:
                                raise RuntimeError(
                                    f"unexpected input binding {spec.name!r}")
                        o_bind = make_binding(out_spec.id, out)
                        ref = np.asarray(expected, dtype=np.float16)
                        warmup_checks = []
                        warmup_valid = True
                        for _ in range(args.warmup):
                            out.fill(0)
                            ws.run(input_binds, [o_bind])
                            check = check_output(out, ref)
                            warmup_checks.append(check)
                            warmup_valid = warmup_valid and check["numeric_pass"]
                        times = []
                        timed_checks = []
                        timed_valid = True
                        for _ in range(args.repeats):
                            out.fill(0)
                            t0 = time.perf_counter()
                            report = ws.run(input_binds, [o_bind])
                            times.append((time.perf_counter() - t0) * 1e6)
                            check = check_output(out, ref)
                            timed_checks.append(check)
                            timed_valid = timed_valid and check["numeric_pass"]
                        candidate_valid = bool(warmup_valid and timed_valid)
                        rec.update({
                            "warmup_checks": warmup_checks,
                            "timed_checks": timed_checks,
                            "candidate_valid": candidate_valid,
                            "bitwise_equal": bool(timed_checks[-1]["bitwise_equal"])
                            if timed_checks else False,
                            "numeric_pass": bool(timed_checks[-1]["numeric_pass"])
                            if timed_checks else False,
                            "finite": bool(timed_checks[-1]["finite"])
                            if timed_checks else False,
                            "reference_max_abs": (timed_checks[-1]["max_abs"]
                                                  if timed_checks else None),
                            "output_sha256": sha_array(out),
                            "times_us": times,
                            "median_us": float(statistics.median(times)),
                            "min_us": float(min(times)),
                            "mean_us": float(statistics.mean(times)),
                            "upload_calls": int(report.upload_calls),
                            "download_calls": int(report.download_calls),
                            "task_report_count": int(report.task_count),
                            "submit_report_count": int(report.submit_count),
                        })
                    except Exception as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        rows.append(rec)
                        fatal = rec["error"]
                        break
                    finally:
                        if ws is not None:
                            ws.destroy()
                        if cplan is not None:
                            cplan.destroy()
                    rows.append(rec)
                    print(f"{case['id']} {candidate['name']} "
                          f"median={rec.get('median_us', 0)/1000:.3f}ms "
                          f"numeric_pass={rec.get('numeric_pass')} "
                          f"bitwise={rec.get('bitwise_equal')} "
                          f"valid={rec.get('candidate_valid')}", flush=True)
                candidate_rows.append({**{k: case[k] for k in
                                          ("id", "kind", "m", "k", "n")},
                                       "candidates": rows})
                if fatal:
                    break
                valid = [r for r in rows if r.get("candidate_valid")
                         and r.get("median_us") is not None]
                if not valid:
                    raise RuntimeError(f"{case['id']}: no valid candidate")
                measured_best = min(valid, key=lambda r: r["median_us"])
                default = next((r for r in valid if r["name"] == "default"),
                               None)
                selected = measured_best
                reason = "autotuner_measured_best"
                if default is not None and measured_best is not default and \
                        measured_best["median_us"] >= default["median_us"] * 0.98:
                    selected = default
                    reason = "default_within_2pct_of_best"
                selected["selected"] = True
                selected["selection_reason"] = reason
                schedule = selected.get("applied_schedule")
                identity = cache_identity(
                    case, schedule, selected["plan_sha256"],
                    package_manifest, lib_sha,
                    package_manifest["compiler"]["profile_hash"])
                policy_cases.append({
                    "id": case["id"],
                    "kind": case["kind"],
                    "m": case["m"], "k": case["k"], "n": case["n"],
                    "selected_plan": selected["plan"],
                    "selected_plan_sha256": selected["plan_sha256"],
                    "selected_plan_hash": selected["plan_hash"],
                    "selected_candidate": selected["name"],
                    "selected_source": selected["source"],
                    "selected_schedule": schedule,
                    "selection_reason": reason,
                    "measured_best": measured_best["name"],
                    "default_median_us": (None if default is None
                                          else default["median_us"]),
                    "winner_median_us": selected["median_us"],
                    "cache_key": cache_key(identity),
                    "identity": identity,
                })
        finally:
            try:
                lib.cdll.rocket_close(fd)
            except Exception:
                pass
    if fatal:
        raise RuntimeError(f"device/candidate failure; search stopped: {fatal}")
    return {
        "schema_version": 1,
        "kind": "smollm2-schedule-tuning-run",
        "manifest": str(plans_root / "manifest.json"),
        "manifest_sha256": sha(plans_root / "manifest.json"),
        "package_manifest_sha256": package_manifest["_sha256"],
        "package_weights_sha256": package_manifest["weights"]["file_sha256"],
        "plan_lib_sha256": lib_sha,
        "cases": candidate_rows,
    }, {
        "schema_version": 1,
        "kind": "smollm2-schedule-policy",
        "source": "autotuner",
        "package_manifest_sha256": package_manifest["_sha256"],
        "package_weights_sha256": package_manifest["weights"]["file_sha256"],
        "plan_lib_sha256": lib_sha,
        "cases": policy_cases,
    }


def run(args) -> int:
    plans_root = Path(args.plans).resolve()
    package = Path(args.package).resolve()
    out = Path(args.out).resolve()
    result, policy = run_case_candidates(args)
    write_json(out, result)
    write_json(plans_root / "policy.json", policy)
    write_json(plans_root / "policy-summary.json", {
        "kind": "smollm2-schedule-policy-summary",
        "cases": [{"id": c["id"], "selected": c["selected_candidate"],
                   "reason": c["selection_reason"],
                   "winner_median_us": c["winner_median_us"],
                   "cache_key": c["cache_key"]}
                  for c in policy["cases"]],
    })
    print(json.dumps((plans_root / "policy-summary.json").read_text(),
                     indent=1)[:3000])
    return 0


def confirm(args) -> int:
    """Independent interleaved baseline/winner confirmation."""
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans_root = Path(args.plans).resolve()
    package = Path(args.package).resolve()
    manifest = json.loads((plans_root / 'manifest.json').read_text())
    policy = json.loads((plans_root / 'policy.json').read_text())
    case = next((c for c in manifest['cases'] if c['id'] == args.case), None)
    if case is None:
        raise SystemExit(f'unknown confirmation case {args.case!r}')
    pcase = next((c for c in policy['cases'] if c['id'] == args.case), None)
    if pcase is None:
        raise SystemExit('policy lacks confirmation case')
    default = next((c for c in case['candidates'] if c['name'] == 'default'),
                   None)
    if default is None or not default.get('c_verifier_pass'):
        raise SystemExit('default candidate is unavailable')
    if pcase['selected_candidate'] == 'default' and not args.candidate:
        write_json(Path(args.out), {
            'schema_version': 1,
            'kind': 'smollm2-schedule-confirmation',
            'case': args.case,
            'pass': True,
            'skipped': 'default_wins',
            'reason': 'autotuner retained the fixed 256 default; no A/B',
        })
        print(json.dumps({'case': args.case, 'skipped': 'default_wins'},
                         indent=1))
        return 0
    selected = pcase['selected_plan']
    if args.candidate:
        candidate = next((c for c in case['candidates']
                          if c['name'] == args.candidate), None)
        if candidate is None:
            raise SystemExit(f'unknown candidate {args.candidate!r}')
        selected = candidate['plan']
    winner = next((c for c in case['candidates']
                   if c['plan'] == selected), None)
    if winner is None:
        raise SystemExit('winner plan is missing from manifest')
    package_manifest, weights = package_arrays(package)
    package_manifest['_sha256'] = sha(package / 'manifest.json')
    with np.load(plans_root / manifest['activation_file'],
                 allow_pickle=False) as acts:
        a, b = case_arrays(case, acts, weights)
        expected = reference_mm_tiled(a, b)
        expected = np.asarray(expected, dtype=np.float16)
        if getattr(args, 'mock', False):
            lib_path = os.environ.get('RKNPU_PLAN_MOCK_LIB', '')
            if not lib_path:
                raise SystemExit('RKNPU_PLAN_MOCK_LIB is required')
            lib = LibRKPlan(lib_path)
            fd = 0
        else:
            lib_path = os.environ.get('RKNPU_PLAN_LIB', '')
            if not lib_path:
                raise SystemExit('RKNPU_PLAN_LIB is required')
            lib = LibRKPlan(lib_path)
            configure_rocket_abi(lib.cdll)
            fd = int(lib.cdll.rocket_open())
            if fd < 0:
                raise SystemExit(f'rocket_open failed: {fd}')
        def execute(plan_info):
            plan = load_plan(str(plans_root / plan_info['plan']))
            cplan = lib.create_plan(plan)
            try:
                consts = list(plan.constants)
                if case.get('b_is_input'):
                    ws = lib.create_workspace(cplan, [], fd=fd)
                else:
                    ws = lib.create_workspace(
                        cplan, [make_binding(consts[0].binding_id, b)], fd=fd)
                try:
                    in_specs = [x for x in plan.bindings if x.role == 'input']
                    out_spec = next(x for x in plan.bindings if x.role == 'output')
                    out = np.zeros(tuple(out_spec.shape), dtype=np.float16)
                    binds = []
                    for spec in in_specs:
                        if spec.name == 'A':
                            binds.append(make_binding(spec.id, a))
                        elif spec.name == 'B' and case.get('b_is_input'):
                            binds.append(make_binding(spec.id, b))
                        else:
                            raise SystemExit(f'unexpected input {spec.name!r}')
                    t0 = time.perf_counter()
                    report = ws.run(binds, [make_binding(out_spec.id, out)])
                    elapsed = (time.perf_counter() - t0) * 1e6
                    return elapsed, check_output(out, expected), report
                finally:
                    ws.destroy()
            finally:
                cplan.destroy()
        rounds = []
        default_times = []
        winner_times = []
        all_valid = True
        try:
            for index in range(args.rounds):
                order = ((default, winner) if index % 2 == 0
                         else (winner, default))
                row = {'round': index, 'order': [x['name'] for x in order]}
                for item in order:
                    elapsed, check, report = execute(item)
                    row[item['name']] = {'us': elapsed, 'check': check,
                                         'task_count': report.task_count,
                                         'submit_count': report.submit_count}
                    if item is default:
                        default_times.append(elapsed)
                    else:
                        winner_times.append(elapsed)
                    all_valid = all_valid and check['numeric_pass']
                rounds.append(row)
        finally:
            try:
                lib.cdll.rocket_close(fd)
            except Exception:
                pass
        def stats(values):
            med = statistics.median(values)
            return {'median_us': med, 'min_us': min(values),
                    'max_us': max(values),
                    'mean_us': statistics.mean(values),
                    'mad_us': statistics.median([abs(x - med) for x in values])}
        speedup = statistics.median(default_times) / statistics.median(winner_times)
        confirmed = bool(all_valid and speedup > 1.0)
        result = {
            'schema_version': 1,
            'kind': 'smollm2-schedule-confirmation',
            'case': args.case,
            'backend': 'mock' if getattr(args, 'mock', False) else 'board',
            'rounds_requested': args.rounds,
            'default': stats(default_times),
            'winner': stats(winner_times),
            'speedup': speedup,
            'confirmed': confirmed,
            'all_valid': all_valid,
            'rounds': rounds,
            'package_manifest_sha256': package_manifest['_sha256'],
            'package_weights_sha256': package_manifest['weights']['file_sha256'],
            'plan_lib_sha256': sha(Path(lib_path)),
            'cache_key': pcase['cache_key'],
        }
        if not confirmed:
            # The search winner did not survive the independent interleaved
            # confirmation.  Retain the existing default policy rather than
            # promoting a noisy candidate.
            pcase['selected_plan'] = default['plan']
            pcase['selected_plan_sha256'] = default['plan_sha256']
            pcase['selected_plan_hash'] = default['plan_hash']
            pcase['selected_candidate'] = 'default'
            pcase['selected_source'] = 'default'
            pcase['selected_schedule'] = default.get('applied_schedule')
            pcase['selection_reason'] = 'independent_confirmation_rejected'
            pcase['confirmed'] = False
            pcase['confirmation_speedup'] = speedup
            pcase['winner_median_us'] = default.get('median_us')
            identity = cache_identity(
                case, default.get('applied_schedule'), default['plan_sha256'],
                package_manifest, sha(Path(lib_path)),
                package_manifest['compiler']['profile_hash'])
            pcase['identity'] = identity
            pcase['cache_key'] = cache_key(identity)
            write_json(plans_root / 'policy.json', policy)
            summary = {
                'kind': 'smollm2-schedule-policy-summary',
                'cases': [{'id': c['id'], 'selected': c['selected_candidate'],
                           'reason': c['selection_reason'],
                           'winner_median_us': c.get('winner_median_us'),
                           'cache_key': c['cache_key']}
                          for c in policy['cases']],
            }
            write_json(plans_root / 'policy-summary.json', summary)
            result['policy_updated_to_default'] = True
    write_json(Path(args.out), result)
    print(json.dumps({k: result[k] for k in ('case', 'speedup', 'all_valid')},
                     indent=1))
    return 0 if all_valid else 1


def replay(args) -> int:
    from rknpu_backend import load_plan
    from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                           make_binding)
    plans_root = Path(args.plans).resolve()
    package = Path(args.package).resolve()
    policy = json.loads((plans_root / "policy.json").read_text())
    if not isinstance(policy.get("cases"), list) or not policy["cases"]:
        raise SystemExit("empty policy is not a valid replay identity")
    case_ids = [c.get("id") for c in policy["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("duplicate case id in policy")
    package_manifest, weights = package_arrays(package)
    package_manifest["_sha256"] = sha(package / "manifest.json")
    if getattr(args, "mock", False):
        lib_path = os.environ.get("RKNPU_PLAN_MOCK_LIB", "")
        if not lib_path:
            raise SystemExit("RKNPU_PLAN_MOCK_LIB is required for mock")
    else:
        lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
        if not lib_path:
            raise SystemExit("RKNPU_PLAN_LIB is required")
    lib_sha = sha(Path(lib_path))
    manifest = json.loads((plans_root / "manifest.json").read_text())
    by_id = {c["id"]: c for c in manifest["cases"]}
    activations = np.load(plans_root / manifest["activation_file"],
                          allow_pickle=False)
    lib = LibRKPlan(lib_path)
    if getattr(args, "mock", False):
        fd = 0
    else:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
    rows = []
    try:
        for pc in policy["cases"]:
            if pc["id"] not in by_id:
                raise SystemExit(f"policy names unknown case {pc['id']!r}")
            case = by_id[pc["id"]]
            a, b = case_arrays(case, activations, weights)
            expected = reference_mm_tiled(a, b)
            plan = load_plan(str(plans_root / pc["selected_plan"]))
            if plan.plan_hash != pc["selected_plan_hash"] or \
                    sha(plans_root / pc["selected_plan"]) != \
                    pc["selected_plan_sha256"]:
                raise SystemExit(f"{pc['id']}: selected plan identity changed")
            ident = cache_identity(case, pc["selected_schedule"],
                                   pc["selected_plan_sha256"], package_manifest,
                                   lib_sha,
                                   package_manifest["compiler"]["profile_hash"])
            if cache_key(ident) != pc["cache_key"]:
                raise SystemExit(f"{pc['id']}: cache key mismatch")
            cplan = lib.create_plan(plan)
            try:
                consts = list(plan.constants)
                if case.get("b_is_input"):
                    if consts:
                        raise SystemExit(f"{pc['id']}: unexpected constants")
                    ws = lib.create_workspace(cplan, [], fd=fd)
                else:
                    ws = lib.create_workspace(
                        cplan, [make_binding(consts[0].binding_id, b)], fd=fd)
                try:
                    in_specs = [x for x in plan.bindings if x.role == "input"]
                    out_spec = next(x for x in plan.bindings if x.role == "output")
                    out = np.zeros(tuple(out_spec.shape), dtype=np.float16)
                    input_binds = []
                    for spec in in_specs:
                        if spec.name == "A":
                            input_binds.append(make_binding(spec.id, a))
                        elif spec.name == "B" and case.get("b_is_input"):
                            input_binds.append(make_binding(spec.id, b))
                        else:
                            raise SystemExit(
                                f"{pc['id']}: unexpected input {spec.name!r}")
                    ref = np.asarray(expected, dtype=np.float16)
                    times = []
                    checks = []
                    for _ in range(args.repeats):
                        out.fill(0)
                        t0 = time.perf_counter()
                        report = ws.run(input_binds,
                                        [make_binding(out_spec.id, out)])
                        times.append((time.perf_counter() - t0) * 1e6)
                        checks.append(check_output(out, ref))
                    valid = all(c["numeric_pass"] for c in checks)
                    rows.append({
                        "id": pc["id"], "candidate": pc["selected_candidate"],
                        "cache_key": pc["cache_key"],
                        "plan_hash": plan.plan_hash,
                        "times_us": times,
                        "median_us": float(statistics.median(times)),
                        "checks": checks,
                        "bitwise_equal": bool(checks[-1]["bitwise_equal"]),
                        "numeric_pass": bool(checks[-1]["numeric_pass"]),
                        "candidate_valid": valid,
                        "reference_max_abs": checks[-1]["max_abs"],
                        "task_count": report.task_count,
                        "submit_count": report.submit_count,
                    })
                finally:
                    ws.destroy()
            finally:
                cplan.destroy()
    finally:
        try:
            lib.cdll.rocket_close(fd)
        except Exception:
            pass
        activations.close()
    result = {
        "schema_version": 1,
        "kind": "smollm2-schedule-replay",
        "pass": bool(rows) and all(r["candidate_valid"] for r in rows),
        "package_manifest_sha256": package_manifest["_sha256"],
        "package_weights_sha256": package_manifest["weights"]["file_sha256"],
        "plan_lib_sha256": lib_sha,
        "cases": rows,
    }
    write_json(Path(args.out), result)
    print(json.dumps({"pass": result["pass"], "cases": len(rows)}, indent=1))
    return 0 if result["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--plans", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run)
    p = sub.add_parser("replay")
    p.add_argument("--plans", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=replay)
    p = sub.add_parser("confirm")
    p.add_argument("--plans", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--case", default="attention_qk_prefill")
    p.add_argument("--candidate", default="")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=confirm)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
