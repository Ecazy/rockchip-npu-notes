#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the RK3588 admission campaign through the production RKPlan C ABI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def environment(lib) -> dict:
    compat = read_text("/proc/device-tree/compatible")
    return {
        # Do not archive the login hostname or the raw board compatible string:
        # a deployment may use either as (part of) a credential.  The model and
        # chip capability are sufficient board identity for this report.
        "board_model": read_text("/proc/device-tree/model").replace("\x00", ""),
        "chip": "rk3588" if "rk3588" in compat else "unknown",
        "machine": os.uname().machine,
        "kernel": os.uname().release,
        "driver_version": read_text("/sys/kernel/debug/rknpu/version"),
        "npu_freq_hz": read_text("/sys/kernel/debug/rknpu/freq"),
        "npu_load_before": read_text("/sys/kernel/debug/rknpu/load"),
        "core_mask_env": os.environ.get("RKNPU_CORE_MASK", "unset"),
        "runtime_abi_version": lib.abi_version(),
        "expected_profile_hash": (
            json.loads((Path(__file__).resolve().parents[1] /
                        "manifest.json").read_text()).get("profile_hash")
            if (Path(__file__).resolve().parents[1] /
                "manifest.json").is_file() else ""),
        "plan_lib_basename": os.path.basename(
            os.environ.get("RKNPU_PLAN_LIB", "")),
    }


def f16_shape(binding) -> tuple[int, int]:
    return tuple(binding.shape)


def make_constant_arrays(plan, npz, *, bad: bool = False):
    arrays = []
    for spec in plan.constants:
        if spec.generator == "identity":
            array = np.asarray(np.eye(spec.shape[0], dtype=np.float32),
                               dtype=np.float16)
        else:
            key = f"constant.{spec.name}"
            if bad:
                bad_key = f"bad_constant.{spec.name}"
                if bad_key in npz:
                    key = bad_key
            if key not in npz:
                raise KeyError(f"missing constant array {key}")
            array = np.asarray(npz[key], dtype=np.float16)
        arrays.append(array)
    return arrays


def binding_array_map(plan, arrays):
    from rknpu_backend.runtime.ffi import make_binding
    return [make_binding(spec.binding_id, array)
            for spec, array in zip(plan.constants, arrays)]


def run_one(case: dict, campaign: Path, manifest: dict, lib,
            fake_fd: int | None = None) -> dict:
    from rknpu_backend import default_profile, load_plan
    from rknpu_backend.errors import BackendError
    from rknpu_backend.runtime.ffi import make_binding

    plan = load_plan(str(campaign / case["plan"]))
    profile = default_profile()
    if plan.profile_hash != profile.profile_hash:
        raise AssertionError("plan/profile hash mismatch")
    if plan.accounting.task_count != case["task_count"] or \
            len(plan.submits) != case["submit_count"] or \
            len(plan.publish) != case["publish_count"]:
        raise AssertionError("plan run counts differ from generated manifest")
    templates = [a.template_id for a in plan.actions if a.kind == "npu_task"]
    if templates != case["templates"]:
        raise AssertionError("plan template sequence differs from manifest")

    data = np.load(campaign / case["data"], allow_pickle=False)
    arrays_keepalive = []
    constants = make_constant_arrays(
        plan, data, bad=(case.get("guard") == "nan_constant"))
    arrays_keepalive.extend(constants)
    cplan = lib.create_plan(plan)
    result = {
        "case": case["name"],
        "kind": case["kind"],
        "core_mask_env": os.environ.get("RKNPU_CORE_MASK", "unset"),
        "plan_hash": plan.plan_hash,
        "profile_hash": plan.profile_hash,
        "task_count": plan.accounting.task_count,
        "submit_count": len(plan.submits),
        "publish_count": len(plan.publish),
        "layout_ids": sorted({view.tensor.layout for view in plan.views}),
        "bindings": [{"name": b.name, "role": b.role,
                      "shape": list(b.shape)} for b in plan.bindings],
        "submits": [{"mode": s.mode, "recipe_id": s.recipe_id,
                     "tasks": len(s.task_ids)} for s in plan.submits],
        "rounds": [],
        "guard": case.get("guard"),
        "status": "PASS",
    }
    fd = -1
    close_fd = False
    ws = None
    try:
        if fake_fd is None:
            lib.cdll.rocket_open.argtypes = []
            lib.cdll.rocket_open.restype = __import__("ctypes").c_int
            lib.cdll.rocket_close.argtypes = [__import__("ctypes").c_int]
            lib.cdll.rocket_close.restype = __import__("ctypes").c_int
            fd = int(lib.cdll.rocket_open())
            close_fd = True
            if fd < 0:
                raise RuntimeError(f"rocket_open failed with {fd}")
        else:
            fd = int(fake_fd)

        try:
            ws = lib.create_workspace(cplan, binding_array_map(plan, constants),
                                      fd=fd)
        except BackendError as exc:
            if case.get("guard") != "nan_constant":
                raise
            diag = exc.diagnostics[0]
            result["guard_code"] = diag.code
            result["guard_message"] = diag.message
            if diag.code != -4:
                raise AssertionError(
                    f"NaN constant returned {diag.code}, expected RK_E_NUMERIC_CONTRACT")
            return result

        output_bindings = [b for b in plan.bindings if b.role == "output"]
        input_bindings = [b for b in plan.bindings if b.role == "input"]
        for round_index in range(case["rounds"]):
            inputs = {}
            for binding in input_bindings:
                key = f"input.{binding.name}.{round_index}"
                if key not in data:
                    raise KeyError(f"missing input array {key}")
                array = np.asarray(data[key], dtype=np.float16)
                arrays_keepalive.append(array)
                inputs[binding.name] = array
            output_arrays = {}
            for binding in output_bindings:
                array = np.zeros(f16_shape(binding), dtype=np.float16)
                arrays_keepalive.append(array)
                output_arrays[binding.name] = array

            if case.get("guard") in {"nan_input", "nonfinite_output"}:
                snapshot = [a.tobytes() for a in output_arrays.values()]
                try:
                    ws.run(
                        [make_binding(b.id, inputs[b.name])
                         for b in input_bindings],
                        [make_binding(b.id, output_arrays[b.name])
                         for b in output_bindings])
                except BackendError as exc:
                    diag = exc.diagnostics[0]
                    result["guard_code"] = diag.code
                    result["guard_message"] = diag.message
                    if diag.code != -4:
                        raise AssertionError(
                            f"{case.get('guard')} returned {diag.code}, expected "
                            "RK_E_NUMERIC_CONTRACT")
                    after = [a.tobytes() for a in output_arrays.values()]
                    if snapshot != after:
                        raise AssertionError("failed guard run modified the output")
                    result["guard_output_unchanged"] = True
                    break
                raise AssertionError(
                    f"{case.get('guard')} was not rejected by the runtime guard")

            report = ws.run(
                [make_binding(b.id, inputs[b.name]) for b in input_bindings],
                [make_binding(b.id, output_arrays[b.name])
                 for b in output_bindings])
            round_info = {
                "round": round_index,
                "task_count": int(report.task_count),
                "submit_count": int(report.submit_count),
                "upload_calls": int(report.upload_calls),
                "upload_bytes": int(report.upload_bytes),
                "download_calls": int(report.download_calls),
                "download_bytes": int(report.download_bytes),
                "pack_us": float(report.pack_us),
                "emit_us": float(report.emit_us),
                "submit_wait_us": float(report.submit_wait_us),
                "output_sync_us": float(report.output_sync_us),
            }
            if case["kind"] != "guard":
                for binding in output_bindings:
                    key = f"expected.{binding.name}.{round_index}"
                    expected = np.asarray(data[key], dtype=np.float16)
                    got = output_arrays[binding.name]
                    if got.shape != expected.shape:
                        raise AssertionError(
                            f"output shape {got.shape} != expected {expected.shape}")
                    diff = np.abs(got.astype(np.float64) -
                                  expected.astype(np.float64))
                    denom = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
                    rel = diff / denom
                    atol = float(case.get("tolerance", {}).get("atol", 0.0))
                    rtol = float(case.get("tolerance", {}).get("rtol", 0.0))
                    allclose = bool(np.allclose(
                        got.astype(np.float64), expected.astype(np.float64),
                        rtol=rtol, atol=atol, equal_nan=False))
                    round_info[binding.name] = {
                        "allclose": allclose,
                        "max_abs": float(diff.max()) if diff.size else 0.0,
                        "max_rel": float(rel.max()) if rel.size else 0.0,
                        "mismatch_elements": int(np.count_nonzero(got != expected)),
                        "element_count": int(got.size),
                    }
                    if not allclose:
                        result["status"] = "FAIL"
            result["rounds"].append(round_info)
        return result
    finally:
        if ws is not None:
            ws.destroy()
        if close_fd and fd >= 0:
            lib.cdll.rocket_close(fd)
        cplan.destroy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fd", type=int, default=None,
                        help="host preflight only: use this fd instead of rocket_open")
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    from rknpu_backend.runtime.ffi import default_library
    lib = default_library()
    env = environment(lib)
    (out / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n")
    failed = 0
    for case in manifest["cases"]:
        started = time.time()
        try:
            result = run_one(case, campaign, manifest, lib, fake_fd=args.fd)
        except Exception as exc:
            result = {
                "case": case.get("name", "<unknown>"),
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "core_mask_env": os.environ.get("RKNPU_CORE_MASK", "unset"),
            }
        result["wall_seconds"] = time.time() - started
        (out / f"{case['name']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        if result.get("status") != "PASS":
            failed += 1
    env["npu_load_after"] = read_text("/sys/kernel/debug/rknpu/load")
    (out / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
