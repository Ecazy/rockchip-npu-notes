#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Board half of the refactor smoke test: the artifact API against the real NPU.

Runs one case at a time, serially, and stops at the first failure so a device
submission fault can never be followed by another submission.

Per case it exercises, in this order:

1. ``Module.from_dict`` of the normalized graph the Host produced from a **real
   TileLang PrimFunc**, then ``compile_artifact`` and ``explain`` -- compiled on the
   board, and its plan hash must equal the Host artifact's (cross-machine identity).
2. ``artifact.instantiate(fd=...)`` -> ``run(A)`` -> ``run(B)`` -> ``run(A)`` on the
   *same* workspace, so input updates and buffer reuse are covered.
3. two independent instantiations kept alive at the same time, each run once.
4. ``artifact.export_plan`` -> ``load_plan`` -> ``Workspace.create`` -> ``run(A)``,
   the offline replay path that deployment uses.

Raw outputs are written as ``.npy`` for the Host-side verifier; nothing is compared
against the model tolerances here, so the board cannot "pass itself".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _env(name: str) -> str:
    try:
        return Path(name).read_text().strip()
    except Exception:
        return ""


def environment() -> dict:
    compat = _env("/proc/device-tree/compatible")
    return {
        "python": sys.version.split()[0],
        "machine": os.uname().machine,
        "kernel": os.uname().release,
        # Never archive the login hostname: only the board model/chip.
        "board_model": _env("/proc/device-tree/model").replace("\x00", ""),
        "chip": "rk3588" if "rk3588" in compat else "unknown",
        "driver_version": _env("/sys/kernel/debug/rknpu/version"),
        "npu_freq_hz": _env("/sys/kernel/debug/rknpu/freq"),
        "npu_load_before": _env("/sys/kernel/debug/rknpu/load"),
        "core_mask_env": os.environ.get("RKNPU_CORE_MASK", "unset"),
        "plan_lib": os.path.basename(os.environ.get("RKNPU_PLAN_LIB", "")),
        "chip_env": os.environ.get("RKNPU_CHIP", "unset"),
    }


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def _load_inputs(case_dir: Path, inputs: list[str]) -> tuple[dict, dict]:
    a = {name: np.load(case_dir / "inputs" / f"a_{name}.npy",
                       allow_pickle=False) for name in inputs}
    b = {name: np.load(case_dir / "inputs" / f"b_{name}.npy",
                       allow_pickle=False) for name in inputs}
    return a, b


def _load_constants(case_dir: Path, meta: dict) -> dict:
    out = {}
    for spec in meta["constants"]:
        if spec.get("generator"):
            continue            # the runtime materialises identity constants
        out[spec["name"]] = np.load(case_dir / "constants" / f"{spec['name']}.npy",
                                    allow_pickle=False)
    return out


def _task_report(workspace) -> dict:
    report = getattr(workspace, "last_report", {}) or {}
    keep = ("task_count", "submit_count", "upload_calls", "download_calls",
            "numeric_mode")
    return {key: report.get(key) for key in keep if key in report}


def run_case(name: str, case_dir: Path, out_dir: Path, *, fd: int) -> dict:
    from rknpu_backend import (CompileOptions, compile_artifact, default_profile,
                               load_plan)
    from rknpu_backend.ir import Module
    from rknpu_backend.runtime.bindings import Workspace
    from rknpu_backend.serialize import canonical_json, load_json_safe

    meta = load_json_safe(case_dir / "meta.json")
    result: dict[str, Any] = {
        "case": name,
        "plan_hash_expected": meta["plan_hash"],
        "status": "running",
        "steps": [],
    }
    profile = default_profile()
    options = CompileOptions(**meta["options"])

    # 1. artifact path, compiled on the board from the normalized graph.
    graph = Module.from_dict(load_json_safe(case_dir / "rkir.json"))
    artifact = compile_artifact(graph, profile=profile, options=options)
    report = artifact.explain()
    result["plan_hash_board"] = artifact.plan_hash
    result["plan_hash_match"] = artifact.plan_hash == meta["plan_hash"]
    result["explain_counts"] = report["counts"]
    result["explain_policy"] = report["policy"]
    result["explain_matches_plan"] = (
        report["counts"]["tasks"] == meta["tasks"]
        and report["counts"]["submits"] == meta["submits"])
    result["steps"].append("compile_artifact")
    result["steps"].append("explain")
    if not result["plan_hash_match"]:
        raise RuntimeError(
            f"board plan hash {artifact.plan_hash} != host {meta['plan_hash']}")

    inputs_a, inputs_b = _load_inputs(case_dir, meta["inputs"])
    constants = _load_constants(case_dir, meta)
    outputs = meta["outputs"]

    # 2. one workspace, three runs: A -> B -> A.
    workspace = artifact.instantiate(fd=fd, constants=constants)
    try:
        run_a1 = workspace.run(inputs_a)
        run_b = workspace.run(inputs_b)
        run_a2 = workspace.run(inputs_a)
        result["report_single"] = _task_report(workspace)
    finally:
        workspace.close()
    for binding in outputs:
        _save(out_dir / "board" / "a1" / f"{binding}.npy", run_a1[binding])
        _save(out_dir / "board" / "b" / f"{binding}.npy", run_b[binding])
        _save(out_dir / "board" / "a2" / f"{binding}.npy", run_a2[binding])
    result["steps"].append("instantiate_run_a_b_a")

    # 3. two independent instantiations alive at the same time.
    first = artifact.instantiate(fd=fd, constants=constants)
    second = artifact.instantiate(fd=fd, constants=constants)
    try:
        independent_1 = first.run(inputs_a)
        independent_2 = second.run(inputs_a)
        result["report_independent"] = [_task_report(first),
                                        _task_report(second)]
    finally:
        second.close()
        first.close()
    for binding in outputs:
        _save(out_dir / "board" / "independent_1" / f"{binding}.npy",
              independent_1[binding])
        _save(out_dir / "board" / "independent_2" / f"{binding}.npy",
              independent_2[binding])
    result["steps"].append("two_independent_instantiations")

    # 4. export -> load_plan -> run (the deployment/replay path).
    exported = out_dir / "exported_plan.json"
    artifact.export_plan(str(exported))
    reloaded = load_plan(str(exported))
    result["plan_hash_reloaded"] = reloaded.plan_hash
    result["plan_hash_reloaded_match"] = reloaded.plan_hash == meta["plan_hash"]
    replayed = Workspace.create(reloaded, fd=fd, constants=constants)
    try:
        run_replay = replayed.run(inputs_a)
        result["report_replay"] = _task_report(replayed)
    finally:
        replayed.close()
    for binding in outputs:
        _save(out_dir / "board" / "replay" / f"{binding}.npy",
              run_replay[binding])
    result["steps"].append("export_load_run")

    result["status"] = "ok"
    result_path = out_dir / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fd", type=int, default=None,
                        help="caller-owned fd; omit to open the device here")
    parser.add_argument("--only", default=None,
                        help="comma separated case names")
    args = parser.parse_args(argv)

    from rknpu_backend.runtime.ffi import configure_rocket_abi, default_library
    from rknpu_backend.serialize import canonical_json, load_json_safe

    campaign = Path(args.campaign)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json_safe(campaign / "manifest.json")
    only = set(args.only.split(",")) if args.only else None

    lib = default_library()
    owns_fd = args.fd is None
    if owns_fd:
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            print("rocket_open failed; no supported RKNPU fd", file=sys.stderr)
            return 3
    else:
        fd = int(args.fd)

    summary: dict[str, Any] = {
        "kind": "artifact-smoke-board",
        "environment": environment(),
        "abi_version": lib.abi_version(),
        "campaign_manifest_sha256": __import__("hashlib").sha256(
            (campaign / "manifest.json").read_bytes()).hexdigest(),
        "fd_owned_by_runner": owns_fd,
        "cases": [],
    }
    status = 0
    try:
        for entry in manifest["cases"]:
            name = entry["name"]
            if only and name not in only:
                continue
            case_dir = campaign / entry["dir"]
            case_out = out_root / name
            case_out.mkdir(parents=True, exist_ok=True)
            try:
                result = run_case(name, case_dir, case_out, fd=fd)
            except BaseException as exc:  # noqa: BLE001 - reported, then stop
                status = 1
                failure = {
                    "case": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-12:],
                }
                (case_out / "result.json").write_text(
                    canonical_json(failure) + "\n", encoding="utf-8")
                print(f"{name}: FAILED {failure['error']}", flush=True)
                summary["cases"].append(failure)
                # Stop immediately: no further device submission after a fault.
                break
            print(f"{name}: ok plan={result['plan_hash_board'][:12]} "
                  f"steps={','.join(result['steps'])}", flush=True)
            summary["cases"].append({
                "case": name,
                "status": "ok",
                "plan_hash_board": result["plan_hash_board"],
                "plan_hash_match": result["plan_hash_match"],
                "report_single": result.get("report_single"),
                "report_independent": result.get("report_independent"),
                "report_replay": result.get("report_replay"),
                "explain_counts": result.get("explain_counts"),
            })
    finally:
        summary["npu_load_after"] = _env("/sys/kernel/debug/rknpu/load")
        (out_root / "summary.json").write_text(canonical_json(summary) + "\n",
                                               encoding="utf-8")
        if owns_fd:
            # rocket_close is void in the vendor header; never reset the device.
            try:
                lib.cdll.rocket_close(fd)
            except Exception:
                pass
    return status


if __name__ == "__main__":
    raise SystemExit(main())
