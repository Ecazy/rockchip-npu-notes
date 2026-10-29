#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-step independent calibration/acceptance for approximate device RMSNorm.

Step 1 (calibrate) runs only the calibration split and freezes a spec containing
the plan/profile/library identities and gates.  Step 2 (verify) is a separate
process that loads that spec, checks identities, runs only the acceptance split,
and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNIT = 1.0 / 4096.0
D = 64
M = 16
LUT_SCALE = 0.75
MEMBERS = 32


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def sha_array(a: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def ceil_unit(x: float) -> float:
    return math.ceil(max(0.0, float(x)) / UNIT) * UNIT


def f16_spacing(scale: float) -> float:
    value = np.float16(max(abs(float(scale)), np.finfo(np.float16).tiny))
    return float(abs(np.spacing(value)))


def make_gamma(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.uniform(0.5, 1.5, size=(D, 1)).astype(np.float16))


def make_rows(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    calls = []
    for _ in range(MEMBERS):
        x = rng.normal(size=(M, D)).astype(np.float32)
        norm = np.sqrt(np.mean(x * x, axis=1, keepdims=True))
        target_rms = rng.uniform(math.sqrt(0.17), math.sqrt(6.10),
                                 size=(M, 1)).astype(np.float32)
        x = x / np.maximum(norm, np.float32(1e-12)) * target_rms
        x[0] = x[0] / np.sqrt(np.mean(x[0] ** 2)) * math.sqrt(0.17)
        x[1] = x[1] / np.sqrt(np.mean(x[1] ** 2)) * math.sqrt(6.10)
        calls.append(np.ascontiguousarray(x.astype(np.float16)))
    return np.stack(calls, axis=0)


def ref_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    scale = 1.0 / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-6)
    return np.ascontiguousarray(
        xf * scale * gamma.astype(np.float32).reshape(1, D),
        dtype=np.float16)


def run_device(plan, lib, calls: np.ndarray, gamma: np.ndarray,
               fd: int) -> np.ndarray:
    from rknpu_backend.models import PlanCall
    consts = {
        "Ones": np.ones((1, D), dtype=np.float16),
        "InvD": np.full((M, 1), LUT_SCALE / D, dtype=np.float16),
        "Eps": np.full((M, 1), LUT_SCALE * 1e-6, dtype=np.float16),
        "GammaW": np.ascontiguousarray(
            (gamma.astype(np.float32) * math.sqrt(LUT_SCALE)).astype(
                np.float16).reshape(D, 1)),
        "__identity_32": np.eye(32, dtype=np.float16),
        "__identity_64": np.eye(64, dtype=np.float16),
    }
    outs = []
    with PlanCall(lib, plan, consts, fd=fd) as call:
        for x in calls:
            y = np.empty((M, D), dtype=np.float16)
            call.run_into([np.ascontiguousarray(x)], [y])
            outs.append(y)
    return np.stack(outs, axis=0)


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    e = g - r
    return {
        "finite": bool(np.isfinite(g).all()),
        "max_abs": float(np.max(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "ref_absmax": float(np.max(np.abs(r))),
        "ref_rms": float(np.sqrt(np.mean(r * r))),
    }


def open_backend(args):
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    lib = LibRKPlan(args.lib) if args.lib else LibRKPlan()
    fd = int(args.fd)
    owns_fd = False
    if args.backend == "board":
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
        owns_fd = True
    return lib, fd, owns_fd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("calibrate", "verify"), required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--plan", default=str(
        ROOT / "models" / "compiled_rmsnorm_lut" /
        "tiny_vit_b1_rmsnorm1_approx.rkplan.json"))
    ap.add_argument("--lib", default="")
    ap.add_argument("--fd", type=int, default=0)
    ap.add_argument("--backend", choices=("mock", "board"), default="mock")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from rknpu_backend import load_plan
    plan_path = Path(args.plan).resolve()
    spec_path = Path(args.spec).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plan = load_plan(str(plan_path))
    lib, fd, owns_fd = open_backend(args)
    gamma = make_gamma(20261028)
    try:
        if args.mode == "calibrate":
            if spec_path.exists():
                raise SystemExit(f"refusing to overwrite frozen spec: {spec_path}")
            cal_x = make_rows(20261029)
            cal_y = run_device(plan, lib, cal_x, gamma, fd)
            cal_ref = np.stack([ref_norm(x, gamma) for x in cal_x], axis=0)
            cal_m = metrics(cal_y, cal_ref)
            atol = ceil_unit(max(2.0 * cal_m["max_abs"],
                                 2.0 * f16_spacing(cal_m["ref_absmax"])))
            rmse_limit = ceil_unit(max(2.0 * cal_m["rmse"], 0.25 * atol))
            spec = {
                "schema_version": 2,
                "kind": "rmsnorm-approx-operator-acceptance",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_name": plan.name,
                "plan_hash": plan.plan_hash,
                "profile_hash": plan.profile_hash,
                "plan_file_sha256": sha_file(plan_path),
                "library_sha256": sha_file(Path(args.lib).resolve())
                    if args.lib else None,
                "numeric_policy": {
                    "lut_scale": LUT_SCALE,
                    "lut_table_domain_scaled": [0.125, 6.32],
                    "validated_original_mean_square": [0.17, 6.10],
                    "reference": "float32 mean/rsqrt/gamma then float16 writeback",
                },
                "calibration": {
                    "seed": 20261029,
                    "members": MEMBERS,
                    "rows": MEMBERS * M,
                    "input_sha256": sha_array(cal_x),
                    "output_sha256": sha_array(cal_y),
                    "metrics": cal_m,
                },
                "gates": {
                    "atol": atol,
                    "rmse_limit": rmse_limit,
                    "finite_required": True,
                },
            }
            # The spec hash is a sidecar so the verifier can bind the exact bytes.
            body = json.dumps(spec, indent=1, sort_keys=True) + "\n"
            spec_path.write_text(body)
            spec_hash = sha_bytes(body.encode())
            spec_path.with_name(spec_path.name + ".sha256").write_text(spec_hash + "\n")
            out.write_text(json.dumps({
                "mode": "calibrate", "pass": True,
                "calibration": cal_m, "gates": spec["gates"],
                "spec_sha256": spec_hash,
            }, indent=1, sort_keys=True) + "\n")
            print(json.dumps({"mode": "calibrate", "pass": True,
                              "calibration": cal_m,
                              "gates": spec["gates"],
                              "spec_sha256": spec_hash}, indent=1))
            return 0
        # verify: load frozen spec, bind identity, run acceptance only.
        if not spec_path.exists():
            raise SystemExit(f"missing frozen spec: {spec_path}")
        raw = spec_path.read_bytes()
        spec = json.loads(raw)
        expected_spec_hash = spec_path.with_name(spec_path.name + ".sha256").read_text().strip()
        current_spec_hash = sha_bytes(raw)
        problems = []
        if current_spec_hash != expected_spec_hash:
            problems.append("spec_sha256")
        if spec.get("plan_hash") != plan.plan_hash:
            problems.append("plan_hash")
        if spec.get("profile_hash") != plan.profile_hash:
            problems.append("profile_hash")
        if spec.get("plan_file_sha256") != sha_file(plan_path):
            problems.append("plan_file_sha256")
        if args.lib and spec.get("library_sha256") != sha_file(
                Path(args.lib).resolve()):
            problems.append("library_sha256")
        acc_x = make_rows(20261030)
        acc_y = run_device(plan, lib, acc_x, gamma, fd)
        acc_ref = np.stack([ref_norm(x, gamma) for x in acc_x], axis=0)
        acc_m = metrics(acc_y, acc_ref)
        gates = spec["gates"]
        if not acc_m["finite"]:
            problems.append("nonfinite")
        if acc_m["max_abs"] > gates["atol"]:
            problems.append("max_abs")
        if acc_m["rmse"] > gates["rmse_limit"]:
            problems.append("rmse")
        result = {
            "mode": "verify", "pass": not problems, "problems": problems,
            "spec_sha256": current_spec_hash,
            "plan_hash": plan.plan_hash, "profile_hash": plan.profile_hash,
            "library_sha256": spec.get("library_sha256"),
            "acceptance": acc_m, "gates": gates,
            "acceptance_input_sha256": sha_array(acc_x),
            "acceptance_output_sha256": sha_array(acc_y),
        }
        out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        np.save(out.with_suffix(".acceptance.npy"), acc_y)
        print(json.dumps(result, indent=1))
        return 0 if not problems else 1
    finally:
        if owns_fd:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
