#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise compile-rkir/validate/explain/run through the CLI on RK3588."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

from rknpu_backend.cli import main as cli_main


def fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def call(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = int(cli_main(args))
    return rc, out.getvalue(), err.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fd", type=int, default=None,
                        help="test-only: pass a caller-owned fd to CLI run")
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    cli = manifest["cli"]
    plan = out / "matmul.rkplan.json"
    report = {"status": "PASS", "steps": {}, "fd": []}

    compile_rc, _, compile_err = call([
        "compile-rkir", str(campaign / cli["rkir"]),
        "--profile", "rk3588_vendor_co_v1", "--numeric-mode", "rk_f16_tiled",
        "--out", str(plan)])
    report["steps"]["compile"] = {"rc": compile_rc, "stderr": compile_err}
    if compile_rc != 0 or not plan.is_file():
        report["status"] = "FAIL"
        (out / "cli-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 1

    validate_rc, _, validate_err = call(["validate", str(plan)])
    report["steps"]["validate"] = {"rc": validate_rc,
                                   "stderr": validate_err}
    explain_rc, explain_out, explain_err = call([
        "explain", str(plan), "--format", "json"])
    report["steps"]["explain"] = {"rc": explain_rc, "stderr": explain_err}
    try:
        explained = json.loads(explain_out)
    except Exception as exc:
        report["status"] = "FAIL"
        report["steps"]["explain"]["parse_error"] = str(exc)
        explained = {}
    if validate_rc != 0 or explain_rc != 0 or explained.get("kind") != "rkplan":
        report["status"] = "FAIL"

    expected = np.load(campaign / cli["expected_output"], allow_pickle=False)
    atol = float(cli["tolerance"]["atol"])
    rtol = float(cli["tolerance"]["rtol"])
    baseline_fds = fd_count()
    for round_index in range(2):
        run_out = out / f"run-{round_index}"
        before = fd_count()
        run_args = ["run", str(plan), "--bindings",
                    str(campaign / cli["bindings_success"]),
                    "--output-dir", str(run_out)]
        if args.fd is not None:
            run_args += ["--fd", str(args.fd)]
        run_rc, _, run_err = call(run_args)
        after = fd_count()
        result_path = run_out / "Y.npy"
        numeric_ok = False
        max_abs = None
        if result_path.is_file():
            got = np.load(result_path, allow_pickle=False)
            if got.shape == expected.shape:
                max_abs = float(np.max(np.abs(
                    got.astype(np.float64) - expected.astype(np.float64))))
                numeric_ok = bool(np.allclose(
                    got.astype(np.float64), expected.astype(np.float64),
                    atol=atol, rtol=rtol, equal_nan=False))
        report["steps"][f"run_{round_index}"] = {
            "rc": run_rc, "stderr": run_err, "max_abs": max_abs,
            "numeric_ok": numeric_ok, "fd_before": before,
            "fd_after": after,
        }
        report["fd"].append({"round": round_index, "before": before,
                             "after": after})
        if run_rc != 0 or not numeric_ok or before != after:
            report["status"] = "FAIL"

    bad_out = out / "run-bad-binding"
    before = fd_count()
    bad_args = ["run", str(plan), "--bindings",
                str(campaign / cli["bindings_bad"]),
                "--output-dir", str(bad_out)]
    if args.fd is not None:
        bad_args += ["--fd", str(args.fd)]
    bad_rc, _, bad_err = call(bad_args)
    after = fd_count()
    bad_files = list(bad_out.glob("*.npy")) if bad_out.exists() else []
    report["steps"]["bad_binding"] = {
        "rc": bad_rc, "stderr": bad_err, "published": bool(bad_files),
        "fd_before": before, "fd_after": after,
    }
    report["fd"].append({"round": "bad_binding", "before": before,
                         "after": after})
    if bad_rc != 2 or bad_files or before != after:
        report["status"] = "FAIL"
    report["baseline_fds"] = baseline_fds
    (out / "cli-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
