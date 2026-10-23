#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent gate for an RK3588 admission result tree and raw outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text())


def metrics(got: np.ndarray, reference: np.ndarray, *, atol: float,
            rtol: float) -> dict:
    if got.shape != reference.shape:
        raise ValueError(f"shape {got.shape} != reference {reference.shape}")
    diff = np.abs(got.astype(np.float64) - reference.astype(np.float64))
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float((diff / denom).max()) if diff.size else 0.0,
        "mismatched_elements": int(np.count_nonzero(got != reference)),
        "element_count": int(got.size),
        "allclose": bool(np.allclose(got.astype(np.float64),
                                     reference.astype(np.float64),
                                     atol=atol, rtol=rtol, equal_nan=False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    campaign = Path(args.manifest).resolve().parent
    manifest = load_json(campaign / "manifest.json")
    results = Path(args.results).resolve()
    problems: list[str] = []
    core_reports = {}
    specs = {case["name"]: case for case in manifest["cases"]}
    expected_cases = list(specs)
    expected_driver = manifest.get("expected_driver", "")
    expected_profile_hash = manifest.get("profile_hash", "")
    expected_abi = int(manifest.get("expected_runtime_abi_version", 0))

    run_root = results.parent
    source_archive = run_root / "source-bundle.tgz"
    source_hash_path = run_root / "source-bundle.sha256"
    source_hash_record = None
    source_hash_recomputed = None
    if not source_archive.is_file() or not source_hash_path.is_file():
        problems.append("source bundle/hash record missing")
    else:
        source_hash_record = source_hash_path.read_text().split()[0]
        digest = hashlib.sha256()
        with open(source_archive, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(chunk)
        source_hash_recomputed = digest.hexdigest()
        if source_hash_recomputed != source_hash_record:
            problems.append("source bundle hash recomputation mismatch")
    library_hash_record = None
    library_hashes = []
    lib_record_path = results / "library.sha256"
    if lib_record_path.is_file():
        library_hash_record = lib_record_path.read_text().split()[0]
    else:
        problems.append("loaded-library sha256 record missing")

    for mask in (1, 2, 4):
        core = results / f"core-{mask}"
        if not core.is_dir():
            problems.append(f"core {mask}: missing result directory")
            continue
        env_path = core / "environment.json"
        if not env_path.is_file():
            problems.append(f"core {mask}: missing environment.json")
            env = {}
        else:
            env = load_json(env_path)
        driver = str(env.get("driver_version", ""))
        chip = str(env.get("chip", ""))
        if expected_driver and expected_driver not in driver:
            problems.append(
                f"core {mask}: driver {driver!r} does not match {expected_driver!r}")
        if chip != "rk3588":
            problems.append(f"core {mask}: device chip is not RK3588")
        if str(env.get("core_mask_env")) != str(mask):
            problems.append(
                f"core {mask}: runner core_mask_env={env.get('core_mask_env')!r}")
        if int(env.get("target_check_rc", -1)) != 0:
            problems.append(
                f"core {mask}: C runtime target check failed: "
                f"{env.get('target_check_message', '')}")
        if env.get("expected_profile_hash") != expected_profile_hash:
            problems.append(f"core {mask}: profile hash mismatch")
        if int(env.get("runtime_abi_version", -1)) != expected_abi:
            problems.append(f"core {mask}: runtime ABI mismatch")
        if library_hash_record is not None:
            if env.get("plan_lib_sha256") != library_hash_record:
                problems.append(f"core {mask}: loaded library hash record mismatch")
            library_hashes.append(env.get("plan_lib_sha256"))
        exit_path = core / "exit.txt"
        if not exit_path.is_file() or exit_path.read_text().strip() != "0":
            problems.append(f"core {mask}: native campaign exit is not zero")

        cases = {}
        numeric_detail = {}
        guard_detail = {}
        raw_cache = {}
        for name in expected_cases:
            spec = specs[name]
            path = core / f"{name}.json"
            if not path.is_file():
                problems.append(f"core {mask}/{name}: missing case result")
                cases[name] = {"status": "MISSING"}
                continue
            report = load_json(path)
            cases[name] = report
            if report.get("profile_hash") != expected_profile_hash:
                problems.append(f"core {mask}/{name}: plan profile hash mismatch")
            rounds = report.get("rounds") or []
            if int(report.get("task_count", -1)) != int(spec["task_count"]):
                problems.append(f"core {mask}/{name}: static task count mismatch")
            if int(report.get("submit_count", -1)) != int(spec["submit_count"]):
                problems.append(f"core {mask}/{name}: static submit count mismatch")
            if report.get("status") != "PASS":
                problems.append(
                    f"core {mask}/{name}: runner status={report.get('status')!r} "
                    f"error={report.get('error', '')}")

            guard = report.get("guard")
            if guard:
                guard_detail[name] = {
                    "kind": guard,
                    "guard_code": int(report.get("guard_code", 0)),
                    "output_unchanged": bool(
                        report.get("guard_output_unchanged", False)),
                }
                if int(report.get("guard_code", 0)) != -4:
                    problems.append(
                        f"core {mask}/{name}: guard code is not RK_E_NUMERIC_CONTRACT")
                raw_path = core / "outputs" / f"{name}.npz"
                if guard in {"nan_input", "nonfinite_output"}:
                    if not report.get("guard_output_unchanged", False):
                        problems.append(
                            f"core {mask}/{name}: guard did not prove output unchanged")
                    if not raw_path.is_file():
                        problems.append(
                            f"core {mask}/{name}: guard sentinel output missing")
                    else:
                        raw = np.load(raw_path, allow_pickle=False)
                        for out_name in spec["output_names"]:
                            for r in range(int(spec["rounds"])):
                                key = f"guard_preexisting.{out_name}.{r}"
                                if key not in raw:
                                    problems.append(
                                        f"core {mask}/{name}: missing {key}")
                elif guard == "nan_constant":
                    if report.get("guard_output_unchanged", False):
                        problems.append(
                            f"core {mask}/{name}: workspace-create guard must not "
                            "claim output-binding execution")
                    if raw_path.exists():
                        problems.append(
                            f"core {mask}/{name}: unexpected raw output for "
                            "workspace-create rejection")
                continue

            try:
                case_data = np.load(campaign / spec["data"], allow_pickle=False)
            except Exception as exc:
                problems.append(f"core {mask}/{name}: campaign data unreadable: {exc}")
                continue
            raw_path = core / "outputs" / f"{name}.npz"
            if not raw_path.is_file():
                problems.append(f"core {mask}/{name}: raw outputs missing")
                continue
            try:
                raw = np.load(raw_path, allow_pickle=False)
            except Exception as exc:
                problems.append(f"core {mask}/{name}: raw outputs unreadable: {exc}")
                continue
            raw_cache[name] = raw
            if len(rounds) != int(spec["rounds"]):
                problems.append(
                    f"core {mask}/{name}: rounds {len(rounds)} != {spec['rounds']}")
            expected_keys = {
                f"output.{out_name}.{r}"
                for out_name in spec["output_names"]
                for r in range(int(spec["rounds"]))
            }
            if set(raw.files) != expected_keys:
                problems.append(
                    f"core {mask}/{name}: raw output set mismatch")
            atol = float(spec["tolerance"]["atol"])
            rtol = float(spec["tolerance"]["rtol"])
            declared_max = 0.0
            math_max = 0.0
            mismatched = 0
            elements = 0
            for round_info in rounds:
                r = int(round_info.get("round", -1))
                if int(round_info.get("task_count", -1)) != int(
                        spec["task_count"]):
                    problems.append(
                        f"core {mask}/{name} round {r}: task count mismatch")
                if int(round_info.get("submit_count", -1)) != int(
                        spec["submit_count"]):
                    problems.append(
                        f"core {mask}/{name} round {r}: submit count mismatch")
                for out_name in spec["output_names"]:
                    raw_key = f"output.{out_name}.{r}"
                    declared_key = f"declared.{out_name}.{r}"
                    math_key = f"fp64_math.{out_name}.{r}"
                    if raw_key not in raw or declared_key not in case_data or \
                            math_key not in case_data:
                        problems.append(
                            f"core {mask}/{name} round {r}: missing {out_name} "
                            "raw/reference array")
                        continue
                    got = np.asarray(raw[raw_key], dtype=np.float16)
                    expected_shape = tuple(spec["output_shapes"][out_name])
                    if got.shape != expected_shape:
                        problems.append(
                            f"core {mask}/{name} round {r} {out_name}: shape "
                            f"{got.shape} != {expected_shape}")
                        continue
                    declared = np.asarray(case_data[declared_key],
                                          dtype=np.float16)
                    math_ref = np.asarray(case_data[math_key], dtype=np.float64)
                    got_metrics = metrics(got, declared, atol=atol, rtol=rtol)
                    math_metrics = metrics(got, math_ref, atol=atol, rtol=rtol)
                    declared_max = max(declared_max, got_metrics["max_abs"])
                    math_max = max(math_max, math_metrics["max_abs"])
                    mismatched += got_metrics["mismatched_elements"]
                    elements += got_metrics["element_count"]
                    if not got_metrics["allclose"]:
                        problems.append(
                            f"core {mask}/{name} round {r} {out_name}: declared "
                            f"path allclose failed max_abs={got_metrics['max_abs']}")
            numeric_detail[name] = {
                "declared_max_abs": declared_max,
                "fp64_math_max_abs": math_max,
                "mismatched_elements": mismatched,
                "element_count": elements,
                "bit_exact_declared": mismatched == 0,
                "rounds": len(rounds),
                "tasks_per_round": int(spec["task_count"]),
                "submits_per_round": int(spec["submit_count"]),
            }

        # Direct chain/separate hardware-output comparison.
        pair_detail = {}
        for left, right in (("projection_chain", "projection_separate"),
                            ("ffn_chain", "ffn_separate"),
                            ("ragged_chain", "ragged_separate")):
            if left not in raw_cache or right not in raw_cache:
                problems.append(
                    f"core {mask}: cannot compare raw outputs {left}/{right}")
                continue
            a, b = raw_cache[left], raw_cache[right]
            if set(a.files) != set(b.files):
                problems.append(f"core {mask}: {left}/{right} output keys differ")
                continue
            bit_equal = True
            max_abs = 0.0
            for key in a.files:
                av = np.asarray(a[key])
                bv = np.asarray(b[key])
                if av.dtype != bv.dtype or av.shape != bv.shape or \
                        av.tobytes() != bv.tobytes():
                    bit_equal = False
                if av.size:
                    max_abs = max(max_abs, float(np.max(
                        np.abs(av.astype(np.float64) -
                               bv.astype(np.float64)))))
            pair_detail[f"{left}/{right}"] = {
                "bit_equal": bit_equal, "max_abs": max_abs}
            if not bit_equal or max_abs != 0.0:
                problems.append(
                    f"core {mask}: raw chain/separate mismatch {left}/{right} "
                    f"(bit_equal={bit_equal}, max_abs={max_abs})")
        cli_dir = core / "cli"
        cli_report_path = cli_dir / "cli-report.json"
        cli_report = {}
        if not cli_report_path.is_file():
            problems.append(f"core {mask}: CLI report missing")
        else:
            cli_report = load_json(cli_report_path)
            if cli_report.get("status") != "PASS":
                problems.append(f"core {mask}: CLI path status != PASS")
            steps = cli_report.get("steps", {})
            expected_codes = {"compile": 0, "validate": 0, "explain": 0,
                              "run_0": 0, "run_1": 0, "bad_binding": 2}
            for step, expected in expected_codes.items():
                if int(steps.get(step, {}).get("rc", -999)) != expected:
                    problems.append(
                        f"core {mask}: CLI {step} exit code mismatch")
            for step in ("run_0", "run_1"):
                if not steps.get(step, {}).get("numeric_ok"):
                    problems.append(f"core {mask}: CLI {step} numeric mismatch")
            if steps.get("bad_binding", {}).get("published"):
                problems.append(
                    f"core {mask}: CLI bad-binding run published output")
            for item in cli_report.get("fd", []):
                if item.get("before") != item.get("after"):
                    problems.append(
                        f"core {mask}: CLI fd ownership leak at "
                        f"{item.get('round')}")
        cli_exit = cli_dir / "exit.txt"
        if not cli_exit.is_file() or cli_exit.read_text().strip() != "0":
            problems.append(f"core {mask}: CLI runner exit is not zero")

        core_reports[str(mask)] = {
            "environment": env, "cases": cases,
            "numeric_detail": numeric_detail, "guard_detail": guard_detail,
            "chain_separate": pair_detail, "cli": cli_report,
        }

    summary = {
        "kind": "rk3588-board-admission-summary",
        "overall_exit": 0 if not problems else 1,
        "driver_expected": expected_driver,
        "profile_hash": expected_profile_hash,
        "runtime_abi_version": expected_abi,
        "cores": sorted(core_reports, key=int),
        "cases_expected": expected_cases,
        "tolerance_contract": manifest.get("tolerance_contract"),
        "artifact_verification": {
            "source_bundle_sha256": {
                "recorded": source_hash_record,
                "recomputed": source_hash_recomputed,
                "verified": (source_hash_record is not None and
                             source_hash_record == source_hash_recomputed),
            },
            "loaded_library_sha256": {
                "recorded": library_hash_record,
                "per_core_records": library_hashes,
                "verified": False,
                "note": "recorded from the remote board; the library bytes were "
                        "not independently rehashed on the Host",
            },
        },
        "problems": problems,
        "core_reports": core_reports,
    }
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# RK3588 board admission",
        "",
        f"overall: **{'PASS' if not problems else 'FAIL'}**",
        f"driver expected: `{expected_driver}`",
        f"profile hash: `{expected_profile_hash}`",
        "",
        "| core | cases | declared numeric | guards | chain/separate bit-equal | CLI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mask in (1, 2, 4):
        report = core_reports.get(str(mask), {})
        cases = report.get("cases", {})
        passed = sum(1 for c in cases.values() if c.get("status") == "PASS")
        numeric = len(report.get("numeric_detail", {}))
        guards = len(report.get("guard_detail", {}))
        pairs = report.get("chain_separate", {})
        pair_ok = sum(1 for item in pairs.values()
                      if item.get("bit_equal") and item.get("max_abs") == 0.0)
        cli_status = report.get("cli", {}).get("status", "MISSING")
        lines.append(f"| {mask} | {passed}/{len(expected_cases)} | {numeric} | "
                     f"{guards} | {pair_ok}/{len(pairs)} | {cli_status} |")
    lines += ["", "## Numeric detail", "",
              "Declared acceptance: `abs(got-declared) <= atol + rtol*abs(declared)`.",
              "The FP64 mathematical reference is reported for context and is not "
              "the acceptance predicate.", "",
              "| core | case | declared max abs | FP64 math max abs | "
              "mismatch/elements | tasks/submits | bit exact |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for mask in (1, 2, 4):
        detail = core_reports.get(str(mask), {}).get("numeric_detail", {})
        for name in expected_cases:
            if name not in detail:
                continue
            item = detail[name]
            lines.append(
                f"| {mask} | {name} | {item['declared_max_abs']:.10g} | "
                f"{item['fp64_math_max_abs']:.10g} | "
                f"{item['mismatched_elements']}/{item['element_count']} | "
                f"{item['tasks_per_round']}/{item['submits_per_round']} | "
                f"{'yes' if item['bit_exact_declared'] else 'no'} |")
    lines += ["", "## Artifact verification", "",
              f"- source bundle SHA-256 recorded: `{source_hash_record}`",
              f"- source bundle SHA-256 recomputed: `{source_hash_recomputed}`",
              f"- source bundle bytes independently verified: "
              f"`{source_hash_record == source_hash_recomputed}`",
              f"- loaded library SHA-256 (recorded remote): "
              f"`{library_hash_record}`; independently rehashed: `False`",
              "", "## Chain/separate raw-output comparison", ""]
    for mask in (1, 2, 4):
        pairs = core_reports.get(str(mask), {}).get("chain_separate", {})
        for name, item in sorted(pairs.items()):
            lines.append(
                f"- core {mask} `{name}`: bit_equal={item['bit_equal']}, "
                f"max_abs={item['max_abs']:.10g}")
    lines += ["", "## Guard detail", ""]
    for mask in (1, 2, 4):
        guards = core_reports.get(str(mask), {}).get("guard_detail", {})
        for name, item in sorted(guards.items()):
            lines.append(
                f"- core {mask} `{name}`: code={item['guard_code']}, "
                f"output_unchanged={item['output_unchanged']}")
    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    else:
        lines += ["", "All requested cases passed on all three single-core masks."]
    (results / "summary.md").write_text("\n".join(lines) + "\n")
    print(results / "summary.md")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
