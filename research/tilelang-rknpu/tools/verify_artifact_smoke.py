#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent Host verification of the board artifact smoke test.

The board only saves raw outputs; every numeric and structural judgement happens
here, against the float64-with-explicit-FP16 oracle the campaign builder wrote.
Nothing about the board's run is trusted:

* plan identity -- the board-recompiled plan hash and the reloaded plan hash must
  equal the Host artifact's,
* ``explain`` consistency and the workspace's own task/submit accounting must match
  the plan,
* A->B->A on one workspace: the two A runs must be bit-identical, and B must
  produce a different value (a stale-input pass would otherwise look correct),
* two independent instantiations must agree bit-for-bit,
* ``export -> load_plan -> run`` must agree bit-for-bit with the artifact path,
* numeric acceptance against the Host oracle using the plan's **declared** tolerance,
  which is read from the campaign metadata and never widened here.

Exit status is non-zero on any failed check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(actual: np.ndarray, expected: np.ndarray, tolerance: dict) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        return {"ok": False, "reason": f"shape {a.shape} != {e.shape}"}
    finite = bool(np.isfinite(a).all())
    delta = np.abs(a - e)
    allowed = tolerance["atol"] + tolerance["rtol"] * np.abs(e)
    if not finite:
        return {"ok": False, "reason": "non-finite values on the device",
                "max_abs_err": None}
    ok = bool((delta <= allowed).all())
    return {
        "ok": ok,
        "max_abs_err": float(delta.max()) if delta.size else 0.0,
        "mean_abs_err": float(delta.mean()) if delta.size else 0.0,
        "worst_allowed": float(allowed.max()) if allowed.size else 0.0,
        "atol": tolerance["atol"],
        "rtol": tolerance["rtol"],
        "elements": int(delta.size),
        "all_finite": finite,
    }


def exact_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(first), np.asarray(second)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--board", required=True,
                        help="directory containing the fetched board results")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    campaign = Path(args.campaign)
    board = Path(args.board)
    # The board writes one directory per core mask; accept either the mask
    # directory itself or the fetched results root that contains exactly one.
    if not (board / "summary.json").is_file():
        candidates = sorted(board.glob("core-*/summary.json"))
        if len(candidates) == 1:
            board = candidates[0].parent
    out_path = Path(args.out)

    manifest = load_json(campaign / "manifest.json")
    board_summary_path = board / "summary.json"
    report: dict[str, Any] = {
        "kind": "artifact-smoke-verification",
        "campaign": str(campaign),
        "board": str(board),
        "board_summary_present": board_summary_path.is_file(),
        "cases": {},
        "failures": [],
    }
    if not board_summary_path.is_file():
        report["failures"].append("board summary.json is missing")
        out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print("FAIL: board summary.json is missing")
        return 1
    board_summary = load_json(board_summary_path)
    report["board_environment"] = board_summary.get("environment", {})
    report["board_status_by_case"] = {
        entry["case"]: entry.get("status") for entry in board_summary.get("cases", [])}

    for entry in manifest["cases"]:
        name = entry["name"]
        case_dir = campaign / entry["dir"]
        meta = load_json(case_dir / "meta.json")
        case_board = board / name
        verdict: dict[str, Any] = {"tolerance": meta["tolerance"],
                                   "checks": {}, "outputs": {}}
        case_result_path = case_board / "result.json"
        if not case_result_path.is_file():
            report["failures"].append(f"{name}: board result.json missing")
            verdict["checks"]["board_result_present"] = False
            report["cases"][name] = verdict
            continue
        result = load_json(case_result_path)
        verdict["checks"]["board_status_ok"] = result.get("status") == "ok"
        verdict["checks"]["steps_complete"] = result.get("steps") == [
            "compile_artifact", "explain", "instantiate_run_a_b_a",
            "two_independent_instantiations", "export_load_run"]
        verdict["checks"]["plan_hash_board_matches_host"] = bool(
            result.get("plan_hash_match")) and result.get("plan_hash_board") == meta["plan_hash"]
        verdict["checks"]["plan_hash_reload_matches_host"] = bool(
            result.get("plan_hash_reloaded_match")) and result.get(
                "plan_hash_reloaded") == meta["plan_hash"]
        verdict["checks"]["explain_matches_plan"] = bool(
            result.get("explain_matches_plan"))
        explain_counts = result.get("explain_counts") or {}
        verdict["checks"]["explain_counts_match"] = (
            explain_counts.get("tasks") == meta["tasks"]
            and explain_counts.get("submits") == meta["submits"]
            and explain_counts.get("data_bos") == meta["data_bos"])
        report_single = result.get("report_single") or {}
        verdict["checks"]["workspace_accounting_matches"] = (
            report_single.get("task_count") == meta["tasks"]
            and report_single.get("submit_count") == meta["submits"])

        for binding in meta["outputs"]:
            def load(stage: str) -> np.ndarray | None:
                path = case_board / "board" / stage / f"{binding}.npy"
                if not path.is_file():
                    return None
                return np.load(path, allow_pickle=False)

            a1 = load("a1")
            a2 = load("a2")
            b = load("b")
            ind1 = load("independent_1")
            ind2 = load("independent_2")
            replay = load("replay")
            missing = [stage for stage, array in
                       (("a1", a1), ("a2", a2), ("b", b), ("independent_1", ind1),
                        ("independent_2", ind2), ("replay", replay))
                       if array is None]
            if missing:
                report["failures"].append(
                    f"{name}/{binding}: missing board output(s) {missing}")
                verdict["outputs"][binding] = {"missing": missing}
                continue
            reference_a = np.load(case_dir / "reference" / f"a_{binding}.npy",
                                  allow_pickle=False)
            reference_b = np.load(case_dir / "reference" / f"b_{binding}.npy",
                                  allow_pickle=False)
            verdict["outputs"][binding] = {
                "numeric_a": compare(a1, reference_a, meta["tolerance"]),
                "numeric_b": compare(b, reference_b, meta["tolerance"]),
                "a_repeat_bitwise_equal": exact_equal(a1, a2),
                "b_differs_from_a": not exact_equal(a1, b),
                "independent_bitwise_equal": exact_equal(ind1, ind2),
                "independent_equals_first": exact_equal(ind1, a1),
                "replay_bitwise_equal": exact_equal(replay, a1),
            }

        failures = []
        for check, ok in verdict["checks"].items():
            if not ok:
                failures.append(f"{name}: check {check} failed")
        for binding, detail in verdict["outputs"].items():
            if "missing" in detail:
                continue
            if not detail["numeric_a"]["ok"]:
                failures.append(f"{name}/{binding}: numeric A mismatch "
                                f"(max_abs_err={detail['numeric_a'].get('max_abs_err')}, "
                                f"atol={meta['tolerance']['atol']})")
            if not detail["numeric_b"]["ok"]:
                failures.append(f"{name}/{binding}: numeric B mismatch "
                                f"(max_abs_err={detail['numeric_b'].get('max_abs_err')}, "
                                f"atol={meta['tolerance']['atol']})")
            if not detail["a_repeat_bitwise_equal"]:
                failures.append(f"{name}/{binding}: repeated A run differs")
            if not detail["b_differs_from_a"]:
                failures.append(f"{name}/{binding}: B input produced the A result "
                                "(input update not applied)")
            if not detail["independent_bitwise_equal"]:
                failures.append(f"{name}/{binding}: independent instantiations differ")
            if not detail["independent_equals_first"]:
                failures.append(f"{name}/{binding}: independent run differs from the "
                                "first workspace")
            if not detail["replay_bitwise_equal"]:
                failures.append(f"{name}/{binding}: export/load replay differs")
        report["failures"].extend(failures)
        report["cases"][name] = verdict

    report["passed"] = not report["failures"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")

    print(f"cases={len(report['cases'])} failures={len(report['failures'])}")
    for name, verdict in report["cases"].items():
        numeric = [detail["numeric_a"].get("max_abs_err")
                   for detail in verdict["outputs"].values()
                   if isinstance(detail.get("numeric_a"), dict)]
        pretty = ", ".join(f"{value:.3e}" for value in numeric if value is not None)
        print(f"  {name:16s} max_abs_err=[{pretty}] "
              f"tolerance={verdict['tolerance']['atol']}")
    for failure in report["failures"][:12]:
        print(f"  FAIL {failure}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
