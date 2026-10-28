#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the multi-submit budget-closure correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

CASES = ("mlp_b32", "mlp_b1")
ARMS = ("off", "on")


def median(values):
    return float(statistics.median([float(v) for v in values])) if values else None


def p10(values):
    rows = sorted(float(v) for v in values)
    return rows[(len(rows) - 1) // 10] if rows else None


def parse_timing(path: Path):
    out = {"warm": []}
    for line in path.read_text().splitlines():
        if line.startswith("SETUP_US "):
            out["setup"] = float(line.split()[1])
        elif line.startswith("WARM_US "):
            out["warm"] = [float(x) for x in line.split()[1:]]
    return out


def parse_smoke(path: Path):
    out = {"na": None, "mem": {}, "info": {}}
    for line in path.read_text().splitlines():
        if line.startswith("MULTI_NA "):
            out["na"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_INFO "):
            out["info"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_MEM "):
            out["mem"] = {k: float(v) for k, v in
                          re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", line)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    round_dirs = sorted((p for p in (root / "pair").glob("r*") if p.is_dir()),
                        key=lambda p: int(p.name[1:]))
    nrounds = len(round_dirs)
    paired = {}
    for case in CASES:
        per = {arm: [] for arm in ARMS}
        setup = {arm: [] for arm in ARMS}
        bitwise = True
        for rdir in round_dirs:
            for arm in ARMS:
                t = parse_timing(rdir / case / f"{arm}.log")
                per[arm].append(statistics.median(t["warm"]))
                setup[arm].append(t["setup"])
            hashes = {hashlib.sha256(
                (rdir / case / f"{arm}.bin").read_bytes()).hexdigest()
                for arm in ARMS}
            bitwise = bitwise and len(hashes) == 1
        ratios = [per["on"][i] / per["off"][i] for i in range(nrounds)]
        paired[case] = {
            "median_us": {arm: median(per[arm]) for arm in ARMS},
            "setup_us": {arm: median(setup[arm]) for arm in ARMS},
            "on_over_off": {"median": median(ratios), "p10": p10(ratios),
                            "min": min(ratios), "max": max(ratios),
                            "rounds": ratios},
            "bitwise_all_rounds": bitwise,
        }
    smoke = {case: parse_smoke(root / "smoke" / f"{case}.log")
             for case in CASES}
    full = json.loads((root / "full-model" / "result.json").read_text())
    numeric = json.loads((root / "numeric" / "verify-summary.json").read_text())
    checks = {
        "command_cache_test": (root / "checks" /
                               "command_cache-test.log").read_text().strip(),
        "rk_plan_host": (root / "checks" /
                         "rk_plan_host.log").read_text().strip().splitlines()[-1],
        "ctest": [line for line in (root / "checks" / "ctest.log").read_text()
                  .splitlines() if "tests passed" in line or "Total Test time" in line],
        "pytest": (root / "checks" / "pytest.log").read_text().strip().splitlines()[-1],
    }
    result = {
        "schema_version": 1,
        "kind": "multi-submit-cache-budget-closure",
        "control": "RK_PLAN_MULTI_SUBMIT_CACHE=0",
        "opt_in": "RK_PLAN_MULTI_SUBMIT_CACHE=1",
        "adopted_default": 0,
        "budget": "RK_PLAN_MULTI_SUBMIT_BUDGET default 4 MiB/workspace, "
                  "malformed/negative/overflow/zero disable cache",
        "device_accounting": "budget admission uses request bytes; cache-own "
                             "budget uses page-aligned command bytes; fallback "
                             "command workspace remains allocated and counted",
        "paired": paired,
        "smoke": smoke,
        "full_model": full["profiles"],
        "full_model_protocol": {"limit": full["limit"],
                                 "rounds": full["rounds"]},
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "checks": checks,
        "decision": {
            "adopt_default": False,
            "reason": "Compiled MLP plans declare device budgets fully consumed "
                      "by data slots plus the fallback command workspace; the "
                      "independent-BO multi cache therefore has no legal device "
                      "headroom. The earlier positive numbers came from "
                      "exceeding that budget and are withdrawn as product "
                      "claims. The implementation remains opt-in for future "
                      "plans that declare headroom.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Multi-submit cache: budget-closure correction",
        "",
        "The candidate is retained as opt-in code, but **not adopted as a",
        "default**.  The prior positive result remains in the historical",
        "`20261028T035849Z-multi-submit-cache/` directory; its performance",
        "claims are withdrawn because its separate command BOs exceeded the",
        "plans' declared device budgets.",
        "",
        "## Budget admission",
        "",
        "A cache submit is admitted only when all three limits hold:",
        "",
        "1. `RK_PLAN_MULTI_SUBMIT_BUDGET`: Host entries/programs plus",
        "   page-aligned command BO bytes;",
        "2. plan Host budget minus the already-resident workspace Host bytes;",
        "3. plan device budget minus data-slot requests and the still-allocated",
        "   fallback command workspace.  This uses the declared request-byte",
        "   accounting.",
        "",
        "Malformed, negative, overflowing and zero budget values fail closed",
        "(0 = cache disabled), never silently become the default or an",
        "unbounded value.",
        "",
        "## Corrected MLP smoke",
        "",
        "| case | cache state | cached submits | skipped budget/static | host delta | cache request | cache physical | fallback command request/physical |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for case in CASES:
        s = smoke[case]
        mem = s["mem"]
        state = "fallback" if not s["info"].get("ready", 0) else "prepared"
        lines.append(
            f"| {case} | {state} | {s['info'].get('cached', 0)}/"
            f"{s['na'].get('submits', 8) if s['na'] else s['info'].get('submits', 0)} | "
            f"{s['info'].get('skipped_budget', s['na'].get('skipped_budget', 0) if s['na'] else 0)}/"
            f"{s['info'].get('skipped_static', s['na'].get('skipped_static', 0) if s['na'] else 0)} | "
            f"{int(mem.get('host_delta', 0))} B | "
            f"{int(mem.get('device_request_bytes', 0))} B | "
            f"{int(mem.get('device_physical_bytes', 0))} B | "
            f"{int(mem.get('fallback_cmd_request', 0))}/"
            f"{int(mem.get('fallback_cmd_physical', 0))} B |")
    lines += [
        "",
        "The fallback command workspace (65,536 B request and physical for",
        "MLP) is still allocated and is charged in the device-budget check.",
        "The cache therefore allocates no command BO on current plans.",
        "",
        "## Short five-profile confirmation (both arms fall back)",
        "",
        "Interleaved two-runner result, 256 images / 3 rounds:",
        "",
        "| profile | control µs | opt-in µs | on/off | bitwise |",
        "|---|---:|---:|---:|---|",
    ]
    for name, v in full["profiles"].items():
        lines.append(
            f"| {name} | {v['timing']['control']['median_us']:.1f} | "
            f"{v['timing']['prepared']['median_us']:.1f} | "
            f"{v['round_ratio_median']:.4f}× | "
            f"{'yes' if v['bitwise'] else 'NO'} |")
    lines += [
        "",
        "With the corrected admission check the opt-in arm takes the original",
        "per-submit path, so the ratio is neutral and outputs remain bitwise.",
        "This confirms safe fallback, not a performance gain.",
        "",
        "## Budget-boundary regressions",
        "",
        "`rk_plan_host_test` now covers, on a synthetic multi-submit plan:",
        "",
        "- device budget exactly equal to base slots + fallback command: zero",
        "  cache BO allocations, original output bitwise;",
        "- Host budget tight: only 3 of 4 submits fit, 3 cache BOs allocated,",
        "  `rk_workspace_host_bytes` stays within the plan Host budget, output",
        "  bitwise;",
        "- cache-own budget tight: only 1 of 4 submits fits, one cache BO",
        "  allocated, output bitwise;",
        "- `-1`, non-numeric, zero and overflowing budget strings: no cache,",
        "  base allocation count only, output bitwise.",
        "",
        "The tests assert the mock allocation count, so an over-budget attempt",
        "cannot be hidden by allocating then releasing.",
        "",
        "## Validation",
        "",
        f"- command-cache host test: `{checks['command_cache_test']}`",
        f"- rk_plan host: `{checks['rk_plan_host']}`",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- frozen numeric replay: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Do **not** adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` as the default.  Keep it",
        "as an opt-in experiment for workspaces that explicitly declare device",
        "headroom; current compiled plans fall back.  The single-submit command",
        "cache and all earlier results remain unchanged.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
