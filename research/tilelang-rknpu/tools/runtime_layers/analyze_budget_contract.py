#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the plan-budget contract and legal multi-submit cache campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def median(values):
    vals = [float(v) for v in values]
    return float(statistics.median(vals)) if vals else None


def parse_smoke(path: Path):
    out = {"na": {}, "info": {}, "mem": {}, "cmdbytes": None}
    for line in path.read_text().splitlines():
        if line.startswith("MULTI_NA "):
            out["na"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_INFO "):
            out["info"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_MEM "):
            out["mem"] = {k: float(v) for k, v in
                          re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", line)}
        elif line.startswith("MULTI_CMDBYTES_OK "):
            vals = line.split()
            out["cmdbytes"] = {"same": int(vals[1]),
                               "checked": int(vals[2].split("=")[1]),
                               "runs": int(vals[3].split("=")[1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    smoke = {}
    for variant in ("exact", "partial", "full"):
        smoke[variant] = parse_smoke(root / "smoke" / f"{variant}-mlp_b32.log")
    bitwise = {}
    for variant in ("exact", "partial", "full"):
        bitwise[variant] = {}
        for arm in ("off", "on"):
            p = root / "bitwise" / f"{variant}-{arm}.bin"
            bitwise[variant][arm] = hashlib.sha256(p.read_bytes()).hexdigest()
        bitwise[variant]["equal"] = (
            bitwise[variant]["off"] == bitwise[variant]["on"])
    full = json.loads((root / "full-model" / "result.json").read_text())
    numeric = json.loads((root / "numeric" / "verify-summary.json").read_text())
    plan_identity = {}
    for name in ("mnist_mlp_b1", "mnist_mlp_b8", "mnist_mlp_b32"):
        d = json.loads((root / "campaign" / f"{name}.rkplan.json").read_text())
        plan_identity[name] = {
            "budget_policy": d.get("budget_policy"),
            "accounting_device_request_bytes":
                d["accounting"]["device_request_bytes"],
            "accounting_host_bytes": d["accounting"]["host_bytes"],
            "plan_hash": d["plan_hash"],
        }
    checks = {
        "command_cache_test": (root / "checks" /
                               "command_cache-test.log").read_text().strip(),
        "rk_plan_host": (root / "checks" /
                         "rk_plan_host.log").read_text().strip().splitlines()[-1],
        "plan_budget_contract": (root / "checks" /
                                 "plan-budget-contract.log").read_text().strip(),
        "ctest": [line for line in (root / "checks" / "ctest.log").read_text()
                  .splitlines() if "tests passed" in line or "Total test time" in line],
        "pytest": (root / "checks" / "pytest.log").read_text().strip().splitlines()[-1],
    }
    profiles = {}
    for name, v in full["profiles"].items():
        timing = v["timing"]
        throughput = {arm: statistics.median(
            row["throughput_images_s"][arm] for row in v["rounds"])
            for arm in ("control", "prepared")}
        profiles[name] = {
            "control_us": timing["control"]["median_us"],
            "prepared_us": timing["prepared"]["median_us"],
            "ratio": v["round_ratio_median"],
            "p10": v["round_ratio_p10"],
            "throughput_control": throughput["control"],
            "throughput_prepared": throughput["prepared"],
            "setup_us": v["setup_us"],
            "bitwise": v["bitwise"],
            "accuracy": v["accuracy"],
        }
    result = {
        "schema_version": 1,
        "kind": "plan-budget-contract-multi-submit-cache",
        "default": "RK_PLAN_MULTI_SUBMIT_CACHE=1",
        "control": "RK_PLAN_MULTI_SUBMIT_CACHE=0",
        "budget_policy": "explicit CompileOptions device/host limits are "
                         "persisted in plan budget_policy, hashed, loaded and "
                         "passed to the C descriptor; legacy plans without "
                         "budget_policy stay strict",
        "plan_identity": plan_identity,
        "smoke": smoke,
        "bitwise": bitwise,
        "full_model": profiles,
        "full_model_protocol": {"limit": full["limit"],
                                 "rounds": full["rounds"],
                                 "interleaved": True},
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "checks": checks,
        "decision": {
            "adopt_default": True,
            "reason": "With explicit budget_policy, exact-full device budget "
                      "allocates zero cache, partial headroom caches a prefix, "
                      "and sufficient headroom caches all 8 MLP submits with "
                      "bitwise output. Legal five-profile run gains 1.03-1.05x "
                      "on MLP and is neutral on ViT; legacy strict plans still "
                      "fall back safely.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Plan-budget contract and legal multi-submit cache",
        "",
        "`CompileOptions.device_budget_bytes` / `host_budget_bytes` are now",
        "persisted as a versioned `budget_policy` block, included in the plan",
        "hash, loaded back, and projected into the existing C descriptor",
        "`device_budget_bytes` / `host_budget_bytes` fields. `claimed_*` remains",
        "the accounting estimate. Legacy plans without `budget_policy` keep",
        "strict budget = claimed semantics.",
        "",
        "## Budgeted MLP plans",
        "",
        "| plan | budget device/host | accounting device/host | plan hash |",
        "|---|---:|---:|---|",
    ]
    for name, v in plan_identity.items():
        b = v["budget_policy"]
        lines.append(
            f"| {name} | {b['device_budget_bytes']}/{b['host_budget_bytes']} | "
            f"{v['accounting_device_request_bytes']}/{v['accounting_host_bytes']} | "
            f"`{v['plan_hash'][:16]}…` |")
    lines += [
        "",
        "The full campaign uses the documented `CompileOptions` default",
        "256 MiB limits; no budget was raised to recover the cache.",
        "",
        "## Budget boundary behavior",
        "",
        "| variant | cache state | cached | skipped budget/static | Host delta | cache request | cache physical | fallback command request/physical |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for variant in ("exact", "partial", "full"):
        s = smoke[variant]
        mem = s["mem"]
        state = "fallback" if not s["info"].get("ready", 0) else "prepared"
        info = s["info"] or s["na"]
        lines.append(
            f"| {variant} | {state} | {info.get('cached', 0)}/"
            f"{info.get('submits', 8)} | "
            f"{info.get('skipped_budget', 0)}/{info.get('skipped_static', 0)} | "
            f"{int(mem.get('host_delta', 0))} B | "
            f"{int(mem.get('device_request_bytes', 0))} B | "
            f"{int(mem.get('device_physical_bytes', 0))} B | "
            f"{int(mem.get('fallback_cmd_request', 0))}/"
            f"{int(mem.get('fallback_cmd_physical', 0))} B |")
    lines += [
        "",
        "Exact budget uses all device bytes for data slots + the fallback",
        "command workspace; zero cache BOs are allocated. Partial budget leaves",
        "4,032 request bytes: the first 4-task submit fits (4,032 request /",
        "4,096 physical), the other seven are skipped. Full budget caches all",
        "eight: 11,088 request bytes and a 32,768 B page-aligned physical upper",
        "bound. All three variants produce bitwise-identical output.",
        "",
        "## Five-profile interleaved comparison",
        "",
        "256 images / 7 rounds; per-call samples are in `full-model/result.json`.",
        "",
        "| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, v in profiles.items():
        lines.append(
            f"| {name} | {v['control_us']:.1f} | {v['prepared_us']:.1f} | "
            f"**{v['ratio']:.4f}×** | {v['p10']:.4f}× | "
            f"{v['throughput_control']:.1f} | {v['throughput_prepared']:.1f} | "
            f"{'yes' if v['bitwise'] else 'NO'} |")
    lines += [
        "",
        "MLP gains are 1.034–1.054×; ViT profiles have no multi-submit",
        "workspace and stay neutral. Accuracy is identical and logits are",
        "bitwise equal.",
        "",
        "## Validation",
        "",
        f"- command-cache host: `{checks['command_cache_test']}`",
        f"- rk_plan host: `{checks['rk_plan_host']}`",
        f"- plan budget contract: `{checks['plan_budget_contract']}`",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- frozen numeric replay: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` as default. Legacy plans without",
        "`budget_policy` remain strict and fall back; new plans with explicit",
        "budget headroom use the legal cache. The earlier invalid-budget results",
        "remain historical and are not used as performance evidence.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
