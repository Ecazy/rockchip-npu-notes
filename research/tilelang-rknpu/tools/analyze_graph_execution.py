#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyze the Tiny ViT graph-execution prototype and external references."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROFILES = ("tiny_vit.b8", "tiny_vit.b1")


def median(values):
    rows = [float(value) for value in values if value is not None]
    return float(statistics.median(rows)) if rows else None


def load_external(results: Path, arm: str) -> dict:
    profiles = {}
    walls = []
    for round_dir in sorted(results.glob(f"*{arm}")):
        data = json.loads((round_dir / f"{arm}.json").read_text())
        walls.append(float((round_dir / "wall_seconds.txt").read_text()))
        for row in data.get("profiles", []):
            key = f"{row['model']}.b{int(row['batch'])}"
            calls = int(row["batch_schedule"]["calls"])
            profiles.setdefault(key, []).append({
                "warm_ms": 1e3 * float(row["model_call_seconds"]) / calls,
                "e2e_ms": 1e3 * float(row["end_to_end_seconds"]) / calls,
                "setup_ms": 1e3 * float(row["setup_seconds"]),
                "accuracy": float(row["metrics"]["accuracy"]),
                "top1": float(row["metrics"]["top1_agreement"]),
                "max_abs_logit_error": float(
                    row["metrics"]["max_abs_logit_error"]),
                "n": int(row["n"]),
            })
    out = {"arm": arm, "rounds": len(walls), "wall_median_s": median(walls),
           "profiles": {}}
    for key, rows in profiles.items():
        out["profiles"][key] = {
            field: median(row[field] for row in rows)
            for field in rows[0]
        }
    return out


def write_markdown(summary: dict, path: Path) -> None:
    profiles = summary["profiles"]
    lines = [
        "# Graph-execution prototype",
        "",
        f"- status: **{summary['status']}**",
        f"- full test images: {summary['limit']}",
        f"- rounds: {summary['rounds']}",
        "- application timing: `runner.run(..., stats=None)`",
        "- instrumentation: separate `RunStats` sample, not in app interval",
        "- CPU governor: `performance`, pinned CPU 6 at 2.304 GHz",
        "- NPU: one core at 1.0 GHz",
        "",
        "## Same numeric policy — Tiny ViT",
        "",
        "| profile | generic ms | prepared ms | auto graph ms | "
        "auto/generic | auto/prepared | tasks/submits | view/skip |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for key in PROFILES:
        row = profiles[key]
        generic = row["generic"]["warm_median_ms"]
        prepared = row["prepared"]["warm_median_ms"]
        auto = row["auto"]["warm_median_ms"]
        inst = row["auto"]["instrumented"]["stats_per_call"]
        lines.append(
            f"| {key} | {generic:.3f} | {prepared:.3f} | {auto:.3f} | "
            f"{generic / auto:.3f}x | {prepared / auto:.3f}x | "
            f"{inst['task_count']:.0f}/{inst['submit_count']:.0f} | "
            f"{inst['graph_view_ops']:.0f}/"
            f"{inst['graph_skipped_trace_output_writes']:.0f} |")
    lines += [
        "",
        "Auto executes the compiled flat graph for the high-binding-fanout ViT",
        "profiles. It removes the metadata reshape action, skips two production",
        "writes of trace-only `gate/up` values, and preserves all 8 RKPlan submits,",
        "transfers and device tasks. Generic/prepared are same-policy controls.",
        "",
        "## Application and instrumentation",
        "",
        "| profile | arm | app ms/call | instrumented ms/call | Host actions ms | "
        "C workspace ms | executor ms | setup ms | resident B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in PROFILES:
        for arm in ("generic", "prepared", "auto"):
            row = profiles[key][arm]
            inst = row["instrumented"]["stats_per_call"]
            lines.append(
                f"| {key} | {arm} | {row['warm_median_ms']:.3f} | "
                f"{row['instrumented']['median_ms_per_call']:.3f} | "
                f"{inst['host_action_us']/1e3:.3f} | "
                f"{inst['plan_workspace_us']/1e3:.3f} | "
                f"{inst['executor_overhead_us']/1e3:.3f} | "
                f"{row['runner_setup_ms']:.3f} | {row['resident_bytes']} |")
    lines += [
        "",
        "## External references (separate numeric policies)",
        "",
        "| profile | auto graph ms | vendor ms | CPU FP32 ms | vendor/auto | "
        "CPU/auto | vendor max err | packed atol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in PROFILES:
        auto = profiles[key]["auto"]["warm_median_ms"]
        vendor = summary["external"]["vendor"]["profiles"][key]
        cpu = summary["external"]["cpu"]["profiles"][key]
        lines.append(
            f"| {key} | {auto:.3f} | {vendor['warm_ms']:.3f} | "
            f"{cpu['warm_ms']:.3f} | {auto/vendor['warm_ms']:.3f}x | "
            f"{auto/cpu['warm_ms']:.3f}x | {vendor['max_abs_logit_error']:.5f} | "
            "0.0625 |")
    lines += [
        "",
        "Vendor uses non-quantized FP16 with graph fusion and does not satisfy the",
        "packed per-logit tolerance on ViT. CPU is FP32 reference math. Their times",
        "are references, not a same-policy ranking. The vendor one-call runtime API",
        "does not reveal its internal device Submit count.",
    ]
    if summary.get("independent_numeric_policy"):
        lines += [
            "",
            "## Independent numeric-policy candidate — device FFN",
            "",
            "| profile | same-policy auto ms | device FFN ms | device/policy | "
            "max logit delta | tasks/submits | upload/download B |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for key in PROFILES:
            row = summary["independent_numeric_policy"]["profiles"].get(key)
            if row is None:
                continue
            lines.append(
                f"| {key} | {row['same_policy_ms']:.3f} | "
                f"{row['deviceffn_ms']:.3f} | "
                f"{row['deviceffn_over_same_policy']:.3f}x | "
                f"{row['max_abs_delta_same_policy']:.5f} | "
                f"{row['task_count']:.0f}/{row['submit_count']:.0f} | "
                f"{row['upload_bytes']:.0f}/{row['download_bytes']:.0f} |")
        lines += [
            "",
            "Device FFN uses the existing independent numeric policy; it is not a "
            "drop-in packed-logit path. It removes Host SwiGLU/residual work and "
            "bridge bytes but adds device tasks and submits. The full-test probe "
            "shows a b8 gain and a b1 regression, so it is not included in the "
            "default auto path. Adoption would require an explicit numeric-policy "
            "decision and a uniform per-profile acceptance story.",
        ]
    lines += [
        "",
        "## Host boundary ledger",
        "",
        "| profile | Host actions | Host total ms | provably removed now ms | "
        "view aliases | trace-only tensors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, ledger in summary["boundaries"].items():
        lines.append(
            f"| {key} | {ledger['host_action_count']} | "
            f"{ledger['measured_host_total_ms']:.3f} | "
            f"{ledger['same_policy_removed_total_ms']:.3f} | "
            f"{len(ledger['view_aliases'])} | "
            f"{len(ledger['trace_only_tensors'])} |")
    lines += [
        "",
        "The Host total is observed action dwell, not a recoverable bound.",
        "Numeric Host arithmetic is retained; only the metadata reshape is counted",
        "as directly removable in the ledger. The flat executor's measured gain",
        "also includes eliminating per-node Python/template work and trace-only",
        "production writes, which the action sum does not represent.",
        "",
        "## Problems",
        "",
    ]
    lines += [f"- {problem}" for problem in summary["problems"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--external", type=Path, required=True)
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--independent-benchmark", type=Path,
                    help="optional device-FFN candidate benchmark with "
                         "same-policy and independent-policy arms")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    external = {
        "vendor": load_external(args.external, "vendor"),
        "cpu": load_external(args.external, "cpu"),
    }
    profiles = {}
    for key, arms in benchmark["profiles"].items():
        profiles[key] = {}
        for name, row in arms.items():
            profiles[key][name] = {
                "warm_median_ms": median(
                    item["warm_ms_per_call"] for item in row["rounds"]),
                "warm_rounds_ms": [item["warm_ms_per_call"]
                                   for item in row["rounds"]],
                "paired_vs_generic_median": median(
                    item.get("paired_vs_generic_median")
                    for item in row["rounds"] if name != "generic"),
                "instrumented": row["instrumented"],
                "runner_setup_ms": row["runner_setup_ms"],
                "resident_bytes": row["resident_bytes"],
                "backend": row["backend"],
                "selected_mode": row["selected_mode"],
                "selection": row["selection"],
                "trace_bitwise_equal_generic":
                    row["trace_bitwise_equal_generic"],
            }
    boundaries = {}
    for path in sorted(args.boundaries.glob("*.json")):
        data = json.loads(path.read_text())
        boundaries[data["profile_key"]] = data
    problems = list(benchmark.get("problems", []))
    if benchmark.get("status") != "PASS":
        problems.append("benchmark campaign did not pass")
    for key in PROFILES:
        if key not in profiles:
            problems.append(f"missing profile {key}")
            continue
        if key not in boundaries:
            problems.append(f"missing boundary ledger {key}")
        for arm in ("prepared", "auto"):
            if arm not in profiles[key]:
                problems.append(f"missing arm {arm}/{key}")
                continue
            for rnd in profiles[key][arm]["warm_rounds_ms"]:
                if rnd <= 0:
                    problems.append(f"nonpositive warm interval {arm}/{key}")
        if profiles[key]["auto"]["backend"] != "compiled_graph":
            problems.append(f"auto did not select compiled graph for {key}")
        for arm in external:
            row = external[arm]["profiles"].get(key)
            if row is None or row["n"] != 10000:
                problems.append(f"external {arm}/{key} did not cover 10k")
    independent = None
    if args.independent_benchmark is not None:
        raw = json.loads(args.independent_benchmark.read_text())
        independent = {"status": raw.get("status"),
                       "problems": raw.get("problems", []),
                       "profiles": {}}
        for profile, arms in raw.get("profiles", {}).items():
            if "default" not in arms or "deviceffn" not in arms:
                problems.append(
                    f"independent benchmark {profile} lacks default/deviceffn")
                continue
            default = arms["default"]
            device = arms["deviceffn"]
            default_ms = median(
                item["warm_ms_per_call"] for item in default["rounds"])
            device_ms = median(
                item["warm_ms_per_call"] for item in device["rounds"])
            deltas = [
                item.get("max_abs_delta_generic")
                for item in device["rounds"]
                if item.get("max_abs_delta_generic") is not None]
            independent["profiles"][profile] = {
                "same_policy_ms": default_ms,
                "deviceffn_ms": device_ms,
                "deviceffn_over_same_policy": device_ms / default_ms,
                "accuracy": median(item["accuracy"]
                                   for item in device["rounds"]),
                "max_abs_delta_same_policy": max(deltas) if deltas else None,
                "task_count": device["instrumented"]["stats_per_call"][
                    "task_count"],
                "submit_count": device["instrumented"]["stats_per_call"][
                    "submit_count"],
                "upload_bytes": device["instrumented"]["stats_per_call"][
                    "upload_bytes"],
                "download_bytes": device["instrumented"]["stats_per_call"][
                    "download_bytes"],
                "numeric_policy": "independent device FFN",
            }
    result = {
        "schema_version": 1,
        "kind": "tiny-vit-graph-execution-analysis",
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "limit": benchmark["limit"],
        "rounds": benchmark["rounds"],
        "profiles": profiles,
        "external": external,
        "boundaries": boundaries,
        "independent_numeric_policy": independent,
        "benchmark": benchmark,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(result, args.markdown)
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "problems": problems[:20]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
