#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit an inspectable Host-boundary ledger for a frozen hybrid graph.

Every Host action is reported with the reason it exists, its measured Host time
when a campaign ledger is supplied, the part that the current compiled-graph
lowering actually removes, and the missing device templates for a different
numeric-policy route.  Host compute is never assumed to be fully recoverable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.passes.graph_execution import analyze_graph_execution


def _host_times(analysis: Path | None, benchmark: Path | None,
                profile: str) -> dict[str, float]:
    if benchmark is not None:
        payload = json.loads(benchmark.read_text())
        row = payload.get("profiles", {}).get(profile, {}).get(
            "prepared", {})
        actions = row.get("instrumented", {}).get("host_actions", {})
        return {name: float(value) / 1e3 for name, value in actions.items()}
    if analysis is None:
        return {}
    payload = json.loads(analysis.read_text())
    row = payload.get("profiles", {}).get(profile, {}).get("auto", {})
    return {name: float(value) / 1e3
            for name, value in row.get("host_actions", {}).items()}


def _write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Host-boundary ledger",
        "",
        f"- graph: `{result['manifest']}`",
        f"- profile ledger: `{result['profile_key']}`",
        "",
        "| node | action | causes | measured ms | removable now ms | "
        "trace-only outputs | trace-write estimate ms | necessary added cost | "
        "graph device gap |",
        "|---|---|---|---:|---:|---|---:|---|---|",
    ]
    for row in result["boundaries"]:
        lines.append(
            f"| {row['node']} | `{row['action']}` | "
            f"{', '.join(row['causes'])} | "
            f"{row['measured_host_ms']:.3f} | "
            f"{row['same_policy_removable_ms']:.3f} | "
            f"{', '.join(row['trace_only_outputs']) or '-'} | "
            f"{'; '.join(row['device_missing']) or '-'} |")
    lines += [
        "",
        "`same_policy_removable_ms` is zero unless the current prototype",
        "provably removes the action (metadata reshape view). Retention of",
        "numeric Host arithmetic is not counted as loss. Trace-only writes are",
        "reported as a counter/bytes estimate but not added to removable time.",
        "",
        "## View aliases",
        "",
    ]
    if result["view_aliases"]:
        for alias in result["view_aliases"]:
            lines.append(
                f"- `{alias['node']}`: `{alias['input']}` -> "
                f"`{alias['output']}` shape `{alias['shape']}`")
    else:
        lines.append("- none")
    lines += ["", "## Summary", ""]
    lines.append(
        f"- production tensors: {len(result['production_tensors'])}; "
        f"trace-only tensors: {len(result['trace_only_tensors'])}")
    lines.append(
        f"- Host actions: {result['host_action_count']}; "
        f"view-elided: {len(result['view_aliases'])}; "
        f"measured Host total: {result['measured_host_total_ms']:.3f} ms")
    lines.append(
        f"- explicitly removed by current same-policy prototype: "
        f"{result['same_policy_removed_total_ms']:.3f} ms "
        "(not the whole Host total)")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--profile-key", default="")
    ap.add_argument("--analysis", type=Path,
                    help="execution-candidate analysis.json for Host timings")
    ap.add_argument("--benchmark", type=Path,
                    help="graph-execution benchmark.json for Host timings")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    program = load_hybrid_program(
        args.plans_dir / f"{args.manifest}.rkhybrid.json",
        plans_dir=args.plans_dir)
    plan = analyze_graph_execution(program)
    profile_key = args.profile_key or f"{program.model}.b{program.batch}"
    timings = _host_times(args.analysis, args.benchmark, profile_key)
    rows = []
    for boundary in plan.boundaries:
        if boundary.capture_only:
            continue
        measured = float(timings.get(boundary.node_id, 0.0))
        removed = measured if boundary.lowered_as_view else 0.0
        trace_bytes = sum(program.tensors[name].nbytes
                          for name in boundary.trace_only_outputs)
        rows.append({
            "node": boundary.node_id,
            "action": boundary.action_id,
            "phase": boundary.phase,
            "causes": list(boundary.causes),
            "measured_host_ms": measured,
            "same_policy_removable_ms": removed,
            "outputs": list(boundary.output_tensors),
            "trace_only_outputs": list(boundary.trace_only_outputs),
            "trace_only_write_bytes": trace_bytes,
            "trace_only_write_estimate_ms": trace_bytes / 8e6,
            "device_missing": list(boundary.device_missing),
            "lowered_as_view": boundary.lowered_as_view,
            "retained_numeric_work": not boundary.lowered_as_view,
            "necessary_added_cost": (
                "one-time view construction; no per-call arithmetic/copy"
                if boundary.lowered_as_view else
                "unpriced without a validated device/layout recipe; current "
                "Host compute and any required conversion/submit remain"),
        })
    result = {
        "schema_version": 1,
        "kind": "graph-host-boundary-ledger",
        "manifest": args.manifest,
        "profile_key": profile_key,
        "boundaries": rows,
        "view_aliases": [{
            "node": item.node_id, "input": item.input_tensor,
            "output": item.output_tensor, "shape": list(item.shape)}
            for item in plan.view_aliases],
        "production_tensors": list(plan.production_tensors),
        "trace_only_tensors": list(plan.trace_only_tensors),
        "host_action_count": len(rows),
        "measured_host_total_ms": sum(row["measured_host_ms"] for row in rows),
        "same_policy_removed_total_ms": sum(
            row["same_policy_removable_ms"] for row in rows),
        "trace_only_output_bytes": sum(
            row["trace_only_write_bytes"] for row in rows),
        "trace_only_write_estimate_ms": sum(
            row["trace_only_write_estimate_ms"] for row in rows),
        "disclosure": (
            "Host total is the observed action dwell, not a recoverable bound. "
            "Only metadata-only view elimination is counted as same-policy "
            "removal here; numeric Host work remains unless a new validated "
            "device recipe and numeric policy are accepted."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(result, args.markdown)
    print(json.dumps({
        "status": "ok", "out": str(args.out),
        "host_actions": result["host_action_count"],
        "view_aliases": len(result["view_aliases"]),
        "removed_ms": result["same_policy_removed_total_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
