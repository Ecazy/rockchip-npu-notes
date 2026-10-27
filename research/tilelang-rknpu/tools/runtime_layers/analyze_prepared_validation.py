#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the same-implementation prepared-validation A/B experiment."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PRIMARY = ("qk_b8", "pv_b8")
OTHER = ("mlp_b1", "mlp_b32", "qkv_b8")
ATTR_FIELDS = ("validate_us", "pack_us", "emit_us", "command_us",
               "submit_alloc_us", "submit_wait_us", "output_sync_us",
               "download_us", "guard_us", "publish_us", "alloc_us",
               "total_us", "pack_layout_us")


def median(values):
    rows = [float(x) for x in values if x is not None]
    return float(statistics.median(rows)) if rows else None


def p10(values):
    rows = sorted(float(x) for x in values)
    return rows[(len(rows) - 1) // 10] if rows else None


def parse_log(path: Path):
    out = {"warm": []}
    for line in path.read_text().splitlines():
        if line.startswith("SETUP_US "):
            out["setup"] = float(line.split()[1])
        elif line.startswith("FIRST_US "):
            out["first"] = float(line.split()[1])
        elif line.startswith("WARM_US "):
            out["warm"] = [float(x) for x in line.split()[1:]]
    return out


def paired_table(root: Path, cases):
    rows = {}
    for case in cases:
        med = {0: [], 1: []}
        ratios = []
        setup = {0: [], 1: []}
        first = {0: [], 1: []}
        for r in range(1, 8):
            vals = {}
            for mode in (0, 1):
                log = parse_log(root / f"r{r}" / case / f"static-{mode}.log")
                vals[mode] = statistics.median(log["warm"])
                setup[mode].append(log.get("setup"))
                first[mode].append(log.get("first"))
            med[0].append(vals[0]); med[1].append(vals[1])
            ratios.append(vals[0] / vals[1])
        rows[case] = {
            "static_0_median_us": median(med[0]),
            "static_1_median_us": median(med[1]),
            "round_ratio_median": median(ratios),
            "round_ratio_p10": p10(ratios),
            "round_ratio_min": min(ratios),
            "round_ratio_max": max(ratios),
            "round_ratios": ratios,
            "setup_us": {str(k): median(v) for k, v in setup.items()},
            "first_us": {str(k): median(v) for k, v in first.items()},
        }
    return rows


def attr_medians(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("ATTR_US "):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) == len(ATTR_FIELDS):
                rows.append(dict(zip(ATTR_FIELDS, vals)))
    return {k: median(row[k] for row in rows) for k in ATTR_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    primary = paired_table(root, PRIMARY)
    other = paired_table(root / "other", OTHER)
    profile = {}
    for case in PRIMARY:
        profile[case] = {}
        for mode in (0, 1):
            profile[case][str(mode)] = attr_medians(
                root / "profile-static" / f"{case}-{mode}.log")
    full = {}
    for name in ("full-model", "full-model-b1"):
        full[name] = json.loads((root / name / "result.json").read_text())
    checks = {}
    for case in PRIMARY:
        checks[case] = json.loads((root / "checks" / f"{case}.json").read_text())
    result = {
        "schema_version": 1,
        "kind": "prepared-validation-same-implementation",
        "primary": primary,
        "other": other,
        "profile": profile,
        "full_model": full,
        "checks": checks,
        "protocol": "B env=0 versus B env=1 in separate processes per round; "
                    "same descriptor/pack/emitter/executor/guard/download/"
                    "publication; 7 rounds, changed-input frames, order rotated. "
                    "Primary estimate is median of per-round medians; p10 is "
                    "auxiliary. Full-model arms are two compiled-graph runners "
                    "in one process, env snapshotted at workspace creation; no "
                    "adapter/pointer rebuilding or node proxies.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    labels = {"qk_b8": "QK group b8", "pv_b8": "PV group b8",
              "mlp_b1": "MLP b1", "mlp_b32": "MLP b32", "qkv_b8": "QKV b8"}
    lines = [
        "# Same-implementation prepared-validation A/B",
        "",
        "This is the corrected causal experiment. Both arms use exactly the",
        "same C implementation, pack, emitter, command workspace, executor,",
        "guard, download and publication. Only the workspace-snapshotted",
        "`RK_PLAN_PREPARED_VALIDATION` flag differs. The flag is experimental",
        "and defaults off; no public ABI or default policy changed.",
        "",
        "Primary estimate is the median of **per-round paired medians**; p10",
        "is auxiliary. Finite input scans remain per call, and changed",
        "bindings fall back to full validation before cache refresh.",
        "",
        "## Isolated cases",
        "",
        "| case | static=0 µs | static=1 µs | per-round ratio median | "
        "ratio range | p10 ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in PRIMARY + OTHER:
        r = primary.get(case) or other[case]
        lines.append(
            f"| {labels[case]} | {r['static_0_median_us']:.1f} | "
            f"{r['static_1_median_us']:.1f} | "
            f"{r['round_ratio_median']:.4f}× | "
            f"{r['round_ratio_min']:.4f}–{r['round_ratio_max']:.4f} | "
            f"{r['round_ratio_p10']:.4f}× |")
    lines += [
        "",
        "## Validation phase split",
        "",
        "Same `battr` path, one sample per arm; fields are µs.",
        "",
        "| case | arm | validate | pack | command | submit/wait | output-sync | "
        "download | guard | total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in PRIMARY:
        for mode in ("0", "1"):
            p = profile[case][mode]
            lines.append(
                f"| {labels[case]} | static={mode} | {p['validate_us']:.1f} | "
                f"{p['pack_us']:.1f} | {p['command_us']:.1f} | "
                f"{p['submit_wait_us']:.1f} | {p['output_sync_us']:.1f} | "
                f"{p['download_us']:.1f} | {p['guard_us']:.1f} | "
                f"{p['total_us']:.1f} |")
    lines += [
        "",
        "The static cache removes only the generic metadata/alias portion;",
        "the finite input scan remains and is the dominant `validate` cost.",
        "For QK/PV the measured validate delta is about 13–14 µs, not the",
        "hundreds of µs previously inferred from a different finite-scan/pack",
        "implementation. The earlier B/C delta therefore cannot be assigned",
        "to the RKPlan metadata layer.",
        "",
        "## Full model",
        "",
        "| model | current median µs | experimental median µs | "
        "per-round ratio median | p10 ratio | bitwise |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, label in (("full-model", "Tiny ViT b8"),
                        ("full-model-b1", "Tiny ViT b1")):
        f = full[name]
        c = f["trusted_timing"]["current"]["median_us"]
        e = f["trusted_timing"]["experimental"]["median_us"]
        lines.append(
            f"| {label} | {c:.1f} | {e:.1f} | "
            f"{f['round_ratio_median']:.4f}× | {f['round_ratio_p10']:.4f}× | "
            f"{'yes' if f['bitwise'] else 'NO'} |")
    lines += [
        "",
        "The b8 arm shows a small but repeatable gain; b1 does not. The b1",
        "result is included only after the b8 gain was observed.",
        "",
        "## Safety checks",
        "",
    ]
    for case, c in checks.items():
        lines.append(
            f"- {labels[case]}: status **{c['status']}**; wrong shape rejected "
            f"{c['rejected_shape']}; non-finite rejected {c['rejected_finite']}; "
            f"output unchanged on both failures {c['shape_output_unchanged']}/"
            f"{c['finite_output_unchanged']}; address change revalidated "
            f"{c['address_revalidated']}; restored output unchanged "
            f"{c['restored_output_unchanged']}.")
    lines += [
        "",
        "## Decision",
        "",
        "The static-only cache gives only a small b8 gain and essentially no",
        "b1 gain. It is not enabled by default and is not productized in this",
        "stage. Guard, synchronization, finite checks and publication remain",
        "unchanged. A future product decision would need a separate safety/",
        "lifetime contract; the observed gain alone is not sufficient.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
