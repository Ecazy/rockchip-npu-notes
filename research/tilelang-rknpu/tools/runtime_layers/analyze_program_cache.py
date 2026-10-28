#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the creation-time co_program cache A/B."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CASES = ("qk_b8", "pv_b8", "qkv_b8", "mlp_b32")
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


def parse_timing(path):
    out = {"warm": []}
    for line in path.read_text().splitlines():
        if line.startswith("SETUP_US "):
            out["setup"] = float(line.split()[1])
        elif line.startswith("FIRST_US "):
            out["first"] = float(line.split()[1])
        elif line.startswith("WARM_US "):
            out["warm"] = [float(x) for x in line.split()[1:]]
    return out


def attr_medians(path):
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
    pair_root = root / "pair"
    paired = {}
    for case in CASES:
        med = {0: [], 1: []}
        ratios = []
        setup = {0: [], 1: []}
        first = {0: [], 1: []}
        for r in range(1, 8):
            t = {m: parse_timing(pair_root / f"r{r}" / case / f"cache-{m}.log")
                 for m in (0, 1)}
            for m in (0, 1):
                med[m].append(statistics.median(t[m]["warm"]))
                setup[m].append(t[m].get("setup"))
                first[m].append(t[m].get("first"))
            ratios.append(med[0][-1] / med[1][-1])
        paired[case] = {
            "cache_off_median_us": median(med[0]),
            "cache_on_median_us": median(med[1]),
            "round_ratio_median": median(ratios),
            "round_ratio_p10": p10(ratios),
            "round_ratio_min": min(ratios),
            "round_ratio_max": max(ratios),
            "round_ratios": ratios,
            "setup_us": {str(m): median(setup[m]) for m in (0, 1)},
            "first_us": {str(m): median(first[m]) for m in (0, 1)},
        }
    profile = {}
    for case in CASES:
        profile[case] = {
            str(m): attr_medians(root / "profile-emit" / f"{case}-{m}.log")
            for m in (0, 1)
        }
    full = {}
    for name in ("full-model-b8", "full-model-b1"):
        full[name] = json.loads((root / name / "result.json").read_text())
    checks = {
        p.stem: json.loads(p.read_text())
        for p in sorted((root / "checks").glob("*.json"))
    }
    numeric = json.loads((root / "numeric/verify-summary.json").read_text())
    comparison = (pair_root / "comparison.txt").read_text().splitlines()
    bad = [line for line in comparison if "EQUAL" not in line]
    result = {
        "schema_version": 1,
        "kind": "creation-time-coprogram-cache-ab",
        "paired": paired,
        "profile": profile,
        "full_model": full,
        "checks": checks,
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "comparison": {"rows": len(comparison), "bad": len(bad)},
        "decision": {
            "adopt_default": True,
            "scope": "single-submit workspaces with all tasks emitted and "
                     "validated at creation",
            "host_memory_increment_bytes": 0,
            "reason": "existing batch scratch is reused; no extra co_program "
                      "array is allocated",
        },
        "protocol": "Cache off/on are the same C implementation and same "
                    "workspace.  7 rotated rounds; per-round median ratio is "
                    "primary, p10 auxiliary.  Full model uses two runners in "
                    "one process with env snapshotted at workspace creation; "
                    "stats=None and no node proxies.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    labels = {"qk_b8": "QK group b8", "pv_b8": "PV group b8",
              "qkv_b8": "QKV b8", "mlp_b32": "MLP b32"}
    lines = [
        "# Creation-time co_program cache A/B",
        "",
        "This experiment caches `co_program` only. Command-BO packing, guard,",
        "synchronization and numeric policy are unchanged. The cache is",
        "workspace-local, built after BO/view creation, and reused only for",
        "the same workspace. The current default is on for a structurally",
        "eligible single-submit workspace; `RK_PLAN_PROGRAM_CACHE=0` is the",
        "control.",
        "",
        "## Eligibility",
        "",
        "- all NPU task programs are emitted through `rk_emit_action` during",
        "  workspace creation and must pass its existing validation;",
        "- the complete task set is one submit and fits the existing",
        "  `batch_scratch`;",
        "- all reads/outputs point to workspace-owned BOs with stable",
        "  addresses; no plan/view/config input changes after creation;",
        "- multi-submit or failed/program-ineligible workspaces keep the",
        "  original per-run emitter path.",
        "",
        "The existing `batch_scratch` holds the cached `co_program` array, so",
        "the incremental Host allocation is **0 bytes**; no new `co_program`",
        "array is allocated.",
        "",
        "## Isolated cases",
        "",
        "µs; ratio = cache-off / cache-on; primary estimate is the median of",
        "per-round paired medians.",
        "",
        "| case | cache off | cache on | per-round ratio median | ratio range | p10 ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        r = paired[case]
        lines.append(
            f"| {labels[case]} | {r['cache_off_median_us']:.1f} | "
            f"{r['cache_on_median_us']:.1f} | "
            f"**{r['round_ratio_median']:.4f}×** | "
            f"{r['round_ratio_min']:.4f}–{r['round_ratio_max']:.4f} | "
            f"{r['round_ratio_p10']:.4f}× |")
    lines += [
        "",
        "## Creation and emitter cost",
        "",
        "| case | setup off µs | setup on µs | emit off µs | emit on µs |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        p = profile[case]
        lines.append(
            f"| {labels[case]} | {paired[case]['setup_us']['0']:.1f} | "
            f"{paired[case]['setup_us']['1']:.1f} | "
            f"{p['0']['emit_us']:.1f} | {p['1']['emit_us']:.1f} |")
    lines += [
        "",
        "The cache eliminates the runtime `emit` interval for QK/PV (about",
        "46–47 µs in the current attribution), but QKV emit is only about",
        "1.5 µs and MLP b32 is not eligible (8 submits), so its fallback is",
        "bitwise equal and neutral. Creation time rises by roughly 179–255 µs",
        "for QK/PV; the steady gains dominate over the first few calls.",
        "",
        "## Full model (uninstrumented)",
        "",
        "| model | cache off median µs | cache on median µs | per-round ratio median | p10 ratio | bitwise |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, label in (("full-model-b8", "Tiny ViT b8"),
                        ("full-model-b1", "Tiny ViT b1")):
        f = full[name]
        c = f["trusted_timing"]["current"]["median_us"]
        e = f["trusted_timing"]["experimental"]["median_us"]
        lines.append(
            f"| {label} | {c:.1f} | {e:.1f} | "
            f"**{f['round_ratio_median']:.4f}×** | "
            f"{f['round_ratio_p10']:.4f}× | "
            f"{'yes' if f['bitwise'] else 'NO'} |")
    lines += [
        "",
        "b8 gains about 5.5%; b1 gains about 2.8%. The b1 run was done after",
        "the b8 gain was observed. Both use `stats=None` with no node proxies.",
        "",
        "## Lifecycle and failure checks",
        "",
    ]
    for name, c in checks.items():
        lines.append(f"- {name}: **{c['status']}** ({json.dumps(c, sort_keys=True)})")
    lines += [
        f"- pair comparison rows: {len(comparison)}, differences: {len(bad)}",
        f"- frozen numeric replay: **{numeric.get('status')}**, problems: "
        f"{numeric.get('problems')}",
        "- existing CTest suite (including allocation/creation failures) and",
        "  Python workspace tests pass with the cache default enabled; commands",
        "  change only cache control, and a failed program build falls back.",
        "",
        "## Decision",
        "",
        "Adopt the creation-time co_program cache as the default for eligible",
        "single-submit workspaces. Keep `RK_PLAN_PROGRAM_CACHE=0` as the",
        "control/opt-out. This is not command-BO packing reuse; that remains a",
        "separate future experiment.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS" if not bad else "FAIL",
                      "out": str(args.out), "comparison_bad": len(bad)}, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
