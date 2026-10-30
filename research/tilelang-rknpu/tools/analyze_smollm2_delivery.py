#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize layered SmolLM2 decode profiles and interleaved A/B runs.

All per-forward values use the actual ``decode_forwards`` count.  Profile
subdivisions are reported with explicit parent/child relationships and measured
residuals; they are never silently added across levels.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def median(values):
    return float(statistics.median(values)) if values else 0.0


def load_run(path: Path) -> dict[str, Any]:
    return json.loads((path / "run-00" / "result.json").read_text())


def per_forward(values, forward_counts):
    """Convert microsecond samples to milliseconds per actual forward."""
    return median([float(v) / float(n) / 1000.0
                   for v, n in zip(values, forward_counts)])


def profile_summary(root: Path) -> dict[str, Any]:
    run = json.loads((root / "run.json").read_text())
    result = load_run(root)
    rows = result["runs"]
    profiles = [r["profile"] for r in rows]
    forwards = [int(r["decode_forwards"]) for r in rows]
    if any(n != 31 for n in forwards):
        raise SystemExit(f"unexpected decode forward counts: {forwards}")
    for p, n in zip(profiles, forwards):
        if int(p["decode_forwards"]) != n:
            raise SystemExit("profile forward count differs from result")
    top = {
        name: per_forward([p["top_level"][name] for p in profiles], forwards)
        for name in profiles[0]["top_level"]
    }
    layer_ops = {
        name: per_forward([p["layer_ops"][name] for p in profiles], forwards)
        for name in profiles[0]["layer_ops"]
    }
    attention_ops = {
        name: per_forward([p["attention_ops"][name] for p in profiles],
                          forwards)
        for name in profiles[0]["attention_ops"]
    }
    plan_node = {
        name: per_forward([p["plan_node_subdivision"][name] for p in profiles],
                          forwards)
        for name in profiles[0]["plan_node_subdivision"]
    }
    plan_call = {
        name: per_forward([p["plan_call_subdivision"][name] for p in profiles],
                          forwards)
        for name in profiles[0]["plan_call_subdivision"]
    }
    c_runtime = {
        name: per_forward([p["c_runtime_subdivision"][name] for p in profiles],
                          forwards)
        for name in profiles[0]["c_runtime_subdivision"]
    }
    decode_ms = per_forward([p["decode_wall_us"] for p in profiles], forwards)
    top_sum = sum(top.values())
    top_residual = per_forward(
        [p["top_level_residual_us"] for p in profiles], forwards)
    c_total = per_forward([p["c_runtime_total_us"] for p in profiles],
                          forwards)
    c_child_sum = per_forward([p["c_runtime_children_sum_us"] for p in profiles],
                              forwards)
    c_residual = per_forward([p["c_runtime_residual_us"] for p in profiles],
                             forwards)
    # The wrapper's measured C-call wall can exceed the C library's own
    # reported total.  Keep that Python/ctypes seam visible instead of folding
    # it into a C-runtime child.
    c_wrapper = plan_call["c_runtime"]
    c_python_seam = max(0.0, c_wrapper - c_total)
    return {
        "backend_options": run.get("backend_options", {}),
        "runs": len(rows),
        "decode_forwards_per_run": forwards,
        "decode_ms_per_forward": decode_ms,
        "top_level_ms_per_forward": top,
        "top_level_sum_ms_per_forward": top_sum,
        "top_level_residual_ms_per_forward": top_residual,
        "top_level_residual_fraction": (
            top_residual / decode_ms if decode_ms else 0.0),
        "layer_ops_ms_per_forward": layer_ops,
        "layer_ops_sum_ms_per_forward": sum(layer_ops.values()),
        "layer_residual_ms_per_forward": per_forward(
            [p["layer_residual_us"] for p in profiles], forwards),
        "attention_ops_ms_per_forward": attention_ops,
        "attention_ops_sum_ms_per_forward": sum(attention_ops.values()),
        "attention_residual_ms_per_forward": per_forward(
            [p["attention_residual_us"] for p in profiles], forwards),
        "plan_node_ms_per_forward": plan_node,
        "plan_node_sum_ms_per_forward": sum(plan_node.values()),
        "plan_call_ms_per_forward": plan_call,
        "plan_call_sum_ms_per_forward": sum(plan_call.values()),
        "c_runtime_total_ms_per_forward": c_total,
        "c_runtime_children_ms_per_forward": c_runtime,
        "c_runtime_children_sum_ms_per_forward": c_child_sum,
        "c_runtime_residual_ms_per_forward": c_residual,
        "c_runtime_python_seam_ms_per_forward": c_python_seam,
        "host_action_ms_per_forward": per_forward(
            [p["host_action_us"] for p in profiles], forwards),
        "peak_rss_kb": int(run.get("process", {}).get("ru_maxrss_kb", 0)),
        "load_us": float(run.get("load_us", 0.0)),
        "generated_ids": [r.get("generated_ids") for r in rows],
    }


def ab_summary(root: Path) -> dict[str, Any]:
    rows = []
    for mode in ("baseline", "candidate"):
        for pair in (1, 2, 3):
            p = root / f"pair{pair}-{mode}" / "run-00" / "result.json"
            if not p.is_file():
                raise SystemExit(f"missing A/B result {p}")
            d = json.loads(p.read_text())
            r = d["runs"][0]
            run_meta = json.loads((p.parent.parent / "run.json").read_text())
            rows.append({
                "pair": pair,
                "mode": mode,
                "ttft_ms": float(r["ttft_us"]) / 1000.0,
                "generate_wall_ms": float(r["generate_wall_us"]) / 1000.0,
                "decode_ms": float(r["decode_us"]) / 1000.0,
                "decode_forwards": int(r["decode_forwards"]),
                "output_tokens": int(r["output_tokens"]),
                "output_tokens_per_s": (
                    float(r["output_tokens"]) /
                    (float(r["generate_wall_us"]) / 1e6)),
                "decode_forward_per_s": (
                    float(r["decode_forwards"]) /
                    (float(r["decode_us"]) / 1e6)),
                "peak_rss_kb": int(run_meta.get("process", {}).get(
                    "ru_maxrss_kb", 0)),
                "final_vmrss_kb": int(str(run_meta.get("process", {}).get(
                    "vmrss", "0 kB")).split()[0]),
                "generated_ids": r["generated_ids"],
            })
    by_mode = {}
    for mode in ("baseline", "candidate"):
        m = [x for x in rows if x["mode"] == mode]
        by_mode[mode] = {
            "runs": len(m),
            "ttft_ms_median": median([x["ttft_ms"] for x in m]),
            "generate_wall_ms_median": median(
                [x["generate_wall_ms"] for x in m]),
            "decode_ms_median": median([x["decode_ms"] for x in m]),
            "decode_ms_per_forward": median(
                [x["decode_ms"] / x["decode_forwards"] for x in m]),
            "decode_forward_per_s_median": median(
                [x["decode_forward_per_s"] for x in m]),
            "output_tokens_per_s_median": median(
                [x["output_tokens_per_s"] for x in m]),
            "peak_rss_kb_median": median([x["peak_rss_kb"] for x in m]),
            "final_vmrss_kb_median": median(
                [x["final_vmrss_kb"] for x in m]),
            "generated_ids_equal": all(
                x["generated_ids"] == m[0]["generated_ids"] for x in m),
        }
    b, c = by_mode["baseline"], by_mode["candidate"]
    return {
        "rows": rows,
        "baseline": b,
        "candidate": c,
        "candidate_vs_baseline": {
            "generate_wall_speedup": (
                b["generate_wall_ms_median"] / c["generate_wall_ms_median"]),
            "decode_speedup": (
                b["decode_ms_median"] / c["decode_ms_median"]),
            "generate_wall_reduction_fraction": (
                1.0 - c["generate_wall_ms_median"] /
                b["generate_wall_ms_median"]),
            "decode_reduction_fraction": (
                1.0 - c["decode_ms_median"] / b["decode_ms_median"]),
            "added_peak_rss_kb": c["peak_rss_kb_median"] -
                                 b["peak_rss_kb_median"],
            "added_final_vmrss_kb": c["final_vmrss_kb_median"] -
                                    b["final_vmrss_kb_median"],
            "generated_ids_equal_between_modes": all(
                x["generated_ids"] ==
                next(y["generated_ids"] for y in rows if y["mode"] ==
                     "baseline") for x in rows if x["mode"] == "candidate"),
        },
    }


def markdown(profile: dict, ab: dict) -> str:
    b = profile["baseline"]
    c = profile["candidate"]
    lines = [
        "# SmolLM2 V5 delivery latency and host-weight-cache report",
        "",
        "All values are medians across independent 32-token generations. "
        "Per-forward values divide by the actual 31 decode forwards.",
        "",
        "## Layered decode profile (profiled runs)",
        "",
        "| Component | Baseline ms/forward | Candidate ms/forward |",
        "|---|---:|---:|",
    ]
    labels = [
        ("decode wall", "decode_ms_per_forward"),
        ("embed (runner top)", None),
        ("30 layers total", None),
        ("final RMSNorm", None),
        ("lm_head plans total", None),
        ("logits concat", None),
        ("top-level measured residual", "top_level_residual_ms_per_forward"),
        ("PlanCall wrapper total", None),
        ("C runtime reported total", "c_runtime_total_ms_per_forward"),
        ("Host actions (cross-cut)", "host_action_ms_per_forward"),
    ]
    for label, key in labels:
        if key:
            lines.append(f"| {label} | {b[key]:.3f} | {c[key]:.3f} |")
        elif label == "embed (runner top)":
            lines.append(f"| {label} | {b['top_level_ms_per_forward']['embed']:.3f} | {c['top_level_ms_per_forward']['embed']:.3f} |")
        elif label == "30 layers total":
            lines.append(f"| {label} | {b['top_level_ms_per_forward']['layer_total']:.3f} | {c['top_level_ms_per_forward']['layer_total']:.3f} |")
        elif label == "final RMSNorm":
            lines.append(f"| {label} | {b['top_level_ms_per_forward']['final_norm']:.3f} | {c['top_level_ms_per_forward']['final_norm']:.3f} |")
        elif label == "lm_head plans total":
            lines.append(f"| {label} | {b['top_level_ms_per_forward']['lm_head_total']:.3f} | {c['top_level_ms_per_forward']['lm_head_total']:.3f} |")
        elif label == "logits concat":
            lines.append(f"| {label} | {b['top_level_ms_per_forward']['concat']:.3f} | {c['top_level_ms_per_forward']['concat']:.3f} |")
        elif label == "PlanCall wrapper total":
            lines.append(f"| {label} | {b['plan_call_ms_per_forward']['c_runtime']:.3f} | {c['plan_call_ms_per_forward']['c_runtime']:.3f} |")
    lines += [
        "",
        "## Layer subdivision (candidate, ms/forward)",
        "",
        "| Operation | ms/forward |",
        "|---|---:|",
    ]
    for k, v in c["layer_ops_ms_per_forward"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += [
        f"| measured layer residual | {c['layer_residual_ms_per_forward']:.3f} |",
        f"| attention residual | {c['attention_residual_ms_per_forward']:.3f} |",
        "",
        "## PlanCall / C runtime seam (candidate, ms/forward)",
        "",
        "| Segment | ms/forward |",
        "|---|---:|",
    ]
    for k, v in c["plan_call_ms_per_forward"].items():
        lines.append(f"| plan_call.{k} | {v:.3f} |")
    lines += [
        f"| C runtime reported total | {c['c_runtime_total_ms_per_forward']:.3f} |",
        f"| C children sum | {c['c_runtime_children_sum_ms_per_forward']:.3f} |",
        f"| C measured residual | {c['c_runtime_residual_ms_per_forward']:.3f} |",
        f"| Python/ctypes seam (wrapper minus C total) | {c['c_runtime_python_seam_ms_per_forward']:.3f} |",
        "",
        "C children:",
        "",
        "| C interval | ms/forward |",
        "|---|---:|",
    ]
    for k, v in c["c_runtime_children_ms_per_forward"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += [
        "",
        "## Interleaved final A/B (no profiler)",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| TTFT median ms | {ab['baseline']['ttft_ms_median']:.3f} | {ab['candidate']['ttft_ms_median']:.3f} |",
        f"| generate wall median ms | {ab['baseline']['generate_wall_ms_median']:.3f} | {ab['candidate']['generate_wall_ms_median']:.3f} |",
        f"| decode median ms | {ab['baseline']['decode_ms_median']:.3f} | {ab['candidate']['decode_ms_median']:.3f} |",
        f"| decode ms/forward | {ab['baseline']['decode_ms_per_forward']:.3f} | {ab['candidate']['decode_ms_per_forward']:.3f} |",
        f"| decode forward/s | {ab['baseline']['decode_forward_per_s_median']:.3f} | {ab['candidate']['decode_forward_per_s_median']:.3f} |",
        f"| full-generate token/s | {ab['baseline']['output_tokens_per_s_median']:.3f} | {ab['candidate']['output_tokens_per_s_median']:.3f} |",
        f"| peak RSS median kB | {ab['baseline']['peak_rss_kb_median']:.0f} | {ab['candidate']['peak_rss_kb_median']:.0f} |",
        f"| final vmRSS median kB | {ab['baseline']['final_vmrss_kb_median']:.0f} | {ab['candidate']['final_vmrss_kb_median']:.0f} |",
        "",
        f"Generate-wall speedup: `{ab['candidate_vs_baseline']['generate_wall_speedup']:.3f}x`; "
        f"decode speedup: `{ab['candidate_vs_baseline']['decode_speedup']:.3f}x`; "
        f"added peak RSS: `{ab['candidate_vs_baseline']['added_peak_rss_kb']:.0f} kB`; "
        f"added final vmRSS: `{ab['candidate_vs_baseline']['added_final_vmrss_kb']:.0f} kB`.",
        "",
        "The profiler is a diagnostic run. The A/B table above is measured "
        "with profiling disabled; C-runtime child intervals are not added to "
        "the runner top-level totals.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", required=True)
    ap.add_argument("--ab-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)
    profile = {
        "baseline": profile_summary(Path(args.profile_root) /
                                    "board-profile-baseline"),
        "candidate": profile_summary(Path(args.profile_root) /
                                     "board-profile-candidate"),
    }
    ab = ab_summary(Path(args.ab_root))
    result = {
        "schema_version": 1,
        "kind": "smollm2-delivery-latency",
        "profile": profile,
        "ab": ab,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    Path(args.out_md).write_text(markdown(profile, ab) + "\n")
    print(json.dumps({
        "baseline_decode_ms_per_forward":
            profile["baseline"]["decode_ms_per_forward"],
        "candidate_decode_ms_per_forward":
            profile["candidate"]["decode_ms_per_forward"],
        "baseline_embed_ms_per_forward":
            profile["baseline"]["top_level_ms_per_forward"]["embed"],
        "candidate_embed_ms_per_forward":
            profile["candidate"]["top_level_ms_per_forward"]["embed"],
        "ab_generate_speedup":
            ab["candidate_vs_baseline"]["generate_wall_speedup"],
        "added_peak_rss_kb":
            ab["candidate_vs_baseline"]["added_peak_rss_kb"],
        "added_final_vmrss_kb":
            ab["candidate_vs_baseline"]["added_final_vmrss_kb"],
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
