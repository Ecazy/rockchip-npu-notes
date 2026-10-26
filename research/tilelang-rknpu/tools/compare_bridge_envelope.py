#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare bridge-envelope profiles across the arms of a campaign.

The campaign measures the arms **round by round** (arm A, arm B, arm C, then the
next round), so pooling the per-round samples is what makes a comparison fair
against machine-state drift.  This tool therefore:

* pools ``wall_us.samples`` from every ``*_r<round>.json`` per arm/profile and
  reports min/p10/median/p90/max over the pooled set;
* also reports the per-round ratios, so a reader can see whether the arms
  separate consistently or whether a single round disagrees;
* compares the first arm (the reference) against every other arm;
* compares structural counters (tasks/submits/upload/download calls and bytes)
  exactly, and says so when they differ;
* reports the envelope attribution only for arms that have one, and states
  "unavailable" otherwise instead of printing zeros as if they were measured.

A non-zero ``*.exit``, a missing profile or a non-finite wall sample is a hard
error.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ENVELOPE_FIELDS = ("validate_us", "pack_us", "emit_us", "command_us",
                   "submit_alloc_us", "submit_wait_us", "output_sync_us",
                   "download_us", "guard_us", "publish_us", "alloc_us")
STRUCTURAL_FIELDS = ("task_count", "submit_count", "upload_calls",
                     "download_calls")
BYTE_FIELDS = ("upload_bytes", "download_bytes")


def quantiles(samples: list[float]) -> dict:
    if not samples:
        raise SystemExit("empty sample set")
    s = sorted(samples)
    n = len(s)
    for value in s:
        if not math.isfinite(value):
            raise SystemExit(f"non-finite wall sample: {value!r}")
    return {"n": n, "min": s[0], "p10": s[n // 10], "median": s[n // 2],
            "p90": s[(9 * n) // 10], "max": s[-1],
            "mean": float(statistics.fmean(s))}


def _add_row(profiles: dict, tag: str, row: dict, round_name: str) -> None:
    entry = profiles.setdefault(tag, {"rounds": [], "rows": []})
    entry["rounds"].append({"round": round_name,
                            "median_us": row["wall_us"]["median"],
                            "n": len(row["wall_us"]["samples"])})
    entry["rows"].append(row)


def load_arm(root: Path, arm: str) -> dict:
    """Return {profile_tag: {"rounds": [...], "rows": [...], "stats": ...}}."""
    out = root / f"out-{arm}"
    if not out.is_dir():
        raise SystemExit(f"missing arm output directory: {out}")
    profiles: dict[str, dict] = {}
    for path in sorted(out.glob("*_r*.json")):
        if path.name.startswith("primitives"):
            continue
        row = json.loads(path.read_text())
        _add_row(profiles, f"{row['model']}_b{row['batch']}", row,
                 path.stem.rsplit("_r", 1)[-1])
    if not profiles:
        for path in sorted(out.glob("*.json")):
            if path.name.startswith("primitives"):
                continue
            row = json.loads(path.read_text())
            _add_row(profiles, f"{row['model']}_b{row['batch']}", row, "0")
    if not profiles:
        raise SystemExit(f"arm {arm} produced no profile json in {out}")
    for entry in profiles.values():
        entry["samples"] = [float(x) for row in entry["rows"]
                            for x in row["wall_us"]["samples"]]
        entry["stats"] = quantiles(entry["samples"])
        # Counters and attribution are constant per arm; the first row carries
        # them.  Only the wall time varies across rounds.
        entry["row"] = entry["rows"][0]
    return profiles


def check_exit(root: Path, arm: str) -> None:
    failed = []
    out = root / f"out-{arm}"
    for path in sorted(out.glob("*.exit")):
        text = path.read_text().strip()
        if text and text != "0":
            failed.append(f"{path.name}={text}")
    if failed:
        raise SystemExit(f"arm {arm} has failing measurements: {failed}")


def arm_summary(name: str, entry: dict) -> dict:
    row = entry["row"]
    env = row["envelope"]
    available = bool(env.get("envelope_available", False))
    # Throughput is derived from the pooled median so it cannot disagree with the
    # latency column.  Everything below it (envelope, attribution, plan-node and
    # host-action splits) is read from ONE representative run and is labelled with
    # the round it came from: those are per-run decompositions, not pooled
    # statistics, and mixing the two is what the review caught.
    batch = int(row["batch"])
    median_us = entry["stats"]["median"]
    return {
        "arm": name,
        "batch": batch,
        "plan_entry": row.get("plan_entry", "attributed"),
        "envelope_available": available,
        "wall": entry["stats"],
        "images_per_s": (batch / (median_us / 1e6)) if median_us else 0.0,
        "images_per_s_source": "pooled_median",
        "single_sample_round": entry["rounds"][0]["round"],
        "single_sample_label": row.get("label", ""),
        "rounds": entry["rounds"],
        "workspace_us": env["plan_workspace_us"],
        "plan_call_us": env["plan_call_us"],
        "python_plan_node_us": env["python_plan_node_us"],
        "host_action_us": env["host_action_us"],
        "executor_overhead_us": env["executor_overhead_us"],
        "attribution": ({f: env.get(f, 0.0) for f in ENVELOPE_FIELDS}
                        if available else None),
        "total_us": env.get("plan_total_us", 0.0) if available else 0.0,
        "clock_us": env.get("plan_clock_us", 0.0) if available else 0.0,
        "residual_us": env.get("plan_envelope_residual_us", 0.0)
        if available else 0.0,
        "structural": {f: int(env[f]) for f in STRUCTURAL_FIELDS},
        "bytes": {f: int(env[f]) for f in BYTE_FIELDS},
        "cpu_affinity": row.get("host", {}).get("cpu_affinity"),
        "npu_freq": row.get("host", {}).get("npu_freq"),
    }


def compare(tag: str, ref: dict, other: dict) -> dict:
    r, o = ref["wall"], other["wall"]
    out = {
        "profile": tag,
        "batch": ref["batch"],
        "reference": ref,
        "other": other,
        "wall_ratio": (r["median"] / o["median"]) if o["median"] else None,
        "wall_delta_us": o["median"] - r["median"],
        "per_round_ratio": [],
        "structure": {},
    }
    for i in range(min(len(ref["rounds"]), len(other["rounds"]))):
        rb = ref["rounds"][i]["median_us"]
        ro = other["rounds"][i]["median_us"]
        out["per_round_ratio"].append(
            {"round": other["rounds"][i]["round"],
             "reference_us": rb, "other_us": ro,
             "ratio": (rb / ro) if ro else None})
    for field in STRUCTURAL_FIELDS:
        out["structure"][field] = {"reference": ref["structural"][field],
                                   "other": other["structural"][field],
                                   "same": ref["structural"][field] ==
                                           other["structural"][field]}
    for field in BYTE_FIELDS:
        out["structure"][field] = {"reference": ref["bytes"][field],
                                   "other": other["bytes"][field],
                                   "same": ref["bytes"][field] ==
                                           other["bytes"][field]}
    return out


def render(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        ref, other = row["reference"], row["other"]
        lines.append(f"[{row['profile']}] batch={row['batch']}   "
                     f"reference={ref['arm']}   other={other['arm']}")
        lines.append("  arm           n      min      p10      p50      p90"
                     "      max     img/s")
        for side in (ref, other):
            s = side["wall"]
            lines.append(f"  {side['arm']:<11} {s['n']:4d} "
                         f"{s['min'] / 1e3:8.3f} {s['p10'] / 1e3:8.3f} "
                         f"{s['median'] / 1e3:8.3f} {s['p90'] / 1e3:8.3f} "
                         f"{s['max'] / 1e3:8.3f} {side['images_per_s']:9.1f}")
        ratio = row["wall_ratio"]
        lines.append(f"  pooled ratio   {ratio:.4f}x" if ratio
                     else "  pooled ratio   n/a")
        lines.append("  per-round      " + ", ".join(
            f"r{p['round']}:{(p['ratio'] or 0):.3f}x"
            for p in row["per_round_ratio"]))
        same = all(v["same"] for v in row["structure"].values())
        lines.append(f"  structure      {'identical' if same else 'DIFFERS'}")
        if not same:
            for field, item in row["structure"].items():
                if not item["same"]:
                    lines.append(f"    {field}: {item['reference']} -> "
                                 f"{item['other']}")
        for side in (ref, other):
            if not side["envelope_available"]:
                lines.append(f"  {side['arm']:<11} envelope: unavailable "
                             "(probes compiled out / legacy entry)")
                continue
            total = side["total_us"] or 1.0
            lines.append(f"  {side['arm']:<11} envelope "
                         f"[single sample, round {side['single_sample_round']}]: "
                         f"C total {side['total_us'] / 1e3:.3f} ms, probes "
                         f"{side['clock_us'] / 1e3:.3f} ms, residual "
                         f"{side['residual_us'] / 1e3:.3f} ms")
            for field, value in side["attribution"].items():
                lines.append(f"      {field:<16} {value / 1e3:9.3f} ms "
                             f"{value / total * 100:5.1f}%")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arms", default="baseline,candidate,noprobe")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    if len(arms) < 2:
        raise SystemExit("comparison needs at least two arms")
    ref_arm, other_arms = arms[0], arms[1:]
    data = {}
    for arm in arms:
        check_exit(args.root, arm)
        data[arm] = load_arm(args.root, arm)
    tags = set(data[ref_arm])
    for arm in other_arms:
        if set(data[arm]) != tags:
            raise SystemExit(f"profile sets differ between {ref_arm} and {arm}: "
                             f"{sorted(set(data[arm]) ^ tags)}")
    rows = []
    for tag in sorted(tags):
        ref = arm_summary(ref_arm, data[ref_arm][tag])
        for arm in other_arms:
            rows.append(compare(tag, ref, arm_summary(arm, data[arm][tag])))
    sys.stdout.write(render(rows))
    args.out.write_text(json.dumps({
        "schema_version": 2,
        "kind": "bridge-envelope-comparison",
        "reference_arm": ref_arm,
        "arms": arms,
        "pooled_across_rounds": True,
        "comparisons": rows,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
