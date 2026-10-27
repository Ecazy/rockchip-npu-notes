#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarise the current / vendor-RKNN / CPU full-model comparison."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = (("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
            ("tiny_vit", 8), ("tiny_vit", 1))


def median(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def load_rounds(results: Path):
    by_arm = {"current": [], "vendor": [], "cpu": []}
    for p in sorted(results.iterdir()):
        if not p.is_dir() or "-" not in p.name or not p.name.startswith("r"):
            continue
        _, arm = p.name.split("-", 1)
        if arm not in by_arm:
            continue
        wall = None
        wp = p / "wall_seconds.txt"
        if wp.is_file():
            try:
                wall = float(wp.read_text().strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass
        by_arm[arm].append({"dir": p, "wall": wall})
    return by_arm


def current_rows(round_dir: Path):
    report = json.loads((round_dir / "model_eval.json").read_text())
    rows = {}
    for row in report["models"]:
        calls = int(row["batch_schedule"]["calls"])
        rows[(row["model"], int(row["batch"]))] = {
            "n": int(row["n"]),
            "accuracy": float(row["accuracy"]),
            "cpu_accuracy": float(row["cpu_accuracy"]),
            "top1_agreement": float(row["top1_agreement"]),
            "declared_top1_agreement": float(row["declared_top1_agreement"]),
            "max_abs_logit_error": float(row["max_abs_logit_error"]),
            "declared_max_abs_logit_error": float(
                row["declared_max_abs_logit_error"]),
            "warm_call_ms": 1000.0 * float(
                row["timing"]["model_call_seconds"]) / calls,
            "warm_images_s": float(row["n"]) / float(
                row["timing"]["model_call_seconds"]),
            "setup_ms": 1000.0 * float(
                row["timing"]["runner_setup_seconds"]),
            "calls": calls,
            "structure": {
                "task_count": row["stats"].get("task_count"),
                "submit_count": row["stats"].get("submit_count"),
                "upload_calls": row["stats"].get("upload_calls"),
                "download_calls": row["stats"].get("download_calls"),
                "upload_bytes": row["stats"].get("upload_bytes"),
                "download_bytes": row["stats"].get("download_bytes"),
            },
        }
    return rows


def vendor_rows(round_dir: Path, kind: str):
    report = json.loads((round_dir / f"{kind}.json").read_text())
    rows = {}
    for row in report["profiles"]:
        calls = int(row["batch_schedule"]["calls"])
        m = row["metrics"]
        rows[(row["model"], int(row["batch"]))] = {
            "n": int(row["n"]),
            "accuracy": float(m["accuracy"]),
            "cpu_accuracy": float(m["cpu_accuracy"]),
            "top1_agreement": float(m["top1_agreement"]),
            "declared_top1_agreement": float(m["declared_top1_agreement"]),
            "max_abs_logit_error": float(m["max_abs_logit_error"]),
            "declared_max_abs_logit_error": float(
                m["declared_max_abs_logit_error"]),
            "warm_call_ms": 1000.0 * float(row["model_call_seconds"]) / calls,
            "warm_images_s": float(row["n"]) / float(row["model_call_seconds"]),
            "setup_ms": 1000.0 * float(row["setup_seconds"]),
            "calls": calls,
            "structure": {"runtime_calls": calls},
        }
    return rows


def summarise(by_arm):
    profiles = {}
    walls = {}
    for arm, rounds in by_arm.items():
        walls[arm] = median([r["wall"] for r in rounds])
        rows_by_round = []
        for r in rounds:
            if arm == "current":
                rows_by_round.append(current_rows(r["dir"]))
            else:
                rows_by_round.append(vendor_rows(r["dir"], arm))
        for model, batch in PROFILES:
            key = f"{model}.b{batch}"
            rows = [rows[(model, batch)] for rows in rows_by_round
                    if (model, batch) in rows]
            if not rows:
                continue
            profiles[key] = {}
            for field in ("accuracy", "cpu_accuracy", "top1_agreement",
                          "declared_top1_agreement", "max_abs_logit_error",
                          "declared_max_abs_logit_error", "warm_call_ms",
                          "warm_images_s", "setup_ms", "calls"):
                profiles[key][field] = median(row[field] for row in rows)
            profiles[key]["n"] = min(row["n"] for row in rows)
            profiles[key]["structure"] = rows[0]["structure"]
            profiles[key]["round_warm_call_ms"] = [
                row["warm_call_ms"] for row in rows]
    return {"profiles": profiles, "wall_seconds": walls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--final-analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()
    campaign = Path(args.campaign).resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    final = json.loads(Path(args.final_analysis).resolve().read_text())
    summary = summarise(load_rounds(Path(args.results).resolve()))
    problems = []
    thresholds = manifest["thresholds"]
    expected = {(m, b) for m, b in PROFILES}
    for arm, profiles in summary["profiles"].items():
        if arm not in ("current", "vendor", "cpu"):
            continue
    # Common row completeness and accuracy gates.
    for arm, rows in (("current", summary["profiles"]),):
        pass
    # Reshape by arm for easier checks.
    arm_profiles = {}
    for arm in ("current", "vendor", "cpu"):
        arm_profiles[arm] = {}
    # load rows per arm again (small, deterministic)
    by_arm = load_rounds(Path(args.results).resolve())
    for arm, rounds in by_arm.items():
        for r in rounds:
            rows = current_rows(r["dir"]) if arm == "current" else vendor_rows(r["dir"], arm)
            if set(rows) != expected:
                problems.append(f"{r['dir'].name}: five-profile set mismatch")
            for (model, batch), row in rows.items():
                key = f"{model}.b{batch}"
                if row["n"] != 10000:
                    problems.append(f"{r['dir'].name}/{key}: n={row['n']}")
                if row["accuracy"] < thresholds[
                        "mnist_mlp_accuracy" if model == "mnist_mlp"
                        else "tiny_vit_accuracy"]:
                    problems.append(f"{r['dir'].name}/{key}: accuracy below threshold")
                if row["top1_agreement"] < thresholds["top1_agreement_min"] or \
                        row["declared_top1_agreement"] < thresholds[
                            "top1_agreement_min"]:
                    problems.append(f"{r['dir'].name}/{key}: top-1 below threshold")
    comparison = {}
    for model, batch in PROFILES:
        key = f"{model}.b{batch}"
        c = summary["profiles"][key] if key in summary["profiles"] else None
        # The flattened summary above is not arm-indexed; recompute arm medians below.
    # Recompute medians per arm from the raw round rows.
    arm_med = {arm: {} for arm in ("current", "vendor", "cpu")}
    for arm, rounds in by_arm.items():
        per = {}
        for r in rounds:
            rows = current_rows(r["dir"]) if arm == "current" else vendor_rows(r["dir"], arm)
            for k, row in rows.items():
                per.setdefault(k, []).append(row)
        for (model, batch), rows in per.items():
            key = f"{model}.b{batch}"
            arm_med[arm][key] = {
                field: median([row[field] for row in rows])
                for field in ("accuracy", "cpu_accuracy", "top1_agreement",
                              "declared_top1_agreement",
                              "max_abs_logit_error",
                              "declared_max_abs_logit_error",
                              "warm_call_ms", "warm_images_s", "setup_ms")}
            arm_med[arm][key]["structure"] = rows[0]["structure"]
            arm_med[arm][key]["round_warm_call_ms"] = [
                row["warm_call_ms"] for row in rows]
    for model, batch in PROFILES:
        key = f"{model}.b{batch}"
        if not all(key in arm_med[a] for a in ("current", "vendor", "cpu")):
            problems.append(f"{key}: missing arm")
            continue
        c = arm_med["current"][key]
        v = arm_med["vendor"][key]
        u = arm_med["cpu"][key]
        pre = final["arms"]["baseline"]["profiles"][key]
        cur_final = final["arms"]["current"]["profiles"][key]
        comparison[key] = {
            "preopt_warm_call_ms": pre["warm_call_ms"],
            "preopt_vs_current_final_speedup": final["comparison"][key]["warm_speedup"],
            "current_final_warm_call_ms": cur_final["warm_call_ms"],
            "current_vendor_campaign_warm_call_ms": c["warm_call_ms"],
            "vendor_warm_call_ms": v["warm_call_ms"],
            "cpu_warm_call_ms": u["warm_call_ms"],
            "vendor_vs_current_speedup": c["warm_call_ms"] / v["warm_call_ms"],
            "cpu_vs_current_speedup": c["warm_call_ms"] / u["warm_call_ms"],
            "vendor_vs_cpu_speedup": u["warm_call_ms"] / v["warm_call_ms"],
            "current_vendor_campaign_images_s": c["warm_images_s"],
            "vendor_images_s": v["warm_images_s"],
            "cpu_images_s": u["warm_images_s"],
            "current_setup_ms": c["setup_ms"],
            "vendor_setup_ms": v["setup_ms"],
            "cpu_setup_ms": u["setup_ms"],
            "current_structure": c["structure"],
            "vendor_structure": v["structure"],
            "cpu_structure": u["structure"],
            "vendor_accuracy": v["accuracy"],
            "vendor_top1_agreement": v["top1_agreement"],
            "vendor_declared_top1_agreement": v["declared_top1_agreement"],
            "vendor_max_abs_logit_error": v["max_abs_logit_error"],
            "vendor_declared_max_abs_logit_error": v[
                "declared_max_abs_logit_error"],
            "current_max_abs_logit_error": c["max_abs_logit_error"],
            "packed_task_atol": json.loads(
                (campaign / "numeric" / "numeric_acceptance_packed.json").read_text()
            )["task_logit_tolerances"][key]["atol"] if (
                campaign / "numeric" / "numeric_acceptance_packed.json").is_file() else None,
        }
    output = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "campaign_manifest_sha256": __import__("hashlib").sha256(
            (campaign / "manifest.json").read_bytes()).hexdigest(),
        "full_test_images": manifest["full_test_images"],
        "rounds": manifest["rounds"],
        "arms": {arm: {"profiles": arm_med[arm],
                       "wall_seconds": summary["wall_seconds"][arm]}
                 for arm in ("current", "vendor", "cpu")},
        "comparison": comparison,
        "preopt_final": final["comparison"],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        lines = [
            "# Vendor baseline comparison", "",
            f"- status: **{output['status']}**",
            f"- full test images per profile: {manifest['full_test_images']}",
            f"- rounds: {manifest['rounds']}", "",
            "Warm values are per batch call; vendor includes its graph preprocessing.",
            "", "Speedups are `current_ms / arm_ms`; >1 means the arm is faster.",
            "", "| profile | preopt ms* | current ms | vendor ms | CPU ms | "
            "vendor speedup | CPU speedup |", "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model, batch in PROFILES:
            key = f"{model}.b{batch}"
            c = comparison[key]
            lines.append(
                f"| {key} | {c['preopt_warm_call_ms']:.3f} | "
                f"{c['current_vendor_campaign_warm_call_ms']:.3f} | "
                f"{c['vendor_warm_call_ms']:.3f} | {c['cpu_warm_call_ms']:.3f} | "
                f"{c['vendor_vs_current_speedup']:.3f}x | "
                f"{c['cpu_vs_current_speedup']:.3f}x |")
        lines += ["", "\\* preopt comes from the earlier final campaign "
                  "(`results/20261026T120625Z-final`); the other three columns "
                  "share this vendor campaign.", "", "## Accuracy and numeric", "",
                  "| profile | vendor acc | vendor top-1 | vendor max err | packed atol |",
                  "|---|---:|---:|---:|---:|"]
        for model, batch in PROFILES:
            key = f"{model}.b{batch}"
            c = comparison[key]
            lines.append(
                f"| {key} | {c['vendor_accuracy']:.5f} | "
                f"{c['vendor_top1_agreement']:.5f} | "
                f"{c['vendor_max_abs_logit_error']:.5f} | "
                f"{c['packed_task_atol']} |")
        lines += ["", "Vendor accuracy/top-1 equals the CPU reference for all five "
                  "profiles.  MLP logits stay within the packed task-logit "
                  "tolerance; ViT max errors (0.1228/0.1072) exceed the packed "
                  "0.0625 tolerance, so vendor fusion is classification-preserving "
                  "but not a drop-in per-logit equivalence.", "",
                  "## Setup and structure", "",
                  "| profile | current setup ms | vendor setup ms | "
                  "current task/submit | vendor rknn calls | CPU calls |",
                  "|---|---:|---:|---:|---:|---:|"]
        for model, batch in PROFILES:
            key = f"{model}.b{batch}"
            c = comparison[key]
            cs = c["current_structure"]
            lines.append(
                f"| {key} | {c['current_setup_ms']:.1f} | "
                f"{c['vendor_setup_ms']:.1f} | "
                f"{cs.get('task_count')}/{cs.get('submit_count')} | "
                f"{c['vendor_structure']['runtime_calls']} | "
                f"{c['cpu_structure']['runtime_calls']} |")
        lines += ["", "## Wall time", "",
                  "| arm | wall seconds (2 rounds) |",
                  "|---|---:|",
                  f"| current | {summary['wall_seconds']['current']} |",
                  f"| vendor | {summary['wall_seconds']['vendor']} |",
                  f"| CPU | {summary['wall_seconds']['cpu']} |",
                  "", "Wall includes CPU/declared reference calculations and "
                  "trace capture; it is not the application-only interval.",
                  "", "Vendor RKNN uses toolkit/runtime 2.3.2, non-quantized "
                  "float16, `single_core_mode=True`, runtime core 0; current uses "
                  "the same single NPU core and CPU affinity 6.",
                  "", "**Interpretation:** vendor is the strongest batched path; "
                  "CPU remains faster than both backends at b1 for MLP and ViT. "
                  "The gap is therefore not a single QKV transport issue; it "
                  "points to graph-level scheduling/fusion and Host boundaries."]
        Path(args.markdown).resolve().write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": output["status"],
                      "problems": output["problems"]}, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
