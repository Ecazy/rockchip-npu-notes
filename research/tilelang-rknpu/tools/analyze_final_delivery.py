#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Aggregate the final baseline/current full-model delivery campaign.

This is a delivery analysis, not a new numeric policy.  It recomputes the CPU
and declared FP16 references from the campaign payload, checks the original
accuracy/top-1 thresholds, applies the frozen packed task-logit tolerances, and
summarises warm latency, throughput, creation cost and structural counts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_model_results import (  # noqa: E402
    read_idx, _refs_with_schedule, _row_metrics,
    logits_within_frozen_tolerance)
from rknpu_backend.models import load_weight_npz  # noqa: E402

SHORT = {"mnist_mlp": "mlp", "tiny_vit": "vit"}
EXPECTED = (("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
            ("tiny_vit", 8), ("tiny_vit", 1))


def _median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(report_path: Path, model: str, batch: int, kind: str) -> Path:
    return report_path.with_name(
        report_path.stem + f".{SHORT[model]}_b{batch}.{kind}.npy")


def analyze(campaign: Path, results: Path, arms: tuple[str, ...]) -> dict:
    manifest = json.loads((campaign / "manifest.json").read_text())
    expected_n = int(manifest["full_test_images"])
    thresholds = manifest["thresholds"]
    packed_spec = json.loads(
        (campaign / "numeric" / "numeric_acceptance_packed.json").read_text())
    if packed_spec.get("status") != "frozen":
        raise RuntimeError("packed numeric spec is not frozen")
    tols = packed_spec["task_logit_tolerances"]
    data_dir = campaign / "data"
    weights_dir = campaign / "weights"
    x = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    weights = {
        "mnist_mlp": load_weight_npz(weights_dir / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(weights_dir / "tiny_vit.npz"),
    }
    refs: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    problems: list[str] = []
    rounds = sorted(p.name for p in results.iterdir()
                    if p.is_dir() and p.name.startswith("r") and "-" in p.name)
    by_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    per_round: dict[tuple[int, str], dict] = {}
    raw_logits: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for dirname in rounds:
        try:
            round_no, arm = dirname[1:].split("-", 1)
            round_no = int(round_no)
        except ValueError:
            continue
        if arm not in by_arm:
            continue
        report_path = results / dirname / "model_eval.json"
        if not report_path.is_file():
            problems.append(f"{dirname}: missing model_eval.json")
            continue
        report = json.loads(report_path.read_text())
        if int(report.get("test_images", -1)) != expected_n:
            problems.append(
                f"{dirname}: test_images {report.get('test_images')} != {expected_n}")
        wall_path = results / dirname / "wall_seconds.txt"
        wall = None
        if wall_path.is_file():
            try:
                wall = float(wall_path.read_text().strip().splitlines()[-1])
            except (ValueError, IndexError):
                problems.append(f"{dirname}: invalid wall_seconds.txt")
        rows = {(row["model"], int(row["batch"])): row
                for row in report.get("models", [])}
        if set(rows) != set(EXPECTED):
            problems.append(f"{dirname}: five-profile set mismatch")
        summary_rows: dict[str, dict] = {}
        raw_logits[(round_no, arm)] = {}
        for model, batch in EXPECTED:
            key = f"{model}.b{batch}"
            row = rows.get((model, batch))
            if row is None:
                continue
            n = int(row["n"])
            if n != expected_n:
                problems.append(f"{dirname}/{key}: n={n} != full test {expected_n}")
            logits_path = _sidecar(report_path, model, batch, "logits")
            declared_path = _sidecar(report_path, model, batch, "declared")
            if not logits_path.is_file() or not declared_path.is_file():
                problems.append(f"{dirname}/{key}: missing raw logits/declared sidecar")
                continue
            logits = np.load(logits_path, allow_pickle=False)
            if logits.shape != (n, 10) or not np.isfinite(logits).all():
                problems.append(f"{dirname}/{key}: invalid raw logits")
                continue
            raw_logits[(round_no, arm)][key] = logits
            cache_key = (model, batch, n)
            if cache_key not in refs:
                cpu = _refs_with_schedule("cpu", model, weights[model],
                                          x[:n], batch)
                declared = _refs_with_schedule(
                    "declared", model, weights[model], x[:n], batch)
                refs[cache_key] = (cpu, declared)
            cpu, declared = refs[cache_key]
            metrics = _row_metrics(logits, cpu, declared, y[:n])
            for field in ("accuracy", "cpu_accuracy", "accuracy_delta",
                          "top1_agreement", "declared_accuracy",
                          "declared_top1_agreement"):
                reported = row.get(field)
                if not isinstance(reported, (int, float)) or \
                        abs(float(reported) - float(metrics[field])) > 1e-9:
                    problems.append(
                        f"{dirname}/{key}: reported {field} differs from refs")
            if metrics["accuracy"] < thresholds[
                    "mnist_mlp_accuracy" if model == "mnist_mlp"
                    else "tiny_vit_accuracy"]:
                problems.append(f"{dirname}/{key}: accuracy below threshold")
            if metrics["accuracy_delta"] < -float(thresholds["accuracy_drop_max"]):
                problems.append(f"{dirname}/{key}: accuracy drop exceeds threshold")
            if metrics["top1_agreement"] < float(thresholds["top1_agreement_min"]) or \
                    metrics["declared_top1_agreement"] < float(
                        thresholds["top1_agreement_min"]):
                problems.append(f"{dirname}/{key}: top-1 agreement below threshold")
            if key not in tols:
                problems.append(f"{dirname}/{key}: missing packed task tolerance")
                ok, error = False, float("inf")
            else:
                ok, error = logits_within_frozen_tolerance(
                    logits, declared, tols[key])
                if not ok:
                    problems.append(
                        f"{dirname}/{key}: raw logits outside packed tolerance")
            stats = row.get("stats", {})
            timing = row.get("timing", {})
            calls = int(row.get("batch_schedule", {}).get("calls", 0))
            model_call = float(timing.get("model_call_seconds", -1))
            end_to_end = float(timing.get("end_to_end_seconds", -1))
            setup = float(timing.get("runner_setup_seconds", -1))
            item = {
                "n": n,
                "accuracy": metrics["accuracy"],
                "cpu_accuracy": metrics["cpu_accuracy"],
                "top1_agreement": metrics["top1_agreement"],
                "declared_accuracy": metrics["declared_accuracy"],
                "declared_top1_agreement": metrics["declared_top1_agreement"],
                "max_abs_logit_error": metrics["max_abs_logit_error"],
                "declared_max_abs_logit_error": error,
                "packed_tolerance_ok": bool(ok),
                "batch_calls": calls,
                "warm_call_ms": 1000.0 * model_call / calls if calls else None,
                "warm_throughput_images_s": n / model_call if model_call > 0 else None,
                "end_to_end_throughput_images_s": (
                    n / end_to_end if end_to_end > 0 else None),
                "runner_setup_ms": 1000.0 * setup if setup >= 0 else None,
                "warmup_ms": 1000.0 * float(timing.get("warmup_seconds", -1)),
                "weight_load_ms": 1000.0 * float(
                    timing.get("weight_load_seconds", -1)),
                "model_call_seconds": model_call,
                "end_to_end_seconds": end_to_end,
                "reference_seconds": timing.get("reference_seconds"),
                "structure": {
                    "task_count": stats.get("task_count"),
                    "submit_count": stats.get("submit_count"),
                    "run_calls": stats.get("run_calls"),
                    "upload_calls": stats.get("upload_calls"),
                    "download_calls": stats.get("download_calls"),
                    "upload_bytes": stats.get("upload_bytes"),
                    "download_bytes": stats.get("download_bytes"),
                    "plan_workspace_ms": 1e-3 * float(
                        stats.get("plan_workspace_us", 0.0)),
                    "host_action_ms": 1e-3 * float(
                        stats.get("host_action_us", 0.0)),
                    "executor_overhead_ms": 1e-3 * float(
                        stats.get("executor_overhead_us", 0.0)),
                },
            }
            summary_rows[key] = item
        per_round[(round_no, arm)] = {
            "wall_seconds": wall,
            "rows": summary_rows,
        }
        by_arm[arm].append(per_round[(round_no, arm)])

    # Arm-level medians and structure checks.
    arm_summary: dict[str, dict] = {}
    for arm, rounds_for_arm in by_arm.items():
        if not rounds_for_arm:
            problems.append(f"{arm}: no rounds")
            continue
        profiles = {}
        for model, batch in EXPECTED:
            key = f"{model}.b{batch}"
            rows = [r["rows"][key] for r in rounds_for_arm if key in r["rows"]]
            if len(rows) != len(rounds_for_arm):
                problems.append(f"{arm}/{key}: missing round row")
                continue
            structures = [json.dumps(row["structure"], sort_keys=True)
                          for row in rows]
            if len(set(structures)) != 1:
                problems.append(f"{arm}/{key}: structure differs across rounds")
            profiles[key] = {
                "rounds": len(rows),
                "accuracy": _median(row["accuracy"] for row in rows),
                "top1_agreement": _median(row["top1_agreement"] for row in rows),
                "declared_top1_agreement": _median(
                    row["declared_top1_agreement"] for row in rows),
                "max_abs_logit_error": _median(
                    row["max_abs_logit_error"] for row in rows),
                "declared_max_abs_logit_error": _median(
                    row["declared_max_abs_logit_error"] for row in rows),
                "warm_call_ms": _median(row["warm_call_ms"] for row in rows),
                "warm_throughput_images_s": _median(
                    row["warm_throughput_images_s"] for row in rows),
                "end_to_end_throughput_images_s": _median(
                    row["end_to_end_throughput_images_s"] for row in rows),
                "runner_setup_ms": _median(
                    row["runner_setup_ms"] for row in rows),
                "weight_load_ms": _median(
                    row["weight_load_ms"] for row in rows),
                "warmup_ms": _median(row["warmup_ms"] for row in rows),
                "model_call_seconds": _median(
                    row["model_call_seconds"] for row in rows),
                "end_to_end_seconds": _median(
                    row["end_to_end_seconds"] for row in rows),
                "reference_seconds": rows[0]["reference_seconds"],
                "structure": rows[0]["structure"],
                "round_warm_call_ms": [row["warm_call_ms"] for row in rows],
                "round_warm_throughput_images_s": [
                    row["warm_throughput_images_s"] for row in rows],
            }
        arm_summary[arm] = {
            "profiles": profiles,
            "full_eval_wall_seconds": _median(
                r["wall_seconds"] for r in rounds_for_arm),
            "round_wall_seconds": [r["wall_seconds"] for r in rounds_for_arm],
            "rounds": len(rounds_for_arm),
        }

    comparison = {}
    if "baseline" in arm_summary and "current" in arm_summary:
        for model, batch in EXPECTED:
            key = f"{model}.b{batch}"
            b = arm_summary["baseline"]["profiles"].get(key)
            c = arm_summary["current"]["profiles"].get(key)
            if not b or not c:
                continue
            exact = []
            maxdiff = []
            for round_no in sorted({r for r, _ in per_round}):
                lb = raw_logits.get((round_no, "baseline"), {}).get(key)
                lc = raw_logits.get((round_no, "current"), {}).get(key)
                if lb is not None and lc is not None:
                    exact.append(bool(np.array_equal(lb, lc)))
                    maxdiff.append(float(np.max(np.abs(
                        lb.astype(np.float64) - lc.astype(np.float64)),
                        initial=0.0)))
            comparison[key] = {
                "warm_speedup": (b["warm_call_ms"] / c["warm_call_ms"]
                                 if c["warm_call_ms"] else None),
                "warm_throughput_ratio": (
                    c["warm_throughput_images_s"] /
                    b["warm_throughput_images_s"]
                    if b["warm_throughput_images_s"] else None),
                "end_to_end_throughput_ratio": (
                    c["end_to_end_throughput_images_s"] /
                    b["end_to_end_throughput_images_s"]
                    if b["end_to_end_throughput_images_s"] else None),
                "runner_setup_ratio": (
                    b["runner_setup_ms"] / c["runner_setup_ms"]
                    if c["runner_setup_ms"] else None),
                "byte_exact_rounds": exact,
                "max_abs_baseline_current_logit_delta": (
                    max(maxdiff) if maxdiff else None),
                "baseline_structure": b["structure"],
                "current_structure": c["structure"],
            }
        bwall = arm_summary["baseline"]["full_eval_wall_seconds"]
        cwall = arm_summary["current"]["full_eval_wall_seconds"]
        comparison["full_eval_wall"] = {
            "baseline_seconds": bwall,
            "current_seconds": cwall,
            "ratio": (bwall / cwall if cwall else None),
        }

    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "campaign_manifest_sha256": _sha(campaign / "manifest.json"),
        "full_test_images": expected_n,
        "rounds": len(by_arm.get("baseline", [])),
        "arms": arm_summary,
        "comparison": comparison,
        "packed_spec": {
            "status": packed_spec.get("status"),
            "compiled_manifest_sha256": packed_spec.get(
                "compiled_manifest_sha256"),
            "tolerances": tols,
        },
    }


def markdown(summary: dict) -> str:
    lines = ["# Final model delivery analysis", "",
             f"- status: **{summary['status']}**",
             f"- full test images: {summary['full_test_images']}",
             f"- rounds: {summary['rounds']}"]
    if summary["problems"]:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in summary["problems"]]
    lines += ["", "## Profile comparison", "",
              "| profile | baseline warm ms | current warm ms | speedup | "
              "baseline img/s | current img/s | acc b/c | top1 b/c |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, cmp in summary["comparison"].items():
        if key == "full_eval_wall":
            continue
        b = summary["arms"]["baseline"]["profiles"][key]
        c = summary["arms"]["current"]["profiles"][key]
        lines.append(
            f"| {key} | {b['warm_call_ms']:.3f} | {c['warm_call_ms']:.3f} | "
            f"{cmp['warm_speedup']:.3f}x | "
            f"{b['warm_throughput_images_s']:.1f} | "
            f"{c['warm_throughput_images_s']:.1f} | "
            f"{b['accuracy']:.5f}/{c['accuracy']:.5f} | "
            f"{b['top1_agreement']:.5f}/{c['top1_agreement']:.5f} |")
    wall = summary["comparison"].get("full_eval_wall", {})
    lines += ["", "## Full evaluation wall", "",
              f"- baseline: {wall.get('baseline_seconds')} s",
              f"- current: {wall.get('current_seconds')} s",
              f"- ratio: {wall.get('ratio')}x"]
    lines += ["", "## Creation cost (runner setup)", "",
              "| profile | baseline ms | current ms | ratio |",
              "|---|---:|---:|---:|"]
    for model_batch in ("mnist_mlp.b32", "mnist_mlp.b8", "mnist_mlp.b1",
                        "tiny_vit.b8", "tiny_vit.b1"):
        b = summary["arms"]["baseline"]["profiles"][model_batch]
        c = summary["arms"]["current"]["profiles"][model_batch]
        ratio = (b["runner_setup_ms"] / c["runner_setup_ms"]
                 if c["runner_setup_ms"] else None)
        lines.append(f"| {model_batch} | {b['runner_setup_ms']:.3f} | "
                     f"{c['runner_setup_ms']:.3f} | {ratio:.3f}x |")
    lines += ["", "## Structure", "",
              "| profile | baseline task/submit | current task/submit | "
              "baseline upload/download | current upload/download |",
              "|---|---:|---:|---:|---:|"]
    for model_batch in ("mnist_mlp.b32", "mnist_mlp.b8", "mnist_mlp.b1",
                        "tiny_vit.b8", "tiny_vit.b1"):
        bs = summary["arms"]["baseline"]["profiles"][model_batch]["structure"]
        cs = summary["arms"]["current"]["profiles"][model_batch]["structure"]
        lines.append(
            f"| {model_batch} | {bs['task_count']}/{bs['submit_count']} | "
            f"{cs['task_count']}/{cs['submit_count']} | "
            f"{bs['upload_calls']}/{bs['download_calls']} | "
            f"{cs['upload_calls']}/{cs['download_calls']} |")
    lines += ["", "Packed task-logit tolerances are listed in the JSON output."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default="")
    ap.add_argument("--arms", default="baseline,current")
    args = ap.parse_args()
    summary = analyze(Path(args.campaign).resolve(),
                      Path(args.results).resolve(),
                      tuple(args.arms.split(",")))
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        Path(args.markdown).resolve().write_text(markdown(summary))
    print(json.dumps({"status": summary["status"],
                      "problems": summary["problems"]}, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
