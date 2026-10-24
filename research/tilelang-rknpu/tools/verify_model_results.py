#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed verifier for the complete-model board campaign."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    results = Path(args.results)
    th = manifest["thresholds"]
    problems = []
    summary = {"status": "PASS", "cores": {}, "thresholds": th}

    for mask in (1, 2, 4):
        core = results / f"core-{mask}"
        path = core / "model_eval.json"
        if not path.is_file():
            problems.append(f"core-{mask}: missing model_eval.json")
            continue
        report = json.loads(path.read_text())
        if report.get("backend") != "board":
            problems.append(f"core-{mask}: backend is not board")
        rows = {(m["model"], m["batch"]): m for m in report.get("models", [])}
        expected_n = 10000 if mask == 1 else 512
        required = [("mnist_mlp", 32), ("tiny_vit", 8)]
        core_summary = {"expected_images": expected_n, "rows": {}}
        for name, batch in required:
            row = rows.get((name, batch))
            if row is None:
                problems.append(f"core-{mask}: missing {name} batch {batch}")
                continue
            if not row.get("full_test_split"):
                problems.append(f"core-{mask}: {name} batch {batch} is not the full profile")
            if row.get("n") != expected_n:
                problems.append(f"core-{mask}: {name} n={row.get('n')} != {expected_n}")
            threshold = th["mnist_mlp_accuracy"] if name == "mnist_mlp" else th["tiny_vit_accuracy"]
            if float(row.get("accuracy", -1)) < threshold:
                problems.append(f"core-{mask}: {name} accuracy below {threshold}")
            if float(row.get("accuracy_delta", -1)) < -float(th["accuracy_drop_max"]):
                problems.append(f"core-{mask}: {name} drops more than {th['accuracy_drop_max']}")
            if float(row.get("top1_agreement", -1)) < float(th["top1_agreement_min"]):
                problems.append(f"core-{mask}: {name} CPU top1 agreement below threshold")
            if float(row.get("declared_top1_agreement", -1)) < float(th["top1_agreement_min"]):
                problems.append(f"core-{mask}: {name} declared top1 agreement below threshold")
            if not math.isfinite(float(row.get("max_abs_logit_error", float("nan")))):
                problems.append(f"core-{mask}: {name} invalid logit error")
            core_summary["rows"][name] = {
                "batch": batch, "n": row.get("n"),
                "accuracy": row.get("accuracy"),
                "cpu_accuracy": row.get("cpu_accuracy"),
                "accuracy_delta": row.get("accuracy_delta"),
                "top1_agreement": row.get("top1_agreement"),
                "declared_top1_agreement": row.get("declared_top1_agreement"),
                "max_abs_logit_error": row.get("max_abs_logit_error"),
                "declared_max_abs_logit_error": row.get("declared_max_abs_logit_error"),
                "host_compute_us": row.get("stats", {}).get("host_compute_us"),
                "upload_bytes": row.get("stats", {}).get("upload_bytes"),
                "download_bytes": row.get("stats", {}).get("download_bytes"),
                "task_count": row.get("stats", {}).get("task_count"),
            }
        # Every additional profile (batch=1/8 etc.) must also agree with CPU and
        # declared references; these are validation rows, not full-split claims.
        for (name, batch), row in rows.items():
            if (name, batch) in required:
                continue
            if float(row.get("top1_agreement", -1)) < float(th["top1_agreement_min"]):
                problems.append(f"core-{mask}: {name} batch {batch} CPU agreement low")
            if float(row.get("declared_top1_agreement", -1)) < float(th["top1_agreement_min"]):
                problems.append(f"core-{mask}: {name} batch {batch} declared agreement low")
        summary["cores"][f"core-{mask}"] = core_summary
    cpu_path = results / "core-1" / "cpu_benchmark.json"
    if not cpu_path.is_file():
        problems.append("core-1: missing CPU benchmark")
    else:
        cpu = json.loads(cpu_path.read_text())
        if cpu.get("test_images") != 10000:
            problems.append("core-1: CPU benchmark did not use 10000 images")
        if not cpu.get("cpu_count"):
            problems.append("core-1: CPU benchmark lacks cpu_count")
        if not (cpu.get("cpuinfo") or {}).get("mean_mhz"):
            problems.append("core-1: CPU benchmark lacks measured frequency")
        cm = cpu.get("models", {})
        if float(cm.get("mnist_mlp", {}).get("fp32_accuracy", -1)) < th["mnist_mlp_accuracy"]:
            problems.append("core-1: CPU MLP baseline below accuracy threshold")
        if float(cm.get("tiny_vit", {}).get("fp32_accuracy", -1)) < th["tiny_vit_accuracy"]:
            problems.append("core-1: CPU ViT baseline below accuracy threshold")
        summary["cpu_benchmark"] = {
            "cpu_count": cpu.get("cpu_count"),
            "cpu_thread_env": cpu.get("cpu_thread_env"),
            "cpuinfo": cpu.get("cpuinfo"),
            "mnist_mlp": cm.get("mnist_mlp"),
            "tiny_vit": cm.get("tiny_vit"),
        }
    if problems:
        summary["status"] = "FAIL"
        summary["problems"] = problems
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
