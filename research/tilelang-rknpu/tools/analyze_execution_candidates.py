#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyze a same-payload execution-plan campaign on one RK3588 board.

The campaign layout is intentionally small and explicit::

    <results>/r1-generic/model_eval.json
    <results>/r1-prepared/model_eval.json
    <results>/r1-hostfused/model_eval.json
    <results>/r1-vendor/vendor.json
    <results>/r1-cpu/cpu.json

Custom RKPlan arms use the evaluator's native schema; RKNN and NumPy reference
arms use the vendor benchmark schema.  This tool keeps the two groups separate:
only the custom arms share one numeric policy, so only those are ranked against
each other.  Vendor and CPU values are still reported side by side with their
precision/contract caveats.

Warm latency is the application-side model interval per static batch call:
raw input preparation + execution + output materialisation, excluding argmax,
reference computation and the separately recorded setup.  Full process wall time
is not used as a ranking statistic.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

PROFILES = (
    ("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
    ("tiny_vit", 8), ("tiny_vit", 1),
)
PROFILE_KEYS = tuple(f"{model}.b{batch}" for model, batch in PROFILES)
CUSTOM_STATS = (
    "plan_call_us", "plan_workspace_us", "host_action_us",
    "executor_overhead_us", "plan_pack_us", "plan_submit_wait_us",
    "plan_download_us", "plan_guard_us", "task_count", "submit_count",
    "upload_calls", "download_calls", "upload_bytes", "download_bytes",
)


def _median(values):
    rows = [float(v) for v in values if v is not None]
    return statistics.median(rows) if rows else None


def _profile_rows(path: Path, arm: str) -> tuple[dict[str, Any], str]:
    """Return {profile: normalized row} and the arm family."""
    custom = path / "model_eval.json"
    if custom.is_file():
        payload = json.loads(custom.read_text())
        out: dict[str, Any] = {}
        for result in payload.get("models", []):
            key = f"{result['model']}.b{int(result['batch'])}"
            calls = int(result["batch_schedule"]["calls"])
            stats = result.get("stats", {})
            timing = result.get("timing", {})
            per_call = {
                field: float(stats.get(field, 0.0)) / max(calls, 1)
                for field in CUSTOM_STATS
            }
            out[key] = {
                "warm_ms": 1e3 * float(timing["model_call_seconds"]) / calls,
                "e2e_ms": 1e3 * float(timing["end_to_end_seconds"]) / calls,
                "accuracy": float(result["accuracy"]),
                "top1": float(result["top1_agreement"]),
                "max_abs_logit_error": float(result["max_abs_logit_error"]),
                "declared_max_abs_logit_error": float(
                    result.get("declared_max_abs_logit_error", 0.0)),
                "n": int(result["n"]),
                "full_test_split": bool(result["full_test_split"]),
                "calls": calls,
                "setup_ms": 1e3 * float(timing.get("runner_setup_seconds", 0.0)),
                "weight_load_ms": 1e3 * float(
                    timing.get("weight_load_seconds", 0.0)),
                "per_call": per_call,
                "host_actions": {
                    row["name"]: float(row["elapsed_us"]) / calls
                    for row in stats.get("host_actions", [])
                },
                "execution_mode": payload.get("execution_mode", "generic"),
                "hybrid_suffix": payload.get("hybrid_suffix", ""),
            }
        return out, "custom"
    for filename, family in ((f"{arm}.json", "external"),):
        candidate = path / filename
        if candidate.is_file():
            payload = json.loads(candidate.read_text())
            out = {}
            for result in payload.get("profiles", []):
                key = f"{result['model']}.b{int(result['batch'])}"
                calls = int(result["batch_schedule"]["calls"])
                metrics = result.get("metrics", {})
                out[key] = {
                    "warm_ms": 1e3 * float(result["model_call_seconds"]) / calls,
                    "e2e_ms": 1e3 * float(result["end_to_end_seconds"]) / calls,
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "top1": float(metrics.get("top1_agreement", 0.0)),
                    "max_abs_logit_error": float(
                        metrics.get("max_abs_logit_error", 0.0)),
                    "declared_max_abs_logit_error": float(
                        metrics.get("declared_max_abs_logit_error", 0.0)),
                    "n": int(result["n"]),
                    "full_test_split": int(result["n"]) >= 10000,
                    "calls": calls,
                    "setup_ms": 1e3 * float(result.get("setup_seconds", 0.0)),
                    "weight_load_ms": None,
                    "per_call": {},
                    "host_actions": {},
                    "execution_mode": payload.get("backend", family),
                    "hybrid_suffix": "",
                }
            return out, family
    raise FileNotFoundError(f"no custom or external result under {path}")


def _round_dirs(results: Path, arm: str) -> list[Path]:
    found = []
    for path in sorted(results.glob(f"r*-{arm}")):
        if path.is_dir():
            found.append(path)
    return found


def _arm_table(results: Path, arm: str) -> dict:
    rounds = _round_dirs(results, arm)
    if not rounds:
        raise FileNotFoundError(f"no rounds for arm {arm!r} under {results}")
    parsed = [_profile_rows(path, arm) for path in rounds]
    family = parsed[0][1]
    if any(item[1] != family for item in parsed):
        raise RuntimeError(f"arm {arm!r} mixes custom and external schemas")
    out = {"arm": arm, "family": family, "rounds": len(rounds),
           "profiles": {}}
    for key in PROFILE_KEYS:
        rows = [data[0].get(key) for data in parsed]
        if any(row is None for row in rows):
            raise RuntimeError(f"arm {arm!r} lacks profile {key}")
        prof = {
            field: _median(row.get(field) for row in rows)
            for field in (
                "warm_ms", "e2e_ms", "accuracy", "top1",
                "max_abs_logit_error", "declared_max_abs_logit_error",
                "n", "calls", "setup_ms", "weight_load_ms")
        }
        prof["full_test_split"] = all(bool(row["full_test_split"])
                                      for row in rows)
        prof["round_warm_ms"] = [float(row["warm_ms"]) for row in rows]
        prof["execution_mode"] = rows[0].get("execution_mode")
        prof["per_call"] = {
            field: _median(row["per_call"].get(field) for row in rows)
            for field in CUSTOM_STATS
        }
        names = sorted(set().union(*(row["host_actions"] for row in rows)))
        prof["host_actions"] = {
            name: _median(row["host_actions"].get(name) for row in rows)
            for name in names
        }
        out["profiles"][key] = prof
    return out


def _logits_path(path: Path, arm: str, key: str) -> Path | None:
    model, batch_tag = key.split(".b")
    prefix = "mlp" if model == "mnist_mlp" else "vit"
    candidates = (
        path / f"model_eval.{prefix}_b{batch_tag}.logits.npy",
        path / f"{arm}.{model}_b{batch_tag}.logits.npy",
    )
    return next((p for p in candidates if p.is_file()), None)


def _numeric_comparison(results: Path, arms: list[str],
                        reference_arm: str) -> dict:
    cmp: dict[str, Any] = {}
    for arm in arms:
        if arm == reference_arm:
            continue
        rounds = []
        for ref_dir in _round_dirs(results, reference_arm):
            suffix = ref_dir.name.split("-", 1)[1]
            arm_dir = ref_dir.with_name(ref_dir.name[:-len(suffix)] + arm)
            if not arm_dir.is_dir():
                continue
            profile_rows = {}
            for key in PROFILE_KEYS:
                ref_path = _logits_path(ref_dir, reference_arm, key)
                got_path = _logits_path(arm_dir, arm, key)
                if ref_path is None or got_path is None:
                    continue
                ref = np.load(ref_path)
                got = np.load(got_path)
                if ref.shape != got.shape:
                    raise RuntimeError(
                        f"{arm}/{key}: logits shape {got.shape} != {ref.shape}")
                delta = np.abs(ref.astype(np.float64) -
                               got.astype(np.float64))
                profile_rows[key] = {
                    "bitwise_equal": bool(np.array_equal(ref, got)),
                    "max_abs": float(delta.max()) if delta.size else 0.0,
                    "n": int(ref.shape[0]),
                }
            trace_rows = {}
            ref_trace = ref_dir / "model_eval.layers.npz"
            got_trace = arm_dir / "model_eval.layers.npz"
            if ref_trace.is_file() and got_trace.is_file():
                with np.load(ref_trace, allow_pickle=False) as a, \
                        np.load(got_trace, allow_pickle=False) as b:
                    common = sorted(set(a.files) & set(b.files))
                    trace_rows = {
                        name: {
                            "bitwise_equal": bool(np.array_equal(a[name], b[name])),
                            "shape_equal": list(a[name].shape) == list(b[name].shape),
                        }
                        for name in common
                    }
            rounds.append({"reference_round": ref_dir.name,
                           "profiles": profile_rows, "trace": trace_rows})
        cmp[arm] = rounds
    return cmp


def _external_gap(rows: dict[str, dict], external: str,
                  reference: str) -> dict[str, float | None]:
    return {
        key: (rows[reference]["profiles"][key]["warm_ms"] /
              rows[external]["profiles"][key]["warm_ms"])
        for key in PROFILE_KEYS
    }


def _write_markdown(summary: dict, path: Path) -> None:
    profiles = summary["profiles"]
    custom = summary["custom_arms"]
    external = summary["external_arms"]
    reference = summary.get("reference_arm", "generic")
    lines = [
        "# Execution-plan candidate comparison",
        "",
        f"- status: **{summary['status']}**",
        f"- rounds: {summary['custom_rounds']} custom, "
        f"{summary['external_rounds']} external",
        f"- full 10,000-image profiles: {summary['full_10k_all_profiles']}",
        f"- same-policy reference arm: `{reference}`",
        "",
        "Warm latency is per static batch call: raw preparation + execution + "
        "output materialisation, excluding argmax and references.  Speedup is "
        "`reference_ms / arm_ms`; >1 means the arm is faster.",
        "",
        "## Same-policy compiler arms",
        "",
        "| profile | " + " | ".join(f"{a} ms" for a in custom) +
        " | " + " | ".join(f"{a}/{reference}" for a in custom if a != reference) +
        " |",
        "|---|" + "---:|" * (2 * len(custom) - 1),
    ]
    for key in PROFILE_KEYS:
        vals = [profiles[key][arm]["warm_ms"] for arm in custom]
        ratios = [profiles[key][reference]["warm_ms"] /
                  profiles[key][arm]["warm_ms"]
                  for arm in custom if arm != reference]
        lines.append("| " + key + " | " +
                     " | ".join(f"{v:.3f}" for v in vals) + " | " +
                     " | ".join(f"{v:.3f}x" for v in ratios) + " |")
    lines += [
        "",
        "## External baselines (separate numeric contracts)",
        "",
        "| profile | same-policy reference ms | vendor ms | CPU ms | "
        "vendor/ref | CPU/ref | vendor max err | packed atol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in PROFILE_KEYS:
        ref = profiles[key][reference]["warm_ms"]
        v = profiles[key].get("vendor")
        c = profiles[key].get("cpu")
        atol = profiles[key].get("packed_atol")
        ev = f"{ref / v['warm_ms']:.3f}x" if v else "-"
        ec = f"{ref / c['warm_ms']:.3f}x" if c else "-"
        vm = f"{v['warm_ms']:.3f}" if v else "-"
        cm = f"{c['warm_ms']:.3f}" if c else "-"
        ve = f"{v['max_abs_logit_error']:.5f}" if v else "-"
        lines.append(f"| {key} | {ref:.3f} | {vm} | {cm} | {ev} | {ec} | "
                     f"{ve} | {atol if atol is not None else '-'} |")
    lines += [
        "",
        "Vendor is class-preserving but uses non-quantized FP16 graph fusion; "
        "its per-logit contract is not the custom packed policy.  CPU is FP32 "
        "reference math.  Neither is ranked against the custom arms.",
        "",
        "## Prepared lowering cost decomposition",
        "",
        "| profile | arm | warm ms | Python plan-node ms | Host actions ms | "
        "C workspace ms | executor ms | submit/call | tasks/call |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in PROFILE_KEYS:
        for arm in custom:
            row = profiles[key][arm]
            pc = row.get("per_call", {})
            if not pc:
                continue
            py_ms = (pc.get("plan_call_us", 0.0) -
                     pc.get("plan_workspace_us", 0.0)) / 1e3
            lines.append(
                f"| {key} | {arm} | {row['warm_ms']:.3f} | {py_ms:.3f} | "
                f"{pc.get('host_action_us', 0.0)/1e3:.3f} | "
                f"{pc.get('plan_workspace_us', 0.0)/1e3:.3f} | "
                f"{pc.get('executor_overhead_us', 0.0)/1e3:.3f} | "
                f"{pc.get('submit_count', 0.0):.1f} | "
                f"{pc.get('task_count', 0.0):.1f} |")
    lines += [
        "",
        "## Numeric equality against the same-policy reference",
        "",
        "| arm | all profiles bitwise | max logit delta | all trace bitwise |",
        "|---|---:|---:|---:|",
    ]
    for arm, rounds in summary["numeric_compare"].items():
        profile_rows = [row for rnd in rounds
                        for row in rnd["profiles"].values()]
        traces = [row for rnd in rounds for row in rnd["trace"].values()]
        bitwise = all(row["bitwise_equal"] for row in profile_rows)
        max_delta = max((row["max_abs"] for row in profile_rows), default=0.0)
        trace_bitwise = all(row["bitwise_equal"] for row in traces)
        lines.append(f"| {arm} | {bitwise} | {max_delta:.9g} | "
                     f"{trace_bitwise} |")
    lines += ["", "## Problems", ""]
    lines += [f"- {item}" for item in summary["problems"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--arms", nargs="+",
                    default=["generic", "prepared", "hostfused", "vendor", "cpu"])
    ap.add_argument("--reference-arm", default="generic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    results = args.results.resolve()
    campaign = json.loads((args.campaign / "manifest.json").read_text())
    rows = {arm: _arm_table(results, arm) for arm in args.arms}
    custom = [arm for arm in args.arms if rows[arm]["family"] == "custom"]
    external = [arm for arm in args.arms if rows[arm]["family"] == "external"]
    if args.reference_arm not in custom:
        raise SystemExit("reference arm must be a custom arm")

    packed_path = args.campaign / campaign["numeric"]["packed_spec"]["file"]
    packed = json.loads(packed_path.read_text())
    atol = {key: float(value["atol"])
            for key, value in packed["task_logit_tolerances"].items()}

    profiles = {}
    problems: list[str] = []
    for key in PROFILE_KEYS:
        profiles[key] = {}
        for arm in args.arms:
            row = rows[arm]["profiles"][key]
            profiles[key][arm] = row
        profiles[key]["packed_atol"] = atol.get(key)
        if not all(rows[arm]["profiles"][key]["full_test_split"]
                   for arm in custom):
            problems.append(f"{key}: a custom arm did not evaluate 10,000 images")
        if not all(rows[arm]["profiles"][key]["n"] == 10000
                   for arm in custom):
            problems.append(f"{key}: custom sample count is not 10000")
        ref = profiles[key][args.reference_arm]
        for arm in custom:
            got = profiles[key][arm]
            if got["accuracy"] + 1e-12 < ref["accuracy"]:
                problems.append(f"{key}: {arm} accuracy is below reference")
            if got["top1"] + 1e-12 < ref["top1"]:
                problems.append(f"{key}: {arm} top-1 agreement is below reference")
        for arm in external:
            got = profiles[key][arm]
            if got["n"] != 10000:
                problems.append(f"{key}: {arm} sample count is not 10000")

    numeric = _numeric_comparison(results, custom, args.reference_arm)
    for arm, rounds in numeric.items():
        if not rounds:
            problems.append(f"{arm}: no paired rounds with reference")
        for rnd in rounds:
            if set(rnd["profiles"]) != set(PROFILE_KEYS):
                problems.append(f"{arm}: missing profile in paired logits")
            for key, row in rnd["profiles"].items():
                if not row["bitwise_equal"]:
                    problems.append(
                        f"{arm}/{key}: logits differ from {args.reference_arm} "
                        f"(max {row['max_abs']:.9g})")
            for name, row in rnd["trace"].items():
                if not row["shape_equal"] or not row["bitwise_equal"]:
                    problems.append(
                        f"{arm}: trace tensor {name} differs from "
                        f"{args.reference_arm}")

    ratios = {}
    for arm in custom:
        if arm == args.reference_arm:
            continue
        ratios[arm] = {
            key: profiles[key][args.reference_arm]["warm_ms"] /
                 profiles[key][arm]["warm_ms"]
            for key in PROFILE_KEYS
        }
    full_10k = all(rows[arm]["profiles"][key]["full_test_split"]
                   for arm in args.arms for key in PROFILE_KEYS)
    status = "PASS" if not problems else "FAIL"
    summary = {
        "schema_version": 1,
        "kind": "execution-candidate-comparison",
        "status": status,
        "campaign": campaign,
        "custom_arms": custom,
        "external_arms": external,
        "reference_arm": args.reference_arm,
        "custom_rounds": min((rows[a]["rounds"] for a in custom), default=0),
        "external_rounds": min((rows[a]["rounds"] for a in external),
                               default=0),
        "full_10k_all_profiles": bool(full_10k),
        "profiles": profiles,
        "speedups_vs_reference": ratios,
        "numeric_compare": numeric,
        "problems": problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, args.markdown)
    print(json.dumps({"status": status, "out": str(args.out),
                      "problems": problems[:20]}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
