#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent, fail-closed verifier for the complete-model board campaign.

The verifier trusts neither summary accuracy nor stored reference arrays.  It
loads raw hardware logits, recomputes labels/CPU/declared references from the
campaign weights and dataset, checks raw finite values and frozen tolerances, and
recomputes deterministic layer traces.  A classification-preserving logit scale
is therefore still rejected.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

TILELANG_ROOT = Path(__file__).resolve().parents[1]
if str(TILELANG_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(TILELANG_ROOT / "python"))

from rknpu_backend import load_plan  # noqa: E402
from rknpu_backend.hybrid import load_hybrid_program  # noqa: E402
from rknpu_backend.model_reference import (mlp_logits, mlp_logits_declared,  # noqa: E402
                                           tiny_vit_logits,
                                           tiny_vit_logits_declared)
from rknpu_backend.models import load_weight_npz  # noqa: E402
from rknpu_backend.numeric import golden_arrays  # noqa: E402

EXPECTED_ROWS = {
    "mnist_mlp": (32, 8, 1),
    "tiny_vit": (8, 1),
}
PRIMARY = {"mnist_mlp": 32, "tiny_vit": 8}
SHORT = {"mnist_mlp": "mlp", "tiny_vit": "vit"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def schedule_for(model: str, batch: int, n: int) -> list[tuple[int, int]]:
    if model == "mnist_mlp":
        if batch == 32:
            rem, tail_profiles = n % 32, (8, 1)
        elif batch == 8:
            rem, tail_profiles = n % 8, (1,)
        elif batch == 1:
            rem, tail_profiles = 0, ()
        else:
            raise ValueError(batch)
    elif model == "tiny_vit":
        if batch not in (1, 8):
            raise ValueError(batch)
        rem, tail_profiles = n % batch, ()
    else:
        raise ValueError(model)
    rows = [(i * batch, batch) for i in range(n // batch)]
    offset = (n // batch) * batch
    while rem:
        size = next((b for b in tail_profiles if b <= rem), None)
        if size is None:
            raise ValueError(f"uncompiled tail for {model} b{batch} n={n}")
        rows.append((offset, size))
        offset += size
        rem -= size
    if len(rows) == 0 or sum(size for _, size in rows) != n:
        raise ValueError("batch schedule does not cover samples")
    return rows


def _refs_with_schedule(kind: str, model: str, weights, x: np.ndarray,
                        batch: int) -> np.ndarray:
    fn = {"cpu": {"mnist_mlp": mlp_logits, "tiny_vit": tiny_vit_logits},
          "declared": {"mnist_mlp": mlp_logits_declared,
                       "tiny_vit": tiny_vit_logits_declared}}[kind][model]
    out = np.empty((len(x), 10), dtype=np.float32)
    for offset, size in schedule_for(model, batch, len(x)):
        out[offset:offset + size] = fn(weights, x[offset:offset + size])
    return out


def logits_within_frozen_tolerance(logits: np.ndarray, expected: np.ndarray,
                                   tolerance: dict) -> tuple[bool, float]:
    """Return (accepted, max_abs_error); classification is not consulted."""
    a = np.asarray(logits)
    b = np.asarray(expected)
    if a.shape != b.shape or a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
        return False, float("inf")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return False, float("inf")
    atol = float(tolerance.get("atol", -1.0))
    rtol = float(tolerance.get("rtol", 0.0))
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        return False, float("inf")
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    limit = atol + rtol * np.abs(b.astype(np.float64))
    return bool(np.all(error <= limit + 1e-15)), float(error.max(initial=0.0))


def _verify_hashes(manifest: dict, campaign: Path, problems: list[str]
                   ) -> dict:
    report = {"weights": {}, "data": {}, "plans": {},
              "hybrid_manifests": {}, "numeric": {}}
    for name, expected in manifest.get("weights", {}).items():
        path = campaign / "weights" / name
        actual = sha256_file(path) if path.is_file() else None
        report["weights"][name] = {"expected": expected, "actual": actual,
                                   "ok": actual == expected}
        if actual != expected:
            problems.append(f"weights hash mismatch: {name}")
    for name, expected in manifest.get("data", {}).items():
        path = campaign / "data" / name
        actual = sha256_file(path) if path.is_file() else None
        report["data"][name] = {"expected": expected, "actual": actual,
                                "ok": actual == expected}
        if actual != expected:
            problems.append(f"dataset hash mismatch: {name}")
    compiled = manifest.get("compiled_manifest", {})
    for name, info in compiled.get("plans", {}).items():
        path = campaign / "plans" / info["path"]
        try:
            plan = load_plan(str(path))
            ok = plan.plan_hash == info["plan_hash"]
        except Exception as exc:
            report["plans"][name] = {"ok": False, "error": repr(exc)}
            problems.append(f"plan cannot be loaded/verified: {name}")
            continue
        report["plans"][name] = {"expected": info["plan_hash"],
                                 "actual": plan.plan_hash, "ok": ok}
        if not ok:
            problems.append(f"plan hash mismatch: {name}")
    for name, info in compiled.get("hybrid_manifests", {}).items():
        path = campaign / "plans" / info["path"]
        actual = sha256_file(path) if path.is_file() else None
        report["hybrid_manifests"][name] = {
            "expected": info["sha256"], "actual": actual,
            "ok": actual == info["sha256"]}
        if actual != info["sha256"]:
            problems.append(f"hybrid manifest hash mismatch: {name}")
        else:
            try:
                load_hybrid_program(path, plans_dir=campaign / "plans")
            except Exception as exc:
                problems.append(f"hybrid manifest validation failed: {name}: {exc}")
                report["hybrid_manifests"][name]["ok"] = False
    numeric_meta = manifest.get("numeric", {})
    for name, info in numeric_meta.items():
        path = campaign / "numeric" / info["file"]
        actual = sha256_file(path) if path.is_file() else None
        report["numeric"][name] = {"expected": info["sha256"], "actual": actual,
                                   "ok": actual == info["sha256"]}
        if actual != info["sha256"]:
            problems.append(f"numeric artifact hash mismatch: {name}")
    # Verify source hashes against the local source tree used to make the bundle.
    local = TILELANG_ROOT
    source_checks = {}
    for rel, expected in manifest.get("source_hashes", {}).items():
        path = local / rel
        ok = path.is_file() and sha256_file(path) == expected
        source_checks[rel] = {"expected": expected,
                              "ok": ok, "exists": path.is_file()}
        if not ok:
            problems.append(f"source hash mismatch: {rel}")
    report["source_hashes"] = source_checks
    return report


def _row_metrics(logits: np.ndarray, cpu: np.ndarray, declared: np.ndarray,
                 labels: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    cpu_pred = cpu.argmax(axis=1)
    dec_pred = declared.argmax(axis=1)
    cpu_diff = np.abs(logits.astype(np.float32) - cpu.astype(np.float32))
    dec_diff = np.abs(logits.astype(np.float32) - declared.astype(np.float32))
    return {
        "accuracy": float((pred == labels).mean()),
        "cpu_accuracy": float((cpu_pred == labels).mean()),
        "accuracy_delta": float((pred == labels).mean() -
                                (cpu_pred == labels).mean()),
        "top1_agreement": float((pred == cpu_pred).mean()),
        "declared_accuracy": float((dec_pred == labels).mean()),
        "declared_top1_agreement": float((pred == dec_pred).mean()),
        "mean_abs_logit_error": float(cpu_diff.mean()),
        "max_abs_logit_error": float(cpu_diff.max(initial=0.0)),
        "declared_mean_abs_logit_error": float(dec_diff.mean()),
        "declared_max_abs_logit_error": float(dec_diff.max(initial=0.0)),
    }


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _verify_timing(row: dict, row_key: str, n: int, problems: list[str]
                   ) -> dict:
    timing = row.get("timing")
    stats = row.get("stats", {})
    if not isinstance(timing, dict):
        problems.append(f"{row_key}: missing timing schema v2")
        return {}
    required = ("weight_load_seconds", "runner_setup_seconds", "warmup_seconds",
                "preprocess_seconds", "execution_seconds",
                "postprocess_seconds", "argmax_seconds", "model_call_seconds",
                "end_to_end_seconds", "plan_call_seconds",
                "plan_workspace_seconds", "host_action_seconds",
                "executor_overhead_seconds", "reference_seconds")
    missing = [key for key in required if key not in timing]
    if missing:
        problems.append(f"{row_key}: timing missing {missing}")
    if timing.get("timing_schema_version") != 2:
        problems.append(f"{row_key}: timing_schema_version is not 2")
    if timing.get("reference_timing") != "excluded_from_model_interval":
        problems.append(f"{row_key}: reference timing is not excluded")
    for key in ("preprocess_seconds", "execution_seconds",
                "postprocess_seconds", "model_call_seconds",
                "end_to_end_seconds", "plan_call_seconds",
                "host_action_seconds", "weight_load_seconds",
                "runner_setup_seconds", "warmup_seconds"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
                or float(value) < 0:
            problems.append(f"{row_key}: invalid timing {key}")
    end = float(timing.get("end_to_end_seconds", -1))
    model_s = float(timing.get("model_call_seconds", -1))
    argmax_s = float(timing.get("argmax_seconds", -1))
    if end <= 0 or model_s <= 0:
        problems.append(f"{row_key}: non-positive model interval")
    if not _close(end, model_s + argmax_s, max(1e-9, end * 1e-6)):
        problems.append(f"{row_key}: end-to-end is not model+argmax")
    expected_throughput = n / max(1e-12, end)
    if not _close(float(row.get("throughput_images_s", -1)),
                  expected_throughput, max(1e-6, expected_throughput * 1e-5)):
        problems.append(f"{row_key}: throughput does not use end-to-end interval")
    if row.get("throughput_basis", "").find("references excluded") < 0:
        problems.append(f"{row_key}: throughput basis does not exclude references")
    refs = timing.get("reference_seconds", {})
    if not isinstance(refs, dict) or set(refs) != {"cpu_fp32",
                                                    "declared_fp16_path"}:
        problems.append(f"{row_key}: reference seconds are not separate")
    if stats.get("timing_schema_version") != 2:
        problems.append(f"{row_key}: stats timing schema is not 2")
    if stats.get("reference_timing") != "excluded":
        problems.append(f"{row_key}: stats include reference time")
    if float(stats.get("executor_overhead_us", -1)) < 0:
        problems.append(f"{row_key}: executor overhead is not measured")
    plan_us = float(stats.get("plan_call_us", -1))
    end_us = float(stats.get("end_to_end_us", -1))
    workspace_us = float(stats.get("plan_workspace_us", -1))
    if plan_us < 0 or workspace_us < 0 or workspace_us > plan_us * 1.01 + 1.0:
        problems.append(f"{row_key}: plan workspace interval exceeds plan node")
    unattributed = float(timing.get("unattributed_seconds", -1))
    if unattributed < 0 or unattributed > max(0.01, end * 0.01):
        problems.append(f"{row_key}: model interval has unaccounted Host time")
    action_names = {item.get("name") for item in stats.get("host_actions", [])}
    plan_names = {item.get("name") for item in stats.get("plans", [])}
    if "mnist_mlp" in row_key:
        expected_actions = {"normalize", "pad_k", "export_logits"}
        expected_plans = {"mlp"}
    else:
        expected_actions = {
            "patchify", "patch_pad", "position_add", "rmsnorm1",
            "softmax", "merge_heads", "residual_attention", "rmsnorm2",
            "swiglu", "residual_ffn", "reshape_tokens", "mean_pool",
            "export_logits",
        }
        expected_plans = {"patch", "qkv", "qk", "pv", "attn_out",
                          "gate_up", "down", "classifier"}
    if not expected_actions <= action_names:
        problems.append(f"{row_key}: missing measured Host action nodes")
    if not expected_plans <= plan_names:
        problems.append(f"{row_key}: missing measured RKPlan nodes")
    schedule = row.get("batch_schedule", {})
    if schedule.get("no_dynamic_shapes") is not True:
        problems.append(f"{row_key}: batch schedule does not forbid dynamic shapes")
    return timing


def _load_numeric(campaign: Path, manifest: dict, problems: list[str]
                  ) -> dict:
    numeric_meta = manifest.get("numeric", {})
    spec_path = campaign / "numeric" / numeric_meta["spec"]["file"]
    golden_path = campaign / "numeric" / numeric_meta["golden"]["file"]
    spec = json.loads(spec_path.read_text())
    golden = np.load(golden_path, allow_pickle=False)
    if sha256_file(golden_path) != spec.get("golden_sha256"):
        problems.append("golden hash differs from numeric spec")
    if spec.get("status") != "frozen" or \
            spec.get("calibration", {}).get("status") != "frozen":
        problems.append("numeric acceptance is not frozen")
    indices = [int(v) for v in spec.get("sample_policy", {}).get("indices", [])]
    if indices != [int(v) for v in golden.get("sample_indices", [])]:
        problems.append("numeric sample indices differ from golden")
    if len(indices) != 8 or len(set(indices)) != len(indices):
        problems.append("numeric sample set is not eight unique indices")
    return {"spec_path": spec_path, "golden_path": golden_path,
            "spec": spec, "golden": golden}


def _expected_trace_keys(spec: dict) -> set[str]:
    keys: set[str] = set()
    for name, item in spec.get("trace_contract", {}).items():
        keys.update(item.get("keys", []))
    return keys


def _verify_layer_trace(result_path: Path, report: dict, numeric: dict,
                        weights: dict, x: np.ndarray, core: str,
                        problems: list[str]) -> dict:
    trace_info = report.get("layer_trace", {})
    if trace_info.get("status") != "captured":
        problems.append(f"{core}: deterministic layer trace not captured")
        return {"ok": False}
    trace_path = result_path.parent / trace_info.get("path", "")
    if not trace_path.is_file():
        problems.append(f"{core}: missing layer trace file")
        return {"ok": False}
    observed = np.load(trace_path, allow_pickle=False)
    spec, golden = numeric["spec"], numeric["golden"]
    indices = [int(v) for v in spec["sample_policy"]["indices"]]
    got_indices = [int(v) for v in observed.get("sample_indices", [])]
    if got_indices != indices:
        problems.append(f"{core}: layer trace sample set mismatch")
    expected_keys = _expected_trace_keys(spec)
    missing = expected_keys - set(observed.files)
    if missing:
        problems.append(f"{core}: layer trace missing {sorted(missing)}")
    extra = set(observed.files) - expected_keys - {"schema_version",
                                                     "sample_indices"}
    if extra:
        problems.append(f"{core}: unexpected layer trace tensors {sorted(extra)}")
    if [int(v) for v in trace_info.get("sample_indices", [])] != indices:
        problems.append(f"{core}: reported trace sample set mismatch")
    tolerances = spec.get("layer_tolerances", {})
    max_errors = {}
    for key in sorted(expected_keys & set(observed.files)):
        if key not in golden.files or key not in tolerances:
            problems.append(f"{core}: no golden/tolerance for {key}")
            continue
        tol = tolerances[key]
        ok, error = logits_within_frozen_tolerance(
            observed[key], golden[key], tol)
        max_errors[key] = {"max_abs": error, "atol": tol.get("atol"),
                           "ok": ok}
        if not ok:
            problems.append(f"{core}: layer tolerance exceeded: {key}")
        if not np.isfinite(observed[key]).all():
            problems.append(f"{core}: non-finite layer output: {key}")
    # Independent regeneration of the golden file from campaign weights and
    # dataset, not from any board artifact.
    recomputed = golden_arrays(weights["mnist_mlp"], weights["tiny_vit"], x)
    golden_checks = {}
    for key in sorted(expected_keys):
        if key not in golden.files or key not in recomputed:
            continue
        a, b = np.asarray(golden[key]), np.asarray(recomputed[key])
        diff = float(np.max(np.abs(a.astype(np.float64) -
                                   b.astype(np.float64)), initial=0.0))
        ok = a.shape == b.shape and diff <= 1e-6
        golden_checks[key] = {"max_abs_diff": diff, "ok": ok}
        if not ok:
            problems.append(f"{core}: golden disagrees with independent rebuild: {key}")
    return {"ok": not missing, "path": trace_path.name,
            "layers": max_errors, "golden_rebuild": golden_checks}


def verify_campaign(manifest_path: str | Path, results: str | Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    campaign = manifest_path.parent
    results = Path(results).resolve()
    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []
    summary: dict = {"status": "PASS", "problems": [],
                     "campaign_manifest": manifest_path.name,
                     "thresholds": manifest.get("thresholds", {}),
                     "hashes": _verify_hashes(manifest, campaign, problems),
                     "cores": {}}
    data_dir = campaign / manifest.get("payload", {}).get("data_dir", "data")
    weights_dir = campaign / manifest.get("payload", {}).get(
        "weights_dir", "weights")
    x = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    y = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    weights = {
        "mnist_mlp": load_weight_npz(weights_dir / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(weights_dir / "tiny_vit.npz"),
    }
    numeric = _load_numeric(campaign, manifest, problems)
    spec = numeric["spec"]
    th = manifest["thresholds"]
    ref_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for mask in (1, 2, 4):
        core = f"core-{mask}"
        core_dir = results / core
        result_path = core_dir / "model_eval.json"
        core_summary: dict = {"rows": {}}
        if not result_path.is_file():
            problems.append(f"{core}: missing model_eval.json")
            summary["cores"][core] = core_summary
            continue
        report = json.loads(result_path.read_text())
        expected_n = 10000 if mask == 1 else 512
        if report.get("backend") != "board":
            problems.append(f"{core}: backend is not board")
        if report.get("test_images") != expected_n:
            problems.append(f"{core}: test image count mismatch")
        expected_set = "full_test_split" if mask == 1 else "admission_subsample"
        if report.get("evaluation_set") != expected_set:
            problems.append(f"{core}: evaluation_set mismatch")
        if report.get("timing_schema_version") != 2:
            problems.append(f"{core}: report timing schema is not 2")
        if report.get("numeric_spec", {}).get("calibration_status") != "frozen":
            problems.append(f"{core}: evaluator did not use a frozen numeric spec")
        thread_cfg = report.get("cpu_thread_config", {})
        if not thread_cfg.get("cpu_affinity_count"):
            problems.append(f"{core}: CPU affinity configuration missing")
        rows = {(row.get("model"), row.get("batch")): row
                for row in report.get("models", [])}
        expected_rows = {(model, batch) for model in EXPECTED_ROWS
                         for batch in EXPECTED_ROWS[model]}
        if set(rows) != expected_rows:
            problems.append(f"{core}: profile set mismatch")
        for model in EXPECTED_ROWS:
            for batch in EXPECTED_ROWS[model]:
                key = f"{model}.b{batch}"
                row = rows.get((model, batch))
                if row is None:
                    continue
                if mask == 1:
                    is_primary = batch == PRIMARY[model]
                    row_n = 10000 if is_primary else 512
                    row_set = "full_test_split" if is_primary else "admission_subsample"
                    row_full = is_primary
                else:
                    row_n = 512
                    row_set = "admission_subsample"
                    row_full = False
                if row.get("n") != row_n:
                    problems.append(f"{core}/{key}: n mismatch")
                    continue
                if report.get("test_images", 0) < row_n:
                    problems.append(f"{core}/{key}: dataset too short")
                    continue
                if row.get("evaluation_set") != row_set or \
                        bool(row.get("full_test_split")) != bool(row_full):
                    problems.append(f"{core}/{key}: evaluation-set flags mismatch")
                short = SHORT[model]
                logits_path = result_path.with_name(
                    result_path.stem + f".{short}_b{batch}.logits.npy")
                if not logits_path.is_file():
                    problems.append(f"{core}/{key}: missing raw logits")
                    continue
                logits = np.load(logits_path, allow_pickle=False)
                if logits.shape != (row_n, 10) or logits.dtype != np.float32:
                    problems.append(f"{core}/{key}: raw logits shape/dtype invalid")
                    continue
                if not np.isfinite(logits).all():
                    problems.append(f"{core}/{key}: non-finite raw logits")
                cache_key = (model, batch, row_n)
                if cache_key not in ref_cache:
                    cpu = _refs_with_schedule("cpu", model,
                                              weights[model], x[:row_n], batch)
                    declared = _refs_with_schedule(
                        "declared", model, weights[model], x[:row_n], batch)
                    ref_cache[cache_key] = (cpu, declared)
                cpu, declared = ref_cache[cache_key]
                recomputed = _row_metrics(logits, cpu, declared, y[:row_n])
                exact_fields = (
                    "accuracy", "cpu_accuracy", "accuracy_delta",
                    "top1_agreement", "declared_accuracy",
                    "declared_top1_agreement",
                )
                for field in exact_fields:
                    reported = row.get(field)
                    if not isinstance(reported, (int, float)) or \
                            not _close(float(reported), recomputed[field], 1e-9):
                        problems.append(
                            f"{core}/{key}: reported {field} differs from raw logits")
                # FP32/FP16 reference reductions are not bit-portable across
                # x86 and AArch64 BLAS.  Error summaries therefore allow a small
                # reference-portability delta; the raw-logit tolerance gate above
                # remains against this verifier's own recomputed reference.
                portability = {
                    "mean_abs_logit_error": 1e-3,
                    "declared_mean_abs_logit_error": 1e-3,
                    "max_abs_logit_error": 2e-2,
                    "declared_max_abs_logit_error": 2e-2,
                }
                for field, limit in portability.items():
                    reported = row.get(field)
                    if not isinstance(reported, (int, float)) or \
                            not math.isfinite(float(reported)) or \
                            abs(float(reported) - recomputed[field]) > limit:
                        problems.append(
                            f"{core}/{key}: reported {field} not consistent with "
                            "independent reference within portability limit")
                tol = spec.get("task_logit_tolerances", {}).get(key)
                if not isinstance(tol, dict):
                    problems.append(f"{core}/{key}: missing frozen logit tolerance")
                else:
                    ok, error = logits_within_frozen_tolerance(
                        logits, declared, tol)
                    if not ok:
                        problems.append(
                            f"{core}/{key}: raw logits outside frozen tolerance")
                threshold = th["mnist_mlp_accuracy"] if model == "mnist_mlp" \
                    else th["tiny_vit_accuracy"]
                if recomputed["accuracy"] < threshold:
                    problems.append(f"{core}/{key}: accuracy below threshold")
                if recomputed["accuracy_delta"] < -float(th["accuracy_drop_max"]):
                    problems.append(f"{core}/{key}: accuracy drop exceeds limit")
                if recomputed["top1_agreement"] < float(th["top1_agreement_min"]):
                    problems.append(f"{core}/{key}: CPU top-1 agreement low")
                if recomputed["declared_top1_agreement"] < \
                        float(th["top1_agreement_min"]):
                    problems.append(f"{core}/{key}: declared top-1 agreement low")
                if not np.isfinite(logits).all():
                    problems.append(f"{core}/{key}: non-finite hardware logits")
                _verify_timing(row, f"{core}/{key}", row_n, problems)
                core_summary["rows"][key] = {
                    **{k: recomputed[k] for k in
                       ("accuracy", "cpu_accuracy", "accuracy_delta",
                        "top1_agreement", "declared_top1_agreement",
                        "max_abs_logit_error",
                        "declared_max_abs_logit_error")},
                    "n": row_n,
                    "timing": row.get("timing"),
                    "batch_schedule": row.get("batch_schedule"),
                }
        core_summary["layer_trace"] = _verify_layer_trace(
            result_path, report, numeric, weights, x, core, problems)
        summary["cores"][core] = core_summary
    cpu_path = results / "core-1" / "cpu_benchmark.json"
    if not cpu_path.is_file():
        problems.append("core-1: missing matched-batch CPU benchmark")
    else:
        cpu = json.loads(cpu_path.read_text())
        summary["cpu_benchmark"] = {"batch_policy": cpu.get("batch_policy"),
                                    "thread_config": cpu.get("cpu_thread_config")}
        if cpu.get("test_images") != 10000:
            problems.append("core-1: CPU benchmark did not use 10000 images")
        policy = cpu.get("batch_policy", {})
        if policy.get("mnist_mlp", {}).get("batch_histogram") != \
                {"32": 312, "8": 2}:
            problems.append("core-1: CPU MLP schedule is not matched to NPU")
        if policy.get("tiny_vit", {}).get("batch_histogram") != {"8": 1250}:
            problems.append("core-1: CPU ViT schedule is not matched to NPU")
        cfg = cpu.get("cpu_thread_config", {})
        env = cfg.get("cpu_thread_env", {})
        explicit_single = all(str(env.get(key)) == "1" for key in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "MKL_NUM_THREADS"))
        if not cfg.get("cpu_affinity_count") or not (
                cfg.get("blas") or explicit_single):
            problems.append("core-1: CPU benchmark thread configuration missing")
        if not explicit_single:
            problems.append("core-1: CPU benchmark is not explicitly "
                            "single-threaded (thread env not all 1)")
        cm = cpu.get("models", {})
        if float(cm.get("mnist_mlp", {}).get("fp32_accuracy", -1)) < \
                th["mnist_mlp_accuracy"]:
            problems.append("core-1: CPU MLP accuracy below threshold")
        if float(cm.get("tiny_vit", {}).get("fp32_accuracy", -1)) < \
                th["tiny_vit_accuracy"]:
            problems.append("core-1: CPU ViT accuracy below threshold")
        for model in ("mnist_mlp", "tiny_vit"):
            item = cm.get(model, {})
            end = float(item.get("end_to_end_seconds", -1))
            execution = float(item.get("execution_seconds", -1))
            post = float(item.get("postprocess_seconds", -1))
            if end <= 0 or execution <= 0 or not _close(end, execution + post,
                                                         max(1e-9, end * 1e-6)):
                problems.append(f"core-1: CPU {model} timing decomposition invalid")
            throughput = float(item.get("throughput_images_s", -1))
            if not _close(throughput, 10000.0 / max(1e-12, end),
                          max(1e-6, throughput * 1e-5)):
                problems.append(f"core-1: CPU {model} throughput basis invalid")
    summary["numeric"] = {
        "status": spec.get("status"),
        "sample_indices": spec.get("sample_policy", {}).get("indices"),
        "calibration": spec.get("calibration"),
        "task_logit_tolerances": spec.get("task_logit_tolerances"),
    }
    summary["problems"] = problems
    if problems:
        summary["status"] = "FAIL"
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    summary = verify_campaign(args.manifest, args.results)
    out = Path(args.out) if args.out else Path(args.results) / "verify-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
