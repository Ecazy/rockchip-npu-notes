#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify train-calibrated numeric evidence and disjoint post-freeze test data."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from rknpu_backend import load_plan  # noqa: E402
from rknpu_backend.models import load_weight_npz  # noqa: E402
from rknpu_backend.numeric_policy import (freeze_atol,  # noqa: E402
                                          within_tolerance)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()




def layer_checks(expected: np.lib.npyio.NpzFile,
                 observed: np.lib.npyio.NpzFile,
                 tolerances: dict, *, label: str,
                 require_freezing: bool = False) -> tuple[dict, list[str]]:
    rows: dict[str, dict] = {}
    problems: list[str] = []
    for key in sorted(tolerances):
        tol = tolerances[key]
        if not isinstance(tol, dict):
            problems.append(f"{label}: missing tolerance for layer {key}")
            continue
        if key not in expected.files or key not in observed.files:
            problems.append(f"{label}: missing layer array {key}")
            continue
        expected_value = expected[key]
        observed_value = observed[key]
        ok, error = within_tolerance(observed_value, expected_value, tol)
        row = {"max_abs_error": error, "atol": tol.get("atol"),
               "rtol": tol.get("rtol"), "ok": ok}
        if require_freezing:
            measured = float(tol.get("observed_max_abs", float("nan")))
            expected_atol, _basis = freeze_atol(
                measured, expected_value, key)
            freeze_ok = (_close(error, measured) and
                         _close(float(tol.get("atol")), expected_atol))
            row["freeze_ok"] = freeze_ok
            if not freeze_ok:
                problems.append(
                    f"{label}: layer {key} threshold was not derived solely "
                    "from this calibration observation")
        if not ok:
            problems.append(f"{label}: layer tolerance exceeded for {key}")
        rows[key] = row
    return rows, problems


def profile_checks(expected: np.lib.npyio.NpzFile,
                   observed: np.lib.npyio.NpzFile,
                   tolerances: dict, *, label: str,
                   require_freezing: bool = False) -> tuple[dict, list[str]]:
    rows: dict[str, dict] = {}
    problems: list[str] = []
    for key in sorted(tolerances):
        tol = tolerances[key]
        if not isinstance(tol, dict):
            problems.append(f"{label}: missing tolerance for profile {key}")
            continue
        array_key = f"profile.{key}.logits"
        if array_key not in expected.files or array_key not in observed.files:
            problems.append(f"{label}: missing profile logits {key}")
            continue
        expected_value = expected[array_key]
        observed_value = observed[array_key]
        ok, error = within_tolerance(observed_value, expected_value, tol)
        row = {"max_abs_error": error, "atol": tol.get("atol"),
               "rtol": tol.get("rtol"), "ok": ok,
               "same_top1": bool(np.array_equal(
                   observed_value.argmax(axis=1),
                   expected_value.argmax(axis=1)))}
        if require_freezing:
            measured = float(tol.get("observed_max_abs", float("nan")))
            expected_atol, _basis = freeze_atol(
                measured, expected_value, f"profile.{key}")
            freeze_ok = (_close(error, measured) and
                         _close(float(tol.get("atol")), expected_atol))
            row["freeze_ok"] = freeze_ok
            if not freeze_ok:
                problems.append(
                    f"{label}: profile {key} threshold was not derived solely "
                    "from this calibration observation")
        if not ok:
            problems.append(f"{label}: profile tolerance exceeded for {key}")
        rows[key] = row
    return rows, problems


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def _schedule(model: str, batch: int, n: int) -> list[tuple[int, int]]:
    if model == "mnist_mlp":
        rem, tails = n % batch, ((8, 1) if batch == 32 else
                                 ((1,) if batch == 8 else ()))
    elif model == "tiny_vit":
        if batch not in (1, 8):
            raise ValueError(batch)
        rem, tails = n % batch, ()
    else:
        raise ValueError(model)
    rows = [(i * batch, batch) for i in range(n // batch)]
    offset = (n // batch) * batch
    while rem:
        size = next((v for v in tails if v <= rem), None)
        if size is None:
            raise ValueError("no compiled tail")
        rows.append((offset, size))
        offset += size
        rem -= size
    return rows


def _declared_for_campaign(model: str, weights, x: np.ndarray, batch: int):
    from rknpu_backend.model_reference import (mlp_logits_declared,
                                              tiny_vit_logits_declared)
    fn = mlp_logits_declared if model == "mnist_mlp" else tiny_vit_logits_declared
    out = np.empty((len(x), 10), dtype=np.float32)
    for offset, size in _schedule(model, batch, len(x)):
        out[offset:offset + size] = fn(weights, x[offset:offset + size])
    return out


def _check_existing_campaign(spec: dict, campaign_manifest: Path,
                             campaign_results: Path, problems: list[str]) -> dict:
    from tools.verify_model_results import _refs_with_schedule, _row_metrics
    manifest = json.loads(campaign_manifest.read_text())
    campaign = campaign_manifest.parent
    data_dir = campaign / manifest.get("payload", {}).get("data_dir", "data")
    weights_dir = campaign / manifest.get("payload", {}).get(
        "weights_dir", "weights")
    x = _read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    y = _read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    weights = {
        "mnist_mlp": load_weight_npz(weights_dir / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(weights_dir / "tiny_vit.npz"),
    }
    short = {"mnist_mlp": "mlp", "tiny_vit": "vit"}
    summary: dict = {"rows": {}}
    refs: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for mask in (1, 2, 4):
        result_path = campaign_results / f"core-{mask}" / "model_eval.json"
        if not result_path.is_file():
            problems.append(f"existing campaign core-{mask} missing")
            continue
        report = json.loads(result_path.read_text())
        for row in report.get("models", []):
            model, batch, n = row["model"], int(row["batch"]), int(row["n"])
            key = f"{model}.b{batch}"
            if key not in spec["task_logit_tolerances"]:
                problems.append(f"existing campaign profile lacks tolerance: {key}")
                continue
            logits_path = result_path.with_name(
                result_path.stem + f".{short[model]}_b{batch}.logits.npy")
            if not logits_path.is_file():
                problems.append(f"existing campaign raw logits missing: core-{mask}/{key}")
                continue
            logits = np.load(logits_path, allow_pickle=False)
            if logits.shape != (n, 10) or not np.isfinite(logits).all():
                problems.append(f"existing campaign raw logits invalid: core-{mask}/{key}")
                continue
            cache_key = (model, batch, n)
            if cache_key not in refs:
                cpu = _refs_with_schedule("cpu", model, weights[model],
                                          x[:n], batch)
                declared = _refs_with_schedule("declared", model,
                                               weights[model], x[:n], batch)
                refs[cache_key] = (cpu, declared)
            cpu, declared = refs[cache_key]
            ok, error = within_tolerance(
                logits, declared, spec["task_logit_tolerances"][key])
            metrics = _row_metrics(logits, cpu, declared, y[:n])
            row_summary = {
                "n": n, "max_abs_declared_error": error,
                "atol": spec["task_logit_tolerances"][key]["atol"],
                "ok": ok, "accuracy": metrics["accuracy"],
                "cpu_accuracy": metrics["cpu_accuracy"],
                "top1_agreement": metrics["top1_agreement"],
            }
            summary["rows"][f"core-{mask}/{key}"] = row_summary
            if not ok:
                problems.append(
                    f"existing campaign raw logits exceed train-frozen "
                    f"tolerance: core-{mask}/{key}")
    return summary


def verify(spec_path: Path, calibration_observation: Path,
           acceptance_observation: Path, *,
           campaign_manifest: Path | None = None,
           campaign_results: Path | None = None,
           compiled_manifest: Path | None = None,
           plans_dir: Path | None = None) -> dict:
    spec = json.loads(spec_path.read_text())
    problems: list[str] = []
    row: dict = {"status": "PASS", "problems": problems,
                 "spec": spec_path.as_posix()}
    if spec.get("status") != "frozen":
        problems.append("numeric spec is not frozen")
    if spec.get("calibration_rule", {}).get("status") != "frozen":
        problems.append("calibration rule is not frozen")
    if spec.get("calibration_rule", {}).get("test_data_used") is not False:
        problems.append("calibration rule used test data")
    if spec.get("calibration", {}).get("split") != "train":
        problems.append("calibration split is not train")
    if spec.get("acceptance", {}).get("split") != "test":
        problems.append("acceptance split is not test")
    expected_cal_path = spec_path.parent / spec["calibration"]["expected_file"]
    expected_acc_path = spec_path.parent / spec["acceptance"]["expected_file"]
    if sha256(expected_cal_path) != spec["calibration"]["expected_sha256"]:
        problems.append("train calibration expected-file hash mismatch")
    if sha256(expected_acc_path) != spec["acceptance"]["expected_sha256"]:
        problems.append("test acceptance expected-file hash mismatch")
    cal_meta_path = calibration_observation.with_suffix(".json")
    acc_meta_path = acceptance_observation.with_suffix(".json")
    cal_meta = json.loads(cal_meta_path.read_text())
    acc_meta = json.loads(acc_meta_path.read_text())
    if sha256(calibration_observation) != cal_meta.get("observation_sha256"):
        problems.append("calibration observation file hash mismatch")
    if sha256(acceptance_observation) != acc_meta.get("observation_sha256"):
        problems.append("acceptance observation file hash mismatch")
    if cal_meta.get("mode") != "calibration" or cal_meta.get("split") != "train":
        problems.append("calibration observation is not train calibration")
    if cal_meta.get("labels_read") is not False:
        problems.append("calibration observation read labels")
    if acc_meta.get("mode") != "acceptance" or acc_meta.get("split") != "test":
        problems.append("acceptance observation is not test acceptance")
    if acc_meta.get("labels_read") is not False:
        problems.append("acceptance observation read labels")
    if acc_meta.get("spec_sha256") != sha256(spec_path):
        problems.append("acceptance observation did not use this frozen spec")
    if cal_meta.get("sample_indices") != spec["calibration"]["indices"]:
        problems.append("calibration indices differ from spec")
    if acc_meta.get("sample_indices") != spec["acceptance"]["indices"]:
        problems.append("acceptance indices differ from spec")
    if cal_meta.get("sample_sha256") != spec["calibration"]["sample_sha256"]:
        problems.append("calibration sample hashes differ from spec")
    if acc_meta.get("sample_sha256") != spec["acceptance"]["sample_sha256"]:
        problems.append("acceptance sample hashes differ from spec")
    if cal_meta.get("source_sha256") != spec["calibration"]["source_sha256"]:
        problems.append("calibration source file hash mismatch")
    if acc_meta.get("source_sha256") != spec["acceptance"]["source_sha256"]:
        problems.append("acceptance source file hash mismatch")
    if cal_meta.get("weights_sha256") != spec["weights_sha256"]:
        problems.append("calibration weights hash mismatch")
    if acc_meta.get("weights_sha256") != spec["weights_sha256"]:
        problems.append("acceptance weights hash mismatch")
    plans_root = (plans_dir if plans_dir is not None
                  else ROOT / "models" / "compiled")
    compiled_path = (compiled_manifest if compiled_manifest is not None
                     else plans_root / "manifest.json")
    if spec.get("compiled_manifest_sha256") != sha256(compiled_path):
        problems.append("spec compiled-manifest hash mismatch")
    if cal_meta.get("compiled_manifest_sha256") != spec.get(
            "compiled_manifest_sha256"):
        problems.append("calibration compiled-manifest hash mismatch")
    if acc_meta.get("compiled_manifest_sha256") != spec.get(
            "compiled_manifest_sha256"):
        problems.append("acceptance compiled-manifest hash mismatch")
    compiled = json.loads(compiled_path.read_text())
    for label, meta in (("calibration", cal_meta), ("acceptance", acc_meta)):
        for key, info in meta.get("plans", {}).items():
            package = compiled.get("plans", {}).get(key)
            if package is None or package.get("plan_hash") != info.get(
                    "plan_hash"):
                problems.append(f"{label} plan hash mismatch for {key}")
            else:
                actual_plan = load_plan(str(plans_root / package["path"]))
                if actual_plan.plan_hash != info.get("plan_hash"):
                    problems.append(f"{label} actual plan hash mismatch for {key}")
    # Explicit no-overlap proof: different official split and no repeated pixel
    # image.  The (split,index) identities are also disjoint by construction.
    cal_hashes = set(spec["calibration"]["sample_sha256"])
    acc_hashes = set(spec["acceptance"]["sample_sha256"])
    disjoint = bool(cal_hashes.isdisjoint(acc_hashes))
    if not disjoint:
        problems.append("calibration and acceptance samples overlap")
    if spec["calibration"]["source_sha256"] == \
            spec["acceptance"]["source_sha256"]:
        problems.append("calibration and acceptance use the same image source")
    expected_cal = np.load(expected_cal_path, allow_pickle=False)
    observed_cal = np.load(calibration_observation, allow_pickle=False)
    expected_acc = np.load(expected_acc_path, allow_pickle=False)
    observed_acc = np.load(acceptance_observation, allow_pickle=False)
    if list(expected_cal["sample_sha256"]) != list(observed_cal["sample_sha256"]):
        problems.append("calibration sample pixels differ from expected file")
    if list(expected_acc["sample_sha256"]) != list(observed_acc["sample_sha256"]):
        problems.append("acceptance sample pixels differ from expected file")
    cal_layers, p = layer_checks(
        expected_cal, observed_cal, spec["layer_tolerances"],
        label="calibration", require_freezing=True)
    problems.extend(p)
    cal_profiles, p = profile_checks(
        expected_cal, observed_cal, spec["task_logit_tolerances"],
        label="calibration", require_freezing=True)
    problems.extend(p)
    acc_layers, p = layer_checks(
        expected_acc, observed_acc, spec["layer_tolerances"],
        label="acceptance", require_freezing=False)
    problems.extend(p)
    acc_profiles, p = profile_checks(
        expected_acc, observed_acc, spec["task_logit_tolerances"],
        label="acceptance", require_freezing=False)
    problems.extend(p)
    campaign_summary = None
    if campaign_manifest is not None and campaign_results is not None:
        campaign_summary = _check_existing_campaign(
            spec, campaign_manifest, campaign_results, problems)
    row.update({
        "calibration": {"layers": cal_layers, "profiles": cal_profiles},
        "acceptance": {"layers": acc_layers, "profiles": acc_profiles},
        "sample_disjoint": {
            "train_test_splits_distinct": (
                spec["calibration"]["split"] != spec["acceptance"]["split"]),
            "pixel_hash_sets_disjoint": disjoint,
        },
        "existing_campaign": campaign_summary,
        "non_blind_disclosure": spec.get("non_blind_disclosure"),
    })
    if problems:
        row["status"] = "FAIL"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(ROOT / "models" /
                                           "numeric_acceptance_train.json"))
    ap.add_argument("--calibration-observation", required=True)
    ap.add_argument("--acceptance-observation", required=True)
    ap.add_argument("--campaign-manifest", default="")
    ap.add_argument("--campaign-results", default="")
    ap.add_argument("--compiled-manifest", default="")
    ap.add_argument("--plans-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    summary = verify(
        Path(args.spec), Path(args.calibration_observation),
        Path(args.acceptance_observation),
        campaign_manifest=(Path(args.campaign_manifest)
                           if args.campaign_manifest else None),
        campaign_results=(Path(args.campaign_results)
                          if args.campaign_results else None),
        compiled_manifest=(Path(args.compiled_manifest)
                           if args.compiled_manifest else None),
        plans_dir=(Path(args.plans_dir) if args.plans_dir else None))
    out = Path(args.out) if args.out else Path(
        args.acceptance_observation).with_suffix(".verify.json")
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
