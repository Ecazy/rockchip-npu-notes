#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze tolerances from the fixed train-split calibration observations only.

This tool never reads or hashes the acceptance reference file and never opens any
label file.  A failed acceptance cannot trigger a threshold recalibration here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rknpu_backend.numeric import CALIBRATION_SPLIT
from rknpu_backend.numeric_policy import freeze_atol

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def max_abs(expected: np.ndarray, observed: np.ndarray) -> float:
    if expected.shape != observed.shape:
        raise SystemExit(f"shape mismatch {expected.shape} != {observed.shape}")
    return float(np.max(np.abs(expected.astype(np.float64) -
                              observed.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(ROOT / "models" /
                                           "numeric_acceptance_train.json"))
    ap.add_argument("--calibration", default=str(
        ROOT / "models" / "numeric_calibration_train.npz"))
    ap.add_argument("--observation", required=True)
    ap.add_argument("--metadata", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    if spec.get("status") != "uncalibrated":
        raise SystemExit("refusing to overwrite a non-uncalibrated spec")
    if spec["calibration"]["split"] != CALIBRATION_SPLIT:
        raise SystemExit("calibration split is not train")
    expected_path = Path(args.calibration)
    if sha256(expected_path) != spec["calibration"]["expected_sha256"]:
        raise SystemExit("calibration expected-file hash mismatch")
    obs_path = Path(args.observation)
    meta_path = (Path(args.metadata) if args.metadata
                 else obs_path.with_suffix(".json"))
    obs_meta = json.loads(meta_path.read_text())
    if obs_meta.get("mode") != "calibration" or \
            obs_meta.get("split") != CALIBRATION_SPLIT:
        raise SystemExit("observation is not a train calibration run")
    if obs_meta.get("labels_read") is not False:
        raise SystemExit("calibration observation must not read labels")
    if obs_meta.get("sample_indices") != spec["calibration"]["indices"]:
        raise SystemExit("calibration sample indices mismatch")
    if obs_meta.get("sample_sha256") != spec["calibration"]["sample_sha256"]:
        raise SystemExit("calibration sample hashes mismatch")
    if obs_meta.get("weights_sha256") != spec["weights_sha256"]:
        raise SystemExit("calibration weights hashes mismatch")
    if obs_meta.get("source_sha256") != spec["calibration"]["source_sha256"]:
        raise SystemExit("calibration source hash mismatch")
    expected = np.load(expected_path, allow_pickle=False)
    observed = np.load(obs_path, allow_pickle=False)
    if not np.array_equal(expected["sample_indices"],
                          observed["sample_indices"]):
        raise SystemExit("calibration observation/reference indices differ")
    if list(expected["sample_sha256"]) != list(observed["sample_sha256"]):
        raise SystemExit("calibration observation/reference sample hashes differ")
    # Calibration intentionally does not open spec["acceptance"]["expected_file"].
    layer_tolerances = {}
    for key in sorted(spec["layer_tolerances"]):
        if key not in expected.files or key not in observed.files:
            raise SystemExit(f"missing calibration layer {key}")
        expected_value = expected[key]
        err = max_abs(expected_value, observed[key])
        atol, basis = freeze_atol(err, expected_value, key)
        layer_tolerances[key] = {
            "atol": atol, "rtol": 0.0,
            "observed_max_abs": err,
            "f16_rounding_floor": basis["f16_rounding_floor"],
            "expected_absmax_train": basis["expected_absmax_train"],
            "basis": basis["formula"] + "; train split only; labels never "
                     "read; test data not used",
        }
    task_tolerances = {}
    for key in sorted(spec["task_logit_tolerances"]):
        array_key = f"profile.{key}.logits"
        if array_key not in expected.files or array_key not in observed.files:
            raise SystemExit(f"missing calibration profile logits {key}")
        expected_value = expected[array_key]
        err = max_abs(expected_value, observed[array_key])
        atol, basis = freeze_atol(err, expected_value, f"profile.{key}")
        task_tolerances[key] = {
            "atol": atol, "rtol": 0.0,
            "observed_max_abs": err,
            "f16_rounding_floor": basis["f16_rounding_floor"],
            "expected_absmax_train": basis["expected_absmax_train"],
            "basis": basis["formula"] + "; train split only; labels never "
                     "read; test data not used",
        }
    spec["status"] = "frozen"
    spec["layer_tolerances"] = layer_tolerances
    spec["task_logit_tolerances"] = task_tolerances
    spec["calibration_rule"] = {
        "status": "frozen",
        "formula": "device f16: atol=ceil_2^-12(max(2*observed,"
                   "2*f16_spacing(train_expected_absmax))); exact Host layout: "
                   "atol=ceil_2^-12(2*observed); rtol=0",
        "data_split": CALIBRATION_SPLIT,
        "labels_read": False,
        "test_data_used": False,
        "scope": "empirical envelope for the frozen model package and execution "
                 "paths on RK3588; not a theoretical accuracy or error bound",
        "failure_policy": "acceptance failure must be investigated as a numeric "
                          "or implementation defect; do not widen limits using "
                          "test-set maxima",
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observation_file": obs_path.name,
        "observation_sha256": sha256(obs_path),
        "observation_metadata_sha256": sha256(meta_path),
    }
    out = Path(args.out) if args.out else spec_path
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(out)
    for key, value in layer_tolerances.items():
        print(key, value["atol"])
    for key, value in task_tolerances.items():
        print("task", key, value["atol"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
