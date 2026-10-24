#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build disjoint train-calibration / test-acceptance numeric reference sets.

Only image files are read.  No label file is opened in this tool, and no
tolerance is chosen here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from rknpu_backend.models import load_weight_npz
from rknpu_backend.numeric import (ACCEPTANCE_SPLIT, CALIBRATION_SPLIT,
                                   TRAIN_NUMERIC_SCHEMA_VERSION,
                                   expected_arrays, layer_contract_keys,
                                   profile_contract_keys, sample_image_hashes,
                                   save_npz_deterministic, selected_images,
                                   selection_metadata)

ROOT = Path(__file__).resolve().parents[1]


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--calibration-out", default=str(
        ROOT / "models" / "numeric_calibration_train.npz"))
    ap.add_argument("--acceptance-out", default=str(
        ROOT / "models" / "numeric_acceptance_test.npz"))
    ap.add_argument("--spec-out", default=str(
        ROOT / "models" / "numeric_acceptance_train.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    weights_mlp = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
    weights_vit = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
    train_images = read_idx(data_dir / "train-images-idx3-ubyte.gz", 3)
    test_images = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    calibration = expected_arrays(weights_mlp, weights_vit, train_images,
                                  split=CALIBRATION_SPLIT)
    acceptance = expected_arrays(weights_mlp, weights_vit, test_images,
                                 split=ACCEPTANCE_SPLIT)
    save_npz_deterministic(args.calibration_out, calibration)
    save_npz_deterministic(args.acceptance_out, acceptance)
    cal_meta = selection_metadata(CALIBRATION_SPLIT)
    acc_meta = selection_metadata(ACCEPTANCE_SPLIT)
    cal_images = selected_images(train_images, CALIBRATION_SPLIT)
    acc_images = selected_images(test_images, ACCEPTANCE_SPLIT)
    cal_source = data_dir / "train-images-idx3-ubyte.gz"
    acc_source = data_dir / "t10k-images-idx3-ubyte.gz"
    cal_meta.update({
        "source_file": cal_source.name,
        "source_sha256": sha256(cal_source),
        "expected_file": Path(args.calibration_out).name,
        "expected_sha256": sha256(Path(args.calibration_out)),
        "sample_sha256": sample_image_hashes(cal_images),
        "labels_read": False,
    })
    acc_meta.update({
        "source_file": acc_source.name,
        "source_sha256": sha256(acc_source),
        "expected_file": Path(args.acceptance_out).name,
        "expected_sha256": sha256(Path(args.acceptance_out)),
        "sample_sha256": sample_image_hashes(acc_images),
        "labels_read": False,
    })
    layer_keys = layer_contract_keys(calibration)
    spec = {
        "schema_version": TRAIN_NUMERIC_SCHEMA_VERSION,
        "weights_sha256": {
            "mnist_mlp.npz": sha256(Path(args.weights_dir) / "mnist_mlp.npz"),
            "tiny_vit.npz": sha256(Path(args.weights_dir) / "tiny_vit.npz"),
        },
        "compiled_manifest_sha256": sha256(
            ROOT / "models" / "compiled" / "manifest.json"),
        "kind": "train-calibrated-model-numeric-acceptance",
        "status": "uncalibrated",
        "calibration": cal_meta,
        "acceptance": acc_meta,
        "trace_contract": {
            "mlp_hidden": [f"mlp.{key}" for key in (
                "x_padded", "h1_pre_relu", "r1_relu", "h2_pre_relu",
                "r2_relu", "final_preactivation", "logits")],
            "tiny_vit": [key for key in layer_keys
                         if key.startswith("tiny_vit.")],
            "relu_probe": ["relu_probe.output"],
            "profiles": profile_contract_keys(),
        },
        "layer_tolerances": {key: None for key in layer_keys},
        "task_logit_tolerances": {key: None for key in profile_contract_keys()},
        "calibration_rule": {
            "status": "pending",
            "formula": "device f16: atol=ceil_2^-12(max(2*observed,"
                       "2*f16_spacing(train_expected_absmax))); exact Host "
                       "layout: atol=ceil_2^-12(2*observed); rtol=0",
            "observations": "train split only; labels never read",
            "test_data_used_for_tolerance": False,
            "scope": "empirical envelope for the frozen model package and "
                     "execution paths; not a theoretical accuracy guarantee",
        },
        "non_blind_disclosure": (
            "The complete test set and earlier test results were inspected before "
            "this train-calibration boundary was introduced. This is a disjoint "
            "post-freeze numerical acceptance rerun, not a blind benchmark."),
    }
    Path(args.spec_out).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(args.calibration_out)
    print(args.acceptance_out)
    print(args.spec_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
