#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build golden deterministic layer outputs and a calibration-ready spec."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from rknpu_backend.models import load_weight_npz
from rknpu_backend.numeric import (NUMERIC_SCHEMA_VERSION, SAMPLE_INDICES,
                                   golden_arrays)

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
    ap.add_argument("--golden", default=str(ROOT / "models" /
                                            "numeric_golden.npz"))
    ap.add_argument("--spec", default=str(ROOT / "models" /
                                           "numeric_acceptance.json"))
    args = ap.parse_args()
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    mlp = load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz")
    vit = load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz")
    golden = golden_arrays(mlp, vit, x)
    golden_path = Path(args.golden)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(golden_path, **golden)
    mlp_keys = ["mlp.x_padded", "mlp.logits"]
    vit_keys = [key for key in golden if key.startswith("tiny_vit.")]
    probe_keys = ["relu_probe.output"]
    layer_keys = mlp_keys + vit_keys + probe_keys
    task_profiles = [
        "mnist_mlp.b32", "mnist_mlp.b8", "mnist_mlp.b1",
        "tiny_vit.b8", "tiny_vit.b1",
    ]
    spec = {
        "schema_version": NUMERIC_SCHEMA_VERSION,
        "kind": "complete-model-numeric-acceptance",
        "status": "uncalibrated",
        "golden_file": golden_path.name,
        "golden_sha256": sha256(golden_path),
        "sample_policy": {
            "seed": 20261024,
            "population": "first 512 images of the MNIST test split",
            "method": "numpy_default_rng_choice_without_replacement",
            "count": len(SAMPLE_INDICES),
            "indices": list(SAMPLE_INDICES),
            "labels_used_for_selection": False,
        },
        "trace_contract": {
            "mlp": {
                "profile": "mnist_mlp_b8",
                "keys": mlp_keys,
            },
            "tiny_vit": {
                "profile": "tiny_vit_b8",
                "keys": vit_keys,
            },
            "relu_probe": {
                "plan": "mnist_relu_probe",
                "keys": probe_keys,
            },
        },
        "layer_tolerances": {key: None for key in layer_keys},
        "task_logit_tolerances": {profile: None for profile in task_profiles},
        "calibration": {
            "status": "pending",
            "basis": "Each layer limit is frozen only after observing the "
                     "deterministic subset on the mock and all RK3588 core "
                     "masks; limits are not selected from top-1 accuracy.",
            "observations": [],
        },
    }
    Path(args.spec).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(golden_path)
    print(args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
