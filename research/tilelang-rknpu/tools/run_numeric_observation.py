#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the fixed train-calibration or test-acceptance numeric observation set.

This tool never opens the label files.  Calibration observations run before the
threshold spec is frozen; acceptance observations require an already-frozen spec
and record its SHA-256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rknpu_backend.numeric import (ACCEPTANCE_SPLIT, CALIBRATION_SPLIT,
                                   TRAIN_NUMERIC_SCHEMA_VERSION,
                                   sample_image_hashes, save_npz_deterministic,
                                   selected_images, split_indices)
from rknpu_backend.numeric_observation import observe_split
from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                      default_library)

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
    ap.add_argument("--mode", choices=("calibration", "acceptance"),
                    required=True)
    ap.add_argument("--plans-dir", default=str(ROOT / "models" / "compiled"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--spec", default=str(ROOT / "models" /
                                           "numeric_acceptance_train.json"))
    ap.add_argument("--lib", default=os.environ.get("RKNPU_PLAN_LIB", ""))
    ap.add_argument("--fd", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-key", action="append", default=[],
                    metavar="OLD=NEW",
                    help="override an rkhybrid manifest stem for this observation")
    args = ap.parse_args()
    plans_dir = Path(args.plans_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    mode = args.mode
    split = CALIBRATION_SPLIT if mode == "calibration" else ACCEPTANCE_SPLIT
    manifest_keys: dict[str, str] = {}
    for item in args.manifest_key:
        if "=" not in item or not all(item.split("=", 1)):
            raise SystemExit(f"invalid --manifest-key {item!r}")
        old, new = item.split("=", 1)
        manifest_keys[old] = new
    spec = None
    spec_hash = None
    if mode == "calibration":
        source_name = "train-images-idx3-ubyte.gz"
        indices = list(split_indices(split))
    else:
        spec_path = Path(args.spec).resolve()
        spec = json.loads(spec_path.read_text())
        if spec.get("status") != "frozen" or \
                spec.get("calibration_rule", {}).get("status") != "frozen":
            raise SystemExit("acceptance observations require a frozen spec")
        if spec["acceptance"]["split"] != ACCEPTANCE_SPLIT:
            raise SystemExit("frozen spec acceptance split is not test")
        source_name = spec["acceptance"]["source_file"]
        indices = [int(v) for v in spec["acceptance"]["indices"]]
        if indices != list(split_indices(split)):
            raise SystemExit("frozen spec acceptance indices differ from code")
        spec_hash = sha256(spec_path)
    source_path = data_dir / source_name
    raw = read_idx(source_path, 3)
    images = selected_images(raw, split)
    if [int(v) for v in indices] != list(split_indices(split)):
        raise SystemExit("selected numeric indices mismatch")
    if args.lib:
        lib = LibRKPlan(args.lib)
    else:
        lib = default_library()
    configure_rocket_abi(lib.cdll)
    fd = args.fd
    own_fd = False
    if fd is None:
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise RuntimeError(f"rocket_open failed: {fd}")
        own_fd = True
    try:
        arrays = observe_split(lib, plans_dir, weights_dir, raw, split, fd=fd,
                               manifest_keys=manifest_keys)
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_npz_deterministic(out, arrays)
        manifest = json.loads((plans_dir / "manifest.json").read_text())
        used_plans = {
            key: {"path": info["path"], "plan_hash": info["plan_hash"]}
            for key, info in manifest["plans"].items()
        }
        metadata = {
            "schema_version": TRAIN_NUMERIC_SCHEMA_VERSION,
            "kind": "model-numeric-observation",
            "mode": mode,
            "split": split,
            "source_file": source_name,
            "source_sha256": sha256(source_path),
            "sample_indices": indices,
            "sample_sha256": sample_image_hashes(images),
            "labels_read": False,
            "compiled_manifest_sha256": sha256(plans_dir / "manifest.json"),
            "profiles": [f"{m}.b{b}" for m, b in (
                ("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
                ("tiny_vit", 8), ("tiny_vit", 1))],
            "plans": used_plans,
            "weights_sha256": {
                "mnist_mlp.npz": sha256(weights_dir / "mnist_mlp.npz"),
                "tiny_vit.npz": sha256(weights_dir / "tiny_vit.npz"),
            },
            "spec_sha256": spec_hash,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observation_file": out.name,
            "observation_sha256": sha256(out),
        }
        meta_path = out.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(meta_path)
    finally:
        if own_fd:
            lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
