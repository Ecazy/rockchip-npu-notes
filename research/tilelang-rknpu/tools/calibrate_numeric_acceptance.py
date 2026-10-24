#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze per-layer and per-profile logit tolerances from observed traces.

The rule is mechanical and classification-independent:

    atol = ceil_to_2^-12(2 * max_observed_abs_error), rtol = 0

Observed values are the deterministic subset on the mock plus every RK3588 core
mask, and every profile's raw task logits against its declared reference.  The
tool records the individual observations so the margin has an auditable basis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import gzip
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRID = 4096.0
TASK_PROFILES = (
    ("mnist_mlp", 32, "mlp"),
    ("mnist_mlp", 8, "mlp"),
    ("mnist_mlp", 1, "mlp"),
    ("tiny_vit", 8, "vit"),
    ("tiny_vit", 1, "vit"),
)


def label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_limit(observed: float) -> tuple[float, float]:
    if observed <= 0.0:
        return 0.0, 0.0
    return math.ceil((2.0 * observed) * GRID - 1e-12) / GRID, observed


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} != {b.shape}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(ROOT / "models" /
                                           "numeric_acceptance.json"))
    ap.add_argument("--golden", default=str(ROOT / "models" /
                                             "numeric_golden.npz"))
    ap.add_argument("--layers", action="append", default=[],
                    help="observed model_eval.layers.npz; repeat per source")
    ap.add_argument("--results", action="append", default=[],
                    help="observed model_eval.json; repeat per core")
    ap.add_argument("--data-dir", default=str(ROOT / ".data" / "mnist"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    golden_path = Path(args.golden)
    golden = np.load(golden_path, allow_pickle=False)
    if sha256(golden_path) != spec["golden_sha256"]:
        raise SystemExit("golden hash differs from spec")
    expected_indices = [int(v) for v in spec["sample_policy"]["indices"]]
    required_layers = list(spec["layer_tolerances"])
    observations: dict[str, dict[str, float]] = {
        key: {} for key in required_layers}
    trace_sources = []
    for raw_path in args.layers:
        path = Path(raw_path).resolve()
        z = np.load(path, allow_pickle=False)
        got_indices = [int(v) for v in z["sample_indices"].tolist()]
        if got_indices != expected_indices:
            raise SystemExit(f"sample indices differ in {path}")
        source_label = label(path)
        trace_sources.append({"path": source_label, "sha256": sha256(path),
                              "sample_indices": got_indices})
        for key in required_layers:
            if key not in z.files or key not in golden.files:
                raise SystemExit(f"missing layer {key!r} in golden or {path}")
            value = _max_abs(golden[key], z[key])
            observations[key][source_label] = value
    for key, source in observations.items():
        if not source:
            raise SystemExit(f"no observed trace supplies layer {key!r}")

    # Task logits: recompute the declared reference here exactly as the final
    # independent verifier does.  This captures cross-architecture FP32/BLAS
    # reduction differences without trusting an evaluator sidecar.
    from tools.verify_model_results import _refs_with_schedule
    from rknpu_backend.models import load_weight_npz
    x = read_idx(Path(args.data_dir) / "t10k-images-idx3-ubyte.gz", 3)
    weights = {
        "mnist_mlp": load_weight_npz(Path(args.weights_dir) / "mnist_mlp.npz"),
        "tiny_vit": load_weight_npz(Path(args.weights_dir) / "tiny_vit.npz"),
    }
    task_obs: dict[str, dict[str, float]] = {
        f"{model}.b{batch}": {} for model, batch, _short in TASK_PROFILES}
    result_sources = []
    for raw_path in args.results:
        result_path = Path(raw_path).resolve()
        report = json.loads(result_path.read_text())
        source_label = label(result_path)
        result_sources.append({"path": source_label,
                               "sha256": sha256(result_path)})
        for row in report.get("models", []):
            key = f"{row['model']}.b{row['batch']}"
            if key not in task_obs:
                continue
            short = "mlp" if row["model"] == "mnist_mlp" else "vit"
            logits_path = result_path.with_name(
                result_path.stem + f".{short}_b{row['batch']}.logits.npy")
            if not logits_path.is_file():
                raise SystemExit(f"missing raw logits for {key}")
            logits = np.load(logits_path, allow_pickle=False)
            n = int(row["n"])
            if logits.shape != (n, 10) or n > len(x):
                raise SystemExit(f"raw logits n mismatch for {key}")
            declared = _refs_with_schedule(
                "declared", row["model"], weights[row["model"]],
                x[:n], int(row["batch"]))
            value = _max_abs(declared, logits)
            task_obs[key][source_label] = value
    for key, source in task_obs.items():
        if not source:
            raise SystemExit(f"no observed result supplies task profile {key!r}")

    layer_tolerances = {}
    for key, source in observations.items():
        observed = max(source.values())
        atol, raw = freeze_limit(observed)
        layer_tolerances[key] = {
            "atol": atol, "rtol": 0.0,
            "observed_max_abs_by_source": dict(sorted(source.items())),
            "basis": "2x max observed |hardware-declared| over mock and all "
                     "core masks, rounded up to 2^-12; rtol=0; no top-1 data used",
        }
    task_tolerances = {}
    for key, source in task_obs.items():
        observed = max(source.values())
        atol, raw = freeze_limit(observed)
        task_tolerances[key] = {
            "atol": atol, "rtol": 0.0,
            "observed_max_abs_by_source": dict(sorted(source.items())),
            "basis": "2x max observed raw-logit error for this exact static "
                     "batch profile across all core masks, rounded up to 2^-12; "
                     "rtol=0; classification output is not a tolerance input",
        }
    spec["status"] = "frozen"
    spec["layer_tolerances"] = layer_tolerances
    spec["task_logit_tolerances"] = task_tolerances
    spec["calibration"] = {
        "status": "frozen",
        "rule": "atol=ceil_2^-12(2*max_observed_abs_error); rtol=0",
        "uses_classification_metric": False,
        "observed_layer_traces": trace_sources,
        "observed_results": result_sources,
        "basis": "Limits are frozen from deterministic layer outputs and raw "
                 "task-logit comparisons; the same limits are used by the "
                 "independent verifier.",
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
