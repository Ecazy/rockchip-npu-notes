# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic numeric-acceptance inputs shared by builder, runner, verifier.

The sample set is a fixed subset of the 512-image admission split, so every core
mask can execute the exact same layer trace.  It is selected once from the test
split by a frozen seed; no labels participate in selection.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NUMERIC_SCHEMA_VERSION = 1
SAMPLE_SEED = 20261024
SAMPLE_INDICES = (10, 73, 171, 249, 279, 349, 383, 405)
PROBE_SEED = 20261024
PROBE_SHAPE = (2, 256)


def sample_images(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw)
    if x.shape[0] <= max(SAMPLE_INDICES):
        raise ValueError("test split does not contain the frozen numeric sample")
    return x[list(SAMPLE_INDICES)]


def relu_probe_vectors(weights: dict[str, np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic probe X/B, pre-activation and exact ReLU output."""
    rng = np.random.default_rng(PROBE_SEED)
    x = rng.normal(0.0, 0.5, PROBE_SHAPE).astype(np.float16)
    bias = np.asarray(weights["b1"], dtype=np.float16)
    if bias.shape != (PROBE_SHAPE[1],):
        raise ValueError("b1 shape does not match the ReLU probe width")
    b = np.tile(bias[None, :], (PROBE_SHAPE[0], 1))
    pre = (x.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    out = np.maximum(pre, np.float16(0))
    return x, b, pre, out


def relu_probe_constants(plan: Any, weights: dict[str, np.ndarray]
                         ) -> dict[str, np.ndarray]:
    """Materialise every constant/input slot of the frozen ReLU probe plan."""
    x, b, _, _ = relu_probe_vectors(weights)
    named = {"X": x, "B": b}
    out: dict[str, np.ndarray] = {}
    for spec in plan.constants:
        if spec.name in named:
            arr = np.asarray(named[spec.name], dtype=np.float16)
        elif spec.generator == "identity":
            arr = np.eye(spec.shape[0], dtype=np.float16)
        elif spec.generator == "zero":
            arr = np.zeros(spec.shape, dtype=np.float16)
        else:
            raise KeyError(f"probe plan constant {spec.name!r} has no source")
        if arr.shape != tuple(spec.shape):
            raise ValueError(f"probe constant {spec.name!r} shape mismatch")
        out[spec.name] = np.ascontiguousarray(arr, dtype=np.float16)
    return out


def golden_arrays(mlp_weights: dict[str, np.ndarray],
                  vit_weights: dict[str, np.ndarray],
                  raw: np.ndarray) -> dict[str, np.ndarray]:
    """Build the frozen expected layer values from independent declared paths."""
    from .model_reference import (mlp_layer_trace_declared,
                                  tiny_vit_layer_trace_declared)
    x = sample_images(raw)
    mlp = mlp_layer_trace_declared(mlp_weights, x)
    vit = tiny_vit_layer_trace_declared(vit_weights, x)
    px, pb, pre, pout = relu_probe_vectors(mlp_weights)
    out: dict[str, np.ndarray] = {
        "schema_version": np.asarray(NUMERIC_SCHEMA_VERSION, dtype=np.int64),
        "sample_indices": np.asarray(SAMPLE_INDICES, dtype=np.int64),
        "mlp.x_padded": np.asarray(mlp["x_padded"], dtype=np.float16),
        "mlp.logits": np.asarray(mlp["logits"], dtype=np.float32),
        "relu_probe.input": px,
        "relu_probe.bias": pb,
        "relu_probe.pre_relu": pre,
        "relu_probe.output": pout,
    }
    for key, value in vit.items():
        out[f"tiny_vit.{key}"] = np.asarray(value)
    return out


# ---------------------------------------------------------------------------
# Train-calibrated / test-accepted numeric evidence (schema v2).
# ---------------------------------------------------------------------------
TRAIN_NUMERIC_SCHEMA_VERSION = 2
CALIBRATION_SPLIT = "train"
CALIBRATION_SEED = 20261024
CALIBRATION_COUNT = 32
CALIBRATION_INDICES = (
    1244, 2363, 2999, 4556, 4891, 8580, 14022, 14891,
    18780, 20214, 23747, 23823, 29431, 29453, 29495, 32891,
    32984, 33660, 34481, 38748, 40912, 42965, 43746, 44188,
    45480, 46056, 48201, 51374, 53224, 53271, 54801, 59181,
)
ACCEPTANCE_SPLIT = "test"
ACCEPTANCE_SEED = 20261025
ACCEPTANCE_COUNT = 32
ACCEPTANCE_INDICES = (
    354, 531, 622, 816, 1146, 1420, 1597, 1684,
    1772, 2288, 2718, 3049, 3116, 3200, 4242, 5447,
    5464, 5832, 6214, 6283, 6324, 6365, 6398, 6678,
    7415, 7903, 8733, 9118, 9308, 9346, 9440, 9830,
)
PROFILE_SPECS = (
    ("mnist_mlp", 1), ("mnist_mlp", 8), ("mnist_mlp", 32),
    ("tiny_vit", 1), ("tiny_vit", 8),
)
MLP_HIDDEN_TRACE_KEYS = (
    "x_padded", "h1_pre_relu", "r1_relu", "h2_pre_relu",
    "r2_relu", "final_preactivation", "logits",
)


def selection_metadata(split: str) -> dict[str, Any]:
    if split == CALIBRATION_SPLIT:
        return {
            "split": split, "seed": CALIBRATION_SEED,
            "count": CALIBRATION_COUNT,
            "indices": list(CALIBRATION_INDICES),
            "source_file": "train-images-idx3-ubyte.gz",
            "population": "MNIST training images",
            "method": "numpy_default_rng_choice_without_replacement",
            "labels_read_for_selection": False,
        }
    if split == ACCEPTANCE_SPLIT:
        return {
            "split": split, "seed": ACCEPTANCE_SEED,
            "count": ACCEPTANCE_COUNT,
            "indices": list(ACCEPTANCE_INDICES),
            "source_file": "t10k-images-idx3-ubyte.gz",
            "population": "MNIST test images",
            "method": "numpy_default_rng_choice_without_replacement",
            "labels_read_for_selection": False,
        }
    raise ValueError(f"unknown numeric split {split!r}")


def split_indices(split: str) -> tuple[int, ...]:
    return tuple(selection_metadata(split)["indices"])


def selected_images(raw: np.ndarray, split: str) -> np.ndarray:
    x = np.asarray(raw)
    indices = split_indices(split)
    if x.shape[0] <= max(indices):
        raise ValueError(f"{split} split does not contain selected numeric samples")
    return x[list(indices)]


def image_sha256(image: np.ndarray) -> str:
    import hashlib
    a = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()


def sample_image_hashes(images: np.ndarray) -> list[str]:
    return [image_sha256(image) for image in np.asarray(images)]


def _concat_traces(fn, weights: dict[str, np.ndarray],
                   images: np.ndarray, chunk: int = 8) -> dict[str, np.ndarray]:
    chunks = [fn(weights, images[i:i + chunk])
              for i in range(0, len(images), chunk)]
    if not chunks:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        out[key] = np.concatenate([part[key] for part in chunks], axis=0)
    return out


def profile_expected_logits(weights: dict[str, np.ndarray], raw: np.ndarray,
                            model: str, batch: int) -> np.ndarray:
    from .model_reference import (mlp_logits_declared,
                                  tiny_vit_logits_declared)
    fn = (mlp_logits_declared if model == "mnist_mlp"
          else tiny_vit_logits_declared)
    out = np.empty((len(raw), 10), dtype=np.float32)
    for i in range(0, len(raw), batch):
        part = raw[i:i + batch]
        out[i:i + len(part)] = fn(weights, part)
    return out


def expected_arrays(mlp_weights: dict[str, np.ndarray],
                    vit_weights: dict[str, np.ndarray],
                    raw: np.ndarray, *, split: str) -> dict[str, np.ndarray]:
    """Independent declared reference for a fixed train/test numeric subset."""
    from .model_reference import (mlp_layer_trace_declared,
                                  tiny_vit_layer_trace_declared)
    images = selected_images(raw, split)
    mlp = _concat_traces(mlp_layer_trace_declared, mlp_weights, images, 8)
    vit = _concat_traces(tiny_vit_layer_trace_declared, vit_weights, images, 8)
    px, pb, pre, pout = relu_probe_vectors(mlp_weights)
    out: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRAIN_NUMERIC_SCHEMA_VERSION,
                                     dtype=np.int64),
        "sample_indices": np.asarray(split_indices(split), dtype=np.int64),
        "images": np.ascontiguousarray(images, dtype=np.uint8),
        "sample_sha256": np.asarray(sample_image_hashes(images)),
        "mlp.x_padded": np.asarray(mlp["x_padded"], dtype=np.float16),
        "mlp.h1_pre_relu": np.asarray(mlp["h1"], dtype=np.float16),
        "mlp.r1_relu": np.asarray(mlp["relu1"], dtype=np.float16),
        "mlp.h2_pre_relu": np.asarray(mlp["h2"], dtype=np.float16),
        "mlp.r2_relu": np.asarray(mlp["relu2"], dtype=np.float16),
        "mlp.final_preactivation": np.asarray(mlp["logits_padded"],
                                               dtype=np.float16),
        "mlp.logits": np.asarray(mlp["logits"], dtype=np.float32),
        "relu_probe.input": px,
        "relu_probe.bias": pb,
        "relu_probe.pre_relu": pre,
        "relu_probe.output": pout,
    }
    for key, value in vit.items():
        out[f"tiny_vit.{key}"] = np.asarray(value)
    for model, batch in PROFILE_SPECS:
        weights = mlp_weights if model == "mnist_mlp" else vit_weights
        out[f"profile.{model}.b{batch}.logits"] = profile_expected_logits(
            weights, images, model, batch)
    return out


def layer_contract_keys(expected: dict[str, np.ndarray]) -> list[str]:
    keys = [key for key in expected if key.startswith(("mlp.", "tiny_vit."))]
    keys.append("relu_probe.output")
    return sorted(keys)


def profile_contract_keys() -> list[str]:
    return [f"{model}.b{batch}" for model, batch in PROFILE_SPECS]


# ---------------------------------------------------------------------------
# Real-TileLang MLP hidden-layer diagnostic plan constants.
# ---------------------------------------------------------------------------
def mlp_diag_constants(plan: Any, weights: dict[str, np.ndarray],
                       *, batch: int = 8) -> dict[str, np.ndarray]:
    w1 = np.zeros((1024, 256), dtype=np.float16)
    w1[:784] = np.asarray(weights["W1"], dtype=np.float16)
    b1 = np.tile(np.asarray(weights["b1"], dtype=np.float16)[None, :],
                 (batch, 1))
    w2 = np.asarray(weights["W2"], dtype=np.float16)
    b2 = np.tile(np.asarray(weights["b2"], dtype=np.float16)[None, :],
                 (batch, 1))
    w3 = np.zeros((128, 16), dtype=np.float16)
    w3[:, :10] = np.asarray(weights["W3"], dtype=np.float16)
    b3 = np.zeros((batch, 16), dtype=np.float16)
    b3[:, :10] = np.asarray(weights["b3"], dtype=np.float16)
    named = {"W1": w1, "B1": b1, "W2": w2, "B2": b2, "W3": w3, "B3": b3}
    out: dict[str, np.ndarray] = {}
    for spec in plan.constants:
        if spec.name in named:
            arr = np.asarray(named[spec.name], dtype=np.float16)
            if arr.shape != tuple(spec.shape):
                raise ValueError(f"diagnostic constant {spec.name} shape mismatch")
        elif spec.generator == "identity":
            arr = np.eye(spec.shape[0], dtype=np.float16)
        elif spec.generator == "zero":
            arr = np.zeros(spec.shape, dtype=np.float16)
        else:
            raise KeyError(f"no diagnostic constant source for {spec.name}")
        out[spec.name] = np.ascontiguousarray(arr, dtype=np.float16)
    return out


def save_npz_deterministic(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed ZIP timestamps for stable evidence hashes."""
    import io
    import zipfile
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy",
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload.getvalue())
