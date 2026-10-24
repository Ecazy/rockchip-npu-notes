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
