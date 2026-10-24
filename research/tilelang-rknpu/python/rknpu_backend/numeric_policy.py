# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen numeric tolerance policy shared by calibration and verification."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

GRID = 4096.0
HOST_EXACT_KEYS = {
    "mlp.x_padded",
    "tiny_vit.patches",
    "tiny_vit.patch_flat",
}


def _ceil_grid(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return math.ceil(value * GRID - 1e-12) / GRID


def f16_spacing_floor(expected: np.ndarray) -> float:
    """Two FP16 ULPs at the train-reference absolute maximum."""
    a = np.asarray(expected)
    if a.size == 0:
        return 0.0
    peak = float(np.max(np.abs(a.astype(np.float32))))
    if peak <= 0.0:
        return 0.0
    spacing = float(np.spacing(np.dtype(np.float16).type(peak)))
    return float(2.0 * spacing)


def freeze_atol(observed_max_abs: float, expected: np.ndarray,
                key: str) -> tuple[float, dict[str, Any]]:
    """Return a train-only tolerance and its auditable basis.

    Host-only exact transforms use only 2x the train observation.  Device-derived
    f16 outputs additionally get a two-ULP representational floor computed from
    the train reference's absolute maximum.  This is an empirical envelope plus a
    dtype-local floor, not a theoretical error bound.
    """
    observed = float(observed_max_abs)
    exact_host = key in HOST_EXACT_KEYS
    floor = 0.0 if exact_host else f16_spacing_floor(expected)
    raw = max(2.0 * observed, floor)
    atol = _ceil_grid(raw)
    basis = {
        "observed_max_abs": observed,
        "f16_rounding_floor": floor,
        "expected_absmax_train": float(np.max(np.abs(
            np.asarray(expected).astype(np.float32)))) if np.size(expected) else 0.0,
        "exact_host": exact_host,
        "formula": ("atol=ceil_2^-12(2*observed)"
                    if exact_host else
                    "atol=ceil_2^-12(max(2*observed,2*f16_spacing("
                    "train_expected_absmax)))"),
    }
    return atol, basis


def within_tolerance(observed: np.ndarray, expected: np.ndarray,
                     tolerance: dict[str, Any]) -> tuple[bool, float]:
    """Absolute frozen-tolerance gate; classification is not consulted."""
    a, b = np.asarray(observed), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False, float("inf")
    atol = float(tolerance.get("atol", -1.0))
    rtol = float(tolerance.get("rtol", -1.0))
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        return False, float("inf")
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    allowed = atol + rtol * np.abs(b.astype(np.float64))
    return bool(np.all(error <= allowed + 1e-15)), float(error.max(initial=0.0))
