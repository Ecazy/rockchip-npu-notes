# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute fixed train/test numeric observation subsets through production paths.

This module only gathers observations.  It never computes or changes tolerances,
and it does not read labels.  Threshold freezing is a separate Host step over a
train-split observation file.
"""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid import HybridRunner, load_hybrid_program
from .models import PlanCall, preprocess_mnist
from .numeric import (mlp_diag_constants, relu_probe_constants,
                      sample_image_hashes, selected_images, split_indices)
from .runtime.ffi import LibRKPlan
from .serialize import load_plan

DIAG_PLAN_KEYS = (
    "mnist_mlp_diag_l1_bias",
    "mnist_mlp_diag_l1_relu",
    "mnist_mlp_diag_l2_bias",
    "mnist_mlp_diag_l2_relu",
    "mnist_mlp_diag_l3_bias",
)
PROFILE_SPECS = (
    ("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
    ("tiny_vit", 8), ("tiny_vit", 1),
)


def _hybrid_path(manifest: dict[str, Any], plans_dir: Path, key: str) -> Path:
    entry = manifest.get("hybrid_manifests", {}).get(key)
    if entry is None:
        raise RuntimeError(f"missing rkhybrid manifest {key!r}")
    return plans_dir / entry["path"]


def _plan(manifest: dict[str, Any], plans_dir: Path, key: str):
    info = manifest["plans"].get(key)
    if info is None:
        raise RuntimeError(f"missing diagnostic RKPlan {key!r}")
    plan = load_plan(str(plans_dir / info["path"]))
    if plan.plan_hash != info["plan_hash"]:
        raise RuntimeError(f"diagnostic plan hash mismatch for {key!r}")
    return plan


def _run_profile(lib: LibRKPlan, manifest: dict[str, Any], plans_dir: Path,
                 model: str, batch: int, weights: dict[str, np.ndarray],
                 weights_path: Path | None, images: np.ndarray,
                 fd: int) -> np.ndarray:
    key = f"{model}_b{batch}"
    program = load_hybrid_program(_hybrid_path(manifest, plans_dir, key),
                                  plans_dir=plans_dir)
    out = np.empty((len(images), 10), dtype=np.float32)
    with HybridRunner(lib, program, weights, weights_path=weights_path,
                      fd=fd) as runner:
        for i in range(0, len(images), batch):
            part = images[i:i + batch]
            if len(part) != batch:
                raise RuntimeError(f"{key} numeric subset is not a full batch")
            out[i:i + batch] = runner.run(part).outputs["logits"]
    return out


def _run_mlp_trace(lib: LibRKPlan, manifest: dict[str, Any], plans_dir: Path,
                   weights: dict[str, np.ndarray], weights_path: Path,
                   images: np.ndarray, fd: int) -> dict[str, np.ndarray]:
    program = load_hybrid_program(
        _hybrid_path(manifest, plans_dir, "mnist_mlp_b8"),
        plans_dir=plans_dir)
    chunks: list[dict[str, np.ndarray]] = []
    with HybridRunner(lib, program, weights, weights_path=weights_path,
                      fd=fd) as runner:
        for i in range(0, len(images), 8):
            result = runner.run(images[i:i + 8], capture=True)
            chunks.append(dict(result.trace))
    out: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        out[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return out


def _run_mlp_diagnostics(lib: LibRKPlan, manifest: dict[str, Any],
                         plans_dir: Path, weights: dict[str, np.ndarray],
                         images: np.ndarray, fd: int) -> dict[str, np.ndarray]:
    plans = {key: _plan(manifest, plans_dir, key) for key in DIAG_PLAN_KEYS}
    result_lists = {key: [] for key in (
        "h1_pre_relu", "r1_relu", "h2_pre_relu", "r2_relu",
        "final_preactivation")}
    with ExitStack() as stack:
        calls = {
            key: stack.enter_context(PlanCall(
                lib, plan, mlp_diag_constants(plan, weights, batch=8), fd=fd))
            for key, plan in plans.items()
        }
        for i in range(0, len(images), 8):
            part = images[i:i + 8]
            if len(part) != 8:
                raise RuntimeError("MLP diagnostic subset is not a full b8 batch")
            x = preprocess_mnist(part).astype(np.float16)
            xp = np.zeros((8, 1024), dtype=np.float16)
            xp[:, :784] = x
            h1, _r1, _t1 = calls["mnist_mlp_diag_l1_bias"].run([xp])
            r1, _r1r, _t1r = calls["mnist_mlp_diag_l1_relu"].run([xp])
            h2, _r2, _t2 = calls["mnist_mlp_diag_l2_bias"].run([r1[0]])
            r2, _r2r, _t2r = calls["mnist_mlp_diag_l2_relu"].run([r1[0]])
            final, _r3, _t3 = calls["mnist_mlp_diag_l3_bias"].run([r2[0]])
            result_lists["h1_pre_relu"].append(h1[0])
            result_lists["r1_relu"].append(r1[0])
            result_lists["h2_pre_relu"].append(h2[0])
            result_lists["r2_relu"].append(r2[0])
            result_lists["final_preactivation"].append(final[0])
    return {key: np.concatenate(values, axis=0)
            for key, values in result_lists.items()}


def _run_relu_probe(lib: LibRKPlan, manifest: dict[str, Any], plans_dir: Path,
                    weights: dict[str, np.ndarray], fd: int) -> np.ndarray:
    plan = _plan(manifest, plans_dir, "mnist_relu_probe")
    with PlanCall(lib, plan, relu_probe_constants(plan, weights), fd=fd) as call:
        outputs, _report, _elapsed = call.run([])
    return np.asarray(outputs[0], dtype=np.float16)


def _run_vit_trace(lib: LibRKPlan, manifest: dict[str, Any], plans_dir: Path,
                   weights: dict[str, np.ndarray], weights_path: Path,
                   images: np.ndarray, fd: int) -> dict[str, np.ndarray]:
    program = load_hybrid_program(
        _hybrid_path(manifest, plans_dir, "tiny_vit_b8"),
        plans_dir=plans_dir)
    chunks: list[dict[str, np.ndarray]] = []
    with HybridRunner(lib, program, weights, weights_path=weights_path,
                      fd=fd) as runner:
        for i in range(0, len(images), 8):
            chunks.append(dict(runner.run(images[i:i + 8],
                                          capture=True).trace))
    out: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        out[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return out


def observe_split(lib: LibRKPlan, plans_dir: str | Path,
                  weights_dir: str | Path, raw_images: np.ndarray,
                  split: str, *, fd: int = 0) -> dict[str, np.ndarray]:
    """Return all fixed observations for one split in one NPZ-like mapping."""
    plans_dir = Path(plans_dir).resolve()
    weights_dir = Path(weights_dir).resolve()
    manifest = json.loads((plans_dir / "manifest.json").read_text())
    mlp_path = weights_dir / "mnist_mlp.npz"
    vit_path = weights_dir / "tiny_vit.npz"
    mlp = dict(np.load(mlp_path, allow_pickle=False))
    vit = dict(np.load(vit_path, allow_pickle=False))
    images = selected_images(raw_images, split)
    arrays: dict[str, np.ndarray] = {
        "sample_indices": np.asarray(split_indices(split), dtype=np.int64),
        "images": np.ascontiguousarray(images, dtype=np.uint8),
        "sample_sha256": np.asarray(sample_image_hashes(images)),
    }
    for model, batch, weights, weights_path in (
            ("mnist_mlp", 32, mlp, mlp_path),
            ("mnist_mlp", 8, mlp, mlp_path),
            ("mnist_mlp", 1, mlp, mlp_path),
            ("tiny_vit", 8, vit, vit_path),
            ("tiny_vit", 1, vit, vit_path)):
        arrays[f"profile.{model}.b{batch}.logits"] = _run_profile(
            lib, manifest, plans_dir, model, batch, weights, weights_path,
            images, fd)
    mlp_trace = _run_mlp_trace(lib, manifest, plans_dir, mlp, mlp_path,
                               images, fd)
    arrays["mlp.x_padded"] = mlp_trace["x_padded"]
    arrays["mlp.logits"] = mlp_trace["logits"]
    hidden = _run_mlp_diagnostics(lib, manifest, plans_dir, mlp, images, fd)
    arrays.update({f"mlp.{key}": value for key, value in hidden.items()})
    arrays["relu_probe.output"] = _run_relu_probe(lib, manifest, plans_dir,
                                                  mlp, fd)
    vit_trace = _run_vit_trace(lib, manifest, plans_dir, vit, vit_path,
                               images, fd)
    arrays.update({f"tiny_vit.{key}": value for key, value in vit_trace.items()})
    return arrays
