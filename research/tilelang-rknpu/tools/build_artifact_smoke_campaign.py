#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the artifact smoke campaign: real TileLang kernels -> portable case data.

This is the Host half of the refactor smoke test.  For every representative kernel
it runs the *production* compile entry point on the Host:

    real TileLang PrimFunc
      -> compile_artifact(...)        (RKIR carrier, same codegen pipeline)
      -> explain()                    (must not touch a device)
      -> export_plan(...)             (the replayable RKPlan file shipped to the board)
      -> normalized graph JSON        (so the board can re-run the artifact path)

It also writes, per case:

* ``inputs/a.npy`` / ``inputs/b.npy`` -- two different input sets for the A->B->A
  update check on one workspace,
* ``constants/<name>.npy`` -- the runtime constants the plan expects,
* ``reference/a.npy`` / ``reference/b.npy`` -- the Host oracle, computed in float64
  with explicit FP16 rounding at every declared narrowing point (the frozen
  ``rk_f16_tiled`` policy: ordered 256-wide K tiles written back as FP16),
* ``meta.json`` -- bindings, the plan's declared tolerance, plan hash and the
  identity of the sources used.

Nothing here opens a device.  ``--expect-device`` is deliberately absent: the only
way this script can fail on a device issue is if it tried to use one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

K_TILE = 256
F16 = np.float16


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_smoke_{name}",
                                                  FIXTURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f16(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=F16))


def kchain(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Frozen numeric policy: ordered K tiles, FP16 written back after each tile."""
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    acc = f16(a[:, :K_TILE] @ b[:K_TILE, :])
    for k0 in range(K_TILE, a.shape[1], K_TILE):
        acc = f16(acc.astype(np.float64) + a[:, k0:k0 + K_TILE] @ b[k0:k0 + K_TILE, :])
    return acc


def silu(z: np.ndarray) -> np.ndarray:
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    return f16(z.astype(np.float32) * sig.astype(np.float32))


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _case_specs():
    semantics = _load("tilelang_semantics")
    ragged = _load("tilelang_ragged")
    multi = _load("tilelang_multi_output")
    tiled = dict(numeric_mode="rk_f16_tiled")
    tiled_lut = dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)

    # Each spec returns the kernel, its options, a *runtime-input* generator and an
    # oracle that maps (inputs, constants) to expected outputs.  Constants stay fixed
    # across the A and B input sets, so the oracle for B is recomputed with the same
    # weights -- generating B's expectation from a different weight set would look
    # like a device mismatch.
    def single_matmul():
        prim = semantics._valid_kernel(16, 256, 16)
        constants = {"B": f16(np.random.default_rng(9001).normal(0, 0.10,
                                                               (256, 16)))}

        def inputs(seed: int):
            rng = np.random.default_rng(seed)
            return {"A": f16(rng.normal(0, 0.15, (16, 256)))}

        def oracle(data):
            return {"Y": kchain(data["A"], constants["B"])}

        return prim, tiled, constants, inputs, oracle

    def matmul_silu():
        prim = semantics._silu_kernel(16, 256, 16)
        constants = {"B": f16(np.random.default_rng(9002).normal(0, 0.10,
                                                               (256, 16)))}

        def inputs(seed: int):
            rng = np.random.default_rng(seed)
            return {"A": f16(rng.normal(0, 0.15, (16, 256)))}

        def oracle(data):
            return {"Y": silu(kchain(data["A"], constants["B"]))}

        return prim, tiled_lut, constants, inputs, oracle

    def ragged_k257():
        prim = ragged.ragged_tail(13, 33, 257, 16, 64, 256)
        constants = {"B": f16(np.random.default_rng(9003).normal(0, 0.10,
                                                               (257, 33)))}

        def inputs(seed: int):
            rng = np.random.default_rng(seed)
            return {"A": f16(rng.normal(0, 0.15, (13, 257)))}

        def oracle(data):
            return {"C": kchain(data["A"], constants["B"])}

        return prim, tiled, constants, inputs, oracle

    def branch_outputs():
        prim = multi.branch_outputs(8, 64, 32, 48)
        rng = np.random.default_rng(9004)
        constants = {"W1": f16(rng.normal(0, 0.12, (64, 32))),
                     "W2": f16(rng.normal(0, 0.12, (64, 48)))}

        def inputs(seed: int):
            rng = np.random.default_rng(seed)
            return {"X": f16(rng.normal(0, 0.15, (8, 64)))}

        def oracle(data):
            return {"Y1": kchain(data["X"], constants["W1"]),
                    "Y2": kchain(data["X"], constants["W2"])}

        return prim, tiled, constants, inputs, oracle

    return [
        ("single_matmul", single_matmul),
        ("matmul_silu", matmul_silu),
        ("ragged_k257", ragged_k257),
        ("branch_outputs", branch_outputs),
    ]


def _contract_tolerance(graph) -> dict[str, float]:
    """The declared (frozen) tolerance, read from the compiled graph itself."""
    tolerances = []
    for op in graph.ops:
        contract = op.attrs.get("numeric_contract") or {}
        tolerance = contract.get("tolerance")
        if tolerance:
            tolerances.append({"atol": float(tolerance["atol"]),
                               "rtol": float(tolerance["rtol"])})
    if not tolerances:
        raise SystemExit("compiled graph declares no numeric tolerance")
    # A single policy per kernel: every op must agree, otherwise the message would
    # hide an inconsistency behind a widened bound.
    unique = {(round(t["atol"], 12), round(t["rtol"], 12)) for t in tolerances}
    if len(unique) != 1:
        raise SystemExit(f"mixed tolerances in one kernel: {sorted(unique)}")
    atol, rtol = unique.pop()
    return {"atol": atol, "rtol": rtol}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7100)
    args = parser.parse_args(argv)

    from rknpu_backend import (CompileOptions, compile_artifact, default_profile,
                               explain_plan)
    from rknpu_backend.codegen import pipeline
    from rknpu_backend.serialize import canonical_json

    if not (ROOT / "python" / "rknpu_backend").is_dir():
        raise SystemExit("run from the tilelang-rknpu tree")
    profile = default_profile()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "kind": "artifact-smoke-campaign",
        "schema_version": 1,
        "created_by": "tools/build_artifact_smoke_campaign.py",
        "profile_hash": profile.profile_hash,
        "numeric_mode": "rk_f16_tiled",
        "k_tile": K_TILE,
        "cases": [],
    }

    for index, (name, spec) in enumerate(_case_specs()):
        seed_a = args.seed + index
        seed_b = seed_a + 1000
        prim, options_kwargs, constants, make_inputs, oracle = spec()
        data_a = make_inputs(seed_a)
        data_b = make_inputs(seed_b)
        if set(data_a) != set(data_b):
            raise SystemExit(f"{name}: input names differ between input sets")
        # Same constants, different inputs: exactly the A -> B -> A update.
        reference_a = oracle(data_a)
        reference_b = oracle(data_b)
        if all(np.array_equal(data_a[k], data_b[k]) for k in data_a):
            raise SystemExit(f"{name}: the two input sets are identical")
        prim_a = prim

        options = CompileOptions(**options_kwargs)
        artifact = compile_artifact(prim_a, profile=profile, options=options,
                                    b_is_input=False)
        graph = pipeline.normalize_tilelang(prim_a, profile=profile,
                                           options=options)
        report = artifact.explain()
        # explain() must describe the same plan without touching a device.
        if report["plan_hash"] != artifact.plan_hash:
            raise SystemExit(f"{name}: explain() reported a different plan"

                             )
        if explain_plan(artifact.plan)["counts"] != report["counts"]:
            raise SystemExit(f"{name}: explain() disagrees with explain_plan()")

        case_dir = out_dir / "cases" / name
        (case_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (case_dir / "constants").mkdir(parents=True, exist_ok=True)
        (case_dir / "reference").mkdir(parents=True, exist_ok=True)

        artifact.export_plan(str(case_dir / "plan.json"))
        (case_dir / "rkir.json").write_text(
            canonical_json(graph.to_dict()) + "\n", encoding="utf-8")
        for binding, array in data_a.items():
            np.save(case_dir / "inputs" / f"a_{binding}.npy", array)
        for binding, array in data_b.items():
            np.save(case_dir / "inputs" / f"b_{binding}.npy", array)
        for binding, array in constants.items():
            np.save(case_dir / "constants" / f"{binding}.npy", array)
        for binding, array in reference_a.items():
            np.save(case_dir / "reference" / f"a_{binding}.npy", array)
        for binding, array in reference_b.items():
            np.save(case_dir / "reference" / f"b_{binding}.npy", array)

        tolerance = _contract_tolerance(graph)
        meta = {
            "name": name,
            "tolerance": tolerance,
            "options": {"numeric_mode": options.numeric_mode,
                        "allow_approx_lut": options.allow_approx_lut,
                        "submit_mode": options.submit_mode},
            "plan_hash": artifact.plan_hash,
            "profile_hash": artifact.profile_hash,
            "source_hash": artifact.source_hash,
            "bindings": [{"name": b.name, "role": b.role, "shape": list(b.shape)}
                         for b in artifact.plan.bindings],
            "inputs": sorted(data_a),
            "constants": [{"name": spec.name, "shape": list(spec.shape),
                           "generator": spec.generator}
                          for spec in artifact.plan.constants],
            "outputs": sorted(reference_a),
            "tasks": artifact.plan.accounting.task_count,
            "submits": artifact.plan.accounting.submit_count,
            "data_bos": artifact.plan.accounting.data_bo_count,
            "explain_counts": report["counts"],
            "inputs_sha256": {k: _digest(v) for k, v in
                              {**{f"a_{k}": v for k, v in data_a.items()},
                               **{f"b_{k}": v for k, v in data_b.items()}}.items()},
            "constants_sha256": {k: _digest(v) for k, v in constants.items()},
            "reference_sha256": {k: _digest(v) for k, v in
                                 {**{f"a_{k}": v for k, v in reference_a.items()},
                                  **{f"b_{k}": v for k, v in reference_b.items()}}.items()},
        }
        (case_dir / "meta.json").write_text(
            canonical_json(meta) + "\n", encoding="utf-8")
        manifest["cases"].append({"name": name, "dir": f"cases/{name}",
                                  "plan_hash": artifact.plan_hash,
                                  "tolerance": tolerance,
                                  "tasks": meta["tasks"],
                                  "inputs": meta["inputs"],
                                  "constants": [spec.name for spec in
                                                artifact.plan.constants
                                                if spec.generator is None],
                                  "outputs": meta["outputs"]})
        print(f"{name:16s} plan={artifact.plan_hash[:12]} "
              f"tasks={meta['tasks']} tolerance={tolerance}", flush=True)

    (out_dir / "manifest.json").write_text(canonical_json(manifest) + "\n",
                                          encoding="utf-8")
    print(f"campaign written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
