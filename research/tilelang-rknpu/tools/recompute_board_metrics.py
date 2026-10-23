#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline recompute declared/FP64-math metrics from archived raw outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from make_board_campaign import (activation_declared, activation_math,
                                 broadcast_declared, broadcast_math,
                                 ffn_math_reference, ffn_nk_declared,
                                 ffn_nk_math, ffn_reference,
                                 kchain_nk_declared, matmul_activation_declared,
                                 matmul_activation_math, matmul_math,
                                 matmul_math_reference, matmul_reference,
                                 projection_math_reference,
                                 projection_nk_declared, projection_nk_math,
                                 projection_reference, ragged_reference,
                                 staged_ffn_math, staged_ffn_reference,
                                 staged_projection_math,
                                 staged_projection_reference)


def metrics(got, ref, atol, rtol):
    got64 = np.asarray(got, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    if got64.shape != ref64.shape:
        raise ValueError(f"shape {got64.shape} != {ref64.shape}")
    diff = np.abs(got64 - ref64)
    denom = np.maximum(np.abs(ref64), 1e-12)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float((diff / denom).max()) if diff.size else 0.0,
        "mismatched_elements": int(np.count_nonzero(got64 != ref64)),
        "element_count": int(got64.size),
        "allclose": bool(np.allclose(got64, ref64, atol=atol, rtol=rtol,
                                     equal_nan=False)),
    }


def reference_arrays(case, data, round_index):
    kind = case["kind"]

    def inp(name):
        return np.asarray(data[f"input.{name}.{round_index}"], dtype=np.float16)

    def const(name):
        return np.asarray(data[f"constant.{name}"], dtype=np.float16)

    inputs = set(case.get("input_names", []))
    constants = set(case.get("constant_names", []))
    if kind == "matmul":
        x, w = inp("A"), const("B")
        return ({"Y": matmul_reference(x, w)},
                {"Y": matmul_math_reference(x, w)})
    if kind == "projection":
        x, w1, w2 = inp("X"), const("W1"), const("W2")
        if w1.shape[1] == x.shape[1]:  # RKIR [N,K]
            return ({"Y": projection_nk_declared(x, w1, w2)},
                    {"Y": projection_nk_math(x, w1, w2)})
        return ({"Y": projection_reference(x, w1, w2)},
                {"Y": projection_math_reference(x, w1, w2)})
    if kind == "ffn":
        x, wg, wu, wd = inp("X"), const("Wg"), const("Wu"), const("Wd")
        if wg.shape[1] == x.shape[1]:  # RKIR [N,K]
            return ({"Y": ffn_nk_declared(x, wg, wu, wd)},
                    {"Y": ffn_nk_math(x, wg, wu, wd)})
        return ({"Y": ffn_reference(x, wg, wu, wd)},
                {"Y": ffn_math_reference(x, wg, wu, wd)})
    if kind in {"ragged", "kchain", "mntile"}:
        x, w = inp("A"), const("B")
        k = x.shape[1]
        if kind == "ragged":
            return ({"C": ragged_reference(x, w, k)},
                    {"C": x.astype(np.float64) @ w.astype(np.float64)})
        # The campaign's kchain/mntile builders use natural [K,N] weights and
        # the generic K-chain reference.
        from make_board_campaign import kchain_declared
        return ({"Y": kchain_declared(x, w, k)},
                {"Y": matmul_math(x, w)})
    if kind == "broadcast":
        x, w, bias = inp("X"), const("W"), const("B")
        templates = case.get("templates", [])
        op = "add" if any("ew_add" in t for t in templates) else "mul"
        return ({"Y": broadcast_declared(x, w, bias, op)},
                {"Y": broadcast_math(x, w, bias, op)})
    if kind == "activation":
        if "X" in inputs and not constants:
            x = inp("X")
            names = case["output_names"]
            name = names[0]
            return ({name: activation_declared(x)},
                    {name: activation_math(x)})
        if "A" in inputs and "B" in constants:
            x, w = inp("A"), const("B")
            name = case["output_names"][0]
            return ({name: matmul_activation_declared(x, w)},
                    {name: matmul_activation_math(x, w)})
    if kind == "multi_output":
        x, w1, w2 = inp("X"), const("W1"), const("W2")
        return ({"Y1": matmul_reference(x, w1),
                 "Y2": matmul_reference(x, w2)},
                {"Y1": matmul_math(x, w1),
                 "Y2": matmul_math(x, w2)})
    if kind in {"projection_multitile", "ffn_multitile"}:
        x = inp("X")
        if kind == "projection_multitile":
            w1, w2 = const("W1"), const("W2")
            return ({"Y": projection_nk_declared(x, w1, w2)},
                    {"Y": projection_nk_math(x, w1, w2)})
        wg, wu, wd = const("Wg"), const("Wu"), const("Wd")
        return ({"Y": ffn_nk_declared(x, wg, wu, wd)},
                {"Y": ffn_nk_math(x, wg, wu, wd)})
    if kind == "branch_multitile":
        x, w1, w2 = inp("X"), const("W1"), const("W2")
        return ({"Y1": kchain_nk_declared(x, w1),
                 "Y2": kchain_nk_declared(x, w2)},
                {"Y1": x.astype(np.float64) @ w1.T.astype(np.float64),
                 "Y2": x.astype(np.float64) @ w2.T.astype(np.float64)})
    if kind == "activation_multitile":
        x = inp("X")
        return ({"Y": activation_declared(x)},
                {"Y": activation_math(x)})
    if kind == "projection_tilelang_staged":
        x, w1, w2 = inp("X"), const("W1"), const("W2")
        return ({"Y": staged_projection_reference(x, w1, w2)},
                {"Y": staged_projection_math(x, w1, w2)})
    if kind == "ffn_tilelang_staged":
        x, wg, wu, wd = inp("X"), const("Wg"), const("Wu"), const("Wd")
        return ({"Y": staged_ffn_reference(x, wg, wu, wd)},
                {"Y": staged_ffn_math(x, wg, wu, wd)})
    if kind == "branch_tilelang_staged":
        x, w1, w2 = inp("X"), const("W1"), const("W2")
        return ({"Y1": kchain_nk_declared(x, w1) if False else
                        __import__("make_board_campaign").kchain_declared(
                            x, w1, 256),
                 "Y2": __import__("make_board_campaign").kchain_declared(
                            x, w2, 256)},
                {"Y1": x.astype(np.float64) @ w1.astype(np.float64),
                 "Y2": x.astype(np.float64) @ w2.astype(np.float64)})
    raise ValueError(f"unsupported case kind {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    results = Path(args.results).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    detail = {}
    problems = []
    for mask in (1, 2, 4):
        detail[str(mask)] = {}
        for case in manifest["cases"]:
            if case["kind"] == "guard":
                continue
            data = np.load(campaign / case["data"], allow_pickle=False)
            raw = np.load(results / f"core-{mask}" / "outputs" /
                          f"{case['name']}.npz", allow_pickle=False)
            atol = float(case["tolerance"]["atol"])
            rtol = float(case["tolerance"]["rtol"])
            declared_max = math_max = 0.0
            allclose = True
            elements = 0
            for r in range(int(case["rounds"])):
                declared, math_ref = reference_arrays(case, data, r)
                for name in case["output_names"]:
                    got = np.asarray(raw[f"output.{name}.{r}"],
                                     dtype=np.float16)
                    dmetrics = metrics(got, declared[name], atol, rtol)
                    mmetrics = metrics(got, math_ref[name], atol, rtol)
                    declared_max = max(declared_max, dmetrics["max_abs"])
                    math_max = max(math_max, mmetrics["max_abs"])
                    elements += dmetrics["element_count"]
                    allclose = allclose and dmetrics["allclose"]
            detail[str(mask)][case["name"]] = {
                "declared_max_abs": declared_max,
                "fp64_math_max_abs": math_max,
                "declared_allclose": allclose,
                "element_count": elements,
            }
            if not allclose:
                problems.append(f"core {mask}/{case['name']}: declared mismatch")
    summary = {"kind": "offline-recomputed-metrics", "overall_exit":
               0 if not problems else 1, "problems": problems,
               "detail": detail}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Offline recomputed metrics", "",
             f"overall: **{'PASS' if not problems else 'FAIL'}**", "",
             "The FP64 math reference is unquantized and retained as float64. "
             "Declared-path acceptance uses the recorded allclose policy.", "",
             "| core | case | declared max abs | FP64 math max abs | "
             "declared allclose |", "| --- | --- | --- | --- | --- |"]
    for mask in (1, 2, 4):
        for name, item in sorted(detail[str(mask)].items()):
            lines.append(
                f"| {mask} | {name} | {item['declared_max_abs']:.10g} | "
                f"{item['fp64_math_max_abs']:.10g} | "
                f"{item['declared_allclose']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(out / "summary.md")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
