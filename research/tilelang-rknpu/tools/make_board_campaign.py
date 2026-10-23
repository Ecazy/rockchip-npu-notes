#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the RK3588 admission campaign plans and numpy fixtures on the Host."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f16(x):
    return np.asarray(x, dtype=np.float16)


def matmul_reference(x, w):
    # Natural [K,N] weight; one 256-wide K tile.
    return f16(np.matmul(x.astype(np.float64), w.astype(np.float64)))


def projection_reference(x, w1, w2):
    z = f16(np.matmul(x.astype(np.float64), w1.astype(np.float64)))
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    silu = f16(z.astype(np.float32) * sig.astype(np.float32))
    p = f16(np.matmul(silu.astype(np.float64), w2.astype(np.float64)))
    return f16(p.astype(np.float32) + x.astype(np.float32))


def ffn_reference(x, wg, wu, wd):
    gate = f16(np.matmul(x.astype(np.float64), wg.astype(np.float64)))
    up = f16(np.matmul(x.astype(np.float64), wu.astype(np.float64)))
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))
    act = f16(silu.astype(np.float32) * up.astype(np.float32))
    proj = f16(np.matmul(act.astype(np.float64), wd.astype(np.float64)))
    return f16(proj.astype(np.float32) + x.astype(np.float32))


def matmul_math_reference(x, w):
    """Unquantized FP64 mathematical reference (no FP16 intermediate cast)."""
    return x.astype(np.float64) @ w.astype(np.float64)


def projection_math_reference(x, w1, w2):
    """Unquantized FP64 mathematical reference (no FP16 intermediate cast)."""
    z = x.astype(np.float64) @ w1.astype(np.float64)
    gate = 1.0 / (1.0 + np.exp(-z))
    silu = z * gate
    proj = silu @ w2.astype(np.float64)
    return proj + x.astype(np.float64)


def ffn_math_reference(x, wg, wu, wd):
    """Unquantized FP64 SwiGLU mathematical reference (no intermediate cast)."""
    gate = x.astype(np.float64) @ wg.astype(np.float64)
    up = x.astype(np.float64) @ wu.astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    act = silu * up
    proj = act @ wd.astype(np.float64)
    return proj + x.astype(np.float64)


def activation_declared(x):
    sig = f16(1.0 / (1.0 + np.exp(-x.astype(np.float32))))
    return f16(x.astype(np.float32) * sig.astype(np.float32))


def activation_math(x):
    x64 = x.astype(np.float64)
    return x64 / (1.0 + np.exp(-x64))


def matmul_activation_declared(x, w):
    z = f16(x.astype(np.float64) @ w.astype(np.float64))
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    return f16(z.astype(np.float32) * sig.astype(np.float32))


def matmul_activation_math(x, w):
    z = x.astype(np.float64) @ w.astype(np.float64)
    return z / (1.0 + np.exp(-z))


def kchain_declared(x, w, k):
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    first = min(256, k)
    acc = f16(a[:, :first] @ b[:first, :])
    for k0 in range(256, k, 256):
        k1 = min(k, k0 + 256)
        acc = f16(acc.astype(np.float64) + (a[:, k0:k1] @ b[k0:k1, :]))
    return acc


def matmul_math(x, w):
    return x.astype(np.float64) @ w.astype(np.float64)


def kchain_nk_declared(x, w):
    """Declared K tile chain for RKIR [N,K] weights."""
    a = x.astype(np.float64)
    b = w.T.astype(np.float64)
    first = min(256, a.shape[1])
    acc = f16(a[:, :first] @ b[:first, :])
    for k0 in range(256, a.shape[1], 256):
        k1 = min(a.shape[1], k0 + 256)
        acc = f16(acc.astype(np.float64) + (a[:, k0:k1] @ b[k0:k1, :]))
    return acc


def projection_nk_declared(x, w1, w2):
    z = kchain_nk_declared(x, w1)
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    silu = f16(z.astype(np.float32) * sig.astype(np.float32))
    p = kchain_nk_declared(silu, w2)
    return f16(p.astype(np.float32) + x.astype(np.float32))


def projection_nk_math(x, w1, w2):
    z = x.astype(np.float64) @ w1.T.astype(np.float64)
    silu = z / (1.0 + np.exp(-z))
    p = silu @ w2.T.astype(np.float64)
    return p + x.astype(np.float64)


def ffn_nk_declared(x, wg, wu, wd):
    gate = kchain_nk_declared(x, wg)
    up = kchain_nk_declared(x, wu)
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))
    act = f16(silu.astype(np.float32) * up.astype(np.float32))
    proj = kchain_nk_declared(act, wd)
    return f16(proj.astype(np.float32) + x.astype(np.float32))


def ffn_nk_math(x, wg, wu, wd):
    x64 = x.astype(np.float64)
    gate = x64 @ wg.T.astype(np.float64)
    up = x64 @ wu.T.astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    prov = (silu * up) @ wd.T.astype(np.float64)
    return prov + x64


def staged_projection_reference(x, w1, w2):
    z = kchain_declared(x, w1, x.shape[1])
    sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
    silu = f16(z.astype(np.float32) * sig.astype(np.float32))
    p = kchain_declared(silu, w2, w2.shape[0])
    return f16(p.astype(np.float32) + x.astype(np.float32))


def staged_projection_math(x, w1, w2):
    z = x.astype(np.float64) @ w1.astype(np.float64)
    silu = z / (1.0 + np.exp(-z))
    p = silu @ w2.astype(np.float64)
    return p + x.astype(np.float64)


def staged_ffn_reference(x, wg, wu, wd):
    gate = kchain_declared(x, wg, x.shape[1])
    up = kchain_declared(x, wu, x.shape[1])
    sig = f16(1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    silu = f16(gate.astype(np.float32) * sig.astype(np.float32))
    act = f16(silu.astype(np.float32) * up.astype(np.float32))
    p = kchain_declared(act, wd, wd.shape[0])
    return f16(p.astype(np.float32) + x.astype(np.float32))


def staged_ffn_math(x, wg, wu, wd):
    x64 = x.astype(np.float64)
    gate = x64 @ wg.astype(np.float64)
    up = x64 @ wu.astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ wd.astype(np.float64) + x64


def broadcast_declared(x, w, bias, op):
    z = f16(x.astype(np.float64) @ w.astype(np.float64))
    if op == "add":
        return f16(z.astype(np.float32) + bias.astype(np.float32))
    return f16(z.astype(np.float32) * bias.astype(np.float32))


def broadcast_math(x, w, bias, op):
    z = x.astype(np.float64) @ w.astype(np.float64)
    b = bias.astype(np.float64)
    return z + b if op == "add" else z * b


def ragged_reference(x, w, k):
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    first = min(256, k)
    acc = f16(a[:, :first] @ b[:first, :])
    for k0 in range(256, k, 256):
        k1 = min(k, k0 + 256)
        acc = f16(acc.astype(np.float64) + (a[:, k0:k1] @ b[k0:k1, :]))
    return acc


def plan_summary(plan):
    tasks = [a for a in plan.actions if a.kind == "npu_task"]
    return {
        "plan_hash": plan.plan_hash,
        "profile_hash": plan.profile_hash,
        "templates": [a.template_id for a in tasks],
        "task_count": plan.accounting.task_count,
        "submit_count": len(plan.submits),
        "upload_count": sum(1 for a in plan.actions if a.kind == "upload"),
        "download_count": sum(1 for a in plan.actions if a.kind == "download"),
        "publish_count": len(plan.publish),
        "submit_mode": "chain" if plan.submits and plan.submits[0].mode == "chain"
                       else "separate",
        "publish": [{"rows": p.rows, "cols": p.cols,
                     "dst_row": p.dst_row, "dst_col": p.dst_col}
                    for p in plan.publish],
        "source_tail_contract": next(
            (op.attrs.get("source_tail_contract") for op in []), None),
    }


def add_case(campaign: Path, manifest: dict, *, name: str, prim_func,
             options, inputs: list[dict], constants: dict,
             expected: list[dict], math_expected: list[dict],
             kind: str, rounds: int, guard: str | None = None,
             bad_constants: dict | None = None, compile_fn=None,
             tolerance: dict | None = None):
    from rknpu_backend import compile_rkir, compile_tilelang, save_plan
    from rknpu_backend.frontend.tilelang_adapter import extract
    from rknpu_backend.ir import NumericContract
    if tolerance is None:
        module = extract(prim_func, options=options,
                         profile_hash=manifest["_profile"].profile_hash)
        matmul_op = next(op for op in module.ops if op.opcode == "matmul")
        contract = NumericContract.from_dict(
            matmul_op.attrs["numeric_contract"])
        tolerance = contract.tolerance
    result = (compile_fn or compile_tilelang)(
        prim_func, profile=manifest["_profile"], options=options)
    plan = result.plan
    if tolerance.get("atol") != 8e-3 or tolerance.get("rtol") != 8e-3:
        raise RuntimeError(f"unexpected plan tolerance for {name}: {tolerance}")
    plans_dir = campaign / "plans"
    data_dir = campaign / "cases"
    plans_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"{name}.rkplan.json"
    save_plan(plan, plan_path)
    data_path = data_dir / f"{name}.npz"
    payload = {}
    for r, values in enumerate(inputs):
        for key, array in values.items():
            payload[f"input.{key}.{r}"] = f16(array)
    if expected:
        for r, values in enumerate(expected):
            for key, array in values.items():
                payload[f"declared.{key}.{r}"] = f16(array)
    if math_expected:
        for r, values in enumerate(math_expected):
            for key, array in values.items():
                payload[f"fp64_math.{key}.{r}"] = np.asarray(
                    array, dtype=np.float64)
    for key, array in constants.items():
        payload[f"constant.{key}"] = f16(array)
    if bad_constants:
        for key, array in bad_constants.items():
            payload[f"bad_constant.{key}"] = f16(array)
    np.savez_compressed(data_path, **payload)
    summary = plan_summary(plan)
    summary["source_tail_contract"] = (
        result.report.get("source", {}).get("source_tail_contract"))
    manifest["cases"].append({
        "name": name,
        "kind": kind,
        "plan": str(plan_path.relative_to(campaign)),
        "data": str(data_path.relative_to(campaign)),
        "rounds": rounds,
        "guard": guard,
        "input_names": sorted(set().union(*(v.keys() for v in inputs))),
        "constant_names": sorted(constants),
        "output_names": [b.name for b in plan.bindings if b.role == "output"],
        "output_shapes": {b.name: list(b.shape) for b in plan.bindings
                          if b.role == "output"},
        "tolerance": {
            "atol": float(tolerance.get("atol", 8e-3)),
            "rtol": float(tolerance.get("rtol", 8e-3)),
            "source": "plan matmul NumericContract.tolerance; generated by " \
                      "compile_tilelang for rk_f16_tiled",
            "acceptance": "numpy.allclose: abs(got-declared) <= atol + " \
                          "rtol*abs(declared)",
            "scope": "declared staged-FP16 path only; not a mathematical-FP64 " \
                     "precision guarantee",
        },
        **summary,
    })


def main() -> int:
    from rknpu_backend import CompileOptions, default_profile

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    campaign = Path(args.out_dir).resolve()
    if campaign.exists() and any(campaign.iterdir()):
        raise SystemExit(f"campaign directory is not empty: {campaign}")
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "tools").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "tools" / "board_runner.py",
                 campaign / "tools" / "board_runner.py")
    shutil.copy2(ROOT / "tools" / "board_cli_check.py",
                 campaign / "tools" / "board_cli_check.py")
    profile = default_profile()
    manifest = {"schema": 1, "kind": "rk3588-board-admission",
                "chip": "rk3588", "profile_id": "rk3588_vendor_co_v1",
                "profile_hash": profile.profile_hash,
                "expected_driver": "RKNPU driver: v0.9.8",
                "expected_runtime_abi_version": 1,
                "tolerance_contract": {
                    "atol": 8e-3,
                    "rtol": 8e-3,
                    "formula": "numpy.allclose: abs(got-declared) <= atol + "
                               "rtol*abs(declared)",
                    "source": "rk_f16_tiled NumericContract.tolerance in each "
                              "generated plan",
                    "scope": "declared staged-FP16 path and listed small shapes; "
                             "not a mathematical-FP64 precision claim",
                },
                "cases": [], "_profile": profile}

    matmul = load_fixture("board_matmul", ROOT / "examples" / "matmul.py")
    projection = load_fixture("board_projection",
                              FIXTURES / "tilelang_silu_proj.py")
    ffn = load_fixture("board_ffn", FIXTURES / "tilelang_ffn.py")
    ragged = load_fixture("board_ragged", FIXTURES / "tilelang_ragged.py")
    broadcast = load_fixture("board_broadcast",
                             FIXTURES / "tilelang_broadcast.py")
    mntile = load_fixture("board_mntile", FIXTURES / "tilelang_mntile.py")
    kloop = load_fixture("board_kloop", FIXTURES / "tilelang_kloop.py")
    multiout = load_fixture("board_multiout",
                            FIXTURES / "tilelang_multi_output.py")
    projection_staged = load_fixture(
        "board_projection_staged", FIXTURES / "tilelang_projection_staged.py")
    ffn_staged = load_fixture("board_ffn_staged",
                              FIXTURES / "tilelang_ffn_staged.py")
    branch_staged = load_fixture("board_branch_staged",
                                 FIXTURES / "tilelang_branch_staged.py")

    rng = np.random.default_rng(20261023)
    x = f16(rng.normal(0, 0.3, (32, 64)))
    x2 = f16(rng.normal(0, 0.3, (32, 64)))
    w = f16(rng.normal(0, 0.2, (64, 48)))
    add_case(
        campaign, manifest, name="matmul_chain",
        prim_func=matmul.make_kernel(m=32, k=64, n=48),
        options=CompileOptions(numeric_mode="rk_f16_tiled", submit_mode="chain"),
        inputs=[{"A": x}, {"A": x2}], constants={"B": w},
        expected=[{"Y": matmul_reference(x, w)},
                  {"Y": matmul_reference(x2, w)}],
        math_expected=[{"Y": matmul_math_reference(x, w)},
                       {"Y": matmul_math_reference(x2, w)}],
        kind="matmul", rounds=2)

    px = f16(rng.normal(0, 0.2, (13, 64)))
    px2 = f16(rng.normal(0, 0.2, (13, 64)))
    w1 = f16(rng.normal(0, 0.15, (64, 32)))
    w2 = f16(rng.normal(0, 0.15, (32, 64)))
    for mode in ("chain", "separate"):
        add_case(
            campaign, manifest, name=f"projection_{mode}",
            prim_func=projection.silu_proj(m=13, d=64, h=32),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True, submit_mode=mode),
            inputs=[{"X": px}, {"X": px2}],
            constants={"W1": w1, "W2": w2},
            expected=[{"Y": projection_reference(px, w1, w2)},
                      {"Y": projection_reference(px2, w1, w2)}],
            math_expected=[{"Y": projection_math_reference(px, w1, w2)},
                           {"Y": projection_math_reference(px2, w1, w2)}],
            kind="projection", rounds=2)
    x_big = np.full((32, 64), 1000.0, dtype=np.float16)
    w_big = np.full((64, 48), 1000.0, dtype=np.float16)
    add_case(
        campaign, manifest, name="matmul_guard_nonfinite_output",
        prim_func=matmul.make_kernel(m=32, k=64, n=48),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"A": x_big}], constants={"B": w_big}, expected=[],
        math_expected=[], kind="guard", rounds=1,
        guard="nonfinite_output")
    add_case(
        campaign, manifest, name="projection_guard_nan_input",
        prim_func=projection.silu_proj(m=13, d=64, h=32),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": np.full((13, 64), np.nan, dtype=np.float16)}],
        constants={"W1": w1, "W2": w2}, expected=[], math_expected=[],
        kind="guard", rounds=1, guard="nan_input")
    add_case(
        campaign, manifest, name="projection_guard_nan_constant",
        prim_func=projection.silu_proj(m=13, d=64, h=32),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[], constants={"W1": w1, "W2": w2}, expected=[],
        math_expected=[], kind="guard", rounds=1, guard="nan_constant",
        bad_constants={"W1": np.full((64, 32), np.nan, dtype=np.float16)})

    fx = f16(rng.normal(0, 0.2, (8, 64)))
    fx2 = f16(rng.normal(0, 0.2, (8, 64)))
    wg = f16(rng.normal(0, 0.1, (64, 32)))
    wu = f16(rng.normal(0, 0.1, (64, 32)))
    wd = f16(rng.normal(0, 0.1, (32, 64)))
    for mode in ("chain", "separate"):
        add_case(
            campaign, manifest, name=f"ffn_{mode}",
            prim_func=ffn.swiglu_ffn_f16(m=8, d=64, h=32, bk=64),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True, submit_mode=mode),
            inputs=[{"X": fx}, {"X": fx2}],
            constants={"Wg": wg, "Wu": wu, "Wd": wd},
            expected=[{"Y": ffn_reference(fx, wg, wu, wd)},
                      {"Y": ffn_reference(fx2, wg, wu, wd)}],
            math_expected=[{"Y": ffn_math_reference(fx, wg, wu, wd)},
                           {"Y": ffn_math_reference(fx2, wg, wu, wd)}],
            kind="ffn", rounds=2)

    rx = f16(rng.normal(0, 0.2, (13, 257)))
    rx2 = f16(rng.normal(0, 0.2, (13, 257)))
    rw = f16(rng.normal(0, 0.1, (257, 33)))
    for mode in ("chain", "separate"):
        add_case(
            campaign, manifest, name=f"ragged_{mode}",
            prim_func=ragged.ragged_tail(m=13, n=33, k=257, bm=16, bn=64),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   submit_mode=mode),
            inputs=[{"A": rx}, {"A": rx2}], constants={"B": rw},
            expected=[{"C": ragged_reference(rx, rw, 257)},
                      {"C": ragged_reference(rx2, rw, 257)}],
            math_expected=[{"C": rx.astype(np.float64) @ rw.astype(np.float64)},
                           {"C": rx2.astype(np.float64) @ rw.astype(np.float64)}],
            kind="ragged", rounds=2)

    # Constant scalar/per-channel Add/Mul, using the existing real TileLang
    # broadcast fixtures.
    bx = f16(rng.normal(0, 0.25, (8, 64)))
    bw = f16(rng.normal(0, 0.15, (64, 33)))
    scalar = f16(np.asarray(0.25))
    channel = f16(np.linspace(-0.5, 0.5, 33, dtype=np.float32))
    broadcast_cases = (
        ("broadcast_scalar_add_chain", broadcast.scalar_add, "add", scalar,
         "chain"),
        ("broadcast_scalar_mul_chain", broadcast.scalar_mul, "mul", scalar,
         "chain"),
        ("broadcast_channel_add_separate", broadcast.channel_add, "add",
         channel, "separate"),
        ("broadcast_channel_mul_separate", broadcast.channel_mul, "mul",
         channel, "separate"),
    )
    for name, factory, op, bias, mode in broadcast_cases:
        add_case(
            campaign, manifest, name=name,
            prim_func=factory(m=8, k=64, n=33),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   submit_mode=mode),
            inputs=[{"X": bx}], constants={"W": bw, "B": bias},
            expected=[{"Y": broadcast_declared(bx, bw, bias, op)}],
            math_expected=[{"Y": broadcast_math(bx, bw, bias, op)}],
            kind="broadcast", rounds=1)

    # Independent activation (generic per-node graph, no matmul).
    from rknpu_backend import (Module, Op, TensorType, Value,
                               activation_module, branch_outputs_module,
                               compile_rkir, compile_tilelang,
                               matmul_module, projection_module,
                               swiglu_module)
    act_x = f16(rng.normal(0, 0.5, (8, 33)))
    act_module = Module(
        name="independent-silu",
        values=(Value(0, "X", TensorType("float16", (8, 33)), None, "input"),
                Value(1, "Y", TensorType("float16", (8, 33)), 0, "output")),
        ops=(Op(0, "silu", (0,), (1,), {}),),
        inputs=(0,), outputs=(1,), source_hash="campaign:independent-silu")
    add_case(
        campaign, manifest, name="independent_silu_chain",
        prim_func=act_module,
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": act_x}], constants={},
        expected=[{"Y": activation_declared(act_x)}],
        math_expected=[{"Y": activation_math(act_x)}],
        kind="activation", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    # 256x256 physical tile forces multiple LUT chunks.
    lut_x = f16(rng.normal(0, 0.15, (256, 256)))
    lut_w = f16(rng.normal(0, 0.08, (256, 256)))
    add_case(
        campaign, manifest, name="large_lut_256x256_chain",
        prim_func=matmul_module(m=256, k=256, n=256, activation="silu"),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"A": lut_x}], constants={"B": lut_w},
        expected=[{"A1": matmul_activation_declared(lut_x, lut_w)}],
        math_expected=[{"A1": matmul_activation_math(lut_x, lut_w)}],
        kind="activation", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    # Two independent outputs from the existing real TileLang branch fixture.
    mo_x = f16(rng.normal(0, 0.2, (8, 64)))
    mo_w1 = f16(rng.normal(0, 0.1, (64, 33)))
    mo_w2 = f16(rng.normal(0, 0.1, (64, 47)))
    add_case(
        campaign, manifest, name="dual_output_chain",
        prim_func=multiout.branch_outputs(m=8, k=64, n1=33, n2=47),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"X": mo_x}], constants={"W1": mo_w1, "W2": mo_w2},
        expected=[{"Y1": matmul_reference(mo_x, mo_w1),
                   "Y2": matmul_reference(mo_x, mo_w2)}],
        math_expected=[{"Y1": matmul_math(mo_x, mo_w1),
                        "Y2": matmul_math(mo_x, mo_w2)}],
        kind="multi_output", rounds=1)

    # M/N multi-tile through the existing real TileLang fixture.
    mn_x = f16(rng.normal(0, 0.15, (512, 512)))
    mn_w = f16(rng.normal(0, 0.08, (512, 384)))
    add_case(
        campaign, manifest, name="mntile_512x384x512_chain",
        prim_func=mntile.mn_tiled(m=512, n=384, k=512, bm=256, bn=128,
                                   bk=256),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"A": mn_x}], constants={"B": mn_w},
        expected=[{"Y": kchain_declared(mn_x, mn_w, 512)}],
        math_expected=[{"Y": matmul_math(mn_x, mn_w)}],
        kind="mntile", rounds=1)

    # Even and odd K shard landing.
    ke_x = f16(rng.normal(0, 0.2, (8, 512)))
    ke_w = f16(rng.normal(0, 0.1, (512, 33)))
    add_case(
        campaign, manifest, name="k_even_512_chain",
        prim_func=kloop.k_loop(m=8, n=33, k=512, bk=256),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"A": ke_x}], constants={"B": ke_w},
        expected=[{"Y": kchain_declared(ke_x, ke_w, 512)}],
        math_expected=[{"Y": matmul_math(ke_x, ke_w)}],
        kind="kchain", rounds=1)
    ko_x = f16(rng.normal(0, 0.2, (8, 257)))
    ko_w = f16(rng.normal(0, 0.1, (257, 33)))
    add_case(
        campaign, manifest, name="k_odd_257_chain",
        prim_func=matmul_module(m=8, k=257, n=33),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"A": ko_x}], constants={"B": ko_w},
        expected=[{"Y": kchain_declared(ko_x, ko_w, 257)}],
        math_expected=[{"Y": matmul_math(ko_x, ko_w)}],
        kind="kchain", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    # Multi-tile per-node dataflow: projection, then FFN and a branch.
    mt_m = mt_d = mt_h = 300
    mt_x = f16(rng.normal(0, 0.15, (mt_m, mt_d)))
    mt_w1 = f16(rng.normal(0, 0.08, (mt_h, mt_d)))
    mt_w2 = f16(rng.normal(0, 0.08, (mt_d, mt_h)))
    add_case(
        campaign, manifest, name="projection_multitile_300_chain",
        prim_func=projection_module(m=mt_m, d=mt_d, h=mt_h),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": mt_x}], constants={"W1": mt_w1, "W2": mt_w2},
        expected=[{"Y": projection_nk_declared(mt_x, mt_w1, mt_w2)}],
        math_expected=[{"Y": projection_nk_math(mt_x, mt_w1, mt_w2)}],
        kind="projection_multitile", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    mt_wg = f16(rng.normal(0, 0.08, (mt_h, mt_d)))
    mt_wu = f16(rng.normal(0, 0.08, (mt_h, mt_d)))
    mt_wd = f16(rng.normal(0, 0.08, (mt_d, mt_h)))
    add_case(
        campaign, manifest, name="ffn_multitile_300_chain",
        prim_func=swiglu_module(m=mt_m, d=mt_d, h=mt_h),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": mt_x}],
        constants={"Wg": mt_wg, "Wu": mt_wu, "Wd": mt_wd},
        expected=[{"Y": ffn_nk_declared(mt_x, mt_wg, mt_wu, mt_wd)}],
        math_expected=[{"Y": ffn_nk_math(mt_x, mt_wg, mt_wu, mt_wd)}],
        kind="ffn_multitile", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    br_x = f16(rng.normal(0, 0.15, (mt_m, mt_d)))
    br_w1 = f16(rng.normal(0, 0.08, (mt_h, mt_d)))
    br_n2 = 200
    br_w2 = f16(rng.normal(0, 0.08, (br_n2, mt_d)))
    add_case(
        campaign, manifest, name="branch_multitile_300_chain",
        prim_func=branch_outputs_module(m=mt_m, k=mt_d, n1=mt_h,
                                        n2=br_n2),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"X": br_x}], constants={"W1": br_w1, "W2": br_w2},
        expected=[{"Y1": kchain_nk_declared(br_x, br_w1),
                   "Y2": kchain_nk_declared(br_x, br_w2)}],
        math_expected=[{"Y1": br_x.astype(np.float64) @ br_w1.T.astype(np.float64),
                        "Y2": br_x.astype(np.float64) @ br_w2.T.astype(np.float64)}],
        kind="branch_multitile", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    at_x = f16(rng.normal(0, 0.4, (300, 300)))
    add_case(
        campaign, manifest, name="activation_multitile_300_chain",
        prim_func=activation_module(m=300, n=300),
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": at_x}], constants={},
        expected=[{"Y": activation_declared(at_x)}],
        math_expected=[{"Y": activation_math(at_x)}],
        kind="activation_multitile", rounds=1, compile_fn=compile_rkir,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    # Real multi-block TileLang staged entry -> same tiled RKIR backend.
    sp_x = f16(rng.normal(0, 0.15, (512, 256)))
    sp_w1 = f16(rng.normal(0, 0.08, (256, 512)))
    sp_w2 = f16(rng.normal(0, 0.08, (512, 256)))
    add_case(
        campaign, manifest, name="projection_tilelang_staged_chain",
        prim_func=projection_staged.projection_staged,
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": sp_x}], constants={"W1": sp_w1, "W2": sp_w2},
        expected=[{"Y": staged_projection_reference(sp_x, sp_w1, sp_w2)}],
        math_expected=[{"Y": staged_projection_math(sp_x, sp_w1, sp_w2)}],
        kind="projection_tilelang_staged", rounds=1, compile_fn=compile_tilelang,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    sf_x = f16(rng.normal(0, 0.15, (512, 256)))
    sf_wg = f16(rng.normal(0, 0.08, (256, 512)))
    sf_wu = f16(rng.normal(0, 0.08, (256, 512)))
    sf_wd = f16(rng.normal(0, 0.08, (512, 256)))
    add_case(
        campaign, manifest, name="ffn_tilelang_staged_chain",
        prim_func=ffn_staged.ffn_staged,
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               allow_approx_lut=True, submit_mode="chain"),
        inputs=[{"X": sf_x}],
        constants={"Wg": sf_wg, "Wu": sf_wu, "Wd": sf_wd},
        expected=[{"Y": staged_ffn_reference(sf_x, sf_wg, sf_wu, sf_wd)}],
        math_expected=[{"Y": staged_ffn_math(sf_x, sf_wg, sf_wu, sf_wd)}],
        kind="ffn_tilelang_staged", rounds=1, compile_fn=compile_tilelang,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    sb_x = f16(rng.normal(0, 0.15, (512, 256)))
    sb_w1 = f16(rng.normal(0, 0.08, (256, 512)))
    sb_w2 = f16(rng.normal(0, 0.08, (256, 256)))
    add_case(
        campaign, manifest, name="branch_tilelang_staged_chain",
        prim_func=branch_staged.branch_staged,
        options=CompileOptions(numeric_mode="rk_f16_tiled",
                               submit_mode="chain"),
        inputs=[{"X": sb_x}], constants={"W1": sb_w1, "W2": sb_w2},
        expected=[{"Y1": kchain_declared(sb_x, sb_w1, 256),
                   "Y2": kchain_declared(sb_x, sb_w2, 256)}],
        math_expected=[{"Y1": sb_x.astype(np.float64) @ sb_w1.astype(np.float64),
                        "Y2": sb_x.astype(np.float64) @ sb_w2.astype(np.float64)}],
        kind="branch_tilelang_staged", rounds=1,
        tolerance={"atol": 8e-3, "rtol": 8e-3})

    # Board CLI compile-rkir -> validate -> explain -> run fixtures.
    from rknpu_backend.serialize import canonical_json
    cli_dir = campaign / "cli"
    cli_dir.mkdir(exist_ok=True)
    cli_rkir = matmul_module(m=32, k=64, n=48)
    (cli_dir / "matmul.rkir.json").write_text(
        canonical_json(cli_rkir.to_dict()) + "\n")
    np.save(cli_dir / "A.npy", x)
    np.save(cli_dir / "B.npy", w)
    np.save(cli_dir / "expected_Y.npy", matmul_reference(x, w))
    (cli_dir / "bindings_success.json").write_text(json.dumps({
        "A": "A.npy", "B": "B.npy"}, indent=2) + "\n")
    (cli_dir / "bindings_missing_constant.json").write_text(json.dumps({
        "A": "A.npy"}, indent=2) + "\n")
    manifest["cli"] = {
        "rkir": "cli/matmul.rkir.json",
        "bindings_success": "cli/bindings_success.json",
        "bindings_bad": "cli/bindings_missing_constant.json",
        "expected_output": "cli/expected_Y.npy",
        "tolerance": {"atol": 8e-3, "rtol": 8e-3},
    }

    manifest.pop("_profile")
    (campaign / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
