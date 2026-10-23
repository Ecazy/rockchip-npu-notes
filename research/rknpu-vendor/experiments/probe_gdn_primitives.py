#!/usr/bin/env python3
"""Probe the RKNN lowering of the primitive ops the GDN graph is built from.

The full GDN layer converts but does not reproduce the CPU reference, so the
first question is which primitive is mis-lowered.  Every case here is a
single-op or few-op micro-graph with one to two inputs, so it can be run with
``gdn_feed`` on the board and compared against onnxruntime on the host.

Usage:
  python3 probe_gdn_primitives.py --out-dir <dir> [--fp16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnx_mapping_shim  # noqa: E402

rng = np.random.default_rng(0)


def vi(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def const(name: str, array) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name)


def save(nodes, inputs, outputs, inits, path: Path) -> Path:
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def case_rmsnorm_chain():
    x = (rng.standard_normal((196, 4096)) * 0.013).astype(np.float32)
    w = np.ones(4096, np.float32)
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["mean", "eps"], ["den"]),
        helper.make_node("Sqrt", ["den"], ["root"]),
        helper.make_node("Div", ["x", "root"], ["div"]),
        helper.make_node("Mul", ["div", "w"], ["y"]),
    ]
    inits = [const("eps", np.float32(1e-6)), const("w", w)]
    save(nodes, [vi("x", x.shape)], [vi("y", x.shape)], inits, Path("rmsnorm_chain.onnx"))
    ref = {}


def case_reduce_mean_last():
    x = (rng.standard_normal((196, 4096)) * 0.013).astype(np.float32)
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["y"], axes=[-1], keepdims=1),
    ]
    save(nodes, [vi("x", x.shape)], [vi("y", [196, 1])], [], Path("reduce_mean_last.onnx"))


def case_sqrt_div():
    x = rng.standard_normal((196, 64)).astype(np.float32)
    d = np.abs(rng.standard_normal((196, 1)).astype(np.float32)) + 0.5
    nodes = [
        helper.make_node("Sqrt", ["d"], ["r"]),
        helper.make_node("Div", ["x", "r"], ["y"]),
    ]
    save(nodes, [vi("x", x.shape), vi("d", d.shape)], [vi("y", x.shape)], [], Path("sqrt_div.onnx"))


def case_reduce_sum_3d():
    x = rng.standard_normal((4, 64, 128)).astype(np.float32)
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["sq"]),
        helper.make_node("ReduceSum", ["sq", "axes"], ["y"], keepdims=1),
    ]
    save(nodes, [vi("x", x.shape)], [vi("y", [4, 64, 1])], [const("axes", np.array([-1], np.int64))], Path("reduce_sum_3d.onnx"))


def case_l2norm_3d():
    x = rng.standard_normal((196, 32, 128)).astype(np.float32)
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["sq"]),
        helper.make_node("ReduceSum", ["sq", "axes"], ["s"], keepdims=1),
        helper.make_node("Add", ["s", "eps"], ["d"]),
        helper.make_node("Sqrt", ["d"], ["r"]),
        helper.make_node("Div", ["x", "r"], ["y"]),
    ]
    inits = [const("axes", np.array([-1], np.int64)), const("eps", np.float32(1e-6))]
    save(nodes, [vi("x", x.shape)], [vi("y", x.shape)], inits, Path("l2norm_3d.onnx"))


def case_batched_matmul_3d():
    a = (rng.standard_normal((32, 64, 128)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((32, 128, 128)) * 0.1).astype(np.float32)
    nodes = [helper.make_node("MatMul", ["a", "b"], ["y"])]
    save(nodes, [vi("a", a.shape), vi("b", b.shape)], [vi("y", [32, 64, 128])], [], Path("batched_matmul_3d.onnx"))


def case_batched_matmul_2d():
    a = (rng.standard_normal((196, 4096)) * 0.013).astype(np.float32)
    b = (rng.standard_normal((4096, 512)) * 0.02).astype(np.float32)
    nodes = [helper.make_node("MatMul", ["a", "b"], ["y"])]
    save(nodes, [vi("a", a.shape), vi("b", b.shape)], [vi("y", [196, 512])], [], Path("matmul_2d.onnx"))


def case_cumsum_exp():
    x = (-np.abs(rng.standard_normal((128, 64)).astype(np.float32)) * 2).astype(np.float32)
    nodes = [
        helper.make_node("CumSum", ["x", "axis"], ["cum"], exclusive=0, reverse=0),
        helper.make_node("Exp", ["cum"], ["y"]),
    ]
    save(nodes, [vi("x", x.shape)], [vi("y", x.shape)], [const("axis", np.array([1], np.int64))], Path("cumsum_exp.onnx"))


def case_softplus_chain():
    x = rng.standard_normal((196, 32)).astype(np.float32)
    nodes = [
        helper.make_node("Abs", ["x"], ["abs"]),
        helper.make_node("Neg", ["abs"], ["neg"]),
        helper.make_node("Exp", ["neg"], ["e"]),
        helper.make_node("Add", ["e", "one"], ["a1"]),
        helper.make_node("Log", ["a1"], ["l"]),
        helper.make_node("Max", ["x", "zero"], ["relu"]),
        helper.make_node("Add", ["l", "relu"], ["y"]),
    ]
    inits = [const("one", np.float32(1)), const("zero", np.float32(0))]
    save(nodes, [vi("x", x.shape)], [vi("y", x.shape)], inits, Path("softplus_chain.onnx"))


def case_pad3d():
    x = rng.standard_normal((196, 4, 128)).astype(np.float32)
    nodes = [helper.make_node("Pad", ["x", "pads", "zero"], ["y"])]
    inits = [const("pads", np.array([0, 0, 0, 60, 0, 0], np.int64)), const("zero", np.float32(0))]
    save(nodes, [vi("x", x.shape)], [vi("y", [256, 4, 128])], inits, Path("pad3d.onnx"))


def case_slice_reshape():
    x = rng.standard_normal((32, 4, 64, 128)).astype(np.float32)
    nodes = [
        helper.make_node("Slice", ["x", "st", "en", "ax"], ["s"]),
        helper.make_node("Reshape", ["s", "shape"], ["y"]),
    ]
    inits = [
        const("st", np.array([1], np.int64)),
        const("en", np.array([2], np.int64)),
        const("ax", np.array([1], np.int64)),
        const("shape", np.array([32, 64, 128], np.int64)),
    ]
    save(nodes, [vi("x", x.shape)], [vi("y", [32, 64, 128])], inits, Path("slice_reshape.onnx"))


def case_triu_mask_exp():
    cum = (-np.abs(rng.standard_normal((128, 64)).astype(np.float32)) * 2).astype(np.float32)
    bias = ((np.tril(np.ones((64, 64), np.float32)) - 1.0) * 1e4).astype(np.float32)
    nodes = [
        helper.make_node("Unsqueeze", ["cum", "ax1"], ["ci"]),
        helper.make_node("Unsqueeze", ["cum", "ax2"], ["cj"]),
        helper.make_node("Sub", ["ci", "cj"], ["diff"]),
        helper.make_node("Add", ["diff", "bias"], ["dm"]),
        helper.make_node("Exp", ["dm"], ["y"]),
    ]
    inits = [const("ax1", np.array([-1], np.int64)), const("ax2", np.array([-2], np.int64)), const("bias", bias)]
    save(nodes, [vi("cum", cum.shape)], [vi("y", [128, 64, 64])], inits, Path("triu_mask_exp.onnx"))


def case_nilpotent_chain():
    m = (np.tril(rng.standard_normal((32, 64, 64)), -1) * 0.2).astype(np.float32)
    eye = np.eye(64, dtype=np.float32)
    nodes = [
        helper.make_node("Add", ["m", "eye"], ["p1"]),
        helper.make_node("MatMul", ["m", "m"], ["m2"]),
        helper.make_node("Add", ["m2", "eye"], ["p2"]),
        helper.make_node("MatMul", ["p1", "p2"], ["a2"]),
        helper.make_node("MatMul", ["m2", "m2"], ["m4"]),
        helper.make_node("Add", ["m4", "eye"], ["p4"]),
        helper.make_node("MatMul", ["a2", "p4"], ["y"]),
    ]
    save(nodes, [vi("m", m.shape)], [vi("y", [32, 64, 64])], [const("eye", eye)], Path("nilpotent_chain.onnx"))



def case_slice_only():
    x = rng.standard_normal((32, 4, 64, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])]
    inits = [const("st", np.array([1], np.int64)), const("en", np.array([2], np.int64)), const("ax", np.array([1], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [32, 1, 64, 128])], inits, Path("slice_only.onnx"))


def case_reshape_only():
    x = rng.standard_normal((32, 4, 64, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["y"])]
    inits = [const("shape", np.array([128, 64, 128], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [128, 64, 128])], inits, Path("reshape_only.onnx"))


def case_transpose4d():
    x = rng.standard_normal((32, 4, 64, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Transpose", ["x"], ["y"], perm=[1, 2, 0, 3])]
    save(nodes, [vi("x", x.shape)], [vi("y", [4, 64, 32, 128])], [], Path("transpose4d.onnx"))


def case_concat_axis0():
    x = rng.standard_normal((32, 4, 64, 128)).astype(np.float32) * 0.1
    y2 = rng.standard_normal((32, 4, 64, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Concat", ["x", "y"], ["z"], axis=1)]
    save(nodes, [vi("x", x.shape), vi("y", y2.shape)], [vi("z", [32, 8, 64, 128])], [], Path("concat_axis1.onnx"))


def case_broadcast_mask():
    state = rng.standard_normal((32, 128, 128)).astype(np.float32) * 0.1
    mask = (rng.integers(0, 2, (32, 1, 1))).astype(np.float32)
    nodes = [helper.make_node("Mul", ["state", "mask"], ["y"])]
    save(nodes, [vi("state", state.shape), vi("mask", mask.shape)], [vi("y", state.shape)], [], Path("broadcast_mask.onnx"))


def case_reduce_axis1_4d():
    x = rng.standard_normal((32, 4, 128, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)]
    save(nodes, [vi("x", x.shape)], [vi("y", [32, 1, 128, 128])], [const("axes", np.array([1], np.int64))], Path("reduce_axis1_4d.onnx"))


def case_reshape_and_matmul():
    # [H*C,T,D] layout: matmul against a per-head state tiled to H*C
    a = rng.standard_normal((128, 64, 128)).astype(np.float32) * 0.1
    b = rng.standard_normal((128, 128, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("MatMul", ["a", "b"], ["y"])]
    save(nodes, [vi("a", a.shape), vi("b", b.shape)], [vi("y", [128, 64, 128])], [], Path("matmul_128x64x128.onnx"))


def case_transpose3d():
    x = rng.standard_normal((32, 256, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Transpose", ["x"], ["y"], perm=[1, 0, 2])]
    save(nodes, [vi("x", x.shape)], [vi("y", [256, 32, 128])], [], Path("transpose3d.onnx"))


def case_reshape3d():
    x = rng.standard_normal((32, 256, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["y"])]
    inits = [const("shape", np.array([128, 64, 128], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [128, 64, 128])], inits, Path("reshape3d.onnx"))


def case_reshape_3d_to_2d():
    x = rng.standard_normal((1, 256, 64)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["y"])]
    inits = [const("shape", np.array([256, 64], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [256, 64])], inits, Path("reshape_3d_to_2d.onnx"))


def case_conv1d_depthwise():
    channels, length, kernel = 64, 256, 4
    x = rng.standard_normal((1, channels, 1, length)).astype(np.float32) * 0.1
    w = (rng.standard_normal((channels, 1, 1, kernel)) * 0.5).astype(np.float32)
    nodes = [helper.make_node("Conv", ["x", "w"], ["y"], group=channels,
                              kernel_shape=[1, kernel], pads=[0, kernel - 1, 0, 0])]
    save(nodes, [vi("x", x.shape)], [vi("y", [1, channels, 1, length])], [const("w", w)], Path("conv1d_depthwise.onnx"))


def case_sigmoid_silu():
    x = rng.standard_normal((256, 64)).astype(np.float32) * 0.1
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["s"]),
        helper.make_node("Mul", ["x", "s"], ["y"]),
    ]
    save(nodes, [vi("x", x.shape)], [vi("y", x.shape)], [], Path("sigmoid_silu.onnx"))


def case_matmul_2d_bias():
    a = rng.standard_normal((256, 512)).astype(np.float32) * 0.013
    b = (rng.standard_normal((512, 256)) * 0.02).astype(np.float32)
    nodes = [helper.make_node("MatMul", ["a", "b"], ["y"])]
    save(nodes, [vi("a", a.shape), vi("b", b.shape)], [vi("y", [256, 256])], [], Path("matmul_2d_bias.onnx"))


def case_pad2d():
    x = rng.standard_normal((256, 64)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Pad", ["x", "pads", "zero"], ["y"])]
    inits = [const("pads", np.array([0, 0, 3, 0], np.int64)), const("zero", np.float32(0))]
    save(nodes, [vi("x", x.shape)], [vi("y", [259, 64])], inits, Path("pad2d.onnx"))


def case_slice2d_axis0():
    x = rng.standard_normal((259, 64)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])]
    inits = [const("st", np.array([0], np.int64)), const("en", np.array([256], np.int64)), const("ax", np.array([0], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [256, 64])], inits, Path("slice2d_axis0.onnx"))


def case_slice2d_axis1():
    x = rng.standard_normal((256, 192)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])]
    inits = [const("st", np.array([64], np.int64)), const("en", np.array([128], np.int64)), const("ax", np.array([1], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [256, 64])], inits, Path("slice2d_axis1.onnx"))


def case_tile3d():
    x = rng.standard_normal((32, 1, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Tile", ["x", "reps"], ["y"])]
    inits = [const("reps", np.array([1, 4, 1], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [32, 4, 128])], inits, Path("tile3d.onnx"))


def case_mul_row_broadcast():
    x = rng.standard_normal((256, 64)).astype(np.float32) * 0.1
    w = (rng.standard_normal((1, 64)) * 0.5).astype(np.float32)
    nodes = [
        helper.make_node("Mul", ["x", "w"], ["y"]),
    ]
    save(nodes, [vi("x", x.shape), vi("w", w.shape)], [vi("y", [256, 64])], [], Path("mul_row_broadcast.onnx"))


def case_conv1d_via_shifts():
    length, channels, kernel = 64, 32, 4
    x = rng.standard_normal((length, channels)).astype(np.float32) * 0.1
    w = (rng.standard_normal((channels, kernel)) * 0.5).astype(np.float32)
    nodes = []
    inits = [const("w", w)]
    # x_pad: [length + kernel - 1, channels] with zeros on top
    inits.append(const("pad", np.array([kernel - 1, 0, 0, 0], np.int64)))
    inits.append(const("zero", np.float32(0)))
    nodes.append(helper.make_node("Pad", ["x", "pad", "zero"], ["xp"]))
    inits.append(const("ax0", np.array([0], np.int64)))
    acc = None
    for j in range(kernel):
        st, en = j, j + length
        inits.append(const(f"st{j}", np.array([st], np.int64)))
        inits.append(const(f"en{j}", np.array([en], np.int64)))
        inits.append(const(f"wj{j}", w[:, j].reshape(1, channels)))
        nodes.append(helper.make_node("Slice", ["xp", f"st{j}", f"en{j}", "ax0"], [f"s{j}"]))
        nodes.append(helper.make_node("Mul", [f"s{j}", f"wj{j}"], [f"m{j}"]))
        if acc is None:
            acc = f"m{j}"
        else:
            nodes.append(helper.make_node("Add", [acc, f"m{j}"], [f"a{j}"]))
            acc = f"a{j}"
    nodes.append(helper.make_node("Identity", [acc], ["y"]))
    save(nodes, [vi("x", x.shape)], [vi("y", [length, channels])], inits, Path("conv1d_via_shifts.onnx"))


def case_softmax_free_reduce_c():
    x = rng.standard_normal((128, 64, 128)).astype(np.float32) * 0.1
    nodes = [
        helper.make_node("Reshape", ["x", "shape"], ["r"]),
        helper.make_node("ReduceSum", ["r", "axes"], ["s"], keepdims=1),
        helper.make_node("Reshape", ["s", "shape2"], ["y"]),
    ]
    inits = [const("shape", np.array([32, 4, 64 * 128], np.int64)),
             const("axes", np.array([1], np.int64)),
             const("shape2", np.array([32, 64, 128], np.int64))]
    save(nodes, [vi("x", x.shape)], [vi("y", [32, 128, 128])], inits, Path("reduce_chunk_axis.onnx"))


def case_gather3d_axis1():
    x = rng.standard_normal((196, 16, 128)).astype(np.float32) * 0.1
    idx = np.repeat(np.arange(16), 2).astype(np.int64)
    nodes = [helper.make_node("Gather", ["x", "idx"], ["y"], axis=1)]
    save(nodes, [vi("x", x.shape)], [vi("y", [196, 32, 128])], [const("idx", idx)], Path("gather3d_axis1.onnx"))


def case_concat2d_axis1():
    x = rng.standard_normal((196, 64)).astype(np.float32) * 0.1
    y2 = rng.standard_normal((196, 64)).astype(np.float32) * 0.1
    z = rng.standard_normal((196, 64)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Concat", ["x", "y", "z"], ["w"], axis=1)]
    save(nodes, [vi("x", x.shape), vi("y", y2.shape), vi("z", z.shape)], [vi("w", [196, 192])], [], Path("concat2d_axis1.onnx"))


def case_concat3d_axis0():
    x = rng.standard_normal((32, 64, 128)).astype(np.float32) * 0.1
    y2 = rng.standard_normal((32, 64, 128)).astype(np.float32) * 0.1
    nodes = [helper.make_node("Concat", ["x", "y"], ["z"], axis=0)]
    save(nodes, [vi("x", x.shape), vi("y", y2.shape)], [vi("z", [64, 64, 128])], [], Path("concat3d_axis0.onnx"))


def case_gather3d_axis0_repeat():
    x = rng.standard_normal((32, 128, 128)).astype(np.float32) * 0.1
    idx = np.repeat(np.arange(32), 4).astype(np.int64)
    nodes = [helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)]
    save(nodes, [vi("x", x.shape)], [vi("y", [128, 128, 128])], [const("idx", idx)], Path("gather3d_axis0_repeat.onnx"))


def case_boundary_rmsnorm():
    L, H = 196, 4096
    rng2 = np.random.default_rng(7)
    x = (rng2.standard_normal((1, L, H)) * 0.013).astype(np.float32)
    w = (np.abs(rng2.standard_normal(H)) * 0.05 + 0.01).astype(np.float32)
    nodes = [
        helper.make_node("Reshape", ["x", "shape"], ["xf"]),
        helper.make_node("Mul", ["xf", "xf"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["m"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["m", "eps"], ["d"]),
        helper.make_node("Sqrt", ["d"], ["r"]),
        helper.make_node("Div", ["xf", "r"], ["n"]),
        helper.make_node("Mul", ["n", "w"], ["y"]),
    ]
    inits = [const("shape", np.array([L, H], np.int64)), const("eps", np.float32(1e-6)), const("w", w.reshape(1, H))]
    save(nodes, [vi("x", x.shape)], [vi("y", [L, H])], inits, Path("boundary_rmsnorm.onnx"))

CASES = {
    "rmsnorm_chain": (case_rmsnorm_chain, ["x"], lambda d: d["x"] / np.sqrt((d["x"] ** 2).mean(-1, keepdims=True) + 1e-6)),
    "reduce_mean_last": (case_reduce_mean_last, ["x"], lambda d: (d["x"] ** 2).mean(-1, keepdims=True)),
    "sqrt_div": (case_sqrt_div, ["x", "d"], lambda d: d["x"] / np.sqrt(d["d"])),
    "reduce_sum_3d": (case_reduce_sum_3d, ["x"], lambda d: (d["x"] ** 2).sum(-1, keepdims=True)),
    "l2norm_3d": (case_l2norm_3d, ["x"], lambda d: d["x"] / np.sqrt((d["x"] ** 2).sum(-1, keepdims=True) + 1e-6)),
    "batched_matmul_3d": (case_batched_matmul_3d, ["a", "b"], lambda d: d["a"] @ d["b"]),
    "matmul_2d": (case_batched_matmul_2d, ["a", "b"], lambda d: d["a"] @ d["b"]),
    "cumsum_exp": (case_cumsum_exp, ["x"], lambda d: np.exp(np.cumsum(d["x"], axis=1))),
    "softplus_chain": (case_softplus_chain, ["x"], lambda d: np.log1p(np.exp(-np.abs(d["x"]))) + np.maximum(d["x"], 0)),
    "pad3d": (case_pad3d, ["x"], lambda d: np.pad(d["x"], ((0, 60), (0, 0), (0, 0)))),
    "slice_reshape": (case_slice_reshape, ["x"], lambda d: d["x"][:, 1]),
    "triu_mask_exp": (case_triu_mask_exp, ["cum"], None),
    "nilpotent_chain": (case_nilpotent_chain, ["m"], None),
    "slice_only": (case_slice_only, ["x"], None),
    "reshape_only": (case_reshape_only, ["x"], None),
    "transpose4d": (case_transpose4d, ["x"], None),
    "concat_axis1": (case_concat_axis0, ["x", "y"], None),
    "broadcast_mask": (case_broadcast_mask, ["state", "mask"], None),
    "reduce_axis1_4d": (case_reduce_axis1_4d, ["x"], None),
    "matmul_128x64x128": (case_reshape_and_matmul, ["a", "b"], None),
    "transpose3d": (case_transpose3d, ["x"], None),
    "reshape3d": (case_reshape3d, ["x"], None),
    "reshape_3d_to_2d": (case_reshape_3d_to_2d, ["x"], None),
    "conv1d_depthwise": (case_conv1d_depthwise, ["x"], None),
    "sigmoid_silu": (case_sigmoid_silu, ["x"], None),
    "matmul_2d_bias": (case_matmul_2d_bias, ["a", "b"], None),
    "pad2d": (case_pad2d, ["x"], None),
    "slice2d_axis0": (case_slice2d_axis0, ["x"], None),
    "slice2d_axis1": (case_slice2d_axis1, ["x"], None),
    "tile3d": (case_tile3d, ["x"], None),
    "mul_row_broadcast": (case_mul_row_broadcast, ["x", "w"], None),
    "conv1d_via_shifts": (case_conv1d_via_shifts, ["x"], None),
    "reduce_chunk_axis": (case_softmax_free_reduce_c, ["x"], None),
    "gather3d_axis1": (case_gather3d_axis1, ["x"], None),
    "concat2d_axis1": (case_concat2d_axis1, ["x", "y", "z"], None),
    "concat3d_axis0": (case_concat3d_axis0, ["x", "y"], None),
    "gather3d_axis0_repeat": (case_gather3d_axis0_repeat, ["x"], None),
    "boundary_rmsnorm": (case_boundary_rmsnorm, ["x"], None),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    onnx_mapping_shim.install()
    from rknn.api import RKNN
    import onnxruntime as ort

    manifest = {}
    for name, (builder, input_names, ref_fn) in CASES.items():
        if args.only and name not in args.only:
            continue
        case_dir = args.out_dir / name
        case_dir.mkdir(exist_ok=True)
        import os

        cwd = Path.cwd()
        os.chdir(case_dir)
        try:
            builder()
        finally:
            os.chdir(cwd)
        onnx_path = case_dir / f"{name}.onnx"

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feeds = {
            inp.name: rng.standard_normal(tuple(inp.shape)).astype(np.float32)
            for inp in sess.get_inputs()
        }
        # every case needs realistic magnitudes, not unit normals
        feeds = rescale(name, feeds)
        paths = []
        for idx, key in enumerate(input_names):
            path = case_dir / f"in{idx}_{key}.bin"
            feeds[key].astype(np.float32).tofile(path)
            paths.append(str(path))

        (case_dir / "dataset.txt").write_text(" ".join(paths) + "\n", encoding="utf-8")
        ref_out = sess.run(None, feeds)[0]
        np.save(case_dir / "ref.npy", ref_out)

        rknn = RKNN(verbose=False)
        rknn.config(target_platform="rk3588", quantized_dtype="w8a8", optimization_level=3)
        status = "ok"
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            status = "load_failed"
        elif rknn.build(do_quantization=not args.fp16, dataset=str(case_dir / "dataset.txt")) != 0:
            status = "build_failed"
        elif rknn.export_rknn(str(case_dir / f"{name}.rknn")) != 0:
            status = "export_failed"
        rknn.release()
        manifest[name] = {"status": status, "inputs": paths, "dir": str(case_dir)}

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def rescale(name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if name in ("rmsnorm_chain", "reduce_mean_last", "matmul_2d"):
        for key in list(feeds):
            feeds[key] = (feeds[key] * 0.013).astype(np.float32) if key != "b" else (feeds[key] * 0.02).astype(np.float32)
        if "w" in feeds:
            feeds["w"] = np.ones_like(feeds["w"])
    else:
        for key in list(feeds):
            feeds[key] = (feeds[key] * 0.1).astype(np.float32)
        if "d" in feeds:
            feeds["d"] = np.abs(feeds["d"]) + 0.5
    return feeds


if __name__ == "__main__":
    raise SystemExit(main())
