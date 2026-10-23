#!/usr/bin/env python3
"""Build the Qwen3.5 GatedDeltaNet (GDN) decoder layer-0 subgraph as ONNX.

The graph is a transcription of
``transformers/models/qwen3_5/modeling_qwen3_5.py``
(``Qwen3_5GatedDeltaNet.forward`` + ``torch_chunk_gated_delta_rule``), restricted to
the *prefill* path (no KV/conv cache, no decoding state):

    h   = RMSNorm(x)                                  input_layernorm
    m   = concat(q, k, v) = h @ W_in_qkv^T            [L, 2*key_dim + value_dim]
    z   = h @ W_in_z^T                                [L, value_dim]
    b   = h @ W_in_b^T ; a = h @ W_in_a^T             [L, num_v_heads]
    m   = SiLU(causal_conv1d(m, k=4, groups=conv_dim, left pad 3))
    q,k,v = split(m)
    q   = l2norm(q) * key_head_dim**-0.5 ; k = l2norm(k)
    beta = sigmoid(b)
    g    = -exp(A_log) * softplus(a + dt_bias)        <= 0, log-space decay
    q,k = repeat_interleave(q,k, num_v_heads/num_k_heads)
    core = chunk_gated_delta_rule(q, k, v, g, beta, chunk=64)   # UT transform + scan
    core = RMSNormGated(core, z) * norm.weight
    out  = core @ W_out^T

Two implementations live in this file and must agree:

* ``reference_layer`` — plain numpy, written directly from the equations above
  (used as the CPU oracle and to sanity-check the ONNX graph).
* ``build_onnx`` — the same computation as an ONNX graph of standard ops, so
  rknn-toolkit2 can lower it for the RK3588 NPU.

The chunked delta rule is the only non-obvious part.  HF solves the unit lower
triangular system with ``torch.linalg.solve_triangular``; ONNX has no triangular
solve, so the inverse of (I + L) is built with the nilpotent doubling identity

    (I + L)^-1 = (I + M)(I + M^2)(I + M^4)(I + M^8)(I + M^16)(I + M^32),  M = -L

which is exact for a strictly lower triangular 64x64 ``L`` (M^64 = 0) and costs
5 squarings + 5 multiplies instead of a 64-step forward substitution.  The
graph therefore stays a matmul-dominated, hardware-friendly pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class GDNConfig:
    """Qwen3.5 GatedDeltaNet layer geometry.

    Defaults are the real ``Qwen/Qwen3.5-9B`` text config values.
    """

    hidden: int = 4096
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    seq: int = 196
    chunk: int = 64
    rms_eps: float = 1e-6
    l2_eps: float = 1e-6

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def group(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def seq_pad(self) -> int:
        rem = self.seq % self.chunk
        return self.seq if rem == 0 else self.seq + (self.chunk - rem)

    @property
    def num_chunks(self) -> int:
        return self.seq_pad // self.chunk

    def validate(self) -> None:
        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be a multiple of num_k_heads"
        assert self.head_k_dim == self.head_v_dim or True  # unequal dims are allowed by HF
        assert self.conv_kernel >= 1
        assert self.chunk in (16, 32, 64, 128), "chunk size drives the doubling depth"


WEIGHTS = (
    "input_layernorm",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "conv1d",
    "dt_bias",
    "A_log",
    "norm",
    "out_proj",
)


def random_weights(cfg: GDNConfig, seed: int = 0, scale: float = 0.02) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = {
        "input_layernorm": np.ones(cfg.hidden, np.float32),
        "in_proj_qkv": (rng.standard_normal((cfg.conv_dim, cfg.hidden)) * scale).astype(np.float32),
        "in_proj_z": (rng.standard_normal((cfg.value_dim, cfg.hidden)) * scale).astype(np.float32),
        "in_proj_b": (rng.standard_normal((cfg.num_v_heads, cfg.hidden)) * scale).astype(np.float32),
        "in_proj_a": (rng.standard_normal((cfg.num_v_heads, cfg.hidden)) * scale).astype(np.float32),
        "conv1d": (rng.standard_normal((cfg.conv_dim, 1, cfg.conv_kernel)) * 0.5).astype(np.float32),
        "dt_bias": np.ones(cfg.num_v_heads, np.float32),
        "A_log": np.log(rng.uniform(0.01, 16.0, cfg.num_v_heads)).astype(np.float32),
        "norm": np.ones(cfg.head_v_dim, np.float32),
        "out_proj": (rng.standard_normal((cfg.hidden, cfg.value_dim)) * scale).astype(np.float32),
    }
    return w


# --------------------------------------------------------------------------
# numpy reference (oracle)
# --------------------------------------------------------------------------


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    # log(1 + exp(x)) with the overflow-safe branch HF uses via F.softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _l2norm(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _causal_conv1d(x: np.ndarray, weight: np.ndarray, kernel: int) -> np.ndarray:
    """x: [L, conv_dim], weight: [conv_dim, 1, kernel], left pad = kernel-1."""
    length = x.shape[0]
    out = np.zeros_like(x)
    for j in range(kernel):
        # conv1d with padding=kernel-1: out[t] += w[j] * x[t + j - (kernel-1)]
        idx = np.arange(length) + j - (kernel - 1)
        valid = idx >= 0
        out[valid] += x[idx[valid]] * weight[:, 0, j][None, :]
    return out



def _solve_lower_unit(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve (I + L) X = B for strictly lower-triangular L (forward substitution).

    This is what ``torch.linalg.solve_triangular(..., unitriangular=True)`` does
    inside ``torch_chunk_gated_delta_rule``; the ONNX graph has to use the
    nilpotent-doubling inverse instead, but the numpy oracle follows the
    reference solver so its result is not biased by the doubling formulation.
    """
    X = B.copy()
    if L.ndim == 2:
        for i in range(1, L.shape[0]):
            X[i] -= L[i, :i] @ X[:i]
        return X
    # batched: L [B,T,T], X [B,T,D]
    for i in range(1, L.shape[1]):
        X[:, i] -= np.einsum("bj,bjd->bd", L[:, i, :i], X[:, :i])
    return X

def _chunk_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    cfg: GDNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """q,k: [L, H, Dk]; v: [L, H, Dv]; g, beta: [L, H].  No initial state."""
    length = q.shape[0]
    heads = q.shape[1]
    dk, dv = q.shape[-1], v.shape[-1]
    pad = cfg.seq_pad - length
    if pad:
        q = np.pad(q, ((0, pad), (0, 0), (0, 0)))
        k = np.pad(k, ((0, pad), (0, 0), (0, 0)))
        v = np.pad(v, ((0, pad), (0, 0), (0, 0)))
        g = np.pad(g, ((0, pad), (0, 0)))
        beta = np.pad(beta, ((0, pad), (0, 0)))

    chunks, step = cfg.num_chunks, cfg.chunk
    # [H, C, T, D] -> [H*C, T, D]
    def to_chunks(x, last):
        return x.transpose(1, 0, 2).reshape(heads, chunks, step, last).reshape(heads * chunks, step, last)

    qc, kc, vc = to_chunks(q, dk), to_chunks(k, dk), to_chunks(v, dv)
    gc = g.transpose(1, 0).reshape(heads, chunks, step).reshape(heads * chunks, step)
    bc = beta.transpose(1, 0).reshape(heads, chunks, step).reshape(heads * chunks, step)

    v_beta = vc * bc[..., None]
    k_beta = kc * bc[..., None]
    cum = np.cumsum(gc, axis=-1)  # [HC, T]
    # pairwise decay: exp(cum_i - cum_j) for i >= j, else 0
    diff = cum[:, :, None] - cum[:, None, :]
    tril = np.tril(np.ones((step, step), np.float32))
    pwd = np.exp(np.where(tril > 0, diff, -1e4))

    ut = (k_beta @ kc.transpose(0, 2, 1)) * pwd
    intra = (qc @ kc.transpose(0, 2, 1)) * pwd
    decayed_k_beta = k_beta * np.exp(cum)[..., None]

    # (I + L) X = B with L = strict_lower(ut), forward substitution (reference
    # solver).  The exported ONNX graph uses the nilpotent-doubling inverse
    # instead because ONNX has no triangular solve; that formulation is far more
    # sensitive to conditioning, which is exactly why the oracle must not use it.
    lower = np.tril(ut, -1)
    new_values = _solve_lower_unit(lower, v_beta)
    k_cumdecay = _solve_lower_unit(lower, decayed_k_beta)

    q_dec = qc * np.exp(cum)[..., None]
    k_dec = kc * np.exp(cum[:, -1:] - cum)[..., None]
    chunk_decay = np.exp(cum[:, -1])

    state = np.zeros((heads * chunks, dk, dv), np.float32)
    out = np.zeros((heads * chunks, step, dv), np.float32)
    # chunks are grouped per head: index = head * num_chunks + c
    for h in range(heads):
        state_h = np.zeros((dk, dv), np.float32)
        for c in range(chunks):
            i = h * chunks + c
            v_new = new_values[i] - k_cumdecay[i] @ state_h
            out[i] = q_dec[i] @ state_h + intra[i] @ v_new
            state_h = state_h * chunk_decay[i] + k_dec[i].T @ v_new
        state[h * chunks : (h + 1) * chunks] = state_h

    # [h*C+c, t, d] -> [h, c, t, d] -> [c, t, h, d] -> [c*T+t, h, d]
    out = out.reshape(heads, chunks, step, dv).transpose(1, 2, 0, 3).reshape(cfg.seq_pad, heads, dv)
    state = state.reshape(heads, chunks, dk, dv)[:, -1].transpose(0, 1, 2)
    return out[:length], state


def reference_layer(cfg: GDNConfig, w: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: [1, L, hidden] -> (out [1, L, hidden], final_state [H, Dk, Dv])."""
    length = x.shape[1]
    h = x.reshape(length, cfg.hidden).astype(np.float32)
    var = (h * h).mean(-1, keepdims=True)
    h = (h / np.sqrt(var + cfg.rms_eps)) * w["input_layernorm"][None, :]

    mixed = h @ w["in_proj_qkv"].T
    z = h @ w["in_proj_z"].T
    b = h @ w["in_proj_b"].T
    a = h @ w["in_proj_a"].T

    mixed = _silu(_causal_conv1d(mixed, w["conv1d"], cfg.conv_kernel))
    q = mixed[:, : cfg.key_dim].reshape(length, cfg.num_k_heads, cfg.head_k_dim)
    k = mixed[:, cfg.key_dim : 2 * cfg.key_dim].reshape(length, cfg.num_k_heads, cfg.head_k_dim)
    v = mixed[:, 2 * cfg.key_dim :].reshape(length, cfg.num_v_heads, cfg.head_v_dim)

    q = _l2norm(q, cfg.l2_eps) * (cfg.head_k_dim**-0.5)
    k = _l2norm(k, cfg.l2_eps)
    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(w["A_log"])[None, :] * _softplus(a + w["dt_bias"][None, :])
    if cfg.group > 1:
        q = np.repeat(q, cfg.group, axis=1)
        k = np.repeat(k, cfg.group, axis=1)

    core, state = _chunk_gated_delta_rule(q, k, v, g, beta, cfg)

    rows = core.reshape(-1, cfg.head_v_dim)
    variance = (rows * rows).mean(-1, keepdims=True)
    rows = rows / np.sqrt(variance + cfg.rms_eps)
    rows = rows * w["norm"][None, :]
    rows = rows * _silu(z.reshape(-1, cfg.head_v_dim))
    out = rows.reshape(1, length, cfg.value_dim) @ w["out_proj"].T
    return out, state


# --------------------------------------------------------------------------
# ONNX graph
# --------------------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def const(self, name: str, value: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, op_type: str, inputs: list[str], name: str, **attrs) -> str:
        # the requested name is the tensor name; the node name gets a suffix so a
        # reader can tell graph nodes from tensors in the RKNN op dump
        self.nodes.append(helper.make_node(op_type, inputs, [name], name=f"{name}_node", **attrs))
        self._n += 1
        return name

    def matmul(self, a: str, b: str, name: str) -> str:
        return self.op("MatMul", [a, b], name)

    def unsqueeze(self, tensor: str, axes, name: str) -> str:
        axes_name = f"{name}_axes"
        self.const(axes_name, np.asarray(axes, np.int64))
        return self.op("Unsqueeze", [tensor, axes_name], name)


DEBUG_TENSORS = (
    "rms_scale", "proj_qkv", "proj_z", "conv_silu", "reshape_q", "reshape_k", "reshape_v",
    "l2_scale_q", "beta_sigmoid", "gate_decay", "cum_decay", "pair_decay", "ut_system",
    "intra_chunk_attn", "nilpotent_acc5", "ut_solve_v", "ut_solve_k", "q_decayed",
    "chunk_out0", "chunk_out3", "core_trim", "gated_silu", "out_proj",
)


def _finalize(gb: "_Graph", inputs, outputs, path: Path, debug_outputs: bool) -> None:
    graph = helper.make_graph(gb.nodes, path.stem, inputs, outputs, gb.inits, value_info=[])
    model = helper.make_model(graph, producer_name="rockchip-npu-notes/gdn-layer", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    if debug_outputs:
        model = onnx.shape_inference.infer_shapes(model)
    else:
        onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_onnx(
    cfg: GDNConfig,
    w: dict[str, np.ndarray],
    path: Path,
    debug_outputs: bool = False,
    final_state_output: bool = False,
    value_scale: float = 1.0,
    front_end: bool = False,
    tail: bool = False,
) -> None:
    """Emit the layer as an ONNX graph with **no 4-D tensors**.

    Every tensor here is 2-D or 3-D on purpose.  rknn-toolkit2 2.3.2 treats any
    4-D tensor as an NCHW image (Slice/Reshape/Transpose/Concat on 4-D inputs
    are all mis-lowered, see ``probe_gdn_primitives.py``), so the graph keeps
    the depthwise conv1d as four shifted 2-D multiply-adds and keeps the
    chunked delta rule as ``[heads*chunks, chunk, dim]`` 3-D matmuls with
    constant chunk masks instead of per-chunk slicing.
    """
    gb = _Graph()
    length = cfg.seq
    heads, chunks, step = cfg.num_v_heads, cfg.num_chunks, cfg.chunk
    hc = heads * chunks
    dk, dv = cfg.head_k_dim, cfg.head_v_dim
    pad = cfg.seq_pad - length

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, length, cfg.hidden])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, length, cfg.hidden])
    state_out = helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [heads, dk, dv])

    def ic(name: str, value) -> str:
        gb.const(name, np.asarray(value, np.int64))
        return name

    def fc(name: str, value) -> str:
        gb.const(name, np.asarray(value, np.float32))
        return name

    # --- 1. input norm + projections (all 2-D) ----------------------------
    gb.const("flat_shape", np.array([length, cfg.hidden], np.int64))
    flat = gb.op("Reshape", ["x", "flat_shape"], "x_flat")
    sq = gb.op("Mul", [flat, flat], "rms_sq")
    mean = gb.op("ReduceMean", [sq], "rms_mean", axes=[-1], keepdims=1)
    mean = gb.op("Add", [mean, fc("rms_eps", cfg.rms_eps)], "rms_add")
    root = gb.op("Sqrt", [mean], "rms_sqrt")
    normed = gb.op("Div", [flat, root], "rms_div")
    for key in ("input_layernorm", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        gb.const(f"{key}_w", np.ascontiguousarray(w[key].T))
    h = gb.op("Mul", [normed, "input_layernorm_w"], "rms_scale")
    mixed = gb.matmul(h, "in_proj_qkv_w", "proj_qkv")
    z = gb.matmul(h, "in_proj_z_w", "proj_z")
    b = gb.matmul(h, "in_proj_b_w", "proj_b")
    a = gb.matmul(h, "in_proj_a_w", "proj_a")

    # --- 2. causal depthwise conv1d as shifted 2-D multiply-adds ----------
    gb.const("conv_pad", np.array([cfg.conv_kernel - 1, 0, 0, 0], np.int64))
    padded = gb.op("Pad", [mixed, "conv_pad", fc("zero", 0.0)], "conv_input_pad")
    acc = None
    for j in range(cfg.conv_kernel):
        start = ic(f"conv_slice_{j}_start", [j])
        stop = ic(f"conv_slice_{j}_end", [j + length])
        axis = ic("axis0", [0]) if j == 0 else "axis0"
        shifted = gb.op("Slice", [padded, start, stop, axis], f"conv_shift_{j}")
        gb.const(f"conv_w_{j}", w["conv1d"][:, 0, j].reshape(1, cfg.conv_dim).astype(np.float32))
        term = gb.op("Mul", [shifted, f"conv_w_{j}"], f"conv_term_{j}")
        if acc is None:
            acc = term
        else:
            acc = gb.op("Add", [acc, term], f"conv_acc_{j}")
    sig = gb.op("Sigmoid", [acc], "conv_silu_gate")
    mixed = gb.op("Mul", [acc, sig], "conv_silu")

    def take(name: str, offset: int, size: int, shape) -> str:
        start = ic(f"{name}_start", [offset])
        stop = ic(f"{name}_end", [offset + size])
        part = gb.op("Slice", [mixed, start, stop, "axis1"], f"split_{name}")
        gb.const(f"{name}_shape", np.array(shape, np.int64))
        return gb.op("Reshape", [part, f"{name}_shape"], f"reshape_{name}")

    ic("axis1", [1])
    q = take("q", 0, cfg.key_dim, [length, cfg.num_k_heads, dk])
    k = take("k", cfg.key_dim, cfg.key_dim, [length, cfg.num_k_heads, dk])
    v = take("v", 2 * cfg.key_dim, cfg.value_dim, [length, heads, dv])
    if value_scale != 1.0:
        # The delta-rule state is linear in v, so scaling v (and unscaling the
        # rule output) is exact, and it keeps the recurrence inside the FP16
        # exponent range: the untrained-from-checkpoint state reaches ~2.8e4
        # with real Qwen3.5-9B weights, close to the FP16 ceiling.
        v = gb.op("Mul", [v, fc("value_scale", value_scale)], "v_scaled")

    # --- 3. L2 normalisation, gating, GQA expansion -----------------------
    gb.const("last_axis", np.array([-1], np.int64))
    gb.const("q_scale", np.array(cfg.head_k_dim**-0.5, np.float32))
    fc("l2_eps", cfg.l2_eps)
    for name, tensor in (("q", q), ("k", k)):
        sq = gb.op("Mul", [tensor, tensor], f"l2_sq_{name}")
        ssum = gb.op("ReduceSum", [sq, "last_axis"], f"l2_sum_{name}", keepdims=1)
        ssum = gb.op("Add", [ssum, "l2_eps"], f"l2_add_{name}")
        sroot = gb.op("Sqrt", [ssum], f"l2_sqrt_{name}")
        part = gb.op("Div", [tensor, sroot], f"l2_div_{name}")
        if name == "q":
            q = gb.op("Mul", [part, "q_scale"], "l2_scale_q")
        else:
            k = part

    beta = gb.op("Sigmoid", [b], "beta_sigmoid")
    gb.const("dt_bias", w["dt_bias"].astype(np.float32).reshape(1, heads))
    gb.const("a_log_exp", np.exp(w["A_log"]).astype(np.float32).reshape(1, heads))
    a_dt = gb.op("Add", [a, "dt_bias"], "gate_a_dt")
    absx = gb.op("Abs", [a_dt], "gate_abs")
    neg_abs = gb.op("Neg", [absx], "gate_neg_abs")
    ex = gb.op("Exp", [neg_abs], "gate_exp")
    lg = gb.op("Log", [gb.op("Add", [ex, fc("gate_one", 1.0)], "gate_log1p_add")], "gate_log1p")
    relu = gb.op("Max", [a_dt, fc("gate_zero", 0.0)], "gate_relu")
    softplus = gb.op("Add", [lg, relu], "gate_softplus")
    g = gb.op("Neg", [gb.op("Mul", [softplus, "a_log_exp"], "gate_scale")], "gate_decay")

    if cfg.group > 1:
        repeat_idx = np.repeat(np.arange(cfg.num_k_heads), cfg.group).astype(np.int64)
        gb.const("gqa_index", repeat_idx)
        q = gb.op("Gather", [q, "gqa_index"], "gqa_q", axis=1)
        k = gb.op("Gather", [k, "gqa_index"], "gqa_k", axis=1)

    if front_end:
        # NPU front-end cut: input RMSNorm -> projections -> conv/SiLU -> q/k L2
        # norm -> beta/g gates, plus z for the gated norm of the CPU tail.  One
        # concatenated tensor keeps the model single-output (multi-output models
        # fall back to zeros on this runtime).
        segments = [("q", q, length * heads * dk), ("k", k, length * heads * dk),
                    ("v", v, length * heads * dv), ("beta", beta, length * heads),
                    ("g", g, length * heads), ("z", z, length * cfg.hidden)]
        parts = []
        for name, tensor, size in segments:
            gb.const(f"fe_{name}_shape", np.array([1, size], np.int64))
            parts.append(gb.op("Reshape", [tensor, f"fe_{name}_shape"], f"fe_reshape_{name}"))
        total = sum(size for _, _, size in segments)
        fe_out = gb.op("Concat", parts, "fe_out", axis=1)
        _finalize(gb, [x], [helper.make_tensor_value_info("fe_out", TensorProto.FLOAT, [1, total])], path, False)
        return

    # --- 4. chunk layout: [L, H, D] -> [H*C, T, D] ------------------------
    def chunkify(name: str, tensor: str, last: int) -> str:
        t = gb.op("Transpose", [tensor], f"{name}_to_head", perm=[1, 0, 2])
        t = gb.op("Pad", [t, "seq_pad", fc(f"{name}_pad_zero", 0.0)], f"{name}_pad")
        gb.const(f"{name}_chunk_shape", np.array([hc, step, last], np.int64))
        return gb.op("Reshape", [t, f"{name}_chunk_shape"], f"{name}_chunks")

    gb.const("seq_pad", np.array([0, 0, 0, 0, pad, 0], np.int64))
    qc = chunkify("q", q, dk)
    kc = chunkify("k", k, dk)
    vc = chunkify("v", v, dv)

    for name, tensor in (("beta", beta), ("g", g)):
        t = gb.op("Transpose", [tensor], f"{name}_to_head", perm=[1, 0])
        t = gb.op("Pad", [t, "seq_pad_2d", fc(f"{name}_pad_zero", 0.0)], f"{name}_pad")
        gb.const(f"{name}_chunk_shape", np.array([hc, step], np.int64))
        t = gb.op("Reshape", [t, f"{name}_chunk_shape"], f"{name}_chunks")
        if name == "beta":
            bc = t
        else:
            gc = t
    gb.const("seq_pad_2d", np.array([0, 0, 0, pad], np.int64))

    ic("unsq_last", [-1])
    beta_x = gb.unsqueeze(bc, [-1], "beta_unsq")
    v_beta = gb.op("Mul", [vc, beta_x], "v_beta")
    k_beta = gb.op("Mul", [kc, beta_x], "k_beta")

    cum = gb.op("CumSum", [gc, ic("cum_axis", [1])], "cum_decay", exclusive=0, reverse=0)
    cum_x = gb.unsqueeze(cum, [-1], "cum_unsq")
    cum_exp_x = gb.unsqueeze(gb.op("Exp", [cum], "cum_exp"), [-1], "cum_exp_unsq")
    gb.const("cum_last_start", np.array([step - 1], np.int64))
    gb.const("cum_last_end", np.array([step], np.int64))
    cum_last = gb.op("Slice", [cum, "cum_last_start", "cum_last_end", "axis1"], "cum_last")
    cum_last_x = gb.unsqueeze(cum_last, [-1], "cum_last_unsq")

    gb.const("mask_bias", (np.tril(np.ones((step, step), np.float32)) - 1.0) * 1e4)
    diff = gb.op("Sub", [gb.unsqueeze(cum, [-1], "pair_i"), gb.unsqueeze(cum, [-2], "pair_j")], "pair_diff")
    pwd = gb.op("Exp", [gb.op("Add", [diff, "mask_bias"], "pair_diff_masked")], "pair_decay")

    kT = gb.op("Transpose", [kc], "k_transpose", perm=[0, 2, 1])
    ut = gb.op("Mul", [gb.matmul(k_beta, kT, "ut_matmul"), pwd], "ut_system")
    intra = gb.op("Mul", [gb.matmul(qc, kT, "intra_matmul"), pwd], "intra_chunk_attn")
    decayed_k_beta = gb.op("Mul", [k_beta, cum_exp_x], "k_beta_cumdecay")

    gb.const("strict_lower_neg", -np.tril(np.ones((step, step), np.float32), -1))
    gb.const("identity_step", np.eye(step, dtype=np.float32))
    m = gb.op("Mul", [ut, "strict_lower_neg"], "nilpotent_m")
    inv = gb.op("Add", [m, "identity_step"], "nilpotent_seed")
    power = gb.matmul(m, m, "nilpotent_pow2")
    depth = int(np.log2(step))
    for i in range(depth - 1):
        factor = gb.op("Add", [power, "identity_step"], f"nilpotent_factor{i + 1}")
        inv = gb.matmul(inv, factor, f"nilpotent_acc{i + 1}")
        if i < depth - 2:
            power = gb.matmul(power, power, f"nilpotent_sq{i + 2}")

    new_values = gb.matmul(inv, v_beta, "ut_solve_v")
    k_cumdecay = gb.matmul(inv, decayed_k_beta, "ut_solve_k")

    q_dec = gb.op("Mul", [qc, cum_exp_x], "q_decayed")
    k_dec = gb.op("Mul", [kc, gb.op("Exp", [gb.op("Sub", [cum_last_x, cum_x], "k_decay_exponent")], "k_decay")], "k_decayed")
    chunk_decay = gb.op("Exp", [cum_last], "chunk_decay")

    # --- 5. chunk scan: constant masks instead of per-chunk slicing -------
    state_flat = gb.op("Reshape", [fc("state_init", np.zeros((heads, dk, dv), np.float32)), ic("state_flat_shape", [heads, dk * dv])], "state_flat")
    gb.const("tile_reps", np.array([1, chunks], np.int64))
    gb.const("state_hc_shape", np.array([hc, dk, dv], np.int64))
    gb.const("state_head_shape", np.array([heads, dk, dv], np.int64))
    gb.const("state_contrib_shape", np.array([heads, chunks, dk * dv], np.int64))
    gb.const("state_hc_flat", np.array([heads, chunks, dk * dv], np.int64))
    gb.const("chunk_axis_reduce", np.array([1], np.int64))

    state = gb.op("Reshape", [state_flat, "state_head_shape"], "state_init_3d")
    out_acc = None
    for c in range(chunks):
        tiled = gb.op("Tile", [gb.op("Reshape", [state, "state_flat_shape"], f"state_flat_{c}"), "tile_reps"], f"state_tile_{c}")
        tiled = gb.op("Reshape", [tiled, "state_hc_flat"], f"state_rep_flat_{c}")
        state_rep = gb.op("Reshape", [tiled, "state_hc_shape"], f"state_rep_{c}")

        mask = np.zeros((hc, 1, 1), np.float32)
        mask[c::chunks, 0, 0] = 1.0
        gb.const(f"chunk_mask_{c}", mask)
        mask1 = np.zeros((hc, 1), np.float32)
        mask1[c::chunks, 0] = 1.0
        gb.const(f"chunk_mask1_{c}", mask1)

        v_new = gb.op("Sub", [new_values, gb.matmul(k_cumdecay, state_rep, f"v_new_read{c}")], f"v_new_pre{c}")
        v_new = gb.op("Mul", [v_new, f"chunk_mask_{c}"], f"v_new{c}")
        inter = gb.op("Mul", [gb.matmul(q_dec, state_rep, f"inter_chunk_attn{c}"), f"chunk_mask_{c}"], f"inter_masked{c}")
        intra_out = gb.matmul(intra, v_new, f"intra_out{c}")
        o = gb.op("Add", [inter, intra_out], f"chunk_out{c}")
        out_acc = o if out_acc is None else gb.op("Add", [out_acc, o], f"chunk_out_acc{c}")

        kdT = gb.op("Transpose", [k_dec], f"kdec_t{c}", perm=[0, 2, 1])
        contrib = gb.matmul(kdT, v_new, f"state_update{c}")
        contrib = gb.op("Reshape", [contrib, "state_contrib_shape"], f"state_update_flat{c}")
        contrib = gb.op("ReduceSum", [contrib, "chunk_axis_reduce"], f"state_update_sum{c}", keepdims=1)
        contrib = gb.op("Reshape", [contrib, "state_head_shape"], f"state_update_head{c}")

        decay = gb.op("Mul", [chunk_decay, f"chunk_mask1_{c}"], f"chunk_decay_masked{c}")
        decay = gb.op("Reshape", [decay, ic(f"chunk_decay_shape_{c}", [heads, chunks, 1])], f"chunk_decay_flat{c}")
        decay = gb.op("ReduceSum", [decay, "chunk_axis_reduce"], f"chunk_decay_sum{c}", keepdims=1)
        decay = gb.op("Reshape", [decay, ic(f"chunk_decay_head_{c}", [heads, 1, 1])], f"chunk_decay_head{c}")

        state = gb.op("Add", [gb.op("Mul", [state, decay], f"state_decay{c}"), contrib], f"state_next{c}")

    # --- 6. back to sequence order, gated RMSNorm, output projection ------
    gb.const("core_hc_shape", np.array([heads, chunks * step, dv], np.int64))
    gb.const("core_seq_shape", np.array([cfg.seq_pad, heads, dv], np.int64))
    gb.const("rows_shape", np.array([length * heads, dv], np.int64))
    gb.const("value_shape", np.array([cfg.seq_pad, cfg.value_dim], np.int64))
    gb.const("keep_start", np.array([0], np.int64))
    gb.const("keep_end", np.array([length], np.int64))

    core = gb.op("Reshape", [out_acc, "core_hc_shape"], "core_hc")
    core = gb.op("Transpose", [core], "core_to_seq", perm=[1, 0, 2])
    core = gb.op("Reshape", [core, "core_seq_shape"], "core_seq")
    core = gb.op("Reshape", [core, "value_shape"], "core_value")
    core = gb.op("Slice", [core, "keep_start", "keep_end", "axis0"], "core_trim")
    if value_scale != 1.0:
        core = gb.op("Mul", [core, fc("value_scale_inv", 1.0 / value_scale)], "core_unscaled")
    core = gb.op("Reshape", [core, "rows_shape"], "core_rows")

    gsq = gb.op("Mul", [core, core], "gated_rms_sq")
    gmean = gb.op("ReduceMean", [gsq], "gated_rms_mean", axes=[-1], keepdims=1)
    gmean = gb.op("Add", [gmean, fc("gated_eps", cfg.rms_eps)], "gated_rms_add")
    groot = gb.op("Sqrt", [gmean], "gated_rms_sqrt")
    gnorm = gb.op("Div", [core, groot], "gated_rms_div")
    gb.const("norm_w", w["norm"].astype(np.float32).reshape(1, dv))
    gnorm = gb.op("Mul", [gnorm, "norm_w"], "gated_rms_scale")
    z_rows = gb.op("Reshape", [z, "rows_shape"], "z_rows")
    gated = gb.op("Mul", [gnorm, z_rows], "gated_mul")
    gated = gb.op("Mul", [gated, gb.op("Sigmoid", [z_rows], "z_gate")], "gated_silu")
    gb.const("gated_seq_shape", np.array([length, cfg.value_dim], np.int64))
    gated = gb.op("Reshape", [gated, "gated_seq_shape"], "gated_seq")
    y_out = gb.matmul(gated, "out_proj_w", "out_proj")
    gb.const("y_shape", np.array([1, length, cfg.hidden], np.int64))
    y_out = gb.op("Reshape", [y_out, "y_shape"], "y")

    final_state = gb.op("Reshape", [state, "state_head_shape"], "final_state")

    # rknn-toolkit2 2.3.2 mis-executes this graph when it has a second model
    # output (both outputs come back zero on the board), so final_state is only
    # exported on request for onnxruntime-side checks.
    outputs = [y, state_out] if (final_state_output or debug_outputs) else [y]
    if debug_outputs:
        for name in DEBUG_TENSORS:
            outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    graph = helper.make_graph(
        gb.nodes,
        "qwen35_gdn_layer0",
        [x],
        outputs,
        gb.inits,
        value_info=[],
    )
    model = helper.make_model(graph, producer_name="rockchip-npu-notes/gdn-layer", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    if debug_outputs:
        # debug outputs are declared shape-less; the checker wants shapes
        model = onnx.shape_inference.infer_shapes(model)
    else:
        onnx.checker.check_model(model)
    onnx.save(model, str(path))




def build_tail_onnx(cfg: GDNConfig, w: dict[str, np.ndarray], path: Path) -> None:
    """Tail cut: gated RMSNorm(core, z) + out_proj.

    Inputs are the flattened tensors the CPU recurrence (``core``) and the NPU
    front-end (``z``) produce, so the NPU half of a hybrid split only needs
    elementwise work plus one matmul.
    """
    gb = _Graph()
    rows = cfg.seq * cfg.value_dim
    core_in = helper.make_tensor_value_info("core", TensorProto.FLOAT, [1, rows])
    z_in = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, rows])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cfg.seq, cfg.hidden])

    gb.const("rows_shape", np.array([cfg.seq * cfg.num_v_heads, cfg.head_v_dim], np.int64))
    core = gb.op("Reshape", ["core", "rows_shape"], "core_rows")
    z = gb.op("Reshape", ["z", "rows_shape"], "z_rows")
    sq = gb.op("Mul", [core, core], "tail_rms_sq")
    mean = gb.op("ReduceMean", [sq], "tail_rms_mean", axes=[-1], keepdims=1)
    gb.const("tail_eps", np.array(cfg.rms_eps, np.float32))
    mean = gb.op("Add", [mean, "tail_eps"], "tail_rms_add")
    root = gb.op("Sqrt", [mean], "tail_rms_sqrt")
    normed = gb.op("Div", [core, root], "tail_rms_div")
    gb.const("norm_w", w["norm"].astype(np.float32).reshape(1, cfg.head_v_dim))
    normed = gb.op("Mul", [normed, "norm_w"], "tail_rms_scale")
    gate = gb.op("Sigmoid", [z], "tail_z_gate")
    gated = gb.op("Mul", [normed, z], "tail_gated_mul")
    gated = gb.op("Mul", [gated, gate], "tail_gated_silu")
    gb.const("gated_seq", np.array([cfg.seq, cfg.value_dim], np.int64))
    gated = gb.op("Reshape", [gated, "gated_seq"], "tail_gated_seq")
    gb.const("wout", np.ascontiguousarray(w["out_proj"].T))
    y = gb.matmul(gated, "wout", "tail_out_proj")
    gb.const("y_shape", np.array([1, cfg.seq, cfg.hidden], np.int64))
    gb.op("Reshape", [y, "y_shape"], "y")
    _finalize(gb, [core_in, z_in], [y_out], path, False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_weights(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in WEIGHTS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, help="npz with the 10 layer tensors (default: seeded random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=GDNConfig.hidden)
    ap.add_argument("--num-k-heads", type=int, default=GDNConfig.num_k_heads)
    ap.add_argument("--num-v-heads", type=int, default=GDNConfig.num_v_heads)
    ap.add_argument("--head-k-dim", type=int, default=GDNConfig.head_k_dim)
    ap.add_argument("--head-v-dim", type=int, default=GDNConfig.head_v_dim)
    ap.add_argument("--conv-kernel", type=int, default=GDNConfig.conv_kernel)
    ap.add_argument("--seq", type=int, default=GDNConfig.seq)
    ap.add_argument("--chunk", type=int, default=GDNConfig.chunk)
    ap.add_argument("--debug-outputs", action="store_true", help="expose intermediate tensors as model outputs")
    ap.add_argument("--final-state-output", action="store_true", help="also export the recurrent state (checker only)")
    ap.add_argument("--front-end", action="store_true", help="export only the NPU front-end cut (single concatenated output)")
    ap.add_argument("--tail", action="store_true", help="export only the gated-RMSNorm + out_proj tail (inputs core, z)")
    ap.add_argument("--value-scale", type=float, default=1.0,
                    help="scale v into the chunk rule and unscale its output (exact; keeps FP16 in range)")
    args = ap.parse_args()

    cfg = GDNConfig(
        hidden=args.hidden,
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
        conv_kernel=args.conv_kernel,
        seq=args.seq,
        chunk=args.chunk,
    )
    cfg.validate()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights(args.weights) if args.weights else random_weights(cfg, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    x = (rng.standard_normal((1, cfg.seq, cfg.hidden)) * 1.0).astype(np.float32)

    onnx_path = args.out_dir / "gdn_layer.onnx"
    if args.tail:
        build_tail_onnx(cfg, weights, onnx_path)
        x = np.zeros((1, cfg.seq, cfg.hidden), np.float32)
        ref_out, ref_state = reference_layer(cfg, weights, x)
        np.savez(args.out_dir / "gdn_reference.npz", x=x, y=ref_out, final_state=ref_state)
        (args.out_dir / "gdn_config.json").write_text(json.dumps({"config": asdict(cfg)}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote tail {onnx_path}")
        return 0
    build_onnx(
        cfg,
        weights,
        onnx_path,
        debug_outputs=args.debug_outputs,
        final_state_output=args.final_state_output,
        value_scale=args.value_scale,
        front_end=args.front_end,
    )

    ref_out, ref_state = reference_layer(cfg, weights, x)
    np.savez(
        args.out_dir / "gdn_reference.npz",
        x=x,
        y=ref_out,
        final_state=ref_state,
    )
    (args.out_dir / "gdn_config.json").write_text(
        json.dumps({"config": asdict(cfg), "derived": {
            "key_dim": cfg.key_dim,
            "value_dim": cfg.value_dim,
            "conv_dim": cfg.conv_dim,
            "group": cfg.group,
            "seq_pad": cfg.seq_pad,
            "num_chunks": cfg.num_chunks,
        }}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {onnx_path} ({len(onnx.load(str(onnx_path)).graph.node)} nodes)")
    print(f"reference out rms={np.sqrt((ref_out ** 2).mean()):.6f} state rms={np.sqrt((ref_state ** 2).mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
