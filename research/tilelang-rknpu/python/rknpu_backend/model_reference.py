# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure NumPy FP32 CPU references for the two frozen model packages."""
from __future__ import annotations

from typing import Any

import numpy as np

from .models import patchify_mnist, preprocess_mnist


def _f32(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def mlp_logits(weights: dict[str, np.ndarray], raw_images: np.ndarray) -> np.ndarray:
    x = preprocess_mnist(raw_images).astype(np.float32)
    w1 = _f32(weights["W1"])
    h = np.maximum(x @ w1 + _f32(weights["b1"]), 0.0)
    h = np.maximum(h @ _f32(weights["W2"]) + _f32(weights["b2"]), 0.0)
    return h @ _f32(weights["W3"]) + _f32(weights["b3"])


def _rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    scale = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return xf * scale * gamma.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def tiny_vit_logits(weights: dict[str, np.ndarray],
                    raw_images: np.ndarray) -> np.ndarray:
    """FP32 reference with stable softmax and the exact source patch order."""
    p = patchify_mnist(raw_images).astype(np.float32)
    b = p.shape[0]
    w = weights
    h = p.reshape(b * 16, 49) @ _f32(w["patch_w"])
    h = h.reshape(b, 16, 64) + _f32(w["pos"])[None, :, :]
    z = _rmsnorm(h, w["norm1_gamma"])
    qkv = z.reshape(b * 16, 64) @ _f32(w["qkv_w"])
    qkv = qkv.reshape(b, 16, 3, 4, 16)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    probs = _softmax((q @ k.transpose(0, 1, 3, 2)) / np.float32(4.0))
    a = (probs @ v).transpose(0, 2, 1, 3).reshape(b * 16, 64)
    h = h.reshape(b * 16, 64) + a @ _f32(w["attn_out_w"])
    z = _rmsnorm(h.reshape(b, 16, 64), w["norm2_gamma"]).reshape(b * 16, 64)
    gate = z @ _f32(w["ffn_gate_w"])
    up = z @ _f32(w["ffn_up_w"])
    ffn = (gate / (1.0 + np.exp(-gate)) * up) @ _f32(w["ffn_down_w"])
    h = h + ffn
    pooled = h.reshape(b, 16, 64).mean(axis=1)
    return pooled @ _f32(w["class_w"])


def _mm_declared(a: np.ndarray, b: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Emulate ordered FP16 matmul writeback used by the declared RK path."""
    af = np.asarray(a, dtype=np.float16)
    bf = np.asarray(b, dtype=np.float16)
    k = af.shape[1]
    acc = np.zeros((af.shape[0], bf.shape[1]), dtype=np.float16)
    for k0 in range(0, k, chunk):
        k1 = min(k, k0 + chunk)
        part = (af[:, k0:k1].astype(np.float32) @
                bf[k0:k1].astype(np.float32))
        acc = (acc.astype(np.float32) + part).astype(np.float16)
    return acc


def _mm_batched(a: np.ndarray, b: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Batched ordered FP16 matmul writeback for attention heads."""
    af = np.asarray(a, dtype=np.float16)
    bf = np.asarray(b, dtype=np.float16)
    k = af.shape[-1]
    acc = np.zeros(af.shape[:-1] + (bf.shape[-1],), dtype=np.float16)
    for k0 in range(0, k, chunk):
        k1 = min(k, k0 + chunk)
        part = np.matmul(af[..., k0:k1].astype(np.float32),
                         bf[..., k0:k1, :].astype(np.float32))
        acc = (acc.astype(np.float32) + part).astype(np.float16)
    return acc


def mlp_layer_trace_declared(weights: dict[str, np.ndarray],
                             raw_images: np.ndarray) -> dict[str, np.ndarray]:
    """Declared FP16-path MLP trace using the frozen ordered-K writeback.

    Hidden trace keys document device-resident values; the hybrid graph only
    exposes ``x_padded`` and the public logits.  The separate real-TileLang ReLU
    probe supplies the observable activation-layer acceptance point.
    """
    x = preprocess_mnist(raw_images).astype(np.float16)
    xp = np.zeros((x.shape[0], 1024), dtype=np.float16)
    xp[:, :784] = x
    w1 = np.zeros((1024, 256), dtype=np.float16)
    w1[:784] = np.asarray(weights["W1"], dtype=np.float16)
    z1 = _mm_declared(xp, w1)
    b1 = np.asarray(weights["b1"], dtype=np.float16)[None, :]
    h1 = (z1.astype(np.float32) + b1.astype(np.float32)).astype(np.float16)
    relu1 = np.maximum(h1, np.float16(0))
    w2 = np.asarray(weights["W2"], dtype=np.float16)
    z2 = _mm_declared(relu1, w2)
    b2 = np.asarray(weights["b2"], dtype=np.float16)[None, :]
    h2 = (z2.astype(np.float32) + b2.astype(np.float32)).astype(np.float16)
    relu2 = np.maximum(h2, np.float16(0))
    w3 = np.zeros((128, 16), dtype=np.float16)
    w3[:, :10] = np.asarray(weights["W3"], dtype=np.float16)
    z3 = _mm_declared(relu2, w3)
    b3 = np.zeros((1, 16), dtype=np.float16)
    b3[0, :10] = np.asarray(weights["b3"], dtype=np.float16)
    logits_padded = (z3.astype(np.float32) +
                     b3.astype(np.float32)).astype(np.float16)
    logits = logits_padded[:, :10].astype(np.float32)
    return {
        "x_flat": x,
        "x_padded": xp,
        "z1": z1,
        "h1": h1,
        "relu1": relu1,
        "z2": z2,
        "h2": h2,
        "relu2": relu2,
        "z3": z3,
        "logits_padded": logits_padded,
        "logits": logits,
    }


def mlp_logits_declared(weights: dict[str, np.ndarray],
                        raw_images: np.ndarray) -> np.ndarray:
    return mlp_layer_trace_declared(weights, raw_images)["logits"]


def _mm_head_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One 2-D declared matmul per batch/head, matching the RKPlan call graph."""
    af = np.asarray(a, dtype=np.float16)
    bf = np.asarray(b, dtype=np.float16)
    out = np.empty(af.shape[:-2] + (af.shape[-2], bf.shape[-1]),
                   dtype=np.float16)
    for index in np.ndindex(af.shape[:-2]):
        out[index] = _mm_declared(af[index], bf[index])
    return out


def tiny_vit_layer_trace_declared(weights: dict[str, np.ndarray],
                                  raw_images: np.ndarray
                                  ) -> dict[str, np.ndarray]:
    """Declared Hybrid Tiny Transformer trace matching the rkhybrid nodes."""
    p = patchify_mnist(raw_images).astype(np.float16)
    b = p.shape[0]
    w = {k: np.asarray(v, dtype=np.float16) for k, v in weights.items()}
    pf = np.zeros((b * 16, 64), dtype=np.float16)
    pf[:, :49] = p.reshape(b * 16, 49)
    patch_projection = _mm_declared(
        pf, _pad_weight(w["patch_w"], 64, 64))
    positioned_tokens = (
        patch_projection.reshape(b, 16, 64).astype(np.float32) +
        w["pos"].astype(np.float32)[None]).astype(np.float16)
    positioned = positioned_tokens.reshape(b * 16, 64)
    norm1_tokens = _rmsnorm(positioned_tokens, w["norm1_gamma"]).astype(np.float16)
    norm1 = norm1_tokens.reshape(b * 16, 64)
    qkv = _mm_declared(norm1, w["qkv_w"]).reshape(b, 16, 3, 4, 16)
    q_heads = np.ascontiguousarray(qkv[:, :, 0].transpose(0, 2, 1, 3))
    k_t_heads = np.ascontiguousarray(qkv[:, :, 1].transpose(0, 2, 3, 1))
    v_heads = np.ascontiguousarray(qkv[:, :, 2].transpose(0, 2, 1, 3))
    scores_heads = _mm_head_loop(q_heads, k_t_heads)
    sf = scores_heads.astype(np.float32) * np.float32(0.25)
    sf -= sf.max(axis=-1, keepdims=True)
    e = np.exp(sf)
    probs_heads = (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)
    attn_heads = _mm_head_loop(probs_heads, v_heads)
    attn_merged = np.ascontiguousarray(
        attn_heads.transpose(0, 2, 1, 3).reshape(b * 16, 64))
    attn_projection = _mm_declared(attn_merged, w["attn_out_w"])
    h1 = (positioned.astype(np.float32) +
          attn_projection.astype(np.float32)).astype(np.float16)
    norm2 = _rmsnorm(h1.reshape(b, 16, 64),
                     w["norm2_gamma"]).astype(np.float16).reshape(b * 16, 64)
    gate = _mm_declared(norm2, w["ffn_gate_w"])
    up = _mm_declared(norm2, w["ffn_up_w"])
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    act = (g / (np.float32(1.0) + np.exp(-g)) * u).astype(np.float16)
    ffn = _mm_declared(act, w["ffn_down_w"])
    h2 = (h1.astype(np.float32) + ffn.astype(np.float32)).astype(np.float16)
    h2_tokens = h2.reshape(b, 16, 64)
    pooled = h2_tokens.astype(np.float32).mean(axis=1).astype(np.float16)
    logits_padded = _mm_declared(
        pooled, _pad_weight(w["class_w"], 64, 16))
    logits = logits_padded[:, :10].astype(np.float32)
    return {
        "patches": p,
        "patch_flat": pf,
        "patch_projection": patch_projection,
        "positioned": positioned,
        "norm1": norm1,
        "qkv": qkv.reshape(b * 16, 192),
        "q_heads": q_heads,
        "k_t_heads": k_t_heads,
        "v_heads": v_heads,
        "scores_heads": scores_heads,
        "probs_heads": probs_heads,
        "attn_heads": attn_heads,
        "attn_merged": attn_merged,
        "attn_projection": attn_projection,
        "h1": h1,
        "norm2": norm2,
        "gate": gate,
        "up": up,
        "act": act,
        "ffn": ffn,
        "h2": h2,
        "h2_tokens": h2_tokens,
        "pooled": pooled,
        "logits_padded": logits_padded,
        "logits": logits,
    }


def tiny_vit_logits_declared(weights: dict[str, np.ndarray],
                             raw_images: np.ndarray) -> np.ndarray:
    return tiny_vit_layer_trace_declared(weights, raw_images)["logits"]

def _pad_weight(w: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float16)
    out[:w.shape[0], :w.shape[1]] = w[:rows, :cols]
    return out
