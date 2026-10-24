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


def mlp_logits_declared(weights: dict[str, np.ndarray],
                        raw_images: np.ndarray) -> np.ndarray:
    x = preprocess_mnist(raw_images).astype(np.float16)
    xp = np.zeros((x.shape[0], 1024), dtype=np.float16)
    xp[:, :784] = x
    w1 = np.zeros((1024, 256), dtype=np.float16)
    w1[:784] = np.asarray(weights["W1"], dtype=np.float16)
    h = _mm_declared(xp, w1)
    b1 = np.asarray(weights["b1"], dtype=np.float16)[None, :]
    h = (h.astype(np.float32) + b1.astype(np.float32)).astype(np.float16)
    h = np.maximum(h, np.float16(0))
    h = _mm_declared(h, np.asarray(weights["W2"], dtype=np.float16))
    b2 = np.asarray(weights["b2"], dtype=np.float16)[None, :]
    h = (h.astype(np.float32) + b2.astype(np.float32)).astype(np.float16)
    h = np.maximum(h, np.float16(0))
    w3 = np.zeros((128, 16), dtype=np.float16)
    w3[:, :10] = np.asarray(weights["W3"], dtype=np.float16)
    logits = _mm_declared(h, w3)
    b3 = np.zeros((1, 16), dtype=np.float16)
    b3[0, :10] = np.asarray(weights["b3"], dtype=np.float16)
    logits = (logits.astype(np.float32) + b3.astype(np.float32)).astype(np.float16)
    return logits[:, :10].astype(np.float32)


def tiny_vit_logits_declared(weights: dict[str, np.ndarray],
                             raw_images: np.ndarray) -> np.ndarray:
    p = patchify_mnist(raw_images).astype(np.float16)
    b = p.shape[0]
    w = {k: np.asarray(v, dtype=np.float16) for k, v in weights.items()}
    pf = np.zeros((b * 16, 64), dtype=np.float16)
    pf[:, :49] = p.reshape(b * 16, 49)
    h = _mm_declared(pf, _pad_weight(w["patch_w"], 64, 64))
    h = (h.reshape(b, 16, 64).astype(np.float32) +
         w["pos"].astype(np.float32)[None]).astype(np.float16)
    z = _rmsnorm(h, w["norm1_gamma"]).astype(np.float16)
    qkv = _mm_declared(z.reshape(b * 16, 64), w["qkv_w"]).reshape(b, 16, 3, 4, 16)
    q, k, v = qkv[:, :, 0].transpose(0, 2, 1, 3), \
              qkv[:, :, 1].transpose(0, 2, 1, 3), \
              qkv[:, :, 2].transpose(0, 2, 1, 3)
    scores = _mm_batched(q, k.transpose(0, 1, 3, 2))
    sf = scores.astype(np.float32) / np.float32(4.0)
    sf -= sf.max(axis=-1, keepdims=True)
    probs = (np.exp(sf) / np.exp(sf).sum(axis=-1, keepdims=True)).astype(np.float16)
    attn_heads = _mm_batched(probs, v)
    attn = attn_heads.transpose(0, 2, 1, 3).reshape(b * 16, 64)
    h = (h.reshape(b * 16, 64).astype(np.float32) +
         _mm_declared(attn, w["attn_out_w"]).astype(np.float32)).astype(np.float16)
    z = _rmsnorm(h.reshape(b, 16, 64), w["norm2_gamma"]).astype(np.float16)
    zf = z.reshape(b * 16, 64)
    gate = _mm_declared(zf, w["ffn_gate_w"]).astype(np.float32)
    up = _mm_declared(zf, w["ffn_up_w"]).astype(np.float32)
    act = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float16)
    h = (h.astype(np.float32) +
         _mm_declared(act, w["ffn_down_w"]).astype(np.float32)).astype(np.float16)
    pooled = h.reshape(b, 16, 64).astype(np.float32).mean(axis=1).astype(np.float16)
    logits = _mm_declared(pooled, _pad_weight(w["class_w"], 64, 16))
    return logits[:, :10].astype(np.float32)


def _pad_weight(w: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float16)
    out[:w.shape[0], :w.shape[1]] = w[:rows, :cols]
    return out
