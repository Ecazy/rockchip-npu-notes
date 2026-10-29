#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent declared-reference implementation of SmolLM2-135M-Instruct.

This is a Host oracle for the backend numeric contract, not a production path.
It follows the explicit Kernel Series boundary:

  * weights and activations are FP16;
  * each logical linear operation is split into 256-wide K tiles and the
    accumulator is written back to FP16 after every K tile;
  * RMSNorm keeps the reduction/rsqrt/gamma product in FP32 and narrows once;
  * RoPE uses FP32 frequencies/cos/sin and narrows the final rotation;
  * attention QK/PV are the same FP16 tiled products;
  * causal softmax is FP32 stable softmax narrowed to FP16;
  * residual adds and SwiGLU narrow once to FP16.

No backend planner, runtime library, TileLang node or RKPlan is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

F16 = np.dtype("<f2")
K_TILE = 256


def f16(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mm_tiled(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FP16 A[M,K] x B[K,N] with FP16 writeback after each 256-wide K tile."""
    if a.dtype != F16 or b.dtype != F16 or a.ndim != 2 or b.ndim != 2 or \
            a.shape[1] != b.shape[0]:
        raise ValueError("mm_tiled expects FP16 [M,K] x [K,N]")
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
    for k0 in range(0, a.shape[1], K_TILE):
        kk = min(K_TILE, a.shape[1] - k0)
        part = a[:, k0:k0 + kk].astype(np.float32) @ \
               b[k0:k0 + kk, :].astype(np.float32)
        out = f16(out.astype(np.float32) + part)
    return out


def add16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return f16(a.astype(np.float32) + b.astype(np.float32))


def rmsnorm16(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    scale = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    return f16(xf * scale * gamma.astype(np.float32))


def rope_half(x: np.ndarray, positions: np.ndarray,
              theta: float) -> np.ndarray:
    dim = x.shape[-1]
    inv = np.power(np.float32(theta),
                   -np.arange(0, dim, 2, dtype=np.float32) /
                   np.float32(dim))
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb).astype(np.float32)[:, None, :]
    sin = np.sin(emb).astype(np.float32)[:, None, :]
    xf = x.astype(np.float32)
    half = dim // 2
    rot = np.concatenate((-xf[..., half:], xf[..., :half]), axis=-1)
    return f16(xf * cos + rot * sin)


def causal_softmax16(scores: np.ndarray, valid: np.ndarray,
                     scale: float) -> np.ndarray:
    rows, cols = scores.shape
    out = np.zeros((rows, cols), dtype=np.float16)
    idx = np.arange(cols, dtype=np.int32)[None, :]
    mask = idx < valid[:, None]
    sf = scores.astype(np.float32) * np.float32(scale)
    sf = np.where(mask, sf, np.float32(-np.inf))
    mx = np.max(sf, axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, np.float32(0.0))
    with np.errstate(invalid="ignore", over="ignore"):
        e = np.exp(sf - mx) * mask.astype(np.float32)
    den = np.sum(e, axis=1, keepdims=True)
    probs = np.zeros_like(e, dtype=np.float32)
    np.divide(e, den, out=probs, where=den > 0)
    return f16(probs)


class DeclaredReference:
    def __init__(self, package: Path):
        self.package = package
        manifest = json.loads((package / "manifest.json").read_text())
        self.manifest = manifest
        self.cfg = manifest["config"]
        self.capacity = int(manifest["capacity"])
        self.shards = int(manifest["lm_head_shards"])
        self.weights = np.load(package / manifest["weights"]["path"],
                               allow_pickle=False)
        if sha256_file(package / manifest["weights"]["path"]) != \
                manifest["weights"]["file_sha256"]:
            raise SystemExit("weight package file hash mismatch")

    def weight_t(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(self.weights[name].T)

    def forward(self, token_ids: list[int], *,
                capture_layers: tuple[int, ...] = ()) -> dict[str, Any]:
        c = self.cfg
        hsz = c["hidden_size"]
        nh = c["num_attention_heads"]
        nk = c["num_key_value_heads"]
        hd = c["head_dim"]
        inter = c["intermediate_size"]
        cap = self.capacity
        per = nh // nk
        ids = np.asarray(token_ids, dtype=np.int32)
        tokens = int(ids.size)
        x = f16(self.weights["embed"][ids])
        captured: dict[str, np.ndarray] = {}
        k_cache = np.zeros((nk, cap, hd), dtype=np.float16)
        v_cache = np.zeros_like(k_cache)
        positions = np.arange(tokens, dtype=np.int32)
        for layer in range(c["num_hidden_layers"]):
            capture = layer in capture_layers
            h = rmsnorm16(x, self.weights[f"layer.{layer}.input_norm"],
                          c["rms_norm_eps"])
            qkv = mm_tiled(h, self.weight_t(f"layer.{layer}.qkv"))
            q = qkv[:, :nh * hd].reshape(tokens, nh, hd)
            k = qkv[:, nh * hd:nh * hd + nk * hd].reshape(tokens, nk, hd)
            v = qkv[:, nh * hd + nk * hd:].reshape(tokens, nk, hd)
            q = rope_half(q, positions, c["rope_theta"])
            k = rope_half(k, positions, c["rope_theta"])
            k_cache[:, :tokens, :] = k.transpose(1, 0, 2)
            v_cache[:, :tokens, :] = v.transpose(1, 0, 2)
            ctx = np.empty((nk, per * tokens, hd), dtype=np.float16)
            for g in range(nk):
                a = q[:, g * per:(g + 1) * per].transpose(1, 0, 2).reshape(
                    per * tokens, hd)
                kt = f16(k_cache[g].T)
                scores = mm_tiled(np.ascontiguousarray(a), kt)
                valid = np.repeat(
                    np.arange(1, tokens + 1, dtype=np.int32)[None, :],
                    per, axis=0).reshape(-1)
                probs = causal_softmax16(
                    scores, valid, 1.0 / math.sqrt(float(hd)))
                got = mm_tiled(probs, v_cache[g])
                ctx[g] = got
            merged = ctx.reshape(nk, per, tokens, hd).transpose(
                2, 0, 1, 3).reshape(tokens, nh * hd)
            o = mm_tiled(merged, self.weight_t(f"layer.{layer}.o"))
            x_attn = add16(x, o)
            h2 = rmsnorm16(x_attn,
                           self.weights[f"layer.{layer}.post_norm"],
                           c["rms_norm_eps"])
            gate_up = mm_tiled(h2,
                               self.weight_t(f"layer.{layer}.gate_up"))
            gate = gate_up[:, :inter]
            up = gate_up[:, inter:]
            act = f16(gate.astype(np.float32) *
                      (np.float32(1.0) / (np.float32(1.0) +
                       np.exp(-gate.astype(np.float32)))) *
                      up.astype(np.float32))
            down = mm_tiled(act, self.weight_t(f"layer.{layer}.down"))
            x = add16(x_attn, down)
            if capture:
                p = f"layer.{layer}."
                captured.update({
                    p + "input_norm": h.copy(),
                    p + "qkv": qkv.copy(),
                    p + "q_rope": q.copy(),
                    p + "k_rope": k.copy(),
                    p + "v": v.copy(),
                    p + "attn_context": merged.copy(),
                    p + "o_proj": o.copy(),
                    p + "resid_attn": x_attn.copy(),
                    p + "post_norm": h2.copy(),
                    p + "gate_up": gate_up.copy(),
                    p + "swiglu": act.copy(),
                    p + "down_proj": down.copy(),
                    p + "resid_ffn": x.copy(),
                })
        normed = rmsnorm16(x, self.weights["norm"], c["rms_norm_eps"])
        if capture_layers:
            captured["final_norm"] = normed.copy()
        vocab = c["vocab_size"]
        logits = np.empty((tokens, vocab), dtype=np.float16)
        width = (vocab + self.shards - 1) // self.shards
        wt = np.ascontiguousarray(self.weights["embed"].T)
        shard_hashes = []
        for j in range(self.shards):
            start, stop = j * width, min(vocab, (j + 1) * width)
            shard = mm_tiled(normed, wt[:, start:stop])
            logits[:, start:stop] = shard
            shard_hashes.append(hashlib.sha256(shard.tobytes()).hexdigest())
        return {
            "token_ids": ids.tolist(),
            "tokens": tokens,
            "logits": logits,
            "captured": captured,
            "lm_head_shard_sha256": shard_hashes,
            "declared_reference": {
                "dtype": "float16",
                "k_tile": K_TILE,
                "writeback": "float16_after_each_k_tile",
                "softmax": "float32_stable_then_float16",
                "rmsnorm": "float32_reduction_then_float16",
                "rope": "float32_cos_sin_then_float16",
            },
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--ids-json", required=True,
                    help="JSON object with token_ids, or a bare array")
    ap.add_argument("--capture-layers", default="0,15,29")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    raw = json.loads(Path(args.ids_json).read_text())
    ids = raw["token_ids"] if isinstance(raw, dict) else raw
    layers = tuple(int(v) for v in args.capture_layers.split(",") if v)
    ref = DeclaredReference(Path(args.package))
    result = ref.forward(ids, capture_layers=layers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "logits.npy", result["logits"])
    if result["captured"]:
        np.savez(out / "captured.npz", **result["captured"])
    metadata = {
        "kind": "smollm2-declared-reference",
        "package_manifest_sha256": sha256_file(
            Path(args.package) / "manifest.json"),
        "weights_file_sha256": sha256_file(
            Path(args.package) / ref.manifest["weights"]["path"]),
        "token_ids": result["token_ids"],
        "tokens": result["tokens"],
        "capture_layers": list(layers),
        "logits_shape": list(result["logits"].shape),
        "logits_sha256": hashlib.sha256(
            result["logits"].tobytes()).hexdigest(),
        "lm_head_shard_sha256": result["lm_head_shard_sha256"],
        "declared_reference": result["declared_reference"],
    }
    (out / "meta.json").write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"tokens": result["tokens"],
                      "logits": list(result["logits"].shape),
                      "top1_last": int(np.argmax(result["logits"][-1])),
                      "captures": sorted(result["captured"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
