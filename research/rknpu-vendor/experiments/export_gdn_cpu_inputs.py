#!/usr/bin/env python3
"""Export the GDN layer-0 tensors the C harness needs, and check its numerics.

Weights go to <out>/weights/*.bin (fp32 row-major, the layout gdn_cpu_layer.c
expects); the boundary tensors that the NPU front-end produces go to
<out>/fe/*.bin so the CPU recurrence can be validated and timed against exactly
the same values.

  python3 export_gdn_cpu_inputs.py --weights qwen35_gdn_layer0.npz \
      --inputs qwen35_gdn_inputs.npz --out /tmp/gdn_cpu [--from-board dump.bin]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gdn_layer_onnx import GDNConfig, load_weights, reference_layer  # noqa: E402

SEGMENTS = (
    ("q", lambda cfg: (cfg.seq, cfg.num_v_heads, cfg.head_k_dim)),
    ("k", lambda cfg: (cfg.seq, cfg.num_v_heads, cfg.head_k_dim)),
    ("v", lambda cfg: (cfg.seq, cfg.num_v_heads, cfg.head_v_dim)),
    ("beta", lambda cfg: (cfg.seq, cfg.num_v_heads)),
    ("g", lambda cfg: (cfg.seq, cfg.num_v_heads)),
    ("z", lambda cfg: (cfg.seq, cfg.hidden)),
)


def front_end_reference(cfg: GDNConfig, w: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    """Recompute the front-end boundary tensors from the numpy oracle path."""
    length = x.shape[1]
    h = x.reshape(length, cfg.hidden).astype(np.float32)
    var = (h * h).mean(-1, keepdims=True)
    h = (h / np.sqrt(var + cfg.rms_eps)) * w["input_layernorm"][None, :]
    mixed = h @ w["in_proj_qkv"].T
    z = h @ w["in_proj_z"].T
    b = h @ w["in_proj_b"].T
    a = h @ w["in_proj_a"].T
    # conv + silu
    padded = np.concatenate([np.zeros((cfg.conv_kernel - 1, cfg.conv_dim), np.float32), mixed], axis=0)
    acc = np.zeros_like(mixed)
    for j in range(cfg.conv_kernel):
        acc += padded[j : j + length] * w["conv1d"][:, 0, j][None, :]
    m = acc / (1.0 + np.exp(-acc))
    q = m[:, : cfg.key_dim].reshape(length, cfg.num_k_heads, cfg.head_k_dim)
    k = m[:, cfg.key_dim : 2 * cfg.key_dim].reshape(length, cfg.num_k_heads, cfg.head_k_dim)
    v = m[:, 2 * cfg.key_dim :].reshape(length, cfg.num_v_heads, cfg.head_v_dim)
    q = (q / np.sqrt((q * q).sum(-1, keepdims=True) + cfg.l2_eps)) * (cfg.head_k_dim**-0.5)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + cfg.l2_eps)
    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(w["A_log"])[None, :] * (np.log1p(np.exp(-np.abs(a + w["dt_bias"][None, :]))) +
                                        np.maximum(a + w["dt_bias"][None, :], 0.0))
    q = np.repeat(q, cfg.group, axis=1)
    k = np.repeat(k, cfg.group, axis=1)
    return {"q": q, "k": k, "v": v, "beta": beta, "g": g, "z": z}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-board", type=Path, help="device dump of the front-end output (concatenated)")
    args = ap.parse_args()

    cfg = GDNConfig()
    w = load_weights(args.weights)
    x = np.load(args.inputs)["x"][0:1].astype(np.float32)

    wdir = args.out / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    for name, dst in (
        ("input_layernorm", "ln.bin"),
        ("in_proj_qkv", "wqkv.bin"),
        ("in_proj_z", "wz.bin"),
        ("in_proj_b", "wb.bin"),
        ("in_proj_a", "wa.bin"),
        ("norm", "norm.bin"),
        ("out_proj", "wout.bin"),
        ("dt_bias", "dtb.bin"),
        ("A_log", "alog.bin"),
    ):
        w[name].astype(np.float32).tofile(wdir / dst)
    w["conv1d"][:, 0, :].astype(np.float32).tofile(wdir / "conv.bin")

    ref = front_end_reference(cfg, w, x)
    fedir = args.out / "fe"
    fedir.mkdir(parents=True, exist_ok=True)
    for name, shape_fn in SEGMENTS:
        tensor = ref[name]
        if args.from_board:
            dump = np.fromfile(args.from_board, dtype=np.float32)
            sizes = [int(np.prod(shape_fn(cfg))) for name2, shape_fn in SEGMENTS]
            offset = 0
            for (name2, shape_fn2), size in zip(SEGMENTS, sizes):
                if name2 == name:
                    tensor = dump[offset : offset + size].reshape(shape_fn2(cfg))
                    break
                offset += size
        tensor.astype(np.float32).tofile(fedir / f"{name}.bin")

    x[0].astype(np.float32).tofile(args.out / "x.bin")
    y_ref, _ = reference_layer(cfg, w, x)
    y_ref.astype(np.float32).tofile(args.out / "y_ref.bin")
    # the tensor that feeds gated RMSNorm: core rows = [L*HEADS, DV]
    np.save(args.out / "meta.npy", np.array([cfg.seq, cfg.num_v_heads, cfg.head_v_dim]))
    print(f"wrote {wdir}, {fedir}, x.bin, y_ref.bin under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
