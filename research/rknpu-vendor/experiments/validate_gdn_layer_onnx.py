#!/usr/bin/env python3
"""Check a built GDN ONNX graph against the numpy oracle and, when torch is
available, against the reference HuggingFace ``Qwen3_5GatedDeltaNet`` module.

Usage:
  python3 validate_gdn_layer_onnx.py --dir <build dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--torch-weights", type=Path, help="npz with the same tensors, for HF cross-check")
    args = ap.parse_args()

    import onnxruntime as ort

    ref = np.load(args.dir / "gdn_reference.npz")
    cfg = json.loads((args.dir / "gdn_config.json").read_text())["config"]

    sess = ort.InferenceSession(str(args.dir / "gdn_layer.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": ref["x"]})
    y_onnx = out[0]
    entry = {
        "out_corr": corr(y_onnx, ref["y"]),
        "out_max_abs": float(np.abs(y_onnx - ref["y"]).max()),
    }
    if len(out) > 1:
        state_onnx = out[1]
        entry["state_corr"] = corr(state_onnx, ref["final_state"])
        entry["state_max_abs"] = float(np.abs(state_onnx - ref["final_state"]).max())
    report = {"onnx_vs_numpy": entry}

    if args.torch_weights is not None:
        hf = hf_reference(args.dir, cfg, np.load(args.torch_weights))
        report["hf_vs_numpy"] = {
            "out_corr": corr(hf["y"], ref["y"]),
            "out_max_abs": float(np.abs(hf["y"] - ref["y"]).max()),
            "state_corr": corr(hf["final_state"], ref["final_state"]),
            "state_max_abs": float(np.abs(hf["final_state"] - ref["final_state"]).max()),
        }
        report["hf_vs_onnx"] = {
            "out_corr": corr(hf["y"], y_onnx),
            "out_max_abs": float(np.abs(hf["y"] - y_onnx).max()),
        }

    print(json.dumps(report, indent=2))
    return 0


def hf_reference(build_dir: Path, cfg: dict, w: dict) -> dict:
    """Run the reference HuggingFace Qwen3.5 GDN module on the same weights.

    ``Qwen3_5GatedDeltaNet`` does not contain the decoder pre-norm, so the
    input is normalised here exactly like the ONNX graph does internally.
    """
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    text_cfg = Qwen3_5TextConfig(
        hidden_size=cfg["hidden"],
        linear_num_key_heads=cfg["num_k_heads"],
        linear_num_value_heads=cfg["num_v_heads"],
        linear_key_head_dim=cfg["head_k_dim"],
        linear_value_head_dim=cfg["head_v_dim"],
        linear_conv_kernel_dim=cfg["conv_kernel"],
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        rms_norm_eps=cfg["rms_eps"],
    )
    layer = Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).eval()
    with torch.no_grad():
        layer.in_proj_qkv.weight.copy_(torch.from_numpy(w["in_proj_qkv"]))
        layer.in_proj_z.weight.copy_(torch.from_numpy(w["in_proj_z"]))
        layer.in_proj_b.weight.copy_(torch.from_numpy(w["in_proj_b"]))
        layer.in_proj_a.weight.copy_(torch.from_numpy(w["in_proj_a"]))
        layer.conv1d.weight.copy_(torch.from_numpy(w["conv1d"]))
        layer.dt_bias.copy_(torch.from_numpy(w["dt_bias"]))
        layer.A_log.copy_(torch.from_numpy(w["A_log"]))
        layer.norm.weight.copy_(torch.from_numpy(w["norm"]))
        layer.out_proj.weight.copy_(torch.from_numpy(w["out_proj"]))
        x = torch.from_numpy(np.load(build_dir / "gdn_reference.npz")["x"])
        var = x.pow(2).mean(-1, keepdim=True)
        h = x * torch.rsqrt(var + cfg["rms_eps"]) * torch.from_numpy(w["input_layernorm"])
        y = layer(h, cache_params=None)
    return {"y": y.numpy(), "final_state": np.zeros((cfg["num_v_heads"], cfg["head_k_dim"], cfg["head_v_dim"]))}


if __name__ == "__main__":
    raise SystemExit(main())
