#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the frozen MLP/Tiny-ViT models to fixed-batch ONNX for RKNN.

This is a comparison artifact, not a second model implementation.  The graph
uses the same weights and the same FP32 math as ``model_reference.py``; the
vendor compiler is free to choose its own device placement/fusions.  No
quantisation is requested by the conversion step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "python"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
MEAN = 0.1307
STD = 0.3081


def _tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


class MnistMlp(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, value in weights.items():
            self.register_buffer(name, _tensor(value))

    def forward(self, images):
        x = images / 255.0
        x = (x - MEAN) / STD
        h = torch.relu(x @ self.W1 + self.b1)
        h = torch.relu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3


class TinyVit(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name, value in weights.items():
            self.register_buffer(name, _tensor(value))

    @staticmethod
    def _rmsnorm(x, gamma, eps=1e-6):
        scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x * scale * gamma

    def forward(self, images):
        b = images.shape[0]
        x = images.reshape(b, 1, 28, 28)
        x = x / 255.0
        x = (x - MEAN) / STD
        # [B,1,4,7,4,7] -> [B,1,4,4,7,7] -> [B,16,49]
        p = x.reshape(b, 1, 4, 7, 4, 7)
        p = p.permute(0, 1, 2, 4, 3, 5).reshape(b, 16, 49)
        h = p.reshape(b * 16, 49) @ self.patch_w
        h = h.reshape(b, 16, 64) + self.pos
        z = self._rmsnorm(h, self.norm1_gamma)
        qkv = z.reshape(b * 16, 64) @ self.qkv_w
        qkv = qkv.reshape(b, 16, 3, 4, 16)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        probs = torch.softmax((q @ k.transpose(-1, -2)) * 0.25, dim=-1)
        a = (probs @ v).permute(0, 2, 1, 3).reshape(b * 16, 64)
        h = h.reshape(b * 16, 64) + a @ self.attn_out_w
        z = self._rmsnorm(h.reshape(b, 16, 64), self.norm2_gamma)
        z = z.reshape(b * 16, 64)
        gate = z @ self.ffn_gate_w
        up = z @ self.ffn_up_w
        ffn = (gate * torch.sigmoid(gate) * up) @ self.ffn_down_w
        h = h + ffn
        pooled = h.reshape(b, 16, 64).mean(dim=1)
        return pooled @ self.class_w


def _onnx_reference(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return session.run(None, {name: x.astype(np.float32)})[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    ap.add_argument("--out-dir", default=str(ROOT / "vendor" / "onnx"))
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--check", action="store_true",
                    help="run ONNX Runtime on 64 deterministic test images")
    args = ap.parse_args()
    weights_dir = Path(args.weights_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    if args.profiles == "all":
        profiles = [("mnist_mlp", b) for b in (1, 8, 32)] + \
                   [("tiny_vit", b) for b in (1, 8)]
    else:
        for item in args.profiles.split(","):
            model, batch = item.split(":")
            profiles.append((model, int(batch)))
    mlp = dict(np.load(weights_dir / "mnist_mlp.npz", allow_pickle=False))
    vit = dict(np.load(weights_dir / "tiny_vit.npz", allow_pickle=False))
    x = None
    if args.check:
        import gzip, struct
        with gzip.open(ROOT / ".data" / "mnist" /
                       "t10k-images-idx3-ubyte.gz", "rb") as f:
            f.read(4)
            dims = struct.unpack(">III", f.read(12))
            x = np.frombuffer(f.read(), np.uint8).reshape(dims)
        try:
            from rknpu_backend.model_reference import (mlp_logits,
                                                      tiny_vit_logits)
        except ImportError:
            mlp_logits = tiny_vit_logits = None
    metadata = {}
    for model, batch in profiles:
        net = MnistMlp(mlp) if model == "mnist_mlp" else TinyVit(vit)
        net.eval()
        dummy = torch.zeros(batch, 784, dtype=torch.float32)
        path = out_dir / f"{model}_b{batch}.onnx"
        torch.onnx.export(
            net, dummy, str(path), input_names=["images"],
            output_names=["logits"], opset_version=17, dynamo=False,
            do_constant_folding=True)
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {"model": model, "batch": batch, "path": path.name,
                 "onnx_sha256": digest}
        if args.check:
            flat = x[:batch].reshape(batch, 784)
            y = _onnx_reference(path, flat)
            ref = (mlp_logits if model == "mnist_mlp" else tiny_vit_logits)(
                mlp if model == "mnist_mlp" else vit, x[:batch])
            entry["check_max_abs"] = float(np.max(np.abs(
                y.astype(np.float64) - ref.astype(np.float64))))
            entry["check_top1_match"] = bool(np.array_equal(
                y.argmax(1), ref.argmax(1)))
        metadata[f"{model}_b{batch}"] = entry
        print(path, flush=True)
    (out_dir / "export-manifest.json").write_text(
        json.dumps({"schema_version": 1, "profiles": metadata},
                   indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
