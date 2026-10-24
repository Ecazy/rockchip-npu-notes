#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train and freeze the two milestone models with a fixed seed.

This is the Host training/reference step.  It reads the official MNIST idx files,
trains FP32 CPU models, evaluates the full test split, and writes safe ``.npz``
weights plus a manifest.  No device is opened here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".data" / "mnist"
MODELS = ROOT / "models"
EVIDENCE = ROOT / "evidence"
SEED = 20261024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(path: Path, expected_rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        rank = magic & 0xFF
        if rank != expected_rank:
            raise ValueError(f"{path}: rank {rank} != {expected_rank}")
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path}: payload size mismatch")
    return data.reshape(dims)


def load_mnist():
    train_x = read_idx(DATA / "train-images-idx3-ubyte.gz", 3).astype(np.float32)
    train_y = read_idx(DATA / "train-labels-idx1-ubyte.gz", 1).astype(np.int64)
    test_x = read_idx(DATA / "t10k-images-idx3-ubyte.gz", 3).astype(np.float32)
    test_y = read_idx(DATA / "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
    train_x = train_x[:, None, :, :] / 255.0
    test_x = test_x[:, None, :, :] / 255.0
    mean, std = 0.1307, 0.3081
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, train_y, test_x, test_y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TinyViT(nn.Module):
    """16 patch tokens, D=64, 4 heads, one pre-RMSNorm encoder layer."""

    def __init__(self, dim=64, heads=4, mlp=128, classes=10, tokens=16,
                 patch=7, image=28):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.tokens = tokens
        self.patch = patch
        self.image = image
        self.patch_w = nn.Parameter(torch.empty(patch * patch, dim))
        self.pos = nn.Parameter(torch.empty(tokens, dim))
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.gate = nn.Linear(dim, mlp, bias=False)
        self.up = nn.Linear(dim, mlp, bias=False)
        self.down = nn.Linear(mlp, dim, bias=False)
        self.classifier = nn.Linear(dim, classes, bias=False)
        self.eps = 1e-6
        nn.init.normal_(self.patch_w, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    @staticmethod
    def rmsnorm(x, gamma, eps):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma

    def forward(self, x):
        b = x.shape[0]
        patches = F.unfold(x, kernel_size=self.patch, stride=self.patch)
        patches = patches.transpose(1, 2)  # [B, tokens, patch*patch]
        h = patches @ self.patch_w + self.pos
        z = self.rmsnorm(h, self.norm1, self.eps)
        qkv = self.qkv(z)
        q, k, v = qkv.split(self.dim, dim=-1)
        q = q.reshape(b, self.tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, self.tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, self.tokens, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        attn = (probs @ v).transpose(1, 2).reshape(b, self.tokens, self.dim)
        h = h + self.attn_out(attn)
        z = self.rmsnorm(h, self.norm2, self.eps)
        ffn = (F.silu(self.gate(z)) * self.up(z)) @ self.down.weight.T
        h = h + ffn
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


def train_epoch(model, opt, x, y, batch_size):
    model.train()
    perm = torch.randperm(x.shape[0])
    total = correct = 0
    loss_sum = 0.0
    for start in range(0, x.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        xb = torch.from_numpy(x[idx]).float()
        yb = torch.from_numpy(y[idx])
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        loss_sum += float(loss) * xb.shape[0]
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum())
        total += xb.shape[0]
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, x, y, batch_size=512, return_logits=False):
    model.eval()
    correct = 0
    logits_all = []
    nll_sum = 0.0
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).float()
        yb = torch.from_numpy(y[start:start + batch_size])
        logits = model(xb)
        nll_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(dim=1) == yb).sum())
        if return_logits:
            logits_all.append(logits.cpu().numpy().astype(np.float32))
    logits_np = np.concatenate(logits_all) if return_logits else None
    return correct / y.shape[0], nll_sum / y.shape[0], logits_np


def save_manifest(entries):
    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "dataset": {
            "name": "MNIST",
            "source": "https://storage.googleapis.com/cvdf-datasets/mnist/",
            "files": {
                p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(DATA.glob("*.gz"))
            },
            "train_count": 60000,
            "test_count": 10000,
            "preprocessing": {"normalize_mean": 0.1307,
                              "normalize_std": 0.3081},
        },
        "models": entries,
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    MODELS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    train_x, train_y, test_x, test_y = load_mnist()
    entries = []

    # MLP: enough epochs for a stable >97% FP32 baseline.
    mlp = MLP()
    opt = torch.optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    for epoch in range(12):
        loss, acc = train_epoch(mlp, opt, train_x, train_y, 128)
        sched.step()
        print(f"MLP epoch {epoch+1:02d}: loss={loss:.5f} train_acc={acc:.4f}", flush=True)
    mlp_acc, mlp_nll, mlp_logits = evaluate(mlp, test_x, test_y, return_logits=True)
    np.savez(
        MODELS / "mnist_mlp.npz",
        W1=mlp.fc1.weight.detach().numpy().T.copy(),
        b1=mlp.fc1.bias.detach().numpy().copy(),
        W2=mlp.fc2.weight.detach().numpy().T.copy(),
        b2=mlp.fc2.bias.detach().numpy().copy(),
        W3=mlp.fc3.weight.detach().numpy().T.copy(),
        b3=mlp.fc3.bias.detach().numpy().copy(),
    )
    np.save(EVIDENCE / "mnist_mlp_cpu_test_logits.npy", mlp_logits)
    np.save(EVIDENCE / "mnist_test_labels.npy", test_y)
    entries.append({
        "name": "mnist_mlp", "file": "models/mnist_mlp.npz",
        "architecture": "Flatten(784)->Linear(784,256)->ReLU->Linear(256,128)->ReLU->Linear(128,10)",
        "input": "[N,1,28,28] float32 normalized",
        "output": "[N,10] logits; argmax application postprocess",
        "weights": ["W1[784,256]", "b1[256]", "W2[256,128]", "b2[128]",
                    "W3[128,10]", "b3[10]"],
        "train_epochs": 12, "optimizer": "AdamW", "lr": 1.5e-3,
        "batch_size": 128, "test_accuracy": float(mlp_acc),
        "test_nll": float(mlp_nll),
        "weights_sha256": sha256(MODELS / "mnist_mlp.npz"),
    })
    print(f"MLP test accuracy={mlp_acc:.4f} nll={mlp_nll:.5f}", flush=True)

    # Tiny ViT: 16 patch tokens, D=64, 4 heads, one encoder, SwiGLU FFN=128.
    vit = TinyViT()
    opt = torch.optim.AdamW(vit.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for epoch in range(30):
        loss, acc = train_epoch(vit, opt, train_x, train_y, 128)
        sched.step()
        print(f"ViT epoch {epoch+1:02d}: loss={loss:.5f} train_acc={acc:.4f}", flush=True)
    vit_acc, vit_nll, vit_logits = evaluate(vit, test_x, test_y, return_logits=True)
    np.savez(
        MODELS / "tiny_vit.npz",
        patch_w=vit.patch_w.detach().numpy().copy(),
        pos=vit.pos.detach().numpy().copy(),
        norm1_gamma=vit.norm1.detach().numpy().copy(),
        qkv_w=vit.qkv.weight.detach().numpy().T.copy(),
        attn_out_w=vit.attn_out.weight.detach().numpy().T.copy(),
        norm2_gamma=vit.norm2.detach().numpy().copy(),
        ffn_gate_w=vit.gate.weight.detach().numpy().T.copy(),
        ffn_up_w=vit.up.weight.detach().numpy().T.copy(),
        ffn_down_w=vit.down.weight.detach().numpy().T.copy(),
        class_w=vit.classifier.weight.detach().numpy().T.copy(),
    )
    np.save(EVIDENCE / "tiny_vit_cpu_test_logits.npy", vit_logits)
    entries.append({
        "name": "tiny_vit", "file": "models/tiny_vit.npz",
        "architecture": "16x7x7 patch -> D64; 1 pre-RMSNorm attention layer (4 heads); pre-RMSNorm SwiGLU FFN(128); mean pool; linear 10",
        "input": "[N,1,28,28] float32 normalized",
        "output": "[N,10] logits; mean-pooled 16 tokens",
        "weights": ["patch_w[49,64]", "pos[16,64]", "norm1_gamma[64]",
                    "qkv_w[64,192]", "attn_out_w[64,64]", "norm2_gamma[64]",
                    "ffn_gate_w[64,128]", "ffn_up_w[64,128]", "ffn_down_w[128,64]",
                    "class_w[64,10]"],
        "train_epochs": 30, "optimizer": "AdamW", "lr": 3e-3,
        "batch_size": 128, "test_accuracy": float(vit_acc),
        "test_nll": float(vit_nll), "eps": 1e-6,
        "weights_sha256": sha256(MODELS / "tiny_vit.npz"),
    })
    print(f"ViT test accuracy={vit_acc:.4f} nll={vit_nll:.5f}", flush=True)
    save_manifest(entries)
    print(json.dumps({"mlp_accuracy": mlp_acc, "vit_accuracy": vit_acc}, indent=2))


if __name__ == "__main__":
    main()
