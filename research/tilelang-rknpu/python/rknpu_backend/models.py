# SPDX-License-Identifier: GPL-3.0-or-later
"""Model-level runners for the frozen MNIST MLP and hybrid Tiny Transformer.

The runners only orchestrate already-compiled RKPlan files (or compile them from
real TileLang sources on the Host).  All matrix computation is delegated to the
production ``rkplan`` C ABI; the Transformer's first-round RMSNorm, stable
softmax, layout bridge and mean pooling are explicit Host work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RK_E_INVALID, RK_E_SHAPE, RK_E_UNSUPPORTED, reject
from .plan import Plan
from .runtime.ffi import CWorkspace, LibRKPlan, make_binding

MNIST_MEAN = np.float32(0.1307)
MNIST_STD = np.float32(0.3081)


def _f16(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16)


def preprocess_mnist(raw: np.ndarray) -> np.ndarray:
    """Convert uint8/float raw ``[N,1,28,28]`` or ``[N,784]`` to normalized f32."""
    a = np.asarray(raw)
    if a.ndim == 3:
        if a.shape[1:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,28,28]", stage="preprocess",
                   actual=list(a.shape))
        a = a.reshape(a.shape[0], 784)
    elif a.ndim == 4:
        if a.shape[1] != 1 or a.shape[2:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,1,28,28]", stage="preprocess",
                   actual=list(a.shape))
        a = a[:, 0, :, :]
    if a.ndim != 2 or a.shape[1] != 784:
        reject(RK_E_SHAPE, "MNIST images must flatten to [N,784]", stage="preprocess",
               actual=list(a.shape))
    af = a.astype(np.float32)
    if af.size and float(np.max(af)) > 1.5:
        af /= 255.0
    return (af - MNIST_MEAN) / MNIST_STD


def patchify_mnist(raw: np.ndarray) -> np.ndarray:
    """Return normalized ``[N,16,49]`` 7x7 patches in raster token order."""
    a = np.asarray(raw)
    if a.ndim == 2 and a.shape[1] == 784:
        a = a.reshape(-1, 1, 28, 28)
    elif a.ndim == 3 and a.shape[1:] == (28, 28):
        a = a[:, None, :, :]
    if a.ndim != 4 or a.shape[1] != 1 or a.shape[2:] != (28, 28):
        reject(RK_E_SHAPE, "MNIST images must be [N,1,28,28]", stage="preprocess",
               actual=list(a.shape))
    af = a.astype(np.float32)
    if af.size and float(np.max(af)) > 1.5:
        af /= 255.0
    af = (af - MNIST_MEAN) / MNIST_STD
    # [N,1,4,7,4,7] -> [N,1,4,4,7,7] -> [N,16,49]
    p = af.reshape(a.shape[0], 1, 4, 7, 4, 7).transpose(0, 1, 2, 4, 3, 5)
    return p.reshape(a.shape[0], 16, 49).astype(np.float32)


@dataclass
class RunStats:
    elapsed_us: float = 0.0
    host_compute_us: float = 0.0
    upload_calls: int = 0
    upload_bytes: int = 0
    download_calls: int = 0
    download_bytes: int = 0
    task_count: int = 0
    submit_count: int = 0
    plans: list[dict[str, Any]] = field(default_factory=list)

    def add_report(self, name: str, report: Any, elapsed_us: float) -> None:
        self.elapsed_us += elapsed_us
        self.upload_calls += int(report.upload_calls)
        self.upload_bytes += int(report.upload_bytes)
        self.download_calls += int(report.download_calls)
        self.download_bytes += int(report.download_bytes)
        self.task_count += int(report.task_count)
        self.submit_count += int(report.submit_count)
        # Aggregate repeated calls so a full test-set evaluation has a bounded
        # report (one row per logical plan) rather than one row per batch.
        row = next((r for r in self.plans if r["name"] == name), None)
        if row is None:
            row = {"name": name, "calls": 0, "elapsed_us": 0.0,
                   "pack_us": 0.0, "emit_us": 0.0,
                   "submit_wait_us": 0.0, "output_sync_us": 0.0,
                   "upload_calls": 0, "upload_bytes": 0,
                   "download_calls": 0, "download_bytes": 0,
                   "task_count": 0, "submit_count": 0}
            self.plans.append(row)
        row["calls"] += 1
        row["elapsed_us"] += elapsed_us
        row["pack_us"] += float(report.pack_us)
        row["emit_us"] += float(report.emit_us)
        row["submit_wait_us"] += float(report.submit_wait_us)
        row["output_sync_us"] += float(report.output_sync_us)
        row["upload_calls"] += int(report.upload_calls)
        row["upload_bytes"] += int(report.upload_bytes)
        row["download_calls"] += int(report.download_calls)
        row["download_bytes"] += int(report.download_bytes)
        row["task_count"] += int(report.task_count)
        row["submit_count"] += int(report.submit_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_us": self.elapsed_us,
            "host_compute_us": self.host_compute_us,
            "upload_calls": self.upload_calls,
            "upload_bytes": self.upload_bytes,
            "download_calls": self.download_calls,
            "download_bytes": self.download_bytes,
            "task_count": self.task_count,
            "submit_count": self.submit_count,
            "plans": list(self.plans),
        }


class PlanCall:
    """Reusable workspace for one immutable plan and its constants."""

    def __init__(self, lib: LibRKPlan, plan: Plan, constants: dict[str, np.ndarray],
                 *, fd: int = 0):
        self.lib = lib
        self.plan = plan
        self.cplan = lib.create_plan(plan)
        self._const_arrays = []
        bindings = []
        for spec in plan.constants:
            if spec.name not in constants:
                raise KeyError(f"missing constant {spec.name!r} for {plan.name}")
            arr = _f16(constants[spec.name])
            expected = tuple(spec.shape)
            if arr.shape != expected:
                reject(RK_E_SHAPE, f"constant {spec.name!r} shape mismatch",
                       stage="model", actual=list(arr.shape), required=list(expected))
            self._const_arrays.append(arr)
            bindings.append(make_binding(spec.binding_id, arr))
        self.workspace: CWorkspace = lib.create_workspace(
            self.cplan, bindings, fd=fd)
        self.inputs = [b for b in plan.bindings if b.role == "input"]
        self.outputs = [b for b in plan.bindings if b.role == "output"]

    def run(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], Any, float]:
        if len(inputs) != len(self.inputs):
            reject(RK_E_SHAPE, "model plan input count mismatch", stage="model",
                   actual=len(inputs), required=len(self.inputs))
        in_bind = []
        keep_in = []
        for spec, arr in zip(self.inputs, inputs):
            a = _f16(arr)
            if a.shape != tuple(spec.shape):
                reject(RK_E_SHAPE, "model plan input shape mismatch", stage="model",
                       actual=list(a.shape), required=list(spec.shape))
            keep_in.append(a)
            in_bind.append(make_binding(spec.id, a))
        out_arrays = [np.empty(tuple(b.shape), dtype=np.float16)
                      for b in self.outputs]
        out_bind = [make_binding(b.id, a)
                    for b, a in zip(self.outputs, out_arrays)]
        # The mock harness snapshots each emitted co_program.  Clear that
        # test-only cache between calls; production builds have no such symbol.
        if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
            self.lib.mock_control("rk_mock_reset_tags")
        t0 = time.perf_counter()
        report = self.workspace.run(in_bind, out_bind)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return out_arrays, report, elapsed_us

    def close(self) -> None:
        self.workspace.destroy()
        self.cplan.destroy()


def _generated_constant(spec, weights: dict[str, np.ndarray]) -> np.ndarray:
    name = spec.name
    if name.startswith("__identity_"):
        n = spec.shape[0]
        return np.eye(n, dtype=np.float16)
    if name.startswith("__zero_"):
        return np.zeros(spec.shape, dtype=np.float16)
    if name in weights:
        return np.asarray(weights[name], dtype=np.float16)
    raise KeyError(f"cannot synthesize constant {name!r}")


class MLPRunner:
    """Complete learned MNIST MLP, one static batch profile per runner."""

    def __init__(self, lib: LibRKPlan, plan: Plan, weights: dict[str, np.ndarray],
                 batch: int, *, fd: int = 0):
        self.lib = lib
        self.plan = plan
        self.batch = int(batch)
        self.call = PlanCall(lib, plan, self._constants(plan, weights, self.batch),
                             fd=fd)

    @staticmethod
    def _constants(plan: Plan, w: dict[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
        # Source W arrays are [K,N]; bias arrays are expanded to the static tile.
        w1 = np.zeros((1024, 256), dtype=np.float16)
        w1[:784] = np.asarray(w["W1"], dtype=np.float16)
        w2 = np.asarray(w["W2"], dtype=np.float16)
        w3 = np.zeros((128, 16), dtype=np.float16)
        w3[:, :10] = np.asarray(w["W3"], dtype=np.float16)
        b1 = np.repeat(np.asarray(w["b1"], dtype=np.float16)[None, :], batch, axis=0)
        b2 = np.repeat(np.asarray(w["b2"], dtype=np.float16)[None, :], batch, axis=0)
        b3 = np.zeros((batch, 16), dtype=np.float16)
        b3[:, :10] = np.asarray(w["b3"], dtype=np.float16)
        named = {"W1": w1, "B1": b1, "W2": w2, "B2": b2, "W3": w3, "B3": b3}
        out = {}
        for spec in plan.constants:
            out[spec.name] = _generated_constant(spec, named)
        return out

    def run(self, raw_images: np.ndarray, *, stats: RunStats | None = None) -> np.ndarray:
        x = preprocess_mnist(raw_images)
        if x.shape[0] != self.batch:
            reject(RK_E_SHAPE, "MLP raw batch does not match compiled profile",
                   stage="model", actual=x.shape[0], required=self.batch)
        xp = np.zeros((self.batch, 1024), dtype=np.float16)
        xp[:, :784] = x.astype(np.float16)
        outs, report, elapsed = self.call.run([xp])
        if stats is not None:
            stats.add_report("mlp", report, elapsed)
        return np.asarray(outs[0][:, :10], dtype=np.float32)

    def close(self) -> None:
        self.call.close()


class TinyViTRunner:
    """Hybrid 16-token, one-layer, 4-head Tiny Transformer image classifier."""

    def __init__(self, lib: LibRKPlan, plans: dict[str, Plan],
                 weights: dict[str, np.ndarray], batch: int, *, fd: int = 0):
        self.lib = lib
        self.batch = int(batch)
        self.weights = {k: np.asarray(v, dtype=np.float16) for k, v in weights.items()}
        self.calls: dict[str, PlanCall] = {}
        for name, plan in plans.items():
            name_consts = {}
            for spec in plan.constants:
                name_consts[spec.name] = self._constant_for(name, spec)
            self.calls[name] = PlanCall(lib, plan, name_consts, fd=fd)

    def _constant_for(self, plan_name: str, spec) -> np.ndarray:
        w = self.weights
        if spec.name == "B":
            if plan_name == "patch":
                b = np.zeros((64, 64), dtype=np.float16)
                b[:49] = w["patch_w"]
                return b
            if plan_name == "qkv":
                return w["qkv_w"]
            if plan_name == "attn_out":
                return w["attn_out_w"]
            if plan_name == "gate":
                return w["ffn_gate_w"]
            if plan_name == "up":
                return w["ffn_up_w"]
            if plan_name == "down":
                return w["ffn_down_w"]
            if plan_name == "classifier":
                b = np.zeros((64, 16), dtype=np.float16)
                b[:, :10] = w["class_w"]
                return b
        if spec.name.startswith("__identity_"):
            n = spec.shape[0]
            return np.eye(n, dtype=np.float16)
        if spec.name.startswith("__zero_"):
            return np.zeros(spec.shape, dtype=np.float16)
        raise KeyError(f"no Transformer constant for {plan_name}/{spec.name}")

    @staticmethod
    def _rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        xf = x.astype(np.float32)
        scale = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
        return np.asarray(xf * scale * gamma.astype(np.float32), dtype=np.float16)

    def run(self, raw_images: np.ndarray, *, stats: RunStats | None = None) -> np.ndarray:
        b = self.batch
        patches = patchify_mnist(raw_images)
        if patches.shape[0] != b:
            reject(RK_E_SHAPE, "Transformer raw batch does not match profile",
                   stage="model", actual=patches.shape[0], required=b)
        host_us = 0.0
        t0 = time.perf_counter()
        pflat = np.zeros((b * 16, 64), dtype=np.float16)
        pflat[:, :49] = patches.reshape(b * 16, 49).astype(np.float16)
        host_us += (time.perf_counter() - t0) * 1e6

        patch_out, rep, us = self.calls["patch"].run([pflat])
        if stats is not None: stats.add_report("patch", rep, us)
        h = patch_out[0].reshape(b, 16, 64).astype(np.float32)
        h += self.weights["pos"][None, :, :].astype(np.float32)
        h = h.astype(np.float16)

        t0 = time.perf_counter()
        z = self._rmsnorm(h, self.weights["norm1_gamma"])
        host_us += (time.perf_counter() - t0) * 1e6
        qkv, rep, us = self.calls["qkv"].run([z.reshape(b * 16, 64)])
        if stats is not None: stats.add_report("qkv", rep, us)
        qkv = qkv[0].reshape(b, 16, 3, 4, 16)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        heads = np.empty((b, 4, 16, 16), dtype=np.float16)
        host_attn = 0.0
        for bi in range(b):
            for hi in range(4):
                qh = np.ascontiguousarray(q[bi, hi], dtype=np.float16)
                kh = np.ascontiguousarray(k[bi, hi], dtype=np.float16)
                vh = np.ascontiguousarray(v[bi, hi], dtype=np.float16)
                scores, rep, us = self.calls["qk"].run([qh, kh.T])
                if stats is not None: stats.add_report("qk", rep, us)
                t1 = time.perf_counter()
                sf = scores[0].astype(np.float32) / np.float32(4.0)
                sf = sf - sf.max(axis=-1, keepdims=True)
                e = np.exp(sf)
                probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)
                host_attn += (time.perf_counter() - t1) * 1e6
                out, rep, us = self.calls["pv"].run([probs, vh])
                if stats is not None: stats.add_report("pv", rep, us)
                heads[bi, hi] = out[0]
        host_us += host_attn
        merged = heads.transpose(0, 2, 1, 3).reshape(b * 16, 64)
        attn, rep, us = self.calls["attn_out"].run([np.ascontiguousarray(merged)])
        if stats is not None: stats.add_report("attn_out", rep, us)
        h = (h.reshape(b * 16, 64).astype(np.float32) +
             attn[0].astype(np.float32)).astype(np.float16)

        t0 = time.perf_counter()
        z = self._rmsnorm(h.reshape(b, 16, 64), self.weights["norm2_gamma"])
        host_us += (time.perf_counter() - t0) * 1e6
        zf = z.reshape(b * 16, 64)
        gate, rep, us = self.calls["gate"].run([zf])
        if stats is not None: stats.add_report("gate", rep, us)
        up, rep, us = self.calls["up"].run([zf])
        if stats is not None: stats.add_report("up", rep, us)
        t0 = time.perf_counter()
        g = gate[0].astype(np.float32)
        u = up[0].astype(np.float32)
        act = (g / (1.0 + np.exp(-g)) * u).astype(np.float16)
        host_us += (time.perf_counter() - t0) * 1e6
        down, rep, us = self.calls["down"].run([act])
        if stats is not None: stats.add_report("down", rep, us)
        h = (h.reshape(b * 16, 64).astype(np.float32) +
             down[0].astype(np.float32)).astype(np.float16)

        t0 = time.perf_counter()
        pooled = h.reshape(b, 16, 64).astype(np.float32).mean(axis=1).astype(np.float16)
        host_us += (time.perf_counter() - t0) * 1e6
        logits, rep, us = self.calls["classifier"].run([pooled])
        if stats is not None: stats.add_report("classifier", rep, us)
        if stats is not None:
            stats.host_compute_us += host_us
        return np.asarray(logits[0][:, :10], dtype=np.float32)

    def close(self) -> None:
        for c in self.calls.values():
            c.close()


def load_weight_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}
