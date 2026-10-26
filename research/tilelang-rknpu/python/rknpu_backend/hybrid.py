# SPDX-License-Identifier: GPL-3.0-or-later
"""Versioned, inspectable model execution manifests for the frozen models.

This is intentionally *not* a model importer.  A ``rkhybrid`` manifest is a
closed, statically validated graph whose NPU nodes name immutable RKPlan files
and whose Host nodes name actions in a small versioned registry.  Shapes, axes,
dependencies, numeric contracts and bridge accounting are part of the manifest.
Unknown fields, actions, plan hashes or shape/dtype combinations fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import (RK_E_INVALID, RK_E_NUMERIC_CONTRACT, RK_E_SCHEMA,
                     RK_E_SHAPE, RK_E_UNSUPPORTED, reject)
from .models import PlanCall, RunStats
from .plan import Plan
from .runtime.ffi import LibRKPlan

HYBRID_SCHEMA_VERSION = 1
HYBRID_KIND = "rkhybrid"
HOST_ACTIONS_VERSION = 1
DTYPES = {"uint8": np.uint8, "float16": np.float16, "float32": np.float32}
PHASES = {"preprocess", "execution", "postprocess"}
_F16 = np.dtype("<f2")


def _keys(obj: Any, required: set[str], optional: set[str] | None = None,
          *, where: str) -> None:
    if not isinstance(obj, dict):
        reject(RK_E_SCHEMA, f"{where} must be an object", stage="hybrid")
    optional = optional or set()
    missing = sorted(required - obj.keys())
    unknown = sorted(obj.keys() - required - optional)
    if missing:
        reject(RK_E_SCHEMA, f"{where} missing fields: {', '.join(missing)}",
               stage="hybrid")
    if unknown:
        reject(RK_E_SCHEMA, f"{where} unknown fields: {', '.join(unknown)}",
               stage="hybrid")


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        reject(RK_E_SCHEMA, f"{where} must be a non-empty string",
               stage="hybrid")
    return value


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        reject(RK_E_SCHEMA, f"{where} must be an integer >= {minimum}",
               stage="hybrid")
    return value


def _int_tuple(value: Any, where: str, *, positive: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list):
        reject(RK_E_SCHEMA, f"{where} must be an array", stage="hybrid")
    out = tuple(_int(v, f"{where}[{i}]",
                     minimum=1 if positive else 0)
                for i, v in enumerate(value))
    return out


def _tensor_spec(obj: Any, where: str) -> "TensorSpec":
    _keys(obj, {"name", "dtype", "shape", "axes"}, where=where)
    name = _string(obj["name"], f"{where}.name")
    dtype = _string(obj["dtype"], f"{where}.dtype")
    if dtype not in DTYPES:
        reject(RK_E_UNSUPPORTED, f"{where}.dtype {dtype!r} is not supported",
               stage="hybrid")
    shape = _int_tuple(obj["shape"], f"{where}.shape", positive=True)
    axes = obj["axes"]
    if not isinstance(axes, list) or len(axes) != len(shape):
        reject(RK_E_SCHEMA, f"{where}.axes must match tensor rank",
               stage="hybrid", actual=axes)
    parsed_axes = tuple(_string(axis, f"{where}.axes[{i}]")
                        for i, axis in enumerate(axes))
    if len(set(parsed_axes)) != len(parsed_axes):
        reject(RK_E_SCHEMA, f"{where}.axes contains duplicates", stage="hybrid")
    return TensorSpec(name, dtype, shape, parsed_axes)


def _expected_nbytes(spec: "TensorSpec") -> int:
    return int(math.prod(spec.shape)) * np.dtype(DTYPES[spec.dtype]).itemsize


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]

    @property
    def nbytes(self) -> int:
        return _expected_nbytes(self)


@dataclass(frozen=True)
class TensorRef:
    tensor: str
    index: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class RepeatAxis:
    name: str
    extent: int
    axis: str


@dataclass(frozen=True)
class HostConstant:
    spec: TensorSpec
    weight: str


@dataclass(frozen=True)
class PlanInfo:
    key: str
    path: Path
    plan: Plan
    plan_hash: str
    source_hash: str
    dynamic_operand: str | None


@dataclass(frozen=True)
class HostAction:
    action_id: str
    version: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    func: Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, np.ndarray]]


@dataclass(frozen=True)
class NpuNode:
    id: str
    plan: str
    inputs: dict[str, TensorRef]
    outputs: dict[str, TensorRef]
    constants: dict[str, dict[str, Any]]
    depends_on: tuple[str, ...]
    repeat: tuple[RepeatAxis, ...]
    numeric_contract: dict[str, Any]
    bridge: dict[str, Any]
    phase: str
    # Optional static-repeat grouping policy.  ``repeat_submit`` selects how the
    # declared repeat set is executed; ``group_plan`` names the pre-expanded plan
    # (3N bindings: A_i, B_i, Y_i) that the shared path submits in one chain.
    repeat_submit: str = "per_repeat_submits"
    group_plan: str | None = None
    # Numeric-observation probes are statically present in the manifest but
    # execute only when the caller asks for a capture.  Their outputs are
    # diagnostic evidence from a separate plan execution, not a sample of the
    # production plan's internal state; ordinary nodes may not consume them.
    capture_only: bool = False


@dataclass(frozen=True)
class HostNode:
    id: str
    action: HostAction
    inputs: dict[str, TensorRef]
    outputs: dict[str, TensorRef]
    attrs: dict[str, Any]
    depends_on: tuple[str, ...]
    repeat: tuple[RepeatAxis, ...]
    numeric_contract: dict[str, Any]
    bridge: dict[str, Any]
    phase: str


Node = NpuNode | HostNode


@dataclass(frozen=True)
class _RepeatRef:
    """One resolved binding access for a single repeat instance.

    Everything that depends only on the immutable manifest (which tensor, which
    repeat indices, whether the whole tensor must be allocated on first write and
    the exact element shape) is resolved when the runner is created, so the hot
    loop performs no ref parsing, dict lookup by binding name or shape checks.
    """

    tensor: str
    index: tuple[int, ...]
    allocate: tuple[int, ...] | None
    dtype: str


@dataclass(frozen=True)
class _NpuBindingPlan:
    """Static execution shape of one RKPlan node.

    ``mode`` is the effective repeat-submit policy for this node:
      ``independent_calls`` -- one PlanCall per repeat (the original path)
      ``per_repeat_submits`` -- one native group entry, N submits (B1)
      ``shared_group`` -- one pre-expanded plan, one chained submit (B2), with the
                          flattened binding list in ``flat``.
    """

    plan: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    repeats: tuple[tuple[tuple[_RepeatRef, ...], tuple[_RepeatRef, ...]], ...]
    mode: str = "per_repeat_submits"
    flat_inputs: tuple[_RepeatRef, ...] = ()
    flat_outputs: tuple[_RepeatRef, ...] = ()


@dataclass
class HybridProgram:
    path: Path
    name: str
    model: str
    batch: int
    profile_hash: str
    execution_policy: dict[str, Any]
    bridge_policy: dict[str, Any]
    input_spec: TensorSpec
    host_constants: dict[str, HostConstant]
    tensors: dict[str, TensorSpec]
    plans: dict[str, PlanInfo]
    nodes: tuple[Node, ...]
    outputs: dict[str, str]
    trace_tensors: tuple[str, ...]
    weights: dict[str, Any]


@dataclass
class HybridResult:
    outputs: dict[str, np.ndarray]
    trace: dict[str, np.ndarray]


def _parse_ref(obj: Any, where: str, repeat_names: set[str]) -> TensorRef:
    _keys(obj, {"tensor"}, {"index"}, where=where)
    tensor = _string(obj["tensor"], f"{where}.tensor")
    raw_index = obj.get("index", [])
    if not isinstance(raw_index, list):
        reject(RK_E_SCHEMA, f"{where}.index must be an array", stage="hybrid")
    index: list[int | str] = []
    for i, token in enumerate(raw_index):
        if isinstance(token, str):
            if token not in repeat_names:
                reject(RK_E_SCHEMA,
                       f"{where}.index[{i}] uses unknown repeat variable "
                       f"{token!r}", stage="hybrid")
            index.append(token)
        elif isinstance(token, int) and not isinstance(token, bool) and token >= 0:
            index.append(token)
        else:
            reject(RK_E_SCHEMA,
                   f"{where}.index[{i}] must be a non-negative integer or "
                   "repeat variable", stage="hybrid")
    return TensorRef(tensor, tuple(index))


def _parse_repeat(obj: Any, where: str) -> tuple[RepeatAxis, ...]:
    if obj is None:
        return ()
    if not isinstance(obj, list):
        reject(RK_E_SCHEMA, f"{where} must be an array", stage="hybrid")
    axes: list[RepeatAxis] = []
    for i, item in enumerate(obj):
        _keys(item, {"name", "extent", "axis"}, where=f"{where}[{i}]")
        axes.append(RepeatAxis(
            _string(item["name"], f"{where}[{i}].name"),
            _int(item["extent"], f"{where}[{i}].extent", minimum=1),
            _string(item["axis"], f"{where}[{i}].axis")))
    if len({axis.name for axis in axes}) != len(axes):
        reject(RK_E_SCHEMA, f"{where} repeat variable names are not unique",
               stage="hybrid")
    return tuple(axes)


def _parse_numeric(obj: Any, where: str) -> dict[str, Any]:
    _keys(obj, {"storage", "accumulation", "writeback", "semantics"},
          {"dynamic_b", "scale", "ordered_k_tile", "rounding"},
          where=where)
    for key in ("storage", "accumulation", "writeback", "semantics"):
        _string(obj[key], f"{where}.{key}")
    if obj.get("dynamic_b") is not None and \
            not isinstance(obj["dynamic_b"], bool):
        reject(RK_E_SCHEMA, f"{where}.dynamic_b must be boolean",
               stage="hybrid")
    return dict(obj)


def _parse_bridge(obj: Any, where: str, kind: str) -> dict[str, Any]:
    _keys(obj, {"kind", "accounting"}, where=where)
    expected = ("rkplan_transfer", "runtime_report") if kind == "rkplan" else \
               ("host_tensor_materialization", "tensor_bytes")
    if (obj["kind"], obj["accounting"]) != expected:
        reject(RK_E_SCHEMA, f"{where} must declare {expected}", stage="hybrid",
               actual=[obj["kind"], obj["accounting"]])
    return dict(obj)


def _parse_constant(obj: Any, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        reject(RK_E_SCHEMA, f"{where} must be an object", stage="hybrid")
    kind = obj.get("kind")
    if kind == "weight":
        _keys(obj, {"kind", "name"}, where=where)
        return {"kind": "weight",
                "name": _string(obj["name"], f"{where}.name")}
    if kind == "weight_concat":
        _keys(obj, {"kind", "parts", "axis"}, where=where)
        parts = obj["parts"]
        if not isinstance(parts, list) or len(parts) < 2:
            reject(RK_E_SCHEMA, f"{where}.parts must list at least two weights",
                   stage="hybrid")
        axis = _int(obj["axis"], f"{where}.axis")
        if axis not in (0, 1):
            reject(RK_E_UNSUPPORTED, f"{where}.axis must be 0 or 1",
                   stage="hybrid")
        return {"kind": kind,
                "parts": [_string(part, f"{where}.parts[{i}]")
                          for i, part in enumerate(parts)],
                "axis": axis}
    if kind == "weight_repeat_pad":
        _keys(obj, {"kind", "name", "reshape"},
              {"repeat_rows"}, where=where)
        reshape = _int_tuple(obj["reshape"], f"{where}.reshape", positive=True)
        if len(reshape) != 2:
            reject(RK_E_SHAPE, f"{where}.reshape must be 2-D", stage="hybrid")
        repeat_rows = _int(obj.get("repeat_rows", 1),
                           f"{where}.repeat_rows", minimum=1)
        return {"kind": kind, "name": _string(obj["name"], f"{where}.name"),
                "reshape": reshape, "repeat_rows": repeat_rows}
    if kind == "generated":
        _keys(obj, {"kind", "generator"}, where=where)
        generator = _string(obj["generator"], f"{where}.generator")
        if generator not in {"identity", "zero"}:
            reject(RK_E_UNSUPPORTED,
                   f"{where}.generator {generator!r} is not registered",
                   stage="hybrid")
        return {"kind": kind, "generator": generator}
    reject(RK_E_UNSUPPORTED, f"{where}.kind {kind!r} is not registered",
           stage="hybrid")


# ---------------------------------------------------------------------------
# Registered Host actions.  Each action is small, versioned and independently
# testable; the transformer is never hidden behind one opaque Python call.
# ---------------------------------------------------------------------------

def _action_attrs(attrs: dict[str, Any], required: set[str],
                  optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = sorted(required - attrs.keys())
    unknown = sorted(attrs.keys() - required - optional)
    if missing:
        reject(RK_E_SCHEMA, f"Host action missing attrs: {', '.join(missing)}",
               stage="hybrid")
    if unknown:
        reject(RK_E_SCHEMA,
               f"Host action has unknown attrs: {', '.join(unknown)}",
               stage="hybrid")


def _norm_images(raw: np.ndarray) -> np.ndarray:
    a = np.asarray(raw)
    if a.ndim == 3:
        if a.shape[1:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,28,28]",
                   stage="host_action", actual=list(a.shape))
        flat = a.reshape(a.shape[0], 784)
    elif a.ndim == 4:
        if a.shape[1] != 1 or a.shape[2:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,1,28,28]",
                   stage="host_action", actual=list(a.shape))
        flat = a[:, 0]
    else:
        reject(RK_E_SHAPE, "MNIST images must be rank 3 or 4",
               stage="host_action", actual=list(a.shape))
    value = flat.astype(np.float32)
    if value.size and float(np.max(value)) > 1.5:
        value /= np.float32(255.0)
    value = (value - np.float32(0.1307)) / np.float32(0.3081)
    return value.astype(np.float32)


def _mnist_normalize_flat(inputs: dict[str, np.ndarray],
                          attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, set())
    x = _norm_images(inputs["raw"])
    return {"x": np.ascontiguousarray(x, dtype=np.float16)}


def _mnist_patchify_norm(inputs: dict[str, np.ndarray],
                         attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, set())
    flat = _norm_images(inputs["raw"]).astype(np.float32)
    b = flat.shape[0]
    # [N,4,7,4,7] -> [N,4,4,7,7] -> [N,16,49], source raster order.
    patches = flat.reshape(b, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4)
    return {"patches": np.ascontiguousarray(
        patches.reshape(b, 16, 49), dtype=np.float16)}


def _tensor_pad_2d(inputs: dict[str, np.ndarray],
                   attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"input_shape", "copy_shape", "output_shape"})
    source = inputs["x"]
    if source.dtype != _F16:
        reject(RK_E_INVALID, "tensor.pad_2d input must be FP16", stage="host_action")
    input_shape = _int_tuple(attrs["input_shape"], "pad.input_shape", positive=True)
    copy_shape = _int_tuple(attrs["copy_shape"], "pad.copy_shape", positive=True)
    output_shape = _int_tuple(attrs["output_shape"], "pad.output_shape", positive=True)
    if len(input_shape) != 2 or len(copy_shape) != 2 or len(output_shape) != 2:
        reject(RK_E_SHAPE, "tensor.pad_2d shapes must be rank 2",
               stage="host_action")
    source = np.ascontiguousarray(source.reshape(input_shape), dtype=np.float16)
    if source.shape != copy_shape:
        reject(RK_E_SHAPE, "tensor.pad_2d source/copy shape mismatch",
               stage="host_action", actual=list(source.shape),
               required=list(copy_shape))
    if copy_shape[0] > output_shape[0] or copy_shape[1] > output_shape[1]:
        reject(RK_E_SHAPE, "tensor.pad_2d copy exceeds output shape",
               stage="host_action")
    out = np.zeros(output_shape, dtype=np.float16)
    out[:copy_shape[0], :copy_shape[1]] = source
    return {"y": out}


def _tensor_reshape(inputs: dict[str, np.ndarray],
                    attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"shape"})
    shape = _int_tuple(attrs["shape"], "reshape.shape", positive=True)
    x = inputs["x"]
    if int(np.prod(x.shape)) != int(np.prod(shape)):
        reject(RK_E_SHAPE, "reshape must preserve element count",
               stage="host_action", actual=list(x.shape), required=list(shape))
    return {"y": np.ascontiguousarray(x.reshape(shape), dtype=x.dtype)}


def _tensor_add_f16(inputs: dict[str, np.ndarray],
                    attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, set())
    a, b = np.asarray(inputs["a"]), np.asarray(inputs["b"])
    if a.dtype != _F16 or b.dtype != _F16 or a.shape != b.shape:
        reject(RK_E_SHAPE,
               "tensor.add requires equal-shape FP16 inputs", stage="host_action",
               actual={"a": [str(a.dtype), list(a.shape)],
                       "b": [str(b.dtype), list(b.shape)]})
    return {"y": np.ascontiguousarray(
        a.astype(np.float32) + b.astype(np.float32), dtype=np.float16)}


def _tiny_vit_position_add(inputs: dict[str, np.ndarray],
                           attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"batch", "tokens"})
    batch = _int(attrs["batch"], "position_add.batch", minimum=1)
    tokens = _int(attrs["tokens"], "position_add.tokens", minimum=1)
    x, pos = inputs["x"], inputs["pos"]
    if x.dtype != _F16 or pos.dtype != _F16:
        reject(RK_E_INVALID, "position_add inputs must be FP16",
               stage="host_action")
    if x.shape != (batch * tokens, pos.shape[1]) or pos.shape != (tokens, x.shape[1]):
        reject(RK_E_SHAPE, "position_add shape mismatch", stage="host_action",
               actual={"x": list(x.shape), "pos": list(pos.shape)})
    value = x.reshape(batch, tokens, x.shape[1]).astype(np.float32)
    value += pos.astype(np.float32)[None, :, :]
    return {"y": np.ascontiguousarray(
        value.reshape(batch * tokens, x.shape[1]), dtype=np.float16)}


def _tensor_add_rmsnorm(inputs: dict[str, np.ndarray],
                        attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Same-shape FP16 residual add followed by last-axis RMSNorm.

    This is the literal composition of ``tensor.add.v1`` and
    ``rmsnorm.last_axis.v1``: the residual sum is narrowed to FP16 before the
    RMSNorm input is read, and the RMSNorm reduction/rsqrt/gamma product stays in
    FP32 with one final FP16 store.  It removes a Host action boundary without
    changing any arithmetic or narrowing point.
    """
    _action_attrs(attrs, {"eps"})
    eps = float(attrs["eps"])
    if not math.isfinite(eps) or eps <= 0:
        reject(RK_E_INVALID, "rmsnorm eps must be finite and positive",
               stage="host_action")
    a, b, gamma = inputs["a"], inputs["b"], inputs["gamma"]
    if a.dtype != _F16 or b.dtype != _F16 or gamma.dtype != _F16 or \
            a.shape != b.shape:
        reject(RK_E_SHAPE,
               "tensor.add_rmsnorm requires equal-shape FP16 a/b and FP16 gamma",
               stage="host_action",
               actual={"a": [str(a.dtype), list(a.shape)],
                       "b": [str(b.dtype), list(b.shape)],
                       "gamma": [str(gamma.dtype), list(gamma.shape)]})
    if gamma.shape != (a.shape[-1],):
        reject(RK_E_SHAPE, "rmsnorm gamma shape mismatch", stage="host_action",
               actual=list(gamma.shape), required=[a.shape[-1]])
    h = np.ascontiguousarray(
        a.astype(np.float32) + b.astype(np.float32), dtype=np.float16)
    xf = h.astype(np.float32)
    scale = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    y = np.ascontiguousarray(
        xf * scale * gamma.astype(np.float32), dtype=np.float16)
    return {"h": h, "y": y}


def _rmsnorm_last_axis(inputs: dict[str, np.ndarray],
                       attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"eps"})
    eps = float(attrs["eps"])
    if not math.isfinite(eps) or eps <= 0:
        reject(RK_E_INVALID, "rmsnorm eps must be finite and positive",
               stage="host_action")
    x, gamma = inputs["x"], inputs["gamma"]
    if x.dtype != _F16 or gamma.dtype != _F16:
        reject(RK_E_INVALID, "rmsnorm inputs must be FP16", stage="host_action")
    if gamma.shape != (x.shape[-1],):
        reject(RK_E_SHAPE, "rmsnorm gamma shape mismatch", stage="host_action",
               actual=list(gamma.shape), required=[x.shape[-1]])
    xf = x.astype(np.float32)
    scale = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    return {"y": np.ascontiguousarray(
        xf * scale * gamma.astype(np.float32), dtype=np.float16)}


def _qkv_split_heads(inputs: dict[str, np.ndarray],
                     attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"batch", "tokens", "heads", "head_dim"})
    batch = _int(attrs["batch"], "split.batch", minimum=1)
    tokens = _int(attrs["tokens"], "split.tokens", minimum=1)
    heads = _int(attrs["heads"], "split.heads", minimum=1)
    head_dim = _int(attrs["head_dim"], "split.head_dim", minimum=1)
    qkv = inputs["qkv"]
    if qkv.dtype != _F16:
        reject(RK_E_INVALID, "qkv split input must be FP16", stage="host_action")
    expected = (batch * tokens, 3 * heads * head_dim)
    if qkv.shape != expected:
        reject(RK_E_SHAPE, "qkv split input shape mismatch", stage="host_action",
               actual=list(qkv.shape), required=list(expected))
    value = qkv.reshape(batch, tokens, 3, heads, head_dim)
    q = value[:, :, 0].transpose(0, 2, 1, 3)
    kt = value[:, :, 1].transpose(0, 2, 3, 1)
    v = value[:, :, 2].transpose(0, 2, 1, 3)
    return {"q": np.ascontiguousarray(q, dtype=np.float16),
            "k_t": np.ascontiguousarray(kt, dtype=np.float16),
            "v": np.ascontiguousarray(v, dtype=np.float16)}


def _scaled_softmax(inputs: dict[str, np.ndarray],
                    attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"scale", "axis"})
    if int(attrs["axis"]) != -1:
        reject(RK_E_UNSUPPORTED, "scaled_softmax currently requires axis=-1",
               stage="host_action", actual=attrs["axis"])
    score = inputs["scores"]
    if score.dtype != _F16:
        reject(RK_E_INVALID, "softmax input must be FP16", stage="host_action")
    scale = float(attrs["scale"])
    if not math.isfinite(scale) or scale <= 0:
        reject(RK_E_INVALID, "softmax scale must be finite and positive",
               stage="host_action")
    sf = score.astype(np.float32) * np.float32(scale)
    sf -= sf.max(axis=-1, keepdims=True)
    e = np.exp(sf)
    probs = e / e.sum(axis=-1, keepdims=True)
    return {"y": np.ascontiguousarray(probs, dtype=np.float16)}


def _merge_heads(inputs: dict[str, np.ndarray],
                 attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"batch", "tokens", "heads", "head_dim"})
    batch = _int(attrs["batch"], "merge.batch", minimum=1)
    tokens = _int(attrs["tokens"], "merge.tokens", minimum=1)
    heads = _int(attrs["heads"], "merge.heads", minimum=1)
    head_dim = _int(attrs["head_dim"], "merge.head_dim", minimum=1)
    x = inputs["heads"]
    if x.dtype != _F16 or x.shape != (batch, heads, tokens, head_dim):
        reject(RK_E_SHAPE, "merge_heads expects [B,H,T,D] FP16",
               stage="host_action", actual=list(x.shape))
    merged = x.transpose(0, 2, 1, 3).reshape(batch * tokens, heads * head_dim)
    return {"y": np.ascontiguousarray(merged, dtype=np.float16)}


def _swiglu(inputs: dict[str, np.ndarray],
            attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, set())
    gate, up = inputs["gate"], inputs["up"]
    if gate.dtype != _F16 or up.dtype != _F16 or gate.shape != up.shape:
        reject(RK_E_SHAPE, "swiglu requires equal-shape FP16 gate/up",
               stage="host_action",
               actual={"gate": list(gate.shape), "up": list(up.shape)})
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-g))
    return {"y": np.ascontiguousarray(g * sigmoid * u, dtype=np.float16)}


def _swiglu_packed(inputs: dict[str, np.ndarray],
                   attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Split a packed [gate|up] GEMM result and apply exact FP32 SwiGLU.

    The device computes the two branches as one N=2H GEMM with concatenated
    static weights; the host split is a pure layout view.  The activation product
    is still narrowed exactly once, so this preserves the established host-SwiGLU
    numeric path.
    """
    _action_attrs(attrs, set())
    packed = inputs["gate_up"]
    if packed.dtype != _F16 or packed.ndim != 2 or packed.shape[1] % 2:
        reject(RK_E_SHAPE,
               "swiglu.packed_gate_up requires FP16 [M,2H] input",
               stage="host_action", actual=list(packed.shape))
    half = packed.shape[1] // 2
    gate = np.ascontiguousarray(packed[:, :half], dtype=np.float16)
    up = np.ascontiguousarray(packed[:, half:], dtype=np.float16)
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-g))
    act = np.ascontiguousarray(g * sigmoid * u, dtype=np.float16)
    return {"gate": gate, "up": up, "act": act}


def _mean_axis(inputs: dict[str, np.ndarray],
               attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"axis"})
    axis = int(attrs["axis"])
    x = inputs["x"]
    if x.dtype != _F16:
        reject(RK_E_INVALID, "mean_axis input must be FP16", stage="host_action")
    return {"y": np.ascontiguousarray(
        x.astype(np.float32).mean(axis=axis), dtype=np.float16)}


def _logits_prefix(inputs: dict[str, np.ndarray],
                   attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    _action_attrs(attrs, {"cols"})
    cols = _int(attrs["cols"], "logits_prefix.cols", minimum=1)
    x = inputs["logits"]
    if x.dtype != _F16 or cols > x.shape[-1]:
        reject(RK_E_SHAPE, "logits_prefix requires FP16 logits and valid cols",
               stage="host_action", actual=list(x.shape))
    return {"y": np.ascontiguousarray(x[:, :cols], dtype=np.float32)}


HOST_ACTIONS: dict[str, HostAction] = {
    action.action_id: action for action in (
        HostAction("mnist.normalize_flatten.v1", HOST_ACTIONS_VERSION,
                   ("raw",), ("x",), _mnist_normalize_flat),
        HostAction("mnist.patchify_norm.v1", HOST_ACTIONS_VERSION,
                   ("raw",), ("patches",), _mnist_patchify_norm),
        HostAction("tensor.pad_2d.v1", HOST_ACTIONS_VERSION,
                   ("x",), ("y",), _tensor_pad_2d),
        HostAction("tensor.reshape.v1", HOST_ACTIONS_VERSION,
                   ("x",), ("y",), _tensor_reshape),
        HostAction("tensor.add.v1", HOST_ACTIONS_VERSION,
                   ("a", "b"), ("y",), _tensor_add_f16),
        HostAction("tiny_vit.position_add.v1", HOST_ACTIONS_VERSION,
                   ("x", "pos"), ("y",), _tiny_vit_position_add),
        HostAction("rmsnorm.last_axis.v1", HOST_ACTIONS_VERSION,
                   ("x", "gamma"), ("y",), _rmsnorm_last_axis),
        HostAction("tensor.add_rmsnorm.last_axis.v1", HOST_ACTIONS_VERSION,
                   ("a", "b", "gamma"), ("h", "y"), _tensor_add_rmsnorm),
        HostAction("tiny_vit.qkv_split_heads.v1", HOST_ACTIONS_VERSION,
                   ("qkv",), ("q", "k_t", "v"), _qkv_split_heads),
        HostAction("attention.scaled_softmax.v1", HOST_ACTIONS_VERSION,
                   ("scores",), ("y",), _scaled_softmax),
        HostAction("tiny_vit.merge_heads.v1", HOST_ACTIONS_VERSION,
                   ("heads",), ("y",), _merge_heads),
        HostAction("swiglu.gate_up.v1", HOST_ACTIONS_VERSION,
                   ("gate", "up"), ("y",), _swiglu),
        HostAction("swiglu.packed_gate_up.v1", HOST_ACTIONS_VERSION,
                   ("gate_up",), ("gate", "up", "act"), _swiglu_packed),
        HostAction("tensor.mean_axis.v1", HOST_ACTIONS_VERSION,
                   ("x",), ("y",), _mean_axis),
        HostAction("logits.prefix.v1", HOST_ACTIONS_VERSION,
                   ("logits",), ("y",), _logits_prefix),
    )
}


def load_hybrid_program(path: str | Path, *,
                        plans_dir: str | Path | None = None) -> HybridProgram:
    source = Path(path).resolve()
    try:
        obj = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        reject(RK_E_SCHEMA, f"cannot read hybrid manifest: {exc}",
               stage="hybrid")
    _keys(obj, {"schema_version", "kind", "name", "model", "batch",
                "profile_hash", "execution_policy", "bridge_policy", "input",
                "host_constants", "tensors", "plans", "nodes", "outputs",
                "trace_tensors", "weights"}, where="hybrid manifest")
    if obj["schema_version"] != HYBRID_SCHEMA_VERSION:
        reject(RK_E_SCHEMA, "unsupported hybrid manifest schema_version",
               stage="hybrid", actual=obj["schema_version"],
               required=HYBRID_SCHEMA_VERSION)
    if obj["kind"] != HYBRID_KIND:
        reject(RK_E_SCHEMA, "hybrid manifest kind must be rkhybrid",
               stage="hybrid")
    policy = obj["execution_policy"]
    _keys(policy, {"mode", "host_actions_version", "plan_reuse",
                   "host_fallback", "npu_fallback", "dynamic_shapes",
                   "unknown_action"}, where="execution_policy")
    expected_policy = {
        "mode": "explicit_hybrid",
        "host_actions_version": HOST_ACTIONS_VERSION,
        "plan_reuse": "per_plan_workspace",
        "host_fallback": "forbidden",
        "npu_fallback": "forbidden",
        "dynamic_shapes": "forbidden",
        "unknown_action": "reject",
    }
    if policy != expected_policy:
        reject(RK_E_UNSUPPORTED, "hybrid execution policy is not the frozen v1 "
               "closed policy", stage="hybrid", actual=policy,
               required=expected_policy)
    bridge_policy = obj["bridge_policy"]
    _keys(bridge_policy, {"host_action", "rkplan", "final_output",
                          "npu_fallback"}, where="bridge_policy")
    expected_bridge = {
        "host_action": "sum_resolved_input_output_tensor_bytes",
        "rkplan": "C_RKRunReport_upload_plus_download_bytes",
        "final_output": "host_materialized",
        "npu_fallback": "forbidden",
    }
    if bridge_policy != expected_bridge:
        reject(RK_E_UNSUPPORTED, "hybrid bridge policy is not the frozen v1 "
               "accounting contract", stage="hybrid")

    input_spec = _tensor_spec(obj["input"], "input")
    if not isinstance(obj["host_constants"], dict):
        reject(RK_E_SCHEMA, "host_constants must be an object", stage="hybrid")
    if not isinstance(obj["tensors"], list):
        reject(RK_E_SCHEMA, "tensors must be an array", stage="hybrid")
    if not isinstance(obj["plans"], dict):
        reject(RK_E_SCHEMA, "plans must be an object", stage="hybrid")
    if not isinstance(obj["nodes"], list):
        reject(RK_E_SCHEMA, "nodes must be an array", stage="hybrid")
    host_constants: dict[str, HostConstant] = {}
    for name, item in obj["host_constants"].items():
        if not isinstance(item, dict):
            reject(RK_E_SCHEMA, f"host_constants.{name} must be an object",
                   stage="hybrid")
        _keys(item, {"weight", "dtype", "shape", "axes"},
              where=f"host_constants.{name}")
        spec = _tensor_spec({"name": name, **{k: item[k] for k in
                                             ("dtype", "shape", "axes")}},
                            f"host_constants.{name}")
        host_constants[name] = HostConstant(
            spec, _string(item["weight"], f"host_constants.{name}.weight"))
    tensors: dict[str, TensorSpec] = {}
    for i, item in enumerate(obj["tensors"]):
        spec = _tensor_spec(item, f"tensors[{i}]")
        if spec.name in tensors or spec.name == input_spec.name or \
                spec.name in host_constants:
            reject(RK_E_SCHEMA, f"duplicate graph value {spec.name!r}",
                   stage="hybrid")
        tensors[spec.name] = spec

    base = Path(plans_dir).resolve() if plans_dir is not None else source.parent
    plans: dict[str, PlanInfo] = {}
    for key, item in obj["plans"].items():
        if not isinstance(item, dict):
            reject(RK_E_SCHEMA, f"plans.{key} must be an object", stage="hybrid")
        _keys(item, {"path", "plan_hash", "source_hash", "dynamic_operand"},
              where=f"plans.{key}")
        rel = Path(_string(item["path"], f"plans.{key}.path"))
        if rel.is_absolute() or ".." in rel.parts:
            reject(RK_E_SCHEMA, f"plans.{key}.path must stay inside plans-dir",
                   stage="hybrid")
        plan_path = (base / rel).resolve()
        try:
            plan_path.relative_to(base)
        except ValueError:
            reject(RK_E_SCHEMA, f"plans.{key}.path escapes plans-dir",
                   stage="hybrid")
        if not plan_path.is_file():
            reject(RK_E_SCHEMA, f"missing RKPlan {plan_path}", stage="hybrid")
        from . import load_plan
        plan = load_plan(str(plan_path))
        plan_hash = _string(item["plan_hash"], f"plans.{key}.plan_hash")
        if plan.plan_hash != plan_hash:
            reject(RK_E_SCHEMA, f"plans.{key} hash mismatch", stage="hybrid",
                   actual=plan.plan_hash, required=plan_hash)
        if plan.profile_hash != obj["profile_hash"]:
            reject(RK_E_SCHEMA, f"plans.{key} profile hash differs from graph",
                   stage="hybrid", actual=plan.profile_hash,
                   required=obj["profile_hash"])
        dynamic = item["dynamic_operand"]
        if dynamic is not None:
            dynamic = _string(dynamic, f"plans.{key}.dynamic_operand")
            binding = next((b for b in plan.bindings if b.name == dynamic), None)
            if binding is None or binding.role != "input":
                reject(RK_E_SCHEMA,
                       f"plans.{key}.dynamic_operand {dynamic!r} is not a plan "
                       "input", stage="hybrid")
        plans[key] = PlanInfo(key, plan_path, plan, plan_hash,
                              _string(item["source_hash"],
                                      f"plans.{key}.source_hash"), dynamic)
    if not plans:
        reject(RK_E_SCHEMA, "hybrid manifest has no plans", stage="hybrid")

    weights = obj["weights"]
    _keys(weights, {"file", "sha256", "keys", "host_precision"},
          where="weights")
    if not isinstance(weights["keys"], dict) or not weights["keys"]:
        reject(RK_E_SCHEMA, "weights.keys must be a non-empty object",
               stage="hybrid")
    for name, item in weights["keys"].items():
        _keys(item, {"shape", "dtype"}, where=f"weights.keys.{name}")
        _int_tuple(item["shape"], f"weights.keys.{name}.shape", positive=True)
        if item["dtype"] not in {"float16", "float32"}:
            reject(RK_E_UNSUPPORTED, f"weights.keys.{name}.dtype unsupported",
                   stage="hybrid")
    if weights["host_precision"] != "float16":
        reject(RK_E_UNSUPPORTED, "host_precision must be float16",
               stage="hybrid")

    nodes: list[Node] = []
    node_by_id: dict[str, Node] = {}
    producer: dict[str, str] = {input_spec.name: "$input"}
    capture_only_tensor: dict[str, bool] = {}
    for name in host_constants:
        producer[name] = "$host_constant"
    plan_constants_by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for index, raw_node in enumerate(obj["nodes"]):
        if not isinstance(raw_node, dict):
            reject(RK_E_SCHEMA, f"nodes[{index}] must be an object",
                   stage="hybrid")
        kind = raw_node.get("kind")
        if kind not in {"rkplan", "host_action"}:
            reject(RK_E_UNSUPPORTED,
                   f"nodes[{index}].kind {kind!r} is not registered",
                   stage="hybrid")
        common = {"id", "kind", "inputs", "outputs", "depends_on", "repeat",
                  "numeric_contract", "bridge", "phase"}
        if kind == "rkplan":
            _keys(raw_node, common | {"plan", "constants"},
                  {"repeat_submit", "group_plan", "capture_only"},
                  where=f"nodes[{index}]")
        else:
            _keys(raw_node, common | {"action", "action_version", "attrs"},
                  where=f"nodes[{index}]")
        node_id = _string(raw_node["id"], f"nodes[{index}].id")
        if node_id in node_by_id:
            reject(RK_E_SCHEMA, f"duplicate node id {node_id!r}", stage="hybrid")
        repeat = _parse_repeat(raw_node["repeat"], f"nodes[{index}].repeat")
        repeat_names = {axis.name for axis in repeat}
        if not isinstance(raw_node["inputs"], dict) or \
                not isinstance(raw_node["outputs"], dict):
            reject(RK_E_SCHEMA, f"nodes[{index}] inputs/outputs must be objects",
                   stage="hybrid")
        deps = raw_node["depends_on"]
        if not isinstance(deps, list):
            reject(RK_E_SCHEMA, f"nodes[{index}].depends_on must be an array",
                   stage="hybrid")
        dep_ids = tuple(_string(v, f"nodes[{index}].depends_on[{i}]")
                        for i, v in enumerate(deps))
        if len(set(dep_ids)) != len(dep_ids):
            reject(RK_E_SCHEMA, f"nodes[{index}] has duplicate dependencies",
                   stage="hybrid")
        for dep in dep_ids:
            if dep not in node_by_id:
                reject(RK_E_SCHEMA,
                       f"nodes[{index}] depends on non-prior node {dep!r}",
                       stage="hybrid")
        refs_in = {key: _parse_ref(value, f"nodes[{index}].inputs.{key}",
                                   repeat_names)
                   for key, value in raw_node["inputs"].items()}
        refs_out = {key: _parse_ref(value, f"nodes[{index}].outputs.{key}",
                                    repeat_names)
                    for key, value in raw_node["outputs"].items()}
        phase = _string(raw_node["phase"], f"nodes[{index}].phase")
        if phase not in PHASES:
            reject(RK_E_SCHEMA, f"nodes[{index}].phase is not registered",
                   stage="hybrid")
        numeric = _parse_numeric(raw_node["numeric_contract"],
                                 f"nodes[{index}].numeric_contract")
        bridge = _parse_bridge(raw_node["bridge"], f"nodes[{index}].bridge", kind)
        if kind == "rkplan":
            plan_key = _string(raw_node["plan"], f"nodes[{index}].plan")
            if plan_key not in plans:
                reject(RK_E_SCHEMA, f"nodes[{index}] names unknown plan "
                       f"{plan_key!r}", stage="hybrid")
            if phase != "execution":
                reject(RK_E_SCHEMA, "RKPlan nodes must use execution phase",
                       stage="hybrid")
            plan = plans[plan_key].plan
            capture_only = raw_node.get("capture_only", False)
            if not isinstance(capture_only, bool):
                reject(RK_E_SCHEMA,
                       f"nodes[{index}].capture_only must be boolean",
                       stage="hybrid")
            repeat_submit = raw_node.get("repeat_submit", "per_repeat_submits")
            if repeat_submit not in REPEAT_SUBMIT_POLICIES:
                reject(RK_E_UNSUPPORTED,
                       f"nodes[{index}].repeat_submit {repeat_submit!r} is not "
                       "registered", stage="hybrid", actual=repeat_submit)
            group_plan_key = raw_node.get("group_plan")
            if group_plan_key is not None:
                group_plan_key = _string(group_plan_key,
                                         f"nodes[{index}].group_plan")
                if group_plan_key not in plans:
                    reject(RK_E_SCHEMA,
                           f"nodes[{index}] names unknown group plan "
                           f"{group_plan_key!r}", stage="hybrid")
            if repeat_submit == "shared_group":
                if group_plan_key is None:
                    reject(RK_E_SCHEMA,
                           "repeat_submit=shared_group requires a group_plan",
                           stage="hybrid")
                if not repeat:
                    reject(RK_E_SCHEMA,
                           "repeat_submit=shared_group requires a repeat axis",
                           stage="hybrid")
                _validate_group_plan(plan, plans[group_plan_key].plan,
                                     [a.extent for a in repeat],
                                     where=f"nodes[{index}]")
            elif group_plan_key is not None:
                reject(RK_E_SCHEMA,
                       "group_plan is only meaningful with "
                       "repeat_submit=shared_group", stage="hybrid")
            plan_inputs = [b.name for b in plan.bindings if b.role == "input"]
            plan_outputs = [b.name for b in plan.bindings if b.role == "output"]
            if set(refs_in) != set(plan_inputs):
                reject(RK_E_SHAPE,
                       f"nodes[{index}] inputs differ from RKPlan inputs",
                       stage="hybrid", actual=sorted(refs_in),
                       required=sorted(plan_inputs))
            if set(refs_out) != set(plan_outputs):
                reject(RK_E_SHAPE,
                       f"nodes[{index}] outputs differ from RKPlan outputs",
                       stage="hybrid", actual=sorted(refs_out),
                       required=sorted(plan_outputs))
            expected_out_index = tuple(axis.name for axis in repeat)
            for name, ref in refs_out.items():
                if ref.index != expected_out_index:
                    reject(RK_E_SHAPE,
                           f"nodes[{index}].outputs.{name} must be indexed by "
                           "all repeat axes", stage="hybrid",
                           actual=list(ref.index),
                           required=list(expected_out_index))
            for name, ref in refs_in.items():
                shape = _shape_after_ref_obj(program_input=input_spec,
                                             host_constants=host_constants,
                                             tensors=tensors,
                                             ref=ref, where=f"nodes[{index}].inputs.{name}")
                binding = next(b for b in plan.bindings if b.name == name)
                if shape != tuple(binding.shape):
                    reject(RK_E_SHAPE,
                           f"nodes[{index}].inputs.{name} shape does not match "
                           "RKPlan binding", stage="hybrid",
                           actual=list(shape), required=list(binding.shape))
            for name, ref in refs_out.items():
                shape = _shape_after_ref_obj(program_input=input_spec,
                                             host_constants=host_constants,
                                             tensors=tensors,
                                             ref=ref, where=f"nodes[{index}].outputs.{name}")
                binding = next(b for b in plan.bindings if b.name == name)
                if shape != tuple(binding.shape):
                    reject(RK_E_SHAPE,
                           f"nodes[{index}].outputs.{name} target suffix shape "
                           "does not match RKPlan output", stage="hybrid",
                           actual=list(shape), required=list(binding.shape))
            constants = raw_node["constants"]
            if not isinstance(constants, dict):
                reject(RK_E_SCHEMA, f"nodes[{index}].constants must be an object",
                       stage="hybrid")
            expected_constants = {spec.name for spec in plan.constants}
            if set(constants) != expected_constants:
                reject(RK_E_SCHEMA,
                       f"nodes[{index}] must declare every RKPlan constant",
                       stage="hybrid", actual=sorted(constants),
                       required=sorted(expected_constants))
            parsed_constants = {
                name: _parse_constant(value,
                                      f"nodes[{index}].constants.{name}")
                for name, value in constants.items()}
            prior = plan_constants_by_key.setdefault(plan_key,
                                                     parsed_constants)
            if prior != parsed_constants:
                reject(RK_E_SCHEMA,
                       f"plan {plan_key!r} is reused with different constants",
                       stage="hybrid")
            node: Node = NpuNode(node_id, plan_key, refs_in, refs_out,
                                 parsed_constants, dep_ids, repeat, numeric,
                                 bridge, phase, repeat_submit, group_plan_key,
                                 capture_only)
        else:
            action_id = _string(raw_node["action"],
                                f"nodes[{index}].action")
            action = HOST_ACTIONS.get(action_id)
            if action is None:
                reject(RK_E_UNSUPPORTED,
                       f"unregistered Host action {action_id!r}",
                       stage="hybrid")
            version = _int(raw_node["action_version"],
                           f"nodes[{index}].action_version", minimum=1)
            if version != action.version:
                reject(RK_E_SCHEMA,
                       f"Host action {action_id!r} version mismatch",
                       stage="hybrid", actual=version,
                       required=action.version)
            if repeat:
                reject(RK_E_UNSUPPORTED,
                       "Host action nodes may not repeat in v1; emit one node "
                       "per static Host action", stage="hybrid")
            if set(refs_in) != set(action.input_names):
                reject(RK_E_SCHEMA,
                       f"Host action {action_id!r} input names mismatch",
                       stage="hybrid", actual=sorted(refs_in),
                       required=sorted(action.input_names))
            if set(refs_out) != set(action.output_names):
                reject(RK_E_SCHEMA,
                       f"Host action {action_id!r} output names mismatch",
                       stage="hybrid", actual=sorted(refs_out),
                       required=sorted(action.output_names))
            if any(ref.index for ref in refs_in.values()) or \
                    any(ref.index for ref in refs_out.values()):
                reject(RK_E_UNSUPPORTED,
                       "Host action node references may not be indexed in v1",
                       stage="hybrid")
            attrs = raw_node["attrs"]
            if not isinstance(attrs, dict):
                reject(RK_E_SCHEMA, f"nodes[{index}].attrs must be an object",
                       stage="hybrid")
            node = HostNode(node_id, action, refs_in, refs_out, dict(attrs),
                            dep_ids, repeat, numeric, bridge, phase)

        # Register producer edges and enforce explicit dependency closure.
        referenced_nodes: set[str] = set()
        node_capture_only = getattr(node, "capture_only", False)
        for ref in node.inputs.values():
            owner = producer.get(ref.tensor)
            if owner is None:
                reject(RK_E_SCHEMA,
                       f"nodes[{index}] reads {ref.tensor!r} before it is "
                       "produced", stage="hybrid")
            if owner.startswith("$"):
                continue
            if capture_only_tensor.get(ref.tensor) and not node_capture_only:
                reject(RK_E_SCHEMA,
                       f"nodes[{index}] reads {ref.tensor!r} produced by "
                       f"capture-only node {owner!r}; capture-only values may "
                       "not feed ordinary execution", stage="hybrid")
            referenced_nodes.add(owner)
        if referenced_nodes != set(node.depends_on):
            reject(RK_E_SCHEMA,
                   f"nodes[{index}] depends_on does not equal producer edges",
                   stage="hybrid", actual=sorted(node.depends_on),
                   required=sorted(referenced_nodes))
        for ref in node.outputs.values():
            if ref.tensor in producer:
                reject(RK_E_SCHEMA,
                       f"nodes[{index}] writes tensor {ref.tensor!r} more than "
                       "once", stage="hybrid")
            producer[ref.tensor] = node_id
            capture_only_tensor[ref.tensor] = node_capture_only
        node_by_id[node_id] = node
        nodes.append(node)

    for spec in tensors.values():
        if spec.name not in producer or producer[spec.name].startswith("$"):
            reject(RK_E_SCHEMA, f"declared tensor {spec.name!r} has no producer",
                   stage="hybrid")
    outputs = obj["outputs"]
    if not isinstance(outputs, dict) or not outputs:
        reject(RK_E_SCHEMA, "outputs must be a non-empty object", stage="hybrid")
    out_map: dict[str, str] = {}
    for key, value in outputs.items():
        out_map[_string(key, "outputs key")] = _string(
            value, f"outputs.{key}")
    for name in out_map.values():
        if name not in tensors:
            reject(RK_E_SCHEMA, f"output tensor {name!r} is not declared",
                   stage="hybrid")
        if capture_only_tensor.get(name):
            reject(RK_E_SCHEMA,
                   f"output tensor {name!r} is produced by a capture-only "
                   "node; capture-only values may not be model outputs",
                   stage="hybrid")
    trace = obj["trace_tensors"]
    if not isinstance(trace, list):
        reject(RK_E_SCHEMA, "trace_tensors must be an array", stage="hybrid")
    trace_names = tuple(_string(v, f"trace_tensors[{i}]")
                        for i, v in enumerate(trace))
    if len(set(trace_names)) != len(trace_names):
        reject(RK_E_SCHEMA, "trace_tensors contains duplicates", stage="hybrid")
    for name in trace_names:
        if name not in tensors:
            reject(RK_E_SCHEMA, f"trace tensor {name!r} is not declared",
                   stage="hybrid")
    return HybridProgram(source, _string(obj["name"], "name"),
                         _string(obj["model"], "model"),
                         _int(obj["batch"], "batch", minimum=1),
                         _string(obj["profile_hash"], "profile_hash"),
                         dict(policy), dict(bridge_policy), input_spec,
                         host_constants, tensors, plans, tuple(nodes), out_map,
                         trace_names, dict(weights))


REPEAT_SUBMIT_POLICIES = ("per_repeat_submits", "shared_group")


def _validate_group_plan(single: "Plan", group: "Plan", extents: list[int],
                         *, where: str) -> None:
    """Prove the group plan is the N-way expansion of the single-instance plan.

    The manifest may not name an arbitrary second plan: the group must carry
    exactly 3N bindings named ``<input>_<i>`` / ``<output>_<i>`` with the same
    shapes as the single plan's bindings, and N must equal the declared repeat
    product.  The C verifier independently re-checks the chained task structure.
    """
    nrepeat = 1
    for extent in extents:
        nrepeat *= int(extent)
    single_names = [b.name for b in single.bindings]
    single_shapes = {b.name: tuple(b.shape) for b in single.bindings}
    expected = []
    for i in range(nrepeat):
        for name in single_names:
            expected.append(f"{name}_{i}")
    got = [b.name for b in group.bindings]
    if got != expected:
        reject(RK_E_SHAPE,
               f"{where}.group_plan bindings are not the {nrepeat}-way expansion "
               "of the single plan", stage="hybrid", actual=got[:8],
               required=expected[:8])
    for i in range(nrepeat):
        for name in single_names:
            binding = next(b for b in group.bindings if b.name == f"{name}_{i}")
            if tuple(binding.shape) != single_shapes[name]:
                reject(RK_E_SHAPE,
                       f"{where}.group_plan tile {name}_{i} shape differs from "
                       "the single plan", stage="hybrid",
                       actual=list(binding.shape),
                       required=list(single_shapes[name]))
    if group.accounting.task_count != nrepeat or \
            group.accounting.submit_count != 1:
        reject(RK_E_SCHEMA,
               f"{where}.group_plan must be one submit of {nrepeat} tasks",
               stage="hybrid",
               actual=[group.accounting.task_count,
                       group.accounting.submit_count])


def _shape_after_ref_obj(*, program_input: TensorSpec,
                         host_constants: dict[str, HostConstant],
                         tensors: dict[str, TensorSpec],
                         ref: TensorRef, where: str) -> tuple[int, ...]:
    spec = tensors.get(ref.tensor)
    if spec is None and ref.tensor in host_constants:
        spec = host_constants[ref.tensor].spec
    if spec is None and ref.tensor == program_input.name:
        spec = program_input
    if spec is None:
        reject(RK_E_SCHEMA, f"{where} references unknown tensor {ref.tensor!r}",
               stage="hybrid")
    if len(ref.index) > len(spec.shape):
        reject(RK_E_SHAPE, f"{where} has too many indices", stage="hybrid")
    return spec.shape[len(ref.index):]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _materialize_constant(desc: dict[str, Any], plan_spec: Any,
                          weights: dict[str, np.ndarray],
                          weights_meta: dict[str, Any]) -> np.ndarray:
    target = tuple(plan_spec.shape)
    if desc["kind"] == "generated":
        if plan_spec.generator != desc["generator"]:
            reject(RK_E_SCHEMA,
                   f"generated constant {plan_spec.name!r} has mismatched "
                   "generator", stage="hybrid")
        if desc["generator"] == "identity":
            return np.eye(target[0], dtype=np.float16)
        return np.zeros(target, dtype=np.float16)
    if desc["kind"] == "weight_concat":
        parts = []
        for part in desc["parts"]:
            if part not in weights:
                reject(RK_E_SCHEMA, f"missing weight {part!r}", stage="hybrid")
            meta = weights_meta["keys"].get(part)
            value = np.asarray(weights[part])
            if meta is None or tuple(meta["shape"]) != tuple(value.shape) or \
                    str(np.dtype(meta["dtype"])) != str(value.dtype):
                reject(RK_E_SCHEMA, f"weight {part!r} metadata mismatch",
                       stage="hybrid")
            parts.append(value.astype(np.float16))
        arr = np.concatenate(parts, axis=int(desc["axis"]))
        if arr.shape != target:
            reject(RK_E_SHAPE, f"concatenated constant {plan_spec.name!r} "
                   "shape mismatch", stage="hybrid", actual=list(arr.shape),
                   required=list(target))
        return np.ascontiguousarray(arr, dtype=np.float16)
    name = desc["name"]
    if name not in weights:
        reject(RK_E_SCHEMA, f"missing weight {name!r}", stage="hybrid")
    source = np.asarray(weights[name])
    meta = weights_meta["keys"].get(name)
    if meta is None or tuple(meta["shape"]) != tuple(source.shape) or \
            str(np.dtype(meta["dtype"])) != str(source.dtype):
        reject(RK_E_SCHEMA, f"weight {name!r} metadata mismatch",
               stage="hybrid", actual={"shape": list(source.shape),
                                       "dtype": str(source.dtype)})
    arr = source.astype(np.float16)
    if desc["kind"] == "weight_repeat_pad":
        reshape = tuple(desc["reshape"])
        if int(np.prod(arr.shape)) != int(np.prod(reshape)):
            reject(RK_E_SHAPE, f"weight {name!r} reshape changes element count",
                   stage="hybrid", actual=list(arr.shape),
                   required=list(reshape))
        arr = arr.reshape(reshape)
        repeat_rows = int(desc.get("repeat_rows", 1))
        if arr.shape[0] * repeat_rows > target[0] or arr.shape[1] > target[1]:
            reject(RK_E_SHAPE, f"weight {name!r} repeat/pad exceeds target",
                   stage="hybrid", actual=list(arr.shape),
                   required=list(target))
        if repeat_rows > 1:
            arr = np.tile(arr, (repeat_rows, 1))
    if arr.shape != target:
        if arr.shape[0] > target[0] or arr.shape[1] > target[1] or \
                desc["kind"] != "weight_repeat_pad":
            reject(RK_E_SHAPE, f"constant {plan_spec.name!r} shape mismatch",
                   stage="hybrid", actual=list(arr.shape),
                   required=list(target))
        padded = np.zeros(target, dtype=np.float16)
        padded[:arr.shape[0], :arr.shape[1]] = arr
        arr = padded
    return np.ascontiguousarray(arr, dtype=np.float16)


class HybridRunner:
    """Execute one frozen rkhybrid graph with plan reuse and explicit stats."""

    def __init__(self, lib: LibRKPlan, program: HybridProgram,
                 weights: dict[str, np.ndarray], *,
                 weights_path: str | Path | None = None, fd: int = 0,
                 plan_entry: str = "attributed",
                 repeat_submit: str = "manifest"):
        if plan_entry not in ("attributed", "legacy"):
            reject(RK_E_INVALID, f"unknown plan entry {plan_entry!r}",
                   stage="model")
        if repeat_submit not in ("manifest", "separate", "b1", "shared",
                                 "chain_shared"):
            reject(RK_E_INVALID, f"unknown repeat_submit {repeat_submit!r}",
                   stage="model")
        self.plan_entry = plan_entry
        self.repeat_submit = repeat_submit
        t0 = time.perf_counter()
        self.lib = lib
        self.program = program
        self.fd = fd
        self.weights = {name: np.asarray(value)
                        for name, value in weights.items()}
        self.calls: dict[str, PlanCall] = {}
        self.host_constant_values: dict[str, np.ndarray] = {}
        self._closed = False
        self._validate_weights(weights_path)
        self._prepare_host_constants()
        try:
            for key, info in program.plans.items():
                node = next((n for n in program.nodes
                             if isinstance(n, NpuNode) and n.plan == key), None)
                if node is None:
                    # A pre-expanded group plan: no constants (the expansion
                    # re-slices tiles, it does not invent weights).
                    if info.plan.constants:
                        reject(RK_E_UNSUPPORTED,
                               f"group plan {key!r} carries constants; the shared "
                               "repeat path does not re-slice them",
                               stage="model")
                    self.calls[key] = PlanCall(lib, info.plan, {}, fd=fd,
                                               entry=self.plan_entry)
                    continue
                plan_constants = {}
                lookup = {spec.name: spec for spec in info.plan.constants}
                for name, desc in node.constants.items():
                    plan_constants[name] = _materialize_constant(
                        desc, lookup[name], self.weights, program.weights)
                self.calls[key] = PlanCall(
                    lib, info.plan, plan_constants, fd=fd,
                    entry=self.plan_entry)
        except BaseException as exc:
            cleanup_errors = self._cleanup_calls()
            if cleanup_errors:
                exc.add_note(
                    "HybridRunner construction cleanup also failed: " +
                    "; ".join(repr(e) for e in cleanup_errors))
            raise
        self.node_plans: dict[str, _NpuBindingPlan] = {}
        try:
            for node in program.nodes:
                if isinstance(node, NpuNode):
                    self.node_plans[node.id] = self._static_binding_plan(node)
        except BaseException as exc:
            cleanup_errors = self._cleanup_calls()
            if cleanup_errors:
                exc.add_note(
                    "HybridRunner static binding-plan cleanup also failed: " +
                    "; ".join(repr(e) for e in cleanup_errors))
            raise
        self.setup_us = (time.perf_counter() - t0) * 1e6

    def _static_binding_plan(self, node: NpuNode) -> _NpuBindingPlan:
        """Resolve a repeated RKPlan node once, at construction time.

        Repeat indices are materialised here instead of being rebuilt every run,
        and the shape/lifetime rules that ``_target_for_output`` re-derived on
        every call are proven once against the immutable manifest.  Anything that
        depends on the caller's data is still validated for every repeat.
        """
        info = self.program.plans[node.plan]
        bindings = {b.name: b for b in info.plan.bindings}
        input_names = tuple(b.name for b in info.plan.bindings
                            if b.role == "input")
        output_names = tuple(b.name for b in info.plan.bindings
                             if b.role == "output")
        expected_out_index = tuple(axis.name for axis in node.repeat)
        repeats = []
        for ordinal, repeat_vars in enumerate(self._repeat_bindings(node)):
            in_refs = []
            for name in input_names:
                ref = node.inputs[name]
                index = tuple(repeat_vars[token] if isinstance(token, str)
                              else token for token in ref.index)
                spec = self._tensor_spec(ref.tensor)
                suffix = tuple(spec.shape[len(index):])
                if suffix != tuple(bindings[name].shape):
                    reject(RK_E_SHAPE,
                           f"{node.id}.inputs.{name} slice shape does not match "
                           "the RKPlan input", stage="hybrid",
                           actual=list(suffix),
                           required=list(bindings[name].shape))
                in_refs.append(_RepeatRef(ref.tensor, index, None, spec.dtype))
            out_refs = []
            for name in output_names:
                ref = node.outputs[name]
                if ref.index and ref.index != expected_out_index:
                    reject(RK_E_SHAPE,
                           f"{node.id}.outputs.{name} index must be all repeat "
                           "axes", stage="hybrid", actual=list(ref.index),
                           required=list(expected_out_index))
                index = tuple(repeat_vars[token] if isinstance(token, str)
                              else token for token in ref.index)
                spec = self._tensor_spec(ref.tensor)
                suffix = tuple(spec.shape[len(index):])
                if suffix != tuple(bindings[name].shape):
                    reject(RK_E_SHAPE,
                           f"{node.id}.outputs.{name} target shape does not "
                           "match the RKPlan output", stage="hybrid",
                           actual=list(suffix), required=list(bindings[name].shape))
                # A sliced write allocates the whole destination on the first
                # repeat; a whole-tensor write allocates on every repeat so a
                # second write to the same tensor is still rejected at run time.
                allocate = None
                if not index or ordinal == 0:
                    allocate = tuple(spec.shape)
                out_refs.append(_RepeatRef(ref.tensor, index, allocate,
                                           spec.dtype))
            repeats.append((tuple(in_refs), tuple(out_refs)))
        mode = "per_repeat_submits" if len(repeats) > 1 else "independent_calls"
        flat_inputs: tuple[_RepeatRef, ...] = ()
        flat_outputs: tuple[_RepeatRef, ...] = ()
        if node.repeat_submit == "shared_group":
            if self.repeat_submit in ("separate", "b1"):
                mode = ("independent_calls" if self.repeat_submit == "separate"
                        else "per_repeat_submits")
            else:
                info_group = self.program.plans[node.group_plan]
                # The group plan's binding order is the single plan's binding
                # order repeated N times (A_0, B_0, Y_0, A_1, ...), which the
                # manifest loader already proved.  Map each group binding back to
                # the repeat instance that supplies its ref.
                per_repeat = []
                for in_refs, out_refs in repeats:
                    mapping = {name: ref for name, ref
                               in zip(input_names, in_refs)}
                    mapping.update({name: ref
                                    for name, ref in zip(output_names, out_refs)})
                    per_repeat.append(mapping)
                flat_in: list[_RepeatRef] = []
                flat_out: list[_RepeatRef] = []
                for binding in info_group.plan.bindings:
                    name = binding.name
                    base, _, index = name.rpartition("_")
                    if not index.isdigit() or int(index) >= len(per_repeat) or \
                            name != f"{base}_{int(index)}" or \
                            base not in per_repeat[int(index)]:
                        reject(RK_E_SHAPE,
                               f"{node.id}.group_plan binding {name!r} does not "
                               "correspond to a declared repeat",
                               stage="hybrid")
                    (flat_out if binding.role == "output" else flat_in).append(
                        per_repeat[int(index)][base])
                mode = "shared_group"
                flat_inputs = tuple(flat_in)
                flat_outputs = tuple(flat_out)
        elif self.repeat_submit in ("shared", "chain_shared") and node.repeat:
            reject(RK_E_UNSUPPORTED,
                   f"node {node.id!r} has no group_plan; the runner cannot force "
                   "the shared repeat path", stage="model")
        return _NpuBindingPlan(node.plan, input_names, output_names,
                               tuple(repeats), mode, flat_inputs, flat_outputs)

    def _tensor_spec(self, name: str) -> TensorSpec:
        spec = self.program.tensors.get(name)
        if spec is None:
            reject(RK_E_SCHEMA, f"binding references unknown tensor {name!r}",
                   stage="hybrid")
        return spec

    def _validate_weights(self, weights_path: str | Path | None) -> None:
        meta = self.program.weights
        if weights_path is not None:
            actual = _sha256_file(Path(weights_path))
            if actual != meta["sha256"]:
                reject(RK_E_SCHEMA, "model weights hash mismatch", stage="hybrid",
                       actual=actual, required=meta["sha256"])
        expected = meta["keys"]
        for name, spec in expected.items():
            if name not in self.weights:
                reject(RK_E_SCHEMA, f"missing model weight {name!r}",
                       stage="hybrid")
            arr = self.weights[name]
            if tuple(arr.shape) != tuple(spec["shape"]) or \
                    str(arr.dtype) != spec["dtype"]:
                reject(RK_E_SCHEMA, f"weight {name!r} shape/dtype mismatch",
                       stage="hybrid",
                       actual={"shape": list(arr.shape), "dtype": str(arr.dtype)},
                       required={"shape": spec["shape"], "dtype": spec["dtype"]})
        extra = set(self.weights) - set(expected)
        if extra:
            reject(RK_E_SCHEMA, f"unexpected model weights: {sorted(extra)}",
                   stage="hybrid")

    def _prepare_host_constants(self) -> None:
        for name, item in self.program.host_constants.items():
            if item.weight not in self.weights:
                reject(RK_E_SCHEMA, f"missing Host constant weight "
                       f"{item.weight!r}", stage="hybrid")
            arr = self.weights[item.weight]
            if tuple(arr.shape) != item.spec.shape:
                reject(RK_E_SHAPE, f"Host constant {name!r} shape mismatch",
                       stage="hybrid", actual=list(arr.shape),
                       required=list(item.spec.shape))
            self.host_constant_values[name] = np.ascontiguousarray(
                arr.astype(np.float16), dtype=np.float16)

    def _cleanup_calls(self) -> list[BaseException]:
        errors: list[BaseException] = []
        remaining: dict[str, PlanCall] = {}
        for key, call in reversed(list(self.calls.items())):
            try:
                call.close()
            except BaseException as exc:
                errors.append(exc)
                remaining[key] = call
        self.calls = remaining
        return errors

    def _resolve(self, ref: TensorRef, values: dict[str, np.ndarray],
                 repeat_vars: dict[str, int], where: str) -> np.ndarray:
        if ref.tensor not in values:
            reject(RK_E_SCHEMA, f"{where} reads unavailable tensor "
                   f"{ref.tensor!r}", stage="hybrid")
        index = tuple(repeat_vars[token] if isinstance(token, str) else token
                      for token in ref.index)
        value = values[ref.tensor]
        if index:
            value = value[index]
        return value

    @staticmethod
    def _read_ref(values: dict[str, np.ndarray], ref: _RepeatRef) -> np.ndarray:
        value = values.get(ref.tensor)
        if value is None:
            reject(RK_E_SCHEMA, f"reads unavailable tensor {ref.tensor!r}",
                   stage="hybrid")
        return value[ref.index] if ref.index else value

    @staticmethod
    def _write_ref(values: dict[str, np.ndarray],
                   ref: _RepeatRef) -> np.ndarray:
        if ref.allocate is not None:
            if not ref.index:
                if ref.tensor in values:
                    reject(RK_E_SCHEMA, f"writes tensor {ref.tensor!r} more "
                           "than once", stage="hybrid")
            target = np.empty(ref.allocate, dtype=DTYPES[ref.dtype])
            if not ref.index:
                values[ref.tensor] = target
                return target
            values[ref.tensor] = target
            return target[ref.index]
        holder = values.get(ref.tensor)
        if holder is None:
            reject(RK_E_SCHEMA, f"writes unavailable tensor {ref.tensor!r}",
                   stage="hybrid")
        return holder[ref.index]

    def _target_for_output(self, node: Node, ref: TensorRef,
                           values: dict[str, np.ndarray],
                           repeat_vars: dict[str, int], where: str) -> np.ndarray:
        spec = self.program.tensors[ref.tensor]
        if ref.index:
            if ref.tensor not in values:
                values[ref.tensor] = np.empty(spec.shape,
                                              dtype=DTYPES[spec.dtype])
            expected_index = tuple(axis.name for axis in node.repeat)
            if ref.index != expected_index:
                reject(RK_E_SHAPE,
                       f"{where} output index must be all repeat axes",
                       stage="hybrid", actual=list(ref.index),
                       required=list(expected_index))
            index = tuple(repeat_vars[token] for token in ref.index)
            target = values[ref.tensor][index]
        else:
            if ref.tensor in values:
                reject(RK_E_SCHEMA, f"{where} writes tensor more than once",
                       stage="hybrid")
            target = np.empty(spec.shape, dtype=DTYPES[spec.dtype])
            values[ref.tensor] = target
        if target.shape != tuple(spec.shape[len(ref.index):]):
            reject(RK_E_SHAPE, f"{where} target shape mismatch", stage="hybrid",
                   actual=list(target.shape),
                   required=list(spec.shape[len(ref.index):]))
        return target

    @staticmethod
    def _repeat_bindings(node: Node) -> list[dict[str, int]]:
        if not node.repeat:
            return [{}]
        out: list[dict[str, int]] = [{}]
        for axis in node.repeat:
            out = [{**prior, axis.name: value}
                   for prior in out for value in range(axis.extent)]
        return out

    def run(self, raw: np.ndarray, *, stats: RunStats | None = None,
            capture: bool = False) -> HybridResult:
        if self._closed:
            reject(RK_E_INVALID, "hybrid runner is already closed", stage="model")
        arr = np.asarray(raw)
        spec = self.program.input_spec
        if arr.shape != spec.shape:
            reject(RK_E_SHAPE, "hybrid graph input shape mismatch", stage="model",
                   actual=list(arr.shape), required=list(spec.shape))
        if arr.dtype != DTYPES[spec.dtype]:
            reject(RK_E_INVALID, "hybrid graph input dtype mismatch",
                   stage="model", actual=str(arr.dtype),
                   required=spec.dtype)
        if not arr.flags.c_contiguous:
            reject(RK_E_INVALID, "hybrid graph input must be C-contiguous",
                   stage="model")
        values: dict[str, np.ndarray] = {spec.name: arr}
        values.update(self.host_constant_values)
        phase_before_us = (stats.preprocess_us + stats.execution_us +
                           stats.postprocess_us) if stats is not None else 0.0
        t_run = time.perf_counter()
        for node in self.program.nodes:
            if isinstance(node, NpuNode) and node.capture_only and not capture:
                continue
            if isinstance(node, HostNode):
                t0 = time.perf_counter()
                inputs = {name: self._resolve(
                    ref, values, {}, f"{node.id}.inputs.{name}")
                    for name, ref in node.inputs.items()}
                outputs = node.action.func(inputs, node.attrs)
                if not isinstance(outputs, dict) or \
                        set(outputs) != set(node.action.output_names):
                    reject(RK_E_SCHEMA,
                           f"Host action {node.action.action_id!r} returned "
                           "wrong output keys", stage="host_action")
                produced: list[np.ndarray] = []
                for name, ref in node.outputs.items():
                    value = np.asarray(outputs[name])
                    target_spec = self.program.tensors[ref.tensor]
                    dtype = DTYPES[target_spec.dtype]
                    if value.dtype != np.dtype(dtype):
                        reject(RK_E_INVALID,
                               f"{node.id}.{name} dtype mismatch",
                               stage="host_action",
                               actual=str(value.dtype),
                               required=target_spec.dtype)
                    if value.shape != target_spec.shape:
                        reject(RK_E_SHAPE,
                               f"{node.id}.{name} shape mismatch",
                               stage="host_action", actual=list(value.shape),
                               required=list(target_spec.shape))
                    if not value.flags.c_contiguous:
                        value = np.ascontiguousarray(value)
                    values[ref.tensor] = value
                    produced.append(value)
                elapsed_us = (time.perf_counter() - t0) * 1e6
                if stats is not None:
                    stats.add_host_action(
                        node.id, node.phase, elapsed_us,
                        inputs=tuple(inputs.values()), outputs=tuple(produced))
            else:
                binding_plan = self.node_plans[node.id]
                repeats = binding_plan.repeats
                if binding_plan.mode == "shared_group":
                    # One pre-expanded plan, one chained submit.  Bindings are
                    # flattened in the group plan's order and every tile is still
                    # validated by PlanCall before the pointer reaches C.
                    call = self.calls[node.group_plan]
                    t_node = time.perf_counter()
                    inputs = [self._read_ref(values, ref)
                              for ref in binding_plan.flat_inputs]
                    outputs = [self._write_ref(values, ref)
                               for ref in binding_plan.flat_outputs]
                    report, workspace_us = call.run_into(inputs, outputs)
                    node_us = (time.perf_counter() - t_node) * 1e6
                    if stats is not None:
                        stats.add_report(node.id, report, node_us,
                                         phase=node.phase,
                                         workspace_us=workspace_us)
                    continue
                call = self.calls[node.plan]
                if binding_plan.mode == "per_repeat_submits" and len(repeats) > 1:
                    # Static repeat grouping: the manifest declared N independent
                    # instances of one immutable plan, so they run through one
                    # native entry.  The device submit count is unchanged and every
                    # per-repeat check still runs; only the Host transition, the
                    # per-repeat guard pass and the per-repeat scratch allocation
                    # are shared.
                    t_node = time.perf_counter()
                    inputs = [[self._read_ref(values, ref) for ref in in_refs]
                              for in_refs, _ in repeats]
                    outputs = [[self._write_ref(values, ref) for ref in out_refs]
                               for _, out_refs in repeats]
                    report, workspace_us = call.run_group_into(inputs, outputs)
                    node_us = (time.perf_counter() - t_node) * 1e6
                    if stats is not None:
                        stats.add_group_report(node.id, report, node_us,
                                               repeats=len(repeats),
                                               phase=node.phase,
                                               workspace_us=workspace_us)
                    continue
                # independent_calls: one PlanCall per repeat (the original
                # path, kept as the explicit control for both group policies).
                for in_refs, out_refs in repeats:
                    t_node = time.perf_counter()
                    inputs = [self._read_ref(values, ref) for ref in in_refs]
                    outputs = [self._write_ref(values, ref) for ref in out_refs]
                    report, workspace_us = call.run_into(inputs, outputs)
                    node_us = (time.perf_counter() - t_node) * 1e6
                    if stats is not None:
                        stats.add_report(node.id, report, node_us,
                                         phase=node.phase,
                                         workspace_us=workspace_us)
        public_outputs: dict[str, np.ndarray] = {}
        for public, tensor in self.program.outputs.items():
            if tensor not in values:
                reject(RK_E_SCHEMA, f"graph output {tensor!r} was not produced",
                       stage="hybrid")
            value = values[tensor]
            spec = self.program.tensors[tensor]
            if value.shape != spec.shape or value.dtype != np.dtype(DTYPES[spec.dtype]):
                reject(RK_E_SHAPE, f"graph output {tensor!r} invalid",
                       stage="hybrid")
            if not np.isfinite(value).all():
                reject(RK_E_NUMERIC_CONTRACT,
                       f"graph output {tensor!r} contains non-finite values",
                       stage="model")
            public_outputs[public] = value
        trace = {}
        if capture:
            for name in self.program.trace_tensors:
                if name in values:
                    trace[name] = np.array(values[name], copy=True)
        if stats is not None:
            elapsed_us = (time.perf_counter() - t_run) * 1e6
            phase_now_us = (stats.preprocess_us + stats.execution_us +
                            stats.postprocess_us)
            residual_us = max(0.0, elapsed_us -
                              (phase_now_us - phase_before_us))
            if residual_us > 0.0:
                stats.add_executor_overhead(residual_us)
            stats.record_run(elapsed_us)
        return HybridResult(public_outputs, trace)

    def close(self) -> None:
        if self._closed:
            return
        errors = self._cleanup_calls()
        if not errors:
            self._closed = True
        if errors:
            raise errors[0]

    def __enter__(self) -> "HybridRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                exc.add_note("HybridRunner cleanup also failed: " +
                             repr(cleanup))
                return False
            raise
        return False
