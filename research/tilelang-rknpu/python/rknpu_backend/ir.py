# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen RKIR data model for the static FP16 rank-2 MVP.

The model intentionally contains no device handles, fd, IOVA, TVM objects or Python
callables.  It is JSON-persisted with a versioned schema and strict decoding.  The
first version accepts only the opcodes listed in :data:`SUPPORTED_OPS`; anything
else is rejected before any lowering or device work.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .errors import (RK_E_INVALID, RK_E_SCHEMA, RK_E_SHAPE,
                     RK_E_UNSUPPORTED, reject)

SCHEMA_VERSION = 1
KIND = "rkir"
DTYPE_F16 = "float16"
SUPPORTED_DTYPES = {DTYPE_F16}
SUPPORTED_OPS = {"matmul", "add", "mul", "silu", "gelu", "sigmoid", "hardsigmoid"}
UNARY_OPS = {"silu", "gelu", "sigmoid", "hardsigmoid"}
BINARY_OPS = {"add", "mul"}
MAX_EXTENT = 4096


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_tuple(value: Any, name: str, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        reject(RK_E_SCHEMA, f"{name} must be an array", stage="schema")
    out = tuple(value)
    if length is not None and len(out) != length:
        reject(RK_E_SCHEMA, f"{name} must have length {length}", stage="schema")
    for item in out:
        if not _is_int(item) or item < 0:
            reject(RK_E_SCHEMA, f"{name} entries must be non-negative integers",
                   stage="schema")
    return out


def _check_keys(obj: dict[str, Any], required: Iterable[str], optional: Iterable[str],
                where: str) -> None:
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - obj.keys())
    unknown = sorted(obj.keys() - required_set - optional_set)
    if missing:
        reject(RK_E_SCHEMA, f"{where} missing fields: {', '.join(missing)}",
               stage="schema")
    if unknown:
        reject(RK_E_SCHEMA, f"{where} has unknown fields: {', '.join(unknown)}",
               stage="schema")


@dataclass(frozen=True)
class TensorType:
    """Logical tensor type.  MVP tensors are static rank-2 FP16 matrices."""

    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...] = ("row", "col")
    stride: tuple[int, ...] | None = None
    index_map: str = "identity"
    finite: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            reject(RK_E_UNSUPPORTED, f"unsupported dtype {self.dtype!r}",
                   stage="infer", actual=self.dtype, required=sorted(SUPPORTED_DTYPES))
        if len(self.shape) != 2:
            reject(RK_E_SHAPE, "MVP tensors must be rank 2", stage="infer",
                   actual=list(self.shape), required=2)
        for dim in self.shape:
            if not _is_int(dim) or dim <= 0:
                reject(RK_E_SHAPE, "tensor dimensions must be static positive integers",
                       stage="infer", actual=list(self.shape))
            if dim > MAX_EXTENT:
                reject(RK_E_SHAPE, f"tensor extent {dim} exceeds {MAX_EXTENT}",
                       stage="infer", actual=dim, required=f"<= {MAX_EXTENT}")
        if len(self.axes) != 2 or any(not isinstance(a, str) or not a for a in self.axes):
            reject(RK_E_SCHEMA, "axes must contain two non-empty strings", stage="schema")
        if self.stride is not None:
            if len(self.stride) != 2 or any(not _is_int(v) for v in self.stride):
                reject(RK_E_SCHEMA, "stride must contain two integers", stage="schema")
            if any(v < 0 for v in self.stride):
                reject(RK_E_UNSUPPORTED, "negative strides are not supported",
                       stage="infer", actual=list(self.stride))
        if self.index_map != "identity":
            reject(RK_E_UNSUPPORTED,
                   f"index map {self.index_map!r} is not supported", stage="infer")

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "index_map": self.index_map,
            "finite": self.finite,
        }
        if self.stride is not None:
            out["stride"] = list(self.stride)
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> "TensorType":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "TensorType must be an object", stage="schema")
        _check_keys(obj, {"dtype", "shape", "axes", "index_map", "finite"},
                    {"stride"}, "TensorType")
        stride = obj.get("stride")
        if stride is not None:
            stride = _int_tuple(stride, "TensorType.stride", 2)
        return cls(
            dtype=obj["dtype"],
            shape=_int_tuple(obj["shape"], "TensorType.shape", 2),
            axes=tuple(obj["axes"]) if isinstance(obj["axes"], list) else (),
            stride=stride,
            index_map=obj["index_map"],
            finite=obj["finite"],
        )


@dataclass(frozen=True)
class NumericContract:
    """Requested arithmetic semantics, not a promise made by the hardware."""

    input_dtype: str = DTYPE_F16
    output_dtype: str = DTYPE_F16
    accumulator_dtype: str = "float32"
    accumulation: str = "strict_unspecified"
    narrowing_points: tuple[str, ...] = ()
    approximate: bool = False
    tolerance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dtype not in SUPPORTED_DTYPES or self.output_dtype not in SUPPORTED_DTYPES:
            reject(RK_E_NUMERIC_CONTRACT, "MVP numeric contracts require FP16 I/O",
                   stage="infer")
        if self.accumulator_dtype not in {"float16", "float32"}:
            reject(RK_E_NUMERIC_CONTRACT,
                   f"unknown accumulator dtype {self.accumulator_dtype!r}",
                   stage="infer")
        if self.accumulation not in {
            "strict_unspecified", "ordered_k_tiles_with_f16_writeback"
        }:
            reject(RK_E_NUMERIC_CONTRACT,
                   f"unknown accumulation policy {self.accumulation!r}", stage="infer")
        for key, value in self.tolerance.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                reject(RK_E_NUMERIC_CONTRACT,
                       "tolerance values must be non-negative finite numbers",
                       stage="infer")

    @property
    def allows_f16_tiled_path(self) -> bool:
        return (self.accumulator_dtype == "float16" and
                self.accumulation == "ordered_k_tiles_with_f16_writeback" and
                "k_tile_boundary" in self.narrowing_points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "accumulation": self.accumulation,
            "narrowing_points": list(self.narrowing_points),
            "approximate": self.approximate,
            "tolerance": dict(sorted(self.tolerance.items())),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "NumericContract":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "NumericContract must be an object", stage="schema")
        _check_keys(obj, {"input_dtype", "output_dtype", "accumulator_dtype",
                          "accumulation", "narrowing_points", "approximate",
                          "tolerance"}, set(), "NumericContract")
        return cls(
            input_dtype=obj["input_dtype"], output_dtype=obj["output_dtype"],
            accumulator_dtype=obj["accumulator_dtype"],
            accumulation=obj["accumulation"],
            narrowing_points=tuple(obj["narrowing_points"]),
            approximate=obj["approximate"],
            tolerance=dict(obj["tolerance"]),
        )


@dataclass(frozen=True)
class Value:
    id: int
    name: str
    type: TensorType
    producer: int | None
    role: str

    ROLES = {"input", "constant", "temporary", "output", "state"}

    def __post_init__(self) -> None:
        if not _is_int(self.id) or self.id < 0:
            reject(RK_E_SCHEMA, "Value.id must be a non-negative integer", stage="schema")
        if not isinstance(self.name, str) or not self.name:
            reject(RK_E_SCHEMA, "Value.name must be non-empty", stage="schema")
        if self.producer is not None and (not _is_int(self.producer) or self.producer < 0):
            reject(RK_E_SCHEMA, "Value.producer must be null or a non-negative integer",
                   stage="schema")
        if self.role not in self.ROLES:
            reject(RK_E_SCHEMA, f"unknown Value.role {self.role!r}", stage="schema")
        if self.role == "state":
            reject(RK_E_UNSUPPORTED, "state is not supported by the MVP",
                   stage="infer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.to_dict(),
            "producer": self.producer,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "Value":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "Value must be an object", stage="schema")
        _check_keys(obj, {"id", "name", "type", "producer", "role"}, set(), "Value")
        return cls(obj["id"], obj["name"], TensorType.from_dict(obj["type"]),
                   obj["producer"], obj["role"])


@dataclass(frozen=True)
class Op:
    id: int
    opcode: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    side_effects: bool = False

    def __post_init__(self) -> None:
        if not _is_int(self.id) or self.id < 0:
            reject(RK_E_SCHEMA, "Op.id must be a non-negative integer", stage="schema")
        if self.opcode not in SUPPORTED_OPS:
            reject(RK_E_UNSUPPORTED, f"unsupported opcode {self.opcode!r}",
                   stage="frontend", op=self.opcode, source=self.source)
        if len(self.outputs) != 1:
            reject(RK_E_UNSUPPORTED, "MVP ops must have exactly one result",
                   stage="frontend", op=self.opcode, source=self.source)
        if self.opcode == "matmul":
            if len(self.inputs) != 2:
                reject(RK_E_SHAPE, "matmul needs exactly two inputs",
                       stage="infer", op=self.opcode, source=self.source)
            b_layout = self.attrs.get("b_layout", "kn")
            if b_layout not in {"kn", "nk"}:
                reject(RK_E_UNSUPPORTED, "matmul b_layout must be 'kn' or 'nk'",
                       stage="infer", op=self.opcode, source=self.source)
        elif self.opcode in UNARY_OPS:
            if len(self.inputs) != 1:
                reject(RK_E_SHAPE, f"{self.opcode} needs exactly one input",
                       stage="infer", op=self.opcode, source=self.source)
            if self.attrs:
                reject(RK_E_UNSUPPORTED, f"{self.opcode} attributes are not supported",
                       stage="infer", op=self.opcode, source=self.source)
        elif self.opcode in BINARY_OPS:
            if len(self.inputs) != 2:
                reject(RK_E_SHAPE, f"{self.opcode} needs exactly two inputs",
                       stage="infer", op=self.opcode, source=self.source)
            if self.attrs:
                reject(RK_E_UNSUPPORTED,
                       "broadcast/general elementwise attributes are not supported",
                       stage="infer", op=self.opcode, source=self.source)
        if self.side_effects:
            reject(RK_E_UNSUPPORTED, "side-effecting ops are not supported by the MVP",
                   stage="frontend", op=self.opcode, source=self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "opcode": self.opcode,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _jsonable_attrs(self.attrs),
            "source": self.source,
            "side_effects": self.side_effects,
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "Op":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "Op must be an object", stage="schema")
        _check_keys(obj, {"id", "opcode", "inputs", "outputs", "attrs", "source",
                          "side_effects"}, set(), "Op")
        return cls(
            id=obj["id"], opcode=obj["opcode"],
            inputs=tuple(_int_tuple(obj["inputs"], "Op.inputs")),
            outputs=tuple(_int_tuple(obj["outputs"], "Op.outputs")),
            attrs=dict(obj["attrs"]), source=obj["source"],
            side_effects=obj["side_effects"],
        )


def _jsonable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted(attrs.items()):
        if isinstance(value, NumericContract):
            out[key] = value.to_dict()
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class RuntimeParam:
    name: str
    dtype: str
    shape: tuple[int, ...]
    binding: str
    lifetime: str = "call"
    check: str = "finite"

    def __post_init__(self) -> None:
        if self.binding not in {"input", "constant", "scalar"}:
            reject(RK_E_SCHEMA, f"unknown runtime binding role {self.binding!r}",
                   stage="schema")
        if self.check not in {"finite", "caller_precondition", "none"}:
            reject(RK_E_SCHEMA, f"unknown check policy {self.check!r}", stage="schema")


@dataclass(frozen=True)
class TensorView:
    """A logical view description, never a device address."""

    buffer: str
    offset: int
    shape: tuple[int, ...]
    dtype: str = DTYPE_F16

    def __post_init__(self) -> None:
        if self.offset % 2:
            reject(RK_E_SHAPE, "RKIR tensor views must be element-aligned", stage="infer")


@dataclass(frozen=True)
class State:
    state_in: str
    state_out: str
    version: int
    submit_group: str
    concurrency: str = "exclusive"

    def __post_init__(self) -> None:
        reject(RK_E_UNSUPPORTED, "state versions are not supported by the MVP",
               stage="infer")


@dataclass(frozen=True)
class Module:
    name: str
    values: tuple[Value, ...]
    ops: tuple[Op, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    profile_hash: str = ""
    source_hash: str = ""
    state_signature: str = ""
    schema_version: int = SCHEMA_VERSION
    kind: str = KIND

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            reject(RK_E_SCHEMA, f"unsupported RKIR schema version {self.schema_version}",
                   stage="schema")
        if self.kind != KIND:
            reject(RK_E_SCHEMA, f"expected kind {KIND!r}, got {self.kind!r}",
                   stage="schema")
        if not self.name:
            reject(RK_E_SCHEMA, "Module.name must be non-empty", stage="schema")
        if len({v.id for v in self.values}) != len(self.values):
            reject(RK_E_SCHEMA, "duplicate Value IDs", stage="schema")
        if [v.id for v in self.values] != list(range(len(self.values))):
            reject(RK_E_SCHEMA, "Value IDs must be contiguous from 0", stage="schema")
        if len({op.id for op in self.ops}) != len(self.ops):
            reject(RK_E_SCHEMA, "duplicate Op IDs", stage="schema")
        if [op.id for op in self.ops] != list(range(len(self.ops))):
            reject(RK_E_SCHEMA, "Op IDs must be contiguous from 0", stage="schema")
        by_id = {v.id: v for v in self.values}
        for vid in self.inputs:
            if vid not in by_id or by_id[vid].role != "input":
                reject(RK_E_SHAPE, f"module input value {vid} is not an input",
                       stage="infer")
        if len(set(self.inputs)) != len(self.inputs):
            reject(RK_E_SHAPE, "duplicate module inputs", stage="infer")
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            reject(RK_E_SHAPE, "module outputs must be non-empty and unique",
                   stage="infer")
        for vid in self.outputs:
            if vid not in by_id or by_id[vid].role != "output":
                reject(RK_E_SHAPE, f"module output value {vid} is not an output",
                       stage="infer")
            if by_id[vid].producer is None:
                reject(RK_E_SHAPE,
                       f"module output value {vid} has no producer operation",
                       stage="infer")
        for op in self.ops:
            for vid in op.inputs:
                val = by_id.get(vid)
                if val is None:
                    reject(RK_E_SHAPE, f"op {op.id} references missing value {vid}",
                           stage="infer", op=op.opcode, source=op.source)
                if val.producer is not None and val.producer >= op.id:
                    reject(RK_E_SHAPE,
                           f"op {op.id} consumes value {vid} before its producer",
                           stage="infer", op=op.opcode, source=op.source)
            for vid in op.outputs:
                val = by_id.get(vid)
                if val is None or val.producer != op.id:
                    reject(RK_E_SHAPE,
                           f"op {op.id} result value {vid} has inconsistent producer",
                           stage="infer", op=op.opcode, source=op.source)
        produced = {v.producer for v in self.values if v.producer is not None}
        if produced != set(range(len(self.ops))):
            reject(RK_E_SHAPE, "every operation must produce one value", stage="infer")
        self._infer_and_verify()

    def _infer_and_verify(self) -> None:
        by_id = {v.id: v for v in self.values}
        for op in self.ops:
            out = by_id[op.outputs[0]]
            if op.opcode == "matmul":
                a, b = by_id[op.inputs[0]], by_id[op.inputs[1]]
                if a.type.dtype != DTYPE_F16 or b.type.dtype != DTYPE_F16:
                    reject(RK_E_NUMERIC_CONTRACT, "matmul inputs must be FP16",
                           stage="infer", op=op.opcode, source=op.source)
                b_layout = op.attrs.get("b_layout", "kn")
                expected_b = (a.type.cols, None) if b_layout == "kn" else (None, a.type.cols)
                if b_layout == "kn":
                    if b.type.shape != (a.type.cols, out.type.cols):
                        reject(RK_E_SHAPE,
                               "matmul B must have shape [K,N] for b_layout='kn'",
                               stage="infer", op=op.opcode, source=op.source,
                               actual=list(b.type.shape))
                else:
                    if b.type.shape != (out.type.cols, a.type.cols):
                        reject(RK_E_SHAPE,
                               "matmul B must have shape [N,K] for b_layout='nk'",
                               stage="infer", op=op.opcode, source=op.source,
                               actual=list(b.type.shape))
                if out.type.shape != (a.type.rows, b.type.shape[1] if b_layout == "kn"
                                      else b.type.shape[0]):
                    reject(RK_E_SHAPE, "matmul result shape mismatch", stage="infer",
                           op=op.opcode, source=op.source)
                if "numeric_contract" not in op.attrs:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "matmul must carry an explicit NumericContract",
                           stage="infer", op=op.opcode, source=op.source)
                NumericContract.from_dict(op.attrs["numeric_contract"])
            elif op.opcode in UNARY_OPS:
                x = by_id[op.inputs[0]]
                if x.type != out.type:
                    reject(RK_E_SHAPE, f"{op.opcode} requires identical input/output type",
                           stage="infer", op=op.opcode, source=op.source)
            elif op.opcode in BINARY_OPS:
                a, b = by_id[op.inputs[0]], by_id[op.inputs[1]]
                if a.type != b.type or a.type != out.type:
                    reject(RK_E_UNSUPPORTED,
                           "MVP binary ops require exact same-shape operands; "
                           "broadcast is rejected",
                           stage="infer", op=op.opcode, source=op.source)

    def value(self, value_id: int) -> Value:
        if value_id < 0 or value_id >= len(self.values):
            reject(RK_E_SHAPE, f"missing value {value_id}", stage="infer")
        return self.values[value_id]

    def op(self, op_id: int) -> Op:
        if op_id < 0 or op_id >= len(self.ops):
            reject(RK_E_SHAPE, f"missing op {op_id}", stage="infer")
        return self.ops[op_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "profile_hash": self.profile_hash,
            "source_hash": self.source_hash,
            "state_signature": self.state_signature,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "values": [v.to_dict() for v in self.values],
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "Module":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "Module must be an object", stage="schema")
        _check_keys(obj, {"schema_version", "kind", "name", "profile_hash",
                          "source_hash", "state_signature", "inputs", "outputs",
                          "values", "ops"}, set(), "Module")
        return cls(
            name=obj["name"],
            values=tuple(Value.from_dict(v) for v in obj["values"]),
            ops=tuple(Op.from_dict(o) for o in obj["ops"]),
            inputs=tuple(_int_tuple(obj["inputs"], "Module.inputs")),
            outputs=tuple(_int_tuple(obj["outputs"], "Module.outputs")),
            profile_hash=obj["profile_hash"],
            source_hash=obj["source_hash"],
            state_signature=obj["state_signature"],
            schema_version=obj["schema_version"],
            kind=obj["kind"],
        )


def matmul_module(*, m: int, k: int, n: int, name: str = "matmul",
                  b_layout: str = "kn", activation: str | None = None,
                  numeric_mode: str = "rk_f16_tiled",
                  source: str = "rk-builder:matmul") -> Module:
    """Build a tiny, explicit RKIR module for the closed MVP path.

    This is a debug/offline construction helper, not a TileLang frontend.  It makes
    the semantic contract visible to tests and examples while the optional real
    TileLang extractor is unavailable.
    """
    if activation is not None and activation not in UNARY_OPS:
        reject(RK_E_UNSUPPORTED, f"activation {activation!r} is not supported",
               stage="legalize")
    a_ty = TensorType(DTYPE_F16, (m, k))
    b_shape = (k, n) if b_layout == "kn" else (n, k)
    b_ty = TensorType(DTYPE_F16, b_shape)
    y_ty = TensorType(DTYPE_F16, (m, n))
    a = Value(0, "A", a_ty, None, "input")
    b = Value(1, "B", b_ty, None, "constant")
    values = [a, b]
    ops: list[Op] = []
    if numeric_mode == "strict":
        contract = NumericContract(
            accumulator_dtype="float32", accumulation="strict_unspecified",
            narrowing_points=(), approximate=False,
            tolerance={"atol": 1e-3, "rtol": 1e-3})
    elif numeric_mode == "rk_f16_tiled":
        contract = NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=("k_tile_boundary",), approximate=False,
            tolerance={"atol": 8e-3, "rtol": 8e-3})
    else:
        reject(RK_E_NUMERIC_CONTRACT, f"unknown numeric mode {numeric_mode!r}",
               stage="frontend")
    matmul_out_role = "temporary" if activation else "output"
    c = Value(2, "Z" if activation else "Y", y_ty, 0, matmul_out_role)
    values.append(c)
    ops.append(Op(0, "matmul", (0, 1), (2,),
                  {"b_layout": b_layout,
                   "numeric_contract": contract.to_dict()}, source))
    if activation:
        out = Value(3, "Y", y_ty, 1, "output")
        values.append(out)
        ops.append(Op(1, activation, (2,), (3,), {}, source))
        outputs = (3,)
        final = 3
    else:
        outputs = (2,)
        final = 2
    return Module(name=name, values=tuple(values), ops=tuple(ops),
                  inputs=(0,), outputs=outputs, source_hash="builder",
                  state_signature="")
