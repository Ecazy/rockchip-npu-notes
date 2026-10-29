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
from .immutable import freeze, thaw

SCHEMA_VERSION = 1
KIND = "rkir"
DTYPE_F16 = "float16"
SUPPORTED_DTYPES = {DTYPE_F16}
SUPPORTED_OPS = {"matmul", "add", "mul", "relu", "silu", "gelu", "sigmoid",
                 "hardsigmoid", "rsqrt"}
UNARY_OPS = {"relu", "silu", "gelu", "sigmoid", "hardsigmoid", "rsqrt"}
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
        if len(self.shape) > 2:
            reject(RK_E_SHAPE, "MVP tensors must be rank 0, 1 or 2", stage="infer",
                   actual=list(self.shape), required="<= 2")
        for dim in self.shape:
            if not _is_int(dim) or dim <= 0:
                reject(RK_E_SHAPE, "tensor dimensions must be static positive integers",
                       stage="infer", actual=list(self.shape))
            if dim > MAX_EXTENT:
                reject(RK_E_SHAPE, f"tensor extent {dim} exceeds {MAX_EXTENT}",
                       stage="infer", actual=dim, required=f"<= {MAX_EXTENT}")
        if len(self.axes) != len(self.shape) or \
                any(not isinstance(a, str) or not a for a in self.axes):
            reject(RK_E_SCHEMA,
                   "axes must contain one non-empty string per tensor rank",
                   stage="schema", actual=list(self.axes))
        if self.stride is not None:
            if len(self.stride) != len(self.shape) or \
                    any(not _is_int(v) for v in self.stride):
                reject(RK_E_SCHEMA,
                       "stride must contain one integer per tensor rank",
                       stage="schema")
            if any(v < 0 for v in self.stride):
                reject(RK_E_UNSUPPORTED, "negative strides are not supported",
                       stage="infer", actual=list(self.stride))
            # The lowering packs dense row-major tiles only.  A logical stride that
            # is not ``(cols, 1)`` would silently change which elements are copied
            # and contracted.  Rank-0/1 broadcast constants are unpacked by the
            # explicit Host upload path, so they must not claim a strided view.
            if len(self.shape) == 2 and tuple(self.stride) != (self.shape[1], 1):
                reject(RK_E_UNSUPPORTED,
                       "only dense row-major strides are lowered; a strided "
                       "tensor view would change the packing", stage="infer",
                       actual=list(self.stride), required=[self.shape[1], 1])
            if len(self.shape) < 2 and any(v != 1 for v in self.stride):
                reject(RK_E_UNSUPPORTED,
                       "broadcast constants require a contiguous scalar/channel "
                       "source", stage="infer", actual=list(self.stride))
        if self.index_map != "identity":
            reject(RK_E_UNSUPPORTED,
                   f"index map {self.index_map!r} is not supported", stage="infer")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def rows(self) -> int:
        if self.rank != 2:
            reject(RK_E_SHAPE, "rows is only defined for rank-2 tensors",
                   stage="infer", actual=list(self.shape))
        return self.shape[0]

    @property
    def cols(self) -> int:
        if self.rank != 2:
            reject(RK_E_SHAPE, "cols is only defined for rank-2 tensors",
                   stage="infer", actual=list(self.shape))
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
            shape=_int_tuple(obj["shape"], "TensorType.shape"),
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
        if self.role != "constant" and self.type.rank != 2:
            reject(RK_E_SHAPE,
                   "only constants may be rank-0/1 broadcast sources; runtime "
                   "tensors and device values are rank 2", stage="infer",
                   actual={"role": self.role, "shape": list(self.type.shape)})

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


def _check_elementwise_attrs(op: "Op", allowed: str) -> None:
    unknown = sorted(set(op.attrs) - {allowed})
    if unknown:
        reject(RK_E_UNSUPPORTED,
               "broadcast/general elementwise attributes are not supported",
               stage="infer", op=op.opcode, source=op.source,
               actual=unknown)
    if allowed in op.attrs:
        contract = op.attrs[allowed]
        if not isinstance(contract, dict):
            reject(RK_E_SCHEMA, f"{allowed} must be an object", stage="schema",
                   op=op.opcode, source=op.source)
        NumericContract.from_dict(contract)


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
            _check_elementwise_attrs(self, "numeric_contract")
        elif self.opcode in BINARY_OPS:
            if len(self.inputs) != 2:
                reject(RK_E_SHAPE, f"{self.opcode} needs exactly two inputs",
                       stage="infer", op=self.opcode, source=self.source)
            # Broadcast and other arithmetic attributes remain unsupported; a
            # NumericContract is metadata about explicit FP16 narrowing points,
            # not an arithmetic attribute.
            _check_elementwise_attrs(self, "numeric_contract")
        if self.side_effects:
            reject(RK_E_UNSUPPORTED, "side-effecting ops are not supported by the MVP",
                   stage="frontend", op=self.opcode, source=self.source)
        # Same boundary as plan.Action: copy the caller's mapping and make the
        # result read-only, so a recognized graph cannot be edited behind the
        # planner's back.
        object.__setattr__(self, "attrs", freeze(dict(self.attrs)))

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
            out[key] = thaw(value)
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
                if a.type.rank != 2 or b.type.rank != 2 or out.type.rank != 2:
                    reject(RK_E_SHAPE,
                           "matmul operands and result must be rank 2",
                           stage="infer", op=op.opcode, source=op.source)
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
                if a.type == b.type == out.type:
                    continue
                # Compile-time broadcast is allowed only when exactly one
                # operand is a constant rank-0 scalar or rank-1 [N] channel
                # vector and the other operand is the exact output tensor.
                constant_side = None
                tensor_side = None
                for value, other in ((a, b), (b, a)):
                    if value.role == "constant" and value.type.rank < 2:
                        if constant_side is not None:
                            reject(RK_E_UNSUPPORTED,
                                   "two broadcast constants need an explicit "
                                   "materialised tensor", stage="infer",
                                   op=op.opcode, source=op.source)
                        constant_side, tensor_side = value, other
                if constant_side is None or tensor_side is None or \
                        tensor_side.type.rank != 2 or tensor_side.type != out.type:
                    reject(RK_E_UNSUPPORTED,
                           "MVP binary ops accept exact same-shape operands or "
                           "one explicit rank-0/rank-1 constant broadcast",
                           stage="infer", op=op.opcode, source=op.source,
                           actual={"a": list(a.type.shape),
                                   "b": list(b.type.shape),
                                   "out": list(out.type.shape)})
                if constant_side.type.rank == 1 and \
                        constant_side.type.shape[0] != tensor_side.type.cols:
                    reject(RK_E_SHAPE,
                           "per-channel constant length must equal the tensor "
                           "column count", stage="infer", op=op.opcode,
                           source=op.source,
                           actual=list(constant_side.type.shape),
                           required=tensor_side.type.cols)

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


def swiglu_module(*, m: int, d: int, h: int, name: str = "swiglu",
                  numeric_mode: str = "rk_f16_tiled",
                  source: str = "rk-builder:swiglu") -> Module:
    """Build the closed SwiGLU FFN graph ``Y = (SiLU(X@Wg^T)*(X@Wu^T))@Wd^T + X``.

    ``d`` is the model width (input and output), ``h`` the hidden width.
    """
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
    x_ty = TensorType(DTYPE_F16, (m, d))
    hid_ty = TensorType(DTYPE_F16, (m, h))
    out_ty = TensorType(DTYPE_F16, (m, d))
    values = [
        Value(0, "X", x_ty, None, "input"),
        Value(1, "Wg", TensorType(DTYPE_F16, (h, d)), None, "constant"),
        Value(2, "Wu", TensorType(DTYPE_F16, (h, d)), None, "constant"),
        Value(3, "Wd", TensorType(DTYPE_F16, (d, h)), None, "constant"),
        Value(4, "Zg", hid_ty, 0, "temporary"),
        Value(5, "Zu", hid_ty, 1, "temporary"),
        Value(6, "S", hid_ty, 2, "temporary"),
        Value(7, "A", hid_ty, 3, "temporary"),
        Value(8, "H", out_ty, 4, "temporary"),
        Value(9, "Y", out_ty, 5, "output"),
    ]
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    ops = [
        Op(0, "matmul", (0, 1), (4,), dict(attrs), source),
        Op(1, "matmul", (0, 2), (5,), dict(attrs), source),
        Op(2, "silu", (4,), (6,), {}, source),
        Op(3, "mul", (6, 5), (7,), {}, source),
        Op(4, "matmul", (7, 3), (8,), dict(attrs), source),
        Op(5, "add", (8, 0), (9,), {}, source),
    ]
    return Module(name=name, values=tuple(values), ops=tuple(ops), inputs=(0,),
                  outputs=(9,), source_hash="builder", state_signature="")


def matmul_module(*, m: int, k: int, n: int, name: str = "matmul",
                  residual: str | None = None,
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
    matmul_out_role = "temporary" if (activation or residual) else "output"
    c = Value(2, "Z" if activation else "Y", y_ty, 0, matmul_out_role)
    values.append(c)
    ops.append(Op(0, "matmul", (0, 1), (2,),
                  {"b_layout": b_layout,
                   "numeric_contract": contract.to_dict()}, source))
    next_id = len(values)
    op_id = len(ops)
    previous = 2
    if activation:
        out = Value(next_id, "A1", y_ty, op_id, "temporary")
        values.append(out)
        ops.append(Op(op_id, activation, (previous,), (next_id,), {}, source))
        previous = next_id
        next_id += 1
        op_id += 1
    if residual is not None:
        if residual not in {"add", "mul"}:
            reject(RK_E_UNSUPPORTED, f"unsupported residual op {residual!r}",
                   stage="frontend")
        operand = Value(next_id, "R", y_ty, None, "input")
        values.append(operand)
        next_id += 1
        out = Value(next_id, "Y", y_ty, op_id, "output")
        values.append(out)
        ops.append(Op(op_id, residual, (previous, operand.id), (out.id,), {},
                      source))
        previous = out.id
        inputs = (0, operand.id)
    else:
        out = values[previous]
        # The last produced value becomes the exported output.
        values[previous] = replace(out, role="output")
        inputs = (0,)
    return Module(name=name, values=tuple(values), ops=tuple(ops),
                  inputs=inputs, outputs=(previous,), source_hash="builder",
                  state_signature="")


def _builder_contract(numeric_mode: str) -> NumericContract:
    if numeric_mode == "strict":
        return NumericContract(
            accumulator_dtype="float32", accumulation="strict_unspecified",
            narrowing_points=(), approximate=False,
            tolerance={"atol": 1e-3, "rtol": 1e-3})
    if numeric_mode == "rk_f16_tiled":
        return NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=("k_tile_boundary",), approximate=False,
            tolerance={"atol": 8e-3, "rtol": 8e-3})
    reject(RK_E_NUMERIC_CONTRACT, f"unknown numeric mode {numeric_mode!r}",
           stage="frontend")


def projection_module(*, m: int, d: int, h: int, name: str = "projection",
                      b_layout: str = "nk",
                      numeric_mode: str = "rk_f16_tiled",
                      source: str = "rk-builder:projection") -> Module:
    """Build ``Y = SiLU(X@W1^T) @ W2^T + X``.

    With ``b_layout='nk'`` the weights use the custom-ops ``[N,K]`` convention.
    The graph is a debug/offline construction helper for tiled dataflow tests.
    """
    contract = _builder_contract(numeric_mode)
    w1_shape = (h, d) if b_layout == "nk" else (d, h)
    w2_shape = (d, h) if b_layout == "nk" else (h, d)
    x_ty = TensorType(DTYPE_F16, (m, d))
    hid_ty = TensorType(DTYPE_F16, (m, h))
    out_ty = TensorType(DTYPE_F16, (m, d))
    values = (
        Value(0, "X", x_ty, None, "input"),
        Value(1, "W1", TensorType(DTYPE_F16, w1_shape), None, "constant"),
        Value(2, "W2", TensorType(DTYPE_F16, w2_shape), None, "constant"),
        Value(3, "Z", hid_ty, 0, "temporary"),
        Value(4, "S", hid_ty, 1, "temporary"),
        Value(5, "G", hid_ty, 2, "temporary"),
        Value(6, "P", out_ty, 3, "temporary"),
        Value(7, "Y", out_ty, 4, "output"),
    )
    attrs = {"b_layout": b_layout, "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    ops = (
        Op(0, "matmul", (0, 1), (3,), dict(attrs), source),
        Op(1, "sigmoid", (3,), (4,), {}, source),
        Op(2, "mul", (4, 3), (5,), {}, source),
        Op(3, "matmul", (5, 2), (6,), dict(attrs), source),
        Op(4, "add", (6, 0), (7,), {}, source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0,), outputs=(7,),
                  source_hash="builder", state_signature="")


def rmsnorm_approx_module(*, m: int, d: int, name: str = "rmsnorm_approx",
                          source: str = "rk-builder:rmsnorm-approx") -> Module:
    """Build the explicit full-device RMSNorm reduction chain.

    The graph is intentionally expressed in the existing primitive set:
    square -> row sum via an MM with a ones weight -> mean -> +eps -> rsqrt LUT
    -> gamma expansion via a second MM -> multiply.  Every intermediate is a
    device value.  The LUT rsqrt is an approximate, bounded-domain path and is
    therefore a distinct numeric policy from the Host FP32 reduction/rsqrt path.
    """
    contract = _builder_contract("rk_f16_tiled")
    x_ty = TensorType(DTYPE_F16, (m, d))
    row_ty = TensorType(DTYPE_F16, (m, 1))
    values = (
        Value(0, "X", x_ty, None, "input"),
        Value(1, "Ones", TensorType(DTYPE_F16, (1, d)), None, "constant"),
        Value(2, "InvD", row_ty, None, "constant"),
        Value(3, "Eps", row_ty, None, "constant"),
        Value(4, "GammaW", TensorType(DTYPE_F16, (d, 1)), None, "constant"),
        Value(5, "Sq", x_ty, 0, "temporary"),
        Value(6, "Stats", row_ty, 1, "temporary"),
        Value(7, "Mean", row_ty, 2, "temporary"),
        Value(8, "MeanEps", row_ty, 3, "temporary"),
        Value(9, "Scale", row_ty, 4, "temporary"),
        Value(10, "Coeff", x_ty, 5, "temporary"),
        Value(11, "Y", x_ty, 6, "output"),
    )
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    ops = (
        Op(0, "mul", (0, 0), (5,), {}, source),
        Op(1, "matmul", (5, 1), (6,), dict(attrs), source),
        Op(2, "mul", (6, 2), (7,), {}, source),
        Op(3, "add", (7, 3), (8,), {}, source),
        Op(4, "rsqrt", (8,), (9,), {}, source),
        Op(5, "matmul", (9, 4), (10,), dict(attrs), source),
        Op(6, "mul", (0, 10), (11,), {}, source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0,),
                  outputs=(11,), source_hash="builder", state_signature="")


def rope_device_module(*, m: int, width: int,
                       name: str = "rope_device",
                       source: str = "rk-builder:rope-device") -> Module:
    """Single-head (or one contiguous head block) device RoPE reorder.

    ``P`` is a block-diagonal signed permutation constant supplied by the
    caller.  ``Cos`` and ``Sin`` are runtime inputs carrying the precomputed
    position coefficients; non-rotary dimensions use cos=1, sin=0.  Arithmetic
    after the reorder is exactly three FP16 EW tasks.  The Host never reads the
    activation: P/Cos/Sin enter as uploads and the output is a device value.
    """
    contract = _builder_contract("rk_f16_tiled")
    ty = TensorType(DTYPE_F16, (m, width))
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    values = (
        Value(0, "X", ty, None, "input"),
        Value(1, "Cos", ty, None, "input"),
        Value(2, "Sin", ty, None, "input"),
        Value(3, "P", TensorType(DTYPE_F16, (width, width)), None,
              "constant"),
        Value(4, "Rot", ty, 0, "temporary"),
        Value(5, "XCos", ty, 1, "temporary"),
        Value(6, "RotSin", ty, 2, "temporary"),
        Value(7, "Y", ty, 3, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 3), (4,), dict(attrs), source),
        Op(1, "mul", (0, 1), (5,), {}, source),
        Op(2, "mul", (4, 2), (6,), {}, source),
        Op(3, "add", (5, 6), (7,), {}, source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0, 1, 2),
                  outputs=(7,), source_hash="builder", state_signature="")


def rope_device_constant_module(*, m: int, width: int,
                                name: str = "rope_device_constant",
                                source: str = "rk-builder:rope-device-constant") -> Module:
    """Device RoPE reorder with Cos/Sin supplied as plan constants.

    For a fixed prefill bucket the coefficient table is uploaded once when the
    PlanCall/workspace is created and can be reused by every layer.  Decode needs
    a dynamic position-indexed device view, which is not represented here.
    """
    contract = _builder_contract("rk_f16_tiled")
    ty = TensorType(DTYPE_F16, (m, width))
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    values = (
        Value(0, "X", ty, None, "input"),
        Value(1, "Cos", ty, None, "constant"),
        Value(2, "Sin", ty, None, "constant"),
        Value(3, "P", TensorType(DTYPE_F16, (width, width)), None,
              "constant"),
        Value(4, "Rot", ty, 0, "temporary"),
        Value(5, "XCos", ty, 1, "temporary"),
        Value(6, "RotSin", ty, 2, "temporary"),
        Value(7, "Y", ty, 3, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 3), (4,), dict(attrs), source),
        Op(1, "mul", (0, 1), (5,), {}, source),
        Op(2, "mul", (4, 2), (6,), {}, source),
        Op(3, "add", (5, 6), (7,), {}, source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0,),
                  outputs=(7,), source_hash="builder", state_signature="")


def rope_decode_indexed_module(*, width: int, capacity: int,
                               name: str = "rope_decode_indexed",
                               source: str = "rk-builder:rope-decode-indexed") -> Module:
    """Decode RoPE with a resident coefficient table and one-hot position gather.

    CosW/SinW are resident WEIGHT constants [width,capacity].  The Host only
    uploads a one-hot F16 position vector [1,capacity]; the device MM gathers the
    coefficient row.  The gathered Cos/Sin then feed the same FP16 reorder and
    three EW tasks as the other RoPE variants.
    """
    contract = _builder_contract("rk_f16_tiled")
    x_ty = TensorType(DTYPE_F16, (1, width))
    pos_ty = TensorType(DTYPE_F16, (1, capacity))
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    values = (
        Value(0, "X", x_ty, None, "input"),
        Value(1, "PosOneHot", pos_ty, None, "input"),
        Value(2, "CosW", TensorType(DTYPE_F16, (width, capacity)), None,
              "constant"),
        Value(3, "SinW", TensorType(DTYPE_F16, (width, capacity)), None,
              "constant"),
        Value(4, "P", TensorType(DTYPE_F16, (width, width)), None,
              "constant"),
        Value(5, "Cos", x_ty, 0, "temporary"),
        Value(6, "Sin", x_ty, 1, "temporary"),
        Value(7, "Rot", x_ty, 2, "temporary"),
        Value(8, "XCos", x_ty, 3, "temporary"),
        Value(9, "RotSin", x_ty, 4, "temporary"),
        Value(10, "Y", x_ty, 5, "output"),
    )
    ops = (
        Op(0, "matmul", (1, 2), (5,), dict(attrs), source),
        Op(1, "matmul", (1, 3), (6,), dict(attrs), source),
        Op(2, "matmul", (0, 4), (7,), dict(attrs), source),
        Op(3, "mul", (0, 5), (8,), {}, source),
        Op(4, "mul", (7, 6), (9,), {}, source),
        Op(5, "add", (8, 9), (10,), {}, source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0, 1),
                  outputs=(10,), source_hash="builder", state_signature="")


def pv_o_device_module(*, m: int, heads: int = 4, head_dim: int = 16,
                       out_dim: int = 64, name: str = "pv_o_device",
                       source: str = "rk-builder:pv-o-device") -> Module:
    """Attention PV heads feeding the O projection entirely on device.

    Each ``Vt_h`` is the transposed V head ``[head_dim,head_dim]`` because the
    MMA B operand convention is ``[n,k]``.  Each ``W_h`` is the corresponding
    O-weight slice ``[out_dim,head_dim]``.  The four PV results are direct
    producers for four O matmuls, followed by three FP16 EW adds.  No Host
    merge of the head outputs is required.
    """
    contract = _builder_contract("rk_f16_tiled")
    ty_pv = TensorType(DTYPE_F16, (m, head_dim))
    ty_o = TensorType(DTYPE_F16, (m, out_dim))
    values: list[Value] = []
    for h in range(heads):
        values.append(Value(len(values), f"P{h}", ty_pv, None, "input"))
    for h in range(heads):
        values.append(Value(len(values), f"Vt{h}",
                            TensorType(DTYPE_F16, (head_dim, head_dim)),
                            None, "input"))
    for h in range(heads):
        values.append(Value(len(values), f"W{h}",
                            TensorType(DTYPE_F16, (out_dim, head_dim)),
                            None, "constant"))
    pv_ids = []
    for h in range(heads):
        pv_ids.append(len(values))
        values.append(Value(len(values), f"PV{h}", ty_pv, len(pv_ids) - 1,
                            "temporary"))
    o_ids = []
    for h in range(heads):
        o_ids.append(len(values))
        values.append(Value(len(values), f"O{h}", ty_o, heads + h,
                            "temporary"))
    add_ids = []
    for h in range(heads - 1):
        add_ids.append(len(values))
        role = "output" if h == heads - 2 else "temporary"
        values.append(Value(len(values), f"A{h + 1}", ty_o,
                            len(pv_ids) + len(o_ids) + h, role))
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    ops: list[Op] = []
    for h in range(heads):
        ops.append(Op(len(ops), "matmul", (h, heads + h), (pv_ids[h],),
                      dict(attrs), source))
    for h in range(heads):
        ops.append(Op(len(ops), "matmul", (pv_ids[h], 2 * heads + h),
                      (o_ids[h],), dict(attrs), source))
    prev = o_ids[0]
    for h in range(heads - 1):
        ops.append(Op(len(ops), "add", (prev, o_ids[h + 1]),
                      (add_ids[h],), {}, source))
        prev = add_ids[h]
    return Module(name=name, values=tuple(values), ops=tuple(ops),
                  inputs=tuple(range(2 * heads)),
                  outputs=(add_ids[-1],), source_hash="builder",
                  state_signature="")


def pv_o_device_batches_module(*, batches: int = 8, m: int = 16,
                               heads: int = 4, head_dim: int = 16,
                               out_dim: int = 64,
                               name: str = "pv_o_device_b8",
                               source: str = "rk-builder:pv-o-device-b8") -> Module:
    """Eight independent Tiny-ViT PV→O chains in one workspace.

    Each batch contributes four PV head MM tasks, four O slice MM tasks and
    three FP16 EW adds.  The chains have no cross-batch dependency; the planner
    may keep them as multiple Submit groups, but all intermediates remain in the
    same device workspace.
    """
    contract = _builder_contract("rk_f16_tiled")
    ty_pv = TensorType(DTYPE_F16, (m, head_dim))
    ty_o = TensorType(DTYPE_F16, (m, out_dim))
    values: list[Value] = []
    for b in range(batches):
        for h in range(heads):
            values.append(Value(len(values), f"P{b}_{h}", ty_pv, None, "input"))
    for b in range(batches):
        for h in range(heads):
            values.append(Value(len(values), f"Vt{b}_{h}",
                                TensorType(DTYPE_F16, (head_dim, head_dim)),
                                None, "input"))
    for h in range(heads):
        values.append(Value(len(values), f"W{h}",
                            TensorType(DTYPE_F16, (out_dim, head_dim)),
                            None, "constant"))
    attrs = {"b_layout": "nk", "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    ops: list[Op] = []
    outputs: list[int] = []
    op_id = 0
    for b in range(batches):
        pv_ids = []
        for h in range(heads):
            pv_ids.append(len(values))
            values.append(Value(len(values), f"PV{b}_{h}", ty_pv, op_id,
                                "temporary"))
            ops.append(Op(op_id, "matmul",
                          (b * heads + h,
                           batches * heads + b * heads + h),
                          (pv_ids[h],), dict(attrs), source))
            op_id += 1
        o_ids = []
        for h in range(heads):
            o_ids.append(len(values))
            values.append(Value(len(values), f"O{b}_{h}", ty_o, op_id,
                                "temporary"))
            ops.append(Op(op_id, "matmul",
                          (pv_ids[h], 2 * batches * heads + h),
                          (o_ids[h],), dict(attrs), source))
            op_id += 1
        prev = o_ids[0]
        for h in range(heads - 1):
            out_id = len(values)
            role = "output" if h == heads - 2 else "temporary"
            values.append(Value(out_id, f"A{b}_{h + 1}", ty_o, op_id, role))
            ops.append(Op(op_id, "add", (prev, o_ids[h + 1]), (out_id,), {},
                          source))
            prev = out_id
            op_id += 1
        outputs.append(prev)
    return Module(name=name, values=tuple(values), ops=tuple(ops),
                  inputs=tuple(range(2 * batches * heads)),
                  outputs=tuple(outputs), source_hash="builder",
                  state_signature="")


def branch_outputs_module(*, m: int, k: int, n1: int, n2: int,
                          name: str = "branch",
                          b_layout: str = "nk",
                          numeric_mode: str = "rk_f16_tiled",
                          source: str = "rk-builder:branch") -> Module:
    """Build two independent matmul outputs from one shared input."""
    contract = _builder_contract(numeric_mode)
    w1_shape = (n1, k) if b_layout == "nk" else (k, n1)
    w2_shape = (n2, k) if b_layout == "nk" else (k, n2)
    attrs = {"b_layout": b_layout, "numeric_contract": contract.to_dict(),
             "source_accumulator_dtype": "float32"}
    values = (
        Value(0, "X", TensorType(DTYPE_F16, (m, k)), None, "input"),
        Value(1, "W1", TensorType(DTYPE_F16, w1_shape), None, "constant"),
        Value(2, "W2", TensorType(DTYPE_F16, w2_shape), None, "constant"),
        Value(3, "Y1", TensorType(DTYPE_F16, (m, n1)), 0, "output"),
        Value(4, "Y2", TensorType(DTYPE_F16, (m, n2)), 1, "output"),
    )
    ops = (
        Op(0, "matmul", (0, 1), (3,), dict(attrs), source),
        Op(1, "matmul", (0, 2), (4,), dict(attrs), source),
    )
    return Module(name=name, values=values, ops=ops, inputs=(0,),
                  outputs=(3, 4), source_hash="builder", state_signature="")


def activation_module(*, m: int, n: int, activation: str = "silu",
                      name: str = "activation",
                      source: str = "rk-builder:activation") -> Module:
    """Build one standalone unary activation graph."""
    if activation not in UNARY_OPS:
        reject(RK_E_UNSUPPORTED, f"activation {activation!r} is not supported",
               stage="frontend")
    ty = TensorType(DTYPE_F16, (m, n))
    values = (Value(0, "X", ty, None, "input"),
              Value(1, "Y", ty, 0, "output"))
    ops = (Op(0, activation, (0,), (1,), {}, source),)
    return Module(name=name, values=values, ops=ops, inputs=(0,), outputs=(1,),
                  source_hash="builder", state_signature="")
