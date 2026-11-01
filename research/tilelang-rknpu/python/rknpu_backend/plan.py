# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen RKPlan model, canonical hashing and plan-level constants.

A logical buffer, an allocation slot and a real BO are deliberately separate:

* ``Buffer`` is a logical lifetime/capacity record.  It names a ``slot_id``.
* A slot is instantiated once by the C runtime and may back several logical
  buffers whose lifetimes do not overlap.
* A ``View`` has an offset into its logical buffer.  The C runtime adds the
  buffer's slot offset and validates the resulting device address.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import (RK_E_ALIAS, RK_E_SCHEMA, RK_E_SHAPE, RK_E_UNSUPPORTED,
                     reject)
from .immutable import freeze, thaw
from .serialize import sha256_json

# Mirrors RK_MAX_BUFFERS_PER_SLOT in custom-ops/include/rk_plan.h, which is a
# verifier bookkeeping bound (bounded pairwise alias checks), not a device
# limit.  The planner keeps every slot below it and plan verification re-checks
# it, so a plan is never handed on that the device-level verifier must refuse.
MAX_BUFFERS_PER_SLOT = 64

# One NPU task programs one CUBE K tile; the same bound is enforced by the C
# verifier (CO_K_TILE) and by the emitter.  Kept here so the plan checker does
# not depend on the planner-side tile helpers.
K_TILE_LIMIT = 256
PLAN_SCHEMA_VERSION = 2
PLAN_KIND = "rkplan"
LAYOUT_IDS = {"co.cube.f16.v1", "co.weight.f16.v1", "co.linear.f16.v1"}
TEMPLATE_IDS = {
    "co.mm.f16.v1", "co.mm_acc.f16.v1", "co.ew_add.f16.v1",
    "co.ew_mul.f16.v1", "co.ew_max.f16.v1", "co.ew_add_relu.f16.v1",
    "co.lut.sigmoid.f16.v1", "co.lut.hardsigmoid.f16.v1",
    "co.lut.gelu_gate.f16.v1", "co.lut.rsqrt.f16.v1",
}
# The plan schema knows every step a pipeline stage may express, including the
# transient host steps a draft can carry while it is being built.
ACTION_KINDS = {"host_pack", "host_expand", "host_check", "upload", "npu_task",
                "download", "host_compute", "commit"}
# What the runtime can actually execute, and therefore the only kinds a finished
# plan may contain.  Kept next to ACTION_KINDS so the two lists stay in view.
EXECUTABLE_ACTION_KINDS = frozenset({"upload", "npu_task", "download"})
BINDING_ROLES = {"input", "constant", "output"}
# A submit's recipe must match the task sequence it actually carries.  This is the
# plan-level half of the "no arbitrary chain" rule; the C verifier re-checks the
# per-template contracts but cannot infer a recipe from a task list.
RECIPE_SIGNATURES = {
    # recipe -> list of (head, repeated middle or None, optional final task)
    # A K chain, or a single accumulating task when submits are split
    "linear_k_acc_v1": [("co.mm.f16.v1", "co.mm_acc.f16.v1", None),
                        ("co.mm_acc.f16.v1", None, None)],
    # Static-repeat group: N independent plain matmuls, each writing its own
    # output, chained into one submit.  The C verifier admits this shape by
    # proving structural independence (pairwise distinct written buffers and no
    # task reading a buffer written earlier in the same submit); this entry is
    # the plan-level half of that admission.
    "independent_mm_v1": [("co.mm.f16.v1", "co.mm.f16.v1", None)],
    # Multiple independent output K-accumulation chains in one submit.  The
    # exact per-chain K order is re-proven by the C verifier from explicit
    # source_k_index/source_k_count/deps metadata.
    "multi_k_acc_v1": [],
    # Each SwiGLU stage is its own submit; the recipe names the stage shapes.
    "swiglu_ffn_v1": [
        ("co.mm.f16.v1", None, None),
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1"),
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", None),
        ("co.ew_mul.f16.v1", None, None),
        ("co.ew_add.f16.v1", None, None),
    ],
    # A mixed-template chain admitted only after a board probe.  The C
    # verifier independently proves action-order contiguity, dependency order,
    # disjoint writes and the existing read/write alias contract.
    "dependency_chain_v1": [],
    "gate_ew_v1": [
        # chunked LUT, optionally followed by the gated multiply
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", None),
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1"),
        ("co.lut.gelu_gate.f16.v1", "co.lut.gelu_gate.f16.v1", None),
        ("co.lut.gelu_gate.f16.v1", "co.lut.gelu_gate.f16.v1", "co.ew_mul.f16.v1"),
        ("co.lut.hardsigmoid.f16.v1", "co.lut.hardsigmoid.f16.v1", None),
        # a single element-wise task (residual add / runtime-operand multiply)
        ("co.ew_add.f16.v1", None, None),
        ("co.ew_mul.f16.v1", None, None),
        ("co.ew_max.f16.v1", None, None),
        ("co.ew_add_relu.f16.v1", None, None),
    ],
}


def _recipe_matches(recipe: str, templates: list[str]) -> bool:
    """Prove a submit's declared recipe describes its actual task sequence."""
    if recipe == "linear_epilogue_v1":
        if not templates or templates[0] != "co.mm.f16.v1":
            return False
        i = 1
        while i < len(templates) and templates[i] == "co.mm_acc.f16.v1":
            i += 1
        if i == len(templates):
            return False
        if templates[i] == "co.ew_add_relu.f16.v1":
            return i + 1 == len(templates)
        if templates[i] != "co.ew_add.f16.v1":
            return False
        i += 1
        if i < len(templates):
            if templates[i] != "co.ew_max.f16.v1":
                return False
            i += 1
        return i == len(templates)
    if recipe == "ew_add_relu_v1":
        return templates == ["co.ew_add.f16.v1", "co.ew_max.f16.v1"]
    if recipe == "dependency_chain_v1":
        return bool(templates)
    if recipe == "multi_k_acc_v1":
        return (bool(templates) and templates[0] == "co.mm.f16.v1" and
                all(t in {"co.mm.f16.v1", "co.mm_acc.f16.v1"}
                    for t in templates) and
                "co.mm_acc.f16.v1" in templates)
    signatures = RECIPE_SIGNATURES.get(recipe)
    if not signatures or not templates:
        return False
    for head, repeat, final in signatures:
        if templates[0] != head:
            continue
        body = templates[1:]
        if final is not None:
            if body and body[-1] == final:
                body = body[:-1]
            elif len(templates) > 1:
                continue
        if not body:
            if len(templates) == 1 or final is not None:
                return True
            continue
        if repeat is not None and all(t == repeat for t in body):
            return True
    return False
BUFFER_ROLES = {"input", "input_slice", "weight", "accumulator", "gate",
                "output", "workspace", "constant", "identity", "state"}
PADDING_STATES = {"zero", "finite", "unknown", "valid_only"}


def _require(condition: bool, code: int, message: str, **kwargs: Any) -> None:
    if not condition:
        reject(code, message, **kwargs)


def _int(value: Any, name: str, *, minimum: int = 0, where: str = "plan") -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        reject(RK_E_SCHEMA, f"{where}.{name} must be an integer >= {minimum}",
               stage="schema")
    return value


def _str(value: Any, name: str, *, where: str = "plan") -> str:
    if not isinstance(value, str) or not value:
        reject(RK_E_SCHEMA, f"{where}.{name} must be a non-empty string",
               stage="schema")
    return value


def _tuple_ints(value: Any, name: str, *, where: str = "plan") -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        reject(RK_E_SCHEMA, f"{where}.{name} must be an array", stage="schema")
    return tuple(_int(v, f"{name}[{i}]", where=where) for i, v in enumerate(value))


def _keys(obj: dict[str, Any], required: set[str], optional: set[str] | None = None,
          where: str = "plan") -> None:
    optional = optional or set()
    missing = sorted(required - obj.keys())
    unknown = sorted(obj.keys() - required - optional)
    if missing:
        reject(RK_E_SCHEMA, f"{where} missing fields: {', '.join(missing)}",
               stage="schema")
    if unknown:
        reject(RK_E_SCHEMA, f"{where} unknown fields: {', '.join(unknown)}",
               stage="schema")


@dataclass(frozen=True)
class Binding:
    id: int
    name: str
    role: str
    dtype: str
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.role not in BINDING_ROLES:
            reject(RK_E_SCHEMA, f"unknown binding role {self.role!r}", stage="schema")
        if self.dtype != "float16":
            reject(RK_E_SCHEMA, "MVP bindings must be float16", stage="schema")
        if len(self.shape) > 2:
            reject(RK_E_SHAPE, "binding rank must be 0, 1 or 2", stage="shape",
                   actual=list(self.shape))
        if self.role in {"input", "output"} and len(self.shape) != 2:
            reject(RK_E_SHAPE,
                   "runtime input/output bindings must be rank 2",
                   stage="shape", actual={"role": self.role,
                                          "shape": list(self.shape)})
        if any(v <= 0 or v > 4096 for v in self.shape):
            reject(RK_E_SHAPE,
                   "binding dimensions must be static within 1..4096",
                   stage="shape", actual=list(self.shape))

    @property
    def nbytes(self) -> int:
        elements = 1
        for dim in self.shape:
            elements *= dim
        return elements * 2

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "role": self.role,
                "dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, obj: Any) -> "Binding":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Binding must be an object",
                 stage="schema")
        _keys(obj, {"id", "name", "role", "dtype", "shape"}, where="Binding")
        return cls(_int(obj["id"], "id", where="Binding"),
                   _str(obj["name"], "name", where="Binding"),
                   obj["role"], obj["dtype"],
                   tuple(_tuple_ints(obj["shape"], "shape", where="Binding")))


@dataclass(frozen=True)
class ConstantSpec:
    name: str
    binding_id: int
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str
    pack_key: str
    generator: str | None = None

    def __post_init__(self) -> None:
        if self.dtype != "float16":
            reject(RK_E_SCHEMA, "constant dtype must be float16", stage="schema")
        if len(self.sha256) != 64:
            reject(RK_E_SCHEMA, "constant sha256 must be 64 hex characters",
                   stage="schema")
        if self.generator not in {None, "identity", "zero"}:
            reject(RK_E_SCHEMA, f"unknown constant generator {self.generator!r}",
                   stage="schema")
        if len(self.shape) > 2:
            reject(RK_E_SCHEMA, "constant rank must be 0, 1 or 2", stage="schema",
                   actual=list(self.shape))
        elements = 1
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                reject(RK_E_SCHEMA,
                       "constant dimensions must be positive integers",
                       stage="schema", actual=list(self.shape))
            elements *= dim
        expected = elements * 2
        if self.nbytes != expected:
            reject(RK_E_SCHEMA, "constant nbytes does not match shape", stage="schema",
                   actual=self.nbytes, required=expected)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name, "binding_id": self.binding_id, "dtype": self.dtype,
            "shape": list(self.shape), "nbytes": self.nbytes,
            "sha256": self.sha256, "pack_key": self.pack_key,
        }
        if self.generator is not None:
            out["generator"] = self.generator
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> "ConstantSpec":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "ConstantSpec must be an object",
                 stage="schema")
        _keys(obj, {"name", "binding_id", "dtype", "shape", "nbytes", "sha256",
                    "pack_key"}, {"generator"}, "ConstantSpec")
        return cls(
            name=_str(obj["name"], "name", where="ConstantSpec"),
            binding_id=_int(obj["binding_id"], "binding_id", where="ConstantSpec"),
            dtype=obj["dtype"],
            shape=tuple(_tuple_ints(obj["shape"], "shape", where="ConstantSpec")),
            nbytes=_int(obj["nbytes"], "nbytes", where="ConstantSpec"),
            sha256=_str(obj["sha256"], "sha256", where="ConstantSpec"),
            pack_key=_str(obj["pack_key"], "pack_key", where="ConstantSpec"),
            generator=obj.get("generator"),
        )


@dataclass(frozen=True)
class TensorLayout:
    dtype: str
    layout: str
    rows: int
    cols: int
    physical_rows: int
    physical_cols: int
    atom: int
    bytes: int

    def __post_init__(self) -> None:
        if self.dtype != "float16":
            reject(RK_E_SCHEMA, "MVP layout dtype must be float16", stage="schema")
        if self.layout not in {"co.cube.f16.v1", "co.weight.f16.v1",
                               "co.linear.f16.v1"}:
            reject(RK_E_SCHEMA, f"unknown layout {self.layout!r}", stage="schema")
        if not (0 < self.rows <= self.physical_rows <= 256):
            reject(RK_E_SHAPE, "layout rows must satisfy 0 < rows <= physical_rows <= 256",
                   stage="shape")
        if not (0 < self.cols <= self.physical_cols <= 65520):
            reject(RK_E_SHAPE, "layout cols must satisfy 0 < cols <= physical_cols",
                   stage="shape")
        expected = self.physical_rows * self.physical_cols * 2
        if self.bytes != expected:
            reject(RK_E_SHAPE, "layout byte size is inconsistent", stage="shape")
        # Physical geometry is fixed by the C layout authority (co_tensor_*), not
        # a free tuning knob: two views of one buffer must agree on addressing.
        if self.layout == "co.linear.f16.v1":
            ok = (self.atom == 1 and self.physical_rows == 1 and
                  self.physical_cols == self.cols)
            expected_rule = "atom=1, physical_rows=1, physical_cols=cols"
        elif self.layout == "co.weight.f16.v1":
            ok = (self.atom == 32 and
                  self.physical_rows == -(-self.rows // 32) * 32 and
                  self.physical_cols == -(-self.cols // 32) * 32)
            expected_rule = ("atom=32, rows/cols padded to a multiple of 32")
        else:
            ok = (self.atom == 8 and
                  self.physical_rows == -(-max(self.rows, 12) // 4) * 4 and
                  self.physical_cols == -(-self.cols // 32) * 32)
            expected_rule = ("atom=8, physical_rows=round_up(max(rows,12),4), "
                             "physical_cols=round_up(cols,32)")
        if not ok:
            reject(RK_E_SHAPE,
                   f"{self.layout} physical geometry is not canonical: "
                   f"{expected_rule}", stage="shape",
                   actual={"rows": self.rows, "cols": self.cols,
                           "physical_rows": self.physical_rows,
                           "physical_cols": self.physical_cols,
                           "atom": self.atom})

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype, "layout": self.layout, "rows": self.rows,
            "cols": self.cols, "physical_rows": self.physical_rows,
            "physical_cols": self.physical_cols, "atom": self.atom,
            "bytes": self.bytes,
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "TensorLayout":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "TensorLayout must be an object",
                 stage="schema")
        _keys(obj, {"dtype", "layout", "rows", "cols", "physical_rows",
                    "physical_cols", "atom", "bytes"}, where="TensorLayout")
        return cls(
            dtype=obj["dtype"], layout=obj["layout"],
            rows=_int(obj["rows"], "rows", minimum=1, where="TensorLayout"),
            cols=_int(obj["cols"], "cols", minimum=1, where="TensorLayout"),
            physical_rows=_int(obj["physical_rows"], "physical_rows", minimum=1,
                               where="TensorLayout"),
            physical_cols=_int(obj["physical_cols"], "physical_cols", minimum=1,
                               where="TensorLayout"),
            atom=_int(obj["atom"], "atom", minimum=1, where="TensorLayout"),
            bytes=_int(obj["bytes"], "bytes", minimum=1, where="TensorLayout"),
        )


@dataclass(frozen=True)
class Buffer:
    id: int
    role: str
    slot_id: int
    offset: int
    capacity: int
    alignment: int
    layout_id: str
    lifetime: tuple[int, int]
    padding_state: str
    must_not_share_bo: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in BUFFER_ROLES:
            reject(RK_E_SCHEMA, f"unknown buffer role {self.role!r}", stage="schema")
        if self.layout_id not in LAYOUT_IDS:
            reject(RK_E_SCHEMA, f"unknown buffer layout_id {self.layout_id!r}",
                   stage="schema")
        if self.padding_state not in PADDING_STATES:
            reject(RK_E_SCHEMA, f"unknown padding state {self.padding_state!r}",
                   stage="schema")
        if self.alignment < 16 or self.alignment & (self.alignment - 1):
            reject(RK_E_SHAPE, "buffer alignment must be a power of two >= 16",
                   stage="shape")
        if self.offset % self.alignment:
            reject(RK_E_SHAPE, "buffer offset must satisfy declared alignment",
                   stage="shape")
        if self.lifetime[0] < 0 or self.lifetime[1] < self.lifetime[0]:
            reject(RK_E_SHAPE, "buffer lifetime is invalid", stage="shape")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "role": self.role, "slot_id": self.slot_id,
            "offset": self.offset, "capacity": self.capacity,
            "alignment": self.alignment, "layout_id": self.layout_id,
            "lifetime": list(self.lifetime), "padding_state": self.padding_state,
            "must_not_share_bo": list(self.must_not_share_bo),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "Buffer":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Buffer must be an object",
                 stage="schema")
        _keys(obj, {"id", "role", "slot_id", "offset", "capacity", "alignment",
                    "layout_id", "lifetime", "padding_state",
                    "must_not_share_bo"}, where="Buffer")
        lifetime = _tuple_ints(obj["lifetime"], "lifetime", where="Buffer")
        if len(lifetime) != 2:
            reject(RK_E_SCHEMA, "Buffer.lifetime must have two entries", stage="schema")
        return cls(
            id=_int(obj["id"], "id", where="Buffer"), role=obj["role"],
            slot_id=_int(obj["slot_id"], "slot_id", where="Buffer"),
            offset=_int(obj["offset"], "offset", where="Buffer"),
            capacity=_int(obj["capacity"], "capacity", minimum=1, where="Buffer"),
            alignment=_int(obj["alignment"], "alignment", minimum=16, where="Buffer"),
            layout_id=obj["layout_id"], lifetime=lifetime,
            padding_state=obj["padding_state"],
            must_not_share_bo=_tuple_ints(obj["must_not_share_bo"],
                                          "must_not_share_bo", where="Buffer"),
        )


@dataclass(frozen=True)
class View:
    id: int
    buffer_id: int
    offset_bytes: int
    tensor: TensorLayout
    valid_region: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.offset_bytes % 16:
            reject(RK_E_SHAPE, "view offset must be 16-byte aligned", stage="shape")
        r0, nr, c0, nc = self.valid_region
        if min(r0, nr, c0, nc) < 0:
            reject(RK_E_SHAPE, "view valid_region entries must be non-negative",
                   stage="shape", actual=list(self.valid_region))
        # Start and length are checked separately *and* together: a small extent
        # still runs past the end when the region does not start at 0.  This is
        # the logical tensor bound; the buffer byte bound is the separate
        # "view exceeds its logical buffer" rule in _validate_plan.
        if r0 > self.tensor.rows or c0 > self.tensor.cols:
            reject(RK_E_SHAPE, "view valid_region starts outside the logical tensor",
                   stage="shape", actual=list(self.valid_region),
                   required={"rows": self.tensor.rows,
                             "cols": self.tensor.cols})
        if r0 + nr > self.tensor.rows or c0 + nc > self.tensor.cols:
            reject(RK_E_SHAPE, "view valid_region is outside the logical tensor",
                   stage="shape", actual=list(self.valid_region),
                   required={"end_row": self.tensor.rows,
                             "end_col": self.tensor.cols})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "buffer_id": self.buffer_id,
            "offset_bytes": self.offset_bytes, "tensor": self.tensor.to_dict(),
            "valid_region": list(self.valid_region),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "View":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "View must be an object",
                 stage="schema")
        _keys(obj, {"id", "buffer_id", "offset_bytes", "tensor", "valid_region"},
              where="View")
        region = _tuple_ints(obj["valid_region"], "valid_region", where="View")
        if len(region) != 4:
            reject(RK_E_SCHEMA, "View.valid_region must have four entries",
                   stage="schema")
        return cls(_int(obj["id"], "id", where="View"),
                   _int(obj["buffer_id"], "buffer_id", where="View"),
                   _int(obj["offset_bytes"], "offset_bytes", where="View"),
                   TensorLayout.from_dict(obj["tensor"]), region)


@dataclass(frozen=True)
class Action:
    id: int
    kind: str
    template_id: str | None
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    deps: tuple[int, ...]
    origin: str
    attrs: dict[str, Any] = field(default_factory=dict)
    # Which logical ops this task implements.  Carried by the planner so the
    # attribution survives action renumbering (upload hoisting, submit
    # merging) instead of being re-derived from the origin string afterwards.
    # Deliberately outside the frozen plan identity: it is excluded from
    # equality and from serialization, because the plan states the association
    # once, in Plan.logical_ops.
    source_ops: tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            reject(RK_E_SCHEMA, f"unknown action kind {self.kind!r}", stage="schema")
        if self.kind == "npu_task":
            if self.template_id not in TEMPLATE_IDS:
                reject(RK_E_SCHEMA, f"unknown template {self.template_id!r}",
                       stage="schema")
        elif self.template_id is not None:
            reject(RK_E_SCHEMA, "non-NPU actions must not carry a template", stage="schema")
        if self.kind not in {"npu_task", "host_compute", "commit", "host_pack",
                             "host_expand", "host_check", "upload", "download"}:
            reject(RK_E_SCHEMA, "unknown action kind", stage="schema")
        if not isinstance(self.source_ops, tuple) or not all(
                isinstance(op, int) and not isinstance(op, bool) and op >= 0
                for op in self.source_ops):
            reject(RK_E_SCHEMA, "action source_ops must be logical op indices",
                   stage="schema", action_id=self.id)
        # ``frozen=True`` only stops attribute assignment.  Two separate aliasing
        # paths remain, and both are closed here:
        #   * the caller keeps the mapping it passed in and edits it later,
        #   * a caller edits this mapping afterwards through ``plan.actions[i]``.
        # ``freeze`` copies the whole tree and returns read-only containers, so a
        # descriptor can never be encoded from attributes that changed after the
        # plan hash was computed.
        object.__setattr__(self, "attrs", freeze(dict(self.attrs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "template_id": self.template_id,
            "inputs": list(self.inputs), "outputs": list(self.outputs),
            "deps": list(self.deps), "origin": self.origin,
            # Plain (thawed) projection: JSON and the canonical plan hash are
            # unchanged by the read-only in-memory representation.
            "attrs": thaw(dict(sorted(self.attrs.items()))),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "Action":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Action must be an object",
                 stage="schema")
        _keys(obj, {"id", "kind", "template_id", "inputs", "outputs", "deps",
                    "origin", "attrs"}, where="Action")
        return cls(_int(obj["id"], "id", where="Action"), obj["kind"],
                   obj["template_id"],
                   _tuple_ints(obj["inputs"], "inputs", where="Action"),
                   _tuple_ints(obj["outputs"], "outputs", where="Action"),
                   _tuple_ints(obj["deps"], "deps", where="Action"),
                   _str(obj["origin"], "origin", where="Action"), dict(obj["attrs"]))


@dataclass(frozen=True)
class Submit:
    id: int
    task_ids: tuple[int, ...]
    recipe_id: str
    mode: str
    core_policy: str
    deps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"chain", "separate"}:
            reject(RK_E_SCHEMA, f"unknown submit mode {self.mode!r}", stage="schema")
        if self.core_policy not in {"single_core_per_submit", "auto"}:
            reject(RK_E_SCHEMA, f"unknown core policy {self.core_policy!r}",
                   stage="schema")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "task_ids": list(self.task_ids),
                "recipe_id": self.recipe_id, "mode": self.mode,
                "core_policy": self.core_policy, "deps": list(self.deps)}

    @classmethod
    def from_dict(cls, obj: Any) -> "Submit":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Submit must be an object",
                 stage="schema")
        _keys(obj, {"id", "task_ids", "recipe_id", "mode", "core_policy", "deps"},
              where="Submit")
        return cls(_int(obj["id"], "id", where="Submit"),
                   _tuple_ints(obj["task_ids"], "task_ids", where="Submit"),
                   _str(obj["recipe_id"], "recipe_id", where="Submit"),
                   obj["mode"], obj["core_policy"],
                   _tuple_ints(obj["deps"], "deps", where="Submit"))


@dataclass(frozen=True)
class Publish:
    """One written tile of an output binding.

    ``dst_row``/``dst_col`` place the tile's logical ``rows``x``cols`` inside the
    row-major output binding.  A binding is published by a set of non-overlapping
    tiles that exactly cover it; nothing may be published twice.
    """

    name: str
    binding_id: int
    view_id: int
    dst_row: int
    dst_col: int
    rows: int
    cols: int

    def __post_init__(self) -> None:
        for field in ("dst_row", "dst_col", "rows", "cols"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                reject(RK_E_SCHEMA, f"Publish.{field} must be a non-negative integer",
                       stage="schema")
        if self.rows <= 0 or self.cols <= 0:
            reject(RK_E_SCHEMA, "Publish tile must have positive extents",
                   stage="schema")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "binding_id": self.binding_id,
                "view_id": self.view_id, "dst_row": self.dst_row,
                "dst_col": self.dst_col, "rows": self.rows, "cols": self.cols}

    @classmethod
    def from_dict(cls, obj: Any) -> "Publish":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Publish must be an object",
                 stage="schema")
        _keys(obj, {"name", "binding_id", "view_id", "dst_row", "dst_col",
                    "rows", "cols"}, where="Publish")
        return cls(_str(obj["name"], "name", where="Publish"),
                   _int(obj["binding_id"], "binding_id", where="Publish"),
                   _int(obj["view_id"], "view_id", where="Publish"),
                   _int(obj["dst_row"], "dst_row", where="Publish"),
                   _int(obj["dst_col"], "dst_col", where="Publish"),
                   _int(obj["rows"], "rows", minimum=1, where="Publish"),
                   _int(obj["cols"], "cols", minimum=1, where="Publish"))


BUDGET_POLICY_VERSION = 1


@dataclass(frozen=True)
class PlanBudget:
    """Explicit user-authorized limits persisted with a plan.

    This is deliberately separate from :class:`Accounting`, which remains the
    compiler's estimate of required resources.  Plans written before this block
    existed have no stored budget and are loaded with the historical strict
    semantics: budget == claimed usage.
    """
    device_budget_bytes: int
    host_budget_bytes: int
    schema_version: int = BUDGET_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BUDGET_POLICY_VERSION:
            reject(RK_E_SCHEMA,
                   f"unsupported budget policy schema {self.schema_version}",
                   stage="schema")
        for name in ("device_budget_bytes", "host_budget_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                reject(RK_E_BUDGET, f"{name} must be a positive integer",
                       stage="schema")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version,
                "device_budget_bytes": self.device_budget_bytes,
                "host_budget_bytes": self.host_budget_bytes}

    @classmethod
    def from_dict(cls, obj: Any) -> "PlanBudget":
        _require(isinstance(obj, dict), RK_E_SCHEMA,
                 "budget_policy must be an object", stage="schema")
        _keys(obj, {"schema_version", "device_budget_bytes",
                    "host_budget_bytes"}, where="budget_policy")
        return cls(device_budget_bytes=_int(
                       obj["device_budget_bytes"], "device_budget_bytes",
                       where="budget_policy"),
                   host_budget_bytes=_int(
                       obj["host_budget_bytes"], "host_budget_bytes",
                       where="budget_policy"),
                   schema_version=_int(obj["schema_version"], "schema_version",
                                       where="budget_policy"))


@dataclass(frozen=True)
class Accounting:
    peak_payload_bytes: int
    reserved_bytes: int
    data_bo_count: int
    device_request_bytes: int
    command_workspace_bytes: int
    host_bytes: int
    task_count: int
    submit_count: int
    transfers: int = 0
    broadcast_expand_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "peak_payload_bytes": self.peak_payload_bytes,
            "reserved_bytes": self.reserved_bytes,
            "data_bo_count": self.data_bo_count,
            "device_request_bytes": self.device_request_bytes,
            "command_workspace_bytes": self.command_workspace_bytes,
            "host_bytes": self.host_bytes,
            "task_count": self.task_count,
            "submit_count": self.submit_count,
            "transfers": self.transfers,
        }
        if self.broadcast_expand_bytes:
            out["broadcast_expand_bytes"] = self.broadcast_expand_bytes
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> "Accounting":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Accounting must be an object",
                 stage="schema")
        required = {"peak_payload_bytes", "reserved_bytes", "data_bo_count",
                    "device_request_bytes", "command_workspace_bytes", "host_bytes",
                    "task_count", "submit_count", "transfers"}
        _keys(obj, required, {"broadcast_expand_bytes"}, "Accounting")
        return cls(**{key: _int(obj[key], key, where="Accounting")
                      for key in required},
                   broadcast_expand_bytes=_int(
                       obj.get("broadcast_expand_bytes", 0),
                       "broadcast_expand_bytes", where="Accounting"))


@dataclass(frozen=True)
class LogicalOp:
    """The plan's own record of one source-graph op.

    Smaller than RKIR on purpose: index, opcode, value names, the caller's
    NumericContract (or None), and the device tasks that implement it.  It
    exists so a numeric requirement can be checked against the graph it claims
    to describe instead of being self-consistent on its own.  It is a record,
    not a proof: nothing here says the device met the tolerance, or that the
    module really came from the source named in source_hash.
    """

    index: int
    opcode: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    contract: dict | None
    tasks: tuple[int, ...]
    # True when the op's effect is carried by a view/layout/zero-copy
    # restructuring rather than by a device task (for example the gate/up half
    # split of a packed FFN).  Declared by the planner, never inferred.
    auxiliary: bool = False
    # Set by a plan-to-plan conversion (repeat expansion) so a copy of one
    # computation still answers "whose copy am I": the plan it was derived
    # from and the logical op it came from.  None for a directly compiled op.
    derived_from: dict | None = None

    def __post_init__(self) -> None:
        from .ir import NumericContract
        if not isinstance(self.index, int) or isinstance(self.index, bool) or (
                self.index < 0):
            reject(RK_E_SCHEMA, "logical op index must be >= 0", stage="schema")
        if not isinstance(self.opcode, str) or not self.opcode:
            reject(RK_E_SCHEMA, "logical op opcode must be non-empty",
                   stage="schema")
        for name in ("inputs", "outputs"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                    isinstance(v, str) and v for v in values):
                reject(RK_E_SCHEMA,
                       f"logical op {name} must be non-empty value names",
                       stage="schema", actual=name)
        if not self.inputs:
            reject(RK_E_SCHEMA, "logical op must list at least one input",
                   stage="schema", actual=self.index)
        if not self.outputs:
            reject(RK_E_SCHEMA, "logical op must list at least one output",
                   stage="schema", actual=self.index)
        if self.contract is not None:
            if not isinstance(self.contract, dict):
                reject(RK_E_SCHEMA, "logical op contract must be an object",
                       stage="schema", actual=self.index)
            NumericContract.from_dict(self.contract)
        if not isinstance(self.tasks, tuple) or not all(
                isinstance(task, int) and not isinstance(task, bool) and
                task >= 0 for task in self.tasks):
            reject(RK_E_SCHEMA, "logical op tasks must be action ids",
                   stage="schema", actual=self.index)
        if not isinstance(self.auxiliary, bool):
            reject(RK_E_SCHEMA, "logical op auxiliary must be a boolean",
                   stage="schema", actual=self.index)
        if self.derived_from is not None:
            if not isinstance(self.derived_from, dict) or \
                    set(self.derived_from) != {"plan_hash", "op_index"}:
                reject(RK_E_SCHEMA,
                       "logical op derived_from must record plan_hash and "
                       "op_index", stage="schema", actual=self.index)
            parent = self.derived_from["plan_hash"]
            if not isinstance(parent, str) or len(parent) != 64 or \
                    any(c not in "0123456789abcdef" for c in parent):
                reject(RK_E_SCHEMA,
                       "logical op derived_from.plan_hash must be a plan "
                       "hash", stage="schema", actual=repr(parent)[:80])
            parent_op = self.derived_from["op_index"]
            if not isinstance(parent_op, int) or \
                    isinstance(parent_op, bool) or parent_op < 0:
                reject(RK_E_SCHEMA,
                       "logical op derived_from.op_index must be >= 0",
                       stage="schema", actual=repr(parent_op))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "opcode": self.opcode,
            "inputs": list(self.inputs), "outputs": list(self.outputs),
            "contract": (dict(sorted(self.contract.items()))
                         if self.contract is not None else None),
            "tasks": list(self.tasks), "auxiliary": self.auxiliary,
            # Only emitted when set, so a directly compiled plan is unchanged.
            **({"derived_from": dict(sorted(self.derived_from.items()))}
               if self.derived_from is not None else {}),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "LogicalOp":
        _require(isinstance(obj, dict), RK_E_SCHEMA,
                 "logical op must be an object", stage="schema")
        _keys(obj, {"index", "opcode", "inputs", "outputs", "contract",
                    "tasks", "auxiliary"}, {"derived_from"}, "LogicalOp")
        derived_from = obj.get("derived_from")
        if derived_from is not None and not isinstance(derived_from, dict):
            reject(RK_E_SCHEMA, "logical op derived_from must be an object",
                   stage="schema")
        contract = obj["contract"]
        if contract is not None and not isinstance(contract, dict):
            reject(RK_E_SCHEMA, "logical op contract must be an object or null",
                   stage="schema")
        auxiliary = obj["auxiliary"]
        if not isinstance(auxiliary, bool):
            reject(RK_E_SCHEMA, "logical op auxiliary must be a boolean",
                   stage="schema")
        return cls(_int(obj["index"], "index", where="LogicalOp"),
                   _str(obj["opcode"], "opcode", where="LogicalOp"),
                   tuple(_str(v, "inputs", where="LogicalOp")
                         for v in obj["inputs"]),
                   tuple(_str(v, "outputs", where="LogicalOp")
                         for v in obj["outputs"]),
                   dict(contract) if contract is not None else None,
                   tuple(_int(v, "tasks", where="LogicalOp")
                         for v in obj["tasks"]),
                   auxiliary,
                   ({"plan_hash": _str(derived_from["plan_hash"],
                                      "plan_hash", where="LogicalOp"),
                     "op_index": _int(derived_from["op_index"],
                                       "op_index", where="LogicalOp")}
                    if derived_from is not None else None))


@dataclass(frozen=True)
class NumericRequirement:
    """What the caller asked of one logical op, and which tasks implement it.

    A requirement record, not evidence that the device met it: the contract says
    what arithmetic was requested, execution says what the planner emitted, and
    nothing here claims the tolerance holds.
    """

    op_index: int
    opcode: str
    contract: dict
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tasks: tuple[int, ...]
    execution: dict

    EXECUTION_KEYS = {"k_chain", "k_tile", "templates", "output_dtypes",
                      "attributed", "approximated"}

    def __post_init__(self) -> None:
        from .ir import NumericContract
        if not isinstance(self.op_index, int) or \
                isinstance(self.op_index, bool) or self.op_index < 0:
            reject(RK_E_SCHEMA, "numeric requirement op_index must be >= 0",
                   stage="schema")
        if not isinstance(self.opcode, str) or not self.opcode:
            reject(RK_E_SCHEMA, "numeric requirement opcode must be non-empty",
                   stage="schema")
        NumericContract.from_dict(self.contract)
        for name in ("inputs", "outputs"):
            values = getattr(self, name)
            if not values or not all(isinstance(v, str) and v for v in values):
                reject(RK_E_SCHEMA,
                       f"numeric requirement {name} must be non-empty names",
                       stage="schema")
        if not all(isinstance(task, int) and not isinstance(task, bool)
                   and task >= 0 for task in self.tasks):
            reject(RK_E_SCHEMA, "numeric requirement tasks must be action ids",
                   stage="schema")
        unknown = sorted(set(self.execution) - self.EXECUTION_KEYS)
        if unknown:
            reject(RK_E_SCHEMA,
                   "numeric requirement execution has unsupported fields",
                   stage="schema", actual=unknown,
                   required=sorted(self.EXECUTION_KEYS))

    def to_dict(self) -> dict[str, Any]:
        return {"op_index": self.op_index, "opcode": self.opcode,
                "contract": dict(sorted(self.contract.items())),
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "tasks": list(self.tasks), "execution": dict(self.execution)}

    @classmethod
    def from_dict(cls, obj: Any) -> "NumericRequirement":
        _require(isinstance(obj, dict), RK_E_SCHEMA,
                 "numeric requirement must be an object", stage="schema")
        _keys(obj,
              {"op_index", "opcode", "contract", "inputs", "outputs",
               "tasks", "execution"},
              set(), "NumericRequirement")
        return cls(_int(obj["op_index"], "op_index",
                        where="NumericRequirement"),
                   _str(obj["opcode"], "opcode",
                        where="NumericRequirement"),
                   dict(obj["contract"]),
                   tuple(_str(v, "inputs", where="NumericRequirement")
                         for v in obj["inputs"]),
                   tuple(_str(v, "outputs", where="NumericRequirement")
                         for v in obj["outputs"]),
                   tuple(_int(v, "tasks", where="NumericRequirement")
                         for v in obj["tasks"]),
                   dict(obj["execution"]))


@dataclass(frozen=True)
class Plan:
    name: str
    profile_hash: str
    source_hash: str
    bindings: tuple[Binding, ...]
    constants: tuple[ConstantSpec, ...]
    buffers: tuple[Buffer, ...]
    views: tuple[View, ...]
    actions: tuple[Action, ...]
    submits: tuple[Submit, ...]
    publish: tuple[Publish, ...]
    accounting: Accounting
    plan_hash: str = ""
    schema_version: int = PLAN_SCHEMA_VERSION
    kind: str = PLAN_KIND
    budget: PlanBudget | None = None
    # Empty for plans written before the field existed: the load path then
    # reports "not stored" instead of inventing a default requirement.
    numeric_requirements: tuple[NumericRequirement, ...] = ()
    # The source graph, as far as a plan can restate it.  Also empty for plans
    # written before the field existed; those load and re-save unchanged and
    # skip the requirement-vs-graph cross-checks rather than inventing a graph.
    logical_ops: tuple[LogicalOp, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            reject(RK_E_SCHEMA, f"unsupported plan schema version {self.schema_version}",
                   stage="schema")
        if self.kind != PLAN_KIND:
            reject(RK_E_SCHEMA, f"expected kind {PLAN_KIND!r}, got {self.kind!r}",
                   stage="schema")
        _validate_plan(self)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "profile_hash": self.profile_hash,
            "source_hash": self.source_hash,
            "bindings": [b.to_dict() for b in self.bindings],
            "constants": [c.to_dict() for c in self.constants],
            "buffers": [b.to_dict() for b in self.buffers],
            "views": [v.to_dict() for v in self.views],
            "actions": [a.to_dict() for a in self.actions],
            "submits": [s.to_dict() for s in self.submits],
            "publish": [p.to_dict() for p in self.publish],
            "accounting": self.accounting.to_dict(),
        }
        if self.budget is not None:
            out["budget_policy"] = self.budget.to_dict()
        if self.logical_ops:
            out["logical_ops"] = [op.to_dict() for op in self.logical_ops]
        if self.numeric_requirements:
            out["numeric_requirements"] = [requirement.to_dict()
                                           for requirement in
                                           self.numeric_requirements]
        if include_hash:
            out["plan_hash"] = self.plan_hash
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> "Plan":
        _require(isinstance(obj, dict), RK_E_SCHEMA, "Plan must be an object",
                 stage="schema")
        required = {"schema_version", "kind", "name", "profile_hash", "source_hash",
                    "bindings", "constants", "buffers", "views", "actions",
                    "submits", "publish", "accounting", "plan_hash"}
        _keys(obj, required,
              {"budget_policy", "numeric_requirements", "logical_ops"},
              where="Plan")
        plan = cls(
            name=_str(obj["name"], "name", where="Plan"),
            profile_hash=_str(obj["profile_hash"], "profile_hash", where="Plan"),
            source_hash=obj["source_hash"],
            bindings=tuple(Binding.from_dict(v) for v in obj["bindings"]),
            constants=tuple(ConstantSpec.from_dict(v) for v in obj["constants"]),
            buffers=tuple(Buffer.from_dict(v) for v in obj["buffers"]),
            views=tuple(View.from_dict(v) for v in obj["views"]),
            actions=tuple(Action.from_dict(v) for v in obj["actions"]),
            submits=tuple(Submit.from_dict(v) for v in obj["submits"]),
            publish=tuple(Publish.from_dict(v) for v in obj["publish"]),
            accounting=Accounting.from_dict(obj["accounting"]),
            plan_hash=obj["plan_hash"],
            schema_version=obj["schema_version"], kind=obj["kind"],
            budget=(PlanBudget.from_dict(obj["budget_policy"])
                    if "budget_policy" in obj else None),
            numeric_requirements=tuple(
                NumericRequirement.from_dict(item)
                for item in obj.get("numeric_requirements", ())),
            logical_ops=tuple(LogicalOp.from_dict(item)
                              for item in obj.get("logical_ops", ())),
        )
        expected = compute_plan_hash(plan)
        if plan.plan_hash != expected:
            reject(RK_E_SCHEMA, "plan_hash does not match canonical plan bytes",
                   stage="schema", actual=plan.plan_hash, required=expected)
        return plan

    def binding(self, name: str) -> Binding:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        reject(RK_E_SHAPE, f"plan has no binding named {name!r}", stage="binding")


def compute_plan_hash(plan: Plan) -> str:
    return sha256_json(plan.to_dict(include_hash=False))


def with_plan_hash(plan: Plan) -> Plan:
    from dataclasses import replace
    tmp = replace(plan, plan_hash="")
    digest = compute_plan_hash(tmp)
    return replace(tmp, plan_hash=digest)


def _validate_plan(plan: Plan) -> None:
    # IDs are contiguous and maps are built explicitly, never by trusting fields.
    for collection, label in ((plan.bindings, "binding"), (plan.buffers, "buffer"),
                              (plan.views, "view"), (plan.actions, "action"),
                              (plan.submits, "submit")):
        ids = [item.id for item in collection]
        if ids != list(range(len(collection))):
            reject(RK_E_SCHEMA, f"{label} IDs must be contiguous from 0",
                   stage="schema")
    binding_by_id = {b.id: b for b in plan.bindings}
    if len(binding_by_id) != len(plan.bindings):
        reject(RK_E_SCHEMA, "duplicate binding IDs", stage="schema")
    buffer_by_id = {b.id: b for b in plan.buffers}
    view_by_id = {v.id: v for v in plan.views}
    action_by_id = {a.id: a for a in plan.actions}
    if len(view_by_id) != len(plan.views):
        reject(RK_E_SCHEMA, "duplicate view IDs", stage="schema")
    if len(action_by_id) != len(plan.actions):
        reject(RK_E_SCHEMA, "duplicate action IDs", stage="schema")

    # The runtime executes exactly three action kinds.  The plan schema also knows
    # the transient planning steps (host_pack/expand/check, host_compute, commit);
    # they are legal while a draft is being built, but a plan that leaves this
    # function is the one handed to the runtime, so a transient step here would be
    # a request nobody can execute.  Refusing it names the step instead of letting
    # the plan fail later in the verifier.
    for action in plan.actions:
        if action.kind in EXECUTABLE_ACTION_KINDS:
            continue
        if action.kind == "host_compute":
            reject(RK_E_UNSUPPORTED,
                   "MVP forbids hidden host arithmetic actions",
                   stage="schedule", action_id=action.id,
                   actual=action.kind,
                   required=sorted(EXECUTABLE_ACTION_KINDS))
        reject(RK_E_UNSUPPORTED,
               f"plan action kind {action.kind!r} is a transient planning "
               "step, not an executable action",
               stage="schedule", action_id=action.id, actual=action.kind,
               required=sorted(EXECUTABLE_ACTION_KINDS))

    names = [b.name for b in plan.bindings]
    if len(set(names)) != len(names):
        reject(RK_E_SCHEMA, "binding names must be unique", stage="schema")
    constants_by_name = {c.name: c for c in plan.constants}
    for c in plan.constants:
        if c.binding_id not in binding_by_id:
            reject(RK_E_SCHEMA, f"constant {c.name} references missing binding",
                   stage="schema")
        if binding_by_id[c.binding_id].role != "constant":
            reject(RK_E_SCHEMA, f"constant {c.name} binding is not constant",
                   stage="schema")
        if (binding_by_id[c.binding_id].shape != c.shape or
                binding_by_id[c.binding_id].dtype != c.dtype):
            reject(RK_E_SCHEMA, f"constant {c.name} metadata mismatch",
                   stage="schema")

    for buffer in plan.buffers:
        for other in buffer.must_not_share_bo:
            if other not in buffer_by_id:
                reject(RK_E_ALIAS, "buffer references missing must_not_share_bo",
                       stage="storage")
            if other == buffer.id:
                reject(RK_E_ALIAS, "buffer cannot forbid sharing with itself",
                       stage="storage")
            if buffer.slot_id == buffer_by_id[other].slot_id:
                reject(RK_E_ALIAS,
                       "must_not_share_bo buffers were assigned the same BO",
                       stage="storage", actual=[buffer.id, other])
    slot_capacity: dict[int, int] = {}
    for buffer in plan.buffers:
        slot_capacity[buffer.slot_id] = max(slot_capacity.get(buffer.slot_id, 0),
                                            buffer.offset + buffer.capacity)
    for a in plan.actions:
        for vid in a.inputs + a.outputs:
            if vid not in view_by_id:
                reject(RK_E_SCHEMA, f"action {a.id} references missing view {vid}",
                       stage="schema")
        for dep in a.deps:
            if dep not in action_by_id:
                reject(RK_E_SCHEMA, f"action {a.id} has missing dependency {dep}",
                       stage="schema")
            if dep >= a.id:
                reject(RK_E_SCHEMA, "action dependencies must precede the action",
                       stage="schema")
        for vid in a.inputs + a.outputs:
            view = view_by_id[vid]
            if view.buffer_id not in buffer_by_id:
                reject(RK_E_SCHEMA, f"view {vid} references missing buffer",
                       stage="schema")
            buffer = buffer_by_id[view.buffer_id]
            if view.offset_bytes + view.tensor.bytes > buffer.capacity:
                reject(RK_E_SHAPE, f"view {vid} exceeds its logical buffer",
                       stage="storage")
        if a.kind == "npu_task":
            _validate_action_template(plan, a, view_by_id, buffer_by_id)

    # All NPU actions are assigned to exactly one submit, and submit task arrays
    # contain only NPU actions.  Host actions are hard boundaries, not chain filler.
    upload_bindings: set[int] = set()
    download_bindings: set[int] = set()
    for action in plan.actions:
        if action.kind == "upload":
            upload_bindings.add(int(action.attrs.get("source_binding", -1)))
        if action.kind == "download":
            download_bindings.add(int(action.attrs.get("binding_id", -1)))
    for binding in plan.bindings:
        if binding.role in {"input", "constant"} and \
                binding.id not in upload_bindings:
            reject(RK_E_SCHEMA,
                   f"{binding.role} binding {binding.name!r} has no upload "
                   "action: a declared input has to be consumed by at least "
                   "one upload to become a plan parameter",
                   stage="storage")
        if binding.role == "output" and binding.id not in download_bindings:
            reject(RK_E_SCHEMA,
                   f"output binding {binding.name!r} has no download action",
                   stage="publish")

    # Recipe validation: the declared recipe must describe the actual task order.
    for submit in plan.submits:
        if not 1 <= len(submit.task_ids) <= 64:
            reject(RK_E_SHAPE, "a submit group must carry 1..64 tasks",
                   stage="schedule", actual=len(submit.task_ids))
        templates = []
        for task_id in submit.task_ids:
            action = action_by_id.get(task_id)
            if action is None or action.kind != "npu_task":
                reject(RK_E_SCHEMA,
                       "submit references a task id that is not an NPU task",
                       stage="schedule", actual=task_id)
            templates.append(action.template_id)
        if not _recipe_matches(submit.recipe_id, templates):
            reject(RK_E_SCHEMA,
                   f"submit recipe {submit.recipe_id!r} does not match its task "
                   "sequence", stage="schedule", actual=templates)

    seen_tasks: set[int] = set()
    for submit in plan.submits:
        if not submit.task_ids:
            reject(RK_E_SHAPE, "a submit must contain at least one task",
                   stage="schedule")
        if len(submit.task_ids) > 64:
            reject(RK_E_SHAPE, "a submit group cannot exceed 64 tasks",
                   stage="schedule")
        for task_id in submit.task_ids:
            action = action_by_id.get(task_id)
            if action is None or action.kind != "npu_task":
                reject(RK_E_SCHEMA, "submit contains a non-NPU action",
                       stage="schedule")
            if task_id in seen_tasks:
                reject(RK_E_SCHEMA, "a task may belong to only one submit",
                       stage="schedule")
            seen_tasks.add(task_id)
        for dep in submit.deps:
            if dep >= submit.id:
                reject(RK_E_SCHEMA, "submit dependencies must precede the submit",
                       stage="schedule")
    npu_ids = {a.id for a in plan.actions if a.kind == "npu_task"}
    if seen_tasks != npu_ids:
        reject(RK_E_SCHEMA, "every NPU task must be assigned to exactly one submit",
               stage="schedule")

    # The runtime walks actions in ascending order and flushes a submit when it
    # reaches that submit's last listed task, so the emission order is the
    # action order.  A submit therefore owns a contiguous run of actions:
    # ascending task ids, no host action inside the run, no other submit's task
    # inside the run, and runs that follow the declared submit order.  Without
    # these three, a plan can claim one execution order while the runtime runs
    # another (and the flushed batch would not even contain the skipped task).
    owner_of_task = {task: submit.id for submit in plan.submits
                     for task in submit.task_ids}
    previous_last = -1
    for submit in plan.submits:
        if list(submit.task_ids) != sorted(submit.task_ids):
            reject(RK_E_SCHEMA,
                   "submit tasks must be listed in ascending emission order",
                   stage="schedule", actual=list(submit.task_ids))
        first_task, last_task = submit.task_ids[0], submit.task_ids[-1]
        if first_task <= previous_last:
            reject(RK_E_SCHEMA,
                   "submit task runs must appear in submit order",
                   stage="schedule",
                   actual={"submit": submit.id, "first_task": first_task,
                           "previous_last": previous_last})
        for action_id in range(first_task, last_task + 1):
            action = action_by_id.get(action_id)
            if action is None or action.kind != "npu_task":
                reject(RK_E_SCHEMA,
                       "a host action is interleaved inside a task group",
                       stage="schedule", action_id=action_id,
                       actual=submit.id)
            if owner_of_task.get(action_id) != submit.id:
                reject(RK_E_SCHEMA,
                       "another submit's task is interleaved inside a task "
                       "group; a submit must own a contiguous run of tasks",
                       stage="schedule", action_id=action_id,
                       actual={"submit": submit.id,
                               "owner": owner_of_task.get(action_id)})
        previous_last = last_task

    # CBUF operand reuse: the bit means "the previous task on this core already
    # loaded this operand", so a plan may only set it where the previous task of
    # the same submit carries the same operand view at the same K index, and
    # never on the first task of a submit.  The C verifier enforces the same
    # rule; checking it here reports it before a descriptor exists.
    for submit in plan.submits:
        previous = None
        for task_id in submit.task_ids:
            action = action_by_id[task_id]
            wants_data = bool(action.attrs.get("data_reuse"))
            wants_weight = bool(action.attrs.get("weight_reuse"))
            if not wants_data and not wants_weight:
                previous = action
                continue
            if previous is None:
                reject(RK_E_SCHEMA,
                       "the first task of a submit cannot reuse CBUF operands",
                       stage="schedule", action_id=action.id,
                       actual=submit.id)
            same_k = (previous.attrs.get("k_index")
                      == action.attrs.get("k_index"))
            if wants_data and (not same_k or previous.inputs[0] != action.inputs[0]):
                reject(RK_E_SCHEMA,
                       "data_reuse must follow the same input view at the same "
                       "K index in the same submit",
                       stage="schedule", action_id=action.id,
                       actual={"previous": previous.inputs[0],
                               "current": action.inputs[0],
                               "k_index": [previous.attrs.get("k_index"),
                                           action.attrs.get("k_index")]})
            if wants_weight and (not same_k or
                                 previous.inputs[1] != action.inputs[1]):
                reject(RK_E_SCHEMA,
                       "weight_reuse must follow the same weight view at the same "
                       "K index in the same submit",
                       stage="schedule", action_id=action.id,
                       actual={"previous": previous.inputs[1],
                               "current": action.inputs[1],
                               "k_index": [previous.attrs.get("k_index"),
                                           action.attrs.get("k_index")]})
            previous = action

    published_bindings = {p.binding_id for p in plan.publish}
    output_bindings = {b.id for b in plan.bindings if b.role == "output"}
    if published_bindings != output_bindings:
        reject(RK_E_SCHEMA, "publish set must cover every output binding",
               stage="publish")
    if len({p.view_id for p in plan.publish}) != len(plan.publish):
        reject(RK_E_SCHEMA, "the same view cannot be published twice",
               stage="publish")
    for item in plan.publish:
        binding = binding_by_id.get(item.binding_id)
        if binding is None or binding.role != "output":
            reject(RK_E_SCHEMA, "publish target is not an output binding",
                   stage="publish")
        if item.view_id not in view_by_id:
            reject(RK_E_SCHEMA, "publish references missing view", stage="publish")
        view = view_by_id[item.view_id]
        if item.rows != view.tensor.rows or item.cols != view.tensor.cols:
            reject(RK_E_SHAPE, "publish tile must publish the whole view",
                   stage="publish")
        if item.dst_row + item.rows > binding.shape[0] or \
                item.dst_col + item.cols > binding.shape[1]:
            reject(RK_E_SHAPE, "publish tile exceeds the output binding",
                   stage="publish")
        for other in plan.publish:
            if other is item or other.binding_id != item.binding_id:
                continue
            if (other.dst_row < item.dst_row + item.rows and
                    item.dst_row < other.dst_row + other.rows and
                    other.dst_col < item.dst_col + item.cols and
                    item.dst_col < other.dst_col + other.cols):
                reject(RK_E_SHAPE, "publish tiles overlap", stage="publish")
    for binding in plan.bindings:
        if binding.role != "output":
            continue
        area = sum(p.rows * p.cols for p in plan.publish
                   if p.binding_id == binding.id)
        if area != binding.shape[0] * binding.shape[1]:
            reject(RK_E_SHAPE,
                   "publish tiles do not cover the output binding exactly",
                   stage="publish", actual=area,
                   required=binding.shape[0] * binding.shape[1])
    # Every published view must be the source of a declared download boundary.
    download_views = {(int(a.attrs.get("binding_id", -1)), a.inputs[0])
                      for a in plan.actions if a.kind == "download"}
    for item in plan.publish:
        if (item.binding_id, item.view_id) not in download_views:
            reject(RK_E_SCHEMA, "publish view has no matching download action",
                   stage="publish")
    for action in plan.actions:
        if action.kind != "download":
            continue
        matches = [p for p in plan.publish
                   if p.binding_id == int(action.attrs.get("binding_id", -1))
                   and action.inputs and p.view_id == action.inputs[0]]
        if len(matches) != 1:
            reject(RK_E_SCHEMA,
                   "every download must map to exactly one publish tile",
                   stage="publish")

    # A constant upload happens once at workspace creation, so the buffers it
    # fills must own their slot for the whole workspace lifetime: a per-run
    # buffer sharing that BO would silently corrupt the cached constant.
    persistent: set[int] = set()
    for action in plan.actions:
        if action.kind != "upload":
            continue
        binding = binding_by_id.get(int(action.attrs.get("source_binding", -1)))
        if action.attrs.get("cache") == "constant" or \
                (binding is not None and binding.role == "constant"):
            for view_id in action.outputs:
                persistent.add(view_by_id[view_id].buffer_id)
    slot_owner: dict[int, list[int]] = {}
    for buffer in plan.buffers:
        slot_owner.setdefault(buffer.slot_id, []).append(buffer.id)
    for buffer in plan.buffers:
        if buffer.id in persistent and len(slot_owner[buffer.slot_id]) != 1:
            reject(RK_E_ALIAS,
                   "a constant workspace buffer must own its slot for the whole "
                   "workspace lifetime", stage="storage",
                   actual=[buffer.id, buffer.slot_id])

    # Conservative BO alias contract.  Accumulation and EW/LUT outputs must never
    # alias the buffers they read, even when byte ranges happen to be disjoint.
    for a in plan.actions:
        if a.kind != "npu_task":
            continue
        in_buffers = [view_by_id[v].buffer_id for v in a.inputs]
        for vid in a.outputs:
            out_buffer = view_by_id[vid].buffer_id
            if out_buffer in in_buffers:
                reject(RK_E_ALIAS, f"action {a.id} output aliases an input buffer",
                       stage="verify", action_id=a.id)
            out_slot = buffer_by_id[out_buffer].slot_id
            for in_buffer in in_buffers:
                if buffer_by_id[in_buffer].slot_id == out_slot:
                    reject(RK_E_ALIAS,
                           f"action {a.id} output and input share a BO",
                           stage="verify", action_id=a.id)
        if a.template_id == "co.mm_acc.f16.v1":
            prior = view_by_id[a.inputs[2]]
            out = view_by_id[a.outputs[0]]
            if prior.buffer_id == out.buffer_id:
                reject(RK_E_ALIAS, "mm_acc prior and output must use different buffers",
                       stage="verify", action_id=a.id)
            pb = buffer_by_id[prior.buffer_id]
            ob = buffer_by_id[out.buffer_id]
            if pb.slot_id == ob.slot_id:
                reject(RK_E_ALIAS, "mm_acc prior and output must use different BOs",
                       stage="verify", action_id=a.id)

    # The logical op table is the plan's own statement of the graph it came
    # from.  When present it is closed over the device tasks it claims, so a
    # requirement can no longer describe a graph the plan does not contain.
    # When absent the plan predates the field: it loads, re-saves and replays
    # unchanged, and only the historical requirement checks run.
    logical_by_index: dict[int, LogicalOp] = {}
    if plan.logical_ops:
        previous = -1
        for logical in plan.logical_ops:
            if logical.index <= previous:
                reject(RK_E_SCHEMA,
                       "logical ops must appear once in ascending index order",
                       stage="schema", actual=logical.index,
                       required=previous + 1)
            previous = logical.index
            logical_by_index[logical.index] = logical
            if logical.auxiliary and logical.tasks:
                reject(RK_E_SCHEMA,
                       "an auxiliary logical op cannot own device tasks",
                       stage="schema", actual=list(logical.tasks))
            for task in logical.tasks:
                action = action_by_id.get(task)
                if action is None:
                    reject(RK_E_SCHEMA,
                           "logical op references a missing action",
                           stage="schema", action_id=task,
                           actual=logical.index)
                if action.kind != "npu_task":
                    reject(RK_E_SCHEMA,
                           "logical op references a non-NPU action",
                           stage="schema", action_id=task)
        attributed_tasks: set[int] = set()
        for logical in plan.logical_ops:
            attributed_tasks.update(logical.tasks)
        unclaimed = sorted(action.id for action in plan.actions
                           if action.kind == "npu_task" and
                           action.id not in attributed_tasks)
        if unclaimed:
            reject(RK_E_SCHEMA,
                   "device tasks are not attributed to any logical op",
                   stage="schema", actual=unclaimed)

        # Every name a logical op mentions must exist in this plan: a binding,
        # or the output of some recorded op.  A conversion that renames
        # bindings but leaves the description pointing at the old ones is
        # therefore rejected instead of reading as "the same computation".
        defined = {binding.name for binding in plan.bindings}
        for logical in plan.logical_ops:
            defined.update(logical.outputs)
        for logical in plan.logical_ops:
            unknown = sorted(name for name in
                             tuple(logical.inputs) + tuple(logical.outputs)
                             if name not in defined)
            if unknown:
                reject(RK_E_SCHEMA,
                       "logical op names a value this plan does not define",
                       stage="schema",
                       actual={"op_index": logical.index, "names": unknown})

        # The description must also agree with what the tasks actually read and
        # write.  Only host uploads identify a binding, and generated identity/
        # zero constants are hardware plumbing rather than logical inputs, so
        # those are the two exclusions -- nothing is inferred from names.
        upload_of_view: dict[int, Action] = {}
        downloads_of_view: dict[int, list[Action]] = {}
        for action in plan.actions:
            if action.kind == "upload":
                for view_id in action.outputs:
                    upload_of_view[view_id] = action
            elif action.kind == "download":
                for view_id in action.inputs:
                    downloads_of_view.setdefault(view_id, []).append(action)
        generated = {spec.binding_id for spec in plan.constants
                     if spec.generator is not None}
        claims: dict[int, list[LogicalOp]] = {}
        for logical in plan.logical_ops:
            for task in logical.tasks:
                claims.setdefault(task, []).append(logical)
        for task, owners in sorted(claims.items()):
            action = action_by_id[task]
            # A fused task implements several ops at once, so its inputs and
            # outputs are the union of the ops that claim it.
            named_in = sorted({name for owner in owners
                               for name in owner.inputs})
            named_out = sorted({name for owner in owners
                                for name in owner.outputs})
            op_index = [owner.index for owner in owners]
            for view_id in action.inputs:
                producer = upload_of_view.get(view_id)
                if producer is None:
                    continue
                source = producer.attrs.get("source_binding")
                if not isinstance(source, int) or isinstance(source, bool):
                    continue
                binding = binding_by_id.get(source)
                if binding is None:
                    reject(RK_E_SCHEMA,
                           "a task reads a binding this plan does not "
                           "contain", stage="schema", action_id=task,
                           actual=source)
                if source in generated:
                    continue
                if binding.name not in named_in:
                    reject(RK_E_SCHEMA,
                           "the logical ops claiming a task do not name the "
                           "input it reads", stage="schema", action_id=task,
                           actual={"op_index": op_index,
                                   "binding": binding.name},
                           required=named_in)
            for view_id in action.outputs:
                for download in downloads_of_view.get(view_id, ()):
                    target = download.attrs.get("binding_id")
                    if not isinstance(target, int) or isinstance(target, bool):
                        continue
                    binding = binding_by_id.get(target)
                    if binding is None:
                        continue
                    if binding.name not in named_out:
                        reject(RK_E_SCHEMA,
                               "the logical ops claiming a task do not name "
                               "the output it publishes", stage="schema",
                               action_id=task,
                               actual={"op_index": op_index,
                                       "binding": binding.name},
                               required=named_out)
    elif plan.numeric_requirements:
        reject(RK_E_SCHEMA,
               "numeric requirements need the logical op table that makes "
               "them checkable", stage="schema",
               actual=len(plan.numeric_requirements))

    # Numeric requirements: the record must point at real tasks, and the facts
    # it carries must agree with what those tasks actually emit.  This checks
    # bookkeeping and structure only; it never claims the tolerance is met.
    claimed_tasks: dict[int, int] = {}
    seen_ops: set[int] = set()
    for requirement in plan.numeric_requirements:
        if requirement.op_index in seen_ops:
            reject(RK_E_SCHEMA,
                   "one numeric requirement per logical op",
                   stage="schema", actual=requirement.op_index)
        seen_ops.add(requirement.op_index)
        emitted = [action_by_id[task] for task in requirement.tasks
                   if task in action_by_id]
        if len(emitted) != len(requirement.tasks):
            reject(RK_E_SCHEMA,
                   "numeric requirement references a missing action",
                   stage="schema", actual=list(requirement.tasks))
        for action in emitted:
            if action.kind != "npu_task":
                reject(RK_E_SCHEMA,
                       "numeric requirement references a non-NPU action",
                       stage="schema", action_id=action.id)
            # A fused task may implement more than one logical op (one
            # co.ew_add_relu task serves add and relu).  That is legal exactly
            # while the plan's own op table declares the shared claim, which
            # the table cross-check below verifies; without a table the old
            # exclusive rule still applies.
            if action.id in claimed_tasks and not plan.logical_ops:
                reject(RK_E_SCHEMA,
                       "a task cannot implement two numeric requirements",
                       stage="schema", action_id=action.id)
            claimed_tasks.setdefault(action.id, requirement.op_index)

        if plan.logical_ops:
            owner = logical_by_index.get(requirement.op_index)
            if owner is None:
                reject(RK_E_SCHEMA,
                       "numeric requirement names no logical op of this plan",
                       stage="schema", actual=requirement.op_index)
            for name, claimed in (
                    ("opcode", requirement.opcode),
                    ("inputs", requirement.inputs),
                    ("outputs", requirement.outputs),
                    ("contract", requirement.contract),
                    ("tasks", requirement.tasks)):
                declared = getattr(owner, name)
                if claimed != declared:
                    reject(RK_E_SCHEMA,
                           f"numeric requirement {name} disagrees with the "
                           f"logical op it names", stage="schema",
                           actual=str(claimed), required=str(declared))

        if set(requirement.inputs) & set(requirement.outputs):
            reject(RK_E_SCHEMA,
                   "a numeric requirement must not list a value as both "
                   "input and output", stage="schema",
                   actual=sorted(set(requirement.inputs)
                                 & set(requirement.outputs)))
        attributed = requirement.execution.get("attributed")
        if attributed is not None and bool(attributed) != bool(emitted):
            reject(RK_E_SCHEMA,
                   "numeric requirement attribution flag disagrees with tasks",
                   stage="schema", actual=attributed)
        dtypes = requirement.execution.get("output_dtypes")
        if dtypes:
            actual = sorted({view_by_id[action.outputs[0]].tensor.dtype
                             for action in emitted})
            if actual != sorted(dtypes):
                reject(RK_E_SCHEMA,
                       "numeric requirement output dtype disagrees with tasks",
                       stage="schema", actual=actual, required=list(dtypes))
        templates = requirement.execution.get("templates")
        if templates:
            actual = sorted({action.template_id for action in emitted})
            if actual != sorted(templates):
                reject(RK_E_SCHEMA,
                       "numeric requirement template list disagrees with tasks",
                       stage="schema", actual=actual, required=list(templates))
        chain = requirement.execution.get("k_chain") or []
        if chain:
            expected = list(range(0, chain[-1] + 256, 256))
            if chain != expected:
                reject(RK_E_SCHEMA,
                       "numeric requirement K chain must be ascending 256 slices",
                       stage="schema", actual=list(chain),
                       required=expected)
            actual = sorted({action.attrs.get("k0") for action in emitted
                             if action.attrs.get("k0") is not None})
            if actual != list(chain):
                reject(RK_E_SCHEMA,
                       "numeric requirement K chain disagrees with its tasks",
                       stage="schema", actual=actual, required=list(chain))
        contract = requirement.contract
        # A lookup table is an approximation, so an op lowered through one must
        # be recorded as approximated.  The contract cannot carry that: the
        # contract is what the source asked for, while admitting an approximate
        # lowering is a compile-time decision (allow_approx_lut).  The record
        # therefore lives in execution, and the two directions are checked: no
        # LUT task may implement a non-approximate contract unless the plan
        # records the admitted approximation, and a recorded approximation must
        # actually be implemented by a lookup table.
        lut = [action.id for action in emitted
               if (action.template_id or "").startswith("co.lut.")]
        approximated = requirement.execution.get("approximated")
        if approximated is not None and not isinstance(approximated, bool):
            reject(RK_E_SCHEMA,
                   "numeric requirement approximation flag must be a boolean",
                   stage="schema", actual=repr(approximated))
        if approximated:
            if not lut:
                reject(RK_E_SCHEMA,
                       "a requirement claims an approximated implementation "
                       "that no lookup-table task implements",
                       stage="schema", actual=requirement.op_index)
        elif lut and contract.get("approximate") is False:
            reject(RK_E_SCHEMA,
                   "a non-approximate contract cannot be implemented by a "
                   "lookup-table task unless the plan records the admitted "
                   "approximation", stage="schema", actual=lut)
        if contract.get("accumulation") == \
                "ordered_k_tiles_with_f16_writeback" and len(chain) > 1 and \
                "k_tile_boundary" not in contract.get("narrowing_points", ()):
            reject(RK_E_SCHEMA,
                   "a multi-K contract must declare k_tile_boundary narrowing",
                   stage="schema",
                   actual=list(contract.get("narrowing_points", ())))
    if plan.logical_ops:
        missing = sorted(entry.index for entry in plan.logical_ops
                         if entry.contract is not None and
                         entry.index not in seen_ops)
        if missing:
            reject(RK_E_SCHEMA,
                   "a logical op with a contract has no numeric requirement",
                   stage="schema", actual=missing)
        stray = sorted(index for index in seen_ops
                       if index in logical_by_index and
                       logical_by_index[index].contract is None)
        if stray:
            reject(RK_E_SCHEMA,
                   "a numeric requirement names a logical op with no contract",
                   stage="schema", actual=stray)


def _validate_action_template(plan: Plan, action: Action, view_by_id: dict[int, View],
                              buffer_by_id: dict[int, Buffer]) -> None:
    views = [view_by_id[v] for v in action.inputs + action.outputs]
    for view in views:
        if view.offset_bytes % 16:
            reject(RK_E_SHAPE, "view offset is not 16-byte aligned", stage="shape")
    if action.template_id == "co.mm.f16.v1":
        if len(action.inputs) != 2 or len(action.outputs) != 1:
            reject(RK_E_SHAPE, "co.mm requires two inputs and one output",
                   stage="verify", action_id=action.id)
        x, w, out = (view_by_id[v] for v in action.inputs + action.outputs)
        if x.tensor.layout != "co.cube.f16.v1" or \
                w.tensor.layout != "co.weight.f16.v1" or \
                out.tensor.layout != "co.cube.f16.v1":
            reject(RK_E_SHAPE, "co.mm layout mismatch", stage="verify",
                   action_id=action.id)
        if x.tensor.rows != out.tensor.rows or x.tensor.cols != w.tensor.cols or \
                w.tensor.rows != out.tensor.cols:
            reject(RK_E_SHAPE, "co.mm tensor extents mismatch", stage="verify",
                   action_id=action.id)
    elif action.template_id == "co.mm_acc.f16.v1":
        if len(action.inputs) != 3 or len(action.outputs) != 1:
            reject(RK_E_SHAPE, "co.mm_acc requires three inputs and one output",
                   stage="verify", action_id=action.id)
        x, w, prior, out = (view_by_id[v] for v in action.inputs + action.outputs)
        if prior.tensor != out.tensor:
            reject(RK_E_SHAPE, "co.mm_acc prior/output tensors must be identical",
                   stage="verify", action_id=action.id)
        if x.tensor.layout != "co.cube.f16.v1" or \
                w.tensor.layout != "co.weight.f16.v1" or \
                out.tensor.layout != "co.cube.f16.v1":
            reject(RK_E_SHAPE, "co.mm_acc layout mismatch", stage="verify",
                   action_id=action.id)
        if x.tensor.rows != out.tensor.rows or x.tensor.cols != w.tensor.cols or \
                w.tensor.rows != out.tensor.cols:
            reject(RK_E_SHAPE, "co.mm_acc tensor extents mismatch", stage="verify",
                   action_id=action.id)
        # One co.mm_acc task programs a single CUBE K tile; the emitter has no
        # register description for a wider extent, so a plan that claims one is
        # accepted nowhere in the pipeline.  co.mm above is bounded the same
        # way, and the K tail (K < 256) stays legal.
        if x.tensor.cols > K_TILE_LIMIT or w.tensor.cols > K_TILE_LIMIT or \
                x.tensor.physical_cols > K_TILE_LIMIT or \
                w.tensor.physical_cols > K_TILE_LIMIT:
            reject(RK_E_SHAPE,
                   "co.mm_acc K extent exceeds one 256-wide tile",
                   stage="verify", action_id=action.id,
                   actual={"x_cols": x.tensor.cols,
                           "w_cols": w.tensor.cols,
                           "x_physical_cols": x.tensor.physical_cols,
                           "w_physical_cols": w.tensor.physical_cols},
                   required={"max_k": K_TILE_LIMIT})
        # The accumulation reads the prior tile back and writes the next one in
        # the same physical geometry the emitter programs.
        if x.tensor.physical_rows != out.tensor.physical_rows or \
                x.tensor.physical_cols != w.tensor.physical_cols or \
                w.tensor.physical_rows != out.tensor.physical_cols:
            reject(RK_E_SHAPE, "co.mm_acc physical geometry mismatch",
                   stage="verify", action_id=action.id)
    elif action.template_id in {"co.ew_add.f16.v1", "co.ew_mul.f16.v1",
                                "co.ew_max.f16.v1",
                                "co.ew_add_relu.f16.v1"}:
        if len(action.inputs) != 3 or len(action.outputs) != 1:
            reject(RK_E_SHAPE, "co.ew requires three inputs and one output",
                   stage="verify", action_id=action.id)
        x, operand, identity, out = (view_by_id[v] for v in action.inputs + action.outputs)
        if x.tensor != operand.tensor or x.tensor != out.tensor:
            reject(RK_E_SHAPE, "co.ew requires exact same-shape operands and output",
                   stage="verify", action_id=action.id)
        # The emitter runs the element-wise unit through the matrix pipe, which
        # only reads CUBE tiles; a non-CUBE operand has no emit path at all.
        if x.tensor.layout != "co.cube.f16.v1":
            reject(RK_E_SHAPE, "co.ew operands and output must be CUBE tiles",
                   stage="verify", action_id=action.id,
                   actual={"layout": x.tensor.layout},
                   required={"layout": "co.cube.f16.v1"})
        pc = x.tensor.physical_cols
        if identity.tensor.layout != "co.weight.f16.v1" or \
                identity.tensor.rows != pc or identity.tensor.cols != pc:
            reject(RK_E_SHAPE, "co.ew identity weight dimensions mismatch",
                   stage="verify", action_id=action.id)
        if identity.tensor.physical_rows != pc or \
                identity.tensor.physical_cols != pc:
            reject(RK_E_SHAPE,
                   "co.ew identity weight physical geometry mismatch",
                   stage="verify", action_id=action.id,
                   actual={"physical_rows": identity.tensor.physical_rows,
                           "physical_cols": identity.tensor.physical_cols},
                   required={"physical": pc})
    elif action.template_id and action.template_id.startswith("co.lut."):
        if len(action.inputs) != 1 or len(action.outputs) != 1:
            reject(RK_E_SHAPE, "LUT requires one input and one output",
                   stage="verify", action_id=action.id)
        x = view_by_id[action.inputs[0]]
        out = view_by_id[action.outputs[0]]
        if x.tensor.bytes != out.tensor.bytes:
            reject(RK_E_SHAPE, "LUT input/output physical sizes must match",
                   stage="verify", action_id=action.id)
        # Equal byte counts are not enough: the emitter indexes both tiles with
        # one descriptor (co_emit_lut requires same(x, o)), so a plan whose
        # input and output share bytes but not geometry would be accepted and
        # then fail at emit time.
        if x.tensor != out.tensor:
            reject(RK_E_SHAPE,
                   "LUT input and output must be the same tile: layout, "
                   "logical shape and physical geometry have to match",
                   stage="verify", action_id=action.id,
                   actual={"in": [x.tensor.layout, x.tensor.rows,
                                  x.tensor.cols, x.tensor.physical_rows,
                                  x.tensor.physical_cols, x.tensor.atom],
                           "out": [out.tensor.layout, out.tensor.rows,
                                   out.tensor.cols, out.tensor.physical_rows,
                                   out.tensor.physical_cols, out.tensor.atom]})
        if x.tensor.layout == "co.linear.f16.v1":
            if out.tensor.layout != "co.linear.f16.v1":
                reject(RK_E_SHAPE, "linear LUT requires linear output",
                       stage="verify", action_id=action.id)
        elif x.tensor.layout != "co.cube.f16.v1" or \
                out.tensor.layout != "co.cube.f16.v1":
            reject(RK_E_SHAPE, "unsupported LUT layout", stage="verify",
                   action_id=action.id)
        physical_elements = x.tensor.physical_rows * x.tensor.physical_cols
        if x.tensor.layout == "co.cube.f16.v1" and physical_elements > 65528:
            reject(RK_E_SHAPE, "LUT tile exceeds chunking contract; split into views",
                   stage="verify", action_id=action.id)
