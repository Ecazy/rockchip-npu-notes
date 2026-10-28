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

from .errors import RK_E_ALIAS, RK_E_SCHEMA, RK_E_SHAPE, reject
from .immutable import freeze, thaw
from .serialize import sha256_json

PLAN_SCHEMA_VERSION = 2
PLAN_KIND = "rkplan"
LAYOUT_IDS = {"co.cube.f16.v1", "co.weight.f16.v1", "co.linear.f16.v1"}
TEMPLATE_IDS = {
    "co.mm.f16.v1", "co.mm_acc.f16.v1", "co.ew_add.f16.v1",
    "co.ew_mul.f16.v1", "co.ew_max.f16.v1", "co.ew_add_relu.f16.v1",
    "co.lut.sigmoid.f16.v1", "co.lut.hardsigmoid.f16.v1",
    "co.lut.gelu_gate.f16.v1",
}
ACTION_KINDS = {"host_pack", "host_expand", "host_check", "upload", "npu_task",
                "download", "host_compute", "commit"}
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
    # Each SwiGLU stage is its own submit; the recipe names the stage shapes.
    "swiglu_ffn_v1": [
        ("co.mm.f16.v1", None, None),
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", "co.ew_mul.f16.v1"),
        ("co.lut.sigmoid.f16.v1", "co.lut.sigmoid.f16.v1", None),
        ("co.ew_mul.f16.v1", None, None),
        ("co.ew_add.f16.v1", None, None),
    ],
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
        if min(r0, nr, c0, nc) < 0 or nr > self.tensor.rows or nc > self.tensor.cols:
            reject(RK_E_SHAPE, "view valid_region is outside the logical tensor",
                   stage="shape")

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
        _keys(obj, required, {"budget_policy"}, where="Plan")
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
                   f"{binding.role} binding {binding.name!r} has no upload action",
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
        pc = x.tensor.physical_cols
        if identity.tensor.layout != "co.weight.f16.v1" or \
                identity.tensor.rows != pc or identity.tensor.cols != pc:
            reject(RK_E_SHAPE, "co.ew identity weight dimensions mismatch",
                   stage="verify", action_id=action.id)
    elif action.template_id and action.template_id.startswith("co.lut."):
        if len(action.inputs) != 1 or len(action.outputs) != 1:
            reject(RK_E_SHAPE, "LUT requires one input and one output",
                   stage="verify", action_id=action.id)
        x = view_by_id[action.inputs[0]]
        out = view_by_id[action.outputs[0]]
        if x.tensor.bytes != out.tensor.bytes:
            reject(RK_E_SHAPE, "LUT input/output physical sizes must match",
                   stage="verify", action_id=action.id)
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
