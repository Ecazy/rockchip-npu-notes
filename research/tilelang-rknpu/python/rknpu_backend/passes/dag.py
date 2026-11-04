# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the device task DAG and logical views for MatMul [K-chain] -> activation."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, replace

from ..capabilities import TargetProfile
from ..errors import RK_E_BUDGET, RK_E_NUMERIC_CONTRACT, RK_E_SCHEMA, \
    RK_E_SHAPE, RK_E_UNSUPPORTED, reject
from ..ir import Module, NumericContract
from ..options import CompileOptions
from ..plan import (Action, Binding, ConstantSpec, LogicalOp,
                    NumericRequirement, Publish, TensorLayout, View)
from ..serialize import sha256_json
from .legalize import LegalizedGraph
from .tile import K_TILE, MN_TILE_ORDERS, MNTileSchedule, TILE_LIMIT, \
    cube_layout, input_layout, k_slices, lut_chunks, mn_schedule_from_attrs, \
    output_tiles, round_up, weight_layout

CUBE_LAYOUT = "co.cube.f16.v1"
WEIGHT_LAYOUT = "co.weight.f16.v1"
LINEAR_LAYOUT = "co.linear.f16.v1"


@dataclass(frozen=True)
class _TaskRect:
    """One rank-2 rectangle used by the current linear task chain only."""

    row: int
    col: int
    rows: int
    cols: int

    def contains(self, other: "_TaskRect") -> bool:
        return (self.row <= other.row and self.col <= other.col and
                self.row + self.rows >= other.row + other.rows and
                self.col + self.cols >= other.col + other.cols)

    def overlaps(self, other: "_TaskRect") -> bool:
        return not (self.row + self.rows <= other.row or
                    other.row + other.rows <= self.row or
                    self.col + self.cols <= other.col or
                    other.col + other.cols <= self.col)


@dataclass
class _TaskBufferState:
    """Actual region/version/producer state for the consumed task stream."""

    zero: _TaskRect | None = None
    zero_producer: int | None = None
    value: _TaskRect | None = None
    value_producer: int | None = None
    value_version: int = 0
    value_view: int | None = None
    value_action: int | None = None
    value_tile: tuple[int, int, int, int] | None = None
    source_region: _TaskRect | None = None

    def value_covers(self, region: _TaskRect) -> bool:
        return self.value is not None and self.value.contains(region)

    def zero_covers(self, region: _TaskRect) -> bool:
        return self.zero is not None and self.zero.contains(region)

    def covers(self, region: _TaskRect) -> bool:
        return self.value_covers(region) or self.zero_covers(region)

    def enter_zero(self, area: _TaskRect, producer: int) -> None:
        self.zero = area
        self.zero_producer = producer
        self.value = None
        self.value_producer = None
        self.value_view = None
        self.value_action = None
        self.value_tile = None
        self.value_version += 1

    def adopt_value(self, area: _TaskRect, source: "_TaskBufferState") -> None:
        self.value = area
        self.value_producer = source.value_producer
        self.value_view = source.value_view
        self.value_action = source.value_action
        self.value_tile = source.value_tile
        self.source_region = source.source_region
        self.value_version = source.value_version + 1

    def record_value(self, area: _TaskRect, producer: int,
                     view: int | None, action: int | None,
                     tile: tuple[int, int, int, int] | None = None,
                     source_region: _TaskRect | None = None) -> None:
        self.value = area
        self.value_producer = producer
        self.value_view = view
        self.value_action = action
        self.value_tile = tile
        self.source_region = source_region
        self.value_version += 1


@dataclass
class DraftBuffer:
    id: int
    role: str
    capacity: int
    layout_id: str
    first: int
    last: int
    padding_state: str
    reuse: bool = True
    must_not_share_bo: tuple[int, ...] = ()


@dataclass
class DraftPlan:
    name: str
    profile_hash: str
    source_hash: str
    bindings: list[Binding]
    constants: list[ConstantSpec]
    buffers: list[DraftBuffer]
    views: list[View]
    actions: list[Action]
    publish: list[Publish]
    numeric_requirements: tuple = ()
    logical_ops: tuple = ()


class _Builder:
    def __init__(self, module: Module, legalized: LegalizedGraph, *,
                 profile: TargetProfile, options: CompileOptions):
        self.module = module
        self.legalized = legalized
        self.profile = profile
        self.options = options
        self.bindings: list[Binding] = []
        self.binding_by_value: dict[int, int] = {}
        self.constants: list[ConstantSpec] = []
        self.buffers: list[DraftBuffer] = []
        self.views: list[View] = []
        self.actions: list[Action] = []
        self.publish: list[Publish] = []
        self._identity_buffers: dict[int, tuple[int, int]] = {}
        self._identity_values: dict[int, int] = {}
        self._zero_buffers: dict[tuple[int, int], tuple[int, int]] = {}
        self._zero_values: dict[int, int] = {}
        # Static-dataflow mapping: every RKIR value names the device view that
        # holds it and the action that produced it.  A later node consumes the
        # view directly; intermediates never make a host round trip.
        self.value_views: dict[int, int] = {}
        self.value_action: dict[int, int] = {}
        self._fused_add_relu: dict[int, int] = {}
        self._skip_fused_relu: set[int] = set()
        self._create_bindings()
        if options.fuse_add_relu and any(
                cap.id == "co.ew_add_relu.f16.v1" and cap.status == "enabled"
                for cap in profile.capabilities):
            self._find_add_relu_fusions()

    def _find_add_relu_fusions(self) -> None:
        """Find exact add -> relu pairs with no other consumer.

        The capability gate is checked by the caller; this method only proves
        the graph-side condition.  The add value must be a temporary consumed
        exactly once by the ReLU, and must not be a module output.
        """
        consumers: dict[int, list[int]] = {}
        for op in self.module.ops:
            for value_id in op.inputs:
                consumers.setdefault(value_id, []).append(op.id)
        outputs = set(self.module.outputs)
        for relu_op in self.module.ops:
            if relu_op.opcode != "relu" or len(relu_op.inputs) != 1:
                continue
            add_out = relu_op.inputs[0]
            if add_out in outputs:
                continue
            producer = self.module.value(add_out).producer
            if producer is None:
                continue
            add_op = self.module.op(producer)
            if add_op.opcode != "add" or len(add_op.outputs) != 1:
                continue
            if consumers.get(add_out) != [relu_op.id]:
                continue
            if self.module.value(add_out) is self.module.value(relu_op.outputs[0]):
                continue
            self._fused_add_relu[add_op.id] = relu_op.id
            self._skip_fused_relu.add(relu_op.id)

    def _create_bindings(self) -> None:
        for value in self.module.values:
            if value.role not in {"input", "constant", "output"}:
                continue
            bid = len(self.bindings)
            self.binding_by_value[value.id] = bid
            self.bindings.append(Binding(bid, value.name, value.role,
                                         value.type.dtype, value.type.shape))
            if value.role == "constant":
                shape = value.type.shape
                version = sha256_json({
                    "name": value.name, "shape": list(shape),
                    "dtype": value.type.dtype, "kind": "external-runtime-content",
                })
                elements = 1
                for dim in shape:
                    elements *= dim
                shape_key = "x".join(str(dim) for dim in shape) or "scalar"
                self.constants.append(ConstantSpec(
                    name=value.name, binding_id=bid, dtype=value.type.dtype,
                    shape=shape, nbytes=elements * 2,
                    sha256=version,
                    pack_key=f"f16:host-contract:{shape_key}:v1",
                    generator=None))
        names = [b.name for b in self.bindings]
        if len(names) != len(set(names)):
            reject(RK_E_SCHEMA, "binding names must be unique", stage="dag")
        for value_id in self.module.inputs:
            if value_id not in self.binding_by_value:
                reject(RK_E_SHAPE, "module input is not present in bindings",
                       stage="dag")
        for value_id in self.module.outputs:
            if value_id not in self.binding_by_value:
                reject(RK_E_SHAPE, "module output is not present in bindings",
                       stage="dag")

    def add_buffer(self, *, role: str, capacity: int, layout_id: str,
                   padding_state: str = "finite", reuse: bool = True,
                   must_not_share_bo: tuple[int, ...] = ()) -> int:
        bid = len(self.buffers)
        self.buffers.append(DraftBuffer(
            id=bid, role=role, capacity=capacity, layout_id=layout_id,
            first=10**9, last=-1, padding_state=padding_state, reuse=reuse,
            must_not_share_bo=must_not_share_bo))
        return bid

    def add_view(self, buffer_id: int, offset: int, tensor: TensorLayout,
                 region: tuple[int, int, int, int] | None = None) -> int:
        vid = len(self.views)
        if region is None:
            region = (0, tensor.rows, 0, tensor.cols)
        self.views.append(View(vid, buffer_id, offset, tensor, region))
        return vid

    def touch(self, buffer_id: int, action_id: int) -> None:
        buf = self.buffers[buffer_id]
        buf.first = min(buf.first, action_id)
        buf.last = max(buf.last, action_id)

    def add_action(self, kind: str, *, template_id: str | None = None,
                   inputs: tuple[int, ...] = (), outputs: tuple[int, ...] = (),
                   deps: tuple[int, ...] = (), origin: str = "",
                   attrs: dict | None = None,
                   op_ids: tuple[int, ...] = ()) -> int:
        aid = len(self.actions)
        self.actions.append(Action(aid, kind, template_id, inputs, outputs,
                                   deps, origin, attrs or {},
                                   tuple(op_ids)))
        for vid in inputs + outputs:
            self.touch(self.views[vid].buffer_id, aid)
        return aid

    def value_view(self, value_id: int, tensor: TensorLayout,
                   offset: int = 0) -> int:
        # One logical buffer per value in the minimal path.
        buf = self.add_buffer(role="workspace", capacity=tensor.bytes,
                              layout_id=tensor.layout)
        return self.add_view(buf, offset, tensor)

    def identity_for(self, physical_cols: int) -> tuple[int, int]:
        """Return (value_id, view_id) for an identity weight constant."""
        if physical_cols in self._identity_buffers:
            return self._identity_buffers[physical_cols]
        _Float16_one = struct.pack("<e", 1.0)
        _Float16_zero = struct.pack("<e", 0.0)
        payload = bytearray()
        for r in range(physical_cols):
            for c in range(physical_cols):
                payload += _Float16_one if r == c else _Float16_zero
        digest = hashlib.sha256(payload).hexdigest()
        value_id = 10_000 + len(self._identity_values)
        # This value is internal to lowering and never appears in RKIR.  It has no
        # Python value object; binding/lookup is by a synthetic id.
        binding_id = len(self.bindings)
        name = f"__identity_{physical_cols}"
        self.bindings.append(Binding(binding_id, name, "constant", "float16",
                                     (physical_cols, physical_cols)))
        self.constants.append(ConstantSpec(
            name=name, binding_id=binding_id, dtype="float16",
            shape=(physical_cols, physical_cols),
            nbytes=physical_cols * physical_cols * 2, sha256=digest,
            pack_key=f"identity:f16:{physical_cols}:v1", generator="identity"))
        layout = TensorLayout("float16", WEIGHT_LAYOUT, physical_cols,
                              physical_cols, physical_cols, physical_cols, 32,
                              physical_cols * physical_cols * 2)
        buffer_id = self.add_buffer(role="identity", capacity=layout.bytes,
                                    layout_id=WEIGHT_LAYOUT, reuse=False,
                                    padding_state="zero")
        view_id = self.add_view(buffer_id, 0, layout)
        # Upload the generated constant directly.  The synthetic value uses a
        # high id so it can never collide with a real RKIR value id.
        self.add_action("upload", outputs=(view_id,),
                        origin="lowering:identity",
                        attrs={"source_binding": binding_id,
                               "src_row_offset": 0, "src_col_offset": 0,
                               "src_row_stride": physical_cols,
                               "transpose_source": False,
                               "cache": "constant"})
        self._identity_buffers[physical_cols] = (value_id, view_id)
        self._identity_values[value_id] = binding_id
        return value_id, view_id

    def zero_for(self, rows: int, cols: int) -> tuple[int, int]:
        """Return (value_id, view_id) for an internal zero cube constant."""
        key = (rows, cols)
        if key in self._zero_buffers:
            return self._zero_buffers[key]
        cr, cc, cbytes = cube_layout(rows, cols)
        payload = bytes(max(1, rows * cols * 2))
        digest = hashlib.sha256(payload).hexdigest()
        value_id = 20_000 + len(self._zero_values)
        binding_id = len(self.bindings)
        name = f"__zero_{rows}x{cols}"
        self.bindings.append(Binding(binding_id, name, "constant", "float16",
                                     (rows, cols)))
        self.constants.append(ConstantSpec(
            name=name, binding_id=binding_id, dtype="float16",
            shape=(rows, cols), nbytes=rows * cols * 2, sha256=digest,
            pack_key=f"zero:f16:{rows}x{cols}:v1", generator="zero"))
        layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, cr, cc, 8,
                              cr * cc * 2)
        buffer_id = self.add_buffer(role="constant", capacity=cbytes,
                                    layout_id=CUBE_LAYOUT, reuse=False,
                                    padding_state="zero")
        view_id = self.add_view(buffer_id, 0, layout)
        self.add_action("upload", outputs=(view_id,),
                        origin="lowering:zero",
                        attrs={"source_binding": binding_id,
                               "src_row_offset": 0, "src_col_offset": 0,
                               "src_row_stride": cols,
                               "transpose_source": False, "cache": "constant"})
        self._zero_buffers[key] = (value_id, view_id)
        self._zero_values[value_id] = binding_id
        return value_id, view_id

    def _broadcast_targets(self) -> dict[int, tuple[int, int]]:
        """Return the expanded [M,N] target for each rank-0/1 constant.

        Only compile-time constants are allowed to broadcast.  A constant used
        with incompatible target tiles is rejected instead of being silently
        expanded twice.
        """
        targets: dict[int, tuple[int, int]] = {}
        for op in self.module.ops:
            if op.opcode not in {"add", "mul"}:
                continue
            out_shape = self.module.value(op.outputs[0]).type.shape
            if len(out_shape) != 2:
                reject(RK_E_SHAPE,
                       "binary broadcast result must be rank 2", stage="dag")
            for vid, other_id in ((op.inputs[0], op.inputs[1]),
                                  (op.inputs[1], op.inputs[0])):
                value = self.module.value(vid)
                if value.role != "constant" or value.type.rank >= 2:
                    continue
                other = self.module.value(other_id)
                if other.type.rank != 2 or other.type.shape != out_shape:
                    reject(RK_E_SHAPE,
                           "broadcast target must be the full output tensor",
                           stage="dag", actual={
                               "constant": list(value.type.shape),
                               "other": list(other.type.shape),
                               "out": list(out_shape)})
                if value.type.rank == 1 and value.type.shape[0] != out_shape[1]:
                    reject(RK_E_SHAPE,
                           "per-channel broadcast length must equal N", stage="dag",
                           actual=list(value.type.shape), required=out_shape[1])
                previous = targets.get(vid)
                if previous is not None and previous != out_shape:
                    reject(RK_E_UNSUPPORTED,
                           "one broadcast constant is used with different target "
                           "shapes", stage="dag",
                           actual={"previous": list(previous),
                                   "current": list(out_shape)})
                targets[vid] = out_shape
        return targets

    def _external_layouts(self) -> dict[int, tuple[str, str]]:
        """Infer the one device layout each input/constant binding needs.

        Returns ``value_id -> (kind, b_layout)`` where kind is ``"cube"`` or
        ``"weight"``.  A value used in incompatible roles is rejected rather
        than silently given two uploads with different semantics.
        """
        cube_users: set[int] = set()
        weight_users: dict[int, set[str]] = {}
        for op in self.module.ops:
            if op.opcode == "matmul":
                cube_users.add(op.inputs[0])
                b_layout = op.attrs.get("b_layout", "kn")
                weight_users.setdefault(op.inputs[1], set()).add(b_layout)
            elif op.opcode in {"add", "mul"}:
                for vid in op.inputs:
                    value = self.module.value(vid)
                    if not (value.role == "constant" and value.type.rank < 2):
                        cube_users.add(vid)
            else:
                cube_users.add(op.inputs[0])
        out: dict[int, tuple[str, str]] = {}
        for value in self.module.values:
            if value.role not in {"input", "constant"}:
                continue
            layouts = weight_users.get(value.id, set())
            if len(layouts) > 1:
                reject(RK_E_UNSUPPORTED,
                       "the same weight is used with conflicting b_layout "
                       "values", stage="dag", actual=sorted(layouts))
            if layouts:
                if value.id in cube_users:
                    reject(RK_E_UNSUPPORTED,
                           "a binding cannot be both a feature/residual tile and "
                           "a packed weight", stage="dag",
                           actual={"value": value.id, "name": value.name})
                out[value.id] = ("weight", next(iter(layouts)))
            else:
                out[value.id] = ("cube", "")
        return out

    def _upload_external_values(self) -> None:
        layouts = self._external_layouts()
        broadcast_targets = self._broadcast_targets()
        for value in self.module.values:
            if value.role not in {"input", "constant"}:
                continue
            kind, b_layout = layouts[value.id]
            binding = self.binding_by_value[value.id]
            broadcast = None
            if kind == "weight":
                if value.type.rank != 2:
                    reject(RK_E_SHAPE, "weight bindings must be rank 2",
                           stage="dag", actual=list(value.type.shape))
                source_rows, source_cols = value.type.shape
                if b_layout == "kn":
                    rows, cols = source_cols, source_rows
                    transpose = True
                else:
                    rows, cols = source_rows, source_cols
                    transpose = False
                pr, pc, pbytes = weight_layout(rows, cols)
                layout = TensorLayout("float16", WEIGHT_LAYOUT, rows, cols,
                                      pr, pc, 32, pbytes)
                buffer_id = self.add_buffer(role="weight", capacity=pbytes,
                                            layout_id=WEIGHT_LAYOUT,
                                            reuse=False, padding_state="zero")
                view_id = self.add_view(buffer_id, 0, layout)
                cache = "constant"
                source_stride = value.type.shape[1]
            else:
                if value.type.rank < 2:
                    target = broadcast_targets.get(value.id)
                    if target is None:
                        reject(RK_E_SHAPE,
                               "rank-0/1 constant has no binary broadcast "
                               "consumer", stage="dag",
                               actual={"value": value.id, "name": value.name})
                    rows, cols = target
                    broadcast = "scalar" if value.type.rank == 0 else "channel"
                    source_stride = 0
                else:
                    rows, cols = value.type.shape
                    source_stride = value.type.shape[1]
                cr, cc, cbytes = cube_layout(rows, cols)
                layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cr, cc, 8, cbytes)
                buffer_id = self.add_buffer(role="input_slice", capacity=cbytes,
                                            layout_id=CUBE_LAYOUT, reuse=False,
                                            padding_state="zero")
                view_id = self.add_view(buffer_id, 0, layout)
                transpose = False
                cache = "constant" if value.role == "constant" else "input"
            attrs = {"source_binding": binding, "src_row_offset": 0,
                     "src_col_offset": 0, "src_row_stride": source_stride,
                     "transpose_source": transpose, "cache": cache}
            if broadcast is not None:
                attrs["broadcast"] = broadcast
                attrs["broadcast_target"] = [rows, cols]
            upload = self.add_action(
                "upload", outputs=(view_id,), origin="host_pack:dataflow",
                attrs=attrs)
            self.value_views[value.id] = view_id
            self.value_action[value.id] = upload

    def _deps_for(self, value_ids: tuple[int, ...]) -> tuple[int, ...]:
        deps: list[int] = []
        for value_id in value_ids:
            action = self.value_action.get(value_id)
            if action is None:
                reject(RK_E_SHAPE,
                       f"value {value_id} has no device producer before use",
                       stage="dag")
            if action not in deps:
                deps.append(action)
        return tuple(deps)

    def _emit_binary_views(self, *, opcode: str, rows: int, cols: int,
                           cube_rows: int, cube_cols: int,
                           left_view: int, right_view: int,
                           deps: tuple[int, ...], group: str,
                           relu: bool = False,
                           role: str = "workspace",
                           reuse: bool = True,
                           op_ids: tuple[int, ...] = (),
                           out_view: int | None = None,
                           extra_attrs: dict[str, Any] | None = None) -> int:
        _, identity_view = self.identity_for(cube_cols)
        identity_upload = self._last_upload_of(identity_view)
        layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, cube_rows,
                              cube_cols, 8, cube_rows * cube_cols * 2)
        if out_view is None:
            out_buffer = self.add_buffer(role=role, capacity=layout.bytes,
                                         layout_id=CUBE_LAYOUT,
                                         padding_state="finite", reuse=reuse)
            out_view = self.add_view(out_buffer, 0, layout)
        if relu:
            if opcode != "add":
                reject(RK_E_UNSUPPORTED,
                       "only add can fuse the ReLU primitive", stage="dag")
            template = "co.ew_add_relu.f16.v1"
        else:
            template = ("co.ew_add.f16.v1" if opcode == "add"
                        else "co.ew_mul.f16.v1")
        self.add_action(
            "npu_task", template_id=template,
            inputs=(left_view, right_view, identity_view), outputs=(out_view,),
            deps=deps + ((identity_upload,) if identity_upload is not None else ()),
            origin=("lowering:dataflow:add_relu" if relu
                    else f"lowering:dataflow:{opcode}"),
            attrs={"submit_group": group, **(extra_attrs or {})},
            op_ids=op_ids)
        return out_view

    def _consume_multi_op_tasks(self, *, tasks: list[dict[str, Any]],
                                facts: dict[str, Any]) -> DraftPlan:
        """Consume token/region tasks with the shared value-state rules.

        The tasks carry actual buffer tokens and source/destination regions.
        The module supplies binding identity, NumericContract and attribution;
        execution order and publication come from the ordered tasks.
        """
        values = {value.id: value for value in self.module.values}
        ops = {op.id: op for op in self.module.ops}
        self._upload_external_values()
        identity_cols: set[int] = set()
        for task in tasks:
            if task["kind"] in {"silu", "sigmoid", "mul", "add"}:
                region = task["region"]
                identity_cols.add(cube_layout(region["rows"], region["cols"])[1])
        for cols in sorted(identity_cols):
            self.identity_for(cols)

        def rect(raw: dict[str, int]) -> _TaskRect:
            return _TaskRect(int(raw["row"]), int(raw["col"]),
                             int(raw["rows"]), int(raw["cols"]))

        states: dict[int, _TaskBufferState] = {}
        for token, value_id in (facts.get("external_tokens") or {}).items():
            value = values[int(value_id)]
            if value.role not in {"input", "constant"}:
                continue
            view = self.value_views.get(int(value_id))
            action = self.value_action.get(int(value_id))
            if view is None or action is None:
                reject(RK_E_SHAPE,
                       "an external token has no upload view", stage="dag",
                       actual=value_id)
            tensor = self.views[view].tensor
            shape = tuple(int(v) for v in value.type.shape)
            rows, cols = ((shape[0], shape[1]) if len(shape) == 2 else
                          (tensor.rows, tensor.cols))
            states[int(token)] = _TaskBufferState(
                value=_TaskRect(0, 0, rows, cols),
                value_view=view, value_action=action,
                value_producer=action, value_version=1)

        def state(token: int, role: str) -> _TaskBufferState:
            result = states.get(token)
            if result is None:
                reject(RK_E_SHAPE, f"{role} token has no established state",
                       stage="dag", actual=token)
            return result

        sigmoid_actions: dict[int, int] = {}
        for task in tasks:
            kind = str(task["kind"])
            if kind == "fill":
                token = int(task["buffer_token"])
                if int(task["value"]) != 0:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "a multi-op task-chain fill is not zero", stage="dag",
                           actual=task)
                zero = rect(task["region"])
                current = states.setdefault(token, _TaskBufferState())
                current.enter_zero(zero, len(self.actions))
                continue
            if kind in {"copy", "cast"}:
                source_token = int(task["source_token"])
                destination_token = int(task["destination_token"])
                source_region = rect(task["source_region"])
                destination_region = rect(task["destination_region"])
                source = states.get(source_token)
                if source is None:
                    source_value = int(task["source_value"])
                    source_view = self.value_views.get(source_value)
                    source_action = self.value_action.get(source_value)
                    if source_view is None or source_action is None:
                        reject(RK_E_SHAPE,
                               "a copy source has no device producer",
                               stage="dag", actual=source_value)
                    source = _TaskBufferState(
                        value=source_region, value_view=source_view,
                        value_action=source_action,
                        value_producer=source_action, value_version=1)
                if not source.value_covers(source_region):
                    reject(RK_E_SCHEMA,
                           "a copy source region is not covered by its value",
                           stage="dag", actual=task)
                if (source_region.rows, source_region.cols) != \
                        (destination_region.rows, destination_region.cols):
                    reject(RK_E_SHAPE,
                           "a fused copy/cast changes the region extent",
                           stage="dag", actual=task)
                destination = states.setdefault(
                    destination_token, _TaskBufferState())
                destination.adopt_value(destination_region, source)
                continue
            if kind == "store":
                source_token = int(task["source_token"])
                destination_token = int(task["destination_token"])
                source_region = rect(task["source_region"])
                destination_region = rect(task["destination_region"])
                source = state(source_token, "store source")
                if not source.value_covers(source_region) or \
                        source.value_view is None or source.value_action is None:
                    reject(RK_E_SCHEMA,
                           "store source has no covered device value",
                           stage="dag", actual=task)
                if (source_region.rows, source_region.cols) != \
                        (destination_region.rows, destination_region.cols):
                    reject(RK_E_SHAPE,
                           "the output store changes the region extent",
                           stage="dag", actual=task)
                binding_id = self.binding_by_value.get(
                    int(task["output_value"]))
                if binding_id is None:
                    reject(RK_E_SCHEMA,
                           "store output has no binding identity", stage="dag",
                           actual=task["output_value"])
                self.buffers[self.views[source.value_view].buffer_id].role = \
                    "output"
                download = self.add_action(
                    "download", inputs=(source.value_view,),
                    deps=(source.value_action,), origin="host:download",
                    attrs={"binding_id": binding_id, "submit_group": None})
                self.buffers[self.views[source.value_view].buffer_id].last = max(
                    self.buffers[self.views[source.value_view].buffer_id].last,
                    download)
                self.publish.append(Publish(
                    self.bindings[binding_id].name, binding_id,
                    source.value_view,
                    dst_row=destination_region.row,
                    dst_col=destination_region.col,
                    rows=destination_region.rows,
                    cols=destination_region.cols))
                destination = states.setdefault(
                    destination_token, _TaskBufferState())
                destination.adopt_value(destination_region, source)
                continue
            op_id = int(task["op_id"])
            op = ops[op_id]
            output_id = int(task["output_value"])
            output_token = int(task["output_token"])
            region = rect(task["region"])
            if kind == "gemm":
                input_tokens = [int(v) for v in task["input_tokens"]]
                a = state(input_tokens[0], "GEMM A")
                b = state(input_tokens[1], "GEMM B")
                a_region = rect(task["a_region"])
                b_region = rect(task["b_region"])
                if not a.value_covers(a_region) or \
                        not b.value_covers(b_region):
                    reject(RK_E_SCHEMA,
                           "GEMM operands do not cover their task regions",
                           stage="dag", actual=task)
                output = states.setdefault(output_token, _TaskBufferState())
                if output.zero is None or not output.zero.contains(region):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "a GEMM target has no consumed zero fill",
                           stage="dag", actual=output_token)
                cr, cc, cbytes = cube_layout(region.rows, region.cols)
                layout = TensorLayout("float16", CUBE_LAYOUT, region.rows,
                                      region.cols, cr, cc, 8, cbytes)
                buffer_id = self.add_buffer(role="accumulator",
                                            capacity=cbytes,
                                            layout_id=CUBE_LAYOUT,
                                            padding_state="finite")
                out_view = self.add_view(buffer_id, 0, layout)
                deps = tuple(dict.fromkeys(
                    action for action in (a.value_action, b.value_action)
                    if action is not None))
                group = f"mm:dataflow:{op_id}"
                action = self.add_action(
                    "npu_task", template_id="co.mm.f16.v1",
                    inputs=(a.value_view, b.value_view), outputs=(out_view,),
                    deps=deps, origin=f"lowering:dataflow:matmul:{op_id}",
                    op_ids=(op_id,),
                    attrs={"submit_group": group,
                           "source_k_loop": op.attrs.get("source_k_loop"),
                           "k0": int(task["k0"]), "kt": int(task["kt"])})
                output.record_value(region, action, out_view, action)
                self.value_views[output_id] = out_view
                self.value_action[output_id] = action
                continue
            if kind in {"silu", "sigmoid"}:
                source = state(int(task["input_tokens"][0]),
                               "activation input")
                if not source.value_covers(region) or source.value_view is None:
                    reject(RK_E_SCHEMA,
                           "SiLU input does not cover the task region",
                           stage="dag", actual=task)
                cr, cc, _ = cube_layout(region.rows, region.cols)
                group = f"act:dataflow:{op_id}"
                out_view = self._emit_activation(
                    activation=("silu" if kind == "silu" else "sigmoid"),
                    rows=region.rows, cols=region.cols,
                    cube_rows=cr, cube_cols=cc, input_view=source.value_view,
                    input_buffer=self.views[source.value_view].buffer_id,
                    deps=(source.value_action,), group=group,
                    op_ids=(op_id,))
                action = self._last_task_of_group(group)
                if kind == "sigmoid":
                    sigmoid_actions[output_token] = action
                output = states.setdefault(output_token, _TaskBufferState())
                output.record_value(region, action, out_view, action)
                self.value_views[output_id] = out_view
                self.value_action[output_id] = action
                continue
            if kind in {"mul", "add"}:
                input_tokens = [int(v) for v in task["input_tokens"]]
                input_values = [int(v) for v in task.get("input_values", ())]
                broadcasts = list(task.get("input_broadcasts", ()))
                if len(input_values) != 2 or len(broadcasts) != 2:
                    reject(RK_E_SCHEMA,
                           "a binary task must carry both operand values and "
                           "their broadcast relation", stage="dag", actual=task)
                for value_id, broadcast_kind in zip(input_values, broadcasts):
                    shape = tuple(int(v) for v in values[value_id].type.shape)
                    if broadcast_kind == "scalar":
                        valid = shape == ()
                    elif broadcast_kind == "channel":
                        valid = shape == (region.cols,)
                    elif broadcast_kind == "none":
                        valid = shape == (region.rows, region.cols)
                    else:
                        valid = False
                    if not valid:
                        reject(RK_E_SHAPE,
                               "the task broadcast relation does not match the "
                               "operand's carrier rank/shape", stage="dag",
                               actual={"value": value_id, "shape": shape,
                                       "broadcast": broadcast_kind,
                                       "region": region})
                left = state(input_tokens[0], "left operand")
                right = state(input_tokens[1], "right operand")
                if not left.value_covers(region) or \
                        not right.value_covers(region) or \
                        left.value_view is None or right.value_view is None:
                    reject(RK_E_SCHEMA,
                           "binary operands do not cover the task region",
                           stage="dag", actual=task)
                cr, cc, _ = cube_layout(region.rows, region.cols)
                group_id = int(task.get("group_op_id", op_id))
                group_kind = str(task.get("group_kind", "ew"))
                group = f"{group_kind}:dataflow:{group_id}"
                if kind == "mul" and group_kind == "act":
                    # The explicit sigmoid task already depends on the producer
                    # of its source value.  Keep that direct dependency and let
                    # the identity upload be added by _emit_binary_views.
                    deps = tuple(dict.fromkeys(
                        sigmoid_actions[token] for token in input_tokens
                        if token in sigmoid_actions))
                else:
                    deps = (left.value_action, right.value_action)
                out_view = self._emit_binary_views(
                    opcode=kind, rows=region.rows, cols=region.cols,
                    cube_rows=cr, cube_cols=cc,
                    left_view=left.value_view, right_view=right.value_view,
                    deps=deps, group=group, op_ids=(op_id,))
                action = self._last_task_of_group(group)
                output = states.setdefault(output_token, _TaskBufferState())
                output.value = region
                output.value_view = out_view
                output.value_action = action
                output.value_producer = action
                output.value_version += 1
                self.value_views[output_id] = out_view
                self.value_action[output_id] = action
                continue
            reject(RK_E_UNSUPPORTED,
                   f"no shared task consumer for {kind!r}", stage="dag",
                   actual={"task": task})
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build_instantiated_staged(self, *, tasks: list[dict[str, Any]],
                                  facts: dict[str, Any]) -> DraftPlan:
        """Consume per-launch tasks in one builder and one global value space."""
        if tasks:
            reject(RK_E_SCHEMA,
                   "staged consumption receives its task streams in facts",
                   stage="schedule")
        stages = list(facts.get("staged_stages") or ())
        if not stages:
            reject(RK_E_SHAPE, "staged task stream is empty", stage="schedule")
        staged_scheduled = bool(facts.get("staged_schedule_contract"))
        stage_op_ids = list(facts.get("staged_op_ids") or range(len(stages)))
        binding_by_name = {binding.name: binding.id for binding in self.bindings}
        binding_value = {value.name: value.id for value in self.module.values
                         if value.role in {"input", "constant", "output"}}
        global_regions: dict[tuple[str, int, int, int, int], tuple[int, int]] = {}
        pending_downloads: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
        upload_cache: dict[tuple[Any, ...], tuple[int, int]] = {}
        accumulator: dict[tuple[int, int, int], dict[str, Any]] = {}

        def lift_region(raw: dict[str, int]) -> tuple[int, int, int, int]:
            return (int(raw["row"]), int(raw["col"]),
                    int(raw["rows"]), int(raw["cols"]))

        def task_region(task: dict[str, Any], key: str | None = None
                        ) -> tuple[int, int, int, int]:
            if key is None:
                return (int(task["m0"]), int(task["n0"]),
                        int(task["rows"]), int(task["cols"]))
            raw = task[key]
            return (int(raw["row"]), int(raw["col"]),
                    int(raw["rows"]), int(raw["cols"]))

        producers: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
        consumers: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
        for stage in stages:
            index = int(stage["index"])
            for task in stage["tasks"]:
                if task.get("kind") == "store":
                    producers.setdefault(str(task["destination"]), []).append(
                        (index, task_region(task)))
                elif task.get("kind") == "copy" and task.get("source") not in binding_by_name:
                    consumers.setdefault(str(task["source"]), []).append(
                        (index, task_region(task, "source_region")))

        consumer_composites: dict[
            tuple[str, int, int, int, int], tuple[int, tuple[tuple[int, int, int, int], ...]]
        ] = {}
        producer_composites: dict[
            tuple[str, int, int, int, int], tuple[str, int, int, int, int]
        ] = {}
        stage_out_views: dict[tuple[int, str, int, int], int] = {}
        producer_subviews: dict[tuple[str, int, int, int, int], int] = {}
        for name, reads in consumers.items():
            writes = producers.get(name, [])
            for _consumer_stage, creg in reads:
                if (name, *creg) in consumer_composites:
                    continue
                exact = [item for item in writes if item[1] == creg]
                if len(exact) == 1:
                    continue
                inside = [item for item in writes
                          if item[1][0] >= creg[0] and item[1][1] >= creg[1] and
                          item[1][0] + item[1][2] <= creg[0] + creg[2] and
                          item[1][1] + item[1][3] <= creg[1] + creg[3]]
                if not inside:
                    continue
                rows, cols = creg[2], creg[3]
                if any(item[1][2] != rows for item in inside):
                    reject(RK_E_SHAPE,
                           "composite producer rows disagree with consumer rows",
                           stage="dag", actual={"consumer": creg,
                                                "producers": [i[1] for i in inside]})
                ordered = sorted(inside, key=lambda item: item[1][1])
                cursor = creg[1]
                for _idx, region in ordered:
                    if region[1] != cursor or region[1] % 8:
                        reject(RK_E_SHAPE,
                               "composite producer columns are not aligned or "
                               "contiguous", stage="dag", actual=region)
                    cursor += region[3]
                if cursor != creg[1] + creg[3]:
                    reject(RK_E_SHAPE,
                           "composite producers do not fully cover the consumer",
                           stage="dag", actual={"cursor": cursor,
                                                "consumer": creg})
                cr, cc, cbytes = cube_layout(rows, cols)
                full_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                           cr, cc, 8, cbytes)
                bid = self.add_buffer(role="workspace", capacity=cbytes,
                                      layout_id=CUBE_LAYOUT, reuse=False,
                                      padding_state="finite")
                full_view = self.add_view(bid, 0, full_layout)
                key = (name, *creg)
                consumer_composites[key] = (full_view, tuple(
                    item[1] for item in ordered))
                for stage_index, region in ordered:
                    rel_row = region[0] - creg[0]
                    rel_col = region[1] - creg[1]
                    if rel_row % 8:
                        reject(RK_E_SHAPE,
                               "composite producer row is not atom-aligned",
                               stage="dag", actual=region)
                    pr, pc, _ = cube_layout(region[2], region[3])
                    if pr != cr:
                        reject(RK_E_SHAPE,
                               "composite producer physical rows differ",
                               stage="dag", actual={"producer": pr,
                                                    "consumer": cr})
                    offset = ((rel_col // 8) * cr + rel_row) * 8 * 2
                    sub_layout = TensorLayout("float16", CUBE_LAYOUT,
                                              region[2], region[3], pr, pc, 8,
                                              pr * pc * 2)
                    sub = self.add_view(bid, offset, sub_layout)
                    stage_out_views[(stage_index, "", region[0], region[1])] = sub
                    producer_composites[(name, *region)] = key
                    producer_subviews[(name, *region)] = sub

        composite_actions: dict[tuple[str, int, int, int, int], list[int]] = {}

        def rect(raw: dict[str, int]) -> tuple[int, int, int, int]:
            return (int(raw["row"]), int(raw["col"]),
                    int(raw["rows"]), int(raw["cols"]))

        def upload(source: str, region: dict[str, int], *, weight: bool,
                   stage: dict[str, Any]) -> tuple[int, int]:
            if source not in binding_by_name:
                reject(RK_E_SHAPE,
                       "only an external binding can create an upload",
                       stage="schedule", actual=source)
            binding = binding_by_name[source]
            key = (binding, rect(region), weight)
            if key in upload_cache:
                return upload_cache[key]
            rows, cols = int(region["rows"]), int(region["cols"])
            if weight:
                pr, pc, pbytes = weight_layout(cols, rows)
                layout = TensorLayout("float16", WEIGHT_LAYOUT, cols, rows,
                                      pr, pc, 32, pbytes)
                bid = self.add_buffer(role="weight", capacity=pbytes,
                                      layout_id=WEIGHT_LAYOUT, reuse=False,
                                      padding_state="zero")
                view = self.add_view(bid, 0, layout)
                action = self.add_action(
                    "upload", outputs=(view,),
                    origin="host_pack:staged:weight",
                    attrs={"source_binding": binding,
                           "src_row_offset": int(region["row"]),
                           "src_col_offset": int(region["col"]),
                           "src_row_stride": int(self.bindings[binding].shape[1]),
                           "transpose_source": True,
                           "cache": ("constant" if
                                     self.bindings[binding].role == "constant"
                                     else "input")})
            else:
                pr, pc, pbytes = input_layout(rows, cols)
                layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      pr, pc, 8, pbytes)
                bid = self.add_buffer(role="input_slice", capacity=pbytes,
                                      layout_id=CUBE_LAYOUT, reuse=False,
                                      padding_state="zero")
                view = self.add_view(bid, 0, layout)
                action = self.add_action(
                    "upload", outputs=(view,),
                    origin="host_pack:staged:input",
                    attrs={"source_binding": binding,
                           "src_row_offset": int(region["row"]),
                           "src_col_offset": int(region["col"]),
                           "src_row_stride": int(self.bindings[binding].shape[1]),
                           "transpose_source": False, "cache": "input"})
            upload_cache[key] = (view, action)
            return upload_cache[key]

        # Uploads are host actions; establish every external pack before any
        # device task group so a launch cannot contain an interleaved host step.
        for stage in stages:
            for task in stage["tasks"]:
                if task.get("kind") != "copy":
                    continue
                source = str(task["source"])
                if source not in binding_by_name:
                    continue
                weight = self.bindings[binding_by_name[source]].role == "constant"
                upload(source, task["source_region"], weight=weight, stage=stage)

        for stage_index, stage in enumerate(stages):
            output_shapes = {(int(t["m0"]), int(t["n0"])):
                             (int(t["rows"]), int(t["cols"]))
                             for t in stage["tasks"] if t.get("kind") == "store"}
            local_views: dict[str, tuple[int, int] | None] = {}
            local_deps: dict[str, tuple[int, ...]] = {}
            local_sources: dict[str, dict[str, Any]] = {}
            local_zero: set[str] = set()
            current_group = None
            for task in stage["tasks"]:
                kind = task.get("kind")
                if kind == "fill":
                    name = str(task["destination"])
                    local_views[name] = None
                    local_zero.add(name)
                    continue
                if kind == "copy":
                    source = str(task["source"])
                    destination = str(task["destination"])
                    source_region = task["source_region"]
                    if source in binding_by_name:
                        weight = self.bindings[binding_by_name[source]].role == "constant"
                        uploaded = upload(
                            source, source_region, weight=weight, stage=stage)
                        local_views[destination] = uploaded
                        local_sources[destination] = {
                            "source": source, "region": dict(source_region),
                            **{"view": uploaded[0], "action": uploaded[1]}}
                    elif (source, *rect(source_region)) in consumer_composites:
                        key = (source, *rect(source_region))
                        full_view, _regions = consumer_composites[key]
                        actions = tuple(composite_actions.get(key, ()))
                        if not actions:
                            reject(RK_E_SHAPE,
                                   "composite consumer has no producer actions",
                                   stage="schedule", actual=key)
                        local_views[destination] = (full_view, actions[0])
                        local_deps[destination] = actions
                        local_sources[destination] = {
                            "source": source, "region": dict(source_region),
                            "view": full_view, "action": actions[0]}
                    elif any(key[0] == source for key in global_regions):
                        key = (source, *rect(source_region))
                        found = global_regions.get(key)
                        if found is None:
                            reject(RK_E_SHAPE,
                                   "intermediate copy region has no prior write",
                                   stage="schedule", actual=key)
                        local_views[destination] = found
                        local_sources[destination] = {
                            "source": source, "region": dict(source_region),
                            **{"view": found[0], "action": found[1]}}
                    elif source in local_views:
                        local_views[destination] = local_views[source]
                    else:
                        reject(RK_E_SHAPE,
                               "copy source is neither external nor staged",
                               stage="schedule", actual=source)
                    local_zero.discard(destination)
                    continue
                if kind in {"sigmoid", "mul", "add"}:
                    names = list(task["inputs"])
                    states = [local_views.get(str(name)) for name in names]
                    if any(state is None for state in states):
                        reject(RK_E_SHAPE,
                               "elementwise inputs have no staged source",
                               stage="schedule", actual=names)
                    region_raw = task["region"]
                    region = (int(region_raw["row"]), int(region_raw["col"]),
                              int(region_raw["rows"]), int(region_raw["cols"]))
                    rows, cols = region[2], region[3]
                    cr, cc, _ = cube_layout(rows, cols)
                    self.identity_for(cc)
                    m0, n0 = int(task.get("m0", 0)), int(task.get("n0", 0))
                    input_offsets = list(task.get("input_offsets") or
                                         [0] * len(states))
                    producer_sources = []
                    for name, state, offset in zip(names, states,
                                                   input_offsets):
                        producer_sources.append({
                            "input": name, "view": state[0],
                            "action": state[1], "offset": int(offset),
                            "half": "gate" if int(offset) == 0 else "up"})
                    extra = {
                        "source_stage_index": stage_index,
                        "source_stage_count": len(stages),
                        "source_stage_boundary": "kernel_launch",
                        "stage_barrier": "all_previous_submits",
                        "producer_sources": producer_sources}
                    if kind == "sigmoid":
                        extra["gate_source"] = producer_sources[0]
                    if kind == "mul":
                        for source in producer_sources:
                            if source["half"] == "up":
                                extra["up_source"] = source
                            else:
                                extra["gate_source"] = source
                    marker = "act" if kind == "sigmoid" else "ew"
                    if len(stages) > 2 and not staged_scheduled:
                        group = f"{marker}:dataflow_tiled:{stage_index}:{m0}:{n0}"
                    else:
                        group = f"{marker}:s{stage_index}:{m0}:{n0}"
                    if kind == "sigmoid":
                        out_view = self._emit_activation(
                            activation="sigmoid", rows=rows, cols=cols,
                            cube_rows=cr, cube_cols=cc,
                            input_view=states[0][0],
                            input_buffer=self.views[states[0][0]].buffer_id,
                            deps=(states[0][1],), group=group,
                            op_ids=(stage_op_ids[stage_index],),
                            extra_attrs=extra)
                    else:
                        if len(states) != 2:
                            reject(RK_E_SCHEMA,
                                   "staged binary task needs two inputs",
                                   stage="schedule", actual=task)
                        store_task = next(
                            (t for t in stage["tasks"]
                             if t.get("kind") == "store" and
                             t.get("source") == task["output"] and
                             int(t.get("m0", -1)) == m0 and
                             int(t.get("n0", -1)) == n0), None)
                        explicit = None
                        if store_task is not None:
                            dreg = lift_region(store_task["destination_region"])
                            explicit = producer_subviews.get(
                                (str(store_task["destination"]), *dreg))
                        out_view = self._emit_binary_views(
                            opcode=kind, rows=rows, cols=cols,
                            cube_rows=cr, cube_cols=cc,
                            left_view=states[0][0], right_view=states[1][0],
                            deps=(states[0][1], states[1][1]),
                            group=group, op_ids=(stage_op_ids[stage_index],),
                            out_view=explicit, extra_attrs=extra)
                    action = self._last_task_of_group(group)
                    local_views[str(task["output"])] = (out_view, action)
                    continue
                if kind != "gemm":
                    if kind == "store":
                        source = str(task["source"])
                        destination = str(task["destination"])
                        source_state = local_views.get(source)
                        if source_state is None:
                            reject(RK_E_SHAPE,
                                   "store source has no staged value",
                                   stage="schedule", actual=source)
                        view, action = source_state
                        region = (int(task["m0"]), int(task["n0"]),
                                  int(task["rows"]), int(task["cols"]))
                        if destination in binding_by_name and                                 self.bindings[binding_by_name[destination]].role == "output":
                            pending_downloads.append(
                                (binding_by_name[destination], view,
                                 action, region))
                        else:
                            global_regions[(destination, *region)] = source_state
                            composite_key = producer_composites.get(
                                (destination, *region))
                            if composite_key is not None:
                                composite_actions.setdefault(
                                    composite_key, []).append(action)
                        continue
                    reject(RK_E_UNSUPPORTED,
                           f"staged task kind {kind!r} is not consumed",
                           stage="schedule", actual=task)
                tiles = task.get("tiles")
                if not isinstance(tiles, list) or len(tiles) != 3:
                    reject(RK_E_SCHEMA, "staged GEMM task lacks tile names",
                           stage="schedule", actual=task)
                a_state = local_views.get(str(tiles[0]))
                b_state = local_views.get(str(tiles[1]))
                a_source = local_sources.get(str(tiles[0]))
                if a_state is None or b_state is None:
                    reject(RK_E_SHAPE,
                           "staged GEMM operands have no source view",
                           stage="schedule", actual=tiles)
                m0, n0 = int(task["m0"]), int(task["n0"])
                rows, cols = output_shapes.get(
                    (m0, n0), (int(task["rows"]), int(task["cols"])))
                k0, kt = int(task["k0"]), int(task["kt"])
                effective_kt = min(kt, int(stage["logical_k"]) - k0)
                group_key = (stage_index, m0, n0)
                group = accumulator.get(group_key)
                if group is None:
                    cr, cc, cbytes = cube_layout(rows, cols)
                    layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                          cr, cc, 8, cbytes)
                    group = {"views": [], "layout": layout,
                             "capacity": (cr, cc, cbytes),
                             "action": None, "last": None, "count": 0}
                    accumulator[group_key] = group
                index = int(group["count"])
                group["count"] = index + 1
                slot = index & 1
                while len(group["views"]) <= slot:
                    _cr, _cc, _cbytes = group["capacity"]
                    bid = self.add_buffer(role="accumulator",
                                          capacity=_cbytes,
                                          layout_id=CUBE_LAYOUT,
                                          padding_state="finite")
                    group["views"].append(
                        self.add_view(bid, 0, group["layout"]))
                out_view = group["views"][slot]
                a_view, a_action = a_state
                b_view, b_action = b_state
                cl_state = local_views.get(str(tiles[2]))
                deps = list(local_deps.get(str(tiles[0]), ()))
                deps += list(local_deps.get(str(tiles[1]), ()))
                if not deps:
                    deps = [a_action, b_action]
                elif a_action not in deps:
                    deps.append(a_action)
                if b_action not in deps:
                    deps.append(b_action)
                z_attrs = {}
                if a_source is not None and a_source["source"] == "Z":
                    source_region = a_source["region"]
                    z_attrs["z_source"] = {
                        "value": "Z",
                        "producer_view": int(a_view),
                        "producer_action": int(a_action),
                        "producer_tile": [int(source_region["row"]),
                                          int(source_region["col"]),
                                          int(source_region["rows"]),
                                          int(source_region["cols"])],
                        "consumer_region": [int(source_region["row"]),
                                            int(source_region["col"]),
                                            int(source_region["rows"]),
                                            int(source_region["cols"])],
                        "logical_k": effective_kt,
                        "physical_cols": effective_kt,
                        "row_offset": 0,
                        "padding": "consumer reads valid columns only",
                    }
                if cl_state is None:
                    action = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(a_view, b_view), outputs=(out_view,),
                        deps=tuple(deps), origin=f"lowering:staged:stage{stage_index}:mm:{k0}",
                        attrs={"submit_group": (
                            f"mm:s{stage_index}:{stage_index}:{m0}:{n0}"
                            if len(stages) == 2 else
                            f"mm:s{stage_index}:{m0}:{n0}"
                            if staged_scheduled else
                            f"mm:dataflow_tiled:{stage_index}:{m0}:{n0}"),
                               "source_k_loop": stage.get("source_k_loop"),
                               "k0": 0,
                               "kt": effective_kt,
                               "k_physical_extent": effective_kt,
                               "k_padding": 0,
                               "source_stage_index": stage_index,
                               "source_stage_count": len(stages),
                               "source_stage_boundary": "kernel_launch",
                               "source_stage_schedule_contract":
                                   "staged_mn_tiles_v1",
                               "source_schedule": {
                                   "kind": "mn_tiles_v1",
                                   "tile_m": int(stage["tile_m"]),
                                   "tile_n": int(stage["tile_n"]),
                                   "order": str(stage["order"]),
                                   "k_tile": 256,
                                   "accumulation":
                                       "ordered_ascending_f16_writeback"},
                               "stage_barrier": "all_previous_submits",
                               "producer_sources": [
                                   {"input": "A", "view": a_view,
                                    "action": a_action,
                                    "actions": list(local_deps.get(str(tiles[0]), (a_action,)))},
                                   {"input": "B", "view": b_view,
                                    "action": b_action,
                                    "actions": list(local_deps.get(str(tiles[1]), (b_action,)))}],
                               "activation_sources": [
                                   {"view": a_view, "action": act}
                                   for act in local_deps.get(str(tiles[0]),
                                                             (a_action,))],
                               **z_attrs},
                        op_ids=(stage_op_ids[stage_index],))
                else:
                    action = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(a_view, b_view, cl_state[0]),
                        outputs=(out_view,),
                        deps=tuple(deps + [cl_state[1]]),
                        origin=f"lowering:staged:stage{stage_index}:mm_acc:{k0}",
                        attrs={"submit_group": (
                            f"mm:s{stage_index}:{stage_index}:{m0}:{n0}"
                            if len(stages) == 2 else
                            f"mm:s{stage_index}:{m0}:{n0}"
                            if staged_scheduled else
                            f"mm:dataflow_tiled:{stage_index}:{m0}:{n0}"),
                               "source_k_loop": stage.get("source_k_loop"),
                               "k0": k0,
                               "kt": effective_kt,
                               "k_physical_extent": effective_kt,
                               "k_padding": 0,
                               "source_stage_index": stage_index,
                               "source_stage_count": len(stages),
                               "source_stage_boundary": "kernel_launch",
                               "source_stage_schedule_contract":
                                   "staged_mn_tiles_v1",
                               "source_schedule": {
                                   "kind": "mn_tiles_v1",
                                   "tile_m": int(stage["tile_m"]),
                                   "tile_n": int(stage["tile_n"]),
                                   "order": str(stage["order"]),
                                   "k_tile": 256,
                                   "accumulation":
                                       "ordered_ascending_f16_writeback"},
                               "stage_barrier": "all_previous_submits",
                               "producer_sources": [
                                   {"input": "A", "view": a_view,
                                    "action": a_action,
                                    "actions": list(local_deps.get(str(tiles[0]), (a_action,)))},
                                   {"input": "B", "view": b_view,
                                    "action": b_action,
                                    "actions": list(local_deps.get(str(tiles[1]), (b_action,)))}],
                               "activation_sources": [
                                   {"view": a_view, "action": act}
                                   for act in local_deps.get(str(tiles[0]),
                                                             (a_action,))],
                               **z_attrs},
                        op_ids=(stage_op_ids[stage_index],))
                group["last"] = out_view
                group["action"] = action
                local_views[str(tiles[2])] = (out_view, action)
                local_zero.discard("Acc")
        for binding, view, producer, region in pending_downloads:
            download = self.add_action(
                "download", inputs=(view,), deps=(producer,),
                origin="host:download:staged",
                attrs={"binding_id": binding, "submit_group": None})
            self.buffers[self.views[view].buffer_id].last = max(
                self.buffers[self.views[view].buffer_id].last, download)
            self.publish.append(Publish(
                self.bindings[binding].name, binding, view,
                dst_row=region[0], dst_col=region[1],
                rows=region[2], cols=region[3]))
        return DraftPlan(name=self.module.name,
                         profile_hash=self.profile.profile_hash,
                         source_hash=self.module.source_hash,
                         bindings=self.bindings, constants=self.constants,
                         buffers=self.buffers, views=self.views,
                         actions=self.actions, publish=self.publish)

    def build_dataflow(self) -> DraftPlan:
        """Lower a legalized per-node SSA graph onto device views.

        Every RKIR value gets exactly one device view and one producer action.
        The second GEMM consumes the first stage's cube view directly, so the
        activation result is never downloaded and re-uploaded.
        """
        self._upload_external_values()
        # The C runtime stages every Host upload before it runs any NPU submit.
        # Identity weights must therefore be uploaded here, before the first
        # task action: allocating them lazily between submit groups would make
        # action-id liveness disagree with the runtime's Host-first phase.
        identity_cols: set[int] = set()
        for op in self.module.ops:
            if op.opcode in {"relu", "silu", "gelu", "rsqrt"}:
                rows, cols = self.module.value(op.inputs[0]).type.shape
                identity_cols.add(cube_layout(rows, cols)[1])
            elif op.opcode in {"add", "mul"}:
                rows, cols = self.module.value(op.outputs[0]).type.shape
                identity_cols.add(cube_layout(rows, cols)[1])
        for cols in sorted(identity_cols):
            self.identity_for(cols)
        for op in self.module.ops:
            if op.opcode == "relu" and op.id not in self._skip_fused_relu:
                rows, cols = self.module.value(op.inputs[0]).type.shape
                self.zero_for(rows, cols)
        for op in self.module.ops:
            if op.opcode == "matmul":
                x_view = self.value_views.get(op.inputs[0])
                w_view = self.value_views.get(op.inputs[1])
                if x_view is None or w_view is None:
                    reject(RK_E_SHAPE,
                           "matmul operand has no device view", stage="dag",
                           op=op.opcode, action_id=None)
                out_value = self.module.value(op.outputs[0])
                rows, cols = out_value.type.shape
                cr, cc, cbytes = cube_layout(rows, cols)
                layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cr, cc, 8, cbytes)
                buffer_id = self.add_buffer(role="accumulator",
                                            capacity=cbytes,
                                            layout_id=CUBE_LAYOUT,
                                            padding_state="finite")
                out_view = self.add_view(buffer_id, 0, layout)
                group = f"mm:dataflow:{op.id}"
                task = self.add_action(
                    "npu_task", template_id="co.mm.f16.v1",
                    inputs=(x_view, w_view), outputs=(out_view,),
                    deps=self._deps_for(op.inputs),
                    origin=f"lowering:dataflow:matmul:{op.id}",
                    op_ids=(op.id,),
                    attrs={"submit_group": group,
                           "source_k_loop": op.attrs.get("source_k_loop"),
                           "k0": 0, "kt": self.module.value(op.inputs[0]).type.cols})
                self.value_views[out_value.id] = out_view
                self.value_action[out_value.id] = task
                continue

            if op.id in self._skip_fused_relu:
                continue
            if op.opcode in {"relu", "silu", "gelu", "sigmoid", "hardsigmoid",
                             "rsqrt"}:
                in_id = op.inputs[0]
                in_view = self.value_views.get(in_id)
                if in_view is None:
                    reject(RK_E_SHAPE,
                           "activation input has no device view", stage="dag",
                           op=op.opcode, action_id=None)
                rows, cols = self.module.value(in_id).type.shape
                cr, cc, _ = cube_layout(rows, cols)
                group = f"act:dataflow:{op.id}"
                out_view = self._emit_activation(
                    activation=op.opcode, rows=rows, cols=cols,
                    cube_rows=cr, cube_cols=cc, input_view=in_view,
                    input_buffer=self.views[in_view].buffer_id,
                    deps=self._deps_for(op.inputs), group=group,
                    op_ids=(op.id,))
                last = self._last_task_of_group(group)
                out_buffer = self.views[out_view].buffer_id
                self.buffers[out_buffer].role = "workspace"
                out_id = op.outputs[0]
                self.value_views[out_id] = out_view
                self.value_action[out_id] = last
                continue

            if op.opcode == "add" and op.id in self._fused_add_relu:
                relu_id = self._fused_add_relu[op.id]
                relu_out = self.module.op(relu_id).outputs[0]
                left_id, right_id = op.inputs
                left_view = self.value_views.get(left_id)
                right_view = self.value_views.get(right_id)
                if left_view is None or right_view is None:
                    reject(RK_E_SHAPE,
                           "fused add operands have no device view", stage="dag",
                           op=op.opcode, action_id=None)
                out_value = self.module.value(relu_out)
                rows, cols = out_value.type.shape
                cr, cc, _ = cube_layout(rows, cols)
                left = self.views[left_view].tensor
                right = self.views[right_view].tensor
                if left.physical_rows != cr or left.physical_cols != cc or \
                        right.physical_rows != cr or right.physical_cols != cc:
                    reject(RK_E_SHAPE,
                           "fused add operands do not share the same physical cube "
                           "layout", stage="dag", op=op.opcode)
                group = f"ew:dataflow:{op.id}"
                out_view = self._emit_binary_views(
                    opcode="add", rows=rows, cols=cols,
                    cube_rows=cr, cube_cols=cc, left_view=left_view,
                    right_view=right_view, deps=self._deps_for(op.inputs),
                    group=group, relu=True,
                    role=("output" if relu_out in self.module.outputs
                          else "workspace"),
                    reuse=relu_out not in self.module.outputs,
                    op_ids=(op.id, relu_id))
                last = self._last_task_of_group(group)
                self.value_views[op.outputs[0]] = out_view
                self.value_action[op.outputs[0]] = last
                self.value_views[relu_out] = out_view
                self.value_action[relu_out] = last
                continue

            if op.opcode in {"add", "mul"}:
                left_id, right_id = op.inputs
                left_view = self.value_views.get(left_id)
                right_view = self.value_views.get(right_id)
                if left_view is None or right_view is None:
                    reject(RK_E_SHAPE,
                           "binary operand has no device view", stage="dag",
                           op=op.opcode, action_id=None)
                out_value = self.module.value(op.outputs[0])
                rows, cols = out_value.type.shape
                cr, cc, _ = cube_layout(rows, cols)
                left = self.views[left_view].tensor
                right = self.views[right_view].tensor
                if left.physical_rows != cr or left.physical_cols != cc or \
                        right.physical_rows != cr or right.physical_cols != cc:
                    reject(RK_E_SHAPE,
                           "binary operands do not share the same physical cube "
                           "layout", stage="dag", op=op.opcode)
                group = f"ew:dataflow:{op.id}"
                out_view = self._emit_binary_views(
                    opcode=op.opcode, rows=rows, cols=cols,
                    cube_rows=cr, cube_cols=cc, left_view=left_view,
                    right_view=right_view, deps=self._deps_for(op.inputs),
                    group=group, op_ids=(op.id,))
                last = self._last_task_of_group(group)
                out_id = op.outputs[0]
                self.value_views[out_id] = out_view
                self.value_action[out_id] = last
                continue

            reject(RK_E_UNSUPPORTED,
                   f"no device lowering for legalized node {op.opcode!r}",
                   stage="dag", op=op.opcode, source=op.source)

        # Export every declared output.  An output may be an intermediate that
        # also feeds a later node; its view is never rewritten in place, and the
        # download action extends its lifetime until publish.
        for output_id in self.module.outputs:
            final_view = self.value_views.get(output_id)
            final_task = self.value_action.get(output_id)
            if final_view is None or final_task is None:
                reject(RK_E_SHAPE, "module output has no device producer",
                       stage="dag", actual=output_id)
            self.buffers[self.views[final_view].buffer_id].role = "output"
            binding_id = self.binding_by_value[output_id]
            download = self.add_action(
                "download", inputs=(final_view,), deps=(final_task,),
                origin="host:download",
                attrs={"binding_id": binding_id, "submit_group": None})
            self.buffers[self.views[final_view].buffer_id].last = max(
                self.buffers[self.views[final_view].buffer_id].last, download)
            rows, cols = self.module.value(output_id).type.shape
            self.publish.append(Publish(
                self.bindings[binding_id].name, binding_id, final_view,
                dst_row=0, dst_col=0, rows=rows, cols=cols))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build_dataflow_tiled(self) -> DraftPlan:
        """Lower a legalized SSA graph with a tile set per RKIR value.

        Each rank-2 value is partitioned into the fixed 256x256 logical grid.  A
        consumer reads the producer tile that owns the required (row, K) or
        (row, column) rectangle; device intermediates are never downloaded and
        re-uploaded.  All Host uploads (feature/constant tiles, packed weight
        tiles, broadcast expansions and identity weights) are emitted before the
        first NPU task, and downloads are emitted only after every task.
        """
        for op in self.module.ops:
            if op.opcode == "matmul" and "source_schedule" in op.attrs:
                reject(RK_E_UNSUPPORTED,
                       ("this source proves M/N blocks that the multi-node "
                        "dataflow planner cannot honour: it partitions every "
                        "value on the fixed 256x256 grid.  Annotate blocks the "
                        "admitted contract supports, or write the kernel on the "
                        "256 grid instead of leaving its blocks implicit"
                        if op.attrs.get("source_schedule_origin") == "derived"
                        else "explicit M/N scheduling is not implemented for "
                             "multi-node dataflow plans yet; refusing instead of "
                             "silently ignoring source_schedule"),
                       stage="schedule", op=op.opcode, source=op.source,
                       actual=op.attrs.get("source_schedule"))
        external = self._external_layouts()
        broadcast_targets = self._broadcast_targets()

        value_tiles: dict[int, tuple] = {}
        for value in self.module.values:
            if value.type.rank == 2:
                value_tiles[value.id] = output_tiles(*value.type.shape)
        tile_views: dict[int, dict[tuple[int, int], int]] = {
            value.id: {} for value in self.module.values}
        tile_actions: dict[int, dict[tuple[int, int], int]] = {
            value.id: {} for value in self.module.values}

        def unique(items):
            out = []
            for item in items:
                if item is not None and item not in out:
                    out.append(item)
            return tuple(out)

        # 1. Upload every external rank-2 cube value.  A value wider than one
        # 256-column tile is packed once per 256-row tile and reused through
        # stable column-offset views.  The previous shape-per-tile uploads
        # multiplied BO slots, cache prep/fini calls and publication-gate scan
        # work without changing bytes or math.  Column offset n0*physical_rows*2
        # is the canonical K-major CUBE offset because every tile boundary is
        # 256 columns, a multiple of the 8-element atom.
        for value in self.module.values:
            if value.role not in {"input", "constant"}:
                continue
            kind, _ = external[value.id]
            if kind != "cube" or value.type.rank != 2:
                continue
            binding = self.binding_by_value[value.id]
            value_rows, value_cols = value.type.shape
            for m0 in range(0, value_rows, TILE_LIMIT):
                tile_rows = min(TILE_LIMIT, value_rows - m0)
                physical_rows = round_up(max(tile_rows, 12), 4)
                physical_cols = round_up(value_cols, 32)
                full_bytes = physical_rows * physical_cols * 2
                full_layout = TensorLayout(
                    "float16", CUBE_LAYOUT, tile_rows, value_cols,
                    physical_rows, physical_cols, 8, full_bytes)
                buffer_id = self.add_buffer(
                    role="input_slice", capacity=full_bytes,
                    layout_id=CUBE_LAYOUT, reuse=False, padding_state="zero")
                full_view = self.add_view(buffer_id, 0, full_layout)
                upload = self.add_action(
                    "upload", outputs=(full_view,),
                    origin="host_pack:dataflow_tiled:cube",
                    attrs={
                        "source_binding": binding,
                        "src_row_offset": m0,
                        "src_col_offset": 0,
                        "src_row_stride": value_cols,
                        "transpose_source": False,
                        "cache": ("constant" if value.role == "constant"
                                  else "input"),
                    })
                for n0 in range(0, value_cols, TILE_LIMIT):
                    cols = min(TILE_LIMIT, value_cols - n0)
                    cr, cc, cbytes = cube_layout(tile_rows, cols)
                    # cube_layout's physical rows must agree with the full
                    # buffer: rows are rounded to 4 with a 12-row floor.
                    if cr != physical_rows:
                        reject(RK_E_SHAPE,
                               "external tile row layout drifted from its "
                               "packed buffer", stage="dag",
                               actual={"value": value.name,
                                       "tile": cr, "buffer": physical_rows})
                    layout = TensorLayout("float16", CUBE_LAYOUT, tile_rows,
                                          cols, cr, cc, 8, cbytes)
                    view = self.add_view(
                        buffer_id, n0 * physical_rows * 2, layout)
                    key = (m0, n0)
                    tile_views[value.id][key] = view
                    tile_actions[value.id][key] = upload

        # 2. Pack every (N tile, K slice) weight rectangle needed by a matmul.
        weight_views: dict[tuple, int] = {}
        weight_actions: dict[tuple, int] = {}
        for op in self.module.ops:
            if op.opcode != "matmul":
                continue
            b_value = self.module.value(op.inputs[1])
            b_layout = op.attrs.get("b_layout", "kn")
            k = self.module.value(op.inputs[0]).type.cols
            if b_value.type.rank != 2:
                reject(RK_E_SHAPE, "matmul weight must be rank 2", stage="dag",
                       op=op.opcode)
            binding = self.binding_by_value[b_value.id]
            for tile in value_tiles[op.outputs[0]]:
                for kslice in k_slices(k):
                    key = (b_value.id, tile.n0, kslice.k0, tile.cols,
                           kslice.extent, b_layout)
                    if key in weight_views:
                        continue
                    wr, wc, wbytes = weight_layout(tile.cols, kslice.extent)
                    layout = TensorLayout("float16", WEIGHT_LAYOUT, tile.cols,
                                          kslice.extent, wr, wc, 32, wbytes)
                    buffer_id = self.add_buffer(
                        role="weight", capacity=wbytes,
                        layout_id=WEIGHT_LAYOUT, reuse=False,
                        padding_state="zero")
                    view = self.add_view(buffer_id, 0, layout)
                    if b_layout == "kn":
                        src_row, src_col = kslice.k0, tile.n0
                        transpose = True
                    else:
                        src_row, src_col = tile.n0, kslice.k0
                        transpose = False
                    upload = self.add_action(
                        "upload", outputs=(view,),
                        origin="host_pack:dataflow_tiled:weight",
                        attrs={
                            "source_binding": binding,
                            "src_row_offset": src_row,
                            "src_col_offset": src_col,
                            "src_row_stride": b_value.type.shape[1],
                            "transpose_source": transpose,
                            "cache": ("constant"
                                      if b_value.role == "constant"
                                      else "input"),
                        })
                    weight_views[key] = view
                    weight_actions[key] = upload

        # 3. Expand scalar/channel constants once per output tile.  The source
        # column offset selects the channel window for this tile.
        broadcast_views: dict[tuple[int, int, int], int] = {}
        broadcast_actions: dict[tuple[int, int, int], int] = {}
        for op in self.module.ops:
            if op.opcode not in {"add", "mul"}:
                continue
            out_id = op.outputs[0]
            for value_id in op.inputs:
                value = self.module.value(value_id)
                if not (value.role == "constant" and value.type.rank < 2):
                    continue
                target = broadcast_targets.get(value_id)
                if target is None or target != self.module.value(out_id).type.shape:
                    reject(RK_E_SHAPE,
                           "broadcast constant target does not match binary output",
                           stage="dag", op=op.opcode)
                binding = self.binding_by_value[value_id]
                for tile in value_tiles[out_id]:
                    key = (value_id, tile.m0, tile.n0)
                    if key in broadcast_views:
                        continue
                    cr, cc, cbytes = cube_layout(tile.rows, tile.cols)
                    layout = TensorLayout("float16", CUBE_LAYOUT, tile.rows,
                                          tile.cols, cr, cc, 8, cbytes)
                    buffer_id = self.add_buffer(
                        role="input_slice", capacity=cbytes,
                        layout_id=CUBE_LAYOUT, reuse=False,
                        padding_state="zero")
                    view = self.add_view(buffer_id, 0, layout)
                    upload = self.add_action(
                        "upload", outputs=(view,),
                        origin="host_pack:dataflow_tiled:broadcast",
                        attrs={
                            "source_binding": binding,
                            "src_row_offset": 0,
                            "src_col_offset": (tile.n0
                                               if value.type.rank == 1 else 0),
                            "src_row_stride": 0,
                            "transpose_source": False,
                            "cache": "constant",
                            "broadcast": ("scalar" if value.type.rank == 0
                                          else "channel"),
                            "broadcast_target": [tile.rows, tile.cols],
                        })
                    broadcast_views[key] = view
                    broadcast_actions[key] = upload

        # 4. Identity weights are Host uploads too, so create every distinct
        # physical width before the first task action.
        identity_cols: set[int] = set()
        for op in self.module.ops:
            if op.opcode in {"relu", "silu", "gelu", "sigmoid", "hardsigmoid"}:
                for tile in value_tiles[op.inputs[0]]:
                    identity_cols.add(cube_layout(tile.rows, tile.cols)[1])
            elif op.opcode in {"add", "mul"}:
                for tile in value_tiles[op.outputs[0]]:
                    identity_cols.add(cube_layout(tile.rows, tile.cols)[1])
        for cols in sorted(identity_cols):
            self.identity_for(cols)
        for op in self.module.ops:
            if op.opcode == "relu" and op.id not in self._skip_fused_relu:
                for tile in value_tiles[op.inputs[0]]:
                    self.zero_for(tile.rows, tile.cols)

        # 5. Emit NPU tasks tile by tile in topological order.
        for op in self.module.ops:
            if op.opcode == "matmul":
                a_value = self.module.value(op.inputs[0])
                b_value = self.module.value(op.inputs[1])
                out_value = self.module.value(op.outputs[0])
                b_layout = op.attrs.get("b_layout", "kn")
                ktiles = k_slices(a_value.type.cols)
                for tile in value_tiles[out_value.id]:
                    key_out = (tile.m0, tile.n0)
                    cr, cc, cbytes = cube_layout(tile.rows, tile.cols)
                    acc_views: list[int] = []
                    for _ in range(1 if len(ktiles) == 1 else 2):
                        layout = TensorLayout("float16", CUBE_LAYOUT,
                                              tile.rows, tile.cols, cr, cc, 8,
                                              cbytes)
                        buffer_id = self.add_buffer(
                            role="accumulator", capacity=cbytes,
                            layout_id=CUBE_LAYOUT, padding_state="finite")
                        acc_views.append(self.add_view(buffer_id, 0, layout))
                    group = (f"mm:dataflow_tiled:{op.id}:"
                             f"{tile.m0}:{tile.n0}")
                    current = acc_views[0]
                    prior_view: int | None = None
                    tasks: list[int] = []
                    for index, kslice in enumerate(ktiles):
                        a_key = (tile.m0, kslice.k0)
                        a_view = tile_views[op.inputs[0]].get(a_key)
                        a_action = tile_actions[op.inputs[0]].get(a_key)
                        w_key = (b_value.id, tile.n0, kslice.k0, tile.cols,
                                 kslice.extent, b_layout)
                        w_view = weight_views.get(w_key)
                        w_action = weight_actions.get(w_key)
                        if a_view is None or w_view is None or \
                                a_action is None or w_action is None:
                            reject(RK_E_SHAPE,
                                   "matmul tile operand has no device producer",
                                   stage="dag", op=op.opcode,
                                   actual={"a_key": a_key,
                                           "w_key": list(w_key)})
                        deps = unique((a_action, w_action,
                                       tasks[-1] if tasks else None))
                        if index == 0:
                            action = self.add_action(
                                "npu_task", template_id="co.mm.f16.v1",
                                inputs=(a_view, w_view),
                                outputs=(current,), deps=deps,
                                origin=(f"lowering:dataflow_tiled:matmul:"
                                        f"{op.id}:{kslice.k0}"),
                                op_ids=(op.id,),
                                attrs={"submit_group": group,
                                       "k0": kslice.k0,
                                       "kt": kslice.extent,
                                       "source_k_loop":
                                           op.attrs.get("source_k_loop"),
                                       "source_mn_tiling":
                                           op.attrs.get("source_mn_tiling")})
                        else:
                            other = acc_views[index & 1]
                            action = self.add_action(
                                "npu_task", template_id="co.mm_acc.f16.v1",
                                inputs=(a_view, w_view, prior_view),
                                outputs=(other,), deps=deps,
                                origin=(f"lowering:dataflow_tiled:matmul_acc:"
                                        f"{op.id}:{kslice.k0}"),
                                op_ids=(op.id,),
                                attrs={"submit_group": group,
                                       "k0": kslice.k0,
                                       "kt": kslice.extent,
                                       "source_k_loop":
                                           op.attrs.get("source_k_loop"),
                                       "source_mn_tiling":
                                           op.attrs.get("source_mn_tiling")})
                            current = other
                        tasks.append(action)
                        prior_view = current
                    tile_views[out_value.id][key_out] = current
                    tile_actions[out_value.id][key_out] = tasks[-1]
                    self.buffers[self.views[current].buffer_id].role = (
                        "output" if out_value.id in self.module.outputs
                        else "workspace")
                continue

            if op.id in self._skip_fused_relu:
                continue
            if op.opcode in {"relu", "silu", "gelu", "sigmoid", "hardsigmoid"}:
                in_id = op.inputs[0]
                in_value = self.module.value(in_id)
                out_id = op.outputs[0]
                for tile in value_tiles[out_id]:
                    key = (tile.m0, tile.n0)
                    in_view = tile_views[in_id].get(key)
                    in_action = tile_actions[in_id].get(key)
                    if in_view is None or in_action is None:
                        reject(RK_E_SHAPE,
                               "activation tile input has no device producer",
                               stage="dag", op=op.opcode)
                    cr, cc, _ = cube_layout(tile.rows, tile.cols)
                    group = (f"act:dataflow_tiled:{op.id}:"
                             f"{tile.m0}:{tile.n0}")
                    out_view = self._emit_activation(
                        activation=op.opcode, rows=tile.rows, cols=tile.cols,
                        cube_rows=cr, cube_cols=cc, input_view=in_view,
                        input_buffer=self.views[in_view].buffer_id,
                        deps=(in_action,), group=group, op_ids=(op.id,))
                    last = self._last_task_of_group(group)
                    tile_views[out_id][key] = out_view
                    tile_actions[out_id][key] = last
                    self.buffers[self.views[out_view].buffer_id].role = (
                        "output" if out_id in self.module.outputs
                        else "workspace")
                continue

            if op.opcode == "add" and op.id in self._fused_add_relu:
                add_out_id = op.outputs[0]
                relu_id = self._fused_add_relu[op.id]
                relu_out_id = self.module.op(relu_id).outputs[0]
                left_id, right_id = op.inputs
                for tile in value_tiles[relu_out_id]:
                    key = (tile.m0, tile.n0)
                    operand_views: list[int] = []
                    operand_actions: list[int] = []
                    for value_id in (left_id, right_id):
                        value = self.module.value(value_id)
                        if value.role == "constant" and value.type.rank < 2:
                            bkey = (value_id, tile.m0, tile.n0)
                            view = broadcast_views.get(bkey)
                            action = broadcast_actions.get(bkey)
                        else:
                            view = tile_views[value_id].get(key)
                            action = tile_actions[value_id].get(key)
                        if view is None or action is None:
                            reject(RK_E_SHAPE,
                                   "fused add operand has no device producer",
                                   stage="dag", op=op.opcode)
                        operand_views.append(view)
                        operand_actions.append(action)
                    cr, cc, _ = cube_layout(tile.rows, tile.cols)
                    group = (f"ew:dataflow_tiled:{op.id}:"
                             f"{tile.m0}:{tile.n0}")
                    out_view = self._emit_binary_views(
                        opcode="add", rows=tile.rows, cols=tile.cols,
                        cube_rows=cr, cube_cols=cc,
                        left_view=operand_views[0],
                        right_view=operand_views[1],
                        deps=unique(operand_actions), group=group, relu=True,
                        role=("output" if relu_out_id in self.module.outputs
                              else "workspace"),
                        reuse=relu_out_id not in self.module.outputs,
                        op_ids=(op.id, relu_id))
                    last = self._last_task_of_group(group)
                    tile_views[add_out_id][key] = out_view
                    tile_actions[add_out_id][key] = last
                    tile_views[relu_out_id][key] = out_view
                    tile_actions[relu_out_id][key] = last
                continue

            if op.opcode in {"add", "mul"}:
                out_id = op.outputs[0]
                left_id, right_id = op.inputs
                for tile in value_tiles[out_id]:
                    key = (tile.m0, tile.n0)
                    operand_views: list[int] = []
                    operand_actions: list[int] = []
                    for value_id in (left_id, right_id):
                        value = self.module.value(value_id)
                        if value.role == "constant" and value.type.rank < 2:
                            bkey = (value_id, tile.m0, tile.n0)
                            view = broadcast_views.get(bkey)
                            action = broadcast_actions.get(bkey)
                        else:
                            view = tile_views[value_id].get(key)
                            action = tile_actions[value_id].get(key)
                        if view is None or action is None:
                            reject(RK_E_SHAPE,
                                   "binary tile operand has no device producer",
                                   stage="dag", op=op.opcode)
                        operand_views.append(view)
                        operand_actions.append(action)
                    cr, cc, _ = cube_layout(tile.rows, tile.cols)
                    group = (f"ew:dataflow_tiled:{op.id}:"
                             f"{tile.m0}:{tile.n0}")
                    out_view = self._emit_binary_views(
                        opcode=op.opcode, rows=tile.rows, cols=tile.cols,
                        cube_rows=cr, cube_cols=cc,
                        left_view=operand_views[0],
                        right_view=operand_views[1],
                        deps=unique(operand_actions), group=group,
                        op_ids=(op.id,))
                    last = self._last_task_of_group(group)
                    tile_views[out_id][key] = out_view
                    tile_actions[out_id][key] = last
                    self.buffers[self.views[out_view].buffer_id].role = (
                        "output" if out_id in self.module.outputs
                        else "workspace")
                continue

            reject(RK_E_UNSUPPORTED,
                   f"no tiled device lowering for node {op.opcode!r}",
                   stage="dag", op=op.opcode, source=op.source)

        # 6. Export each final value tile.  Downloads are Host actions and must
        # follow every NPU task; the runtime assembles the publish rectangles.
        for output_id in self.module.outputs:
            if output_id not in value_tiles:
                reject(RK_E_SHAPE, "tiled dataflow output must be rank 2",
                       stage="dag", actual=output_id)
            binding_id = self.binding_by_value[output_id]
            for tile in value_tiles[output_id]:
                key = (tile.m0, tile.n0)
                view = tile_views[output_id].get(key)
                producer = tile_actions[output_id].get(key)
                if view is None or producer is None:
                    reject(RK_E_SHAPE, "module output tile has no producer",
                           stage="dag", actual={"value": output_id,
                                                "m0": tile.m0,
                                                "n0": tile.n0})
                self.buffers[self.views[view].buffer_id].role = "output"
                download = self.add_action(
                    "download", inputs=(view,), deps=(producer,),
                    origin="host:download:dataflow_tiled",
                    attrs={"binding_id": binding_id, "submit_group": None})
                self.buffers[self.views[view].buffer_id].last = max(
                    self.buffers[self.views[view].buffer_id].last, download)
                self.publish.append(Publish(
                    self.bindings[binding_id].name, binding_id, view,
                    dst_row=tile.m0, dst_col=tile.n0,
                    rows=tile.rows, cols=tile.cols))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)


    def build_scheduled_linear(self, *, instances=None) -> DraftPlan:
        """A/B/C/D task order and submit grouping for one FP16 matmul."""
        if len(self.module.ops) != 1 or self.module.ops[0].opcode != "matmul":
            reject(RK_E_UNSUPPORTED,
                   "linear schedule policy requires exactly one matmul op",
                   stage="schedule")
        op = self.module.ops[0]
        a = self.module.value(op.inputs[0])
        w = self.module.value(op.inputs[1])
        y = self.module.value(op.outputs[0])
        m, k = a.type.shape
        n = y.type.shape[1]
        if w.type.shape != (k, n) or op.attrs.get("b_layout", "kn") != "kn":
            reject(RK_E_UNSUPPORTED,
                   "linear schedule policy supports only B=[K,N] kn layout",
                   stage="schedule", actual=list(w.type.shape))
        policy = self.options.linear_task_order
        if policy not in {"baseline", "scheduled_baseline", "merged",
                          "interleaved", "interleaved_reuse",
                          "weight_stationary", "weight_stationary_reuse",
                          "grouped_linear"}:
            reject(RK_E_UNSUPPORTED, f"unknown linear task policy {policy!r}",
                   stage="schedule")
        # These policies choose task order and submit grouping; they are not a
        # shape feature.  What they actually require is checked where the
        # geometry is decided:
        #   * one matmul in B=[K,N] kn form -- checked above;
        #   * a grouped policy needs a real K chain, because it puts every
        #     tile's slice into one multi_k_acc_v1 submit and that recipe is
        #     admitted only when a co.mm_acc task exists (plan.py:
        #     _recipe_matches); a single 256-wide slice has nothing to
        #     accumulate;
        #   * every task of the chosen policy must fit one submit (64), checked
        #     below against the tiles this plan will really emit;
        #   * the primitive M/N tile stays within 1..TILE_LIMIT, either from the
        #     declaration (mn_schedule_from_attrs) or from the defaults below.
        # The previous code additionally pinned M to {1, 256} and K, N to the
        # SmolLM lm_head shape.  That pin recorded where the first board
        # evidence came from, not a property of the lowering; the conditions it
        # stood in for are the ones listed above.
        if policy in {"merged", "interleaved", "interleaved_reuse",
                      "weight_stationary", "weight_stationary_reuse",
                      "grouped_linear"} and \
                k <= K_TILE:
            reject(RK_E_UNSUPPORTED,
                   "this linear task policy groups a K-accumulation chain, so "
                   "it needs more than one 256-wide K slice; a single K slice "
                   "is a plain matmul and is planned by "
                   "linear_task_order=baseline",
                   stage="schedule", actual={"k": k, "policy": policy},
                   required={"k_greater_than": K_TILE})
        if instances is not None:
            # The converted program states the schedule: it is not read from
            # synthetic op attributes.
            schedule = MNTileSchedule(tile_m=instances["tile_m"],
                                      tile_n=instances["tile_n"],
                                      order=instances["order"])
            declared_schedule = schedule.to_dict()
        else:
            declared_schedule = op.attrs.get("source_schedule")
            schedule = mn_schedule_from_attrs(op.attrs, where="matmul")
            if declared_schedule is None:
                tile_m = 1 if m == 1 else 256
                tile_n = TILE_LIMIT
                order = "m_major"
            else:
                tile_m, tile_n, order = (schedule.tile_m, schedule.tile_n,
                                         schedule.order)
        if instances is None:
            tiles = output_tiles(m, n, tile_m=tile_m, tile_n=tile_n, order=order)
            ktiles = k_slices(k)
        else:
            # The instances come from the converted program (one per tile and K
            # slice, in program order).  The module only supplies binding
            # identity here, so the geometry is checked for agreement instead of
            # being re-derived.
            if a.type.shape != (instances["m"], instances["k"]) or \
                    y.type.shape != (instances["m"], instances["n"]):
                reject(RK_E_SHAPE,
                       "the plan bindings disagree with the instantiated geometry",
                       stage="schedule",
                       actual={"a": list(a.type.shape), "y": list(y.type.shape),
                               "instances": [instances["m"], instances["k"],
                                             instances["n"]]})
            tile_m = instances["tile_m"]
            tile_n = instances["tile_n"]
            order = instances["order"]
            tiles = tuple(instances["tiles"])
            ktiles = tuple(instances["ktiles"])
            operands = instances.get("task_operands")
            if operands and (operands[0] != a.name or operands[1] != w.name
                             or operands[2] != y.name):
                reject(RK_E_SCHEMA,
                       "the plan bindings do not match the operands the "
                       "converted tasks read and write",
                       stage="schedule",
                       actual={"tasks": list(operands),
                               "bindings": [a.name, w.name, y.name]})
            if not tiles or not ktiles:
                reject(RK_E_SHAPE,
                       "the instantiated program produced no task geometry",
                       stage="schedule")

        n_m_tiles = len({tile.m0 for tile in tiles})
        ws_per_n = len(ktiles) * n_m_tiles
        # Weight-stationary runs are grouped by N tile, and whole N tiles are
        # packed into one submit as long as the group stays inside the 64-task
        # bound.  The chain of an output tile never leaves its group.
        ws_n_per_submit = max(1, 64 // ws_per_n) if ws_per_n <= 64 else 0
        if self.options.linear_group_n_tiles:
            requested = self.options.linear_group_n_tiles
            if requested * ws_per_n > 64:
                reject(RK_E_SHAPE,
                       "linear_group_n_tiles would exceed the 64-task submit "
                       "bound",
                       stage="schedule",
                       actual={"n_tiles": requested,
                               "tasks_per_n_tile": ws_per_n},
                       required={"max_tasks_per_submit": 64})
            ws_n_per_submit = requested

        def _per_submit(name: str) -> int:
            if name in {"weight_stationary", "weight_stationary_reuse",
                        "grouped_linear"}:
                return (ws_per_n * ws_n_per_submit) if ws_n_per_submit else ws_per_n
            return (len(ktiles) if name in {"baseline", "scheduled_baseline"}
                    else len(tiles) * len(ktiles))

        if _per_submit(policy) > 64:
            fits = [name for name in ("scheduled_baseline", "merged",
                                      "interleaved", "interleaved_reuse")
                    if _per_submit(name) <= 64]
            reject(RK_E_SHAPE,
                   "declared M/N tiling and this linear task policy need more "
                   "than one submit's 64-task bound",
                   stage="schedule",
                   actual={"policy": policy, "tile_m": tile_m,
                           "tile_n": tile_n, "output_tiles": len(tiles),
                           "k_slices": len(ktiles),
                           "tasks_per_submit": _per_submit(policy)},
                   required={"max_tasks_per_submit": 64,
                             "policies_that_fit": fits})
        a_binding = self.binding_by_value[a.id]
        w_binding = self.binding_by_value[w.id]
        y_binding = self.binding_by_value[y.id]
        input_cache: dict[tuple, tuple[int, int]] = {}
        weight_cache: dict[tuple, tuple[int, int]] = {}

        def input_upload(tile, ks):
            key = (a.id, tile.m0, ks.k0, tile.rows, ks.extent)
            if key not in input_cache:
                pr, pc, pbytes = input_layout(tile.rows, ks.extent)
                layout = TensorLayout("float16", CUBE_LAYOUT, tile.rows,
                                      ks.extent, pr, pc, 8, pbytes)
                b = self.add_buffer(role="input_slice", capacity=pbytes,
                                    layout_id=CUBE_LAYOUT, reuse=False,
                                    padding_state="zero")
                v = self.add_view(b, 0, layout)
                act = self.add_action(
                    "upload", outputs=(v,), origin="host_pack:linear:input",
                    attrs={"source_binding": a_binding,
                           "src_row_offset": tile.m0,
                           "src_col_offset": ks.k0,
                           "src_row_stride": k,
                           "transpose_source": False, "cache": "input"})
                input_cache[key] = (v, act)
            return input_cache[key]

        def weight_upload(tile, ks):
            key = (w.id, tile.n0, ks.k0, tile.cols, ks.extent)
            if key not in weight_cache:
                pr, pc, pbytes = weight_layout(tile.cols, ks.extent)
                layout = TensorLayout("float16", WEIGHT_LAYOUT, tile.cols,
                                      ks.extent, pr, pc, 32, pbytes)
                b = self.add_buffer(role="weight", capacity=pbytes,
                                    layout_id=WEIGHT_LAYOUT, reuse=False,
                                    padding_state="zero")
                v = self.add_view(b, 0, layout)
                act = self.add_action(
                    "upload", outputs=(v,), origin="host_pack:linear:weight",
                    attrs={"source_binding": w_binding,
                           "src_row_offset": ks.k0,
                           "src_col_offset": tile.n0,
                           "src_row_stride": n,
                           "transpose_source": True, "cache": "constant"})
                weight_cache[key] = (v, act)
            return weight_cache[key]

        a_views = {(t.m0, t.n0): [input_upload(t, ks)[0] for ks in ktiles]
                   for t in tiles}
        w_views = {(t.m0, t.n0): [weight_upload(t, ks)[0] for ks in ktiles]
                   for t in tiles}
        a_uploads = {(t.m0, t.n0): [input_upload(t, ks)[1] for ks in ktiles]
                     for t in tiles}
        w_uploads = {(t.m0, t.n0): [weight_upload(t, ks)[1] for ks in ktiles]
                     for t in tiles}
        accs: dict[tuple[int, int], list[int]] = {}
        final_view: dict[tuple[int, int], int] = {}
        for tile in tiles:
            cr, cc, cbytes = cube_layout(tile.rows, tile.cols)
            pair = []
            for _ in range(2 if len(ktiles) > 1 else 1):
                layout = TensorLayout("float16", CUBE_LAYOUT, tile.rows,
                                      tile.cols, cr, cc, 8, cbytes)
                b = self.add_buffer(role="accumulator", capacity=cbytes,
                                    layout_id=CUBE_LAYOUT,
                                    padding_state="finite")
                pair.append(self.add_view(b, 0, layout))
            accs[(tile.m0, tile.n0)] = pair

        group = (f"mm:{tiles[0].m0}:{tiles[0].n0}"
                 if policy in {"baseline", "scheduled_baseline"}
                 else "mk:linear")
        def emit(tile, kidx, weight_reuse=False, group=None):
            ks = ktiles[kidx]
            key = (tile.m0, tile.n0)
            prior = None if kidx == 0 else accs[key][(kidx - 1) & 1]
            out = accs[key][kidx & 1]
            deps = tuple(dict.fromkeys((a_uploads[key][kidx],
                                        w_uploads[key][kidx])))
            if prior is not None:
                prev_id = self._linear_prev[key][kidx - 1]
                deps += (prev_id,)
            reuse = (policy == "interleaved_reuse" and
                     tile is not tiles[0])
            submit = (f"mm:{tile.m0}:{tile.n0}"
                      if policy in {"baseline", "scheduled_baseline"}
                      else ("mk:linear" if group is None else group))
            # Explicit recipe requirement: the submit builder reads this field
            # instead of guessing from the human readable group name.
            recipe = ("linear_k_acc_v1"
                      if policy in {"baseline", "scheduled_baseline"}
                      else "multi_k_acc_v1")
            attrs = {"submit_group": submit, "submit_recipe": recipe,
                     "k0": ks.k0, "k_index": kidx, "kt": ks.extent,
                     "k_count": len(ktiles),
                     "data_reuse": bool(reuse),
                     "weight_reuse": bool(weight_reuse),
                     "linear_task_policy": policy}
            if declared_schedule is not None:
                attrs["source_schedule"] = declared_schedule
                attrs["source_k_rounding"] = op.attrs.get("source_k_rounding")
                attrs["source_mn_tiling"] = op.attrs.get("source_mn_tiling")
                attrs["applied_schedule"] = schedule.to_dict()
            if instances is not None and instances.get("hints"):
                # Source requests this path did not apply, recorded so they
                # survive save/load instead of living only in the returned facts.
                attrs["source_hints"] = dict(instances["hints"])
            if prior is None:
                tid = self.add_action(
                    "npu_task", template_id="co.mm.f16.v1",
                    inputs=(a_views[key][kidx], w_views[key][kidx]),
                    outputs=(out,), deps=deps,
                    origin=f"lowering:linear:{policy}:k{kidx}:n{tile.n0}",
                    op_ids=(op.id,),
                    attrs=attrs)
            else:
                tid = self.add_action(
                    "npu_task", template_id="co.mm_acc.f16.v1",
                    inputs=(a_views[key][kidx], w_views[key][kidx], prior),
                    outputs=(out,), deps=deps,
                    origin=f"lowering:linear:{policy}:k{kidx}:n{tile.n0}",
                    op_ids=(op.id,),
                    attrs=attrs)
            self._linear_prev[key][kidx] = tid
        self._linear_prev = {(t.m0, t.n0): {} for t in tiles}
        if policy in {"baseline", "scheduled_baseline", "merged"}:
            for tile in tiles:
                group = (f"mm:{tile.m0}:{tile.n0}"
                         if policy in {"baseline", "scheduled_baseline"}
                         else "mk:linear")
                for kidx in range(len(ktiles)):
                    emit(tile, kidx)
        elif policy == "grouped_linear":
            # Same group partition as the weight-stationary policies, but the
            # historical intra-group order (tile major, K innermost).  This is
            # the control that isolates the intra-group task order.
            ordered = sorted(tiles, key=lambda t: (t.n0, t.m0))
            n_values = sorted({tile.n0 for tile in ordered})
            chunk = ws_n_per_submit or 1
            for start in range(0, len(n_values), chunk):
                block = set(n_values[start:start + chunk])
                group = f"mk:grp:{start // chunk}"
                for tile in ordered:
                    if tile.n0 not in block:
                        continue
                    for kidx in range(len(ktiles)):
                        emit(tile, kidx, group=group)
        elif policy in {"weight_stationary", "weight_stationary_reuse"}:
            # Fixed (N tile, K slice), iterating the M tiles: the weight view of
            # the previous task is then the same operand at the same K index,
            # which is exactly the condition the verifier attaches to the CBUF
            # weight-reuse bit ("reuse the previous task's weight on this
            # core").  Whole N tiles are packed into one submit up to the
            # 64-task bound, so an output tile's K chain never crosses a submit
            # and the first task of every submit starts with reuse off.
            ordered = sorted(tiles, key=lambda t: (t.n0, t.m0))
            n_values = sorted({tile.n0 for tile in ordered})
            chunk = ws_n_per_submit or 1
            for start in range(0, len(n_values), chunk):
                block = set(n_values[start:start + chunk])
                # The mk: prefix makes the submit builder treat the group as a
                # multi-K accumulation run (its matcher accepts any mm/mm_acc
                # sequence), which is what this order is.
                group = f"mk:ws:{start // chunk}"
                for kidx in range(len(ktiles)):
                    previous = None
                    for tile in ordered:
                        if tile.n0 not in block:
                            continue
                        reuse = (policy == "weight_stationary_reuse" and
                                 previous is not None and
                                 previous.n0 == tile.n0)
                        emit(tile, kidx, weight_reuse=reuse, group=group)
                        previous = tile
        else:
            for kidx in range(len(ktiles)):
                for tile in tiles:
                    emit(tile, kidx)
        for tile in tiles:
            key = (tile.m0, tile.n0)
            final_view[key] = accs[key][(len(ktiles) - 1) & 1]
            binding = y_binding
            download = self.add_action(
                "download", inputs=(final_view[key],),
                deps=(self._linear_prev[key][len(ktiles) - 1],),
                origin="host:download:linear",
                attrs={"binding_id": binding, "submit_group": None})
            self.buffers[self.views[final_view[key]].buffer_id].role = "output"
            self.publish.append(Publish(
                self.bindings[binding].name, binding, final_view[key],
                dst_row=tile.m0, dst_col=tile.n0, rows=tile.rows,
                cols=tile.cols))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build_scheduled_two_stage(self) -> DraftPlan:
        """Lower X->W1->Z->W2->Y with a schedule-aware intermediate map.

        Only the restricted two-stage contract is admitted.  Stage 0 must use a
        256-column logical tile so every consumer K slice maps to one producer
        tile.  Stage 1 may use a smaller M tile when it divides stage 0's M tile
        and never crosses a producer row tile.  Z never leaves the device.
        """
        if len(self.module.ops) != 2 or \
                any(op.opcode != "matmul" for op in self.module.ops):
            reject(RK_E_UNSUPPORTED,
                   "scheduled staged lowering requires exactly two matmuls",
                   stage="schedule", actual=[op.opcode for op in self.module.ops])
        op0, op1 = self.module.ops
        x = self.module.value(op0.inputs[0])
        w1 = self.module.value(op0.inputs[1])
        z = self.module.value(op0.outputs[0])
        w2 = self.module.value(op1.inputs[1])
        y = self.module.value(op1.outputs[0])
        if op1.inputs[0] != z.id or z.role != "temporary" or \
                w2.id in {x.id, w1.id, z.id} or y.id in {x.id, w1.id, z.id}:
            reject(RK_E_UNSUPPORTED,
                   "scheduled staged dataflow must be X->W1->Z->W2->Y",
                   stage="schedule")
        m, k = x.type.shape
        h = z.type.shape[1]
        n = y.type.shape[1]
        if z.type.shape != (m, h) or w1.type.shape != (k, h) or \
                w2.type.shape != (h, n) or y.type.shape != (m, n):
            reject(RK_E_SHAPE,
                   "scheduled staged operand shapes do not compose",
                   stage="schedule",
                   actual={"X": list(x.type.shape), "W1": list(w1.type.shape),
                           "Z": list(z.type.shape), "W2": list(w2.type.shape),
                           "Y": list(y.type.shape)})
        s0 = mn_schedule_from_attrs(op0.attrs, where="stage0")
        s1 = mn_schedule_from_attrs(op1.attrs, where="stage1")
        if s0.tile_n != 256:
            reject(RK_E_UNSUPPORTED,
                   "stage0 tile_n must be 256 so each consumer K slice has one "
                   "producer tile", stage="schedule", actual=s0.tile_n,
                   required=256)
        if s1.tile_m != s0.tile_m:
            reject(RK_E_UNSUPPORTED,
                   "the applied stage0/stage1 M tile must match in this "
                   "milestone; a different source M tile can be retiled to "
                   "the consumer-safe applied tile, but the current CUBE view "
                   "cannot expose a row subview with a different physical "
                   "row geometry without an explicit repack",
                   stage="schedule",
                   actual={"stage0_tile_m": s0.tile_m,
                           "stage1_tile_m": s1.tile_m})
        k_tail = h - ((h + 255) // 256 - 1) * 256
        if k_tail % 32:
            reject(RK_E_UNSUPPORTED,
                   "the intermediate H tail must be 32-element aligned; the "
                   "current CUBE view cannot expose a narrower physical tail",
                   stage="schedule", actual={"H": h, "tail": k_tail})

        x_upload_cache: dict[tuple, tuple[int, int]] = {}
        w_upload_cache: dict[tuple, tuple[int, int]] = {}

        def add_input_upload(value_id: int, binding_id: int, m0: int,
                             k0: int, rows: int, kt: int,
                             stride: int) -> tuple[int, int]:
            pr, pc, pbytes = input_layout(rows, kt)
            key = (value_id, m0, k0, rows, kt)
            cached = x_upload_cache.get(key)
            if cached is not None:
                return cached
            layout = TensorLayout("float16", CUBE_LAYOUT, rows, kt,
                                  pr, pc, 8, pbytes)
            buffer_id = self.add_buffer(role="input_slice", capacity=pbytes,
                                        layout_id=CUBE_LAYOUT, reuse=False,
                                        padding_state="zero")
            view = self.add_view(buffer_id, 0, layout)
            action = self.add_action(
                "upload", outputs=(view,), origin="host_pack:staged:input",
                attrs={"source_binding": binding_id,
                       "src_row_offset": m0, "src_col_offset": k0,
                       "src_row_stride": stride,
                       "transpose_source": False, "cache": "input"})
            cached = (view, action)
            x_upload_cache[key] = cached
            return cached

        def add_weight_upload(value_id: int, binding_id: int, n0: int,
                              k0: int, cols: int, kt: int,
                              stride: int) -> tuple[int, int]:
            key = (value_id, n0, k0, cols, kt)
            cached = w_upload_cache.get(key)
            if cached is not None:
                return cached
            wr, wc, wbytes = weight_layout(cols, kt)
            layout = TensorLayout("float16", WEIGHT_LAYOUT, cols, kt,
                                  wr, wc, 32, wbytes)
            buffer_id = self.add_buffer(role="weight", capacity=wbytes,
                                        layout_id=WEIGHT_LAYOUT, reuse=False,
                                        padding_state="zero")
            view = self.add_view(buffer_id, 0, layout)
            action = self.add_action(
                "upload", outputs=(view,), origin="host_pack:staged:weight",
                attrs={"source_binding": binding_id,
                       "src_row_offset": k0, "src_col_offset": n0,
                       "src_row_stride": stride,
                       "transpose_source": True, "cache": "constant"})
            cached = (view, action)
            w_upload_cache[key] = cached
            return cached

        def emit_chain(*, op_id: int, source_op: Any, out_value: Any,
                       tile: Any,
                       tile_m: int, tile_n: int, order: str,
                       a_source: str, b_value: Any, b_binding: int,
                       b_stride: int, a_binding: int | None,
                       a_stride: int | None, z_map: dict | None,
                       z_stage: dict | None) -> tuple[int, int]:
            ktiles = k_slices(a_source == "X" and k or h)
            cr, cc, cbytes = cube_layout(tile.rows, tile.cols)
            acc_views: list[int] = []
            for _ in range(1 if len(ktiles) == 1 else 2):
                layout = TensorLayout("float16", CUBE_LAYOUT,
                                      tile.rows, tile.cols,
                                      cr, cc, 8, cbytes)
                buffer_id = self.add_buffer(
                    role="accumulator", capacity=cbytes,
                    layout_id=CUBE_LAYOUT, padding_state="finite")
                acc_views.append(self.add_view(buffer_id, 0, layout))
            group = f"mm:s{op_id}:{op_id}:{tile.m0}:{tile.n0}"
            current = acc_views[0]
            prior_view: int | None = None
            tasks: list[int] = []
            z_source_meta: dict[str, Any] | None = None
            for index, kslice in enumerate(ktiles):
                if a_source == "X":
                    assert a_binding is not None and a_stride is not None
                    a_view, a_action = add_input_upload(
                        x.id, a_binding, tile.m0, kslice.k0,
                        tile.rows, kslice.extent, a_stride)
                else:
                    assert z_map is not None and z_stage is not None
                    pm0 = (tile.m0 // z_stage["tile_m"]) * z_stage["tile_m"]
                    pn0 = kslice.k0
                    z_view = z_map.get((pm0, pn0))
                    if z_view is None:
                        reject(RK_E_SHAPE,
                               "consumer input region has no producer tile",
                               stage="schedule",
                               actual={"stage1_m0": tile.m0,
                                       "k0": kslice.k0,
                                       "producer_m0": pm0})
                    parent = self.views[z_view]
                    parent_t = parent.tensor
                    row_off = tile.m0 - pm0
                    if row_off < 0 or row_off + tile.rows > parent_t.rows or \
                            kslice.extent != parent_t.cols:
                        reject(RK_E_UNSUPPORTED,
                               "consumer region crosses or misaligns a producer "
                               "Z tile", stage="schedule",
                               actual={"producer": [pm0, pn0,
                                                    parent_t.rows,
                                                    parent_t.cols],
                                       "consumer": [tile.m0, kslice.k0,
                                                    tile.rows,
                                                    kslice.extent]})
                    pr = round_up(max(tile.rows, 12), 4)
                    if row_off + pr > parent_t.physical_rows:
                        reject(RK_E_UNSUPPORTED,
                               "consumer physical rows exceed the producer Z "
                               "tile; a narrower consumer M tile would need an "
                               "unsupported view", stage="schedule",
                               actual={"row_off": row_off,
                                       "consumer_physical_rows": pr,
                                       "producer_physical_rows":
                                           parent_t.physical_rows})
                    sub_layout = TensorLayout(
                        "float16", CUBE_LAYOUT, tile.rows, kslice.extent,
                        pr, parent_t.physical_cols, 8,
                        pr * parent_t.physical_cols * 2)
                    offset = parent.offset_bytes + \
                        row_off * parent_t.physical_cols * 2
                    a_view = self.add_view(parent.buffer_id, offset, sub_layout)
                    a_action = z_map_action[(pm0, pn0)]
                    z_source_meta = {
                        "value": z.name,
                        "producer_view": z_view,
                        "producer_action": a_action,
                        "producer_tile": [pm0, pn0, parent_t.rows,
                                          parent_t.cols],
                        "consumer_region": [tile.m0, kslice.k0,
                                            tile.rows, kslice.extent],
                        "row_offset": row_off,
                        "logical_k": kslice.extent,
                        "physical_cols": parent_t.physical_cols,
                        "padding": "consumer reads valid columns only; "
                                   "A/B zero padding is neutralised by the "
                                   "zero-padded weight tail",
                    }
                w_view, w_action = add_weight_upload(
                    b_value.id, b_binding, tile.n0, kslice.k0,
                    tile.cols, kslice.extent, b_stride)
                deps = tuple(dict.fromkeys(
                    (a_action, w_action,
                     tasks[-1] if tasks else None)))
                deps = tuple(dep for dep in deps if dep is not None)
                if index == 0:
                    action = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(a_view, w_view), outputs=(current,),
                        deps=deps,
                        origin=f"lowering:staged:stage{op_id}:mm:{kslice.k0}",
                        op_ids=(source_op.id,),
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "k_physical_extent":
                                   ((kslice.extent + 31) // 32) * 32,
                               "k_padding":
                                   (((kslice.extent + 31) // 32) * 32
                                    - kslice.extent),
                               "source_schedule": op0.attrs.get(
                                   "source_schedule") if op_id == 0 else
                                   op1.attrs.get("source_schedule"),
                               "applied_schedule": (s0.to_dict()
                                                    if op_id == 0
                                                    else s1.to_dict()),
                               "source_stage_index": op_id,
                               "stage_barrier": "all_previous_submits",
                               **({"z_source": z_source_meta}
                                  if z_source_meta is not None else {})})
                else:
                    other = acc_views[index & 1]
                    action = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(a_view, w_view, prior_view),
                        outputs=(other,), deps=deps,
                        origin=f"lowering:staged:stage{op_id}:mm_acc:"
                               f"{kslice.k0}",
                        op_ids=(source_op.id,),
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "k_physical_extent":
                                   ((kslice.extent + 31) // 32) * 32,
                               "k_padding":
                                   (((kslice.extent + 31) // 32) * 32
                                    - kslice.extent),
                               "source_schedule": op0.attrs.get(
                                   "source_schedule") if op_id == 0 else
                                   op1.attrs.get("source_schedule"),
                               "applied_schedule": (s0.to_dict()
                                                    if op_id == 0
                                                    else s1.to_dict()),
                               "source_stage_index": op_id,
                               "stage_barrier": "all_previous_submits",
                               **({"z_source": z_source_meta}
                                  if z_source_meta is not None else {})})
                    current = other
                tasks.append(action)
                prior_view = current
            return current, tasks[-1]

        z_map: dict[tuple[int, int], int] = {}
        z_map_action: dict[tuple[int, int], int] = {}
        w1_binding = self.binding_by_value[w1.id]
        w2_binding = self.binding_by_value[w2.id]
        x_binding = self.binding_by_value[x.id]

        # Every Host upload must precede the first NPU task.  The pass below
        # creates the exact same cache keys that emit_chain will consume.
        for tile in output_tiles(m, h, tile_m=s0.tile_m,
                                 tile_n=s0.tile_n, order=s0.order):
            for kslice in k_slices(k):
                add_input_upload(x.id, x_binding, tile.m0, kslice.k0,
                                 tile.rows, kslice.extent, k)
                add_weight_upload(w1.id, w1_binding, tile.n0, kslice.k0,
                                  tile.cols, kslice.extent, h)
        for tile in output_tiles(m, n, tile_m=s1.tile_m,
                                 tile_n=s1.tile_n, order=s1.order):
            for kslice in k_slices(h):
                add_weight_upload(w2.id, w2_binding, tile.n0, kslice.k0,
                                  tile.cols, kslice.extent, n)

        for tile in output_tiles(m, h, tile_m=s0.tile_m,
                                 tile_n=s0.tile_n, order=s0.order):
            final_view, final_action = emit_chain(
                op_id=0, source_op=op0, out_value=z, tile=tile,
                tile_m=s0.tile_m,
                tile_n=s0.tile_n, order=s0.order, a_source="X",
                b_value=w1, b_binding=w1_binding, b_stride=h,
                a_binding=x_binding, a_stride=k, z_map=None, z_stage=None)
            z_map[(tile.m0, tile.n0)] = final_view
            z_map_action[(tile.m0, tile.n0)] = final_action
            self.buffers[self.views[final_view].buffer_id].role = "workspace"

        y_tiles: dict[tuple[int, int], int] = {}
        y_actions: dict[tuple[int, int], int] = {}
        for tile in output_tiles(m, n, tile_m=s1.tile_m,
                                 tile_n=s1.tile_n, order=s1.order):
            final_view, final_action = emit_chain(
                op_id=1, source_op=op1, out_value=y, tile=tile,
                tile_m=s1.tile_m,
                tile_n=s1.tile_n, order=s1.order, a_source="Z",
                b_value=w2, b_binding=w2_binding, b_stride=n,
                a_binding=None, a_stride=None, z_map=z_map,
                z_stage={"tile_m": s0.tile_m})
            y_tiles[(tile.m0, tile.n0)] = final_view
            y_actions[(tile.m0, tile.n0)] = final_action
            self.buffers[self.views[final_view].buffer_id].role = "output"

        for tile in output_tiles(m, n, tile_m=s1.tile_m,
                                 tile_n=s1.tile_n, order=s1.order):
            view = y_tiles[(tile.m0, tile.n0)]
            producer = y_actions[(tile.m0, tile.n0)]
            binding_id = self.binding_by_value[y.id]
            download = self.add_action(
                "download", inputs=(view,), deps=(producer,),
                origin="host:download:staged", attrs={
                    "binding_id": binding_id, "submit_group": None})
            self.buffers[self.views[view].buffer_id].last = max(
                self.buffers[self.views[view].buffer_id].last, download)
            self.publish.append(Publish(
                self.bindings[binding_id].name, binding_id, view,
                dst_row=tile.m0, dst_col=tile.n0,
                rows=tile.rows, cols=tile.cols))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build_scheduled_swiglu_ffn(self) -> DraftPlan:
        """Schedule-aware packed gate/up -> SwiGLU -> down -> residual."""
        ops = self.module.ops
        expected = ["matmul", "view", "view", "sigmoid", "mul", "mul",
                    "matmul", "add"]
        if [op.opcode for op in ops] != expected:
            reject(RK_E_UNSUPPORTED,
                   "scheduled FFN lowerer requires the closed eight-op graph",
                   stage="schedule", actual=[op.opcode for op in ops])
        op0, vg, vu, sig, silu, act, op4, res = ops
        x = self.module.value(op0.inputs[0])
        wgu = self.module.value(op0.inputs[1])
        zgu = self.module.value(op0.outputs[0])
        g = self.module.value(vg.outputs[0])
        u = self.module.value(vu.outputs[0])
        s = self.module.value(sig.outputs[0])
        silu_v = self.module.value(silu.outputs[0])
        a = self.module.value(act.outputs[0])
        wd = self.module.value(op4.inputs[1])
        p = self.module.value(op4.outputs[0])
        y = self.module.value(res.outputs[0])
        r = self.module.value(res.inputs[1])
        m, d = x.type.shape
        if r.id == x.id:
            reject(RK_E_UNSUPPORTED,
                   "FFN H and R must be distinct input values; alias is not "
                   "admitted", stage="schedule")
        if r.type.shape != (m, d) or r.role != "input":
            reject(RK_E_SHAPE,
                   "FFN residual R must be an independent input with H shape",
                   stage="schedule", actual={"shape": list(r.type.shape),
                                              "role": r.role})
        twoh = zgu.type.shape[1]
        h = twoh // 2
        if twoh % 2 or a.type.shape != (m, h) or s.type.shape != (m, h) or \
                silu_v.type.shape != (m, h) or u.type.shape != (m, h) or \
                g.type.shape != (m, h) or wgu.type.shape != (d, twoh) or \
                wd.type.shape != (h, d) or p.type.shape != (m, d) or \
                y.type.shape != (m, d):
            reject(RK_E_SHAPE, "scheduled FFN operand shapes do not compose",
                   stage="schedule")
        if vg.attrs != {"row_offset": 0, "col_offset": 0,
                        "rows": m, "cols": h} or \
                vu.attrs != {"row_offset": 0, "col_offset": h,
                             "rows": m, "cols": h}:
            reject(RK_E_UNSUPPORTED,
                   "FFN view ops must be the exact gate/up halves",
                   stage="schedule", actual={"gate": dict(vg.attrs),
                                              "up": dict(vu.attrs)})
        if sig.inputs != (g.id,) or set(silu.inputs) != {g.id, s.id} or \
                set(act.inputs) != {silu_v.id, u.id} or \
                res.inputs != (p.id, r.id):
            reject(RK_E_UNSUPPORTED,
                   "scheduled FFN graph is not packed-gate/up -> SwiGLU -> down",
                   stage="schedule")
        s0 = mn_schedule_from_attrs(op0.attrs, where="stage0")
        s4 = mn_schedule_from_attrs(op4.attrs, where="stage4")
        if s0.tile_n != 256:
            reject(RK_E_UNSUPPORTED,
                   "FFN stage0 tile_n must be 256", stage="schedule",
                   actual=s0.tile_n)
        if h % 256 or twoh % 256 or d % s4.tile_n:
            reject(RK_E_UNSUPPORTED,
                   "FFN intermediate width must be 256-aligned and down output "
                   "must be tile-aligned", stage="schedule",
                   actual={"h": h, "twoh": twoh, "d": d,
                           "down_tile_n": s4.tile_n})
        def ew_schedule(op: Any, index: int) -> dict:
            raw = op.attrs.get("source_elementwise_schedule")
            if not isinstance(raw, dict) or raw.get("kind") != "elementwise_tiles_v1":
                reject(RK_E_UNSUPPORTED,
                       f"FFN stage{index} lacks its elementwise schedule",
                       stage="schedule", actual=raw)
            tm, tn = raw.get("tile_m"), raw.get("tile_n")
            order = raw.get("order")
            if not isinstance(tm, int) or not isinstance(tn, int) or \
                    not 1 <= tm <= 256 or not 1 <= tn <= 256 or \
                    order not in MN_TILE_ORDERS:
                reject(RK_E_UNSUPPORTED,
                       f"FFN stage{index} elementwise schedule is invalid",
                       stage="schedule", actual=raw)
            if tm != s0.tile_m:
                reject(RK_E_UNSUPPORTED,
                       "all FFN stages must use the same applied M tile",
                       stage="schedule", actual={"stage": index, "tile_m": tm,
                                                 "stage0_tile_m": s0.tile_m})
            return raw
        e1 = ew_schedule(sig, 1)
        e2 = ew_schedule(silu, 2)
        e3 = ew_schedule(act, 3)
        e5 = ew_schedule(res, 5)
        if e1["tile_n"] != 256 or e2["tile_n"] != 256 or e3["tile_n"] != 256:
            reject(RK_E_UNSUPPORTED,
                   "SwiGLU stages must use 256-wide elementwise tiles",
                   stage="schedule")
        if s4.tile_m != s0.tile_m or e5["tile_m"] != s4.tile_m or \
                e5["tile_n"] != s4.tile_n:
            reject(RK_E_UNSUPPORTED,
                   "down and residual must share the applied tile geometry",
                   stage="schedule")

        x_binding = self.binding_by_value[x.id]
        r_binding = self.binding_by_value[r.id]
        wgu_binding = self.binding_by_value[wgu.id]
        wd_binding = self.binding_by_value[wd.id]
        input_cache: dict[tuple, tuple[int, int]] = {}
        weight_cache: dict[tuple, tuple[int, int]] = {}

        def cube_view(rows: int, cols: int) -> tuple[int, int, int, int]:
            pr, pc, pbytes = cube_layout(rows, cols)
            layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, pr, pc,
                                  8, pbytes)
            buffer_id = self.add_buffer(role="workspace", capacity=pbytes,
                                        layout_id=CUBE_LAYOUT,
                                        padding_state="finite")
            view_id = self.add_view(buffer_id, 0, layout)
            return buffer_id, view_id, pr, pc

        def input_upload(value_id: int, binding: int, m0: int, c0: int,
                         rows: int, cols: int, stride: int) -> tuple[int, int]:
            key = (value_id, m0, c0, rows, cols, stride)
            cached = input_cache.get(key)
            if cached is not None:
                return cached
            pr, pc, pbytes = input_layout(rows, cols)
            layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, pr, pc,
                                  8, pbytes)
            buffer_id = self.add_buffer(role="input_slice", capacity=pbytes,
                                        layout_id=CUBE_LAYOUT, reuse=False,
                                        padding_state="zero")
            view = self.add_view(buffer_id, 0, layout)
            action = self.add_action(
                "upload", outputs=(view,), origin="host_pack:ffn:input",
                attrs={"source_binding": binding,
                       "src_row_offset": m0, "src_col_offset": c0,
                       "src_row_stride": stride,
                       "transpose_source": False, "cache": "input"})
            input_cache[key] = (view, action)
            return view, action

        def weight_upload(value_id: int, binding: int, n0: int, k0: int,
                          cols: int, kt: int) -> tuple[int, int]:
            key = (value_id, n0, k0, cols, kt)
            cached = weight_cache.get(key)
            if cached is not None:
                return cached
            pr, pc, pbytes = weight_layout(cols, kt)
            layout = TensorLayout("float16", WEIGHT_LAYOUT, cols, kt, pr, pc,
                                  32, pbytes)
            buffer_id = self.add_buffer(role="weight", capacity=pbytes,
                                        layout_id=WEIGHT_LAYOUT, reuse=False,
                                        padding_state="zero")
            view = self.add_view(buffer_id, 0, layout)
            action = self.add_action(
                "upload", outputs=(view,), origin="host_pack:ffn:weight",
                attrs={"source_binding": binding,
                       "src_row_offset": k0, "src_col_offset": n0,
                       "src_row_stride": d if value_id == wd.id else twoh,
                       "transpose_source": True, "cache": "constant"})
            weight_cache[key] = (view, action)
            return view, action

        def matmul_chain(tile: Any, a_views: list[int], w_views: list[int],
                         uploads: list[int], ktiles: list[Any], *,
                         stage: int, group: str, source_op: Any,
                         deps_extra: tuple[int, ...] = (),
                         extra_attrs: dict | None = None) -> tuple[int, int]:
            cr, cc, cbytes = cube_layout(tile.rows, tile.cols)
            acc: list[int] = []
            for _ in range(1 if len(ktiles) == 1 else 2):
                layout = TensorLayout("float16", CUBE_LAYOUT, tile.rows,
                                      tile.cols, cr, cc, 8, cbytes)
                b = self.add_buffer(role="accumulator", capacity=cbytes,
                                    layout_id=CUBE_LAYOUT,
                                    padding_state="finite")
                acc.append(self.add_view(b, 0, layout))
            current = acc[0]
            prior = None
            tasks: list[int] = []
            for index, kslice in enumerate(ktiles):
                deps = tuple(dict.fromkeys(
                    tuple(uploads) + deps_extra +
                    (() if index == 0 else (tasks[-1],))))
                attrs = {"submit_group": group, "k0": kslice.k0,
                         "kt": kslice.extent,
                         "k_physical_extent": ((kslice.extent + 31) // 32) * 32,
                         "k_padding": (((kslice.extent + 31) // 32) * 32
                                       - kslice.extent),
                         "source_stage_index": stage,
                         "stage_barrier": "all_previous_submits",
                         "source_schedule": (op0.attrs.get("source_schedule")
                                             if stage == 0 else
                                             op4.attrs.get("source_schedule")),
                         "applied_schedule": (s0.to_dict() if stage == 0
                                              else s4.to_dict())}
                if extra_attrs:
                    attrs.update(extra_attrs)
                if index == 0:
                    task = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(a_views[index], w_views[index]),
                        outputs=(current,), deps=deps,
                        origin=f"lowering:ffn:stage{stage}:mm:{kslice.k0}",
                        op_ids=(source_op.id,),
                        attrs=attrs)
                else:
                    other = acc[index & 1]
                    task = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(a_views[index], w_views[index], prior),
                        outputs=(other,), deps=deps,
                        origin=f"lowering:ffn:stage{stage}:mm_acc:{kslice.k0}",
                        op_ids=(source_op.id,),
                        attrs=attrs)
                    current = other
                tasks.append(task)
                prior = current
            return current, tasks[-1]

        # ----- Host uploads first ------------------------------------------
        s0_tiles = output_tiles(m, twoh, tile_m=s0.tile_m,
                                tile_n=s0.tile_n, order=s0.order)
        for tile in s0_tiles:
            for kslice in k_slices(d):
                input_upload(x.id, x_binding, tile.m0, kslice.k0,
                             tile.rows, kslice.extent, d)
                weight_upload(wgu.id, wgu_binding, tile.n0, kslice.k0,
                              tile.cols, kslice.extent)
        s4_tiles = output_tiles(m, d, tile_m=s4.tile_m,
                                tile_n=s4.tile_n, order=s4.order)
        for tile in s4_tiles:
            for kslice in k_slices(h):
                weight_upload(wd.id, wd_binding, tile.n0, kslice.k0,
                              tile.cols, kslice.extent)
            input_upload(r.id, r_binding, tile.m0, tile.n0,
                         tile.rows, tile.cols, d)
        _, identity256 = self.identity_for(256)
        identity256_upload = self._last_upload_of(identity256)
        _, identity_out = self.identity_for(s4.tile_n)
        identity_out_upload = self._last_upload_of(identity_out)

        # ----- stage 0: packed gate/up ------------------------------------
        z_tiles: dict[tuple[int, int], int] = {}
        z_tasks: dict[tuple[int, int], int] = {}
        for tile in s0_tiles:
            ktiles = k_slices(d)
            a_views = [input_upload(x.id, x_binding, tile.m0, k.k0,
                                    tile.rows, k.extent, d)[0]
                       for k in ktiles]
            w_views = [weight_upload(wgu.id, wgu_binding, tile.n0, k.k0,
                                     tile.cols, k.extent)[0]
                       for k in ktiles]
            uploads = [input_upload(x.id, x_binding, tile.m0, k.k0,
                                     tile.rows, k.extent, d)[1]
                       for k in ktiles]
            uploads += [weight_upload(wgu.id, wgu_binding, tile.n0, k.k0,
                                      tile.cols, k.extent)[1]
                        for k in ktiles]
            view, task = matmul_chain(
                tile, a_views, w_views, uploads, ktiles, stage=0,
                group=f"mm:s0:{tile.m0}:{tile.n0}", source_op=op0)
            z_tiles[(tile.m0, tile.n0)] = view
            z_tasks[(tile.m0, tile.n0)] = task
            self.buffers[self.views[view].buffer_id].role = "workspace"

        # ----- stages 1-3: sigmoid, silu, gated mul ------------------------
        sig_tiles: dict[tuple[int, int], int] = {}
        sig_tasks: dict[tuple[int, int], int] = {}
        act_tiles: dict[tuple[int, int], int] = {}
        act_tasks: dict[tuple[int, int], int] = {}
        h_tiles = output_tiles(m, h, tile_m=s0.tile_m, tile_n=256,
                               order=e1["order"])
        for tile in h_tiles:
            gate = z_tiles.get((tile.m0, tile.n0))
            if gate is None:
                reject(RK_E_SHAPE, "gate half has no producer tile",
                       stage="schedule", actual=[tile.m0, tile.n0])
            _, sig_view, pr, pc = cube_view(tile.rows, tile.cols)
            chunks = lut_chunks(pr * pc)
            last = None
            for index, (offset, count) in enumerate(chunks):
                linear = TensorLayout("float16", LINEAR_LAYOUT, 1, count, 1,
                                      count, 1, count * 2)
                src = self.add_view(self.views[gate].buffer_id, offset * 2,
                                    linear, (0, 1, 0, count))
                dst = self.add_view(self.views[sig_view].buffer_id,
                                    offset * 2, linear, (0, 1, 0, count))
                last = self.add_action(
                    "npu_task", template_id="co.lut.sigmoid.f16.v1",
                    inputs=(src,), outputs=(dst,),
                    deps=(z_tasks[(tile.m0, tile.n0)],) if index == 0
                    else (last,),
                    origin=f"lowering:ffn:sigmoid:{index}",
                    op_ids=(sig.id,),
                    attrs={"submit_group": f"ew:s1:{tile.m0}:{tile.n0}",
                           "source_stage_index": 1, "chunk_index": index,
                           "source_elementwise_schedule": e1,
                           "gate_source": {"view": gate,
                                           "action": z_tasks[(tile.m0, tile.n0)],
                                           "tile": [tile.m0, tile.n0,
                                                    tile.rows, tile.cols],
                                           "half": "gate"}})
            sig_tiles[(tile.m0, tile.n0)] = sig_view
            sig_tasks[(tile.m0, tile.n0)] = last

        h_tiles = output_tiles(m, h, tile_m=s0.tile_m, tile_n=256,
                               order=e2["order"])
        silu_tiles: dict[tuple[int, int], int] = {}
        silu_tasks: dict[tuple[int, int], int] = {}
        for tile in h_tiles:
            gate = z_tiles[(tile.m0, tile.n0)]
            sig_view = sig_tiles[(tile.m0, tile.n0)]
            _, silu_view, _, _ = cube_view(tile.rows, tile.cols)
            deps = (sig_tasks[(tile.m0, tile.n0)], z_tasks[(tile.m0, tile.n0)])
            if identity256_upload is not None:
                deps += (identity256_upload,)
            task = self.add_action(
                "npu_task", template_id="co.ew_mul.f16.v1",
                inputs=(gate, sig_view, identity256), outputs=(silu_view,),
                deps=deps, origin="lowering:ffn:silu_mul",
                op_ids=(silu.id,),
                attrs={"submit_group": f"ew:s2:{tile.m0}:{tile.n0}",
                       "source_stage_index": 2,
                       "source_elementwise_schedule": e2,
                       "gate_source": {"view": gate,
                                       "action": z_tasks[(tile.m0, tile.n0)]},
                       "sig_source": {"view": sig_view,
                                      "action": sig_tasks[(tile.m0, tile.n0)]}})
            silu_tiles[(tile.m0, tile.n0)] = silu_view
            silu_tasks[(tile.m0, tile.n0)] = task

        h_tiles = output_tiles(m, h, tile_m=s0.tile_m, tile_n=256,
                               order=e3["order"])
        for tile in h_tiles:
            silu_view = silu_tiles[(tile.m0, tile.n0)]
            up = z_tiles.get((tile.m0, h + tile.n0))
            if up is None:
                reject(RK_E_SHAPE, "up half has no producer tile",
                       stage="schedule", actual=[tile.m0, h + tile.n0])
            _, act_view, _, _ = cube_view(tile.rows, tile.cols)
            deps = (silu_tasks[(tile.m0, tile.n0)],
                    z_tasks[(tile.m0, h + tile.n0)])
            if identity256_upload is not None:
                deps += (identity256_upload,)
            task = self.add_action(
                "npu_task", template_id="co.ew_mul.f16.v1",
                inputs=(silu_view, up, identity256), outputs=(act_view,),
                deps=deps, origin="lowering:ffn:up_mul",
                op_ids=(act.id,),
                attrs={"submit_group": f"ew:s3:{tile.m0}:{tile.n0}",
                       "source_stage_index": 3,
                       "source_elementwise_schedule": e3,
                       "silu_source": {"view": silu_view,
                                       "action": silu_tasks[(tile.m0, tile.n0)]},
                       "up_source": {"view": up,
                                     "action": z_tasks[(tile.m0, h + tile.n0)],
                                     "half": "up"}})
            act_tiles[(tile.m0, tile.n0)] = act_view
            act_tasks[(tile.m0, tile.n0)] = task
            self.buffers[self.views[act_view].buffer_id].role = "workspace"

        # ----- stage 4: down projection -----------------------------------
        p_tiles: dict[tuple[int, int], int] = {}
        p_tasks: dict[tuple[int, int], int] = {}
        for tile in s4_tiles:
            ktiles = k_slices(h)
            a_views = []
            for k in ktiles:
                av = act_tiles.get((tile.m0, k.k0))
                if av is None:
                    reject(RK_E_SHAPE,
                           "down K region has no produced activation tile",
                           stage="schedule", actual=[tile.m0, k.k0])
                a_views.append(av)
            w_views = [weight_upload(wd.id, wd_binding, tile.n0, k.k0,
                                     tile.cols, k.extent)[0]
                       for k in ktiles]
            uploads = [weight_upload(wd.id, wd_binding, tile.n0, k.k0,
                                      tile.cols, k.extent)[1]
                       for k in ktiles]
            deps = tuple(act_tasks[(tile.m0, k.k0)] for k in ktiles)
            act_meta = [{"view": a_views[i],
                         "action": act_tasks[(tile.m0, k.k0)],
                         "k0": k.k0, "kt": k.extent}
                        for i, k in enumerate(ktiles)]
            view, task = matmul_chain(
                tile, a_views, w_views, uploads, ktiles, stage=4,
                group=f"mm:s4:{tile.m0}:{tile.n0}", source_op=op4,
                deps_extra=deps,
                extra_attrs={"activation_sources": act_meta})
            p_tiles[(tile.m0, tile.n0)] = view
            p_tasks[(tile.m0, tile.n0)] = task

        # ----- stage 5: residual add and publish ---------------------------
        for tile in output_tiles(m, d, tile_m=s4.tile_m,
                                 tile_n=s4.tile_n, order=e5["order"]):
            pv = p_tiles[(tile.m0, tile.n0)]
            xv, xu = input_upload(r.id, r_binding, tile.m0, tile.n0,
                                  tile.rows, tile.cols, d)
            _, yv, _, _ = cube_view(tile.rows, tile.cols)
            deps = (p_tasks[(tile.m0, tile.n0)], xu)
            if identity_out_upload is not None:
                deps += (identity_out_upload,)
            resid = self.add_action(
                "npu_task", template_id="co.ew_add.f16.v1",
                inputs=(pv, xv, identity_out), outputs=(yv,), deps=deps,
                origin="lowering:ffn:residual", op_ids=(res.id,),
                attrs={"submit_group": f"ew:s5:{tile.m0}:{tile.n0}",
                       "source_stage_index": 5,
                       "source_elementwise_schedule": e5,
                       "down_source": {"view": pv,
                                       "action": p_tasks[(tile.m0, tile.n0)]},
                       "residual_source": {"view": xv, "action": xu}})
            binding = self.binding_by_value[y.id]
            download = self.add_action(
                "download", inputs=(yv,), deps=(resid,),
                origin="host:download:ffn",
                attrs={"binding_id": binding, "submit_group": None})
            self.buffers[self.views[yv].buffer_id].role = "output"
            self.publish.append(Publish(
                self.bindings[binding].name, binding, yv,
                dst_row=tile.m0, dst_col=tile.n0, rows=tile.rows,
                cols=tile.cols))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build_swiglu(self) -> DraftPlan:
        """Lower the matched SwiGLU FFN, one primitive tile per stage.

        Every stage is its own submit group; no stage is fused, so the plan shows
        the real device passes instead of pretending to be one kernel.
        """
        ffn = self.legalized.swiglu
        x_value = self.module.value(ffn.input_id)
        m, d = x_value.type.shape
        h = ffn.hidden
        d_out = ffn.output
        contract = NumericContract.from_dict(
            ffn.matmul_gate.attrs["numeric_contract"])
        notes = {"composite": "swiglu_ffn_v1",
                 "stages": "gate,up,silu,mul,projection,residual",
                 "narrowing": list(contract.narrowing_points),
                 "source_ops": len(self.module.ops),
                 "fused": False}

        hid_phys = cube_layout(m, h)
        out_phys = cube_layout(m, d_out)

        def cube_tile(role, rows, cols, phys, padding="finite", reuse=True):
            layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, phys[0],
                                  phys[1], 8, phys[2])
            buffer_id = self.add_buffer(role=role, capacity=layout.bytes,
                                        layout_id=CUBE_LAYOUT, reuse=reuse,
                                        padding_state=padding)
            return buffer_id, self.add_view(buffer_id, 0, layout)

        def weight_tile(cols, kt, constant_value):
            pr, pc, pbytes = weight_layout(cols, kt)
            layout = TensorLayout("float16", WEIGHT_LAYOUT, cols, kt, pr, pc, 32,
                                  pbytes)
            buffer_id = self.add_buffer(role="weight", capacity=pbytes,
                                        layout_id=WEIGHT_LAYOUT, reuse=False,
                                        padding_state="zero")
            view_id = self.add_view(buffer_id, 0, layout)
            binding = self.binding_by_value[constant_value.id]
            # The RKIR weights already use the custom-ops [N,K] convention, so the
            # pack is a straight copy: no source transposition is involved.
            upload = self.add_action(
                "upload", outputs=(view_id,), origin="host_pack:swiglu_weight",
                attrs={"source_binding": binding, "src_row_offset": 0,
                       "src_col_offset": 0, "src_row_stride": kt,
                       "transpose_source": False, "cache": "constant"})
            return view_id, upload

        # one shared input tile feeds both branches and the residual
        x_buffer, x_view = cube_tile("input_slice", m, d, cube_layout(m, d),
                                     padding="zero", reuse=False)
        x_binding = self.binding_by_value[x_value.id]
        x_upload = self.add_action(
            "upload", outputs=(x_view,), origin="host_pack:swiglu_input",
            attrs={"source_binding": x_binding, "src_row_offset": 0,
                   "src_col_offset": 0, "src_row_stride": d,
                   "transpose_source": False, "cache": "input"})

        wg_value, wu_value, wd_value = (self.module.value(ffn.matmul_gate.inputs[1]),
                                        self.module.value(ffn.matmul_up.inputs[1]),
                                        self.module.value(ffn.matmul_proj.inputs[1]))
        wg_view, wg_upload = weight_tile(h, d, wg_value)
        wu_view, wu_upload = weight_tile(h, d, wu_value)
        wd_view, wd_upload = weight_tile(d_out, h, wd_value)

        uploads = (x_upload, wg_upload, wu_upload, wd_upload)
        _, gate_view = cube_tile("accumulator", m, h, hid_phys)
        gate = self.add_action("npu_task", template_id="co.mm.f16.v1",
                               inputs=(x_view, wg_view), outputs=(gate_view,),
                               deps=uploads, origin="lowering:swiglu:gate",
                               op_ids=(ffn.matmul_gate.id,),
                               attrs={"submit_group": "swiglu:gate",
                                      "attrs_note": notes})
        _, up_view = cube_tile("accumulator", m, h, hid_phys)
        up = self.add_action("npu_task", template_id="co.mm.f16.v1",
                             inputs=(x_view, wu_view), outputs=(up_view,),
                             deps=uploads, origin="lowering:swiglu:up",
                             op_ids=(ffn.matmul_up.id,),
                             attrs={"submit_group": "swiglu:up"})

        # SiLU gate: chunked LUT then multiply, exactly like the single-matmul path
        _, identity_view = self.identity_for(hid_phys[1])
        identity_upload = self._last_upload_of(identity_view)
        _, gate_lut_buffer = cube_tile("gate", m, h, hid_phys, reuse=False)
        chunks = lut_chunks(hid_phys[0] * hid_phys[1])
        lut_tasks: list[int] = []
        for index, (offset, count) in enumerate(chunks):
            linear = TensorLayout("float16", LINEAR_LAYOUT, 1, count, 1, count, 1,
                                  count * 2)
            src = self.add_view(self.views[gate_view].buffer_id, offset * 2,
                                linear, (0, 1, 0, count))
            dst = self.add_view(gate_lut_buffer, offset * 2, linear,
                                (0, 1, 0, count))
            lut_tasks.append(self.add_action(
                "npu_task", template_id="co.lut.sigmoid.f16.v1",
                inputs=(src,), outputs=(dst,),
                deps=(gate,) if index == 0 else (lut_tasks[-1],),
                origin=f"lowering:swiglu:lut:{index}",
                op_ids=(ffn.activation.id,),
                attrs={"submit_group": "swiglu:silu", "chunk_index": index}))
        _, act_view = cube_tile("accumulator", m, h, hid_phys)
        act = self.add_action(
            "npu_task", template_id="co.ew_mul.f16.v1",
            inputs=(gate_view, self._gate_cube_view(gate_lut_buffer, m, h,
                                                    hid_phys), identity_view),
            outputs=(act_view,),
            deps=(lut_tasks[-1],) + ((identity_upload,)
                                     if identity_upload is not None else ()),
            origin="lowering:swiglu:silu_mul",
            op_ids=(ffn.activation.id,),
            attrs={"submit_group": "swiglu:silu"})
        _, gated_view = cube_tile("accumulator", m, h, hid_phys)
        gated = self.add_action(
            "npu_task", template_id="co.ew_mul.f16.v1",
            inputs=(act_view, up_view, identity_view), outputs=(gated_view,),
            deps=(act, up) + ((identity_upload,) if identity_upload is not None
                              else ()),
            origin="lowering:swiglu:up_mul", op_ids=(ffn.multiply.id,),
            attrs={"submit_group": "swiglu:mul"})

        _, proj_view = cube_tile("accumulator", m, d_out, out_phys)
        proj = self.add_action("npu_task", template_id="co.mm.f16.v1",
                               inputs=(gated_view, wd_view),
                               outputs=(proj_view,), deps=(gated, wd_upload),
                               origin="lowering:swiglu:projection",
                               op_ids=(ffn.matmul_proj.id,),
                               attrs={"submit_group": "swiglu:proj"})
        _, out_view = cube_tile("output", m, d_out, out_phys)
        # The residual EW works on the output tile, whose physical width differs
        # from the hidden tile, so it needs its own identity weight.
        _, out_identity = self.identity_for(out_phys[1])
        out_identity_upload = self._last_upload_of(out_identity)
        residual = self.add_action(
            "npu_task", template_id="co.ew_add.f16.v1",
            inputs=(proj_view, x_view, out_identity), outputs=(out_view,),
            deps=(proj, x_upload) + ((out_identity_upload,)
                                     if out_identity_upload is not None else ()),
            origin="lowering:swiglu:residual", op_ids=(ffn.add.id,),
            attrs={"submit_group": "swiglu:residual"})
        binding_id = self.binding_by_value[ffn.add.outputs[0]]
        self.add_action("download", inputs=(out_view,), deps=(residual,),
                        origin="host:download",
                        attrs={"binding_id": binding_id, "submit_group": None})
        self.publish.append(Publish(self.bindings[binding_id].name, binding_id,
                                    out_view, dst_row=0, dst_col=0, rows=m,
                                    cols=d_out))
        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def _gate_cube_view(self, buffer_id: int, rows: int, cols: int,
                        phys) -> int:
        layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, phys[0],
                              phys[1], 8, phys[2])
        return self.add_view(buffer_id, 0, layout)

    def build_instantiated_linear(self, *, tasks: list[dict[str, Any]],
                                  facts: dict[str, Any]) -> DraftPlan:
        """Assemble a plan directly from the instantiated target task sequence.

        The lowering has already proved the source program and emitted
        ``rk_fill``/``rk_copy``/``rk_gemm``/``rk_silu``/``rk_store`` calls in
        execution order.  This method consumes that stream once, in order, as
        the authority for values, regions, producer/consumer relations and
        publication.  It only asks the existing _Builder machinery for layout,
        packing, buffer/view storage, templates, submits and accounting.

        Each actual operation updates or consumes an established value state:

        * fill               -> establish a zero region for a new lifetime;
        * param -> staging   -> create/reuse upload and attach pack-padding;
        * running -> Cl      -> record the accumulator load relation;
        * Cl -> running      -> seal one MM/MM_ACC writeback and value version;
        * gemm               -> emit MM or MM_ACC from the visible operand and
                                accumulator states, never by K-index grouping;
        * silu               -> consume the actual value and emit LUT + multiply;
        * store              -> consume the actual result value, then emit the
                                download and published rectangle.

        There is no GEMM-neighborhood search and no collection of requests that
        are later regrouped into a new execution skeleton.  Dependency
        finalization may attach known task counts and materialize deferred host
        downloads, but it never reconstructs tiles, K slices or task order.
        """
        if tasks and any("input_tokens" in task or
                         "buffer_token" in task for task in tasks):
            return self._consume_multi_op_tasks(tasks=tasks, facts=facts)
        matmul_ops = [item for item in self.module.ops
                      if item.opcode == "matmul"]
        activation_ops = [item for item in self.module.ops
                          if item.opcode == "silu"]
        if len(matmul_ops) != 1 or len(activation_ops) > 1 or \
                len(self.module.ops) != 1 + len(activation_ops):
            reject(RK_E_UNSUPPORTED,
                   "task assembly requires one matmul binding op and at most "
                   "one SiLU op", stage="schedule",
                   actual=[item.opcode for item in self.module.ops])
        if not isinstance(tasks, list) or not tasks:
            reject(RK_E_SHAPE, "the instantiated task sequence is empty",
                   stage="schedule")
        op = matmul_ops[0]
        activation_op = activation_ops[0] if activation_ops else None
        a_value = self.module.value(op.inputs[0])
        b_value = self.module.value(op.inputs[1])
        matmul_out = self.module.value(op.outputs[0])
        y_value = self.module.value(self.module.outputs[0])
        m, k = a_value.type.shape
        n = y_value.type.shape[1]
        if activation_op is None:
            if matmul_out.id != y_value.id:
                reject(RK_E_SHAPE,
                       "the matmul binding carrier has no final output",
                       stage="schedule")
        elif activation_op.inputs != (matmul_out.id,) or \
                activation_op.outputs != (y_value.id,):
            reject(RK_E_SHAPE,
                   "the SiLU binding op does not connect the matmul result to "
                   "the final output", stage="schedule")
        if (facts.get("m"), facts.get("k"), facts.get("n")) != (m, k, n):
            reject(RK_E_SHAPE,
                   "task facts disagree with the binding carrier",
                   stage="schedule",
                   actual={"facts": [facts.get("m"), facts.get("k"),
                                     facts.get("n")],
                           "bindings": [m, k, n]})
        if op.attrs.get("b_layout", "kn") != "kn" or                 b_value.type.shape != (k, n):
            reject(RK_E_UNSUPPORTED,
                   "task assembly supports only B=[K,N] in kn layout",
                   stage="schedule")
        order = facts.get("order", "m_major")
        if order not in MN_TILE_ORDERS:
            reject(RK_E_UNSUPPORTED, f"unknown tile order {order!r}",
                   stage="schedule", actual=order)

        def integer(field: str, value: Any, where: str) -> int:
            if type(value) is not int:
                reject(RK_E_SCHEMA,
                       f"{where}.{field} must be a static integer",
                       stage="schedule", actual={field: value})
            return value

        def region(task: dict[str, Any], key: str, where: str) -> dict[str, int]:
            raw = task.get(key)
            if not isinstance(raw, dict):
                reject(RK_E_SCHEMA, f"{where}.{key} must be an object",
                       stage="schedule", actual=raw)
            return {axis: integer(axis, raw.get(axis), where + "." + key)
                    for axis in ("row", "col", "rows", "cols")}

        def task_rect(task: dict[str, Any], key: str, where: str) -> _TaskRect:
            raw = region(task, key, where)
            return _TaskRect(raw["row"], raw["col"], raw["rows"], raw["cols"])

        def local_rect(task: dict[str, Any], where: str) -> _TaskRect:
            return _TaskRect(0, 0,
                             integer("rows", task.get("rows"), where),
                             integer("cols", task.get("cols"), where))

        policy = self.options.linear_task_order
        if policy not in {"baseline", "scheduled_baseline", "merged",
                          "interleaved", "interleaved_reuse",
                          "weight_stationary", "weight_stationary_reuse",
                          "grouped_linear"}:
            reject(RK_E_UNSUPPORTED, f"unknown linear task policy {policy!r}",
                   stage="schedule")
        if activation_op is not None and policy == "merged":
            reject(RK_E_UNSUPPORTED,
                   "merged linear grouping is not admitted for SiLU plans",
                   stage="schedule")
        legacy_linear = bool(facts.get("guarded_tail") or
                             facts.get("linear_k_loop"))
        compat_records = bool(facts.get("formal_guarded_entry") or
                              facts.get("linear_k_loop"))
        schedule = MNTileSchedule(tile_m=facts["tile_m"],
                                  tile_n=facts["tile_n"], order=order)
        schedule_dict = schedule.to_dict()
        a_binding = self.binding_by_value[a_value.id]
        b_binding = self.binding_by_value[b_value.id]
        y_binding = self.binding_by_value[y_value.id]

        source_tasks = [dict(task) for task in tasks]
        for task in source_tasks:
            if task.get("kind") != "copy":
                continue
            source = str(task.get("source"))
            if source not in (a_value.name, b_value.name):
                continue
            task["_upload_spec"] = {
                "source_region": region(task, "source_region", "copy source"),
                "weight": source == b_value.name,
            }
        tasks = source_tasks
        # A pack is still a pure function of the actual source region and target
        # layout.  The cache is only a resource-reuse decision; each task keeps
        # its explicit upload dependency.
        input_cache: dict[tuple[Any, ...], tuple[int, int]] = {}
        weight_cache: dict[tuple[Any, ...], tuple[int, int]] = {}

        def source_upload(source: dict[str, int], binding: int,
                          *, weight: bool) -> tuple[int, int]:
            key = (binding, source["row"], source["col"], source["rows"],
                   source["cols"])
            cache = weight_cache if weight else input_cache
            if key in cache:
                return cache[key]
            if weight:
                pr, pc, pbytes = weight_layout(source["cols"], source["rows"])
                layout = TensorLayout("float16", WEIGHT_LAYOUT, source["cols"],
                                      source["rows"], pr, pc, 32, pbytes)
                buffer_id = self.add_buffer(role="weight", capacity=pbytes,
                                            layout_id=WEIGHT_LAYOUT,
                                            reuse=False,
                                            padding_state="zero")
                view = self.add_view(buffer_id, 0, layout)
                action = self.add_action(
                    "upload", outputs=(view,), origin="host_pack:linear:weight",
                    attrs={"source_binding": binding,
                           "src_row_offset": source["row"],
                           "src_col_offset": source["col"],
                           "src_row_stride": n,
                           "transpose_source": True,
                           "cache": ("constant"
                                     if b_value.role == "constant" else "input")})
            else:
                pr, pc, pbytes = input_layout(source["rows"], source["cols"])
                layout = TensorLayout("float16", CUBE_LAYOUT, source["rows"],
                                      source["cols"], pr, pc, 8, pbytes)
                buffer_id = self.add_buffer(role="input_slice", capacity=pbytes,
                                            layout_id=CUBE_LAYOUT, reuse=False,
                                            padding_state="zero")
                view = self.add_view(buffer_id, 0, layout)
                action = self.add_action(
                    "upload", outputs=(view,), origin="host_pack:linear:input",
                    attrs={"source_binding": binding,
                           "src_row_offset": source["row"],
                           "src_col_offset": source["col"],
                           "src_row_stride": k,
                           "transpose_source": False, "cache": "input"})
            cache[key] = (view, action)
            return cache[key]

        # Host pack actions are materialized from the actual A/B copy tasks
        # before task groups.  This is a dependency/resource decision, not a
        # GEMM-chain search: each upload is keyed to its concrete copy task.
        task_uploads: dict[int, tuple[int, int]] = {}
        baseline_segment_uploads: dict[int, list[int]] = {}
        current_segment_uploads: list[int] | None = None
        a_tile_names: set[str] = set()
        b_tile_names: set[str] = set()
        c_tile_names: set[str] = set()
        for task in tasks:
            if task.get("kind") == "gemm" and \
                    isinstance(task.get("tiles"), list) and \
                    len(task["tiles"]) == 3:
                a_tile_names.add(str(task["tiles"][0]))
                b_tile_names.add(str(task["tiles"][1]))
                c_tile_names.add(str(task["tiles"][2]))
        if legacy_linear and policy == "baseline":
            segment: list[int] = []
            for index, task in enumerate(tasks):
                segment.append(index)
                if task.get("kind") == "store":
                    copies = [
                        item for item in segment
                        if tasks[item].get("kind") == "copy" and
                        str(tasks[item].get("source")) in
                        (a_value.name, b_value.name)]
                    baseline_segment_uploads[segment[0]] = sorted(
                        copies, key=lambda item: (
                            0 if str(tasks[item].get("source")) ==
                            a_value.name else 1, item))
                    segment = []
            if segment:
                reject(RK_E_SCHEMA,
                       "the baseline task stream ends without a store",
                       stage="schedule")
        else:
            # Upload actions are materialised in source task order.  Policy
            # reordering only changes execution order, not the packed upload
            # identities or their lifetimes.
            for weight_pass in (False, True):
                for task in source_tasks:
                    spec = task.get("_upload_spec")
                    if spec is None or bool(spec["weight"]) != weight_pass:
                        continue
                    source_upload(
                        spec["source_region"],
                        b_binding if weight_pass else a_binding,
                        weight=weight_pass)



        # Actual region/version/producer state.  A zero fill starts a new zero
        # lifetime; a value write records the task ordinal, plan action and view
        # that produced the value.  No GEMM-neighborhood search is performed.
        states: dict[str, _TaskBufferState] = {
            a_value.name: _TaskBufferState(
                value=_TaskRect(0, 0, m, k), value_version=1),
            b_value.name: _TaskBufferState(
                value=_TaskRect(0, 0, k, n), value_version=1),
        }
        linear_actions: list[dict[str, Any]] = []

        def begin_zero(name: str, area: _TaskRect, task: int) -> _TaskBufferState:
            state = states.setdefault(name, _TaskBufferState())
            state.enter_zero(area, task)
            return state

        def write_value(name: str, area: _TaskRect, task: int,
                        action: int | None, view: int | None,
                        tile: tuple[int, int, int, int] | None,
                        source_region: _TaskRect | None = None) -> None:
            state = states.setdefault(name, _TaskBufferState())
            if state.zero is not None and not state.zero.contains(area) and \
                    not area.contains(state.zero):
                reject(RK_E_UNSUPPORTED,
                       "a value write overlaps a zero state that this task "
                       "stream cannot represent", stage="schedule",
                       actual={"buffer": name, "write": area,
                               "zero": state.zero})
            state.record_value(area, task, view, action, tile, source_region)

        acc_views: dict[tuple[int, int, int, int], list[int]] = {}
        tile_k_count: dict[tuple[int, int, int, int], int] = {}
        tile_k_ranges: dict[tuple[int, int, int, int],
                            list[tuple[int, int]]] = {}
        mm_task_ids: dict[tuple[int, int, int, int], list[int]] = {}
        pending_loads: dict[str, dict[str, Any]] = {}
        pending_gemms: dict[str, dict[str, Any]] = {}
        publish_records: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        deferred_publishes: list[tuple[int, int, int, int]] = []
        silu_count = 0
        guarded_last_mm: int | None = None
        # Local pass diagnostics only; this is not a public plan artifact.
        fusion_records: list[dict[str, Any]] = []

        def accumulator_view(tile: tuple[int, int, int, int],
                             k_index: int) -> int:
            pair = acc_views.setdefault(tile, [])
            slot = k_index & 1
            if slot >= len(pair):
                m0, n0, rows, cols = tile
                cr, cc, cbytes = cube_layout(rows, cols)
                layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cr, cc, 8, cbytes)
                buffer_id = self.add_buffer(role="accumulator", capacity=cbytes,
                                            layout_id=CUBE_LAYOUT,
                                            padding_state="finite")
                pair.append(self.add_view(buffer_id, 0, layout))
            return pair[slot]

        def materialize_publish(tile: tuple[int, int, int, int]) -> int:
            record = publish_records[tile]
            view = record["view"]
            download = self.add_action(
                "download", inputs=(view,), deps=(record["action"],),
                origin="host:download:linear",
                attrs={"binding_id": y_binding, "submit_group": None})
            self.buffers[self.views[view].buffer_id].role = "output"
            self.buffers[self.views[view].buffer_id].last = max(
                self.buffers[self.views[view].buffer_id].last, download)
            self.publish.append(Publish(
                y_value.name, y_binding, view,
                dst_row=record["m0"], dst_col=record["n0"],
                rows=record["rows"], cols=record["cols"]))
            return download

        for index, task in enumerate(tasks):
            if index in baseline_segment_uploads:
                current_segment_uploads = []
                for copy_index in baseline_segment_uploads[index]:
                    copy_task = tasks[copy_index]
                    source = str(copy_task.get("source"))
                    source_region = region(
                        copy_task, "source_region", "copy source")
                    task_uploads[copy_index] = source_upload(
                        source_region,
                        a_binding if source == a_value.name else b_binding,
                        weight=source == b_value.name)
                    current_segment_uploads.append(task_uploads[copy_index][1])
            kind = task.get("kind")
            if kind not in {"fill", "copy", "gemm", "silu", "store"}:
                reject(RK_E_UNSUPPORTED,
                       "the instantiated task kind is not consumed",
                       stage="schedule", actual=kind)

            if kind == "fill":
                destination = str(task.get("destination"))
                value = task.get("value")
                if value != 0:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "a fill consumed by the task chain is not zero",
                           stage="schedule", actual=task)
                area = task_rect(task, "region", "fill")
                state = begin_zero(destination, area, index)
                fusion_records.append({
                    "kind": "zero-lifecycle", "task": index,
                    "buffer": destination, "region": area,
                    "zero_producer": state.zero_producer,
                })
                continue

            if kind == "copy":
                source = str(task.get("source"))
                destination = str(task.get("destination"))
                source_region = task_rect(task, "source_region", "copy source")
                destination_region = task_rect(
                    task, "destination_region", "copy destination")

                if source == a_value.name or source == b_value.name:
                    spec = task.get("_upload_spec")
                    if spec is not None:
                        task_uploads[index] = source_upload(
                            spec["source_region"],
                            b_binding if spec["weight"] else a_binding,
                            weight=spec["weight"])
                    source_state = states.get(source)
                    if source_state is None or \
                            not source_state.value_covers(source_region):
                        reject(RK_E_SCHEMA,
                               "a staging copy reads an unestablished source "
                               "region", stage="schedule",
                               actual={"source": source,
                                       "region": source_region})
                    destination_state = states.get(destination)
                    if destination_state is None or \
                            not destination_state.zero_covers(destination_region):
                        reject(RK_E_NUMERIC_CONTRACT,
                               "an operand staging copy lacks a zero-filled "
                               "write region", stage="schedule",
                               actual={"destination": destination,
                                       "region": destination_region})
                    view, upload = task_uploads[index]
                    zero_task = destination_state.zero_producer
                    write_value(destination, destination_region, index,
                                upload, view, None,
                                source_region=source_region)
                    fusion_records.append({
                        "kind": "pack-padding", "task": index,
                        "source": source, "destination": destination,
                        "zero_task": zero_task, "upload": upload,
                        "value_version": states[destination].value_version,
                    })
                    continue

                if destination in a_tile_names or destination in b_tile_names:
                    reject(RK_E_SCHEMA,
                           "a staging task does not copy the bound input into "
                           "the GEMM tile", stage="schedule",
                           actual={"source": source,
                                   "destination": destination})
                if destination in pending_gemms or source in pending_gemms:
                    pending = pending_gemms.pop(source, None)
                    if pending is None:
                        reject(RK_E_SCHEMA,
                               "an accumulator writeback has no preceding GEMM",
                               stage="schedule", actual=task)
                    if destination not in states:
                        reject(RK_E_SCHEMA,
                               "an accumulator writeback has no running state",
                               stage="schedule", actual=destination)
                    state = states[destination]
                    write_value(destination, destination_region, index,
                                pending["action"], pending["view"],
                                pending["tile"])
                    fusion_records.append({
                        "kind": "mm-writeback", "task": index,
                        "gemm_task": pending["task"],
                        "action": pending["action"],
                        "destination": destination,
                        "value_version": state.value_version,
                    })
                    continue

                source_state = states.get(source)
                if source_state is None or not (
                        source_state.value_covers(source_region) or
                        source_state.zero_covers(source_region)):
                    if source_state is not None and \
                            source_state.zero is not None:
                        reject(RK_E_NUMERIC_CONTRACT,
                               "the first accumulator load must read the "
                               "complete zero-filled running state",
                               stage="schedule",
                               actual={"source": source,
                                       "region": source_region,
                                       "zero": source_state.zero})
                    reject(RK_E_NUMERIC_CONTRACT,
                           "the accumulator load has no producer state",
                           stage="schedule",
                           actual={"source": source,
                                   "region": source_region})
                pending_loads[destination] = {
                    "task": index,
                    "source": source,
                    "source_region": source_region,
                    "destination_region": destination_region,
                    "prior_value": source_state.value is not None and
                                   source_state.value_covers(source_region),
                    "prior_view": source_state.value_view,
                    "prior_action": source_state.value_action,
                    "prior_producer": source_state.value_producer,
                    "prior_tile": source_state.value_tile,
                    "prior_version": source_state.value_version,
                    "zero_producer": source_state.zero_producer,
                }
                cl_state = states.setdefault(destination, _TaskBufferState())
                cl_state.zero = (destination_region
                                 if pending_loads[destination]["prior_value"]
                                 is False else None)
                cl_state.zero_producer = source_state.zero_producer
                cl_state.value = (destination_region
                                  if pending_loads[destination]["prior_value"]
                                  else None)
                cl_state.value_producer = source_state.value_producer
                cl_state.value_view = source_state.value_view
                cl_state.value_action = source_state.value_action
                cl_state.value_tile = source_state.value_tile
                cl_state.value_version = source_state.value_version
                fusion_records.append({
                    "kind": "accumulator-load", "task": index,
                    "source": source, "destination": destination,
                    "prior_value": pending_loads[destination]["prior_value"],
                    "prior_action": pending_loads[destination]["prior_action"],
                })
                continue

            if kind == "gemm":
                if task.get("operands") != [a_value.name, b_value.name,
                                            matmul_out.name]:
                    reject(RK_E_SCHEMA,
                           "a GEMM task names operands the binding carrier does "
                           "not bind", stage="schedule",
                           actual={"task": task.get("operands"),
                                   "bindings": [a_value.name, b_value.name,
                                                matmul_out.name]})
                m0 = integer("m0", task.get("m0"), f"task[{index}]")
                n0 = integer("n0", task.get("n0"), f"task[{index}]")
                rows = integer("rows", task.get("rows"), f"task[{index}]")
                cols = integer("cols", task.get("cols"), f"task[{index}]")
                k0 = integer("k0", task.get("k0"), f"task[{index}]")
                kt = integer("kt", task.get("kt"), f"task[{index}]")
                if min(m0, n0, rows, cols, k0, kt) < 0 or \
                        rows <= 0 or cols <= 0 or kt <= 0:
                    reject(RK_E_SHAPE, "a GEMM task carries an invalid rectangle",
                           stage="schedule", actual=task)
                tiles = task.get("tiles")
                if not isinstance(tiles, list) or len(tiles) != 3:
                    reject(RK_E_SCHEMA,
                           "a GEMM task must name three tile buffers",
                           stage="schedule", actual=tiles)
                a_tile, b_tile, c_tile = (str(item) for item in tiles)
                a_state = states.get(a_tile)
                b_state = states.get(b_tile)
                c_state = states.get(c_tile)
                a_full = _TaskRect(0, 0, rows, kt)
                b_full = _TaskRect(0, 0, kt, cols)
                c_full = _TaskRect(0, 0, rows, cols)
                if a_state is None or not a_state.zero_covers(a_full) or \
                        a_state.value is None or \
                        not a_full.contains(a_state.value):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "GEMM operand A does not have a logical value over "
                           "a zero-filled physical tile", stage="schedule",
                           actual={"tile": a_tile, "physical": a_full,
                                   "state": a_state})
                if b_state is None or not b_state.zero_covers(b_full) or \
                        b_state.value is None or \
                        not b_full.contains(b_state.value):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "GEMM operand B does not have a logical value over "
                           "a zero-filled physical tile", stage="schedule",
                           actual={"tile": b_tile, "physical": b_full,
                                   "state": b_state})
                logical_rows = a_state.value.rows
                logical_cols = b_state.value.cols
                logical_kt = a_state.value.cols
                if a_state.source_region is None or b_state.source_region is None:
                    reject(RK_E_SCHEMA,
                           "a GEMM staging tile has no actual source region",
                           stage="schedule")
                if a_state.source_region.row != m0 or \
                        a_state.source_region.col != k0 or \
                        a_state.source_region.rows != logical_rows or \
                        a_state.source_region.cols != logical_kt:
                    reject(RK_E_SCHEMA,
                           "the A staging task does not copy the bound input "
                           "into the GEMM A tile", stage="schedule",
                           actual={"source": a_state.source_region,
                                   "expected": [m0, k0, logical_rows,
                                                logical_kt]})
                if b_state.source_region.row != k0 or \
                        b_state.source_region.col != n0 or \
                        b_state.source_region.rows != logical_kt or \
                        b_state.source_region.cols != logical_cols:
                    reject(RK_E_SCHEMA,
                           "the B staging task does not copy the bound weight "
                           "into the GEMM B tile", stage="schedule",
                           actual={"source": b_state.source_region,
                                   "expected": [k0, n0, logical_kt,
                                                logical_cols]})
                if c_state is None or not (
                        c_state.value_covers(c_full) or
                        c_state.zero_covers(c_full)):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "GEMM has no accumulated Cl state", stage="schedule",
                           actual={"tile": c_tile, "physical": c_full})
                if c_tile not in pending_loads:
                    reject(RK_E_SCHEMA,
                           "GEMM consumes a Cl tile with no accumulator load",
                           stage="schedule", actual=c_tile)
                pending_loads.pop(c_tile, None)
                tile = (m0, n0, logical_rows, logical_cols)
                kidx = tile_k_count.get(tile, 0)
                output_view = accumulator_view(tile, kidx)
                prior_view = c_state.value_view if c_state.value is not None \
                    else None
                prior_action = c_state.value_action if c_state.value is not None \
                    else None
                deps = tuple(dict.fromkeys(
                    action for action in (
                        a_state.value_action, b_state.value_action, prior_action)
                    if action is not None))
                if legacy_linear and policy == "baseline":
                    if prior_view is None:
                        deps = tuple(dict.fromkeys(
                            tuple(current_segment_uploads or ()) + deps +
                            ((guarded_last_mm,)
                             if guarded_last_mm is not None else ())))
                    else:
                        # The baseline planner pins all uploads for this tile
                        # group to the first MM task and keeps later K tasks on
                        # the accumulation dependency only.
                        deps = (prior_action,)
                if facts.get("linear_k_loop") and policy == "baseline":
                    submit_group = f"mm:{tile[0]}:{tile[1]}"
                elif legacy_linear and policy == "baseline":
                    submit_group = (
                        f"mm:{m0 // facts['tile_m']}:"
                        f"{n0 // facts['tile_n']}")
                elif policy == "scheduled_baseline":
                    submit_group = f"mm:{tile[0]}:{tile[1]}"
                elif policy == "baseline":
                    submit_group = f"mm:{tile[0]}:{tile[1]}"
                else:
                    submit_group = "mk:linear"
                recipe = ("linear_k_acc_v1"
                          if policy in {"baseline", "scheduled_baseline"}
                          else "multi_k_acc_v1")
                physical_kt = logical_kt
                if a_state.value_view is not None:
                    physical_kt = int(
                        self.views[a_state.value_view].tensor.physical_cols)
                attrs = {"submit_group": submit_group,
                         "k0": k0, "k_index": kidx, "kt": logical_kt}
                if not (legacy_linear and policy == "baseline"):
                    attrs.update({
                        "submit_recipe": recipe,
                        "data_reuse": bool(task.get("policy_data_reuse")),
                        "weight_reuse": bool(task.get("policy_weight_reuse")),
                        "linear_task_policy": policy,
                    })
                if facts.get("linear_k_loop") and \
                        not facts.get("schedule_declared"):
                    attrs["source_schedule"] = None
                    attrs["source_k_rounding"] = facts.get("rounding")
                    attrs["source_mn_tiling"] = None
                    attrs["applied_schedule"] = schedule_dict
                elif facts.get("schedule_declared") or policy in {
                        "baseline", "scheduled_baseline"}:
                    attrs["source_schedule"] = schedule_dict
                    attrs["source_k_rounding"] = facts.get("rounding")
                    attrs["source_mn_tiling"] = None
                    attrs["applied_schedule"] = schedule_dict
                if facts.get("hints") and not compat_records:
                    attrs["source_hints"] = dict(facts["hints"])
                if prior_view is None:
                    task_id = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(a_state.value_view, b_state.value_view),
                        outputs=(output_view,), deps=deps,
                        origin=(f"lowering:linear:{policy}:k{kidx}:"
                                f"n{n0}"),
                        op_ids=(op.id,), attrs=attrs)
                else:
                    task_id = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(a_state.value_view, b_state.value_view,
                                prior_view),
                        outputs=(output_view,), deps=deps,
                        origin=(f"lowering:linear:{policy}:k{kidx}:"
                                f"n{n0}"),
                        op_ids=(op.id,), attrs=attrs)
                if legacy_linear:
                    guarded_last_mm = task_id
                    linear_actions.append({
                        "action": task_id, "tile": tile,
                        "k_index": kidx, "k0": k0, "kt": logical_kt,
                    })
                pending_gemms[c_tile] = {
                    "task": index, "action": task_id, "view": output_view,
                    "tile": tile, "region": c_full,
                }
                write_value(c_tile, c_full, index, task_id, output_view, tile)
                tile_k_count[tile] = kidx + 1
                tile_k_ranges.setdefault(tile, []).append((k0, logical_kt))
                mm_task_ids.setdefault(tile, []).append(task_id)
                fusion_records.append({
                    "kind": "mm-chain", "task": index,
                    "action": task_id, "tile": tile, "k0": k0, "kt": kt,
                    "prior_action": prior_action,
                    "operands": [a_state.value_action,
                                 b_state.value_action],
                })
                continue

            if kind == "silu":
                source = str(task.get("source"))
                area = local_rect(task, "silu")
                state = states.get(source)
                if state is None or not state.value_covers(area) or \
                        state.value_view is None or state.value_action is None:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "SiLU does not consume an established value",
                           stage="schedule",
                           actual={"source": source, "region": area})
                if activation_op is None:
                    reject(RK_E_SCHEMA,
                           "a SiLU task has no activation binding carrier",
                           stage="schedule")
                tile = state.value_tile
                if tile is None or tile[2] != area.rows or \
                        tile[3] != area.cols:
                    reject(RK_E_SCHEMA,
                           "SiLU region does not match its value tile",
                           stage="schedule", actual=tile)
                input_action = state.value_action
                input_view = state.value_view
                input_version = state.value_version
                cr, cc, _ = cube_layout(area.rows, area.cols)
                group = f"act:{tile[0]}:{tile[1]}"
                activated = self._emit_activation(
                    activation="silu", rows=area.rows, cols=area.cols,
                    cube_rows=cr, cube_cols=cc, input_view=input_view,
                    input_buffer=self.views[input_view].buffer_id,
                    deps=(input_action,), group=group,
                    op_ids=(activation_op.id,))
                action_id = self._last_task_of_group(group)
                write_value(source, area, index, action_id, activated, tile)
                silu_count += 1
                fusion_records.append({
                    "kind": "activation", "task": index,
                    "input_action": input_action,
                    "input_view": input_view,
                    "input_version": input_version,
                    "output_action": action_id,
                    "output_view": activated,
                    "output_version": states[source].value_version,
                    "tile": tile,
                })
                continue

            # rk_store: consume the actual value produced by the preceding
            # chain, then apply the existing baseline/deferred download policy.
            source = str(task.get("source"))
            if str(task.get("destination")) != y_value.name:
                reject(RK_E_SCHEMA,
                       "a store task writes a binding other than the output",
                       stage="schedule", actual=task.get("destination"))
            area = local_rect(task, "store")
            m0 = integer("m0", task.get("m0"), f"task[{index}]")
            n0 = integer("n0", task.get("n0"), f"task[{index}]")
            tile = (m0, n0, area.rows, area.cols)
            state = states.get(source)
            if state is None or not state.value_covers(area) or \
                    state.value_view is None or state.value_action is None:
                reject(RK_E_NUMERIC_CONTRACT,
                       "store does not read an established result value",
                       stage="schedule",
                       actual={"source": source, "region": area})
            if tile in publish_records:
                reject(RK_E_SCHEMA,
                       "an instantiated tile has more than one output store",
                       stage="schedule", actual=tile)
            if state.value_tile is not None and state.value_tile != tile:
                reject(RK_E_SCHEMA,
                       "the store lacks its FP16 accumulator writeback for "
                       "the published rectangle", stage="schedule",
                       actual={"store": tile, "value": state.value_tile})
            publish_records[tile] = {
                "m0": m0, "n0": n0, "rows": area.rows, "cols": area.cols,
                "view": state.value_view, "action": state.value_action,
                "value_version": state.value_version,
            }
            if policy == "baseline":
                materialize_publish(tile)
                current_segment_uploads = None
            else:
                deferred_publishes.append(tile)

        if pending_loads or pending_gemms:
            reject(RK_E_SCHEMA,
                   "the task stream ends with an unfinished accumulator chain",
                   stage="schedule",
                   actual={"loads": sorted(pending_loads),
                           "gemms": sorted(pending_gemms)})
        if bool(silu_count) != (activation_op is not None):
            reject(RK_E_SCHEMA,
                   "the SiLU task stream and binding carrier disagree on "
                   "activation", stage="schedule",
                   actual={"silu_tasks": silu_count,
                           "binding_activation": None if activation_op is None
                                                  else activation_op.opcode})
        if not publish_records:
            reject(RK_E_SHAPE, "the task stream publishes no output tile",
                   stage="schedule")
        if set(acc_views) != set(publish_records):
            reject(RK_E_SHAPE,
                   "every produced tile needs exactly one store",
                   stage="schedule",
                   actual={"tiles": sorted(acc_views),
                           "stores": sorted(publish_records)})

        # Finalization only attaches records already implied by the consumed
        # stream.  It does not re-group or reconstruct the task sequence.
        for tile, ranges in tile_k_ranges.items():
            cursor = 0
            for k0, kt in ranges:
                if k0 != cursor or kt <= 0:
                    reject(RK_E_SHAPE,
                           "consumed K slices do not form an ordered partition",
                           stage="schedule",
                           actual={"tile": tile, "ranges": ranges})
                cursor += kt
            if cursor != k:
                reject(RK_E_SHAPE,
                       "consumed K slices do not cover the whole reduction",
                       stage="schedule",
                       actual={"tile": tile, "covered": cursor, "k": k})

        from dataclasses import replace as _replace_action
        from ..immutable import thaw as _thaw
        reference_ranges = next(iter(tile_k_ranges.values()), [])
        pipeline_hint = dict(
            (facts.get("hints") or {}).get("pipeline_hint") or {})
        pipeline_stages = int(pipeline_hint.get("num_stages", 0) or 0)
        loop_annotations = ({"num_stages": pipeline_stages}
                            if pipeline_stages > 0 else {})
        k_blocks = [{"index": index, "k0": k0, "extent": kt}
                    for index, (k0, kt) in enumerate(reference_ranges)]
        source_k_loop = {
            "source_k_block": int(facts.get("block_k", 256)),
            "source_k_trips": len(reference_ranges),
            "logical_k": k,
            "k_blocks": k_blocks,
            "source_rounding": facts.get("rounding"),
            "rounding_points": (
                "after_each_k_block"
                if facts.get("rounding") == "explicit_per_k_fp16"
                else "final_output_only"),
            "lowering_k_tile": 256,
            "source_loop_annotations": loop_annotations,
            "source_pipeline_stages": pipeline_stages,
            "pipeline_hint_dropped": bool(pipeline_stages),
            "pipeline_not_applied_reason": (
                "no software pipeline is implemented on this target: the K "
                "chain is emitted as an ordered accumulation of 256-wide "
                "tasks, so the request is recorded but not applied"
                if pipeline_stages else None),
        }
        if reference_ranges and                 reference_ranges[-1][1] < int(facts.get("block_k", 256)):
            source_k_loop["k_tail"] = reference_ranges[-1][1]
        grid_x = 1 + max((tile[1] // facts["tile_n"]
                          for tile in publish_records), default=0)
        grid_y = 1 + max((tile[0] // facts["tile_m"]
                          for tile in publish_records), default=0)
        ragged = any(tile[2] < facts["tile_m"] or
                     tile[3] < facts["tile_n"]
                     for tile in publish_records)
        source_mn_tiling = {
            "block_m": facts["tile_m"], "block_n": facts["tile_n"],
            "grid_x": grid_x, "grid_y": grid_y,
            "coverage": "ragged_tail" if ragged else "exact_once",
        }
        for tile, task_ids in mm_task_ids.items():
            for task_id in task_ids:
                action = self.actions[task_id]
                attrs = _thaw(action.attrs)
                attrs["k_count"] = len(task_ids)
                if compat_records:
                    attrs.pop("source_mn_tiling", None)
                    attrs.update({
                        "k_physical_extent": physical_kt,
                        "k_padding": max(0, physical_kt - logical_kt),
                        "source_k_loop": dict(source_k_loop),
                        "source_mn_tiling": dict(source_mn_tiling),
                    })
                self.actions[task_id] = _replace_action(action, attrs=attrs)
        for tile in deferred_publishes:
            materialize_publish(tile)

        spans_m = sorted({(tile[0], tile[2]) for tile in publish_records})
        spans_n = sorted({(tile[1], tile[3]) for tile in publish_records})
        for spans, logical, label in ((spans_m, m, "M"), (spans_n, n, "N")):
            cursor = 0
            for start, extent in spans:
                if start != cursor or extent <= 0:
                    reject(RK_E_SHAPE,
                           f"published {label} tiles do not partition output",
                           stage="schedule", actual=spans)
                cursor += extent
            if cursor != logical:
                reject(RK_E_SHAPE,
                       f"published {label} tiles do not cover output",
                       stage="schedule",
                       actual={"covered": cursor, "logical": logical})

        if legacy_linear and policy not in {"baseline", "scheduled_baseline"}:
            tile_order: list[tuple[int, int, int, int]] = []
            tile_rank: dict[tuple[int, int, int, int], int] = {}
            tile_meta: dict[tuple[int, int, int, int], dict[str, Any]] = {}
            for entry in linear_actions:
                tile = entry["tile"]
                if tile not in tile_rank:
                    tile_rank[tile] = len(tile_order)
                    tile_order.append(tile)
                tile_meta[tile] = entry
            m_values = sorted({tile[0] for tile in tile_order})
            n_values = sorted({tile[1] for tile in tile_order})
            k_count = max(meta["k_index"] for meta in tile_meta.values()) + 1
            ws_per_n = len(m_values) * k_count
            group_n = self.options.linear_group_n_tiles
            if policy in {"grouped_linear", "weight_stationary",
                          "weight_stationary_reuse"}:
                if group_n:
                    if group_n * ws_per_n > 64:
                        reject(RK_E_SHAPE,
                               "linear_group_n_tiles would exceed the 64-task "
                               "submit bound", stage="schedule",
                               actual={"n_tiles": group_n,
                                       "tasks_per_n_tile": ws_per_n},
                               required={"max_tasks_per_submit": 64})
                    n_per_group = group_n
                elif ws_per_n <= 64:
                    n_per_group = max(1, 64 // ws_per_n)
                else:
                    reject(RK_E_SHAPE,
                           "declared M/N tiling and this linear task policy "
                           "need more than one submit's 64-task bound",
                           stage="schedule",
                           actual={"tasks_per_n_tile": ws_per_n},
                           required={"max_tasks_per_submit": 64})
            else:
                n_per_group = max(1, len(n_values))
            if policy in {"merged", "interleaved", "interleaved_reuse"} and \
                    len(tile_order) * k_count > 64:
                reject(RK_E_SHAPE,
                       "declared M/N tiling and this linear task policy need "
                       "more than one submit's 64-task bound", stage="schedule",
                       actual={"policy": policy,
                               "output_tiles": len(tile_order),
                               "k_slices": k_count,
                               "tasks_per_submit": len(tile_order) * k_count},
                       required={"max_tasks_per_submit": 64})
            chunks = [n_values[start:start + n_per_group]
                      for start in range(0, len(n_values), n_per_group)]

            def tile_group(tile: tuple[int, int, int, int]) -> str:
                if policy in {"merged", "interleaved", "interleaved_reuse"}:
                    return "mk:linear"
                for index, chunk in enumerate(chunks):
                    if tile[1] in chunk:
                        return (f"mk:{'grp' if policy == 'grouped_linear' else 'ws'}:"
                                f"{index}")
                reject(RK_E_SCHEMA, "a tile has no policy submit group",
                       stage="schedule", actual=tile)

            def priority(entry: dict[str, Any]):
                tile = entry["tile"]
                rank = tile_rank[tile]
                child = entry["k_index"]
                group = tile_group(tile)
                if policy == "merged":
                    return (rank, child)
                if policy in {"interleaved", "interleaved_reuse"}:
                    return (child, rank)
                if policy == "grouped_linear":
                    return (group, tile[1], tile[0], child)
                return (group, child, tile[1], tile[0])

            action_by_id = {action.id: action for action in self.actions}
            by_action_id = {entry["action"]: entry for entry in linear_actions}
            npu_ids = set(by_action_id)
            children: dict[int, list[int]] = {aid: [] for aid in npu_ids}
            indegree: dict[int, int] = {aid: 0 for aid in npu_ids}
            for aid in npu_ids:
                for dep in action_by_id[aid].deps:
                    if dep in npu_ids:
                        children[dep].append(aid)
                        indegree[aid] += 1
            import heapq

            ready = [(priority(entry), entry["action"])
                     for entry in linear_actions if indegree[entry["action"]] == 0]
            heapq.heapify(ready)
            ordered_ids: list[int] = []
            while ready:
                _, aid = heapq.heappop(ready)
                ordered_ids.append(aid)
                for child in children[aid]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        heapq.heappush(ready, (priority(by_action_id[child]),
                                               child))
            if len(ordered_ids) != len(npu_ids):
                reject(RK_E_SCHEMA,
                       "the policy order cannot satisfy task dependencies",
                       stage="schedule", actual=len(ordered_ids),
                       required=len(npu_ids))
            attrs_by_id: dict[int, dict[str, Any]] = {}
            previous_by_group: dict[str, tuple[int | None, int | None]] = {}
            for aid in ordered_ids:
                action = action_by_id[aid]
                entry = by_action_id[aid]
                group = tile_group(entry["tile"])
                previous = previous_by_group.get(group)
                b_view = action.inputs[1] if len(action.inputs) > 1 else None
                weight_reuse = bool(
                    policy == "weight_stationary_reuse" and
                    previous is not None and previous[0] == b_view and
                    previous[1] == entry["k_index"])
                data_reuse = bool(
                    policy == "interleaved_reuse" and
                    tile_rank[entry["tile"]] != 0)
                previous_by_group[group] = (b_view, entry["k_index"])
                attrs = _thaw(action.attrs)
                attrs.update({
                    "submit_group": group,
                    "submit_recipe": "multi_k_acc_v1",
                    "linear_task_policy": policy,
                    "weight_reuse": weight_reuse,
                    "data_reuse": data_reuse,
                })
                attrs_by_id[aid] = attrs
            positions = [index for index, action in enumerate(self.actions)
                         if action.id in npu_ids]
            ordered_actions = [action_by_id[aid] for aid in ordered_ids]
            new_actions = list(self.actions)
            for position, action in zip(positions, ordered_actions):
                new_actions[position] = action
            mapping = {action.id: new_id
                       for new_id, action in enumerate(new_actions)}
            final_actions = []
            for new_id, action in enumerate(new_actions):
                attrs = attrs_by_id.get(action.id, action.attrs)
                final_actions.append(replace(
                    action, id=new_id,
                    deps=tuple(mapping[dep] for dep in action.deps),
                    attrs=attrs))
            self.actions = final_actions
            for buffer in self.buffers:
                buffer.first = len(final_actions) + 1
                buffer.last = -1
            for action in self.actions:
                for view_id in action.inputs + action.outputs:
                    self.touch(self.views[view_id].buffer_id, action.id)
            self.value_action = {
                value: mapping[action_id]
                for value, action_id in self.value_action.items()}

        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers, views=self.views,
            actions=self.actions, publish=self.publish)

    def build(self) -> DraftPlan:
        a_value = self.module.value(self.legalized.matmul_op.inputs[0])
        b_value = self.module.value(self.legalized.matmul_op.inputs[1])
        out_value = self.module.value(self.legalized.matmul_op.outputs[0])
        m, k = a_value.type.shape
        n = out_value.type.shape[1]
        b_layout = self.legalized.matmul_op.attrs.get("b_layout", "kn")
        a_binding = self.binding_by_value[a_value.id]
        b_binding = self.binding_by_value[b_value.id]
        b_is_constant = b_value.role == "constant"
        ktiles = k_slices(k)
        activation = self.legalized.activation_op.opcode \
            if self.legalized.activation_op is not None else None
        binary = self.legalized.binary_op
        self._assert_chain_edges(activation, binary)
        previous_task: int | None = None
        previous_group: str | None = None
        source_k_loop = self.legalized.matmul_op.attrs.get("source_k_loop")
        source_mn_tiling = self.legalized.matmul_op.attrs.get("source_mn_tiling")
        schedule = mn_schedule_from_attrs(self.legalized.matmul_op.attrs)
        schedule_dict = schedule.to_dict()
        # The source M/N tiling is an instruction for the supported
        # single-GEMM path.  K blocking and accumulation order remain fixed.
        tiles = output_tiles(m, n, tile_m=schedule.tile_m,
                             tile_n=schedule.tile_n, order=schedule.order)

        # A packed external slice is a pure function of (binding identity,
        # source region, source stride/transpose, destination layout).  Several
        # output tiles read the *same* A row block and K slice; packing it once
        # per tile wasted upload actions, BO slots and device bytes without
        # changing the math.  The cache is deliberately per-lowering-call: it
        # is only valid for one static execution/dataflow version, where every
        # upload reads the caller's binding once before any NPU submit.  There
        # is no cross-run content assumption and no Host-pointer identity test.
        #
        # Dynamic B is cached with the same rule inside this call.  This covers
        # the M-tiled attention plans whose K/V binding is repeated for every M
        # tile; immutable constants also get this dedup for free.
        x_tile_cache: dict[tuple[Any, ...], tuple[int, int, int]] = {}
        w_tile_cache: dict[tuple[Any, ...], tuple[int, int, int]] = {}

        for tile_index, tile in enumerate(tiles):
            rows, cols = tile.rows, tile.cols
            cube_rows, cube_cols, cube_bytes = cube_layout(rows, cols)

            x_views: list[int] = []
            w_views: list[int] = []
            upload_ids: list[int] = []
            for kslice in ktiles:
                # Keep the tile extent in the identity: a ragged tail K slice
                # has different physical columns and must never alias the full
                # 256-wide slice even if its source offset happened to match.
                x_key = (a_binding, tile.m0, rows, kslice.k0,
                         kslice.extent, k)
                cached = x_tile_cache.get(x_key)
                if cached is None:
                    pr, pc, pbytes = input_layout(rows, kslice.extent)
                    x_layout = TensorLayout("float16", CUBE_LAYOUT, rows,
                                            kslice.extent, pr, pc, 8, pbytes)
                    xb = self.add_buffer(role="input_slice", capacity=pbytes,
                                         layout_id=CUBE_LAYOUT, reuse=False,
                                         padding_state="zero")
                    xv = self.add_view(xb, 0, x_layout)
                    upload = self.add_action(
                        "upload", outputs=(xv,), origin="host_pack:input",
                        attrs={"source_binding": a_binding,
                               "src_row_offset": tile.m0,
                               "src_col_offset": kslice.k0,
                               "src_row_stride": k,
                               "transpose_source": False,
                               "cache": "input"})
                    cached = (xb, xv, upload)
                    x_tile_cache[x_key] = cached
                x_views.append(cached[1])
                upload_ids.append(cached[2])
            for kslice in ktiles:
                if b_layout == "kn":
                    src_row, src_col, stride = kslice.k0, tile.n0, n
                else:
                    src_row, src_col, stride = tile.n0, kslice.k0, k
                w_key = (b_binding, tile.n0, cols, kslice.k0,
                         kslice.extent, b_layout, stride)
                cached = w_tile_cache.get(w_key)
                if cached is None:
                    wr, wc, wbytes = weight_layout(cols, kslice.extent)
                    w_layout = TensorLayout("float16", WEIGHT_LAYOUT, cols,
                                            kslice.extent, wr, wc, 32, wbytes)
                    wb = self.add_buffer(role="weight", capacity=wbytes,
                                         layout_id=WEIGHT_LAYOUT, reuse=False,
                                         padding_state="zero")
                    wv = self.add_view(wb, 0, w_layout)
                    upload = self.add_action(
                        "upload", outputs=(wv,), origin="host_pack:weight",
                        attrs={"source_binding": b_binding,
                               "src_row_offset": src_row,
                               "src_col_offset": src_col,
                               "src_row_stride": stride,
                               "transpose_source": b_layout == "kn",
                               "cache": ("constant" if b_is_constant
                                         else "input")})
                    cached = (wb, wv, upload)
                    w_tile_cache[w_key] = cached
                w_views.append(cached[1])
                upload_ids.append(cached[2])
            # A tile can mention the same packed slice twice only if a future
            # K schedule revisits it; deps are sets at this boundary.
            upload_ids = list(dict.fromkeys(upload_ids))

            acc_buffers: list[int] = []
            acc_views: list[int] = []
            for _ in range(1 if len(ktiles) == 1 else 2):
                layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cube_rows, cube_cols, 8, cube_bytes)
                b = self.add_buffer(role="accumulator", capacity=cube_bytes,
                                    layout_id=CUBE_LAYOUT, padding_state="finite")
                acc_buffers.append(b)
                acc_views.append(self.add_view(b, 0, layout))

            group = f"mm:{tile.mi}:{tile.ni}"
            # Every host action that fills this tile's inputs is declared on the
            # first task, so later tasks in the tile inherit them transitively.
            tile_upload_ids: list[int] = list(upload_ids)
            mm_tasks: list[int] = []
            current = acc_views[0]
            prior_view: int | None = None
            for index, kslice in enumerate(ktiles):
                if index == 0:
                    action = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(x_views[index], w_views[index]),
                        outputs=(current,), deps=tuple(tile_upload_ids),
                        origin=f"lowering:matmul:{kslice.k0}",
                        op_ids=(self.legalized.matmul_op.id,),
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "k_physical_extent":
                                   ((kslice.extent + 31) // 32) * 32,
                               "k_padding":
                                   (((kslice.extent + 31) // 32) * 32
                                    - kslice.extent),
                               "source_k_loop": source_k_loop,
                               "source_mn_tiling": source_mn_tiling,
                               "source_k_rounding":
                                   self.legalized.matmul_op.attrs.get(
                                       "source_k_rounding"),
                               "source_schedule":
                                   self.legalized.matmul_op.attrs.get(
                                       "source_schedule"),
                               "applied_schedule": schedule_dict})
                else:
                    other = acc_views[index & 1]
                    action = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(x_views[index], w_views[index], prior_view),
                        outputs=(other,), deps=(mm_tasks[-1],),
                        origin=f"lowering:matmul_acc:{kslice.k0}",
                        op_ids=(self.legalized.matmul_op.id,),
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "k_physical_extent":
                                   ((kslice.extent + 31) // 32) * 32,
                               "k_padding":
                                   (((kslice.extent + 31) // 32) * 32
                                    - kslice.extent),
                               "source_k_loop": source_k_loop,
                               "source_mn_tiling": source_mn_tiling,
                               "source_k_rounding":
                                   self.legalized.matmul_op.attrs.get(
                                       "source_k_rounding"),
                               "source_schedule":
                                   self.legalized.matmul_op.attrs.get(
                                       "source_schedule"),
                               "applied_schedule": schedule_dict})
                    current = other
                mm_tasks.append(action)
                prior_view = current

            # MatMul result view is the final ping-pong view.  Reuse its buffer for
            # activation scratch only after its last read.
            final_mm_view = current
            final_mm_buffer = self.views[final_mm_view].buffer_id
            final_task: int = mm_tasks[-1]
            act_group: str | None = None
            if activation is None:
                self.buffers[final_mm_buffer].role = "output"
                final_view = final_mm_view
            else:
                act_group = f"act:{tile.mi}:{tile.ni}"
                final_view = self._emit_activation(
                    activation=activation, rows=rows, cols=cols,
                    cube_rows=cube_rows, cube_cols=cube_cols,
                    input_view=final_mm_view,
                    input_buffer=final_mm_buffer,
                    deps=(mm_tasks[-1],), group=act_group,
                    op_ids=(self.legalized.activation_op.id,))
                self.buffers[self.views[final_view].buffer_id].role = "output"
                final_task = self._last_task_of_group(act_group)

            if binary is not None:
                final_view = self._emit_binary(
                    binary=binary, tile=tile, rows=rows, cols=cols,
                    cube_rows=cube_rows, cube_cols=cube_cols,
                    input_view=final_view, deps=(final_task,),
                    group=f"ew:{tile.mi}:{tile.ni}", op_ids=(binary.id,))
                self.buffers[self.views[final_view].buffer_id].role = "output"
                final_task = self._last_task_of_group(f"ew:{tile.mi}:{tile.ni}")

            out_value_id = self.module.outputs[0] if (activation is not None or
                                                     binary is not None) \
                else out_value.id
            binding_id = self.binding_by_value[out_value_id]
            # Keep output alive through the download action.  The download action
            # is also a hard Host boundary and makes liveness explicit.
            download = self.add_action(
                "download", inputs=(final_view,), deps=(final_task,),
                origin="host:download",
                attrs={"binding_id": binding_id, "submit_group": None})
            self.buffers[self.views[final_view].buffer_id].last = max(
                self.buffers[self.views[final_view].buffer_id].last, download)
            # Multi-tile outputs publish one placement rectangle per tile; the
            # C runtime concatenates them into the row-major output binding.
            self.publish.append(Publish(
                self.bindings[binding_id].name, binding_id, final_view,
                dst_row=tile.m0, dst_col=tile.n0, rows=rows, cols=cols))

            # Serialize output tiles conservatively.  This removes any dependence
            # on slot reuse across independently submitted jobs.
            if previous_group is not None:
                # schedule.py reads this dependency list; the first activation or
                # first MM task of this tile depends on the previous tile's last
                # group.
                for action_id in (mm_tasks[0],):
                    action = self.actions[action_id]
                    self.actions[action_id] = replace(
                        action, deps=action.deps + (previous_task,))
            if act_group is not None:
                # record predecessor group for schedule; actual task deps already
                # include mm_tasks[-1] and upload actions.
                pass
            previous_group = act_group or group
            # Last NPU action in this tile is a stable predecessor marker.
            previous_task = mm_tasks[-1] if act_group is None else None
            if act_group is not None:
                # Find last task with this group.
                for action in reversed(self.actions):
                    if action.kind == "npu_task" and \
                            action.attrs.get("submit_group") == act_group:
                        previous_task = action.id
                        break

        return DraftPlan(
            name=self.module.name, profile_hash=self.profile.profile_hash,
            source_hash=self.module.source_hash, bindings=self.bindings,
            constants=self.constants, buffers=self.buffers,
            views=self.views, actions=self.actions, publish=self.publish)

    def _assert_chain_edges(self, activation, binary) -> None:
        """Re-check the graph edges the lowering is about to linearise.

        Legalization already rejects broken chains; this repeats the check where
        the values are actually consumed so a future pass cannot reorder the graph
        without the lowering noticing.
        """
        produced = self.legalized.matmul_op.outputs[0]
        if activation is not None:
            if tuple(self.legalized.activation_op.inputs) != (produced,):
                reject(RK_E_SHAPE,
                       "activation no longer consumes the matmul result",
                       stage="dag")
            produced = self.legalized.activation_op.outputs[0]
        if binary is not None:
            if produced not in binary.inputs:
                reject(RK_E_SHAPE,
                       "add/mul no longer consumes the produced value",
                       stage="dag")
            operand = self.legalized.binary_operand_id
            if operand not in binary.inputs:
                reject(RK_E_SHAPE,
                       "resolved binary operand is not an input of the op",
                       stage="dag")
            produced = binary.outputs[0]
        if tuple(self.module.outputs) != (produced,):
            reject(RK_E_SHAPE, "module output is not the produced value",
                   stage="dag")

    def _last_upload_of(self, view_id: int) -> int | None:
        """Return the upload action that filled ``view_id``, if any."""
        for action in reversed(self.actions):
            if action.kind == "upload" and view_id in action.outputs:
                return action.id
        return None

    def _last_task_of_group(self, group: str) -> int:
        for action in reversed(self.actions):
            if action.kind == "npu_task" and \
                    action.attrs.get("submit_group") == group:
                return action.id
        reject(RK_E_SHAPE, f"task group {group!r} has no NPU task", stage="dag")

    def _emit_binary(self, *, binary, tile, rows: int, cols: int,
                     cube_rows: int, cube_cols: int, input_view: int,
                     deps: tuple[int, ...], group: str,
                     op_ids: tuple[int, ...] = ()) -> int:
        """Emit one same-shape add/mul against a host-supplied tensor tile.

        The operand is the one *legalization* resolved from the real graph edges,
        not a fixed operand position: ``add``/``mul`` are commutative, so the
        produced value may appear on either side.
        """
        operand_id = self.legalized.binary_operand_id
        if operand_id is None:
            reject(RK_E_SHAPE, "legalization did not resolve the binary operand",
                   stage="dag", action_id=None)
        if operand_id in binary.inputs and operand_id == binary.outputs[0]:
            reject(RK_E_SHAPE, "binary operand cannot be its own result",
                   stage="dag")
        operand_value = self.module.value(operand_id)
        binding_id = self.binding_by_value[operand_value.id]
        n = self.module.value(self.legalized.matmul_op.outputs[0]).type.shape[1]
        layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, cube_rows,
                              cube_cols, 8, cube_rows * cube_cols * 2)
        operand_buffer = self.add_buffer(role="input_slice",
                                         capacity=layout.bytes,
                                         layout_id=CUBE_LAYOUT, reuse=False,
                                         padding_state="zero")
        operand_view = self.add_view(operand_buffer, 0, layout)
        upload = self.add_action(
            "upload", outputs=(operand_view,), origin="host_pack:binary_operand",
            attrs={"source_binding": binding_id,
                   "src_row_offset": tile.m0, "src_col_offset": tile.n0,
                   "src_row_stride": n, "transpose_source": False,
                   "cache": "constant" if operand_value.role == "constant"
                            else "input"})
        # The identity weight upload is a host action and must precede the group.
        _, identity_view = self.identity_for(cube_cols)
        identity_upload = self._last_upload_of(identity_view)
        out_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, cube_rows,
                                  cube_cols, 8, cube_rows * cube_cols * 2)
        out_buffer = self.add_buffer(role="output", capacity=out_layout.bytes,
                                     layout_id=CUBE_LAYOUT, padding_state="finite")
        out_view = self.add_view(out_buffer, 0, out_layout)
        template = ("co.ew_add.f16.v1" if binary.opcode == "add"
                    else "co.ew_mul.f16.v1")
        self.add_action("npu_task", template_id=template,
                        inputs=(input_view, operand_view, identity_view),
                        outputs=(out_view,),
                        deps=deps + (upload,) +
                             ((identity_upload,) if identity_upload is not None else ()),
                        origin=f"lowering:{binary.opcode}",
                        attrs={"submit_group": group}, op_ids=op_ids)
        return out_view

    def _emit_activation(self, *, activation: str, rows: int, cols: int,
                         cube_rows: int, cube_cols: int, input_view: int,
                         input_buffer: int, deps: tuple[int, ...],
                         group: str, op_ids: tuple[int, ...] = (),
                         extra_attrs: dict[str, Any] | None = None) -> int:
        if activation == "relu":
            _, zero_view = self.zero_for(rows, cols)
            zero_upload = self._last_upload_of(zero_view)
            _, identity_view = self.identity_for(cube_cols)
            identity_upload = self._last_upload_of(identity_view)
            out_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cube_rows, cube_cols, 8,
                                      cube_rows * cube_cols * 2)
            out_buffer = self.add_buffer(role="output", capacity=out_layout.bytes,
                                         layout_id=CUBE_LAYOUT,
                                         padding_state="finite", reuse=False)
            out_view = self.add_view(out_buffer, 0, out_layout)
            deps = tuple(d for d in (deps[0] if deps else None, zero_upload,
                                      identity_upload) if d is not None)
            self.add_action(
                "npu_task", template_id="co.ew_max.f16.v1",
                inputs=(input_view, zero_view, identity_view),
                outputs=(out_view,), deps=deps,
                origin=f"lowering:{activation}",
                attrs={"submit_group": group, **(extra_attrs or {})}, op_ids=op_ids)
            return out_view

        chunks = lut_chunks(cube_rows * cube_cols)
        requires_gate = activation in {"silu", "gelu"}
        if requires_gate:
            # The identity weight upload is a host action, so it must exist
            # before the task group starts: a host action inside a submit group
            # is a hard boundary and would make the group non-contiguous.
            _, identity_view = self.identity_for(cube_cols)
            identity_upload = self._last_upload_of(identity_view)
            gate_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                       cube_rows, cube_cols, 8,
                                       cube_rows * cube_cols * 2)
            gate_buffer = self.add_buffer(role="gate",
                                          capacity=gate_layout.bytes,
                                          layout_id=CUBE_LAYOUT,
                                          padding_state="finite", reuse=False)
            gate_cube_view = self.add_view(gate_buffer, 0, gate_layout)
            gate_linear_views: list[int] = []
            input_linear_views: list[int] = []
            for offset, count in chunks:
                linear = TensorLayout("float16", LINEAR_LAYOUT, 1, count, 1,
                                      count, 1, count * 2)
                byte_offset = offset * 2  # lut_chunks counts FP16 elements
                input_linear_views.append(
                    self.add_view(input_buffer,
                                  self.views[input_view].offset_bytes + byte_offset,
                                  linear, (0, 1, 0, count)))
                gate_linear_views.append(
                    self.add_view(gate_buffer, byte_offset, linear,
                                  (0, 1, 0, count)))
            lut_tasks: list[int] = []
            for index, (src_view, dst_view) in enumerate(
                    zip(input_linear_views, gate_linear_views)):
                task_deps = deps if index == 0 else (lut_tasks[-1],)
                lut_tasks.append(self.add_action(
                    "npu_task",
                    template_id=("co.lut.sigmoid.f16.v1" if activation == "silu"
                                 else "co.lut.gelu_gate.f16.v1"),
                    inputs=(src_view,), outputs=(dst_view,), deps=task_deps,
                    origin=f"lowering:{activation}:lut:{index}",
                    attrs={"submit_group": group, "chunk_index": index, **(extra_attrs or {})},
                    op_ids=op_ids))
            out_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                      cube_rows, cube_cols, 8,
                                      cube_rows * cube_cols * 2)
            out_buffer = self.add_buffer(role="output", capacity=out_layout.bytes,
                                         layout_id=CUBE_LAYOUT,
                                         padding_state="finite")
            out_view = self.add_view(out_buffer, 0, out_layout)
            self.add_action(
                "npu_task", template_id="co.ew_mul.f16.v1",
                inputs=(input_view, gate_cube_view, identity_view),
                outputs=(out_view,),
                deps=(lut_tasks[-1],) + ((identity_upload,)
                                         if identity_upload is not None else ()),
                origin=f"lowering:{activation}:mul",
                attrs={"submit_group": group, **(extra_attrs or {})}, op_ids=op_ids)
            return out_view

        # Direct sigmoid/HardSigmoid: chunked LUT writes are the final result.
        out_layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols,
                                  cube_rows, cube_cols, 8,
                                  cube_rows * cube_cols * 2)
        out_buffer = self.add_buffer(role="output", capacity=out_layout.bytes,
                                     layout_id=CUBE_LAYOUT,
                                     padding_state="finite")
        out_cube_view = self.add_view(out_buffer, 0, out_layout)
        input_linear_views: list[int] = []
        out_linear_views: list[int] = []
        for offset, count in chunks:
            linear = TensorLayout("float16", LINEAR_LAYOUT, 1, count, 1, count,
                                  1, count * 2)
            byte_offset = offset * 2
            input_linear_views.append(
                self.add_view(input_buffer,
                              self.views[input_view].offset_bytes + byte_offset,
                              linear, (0, 1, 0, count)))
            out_linear_views.append(
                self.add_view(out_buffer, byte_offset, linear, (0, 1, 0, count)))
        lut_tasks: list[int] = []
        template = ("co.lut.hardsigmoid.f16.v1"
                    if activation == "hardsigmoid" else
                    ("co.lut.rsqrt.f16.v1" if activation == "rsqrt"
                     else "co.lut.sigmoid.f16.v1"))
        for index, (src_view, dst_view) in enumerate(
                zip(input_linear_views, out_linear_views)):
            task_deps = deps if index == 0 else (lut_tasks[-1],)
            lut_tasks.append(self.add_action(
                "npu_task", template_id=template, inputs=(src_view,),
                outputs=(dst_view,), deps=task_deps,
                origin=f"lowering:{activation}:lut:{index}",
                attrs={"submit_group": group, "chunk_index": index, **(extra_attrs or {})},
                op_ids=op_ids))
        return out_cube_view





# Ops whose effect the plan carries as a view/layout restructuring instead of
# a device task.  Stated by opcode, never inferred from "no task found".
AUXILIARY_OPCODES = frozenset({"view"})


def attach_plan_records(module: Module, draft: DraftPlan,
                        options: CompileOptions) -> DraftPlan:
    """Record the logical graph and every NumericContract in the plan.

    Attribution is structural: a lowering records the logical ops it implements
    when it emits the task (Action.source_ops), so nothing here parses an origin
    string.  This runs after submit scheduling because upload hoisting renumbers
    action ids -- the ids written here are the final ones.  origin stays as
    diagnostic text only.
    """
    from dataclasses import replace as _replace

    by_op: dict[int, list[int]] = {}
    for action in draft.actions:
        if action.kind != "npu_task":
            continue
        for op_id in action.source_ops:
            by_op.setdefault(op_id, []).append(action.id)
    known = {op.id for op in module.ops}
    if stray := sorted(set(by_op) - known):
        reject(RK_E_SCHEMA,
               "a device task names a logical op the module does not contain",
               stage="dag", actual=stray)

    logical_ops = []
    requirements = []
    for op in module.ops:
        tasks = tuple(sorted(by_op.get(op.id, ())))
        inputs = tuple(module.value(value_id).name for value_id in op.inputs)
        outputs = tuple(module.value(value_id).name for value_id in op.outputs)
        contract = op.attrs.get("numeric_contract")
        logical_ops.append(LogicalOp(
            index=op.id, opcode=op.opcode, inputs=inputs, outputs=outputs,
            contract=dict(contract) if contract is not None else None,
            tasks=tasks, auxiliary=op.opcode in AUXILIARY_OPCODES))
        if contract is None:
            continue
        emitters = [draft.actions[task] for task in tasks]
        # Whether this op's implementation uses a lookup-table approximation
        # that the compile admitted.  The contract cannot state this:
        # approximate-ness is a compile-time admission (allow_approx_lut), the
        # contract is what the source asked for.
        uses_lut = any((action.template_id or "").startswith("co.lut.")
                       for action in emitters)
        k_chain = sorted({action.attrs.get("k0") for action in emitters
                          if action.attrs.get("k0") is not None})
        k_extents = sorted({action.attrs.get("kt") for action in emitters
                            if action.attrs.get("kt") is not None})
        requirements.append(NumericRequirement(
            op_index=op.id, opcode=op.opcode, contract=dict(contract),
            inputs=inputs, outputs=outputs, tasks=tasks,
            execution={
                "k_chain": k_chain,
                "k_tile": (k_extents[0] if len(k_extents) == 1 else
                           (max(k_extents) if k_extents else 0)),
                "templates": sorted({action.template_id
                                     for action in emitters}),
                "output_dtypes": sorted({draft.views[action.outputs[0]]
                                             .tensor.dtype
                                         for action in emitters}),
                "attributed": bool(tasks),
                "approximated": bool(uses_lut and
                                     options.allow_approx_lut)}))
    return _replace(draft, numeric_requirements=tuple(requirements),
                    logical_ops=tuple(logical_ops))


def build_draft(module: Module, legalized: LegalizedGraph, *,
                profile: TargetProfile,
                options: CompileOptions) -> DraftPlan:
    if options.max_ir_nodes < len(module.values) + len(module.ops):
        reject(RK_E_BUDGET, "max_ir_nodes exceeded", stage="dag",
               actual=len(module.values) + len(module.ops),
               required=options.max_ir_nodes)
    builder = _Builder(module, legalized, profile=profile, options=options)
    scheduled: list[Op] = []
    two_stage = False
    ffn_scheduled = False
    if legalized.graph_ops is not None:
        scheduled = [op for op in module.ops
                     if op.opcode == "matmul" and
                     "source_schedule" in op.attrs]
        opcodes = [op.opcode for op in module.ops]
        two_stage = (
            len(module.ops) == 2 and
            all(op.opcode == "matmul" for op in module.ops) and
            module.ops[1].inputs[0] == module.ops[0].outputs[0] and
            module.value(module.ops[0].outputs[0]).role == "temporary"
        )
        ffn_scheduled = opcodes == [
            "matmul", "view", "view", "sigmoid", "mul", "mul",
            "matmul", "add"]
        if scheduled and not ((two_stage or ffn_scheduled) and
                             len(scheduled) == 2):
            first = scheduled[0]
            if first.attrs.get("source_schedule_origin") == "derived":
                reject(RK_E_UNSUPPORTED,
                       "this source proves M/N blocks that no admitted "
                       "multi-node contract can honour: only the restricted "
                       "two-stage X->W1->Z->W2->Y graph consumes a schedule.  "
                       "Annotate a schedule the supported contract admits, or "
                       "write the kernels on the fixed 256x256 grid",
                       stage="schedule", op=first.opcode, source=first.source,
                       actual=first.attrs.get("source_schedule"))
            reject(RK_E_UNSUPPORTED,
                   "explicit M/N scheduling is admitted only for the "
                   "restricted two-stage X->W1->Z->W2->Y contract",
                   stage="schedule", op=first.opcode, source=first.source)
    if (options.linear_task_order != "baseline" and
            len(module.ops) == 1 and module.ops[0].opcode == "matmul"):
        draft = builder.build_scheduled_linear()
    elif ffn_scheduled and len(scheduled) == 2:
        draft = builder.build_scheduled_swiglu_ffn()
    elif two_stage and len(scheduled) == 2:
        draft = builder.build_scheduled_two_stage()
    elif legalized.swiglu is not None:
        draft = builder.build_swiglu()
    elif legalized.graph_ops is not None:
        tiled = any(v.type.rank == 2 and
                    (v.type.shape[0] > 256 or v.type.shape[1] > 256)
                    for v in module.values)
        tiled = tiled or any(
            op.opcode == "matmul" and
            module.value(op.inputs[0]).type.cols > 256
            for op in module.ops)
        draft = (builder.build_dataflow_tiled() if tiled
                 else builder.build_dataflow())
    else:
        draft = builder.build()
    if len(draft.actions) > options.max_actions:
        reject(RK_E_BUDGET, "max_actions exceeded", stage="dag",
               actual=len(draft.actions), required=options.max_actions)
    if legalized.matmul_op is not None:
        if len(ktiles := k_slices(
                module.value(legalized.matmul_op.inputs[0]).type.cols)) > 64:
            reject(RK_E_SHAPE,
                   "K tile chain cannot exceed one submit's 64-task limit",
                   stage="dag")
    return draft


def build_from_tasks(module: Module, *, tasks: list[dict[str, Any]],
                     facts: dict[str, Any],
                     legalized: LegalizedGraph,
                     profile: TargetProfile,
                     options: CompileOptions) -> DraftPlan:
    """Plan from the instantiated target operations, not a shape summary.

    ``tasks`` is the ordered ``rk_fill``/``rk_copy``/``rk_gemm``/``rk_store``
    sequence read back from TIR.  The synthetic matmul in ``module`` is only a
    binding and NumericContract carrier; this function does not use its shape to
    regenerate tiles, K slices or task order.
    """
    builder = _Builder(module, legalized, profile=profile,
                       options=options)
    if facts.get("staged_stages"):
        draft = builder.build_instantiated_staged(tasks=tasks, facts=facts)
    else:
        draft = builder.build_instantiated_linear(tasks=tasks, facts=facts)
    if len(draft.actions) > options.max_actions:
        reject(RK_E_BUDGET, "max_actions exceeded", stage="dag",
               actual=len(draft.actions), required=options.max_actions)
    return draft
