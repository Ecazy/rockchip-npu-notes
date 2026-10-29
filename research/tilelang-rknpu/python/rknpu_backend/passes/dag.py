# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the device task DAG and logical views for MatMul [K-chain] -> activation."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, replace

from ..capabilities import TargetProfile
from ..errors import RK_E_BUDGET, RK_E_SCHEMA, RK_E_SHAPE, reject
from ..ir import Module, NumericContract
from ..options import CompileOptions
from ..plan import (Action, Binding, ConstantSpec, Publish, TensorLayout, View)
from ..serialize import sha256_json
from .legalize import LegalizedGraph
from .tile import K_TILE, TILE_LIMIT, cube_layout, input_layout, k_slices, \
    lut_chunks, output_tiles, round_up, weight_layout

CUBE_LAYOUT = "co.cube.f16.v1"
WEIGHT_LAYOUT = "co.weight.f16.v1"
LINEAR_LAYOUT = "co.linear.f16.v1"


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
                   attrs: dict | None = None) -> int:
        aid = len(self.actions)
        self.actions.append(Action(aid, kind, template_id, inputs, outputs,
                                   deps, origin, attrs or {}))
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
                           reuse: bool = True) -> int:
        _, identity_view = self.identity_for(cube_cols)
        identity_upload = self._last_upload_of(identity_view)
        layout = TensorLayout("float16", CUBE_LAYOUT, rows, cols, cube_rows,
                              cube_cols, 8, cube_rows * cube_cols * 2)
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
            attrs={"submit_group": group})
        return out_view

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
                    deps=self._deps_for(op.inputs), group=group)
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
                    reuse=relu_out not in self.module.outputs)
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
                    group=group)
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
                        deps=(in_action,), group=group)
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
                        reuse=relu_out_id not in self.module.outputs)
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
                        deps=unique(operand_actions), group=group)
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
                               attrs={"submit_group": "swiglu:gate",
                                      "attrs_note": notes})
        _, up_view = cube_tile("accumulator", m, h, hid_phys)
        up = self.add_action("npu_task", template_id="co.mm.f16.v1",
                             inputs=(x_view, wu_view), outputs=(up_view,),
                             deps=uploads, origin="lowering:swiglu:up",
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
            attrs={"submit_group": "swiglu:silu"})
        _, gated_view = cube_tile("accumulator", m, h, hid_phys)
        gated = self.add_action(
            "npu_task", template_id="co.ew_mul.f16.v1",
            inputs=(act_view, up_view, identity_view), outputs=(gated_view,),
            deps=(act, up) + ((identity_upload,) if identity_upload is not None
                              else ()),
            origin="lowering:swiglu:up_mul",
            attrs={"submit_group": "swiglu:mul"})

        _, proj_view = cube_tile("accumulator", m, d_out, out_phys)
        proj = self.add_action("npu_task", template_id="co.mm.f16.v1",
                               inputs=(gated_view, wd_view),
                               outputs=(proj_view,), deps=(gated, wd_upload),
                               origin="lowering:swiglu:projection",
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
            origin="lowering:swiglu:residual",
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
        tiles = output_tiles(m, n)
        activation = self.legalized.activation_op.opcode \
            if self.legalized.activation_op is not None else None
        binary = self.legalized.binary_op
        self._assert_chain_edges(activation, binary)
        previous_task: int | None = None
        previous_group: str | None = None
        source_k_loop = self.legalized.matmul_op.attrs.get("source_k_loop")
        source_mn_tiling = self.legalized.matmul_op.attrs.get("source_mn_tiling")

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
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "source_k_loop": source_k_loop,
                               "source_mn_tiling": source_mn_tiling})
                else:
                    other = acc_views[index & 1]
                    action = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(x_views[index], w_views[index], prior_view),
                        outputs=(other,), deps=(mm_tasks[-1],),
                        origin=f"lowering:matmul_acc:{kslice.k0}",
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent,
                               "source_k_loop": source_k_loop,
                               "source_mn_tiling": source_mn_tiling})
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
                    deps=(mm_tasks[-1],), group=act_group)
                self.buffers[self.views[final_view].buffer_id].role = "output"
                final_task = self._last_task_of_group(act_group)

            if binary is not None:
                final_view = self._emit_binary(
                    binary=binary, tile=tile, rows=rows, cols=cols,
                    cube_rows=cube_rows, cube_cols=cube_cols,
                    input_view=final_view, deps=(final_task,),
                    group=f"ew:{tile.mi}:{tile.ni}")
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
                     deps: tuple[int, ...], group: str) -> int:
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
                        attrs={"submit_group": group})
        return out_view

    def _emit_activation(self, *, activation: str, rows: int, cols: int,
                         cube_rows: int, cube_cols: int, input_view: int,
                         input_buffer: int, deps: tuple[int, ...],
                         group: str) -> int:
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
                attrs={"submit_group": group})
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
                    attrs={"submit_group": group, "chunk_index": index}))
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
                attrs={"submit_group": group})
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
                attrs={"submit_group": group, "chunk_index": index}))
        return out_cube_view



def build_draft(module: Module, legalized: LegalizedGraph, *,
                profile: TargetProfile,
                options: CompileOptions) -> DraftPlan:
    if options.max_ir_nodes < len(module.values) + len(module.ops):
        reject(RK_E_BUDGET, "max_ir_nodes exceeded", stage="dag",
               actual=len(module.values) + len(module.ops),
               required=options.max_ir_nodes)
    builder = _Builder(module, legalized, profile=profile, options=options)
    if legalized.swiglu is not None:
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
