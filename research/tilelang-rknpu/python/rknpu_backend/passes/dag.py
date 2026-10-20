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
from .tile import K_TILE, cube_layout, input_layout, k_slices, lut_chunks, \
    output_tiles, weight_layout

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
        self._create_bindings()

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
                self.constants.append(ConstantSpec(
                    name=value.name, binding_id=bid, dtype=value.type.dtype,
                    shape=shape, nbytes=shape[0] * shape[1] * 2,
                    sha256=version,
                    pack_key=f"f16:host-contract:{shape[0]}x{shape[1]}:v1",
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
        previous_task: int | None = None
        previous_group: str | None = None

        for tile_index, tile in enumerate(tiles):
            rows, cols = tile.rows, tile.cols
            cube_rows, cube_cols, cube_bytes = cube_layout(rows, cols)

            x_buffers: list[int] = []
            x_views: list[int] = []
            w_buffers: list[int] = []
            w_views: list[int] = []
            upload_ids: list[int] = []
            for kslice in ktiles:
                pr, pc, pbytes = input_layout(rows, kslice.extent)
                x_layout = TensorLayout("float16", CUBE_LAYOUT, rows,
                                        kslice.extent, pr, pc, 8, pbytes)
                xb = self.add_buffer(role="input_slice", capacity=pbytes,
                                     layout_id=CUBE_LAYOUT, reuse=False,
                                     padding_state="zero")
                xv = self.add_view(xb, 0, x_layout)
                x_buffers.append(xb)
                x_views.append(xv)
                upload_ids.append(self.add_action(
                    "upload", outputs=(xv,), origin="host_pack:input",
                    attrs={"source_binding": a_binding,
                           "src_row_offset": tile.m0,
                           "src_col_offset": kslice.k0,
                           "src_row_stride": k,
                           "transpose_source": False,
                           "cache": "input"}))

                wr, wc, wbytes = weight_layout(cols, kslice.extent)
                w_layout = TensorLayout("float16", WEIGHT_LAYOUT, cols,
                                        kslice.extent, wr, wc, 32, wbytes)
                wb = self.add_buffer(role="weight", capacity=wbytes,
                                     layout_id=WEIGHT_LAYOUT, reuse=False,
                                     padding_state="zero")
                wv = self.add_view(wb, 0, w_layout)
                w_buffers.append(wb)
                w_views.append(wv)
                if b_layout == "kn":
                    src_row, src_col, stride = kslice.k0, tile.n0, n
                else:
                    src_row, src_col, stride = tile.n0, kslice.k0, k
                upload_ids.append(self.add_action(
                    "upload", outputs=(wv,), origin="host_pack:weight",
                    attrs={"source_binding": b_binding,
                           "src_row_offset": src_row,
                           "src_col_offset": src_col,
                           "src_row_stride": stride,
                           "transpose_source": b_layout == "kn",
                           "cache": "constant" if b_is_constant else "input"}))

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
            mm_tasks: list[int] = []
            current = acc_views[0]
            prior_view: int | None = None
            for index, kslice in enumerate(ktiles):
                if index == 0:
                    action = self.add_action(
                        "npu_task", template_id="co.mm.f16.v1",
                        inputs=(x_views[index], w_views[index]),
                        outputs=(current,), deps=tuple(upload_ids),
                        origin=f"lowering:matmul:{kslice.k0}",
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent})
                else:
                    other = acc_views[index & 1]
                    action = self.add_action(
                        "npu_task", template_id="co.mm_acc.f16.v1",
                        inputs=(x_views[index], w_views[index], prior_view),
                        outputs=(other,), deps=(mm_tasks[-1],),
                        origin=f"lowering:matmul_acc:{kslice.k0}",
                        attrs={"submit_group": group,
                               "k0": kslice.k0, "kt": kslice.extent})
                    current = other
                mm_tasks.append(action)
                prior_view = current

            # MatMul result view is the final ping-pong view.  Reuse its buffer for
            # activation scratch only after its last read.
            final_mm_view = current
            final_mm_buffer = self.views[final_mm_view].buffer_id
            if activation is None:
                self.buffers[final_mm_buffer].role = "output"
                final_view = final_mm_view
                act_group = None
            else:
                act_group = f"act:{tile.mi}:{tile.ni}"
                final_view = self._emit_activation(
                    activation=activation, rows=rows, cols=cols,
                    cube_rows=cube_rows, cube_cols=cube_cols,
                    input_view=final_mm_view,
                    input_buffer=final_mm_buffer,
                    deps=(mm_tasks[-1],), group=act_group)
                self.buffers[self.views[final_view].buffer_id].role = "output"

            out_value_id = self.module.outputs[0] if activation is not None \
                else out_value.id
            binding_id = self.binding_by_value[out_value_id]
            # Keep output alive through the download action.  The download action
            # is also a hard Host boundary and makes liveness explicit.
            download = self.add_action(
                "download", inputs=(final_view,),
                origin="host:download",
                attrs={"binding_id": binding_id, "submit_group": None})
            self.buffers[self.views[final_view].buffer_id].last = max(
                self.buffers[self.views[final_view].buffer_id].last, download)
            self.publish.append(Publish(self.bindings[binding_id].name,
                                        binding_id, final_view))

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

    def _emit_activation(self, *, activation: str, rows: int, cols: int,
                         cube_rows: int, cube_cols: int, input_view: int,
                         input_buffer: int, deps: tuple[int, ...],
                         group: str) -> int:
        chunks = lut_chunks(cube_rows * cube_cols)
        requires_gate = activation in {"silu", "gelu"}
        if requires_gate:
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
                input_linear_views.append(
                    self.add_view(input_buffer, offset, linear, (0, 1, 0, count)))
                gate_linear_views.append(
                    self.add_view(gate_buffer, offset, linear, (0, 1, 0, count)))
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
            _, identity_view = self.identity_for(cube_cols)
            self.add_action(
                "npu_task", template_id="co.ew_mul.f16.v1",
                inputs=(input_view, gate_cube_view, identity_view),
                outputs=(out_view,), deps=(lut_tasks[-1],),
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
            input_linear_views.append(
                self.add_view(input_buffer, offset, linear, (0, 1, 0, count)))
            out_linear_views.append(
                self.add_view(out_buffer, offset, linear, (0, 1, 0, count)))
        lut_tasks: list[int] = []
        template = ("co.lut.hardsigmoid.f16.v1"
                    if activation == "hardsigmoid"
                    else "co.lut.sigmoid.f16.v1")
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
    draft = builder.build()
    if len(draft.actions) > options.max_actions:
        reject(RK_E_BUDGET, "max_actions exceeded", stage="dag",
               actual=len(draft.actions), required=options.max_actions)
    if len(ktiles := k_slices(module.value(legalized.matmul_op.inputs[0]).type.cols)) \
            > 64:
        reject(RK_E_SHAPE, "K tile chain cannot exceed one submit's 64-task limit",
               stage="dag")
    return draft
