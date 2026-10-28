# SPDX-License-Identifier: GPL-3.0-or-later
"""The single RKPlan descriptor encoding used by the whole backend.

This module is the *encoding* half of the codegen boundary: it turns one
already-planned, already-verified :class:`~rknpu_backend.plan.Plan` into the
``rk_plan.h`` ctypes descriptor tree.  It performs Host-memory work only -- it
never imports the runtime library, never opens a device and never allocates a BO.

Everything here is a pure projection of the authoritative plan: the C side
re-validates and deep-copies the descriptor, so an encoding bug can only produce
a rejection, never a silently different execution.
"""
from __future__ import annotations

import ctypes
from typing import Any, Iterable

from ..errors import RK_E_UNSUPPORTED, reject
from ..plan import Plan


RK_ABI_VERSION = 1
RK_SCHEMA_VERSION = 2

RK_ROLE_INPUT = 0
RK_ROLE_CONSTANT = 1
RK_ROLE_OUTPUT = 2
RK_ROLE_SCRATCH = 3
RK_ROLE_INPUT_SLICE = 4
RK_ROLE_WEIGHT = 5
RK_ROLE_ACCUMULATOR = 6
RK_ROLE_GATE = 7
RK_ROLE_IDENTITY = 8
RK_ROLE_STATE = 9

RK_DTYPE_F16 = 0
RK_LAYOUT_CUBE_F16 = 1
RK_LAYOUT_WEIGHT_F16 = 2
RK_LAYOUT_LINEAR_F16 = 3

RK_ACTION_HOST_PACK = 1
RK_ACTION_HOST_EXPAND = 2
RK_ACTION_HOST_CHECK = 3
RK_ACTION_UPLOAD = 4
RK_ACTION_NPU_TASK = 5
RK_ACTION_DOWNLOAD = 6
RK_ACTION_HOST_COMPUTE = 7
RK_ACTION_COMMIT = 8

RK_TEMPLATE_MM_F16 = 1
RK_TEMPLATE_MM_ACC_F16 = 2
RK_TEMPLATE_EW_ADD_F16 = 3
RK_TEMPLATE_EW_MUL_F16 = 4
RK_TEMPLATE_LUT_SIGMOID_F16 = 5
RK_TEMPLATE_LUT_HARDSIGMOID_F16 = 6
RK_TEMPLATE_LUT_GELU_GATE_F16 = 7
RK_TEMPLATE_EW_MAX_F16 = 8

RK_SUBMIT_CHAIN = 0
RK_SUBMIT_SEPARATE = 1
RK_CORE_SINGLE = 0
RK_CORE_AUTO = 1

RK_ACTION_FLAG_TRANSPOSE_SOURCE = 1
RK_ACTION_FLAG_CACHE_CONSTANT = 2
RK_ACTION_FLAG_BROADCAST_SCALAR = 4
RK_ACTION_FLAG_BROADCAST_CHANNEL = 8

BUFFER_ROLE_MAP = {
    "input": RK_ROLE_INPUT,
    "input_slice": RK_ROLE_INPUT_SLICE,
    "weight": RK_ROLE_WEIGHT,
    "accumulator": RK_ROLE_ACCUMULATOR,
    "gate": RK_ROLE_GATE,
    "output": RK_ROLE_OUTPUT,
    "workspace": RK_ROLE_SCRATCH,
    "constant": RK_ROLE_CONSTANT,
    "identity": RK_ROLE_IDENTITY,
    "state": RK_ROLE_STATE,
}
LAYOUT_MAP = {
    "co.cube.f16.v1": RK_LAYOUT_CUBE_F16,
    "co.weight.f16.v1": RK_LAYOUT_WEIGHT_F16,
    "co.linear.f16.v1": RK_LAYOUT_LINEAR_F16,
}
KIND_MAP = {
    "host_pack": RK_ACTION_HOST_PACK,
    "host_expand": RK_ACTION_HOST_EXPAND,
    "host_check": RK_ACTION_HOST_CHECK,
    "upload": RK_ACTION_UPLOAD,
    "npu_task": RK_ACTION_NPU_TASK,
    "download": RK_ACTION_DOWNLOAD,
    "host_compute": RK_ACTION_HOST_COMPUTE,
    "commit": RK_ACTION_COMMIT,
}
TEMPLATE_MAP = {
    "co.mm.f16.v1": RK_TEMPLATE_MM_F16,
    "co.mm_acc.f16.v1": RK_TEMPLATE_MM_ACC_F16,
    "co.ew_add.f16.v1": RK_TEMPLATE_EW_ADD_F16,
    "co.ew_mul.f16.v1": RK_TEMPLATE_EW_MUL_F16,
    "co.ew_max.f16.v1": RK_TEMPLATE_EW_MAX_F16,
    "co.lut.sigmoid.f16.v1": RK_TEMPLATE_LUT_SIGMOID_F16,
    "co.lut.hardsigmoid.f16.v1": RK_TEMPLATE_LUT_HARDSIGMOID_F16,
    "co.lut.gelu_gate.f16.v1": RK_TEMPLATE_LUT_GELU_GATE_F16,
}


class RKValueDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("role", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32), ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 2), ("nbytes", ctypes.c_uint64),
        ("name", ctypes.c_char_p), ("name_len", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class RKTensorDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32), ("layout", ctypes.c_uint32),
        ("rows", ctypes.c_uint32), ("cols", ctypes.c_uint32),
        ("physical_rows", ctypes.c_uint32), ("physical_cols", ctypes.c_uint32),
        ("atom", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
    ]


class RKBufferDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("role", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32), ("alignment", ctypes.c_uint32),
        ("layout_id", ctypes.c_uint32), ("padding_state", ctypes.c_uint32),
        ("lifetime_first", ctypes.c_uint32), ("lifetime_last", ctypes.c_uint32),
        ("offset", ctypes.c_uint64), ("capacity", ctypes.c_uint64),
    ]


class RKViewDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("buffer_id", ctypes.c_uint32),
        ("offset_bytes", ctypes.c_uint64),
        ("valid_region", ctypes.c_uint32 * 4),
        ("tensor", RKTensorDesc),
    ]


class RKActionDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("kind", ctypes.c_uint32),
        ("template_id", ctypes.c_uint32), ("ninputs", ctypes.c_uint32),
        ("noutputs", ctypes.c_uint32), ("ndeps", ctypes.c_uint32),
        ("source_binding", ctypes.c_uint32), ("source_row_offset", ctypes.c_uint32),
        ("source_col_offset", ctypes.c_uint32),
        ("source_row_stride", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("act_kind", ctypes.c_uint32),
        ("inputs", ctypes.POINTER(ctypes.c_uint32)),
        ("outputs", ctypes.POINTER(ctypes.c_uint32)),
        ("deps", ctypes.POINTER(ctypes.c_uint32)),
        ("origin", ctypes.c_char * 64),
    ]


class RKSubmitDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("mode", ctypes.c_uint32),
        ("core_policy", ctypes.c_uint32), ("ntask_ids", ctypes.c_uint32),
        ("ndeps", ctypes.c_uint32),
        ("task_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("deps", ctypes.POINTER(ctypes.c_uint32)),
    ]


class RKPublishDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("binding_id", ctypes.c_uint32), ("view_id", ctypes.c_uint32),
        ("dst_row_offset", ctypes.c_uint32), ("dst_col_offset", ctypes.c_uint32),
        ("rows", ctypes.c_uint32), ("cols", ctypes.c_uint32),
    ]


class RKPlanDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("profile_hash", ctypes.c_uint64), ("source_hash", ctypes.c_uint64),
        ("values", ctypes.POINTER(RKValueDesc)), ("nvalues", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("buffers", ctypes.POINTER(RKBufferDesc)), ("nbuffers", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("views", ctypes.POINTER(RKViewDesc)), ("nviews", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("actions", ctypes.POINTER(RKActionDesc)), ("nactions", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint32),
        ("submits", ctypes.POINTER(RKSubmitDesc)), ("nsubmits", ctypes.c_uint32),
        ("reserved4", ctypes.c_uint32),
        ("publish", ctypes.POINTER(RKPublishDesc)), ("npublish", ctypes.c_uint32),
        ("reserved5", ctypes.c_uint32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("host_budget_bytes", ctypes.c_uint64),
        ("claimed_device_bytes", ctypes.c_uint64),
        ("claimed_host_bytes", ctypes.c_uint64),
        ("runtime_abi_version", ctypes.c_uint32),
        ("reserved6", ctypes.c_uint32),
    ]


def _c_string(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _hash64(value: str) -> int:
    value = value or "0"
    try:
        return int(value[:16], 16) & ((1 << 64) - 1)
    except ValueError:
        import hashlib
        return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8],
                              "little")


def _array(seq: Iterable[int]):
    values = list(seq)
    if not values:
        return None, 0, None
    arr = (ctypes.c_uint32 * len(values))(*map(int, values))
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32)), len(values), arr


def build_plan_descriptor(plan: Plan):
    """Build descriptor arrays plus a holder list keeping every pointer alive."""
    holder: list[Any] = []
    values = (RKValueDesc * len(plan.bindings))()
    for i, binding in enumerate(plan.bindings):
        name = _c_string(binding.name)
        holder.append(name)
        shape = list(binding.shape) + [0] * (2 - len(binding.shape))
        values[i] = RKValueDesc(
            abi_version=RK_ABI_VERSION,
            struct_size=ctypes.sizeof(RKValueDesc), id=binding.id,
            role={"input": RK_ROLE_INPUT, "constant": RK_ROLE_CONSTANT,
                  "output": RK_ROLE_OUTPUT}[binding.role],
            dtype=RK_DTYPE_F16, rank=len(binding.shape),
            shape=(ctypes.c_uint32 * 2)(*shape),
            nbytes=binding.nbytes,
            name=name, name_len=len(name), reserved=0)
    holder.append(values)

    buffers = (RKBufferDesc * len(plan.buffers))()
    for i, buffer in enumerate(plan.buffers):
        buffers[i] = RKBufferDesc(
            abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKBufferDesc),
            id=buffer.id, role=BUFFER_ROLE_MAP[buffer.role],
            slot_id=buffer.slot_id, alignment=buffer.alignment,
            layout_id=LAYOUT_MAP[buffer.layout_id],
            padding_state={"zero": 0, "finite": 1, "unknown": 2,
                           "valid_only": 3}[buffer.padding_state],
            lifetime_first=buffer.lifetime[0], lifetime_last=buffer.lifetime[1],
            offset=buffer.offset, capacity=buffer.capacity)
    holder.append(buffers)

    views = (RKViewDesc * len(plan.views))()
    for i, view in enumerate(plan.views):
        tensor = view.tensor
        views[i] = RKViewDesc(
            abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKViewDesc),
            id=view.id, buffer_id=view.buffer_id, offset_bytes=view.offset_bytes,
            valid_region=(ctypes.c_uint32 * 4)(*view.valid_region),
            tensor=RKTensorDesc(
                abi_version=RK_ABI_VERSION,
                struct_size=ctypes.sizeof(RKTensorDesc),
                dtype=RK_DTYPE_F16, layout=LAYOUT_MAP[tensor.layout],
                rows=tensor.rows, cols=tensor.cols,
                physical_rows=tensor.physical_rows,
                physical_cols=tensor.physical_cols, atom=tensor.atom,
                reserved=0, bytes=tensor.bytes))
    holder.append(views)

    actions = (RKActionDesc * len(plan.actions))()
    for i, action in enumerate(plan.actions):
        in_ptr, ninputs, in_arr = _array(action.inputs)
        out_ptr, noutputs, out_arr = _array(action.outputs)
        dep_ptr, ndeps, dep_arr = _array(action.deps)
        holder.extend(x for x in (in_arr, out_arr, dep_arr) if x is not None)
        attrs = action.attrs
        flags = 0
        if bool(attrs.get("transpose_source", False)):
            flags |= RK_ACTION_FLAG_TRANSPOSE_SOURCE
        if attrs.get("cache") == "constant":
            flags |= RK_ACTION_FLAG_CACHE_CONSTANT
        if attrs.get("broadcast") == "scalar":
            flags |= RK_ACTION_FLAG_BROADCAST_SCALAR
        elif attrs.get("broadcast") == "channel":
            flags |= RK_ACTION_FLAG_BROADCAST_CHANNEL
        elif attrs.get("broadcast") is not None:
            reject(RK_E_UNSUPPORTED,
                   f"unknown broadcast kind {attrs.get('broadcast')!r}",
                   stage="ffi")
        origin = _c_string(action.origin)[:63]
        actions[i] = RKActionDesc(
            abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKActionDesc),
            id=action.id, kind=KIND_MAP[action.kind],
            template_id=(TEMPLATE_MAP[action.template_id]
                         if action.template_id else 0),
            ninputs=ninputs, noutputs=noutputs, ndeps=ndeps,
            source_binding=int(attrs.get("source_binding",
                                        attrs.get("binding_id", 0))),
            source_row_offset=int(attrs.get("src_row_offset", 0)),
            source_col_offset=int(attrs.get("src_col_offset", 0)),
            source_row_stride=int(attrs.get("src_row_stride", 0)),
            flags=flags, act_kind=0,
            inputs=in_ptr, outputs=out_ptr, deps=dep_ptr,
            origin=origin)
    holder.append(actions)

    submits = (RKSubmitDesc * len(plan.submits))()
    for i, submit in enumerate(plan.submits):
        task_ptr, ntask, task_arr = _array(submit.task_ids)
        dep_ptr, ndeps, dep_arr = _array(submit.deps)
        holder.extend(x for x in (task_arr, dep_arr) if x is not None)
        submits[i] = RKSubmitDesc(
            abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKSubmitDesc),
            id=submit.id,
            mode=RK_SUBMIT_CHAIN if submit.mode == "chain" else RK_SUBMIT_SEPARATE,
            core_policy=(RK_CORE_SINGLE if submit.core_policy ==
                         "single_core_per_submit" else RK_CORE_AUTO),
            ntask_ids=ntask, ndeps=ndeps, task_ids=task_ptr, deps=dep_ptr)
    holder.append(submits)

    publish = (RKPublishDesc * len(plan.publish))()
    for i, item in enumerate(plan.publish):
        publish[i] = RKPublishDesc(
            abi_version=RK_ABI_VERSION,
            struct_size=ctypes.sizeof(RKPublishDesc),
            binding_id=item.binding_id, view_id=item.view_id,
            dst_row_offset=item.dst_row, dst_col_offset=item.dst_col,
            rows=item.rows, cols=item.cols)
    holder.append(publish)

    desc = RKPlanDesc(
        abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKPlanDesc),
        schema_version=plan.schema_version, flags=0,
        profile_hash=_hash64(plan.profile_hash),
        source_hash=_hash64(plan.source_hash),
        values=values, nvalues=len(values), reserved0=0,
        buffers=buffers, nbuffers=len(buffers), reserved1=0,
        views=views, nviews=len(views), reserved2=0,
        actions=actions, nactions=len(actions), reserved3=0,
        submits=submits, nsubmits=len(submits), reserved4=0,
        publish=publish, npublish=len(publish), reserved5=0,
        device_budget_bytes=(plan.budget.device_budget_bytes
                             if plan.budget is not None
                             else plan.accounting.device_request_bytes),
        host_budget_bytes=(plan.budget.host_budget_bytes
                           if plan.budget is not None
                           else plan.accounting.host_bytes),
        claimed_device_bytes=plan.accounting.device_request_bytes,
        claimed_host_bytes=plan.accounting.host_bytes,
        runtime_abi_version=RK_ABI_VERSION, reserved6=0)
    holder.append(desc)
    return desc, holder
