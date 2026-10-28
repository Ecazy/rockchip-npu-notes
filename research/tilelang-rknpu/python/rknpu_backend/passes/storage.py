# SPDX-License-Identifier: GPL-3.0-or-later
"""Lifetime-based slot assignment and explicit device/host budget accounting."""
from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import TargetProfile, check_budgets
from ..errors import RK_E_BUDGET, RK_E_SHAPE, reject
from ..options import CompileOptions
from ..plan import Accounting, Buffer, Plan, PlanBudget, Submit, with_plan_hash
from .dag import DraftBuffer, DraftPlan

CO_SLACK = 32768
CO_GUARD = 64
# sizeof(co_program) in the C seam: the emission array is one of these per submit
# task.  Pinned by tests/test_ffi.py against the C ABI dump so a change to
# co_program cannot silently understate the Host requirement.
CO_PROGRAM_BYTES = 16624
CO_MAX_WORDS = 2048


@dataclass
class _Slot:
    id: int
    capacity: int
    last: int
    reusable: bool = True


def assign_storage(draft: DraftPlan, *, profile: TargetProfile,
                   options: CompileOptions, submits: list[Submit]) -> Plan:
    slots: list[_Slot] = []
    slot_by_buffer: dict[int, int] = {}
    for buf in draft.buffers:
        if buf.first > buf.last:
            reject(RK_E_SHAPE, f"buffer {buf.id} is never used", stage="storage")
        chosen: _Slot | None = None
        if buf.reuse:
            candidates = [
                slot for slot in slots
                if slot.reusable and slot.last < buf.first and
                slot.capacity >= buf.capacity
            ]
            candidates.sort(key=lambda slot: (slot.capacity, slot.id))
            if candidates:
                chosen = candidates[0]
        if chosen is None:
            chosen = _Slot(len(slots), buf.capacity, buf.last, buf.reuse)
            slots.append(chosen)
        else:
            chosen.capacity = max(chosen.capacity, buf.capacity)
            chosen.last = max(chosen.last, buf.last)
            chosen.reusable = chosen.reusable and buf.reuse
        slot_by_buffer[buf.id] = chosen.id

    # The C verifier must not have to rediscover this invariant.  Explicitly mark
    # every accumulation output as non-shareable with each live input of the same
    # action; slot assignment above already gives distinct slots because the
    # prior is live at the accumulation step.
    for action in draft.actions:
        if action.template_id == "co.mm_acc.f16.v1":
            prior_buffer = draft.views[action.inputs[2]].buffer_id
            out_buffer = draft.views[action.outputs[0]].buffer_id
            if slot_by_buffer[prior_buffer] == slot_by_buffer[out_buffer]:
                reject(RK_E_SHAPE, "mm_acc prior/output were assigned the same BO",
                       stage="storage", action_id=action.id)

    buffers = tuple(Buffer(
        id=buf.id, role=buf.role, slot_id=slot_by_buffer[buf.id], offset=0,
        capacity=buf.capacity, alignment=16, layout_id=buf.layout_id,
        lifetime=(buf.first, buf.last), padding_state=buf.padding_state,
        must_not_share_bo=tuple(buf.must_not_share_bo),
    ) for buf in draft.buffers)

    slot_capacity = {slot.id: slot.capacity for slot in slots}
    peak_payload = sum(slot_capacity.values())
    reserved = sum(capacity + CO_SLACK + CO_GUARD
                   for capacity in slot_capacity.values())
    max_tasks = max(len(submit.task_ids) for submit in submits)
    command_workspace = max_tasks * CO_MAX_WORDS * 8
    device_request = reserved + command_workspace

    input_bytes = sum(b.nbytes for b in draft.bindings if b.role == "input")
    output_bytes = sum(b.nbytes for b in draft.bindings if b.role == "output")
    broadcast_expand_bytes = sum(
        draft.views[action.outputs[0]].tensor.bytes
        for action in draft.actions
        if action.kind == "upload" and action.attrs.get("broadcast") is not None)
    # The caller still supplies the scalar/channel source; the expanded tile is
    # an additional Host preparation allocation, not an extra device transfer.
    constant_bytes = sum(c.nbytes for c in draft.constants) + broadcast_expand_bytes
    # Runtime-owned Host scratch is part of the plan's Host requirement: the
    # per-submit emission array (one co_program per submit task), the download
    # tile and the per-output bookkeeping all live for the workspace's lifetime.
    # CO_PROGRAM_BYTES is pinned to sizeof(co_program) by a C/Python test.
    largest_output = max((b.nbytes for b in draft.bindings if b.role == "output"),
                         default=0)
    runtime_scratch = (max_tasks * CO_PROGRAM_BYTES + largest_output +
                       len([b for b in draft.bindings if b.role == "output"]) *
                       (8 + 4))
    host_bytes = input_bytes + output_bytes + constant_bytes + \
        max(output_bytes, 65536) + runtime_scratch

    check_budgets(profile, device_bytes=device_request, host_bytes=host_bytes)
    if device_request > options.device_budget_bytes:
        reject(RK_E_BUDGET, "device budget exceeded", stage="storage",
               actual=device_request, required=options.device_budget_bytes,
               target_profile=profile.profile_hash)
    if host_bytes > options.host_budget_bytes:
        reject(RK_E_BUDGET, "host budget exceeded", stage="storage",
               actual=host_bytes, required=options.host_budget_bytes,
               target_profile=profile.profile_hash)

    accounting = Accounting(
        peak_payload_bytes=peak_payload,
        reserved_bytes=reserved,
        data_bo_count=len(slots),
        device_request_bytes=device_request,
        command_workspace_bytes=command_workspace,
        host_bytes=host_bytes,
        task_count=sum(len(submit.task_ids) for submit in submits),
        submit_count=len(submits),
        transfers=sum(1 for action in draft.actions
                      if action.kind in {"upload", "download"}),
        broadcast_expand_bytes=broadcast_expand_bytes,
    )
    plan = Plan(
        name=draft.name, profile_hash=draft.profile_hash,
        source_hash=draft.source_hash, bindings=tuple(draft.bindings),
        constants=tuple(draft.constants), buffers=buffers,
        views=tuple(draft.views), actions=tuple(draft.actions),
        submits=tuple(submits), publish=tuple(draft.publish),
        accounting=accounting,
        budget=PlanBudget(device_budget_bytes=options.device_budget_bytes,
                          host_budget_bytes=options.host_budget_bytes),
    )
    return with_plan_hash(plan)
