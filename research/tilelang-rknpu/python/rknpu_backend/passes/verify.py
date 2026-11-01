# SPDX-License-Identifier: GPL-3.0-or-later
"""Final plan-level verification before serialization."""
from __future__ import annotations

from ..errors import RK_E_SCHEMA, RK_E_SHAPE, reject
from ..plan import MAX_BUFFERS_PER_SLOT, Plan


def verify_plan(plan: Plan) -> Plan:
    """Re-check invariants that must not depend on a caller's ``verified`` bit."""
    task_count = sum(1 for action in plan.actions if action.kind == "npu_task")
    submit_tasks = sum(len(submit.task_ids) for submit in plan.submits)
    if task_count != submit_tasks:
        reject(RK_E_SHAPE, "task count differs from submitted task count",
               stage="verify")
    if plan.accounting.task_count != task_count:
        reject(RK_E_SHAPE, "accounting task_count is stale", stage="verify")
    if plan.accounting.submit_count != len(plan.submits):
        reject(RK_E_SHAPE, "accounting submit_count is stale", stage="verify")
    # The C verifier refuses a plan whose slot carries more buffers than its
    # bookkeeping bound.  Re-checking it here keeps "Python verified" honest.
    per_slot: dict[int, int] = {}
    for buffer in plan.buffers:
        count = per_slot.get(buffer.slot_id, 0) + 1
        per_slot[buffer.slot_id] = count
        if count > MAX_BUFFERS_PER_SLOT:
            reject(RK_E_SCHEMA, "too many buffers share one slot",
                   stage="verify",
                   actual={"slot_id": buffer.slot_id,
                           "buffers_in_slot": count},
                   required={"max_buffers_per_slot": MAX_BUFFERS_PER_SLOT})
    for action in plan.actions:
        if action.kind == "host_compute":
            reject(RK_E_SHAPE, "MVP forbids hidden host arithmetic actions",
                   stage="verify", action_id=action.id)
    return plan
