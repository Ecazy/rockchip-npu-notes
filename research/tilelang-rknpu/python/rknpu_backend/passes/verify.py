# SPDX-License-Identifier: GPL-3.0-or-later
"""Final plan-level verification before serialization."""
from __future__ import annotations

from ..errors import RK_E_SHAPE, reject
from ..plan import Plan


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
    for action in plan.actions:
        if action.kind == "host_compute":
            reject(RK_E_SHAPE, "MVP forbids hidden host arithmetic actions",
                   stage="verify", action_id=action.id)
    return plan
