# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative submit grouping and deterministic topological ordering."""
from __future__ import annotations

from dataclasses import replace

from ..errors import RK_E_SHAPE, reject
from ..options import CompileOptions
from ..plan import Submit
from .dag import DraftPlan


def build_submits(draft: DraftPlan, *, options: CompileOptions) -> list[Submit]:
    groups: list[str] = []
    by_group: dict[str, list[int]] = {}
    for action in draft.actions:
        if action.kind != "npu_task":
            continue
        group = action.attrs.get("submit_group")
        if not isinstance(group, str) or not group:
            reject(RK_E_SHAPE, f"NPU task {action.id} has no submit group",
                   stage="schedule", action_id=action.id)
        if group not in by_group:
            groups.append(group)
            by_group[group] = []
        by_group[group].append(action.id)

    submits: list[Submit] = []
    previous: int | None = None
    for group in groups:
        tasks = by_group[group]
        if len(tasks) > 64:
            reject(RK_E_SHAPE, "submit group exceeds 64 tasks", stage="schedule")
        recipe = ("linear_k_acc_v1" if group.startswith("mm:")
                  else "gate_ew_v1")
        if options.submit_mode == "chain":
            sid = len(submits)
            submits.append(Submit(sid, tuple(tasks), recipe, "chain",
                                  "single_core_per_submit",
                                  () if previous is None else (previous,)))
            previous = sid
        else:
            for task in tasks:
                sid = len(submits)
                submits.append(Submit(sid, (task,), recipe, "separate",
                                      "single_core_per_submit",
                                      () if previous is None else (previous,)))
                previous = sid
    if not submits:
        reject(RK_E_SHAPE, "plan contains no NPU submit", stage="schedule")
    return submits
