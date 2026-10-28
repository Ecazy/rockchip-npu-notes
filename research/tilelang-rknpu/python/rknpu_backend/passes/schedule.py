# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative submit grouping and deterministic topological ordering."""
from __future__ import annotations

from dataclasses import replace

from ..errors import RK_E_SHAPE, reject
from ..options import CompileOptions
from ..plan import Submit
from .dag import DraftPlan


MM_F16 = "co.mm.f16.v1"
MM_ACC_F16 = "co.mm_acc.f16.v1"
EW_ADD_F16 = "co.ew_add.f16.v1"
EW_MAX_F16 = "co.ew_max.f16.v1"
EW_ADD_RELU_F16 = "co.ew_add_relu.f16.v1"


def _is_linear_epilogue(templates: list[str]) -> bool:
    """MM/MM_ACC chain followed by exactly an optional ADD then MAX."""
    if not templates or templates[0] != MM_F16:
        return False
    i = 1
    while i < len(templates) and templates[i] == MM_ACC_F16:
        i += 1
    if i == len(templates):
        return False
    if templates[i] == EW_ADD_RELU_F16:
        return i + 1 == len(templates)
    if templates[i] != EW_ADD_F16:
        return False
    i += 1
    if i < len(templates):
        if templates[i] != EW_MAX_F16:
            return False
        i += 1
    return i == len(templates)


def _is_ew_add_relu(templates: list[str]) -> bool:
    return templates == [EW_ADD_F16, EW_MAX_F16]


def _fuse_epilogue_submits(draft: DraftPlan, submits: list[Submit]) -> list[Submit]:
    """Merge directly dependent Linear/epilogue submits into one chain.

    The merge only re-bundles the exact same tasks in the same order.  It does
    not change templates, arithmetic, transfers, buffers or dependencies.
    """
    by_id = {a.id: a for a in draft.actions}
    action_index = {a.id: i for i, a in enumerate(draft.actions)}
    merged: list[tuple[list[int], str, str, str, tuple[int, ...], list[int]]] = []
    index = 0
    while index < len(submits):
        current = submits[index]
        tasks = list(current.task_ids)
        group_ids = [current.id]
        deps = tuple(current.deps)
        recipe = current.recipe_id
        last_id = current.id
        cursor = index + 1
        while cursor < len(submits):
            nxt = submits[cursor]
            if nxt.mode != "chain" or nxt.core_policy != current.core_policy:
                break
            if nxt.deps != (last_id,):
                break
            prev_last = tasks[-1]
            if prev_last not in by_id[nxt.task_ids[0]].deps:
                break
            # The merged submit must remain contiguous in action order: a host
            # action between two task groups cannot be hidden inside one Submit.
            if action_index[nxt.task_ids[0]] != action_index[prev_last] + 1:
                break
            next_indices = [action_index[t] for t in nxt.task_ids]
            if next_indices != list(range(next_indices[0],
                                          next_indices[0] + len(next_indices))):
                break
            dependents = [a.id for a in draft.actions
                          if prev_last in a.deps and a.id != nxt.task_ids[0]]
            if dependents:
                break
            combined = tasks + list(nxt.task_ids)
            templates = [by_id[t].template_id for t in combined]
            if _is_linear_epilogue(templates):
                new_recipe = "linear_epilogue_v1"
            elif _is_ew_add_relu(templates):
                new_recipe = "ew_add_relu_v1"
            else:
                break
            tasks = combined
            group_ids.append(nxt.id)
            recipe = new_recipe
            last_id = nxt.id
            cursor += 1
        merged.append((tasks, recipe, current.mode, current.core_policy,
                       deps, group_ids))
        index = cursor
    # Renumber and remap submit dependencies.  Every original submit id in a
    # fused group maps to the one new id.
    id_map: dict[int, int] = {}
    for new_id, (_, _, _, _, _, group_ids) in enumerate(merged):
        for old_id in group_ids:
            id_map[old_id] = new_id
    out: list[Submit] = []
    for new_id, (tasks, recipe, mode, core_policy, deps, _) in enumerate(merged):
        new_deps = tuple(sorted({id_map[d] for d in deps if d in id_map}))
        out.append(Submit(new_id, tuple(tasks), recipe, mode, core_policy,
                          new_deps))
    return out


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
        if group.startswith("mm:"):
            recipe = "linear_k_acc_v1"
        elif group.startswith("swiglu:"):
            recipe = "swiglu_ffn_v1"
        else:
            recipe = "gate_ew_v1"
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
    if options.fuse_epilogue and options.submit_mode == "chain":
        submits = _fuse_epilogue_submits(draft, submits)
    return submits
