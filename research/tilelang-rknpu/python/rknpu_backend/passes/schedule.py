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


def _hoist_host_uploads(draft: DraftPlan) -> None:
    """Move all upload actions before the first device task.

    A Host upload cannot be placed inside a hardware PC chain, so uploads that
    were lazily emitted between two device stages prevent an otherwise legal
    Submit merge.  Uploads read only external bindings and write their declared
    input/constant destination; moving them earlier is safe after dependencies,
    action ids and buffer live ranges are rebuilt consistently.  This is only
    invoked for the opt-in ``fuse_task_chain`` path.
    """
    uploads = [a for a in draft.actions if a.kind == "upload"]
    rest = [a for a in draft.actions if a.kind != "upload"]
    ordered = uploads + rest
    remap: dict[int, int] = {}
    rebuilt = []
    for new_id, old in enumerate(ordered):
        deps = []
        for dep in old.deps:
            if dep not in remap:
                reject(RK_E_SHAPE,
                       "upload hoisting encountered a non-topological dependency",
                       stage="schedule", action_id=old.id)
            deps.append(remap[dep])
        remap[old.id] = new_id
        rebuilt.append(replace(old, id=new_id, deps=tuple(deps)))
    draft.actions = rebuilt
    for buf in draft.buffers:
        buf.first = 10**9
        buf.last = -1
    for action in draft.actions:
        for view_id in action.inputs + action.outputs:
            buf = draft.buffers[draft.views[view_id].buffer_id]
            buf.first = min(buf.first, action.id)
            buf.last = max(buf.last, action.id)


# Mixed-template chains are admitted by exact template sequence, not by a
# generic "all dependencies happened to be earlier" rule.  The only sequence
# with board evidence is the staged SwiGLU FFN.  Extending this list requires a
# new C verifier entry and a new board probe.
ALLOWED_DEPENDENCY_CHAINS: tuple[tuple[str, ...], ...] = (
    (
        "co.mm.f16.v1",
        "co.mm.f16.v1",
        "co.lut.sigmoid.f16.v1",
        "co.ew_mul.f16.v1",
        "co.ew_mul.f16.v1",
        "co.mm.f16.v1",
        "co.ew_add.f16.v1",
    ),
    (
        "co.ew_mul.f16.v1",
        "co.mm.f16.v1",
        "co.ew_mul.f16.v1",
        "co.ew_add.f16.v1",
        "co.lut.rsqrt.f16.v1",
        "co.mm.f16.v1",
        "co.ew_mul.f16.v1",
    ),
    (
        "co.mm.f16.v1",
        "co.ew_mul.f16.v1",
        "co.ew_mul.f16.v1",
        "co.ew_add.f16.v1",
    ),
    tuple(["co.mm.f16.v1"] * 8 + ["co.ew_add.f16.v1"] * 3),
)


def _is_allowed_chain_prefix(templates: list[str]) -> bool:
    return any(len(templates) <= len(chain) and
               tuple(templates) == chain[:len(templates)]
               for chain in ALLOWED_DEPENDENCY_CHAINS)


def _is_allowed_chain(templates: list[str]) -> bool:
    return tuple(templates) in ALLOWED_DEPENDENCY_CHAINS


def _fuse_task_chain_submits(draft: DraftPlan,
                             submits: list[Submit]) -> list[Submit]:
    """Merge adjacent dependency-connected Submit groups into one chain.

    This is a narrowly admitted extension, not a generic graph superoptimizer.
    A candidate must match a prefix of an exact board-validated template
    sequence, be contiguous in action order, have a serial Submit predecessor,
    have every input produced by a prior task in the candidate or by an action
    from an earlier Submit, have pairwise-disjoint writes, and remain within the
    ABI task limit.  If the full sequence is not reached, the original Submit
    groups are emitted unchanged.
    """
    if len(submits) < 2:
        return submits
    by_id = {a.id: a for a in draft.actions}
    action_index = {a.id: i for i, a in enumerate(draft.actions)}
    writers: dict[int, list[int]] = {}
    for a in draft.actions:
        if a.kind != "npu_task":
            continue
        for view_id in a.outputs:
            buf = draft.views[view_id].buffer_id
            writers.setdefault(buf, []).append(a.id)

    runs: list[tuple[list[int], tuple[int, ...]]] = []
    index = 0
    while index < len(submits):
        current = submits[index]
        original_ids = [current.id]
        tasks = list(current.task_ids)
        if not tasks:
            runs.append((tasks, tuple(original_ids)))
            index += 1
            continue
        last_id = current.id
        cursor = index + 1
        while cursor < len(submits):
            nxt = submits[cursor]
            if nxt.mode != "chain" or nxt.core_policy != current.core_policy:
                break
            if nxt.deps != (last_id,):
                break
            if len(tasks) + len(nxt.task_ids) > 64:
                break
            candidate = tasks + list(nxt.task_ids)
            first = action_index[candidate[0]]
            indices = [action_index[t] for t in candidate]
            if indices != list(range(first, first + len(candidate))):
                break
            templates = [by_id[t].template_id for t in candidate]
            if not _is_allowed_chain_prefix(templates):
                break
            candidate_set = set(candidate)
            ok = True
            seen_outputs: set[int] = set()
            for task_id in candidate:
                task = by_id[task_id]
                # SSA: one device production per candidate output buffer.
                for view_id in task.outputs:
                    buf = draft.views[view_id].buffer_id
                    if buf in seen_outputs:
                        ok = False
                        break
                    seen_outputs.add(buf)
                # Explicit task dependencies must be earlier.
                for dep in task.deps:
                    dep_action = by_id[dep]
                    if dep_action.kind != "npu_task":
                        continue
                    if dep in candidate_set:
                        if action_index[dep] >= action_index[task_id]:
                            ok = False
                            break
                    elif action_index[dep] >= first:
                        ok = False
                        break
                if not ok:
                    break
                # Every device input buffer must already be produced either
                # inside this candidate before the consumer or before the
                # candidate's first action.
                for view_id in task.inputs:
                    buf = draft.views[view_id].buffer_id
                    producers = writers.get(buf, [])
                    if not producers:
                        continue  # upload/constant view
                    if not any(action_index[w] < action_index[task_id]
                               for w in producers):
                        ok = False
                        break
                    if any(first <= action_index[w] < action_index[task_id]
                           and w not in candidate_set for w in producers):
                        ok = False
                        break
                    if any(action_index[w] >= action_index[task_id]
                           for w in producers):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            tasks = candidate
            original_ids.append(nxt.id)
            last_id = nxt.id
            cursor += 1
        templates = [by_id[t].template_id for t in tasks]
        if len(original_ids) > 1 and not _is_allowed_chain(templates):
            # Do not publish a partial merge.  Re-emit the untouched groups.
            for sid in original_ids:
                src = submits[index + original_ids.index(sid)]
                runs.append((list(src.task_ids), (sid,)))
        else:
            runs.append((tasks, tuple(original_ids)))
        index = cursor

    # Renumber the serial submit stream.  The compiler's groups are serial;
    # asserting that here prevents a silent dependency rewrite on a future DAG.
    out: list[Submit] = []
    old_to_new: dict[int, int] = {}
    for new_id, (tasks, old_ids) in enumerate(runs):
        for old_id in old_ids:
            old_to_new[old_id] = new_id
        original = submits[old_ids[0]]
        merged = len(tasks) > len(original.task_ids)
        recipe = "dependency_chain_v1" if merged else original.recipe_id
        deps = () if new_id == 0 else (new_id - 1,)
        out.append(Submit(new_id, tuple(tasks), recipe, original.mode,
                          original.core_policy, deps))
    return out


def build_submits(draft: DraftPlan, *, options: CompileOptions) -> list[Submit]:
    if options.fuse_task_chain and options.submit_mode == "chain":
        _hoist_host_uploads(draft)
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
    if options.fuse_task_chain and options.submit_mode == "chain":
        submits = _fuse_task_chain_submits(draft, submits)
    return submits
