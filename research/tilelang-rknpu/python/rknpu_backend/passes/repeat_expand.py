# SPDX-License-Identifier: GPL-3.0-or-later
"""Expand a single-MM plan into a static-repeat group plan.

A manifest may declare that one immutable plan is instantiated N times over a
static repeat axis (the Tiny Transformer's per-(batch, head) QK/PV work).  This
pass turns such a single-task plan into one plan with N independent tasks that
share three device buffers and submit as one chain, so the N instantiations cost
one Host transition instead of N.

Admission is deliberately narrow and explicit.  Everything not matching the
shape below is refused rather than guessed at:

* exactly one submit, one submit task, and that task is ``co.mm.f16.v1``;
* the matmul has no K chaining and no M/N tiling (one primitive tile), because
  the only board evidence covers independent tasks of that geometry;
* exactly three buffers/views (input tile, weight tile, output tile) and the
  action sequence ``upload, upload, npu_task, download`` with one publish entry.

The generated plan keeps the parent's layout and layout ids, so the packing,
padding and numeric behaviour of each task are the ones already accepted; only
the number of instances and the buffer slicing change.  A separate-submit plan
of the same shape is always kept available by leaving the parent plan untouched.
"""
from __future__ import annotations

from dataclasses import replace

from .storage import CO_GUARD, CO_MAX_WORDS, CO_PROGRAM_BYTES, CO_SLACK
from ..errors import RK_E_BUDGET, RK_E_SHAPE, RK_E_UNSUPPORTED, reject
from ..plan import (Accounting, Action, Binding, Buffer, ConstantSpec, Plan,
                    Publish, Submit, View, with_plan_hash)

GROUP_RECIPE = "independent_mm_v1"
GROUP_TEMPLATE = "co.mm.f16.v1"
# Validation domain, not a convenience limit.  The only board evidence behind an
# independent MM chain is one geometry (16x16x16 with a 32-wide packed tile) and
# up to 32 tasks (results/20261026T033032Z-b2probe).  Structure independence is a
# necessary condition, not a license to generalise.  These numbers are pinned to
# the C verifier's constants by a test that reads the ABI dump, so the two sides
# cannot drift apart.
MAX_GROUP_TASKS = 32
GROUP_M = 16
GROUP_N = 16
GROUP_K = 16
GROUP_PHYS_COLS = 32
HOST_STAGING_FLOOR = 65536


def _admit(plan: Plan) -> tuple[Action, list[Action]]:
    """Return the MM task and its two uploads, or refuse the plan."""
    if len(plan.submits) != 1:
        reject(RK_E_UNSUPPORTED, "repeat expansion needs exactly one submit",
               stage="repeat_expand", actual=len(plan.submits))
    submit = plan.submits[0]
    if len(submit.task_ids) != 1:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion needs exactly one task per submit (K chaining "
               "is a different recipe and has no evidence for this shape)",
               stage="repeat_expand", actual=len(submit.task_ids))
    tasks = [a for a in plan.actions if a.kind == "npu_task"]
    if len(tasks) != 1 or tasks[0].template_id != GROUP_TEMPLATE:
        reject(RK_E_UNSUPPORTED,
               f"repeat expansion only admits one {GROUP_TEMPLATE} task",
               stage="repeat_expand",
               actual=[a.template_id for a in tasks])
    task = tasks[0]
    attrs = task.attrs
    if attrs.get("source_k_loop") is not None or \
            attrs.get("source_mn_tiling") is not None:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion only admits a single primitive tile "
               "(no K loop, no M/N tiling)", stage="repeat_expand",
               actual={k: attrs.get(k) for k in ("source_k_loop",
                                                 "source_mn_tiling")})
    if len(plan.bindings) != 3 or len(plan.buffers) != 3 or len(plan.views) != 3:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion only admits a 3-tensor plan (input, weight, "
               "output)", stage="repeat_expand",
               actual={"bindings": len(plan.bindings),
                       "buffers": len(plan.buffers),
                       "views": len(plan.views)})
    if len(plan.constants) != 0:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion does not yet re-slice plan constants",
               stage="repeat_expand", actual=len(plan.constants))
    uploads = [a for a in plan.actions if a.kind == "upload"]
    downloads = [a for a in plan.actions if a.kind == "download"]
    if len(uploads) != 2 or len(downloads) != 1:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion admits exactly two uploads and one download",
               stage="repeat_expand",
               actual={"uploads": len(uploads), "downloads": len(downloads)})
    roles = [b.role for b in plan.bindings]
    if roles != ["input", "input", "output"]:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion admits two inputs and one output",
               stage="repeat_expand", actual=roles)
    if len(plan.publish) != 1:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion admits exactly one publish tile",
               stage="repeat_expand", actual=len(plan.publish))
    # The geometry, not just the action shape.  The probe covered one M/N/K and a
    # single packed tile; anything else has no evidence and is refused here as
    # well as in the C verifier.
    for i, (binding, view) in enumerate(((plan.bindings[0], plan.views[0]),
                                        (plan.bindings[1], plan.views[1]),
                                        (plan.bindings[2], plan.views[2]))):
        tensor = view.tensor
        want = ((GROUP_M, GROUP_N, GROUP_PHYS_COLS),
                (GROUP_N, GROUP_K, GROUP_PHYS_COLS),
                (GROUP_M, GROUP_N, GROUP_PHYS_COLS))[i]
        if (tensor.rows, tensor.cols, tensor.physical_cols) != want or \
                (binding.shape != (want[0], want[1])):
            reject(RK_E_UNSUPPORTED,
                   "repeat expansion only admits the validated 16x16x16 "
                   "geometry (32-wide packed tile)", stage="repeat_expand",
                   actual={"binding": i, "shape": list(binding.shape),
                           "tensor": [tensor.rows, tensor.cols,
                                      tensor.physical_cols]})
    if int(task.attrs.get("k0", 0)) != 0 or \
            int(task.attrs.get("kt", -1)) != GROUP_K:
        reject(RK_E_UNSUPPORTED,
               "repeat expansion only admits a single full K tile of "
               f"{GROUP_K}", stage="repeat_expand",
               actual={"k0": task.attrs.get("k0"), "kt": task.attrs.get("kt")})
    return task, uploads


def expand_static_repeat(plan: Plan, nrepeat: int, *,
                         shared_bo: bool = False) -> Plan:
    """Return a ``GROUP_RECIPE`` plan with ``nrepeat`` independent instances.

    ``shared_bo=False`` keeps one BO per task (3N slots) -- the layout the first
    board probe validated, with the alias contracts of the single-task plan.
    ``shared_bo=True`` packs the N tiles of each role into one BO as disjoint
    slices (three BOs total), which the limited region-granular admission now
    allows; it is the layout that makes the chained submit pay off, because the
    per-call guard cost scales with the number of BOs.
    """
    if not isinstance(nrepeat, int) or nrepeat < 1:
        reject(RK_E_SHAPE, "repeat count must be a positive integer",
               stage="repeat_expand", actual=nrepeat)
    if nrepeat > MAX_GROUP_TASKS:
        reject(RK_E_UNSUPPORTED,
               f"repeat group of {nrepeat} exceeds the admitted maximum "
               f"{MAX_GROUP_TASKS}", stage="repeat_expand")
    if nrepeat == 1:
        # Nothing to group: the caller keeps the original separate-submit plan.
        return plan

    task, uploads = _admit(plan)
    in_a, in_b, out_y = plan.bindings
    buf_a, buf_w, buf_y = plan.buffers
    view_a, view_w, view_y = plan.views
    up_a = next(u for u in uploads if u.outputs[0] == view_a.id)
    up_w = next(u for u in uploads if u.outputs[0] == view_w.id)
    down = next(a for a in plan.actions if a.kind == "download")
    pub = plan.publish[0]
    submit = plan.submits[0]

    tiles = (buf_a.capacity, buf_w.capacity, buf_y.capacity)
    if shared_bo:
        # Three BOs, one per role; tile i of a role is a view at i * tile bytes.
        # Uploads of one role therefore write disjoint slices of one buffer, which
        # the verifier admits by region and rejects on any overlap.
        first_task = 2 * nrepeat
        last_task = 3 * nrepeat - 1
        last_action = 4 * nrepeat - 1
        bindings: list[Binding] = []
        for i in range(nrepeat):
            for src in (in_a, in_b, out_y):
                bindings.append(Binding(id=len(bindings), name=f"{src.name}_{i}",
                                        role=src.role, dtype=src.dtype,
                                        shape=src.shape))
        buffers = (
            Buffer(id=0, role=buf_a.role, slot_id=0, offset=0,
                   capacity=tiles[0] * nrepeat, alignment=buf_a.alignment,
                   layout_id=buf_a.layout_id, lifetime=(0, last_task),
                   padding_state=buf_a.padding_state,
                   must_not_share_bo=buf_a.must_not_share_bo),
            Buffer(id=1, role=buf_w.role, slot_id=1, offset=0,
                   capacity=tiles[1] * nrepeat, alignment=buf_w.alignment,
                   layout_id=buf_w.layout_id, lifetime=(nrepeat, last_task),
                   padding_state=buf_w.padding_state,
                   must_not_share_bo=buf_w.must_not_share_bo),
            Buffer(id=2, role=buf_y.role, slot_id=2, offset=0,
                   capacity=tiles[2] * nrepeat, alignment=buf_y.alignment,
                   layout_id=buf_y.layout_id, lifetime=(first_task, last_action),
                   padding_state=buf_y.padding_state,
                   must_not_share_bo=buf_y.must_not_share_bo),
        )
        views = []
        for i in range(nrepeat):
            for slot, (src_view, tile) in enumerate((
                    (view_a, tiles[0]), (view_w, tiles[1]), (view_y, tiles[2]))):
                views.append(View(id=len(views), buffer_id=slot,
                                  offset_bytes=i * tile,
                                  tensor=src_view.tensor,
                                  valid_region=src_view.valid_region))
        data_bo_count = 3
    else:
        # First round's layout: ONE BO PER TASK (3N slots).  Every existing alias
        # and writer contract is untouched, at the cost of 32 KiB slack per BO.
        first_task = 2 * nrepeat
        last_task = 3 * nrepeat - 1
        last_action = 4 * nrepeat - 1
        bindings = []
        for i in range(nrepeat):
            for src in (in_a, in_b, out_y):
                bindings.append(Binding(id=len(bindings), name=f"{src.name}_{i}",
                                        role=src.role, dtype=src.dtype,
                                        shape=src.shape))
        buffers = []
        views = []
        for i in range(nrepeat):
            for role, (src_buf, src_view) in enumerate((
                    (buf_a, view_a), (buf_w, view_w), (buf_y, view_y))):
                bid = len(buffers)
                if role == 0:
                    lifetime = (i, first_task + i)
                elif role == 1:
                    lifetime = (nrepeat + i, first_task + i)
                else:
                    lifetime = (first_task + i, 3 * nrepeat + i)
                buffers.append(Buffer(
                    id=bid, role=src_buf.role, slot_id=bid, offset=0,
                    capacity=src_buf.capacity, alignment=src_buf.alignment,
                    layout_id=src_buf.layout_id, lifetime=lifetime,
                    padding_state=src_buf.padding_state,
                    must_not_share_bo=src_buf.must_not_share_bo))
                views.append(View(id=len(views), buffer_id=bid, offset_bytes=0,
                                  tensor=src_view.tensor,
                                  valid_region=src_view.valid_region))
        buffers = tuple(buffers)
        data_bo_count = 3 * nrepeat
    # views[3i], views[3i+1], views[3i+2] are A_i, W_i, Y_i.

    # Action order: all uploads, then all tasks, then all downloads.  A host
    # action may not sit inside a task group, and the verifier requires the
    # nearest preceding producer of every read to be a declared dependency.
    # With one BO per task each tile has exactly one producer, so task i declares
    # its own two uploads and download i declares its own task.
    actions: list[Action] = []
    for i in range(nrepeat):
        actions.append(Action(
            id=len(actions), kind="upload", template_id=None, inputs=(),
            outputs=(views[3 * i].id,), deps=(),
            origin=f"repeat_expand:upload_a:{i}",
            attrs=dict(up_a.attrs, source_binding=bindings[3 * i].id)))
    for i in range(nrepeat):
        actions.append(Action(
            id=len(actions), kind="upload", template_id=None, inputs=(),
            outputs=(views[3 * i + 1].id,), deps=(),
            origin=f"repeat_expand:upload_w:{i}",
            attrs=dict(up_w.attrs, source_binding=bindings[3 * i + 1].id)))
    first_task = len(actions)
    for i in range(nrepeat):
        actions.append(Action(
            id=len(actions), kind="npu_task", template_id=GROUP_TEMPLATE,
            inputs=(views[3 * i].id, views[3 * i + 1].id),
            outputs=(views[3 * i + 2].id,),
            deps=(i, nrepeat + i),
            origin=f"repeat_expand:task:{i}",
            attrs=dict(task.attrs, submit_group=None)))
    last_task = first_task + nrepeat - 1
    for i in range(nrepeat):
        # Each output BO has exactly one producer (its own task), so the
        # download declares that task.  The single chained submit still only
        # completes as a unit, which is why publication waits for the group gate.
        actions.append(Action(
            id=len(actions), kind="download", template_id=None,
            inputs=(views[3 * i + 2].id,), outputs=(),
            deps=(first_task + i,),
            origin=f"repeat_expand:download:{i}",
            attrs=dict(down.attrs, binding_id=bindings[3 * i + 2].id)))

    publish = tuple(Publish(name=bindings[3 * i + 2].name,
                            binding_id=bindings[3 * i + 2].id,
                            view_id=views[3 * i + 2].id, rows=pub.rows,
                            cols=pub.cols, dst_row=0, dst_col=0)
                    for i in range(nrepeat))
    submits = (Submit(id=0,
                      task_ids=tuple(range(first_task, first_task + nrepeat)),
                      recipe_id=GROUP_RECIPE, mode="chain",
                      core_policy=submit.core_policy, deps=()),)

    # Accounting: same formulas the planner and the C verifier use.
    if shared_bo:
        slot_caps = [tiles[0] * nrepeat, tiles[1] * nrepeat, tiles[2] * nrepeat]
    else:
        slot_caps = [tiles[role] for _ in range(nrepeat) for role in range(3)]
    peak_payload = sum(slot_caps)
    reserved = sum(cap + CO_SLACK + CO_GUARD for cap in slot_caps)
    command_workspace = nrepeat * CO_MAX_WORDS * 8
    device_request = reserved + command_workspace
    input_bytes = 2 * nrepeat * (in_a.shape[0] * in_a.shape[1] * 2)
    output_bytes = nrepeat * (out_y.shape[0] * out_y.shape[1] * 2)
    largest_output = out_y.shape[0] * out_y.shape[1] * 2
    runtime_scratch = (nrepeat * CO_PROGRAM_BYTES + largest_output +
                       nrepeat * (8 + 4))
    host_bytes = (input_bytes + output_bytes +
                  max(output_bytes, HOST_STAGING_FLOOR) + runtime_scratch)
    if device_request <= 0 or host_bytes <= 0:
        reject(RK_E_BUDGET, "repeat expansion produced an empty budget",
               stage="repeat_expand")
    accounting = Accounting(
        peak_payload_bytes=peak_payload, reserved_bytes=reserved,
        data_bo_count=data_bo_count, device_request_bytes=device_request,
        command_workspace_bytes=command_workspace, host_bytes=host_bytes,
        task_count=nrepeat, submit_count=1, transfers=3 * nrepeat)

    return with_plan_hash(Plan(
        name=f"{plan.name}+repeat{nrepeat}" + ("+shared" if shared_bo else ""),
        profile_hash=plan.profile_hash,
        source_hash=f"{plan.source_hash}|repeat_expand:n={nrepeat}",
        bindings=tuple(bindings), constants=tuple(),
        buffers=tuple(buffers), views=tuple(views), actions=tuple(actions),
        submits=submits, publish=publish, accounting=accounting,
        budget=plan.budget))


def remap_group_inputs(group: Plan, nrepeat: int, specs: list[dict]) -> Plan:
    """Apply an explicit source-view mapping to a shared-BO repeat group.

    ``specs`` has one entry per single-instance input, in binding order, with:

    * ``shape``: full source binding shape seen by the C upload packer;
    * ``row_stride``: source row length;
    * ``col_base`` / ``col_stride``: per-repeat column slice;
    * ``transpose``: source transpose flag.

    The device buffers, task sequence, output slices and submission structure are
    unchanged.  Every repeat's upload action is rewritten explicitly; the C
    verifier independently checks the resulting source capacity and index range.
    """
    if not isinstance(specs, list) or not specs:
        reject(RK_E_UNSUPPORTED, "mapped group needs explicit input specs",
               stage="repeat_expand")
    expected = 2 * nrepeat
    uploads = [a for a in group.actions if a.kind == "upload"]
    if len(uploads) != expected:
        reject(RK_E_SHAPE, "mapped group upload count mismatch",
               stage="repeat_expand", actual=len(uploads), required=expected)
    bindings = list(group.bindings)
    checked = [False] * expected
    for action in uploads:
        origin = action.origin or ""
        role = 0 if "upload_a:" in origin else 1 if "upload_w:" in origin else -1
        try:
            repeat = int(origin.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            reject(RK_E_SCHEMA, "mapped group upload origin is not indexed",
                   stage="repeat_expand", actual=origin)
        if role < 0 or role >= len(specs) or not 0 <= repeat < nrepeat:
            reject(RK_E_SHAPE, "mapped group upload origin is outside its map",
                   stage="repeat_expand", actual=origin)
        spec = specs[role]
        shape = tuple(spec["shape"])
        if len(shape) != 2 or any(not isinstance(v, int) or v <= 0
                                  for v in shape):
            reject(RK_E_SHAPE, "mapped source shape must be two positive ints",
                   stage="repeat_expand", actual=list(shape))
        if not isinstance(spec.get("row_stride"), int) or \
                spec["row_stride"] != shape[1]:
            reject(RK_E_SHAPE, "mapped source row stride must equal its width",
                   stage="repeat_expand", actual=spec.get("row_stride"))
        if not isinstance(spec.get("transpose"), bool):
            reject(RK_E_SCHEMA, "mapped transpose flag must be boolean",
                   stage="repeat_expand")
        col_base = int(spec["col_base"])
        col_stride = int(spec["col_stride"])
        col_modulus = int(spec.get("col_modulus", 1))
        if col_modulus <= 0:
            reject(RK_E_SHAPE, "mapped column modulus must be positive",
                   stage="repeat_expand", actual=col_modulus)
        col_offset = col_base + (repeat % col_modulus) * col_stride
        if col_offset < 0:
            reject(RK_E_SHAPE, "mapped source column offset is negative",
                   stage="repeat_expand", actual=col_offset)
        attrs = dict(action.attrs)
        attrs.update(src_row_stride=shape[1],
                     src_col_offset=col_offset,
                     transpose_source=spec["transpose"])
        actions = list(group.actions)
        actions[action.id] = replace(action, attrs=attrs)
        group = replace(group, actions=tuple(actions))
        binding_index = 3 * repeat + role
        bindings[binding_index] = replace(bindings[binding_index],
                                          shape=shape)
        checked[2 * repeat + role] = True
    if not all(checked):
        reject(RK_E_SCHEMA, "mapped group did not bind every upload",
               stage="repeat_expand")
    bindings = tuple(bindings)
    input_bytes = sum(b.nbytes for b in bindings if b.role == "input")
    output_bytes = sum(b.nbytes for b in bindings if b.role == "output")
    largest_output = max((b.nbytes for b in bindings if b.role == "output"),
                         default=0)
    output_total = max(output_bytes, HOST_STAGING_FLOOR)
    runtime_scratch = (nrepeat * CO_PROGRAM_BYTES + largest_output +
                       nrepeat * (8 + 4))
    host_bytes = input_bytes + output_bytes + output_total + runtime_scratch
    acct = replace(group.accounting, host_bytes=host_bytes,
                   transfers=3 * nrepeat)
    return with_plan_hash(replace(
        group, name=f"{group.name}+input_map", bindings=bindings,
        accounting=acct))
