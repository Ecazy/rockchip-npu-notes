# Runtime BO lifecycle and failure classification

## Closure status (2026-10-23)

| Item | Status | Primary evidence |
|---|---|---|
| `rocket_close` ctypes ABI | **CLOSED** | vendor header/C probe/ctypes test in `tests/test_runtime_abi.py`; configured as `void(int)` with no status inspection |
| workspace `poisoned` concurrency | **CLOSED** | `atomic_int` + busy-gate ordering; concurrent failing-run regression and structural source check |
| shared provider-context quarantine | **CLOSED (Host/provider boundary)** | per-fd provider `faulted` set under `submit_lock`; queued/shared-fd/new-workspace refusal; faulted close retains fd/task BO; C `provider_context` + shared-fd worker tests |

These close the **code isolation** items.  Physical recovery after a real device
fault remains **unmeasured and is explicitly out of scope** for this delivery;
the supported behavior after unknown device state is quarantine, refuse further
submission, retain resources and stop.  Failure isolation is verified; recovery is
not claimed.  See the evidence boundary below.

This audit follows `rk_workspace_create -> rk_workspace_run -> rk_emit_action ->
co_execute_profiled -> rocket_submit_tasks_flags` and the destruction path.  It
distinguishes failures that occur **before any submit** from failures where the
device may still touch BOs.

## Ownership

- Data BOs: allocated per plan slot by `co_buffer_alloc`; owned by `rk_workspace`.
- Command BO: allocated by `co_execute_workspace_init` into
  `rk_workspace.execute_workspace.commands`; owned by the same workspace.
- Task descriptors / provider task BO: the runtime passes a Host stack descriptor
  array to the provider.  The provider owns `rknpu_dev.taskbo`, the kernel task
  array, and its fd/context.  `rocket_close(int fd)` is a **void** vendor API and
  frees that provider context (including `taskbo`); it has no return status.
- Views point into data BOs.  No view owns an fd or a BO.

The `live_bos == 7` check in the C host test proves only the count for one
six-data-slot plan plus one command BO.  It is **not** the lifecycle proof below.
The lifecycle proof is the ownership and failure path, plus the seam regressions.

## Creation failure

`rk_workspace_create` builds a plan clone, validates constants, allocates all slot
BOs, then allocates the command workspace.  Any failure after allocation enters
the `fail:` path, which calls `co_execute_workspace_destroy` when needed and
`co_buffer_free` for every allocated slot before returning.  No partially valid
handle is published.

## Pre-submit failures

These are classified safe:

- binding count/dtype/shape/role/finite checks;
- input/output Host alias checks;
- upload packing and upload BO preparation failures before any NPU submit.

The runtime returns the specific error without setting `poisoned`, leaves caller
output buffers untouched, and permits another run or a healthy destroy.  Destroy
then frees the command BO first and every data slot afterwards.

Regressions:

- `tests/test_ws_poison.py::test_binding_error_before_submit_is_safe`
- `tests/test_ws_poison.py::test_nonfinite_input_is_a_safe_presubmit_failure`
- `tests/test_ws_poison.py::test_upload_prep_failure_is_safe_presubmit_and_frees_all_bos`

## Shared provider context quarantine

The unknown-state flag is not workspace-local.  `rknpu_provider.c` now keeps an
`atomic_int faulted` in the per-fd `struct rknpu_dev`:

- `submit_kick()` checks it after taking `submit_lock`;
- any `submit_locked()` failure sets it **before `submit_lock` is released**;
- a queued submit that acquires the lock afterwards returns `-EPIPE` without
  entering the task-BO fill/ioctl path;
- `rocket_context_faulted(fd)` exposes `1/0/<negative errno>` to the runtime;
- `rk_workspace_create` and `rk_workspace_run` reject a faulted shared fd, so an
  existing second workspace and a newly created workspace cannot submit;
- `rocket_close()` on a faulted context returns without destroying `taskbo` and
  without closing the fd, retaining the provider context for process-level
  recovery.  A normal healthy close is unchanged.

The upstream patch is reconstructable from
`provenance/upstream-patches.json` and
`provenance/patches/rknpu-submit-context-fault.patch` (base commit
`6b144b19bb6700f96fc1638ccc7f20ad8719d527`).

## Unknown device state

Any non-zero result from `co_execute_profiled` is treated as **unclassified**:
the execution adapter does not prove whether the provider submitted, whether DMA
started, or whether completion occurred.  The runtime therefore:

1. marks the workspace `poisoned`;
2. does not publish staged outputs;
3. refuses subsequent runs with `RK_E_POISONED`;
4. makes `rk_workspace_destroy` return `RK_E_POISONED` and retain the handle,
   command BO, task resources and all data BOs;
5. requires the caller to retain the provider fd/context too.  In particular it
   must not call `rocket_close` after a poisoned destroy, because that would free
   the provider task BO while the device state is unknown.  Process-level exit is
   the conservative recovery boundary.

`poisoned` is an `atomic_int`.  A run first acquires the atomic `busy` gate and
only then observes poison; a failing run writes poison while holding that gate.
Destroy also acquires the busy gate before reading poison.  No plain unsynchronised
read/write of `poisoned` remains.

Regressions:
`tests/test_ws_poison.py::test_unclassified_submit_failure_poisons_and_retains`
runs in a child process, verifies no publish, no resubmit after poison, a retained
handle, and an unchanged live-BO/resident-byte set after the refused destroy.
`tests/test_workspace_busy.py::test_failing_run_poison_with_concurrent_entrant`
races a failing run with a second entrant: the second receives `RK_E_BUSY`, no
second submit occurs, and the third call receives `RK_E_POISONED`.  The structural
atomic/ordering contract is checked by `tests/test_runtime_abi.py`.
`tests/test_workspace_busy.py::test_shared_fd_context_fault_blocks_other_and_new_workspaces`
creates two workspaces on one fd, fails the first, then verifies the second and a
new workspace are refused with `RK_E_POISONED`, executor count does not increase,
outputs are untouched, and handles/BOs are retained.
The provider-boundary C test `provider_context` seeds a fake per-fd context and
queues two real `submit_kick` callers; the loser sees `-EPIPE` without rewriting
the task array, and `rocket_close` on the faulted context retains the fd/task BO.
The same test exercises a healthy context close and verifies the existing
BO-destroy/slot-retire behavior is unchanged.

## BUSY and concurrent ownership

`atomic_exchange(&ws->busy, 1)` prevents two runs from sharing command/data BOs.
A competing run returns `RK_E_BUSY` before mutation; destroy also refuses while
BUSY.  `tests/test_workspace_busy.py` covers this with a delayed mock executor.

## CLI cleanup and fd ownership

The CLI now:

- owns the fd only when it called `rocket_open`; with `--fd N` it never closes
  the caller's descriptor;
- uses the real vendor `void rocket_close(int)` ABI; it does not inspect a
  fabricated return value or claim confirmed fd closure;
- preserves the original execution error when `ws.close()` also fails and writes
  every cleanup diagnostic to stderr;
- returns a cleanup error (resource exit code 3) if execution succeeded but
  workspace destruction failed;
- does **not** call `rocket_close` when destroy did not confirm quiescence; it
  reports that the provider fd/task BO are retained for process-level recovery;
- never returns 0 when a known cleanup failure occurred.

`tests/test_cli.py` patches the workspace/library seam to cover run-success +
destroy-failure, run-error + destroy-failure, healthy close, and caller-owned fd
behavior.  `tests/test_runtime_abi.py` checks the vendor header declaration, the
ctypes `void` configuration and the compile-time C probe against `rocket_npu.h`.

## Shared immutable packed constants

`rk_shared_constant_get_or_create` (declared in `include/rk_plan.h`) caches one
already-packed upload destination BO.  The caller supplies an opaque key; the
Python planner derives it from the actual FP16 source bytes plus source region,
stride/transpose flags, destination tensor geometry, layout id, buffer
capacity/offset/alignment and padding state.  The C runtime additionally
re-checks the action's source region, stride/transpose flags and destination
tensor/padding geometry, and stores the provider fd with the handle.

The sharing domain is deliberately narrow:

- same live `fd`/provider context only; there is no cross-fd or cross-IOMMU
  domain BO import in this target;
- the handle must stay alive; cache entries disappear when the last workspace
  releases them;
- a workspace retains each unique handle and releases it only after a confirmed
  workspace destroy;
- `rk_workspace_create_shared` skips creation-time uploads for those exact
  actions and maps all plan views of that destination buffer to the shared BO;
- a destination buffer may not mix shared and owned uploads.

The key is never based on shape alone or on a Host pointer.  Unit coverage is in
`tests/test_shared_constants.py` (same-content reuse, same-pointer content
mutation, changed weights, fd mismatch, incompatible destination geometry,
A/B/A release order and run-time numerical equality).

## Persistent upload state

`rk_workspace_set_persistent_upload` nominates one existing single-output upload
action whose destination BO survives across runs.  Normal run uploads are
skipped for that action.  The caller must initialize, clear and incrementally
update the destination through:

- `rk_workspace_update_upload_region` — packs one source rectangle into one
  destination logical rectangle using an explicitly selected row-major or
  transposed source mapping; it does not clear untouched bytes;
- `rk_workspace_clear_upload_destination` — zeros the complete destination for
  reset/initialization boundaries.

Every state mutation takes the same non-blocking `busy` ownership gate as a run,
checks poison/provider-context state, validates the full source and destination
ranges before writing, and performs one `rocket_bo_prep`/`fini` pair around the
copy.  A pre-submit validation failure leaves the workspace healthy.  A provider
failure follows the existing quarantine/stop contract.

The Smol runner creates decode QK/PV workspaces per layer and KV head, performs a
full capacity initialization after reset/prefill, then appends only the new token
stripes.  The first and reset boundaries are counted separately
(`kv_state_init_*`, `kv_state_clear_*`); steady-state append bytes are reported as
`kv_state_append_*`.  Capacity-256 prefill/incremental, reset replay, A/B/A
interleaving, 257-token rejection and post-rejection health are covered in
`results/baseline-layout-reuse/board-final4-current-*`.

## Evidence boundary

No dangerous real-device timeout or submit failure has been injected.  The
real-device submit-failure/recovery path is therefore **unmeasured on RK3588**;
only the Host/mock ownership and poisoning policy above is verified.  This record
does not claim that driver quiescence, context recovery, or physical resource
release has succeeded.

## Dependency-ordered mixed-template task chains

The planner can emit a `dependency_chain_v1` Submit when `fuse_task_chain=true`.
Before the C descriptor is built, the planner hoists upload actions ahead of all
device tasks and rebuilds action ids, dependencies and buffer live ranges.  It
merges only adjacent Submit groups whose action ids are contiguous, whose NPU
dependencies are earlier tasks in the merged group (or an earlier Submit), and
whose task count stays within the ABI limit.

The C verifier independently requires: consecutive task ids, every NPU
dependency earlier in the chain or in an earlier Submit, at least one non-MM
task, and pairwise-disjoint written regions.  The normal read-after-write,
alias, publish and lifecycle checks still run.  This is an envelope change only:
tasks, buffers, uploads/downloads, numeric points and device-internal DDR writes
are unchanged.  Unit coverage is in `tests/test_graph_fusion.py`; the device FFN
control and fused arm are bitwise identical when only this envelope changes.

## Approximate rsqrt capability profile

`co.lut.rsqrt.f16.v1` is exposed only by
`capabilities/rk3588_vendor_co_v1_rmsnorm_lut.json`; the default profile remains
unchanged and continues to reject it.  The template maps to the bounded
`CO_RSQRT_BOUNDED` LUT emitter and is intended for independently calibrated
experimental compositions.  `rmsnorm_approx_module` is a generic seven-task
device graph, not a model-specific runtime shortcut.  Its input domain and FP16
narrowing points are declared in the resulting hybrid manifest metadata; it is
not a replacement for the default Host FP32 RMSNorm path.

The mixed-template allowlist is intentionally closed.  As of the 2026-10-28
stage-2 probe the only admitted sequences are the seven-task staged SwiGLU FFN,
the seven-task approximate RMSNorm chain, the four-task device RoPE reorder,
and the eleven-task PV→O composition.  Each has a matching Python tuple in
`passes/schedule.py` and a C table entry in `rk_plan_validate.c`; a structural
dependency chain outside this set is rejected even if all action ids happen to
be ordered.  Negative tests cover non-consecutive ids, template mismatch,
overlapping writes, and a device-produced value used as an MM weight operand.
