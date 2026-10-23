# Runtime BO lifecycle and failure classification

## Closure status (2026-10-23)

| Item | Status | Primary evidence |
|---|---|---|
| `rocket_close` ctypes ABI | **CLOSED** | vendor header/C probe/ctypes test in `tests/test_runtime_abi.py`; configured as `void(int)` with no status inspection |
| workspace `poisoned` concurrency | **CLOSED** | `atomic_int` + busy-gate ordering; concurrent failing-run regression and structural source check |
| shared provider-context quarantine | **CLOSED (Host/provider boundary)** | per-fd provider `faulted` set under `submit_lock`; queued/shared-fd/new-workspace refusal; faulted close retains fd/task BO; C `provider_context` + shared-fd worker tests |

These close the **code isolation** items.  Physical recovery after a real device
fault remains **unmeasured**; see the evidence boundary below.

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

## Evidence boundary

No dangerous real-device timeout or submit failure has been injected.  The
real-device submit-failure/recovery path is therefore **unmeasured on RK3588**;
only the Host/mock ownership and poisoning policy above is verified.  This record
does not claim that driver quiescence, context recovery, or physical resource
release has succeeded.
