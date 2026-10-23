# Runtime BO lifecycle and failure classification

This audit follows `rk_workspace_create -> rk_workspace_run -> rk_emit_action ->
co_execute_profiled -> rocket_submit_tasks_flags` and the destruction path.  It
distinguishes failures that occur **before any submit** from failures where the
device may still touch BOs.

## Ownership

- Data BOs: allocated per plan slot by `co_buffer_alloc`; owned by `rk_workspace`.
- Command BO: allocated by `co_execute_workspace_init` into
  `rk_workspace.execute_workspace.commands`; owned by the same workspace.
- Task descriptors: a Host stack array passed to the provider; the provider owns
  any copy it needs for the lifetime of a submit.  The task-descriptor array is
  never freed or reused by the runtime while a submit may be in flight.
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

## Unknown device state

Any non-zero result from `co_execute_profiled` is treated as **unclassified**:
the execution adapter does not prove whether the provider submitted, whether DMA
started, or whether completion occurred.  The runtime therefore:

1. marks the workspace `poisoned`;
2. does not publish staged outputs;
3. refuses subsequent runs with `RK_E_POISONED`;
4. makes `rk_workspace_destroy` return `RK_E_POISONED` and retain the handle,
   command BO, task resources and all data BOs;
5. requires process-level recovery/context destruction outside this library.

Regression:
`tests/test_ws_poison.py::test_unclassified_submit_failure_poisons_and_retains`
runs in a child process, verifies no publish, no resubmit after poison, a retained
handle, and an unchanged live-BO/resident-byte set after the refused destroy.

## BUSY and concurrent ownership

`atomic_exchange(&ws->busy, 1)` prevents two runs from sharing command/data BOs.
A competing run returns `RK_E_BUSY` before mutation; destroy also refuses while
BUSY.  `tests/test_workspace_busy.py` covers this with a delayed mock executor.

## CLI cleanup and fd ownership

The CLI now:

- owns the fd only when it called `rocket_open`; with `--fd N` it never closes
  the caller's descriptor;
- preserves the original execution error when `ws.close()` also fails and writes
  every cleanup diagnostic to stderr;
- returns a cleanup error (resource exit code 3) if execution succeeded but
  workspace destruction or owned-fd close failed;
- never returns 0 when cleanup failed.

`tests/test_cli.py` patches the workspace/library seam to cover run-success +
destroy-failure, run-error + destroy-failure, and caller-owned fd behavior.

## Evidence boundary

No dangerous real-device timeout or submit failure has been injected.  The
real-device submit-failure/recovery path is therefore **unmeasured on RK3588**;
only the Host/mock ownership and poisoning policy above is verified.  This record
does not claim that driver quiescence, context recovery, or physical resource
release has succeeded.
