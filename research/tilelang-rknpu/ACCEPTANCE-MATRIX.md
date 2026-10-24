# MVP acceptance matrix

Checkpoint evidence: board `results/20261023T144054Z-rk3588/`, Host
`results/20261024T032922Z-host/` (219 pytest / 41 subtests, 27/27 ctest, 711 C
checks, sanitizer enabled).  Status terms are deliberately narrow:
**implemented+verified**, **verification gap**, **explicitly unsupported**,
**out of scope**, and **future extension**.

## P0–P9

| Item | Status | Code | Tests / logs |
|---|---|---|---|
| P0 source/dependency lock | **implemented+verified** | `provenance/source-lock.json`, `requirements.lock`, `scripts/lock_source.py`, `scripts/lock_deps.sh`; `provenance/p0-source-build.json` | source/dependency scopes and isolated-wheel runs are recorded in `P0-SOURCE-LOCK.md`. Locked TileLang/TVM/tvm-ffi/dlpack/libbacktrace archives were independently built with exact commit/hash/source-tree inventory, CMakeCache/compile commands, toolchain and library hashes. Source-built import full pytest 213/40 and CTest 27/27; 27 gate plans semantically equal; same board campaign 81/81. |
| P1 frontend | **implemented+verified for finite accepted forms** | `python/rknpu_backend/frontend/tilelang_adapter.py` | `test_frontend*.py` and `test_staged_tilelang.py`; real multi-block TileLang projection/FFN/branch fixtures now feed the tiled RKIR backend. Same-grid cross-block dependencies, FP32-only K accumulation and inexact block coverage are rejected. Board cases `*_tilelang_staged_chain` pass on all three masks. |
| P2 RKIR/RKPlan/schema/capabilities | **implemented+verified** | `ir.py`, `plan.py`, `schema/*.json`, capability manifest | `test_ir.py`, `test_capabilities.py`, `test_plan_verify.py`; C contract 26/26. |
| P3 planner | **implemented+verified for the finite representative shape list** | `passes/dag.py`, `passes/tile.py`, `passes/storage.py` | `test_lowering.py`, `test_storage.py`, `test_dataflow_multitile.py`, `test_staged_tilelang.py`; representative boundaries include 1/13/257/300/512-wide M/N/K, ragged K=257, M/N multi-tile, odd/even K shards, and staged source shapes `512x256x512`. The 1..4096 limits are rule-checked; exhaustive combination enumeration is explicitly not required. |
| P4 combinations/storage/residual liveness | **implemented+verified for listed nodes** | `passes/legalize.py`, `passes/dag.py`, `passes/storage.py` | `test_dataflow.py`, `test_dataflow_multitile.py`, `test_staged_tilelang.py`, `test_broadcast_execution.py`, `test_multi_output_execution.py`; projection/FFN/branch/ragged/multi-output on board. 3-D/dynamic/state/arbitrary broadcast are explicitly unsupported. |
| P5 C descriptor/FFI | **implemented+verified** | `custom-ops/src/rk_plan_validate.c`, `include/rk_plan.h`, `runtime/ffi.py` | `rk_plan_contract_test` 26 checks, `rk_plan_host_test` 711 checks; `test_ffi.py`, `test_plan_verify.py`. |
| P6 instantiate/execute | **implemented+verified for the declared safety contract** | `custom-ops/src/rk_plan_runtime.c`, `RUNTIME-LIFECYCLE.md` | Pre-submit binding/nonfinite/upload failures are safe; executor errors poison and retain command+data BOs; BUSY and final publication guard regressions pass. Board success path passes. The supported post-failure behavior is quarantine/stop; recovery is out of scope. |
| P7 CLI | **implemented+verified on Host and RK3588** | `python/rknpu_backend/cli.py`, `tools/board_cli_check.py` | `test_cli.py` covers caller/owned fd, cleanup failure not masked, and rc mapping; board `core-*/cli/cli-report.json` shows compile/validate/explain rc0, two successful runs with stable fd count, bad binding rc2/no publish. |
| P8 board admission | **implemented+verified for the declared MVP scope** | campaign/verifier scripts | Final source-built rerun `results/20261023T144054Z-rk3588/`: **27 cases × 3 serial masks = 81/81 PASS**, including projection/FFN/branch from real multi-block TileLang staged sources, tiled RKIR cases, broadcast, dual output, ragged tail, K shards, guards and the full CLI path. Source hash recomputed true; raw outputs and dual references archived. |
| P9 expansion | **future extension** | n/a | Dynamic host broadcast, state/recurrent/hybrid, online core scheduling and unverified composite templates remain outside P0–P8 closure. |

## Next milestone (separate from MVP)

The complete-model milestone is reported separately in
`MODEL-MILESTONE-STATUS.md`: full 10,000-image MNIST test evaluation on RK3588
core 1 passes for the learned MLP (98.31%) and Tiny Transformer (98.21%), with
100% CPU/declared-path top-1 agreement and a PASS campaign verifier.  It adds
ReLU, bias, model package/runtime, static multi-token/head layout, explicit
RMSNorm/Softmax/pooling bridges and hybrid scheduling without changing the MVP
scope above.

## Specific cross-cutting conditions

| Condition | Status | Code / proof | Tests / logs |
|---|---|---|---|
| Target identity and no silent widening | **implemented+verified** | `custom-ops/src/rk_plan_target.c` exact NUL-token compatible and exact driver string; `capabilities.py` profile validation | `test_target_check.py`, `test_capabilities.py`; board `target_check_rc=0`, profile hash match and ABI 1. |
| Binding dtype/shape/role/finite checks | **implemented+verified** | `binding_matches`, `check_finite_binding`, `rk_workspace_run` input/output overlap checks | `test_ws_poison.py::test_binding_error_before_submit_is_safe`, C host alias checks, board NaN input guard. |
| BO independence / no accidental sharing | **implemented+verified** | `must_not_share_bo`, slot/lifetime validator in `plan.py` and `rk_plan_validate.c` | `test_storage.py::ConstantLifetimeTests`, `test_plan_verify.py::test_mm_acc_alias_is_rejected`, C `aliases/shared BO` negative. |
| Device/host budgets | **implemented+verified** | `passes/storage.py`, C `claimed_*` validation and `CO_SLACK`/`CO_GUARD`/command workspace accounting | `test_storage.py`, C `multi_output_budget_case`, board environment budget snapshot. |
| Concurrent BUSY | **implemented+verified (Host/mock)** | atomic `ws->busy` exchange before mutation | `test_workspace_busy.py` (second concurrent run `RK_E_BUSY`, no publish, no leak). |
| Submit failure / poisoned recovery | **implemented+verified (Host/mock); real-device gap** | runtime poison path; no output publication before success | `test_ws_poison.py`, `test_guard_failure.py`; board intentionally avoids unsafe timeout injection. |
| CLI fd ownership, cleanup failure, void close ABI and error codes | **implemented+verified** | `cli.py` uses real `void rocket_close(int)`, opens/closes only its own fd after quiescent destroy, preserves `--fd N` ownership, never closes provider fd after poisoned destroy, reports destroy failures, preserves original run error, and never returns success when cleanup failed | `test_cli.py`, `test_runtime_abi.py` (header + ctypes + C probe), `test_workspace_busy.py`; board `cli-report.json` records fd equality and bad-binding rc2/no publication. |
| Shared-fd provider context quarantine | **implemented+verified on Host/mock/provider boundary; real-device fault unmeasured** | `rknpu_provider.c`: atomic per-fd `faulted` set inside `submit_lock` before unlock; later submits return `-EPIPE` without task-BO rewrite; runtime rejects same-fd existing/new workspaces; faulted `rocket_close` retains fd/task BO. Patch record: `provenance/upstream-patches.json` | C `provider_context` queued-submit + healthy-close test; `test_workspace_busy.py::test_shared_fd_context_fault_blocks_other_and_new_workspaces`; `test_runtime_abi.py`. |
| Command/task/data and provider-context failure lifecycle | **implemented+verified on Host/mock; real-device unknown state unmeasured** | `RUNTIME-LIFECYCLE.md`: creation failure frees command+data BOs; pre-submit failures are safe; executor errors poison the workspace; provider `atomic_int faulted` is set under `submit_lock` before unlock, blocks queued/shared-fd/new-workspace submits, and faulted `rocket_close` retains fd/task BO. Upstream patch recorded in `provenance/upstream-patches.json`. | `test_ws_poison.py`; `test_workspace_busy.py` concurrent failing-run and shared-fd two-workspace/new-workspace paths; `test_runtime_abi.py`; C `provider_context` queued-submit task-BO/close test. `live_bos == 7` is only an allocation-count check, not lifecycle proof. |
| Source/library provenance | **source independently verified; library recorded-only** | `scripts/run-board-admission.sh`, verifier `artifact_verification` | board `source-bundle.sha256` recomputed equal; `board/library.sha256` is a remote record and is explicitly `verified=false` on the Host. |
| Credential hygiene | **implemented+verified** | `.env` ignored; askpass + exact-value scan | board campaign `credential scan: PASS`; exact scan of `results/20261023T144054Z-rk3588/` is zero-match. |

## MVP scope boundary

The exhaustive-shape, multi-board, concurrency, long-stress and undefined
“full release gate” items are removed.  The finite representative matrix, P0
source rebuild and same-board rerun are complete.  There is no remaining required
MVP acceptance condition for the declared scope.

**Explicit scope boundary:** real RK3588 submit/recovery after an unknown fault is
not implemented, not tested and not promised.  Host/mock verifies pre-submit safety,
BUSY, poison-on-unknown, output non-publication and command/data BO retention; the
real-device success path is verified.  The accepted failure contract is quarantine,
refuse retry and stop, not automatic recovery.  This limitation is not counted as
an implicit MVP gate.
