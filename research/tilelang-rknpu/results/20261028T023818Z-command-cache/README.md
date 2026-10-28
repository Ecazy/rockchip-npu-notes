# Creation-time command-BO packing reuse

This experiment evaluates one bounded candidate: for a workspace whose
creation-time `co_program` cache is ready for its single submit, pack the
command BO, chain links, task descriptors and handle lists once at creation
and reuse them for every submit. The original per-submit packing path remains
the control and the fallback.

The command BO itself is not new: it is the existing
`co_execute_workspace.commands` allocation. The cache adds only a
`co_command_plan` descriptor block (3,184 B per eligible workspace) and
allocates **0 B** of BO memory.

## Eligibility

- `program_cache_ready` is set for the same single submit;
- the submit is `CHAIN`, or `CO_SEPARATE` with exactly one task;
- the cached task set fits `max_submit_tasks` and the command BO;
- the plan is workspace-local and borrows that workspace's command BO.

Multi-submit workspaces, `CO_SEPARATE` with more than one task, unsupported
emitters and any failed build keep the original per-submit packing path. No
cache state crosses a workspace, fd, IOMMU domain or BO reallocation.

## Three arms

- `control`: `RK_PLAN_COMMAND_CACHE=0`, pack every submit;
- `cache_sync`: cache on, `RK_PLAN_COMMAND_CACHE_SYNC=1`, reuse the pack but
  keep the historical command-BO prep/fini each submit;
- `cache`: cache on, `RK_PLAN_COMMAND_CACHE_SYNC=0` (adopted default).

Output sync, finite checks, guard, download, publication and failure isolation
are identical in all arms.

## Isolated cases (µs, seven rotated rounds)

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.7 | 336.3 | 320.0 | **1.0866×** | 1.0846× | 1.0846–1.0952 | yes |
| PV group b8 | 333.4 | 322.6 | 305.1 | **1.0923×** | 1.0917× | 1.0917–1.0988 | yes |
| QKV b8 | 168.6 | 166.2 | 165.1 | **1.0247×** | 0.8374× | 0.8374–1.0603 | yes |
| MLP b32 | 593.0 | 595.3 | 592.4 | 1.0000× | 0.9971× | 0.9971–1.0039 | yes |

The QK/PV p10 values agree with their medians. The QKV p10 contains one
process-level outlier while its median stays positive. MLP b32 is ineligible
(8 submits) and is retained only as proof that the fallback stays bitwise
neutral, not as a speed estimate.

## Command-build breakdown (µs, median of 64 builds)

| case | clear | pack | verify | handles | command sync | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.50 | 15.17 | 19.25 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.79 | 15.75 | 21.87 |
| QKV b8 | 0.29 | 0.00 | 0.00 | 0.00 | 0.88 | 1.46 | 2.04 |

The first-build interval is the direct creation-time cost. Process-level
`SETUP_US` deltas are +39.1 µs (QK), +47.0 µs (PV), −8.5 µs (QKV) and include
process noise. The steady gain is 15–29 µs per QK/PV submit, so creation cost
is recovered after roughly one to two calls.

## Full model (uninstrumented)

| model | control | cache+sync | cache | control/cache | p10 | bitwise |
|---|---:|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3450.1 | 3386.2 | 3327.0 | **1.0365×** | 1.0363× | yes |
| Tiny ViT b1 | 1505.3 | 1481.1 | 1464.7 | **1.0276×** | 1.0190× | yes |

The sync-only contribution is 1.0181× (b8) and 1.0108× (b1). The rest is the
removed CPU clear, pack, chain verify and handle-list construction.

## Command-BO contract evidence

- The command BO is written only during creation: clear, regcmd copy, chain
  seal and `rkt_chain_verify`, followed by one `rocket_bo_fini`.
- The cached runtime path contains no command-BO write. It submits the stored
  task descriptors and handle lists through the existing provider call.
- `bcmdverify` ran 512 (QK/PV) and 1024 (QKV) real device submissions, then
  explicitly synced the command BO FROM_DEVICE and compared it byte for byte
  with the creation snapshot: **unchanged** (32,256 / 32,256 / 1,008 bytes).
- Command bytes, task order, chain tail, flags and handle sets are compared
  against a second build of the same implementation in `command_cache_test`;
  the independent workspace uses its own command BO and fails identity checks
  if a stale plan is paired with it.
- Wrong fd, wrong workspace, wrong mode, wrong program array and a reallocated
  command BO are refused. A failed submit leaves bytes unchanged, performs no
  output sync, poisons through the existing runtime path and publishes nothing.

## Validation

- `command_cache_test`: 254 checks, 0 failures.
- CTest: 33/33 passed (includes creation/allocation failure, submit failure,
  guard failure, non-finite input and failure-without-publication scenarios).
- pytest: 351 passed, 71 subtests; targeted failure atomicity: 4 passed, 1
  selected model failure test passed.
- Frozen numeric replay against the packed spec: **PASS**, 0 problems.
- Default-on check: outputs with no override and with
  `RK_PLAN_COMMAND_CACHE=0` are bitwise equal for QK/PV/QKV.

## Decision

Adopt `RK_PLAN_COMMAND_CACHE=1` as the default for eligible single-submit
workspaces. Keep `RK_PLAN_COMMAND_CACHE=0` as the control and rollback.
`RK_PLAN_COMMAND_CACHE_SYNC=1` remains the audit arm that restores only the
repeated command-BO sync.

## Files

- `analysis.md` / `analysis.json`: generated tables and decision.
- `pair/r1..r7/`: isolated three-arm logs, outputs and process status.
- `bcmd/`: clear/pack/verify/sync/handle breakdown and Host accounting.
- `cmdbytes/`: real-device command-BO byte-stability runs.
- `full-model-b8/`, `full-model-b1/`: uninstrumented end-to-end runs.
- `numeric/`: frozen acceptance observation and verifier output.
- `defaultcheck/`: default-on vs explicit-off output hashes.
- `checks/`: host test and failure-atomicity logs.
- `artifacts/`: final board library, probe, build logs and environment.
