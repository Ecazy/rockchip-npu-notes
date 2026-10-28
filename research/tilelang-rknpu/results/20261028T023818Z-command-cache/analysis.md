# Command-BO packing reuse A/B

This experiment reuses the command BO packed at workspace creation for
eligible single-submit workspaces.  The control packs every submit; the
sync arm reuses the pack but keeps the historical command-BO prep/fini;
the adopted arm also removes that repeated CPU cache maintenance.
Output sync, guard, download, publication and failure isolation are
identical in all arms.

## Isolated cases

µs; ratio = control / arm; primary is the median of per-round paired
medians over seven rotated rounds.

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.7 | 336.3 | 320.0 | **1.0866×** | 1.0846× | 1.0846–1.0952 | yes |
| PV group b8 | 333.4 | 322.6 | 305.1 | **1.0923×** | 1.0917× | 1.0917–1.0988 | yes |
| QKV b8 | 168.6 | 166.2 | 165.1 | **1.0247×** | 0.8374× | 0.8374–1.0603 | yes |
| MLP b32 | 593.0 | 595.3 | 592.4 | **1.0000×** | 0.9971× | 0.9971–1.0039 | yes |

The cache-on-with-sync arm isolates the command-sync component:

| case | control/cache+sync | cache+sync/cache |
|---|---:|---:|
| QK group b8 | 1.0346× | 1.0510× |
| PV group b8 | 1.0325× | 1.0574× |
| QKV b8 | 1.0123× | 1.0071× |
| MLP b32 | 0.9971× | 1.0049× |


Creation cost (process setup medians from the same seven paired runs;
the direct first-build interval is the cleaner attribution):

| case | setup control | setup cache | setup delta | first command build |
|---|---:|---:|---:|---:|
| QK group b8 | 1193.5 | 1232.6 | +39.1 | 19.25 |
| PV group b8 | 1192.3 | 1239.3 | +47.0 | 21.87 |
| QKV b8 | 422.9 | 414.5 | -8.5 | 2.04 |
| MLP b32 | 3716.7 | 3685.5 | -31.2 | n/a (ineligible) |

## Command preparation breakdown

Medians of 64 builds into a separate command workspace on the same fd,
using the creation-time `co_program` array.  `sync` is the command-BO
prep+fini pair; clear, pack, verify and handles are the rest of the
removed build.  MLP b32 is ineligible (8 submits).

| case | clear | pack | verify | handles | sync | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.50 | 15.17 | 19.25 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.79 | 15.75 | 21.87 |
| QKV b8 | 0.29 | 0.00 | 0.00 | 0.00 | 0.88 | 1.46 | 2.04 |

Host accounting measured with `rk_workspace_host_bytes`: the cache adds
**3184 B** per eligible workspace (`sizeof(co_command_plan)`), and **0 B** of command BO because the
existing `co_execute_workspace` command BO is reused.

## Full model (uninstrumented)

| model | control | cache+sync | cache | control/cache | p10 | bitwise |
|---|---:|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3450.1 | 3386.2 | 3327.0 | **1.0365×** | 1.0352× | yes |
| Tiny ViT b1 | 1505.3 | 1481.1 | 1464.7 | **1.0276×** | 1.0250× | yes |

The sync-only contribution is 1.0181× for b8 and 1.0108× for b1; the rest is the removed CPU clear/pack/verify work.


Default-on check: the isolated outputs with no environment override and
with `RK_PLAN_COMMAND_CACHE=0` are bitwise equal for qk_b8 (yes), pv_b8 (yes), qkv_b8 (yes).

## Contract evidence

- The cached plan borrows the workspace's existing command BO; no new BO
  is allocated, no handle crosses a workspace/fd/domain, and the plan is
  freed with the workspace.
- The runtime only reaches the cached path when `program_cache_ready` is
  set for the same single submit; multi-submit, `CO_SEPARATE` with more
  than one task, failed builds and all other workspaces keep the original
  per-submit packing path.
- `command_cache_test` compares cached and freshly built plans with the
  same implementation: command bytes, task order, chain tail, flags and
  handle lists match; a failed submit leaves the plan and bytes unchanged
  and performs no output sync; wrong fd/workspace/reallocation is refused.
- Board `bcmdverify` pv_b8: 512 cached executions, 32256 command bytes, unchanged=True after an explicit FROM_DEVICE sync.
- Board `bcmdverify` qk_b8: 512 cached executions, 32256 command bytes, unchanged=True after an explicit FROM_DEVICE sync.
- Board `bcmdverify` qkv_b8: 1024 cached executions, 1008 command bytes, unchanged=True after an explicit FROM_DEVICE sync.

## Validation

- command-cache host test: `command_cache: 254 checks, 0 failures`
- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `351 passed, 71 subtests passed in 30.58s`
- failure atomicity: `4 passed in 0.19s`
- frozen numeric replay: **PASS**, problems: []

## Decision

Adopt `RK_PLAN_COMMAND_CACHE=1` as the default for eligible
single-submit workspaces and keep `RK_PLAN_COMMAND_CACHE=0` as the
control.  `RK_PLAN_COMMAND_CACHE_SYNC=1` remains the audit arm that
restores the repeated command-BO sync without restoring packing.

