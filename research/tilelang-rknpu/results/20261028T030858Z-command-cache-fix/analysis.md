# Command-BO packing reuse A/B

This experiment reuses the command BO packed at workspace creation for
eligible single-submit workspaces.  The control packs every submit; the
sync arm reuses the pack but keeps the historical command-BO prep/fini;
the adopted arm also removes that repeated CPU cache maintenance.
Output sync, guard, download, publication and failure isolation are
identical in all arms.

## Isolated cases

µs; ratio = control / arm; primary is the median of per-round paired
medians over 7 rotated rounds.

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.4 | 336.6 | 320.2 | **1.0858×** | 1.0792× | 1.0792–1.0973 | yes |
| PV group b8 | 334.2 | 325.8 | 305.7 | **1.0937×** | 0.9727× | 0.9727–1.0982 | yes |
| QKV b8 | 162.7 | 163.0 | 160.4 | **1.0127×** | 1.0035× | 1.0035–1.0401 | yes |
| MLP b32 | 590.3 | 591.5 | 591.5 | **0.9995×** | 0.9946× | 0.9946–1.0034 | yes |

The cache-on-with-sync arm isolates the command-sync component:

| case | control/cache+sync | cache+sync/cache |
|---|---:|---:|
| QK group b8 | 1.0330× | 1.0510× |
| PV group b8 | 1.0313× | 1.0658× |
| QKV b8 | 1.0018× | 1.0164× |
| MLP b32 | 0.9995× | 1.0000× |


Creation cost (process setup medians from the same 7 paired runs;
the direct first-build interval is the cleaner attribution):

| case | setup control | setup cache | setup delta | first command build |
|---|---:|---:|---:|---:|
| QK group b8 | 1204.3 | 1234.3 | +30.0 | 30.04 |
| PV group b8 | 1207.2 | 1242.5 | +35.3 | 32.67 |
| QKV b8 | 408.3 | 402.5 | -5.8 | 3.79 |
| MLP b32 | 3716.1 | 3696.8 | -19.3 | n/a (ineligible) |

## Command preparation breakdown

Medians of 64 builds into a separate command workspace on the same fd,
using the creation-time `co_program` array.  `sync` is the command-BO
prep+fini pair; clear, pack, verify and handles are the rest of the
removed build.  `total` is one continuous outer interval; sub-parts are
probe-only and may not sum to it.  MLP b32 is ineligible (8 submits).

| case | clear | pack | verify | handles | sync | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 19.25 | 24.21 | 30.04 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 20.12 | 25.37 | 32.67 |
| QKV b8 | 0.00 | 0.29 | 0.00 | 0.00 | 1.75 | 2.62 | 3.79 |

Host accounting measured with `rk_workspace_host_bytes`: the cache adds
**3184 B** per eligible workspace (`sizeof(co_command_plan)`), and **0 B** of command BO because the
existing `co_execute_workspace` command BO is reused.

## Full model (uninstrumented)

| model | control | cache+sync | cache | control/cache | p10 | bitwise |
|---|---:|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3536.7 | 3473.6 | 3409.3 | **1.0392×** | 1.0389× | yes |
| Tiny ViT b1 | 1489.5 | 1474.5 | 1453.1 | **1.0268×** | 1.0248× | yes |

The sync-only contribution is 1.0185× for b8 and 1.0156× for b1; the rest is the removed CPU clear/pack/verify work.


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

- command-cache host test: `command_cache: 259 checks, 0 failures`
- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `351 passed, 71 subtests passed in 30.06s`
- failure atomicity: `4 passed in 0.18s`
- frozen numeric replay: **PASS**, problems: []

## Decision

Adopt `RK_PLAN_COMMAND_CACHE=1` as the default for eligible
single-submit workspaces and keep `RK_PLAN_COMMAND_CACHE=0` as the
control.  `RK_PLAN_COMMAND_CACHE_SYNC=1` remains the audit arm that
restores the repeated command-BO sync without restoring packing.

