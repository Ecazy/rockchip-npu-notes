# Budgeted multi-submit static preparation

Control keeps the existing single-submit strategy.  The prepared arm
caches each eligible submit's programs and packs an independent command
BO per submit; the default budget is 4 MiB per workspace and covers
Host program/entry bytes plus device command BO bytes.

## Current multi-submit cost (control arm, same A/B)

Per-call medians from the control arm (`attr/<case>-off.log`), plus the
independent pre-candidate snapshot in `baseline-cost/`:

| case | emit µs | command µs | total µs |
|---|---:|---:|---:|
| MLP b32 | 15.5 | 35.0 | 625.0 |
| MLP b1 | 15.9 | 34.7 | 447.7 |

## Isolated MLP A/B

7 rotated rounds; per-call samples are in `pair/r*/.../WARM_US`.

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP b32 | 625.3 | 580.1 | 572.2 | **1.0918×** | 1.0886× | 1.0886–1.0970 | yes |
| MLP b1 | 446.5 | 401.9 | 393.5 | **1.1341×** | 1.1328× | 1.1328–1.1397 | yes |

Attribution confirms the removed work (medians, µs):

| case | arm | emit | command | output sync | total |
|---|---|---:|---:|---:|---:|
| MLP b32 | off | 15.5 | 35.0 | 29.9 | 625.0 |
| MLP b32 | sync | 0.0 | 8.5 | 30.0 | 579.2 |
| MLP b32 | nosync | 0.0 | 0.4 | 30.0 | 572.5 |
| MLP b1 | off | 15.9 | 34.7 | 25.5 | 447.7 |
| MLP b1 | sync | 0.0 | 8.5 | 25.4 | 402.6 |
| MLP b1 | nosync | 0.0 | 0.3 | 25.7 | 393.7 |

## Creation cost and cache memory

| case | setup control µs | setup cache µs | delta µs | first |
|---|---:|---:|---:|---:|
| MLP b32 | 3925.8 | 4126.5 | +200.7 | 3764.5 |
| MLP b1 | 3663.3 | 3840.9 | +177.6 | 3647.5 |

| case | cached submits | device command bytes | Host delta | unique command BOs | bytes unchanged |
|---|---:|---:|---:|---|---|
| MLP b32 | 8/8 | 11088 | 208528 B | yes | yes (8 BOs / 1024 runs) |
| MLP b1 | 8/8 | 11088 | 208528 B | yes | yes (8 BOs / 1024 runs) |

All MLP b1/b8/b32 plans share the same 8-submit/11-task structure, so
the per-workspace cache footprint is the same 208,528 B Host + 11,088 B
device.  The 4 MiB budget is a hard per-workspace cap, and a tight
budget caches only the prefix of submits that fits.

## Command regions

- Each cached submit is packed into its own `rocket_bo`, allocated with
  the exact `words * 8` size by `co_command_plan_build(..., workspace=NULL)`.
- No workspace command BO pointer is adjusted or faked into a slice;
  `rkt_chain_pack_at` derives regcmd addresses from that BO's own DMA base
  and `rkt_chain_verify` validates the chain inside the same object.
- The eight MLP submits report eight unique handles; no later submit can
  overwrite an earlier cache.
- `command_cache_test` independently builds two owned command plans and
  asserts the second submit leaves the first plan's bytes unchanged.

## Five-profile full model (interleaved, 256 images, 7 rounds)

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b1 | 601.1 | 549.2 | **1.0954×** | 1.0472× | 1571.3 | 1703.8 | yes |
| mlp_b32 | 1059.2 | 1006.5 | **1.0511×** | 1.0303× | 30233.0 | 31612.6 | yes |
| mlp_b8 | 701.0 | 649.2 | **1.0781×** | 1.0397× | 10292.1 | 10974.2 | yes |
| vit_b1 | 1442.4 | 1438.9 | **1.0021×** | 1.0010× | 707.2 | 710.2 | yes |
| vit_b8 | 3375.3 | 3362.4 | **1.0030×** | 1.0014× | 2372.6 | 2381.5 | yes |

All five profiles use the same compiled graphs, weights, fd and timed
interval; control and prepared runners alternate order per call and round.
Accuracy is identical and all logits are bitwise equal.

## Validation

- command-cache host test: `command_cache: 270 checks, 0 failures`
- rk_plan host: `RK_PLAN host checks=820 failures=0`
- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `352 passed, 71 subtests passed in 30.39s`
- frozen numeric replay: **PASS**, problems: []

## Decision

Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` with the 4 MiB default budget.
`RK_PLAN_MULTI_SUBMIT_CACHE=0` remains the control.  No hardware recipe,
numeric policy or asynchronous execution changed.

