# Budgeted multi-submit static preparation

This is the first complete optimization package after the command-cache stage.
It extends the existing creation-time `co_program` and command-BO caches from
single-submit workspaces to a per-submit plan for multi-submit workspaces.

## Control and prepared arms

- control: `RK_PLAN_MULTI_SUBMIT_CACHE=0`; existing single-submit strategy and
  the original per-submit packer for multi-submit plans.
- prepared: `RK_PLAN_MULTI_SUBMIT_CACHE=1` (adopted default); each eligible
  submit is emitted and packed once at workspace creation.
- budget: `RK_PLAN_MULTI_SUBMIT_BUDGET`, default 4 MiB per workspace. It covers
  Host program/entry bytes plus device command BO bytes. A tight budget caches
  only the prefix of submits that fits; no cache is a fallback, not an error.
- sync audit: `RK_PLAN_COMMAND_CACHE_SYNC=1` also re-syncs each prepared command
  BO on every submit, isolating the sync component.

Eligibility is structural: the workspace has more than one submit; each submit
has a non-zero task set, all tasks are `RK_ACTION_NPU_TASK` and successfully
emitted through `rk_emit_action`, and the submit is `CHAIN` or one-task
`CO_SEPARATE`. Multi-task `CO_SEPARATE` and failed emission stay on the original
path. No model-name special case exists.

## Independent command regions

Each prepared submit owns a separate `rocket_bo` allocated with the exact
`words * 8` size by `co_command_plan_build(..., workspace=NULL)`. There is no
pointer adjustment or fake slice of another BO. `rkt_chain_pack_at` derives the
regcmd addresses from that BO's own DMA base and `rkt_chain_verify` validates
the chain inside the same object. The MLP plan reports eight unique command
handles for eight submits, so no later preparation can overwrite an earlier
cache.

`command_cache_test` independently builds two owned command plans and asserts
that submitting the second leaves the first plan's bytes unchanged. It also
covers legal two-program `CO_SEPARATE`, a later-invalid program with zero
submits, cached-plan identity, submit failure and build-failure cleanup.

## Current multi-submit cost (control arm)

Per-call medians, µs:

| case | emit | command prep | total |
|---|---:|---:|---:|
| MLP b32 | 15.5 | 35.0 | 625.0 |
| MLP b1 | 15.9 | 34.7 | 447.7 |

The pre-candidate independent snapshot in `baseline-cost/` agrees on the
component costs (emit 15-16 µs, command 31-35 µs).

## Isolated MLP A/B (7 rotated rounds)

Per-call samples are in `pair/r*/<case>/<arm>.log` as `WARM_US`.

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP b32 | 625.3 | 580.1 | 572.2 | **1.0918×** | 1.0886× | 1.0886–1.0970 | yes |
| MLP b1 | 446.5 | 401.9 | 393.5 | **1.1341×** | 1.1328× | 1.1328–1.1397 | yes |

Attribution confirms the mechanism (medians, µs):

| case | arm | emit | command | output sync | total |
|---|---|---:|---:|---:|---:|
| MLP b32 | off | 15.5 | 35.0 | 29.9 | 625.0 |
| MLP b32 | sync | 0.0 | 8.5 | 30.0 | 579.2 |
| MLP b32 | cache | 0.0 | 0.4 | 30.0 | 572.5 |
| MLP b1 | off | 15.9 | 34.7 | 25.5 | 447.7 |
| MLP b1 | sync | 0.0 | 8.5 | 25.4 | 402.6 |
| MLP b1 | cache | 0.0 | 0.3 | 25.7 | 393.7 |

Output sync, guard, download and publication are unchanged. The remaining
`command` in the sync arm is the opt-in per-submit PREP/FINI pair.

## Creation cost, memory and command stability

| case | setup control µs | setup cache µs | delta | cache-on setup µs |
|---|---:|---:|---:|---:|
| MLP b32 | 3925.8 | 4126.5 | +200.7 | 3764.5 |
| MLP b1 | 3663.3 | 3840.9 | +177.6 | 3647.5 |

| case | cached submits | device command bytes | Host delta | unique BOs | bytes unchanged |
|---|---:|---:|---:|---|---|
| MLP b32 | 8/8 | 11,088 | 208,528 B | yes | yes (8 BOs / 1024 runs) |
| MLP b1 | 8/8 | 11,088 | 208,528 B | yes | yes (8 BOs / 1024 runs) |

The three MLP profiles share the same 8-submit/11-task structure, so the cache
footprint is the same 208,528 B Host + 11,088 B device per workspace. The 4 MiB
budget is a hard per-workspace cap.

## Five-profile full model (interleaved, 256 images, 7 rounds)

Two runners per profile share one fd and alternate call order. Per-call samples
are retained in `full-model/result.json` under `samples`.

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b32 | 1059.2 | 1006.5 | **1.0511×** | 1.0303× | 30233.0 | 31612.6 | yes |
| mlp_b8 | 701.0 | 649.2 | **1.0781×** | 1.0397× | 10292.1 | 10974.2 | yes |
| mlp_b1 | 601.1 | 549.2 | **1.0954×** | 1.0472× | 1571.3 | 1703.8 | yes |
| vit_b8 | 3375.3 | 3362.4 | 1.0030× | 1.0014× | 2372.6 | 2381.5 | yes |
| vit_b1 | 1442.4 | 1438.9 | 1.0021× | 1.0010× | 707.2 | 710.2 | yes |

All five profiles use the same compiled graphs, weights and timed interval.
Accuracy is identical and every logit tensor is bitwise equal. The ViT
profiles have no multi-submit workspace in the default graph; their ratios are
neutral, as expected.

## Validation

- `command_cache_test`: **270 checks / 0 failures**.
- `rk_plan_host_test`: multi-submit cache, partial-budget fallback and
  original-path outputs pass.
- CTest: **33/33 passed**.
- pytest: **352 passed, 71 subtests passed**.
- Frozen numeric acceptance replay: **PASS / 0 problems**.
- Default-on `bmulti`: 8/8 submits cached, 8 unique command BOs, command bytes
  unchanged after 1,024 cached executions.
- No hardware recipe, numeric policy or asynchronous execution changed.

## Decision

Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` with the 4 MiB default budget.
`RK_PLAN_MULTI_SUBMIT_CACHE=0` remains the control. All earlier single-submit
and command-cache results remain unchanged historical evidence.

## Files

- `pair/`: isolated MLP three-arm logs with per-call `WARM_US` samples.
- `attr/`: emit/command/output-sync attribution per arm.
- `baseline-cost/`: pre-candidate current-runtime snapshot.
- `cmdbytes/`: default-on `bmulti` memory, handle and byte-stability evidence.
- `full-model/`: five-profile interleaved result, logits and per-call samples.
- `numeric/`: frozen acceptance observation and verifier output.
- `checks/`: command-cache/rk_plan C tests, CTest, pytest and failure logs.
- `artifacts/`: final board library, probe, environment and build logs.
