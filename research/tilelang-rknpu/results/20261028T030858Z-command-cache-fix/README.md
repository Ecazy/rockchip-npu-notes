# Command-BO cache follow-up: validation and timing fix

This directory is the corrected follow-up to
`../20261028T023818Z-command-cache/`, which remains unchanged as historical
evidence. The two code fixes are:

1. restore whole-array validation before `CO_SEPARATE` fan-out, so a later
   invalid program cannot leave an earlier valid program already submitted;
2. count `rocket_bo_prep` and `rocket_bo_fini` together in the command-sync
   probe, and report `total_us` as one continuous outer interval rather than a
   sum of incomplete sub-intervals. Public `command_us` again covers the same
   interval as before the refactor. Detail timestamps exist only in the
   independent breakdown probe; the production control path keeps the original
   two-timestamp overhead.

A follow-up micro-fix also avoids re-validating an already fully validated
single-program array inside the packer, so the control path keeps the original
validation count.

## Eligibility and contract

Unchanged from the historical candidate: only `program_cache_ready`
single-submit workspaces, `CHAIN` or one-task `CO_SEPARATE`; command BO,
task descriptors, BO addresses and submit configuration are fixed at creation;
the plan borrows the existing `co_execute_workspace.commands` BO and is freed
with the workspace. All other paths use the original per-submit packer.

## Isolated short A/B (7 rotated rounds)

µs; ratio = control / arm; per-call samples are retained in
`pair/r*/<case>/<arm>.log` as `WARM_US`.

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.4 | 336.6 | 320.2 | **1.0858×** | 1.0792× | 1.0792–1.0973 | yes |
| PV group b8 | 334.2 | 325.8 | 305.7 | **1.0937×** | 0.9727× | 0.9727–1.0982 | yes |
| QKV b8 | 162.7 | 163.0 | 160.4 | **1.0127×** | 1.0035× | 1.0035–1.0401 | yes |
| MLP b32 | 590.3 | 591.5 | 591.5 | 0.9995× | 0.9946× | 0.9946–1.0034 | yes |

The PV outlier round is retained. QK/PV gains remain about 8.6–9.4%; MLP b32
is ineligible (8 submits) and remains neutral. The cache+sync-to-cache ratios
are QK 1.0510×, PV 1.0658× and QKV 1.0164×, which isolates the repeated command
sync.

## Corrected command build breakdown (µs)

| case | clear | pack | verify | handles | sync (PREP+FINI) | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 19.25 | 24.21 | 30.04 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 20.12 | 25.37 | 32.67 |
| QKV b8 | 0.00 | 0.29 | 0.00 | 0.00 | 1.75 | 2.62 | 3.79 |

`sync` now includes both PREP and FINI. `total` is the continuous
pre-PREP-to-post-handle interval and is the value public `command_us` uses; it
is not a sum of the probe sub-intervals. The previous report's claim that an
unexplained end-to-end gap was "cache/sync perturbation" is withdrawn: the
omitted interval was not excluded before making that attribution.

Host accounting is unchanged: **3,184 B** per eligible workspace and **0 B**
of new command BO.

## Short full-model re-test (64 images, 3 rounds)

| model | control | cache+sync | cache | control/cache | p10 | bitwise |
|---|---:|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3536.7 | 3473.6 | 3409.3 | **1.0392×** | 1.0389× | yes |
| Tiny ViT b1 | 1489.5 | 1474.5 | 1453.1 | **1.0268×** | 1.0248× | yes |

The original 7-round/256-image campaign remains historical evidence at
`../20261028T023818Z-command-cache/` (b8 1.0365×, b1 1.0276×). This short
re-test was run with the fixed implementation and confirms the direction.

## Regression evidence

- `CO_SEPARATE` full-array validation: `command_cache_test` now has a case with
  first program valid and second invalid that asserts **zero submits**; the
  legal two-program separate path still performs exactly two submits.
- `command_cache_test`: 259 checks, 0 failures.
- CTest: 33/33 passed.
- pytest: 351 passed, 71 subtests passed.
- Targeted failure atomicity / poison tests pass.
- Frozen numeric acceptance replay: **PASS / 0 problems**.
- `bcmdverify`: command bytes unchanged after 512/512/1024 real submissions
  (32,256 / 32,256 / 1,008 bytes).
- default-on vs `RK_PLAN_COMMAND_CACHE=0` outputs are bitwise equal.

## Decision

Keep `RK_PLAN_COMMAND_CACHE=1` as the default for eligible single-submit
workspaces. The validation and timing corrections do not change the execution
contract; the corrected short A/B still shows a clear benefit.

## Files

- `pair/r1..r7/`: three-arm isolated logs with per-call `WARM_US` samples.
- `bcmd/`: corrected PREP+FINI/total breakdown and host accounting.
- `cmdbytes/`, `defaultcheck/`: command-BO stability and default-on hashes.
- `full-model-b8/`, `full-model-b1/`: short uninstrumented re-test.
- `numeric/`: frozen replay observation and verifier output.
- `checks/`: CTest, pytest, command-cache regression and failure logs.
- `artifacts/`: corrected board library, probe, build logs and environment.
