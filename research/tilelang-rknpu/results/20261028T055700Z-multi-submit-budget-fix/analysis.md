# Multi-submit cache: budget-closure correction

The candidate is retained as opt-in code, but **not adopted as a
default**.  The prior positive result remains in the historical
`20261028T035849Z-multi-submit-cache/` directory; its performance
claims are withdrawn because its separate command BOs exceeded the
plans' declared device budgets.

## Budget admission

A cache submit is admitted only when all three limits hold:

1. `RK_PLAN_MULTI_SUBMIT_BUDGET`: Host entries/programs plus
   page-aligned command BO bytes;
2. plan Host budget minus the already-resident workspace Host bytes;
3. plan device budget minus data-slot requests and the still-allocated
   fallback command workspace.  This uses the declared request-byte
   accounting.

Malformed, negative, overflowing and zero budget values fail closed
(0 = cache disabled), never silently become the default or an
unbounded value.

## Corrected MLP smoke

| case | cache state | cached submits | host delta | cache request | cache physical | fallback command request/physical |
|---|---|---:|---:|---:|---:|---|
| mlp_b32 | fallback | 0/8 | 0 B | 0 B | 0 B | 65536/65536 B |
| mlp_b1 | fallback | 0/8 | 0 B | 0 B | 0 B | 65536/65536 B |

The fallback command workspace (65,536 B request and physical for
MLP) is still allocated and is charged in the device-budget check.
The cache therefore allocates no command BO on current plans.

## Short five-profile confirmation (both arms fall back)

Interleaved two-runner result, 256 images / 3 rounds:

| profile | control µs | opt-in µs | on/off | bitwise |
|---|---:|---:|---:|---|
| mlp_b1 | 902.1 | 900.1 | 1.0029× | yes |
| mlp_b32 | 1355.9 | 1353.3 | 1.0016× | yes |
| mlp_b8 | 1001.9 | 1000.1 | 1.0022× | yes |
| vit_b1 | 1198.4 | 1179.9 | 1.0154× | yes |
| vit_b8 | 3112.5 | 3093.1 | 1.0055× | yes |

With the corrected admission check the opt-in arm takes the original
per-submit path, so the ratio is neutral and outputs remain bitwise.
This confirms safe fallback, not a performance gain.

## Budget-boundary regressions

`rk_plan_host_test` now covers, on a synthetic multi-submit plan:

- device budget exactly equal to base slots + fallback command: zero
  cache BO allocations, original output bitwise;
- Host budget tight: only 3 of 4 submits fit, 3 cache BOs allocated,
  `rk_workspace_host_bytes` stays within the plan Host budget, output
  bitwise;
- cache-own budget tight: only 1 of 4 submits fits, one cache BO
  allocated, output bitwise;
- `-1`, non-numeric, zero and overflowing budget strings: no cache,
  base allocation count only, output bitwise.

The tests assert the mock allocation count, so an over-budget attempt
cannot be hidden by allocating then releasing.

## Validation

- command-cache host test: `command_cache: 270 checks, 0 failures`
- rk_plan host: `RK_PLAN host checks=875 failures=0`
- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `352 passed, 71 subtests passed in 30.58s`
- frozen numeric replay: **PASS**, problems: []

## Decision

Do **not** adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` as the default.  Keep it
as an opt-in experiment for workspaces that explicitly declare device
headroom; current compiled plans fall back.  The single-submit command
cache and all earlier results remain unchanged.

