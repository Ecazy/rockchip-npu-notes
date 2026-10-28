# Multi-submit cache: budget-closure correction

This directory is the corrected budget record for the opt-in multi-submit
static-preparation candidate. The earlier positive result
`../20261028T035849Z-multi-submit-cache/` remains untouched as historical
evidence, but its MLP speedups are **withdrawn**: the separate cache command
BOs were not charged against the plans' declared device budgets.

## Decision

`RK_PLAN_MULTI_SUBMIT_CACHE` is **not adopted as a default**. The implementation
remains as an opt-in experiment (`=1`) for future plans that explicitly declare
device-budget headroom. Current compiled MLP plans have no such headroom, so
the prepared path falls back and the measured result is neutral.

## Three-budget admission

Before every cache allocation:

1. **cache-own budget**: Host entry/program bytes plus page-aligned command BO
   bytes <= `RK_PLAN_MULTI_SUBMIT_BUDGET` (default 4 MiB/workspace);
2. **workspace Host room**: current `rk_workspace_host_bytes(ws, 1)` plus the
   new Host bytes <= the plan Host budget;
3. **device room**: base data-slot request bytes plus the still-allocated
   fallback command workspace plus the new command request <=
   `plan->device_budget_bytes`.

The fallback command workspace is not treated as replaced. For MLP it remains
allocated at **65,536 B request/physical**, and the base device request already
exhausts the declared budget. Therefore the cache allocates no command BO.

## Device byte accounting

- Plan budget validation uses **request bytes**.
- The cache-own cap uses **page-aligned physical bytes** per command BO, because
  the provider rounds each BO to 4 KiB.
- If the candidate were active on current MLP geometry, its historical request
  total 11,088 B would be eight separate BOs and therefore a 32,768 B physical
  upper bound. Under the corrected budget check it is not allocated at all
  (request/physical increment 0 B).

## Budget parsing

`RK_PLAN_MULTI_SUBMIT_BUDGET` fails closed: unset/empty uses the default;
negative, signed, whitespace/word, trailing-garbage, overflowing and explicit
zero values disable the cache. They never become an unbounded budget or fall
back to the default.

## Regressions

`rk_plan_host_test` (875 checks) uses a synthetic four-submit plan and asserts
mock BO allocation counts, so an over-budget attempt cannot be hidden by an
allocate-then-free pattern:

- device budget exactly equals data slots + fallback command: 0 cache BOs;
- Host budget tight: 3/4 submits cached, Host footprint remains within budget;
- cache-own budget tight: 1/4 submits cached;
- `-1`, non-numeric, `0`, overflow: no cache, base allocation count only;
- every case compares the output bitwise with the original path.

`command_cache_test` (270 checks) covers independent command BOs, legal/illegal
`CO_SEPARATE`, zero submits for a later-invalid program, submit failure and
build-failure cleanup.

## Corrected MLP smoke

| case | cache state | cached | host delta | cache request | cache physical | fallback command request/physical |
|---|---|---:|---:|---:|---:|---|
| MLP b32 | fallback | 0/8 | 0 B | 0 B | 0 B | 65,536 B / 65,536 B |
| MLP b1 | fallback | 0/8 | 0 B | 0 B | 0 B | 65,536 B / 65,536 B |

## Short five-profile confirmation (both arms fall back)

Interleaved two-runner run, 256 images / 3 rounds; per-call samples are in
`full-model/result.json`:

| profile | control µs | opt-in µs | on/off | bitwise |
|---|---:|---:|---:|---|
| mlp_b32 | 1355.9 | 1353.3 | 1.0016× | yes |
| mlp_b8 | 1001.9 | 1000.1 | 1.0022× | yes |
| mlp_b1 | 902.1 | 900.1 | 1.0029× | yes |
| vit_b8 | 3112.5 | 3093.1 | 1.0055× | yes |
| vit_b1 | 1198.4 | 1179.9 | 1.0154× | yes |

This is a neutral safety confirmation, not a performance claim.

## Validation

- `command_cache_test`: **270 checks / 0 failures**
- `rk_plan_host_test`: **875 checks / 0 failures**
- CTest: **33/33 passed**
- pytest: **352 passed, 71 subtests passed**
- Frozen numeric replay: **PASS / 0 problems**

## Files

- `smoke/`: `bmulti` fallback and memory/command-workpace evidence.
- `pair/`: short MLP off/on logs with per-call `WARM_US` samples.
- `full-model/`: five-profile interleaved neutral confirmation and samples.
- `numeric/`: frozen replay observation and verifier output.
- `checks/`: command-cache/rk_plan C tests, CTest, pytest.
- `artifacts/`: corrected board library, probe, environment and build logs.
