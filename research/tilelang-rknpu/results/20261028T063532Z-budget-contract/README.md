# Plan-budget contract and legal multi-submit cache

This package closes the budget-contract gap between compile options, plan
persistence/hash/loading and the C descriptor, then re-evaluates the
multi-submit cache only under explicitly authorized headroom.

## Budget contract

`CompileOptions.device_budget_bytes` / `host_budget_bytes` are now written to a
versioned `budget_policy` block:

```json
"budget_policy": {
  "schema_version": 1,
  "device_budget_bytes": 268435456,
  "host_budget_bytes": 268435456
}
```

The block is part of the canonical plan hash, survives serialize/load, and is
projected by `build_plan_descriptor()` into the existing C fields:

- `device_budget_bytes` / `host_budget_bytes` = explicit user-authorized limits;
- `claimed_device_bytes` / `claimed_host_bytes` = accounting estimate.

No public C ABI field was added or renumbered.

Legacy plan JSON without `budget_policy` loads with `budget=None`; its
descriptor budget remains equal to the accounting estimate, preserving the
historical strict semantics. It is not implicitly filled with 256 MiB.

## Pre-allocation checks

`build_multi_submit_cache()` now checks the three limits before each persistent
allocation:

1. cache-own budget: Host entries/programs + page-aligned command bytes;
2. workspace Host room: plan Host budget minus current
   `rk_workspace_host_bytes(ws, 1)`;
3. device room: plan device budget minus data-slot requests and the
   still-allocated fallback command workspace request.

The `entries` array is rejected before allocation if it cannot fit Host room.
Each program slice is first emitted into the existing `batch_scratch` as a cost
probe; only after Host/device/cache checks pass does the code allocate the
persistent slice, re-emit into it, and build the command BO.

`RK_PLAN_MULTI_SUBMIT_BUDGET` fails closed for negative, signed, non-numeric,
trailing-garbage, overflowing and zero values.

## Budget-boundary behavior (MLP b32)

| variant | cache state | cached | skipped budget/static | Host delta | cache request | cache physical | fallback command request/physical |
|---|---|---:|---|---:|---:|---:|---|
| exact | fallback | 0/8 | 8/0 | 0 B | 0 B | 0 B | 65,536/65,536 B |
| partial | prepared | 1/8 | 7/0 | 92,160 B | 4,032 B | 4,096 B | 65,536/65,536 B |
| full | prepared | 8/8 | 0/0 | 208,528 B | 11,088 B | 32,768 B | 65,536/65,536 B |

The exact-budget plan leaves no room beyond data slots + the fallback command
workspace, so zero cache BOs are allocated. The partial plan leaves 4,032
device request bytes, exactly enough for the first 4-task submit. The full plan
uses the documented `CompileOptions` default 256 MiB limits; no budget was
raised to recover the cache.

Request bytes follow the plan's declared accounting. Physical bytes are the
per-BO 4 KiB page-aligned upper bound: the eight full-cache command BOs are
32,768 B physically, not 11,088 B. The fallback command workspace remains
allocated and charged.

All exact/partial/full variants produce bitwise-identical output.

## Legal five-profile comparison

Interleaved two-runner run on the budgeted MLP plans + unchanged ViT plans,
256 images / 7 rounds; per-call samples are in `full-model/result.json`.

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b32 | 1073.8 | 1037.3 | **1.0538×** | 1.0247× | 28172.9 | 28377.8 | yes |
| mlp_b8 | 1005.9 | 969.0 | **1.0386×** | 1.0357× | 7974.1 | 8268.5 | yes |
| mlp_b1 | 898.3 | 859.2 | **1.0459×** | 1.0449× | 1227.4 | 1298.7 | yes |
| vit_b8 | 3391.6 | 3395.7 | 0.9999× | 0.9975× | 2373.4 | 2371.5 | yes |
| vit_b1 | 1477.1 | 1470.0 | 1.0059× | 1.0053× | 694.7 | 698.9 | yes |

Accuracy is identical and every logit tensor is bitwise equal. ViT profiles
have no multi-submit workspace in the default graph and remain neutral.

## Validation

- `plan-budget-contract`: 4 tests including compile/persist/hash/load/hybrid
  manifest/C-descriptor separation and legacy strict behavior.
- `rk_plan_host_test`: 875 checks, including exact-full device budget,
  Host-limited partial cache, cache-limited partial cache, malformed budget
  fail-closed and allocation-count assertions.
- `command_cache_test`: 270 checks.
- CTest: 33/33.
- pytest: 356 passed, 71 subtests.
- Frozen numeric replay on the budgeted plan package: **PASS / 0 problems**.
  The derived spec updates only the compiled-manifest identity; thresholds and
  expected arrays are unchanged.

## Decision

Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` as the default. Legacy plans without
`budget_policy` remain strict and safely fall back; new plans with explicit
budget headroom use the cache. The earlier invalid-budget result and the
conservative fallback-only correction remain historical and are not used as
performance evidence.

## Files

- `campaign/`: budgeted MLP plans, hybrid manifests, package manifest and the
  regeneration script.
- `smoke/`: exact/partial/full `bmulti` allocation, skip-reason and memory logs.
- `bitwise/`: exact/partial/full output hashes.
- `full-model/`: legal five-profile interleaved result and per-call samples.
- `numeric/`: budgeted calibration/acceptance observations, derived spec,
  plan package and verifier output.
- `checks/`: plan-budget, command-cache, rk_plan, CTest and pytest logs.
- `artifacts/`: final board library, probe, environment and build logs.
