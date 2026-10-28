# Plan-budget contract and legal multi-submit cache

`CompileOptions.device_budget_bytes` / `host_budget_bytes` are now
persisted as a versioned `budget_policy` block, included in the plan
hash, loaded back, and projected into the existing C descriptor
`device_budget_bytes` / `host_budget_bytes` fields. `claimed_*` remains
the accounting estimate. Legacy plans without `budget_policy` keep
strict budget = claimed semantics.

## Budgeted MLP plans

| plan | budget device/host | accounting device/host | plan hash |
|---|---:|---:|---|
| mnist_mlp_b1 | 268435456/268435456 | 1518528/895532 | `ee8ac3e8ab4f661f…` |
| mnist_mlp_b8 | 268435456/268435456 | 1518528/921292 | `d295f0c028a4d604…` |
| mnist_mlp_b32 | 268435456/268435456 | 1627328/1009612 | `3efd5e8dbe803938…` |

The full campaign uses the documented `CompileOptions` default
256 MiB limits; no budget was raised to recover the cache.

## Budget boundary behavior

| variant | cache state | cached | skipped budget/static | Host delta | cache request | cache physical | fallback command request/physical |
|---|---|---:|---|---:|---:|---:|---|
| exact | fallback | 0/8 | 8/0 | 0 B | 0 B | 0 B | 65536/65536 B |
| partial | prepared | 1/8 | 7/0 | 92160 B | 4032 B | 4096 B | 65536/65536 B |
| full | prepared | 8/8 | 0/0 | 208528 B | 11088 B | 32768 B | 65536/65536 B |

Exact budget uses all device bytes for data slots + the fallback
command workspace; zero cache BOs are allocated. Partial budget leaves
4,032 request bytes: the first 4-task submit fits (4,032 request /
4,096 physical), the other seven are skipped. Full budget caches all
eight: 11,088 request bytes and a 32,768 B page-aligned physical upper
bound. All three variants produce bitwise-identical output.

## Five-profile interleaved comparison

256 images / 7 rounds; per-call samples are in `full-model/result.json`.

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b1 | 898.3 | 859.2 | **1.0459×** | 1.0449× | 1227.4 | 1298.7 | yes |
| mlp_b32 | 1073.8 | 1037.3 | **1.0538×** | 1.0247× | 28172.9 | 28377.8 | yes |
| mlp_b8 | 1005.9 | 969.0 | **1.0386×** | 1.0357× | 7974.1 | 8268.5 | yes |
| vit_b1 | 1477.1 | 1470.0 | **1.0059×** | 1.0053× | 694.7 | 698.9 | yes |
| vit_b8 | 3391.6 | 3395.7 | **0.9999×** | 0.9975× | 2373.4 | 2371.5 | yes |

MLP gains are 1.034–1.054×; ViT profiles have no multi-submit
workspace and stay neutral. Accuracy is identical and logits are
bitwise equal.

## Validation

- command-cache host: `command_cache: 270 checks, 0 failures`
- rk_plan host: `RK_PLAN host checks=875 failures=0`
- plan budget contract: `....                                                                     [100%]
4 passed in 0.99s`
- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `356 passed, 71 subtests passed in 30.94s`
- frozen numeric replay: **PASS**, problems: []

## Decision

Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` as default. Legacy plans without
`budget_policy` remain strict and fall back; new plans with explicit
budget headroom use the legal cache. The earlier invalid-budget results
remain historical and are not used as performance evidence.

