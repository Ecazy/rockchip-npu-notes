# Add + ReLU primitive fusion

Baseline: `963fd0d` Submit grouping fusion + multi-submit cache. The
control keeps the original two-task `EW_ADD` → `EW_MAX` path. The
candidate replaces it only when the add value is a non-exported,
single-consumer temporary, using `co.ew_add_relu.f16.v1`.

## Hardware path

`EW_CFG_ALU(ADD)` sets `EW_RELU_BYPASS` bit 9 and leaves
`DPU_EW_RELUX_CMP_VALUE` at zero. The candidate clears only bit 9; the
identity-conv main feed, ERDMA operand, EDATA_SIZE and OUT_CVT stay
unchanged. The hardware probe compares this against the separate
FP16 add then FP16 max path with exact input bits.

## Probe

| elements | mismatches | finite | special | control µs | fused µs |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0 | 0 | 0 | 438.7 | 324.3 |
| 1000000 | 0 | 0 | 0 | 29176.0 | 26140.0 |

The suite includes every finite pair from a boundary set (signed zero,
min/max subnormal, min normal, exact cancellation, max finite),
near-cancellation, deterministic random finite values, and separate
NaN/Inf cases. Signed-zero pairs all report `+0` for both paths. The
generator rejects `ew_relu` with no accumulation, MAX, or MUL.

## Structural result (MLP)

| plan | tasks | submits | data BOs | transfers | peak payload | reserved | command workspace | device request | Host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b1 | 11→9 | 3→3 | 19→15 | 16→14 | 829184→810752 B | 1452992→1303232 B | 98304→81920 B | 1551296→1385152 B | 928780→911388 B |
| b8 | 11→9 | 3→3 | 19→15 | 16→14 | 829184→810752 B | 1452992→1303232 B | 98304→81920 B | 1551296→1385152 B | 954540→931772 B |
| b32 | 11→9 | 3→3 | 19→15 | 16→14 | 937984→888832 B | 1561792→1381312 B | 98304→81920 B | 1660096→1463232 B | 1042860→1001660 B |

For b32 the primitive removes 2 NPU tasks, 4 data BOs, 2 transfers,
49,152 B peak payload, 180,480 B reserved device memory and 41,200 B
Host estimate. Submits stay at the 3 achieved by the prior grouping
fusion, so this arm does not double-count that benefit.

## Multi-submit cache

| arm | cached | command request | command physical | Host delta | fallback command request/physical |
|---|---:|---:|---:|---:|---|
| control | 3/3 | 11088 B | 16384 B | 192488 B | 98304/98304 B |
| fused | 3/3 | 9072 B | 16384 B | 159240 B | 81920/81920 B |

## Isolated MLP timing

| arm | total µs | submit_wait µs | output_sync µs | guard µs | pack µs |
|---|---:|---:|---:|---:|---:|
| control b32 | 602.6 | 253.2 | 21.0 | 157.5 | 156.3 |
| control b1 | 438.1 | 244.4 | 19.2 | 159.2 | 10.8 |
| fused b32 | 548.0 | 232.2 | 15.5 | 129.8 | 156.3 |
| fused b1 | 389.4 | 226.6 | 14.0 | 133.0 | 11.1 |

## Full-model interleaved comparison

256 images, 7 rounds, two independent runs, same fd and multi-submit
policy. Ratios combine all 14 round ratios.

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.0558×** | 1.0511× | 1.0595× | 1059.0→1002.4 | 1060.3→1006.7 | yes |
| mlp_b8 | **1.0783×** | 1.0714× | 1.0827× | 707.6→656.2 | 710.8→660.6 | yes |
| mlp_b1 | **1.0957×** | 1.0936× | 1.1260× | 607.5→555.0 | 506.5→452.7 | yes |
| vit_b8 | **1.0010×** | 0.9985× | 1.0100× | 3366.1→3366.5 | 3081.0→3075.9 | yes |
| vit_b1 | **1.0073×** | 1.0040× | 1.0990× | 1457.4→1445.6 | 1472.8→1462.1 | yes |

MLP improves in every run; ViT shows no stable change. Accuracy is
identical and every logit tensor is bitwise equal.

## Validation

- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `358 passed, 71 subtests passed in 31.01s`
- epilogue/primitive tests: `..                                                                       [100%]
2 passed in 1.24s`
- command cache: `command_cache: 270 checks, 0 failures`
- rk_plan host: `RK_PLAN host checks=875 failures=0`
- frozen numeric replay on fused plans: **PASS**, problems: []

## Decision

Adopt `fuse_add_relu=True` as the compile default. Keep the original
two-task path available through `fuse_add_relu=False`. No model-name
branch and no runtime/cache/budget expansion were introduced.

