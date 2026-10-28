# Linear → bias → ReLU Submit grouping fusion

Generic rule: a directly dependent `MM/MM_ACC` chain may absorb an
immediately following `EW_ADD` and `EW_MAX` when the actions are
contiguous in plan order. The tasks, templates, arithmetic, buffers,
transfers and numeric narrowing are unchanged; only the Submit
envelope is reduced. `fuse_epilogue=False` keeps the original
per-stage lowering as the control.

## Structural comparison (MLP)

| plan | tasks | submits control→fused | command workspace | device request | host bytes | peak/reserved | BOs | transfers | data path identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b1 | 11 | 8→3 | 65536→98304 B | 1518528→1551296 B | 895532→928780 B | 829184/1452992 B | 19 | 16 | yes |
| b8 | 11 | 8→3 | 65536→98304 B | 1518528→1551296 B | 921292→954540 B | 829184/1452992 B | 19 | 16 | yes |
| b32 | 11 | 8→3 | 65536→98304 B | 1627328→1660096 B | 1009612→1042860 B | 937984/1561792 B | 19 | 16 | yes |

The fused plan pays +32,768 B device request and +33,248 B Host for the
larger worst-case fallback command workspace (6 tasks instead of 4),
but the data path is byte-identical: same 19 data BOs, same peak payload,
same allocator slack, same upload/download count and same tensor views.
With multi-submit cache on, the fused plan needs 3 command BOs / 16 KiB
physical instead of 8 command BOs / 32 KiB, offsetting part of that
fallback increase.

## Multi-submit-cache allocation

| plan | cache | cached | request | physical | Host delta | fallback command request/physical |
|---|---|---:|---:|---:|---:|---|
| control | prepared | 8/8 | 11088 B | 32768 B | 208528 B | 65536/65536 B |
| fused | prepared | 3/3 | 11088 B | 16384 B | 192488 B | 98304/98304 B |

## Isolated MLP timing

| plan | total µs | submit_wait µs | output_sync µs | pack µs |
|---|---:|---:|---:|---:|
| control b32 | 548.2 | 185.5 | 24.9 | 158.1 |
| control b1 | 394.0 | 175.0 | 25.7 | 11.7 |
| fused b32 | 499.3 | 139.7 | 22.2 | 158.1 |
| fused b1 | 346.5 | 130.7 | 22.8 | 11.7 |

Submit wait falls by about 45 µs on both b32 and b1; pack and data
movement are unchanged.

## Full-model interleaved comparison

256 images, 7 rounds, two independent runs, same fd and multi-submit
cache policy. Ratios are control/fused from all 14 rounds.

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.1413×** | 1.0498× | 1.2348× | 999.5→949.9 | 1307.1→1061.7 | yes |
| mlp_b8 | **1.0845×** | 1.0663× | 1.3616× | 647.6→597.0 | 644.3→598.5 | yes |
| mlp_b1 | **1.4185×** | 1.1038× | 1.4205× | 863.3→608.4 | 861.0→606.7 | yes |
| vit_b8 | **1.0026×** | 0.9991× | 1.0061× | 3323.1→3313.7 | 3060.0→3053.6 | yes |
| vit_b1 | **1.0039×** | 0.9075× | 1.0072× | 1433.2→1419.4 | 1394.3→1437.0 | yes |

MLP improves in every run; ViT has no Linear-epilogue fusion in its
default graph and shows no stable change; the cause of its reversing
round is not yet identified and is not attributed to process noise.
Accuracy is identical and every logit tensor is bitwise equal.

## Validation

- CTest: `100% tests passed, 0 tests failed out of 33`
- pytest: `357 passed, 71 subtests passed in 30.63s`
- epilogue fusion test: `.                                                                        [100%]
1 passed in 1.10s`
- command cache: `command_cache: 270 checks, 0 failures`
- rk_plan host: `RK_PLAN host checks=875 failures=0`
- frozen numeric replay on fused plans: **PASS**, problems: []

## Decision

Adopt the generic `fuse_epilogue=True` compile default for new plans.
The original lowering remains available with `fuse_epilogue=False` and
is the control used above. No model-name branch, no changed numeric
policy and no runtime expansion were introduced.

