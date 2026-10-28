# Add + ReLU primitive fusion

Baseline: `963fd0d` Submit grouping fusion + multi-submit cache. The control
keeps the original two-task `EW_ADD` → `EW_MAX` path; the candidate replaces
it with `co.ew_add_relu.f16.v1`, clearing only `EW_RELU_BYPASS` (bit 9) while
leaving the identity-conv main feed, ERDMA operand, EDATA_SIZE and OUT_CVT
unchanged. `fuse_add_relu=False` preserves the control path.

## Candidate analysis

- The fused ReLU comparator is in the EW stage before OUT_CVT; the control
  writes an fp16 Add result and reads it in a second EW MAX.
- No `max(x, -b)` substitution and no bias-into-GEMM change.
- Hardware probe uses exact input bits over signed zero, subnormal/min-normal,
  max-finite, cancellation, near-cancellation, random finite and separate
  NaN/Inf.
  - 8192 elements: 0 mismatches.
  - 1,000,000 elements: 0 mismatches.
  - Signed-zero pairs: both paths output `+0x0000`.
  - Invalid `ew_relu` combinations are rejected by the generator.
- Probe layout uses the real NC1HWC2 cube, identity weight, `M>=12`, `M%4`
  and CBUF guard padding; only valid rows are read.

## Structural result (MLP b32)

| metric | control | fused |
|---|---:|---:|
| NPU tasks | 11 | 9 |
| Submits | 3 | 3 |
| data BOs | 19 | 15 |
| transfers | 16 | 14 |
| peak payload | 937,984 B | 888,832 B |
| reserved | 1,561,792 B | 1,381,312 B |
| command workspace | 98,304 B | 81,920 B |
| device request | 1,660,096 B | 1,463,232 B |
| Host estimate | 1,042,860 B | 1,001,660 B |

The two removed NPU tasks are the two ReLUs; the four data BOs are the two Add
intermediates plus two zero-constant BOs. Submits stay at the previous
package's 3, so this arm does not re-count the Submit grouping gain.

## Multi-submit-cache allocation

| arm | cached | request | physical | Host delta | fallback command request/physical |
|---|---:|---:|---:|---:|---|
| control | 3/3 | 11,088 B | 16,384 B | 192,488 B | 98,304/98,304 B |
| fused | 3/3 | 9,072 B | 16,384 B | 159,240 B | 81,920/81,920 B |

## Isolated MLP timing

| arm | total µs | submit_wait µs | output_sync µs | guard µs | pack µs |
|---|---:|---:|---:|---:|---:|
| control b32 | 602.6 | 253.2 | 21.0 | 157.5 | 156.3 |
| fused b32 | 548.0 | 232.2 | 15.5 | 129.8 | 156.3 |
| control b1 | 438.1 | 244.4 | 19.2 | 159.2 | 10.8 |
| fused b1 | 389.4 | 226.6 | 14.0 | 133.0 | 11.1 |

## Full-model interleaved result

256 images, 7 rounds, two independent runs, same fd and multi-submit policy.

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.0558×** | 1.0511× | 1.0595× | 1059.0→1002.4 | 1060.3→1006.7 | yes |
| mlp_b8 | **1.0783×** | 1.0714× | 1.0827× | 707.6→656.2 | 710.8→660.6 | yes |
| mlp_b1 | **1.0957×** | 1.0936× | 1.1260× | 607.5→555.0 | 506.5→452.7 | yes |
| vit_b8 | 1.0010× | 0.9985× | 1.0100× | 3366.1→3366.5 | 3081.0→3075.9 | yes |
| vit_b1 | 1.0073× | 1.0040× | 1.0990× | 1457.4→1445.6 | 1472.8→1462.1 | yes |

Accuracy is identical and every logit tensor is bitwise equal. ViT shows no
stable change; the reversing-round cause has not been identified and is not
attributed to process noise.

## Validation

- `test_epilogue_fusion`: 2 tests covering Submit grouping and primitive
  fusion; the latter asserts 11→9 tasks, 19→15 BOs, 16→14 transfers and
  bitwise mock output.
- CTest: 33/33.
- pytest: 358 passed, 71 subtests.
- `rk_plan_host_test`: 875 checks / 0 failures.
- `command_cache_test`: 270 checks / 0 failures.
- Frozen numeric replay on fused plans: PASS / 0 problems.
- Hardware probe: 8192 and 1,000,000 elements, 0 mismatches; invalid
  combinations rejected.

## Decision

Adopt `fuse_add_relu=True` as the compile default. The original two-task path
is retained via `fuse_add_relu=False`. No runtime/cache/budget expansion and no
model-name special case.

## Files

- `probe/`: hardware boundary-suite logs at 8192 and 1,000,000 elements.
- `campaign/control`, `campaign/fused`: MLP plan/hybrid/manifest packages.
- `smoke/`, `battr/`: cache allocation and isolated timing logs.
- `full-model/run1`, `full-model/run2`: two interleaved five-profile runs.
- `numeric/`: fused calibration/acceptance observations, derived spec,
  plan package and verifier output.
- `checks/`: CTest, pytest, epilogue-fusion, command-cache, rk_plan logs.
- `artifacts/`: final board library/probe and build logs.

## Identity

- runtime SHA-256:
  `e5eeeb4e510051ca0d36e39f1443a6828f6597804e7ff3d4f74d27c80e79c61e`
- runtime probe SHA-256:
  `9bd3198ca5cd8c7c40e453a7641601ce33152d6987d9f498c7ef3d814615c009`
- primitive probe SHA-256:
  `a1026160102b10feeb2d35ca14c164defdab837baafa75e3488f9452890a8346`
