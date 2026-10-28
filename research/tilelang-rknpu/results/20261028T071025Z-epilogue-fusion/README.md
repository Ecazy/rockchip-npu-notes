# MLP Linear → bias → ReLU operator/compiler optimization

This package compares the high-potential Linear-epilogue options, keeps the
existing numeric narrowing/accumulation policy, and adopts only a generic
Submit-envelope fusion that is bitwise equivalent.

## Candidates and disposition

| candidate | structural effect | numeric effect | disposition |
|---|---|---|---|
| Fuse bias into the GEMM DPU BS stage | removes the bias EW task | moves the FP16 bias add before the final narrowing/accumulation order; not bitwise with the current policy | rejected for default; would need a new numeric policy and a separate acceptance |
| Fuse bias+ReLU as one `max(x, -b)` EW task | removes one task | `fp16(x+b)` threshold is not equivalent to `x >= -b` in general; rounding boundary differs | rejected for default; requires an explicit approximation policy |
| Change K tiling / GEMM template to fewer tasks | fewer tasks and less intermediate traffic | changes the declared FP16 writeback per K tile / accumulation semantics | rejected; current K=256 policy is the control |
| Generic Submit-envelope fusion (`fuse_epilogue`) | 8→3 Submits, same 11 tasks, same data path | none: same tasks in the same order, same templates, buffers and transfers | **adopted** |

The hardware does expose DPU BS bias and EW max/min, but same hardwaresupport
does not imply numeric equivalence with the current separate FP16 bias-add and
FP16 ReLU tasks. No new hardware recipe or numeric policy is enabled here.

## Generic rule

A directly dependent `MM/MM_ACC` chain may absorb an immediately following
`EW_ADD` and `EW_MAX` when all tasks are contiguous in action order and the
intermediate has no other consumer. The same rule can merge a standalone
`EW_ADD` + `EW_MAX` pair. It is triggered by template/dependency/action-order
predicates, not model names. `fuse_epilogue=False` preserves the original
per-stage lowering as the control.

The C validator admits only:
- `MM` / `MM_ACC*` followed by optional `EW_ADD` then optional `EW_MAX`, with
  each epilogue task directly depending on the previous task;
- `EW_ADD` followed by `EW_MAX`, with the same direct dependency.

## Structural result (MLP b32)

| metric | control | fused |
|---|---:|---:|
| NPU tasks | 11 | 11 |
| Submits | 8 | **3** |
| command workspace | 65,536 B | 98,304 B |
| device request | 1,627,328 B | 1,660,096 B |
| Host estimate | 1,009,612 B | 1,042,860 B |
| peak payload / reserved | 937,984 / 1,561,792 B | 937,984 / 1,561,792 B |
| data BOs | 19 | 19 |
| transfers | 16 | 16 |
| tensor views / padding / uploads / downloads | unchanged | unchanged |

The fused plan pays +32,768 B device request and +33,248 B Host for the larger
worst-case fallback command workspace (6 tasks vs 4), but the data path is
byte-identical: same BO count, peak payload, allocator slack, tensor views,
upload/download count and transfer count. With multi-submit cache on, the fused
plan needs 3 command BOs / 16 KiB physical instead of 8 / 32 KiB.

## Multi-submit-cache behavior

| plan | cached submits | command request | command physical | Host delta | fallback command request/physical |
|---|---:|---:|---:|---:|---|
| control | 8/8 | 11,088 B | 32,768 B | 208,528 B | 65,536/65,536 B |
| fused | 3/3 | 11,088 B | 16,384 B | 192,488 B | 98,304/98,304 B |

## Isolated MLP timing

| plan | total µs | submit_wait µs | output_sync µs | pack µs |
|---|---:|---:|---:|---:|
| control b32 | 548.2 | 185.5 | 24.9 | 158.1 |
| fused b32 | 499.3 | 139.7 | 22.2 | 158.1 |
| control b1 | 394.0 | 175.0 | 25.7 | 11.7 |
| fused b1 | 346.5 | 130.7 | 22.8 | 11.7 |

Submit wait falls by about 45 µs; pack/layout and transfers are unchanged.

## Full-model interleaved result

256 images, 7 rounds, two independent runs, same fd and multi-submit-cache
policy. Ratios combine all 14 round ratios; per-call samples are in
`full-model/run1/result.json` and `full-model/run2/result.json`.

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.1413×** | 1.0498× | 1.2348× | 999.5→949.9 | 1307.1→1061.7 | yes |
| mlp_b8 | **1.0845×** | 1.0663× | 1.3616× | 647.6→597.0 | 644.3→598.5 | yes |
| mlp_b1 | **1.4185×** | 1.1038× | 1.4205× | 863.3→608.4 | 861.0→606.7 | yes |
| vit_b8 | 1.0026× | 0.9991× | 1.0061× | 3323.1→3313.7 | 3060.0→3053.6 | yes |
| vit_b1 | 1.0039× | 0.9075× | 1.0072× | 1433.2→1419.4 | 1394.3→1437.0 | yes |

Accuracy is identical; every logit tensor is bitwise equal. ViT has no
Linear-epilogue fusion in its default graph and remains neutral; its one
sub-1.0 round is process noise, not a claimed regression.

## Validation

- `test_epilogue_fusion`: control 8 submits / fused 3 submits, same 11 task
  templates, same transfers, same accounting exchange; mock execution bitwise.
- CTest: 33/33.
- pytest: 357 passed, 71 subtests.
- `rk_plan_host_test`: 875 checks / 0 failures.
- `command_cache_test`: 270 checks / 0 failures.
- Frozen numeric replay on the fused plan package: **PASS / 0 problems**.
  The derived spec updates only compiled-manifest identity; thresholds and
  expected arrays are unchanged.

## Decision

Adopt `fuse_epilogue=True` as the compile default for new plans. The original
lowering remains available through `fuse_epilogue=False` and is the control
used above. No runtime/cache/budget expansion and no model-name special case
were introduced.

## Files

- `campaign/control/`, `campaign/fused/`: MLP plan/hybrid/manifest packages and
  the budgeted-campaign builder.
- `smoke/`: multi-submit-cache allocation logs for control/fused.
- `battr/`: isolated per-stage timing logs.
- `full-model/run1`, `full-model/run2`: two interleaved five-profile runs with
  per-call samples and logits tensors.
- `numeric/`: fused calibration/acceptance observations, derived spec, plan
  package and verifier output.
- `checks/`: epilogue, CTest, pytest, command-cache and rk_plan logs.
- `artifacts/`: final board library/probe and build logs.
  - runtime SHA-256:
    `677163cb54409831e22fe1c4f7d00c1f931b94a3af324b5b4cb6b8f2c014762b`
  - probe SHA-256:
    `c345cc114e66532de92c17fe6aedbb78f2f6ff2046bf6cf5b5b6d49a7bd1ccb3`
