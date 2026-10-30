# TileLang 后端与完整模型验收

整理日期：2026-10-27。

本文件合并原 MVP 验收矩阵与完整模型里程碑记录。两阶段分别验收，历史测试数量按对应日期解释。当前默认路径、最终性能与厂商对照统一见 [优化总报告](OPTIMIZATION-SUMMARY.md)，支持边界见 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)，运行命令见 [REPLAY.md](REPLAY.md)。

## 内容索引

- [后端 MVP 验收矩阵](#mvp)（原 `ACCEPTANCE-MATRIX.md`）。
- [完整模型里程碑验收](#models)（原 `MODEL-MILESTONE-STATUS.md`）。

---

<a id="mvp"></a>

## 后端 MVP 验收矩阵

原文件：`ACCEPTANCE-MATRIX.md`。以下为该文件历史内容。

<a id="mvp--mvp-acceptance-matrix"></a>
### MVP acceptance matrix

MVP checkpoint evidence: board `results/20261023T144054Z-rk3588/`; final Host
verification `results/20261024T092339Z-host/` (**235 pytest, 41 subtests, 27/27
ctest, 26 contract + 711 host + 2,400,033 graph C checks**, sanitizer enabled).  Status terms are deliberately narrow:
**implemented+verified**, **verification gap**, **explicitly unsupported**,
**out of scope**, and **future extension**.

<a id="mvp--p0p9"></a>
#### P0–P9

| Item | Status | Code | Tests / logs |
|---|---|---|---|
| P0 source/dependency lock | **implemented+verified** | `provenance/source-lock.json`, `requirements.lock`, `scripts/lock_source.py`, `scripts/lock_deps.sh`; `provenance/p0-source-build.json` | source/dependency scopes and isolated-wheel runs are recorded in `P0-SOURCE-LOCK.md`. Locked TileLang/TVM/tvm-ffi/dlpack/libbacktrace archives were independently built with exact commit/hash/source-tree inventory, CMakeCache/compile commands, toolchain and library hashes. Source-built import full pytest 213/40 and CTest 27/27; 27 gate plans semantically equal; same board campaign 81/81. |
| P1 frontend | **implemented+verified for finite accepted forms** | `python/rknpu_backend/frontend/tilelang_adapter.py` | `test_frontend*.py` and `test_staged_tilelang.py`; real multi-block TileLang projection/FFN/branch fixtures now feed the tiled RKIR backend. Same-grid cross-block dependencies, FP32-only K accumulation and inexact block coverage are rejected. Board cases `*_tilelang_staged_chain` pass on all three masks. |
| P2 RKIR/RKPlan/schema/capabilities | **implemented+verified** | `ir.py`, `plan.py`, `schema/*.json`, capability manifest | `test_ir.py`, `test_capabilities.py`, `test_plan_verify.py`; C contract 26/26. |
| P3 planner | **implemented+verified for the finite representative shape list** | `passes/dag.py`, `passes/tile.py`, `passes/storage.py` | `test_lowering.py`, `test_storage.py`, `test_dataflow_multitile.py`, `test_staged_tilelang.py`; representative boundaries include 1/13/257/300/512-wide M/N/K, ragged K=257, M/N multi-tile, odd/even K shards, and staged source shapes `512x256x512`. The 1..4096 limits are rule-checked; exhaustive combination enumeration is explicitly not required. |
| P4 combinations/storage/residual liveness | **implemented+verified for listed nodes** | `passes/legalize.py`, `passes/dag.py`, `passes/storage.py` | `test_dataflow.py`, `test_dataflow_multitile.py`, `test_staged_tilelang.py`, `test_broadcast_execution.py`, `test_multi_output_execution.py`; projection/FFN/branch/ragged/multi-output on board. 3-D/dynamic/state/arbitrary broadcast are explicitly unsupported. |
| P5 C descriptor/FFI | **implemented+verified** | `custom-ops/src/rk_plan_validate.c`, `include/rk_plan.h`, `runtime/ffi.py` | `rk_plan_contract_test` 26 checks, `rk_plan_host_test` 711 checks; `test_ffi.py`, `test_plan_verify.py`. |
| P6 instantiate/execute | **implemented+verified for the declared safety contract** | `custom-ops/src/rk_plan_runtime.c`, `RUNTIME-LIFECYCLE.md` | Pre-submit binding/nonfinite/upload failures are safe; executor errors poison and retain command+data BOs; BUSY and final publication guard regressions pass. Board success path passes. The supported post-failure behavior is quarantine/stop; recovery is out of scope. |
| P7 CLI | **implemented+verified on Host and RK3588** | `python/rknpu_backend/cli.py`, `tools/board_cli_check.py` | `test_cli.py` covers caller/owned fd, cleanup failure not masked, and rc mapping; board `core-*/cli/cli-report.json` shows compile/validate/explain rc0, two successful runs with stable fd count, bad binding rc2/no publish. |
| P8 board admission | **implemented+verified for the declared MVP scope** | campaign/verifier scripts | Final source-built rerun `results/20261023T144054Z-rk3588/`: **27 cases × 3 serial masks = 81/81 PASS**, including projection/FFN/branch from real multi-block TileLang staged sources, tiled RKIR cases, broadcast, dual output, ragged tail, K shards, guards and the full CLI path. Source hash recomputed true; raw outputs and dual references archived. |
| P9 expansion | **future extension** | n/a | Dynamic host broadcast, state/recurrent/hybrid, online core scheduling and unverified composite templates remain outside P0–P8 closure. |

<a id="mvp--next-milestone-separate-from-mvp"></a>
#### Next milestone (separate from MVP)

The complete-model milestone is reported separately in
`MODEL-MILESTONE-STATUS.md` and is fully closed.  Full task evidence remains
`results/20261024T071144Z-models/`: 10,000-image evaluation on core 1 gives MLP
98.31% and Tiny Transformer 98.21%, with 100% CPU/declared top-1 agreement; masks
2/4 run the fixed 512-image profile set.  Independent numeric evidence is
`results/20261024T091634Z-numeric/`: calibration uses 32 fixed unlabeled MNIST
train images, thresholds are frozen, then a disjoint 32-image test subset is
observed.  The verifier recomputes all 33 layer gates (including real-TileLang
MLP hidden linear+bias/ReLU diagnostics), five profile logit gates and the
existing full-test raw logits under the same train-frozen limits.  It additionally
rejects same-top-1 hidden-trace or logits amplification.  Previously viewed test
results are explicitly disclosed; this is not claimed as a blind benchmark.  The
old test-calibrated envelope is retained only as regression evidence.

<a id="mvp--specific-cross-cutting-conditions"></a>
#### Specific cross-cutting conditions

| Condition | Status | Code / proof | Tests / logs |
|---|---|---|---|
| Target identity and no silent widening | **implemented+verified** | `custom-ops/src/rk_plan_target.c` exact NUL-token compatible and exact driver string; `capabilities.py` profile validation | `test_target_check.py`, `test_capabilities.py`; board `target_check_rc=0`, profile hash match and ABI 1. |
| Binding dtype/shape/role/finite checks | **implemented+verified** | `binding_matches`, `check_finite_binding`, `rk_workspace_run` input/output overlap checks | `test_ws_poison.py::test_binding_error_before_submit_is_safe`, C host alias checks, board NaN input guard. |
| BO independence / no accidental sharing | **implemented+verified** | `must_not_share_bo`, slot/lifetime validator in `plan.py` and `rk_plan_validate.c` | `test_storage.py::ConstantLifetimeTests`, `test_plan_verify.py::test_mm_acc_alias_is_rejected`, C `aliases/shared BO` negative. |
| Device/host budgets | **implemented+verified** | `passes/storage.py`, C `claimed_*` validation and `CO_SLACK`/`CO_GUARD`/command workspace accounting | `test_storage.py`, C `multi_output_budget_case`, board environment budget snapshot. |
| Concurrent BUSY | **implemented+verified (Host/mock)** | atomic `ws->busy` exchange before mutation | `test_workspace_busy.py` (second concurrent run `RK_E_BUSY`, no publish, no leak). |
| Submit failure / poisoned recovery | **implemented+verified (Host/mock); real-device gap** | runtime poison path; no output publication before success | `test_ws_poison.py`, `test_guard_failure.py`; board intentionally avoids unsafe timeout injection. |
| CLI fd ownership, cleanup failure, void close ABI and error codes | **implemented+verified** | `cli.py` uses real `void rocket_close(int)`, opens/closes only its own fd after quiescent destroy, preserves `--fd N` ownership, never closes provider fd after poisoned destroy, reports destroy failures, preserves original run error, and never returns success when cleanup failed | `test_cli.py`, `test_runtime_abi.py` (header + ctypes + C probe), `test_workspace_busy.py`; board `cli-report.json` records fd equality and bad-binding rc2/no publication. |
| Shared-fd provider context quarantine | **implemented+verified on Host/mock/provider boundary; real-device fault unmeasured** | `rknpu_provider.c`: atomic per-fd `faulted` set inside `submit_lock` before unlock; later submits return `-EPIPE` without task-BO rewrite; runtime rejects same-fd existing/new workspaces; faulted `rocket_close` retains fd/task BO. Patch record: `provenance/upstream-patches.json` | C `provider_context` queued-submit + healthy-close test; `test_workspace_busy.py::test_shared_fd_context_fault_blocks_other_and_new_workspaces`; `test_runtime_abi.py`. |
| Command/task/data and provider-context failure lifecycle | **implemented+verified on Host/mock; real-device unknown state unmeasured** | `RUNTIME-LIFECYCLE.md`: creation failure frees command+data BOs; pre-submit failures are safe; executor errors poison the workspace; provider `atomic_int faulted` is set under `submit_lock` before unlock, blocks queued/shared-fd/new-workspace submits, and faulted `rocket_close` retains fd/task BO. Upstream patch recorded in `provenance/upstream-patches.json`. | `test_ws_poison.py`; `test_workspace_busy.py` concurrent failing-run and shared-fd two-workspace/new-workspace paths; `test_runtime_abi.py`; C `provider_context` queued-submit task-BO/close test. `live_bos == 7` is only an allocation-count check, not lifecycle proof. |
| Source/library provenance | **source independently verified; library recorded-only** | `scripts/run-board-admission.sh`, verifier `artifact_verification` | board `source-bundle.sha256` recomputed equal; `board/library.sha256` is a remote record and is explicitly `verified=false` on the Host. |
| Credential hygiene | **implemented+verified** | `.env` ignored; askpass + exact-value scan | board campaign `credential scan: PASS`; exact scan of `results/20261023T144054Z-rk3588/` is zero-match. |

<a id="mvp--mvp-scope-boundary"></a>
#### MVP scope boundary

The exhaustive-shape, multi-board, concurrency, long-stress and undefined
“full release gate” items are removed.  The finite representative matrix, P0
source rebuild and same-board rerun are complete.  There is no remaining required
MVP acceptance condition for the declared scope.

**Explicit scope boundary:** real RK3588 submit/recovery after an unknown fault is
not implemented, not tested and not promised.  Host/mock verifies pre-submit safety,
BUSY, poison-on-unknown, output non-publication and command/data BO retention; the
real-device success path is verified.  The accepted failure contract is quarantine,
refuse retry and stop, not automatic recovery.  This limitation is not counted as
an implicit MVP gate.

---

<a id="models"></a>

## 完整模型里程碑验收

原文件：`MODEL-MILESTONE-STATUS.md`。以下为该文件历史内容。

<a id="models--next-milestone-two-complete-models"></a>
### Next milestone: two complete models

Status: **complete** (2026-10-24).  This is the independent N0-N3 milestone
described in [NEXT-MILESTONE.md](archive/development-history.md#model-requirements); it does not retroactively
change the already accepted MVP.

The first checkpoint proved that both learned models run end to end.  The
closure then fixed model timing boundaries, introduced the registered/planized
`rkhybrid` boundary, and separated numeric calibration (train split, no labels)
from acceptance (disjoint test split).  A previously test-calibrated envelope is
retained only as historical regression evidence.  The earlier performance
fields are still explicitly invalid.

Final optimisation delivery re-acceptance (2026-10-26): all five static profiles
were re-run on the full 10,000-image test split in one baseline/current campaign;
MLP accuracy 0.98310 and Tiny ViT accuracy 0.98210, top-1 agreement 1.00000, and
the current build passed the frozen packed numeric replay.  See
[OPTIMIZATION-SUMMARY.md](OPTIMIZATION-SUMMARY.md) and
`results/20261026T120625Z-final/`.  This does not redefine or add conditions to
N0-N3.  A separate 2026-10-27 vendor RKNN / CPU comparison is recorded in
`results/20261027T015302Z-vendor/`; it is performance evidence, not a replacement
for this milestone's packed numeric acceptance.

<a id="models--delivered-models"></a>
#### Delivered models

<a id="models--mnist-mlp"></a>
##### MNIST MLP

- Learned weights: `models/mnist_mlp.npz` (safe `npz`, no pickle).
- Structure: `784 -> 256 -> 128 -> 10`, bias and exact ReLU at each hidden layer.
- Training: fixed seed 20261024, MNIST train split only, 12 epochs AdamW.
- CPU FP32 test accuracy: **0.9831** (9,831 / 10,000).
- NPU path: all linear layers, biases and ReLUs execute through real TileLang
  staged PrimFuncs -> RKIR -> RKPlan -> production runtime.
- Intermediate activations remain in device BOs; the plan has one final output
  download and no intermediate download/re-upload bridge.
- Static profiles: batch 1, 8 and 32 are separately compiled.  The logical
  784-wide input is zero-padded to 1024 because the locked ordered-K primitive
  uses full 256-wide tiles; the first 10 of the padded 16 logits are returned.
  The 10,000-image split is processed as 312 × batch 32 plus two compiled batch-8
  tail calls; no dynamic shape is used.
- Source: `models/tilelang_mlp.py`; plan compiler: `tools/compile_models.py`.

<a id="models--tiny-transformer-image-classifier"></a>
##### Tiny Transformer image classifier

- Learned weights: `models/tiny_vit.npz` (safe `npz`, no pickle).
- Structure: 16 raster 7x7 patch tokens, D=64, 4 heads, one pre-RMSNorm encoder
  layer, pre-RMSNorm SwiGLU FFN=128, residual connections, mean pooling, 10-way
  linear classifier.
- Training: fixed seed 20261024, MNIST train split only, 30 epochs AdamW.
- CPU FP32 test accuracy: **0.9821** (9,821 / 10,000).
- NPU matrix subgraphs: patch projection, QKV, QK, PV, attention output
  projection, FFN gate/up/down and classifier.  Every one is a real TileLang
  PrimFunc -> RKIR -> RKPlan -> production runtime.
- QK/PV use the immutable 16x16 dynamic-B primitive plus pre-expanded shared
  group plans; each group plan has one reused workspace across all batches and
  heads.  The default mapped-QKV manifest feeds Q/K/V column windows directly;
  q/k/v materialisation is capture-only/diagnostic.
- Static profiles: batch 1 and 8.
- Source: `models/tilelang_matmul.py`; plan compiler: `tools/compile_models.py`.

<a id="models--planized-hybrid-execution"></a>
#### Planized hybrid execution

The Python model runner is now a thin executor over a versioned, closed
`rkhybrid` manifest.  It is not an importer and contains no second model
orchestration path.

- MLP profile: 4 nodes = 3 registered Host-action nodes + 1 RKPlan node.
- Tiny Transformer default profile: 22 nodes = 14 Host-node declarations
  (one capture-only) + 8 RKPlan nodes, referencing 9 immutable RKPlan files.
- Every graph node declares inputs/outputs, static shape and semantic axes,
  dependencies, numeric contract and bridge-accounting policy.
- NPU nodes name an immutable RKPlan key, hash and source hash.  Host nodes name
  an action in `rknpu_backend.hybrid.HOST_ACTIONS` and its exact version.
- The QK and PV nodes declare batch/head repeat axes and explicit source
  view/slice/transpose maps; they do not call an opaque all-Transformer node.
- Unknown actions, unknown fields, plan-hash mismatches, shape/dtype mismatches,
  fallback policies and dynamic shapes are rejected.
- Host arithmetic is explicit and timed per node: patch rasterization/padding,
  position add, RMSNorm, stable softmax, merge, residual adds, SwiGLU, reshape,
  mean pooling and logits export.  QKV split is capture-only in the default
  mapped manifest.
- There is no implicit CPU matrix fallback.  A missing/unknown plan or Host
  action is an error.

Manifests are in `models/compiled/*.rkhybrid.json`; they are generated by
`tools/build_hybrid_manifests.py` from the frozen model architecture and the
compiled RKPlan manifest.

<a id="models--numeric-acceptance"></a>
#### Numeric acceptance

The current independent boundary is documented in
[models/NUMERIC-ACCEPTANCE.md](models/NUMERIC-ACCEPTANCE.md).

<a id="models--calibrationacceptance-separation"></a>
##### Calibration/acceptance separation

- Calibration uses **32 fixed MNIST train images** selected by seed
  `20261024`, without reading labels.
- Acceptance uses **32 fixed MNIST test images** selected by seed `20261025`,
  also without reading labels during observation.
- Train and test source-image SHA-256 hashes, selected indices and per-image raw
  pixel SHA-256 are recorded in `models/numeric_acceptance_train.json`.
- The verifier requires different official splits, different source-image file
  hashes and disjoint per-image pixel-hash sets.
- RK3588 order was: train calibration observation -> Host freeze -> SHA-256 of
  frozen spec recorded by the acceptance observation -> test observation.
- **Disclosure:** the full test set and prior task results were seen before this
  boundary was introduced.  This is a disjoint post-freeze numerical acceptance
  rerun, **not a blind benchmark**.

<a id="models--frozen-policy"></a>
##### Frozen policy

- `models/numeric_acceptance_train.json` freezes **33 layer gates** and
  **5 static-profile raw-logit gates** with `rtol = 0`.
- Device-derived FP16 outputs use:

  ```text
  atol = ceil_to_2^-12(max(2 * train_observed_max_abs_error,
                           2 * f16_spacing(train_expected_absmax)))
  ```

- Host-only exact layout/normalization outputs use only
  `ceil_to_2^-12(2 * train_observed_max_abs_error)`.
- No test-set error, maximum, logit, label or top-1 result enters the threshold
  calculation.
- The floor is a representational-scale FP16 allowance, not a theoretical
  end-to-end error bound.  The accepted scope is the frozen package, weights,
  plans and RK3588 execution paths.
- If acceptance fails, the policy is to investigate the numeric path or
  implementation.  Enlarging limits from test maxima is forbidden by the recorded
  calibration rule.

<a id="models--deterministic-evidence"></a>
##### Deterministic evidence

- `models/numeric_calibration_train.npz` stores the fixed train images,
  independent declared expected values and profile logits.
- `models/numeric_acceptance_test.npz` stores the disjoint test images and
  independent expected values.
- `models/compiled/mnist_relu_probe.rkplan.json` is the real-TileLang
  `co.ew_add.f16.v1 -> co.ew_max.f16.v1` exact-ReLU probe.
- The Transformer trace covers patch projection, positioned input, both RMSNorms,
  QKV, per-head Q/K/V, QK scores, softmax, per-head PV, attention projection,
  residuals, SwiGLU, FFN, pooling and logits.  Dynamic-B QK/PV are included.

<a id="models--real-mlp-hidden-layer-diagnostics"></a>
##### Real MLP hidden-layer diagnostics

Production `mnist_mlp_b*` plans are unchanged and still download only final
logits.  Separate real-TileLang diagnostics export the actual hidden-layer
linear+bias and exact-ReLU values:

- `mnist_mlp_diag_l1_bias`: layer-1 `matmul(W1)+b1` -> pre-ReLU;
- `mnist_mlp_diag_l1_relu`: layer-1 bias + exact ReLU;
- `mnist_mlp_diag_l2_bias`: layer-2 `matmul(W2)+b2` from the **hardware ReLU1**
  output -> pre-ReLU;
- `mnist_mlp_diag_l2_relu`: layer-2 exact ReLU from the hardware ReLU1 path;
- `mnist_mlp_diag_l3_bias`: layer-3 `matmul(W3)+b3` from the **hardware ReLU2**
  output -> final 16-wide preactivation+bias.

Observed post-freeze test-subset errors and train-frozen limits:

| MLP trace | max abs error | frozen atol |
|---|---:|---:|
| layer-1 pre-ReLU linear+bias | 0.001953125 | 0.03125 |
| layer-1 exact ReLU | 0.0009765625 | 0.03125 |
| layer-2 pre-ReLU linear+bias | 0.00390625 | 0.03125 |
| layer-2 exact ReLU | 0.001953125 | 0.03125 |
| final preactivation+bias | 0.015625 | 0.0625 |
| final logits | 0.015625 | 0.0625 |
| ReLU primitive probe | 0.0 | 0.001953125 |

<a id="models--independent-checks"></a>
##### Independent checks

`tools/verify_numeric_acceptance.py` checks:

- train/test source hashes, indices, per-image hashes and observation hashes;
- disjoint calibration/acceptance sample sets;
- every frozen limit recomputes exactly from train observations;
- finite values, expected shapes and all 33 hidden/layer traces;
- all five profile raw-logit arrays;
- the frozen spec SHA-256 recorded by the post-freeze acceptance run;
- all used RKPlan hashes and the compiled-manifest hash;
- the existing full 10,000-image campaign raw logits against the same
  train-frozen profile limits.

Final numeric campaign:
`results/20261024T091634Z-numeric/`.
Its `verify-summary.json` is **PASS / 0 problems**, with 15 existing-campaign raw
logit rows rechecked.  The negative regression feeds a hidden MLP trace outside
its frozen limit while leaving the final logits/top-1 unchanged; the layer gate
still rejects it.

<a id="models--historical-test-envelope-v1"></a>
##### Historical test envelope v1

`models/numeric_acceptance.json`, `models/numeric_golden.npz` and
`evidence/numeric-calibration/` are retained for reproduction of the first-round
run.  They were calibrated from previously viewed test-split observations and are
therefore labeled **regression envelope only**, not independent acceptance.

<a id="models--full-task-results"></a>
#### Full task results

Board campaign: `results/20261024T071144Z-models/`.

| Model | Profile / split | CPU FP32 acc | RK3588 acc | Δ | CPU top-1 agreement | Declared top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| MLP | b32 / 10,000 | 0.9831 | 0.9831 | 0.0000 | 1.0000 | 1.0000 |
| Tiny Transformer | b8 / 10,000 | 0.9821 | 0.9821 | 0.0000 | 1.0000 | 1.0000 |

Additional serial admission rows on the same fixed 512-image subset:

- MLP b32/b8/b1: accuracy **0.986328125**, CPU/declared top-1 agreement 1.0;
- Transformer b8/b1: accuracy **0.986328125**, CPU/declared top-1 agreement 1.0;
- all five profile rows and all three core masks are checked by the independent
  verifier; `model-results/verify-summary.json` is **PASS** with zero problems.

Full-split logit errors:

- MLP: max |hardware − CPU FP32| = **0.05069**; max |hardware − independently
  recomputed declared FP16 path| = **0.03125**.
- Transformer: max |hardware − CPU FP32| = **0.03843**; max |hardware −
  independently recomputed declared FP16 path| = **0.03125**.

The declared path is a reproducible Host-side emulation of the graph's ordered
FP16 writeback and registered Host nodes.  It is not a substitute for CPU FP32.

<a id="models--corrected-timing-and-cost"></a>
#### Corrected timing and cost

Timing schema v2 defines the model interval as:

```
preprocess + execution + postprocess
  execution = all RKPlan node dwell + all registered Host-action nodes
              + measured executor dispatch overhead
```

Weight loading, runner/constant setup and warm-up are outside the interval.  CPU
FP32 and declared reference calculations are separate fields and are outside the
model interval.  `unattributed_seconds` is zero in the final campaign and is
checked by the verifier.

RK3588 core 1, completed 10,000-image split:

| Model | End-to-end | Throughput | Preprocess | Execution | Postprocess | RKPlan node | Native workspace | Registered Host | Dispatch overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b32 | **0.9909 s** | **10,092 img/s** | 0.1288 s | 0.8433 s | 0.0118 s | 0.8158 s | 0.7578 s | 0.1401 s | 0.0275 s |
| Transformer b8 | **50.724 s** | **197.1 img/s** | 0.3001 s | 50.3097 s | 0.0523 s | 45.4405 s | 35.1827 s | 2.9928 s | 2.2284 s |

Reference calculations (excluded from the table above): MLP CPU/declared
2.156 s / 3.603 s; Transformer CPU/declared 7.337 s / 11.221 s.

Full-split transfers and task counts:

- MLP: 20.496 MB upload, 0.641 MB download, 3,454 NPU tasks / 2,512 submits.
- Transformer: 391.040 MB upload, 287.680 MB download, 88,750 NPU tasks /
  88,750 submits; 699.44 MB of registered Host tensor materialization.
- MLP full split uses 312 × b32 + 2 × b8 tail calls.  Transformer uses 1,250 ×
  b8 calls.  No dynamic shape or un-compiled tail profile is used.

Same-board CPU baseline is now matched to the NPU batch schedule:

- 8 CPUs, frequency range 408–2,304 MHz, affinity count 8;
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` was set for the
  benchmark and is recorded in the report;
- MLP FP32: 2.141 s, 4,670.8 img/s;
- Transformer FP32: 7.308 s, 1,368.3 img/s.

These are different numeric paths (CPU FP32, single-thread, no bridge versus NPU
FP16 with its graph/Host bridge).  The final data does not support a blanket
"NPU is faster" claim: this MLP profile is faster than the stated CPU baseline,
while the first-round hybrid Transformer remains slower because of its Host
bridge and per-head plan-call count.  No speedup threshold is part of this
milestone.

The historical campaign `results/20261024T031249Z-models/` is retained, but its
`wall_seconds` and throughput fields are invalid because references were inside
the interval and the CPU batch policy differed.  See
`results/20261024T031249Z-models/PERFORMANCE-RETRACTION.md`.

<a id="models--reuse-failure-behavior-and-cleanup"></a>
#### Reuse, failure behavior and cleanup

- `PlanCall` validates/materialises all Host constants before creating any C
  plan, creates one CPlan/workspace per plan key, uploads constants once and
  reuses the workspace for input updates.
- Partial creation failure destroys already-created CPlans/workspaces.  An
  exception during creation remains the original exception; cleanup failures
  are attached as notes instead of replacing it.
- `HybridRunner` and `PlanCall` are context managers.  A failed run remains
  primary while cleanup is reported separately; poisoned/BUSY resources retain
  the runtime's ownership policy.
- A model-level injected submit failure returns no result, publishes no output,
  leaves the workspace POISONED and does not retry.  The new graph runner cannot
  bypass these runtime rules.
- Shared-fd/provider-context quarantine behavior is unchanged.

<a id="models--evidence-and-reproduction"></a>
#### Evidence and reproduction

Primary evidence:

- Final RK3588 campaign and independent verifier:
  `results/20261024T071144Z-models/` (`verify-summary.json` PASS).
- Raw board logits and deterministic layer traces:
  `results/20261024T071144Z-models/model-results/core-*/`.
- Train-calibrated numeric package: `models/numeric_acceptance_train.json`,
  `models/numeric_calibration_train.npz`, `models/numeric_acceptance_test.npz`,
  and the five `models/compiled/mnist_mlp_diag_*.rkplan.json` plans.
- Numeric acceptance campaign:
  `results/20261024T091634Z-numeric/` (`verify-summary.json` PASS).
- Historical test-only regression envelope: `models/numeric_acceptance.json`,
  `models/numeric_golden.npz`, `evidence/numeric-calibration/`.
- Hybrid manifests: `models/compiled/*.rkhybrid.json`.
- Host/mock complete split: `evidence/model_eval_mock_v2.json`; matched-batch CPU
  baseline: `evidence/cpu_benchmark_v2.json`.
- Host C/Python/sanitizer verification: `results/20261024T092339Z-host/`
  is **PASS**: **235 pytest passed, 0 failed, 0 skipped**; **27/27 ctest**;
  contract **26** checks, host **711** checks, graph host **2,400,033** checks;
  ASan/UBSan enabled explicitly.

Replay commands:

```bash
cd research/tilelang-rknpu

# Rebuild real TileLang RKPlans, ReLU probe and rkhybrid manifests
PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
python3 tools/compile_models.py

# Rebuild the train-calibrated expected references and initial policy
PYTHONPATH=$PWD/python python3 tools/build_numeric_train_acceptance.py

# Reproduce the matched-batch host evaluation
PYTHONPATH=$PWD/python RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
python3 tools/evaluate_models.py --backend mock --models both \
  --global-limit 10000 --profile-samples 512 \
  --out evidence/model_eval_mock_v2.json

# Full model campaign (accuracy/timing; core 1 full split; masks 2/4 512 subset)
./scripts/run-model-admission.sh

# Train-calibrated numeric evidence: calibration on train, freeze, then a
# disjoint post-freeze test observation on RK3588 core 1
./scripts/run-numeric-acceptance.sh

# Verify frozen limits, sample disjointness, MLP hidden traces and existing
# full-test raw logits under the train-frozen profile limits
PYTHONPATH=$PWD/python:$PWD \
python3 tools/verify_numeric_acceptance.py \
  --spec models/numeric_acceptance_train.json \
  --calibration-observation results/20261024T091634Z-numeric/calibration/observation.npz \
  --acceptance-observation results/20261024T091634Z-numeric/acceptance/observation.npz \
  --campaign-manifest results/20261024T071144Z-models/campaign/manifest.json \
  --campaign-results results/20261024T071144Z-models/model-results

# Historical-regression-envelope verifier only (not independent numeric proof)
python3 tools/verify_model_results.py \
  --manifest results/20261024T071144Z-models/campaign/manifest.json \
  --results results/20261024T071144Z-models/model-results
```

<a id="models--explicit-non-claims--next-boundary"></a>
#### Explicit non-claims / next boundary

This milestone does **not** add dynamic shapes, generic ONNX import, arbitrary
TileLang control flow/masks, CNN operator coverage, NPU training, INT8/INT4,
full LLM loading, KV cache/state, RK1228 support, multi-board/concurrent
throughput, or post-fault recovery.  The Transformer's Host bridge is reported
and timed rather than hidden; it is not claimed to be pure NPU.

<a id="smollm2-135m-instruct"></a>
## SmolLM2-135M-Instruct 完整推理验收（2026-10-28）

> 历史 V2 记录：本节数字和 spec 指针属于 2026-10-28 的 V2 campaign。V2 spec
> 已归档为 `models/specs/smollm2_v2_20261028T094650Z.json`，旧 manifest 的
> spec 指针已加 correction 字段。当前可审计验收以
> [2026-10-29 v5 验收](#smollm2-v5-acceptance) 为准。

### 结论与设备

在 RK3588（Orange Pi 5 Plus，kernel
`6.1.43-rockchip-rk3588-rknpu098`，driver v0.9.8，单核 mask 1）上，以固定
revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` 的
HuggingFaceTB/SmolLM2-135M-Instruct 完成 batch=1、容量 256 token 的真实权重推理：
30 层全部执行，词表 49152 全量执行；prefill、持久 KV cache 的逐 token decode、
chat template、EOS/max_new_tokens、reset 重放和独立数值/质量验收均通过。

本里程碑不是 RKNN/RKLLM 路径，也没有 CPU 整模型执行或 CPU 矩阵 fallback。
Embedding、RMSNorm、RoPE、GQA layout/merge、稳定 causal softmax、SwiGLU
elementwise、residual add、logits 拼接和采样是明确注册并计量的 Host action；
所有 Q/K/V/O、gate/up/down、attention QK/PV 和 lm_head 分片的矩阵任务均来自
`models/tilelang_matmul.py:make_matmul_prim` 的真实 TileLang PrimFunc，经
`compile_tilelang` 生成 RKPlan 后在板端提交。`lm_head` 用 12 个 4096 宽分片覆盖
49152 词表。

### 固定模型身份

| 项 | 值 |
|---|---|
| revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| config / tokenizer / weights SHA-256 | 见 `results/20261028T094650Z-smollm2/manifest.json` |
| layers / hidden / intermediate / vocab | 30 / 576 / 1536 / 49152 |
| Q heads / KV heads / head_dim | 9 / 3 / 64 |
| activation / eps / RoPE | SiLU / 1e-5 / theta=100000, HF half-split |
| tied embeddings | true |
| BOS / EOS / pad | 1 `<|im_start|>` / 2 `<|im_end|>` / 2 |
| chat template | tokenizer_config 内嵌模板，Host 侧应用 |

### 计划与执行资源

| 项 | 实测/声明 |
|---|---:|
| RKPlan 数 | 268 |
| NPU tasks / submits | 5660 / 1712 |
| 每 decoder 层每 token 的矩阵计划调用 | QKV + O + gate/up + down + 3 QK + 3 PV = 9 |
| decode lm_head 计划 | 12 |
| 每 token 计划调用总数 | 282（30×9 + 12） |
| 权重文件 | 269,077,440 B FP16 |
| 部署计划常量（decode+prefill） | 537,919,488 B |
| Host KV cache | 30×3×256×64×2×2 = 5,898,240 B |
| Plan artifact 文本 | 10,784,213 B |
| 峰值 RSS | 2,049,200 kB |

268 个计划的 `templates` 均只包含 `co.mm.f16.v1` / `co.mm_acc.f16.v1`；无
Host 矩阵计划、未知 action 或静默 fallback。QK/PV 使用运行时 B operand；KV 追加
仍是 Host 侧显式、带边界检查的 in-place action。

### 数值验收

门限只用 calibration prompt（“Name the capital of France.”，36 token）上的
观测冻结，验收使用不重叠的 prompt（“Write a Python function that returns the
square of an integer.”，42 token）。完整 spec 与哈希：
`models/smollm2_numeric_acceptance.json`（56 个 gate）。独立 declared reference
显式实现 FP16 权重/激活、每个 256-wide K tile 的 FP16 writeback、FP32 RMSNorm
归约、FP32 RoPE/softmax 和单次 FP16 收窄；官方参考为 HF Transformers float32、
`attn_implementation=eager`、固定随机种子。

| 检查 | 观测 | 冻结门限 | 结果 |
|---|---|---|---|
| full logits vs declared reference | max_abs 0.138671875, RMSE 0.0256602, top1 42/42, top5 均值 4.952 | atol 0.296875, RMSE 0.07421875, top1 ≥0.99 | PASS |
| full logits vs official FP32 | max_abs 0.1135664, RMSE 0.0224003, top1 42/42, top5 均值 4.976 | atol 0.1955566, RMSE 0.0625, top1 ≥0.99 | PASS |
| teacher-forcing 74 positions vs declared reference | max_abs 0.1464844, RMSE 0.0239630, top1 74/74 | atol 0.34765625, RMSE 0.0869141, top1 ≥0.99 | PASS |
| incremental KV decode vs full-prefix recompute | 9/9 位置逐元素完全相等（max_abs=0, RMSE=0） | atol/RMSE 2^-12, top1 ≥0.99 | PASS |
| 首/中/末层 captures（layer 0/15/29） | 见 verifier JSON | 校准冻结的 per-tensor atol/RMSE | PASS（56/56） |
| lm_head shard 覆盖 | 12×4096=49152；每个 shard 与 declared reference 数值 gate | 校准冻结 | PASS |

补充：per-shard 内部 rank 本身不作为 gate。验收中 shard 08 在 42 个位置里有 1
个近 tie（top1 从 2670 变 1317，差 0.0078，整词表 top1 仍全对）；这属于硬件
FP16 近 tie，不是词表索引/拼接错误。V1 policy 曾误把 per-shard rank 作为 gate，
并误把官方 continuation 本身作为 gate；V2 保留所有 atol/RMSE 值与 full-vocab
rank gate 不变，只把这两项改为诊断，原始 V1 观测保留在 evidence 中。

### 质量与自由生成

| 项 | Backend | Official FP32 |
|---|---:|---:|
| 验收文本 1275 scored tokens, mean NLL | 3.23915684 | 3.23910809 |
| 验收文本 PPL | 25.512202 | 25.510958 |
| 验收文本 top1 accuracy | 0.43843137 | 0.43843137 |
| 校准文本 1275 tokens, mean NLL | 2.88394461 | 2.88385677 |
| 校准文本 PPL | 17.884682 | 17.883112 |

固定 instruction prompts（chat template，greedy，32 token）：
- prompt 0 “What is 2+2? Answer with one short sentence.”：32/32 token 与官方
  一致，首个分叉 None；
- prompt 1 “Name the capital of France.”：21/32 一致，首个分叉 index 20；
- prompt 2 “Write a Python function that returns the square of an integer.”：
  8/32 一致，首个分叉 index 6；
- prompt 3 “In one sentence, explain why the sky appears blue.”：32/32 一致，
  首个分叉 None。
Prompt 0 的实际文本为
`Two and two is equal to four.<|im_end|>`；这一历史 4-prompt 批次中，teacher
forcing 对当时官方 continuation 的 top1 为 31/32（首个分叉 index 6），仅作为
当时的质量差异报告，不作为后端数值 gate；V5 的对应记录是另一组 8-prompt
批次，见下文。

### 性能与资源

3 次独立 generate（同一进程，每次 reset，32 new tokens）：

| 指标 | 观测 |
|---|---:|
| Host 权重转换 + 268 plan 编译（本机实测） | 4.977 s |
| 进程内模型/包/268 workspace 初始化 | 7.713 s |
| 冷 TTFT（初始化 + prefill，不含 SSH/tokenization） | 10.594 s |
| 暖 prefill / TTFT | 2.644–2.907 s |
| decode 总时长（32 token） | 13.049–15.192 s |
| decode 稳态中位/均值 | 465.85 / 445.17 ms/token |
| decode tokens/s（warm） | 2.45 |
| 3 次总量：plan dwell / Host actions | 23.484 s / 5.898 s |
| 每 decode token 平均 upload / download | 29.92 MB / 9.51 MB |
| 每 decode token pack / submit wait | 60.44 ms / 87.79 ms |
| 峰值 RSS | 2,049,200 kB |

计时边界：`prefill_us` 从 Host `prefill()` 入口（embedding 起）到 12 个
lm_head shard 和 logits 拼接完成；`decode_us` 从 `decode_one()` 入口到下一 token
logits 完成，包含 30 层、QK/PV、head shards 和所有 Host action。TTFT 不含 Host
chat template/tokenization、SSH、结果下载。未与 RKNN/RKLLM/CPU 做同口径加速比较。

### 证据与重放

- 汇总与全部文件哈希：`results/20261028T094650Z-smollm2/manifest.json`
- 验收 verifier：`results/20261028T094650Z-smollm2/verify/verify-summary.json`
- raw arrays（all-logits、teacher-forcing、reference、official）：本地 ignored
  `.data/evidence/smollm2-20261028T094650Z/`，路径和哈希见上述 manifest
- 数值 spec：`models/smollm2_numeric_acceptance.json`
- model package compile/fetch/chat/replay：`models/README.md#smollm2-135m-instruct`
  与 [REPLAY.md](REPLAY.md#smollm2-135m-instruct)
- host 回归：371 pytest PASS、33/33 ctest PASS；日志在 evidence `host/`
- board 回归：既有 current MLP/TinyViT hybrid plans 各 8 张图的 admission smoke
  PASS（MLP b1/b8/b32 与 TinyViT b1/b8 accuracy 1.0，top1 agreement 1.0），
  证明本次通用 extractor/identity 修改未破坏既有模型路径；这不是历史
  full-split 验收的替代。原始结果在 evidence `regression/`。
- 边界：prompt len 1/2/3/255/256、257 拒绝、255+2 拒绝、空 prompt 拒绝、
  EOS 停止、max_new_tokens 停止、两次 reset 结果一致。

### 非声明

本验收只覆盖上述单板、单核、batch=1、FP16、固定 256-token profile 的
SmolLM2 几何；不声明任意 LLM importer、动态 shape、设备驻 KV cache、量化、
并发请求、多板、故障后恢复、RK1228 或性能优势。

<a id="smollm2-v5-acceptance"></a>
## SmolLM2-135M-Instruct 验收闭环修订与 v5 重新验收（2026-10-29）

### 结论

本轮不以新增功能为目标，完成的是验收闭环修复：修复 verifier 漏判、完整 gate
集合/shape/有限性/输入身份/质量输入校验，新增负例测试，冻结 v5 政策与独立输入，
重新执行当前实现的完整板端验收。最终结果：

- **complete acceptance PASS**；
- 56/56 gate 通过，`gate_failures=[]`、`failures=[]`、进程退出码 0；
- 数值：44-token 全词表 logits vs declared reference top1 44/44；vs official FP32
  top1 44/44；teacher-forcing **vs declared reference** 76/76。对归档官方
  continuation 的质量诊断是 31/32、first divergence 0；该归档的 official
  generation 口径问题见下方“生成分叉诊断”，不改写 V5 输入或门限；
- incremental KV decode 与完整前缀重算 17/17 个 logits 位置逐元素完全相等；
  其中 1 个位置来自初始 prefill 的最后一个 logits，另外 16 个来自 16 次
  decode，不是“17 次 decode”；
- 质量：验收文本 1275 scored tokens，backend mean NLL 4.78984792，official
  4.78982925，PPL 120.2831 vs 120.2808；
- 当前执行路径仍满足：batch=1、容量 256、真实全 30 层与完整 49152 词表、
  所有目标矩阵经 TileLang→RKNPU、Host actions 显式且计量。

### Verifier 修复

`tools/verify_smollm2.py` 原先只把失败 gate 写入 `first_failed`，总 `pass` 未计入
`gate_failures`。本轮修复后：

- 任意 numeric/rank gate 失败都会使 `pass=false` 且 CLI 返回非零；
- 预期 gate 集合与执行 gate 集合必须完全一致；缺失/额外 capture tensor 均失败；
- observation/reference shape 必须完全相等，禁止 NumPy broadcast；
- observation/reference/threshold/quality 全部要求有限值，NaN threshold 失败；
- 正式 complete 模式强制要求 numeric + quality + sealed calibration/acceptance
  observation bundles；`--numeric-only` 明确标记为不完整验收；
- 核对 board/ref/official 的 token IDs、continuation、词表和窗口身份；
- 质量对比逐窗口核对相同 token IDs、labels、scored token 数，而非只比汇总值；
- summarizer 拒绝失败、缺项或非 complete 的 verifier JSON，并禁止用总计数除以
  硬编码 96 推算 per-decode 指标。

负例测试位于 `tests/test_verify_smollm2.py`，机器可读矩阵在
`results/20261029T120416Z-smollm2-v5/verify/negative-matrix.json`，覆盖：

| 负例 | 预期 | 实际 |
|---|---|---|
| `logits.full` atol/rmse=0 但 max_abs>0，top1 不变 | gate fail / exit≠0 | `gate_failures=["logits.full"]`, exit 1 |
| top1 rank gate 失败但数值误差小 | gate fail / exit≠0 | exit 1 |
| 缺失 capture gate | gate fail / exit≠0 | `capture_tensor_set` + gate fail |
| shape 可广播但不相等 | shape_mismatch / exit≠0 | exit 1 |
| observation 非有限 / reference 非有限 | nonfinite / exit≠0 | exit 1 |
| NaN threshold | invalid spec / exit≠0 | exit 1 |
| quality 缺失 | command rejected | exit 2 |
| quality NLL 非有限 | quality fail / exit≠0 | exit 1 |
| quality 窗口 token 不一致 | quality fail / exit≠0 | exit 1 |
| quality scored-token 数不一致 | quality fail / exit≠0 | exit 1 |
| input token / continuation 不一致 | identity fail / exit≠0 | exit 1 |
| package manifest/weights 不一致 | identity fail / exit≠0 | exit 1 |
| summarizer 收到失败 verifier | 拒绝汇总 | exit 1 |

### 政策与输入身份

| 项 | 值 |
|---|---|
| 当前 spec | `models/smollm2_numeric_acceptance_v5.json` |
| canonical content SHA-256 | `1a8c68d2da51eb9bc6885a4d540950f74011e8bcbf150a5d975d75a7575f368b` |
| file SHA-256 | `04c4011c7eb7daec4f4f607d65d234f8ec94ddba112a931fc2fd24056f70e38f` |
| 输入 manifest | `models/smollm2_v5_inputs.json` |
| 输入 manifest frozen SHA-256 | `8a7928796f49648db4bea24403bc8e348393da772f5c1a072ac533d59c5a9ed7` |
| calibration prompts | 6 条独立 instruction prompts |
| calibration text | `RUNTIME-LIFECYCLE.md`，5×256 tokens，1275 scored tokens |
| acceptance prompts | 8 条 v5 新 instruction prompts |
| acceptance text | `SUPPORT-DOMAIN.md`，5×256 tokens，1275 scored tokens |
| numeric acceptance prompt | “Write a short Python function that returns the largest number in a list.” |
| teacher forcing | 该 prompt + 32-token 归档 official greedy continuation，共 76 tokens；数值 gate 只对照 declared reference |
| incremental continuation | 前 16 token；与 full-prefix teacher forcing 对照 1 个 prefill logits + 16 个 decode logits = 17 rows |
| rank policy | 预声明固定 `top1>=0.95`、`top5 mean>=4.0` |
| numeric threshold | calibration 6 prompts / 3 layers / 39 tensors / full+12 shard logits / 6 TF sequences |

**V5 spec 元数据勘误（不改门槛、不改 spec 身份）**：冻结的 V5 spec
`policy_notes[0]` 仍写有 “three calibration prompts”，这是冻结前遗留的文本，
不是实际校准输入。实际校准为 6 条 prompt；`models/smollm2_v4_calibration_bundle.json`
和 `models/smollm2_v5_inputs.json` 均记录 6 条，且 `calibration_metrics.*.calibration_sources`
为 6。为保持 V5 spec canonical/file SHA-256 和全部已封存观测不变，本轮不编辑
该历史 spec，只在 `results/20261030T013910Z-smollm2-v5-delivery/POLICY-NOTE-ERRATUM.json`
记录勘误；`tools/freeze_smollm2_v3.py` 已改为从 `len(cal_prompts)` 生成该文本，
避免未来新 spec 重复该错误。

V1–V4 的处理：

- V1 为历史政策重建：固定 verifier 下仍失败于 `logits.shard08` 内部近 tie；原始
  V1 spec 文件未归档，故明确标为 reconstructed；
- V2 为历史 V2 政策重放：固定 verifier + 归档 V2 spec + 原始观测下 PASS；
- V3 与本轮当前实现的新输入的首次验收失败于 `layer.15.input_norm` 校准阈值，
  已降级为 development；
- V4 失败于 full-vocab/official top1 rate 0.9778 < 校准派生 0.99，已降级为
  development；随后 v5 在输入观测前预声明固定 rank floor，而不是事后放宽旧阈值。

历史/开发结果均保留在
`results/20261029T120416Z-smollm2-v5/history/`；政策身份总表在
`models/specs/smollm2_policy_history.json`。

### v5 验收数值

| 检查 | 观测 | 冻结门限 | 结果 |
|---|---|---|---|
| full logits vs declared reference | max_abs 0.1328125, RMSE 0.0243632, top1 44/44, top5 4.9773 | atol 0.359375, RMSE 0.08984375, top1≥0.95 | PASS |
| full logits vs official FP32 | max_abs 0.0977511, RMSE 0.0206828, top1 44/44, top5 4.9773 | atol 0.2268066, RMSE 0.0625, top1≥0.95 | PASS |
| teacher forcing vs declared reference | max_abs 0.1328125, RMSE 0.0234402, top1 76/76 | atol 0.3203125, RMSE 0.0800781 | PASS |
| incremental vs full prefix | 17/17 个 logits 位置逐元素完全相等（1 个 prefill + 16 个 decode），max_abs=0, RMSE=0 | atol/RMSE 2^-12 | PASS |
| layer 0/15/29 captures | 40 capture gates（39 layer tensors + final_norm）+ 12 shard gates | 6-prompt calibration frozen | PASS |
| full vocabulary | 12×4096=49152 | exact shard positions + concat | PASS |
| teacher forcing vs 归档 official continuation | top1 31/32，first divergence 0 | 仅历史质量诊断，不参与数值 gate；已定位到 official generation 的 mask/position bug | 报告 |
| EOS / max_new / capacity | EOS 停止、max_new 停止、255 容量边界、257/255+2/empty 拒绝 | 见 board/acceptance | PASS |
| reset 重放 | persistent-KV reset 再次请求稳定 | 见 tests + board | PASS |

### 当前计划与资源

| 项 | 当前实测/声明 |
|---|---:|
| 编译 RKPlan 数 | 268 |
| 运行时加载 workspace 数 | 446 |
| NPU tasks / submits | 5660 / 1712 |
| plan-declared data BO | 7168 |
| plan artifact 文本 | 7,821,599 B |
| 部署 plan constants | 537,919,488 B |
| Host 权重文件 | 269,077,440 B |
| Host KV cache | 5,898,240 B |
| 峰值 RSS | 1,233,324 kB |

### 修正后的性能口径

固定请求：v5 numeric acceptance prompt（44 token），3 次独立 `generate()`，
每次 32 new tokens，core mask 1。

| 指标 | 实测 |
|---|---:|
| Host 权重转换 + 268 plan 编译（外部 wall） | 4.900 s |
| 进程内 package/446 workspaces 初始化 | 10.064 s |
| 冷启动到首 token（init+reset+prefill+first sample） | 14.026 s |
| TTFT（不含加载；reset+prefill+first sample） | 3.963–4.094 s，中位 4.013 s |
| prefill 中位 | 4.006 s |
| decode forward 中位 | 18.771 s / 31 forwards = 606 ms/forward |
| decode forward rate 中位 | 1.651 forward/s |
| 全 generate wall 中位 | 22.765 s |
| 全 generate output rate 中位 | 1.406 token/s |
| 输出 token / decode forward | 32 / 31（最后一次采样后无多余 forward） |
| decode 每 forward upload / download | 3,207,168 B / 5,603,328 B |
| decode 每 forward pack / submit wait | 7.67 ms / 163.90 ms |
| 稳态 KV append | 180 次 stripe / token，23,040 B/token，7.66 ms/token |
| NV init | 5,898,240 B / run，首次 decode 约 35.75 ms |
| 峰值 RSS | 1,233,324 kB |

说明：`decode_forwards=output_tokens-1` 是生成停止逻辑的显式断言；不再有最后
一次采样后额外的 `decode_one()`。`decode_ms`、`prefill_ms`、`sampling_ms` 和
`generate_wall_ms` 分别报告；per-forward 传输/pack/submit 由实际 stage delta
除以实际 forward 数得到。总 aggregate 计数不再被硬编码 token 数除。

### 质量

| 项 | Backend | Official FP32 |
|---|---:|---:|
| v5 acceptance text, 1275 tokens, mean NLL | 4.78984792 | 4.78982925 |
| v5 acceptance text PPL | 120.283075 | 120.280838 |
| v5 acceptance text top1 | 0.27686275 | 0.27764706 |
| calibration text, 1275 tokens, mean NLL | 4.31941969 | 4.31948853 |
| calibration text PPL | 75.145008 | 75.150177 |

8 条 v5 instruction prompts，greedy 32 token：**归档** official generation 的
一致数分别为 32/32、3/32、1/32、0/32、12/32、32/32、21/32、1/32；首个分叉
分别为 None、0、0、0、7、None、20、0。该结果保留为历史质量诊断，不作为
后端数值 gate。

### 生成分叉诊断（交付收口，不改变 V5 门槛）

对 8 条 prompt 在首次分叉的相同前缀上重放 backend、declared reference 和
official FP32 logits，8/8 的 top1 三者一致；记录 top1/top2 间隔和 top5
candidate，未把任何差异自动称为 near-tie。

进一步定位到归档 official generation 的参考工具缺陷：`model.generate()`
未显式传入 attention mask，而 token id 2 同时是 chat 模板分隔符和 tokenizer
pad id，transformers 把真实 prompt token 2 当成 padding，并据
`cumsum(mask)-1` 重置 RoPE position。使用显式全 1 attention mask 修正后，
8 条 prompt 的 corrected official generation 与 backend **256/256 token**
一致（每条 32/32），同一前缀 top1 也一致。结论是：归档的 102/256 一致率是
official generation 工具口径错误，不是 backend 矩阵实现偏差。修正后的
official generation 仅保存在独立交付目录，不覆盖 V5 输入、门槛或封存观测。

证据：

- `results/20261030T013910Z-smollm2-v5-delivery/divergence/diagnostic.json`
- `results/20261030T013910Z-smollm2-v5-delivery/divergence/DIVERGENCE-REPORT.md`
- `results/20261030T013910Z-smollm2-v5-delivery/divergence-probe-request.json`
- `tools/diagnose_smollm2_divergence.py`

### Host 权重缓存优化与同口径 A/B（V5 后交付）

候选优化只缓存不可变 package 的 FP16 embedding 表和 norm gamma，数值/布局/
检查不变；逻辑新增常驻 `56,693,376 B`（embedding `56,623,104 B` + 61 个
gamma `70,272 B`）。A/B 使用同一 request、同一 package、同一 board、core
mask 1、FP16，交错运行 3 组 baseline/candidate，profiler 关闭。

分层 profiler（31 个实际 decode forwards）：

| 指标 | baseline | candidate |
|---|---:|---:|
| decode wall | 619.099 ms/fwd | 468.947 ms/fwd |
| runner embed | 125.407 ms/fwd | 0.172 ms/fwd |
| 30 layers total | 436.013 ms/fwd | 411.954 ms/fwd |
| lm_head plans | 55.701 ms/fwd | 55.877 ms/fwd |
| PlanCall wrapper total | 327.085 ms/fwd | 332.750 ms/fwd |
| C runtime reported total | 316.254 ms/fwd | 321.635 ms/fwd |
| host actions（cross-cut） | 88.427 ms/fwd | 89.159 ms/fwd |
| top-level measured residual | 0.137 ms/fwd | 0.121 ms/fwd |

最终无 profiler A/B：

| 指标 | baseline | candidate |
|---|---:|---:|
| TTFT median | 3,870.855 ms | 3,865.049 ms |
| generate wall median | 22,711.064 ms | 17,854.156 ms |
| decode forward | 608.435 ms/fwd | 455.652 ms/fwd |
| decode forward/s | 1.644 | 2.195 |
| full-generate token/s | 1.409 | 1.792 |
| peak RSS median | 1,237,348 kB | 1,243,836 kB |
| final vmRSS median | 1,182,176 kB | 1,233,236 kB |

generate-wall speedup `1.272x`，decode speedup `1.335x`。profiler 只用于解释
开销；C runtime 子区间只作为 `PlanCall` 父区间的细分，不能重复相加。逐元素
等价检查覆盖 44 个 arrays（capture 39 tensors + all-logits、TF logits、
incremental logits 等）且 44/44 相等，同时 capture/TF/incremental 请求的
token identity 一致。

证据：

- `results/20261030T013910Z-smollm2-v5-delivery/LATENCY-REPORT.md`
- `results/20261030T013910Z-smollm2-v5-delivery/latency.json`
- `results/20261030T013910Z-smollm2-v5-delivery/equivalence.json`
- `results/20261030T013910Z-smollm2-v5-delivery/board-profile-baseline/`
- `results/20261030T013910Z-smollm2-v5-delivery/board-profile-candidate/`
- `results/20261030T013910Z-smollm2-v5-delivery/ab/pair{1,2,3}-{baseline,candidate}/`

### 复验与回归

- verifier 负例：`tests/test_verify_smollm2.py` 14/14 PASS；附加 `verify/negative-matrix.json` 15/15，基线 exit=0/pass=true，数值/rank/shape/identity/quality 负例 exit=1/pass=false，complete 缺 quality exit=2，summarizer 拒绝失败结果 exit=1；
- 自动重放 `reuse`：从独立 spec hash 记录读取 V5 身份，complete PASS，gate 56/56，退出 0；
- 固定 V5 验收重跑：candidate 默认 Host 缓存路径上板，`complete_acceptance=true`、56/56 gate、exit 0；`verify-summary.json` SHA-256 `2c4165d8c48f339bb2a715670eec34eedad745fba38967a2cb1ec8d8f2f0672e`，acceptance bundle SHA-256 `37160a1920c35e841b863370b84f3319c3d3ae1688dfef85a760a168ece22111`；
- 重跑与归档 V5 board 数组逐元素比较：44/44 arrays 相等、19/19 generation request token lists 相等；PPL NLL 5/5 arrays 相等；
- generation stop/count/timing：`tests/test_smollm2_generate_timing.py` 3/3 PASS；
- host：432 tests collected，431 passed、1 skipped、71 subtests passed，退出 0；
- 本轮新增针对性测试：Host 权重缓存/分层计时 4/4、首次分叉诊断 2/2、自动重放身份/请求顺序 3/3；
- C：33/33 ctest PASS；
- board 回归：既有 MLP b1/b8/b32 与 TinyViT b1/b8 各 8 图 smoke，
  accuracy 1.0、top1 agreement 1.0。

### 证据与重放

- 汇总：`results/20261029T120416Z-smollm2-v5/manifest.json`
- verifier：`results/20261029T120416Z-smollm2-v5/verify/verify-summary.json`
- spec/policy/input/bundles：`results/20261029T120416Z-smollm2-v5/specs/`、
  `models/smollm2_numeric_acceptance_v5.json`、`models/smollm2_v5_inputs.json`
- 原始观测：`.data/evidence/smollm2-20261029T120416Z/`（149 MB 级大数组、
  calibration/acceptance board、references、official logits、PPL）
- 历史 V1/V2/V3/V4：`results/20261029T120416Z-smollm2-v5/history/`
- 旧 V2 manifest 的 spec 指针已修正为归档 spec，并保存 original pointer、
  original manifest SHA 和 correction 说明
- 完整 replay：`REPLAY.md#smollm2-135m-instruct`；可执行入口
  `scripts/replay-smollm2-v5.sh {reuse|rerun-fixed-acceptance|campaign}`
- 交付收口与优化证据：`results/20261030T013910Z-smollm2-v5-delivery/`
  （`DELIVERY-REPORT.md`、`DELIVERY-MANIFEST.json`、`LATENCY-REPORT.md`、
  `divergence/`、`equivalence.json`、`acceptance-equivalence.json`）
- 策略元数据勘误：`results/20261030T013910Z-smollm2-v5-delivery/POLICY-NOTE-ERRATUM.json`

### 非声明

只声明上述单板、单核、batch=1、FP16、容量 256 的固定 SmolLM2 几何。不声明
任意 LLM importer、动态 shape、device-resident KV、量化、并发、多板、故障后恢复、
RK1228 或性能优势。
