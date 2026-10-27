# Results

Each run creates an immutable directory. Only selected text checkpoints are
committed; tensor payloads, source bundles and local archives stay local (see
`.gitignore`). A text checkpoint is a report, not a replacement for the raw
payloads needed for independent numerical replay.

## Vendor RKNN baseline checkpoint (2026-10-27, v14)

- `results/20261027T015302Z-vendor/`: two rounds, full 10,000-image test split
  for all five profiles, current custom backend vs vendor RKNN 2.3.2 vs CPU
  FP32.  Vendor warm latency is **2.40x/2.54x/2.86x** faster than current on
  MLP b32/b8/b1 and **2.53x/3.01x** faster on Tiny ViT b8/b1.
- CPU is faster than current on MLP b1 (4.86x) and Tiny ViT b1 (3.51x); the
  multi-submit/Host-boundary gap remains the main issue.
- Vendor accuracy/top-1 match CPU for all five profiles.  Vendor MLP max logit
  error 0.03049 is within packed 0.0625; Tiny ViT 0.12277/0.10715 exceeds it,
  so vendor fusion is not a drop-in packed per-logit equivalent.
- Artifacts/commands: [vendor/README.md](../vendor/README.md); analysis:
  `analysis.md` / `analysis.json` inside the result directory.
- `vendor-perf/`: per-layer placement probe; Tiny ViT b8 is 50 NPU layers +
  5 CPU layers (input/output, final reshape, two RMSNorm Div), showing that the
  vendor compiler absorbs most of the Host boundaries.
- The pre-optimisation baseline row in that analysis is from
  `results/20261026T120625Z-final/`; the current/vendor/CPU columns share the
  vendor campaign.

## Final model delivery checkpoint (2026-10-26, final)

- `results/20261026T120625Z-final/`: three interleaved rounds, full 10,000-image
  MNIST test split for all five static profiles, baseline `42f870f` vs current
  runtime `60f4a82`.  Warm b8 34.631 → **10.603 ms** (3.266x), b1 7.138 →
  **3.329 ms** (2.144x); full five-profile evaluation wall 224.214 → **148.204 s**
  (1.513x).  Both arms have identical raw logits in every round.
- Accuracy: MLP **0.98310**, Tiny ViT **0.98210**; top-1 agreement **1.00000**
  for every profile.  Fresh current-build packed numeric replay:
  **PASS / 0 problems** (33 layers, 5 profile logits), no recalibration.
- Creation cost is reported separately (Tiny ViT b8 18.0 → 62.2 ms, amortised);
  no cold-start speedup is claimed.
- Summary: [OPTIMIZATION-SUMMARY.md](../OPTIMIZATION-SUMMARY.md); replay:
  `RK_FINAL_LIMIT=10000 RK_FINAL_ROUNDS=3 ./scripts/run-final-delivery.sh`.

## QKV mapped-input checkpoint (2026-10-26, v13)

Seventh optimisation package; report:
[OPTIMIZATION-RESULTS-7.md](../OPTIMIZATION-RESULTS-7.md).

- `results/20261026T113808Z-qkvmap/`: split_heads materialization vs mapped
  QKV input, 4 rounds x 32 runs.  b8 10.784 -> **10.760 ms** (1.0022x, mixed
  rounds), b1 3.521 -> **3.428 ms** (1.0270x).  Structure/transfers unchanged.
  Production removes split_heads and three q/k/v Host arrays; capture still
  materialises them for diagnostics.
- `results/20261026T113650Z-numeric-replay/`: packed frozen replay,
  **PASS / 0 problems**, no recalibration.
- `models/compiled/tiny_vit_bN_splitheads.rkhybrid.json` remains the explicit
  materialized control; `group_layout` v1 is checked by Python and C plan
  validation (source capacity, offsets, transpose, bounds).
- Host verification for this state: 267 pytest, 30/30 ctest.

## Python binding-reuse checkpoint (2026-10-26, v12)

Sixth optimisation package; report:
[OPTIMIZATION-RESULTS-6.md](../OPTIMIZATION-RESULTS-6.md).

- `results/20261026T105439Z-pybind/`: ctypes binding descriptor reuse vs
  per-call `make_binding`, 4 rounds x 32 runs, same payload/source.
  b8 11.579 -> **10.043 ms** (1.153x), b1 3.785 -> **3.356 ms** (1.128x);
  four-round direction consistent; structure and transfer bytes unchanged.
  The descriptor cache is now protected through the synchronous call by a
  non-blocking BUSY-preserving lock, covering single and flat group caches.
- `results/20261026T105527Z-numeric-replay/`: packed frozen replay,
  **PASS / 0 problems**, no recalibration.
- Download-sync coalescing was measured in
  `results/20261026T095856Z-download/` but showed no stable end-to-end win and
  was reverted; per-slice download remains the default.
- Residual+reshape+mean tail fusion was measured in
  `results/20261026T110224Z-tail/` and `results/20261026T110622Z-cumulative/`;
  b8 contradicted the isolated gain against binding-only, so it was reverted.
  Default keeps the three original Host actions.
- `results/20261026T111038Z-pybind/`: three-arm ctypes comparison
  (control / descriptors / prepared arrays).  Descriptors are the adopted
  improvement (b8 1.163x, b1 1.069x); caller-owned prepared arrays were neutral
  on b8 and only helped b1, so that layer was reverted to avoid maintaining a
  second cache/lifetime path.
- Host verification for this state: 265 pytest, 30/30 ctest.

## Upload-policy contract and layout checkpoint (2026-10-26, v11)

Fifth optimisation package; report:
[OPTIMIZATION-RESULTS-5.md](../OPTIMIZATION-RESULTS-5.md).

- `results/20261026T094847Z-layout/`: fixed 16x16 CUBE/WEIGHT pack vs generic,
  coalescing on, 4 rounds x 32 runs.  b8 11.637 -> **11.414 ms** (1.0195x);
  b1 3.706 -> **3.663 ms** (1.0117x).  b8 layout 439 -> 260 us; b1 56.9 -> 32.1 us.
- `results/20261026T094941Z-numeric-replay/`: packed frozen replay,
  **PASS / 0 problems**, no recalibration.
- Runtime fixes: >128 consecutive uploads safely fall back to per-slice path;
  upload policies are snapshotted at workspace creation and reported in
  attribution; coalescing is restricted to the verified independent-MM group.
- Host verification for this state: 263 pytest, 30/30 ctest.

## QK/PV upload-sync checkpoint (2026-10-26, v10)

Fourth optimisation package, shared-group upload preparation; report:
[OPTIMIZATION-RESULTS-4.md](../OPTIMIZATION-RESULTS-4.md).

- `results/20261026T093305Z-upload/`: coalesced vs per-slice prep/fini,
  4 rounds x 32 runs.  b8 11.969 -> **11.721 ms** (1.021x), b1 3.718 ->
  **3.660 ms** (1.016x); tasks/submits and transfer bytes unchanged.
- b8 pack split: 1122.9 -> 597.6 us; prep 300.4 -> 41.1 us,
  fini 273.0 -> 56.3 us; prep/fini calls 134 -> 10.  Layout packing unchanged.
- `results/20261026T093427Z-numeric-replay/`: packed frozen policy replay,
  **PASS / 0 problems**, no recalibration.
- Default remains coalesced; `RK_PLAN_UPLOAD_COALESCE=0` is the explicit
  per-slice control.  Host verification for this state: 261 pytest, 30/30 ctest.

## RMSNorm-boundary checkpoint (2026-10-26, v9)

Third optimisation package, RMSNorm portion of plan section 6/7 D; report:
[OPTIMIZATION-RESULTS-3.md](../OPTIMIZATION-RESULTS-3.md).

- `results/20261026T091322Z-rmsnorm/`: default packed FFN + chain_shared,
  `control` vs fused residual+RMSNorm Host action, 4 rounds x 32 runs.
  b8 11.885 -> 11.921 ms (0.997x), b1 3.587 -> 3.628 ms (0.989x); structure and
  transfers identical.  Default retained.
- `results/20261026T091548Z-numeric-replay/` and
  `...T091612Z-numeric-replay/`: default and fused-manifest replays under the
  existing packed tolerances, both **PASS / 0 problems**, no recalibration.
- `results/20261026T091954Z-numeric/`: separately mapped `_deviceffn` acceptance,
  **PASS / 0 problems** under its independent `atol=0.125` policy.
- Existing `co_row_ops` RMSNorm probe: 0.854 ms/run (b8) and 0.266 ms/run (b1)
  with 2 submits / 3 tasks / 2 uploads / 2 downloads, slower than the current
  Host RMSNorm + adjacent upload.  No device RMSNorm task was added.
- Host verification for this state: 260 pytest, 30/30 ctest.

## QK/PV policy revalidation on packed FFN (2026-10-26, v8)

- `results/20261026T085859Z-policy/`: packed FFN default, QK/PV arms
  `b1 / separate / chain_shared`, 4 rounds x 32 runs, pinned core 1.
  b8 p50 14.584 / 22.344 / **11.692 ms** (chain_shared 1.247x vs B1);
  b1 3.849 / 4.700 / **3.613 ms** (1.065x; one round reversed).
  b8 submits 70 -> 8 with task/upload/download bytes unchanged.
- `results/20261026T085820Z-numeric-replay/`: packed frozen policy replayed
  without recalibration, **PASS / 0 problems**; 33 layer gates and 5 profile
  logit gates under the actual model inputs.
- `capture_only` is now enforced at manifest load time: ordinary nodes cannot
  read its outputs and it cannot be a model output. The trace plan remains
  diagnostic-only evidence, not a sample of production-plan internals.
- Host verification for this state: see the final pytest/ctest checkpoint.

## FFN boundary checkpoint (2026-10-26, v7)

Second optimisation package from
[TRANSFORMER-OPTIMIZATION-PLAN.md](../TRANSFORMER-OPTIMIZATION-PLAN.md) section 6;
the implementation report is
[OPTIMIZATION-RESULTS-2.md](../OPTIMIZATION-RESULTS-2.md).

- `results/20261026T085009Z-ffn/`: RK3588 core 1, three interleaved arms
  (packed / device / host), 4 rounds x 32 runs, pinned CPU.  Default packed FFN:
  b8 12.176 -> **11.772 ms** (1.034x), b1 3.827 -> **3.585 ms** (1.067x).
  Per b8 call: tasks 71 -> 70, submits 9 -> 8, upload bytes 312832 -> 296448;
  downloads stay byte-identical.  Packed logits and all FFN trace tensors are
  bitwise equal to the old host-SwiGLU path.
- `results/20261026T084825Z-numeric/`: frozen packed-path numeric acceptance,
  **PASS / 0 problems**; tolerances are item-for-item identical to v1.
  `results/20261026T085111Z-numeric-replay/` is the post-freeze replay, also
  **PASS / 0 problems**.
- `results/20261026T084901Z-numeric/`: separate fully device-resident FFN policy
  (explicit opt-in `_deviceffn` manifest), independently recalibrated:
  `act/ffn/h2/h2_tokens` atol 0.0625 -> 0.125; **PASS / 0 problems**.  This path
  is not the default and does not inherit the packed bit-exact claim.
- Host verification for this state: 258 pytest, 30/30 ctest.

## Bridge-envelope optimisation checkpoint (2026-10-26, v5)

First optimisation package from
[TRANSFORMER-OPTIMIZATION-PLAN.md](../TRANSFORMER-OPTIMIZATION-PLAN.md); the
implementation report (v5, after four read-only reviews) is
[OPTIMIZATION-RESULTS-1.md](../OPTIMIZATION-RESULTS-1.md).

- `results/20261026T055654Z-envelope/`: RK3588 core 1, three arms measured
  **round by round** (4 rounds x 32 runs, pooled to 128 samples) on a pinned CPU,
  against the current plan package: `baseline` (HEAD source tree), `candidate`
  (working tree) and `noprobe` (same sources, timing probes compiled out).
  Tiny Transformer b8 33.844 -> **14.768 ms** (2.292x, 236.4 -> 541.7 img/s),
  b1 7.199 -> **4.139 ms** (1.739x), MNIST MLP b32 2.832 -> **2.198 ms**
  (1.289x). Throughput is `batch / pooled p50`; the per-arm envelope is a single
  representative run and is labelled with its round. Tasks and submits are
  asserted identical per arm (b8: 71/71, 135 uploads, 71 downloads,
  312832 B up, 230144 B down).
- `results/20261026T053818Z-numeric/`: the plan package was regenerated with the
  corrected Host-requirement formula (runtime scratch now counted by the planner
  and by the C verifier, with a fail-closed check at workspace creation). The
  numeric acceptance was re-run **under the frozen policy**: layer and logit
  tolerances are byte-identical to v1 (diffed), only the package hashes moved.
  Verifier output: **PASS / 0 problems**. The host-only 10,000-image mock
  evaluation is bit-identical to the milestone record.
- `results/20261026T033032Z-b2probe/`: corrected B2 chain-validity probe. Two
  earlier variants were wrong (shared reference buffer; self-comparing negative
  control); the current one stores one reference per task, compares against its
  own slice, and its comparison logic has a host test
  (`tests/c/task_ref_test.c`). With that, every shape passes: 32 independent
  matmuls in one chained submit are bit-exact at 16.0 us/task versus 105.5 us
  for 32 separate submits (per-task costs; the separate-side ratios vary a lot
  between rounds). B2 is device-viable but **not implemented**: it still needs
  static-repeat plan expansion, an `independent_mm_v1` recipe and manifest
  versioning.
- `results/20261026T021229Z-envelope/`, `...T020729Z-envelope/` and
  `results/20261026T030156Z-numeric-replay/`, `...T033318Z-numeric-replay/`
  remain as measurement history (sequential-unpinned campaign, intermediate
  build, and replays against earlier package states).
- `results/20261026T020845Z-numeric/` (local only) is an aborted attempt to
  re-run the full train calibration: `tools/calibrate_numeric_train.py` refuses
  to overwrite an already frozen spec.
- Host verification for this state: 244 pytest / 41 subtests, 30/30 ctest,
  787 C host checks, an exhaustive 65536-pattern FP16 scan test, a per-task
  reference-comparison test, and ASan/UBSan clean host tests.

## Current checkpoint (2026-10-24)

- Complete-model task/timing campaign: `results/20261024T071144Z-models/`.
- Train-calibrated numeric campaign:
  `results/20261024T091634Z-numeric/`; calibration uses 32 unlabeled train images,
  then a disjoint 32-image test subset is observed after freezing thresholds.
  The verifier also rechecks existing full-test raw logits under the same limits.
- Final Host verification: `results/20261024T092339Z-host/` (235 pytest /
  41 subtests, 27/27 ctest, 26 contract + 711 host + 2,400,033 graph C checks,
  explicit ASan/UBSan).
- Historical test-calibrated numeric envelope:
  `models/numeric_acceptance.json` and `evidence/numeric-calibration/`; retained
  only as regression evidence, not independent acceptance.
- Old model performance fields: see
  `results/20261024T031249Z-models/PERFORMANCE-RETRACTION.md`.

## Workspace checkpoint cleanup (2026-10-23)

- The current implementation checkpoint is board `20261023T144054Z-rk3588`
  (27 cases × 3 serial masks) and Host `20261024T032922Z-host` (219 pytest,
  27/27 ctest). The cleanup preflight is `20261023T122811Z-host`; the final cleanup
  verification is `20261023T123836Z-host` (212 pytest, 26/26 ctest,
  C and Python sanitizer steps passed).
- Twelve expanded checkpoints retain the dependency baseline, the known negative
  controls, the EW failure and the main hardware milestones. Their run contents
  were not rewritten during cleanup.
- Thirty-four other historical runs were packed into the ignored local archive
  `.archive/historical-runs-20261023.tar.gz`. Every archived regular file was
  compared by SHA-256 with its expanded original before the expanded copy was
  removed. [ARCHIVE-INDEX.json](ARCHIVE-INDEX.json) records run IDs, outcomes,
  archive hash and restoration command. An archived path mentioned by an older
  note can be restored locally; the compressed archive is not distributed by Git.
- Superseded exploratory mock logs are retained locally in
  `.archive/superseded-mock-debug-logs.tar.gz`; they are not current sanitizer
  evidence. Active regression tests and the final campaign replace those probes.
- Four automatically generated vendor `check[0-9]_*.onnx` optimization
  intermediates (481,534,183 bytes total) were discarded. Model generation and
  conversion scripts are retained. Downloaded SDK/source material, the locked
  Python environment, board raw outputs and source bundles were preserved.
- The redundant `rknpu_plan_mock_asan` CMake target was removed. Sanitizer workers
  use `rknpu_plan_mock` built with all dependencies in the separate sanitizer
  build directory, as the verification script already requires.


## Host verification runs

Produced by `scripts/run-verification.sh` (no device access):

- `commands.log` — every command, its start time and its exit code
- `environment.log` — git HEAD, scoped `git status`, toolchain and Python package
  identity, optional `.pydeps` distribution list
- `source-hashes.json` — SHA-256 of every source/test/schema/capability file
- `manifest.json` — machine-readable step exit codes, C check counts, ctest ratio
  and pytest pass/skip counts
- `summary.md` — human-readable summary
- one `*.log` per step (configure, build, ctest, contract, host, sanitizer,
  pytest)

A host run proves Host behaviour only.  It is **not** an RK3588 acceptance report
and never claims board results.

Guarantees encoded in `scripts/run-verification.sh`:

- the Python/`LD_LIBRARY_PATH` environment is exported **before** identity capture,
  so `environment.log` describes what was actually tested; duplicate distribution
  names fail the run;
- both build trees are recreated from scratch (use `--keep-build` to opt out), and
  the sanitizer is enabled **explicitly** on the configure line with
  `-DCMAKE_C_FLAGS`/`-DCMAKE_EXE_LINKER_FLAGS`/`-DCMAKE_SHARED_LINKER_FLAGS`;
- the compiler and linker lines that were really used are archived under
  `compile-flags/`, and the run fails if the sanitizer flag is missing from them;
- evidence generation is itself a checked step: a missing artifact or an
  unparseable statistic sets `evidence_complete: false` and fails the run while
  still writing `manifest.json`/`summary.md` so a failed run stays auditable.


## Complete-model milestone checkpoint (2026-10-24)

- **Model package:** `models/manifest.json`, `models/mnist_mlp.npz`,
  `models/tiny_vit.npz`, `models/numeric_golden.npz`,
  `models/numeric_acceptance.json`, and `models/compiled/` (RKPlan plus
  versioned rkhybrid manifests).
- **Final board campaign:** `results/20261024T071144Z-models/`; core 1 ran the
  complete 10,000-image MNIST test split for both models, while masks 2 and 4
  ran the fixed 512-image admission subset.  Accuracy: MLP 98.31%, Tiny
  Transformer 98.21%; CPU/declared top-1 agreement 100%; independent verifier
  PASS, including deterministic per-layer traces and a same-top-1 amplified
  logits rejection test.
- **Host/mock evidence:** `evidence/model_eval_mock_v2.json`,
  `evidence/cpu_benchmark_v2.json`, and `evidence/numeric-calibration/`.
  Raw board logits and layer traces are retained with the final campaign.
- **Final Host checks:** `results/20261024T092339Z-host/` is PASS with
  **235 pytest / 41 subtests, 27/27 ctest**, 26 contract + 711 host +
  2,400,033 graph C checks, and explicit ASan/UBSan.
- **Numeric acceptance:** `results/20261024T091634Z-numeric/` PASS with
  train-only calibration, disjoint test observation, MLP hidden-layer traces and
  the existing full-test raw-logit recheck.  The earlier test-calibrated envelope
  is retained only as regression evidence.
- **Timing/performance:** schema-v2 timing excludes reference calculation,
  measures all registered Host nodes and uses a matched static batch schedule
  for the explicitly single-threaded CPU baseline.
- **Historical report:** `results/20261024T031249Z-models/` is retained, but its
  `wall_seconds`/`throughput_images_s` fields are invalid; see
  `PERFORMANCE-RETRACTION.md`.
- **Single-core scope:** masks 1/2/4 are serial, not concurrent throughput.
  The first-round Transformer Host bridge cost is reported in
  `MODEL-MILESTONE-STATUS.md`.

## P0 / regression records

| run | mode | dependency tree | result |
|---|---|---|---|
| `20261022T130044Z-host` | fresh build | `.pydeps-p0` (28 hash-locked artifacts), `.venv-p0` interpreter | **pass**: 26/26 ctest, 208 + 2,400,033 C checks, 130 pytest, 0 failed, 0 skipped, isolation OK |
| `20261022T084651Z-host` | `--keep-build` | same | **pass**: reused the kept trees, same counts |
| `20261021T114612Z-host` | fresh build | same | pass at the 60-test state (referenced by the review) |

Every run recreated or validated its build trees, used the isolated interpreter and
recorded `isolation.log`, `compile-flags/` and source hashes.  The earlier `.pydeps`
wheel-baseline run produced the same counts and is recorded in
`provenance.json`.

## Negative controls

All three are intentional failures kept as evidence; none is a verification pass.

| run | what was injected | expected outcome |
|---|---|---|
| `20261020T085356Z-host` | a `cmake` that always fails | `overall_exit: 1`, `evidence_complete: false`, failing step exit codes recorded |
| `20261021T114712Z-host` | `--keep-build` **before** the fix | exit 1: the script demanded a cache-free tree even in keep mode (review reproduction) |
| `20261021T115302Z-host` | `--keep-build` **after** the fix, with the sanitizer flags stripped from the kept cache | exit 1: "kept sanitizer tree has no sanitizer flags" |

## RK3588 board admission runs

Produced by `scripts/run-board-admission.sh` (credentials read from the ignored
repository `.env`; the script fails closed if any credential value appears in the
result tree):

- `campaign/manifest.json` — plans, case set, profile hash, runtime ABI, layout
  expectations and numeric tolerance;
- `campaign/plans/*.rkplan.json` — Host-generated immutable plans;
- `campaign/cases/*.npz` — local tensor payloads (ignored by git);
- `board/core-{1,2,4}/` — real-device results for the three single-core masks;
- `board/core-*/environment.json` — driver, chip/model, ABI and NPU state;
- `board/core-*/*.json` — per-case task/submit/transfer counts, numeric/guard
  outcome and library identity;
- `board/core-*/outputs/*.npz` — raw per-round hardware outputs (git-ignored);
- `campaign/cases/*.npz` — inputs plus declared staged-FP16 and unquantized
  float64 math references per round (git-ignored);
- `source-bundle.tgz` / `source-bundle.sha256` — exact source deployed;
- `board/library.sha256` — remote recorded loaded-production-library hash
  (not independently rehashed by the Host verifier);
- `board/summary.json` / `board/summary.md` — independent fail-closed campaign
  gate (rounds, output sets/shapes, declared numerics, task/submit counts,
  chain/separate bytes, target/profile/ABI, source-bundle hash recomputation,
  password scan).

Latest campaign: `20261023T144054Z-rk3588/` on RK3588 OPi 5 Plus, driver v0.9.8,
runtime ABI 1: **27 cases × 3 serial single-core masks = 81/81 PASS**.  Coverage
includes MatMul, projection and staged-FP16 FFN chain/separate, one explicit
guarded ragged tail, scalar/per-channel broadcast, independent activation, a
256x256 LUT split, dual output, M/N multi-tile, K=512/257 sharding, tiled per-node
projection/FFN/branch/activation, real multi-block TileLang staged
projection/FFN/branch, and the full board CLI path.  This latest campaign was
regenerated with the source-built TileLang/TVM tree; the wheel and source-built
plans are semantically equal after excluding source-span/hash metadata.  Numeric
guards remain included.
Publishing is preceded by the existing `co_guard_check` for every data BO; a Host
corruption regression proves failure does not publish and poisons the workspace.
The declared-path acceptance formula and its 8e-3/8e-3 provenance are recorded in
`campaign/manifest.json`; FP64 math references are stored separately but are not
the acceptance predicate.  This is finite representative small-shape admission
evidence.  P0 source rebuild is complete.  Real-device recovery after an unknown
fault is explicitly out of scope, unmeasured and not claimed; the accepted failure
contract is quarantine and stop.  Offline recomputation from archived
raw outputs is in
`recomputed/summary.md`; it recomputes both declared and unquantized float64 math
references.  Multi-board, concurrent three-core, performance and long-duration
stress are extension validation, not P8 blockers.




_The notes below are historical incident records.  The MVP Host run at that time
was `20261024T032922Z-host` (219 pytest, 27/27 ctest); the final complete-model
Host verification is `20261024T092339Z-host` (235 pytest / 41 subtests, 27/27
ctest, sanitizer enabled)._

**FFN 数值对照已通过**（20261022T130044Z-host）。根因：mock seam 把模板标签写进 `co_program.flags`，而该字段是生产语义（`co_schedule.c:38` 要求为 0，`custom_ops.c:344` 用 `ROCKET_JOB_PPU_DONE`），导致 LUT/EW/MM 误分派（511/512 元素错）。改为 mock 私有旁表（按程序地址记模板与 mul）后，剩余差异仅为参考实现的收窄顺序；按"先 sigmoid→收紧、再乘 gate→收紧 SiLU"修正参考后逐元素严格相等。


**已删除的验收记录**：`results/20261022T091529Z-host`（2026-10-22 09:15Z 生成）——它是 mock 合并过程中的真实失败态（`rk_plan_host` 93 checks / 1 failure，`overall_exit=1`），原因是我批量重命名宏时误伤了诊断 needle 文本与一个早退条件；修复后该记录被删除。**删除事实登记在此**，当时未保留原始日志，故不恢复、不编造内容。

**当前状态（2026-10-22 13:09Z）：验证流水线为红灯，未关闭。**
`results/20261022T130920Z-host` 中 `configure` 步骤失败：脚本内调用 cmake 时报告
`The C compiler "/usr/bin/cc" is not able to compile a simple test program`，
而在同一时刻手动 `cmake -S research/custom-ops -B /tmp/probe-build` **可以成功**。
因此怀疑与脚本内环境（早期步骤 exit 1）或残留 build 目录状态有关，**尚未定位**。
此前加入的 sanitizer worker 三个步骤已从脚本中**回退**（其插桩 mock 构建必须排在
`build-sanitize` 之后，我之前的插入位置错误）。
独立运行仍然通过：`pytest tests/test_ws_poison.py` → 2 passed。
不完整的“仅 mock 插桩”历史记录仍保留，未覆盖。

**红灯根因已确认（2026-10-22）**：我此前在交互 shell 里
与 //，这些变量被后续
继承，导致 **cc1 自身在 ASan 下运行**，configure 的编译器探针失败。
修复：脚本新增  记录九个相关变量，并对普通步骤**unset**
；
CC/CFLAGS/LDFLAGS 保留编译器选择。恢复后  exit 0。

**已删除的 sanitizer 归档登记**：
（含 poison-worker.log、ffn-worker.log、compile-flags.txt、mock-library.sha256、commands.txt）
在本轮清理旧记录时被删除；其 sha256 为 。
**原始日志未保留，不重建历史**；新的 sanitizer 运行需另建目录。

Legacy `custom-ops/build-host` and `build-sanitize` trees and Python test caches
were discarded after verification. The current `build-host-mvp` and
`build-sanitize-rk` trees and pinned dependency environments remain available.

## Static-repeat policy wiring (2026-10-26, v6)

QK/PV now execute as one pre-expanded plan with a single chained submit, selected
by an explicit manifest field (`repeat_submit: shared_group` +
`group_plan: attn_matmul_group`); the `separate` and B1 paths stay available as
runner controls and are asserted bit-identical.

- `results/20261026T081034Z-policy/`: end-to-end campaign over the three policies
  (`b1`, `separate`, `manifest`), arms alternated round by round on a pinned CPU.
  Tiny Transformer b8: b1 14.940 ms, separate 22.622 ms, **manifest (shared group)
  12.350 ms** -- 1.210x over B1, four rounds 1.20/1.26/1.20/1.22, throughput
  535.5 -> 647.7 img/s. b1 profile: b1 4.143, separate 4.904, manifest
  **3.842 ms**. Device structure: submit_count 71 -> 9, task count 71 unchanged,
  uploads/downloads/bytes unchanged.
  C-side attribution (single sample, round 1): 8.522 -> 5.207 ms total, with
  submit_wait 5.008 -> 1.746 ms; guard unchanged (0.321 -> 0.293 ms) because the
  shared layout keeps three BOs per group.
- `results/20261026T081153Z-numeric/`: numeric acceptance re-run under the frozen
  policy after the plan package changed (new `*_qk_group` plans, regenerated
  manifests). Layer and logit tolerances are byte-identical to v1 (diffed), the
  verifier reports **PASS / 0 problems**, and the gates include
  `tiny_vit.scores_heads` (QK output) and `tiny_vit.attn_heads` (PV output), i.e.
  the QK/PV node outputs themselves were compared under real model inputs.
- Default policy: `shared_group` for the QK/PV nodes.  It is bit-identical to both
  controls, passes the numeric gates, and is 1.21x faster than B1 end to end; the
  controls remain one flag away (`HybridRunner(repeat_submit="b1"|"separate")`).
