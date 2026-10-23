# Results

Each run creates an immutable directory. Only selected text checkpoints are
committed; tensor payloads, source bundles and local archives stay local (see
`.gitignore`). A text checkpoint is a report, not a replacement for the raw
payloads needed for independent numerical replay.

## Workspace checkpoint cleanup (2026-10-23)

- The current implementation checkpoint is board `20261023T144054Z-rk3588`
  (27 cases × 3 serial masks) and Host `20261023T131916Z-host` (213 pytest,
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




_The notes below are historical incident records.  The current Host verification
run is green at `20261023T131916Z-host` (213 pytest, 27/27 ctest)._

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
