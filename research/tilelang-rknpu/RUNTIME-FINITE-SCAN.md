# Binary16 finite-scan A/B

Date: 2026-10-27. The default binary16 bit scan passes the board A/B and the
frozen packed numeric replay. The seven-round isolated QK/PV samples and both
complete Tiny ViT batches show stable gains, with bitwise-equal outputs. The
root-approved implementation is suitable for the default runtime path.

## Protocol and identity

The fast library uses the binary16 exponent-bit scan. The reference library
uses the old per-element `isfinite` loop selected by the source-only
`RK_PLAN_FINITE_SCAN_REFERENCE` macro. Both were built from the same staged
source and CMake configuration. The offline audit checks all 63 compile
commands, the runtime target flags and the CMake cache: the reference macro is
present only in the reference library; removing it makes every compile command
identical. Full details and the independent audit output are in
[`artifacts/`](results/20261027T122305Z-finite-scan/artifacts/).

All A/B runs set `RK_PLAN_PREPARED_VALIDATION=0`. The isolated cases call the
native `rk_workspace_run` probe path. Each of seven rounds runs the same 31
changed-input frames per arm; arm launch order alternates by round. The full
model uses two runners over the same compiled graph and weights in one process,
with separate library handles and file descriptors. Calls are interleaved,
`stats=None`, with no adapter, node proxy or pointer-array reconstruction. b1
was run after b8 showed a positive gain. Every isolated output file and every
model logit array compares bitwise equal.

The RK3588 OPi 5 Plus ran with CPU6 pinned at `performance` / 2.304 GHz, NPU
`performance` / 1.0 GHz, DMC `performance` / 2.112 GHz, NPU mask 1, and driver
v0.9.8. These values are recorded in
[`board-environment.log`](results/20261027T122305Z-finite-scan/board-environment.log).
No frequency or governor settings were changed.

Source hashes: `rk_plan_runtime.c` `3f807a406575e282cfb02d2b130a0992569a89864d8ba1ff8fdd3dd35ab77302`;
`rk_plan_download.h` `2bed0c246954f6a71410884532a2af5c45f433f15ada9697585a27d4b7ca63a8`.
The staged package manifest is `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`.
The board library hashes are in
[`artifacts/fast-library.sha256`](results/20261027T122305Z-finite-scan/artifacts/fast-library.sha256)
and [`artifacts/reference-library.sha256`](results/20261027T122305Z-finite-scan/artifacts/reference-library.sha256).

## Performance

Ratio is `reference / fast`, so values above 1 favor the binary16 bit scan.
For isolated QK/PV, arm latency is the median of seven per-round medians; the
paired ratio is the median of those seven round ratios. For full-model runs,
arm latency is the pooled median of per-call samples; the paired ratio remains
the median of seven paired per-round medians. Ranges show all round ratios;
p10 is auxiliary only.

| case | fast median (µs) | reference median (µs) | paired ratio median | ratio range | outputs |
|---|---:|---:|---:|---:|---|
| QK group b8 | 486.786 | 848.156 | 1.7382× | 1.7122–1.8054× | bitwise |
| PV group b8 | 477.160 | 669.950 | 1.4077× | 1.4004–1.4088× | bitwise |
| Tiny ViT b8 | 3883.348 | 4033.408 | 1.0405× | 1.0353–1.0464× | bitwise |
| Tiny ViT b1 | 1252.984 | 1306.213 | 1.0330× | 1.0289–1.0508× | bitwise |

The isolated arm medians are the median of seven per-round medians. Full-model
arm medians are pooled over all timed calls (224 per arm for b8 and 1,792 for
b1); their primary ratios are medians of the seven paired per-round medians.
The per-round ratio arrays are b8 `1.037817, 1.035269, 1.040504, 1.046375,
1.039839, 1.045495, 1.043959` and b1 `1.032629, 1.030997, 1.032981, 1.028851,
1.050765, 1.047003, 1.040043`. Per-call ratios are noisier and include some
individual reversals; the complete paired per-call samples are retained in
`full-model-raw/b{8,1}/result.json`.

The QK/PV raw timing lines and outputs are under
[`rounds/`](results/20261027T122305Z-finite-scan/rounds/); per-call model
timings, paired ratios, logits and stdout/stderr/exit evidence are under
[`full-model-raw/`](results/20261027T122305Z-finite-scan/full-model-raw/). The
initial summary-only model capture remains under
[`full-model/`](results/20261027T122305Z-finite-scan/full-model/). The complete
command log is [`commands.log`](results/20261027T122305Z-finite-scan/commands.log).
The raw-capture harness SHA-256 is
`c17a042ff88e0bf002473be9356702366b2270e7abd52cf6d02039a3a95b3f16`; it
reuses the already built libraries and records each timed call.

## Verification

Host verification after relinking the libraries: pytest **351 passed, 71
subtests**, CTest **31/31 passed**. Exact commands and captured logs are under
[`host/`](results/20261027T122305Z-finite-scan/host/). The Host finite-scan
coverage includes NaN and both infinity signs. The targeted ASan/UBSan rerun
passed both `fp16_scan` (174 checks) and `rk_plan_host` (806 checks).

The packed frozen numeric replay passed with **0 problems**: 33 layer outputs
and 5 model profiles passed, and all profile top-1 results matched. It reused
the frozen packed spec (SHA-256
`2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`) and the
existing calibration observation (SHA-256
`c6f6bcd2722aaa99383baf4375fa83bde7010b4594adcacd8bab371dee6c6131`). No
calibration or threshold changes were made. The replay worker expects the
generic internal filename `numeric_acceptance_train.json`; the archived file's
hash matches the packed frozen spec above. The acceptance observation and
verifier output are in
[`20261027T122757Z-numeric-replay/`](results/20261027T122757Z-numeric-replay/).
The replay-built library hash matches the A/B fast library:
`097a761f1ff66b1570876a74138caabf357b6b0f77d3d6a658a52d95bc86c50f`.

The first board attempt, `20261027T122026Z-finite-scan`, stopped before
measurement because a worker sysfs glob did not match the NPU governor path;
read-only inspection confirmed the expected board frequencies, and no device
setting was changed. The successful run is `20261027T122305Z-finite-scan`.
The first packed replay attempt was also retained; it used a campaign filename
the native worker did not select. The next run used the worker's existing
frozen-spec path and passed. The success and failure directories remain
separate.

Large raw payloads and libraries remain local under the result directories;
their hash inventories are stored beside them. See
[`results/README.md`](results/README.md) for the artifact map.
