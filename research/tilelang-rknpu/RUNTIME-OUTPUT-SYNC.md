# Output BO synchronization deduplication A/B

Date: 2026-10-27. The default path now performs one output PREP/FINI pair per
unique BO after each submit. On the RK3588 QK/PV cases and complete Tiny ViT
b8/b1, seven-round timing favors this path; every checked output is bitwise
equal. The frozen packed numeric replay also passes.

## What changed and why one sync per BO is valid

Only the output synchronization loop in `custom_ops.c` changed. It deduplicates
by `(fd, handle)` within one completed `execute_impl` call and also requires
matching `obj_addr` and size as a consistency guard. It does not add a cache,
public ABI or runtime flag. The
compile-time A/B control `CO_OUTPUT_SYNC_REFERENCE` preserves the previous
per-program PREP/FINI loop. Both libraries use the current default finite
scanner and `RK_PLAN_PREPARED_VALIDATION=0`.

[`rocket_bo_prep` and `rocket_bo_fini`](../upstream-components/rknpu-submit/src/rknpu_provider.c:654)
synchronize the complete object: offset zero through `page_up(bo->size)`. The
provider's [`submit_locked`](../upstream-components/rknpu-submit/src/rknpu_provider.c:904)
sets `RKNPU_JOB_BLOCK`; its documented driver contract says the submit returns
after job retirement ([provider README](../upstream-components/rknpu-submit/README.md:612)).
`CO_CHAIN` sends its whole task stream as one blocking submit. In
[`execute_impl`](../custom-ops/src/custom_ops.c:432), `CO_SEPARATE` recurses once
per program before syncing; the dedup set is local to each call, so it still
synchronizes each completed output before proceeding. The provider also submits
each unchained task as its own blocking job. Shared views in a BO therefore need
one cache-sync pair after a chained submit, while distinct BOs and separate
submits retain their own pairs.

The real `execute_impl` Host test verifies these counts with only the device
BO/submit boundary replaced. For two programs whose output views share a BO,
`CO_CHAIN` makes one output pair in the default build and two in the reference
build. Two distinct output BOs still make two pairs in both builds. `CO_SEPARATE`
with the shared BO makes two pairs in both builds because there are two submits.
PREP and FINI failures stop the loop at the failing operation. The test reports
`OUTPUT_SYNC checks=40 failures=0`; the new test is in CTest's **32/32 PASS**.
The targeted ASan/UBSan run passed `output_sync` (1/1).

## Board protocol and build identity

QK/PV b8 were measured first with seven outer rounds and 31 changed-input
frames per arm per round. Launch order alternated. A separate `battr` run for
each arm records 32 per-call attribution samples, including
`output_sync_us`. The model test then ran two identical compiled-graph runners
in one process, each with its own library and fd, over the same 256 test images;
b8 and b1 each ran seven rounds. Calls are interleaved with `stats=None` and
`RK_PLAN_PREPARED_VALIDATION=0`, with no adapter, proxy or pointer-array rebuild.
Per-call arm times, paired ratios and logits are retained. Each pair of logits
was compared bitwise.

The board was an RK3588 OPi 5 Plus, driver v0.9.8, CPU6 `performance` at
2.304 GHz, NPU `performance` at 1.0 GHz, DMC `performance` at 2.112 GHz, NPU
mask 1. These conditions are recorded in
[`board-environment.log`](results/20261027T130803Z-output-sync/board-environment.log);
the run did not modify frequency or governor state.

Fast and reference were built from the same staged source. The offline compile
audit reports **65** compile commands; every command matches after removing
`-DCO_OUTPUT_SYNC_REFERENCE`, which appears only in the reference build. The
actual caches, compiler commands and flags are under
[`artifacts/`](results/20261027T130803Z-output-sync/artifacts/).

Key source hashes:

- `custom_ops.c`: `cec430900429e1e94dd596c353f81672df15bec689ed64723ac076ffa43d16ec`
- `tests/c/output_sync_test.c`: recorded in
  [`host/source-hashes.sha256`](results/20261027T130803Z-output-sync/host/source-hashes.sha256)
- `CMakeLists.txt`, plans and tool hashes: `campaign/run-manifest.json`; compiled
  plan manifest SHA-256 is `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`.

The new fast library hash is `7fd5a94b200568b942c2920348d9fae8d4d50ebab7cefb282f4b24bd4f34d0d7`.
The old synchronization reference hash is `097a761f1ff66b1570876a74138caabf357b6b0f77d3d6a658a52d95bc86c50f`,
the same hash as the previous finite-scan fast library. This confirms that the
reference path retains that previously measured runtime state.

## Results

For isolated QK/PV, each latency is the median of seven per-round medians; the
paired ratio is the median of seven per-round ratios. Full-model arm latencies
are pooled medians over timed calls (224 per arm for b8, 1,792 for b1); their
paired ratios are the medians of seven paired per-round medians. Ratio is
`reference / fast`, so values above 1 favor deduplication. Ranges are the
seven per-round ratios; p10 is auxiliary.

| case | fast median (µs) | reference median (µs) | paired ratio median | ratio range | outputs |
|---|---:|---:|---:|---:|---|
| QK group b8 | 395.496 | 487.953 | 1.2307× | 1.1752–1.3319× | bitwise |
| PV group b8 | 381.787 | 477.453 | 1.2500× | 1.2485–1.3498× | bitwise |
| Tiny ViT b8 | 3416.395 | 3614.872 | 1.0502× | 1.0442–1.0597× | bitwise |
| Tiny ViT b1 | 1233.735 | 1288.276 | 1.0315× | 1.0163–1.0454× | bitwise |

QK round ratios are `1.229059, 1.230711, 1.225186, 1.175237, 1.231959,
1.331857, 1.230882`; PV ratios are `1.248475, 1.249618, 1.249238, 1.257844,
1.250001, 1.349849, 1.250572`.

The independent attribution sample measured median `output_sync_us` of **3.50
µs** for both QK and PV in the deduplicated build. The reference measured
**94.79 µs** on QK and **98.87 µs** on PV. Each attribution log has 32 calls;
raw lines and the stage summaries are in `profile/` and
`profile-summary.json`. Exact BO pair counts are established by the real
`execute_impl` Host test above; the board profile is the independent timing
measurement.

The per-round full-model ratios are b8 `1.054428, 1.059745, 1.047168, 1.057298,
1.050185, 1.049110, 1.044190` and b1 `1.019985, 1.016290, 1.035839, 1.031471,
1.039220, 1.045428, 1.025873`. Full per-call timings, ratios, logits and
stdout/stderr/exit codes are in
[`full-model/`](results/20261027T130803Z-output-sync/full-model/); isolated
raw samples are in [`rounds/`](results/20261027T130803Z-output-sync/rounds/).

## Verification

Host verification after relinking the runtime: CTest **32/32 passed** and
pytest **351 passed, 71 subtests**. Logs and exact commands are under
[`host/`](results/20261027T130803Z-output-sync/host/).
The focused ASan/UBSan run passed the `output_sync` target (1/1). From
`research/custom-ops`, it used
`ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize-rk -R '^output_sync$' --output-on-failure`;
the retained log is `host/sanitizer-output-sync.log`. The reused
finite-scan analyzer and compile checker also pass with their original default
CLI arguments on the prior finite-scan evidence; those compatibility outputs
are archived in `host/` without changing that earlier result.

The packed frozen numeric replay passed with **0 problems**: 33 layer outputs
and 5 profiles passed, and all top-1 comparisons matched. It used the packed
spec hash `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
and existing calibration observation hash
`c6f6bcd2722aaa99383baf4375fa83bde7010b4594adcacd8bab371dee6c6131`. No
recalibration or numeric limit change occurred. The acceptance observation and
verification output are under
[`20261027T131044Z-numeric-replay/`](results/20261027T131044Z-numeric-replay/).
The replay campaign retains the generic internal name `numeric_acceptance_train.json`;
its manifest hash matches the packed spec above. The replay library hash matches
the fast A/B library. The exact replay invocation was:
`env RK_FROZEN_CAMPAIGN="$PWD/research/tilelang-rknpu/results/20261027T093259Z-template-gap/numeric" RK_FROZEN_SPEC="$PWD/research/tilelang-rknpu/models/numeric_acceptance_packed.json" bash research/tilelang-rknpu/scripts/run-numeric-replay.sh`.

All commands, raw samples, logs, true exit codes, source/library hashes and
SHA256 inventories are retained in
[`results/20261027T130803Z-output-sync/README.md`](results/20261027T130803Z-output-sync/README.md).

To repeat the same-source A/B from the repository root, using the board
connection already configured in `.env`:

```bash
RK_AB_EXPERIMENT=output-sync RK_AB_REFERENCE_MACRO=CO_OUTPUT_SYNC_REFERENCE \
  bash research/tilelang-rknpu/scripts/run-finite-scan-ab.sh
```
