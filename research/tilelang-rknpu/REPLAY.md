# Replay commands

All commands run from `research/tilelang-rknpu` unless stated otherwise.

## Host verification

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh --keep-build
```

Current evidence is recorded after the final rerun in
`results/20261024T072125Z-host/` (231 pytest passed, 27/27 ctest, 26 contract + 711 host + 2,400,033 graph
C checks, sanitizer explicitly enabled; see its `manifest.json` and `summary.md`).

## Complete-model build and evaluation

```bash
cd research/tilelang-rknpu

PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
python3 tools/compile_models.py

PYTHONPATH=$PWD/python python3 tools/build_numeric_acceptance.py

PYTHONPATH=$PWD/python RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
python3 tools/evaluate_models.py --backend mock --models both \
  --global-limit 10000 --profile-samples 512 \
  --out evidence/model_eval_mock_v2.json

PYTHONPATH=$PWD/python python3 tools/benchmark_cpu_models.py \
  --repeats 3 --out evidence/cpu_benchmark_v2.json
```

`tools/compile_models.py` emits static RKPlans, the real-TileLang exact-ReLU
probe and the versioned `*.rkhybrid.json` execution manifests.  The numeric spec
is calibrated only from observed mock/board layer traces; see
`tools/calibrate_numeric_acceptance.py` and `models/numeric_acceptance.json`.

Final model campaign:

```bash
./scripts/run-model-admission.sh
```

Current evidence: `results/20261024T071144Z-models/` (core 1 full 10,000-image
test split for MLP and Transformer; masks 2/4 fixed 512-image subset; independent
verifier PASS).  The report keeps raw logits and deterministic layer traces.
`results/20261024T031249Z-models/PERFORMANCE-RETRACTION.md` marks the older
performance fields invalid.

## Board admission

```bash
./scripts/run-board-admission.sh
```

The script reads the ignored repository `.env`, uses the askpass path, uploads an
isolated source/campaign bundle, runs the three single-core masks serially,
retrieves raw outputs, verifies source/library records and fails on any exact
credential value in the result tree.

Current evidence: `results/20261023T144054Z-rk3588/` (27 cases × 3 masks = 81/81,
chain/separate, tiled RKIR and real multi-block TileLang staged
projection/FFN/branch, CLI path).  This campaign was generated with the independent
source-built TileLang/TVM tree.  Runtime ownership/failure semantics are in
`RUNTIME-LIFECYCLE.md`.

## Independent complete-model verifier

```bash
PYTHONPATH=$PWD/python python3 tools/verify_model_results.py \
  --manifest results/20261024T071144Z-models/campaign/manifest.json \
  --results results/20261024T071144Z-models/model-results
```

The verifier does not trust reported accuracy/reference arrays.  It reloads raw
logits, labels, weights, plans and the dataset; recomputes CPU FP32 and declared
paths; checks all profiles, finite values, sample/layer sets and hashes; applies
the frozen per-layer and per-profile limits; and rejects a same-top-1 logits
amplification negative control.

## Offline reference/raw-output recomputation

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
python3 tools/recompute_board_metrics.py \
  --campaign results/20261023T144054Z-rk3588/campaign \
  --results results/20261023T144054Z-rk3588/board \
  --out results/20261023T144054Z-rk3588/recomputed
```

The recomputation uses archived inputs/raw outputs and unquantized float64 math
references.

## CLI host preflight (mock, caller-owned fd)

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
RKNPU_CHIP=rk3588 \
python3 tools/board_cli_check.py \
  --campaign results/20261023T144054Z-rk3588/campaign \
  --out /tmp/rk-cli-check --fd 0
```

On the board the same helper is run without `--fd`; the CLI opens and closes its
own fd.

## P0 source build (completed)

Earlier short-timeout attempts remain recorded as history:
`provenance/p0-rebuild-attempt.log` (`git_clone_rc=124`) and
`provenance/p0-rebuild-attempt-2.log` (`curl_rc=28`).  The successful build used
resumable mirror downloads, exact locked archives, a separate CMake tree and
source-built `libtilelang`/`libtvm_*` libraries.  Full source list, archive and
library hashes, toolchain, build flags, logs and source-tree inventory are in
`provenance/p0-source-build.json` and `provenance/p0-source-build/README.md`.

Source-built host validation:

```bash
./provenance/p0-source-build/create-source-env.sh
PYTHONPATH=$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env:$PWD/python \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
TILELANG_CACHE_DIR=$PWD/.src/p0-tilelang-cache \
RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan.so \
RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
python3 -m pytest tests
```

Source-built board campaign:

```bash
TILELANG_DEPS_DIR=$PWD/.src/p0-source-env \
TILELANG_SOURCE_PYTHONPATH=$PWD/.src/p0-build-src/tilelang \
TILELANG_SOURCE_LIBDIR=$PWD/.src/p0-build-tilelang/lib \
TILELANG_CACHE_DIR=$PWD/.src/p0-tilelang-cache \
./scripts/run-board-admission.sh
```

## Upstream provider patch

`rknpu-submit` is pinned at `6b144b19bb6700f96fc1638ccc7f20ad8719d527`.
The context-quarantine patch and its SHA-256 are recorded in
`provenance/upstream-patches.json`; the patch itself is
`provenance/patches/rknpu-submit-context-fault.patch`.  Reconstruct with:

```bash
git -C ../upstream-components/rknpu-submit checkout 6b144b19bb6700f96fc1638ccc7f20ad8719d527
git -C ../upstream-components/rknpu-submit apply \
  "$PWD/provenance/patches/rknpu-submit-context-fault.patch"
```

The provider C test `provider_context` covers queued shared-fd submission,
task-BO non-rewrite, faulted-close retention and healthy-close compatibility.
