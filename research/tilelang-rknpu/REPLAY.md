# Replay commands

All commands run from `research/tilelang-rknpu` unless stated otherwise.

## Host verification

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh --keep-build
```

Current evidence: `results/20261023T131916Z-host/` (213 pytest, 27/27 ctest,
711 host C checks, sanitizer enabled).

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
