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

Current evidence: `results/20261023T131610Z-rk3588/` (27 cases × 3 masks = 81/81,
chain/separate, tiled RKIR and real multi-block TileLang staged
projection/FFN/branch, CLI path).  Runtime ownership/failure semantics are in
`RUNTIME-LIFECYCLE.md`.

## Offline reference/raw-output recomputation

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
python3 tools/recompute_board_metrics.py \
  --campaign results/20261023T131610Z-rk3588/campaign \
  --results results/20261023T131610Z-rk3588/board \
  --out results/20261023T131610Z-rk3588/recomputed
```

The recomputation uses archived inputs/raw outputs and unquantized float64 math
references.

## CLI host preflight (mock, caller-owned fd)

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
RKNPU_CHIP=rk3588 \
python3 tools/board_cli_check.py \
  --campaign results/20261023T131610Z-rk3588/campaign \
  --out /tmp/rk-cli-check --fd 0
```

On the board the same helper is run without `--fd`; the CLI opens and closes its
own fd.

## P0 bounded source retry

The 2026-10-23 attempts and their timeout exit codes are recorded in
`provenance/p0-rebuild-attempt.log` (`git_clone_rc=124`) and
`provenance/p0-rebuild-attempt-2.log` (`curl_rc=28`).  Partial downloads are not
used as source evidence; P0 remains open.

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
