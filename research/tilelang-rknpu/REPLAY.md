# Replay commands

All commands run from `research/tilelang-rknpu` unless stated otherwise.
This file is the only place that keeps commands. What each result is allowed to prove,
and the current status of each path, are in [VALIDATION.md](VALIDATION.md) and
[ARCHITECTURE.md](ARCHITECTURE.md); the supported domain is in [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md).

Historical campaign logs live under `results/<UTC timestamp>-<campaign>/` and are not re-listed here;
see [results/README.md](results/README.md) for how to find them. No measurement below is a current
performance or acceptance claim.

## Environment

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
```

TileLang must be imported before the vendored TVM; use the pinned `.pydeps` build
(`import tilelang` then `from tvm import ...`). Do not mix `.pydeps`, `.pydeps-p0` and `.src` versions
in one interpreter.

## Host verification

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh --keep-build
```

## Backend compile / validate / explain / run

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m rknpu_backend.cli compile-tilelang --help
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m rknpu_backend.cli compile-rkir --help
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m rknpu_backend.cli validate --help
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m rknpu_backend.cli explain --help
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m rknpu_backend.cli run --help
```

`run` is the only entry that opens a device; it accepts `--fd` to use a caller-owned descriptor.
See the CLI contract in [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md).

## Formal pass chain (opt-in)

Direct entry:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
import tilelang
from rknpu_backend import CompileOptions, default_profile
from rknpu_backend.frontend import rknpu_pass as R
from tests.test_rknpu_pass_pipeline import gate_up

plan, stages, facts, tasks = R.compile_guarded_tail(
    gate_up, profile=default_profile(),
    options=CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False))
print(plan.plan_hash)
print([stage.name for stage in stages])
PY
```

Registered TileLang backend (opt-in; nothing is registered by import):

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
import tilelang, tvm
from rknpu_backend.frontend import tilelang_backend as B
from tests.test_tilelang_backend_registration import gemm_small

with B.registered():
    artifact = tilelang.engine.lower(tvm.IRModule({"main": gemm_small}),
                                     target=B.rknpu_target())
    plan = B.plan_from_artifact(artifact)
    print(plan.plan_hash, len(artifact.kernel_source))
PY
```

Targeted tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_rknpu_pass_pipeline.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_tilelang_backend_registration.py -q -o addopts=""
```

Plan-only (no device) numeric check with the C mock:

```bash
PYTHONPATH=$PWD:$PWD/python RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
  .venv/bin/python -m pytest tests/test_rknpu_pass_pipeline.py -q -o addopts="" -k MockExecutionTests
```

## Complete-model build and evaluation

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
python3 tools/compile_models.py

PYTHONPATH=$PWD/python python3 tools/build_numeric_train_acceptance.py

PYTHONPATH=$PWD/python RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
python3 tools/evaluate_models.py --backend mock --models both \
  --global-limit 10000 --profile-samples 512 \
  --out evidence/model_eval_mock_v2.json

PYTHONPATH=$PWD/python python3 tools/benchmark_cpu_models.py \
  --repeats 3 --out evidence/cpu_benchmark_v2.json

./scripts/run-model-admission.sh
RK_FINAL_LIMIT=10000 RK_FINAL_ROUNDS=3 ./scripts/run-final-delivery.sh
```

## Train-calibrated numeric acceptance

```bash
./scripts/run-numeric-acceptance.sh
```

Policy and gates: [models/NUMERIC-ACCEPTANCE.md](models/NUMERIC-ACCEPTANCE.md).

## Board admission

```bash
./scripts/run-board-admission.sh
```

Board commands require SSH credentials in the environment (`ORANGE_PI_HOST`/`ORANGE_PI_USER`/password)
and a built host plan library; without them, no device run is possible.

## Independent numeric verifier

```bash
PYTHONPATH=$PWD/python:$PWD \
python3 tools/verify_numeric_acceptance.py \
  --spec models/numeric_acceptance_train.json \
  --calibration-observation results/20261024T091634Z-numeric/calibration/observation.npz \
  --acceptance-observation results/20261024T091634Z-numeric/acceptance/observation.npz \
  --campaign-manifest results/20261024T071144Z-models/campaign/manifest.json \
  --campaign-results results/20261024T071144Z-models/model-results

PYTHONPATH=$PWD/python python3 tools/verify_model_results.py \
  --manifest results/20261024T071144Z-models/campaign/manifest.json \
  --results results/20261024T071144Z-models/model-results
```

## Offline reference / raw-output recomputation

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
python3 tools/recompute_board_metrics.py \
  --campaign results/20261023T144054Z-rk3588/campaign \
  --results results/20261023T144054Z-rk3588/board \
  --out results/20261023T144054Z-rk3588/recomputed
```

## CLI host preflight (mock, caller-owned fd)

```bash
PYTHONPATH=$PWD/python:$PWD/.pydeps-p0 \
RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
RKNPU_CHIP=rk3588 \
python3 tools/board_cli_check.py \
  --campaign results/20261023T144054Z-rk3588/campaign \
  --out /tmp/rk-cli-check --fd 0
```

## P0 source build

```bash
./provenance/p0-source-build/create-source-env.sh

PYTHONPATH=$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env:$PWD/python \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
TILELANG_CACHE_DIR=$PWD/.src/p0-tilelang-cache \
RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan.so \
RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
python3 -m pytest tests
```

P0 build records are kept in `provenance/p0-source-build/`; they are a source/ABI record,
not a current verification.

## Upstream provider patch

```bash
git -C ../upstream-components/rknpu-submit checkout 6b144b19bb6700f96fc1638ccc7f20ad8719d527
git -C ../upstream-components/rknpu-submit apply \
  "$PWD/provenance/patches/rknpu-submit-context-fault.patch"
```

## SmolLM2-135M-Instruct package

```bash
python3 -m pip install --target .pydeps-p0 -r requirements-smollm2.lock
./scripts/run-smollm2.sh fetch
./scripts/run-smollm2.sh compile
./scripts/run-smollm2.sh host-check
./scripts/run-smollm2.sh setup-board
./scripts/run-smollm2.sh chat "Name the capital of France." 32
./scripts/run-smollm2.sh board-run /tmp/smollm-request.json \
  results/$(date -u +%Y%m%dT%H%M%SZ)-smollm2
```

Fixed acceptance replay (v5 bundles are frozen under `models/`):

```bash
./scripts/replay-smollm2-v5.sh reuse \
  --raw .data/evidence/smollm2-20261029T120416Z \
  --results results/20261029T120416Z-smollm2-v5

./scripts/replay-smollm2-v5.sh rerun-fixed-acceptance \
  --evidence-root /tmp/v5-fixed-rerun/evidence \
  --results /tmp/v5-fixed-rerun/results \
  --calibration-root .data/evidence/smollm2-20261029T120416Z/calibration \
  --calibration-bundle models/specs/smollm2_v4_calibration_bundle.json
```

Verifier and reference:

```bash
PYTHONPATH=$PWD/python:$PWD/tools:$PWD/.pydeps-p0 \
python3 tools/smollm2_reference.py \
  --package .data/models/smollm2_135m_instruct_package \
  --ids-json /tmp/calibration-ids.json \
  --capture-layers 0,15,29 --out /tmp/ref-calibration

PYTHONPATH=$PWD/python:$PWD/tools:$PWD/.pydeps-p0 \
python3 tools/verify_smollm2.py \
  --spec models/smollm2_numeric_acceptance.json \
  --board .data/evidence/smollm2-20261028T094650Z/acceptance \
  --reference .data/evidence/smollm2-20261028T094650Z/acceptance-reference \
  --reference-tf .data/evidence/smollm2-20261028T094650Z/acceptance-tf-reference \
  --official-logits .data/evidence/smollm2-20261028T094650Z/official/official-logits-acceptance/logits.npy \
  --ppl-board .data/evidence/smollm2-20261028T094650Z/ppl/acceptance-board \
  --ppl-official .data/evidence/smollm2-20261028T094650Z/official/official-ppl-acceptance/result.json \
  --acceptance-source-json .data/evidence/smollm2-20261028T094650Z/official/ppl-acceptance-windows.json \
  --out /tmp/verify-smollm2.json
```

## Graph-fusion and layout campaigns

The graph-fusion and layout-reuse campaigns have their own commands in the campaign directories
(`results/20261028T071025Z-epilogue-fusion/`, `results/20261030T013910Z-smollm2-v5-delivery/`, and the
`tools/` entry points they name). Their current status is "historical evidence", not a current claim.
