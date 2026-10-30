# Replay commands

All commands run from `research/tilelang-rknpu` unless stated otherwise.

## Host verification

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh --keep-build
```

Current evidence is recorded after the final rerun in
`results/20261024T092339Z-host/` (235 pytest passed, 27/27 ctest, 26 contract + 711 host + 2,400,033 graph
C checks, sanitizer explicitly enabled; see its `manifest.json` and `summary.md`).

## Complete-model build and evaluation

```bash
cd research/tilelang-rknpu

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
```

`tools/compile_models.py` emits static RKPlans, the real-TileLang exact-ReLU
probe, five MLP hidden-layer diagnostics and the versioned `*.rkhybrid.json`
execution manifests.  `models/numeric_acceptance_train.json` is frozen only from
the fixed MNIST train observation; the old test-calibrated envelope remains only
for historical regression.

Final model campaign:

```bash
./scripts/run-model-admission.sh
```

Final optimisation delivery re-acceptance (full 10,000-image test split for all
five static profiles, baseline `42f870f` vs current default, three interleaved
rounds, plus fresh packed numeric observation):

```bash
RK_FINAL_LIMIT=10000 RK_FINAL_ROUNDS=3 ./scripts/run-final-delivery.sh
```

Current evidence: `results/20261026T120625Z-final/` (`analysis.json`,
`analysis.md`, `verify-packed-summary.json`, raw per-round logits/traces).
Summary and explicit non-claims: [OPTIMIZATION-SUMMARY.md](OPTIMIZATION-SUMMARY.md).

Vendor RKNN baseline build and board replay instructions are in
[vendor/README.md](vendor/README.md).  Recorded result:
`results/20261027T015302Z-vendor/` (`analysis.md`, `analysis.json`).

Current evidence: `results/20261024T071144Z-models/` (core 1 full 10,000-image
test split for MLP and Transformer; masks 2/4 fixed 512-image subset; independent
verifier PASS).  The report keeps raw logits and deterministic layer traces.
`results/20261024T031249Z-models/PERFORMANCE-RETRACTION.md` marks the older
performance fields invalid.

## Train-calibrated numeric acceptance

```bash
./scripts/run-numeric-acceptance.sh
```

The script uploads images only (no labels), runs the fixed 32-image train
calibration observation on RK3588, freezes tolerances on the Host, records the
frozen spec SHA-256, then runs the disjoint fixed 32-image test observation on the
same board.  It then invokes `tools/verify_numeric_acceptance.py`, which checks
sample disjointness, all 33 layer gates (including the real MLP hidden
linear+bias/ReLU traces), five profile logit gates and the existing full-test raw
logits under those train-frozen limits.

Current evidence: `results/20261024T091634Z-numeric/`; `verify-summary.json` is
PASS.  The old test-calibrated envelope under `evidence/numeric-calibration/` is
retained only as regression evidence.

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

## Independent numeric verifier

```bash
PYTHONPATH=$PWD/python:$PWD \
python3 tools/verify_numeric_acceptance.py \
  --spec models/numeric_acceptance_train.json \
  --calibration-observation results/20261024T091634Z-numeric/calibration/observation.npz \
  --acceptance-observation results/20261024T091634Z-numeric/acceptance/observation.npz \
  --campaign-manifest results/20261024T071144Z-models/campaign/manifest.json \
  --campaign-results results/20261024T071144Z-models/model-results
```

This verifier reloads raw train/test observations, expected references, weights,
plans and the existing full-test campaign raw logits.  It proves train/test sample
disjointness, recomputes every limit from train observations only, checks all
layer/profile finite values and applies the frozen limits.  It rejects both a
same-top-1 hidden-layer tamper and a same-top-1 logits amplification.

The older `tools/verify_model_results.py` command remains a historical
task/timing and regression-envelope record.  Its PASS is bound to the prior source
snapshot, and its test-calibrated limits are not independent numerical evidence.

```bash
PYTHONPATH=$PWD/python python3 tools/verify_model_results.py \
  --manifest results/20261024T071144Z-models/campaign/manifest.json \
  --results results/20261024T071144Z-models/model-results
```

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

## SmolLM2-135M-Instruct

> The commands in this section that name `models/smollm2_numeric_acceptance.json`
> are historical V2 replay. The archived V2 spec is
> `models/specs/smollm2_v2_20261028T094650Z.json`; do not treat the mutable
> `models/smollm2_numeric_acceptance.json` path as current. The current v5
> complete-acceptance replay is below.

One model driver handles fixed-revision fetch, Host compile, isolated board
setup, chat generation and generic board requests.  Credentials continue to come
from the repository's ignored `.env`; the script never prints them.

```bash
cd research/tilelang-rknpu

# Host-only tokenizer/reference dependencies (torch/numpy are in requirements.lock).
python3 -m pip install --target .pydeps-p0 -r requirements-smollm2.lock

# Fixed model files and SHA-256 verification (HF mirror by default).
./scripts/run-smollm2.sh fetch

# Canonical FP16 weights + 268 real-TileLang RKPlans.
./scripts/run-smollm2.sh compile

# Host regression: full pytest suite and existing C tests.
./scripts/run-smollm2.sh host-check

# One-time isolated board source bundle, production C build and package upload.
./scripts/run-smollm2.sh setup-board

# Chat-template driven greedy generation on the RK3588.
./scripts/run-smollm2.sh chat \
  "Name the capital of France." 32
```

The generic board entry accepts a JSON containing `runs` with
`generate`, `prefill`, `capture`, `teacher_forcing`, `incremental`, `score` or
`expect_reject` modes:

```bash
./scripts/run-smollm2.sh board-run /tmp/smollm-request.json \
  results/$(date -u +%Y%m%dT%H%M%SZ)-smollm2
```

The committed evidence manifest is
`results/20261028T094650Z-smollm2/manifest.json`.  Raw 25 MB
`all-logits`/teacher-forcing arrays and the official reference arrays are kept
under the ignored local path `.data/evidence/smollm2-20261028T094650Z/`; their
hashes and replay identities are in that manifest.

Independent references and frozen acceptance:

```bash
export PYTHONPATH=$PWD/python:$PWD/tools:$PWD/.pydeps-p0

# Fixed text windows for NLL/PPL (calibration and disjoint acceptance sources).
python3 tools/make_smollm2_ppl_windows.py \
  --source .data/models/SmolLM2-135M-Instruct/README.md \
  --out /tmp/smollm-ppl-calibration.json
python3 tools/make_smollm2_ppl_windows.py \
  --source ACCEPTANCE.md --out /tmp/smollm-ppl-acceptance.json

# Official FP32 semantic/quality reference.
python3 tools/smollm2_official.py --mode logits \
  --ids-json /tmp/calibration-ids.json \
  --out /tmp/official-calibration
python3 tools/smollm2_official.py --mode score \
  --windows-json /tmp/smollm-ppl-calibration.json \
  --out /tmp/official-ppl-calibration

# Independent declared-reference implementation (FP16, 256-wide K writeback).
python3 tools/smollm2_reference.py \
  --package .data/models/smollm2_135m_instruct_package \
  --ids-json /tmp/calibration-ids.json \
  --capture-layers 0,15,29 --out /tmp/ref-calibration

# Re-freeze only from calibration observations, then verify disjoint acceptance.
python3 tools/calibrate_smollm2.py \
  --board .data/evidence/smollm2-20261028T094650Z/calibration \
  --reference .data/evidence/smollm2-20261028T094650Z/calibration-reference \
  --reference-tf .data/evidence/smollm2-20261028T094650Z/calibration-tf-reference \
  --official-logits .data/evidence/smollm2-20261028T094650Z/official/official-logits-calibration/logits.npy \
  --ppl-board .data/evidence/smollm2-20261028T094650Z/ppl/calibration-board \
  --ppl-official .data/evidence/smollm2-20261028T094650Z/official/official-ppl-calibration/result.json \
  --ppl-source-json .data/evidence/smollm2-20261028T094650Z/official/ppl-calibration-windows.json \
  --prompt-text "Name the capital of France." \
  --out models/smollm2_numeric_acceptance.json

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

### Current v5 acceptance closure and executable replay (2026-10-29)

The current implementation's complete acceptance is
`models/smollm2_numeric_acceptance_v5.json` with canonical-content hash
`1a8c68d2da51eb9bc6885a4d540950f74011e8bcbf150a5d975d75a7575f368b` and file SHA-256
`04c4011c7eb7daec4f4f607d65d234f8ec94ddba112a931fc2fd24056f70e38f`.
The frozen inputs are `models/smollm2_v5_inputs.json`; the six-prompt
calibration bundle is `models/specs/smollm2_v4_calibration_bundle.json` and the
sealed acceptance bundle is `models/specs/smollm2_v5_acceptance_bundle.json`.
The V5 inputs, requests, gates and historical failures are immutable in this
delivery.

All replay paths now use one executable entry, with fail-fast stage ordering:

```bash
cd research/tilelang-rknpu

# 1. Re-verify the sealed V5 observation without touching the board.
#    The expected file/canonical spec hashes come from the independent record
#    models/specs/smollm2_spec_hashes.json, not from hashing the spec on the
#    spot and then treating that result as an external identity.
./scripts/replay-smollm2-v5.sh reuse \
  --raw .data/evidence/smollm2-20261029T120416Z \
  --results results/20261029T120416Z-smollm2-v5

# 2. Re-run only the acceptance half against the frozen V5 spec/inputs/requests.
#    Calibration is reused by design; this is for implementation or timing
#    changes, not for establishing a new policy.  The script runs the board
#    acceptance and PPL, rebuilds declared and official references, seals the
#    acceptance tree and invokes the fail-closed verifier.
./scripts/replay-smollm2-v5.sh rerun-fixed-acceptance \
  --evidence-root /tmp/v5-fixed-rerun/evidence \
  --results /tmp/v5-fixed-rerun/results \
  --calibration-root .data/evidence/smollm2-20261029T120416Z/calibration \
  --calibration-bundle models/specs/smollm2_v4_calibration_bundle.json

# 3. Full new campaign from an input plan.  The order is enforced:
#      prepare/freeze inputs -> official preliminary continuations ->
#      calibration board/reference/PPL -> seal calibration -> freeze spec ->
#      acceptance board/reference/PPL -> seal acceptance -> verify.
#    Each subprocess is checked; any failure exits non-zero immediately.
./scripts/replay-smollm2-v5.sh campaign \
  --evidence-root /tmp/v5-new/evidence \
  --results /tmp/v5-new/results \
  --input-plan models/smollm2_v5_input_plan.json
```

`reuse` and `rerun-fixed-acceptance` keep V5 calibration/observation identities
fixed.  `campaign` is the only path that writes a new spec; it never overwrites
the V5 result tree.  The virtual environment/package prerequisites are the same
as `./scripts/run-smollm2.sh`; add `--fetch` and/or `--compile` to `campaign`
when those artefacts are not present.

The underlying `verify_smollm2.py` invocation is still available for audit and
unit tests, but is not the replay entry: the replay entry supplies the expected
spec file hash from the independent record, constructs every path, seals the
observation with that same frozen spec, and turns any non-zero verifier exit
into a non-zero replay exit.

Instruction-prompt divergence diagnosis is a separate, finite diagnostic and
does not alter the V5 gates:

```bash
# Select one same-prefix position per prompt and write a board probe request.
python3 tools/diagnose_smollm2_divergence.py prepare \
  --board-run .data/evidence/smollm2-20261029T120416Z/acceptance/board/run.json \
  --official-root .data/evidence/smollm2-20261029T120416Z/acceptance/official-generations \
  --inputs models/smollm2_v5_inputs.json \
  --out-request /tmp/v5-divergence-request.json \
  --out-manifest /tmp/v5-divergence-manifest.json
./scripts/run-smollm2.sh board-run /tmp/v5-divergence-request.json \
  /tmp/v5-divergence-board

# Recompute backend, declared-reference and official same-prefix top candidates,
# including the corrected all-ones official attention-mask path.
python3 tools/diagnose_smollm2_divergence.py analyze \
  --probe-board /tmp/v5-divergence-board \
  --manifest /tmp/v5-divergence-manifest.json \
  --package .data/models/smollm2_135m_instruct_package \
  --model-dir .data/models/SmolLM2-135M-Instruct \
  --out /tmp/v5-divergence
```

The checked-in delivery result is under
`results/20261030T013910Z-smollm2-v5-delivery/divergence/`; its table separates
the archived official generation from the corrected explicit-mask generation
and reports top1/top2 margins rather than calling every difference a near-tie.
The delivery overview, latency decomposition, A/B and exact identities are in
`results/20261030T013910Z-smollm2-v5-delivery/DELIVERY-REPORT.md` and
`DELIVERY-MANIFEST.json`; large board/reference arrays remain local/ignored.

## TileLang M/N schedule experiment (2026-10-30)

The supported schedule contract is `rknpu.schedule=mn_tiles_v1` plus
`rknpu.tile_m`, `rknpu.tile_n` and `rknpu.mn_order`; K remains fixed at 256-wide
ordered FP16 writeback.  See [TILELANG-SCHEDULE.md](TILELANG-SCHEDULE.md).

```bash
python3 tools/run_schedule_experiment.py compile   --source tiled --out /tmp/schedule-small --m 512 --k 512 --n 384   --schedule base_256x128,256,128,m_major   --schedule tile_128x128,128,128,m_major
./scripts/run-smollm2.sh schedule-bench /tmp/schedule-small /tmp/schedule-small-board.json

# Same schedule information can be supplied directly through RKIR op attrs
# (source_schedule) and is exercised by tests/test_schedule_contract.py.
```

The shipped experiment includes SmolLM-shaped prefill QKV, attention QK prefill
and attention QK decode plans.  Explicit schedules on multi-node dataflow are
rejected until that lowerer is schedule-aware.

The finite board tuner and new-process replay are:

```bash
python3 tools/tune_schedule.py prepare   --capture .data/evidence/smollm2-20261029T120416Z/acceptance/board/run-00/captured-00.npz   --out /tmp/schedule-tuning
./scripts/run-smollm2.sh schedule-tune /tmp/schedule-tuning/plans   /tmp/schedule-tuning/board

# Optional whole-model plan override (does not change the accepted package
# manifest/weights identity); the runner validates the policy/plan hash.
RK_SMOLLM_SCHEDULE_POLICY=/tmp/schedule-tuning/model-override.json   ./scripts/run-smollm2.sh board-run   models/smollm2_v5_requests/acceptance.json /tmp/scheduled-acceptance
```

The delivered tuning/cache/replay and V5 model-override results are under
`results/20261030T030803Z-tilelang-schedule/`.

## Layout reuse / A-B campaign (2026-10-28)

Final report: [LAYOUT-REUSE-REPORT.md](LAYOUT-REUSE-REPORT.md).

The benchmark request uses a fixed 42-token prompt and three 16-token greedy
generations.  The baseline package keeps the pre-P0 plans and runs with
`--no-share-constants --no-persistent-kv`; the current package runs with both
features enabled.  The driver accepts either remote package directory.

```bash
# one-time isolated board setup; uploads current and baseline packages
RK_SMOLLM_BASELINE_PACKAGE=$PWD/.data/models/smollm2_baseline_package \
  ./scripts/run-smollm2.sh setup-board

# baseline
RK_SMOLLM_REMOTE_PACKAGE=package-baseline \
RK_SMOLLM_RUNNER_ARGS='--no-share-constants --no-persistent-kv' \
  ./scripts/run-smollm2.sh board-run \
  results/baseline-layout-reuse/request-benchmark.json \
  results/baseline-layout-reuse/board-final4-baseline-r1-benchmark

# current
RK_SMOLLM_REMOTE_PACKAGE=package \
  ./scripts/run-smollm2.sh board-run \
  results/baseline-layout-reuse/request-benchmark.json \
  results/baseline-layout-reuse/board-final4-current-r1-benchmark

# frozen teacher-forcing / capture / incremental acceptance
RK_SMOLLM_REMOTE_PACKAGE=package \
  ./scripts/run-smollm2.sh board-run \
  results/baseline-layout-reuse/request-acceptance.json \
  results/baseline-layout-reuse/board-final4-current-acceptance

PYTHONPATH=$PWD/python:$PWD python3 tools/verify_smollm2.py \
  --spec models/smollm2_numeric_acceptance.json \
  --board results/baseline-layout-reuse/board-final4-current-acceptance \
  --reference .data/evidence/smollm2-20261028T094650Z/acceptance-reference \
  --reference-tf .data/evidence/smollm2-20261028T094650Z/acceptance-tf-reference \
  --official-logits .data/evidence/smollm2-20261028T094650Z/official/official-logits-acceptance/logits.npy \
  --acceptance-prompt "$(cat .data/evidence/smollm2-20261028T094650Z/official/prompt-acceptance.txt)" \
  --acceptance-source-json .data/evidence/smollm2-20261028T094650Z/official/prompt-0-ids.json \
  --out results/baseline-layout-reuse/verify-final4-current-acceptance.json
```

Host contract tests:

```bash
PYTHONPATH=$PWD/python:$PWD RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
  python3 -m pytest -q tests/test_shared_constants.py
PYTHONPATH=$PWD/python:$PWD RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
  RKNPU_SMOLLM_SLOW_TESTS=1 python3 -m pytest -q \
  tests/test_smollm2_host.py::SmolLM2HostActionTests::test_persistent_kv_reset_replays_identical
```

## Graph fusion / device residency campaign (2026-10-28)

Final report: [GRAPH-FUSION-REPORT.md](GRAPH-FUSION-REPORT.md).

Regenerate the two isolated packages:

```bash
# default-policy graph-fusion package (new plans do not replace the frozen
# models/compiled baseline)
python3 tools/compile_models.py --out models/compiled_graph_fusion

# independently named approximate RMSNorm package/profile
python3 tools/compile_models.py \
  --out models/compiled_rmsnorm_lut \
  --profile capabilities/rk3588_vendor_co_v1_rmsnorm_lut.json \
  --include-rmsnorm-approx
```

Paired RK3588 evaluation (the driver builds the current runtime and uploads the
selected package):

```bash
RK_GF_PLANS_NAME=compiled_graph_fusion ./scripts/run-graph-fusion-eval.sh setup
RK_GF_LIMIT=512 RK_GF_ROUNDS=5 \
RK_GF_ARMS=':default:_deviceffn_fused' \
  ./scripts/run-graph-fusion-eval.sh run \
  results/20261028T113459Z-graph-fusion/board-gf-pair-512

# same-policy residual-Add and old device-FFN controls
RK_GF_LIMIT=512 RK_GF_ROUNDS=3 \
RK_GF_ARMS=':default:_deviceffn:_deviceffn_fused:_addfused' \
  ./scripts/run-graph-fusion-eval.sh run \
  results/20261028T113459Z-graph-fusion/board-gf-final-512

# full 10k progress for the explicit fused-FFN policy
RK_GF_PLANS_NAME=compiled_rmsnorm_lut ./scripts/run-graph-fusion-eval.sh setup
RK_GF_LIMIT=0 RK_GF_ROUNDS=1 RK_GF_ARMS=':_deviceffn_fused' \
  ./scripts/run-graph-fusion-eval.sh run \
  results/20261028T113459Z-graph-fusion/board-gf-full2
```

Host fusion/operator tests:

```bash
PYTHONPATH=$PWD/python:$PWD RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so \
  python3 -m pytest -q tests/test_graph_fusion.py
```

## Stage 2: tightened admission, RMSNorm acceptance, RoPE, PV→O, AB/BA

```bash
# AB/BA scheduling attribution (old device FFN vs fused; same numeric path)
for spec in "ab:5::_deviceffn:_deviceffn_fused" \
            "ba:5::_deviceffn_fused:_deviceffn" \
            "dab:5::default:_deviceffn_fused" \
            "dba:5::_deviceffn_fused:default" \
            "nab:5::default:_deviceffn" \
            "nba:5::_deviceffn:default"; do
  IFS=: read -r name rounds _ arms <<<"$spec"
  RK_GF_LIMIT=512 RK_GF_ROUNDS=$rounds RK_GF_ARMS="$arms" \
    ./scripts/run-graph-fusion-eval.sh run \
    results/20261028T113459Z-graph-fusion/abba-$name
done

# RMSNorm operator-level independent acceptance
RK_GF_PLANS_NAME=compiled_rmsnorm_lut ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh rmsnorm-probe \
  results/20261028T113459Z-graph-fusion/rmsnorm-operator-board-domain.json

# RoPE minimal device reorder probe (1 submit, 4 tasks)
RK_GF_PLANS_NAME=compiled_graph_fusion ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh rope-probe \
  results/20261028T113459Z-graph-fusion/rope-device-interleaved-board.json \
  interleaved 64
./scripts/run-graph-fusion-eval.sh rope-probe \
  results/20261028T113459Z-graph-fusion/rope-device-split32-board.json \
  split_half 32

# PV→O real device producer→consumer probe
./scripts/run-graph-fusion-eval.sh pv-o-probe \
  results/20261028T113459Z-graph-fusion/pv-o-device-board.json
```

Negative chain/QKV→QK tests:

```bash
python3 -m pytest -q tests/test_graph_fusion.py
```

## Stage 3: corrected evidence and layout-boundary work

```bash
# RMSNorm: calibration/freeze, then a separate acceptance process
RK_GF_PLANS_NAME=compiled_rmsnorm_lut ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh rmsnorm-calibrate \
  results/20261028T113459Z-graph-fusion/rmsnorm-calibrate-frozen.json
# copy .spec.json/.spec.sha256 to models/rmsnorm_approx_acceptance.json(.sha256)
./scripts/run-graph-fusion-eval.sh rmsnorm-verify \
  results/20261028T113459Z-graph-fusion/rmsnorm-verify-frozen.json \
  models/rmsnorm_approx_acceptance.json

# QKV->QK address-level constraint proof and negative lowering tests
python3 tools/analyze_cube_weight_mapping.py
python3 -m pytest -q tests/test_graph_fusion.py

# Actual Smol shape device RoPE vs production Host action; PV->O b8 grouping
RK_GF_PLANS_NAME=compiled_graph_fusion ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh smollm-rope-probe \
  results/20261028T113459Z-graph-fusion/smollm-rope-device-board.json
./scripts/run-graph-fusion-eval.sh smollm-rope-residency \
  results/20261028T113459Z-graph-fusion/smollm-rope-residency-board.json
./scripts/run-graph-fusion-eval.sh pv-o-b8-probe \
  results/20261028T113459Z-graph-fusion/pv-o-b8-device-board.json

# Opt-in Smol 30-layer device-RoPE integration; acceptance prompt with policy
./scripts/run-smollm2.sh setup-board
./scripts/run-smollm2.sh device-rope-probe \
  .data/evidence/smollm2-20261028T094650Z/official/prompt-1-ids.json \
  results/20261028T113459Z-graph-fusion/smollm-device-rope-policy-verify.json \
  models/smollm2_device_rope_policy.json
```

## Stage 4: continuous decode and position-indexed RoPE

```bash
# Actual Smol-shape decode table gather: resident CosW/SinW/P, one-hot position only
RK_GF_PLANS_NAME=compiled_graph_fusion ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh smollm-rope-indexed \
  results/20261028T113459Z-graph-fusion/smollm-rope-indexed-board.json

# Continuous 32-step opt-in device-RoPE integration
./scripts/run-smollm2.sh setup-board
./scripts/run-smollm2.sh device-rope-probe \
  .data/evidence/smollm2-20261028T094650Z/official/prompt-0-ids.json \
  results/20261028T113459Z-graph-fusion/smollm-device-rope-cont-p0.json
./scripts/run-smollm2.sh device-rope-probe \
  .data/evidence/smollm2-20261028T094650Z/official/prompt-1-ids.json \
  results/20261028T113459Z-graph-fusion/smollm-device-rope-cont-p1-contpolicy.json \
  models/smollm2_device_rope_policy_cont32.json
```

Continuous decode with position-indexed device gather:

```bash
RK_SMOLLM_DECODE_INDEXED=1 ./scripts/run-smollm2.sh device-rope-probe \
  .data/evidence/smollm2-20261028T094650Z/official/prompt-1-ids.json \
  results/20261028T113459Z-graph-fusion/smollm-device-rope-indexed-cont-p1-policy.json \
  models/smollm2_device_rope_policy_cont32_indexed.json
```

QKV→QK candidate-A device probe (int16 TP writer; expected to be falsified by
the current command/geometry):

```bash
RK_GF_PLANS_NAME=compiled_graph_fusion ./scripts/run-graph-fusion-eval.sh setup
./scripts/run-graph-fusion-eval.sh int16-probe \
  results/20261028T113459Z-graph-fusion/int16-tp-layout-board.json
```
