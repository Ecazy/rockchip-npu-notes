#!/usr/bin/env bash
# Same-payload execution-plan campaign on RK3588:
#   generic Host bindings, prepared residency lowering, Host-fused control,
#   vendor RKNN-Tiny2/Runtime 2.3.2, and NumPy FP32 reference.
# All five static profiles use the full 10,000-image MNIST test split.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || {
  echo 'unsafe remote workspace' >&2; exit 2; }

LIMIT=${RK_CANDIDATE_LIMIT:-10000}
CUSTOM_ROUNDS=${RK_CANDIDATE_ROUNDS:-3}
EXTERNAL_ROUNDS=${RK_CANDIDATE_EXTERNAL_ROUNDS:-2}
CPU=${RK_TIME_CPU:-6}
CORE_MASK=${RKNPU_CORE_MASK:-1}
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$CUSTOM_ROUNDS" =~ ^[1-9][0-9]*$ && \
   "$EXTERNAL_ROUNDS" =~ ^[1-9][0-9]*$ && "$CPU" =~ ^[0-9]+$ ]] || {
  echo 'invalid campaign numeric input' >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-exec-$STAMP"
LOCAL="$HERE/results/$STAMP-execution-candidates"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL"
CAMPAIGN="$LOCAL/campaign"
STAGE=$(mktemp -d /tmp/rk-exec-XXXXXX)
askpass=$(mktemp /tmp/rk-exec-askpass-XXXXXX)
cleanup() { rm -rf -- "$STAGE"; rm -f -- "$askpass"; }
trap cleanup EXIT

# ---- immutable source and payload -----------------------------------------
mkdir -p "$STAGE/source/tilelang-rknpu" \
  "$CAMPAIGN/plans/current" "$CAMPAIGN/plans/candidate" \
  "$CAMPAIGN"/{weights,data,numeric,vendor}
git -C "$REPO" archive HEAD research/custom-ops | \
  tar -x -C "$STAGE/source" --strip-components=1
tar -cf - -C "$HERE" python tools scripts models vendor | \
  tar -xf - -C "$STAGE/source/tilelang-rknpu"
cp -a "$HERE/models/compiled/." "$CAMPAIGN/plans/current/"
cp -a "$CAMPAIGN/plans/current/." "$CAMPAIGN/plans/candidate/"
PYTHONPATH="$HERE/python" python3 "$HERE/tools/build_hybrid_manifests.py" \
  --plans-dir "$CAMPAIGN/plans/candidate" --weights-dir "$HERE/models" \
  --include-hostfused \
  > "$LOCAL/candidate-manifest-build.log" 2>&1
cp "$HERE/models/mnist_mlp.npz" "$HERE/models/tiny_vit.npz" \
   "$CAMPAIGN/weights/"
cp "$HERE/.data/mnist/"*.gz "$CAMPAIGN/data/"
cp "$HERE/models/numeric_acceptance.json" \
   "$HERE/models/numeric_acceptance_packed.json" \
   "$HERE/models/numeric_calibration_train.npz" \
   "$HERE/models/numeric_acceptance_test.npz" \
   "$HERE/models/numeric_golden.npz" "$CAMPAIGN/numeric/"
cp -a "$HERE/vendor/." "$CAMPAIGN/vendor/"

python3 - "$REPO" "$STAGE" "$CAMPAIGN" "$LIMIT" "$CUSTOM_ROUNDS" \
  "$EXTERNAL_ROUNDS" "$CORE_MASK" "$CPU" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, stage, camp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                               text=True).strip()
dirty = bool(subprocess.check_output(
    ['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip())
packed = camp / 'numeric/numeric_acceptance_packed.json'
manifest = {
    'schema_version': 1,
    'kind': 'execution-candidate-campaign',
    'created_utc': camp.parent.name,
    'source': {
        'git_head': head,
        'worktree_dirty_at_stage': dirty,
        'custom_ops_archive': 'git archive HEAD research/custom-ops',
        'tilelang_paths': ['python', 'tools', 'scripts', 'models', 'vendor'],
    },
    'board': {
        'target': 'RK3588 OPi 5 Plus',
        'kernel': '6.1.43-rockchip-rk3588-rknpu098',
        'driver': 'v0.9.8',
        'npu_frequency_hz': 1000000000,
        'npu_core_mask': int(sys.argv[7]),
        'cpu_affinity': int(sys.argv[8]),
        'cpu_thread_env': {'OMP_NUM_THREADS': 1,
                           'OPENBLAS_NUM_THREADS': 1,
                           'MKL_NUM_THREADS': 1},
    },
    'evaluation': {
        'images_per_profile': int(sys.argv[4]),
        'full_mnist_test_split': int(sys.argv[4]) == 10000,
        'profiles': ['mnist_mlp.b32', 'mnist_mlp.b8', 'mnist_mlp.b1',
                     'tiny_vit.b8', 'tiny_vit.b1'],
        'custom_rounds': int(sys.argv[5]),
        'external_rounds': int(sys.argv[6]),
        'interleaved': True,
        'warm_interval': ('raw preparation + model execution + output '
                          'materialisation; argmax excluded'),
        'setup_separate': True,
    },
    'arms': {
        'generic': {'family': 'custom', 'plans': 'plans/current',
                    'execution_mode': 'generic', 'hybrid_suffix': '',
                    'numeric_policy': 'packed per-logit'},
        'prepared': {'family': 'custom', 'plans': 'plans/current',
                     'execution_mode': 'prepared', 'hybrid_suffix': '',
                     'numeric_policy': 'packed per-logit'},
        'auto': {'family': 'custom', 'plans': 'plans/current',
                 'execution_mode': 'auto', 'hybrid_suffix': '',
                 'numeric_policy': 'packed per-logit; static binding-fanout '
                                   'selection between generic and prepared'},
        'hostfused': {'family': 'custom', 'plans': 'plans/candidate',
                      'execution_mode': 'prepared', 'hybrid_suffix': '_hostfused',
                      'numeric_policy': 'packed per-logit; bit-preserving '
                                        'residual+RMSNorm Host fusion'},
        'vendor': {'family': 'external', 'backend': 'rknn-toolkit-lite2',
                   'numeric_policy': 'classification-preserving FP16; not '
                                     'ranked with packed custom arms'},
        'cpu': {'family': 'external', 'backend': 'numpy-fp32',
                'numeric_policy': 'FP32 reference; not ranked with packed '
                                  'custom arms'},
    },
    'thresholds': {
        'mnist_mlp_accuracy': 0.97,
        'tiny_vit_accuracy': 0.95,
        'accuracy_drop_max': 0.005,
        'top1_agreement_min': 0.99,
    },
    'numeric': {
        'packed_spec': {'file': 'numeric/numeric_acceptance_packed.json',
                        'sha256': sha(packed)},
        'evaluator_spec': {
            'file': 'numeric/numeric_acceptance.json',
            'sha256': sha(camp / 'numeric/numeric_acceptance.json')},
    },
    'weights': {p.name: sha(p) for p in sorted((camp / 'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp / 'data').glob('*.gz'))},
    'vendor': {
        'runtime_sha256': sha(camp / 'vendor/runtime/2.3.2/librknnrt.so'),
        'lite2_wheel_sha256': sha(next((camp / 'vendor/wheel').glob('*.whl'))),
        'rknn_manifest_sha256': sha(camp / 'vendor/rknn/rknn-manifest.json'),
        'runtime_version': '2.3.2', 'toolkit_version': '2.3.2',
        'compile_precision': 'non-quantized float16',
        'single_core_mode': True,
        'runtime_core': 0,
    },
    'plans': {
        'current_compiled_manifest_sha256':
            sha(camp / 'plans/current/manifest.json'),
        'candidate_compiled_manifest_sha256':
            sha(camp / 'plans/candidate/manifest.json'),
        'candidate_hybrid_manifest_sha256': {
            p.name: sha(p) for p in sorted(
                (camp / 'plans/candidate').glob('*_hostfused.rkhybrid.json'))},
    },
}
(camp / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n')
(camp / 'SHA256SUMS').write_text('')
PY
(cd "$CAMPAIGN" && find . -type f ! -name SHA256SUMS -print0 | \
   sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$STAGE.source.tgz" -C "$STAGE" source
tar -czf "$STAGE.campaign.tgz" -C "$LOCAL" campaign
sha256sum "$STAGE.source.tgz" "$STAGE.campaign.tgz" > "$LOCAL/bundles.sha256"

# ---- transfer and build ----------------------------------------------------
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
     -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$STAGE.source.tgz" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$STAGE.campaign.tgz" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "cd '$REMOTE' && tar -xzf source.tgz && tar -xzf campaign.tgz && \
  ln -sfn '$REMOTE_BASE/tl-final-20261026T120625Z/source/common/upstream-components' \
    '$REMOTE/source/upstream-components' && \
  { date -u +%FT%TZ; uname -srm; tr -d '\\0' </proc/device-tree/model; echo; \
    cat /sys/kernel/debug/rknpu/version; cat /sys/kernel/debug/rknpu/freq; \
    cat /sys/kernel/debug/rknpu/load; cat /proc/loadavg; taskset -pc \$\$; \
    python3 -V 2>&1; gcc --version | head -1; } \
    > '$REMOTE/campaign/environment.before.log'"
build_rc=0
"${SSH[@]}" "bash '$REMOTE/source/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
  build '$REMOTE/source' '$REMOTE/build'" > "$LOCAL/build.log" 2>&1 || build_rc=$?
[[ "$build_rc" == 0 ]] || { echo 'remote build failed' >&2; exit 1; }
"${SSH[@]}" "sha256sum '$REMOTE/build/librknpu_plan.so'" \
  > "$LOCAL/library.sha256"
# Install the exact Lite2 wheel; the two pure-Python dependencies are reused
# from the previously verified wheel site and their versions are logged.
"${SSH[@]}" "set -e; mkdir -p '$REMOTE/site'; \
  python3 -m pip install --quiet --no-deps --target '$REMOTE/site' \
    '$REMOTE/campaign/vendor/wheel/'*.whl; \
  cp -a '$REMOTE_BASE/vendor-bench/site/ruamel' \
        '$REMOTE_BASE/vendor-bench/site/ruamel_yaml-0.19.1.dist-info' \
        '$REMOTE_BASE/vendor-bench/site/psutil' \
        '$REMOTE_BASE/vendor-bench/site/psutil-7.2.2.dist-info' \
        '$REMOTE/site/'; \
  PYTHONPATH='$REMOTE/site' python3 -c 'import rknnlite, ruamel.yaml, psutil; print(*("rknnlite", rknnlite.__file__)); print(*("ruamel", ruamel.yaml.__version__)); print(*("psutil", psutil.__version__))' > '$REMOTE/campaign/vendor-site.log'"

# ---- custom execution arms -------------------------------------------------
run_custom() {
  local round=$1 arm=$2
  local plans mode suffix
  case "$arm" in
    generic) plans="$REMOTE/campaign/plans/current"; mode=generic; suffix="";;
    prepared) plans="$REMOTE/campaign/plans/current"; mode=prepared; suffix="";;
    auto) plans="$REMOTE/campaign/plans/current"; mode=auto; suffix="";;
    hostfused) plans="$REMOTE/campaign/plans/candidate"; mode=prepared; suffix=_hostfused;;
    *) echo "unknown custom arm $arm" >&2; return 2;;
  esac
  "${SSH[@]}" "RK_EXECUTION_MODE='$mode' RK_HYBRID_SUFFIX='$suffix' \
    bash '$REMOTE/source/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
      eval '$REMOTE/source' '$REMOTE/build' '$plans' \
      '$REMOTE/campaign/weights' '$REMOTE/campaign/data' \
      '$REMOTE/campaign/numeric/numeric_acceptance.json' \
      '$REMOTE/out/r${round}-${arm}' '$CPU' '$LIMIT'" \
    > "$LOCAL/round-${round}-${arm}.log" 2>&1
}
for ((round = 1; round <= CUSTOM_ROUNDS; round++)); do
  case $((round % 4)) in
    1) arms=(generic prepared auto hostfused);;
    2) arms=(hostfused auto prepared generic);;
    3) arms=(prepared generic hostfused auto);;
    0) arms=(auto hostfused generic prepared);;
  esac
  for arm in "${arms[@]}"; do
    echo "custom round $round $arm" | tee -a "$LOCAL/progress.log"
    run_custom "$round" "$arm"
  done
done

# ---- vendor and CPU arms ---------------------------------------------------
record_external_env() {
  local out=$1
  "${SSH[@]}" "mkdir -p '$out'; { date -u +%FT%TZ; uname -srm; \
    tr -d '\\0' </proc/device-tree/model; echo; cat /sys/kernel/debug/rknpu/version; \
    cat /sys/kernel/debug/rknpu/freq; cat /sys/kernel/debug/rknpu/load; \
    cat /proc/loadavg; taskset -pc \$\$; python3 -V 2>&1; } > '$out/environment.log'"
}
run_external() {
  local round=$1 arm=$2 out="$REMOTE/out/r${round}-${arm}"
  record_external_env "$out"
  local start_ns end_ns rc=0
  start_ns=$(date +%s%N)
  if [[ "$arm" == vendor ]]; then
    "${SSH[@]}" "set -e; trap 'umount /usr/lib/librknnrt.so 2>/dev/null || true' EXIT; \
      mount --bind '$REMOTE/campaign/vendor/runtime/2.3.2/librknnrt.so' \
        /usr/lib/librknnrt.so; \
      cd '$REMOTE/source/tilelang-rknpu'; \
      PYTHONPATH='$REMOTE/site:$REMOTE/source/tilelang-rknpu/python' \
      taskset -c '$CPU' python3 tools/benchmark_vendor_models.py \
        --mode vendor --data-dir '$REMOTE/campaign/data' \
        --weights-dir '$REMOTE/campaign/weights' \
        --rknn-dir '$REMOTE/campaign/vendor/rknn' \
        --out '$out/vendor.json' --limit '$LIMIT' > '$out/runner.log' 2>&1; \
      umount /usr/lib/librknnrt.so; trap - EXIT" || rc=$?
  else
    "${SSH[@]}" "set -e; cd '$REMOTE/source/tilelang-rknpu'; \
      PYTHONPATH='$REMOTE/site:$REMOTE/source/tilelang-rknpu/python' \
      taskset -c '$CPU' python3 tools/benchmark_vendor_models.py \
        --mode cpu --data-dir '$REMOTE/campaign/data' \
        --weights-dir '$REMOTE/campaign/weights' \
        --rknn-dir '$REMOTE/campaign/vendor/rknn' \
        --out '$out/cpu.json' --limit '$LIMIT' > '$out/runner.log' 2>&1" || rc=$?
  fi
  end_ns=$(date +%s%N)
  python3 - "$start_ns" "$end_ns" > "$LOCAL/round-${round}-${arm}.wall" <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)
PY
  echo "$rc" > "$LOCAL/round-${round}-${arm}.exit"
  return "$rc"
}
for ((round = 1; round <= EXTERNAL_ROUNDS; round++)); do
  if (( round % 2 )); then arms=(vendor cpu); else arms=(cpu vendor); fi
  for arm in "${arms[@]}"; do
    echo "external round $round $arm" | tee -a "$LOCAL/progress.log"
    run_external "$round" "$arm"
  done
done

# ---- auto/default-path frozen numeric observation --------------------------
"${SSH[@]}" "RK_EXECUTION_MODE=auto \
  bash '$REMOTE/source/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
    calibrate '$REMOTE/source' '$REMOTE/build' \
    '$REMOTE/campaign/plans/current' '$REMOTE/campaign/weights' \
    '$REMOTE/campaign/data' '$REMOTE/out/numeric-auto-calibration' '$CPU'" \
  > "$LOCAL/numeric-auto-calibration.log" 2>&1
"${SSH[@]}" "RK_EXECUTION_MODE=auto \
  bash '$REMOTE/source/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
    observe '$REMOTE/source' '$REMOTE/build' \
    '$REMOTE/campaign/plans/current' '$REMOTE/campaign/weights' \
    '$REMOTE/campaign/data' \
    '$REMOTE/campaign/numeric/numeric_acceptance_packed.json' \
    '$REMOTE/out/numeric-auto' '$CPU'" \
  > "$LOCAL/numeric-auto.log" 2>&1
"${SSH[@]}" "{ date -u +%FT%TZ; uname -srm; tr -d '\\0' </proc/device-tree/model; \
  echo; cat /sys/kernel/debug/rknpu/version; cat /sys/kernel/debug/rknpu/freq; \
  cat /sys/kernel/debug/rknpu/load; cat /proc/loadavg; } \
  > '$REMOTE/campaign/environment.after.log'"

# ---- retrieve and analyze --------------------------------------------------
mkdir -p "$LOCAL/model-results"
"${SCP[@]}" -r "$TARGET:$REMOTE/out/." "$LOCAL/model-results/"
"${SCP[@]}" "$TARGET:$REMOTE/campaign/environment.before.log" \
  "$TARGET:$REMOTE/campaign/environment.after.log" \
  "$TARGET:$REMOTE/campaign/vendor-site.log" "$LOCAL/"
PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/analyze_execution_candidates.py" \
  --campaign "$LOCAL/campaign" --results "$LOCAL/model-results" \
  --arms generic prepared auto hostfused vendor cpu \
  --out "$LOCAL/analysis.json" --markdown "$LOCAL/analysis.md" \
  > "$LOCAL/analysis.log" 2>&1 || true
PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/verify_numeric_acceptance.py" \
  --spec "$LOCAL/campaign/numeric/numeric_acceptance_packed.json" \
  --calibration-observation \
    "$LOCAL/model-results/numeric-auto-calibration/observation.npz" \
  --acceptance-observation \
    "$LOCAL/model-results/numeric-auto/observation.npz" \
  --out "$LOCAL/verify-auto-summary.json" \
  > "$LOCAL/verify-auto.log" 2>&1 || true
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  exit 1
fi
unset RK_BOARD_PASSWORD
printf 'results: %s\n' "$LOCAL"
