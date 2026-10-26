#!/usr/bin/env bash
# Final RK3588 delivery campaign:
#   baseline (9347624) vs current HEAD, five full static profiles, one payload,
#   one NPU core, one Host CPU, interleaved rounds, plus a fresh packed numeric
#   acceptance observation from the current build.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }

BASELINE_REF=${RK_BASELINE_REF:-42f870f}
CURRENT_REF=$(git -C "$REPO" rev-parse "${RK_CURRENT_REF:-HEAD}")
BASELINE_TREE=$(git -C "$REPO" rev-parse "$BASELINE_REF^{tree}")
LIMIT=${RK_FINAL_LIMIT:-10000}
ROUNDS=${RK_FINAL_ROUNDS:-3}
CPU=${RK_TIME_CPU:-6}
CORE_MASK=${RKNPU_CORE_MASK:-1}
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid limit/rounds' >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-final-$STAMP"
LOCAL="$HERE/results/$STAMP-final"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL"
CAMPAIGN="$LOCAL/campaign"
mkdir -p "$CAMPAIGN"/{plans/baseline,plans/current,weights,data,numeric,tools}

STAGE=$(mktemp -d /tmp/rk-final-XXXXXX)
askpass=$(mktemp /tmp/rk-final-askpass-XXXXXX)
cleanup() { rm -rf -- "$STAGE"; rm -f -- "$askpass"; }
trap cleanup EXIT

# ---- immutable source trees ------------------------------------------------
mkdir -p "$STAGE/source/baseline/tilelang-rknpu" \
         "$STAGE/source/current/tilelang-rknpu" \
         "$STAGE/source/common/upstream-components"
tar --exclude='__pycache__' --exclude='*.tgz' -cf - \
  -C "$REPO/research/upstream-components" . | \
  tar -x -C "$STAGE/source/common/upstream-components"
ln -s ../common/upstream-components \
  "$STAGE/source/baseline/upstream-components"
ln -s ../common/upstream-components \
  "$STAGE/source/current/upstream-components"

git -C "$REPO" archive "$BASELINE_REF" \
  research/custom-ops \
  research/tilelang-rknpu/python \
  research/tilelang-rknpu/tools \
  research/tilelang-rknpu/models/compiled \
  research/tilelang-rknpu/models/numeric_acceptance.json \
  | tar -x -C "$STAGE/source/baseline" --strip-components=1
git -C "$REPO" archive "$CURRENT_REF" \
  research/custom-ops \
  research/tilelang-rknpu/python \
  research/tilelang-rknpu/tools \
  research/tilelang-rknpu/models/compiled \
  research/tilelang-rknpu/models/numeric_acceptance.json \
  research/tilelang-rknpu/models/numeric_acceptance_packed.json \
  research/tilelang-rknpu/models/numeric_golden.npz \
  | tar -x -C "$STAGE/source/current" --strip-components=1
mkdir -p "$STAGE/source/baseline/tilelang-rknpu/scripts" \
         "$STAGE/source/current/tilelang-rknpu/scripts"
cp "$HERE/scripts/run-final-delivery-native.sh" \
   "$STAGE/source/baseline/tilelang-rknpu/scripts/"
cp "$HERE/scripts/run-final-delivery-native.sh" \
   "$STAGE/source/current/tilelang-rknpu/scripts/"

# ---- shared campaign payload ----------------------------------------------
cp "$STAGE/source/baseline/tilelang-rknpu/models/compiled/"*.json \
   "$CAMPAIGN/plans/baseline/"
cp "$STAGE/source/current/tilelang-rknpu/models/compiled/"*.json \
   "$CAMPAIGN/plans/current/"
cp "$HERE/models/mnist_mlp.npz" "$HERE/models/tiny_vit.npz" \
   "$CAMPAIGN/weights/"
cp "$HERE/.data/mnist/t10k-images-idx3-ubyte.gz" \
   "$HERE/.data/mnist/t10k-labels-idx1-ubyte.gz" \
   "$HERE/.data/mnist/train-images-idx3-ubyte.gz" "$CAMPAIGN/data/"
cp "$HERE/models/numeric_acceptance.json" \
   "$CAMPAIGN/numeric/numeric_acceptance.json"
cp "$HERE/models/numeric_acceptance_packed.json" \
   "$CAMPAIGN/numeric/numeric_acceptance_packed.json"
cp "$HERE/models/numeric_golden.npz" "$CAMPAIGN/numeric/numeric_golden.npz"

python3 - "$HERE" "$CAMPAIGN" "$BASELINE_REF" "$BASELINE_TREE" \
  "$CURRENT_REF" "$LIMIT" "$ROUNDS" "$CORE_MASK" "$CPU" <<'PY'
import hashlib, json, sys
from pathlib import Path
here, camp = Path(sys.argv[1]), Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
manifest = {
    'schema_version': 1,
    'kind': 'final-model-delivery-campaign',
    'baseline': {
        'git_ref': sys.argv[3],
        'tree_sha': sys.argv[4],
        'compiled_manifest_sha256': sha(camp / 'plans/baseline/manifest.json'),
    },
    'current': {
        'git_ref': sys.argv[5],
        'compiled_manifest_sha256': sha(camp / 'plans/current/manifest.json'),
        'default_hybrid': 'tiny_vit_bN',
        'default_tiny_vit_layout': 'mapped_qkv',
        'splitheads_control': 'tiny_vit_bN_splitheads',
        'opt_in_manifests': ['tiny_vit_bN_deviceffn',
                             'tiny_vit_bN_hostffn',
                             'tiny_vit_bN_rmsnorm'],
    },
    'evaluation_images_per_profile': int(sys.argv[6]),
    'full_test_images': int(sys.argv[6]),
    'full_mnist': int(sys.argv[6]) == 10000,
    'profile_samples': int(sys.argv[6]),
    'all_five_profiles_full': True,
    'rounds': int(sys.argv[7]),
    'core_mask': int(sys.argv[8]),
    'cpu_affinity_pin': int(sys.argv[9]),
    'interleaved': True,
    'expected_rows': [
        {'model': 'mnist_mlp', 'batch': 32},
        {'model': 'mnist_mlp', 'batch': 8},
        {'model': 'mnist_mlp', 'batch': 1},
        {'model': 'tiny_vit', 'batch': 8},
        {'model': 'tiny_vit', 'batch': 1},
    ],
    'thresholds': {'mnist_mlp_accuracy': 0.97, 'tiny_vit_accuracy': 0.95,
                   'accuracy_drop_max': 0.005,
                   'top1_agreement_min': 0.99},
    'weights': {p.name: sha(p) for p in sorted((camp/'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp/'data').glob('*.gz'))},
    'numeric': {
        'evaluator_spec': {
            'file': 'numeric/numeric_acceptance.json',
            'sha256': sha(camp/'numeric/numeric_acceptance.json')},
        'packed_spec': {
            'file': 'numeric/numeric_acceptance_packed.json',
            'sha256': sha(camp/'numeric/numeric_acceptance_packed.json')},
        'golden': {
            'file': 'numeric/numeric_golden.npz',
            'sha256': sha(camp/'numeric/numeric_golden.npz')},
    },
    'payload': {'weights_dir': 'weights', 'data_dir': 'data',
                'baseline_plans_dir': 'plans/baseline',
                'current_plans_dir': 'plans/current'},
    'delivery_tools': {
        'scripts/run-final-delivery.sh': sha(here/'scripts/run-final-delivery.sh'),
        'scripts/run-final-delivery-native.sh': sha(
            here/'scripts/run-final-delivery-native.sh'),
        'tools/analyze_final_delivery.py': sha(
            here/'tools/analyze_final_delivery.py'),
    },
}
(camp / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(sha(camp / 'manifest.json'))
PY
(cd "$CAMPAIGN" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

# ---- upload source/campaign -------------------------------------------------
tar -czf "$STAGE.campaign.tgz" -C "$LOCAL" campaign
tar -czf "$STAGE.source.tgz" -C "$STAGE" source
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$STAGE.source.tgz" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$STAGE.campaign.tgz" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "cd '$REMOTE' && tar -xzf source.tgz && tar -xzf campaign.tgz"

# ---- build both arms before any measurement --------------------------------
status=0
for arm in baseline current; do
  rc=0
  "${SSH[@]}" "bash '$REMOTE/source/$arm/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
      build '$REMOTE/source/$arm' '$REMOTE/build-$arm'" \
      > "$LOCAL/$arm.build.log" 2>&1 || rc=$?
  [[ "$rc" == 0 ]] || { echo "build failed: $arm" >&2; status=1; }
done
[[ "$status" == 0 ]] || exit 1

# ---- interleaved full-model rounds -----------------------------------------
for ((round = 1; round <= ROUNDS; round++)); do
  if (( round % 2 )); then arms=(baseline current); else arms=(current baseline); fi
  for arm in "${arms[@]}"; do
    rc=0
    "${SSH[@]}" "bash '$REMOTE/source/$arm/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
        eval '$REMOTE/source/$arm' '$REMOTE/build-$arm' \
        '$REMOTE/campaign/plans/$arm' '$REMOTE/campaign/weights' \
        '$REMOTE/campaign/data' '$REMOTE/campaign/numeric/numeric_acceptance.json' \
        '$REMOTE/out/r${round}-${arm}' '$CPU' '$LIMIT'" \
        > "$LOCAL/round-${round}-${arm}.log" 2>&1 || rc=$?
    [[ "$rc" == 0 ]] || { echo "eval failed: round $round $arm" >&2; status=1; }
  done
done

# ---- fresh packed numeric acceptance observation (current arm) --------------
rc=0
"${SSH[@]}" "bash '$REMOTE/source/current/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
    calibrate '$REMOTE/source/current' '$REMOTE/build-current' \
    '$REMOTE/campaign/plans/current' '$REMOTE/campaign/weights' \
    '$REMOTE/campaign/data' '$REMOTE/out/numeric-current-calibration' '$CPU'" \
    > "$LOCAL/numeric-current-calibration.log" 2>&1 || rc=$?
[[ "$rc" == 0 ]] || { echo 'packed numeric calibration observation failed' >&2; status=1; }
rc=0
"${SSH[@]}" "bash '$REMOTE/source/current/tilelang-rknpu/scripts/run-final-delivery-native.sh' \
    observe '$REMOTE/source/current' '$REMOTE/build-current' \
    '$REMOTE/campaign/plans/current' '$REMOTE/campaign/weights' \
    '$REMOTE/campaign/data' '$REMOTE/campaign/numeric/numeric_acceptance_packed.json' \
    '$REMOTE/out/numeric-current' '$CPU'" \
    > "$LOCAL/numeric-current.log" 2>&1 || rc=$?
[[ "$rc" == 0 ]] || { echo 'packed numeric observation failed' >&2; status=1; }

mkdir -p "$LOCAL/model-results"
scp_rc=0
"${SCP[@]}" -r "$TARGET:$REMOTE/out/." "$LOCAL/model-results/" || scp_rc=$?
[[ "$scp_rc" == 0 ]] || status=1

# ---- independent local checks ----------------------------------------------
analysis_rc=0
PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/analyze_final_delivery.py" \
  --campaign "$CAMPAIGN" --results "$LOCAL/model-results" \
  --out "$LOCAL/analysis.json" --markdown "$LOCAL/analysis.md" \
  > "$LOCAL/analysis.log" 2>&1 || analysis_rc=$?
[[ "$analysis_rc" == 0 ]] || status=1

CAL="$LOCAL/model-results/numeric-current-calibration/observation.npz"
packed_rc=0
if [[ -f "$CAL" && -f "$LOCAL/model-results/numeric-current/observation.npz" ]]; then
  PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/verify_numeric_acceptance.py" \
    --spec "$HERE/models/numeric_acceptance_packed.json" \
    --calibration-observation "$CAL" \
    --acceptance-observation "$LOCAL/model-results/numeric-current/observation.npz" \
    --out "$LOCAL/verify-packed-summary.json" \
    > "$LOCAL/verify-packed.log" 2>&1 || packed_rc=$?
else
  echo 'missing packed numeric observation/calibration' >&2
  packed_rc=1
fi
[[ "$packed_rc" == 0 ]] || status=1

if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
fi
unset RK_BOARD_PASSWORD
printf 'RK3588 final model delivery: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
