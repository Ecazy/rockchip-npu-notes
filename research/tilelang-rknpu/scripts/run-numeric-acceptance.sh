#!/usr/bin/env bash
# Train-calibrate numeric limits, then run a disjoint post-freeze test observation
# on RK3588 core 1. No labels are uploaded; no tolerance is computed from test.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }
NUMERIC_SPEC=${RK_NUMERIC_SPEC:-numeric_acceptance_train.json}
[[ "$NUMERIC_SPEC" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'unsafe numeric spec name' >&2; exit 2; }
[[ -f "$HERE/models/$NUMERIC_SPEC" ]] || { echo "missing numeric spec: $NUMERIC_SPEC" >&2; exit 2; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-numeric-$STAMP"
LOCAL="$HERE/results/$STAMP-numeric"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL/campaign"/{plans,weights,data,tools,numeric}
mkdir -p "$LOCAL/calibration" "$LOCAL/acceptance"
CAMPAIGN="$LOCAL/campaign"
cp "$HERE"/models/compiled/*.rkplan.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/manifest.json "$CAMPAIGN/plans/manifest.json"
cp "$HERE"/models/mnist_mlp.npz "$HERE"/models/tiny_vit.npz "$CAMPAIGN/weights/"
cp "$HERE"/.data/mnist/train-images-idx3-ubyte.gz \
   "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/tools/run_numeric_observation.py "$CAMPAIGN/tools/"
cp "$HERE/models/$NUMERIC_SPEC" "$CAMPAIGN/numeric/"
python3 - "$HERE" "$CAMPAIGN" "$NUMERIC_SPEC" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
here, camp = Path(sys.argv[1]), Path(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
report = {
    'schema_version': 1,
    'kind': 'train-calibrated-numeric-campaign',
    'source_hashes': {
        'python/rknpu_backend/models.py': sha(here / 'python/rknpu_backend/models.py'),
        'python/rknpu_backend/model_reference.py': sha(here / 'python/rknpu_backend/model_reference.py'),
        'python/rknpu_backend/numeric.py': sha(here / 'python/rknpu_backend/numeric.py'),
        'python/rknpu_backend/numeric_policy.py': sha(here / 'python/rknpu_backend/numeric_policy.py'),
        'python/rknpu_backend/numeric_observation.py': sha(here / 'python/rknpu_backend/numeric_observation.py'),
        'models/tilelang_mlp.py': sha(here / 'models/tilelang_mlp.py'),
        'models/compiled/manifest.json': sha(here / 'models/compiled/manifest.json'),
        'models/numeric_calibration_train.npz': sha(here / 'models/numeric_calibration_train.npz'),
        'models/numeric_acceptance_test.npz': sha(here / 'models/numeric_acceptance_test.npz'),
        'tools/run_numeric_observation.py': sha(here / 'tools/run_numeric_observation.py'),
        'tools/calibrate_numeric_train.py': sha(here / 'tools/calibrate_numeric_train.py'),
        'tools/verify_numeric_acceptance.py': sha(here / 'tools/verify_numeric_acceptance.py'),
    },
    'weights': {p.name: sha(p) for p in sorted((camp / 'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp / 'data').glob('*.gz'))},
    'plans': {p.name: sha(p) for p in sorted((camp / 'plans').glob('*.json'))},
    'initial_spec': {'file': sys.argv[3],
                     'sha256': sha(camp / 'numeric' / sys.argv[3])},
    'claims': {'calibration_split': 'train images only',
               'calibration_labels_read': False,
               'acceptance_split': 'test images only',
               'acceptance_labels_read': False,
               'test_used_for_tolerance': False,
               'limits_scope': 'empirical RK3588 envelope for frozen model package'},
}
(camp / 'run-manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(camp / 'run-manifest.json')
PY
(cd "$CAMPAIGN" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
askpass=$(mktemp /tmp/rk-numeric-askpass-XXXXXX)
source_bundle=$(mktemp /tmp/rk-numeric-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/rk-numeric-campaign-XXXXXX.tgz)
cleanup() { rm -f -- "$askpass" "$source_bundle" "$campaign_bundle"; }
trap cleanup EXIT
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
tar --exclude='.git' --exclude='build*' --exclude='results' --exclude='artifacts' \
    --exclude='*.tgz' --exclude='__pycache__' \
    -czf "$source_bundle" -C "$REPO/research" \
    custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit \
    tilelang-rknpu/capabilities tilelang-rknpu/python \
    tilelang-rknpu/scripts/run-numeric-acceptance-native.sh
tar -czf "$campaign_bundle" -C "$LOCAL" campaign
cp -- "$source_bundle" "$LOCAL/source-bundle.tgz"
sha256sum "$LOCAL/source-bundle.tgz" > "$LOCAL/source-bundle.sha256"
"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$source_bundle" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$campaign_bundle" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "tar -xzf '$REMOTE/source.tgz' -C '$REMOTE' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE'"
status=0
"${SSH[@]}" "bash '$REMOTE/tilelang-rknpu/scripts/run-numeric-acceptance-native.sh' build '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote-build.log" 2>&1 || status=1
if (( status == 0 )); then
  "${SSH[@]}" "bash '$REMOTE/tilelang-rknpu/scripts/run-numeric-acceptance-native.sh' calibration '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote-calibration.log" 2>&1 || status=1
fi
if (( status == 0 )); then
  mkdir -p "$LOCAL/calibration"
  "${SCP[@]}" -r "$TARGET:$REMOTE/results/calibration/." "$LOCAL/calibration/" || status=1
fi
if (( status == 0 )); then
  PYTHONPATH="$HERE/python" python3 "$HERE/tools/calibrate_numeric_train.py" \
    --spec "$CAMPAIGN/numeric/$NUMERIC_SPEC" \
    --calibration "$HERE/models/numeric_calibration_train.npz" \
    --observation "$LOCAL/calibration/observation.npz" \
    --out "$HERE/models/$NUMERIC_SPEC" \
    > "$LOCAL/calibration/freeze.log" 2>&1 || status=1
fi
if (( status == 0 )); then
  cp "$HERE/models/$NUMERIC_SPEC" "$LOCAL/frozen_spec.json"
  "${SCP[@]}" "$HERE/models/$NUMERIC_SPEC" "$TARGET:$REMOTE/campaign/numeric/$NUMERIC_SPEC" || status=1
fi
if (( status == 0 )); then
  "${SSH[@]}" "RK_NUMERIC_SPEC='$NUMERIC_SPEC' bash '$REMOTE/tilelang-rknpu/scripts/run-numeric-acceptance-native.sh' acceptance '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote-acceptance.log" 2>&1 || status=1
fi
if (( status == 0 )); then
  mkdir -p "$LOCAL/acceptance"
  "${SCP[@]}" -r "$TARGET:$REMOTE/results/acceptance/." "$LOCAL/acceptance/" || status=1
fi
if (( status == 0 )); then
  mkdir -p "$LOCAL/board"
  "${SCP[@]}" -r "$TARGET:$REMOTE/results/environment.log" "$TARGET:$REMOTE/results/library.sha256" "$LOCAL/board/" || status=1
fi
if (( status == 0 )); then
  PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/verify_numeric_acceptance.py" \
    --spec "$HERE/models/$NUMERIC_SPEC" \
    --calibration-observation "$LOCAL/calibration/observation.npz" \
    --acceptance-observation "$LOCAL/acceptance/observation.npz" \
    --campaign-manifest "$HERE/results/20261024T071144Z-models/campaign/manifest.json" \
    --campaign-results "$HERE/results/20261024T071144Z-models/model-results" \
    --out "$LOCAL/verify-summary.json" > "$LOCAL/verify.log" 2>&1 || status=1
fi
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
fi
unset RK_BOARD_PASSWORD
printf 'RK3588 train-calibrated numeric acceptance: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
