#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Post-freeze numeric acceptance replay.
#
# Re-observes the *disjoint test split* with the current runtime and verifies it
# against the already frozen train-calibrated spec.  Nothing is re-calibrated and
# no limit is recomputed, so this can only fail if the runtime changed a number.
#
# The calibration observation recorded by the original freeze
# (`results/20261024T091634Z-numeric/calibration/observation.npz`) is reused as
# the calibration side of the verifier, which is what makes this a replay of the
# frozen acceptance boundary rather than a new campaign.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }

FROZEN_CAMPAIGN=${RK_FROZEN_CAMPAIGN:-$HERE/results/20261024T091634Z-numeric}
FROZEN_SPEC=${RK_FROZEN_SPEC:-$HERE/models/numeric_acceptance_train.json}
CALIBRATION_OBS="$FROZEN_CAMPAIGN/calibration/observation.npz"
REFERENCE_CAMPAIGN=${RK_REFERENCE_CAMPAIGN:-$HERE/results/20261024T071144Z-models}
[[ -f "$CALIBRATION_OBS" ]] || { echo "missing frozen calibration observation: $CALIBRATION_OBS" >&2; exit 2; }
[[ -f "$FROZEN_SPEC" ]] || { echo "missing frozen spec: $FROZEN_SPEC" >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-numeric-replay-$STAMP"
LOCAL="$HERE/results/$STAMP-numeric-replay"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL/campaign"/{plans,weights,data,tools,numeric}
CAMPAIGN="$LOCAL/campaign"

cp "$HERE"/models/compiled/*.rkplan.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/manifest.json "$CAMPAIGN/plans/manifest.json"
cp "$HERE"/models/mnist_mlp.npz "$HERE"/models/tiny_vit.npz "$CAMPAIGN/weights/"
cp "$HERE"/.data/mnist/train-images-idx3-ubyte.gz \
   "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/tools/run_numeric_observation.py "$CAMPAIGN/tools/"
cp "$FROZEN_SPEC" "$CAMPAIGN/numeric/numeric_acceptance_train.json"

python3 - "$HERE" "$CAMPAIGN" "$FROZEN_SPEC" "$CALIBRATION_OBS" <<'PY'
import hashlib, json, sys
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
    'kind': 'numeric-acceptance-replay',
    'claims': {
        'recalibrated': False,
        'limits_recomputed': False,
        'acceptance_split': 'test images only',
        'acceptance_labels_read': False,
        'frozen_spec_reused': True,
    },
    'frozen_spec': {'file': 'numeric/numeric_acceptance_train.json',
                    'sha256': sha(camp / 'numeric/numeric_acceptance_train.json')},
    'frozen_calibration_observation': {'path': str(sys.argv[4]),
                                       'sha256': sha(Path(sys.argv[4]))},
    'source_hashes': {
        'python/rknpu_backend/models.py': sha(here / 'python/rknpu_backend/models.py'),
        'python/rknpu_backend/hybrid.py': sha(here / 'python/rknpu_backend/hybrid.py'),
        'python/rknpu_backend/runtime/ffi.py': sha(here / 'python/rknpu_backend/runtime/ffi.py'),
        'python/rknpu_backend/numeric_observation.py': sha(here / 'python/rknpu_backend/numeric_observation.py'),
        'custom-ops/src/rk_plan_runtime.c': sha(here / '../custom-ops/src/rk_plan_runtime.c'),
        'custom-ops/src/custom_ops.c': sha(here / '../custom-ops/src/custom_ops.c'),
        'custom-ops/include/rk_plan.h': sha(here / '../custom-ops/include/rk_plan.h'),
        'models/compiled/manifest.json': sha(here / 'models/compiled/manifest.json'),
        'tools/run_numeric_observation.py': sha(here / 'tools/run_numeric_observation.py'),
    },
    'plans': {p.name: sha(p) for p in sorted((camp / 'plans').glob('*.json'))},
    'weights': {p.name: sha(p) for p in sorted((camp / 'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp / 'data').glob('*.gz'))},
}
(camp / 'run-manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(camp / 'run-manifest.json')
PY
(cd "$CAMPAIGN" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

askpass=$(mktemp /tmp/rk-replay-askpass-XXXXXX)
source_bundle=$(mktemp /tmp/rk-replay-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/rk-replay-campaign-XXXXXX.tgz)
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
if (( status == 0 )) && [[ "${RK_REOBSERVE_CALIBRATION:-0}" == "1" ]]; then
  "${SSH[@]}" "bash '$REMOTE/tilelang-rknpu/scripts/run-numeric-acceptance-native.sh' calibration '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote-calibration.log" 2>&1 || status=1
  if (( status == 0 )); then
    mkdir -p "$LOCAL/calibration"
    "${SCP[@]}" -r "$TARGET:$REMOTE/results/calibration/." "$LOCAL/calibration/" || status=1
    CALIBRATION_OBS="$LOCAL/calibration/observation.npz"
  fi
fi
if (( status == 0 )); then
  "${SSH[@]}" "RK_NUMERIC_MANIFEST_MAP='${RK_NUMERIC_MANIFEST_MAP:-}' bash '$REMOTE/tilelang-rknpu/scripts/run-numeric-acceptance-native.sh' acceptance '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote-acceptance.log" 2>&1 || status=1
fi
if (( status == 0 )); then
  mkdir -p "$LOCAL/acceptance" "$LOCAL/board"
  "${SCP[@]}" -r "$TARGET:$REMOTE/results/acceptance/." "$LOCAL/acceptance/" || status=1
  "${SCP[@]}" -r "$TARGET:$REMOTE/results/environment.log" "$TARGET:$REMOTE/results/library.sha256" "$LOCAL/board/" || status=1
fi
if (( status == 0 )); then
  PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/verify_numeric_acceptance.py" \
    --spec "$FROZEN_SPEC" \
    --calibration-observation "$CALIBRATION_OBS" \
    --acceptance-observation "$LOCAL/acceptance/observation.npz" \
    --campaign-manifest "$REFERENCE_CAMPAIGN/campaign/manifest.json" \
    --campaign-results "$REFERENCE_CAMPAIGN/model-results" \
    --out "$LOCAL/verify-summary.json" > "$LOCAL/verify.log" 2>&1 || status=1
fi
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
fi
unset RK_BOARD_PASSWORD
printf 'RK3588 post-freeze numeric acceptance replay: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
