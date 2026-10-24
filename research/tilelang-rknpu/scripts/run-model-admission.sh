#!/usr/bin/env bash
# Upload precompiled complete-model plans and run full/subsample acceptance.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-models-$STAMP"
LOCAL="$HERE/results/$STAMP-models"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL/campaign"/{plans,weights,data,tools,numeric}
CAMPAIGN="$LOCAL/campaign"
cp "$HERE"/models/compiled/*.rkplan.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/manifest.json "$CAMPAIGN/plans/manifest.json"
cp "$HERE"/models/*.npz "$CAMPAIGN/weights/"
cp "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/.data/mnist/t10k-labels-idx1-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/models/numeric_acceptance.json "$CAMPAIGN/numeric/"
cp "$HERE"/models/numeric_golden.npz "$CAMPAIGN/numeric/"
cp "$HERE"/tools/evaluate_models.py "$CAMPAIGN/tools/"
cp "$HERE"/tools/benchmark_cpu_models.py "$CAMPAIGN/tools/"
python3 - "$HERE" "$CAMPAIGN" <<'PY'
import hashlib, json, sys
from pathlib import Path
here, camp = Path(sys.argv[1]), Path(sys.argv[2])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
manifest={
 'schema_version':1,
 'kind':'complete-model-campaign',
 'source_hashes':{
  'models/tilelang_mlp.py':sha(here/'models/tilelang_mlp.py'),
  'models/tilelang_matmul.py':sha(here/'models/tilelang_matmul.py'),
  'models/compiled/manifest.json':sha(here/'models/compiled/manifest.json'),
  'models/numeric_acceptance.json':sha(here/'models/numeric_acceptance.json'),
  'models/numeric_golden.npz':sha(here/'models/numeric_golden.npz'),
  'python/rknpu_backend/hybrid.py':sha(here/'python/rknpu_backend/hybrid.py'),
  'python/rknpu_backend/models.py':sha(here/'python/rknpu_backend/models.py'),
  'python/rknpu_backend/hostinfo.py':sha(here/'python/rknpu_backend/hostinfo.py'),
  'python/rknpu_backend/model_reference.py':sha(here/'python/rknpu_backend/model_reference.py'),
  'python/rknpu_backend/numeric.py':sha(here/'python/rknpu_backend/numeric.py'),
  'tools/evaluate_models.py':sha(here/'tools/evaluate_models.py'),
  'tools/benchmark_cpu_models.py':sha(here/'tools/benchmark_cpu_models.py'),
  'tools/verify_model_results.py':sha(here/'tools/verify_model_results.py'),
 },
 'payload':{'plans_dir':'plans','weights_dir':'weights','data_dir':'data',
            'numeric_dir':'numeric'},
 'numeric':{
  'spec':{'file':'numeric_acceptance.json',
          'sha256':sha(here/'models/numeric_acceptance.json')},
  'golden':{'file':'numeric_golden.npz',
            'sha256':sha(here/'models/numeric_golden.npz')},
 },
 'expected_rows':[
  {'model':'mnist_mlp','batch':32,'primary':True},
  {'model':'mnist_mlp','batch':8,'primary':False},
  {'model':'mnist_mlp','batch':1,'primary':False},
  {'model':'tiny_vit','batch':8,'primary':True},
  {'model':'tiny_vit','batch':1,'primary':False},
 ],
 'weights':{p.name:sha(p) for p in sorted((camp/'weights').glob('*.npz'))},
 'data':{p.name:sha(p) for p in sorted((camp/'data').glob('*.gz'))},
 'compiled_manifest':json.loads((here/'models/compiled/manifest.json').read_text()),
 'thresholds':{'mnist_mlp_accuracy':0.97,'tiny_vit_accuracy':0.95,
               'accuracy_drop_max':0.005,'top1_agreement_min':0.99},
 'full_test_images':10000,'admission_subsample':512,
}
(camp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(sha(camp/'manifest.json'))
PY
(cd "$CAMPAIGN" && find . -type f ! -name '*.npy' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

askpass=$(mktemp /tmp/rk-models-askpass-XXXXXX)
source_bundle=$(mktemp /tmp/rk-models-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/rk-models-campaign-XXXXXX.tgz)
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
    tilelang-rknpu/scripts/run-model-admission-native.sh
tar -czf "$campaign_bundle" -C "$LOCAL" campaign
cp -- "$source_bundle" "$LOCAL/source-bundle.tgz"
sha256sum "$LOCAL/source-bundle.tgz" > "$LOCAL/source-bundle.sha256"

"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$source_bundle" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$campaign_bundle" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "tar -xzf '$REMOTE/source.tgz' -C '$REMOTE' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE'"

status=0
"${SSH[@]}" "bash '$REMOTE/tilelang-rknpu/scripts/run-model-admission-native.sh' '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote.log" 2>&1 || status=$?
scp_rc=0
mkdir -p "$LOCAL/model-results"
"${SCP[@]}" -r "$TARGET:$REMOTE/results/." "$LOCAL/model-results/" || scp_rc=$?
[[ "$scp_rc" == 0 ]] || status=1
if ! python3 "$HERE/tools/verify_model_results.py" --manifest "$CAMPAIGN/manifest.json" \
    --results "$LOCAL/model-results" > "$LOCAL/verify.log" 2>&1; then
  status=1
fi
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
fi
unset RK_BOARD_PASSWORD
printf 'RK3588 complete-model admission: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
