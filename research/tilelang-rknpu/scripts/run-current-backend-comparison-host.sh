#!/usr/bin/env bash
# Secure Host orchestration for the frozen current/vendor/CPU comparison.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
source "$REPO/.env"
: "${ORANGE_PI_HOST:?missing board host}"
: "${ORANGE_PI_USER:?missing board user}"
: "${ORANGE_PI_PASSWORD:?missing board password}"
BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL="$HERE/results/$STAMP-current-backend-comparison"
REMOTE="$BASE/tl-current-backend-$STAMP"
mkdir -p "$LOCAL/campaign" "$LOCAL/board"
C="$LOCAL/campaign"
mkdir -p "$C/plans" "$C/weights" "$C/data" "$C/vendor/rknn" "$C/vendor/runtime/2.3.2" "$C/vendor/wheel" "$C/tools"
cp "$HERE/models/compiled/manifest.json" "$C/plans/manifest.json"
cp "$HERE/models/compiled/"*.rkhybrid.json "$C/plans/"
cp "$HERE/models/compiled/"*.rkplan.json "$C/plans/"
cp "$HERE/models/"{mnist_mlp,tiny_vit}.npz "$C/weights/"
cp "$HERE/.data/mnist/"{t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz} "$C/data/"
cp "$HERE/vendor/rknn/"{rknn-manifest.json,*.rknn} "$C/vendor/rknn/"
cp "$HERE/vendor/runtime/2.3.2/librknnrt.so" "$C/vendor/runtime/2.3.2/"
cp "$HERE/vendor/wheel/"*.whl "$C/vendor/wheel/"
cp "$HERE/models/numeric_acceptance_packed.json" "$C/numeric_acceptance_packed.json"
python3 - "$C/run-manifest.json" "$REPO" "$HERE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
out,repo,here=map(Path,sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={"kind":"current-vendor-cpu-comparison","schema_version":1,"source_head":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),"compiled_manifest_sha256":sha(here/"models/compiled/manifest.json"),"vendor_manifest_sha256":sha(here/"vendor/rknn/rknn-manifest.json"),"vendor_runtime_sha256":sha(here/"vendor/runtime/2.3.2/librknnrt.so"),"numeric_policy":{"frozen_spec_path":"models/numeric_acceptance_packed.json","frozen_spec_sha256":sha(here/"models/numeric_acceptance_packed.json")},"policy":{"current":"frozen current runtime","vendor":"RKNN Toolkit Lite 2.3.2","cpu":"FP32 reference"}}
out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
PY
(cd "$C" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
source_bundle=$(mktemp /tmp/current-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/current-campaign-XXXXXX.tgz)
askpass=$(mktemp /tmp/current-askpass-XXXXXX)
cleanup(){ rm -f -- "$source_bundle" "$campaign_bundle" "$askpass"; }
trap cleanup EXIT
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD" SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
tar --exclude='build*' --exclude='results' --exclude='.venv*' --exclude='.cache' --exclude='__pycache__' -czf "$source_bundle" -C "$REPO/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit tilelang-rknpu/python tilelang-rknpu/tools/benchmark_current_backends.py tilelang-rknpu/tools/evaluate_models.py tilelang-rknpu/scripts/run-current-backend-comparison-native.sh
tar -czf "$campaign_bundle" -C "$LOCAL" campaign
cp "$source_bundle" "$LOCAL/source-bundle.tgz"
sha256sum "$LOCAL/source-bundle.tgz" > "$LOCAL/source-bundle.sha256"
"${SSH[@]}" "ps -eo args | grep -E 'benchmark_current_backends|run-current-backend-comparison' | grep -v grep && exit 9 || true; mountpoint -q /usr/lib/librknnrt.so && exit 10 || true; mkdir -p '$REMOTE'"
"${SCP[@]}" "$source_bundle" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$campaign_bundle" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "mkdir -p '$REMOTE/source' && tar -xzf '$REMOTE/source.tgz' -C '$REMOTE/source' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE' && mkdir -p '$REMOTE/results'"
"${SSH[@]}" "bash '$REMOTE/source/tilelang-rknpu/scripts/run-current-backend-comparison-native.sh' install-site '$REMOTE' '$BASE/vendor-bench/site' '$REMOTE/results/site'" > "$LOCAL/vendor-site-ssh.log" 2>&1
set +e
"${SSH[@]}" "bash '$REMOTE/source/tilelang-rknpu/scripts/run-current-backend-comparison-native.sh' run '$REMOTE' all 32 1 '$REMOTE/results/smoke.json'" > "$LOCAL/smoke-ssh.log" 2>&1
smoke_rc=$?
printf '%s\n' "$smoke_rc" > "$LOCAL/smoke.exit"
if (( smoke_rc == 0 )); then
  "${SSH[@]}" "bash '$REMOTE/source/tilelang-rknpu/scripts/run-current-backend-comparison-native.sh' run '$REMOTE' all 10000 3 '$REMOTE/results/full.json'" > "$LOCAL/full-ssh.log" 2>&1
  full_rc=$?
else full_rc=99; fi
printf '%s\n' "$full_rc" > "$LOCAL/full.exit"
"${SCP[@]}" -r "$TARGET:$REMOTE/results/." "$LOCAL/board/" || full_rc=98
set -e
unset RK_BOARD_PASSWORD
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" >/dev/null 2>&1; then echo 'credential scan failed' >&2; exit 1; fi
printf 'results=%s smoke_exit=%s full_exit=%s\n' "$LOCAL" "$smoke_rc" "$full_rc"
(( smoke_rc == 0 && full_rc == 0 ))
