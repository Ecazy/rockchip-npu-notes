#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# A/B the fixed QK/PV layout pack: 16x16 fast path vs the generic pack.
# Same source, build, payload and rkhybrid manifest; coalescing stays on and
# only RK_PLAN_UPLOAD_FAST_PACK differs.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }

ARMS=${RK_UPLOAD_ARMS:-fast,generic}
PROFILES=${RK_ENVELOPE_PROFILES:-tiny_vit:8,tiny_vit:1}
RUNS=${RK_ENVELOPE_RUNS:-32}
ROUNDS=${RK_ENVELOPE_ROUNDS:-4}
WARMUP=${RK_ENVELOPE_WARMUP:-8}
CORE_MASK=${RKNPU_CORE_MASK:-1}
TIME_CPU=${RK_TIME_CPU:-6}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-layout-$STAMP"
LOCAL="$HERE/results/$STAMP-layout"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL/campaign"/{plans,weights,data,tools}
CAMPAIGN="$LOCAL/campaign"
cp "$HERE"/models/compiled/*.rkplan.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/manifest.json "$CAMPAIGN/plans/manifest.json"
cp "$HERE"/models/tiny_vit.npz "$HERE"/models/mnist_mlp.npz "$CAMPAIGN/weights/"
cp "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz \
   "$HERE"/.data/mnist/t10k-labels-idx1-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/tools/profile_bridge_envelope.py "$CAMPAIGN/tools/"

python3 - "$HERE" "$CAMPAIGN" "$ARMS" "$PROFILES" "$RUNS" "$ROUNDS" "$WARMUP" \
  "$CORE_MASK" "$TIME_CPU" <<'PY'
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
    'kind': 'upload-layout-campaign',
    'arms': sys.argv[3].split(','),
    'profiles': [{'model': p.split(':')[0], 'batch': int(p.split(':')[1])}
                 for p in sys.argv[4].split(',')],
    'runs_per_round': int(sys.argv[5]),
    'rounds': int(sys.argv[6]),
    'warmup': int(sys.argv[7]),
    'core_mask': int(sys.argv[8]),
    'cpu_affinity_pin': int(sys.argv[9]),
    'interleaved': True,
    'source_hashes': {
        'custom-ops/src/rk_plan_runtime.c':
            sha(here / '../custom-ops/src/rk_plan_runtime.c'),
        'custom-ops/include/rk_plan.h':
            sha(here / '../custom-ops/include/rk_plan.h'),
        'python/rknpu_backend/hybrid.py': sha(here / 'python/rknpu_backend/hybrid.py'),
        'python/rknpu_backend/models.py': sha(here / 'python/rknpu_backend/models.py'),
        'tools/profile_bridge_envelope.py':
            sha(here / 'tools/profile_bridge_envelope.py'),
        'models/compiled/manifest.json': sha(here / 'models/compiled/manifest.json'),
    },
    'plans': {p.name: sha(p) for p in sorted((camp / 'plans').glob('*.json'))},
    'weights': {p.name: sha(p) for p in sorted((camp / 'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp / 'data').glob('*.gz'))},
}
(camp / 'run-manifest.json').write_text(json.dumps(manifest, indent=2,
                                                  sort_keys=True) + '\n')
print(camp / 'run-manifest.json')
PY
(cd "$CAMPAIGN" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

STAGE=$(mktemp -d /tmp/rk-upload-XXXXXX)
askpass=$(mktemp /tmp/rk-upload-askpass-XXXXXX)
cleanup() { rm -rf -- "$STAGE"; rm -f -- "$askpass" "$STAGE"*.tgz; }
trap cleanup EXIT
tar --exclude='build*' --exclude='artifacts' --exclude='results' \
    --exclude='__pycache__' --exclude='*.so' --exclude='*.tgz' \
    -czf "$STAGE/source.tgz" -C "$REPO/research" \
    custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit \
    tilelang-rknpu/python tilelang-rknpu/tools
tar -czf "$STAGE/campaign.tgz" -C "$LOCAL" campaign

printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$STAGE/source.tgz" "$TARGET:$REMOTE/source.tgz" >/dev/null
"${SCP[@]}" "$STAGE/campaign.tgz" "$TARGET:$REMOTE/campaign.tgz" >/dev/null

status=0
"${SSH[@]}" "set -e; cd '$REMOTE'; tar -xzf source.tgz; tar -xzf campaign.tgz
  cmake -S custom-ops -B build -DUPSTREAM_ROOT=\$PWD/upstream-components \
      -DCMAKE_BUILD_TYPE=Release > build-config.log 2>&1
  cmake --build build --target rknpu_plan -j4 > build.log 2>&1
  sha256sum build/librknpu_plan.so > library.sha256
  { echo collected_utc=\$(date -u +%FT%TZ); uname -srm
    echo board_model=\$(tr -d '\\0' </proc/device-tree/model)
    echo driver=\$(cat /sys/kernel/debug/rknpu/version)
    echo npu_freq=\$(cat /sys/kernel/debug/rknpu/freq); } > environment.log" \
  > "$LOCAL/remote-build.log" 2>&1 || status=1

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a PROFILE_LIST <<< "$PROFILES"
for ((round = 1; round <= ROUNDS; round++)); do
  for profile in "${PROFILE_LIST[@]}"; do
    model=${profile%%:*}
    batch=${profile##*:}
    for arm in "${ARM_LIST[@]}"; do
      fast=1
      [[ "$arm" == "generic" ]] && fast=0
      rc=0
      "${SSH[@]}" "cd '$REMOTE'
        RKNPU_CORE_MASK='$CORE_MASK' RK_PLAN_UPLOAD_COALESCE='1' RK_PLAN_UPLOAD_FAST_PACK='$fast' \
        PYTHONPATH=\$PWD/tilelang-rknpu/python \
        taskset -c '$TIME_CPU' python3 campaign/tools/profile_bridge_envelope.py \
          --backend board --lib \$PWD/build/librknpu_plan.so --fd 0 \
          --model '$model' --batch '$batch' \
          --plans-dir campaign/plans --weights-dir campaign/weights \
          --data-dir campaign/data --runs '$RUNS' --warmup '$WARMUP' \
          --label '$arm:r$round' \
          --out 'out-$arm/${arm}_${model}_b${batch}_r${round}.json'" \
        >> "$LOCAL/$arm.measure.log" 2>&1 || rc=$?
      [[ "$rc" == 0 ]] || { echo "measure failed: $arm r$round" >&2; status=1; }
    done
  done
done

for arm in "${ARM_LIST[@]}"; do
  "${SCP[@]}" -r "$TARGET:$REMOTE/out-$arm" "$LOCAL/out-$arm" >/dev/null 2>&1 || status=1
done
"${SCP[@]}" "$TARGET:$REMOTE/environment.log" "$TARGET:$REMOTE/library.sha256" \
  "$LOCAL/" >/dev/null 2>&1 || true

if ! grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
else
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
fi
unset RK_BOARD_PASSWORD

if ! python3 "$HERE/tools/compare_bridge_envelope.py" --root "$LOCAL" \
    --arms "$ARMS" --out "$LOCAL/comparison.json"; then
  status=1
fi

printf 'RK3588 upload-layout campaign: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
