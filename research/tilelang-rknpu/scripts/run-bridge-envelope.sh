#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# RK3588 bridge-envelope campaign: build a baseline arm and the working-tree arm
# from the same payload, profile both on the same board, and diff the envelopes.
#
#   baseline  = the committed (HEAD) source tree, i.e. no envelope attribution
#   candidate = the working tree, i.e. the build under evaluation
#
# Only the model payload (plans/weights/data) is shared; each arm uses its own
# C runtime and its own Python package so the comparison is end to end.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }

ARMS=${RK_ENVELOPE_ARMS:-baseline,candidate,noprobe}
PROFILES=${RK_ENVELOPE_PROFILES:-tiny_vit:8,tiny_vit:1}
# Per-round sample count and number of interleaved rounds.  Arms are measured in
# turn every round, so slow machine phases cannot systematically land on one arm.
RUNS=${RK_ENVELOPE_RUNS:-32}
ROUNDS=${RK_ENVELOPE_ROUNDS:-4}
WARMUP=${RK_ENVELOPE_WARMUP:-8}
CORE_MASK=${RKNPU_CORE_MASK:-1}
TIME_CPU=${RK_TIME_CPU:-6}

# arm -> (source tree, cmake target, library file)
arm_source() {
  case "$1" in
    baseline) echo baseline ;;
    candidate|noprobe) echo candidate ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}
arm_target() {
  case "$1" in
    baseline|candidate) echo rknpu_plan ;;
    noprobe) echo rknpu_plan_noprobe ;;
    *) return 1 ;;
  esac
}
arm_library() {
  case "$1" in
    baseline|candidate) echo librknpu_plan.so ;;
    noprobe) echo librknpu_plan_noprobe.so ;;
    *) return 1 ;;
  esac
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-envelope-$STAMP"
LOCAL="$HERE/results/$STAMP-envelope"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL"
CAMPAIGN="$LOCAL/campaign"
mkdir -p "$CAMPAIGN"/{plans,weights,data,tools}

cp "$HERE"/models/compiled/*.rkplan.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$CAMPAIGN/plans/"
cp "$HERE"/models/compiled/manifest.json "$CAMPAIGN/plans/manifest.json"
cp "$HERE"/models/tiny_vit.npz "$HERE"/models/mnist_mlp.npz "$CAMPAIGN/weights/"
cp "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz "$CAMPAIGN/data/"
cp "$HERE"/.data/mnist/t10k-labels-idx1-ubyte.gz "$CAMPAIGN/data/"
# The profiler always comes from the working tree: it can measure a baseline
# library because absent attribution fields read as zero.
cp "$HERE"/tools/profile_bridge_envelope.py "$CAMPAIGN/tools/"

python3 - "$HERE" "$CAMPAIGN" "$ARMS" "$PROFILES" "$RUNS" "$WARMUP" "$CORE_MASK" \
  "$ROUNDS" "$TIME_CPU" <<'PY'
import hashlib, json, sys
from pathlib import Path
here, camp = Path(sys.argv[1]), Path(sys.argv[2])
arms, profiles = sys.argv[3].split(','), sys.argv[4].split(',')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
manifest = {
    'schema_version': 1,
    'kind': 'bridge-envelope-campaign',
    'arms': arms,
    'profiles': [{'model': p.split(':')[0], 'batch': int(p.split(':')[1])}
                 for p in profiles],
    'runs_per_round': int(sys.argv[5]),
    'warmup': int(sys.argv[6]),
    'core_mask': int(sys.argv[7]),
    'rounds': int(sys.argv[8]),
    'cpu_affinity_pin': int(sys.argv[9]),
    'interleaved': True,
    'source_hashes': {
        'python/rknpu_backend/models.py': sha(here / 'python/rknpu_backend/models.py'),
        'python/rknpu_backend/hybrid.py': sha(here / 'python/rknpu_backend/hybrid.py'),
        'python/rknpu_backend/runtime/ffi.py': sha(here / 'python/rknpu_backend/runtime/ffi.py'),
        'tools/profile_bridge_envelope.py': sha(here / 'tools/profile_bridge_envelope.py'),
        'custom-ops/src/rk_plan_runtime.c': sha(here / '../custom-ops/src/rk_plan_runtime.c'),
        'custom-ops/include/rk_plan.h': sha(here / '../custom-ops/include/rk_plan.h'),
        'models/compiled/manifest.json': sha(here / 'models/compiled/manifest.json'),
    },
    'weights': {p.name: sha(p) for p in sorted((camp / 'weights').glob('*.npz'))},
    'data': {p.name: sha(p) for p in sorted((camp / 'data').glob('*.gz'))},
    'payload': {'plans_dir': 'plans', 'weights_dir': 'weights',
                'data_dir': 'data', 'tools_dir': 'tools'},
}
(camp / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(sha(camp / 'manifest.json'))
PY
(cd "$CAMPAIGN" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

# ---- arm source trees -----------------------------------------------------
STAGE=$(mktemp -d /tmp/rk-envelope-XXXXXX)
askpass=$(mktemp /tmp/rk-envelope-askpass-XXXXXX)
trap 'rm -rf -- "$STAGE"; rm -f -- "$askpass" "$STAGE.tgz"' EXIT

prepare_common() {
  local root=$1
  mkdir -p "$root"
  tar --exclude='build*' --exclude='artifacts' --exclude='results' \
      --exclude='__pycache__' --exclude='*.so' --exclude='.pydeps*' \
      --exclude='*.tgz' -cf - -C "$REPO/research" \
      custom-ops upstream-components/rocket-userspace \
      upstream-components/rknpu-submit | tar -x -C "$root"
}

prepare_baseline() {
  local root=$1
  prepare_common "$root"
  rm -rf "$root/custom-ops"
  mkdir -p "$root/tilelang-rknpu"
  # HEAD's committed source for the runtime, Python package and payload layout.
  (cd "$REPO" && git archive --format=tar HEAD research/custom-ops \
      research/tilelang-rknpu/python research/tilelang-rknpu/models \
      research/tilelang-rknpu/tools) | tar -x -C "$root" --strip-components=1
  mkdir -p "$root/tilelang-rknpu/scripts"
  cp "$HERE/scripts/run-bridge-envelope-native.sh" "$root/tilelang-rknpu/scripts/"
  # The profiler always comes from the working tree.
  mkdir -p "$root/tilelang-rknpu/tools"
  cp "$HERE/tools/profile_bridge_envelope.py" "$root/tilelang-rknpu/tools/"
}

prepare_candidate() {
  local root=$1
  prepare_common "$root"
  mkdir -p "$root/tilelang-rknpu"
  tar --exclude='__pycache__' --exclude='.pydeps*' --exclude='*.npz' \
      --exclude='*.npy' -cf - -C "$REPO/research/tilelang-rknpu" \
      python tools scripts | tar -x -C "$root/tilelang-rknpu"
}

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
# Two arms can share one source tree (candidate and the probe-free control only
# differ in the build target), so stage each distinct source exactly once.
SOURCE_ARMS=()
for arm in "${ARM_LIST[@]}"; do
  src=$(arm_source "$arm") || exit 2
  arm_target "$arm" > /dev/null || exit 2
  arm_library "$arm" > /dev/null || exit 2
  [[ " ${SOURCE_ARMS[*]} " == *" $src "* ]] || SOURCE_ARMS+=("$src")
done
for src in "${SOURCE_ARMS[@]}"; do
  case "$src" in
    baseline) prepare_baseline "$STAGE/$src" ;;
    candidate) prepare_candidate "$STAGE/$src" ;;
    *) echo "unknown source arm $src" >&2; exit 2 ;;
  esac
  (cd "$STAGE" && tar -czf "$STAGE.$src.tgz" "$src")
done

# ---- upload ---------------------------------------------------------------
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

TAR_CAMPAIGN="$STAGE.campaign.tgz"
tar -czf "$TAR_CAMPAIGN" -C "$LOCAL" campaign
"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$TAR_CAMPAIGN" "$TARGET:$REMOTE/campaign.tgz" >/dev/null
for src in "${SOURCE_ARMS[@]}"; do
  "${SCP[@]}" "$STAGE.$src.tgz" "$TARGET:$REMOTE/$src.tgz" >/dev/null 2>&1 || status=1
done

status=0
"${SSH[@]}" "set -e; cd '$REMOTE'; tar -xzf campaign.tgz; for arm in ${SOURCE_ARMS[*]}; do tar -xzf \$arm.tgz; done" \
  > "$LOCAL/extract.log" 2>&1 || status=1

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a PROFILE_LIST <<< "$PROFILES"
status=0

# Build every arm before any measurement so build-time load cannot bias one arm.
for arm in "${ARM_LIST[@]}"; do
  srcarm=$(arm_source "$arm") || exit 2
  target=$(arm_target "$arm") || exit 2
  rc=0
  "${SSH[@]}" "bash '$REMOTE/$srcarm/tilelang-rknpu/scripts/run-bridge-envelope-native.sh' \
      build '$arm' '$target' '$REMOTE/build-$srcarm' '$REMOTE/out-$arm'" \
      > "$LOCAL/$arm.build.log" 2>&1 || rc=$?
  [[ "$rc" == 0 ]] || { echo "build failed: $arm" >&2; status=1; }
done
[[ "$status" == 0 ]] || exit 1

# Primitive cost floor on the candidate build (arm independent: custom_ops.c is
# identical in every arm).
"${SSH[@]}" "bash '$REMOTE/candidate/tilelang-rknpu/scripts/run-bridge-envelope-native.sh' \
    primitives '$REMOTE/build-candidate' '$REMOTE/out-candidate'" \
    > "$LOCAL/primitives.pull.log" 2>&1 || true

# Interleaved rounds: every arm measures the same profile inside the same round.
for ((round = 1; round <= ROUNDS; round++)); do
  for profile in "${PROFILE_LIST[@]}"; do
    model=${profile%%:*}
    batch=${profile##*:}
    for arm in "${ARM_LIST[@]}"; do
      srcarm=$(arm_source "$arm") || exit 2
      libname=$(arm_library "$arm") || exit 2
      rc=0
      "${SSH[@]}" "RK_TIME_CPU='$TIME_CPU' RKNPU_CORE_MASK='$CORE_MASK' \
          RK_ENVELOPE_WARMUP='$WARMUP' \
          bash '$REMOTE/$srcarm/tilelang-rknpu/scripts/run-bridge-envelope-native.sh' \
          measure '$arm' '$REMOTE/build-$srcarm/$libname' '$model' '$batch' \
          '$round' '$RUNS' '$REMOTE/campaign' '$REMOTE/out-$arm'" \
          >> "$LOCAL/$arm.measure.log" 2>&1 || rc=$?
      [[ "$rc" == 0 ]] || { echo "measure failed: $arm r$round" >&2; status=1; }
    done
  done
done

for arm in "${ARM_LIST[@]}"; do
  "${SCP[@]}" -r "$TARGET:$REMOTE/out-$arm" "$LOCAL/out-$arm" >/dev/null 2>&1 || status=1
done

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

printf 'RK3588 bridge-envelope campaign: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
