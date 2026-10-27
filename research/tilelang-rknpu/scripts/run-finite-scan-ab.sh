#!/usr/bin/env bash
# Same-source RK3588 A/B for a compile-time isolated runtime helper change.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }
CPU=${RK_TIME_CPU:-6}
CORE_MASK=${RKNPU_CORE_MASK:-1}
EXPERIMENT=${RK_AB_EXPERIMENT:-finite-scan}
REFERENCE_MACRO=${RK_AB_REFERENCE_MACRO:-RK_PLAN_FINITE_SCAN_REFERENCE}
[[ "$EXPERIMENT" =~ ^[a-z0-9-]+$ && "$REFERENCE_MACRO" =~ ^[A-Z][A-Z0-9_]*$ ]] || {
  echo 'invalid experiment label or reference macro' >&2; exit 2; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL="$HERE/results/${STAMP}-${EXPERIMENT}"
REMOTE="$REMOTE_BASE/tl-${EXPERIMENT}-$STAMP"
[[ ! -e "$LOCAL" ]] || { echo 'fresh result path required' >&2; exit 2; }
mkdir -p "$LOCAL"/{campaign,cases,rounds,profile,full-model,artifacts}
touch "$LOCAL/commands.log"
mkdir -p "$LOCAL/host"
if [[ "$EXPERIMENT" == finite-scan ]]; then
  for evidence in /tmp/rknpu-finite-scan-host-build.log /tmp/rknpu-finite-scan-pytest.log \
                  /tmp/rknpu-finite-scan-pytest.xml /tmp/rknpu-finite-scan-ctest.log; do
    [[ ! -f "$evidence" ]] || cp "$evidence" "$LOCAL/host/"
  done
elif [[ "$EXPERIMENT" == output-sync ]]; then
  shopt -s nullglob
  for evidence in /tmp/rknpu-output-sync-*; do
    [[ ! -f "$evidence" ]] || cp "$evidence" "$LOCAL/host/"
  done
fi
STAGE=$(mktemp -d /tmp/rk-finite-scan-XXXXXX)
askpass=$(mktemp /tmp/rk-finite-scan-askpass-XXXXXX)
cleanup() { rm -rf -- "$STAGE"; rm -f -- "$askpass"; }
trap cleanup EXIT

# Build reproducible QK/PV input bundles from the frozen package.
PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/runtime_layers/build_cases.py" \
  --out "$LOCAL/cases" > "$LOCAL/case-build.stdout" 2> "$LOCAL/case-build.stderr"
rm -rf "$LOCAL/cases/mlp_b1" "$LOCAL/cases/mlp_b32" "$LOCAL/cases/qkv_b8"
cp "$HERE/models/compiled/manifest.json" "$LOCAL/campaign/manifest.json"
cp "$HERE/models/compiled/tiny_vit_b8.rkhybrid.json" "$LOCAL/campaign/"
cp "$HERE/models/compiled/"*.rkplan.json "$LOCAL/campaign/"
cp "$HERE/models/compiled/"*.rkhybrid.json "$LOCAL/campaign/"
cp "$HERE/models/tiny_vit.npz" "$LOCAL/campaign/"
cp "$HERE/.data/mnist/t10k-images-idx3-ubyte.gz" "$LOCAL/campaign/"
cp "$HERE/tools/runtime_layers/benchmark_python_layers.py" "$LOCAL/campaign/"
cp "$HERE/tools/runtime_layers/full_model_finite_scan_ab.py" "$LOCAL/campaign/"
mkdir -p "$LOCAL/campaign/cases"
cp -a "$LOCAL/cases/qk_b8" "$LOCAL/campaign/cases/"
cp -a "$LOCAL/cases/pv_b8" "$LOCAL/campaign/cases/"

python3 - "$HERE" "$LOCAL" "$CPU" "$CORE_MASK" "$EXPERIMENT" "$REFERENCE_MACRO" <<'PY'
import hashlib,json,sys
from pathlib import Path
here,out=Path(sys.argv[1]),Path(sys.argv[2])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=['../custom-ops/src/rk_plan_runtime.c','../custom-ops/src/rk_plan_download.h',
       '../custom-ops/src/custom_ops.c','../custom-ops/CMakeLists.txt',
       '../custom-ops/tests/c/output_sync_test.c',
       'tools/runtime_layers/runtime_layer_probe.c','scripts/run-finite-scan-ab.sh',
       'scripts/run-finite-scan-ab-native.sh',
       'tools/runtime_layers/benchmark_python_layers.py',
       'tools/runtime_layers/full_model_finite_scan_ab.py',
       'tools/runtime_layers/analyze_finite_scan_ab.py',
       'models/compiled/manifest.json','models/compiled/tiny_vit_b8.rkhybrid.json',
       'models/tiny_vit.npz']
manifest={'kind':sys.argv[5]+'-ab','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
 'source_hashes':{x:sha((here/x).resolve()) for x in files if (here/x).exists()},
 'cpu_pin':int(sys.argv[3]),'npu_core_mask':int(sys.argv[4]),
 'prepared_validation':0,'rounds':7,'cases':['qk_b8','pv_b8'],
 'reference_macro':sys.argv[6]}
(out/'campaign'/'run-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
(cd "$LOCAL/campaign" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$LOCAL/campaign/SHA256SUMS")

# Deploy a source snapshot and exact campaign inputs.
mkdir -p "$STAGE/source"
tar --exclude='build*' --exclude='results' --exclude='artifacts' --exclude='__pycache__' \
  -czf "$STAGE/source.tgz" -C "$REPO/research" \
  custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit \
  tilelang-rknpu/python tilelang-rknpu/tools tilelang-rknpu/scripts/run-finite-scan-ab-native.sh
tar -czf "$STAGE/campaign.tgz" -C "$LOCAL" campaign
sha256sum "$STAGE/source.tgz" "$STAGE/campaign.tgz" > "$LOCAL/bundle-hashes.txt"
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
"${SSH[@]}" "mkdir -p '$REMOTE/source' && tar -xzf '$REMOTE/source.tgz' -C '$REMOTE/source' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE'" > "$LOCAL/deploy.stdout" 2> "$LOCAL/deploy.stderr"

# Board-side build wrapper records exact status, stdout, stderr and library hashes.
build_cmd="bash '$REMOTE/source/tilelang-rknpu/scripts/run-finite-scan-ab-native.sh' build '$REMOTE/source' '$REMOTE/campaign' '$REMOTE/build-fast' '$REMOTE/build-reference' '$CPU' '$REFERENCE_MACRO'"
printf '%s\n' "SSH $TARGET $build_cmd" >> "$LOCAL/commands.log"
"${SSH[@]}" "$build_cmd" > "$LOCAL/build.stdout" 2> "$LOCAL/build.stderr" || BUILD_RC=$?
BUILD_RC=${BUILD_RC:-0}; echo "$BUILD_RC" > "$LOCAL/build.exit"
if (( BUILD_RC != 0 )); then
  "${SCP[@]}" -r "$TARGET:$REMOTE/source/build-logs" "$LOCAL/" >/dev/null 2>&1 || true
  exit "$BUILD_RC"
fi
mkdir -p "$LOCAL/build-logs"
"${SCP[@]}" -r "$TARGET:$REMOTE/source/build-logs/." "$LOCAL/build-logs/" >/dev/null
"${SCP[@]}" -r "$TARGET:$REMOTE/source/artifacts/." "$LOCAL/artifacts/" >/dev/null
"${SCP[@]}" "$TARGET:$REMOTE/source/environment.log" "$LOCAL/board-environment.log" >/dev/null

status=0
for round in {1..7}; do
  for case in qk_b8 pv_b8; do
    if (( round % 2 )); then arms=(fast reference); else arms=(reference fast); fi
    for arm in "${arms[@]}"; do
      rc=0
      measure_cmd="bash '$REMOTE/source/tilelang-rknpu/scripts/run-finite-scan-ab-native.sh' isolated '$REMOTE/source' '$REMOTE/campaign' '$REMOTE/build-fast' '$REMOTE/build-reference' '$case' '$arm' '$round' '$CPU' '$CORE_MASK' '$REMOTE/out'"
      printf '%s\n' "SSH $TARGET $measure_cmd" >> "$LOCAL/commands.log"
      "${SSH[@]}" "$measure_cmd" \
        > "$LOCAL/rounds/${case}-r${round}-${arm}.stdout" 2> "$LOCAL/rounds/${case}-r${round}-${arm}.stderr" || rc=$?
      echo "$rc" > "$LOCAL/rounds/${case}-r${round}-${arm}.exit"
      if (( rc != 0 )); then
        status=1
        # Do not launch more device work after a failed probe. Still collect
        # the completed logs below so the failure remains reviewable.
        break 3
      fi
    done
  done
done
"${SCP[@]}" -r "$TARGET:$REMOTE/out/." "$LOCAL/rounds/" >/dev/null 2>&1 || status=1

# Attribution profile is diagnostic; Host tests establish the BO call counts.
if (( status == 0 )); then
  for case in qk_b8 pv_b8; do
    if [[ "$case" == qk_b8 ]]; then arms=(fast reference); else arms=(reference fast); fi
    for arm in "${arms[@]}"; do
      rc=0
      profile_cmd="bash '$REMOTE/source/tilelang-rknpu/scripts/run-finite-scan-ab-native.sh' profile '$REMOTE/source' '$REMOTE/campaign' '$REMOTE/build-fast' '$REMOTE/build-reference' '$case' '$arm' '$CPU' '$CORE_MASK' '$REMOTE/out-profile'"
      printf '%s\n' "SSH $TARGET $profile_cmd" >> "$LOCAL/commands.log"
      "${SSH[@]}" "$profile_cmd" \
        > "$LOCAL/profile/${case}-${arm}.stdout" 2> "$LOCAL/profile/${case}-${arm}.stderr" || rc=$?
      echo "$rc" > "$LOCAL/profile/${case}-${arm}.exit"
      if (( rc != 0 )); then status=1; break 3; fi
    done
  done
fi
"${SCP[@]}" -r "$TARGET:$REMOTE/out-profile/." "$LOCAL/profile/" >/dev/null 2>&1 || status=1

# Full-model b8/b1 are run only after isolated QK/PV show a stable positive gain.
if (( status == 0 )) && python3 "$HERE/tools/runtime_layers/analyze_finite_scan_ab.py" --experiment "$EXPERIMENT" --root "$LOCAL/rounds" --out "$LOCAL/isolated-summary.json"; then
  gain=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(all(d["cases"][k]["ratio_reference_over_fast_median"]>1 for k in d["cases"])))' "$LOCAL/isolated-summary.json")
else status=1; gain=0; fi
if (( status == 0 && gain == 1 )); then
  for batch in 8; do
    rc=0
    full_cmd="bash '$REMOTE/source/tilelang-rknpu/scripts/run-finite-scan-ab-native.sh' full '$REMOTE/source' '$REMOTE/campaign' '$REMOTE/build-fast' '$REMOTE/build-reference' '$batch' '$CPU' '$CORE_MASK' '$REMOTE/out-full' '$EXPERIMENT'"
    printf '%s\n' "SSH $TARGET $full_cmd" >> "$LOCAL/commands.log"
    "${SSH[@]}" "$full_cmd" \
      > "$LOCAL/full-model/b${batch}.stdout" 2> "$LOCAL/full-model/b${batch}.stderr" || rc=$?
    echo "$rc" > "$LOCAL/full-model/b${batch}.exit"
    (( rc == 0 )) || status=1
    "${SCP[@]}" -r "$TARGET:$REMOTE/out-full/b${batch}/." "$LOCAL/full-model/b${batch}/" >/dev/null 2>&1 || status=1
  done
  model_gain=$(python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(int(d["round_ratio_median"]>1 and d["bitwise"]))' "$LOCAL/full-model/b8/result.json" 2>/dev/null || echo 0)
  if (( status == 0 && model_gain == 1 )); then
    batch=1; rc=0
    full_cmd="bash '$REMOTE/source/tilelang-rknpu/scripts/run-finite-scan-ab-native.sh' full '$REMOTE/source' '$REMOTE/campaign' '$REMOTE/build-fast' '$REMOTE/build-reference' '$batch' '$CPU' '$CORE_MASK' '$REMOTE/out-full' '$EXPERIMENT'"
    printf '%s\n' "SSH $TARGET $full_cmd" >> "$LOCAL/commands.log"
    "${SSH[@]}" "$full_cmd" > "$LOCAL/full-model/b${batch}.stdout" 2> "$LOCAL/full-model/b${batch}.stderr" || rc=$?
    echo "$rc" > "$LOCAL/full-model/b${batch}.exit"
    (( rc == 0 )) || status=1
    "${SCP[@]}" -r "$TARGET:$REMOTE/out-full/b${batch}/." "$LOCAL/full-model/b${batch}/" >/dev/null 2>&1 || status=1
  fi
fi
unset RK_BOARD_PASSWORD
echo "$status" > "$LOCAL/status.exit"
echo "RK3588 $EXPERIMENT A/B status=$status results=$LOCAL"
exit "$status"
