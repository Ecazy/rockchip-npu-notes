#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# RK3588 artifact smoke test for the codegen refactor.
#
# Scope: single MatMul, MatMul->SiLU, ragged K=257 and a branch/multi-output
# kernel, all compiled on the Host from real TileLang PrimFuncs, then executed on
# the board through the new artifact API (compile -> explain -> instantiate -> run),
# an A->B->A input update on one workspace, two independent instantiations, and the
# export -> load_plan -> run replay path.
#
# Deliberately *not* done here: no performance campaign, no frequency change, no
# device reset, no multi-core masks (single core only), no tolerance change.
#
# Credentials are read from the ignored repository .env and never printed, placed
# in argv, or copied into the bundle/results.  The remote runs in a fresh isolated
# directory and any device failure stops the run.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)

if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$REMOTE_BASE" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'unsafe remote workspace' >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE="$REMOTE_BASE/tl-artifact-smoke-$STAMP"
LOCAL="$HERE/results/$STAMP-artifact-smoke"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL"/{campaign,board,logs}
CAMPAIGN="$LOCAL/campaign"
LOG="$LOCAL/logs"
MASKS=${RK_SMOKE_CORE_MASK:-1}

status_file="$LOCAL/.status"
: >"$status_file"
record() { printf '%s\t%s\n' "$1" "$2" >>"$status_file"; }
note() { echo "== $*" >&2; }

DEPS=${TILELANG_DEPS_DIR:-$HERE/.pydeps}
export PYTHONPATH="$HERE/python:$HERE:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HERE/.cache}"
export RKNPU_CHIP=rk3588

# --- 1. Host: compile real TileLang kernels and build the case data ----------
note "building the artifact smoke campaign (real TileLang -> artifact -> RKPlan)"
python3 "$HERE/tools/build_artifact_smoke_campaign.py" --out-dir "$CAMPAIGN" \
  >"$LOG/build-campaign.log" 2>&1
campaign_rc=$?
record build_campaign "$campaign_rc"
if (( campaign_rc != 0 )); then
  note "campaign build failed; see $LOG/build-campaign.log"
  tail -20 "$LOG/build-campaign.log" >&2
  echo 'artifact smoke: FAIL (campaign build)' >&2
  exit 1
fi

# --- 2. bundle and transfer ---------------------------------------------------
askpass=$(mktemp /tmp/rk-smoke-askpass-XXXXXX)
source_bundle=$(mktemp /tmp/rk-smoke-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/rk-smoke-campaign-XXXXXX.tgz)
cleanup() { rm -f -- "$askpass" "$source_bundle" "$campaign_bundle"; }
trap cleanup EXIT
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
chmod 700 "$askpass"
export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1)
SSH=(setsid -w ssh "${SSH_OPTS[@]}" "$TARGET")
SCP=(setsid -w scp "${SSH_OPTS[@]}")

tar --exclude='.git' --exclude='build*' --exclude='results' --exclude='artifacts' \
    --exclude='*.tgz' --exclude='__pycache__' -czf "$source_bundle" \
    -C "$REPO/research" \
    custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit \
    tilelang-rknpu/python tilelang-rknpu/capabilities \
    tilelang-rknpu/tools/board_artifact_smoke.py
record bundle_source $?
tar -czf "$campaign_bundle" -C "$LOCAL" campaign
record bundle_campaign $?
sha256sum "$source_bundle" >"$LOCAL/source-bundle.sha256"
sha256sum "$campaign_bundle" >"$LOCAL/campaign-bundle.sha256"
cp -- "$campaign_bundle" "$LOCAL/campaign-bundle.tgz"

note "creating isolated remote directory"
"${SSH[@]}" "mkdir -p '$REMOTE'" >"$LOG/remote-mkdir.log" 2>&1
record remote_mkdir $?
"${SCP[@]}" "$source_bundle" "$TARGET:$REMOTE/source.tgz" >"$LOG/scp-source.log" 2>&1
record scp_source $?
"${SCP[@]}" "$campaign_bundle" "$TARGET:$REMOTE/campaign.tgz" >"$LOG/scp-campaign.log" 2>&1
record scp_campaign $?
"${SSH[@]}" "tar -xzf '$REMOTE/source.tgz' -C '$REMOTE' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE'" \
  >"$LOG/remote-extract.log" 2>&1
record remote_extract $?

# --- 3. remote: build the production library, then run one case at a time -----
note "building librknpu_plan.so on the board and running the smoke cases (core mask $MASKS)"
"${SSH[@]}" "bash -s" >"$LOG/remote-run.log" 2>&1 <<REMOTE_SCRIPT
set -uo pipefail
REMOTE='$REMOTE'
MASKS='$MASKS'
export RKNPU_CHIP=rk3588
export ROCKET_CHIP=rk3588
export PYTHONPATH="\$REMOTE/tilelang-rknpu/python"
mkdir -p "\$REMOTE/build" "\$REMOTE/results"
{
  echo "collected_utc=\$(date -u +%FT%TZ)"
  uname -srm
  echo "board_model=\$(tr -d '\\0' </proc/device-tree/model 2>/dev/null)"
  echo "driver=\$(cat /sys/kernel/debug/rknpu/version 2>/dev/null)"
  echo "npu_freq=\$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null)"
  echo "npu_load=\$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)"
  echo "python=\$(python3 -V 2>&1)"
  echo "core_mask=\$MASKS"
} > "\$REMOTE/results/environment.log"

cmake -S "\$REMOTE/custom-ops" -B "\$REMOTE/build" \\
      -DUPSTREAM_ROOT="\$REMOTE/upstream-components" -DCMAKE_BUILD_TYPE=Release \\
      > "\$REMOTE/results/configure.log" 2>&1
cfg=\$?
cmake --build "\$REMOTE/build" --target rknpu_plan -j4 > "\$REMOTE/results/build.log" 2>&1
bld=\$?
echo "configure=\$cfg build=\$bld" >> "\$REMOTE/results/environment.log"
if (( cfg != 0 || bld != 0 )); then
  tail -30 "\$REMOTE/results/configure.log" "\$REMOTE/results/build.log" >&2
  exit 1
fi
export RKNPU_PLAN_LIB="\$REMOTE/build/librknpu_plan.so"
[[ -f "\$RKNPU_PLAN_LIB" ]] || { echo 'librknpu_plan.so missing' >&2; exit 1; }
sha256sum "\$RKNPU_PLAN_LIB" > "\$REMOTE/results/library.sha256"

overall=0
for mask in \$MASKS; do
  out="\$REMOTE/results/core-\$mask"
  mkdir -p "\$out"
  dmesg 2>/dev/null | tail -n 60 > "\$REMOTE/results/kernel-before.log" || true
  RKNPU_CORE_MASK="\$mask" LD_LIBRARY_PATH="\$REMOTE/build\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" \\
    python3 "\$REMOTE/tilelang-rknpu/tools/board_artifact_smoke.py" \\
      --campaign "\$REMOTE/campaign" --out "\$out" > "\$out/runner.log" 2>&1
  rc=\$?
  echo "\$rc" > "\$out/exit.txt"
  dmesg 2>/dev/null | tail -n 120 > "\$REMOTE/results/kernel-after.log" || true
  if (( rc != 0 )); then
    overall=1
    # A submission fault must not be followed by another submission.
    echo "smoke run failed (rc=\$rc); stopping before any further case" >&2
    tail -30 "\$out/runner.log" >&2
    break
  fi
done
exit "\$overall"
REMOTE_SCRIPT
remote_rc=$?
record remote_run "$remote_rc"
cat "$LOG/remote-run.log" >&2

note "fetching raw board outputs"
"${SCP[@]}" -r "$TARGET:$REMOTE/results/." "$LOCAL/board/" >"$LOG/scp-results.log" 2>&1
record scp_results $?

# --- 4. Host: independent verification ---------------------------------------
note "verifying raw board outputs on the HOST"
python3 "$HERE/tools/verify_artifact_smoke.py" \
  --campaign "$CAMPAIGN" --board "$LOCAL/board" --out "$LOCAL/verify.json" \
  >"$LOG/verify.log" 2>&1
verify_rc=$?
record verify "$verify_rc"
cat "$LOG/verify.log" >&2

# --- 5. credential scan -------------------------------------------------------
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" >/dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  record credential_scan 1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
  record credential_scan 0
fi
unset RK_BOARD_PASSWORD

python3 - "$LOCAL" "$REMOTE" "$status_file" <<'PY'
import json
import sys
from pathlib import Path

local = Path(sys.argv[1])
remote = sys.argv[2]
commands = {}
for line in Path(sys.argv[3]).read_text().splitlines():
    if not line.strip():
        continue
    step, code = line.split("\t")
    commands[step] = {"exit_code": int(code)}
manifest = json.loads((local / "campaign" / "manifest.json").read_text())
summary = {
    "kind": "artifact-smoke-evidence",
    "created_utc": local.name.split("-")[0],
    "remote_dir": remote,
    "core_mask": "1",
    "scope": [
        "single MatMul, MatMul->SiLU, ragged K=257, branch/multi-output",
        "all compiled on the Host from real TileLang PrimFuncs",
        "artifact compile -> explain -> instantiate -> run on the board",
        "A -> B -> A input update on one workspace",
        "two independent instantiations",
        "export -> load_plan -> run replay",
        "host-side numeric verification against the plan's declared tolerance",
    ],
    "explicitly_not_done": [
        "no performance campaign, no frequency change, no device reset",
        "no multi-core masks (single core only)",
        "no tolerance relaxation",
        "no TileLang on the board (absent); TileLang normalization ran on the Host",
    ],
    "commands": commands,
    "cases": [case["name"] for case in manifest["cases"]],
}
(local / "manifest.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v["exit_code"] for k, v in commands.items()}, sort_keys=True))
PY
manifest_rc=$?
record manifest "$manifest_rc"

# The status file is the only record of every step, and it includes the manifest
# step itself.  Read it *before* deleting it -- and never fall back to re-parsing
# manifest.json, whose writer is exactly the step that may have just failed.
failed_steps=""
step_count=0
seen_steps=" "
while IFS=$'\t' read -r step code; do
  [[ -n "$step" ]] || continue
  step_count=$((step_count + 1))
  seen_steps+="$step "
  if (( code != 0 )); then failed_steps+="$step($code) "; fi
done < "$status_file"
rm -f "$status_file"

# Belt and braces: even if the status file were truncated, the statuses that matter
# for acceptance are part of the final decision in their own right.
add_failure() {  # <step> <rc>
  if [[ "$seen_steps" != *" $1 "* ]]; then failed_steps+="$1($2) "; fi
}
if (( manifest_rc != 0 )); then add_failure manifest "$manifest_rc"; fi
if (( remote_rc != 0 )); then add_failure remote_run "$remote_rc"; fi
if (( verify_rc != 0 )); then add_failure verify "$verify_rc"; fi
if (( step_count == 0 )); then failed_steps+="no-steps-recorded "; fi

if [[ -n "$failed_steps" ]]; then
  echo "artifact smoke: FAIL (${failed_steps% })" >&2
  echo "evidence: $LOCAL" >&2
  exit 1
fi
echo "artifact smoke: PASS" >&2
echo "evidence: $LOCAL" >&2
exit 0
