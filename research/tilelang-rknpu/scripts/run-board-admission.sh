#!/usr/bin/env bash
# RK3588 admission campaign: Host-generated plans -> remote build -> three cores.
# Credentials are read from the ignored repository .env and are never printed,
# placed in argv, or copied into the bundle/results.
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
REMOTE="$REMOTE_BASE/tl-admission-$STAMP"
LOCAL="$HERE/results/$STAMP-rk3588"
[[ ! -e "$LOCAL" ]] || { echo 'fresh results path required' >&2; exit 2; }
mkdir -p "$LOCAL/board"
CAMPAIGN="$LOCAL/campaign"

DEPS_DIR=${TILELANG_DEPS_DIR:-$HERE/.pydeps-p0}
VENV=${TILELANG_VENV_DIR:-$HERE/.venv-p0}
PY=python3
[[ -x "$VENV/bin/python" ]] && PY="$VENV/bin/python"
export PYTHONPATH="$HERE/python:$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DEPS_DIR/z3/lib:$DEPS_DIR/tilelang/lib:$DEPS_DIR/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TILELANG_CACHE_DIR="$HERE/.cache"
"$PY" "$HERE/tools/make_board_campaign.py" --out-dir "$CAMPAIGN" > "$LOCAL/make-campaign.log" 2>&1

askpass=$(mktemp /tmp/rk-board-askpass-XXXXXX)
source_bundle=$(mktemp /tmp/rk-board-source-XXXXXX.tgz)
campaign_bundle=$(mktemp /tmp/rk-board-campaign-XXXXXX.tgz)
cleanup() {
  rm -f -- "$askpass" "$source_bundle" "$campaign_bundle"
}
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
    tilelang-rknpu/scripts/run-board-admission-native.sh
tar -czf "$campaign_bundle" -C "$LOCAL" campaign
cp -- "$source_bundle" "$LOCAL/source-bundle.tgz"
sha256sum "$LOCAL/source-bundle.tgz" > "$LOCAL/source-bundle.sha256"

"${SSH[@]}" "mkdir -p '$REMOTE'"
"${SCP[@]}" "$source_bundle" "$TARGET:$REMOTE/source.tgz"
"${SCP[@]}" "$campaign_bundle" "$TARGET:$REMOTE/campaign.tgz"
"${SSH[@]}" "tar -xzf '$REMOTE/source.tgz' -C '$REMOTE' && tar -xzf '$REMOTE/campaign.tgz' -C '$REMOTE'"

status=0
"${SSH[@]}" "bash '$REMOTE/tilelang-rknpu/scripts/run-board-admission-native.sh' '$REMOTE/campaign' '$REMOTE/build' '$REMOTE/results'" > "$LOCAL/remote.log" 2>&1 || status=$?
mkdir -p "$LOCAL/board"
"${SCP[@]}" -r "$TARGET:$REMOTE/results/." "$LOCAL/board/" || status=1
if "$PY" "$HERE/tools/verify_board_campaign.py" \
    --manifest "$LOCAL/campaign/manifest.json" --results "$LOCAL/board" \
    > "$LOCAL/verify.log" 2>&1; then
  :
else
  status=1
fi
# Fail closed if any credential value was copied into the evidence tree.  Do not
# print the matches themselves: a filename is safe, a leaked secret is not.
if grep -R -l -F -- "$ORANGE_PI_PASSWORD" "$LOCAL" > /dev/null 2>&1; then
  echo 'credential scan: FAIL (value found in artifacts)' >&2
  status=1
else
  echo 'credential scan: PASS (no credential value in artifacts)' >&2
fi
unset RK_BOARD_PASSWORD
printf 'RK3588 admission: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
printf 'Results: %s\n' "$LOCAL"
exit "$status"
