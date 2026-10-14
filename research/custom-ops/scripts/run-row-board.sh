#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
rc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rc_repo=$(cd -- "$rc_project/../.." && pwd)
if [[ -f "$rc_repo/.env" ]]; then source "$rc_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
rc_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$rc_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
rc_stamp=$(date +%Y%m%d-%H%M%S)
rc_remote="$rc_base/row-$rc_stamp"
rc_local=${1:-$rc_project/results/$(date +%Y-%m-%d)/row-$rc_stamp}
[[ ! -e "$rc_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$rc_local"
rc_bundle=$(mktemp /tmp/row-XXXXXX.tgz)
trap 'rm -f -- "$rc_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$rc_bundle" -C "$rc_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
rc_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$rc_target" "mkdir -p '$rc_remote'"
sshpass -e scp "$rc_bundle" "$rc_target:$rc_remote/source.tgz"
sshpass -e ssh "$rc_target" "tar -xzf '$rc_remote/source.tgz' -C '$rc_remote'"
rc_status=0
sshpass -e ssh "$rc_target" "bash '$rc_remote/custom-ops/scripts/run-row-native.sh' '$rc_remote/custom-ops/build-row' '$rc_remote/results'" || rc_status=$?
sshpass -e scp -r "$rc_target:$rc_remote/results/." "$rc_local/"
python3 "$rc_project/tools/python/row_campaign.py" "$rc_local" || rc_status=$?
printf 'Results: %s\n' "$rc_local"
exit "$rc_status"
