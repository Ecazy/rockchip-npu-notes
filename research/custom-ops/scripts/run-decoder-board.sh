#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
dc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dc_repo=$(cd -- "$dc_project/../.." && pwd)
if [[ -f "$dc_repo/.env" ]]; then source "$dc_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
dc_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$dc_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
dc_stamp=$(date +%Y%m%d-%H%M%S)
dc_remote="$dc_base/decoder-$dc_stamp"
dc_local=${1:-$dc_project/results/$(date +%Y-%m-%d)/decoder-$dc_stamp}
[[ ! -e "$dc_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$dc_local"
dc_bundle=$(mktemp /tmp/decoder-XXXXXX.tgz)
trap 'rm -f -- "$dc_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$dc_bundle" -C "$dc_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
dc_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$dc_target" "mkdir -p '$dc_remote'"
sshpass -e scp "$dc_bundle" "$dc_target:$dc_remote/source.tgz"
sshpass -e ssh "$dc_target" "tar -xzf '$dc_remote/source.tgz' -C '$dc_remote'"
dc_status=0
sshpass -e ssh "$dc_target" "bash '$dc_remote/custom-ops/scripts/run-decoder-native.sh' '$dc_remote/custom-ops/build-decoder' '$dc_remote/results'" || dc_status=$?
sshpass -e scp -r "$dc_target:$dc_remote/results/." "$dc_local/"
python3 "$dc_project/tools/python/decoder_campaign.py" "$dc_local" || dc_status=$?
printf 'Results: %s\n' "$dc_local"
exit "$dc_status"
