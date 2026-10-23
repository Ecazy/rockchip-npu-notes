#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
gx_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gx_repo=$(cd -- "$gx_project/../.." && pwd)
if [[ -f "$gx_repo/.env" ]]; then source "$gx_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
gx_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$gx_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
gx_stamp=$(date +%Y%m%d-%H%M%S)
gx_remote="$gx_base/gdn-experiments-$gx_stamp"
gx_local=${1:-$gx_project/results/$(date +%Y-%m-%d)/gdn-experiments-$gx_stamp}
[[ ! -e "$gx_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$gx_local"
gx_bundle=$(mktemp /tmp/gdn-experiments-XXXXXX.tgz)
trap 'rm -f -- "$gx_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$gx_bundle" -C "$gx_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
gx_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$gx_target" "mkdir -p '$gx_remote'"
sshpass -e scp "$gx_bundle" "$gx_target:$gx_remote/source.tgz"
sshpass -e ssh "$gx_target" "tar -xzf '$gx_remote/source.tgz' -C '$gx_remote'"
gx_status=0
sshpass -e ssh "$gx_target" "bash '$gx_remote/custom-ops/scripts/run-gdn-experiments-native.sh' '$gx_remote/custom-ops/build-gdn-experiments' '$gx_remote/results'" || gx_status=$?
sshpass -e scp -r "$gx_target:$gx_remote/results/." "$gx_local/"

if [[ "$gx_status" == 0 ]]; then
    python3 "$gx_project/tools/python/gdn_experiments_gate.py" "$gx_local" > "$gx_local/experiments.json" || gx_status=$?
fi
printf 'Results: %s\n' "$gx_local"
exit "$gx_status"
