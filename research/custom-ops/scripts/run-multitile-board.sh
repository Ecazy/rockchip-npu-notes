#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
mt_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mt_repo=$(cd -- "$mt_project/../.." && pwd)
if [[ -f "$mt_repo/.env" ]]; then source "$mt_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
mt_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$mt_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
mt_stamp=$(date +%Y%m%d-%H%M%S)
mt_remote="$mt_base/multitile-$mt_stamp"
mt_local=${1:-$mt_project/results/$(date +%Y-%m-%d)/multitile-$mt_stamp}
[[ ! -e "$mt_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$mt_local"
mt_bundle=$(mktemp /tmp/multitile-XXXXXX.tgz)
trap 'rm -f -- "$mt_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$mt_bundle" -C "$mt_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
mt_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$mt_target" "mkdir -p '$mt_remote'"
sshpass -e scp "$mt_bundle" "$mt_target:$mt_remote/source.tgz"
sshpass -e ssh "$mt_target" "tar -xzf '$mt_remote/source.tgz' -C '$mt_remote'"
mt_status=0
sshpass -e ssh "$mt_target" "bash '$mt_remote/custom-ops/scripts/run-multitile-native.sh' '$mt_remote/custom-ops/build-multitile' '$mt_remote/results'" || mt_status=$?
sshpass -e scp -r "$mt_target:$mt_remote/results/." "$mt_local/"
printf 'Results: %s\n' "$mt_local"
exit "$mt_status"
