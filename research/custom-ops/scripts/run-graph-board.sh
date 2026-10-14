#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
gc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gc_repo=$(cd -- "$gc_project/../.." && pwd)
if [[ -f "$gc_repo/.env" ]]; then source "$gc_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
gc_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$gc_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
gc_stamp=$(date +%Y%m%d-%H%M%S)
gc_remote="$gc_base/graph-$gc_stamp"
gc_local=${1:-$gc_project/results/$(date +%Y-%m-%d)/graph-$gc_stamp}
[[ ! -e "$gc_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$gc_local"
gc_bundle=$(mktemp /tmp/graph-XXXXXX.tgz)
trap 'rm -f -- "$gc_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$gc_bundle" -C "$gc_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
gc_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$gc_target" "mkdir -p '$gc_remote'"
sshpass -e scp "$gc_bundle" "$gc_target:$gc_remote/source.tgz"
sshpass -e ssh "$gc_target" "tar -xzf '$gc_remote/source.tgz' -C '$gc_remote'"
gc_status=0
sshpass -e ssh "$gc_target" "bash '$gc_remote/custom-ops/scripts/run-graph-native.sh' '$gc_remote/custom-ops/build-graph' '$gc_remote/results'" || gc_status=$?
sshpass -e scp -r "$gc_target:$gc_remote/results/." "$gc_local/"
printf 'Results: %s\n' "$gc_local"
exit "$gc_status"
