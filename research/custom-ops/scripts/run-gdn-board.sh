#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
gdn_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gdn_repo=$(cd -- "$gdn_project/../.." && pwd)
if [[ -f "$gdn_repo/.env" ]]; then source "$gdn_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
gdn_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$gdn_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
gdn_stamp=$(date +%Y%m%d-%H%M%S)
gdn_remote="$gdn_base/gdn-$gdn_stamp"
gdn_local=${1:-$gdn_project/results/$(date +%Y-%m-%d)/gdn-$gdn_stamp}
[[ ! -e "$gdn_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$gdn_local"
gdn_bundle=$(mktemp /tmp/gdn-XXXXXX.tgz)
trap 'rm -f -- "$gdn_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$gdn_bundle" -C "$gdn_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
gdn_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$gdn_target" "mkdir -p '$gdn_remote'"
sshpass -e scp "$gdn_bundle" "$gdn_target:$gdn_remote/source.tgz"
sshpass -e ssh "$gdn_target" "tar -xzf '$gdn_remote/source.tgz' -C '$gdn_remote'"
gdn_status=0
sshpass -e ssh "$gdn_target" "bash '$gdn_remote/custom-ops/scripts/run-gdn-native.sh' '$gdn_remote/custom-ops/build-gdn' '$gdn_remote/results'" || gdn_status=$?
sshpass -e scp -r "$gdn_target:$gdn_remote/results/." "$gdn_local/"

if [[ "$gdn_status" == 0 ]]; then
    python3 "$gdn_project/tools/python/gdn_campaign.py" "$gdn_local" > "$gdn_local/campaign.json" || gdn_status=$?
fi
printf 'Results: %s\n' "$gdn_local"
exit "$gdn_status"
