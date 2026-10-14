#!/usr/bin/env bash
# Upload actual sources to a new board directory; retrieve failures as well.
set -euo pipefail
ac_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ac_repo=$(cd -- "$ac_project/../.." && pwd)
if [[ -f "$ac_repo/.env" ]]; then source "$ac_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
ac_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$ac_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
ac_stamp=$(date +%Y%m%d-%H%M%S)
ac_remote="$ac_base/attention-$ac_stamp"
ac_local=${1:-$ac_project/results/$(date +%Y-%m-%d)/attention-$ac_stamp}
[[ ! -e "$ac_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$ac_local"
ac_bundle=$(mktemp /tmp/attention-XXXXXX.tgz)
trap 'rm -f -- "$ac_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
    -czf "$ac_bundle" -C "$ac_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
ac_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$ac_target" "mkdir -p '$ac_remote'"
sshpass -e scp "$ac_bundle" "$ac_target:$ac_remote/source.tgz"
sshpass -e ssh "$ac_target" "tar -xzf '$ac_remote/source.tgz' -C '$ac_remote'"
ac_status=0
sshpass -e ssh "$ac_target" "bash '$ac_remote/custom-ops/scripts/run-attention-native.sh' '$ac_remote/custom-ops/build-attention' '$ac_remote/results'" || ac_status=$?
sshpass -e scp -r "$ac_target:$ac_remote/results/." "$ac_local/"
python3 "$ac_project/tools/python/attention_campaign.py" "$ac_local" || ac_status=$?
printf 'Results: %s\n' "$ac_local"
exit "$ac_status"
