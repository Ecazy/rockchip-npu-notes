#!/usr/bin/env bash
set -euo pipefail
mo_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mo_repo=$(cd -- "$mo_project/../.." && pwd)
mo_config=$(readlink -f -- "${1:?Supply a JSON case configuration}")
[[ -f "$mo_config" ]]
source "$mo_repo/.env"
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
mo_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$mo_base" =~ ^/[a-zA-Z0-9_./-]+$ ]]
mo_stamp=$(date +%Y%m%d-%H%M%S)
mo_remote="$mo_base/matched-overlap-$mo_stamp"
mo_local=${2:-$mo_project/results/$(date +%Y-%m-%d)/matched-overlap-$mo_stamp}
[[ ! -e "$mo_local" ]]
mkdir -p "$mo_local"
mo_bundle=$(mktemp /tmp/matched-overlap-XXXXXX.tgz)
trap 'rm -f -- "$mo_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' -czf "$mo_bundle" -C "$mo_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
mo_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$mo_target" "mkdir -p '$mo_remote'"
sshpass -e scp "$mo_bundle" "$mo_target:$mo_remote/source.tgz"
sshpass -e scp "$mo_config" "$mo_target:$mo_remote/config.json"
sshpass -e ssh "$mo_target" "tar -xzf '$mo_remote/source.tgz' -C '$mo_remote'"
mo_status=0
sshpass -e ssh "$mo_target" "bash '$mo_remote/custom-ops/scripts/run-matched-overlap-native.sh' '$mo_remote/build' '$mo_remote/results' '$mo_remote/config.json'" || mo_status=$?
sshpass -e scp -r "$mo_target:$mo_remote/results/." "$mo_local/"
if [[ -f "$mo_project/tools/python/matched_overlap_analyze.py" ]]; then
    python3 "$mo_project/tools/python/matched_overlap_analyze.py" "$mo_local" || mo_status=$?
fi
echo "Results: $mo_local"
exit "$mo_status"
