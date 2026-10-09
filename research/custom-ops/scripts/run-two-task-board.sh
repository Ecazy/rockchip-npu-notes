#!/usr/bin/env bash
set -euo pipefail
tt_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
tt_repo=$(cd -- "$tt_project/../.." && pwd)
if [[ -f "$tt_repo/.env" ]]; then source "$tt_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
tt_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$tt_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Invalid workspace path' >&2; exit 1; }
tt_stamp=$(date +%Y%m%d-%H%M%S)
tt_remote="$tt_base/two-task-$tt_stamp"
tt_local=${1:-$tt_project/results/$(date +%Y-%m-%d)/two-task-$tt_stamp}
[[ ! -e "$tt_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$tt_local"
tt_bundle=$(mktemp /tmp/two-task-XXXXXX.tgz)
trap 'rm -f -- "$tt_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
  -czf "$tt_bundle" -C "$tt_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
tt_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$tt_target" "mkdir -p '$tt_remote'"
sshpass -e scp "$tt_bundle" "$tt_target:$tt_remote/source.tgz"
sshpass -e ssh "$tt_target" "tar -xzf '$tt_remote/source.tgz' -C '$tt_remote'"
tt_status=0
sshpass -e ssh "$tt_target" "bash '$tt_remote/custom-ops/run-two-task-native.sh' '$tt_remote/custom-ops/build-two-task' '$tt_remote/results'" || tt_status=$?
sshpass -e scp -r "$tt_target:$tt_remote/results/." "$tt_local/"
python3 "$tt_project/two_task_campaign.py" "$tt_local" 128 || tt_status=$?
echo "Results: $tt_local"
exit "$tt_status"
