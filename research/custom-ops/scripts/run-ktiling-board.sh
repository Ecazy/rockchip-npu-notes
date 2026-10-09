#!/usr/bin/env bash
# Reads credentials from the ignored repository .env; never prints the password.
# Uploads a fresh isolated source bundle and runs the K-tiling campaign remotely.
set -euo pipefail
kt_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
kt_repo=$(cd -- "$kt_project/../.." && pwd)
if [[ -f "$kt_repo/.env" ]]; then source "$kt_repo/.env"; fi
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
kt_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$kt_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Invalid workspace path' >&2; exit 1; }
kt_reps=${KT_REPS:-24}
(( kt_reps >= 20 )) || { echo 'KT_REPS must be >= 20' >&2; exit 1; }
kt_stamp=$(date +%Y%m%d-%H%M%S)
kt_remote="$kt_base/ktiling-$kt_stamp"
kt_local=${1:-$kt_project/results/$(date +%Y-%m-%d)/ktiling-$kt_stamp}
[[ ! -e "$kt_local" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$kt_local"
kt_bundle=$(mktemp /tmp/ktiling-XXXXXX.tgz)
trap 'rm -f -- "$kt_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
  -czf "$kt_bundle" -C "$kt_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
kt_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$kt_target" "mkdir -p '$kt_remote'"
sshpass -e scp "$kt_bundle" "$kt_target:$kt_remote/source.tgz"
sshpass -e ssh "$kt_target" "tar -xzf '$kt_remote/source.tgz' -C '$kt_remote'"
kt_status=0
# Retrieve evidence on failures too; publishing a PASS requires the campaign gate.
sshpass -e ssh "$kt_target" "KT_REPS=$kt_reps bash '$kt_remote/custom-ops/run-ktiling-native.sh' '$kt_remote/custom-ops/build-ktiling' '$kt_remote/results'" || kt_status=$?
sshpass -e scp -r "$kt_target:$kt_remote/results/." "$kt_local/"
python3 "$kt_project/ktiling_campaign.py" "$kt_local" "$kt_reps" || kt_status=$?
echo "Results: $kt_local"
exit "$kt_status"
