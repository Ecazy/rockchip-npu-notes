#!/usr/bin/env bash
# Host-side Experiment A driver: bundle the current source, run it on the board
# and retrieve the evidence. Reads credentials from the ignored repository .env.
set -euo pipefail
ov_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
ov_repo=$(cd -- "$ov_project/../.." && pwd)
if [[ -f "$ov_repo/.env" ]]; then source "$ov_repo/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD}"
ov_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$ov_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsafe remote workspace path' >&2; exit 1; }
ov_stamp=$(date +%Y%m%d-%H%M%S)
ov_remote="$ov_base/overlap-$ov_stamp"
ov_local=${1:-$ov_project/results/$(date +%Y-%m-%d)/overlap-$ov_stamp}
[[ ! -e "$ov_local" ]] || { echo 'Results destination already exists; use a fresh path' >&2; exit 1; }
mkdir -p "$ov_local"
ov_bundle=$(mktemp /tmp/rk3588-overlap-XXXXXX.tgz)
trap 'rm -f -- "$ov_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
  -czf "$ov_bundle" -C "$ov_repo/research" custom-ops \
  upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
ov_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$ov_target" "mkdir -p '$ov_remote'"
sshpass -e scp "$ov_bundle" "$ov_target:$ov_remote/source.tgz"
sshpass -e ssh "$ov_target" "tar -xzf '$ov_remote/source.tgz' -C '$ov_remote'"
ov_status=0
ov_env="OV_REPS=${OV_REPS:-64} OV_WARMUPS=${OV_WARMUPS:-8} OV_EXPLORE_REPS=${OV_EXPLORE_REPS:-2} OV_PAIR_CPUS=${OV_PAIR_CPUS:-4,5}"
sshpass -e ssh "$ov_target" "$ov_env bash '$ov_remote/custom-ops/run-overlap-native.sh' '$ov_remote/custom-ops/build-overlap' '$ov_remote/results'" || ov_status=$?
sshpass -e scp -r "$ov_target:$ov_remote/results/." "$ov_local/" || ov_status=$?
python3 "$ov_project/overlap_verify.py" "$ov_local" || ov_status=$?
echo "Results: $ov_local"
exit "$ov_status"
