#!/usr/bin/env bash
# Reads credentials from the ignored repository .env; never prints the password.
# Uploads a fresh, isolated source bundle. No dependency on old board workspaces.
set -euo pipefail
co_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
co_repo=$(cd -- "$co_project/../.." && pwd)
if [[ -f "$co_repo/.env" ]]; then source "$co_repo/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD}"
co_remote_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
# Remote command paths are shell arguments: constrain them instead of using JSON quoting.
[[ "$co_remote_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsafe remote workspace path' >&2; exit 1; }
co_stamp=$(date +%Y%m%d-%H%M%S)
co_remote="$co_remote_base/custom-ops-$co_stamp"
co_destination=${1:-$co_project/results/$(date +%Y-%m-%d)/$co_stamp}
[[ ! -e "$co_destination" ]] || { echo 'Results destination already exists; use a fresh path' >&2; exit 1; }
mkdir -p "$co_destination"
co_bundle=$(mktemp /tmp/rk3588-custom-ops-XXXXXX.tgz)
trap 'rm -f -- "$co_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' \
  -czf "$co_bundle" -C "$co_repo/research" custom-ops \
  upstream-components/rocket-userspace upstream-components/rknpu-submit
export SSHPASS="$ORANGE_PI_PASSWORD"
co_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$co_target" "mkdir -p '$co_remote'"
sshpass -e scp "$co_bundle" "$co_target:$co_remote/source.tgz"
sshpass -e ssh "$co_target" "tar -xzf '$co_remote/source.tgz' -C '$co_remote'"
co_status=0
sshpass -e ssh "$co_target" "bash '$co_remote/custom-ops/run-native.sh' '$co_remote/custom-ops/build' '$co_remote/results'" || co_status=$?
# Retrieve evidence on failures too; publishing a PASS requires the summary gate.
sshpass -e scp -r "$co_target:$co_remote/results/." "$co_destination/"
python3 "$co_project/summarize.py" "$co_destination" || co_status=$?
echo "Results: $co_destination"
exit "$co_status"
