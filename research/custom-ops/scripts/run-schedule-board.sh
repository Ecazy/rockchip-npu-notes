#!/usr/bin/env bash
set -euo pipefail
sc_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
sc_repo=$(cd -- "$sc_project/../.." && pwd)
source "$sc_repo/.env"
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
sc_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$sc_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 1
sc_stamp=$(date +%Y%m%d-%H%M%S)
sc_remote="$sc_base/scheduling-$sc_stamp"
sc_local=${1:-$sc_project/results/$(date +%Y-%m-%d)/scheduling-$sc_stamp}
[[ ! -e "$sc_local" ]] || exit 1
mkdir -p "$sc_local"
sc_bundle=$(mktemp /tmp/scheduling-XXXXXX.tgz)
trap 'rm -f -- "$sc_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='artifacts' --exclude='results' --exclude='*.tgz' -czf "$sc_bundle" -C "$sc_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
sha256sum "$sc_bundle" > "$sc_local/source-bundle.sha256"
cp "$sc_bundle" "$sc_local/source.tgz"
export SSHPASS="$ORANGE_PI_PASSWORD"
sc_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh "$sc_target" "mkdir -p '$sc_remote'"
sshpass -e scp "$sc_bundle" "$sc_target:$sc_remote/source.tgz"
sc_status=0
sshpass -e ssh "$sc_target" "tar -xzf '$sc_remote/source.tgz' -C '$sc_remote' && cmake -S '$sc_remote/custom-ops' -B '$sc_remote/custom-ops/build' -DCMAKE_BUILD_TYPE=Release > '$sc_remote/build.log' 2>&1 && cmake --build '$sc_remote/custom-ops/build' -j4 >> '$sc_remote/build.log' 2>&1 && ctest --test-dir '$sc_remote/custom-ops/build' --output-on-failure >> '$sc_remote/build.log' 2>&1 && ROCKET_CHIP=rk3588 python3 '$sc_remote/custom-ops/schedule_capture.py' '$sc_remote/custom-ops/build/schedule_probe' '$sc_remote/results'" || sc_status=$?
sshpass -e scp "$sc_target:$sc_remote/build.log" "$sc_local/build.log"
sshpass -e scp -r "$sc_target:$sc_remote/results/." "$sc_local/" || sc_status=$?
if (( sc_status == 0 )); then python3 "$sc_project/schedule_analyze.py" "$sc_local" || sc_status=$?; fi
echo "Results: $sc_local"
exit "$sc_status"
