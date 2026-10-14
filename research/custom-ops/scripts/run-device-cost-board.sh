#!/usr/bin/env bash
# Deploy exact current sources; preserve failures and never reuse old binaries.
set -euo pipefail
dc_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
dc_repo=$(cd -- "$dc_project/../.." && pwd)
source "$dc_repo/.env"
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
dc_suite=${1:-smoke}; dc_blocks=${2:-1}; dc_operations=${3:-12}; dc_warmup=${4:-2}
[[ "$dc_suite" =~ ^(smoke|cost|concurrency|resident|resident-smoke|saturation|interference|trace|tail)$ ]] || exit 2
for dc_number in "$dc_blocks" "$dc_operations" "$dc_warmup"; do [[ "$dc_number" =~ ^[0-9]+$ ]] || exit 2; done
dc_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$dc_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 2
dc_stamp=$(date +%Y%m%d-%H%M%S)
dc_remote="$dc_base/device-cost-$dc_suite-$dc_stamp"
dc_local="$dc_project/results/$(date +%Y-%m-%d)/device-cost-$dc_suite-$dc_stamp"
[[ ! -e "$dc_local" ]] || exit 1
mkdir -p "$dc_local"
dc_bundle=$(mktemp /tmp/device-cost-source-XXXXXX.tgz)
trap 'rm -f -- "$dc_bundle"' EXIT
tar --exclude='.git' --exclude='build*' --exclude='results' --exclude='artifacts' --exclude='__pycache__' --exclude='*.tgz' \
    -czf "$dc_bundle" -C "$dc_repo/research" custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit
cp "$dc_bundle" "$dc_local/source.tgz"
sha256sum "$dc_local/source.tgz" > "$dc_local/source.sha256"
git -C "$dc_repo" rev-parse HEAD > "$dc_local/base-commit.txt"
git -C "$dc_repo" diff -- research/custom-ops > "$dc_local/workspace.diff"
export SSHPASS="$ORANGE_PI_PASSWORD"
dc_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 "$dc_target" "mkdir -p '$dc_remote'"
sshpass -e scp "$dc_bundle" "$dc_target:$dc_remote/source.tgz"
dc_status=0
sshpass -e ssh "$dc_target" "tar -xzf '$dc_remote/source.tgz' -C '$dc_remote' && cmake -S '$dc_remote/custom-ops' -B '$dc_remote/custom-ops/build' -DCMAKE_BUILD_TYPE=Release > '$dc_remote/build.log' 2>&1 && cmake --build '$dc_remote/custom-ops/build' -j4 >> '$dc_remote/build.log' 2>&1 && ctest --test-dir '$dc_remote/custom-ops/build' --output-on-failure >> '$dc_remote/build.log' 2>&1 && python3 '$dc_remote/custom-ops/tools/python/device_cost_campaign.py' '$dc_remote/custom-ops/build/cost_probe' '$dc_remote/results' --suite '$dc_suite' --blocks '$dc_blocks' --operations '$dc_operations' --warmup '$dc_warmup'" | tee "$dc_local/run.log" || dc_status=$?
sshpass -e scp "$dc_target:$dc_remote/build.log" "$dc_local/build.log" || dc_status=$?
sshpass -e scp -r "$dc_target:$dc_remote/results/." "$dc_local/" || dc_status=$?
if (( dc_status == 0 )); then python3 "$dc_project/tools/python/device_cost_analyze.py" "$dc_local" || dc_status=$?; fi
echo "Results: $dc_local"
exit "$dc_status"
