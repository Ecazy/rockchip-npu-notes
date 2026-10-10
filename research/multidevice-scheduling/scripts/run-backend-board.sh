#!/usr/bin/env bash
# Isolated build + one real submit; no clock/governor or existing workspace changes.
set -euo pipefail
md_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$md_repo/.env"
: "${ORANGE_PI_HOST:?}" "${ORANGE_PI_USER:?}" "${ORANGE_PI_PASSWORD:?}"
md_base=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
[[ "$md_base" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 2
md_stamp=$(date +%Y%m%d-%H%M%S)
md_remote="$md_base/multidevice-backend-$md_stamp"
md_result="$md_repo/research/multidevice-scheduling/results/backend-$md_stamp"
[[ ! -e "$md_result" ]] || exit 1
mkdir -p "$md_result"
md_bundle="$md_result/source.tgz"
tar --exclude='.git' --exclude='build*' --exclude='results' --exclude='artifacts' --exclude='__pycache__' \
 --exclude='data' --exclude='evidence' --exclude='static-analysis' --exclude='*.tgz' \
 -czf "$md_bundle" -C "$md_repo" research/custom-ops research/multidevice-scheduling \
 research/upstream-components/rocket-userspace research/upstream-components/rknpu-submit
sha256sum "$md_bundle" > "$md_result/source.sha256"
export SSHPASS="$ORANGE_PI_PASSWORD"
md_target="$ORANGE_PI_USER@$ORANGE_PI_HOST"
sshpass -e ssh -o ConnectTimeout=10 "$md_target" "mkdir -p '$md_remote'"
sshpass -e scp "$md_bundle" "$md_target:$md_remote/source.tgz"
md_status=0
sshpass -e ssh "$md_target" "tar -xzf '$md_remote/source.tgz' -C '$md_remote' && cmake -S '$md_remote/research/multidevice-scheduling/backends/bridge' -B '$md_remote/build' -DCMAKE_BUILD_TYPE=Release > '$md_remote/build.log' 2>&1 && cmake --build '$md_remote/build' --target co_backend_bridge -j4 >> '$md_remote/build.log' 2>&1 && uname -a > '$md_remote/hardware.txt' && (flock -n 9 && CO_BACKEND_BRIDGE='$md_remote/build/libco_backend_bridge.so' RKNPU_CORE_MASK=1 python3 '$md_remote/research/multidevice-scheduling/backends/smoke.py') 9>/tmp/rk3588-custom-ops.lock > '$md_remote/smoke.json' 2> '$md_remote/smoke.stderr'" || md_status=$?
for md_file in build.log hardware.txt smoke.json smoke.stderr; do
 sshpass -e scp "$md_target:$md_remote/$md_file" "$md_result/$md_file" || md_status=$?
done
printf '%s\n' "$md_status" > "$md_result/exit-code.txt"
echo "Evidence: $md_result"
exit "$md_status"
