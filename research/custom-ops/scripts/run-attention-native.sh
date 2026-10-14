#!/usr/bin/env bash
set -euo pipefail
ac_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ac_build=${1:-$ac_project/build-attention}
ac_results=${2:-$ac_project/artifacts/attention-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$ac_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$ac_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
ac_upstream=${UPSTREAM_ROOT:-$ac_project/../upstream-components}
python3 "$ac_project/tools/python/summarize.py" --sources "$ac_project" "$ac_upstream" > "$ac_results/source-manifest.json"
{ date -Is; uname -a; } > "$ac_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$ac_results/kernel-before.log" 2>&1 || true
ac_cursor=$(sed -n 's/^-- cursor: //p' "$ac_results/kernel-before.log")
ac_capture_kernel() {
    if [[ -n "$ac_cursor" ]]; then
        journalctl -k --after-cursor="$ac_cursor" --no-pager > "$ac_results/kernel-interval.log" 2>&1 || true
    fi
}
trap ac_capture_kernel EXIT
cmake -S "$ac_project" -B "$ac_build" -DUPSTREAM_ROOT="$ac_upstream" -DCMAKE_BUILD_TYPE=Release > "$ac_results/configure.log" 2>&1
cmake --build "$ac_build" -j4 > "$ac_results/build.log" 2>&1
ctest --test-dir "$ac_build" --output-on-failure > "$ac_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
for ac_mask in 1 2 4; do
    for ac_mode in chain separate; do
        RKNPU_CORE_MASK=$ac_mask "$ac_build/attention_probe" "$ac_results/core-$ac_mask-$ac_mode" "$ac_mode" > "$ac_results/core-$ac_mask-$ac_mode.log" 2>&1
        python3 "$ac_project/tools/python/attention_gate.py" "$ac_results/core-$ac_mask-$ac_mode" "$ac_mode" > "$ac_results/core-$ac_mask-$ac_mode-oracle.log" 2>&1
    done
    RKNPU_CORE_MASK=$ac_mask "$ac_build/multitile_probe" > "$ac_results/core-$ac_mask-device-regression.log" 2>&1
done
ac_capture_kernel
python3 "$ac_project/tools/python/attention_campaign.py" "$ac_results"
printf 'ATTENTION hardware campaign PASS\n'
