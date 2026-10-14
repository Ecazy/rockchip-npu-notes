#!/usr/bin/env bash
set -euo pipefail
dc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dc_build=${1:-$dc_project/build-decoder}
dc_results=${2:-$dc_project/artifacts/decoder-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$dc_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$dc_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
dc_upstream=${UPSTREAM_ROOT:-$dc_project/../upstream-components}
python3 "$dc_project/tools/python/summarize.py" --sources "$dc_project" "$dc_upstream" > "$dc_results/source-manifest.json"
{ date -Is; uname -a; } > "$dc_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$dc_results/kernel-before.log" 2>&1 || true
dc_cursor=$(sed -n 's/^-- cursor: //p' "$dc_results/kernel-before.log")
dc_capture_kernel() {
    if [[ -n "$dc_cursor" ]]; then
        journalctl -k --after-cursor="$dc_cursor" --no-pager > "$dc_results/kernel-interval.log" 2>&1 || true
    fi
}
trap dc_capture_kernel EXIT
cmake -S "$dc_project" -B "$dc_build" -DUPSTREAM_ROOT="$dc_upstream" -DCMAKE_BUILD_TYPE=Release > "$dc_results/configure.log" 2>&1
cmake --build "$dc_build" -j4 > "$dc_results/build.log" 2>&1
ctest --test-dir "$dc_build" --output-on-failure > "$dc_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
for dc_mask in 1 2 4; do
    for dc_mode in chain separate; do
        RKNPU_CORE_MASK=$dc_mask "$dc_build/decoder_probe" "$dc_results/core-$dc_mask-$dc_mode" "$dc_mode" > "$dc_results/core-$dc_mask-$dc_mode.log" 2>&1
        python3 "$dc_project/tools/python/decoder_gate.py" "$dc_results/core-$dc_mask-$dc_mode" "$dc_mode" > "$dc_results/core-$dc_mask-$dc_mode-oracle.log" 2>&1
    done
    RKNPU_CORE_MASK=$dc_mask "$dc_build/attention_probe" "$dc_results/core-$dc_mask-attention-regression" chain > "$dc_results/core-$dc_mask-attention-regression.log" 2>&1
    python3 "$dc_project/tools/python/attention_gate.py" "$dc_results/core-$dc_mask-attention-regression" chain > "$dc_results/core-$dc_mask-attention-regression-oracle.log" 2>&1
    RKNPU_CORE_MASK=$dc_mask "$dc_build/multitile_probe" > "$dc_results/core-$dc_mask-device-regression.log" 2>&1
done
dc_capture_kernel
python3 "$dc_project/tools/python/decoder_campaign.py" "$dc_results"
printf 'DECODER hardware campaign PASS\n'
