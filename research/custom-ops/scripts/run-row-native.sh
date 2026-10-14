#!/usr/bin/env bash
set -euo pipefail
rc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rc_build=${1:-$rc_project/build-row}
rc_results=${2:-$rc_project/artifacts/row-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$rc_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$rc_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
rc_upstream=${UPSTREAM_ROOT:-$rc_project/../upstream-components}
python3 "$rc_project/tools/python/summarize.py" --sources "$rc_project" "$rc_upstream" > "$rc_results/source-manifest.json"
{ date -Is; uname -a; } > "$rc_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$rc_results/kernel-before.log" 2>&1 || true
rc_cursor=$(sed -n 's/^-- cursor: //p' "$rc_results/kernel-before.log")
rc_capture_kernel() {
    if [[ -n "$rc_cursor" ]]; then
        journalctl -k --after-cursor="$rc_cursor" --no-pager > "$rc_results/kernel-interval.log" 2>&1 || true
    fi
}
trap rc_capture_kernel EXIT
cmake -S "$rc_project" -B "$rc_build" -DUPSTREAM_ROOT="$rc_upstream" -DCMAKE_BUILD_TYPE=Release > "$rc_results/configure.log" 2>&1
cmake --build "$rc_build" -j4 > "$rc_results/build.log" 2>&1
ctest --test-dir "$rc_build" --output-on-failure > "$rc_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
for rc_mask in 1 2 4; do
    RKNPU_CORE_MASK=$rc_mask python3 "$rc_project/tools/python/row_gate.py" "$rc_build/row_graph" "$rc_results/core-$rc_mask-row" > "$rc_results/core-$rc_mask-row.log" 2>&1
    RKNPU_CORE_MASK=$rc_mask python3 "$rc_project/tools/python/graph_gate.py" "$rc_build/graph_ffn" "$rc_results/core-$rc_mask-graph" --rmsnorm > "$rc_results/core-$rc_mask-graph.log" 2>&1
    RKNPU_CORE_MASK=$rc_mask "$rc_build/multitile_probe" > "$rc_results/core-$rc_mask-device-regression.log" 2>&1
    RKNPU_CORE_MASK=$rc_mask python3 "$rc_project/tools/python/cli_gate.py" "$rc_build/linear_hardsigmoid" "$rc_results/core-$rc_mask-cli-regression" > "$rc_results/core-$rc_mask-cli-regression.log" 2>&1
done
rc_capture_kernel
python3 "$rc_project/tools/python/row_campaign.py" "$rc_results"
printf 'ROW + normalized FFN hardware campaign PASS\n'
