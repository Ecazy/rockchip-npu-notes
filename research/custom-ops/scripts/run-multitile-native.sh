#!/usr/bin/env bash
set -euo pipefail
mt_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mt_build=${1:-$mt_project/build-multitile}
mt_results=${2:-$mt_project/artifacts/multitile-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$mt_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$mt_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
mt_upstream=${UPSTREAM_ROOT:-$mt_project/../upstream-components}
python3 "$mt_project/tools/python/summarize.py" --sources "$mt_project" "$mt_upstream" > "$mt_results/source-manifest.json"
{ date -Is; uname -a; } > "$mt_results/environment.log"
cmake -S "$mt_project" -B "$mt_build" -DUPSTREAM_ROOT="$mt_upstream" -DCMAKE_BUILD_TYPE=Release > "$mt_results/configure.log" 2>&1
cmake --build "$mt_build" -j4 > "$mt_results/build.log" 2>&1
ctest --test-dir "$mt_build" --output-on-failure > "$mt_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
for mt_mask in 1 2 4; do
    RKNPU_CORE_MASK=$mt_mask "$mt_build/multitile_probe" > "$mt_results/core-$mt_mask-device.log" 2>&1
    RKNPU_CORE_MASK=$mt_mask python3 "$mt_project/tools/python/cli_gate.py" "$mt_build/linear_hardsigmoid" "$mt_results/core-$mt_mask-cli" > "$mt_results/core-$mt_mask-cli.log" 2>&1
done
printf 'MULTITILE hardware campaign PASS\n'
