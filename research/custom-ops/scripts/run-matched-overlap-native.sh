#!/usr/bin/env bash
set -euo pipefail
mo_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mo_build=$1
mo_results=$2
mo_config=$3
[[ ! -e "$mo_results" ]]
mkdir -p "$mo_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'NPU campaign lock held' >&2; exit 1; }
{ date -Is; uname -a; cat /sys/kernel/debug/rknpu/version /sys/kernel/debug/rknpu/freq /sys/kernel/debug/rknpu/load; } > "$mo_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$mo_results/kernel-before.log"
mo_cursor=$(sed -n 's/^-- cursor: //p' "$mo_results/kernel-before.log")
[[ -n "$mo_cursor" ]]
trap 'journalctl -k --after-cursor="$mo_cursor" --no-pager > "$mo_results/kernel-after.log"' EXIT
cp "$mo_config" "$mo_results/config.json"
mo_upstream=${UPSTREAM_ROOT:-$mo_project/../upstream-components}
python3 "$mo_project/tools/python/summarize.py" --sources "$mo_project" "$mo_upstream" > "$mo_results/source-manifest.json"
cmake -S "$mo_project" -B "$mo_build" -DCMAKE_BUILD_TYPE=Release -DUPSTREAM_ROOT="$mo_upstream" > "$mo_results/configure.log" 2>&1
cmake --build "$mo_build" -j4 > "$mo_results/build.log" 2>&1
ctest --test-dir "$mo_build" --output-on-failure > "$mo_results/host-tests.log" 2>&1
python3 "$mo_project/tools/python/matched_overlap_campaign.py" "$mo_build/regcmd_overlap_probe" "$mo_config" "$mo_results"
