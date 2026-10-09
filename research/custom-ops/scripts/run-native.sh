#!/usr/bin/env bash
# Native RK3588 board campaign. Hardware runs are serial, stop at first failure.
set -euo pipefail
co_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
co_build=${1:-$co_project/build}
co_results=${2:-$co_project/artifacts/$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$co_results" ]] || { echo 'Results destination already exists; use a fresh path' >&2; exit 1; }
mkdir -p "$co_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another custom-ops campaign is running' >&2; exit 1; }
{
  date -Is
  uname -a
  gcc --version | head -1
  cat /sys/kernel/debug/rknpu/version
  cat /sys/kernel/debug/rknpu/freq
  cat /proc/device-tree/compatible | tr '\0' '\n'
} > "$co_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$co_results/kernel-before.log"
co_cursor=$(sed -n 's/^-- cursor: //p' "$co_results/kernel-before.log")
[[ -n "$co_cursor" ]] || { echo 'Cannot establish kernel journal cursor' >&2; exit 1; }
co_upstream=${UPSTREAM_ROOT:-$co_project/../upstream-components}
python3 "$co_project/summarize.py" --sources "$co_project" "$co_upstream" > "$co_results/source-manifest.json"
cmake -S "$co_project" -B "$co_build" -DUPSTREAM_ROOT="$co_upstream" -DCMAKE_BUILD_TYPE=Release > "$co_results/configure.log" 2>&1
cmake --build "$co_build" -j4 > "$co_results/build.log" 2>&1
ctest --test-dir "$co_build" --output-on-failure > "$co_results/host-tests.log" 2>&1
co_status=0
for co_mask in 1 2 4; do
  for co_rep in 1 2; do
    co_run="$co_results/core-${co_mask}-run-${co_rep}"
    mkdir -p "$co_run"
    RKNPU_CORE_MASK="$co_mask" CO_ARTIFACT_DIR="$co_run" "$co_build/custom_ops_probe" all > "$co_run/probe.log" 2>&1 || co_status=$?
    if (( co_status == 0 )); then
      RKNPU_CORE_MASK="$co_mask" python3 "$co_project/cli_gate.py" "$co_build/linear_hardsigmoid" "$co_run" > "$co_run/cli.log" 2>&1 || co_status=$?
    fi
    echo "$co_status" > "$co_run/exit.txt"
    tail -1 "$co_run/probe.log"
    if (( co_status != 0 )); then break 2; fi
  done
done
journalctl -k --after-cursor="$co_cursor" --no-pager > "$co_results/kernel-after.log"
python3 "$co_project/summarize.py" --create "$co_results" || co_status=$?
exit "$co_status"
