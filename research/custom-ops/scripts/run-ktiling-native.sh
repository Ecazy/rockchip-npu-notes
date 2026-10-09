#!/usr/bin/env bash
# Native RK3588 K-tiling campaign. No clock/driver/kernel-debug writes; serial runs.
set -euo pipefail
kt_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
kt_build=${1:-$kt_project/build-ktiling}
kt_results=${2:-$kt_project/artifacts/ktiling-$(date +%Y%m%d-%H%M%S)}
kt_reps=${KT_REPS:-24}
(( kt_reps >= 20 )) || { echo 'KT_REPS must be >= 20' >&2; exit 1; }
[[ ! -e "$kt_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$kt_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
{
  date -Is
  uname -a
  gcc --version | head -1
  cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
  cat /sys/kernel/debug/rknpu/freq 2>/dev/null || true
  cat /proc/interrupts 2>/dev/null || true
} > "$kt_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$kt_results/kernel-before.log"
kt_cursor=$(sed -n 's/^-- cursor: //p' "$kt_results/kernel-before.log")
[[ -n "$kt_cursor" ]] || { echo 'No kernel journal cursor' >&2; exit 1; }
kt_upstream=${UPSTREAM_ROOT:-$kt_project/../upstream-components}
python3 "$kt_project/summarize.py" --sources "$kt_project" "$kt_upstream" > "$kt_results/source-manifest.json"
cmake -S "$kt_project" -B "$kt_build" -DUPSTREAM_ROOT="$kt_upstream" -DCMAKE_BUILD_TYPE=Release > "$kt_results/configure.log" 2>&1
cmake --build "$kt_build" -j4 > "$kt_results/build.log" 2>&1
ctest --test-dir "$kt_build" --output-on-failure > "$kt_results/host-tests.log" 2>&1
kt_status=0
for kt_mask in 1 2 4; do
  for kt_rep in 1 2; do
    kt_run="$kt_results/core-${kt_mask}-run-${kt_rep}"
    mkdir -p "$kt_run/witnesses"
    RKNPU_CORE_MASK="$kt_mask" KT_PROBE_JSON="$kt_run/probe.jsonl" \
      KT_WITNESS_DIR="$kt_run/witnesses" \
      "$kt_build/ktiling_probe" all "$kt_reps" > "$kt_run/probe.log" 2>&1 || kt_status=$?
    echo "$kt_status" > "$kt_run/exit.txt"
    if (( kt_status == 0 )); then
      python3 "$kt_project/ktiling_gate.py" "$kt_run/witnesses" > "$kt_run/gate.json" 2>&1 || kt_status=$?
      echo "$kt_status" > "$kt_run/exit.txt"
    fi
    tail -1 "$kt_run/probe.log" || true
    if (( kt_status != 0 )); then
      echo "run core-${kt_mask}-run-${kt_rep} exited $kt_status" >&2
      [[ "$kt_status" != 77 ]] || echo 'SKIP: no NPU present' >&2
      break 2
    fi
  done
done
journalctl -k --after-cursor="$kt_cursor" --no-pager > "$kt_results/kernel-after.log"
python3 "$kt_project/ktiling_campaign.py" "$kt_results" "$kt_reps" || kt_status=$?
exit "$kt_status"
