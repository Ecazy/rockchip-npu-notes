#!/usr/bin/env bash
# Controlled A/B/C/D campaign. No clock/driver changes; serial native RK3588 runs.
set -euo pipefail
tt_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
tt_build=${1:-$tt_project/build-two-task}
tt_results=${2:-$tt_project/artifacts/two-task-$(date +%Y%m%d-%H%M%S)}
tt_rounds=${TT_ROUNDS:-128}
[[ ! -e "$tt_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$tt_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
{
  date -Is
  uname -a
  gcc --version | head -1
  cat /sys/kernel/debug/rknpu/version
  cat /sys/kernel/debug/rknpu/freq
  cat /proc/interrupts
} > "$tt_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$tt_results/kernel-before.log"
tt_cursor=$(sed -n 's/^-- cursor: //p' "$tt_results/kernel-before.log")
[[ -n "$tt_cursor" ]] || { echo 'No kernel journal cursor' >&2; exit 1; }
tt_upstream=${UPSTREAM_ROOT:-$tt_project/../upstream-components}
python3 "$tt_project/summarize.py" --sources "$tt_project" "$tt_upstream" > "$tt_results/source-manifest.json"
cmake -S "$tt_project" -B "$tt_build" -DUPSTREAM_ROOT="$tt_upstream" -DCMAKE_BUILD_TYPE=Release > "$tt_results/configure.log" 2>&1
cmake --build "$tt_build" -j4 > "$tt_results/build.log" 2>&1
ctest --test-dir "$tt_build" --output-on-failure > "$tt_results/host-tests.log" 2>&1
tt_status=0
for tt_mask in 1 2 4; do
  for tt_sptr in 14 48; do
    for tt_rep in 1 2; do
      tt_run="$tt_results/core-${tt_mask}-sptr-${tt_sptr}-run-${tt_rep}"
      mkdir -p "$tt_run"
      RKNPU_CORE_MASK="$tt_mask" TT_TRACE_JSON="$tt_run/trace.jsonl" TT_TRACE_DIR="$tt_run" \
        TT_PROBE_JSON="$tt_run/probe.jsonl" LD_PRELOAD="$tt_build/libtwo_task_trace.so" \
        "$tt_build/two_task_probe" "$tt_rounds" "$tt_sptr" > "$tt_run/probe.log" 2>&1 || tt_status=$?
      echo "$tt_status" > "$tt_run/exit.txt"
      gzip -n -9 "$tt_run/trace.jsonl"
      if ((tt_status == 0)); then
        python3 "$tt_project/two_task_verify.py" "$tt_run" > "$tt_run/validation.json" || tt_status=$?
      fi
      tail -1 "$tt_run/probe.log"
      if ((tt_status != 0)); then break 3; fi
    done
  done
done
journalctl -k --after-cursor="$tt_cursor" --no-pager > "$tt_results/kernel-after.log"
python3 "$tt_project/two_task_campaign.py" "$tt_results" "$tt_rounds" || tt_status=$?
exit "$tt_status"
