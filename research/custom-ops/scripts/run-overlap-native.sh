#!/usr/bin/env bash
# Board-side overlap/concurrency run. Builds the current source, then runs:
#   matmul-lut  : the original MatMul <-> DPU-only LUT regression
#   gemm-gemv   : GEMM/GEMV + B8/B16 chains + two-thread same/cross-core (x2)
#   explore     : matmul-lut block-bitmap ablation (controlled timeouts)
# Serial via one NPU lock. Pair threads are pinned to two A76 cores.
set -euo pipefail
ov_project=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
ov_build=${1:-$ov_project/build-overlap}
ov_results=${2:-$ov_project/artifacts/overlap-$(date +%Y%m%d-%H%M%S)}
ov_samples=${OV_REPS:-64}
ov_warmups=${OV_WARMUPS:-8}
ov_pair_cpus=${OV_PAIR_CPUS:-4,5}
[[ ! -e "$ov_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$ov_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
{
  date -Is
  uname -a
  gcc --version | head -1
  cat /sys/kernel/debug/rknpu/version
  cat /sys/kernel/debug/rknpu/freq
  cat /sys/kernel/debug/rknpu/load
  echo "pair_cpus=$ov_pair_cpus"
  taskset -pc $$ 2>/dev/null || true
  cat /proc/interrupts
} > "$ov_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$ov_results/kernel-before.log"
ov_cursor=$(sed -n 's/^-- cursor: //p' "$ov_results/kernel-before.log")
[[ -n "$ov_cursor" ]] || { echo 'No kernel journal cursor' >&2; exit 1; }
ov_upstream=${UPSTREAM_ROOT:-$ov_project/../upstream-components}
python3 "$ov_project/summarize.py" --sources "$ov_project" "$ov_upstream" > "$ov_results/source-manifest.json"
cmake -S "$ov_project" -B "$ov_build" -DUPSTREAM_ROOT="$ov_upstream" -DCMAKE_BUILD_TYPE=Release \
  > "$ov_results/configure.log" 2>&1
cmake --build "$ov_build" -j4 > "$ov_results/build.log" 2>&1
ctest --test-dir "$ov_build" --output-on-failure > "$ov_results/host-tests.log" 2>&1
ov_status=0
ov_launcher=()
command -v taskset >/dev/null 2>&1 && ov_launcher=(taskset -c "$ov_pair_cpus")
run_probe() {
  local label=$1 suite=$2 samples=$3 explore=$4 warmups=$5
  local out="$ov_results/$label"
  mkdir -p "$out"
  OL_PROBE_JSON="$out/probe.jsonl" RKNPU_CORE_MASK=1 OL_EXPLORE="$explore" \
    timeout "${OV_TIMEOUT:-1800}" "${ov_launcher[@]}" "$ov_build/overlap_probe" "$suite" "$samples" 1 "$warmups" \
    > "$out/probe.log" 2>&1 || ov_status=$?
  echo "$ov_status" > "$out/exit.txt"
  tail -1 "$out/probe.log" || true
}
fail() { journalctl -k --after-cursor="$ov_cursor" --no-pager > "$ov_results/kernel-after.log"; exit "$ov_status"; }
run_probe matmul-lut matmul-lut "$ov_samples" 0 "$ov_warmups"
[[ "$ov_status" -eq 0 ]] || fail
run_probe gemm-gemv-a gemm-gemv "$ov_samples" 0 "$ov_warmups"
[[ "$ov_status" -eq 0 ]] || fail
run_probe gemm-gemv-b gemm-gemv "$ov_samples" 0 "$ov_warmups"
[[ "$ov_status" -eq 0 ]] || fail
run_probe explore matmul-lut "${OV_EXPLORE_REPS:-2}" 1 0
journalctl -k --after-cursor="$ov_cursor" --no-pager > "$ov_results/kernel-after.log"
exit "$ov_status"
