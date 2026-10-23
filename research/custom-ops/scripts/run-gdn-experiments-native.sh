#!/usr/bin/env bash
set -euo pipefail
gx_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gx_build=${1:-$gx_project/build-gdn-experiments}
gx_results=${2:-$gx_project/artifacts/gdn-experiments-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$gx_results" ]] || exit 1
mkdir -p "$gx_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || exit 1
gx_upstream=${UPSTREAM_ROOT:-$gx_project/../upstream-components}
python3 "$gx_project/tools/python/summarize.py" --sources "$gx_project" "$gx_upstream" > "$gx_results/source-manifest.json"
journalctl -k -n 0 --show-cursor --no-pager > "$gx_results/kernel-before.log" 2>&1 || true
gx_cursor=$(sed -n 's/^-- cursor: //p' "$gx_results/kernel-before.log")
gx_policy=/sys/devices/system/cpu/cpufreq/policy4
gx_npu=/sys/class/devfreq/fdab0000.npu
gx_old_governor=$(cat "$gx_policy/scaling_governor")
gx_old_min=$(cat "$gx_policy/scaling_min_freq")
gx_old_max=$(cat "$gx_policy/scaling_max_freq")
gx_npu_old_governor=$(cat "$gx_npu/governor")
gx_npu_old_min=$(cat "$gx_npu/min_freq")
gx_npu_old_max=$(cat "$gx_npu/max_freq")
gx_restore() {
    if [[ -w "$gx_npu/min_freq" && -w "$gx_npu/governor" ]]; then
        printf '%s' "$gx_npu_old_min" > "$gx_npu/min_freq" || true
        printf '%s' "$gx_npu_old_governor" > "$gx_npu/governor" || true
    fi
    { cat "$gx_npu/governor" "$gx_npu/min_freq" "$gx_npu/max_freq"; } > "$gx_results/npu-restored.log"
    if [[ -w "$gx_policy/scaling_min_freq" && -w "$gx_policy/scaling_governor" ]]; then
        printf '%s' "$gx_old_min" > "$gx_policy/scaling_min_freq" || true
        printf '%s' "$gx_old_governor" > "$gx_policy/scaling_governor" || true
    fi
    { cat "$gx_policy/scaling_governor" "$gx_policy/scaling_min_freq" "$gx_policy/scaling_max_freq"; } > "$gx_results/cpu-restored.log"
    journalctl -k --after-cursor="$gx_cursor" --no-pager > "$gx_results/kernel-interval.log" 2>&1 || true
}
trap gx_restore EXIT
printf 'governor=%s min=%s max=%s\n' "$gx_old_governor" "$gx_old_min" "$gx_old_max" > "$gx_results/cpu-original.log"
printf 'governor=%s min=%s max=%s\n' "$gx_npu_old_governor" "$gx_npu_old_min" "$gx_npu_old_max" > "$gx_results/npu-original.log"
if [[ -w "$gx_policy/scaling_min_freq" && -w "$gx_policy/scaling_governor" ]]; then
    if printf '%s' performance > "$gx_policy/scaling_governor" && printf '%s' "$gx_old_max" > "$gx_policy/scaling_min_freq"; then
        echo 'Requested CPU4 policy min=max at existing maximum; per-token frequency still recorded.' > "$gx_results/cpu-control.log"
    else
        echo 'Frequency request failed; restore defaults and retain interleaved repeated measurement.' > "$gx_results/cpu-control.log"
        printf '%s' "$gx_old_min" > "$gx_policy/scaling_min_freq" || true
        printf '%s' "$gx_old_governor" > "$gx_policy/scaling_governor" || true
    fi
else
    echo 'CPU policy not writable; retain interleaved repeated measurement.' > "$gx_results/cpu-control.log"
fi
if [[ -w "$gx_npu/min_freq" && -w "$gx_npu/max_freq" ]]; then
    if printf '%s' "$gx_npu_old_max" > "$gx_npu/min_freq" && printf '%s' "$gx_npu_old_max" > "$gx_npu/max_freq"; then
        echo 'Requested NPU min=max at existing maximum; per-token NPU frequency still recorded.' > "$gx_results/npu-control.log"
    else
        echo 'NPU frequency request failed; restore policy and retain interleaved repeated measurement.' > "$gx_results/npu-control.log"
        printf '%s' "$gx_npu_old_min" > "$gx_npu/min_freq" || true
        printf '%s' "$gx_npu_old_max" > "$gx_npu/max_freq" || true
    fi
else
    echo 'NPU devfreq policy not writable; retain interleaved repeated measurement.' > "$gx_results/npu-control.log"
fi
{ date -Is; uname -a; echo '--- NPU devfreq ---'; echo 'governor'; cat "$gx_npu/governor"; echo 'min'; cat "$gx_npu/min_freq"; echo 'max'; cat "$gx_npu/max_freq"; echo 'cur'; cat "$gx_npu/cur_freq"; for gx_p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$gx_p"; cat "$gx_p/scaling_governor" "$gx_p/scaling_min_freq" "$gx_p/scaling_max_freq" "$gx_p/scaling_cur_freq"; done; } > "$gx_results/environment.log"
cmake -S "$gx_project" -B "$gx_build" -DUPSTREAM_ROOT="$gx_upstream" -DCMAKE_BUILD_TYPE=Release > "$gx_results/configure.log" 2>&1
cmake --build "$gx_build" -j4 > "$gx_results/build.log" 2>&1
ctest --test-dir "$gx_build" --output-on-failure > "$gx_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
# Isolate the new register recipe before any dependent recurrence. Failure stops.
RKNPU_CORE_MASK=1 timeout 90 taskset -c 4 "$gx_build/gdn_experiments" native-smoke "$gx_results/native-smoke" > "$gx_results/native-smoke.log" 2>&1
for gx_core in 1 2 4; do
    RKNPU_CORE_MASK=$gx_core timeout 300 taskset -c 4 "$gx_build/gdn_experiments" run "$gx_results/core-$gx_core" > "$gx_results/core-$gx_core.log" 2>&1
done
sha256sum "$gx_build/gdn_experiments" > "$gx_results/binary.sha256"

# Regression of the pre-existing GDN API and failure/guard contract.
RKNPU_CORE_MASK=1 timeout 180 taskset -c 4 "$gx_build/gdn_probe" "$gx_results/legacy-regression" chain 20 > "$gx_results/legacy-regression.log" 2>&1
python3 "$gx_project/tools/python/gdn_gate.py" "$gx_results/legacy-regression" "$gx_results/legacy-regression.log" 20 chain > "$gx_results/legacy-regression.json"
