#!/usr/bin/env bash
set -euo pipefail
gdn_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gdn_build=${1:-$gdn_project/build-gdn}
gdn_results=${2:-$gdn_project/artifacts/gdn-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$gdn_results" ]] || exit 1
mkdir -p "$gdn_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
gdn_upstream=${UPSTREAM_ROOT:-$gdn_project/../upstream-components}
python3 "$gdn_project/tools/python/summarize.py" --sources "$gdn_project" "$gdn_upstream" > "$gdn_results/source-manifest.json"
{ date -Is; uname -a; lscpu; cat /sys/class/devfreq/fdab0000.npu/cur_freq;
  for gdn_policy in /sys/devices/system/cpu/cpufreq/policy*; do
    printf '%s ' "$gdn_policy"; cat "$gdn_policy/scaling_governor" "$gdn_policy/scaling_cur_freq";
  done; } > "$gdn_results/environment.log" 2>&1
journalctl -k -n 0 --show-cursor --no-pager > "$gdn_results/kernel-before.log" 2>&1 || true
gdn_cursor=$(sed -n 's/^-- cursor: //p' "$gdn_results/kernel-before.log")
gdn_capture() {
    if [[ -n "$gdn_cursor" ]]; then journalctl -k --after-cursor="$gdn_cursor" --no-pager > "$gdn_results/kernel-interval.log" 2>&1 || true; fi
}
trap gdn_capture EXIT
cmake -S "$gdn_project" -B "$gdn_build" -DUPSTREAM_ROOT="$gdn_upstream" -DCMAKE_BUILD_TYPE=Release > "$gdn_results/configure.log" 2>&1
cmake --build "$gdn_build" -j4 > "$gdn_results/build.log" 2>&1
ctest --test-dir "$gdn_build" --output-on-failure > "$gdn_results/host-tests.log" 2>&1
sha256sum "$gdn_build/gdn_probe" > "$gdn_results/binary.sha256"
export ROCKET_CHIP=rk3588
for gdn_mask in 1 2 4; do
    for gdn_mode in chain separate; do
        gdn_prefix="$gdn_results/core-$gdn_mask-$gdn_mode"
        RKNPU_CORE_MASK=$gdn_mask timeout 180 taskset -c 4 "$gdn_build/gdn_probe" "$gdn_prefix" "$gdn_mode" 20 > "$gdn_prefix.log" 2>&1
        python3 "$gdn_project/tools/python/gdn_gate.py" "$gdn_prefix" "$gdn_prefix.log" 20 "$gdn_mode" > "$gdn_prefix-summary.json"
    done
done
# In-place alias tests are isolated from production recurrence. Nonzero/hang is
# retained as a diagnostic failure and never enables aliasing in the API.
for gdn_mask in 1 2 4; do
    gdn_alias_status=0
    RKNPU_CORE_MASK=$gdn_mask timeout 60 taskset -c 4 "$gdn_build/gdn_probe" inplace > "$gdn_results/inplace-$gdn_mask.log" 2>&1 || gdn_alias_status=$?
    printf '%s\n' "$gdn_alias_status" > "$gdn_results/inplace-$gdn_mask.exit"
    # Stop after a device error/timeout; do not submit into potentially wedged HW.
    if [[ "$gdn_alias_status" != 0 ]]; then break; fi
done
gdn_capture
python3 "$gdn_project/tools/python/gdn_campaign.py" "$gdn_results" > "$gdn_results/campaign.json"
printf 'GDN hardware campaign PASS\n' > "$gdn_results/PASS"
