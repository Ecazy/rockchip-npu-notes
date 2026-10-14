#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd/workspace/rknpu-open-validation-20261014
exec 8>/tmp/rknpu_lowlevel_submit.lock
exec 9>/tmp/rk3588_npu_submit.lock
flock -n 8
flock -n 9
paths=(/sys/class/devfreq/fdab0000.npu/governor /sys/class/devfreq/dmc/governor /sys/devices/system/cpu/cpufreq/policy4/scaling_governor /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)
old=()
for p in "${paths[@]}"; do old+=("$(cat "$p")"); done
restore() { for i in "${!paths[@]}"; do echo "${old[$i]}" > "${paths[$i]}"; done; }
trap restore EXIT
for p in "${paths[@]}"; do echo performance > "$p"; done
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
{
 date -Is
 for p in "${paths[@]}"; do echo "$p $(cat "$p")"; done
 cat /sys/class/devfreq/fdab0000.npu/cur_freq /sys/class/devfreq/dmc/cur_freq /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
} > controlled-metadata.log
for r in 1 2 3; do
 taskset -c 4-7 ./build/bench_llm_i8 --loops 20 --warmup 3 --npu-mask 1 > controlled-open-$r.log 2>&1
 VALIDATE_NATIVE=1 taskset -c 4-7 ./build/validate_official_llm > controlled-off-native-$r.log 2>&1
 taskset -c 4-7 ./build/validate_official_llm > controlled-off-normal-$r.log 2>&1
 echo "Completed controlled round $r"
done
taskset -c 4-7 ./build/bench_llm_i8 --loops 20 --warmup 3 --npu-mask 7 > controlled-open-3core.log 2>&1
cat /sys/class/thermal/thermal_zone*/temp >> controlled-metadata.log
