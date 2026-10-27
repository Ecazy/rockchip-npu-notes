#!/usr/bin/env bash
# Board worker for run-finite-scan-ab.sh. Outputs remain under the campaign dir.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
case "${1:-}" in
  build)
    [[ $# == 6 ]] || exit 2
    source_root=$(readlink -f "$2"); campaign=$(readlink -f "$3")
    fast=$(readlink -m "$4"); reference=$(readlink -m "$5"); cpu=$6
    mkdir -p "$fast" "$reference" "$source_root/build-logs" "$source_root/artifacts"
    set +e
    cmake -S "$source_root/custom-ops" -B "$fast" \
      -DUPSTREAM_ROOT="$source_root/upstream-components" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      > "$source_root/build-logs/fast-config.stdout" 2> "$source_root/build-logs/fast-config.stderr"
    rc=$?; echo "$rc" > "$source_root/build-logs/fast-config.exit"
    ((rc==0)) || exit "$rc"
    cmake --build "$fast" --target rknpu_plan -j4 \
      > "$source_root/build-logs/fast-build.stdout" 2> "$source_root/build-logs/fast-build.stderr"
    rc=$?; echo "$rc" > "$source_root/build-logs/fast-build.exit"
    ((rc==0)) || exit "$rc"
    cmake -S "$source_root/custom-ops" -B "$reference" \
      -DUPSTREAM_ROOT="$source_root/upstream-components" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_C_FLAGS=-DRK_PLAN_FINITE_SCAN_REFERENCE \
      > "$source_root/build-logs/reference-config.stdout" 2> "$source_root/build-logs/reference-config.stderr"
    rc=$?; echo "$rc" > "$source_root/build-logs/reference-config.exit"
    ((rc==0)) || exit "$rc"
    cmake --build "$reference" --target rknpu_plan -j4 \
      > "$source_root/build-logs/reference-build.stdout" 2> "$source_root/build-logs/reference-build.stderr"
    rc=$?; echo "$rc" > "$source_root/build-logs/reference-build.exit"
    ((rc==0)) || exit "$rc"
    mkdir -p "$source_root/artifacts"
    cp "$fast/CMakeCache.txt" "$source_root/artifacts/fast-CMakeCache.txt"
    cp "$reference/CMakeCache.txt" "$source_root/artifacts/reference-CMakeCache.txt"
    cp "$fast/compile_commands.json" "$source_root/artifacts/fast-compile_commands.json"
    cp "$reference/compile_commands.json" "$source_root/artifacts/reference-compile_commands.json"
    cp "$fast/CMakeFiles/rk_plan_runtime_obj.dir/flags.make" "$source_root/artifacts/fast-runtime-flags.make"
    cp "$reference/CMakeFiles/rk_plan_runtime_obj.dir/flags.make" "$source_root/artifacts/reference-runtime-flags.make"
    python3 - "$source_root/artifacts/fast-runtime-flags.make" \
      "$source_root/artifacts/reference-runtime-flags.make" \
      > "$source_root/artifacts/compile-flag-comparison.txt" <<'PY'
import re,sys
def fields(path):
    out={}
    for line in open(path):
        m=re.match(r'(C_DEFINES|C_INCLUDES|C_FLAGS) = (.*)',line.rstrip())
        if m: out[m.group(1)]=m.group(2)
    return out
a,b=map(fields,sys.argv[1:])
keys=sorted(set(a)|set(b))
macro='-DRK_PLAN_FINITE_SCAN_REFERENCE'
fast_text=' '.join(a.values()); reference_text=' '.join(b.values())
if macro in fast_text or macro not in reference_text:
    print('macro assertion failed: fast must omit, reference must contain RK_PLAN_FINITE_SCAN_REFERENCE')
    raise SystemExit(1)
for k in keys:
    av,bv=a.get(k,''),b.get(k,'')
    normalized=re.sub(r'\s*-DRK_PLAN_FINITE_SCAN_REFERENCE\b','',bv).strip()
    print(f'{k}: '+('ONLY_REFERENCE_HAS_RK_PLAN_FINITE_SCAN_REFERENCE' if av==normalized and av!=bv else 'MATCH' if av==bv else 'MISMATCH'))
if any(a.get(k,'') != re.sub(r'\s*-DRK_PLAN_FINITE_SCAN_REFERENCE\b','',b.get(k,'')).strip() for k in keys):
    raise SystemExit(1)
PY
    rc=$?; echo "$rc" > "$source_root/build-logs/compile-flag-check.exit"
    ((rc==0)) || exit "$rc"
    sha256sum "$fast/librknpu_plan.so" > "$source_root/artifacts/fast-library.sha256"
    sha256sum "$reference/librknpu_plan.so" > "$source_root/artifacts/reference-library.sha256"
    gcc -std=c11 -O2 -I "$source_root/custom-ops/include" \
      -I "$source_root/custom-ops/src" \
      -I "$source_root/upstream-components/rocket-userspace/include" \
      "$source_root/tilelang-rknpu/tools/runtime_layers/runtime_layer_probe.c" \
      -L "$fast" -lrknpu_plan -lm -o "$source_root/artifacts/runtime_layer_probe" \
      > "$source_root/build-logs/probe-build.stdout" 2> "$source_root/build-logs/probe-build.stderr"
    rc=$?; echo "$rc" > "$source_root/build-logs/probe-build.exit"
    ((rc==0)) || exit "$rc"
    sha256sum "$source_root/artifacts/runtime_layer_probe" > "$source_root/artifacts/runtime-probe.sha256"
    npu_gov=$(cat /sys/class/devfreq/*npu*/governor 2>/dev/null | head -1)
    npu_hz=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null | head -1)
    dmc_gov=$(cat /sys/class/devfreq/dmc/governor 2>/dev/null | head -1)
    dmc_hz=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null | head -1)
    cpu_gov=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor 2>/dev/null)
    cpu_khz=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq 2>/dev/null)
    {
      echo "collected_utc=$(date -u +%FT%TZ)"; uname -srm
      echo "board_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
      echo "driver=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || true)"
      echo "npu_governor=$npu_gov"; echo "npu_freq=$npu_hz"
      echo "dmc_governor=$dmc_gov"; echo "dmc_freq=$dmc_hz"
      echo "cpu_governor=$cpu_gov"; echo "cpu_freq=$cpu_khz"
      echo "cpu=$cpu core_mask=1"; taskset -pc "$$"; cat /proc/loadavg
    } > "$source_root/environment.log"
    if [[ "$npu_gov" != performance || "$npu_hz" != 1000000000 ||
          "$dmc_gov" != performance || "$dmc_hz" != 2112000000 ||
          "$cpu_gov" != performance || "$cpu_khz" != 2304000 ]]; then
      echo 'fixed RK3588 frequency condition mismatch; observed values saved in environment.log' >&2
      exit 3
    fi
    ;;
  isolated)
    [[ $# == 11 ]] || exit 2
    source_root=$(readlink -f "$2"); campaign=$(readlink -f "$3")
    fast=$(readlink -m "$4"); reference=$(readlink -m "$5")
    case_name=$6; arm=$7; round=$8; cpu=$9; mask=${10}; out=${11}
    case "$arm" in fast) libdir=$fast;; reference) libdir=$reference;; *) exit 2;; esac
    mkdir -p "$out/$case_name/r$round"
    result="$out/$case_name/r$round/$arm"
    probe="$source_root/artifacts/runtime_layer_probe"
    set +e
    taskset -c "$cpu" env RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 \
      RKNPU_CORE_MASK="$mask" RK_PLAN_PREPARED_VALIDATION=0 \
      LD_LIBRARY_PATH="$libdir" "$probe" "$campaign/cases/$case_name" b 1 "$result.bin" \
      > "$result.stdout" 2> "$result.stderr"
    rc=$?; echo "$rc" > "$result.exit"; exit "$rc"
    ;;
  full)
    [[ $# == 9 ]] || exit 2
    source_root=$(readlink -f "$2"); campaign=$(readlink -f "$3")
    fast=$(readlink -m "$4"); reference=$(readlink -m "$5")
    batch=$6; cpu=$7; mask=$8; out=$9
    dir="$out/b$batch"; mkdir -p "$dir"
    set +e
    taskset -c "$cpu" env RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 \
      RKNPU_CORE_MASK="$mask" RK_PLAN_PREPARED_VALIDATION=0 \
      LD_LIBRARY_PATH="$fast:$reference" PYTHONPATH="$source_root/tilelang-rknpu/python" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 "$source_root/tilelang-rknpu/tools/runtime_layers/full_model_finite_scan_ab.py" \
        --plans-dir "$campaign" --weights-dir "$campaign" --data-dir "$campaign" \
        --lib-fast "$fast/librknpu_plan.so" --lib-reference "$reference/librknpu_plan.so" \
        --batch "$batch" --limit 256 --rounds 7 --out "$dir/result.json" \
        > "$dir/run.stdout" 2> "$dir/run.stderr"
    rc=$?; echo "$rc" > "$dir/run.exit"; exit "$rc"
    ;;
  *) echo 'usage: build|isolated|full ...' >&2; exit 2;;
esac
