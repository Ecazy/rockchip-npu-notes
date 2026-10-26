#!/usr/bin/env bash
# RK3588 final-delivery worker: build one arm, run one full five-profile
# evaluation, or capture the frozen packed numeric acceptance observation.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)

record_env() {
  local out=$1
  {
    echo "collected_utc=$(date -u +%FT%TZ)"
    uname -srm
    echo "board_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    echo "chip=${RKNPU_CHIP:-rk3588}"
    echo "driver=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || true)"
    echo "npu_freq=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null || true)"
    echo "npu_load=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || true)"
    echo "loadavg=$(cat /proc/loadavg)"
    echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null || true)"
    echo "gcc=$(gcc --version | head -1)"
    echo "cmake=$(cmake --version | head -1)"
    echo "python=$(python3 -V 2>&1)"
  } > "$out"
}

case "${1:-}" in
  build)
    [[ $# -eq 3 ]] || { echo 'usage: build <source-root> <build-dir>' >&2; exit 2; }
    source_root=$(readlink -f "$2")
    build=$(readlink -f "$3")
    mkdir -p "$build"
    cmake -S "$source_root/custom-ops" -B "$build" \
      -DUPSTREAM_ROOT="$source_root/upstream-components" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target rknpu_plan -j4
    ;;
  eval)
    [[ $# -eq 10 ]] || {
      echo 'usage: eval <source-root> <build-dir> <plans-dir> <weights-dir> <data-dir> <numeric-spec> <out-dir> <cpu> <limit>' >&2
      exit 2; }
    source_root=$(readlink -f "$2")
    build=$(readlink -f "$3")
    plans=$(readlink -f "$4")
    weights=$(readlink -f "$5")
    data=$(readlink -f "$6")
    numeric=$(readlink -f "$7")
    out_arg=$8
    mkdir -p "$out_arg"
    out=$(readlink -f "$out_arg")
    cpu=$9
    limit=${10}
    rm -rf "$out"
    mkdir -p "$out"
    record_env "$out/environment.log"
    lib="$build/librknpu_plan.so"
    [[ -f "$lib" ]] || { echo "missing $lib" >&2; exit 1; }
    sha256sum "$lib" > "$out/library.sha256"
    set +e
    start_ns=$(date +%s%N)
    taskset -c "$cpu" env \
      RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 \
      RKNPU_PLAN_LIB="$lib" LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$source_root/tilelang-rknpu/python${PYTHONPATH:+:$PYTHONPATH}" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 "$source_root/tilelang-rknpu/tools/evaluate_models.py" \
        --backend board --lib "$lib" \
        --plans-dir "$plans" --weights-dir "$weights" --data-dir "$data" \
        --global-limit "$limit" --profile-samples "$limit" \
        --numeric-spec "$numeric" --out "$out/model_eval.json" \
        > "$out/runner.log" 2>&1
    rc=$?
    end_ns=$(date +%s%N)
    set -e
    python3 - "$start_ns" "$end_ns" > "$out/wall_seconds.txt" <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)
PY
    echo "$rc" > "$out/exit.txt"
    record_env "$out/environment.after.log"
    cat "$out/wall_seconds.txt" 2>/dev/null || true
    exit "$rc"
    ;;
  calibrate)
    [[ $# -eq 8 ]] || {
      echo 'usage: calibrate <source-root> <build-dir> <plans-dir> <weights-dir> <data-dir> <out-dir> <cpu>' >&2
      exit 2; }
    source_root=$(readlink -f "$2")
    build=$(readlink -f "$3")
    plans=$(readlink -f "$4")
    weights=$(readlink -f "$5")
    data=$(readlink -f "$6")
    out_arg=$7
    mkdir -p "$out_arg"
    out=$(readlink -f "$out_arg")
    cpu=$8
    rm -rf "$out"
    mkdir -p "$out"
    record_env "$out/environment.log"
    lib="$build/librknpu_plan.so"
    taskset -c "$cpu" env \
      RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 \
      RKNPU_PLAN_LIB="$lib" LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$source_root/tilelang-rknpu/python${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$source_root/tilelang-rknpu/tools/run_numeric_observation.py" \
        --mode calibration --plans-dir "$plans" --weights-dir "$weights" \
        --data-dir "$data" --lib "$lib" \
        --out "$out/observation.npz" > "$out/runner.log" 2>&1
    ;;
  observe)
    [[ $# -eq 9 ]] || {
      echo 'usage: observe <source-root> <build-dir> <plans-dir> <weights-dir> <data-dir> <spec> <out-dir> <cpu>' >&2
      exit 2; }
    source_root=$(readlink -f "$2")
    build=$(readlink -f "$3")
    plans=$(readlink -f "$4")
    weights=$(readlink -f "$5")
    data=$(readlink -f "$6")
    spec=$(readlink -f "$7")
    out_arg=$8
    mkdir -p "$out_arg"
    out=$(readlink -f "$out_arg")
    cpu=$9
    rm -rf "$out"
    mkdir -p "$out"
    record_env "$out/environment.log"
    lib="$build/librknpu_plan.so"
    taskset -c "$cpu" env \
      RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 \
      RKNPU_PLAN_LIB="$lib" LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$source_root/tilelang-rknpu/python${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$source_root/tilelang-rknpu/tools/run_numeric_observation.py" \
        --mode acceptance --plans-dir "$plans" --weights-dir "$weights" \
        --data-dir "$data" --spec "$spec" --lib "$lib" \
        --out "$out/observation.npz" > "$out/runner.log" 2>&1
    ;;
  *)
    echo "usage: $0 {build|eval|observe} ..." >&2
    exit 2
    ;;
esac
