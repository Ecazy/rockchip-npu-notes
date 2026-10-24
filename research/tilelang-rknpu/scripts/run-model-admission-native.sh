#!/usr/bin/env bash
# Remote RK3588 complete-model evaluator. Credentials never reach this script.
set -uo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 <campaign-dir> <build-dir> <results-dir>" >&2
  exit 2
fi
campaign=$(readlink -f "$1")
build=$(readlink -f "$2")
results=$(readlink -f "$3")
[[ ! -e "$results" ]] || { echo 'Fresh results directory required' >&2; exit 2; }
mkdir -p "$results"
source_root=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
python_pkg="$source_root/python"
eval_script="$campaign/tools/evaluate_models.py"
[[ -f "$eval_script" ]] || { echo "missing model evaluator: $eval_script" >&2; exit 2; }

export RKNPU_CHIP=rk3588
export ROCKET_CHIP=rk3588
export PYTHONPATH="$python_pkg${PYTHONPATH:+:$PYTHONPATH}"
export RKNPU_PLAN_LIB="$build/librknpu_plan.so"
export LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
  echo "collected_utc=$(date -u +%FT%TZ)"
  uname -srm
  echo "board_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
  echo "chip=rk3588"
  echo "driver=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || true)"
  echo "npu_freq=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null || true)"
  echo "npu_load=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || true)"
  echo "gcc=$(gcc --version | head -1)"
  echo "cmake=$(cmake --version | head -1)"
  echo "python=$(python3 -V 2>&1)"
} > "$results/environment.log"
printf 'campaign manifest sha256: '; sha256sum "$campaign/manifest.json"
printf 'source package sha256: '; sha256sum "$python_pkg/rknpu_backend/models.py"

mkdir -p "$build"
if ! cmake -S "$source_root/../custom-ops" -B "$build" \
    -DUPSTREAM_ROOT="$(cd "$source_root/../upstream-components" && pwd)" \
    -DCMAKE_BUILD_TYPE=Release > "$results/configure.log" 2>&1; then
  echo 'configure failed' >&2
  exit 1
fi
if ! cmake --build "$build" --target rknpu_plan -j4 > "$results/build.log" 2>&1; then
  echo 'build failed' >&2
  exit 1
fi
[[ -f "$RKNPU_PLAN_LIB" ]] || { echo 'librknpu_plan.so missing' >&2; exit 1; }
sha256sum "$RKNPU_PLAN_LIB" > "$results/library.sha256"

overall=0
for mask in 1 2 4; do
  core_dir="$results/core-$mask"
  mkdir -p "$core_dir"
  {
    echo "core_mask=$mask"
    echo "collected_utc=$(date -u +%FT%TZ)"
    cat /sys/kernel/debug/rknpu/load 2>/dev/null || true
  } > "$core_dir/environment.log"
  dmesg 2>/dev/null | tail -n 120 > "$core_dir/kernel-before.log" || true
  limit=10000
  [[ "$mask" == "1" ]] || limit=512
  set +e
  RKNPU_CORE_MASK="$mask" python3 "$eval_script" \
      --backend board --lib "$RKNPU_PLAN_LIB" \
      --plans-dir "$campaign/plans" --weights-dir "$campaign/weights" \
      --data-dir "$campaign/data" --global-limit "$limit" \
      --profile-samples 512 \
      --out "$core_dir/model_eval.json" > "$core_dir/runner.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$core_dir/exit.txt"
  cat /sys/kernel/debug/rknpu/load 2>/dev/null >> "$core_dir/environment.log" || true
  dmesg 2>/dev/null | tail -n 160 > "$core_dir/kernel-after.log" || true
  if (( rc != 0 )); then overall=1; fi
  if [[ "$mask" == "1" ]]; then
    if ! python3 "$campaign/tools/benchmark_cpu_models.py" \
        --data-dir "$campaign/data" --weights-dir "$campaign/weights" \
        --out "$core_dir/cpu_benchmark.json" --repeats 2 \
        > "$core_dir/cpu_benchmark.log" 2>&1; then
      overall=1
    fi
  fi
done
exit "$overall"
