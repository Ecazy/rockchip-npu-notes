#!/usr/bin/env bash
# Remote numeric observation/build steps. Build happens once; calibration runs
# before the Host freezes thresholds; acceptance runs only after the frozen spec
# is uploaded back to the campaign.
set -uo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 <build|calibration|acceptance> <campaign> <build-dir> <results>" >&2
  exit 2
fi
mode=$1
campaign=$(readlink -f "$2")
build=$(readlink -f "$3")
results=$(readlink -f "$4")
source_root=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
python_pkg="$source_root/python"
mkdir -p "$results"
export RKNPU_CHIP=rk3588
export ROCKET_CHIP=rk3588
export RKNPU_CORE_MASK=1
export PYTHONPATH="$python_pkg${PYTHONPATH:+:$PYTHONPATH}"
export RKNPU_PLAN_LIB="$build/librknpu_plan.so"
export LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case "$mode" in
  build)
    {
      echo "collected_utc=$(date -u +%FT%TZ)"
      uname -srm
      echo "board_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
      echo "driver=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || true)"
      echo "npu_freq=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null || true)"
      echo "core_mask=1"
      echo "gcc=$(gcc --version | head -1)"
      echo "cmake=$(cmake --version | head -1)"
      echo "python=$(python3 -V 2>&1)"
    } > "$results/environment.log"
    if ! cmake -S "$source_root/../custom-ops" -B "$build" \
        -DUPSTREAM_ROOT="$(cd "$source_root/../upstream-components" && pwd)" \
        -DCMAKE_BUILD_TYPE=Release > "$results/configure.log" 2>&1; then
      echo "configure failed" >&2; exit 1
    fi
    if ! cmake --build "$build" --target rknpu_plan -j4 \
        > "$results/build.log" 2>&1; then
      echo "build failed" >&2; exit 1
    fi
    sha256sum "$build/librknpu_plan.so" > "$results/library.sha256"
    ;;
  calibration)
    mkdir -p "$results/calibration"
    manifest_args=()
    if [[ -n "${RK_NUMERIC_MANIFEST_MAP:-}" ]]; then
      IFS=',' read -r -a manifest_pairs <<< "$RK_NUMERIC_MANIFEST_MAP"
      for pair in "${manifest_pairs[@]}"; do
        [[ "$pair" =~ ^[a-zA-Z0-9_.-]+=[a-zA-Z0-9_.-]+$ ]] || {
          echo "invalid RK_NUMERIC_MANIFEST_MAP entry: $pair" >&2; exit 2; }
        manifest_args+=(--manifest-key "$pair")
      done
    fi
    set +e
    python3 "$campaign/tools/run_numeric_observation.py" \
      --mode calibration \
      --plans-dir "$campaign/plans" --weights-dir "$campaign/weights" \
      --data-dir "$campaign/data" --lib "$build/librknpu_plan.so" \
      "${manifest_args[@]}" \
      --out "$results/calibration/observation.npz" \
      > "$results/calibration/runner.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$results/calibration/exit.txt"
    exit "$rc"
    ;;
  acceptance)
    mkdir -p "$results/acceptance"
    manifest_args=()
    if [[ -n "${RK_NUMERIC_MANIFEST_MAP:-}" ]]; then
      IFS=',' read -r -a manifest_pairs <<< "$RK_NUMERIC_MANIFEST_MAP"
      for pair in "${manifest_pairs[@]}"; do
        [[ "$pair" =~ ^[a-zA-Z0-9_.-]+=[a-zA-Z0-9_.-]+$ ]] || {
          echo "invalid RK_NUMERIC_MANIFEST_MAP entry: $pair" >&2; exit 2; }
        manifest_args+=(--manifest-key "$pair")
      done
    fi
    set +e
    python3 "$campaign/tools/run_numeric_observation.py" \
      --mode acceptance \
      --spec "$campaign/numeric/${RK_NUMERIC_SPEC:-numeric_acceptance_train.json}" \
      --plans-dir "$campaign/plans" --weights-dir "$campaign/weights" \
      --data-dir "$campaign/data" --lib "$build/librknpu_plan.so" \
      "${manifest_args[@]}" \
      --out "$results/acceptance/observation.npz" \
      > "$results/acceptance/runner.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$results/acceptance/exit.txt"
    exit "$rc"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
