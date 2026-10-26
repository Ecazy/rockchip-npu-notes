#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Remote (RK3588) side of the bridge-envelope campaign.
#
#   build      <arm> <target> <build-dir> <out-dir>   build one arm's library
#   measure    <arm> <lib> <model> <batch> <round> <runs> <campaign> <out-dir>
#   primitives <build-dir> <out-dir>                  primitive cost floor
#
# Build and measure are separate so the caller can build every arm first and then
# interleave the measurements round by round against the same machine state.
set -uo pipefail
mode=${1:-}
source_root=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
python_pkg="$source_root/python"
export RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588
export RKNPU_CORE_MASK=${RKNPU_CORE_MASK:-1}
case "$mode" in
  build)
    [[ $# -eq 5 ]] || { echo "usage: $0 build <arm> <target> <build-dir> <out-dir>" >&2; exit 2; }
    arm=$2; target=$3; build=$(readlink -f "$4"); out=$(readlink -f "$5")
    mkdir -p "$build" "$out"
    if ! cmake -S "$source_root/../custom-ops" -B "$build" \
        -DUPSTREAM_ROOT="$(cd "$source_root/../upstream-components" && pwd)" \
        -DCMAKE_BUILD_TYPE=Release > "$out/configure.log" 2>&1; then
      echo "configure failed for $arm" >&2; exit 1
    fi
    if ! cmake --build "$build" --target "$target" -j4 > "$out/build.log" 2>&1; then
      echo "build failed for $arm" >&2; exit 1
    fi
    exit 0
    ;;
  measure)
    [[ $# -eq 9 ]] || {
      echo "usage: $0 measure <arm> <lib> <model> <batch> <round> <runs> <campaign> <out-dir>" >&2
      exit 2; }
    arm=$2; lib=$(readlink -f "$3"); model=$4; batch=$5; round=$6; runs=$7
    campaign=$(readlink -f "$8"); out=$(readlink -f "$9")
    mkdir -p "$out"
    tag="${arm}_${model}_b${batch}_r${round}"
    cpu=${RK_TIME_CPU:-6}
    export PYTHONPATH="$python_pkg${PYTHONPATH:+:$PYTHONPATH}"
    export RKNPU_PLAN_LIB="$lib"
    {
      echo "arm=$arm"; echo "round=$round"; echo "model=$model"; echo "batch=$batch"
      echo "runs=$runs"; echo "core_mask=$RKNPU_CORE_MASK"; echo "cpu=$cpu"
      echo "collected_utc=$(date -u +%FT%TZ)"
      echo "npu_freq=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null || true)"
      echo "npu_load=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || true)"
      echo "affinity=$(taskset -pc $$ 2>&1 || true)"
      echo "loadavg=$(cat /proc/loadavg)"
    } > "$out/${tag}.env"
    sha256sum "$lib" > "$out/${tag}.library.sha256"
    # Pin to one CPU so both arms see the same core; the profiler records the
    # mask it actually ran under from sched_getaffinity.
    taskset -c "$cpu" python3 "$campaign/tools/profile_bridge_envelope.py" \
        --backend board --lib "$lib" --fd 0 \
        --model "$model" --batch "$batch" \
        --plans-dir "$campaign/plans" --weights-dir "$campaign/weights" \
        --data-dir "$campaign/data" \
        --runs "$runs" --warmup "${RK_ENVELOPE_WARMUP:-8}" \
        --label "${arm}:r${round}" \
        --out "$out/${tag}.json" > "$out/${tag}.log" 2>&1
    rc=$?
    echo "$rc" > "$out/${tag}.exit"
    exit "$rc"
    ;;
  primitives)
    [[ $# -eq 3 ]] || { echo "usage: $0 primitives <build-dir> <out-dir>" >&2; exit 2; }
    build=$(readlink -f "$2"); out=$(readlink -f "$3")
    mkdir -p "$out"
    if cmake --build "$build" --target plan_envelope_probe -j4 \
        >> "$out/build.log" 2>&1; then
      taskset -c "${RK_TIME_CPU:-6}" "$build/plan_envelope_probe" \
          > "$out/primitives.jsonl" 2> "$out/primitives.err" || true
    fi
    exit 0
    ;;
  *)
    echo "usage: $0 {build|measure|primitives} ..." >&2
    exit 2
    ;;
esac
