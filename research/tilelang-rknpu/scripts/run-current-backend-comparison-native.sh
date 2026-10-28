#!/usr/bin/env bash
# Board-side helpers for the current/vendor/CPU comparison. Credentials stay local.
set -euo pipefail

record_environment() {
  local root=$1 out=$2
  {
    echo "collected_utc=$(date -u +%FT%TZ)"
    uname -srm
    echo "board_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    echo "driver=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || true)"
    echo "cpu6_governor=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor 2>/dev/null || true)"
    echo "cpu6_freq_khz=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq 2>/dev/null || true)"
    echo "npu_governor=$(cat /sys/class/devfreq/*npu*/governor 2>/dev/null | head -1)"
    echo "npu_freq_hz=$(cat /sys/kernel/debug/rknpu/freq 2>/dev/null | head -1)"
    echo "dmc_governor=$(cat /sys/class/devfreq/dmc/governor 2>/dev/null || true)"
    echo "dmc_freq_hz=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null || true)"
    echo "npu_core_mask=1"
    echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null || true)"
    echo "loadavg=$(cat /proc/loadavg)"
    echo "python=$(python3 -V 2>&1)"
    echo "current_library=$(sha256sum "$root/campaign/current/librknpu_plan.so")"
    echo "vendor_runtime=$(sha256sum "$root/campaign/vendor/runtime/2.3.2/librknnrt.so")"
    echo "system_rknnrt_mount_before=$(if mountpoint -q /usr/lib/librknnrt.so; then echo mounted; else echo unmounted; fi)"
    echo "system_rknnrt_hash_before=$(sha256sum /usr/lib/librknnrt.so 2>/dev/null || true)"
  } > "$out"
}

if [[ ${1:-} == install-site ]]; then
  [[ $# == 4 ]] || { echo 'usage: install-site <root> <old-site> <out-dir>' >&2; exit 2; }
  ROOT=$(readlink -f "$2"); OLD_SITE=$(readlink -f "$3"); OUT=$(readlink -f "$4")
  mkdir -p "$ROOT/site" "$OUT"
  WHEEL=("$ROOT/campaign/vendor/wheel/"*.whl)
  [[ -f ${WHEEL[0]} ]] || { echo 'Toolkit Lite wheel missing' >&2; exit 2; }
  set +e
  python3 -m pip install --quiet --no-deps --target "$ROOT/site" "${WHEEL[0]}" \
    > "$OUT/pip.stdout" 2> "$OUT/pip.stderr"
  rc=$?
  set -e
  echo "$rc" > "$OUT/pip.exit"
  (( rc == 0 )) || exit "$rc"
  for name in ruamel ruamel_yaml-0.19.1.dist-info psutil psutil-7.2.2.dist-info; do
    [[ -e "$OLD_SITE/$name" ]] || { echo "missing reused Python dependency: $name" >&2; exit 2; }
    cp -a "$OLD_SITE/$name" "$ROOT/site/"
  done
  PYTHONPATH="$ROOT/site" python3 -c 'import rknnlite, ruamel.yaml, psutil; print("rknnlite", rknnlite.__file__); print("ruamel", ruamel.yaml.__version__); print("psutil", psutil.__version__)' \
    > "$OUT/vendor-site.stdout" 2> "$OUT/vendor-site.stderr"
  echo 0 > "$OUT/vendor-site.exit"
  exit 0
fi

[[ ${1:-} == run ]] || { echo 'usage: run <root> <mode> <limit> <rounds> <out> [profiles...]' >&2; exit 2; }
[[ $# -ge 6 ]] || exit 2
ROOT=$(readlink -f "$2")
MODE=$3
LIMIT=$4
ROUNDS=$5
OUT=$6
shift 6
mkdir -p "$(dirname "$OUT")"
LIB="$ROOT/campaign/current/librknpu_plan.so"
RUNTIME="$ROOT/campaign/vendor/runtime/2.3.2/librknnrt.so"
SITE="$ROOT/site"
PYTHONPATH="$SITE:$ROOT/source/tilelang-rknpu/python"
LD_LIBRARY_PATH="$(dirname "$LIB"):$SITE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[[ $(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor) == performance ]] || {
  echo 'CPU6 governor mismatch' >&2; exit 3; }
[[ $(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq) == 2304000 ]] || {
  echo 'CPU6 frequency mismatch' >&2; exit 3; }
[[ $(cat /sys/class/devfreq/*npu*/governor | head -1) == performance ]] || {
  echo 'NPU governor mismatch' >&2; exit 3; }
[[ $(cat /sys/kernel/debug/rknpu/freq | head -1) == 1000000000 ]] || {
  echo 'NPU frequency mismatch' >&2; exit 3; }
[[ $(cat /sys/class/devfreq/dmc/governor) == performance ]] || {
  echo 'DMC governor mismatch' >&2; exit 3; }
[[ $(cat /sys/class/devfreq/dmc/cur_freq) == 2112000000 ]] || {
  echo 'DMC frequency mismatch' >&2; exit 3; }

if [[ ! -f "$LIB" ]]; then
  mkdir -p "$(dirname "$LIB")" "$ROOT/build"
  cmake -S "$ROOT/source/custom-ops" -B "$ROOT/build" \
    -DUPSTREAM_ROOT="$ROOT/source/upstream-components" \
    -DCMAKE_BUILD_TYPE=Release > "$OUT.configure.log" 2>&1
  cmake --build "$ROOT/build" --target rknpu_plan -j4 > "$OUT.build.log" 2>&1
  cp "$ROOT/build/librknpu_plan.so" "$LIB"
fi
[[ -f "$LIB" ]] || { echo 'current runtime library missing' >&2; exit 6; }

VENDOR_ARM=0
SYSTEM_HASH_BEFORE=''
cleanup_mount() {
  local rc=$?
  trap - EXIT
  if (( VENDOR_ARM )); then
    if ! umount /usr/lib/librknnrt.so; then
      echo 'FAIL: could not unmount the temporary Toolkit Lite runtime' >> "$OUT.runtime-mount.txt"
      (( rc != 0 )) || rc=90
    else
      local after
      after=$(sha256sum /usr/lib/librknnrt.so | awk '{print $1}')
      printf 'system_rknnrt_hash_after=%s\n' "$after" >> "$OUT.runtime-mount.txt"
      if [[ "$after" != "$SYSTEM_HASH_BEFORE" ]]; then
        echo 'FAIL: system runtime hash changed after unmount' >> "$OUT.runtime-mount.txt"
        (( rc != 0 )) || rc=91
      fi
    fi
  fi
  exit "$rc"
}
trap cleanup_mount EXIT
if [[ "$MODE" == vendor || "$MODE" == all ]]; then
  if mountpoint -q /usr/lib/librknnrt.so; then
    echo 'system librknnrt path is already a mount; refusing to replace it' >&2
    exit 4
  fi
  SYSTEM_HASH_BEFORE=$(sha256sum /usr/lib/librknnrt.so | awk '{print $1}')
  EXPECTED_VENDOR_HASH=$(sha256sum "$RUNTIME" | awk '{print $1}')
  [[ "$EXPECTED_VENDOR_HASH" == d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8 ]] || {
    echo 'unexpected Toolkit Lite 2.3.2 runtime hash' >&2; exit 5; }
  mount --bind "$RUNTIME" /usr/lib/librknnrt.so
  VENDOR_ARM=1
  MOUNTED_HASH=$(sha256sum /usr/lib/librknnrt.so | awk '{print $1}')
  if [[ "$MOUNTED_HASH" != "$EXPECTED_VENDOR_HASH" ]]; then
    umount /usr/lib/librknnrt.so
    echo 'mounted Toolkit Lite runtime hash mismatch' >&2
    exit 5
  fi
  findmnt -n -o SOURCE,TARGET /usr/lib/librknnrt.so > "$OUT.runtime-mount.txt"
  printf 'system_rknnrt_hash_before=%s\nmounted_librknnrt_hash=%s\n' \
    "$SYSTEM_HASH_BEFORE" "$MOUNTED_HASH" >> "$OUT.runtime-mount.txt"
fi

ARGS=(--plans-dir "$ROOT/campaign/plans" --weights-dir "$ROOT/campaign/weights"
      --data-dir "$ROOT/campaign/data" --rknn-dir "$ROOT/campaign/vendor/rknn"
      --lib "$LIB" --limit "$LIMIT" --rounds "$ROUNDS" --out "$OUT" --mode "$MODE")
for profile in "$@"; do ARGS+=(--profile "$profile"); done
set +e
export RKNN_RUNTIME_EXPECTED=/usr/lib/librknnrt.so
taskset -c 6 env RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 \
  RKNPU_PLAN_LIB="$LIB" PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 "$ROOT/source/tilelang-rknpu/tools/benchmark_current_backends.py" \
    "${ARGS[@]}" > "$OUT.stdout" 2> "$OUT.stderr"
rc=$?
echo "$rc" > "$OUT.exit"
exit "$rc"
