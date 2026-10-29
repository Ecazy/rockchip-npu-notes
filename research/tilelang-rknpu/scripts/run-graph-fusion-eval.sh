#!/usr/bin/env bash
# Focused RK3588 evaluator for graph/submit fusion candidates.
# Usage: ./scripts/run-graph-fusion-eval.sh setup
#        RK_GF_LIMIT=512 RK_GF_ROUNDS=2 ./scripts/run-graph-fusion-eval.sh run <out-dir>
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
: "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
: "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
: "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
REMOTE_BASE=${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}
REMOTE_STATE="$HERE/.data/graph_fusion_remote_path"
LIMIT=${RK_GF_LIMIT:-512}
ROUNDS=${RK_GF_ROUNDS:-2}
COREMASK=${RKNPU_CORE_MASK:-1}
CPU=${RK_GF_CPU:-6}
ARMS=${RK_GF_ARMS:-":default:_deviceffn:_deviceffn_fused"}
PLANS_NAME=${RK_GF_PLANS_NAME:-compiled_graph_fusion}

board_env() {
  askpass=$(mktemp /tmp/rk-gf-askpass-XXXXXX)
  printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
  chmod 700 "$askpass"
  export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
  export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
  SSH=(setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o NumberOfPasswordPrompts=1 "$TARGET")
  SCP=(setsid -w scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
}
cleanup() { unset RK_BOARD_PASSWORD; [[ -n "${askpass:-}" ]] && rm -f -- "$askpass"; }
trap cleanup EXIT

cmd_setup() {
  board_env
  local stamp remote src pkg
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  remote="$REMOTE_BASE/graph-fusion-$stamp"
  src=$(mktemp /tmp/rk-gf-src-XXXXXX.tgz)
  pkg=$(mktemp /tmp/rk-gf-pkg-XXXXXX.tgz)
  plans=$(mktemp /tmp/rk-gf-plans-XXXXXX.tgz)
  tar --exclude='.git' --exclude='build*' --exclude='results' --exclude='__pycache__' \
      --exclude='*.so' --exclude='*.o' -czf "$src" -C "$REPO/research" \
      custom-ops upstream-components/rocket-userspace upstream-components/rknpu-submit \
      tilelang-rknpu/python tilelang-rknpu/capabilities \
      tilelang-rknpu/models/tiny_vit.npz tilelang-rknpu/models/mnist_mlp.npz \
      tilelang-rknpu/models/numeric_acceptance.json tilelang-rknpu/models/numeric_golden.npz \
      tilelang-rknpu/tools/evaluate_models.py \
      tilelang-rknpu/tools/evaluate_rmsnorm_approx.py \
      tilelang-rknpu/tools/evaluate_rope_device.py \
      tilelang-rknpu/tools/evaluate_pv_o_device.py \
      tilelang-rknpu/tools/evaluate_smollm_rope_device.py \
      tilelang-rknpu/tools/evaluate_pv_o_b8_device.py \
      tilelang-rknpu/tools/evaluate_smollm_rope_residency.py \
      tilelang-rknpu/tools/evaluate_smollm_rope_indexed.py \
      tilelang-rknpu/.data/models/SmolLM2-135M-Instruct/config.json
  tar -czf "$plans" -C "$HERE/models" "$PLANS_NAME"
  tar -czf "$pkg" -C "$HERE/.data/mnist" t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
  "${SSH[@]}" "mkdir -p '$remote'" >/dev/null
  "${SCP[@]}" "$src" "$TARGET:$remote/source.tgz" >/dev/null
  "${SCP[@]}" "$pkg" "$TARGET:$remote/data.tgz" >/dev/null
  "${SCP[@]}" "$plans" "$TARGET:$remote/plans.tgz" >/dev/null
  "${SSH[@]}" "cd '$remote' && mkdir -p data plans build && tar -xzf source.tgz && tar -xzf data.tgz -C data && tar -xzf plans.tgz -C plans --strip-components=1 && cmake -S custom-ops -B build -DUPSTREAM_ROOT='$remote/upstream-components' -DCMAKE_BUILD_TYPE=Release > configure.log 2>&1 && cmake --build build --target rknpu_plan -j4 > build.log 2>&1 && sha256sum build/librknpu_plan.so" >&2
  printf '%s\n' "$remote" > "$REMOTE_STATE"
  echo "remote=$remote"
}

cmd_run() {
  local out=${1:?output dir required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  mkdir -p "$out"
  for ((round=1; round<=ROUNDS; round++)); do
    local IFS=':'; read -r -a arms <<< "$ARMS"; unset IFS
    for arm in "${arms[@]}"; do
      [[ -n "$arm" ]] || continue
      local suffix="$arm"
      [[ "$arm" == "default" ]] && suffix=""
      local safe=${suffix#_}; [[ -n "$safe" ]] || safe=default
      echo "round=$round arm=$safe suffix=$suffix" | tee -a "$out/run.log"
      "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_models.py' --backend board --lib '$remote/build/librknpu_plan.so' --plans-dir '$remote/plans' --weights-dir '$remote/tilelang-rknpu/models' --data-dir '$remote/data' --models vit --global-limit $LIMIT --profile-samples $LIMIT --hybrid-suffix '$suffix' --out '$remote/out-r${round}-${safe}.json'" >> "$out/run.log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then echo "arm failed rc=$rc" | tee -a "$out/run.log"; return $rc; fi
      "${SCP[@]}" "$TARGET:$remote/out-r${round}-${safe}.json" "$out/" >/dev/null || return 1
      "${SCP[@]}" "$TARGET:$remote/out-r${round}-${safe}.vit_b1.logits.npy" "$out/" >/dev/null || true
      "${SCP[@]}" "$TARGET:$remote/out-r${round}-${safe}.vit_b8.logits.npy" "$out/" >/dev/null || true
    done
  done
}

cmd_rmsnorm_calibrate() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && rm -f rmsnorm-approx-spec.json rmsnorm-approx-spec.sha256 && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_rmsnorm_approx.py' --mode calibrate --backend board --lib '$remote/build/librknpu_plan.so' --plan '$remote/plans/tiny_vit_b1_rmsnorm1_approx.rkplan.json' --spec '$remote/rmsnorm-approx-spec.json' --out '$remote/rmsnorm-calibrate.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/rmsnorm-calibrate.json" "$out" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/rmsnorm-approx-spec.json" "$out.spec.json" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/rmsnorm-approx-spec.sha256" "$out.spec.sha256" >/dev/null
}

cmd_rmsnorm_verify() {
  local out=${1:?output file required}
  local spec=${2:?spec file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  local base; base=$(basename "$spec")
  "${SCP[@]}" "$spec" "$TARGET:$remote/$base" >/dev/null
  "${SCP[@]}" "$spec.sha256" "$TARGET:$remote/$base.sha256" >/dev/null
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_rmsnorm_approx.py' --mode verify --backend board --lib '$remote/build/librknpu_plan.so' --plan '$remote/plans/tiny_vit_b1_rmsnorm1_approx.rkplan.json' --spec '$remote/$base' --out '$remote/rmsnorm-verify.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/rmsnorm-verify.json" "$out" >/dev/null || true
}

cmd_rope_probe() {
  local out=${1:?output file required}
  local style=${2:-interleaved}
  local rotary=${3:-64}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_rope_device.py' --backend board --lib '$remote/build/librknpu_plan.so' --style '$style' --rotary-dim $rotary --calls 200 --warmup 20 --out '$remote/rope-device.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/rope-device.json" "$out" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/rope-device.rkplan.json" "$out.rkplan.json" >/dev/null || true
}

cmd_pv_o_probe() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_pv_o_device.py' --backend board --lib '$remote/build/librknpu_plan.so' --calls 200 --warmup 20 --out '$remote/pv-o-device.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/pv-o-device.json" "$out" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/pv-o-device.rkplan.json" "$out.rkplan.json" >/dev/null || true
}

cmd_smollm_rope_residency() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_smollm_rope_residency.py' --backend board --lib '$remote/build/librknpu_plan.so' --layers 30 --calls 10 --out '$remote/smollm-rope-residency.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/smollm-rope-residency.json" "$out" >/dev/null
  for label in prefill_q prefill_k decode_q decode_k; do
    "${SCP[@]}" "$TARGET:$remote/smollm-rope-residency.$label.rkplan.json" "$out.$label.rkplan.json" >/dev/null || true
  done
}

cmd_smollm_rope_indexed() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_smollm_rope_indexed.py' --backend board --lib '$remote/build/librknpu_plan.so' --calls 100 --out '$remote/smollm-rope-indexed.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/smollm-rope-indexed.json" "$out" >/dev/null
  for label in q k; do
    "${SCP[@]}" "$TARGET:$remote/smollm-rope-indexed.$label.rkplan.json" "$out.$label.rkplan.json" >/dev/null || true
  done
}

cmd_smollm_rope_probe() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_smollm_rope_device.py' --backend board --lib '$remote/build/librknpu_plan.so' --calls 50 --warmup 10 --out '$remote/smollm-rope-device.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/smollm-rope-device.json" "$out" >/dev/null
  for label in prefill_q prefill_k decode_q decode_k; do
    "${SCP[@]}" "$TARGET:$remote/smollm-rope-device.$label.rkplan.json" "$out.$label.rkplan.json" >/dev/null || true
  done
}

cmd_pv_o_b8_probe() {
  local out=${1:?output file required}
  [[ -f "$REMOTE_STATE" ]] || { echo 'run setup first' >&2; exit 2; }
  board_env
  local remote; remote=$(<"$REMOTE_STATE")
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=$COREMASK && taskset -c $CPU python3 '$remote/tilelang-rknpu/tools/evaluate_pv_o_b8_device.py' --backend board --lib '$remote/build/librknpu_plan.so' --calls 100 --warmup 10 --out '$remote/pv-o-b8-device.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/pv-o-b8-device.json" "$out" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/pv-o-b8-device.rkplan.json" "$out.rkplan.json" >/dev/null || true
}




case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  run) shift; cmd_run "$@" ;;
  rmsnorm-calibrate) shift; cmd_rmsnorm_calibrate "$@" ;;
  rmsnorm-verify) shift; cmd_rmsnorm_verify "$@" ;;
  rope-probe) shift; cmd_rope_probe "$@" ;;
  pv-o-probe) shift; cmd_pv_o_probe "$@" ;;
  pv-o-b8-probe) shift; cmd_pv_o_b8_probe "$@" ;;
  smollm-rope-probe) shift; cmd_smollm_rope_probe "$@" ;;
  smollm-rope-residency) shift; cmd_smollm_rope_residency "$@" ;;
  smollm-rope-indexed) shift; cmd_smollm_rope_indexed "$@" ;;
  *) echo "usage: $0 {setup|run <out-dir>|rmsnorm-calibrate <out-file>|rmsnorm-verify <out-file> <spec>|rope-probe <out-file> [style] [rotary]|pv-o-probe <out-file>|pv-o-b8-probe <out-file>|smollm-rope-probe <out-file>|smollm-rope-residency <out-file>|smollm-rope-indexed <out-file>}" >&2; exit 2 ;;
esac
