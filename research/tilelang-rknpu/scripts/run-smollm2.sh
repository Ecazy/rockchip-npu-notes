#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Reproducible SmolLM2-135M-Instruct fetch/compile/board/chat driver.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
MODEL_DIR="$HERE/.data/models/SmolLM2-135M-Instruct"
PACKAGE="$HERE/.data/models/smollm2_135m_instruct_package"
REMOTE_STATE="$HERE/.data/smollm2_remote_path"

host_env() {
  export PYTHONPATH="$HERE:$HERE/python:$HERE/.pydeps-p0:$HERE/.src/p0-build-src/tilelang:$HERE/.src/p0-source-env"
  export LD_LIBRARY_PATH="$HERE/.src/p0-build-tilelang/lib:$HERE/.src/p0-build-tvmffi-python/lib:$HERE/.pydeps-p0/z3/lib"
  export TILELANG_CACHE_DIR="$HERE/.src/p0-tilelang-cache"
  export RKNPU_CHIP=rk3588
}

board_env() {
  if [[ -f "$REPO/.env" ]]; then source "$REPO/.env"; fi
  : "${ORANGE_PI_HOST:?Set ORANGE_PI_HOST in $REPO/.env}"
  : "${ORANGE_PI_USER:?Set ORANGE_PI_USER in $REPO/.env}"
  : "${ORANGE_PI_PASSWORD:?Set ORANGE_PI_PASSWORD in $REPO/.env}"
  local askpass
  askpass=$(mktemp /tmp/rk-smollm-askpass-XXXXXX)
  printf '%s\n' '#!/bin/sh' 'printf "%s\n" "${RK_BOARD_PASSWORD:?}"' > "$askpass"
  chmod 700 "$askpass"
  export RK_BOARD_PASSWORD="$ORANGE_PI_PASSWORD"
  export SSH_ASKPASS="$askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  TARGET="$ORANGE_PI_USER@$ORANGE_PI_HOST"
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
            -o NumberOfPasswordPrompts=1)
  SSH=(setsid -w ssh "${SSH_OPTS[@]}" "$TARGET")
  SCP=(setsid -w scp "${SSH_OPTS[@]}")
  ASKPASS_FILE="$askpass"
}

board_cleanup() {
  unset RK_BOARD_PASSWORD
  [[ -n "${ASKPASS_FILE:-}" ]] && rm -f -- "$ASKPASS_FILE"
}

cmd_fetch() {
  host_env
  python3 "$HERE/tools/fetch_smollm2.py" --out "$MODEL_DIR"
}

cmd_compile() {
  host_env
  python3 "$HERE/tools/compile_smollm2.py" \
    --model-dir "$MODEL_DIR" --out "$PACKAGE"
}

cmd_host_check() {
  host_env
  export RKNPU_PLAN_LIB="$REPO/research/custom-ops/build-host-mvp/librknpu_plan.so"
  export RKNPU_PLAN_MOCK_LIB="$REPO/research/custom-ops/build-host-mvp/librknpu_plan_mock.so"
  python3 -m pytest -q "$HERE/tests"
  ctest --test-dir "$REPO/research/custom-ops/build-host-mvp" \
        --output-on-failure
}

cmd_setup_board() {
  [[ -d "$PACKAGE" ]] || { echo "compile the package first" >&2; exit 2; }
  board_env
  trap board_cleanup EXIT
  local stamp remote source_bundle package_bundle baseline_bundle=""
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  remote="${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}/smollm-work-$stamp"
  source_bundle=$(mktemp /tmp/rk-smollm-src-XXXXXX.tgz)
  package_bundle=$(mktemp /tmp/rk-smollm-pkg-XXXXXX.tgz)
  if [[ -n "${RK_SMOLLM_BASELINE_PACKAGE:-}" ]]; then
    [[ -f "$RK_SMOLLM_BASELINE_PACKAGE/manifest.json" ]] || {
      echo "baseline package manifest missing" >&2; exit 2; }
    baseline_bundle=$(mktemp /tmp/rk-smollm-base-XXXXXX.tgz)
  fi
  trap 'rm -f -- "${source_bundle:-}" "${package_bundle:-}" "${baseline_bundle:-}"; board_cleanup' EXIT
  tar --exclude='.git' --exclude='build*' --exclude='results' \
      --exclude='artifacts' --exclude='__pycache__' --exclude='*.so' \
      --exclude='*.o' -czf "$source_bundle" -C "$REPO/research" \
      custom-ops upstream-components/rocket-userspace \
      upstream-components/rknpu-submit tilelang-rknpu/python \
      tilelang-rknpu/capabilities tilelang-rknpu/models/tilelang_matmul.py \
      tilelang-rknpu/tools/run_smollm2.py \
      tilelang-rknpu/tools/run_schedule_experiment.py \
      tilelang-rknpu/tools/tune_schedule.py \
      tilelang-rknpu/tools/run_smollm_device_rope.py
  tar -czf "$package_bundle" -C "$HERE/.data/models" \
      smollm2_135m_instruct_package
  if [[ -n "$baseline_bundle" ]]; then
    tar -czf "$baseline_bundle" -C "$(dirname "$RK_SMOLLM_BASELINE_PACKAGE")" \
        "$(basename "$RK_SMOLLM_BASELINE_PACKAGE")"
  fi
  "${SSH[@]}" "mkdir -p '$remote/package' '$remote/build'" >/dev/null
  "${SCP[@]}" "$source_bundle" "$TARGET:$remote/source.tgz" >/dev/null
  "${SCP[@]}" "$package_bundle" "$TARGET:$remote/package.tgz" >/dev/null
  if [[ -n "$baseline_bundle" ]]; then
    "${SCP[@]}" "$baseline_bundle" "$TARGET:$remote/baseline.tgz" >/dev/null
    "${SSH[@]}" "mkdir -p '$remote/package-baseline' && tar -xzf '$remote/baseline.tgz' -C '$remote/package-baseline' --strip-components=1" >/dev/null
  fi
  "${SSH[@]}" "cd '$remote' && tar -xzf source.tgz && tar -xzf package.tgz -C package --strip-components=1 && cmake -S custom-ops -B build -DUPSTREAM_ROOT='$remote/upstream-components' -DCMAKE_BUILD_TYPE=Release > configure.log 2>&1 && cmake --build build --target rknpu_plan -j4 > build.log 2>&1 && sha256sum build/librknpu_plan.so"
  printf '%s\n' "$remote" > "$REMOTE_STATE"
  echo "remote=$remote"
}

cmd_board_run() {
  local request=$1 out=$2
  [[ -f "$REMOTE_STATE" ]] || { echo "run setup-board first" >&2; exit 2; }
  [[ -f "$request" ]] || { echo "request JSON missing: $request" >&2; exit 2; }
  board_env
  trap board_cleanup EXIT
  local remote stamp_request remote_out
  remote=$(<"$REMOTE_STATE")
  stamp_request="request-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
  # A fresh remote output name is mandatory: reusing "out-board" would mix a
  # later 5-window PPL run with the earlier 19-run acceptance board directory.
  remote_out="out-${stamp_request%.json}-$(basename "$out")"
  # Trusted local A/B control only; values are expanded into the remote command.
  local runner_args=${RK_SMOLLM_RUNNER_ARGS:-}
  local remote_package=${RK_SMOLLM_REMOTE_PACKAGE:-package}
  local schedule_arg=""
  local schedule_policy=${RK_SMOLLM_SCHEDULE_POLICY:-}
  if [[ -n "$schedule_policy" ]]; then
    [[ -f "$schedule_policy" ]] || {
      echo "schedule policy missing: $schedule_policy" >&2; exit 2; }
    local remote_policy="schedule-policy-${stamp_request}"
    "${SCP[@]}" "$schedule_policy" "$TARGET:$remote/$remote_policy" >/dev/null
    schedule_arg="--schedule-policy '$remote/$remote_policy'"
  fi
  "${SCP[@]}" "$request" "$TARGET:$remote/$stamp_request" >/dev/null
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' && export RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' && export LD_LIBRARY_PATH='$remote/build' && export RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && python3 tilelang-rknpu/tools/run_smollm2.py --package '$remote_package' --request '$stamp_request' --out '$remote_out' $runner_args $schedule_arg"
  rm -rf "$out"
  mkdir -p "$out"
  "${SCP[@]}" -r "$TARGET:$remote/$remote_out/." "$out/" >/dev/null
}

cmd_chat() {
  local prompt=$1 max_new=${2:-32} out
  host_env
  python3 - "$MODEL_DIR" "$prompt" "$max_new" /tmp/smollm-chat-request.json <<'PY'
import json, sys
from transformers import AutoTokenizer
model, prompt, n, out = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(model)
ids = tok.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=True,
    add_generation_prompt=True)
open(out, "w").write(json.dumps({"runs": [{
    "mode": "generate", "token_ids": ids,
    "max_new_tokens": int(n), "stop_on_eos": True, "repeat": 1
}]}, indent=2))
PY
  out="$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)-smollm2-chat"
  cmd_board_run /tmp/smollm-chat-request.json "$out"
  python3 - "$MODEL_DIR" "$out" "$prompt" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
model, out, prompt = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(model)
result = json.loads((Path(out) / "run-00/result.json").read_text())["runs"][0]
text = tok.decode(result["generated_ids"])
summary = {"prompt": prompt, "prompt_ids": result["prompt_ids"],
           "generated_ids": result["generated_ids"],
           "generated_text": text,
           "stopped_by_eos": result["stopped_by_eos"],
           "prefill_us": result["prefill_us"],
           "decode_us": result["decode_us"]}
(Path(out) / "chat.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1, ensure_ascii=False))
PY
}

usage() {
  cat >&2 <<EOF
usage: $0 {fetch|compile|host-check|setup-board|board-run <request.json> <out-dir>|schedule-bench <plans-dir> <out.json>|schedule-tune <plans-dir> <out-dir>|chat <prompt> [max_new]}
EOF
}

cmd_schedule_bench() {
  local plans=$1 out=$2
  [[ -f "$REMOTE_STATE" ]] || { echo "run setup-board first" >&2; exit 2; }
  [[ -f "$plans/manifest.json" ]] || {
    echo "schedule plan manifest missing: $plans/manifest.json" >&2; exit 2; }
  board_env
  trap board_cleanup EXIT
  local remote stamp remote_plans remote_out bundle
  remote=$(<"$REMOTE_STATE")
  stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
  remote_plans="schedule-plans-$stamp"
  remote_out="schedule-result-$stamp.json"
  bundle=$(mktemp /tmp/rk-smollm-schedule-XXXXXX.tgz)
  trap 'rm -f -- "${bundle:-}"; board_cleanup' EXIT
  tar -czf "$bundle" -C "$(dirname "$plans")" "$(basename "$plans")"
  "${SCP[@]}" "$bundle" "$TARGET:$remote/schedule.tgz" >/dev/null
  "${SSH[@]}" "cd '$remote' && rm -rf '$remote_plans' && mkdir -p '$remote_plans' && tar -xzf schedule.tgz -C '$remote_plans' --strip-components=1 && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && python3 '$remote/tilelang-rknpu/tools/run_schedule_experiment.py' run --plans '$remote/$remote_plans' --backend board --repeats 5 --out '$remote/$remote_out'"
  "${SCP[@]}" "$TARGET:$remote/$remote_out" "$out" >/dev/null
}

cmd_schedule_tune() {
  local plans=$1 out=$2
  [[ -f "$REMOTE_STATE" ]] || { echo "run setup-board first" >&2; exit 2; }
  [[ -f "$plans/manifest.json" ]] || {
    echo "tuning manifest missing: $plans/manifest.json" >&2; exit 2; }
  board_env
  trap board_cleanup EXIT
  local remote stamp remote_plans bundle
  remote=$(<"$REMOTE_STATE")
  stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
  remote_plans="tune-plans-$stamp"
  bundle=$(mktemp /tmp/rk-smollm-tune-XXXXXX.tgz)
  trap 'rm -f -- "${bundle:-}"; board_cleanup' EXIT
  mkdir -p "$out"
  tar -czf "$bundle" -C "$(dirname "$plans")" "$(basename "$plans")"
  "${SCP[@]}" "$bundle" "$TARGET:$remote/tune.tgz" >/dev/null
  "${SSH[@]}" "cd '$remote' && rm -rf '$remote_plans' && mkdir -p '$remote_plans' && tar -xzf tune.tgz -C '$remote_plans' --strip-components=1 && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && python3 '$remote/tilelang-rknpu/tools/tune_schedule.py' run --plans '$remote/$remote_plans' --package '$remote/package' --out '$remote/tune-run.json'"
  "${SCP[@]}" "$TARGET:$remote/tune-run.json" "$out/tune-run.json" >/dev/null
  # Separate interleaved baseline/winner confirmation process.
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && python3 '$remote/tilelang-rknpu/tools/tune_schedule.py' confirm --plans '$remote/$remote_plans' --package '$remote/package' --case attention_qk_prefill --rounds 5 --out '$remote/tune-confirm.json'"
  "${SCP[@]}" "$TARGET:$remote/tune-confirm.json" "$out/tune-confirm.json" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/$remote_plans/policy.json" "$out/policy.json" >/dev/null
  "${SCP[@]}" "$TARGET:$remote/$remote_plans/policy-summary.json" "$out/policy-summary.json" >/dev/null
  # New SSH process: this is the new-process replay, not a re-use of the
  # Python process that selected the winner.
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && python3 '$remote/tilelang-rknpu/tools/tune_schedule.py' replay --plans '$remote/$remote_plans' --package '$remote/package' --out '$remote/tune-replay.json'"
  "${SCP[@]}" "$TARGET:$remote/tune-replay.json" "$out/tune-replay.json" >/dev/null
}

cmd_device_rope_probe() {
  local prompt_json=${1:?prompt json required}
  local out=${2:?output file required}
  local policy=${3:-}
  [[ -f "$REMOTE_STATE" ]] || { echo "run setup-board first" >&2; exit 2; }
  [[ -f "$prompt_json" ]] || { echo "prompt json missing: $prompt_json" >&2; exit 2; }
  board_env
  trap board_cleanup EXIT
  local remote base
  remote=$(<"$REMOTE_STATE")
  base=$(basename "$prompt_json")
  "${SCP[@]}" "$prompt_json" "$TARGET:$remote/$base" >/dev/null
  local policy_arg=""
  if [[ "${RK_SMOLLM_DECODE_INDEXED:-0}" == "1" ]]; then
    policy_arg+=" --decode-indexed"
  fi
  if [[ -n "$policy" ]]; then
    local pbase; pbase=$(basename "$policy")
    "${SCP[@]}" "$policy" "$TARGET:$remote/$pbase" >/dev/null
    policy_arg+=" --policy '$remote/$pbase'"
  fi
  "${SSH[@]}" "cd '$remote' && export PYTHONPATH='$remote/tilelang-rknpu/python' RKNPU_PLAN_LIB='$remote/build/librknpu_plan.so' LD_LIBRARY_PATH='$remote/build' RKNPU_CHIP=rk3588 ROCKET_CHIP=rk3588 RKNPU_CORE_MASK=1 && taskset -c 6 python3 '$remote/tilelang-rknpu/tools/run_smollm_device_rope.py' --backend board --lib '$remote/build/librknpu_plan.so' --package '$remote/package' --prompt-json '$remote/$base' $policy_arg --out '$remote/smollm-device-rope.json'" 2>&1 | tee "$out.log"
  "${SCP[@]}" "$TARGET:$remote/smollm-device-rope.json" "$out" >/dev/null
}

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  compile) shift; cmd_compile "$@" ;;
  host-check) shift; cmd_host_check "$@" ;;
  setup-board) shift; cmd_setup_board "$@" ;;
  board-run) shift; cmd_board_run "${1:?request}" "${2:?out-dir}" ;;
  schedule-bench) shift; cmd_schedule_bench "${1:?plans dir}" "${2:?out json}" ;;
  schedule-tune) shift; cmd_schedule_tune "${1:?plans dir}" "${2:?out dir}" ;;
  device-rope-probe) shift; cmd_device_rope_probe "${1:?prompt json}" "${2:?out file}" "${3:-}" ;;
  chat) shift; cmd_chat "${1:?prompt}" "${2:-32}" ;;
  *) usage; exit 2 ;;
esac
