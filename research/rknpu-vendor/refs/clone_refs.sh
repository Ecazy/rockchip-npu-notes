#!/bin/bash
# 浅克隆文档提到的 RKNPU 相关开源库到 refs/
# 用法： bash clone_refs.sh [phase1|phase2]
set -u
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
GIT="git -c advice.detachedHead=false"

log() { echo "[$(date +%H:%M:%S)] $*"; }

clone_depth1() { # $1=dir $2=url
  local d=$1 u=$2
  if [ -d "$d/.git" ]; then log "跳过（已存在）$d"; return 0; fi
  log "clone --depth 1  $d  <- $u"
  if timeout 1500 $GIT clone --depth 1 -q "$u" "$d" >/tmp/_c.out 2>&1; then
    log "  ok $d  $(cd "$d" && $GIT rev-parse --short HEAD)  $(du -sh "$d" | cut -f1)"
  else
    log "  FAIL $d : $(tail -1 /tmp/_c.out)"
  fi
}

clone_pinned() { # $1=dir $2=url $3=sha
  local d=$1 u=$2 sha=$3
  if [ -d "$d/.git" ]; then log "跳过（已存在）$d"; return 0; fi
  log "clone 固定 commit  $d  <- $u @ ${sha:0:8}"
  $GIT clone --depth 1 --filter=blob:none --no-checkout -q "$u" "$d" 2>&1 | tail -2
  ( cd "$d" || exit 1
    if $GIT fetch --depth 1 -q origin "$sha" 2>/dev/null; then
      $GIT checkout -q FETCH_HEAD && echo "  ok(按 sha 取到) $(git rev-parse --short HEAD)"
    else
      echo "  !! 无法按 sha fetch，改为克隆默认分支 tip"
      $GIT fetch --depth 1 -q origin HEAD && $GIT checkout -q FETCH_HEAD
    fi )
  log "  $d  $(cd "$d" && $GIT rev-parse --short HEAD 2>/dev/null)  $(du -sh "$d" | cut -f1)"
}

clone_sparse() { # $1=dir $2=url $3...=paths
  local d=$1 u=$2; shift 2
  if [ -d "$d/.git" ]; then log "跳过（已存在）$d"; return 0; fi
  log "blobless+sparse  $d  <- $u  paths: $*"
  if ! timeout 1800 $GIT clone --depth 1 --filter=blob:none --sparse -q "$u" "$d" >/tmp/_c.out 2>&1; then
    log "  FAIL(clone) $d : $(tail -1 /tmp/_c.out)"; rm -rf "$d"; return 1
  fi
  ( cd "$d" || exit 1
    if ! $GIT sparse-checkout set "$@" >/tmp/_s.out 2>&1; then
      log "  FAIL(sparse) $d : $(tail -1 /tmp/_s.out)"; exit 1
    fi
    echo "  稀疏检出完成 $(git rev-parse --short HEAD)" )
  log "  $d  $(cd "$d" && $GIT rev-parse --short HEAD 2>/dev/null)  $(du -sh "$d" | cut -f1)"
}

phase1() {
  clone_depth1 rknpu-reverse-engineering https://github.com/phhusson/rknpu-reverse-engineering.git
  clone_depth1 rk3588-npu              https://github.com/mtx512/rk3588-npu.git
  clone_depth1 rk3568-npu              https://github.com/iav/rk3568-npu.git
  clone_depth1 rknpu_ddk               https://github.com/airockchip/rknpu_ddk.git
  clone_depth1 npu-oc-tools            https://github.com/sky-rk3588/npu-oc-tools.git
  clone_depth1 rknputop                https://github.com/ramonbroox/rknputop.git
  clone_depth1 ezrknpu                 https://github.com/Pelochus/ezrknpu.git
  clone_depth1 rkllama                 https://github.com/NotPunchnox/rkllama.git
  clone_depth1 onnc                    https://github.com/ONNC/onnc.git
  clone_depth1 mesa-rk3576             https://github.com/gahingwoo/mesa-rk3576.git
  clone_pinned npu      https://github.com/allbilly/npu.git      e02707ca9b010372f1c506c7f3a2fa2c9f2beb8e
  clone_pinned tinygrad https://github.com/allbilly/tinygrad.git 4ccf0a87f96d38424f71063f7e37a61b43484243
}

phase2() {
  # 大仓库：只取真正有知识价值的路径（二进制 wheel / 预编译库 / 历史全部略过）
  clone_sparse rknpu https://github.com/rockchip-linux/rknpu.git \
      drivers rknn
  clone_sparse rknn-toolkit2 https://github.com/airockchip/rknn-toolkit2.git \
      doc rknpu2/examples rknpu2/runtime/Linux/librknn_api/include \
      rknn-toolkit2/doc rknn-toolkit2/examples rknn-toolkit-lite2
  clone_sparse rknn-llm https://github.com/airockchip/rknn-llm.git \
      doc rkllm-toolkit rkllm-runtime rknpu-driver scripts examples
  clone_sparse mesa https://gitlab.freedesktop.org/mesa/mesa.git \
      src/gallium/drivers/rocket src/gallium/drivers/teflon
}

case "${1:-phase1}" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  *) echo "用法: $0 [phase1|phase2]"; exit 2 ;;
esac
log "=== 完成 ==="
