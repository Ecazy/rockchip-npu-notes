#!/bin/bash
# 网络对 blobless（promisor 按需取）不友好：改为整包 --depth 1 浅克隆。
# rknn-toolkit2 太大，先试 blobless+sparse，失败再退化为整包。
set -u
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
# 放宽低速保护 + 强制 HTTP/1.1（HTTP/2 在抖动链路上更容易坏包）
GIT="git -c advice.detachedHead=false -c http.version=HTTP/1.1 \
     -c http.lowSpeedLimit=50 -c http.lowSpeedTime=120 -c http.postBuffer=524288000"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

clone_plain() { # $1=dir $2=url
  local d=$1 u=$2
  for a in 1 2 3; do
    if [ -d "$d/.git" ] && git -C "$d" rev-parse --verify HEAD >/dev/null 2>&1; then
      log "已可用 $d"; return 0; fi
    rm -rf "$d"
    log "整包浅克隆 尝试$a: $d"
    if timeout 2400 $GIT clone --depth 1 --single-branch -q "$u" "$d" >/tmp/_cp.out 2>&1; then
      log "  ok $d $(git -C "$d" rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"; return 0
    fi
    log "  失败: $(tail -2 /tmp/_cp.out | tr '\n' ' ')"; sleep 15
  done
  log "  !! 放弃 $d"; return 1
}

clone_sparse_retry() { # $1=dir $2=url ; 其余=paths
  local d=$1 u=$2; shift 2
  for a in 1 2 3; do
    if [ -d "$d/.git" ] && git -C "$d" sparse-checkout list >/dev/null 2>&1 \
       && [ "$(git -C "$d" sparse-checkout list 2>/dev/null | wc -l)" -gt 0 ]; then
      log "已可用 $d"; return 0; fi
    rm -rf "$d"
    log "blobless+sparse 尝试$a: $d"
    if timeout 2400 $GIT clone --depth 1 --filter=blob:none --sparse -q "$u" "$d" >/tmp/_cs.out 2>&1; then
      if ( cd "$d" && timeout 2400 $GIT sparse-checkout set "$@" >>/tmp/_cs.out 2>&1 ); then
        log "  ok $d $(git -C "$d" rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"
        return 0
      fi
    fi
    log "  失败: $(tail -2 /tmp/_cs.out | tr '\n' ' ')"; sleep 15
  done
  log "  !! blobless 三次失败，$d 退化为整包浅克隆"
  clone_plain "$d" "$u"
}

clone_plain rknpu    https://github.com/rockchip-linux/rknpu.git
clone_plain rknn-llm https://github.com/airockchip/rknn-llm.git
clone_sparse_retry rknn-toolkit2 https://github.com/airockchip/rknn-toolkit2.git \
    doc rknpu2/examples rknpu2/runtime/Linux/librknn_api/include \
    rknn-toolkit2/doc rknn-toolkit2/examples rknn-toolkit-lite2
log "=== 完成 ==="
