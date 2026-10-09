#!/bin/bash
# 修复 rknpu / rknn-llm 的稀疏检出，并补做 rknn-toolkit2
set -u
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
GIT="git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

fix_sparse() { # $1=dir ; 其余=paths
  local d=$1; shift
  log "稀疏检出 $d : $*"
  ( cd "$d" || exit 1
    if $GIT sparse-checkout set "$@" >/tmp/_sp.out 2>&1; then
      log "  ok $d  检出文件 $(git ls-files | wc -l) 个  $(du -sh . | cut -f1)"
    else
      log "  FAIL $d : $(tail -2 /tmp/_sp.out | tr '\n' ' ')"
    fi )
}

fix_sparse rknpu        drivers rknn
fix_sparse rknn-llm     doc rkllm-toolkit rkllm-runtime rknpu-driver scripts examples
fix_sparse mesa         src/gallium/drivers/rocket src/gallium/drivers/teflon

# rknn-toolkit2：仓库约 3.8 GB，只取文档、示例与头文件
d=rknn-toolkit2
u=https://github.com/airockchip/rknn-toolkit2.git
if [ ! -d "$d/.git" ]; then
  log "clone(blobless+sparse) $d"
  if timeout 2400 $GIT clone --depth 1 --filter=blob:none --sparse -q "$u" "$d" >/tmp/_rk.out 2>&1; then
    log "  clone ok"
  else
    log "  FAIL clone: $(tail -2 /tmp/_rk.out | tr '\n' ' ')"; rm -rf "$d"
  fi
fi
if [ -d "$d/.git" ]; then
  fix_sparse "$d" doc rknpu2/examples rknpu2/runtime/Linux/librknn_api/include \
                  rknn-toolkit2/doc rknn-toolkit2/examples rknn-toolkit-lite2
fi
log "=== 修复阶段完成 ==="
