#!/bin/bash
# 带超时与重试的补充克隆（网络偶发卡死时用）
set -u
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
GIT="git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

try() { # $1=dir ; 其余=clone 参数
  local d=$1; shift
  for attempt in 1 2 3; do
    if [ -d "$d/.git" ]; then log "已存在 $d"; return 0; fi
    rm -rf "$d"
    log "尝试 $attempt: $d"
    if timeout 900 $GIT clone -q "$@" "$d" 2>&1 | tail -2; then
      log "  ok $d $(cd "$d" && $GIT rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"
      return 0
    fi
    log "  失败，10s 后重试"; sleep 10
  done
  log "  !! 放弃 $d"
  return 1
}

try onnc --depth 1 https://github.com/ONNC/onnc.git
try mesa-rk3576 --depth 1 https://github.com/gahingwoo/mesa-rk3576.git

# 文档指定的固定 commit：blobless 拿历史后精确 checkout
for spec in "npu|https://github.com/allbilly/npu.git|e02707ca9b010372f1c506c7f3a2fa2c9f2beb8e" \
            "tinygrad|https://github.com/allbilly/tinygrad.git|4ccf0a87f96d38424f71063f7e37a61b43484243"; do
  d=${spec%%|*}; rest=${spec#*|}; u=${rest%%|*}; sha=${rest##*|}
  if [ -d "$d/.git" ]; then log "已存在 $d"; continue; fi
  rm -rf "$d"
  log "固定 commit: $d @ ${sha:0:8}"
  if timeout 1200 $GIT clone -q --filter=blob:none --no-checkout "$u" "$d" 2>&1 | tail -2; then
    ( cd "$d" && $GIT checkout -q "$sha" 2>&1 | tail -2 && \
      log "  ok $d $(git rev-parse --short HEAD) $(du -sh . | cut -f1)" ) || log "  !! checkout 失败 $d"
  else
    log "  !! clone 失败 $d"
  fi
done
log "=== 补充克隆完成 ==="
