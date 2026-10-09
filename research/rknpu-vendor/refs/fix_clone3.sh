#!/bin/bash
# 走 Mac 上的代理重做三个大仓库（之前的 blobless 只检出根目录，是失败的）
set -u
export http_proxy="http://10.8.1.13:7897"
export https_proxy="http://10.8.1.13:7897"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
GIT="git -c advice.detachedHead=false -c http.version=HTTP/1.1 -c http.lowSpeedLimit=50 -c http.lowSpeedTime=180"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 先清掉三个残缺目录
for d in rknpu rknn-llm rknn-toolkit2; do [ -e "$d" ] && { log "清理残缺 $d"; rm -rf "$d"; }; done

clone_plain() { # $1=dir $2=url
  local d=$1 u=$2
  for a in 1 2 3; do
    [ -d "$d/.git" ] && { log "已可用 $d"; return 0; }
    log "整包浅克隆 尝试$a: $d"
    if timeout 2400 $GIT clone --depth 1 --single-branch -q "$u" "$d" >/tmp/_p.out 2>&1; then
      log "  ok $d $(git -C "$d" rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"; return 0
    fi
    log "  失败: $(tail -2 /tmp/_p.out | tr '\n' ' ')"; rm -rf "$d"; sleep 10
  done
  log "  !! 放弃 $d"; return 1
}

clone_sparse() { # $1=dir $2=url ; 其余=paths
  local d=$1 u=$2; shift 2
  for a in 1 2 3; do
    [ -d "$d/.git" ] && { log "已可用 $d"; return 0; }
    log "blobless+sparse 尝试$a: $d"
    rm -rf "$d"
    if timeout 2400 $GIT clone --depth 1 --filter=blob:none --sparse -q "$u" "$d" >/tmp/_s.out 2>&1; then
      if ( cd "$d" && timeout 2400 $GIT sparse-checkout set "$@" >>/tmp/_s.out 2>&1 ); then
        log "  ok $d $(git -C "$d" rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"; return 0
      fi
    fi
    log "  失败: $(tail -2 /tmp/_s.out | tr '\n' ' ')"; rm -rf "$d"; sleep 10
  done
  log "  !! blobless 失败，$d 退化整包"
  clone_plain "$d" "$u"
}

clone_plain  rknpu    https://github.com/rockchip-linux/rknpu.git
clone_plain  rknn-llm https://github.com/airockchip/rknn-llm.git
clone_sparse rknn-toolkit2 https://github.com/airockchip/rknn-toolkit2.git \
    doc rknpu2/examples rknpu2/runtime/Linux/librknn_api/include \
    rknn-toolkit2/doc rknn-toolkit2/examples rknn-toolkit-lite2
log "=== 完成 ==="
