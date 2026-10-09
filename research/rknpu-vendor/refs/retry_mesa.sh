#!/bin/bash
# 只补 mesa-rk3576（上一次因 early EOF 失败；注意不能把 git 的退出码丢给管道）
set -u
cd /home/maoyu/Documents/code/RKNPU/refs || exit 1
GIT="git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
d=mesa-rk3576
u=https://github.com/gahingwoo/mesa-rk3576.git
for attempt in 1 2 3 4; do
  if [ -d "$d/.git" ] && (cd "$d" && $GIT rev-parse --verify HEAD >/dev/null 2>&1); then
    log "已存在且可用 $d"; break
  fi
  rm -rf "$d"
  log "尝试 $attempt: $d"
  if timeout 1500 $GIT clone -q --depth 1 --single-branch "$u" "$d" >/tmp/mesa_clone.out 2>&1; then
    log "  ok $d $(cd "$d" && $GIT rev-parse --short HEAD) $(du -sh "$d"|cut -f1)"
    break
  else
    log "  失败: $(tail -2 /tmp/mesa_clone.out | tr '\n' ' ')"
    sleep 15
  fi
done
log "=== 完成 ==="
