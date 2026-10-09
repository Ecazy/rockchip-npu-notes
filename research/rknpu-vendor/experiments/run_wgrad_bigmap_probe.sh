#!/usr/bin/env bash
# run_wgrad_bigmap_probe.sh — §7.49 更大的特征图（16×16 / 32×32）+ 深 ic 的训练闭环
# 在板端 exp3 目录运行；用法：./run_wgrad_bigmap_probe.sh [BIGMAP_LOG=...]
#
# 背景：`wgrad_real.c` 原先把 pad0 输出尺寸写死成 `6`（只对 IW=IH=8 成立）。改为
# `OW_P0 = IW-(KW-1)` / `OH_P0 = IH-(KH-1)` 后，可用 `-DIW/-DIH/-DIC/-DOC` 编译更大的图。
# 覆盖范围受**单次 submit 的输入 bank** 限制：`in_bytes = IW·IH·elem·IC ≤ 11×32768 = 360448 B`，
# 超出需行切分（下一步）。
set -u
cd "$(dirname "$0")"
log=${BIGMAP_LOG:-wgrad_bigmap_$(date +%Y%m%d_%H%M%S).log}
mkdir -p deep

build() { gcc -O2 -o "deep/wgrad_real_$1" $2 wgrad_real.c -lm || echo "BUILD FAILED $1"; }

show() { grep -E "F16 FWD b=|WGRAD16_TOTAL|F16 FWD2 b=|RESULT:|in_size|CBUF" | sed 's/^/    /'; }

{
    echo "# §7.49 更大特征图 + 深 ic 的训练闭环  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 默认 8x8（回归）"
    build base "" && { ./deep/wgrad_real_base --fp16 --pad 1 --batch 1 --update 2>&1 | show; }

    echo
    echo "## B) 16x16（输入 16·16·2·IC B）"
    for v in "m16_ic32:-DIW=16 -DIH=16 -DIC=32 -DOC=32" "m16_ic128:-DIW=16 -DIH=16 -DIC=128 -DOC=32"; do
        n=${v%%:*}; f=${v#*:}
        build $n "$f" && {
            echo "  -- $n $f --"
            ./deep/wgrad_real_$n --fp16 --pad 1 --batch 1 --update 2>&1 | show
            sleep 0.5
        }
    done

    echo
    echo "## C) 32x32"
    for v in "m32_ic32:-DIW=32 -DIH=32 -DIC=32 -DOC=32" "m32_ic128:-DIW=32 -DIH=32 -DIC=128 -DOC=32"; do
        n=${v%%:*}; f=${v#*:}
        build $n "$f" && {
            echo "  -- $n $f --"
            ./deep/wgrad_real_$n --fp16 --pad 1 --batch 1 --update 2>&1 | show
            sleep 0.5
        }
    done

    echo
    echo "## E) 前向行切分（§7.53）：切分结果必须与不切分逐位一致"
    build tile16 "-DIW=16 -DIH=16 -DIC=32 -DOC=32" && {
        echo "  -- 16x16 IC32 不切分（对照）--"
        ./deep/wgrad_real_tile16 --fp16 --pad 1 --batch 1 --update 2>&1 | show
        for t in 8 4 2; do
            echo "  -- 16x16 IC32 --tile-oh $t --"
            ./deep/wgrad_real_tile16 --fp16 --pad 1 --batch 1 --update --tile-oh $t 2>&1 | \
                grep -E "TILE r0|F16 FWD b=|F16 FWD2 b=|RESULT:" | show
            sleep 0.5
        done
    }
    build tile64 "-DIW=64 -DIH=64 -DIC=64 -DOC=32" && {
        echo "  -- 64x64 IC64 --tile-oh 16（输入 16 bank > 包线，靠切分）仅前向 --"
        timeout 120 ./deep/wgrad_real_tile64 --fp16 --pad 1 --batch 1 --tile-oh 16 2>&1 | \
            grep -E "TILE:|F16 FWD b=|WGRAD16_TOTAL|WARN|RESULT:" | show
    }

    echo
    echo "## D) pad0（OW_P0/OH_P0 路径）"
    build m16p0 "-DIW=16 -DIH=16 -DIC=32 -DOC=32" && {
        ./deep/wgrad_real_m16p0 --fp16 --pad 0 --batch 1 --update 2>&1 | show
    }
} 2>&1 | tee "$log"

echo "SAVED: $log"
