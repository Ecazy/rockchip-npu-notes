#!/usr/bin/env bash
# run_wgrad_deep_probe.sh — §7.37 深通道 fp16 训练闭环（wgrad_real.c，编译期 -DIC/-DOC）
# 在板端 exp3 目录运行；用法：./run_wgrad_deep_probe.sh [DEEP_LOG=...]
#
# 背景：`wgrad_real.c` 的 `IC`/`OC` 原是硬编码 `3`/`16`（fp16 打包/参考里还写着
# `FEATURE_ATOMIC_SIZE=16`），故训练闭环此前只覆盖 ic=3/oc=16。本轮把这两处参数化：
#   · `#ifndef IC/OC/L2_IC` ⇒ 可 `gcc -DIC=64 -DOC=64` 生成深通道变体；
#   · fp16 的输入通道对齐改由 `f16_ic_pad_for(IC)` 决定（≤16→16，否则 32 对齐，与 min_conv_exp 同规则）；
#   · 激活/权重 BO 由 64 KB 放大到 1 MB。
#
# 结果（板端 RKNPU driver v0.9.8）：**单层 fp16 闭环在 ic≤128 / oc≤128 全 BIT-EXACT**：
#   ic3/oc16（默认）、ic64/oc64、ic128/oc32、ic64/oc128 —— 前向、27/576/1152 坐标 wgrad、
#   写回后复前向全部逐位一致；pad0/1、batch 1/2/3 均覆盖。
# 边界（本轮未支持，需另行移植）：
#   · **深通道 int8**：int8 打包器按 16 通道原子写死（`pack_*`/`pack_wgrad_weights`），
#     ic=64 下前向 mismatch、wgrad submit 失败；
#   · ~~深通道 `--two-layer`~~ **已修（§7.40）**：层间偏移改为按几何计算（原固定 36864/8192
#     与层1 输出/权重重叠）+ 层1 权重打包/参考的 ic_pad 由 `f16_ic_pad_for()` 决定（原写死 16）
#     ⇒ ic≤128 的两层闭环全部 BIT-EXACT。
#   · ~~深通道 int8~~ **已修（§7.41）**：`pack_fwd_input`（多平面）／`pack_fwd_weights`／
#     `pack_wgrad_weights` 用 `w_elem_index_i8()` 通用布局 A（保留本工具 int8 链路的「空间转置」
#     约定：tap 嵌套 ((…*kw+kx)*kh+ky)）＋ `f.oc` 用硬件 oc 跨度 `ALIGN_UP(MAX(OC,32),32)`
#     ＋ 输出回读按 16 通道原子分组。注意：深几何必须用 `--gen-only --dir <新目录>` 生成
#     对应尺寸的数据集（旧 wgdata_pad1 的文件尺寸按 ic=3/oc=16 落盘，深构建会读到不完整数据）。
set -u
cd "$(dirname "$0")"
log=${DEEP_LOG:-wgrad_deep_$(date +%Y%m%d_%H%M%S).log}
mkdir -p deep

build() {  # name flags
    gcc -O2 -o "deep/wgrad_real_$1" $2 wgrad_real.c -lm || { echo "BUILD FAILED $1"; return 1; }
}

show() {  # 只抓关键判据行
    grep -E "^== wgrad_real|F16 FWD b=|F16 FWD2 b=|WGRAD16_TOTAL|L2 F1|L2 F2|L2_WGRAD:|RESULT:|COST:|mismatch|SUBMIT-FAILED" \
    | sed -e 's/^/    /'
}

{
    echo "# §7.37 深通道 fp16 训练闭环（wgrad_real.c 参数化 IC/OC）  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 默认 ic=3/oc=16（回归基线）"
    build d3_16 "" && {
        echo "  -- fp16 pad1 B1 update --"; ./deep/wgrad_real_d3_16 --fp16 --pad 1 --batch 1 --update 2>&1 | show
        sleep 0.5
        echo "  -- int8 pad1 B1 update --"; ./deep/wgrad_real_d3_16 --dir wgdata_pad1 --pad 1 --batch 1 --update 2>&1 | show
    }

    echo
    echo "## B) 深通道 ic=64 oc=64"
    build d64_64 "-DIC=64 -DOC=64" && {
        for args in "--fp16 --pad 1 --batch 1 --update" "--fp16 --pad 0 --batch 1 --update" "--fp16 --pad 0 --batch 3 --update"; do
            echo "  -- $args --"; ./deep/wgrad_real_d64_64 $args 2>&1 | show; sleep 0.5;
        done
    }

    echo
    echo "## C) 深通道 ic=128 oc=32"
    build d128_32 "-DIC=128 -DOC=32" && {
        for args in "--fp16 --pad 1 --batch 1 --update" "--fp16 --pad 0 --batch 2 --update"; do
            echo "  -- $args --"; ./deep/wgrad_real_d128_32 $args 2>&1 | show; sleep 0.5;
        done
    }

    echo
    echo "## D) 深 oc ic=64 oc=128"
    build d64_128 "-DIC=64 -DOC=128" && {
        echo "  -- fp16 pad1 B1 update --"; ./deep/wgrad_real_d64_128 --fp16 --pad 1 --batch 1 --update 2>&1 | show
    }

    echo
    echo "## D2) 深通道 --two-layer（§7.40 修：层间偏移按几何 + 层1 权重 ic_pad 参数化）"
    for v in "d3_16:-DIC=3 -DOC=16" "d64_64:-DIC=64 -DOC=64" "d128_32:-DIC=128 -DOC=32" "d48_48:-DIC=48 -DOC=48"; do
        n=${v%%:*}; f=${v#*:}
        build $n "$f" && {
            echo "  -- $n $f --"
            ./deep/wgrad_real_$n --two-layer --pad 1 --batch 1 --seed 20260928 --update 2>&1 |                 grep -E "L2_LAYOUT|L2 F1 b=|L2 F2 b=|L2_WGRAD:|F1. b=|RESULT:" | show
            sleep 0.5
        }
    done

    echo
    echo "## E) 深通道 int8（§7.41：打包器/回读参数化 + 需先用 --gen-only 生成对应几何的数据集）"
    for v in "d3_16:" "d64_64:-DIC=64 -DOC=64" "d128_32:-DIC=128 -DOC=32"; do
        n=${v%%:*}; f=${v#*:}
        build $n "$f" || continue
        ddir="deep/wgdata_${n}_pad1"
        ./deep/wgrad_real_$n --gen-only --dir "$ddir" --pad 1 --seed 20260928 >/dev/null 2>&1
        echo "  -- $n int8（数据集 $ddir, w.bin=$(stat -c%s $ddir/w.bin 2>/dev/null || echo '-') B）--"
        ./deep/wgrad_real_$n --dir "$ddir" --pad 1 --batch 1 --update 2>&1 | \
            grep -E "FWD b=0:|WGRAD_TOTAL|FWD2 b=0:|RESULT:" | show
        sleep 0.5
    done
} 2>&1 | tee "$log"

echo "SAVED: $log"
