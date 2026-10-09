#!/usr/bin/env bash
# run_col_split_probe.sh — §7.35 输出列切分（列 × 行 × ic 三维切分）
# 在板端 exp2 目录运行；用法：./run_col_split_probe.sh [COL_SPLIT_LOG=...]
#
# 问题（handoff §8 下一入口①）：当**单行输入字节**就超过可用 data bank（`iw × 通道字节
# > 可用 bank × 32768`）时，行切分与 ic 切分都救不了——必须按**输出列**切分。
#
# 答案（实测，RKNPU driver v0.9.8）：
#   · `--tile-ow N`（显式列宽）+ `--tile-auto`（列/行联合自动）已落地；列 tile 把输入
#     **子矩形**逐行 stage 成紧凑块（行内字节 = 列宽），`pad_left` 相应下调，输出按行段
#     scatter 回全局/组 partial 面。
#   · 关键约束（本轮发现）：输入 `SURF_STRIDE = L·(staged_ih/4 − 1)` 编码「平面步长 =
#     4 行组」⇒ **staged 输入行数必须 ≥ 4**（否则该式无解/负值 → 结构性算错）。自动模式
#     因此按「4 行预算」选列宽。
#   · 同一几何五种切分（自动列切 / ic4+自动 / ic16+自动 / 显式 tile-ow 128 / 以及不同
#     行×列组合）**全部 BIT-EXACT 且 hash 相同**（`e86f99c0`）。
#   · 三维（行 × 列 × ic）组合可用（见 D 段）。
set -u
cd "$(dirname "$0")"
log=${COL_SPLIT_LOG:-col_split_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"

row() {
    local desc="$1"; shift
    local out res stat hash tile ic split try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    hash=$(echo "$out" | grep -oE "OUT-HASH: [0-9a-f]+" | sed 's/OUT-HASH: //')
    tile=$(echo "$out" | grep -oE "^TILE: [0-9]+ tile\(s\)  tile=[0-9]+x[0-9]+" | sed 's/TILE: //')
    ic=$(echo "$out" | grep -oE "^IC-GROUPS: [0-9]+ group" | sed 's/IC-GROUPS: //')
    split=$(echo "$out" | grep -oE "IC-SPLIT: .*" | head -1)
    printf "%-32s hash=%-9s %-24s %-18s %-16s %s\n" "$desc" "$hash" "$res" "$stat" "$tile" "$ic"
    [ -n "$split" ] && printf "    %s\n" "$split"
}

{
    echo "# §7.35 输出列切分（列 × 行 × ic）  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 单行超 bank：iw1024 ic256 ih8 1×1（行 = 512 KB = 16 bank > 可用 11）"
    echo "   # 期望：全部 BIT-EXACT 且 hash 相同（列/行/ic 的任意组合都不改变结果）"
    row "不切分（越界）"         $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16
    row "tile-auto（自动列切）"  $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --tile-auto
    row "ic-groups 4 + auto"     $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --ic-groups 4  --tile-auto
    row "ic-groups 16 + auto"    $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --ic-groups 16 --tile-auto
    row "显式 --tile-ow 128"     $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --tile-ow 128
    row "显式 --tile-ow 256"     $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --tile-ow 256
    row "显式 --tile-oh 4 --tile-ow 256" $F --kw 1 --kh 1 --iw 1024 --ih 8 --pad 0 --ic 256 --oc 16 --tile-oh 4 --tile-ow 256

    echo
    echo "## B) 3×3 宽输入（行字节 32 KB，×4 行 ≤ bank 时不切列）"
    row "iw512 ic32 ih16 3x3p1 不切分" $F --kw 3 --kh 3 --iw 512 --ih 16 --pad 1 --ic 32 --oc 16
    row "iw512 ic32 ih16 3x3p1 auto"  $F --kw 3 --kh 3 --iw 512 --ih 16 --pad 1 --ic 32 --oc 16 --tile-auto

    echo
    echo "## C) 列切 + pad（pad 非 0 时首/末列 tile 的 pad_left 下调）"
    row "iw256 ic64 ih8 3x3p1 不切分" $F --kw 3 --kh 3 --iw 256 --ih 8 --pad 1 --ic 64 --oc 16
    row "iw256 ic64 ih8 3x3p1 auto"   $F --kw 3 --kh 3 --iw 256 --ih 8 --pad 1 --ic 64 --oc 16 --tile-auto
    echo "   # 强制列切以覆盖 pad_left 下调（首/内/末列 tile）：--tile-ow 64"
    row "iw256 ic64 ih8 3x3p1 --tile-ow 64"        $F --kw 3 --kh 3 --iw 256 --ih 8 --pad 1 --ic 64 --oc 16 --tile-ow 64
    row "iw256 ic64 ih8 3x3p1 --tile-ow 64 --tile-oh 4" $F --kw 3 --kh 3 --iw 256 --ih 8 --pad 1 --ic 64 --oc 16 --tile-ow 64 --tile-oh 4

    echo
    echo "## D) 三维：列 × 行 × ic（iw1024 ic256 ih32 1×1）"
    row "不切分"                  $F --kw 1 --kh 1 --iw 1024 --ih 32 --pad 0 --ic 256 --oc 16
    row "ic4 + auto（列+行+ic）"  $F --kw 1 --kh 1 --iw 1024 --ih 32 --pad 0 --ic 256 --oc 16 --ic-groups 4 --tile-auto

    echo
    echo "## E) 不回归：既有形状 hash 应与 §7.31/§7.33/§7.34 一致"
    row "ic16 3x3p1 ih16"          $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16  --oc 16
    row "ic128 3x3p1 ih16 groups4" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --ic-groups 4
    row "ic256 ih400 1x1 groups4+tile" $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --ic-groups 4 --tile-auto
} 2>&1 | tee "$log"

echo "SAVED: $log"
