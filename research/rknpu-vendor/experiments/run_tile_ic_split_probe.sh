#!/usr/bin/env bash
# run_tile_ic_split_probe.sh — §7.34 行切分 × ic 切分（二维切分）
# 在板端 exp2 目录运行；用法：./run_tile_ic_split_probe.sh [TILE_IC_LOG=...]
#
# 问题（handoff §8 下一入口①）：当**单个 ic 组的一整幅输入**仍超出可用 data bank 时，
# ic 切分不够、行切分也不够（每多少行都不够需按列，或行预算太小）——必须二维切分。
#
# 答案（实测，RKNPU driver v0.9.8）：
#   · `--ic-groups G` 与 `--tile-*` 可**组合**：每组先按输出行 tile，tile 的输入只 stage
#     本组的平面；每个 tile 的 partial 写回**本组的**输出面，最后 host fp32 累加各组。
#   · 同一几何（`iw16 ic256 ih400` 1×1，全图 = 100 bank，单组 = 25 bank）三种策略：
#     行切分（10 tile）/ 2D（3 tile × 4 组）/ 2D（2 tile × 8 组）**hash 完全相同、BIT-EXACT**；
#     纯 ic 切分（组内仍 13 bank > 11 可用）结构性 MISMATCH ⇒ 必须二维。
#   · 2D 下组数/行数的选择不影响结果（hash 相同）；误差仍受 §7.33 的 partial fp16 量化支配
#     （`--in-ruler` 压力下 ≤1 ULP）。
#   · 顺带硬化：激活 BO 现按几何自动放大（输入可达数 MB + (G+1) 个 partial 面），
#     旧默认 1 MB 在大 ic × 大 ih 下会静默踩内存（segfault）。
set -u
cd "$(dirname "$0")"
log=${TILE_IC_LOG:-tile_ic_split_$(date +%Y%m%d_%H%M%S).log}

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
    tile=$(echo "$out" | grep -oE "^TILE: [0-9]+ tile" | sed 's/TILE: //')
    ic=$(echo "$out" | grep -oE "^IC-GROUPS: [0-9]+ group" | sed 's/IC-GROUPS: //')
    split=$(echo "$out" | grep -oE "IC-SPLIT: .*" | head -1)
    printf "%-30s hash=%-9s %-22s %-18s %s %s\n" "$desc" "$hash" "$res" "$stat" "$tile" "$ic"
    [ -n "$split" ] && printf "    %s\n" "$split"
}

{
    echo "# §7.34 二维切分（输出行 × ic）  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 同一几何三种策略（iw16 ic256 ih400 1×1：全图 100 bank，单组 25 bank）"
    echo "   # 行切分与两种 2D 应 hash 相同；纯 ic 因组内 > 可用 bank 而结构性算错"
    row "不切分（越界）"        $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16
    row "行切分 tile-auto"     $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --tile-auto
    row "纯 ic groups16(=8×32)" $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --ic-groups 16
    row "2D groups4+tile-auto"  $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --ic-groups 4  --tile-auto
    row "2D groups8+tile-auto"  $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --ic-groups 8  --tile-auto
    row "2D groups4+tile-oh 100" $F --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 256 --oc 16 --ic-groups 4 --tile-oh 100

    echo
    echo "## B) 组内可用 bank 够时纯 ic 切分即可（iw16 ic256 ih100：全图 25 bank 越界，组 4 bank 够）"
    row "ic256 ih100 不切分（越界）" $F --kw 1 --kh 1 --iw 16 --ih 100 --pad 0 --ic 256 --oc 16
    row "ic256 ih100 groups4"      $F --kw 1 --kh 1 --iw 16 --ih 100 --pad 0 --ic 256 --oc 16 --ic-groups 4
    row "ic256 ih100 groups8"      $F --kw 1 --kh 1 --iw 16 --ih 100 --pad 0 --ic 256 --oc 16 --ic-groups 8

    echo
    echo "## C) 2D + 压力 pattern --in-ruler（继承 §7.33 的 partial fp16 量化 ⇒ ≤1 ULP）"
    row "2D ruler ic96 ih400 1x1"  $F --in-ruler --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 96 --oc 16 --ic-groups 3 --tile-auto
    row "行切分 ruler 同上"        $F --in-ruler --kw 1 --kh 1 --iw 16 --ih 400 --pad 0 --ic 96 --oc 16 --tile-auto

    echo
    echo "## D) 3x3 二维切分（真卷积 + 深 ic）"
    row "3x3 ic128 ih200 不切分"   $F --kw 3 --kh 3 --iw 16 --ih 200 --pad 1 --ic 128 --oc 16
    row "3x3 ic128 ih200 groups4+tile-auto" $F --kw 3 --kh 3 --iw 16 --ih 200 --pad 1 --ic 128 --oc 16 --ic-groups 4 --tile-auto

    echo
    echo "## E) 不回归：既有形状 hash 应与 §7.31/§7.33 一致"
    row "ic16 3x3p1 ih16"        $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16  --oc 16
    row "ic128 3x3p1 ih16 groups4" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --ic-groups 4
    row "ic16 ih64 单发"          $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 16  --oc 16
} 2>&1 | tee "$log"

echo "SAVED: $log"
