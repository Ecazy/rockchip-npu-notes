#!/usr/bin/env bash
# run_fp16_ws1_probe.sh — §7.36：fp16 `WEIGHT_SIZE1` 必须 ×2（根因）+ 权重侧非硬限
# 在板端 exp2 目录运行；用法：./run_fp16_ws1_probe.sh [WS1_LOG=...]
#
# 背景：探索「更大的 fp16 oc」（handoff §3.5 未闭合项）时发现：小输入 + 多 oc 块的 fp16
# 卷积会**静默算错后半个 oc 块**（mismatch 集中在 oc ≥ 32），且表现随几何/次数抖动。
#
# 根因（决定性）：发射器 fp16 路径把 `WEIGHT_SIZE0` ×2（§7.17）却漏了 `WEIGHT_SIZE1`。
# `WEIGHT_SIZE1 = kw·kh·ic` 是**字节数**，fp16 每元素 2 B ⇒ 必须 `kw·kh·ic·2`。
#   判别：失败用例加 `--ws1 kw*kh*ic*2` 立即 ULP-EXACT；改回旧值复现 MISMATCH。
# 修复后：ic∈{32..192} × oc∈{32..256}（含多 ic/oc 块）3×3 全部 ULP-EXACT。
#
# 旁得（修正 §7.31 的措辞）：**权重侧不是硬限**——权重描述符从 DRAM 流式读取，描述符
# 超过 `WEIGHT_BANK`（甚至 21 bank vs WB=11）仍 BIT-EXACT。真正硬的只有
# **输入 tile ≤ DATA_BANK × 32768**。
set -u
cd "$(dirname "$0")"
log=${WS1_LOG:-fp16_ws1_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"

row() {
    local desc="$1"; shift
    local out res stat cfg try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    cfg=$(echo "$out" | grep -oE "CBUF-CFG: .*" | head -1)
    printf "%-34s %-24s %s\n" "$desc" "$res" "$stat"
    [ -n "$cfg" ] && printf "    %s\n" "$cfg"
}

{
    echo "# §7.36 fp16 WEIGHT_SIZE1 必须 ×2（根因）+ 权重侧非硬限  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) A/B：同一失败用例，「旧 WEIGHT_SIZE1（--ws1 kw*kh*ic）」vs 修复后（默认 ×2）"
    echo "   # iw8 ih8 pad0 3x3：小输入 + 多 ic/oc 块 ⇒ 旧值让后半个 oc 块静默取错权重"
    for pair in "64 64" "96 32" "32 128" "128 64"; do
        set -- $pair; ic=$1; oc=$2; old=$((3*3*ic))
        row "ic$ic oc$oc 旧 ws1=$old" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic $ic --oc $oc --ws1 $old
        row "ic$ic oc$oc 修复（默认 ws1=$((old*2))）" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic $ic --oc $oc
        sleep 0.3
    done

    echo
    echo "## B) 修复后：宽 ic × oc 网格（3x3 iw8 ih8 pad0）应全 ULP-EXACT"
    for pair in "32 32" "64 32" "96 32" "128 32" "64 64" "96 64" "128 64" "16 128" "32 96" "32 128" "32 256"; do
        set -- $pair; ic=$1; oc=$2
        row "ic$ic oc$oc" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic $ic --oc $oc
        sleep 0.3
    done

    echo
    echo "## C) 权重侧不是硬限：WEIGHT_BANK 小于权重描述符 / 描述符 21 bank 仍 BIT-EXACT"
    row "3x3 ic256 iw16 ih40 WB=2(默认DB=9)"  $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 0 --ic 256 --oc 16
    row "3x3 ic256 iw16 ih40 --weight-bank 2" $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 0 --ic 256 --oc 16 --weight-bank 2
    row "3x3 ic192 oc192 iw4 ih4（权重 21 bank）" $F --kw 3 --kh 3 --iw 4 --ih 4 --pad 0 --ic 192 --oc 192

    echo
    echo "## D) oc 切分（§7.36 实现）与不切分同结果"
    row "ic192 oc192 不切分"      $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic 192 --oc 192
    row "ic192 oc192 oc-groups 4" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic 192 --oc 192 --oc-groups 4
    row "ic192 oc192 oc-groups 8" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic 192 --oc 192 --oc-groups 8

    echo
    echo "## E) 不回归：既有形状 hash 应与 §7.31/§7.33/§7.34/§7.35 记录一致"
    row "ic16 3x3p1 ih16"           $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16  --oc 16
    row "ic128 3x3p1 ih16 groups4"  $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --ic-groups 4
    row "ic64 iw16 ih64 3x3p1"      $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 64  --oc 16
} 2>&1 | tee "$log"

echo "SAVED: $log"
