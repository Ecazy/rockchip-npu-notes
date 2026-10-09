#!/usr/bin/env bash
# run_fp16_deep_channel_probe.sh — CNA 输入 feature tile 32KB 包线探针（§7.30，修正 §7.29）
# 在板端 exp2 目录运行；用法：./run_fp16_deep_channel_probe.sh
#   [FP16_DEEP_LOG=... 覆盖日志名]
#
# ⚠ §7.31 修正：本探针记录的「32 KB 硬包线」实为**旧发射器硬编码 DATA_BANK=1** 的表现
#   （DATA_BANK × 32768 B；见 run_fp16_cbuf_bank_probe.sh）。本探针保留为**历史对照臂**，
#   全部命令带 `--no-auto-cfg` 以复现 §7.30 的旧配置；默认（auto-cfg）下这些用例均已通过。
#
# 结论（实测，RKNPU driver v0.9.8）：
#   硬件包线不是「多 tap/1x1 区分 c.ic 上限」，而是 **CNA 输入 feature tile 字节数
#   ≤ 32768 B**（即 emitter 的 in_size = iw*ih*对齐通道*每元素字节）：
#     · fp16：iw16 ih8  ic128 = 32768 B 正常；ih9 = 36864 B 算错（iw 亦然）
#     · 同一 iw/ih/ic 下 1x1 与 3x3 同为 OK/BAD ⇒ tap 形状无关（§7.29 的 tap 依赖
#       是探针配置混淆：多 tap 用 16x16、1x1 用 8x8）
#     · int8 同受此限：iw16 ih16 ic128 = 32768 B 正常；ic160 = 40960 B 算错
#   超限硬件静默算错（max|diff| 可达数千，非 ULP 噪声）。修复：按输出行（或 ic 组）
#   切分，使单次 submit 的 in_size ≤ 32768（vendor runtime 深卷积拆多 task 即此因）。
set -u
cd "$(dirname "$0")"
log=${FP16_DEEP_LOG:-fp16_deep_channel_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0 --no-auto-cfg"
I="--pattern --auto-fix --pad-con1 0 --no-auto-cfg"

verdict() {  # 打印：描述 | RESULT | 失配数/max|diff|/max ulp（判别结构性 vs ULP 噪声）
    local desc="$1"; shift
    local out res stat try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT, 疑似 submit 超时)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "total=[0-9]+  (fp16-)?mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    printf "%-34s %-52s %s\n" "$desc" "$res" "$stat"
}

{
    echo "# CNA input feature tile 32KB envelope (corrects §7.29)  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) fp16 3x3p1 固定 iw16 扫 ih：32768 B 边界（ih8 正常，ih9 算错）"
    for ih in 4 8 9 12 16; do
        verdict "3x3 ic128 ih=$ih (in=$((16*ih*128*2))B)" $F --kw 3 --kh 3 --iw 16 --ih "$ih" --pad 1 --ic 128 --oc 16
    done

    echo
    echo "## B) fp16 3x3p1 固定 ih16 扫 iw：iw 与 ih 对称"
    for iw in 8 9 12 16; do
        verdict "3x3 ic128 iw=$iw (in=$((iw*16*128*2))B)" $F --kw 3 --kh 3 --iw "$iw" --ih 16 --pad 1 --ic 128 --oc 16
    done

    echo
    echo "## C) tap 形状无关：同一 iw/ih/ic 下 1x1 与 3x3 同为 OK/BAD"
    verdict "1x1 ic128 iw16 ih8  (32768B)"  $F --kw 1 --kh 1 --iw 16 --ih 8 --pad 0 --ic 128 --oc 16
    verdict "3x3 ic128 iw16 ih8  (32768B)"  $F --kw 3 --kh 3 --iw 16 --ih 8 --pad 1 --ic 128 --oc 16
    verdict "1x1 ic128 iw16 ih9  (36864B)"  $F --kw 1 --kh 1 --iw 16 --ih 9 --pad 0 --ic 128 --oc 16
    verdict "3x3 ic128 iw16 ih9  (36864B)"  $F --kw 3 --kh 3 --iw 16 --ih 9 --pad 1 --ic 128 --oc 16

    echo
    echo "## D) 小 ic 亦可越界（证明由 tile 字节而非 ic）：ic64 16x24 / ic32 16x48"
    verdict "3x3 ic64 iw16 ih24 (49152B)" $F --kw 3 --kh 3 --iw 16 --ih 24 --pad 1 --ic 64 --oc 16
    verdict "3x3 ic32 iw16 ih48 (49152B)" $F --kw 3 --kh 3 --iw 16 --ih 48 --pad 1 --ic 32 --oc 16
    verdict "3x3 ic16 iw16 ih64 (32768B)" $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 16 --oc 16

    echo
    echo "## E) c.ic 对齐跨度决定 tile 字节（真实通道 vs --ic-pad）"
    verdict "ic64 --ic-pad 64（32768B 基线）" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 64 --oc 16 --ic-pad 64
    verdict "ic64 --ic-pad 96（49152B）"      $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 64 --oc 16 --ic-pad 96
    verdict "ic96 --ic-pad 64（32768B）"      $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 96 --oc 16 --ic-pad 64

    echo
    echo "## F) fp16 3x3p1 权重字节无关：oc 放大到 64（73728 B）仍 BIT-EXACT"
    verdict "3x3 ic64 oc64" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 64 --oc 64

    echo
    echo "## G) int8 对照：同受 32768 B tile 限"
    verdict "int8 3x3 ic128 iw16 ih16 (32768B)" $I --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16
    verdict "int8 3x3 ic128 iw16 ih17 (34816B)" $I --kw 3 --kh 3 --iw 16 --ih 17 --pad 1 --ic 128 --oc 16
    verdict "int8 3x3 ic64  iw16 ih32 (32768B)" $I --kw 3 --kh 3 --iw 16 --ih 32 --pad 1 --ic 64  --oc 16
} 2>&1 | tee "$log"

echo "SAVED: $log"
