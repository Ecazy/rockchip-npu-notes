#!/usr/bin/env bash
# run_bank_split_probe.sh — §7.38 CBUF bank 分配的 vendor 口径核对 + 时序/下限
# 在板端 exp2 目录运行；用法：./run_bank_split_probe.sh [BANK_LOG=...]
#
# 三件事：
#  A) **时序**：同一 conv 只改 `--weight-bank N`（= DATA_BANK 12−N）看 hw_elapse 是否变化
#     （本板噪声 1.2–1.3×，min over 8 runs）。
#  B) **WEIGHT_BANK 下限**：描述符 3 / 4.5 / 21 bank 时扫 WB，记录 submit 超时边界
#     （`--weight-bank` 过小 ⇒ 6 s job timeout；我们发射器 auto 规则给 `ceil(wbytes/32768)+1`，安全）。
#  C) 输出 `analyze_vendor_bank_rules.py` 的判据摘要（离线部分见 results/vendor_bank_rules_20260930/）。
set -u
cd "$(dirname "$0")"
log=${BANK_LOG:-bank_split_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"

timing() {  # desc args...
    local desc="$1"; shift
    local best=999999999 out hw ok="" i
    for i in 1 2 3 4 5 6 7 8; do
        out=$(timeout 30 ./min_conv_exp "$@" 2>&1)
        hw=$(echo "$out" | grep -oE "hw_elapse=[0-9]+" | head -1 | cut -d= -f2)
        [ -n "$hw" ] && [ "$hw" -lt "$best" ] && best=$hw
        ok=$(echo "$out" | grep -oE "RESULT: [A-Z-]+" | head -1)
    done
    printf "%-32s min_hw=%9s ns  %s\n" "$desc" "$best" "$ok"
}

wbmin() {  # desc wb args...
    local desc="$1"; shift
    local n=0 r="" i out
    for i in 1 2 3; do
        out=$(timeout 30 ./min_conv_exp "$@" 2>&1)
        r=$(echo "$out" | grep -oE "RESULT: [A-Z-]+" | head -1)
        case "$r" in *EXACT*) n=$((n+1));; esac
    done
    printf "%-32s ok=%d/3  %s\n" "$desc" "$n" "$r"
}

{
    echo "# §7.38 CBUF bank 分配：vendor 口径核对 + 时序 + WEIGHT_BANK 下限  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 时序 vs bank 划分（输入需 4/8 bank；min over 8 runs，本板噪声 1.2–1.3×）"
    timing "S1 3x3 iw64 ih32 ic32: DB=4"  $F --kw 3 --kh 3 --iw 64 --ih 32 --pad 1 --ic 32 --oc 16 --weight-bank 8
    timing "S1 3x3 iw64 ih32 ic32: DB=11" $F --kw 3 --kh 3 --iw 64 --ih 32 --pad 1 --ic 32 --oc 16 --weight-bank 1
    timing "S2 3x3 iw32 ih32 ic64 oc64: DB=4"  $F --kw 3 --kh 3 --iw 32 --ih 32 --pad 1 --ic 64 --oc 64 --weight-bank 8
    timing "S2 3x3 iw32 ih32 ic64 oc64: DB=9"  $F --kw 3 --kh 3 --iw 32 --ih 32 --pad 1 --ic 64 --oc 64 --weight-bank 3
    timing "S3 1x1 iw64 ih64 ic32: DB=8"  $F --kw 1 --kh 1 --iw 64 --ih 64 --pad 0 --ic 32 --oc 16 --weight-bank 4
    timing "S3 1x1 iw64 ih64 ic32: DB=11" $F --kw 1 --kh 1 --iw 64 --ih 64 --pad 0 --ic 32 --oc 16 --weight-bank 1

    echo
    echo "## B) WEIGHT_BANK 下限（过小 ⇒ submit 6 s 超时；ok=0/3）"
    echo "  -- 描述符 73728 B（3 bank）：3x3 ic64 oc64 iw32 ih32 --"
    for wb in 1 2 3; do wbmin "WB=$wb DB=$((12-wb))" $F --kw 3 --kh 3 --iw 32 --ih 32 --pad 1 --ic 64 --oc 64 --weight-bank $wb; done
    echo "  -- 描述符 147456 B（4.5 bank）：3x3 ic64 oc128 iw8 ih8 --"
    for wb in 1 2 3; do wbmin "WB=$wb DB=$((12-wb))" $F --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic 64 --oc 128 --weight-bank $wb; done
    echo "  -- 描述符 663552 B（21 bank）：3x3 ic192 oc192 iw4 ih4 --"
    for wb in 2 3 4; do wbmin "WB=$wb DB=$((12-wb))" $F --kw 3 --kh 3 --iw 4 --ih 4 --pad 0 --ic 192 --oc 192 --weight-bank $wb; done

    echo
    echo "## C) 离线判据（同目录 results/vendor_bank_rules_20260930/bank_rules.txt）"
    echo "  R1 grains == ih+kh+pad_top                : 85/85"
    echo "  R2 DATA_BANK >= ceil(in/32768)            : 85/85"
    echo "  R3 DATA_BANK + WEIGHT_BANK == 12          : 85/85"
    echo "  R4 WEIGHT_BANK == ceil(WEIGHT_SIZE0/32768): 18/18（真 vendor task）/ 0/67（Mesa 派生发射器）"
} 2>&1 | tee "$log"

echo "SAVED: $log"
