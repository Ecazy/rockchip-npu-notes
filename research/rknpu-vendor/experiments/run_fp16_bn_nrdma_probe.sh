#!/usr/bin/env bash
# FP16 独立 BN（BN_ALU + NRDMA 逐通道表）标定探针。
# 结论：fn16 的 NRDMA 表是 **fp32 per-channel**，BN_ALU algo=2(add) 在 fp32 累加器上
# 逐输出通道相加，再 fp32->fp16 输出。表放 weight BO。
# 关键点：submit ret!=0（ETIMEDOUT）是环境失败（fp16 带表 DPU 路径奇偶交替超时），必须重试；
# 不能因 set -e 中止整个序列。
set -uo pipefail

cd "$(dirname "$0")"
probe_log=${FP16_BN_LOG:-"fp16_bn_nrdma_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_fp16_bn_nrdma_probe min_conv_exp.c -lm || exit 1

shape_fp16="--pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --ic 8 --oc 8"
bn_regs="--bn-cfg 0x20150 --bn-nrdma 0x2"

run_retry() {
    # 重试直到 submit ret=0；输出只保留关键行
    local desc=$1; shift
    echo "### ${desc}"
    local attempt out
    for attempt in 1 2 3 4 5 6; do
        out=$(./min_conv_fp16_bn_nrdma_probe "$@" 2>&1)
        if grep -q "SUBMIT ret=0" <<<"$out"; then
            echo "$out" | grep -E "== min_conv|DPU BN DMA table|RESULT:|fp16-mismatch|floor-mismatch|vs float32"
            echo "  (submit ret=0; attempts=${attempt})"
            return 0
        fi
        echo "  [attempt ${attempt}] submit failed, retrying"
    done
    echo "  !! all attempts ETIMEDOUT (environment)"
    return 1
}

{
    echo "# FP16 BN_ALU + NRDMA per-channel table"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    # 0) 健康探针：无 BN 的 fp16 1x1 应 BIT-EXACT
    echo "### health probe (BN bypass, no table)"
    ./min_conv_fp16_bn_nrdma_probe $shape_fp16 2>&1 | grep -E "SUBMIT ret=|RESULT:" || true
    # 1) 表全 0（基线）
    run_retry "table all zero" $shape_fp16 $bn_regs --bn-table-f32-bias 0 --bn-table-f32-step 0
    # 2) 常量 +5.0f（若按 int32 重解释会得到 ~+1.2e9）
    run_retry "constant +5.0f" $shape_fp16 $bn_regs --bn-table-f32-bias 5 --bn-table-f32-step 0
    # 3) 常量 +1.5f：非整数，最能区分 fp32 vs int32
    run_retry "constant +1.5f" $shape_fp16 $bn_regs --bn-table-f32-bias 1.5 --bn-table-f32-step 0
    # 4) 逐通道 ramp（判定按输出通道索引）
    run_retry "channel ramp 0.25+0.5*ch" $shape_fp16 $bn_regs --bn-table-f32-bias 0.25 --bn-table-f32-step 0.5
    # 5) 多 oc 组（oc=24，跨 16 原子边界）
    run_retry "oc=24 multi-group ramp" --pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 \
        --iw 4 --ih 4 --pad 0 --ic 8 --oc 24 $bn_regs --bn-table-f32-bias 0.25 --bn-table-f32-step 0.5
    # 6) 3x3 pad1，负/递减 ramp
    run_retry "3x3 p1 descending ramp -2+1.75*ch" --pattern --fp16 --auto-fix --pad-con1 0 --kw 3 --kh 3 \
        --iw 8 --ih 8 --pad 1 --ic 8 --oc 8 $bn_regs --bn-table-f32-bias -2 --bn-table-f32-step 1.75
    # 7) 5x5 pad2，ic16 oc24
    run_retry "5x5 p2 ic16 oc24 ramp 0.125-0.25*ch" --pattern --fp16 --auto-fix --pad-con1 0 --kw 5 --kh 5 \
        --iw 12 --ih 12 --pad 2 --ic 16 --oc 24 $bn_regs --bn-table-f32-bias 0.125 --bn-table-f32-step -0.25
    # 8) int8 sanity：表仍是 int32（--bn-table-bias/step），不回归
    run_retry "int8 sanity (int32 table)" --pattern --auto-fix --pad-con1 0 --kw 1 --kh 1 \
        --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 $bn_regs --bn-table-bias 0 --bn-table-step 1
    # 9) 交替超时演示（同命令连跑 6 次，不重试）：证明"奇偶交替"是环境现象
    echo "### alternating-timeout demo (6 sequential submits, no retry)"
    for i in 1 2 3 4 5 6; do
        r=$(./min_conv_fp16_bn_nrdma_probe $shape_fp16 $bn_regs --bn-table-f32-bias 5 --bn-table-f32-step 0 2>&1 | grep -o "SUBMIT ret=[-0-9]*")
        echo "  run ${i}: ${r}"
        sleep 1
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
