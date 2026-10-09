#!/usr/bin/env bash
# FP16 BS/BRDMA ALU+MUL 级标定探针（§7.24）。
# 结论：
#   - ALU addend = **fp32**（immediate = BS_ALU_CFG 整个 32 位按 IEEE-754；BRDMA src
#     = bias 表槽内 fp32），先施加。
#   - MUL factor = **fp16**（immediate = BS_MUL_CFG[31:16]；BRDMA src = mul 表槽内
#     16 位按 fp16），后施加，无 shift。
#   - out = f16( (conv + alu_fp32) * mul_fp16 )。
#   - `--bs-cfg 0x20140` 恒输出 0 的真因：默认 BS_MUL_CFG=0 ⇒ mul=+0.0。
# 关键点：submit ret!=0（ETIMEDOUT）是环境失败（fp16 带表 DPU 路径奇偶交替超时），
# 必须重试；不能因 set -e 中止整个序列。
set -uo pipefail

cd "$(dirname "$0")"
probe_log=${FP16_BS_MUL_LOG:-"fp16_bs_mul_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_fp16_bs_mul_probe min_conv_exp.c -lm || exit 1

shape="--pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 --in-ch-id --w-ones"
shape3="--pattern --fp16 --auto-fix --pad-con1 0 --kw 3 --kh 3 --iw 8 --ih 8 --pad 1 --ic 8 --oc 8 --in-ch-id --w-ones"

run_retry() {
    local desc=$1; shift
    echo "### ${desc}"
    local attempt out
    for attempt in 1 2 3 4 5 6; do
        out=$(./min_conv_fp16_bs_mul_probe "$@" 2>&1)
        if grep -q "SUBMIT ret=0" <<<"$out"; then
            echo "$out" | grep -E "P\[oc=0 x=0 y=0\]|RESULT:|fp16-mismatch|max ulp"
            echo "  (submit ret=0; attempts=${attempt})"
            return 0
        fi
        echo "  [attempt ${attempt}] submit failed, retrying"
    done
    echo "  !! all attempts ETIMEDOUT (environment)"
    return 1
}

# BS 激活前先跑一次同几何 BS-bypass 提交武装表状态（§7.19）
prewarm() { ./min_conv_fp16_bs_mul_probe "$@" >/dev/null 2>&1 || true; }

{
    echo "# FP16 BS/BRDMA ALU(add)/MUL 级标定"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo "### health probe (BS bypass) -> BIT-EXACT"
    ./min_conv_fp16_bs_mul_probe $shape 2>&1 | grep -E "SUBMIT ret=|RESULT:" || true

    echo "# --- ALU addend 编码：fp32 全 32 位 ---"
    prewarm $shape
    run_retry "ALU imm fp32 +5.0 (0x40a00000), MUL bypass"  $shape --bs-cfg 0x20050 --bs-alu 0x40a00000
    run_retry "ALU imm fp32 +0.5 (0x3f000000), MUL bypass"  $shape --bs-cfg 0x20050 --bs-alu 0x3f000000
    run_retry "ALU imm fp32 512.5 (0x44008000) discriminator" $shape --bs-cfg 0x20050 --bs-alu 0x44008000
    run_retry "ALU imm fp32 -2.0 (0xc0000000), MUL bypass"  $shape --bs-cfg 0x20050 --bs-alu 0xc0000000

    echo "# --- MUL factor 编码：fp16 @ [31:16] ---"
    prewarm $shape
    run_retry "MUL imm fp16 x2.0 (0x40000000)"   $shape --bs-cfg 0x20140 --bs-mul 0x40000000
    run_retry "MUL imm fp16 x1.75 (0x3f000000)"  $shape --bs-cfg 0x20140 --bs-mul 0x3f000000
    run_retry "MUL imm fp16 x0.5 (0x38000000)"   $shape --bs-cfg 0x20140 --bs-mul 0x38000000
    run_retry "MUL imm fp16 x-2.0 (0xc0000000)"  $shape --bs-cfg 0x20140 --bs-mul 0xc0000000
    run_retry "MUL imm default 0 -> out 0 (mystery resolved)" $shape --bs-cfg 0x20140

    echo "# --- ALU 先、MUL 后 ---"
    prewarm $shape
    run_retry "ALU +2.0 then MUL x2.0 -> (conv+2)*2" $shape --bs-cfg 0x20040 --bs-alu 0x40000000 --bs-mul 0x40000000

    echo "# --- BRDMA 表源：bias 槽 fp32，mul 槽 fp16（0x20140 = ALU src=BRDMA + MUL active） ---"
    prewarm $shape
    run_retry "table bias fp32 5.0 only (0x20150 MUL bypass) -> conv+5" \
        $shape --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000 --brdma-mul 16384
    run_retry "table mul fp16 0.5 (slot 0x3800=14336) -> conv*0.5" \
        $shape --bs-cfg 0x20140 --bs-mul 0x1 --brdma-cfg 0xe --brdma-mul 14336
    run_retry "table bias fp32 5.0 * mul fp16 2.0 -> (conv+5)*2" \
        $shape --bs-cfg 0x20140 --bs-mul 0x1 --brdma-cfg 0xe --brdma-bias 0x40a00000 --brdma-mul 16384
    run_retry "table bias fp32 ramp + mul fp16 ramp" \
        $shape --bs-cfg 0x20140 --bs-mul 0x1 --brdma-cfg 0xe --brdma-bias 0x40200000 --brdma-bias-step 0x100000 \
        --brdma-mul 13824 --brdma-mul-step 512

    echo "# --- 全 pattern 3x3 组合验证 ---"
    prewarm $shape3
    run_retry "3x3 full pattern: ALU fp32 +1.25 * MUL fp16 1.5" \
        $shape3 --bs-cfg 0x20040 --bs-alu 0x3fa00000 --bs-mul 0x3e000000

    echo "# --- int8 不回归（BS bias int32 表） ---"
    run_retry "int8 3x3 BS bias table" --pattern --auto-fix --pad-con1 0 --kw 3 --kh 3 \
        --iw 8 --ih 8 --pad 1 --ic 8 --oc 8 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 5

    echo "### alternating-timeout demo (6 sequential submits, no retry)"
    for i in 1 2 3 4 5 6; do
        r=$(./min_conv_fp16_bs_mul_probe $shape --bs-cfg 0x20140 --bs-mul 0x40000000 2>&1 | grep -o "SUBMIT ret=[-0-9]*")
        echo "  run ${i}: ${r}"
        sleep 1
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
