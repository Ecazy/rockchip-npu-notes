#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${DPU_BN_SIGN_LOG:-"dpu_bn_mul_sign_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_dpu_bn_probe min_conv_exp.c -lm

{
    echo "# BN MUL sign/shift probe: uniform 3x3 conv, ic=3, oc=16"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for input in 0x01 0xff; do
        for shift in 0 1 2; do
            mul_cfg=$(( (3 << 16) | (shift << 8) ))
            echo
            echo "### input=${input} BN_MUL operand=3 shift=${shift} BN_TRUNCATE_SRC=0"
            ./min_conv_dpu_bn_probe \
                --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 --auto-fix \
                --in-byte "$input" --w-byte 1 --bn-cfg 0x42 --bn-mul "$mul_cfg"
        done
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
