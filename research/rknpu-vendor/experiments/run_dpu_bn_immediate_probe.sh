#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${DPU_BN_LOG:-"dpu_bn_immediate_$(date +%Y%m%d_%H%M%S).log"}

gcc -O2 -o min_conv_dpu_bn_probe min_conv_exp.c -lm

{
    echo "# DPU BN immediate ALU/MUL probes; IC=3, OC=16, int8 conv"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo "### baseline: BN bypass"
    ./min_conv_dpu_bn_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 --auto-fix

    echo "### BN ALU add immediate +3: BN_CFG=0x20050"
    ./min_conv_dpu_bn_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
        --auto-fix --bn-cfg 0x20050 --bn-alu 3

    for spec in 0x00030000:0 0x00030100:1 0x00030200:2; do
        mul_cfg=${spec%%:*}
        shift=${spec#*:}
        echo "### BN MUL operand 3, shift ${shift}, BN_TRUNCATE_SRC=0: BN_CFG=0x42"
        ./min_conv_dpu_bn_probe \
            --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
            --auto-fix --bn-cfg 0x42 --bn-mul "$mul_cfg"
    done

    echo "### BN MUL operand 3, pos shift 1, neg shift 1 via DATA_FORMAT=0x400"
    ./min_conv_dpu_bn_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
        --auto-fix --bn-cfg 0x42 --bn-mul 0x00030100 --dpu-data-format 0x400
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
