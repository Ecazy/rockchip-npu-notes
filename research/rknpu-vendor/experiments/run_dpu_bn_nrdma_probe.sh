#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${DPU_NRDMA_LOG:-"dpu_bn_nrdma_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_dpu_bn_nrdma_probe min_conv_exp.c -lm

{
    echo "# INT8 DPU BN ALU immediate vs NRDMA per-channel table"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo "### immediate ALU +3 reference"
    ./min_conv_dpu_bn_nrdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
        --auto-fix --bn-cfg 0x20050 --bn-alu 3

    echo "### NRDMA source, BN_ALU add, coefficients[channel]=channel"
    ./min_conv_dpu_bn_nrdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
        --auto-fix --bn-cfg 0x20150 --bn-nrdma 0x2 --bn-table-step 1

    echo "### NRDMA source, signed coefficients[channel]=channel-8"
    ./min_conv_dpu_bn_nrdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 \
        --auto-fix --bn-cfg 0x20150 --bn-nrdma 0x2 \
        --bn-table-bias -8 --bn-table-step 1
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
