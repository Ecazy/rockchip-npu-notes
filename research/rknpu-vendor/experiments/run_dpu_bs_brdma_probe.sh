#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${DPU_BRDMA_LOG:-"dpu_bs_brdma_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_dpu_brdma_probe min_conv_exp.c -lm

{
    echo "# DPU BS/BRDMA per-channel bias and multiplier probe"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo "### baseline: BS defaults"
    ./min_conv_dpu_brdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 --auto-fix

    echo "### BRDMA: identity gamma, beta[channel]=channel-8"
    ./min_conv_dpu_brdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 --auto-fix \
        --bs-cfg 0x20140 --bs-mul 0x00000e01 --dpu-data-format 0xe0 \
        --brdma-cfg 0x0e --brdma-bias -8 --brdma-bias-step 1 \
        --brdma-mul 16384

    echo "### BRDMA: gamma[channel]=0.5+channel/64, beta[channel]=channel-8"
    ./min_conv_dpu_brdma_probe \
        --pattern --ic 3 --oc 16 --pad 1 --pad-con1 0 --out-zp 128 --auto-fix \
        --bs-cfg 0x20140 --bs-mul 0x00000e01 --dpu-data-format 0xe0 \
        --brdma-cfg 0x0e --brdma-bias -8 --brdma-bias-step 1 \
        --brdma-mul 8192 --brdma-mul-step 256
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
