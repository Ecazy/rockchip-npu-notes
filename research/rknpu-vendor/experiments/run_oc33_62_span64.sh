#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${OC_SPAN64_LOG:-"oc33_62_span64_$(date +%Y%m%d_%H%M%S).log"}

gcc -O2 -o min_conv_oc_span64_probe min_conv_exp.c -lm

{
    echo "# OC=33..62; per-tap span and WEIGHT_KERNELS are both 64"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for oc in $(seq 33 62); do
        echo
        echo "### oc_real=${oc} span=64 WEIGHT_KERNELS=64"
        ./min_conv_oc_span64_probe \
            --pattern --ic 3 --oc "$oc" --pad 1 --pad-con1 0 --out-zp 128 \
            --w-oc2 64 --w-kernels 64
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
