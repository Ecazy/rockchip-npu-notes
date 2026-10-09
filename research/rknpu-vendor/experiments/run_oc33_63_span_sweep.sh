#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${OC_SPAN_LOG:-"oc33_63_span_sweep_$(date +%Y%m%d_%H%M%S).log"}

gcc -O2 -o min_conv_oc_span_probe min_conv_exp.c -lm

{
    echo "# OC=33..63; per-tap span and WEIGHT_KERNELS both align to 2"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for oc in $(seq 33 63); do
        span=$(( (oc + 1) / 2 * 2 ))
        echo
        echo "### oc_real=${oc} span=${span} WEIGHT_KERNELS=${span}"
        ./min_conv_oc_span_probe \
            --pattern --ic 3 --oc "$oc" --pad 1 --pad-con1 0 --out-zp 128 \
            --w-oc2 "$span" --w-kernels "$span"
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
