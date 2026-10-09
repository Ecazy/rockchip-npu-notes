#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${WGRAD_MATRIX_LOG:-"wgrad_coeff_matrix_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_wgrad_matrix_probe min_conv_exp.c -lm

{
    echo "# Synthetic 3x3, ic=3 wgrad coefficients; P=64 spatial dot per weight"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for ic in 0 1 2; do
        for ky in 0 1 2; do
            for kx in 0 1 2; do
                echo
                echo "### weight coordinate kx=${kx} ky=${ky} ic=${ic}"
                ./min_conv_wgrad_matrix_probe \
                    --wgrad-dot --wgrad-kx "$kx" --wgrad-ky "$ky" --wgrad-ic "$ic" \
                    --auto-fix --ic 64 --oc 16 --iw 4 --ih 4 --ow 1 --oh 1 \
                    --kw 1 --kh 1 --pad 0 --out-zp 128
            done
        done
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
