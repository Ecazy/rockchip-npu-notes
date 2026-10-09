#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${WGRAD_CONV_MATRIX_LOG:-"wgrad_conv3x3_matrix_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_wgrad_conv3x3 min_conv_exp.c -lm

{
    echo "# Synthetic valid 3x3 conv wgrad: input 8x8x3, dY 6x6x16"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for ic in 0 1 2; do
        for ky in 0 1 2; do
            for kx in 0 1 2; do
                echo
                echo "### dW coordinate kx=${kx} ky=${ky} ic=${ic}, P=36"
                ./min_conv_wgrad_conv3x3 \
                    --wgrad-conv3x3 --wgrad-kx "$kx" --wgrad-ky "$ky" --wgrad-ic "$ic" \
                    --hi-zero --auto-fix --ic 36 --oc 16 \
                    --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0 \
                    --out-zp 128
            done
        done
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
