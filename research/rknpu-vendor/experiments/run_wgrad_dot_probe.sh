#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${WGRAD_DOT_LOG:-"wgrad_dot_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_wgrad_dot min_conv_exp.c -lm

{
    echo "# wgrad dot reduction: CNA K-axis reduces spatial positions for one kernel element"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for points in 32 64 96; do
        echo
        echo "### reduction points=${points}"
        ./min_conv_wgrad_dot \
            --wgrad-dot --auto-fix --ic "$points" --oc 16 \
            --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0 \
            --out-zp 128
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
