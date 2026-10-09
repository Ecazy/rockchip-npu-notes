#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${WGRAD_BATCH_LOG:-"wgrad_batch_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_wgrad_batch_probe min_conv_exp.c -lm

{
    echo "# Synthetic 3x3 wgrad partial sums across batches"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    for batch in 1 2 3; do
        points=$((36 * batch))
        echo
        echo "### batch=${batch} P=${points} coordinate=(kx=1,ky=1,ic=1)"
        ./min_conv_wgrad_batch_probe \
            --wgrad-conv3x3 --wgrad-batch "$batch" --wgrad-kx 1 --wgrad-ky 1 --wgrad-ic 1 \
            --hi-zero --auto-fix --ic "$points" --oc 16 \
            --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0 --out-zp 128
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
