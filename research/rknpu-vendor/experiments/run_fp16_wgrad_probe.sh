#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
probe_log=${FP16_WGRAD_LOG:-"fp16_wgrad_$(date +%Y%m%d_%H%M%S).log"}
gcc -O2 -o min_conv_fp16_wgrad min_conv_exp.c -lm

common=(--fp16 --f16-pix-elems 8 --auto-fix --pad-con1 0 --oc 16
        --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0 --out-zp 128)

{
    echo "# fp16 wgrad: CNA K-axis reduction, fp16 input/weight, fp32 accumulator"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "# --- A) dot reduction P sweep (ic alignment 32) ---"
    for points in 32 36 40 48 64 96; do
        echo
        echo "### fp16 dot reduction P=${points}"
        ./min_conv_fp16_wgrad --wgrad-dot --pattern --ic "$points" "${common[@]}"
    done

    echo
    echo "# --- B) 3x3 conv wgrad coordinate matrix (P=36) ---"
    for ic in 0 1 2; do
        for ky in 0 1 2; do
            for kx in 0 1 2; do
                echo
                echo "### fp16 dW coordinate kx=${kx} ky=${ky} ic=${ic}, P=36"
                ./min_conv_fp16_wgrad \
                    --wgrad-conv3x3 --wgrad-kx "$kx" --wgrad-ky "$ky" --wgrad-ic "$ic" \
                    --hi-zero --ic 36 "${common[@]}"
            done
        done
    done

    echo
    echo "# --- C) batch accumulation (single coord kx=1 ky=1 ic=1) ---"
    for batch in 1 2 3; do
        points=$((36 * batch))
        echo
        echo "### fp16 batch=${batch} P=${points}"
        ./min_conv_fp16_wgrad \
            --wgrad-conv3x3 --wgrad-batch "$batch" --wgrad-kx 1 --wgrad-ky 1 --wgrad-ic 1 \
            --hi-zero --ic "$points" "${common[@]}"
    done
} | tee "$probe_log"

echo "Saved results to ${probe_log}"
