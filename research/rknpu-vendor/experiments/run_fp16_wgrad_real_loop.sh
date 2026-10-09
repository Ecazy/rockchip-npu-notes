#!/usr/bin/env bash
# run_fp16_wgrad_real_loop.sh — fp16 真实张量 wgrad 闭环（在板端 exp3 目录运行）
# 用法：./run_fp16_wgrad_real_loop.sh   [FP16_WGRAD_REAL_LOG=... 可覆盖日志名]
set -u
cd "$(dirname "$0")"
log=${FP16_WGRAD_REAL_LOG:-fp16_wgrad_real_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o wgrad_real wgrad_real.c -lm || { echo "BUILD FAILED"; exit 1; }

{
    echo "# fp16 wgrad_real closed-loop run  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    echo "npu_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null)"
    echo "thermal=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')"

    echo
    echo "## Phase F16-1: pad1 B=1 closed loop (fwd + 27 wgrad + update + re-fwd)"
    ./wgrad_real --fp16 --pad 1 --batch 1 --update
    echo "rc=$?"

    echo
    echo "## Phase F16-0: valid(pad0) B=1 closed loop"
    ./wgrad_real --fp16 --pad 0 --batch 1 --update
    echo "rc=$?"

    echo
    echo "## Phase F16-B: pad1 batch accumulation B=2,3"
    for b in 2 3; do
        echo "### fp16 pad1 batch=${b}"
        ./wgrad_real --fp16 --pad 1 --batch "$b" --update
        echo "rc=$?"
    done

    echo
    echo "## Phase F16-lr: lr/loss sweep (pad1 B=1)"
    for cfg in "--lr 0.5 --loss 0.125" "--lr 0.125 --loss 0.5" "--lr 1.0 --loss 1.0"; do
        echo "### fp16 $cfg"
        ./wgrad_real --fp16 --pad 1 --batch 1 --update $cfg
        echo "rc=$?"
    done

    echo
    echo "## Phase F16-seed: seed sweep (pad1 B=1)"
    for s in 1 2 999; do
        echo "### fp16 seed=${s}"
        ./wgrad_real --fp16 --pad 1 --batch 1 --seed "$s" --update
        echo "rc=$?"
    done

    echo
    echo "## Baseline recheck: int8 closed loop unaffected"
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 1 --update | tail -3
} 2>&1 | tee "$log"

echo "SAVED: $log"
