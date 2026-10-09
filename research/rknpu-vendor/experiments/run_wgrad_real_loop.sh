#!/usr/bin/env bash
# run_wgrad_real_loop.sh — 真实张量 wgrad 闭环（在板端 exp3 目录运行）
# 用法：./run_wgrad_real_loop.sh   [WGRAD_REAL_LOG=... 可覆盖日志名]
set -u
cd "$(dirname "$0")"
log=${WGRAD_REAL_LOG:-wgrad_real_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o wgrad_real wgrad_real.c -lm || { echo "BUILD FAILED"; exit 1; }

{
    echo "# wgrad_real closed-loop run  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    echo "npu_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null)"
    echo "thermal=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')"

    echo
    echo "## Phase A: pad1 B=1 full closed loop (forward + 27 wgrad + sat + update + re-forward)"
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 1 --update
    echo "phaseA_rc=$?"

    echo
    echo "## Phase B: valid(pad0) B=1 full closed loop"
    ./wgrad_real --dir wgdata_valid --pad 0 --batch 1 --update
    echo "phaseB_rc=$?"

    echo
    echo "## Phase C: pad1 B=2 batch accumulation"
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 2
    echo "phaseC_rc=$?"

    echo
    echo "## Phase D: pad1 B=3 batch accumulation"
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 3
    echo "phaseD_rc=$?"

    echo
    echo "## Phase E: K capacity scan (synthetic, coord kx=1 ky=1 ic=1)"
    ./wgrad_real --kscan 8 --pad 1
    echo "kscan_rc=$?"

    echo
    echo "## Baseline recheck (board health after any failure)"
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 1 | tail -3
} 2>&1 | tee "$log"

echo "SAVED: $log"
