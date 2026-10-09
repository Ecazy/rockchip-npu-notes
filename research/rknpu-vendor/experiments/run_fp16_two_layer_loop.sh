#!/usr/bin/env bash
# run_fp16_two_layer_loop.sh — fp16 双层真实网络闭环（跨层前向 + 分层 wgrad + 权重更新）
# 在板端 exp3 目录运行；用法：./run_fp16_two_layer_loop.sh
#   [FP16_TWO_LAYER_LOG=... 可覆盖日志名] [TL_SWEEP=0 跳过 30×3 宽扫描]
set -u
cd "$(dirname "$0")"
log=${FP16_TWO_LAYER_LOG:-fp16_two_layer_$(date +%Y%m%d_%H%M%S).log}
sweep=${TL_SWEEP:-1}

gcc -O2 -Wall -o wgrad_real wgrad_real.c -lm || { echo "BUILD FAILED"; exit 1; }

{
    echo "# fp16 two-layer closed-loop run  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
    echo "npu_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null)"
    echo "thermal=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')"

    echo
    echo "## Phase L2-pad1: 跨层闭环 pad1 B=1,2,3"
    for b in 1 2 3; do
        echo "### two-layer pad1 batch=${b}"
        ./wgrad_real --two-layer --pad 1 --batch "$b" --seed 20260928 --update
        echo "rc=$?"
    done

    echo
    echo "## Phase L2-pad0: 跨层闭环 pad0 B=1,2,3"
    for b in 1 2 3; do
        echo "### two-layer pad0 batch=${b}"
        ./wgrad_real --two-layer --pad 0 --batch "$b" --seed 20260928 --update
        echo "rc=$?"
    done

    echo
    echo "## Phase L2-seed: seed sweep (pad1 B=1)"
    for s in 1 2 999; do
        echo "### two-layer seed=${s}"
        ./wgrad_real --two-layer --pad 1 --batch 1 --seed "$s" --update
        echo "rc=$?"
    done

    echo
    echo "## Phase L2-lr: lr/loss sweep (pad1 B=1)"
    for cfg in "--lr 0.5 --loss 0.125" "--lr 0.125 --loss 0.5" "--lr 1.0 --loss 1.0"; do
        echo "### two-layer $cfg"
        ./wgrad_real --two-layer --pad 1 --batch 1 --seed 1 --update $cfg
        echo "rc=$?"
    done

    if [ "$sweep" = 1 ]; then
        echo
        echo "## Phase L2-sweep: pad0/1 × seed1..30 × B1..3 判定汇总"
        pass=0; fail=0
        for p in 0 1; do
            for s in $(seq 1 30); do
                for b in 1 2 3; do
                    r=$(./wgrad_real --two-layer --pad "$p" --batch "$b" --seed "$s" --update 2>&1 | grep '^RESULT:')
                    case "$r" in
                        *OK*) pass=$((pass+1)) ;;
                        *)    fail=$((fail+1)); echo "FAIL pad=$p seed=$s b=$b: $r" ;;
                    esac
                done
            done
        done
        echo "SWEEP: PASS=$pass FAIL=$fail"
    fi

    echo
    echo "## Baseline recheck: 单层 fp16 + int8 闭环不受影响"
    ./wgrad_real --fp16 --pad 1 --batch 1 --update | tail -1
    ./wgrad_real --dir wgdata_pad1 --pad 1 --batch 8 --update | tail -1
} 2>&1 | tee "$log"

echo "SAVED: $log"
