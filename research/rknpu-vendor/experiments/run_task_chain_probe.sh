#!/usr/bin/env bash
# run_task_chain_probe.sh — §7.42「一次 submit 多 task」语义实测（--chain-id / --task-count）
# 在板端 exp2 目录运行；用法：./run_task_chain_probe.sh [CHAIN_LOG=...]
#
# 结论（本轮实测）：
#   · `--chain-id`：把同一 conv 作为 task0（→OUT_OFF）与 task1（→OUT_OFF+256KB）放进**一次**
#     submit（task_number=2，两段 regcmd 各自的 regcmd_addr/regcfg_amount）。
#     结果：task0 输出正确（OUT-HASH 与基线一致），**task1 的输出区仍是全零**
#     （CHAIN-HASH = FNV1a(8192×0x00) = bcc31dc5 ⇒ 该区从未被写过）⇒ **task1 的 regcmd 未被执行**。
#   · `--task-count N`（N=1..4，同一 task 重复）：min `hw_elapse` **基本不变**
#     （20708 / 20416 / 19833 / 19542 ns）⇒ 多出来的 task **不产生额外工作量**。
#   · 把 `sub.task_base_addr` 设成 task BO 的 DMA 地址（让 PC 自己按基址取 task 描述符）
#     ⇒ PC-DMA 取指失败、submit 6 s 超时（**不要重试**，每次一个 soft reset）。
# ⇒ 判定：本硬件/驱动下**一次 submit 内的多个 task 不是「顺序依赖链」**，而是**按核分发的
#   并行工作项**（驱动 `subcore_task[i].task_start/task_number` + `core_mask` 决定每个核跑哪个
#   task；同一核上只有第一个 task 的 regcmd 经 PC_DATA_ADDR/AMOUNT 真正执行）。
#   这与 vendor trace 一致（同一 sequence 内多 task、同 op_idx、各自不同 regcmd_addr），
#   也解释了 skill 里「按 core_mask/subcore_task 归属提交」的用法。
set -u
cd "$(dirname "$0")"
log=${CHAIN_LOG:-task_chain_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"
G="--kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16 --oc 16 --warmup-submits 0"

zero_hash() {   # FNV1a32 over N zero bytes
    local n=$1 h=2166136261 i
    for ((i = 0; i < n; i++)); do h=$(( (h ^ 0) * 16777619 & 0xffffffff )); done
    printf "%08x" "$h"
}

{
    echo "# §7.42 一次 submit 多 task 语义实测  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 基线（单 task）"
    ./min_conv_exp $F $G 2>&1 | grep -E "SUBMIT ret|OUT-HASH|RESULT:"

    echo
    echo "## B) 2-task 链（task0 → OUT_OFF，task1 同几何 → OUT_OFF+256KB，task_number=2）"
    ./min_conv_exp $F $G --chain-id 2>&1 | grep -E "CHAIN|SUBMIT ret|OUT-HASH|RESULT:"

    echo
    echo "## C) task_number 对硬件时间的影响（同一 task 重复 N 次，每档 3 次取 min）"
    for tc in 1 2 3 4; do
        best=999999999
        for i in 1 2 3; do
            hw=$(./min_conv_exp $F $G --task-count $tc 2>&1 | grep -oE "hw_elapse=[0-9]+" | head -1 | cut -d= -f2)
            [ -n "$hw" ] && [ "$hw" -lt "$best" ] && best=$hw
        done
        printf "  task-count=%d  min_hw_elapse=%s ns\n" "$tc" "$best"
        sleep 0.3
    done

    echo
    echo "## D) 参考：task1 输出区未写入时的 hash（全零区 FNV1a）"
    echo "  fnv1a(8192 x 0x00) = $(zero_hash 8192)   # 与 B 段 CHAIN-HASH 对比"

    echo
    echo "## F) 多核派发（§7.43：core_mask=0x3 + 每核一个不同 task；对照 vendor 的 flags=0x5）"
    for fl in 7 5; do
        for r in 1 2; do
            echo "  -- job-flags=$fl 第 $r 次 --"
            ./min_conv_exp $F $G --chain-id --multi-core --job-flags $fl 2>&1 | \
                grep -E "MULTICORE|SUBMIT ret|OUT-HASH|CHAIN-HASH" | sed 's/^/    /' 
            sleep 0.5
        done
    done
    echo "  （判读：CHAIN-SUMMARY 的「已写入/与 task0 逐位相同」计数）"

    echo
    echo "## H) 仿 vendor 最小形状：N=3 个**连续** regcmd 块（§7.44）"
    for cfg in "--job-flags 5" "--job-flags 7" "--job-flags 5 --multi-core"; do
        echo "  -- N=3 3x3 $cfg --"
        ./min_conv_exp $F $G --chain-id --chain-n 3 $cfg 2>&1 | \
            grep -E "MULTICORE|CHAIN: |CHAIN-TASK|CHAIN-SUMMARY|SUBMIT ret" | sed 's/^/    /'
        sleep 0.5
    done
    echo "  -- N=3 1x1 最小几何（每段仍 ~130 entries）flags=5 --"
    ./min_conv_exp $F --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 8 --oc 8 --warmup-submits 0 \
        --chain-id --chain-n 3 --job-flags 5 2>&1 | \
        grep -E "CHAIN-TASK|CHAIN-SUMMARY|SUBMIT ret" | sed 's/^/    /'

    echo
    echo "## I) vendor 侧 task 块解剖（离线；见 results/multicore_vendor_20260930/vendor_regcmd_blocks.txt）"
    echo "  attn_relu 的 task 块只含 **DPU 寄存器 0x4004..0x5004**，每块 34/34/4 entries，"
    echo "  逐 task 递增 0x4020（+0x400）⇒ 多 task = **批量发射同一个 DPU 小算子**（不是 conv 链）。"

    echo
    echo "## G) vendor 侧口径（离线，见 results/multicore_vendor_20260930/vendor_submit_rules.txt）"
    echo "  core_mask=0(AUTO)、flags=0x5(无 PINGPONG)、task_base_addr=0、regcfg_offset=0、"
    echo "  subcore_task[0..2]={0, N/3}；task_number 最大 3663；hw=7.19 ms ⇒ ~1.96 µs/task（全部执行过）"

    echo
    echo "## E) 未纳入脚本（每次一个 soft reset，且历史上有楔死风险）："
    echo "  把 sub.task_base_addr 设为 task BO 的 DMA 地址 ⇒ PC-DMA 取指超时（已实测一次，未复现重试）。"
} 2>&1 | tee "$log"

echo "SAVED: $log"
