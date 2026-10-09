#!/usr/bin/env bash
# run_table_irq_probe.sh — §7.39 带表 DPU 提交的「逐次交替超时」根因刻画
# 在板端 exp2 目录运行；用法：./run_table_irq_probe.sh [IRQ_LOG=...]
#
# 现象（本轮刻画）：
#   · 带 DPU 表的提交（BS/BRDMA 或 BN/NRDMA）**每次进程内逐次交替**：
#     submit#1 OK、#2 TIMEOUT(6 s)、#3 OK、#4 TIMEOUT …（与 job_flags 5/7、S_POINTER
#     `--pp-val`、表内容、单次 submit 的 task 数 1/2/3 **全部无关**）。
#   · 纯 conv（无表）从不交替（含 2-task 链）。
#   · 失败的提交硬件其实**做完了**：`hw_elapse` 有值（µs 级）、超时日志里
#     `task counter: 0x0`（PC 无待处理任务）。
#   · 失败提交在超时时刻的 `INT_RAW_STATUS` = `0xc000002a` / `0xc0000015`（低 6 位
#     奇/偶位互补 + 位 30/31）；且 dmesg 中 **没有** `invalid irq status`（0 次）
#     ⇒ **完成中断根本没进驱动的 IRQ 处理函数**（不是"进了但不匹配掩码"）。
# 驱动机制（源码 refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c）：
#   IRQ 判定是 **等值** `rknpu_fuzz_status(status) != job->int_mask[core_index]`，
#   `fuzz_status` 把 6 对位（0x3/0xc/0x30/0xc0/0x300/0xc00）各自整对放大；CONV 完成是
#   0x300、PPU 完成是 0xc00。固定 int_mask 无法匹配"逐次翻转"的状态相位。
# 结论：这是**驱动/中断层面**的带表路径缺陷（硬件完成但中断不来），不是寄存器配置错误；
#   实用配方 = 带重试（并让"被测量的一次"落在好相位），**不要把 timeout 记为语义失败**。
set -u
cd "$(dirname "$0")"
log=${IRQ_LOG:-table_irq_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"
BS="$F --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 --in-ch-id --w-ones \
    --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000 --warmup-submits 3"
BN="$F --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 \
    --bn-cfg 0x20150 --bn-nrdma 0x2 --bn-table-f32-bias 1.5 --warmup-submits 3"

show() {  # desc args...
    local desc="$1"; shift
    local out seq
    out=$(timeout 60 ./min_conv_exp "$@" 2>&1)
    seq=$(echo "$out" | grep -oE "WARMUP [0-9]/[0-9] ret=[0-9-]+" | sed -E 's/.*ret=//' | tr '\n' ',' )
    printf "%-34s 提交序列 ret=[%s]  %s\n" "$desc" "${seq%,}" \
           "$(echo "$out" | grep -oE 'RESULT: [A-Z-]+' | tail -1)"
    sleep 1
}

{
    echo "# §7.39 带表 DPU 提交「逐次交替超时」刻画  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 现象：3 次热身 + 1 次测量（ret 序列，0=ok / -1=ETIMEDOUT）"
    show "BS/BRDMA 表路径" $BS
    show "BN/NRDMA 表路径" $BN
    show "纯 conv（对照，无表）" $F --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 --warmup-submits 3

    echo
    echo "## B) 不变量：改 job_flags / pp_val / 表内容 / task 数，序列不变"
    show "BS --job-flags 5"        $BS --job-flags 5
    show "BS --pp-val 0"           $BS --pp-val 0
    show "BS --brdma-bias 0"       $BS --brdma-bias 0
    show "BS --task-count 2"       $BS --task-count 2
    show "BS --task-count 3"       $BS --task-count 3

    echo
    echo "## C) 失败提交的硬件侧证据（dmesg）"
    dmesg > /tmp/irq_probe_dmesg.txt
    echo "  job timeout 计数                : $(grep -c 'job timeout' /tmp/irq_probe_dmesg.txt)"
    echo "  invalid irq status 计数         : $(grep -c 'invalid irq status' /tmp/irq_probe_dmesg.txt)"
    echo "  soft reset 计数                 : $(grep -c 'soft reset' /tmp/irq_probe_dmesg.txt)"
    echo "  require mask 分布               : $(grep -o 'require mask: 0x[0-9a-f]*' /tmp/irq_probe_dmesg.txt | sort | uniq -c | tr '\n' ' ')"
    echo "  raw status 分布（前 5）         : $(grep -o 'raw status: 0x[0-9a-f]*' /tmp/irq_probe_dmesg.txt | sort | uniq -c | sort -rn | head -5 | tr '\n' ' ')"
    echo "  最近一条超时明细："
    grep -A1 "job timeout" /tmp/irq_probe_dmesg.txt | tail -2 | sed 's/^/    /'
} 2>&1 | tee "$log"

echo "SAVED: $log"
