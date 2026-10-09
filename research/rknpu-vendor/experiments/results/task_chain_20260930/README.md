# 一次 submit 多 task 的语义实测（§7.42）

回答「vendor trace 里一条 submit 有多个 task（同 `op_idx`、各自 `regcmd_addr`/`regcfg_amount`）
到底怎么执行」以及「能否用多 task 做顺序依赖链」。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8（源码 `refs/rknpu-driver-0.9.8/`）。
复跑：板端 exp2 `bash run_task_chain_probe.sh`（日志 `task_chain_probe.txt`）。

## 实现：`--chain-id` 真 2-task submit

发射器新增 `--chain-id`：在同一次 submit 里放两个 task ——
- task0 = 主 conv（cfg 原样，输出 → `OUT_OFF`）；
- task1 = **同几何**的第二个 conv（输出 → `OUT_OFF + 256 KB`，可用 `--chain-off` 覆盖）；
两段 regcmd 写进 RCBO 的不同偏移（`ALIGN_UP(n*8,64)`），task 描述符各自填
`regcmd_addr`/`regcfg_amount`，`task_number=2`；输出区提交前按零点填充。

## 实测结果

| 项 | 观测 | 判读 |
| --- | --- | --- |
| 提交 | `SUBMIT ret=0`，`hw_elapse≈19.8 µs` | 链式 submit 本身合法、不超时 |
| task0 输出 | `OUT-HASH 533d6b86` = 单 task 基线 | task0 的 regcmd 正常执行 |
| task1 输出 | `CHAIN-HASH bcc31dc5`，`diff_vs_task0=7620/8192` | **该区从未被写**（`bcc31dc5` = `FNV1a(8192×0x00)`，本文件 D 段现算验证） |
| `--task-count N`（N=1..4） | min `hw_elapse` 18.1 / 21.3 / 23.9 / 20.7 µs | **无线性增长**（≤1.3× 板噪声）⇒ 多出的 task 不产生工作量 |
| `sub.task_base_addr = task BO dma` | PC-DMA 取指失败、6 s 超时（一次） | **不要重试**；PC_DMA_BASE_ADDR 语义待解 |

## 判定

**同一核上、一次 submit 内的多个 task 不是「顺序依赖链」**：只有该核第一个 task 的 regcmd
经 `PC_DATA_ADDR`/`PC_DATA_AMOUNT`（驱动用 first task 的值编程）真正执行；task 数组更多是给
**驱动**用的：按核分发工作项（`subcore_task[i].task_start/task_number` + `task_mask`）以及
从 **last task** 取完成中断掩码（`job->int_mask[core] = last_task->int_mask`）。

这与 vendor trace 一致：同一 `sequence` 内多 task、**同一个 `op_idx`**、各自不同的 `regcmd_addr`
⇒ 那些 task 是**按核并行的切片**，不是链；也解释了 skill 里「用 `core_mask`/`subcore_task`
把提交归属到核」的做法。

⇒ 想做顺序依赖（conv→conv 不落 CPU）只能体现在**单条 regcmd 流内部**（一个 task 可携带很长
的寄存器序列 / 多次 DPU 处理），或用多次 submit；真正的多核并行分发 = `core_mask` + 每核 task
（历史上 `--subcore 2/3/4` 有挂内核记录，仍是未闭合高风险区）。

## 产物

- `experiments/min_conv_exp.c`：`--chain-id`、`--chain-off`、`CHAIN-HASH` 校验。
- `experiments/run_task_chain_probe.sh`、`task_chain_probe.txt`。
