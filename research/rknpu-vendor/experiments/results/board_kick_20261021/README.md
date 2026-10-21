# submit → kick 的**条件估计**（2026-10-21，板端）

> **修正（评审）**：本文原称「上界」，**已撤回**。估计量把 x* 放在阈值两侧两点的中点，
> 该中点可能**低于**真值（自测实例：真值 5.00 µs → 估计 4.50 µs，测试仍通过）；
> bootstrap 区间是估计量的散布，不是最大延迟保证。x* 还是
> kick 偏移 + 清除延迟 + 采样延迟 + 相位量化的**混合量**，不可分解相加；
> 采样线程与提交线程并行，分量不能串行相加。

- 负载：`min_conv_exp --fp16`（已验证 BIT-EXACT）、`min_conv_exp --fp16 --iw 32`（同）
- 相位：0–30 µs 以 1 µs 步进，另加 34/38/44/52/64/80 µs；每相位 12 次；两轮各 **384 run**
- 读回：`INTERRUPT_STATUS (0x0028)` + `RAW_STATUS (0x002c)`，每 run 一个快照
- 原始数据 `kickA/`、`kickB/`；分析 `analysis/{report.md,summary.json}`

## 结果

| 负载 | 样本 | 置位/未置位 | x* 估计 | 95% 区间 | 含义 |
|---|---|---|---|---|---|
| fp16 | 372 | 214/158 | 5.54 µs | 4.08–6.42 µs | 条件估计，**非上界** |
| iw32 | 373 | 213/160 | 5.10 µs | 4.81–6.42 µs | 条件估计，**非上界** |

两个负载的估计值接近（约 5 µs），区间重叠。**这不构成上界。**

## 推导（与它的前提）

在这个**条件模型**下：驱动在 commit 处写 `OPERATION_ENABLE`，
`hw_elapse = ktime_get()@job_done − hw_commit_time`（含中断与驱动路径）；
中断位由 IRQ 路径清除。于是置位条件写作 `φ < kick_rel + hw + clear_latency`。令 `x = φ − hw`，过渡点
`x* = kick_rel + clear_latency`——这是一个**混合量**，不是 kick 本身。

**这不是上界**：中点估计可能低于真值；"加 3 µs 延迟估计就更大"只说明估计量随延迟单调，
**不**说明它是上界。


## 没有证明的

- kick 的确切时刻；
- 清除延迟、中断延迟与采样延迟未分离；
- 驱动 `ktime_get()`（MONOTONIC）与采样器 RAW 是否同源：**未建立**；
- 不涉及任何 regcmd 位置或 per-regcmd 时长；
- 仅适用于本板、本驱动版本（0.9.8 / 6.1.43）与这两个负载。

## 附带：这次会话里两 task 链的负面结果（未用于任何结论）

| 尝试 | 结果 |
|---|---|
| `min_chain --tasks 2` | 提交成功（`ret=0`），但**自身数值校验 12/12 `MISMATCH`**（`mismatch=623/1024`，`max abs diff 255`，非舍入误差）；`--chain-next/--tail-self/--contig/--alias` 各变体结果相同 |
| `min_conv_exp --fp16 --chain-id`（及其 `--chain-pc/--chain-stub/--chain-align8` 变体） | **job 超时**：`SUBMIT ret=-1 errno=Connection timed out 6242 ms hw_elapse=0`，task1 输出 `UNTOUCHED`，dmesg `job timeout` + `soft reset, num: 6` |

**因此"已验证的两 task 数值正确负载"目前不存在**，多 task 的组位/完成窗口实验**未开展**。

**安全事件（按"时序上最可能、未证实"记录）**：`--chain-id` 那次 6.2 s 超时之后，板子
处于重启后状态（复查时 `up 1 min`，dmesg 从引导开始）。上一次引导的日志未持久化，
**无法证实**重启由该次超时引起。此后 `--chain-id` 已停用；工作区与全部产物完好，
板子当前正常（`BIT-EXACT`、power=on）。dmesg 错误计数在新引导下重新基线（18）。
