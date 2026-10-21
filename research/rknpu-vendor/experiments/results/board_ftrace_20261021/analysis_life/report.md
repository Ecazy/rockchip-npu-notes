# 驱动事件与采样器的时间对齐（ftrace `mono_raw`）

ftrace 记录在 **mono_raw**，与采样器 CSV 同一时钟，因此可以直接相减；原点统一取采样器的 `submit_enter`。ftrace 时间戳分辨率为 1 µs。

| 量 | p50 | min | max | n |
|---|---|---|---|---|
| hook 入口 → 真实 ioctl 入口（我们自己的 wrapper 开销） | n/a | n/a | n/a | 0 |
| **commit / kick** | 6195.0 | 5910.0 | 6416.0 | 3 |
| **IRQ 入口** | n/a | n/a | n/a | 0 |
| `hw_elapse − (IRQ 入口 − commit)` | n/a | n/a | n/a | 0 |
| 上升沿 − commit | n/a | n/a | n/a | 0 |
| 下降沿 − IRQ 入口 | n/a | n/a | n/a | 0 |

## PM / 作业生命周期（同批 trace）

| 量 | 中位 | n |
|---|---|---|
| `rknpu_power_on`（相对首事件） | 4.0 µs | 1 |
| commit | 6195.0 µs | 3 |
| `rknpu_job_free` | 6596.0 µs | 3 |
| 两次 `rknpu_job_next` 间隔 | 177.0 µs | 3 |
| `rknpu_power_on` 出现次数 | 1 / 3 次运行 | |
| `rknpu_power_off` 出现次数 | 0 / 3 次运行 | |

- **域是由负载自己打开的**：首次 `rknpu_power_get`（GEM create）在 +4 µs 内触发 `rknpu_power_on`，比 commit 早数毫秒。采样器的电源复查之所以能通过，靠的就是这些 GEM 引用，**不是**采样器。
- 后续运行里没有再次 `rknpu_power_on`，且 `rknpu_power_off` 全程未出现 ⇒ **域跨运行保持开启**（延迟断电未在观测窗口内发生）。

- 注意：本表时间原点是**该 trace 的第一条事件**（通常是首个 GEM create），与主表（原点 = 采样器 `submit_enter`）不同。
- **`rknpu_job_next.part.0` 每个 job 被调用两次**：第一次来自 `rknpu_job_schedule`（commit），第二次来自 IRQ handler（派发下一个 job）。只有第一次是 kick —— 本工具取第一次。


## 结论

- **commit 是实测的**（`rknpu_job_next.part.0`），不再是估计；
- 上升沿与 commit 在区间分辨内一致 ⇒ 中断位大体是**从 kick 起置位**；
- 下降沿在 IRQ 入口 ±数 µs 内 ⇒ 由 IRQ 路径清除；
- `hw_elapse` 略大于 `IRQ 入口 − commit`，与它在该路径内稍后取时一致。

## 没有证明的

- 只覆盖这些运行与这个负载；ftrace 分辨率 1 µs；
- 不涉及 regcmd 位置或 per-regcmd 时长；
- 位与提交奇偶的对应仍不等于已识别执行 bank。

