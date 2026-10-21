# 驱动事件与采样器的时间对齐（ftrace `mono_raw`）

ftrace 记录在 **mono_raw**，与采样器 CSV 同一时钟，因此可以直接相减；原点统一取采样器的 `submit_enter`。ftrace 时间戳分辨率为 1 µs。

| 量 | p50 | min | max | n |
|---|---|---|---|---|
| hook 入口 → 真实 ioctl 入口（我们自己的 wrapper 开销） | 8.0 | 6.2 | 14.0 | 6 |
| **`rknpu_job_next` 调度路径入口**（来自 `job_schedule`） | 15.9 | 13.2 | 33.0 | 6 |
| **IRQ handler 入口** | 124.1 | 65.0 | 151.3 | 6 |
| `hw_elapse − (IRQ 入口 − 调度入口)` | 6.5 | 1.7 | 8.5 | 6 |
| 上升沿 − 调度路径入口（区间） | 2.8 ~ 15.2 | 0.7 | 19.2 | 2 |
| 下降沿 − IRQ handler 入口（区间） | -1.5 ~ 7.1 | -6.3 | 12.6 | 6 |
| `job_next` 调用次数 / `job_schedule` 调用次数 | 2 / 1 | | | | |

## 语义（严格）

- `rknpu_job_next.part.0` 的**函数入口**只叫「调度路径入口」。源码里它有两条早退（`soft_reseting`、`subcore_data->job` 忙或 todo_list 空），早退路径**不设** `hw_commit_time`，也**不写** `PC_OP_EN`；
- 真正的提交在 `rknpu_job_commit → rknpu_job_subcore_commit → ..._commit_pc`，全是 `static`/`static inline`，**不在 `available_filter_functions`** 里 ⇒ ftrace 无法直接给出 `PC_OP_EN` 写入时刻；
- 因此：调度入口**早于** kick，只能作为 `kick_rel` 的**下界**；
- 两次 `job_next` 调用来源不同（`job_schedule` / `irq_handler`），**不能按次数统计提交或 task**；第二次也可能只是早退检查。
- 位与事件的关系只用**区间**表达，事件语义保留为「未识别其对应的硬件动作」。

## 没有证明的

- 只覆盖这些运行与这个负载；ftrace 分辨率 1 µs；
- 不涉及 regcmd 位置或 per-regcmd 时长；
- 位与提交奇偶的对应仍不等于已识别执行 bank。

