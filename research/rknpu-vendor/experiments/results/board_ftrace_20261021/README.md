# 驱动事件 × 采样器时间对齐（2026-10-21，板端）

## 方法：不改内核，用 ftrace 取驱动时间点

`CONFIG_ROCKCHIP_RKNPU=y`（内建、无 `.ko`、无构建树）⇒ 改驱动要整内核重编换镜像，
在本板风险不可接受。改用 **ftrace**：

- `trace_clock` 支持 **`mono_raw`** —— 与采样器 CSV 同一时钟，可直接相减；
- `available_filter_functions` 里可追踪的 rknpu 函数包括：
  `__rknpu_submit_ioctl`、`rknpu_job_next.part.0`（commit）、`rknpu_core{0,1,2}_irq_handler`、
  `rknpu_power_get/put/on/off`、`rknpu_job_abort`、`rknpu_soft_reset`；
- 过滤器只开这 13 个函数，`function` tracer，运行 6 次，每次同时保存 trace 与采样 CSV。

（`kprobe_events` 不存在，kprobes 未编；`function`/`function_graph` 可用，已够用。
ftrace 时间戳分辨率 **1 µs**。）

## 结果（原点统一为采样器的 `submit_enter`，同一 mono_raw 时钟）

> 表中「调度路径入口」= `rknpu_job_next.part.0` 被 `rknpu_job_schedule` 调用的**函数入口**，
> 不是 `PC_OP_EN` 写入时刻（见下方源码核对）。

| 量 | p50 | min | max | n |
|---|---|---|---|---|
| hook 入口 → 真实 ioctl 入口 | **8.0 µs** | 6.2 | 14.0 | 6 |
| **`job_next` 调度路径入口**（kick 的下界） | **15.9 µs** | 13.2 | 33.0 | 6 |
| **IRQ 入口** | 124.1 µs | 65.0 | 151.3 | 6 |
| `hw_elapse − (IRQ 入口 − 调度入口)` | **6.5 µs** | 1.7 | 8.5 | 6 |
| 上升沿 − 调度入口（区间） | 2.8 ~ 15.2 µs | 0.7 | 19.2 | 2 |
| 下降沿 − IRQ handler 入口（区间） | −1.5 ~ 7.1 µs | −6.3 | 12.6 | 6 |

## 这一批数据推翻了什么、又没能推翻什么

> **二次修正（评审）**：本节最初写的是"实测 commit 13.2–33.0 µs""上升沿−commit = +0.7/+5.0 µs
> ⇒ 位是从 kick 起置位"。源码核对后，这两句都**超出了数据能支持的范围**，已按下面收紧。

**能确认的**：

1. **"submit→kick ≤ 5.5 µs" 作废**。原先的条件估计假设 kick 在提交入口附近；
   实测 `rknpu_job_next` 的**调度路径入口**在 hook 入口后 **13.2–33.0 µs**
   （其中 6.2–14.0 µs 是我们自己的 LD_PRELOAD wrapper 开销，此前完全未计入）。
   调度入口**早于** kick ⇒ 它只是 **kick 的下界**，kick 本身仍未测量。
2. **上升沿与调度入口只有区间关系**：`上升沿 − 调度入口 ∈ [2.8, 15.2] µs`（n=2）。
   它既不能证明"上升沿 = kick"，也不能支持"位就是作业在飞行标志"。

**没能推翻的**：位的硬件语义仍是**未识别**。

**顺带确认的两点**（仍按区间口径）：

- 下降沿 − IRQ handler 入口 = **−1.5 ~ +7.1 µs** ⇒ 清除发生在 IRQ 路径附近；
- `hw_elapse − (IRQ 入口 − 调度入口)` = **1.7–8.5 µs** ⇒ 与"在 `job_done` 内稍后取时"一致。

## 补充：PM / 作业生命周期（同批 + 3 次纯生命周期 trace）

另跑 3 次**不带采样器**的 trace，过滤器扩到
`rknpu_power_get/put/put_delay/on/off/delay_work`、`rknpu_job_free`、`rknpu_job_cleanup_work`。

生命周期序列（相对该 trace 首事件，通常 = 首个 GEM create）：

```
+0.0      power_get   ← GEM create
+4.0      power_on    ← power_get          ← 域在这里打开
+6410     power_get   ← submit ioctl
+6416     job_next    ← job_schedule       ← commit
+6601     job_next    ← irq_handler        ← 第二次调用（IRQ 路径派发下一个 job）
+6771     job_free    ← submit_ioctl
+11975…   power_put_delay ← gem_free_object
          （power_off 全程未出现）
```

**三条生命周期事实**：

1. **域是负载自己打开的**：首个 `rknpu_power_get`（GEM create）在 +4 µs 内触发 `rknpu_power_on`，
   比 commit 早数毫秒 ⇒ 采样器的电源复查能通过，靠的是这些 GEM 引用，**不是采样器**。
2. **`rknpu_power_off` 在 3/3 次运行里都没有出现**，后续运行也没有再次 `power_on`
   ⇒ **域跨运行保持开启**（延迟断电未在观测窗口内发生）。
3. **`rknpu_job_next.part.0` 每个 job 被调用两次**：一次来自 `rknpu_job_schedule`（commit），
   一次来自 IRQ handler（派发下一个 job），两次间隔中位 **177 µs**。
   **只有第一次是 kick** —— 分析器取第一次；这一点如果不查清，会把 IRQ 路径那次误当成第二次提交。

## 评审修正：口径收紧 + 源码核对

### 源码级：`job_next` 入口**不是** commit

```c
static void rknpu_job_next(struct rknpu_device *rknpu_dev, int core_index)
{
    if (rknpu_dev->soft_reseting) return;                       /* 早退 1 */
    ...
    if (subcore_data->job || list_empty(&subcore_data->todo_list)) {
        spin_unlock_irqrestore(&rknpu_dev->irq_lock, flags);
        return;                                                 /* 早退 2 */
    }
    job = list_first_entry(...); list_del_init(...); subcore_data->job = job;
    job->hw_commit_time = ktime_get();        /* 只有出队路径才设 */
    ...
    if (atomic_dec_and_test(&job->run_count)) rknpu_job_commit(job);   /* 真正写 PC_OP_EN */
}
```

- 两条早退路径**不设** `hw_commit_time`、**不写** `PC_OP_EN` ⇒ 函数入口只能叫**调度路径入口**；
- 真正的提交链 `rknpu_job_commit → rknpu_job_subcore_commit → ..._commit_pc` 全是
  `static`/`static inline`，**不在 `available_filter_functions`** ⇒ ftrace **无法**直接给出
  `PC_OP_EN` 写入时刻；
- 也查过 `rknpu_get_task_number`（可追踪）不在 `job_next` 路径上（只在
  `job_schedule`/`job_done`/`job_abort`），**不能**用它区分两条分支。

**因此：调度入口早于 kick，只能作为 `kick_rel` 的下界：`kick ≥ 15.9 µs`（p50）。**

单 job 场景的论证（源码级，非测量）：`job_schedule` 入队 1 个 job → `job_done` 清空
`subcore_data->job` 后调 `job_next`，此时 todo_list 已空 ⇒ **第二次调用必然走早退 2**。
本批 trace 里 `job_schedule` 调用数 = **1**（与单 job 一致），`job_next` 调用数 = 2。

### 分析器修正

| 问题 | 修正 |
|---|---|
| `parse_trace()` 把第一次 `job_next` 直接命名为 commit | 改为**按调用来源分类**：`sched_entry`（← `job_schedule`）/ `irq_entry`（← `irq_handler`）/ `other_entry`；并记录 `job_schedule` 调用数 |
| 边沿差值只取下端点 | 改为**完整区间**：`rise − 调度入口 ∈ (lo, hi]`、`fall − IRQ 入口 ∈ (lo, hi]` |
| 无 trace 丢失检查 | 新增：解析 `entries-in-buffer/entries-written`，报告 `lost` |
| 事件语义被当成"kick / 作业结束" | 报告新增「语义（严格）」小节，保留为**未识别的硬件动作** |

**修正后的实际数值**（n=6）：

| 量 | 结果 |
|---|---|
| 上升沿 − 调度入口 | **2.8 ~ 15.2 µs**（区间，非 +0.7/+5.0 两个点） |
| 下降沿 − IRQ 入口 | **−1.5 ~ +7.1 µs**（区间） |
| `job_next` / `job_schedule` 调用数 | 2 / 1 |
| trace 丢失 | **0/9 份 trace**（全部 `in_buffer == written`） |

### tracing 开销：**测不出来**

同一脚本两次 80 submit（40/臂，ABBA 交错，全部 BIT-EXACT）：

| 轮次 | plain 中位 | tracing 中位 | 差 | 95% CI |
|---|---|---|---|---|
| 第 1 次 | 65.5 µs | 84.3 µs | +18.8 µs | — |
| 第 2 次 | 105.9 µs | 110.7 µs | **+4.8 µs** | **[−20.0, +29.2] µs** |

- 两次结论不一致，且 **plain 臂自身中位数从 65.5 漂到 105.9 µs** ⇒ 负载漂移远大于 tracing 效应；
- 双峰比例在第二次实验里也不稳（12% vs 18%）；
- **结论：在本负载上无法分辨 tracing 开销**，CI 含 0。既不支持"开销可忽略"，也不支持"+19 µs"。
  使用 ftrace 时刻做跨条件比较时必须记住这个噪声底。

## 仍然没有的

- 只有 6 次运行、1 个负载；ftrace 1 µs 分辨率；
- 不涉及任何 regcmd 位置或 per-regcmd 时长；
- 位与提交奇偶的对应仍不等于已识别执行 bank；
- **kick 本身仍未测量**，只有"调度入口"这个下界；`PC_OP_EN` 写入时刻需要不可内联的观测点（改驱动或 kprobe，本板均不可用）；
- tracing 开销无法与负载漂移分离；
- 生命周期只覆盖 1 个负载、3 次运行；延迟断电的实际时刻未观测到（超出窗口）；
- 未观测到 `power_off`，因此"域保持开启多久"没有上界证据。

## 复现

```sh
# 板端（过滤器 + mono_raw + 6 次运行）
./ftrace_probe.sh
# 离线
python3 analyze_driver_events.py --dir ftrace --out-dir analysis
```
