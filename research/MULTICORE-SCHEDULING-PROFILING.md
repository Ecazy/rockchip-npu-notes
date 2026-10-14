# RK3588 可控多核分配与调度 Profiling

2026-10-09，Orange Pi 5 Plus / RK3588 / FP16 / 厂商 RKNPU 0.9.8。

已实现逐次 Submit 指定核，并在真机上用驱动回写 Mask、核 IRQ、独立输出数值三类证据核验。
已把提交请求的排队时间与驱动服务时间分离：排队时间输出可信区间，服务时间使用驱动返回值并与 trace 交叉验证。
仍未获得精确的链表入队瞬间和纯硬件开始计算时间；不会用 `wall - hw_elapse_time` 代替排队时间。

## 多核控制

[co_schedule.h](custom-ops/include/co_schedule.h) / [co_schedule.c](custom-ops/src/co_schedule.c) 提供局部提交适配层，绕开上游 provider 的进程级环境变量缓存。无需修改上游依赖。

| 调用参数 | 含义 | 验证方式 |
|---|---|---|
| `mask=1` | Core 0 | 回写 1，Core 0 IRQ，数值正确 |
| `mask=2` | Core 1 | 回写 2，Core 1 IRQ，数值正确 |
| `mask=4` | Core 2 | 回写 4，Core 2 IRQ，数值正确 |
| `mask=0` | 厂商 AUTO，在三个核中选择 | 读取回写 Mask，再匹配对应核 IRQ |
| `co_schedule_pick(allowed, ordinal, &mask)` | 用户态在指定集合内轮询，随后提交选出的单核 Mask | 回写和 IRQ 必须属于 allowed 集合 |

**不能把 3、5、6、7 直接当成厂商候选核集合。** 当前驱动把这些位组合解释为多核联合执行。
适配层拒绝它们作为 Submit Mask；例如候选集合 `allowed=3` 由用户态轮询选择 `1` 或 `2`，再严格单核提交。
这不是驱动在候选集合中自动择核，策略和实际行为均有明确记录。

```c
unsigned mask;
co_schedule_profile profile;
// independent_job_i -> Core_(i % 3)
mask = 1u << (i % 3);
int rc = co_schedule_execute(fd, programs, count, mask, -1, request_id, &profile);

// 只允许 Core 0 / Core 1：用户态选核，然后单核提交。
co_schedule_pick(3, i, &mask);
rc = co_schedule_execute(fd, programs, count, mask, -1, request_id, &profile);
```

链接 CMake 目标 `co_schedule`。`programs` 来自现有 `co_emit_*`，所有输入/输出 BO 使用同一个 fd。
当前适配器支持 1–64 个 DPU program，每个调用一次阻塞 Submit；PPU 尚未接入此适配器。
`trace_marker_fd=-1` 可关闭标记。

调度粒度是 **job / 一次 Submit**：有依赖的 task 链整体留在一个核上，维持设备 Cube 交接。
需要不同核的独立计算应拆成独立 job。当前没有链内部跨核迁移、抢占或每个描述符的独立调度时间戳。
并行提交由调用方的独立线程/进程安排；候选策略目前是轮询，不是负载预测调度器。

## 实际可观测的时间点

所有 Host 时间使用 `CLOCK_MONOTONIC`；function graph 使用 `trace_clock=mono`。纳秒是存储单位，不意味着纳秒精度。

| 时间点 | 数据字段 | 精度及语义 |
|---|---|---|
| Host 请求开始 | `host_begin_ns` | API 入口附近，包含本次命令/描述符准备；输入生成和上传在此前 |
| Host 即将提交 ioctl | `ioctl_begin_ns` | 在写 trace marker 之前采样，包含 marker 的提交开销 |
| 进入驱动处理 | `driver_entry_ns=[lo,hi]` | `rknpu_submit_ioctl` 函数入口；不声称此时已成功排队 |
| 驱动完成入队 | `enqueue_ns=[lo,hi]` | 根据已审计的指令顺序包围链表插入操作；无精确公开时间戳 |
| 驱动开始服务 | `start_ns=[lo,hi]` | `hw_commit_time=ktime_get()`；随后配置寄存器并启动任务 |
| 驱动记录完成 | `completion_ns=[lo,hi]` | 对应核 IRQ 中 `now=ktime_get()`；不是硬件发出 IRQ 的精确时刻 |
| Host 首次观察到完成 | `host_observed_ns = ioctl_end_ns` | 阻塞 ioctl 返回后采样，尚未完成输出读同步 |
| API 完成读同步、清理 | `host_done_ns` | 最后一次 marker 写入前采样；数值对照在 API 返回之后 |

定义：

```text
response = host_observed - host_begin
service  = submit.hw_elapse_time
queue    ∈ [max(0, start_lo - enqueue_hi), max(0, start_hi - enqueue_lo)]
other    ∈ [response - service - queue_hi, response - service - queue_lo]
api_response = host_done - host_begin
```

因此得到 `T_response = T_queue + T_service + T_other`，并保留相关区间端点。
若把问题中的 `T_execution` 理解为设备服务包络，可用这里的 `T_service`；若要求纯硬件执行时间，当前尚不能等同。
服务包络包括寄存器配置、执行和完成中断处理到取时之间的延迟。`other` 包含命令准备、进入驱动、入队前工作、Host 唤醒及未进一步分类的开销；提交返回后的输出读同步和资源清理由 `api_response-response` 单独体现。

## 如何分离队列和服务

当前内核没有 kprobe / BTF 接口，但支持全局 function graph tracer。
[schedule_capture.py](custom-ops/tools/python/schedule_capture.py) 采集提交、调度、取队列、核 IRQ 和相应 `ktime_get` 调用。
[schedule_analyze.py](custom-ops/tools/python/schedule_analyze.py) 按请求 ID/TID、核内 FIFO 和 IRQ 核号关联记录。
只匹配预期调用层级的时间函数，避免把调度切换或无关调用的取时误认成 job 开始。

2026-10-10补充：过滤掉的调用/中断仍可能在这一显示层级留下多个 `ktime_get`。v2解析器将严格递增候选合成包围区间，再用驱动service时差收紧，不挑选一个调用当成精确时间点；重复/同量化时间的候选与FIFO歧义仍拒绝。原正式轮复算结果保持一致，另见 [并发与干扰诊断](JOB-CONCURRENCY-INTERFERENCE.md)。

入队区间来自以下真实顺序：

```text
rknpu_job_schedule 入口
  IOMMU 与锁处理
  list_add_tail(job, core.todo_list)
  rknpu_get_task_number(job, core) 入口
  解锁
  rknpu_job_next
```

队列插入发生在 schedule 入口与 task-number 入口之间。
同核插入顺序使用持有队列锁时的 task-number 调用确定；开始由真正取出 FIFO job 后的 commit 取时确定。
繁忙核上的下一 job 可以由上一 job 的 IRQ 启动，解析器覆盖这一交接路径。

该顺序同时核对了 [0.9.8 驱动源码](rknpu-vendor/refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c) 和板端 boot Image 的实际指令。
编译后链表写入在 `0xffffffc009003534..353c`，随后的 `0xffffffc009003540` 调用 `rknpu_get_task_number`。
[schedule_kernel_audit.py](custom-ops/tools/python/schedule_kernel_audit.py) 固定核对这段代码的 SHA256，未审计的内核拒绝运行。

function graph 的 leaf 行使用入口事件时间和调用时长；格式依据 Linux v6.1 的
[trace_functions_graph.c](https://github.com/torvalds/linux/blob/v6.1/kernel/trace/trace_functions_graph.c) 的 `print_graph_entry` / `get_return_for_leaf`。
文本时间截断到微秒，因此显式加入 1 μs 量化包络和时长舍入裕量。开启 `sleep-time` 保留调度挂起时间。
驱动服务时长必须落入 `completion-start` 的可行区间，再用该精确时差收紧两端区间。

丢失/覆盖 trace、未标记提交、非预期 IRQ、请求缺失、跨 CPU 同时戳造成的顺序歧义、Mask 越界、数值错误或时差不一致都会拒绝产生 PASS。
同一 CPU/TID 的相同时戳保留程序顺序；不同上下文的歧义不会猜测。
采集需要安静且受控的 NPU 环境，不能把这套无 job 指针的 FIFO 关联方法直接用于任意外部负载。

## 真机结果

正式证据：[scheduling-20261009-213637](custom-ops/results/2026-10-09/scheduling-20261009-213637/summary.json)。
环境：`6.1.43-rockchip-rk3588-rknpu098`、RKNPU 0.9.8、NPU 1 GHz。
每个 task 为 64×64 FP16 HardSigmoid；worker 使用独立输入/输出，逐元素数学参考与 BO guard 验证通过。

| 场景 | 请求数 / task 数 | IRQ 实际核 | 排队上下界中位数 μs | 最大排队下界 μs | 最大区间宽度 μs | 服务时长中位数 μs |
|---|---:|---|---:|---:|---:|---:|
| 逐次显式切核 | 6 / 6 | 0, 1, 2 | 2.626–6.627 | 9.334 | 9.168 | 82.395 |
| 四 worker 同核 | 24 / 1536 | 0 | 436.000–438.988 | 1220.131 | 6.549 | 508.222 |
| 三 worker 分核 | 18 / 1152 | 0, 1, 2 | 1.779–5.530 | 3.717 | 4.876 | 603.158 |
| 候选集合 0/1 | 24 / 1536 | 0, 1 | 3.214–6.617 | 579.425 | 6.513 | 596.450 |
| 驱动 AUTO | 24 / 1536 | 0, 1, 2 | 1.782–5.365 | 599.173 | 5.584 | 737.906 |

这是各请求上下界分别取中位数，不能解释为某一真实请求的区间。
同核竞争与分核实验请求数不同，只用于验证排队与映射，没有据此声称吞吐加速比。
空闲核上的微秒级 enqueue→commit 包括派发开销，不必然意味着排在另一个 job 后面。

具体请求 `139344030004`：

```text
requested_mask = actual_mask = 1，irq_core = 0
queue    ∈ [1220.131, 1222.876] μs
service  = 455.869 μs
response = 4121.775 μs
other    ∈ [2443.030, 2445.775] μs
```

这个请求能区分超过 1.2 ms 的排队和约 456 μs 的服务，排队区间宽约 2.745 μs。
剩余时间确实很大：当前适配器每次分配、打包并同步命令/描述符，不能把这部分全部算进队列。

同一轮还运行了**切换为 `nop`、关闭 function graph hook** 的无 tracing 对照，数值全部通过。
同核服务中位数为 tracing 508.222 μs / 无 tracing 492.764 μs；分核为 603.158 / 716.907 μs。
对照受缓存、Host 排程和多核共享资源变化影响，不能从少量顺序样本推出固定 tracing 开销，也不作无扰动性能结论。

正式轮共 96 个 traced 请求、5766 个 task；另有同规模无 tracing 对照。
全部逐元素误差 ≤ 0.002，trace overrun / dropped events 为零。
`source.tgz` 保存实际板端运行时的源码包；证据目录另附收尾后的解析器、测试与 fixture，记录于 `review-source-manifest.json`。
Host 与板端 CTest 各 9 项通过；解析器另含 17 个正/反例，覆盖 FIFO IRQ 交接、相同时戳、错误映射、时间不一致与证据缺失。
采集时窗内没有 kernel journal 事件。原始 JSONL、trace、统计、配置恢复证明、内核反汇编、源码包和校验信息均保留在证据目录。

## 复跑

```bash
# 使用仓库已有 .env 的板端连接配置；无需 opencode。
bash research/custom-ops/run-schedule-board.sh

# 分析正式证据，严格核验完整轮次。
python3 research/custom-ops/schedule_analyze.py \
  research/custom-ops/results/2026-10-09/scheduling-20261009-213637

# 本机反例测试，无需设备。
python3 research/custom-ops/schedule_analyze_test.py
```

采集器取得 NPU 实验锁及全局 trace 锁，拒绝占用中的 tracer/events，备份已有 trace，自动恢复 tracer、时钟、buffer 大小、过滤器及选项。
它使用全局 tracer，因为当前内核的 trace instance 不支持 function graph。
这会短暂影响全局 tracing；已有 buffer 内容保存在 `previous-trace.txt`，不能重新注入 ring buffer。

若下一步必须得到**精确入队点**及**提交寄存器 kick 时刻**，应在匹配驱动加入带 job ID/core 的 enqueue、commit/kick、IRQ-done tracepoint，或换用支持相应 kprobe/fentry 的内核。
该内核改动尚未实施。当前完成的是已审计、可拒绝歧义的区间分离，以及可靠的逐 Submit 核控制。

后续项目结构整理已将 `.gitignore` 收窄为仅排除构建与大型证据；源码、工具与小型 fixture 可纳入 Git。原始证据与旧轮次源码快照保持不变。
