# Multi-Device Simulator 与单设备回放

`simulator.py` 是标准库实现的整数 ns 离散事件模型。场景描述 devices、tensor 大小与来源、job 输入/输出/依赖/workspace、每设备 duration 和 directed links。设备使用 FIFO，无抢占；task_count 是元数据，一次 Submit 内部的 task 链不迁移。张量生产者依赖自动纳入 DAG，循环、重复 tensor 引用、未知设备、非法 duration/bandwidth 和超出所有候选容量的 Job 会拒绝。

所有策略按 ready 顺序考虑 Job，设备 tie 用场景设备顺序；缺失实测/明确配置的 duration 不作为零成本。

| 基线 | 设备选择 |
|---|---|
| round_robin | 从循环游标选择第一个合法设备 |
| shortest_queue | 最小 `等待 FIFO 长度 + 在途 Job 数` |
| earliest_finish | 最小预计队列释放时间 + 迁移等待/传输 + service |
| locality_aware | 先最小非驻留输入字节，再最小预计完成时间 |

预计队列成本属于启发式，会使用当前驻留、链路占用及已排队 duration；不会预知未来的缓存变化或负载。每次模拟深拷贝输入场景，重复运行和调换策略执行顺序不累积状态。

## 内存与链路

每设备容量计入现有驻留、缺失输入副本、输出预留和 workspace。记录实际模拟峰值与最终占用。生产者的 home 数据在仍有消费者时不能删除；远端副本及 Host 可重取的数据可安全 LRU 驱逐。输入副本保留给后续消费者，最后消费完成后释放；传输在途消费者仍计入引用数。暂时缺内存的队列 head 等待后续完成事件，FIFO 保持；没有未来可推进事件则返回 deadlock，CLI 非零退出。

链路是独立串行资源，时间 `latency_ns + ceil(bytes*1e9/bandwidth_bytes_per_s)`，同一链路有争用。不同链路可重叠。设备在整个输入传输及计算期间被当前 Job 占用，此版本不模拟同一设备计算与传输重叠。

直接 device→device link 代表用户明确配置的有效路由，**不证明 P2P**。若没有直接 link，配置 source→host 与 host→target 可采用两段 Host staging；必须提供 `meta.host_copy_bandwidth_bytes_per_s`，另可设置 `meta.host_staging_capacity_bytes`。copy 使用独立串行 Host 资源，Host staging 按整个路由保守预留容量；报告 Host 峰值/最终量。张量在此模型中是同一组不变字节，真实布局转换的成本需要显式纳入场景或成本模型，不能自动解释为跨设备布局兼容。

事件按 `(time_ns, priority, insertion sequence)` 稳定排序，同一时刻 completion 先于后续开始；`transfer_reserve` 与实际 `transfer_begin` 分开。每 Job 满足：

`end-arrival = dependency_wait + dispatch_wait + queue_wait + transfer_wait + compute`。

`transfer_wait` 是计算前的实际墙钟跨度；`transfer_time` 是链路/复制资源服务时间之和，多个链路重叠时可以超过前者，二者不能再相加。设备 service/transfer 统计和 makespan 同样不能直接相加。输出内存峰值不是性能实测。

## 可复跑场景

```bash
python3 research/multidevice-scheduling/examples/run_example.py
python3 research/multidevice-scheduling/simulator.py \
  research/multidevice-scheduling/examples/two_device_scenario.json --policy all --events
python3 research/multidevice-scheduling/cost_model.py --example
```

`two_device_scenario.json` 的 RK3588/RK182X 服务、容量和链路均为说明性配置。`measured-rk3588.json` 使用数据集内的真实 service 与全 operation BO 分配量，但两台独立设备的拓扑/容量是配置；不将 RK3588 三核冒充独立 DDR 多卡。比较报告：[measured-baselines.json](../results/measured-baselines.json)。当前相同 Job 集的 RR/shortest_queue makespan 为 1055816ns，earliest_finish/locality 为 561157ns；这是成本查表的模拟结果，不是实测加速。

## 单设备回放验收

```bash
python3 research/multidevice-scheduling/replay.py \
  research/multidevice-scheduling/evidence/rk3588-trace-20261010/block-00-cell-001
python3 research/multidevice-scheduling/verify.py
```

`replay.py` 校验原始 `host_begin≤ioctl_begin≤ioctl_end≤host_done`、service 不超 ioctl、raw/interval 字段一致和 request_id 唯一。无 tracing 分解 prepare、service、ioctl residual、post；residual 是未知 queue + Host 开销，不称精确排队。

有 tracing 时使用原 audited entry/enqueue/start/completion hull，校验 service、queue、core 及存在可行 FIFO 时间线。回放先运行最早 release 的下界参考，再用原 start/completion hull 构建一条可行重建，在模拟器中检查事件顺序与 duration；不把重建的单个时间点当硬件精确时间。多个 actual core 使用不同执行队列，但同属一块共享 DDR 设备。

已保存五组 **120 次** tracing 和隔离轮 **200 次**无 tracing 的回放报告，见 `results/replay-*.json`。tracing 中 3 对 enqueue hull 重叠，唯一 enqueue 排序不作强断言，报告 `unique_enqueue_order=false` 与数量；其执行顺序可识别且 FIFO 可行。无 trace 的 FIFO 排序不可观测，明确标记。相同观测 service 的回放通过证明的是时间核算/区间一致性，尚未验证多卡反事实调度的预测精度。

87 项整体验收含独立手算时间线、双链路+Host copy 的 47ns DAG、驻留复用、峰值容量、重复运行、异常输入与原始数据损坏反例。
