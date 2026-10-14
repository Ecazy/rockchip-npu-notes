# RK3588 多 Job 并发与资源干扰 v2

2026-10-10，FP16 / 厂商驱动0.9.8 / NPU 1 GHz。Job=一次阻塞Submit；依赖链整体在单核。mask 1/2/4 为单核，0仅为厂商AUTO；候选集合由用户态产出单个位后提交。

## 正式证据

| 实验 | runs | 正式Submit | 证据 |
|---|---:|---:|---|
| foreground | 435 | 261000 | [analysis](custom-ops/results/2026-10-10/device-cost-concurrency-20261010-170013/analysis.json) |
| resident | 435 | 261000 | [analysis](custom-ops/results/2026-10-10/device-cost-resident-20261010-171054/analysis.json) |
| saturation | 90 | 54000 | [analysis](custom-ops/results/2026-10-10/device-cost-saturation-20261010-172338/analysis.json) |
| interference | 150 | 90000 | [analysis](custom-ops/results/2026-10-10/device-cost-interference-20261010-170421/analysis.json) |
| tail | 30 | 150000 | [analysis](custom-ops/results/2026-10-10/device-cost-tail-20261010-173208/analysis.json) |
| trace | 5 | 120 | [analysis](custom-ops/results/2026-10-10/device-cost-trace-20261010-173033/analysis.json) |

每轮请求预算固定，worker各分得确定配额；同策略对照保持总工作量、形状、program清单、CPU分配和运行路径一致。每个worker独立fd/BO，输入有两种符号变体，输出预置NaN；全部请求逐元素评分，包含padding与BO guard。
foreground逐请求评分；resident预上传输入，每请求保留独立输出，全部worker完成测量后才逐请求验收，不是稀疏抽查。resident测量窗口仍包含现有适配器的每次命令/描述符分配、打包与同步，尚未做持久提交上下文优化。
continuous为一次启动后持续closed-loop；burst为可中止的逐轮启动门，失败不会让其他worker永久卡在barrier。
CPU按worker编号固定到0–7；1/2/3worker在A55，8worker还使用A76。与单worker比较包含Host并行和CPU类型影响，不能直接声称NPU三核加速；以下重点采用同worker数策略对照。

## 较长 Job 的同工作量对照

saturation使用256³ MatMul和四task组合算子。下面为同8worker、每轮600 operation、5个配对block：

| workload | 同核完成/s | 分核完成/s | 分核/同核 speedup [95%CI] | 分核/同核 service中位数比 |
|---|---:|---:|---:|---|
| matmul | 8963 | 17576 | 1.961 [1.943,1.979] | 1.329 |
| mm_bias_hsig | 3928 | 7562 | 1.926 [1.909,1.943] | 1.425 |

分核减少同核排队并提高完成吞吐，同时服务包络膨胀。这个现象说明并发会改变服务成本；未据此确定DDR/CBUF/IRQ中的具体瓶颈。短Job foreground中同核与分核吞吐接近，Host检查和提交开销占比很高，原数据保留。

## 排队与服务的区间证据

tracing诊断与无trace性能轮分开；不能组合成同一请求的分解。内核代码经过既有audit，trace loss和未知提交会拒绝PASS。
filtered-out调用/中断可能在同一显示层级贡献多个ktime_get；解析器保留候选的包围区间，再用驱动service时差收紧，不选择其中一个冒充精确时间点。重复/同量化时刻的候选、FIFO顺序歧义仍拒绝。

| workload / workers / policy | IRQ核 | queue上下界中位数μs | service中位数μs | 最大queue区间宽μs |
|---|---|---:|---:|---:|
| matmul / 8 / fixed | [0] | 535.043–538.002 | 88.957 | 21.000 |
| lut / 1 / fixed | [0] | 1.042–4.460 | 74.957 | 3.543 |
| matmul / 8 / split | [0, 1, 2] | 45.814–48.084 | 132.123 | 4.420 |
| lut / 4 / fixed | [0] | 1.543–4.856 | 76.416 | 4.585 |
| lut / 3 / split | [0, 1, 2] | 1.085–4.584 | 80.353 | 3.752 |

上下界分别取中位数，不是某一真实请求的区间。空闲核的微秒级enqueue→commit也包括派发时间，不必然是等待另一Job。现有观测仍没有精确enqueue/kick时间点或纯硬件执行边界。

## 尾延迟

tail每条件5000 operation/block × 5block=25000请求；以下为block p99的均值及按block bootstrap的95%CI。数值与guard全量通过。

| workload / workers / policy | API operation p99 μs [95%CI] | 完成/s |
|---|---:|---:|
| lut / 1 / fixed | 321.65 [312.78,326.32] | 5201 |
| lut / 4 / fixed | 454.83 [452.79,457.22] | 13391 |
| lut / 4 / split | 447.53 [439.60,455.47] | 13358 |
| matmul / 1 / fixed | 408.28 [396.45,420.30] | 3465 |
| matmul / 4 / fixed | 540.16 [535.37,547.17] | 9126 |
| matmul / 4 / split | 463.28 [458.84,468.59] | 11457 |

公平性用共同active窗口内的完成进展计算Jain，异构负载不给未经归一化的公平性分数；最终相等配额不能证明公平性。request_overlap_fraction只表示Host请求窗口重叠，不能解释成硬件执行重叠。

## CPU与内存背景干扰

victim在CPU0提交；背景固定到A76的CPU4–7。背景私有缓冲总量256MiB，超过已盘点3MiB LLC；读写软件流量、checksum、线程/缓冲参数及生命周期均保留。软件GB/s不是DDR硬件计数器。DMC频率固定并回读，线程数仅为背景强度参数。

| victim | 背景 | service slowdown [95%CI] |
|---|---|---:|
| lut | compute:1 | 0.995 [0.990,1.000] |
| lut | compute:2 | 0.992 [0.987,0.997] |
| lut | compute:4 | 1.000 [1.000,1.000] |
| lut | stream:1 | 1.197 [1.192,1.203] |
| lut | stream:2 | 1.421 [1.415,1.423] |
| lut | stream:4 | 1.679 [1.664,1.695] |
| matmul | compute:1 | 1.000 [1.000,1.000] |
| matmul | compute:2 | 1.011 [1.000,1.032] |
| matmul | compute:4 | 1.011 [1.000,1.032] |
| matmul | stream:1 | 1.263 [1.247,1.279] |
| matmul | stream:2 | 1.653 [1.621,1.684] |
| matmul | stream:4 | 2.100 [2.063,2.137] |

不同核的同质/异质pair覆盖核集合0/1、0/2、1/2，另有三核同质Job；异质条件交换victim/competitor类型。service包含驱动配置与IRQ延迟，部分worker的CPU类型不同，具体资源归因保留unresolved。
[干扰校准表](custom-ops/results/2026-10-10/device-cost-interference-20261010-170421/interference-model.json) 用前4个block预测第5个block；service乘数留出误差p50=0.56%、p90=2.90%。
另提供固定常数/线性基线的组合留出检查；这是事后探索性检验，未提升为普适预测能力。校准仅覆盖64×64所测条件，不支持未知形状、背景或由线程数推导反事实DDR带宽。

## 实现、复跑与限制

```bash
bash research/custom-ops/scripts/run-device-cost-board.sh resident 5 600 20
bash research/custom-ops/scripts/run-device-cost-board.sh saturation 5 600 20
bash research/custom-ops/scripts/run-device-cost-board.sh interference 5 600 20
bash research/custom-ops/scripts/run-device-cost-board.sh tail 5 5000 20
bash research/custom-ops/scripts/run-device-cost-board.sh trace 1 24 2
```

入口：tools/c/cost_probe.c、interference_load.c、tools/python/device_cost_campaign.py、device_cost_schema.py、device_cost_analyze.py。只使用C11/POSIX和Python标准库。
外层持有一次NPU锁及trace锁；保存/恢复NPU min_freq、DMC/CPU governor和tracer设置，回读核验。全局旧trace内容保存但不能重新注入ring buffer；SIGKILL无法保证自动恢复，下一次须预检。
失败保留构建日志、部分JSONL和原始trace；不自动重试。两个早期构建失败和首次trace解析拒绝已保留，不计入正式通过样本。修复后使用新源码包和新目录验收。
无抢占、链内迁移或精确queue-depth观测；worker数不等于队列深度。没有采用M/M/1泊松假设或预测调度器。持久Submit上下文、精确job-ID tracepoint、PPU/大K成本校准为后续扩展。
