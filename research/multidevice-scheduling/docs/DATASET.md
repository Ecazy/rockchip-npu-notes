# Job Cost Dataset 验收

数据入口：[job-costs.json](../data/job-costs.json)，[完整来源清单](../data/source-manifest.json)。2026-10-10 七轮真实 RK3588 实验导入得到 **608 个条件分组、945120 次有效 Submit**，其中 945000 次正式性能测量与 120 次 tracing 诊断分开分组。独立原始日志计数见 [source-census.json](../results/source-census.json)。数据构建没有运行 NPU，也没有补写测量值。

| 正式来源 | 有效 Submit | raw 文件 |
|---|---:|---:|
| cost | 129000 | 510 |
| concurrency | 261000 | 435 |
| resident | 261000 | 435 |
| saturation | 54000 | 90 |
| interference | 90000 | 150 |
| tail | 150000 | 30 |
| trace（诊断） | 120 | 5 |

每个 cell 的 condition 保留 campaign、FP16、M/K/N shape、programs、Submit 索引、实际 task_count、mode、Host I/O/device_ready 路径、workers、选核策略、requested/actual/allowed mask、burst/continuous、background、评分方式与 CPU affinity。不同环境、几何、并发、背景、实际核、tracing 状态不混合。一条 separate operation 的多个 Submit 是多个 Job，operation 的上传/下载窗口会重复关联到各 Submit；**不能将这些重复字段相加作为独占 Job 成本**。

`service` 保留每个整数 ns 值及频数的完整直方图，并提供 nearest-rank p50/p90/p95/p99。它继续使用原实验的驱动 commit→IRQ 包络，包括配置和部分 IRQ 延迟，不代表纯计算。Host prepare/ioctl/post/API response 和 operation upload/post 窗口分别给出分布摘要；operation post 不是已识别的纯下载时间。DDR 访问估计、真实 BO 容量和 Host 传输窗口各有独立字段。

`confidence` 是 **mean of block p50 的 95% block-bootstrap CI**：以独立随机化 block 为重采样单位，seed=20261010，1000 次 percentile bootstrap。它不是 pooled p99 的置信区间，也不独立抽样同一进程中的请求。少于两个独立 block 时标 `insufficient_blocks`、CI=null；因此单 block tracing 数据只有经验分布，不冒充具有跨轮测量置信度。尾部百分位不保证对未测负载泛化。

数据检查排除所有 warmup；失败 result、bad_count、guard 或缺失 operation 不进入有效分布。非法时间核算直接拒绝；正式来源必须 campaign PASS 且各 run exit_code=0。`census` 保留 warmup/invalid 的计数。所有有效来源 invalid=0。

硬件记录包含 RK3588、实验身份 Orange Pi 5 Plus、实际 uname、driver、NPU/DMC 回读和 affinity，以及原始环境路径/哈希。设备序列号未测，serial=null；1GHz NPU/0.9.8 驱动范围沿用已有报告。温度、频率采样和背景流量的原日志也在 source 清单里。驻留定义为共享 DDR 中设备可访问 BO；无法从这些日志判定 CBUF 常驻或精确 queue depth，workers 不等于 queue depth。

未经 tracing 的 `queue_ns=null`；tracing 的 audited queue hull 保留在原始 `queue-bounds.json`，不拿其替代非 tracing 请求的排队成本。跨卡 migration 没有实测，`migration_cost_ns=null`。RK182X 不在此测量数据域内。

## 重建、校验与查询

从仓库根运行：

```bash
python3 research/multidevice-scheduling/dataset.py build
python3 research/multidevice-scheduling/dataset.py validate
python3 research/multidevice-scheduling/dataset.py validate --full-sources
python3 research/multidevice-scheduling/verify.py --full-sources
```

`build` 从指定的七轮目录流式读取源文件，每个 run 独立解析；输出是确定性的派生 JSON，可重建覆盖，不修改原始证据。`--root` 支持恢复到另一个仓库根；所有 provenance 引用均为仓库相对路径。结构验证与来源哈希验证独立，输出明确 `full_sources_verified`。原始性能数据受历史 `.gitignore` 排除，完整重建需要恢复原轮次，缺失文件会失败。

五组原始 trace 与请求日志有模块内的完整[归档](../evidence/rk3588-trace-20261010/archive-manifest.json)，34 个原文件均逐字节保存；不依赖历史结果目录即可执行 tracing 回放与默认 `verify.py`。完整来源清单同时索引 6682 个原文件，含采集配置、环境、源码身份与背景数据。

```python
import sys
sys.path.insert(0, 'research/multidevice-scheduling')
from dataset import JobCostDataset
data = JobCostDataset()
cells = data.match(workload='lut', workers=1, residency='device_ready', tracing=False)
# match 是条件筛选；cost 要求完整 cell.condition，不能用部分条件冒充唯一预测。
cost = data.cost(cells[0]['condition'], percentile=99)
print(cost['duration_ns'], cost['confidence'])
```

这是实测条件查表；未知 shape、设备、路径或不完整条件会拒绝。原有 DeviceCostModel 的回归状态继续由原报告控制，本模块没有将实验性 MatMul 预测升级为已验证模型。
