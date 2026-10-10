# 多设备驻留、迁移与调度研究

完整证据与验收状态见 [验收记录](docs/ACCEPTANCE.md)。

本模块在 2026-10-10 的 RK3588 成本、驻留、并发和干扰实验上继续构建。调度单位是一次 Submit：依赖 task 链整体留在同一个设备，不在链内迁移。RK3588 片内三核与 RK182X 多卡是两种不同拓扑，设备 ID、内存所有权和命令 ABI 独立。

| 验收成果 | 实现与证据入口 | 边界 |
|---|---|---|
| Job Cost Dataset | [DATASET.md](docs/DATASET.md)、`dataset.py`、`data/` | 真实 RK3588 数据；原始日志和 trace 保留来源、哈希与测量置信度 |
| Multi-Device Simulator | [SIMULATOR.md](docs/SIMULATOR.md)、`simulator.py`、`replay.py`、`examples/`、`results/` | 整数 ns 事件、设备队列、内存约束、依赖、链路及四种基线；单设备回放单独验收 |
| Hardware Backend Interface | [BACKENDS.md](docs/BACKENDS.md)、`backends/` | 统一 submit/wait/transfer/query_memory；RK3588 复用既有执行，RK182X 独立适配 |
| 驻留与迁移成本模型 | [COST-MODEL.md](docs/COST-MODEL.md)、`cost_model.py` | 实测服务查表 + 显式配置迁移/双端内存门禁；未知成本不填零 |
| RKNN3 静态分析 | [RKNN3-STATIC-ANALYSIS.md](docs/RKNN3-STATIC-ANALYSIS.md)、[源码清单](static-analysis/source-manifest.json) | 固定官方版本的公开头文件、示例和文档，不代表 RK182X 实机验证 |

主机验收从仓库根目录运行：

```bash
python3 -m unittest discover -s research/multidevice-scheduling/tests -p 'test_*.py' -v
```

数据构建、完整来源校验、模拟与回放命令见各成果文档。Python 工具只依赖标准库；RK3588 原生桥构建依赖仓库已有固定版本源码。

五组 tracing 诊断的原始 `capture.trace`、完整 `raw.jsonl`、排队区间、采集配置及硬件身份已逐字节归档在 [evidence/rk3588-trace-20261010](evidence/rk3588-trace-20261010/archive-manifest.json)，归档清单记录原路径与 SHA256；可用于保留原始事件的离线验收。

## 成本与证据约定

`service` 继续采用原实验的驱动 commit→IRQ 包络，包含寄存器配置和部分 IRQ 延迟。Host API wall time、设备排队区间、上传下载和迁移分别记录；未观测的排队/跨卡成本保持未知。没有 tracing 的请求不能从 `wall-service` 得到精确 queue。

驻留在 RK3588 上是设备可访问 DDR BO，不表示 CBUF 常驻。内存容量使用实际 BO 分配或 SDK 对齐容量，包含当前任务的 workspace 和输出；访问量估算不等于容量。模型、张量及状态的迁移须绑定来源/目标设备、布局和版本，不能共享物理地址或 Regcmd。

现有七轮证据含 945000 次正式 Submit，另有 120 次 tracing 诊断请求；warmup、失败记录和 tracing 轮不混入普通性能分布。原始证据位于 `research/custom-ops/results/2026-10-10/`，该历史目录受既有 `.gitignore` 排除。此模块引用原文件并提交派生数据及来源清单；完整重算/校验需要保留或恢复原始证据，不能用派生分位数代替 raw trace。

模拟器链路带宽、延迟及 RK182X 示例服务时间都是明确配置的场景参数。公开 RKNN3 多卡示例经 Host 回调复制与队列交接，不足以证明 P2P、底层抢占或与 RK3588 命令兼容。RK182X 的实测服务分布和链路校准须在对应硬件上独立补充。
