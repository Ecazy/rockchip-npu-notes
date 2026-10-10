# 2026-10-10 验收记录

模块入口：[README](../README.md)。新增源码和派生证据全部位于 `research/multidevice-scheduling/`，研究索引已增加入口。现有 RK3588 `custom-ops` 实现不修改；任务开始前的代码/文档改动保留。委派中误还原的能力文档按先前原始补丁恢复，最终与保护快照一致。

| 成果 | 已交付 | 证据 |
|---|---|---|
| Job Cost Dataset | 608 条件、945000 正式 Submit +120 tracing；原始服务时间直方图、shape/task/驻留/并发、硬件域、block CI | [数据](../data/job-costs.json)、[来源](../data/source-manifest.json)、[独立计数](../results/source-census.json) |
| 内存驻留/迁移模型 | CacheIdentity、真实 BO 容量门禁、下载/复制/布局/上传/setup、双端预算、成本查表与 break-even | [模型](COST-MODEL.md)、[实测成本驱动的基线模拟](../results/measured-baselines.json) |
| Multi-Device Simulator | FIFO 队列、内存/安全驻留、DAG、配置链路/争用、Host staging、四基线 | [语义与命令](SIMULATOR.md)、独立 47ns/驻留/等待/失败时间线测试 |
| 单设备回放 | 120 traced +200 untraced、服务/Host 核算、原区间约束和可行事件顺序 | `results/replay-*.json`，34 个原始诊断证据文件 |
| Hardware Backend Interface | submit/wait/transfer/query_memory、所有权/超时/生命周期、RK3588 原生桥、RK182X 独立契约层 | [接口](BACKENDS.md)、[真实 ABI 2 smoke](../results/backend-20261010-191327/smoke.json) |
| RKNN3 静态分析 | 两个固定官方 commit、10 份源码快照、x86_64 Runtime 动态表与完整符号 | [分析](RKNN3-STATIC-ANALYSIS.md)、`static-analysis/*manifest.json` |

验证结果：

- [87 项主机测试](../results/host-tests.json)：0 failure/error/skip，包含数据损坏、模拟时间线、迁移双端门禁、错误 ABI/句柄、超时后继续 wait、缓存重用、内存暂时不足等待等反例。
- [11 项既有 C 框架主机检查](../results/legacy-host-tests.log)：全部通过；新 bridge 与原静态目标构建/链接成功。
- [完整来源与归档核验](../results/verification.json)：6682 个历史来源 SHA256 校验、64 个模块静态/归档/硬件证据文件核验，120 次 trace 回放通过。
- [确定性重建](../results/data-determinism.json)：从全部原始轮次独立重建，两个派生数据 JSON 逐字节一致。
- 两次新适配器真机 smoke 在各自新工作目录由实际源码构建，ABI 1/2 分开保留；最新 ABI 2 的 1024 个 FP16 HardSigmoid 元素及整个 slack/guard 通过。持有两个 BO 共 69760 字节；未发布新的吞吐校准。

复查命令：

```bash
python3 -m unittest discover -s research/multidevice-scheduling/tests -p 'test_*.py' -v
python3 research/multidevice-scheduling/verify.py --full-sources
python3 research/multidevice-scheduling/cost_model.py --example
```

未提升的结论：RK182X 尚未绑定真实 Runtime，也未测服务/链路/迁移。四基线结果属于配置拓扑下的模拟，不是多卡硬件性能。3 对 tracing enqueue hull 重叠，报告保留非唯一排序；无 trace 的 ioctl residual 仍是未拆分的 queue+Host。单设备回放验证一致性，不证明反事实预测精度。
