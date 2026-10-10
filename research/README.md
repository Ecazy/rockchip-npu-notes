# 研究目录索引

| 目录/报告 | 职责 | 状态 |
|---|---|---|
| [custom-ops](custom-ops/README.md) | 当前可复用接口、组合算子、提交适配、工具、验收 | 当前开发入口 |
| [rknpu-vendor](rknpu-vendor/INTEGRATION.md) | 厂商逆向 Hook、旧发射器、实验台账、原始捕获与驱动参考 | 历史实验，独立保留 |
| [upstream-components](upstream-components/README.md) | 固定版本组件索引与独立源码 checkout | 外部依赖 |
| [multidevice-scheduling](multidevice-scheduling/README.md) | Job 成本数据、驻留与迁移、多设备模拟、硬件适配和 RKNN3 静态分析 | 新研究模块；RK182X 实机成本待测 |
| [FP16 K 分片](FP16-K-TILING.md) | K>256 的设备累计及融合对照 | 真机验证 |
| [Two-Task Chaining](TWO-TASK-CHAINING.md) | 描述符、寄存器组、IRQ 和 BO 生命周期 | 真机验证 |
| [多核调度 Profiling](MULTICORE-SCHEDULING-PROFILING.md) | 核选择、队列区间、服务包络、观测边界 | 真机验证 |

新增的可复用能力放 `custom-ops`；探索性捕获按日期放在现有实验结果目录。
跨模块的支持范围更新 [能力总览](../docs/CAPABILITIES.md)，不要把“源码具备”直接登记为“本机已验证”。
上游知识仍按根目录 `chips/`、`encodings/`、`perf/`、`guide/` 分类；本轮不移动这些历史路径。
