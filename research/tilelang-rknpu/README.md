# TileLang → RKNPU

复用 TileLang/TVM，将受支持的静态计算编译到 RK3588，沿用 custom-ops/runtime 执行。离线 compile/validate/explain 不打开设备。

## 当前状态

- 后端 MVP 已在声明支持域内完成；仍是实验性后端，不承诺未知设备故障后的恢复。
- MNIST MLP 与显式 hybrid Tiny Transformer 完整模型里程碑已关闭，证据统一见 [ACCEPTANCE.md](ACCEPTANCE.md)。
- 默认模型优化、最终 baseline/current 复验及厂商 RKNN 对照统一见 [OPTIMIZATION-SUMMARY.md](OPTIMIZATION-SUMMARY.md)。不同数值政策及不同 campaign 的性能分开解释。
- 后端重构是新的待实施任务，不代表当前代码已迁移。原始设计与重构方案各保留一对 overview/specific，按下面的适用范围阅读。

## 开发规范

| 用途 | 宏观设计 | 实施细则 |
|---|---|---|
| 原始后端设计及既有契约 | [overview.md](overview.md) | [specific.md](specific.md) |
| 当前重构方案与执行要求 | [refactor-overview.md](refactor-overview.md) | [refactor-specific.md](refactor-specific.md) |

重构的接口、工作包、验证与进度只写入 refactor-specific；不再为该任务另建状态、实验报告或交接文档。原始设计不作为重构必须照搬的表示形式。

## 使用与验收

| 要做什么 | 唯一首选入口 |
|---|---|
| 判断输入是否支持 | [支持域](SUPPORT-DOMAIN.md) |
| 理解 BO、fd、并发与失败处理 | [运行时契约](RUNTIME-LIFECYCLE.md) |
| 构建、恢复依赖、重放、上板 | [REPLAY.md](REPLAY.md) |
| 查看 MVP 与完整模型验收 | [ACCEPTANCE.md](ACCEPTANCE.md) |
| 查看当前优化结论与外部对照 | [OPTIMIZATION-SUMMARY.md](OPTIMIZATION-SUMMARY.md) |
| 使用模型包与数值政策 | [models/README.md](models/README.md)、[数值验收](models/NUMERIC-ACCEPTANCE.md) |
| 找原始 campaign 和日志 | [results/README.md](results/README.md) |
| 研究独立 RK1228 目标 | [RK1228-TARGET.md](RK1228-TARGET.md) |

## 历史材料

- [开发与需求历史](archive/development-history.md)：合并 MVP 实施、审查与交接、P0 源码锁、已完成模型需求；各节保留来源文件名及稳定锚点。
- [优化方案与逐轮历史](archive/optimization-history.md)：合并原优化方案与第 1–7 轮结果；不是当前任务清单，最终结论从优化总报告查起。

旧文档原路径不再保留占位文件。历史源码包、provenance 和 campaign 中的原文件名按当时快照解释；本次不修改其内容或哈希。被删除的旧长首页快照与 Git 中原版本重复，可从版本历史追溯。

## 维护规则

首页只维护导航与简短状态；支持域、运行契约、重放、验收和优化总报告各有一处权威说明。阶段记录归并到对应历史材料，不继续新增逐轮独立报告或重复交接。原始证据按日期保留原位，不移动、不重算旧身份，也不以历史通过代替当前验证。
