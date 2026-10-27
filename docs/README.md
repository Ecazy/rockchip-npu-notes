# 文档导航

先按用途选入口，无需通读整个仓库。当前开发主要使用研究模块文档；硬件知识、早期实验和原始结果按来源与日期解释。

## 当前开发与使用

| 需求 | 入口 |
|---|---|
| 查看当前能力、证据等级与缺口 | [CAPABILITIES.md](CAPABILITIES.md) |
| 理解源码、依赖与扩展约定 | [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) |
| TileLang 后端、完整模型、后续优化 | [TileLang RKNPU 首页](../research/tilelang-rknpu/README.md) |
| custom-ops 公共 API、组合算子和工具 | [custom-ops 首页](../research/custom-ops/README.md) |
| 找研究模块和专题验收 | [research 索引](../research/README.md) |
| 恢复固定版本外部依赖 | [upstream-components](../research/upstream-components/README.md) |

## 硬件与性能参考

| 主题 | 入口及范围 |
|---|---|
| 芯片差异 | [chips](../chips/README.md)，区分 RK3588 与其他目标 |
| 硬件和数据类型 | [hardware-overview](../hardware-overview.md)、[datatypes](../datatypes.md) |
| MatMul 映射与布局 | [matmul-as-conv](../matmul-as-conv.md)、[tile-layouts](../encodings/tile-layouts.md) |
| 寄存器命令与 task | [regcmd-task-model](../encodings/regcmd-task-model.md) |
| 本地算子、图和调度实证 | [research 专题索引](../research/README.md#专题报告) |
| 上游性能笔记 | [benchmarks](../perf/benchmarks.md)、[TUNING](../TUNING.md) |
| 上游完整主题目录 | [旧综合索引](archive/LEGACY-README.md) |
| 主线 Rocket 使用指南 | [guide](../guide/README.md)，与本地厂商提交路径区分 |
| 第三方来源 | [SOURCES.md](../SOURCES.md) |

## 历史、数据与证据

| 材料 | 如何使用 |
|---|---|
| [RKNPU-STATUS.md](../RKNPU-STATUS.md) | 早期进展快照，当前能力从 CAPABILITIES 查起 |
| [MODEL-NOTES.md](../MODEL-NOTES.md) | 模型专题参考；TileLang 模型的验收从其模块首页查起 |
| [知识库审计](KNOWLEDGE-BASE-AUDIT-2026-10-14.md) | 2026-10-14 的审计记录 |
| [厂商实验台账](../research/rknpu-vendor/INTEGRATION.md) | 历史 Hook、捕获与逆向证据 |
| `perf/data/` | 各模型、会话及运行环境下的历史性能数据 |
| 各模块 `results/`、`provenance/` | 原始日志、输入输出、源码与依赖摘要；部分材料仅在本地存在 |

历史路径保留，以免破坏实验引用与哈希。`results/` 中的源码快照和 README 属于当轮证据，不作为新的开发入口。

## 维护约定

- 仓库首页只保留项目说明、构建入口和路由；各模块首页负责该模块的导航。
- 能力边界、接口契约、运行命令、验收结果各有所属文档，通过链接引用，避免复制完整段落。
- 阶段计划完成后注明“已完成需求”并链接验收；交接与审查记录标明日期，不持续追加为当前任务清单。
- 新实验保存独立结果与适用范围；不改写历史产物，不把候选方案或 mock 通过写成硬件能力。
- 同一数据只在一处维护，其他文档写摘要和链接。不同 campaign 的延迟不直接拼接为加速比。
