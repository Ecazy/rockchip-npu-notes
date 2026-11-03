# TileLang → RKNPU

复用 TileLang/TVM，把受支持的静态计算编译到 RK3588，沿用 `custom-ops`/runtime 执行。
离线 compile / validate / explain 不打开设备。

## 当前状态

- 实验性后端，支持域由 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md) 声明；域外输入 fail closed。
- TileLang 入口仍走 adapter：`compile_tilelang` 与 `compile_artifact(carrier="rkir")`（PrimFunc 输入）
  先经 `tilelang_adapter.extract` 识别结构。
- `compile_rkir` **不经过 adapter**：它直接接收已归一的 RKIR `ir.Module`。
- 三者共用 `codegen.pipeline.compile_graph`。
- 正式 pass 试点（`frontend/rknpu_pass.py` + 可选的 TileLang backend 注册）是 **opt-in**：
  只支持 guarded-tail FP16 GEMM，尚未接入 CLI，也未切为默认。
- 判断"能做到什么"看支持域，判断"证据能证明什么"看验收方法；两者都不以历史 PASS 计数为依据。

## 使用与验收

| 要做什么 | 唯一入口 |
|---|---|
| 判断输入是否支持、为什么被拒 | [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md) |
| 理解架构、模块边界、pass 链契约与产物边界 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 判断什么算证据、能证明什么、不能声明什么 | [VALIDATION.md](VALIDATION.md) |
| 构建、恢复依赖、重放、上板 | [REPLAY.md](REPLAY.md) |
| 理解 BO、fd、并发与失败处理 | [RUNTIME-LIFECYCLE.md](RUNTIME-LIFECYCLE.md) |
| 查看 TileLang M/N 调度契约 | [TILELANG-SCHEDULE.md](TILELANG-SCHEDULE.md) |
| 使用冻结模型包与数值政策 | [models/README.md](models/README.md)、[数值政策](models/NUMERIC-ACCEPTANCE.md) |
| 找原始 campaign、日志与哈希清单 | [results/README.md](results/README.md) |
| 开发算子、runtime、GEMM、量化等实现技能 | [AGENTS.md](AGENTS.md) |

## 冻结输入材料

`GRAPH-FUSION-REPORT.md` 与 `LAYOUT-REUSE-REPORT.md` 是 SmolLM2 calibration/acceptance 文本窗口的
声明来源（见 [VALIDATION.md](VALIDATION.md) §4），按**冻结输入材料保留原文**；它们是历史实验报告，
不是当前技术说明，也不描述当前实现或当前结论。

## 维护规则

- 支持域、架构与契约、验收方法、命令、数值政策各自只有一处权威（上表）。
- 不新增逐轮报告、交接文档或状态摘要；阶段记录与历史方案通过 Git 历史查看。
- 证据按日期目录原地保存，不移动、不重算旧身份；文档不复制历史 PASS 表。
- 未实现的能力不写成现状；未被当前代码验证的结论不写成当前结论。
