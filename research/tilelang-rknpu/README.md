# TileLang → RKNPU

复用 TileLang/TVM，把受支持的静态计算编译到 RK3588，沿用 `custom-ops`/runtime 执行。
离线 compile / validate / explain 不打开设备。

## 当前状态

- 实验性后端，支持域由 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md) 声明；域外输入 fail closed。
- `compile_tilelang` 与 `compile_artifact(carrier="rkir")` 先做结构性输入分派：closed
  full-logical FP16 GEMM 直接进入目标 TIR → 任务消费链；其他尚未迁移的 PrimFunc 形式仍走
  `tilelang_adapter.extract`。
- `compile_rkir` **不经过 adapter**：它直接接收已归一的 RKIR `ir.Module`。
- `compile_rkir` 与剩余 adapter 输入共用 `codegen.pipeline.compile_graph`；已迁移的
  full-logical/SiLU 输入使用 `compile_from_tasks` 任务消费链。
- closed full-logical QKV 与现有 SiLU accumulator 更新已接入正式
  `compile_tilelang`/`compile_artifact`：`schedule_full_logical_source` →
  `lower_scheduled_full_logical` → `pass_instantiate` → `compile_from_tasks`。
  这是对真实源 fill/copy/GEMM/store/output-copy 节点的改写；`_LogicalSource`、
  `_prove_full_logical` 与 `_target_tir` 已删除。
- guarded-tail 的 guarded single-trip、三 buffer final-only、显式 round-trip 及
  `baseline / scheduled_baseline / weight_stationary / weight_stationary_reuse /
  grouped_linear / merged` 已接入正式 `compile_tilelang`/`compile_artifact(rkir)`，
  复用 localise/instantiate 与顺序任务消费链。无 guard 的 final-only/round-trip
  K-loop 也通过源操作局部 lowering 进入同一任务链；adapter 的 `_extract_ragged_tail`
  与旧 K-loop 分支均已删除。
- 两条真实 SwiGLU FFN、`silu_proj` 投影链、`branch_outputs` 双输出以及
  scalar/channel broadcast 输入已接入同一任务消费链：源端按实际
  fill/copy/cast/sigmoid/GEMM/store 和局部表达式逐项改写；整区域 cast/copy 才允许
  融合。显式 sigmoid 与后续乘法保持独立 op 归因；每个实际 output copy 独立生成
  `rk_store` 并驱动 download/publish。rank-0/rank-1 broadcast 只从实际 `B[()]` /
  `B[j]` 读取和所属循环承接，carrier 保留原始 rank，任务记录广播映射。adapter 的
  `_extract_ffn_branch`、`_extract_static_dataflow`、`_extract_branch_outputs` 与
  `_extract_broadcast_tail` 已删除。多 launch 输入由 `staged_launch` 按实际
  `T.Kernel` 作用域分割，保留 stage 注解、跨 launch 全局值和提交边界。两 launch
  `make_two_stage` 已由真实 per-stage 任务消费生成计划：声明调度先落实为应用后 tile，
  中间 Z 仅登记设备 value/view/producer，stage1 直接继承其 region/producer；正式入口
  不再回退 adapter。
- 注册 TileLang backend 仍为显式 opt-in；CLI 与模型构建调用尚未整体切换。
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
