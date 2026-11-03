# 当前复核状态：§81修复通过（主报告§83）

2026-11-03独立复核关闭§81 P1：四份原探针在两个入口均拒绝；新增效果检查7项和
注册/产物6项独立通过。当前实现SHA-256为
`65a502dc01bd6f2d2f308ed34e558402efd5159d1d35fab0ad143d396f899c84`，测试为
`9ab554d43f13f65028993246bdbc496657d4c1f95fc849de7c81405039c252b4`；backend未改。
完整证据：`results/codex-pass-81-recheck/README.md`。执行者全量909 passed/2 skipped，
本次未重复全量，无新设备实验。只验收现有guarded-tail FP16 GEMM试点与§81修复，
不自动扩大算子或切换默认；本次未修改实现、正式测试或Goal工具状态。
旧STATE逐字保存在该证据目录的STATE-before-review.md；下列内容为历史状态，
历史哈希不代表上述当前实现。

# 历史调度状态：§80曾判本阶段最终验收PASS

实际源码、工具状态与可重跑证据优先；BASELINE.json保留不可改写的接手快照。
旧STATE保存在results/handoff-final-lead-audit-r4/STATE-before-final.md；历史过程
见唯一主报告§74–§80，本文件只记录最终现状。

## 目标与停止状态

- Goal：01a19cd9-cf2e-7562-9810-eb57f8724acd。当前阶段已独立验收，目标完成条件全部满足；最终工具完成记录由本会话保存。
- guarded-tail FP16 GEMM当前阶段PASS。已确认阻断项全部关闭，无剩余实施项。
- 主线程/root只规划、调度与独立审计，未改实现/正式测试/配置/构建脚本。
- 执行者/root/executor_w01已最终停止全部写入，无后台任务。原spawn显式请求gpt-6-luna/low/fork_turns=none；工具未提供实际路由证明，不额外声称已验证。
- 无运行中的测试/构建/板测或其它writer；停止于本阶段，不自动启动下一目标。
- HEAD da4eeabd1c022a6e077bddfef4c36f2c46e6b11f；继承dirty/untracked保留，未reset/clean/整文件回退或混合暂存。

## 最终冻结身份

| 文件 | SHA-256 |
|---|---|
| rknpu_pass.py | e072e45e47d9913fe4913d7c4029a6b872bffb1478bb80aaab92ff24a6e927dd |
| tilelang_backend.py | 471c4e911ac0a25b8ce5f3f15b05cdece2ff9f17bd3d373384ef90fc45ca1463 |
| test_rknpu_pass_pipeline.py | 5afd2796902035e91d5bf8c902bbfea44b222cfebbb22f48372a01da1e31b35d |
| test_tilelang_backend_registration.py | f564bfb33a1ec46e88beb9afe7b88011ad92ec4090da1b466a9098c371f9807b |

r4证据根results/handoff-final-lead-audit-r4。185文件before/after完全相同，
最后实时复核一致；r3→r4只改pipeline测试断言，未改实现。身份索引含文档/日志。

## 独立结论与证据

- 最终审计INTENT-FIDELITY-AUDIT.md §80。§74谓词、§75 store mask、§76调用/offset/flag、§78声明覆盖/module边界/hint持久化全部关闭。
- r3独立control18+fields33符合预期，七组历史脚本39状态零偏差，module public/engine十结果符合预期。
- 两种tail/decode形状的direct/simplify/legacy/engine/roundtrip执行描述、常量与contract相等；四次当前计划mock输出576零不符；artifact清空全局并删局部源引用后可读。
- 最终主线程完整pytest tests：902 passed、2 skipped、355 subtests passed，exit0，1088.58秒（driver1089.0038秒），session86132已结束。含两相关正式文件。
- skip：transformers未安装的AutoTokenizer入口；未启用RKNPU_SMOLLM_SLOW_TESTS=1的包级mock回放。没有将skip当通过。
- r3首次相关回归10failed/111passed/66subtests日志保留。原89方法全部保留；只按完整执行等同性和准确拒绝诊断迁移断言，未放宽数值。
- 同一C/库独立CTest33/33、5.64秒，既有build未重建；两host库与BASELINE相同。
- 当前256/576/3072与历史真实设备计划完整C执行字段相同，仅36条origin诊断不同；常量/contract相同，JSON记录差异另列。本轮无真机实验，mock不是真机。旧manifest两中间TIR既有漂移未改写，关键计划/脚本/设备记录身份有效。
- refactor-specific.md与W02 JSON已回填最终结果并核对日志/命令SHA，git diff --check通过。

## 限制与后续

动态/选择性block predicate、非平凡mask、block iter/init/match、非零/未知offset
及其它未支持结构按契约拒绝。不实现通用GPU语义或其它算子。旧adapter、默认、
数值政策和算子范围不变。无新构建、性能搜索或权重复用实验；当前阶段完成后停止。
