# 研究目录索引

| 目录/报告 | 职责 | 状态 |
|---|---|---|
| [custom-ops](custom-ops/README.md) | 当前可复用接口、组合算子、提交适配、工具、验收 | 当前开发入口 |
| [rknpu-vendor](rknpu-vendor/INTEGRATION.md) | 厂商逆向 Hook、旧发射器、实验台账、原始捕获与驱动参考 | 历史实验，独立保留 |
| [upstream-components](upstream-components/README.md) | 固定版本组件索引与独立源码 checkout | 外部依赖 |
| [multidevice-scheduling](multidevice-scheduling/README.md) | Job 成本数据、驻留与迁移、多设备模拟、硬件适配和 RKNN3 静态分析 | 新研究模块；RK182X 实机成本待测 |
| [FP16 K 分片](FP16-K-TILING.md) | K>256 的设备累计及融合对照 | 真机验证 |
| [Two-Task Chaining](TWO-TASK-CHAINING.md) | 描述符、寄存器组、IRQ 和 BO 生命周期 | 真机验证 |
| [静态图编译与 SwiGLU FFN](STATIC-GRAPH-COMPILER.md) | 有限 FP16 图的形状、tiling、生命周期与 BO 规划 | 三核 FFN 数值及边界传输验收 |
| [跨 tile RMSNorm 与 Softmax](CROSS-TILE-ROW-OPS.md) | 混合归约、mask、指数分段与前置 RMSNorm FFN | 真机；含明确的 CPU 阶段 |
| [RoPE、KV cache 与 self-attention](ATTENTION-KERNELS.md) | 固定容量 MHA/GQA、prefill/decode、主机 KV 事务 | 三核双模式验收；混合实现 |
| [Transformer decoder block](TRANSFORMER-DECODER-BLOCK.md) | 多输出 Q/K/V 图、残差/FFN、KV 回滚与单行 decode | 真机功能验收；API 收益按形状限定 |
| [同核 MatMul/DPU 重叠](SAME-CORE-OVERLAP.md) | 异质程序链、Submit 包络摊销、位图消融与证据边界 | 真机；已测链未观察到额外重叠收益 |
| [同核 GEMM/GEMV 并发](GEMM-GEMV-CONCURRENCY.md) | 链式包络、B8/B16 放大组、同核与跨核双进程对照 | 真机；链式只省 Submit 开销，跨核重叠为正对照 |
| [多核调度 Profiling](MULTICORE-SCHEDULING-PROFILING.md) | 核选择、队列区间、服务包络、观测边界 | 真机验证 |
| [单设备成本模型](DEVICE-COST-MODEL.md) | 服务/Host成本预测、分组留出误差及范围 | 真机；纯MatMul预测仍为实验性 |
| [Job并发与资源干扰](JOB-CONCURRENCY-INTERFERENCE.md) | 等工作量策略对照、排队区间、尾延迟及背景负载 | 真机；945000正式Submit，另120 tracing请求 |
| [成本模型实施记录](custom-ops/implementation-notes.md) | v2 组件、正式轮预算及验证边界 | 工具及本地证据已交付 |
| [INT8 MatMul 官方对照复核](rknpu2-matmul-validation/README.md) | 资源与主机布局匹配的单核完整调用对照 | 大 M prefill 有形状限定收益；decode 单核略慢，非整模型加速 |

新增的可复用能力放 `custom-ops`；探索性捕获按日期放在现有实验结果目录。
跨模块的支持范围更新 [能力总览](../docs/CAPABILITIES.md)，不要把“源码具备”直接登记为“本机已验证”。
上游知识仍按根目录 `chips/`、`encodings/`、`perf/`、`guide/` 分类；本轮不移动这些历史路径。
