# RK3588 算子开发技能导航

本项目的跨 agent 技能位于 `.agents/skills/`，使用标准 YAML frontmatter + `SKILL.md`，不依赖 Codex 专用配置。支持自动发现的 agent 可直接加载对应技能；其它工具或人工可从下面的链接读取。

| 任务 | 技能入口 |
|---|---|
| 算子设计、源码全景、TileLang 后端迁移 | [rknpu-operator-dev](.agents/skills/rknpu-operator-dev/SKILL.md) |
| GEM/DMA、IOMMU、同步、PC-chain、多核 | [rknpu-runtime](.agents/skills/rknpu-runtime/SKILL.md) |
| FP16/BF16/TF32、GEMM/BMM、布局和分块 | [rknpu-gemm](.agents/skills/rknpu-gemm/SKILL.md) |
| W8/W4、MSD、FlatQuant、DCOMP | [rknpu-quantization](.agents/skills/rknpu-quantization/SKILL.md) |
| GQA、MLA、KV cache、softmax | [rknpu-attention](.agents/skills/rknpu-attention/SKILL.md) |
| MoE、SwiGLU、LUT、Add+RMSNorm | [rknpu-fused-operators](.agents/skills/rknpu-fused-operators/SKILL.md) |
| autotune、cache、基准和正确性验证 | [rknpu-tuning-validation](.agents/skills/rknpu-tuning-validation/SKILL.md) |

技能来自 `rknpu2_matmul_open` 的固定源码版本，面向本项目后端开发。按当前任务读取相关技能及其参考文件；源码配方是候选依据，项目当前的支持域、数值政策、runtime 生命周期和能力验收仍以现有文档/实现为准。RK3588 经验不自动扩展到 RK1228。

示例：实现矩阵算子时读取 operator-dev + gemm；优化量化流水时读取 quantization + runtime + tuning-validation；优化 Transformer head 分组时读取 attention + runtime 及 operator-dev 的迁移参考。
