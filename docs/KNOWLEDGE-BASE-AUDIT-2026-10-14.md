# knowledge-base 与 RKNPU 工作区对照审查

审查日期：2026-10-14。知识库路径：`/home/maoyu/Documents/knowledge-base`，全程只读。

结论：RKNPU 专题的主干结构可保留，但累加、布局转换、模块调度和性能建议存在过度概括；当前工作区新增的可调用接口与实机成果尚未收录。以下区分正文错误、条件缺失和新增成果遗漏，不能将本地分支的新增能力反推为其引用的上游历史版本已经提供。

## 1. 范围与证据

- 对知识库全部 78 篇 Markdown 做 RKNPU 相关内容检索和静态相对链接筛查；精读 RKNPU 的五篇专题（737 行）及相关入口页。
- 对照本工作区 `chips/`、`encodings/`、`perf/`、`research/`、公共头文件、实现、现存结果及 Git 变更。没有对无关论文专题作科学内容的全面审稿。
- 知识库五篇专题主要引用上游 `81b0d17da0b8a472c1a1e6f7d2ddba6b34d10111`，访问日期为 2026-10-09；本工作区有合并后的上游更新及 10-10、10-13、10-14 本地成果。历史快照与当前分支必须分别说明。
- 本次重新构建 `research/custom-ops` 并运行 Host CTest：18/18 通过。主机测试不是新增真机验收。
- 本次对两轮 overlap 原始记录重跑核验器，并重算已有 attention / decoder 六个核与模式组合的独立数学参考、文件哈希及缓存事务记录。没有向板子提交新的硬件任务。
- 原始大型 `custom-ops/results/` 沿用 Git 忽略规则，只保留本地；报告、源码和小型验证资料可提交。已保存成果提交：`96a65a7`（算子/图/decoder/成本与并发）、`8b859e7`（INT8 官方对照与紧凑原始资料）。核心成果和少量结果入口的哈希索引见 [审查快照](KNOWLEDGE-BASE-AUDIT-2026-10-14.json)。

下文 `KB/` 表示知识库的 `content/hardware/on-device-npu/rknpu/`。行号按审查时文件计，不是未来更新后的行号。

## 2. 优先修正的错误与条件缺失

### F01：将 CORE 无跨层累加推成 K 分块必须回读；又把 DPU 累加称为片上累加

**优先级：高；类型：正文矛盾、存储边界错误。**

- 位置：`KB/block-internals.md:75`、`KB/architecture.md:124`、`:129`、`:138`。
- 原文：“没有跨层累加，这是 K 分块必须回读的根因”；另一处又称“fp16 是唯一有片上 K 累加的”。
- CORE/CACC 不能跨 task 保留部分和，但 FP16 DPU EW 可以从设备可访问 DDR BO 读取上次部分和，与新部分和相加后写入另一个 BO，CPU 最后才下载输出。**不回读主机不等于不经过 DDR，更不等于 CBUF/CACC 常驻。**
- 证据：[上游 K 累加说明](../encodings/k-accumulation.md)、[本地 FP16 K 分片](../research/kernels/FP16-K-TILING.md)、[Linear 实现](../research/custom-ops/src/linear_hardsigmoid.c)。
- 建议表述：“CACC 不支持跨 task 累加；本工作区 FP16 路径以 DPU EW 和双 DDR BO 完成设备侧部分和累计，每片会舍入到 FP16。外部 `rknpu2_matmul_open` 另有可选 INT32 DPU split-K 累加接口，见 §7；该接口尚未由本次归档性能实验验收，不能把本框架未实现写成硬件不支持。”

### F02：将无 RUBIK 推成布局变换只能在主机上完成

**优先级：高；类型：过强的否定结论。**

- 位置：`KB/architecture.md:28`、`:31`、`:158`，`KB/block-internals.md:10`，`KB/programming-model.md:150`。
- 原文：“主机侧 de-tile 不可消除”“布局变换留在主机侧”“没有寄存器能转换布局”。
- 同一专题已经承认 MatMul 可以用 stride 直接保持 row-major，因而“不可消除”在文内也不成立。本地还提供真机验证的固定 `8×8×8` reformat：`out[c,a,b]=in[a,b,c]`。
- 证据：[row-major 寄存器配方](../matmul-as-conv.md)、[当前接口与重排契约](../research/custom-ops/README.md)、[能力总览](CAPABILITIES.md)。
- 建议区分：无独立 RUBIK/通用 transpose 引擎；权重通常仍须预打包；部分 MatMul 布局可由 stride/notch 表达；已测固定重排可在设备执行。不能推广成任意 shape/perm，固定重排也不等于完整 Runtime 外部布局转换。

### F03：独立块、乒乓寄存器组与模块并发之间缺少关键限制

**优先级：高；类型：能力边界遗漏与文内歧义。**

- 位置：`KB/architecture.md:47`，对照 `KB/programming-model.md:89`。
- 前者称有依赖的后继必须由软件等生产者完成后再调度，后者又允许一次链式 Submit 内直接交接依赖输出。需要说明“软件安排 task 顺序/完成机制”与“每段必须独立回到主机等待”的区别。
- 更重要的是，DPU-only LUT 可以独立发射，不代表同核 CNA/CORE 与 DPU 有两条用户可独立调度的计算队列。完整混合 Softmax 还使用 Host、identity MatMul、归约 MatMul 和 DPU。
- 证据：[Two-Task 依赖链](../research/scheduling/TWO-TASK-CHAINING.md)、[MatMul/LUT 对照](../research/scheduling/SAME-CORE-OVERLAP.md)、[GEMM/GEMV 对照](../research/scheduling/GEMM-GEMV-CONCURRENCY.md)。本次两轮记录核验 PASS。
- 已测链式收益主要为一次约 52–53 μs 的固定 Submit 服务包络；同核双 job 在当前厂商路径串行；跨核服务重叠约 1.31–1.32×。这些不能等同于纯 MAC 时间线，也不能证明所有潜在硬件编排均不可能。

### F04：M=1 pad 到 4 被写成通用 MatMul/组合算子的规则

**优先级：高；类型：条件缺失。**

- 位置：`KB/programming-model.md:154`。
- 原文：“M<4 会算错，必须 M%4=0（单行 matmul 要 pad 到 4 行）”。这是特定历史路径的规则；不能当成所有 FP16 DPU EW 和当前 API 的最小物理行数。
- 当前 `co_tensor_matrix` / `co_tensor_input` 采用 `round_up(max(M,12),4)`，逻辑 M=1 被支持，但物理执行至少 12 行。原 primitive 的 M/K/N≤256 与组合接口逻辑 1..4096 也须区分。
- 证据：[张量构造实现](../research/custom-ops/src/custom_ops.c)、[EW 的 M≥12 配方](../encodings/dpu-lut-activation.md)、[decoder 几何优化](../research/graphs/TRANSFORMER-DECODER-BLOCK.md)。
- 建议按驱动、发射器和算子列出 padding 契约；不要把接口的保守范围写成硅片普适限制。

### F05：460 GOP/s、三核 3×、默认 200 MHz 等数字被抽离测量条件

**优先级：中；类型：性能与部署条件缺失。**

- 位置：`KB/README.md:84`，`KB/architecture.md:147`，`KB/programming-model.md:159`、`:164`。
- 460 GOP/s 与量化近似等速来自特定 Rocket / 600 MHz / 权重驻留工作点，不能支撑所有形状及路径“量化不省时间”。时钟 200/600/900 MHz 的建议同样属于相应主线内核和调频实现。
- 本地厂商实验运行在 NPU 1 GHz；等工作量、同 8 worker 的长 Job 分核/同核吞吐约 1.96×，服务包络还会膨胀，不能套用理想 3×。
- 证据：[历史性能条件](../perf/not-mac-bound.md)、[时钟说明](../perf/clock.md)、[当前并发与干扰](../research/scheduling/JOB-CONCURRENCY-INTERFERENCE.md)、[INT8 官方对照](../research/rknpu2-matmul-validation/README.md)。
- 建议每个数字带上 shape、dtype、核数、频率、驱动、驻留方式和计时边界；“service”不写成纯硬件执行时间。

### F06：“单行 decode 不要上 NPU”是部署建议，不是硬件能力禁令

**优先级：中；类型：过度概括、遗漏目标函数。**

- 位置：`KB/programming-model.md:162`。
- 本地 decoder 已实现逻辑 M=1；max_query=129 时 warm 单层 API 的特定对照有 1.491–1.760× 收益，小形状仍可能变慢。这是相对固定多行计算基线，不是相对 CPU decode 的同等收益。
- 上游最新 Whisper 单步对照也表明：NPU 相对四线程 CPU 墙钟约慢 2.6–2.8×，但 CPU 时间约减半，说明时延和 CPU 腾挪是不同目标。82× 的 GEMV/GEMM 数字不能读成相对 CPU 的 slowdown。
- 证据：[decoder 报告](../research/graphs/TRANSFORMER-DECODER-BLOCK.md)、[上游 decode 说明与 Whisper 补充](../perf/decode-gemv.md)。
- 建议保留“该 Rocket 部署通常优先 CPU decode”，同时说明路径、比较基线与目标；不能由此否认 NPU decode 可实现。

### F07：软件栈对上游 notes 的定位没有与当前工作区分开

**优先级：中；类型：新增成果遗漏，不是历史引用错误。**

- 位置：`KB/software-stack.md:6`、`:47`、`:83`。
- “不是运行时库、不是编译器、只有文档”可以描述其引用的上游 notes 快照，却不能描述本地扩展分支。本地已有 `custom-ops` C 静态库、有限 FP16 DAG 编译器、调度适配器和 CLI。
- 证据：[工作区首页](../README.md)、[能力总览](CAPABILITIES.md)、[静态图编译](../research/graphs/STATIC-GRAPH-COMPILER.md)、[CMake 目标](../research/custom-ops/CMakeLists.txt)。
- 建议分别列“上游硬件参考”“本地编程与实验扩展”“独立 upstream checkout”。有限图编译器仍没有 ONNX/GGUF 导入、动态 shape 或任意 kernel 编译能力。

### F08：将 LE/LO 的负/正分支配方写成 LUT 的普遍定义

**优先级：中；类型：配方与硬件语义混淆。**

- 位置：`KB/block-internals.md:99`。
- “两张 513 项表（负/正分支）”只覆盖一类已测布表方式。上游还明确描述 LE 为 X/raw、LO 为 Y/density 的 hybrid，以及将整个正值域移至 LO 的 sqrt/rsqrt/exp 配方。
- 证据：[LUT hybrid 与分域说明](../encodings/dpu-lut-activation.md)。
- 建议保留 RK3588 每表 513 项的事实，将负/正称为示例配置。补上 `q=0` 输出故障、平坦区 mux 和有符号输出 x≈0 尖峰；SiLU/GELU 常用 gate+EW，而非任意输入下一次 LUT 就可靠。

### F09：“没有 matmul 原语”需要说明硬件与 SDK 两个层级

**优先级：中；类型：术语范围缺失。**

- 位置：`KB/README.md:6`、`:78`，`KB/programming-model.md:128`。
- CNA/CORE 用 1×1 Conv 数据通路实现矩阵乘的描述成立，但官方存在 `rknn_matmul_*` API，本地也有 `co_emit_mm` / Linear API。当前字面表述容易让读者误以为没有可调用的 MatMul 接口。
- 证据：[MatMul 数据通路](../matmul-as-conv.md)、[官方 MatMul 独立验证代码](../research/rknpu2-matmul-validation/validate_official_llm.cpp)。
- 建议：“寄存器级矩阵乘复用卷积通路；用户态库/SDK 可提供 MatMul API。”

### F10：将“无可用字节计数器”写成全系统的否定

**优先级：低；类型：观测范围缺失。**

- 位置：`KB/block-internals.md:144`。
- “DDMA/SDMA：字节计数（无可用计数器）”应限定为当前安全可用的 NPU per-op 计数接口。上游还研究了 DDR controller PMU 的全系统计数，它可用于差分实验，但不是每个 NPU task 的精确流量归因。
- 证据：[NPU/DDR 字节计数器调查](../perf/hw-byte-counters.md)、[本地成本模型](../research/scheduling/DEVICE-COST-MODEL.md)。本地 `estimated_bytes` 是访问量估计，不是已测 DDR bytes。

## 3. 应补录的成果及收录边界

这些是相对当前知识库的遗漏；不要求把底层架构笔记扩写成所有实验日志。

| 编号 | 建议知识点 | 本工作区入口 | 必须保留的边界 |
|---|---|---|---|
| O01 | FP16 多 tile Linear 与设备 K 累加 | [K 分片](../research/kernels/FP16-K-TILING.md)、[接口](../research/custom-ops/README.md) | 逻辑 M/N/K 1..4096；primitive≤256；DDR BO 双缓冲、每片 FP16 舍入；不是 CBUF 常驻 |
| O02 | 有限静态图编译与 SwiGLU FFN | [静态图](../research/graphs/STATIC-GRAPH-COMPILER.md) | 形状/tiling/生命周期/BO 预算；每节点每输出 tile Submit；无模型 importer、自动性能搜索或跨核图执行 |
| O03 | 跨 tile RMSNorm / 稳定 Softmax | [跨 tile 行操作](../research/kernels/CROSS-TILE-ROW-OPS.md) | 明确 CPU 统计、系数及 logits 处理；FP16 平方和/值域保护；不是纯 NPU 任意幅值归一化 |
| O04 | RoPE / GQA / KV / decoder 的功能闭环 | [attention](../research/kernels/ATTENTION-KERNELS.md)、[decoder](../research/graphs/TRANSFORMER-DECODER-BLOCK.md) | 主机 KV、固定容量、动态权重重排、混合 Softmax；错误回滚有受控调用层证据；不是 FlashAttention 或真实完整 LLM runtime |
| O05 | 调度观测、成本模型、同核与跨核并发 | [profiling](../research/scheduling/MULTICORE-SCHEDULING-PROFILING.md)、[成本模型](../research/scheduling/DEVICE-COST-MODEL.md)、[资源干扰](../research/scheduling/JOB-CONCURRENCY-INTERFERENCE.md)、[重叠](../research/scheduling/GEMM-GEMV-CONCURRENCY.md) | mask 1/2/4 与 AUTO=0；候选集合在用户态选单核；queue 为区间、service 为驱动包络；纯 MatMul 预测仍 experimental |
| O06 | INT8 开源实现对官方 SDK 的公平比较 | [官方对照复核](../research/rknpu2-matmul-validation/README.md) | 单核、完整主机调用、normal/native 取较快者；M=128 三形状 2.01/2.05/4.10×，M=1/4 略慢；正确性为 4096 点抽样，不是全矩阵检查或整模型 tokens/s |

推荐组织：先修正 F01–F06，再补软件栈的本地分支入口和 O01–O05；O06 独立成性能案例。每篇标注来源版本与证据等级，引用本地新增函数时使用**实际提交后**的可发布 commit 链接，不将它们挂到上游历史 commit。

## 4. 引用与结构问题

**S01：代码标识符引用未精确到支撑实现行。** 五篇专题虽 pin 了主参考 commit，但大量函数/字段/寄存器名只引用仓库 tree 或 `npu_hw.h:25–33`。该短区间不能替代每个 uAPI 字段、调度函数和布局生成器的对应定义/实现。按知识库 `AGENTS.md` 的“开源代码引用（必须精确到行）”规则，应逐项附对应文件的 pinned 行链接。这里没有进行外网全链接可达性与所有行内容复核，不把来源不充分说成链接必然失效。

**S02：一处真实相对链接错误。** `content/genai/README.md:74` 的 `../AGENTS.md` 实际指向 `content/AGENTS.md`，该文件不存在；仓库根文件应为 `../../AGENTS.md`。`writing-guide.md:23` 的 `xxx.md` 位于行内代码，是示例，不计为断链。其他静态存在性检查不能代替 HonKit 渲染、锚点或公式检查。

知识库构建会写入 `_book/`，因此本次没有在知识库执行 `npm run build`。本次未改知识库任何文件。

## 5. 本工作区需要保留的限制

- `docs/CAPABILITIES.md` 已提示历史 `research/validation/2026-10-09/README.md` 不存在，不引用旧“核心库 95、ggml 8”等统计作为当前可审计结果。
- 旧厂商 Conv 连续链的数值故障没有因新 LUT/Linear 链通过而被证明修复；历史梯度/训练微步不是完整 Loss 自动反传或收敛验收。
- 官方对照资料保留原始日志、CSV 换行与 patch 上下文；其归档 diff 有空白告警，不为消除格式告警改写实验记录。
- `custom-ops/results/` 不随 Git 分发；仓库文本中的结果链接在这台机器可读，但外部 clone 不保证存在。此次 JSON 索引仅固定小型成果摘要/manifest 的哈希，不替代原始张量、trace 和源码包。
- RK3588 片内三核成果不代表 RK182X 多卡、跨芯片通信或多设备成本已经实测。

## 6. 本次验证

```bash
cmake -S research/custom-ops -B /tmp/rknpu-audit-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rknpu-audit-build -j 4
ctest --test-dir /tmp/rknpu-audit-build --output-on-failure
python3 scripts/check_docs.py research
python3 scripts/check_docs.py docs
python3 research/custom-ops/tools/python/overlap_verify.py research/custom-ops/results/2026-10-14/overlap-20261014-145427
python3 research/custom-ops/tools/python/overlap_verify.py research/custom-ops/results/2026-10-14/overlap-20261014-155212
python3 research/custom-ops/tools/python/attention_campaign.py research/custom-ops/results/2026-10-13/attention-20261013-174358
python3 research/custom-ops/tools/python/decoder_campaign.py research/custom-ops/results/2026-10-13/decoder-20261013-184941
```

构建成功；CTest 18/18；文档入口链接检查通过；两轮 overlap 核验通过；已有 attention/decoder 数学参考与证据核验通过。知识库审查前后文件 SHA256 集合与 Git 状态一致，见审查快照。没有发布、推送或修改知识库。

## 7. 直接参考 rknpu2_matmul_open 的补充审查

上一轮参考的是本工作区归档的性能复核，未重新精读外部 checkout。用户追问后，直接只读检查 `/home/maoyu/Documents/code/rknpu2_matmul_open` 的 README、公共头文件、实现和相关文档。其 HEAD 为 `192f6ca25407fac7c0880123c0c394bd3e2f74f7`，与已归档性能报告的版本一致；Git 状态干净。没有修改该仓库，也没有新增该库的硬件测试。

### R01：整数设备侧 K 累加已有外部公共接口与执行实现

`rk_npu_quant_matmul.h` 明确提供默认关闭的 `RK_NPU_I8_NPU_REDUCE=1`：通过多核 PC chain 和 offline DPU INT32 split-K 累加。浮点算子的可选 NPU dequant 使用 FP16 舍入系数，是独立精度折衷。当前限制 M≤128、K≤131071、未压缩 W8 权重，不支持的配置报错。

- [公共接口与限制，L23–L33](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/include/rk_npu_quant_matmul.h#L23-L33)
- [实际 npu_reduce 执行分支，L825–L855](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/src/rk_npu_quant_matmul.cpp#L825-L855)

这进一步限制 F01 中“只有 FP16”的适用范围：它最多描述特定历史/本地框架，不能概括全部 RKNPU 软件。归档 INT8 官方对照使用默认 CPU split-K，不能用该性能表证明可选 NPU 累加后端已验收。

### R02：DCOMP 不是仅有 dense 直通的研究状态

知识库 `KB/block-internals.md:45` 称“寄存器映射目前只解出 dense 直通模式”“使能值未确认”。外部库已有 INT8 DCOMP encode/decode、压缩权重准备和非 dense 寄存器绑定：实际选择 `DCOMP_NORMAL_REGNUM`、控制字以及 16 路 amount。

- [CPU codec 定义与 native 布局，L9–L18](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/src/rk_npu_dcomp.h#L9-L18)
- [压缩寄存器绑定，L742–L755](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/src/rk_npu_matmul_i8.cpp#L742-L755)

建议把原否定改成“所引历史 capture 尚未覆盖压缩路径”，另列该外部实现。源码存在不证明知识库所引上游历史实验已有 capture；本次也未补做 DCOMP 真机数值验证。

### R03：本地混合归一化不代表所有实现都必须 CPU 统计

外部库已有 fixed-template native Add+RMSNorm：M=1/128、D=4096、eps=1e-5，输入输出采用 block-8 FP16；执行入口绑定地址后提交 task 链，没有本工作区混合 RMSNorm 所用的每行统计下载/CPU rsqrt/系数回传阶段。

- [公共接口与固定形状，L30–L55](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/include/rk_npu_add_rmsnorm_f16.h#L30-L55)
- [运行路径，L267–L306](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/src/rk_npu_add_rmsnorm_f16.cpp#L267-L306)

O03 应补录“通用跨 tile 混合实现”与“外部固定形状 native 模板”两个层级。本次只确认后者的源码路径，不主张任意形状、幅值或全模型下已经可靠。

### R04：外部 attention 有 native device KV 与有效长度计算

本工作区 `co_attention` 的主机 KV、固定满容量工作量是该实现的边界，不能推广成硬件限制。外部接口提供 DDR GEM 中的 native KV、容量增长，以及按有效 prefix/cache length 计算。它同时提供 fixed-shift experimental 与稳定 CPU Softmax 两条显式路径；不同模式、配置和文档中的历史支持范围要分别核对。

- [两种显式模式，L11–L14](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/include/rk_npu_attention_f16.h#L11-L14)
- [native KV 与有效长度，L91–L119](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/include/rk_npu_attention_f16.h#L91-L119)

其 decode submit/sync 失败会 poison 状态并要求 reload/recreate，不能与本地受控错误下的 KV 回滚混为一谈。README 的“Very fast fused”不是本次性能证明，稳定模式有 CPU Softmax，也不能统一称为纯 NPU FlashAttention。

### R05：软件栈应补充独立开源 SDK 替代路线

该库直接使用厂商 `rknpu` DRM ABI，运行核心接口不依赖 `librknnrt` 或 RKNN compiler/model；主线 Rocket 支持仍列 TODO。公共 raw MM/BMM 提供 FP16/BF16/TF32、strides/transpose、多核与 split-K，其默认 split-K reduction 仍在 CPU FP32。它是厂商内核上的开源寄存器库，不能等同于闭源 RKNN 路线或本地 custom-ops 的接口。

- [raw MM 数据类型、split-K 与数值契约，L1–L75](https://github.com/happyme531/rknpu2_matmul_open/blob/192f6ca25407fac7c0880123c0c394bd3e2f74f7/include/rk_npu_mm.h#L1-L75)

知识库软件栈建议增加该路线，并补充 DCOMP、整数设备累加、固定 native RMSNorm、device KV 和量化/MLA/MoE 等“外部源码可评估能力”。这些均须与本工作区已完成的真机验收分列；没有由现有 INT8 MatMul 性能表验证全部库功能。
