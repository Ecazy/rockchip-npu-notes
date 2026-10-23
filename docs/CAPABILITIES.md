# RKNPU 当前能力总览

整理日期：2026-10-14。依据当前工作区代码与现存证据；目标环境为 Orange Pi 5 Plus / RK3588，厂商 RKNPU 0.9.8，内核 `6.1.43-rockchip-rk3588-rknpu098`。各正式轮的 NPU 频率及控制方式以专项报告为准。

这里的“自写算子”是自行安排 Tensor/BO 布局、寄存器命令、task 顺序、同步和临时缓冲，组合 CNA、CORE、DPU、PPU 等固定硬件功能。当前没有通用指令内核、任意 C/CUDA kernel 编译器；新增静态 FP16 rank-2 子图编译器支持有限节点集合，尚无模型导入或通用图编译。

## 1. 证据口径

| 标记 | 判断依据 | 如何使用 |
|---|---|---|
| **本地真机** | 当前框架源码与现存真机日志/独立数值检查 | 按接口、形状、值域和驱动版本复用 |
| **历史厂商** | 导入实验台账与捕获，采用当时的发射器 | 作为已知机制/配方，不等于当前统一 API |
| **组件/上游** | 已获取组件源码或上游硬件报告 | 可以接入，但仍须本机完整验收 |
| **待补齐** | 尚无对应实现或足够证据 | 不能据此声称硬件不支持 |

**核对提醒：** 旧首页/进展快照提到 `research/validation/2026-10-09/README.md` 与“核心库 95、ggml 8 等通过”统计，但该报告目录在本次工作区检查中不存在。这里不将这些统计作为当前可审计验收；也不否定其历史陈述。当前自写算子、K 分片、task 链及调度正式证据均存在。

## 2. 底层逆向与控制能力

| 能力 | 当前可做的工作 | 证据与边界 |
|---|---|---|
| 寄存器命令编码 | 生成/解码 64 位 regcmd，分辨 target/value/register | 历史厂商与组件；字段编码见 [命令模型](../encodings/regcmd-task-model.md) |
| 厂商提交 ABI | 构造 packed 40 B task、regcmd 链、subcore 范围和阻塞 Submit | 本地真机；[task 链](../research/scheduling/TWO-TASK-CHAINING.md) |
| CNA/CORE MatMul | 用 1×1 Conv 路径组合矩阵计算，控制张量、权重和输出布局 | 本地 FP16 真机；整数/其他精度的范围另行登记 |
| DPU 后处理 | EW add/mul、bias、LUT，组合到 MatMul 后处理 | 本地真机；没有任意公式解释器 |
| PPU | 有界池化与空间归约配方 | 本地真机；独立 PPU 路径，不假设可任意混入 DPU 链 |
| BO/IOVA 生命周期 | 分配、映射、上传/下载、同步、View 偏移、设备直接交接、guard | 本地真机；地址须保持 fd/domain 身份与 BO 生存期 |
| Cube 与权重打包 | Host row-major 与设备 Cube 显式转换，K 片偏移 View | 本地真机；Cube 驻留不等于硬件 CBUF 一直驻留 |
| Task 链 | 下一段地址/长度、终端封口、两组寄存器状态交接、一次 Submit 多 task | 本地真机；只推广已验收的配方 |
| Capture/Hook | 捕获 ioctl、描述符、命令、寄存器组、IRQ 和 BO 生命周期 | 历史厂商 + 本地 Hook；完整捕获和实际执行范围分别检查 |
| 寄存器消融 | 用关闭/替换字段和独立输出检查分析任务行为 | 历史实验方法；不是全部字段均已穷举验证 |

CBUF/bank 容量规划、SDP precision、DMA stride 等可参考 [硬件概览](../hardware-overview.md)、[布局](../encodings/tile-layouts.md)、[精度字段](../encodings/precision-field.md)。主线 Rocket 和厂商驱动的调度、地址空间及同步语义分别判断。

## 3. 当前自写框架可以直接调用的能力

公共接口：[custom_ops.h](../research/custom-ops/include/custom_ops.h)，实现：[src](../research/custom-ops/src/)。

| 能力/API | 支持范围 | 状态与限制 |
|---|---|---|
| Tensor/Buffer/View | FP16/FP32 输出描述、逻辑/物理维度、偏移、padding、同步、guard | 本地真机；FP32 输出不代表原生 FP32 输入 |
| `co_emit_mm` | 有界 FP16 MatMul，单 primitive tile 的 M/K/N ≤256 | 本地真机；Host 形状可以非对齐，物理维度按契约补齐 |
| `co_emit_mm_acc` | MatMul 后融合 DPU EW 累加，双 BO 乒乓 | 本地真机；每片累计收窄到 FP16，禁止原位别名 |
| `co_graph_*` | 静态 FP16 Input/Linear/Add/Mul/SiLU/GELU/HardSigmoid 图及混合RMSNorm/Softmax/RoPE/self-attention；自动形状、tiling与生命周期规划 | 本地真机 SwiGLU FFN 三核验收；Linear/逐元素仅边界搬运，混合归约另有CPU处理；无 ONNX/GGUF 导入或动态 shape；attention为固定容量混合实现 |
| `co_gdn_*` | 固定单 head、dv/dk 1..256、FP16 六 task 状态更新；State 双 BO 保持 feature 布局；CPU-FP16/FP32 完整基线及实验性 beta-fold/merged/native-alpha 路线 | 三核五形状双模式及 32 步递推验收；三形状七路线在固定 CPU 2.304 GHz/NPU 1 GHz 下完成端到端与成本拆解；动态 k/q/v 仍 Host 打包；native 标量仅是已验证探针，未放开通用广播或原位别名；[结果](../research/kernels/GDN-EXECUTION.md) |
| `co_decoder_*` | 完整bias-free pre-RMSNorm decoder block；Q/K/V、RoPE/GQA、Wo/residual、SwiGLU/residual；persistent KV及M=1几何 | 本地真机三核双模式；完整block和post-FFN回滚通过；混合路径、容量固定、非真实模型runtime，API优化收益按形状限定 |
| `co_attention_*` | 固定容量MHA/GQA/MQA self-attention；RoPE、causal/window/additive mask、prefill与增量decode、事务式主机KV | 本地真机三核双模式，QK/PV在NPU；CPU KV/动态权重/Softmax，固定满容量计算；未测整模型/优化decode |
| `co_rope_*` | interleaved/split-half、partial、跨tile配对 | 本地真机；CPU配对/系数，NPU FP16乘加；head_dim≤256、总通道≤4096 |
| `co_rowop_*` | 跨tile RMSNorm/稳定Softmax，M/N≤4096，mask/负尾处理 | 本地真机三核归约及前置RMSNorm FFN验收；明确CPU统计/系数，Softmax还读取logits；RMSNorm保守拒绝tile行平方和≥65504，尚非纯NPU归约 |
| `co_linear_*` | 通用多 tile Linear，可选 bias/HardSigmoid，统一 M/K tile 输入 | 本地真机 FFN 复用；K 部分和设备 FP16 累加；每个输出 tile 一次 chain Submit |
| `co_linear_hsig_*` | `Y=HardSigmoid(XW.T+b)`；M/N/K 1..4096，自动分解 | 本地真机覆盖 M/N/K 同时分片和尾块；Host、单块 Cube、设备 tile 集合入口；链模式每个输出块一次 Submit |
| `linear_hardsigmoid` | FP16 row-major 文件 CLI，重复执行与 chain/separate 对照 | 本地真机；非法输入拒绝，无隐藏 CPU 计算回退 |
| `co_emit_ew` | 同 shape add/mul | 本地真机；通过 CNA identity + DPU EW，不是任意广播 |
| `co_emit_lut` | HardSigmoid、sigmoid、GELU gate，完整 LE/LO loader | 本地真机；近似精度以配方/验收为准 |
| `co_gated_activation` | SiLU、erf-GELU；LUT gate + EW multiply | 本地真机；设备中间 Cube，无 CPU 激活阶段 |
| `CO_EXP_BOUNDED` | exp，输入域 `[-4,4]` | 本地真机原型；当前验收绝对误差门限 0.0625；不等于通用 Softmax |
| `CO_EXP_NEG4` | exp，输入域 `[-4,0]`、unit scale | 用于分段稳定Softmax；不直接接受任意负尾输入 |
| `CO_RSQRT_BOUNDED` | rsqrt，输入域 `[0.25,4]` | 本地真机原型；当前验收门限 0.003；调用者保证值域 |
| `co_ones_upload` + MatMul | 末维 sum/mean，K 在单 tile 范围内 | 本地真机；mean 系数先量化到 FP16，不是任意轴归约 |
| `co_emit_pool` | FP16 max/average/global pooling 配方 | 本地真机；无 padding，kernel=stride，各轴整除，空间面≤256；核各轴1..16，average各轴至少2 |
| `co_emit_reformat_8x8x8` | 固定 512 元素重排，`out[c,a,b]=in[a,b,c]` | 本地真机；不是通用 ONNX transpose |
| `co_norm_hybrid` | RMSNorm/LayerNorm；NPU square/reduce/affine | 本地真机；CPU 统计、rsqrt、系数展开；FP16 square 非有限结果拒绝 |
| `co_execute` | separate 或 task-chain 提交 | 本地真机；BO 必须存活至同步调用返回 |

**K 扩展不是把 4096 塞进一个 tile：**X 全 Cube 按 256 通道拆成 View，权重预打包，每片 MatMul 用 DPU 累计，最后才执行 bias 与激活。M/N 现已自动分块，并提供输出块容量与 task/Submit 数规划；静态子图使用统一 tile 集合直接交接；全局大 Cube 的自动行切片、任意 K 和通用模型图分解尚未实现。

## 4. 多核分配与调度观测

接口：[co_schedule.h](../research/custom-ops/include/co_schedule.h)，[详细报告](../research/scheduling/MULTICORE-SCHEDULING-PROFILING.md)。

| 能力 | 可实现内容 | 不能误解为 |
|---|---|---|
| 逐 Submit 单核指定 | Mask 1/2/4 → Core 0/1/2；回写 Mask + 对应 IRQ + 数值核验 | 链内部 task 跨核迁移或抢占 |
| 候选集合 | 用户态 `co_schedule_pick` 在 allowed 集合轮询后，提交一个单核 Mask | 厂商把 3/5/6/7 当候选集合自动选核；这些是多核联合 Mask |
| AUTO | Mask 0，由当前驱动选择核，再读回实际 Mask | 自定义候选子集或稳定负载均衡保证 |
| Host 起止时间 | 请求开始、ioctl 前、阻塞返回、读同步/清理后的 API 完成 | 纯设备运行时间 |
| 入队时间 | 已审计指令顺序包围的时间区间 | 精确链表插入时间戳 |
| 开始/完成 | commit `ktime_get` 与核 IRQ 完成取时区间 | 精确寄存器 kick、硬件开始计算或硬件 IRQ 发出时刻 |
| 排队与服务分离 | queue 区间 + 驱动 `hw_elapse_time` + other 区间，交叉核验 | `wall-hw` 全部算 queue |

响应边界是 Host 首次观察到阻塞提交完成：

$$
T_{\mathrm{response}}=T_{\mathrm{queue}}+T_{\mathrm{service}}+T_{\mathrm{other}}.
$$

`service` 为寄存器配置到完成 IRQ 取时的驱动服务包络，包括配置和中断延迟；纯硬件执行时间仍待补齐。
输出读同步/资源清理由另一个 `api_response` 边界记录。

调度适配器接任意已发射 program（DPU 原语及 CNA/CORE MatMul），1–64 task/job，一个阻塞 Submit；有依赖的链整体在同一核。
同核 MatMul 与 DPU-only 的完整链/两次 Submit/regcmd 重划分统一见 [同核重叠实验](../research/scheduling/SAME-CORE-OVERLAP.md)：32 tile 上模块重划分相对完整链额外节省7.14%，不等于完整 attention 端到端加速。
同核 GEMM/GEMV 的单项、B8/B16 链式放大组及双进程并发对照见 [GEMM/GEMV 并发](../research/scheduling/GEMM-GEMV-CONCURRENCY.md)：已测链只摊销提交包络，同核 job 串行，跨核服务包络重叠；该报告只测完整 task/job，模块重划分的正结果见同核重叠实验。
抓取工具固定审核匹配内核代码与 tracer 能力，缺失/丢失/歧义证据拒绝通过。不存在可泛化到任意背景负载的无指针 job 关联保证。

### 单设备成本、并发与干扰 v2

本地真机完成945000次正式Submit及另120次受控tracing请求。工具为 `custom-ops/tools/c/cost_probe.c`、`interference_load.c` 和 `tools/python/device_cost_*.py`；部署入口为 `scripts/run-device-cost-board.sh`。

| 能力 | 本轮证据与适用范围 |
|---|---|
| 隔离成本模型 | 服务包络与Host API成本分别拟合；LUT及组合链分组几何留出达标，纯MatMul仍标experimental；不支持纯硬件/queue预测 |
| 等工作量并发 | foreground与延后全量验收resident分开；1/2/3/4/6/8worker，固定单核、分核、候选轮询、AUTO及burst |
| 较长Job竞争 | 同8worker的256³ MatMul分核/同核吞吐约1.96×，服务包络约1.33×；不解释为理想三核倍数 |
| 排队诊断 | 同核8worker MatMul queue上下界中位数535–538μs，分核46–48μs；仍是区间且与性能轮分开 |
| 背景干扰 | CPU compute与内存streaming成对；4线程streaming下64³ MatMul服务约2.1×；DDR/CBUF具体归因未证实 |
| 尾延迟 | 每条件25000请求，全量数值/padding/guard核验；block级p99及置信区间 |

详见 [单设备成本模型](../research/scheduling/DEVICE-COST-MODEL.md)、[并发与资源干扰](../research/scheduling/JOB-CONCURRENCY-INTERFERENCE.md)。校准模型只在所测精度、形状、执行路径、频率和驱动范围内使用。

## 5. 历史厂商实验可复用的机制

这些能力的源码与证据在 [厂商整合目录](../research/rknpu-vendor/INTEGRATION.md)，尚未全部迁入当前公共 API。

| 能力 | 已有证据 | 当前工程边界 |
|---|---|---|
| INT8/FP16 Conv | 历史数值回归，pad/stride/通道和几何覆盖 | 旧发射器；重复同构 Conv 链仍有独立数值故障记录 |
| INT8/FP16 Deconv | 步长2/4/8及核方向配方 | 不推广到任意步长 |
| PPU 行切分 | 高度受限配置与大图切分 | 个别大图超时未解；不是通用硅片高度限制 |
| BS/BN affine、Mul | 精度/系数和部分逐位对照 | 任意广播及连续状态切换另验 |
| 特定 transpose | 已分析若干 perm 的厂商程序 | 没有通用 transpose 发射器 |
| LayerNorm/Attention 捕获 | 多 task 中间 BO、LUT 与输入来源分析 | 没有完整纯 NPU 手写 LN/SDPA 数值闭环 |
| dW/dX 与训练微步 | 历史梯度数值、原 BO 权重更新后前向 | dY 注入/合成，包含 CPU 处理；未证真实 Loss 自动反传或端到端收敛 |
| 厂商多核分片 | 不同核使用各自 subcore task 范围 | 与当前独立 job 指定核分开；不推广到跨芯片多卡 |

旧 Conv 链问题未因新 HardSigmoid/LUT 链通过而被判定已修复。

## 6. 外部组件可以提供的能力

版本锁定在 [provenance.json](../research/upstream-components/provenance.json)；组件是独立 checkout，不是本仓库内置通用能力。

| 组件 | 可接入用途 | 当前应采用的口径 |
|---|---|---|
| `rocket-userspace` | 寄存器生成、BO/布局、MatMul/算子和 graph 原语 | 当前自写框架已复用其中 portable generator；完整库不同路径另验 |
| `rknpu-submit` | 将 Rocket 用户态提交语义适配厂商内核 | 当前自写框架已使用；调度细化由本地适配层提供 |
| `ggml-rocket` | ggml 后端，LLM/Whisper 算子对接 | 源码可接入；不能从现存自写算子证据声称整模型已跑通 |
| `tflite-rocket` | TFLite delegate，Conv/量化/CNN 对接 | 源码可接入；转换成功不等于硬件推理通过 |
| `ort-rocket` | ONNX Runtime execution provider | 源码可接入；没有当前完整 ORT 本机验收可供引用 |
| `patches` | 内核/平台补丁，Rocket 路径扩展 | 核对匹配内核；当前设备未自动应用这些补丁 |

上游笔记另报告 INT4/INT8/INT16/FP16/BF16/TF32 MatMul、requant、CBUF reuse 与更宽布局配方。
这些属于上游环境的硬件报告/组件能力，不能与本地 FP16 厂商实现的支持矩阵合并。见 [数据类型](../datatypes.md)、[CBUF reuse](../encodings/cbuf-reuse.md)、[requant](../encodings/out-cvt-converter.md)。

## 7. 关键验收与当前缺口

| 现存证据 | 结果 | 范围 |
|---|---|---|
| [完整算子交付](../research/custom-ops/results/2026-10-09/OPERATOR-DELIVERY.md) | 三核各两进程，2094 个在线检查、102 个文件接口检查 | 初版 K≤256 与原语/混合路径；不是2094种算子 |
| [Two-Task](../research/scheduling/TWO-TASK-CHAINING.md) | 12进程，各128轮，15360个数值对照 | A/B/C/D、组状态、IRQ、BO与padding |
| [K 分片](../research/kernels/FP16-K-TILING.md) | K≤4096，三核各两进程；两个对照形状 wall 加速中位数约1.43× | 单次Submit、Cube交接；相对同条件未融合设备链 |
| [调度](../research/scheduling/MULTICORE-SCHEDULING-PROFILING.md) | 96 traced job / 5766 task，另有同规模无 tracing 对照 | 核绑定、候选集合、AUTO与queue/service一致性 |
| [同核重叠](../research/scheduling/SAME-CORE-OVERLAP.md) | 最新6配置、8400次真实Submit；输出、IRQ与实际regcmd核验通过 | Core 0；32 tile 完整链比两次Submit省2.07%，重划分再省7.14%；模块周期未测 |
| [GEMM/GEMV 并发](../research/scheduling/GEMM-GEMV-CONCURRENCY.md) | 两轮各64采样；链式/放大组/双进程同核与跨核对照 | 链式只省约52µs Submit包络；同核并发串行、跨核重叠1.31× |

例如调度请求 `139344030004`：排队1220.131–1222.876 μs，服务455.869 μs，响应4121.775 μs；可以识别排队，而不是靠总时长猜测。
性能结论只在各报告的形状、精度、时钟、对照和时间边界下成立。

新增 [静态图编译与 SwiGLU FFN](../research/graphs/STATIC-GRAPH-COMPILER.md) 已完成 M/N/K 分解和静态子图 BO 规划。优先扩展方向：精确 enqueue/kick/job-ID tracepoint；减少 Submit 打包/分配开销；有数值契约的 Softmax/纯 NPU 归一化；将历史 Conv/梯度配方迁入统一接口。
RK182X 多卡和跨芯片通信没有由这些 RK3588 片内三核实验得到验证。

## 8. 开发入口

- 新寄存器原语：`research/custom-ops/src/custom_ops.c`，同步更新公共头与几何/别名契约。
- 新组合算子：`src/` 新模块，复用设备 View 与链；明确 CPU/NPU 分工。
- 新调度策略：`src/co_schedule.c` 上层选核，扩展 `tools/c/schedule_probe.c` 的受控场景。
- 新捕获/解析：`tools/python/`；缺失时间点保留“不可观测”，用反例测试约束推断。
- 测试与结果：`tests/` 放可版本控制的小 fixture，`results/<日期>/<轮次>/` 放不可变原始证据。

目录与兼容入口见 [项目结构](PROJECT-STRUCTURE.md)。
