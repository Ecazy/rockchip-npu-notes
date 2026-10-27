# RKNPU 逆向进展与已获信息

> 历史范围快照：当前支持矩阵以 [能力总览](docs/CAPABILITIES.md) 为准。K 分片和调度后续成果已验收；本文提及的旧 `research/validation` 报告在当前工作区缺失，不作为本轮可审计证据。

整理日期：2026-10-09。检查对象为当前工作区，厂商实验最新条目为台账 §7.194（2026-10-03）。本次执行文件校验、捕获完整性检查和脚本语法检查，没有连接板卡或重跑硬件实验。

历史记录称后续已完成上游源码获取及 README 能力真机验证（原报告 `research/validation/2026-10-09/README.md` 当前缺失）：RK3588 厂商驱动上核心库 95 通过/22 跳过、ggml 8 通过/1 跳过、TFLite 转换门通过，均无失败。下文保留原始离线审计范围；该历史结论不作为当前可审计验收，也不代表旧厂商实验发射器的缺陷已修复。

2026-10-09 后续新增 [自写组合算子框架](research/custom-ops/README.md) 及 [完整算子交付验收](research/custom-ops/results/2026-10-09/OPERATOR-DELIVERY.md)：三个核各两个独立进程，共 2,094 个数值对照和 102 个文件接口检查通过，测试区间无内核异常日志。它提供可复用的 `HardSigmoid(XW+b)` 上下文与用户 FP16 文件 CLI、设备 BO 直接交接和单次 submit 依赖链，以及 EW、LUT 激活、归约、PPU、固定 transpose 与混合 RMSNorm/LayerNorm。归一化的统计/rsqrt 和系数展开仍在 CPU；旧发射器重复 Conv 链损坏根因仍未定位。下面原有表格与离线审计结论保留原范围，新实现的具体支持边界以该目录文档为准。

**整体判断：已达到寄存器级算子实现、受控多核调度及小规模训练闭环阶段；尚未达到通用图编译器或完整厂商运行时替代阶段。** 基础 CNN 路径证据较强；复杂 task 状态、归一化、重排和 attention 仍存在实现或验证缺口。不宜给出“逆向完成百分比”：寄存器数量、已解码语义与可稳定运行的算子不是同一口径。

## 1. 两类资料的覆盖范围

| 资料 | 环境与来源 | 可以据此判断什么 |
| --- | --- | --- |
| 本地厂商实验 | Orange Pi 5 Plus / RK3588，内核 `6.1.43-rockchip-rk3588-rknpu098`，`rknpu 0.9.8`，1 GHz；RKNN Toolkit/Runtime 2.3.2 | 当前手写发射器能力、厂商 ABI、原始捕获及历史真机结果 |
| 上游主题笔记 | 导入 `gregordinary/rockchip-npu-notes` 的 `81b0d17`；主要为 Turing RK1 / RK3588 的 Rocket 路径，另有 RK3576 | 更广的数据类型、matmul、LUT、归约、模型和性能知识；不能自动视为本地代码已实现或复现 |

本仓库主体是笔记与实验工具。上游指南中的 `rocket-userspace`、`ggml-rocket`、`tflite-rocket`、`ort-rocket`、`rknpu-submit` 和 `patches` 已于 2026-10-09 获取为独立浅克隆，位于 [research/upstream-components/](research/upstream-components/README.md)，来源版本见该目录清单。源码获取尚未包含构建、安装和本地硬件验证；`refs/` 的参考源码也不等于当前实验发射器的能力。

入口：[整合与边界](research/rknpu-vendor/INTEGRATION.md)、[本地能力指南](research/rknpu-vendor/docs/guide.md)、[上游使用指南](guide/README.md)。

## 2. 已掌握的底层机制

| 主题 | 已获得信息 | 证据与边界 |
| --- | --- | --- |
| 硬件结构 | RK3588 三核，NVDLA 派生；CNA→CORE→DPU 计算路径，PPU 池化；CBUF 为 12×32 KiB | [硬件笔记](hardware-overview.md)；本地指南 §4 |
| 命令编码 | 64 位小端，`[63:48]=target`、`[47:16]=value`、`[15:0]=register` | 台账 A3；Hook 与发射器实现 |
| 厂商 ABI | 当前装置用 DRM ioctl magic `d`；task 描述符 packed 40 B；`regcfg_amount` 为 64 位命令数，读取 `round_up(amount+4,2)` 条，包含尾部 | 台账 A1、A2、A5；仅对应已核对的驱动版本 |
| 采集可靠性 | 已修正 IOMMU domain 混淆、IOVA 符号扩展、命令长度减半、固定上限截断；有 strict 和 strict-effective 检查 | 台账 A6–A10；完整捕获与执行范围分别验收 |
| 寄存器与字段 | 本地记录参考表 199 个寄存器、655 个位域；已有名称解码与算子指纹工具 | 台账 A4；这是参考表规模，不代表所有位域都经独立硬件验证 |
| 单元使能 | 本地证据支持 bit0=主使能/PC、bit2=CNA、bit3=DPU、bit4=DPU_RDMA、bit5=PPU、bit6=PPU_RDMA；CORE 未观察到独立对应位 | §7.194；bit1 语义未知；上游部分命名仍有不一致 |
| 命令族 | `0x18`：DPU+RDMA；`0x0d`：CNA+DPU，部分含 CORE；`0x1d`：增加 DPU_RDMA | §7.194；是单元参与组合，不能单凭 mask 唯一判定算子 |
| 数据布局 | INT8/FP16 的输入、权重、输出分组布局及矩形核 tap 顺序已有实验；FP16 像素原子为 8 元素/16 B | 指南 §4、[tile-layouts](encodings/tile-layouts.md)；Conv 与 Pool 工具的文件布局不同 |
| 容量规划 | 已有 bank/grains 自动规划及行、列、IC、OC 切分；FP16 IC 分片以 host FP32 累加 | §7.31–§7.38；具体余量公式须结合精度和配置，不能混用两条驱动路径的配方 |
| 数值路径 | FP16 `2048+1−2048=1` 探针支持至少 FP32 累加精度；BS/BN 表的 FP16/FP32 混合精度与 affine 顺序已表征 | §7.17、§7.23、§7.24；累加精度不等于原生 FP32 输入能力 |

主线 Rocket 的每 fd 调度及 4 GB IOVA 窗口，与厂商 `core_mask/subcore_tasks` 接口分开记录。相同 IOVA 数值也必须保留 domain 身份。[多核与地址空间笔记](perf/iova-and-multicore.md)

## 3. 算子逆向与实现程度

| 算子或能力 | 当前本地状态 | 最强已有证据 / 尚缺什么 |
| --- | --- | --- |
| INT8 Conv | 手写发射、历史逐值正确 | 硬化回归 74/74；覆盖矩形/1D/方形核、pad、stride、多通道组 |
| FP16 Conv | 手写发射、历史数值验证 | 最新套件 79 pass / 0 fail / 0 envfail；不同用例为位一致或允许 ULP 误差，不能全称 bit-exact |
| INT8/FP16 Deconv | 已验证 | 步长 2/4/8，核方向及 `S−1` 编码；不能扩展到所有步长 |
| PPU MaxPool | 已验证、支持行切分 | 当前配置 `ih≤63`；切分后 `ih=128` 也有整图 0 失配记录；部分大图 `k≥3,s=1` 超时未解 |
| DPU BS/BN/BRDMA affine | 已验证 | INT8 round-exact；FP16 bias 与独立 BN 有 bit-exact 用例；连续状态切换仍有偶发异常记录 |
| Mul | 厂商通路已表征 | BS_MUL；不等于任意广播逐元素算子均已实现 |
| Depthwise、FC/matmul | 本地机制已知 | 当前本地硬化回归未完整覆盖；上游分别有实现与硬件验证笔记 |
| Sigmoid/Tanh/Softmax | 本地表语义已解到不同程度 | 手写 loader/算子发射尚未工程化；上游已记录 LUT 激活、exp、rsqrt 等实现 |
| Transpose/Reformat | 特定形状与 perm 编码已解 | `p0123/0213/0231/0312` 的 RDMA 路径；`p0321` 降为 24-task CNA 链；尚无通用发射器 |
| LayerNorm | 厂商数值链已解释 | 18-task 链及中间 BO、γ/β 地址已定位；中段 NR 迭代未逐 task 完全配属；尚无手写发射器 |
| Attention/SDPA | 可喂真实多输入、采集与分析 | 有 Q/K/V 输入及 LUT 对账；尚无本地完整手写 attention 数值闭环 |
| 任意轴归约、Loss | 本地未提供通用接口 | 上游 ones-matmul/feature reduce/PPU reduce 可作为实现方向；未实现不能归因于硅片不支持 |

上游记录 RK3588 的 INT4、INT8、INT16、FP16、BF16、TF32 原生 matmul 均已硬件验证；当前本地厂商实验主要覆盖 INT8/FP16。INT16 输出超时的旧本地说法也不能覆盖上游已验证的 INT16 路径。[数据类型矩阵](datatypes.md)

## 4. 已验证的组合计算与训练

- **三层推理**：`conv→maxpool→conv` 历史逐值正确，池化有独立 Python 参考，两次独立进程输出 SHA256 一致。连接方式为工具间文件读写及 CPU 布局转换，并非单次 submit 的全驻留融合图。[§7.192 证据](research/rknpu-vendor/experiments/results/conv_pool_conv_20261003/README.md)
- **梯度原语**：INT8 dW 的 dot、系数矩阵和真实张量闭环；FP16 dW 的 dot/3×3/batch 42/42；反卷积形式的 dgrad；深通道及双层闭环均有历史记录。[能力指南 §1.2](research/rknpu-vendor/docs/guide.md)
- **真实激活上的训练微步**：用前两层 NPU 输出训练第三层；9216/9216 梯度 round-exact，原 BO 权重更新后再前向与 CPU(W′) 一致。稀疏 dY 密度 1/64 的用例饱和数为 0，7296/9216 权重改变。[§7.193 证据](research/rknpu-vendor/experiments/results/conv_pool_conv_train_20261003/README.md)

这些结果证明训练原语、数据对接和手工更新可闭环。dY 为注入或合成数据，尚未证明从真实 Loss 自动反传的完整训练图、端到端收敛或全 NPU 优化器；已有脚本包含 CPU 处理。

## 5. 调度与 task 链：必须分开的验收层次

1. **厂商多核分片已验证**：`--subcore-hi` 配合槽位 2..4 和 `core_mask=0x7`，每核一个 task；历史吞吐提升约 1.95×，限原实验环境。
2. **厂商异构依赖链可执行**：忠实重放 conv/BN 链后，两个输出区均有实际写入；成功路径 `task_counter=task_number` 是驱动回显，输出证据比返回计数更有意义。
3. **手写自链协议已修复**：终端尾写形态与段拷贝错位修复后可完成提交，后续 WDMA 落点正确。
4. **重复同构 conv 链的数值可靠性未解决**：§7.188 仍记录 task0 正确、task1 非确定损坏；重新同步并等候 500 ms 后不变。台账将其解释为管线竞态，但目前仍是根因假说，应与上游乒乓寄存器状态证据交叉核对。

因此指南“一次 submit 多 task 全部闭合”只能理解为已有路径的协议完成，不能推广为任意多 task 程序都数值正确。下一轮应核对 `S_POINTER`/执行组、每段尾写、输入依赖和独立输出参考。[整合文档 §3.2](research/rknpu-vendor/INTEGRATION.md)

## 6. 已修正结论与当前证据风险

| 容易误读的信息 | 当前应采用的口径 |
| --- | --- |
| sigmoid/tanh 每突发 443 项 | 捕获上限为 4096 B，属于捕获突发，不能当作整表容量；上游记录 LE/LO 各 513 项，完整 loader 为 1026 项 |
| sigmoid/tanh “精确解析重建” | `S=32767` 拟合在 ≤0.5 LSB 内分别命中约 99.1%/99.3%，最大残差约 0.519/0.513 LSB；语义吻合，但不是逐项无损重建 |
| attention 表可用两参数无损压缩 | 指数模型逐项命中率约 80.5%、最大误差 3；精确重放仍需原表 |
| FP16 无 FP32 累加 | 已被累加探针推翻 |
| 厂商不链 Conv | 已被厂商捕获及忠实重放推翻 |
| Pool 文件输出非确定即硬件异常 | 曾为 `bo_sync(from-device)` 前 dump 的缓存问题；代码已修复，不能混同重复 Conv 链问题 |
| 参考与硬件一致即输入管线正确 | 旧组间转置漏项仍可因参考读取同一错误输入而自洽；跨层需独立验证实际输入来源 |
| 方图要求、PPU 高度 63 是普适硅片限制 | 当前本地发射配置的观察；上游有更广路径，需在相同配置下复核 |
| `0x8034/38/3c` 可直接用来量 DMA 字节 | 本地指南列为计数器；上游认为该组为配置用途，另有读取硬锁记录；当前未取得本地有效字节计数证据 |
| matmul 行数必须是 4 的倍数 | 上游 matmul 配方的实测约束，不应直接推广到所有本地卷积几何 |

依据：[LUT 拟合数据](research/rknpu-vendor/experiments/results/lut_act_20261003/fit_summary.json)、[attention 拟合数据](research/rknpu-vendor/experiments/results/lut_attn_20261003/fit_summary.json)、[DMA 计数笔记](perf/hw-byte-counters.md)、台账 §7.188、§7.192–§7.194。

## 7. 当前快照完整性与本次检查

| 检查 | 本次离线结果 |
| --- | --- |
| 迁移来源清单 | 484 个文件全部存在；483 个 SHA256 相同；仅 `.gitignore` 不同，包含新增 `.archive/` 排除规则 |
| 本地参考依赖 | 18 个源码符号链接目标均存在；换机器或 Git checkout 不会自动得到这些依赖 |
| 关键 LN 证据 | trace、`em0d_analyze.py`、post-run `ln_bos/02.bin` 均存在 |
| LN 捕获完整性 | `analyze_rknpu_ioctl.py --strict --strict-effective` 退出 0；54/54 task 命令可解析，完整性与执行范围检查通过 |
| LN 实际执行范围 | 54 个候选 task，按 subcore 范围判定有效 18 个；不能据此声称三核同时执行 LN |
| 脚本静态检查 | 实验及 Hook 下 40 个 Python 文件 AST 解析通过；实验根目录 44 个 Shell 脚本 `bash -n` 通过 |
| 原仓库历史 | 本地 `.archive/rknpu-history.bundle` 存在；未纳入 Git |

本次没有复算所有算子数值，也没有编译或提交板端程序。INT8 74/74、FP16 79/0/0 均为历史结果，其中 FP16 suite 存在不计入 pass/fail 的信息性超时探针；“回归全绿”不表示所有探针均无超时。第 124 轮 R1–R6 收官日志也是历史验收，不是本次检查结果。

原始 trace、模型、二进制及日志部分受 `.gitignore` 排除；当前本机存在不代表发布后的 clone 完整。外部 `findings/`、部分板端日志仍未迁入。`em0d_analyze.py` 还硬编码旧工作区寄存器头路径，一些复跑脚本依赖 `/mnt/ssd/workspace/rknpu-re/`；目前可以查阅证据，但工具尚未全面做到搬迁后独立运行。

## 8. 后续优先级

1. **先解决 task 状态与数值验收**：以两组寄存器状态和每 task 独立参考验证重复链；将协议完成与数值正确分别设门。
2. **补齐 LUT 工程实现**：统一访问地址、bank 和捕获突发索引，以完整原表做基准；先独立激活，再连接 softmax。
3. **实现归约、EW、Transpose 与 LayerNorm**：从已知 ones-matmul/PPU 配方及简单 perm 开始；保留中间 BO 对照，逐步还原 LN 迭代。
4. **推进完整图与训练**：验证真实 dY 来源、Loss 和反向图，再讨论优化器、驻留融合与性能；已有微步不能替代这些验收。
5. **改善复现与跨芯片验证**：去除旧绝对路径、记录构建/输入/输出哈希、补齐被忽略的证据清单；RK3576 上游已验证，但本地厂商发射器仍需独立移植与测试，RK3566 仍为计划状态。

本地源码入口：[Hook/解码器](research/rknpu-vendor/source/hook/rknpu/)、[Conv 发射器](research/rknpu-vendor/experiments/min_conv_exp.c)、[Pool 发射器](research/rknpu-vendor/experiments/min_pool.c)、[梯度与更新](research/rknpu-vendor/experiments/wgrad_real.c)、[厂商重放](research/rknpu-vendor/experiments/vendor_replay.c)、[实验与回归入口](research/rknpu-vendor/experiments/README.md)。
