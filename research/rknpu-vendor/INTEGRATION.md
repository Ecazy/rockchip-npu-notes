# 厂商 RKNPU 实验与主线 Rocket 笔记整合

后续分析以 `rockchip-npu-notes` 为工作目录。本文连接上游主题笔记与原 RKNPU 实验。
原始文档和实验文件保留目录关系，便于复核历史结论。

## 1. 工作入口与来源

- [上游主题入口](../../README.md)：硬件结构、编码、性能与主线使用指南。
- [原能力指南](docs/guide.md)：厂商路径的现有能力与工具用法。
- [原实验台账](docs/RKNPU-实验台账.md)：截至 §7.194 的实验、更正和负结果。
- [实验与复跑入口](experiments/README.md)：发射器、回归与分析脚本。
- [来源清单](provenance.json)：来源 commit、文件 SHA256 和本地依赖。

| 项目 | 原工作区 | 新基础项目 |
| --- | --- | --- |
| 仓库 | 本地 `RKNPU` | `gregordinary/rockchip-npu-notes` |
| 基准 commit | `8cc08b1`，完整值见来源清单 | `81b0d17`，完整值见来源清单 |
| 板卡 | Orange Pi 5 Plus，RK3588 | 上游记录主要采用 Turing RK1；另有 RK3576 |
| 原实验运行环境 | 内核 `6.1.43-rockchip-rk3588-rknpu098`；驱动 `rknpu 0.9.8`；1.0 GHz | 每项以上游标签记录的板卡、内核、驱动和频率为准 |
| SDK | RKNN Toolkit/Runtime `2.3.2` | 主线 `rocket`、自建用户态栈；部分笔记也有厂商路径对照 |

上游文件保留原文。本轮新增对应索引，没有把待复核结论写入上游主题正文。
本轮没有运行板端实验，表中的数值均为历史记录。

## 2. 主题与证据对应

下表的“本地证据”路径均相对于本文所在目录。台账 §7.x 使用原编号。

| 主题 | 上游主题文件 | 本地证据与补充 | 合并判断 |
| --- | --- | --- | --- |
| 命令与任务 ABI | [regcmd-task-model](../../encodings/regcmd-task-model.md) | 台账 A1～A10；`source/hook/rknpu/` | 64 位命令编码可核对；ioctl 和 task 描述符分别记录 |
| block 使能位 | [regcmd-task-model](../../encodings/regcmd-task-model.md)、[ppu-pooling](../../encodings/ppu-pooling.md) | §7.194；`experiments/results/vendor_op_semantics_20261002/` | CORE 无独立位的观察可补充；bit0 的命名存在差异 |
| INT8、FP16 布局 | [tile-layouts](../../encodings/tile-layouts.md)、[matmul-as-conv](../../matmul-as-conv.md) | §5、§7.15、§7.18、§7.36；`min_conv_exp.c` | 按输入、权重、输出及数据类型分别核对；保留矩形核判别数据 |
| CBUF 与 tile | [cbuf-bank-slack](../../encodings/cbuf-bank-slack.md)、[cbuf-reuse](../../encodings/cbuf-reuse.md) | §7.31、§7.32、§7.38；`experiments/results/vendor_bank_rules_20260930/` | 12 × 32 KiB 可对照；bank 余量、grains 公式需匹配发射配置 |
| 多核与 IOVA | [iova-and-multicore](../../perf/iova-and-multicore.md) | A6、A7、§7.170、§7.171；`experiments/results/multicore_vendor_20260930/` | 主线 fd 调度与厂商 subcore 槽位分别记录；IOVA 查找保留 domain 条件 |
| task 链与寄存器状态 | [regcmd-task-model](../../encodings/regcmd-task-model.md)、[cross-op-chaining](../../encodings/cross-op-chaining.md) | §7.173～§7.188；`experiments/results/chain_cbc_20261003/` | 分开核对协议完成、输出正确与跨算子 BO 复用 |
| BS、BN、EW 精度 | [sdp-stage-precision](../../encodings/sdp-stage-precision.md)、[out-cvt-converter](../../encodings/out-cvt-converter.md) | §7.12、§7.23、§7.24、§7.175 | affine 表布局和 integer EW 模式互补；逐字段核对精度设置 |
| LUT 装载与激活 | [dpu-lut-activation](../../encodings/dpu-lut-activation.md) | A13～A16、§7.190、§7.191；`experiments/results/lut_act_20261003/`、`lut_attn_20261003/` | 上游提供发射配方，本地提供厂商拟合；区分完整表与捕获突发 |
| PPU 与归约 | [ppu-pooling](../../encodings/ppu-pooling.md)、[ppu-reduce-mean](../../encodings/ppu-reduce-mean.md)、[feature-reduce](../../encodings/feature-reduce.md) | §7.13、§7.90；`min_pool.c` | 上游补充平均池化和 ones-matmul 归约；本地高度边界保留为当前配置观察 |
| 转置卷积 | [conv-transpose](../../encodings/conv-transpose.md) | §7.22、§7.27；`experiments/results/deconv_boundary_20260930/` | 核方向、S−1 编码和 2/4/8 步长可交叉核对 |
| LayerNorm 与 Transpose | [norm-vision](../../encodings/norm-vision.md)、[rmsnorm-onnpu](../../encodings/rmsnorm-onnpu.md) | §7.194；`experiments/results/bn_lower_20260928/em0d/`；`em0d_analyze.py` | 厂商数值链与 perm 编码是补充证据；手写发射仍待实现 |
| 训练原语与闭环 | [feature-reduce](../../encodings/feature-reduce.md)、[whisper-encoder](../../encodings/whisper-encoder.md) | §7.25～§7.41、§7.192、§7.193；`wgrad_real.c` | 本地补充 dW、权重更新与真实激活闭环；尚无通用自动微分接口 |

## 3. 必须保持边界的结论

### 3.1 调度机制依赖驱动

`rocket` 的多 fd 策略与厂商的 `core_mask/subcore_tasks` 是两套接口。
厂商 `--subcore-hi` 的槽位规则不能用于主线提交。
本地 Hook 只针对厂商 ABI；主线采集需要对应的驱动接口工具。
同一 IOVA 数值可能属于不同 domain，解码必须保留 domain 身份。

### 3.2 task 链的完成与数值正确分别验收

台账 §7.188 记录了协议完成，但后续同构 task 输出受损。
该条对竞态的解释仍需与寄存器状态实验交叉验证。
上游记录了乒乓组保留状态，以及默认配置下隔一个 task 继承状态。
上游也明确其 delta 实测未覆盖自链任务。
这些证据形成新的排查方向，还不足以确定本地损坏的根因。

后续先比较四个 `S_POINTER` 的写入与执行组，再检查尾写和每 task 输出。
采用不同几何、不同输入和输出哨兵，避免同构数据掩盖状态继承。

### 3.3 LUT 表项数与拟合口径

上游描述 LE、LO 各 513 项，与本地完整 loader 的 1,026 项可对照。
本地 §7.191 的 443 项属于对应捕获突发，不能解释为硬件总容量。
该条采用 `S=32767`；早期指南使用 `32768` 的近似描述。
后续以突发边界、`LUT_ACCESS_CFG` 地址和原始表项建立统一索引。
§7.190 的指数拟合逐位命中率为 80.5%，两参数表示属于近似模型。
重建原表仍需原始 CSV 或无损表示。

### 3.4 能力限制需要配置范围

原指南的任意轴归约“未提供”描述当前发射器能力。
上游的 feature reduce 和 PPU reduce 提供了可移植的实现方向。
原指南的方图准备要求与 PPU `ih ≤ 63` 边界也来自特定配置。
引用时附带程序、数据类型、形状和寄存器设置。

本地 §7.194 给出了特定 perm 的 DPU reformat 编码。
这需要与上游关于布局转换的概括逐例核对。
区分特定硬件重排、CPU 打包，以及通用布局转换接口。

### 3.5 使能位命名存在上游内部差异

上游 task 笔记将 bit0 解释为 PC，pool 笔记的一段说明将其命名为 CNA。
pool 笔记也在 `0x1d` 的说明中列出 CORE。
本地 §7.194 记录 CORE 有无与 mask 不对应，并给出 bit2=CNA 的证据。
这些名称需统一核对，不能直接照抄成最终位表。

## 4. 下一轮分析顺序

1. **离线核对使能位与状态。** 对照 §7.194 和 task/pool 笔记，整理字段级证据。
2. **建立 LUT 表索引。** 结合上游发射配方与本地原表，先验证装载、地址和转换链。
3. **复核 task 链数值。** 用独立 task 输出对照乒乓组配置，重新验证 §7.188 的解释。
4. **移植归约与归一化。** 从 ones-matmul 和独立 EW 原语开始，再覆盖厂商 LN 数值链。
5. **扩展训练闭环。** 在前述原语通过后，连接已有 dW、dgrad 与真实激活实验。

每项新结论记录板卡、内核、驱动、频率和 shape。
同时保存完整命令、输入和输出 SHA256。
采用上游证据标签时按实际方法标注：观测命令流不等同于 `[HW sweep]`。
性能数字同时附带计时边界；历史通过次数不等同于本轮复跑结果。

## 5. 本地复用与可移植性

从新仓库进入原工具目录后，继续使用历史命令：

```bash
cd research/rknpu-vendor
python3 source/hook/rknpu/analyze_rknpu_ioctl.py --strict trace.jsonl
python3 experiments/rknpu_trace_report.py trace.jsonl
```

`trace.jsonl` 替换为对应捕获文件。Shell 脚本的板端绝对路径继续沿用原配置。
`infra/` 仍依赖原工程，当前迁移没有补齐该构建环境。

迁移保留了全部 272 个 Git 跟踪文件。
另将本机已有的结果日志、二进制依据和模型复制到本地。
这些文件继续遵循原 `.gitignore`，部分不会出现在以后发布的 Git checkout 中。
[来源清单](provenance.json) 的 `tracked_in_source` 区分两类文件。
外部 `findings/` 和台账引用的部分板端文件仍未包含在当前快照中。

`refs/` 下的源码库通过本地符号链接复用旧工作区，避免重复复制约 4 GB。
这些链接不纳入 Git；换机器需按 [refs 索引](refs/README.md) 重新准备依赖。
旧仓库全历史保存在本地 `.archive/rknpu-history.bundle`，该文件也不纳入 Git。
可用它查看台账引用的清理前文件：

```bash
git clone research/rknpu-vendor/.archive/rknpu-history.bundle /tmp/rknpu-history-review
git -C /tmp/rknpu-history-review show 972d97c:README.md
```

上游保留其 CC BY 4.0 许可与来源说明。
原工作区没有单独的顶层许可声明，迁入文件保留原有版权与来源信息。
本次本地整合没有自动为这些代码指定新许可。
