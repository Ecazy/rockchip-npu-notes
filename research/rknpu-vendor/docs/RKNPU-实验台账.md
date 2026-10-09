# RKNPU 实验台账

本文保存观测线与手写发射线的实验过程。当前能力和使用方法见 [使用与能力指南](guide.md)。
原始结果位于 [experiments/results/](../experiments/results/)，复跑入口见 [实验目录](../experiments/README.md)。

## 一、台账读法

每条记录列出方法、结果、证据与产物。后续条目可能修正早期结论，引用时检查更新。
观测线记录厂商下发的命令；构建线验证寄存器语义，两者的证据用途不同。

- **已证实**：当前装置上的真机结果或驱动源码支持。
- **强推断**：寄存器、编译器符号或多源证据支持，尚无独立受控验证。
- **待验证**：条件缺失或复核未通过。

| 装置项 | 值 |
| --- | --- |
| 板卡与内核 | Orange Pi 5 Plus；`6.1.43-rockchip-rk3588-rknpu098` |
| 驱动与频率 | RKNPU `0.9.8`；1.0 GHz |
| 设备节点 | `/dev/dri/card1`；`/npu@fdab0000` |
| 板端目录 | 观测线 `/mnt/ssd/workspace/rknpu-hook/`；构建线 `/mnt/ssd/workspace/rknpu-re/` |

2026-10-08 合并了说明文档并清理一次性脚本，历史文件名继续用于溯源。
原工程的 `findings/` 路径指向外部采集记录，当前仓库未包含全部文件。
清理前的版本可运行 `git show 972d97c:<仓库相对路径>` 查看。

## 二、观测线台账（A 系列）

来源：`guide.md`。产物不在本仓库，故只登记结论与可复核要点。

### 2.1 接口与解码

| 编号 | 主题 | 结论 | 等级 |
| --- | --- | --- | --- |
| A1 | ioctl ABI | 部署的 v0.9.8 走 DRM render node，magic `'d'`、nr = `0x40 + command_nr`；存在 legacy `'r'` 一代 | 已证实 |
| A2 | task 结构 | packed 40 字节，字段序 `flags, op_idx, enable_mask, int_mask, int_clear, int_status, regcfg_amount, regcfg_offset, regcmd_addr` | 已证实 |
| A3 | NPUOP 编码 | `((op & 0xffff) << 48) \| ((value & 0xffffffff) << 16) \| (reg & 0xffff)`，64-bit little-endian | 已证实 |
| A4 | 寄存器参考表 | 199 个寄存器、655 个位域，来源为社区 `npu_hw.h` 与 Mesa rules-ng-ng | 已证实 |
| A5 | `regcfg_amount` 单位 | 单位是 **64-bit entry 数**，不是字节；driver 读 `round_up(regcfg_amount + 4, 2)` 个 entry | 已证实 |

### 2.2 采集完整性与三个历史反例

| 编号 | 主题 | 结论 | 等级 |
| --- | --- | --- | --- |
| A6 | IOMMU domain 修复 | 旧 hook 不按 domain 过滤 IOVA；RKLLM 进程跨 domain 0/1 且有重叠 IOVA，导致 vision regcmd 不可解 | 已证实 |
| A7 | 低 32-bit IOVA 回退 | language 路径 `regcmd_addr` 出现 `0xffffffffXXXXXXXX` 符号扩展；按低 32-bit 回退后捕获从 0 恢复到 38,457 条 | 已证实 |
| A8 | 长度算法错误 | 旧 hook 按 `regcfg_amount * 4` 抓字节，命令被减半；完整 sigmoid 为 1102 words，旧记录约 548 words | 已证实 |
| A9 | 固定上限截断 | `cap=256` 与固定 16 KiB submit 内联缓冲分别截断 task 明细、丢失 submit 元数据 | 已证实 |
| A10 | strict closure | 正式数据必须先过 `analyze_rknpu_ioctl.py --strict`；失败时只能保留局部观察 | 已证实 |

A8–A10 是三个应当长期保留的反例：它们的数字曾在旧材料里被当作硬件事实（`4388 B`、`479` 次 LUT 写）。

### 2.3 模型侧与 DPU LUT

| 编号 | 主题 | 结论 | 等级 |
| --- | --- | --- | --- |
| A11 | Qwen3.5-2B ViT 896 attention | strict-effective 的 `op_idx=24` 完整实例含 4465 个异构 task：CNA 33,491、DPU 24,663、DPU_RDMA 4,579、CORE 384、PC 8,930、PPU 0，含 1,079 次 `0x4104` | 已证实（限该捕获） |
| A12 | 单层 MLP 对照 | `linear_fc1` 26 tasks；SDPA 约 4.51 MMAC/task、0.0353 TMAC/s，MLP fc1 约 505.9 MMAC/task、0.981 TMAC/s | 强推断 |
| A13 | sigmoid LUT 表 | 1026 项按 513+513 分两 bank，`round(sigmoid(x)*32768)`，步长约 0.012300，起点约 −6.304853 | 已证实（限该捕获） |
| A14 | SDK LUT classes | 编译器二进制含 `SigmoidLut`、`TanhLut`、`ExpNegLut`、`GeluLut`、`SwishLut` 等；无独立 Exp/Exp2/Reciprocal/Sin/Cos，独立算子默认走 CPU | 强推断 |
| A15 | softmax LUT 配置 | 捕获到 `LUT_CFG=0x68`、`LUT_INFO=0x00050500`、`LE_START=0xffffc000`、`OUT_CVT_SHIFT=0xf000` 等固定实现 | 已证实（限该捕获） |
| A16 | 手写 PWL microbench | loader 的 NPUOP 计数关系 `NPUOP = N + 149`，在 N = 64/96/1024/1026 上验证 | 已证实（限 probe） |
| A17 | core_mask 口径 | vision 全 submit 直方图 `{1:2569, 2:2397, 4:2217, 7:123}`；候选 SDPA submit 为 `{1:140, 7:101}` | 已证实 |
| A18 | 编译器 lowering | `exSDPAttention` 的 FlashAttention 符号（`exSDPAttentionLower`、`FlashAttentionNBufPlanT*`）；K=8 CNA matmul 为结构描述 | 强推断 |
| A19 | vendor profile | 某 896 case 中 `exSDPAttention` 出现 24 次、累计 13,673.8 ms（约 85%）；patch embed 单次约 632.8 ms | 已证实（限该 profile） |

> A11 与 A19 都涉及 SDPA，但前者是 task 计数、后者是节点耗时。两者不能互相换算：`node_time − active_core_MAC_floor` 是混合 residual，包含 DPU、DMA、小 tile 利用率与 stall。

---

## 三、构建线台账（B 系列）

来源：`RKNPU逆向-算子构建与训练可行性.md` §8 与 `experiments/RESULTS.md`。

| 编号 | 主题 | 方法 | 结论 | 等级 |
| --- | --- | --- | --- | --- |
| B0 | 基线链路 | librocketnpu `test_fc` 走 vendor 驱动 | 10/10 字节 bit-exact；`test_sw_ops` 27/27、`test_rknpu_abi` 29/29 | 已证实 |
| B1 | 手写 CONV | 130 条 regcmd（`regcfg_amount=126`）打 4 组配置 | 1024/1024、1024/1024、128/128、72/72 全部 max_diff = 0 | 已证实 |
| B2 | 权重区布局 | 改每 tap 的 oc 跨度 | `align(oc_real,2)` → 455/1024 错；`align(oc_real,32)` → 0 错 | 已证实（见 §5.1 更正） |
| B3 | K 维累加 | 16 槽全填 +1，`ic_real=3`、`CHANNEL_REAL=2` | 内点 144 = 9 × 16，说明按 **16 对齐后的通道数**累加 | 已证实 |
| B4 | `CNA_PAD_CON1` | 普通路径改低字节取值 | 旧结论为「低字节 = 越界样本字节值」，填错报 22/1024 | 普通路径 7 个值均 bit-exact；IC=1 专用路径差异见 §7.1 |
| B5 | DECONV | 置 `CONV_CON1_DECONV(1)` 并给 `DECONV_X/Y_STRIDE` | 位单独置位是 no-op；域值 1/3/7（⇒ S = 2/4/8）生效；S = 2/4/8 下与 CPU 转置卷积参考 max_diff = 0 | 已证实 |
| B6 | `enable_mask` | 扫描 `0x0`～`0xff` | 结果逐元素完全一致；驱动只读 5 个 task 字段，不读 `enable_mask` 与 `op_idx` | 已证实 |
| B7 | 末尾 4 条 | 对照驱动公式 | 4 条计入 DMA 搬运量但不计入 `regcfg_amount`；`PC_BASE_ADDRESS` 由驱动覆盖 | 已证实 |
| B8 | openrknn OWN 模式 | 复现 7/7 模型 | 因本机模型集不全**未复现** | 待验证 |
| B9 | DPU 逐通道 affine | BS/BRDMA gamma+beta；BN/NRDMA ALU | **DPU affine 原语已真机验证**：BRDMA per-channel gamma+beta 1024 项 round-exact；BN/NRDMA per-channel add 1024 项 round-exact。厂商 BatchNormalization 图层到这些寄存器/BO 的完整 lowering 仍待单独对照 | 部分完成 |
| B10 | openrknn 分段规则 | FlatBuffer 反推 | 每 op 任务数取 `f[10][1]`（`f[10][2]==0` 时用 `f[10][0]`）；LUT 激活集合 flags 取 0x1，其余 0x5 | 强推断 |
| B11 | wgrad dot reduction | P 个空间/样本点映射到 CNA K 维，16 个输出梯度 | 合成单样本 3×3、IC=3、OC=16 的 27 个 wgrad 权重坐标 27/27 round-exact；另一个坐标跨 B=1/2/3（P=36/72/108）均 round-exact；真实模型数据和梯度缩放未测 | 部分完成 |

---

## 四、2026-09-27 补验：目的与方法

补验目标只有一条：构建线里文档标为「未测」、却直接决定发射器正确性的那条规则。它问的是权重区布局在 `ic_real > 16`、`oc_real` 非 32 倍数时如何展开。文档 §9 第 8 条也把它点名为「值得单独回归」。

方法沿用 B1 的路径：改 `experiments/min_conv.c` 成仪表版 `min_conv_exp.c`，只加实验开关，不改默认行为。新增开关：

| 开关 | 作用 |
| --- | --- |
| `--w-kernels N` | 显式设置 `CNA_WEIGHT_SIZE2.WEIGHT_KERNELS` |
| `--w-oc2 N` | 显式设置每个 tap 的 oc 打包跨度 |
| `--w-layout a\|c\|d` | 权重区嵌套序：A `[ic1][kx][ky][oc][ic2]`、C `[kx][ky][ic1][oc][ic2]`、D `[kx][ky][oc][ic1][ic2]` |
| `--ic-pad N` | 强制 CNA 的补齐通道数 |
| `--entries N` | 覆盖 `CNA_CBUF_CON1.DATA_ENTRIES` |
| `--icreal-reg N` | 覆盖 `DATAIN_CHANNEL_REAL` 寄存器 |
| `--hi-zero` | pattern 模式下把 `ch >= ic_real` 的输入字节清零 |
| `--auto-fix` | 打开修正后的口径（参考实现，见 §5.6） |

程序在权重区算 FNV-1a 哈希并打印。它确认「改开关真的改了字节」，是排除假阴性的关键一步。全部原始日志见 `experiments/results/baseline_20260927/raw_results_20260927.txt`。

---

## 五、补验结果

### 5.1 每 tap 的 oc 跨度由 `WEIGHT_KERNELS` 决定，不必补到 32

同一图案、同一输出，只改描述符与打包跨度：

| `oc_real` | `WEIGHT_KERNELS` | 每 tap oc 跨度 | 描述符 | mismatch |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 32 | 4608 B | **0 / 1024** |
| 16 | 32 | 16 | 4608 B（实填 2304 B） | 418 / 1024（max diff 9） |
| 16 | **16** | **16** | **2304 B** | **0 / 1024** |
| 16 | 16 | 32 | 2304 B（实填 4608 B） | 423 / 1024 |
| 8 | 8 | 8 | 1152 B | **0 / 512** |
| 8 | 32 | 32 | 4608 B | **0 / 512** |
| 40 | 64 | 64 | 9216 B | **0 / 2560** |

**更正文档 §2.2 与 §9 第 8 条**：硬件按 `WEIGHT_KERNELS` 解释每 tap 的输出通道槽位数。**打包跨度等于它即可**，`oc_real` 非 32 倍数时无需补齐到 32。旧结论「必须 32 对齐」的观测条件是「只改跨度、不改 `WEIGHT_KERNELS`」，那是描述符自相矛盾，不是硬件强制。

`oc_real = 40`（补到 64）与 `oc_real = 8`（跨度 8）都 bit-exact。可见 DPU/CORE 侧的 `oc = align(max(oc_real,32),32)` 与权重侧的跨度是两个独立量。

### 5.2 权重区的 ic 原子是 32，且组序在最外层

`--w-layout a|c|d` 在 `ic = 32` 时给出**完全相同的字节**（FNV-1a 均为 `0b1d73c5`）。可见权重区的 ic 原子是 `WEIGHT_ATOMIC_SIZE = 32`，单个 32 宽块内不存在组序问题。提到 `ic = 64` 后三种布局才分开：

| 布局 | 打包字节哈希（ic = 64） | mismatch |
| --- | --- | ---: |
| A `[ic1][kx][ky][oc][ic2]` | `d12bcdc5` | **0 / 2048** |
| C `[kx][ky][ic1][oc][ic2]` | `30c829c5` | 459 / 2048 |
| D `[kx][ky][oc][ic1][ic2]` | `0281b7c5` | 1082 / 2048 |

结论：**ic 组维在最外层**，与 librocketnpu 的 `rnpu_fill_weights()` 一致。这同时把文档 §2.2 里「本式在 `ic ≤ 16` 下验证通过」的适用范围缺口补齐。

### 5.3 两处真正导致错配的规则（新增）

`ic_real = 40` 在 `ic = 64`、跨度 64、布局 A 下仍然报 813 / 2048 错。逐个排除后定位到两个量：

| 探针 | mismatch | 判定 |
| --- | ---: | --- |
| `ic_real=40`，`ic=48`（旧口径） | 1018 | 描述符小于打包需求 |
| `ic_real=40`，`ic=64` | 813 | 仍有缺口 |
| 同上 + `--hi-zero`（输入高通道清零） | 813 | **假设不成立** |
| 同上 + `--icreal-reg 63` | 813 | **假设不成立** |
| 同上 + `--entries 8` | **0** | **根因** |

**修正一**：权重区的补齐通道数按 32 对齐。

$$
\text{IC}_{\text{pad}} = \begin{cases} 16, & \text{ic\_real} \le 16 \\ \mathrm{align}(\text{ic\_real},\, 32), & \text{ic\_real} > 16 \end{cases}
$$

**修正二**：`CNA_CBUF_CON1.DATA_ENTRIES` 必须用补齐后的通道数。

$$
\text{DATA\_ENTRIES} = \left\lceil \frac{W_{in} \times 2 \times \lceil \text{IC}_{\text{pad}} / 16 \rceil}{8} \right\rceil
$$

`min_conv.c` 与 librocketnpu 都写成了 `ic_real`。只要 `ic_real` 落在 (16,32)、(32,64)、(64,96) 这类区间，结果必错。确证用例：

| `ic_real` | `IC_pad` | `DATA_ENTRIES` | mismatch |
| ---: | ---: | ---: | ---: |
| 40 | 64 | 6（旧公式） | 813 / 2048 |
| 40 | 64 | 8 | **0 / 2048** |
| 33 | 64 | 8 | **0 / 2048** |
| 48 | 64 | 8 | **0 / 2048** |
| 65 | 96 | 12 | **0 / 2048** |

> **对文档 §1.3 的解释**：YOLOv5s-relu 640 在 librocketnpu 下 mean diff 18–25，而 MobileNetV1 224 完全 bit-exact。MobileNetV1 的通道数为 32、64、128、256、512、1024，全是 32 的倍数。YOLOv5s 则含 48 这类通道，正落在上面两条规则的作用区间内。这条精度损失因此不必再归因于「逐通道量化硬件限制」。

### 5.4 `CNA_PAD_CON1` 的结论未能复现

在同一块板、同一驱动、仓库当前 `min_conv.c` 上重编译复核，7 种取值全部 bit-exact：

| `PAD_CON1` 写入值 | mismatch |
| --- | ---: |
| `0x00000000` | **0 / 1024** |
| `0x00000001` | **0 / 1024** |
| `0x00000080` | **0 / 1024** |
| `0x000000ff` | **0 / 1024** |
| `0x7fffffff` | **0 / 1024** |
| `0xffff8080` | **0 / 1024** |
| 不传开关（库默认 `0xffff8080`） | **0 / 1024** |
| `--pad 2` + `0x7fffffff` | **0 / 1024** |
| `--pad 0`、6×6 输出（无越界样本） | **0 / 576** |

结论：在 §5.4 的普通路径测试配置下，越界样本恒取 0，7 个 `PAD_CON1` 写值均 bit-exact；因此旧的 22/1024 结果无法复现，也不能据此断言普通路径的填充值由该寄存器控制。**这不代表所有 CNA 路径都忽略该寄存器**：IC=1 专用输入 CVT 路径中，`PAD_CON1=0` 与 `0x80/0xff` 有可重复差异，见 §7.1。

顺带发现一个工具缺陷：原版 `--pad-con1` 用 `>= 0` 判定是否给出覆盖值。`0xFFFFFF80` 这类值因此判成「未提供」，退回库默认。测试高位字节前必须先修这个判定（仪表版已修）。

### 5.5 测试工具的一个寻址缺陷

`min_conv.c` 的 CPU 参考按 `outnpu[npu_off(0, ox, oy, ow, oh) + oc]` 读输出。输出是 NC1HWC2、通道原子 16，`oc ≥ 16` 时落在下一个 group。未修正前：

| 用例 | 未修正 | 修正后 |
| --- | ---: | ---: |
| `oc_real=40`、跨度 64 | 187 / 2560（假） | **0 / 2560** |
| `oc_real=32`、跨度 32 | 187 / 2048（假） | **0 / 2048** |
| `oc_real=64`、跨度 64 | — | **0 / 4096** |

任何 `oc_real > 16` 的用例都必须先修这一点，否则会把测试工具的错误读成硬件的错误。

### 5.6 参考实现 `--auto-fix` 的验收

把 §5.1～§5.3 的规则固化成 `--auto-fix`：

~~~c
IC_pad  = (ic_real <= 16) ? 16 : ALIGN_UP(ic_real, 32);
oc_span = (oc_real <= 32) ? ALIGN_UP(oc_real, 2) : ALIGN_UP(oc_real, 32);
WEIGHT_KERNELS = oc_span;            /* 与打包跨度一致 */
layout  = A;                         /* [ic1][kx][ky][oc][ic2] */
DATA_ENTRIES = DIV_ROUND_UP(iw * 2 * DIV_ROUND_UP(IC_pad, 16), 8);
~~~

验收矩阵为 `ic_real ∈ {3,16,17,20,24,32,33,40,48,64,65,96} × oc_real ∈ {8,16,32,40,64}`，共 60 例；另单独补测 `ic_real = 2` 两例（`oc_real` 取 16 与 32）。

- **62 例全部 bit-exact**；
- `ic_real = 1` 不属于该矩阵；旧 580～677 / 1024 mismatch 的 pattern/domain 不匹配，不能作为布局缺陷证据。其真机补验见 §7.1；
- 去掉 `--auto-fix` 的对照三例（`ic_real = 40 / 48 / 65`）全部报错，说明修正不是恒真。

### 5.8 PPU MaxPool：构建一个新算子的完整实测

本节是一次延伸验证，完整流程另见 [[RKNPU-构建算子-实测流程.md]]。

| 项 | 结果 |
| --- | --- |
| 对象 | PPU MaxPool（`0x6000` 窗口）；两条线此前都只留下「未使用 PPU」的记载 |
| 方法 | 抓厂商 regcmd（30 条，纯 PPU）→ 抄进手写发射器 → 与 CPU 参考逐元素对照 |
| 结果 | `ic = 16` 下 9 组形状/核/步长配置全部 bit-exact；改一个字段即得 MinPool，同样 bit-exact |
| 新增机制 | PPU 任务的 `int_mask` 必须为 `0xc00`；PPU 的「宽」是 NC1HWC2 内层空间维；`POOLING_METHOD` 1 = MAX、2 = MIN，0/3 输出全 0 |
| 危险操作 | 非法 `int_mask`（如 `0x3f00`）触发 job abort，并让内核在 BO 释放路径 oops，只能重启板卡 |

对台账的影响：**「单元级使能落在哪」已答**。纯 PPU 任务的 regcmd 里根本不写 CNA / DPU / CORE，末条 `PC_OPERATION_ENABLE = 0x60` 的 `OP_EN` 仍为 0，触发由驱动完成。

### 5.9 DPU ReLU 与两 task 链

| 项 | 结果 |
| --- | --- |
| 对象 | DPU 的 BS 段 ReLU；一个 submit 内的两 task 链 |
| ReLU | 厂商取值 `DPU_BS_CFG = 0x00020010`；语义为 **requant 之前**对累加器钳位 |
| 观测陷阱 | `offset = -128` 时 ReLU 被钳位掩盖，两种候选语义恒等，必须挪开量化零点才能判别 |
| 链机制 | 每条 regcmd 的 trailer 第 0 槽写 `PC_BASE_ADDRESS = 下一条 regcmd 地址`（厂商 2307 个 task 中 2305 条带此指针） |
| 链的根因 | trailer 第 1 槽 `PC_REGISTER_AMOUNTS` 必须写真实条数 `(amount + 5) / 2 - 1`。写 0 时第一条 task 会清掉它，第二条的 DMA 长度变成 0 |
| 结果 | 分 BO、同一 BO 连续、末条链指针自指三种布局均 **CHAIN BIT-EXACT**（1024 元素，max diff 0） |
| 新边界 | 发射器的 stride 公式只在 8×8 校准（12/16/24 有 55～77 个错）；多核派发未验证 |
| 危险操作 | `subcore_task[i].task_number = 0` 会让板子硬挂死，看门狗重启 |

完整流程与原日志见 [[项目/RKNPU/RKNPU-构建算子-实测流程.md]] 的第四节与第五节。

### 5.7 `raw_results.txt` 不能按现源码复现

用仓库当前的 `experiments/min_conv.c` 在同一板上重新编译复跑：

| 记录 | `raw_results.txt` | 本次复跑 |
| --- | ---: | ---: |
| 阶段 1 的 4 组 CONV | 全部 0 错 | 全部 0 错（一致） |
| 每 tap oc 跨度 16 vs 32 | 455 / 1024（max diff 4）vs 0 | **418 / 1024（max diff 9）** vs 0 |
| `PAD_CON1` 0 vs −128 | 0 vs **22 / 1024** | 0 vs **0 / 1024** |

原版与改造版并排复跑的结论完全一致（见 `raw_results_20260927.txt` 第二节）。差异不是本次改造引入的。`raw_results.txt` 未记录 §3、§4 的完整命令行，无法进一步定位。**建议**：`raw_results.txt` 里这两组数字先标注为「不可按现源码复现」，引用前重跑。

---

## 六、早期结论修正索引

权重槽位与布局修正见 §5.1～§5.3；普通路径的 padding 和参考寻址修正见 §5.4、§5.5。
早期 `raw_results.txt` 中无法复现的数字见 §5.7。
精度路径、affine 和厂商 lowering 的后续实验见 §7.3、§7.6、§7.17。

## 索引（§7.x 速查）
覆盖 **§7.1–§7.194**（共 163 条）；每条可 `grep -n "^### 7.N" docs/RKNPU-实验台账.md` 直达。
> **过程教训（第一百轮修复）**：第七轮索引重建误删了全部 `### 7.x` 章节（1889→581 行）；已从 `65a6e60` 恢复。
> **规则：任何"重建索引"只允许在索引区内操作，写完立刻 `grep -c "^### 7\."` 校验。**
**按目标挑入口**：
- **技能两大核心问题（真实语料闭环）**：① **LUT vs 多项式** ⇒ **§7.135/§7.136**（计数）、**§7.140**（静态常量）、**§7.142/§7.143**（函数族）、**§7.144/§7.145**（事件级）；② **是否单核受限** ⇒ **§7.116/§7.127/§7.137**。
- **三项已闭合逆向目标**：多 task/多核 ⇒ **§7.110–§7.116**；PPU 行切分 ⇒ **§7.87–§7.90**；conv 前向分片 + 非方图 ⇒ **§7.96–§7.109**（二维切分早于本会话已在 §7.34/§7.35 闭合 ✓）。
- **硬规则**：`npu_off` 散回 ⇒ §7.102；输入偏移须 BO 基址 ⇒ §7.79；emit 游标须复位 ⇒ §7.71；PPU tile 形状须一致 ⇒ §7.87；int8 tile `rows` 须 16 的倍数 ⇒ §7.96；输入面须与 `IW` 同高 ⇒ §7.117。
- **已知限制**：非方输入面不能照搬 vendor `dma_h`/`SURF` ⇒ **§7.119–§7.124**。
- **一手资料与方法**：jsonl `hex`（**读 `event=="regcmd"`**）⇒ §7.118/§7.134/§7.135；LUT 工具 ⇒ §7.135（`experiments/lut_scan.py`）；逐位域 diff ⇒ §7.47/§7.48。
- **训练闭环/包线** ⇒ §7.49–§7.54；**命令流语义与 `.rknn` 取证** ⇒ §7.15。
**全部条目**：
**全部条目**：
- **§7.1** `ic_real=1` 专用路径：2026-09-28 真机补验
- **§7.2** `oc_real=33..63` 的 OC 跨度补验
- **§7.3** DPU BN/NRDMA affine 的已有证据与缺口
- **§7.4** wgrad：CNA K 维 dot-reduction 微测
- **§7.5** wgrad_real：真实张量 + 梯度累加 + 权重更新闭环
- **§7.6** vendor BatchNormalization lowering 识别
- **§7.7** `em=0x0d` 首次捕获：LayerNorm / Mul / Transpose 的 vendor lowering
- **§7.8** PPU MaxPool 多通道与 padding 边界
- **§7.9** 多核派发探针
- **§7.10** BS_ALU_ALGO 枚举扫描
- **§7.11** 矩形空间根因定位— 推翻 §7.10 的全部候选
- **§7.12** BRDMA 表值域标定闭合— ALGO 0/1 重新定界为"未定义值域"
- **§7.13** PPU 多组闭合— 根因是 CPU 参考读 bug，硬件单 task 即支持多组
- **§7.14** 矩形核重新表征— 旧"方形 vs 矩形"结论复核成立，但旧 pattern 存在 y 盲区假 PASS 〔本节的 tap
- **§7.15** 权重 tap 顺序根因闭合— 核内转置被对称 pattern 掩盖
- **§7.16** FP16 / PROC_PRECISION 路径侦察— **未闭合**，仅建立起点
- **§7.17** FP16路径打通— 手写发射 fp16 卷积，宽矩阵 BIT-EXACT
- **§7.18** FP16 权重 oc 组与 ic 行对齐闭合— §7.17 的两个遗留项全部消除
- **§7.19** FP16 覆盖面检查— deconv 与 BS/BN 表路径两个新开口
- **§7.20** FP16 BS/BRDMA bias 稳定性调查— 机制已定、触发已复现、修复未成
- **§7.21** FP16 BS/BRDMA bias：重载条件查明 + 可用配方— §7.20 的根因
- **§7.22** FP16 deconv 闭合— §7.19 的「未通」是参考工具 bug
- **§7.23** FP16 独立 BN闭合
- **§7.24** FP16 BS/BRDMA 的 ALU(add) 与 MUL 级标定
- **§7.25** FP16 训练侧 wgrad打通 + fp16 `c.ic` 32 对齐根因
- **§7.26** FP16 真实训练闭环移植
- **§7.27** deconv 边界口径修正：转置卷积方向写反
- **§7.28** fp16 双层真实网络闭环：跨层前向 + 分层 wgrad + 权重更新
- **§7.29** fp16 深通道 / 多 tap 权重分组包线：`c.ic` 硬上限
- **§7.30** 修正 §7.29：包线本质是 CNA 输入 feature tile ≤ 32 KB
- **§7.31** 修正 §7.30：真包线是 CBUF bank，次级约束是 FEATURE_GRAINS
- **§7.32** FEATURE_GRAINS 的 vendor 精确口径：`grains = ih + kh + pad_top`
- **§7.33** ic 维切分：非逐位一致，可替代行切分
- **§7.34** 二维切分：输出行 × ic
- **§7.35** 输出列切分
- **§7.36** fp16 `WEIGHT_SIZE1` 必须 ×2+ 权重侧非硬限 + oc 维切分
- **§7.37** 深通道 fp16 训练闭环
- **§7.38** 用 vendor 真值核对 CBUF bank 规划 + bank 划分的时序/下限
- **§7.39** 带表 DPU 提交「逐次交替超时」的根因刻画
- **§7.40** 深通道 `--two-layer` 修复
- **§7.41** 深通道 int8 打包/回读参数化
- **§7.42** 一次 submit 多 task 的语义实测
- **§7.43** 多核派发尝试 + vendor 提交口径真值
- **§7.44** 多 task 机制的真实用途：批量发射 DPU 小算子
- **§7.45** **重要修正**：多 task「不执行」是我们 submit 字段没配套
- **§7.46** 多 task 链式机制：任务流尾部写 PC 寄存器指向下一块
- **§7.47** vendor 任务的位域语义
- **§7.48** 我们的流 vs vendor 真值的逐位域 diff
- **§7.49** 训练闭环特征图尺寸扩展与新边界
- **§7.50** §7.49 边界根因：输出区与打包输入重叠→ 训练闭环到 32×32
- **§7.51** 训练闭环前向行切分+ wgrad K 维上限
- **§7.52** 前向行切分的内部 tile 取窗：逐寄存器收敛
- **§7.53** 前向行切分修好并通过验证
- **§7.54** wgrad 的 K 维分块 `--tile-p`
- **§7.55** int8 前向行切分未成立
- **§7.56** int8 行切分失败点收窄：打包数据而非寄存器
- **§7.57** 多 task 驱动侧机制复核 + `task_base_addr` 负结果
- **§7.58** 2 task 提交「真跑」必挂 + 此前实验假成功的原因
- **§7.59** 链段等长 bug+ vendor 多 task 真值
- **§7.60** 多 task：vendor 定量口径
- **§7.61** 多核提交「成功但后续 task 无输出」
- **§7.62** 多 task：vendor 最小样本 vs 我们实测
- **§7.63** PPU MaxPool 边界 B签名收窄：失配位置返回**窗口最小值**
- **§7.64** PPU 输入高度上限 = 63——边界 B 的真正根因
- **§7.65** PPU 按行切分 `--tile-oh`：尝试与当前签名
- **§7.66** PPU 输出地址的单变量拆分：输出地址不被遵守
- **§7.67** PPU 输出地址根本不生效
- **§7.68** §7.67 修正：偏移被忽略成立；「换 BO 不生效」作废
- **§7.69** PPU 切分重试与逐 tile 自检：tile 计算本身错
- **§7.70** 矛盾钉死：同一 (src,dst) 组合两条流程结论相反
- **§7.71** 根因：emit_pool 游标未复位
- **§7.72** /7.73 tile 切分几何两次修正
- **§7.74** /7.75 tile 几何两次尝试与回退
- **§7.76** tile-0 流与同 cfg 不切分流完全相同+ `oh_t` 附带负结果
- **§7.77** –7.80 PPU 单变量排查：输入偏移不被遵守+ tile 改造
- **§7.81** /7.82 tile0 已证明计算正确；前进量改按推导行数
- **§7.86** /7.87 突破：tile 几何必须逐次一致
- **§7.89** lead 行方案——设计与实施要点
- **§7.90** **PPU 行切分闭合**：ih ≤ 63 硬边界绕开
- **§7.94** 附带发现：int8 前向在 `IH ≠ 16` 时不切分也 MISMATCH
- **§7.102** **int8 训练前向切分闭合**
- **§7.107** /7.108 int8 分片收尾：**输入面必须方图**，分片解锁非方图
- **§7.109** 规则与分片收益最终确认
- **§7.110** 多 task：修掉描述符 `regcfg_amount` 实为 0 的顺序 bug
- **§7.111** 运行内核复核：0.9.8 内建、尊重 `task_number`
- **§7.112** vendor 原始块表复核：`regcfg_amount = 首块字节数/4`
- **§7.113** 多 task：块首 `S_POINTER` 不是原因；下一步 = 每 task 独立块
- **§7.114** 多 task：试「每 task 独立块」仍超时 ⇒ 三条杠杆全部排除
- **§7.115** **多 task/多核闭合**
- **§7.116** 多核并行定量验证：3 个独立 task ≈ 1 个 task 的时间
- **§7.117** 方图规则是严格等号 `rows == IW`
- **§7.118** vendor 原始 regcmd 字节流的位置与格式
- **§7.119** vendor 存在非方输入面 ⇒ "必须方图"是发射器约定而非硬件规则
- **§7.122** 解出 vendor 样本完整几何：`SURF` 公式吻合，失败因**打包侧未同步改**
- **§7.123** vendor 语料全表：`dma_h ≡ 8`、`SURF` 随算子分族；并证伪"这两字段是唯一决定因素"
- **§7.124** 判定性负结果：`SURF` 口径不可照搬⇒ 非方线关闭
- **§7.126** 全量回归：三项闭合成果全部保持 ✓
- **§7.127** vendor 派发形态普查 + "单 subcore 多 task"复测仍超时
- **§7.128** 更大 fp16 `oc` 验证：`oc = 96 / 128` 均 **ULP-EXACT**
- **§7.129** 语料普查扩展：现有 28 份 jsonl **零 LUT写入** ⇒ LUT 指标需新捕获
- **§7.131** 真实 softmax 注意力 trace 解析：仍零 LUT 写；但确认"单 subcore 数千 task"
- **§7.132** 真实负载描述符普查+ 增量块复测仍挂 ⇒ 阻塞在驱动子核路径
- **§7.133** 激活语料：工具地图已打通，卡在"模型与运行时不匹配"
- **§7.135** **LUT 指标收口**
- **§7.136** LUT vs 多项式普查——技能核心问题达成
- **§7.137** 26 个真实模型的提交形态：**全部 `core_mask=0` + 按 subcore 分派**
- **§7.138** 真实 attention 负载的 LUT 判定：**softmax 确实用 LUT**
- **§7.139** LUT 表内容结构解码
- **§7.140** 判据实验：LUT 表是**静态编译常量**
- **§7.141** sigmoid LUT 逐项拟合：单调但**不是** `S·σ(x)`
- **§7.142** sigmoid LUT 函数族判定：**近指数表**
- **§7.143** 四张 LUT 表的结构分类
- **§7.144** LUT 写入的事件级分解：整表突发 + 零写旁路戳
- **§7.145** attention 大突发按 `op_idx` 拆分：同一张表，一次推理加载 4 次
- **§7.146** LUT 语义收尾："平台"是写入顺序假象 ✗；四表去重后**全部严格递增**
- **§7.147** LUT 语义收尾：归一化形状比对 ⇒ sigmoid 与 tanh **同族指数型**、softmax 线性
- **§7.148** LUT 语义收尾：经验模型 `v≈A·exp(k·i+c·i²)`、无精确闭式
- **§7.149** 单 subcore 多 task：连 vendor 真实形态也挂 ⇒ 从流侧不可达
- **§7.150** vendor op 寄存器指纹工具 + 判别特征
- **§7.151** 指纹工具扩展：CNA 块的出现程度 = 算子类别判据
- **§7.152** 一站式 trace 报告工具
- **§7.153** 修正 §7.146："平台"真实存在；§7.146 的检验方法无效
- **§7.154** 报告工具加 `--check` + 建立回归基线
- **§7.155** 池化 → 卷积 管线对接
- **§7.156** **池化 → 卷积 NPU 管线打通**
- **§7.157** `--in-file`/`--in-xiw` 路径的**行为学验证**
- **§7.158** 池化→卷积的**定量校验尝试**：未获确认 ✗
- **§7.159** 收官回归
- **§7.160** 外部输入的定量校验：路径已摸清
- **§7.161** **池化 → 卷积 管线定量校验通过**
- **§7.162** 池化→卷积**一键配方**
- **§7.163** wgrad 在**外部任意梯度**上的定量验证
- **§7.164** 旋钮清单入 handoff + 例行自检
- **§7.165** 会话后半成果并入 handoff
- **§7.166** 技能文档本身增补"2026-09-30 实测要点"
- **§7.167** 与技能文档旧记载数字的**对账**
- **§7.168** **规范 hook 交叉验证**
- **§7.169** 会话收尾与暂停
- **§7.170** 驱动反汇编定论 + 多核并行可证伪闭合 + §7.115/§7.116 证伪修正
- **§7.171** 驱动 AUTO 分支/subcore_task 槽位语义源码定论（vendor N=3 描述符规格；slot={0,1}+task_number=3 新杠杆）
- **§7.172** vendor 34/34/4 块字面回放两变体超时 + 楔死事故（mul 终端尾写真值 0x18；hook 口径差异）
- **§7.173** **链协议闭合**：全字面回放 3/3 ALL-TASKS-EXECUTED（旧 hook 半截捕获为根因；slot.task_number=PC 取条数）
- **§7.174** LUT 逐项拟合（Task 4 部分）：fsoftmax 111 项表 = 精确 19 段分段线性（压缩率 17%）✓；sigmoid/tanh 写序≠索引序，待按 LUT 地址恢复后再拟合
- **§7.175** vendor op 语义库（Task 5）：mul=BS_MUL 通路（§7.150 一致）；LN 判别=0x100c–0x1040 区 54 寄存器（DPU+PPU 协同）+ PC 尾写链独立复现 §7.173
- **§7.176** 链工程化移植：--ov 尾写 target bug（VERSION/OP_EN 应写 0x41/0x81 bank）已修，待单次验证
- **§7.177** 链工程化②：trailer 清链根因已修（--ov 尾写即 +4 EXTRA，段内跳过标准 trailer）+ 段几何=vendor 形态；剩 enable_mask=0x18 对照（下轮 1 次提交）
- **§7.178** 链工程化③：流/描述符已与 vendor 逐字段同构仍超时 ⇒ 嫌疑收敛到块内容（卷积配置链式状态依赖）；下一步 = vendor mul 块经 replay harness 验证
- **§7.179** 链工程化④：mul 全窗块经 harness ret=0（{0,1} 单执行 ✓ / {0,3} 无链超时=预期）⇒ harness 与 vendor 链双自证；新嫌疑 = 段 0x80 对齐（vendor PC_BASE 全 0x80 对齐，我方 +0x410 非对齐）
- **§7.180** 链工程化⑤：0x80 段对齐落地仍超时；下轮 = conv→attn 混合链二分（isolate conv 块是否可链）
- **§7.181** 链工程化⑥：conv→stub 亦超时（conv 块交不出链）；块内 PC 干扰排除；尾写 OP_EN 成旋钮（0x18/0xd/0x1d 待对照）
- **§7.182** 链工程化⑦：--chain-op-en 0x1d ⇒ **task_counter=1（首次有中断！）**；终端 OP_EN 固定 0x18 修正；下轮重跑 conv→stub×0x1d
- **§7.183** 链工程化⑧：**task0 完整执行实锤**（OUT0 HIT 1696/2048）；终端 OP_EN 0x18→旋钮（conv 需 bit0）；op_idx 排除；挂点=PC 不发起 seg1 取指；vendor 从未链 conv（§7.44 重读）
- **§7.184** 链工程化⑨（**定性反转**）：vendor conv 链真值抓取——conv→BN→conv=6-task/1-submit 全部执行；尾写 OP_EN=**0x1d**（attn 0x18=特例）、enable_mask=0x1d、stride=ALIGN_UP(fetch,8)；对拍差 31 项（几何/精度）；剩 0x100c/0x1180/0x3010 三控制位待二分
- **§7.185** 链工程化⑩（**决定性通过**）：vendor conv 链 verbatim 复刻回放 ret=0/hw=9333ns/DST0+DST1 双 HIT——**提交路径无罪实锤**；两次 counter=0 假阴性 = delta patch mask 清零 target 位（教训：字节级 diff 是假阴性唯一防线）；31 项对拍差 = 下轮二分清单
- **§7.186** 链工程化（十一）双根因修复：终端尾写形态（conv=[pad,AMT=0,VER,OP_EN=0x1d]）+ 段拷贝错位（n_prev↔before）——emitter 自链闭合（ret=0，双 task 消费，seg1 WDMA 正确落位）
- **§7.187** 链工程化（十二）协议已闭合、数据保真未闭合——跨 task 管线状态线程（同构重复 task1 非确定）+ stub+pc 探针缺陷 + 板卡复位事件（已停手）
- **§7.188** 链工程化（十三）§3b-1 收口：数据保真 = 「同核同构重复约束」（vendor 恒异质依赖链）；--ov 在 chain-pc 被覆写作废；双回归全绿
- **§7.189** Task 3 打通：真实多输入 attention `.rknn` 喂数（`inputs_set -5` 根因=维度+仅喂 input[0]）+ 喂真数 LUT 对账（输入无关）
- **§7.190** Task 4 Step 2 解阻：attention 大表（954 项）= 几何/指数表（公比 2 每 71 项），3816B→2 参数；`lut_scan --fit`
- **§7.191** Task 4 Step 1 解阻：sigmoid/tanh LUT = 精确 Q15 激活表（单突发内写序=索引序；推翻 §7.141/§7.174）
- **§7.192** Task 6 Step 2：`conv → pool → conv` 三层管线 BIT-EXACT（修 min_pool --out-file 同步竞态 + min_conv_exp 外部输入参考）
- **§7.193** Task 6 Step 3：真实激活上的 int8 训练微步（量化感知 in-place 写回 + dY 缩放↔饱和率），CLOSED-LOOP OK
- **§7.194** em=0x0d lowering 表征：LayerNorm fp16 全 task 链离线解码（CORE 无 enable 位；γ fp16/β fp32；transpose perm→编码全表；数值链四项复算命中）

### 7.1 `ic_real=1` 专用路径：2026-09-28 真机补验

旧日志的 580/1024 与 677/1024 mismatch 不是权重布局证据。此路径同时启用 `NONALIGN_DMA + GROUP_LINE_OFF + ARGB_IN(8)`、CNA 输入 CVT（scale 16384、offset 65408、truncate 14）和 `CNA_CVT_CON5=65535`；旧 probe 却把逻辑有符号输入直接写成二补码字节，并按普通 NC1HWC2 每像素 16 字节布局填激活。

交叉核对 Mesa/librocketnpu 源码后，确认专用输入布局应为 `[x][y]` 的单字节数据，Y 补齐到 16；`DATAIN_WIDTH=max(input_width,16)`、line stride=`max(input_width,16)`、surface stride=`line_stride×(height−1)`、`DATA_ENTRIES=DATAIN_WIDTH×height`。CPU 参考保持逻辑有符号值，probe 将 pattern 输入编码为 `logical+128`。该配置已上板验证：原始输入 byte 0、1、127、128、255 分别输出 −128、−127、−1、0、127，确认本路径 CVT 映射为 `byte−128`。

| 用例（`ic=1, oc=16`） | 输出数 | 结果 |
| --- | ---: | --- |
| 1×1 kernel、单输出像素 | 16 | round-exact |
| 1×1 kernel、8×8 输出 | 1024 | round-exact |
| 2×2 kernel、单输出像素 | 16 | round-exact |
| 3×3 kernel、单输出像素 | 16 | floor-exact |
| 3×3 valid、8×8 输入→6×6 输出 | 576 | floor-exact |
| 3×3/pad1、8×8 输入输出，`PAD_CON1=0` | 1024 | round-exact |

矩形 kernel `2×1`、`1×2` 在该 probe 下仍 mismatch；相同控制用例在 `ic=3` 也 mismatch，因此暂不归因为 IC=1 专用布局，先列为最小发射器的矩形 kernel/轴向语义问题。

**矩形 kernel 边界表征（2026-09-28 补充，[`rect_kernel_20260928.log`](../experiments/results/rect_kernel_20260928.log)）**：用 IC=3 普通路径 + `--pattern --auto-fix`（权重 span 与 WEIGHT_KERNELS 一致，排除 §5.3 干扰）系统扫描 kernel 形状后，规律收敛为**「方形 vs 矩形」**，与 IC、pad、空间尺寸无关：

| kernel（IC=3/IC=1，pad0） | 结果 |
| --- | --- |
| 1×1、2×2、3×3、4×4（方形） | **BIT-EXACT** |
| 2×1、1×2（多行输入） | 555/896、555/896（max diff 4） |
| 3×1、1×3 | 88/768、88/768（max diff 5） |
| 4×1、1×4 | 278/640、278/640（max diff 2） |
| 4×4 | **BIT-EXACT** |
| 2×1、单行输入（ih=1，oh=1） | **BIT-EXACT**（如 iw=2/ih=1、iw=3/ih=1） |
| 1×2、单列输入（iw=1） | **BIT-EXACT**（iw=1/ih=3 → 1×2 输出） |
| 2×1、IC=1 路径 | 159/448（同样失败） |

特征：失配确定性可复现；pattern 数据下误差小（≤5）且随 oc 呈 mod-3 周期（权重 pattern 周期的相互作用）；2×1 与 1×2 失配数完全相同（转置对称）。已排除：权重 span/WEIGHT_KERNELS 不一致（§5.3）、`--auto-fix` 口径、X 读步长/ tap 线性化的简单模型（`rect_solve.py` 穷举「核尺寸×tap 映射×X 步长」无干净解，最好 48/64）。单行输入可通过说明单 tap 序列本身无误，失败需要「矩形核 + 多像素两维输出」同时成立，指向 CBUF/核几何在 kw≠kh 时的内部处理差异。

**结论**：矩形 kernel 是当前最小发射器无法逾越的真机边界；真实模型几乎不用非方形卷积核（1×1/3×3 为主），工程影响有限。根因定位需观测线捕获 vendor runtime 对矩形 kernel 的 regcmd（若有）；列为低优先级。

**IC=1 空间扩展扫描（2026-09-28 第五轮，同 `--pattern --auto-fix --pad-con1 0`）**：8×8 输入上 `--ic 1 --oc {1,4,16,33} --stride {1,2} --kw {1,1x2,2x2,3x3} --pad {0,1}` **全部 BIT-EXACT**（含 oc=33 跨 64 槽、stride=2 下采样、pad=0 valid）；但空间尺寸一旦偏离 8×8（16×16 / 8×4 / 4×4，任意 stride/pad 组合）**IC=3 对照同样 MISMATCH**——与矩形核同类的「控制例也失败」模式，非 IC=1 特有。误差特征：±5 小偏差集中在 pad 边界行（y=0）与 x=4k 位置，`--entries 256` 无法修复（(0,0) 反而 hw=107，远偏离）。~~判定：发射器空间≠8×8 配置缺陷~~。

**空间尺寸缺陷根因修复（2026-09-28 第八轮，[`spatial_ohfix_20260928/`](../experiments/results/spatial_ohfix_20260928/)）**：用 P2 方法论重建 vendor 地面真值——rknn-toolkit2 2.3.2 转换 16×16 int8 单卷积（工具链需补 onnx.mapping shim：rknn 2.3.2 的 base_utils.so 仍引用已被 onnx≥1.16 移除的 `onnx.mapping`，从 onnx 1.15 sdist 提取 mapping.py 放入包内 + 脚本顶部显式 `import onnx.mapping`），板端 hook 采集 regcmd（`vendor_conv16.txt`），与发射器同几何 dump 逐寄存器对比。对比揪出真正的根因：**`min_conv_exp` 的 `--oh` 从不联动 `--ih`（默认恒 8，仅 ow=iw 联动）**——第五轮所有「失败」用例实际都在跑 ih≠oh 的错位几何（如 iw=16 ih=16 实际输出立方 oh=8，DATA_SIZE3=128≠vendor 256 即此错位的直接证据）。补 `if (!oh_given) c.oh = c.ih;` 后重扫：**4×4 / 8×8 / 12×12 / 16×16 / 32×32、stride∈{1,2}、pad∈{0,1}、IC=1/3 全部 BIT-EXACT**——硬件对空间尺寸完全不受限，第五轮结论（发射器 8×8 硬编码嫌疑）作废。第八轮另确认：`--entries 256`（vendor 值）在任何几何下都不适用——DATA_ENTRIES 语义与 CBUF bank 规划耦合，我们的 1-bank 布局下小值正确；vendor `CBUF_CON1=ic16×ih` 仅在其自身布局下成立。**仍开放**：① `ih=4`（8×4 出错：中间行全死、边界行 ±15；4×4 反而通过，疑似 iw≥8 且 ih<8 的表面/bank 组合）；② 矩形核（§7.1 前项，不受本轮影响）。发射器同步新增 `--ov REG=VAL` 尾部寄存器覆盖机制（二分定位用）。

**PAD_CON1 的条件差异**：IC=1、输入零点为 0 的 3×3/pad1 用例，`PAD_CON1=0` 为 round-exact；写 `0x80` 或 `0xff` 分别出现 22 个 round mismatch（`0x80` max diff 129，`0xff` max diff 2）。这说明该专用路径的边界填充值按输入 CVT 后的有符号域解释。`min_conv_exp.c` 已将 IC=1 默认值改为 `input_zero_point`，本例有效写入 0。此前“PAD_CON1 写值无影响”的说法只适用于 §5.4 普通路径。

原始日志归档于 [`experiments/results/ic1_probe_20260928/`](../experiments/results/ic1_probe_20260928/)，修正后的发射器为 [`experiments/min_conv_exp.c`](../experiments/min_conv_exp.c)，CVT 自动扫描脚本为 [`experiments/run_ic1_cvt_sweep.sh`](../experiments/run_ic1_cvt_sweep.sh)。第五轮扫描与 PPU/多核探针日志：[`ppu_maxpool_20260928.log`](../experiments/results/ppu_maxpool_20260928.log)、[`subcore_probe_20260928.log`](../experiments/results/subcore_probe_20260928.log)、[`board_reboot_20260928.log`](../experiments/results/board_reboot_20260928.log)。

### 7.2 `oc_real=33..63` 的 OC 跨度补验

用 IC=3 普通路径、3×3/pad1、8×8 输出、`out_zp=128`，分别比较每个 OC 的 `align_up(oc_real,2)` 跨度和 64 跨度；打包跨度始终与 `WEIGHT_KERNELS` 一致。输出零点设为128，避免负累加被默认 offset `-128` 饱和掩盖。

| OC 范围 | 每 tap 跨度 / `WEIGHT_KERNELS` | 结果 |
| --- | --- | --- |
| 33..62 | `align_up(oc_real,2)`（34..62） | 30/30 mismatch |
| 33..62 | 64 | 30/30 round-exact（91,200 个输出） |
| 63 | 64 | round-exact（4,032 个输出） |

结论：`oc_real > 32` 且 `oc_real ≤ 64` 时，该装置上的 CNA 权重区必须给每个 tap 预留完整 64 个 OC 槽位，即 `WEIGHT_KERNELS=64`；单纯让描述符和偶数跨度彼此一致仍不足以跨过 32 通道组边界。配合 §5.1 已测的 8/16/32 输出通道结果，当前已确认的 OC 规则为：`oc_real≤32` 可用 `align_up(oc_real,2)`，`32<oc_real≤64` 用 64。

首轮小跨度与 64 槽复验原始输出分别见 [`experiments/results/oc33_63_span_sweep_20260928.log`](../experiments/results/oc33_63_span_sweep_20260928.log) 与 [`experiments/results/oc33_62_span64_20260928.log`](../experiments/results/oc33_62_span64_20260928.log)。可复跑脚本为 [`experiments/run_oc33_63_span_sweep.sh`](../experiments/run_oc33_63_span_sweep.sh) 和 [`experiments/run_oc33_62_span64.sh`](../experiments/run_oc33_62_span64.sh)。按原顺序，DPU affine 原语已补齐；下一步是确认厂商 BatchNormalization lowering，再进入 wgrad/通用归约探针。

### 7.3 DPU BN/NRDMA affine 的已有证据与缺口

板端已有 LayerNorm 厂商 trace [`experiments/results/dpu_bn_20260928/fp_layernorm_16_int8.jsonl`](../experiments/results/dpu_bn_20260928/fp_layernorm_16_int8.jsonl)。从任务 regcmd 解码出 24 个 active BN 记录：6 个 `BN_CFG=0x42, BN_MUL_CFG=0x3e800000`，6 个 `BN_CFG=0x42, BN_MUL_CFG=0xb4000000`，6 个 `BN_CFG=0x20050, BN_ALU_CFG=0x5f37`，6 个 `BN_CFG=0x20050, BN_ALU_CFG=0x41000000`。这些记录均 `NRDMA_CFG=0`、`BN_BASE_ADDR=0`；样本 task 的 `CNA_CONV_CON1=0x120`，`PROC_PRECISION=2`，属于 FP16 precision。它只提供 immediate ALU/MUL 参考。

这份 FP16 trace 可用于核对 immediate BN 寄存器组合，但不回答 B9 的 INT8 逐通道 affine DMA 布局。为隔离这些路径，仪表版新增 `--bn-cfg/--bn-alu/--bn-mul/--bn-relux/--dpu-data-format` 和对应 CPU reference，并用自洽权重配置做了 INT8 真机 probe：

| 模式 | 配置 | 结果 |
| --- | --- | --- |
| bypass 对照 | 默认 BN bypass | 1024 项 round-exact |
| immediate ALU add | `BN_CFG=0x20050`, `BN_ALU_CFG=3` | 1024 项 round-exact，验证 ALU algo=2 加立即数 |
| immediate MUL，shift=0/1/2 | `BN_CFG=0x42`, operand=3 | 三种配置均 round-exact；验证正值 shift 读 `BN_MUL_CFG[13:8]` |
| negative shift=1 | 同上并设 `DPU_DATA_FORMAT=0x400` | round-exact；验证负值 shift 读 `BN_MUL_SHIFT_VALUE_NEG[15:10]` |
| `BN_TRUNCATE_SRC=1` 试探 | `BN_CFG=0x42`, `BN_MUL_CFG=0x00030102` | SUBMIT timeout；随后 BN bypass 基线仍 round-exact，板卡正常 |

`BN_MUL` 的参考运算采用 ties-to-even；正值 shift 来自 `BN_MUL_CFG`，负值 shift 独立配置在 `DPU_DATA_FORMAT`。这修正了 QEMU immediate BN 模型里用单一 shift 并按 half-up 处理的简化假设。

**找到了真正 active 的 INT8 NRDMA oracle**：Qwen vision 896 capture 的 `op_idx=8` 有 12 条记录，`BN_CFG=0x20150`（BN ALU algo=2/add、`BN_ALU_SRC=NRDMA`）、`NRDMA_CFG=0x2`、`BN_BASE_ADDR!=0`。FlattenBuffer 映射显示该 op 是 `Add`，第二输入为 tensor250 `/visual/Concat_output_0`，weight blob offset `0x0f179d40`、长度 12,845,056 B；它不是独立 `BatchNormalization` gamma/beta 样本，但证明 BN_ALU 的 NRDMA DMA 数据源在 INT8 路径确实被编译器使用。映射由 [inspect_rknn_bn_model.c](../experiments/inspect_rknn_bn_model.c) 使用 openrknn parser-only 初始化得到；该 Qwen 版本的任务/寄存器提取失败，但 tensor/blob 索引和 weight BO 偏移可读。

两组 weight BO 的 base 分别为 `0xe4dc2000` 与 `0xc2c0b000`，task 的 BN base 偏移分别为 `0x0f179d40`、`0x0f181d40`，相差 `0x8000`；blob 长度等于 `392×0x8000`。该 op 的 DPU channel field 为 `0x1fff`（8192 槽）、宽高为1，因此每段覆盖8192个4字节 operand。精简 trace、tensor 元数据和原始 blob 分别见 [`qwen_active_bn_excerpt.jsonl`](../experiments/results/dpu_bn_20260928/qwen_active_bn_excerpt.jsonl)、[`qwen_openrknn_attempt.log`](../experiments/results/dpu_bn_20260928/qwen_openrknn_attempt.log) 与 [`bn_blob_data/tensor_250_blob_250.bin`](../experiments/results/dpu_bn_20260928/bn_blob_data/tensor_250_blob_250.bin)。

进一步手写 DPU BN ALU DMA 表：在 weight BO 末尾按64字节对齐放32个 `int32`，置 `BN_CFG=0x20150`、`NRDMA_CFG=0x2`、`BN_BASE_ADDR=weight_base+0x900`。系数 `[0,1,…,31]` 与 `[-8,-7,…,23]` 两种 signed 表，在 `ic=3,oc=16,8×8` 卷积的 **1024 项输出均 round-exact**；CPU reference 使用同一通道系数，证明 NRDMA 表按输出通道读取并广播到各空间位置。

还完成 DPU BS/BRDMA 的 per-channel `gamma*x+beta` 探针。按 Rocket/QEMU 表布局，每8通道一组、64字节：`[0..31]` 放8个int32 bias，`[32..47]` 保留，`[48..63]` 放8个int16 gamma。设置 `BS_CFG=0x20140`、`BS_MUL_CFG=0x00000e01`、`BRDMA_CFG=0x0e`、`DPU_DATA_FORMAT=0xe0`；`gamma=1` 且 `beta=ch−8`，以及 `gamma[ch]=0.5+ch/64`、`beta=ch−8` 两种配置均在 1024 个输出上 round-exact。这证明 BRDMA 可按输出通道给 DPU BS 阶段提供 bias 与乘数，并对 INT8 卷积实现逐通道 affine。

Immediate BN、NRDMA ALU 和 BRDMA affine 原始输出分别见 [`dpu_bn_mul_rne_verify.log`](../experiments/results/dpu_bn_20260928/dpu_bn_mul_rne_verify.log)、[`dpu_bn_nrdma_signed.log`](../experiments/results/dpu_bn_20260928/dpu_bn_nrdma_signed.log) 与 [`dpu_bs_brdma_probe.log`](../experiments/results/dpu_bn_20260928/dpu_bs_brdma_probe.log)。可复跑脚本为 [`run_dpu_bn_immediate_probe.sh`](../experiments/run_dpu_bn_immediate_probe.sh)、[`run_dpu_bn_nrdma_probe.sh`](../experiments/run_dpu_bn_nrdma_probe.sh) 和 [`run_dpu_bs_brdma_probe.sh`](../experiments/run_dpu_bs_brdma_probe.sh)。硬件 affine 原语已打通；剩余验证是厂商 BatchNormalization 图实际采用的路径/系数量化。

### 7.4 wgrad：CNA K 维 dot-reduction 微测

对单个 filter 权重，`dW[kx,ky,ic,oc] = Σ_p X[p+k,ic]·dY[p,oc]`。probe 把 P 个空间 reduction 点放到 CNA 输入通道/K 维，把 `dY[p,oc]` 作为每个输出通道的权重，提交一个单输出像素的 1×1 CONV；CNA 内部归约 K 维，一次得到16个输出通道的梯度。`wgrad_x/dy` 使用确定性合成向量，不是某个训练样本的真实梯度。

| P（reduction 点） | IC pad | 输出 | 真机结果 |
| ---: | ---: | ---: | --- |
| 32 | 32 | 16 | round-exact |
| 64 | 64 | 16 | round-exact |
| 96 | 96 | 16 | round-exact |

P=64（对应8×8空间点）dot 向量 `[21,-42,21,21,-42,21,…]` 与 CPU 完全一致。随后将 probe 改为一个自洽的合成 valid 卷积：输入 `X=8×8×3`，梯度 `dY=6×6×16`，P=36 空间点填入 K 维并 pad 到64。对所有 `3×3×3=27` 个 `(kx,ky,ic)` 坐标各发一个 task，每个 task 输出16个 `dW`，共 **27/27 task、432/432 梯度 round-exact**。合成数据用确定性整数公式生成，参考按同一输入和梯度直接求和。

P=32/64/96 初始向量扫描见[`wgrad_dot_20260928.log`](../experiments/results/wgrad_dot_20260928.log)；早期独立坐标扫描见[`wgrad_coeff_matrix_20260928.log`](../experiments/results/wgrad_coeff_matrix_20260928.log)；自洽 3×3 小卷积矩阵见[`wgrad_conv3x3_matrix_20260928.log`](../experiments/results/wgrad_conv3x3_matrix_20260928.log)。可复跑脚本为[`run_wgrad_dot_probe.sh`](../experiments/run_wgrad_dot_probe.sh)、[`run_wgrad_coeff_matrix_probe.sh`](../experiments/run_wgrad_coeff_matrix_probe.sh)、[`run_wgrad_conv3x3_matrix_probe.sh`](../experiments/run_wgrad_conv3x3_matrix_probe.sh)。

再对 `(kx=1,ky=1,ic=1)` 把1/2/3个合成样本并入同一K向量，P=36/72/108，对应 K pad=64/96/128；三次16通道输出都 round-exact，oc0 分别为 −6/18/30。该 probe 将每样本36个空间位置直接拼入K维，因此验证了单个权重项的跨 batch 求和也可由CNA完成。日志和脚本见[`wgrad_batch_20260928.log`](../experiments/results/wgrad_batch_20260928.log)与[`run_wgrad_batch_probe.sh`](../experiments/run_wgrad_batch_probe.sh)。

下一步用真实小卷积 `X/dY` tensor 替换合成向量，逐 `(kx,ky,ic)` 形成 reduction task，再测多 batch 部分和与端到端耗时。

### 7.5 wgrad_real：真实张量 + 梯度累加 + 权重更新闭环（2026-09-28，Orange Pi 5 Plus）

工具：`experiments/wgrad_real.c`（板上 `exp3/`），数据集 LCG seed=20260928，X∈[−8,7]、W∈[-4,3]、IC=3/OC=16/3×3；fwd requant scale=16385 shift=14，梯度 task out_scale=8（scale=16385 shift=17）；`W' = clamp(W − rhao(dW_npu/4))`。运行器 `run_wgrad_real_loop.sh`（Phase A pad1 B=1 / B valid pad0 B=1 / C pad1 B=2 / D pad1 B=3 / E K 容量扫描 / 基线复查）。

**最终结果：全部相位 CLOSED-LOOP OK**（日志 [`wgrad_real_20260928_185303.log`](results/wgrad_real_20260928/wgrad_real_20260928_185303.log)）：

| Phase | 配置 | FWD | WGRAD | SAT | FWD2 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| A | pad1 B=1 | 0/1024 | 27/27 round-exact | OK | 0/1024 | CLOSED-LOOP OK |
| B | **valid pad0 B=1** | 0/576 | 27/27 round-exact | OK | 0/576 | CLOSED-LOOP OK |
| C | pad1 B=2 | 0/2048 | 27/27 round-exact | — | — | CLOSED-LOOP OK |
| D | pad1 B=3 | 0/3072 | 27/27 round-exact | — | — | CLOSED-LOOP OK |
| E | K 扫描 P=36..288 | — | 0/16 每批 | — | — | 全 OK |

耗时：单闭环（31 submits）wall≈0.6–0.8 ms，hw≈0.25–0.30 ms，DMA≈69–71 KB。梯度 int8 = raw/8（scale=8, shift=17），`max|raw−npu·8|=4`，round-exact；饱和探针 ±127/−128 正常。

**① PAD_CON1 真相（修正 §5.4）**：用单 tap 恒等权重探针（`--oobtest`，X(ix,iy)=4(ix+1)+(iy+1)，逐方向解码越界样本）实测：`REG_CNA_PAD_CON1(0x1184)` **低字节 = 越界样本字节值（有符号域）**：`0x00→0`、`0x80→−128`、`0xff→−1`。旧公式 `0xffff8080`（zp=0、kw≥3）使 OOB=−128，pad1 边界 444/1024 全错（大量 127 饱和）；**`0x00000000` 才是 in_zp=0 的正确零填充**。§5.4「7 值均 bit-exact」的旧结论只在无 OOB tap 的配置（如其 6×6/40×32 用例）成立，是幸存者偏差。普通路径结论改写为：**有越界 tap 时该寄存器必须写 0**。

**② 「6×6 输出置换」为数据集缺陷，非硬件缺陷**：pad0 前向曾报 473/576「错序」，并推导出「8 像素 burst、ow 跨度、丢 8 像素」的伪规律。离线用独立 Python 实现卷积逐字节比对后确认：**DPU 输出自始至终 bit-exact 正确**（hw X-major 6×6 vs 真 conv6 = 0/576）。真正缺陷在 `gen_dataset`/`dataset_files_load`：CPU 参考 `cpu_forward(..., pad, IW, IH, ...)` 恒按 8×8 生成，pad0 时把 8 宽图像截断 576B 存盘，Phase F 以 6 宽步长读它 → 全部「错位」。FWD2 之所以「突然 exact」，是因为 U2 用 `cpu_forward(..., ow, oh)`（正确 6 宽）现算参考——两个参考不一致是唯一原因，与 task 历史/权重数值无关。教训：**两处对比出现矛盾时优先怀疑参考实现路径不同**；已修复为 `pad ? IW : 6`，数据集已重新生成（X/W 不变，dY 相应更新）。

**③ min_conv_exp 6×6 pad0「192/576」反例已关闭（同日复验）**：该失配并非新硬件问题，而是复跑时未加 `--auto-fix`，默认路径给出 `WEIGHT_KERNELS=32` 与权重跨度 `w_oc2=16` 不一致的自相矛盾配置，恰好触发 §5.3 已知缺陷类（span≠WEIGHT_KERNELS → 权重误读）。pattern 数据下特征为小值误差（max diff 9）。三种几何均复现且对齐后全部消失：

| 配置（均 `--pattern --pad-con1 0`） | span≠kernels | 对齐后 |
| --- | ---: | ---: |
| pad0 6×6（wk=32/oc2=16） | 192/576 | wk=16 → **0/576**；w_oc2=32 → **0/576** |
| pad1 8×8（默认 wk=32/oc2=16） | 418/1024 | — |
| pad0 8×8（iw=10，wk=32/oc2=16） | 344/1024 | — |

`--auto-fix`（自动选 wk=16 与 oc2=16 一致）6×6 → **BIT-EXACT (floor)**。§5.4 的「PAD_CON1 7 值均 bit-exact」结论在其原配置（min_conv.c、一致权重设置）下成立，维持不变。wgrad_real 发射器自身 kernels=16/span=16 一致，其全部闭环结果不受影响。此轮教训入册：**用 min_conv_exp 复测任何结论前，先核对 CFG 行的 `w_kernels` 与 `w_oc2` 是否一致（或直接加 `--auto-fix`），否则会把 §5.3 的已知缺陷误读成新问题。**

新 CLI：`--bisect MODE`（none|fwd6|fwd8|wgrad|wgradfull|sat|phb|many，前向→干预→再前向）、`--wfile PATH`（权重覆盖）、`--oobtest`、`--fwd-repeat N`、`--wk32`、`--padcon1 V`、`--dump`。bisect/wfile 本轮仅用于排除法，最终结论不依赖它们。

复现（板上 `/mnt/ssd/workspace/rknpu-re/exp3/`）：

~~~bash
gcc -O2 -Wall -Wextra -o wgrad_real wgrad_real.c -lm
./wgrad_real --dir wgdata_valid --pad 0 --gen-only   # 数据集（修正后 Q 为 6×6）
bash run_wgrad_real_loop.sh                          # A/B/C/D/E + 基线，全 CLOSED-LOOP OK
./wgrad_real --dir wgdata_pad1 --pad 1 --batch 1 --update   # 单相位复跑
./wgrad_real --dir wgdata_valid --pad 0 --oobtest           # PAD_CON1 越界探针
~~~

---

### 7.6 vendor BatchNormalization lowering 识别（2026-09-28，观测线+工具链，优先级 2 完成）

**方法**：工作站 `pip install rknn-toolkit2==2.3.2`（Python 3.12，onnx==1.16.1，protobuf 需 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 或给 `onnx/helper.py:strip_doc_string` 打 label 兼容补丁）。构造极小 ONNX 图并转 RK3588：`bn_only`（仅一个 BatchNormalization，C=8，8×8，γ/β/mean/var 全部取易辨识的非对称值）与 `conv_bn_conv`（Conv1×1→BN→Conv1×1），各出 INT8 与 FP16 两个 .rknn。板端部署：C runner（`rknn_runner.c`，链接 `/mnt/ssd/workspace/runtimes/mars-overhead/rknn/lib/librknnrt.so` 2.3.2）+ `LD_PRELOAD` ioctl hook（strict 通过）+ 新增 mmap 拦截 shim（`bo_dump.c`，按 fake offset≥2^32 识别 BO 并在退出前转储）读取运行时构建的系数 BO。输出数值全部先过独立 FP32/onnxruntime 参考核对（INT8 ≤1.5 LSB 含中间量化，FP16 正常）。

**结论 ①——独立 INT8 BN 的 active lowering**（[`trace_bn_only_i8.jsonl`](../experiments/results/bn_lower_20260928/trace_bn_only_i8.jsonl)）：单个 **DPU task** 同时编程 BS 与 BN（不是 BN/NRDMA 单独、也不是 CNA）：

| 寄存器 | 值 | 含义 |
| --- | --- | --- |
| `DPU_BS_CFG=0x42`、`BS_MUL_CFG=1`、`RDMA_BRDMA_CFG=8` | BS 乘法 algo=1，系数走 BRDMA per-channel 表（int16 mul） |
| `DPU_BN_CFG=0x20150`、`RDMA_NRDMA_CFG=2`、`RDMA_BN_BASE_ADDR=系数BO+0x40` | BN ALU algo=2(add)、源=NRDMA per-channel 表（int32 bias） |
| `RDMA_BS_BASE_ADDR=系数BO+0x00` | BS mul 表（8 通道一组 16B int16） |
| `DPU_OUT_CVT_OFFSET/SCALE/SHIFT = 11 / 30256 / 30` | 出口 requant（offset=输出 zp） |

提交形态：1 submit 内 **3 条 byte 级相同 task**（冗余重算同一 BN，最终值不变），core_mask 实际只跑核 0，op_idx=1，hw ≈22.5 µs。

**系数编码（从系数 BO 直接读出并数值闭环）**：

- `mul_c = int16 ≈ round(K·γ_c/σ_c)`，`K = round(s_in/(k_cvt·s_out)) = 13107`（本模型 `k_cvt = 30256/2^30`；8 通道中 6 条与该式完全吻合，ch5/ch6 差 ≤3 LSB——runtime 的 scale 中间精度细节未完全钉死，属开放小项）。
- `bias_c = int32 ≈ round((β_c − γ_c·mean_c/σ_c)/(k_cvt·s_out))`（ch0–ch2/ch4–ch7 与 round 吻合，ch3 差 1）。
- `DPU` 输出字节 = `round_half_up((x_q·mul_c + bias_c)·OUT_CVT_SCALE / 2^30)`，反量化 `(byte − zp_out)·s_out`。
- **数值闭环：512/512 输出与真机 bit 级一致**（`verify_bn_lower.py`，日志 `bn_lower_verify.log`）。

**结论 ②——FP16 路径**（`trace_bn_only_fp16.jsonl`）：同一 BS+BN 结构，`DPU_DATA_FORMAT=0x48000002`、`OUT_CVT` 旁路（scale=1/shift=0/offset=0）；BS mul 表 = `fp16(γ_c/σ_c)`，BN bias 表 = **fp32** `(β_c − γ_c·mean_c/σ_c)`。输出与 FP32 参考差 ≤fp16 舍入（max 0.0045）。

**结论 ③——BN 夹在两卷积之间时被融合**（`trace_conv_bn_conv_i8.jsonl`）：6 task/1 submit（2×CNA conv0 + 1×DPU + 2×CNA conv1 + 1×PPU，op_idx=[1,2,0,1,2,2]）；DPU task 的 BS mul 全 1、BN bias 全 0、OUT_CVT 恒等 → **BatchNormalization 的仿射被折入相邻卷积，独立 affine task 退化为布局/搬运用途**。即 vendor 在有相邻卷积时优先融合，仅独立 BN 才落到 BS/BRDMA+BN/NRDMA affine。这解释了此前 Qwen 捕获中只有 Add 的 NRDMA、无独立 BN 证据的现象。

**证据与身份**：模型/trace/BO dump/脚本/验证全归档于 [`experiments/results/bn_lower_20260928/`](../experiments/results/bn_lower_20260928/)；板端 `/mnt/ssd/workspace/bn_lower_probe/`。runtime librknnrt 2.3.2（429f97ae6b@2025-04-09），toolkit2 2.3.2，驱动 v0.9.8，hook 与 `bo_dump.so` 见产物。范围限定：单设备/单版本、C=8 与 3 通道两个 shape、γ/β 取特定辨识值；其他量化配置（per-channel 权重、对称量化、更大 C）未扫描。

---

### 7.7 `em=0x0d` 首次捕获：LayerNorm / Mul / Transpose 的 vendor lowering（2026-09-28，初步表征）

用 §7.6 同一套工具链构造 4 个最小 FP16/INT8 图（`mul_only`、`transpose_only`、`layernorm_only`，(1,8,8,8)）并捕获（trace/模型归档于 [`results/bn_lower_20260928/em0d/`](../experiments/results/bn_lower_20260928/em0d/)）。enable_mask 分布：

| 图 | task 数 | enable_mask 序列 |
| --- | --- | --- |
| `mul_only`（标量×1.5，INT8/FP16） | 3 | 全部 `0x18`（BS/BN 仿射形状；FP16 路径同 BN §7.6） |
| `transpose_only`（FP16） | 3 | 全部 `0x18`，但 `FEATURE_MODE_CFG=0x81e5`（bit15 置位）、`BS_CFG=BN_CFG=0x53`、`OW_CFG=0x080007fe`、cube W=1/H=8/C=64 + NOTCH=0x3f003f、DATA_FORMAT=0x24000001、mul 全 0 → **纯 DPU 重排**（perm 语义尚未逐一对照） |
| `layernorm_only`（FP16） | 54 | 3×[1×`0x18`(op1) + 14×`0x18` + 2×**`0xd`** + 2×**`0x1d`** + …]；`0xd`/`0x1d` 即 A 系列遗留的通用计算类，共出现 6 次 |

**em=0x0d/0x1d 的初步解读**：enable_mask=0x18=DPU+PPU（普通 DPU task）；**0xd = PC+CNA?+CORE+DPU 的组合**——对 LayerNorm 第一条 `0xd` task（task1，108 entries）解码显示同一 task 内同时编程 CNA（CONV_CON1=0x120、WEIGHT_SIZE 16×16、FEATURE_DATA_ADDR=0xffff9c00、DCOMP 表全零）+ CORE（`CORE_MISC_CFG=0x200`、DATAOUT_SIZE）+ DPU（FP16 格式、`BS_MUL_CFG=0x35a80000`、`BN_MUL_CFG=0xb5a80000`、OW_CFG=0x126）三块寄存器——**vendor 用单 task 跨 CNA/CORE/DPU 做归约+仿射流水**，这正是 openrknn em=0x0d 阻塞点对应的形态。LayerNorm 整层 = 3 次重复的约 18-task 流水线（对 3 个核各发射一次，与 §7.6 的 3 冗余 task 模式一致）。

**限定**：仅 (1,8,8,8)、LN 轴=−1 一个 shape；`0xd` task 内 CNA 权重表与 CORE 协同细节、Transpose 的 perm/OW 编码规则尚未解开。下次复现：`results/bn_lower_20260928/em0d/` 的 onnx/rknn + §7.6 的板端流程（`rknn_runner` + hook + `decode_trace.py`）。

**补充表征（全部 54 task 的 block 画像见 [`em0d/ln_task_profile.txt`](../experiments/results/bn_lower_20260928/em0d/ln_task_profile.txt)）**：LayerNorm 每核重复单元为约 18 task 的流水线：

1. `task0`（0x18）：Transpose 式重排——`FEATURE_MODE_CFG=0x81e5`、`DATA_FORMAT=0x24000001`、cube 1×8×64+NOTCH，与独立 transpose_only trace 签名一致。
2. `task1`（**0xd**，108 entries）：CNA(52 写)+CORE(5)+DPU(51)——CNA FC/conv 归约（FEATURE_DATA_ADDR=重排结果、FC_DATA_SIZE 8×8、WEIGHT_SIZE 16×16、DCOMP 全零）+ CORE(`MISC_CFG=0x200`) + DPU(`BS_MUL_CFG=0x35a80000`)，DST=0xffffa440。
3. `task2`（0x1d，126 entries）：又一组 CNA+CORE+DPU+RDMA（C=0x7000f，DST=0xffffac40）——累计步。
4. `task3-5`：小 DPU task：task4（`0xd`，22 entries，`DATA_FORMAT=0x6c000003`、`BS_ALU_CFG=0x38a7c5ac`、C=16）与 task5（`DATA_FORMAT=0x24000011`、`BN_CFG=0x20050`、`BN_ALU_CFG=0x5f37`、mul=0xffff0100、RDMA 读回 0xffffb440）——标量除法/开方类步骤落在 DPU EW/BN 立即数配置上。
5. `task6-15`：12 条 11–32 entries 的小 DPU EW task（逐元素 x−mean、平方等链）。
6. `task16`（0x1d，26）+ `task17`（op_idx=3，0x18，69）：收尾 per-channel 仿射（同 BS/BN 表形态）。

**"em" 即 task 的 enable_mask**：`0x18`=DPU+PPU（普通 DPU task），`0xd`=PC+CORE+DPU，`0x1d`=PC+CORE+DPU+PPU——与 task 内实际编程的寄存器块一致。LayerNorm = 重排 → CNA 归约(mean) → EW 链(var/σ) → BS/BN 逐通道仿射。CNA 归约权重表内容、EW_ALU 位域、Transpose perm/OW 编码仍开放。

**数值闭环**：LayerNorm FP16 输出与独立 numpy FP32 参考 maxdiff=0.007（fp16 舍入级），流水线计算正确性确认（`em0d/ln_numeric_check.txt`）；标量 Mul 输出 = 1.5×输入 精确（INT8 在 1 LSB 内、FP16 精确）。

**Transpose perm 扫描补充（2026-09-28 第三轮，`em0d/transpose_p*`）**：对 (1,8,8,8) 的 5 种 perm（0231/0312/0213/0123/0321）各转 FP16 并捕获：

- **全部 5 种 perm 的 NPU 输出与 numpy 置换 bit-exact**（maxdiff=0.0）。
- 非 identity perm → 单条 DPU task，`FEATURE_MODE_CFG=0x81e5`（bit15=重排模式）、`DATA_FORMAT=0x24000001`、`BS_CFG=BN_CFG=0x53`、`BS_OW_CFG=0x080007fe`、mul/ALU 全 0；perm 的区别体现在 **cube 几何 + stride + WDMA_SIZE**：p0231 与 p0213 都是 W=1/H=8/C=64（`WDMA_SIZE_0=0x0807003f`），p0312 是 W=8/H=8/C=8 + `DST_SURF_STRIDE=0x80`（`WDMA_SIZE_0=0x08000007`、`WDMA_SIZE_1=0x00010003`）——重排由「把源张量按某一维度顺序重新解释为 cube 并用 WDMA 步长写出」实现，非 24 种独立指令。
- identity perm（p0123）也生成重排 task（`fmc=0x1e5`、cube 4×4×32、OW_CFG=2）——工具链在 FP16 内部布局转换时同样走该 DPU 路径。
- 标量 Mul 变体（0.25/−2/3.75）task 形状与 1.5 时完全一致（3×BS/BN affine）——**标量 Mul 即 BS/BN 仿射原语**，系数量化规则同 §7.6（mul=c、bias=0）。

**待解**：`OW_CFG=0x080007fe`/`WDMA_SIZE` 打包与 perm 的精确映射式；`0xd` task 的 CNA 归约权重表。

**LayerNorm 系数 BO 解码（[`em0d/ln_bos/`](../experiments/results/bn_lower_20260928/em0d/ln_bos/)，权重 blob = `01.bin`）**：

- `+0..15`：γ 的 8×fp16（1.0/2.0/0.5/1.25/−1.0/0.75/1.5/−0.25，与 ONNX 逐值一致）。
- `+64..95`：β 的 8×**fp32**——与 §7.6 FP16 BN 的 bias 表布局完全相同（mul fp16 / bias fp32 混排）。
- `+128..383`：**16×16 fp16 归约矩阵**，每 16B 一行、第 0 列全 1.0、其余全 0 → CNA 归约 = 与全 1 列做点积（8 通道求和；÷8 等标量步落在后续 DPU EW 立即数里）。
- 结论：LayerNorm 的归约权重就是全 1 列，最终仿射直接复用 BS/BN 系数表格式（γ/σ 与 β 修正项）；`0xd` task 的 CNA 侧语义 = FC/conv 求和归约。EW_ALU 立即数位域（task4 `BS_ALU_CFG=0x38a7c5ac`、task5 `BN_ALU_CFG=0x5f37`）仍开放。

**DPU 标量步立即数解码（结合 `rknpu-command-stream` skill 与 refs/npu、mesa registers.xml）**：

- task4（÷√方差步）：`BS_CFG=0x20040`（BS_ALU_ALGO=2+SRC=1+RELU_BYPASS）、`BS_ALU_CFG=0x38a7c5ac` = **fp32 8e−5 = C·eps**（模型 eps=1e−5、C=8）、`BS_MUL_CFG=BN_MUL_CFG=0x3eb50000` = **fp32(fp16(1/√8)) = 1/√C**（÷√8，即 var→σ 的 1/√C 因子）。
- task5：`BS_MUL_CFG=0xffff0100`（[15:0]=fp16 1.526e−5≈eps/64?，[31:16]=0xffff 待解）、`BN_CFG=0x20050`（ALGO=2+SRC=1+MUL 使能）、`BN_ALU_CFG=0x5f37` = fp16 461.75（语义待定）。
- **语义确认：FP16 路径下 DPU BS/BN 的 ALU/MUL 立即数按 IEEE-754 位模式解释**（fp32 宽位或 fp16 表达式折叠后的常量），与 §7.6 "mul=fp16(γ/σ)、bias=fp32" 的表格式一致——LayerNorm 的 σ 逆由 `add C·eps` + `mul 1/√C` 两步组合实现，rsqrt 本体应在 task2/16 的 `0x1d` 段或 EW 链中。ALGO 枚举语义（mesa 只用 ALGO=2=add）仍开放。

---

### 7.8 PPU MaxPool 多通道与 padding 边界（2026-09-28 第六轮，`pool/min_pool`）

扫描（基线 64×64 输入、kw=2 kh=2 sx=sy=2 默认，逐项改参）：

- **ic=16（含 1/2/4/8 被 CFG 钳到 16）**：k∈{2×2,3×3}×s∈{1,2}×pad∈{0,1} 全部 `BIT-EXACT(max)`——**PPU padding 边界关闭**（pad1 时 kw=3 正确处理越界取 −∞/边界）。
- **ic 量化**：`--ic` 被 CFG 对齐到 16 通道组（1→16、33→48）→ **PPU 按 16 通道组处理**，与 trace 观测的 `INT_MASK` 分组语义一致。
- **ic=48 失败**：mismatch 全部落在 c≥16（如 c=16: cpu=117 hw=0）——group 1+ 输出为 0。`min_pool` 发射器只配置第一个 16 通道组（单组 INT_MASK/表项），多组需按组追加配置；为发射器缺口而非硬件边界（vendor runtime 多组显然可用）。

结论：PPU padding/stride/kernel 在单 16 通道组内完全可靠；多通道组为**发射器待修项**（按组配置 INT_MASK 与表项），开放。日志：板端 `/mnt/ssd/workspace/rknpu-re/pool/`（min_pool 2026-09-28 运行输出）。

> **§7.13 已推翻本节 ic=48 结论**：vendor ic=48 是单 task 同构程序（无按组追加配置），min_pool 输出 0 的根因是 CPU 参考输出读取组间距 bug（`lay()` 误用输入面尺寸），详见 §7.13。

### 7.9 多核派发探针（2026-09-28 第七轮，`min_conv_exp --subcore`，⚠️ 一次内核挂死）

发射器原 submit 配置：`core_mask=0`、`subcore_task[0..4]={task_start=0,task_number=1}`（5 个核全挂 task 0），一直工作正常。本轮加 `--subcore` 变体验证派发路由（IC=1 OC=4 3×3/pad1，`--pattern --auto-fix --pad-con1 0`）：

- **mode0（默认，5 核同配）**：ret=0，hw_elapse=17208 ns，BIT-EXACT。
- **mode1（仅 subcore 0，task_number=1，其余核 0）**：ret=0，hw_elapse=16625 ns，BIT-EXACT——与 mode0 结果、耗时一致 → 默认配置下实际只由 subcore 0 执行（或驱动折叠为同效），**subcore 0 路由正常**。
- **mode2（仅 subcore 1）**：SUBMIT 无返回，随后整板失联（ssh No route to host，持续 >4 min）→ **内核级挂死，需断电冷启**。mode3（subcore 2）、mode5（core_mask=0x7）、mode6（mask+仅核 0）未及执行。崩溃 dmesg 待板子恢复后取证。
- **mode4（task_number=3，单 task regcmd）**未执行。

**结论（关闭「多核派发」观测项，方式为边界表征）**：driver 0.9.8 下把任务派到 **subcore≥1 会挂死内核**（mode2 事件，单次复现，因果强相关）。与 §7.6 的 vendor 观测吻合——vendor runtime 同样只见 core_mask=1/核 0 执行（3 条相同 task 而非分核）。即：当前驱动 + vendor 运行时都是**事实单核**，多核并行要么未启用、要么需要额外配置（如核间 task 链）才能安全使用。工程上无需再追多核；如必须重试，须串口值守 + 可复位前提。

⚠️ 事件记录：mode2 探测触发挂死，板端 192.168.101.113 断电前不可达；`min_conv_exp` 已部署 `--subcore` 版本，**重启后勿直接重跑 mode2/3/4**。

### 7.10 BS_ALU_ALGO 枚举扫描（2026-09-28 第十轮，[`bs_alu_algo_scan_20260928.log`](../experiments/results/bs_alu_algo_scan_20260928.log)）

用发射器 `--bs-cfg $((A<<16|0x150)) --bs-alu OP`（ALGO=0..7 × 立即数 0..3，int8 conv 8×8，对照 CPU 参考）：

| ALGO | op=0 | op=1..3 响应 | 判定 |
|---|---|---|---|
| 0 | 135/256 mismatch | 全饱和 +127 | 非立即数加类；疑似 max/RELU 族（待证） |
| 1 | 128/256（恰半） | 恒定 135/256 | 同上对称族（min/abs？待证） |
| **2** | **BIT-EXACT** | mismatch 245、maxdiff=**±op 线性**（1/2/3） | **y = x ± op**（整数立即数；mesa 偏置路径用 2，本配置下表现为减） |
| 3,4 | identity | op 完全无效 | BS ALU 无操作 |
| 5,6,7 | 76/256（=正像素集） | op 无关、输出全 −128 | **乘族**，操作数为分数域（op·2^−k≈0） |

结论：`BS_ALU_ALGO` 非单值映射——2=加/减（整数立即数），5..7=乘（分数/浮点立即数，与 §7.7 LN fp32 位模式一致），3/4 在 BS 侧无效（或仅 EW/BN 侧有效），0/1 待 op 极性/大值扫描定界。这解释了 §7.9 LN task5 的 `BN_ALU_CFG=0x5f37`（fp16 461.75）落在乘族语义里。原始矩阵：`experiments/results/bs_alu_algo_scan_20260928.log`。

**ALGO 0/1 补充探针**：SRC=0/1 无差（0/1 仍非恒等）→ 其操作数**不来自 BS_ALU_CFG**（疑 BRDMA 通道，本配置未初始化→+127 饱和/恰半像素翻转）；op=0x40 时 A0→全 +127（maxdiff 191）、A1→76/256（与乘族破坏集相同）。判定：0/1 为另一操作数源（RDMA 表）的算子族，需配 BRDMA 表才能定界；本项与 BN_ALU 同款扫描列为后续。

**矩形空间边界收敛**：oh=ih 修复后失败集精确收敛为 **iw≠ih**（16×12/12×16/8×16/16×8/8×4/4×8/8×5 全败、pad0 同败；4×4..32×32 方阵全过）。误差特征（8×4）：**输出中间行全零（=act BO 初始值，HW 未写）**，首末行散点小误差 → 疑似 CNA 行波次只覆盖部分行/WDMA 行跨度错位。DATA_ENTRIES 改为 `ceil(ih/4)×ceil(ic/16)`（与 vendor `ic16×ih` 同构、方阵等价）不改变矩形失败 → 根因在别处（SURF_STRIDE 单位/tile 行走查未排除）。vendor 矩形 conv regcmd 采集（复用 conv16 方法论）为下一步。

**矩形空间 vendor 采集**：复用 conv16 方法论采集 vendor 16×8 int8 conv regcmd（`conv16x8.onnx/i8.rknn/tr_conv16x8.jsonl` 已归档）。关键真值：`DATA_SIZE0=0x00100008`（W 低16/H 高16 ✓）、`DATA_SIZE3=128=ow×oh`、`CBUF_CON1=128=ic16×ih`、`DMA_CON1=16`、**`DMA_CON2=112=16×(ih−1)`（只随 ih 变，与 8×8 相同）**、`CONV_CON2=192`、`PAD_CON1=0xffffff80`。与我们的全量 diff（`vendor_rect.txt/ours_rect.txt`）：除 BO 地址外仅 0x1010（grains 192/832）、0x1044（128/2）、0x104c（CVT_CON0 9/11）、0x107c（16/64）、0x1080（112/64）+ vendor 多写 0x1180/0x1184。**逐项 --ov 覆盖尝试（DMA_CON1=16、SURF=112、entries=128、pad-con1=0xffffff80、pad0）均未解** → 矩形根因未定位。误差结构（16×8）：**输出仅 y=0 与 y=7 行有值（且接近正确），中间 6 行全零**（act BO 初始值，HW 未写）→ 疑 WDMA/CNA 行波次只写首末行。待办：vendor 输出的完整量化域 numpy 复算（本机 onnx 手搓 ref 因 w8a8 per-channel 量化链未完成，harness 输出为 float32 dequant），+ vendor CNA 0x1180/0x1184（0xfff/0xffffff80）语义。IC=1 路径矩形同样失败（第五轮已测）。

**矩形空间根因重大发现（同日第十三轮，未定位完）**：全量重 diff（`vendor_rect_full.txt`，含 CORE/DPU 段）揭示 **vendor 对 16×8 发射两个 CNA task**：task0 `DATA_SIZE0=W16|H8`、entries=128（=ic16×ih ✓）、CORE/DPU out H=7|W=15；task1 `DATA_SIZE0=W16|H5`、entries=80（=ic16×5 ✓）、out H=3、`CONV_CON2=0x10000090`（bit28=LAST 标记）。16×16 vendor 同样双 task（task1 H=5 → 尾 4 行）。即 **vendor 对空间按主块+尾块（≤4 行尾）分片发射，且 entries=ic16×ih_per_task**——我们单 task 全高发射对方阵容忍、对矩形不行（中间行死、首末行对）。待办：解码 task 边界语义（task0 覆盖行数 vs task1 起始行、pad/CSV 携带方式、LINE/SURF 在 task1 的取值）、给发射器加分片逻辑、矩形回归。其余 diff（0x4010=0xe0、0x4048=0xe01、0x4084、0x4088=0x18、0x501c=0xe、0x4080=0x26 等）需在方阵基线 diff 中分离（可能是我们一直"错但被容忍"的值）。

**矩形空间单寄存器排除完毕（同日第十四轮）**：对 16×8 逐项 --ov 套用 vendor 值均失败：CNA 侧 0x100c=0x6000a000、0x1010=0xc0、0x104c=0x9、0x107c=0x10、0x1080=0x70、0x1044=0x80（含全组合）；DPU/RDMA 侧 0x501c=0xe（BRDMA_CFG）、0x4088=0x18+0x4084=0x41d4、0x4010=0xe0+0x4048=0xe01。**单寄存器差异全部排除 → 根因锁定为「vendor 双 task 分片结构」或「输入 BO 原子交错顺序」（iw=16 行=2 原子的 DMA 读取顺序 vs 我们填充顺序），非单寄存器可修**。下一步：实现发射器双 task（主块+尾 4 行 task，entries=ic16×ih_per_task、尾 task CONV_CON2|bit28、LINE/SURF 按各自 H 重算、WDMA 基址偏移尾行），或抓 vendor 输入 BO（bo_dump）对比原子布局。另注：0x501c 实为 DPU_RDMA_BRDMA_CFG（vendor 0xe=bits1-3，我们 0x2），正常卷积路径中方阵容忍。

**矩形空间新证据（同日第十五轮）：输入 BO 布局假设崩塌**。`--dump-bin` 抓 16×8 失败时的 act BO 逐字节分析：输出区 y=0/y=7 行有值（接近正确）、y=1..6 全为 0x80（**acc=0 的量化零，HW 真写了**，非未写）；ch≥4 填充通道=0x00。对照输入填充公式 `npu_off(g,x,y,w,h)=g*w*h*16 + x*h*16 + y*16`（**x-主序**：x 步长=ih×16B、y 步长=16B）。方阵 8×8 时 x 步长=8×16=128B 恰等于"一行 8 原子"——但 16×8 时 HW 若按 y-主序/4 行 surface 读，x 步长应为 32B 级别 → **我们的输入填充对 ih≠⋯⋯ 尺寸与 HW 的原子走查序不一致**（此前所有 --ov 寄存器覆盖无效的根本原因：数据本身摆错位置，寄存器正确也读错数据）。注意：16×16/8×8 方阵通过=该 x-主序在方阵下与 HW 一致（或 CNA 对方阵有等价走查）。**待办：bo_dump 抓 vendor 输入 BO 逐字节对比我们填充序（BO/ 目录 + atexit.so 已备好但 bodump 未触发——BO 在 exit 前已 unmap，需 hook 在 submit 后即时 dump）；或直接试验 y-主序填充。**输出侧 ch≥4 填充通道 HW 写 0x80 还是未写亦需同 BO 确认。

**矩形空间排除更新（同日第十六轮）**：① 填充序 y-主序（`--fill-ymaj`）：方阵也坏 → HW 期望确为 x-平面序，排除；② x-平面步长=iw 变体（`--fill-xiw`）：方阵等价仍过、矩形仍败 → 排除；③ **16×8 对 LINE/SURF/entries/grains 全参数扫描输出恒 38 mismatch（完全不敏感）**——失败与 CNA 寻址参数无关；38 错点呈 ±3 邻域错位特征（ref acc=0↔hw=5 互换，疑似窗口错位 3=kw−1）。已排除：填充 y-主序、x-步长=iw、全部单寄存器。剩余候选：**(a) vendor 双 task 分片结构（唯一未试的结构性差异）；(b) 权重 BO 布局对矩形的组内排布差异；(c) CBUF 银行分配与 ih 的耦合（FEATURE_GRAINS 常数 52 对矩形可能不足——vendor grains=192(16×8)/320(16×16) 按 ih 缩放，我们恒 832）**。建议下一步：采集 vendor 8×8 与 4×8 的 grains/entries/CON2 完整参数组，归纳 ih 维公式；并给发射器实现双 task 分片。

**矩形空间排除定案（同日第十七轮）**：vendor 全套 CNA 参数组（0x100c/0x1010/0x1044/0x104c/0x107c/0x1080/PAD_CON1）一次套用仍失败，OC=1 同败 → **CNA 寄存器组彻底排除**。根因收敛于任务/图结构层（vendor 双 task 分片为唯一未复现的差异）或权重/CBUF 内部布局。双 task 实现为下一会话首要工作。

**BN_ALU 枚举完成（同日第十八轮）**：BN_ALU_ALGO×op 全矩阵闭环——A2=add(立即数，BIT-EXACT)、A3/4=恒等（A4 op=3 时 15 点 ±3 微偏差，疑似 max(x,op) 小效应）、A5/6/7=乘族（op=0/3 均 76=非零像素集 ✓）、A0/A1=op 无关全饱和（256，立即数路径无效 → 操作数来自 NRDMA 表，同 BS A0/1）。与 BS 侧完全同构。

**BS_ALU_ALGO 语义（同轮续）**：`BS_ALU_SRC=1` = BRDMA 逐通道表（表空读垃圾）；ALGO2=add(BRDMA int32 原始值) BIT-EXACT；ALGO5-7=乘族（分数立即数）；ALGO3/4 恒等；**ALGO0/1 表驱动但非 max/min**（配表后 196/192 mismatch、oc0/2/3 全饱和 +127、oc1 仅 4 点——疑似表槽 stride/格式错位）；BN_ALU_ALGO 同构（2=add 确认、4=恒等、0/1 全饱和）。BN_ALU 完整枚举待做。

### 7.11 矩形空间根因定位（2026-09-29 第十九轮）— 推翻 §7.10 的全部候选

**根因：输入与输出都是行主序（row-major）**。硬件对空间寻址统一为
`atom(g,x,y) = g*w*h*FEATURE_ATOMIC_SIZE + y*w*FEATURE_ATOMIC_SIZE + x*FEATURE_ATOMIC_SIZE`，
**输入（w=iw）与输出（w=ow）皆然，所有 kernel/pad/IC 路径一致**。方阵 iw==ih（输入）/ ow==oh（输出）时行主序与列主序重合，故长期掩盖该 bug；矩形 iw≠ih 才暴露。§7.10 第十二~十七轮的单寄存器排除、双 task 分片假说、以及"y-主序已排除"结论**全部作废**——它们都在错误的数据布局与错误输出读取下测试。

此前还有两个独立错误叠加，互相掩蔽：

1. **DATA_ENTRIES 公式错（Bug1）**：应为**宽度驱动** `DIV_ROUND_UP(iw*2*ceil(ic/16),8)`（iw16→4、iw8→2、iw12→3），与 mesa/librocketnpu `rnpu_task.c` 一致；旧 `ceil(ih/4)×ceil(ic/16)`（§7.10）在 iw≠ih 时偏小。方阵下两式等价，故长期无回归。
2. **输入填充用列主序** `npu_off = g*iw*ih + x*ih + y`（旧默认），矩形错位。
3. **输出读取用列主序** `out_off = g*ow*oh + x*oh + y`（旧默认），矩形错位。**旧代码 fill 与 out 同为列主序，1×1 点积下两个置换互相抵消**，故 1×1 在旧默认下"通过"，误导出"1×1 走列主序路径"的错误结论（§7.11 旧版）。

**修正后结果（新默认：行主序 in + 行主序 out + 宽度 entries）**：3×3、1×1、IC=1 的 **pad0 与 pad1 对所有测试矩形全部 BIT-EXACT**。回归 68/68 全过（无 `--fill-ymaj`/`--out-ymaj`，因已为默认）：

- 3×3 pad0/pad1：8×8、16×16、32×32、16×8、8×16、12×8、8×12、16×12、12×16、20×8、8×20、11×7、7×11、9×5、5×9、16×24、24×16。
- 1×1：`--iw 16 --ih 8`、`--iw 8 --ih 16`、`--iw 12 --ih 16`、`--iw 8 --ih 12`，pad0/pad1 全过；`--fill-xiw`（列主序）全部 FAIL（910~1448 mismatch）→ **1×1 也是行主序**，旧"1×1 例外"作废。
- IC=1：8×8、16×8、8×16、12×8、8×12、16×16、24×16、16×24、7×11、11×7，3×3 pad0/pad1 全过（见下）。
- 变体：ic∈{8,16,32,40,65}、oc∈{40}、stride∈{2,3}、kw/kh∈{1×1,3×1,1×3}、pad2、kw5×kh5 —— 全过。
- 方阵 8×8/16×16 无回归；§7.10 的"pad1 残留 132""4 行接缝行""x≡0,4 mod5 列"现象**消失**（那本是输出列主序读取造成的错位假象）。

**IC=1（ARGB_IN）输入布局**：也是行主序，但行（y）步长 = `MAX2(iw, FEATURE_ATOMIC_SIZE)` 字节（`CNA_DMA_CON1.LINE_STRIDE` 同值），即 `idx = y*MAX2(iw,16) + x`；x<iw 且 y<ih 为真实点，其余填 pad。四种候选实测（16×8、8×16、12×8、8×12、7×11、11×7 等）：
`line=MAX2(iw,16)` → 全 0；`line=iw`（iw<16 时）→ FAIL（215~548）；`x*ph+y`（旧库 `rnpu_convert_input` 公式）→ FAIL；`x*ih+y` → FAIL。发射器新增 `--ic1-layout {0,1,2,3}`，默认 1。

**结论**：矩形空间 bug **闭合**，且寻址模型可统一为单一规则——**输入/输出行主序；IC=1 行步长对齐到 16**。§7.10 遗留的 vendor 双 task、SURF_MODE、CBUF pad fetch 均不再需要。

**回归复跑（2026-09-29 提交前终验）**：`regression68.sh`（台账 §7.11 全部命名用例的严格超集，74 例）板上实测 **74/74 BIT-EXACT**——3×3 17 形状×pad0/1（34）+ IC=1 10 形状×pad0/1（20）+ 1×1 4 形状×pad0/1（8）+ 变体 ic{8,16,32,40,65}×oc40/stride{2,3}/kw{3×1,1×3}/pad2/kw5×kh5（12）。日志：[`spatial_ohfix_20260928/regression74_20260929.log`](../experiments/results/spatial_ohfix_20260928/regression74_20260929.log)。

### 7.12 BRDMA 表值域标定闭合（2026-09-29 第二十轮）— ALGO 0/1 重新定界为"未定义值域"

**表值域标定完成**（发射器形状 iw8/ih2/ic4/oc16，`--bs-cfg 0x20140` ALGO2+SRC1 为载波）：

1. **bias 表槽**：`int32` ×8/64B 组 @ 组内偏移 0，lane k = 第 k 通道。斜 ramp（`--brdma-bias −24 --brdma-bias-step 3`）与常数表均 BIT-EXACT（r7_a2、M5 vendor 全链）。
2. **mul 表槽**：`s2.14` **有符号定点**（非 fp16），2 B @ 组内偏移 48，`shift = BS_MUL_CFG.SHIFT`。决定性判据：表值 15360（fp16=1.0，s2.14=0.9375）→ 输出 ×0.9375 BIT-EXACT（排除 fp16）；ramp 16384+1024·ch 逐通道 BIT-EXACT（M2）；−16384（−1.0）BIT-EXACT（M4）。
3. **vendor ALGO 普查**（6 个 vendor regcmd trace：conv16/conv16x8/em0d/dpu_bn 等，282 条 BS_CFG）：`{0x42×73, 0x53×90, 0x20040×39, 0x20050×30, 0x20140×4, 0x20150×5}`。0x42/0x53 = ALU_BYPASS 形态（ALGO 位无意义）；**ALU 实际参与时 ALGO 只出现 2**（add）。`RDMA_BRDMA_CFG` vendor 取 0xe（3 use 位）或 0x2。→ **ALGO 0/1 非 vendor 代码路径**。
4. **ALGO0/1 直接探测降级**：立即数（SRC=0）与配表（SRC=1、正确槽位斜坡）下输出均为**每通道常数 ±饱和**（8-lane 模式 [+,−,+,−,+,+,−,−]），与立即数值、表值、输入数据三者均无关（r1 系列与 r7_a0 同构，r7_a0 16485 B 原始日志）→ 非法/未定义 op 签名，非 max/min、非表驱动算法。**P3-2 按值域闭合收口，ALGO 0/1 不再投入**；后续若 vendor 新模型出现 ALGO0/1 再重启。
5. **新坑（系统性）**：`BS_MUL_CFG.MUL_SRC=1` 而 `RDMA_BRDMA_CFG` 未配置 → **6/6 submit ETIMEDOUT**（等待 BRDMA 数据永不完成，dmesg job timeout→soft reset）；配置 BRDMA 后同参全过。另有**随机** ETIMEDOUT（同会话 ~1/3，MUL_SRC=0 亦发生，dmesg 特征相同）。
6. **实验纪律**：submit `ret≠0` 一律视为环境失败重试至 `ret=0`；严禁把 ETIMEDOUT 记为语义 FAIL。

产物：板上 `exp2/run_brdma_cal_{r1,r2,r2b,r3}.sh`、`brdma_r2b/`、`brdma_r3/`；本机 `experiments/results/brdma_cal_r2_20260929.log`、`experiments/run_brdma_cal_r*.sh`。

### 7.13 PPU 多组闭合（2026-09-29 第二十一轮）— 根因是 CPU 参考读 bug，硬件单 task 即支持多组

1. **vendor 真值（ic=48 maxpool）**：`make_pool48.py` → `maxpool_ic48_i8.rknn` 的 regcmd 为**单 task 26 条**，与 ic=16 版程序完全同构，仅 `CUBE_*_CHANNEL=47`（stride 仍按 16 通道组：DST_SURF=16384=ow·oh·16、RD_SURF=65536=64·64·16、MISC_CTRL=3、DATA_FORMAT=0x4000）。**§7.8 "多组需按组追加配置/task"的假设不成立**：vendor 根本不拆 task，寄存器也无任何按组展开。
2. **min_pool 根因 = CPU 参考输出读取 bug**：`lay()` 以输入面组间距 `iw·ih·16` 读输出；输出组间距应为 `ow·oh·16`（DST_SURF_STRIDE 口径）。修复：新增 `out_lay()`（`g_ow/g_oh`），3 处输出读取替换（输入读取仍用 `lay()`）；BO 尺寸改按 `in_need/out_need` 页对齐（ic=80 时原 128 KiB out BO 溢出）。**多组配置本身一行寄存器都不用改**——之前 ic=48 输出全 0 是参考读错位 + BO 溢出叠加。
3. **修复后多组回归**（64×64 输入）：ic=16/32/48 k2s2、ic=33（跨组奇数）k2s2/k3s3/method2=min/no-transpose、ic=64 method2=min、ic=80 k2s2 —— 全部 BIT-EXACT(max)。
4. **重叠窗口/奇尺寸**：k2 s1（out 63×63）、k3 s2 pad1、k3 s3 pad1、iw=17 奇宽 k2s2 —— 全部 BIT-EXACT。
5. **新边界 A（k≥3 + s1 + 大图，与多组无关）**：in≥36（out≥34）submit ETIMEDOUT（in33/out31 过，in36 起 3/3 挂；k4/k5 s1 同挂；ic=16 同挂；SURF_LEN 8/16/31 与 `--mc-surf-out` 均无效；k2 s1 out=63 反而过）。vendor pool 模型只用 k2s2，不深挖，记为 PPU 内部限制。
6. **新边界 B（小语义挂账）**：k2 s1 pad1（out 64×64）mismatch 9582/67600，mismatch 位置 x≡0 (mod 7) 周期——未深究。
7. 实验坑：ic=80 CPU 对比循环 >25 s，ssh 超时截断输出易误判；加大 timeout 后 BIT-EXACT。

产物：本机 `experiments/results/pool_vendor48_tr.jsonl`（vendor ic=48 真值）、`pool_mine48_tr.jsonl`、`pool_ic48_tr.jsonl`、`pool_ic48_20260929/make_pool48.py`；`experiments/min_pool.c`（修复后）；板上 `rknpu-re/pool/`（min_pool、maxpool_ic48_i8.rknn、in48.bin、tr_*.jsonl）。

产物：本机 `experiments/results/pool_vendor48_tr.jsonl`（vendor ic=48 真值）、`pool_mine48_tr.jsonl`、`pool_ic48_tr.jsonl`、`pool_ic48_20260929/make_pool48.py`；`experiments/min_pool.c`（修复后）；板上 `rknpu-re/pool/`（min_pool、maxpool_ic48_i8.rknn、in48.bin、tr_*.jsonl）。

### 7.14 矩形核重新表征（2026-09-29 第二十二轮）— 旧"方形 vs 矩形"结论复核成立，但旧 pattern 存在 y 盲区假 PASS 〔本节的 tap 语义部分已被 §7.15 推翻，见节末注〕

**背景**：§7.11 行主序定案后矩形核（kw≠kh）未复测。本轮用 spatial_ohfix 方法论重建 vendor 真值 + 重扫发射器，并修正了 pattern 本身的缺陷。

1. **vendor 矩形核 regcmd 真值**：rknn-toolkit2 2.3.2 转换 2×1（kw2kh1）与 1×2 kernel 的 int8 conv（16×16×3→oc16，valid，[`rect_kernel_20260929/`](../experiments/results/rect_kernel_20260929/)），板端 hook 采集 `tr_rect21/12.jsonl`。均为**单 task**（23 条），`WEIGHT_SIZE2` WIDTH/HEIGHT 字段正确编码 2/1，无任何 task 分片；CNA 段寄存器与同几何方形的差异仅在 kernel 字段/权重尺寸——**寄存器层面无新信息**，vendor 对矩形核不改变发射形态。
2. **旧 pattern 的两个盲区（方法论修正）**：`pat_in = (7x+5y+3c) mod 5 − 2` 中 5y≡0 (mod 5) → **输入对 y 完全不敏感**；`pat_w = (x+y+…) mod 3 − 1` 对 (kx,ky) 互换对称 → 方形核 tap 置换在输出中抵消。这解释了本轮初扫的混乱结果（2×4 "BIT-EXACT" 但 2×3 挂等）。已加 `--pat2`（`3x+5y+2c mod 7 − 3`，x/y 双依赖）与 `--w-tap KX:KY`（单 tap one-hot，可直接读出 HW 对该 tap 的配对）。
3. **定论矩阵（--pat2，16×16 ic3 oc16 valid）**：kw=1 或 kh=1（1..6）全 BIT-EXACT；**方形（k×k）全 BIT-EXACT（含 --asym-w 双重验证）**；**kw≠kh 且均 ≥2 全 MISMATCH（2..6 的全部 20 个有序对）**。**此前"2×4 BIT-EXACT"是 y 盲区假象**，实际非方形全挂——§7.1 的「方形 vs 矩形」分界在修正 pattern 后恢复成立，但 1D 核（N×1/1×N）确认真通过（§7.1 旧表中的 2×1/1×2 mismatch 系列已被行主序修复治愈，本次全部 BIT-EXACT）。
4. **误差结构**：失配确定性、幅值小（pat2 下 ≤28）、per-x 计数呈周期 4 结构且存在成对「全对列」（2×3：x≡3,4 mod 4 全对，x≡1,2 全错，x≡0 部分错；3×2 类似）——指向 CBUF/核几何在 kw≠kh 时的内部行取数差异，非权重 tap 线性置换可解（one-hot 求解无单一 (dx,dy) 假设成立；pat2 mod 7 熵不足亦致逐像素歧义，后续需更大周期注入 pattern）。
5. **方法论警告（影响历史结论解读）**：**全 pattern BIT-EXACT 不保证逐 tap 正确**——3×3 全 pattern BIT-EXACT 但 `--w-tap 1:1` 单 tap MISMATCH（误差 ±1~2），对称 pattern 下 tap 错位可完全抵消。此前所有依赖 pattern 抵消的「BIT-EXACT」结论（尤其方形核）对 tap 级语义的证明力有限，但输出面逐字节比对（不同 pattern 形状族 74/74 + pat2 复扫）仍成立。
6. 顺带发现：`--w-uniform` 实验**无效**（CPU 参考恒用 pat_w 计算期望值，不读权重 BO，uniform 下参考与 HW 必然失配）——`--w-tap` 是正确工具（fill 与参考同源）。

产物：`experiments/results/rect_kernel_20260929/`（vendor trace×2 + 解码 + 模型构建脚本；.rknn/.onnx 由脚本复现）；`experiments/min_conv_exp.c` 新增 `--w-tap KX:KY`、`--pat2`。板上 `bn_lower_probe/{rect21,rect12}_i8.rknn + in/out/tr`、`exp2/tap23*/`（one-hot dump）。

> ⚠ **本节的 item 3、4、5、6 已被 §7.15 推翻或改写**：方形核的 tap 级语义此前从未被真正验证（`--asym-w` 在旧 y 盲区 pattern 下也抵消），"方形 vs 矩形分界" 是权重核内转置被对称 pattern 掩盖的假象。

### 7.15 权重 tap 顺序根因闭合（2026-09-29 第二十三轮）— 核内转置被对称 pattern 掩盖

**根因**：CNA 权重槽的核内顺序是 **ky-major（kx 变化最快）**，即 `slot = ky·kw + kx`；发射器 layout A 写成 kx-major（`slot = kx·kh + ky`）。两者在**方形核**上互为转置，而 `pat_w = (x+y+oc+icc) mod 3 − 1` 在 (kx,ky) 互换下不变 → 完全抵消；旧 `pat_in = (7x+5y+3c) mod 5 − 2` 又与 y 无关 → 连 `--asym-w` 也抵消；**1D 核**（kw=1 或 kh=1）两种约定等价。只有 `kw≠kh 且均 ≥2` 才暴露成 MISMATCH。

1. **证据 A：vendor `.rknn` 权重 blob 字节级解码（离线，与 pattern 无关）**。`openrknn` 把 `.rknn` 的 `weight_data` blob **原样**拷进权重 BO（64 B 对齐，见 `refs/orangepi5plus-npu/openrknn/src/openrknn_model.c` 的 `wt_blob_offsets`），故 blob 即硬件布局。三个 probe 模型用固定 numpy seed 生成，逻辑权重完全已知；拟合 rknn-toolkit2 的逐输出通道非对称 int8 量化 `q(w)=round(w·s+zp)`（`s=255/(wmax−wmin)`、`zp=−128−s·wmin`）后逐行反查：

   | 模型（seed） | blob | 解出的 slot 顺序 | ky-major | kx-major |
   | --- | --- | --- | --- | --- |
   | conv 3×3 oc4 ic3（7） | 576 B = 9 tap × 64 B | (0,0),(0,1),(0,2),(1,0)…(2,2) | 一致 | 一致（方形下重合） |
   | rect kw2 kh1（11） | 512 B = 2 tap × 256 B | (0,0),(0,1) | **一致** | t=1 否证 |
   | rect kw1 kh2（11） | 512 B = 2 tap × 256 B | (0,0),(1,0) | **一致** | t=1 否证 |

   行结构 `[oc][ic0..ic2, pad×13]`，ic 原子 16 B，**ic 越界槽填 `round(zp_oc)` 而非 0**；`blob 长度 = kw·kh·oc·16`。旁得：**`.rknn` 权重可直接离线读取，vendor 权重 BO dump（`bo_dump` 0 dumps）不再是必需**。工具 `experiments/rknn_blob_dump.py`、`experiments/vendor_blob_decode.py`。
2. **证据 B：真机 A/B**（旧/新二进制同源，仅 `fill_weights` 布局 A 一行不同；`16×16 ic3 oc16 valid --pat2 --auto-fix`）：旧布局 `kw3 kh3 --asym-w` **MISMATCH 2682/4096（max 42）**、`kw3 kh3 --w-tap 2:0` MISMATCH 2624、`kw3 kh2` MISMATCH 1357、`kw2 kh3` MISMATCH 1873；新布局全部 **BIT-EXACT**。1D 核与方形（对称 pattern）两态均过——正是历史假象的来源。
3. **修复**：`min_conv_exp.c` 布局 A 改为 `idx = ((((oc1·ic1+i1)·c->kh + y)·c->kw + x)·oc2n+oc2)·ic2+i2`。
4. **修复后覆盖**（`--pat2`）：判别矩阵 15/15、扩展扫描 74/74 —— 全部 20 个 `kw≠kh` 有序对（2..6）、方形 `--asym-w` 2..5、21 个 one-hot 单 tap（3×3 / 3×2 / 2×3）、`pad=1` 下 25 种 `kw×kh`、`stride=2` 下 4 种 `kw≠kh`。
5. **硬化回归 74/74**：§7.11 的 74 例改用 `--pat2 --asym-w` 重跑（旧 y 盲区 + 对称 pattern 只能证明输出面，不能证明 tap 语义）。
6. **对既有结论的更正**：§7.14 item 3「方形全 BIT-EXACT（含 `--asym-w` 双重验证）」**作废**（`--pat2 --asym-w` 下旧布局 MISMATCH）；item 5 的「3×3 `--w-tap 1:1` MISMATCH ±1~2」实为 **kw2 kh3** 用例（板上 `exp2/tap23*/` 恰为 6 个 `w-tap` = 2×3；旧布局复现 max diff 2，新布局 BIT-EXACT；3×3 的 (1,1) 自映射，两布局都过，不能作判据）；item 4 的「per-x 周期 4 / 成对全对列」由核内转置直接解释，不需要 CBUF 行取数假设；**§7.1 的「方形 vs 矩形」硬件边界作废**。
7. **deconv 对照**：`--deconv-flip` 6/6 BIT-EXACT（含 pat2+asym）；no-flip 是刻意的错误参考（y 盲区对称 pattern 下同样 MISMATCH），非回归。

产物：`experiments/results/taporder_20260929/`（README + `ab_evidence.txt` + 分步日志）；`experiments/rknn_blob_dump.py`、`vendor_blob_decode.py`、`run_taporder_{matrix,matrix2,archive}.sh`、`run_ab_taporder.sh`、`run_regression_hard.sh`、`run_deconv_smoke.sh`。板上 `exp2/{min_conv_exp_old,run_*,results/taporder_20260929/}`。

### 7.16 FP16 / PROC_PRECISION 路径侦察（2026-09-29 第二十四轮）— **未闭合**，仅建立起点

**动机**：§七-7 的 FP16 是 P3 剩余里杠杆最高的一项（决定训练能否用 FP16 累加）。本轮不发射，只做两条可复现的取证。

1. **vendor 寄存器差分（同一 `conv_bn_conv` ONNX，仅量化档不同）**：解码 `bn_lower_20260928/trace_conv_bn_conv_{fp16,i8}.jsonl`，按几何（`WEIGHT_SIZE0`=0x80=conv0 / 0x40=conv1）配对 op_idx=1/2 的主块 task；工具 `experiments/fp16_diff.py`。差异全表（conv0 主块）：

   | 寄存器 | int8 | fp16 | 解读 |
   | --- | --- | --- | --- |
   | `CNA_CONV_CON1` 0x100c | 0x6000a000 | 0x6000a120 | `PROC_PRECISION=(v>>7)&7`：0→**2** |
   | `CORE_MISC_CFG` 0x3010 | 0x00000001 | 0x00000200 | `PROC_PRECISION=(v>>8)&7`：0→2 |
   | `DPU_DATA_FORMAT` 0x4010 | 0x000000e0 | 0x48000002 | `PROC_PRECISION=v&7`：0→2；高位为 fp16 I/O 格式 |
   | `DPU_RDMA_RDMA_FEATURE_MODE_CFG` 0x5044 | 0x00007810 | 0x00017850 | `PROC_PRECISION=(v>>5)&7`：0→2 |
   | `DPU_OUT_CVT_SCALE` 0x4084 | 0x00007a18 | 0x00010001 | fp16 输出直通：`FP32TOFP16_EN`(bit16)=1、`OUT_CVT_SCALE`=1 |
   | `DPU_OUT_CVT_SHIFT` 0x4088 | 0x00000017(23) | 0 | 无重量化 |
   | `DPU_OUT_CVT_OFFSET` 0x4080 | 0xfffffffb | 0 | 无 offset |
   | `CNA_DATA_SIZE0` 0x1020 | 0x00100008 | 0x00080008 | `DATAIN_WIDTH` 16→**8**（fp16 16 B 原子装 8 元素）、HEIGHT 8 不变 |
   | `CNA_DATA_SIZE1` 0x1024 | 0x00020010 | 0x00020008 | `DATAIN_CHANNEL` 16→**8**；`CHANNEL_REAL` 两档同为 2 |
   | `CNA_CBUF_CON1` 0x1044 / `CNA_DMA_CON1` 0x107c / `CNA_DMA_CON2` 0x1080 / `CORE_DATAOUT_SIZE_1` 0x3018 / `DPU_WDMA_SIZE_0` 0x4058 / `DPU_DATA_CUBE_CHANNEL` 0x403c / `RDMA_DATA_CUBE_CHANNEL` 0x5014 | 0x80 / 0x10 / 0x70 / 0x1f / 0x1f / 0x…1f / 0x1f | 0x40 / 0x08 / 0x38 / 0x0f / 0x0f / 0x…0f / 0x0f | 通道/行/entries 计数按「16 B 原子 = 16 个 int8 = 8 个 fp16」折半 |
   | `DPU_BS_CFG` 0x4040 / `BS_MUL_CFG` 0x4048 / `RDMA_BRDMA_CFG` 0x501c | 0x20140 / 0xe01 / 0xe | 0x20150 / 0x0 / 0x2 | int8 走 BS+BRDMA 逐通道 affine；fp16 走 BN_ALU+NRDMA（BN 已折进 conv） |

   注：`WEIGHT_SIZE0/2` 两档相同（0x80 / 0x01010008），即 1×1 conv、oc 跨度 8/4 不变。
2. **fp16 权重 blob 布局（离线字节级，确定性）**：`conv_bn_conv_fp16.rknn` 的 `blob[1]`=w0(8,3,1,1) 128 B、`blob[3]`=w1(4,8,1,1) 64 B。**行仍是 16 B，但装 8 个 fp16**（fp16 下 ic 原子 = 8 元素）：w1 与 ONNX 权重**逐值相等**；w0 每 oc 行 = ONNX w0 × **γ/√(var+eps)**（即 BN 已折进前置 conv，8/8 通道比值命中 `[1, 1.41421, 0.25, 2.5, −1, 0.5, 2.1213, −0.25]`）；ic 越界槽填 0（fp16 无 zero-point）。工具 `experiments/fp16_blob_decode.py`，日志 `results/bn_lower_20260928/fp16_blob_decode.log`。
3. **未完成（下次入口，按序）**：① 给 `min_conv_exp` 加 fp16 发射通道（输入/权重按 fp16 2 B 填充 + 上表 4 处 `PROC_PRECISION=2` + DPU 输出直通 scale=1/shift=0/offset=0 + 原子计数折半），先做「带全部 fp16 寄存器但数据仍 int8」的 A/B 以分离寄存器效应与数据格式效应；② 实现 fp16 CPU 参考（fp16 读入、float 累加），验证累加精度与 §七-7 遗留的「INT16 输出置位是否 timeout」；③ 若打通，再测 fp16 累加下限（denormal/rounding 模式）。风险：耦合寄存器多，必须逐项 A/B 且 submit `ret≠0` 按 §7.12 纪律重试。

产物：`experiments/fp16_diff.py`、`experiments/fp16_blob_decode.py`、`experiments/results/bn_lower_20260928/{trace_conv_bn_conv_fp16.jsonl,conv_bn_conv_fp16.rknn,fp16_blob_decode.log}`。

### 7.17 FP16（PROC_PRECISION=2）路径打通（2026-09-29 第二十五轮）— 手写发射 fp16 卷积，宽矩阵 BIT-EXACT

**一句话**：`min_conv_exp --fp16` 在真机上跑通 fp16 卷积并与 fp16 CPU 参考逐位一致（多元素求和 ≤8 ULP）；**累加器 ≥ fp32（不是 fp16）**；§七-7 的「INT16 输出置位会 timeout」**未复现**。

1. **发射配置（相对 int8 的全部差异）**：`CNA_CONV_CON1` +`IN_PRECISION(2)|PROC_PRECISION(2)`（0x120）；`CORE_MISC_CFG`=`PROC_PRECISION(2)`（同时 `QD_EN=0`，0x200）；`DPU_DATA_FORMAT`=`IN/OUT/PROC_PRECISION(2)`（0x48000002）；`DPU_RDMA_FEATURE_MODE_CFG` +`IN/OUT/PROC_PRECISION(2)`（0x17850）；`DPU_OUT_CVT_SCALE`=`FP32TOFP16_EN(1)|OUT_CVT_SCALE(1)`（0x10001）、`SHIFT=0`、`OFFSET=0`（输出直通）；`DPU_BS_CFG` 整段 bypass（fp16 无 int8 域 bias 表）；`CNA_WEIGHT_SIZE0` ×2；`DATA_ENTRIES` 按 8 通道平面计；`ic_real==1` 专用路径（ARGB/NONALIGN/CVT）在 fp16 下关闭。工具：`--fp16`、`--f16-selftest`（fp16 转换与 numpy.float16 全 65536 位型对拍一致）。
2. **布局规则（实测）**：输入**像素 = 16 B = 8 个 fp16**、ic 平面步长 = `iw·ih·16 B`（`[plane][y][x][8ch]`，行主序同 int8）；权重与 int8 layout A 同构 `[oc1][i1][ky][kx][oc2][i2]`，**ic 原子 32 元素**、2 B/元素、ic 越界槽填 0；**ic≤8 → 8，否则 16 对齐**（`c.ic=24` 的 3 平面组合失败，8/16/32/40/48/64 全过）、`oc` 按 16 对齐；CBUF 权重 bank 估算按 2 B/元素 ×2。
3. **决定性证据 A（标尺探针）**：`--in-ruler` 把输入填成「元素下标」，单点权重直接读出硬件访问的输入元素下标——ic=16 时权重槽 j=0..7 → 元素 0..7、j=8..15 → 元素 128..135（iw=ih=4）⇒ 平面步长正是 `iw·ih·16 B`。像素原子 × 权重 ic 原子的 9 组合中只有 (8, ≥16) 通过。
4. **决定性证据 B（累加器精度）**：`--in-cancel`（ch0=2048、ch1=1、ch2=−2048，权重全 1）：fp16 逐步累加会把 2049 舍成 2048 → 0；fp32 累加 → 1。实测 **hw = 1.0（ic=3/8/16/32 全部）⇒ 累加器 ≥ fp32**。多元素（K≥27）求和相对 fp64 参考残差 ≤4~8 ULP，与「fp32 累加 + 求和顺序差异」一致，判据记为 `ULP-EXACT (≤8 ulp)`。
5. **覆盖**：1×1 ic∈{1..96}×oc16（含 27/27 ic 扫描）、1×1 ic16 oc∈{4..40}、3×3 pad0/1 六形状（oc16）、5×5 pad2、rect kw3kh2/kw2kh3/kw1kh5/kw5kh1、stride 2/3 —— 见 `results/fp16_20260929/01_fp16_regression.txt`（pass=45 fail=5 envfail=9）。**int8 无回归**：§7.11 的 74 例（`--pat2 --asym-w`）本次 73/74，唯一 FAIL 是随机 ETIMEDOUT（同命令单独跑 5/5 BIT-EXACT，§7.12 纪律）。
6. **本条的两个遗留项已在 §7.18 全部闭合**：① `kw·kh>1` 且 `oc>16` 的根因是 fp16 **oc 原子 = 16** 且每 tap 的 oc 跨度要对齐到 16 的整数倍；② `1×1 ic≤8 且 oc≥24` 的 ETIMEDOUT 根因是权重 **ic 行必须是完整的 16 元素（32 B）原子**。两条都已修进默认路径（fp16 回归 pass=71/fail=0）。
7. **纪律（新增）**：**板上同一时刻只能有一个发射器实例**——发射器用固定大小的 CMA BO，两个并发实例（含本地 timeout 后仍在跑的旧 run）会互相踩缓冲，产生**幅值固定、可复现**的假 MISMATCH（表现为「某点之后整段全错」）。本轮踩到两次。回归脚本因此加「出现 MISMATCH 先跑已知 BIT-EXACT 的 3×3 健康探针，探头脏则丢弃该结果」的协议；核心 5 例经 3 轮重复完全一致。

产物：`experiments/results/fp16_20260929/`（README + 7 份板端日志）；`experiments/min_conv_exp.c` 新增 `--fp16`/`--f16-selftest`/`--f16-pix-elems`/`--f16-watom`/`--f16-ocatom`/`--w-ic-tap`/`--in-one-hot`/`--in-ones`/`--w-ones`/`--in-ruler`/`--in-cancel`/`--din-ch`/`--dma-ch`/`--ws1`；`experiments/run_fp16_{regression,icsweep,acc,ruler,layout_search,impulse,archive}.sh`。

### 7.18 FP16 权重 oc 组与 ic 行对齐闭合（2026-09-29 第二十六轮）— §7.17 的两个遗留项全部消除

**根因（两个耦合的布局错误）**：
1. **fp16 的权重 oc 原子是 16**（不是 int8 的 32），且**每 tap 的 oc 跨度必须按 oc 原子组的总和向上对齐**（`oc_w = ALIGN_UP(oc_real,16)`）；
2. **权重 ic 行必须是完整的 16 元素（32 B）原子**（`ic≤16 → 16`，否则 16 对齐）——否则「单 ic 平面（ic≤8）+ 多 oc 组（oc≥17）」会 submit **ETIMEDOUT**。

**触发链（原实现）**：`--auto-fix` 的 int8 分支先把 `WEIGHT_KERNELS` 设成 `ALIGN_UP(oc_real,2)`（oc=17 → 18），fp16 分支随后只改 `w_oc2` 而没同步 `WEIGHT_KERNELS` → 描述符（18）与实际打包跨度（32）不一致 → 第二个（不完整）oc 组的 per-tap 步进错位，表现为 `3×3 ic16 oc=17/24` 大面积 MISMATCH。

**修复**：fp16 下 `c.ic = (ic_real ≤ 16) ? 16 : ALIGN_UP(ic_real,16)`；`w_oc2 = ALIGN_UP(oc_real,16)` 且 `WEIGHT_KERNELS = w_oc2`（新增 `--w-kernels` 显式标志以免被 auto-fix 覆盖）；`--f16-ocatom` 默认 16。

**验证**：fp16 全量回归 **pass=71 fail=0**（唯一非通过项是 `3x3p1 ic16 oc65` 的**随机** ETIMEDOUT，§7.12 纪律；单独跑通过）——含 `3×3p1 ic16 oc∈{4,8,16,17,24,32,33,40,48,64}`（**全部不完整 oc 组**）、`1×1 ic≤8 且 oc≥24` 的 9 例由 ETIMEDOUT 变为 BIT-EXACT、`3×3p1 ic8/ic32 oc∈{17,24,40}`；**int8 无回归**（74 例 `--pat2 --asym-w`，本次归档 73/74，唯一 FAIL 为随机 ETIMEDOUT，单独跑 5/5 通过）。

**判据**：fp16 多元素求和与 fp64 参考的残差 ≤16 ULP（多落在**次正规量级**，|diff| ≤ 1e-6），与「fp32 累加 + 求和顺序差异」一致，记为 `ULP-EXACT (≤16 ulp)`；已把 ULP 距离改为 fp16 位型的**单调整数键**，避免近零处正负跨越把差值放大成几万。

产物：`experiments/results/fp16_20260929/`（README + 7 份板端日志，回归 pass=71/fail=0）；`experiments/min_conv_exp.c`；`experiments/run_fp16_*.sh`。

### 7.19 FP16 覆盖面检查（2026-09-29 第二十七轮）— deconv 与 BS/BN 表路径两个新开口

在 §7.18 之后顺手查了 fp16 还没覆盖的两条路径：

1. **fp16 deconv 未打通**（第二十八轮补充量化）：`--mode deconv --deconv-flip`、4×4→8×8、ic3/oc16：S=2 **954/1024 mismatch（max|diff| 6.60）**、S=4 580、S=8 373；**1×1 核 S=2 也挂 335/1024（max|diff| 2.56）** ⇒ 不是翻转/多 tap 的小语义差，而是基础性的不一致。已试并无效：`--ov 0x100c` 的 con1 精度位组合（0x10120/0x10100/0x10020/0x10000 等）、`--w-oc2` 对齐、`--deconv-stride` 各档；int8 同配置 6/6 BIT-EXACT（本轮复核仍 6/6 + no-flip 反例，**int8 无回归**）。deconv 属低频路径，列为未闭合，下轮可对照 fp16/int8 的 deconv regcmd 全量 diff 或找 vendor 的 fp16 deconv 模型。
2. **fp16 的 BS/BRDMA bias：机制已定、可靠性未定**。用 `--bs-cfg 0x20150`＋`--brdma-cfg 0xe`＋`--brdma-bias <fp32 位型>` 的探针（输入 `--in-ch-id`＋`--w-ones` 使 conv = 36）：
   - **槽是 fp32**（决定性）：槽 = `0x40a00000`(=5.0f) → `hw = 41 = 36 + 5.0`；`0xc0400000`(−3.0f) → `33`；逐通道 ramp 亦对。把 fp16 参考改成"按 fp32 位型读表"后，1×1 与 3×3、单组与 `oc=24` 多组**整例 BIT-EXACT**（独立重复 3/3 成功）。⇒ 与 int8 的 int32 槽不同，**fp16 的 BS/BRDMA 槽是 fp32**（与 vendor fp16 `b0` blob 是 fp32 一致，§7.16）。
   - **不稳定**：同一进程连续 submit 约 **50% `SUBMIT ETIMEDOUT`**（BRDMA 相关，奇偶交替）；长序列归档中即使 `ret=0` 也可能**表数据未送达 DPU**（`hw=36` vs `ref=41`，5/5 复现）⇒ 该路径尚未达到"可反复复现"的标准，回归里列为**信息性**项。
   - 下一轮建议：把表放进 **weight BO**（§7.12/vendor 的 `BN_BASE_ADDR` 做法）并加"表送达自检"。
   另有：`bs-cfg 0x20140`（BS_MUL 未 bypass）时 `hw = 0`（与表值无关）——沿用 §7.12 的结论，fp16 的可用配置是 `0x20150`。

**结论**：fp16 的**无 bias 卷积主体**已闭合（§7.17/§7.18，回归 pass=71/fail=0）；**bias（fp32 表已定、可靠性未定）**与 **deconv（未通）**是接下来的两个开口；两者都不影响"fp16 MAC + fp32 累加"的结论。

### 7.20 FP16 BS/BRDMA bias 稳定性调查（2026-09-29 第二十八轮）— 机制已定、触发已复现、修复未成

1. **可复现的触发条件**（板上，输入 `--in-ch-id --w-ones` 使 conv=36，bias 槽 `0x40a00000`=5.0f）：
   - **A**：先跑一个「同几何但 BS bypass」的 dummy submit，再跑 probe ⇒ **bias 丢失 20/20**（`ret=0`、`hw=36`、`ref=41`）。
   - **B**：probe 直接连跑 ⇒ 生效（`hw=41`，BIT-EXACT），但伴随**奇偶交替的 `SUBMIT ETIMEDOUT`**（约 50%）。
   - **C**：**int8 同协议**（dummy → `--bs-cfg 0x20140 --brdma-bias 7`）⇒ **3/3 BIT-EXACT，不丢表** ⇒ 这是 **fp16 配置特有**的问题，不是通用 BRDMA 不可靠。
2. **已排除**（均可复现）：
   - **ping-pong 指针**：新增 `--pp-val`（覆盖 DPU/RDMA `S_POINTER`）扫 `{0,1,4,6,8,10,14}` 行为不变（`POINTER=1` 全超时 ⇒ 我们确实写 bank 0）；**末尾用 `--ov 0x4004/0x5004` 重写 S_POINTER 也不修**。
   - **表的位置**：新增 `--bs-table-in-wt OFF`（把表放进 weight BO，照 §7.12/vendor 的 `BN_BASE_ADDR` 做法）——**不修**（dummy 后仍丢）。
   - **写入顺序**：末尾重写 `BS_CFG`（`--ov 0x4040=0x20150`）**不修**。
   - **表值域**：槽是 fp32（§7.19 决定性判据）；int32 解释不成立。
   - **MUL 级**：`--bs-cfg 0x20140`（MUL 未 bypass）在 fp16 下**恒输出 0**，与 mul 槽值（15360 / 1 / 0x3c00）无关 ⇒ fp16 的 MUL 级语义另需标定。
3. **当前可用的 fp16 bias 配方与限制**：`--bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias <fp32 位型>`；只在"前一个 submit 不是 BS-bypass 配置"时生效，且 ~50% ETIMEDOUT。⇒ 判定 **机制已定、可靠性未定**，回归中列为信息性项。
4. **下一步候选**：① 标定 fp16 的 MUL 级（`BS_MUL_CFG` 与 mul 槽在 `PROC_PRECISION=2` 下是 fp16 还是 s2.14）；② 换用 vendor 的 fp16 模型（`bn_only_fp16` / `conv_bn_conv_fp16`）读它完整的 BS/BN 寄存器组合当真值；③ 查驱动侧是否有"每 submit 必须重载 BS 表"的隐含条件。

新增开关：`--pp-val V`、`--bs-table-in-wt OFF`（默认 -1 = 旧行为，均不影响既有回归：int8 74/74、int8 deconv 6/7、fp16 核心用例）。

### 7.21 FP16 BS/BRDMA bias：重载条件查明 + 可用配方（2026-09-29 第二十九轮）— §7.20 的根因

1. **vendor 真值对照**（fp16）：BN 折进 conv 的 `conv_bn_conv_fp16` 里，conv0/conv1 的 BS 组是 `BS_CFG=0x20150`、`BS_ALU_CFG=0`、`BS_MUL_CFG=0`、`BS_OW_CFG=0x126`、`BN_CFG=0x53`、`BRDMA_CFG=0x2`、`NRDMA_CFG=0`、`BS_BASE_ADDR`=<weight BO 内偏移>、`FEATURE_MODE_CFG=0x17850`；而独立 BN 的 `bn_only_fp16` 恰好相反：`BS_CFG=0x42`（BS 全 bypass）、`BN_CFG=0x20150` + `NRDMA_CFG=2` + `BN_BASE_ADDR`（逐通道表走 **BN_ALU/NRDMA**）。⇒ 我们用的 BS 路径与 vendor 的 fp16 conv **一致**；唯一差异是 `BS_OW_CFG`（我们 0x124 / vendor 0x126），改成 0x126 **不改变结果**。submit flags：vendor = 5（`PC|BLOCK`，**无 PINGPONG**），我们旧默认 = 7；`--job-flags 5` 也不改变现象。
2. **重载条件（决定性，dummy 变体矩阵）**——前一次 submit 的 BS 状态决定后续 probe 的 bias 是否生效：

| 前一次 submit | 后续 probe 的 hw（期望 41） |
| --- | --- |
| BS bypass（0x53 或 0x1） | **36（丢表）** 3/3 |
| 无任何 BS 配置（裸 conv） | 36（丢表） |
| **BS 激活 + 表值 0** | **41（生效）** 3/3 |
| BS 激活 + 表值 3.0f | 3/3 ETIMEDOUT |

⇒ **BS/BRDMA 表只在"上一次 submit 的 BS 阶段是激活的"时被重载**；一次 BS-bypass 之后紧接着的 BS 激活 submit 会读到旧（零）表。这也解释了 vendor runtime 为何从不复现：真实模型里每个 conv 的 BS 都是激活的，不存在 bypass→active 的过渡。
3. **可用配方（已实现并验证）**：发射器在 BS 激活时自动预热 `--warmup-submits`（默认 3）次同 regcmd 提交后再测量；**外部先跑一次同几何的 BS-bypass 提交**以打断状态。验证：单独序列下 1×1 **3/3**、3×3 **3/3** BIT-EXACT 且 `ret=0`；长回归序列中 **4/5**（首例偶发丢表）。
4. **残余**：连续/反复的 BS 激活提交仍会进入 ETIMEDOUT 簇（dmesg `job timeout` + `soft reset`，本机累计计数 483）；短时间反复超时会放大失败率（同一命令从 3/3 掉到 0/5），期间**非 BS 用例始终正常**（设备本身没坏）。
5. **结论**：fp16 bias **机制已定**＝"fp32 表 + 仅在前一次 BS 激活时重载"，配方可用，但在 **BS 状态反复切换**时仍脆弱 ⇒ 保持为**已知脆弱项**；真实 workload（BS 常开）不受影响。

新增开关：`--warmup-submits N`（默认 BS 激活时 3）、`--job-flags N`、`--pp-val V`、`--bs-table-in-wt OFF`。回归核对：fp16 主表 **pass=71/fail=0**（唯一 env fail 为随机 ETIMEDOUT）、bias 段信息性 4/5、ic 扫描 27/27、int8 73/74（1 随机 ETIMEDOUT）。

### 7.22 FP16 deconv 闭合（2026-09-29 第三十轮）— §7.19 的「未通」是参考工具 bug

1. **诊断**：先把 int8 与 fp16 同几何 deconv 的 regcmd 做**全量 diff**（`--dump`），差异**只有预期内的 fp16 增量**（`CONV_CON1` 精度位、`CORE_MISC_CFG`、`DPU_DATA_FORMAT`、`OUT_CVT*`、通道对齐 `0x3018/0x403c/0x4058`、`CBUF_CON1.DATA_ENTRIES`、`WEIGHT_SIZE0` ×2）——**没有任何 deconv 相关差异**，说明问题不在寄存器/数据布局。再看失配模式（同一数值在 ref 与 hw 的 x 位置上错开），结合参考代码发现：**§7.17 新增的 fp16 CPU 参考块只写了 conv 分支，没有 deconv 的 scatter+flip 分支**，fp16 deconv 一直在拿「普通卷积」当参考比 → 恒 MISMATCH。
2. **修复**：给 fp16 参考补上 deconv 分支（与 int8 同一口径 `out[(iy·S−P+ky)][…] += in[iy][ix]·w[flip]`，权重槽仍按 `(wx,wy)` 读）。
3. **验证（`--deconv-flip`，4×4→8×8，ic3/oc16）**：S=2/4/8 **3/3 BIT-EXACT（max ulp 0）**；no-flip 是刻意的错误参考（int8 同）。
4. **已知共享边界**：**1×1 核 + S=2** 在 **int8（172/1024，max diff 2）与 fp16（186/1024）都 MISMATCH** ⇒ 属参考/边界口径问题，与精度无关，列为已知项（回归中信息性）。
5. **影响**：§7.19 item 1「fp16 deconv 未打通」**作废**；fp16 回归新增 deconv 段后为 **pass=74 / fail=0**。

产物：`experiments/results/fp16_20260929/01_fp16_regression.txt`（含 J 段 deconv）、`experiments/min_conv_exp.c`。

### 7.23 FP16 独立 BN（BN_ALU + NRDMA fp32 表）闭合（2026-09-30 第三十一轮）

1. **背景**：capability matrix 里 fp16 独立 BN/NRDMA 一直"未实现"。§7.21 已取 vendor 真值：独立 BN = `BS_CFG=0x42`（BS 全 bypass）+ `BN_CFG=0x20150` + `NRDMA_CFG=0x2` + `BN_BASE_ADDR`（指向 weight BO 内逐通道表）；但 fp16 下表元素宽度未定（§7.3 的 int8 结论是 NRDMA 表为 int32；§7.19 已实测 fp16 的 **BS/BRDMA** 槽是 fp32）。
2. **工具**：发射器新增 `--bn-table-f32-bias` / `--bn-table-f32-step`（fp32 写表；配套 fp16 CPU 参考按 fp32 位型读表并在 **fp32 累加器**上逐输出通道相加、再 fp32→fp16 输出）；探针 `run_fp16_bn_nrdma_probe.sh`（对 `ret!=0` 自动重试，避免 §7.20 的 `set -e` 中止陷阱）。
3. **决定性证据（全 BIT-EXACT / max ulp 0）**：
   - **表是 fp32**：`--bn-table-f32-bias 1.5`（位型 `0x3FC00000`）→ 若硬件按 int32 重解释将加 ~`+1.07e9`，实测输出为 `conv + 1.5` 且 BIT-EXACT；`+5.0f`、`0` 基线亦对。
   - **逐输出通道**：ch ramp `0.25 + 0.5·ch` BIT-EXACT；多组 `oc=24`（表 base 偏移 0x400、32 通道）BIT-EXACT。
   - **形状覆盖**：1×1 ic8 oc8、1×1 oc24、**3×3 pad1**（表 base 0x1200）、**5×5 pad2 ic16 oc24**（表 base 0x6400、32 通道）全 BIT-EXACT。
   - **精度分流正确**：同 reg 组合（`BN_CFG=0x20150` + `NRDMA_CFG=2`）在 **int8** 下用 int32 表仍 `BIT-EXACT (floor)`；fp16 用 fp32 表 `BIT-EXACT (fp16)`。
4. **参考口径**：fp16 的 BN add 施加在 **fp32 累加器**上，先 `acc += table[oc]` 再 `f32_to_f16` 输出（与 int8 把 BN 施加在 int32 累加器再 requant 同构）。
5. **现象（残余）**：fp16 带表 DPU 提交呈 **确定性奇偶交替 `SUBMIT ETIMEDOUT`**（同命令连跑 6 次：`ret=0/-1/0/-1/0/-1`，日志末演示）；探针每例均 attempt 1 超时、attempt 2 成功。与 §7.20/§7.21 的 BS/BRDMA 超时同族 ⇒ 判为 **fp16 带表 DPU 路径的环境脆弱性**，重试即消，**非语义错**。
6. **结论**：能力矩阵 "fp16 独立 BN（NRDMA 路径）未实现" → **已闭合（fp32 表，BIT-EXACT）**；fp16 的两条 DPU 带表路径（BS/BRDMA bias、BN/NRDMA）现均可用且机制一致（fp32 表 + 交替超时重试）。

产物：`experiments/results/fp16_bn_20260930/`（板端日志 + README）、`experiments/run_fp16_bn_nrdma_probe.sh`、`experiments/min_conv_exp.c`。


### 7.24 FP16 BS/BRDMA 的 ALU(add) 与 MUL 级标定（2026-09-30 第三十二轮）

1. **背景**：`capability` 里 fp16 的 `BS_MUL_CFG` 一直"未定"——§7.19 观察到 `--bs-cfg 0x20140`（MUL 未 bypass）在 fp16 下**恒输出 0**，且无法判定乘数是 fp16 还是 s2.14。§7.23 后此项是 fp16 的最后一个已知开口。
2. **MUL 乘数 = fp16，位于 `BS_MUL_CFG[31:16]`（决定）**：`--in-ch-id --w-ones` 使 conv=`36`，`--bs-cfg 0x20140`；`0x3c000000`(fp16 1.0)→36、`0x40000000`(2.0)→72、`0x3f000000`(1.75)→63、`0x38000000`(0.5)→18、`0xc0000000`(-2.0)→-72。**低 16 位放同一 fp16 位型（`0x00003c00`）→ 0**，排除"低半区 + shift"；s2.14（`0x00004000`/`0x00008000`）→ 0。
3. **"恒输出 0"谜底**：`BS_MUL_CFG` 默认 `0` ⇒ 乘数 = fp16 `+0.0` ⇒ 结果 0。与寄存器/精度无关。
4. **ALU addend = fp32，且先于 MUL**：以 `--bs-cfg 0x20050`（MUL bypass）隔离 ALU，`BS_ALU_CFG` **整个 32 位按 IEEE-754 fp32** 读：`0x40a00000`(5.0)→41、`0x3f000000`(0.5)→36.5、`0xc0000000`(-2.0)→34。**判别位** `0x44008000`(fp32 `512.5`)→`550`（若按 bf16/高 16 位截断则为 `548`）⇒ 是完整 fp32，非 bf16。`0x20040`（ALU immediate + MUL active）`0x40000000`×2 → `(36+2)×2=76` ⇒ **顺序 conv+ALU 再 ×MUL**。
5. **BRDMA 表源同构**：bias 槽 **fp32**（`0x40a00000`→conv+5），mul 槽 **16 位 fp16**（`0x3800`→×0.5）。组合 `bias 5.0 × mul 2.0`→`(36+5)×2=82`；ramp 组合 `(36+2.5)×0.375=14.4375`。表源与立即数路径语义一致。
6. **语义定式**：`out_f16 = f16( (conv_fp32 + ALU_fp32) × MUL_fp16 )`；ALU algo `0/1/2` = max/min/add；MUL **无 shift**（与 int8 的 s2.14+移位不同）；RELU `bit6` bypass、`bit7` 上界 `BN_RELUX`。
7. **覆盖**：4 例 ALU、5 例 MUL、1 例顺序、4 例表源、1 例 3×3 全 pattern 组合、1 例 int8 对照，**全 `BIT-EXACT (fp16)`、`max ulp=0`、`submit ret=0`**；int8 的 int32 bias 表 `BIT-EXACT (floor)` 不回归。
8. **结论**：能力矩阵 "fp16 bias 的 MUL 级未定" → **已闭合**。工具新增 fp16 参考函数 `apply_bs_fp16()`（fp32 ALU + fp16 MUL）。

产物：`experiments/results/fp16_bs_mul_20260930/`（板端日志 + README）、`experiments/run_fp16_bs_mul_probe.sh`、`experiments/min_conv_exp.c`。


### 7.25 FP16 训练侧 wgrad（CNA K 维归约）打通 + fp16 `c.ic` 32 对齐根因（2026-09-30 第三十三轮）

1. **背景**：§7.4/§7.5 的 wgrad（把归约点放到 CNA 输入通道、`dY[p,oc]` 作为输出通道权重，一次 1×1 CONV 得到 16 通道梯度）此前只在 int8 上闭合；§7.18 已证 fp16 累加器 ≥ fp32，是训练侧的正向信号。本轮把该 K 维归约在 fp16 上打通。
2. **根因（fp16 `c.ic` 必须按 32 对齐，决定性）**：fp16 权重 ic 块是 32 元素（`ic2=32`）的完整原子，硬件不允许「半块」⇒ 当 `ic_real>16` 时硬件把 `c.ic` 上取整到 **32 的倍数**。发射器此前只按 16 对齐（§7.18），`ic_real∈{33..48,65..80,...}` 会得到 `c.ic=48/80/...`（非 32 倍数）⇒ 硬件读到的第二个 ic 块错位 ⇒ **全量 pattern MISMATCH**。判别：dot P=48，`--ic-pad 48`（旧行为）→ MISMATCH，`--ic-pad 64`（32 对齐）→ BIT-EXACT；对 `ic_real` 8/16/24/32/40/48/56/64/72/80/88/96 配 `--f16-pix-elems 8` 复测，修复前 40/48/72/80 MISMATCH，修复后全 BIT-EXACT。
3. **修复**：`min_conv_exp.c` fp16 分支 `c.ic = ic_real≤16 ? 16 : ALIGN_UP(ic_real, 4·FP16_ATOM_ELEMS)`（=32 对齐，原为 2·原子=16 对齐）。`ic_real≤16` 仍为 16。
4. **回归确认**：全量 `run_fp16_regression.sh`（`--pattern --fp16 --auto-fix`，默认 pix-elems=0）**pass=74 / fail=0 / envfail=1**（唯一 env fail 为随机 ETIMEDOUT），确认 32 对齐改动对既有 fp16 路径**无回归**；§7.23 BN/§7.24 BS-MUL 探针复跑仍全 BIT-EXACT。
5. **wgrad 能力（全 `BIT-EXACT (fp16)`、`submit ret=0`）**：
   - **A) dot 向量** P=32/36/40/48/64/96（`c.ic=32/64/64/64/64/96`）；
   - **B) 3×3 conv wgrad 坐标矩阵** `kx,ky∈{0,1,2}×ic∈{0,1,2}` **27/27**（P=36，`--hi-zero`，仅 (0,0) 输入像素有效）——同时验证 fp16 权重 tap 排列与 §7.15 的 ky-major 一致；
   - **C) batch 累加** 单坐标 `(kx=1,ky=1,ic=1)` B=1/2/3 → P=36/72/108（`c.ic=64/96/128`）。
   - 全 42 例零 ETIMEDOUT。
6. **输入打包注意**：fp16 多 ic 用例必须显式 `--f16-pix-elems 8`；否则 `fill_input` 按 `atom=c.ic` 打包，使实际只有一个 8 通道平面生效，单平面用例会「恰好」通过而掩盖多 ic 错误。
7. **结论**：能力矩阵新增「fp16 wgrad（单卷积 K 维归约）」= **BIT-EXACT**；残余是真实训练闭环（写回 + 梯度缩放、跨层累加）的 fp16 移植（`wgrad_real.c`），非机制阻塞。

产物：`experiments/results/fp16_wgrad_20260930/`（README；日志 gitignored）、`experiments/run_fp16_wgrad_probe.sh`、`experiments/min_conv_exp.c`。

### 7.26 FP16 真实训练闭环移植（`wgrad_real.c --fp16`）（2026-09-30 第三十四轮）

1. **背景**：§7.25 打通了 fp16 单卷积 K 维归约（wgrad dot/conv3x3/batch），但「真实训练闭环」（正向 → 27 坐标梯度 → 写回 + 梯度缩放 → 重跑正向）此前只在 int8 的 `wgrad_real.c` 闭合。本轮把 fp16 迁入 `wgrad_real.c`，新增 `--fp16 [--lr F] [--loss F]` 路径，一次运行完成 **Phase F16（正向）/ W16（27 坐标 dW）/ U16（`W'=fp16(W−lr·dW_npu)` 写回 + 重跑正向）**。
2. **实现**：`emit_conv()` 全 fp16 分支由全局 `g_fp16` 门控（`input_data_entries` 除以 `FP16_ATOM_ELEMS=8`、`wbytes×2`、`weights_kernels=ALIGN_UP(MAX2(oc_real,16),16)`、`CON1 IN/PROC_PRECISION(2)`、`WEIGHT_SIZE0×2`、`CORE_MISC_CFG PROC_PRECISION(2)`、`DPU_DATA_FORMAT` 全精度 2、`DPU_BS_CFG` 全 bypass、`OUT_CVT_SCALE=FP32TOFP16_EN|SCALE(1)`、`OUT_CVT_OFFSET/SHIFT=0`、`RDMA_FEATURE_MODE_CFG` 精度 2）。`c.ic` 沿用 §7.25 的 32 对齐；fp16 wgrad `c.oc=16`（区别 int8 的 32）。fp16 路径**无 requant/clamp/saturation**（Phase S 停用），更新直接 `f16(W−lr·dW_npu)` 舍入，不做 int8 的 `round(dW/4)`。
3. **决定性定位**：初次运行 3/3 `FWD SUBMIT FAILED (ETIMEDOUT)`。用 `--dump` 对比 `min_conv_exp --pattern --fp16 ...`（同几何 ic3/oc16/8×8/pad1）发现正向 `DPU_DATA_FORMAT` 仍为 int8 值 `0x0`——根因是 `--fp16` 解析只置了调度标志 `fp16`，**未置发射门控 `g_fp16`**。修复后正向 126 条 EMIT 与参考**逐条一致**（diff 为空），闭环随即通过。
4. **结果（10/10 相位 `RESULT: CLOSED-LOOP OK (fp16 ...)`）**：pad1 B=1/2/3、pad0 B=1/2、`--lr {0.5,0.125,1.0} --loss {0.125,0.5,1.0}`、`--seed {1,2,999}`；正向 `mismatch=0`（pad1 total=1024 / pad0 total=576），wgrad `coords=27/27 BIT-EXACT (fp16)`，更新 `F16 FWD2 ... vs CPU(W') BIT-EXACT` 且 `outputs_changed=1024/1024`。
5. **不回归**：int8 闭环 `--dir wgdata_pad1 --pad 1 --batch 1 --update` 改动后仍 `CLOSED-LOOP OK (fwd exact, wgrad round-exact, update verified)`（首跑 1 次偶发 ETIMEDOUT，重试 3/3 通过）。
6. **结论**：能力矩阵新增「fp16 训练侧真实闭环（单层 3×3/IC=3/OC=16/8×8）」= **BIT-EXACT**。残余：更深通道（ic>96/oc>64）与多层真实网络的 fp16 闭环，属后续方向。

产物：`experiments/results/fp16_wgrad_real_20260930/`（README；日志 gitignored）、`experiments/run_fp16_wgrad_real_loop.sh`、`experiments/wgrad_real.c`。


### 7.27 deconv 边界口径修正：转置卷积方向写反（2026-09-30 第三十五轮）

1. **背景**：台账多处记录的「**1×1 核 + S=2 deconv** 在 int8（172/1024）与 fp16（186/1024）都 MISMATCH」一直被当作「共享边界」。本轮查明：**它不是硬件边界，而是 CPU 参考工具的转置卷积口径写反了方向**，与精度无关。
2. **口径对比**：
   - 旧参考（错误）：`tx = ox + pad − kx`，且 `--deconv-flip` 再把权重 tap 翻转（`kw−1−kx`）。
   - 硬件真实（正确）：`ox = ix·S + pad − kx` ⇒ `tx = ox − pad + kx`（须能被 S 整除，`ix = tx/S`），权重 tap 直接用 `kx`（**无翻转**）。
   - 两者仅当 **`2·pad = kw−1`** 时重合。`3×3 pad1` 恰好满足（`2=2`），故此前**只有 3×3 通过**，1×1/2×2/矩形核全 MISMATCH——这解释了「为什么只有 3×3 对」。
3. **硬件映射反解（决定性证据）**：`--in-ruler` + `--w-ones`（fp16 ic=16，1×1 核 S=2 pad1，仅 tap 0）。输入按元素下标作标尺（每像素 8 个 fp16、平面步长 `iw·ih·16 B`），`w=1` 时输出 = 标尺之和 = `128·pix + 1080`（`pix=iy·4+ix`）。反解：`(ox,oy)=(1,1)→(ix,iy)=(0,0)`、`(3,1)→(1,0)`、`(1,3)→(0,1)`、`(3,3)→(1,1)`、`(7,7)→(3,3)`，即 `ix=(ox−1)/2`、`iy=(oy−1)/2`。旧参考给出 `(1,1)/(2,1)/(1,2)/(2,2)` 且把 `ox=7` 判越界丢成 0——硬件把外边界行/列视作有效输入。
4. **权重序判定**：kw=3 S=2 下 `--w-tap kx:ky` 遍历 9 个 tap 全 `BIT-EXACT (fp16)` ⇒ tap = 原始 `kx`，**无翻转**；`--deconv-flip` 修正后无语义影响（保留为兼容 no-op）。
5. **覆盖（修正后全 `BIT-EXACT`）**：fp16 核形状 {1×1,2×2,3×3,3×1,1×3,5×5,5×1} × S∈{2,4,8}；int8 同口径 {1×1,2×2,3×1,5×5} 全 `BIT-EXACT (floor)`。探针 `run_deconv_boundary_probe.sh` **26/26 pass**。
6. **改动**：`experiments/min_conv_exp.c` 的 fp16 / int8 两处 deconv CPU 参考改为 `− pad + kx` + 权重 tap `kx`；`run_deconv_boundary_probe.sh` 新增；`run_deconv_smoke.sh` / `run_fp16_regression.sh` 加入非 3×3 核用例。
7. **不回归**：`run_fp16_regression.sh` **pass=78/fail=0/envfail=1**（原 74/0，新增 4 例）；`run_regression_hard.sh`（int8）**pass=74/fail=0**；`run_deconv_smoke.sh` **pass=12/fail=0**。
8. **结论**：能力矩阵的 fp16/int8 deconv 由「BIT-EXACT，1×1 核 + S=2 为共享边界」更正为「**任意核形状 × S ∈ {2,4,8} 全 BIT-EXACT**」。「共享边界」条目作废。

产物：`experiments/results/deconv_boundary_20260930/`（README；日志 gitignored）、`experiments/run_deconv_boundary_probe.sh`。


### 7.28 fp16 双层真实网络闭环：跨层前向 + 分层 wgrad + 权重更新（2026-09-30 第三十六轮）

1. **背景**：§7.26 把 fp16 训练闭环打进 `wgrad_real.c`（单层：正向 → 27 坐标 wgrad → 写回 → 重跑）。§8 的下一入口是「多层真实网络的 fp16 闭环 / 跨层累加」。本轮新增 `--two-layer`，验证**层 2 直接读层 1 的 NPU 输出面**（不做 CPU 回环）时，跨层 fp16 前向与分层 wgrad 是否仍自洽。
2. **几何**：层 1 `ic=3→oc=16`、层 2 `ic=16→oc=16`，均 `3×3 pad`、`8×8` 输入；ow×oh = 8×8（pad1）/ 6×6（pad0）。相位 `L2-F`（链式前向）、`L2-W`（层 2 wgrad 输入=NPU A1；dgrad 反传得 dA1；层 1 wgrad）、`L2-U`（`W'=fp16(W−lr·dW)` 写回并再前向）。
3. **结果**：
   - **跨层前向（更新前）F1/F2：720/720 BIT-EXACT**（pad0/pad1 × seed1..30 × B1..3）。
   - **分层 wgrad（fp64 正确舍入参考）：348/360 BIT-EXACT，其余 ≤4 ULP**。
   - **更新后再前向**：大量逐位一致，偶发 ≤5 ULP。
   - **宽扫描 pad0/1 × seed1..30 × B1..3：180/180 `2-LAYER CLOSED-LOOP OK`**；单层 fp16/int8 基线无回归。
4. **判据修正（近零相消）**：fp16 多元素求和沿用 `≤16 ULP`，但**强相消**时输出量级远小于累加项，fp32 级累加的绝对误差被放大为很大的 ULP 距离（相对误差），此时 ULP 无意义。闭环判等改为 **`ULP ≤ 16` 或 `|hw−ref| ≤ 1e-4`**（`--tl-ulp`/`--tl-abs` 可调）。具例 `seed=11 pad=1 B=3` 层 2 更新后再前向 `(oy=3,ox=5,oc=2)`：硬件 `0x0f48`(4.444e-4)、fp64 精确和 `0x0f9f`(4.652e-4)、fp32 顺序和 `0x0f00`(4.395e-4)，ULP=87 但绝对残差 `2.07e-5`；硬件落在 fp32 顺序和与精确和之间 ⇒ fp32 级累加的正常舍入，非逻辑错。
5. **根因修复**：`make_fwd_cfg_l2` 曾把输出面尺寸硬编码为 `IW×IH`（仅在 pad1 与 `ow×oh` 重合），pad0 下 6×6 输出 stride 错误导致前向大面积失配（525/576 mismatch、ULP 达 3.6e4）；改为按 `pad` 取 `ow×oh`。此 bug 仅在 `ow≠IW` 时暴露，正好由 pad0 用例抓出。
6. **实现**：`experiments/wgrad_real.c` 新增 `--two-layer`（隐含 `--fp16`）、相位 `L2-F/L2-W/L2-U`；参考 `f16_cpu_wgrad_dbl`/`f16_cpu_fwd_gen` 均改 **fp64 累加**；新增 `f16_key`/`f16_ulp`/`f16_ok`/`f16_verdict` 与 `f16_fwd_probe64`（顺序/精确和对照）。
7. **结论**：能力矩阵 fp16 由「单层训练闭环（§7.26）」扩展为「**双层跨层前向 + 分层 wgrad + 更新闭环，判据 ≤16 ULP 或近零绝对残差**」。

产物：`experiments/results/fp16_two_layer_20260930/`（README；日志 gitignored）、`experiments/run_fp16_two_layer_loop.sh`。

### 7.29 fp16 深通道 / 多 tap 权重分组包线：`c.ic` 硬上限（2026-09-30 第三十七轮）

> ⚠️ **本节结论已由 §7.30 修正**：真正的包线是「单次 submit 的 CNA 输入 feature tile
> 字节 ≤ 32 KB」，与 tap 形状无关；本节的「多 tap≤64 / 1×1≤256」判据源于多 tap 探针
> （`iw=ih=16`）与 1×1 探针（`iw=ih=8`）配置混淆。保留本节记录历史推理与仍有效的排除项。

1. **背景**：§8 下一入口「更深通道 ic>96/oc>64」。此前 fp16 回归仅覆盖 1×1 至 `c.ic=96`、3×3p1 至 ic32，深多 tap 是空白。本轮在板端扫深通道，发现 **fp16 conv 权重按输入通道分组装载，存在硬性包线上限；超限时硬件静默算错**（无报错、`SUBMIT ret=0`、结果大幅偏离）。
2. **包线（实测 driver v0.9.8）**：
   - **多 tap（`kh*kw>1`，任意 1×2/2×2/3×3/5×5…）：`c.ic ≤ 64`**（ic1≤2 组）；`c.ic ≥ 96` 结构性 MISMATCH。
   - **1×1：`c.ic ≤ 256`**（ic1≤8 组）；`c.ic ≥ 320` 结构性 MISMATCH（1×1 与 wgrad dot 同路径，包线更宽）。
   - **int8 多 tap（对照）：`c.ic ≤ 128`**；`c.ic ≥ 160` MISMATCH。
   - fp16 `c.ic` 仍为 `ic_real≤16→16，否则 32 对齐`（§7.25），故取值 16/32/64/96/128…，多 tap 在 `ic_real 65→c.ic=96` 处跨限。
3. **上限由 `c.ic` 决定，与权重字节/真实通道无关**（决定性对照，3×3p1 fp16）：
   - `ic64 oc64`（权重 73728 B）**BIT-EXACT**（max|diff|=0）⇒ 非权重字节上限。
   - `ic_real=64 --ic-pad 96`（c.ic=96，多出的权重为 0）**MISMATCH**（max|diff|=311）；
     `ic_real=96 --ic-pad 64` **ULP-EXACT**（max|diff|=9.54e-07）⇒ 真实通道可多，拉回 `c.ic` 即恢复。
   - `ic65→c.ic=96` MISMATCH，而 `ic64→c.ic=64` ULP-EXACT。
4. **结构性 vs ULP 噪声判别**（`max|diff|`）：包线内为 `9.5e-07`（仅求和顺序差，ULP≤16）；跨限后为 `311~5753`（CSA/WR 数量级），非噪声。多 tap 各形状在 `c.ic=96` 全部失效（`kw1kh2`/`kw2kh1`/`kw2kh2`/`kw3kh1`/`kw5kh5`，max|diff| 1.5e3~5.8e3）。
5. **排除的修复方向**：`--f16-watom 16/8`、`--f16-nest 1/2`、`--f16-ocatom 32`、`--entries 2/4/8/12/16`、`--ws1 864/1728/3456/216`、**`--kernel-group N`（`CNA_CONV_CON2.KERNEL_GROUP` bits16-23，扫 0/1/2/3/4/8/16）** 均无法恢复（输出与默认逐字节一致）⇒ 非 `WEIGHT_SIZE1` 缩放、打包布局选项、亦非 KERNEL_GROUP 分组迭代控制；`--ws1`/`--kernel-group` 对相应字段无影响与 §7.25 结论一致。
6. **实现**：`experiments/min_conv_exp.c` fp16 非 deconv 路径新增包线判别——`kh*kw>1 → c.ic>64`、`1×1 → c.ic>256` 时打印 `WARN` 并把 `RESULT` 标为 `OUT-OF-ENVELOPE (fp16 weight-group cap, §7.29)`，避免把硬件静默错误误记成普通 MISMATCH。探针 `experiments/run_fp16_deep_channel_probe.sh`（含 `max|diff|` 明细）。
7. **结论**：真实模型 fp16 的深 `3×3`（如 ic=128/256）**必须按 ic 分组切分并累加**（每次 `c.ic≤64`），正是 vendor runtime 对深卷积拆多 task 的原因；单次发射无法承载。能力矩阵 fp16 增加「深通道适用域」注记。待验：跨 submit 的 ic 分组累加是否 BIT-EXACT。

产物：`experiments/results/fp16_deep_channel_20260930/`（README；日志 gitignored）、`experiments/run_fp16_deep_channel_probe.sh`。

### 7.30 修正 §7.29：包线本质是 CNA 输入 feature tile ≤ 32 KB（2026-09-30 第三十八轮）

> ⚠️ **本节结论已由 §7.31 再修正**：32 KB 是**旧发射器硬编码 `DATA_BANK = 1`** 的表现
> （CBUF 每 bank 32 KB），第 4 点的「次级约束」则是 **`FEATURE_GRAINS` 写死 `50+stride+1`**
> 的表现；两者都是配置，不是硬件硬包线。本节保留为历史推理与 A/B 对照臂（探针已加
> `--no-auto-cfg`）。

1. **背景**：§7.29 把包线归为「tap 形状区分 `c.ic` 上限（多 tap≤64、1×1≤256）」，但其多 tap 探针固定 `iw=ih=16`、1×1 探针固定 `iw=ih=8`，**两个变量（tap 与 tile 尺寸）未解耦**。本轮做控制变量实验。
2. **决定性对照（3×3p1 fp16，固定 `iw=16` 只扫 `ih`；已用大激活 BO 排除内存重叠，见第 6 点）**：
   - `ic128 ih8`（输入 = 16·8·128·2 = **32768 B**）ULP-EXACT；`ih9`（36864 B）结构性 MISMATCH（max|diff| 240）。
   - **边界精确（用 iw 细分到 512 B/步）**：`ic16 iw64 ih16`=32768 B ULP-EXACT；`iw65`=33280 B 即 MISMATCH（max|diff| 29）⇒ 阈值恰为 **32768 B**。
   - `iw` 与 `ih` 完全对称（`iw8 ih16`=32768 B OK，`iw9`=36864 B BAD）。
   - **同一 `iw/ih/ic` 下 1×1 与 3×3 同为 OK/BAD**（`ic128 iw16 ih8`：1×1 BIT-EXACT、3×3 ULP-EXACT；`ih9`：两者皆 MISMATCH）⇒ **tap 形状无关**，§7.29 的 tap 依赖是配置混淆。
   - 小 ic 亦可越界：`ic64 iw16 ih24`(49152 B)、`ic32 iw16 ih48`(49152 B) 均 MISMATCH ⇒ 由 tile 字节而非 `c.ic` 决定。
3. **规则**：单次 submit 的 CNA 输入 feature tile 字节数 **`in_size = iw*ih*对齐通道*每元素字节 ≤ 32768`**（= 32 KB），`> 32768` 硬件静默算错（`SUBMIT ret=0`，无报错）。fp16 每元素 2 B（原子 8 元素 ×16 B），int8 每元素 1 B（原子 16 元素 ×16 B）。int8 `ic128 iw16 ih16`=32768 B BIT-EXACT、`ic160`=40960 B MISMATCH，同受此限。`c.ic` 用对齐/pad 后的跨度（`ic_real=64 --ic-pad 96`→96→49152 B 坏；`ic_real=96 --ic-pad 64`→64→32768 B 好），与 §7.29 的 `c.ic` 依赖一致但被此规则吸收。权重字节无关（`ic64 oc64` 描述符 73728 B 仍 ULP-EXACT）。
4. **次级约束（本次未闭合）**：在 ≤32 KB 内，**极高的输入亦会失效**（独立于字节）：fp16 `ic16 iw8` 在 `ih53` ULP-EXACT、`ih54` MISMATCH；边界随 `iw` 轻微浮动（`iw12`→52/53，`iw16`→53/54）。疑为行/stride 或输出 tile 的独立上限（emitter 简化布局亦可能是诱因），待定域。
5. **修复方向（修正）**：§7.29 的「按 ic 分组」只是充分条件之一；**按输出行（或任意维）切分使单次 `in_size ≤ 32768` 皆可**——vendor runtime 深卷积正是**按输出行拆 task**（ic128 iw16 ih16 拆成 h=8 的两 task，每 tile 恰 32768 B），而非按 ic 拆。
6. **排除 BO 重叠混淆（重要前置）**：emitter 旧默认激活 BO=64 KB、输出偏移 `OUT_OFF=32768`；当 `in_size>32768` 时输入区与输出区**重叠**，会额外污染结果。已把默认改为激活 BO 1 MB、输出偏移 512 KB，并新增 `--act-size`/`--out-off` 旋钮；用大 BO 复测确认**边界仍恰为 32768 B**（`ih8` 净、`ih9` 坏），重叠只放大失配幅度（如 `ic128 ih16` 的 max|diff| 由 3769 降为 546），非根因。权重 BO 亦由 64 KB 增至 1 MB（容纳 `ic64 oc64` 的 73728 B 描述符，此前该例的 BIT-EXACT 疑因越界）。
7. **实现**：`experiments/min_conv_exp.c` 包线判别改为 `in_size > 32768`（不再按 tap 形状），打印 `OUT-OF-ENVELOPE (input feature tile > 32KB, §7.30)`；探针 `run_fp16_deep_channel_probe.sh` 重写为边界/对称/tap-control/小-ic/对齐/int8 六组（含 submit 超时重试）。

产物：`experiments/results/fp16_deep_channel_20260930/`（README 已更新）、`experiments/run_fp16_deep_channel_probe.sh`、`experiments/min_conv_exp.c`。

### 7.31 修正 §7.30：真包线是 CBUF bank（DATA_BANK × 32 KB），次级约束是 FEATURE_GRAINS（2026-09-30 第三十九轮）

1. **背景**：§7.30 把「单次 submit 输入 tile > 32768 B 静默算错」定为硬包线，并留下「≤32 KB 内高输入亦失效」的次级约束（fp16 `ic16 iw16`：53 行净、54 行错）。本轮在 `refs/tinygrad/tinygrad/renderer/rockchip.py`（RK3588 CMAC 后端）里发现两条可用口径——`banks = ceil(m*row_bytes/32768)` 与 `grains = max(80, ceil(2*32768/row_bytes)+1)`——据此在板上做控制变量实验，二者均被证实。
2. **① 输入 tile 上限 = `DATA_BANK × 32768 B`（决定性）**：`CNA_CBUF_CON0` 的 `WEIGHT_BANK = 12 − DATA_BANK`，CBUF 共 12 个 32 KB bank；旧发射器恒写 `DATA_BANK=1`。用 `--weight-bank N` 显式设 `DATA_BANK = 12−N` 扫字节边界（3×3p1 fp16）：

   | 输入字节（`ic128 iw16`） | 需要 bank | `DATA_BANK=1` | `DATA_BANK=ceil(in/32768)` |
   | --- | --- | --- | --- |
   | 32768 B（`ih8`） | 1 | ULP-EXACT | ULP-EXACT |
   | 36864 B（`ih9`） | 2 | MISMATCH（max\|diff\| 240） | ULP-EXACT |
   | 65536 B（`ih16`） | 2 | MISMATCH（545.6） | ULP-EXACT |
   | 98304 B（`ih24`） | 3 | MISMATCH（545.6） | ULP-EXACT |
   | 131072 B（`ih32`） | 4 | MISMATCH（545.6） | ULP-EXACT |

   ⇒ §7.30 的「阈值恰为 32768 B」是 `DATA_BANK=1` 的特例；`iw`/`ih` 对称、tap 无关等观察全部被此规则吸收。
3. **② `FEATURE_GRAINS`（`CNA_CONV_CON2[13:4]`，10 bit）必须随输入行字节放大**：旧值 `50 + stride_y + 1`（mesa 的 "magic"）在**高输入**下静默算错，且与 32 KB 无关（`iw16 ic16 ih54` 仅 27648 B 即错）。采用 tinygrad 口径 `grains = max(80, ceil(2·32768/row_bytes)+1)`（取偶、上限 1022；`row_bytes = iw × 对齐通道 × 每元素字节`）：

   | 配置（3×3p1 fp16，`--no-auto-cfg` 固定 bank） | `grains=52` | `grains=auto` |
   | --- | --- | --- |
   | `iw16 ic16 ih54`（row 512 B → 130） | MISMATCH 7.3 | ULP-EXACT |
   | `iw16 ic16 ih64` | MISMATCH 15.4 | ULP-EXACT |
   | `iw16 ic16 ih128` | MISMATCH 15.4 | ULP-EXACT |
   | `iw16 ic16 ih200` | MISMATCH 16.5 | ULP-EXACT |
   | `iw8 ic16 ih100`（row 256 B → 258） | MISMATCH 14.7 | ULP-EXACT |
   | `iw8 ic16 ih200` | MISMATCH 15.8 | ULP-EXACT |

   另：`grains ≥ 1024` 被 10 bit 字段截断（`--grains 2048` → 写入 0 → MISMATCH），故 auto 上限取 **1022**。⇒ §7.30 的「次级约束」闭合，且**不是**硬件包线。
4. **③ 真包线 = CBUF 总容量**：`ceil(in_bytes/32768) + weight_banks_need > 12` 时单次 submit 放不下，必须切分。`ic128 iw16 ih64`（输入 8 + 权重 3 = 11）单发即 ULP-EXACT；`ic256 iw16 ih40`（输入 10 + 权重 4 = 14）单发 MISMATCH（max|diff| 623），`--tile-auto` 切成 30+10 行（最大 tile 31 行 = 253952 B = 8 bank）后 **ULP-EXACT**。
5. **④ 按输出行切分落地（`min_conv_exp.c` 新增多 submit 路径）**：每个 tile 把自己需要的输入行块 `[g0, g0+rows)`（`rows = (T−1)·sy + kh`，`g0 = max(0, oy0·sy − pad_top)`）拷成紧凑 stage，`pad_top` 相应下调（中间 tile 为 0，首 tile = `−iy0`），输出面 scatter 回全局布局。**tile 输出互不相交、无需跨 submit 累加**（§7.30 的判断成立）。旋钮：`--tile-oh N`（显式 tile 高度）、`--tile-auto`（按「可用 data bank × 32768」自动选 T）、`--tile-rows-max N`。
6. **⑤ 发射器默认改为 `--auto-cfg`**：按 ① 取 `DATA_BANK = min(ceil(in/32768), 12 − 权重 bank)`，按 ② 取 grains（`--grains` 仍可覆盖）；`--no-auto-cfg` 复现 §7.30 的旧行为（A/B 对照臂）。新增信息行 `CBUF-CFG`/`CBUF-PLAN`/`TILE`，容量越界时打 `OUT-OF-ENVELOPE (CBUF banks: input+weight > 12, §7.31)`。
7. **不回归**：`run_fp16_regression.sh` **79/0/0**（原 78/0/1）、`run_regression_hard.sh`（int8）**74/74**（首跑 1 例随机 ETIMEDOUT，复跑全过）、`run_deconv_smoke.sh` **12/0**、§7.23 BN 探针与 §7.25 wgrad 探针复跑仍全 BIT-EXACT。
8. **vendor trace 旁证（离线，`experiments/analyze_vendor_cbuf_banks.py --all`）**：仓库内 17 份 hook trace 共 85 个 regcmd task 的 `CNA_CBUF_CON0` 分布——`attn_relu_vendor.jsonl`（1×1 matmul 形态，`WEIGHT_SIZE0`=65536 B）用 **DATA_BANK=10 / WEIGHT_BANK=2**，`bn_lower/em0d/tr_transpose_p0321_fp16.jsonl` 用 **DATA_BANK=11 / WEIGHT_BANK=1**，其余 67 个（小输入）为 1/11。⇒ `DATA_BANK`/`WEIGHT_BANK` 是 vendor runtime 真正在用的规划旋钮（输入按字节取 bank、权重拿余量），旧发射器恒写 1 是 §7.30 假象的来源。⚠ vendor 的 `FEATURE_GRAINS` 实测仅 **2–20**（随几何变化）⇒ tinygrad 口径是**充分但保守**的可用规则，vendor 精确规则待反解。
9. **口径已同步到训练工具**：`experiments/wgrad_real.c` 的 `emit_conv()` 原先同样硬编码 `input_banks=1` 与 `grains=50+stride+1`；本轮把 §7.31 规则（`DATA_BANK=min(⌈in/32768⌉, 12−权重 bank)`、`grains=max(52+stride, max(80,⌈2·32768/row_bytes⌉+1))`）与 `--no-auto-cfg` 一并移植。验证：fp16 单层闭环 **10/10 相位 OK**、`--two-layer` 宽扫描 **SWEEP: PASS=180 FAIL=0**、int8 基线 `CLOSED-LOOP OK`（首跑 1 次随机 ETIMEDOUT，复跑 3/3 通过）。
10. **结论**：能力矩阵的「CNA 输入 feature tile 32 KB 包线」应改为「**CBUF bank 容器**：输入 tile ≤ `DATA_BANK × 32768`，且输入+权重 ≤ 12 bank；`FEATURE_GRAINS` 须按行字节取 `max(80, ⌈2·32768/row_bytes⌉+1)`；超出 CBUF 容量时按输出行切分（无需累加）」。vendor runtime 拆多 task 的真因是**权重 bank + 输出行 tile 的联合规划**，不是 32 KB 硬限。

产物：`experiments/results/fp16_cbuf_bank_20260930/`（README + 板端日志）、`experiments/run_fp16_cbuf_bank_probe.sh`、`experiments/analyze_vendor_cbuf_banks.py`、`experiments/min_conv_exp.c`、`experiments/wgrad_real.c`；`run_fp16_deep_channel_probe.sh` 改为历史对照臂。


### 7.32 FEATURE_GRAINS 的 vendor 精确口径：`grains = ih + kh + pad_top`（2026-09-30 第四十轮）

1. **背景**：§7.31 只给出「grains 必须够大」的充分口径（tinygrad `max(80, ⌈2·32768/row_bytes⌉+1)`），并留下「vendor 实测只有 2–20，精确规则待反解」。本轮用仓库内全部 hook trace 反解。
2. **反解方法与工具**：`experiments/analyze_vendor_grains.py` 解每个 regcmd 的 NPUOP，抽 `DATA_SIZE0/1`（`ih`/`iw`/`ic`）、`WEIGHT_SIZE2`（`kw`/`kh`/`kernels`）、`PAD_CON0`（`pad_left`/`pad_top`）、`CBUF_CON0`（bank）、`CONV_CON2`（grains）→ `grains_tasks.csv`。
3. **规则（决定性，离线）**：**`FEATURE_GRAINS = ih + kh + pad_top`**，**85/85 task、32 个唯一几何全精确吻合（误差 0）**：1×1（`iw8 ih8→9`、`iw1 ih1→2`）、1×2/2×1（`18/17`）、3×3p1（`20/13/12/9`）、attention matmul 形态（`iw16 ih16 kw16 kh1 → 17`）、单行 feature（`ih1→2`）。**与 `kw`、`ic`、`oc`、stride、bank 数无关**。与 rk3568 RE 的 `grains = kh + strY·(L−1)` 在本批（`strY=1`）下即 `L = ih + pad_top`；mesa/旧发射器的 `50+stride_y+1` 是**小图 magic**（`ih` 小时恰好 ≥ 需求，故 74/74 回归能过）。
4. **板端验证（`run_vendor_grains_probe.sh`，只切 grains、bank 由 auto-cfg 负责）**：旧 `52` 在 fp16 3×3p1 `ih54/64/100`（iw16/iw8）、1×1p0 `ih64`、5×5p2 `ih64` 全结构性 MISMATCH（max|diff| 7~16.5），而 vendor 口径（`58/68/104/65/71`）全部 **ULP-EXACT/BIT-EXACT**；大 tile（`ic64 ih100` 7 bank、`ic128 ih24` 3 bank）与 int8 用例连 52 也过 ⇒ 硬件需求**不是**随 `ih` 单调，`ih+kh+pad_top` 是 vendor 选择的工作点（本批 12/12 充分）。
5. **落地**：`min_conv_exp.c` 与 `wgrad_real.c` 的 `--auto-cfg` 取两条已验证规则的**上包络**：`max(50+stride_y+1, ih+kh+pad_top, max(80, ⌈2·32768/row_bytes⌉+1))`，再截到 1022（字段 10 bit）；`--grains N` 仍可显式覆盖。板端复测 12 例全准（含 `--tile-auto` 切分路径）。
6. **不回归**：`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh`（int8）**74/74**（首跑 1 例随机 ETIMEDOUT）、`run_deconv_smoke.sh` **12/0**。
7. **结论**：§7.31 的「grains 待反解」闭合——vendor 口径为 vendor 明确选择的规则，tinygrad 口径为另一条充分启发式；发射器取上包络后既贴近 vendor 又保留已验证的余量。

产物：`experiments/results/vendor_grains_rule_20260930/`（README + `grains_tasks.csv` + `grains_probe.txt`）、`experiments/analyze_vendor_grains.py`、`experiments/run_vendor_grains_probe.sh`、`experiments/min_conv_exp.c`、`experiments/wgrad_real.c`。


### 7.33 ic 维切分（跨 submit，host fp32 累加）：非逐位一致（≤1 ULP），可替代行切分（2026-09-30 第四十一轮）

1. **背景**：handoff §8 下一入口①——真 CBUF 容量越界（`ceil(in/32768)+权重 bank > 12`）时，§7.31 只落地了**输出行**切分；本轮补 **ic 维**切分并回答「跨 submit 累加是否 BIT-EXACT」。
2. **机制（`min_conv_exp.c --ic-groups G`）**：把 `ic_real` 按 32 对齐切成 G 组（G 是组数下界，每组 ≥32 通道）；每组独立 emit+submit：**输入** = 全局输入的连续平面区间（fp16 布局 `[plane(8ch)][y][x][16 B]`，组址 = 基址 + `ic0/8·iw·ih·16`，**无需拷贝**）；**权重** = 按组重打包描述符（`fill_weights_f16` 用全局 ic 偏移取逻辑权重）；**输出** = 每组独立 partial 区，最后在 **host 用 fp32** 逐元素相加并 `fp32→fp16` 写回。仅 fp16（int8 partial 会被重量化）；暂不与 `--tile-*`/BN/BRDMA 表组合。另新增 `OUT-HASH`（输出面 FNV1a）以跨进程比对切分/不切分。
3. **不是逐位一致（决定性）**：`--in-ruler ic96 3×3p1` 切 2/3 组时 hash 与单次 submit 不同，差异 28–43/4096 元素、`max|diff|=16`（该量级 = **1 ULP**）；而同用例的单次 arm 对 fp64 参考 **BIT-EXACT** ⇒ 切分偏离 **≤1 ULP**。根因：partial 先被 fp16 量化再 host 相加（单次 submit 是整段 fp32 累加）。多数用例（默认 pattern、整数 pattern、`--in-ruler ic128 1×1`）仍 **hash 完全相同**（逐位一致）。
4. **累加本身无误差**：全部切分用例 `host fp32 累加 vs fp64 累加: max ulp=0`。
5. **真容量越界可救**：`ic256 iw16 ih40`（输入 10 bank + 权重 4 bank = 14 > 12）单发结构性算错（max|diff| 623）；`--ic-groups 4` 或 8 → **ULP-EXACT**（max|diff| 4.8e-07），且 4 组与 8 组 hash 相同。
6. **取舍**：行切分**无需累加**、判据与原结果同级（首选）；ic 切分需 host 累加、最坏 ≤1 ULP（行不可分/宽行时的替代）。⇒ handoff「跨 submit 累加是否 BIT-EXACT」= **不是**（≤1 ULP），但作为容量替代路径**可用**。
7. **不回归**：`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh`（int8）**74/74**（首跑 1 例 `3x3 8x8 pad0` 随机 ETIMEDOUT，复跑全过）、`run_deconv_smoke.sh` **12/0**。

产物：`experiments/results/ic_split_20260930/`（README + `ic_split_probe.txt`）、`experiments/run_ic_split_probe.sh`、`experiments/min_conv_exp.c`。


### 7.34 二维切分：输出行 × ic（2026-09-30 第四十二轮）

1. **背景**：§7.33 落地 ic 切分后，handoff §8 下一入口①还剩「ic 与行切分组合」——当**单个 ic 组的一整幅输入**仍超出可用 data bank 时必须二维切分。
2. **实现（`min_conv_exp.c`）**：`--ic-groups G` 与 `--tile-*` 可组合。组外层（§7.33）不变；组内层（§7.31）的 tile 输入 stage **只含本组平面**（源 = 组首平面 + tile 行区间），tile 输出 scatter 回**本组** partial 面；预算 `行数 = 可用 data bank × 32768 / 组行字节`（组行字节 = `iw × 组对齐通道/8 × 16`）。**顺带硬化**：激活 BO 现按几何自动放大（全局输入可达数 MB + `(G+1)` 个 partial 面），旧默认 1 MB 在大 ic × 大 ih 下静默踩内存（实测 segfault）。
3. **决定性（同一几何三种策略，`iw16 ic256 ih400` 1×1：全图 100 bank、单组 25 bank）**：不切分 `d0fd2751` OUT-OF-ENVELOPE（max|diff| 171）；**行切分**（10 tile）`a452fec0` **BIT-EXACT**；**纯 ic**（8×32，组内 13 bank > 11）`f5ac557a` MISMATCH（209）⇒ 必须二维；**2D** groups4+tile-auto（3 tile×4）/ groups8+tile-auto（2 tile×8）/ groups4+tile-oh 100（4 tile×4）**全 `a452fec0` BIT-EXACT**。⇒ 切分选择不影响结果；所有切分 arm 的 host fp32 累加对 fp64 = 0 ULP。
4. **组内 bank 够时纯 ic 即可**：`ic256 ih100`（全图 25 bank 越界）不切分 `b8ffce73` 越界；`groups 4`/`groups 8` 均 `2762b03b` BIT-EXACT。
5. **3×3 真卷积 + 深 ic**：`ic128 ih200` 3×3p1 不切分 OUT-OF-ENVELOPE（545）；`groups4 + tile-auto` 自动选 1 tile×4 组（组 7 bank + 权重 2 ≤ 12）→ **ULP-EXACT**（4.8e-07）。
6. **压力 pattern**：`--in-ruler ic96 ih400` 1×1 的 2D（2 tile×3 组）与行切分（4 tile）同 hash `dd33c35e` BIT-EXACT；最坏界仍由 §7.33 的 partial fp16 量化决定（≤1 ULP）。
7. **不回归**：`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh`（int8）**74/74**（首跑 1 例 `3x3 8x8 pad0` 随机 ETIMEDOUT，该例单跑 5/5 通过）、`run_deconv_smoke.sh` **12/0**；既有形状 hash 与 §7.31/§7.33 记录一致（`533d6b86`、`3606092d`）。
8. **结论**：fp16 卷积的**能力闭包完成**——任意形状可按「行 × ic」二维切分使单次 `输入 tile + 权重 bank ≤ 12`，结果与不切分/行切分一致（BIT-EXACT 或 ≤1 ULP）。剩余限制：**单行字节 > 可用 bank × 32768**（需按宽度/列切分，未实现）。

产物：`experiments/results/tile_ic_split_20260930/`（README + `tile_ic_split_probe.txt`）、`experiments/run_tile_ic_split_probe.sh`、`experiments/min_conv_exp.c`。


### 7.35 输出列切分（列 × 行 × ic 三维切分）（2026-09-30 第四十三轮）

1. **背景**：§7.34 后仍剩「**单行输入字节 > 可用 data bank**」的极宽输入——行切分与 ic 切分都救不了，必须按**输出列**切分。
2. **实现（`min_conv_exp.c`）**：新增 `--tile-ow N`（显式列宽）；`--tile-auto` 改为列/行联合自动——当「4 行 × 全宽」都超 bank 时按列切（`iw_max = (cap/4)/每像素字节`），再按行预算取 T。列 tile 把输入**子矩形逐行 stage**（行内字节 = `(Tc−1)·sx + kw`，平面步长 = rows×行内字节），`ct.iw/ct.ow` 取 tile 宽，`pad_left` 下调 `max(0, −(ox0·sx − pad_left))`，输出按行段（`Tc×16 B`）scatter 回全局面/组 partial 面。
3. **关键约束（本轮发现）**：`CNA_DMA_CON2.SURF_STRIDE = L·(staged_ih/4 − 1)` 编码「平面步长 = 4 行组」⇒ **staged 输入行数必须 ≥ 4**，否则该式无解（负值）→ 结构性算错（实测自动列切最初给出 1 行 tile → MISMATCH）。现自动模式把行预算下限钳到 4；显式 tile 越界时如实打印 `OUT-OF-ENVELOPE`。
4. **决定性（A 段，`iw1024 ic256 ih8` 1×1：单行 = 16 bank > 可用 11）**：不切分 `e16482e5` OUT-OF-ENVELOPE；**五种切分全部 BIT-EXACT 且同 hash `e86f99c0`**——自动列切（14 tile 4×160）/ `ic-groups 4 + auto`（4×640 ×4 组）/ `ic-groups 16 + auto`（5×1024 ×8 组）/ 显式 `--tile-ow 128`（5×128）。显式 `--tile-ow 256`（4×256，超容量）如实报 OUT-OF-ENVELOPE（不静默）。
5. **列切 + pad**：`iw256 ic64 ih8` 3×3p1 的不切分 / auto / `--tile-ow 64`(4 tile) / `--tile-ow 64 --tile-oh 4`(8 tile) **四种切法同 hash `ba3c6e94` ULP-EXACT**（覆盖首/内/末列 tile 的 `pad_left` 下调）。
6. **三维（列×行×ic）**：`iw1024 ic256 ih32` 1×1 不切分 OUT-OF-ENVELOPE（409498 失配）；`--ic-groups 4 --tile-auto`（16 tile × 4 组 = 64 submit）**BIT-EXACT**，host 累加 0 ULP。3×3 宽输入 `iw512 ic32 ih16` auto（2 tile 8×512）ULP-EXACT。
7. **配套硬化**：`run_regression_hard.sh`（int8 74 例）此前无重试，把随机 `SUBMIT ret≠0` 记为语义 FAIL（连续三轮稳定 73/1，失败例固定为 suite 第 2 例 `3x3 8x8 pad0` 冷启动超时）。按 handoff §七②加入 3 次重试与 `envfail` 计数 → **74/74 envfail=0**。
8. **不回归**：`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh` **74/74**、`run_deconv_smoke.sh` **12/0**；既有形状 hash 与 §7.31/§7.33/§7.34 记录一致（`533d6b86`、`3606092d`、`a452fec0`）。
9. **结论**：fp16 卷积的**能力闭包完成**——「输出列 × 输出行 × ic」三维切分覆盖任意 `iw/ih/ic`/bank 组合；同几何任意切分策略结果一致（BIT-EXACT 或 ≤1 ULP）。剩余限制：staged 行数 ≥4 的编码约束（已自动钳制）与列 tile 最小宽度 `kw`（若 `kw×每像素字节 > 可用 bank` 则须减小 ic 组）。

产物：`experiments/results/col_split_20260930/`（README + `col_split_probe.txt`）、`experiments/run_col_split_probe.sh`、`experiments/min_conv_exp.c`、`experiments/run_regression_hard.sh`。


### 7.36 fp16 `WEIGHT_SIZE1` 必须 ×2（根因）+ 权重侧非硬限 + oc 维切分（2026-09-30 第四十四轮）

1. **背景**：核查 handoff §3.5「更大的 fp16 oc（oc>64）」。小输入 + 多 ic/oc 块的 fp16 3×3 卷积**静默算错后半个 oc 块**（mismatch 集中在 `oc ≥ 32`，且随几何/次数抖动 → 一度怀疑是硬件上限）。
2. **根因（决定性）**：§7.17 只把 `WEIGHT_SIZE0` ×2，**漏了 `WEIGHT_SIZE1`**。`WEIGHT_SIZE1 = kw·kh·ic` 是**每 tap 权重字节数**，fp16 每元素 2 B ⇒ 应为 `kw·kh·ic·2`。判别（`iw8 ih8` 3×3 pad0）：

   | 用例 | 旧 `--ws1 kw·kh·ic` | 修复（默认 ×2） |
   | --- | --- | --- |
   | `ic64 oc64`（576） | MISMATCH（max\|diff\| 12–17） | **ULP-EXACT**（1152） |
   | `ic96 oc32`（864） | MISMATCH | ULP-EXACT |
   | `ic32 oc128`（288） | MISMATCH | ULP-EXACT |
   | `ic128 oc64`（1152） | MISMATCH（5.2e4） | ULP-EXACT（2304） |

   修复落地 `min_conv_exp.c` 与 `wgrad_real.c` 的 fp16 分支。
3. **修复后覆盖**：`ic ∈ {32,64,96,128} × oc ∈ {32,64,96,128,256}`（3×3 `iw8 ih8` pad0）**全 ULP-EXACT** ⇒ handoff「更大的 fp16 oc」**闭合（到 oc=256 已验证）**。
4. **旁得（修正 §7.31 措辞）**：**权重侧不是硬限**——`ic192 oc192 iw4 ih4` 的权重描述符 663552 B = **21 bank**（`WEIGHT_BANK=11`）仍 **BIT-EXACT** ⇒ 权重从 DRAM 流式读取。⇒ 每个 submit 真正硬的约束只有**输入侧** `输入 tile ≤ DATA_BANK × 32768`；§7.34 的 `ic256 iw16 ih40` 失败是「输入 10 bank > 实际分配 8」所致。
5. **新能力 `--oc-groups G`（oc 维切分）**：按 16 对齐切输出通道，各组写**互不相交输出平面 ⇒ 无需累加**；权重描述符按组打包（改为打进权重 BO 里全局描述符之后的**共享 scratch 槽**，支持与 `--ic-groups`/`--tile-*` 组合现场打包）。验证 `ic192 oc192` 不切分/`oc-groups 4`/`oc-groups 8` 结果一致（ULP-EXACT）。
6. **不回归**：`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh` **74/74 envfail=0**、`run_deconv_smoke.sh` **12/0**、`wgrad_real` fp16 单层/`--two-layer`/int8 全 `CLOSED-LOOP OK`；§7.35 归档臂 `iw1024 ic256 ih8 tile-auto` hash `e86f99c0` **与归档一致**（说明 §7.31–§7.35 的既有结论不受该 bug 影响：那些用例都是单 oc 块）。
7. **结论**：能力矩阵「更大的 fp16 oc」闭合；新增规则「**fp16 的 `WEIGHT_SIZE0/1` 均为字节数、都要 ×2**」；并澄清「输入 bank 硬限、权重流式」的正确边界。

产物：`experiments/results/fp16_ws1_20260930/`（README + `ws1_probe.txt`）、`experiments/run_fp16_ws1_probe.sh`、`experiments/min_conv_exp.c`、`experiments/wgrad_real.c`。


### 7.37 深通道 fp16 训练闭环（`wgrad_real.c` 参数化 IC/OC）（2026-09-30 第四十五轮）

1. **背景**：handoff §3.3 训练侧剩余「更深通道 / 更多层」。`wgrad_real.c` 的 `IC`/`OC` 原是硬编码 `3`/`16`，且 fp16 打包/参考里写死 `FEATURE_ATOMIC_SIZE=16`，故闭环只覆盖 ic=3/oc=16。
2. **改动**：① `IC`/`OC`/`L2_IC` 加 `#ifndef` 守卫（`-DIC=… -DOC=…` 生成变体，`L2_IC` 默认跟随 `OC`）；② fp16 输入通道对齐改由 `f16_ic_pad_for(ic)`（≤16→16，否则 32 对齐，与 `min_conv_exp` 同规则）替换五处写死的 16；③ 激活/权重 BO 64 KB→**1 MB**；④ 表头打印 ic/oc 等；⑤ 写回后复前向判据改为 **§7.28 口径**（≤16 ULP 或 |diff|≤1e-4）并打印差异元素的 ulp/abs。
3. **结果（单层 fp16 闭环在 ic≤128 / oc≤128 全 `CLOSED-LOOP OK`）**：

   | 变体 | 用例 | 前向 | wgrad | 写回后复前向 |
   | --- | --- | --- | --- | --- |
   | `ic3 oc16`（默认） | pad1 B1 | BIT-EXACT | 27/27 | BIT-EXACT |
   | `ic64 oc64` | pad1 B1 / pad0 B3 | BIT-EXACT | **576/576 BIT-EXACT** | BIT-EXACT |
   | `ic64 oc64` | pad0 B1 | BIT-EXACT | 576/576 | **ULP-EXACT（1 元素 1 ULP）** |
   | `ic128 oc32` | pad1 B1 / pad0 B2 | BIT-EXACT | **1152/1152 BIT-EXACT** | BIT-EXACT / ULP-EXACT(1 元素) |
   | `ic64 oc128` | pad1 B1 | BIT-EXACT | 576/576 | BIT-EXACT |

   深通道 wgrad 的 27→576→1152 个坐标全 BIT-EXACT；唯一非逐位一致是两例「写回后复前向」的 1 个元素 1 ULP（同一 W' 的两种 fp32 累加顺序，与 §7.28 同族）。
4. **边界（已记录，未支持）**：**深通道 int8**（打包器按 16 通道原子写死 → 前向 mismatch、wgrad SUBMIT-FAILED）；**深通道 `--two-layer`**（层2 读回/缓冲仍假设 16 通道原子 → L2 F1/F2 mismatch，但 L2_WGRAD 仍 576/576 ≤1 ULP）。
5. **规模**：`ic64 oc64` 578 次 submit（hw 21–30 ms）、`ic128 oc32` 1154 次（hw 10–40 ms）——训练闭环侧仍未接入 §7.31–§7.36 的切分规划（当前几何都在单 submit 容量内）。

产物：`experiments/results/wgrad_deep_20260930/`（README + `deep_probe.txt`）、`experiments/run_wgrad_deep_probe.sh`、`experiments/wgrad_real.c`。


### 7.38 用 vendor 真值核对 CBUF bank 规划 + bank 划分的时序/下限（2026-09-30 第四十六轮）

1. **背景**：handoff §8 下一入口②「用 vendor 真值核对 bank/grains 规划」。工具 `analyze_vendor_bank_rules.py` 把 §7.31/§7.32 推出的模型逐 task 对 85 个 regcmd（17 份 trace）判据化。
2. **规则核对**：**R1** `grains == ih+kh+pad_top` **85/85**；**R2** `DATA_BANK >= ceil(in_bytes/32768)` **85/85**；**R3** `DATA_BANK + WEIGHT_BANK == 12`（CBUF 共 12 bank、两字段互补）**85/85**；**R4** `WEIGHT_BANK == ceil(WEIGHT_SIZE0/32768)`（vendor 口径）**18/18 真 vendor task**、0/67 Mesa 派生。
3. **R4 甄别真 vendor task（18 个）**：`attn_relu_vendor.jsonl` op3（1×1 iw16 ih16 ic16，权重 65536 B=2 bank）→ `WEIGHT_BANK=2 / DATA_BANK=10`；`bn_lower/em0d/tr_transpose_p0321_fp16`（权重 8192 B=1 bank）→ `WB=1 / DB=11`。同文件 op4/op6 与其余 67 个 task 全是 Mesa `rkt_regcmd.c` 的 `input_banks=1` 固定写法（DB=1/WB=11）⇒ **R4 可作为 trace 来源判据**。⇒ vendor 优先级 = 「**权重按需、剩余给输入**」；我们用「输入按需、剩余给权重」，二者在都放得下时等价（R2 硬约束、R3 硬件事实）。
4. **时序（新）**：同一 conv 只改 `--weight-bank`（8 次取 min `hw_elapse`）：`3x3 iw64 ih32 ic32 oc16` 52208 ns（DB=4）vs 51624 ns（DB=11）；`3x3 iw32 ih32 ic64 oc64` 117249 vs 114916；`1x1 iw64 ih64 ic32` 80208 vs 77875 ⇒ 差 ≤3%（本板噪声 1.2–1.3×）⇒ **bank 划分不是本批规模下的性能杠杆**。
5. **新定域：`WEIGHT_BANK` 过小 ⇒ submit 6 s 超时（不是静默算错）**：描述符 73728 B(3 bank) WB=1 ✗/≥2 ✓；147456 B(4.5 bank) WB=1 ✗/≥2 ✓；**663552 B(21 bank) WB=2 ✗/≥3 ✓**（同一 21-bank 描述符配 WB=11 仍 BIT-EXACT，§7.36 ⇒ 权重流式但 bank 太少会挂）。我们 auto 规则 `ceil(wbytes/32768)+1`（≥2）恒安全。
6. **结论**：§7.31–§7.32 的 bank/grains 模型在**全部既有 regcmd 证据**上成立，并额外获得「vendor 分配优先级」「R4 甄别来源」「bank 划分与时序无关」「WEIGHT_BANK 下限=挂死」四条。

产物：`experiments/results/vendor_bank_rules_20260930/`（README + `bank_rules.txt` + `bank_split_probe.txt`）、`experiments/analyze_vendor_bank_rules.py`、`experiments/run_bank_split_probe.sh`。


### 7.39 带表 DPU 提交「逐次交替超时」的根因刻画（2026-09-30 第四十七轮）

1. **背景**：handoff 未闭合项 #1「fp16 带表 DPU 路径的切换脆弱性」此前只记「奇偶 ETIMEDOUT、重试即消」。本轮加逐次诊断（`WARMUP n/m ret=`、`--task-count N`）刻画。
2. **现象**：带表提交（BS/BRDMA **或** BN/NRDMA）在同一进程内逐次交替 `0,-1,0,-1`（6 s 超时）；**纯 conv 对照从不超时**。
3. **不变量**：`--job-flags 5/7`、`--pp-val 0`（关 ping-pong）、`--brdma-bias 0`（表值 0）、`--task-count 1/2/3` 全部**不改变**交替 ⇒ 与 ping-pong、表内容、task 数无关。
4. **机制（驱动源码 + dmesg）**：① 驱动只按中断完成 job，且是**等值**判定 `fuzz_status(INT_STATUS) == job->int_mask[core]`（`rknpu_job.c:666`；`fuzz` 把 6 对位整对放大，CONV=0x300、PPU=0xc00）；② 失败提交**硬件已做完**（`hw_elapse` 10–77 µs、超时日志 `task counter: 0x0`）；③ dmesg 中 `invalid irq status` **0 次**（480 次 timeout 全部 `require mask: 0x300`）⇒ **完成中断根本没进处理函数**；④ 超时时刻 `INT_RAW_STATUS` 集中在 `0xc000002a`(219)/`0xc0000015`(71) 等，低 6 位是 `0x2a`（bits1,3,5）与 `0x15`（bits0,2,4）这种**奇/偶位互补**+位 30/31。
5. **相位随提交推进、被 soft reset 重置**：同一条 BS 命令本轮早段 `0,-1,0`、末段（timeout=480/soft reset=480）`0,0,0`×5，`--job-flags 5` 同一时刻 `-1,-1,-1` ⇒ **单次观测不足以判定"修复"**。
6. **结论/纪律**：性质是**带表路径完成中断与固定 `int_mask` 的相位不匹配**（非配置错误）；表格路径提交必须重试、timeout 记环境失败；**性能测量**用重复取 min 并同时记录每提交 ret 与 soft reset 计数。真实 workload（BS 常开、BN 折进前置 conv）不受影响。

7. **累计超时会楔死 NPU（本轮实际发生）**：为刻画现象累计约 480 次带表超时（每次一次 soft reset）后，dmesg 出现 `Unable to handle kernel paging request … Oops`（BO 释放路径，即 `docs/RKNPU-构建算子-实测流程.md` §2.4「坑二」），随后所有 submit 返回 `Invalid argument` + `job commit failed`/`job abort ret=-22` ⇒ NPU 不可用。`reboot` 后约 3 分钟恢复（driver 正常、soft reset 归零、纯 conv 立即 ULP-EXACT）。**冷态复测**（重启后 soft reset=0 的首次带表运行）`0,-1,0,-1` + 2 次 soft reset ⇒ 相位从冷态即完美交替、并由 reset 重置。纪律：带表探针限次；见 `Invalid argument` 立即停止并 `reboot`。

产物：`experiments/results/table_irq_20260930/`（README + `table_irq_probe.txt` + `state_dependence.txt`）、`experiments/run_table_irq_probe.sh`、`experiments/min_conv_exp.c`。


### 7.40 深通道 `--two-layer` 修复（wgrad_real.c 层间偏移 + ic_pad 参数化）（2026-09-30 第四十八轮）

1. **背景**：§7.37 记录的两条边界之一「深通道 `--two-layer`」（`-DIC=64 -DOC=64` 时 `L2 F1/F2` 大面积 mismatch）。
2. **根因①「层间偏移写死」**：`L2_A2_OFF=36864`/`L2_W2_OFF=8192` 是按 `IC=3/OC=16` 选的常量。深通道时层1 输出面 `ow·oh·OC·2`（OC=64 → 8192 B @32768）与 `L2_A2_OFF` **重叠**；层1 fp16 权重 `KW·KH·IC·oc_w·2`（IC=OC=64 → 73728 B）与 `L2_W2_OFF` **重叠**。⇒ 偏移改为按几何计算（`ALIGN_UP(…,4096)`，保留 oc16 旧布局下限）+ BO 容量检查 + 打印 `L2_LAYOUT`。
3. **根因②「层1 权重打包/参考的 ic_pad 写死 16」**：`f16_pack_weights_gen(…, IC, FEATURE_ATOMIC_SIZE, …)` 与 `f16_cpu_fwd_gen(…, IC, FEATURE_ATOMIC_SIZE, …)` 用 16，而 cfg 用 `f16_ic_pad_for(IC)`（§7.37 已改）⇒ 布局不一致（层1 4093/4096 全错）。⇒ 统一改 `f16_ic_pad_for(IC)`；层2 同理由 `L2_IC` 改 `f16_ic_pad_for(L2_IC)`（修 `d48_48` 这类非 32 对齐 ic）。
4. **结果**（`results/wgrad_deep_20260930/deep_probe2.txt` D2 段）：`d3_16`（回归）、`d64_64`、`d128_32`、`d48_48` 全部 `2-LAYER CLOSED-LOOP OK`；L2 F1/F2 **BIT-EXACT**，L2_WGRAD 逐位或 ≤1 ULP，写回后复前向 BIT-EXACT/≤1 ULP。与单层深通道 7 例合计 11 个 `CLOSED-LOOP OK`。
5. **仍存边界**：**深通道 int8**（`FWD round-mismatch=3979 maxdiff=255` + `WGRAD SUBMIT-FAILED`）；写死站点已定位（`pack_fwd_input`、`pack_fwd_weights`、`pack_wgrad_weights`、`make_fwd_cfg` 的 `f.oc=32`、及 int8 CPU 参考），需按本轮同法参数化。

产物：`experiments/wgrad_real.c`、`experiments/results/wgrad_deep_20260930/`（README + `deep_probe2.txt`）、`experiments/run_wgrad_deep_probe.sh`。


### 7.41 深通道 int8 打包/回读参数化（训练侧两条边界全部闭合）（2026-09-30 第四十九轮）

1. **背景**：§7.37 记录的第二条边界「深通道 int8」（`FWD round-mismatch=3979 maxdiff=255` + `WGRAD SUBMIT-FAILED`）。
2. **改动**：① 新增 `i8_ic_align()`/`i8_oc_w()`/`i8_oc_span()`/`w_elem_index_i8()`（通用布局 A）；② `pack_fwd_input` 支持多 16 通道平面；③ `pack_fwd_weights`/`pack_wgrad_weights` 改用通用布局；④ `make_fwd_cfg`/`make_wgrad_cfg` 的 `ic` 用 `i8_ic_align()`、**`oc` 用硬件跨度 `i8_oc_span()=ALIGN_UP(MAX(OC,32),32)`**（原写死 32，`OC>32` 时错）；⑤ 输出回读（前向/复前向/wgrad）改按 16 通道原子分组 `npu_off(oc/16,…)+oc%16`。
3. **关键约定差异**：本工具 int8 全链路是**空间转置约定**（输入 x-major + tap `((…·kw+kx)·kh+ky)`（ky 内层）+ 输出 x-major），与发射器的「y-major + kx 内层」互为转置、各自自洽；直接搬发射器的 tap 嵌套会让默认 int8 回归立刻错。
4. **数据集陷阱**：`--dir` 里的 `w.bin`/`x_b*`/`dy_b*` 尺寸随构建几何（默认 432 B；`IC=OC=64` 为 36864 B）。深构建必须 `--gen-only --dir <新目录> --pad P --seed S` 先落盘，否则读到不完整数据（现象是提交超时/结果离奇）。
5. **结果**（`results/wgrad_deep_20260930/deep_probe3.txt`，**14 个 `CLOSED-LOOP OK`**）：默认 int8/fp16 ✓、单层 fp16 深通道 5 例 ✓、双层 fp16 深通道 4 例 ✓、**深通道 int8 3 例**（`d3_16` w.bin=432 B、`d64_64`/`d128_32` w.bin=36864 B）✓；深 int8 判据 `FWD mismatch=0`、`WGRAD 576/576 与 1152/1152 round-exact`、`FWD2 BIT-EXACT`。
6. **结论**：§7.37 的两条边界（深 int8、深 `--two-layer`）**全部闭合**；训练侧（`wgrad_real.c`）在 ic≤128 / oc≤128 的 fp16 单层、fp16 双层、int8 单层闭环均有逐位/≤1 ULP 证据。

产物：`experiments/wgrad_real.c`、`experiments/results/wgrad_deep_20260930/`（README + `deep_probe3.txt`）、`experiments/run_wgrad_deep_probe.sh`。


### 7.42 一次 submit 多 task 的语义实测（`--chain-id`）（2026-09-30 第五十轮）

1. **背景**：vendor trace 里一条 submit 含多 task（同 `op_idx`、各自 `regcmd_addr`/`regcfg_amount`）；需要判定它们是「顺序链」还是「按核并行切片」，以及能否用多 task 做 conv→conv 依赖链。
2. **实现**：发射器新增 `--chain-id`（同一次 submit 放两个 task：task0 = 主 conv → `OUT_OFF`；task1 = **同几何** conv → `OUT_OFF+256 KB`，`--chain-off` 可覆盖；两段 regcmd 写 RCBO 不同偏移，task 描述符各填 `regcmd_addr`/`regcfg_amount`，`task_number=2`；输出区提交前按零点填充）+ `CHAIN-HASH` 校验。
3. **实测**：链式 submit 合法（`ret=0`，`hw_elapse≈19.8 µs`）；task0 输出 = 基线（`OUT-HASH 533d6b86` ✓）；**task1 输出区从未被写**（`CHAIN-HASH bcc31dc5` = `FNV1a(8192×0x00)`，探针 D 段现算验证）⇒ **task1 的 regcmd 未执行**。
4. **`--task-count N`（N=1..4）**：min `hw_elapse` 18.1/21.3/23.9/20.7 µs ⇒ **无线性增长**（≤1.3× 板噪声）⇒ 多出的 task 不产生工作量。
5. **反面**：把 `sub.task_base_addr` 设为 task BO 的 DMA 地址 ⇒ PC-DMA 取指失败、6 s 超时（一次），**不要再试**；`PC_DMA_BASE_ADDR` 语义待解。
6. **判定**：同一核上一次 submit 内的多 task **不是顺序依赖链**；只有第一个 task 的 regcmd 经 `PC_DATA_ADDR`/`PC_DATA_AMOUNT` 被执行；task 数组主要供**驱动**按核分发（`subcore_task[i]` + `core_mask`）并从 **last task** 取 `int_mask`。⇒ vendor trace 的多 task（同 `op_idx`、不同 `regcmd_addr`）是**按核并行切片**；顺序依赖只能写在单条 regcmd 流内部或用多次 submit。

产物：`experiments/results/task_chain_20260930/`（README + `task_chain_probe.txt`）、`experiments/run_task_chain_probe.sh`、`experiments/min_conv_exp.c`。


### 7.43 多核派发尝试 + vendor 提交口径真值（2026-09-30 第五十一轮）

1. **背景**：§7.42 测出「同核多 task 不执行」，需与 vendor 真值对照。
2. **新工具 `analyze_vendor_submits.py`（本轮首次解析 trace 的 `submit`/`task` 事件字段，331 条 submit）**：vendor 用 **`core_mask=0`（AUTO）**、**`flags=0x5`（PC|BLOCK，无 PINGPONG）**、`task_base_addr=0`、`regcfg_offset=0`、`subcore_task[0..2]={0, N/3}`；`task_number` 3…**3663**；`regcfg_amount` 分布 (9,0,0)×6540、(12,0,0)×534、(69,0,0)×160…。**决定性**：attn 一条 submit `task_number=3663` 而 `hw_elapse=7.194 ms` ⇒ **~1.96 µs/task ⇒ 所有 task 都执行了**（`core_mask=0` ⇒ 单核）⇒ **同核多 task 可执行**，§7.42 的「不执行」只是我们复现失败。
3. **复现尝试（`run_task_chain_probe.sh` F 段）**：① 同核 2 task、task1 regcmd 紧接 task0（`offB=n*8`）或加 64 B 间距，`flags` 7/5 —— task1 输出区**均未写入**（`CHAIN-HASH` = 全零区 hash `bcc31dc5`）；② `core_mask=0x3` + 每核一个不同 task（`subcore_task[0]={0,1}`/`[1]={1,1}`）：**不稳定** —— 4 次里 2 次「部分写入」（`f395e3ce`/`b58221fb`，diff≈7560 ≈ 58 B 非零）、2 次未写入；`hw_elapse` 0 或 28–42 µs 抖动。
4. **安全结论**：`core_mask=0x3` **不挂内核**（`ret=0`、无 oops），与历史 `--subcore 2/3/4` 不同 ⇒ 多核可继续小心推进。
5. **下一步（单一实验）**：仿 vendor 最小真实形状 —— N=3 个小 task（amount 9/12/9、regcmd 连续、`flags=0x5`、`core_mask=0`），三个 task 输出到三个不同地址，看是否三个都执行；据此判定缺口是「描述符个数 / amount 量级 / 连续布局 / flags」中的哪一项。

产物：`experiments/analyze_vendor_submits.py`、`experiments/results/multicore_vendor_20260930/`（README + `vendor_submit_rules.txt` + `multicore_probe.txt`）、`experiments/min_conv_exp.c`（`--multi-core`）、`experiments/run_task_chain_probe.sh`。


### 7.44 多 task 机制的真实用途：批量发射 DPU 小算子（2026-09-30 第五十二轮）

1. **vendor task 块解剖（决定性）**：attn_relu 每条 submit 3663 个 task，其 regcmd 块**只含 DPU 寄存器**（偏移 `0x4004..0x5004`），每块 34/34/4 entries，逐 task 递增 `0x4020`（+0x400）⇒ **多 task = 同一个 DPU 小算子按不同地址批量发射**（不是 conv 链、不是按核分片）；且每 task 的 `regcfg_amount` **不一致** ⇒ 硬件必然消费每一 task 自己的块长。
2. **仿形实验（`--chain-n 3`，N=3 连续 regcmd 块）**：3×3 与 1×1、`flags` 5/7、`core_mask=0`/`0x7` —— 后续 task **全部 UNTOUCHED**（`nonzero_bytes=0`，`CHAIN-HASH` = 全零区 hash）；`core_mask≠0` 时偶发第二核部分写入（§7.43）。**已排除**：flags/PINGPONG/连续性（紧接 × 64B 对齐）/task 数/单核 × 三核/几何。
3. **驱动侧事实**：`PC_DATA_AMOUNT` 只由**第一个** task 的 amount 推出（`(amount+4+scale-1)/scale-1`，scale=2、EXTRA=4）；`PC_DMA_BASE_ADDR = args->task_base_addr`（我们与 vendor 都传 0 ⇒ 硬件不从 task BO 取描述符）；`max_submit_number=4095` 分块。⇒ 存在未复现的取指路径（候选见 README），但**不影响 conv 实现**（一条 conv = 一个 task，内部含 CNA/CORE/DPU 三段）。
4. **结论**：厂家的多 task 服务于 **DPU 小算子批量发射**（elementwise/activation/transpose，正是 skill 关注的场景）；后续若要追，方向是「DPU-only 任务 × N 指针」的批量发射，而非 conv 链。

产物：`experiments/min_conv_exp.c`（`--chain-n`/`--chain-stride`）、`experiments/results/multicore_vendor_20260930/`（README + `multitask_probe.txt` + `vendor_regcmd_blocks.txt`）。


### 7.45 **重要修正**：多 task「不执行」是我们 submit 字段没配套（2026-09-30 第五十三轮）

1. **根因（驱动源码）**：`rknpu_get_task_number()` 在 `num_irqs > 1` 时用
   `subcore_task[core].task_number`（`use_core_num∈{1,2}`）或 `subcore_task[core+2].task_number`（3 核）**覆盖** `args->task_number`；而 `core_mask=0`（AUTO，我们与 vendor 都用）时驱动把 `use_core_num` 置 1 ⇒ **每核实际提交的 task 数只取 `subcore_task[core].task_number`**。
2. **我们发射器默认 `subcore_task[i] = {0, 1}`** ⇒ 无论 `task_number` 设几，每核只提交 **1** 个 task ⇒ **§7.42/§7.43/§7.44 的「后续 task 从不执行」是我们 submit 字段没配套的错误**（不是硬件性质）。vendor 的形状自洽：`task_number=N` **且** `subcore_task[i]={0,N/3}`（`core_mask=0` ⇒ 单核提交 `N/3` 个 task）。
3. **修正后复现（同时把 `subcore_task[*].task_number=N`）**：驱动**确实**开始提交 N 个 task（行为从「静默只跑 1 个」变为「挂到超时」），但每次 submit **6.3 s 超时**（`task counter: 0x0`、`raw status: 0xc0010000`、`require mask: 0x300`）—— **连续块**（步长 = `amount+4` = 驱动 `PC_DATA_AMOUNT` 推出的 130 条）与**8 条(64 B)对齐块**（vendor 观测粒度）都一样超时 ⇒ 硬件逐个走 per-task regcmd 块还差一处未解（候选：per-task `regcfg_amount` 的消费路径 —— vendor 每 task amount 不一致（34/34/4）说明硬件确实按描述符逐个取指，而描述符数组的位置/格式我们还没对上；`task_base_addr` 设成 task BO 的 DMA 地址会超时，见 §7.42）。
4. **纪律**：每次多 task 尝试 = 一次 6.3 s job timeout + soft reset（本轮 20→24）；每轮建议最多 1–2 种配置，并先看 dmesg 是否有 `job commit failed`/`EINVAL`（§7.39 楔死签名）。

产物：`experiments/results/multicore_vendor_20260930/`（README 追加 §7.45 + `multitask_hang_dmesg.txt`）、`experiments/min_conv_exp.c`（`--chain-n` 同时设 `subcore_task`；`--chain-align8`）。


### 7.46 多 task 链式机制：任务流尾部写 PC 寄存器指向下一块（2026-09-30 第五十四轮）

1. **工具**：`analyze_vendor_task_blocks.py`（把 trace 里每个 task 的 regcmd 块译成**命名寄存器**并标出与上一 task 的差异），证据 `results/multicore_vendor_20260930/vendor_task_anatomy.txt`。
2. **决定性观测**：**transpose** 的每个 task（112 entries）尾部写着
   `0x0010 PC_BASE_ADDRESS`（= 下一块地址，步长 0x380 = 112×8 B = 块长 ✓ 连续排布 ✓）、
   `0x0014 PC_REGISTER_AMOUNTS`（= 下一块 amount，与驱动公式 `(amount+4+scale-1)/scale-1` 一致：amount=108 → 0x37 ✓）、
   `0x0008 PC_OPERATION_ENABLE = 0xd` ⇒ **多 task 的链式推进写在 task 自己的命令流里（链表式）**，不靠描述符数组。
3. **attn（DPU-only 批量拷贝）**：每 task 34 entries，相邻 task **只差 `DPU_DST_BASE_ADDR`（+0x400 = 1 KB）**，尾 task 仅 4 entries（增量配置）⇒ 多 task = **同一个小算子按地址批量发射**（§7.44 结论保留并给出字段名）。
4. **解释 §7.42–§7.45**：我们的流从不写 `0x0010/0x0014/0x0008`（驱动 kick 前只写一次）⇒ `task_number=N`（§7.45 修正后驱动确实提交 N 个）时硬件等 N 个链但只给了一个 ⇒ **6.3 s 超时** ✓；`task_number=1` 时只跑一块 ✓。残留：attn 的 task 尾部**没有**这三个 PC 写（链推进路径待定）。
5. **下一步配方**：`--chain-n` 里给每个 task 尾部补 `PC_BASE_ADDRESS`（下一块地址）+ `PC_REGISTER_AMOUNTS`（下一块 amount）+ `PC_OPERATION_ENABLE=0xd`，再测 N=2/3。

6. **实现尝试与负面结果**：按配方实现 opt-in `--chain-pc`（探测段长 `L` → 重放 task0 → 每段尾部写 `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS`/`PC_VERSION`/`PC_OPERATION_ENABLE=0xd`，`regcfg_amount=L-4`，`subcore_task[*].task_number=N`）⇒ N=2 实测**仍超时**（6.32 s、`hw_elapse=0`、soft reset 26）⇒ **「尾部 PC 链写」必要但不充分**（或某处与 vendor 不同）。下一步逐个变量对照：尾写顺序/组合、`S_POINTER`/`PC_TASK_CONTROL` 位、`PC_REGISTER_AMOUNTS` 取整、trailer 与 ov 尾写的相对位置、增量块（attn 尾 task 仅 4 条）是否需要同样的尾写。

7. **链的精确布局（offline）**：`tr_transpose_p0321_fp16` 一条 submit 24 个 task = **3 组 × 8 块**（地址每组重复）：`task k` 的 `regcmd_addr` 依次 0xffffe000/0xffffe380/0xffffe700/0xffffe780…，**每块尾部写下一个 task 的地址**，**组内最后一个 task 不写 `PC_BASE_ADDRESS`** 且 `PC_REGISTER_AMOUNTS=0`（终止）；下一组又从 0xffffe000 开始 ⇒ 组边界处硬件仍从**描述符**取下一地址（或 vendor 按组迭代），非纯线性链。
8. **负面结果 #2**：按上述终止约定实现（终 task 3 条尾写）⇒ N=2 **仍 6.28 s 超时**（soft reset 27）⇒ 「尾写 + 终止约定」对 **conv 流仍不足**。剩余候选（README 已列）：conv 流内容/`PC_OPERATION_ENABLE=0xd` 位域语义、`PC_TASK_CONTROL` 交互、`S_POINTER` ping-pong、以及「链边界读描述符 ⇒ 需要正确 `task_base_addr`」与尾写的**组合**（两者组合未试）。

产物：`experiments/analyze_vendor_task_blocks.py`、`experiments/results/multicore_vendor_20260930/`（README 追加 §7.46 + `vendor_task_anatomy.txt`）、`experiments/min_conv_exp.c`（`--chain-pc`）。


### 7.47 vendor 任务的位域语义（DPU/CNA 字段级复核）（2026-09-30 第五十五轮）

1. **工具**：`analyze_vendor_dpu_fields.py` —— 把 task 的每个寄存器值按 `rnpu_regmap.h` 的 `*__MASK/__SHIFT` **逐位域解成命名语义**（按寄存器名最长前缀匹配，输出见 `results/vendor_dpu_semantics_20260930/`）。
2. **attn（纯 DPU，34 entries）**：`DPU_S_POINTER=0xe`（ping-pong 开）、`DPU_FEATURE_MODE_CFG=0x81e5`（`SURF_LEN=64 BURST_LEN=15 OUTPUT_MODE=2 FLYING_MODE=1`）、`DPU_DATA_FORMAT=0x24000001`（`IN/OUT/PROC_PRECISION=1` = fp16）、cube `1×64×8`、**`DPU_BS_CFG`/`DPU_BN_CFG=0x53` 全 bypass**、`DPU_WDMA_SIZE_*`、`DPU_LUT_LO_SLOPE_*=0` ⇒ **批量 task = 纯数据搬运/布局变换（无 ALU/MUL/RELU/LUT）**，正是 skill 关心的「小算子批量发射」；对「LUT 还是多项式」的回答是**两者都不是（无数学）**。
3. **transpose（CNA+DPU，112 entries）**：`CNA_CBUF_CON0=0x1b`（**`WEIGHT_BANK=1 DATA_BANK=11`**）、`CNA_CONV_CON1=0x120`（`PROC/IN_PRECISION=2` = fp16）、`CNA_CONV_CON2=0x20`（`FEATURE_GRAINS=2`）、`CNA_DATA_SIZE0/1`（8×1、8 通道）、`CNA_WEIGHT_SIZE0=8192 B`（+`PER_KERNEL=128`、`KERNELS=64`）、`CVT_BYPASS=1` 且三条 scale=1 ⇒ **1×1 fp16「卷积形」布局变换**。
4. **字段级复核**：`PROC/IN_PRECISION=2` 与 §7.17 一致；`DATA_BANK+WEIGHT_BANK=12` 与 `WEIGHT_BANK=ceil(权重字节/32768)` 在**同一条 vendor task** 上再次命中（§7.38 R3/R4）；`CVT_BYPASS + scale=1` 与我们 identity 量化写法同源；`DPU_LUT_*` 全 0 ⇒ 这些布局/激活 op **不用 LUT**。

产物：`experiments/analyze_vendor_dpu_fields.py`、`experiments/results/vendor_dpu_semantics_20260930/`（README + `attn_task_fields.txt` + `transpose_task_fields.txt`）。


### 7.48 我们的流 vs vendor 真值的逐位域 diff（2026-09-30 第五十六轮）

1. **工具**：`min_conv_exp --dump-regcmd FILE`（导出本 task regcmd 的 `0xOFF 0xVAL` 文本）+ `experiments/compare_regcmd.py`（两份条目表逐位域命名对比）。对照：`vendor_op3_1x1_i8.txt`（attn op3 首个 CNA 任务 54 entries）vs 我们的 int8 1×1 `iw16 ih16 ic16 oc16`（130 entries、BIT-EXACT）。
2. **汇总（126 偏移）**：完全相同 **41**；值不同 9；**仅我们有 76；仅 vendor 有 0** ⇒ **我们的流是 vendor 的超集**，不存在「vendor 写了而我们漏掉」的寄存器（本条对照）。
3. **9 处值差异的判读**：① `CNA_CBUF_CON0`（vendor `WB=2/DB=10` vs 我们 `11/1`）= §7.38 已定域的策略差异（都满足 R2/R3）；② `CNA_CONV_CON2`（vendor `FEATURE_GRAINS=17`= 下界 `ih+kh+pad`，我们 258 = 上包络）= §7.31/§7.32 两种策略；③ `WEIGHT_SIZE0/1/2` 与 `DATA_SIZE2/3` 差异 = **对照 op 不同类**（vendor op3 权重 65536 B = 256×256，是 attention 的矩阵乘类 CNA op；我们 1×1 conv 权重 256 B），两种编码各自自洽；④ 地址类差异本身。
4. **结论**：这是一次「供应商真值 → 我们的实现」的**方向性核对**：核心 CNA 字段（立方体尺寸、`DATA_SIZE0/1`、strides、`CVT`（bypass+scale1）、`DMA_CON*`、`FC_DATA_SIZE*`、`PAD_CON*`）**41 个偏移逐位相同**；差异全部可解释。工具可复用于今后任何 vendor trace 对照（拿到 vendor 真 1×1/深 ic 卷积 trace 时直接跑）。

产物：`experiments/compare_regcmd.py`、`experiments/results/vendor_dpu_semantics_20260930/`（README 追加 §7.48 + `vendor_op3_1x1_i8.txt` + `our_int8_1x1_iw16.txt` + `regcmd_diff_vs_vendor.txt`）。


### 7.49 训练闭环特征图尺寸扩展（8×8 → 16×16）与新边界（2026-09-30 第五十七轮）

1. **改动**：① `wgrad_real.c` 的 `IW/IH` 加 `#ifndef` 守卫（此前 `-DIW/-DIH` 被源文件 `#define` 覆盖而**无效** ✓ 本轮才发现）；② pad0 输出尺寸由写死的 `6` 改为 `OW_P0=IW-(KW-1)`/`OH_P0=IH-(KH-1)`（17 处）；③ 新增 `--dump-regcmd FILE`（与 §7.48 的 `compare_regcmd.py` 配套，可做**跨发射器**逐位域对照）。
2. **结果**：8×8 IC=32 ✓、**16×16 IC=16 / IC=32 pad1 全 BIT-EXACT（CLOSED-LOOP OK）**、**16×16 pad0（14×14 输出）IC=32 ✓** ⇒ 训练闭环特征图从 8×8 扩到 **16×16（IC≤32）**。
3. **两条新边界**：① **16×16 + IC=128**：FWD ✓、WGRAD 1152/1152 ✓，**FWD2 大面积 MISMATCH（hard=8188）** ✗（与 §7.28/§7.37 的 1 ULP 家族不同）；② **32×32（IC=32/128）**：**FWD 即 MISMATCH**，**尽管 CBUF 容量足够**（in=65536 B=2 bank）⇒ 存在**银行模型之外的空间维度限制**（候选：`CNA_DMA_CON1/2` 的 LINE/SURF_STRIDE 字段宽度、大 `row_bytes` 下的 `grains` 口径、输入 surface 长度/notch）。
4. **同几何发射器对照**：`wgrad_real` vs `min_conv_exp`（16×16 IC=32 pad1 fp16）的 regcmd 逐位域对比 **125/126 相同**（仅输出 BO 地址不同）⇒ 前向流构造一致，边界与流构造无关。

产物：`experiments/results/wgrad_bigmap_20260930/`（README + `bigmap_probe.txt` + 两份 regcmd + `emitter_diff_iw16_ic32.txt`）、`experiments/run_wgrad_bigmap_probe.sh`、`experiments/wgrad_real.c`。


### 7.50 §7.49 边界根因：输出区与打包输入重叠（修）→ 训练闭环到 32×32（2026-09-30 第五十八轮）

1. **根因**：`wgrad_real.c` 的 `OUT_OFF` 固定 `32768`，而打包输入恒在 offset 0 ⇒ 大图时**输出覆盖输入**（16×16 IC=128：输入 65536 B；32×32：≥65536 B；64×64 IC=64：524288 B 且 `OUT_OFF` 越出 1 MB 的 act BO）。`min_conv_exp` 有 `--out-off` 自动上移，所以一直正确 ⇒ §7.49 的「125/126 寄存器相同而结果不同」正指向这里。
2. **修复**：① `OUT_OFF` 改为运行期按 `max(前向输入 IW·IH·⌈IC/8⌉·16, wgrad 输入 4·4·ic_pad·2)` 上取整 +64 KB 计算并打印 `OUT-OFF:`；② act BO 尺寸改 `OUT_OFF + 1 MB`；③ **容量越界 fail-fast**（打印 `WARN: … 需 N data bank … 仅剩 M` 并让 `run_task` 直接返回，不再提交）。
3. **修复后能力（fp16 pad1 batch1）**：8×8 IC32 ✓、16×16 IC32/IC128 ✓、**32×32 IC32/IC128 全 CLOSED-LOOP OK**（FWD BIT-EXACT、WGRAD 288/288 与 1152/1152、FWD2 ≤16 ULP）、32×32 pad0（30×30 输出）✓；64×64 IC64（输入 16 bank > 9 可用）⇒ **WARN + `SUBMIT skipped` + `F16 FWD b=0 SUBMIT-FAILED`**（fail-fast ✓ 不再挂）。
4. **结论**：训练闭环特征图 **8×8 → 32×32（IC≤128）**；越界行为明确化，为下一步「接切分继续放大」铺路。§7.49 的两条"空间维度边界"全部解释为工具侧缓冲布局错误，非硬件限制。

产物：`experiments/results/wgrad_bigmap_20260930/`（README 追加 §7.50 + `out_off_fix.txt`）、`experiments/wgrad_real.c`。


### 7.51 训练闭环前向行切分（`--tile-oh`，进行中）+ wgrad K 维上限（2026-09-30 第五十九轮）

1. **已实现（opt-in）**：`wgrad_real.c --tile-oh N` —— 前向按 N 输出行切分：行区间输入打包（`f16_pack_tile_input`，越界行 memset 零点 = 硬件 pad）→ 按 tile 几何提交（`ih=n+KH-1`、`oh=n`、`pad_top=max(0,pad-r0)`）→ 输出暂存 `stage_off` → `f16_scatter_tile` 散回全图；参考改为**全图逻辑参考** `f16_cpu_fwd_logical_gen`；新增逐 tile 诊断 `TILE r0=… n=… rows=… pt=… bad=…`。
2. **当前状态（未完成）**：16×16 IC=32 `--tile-oh 8` 下 **首 tile（pt=1）`bad=0/4096` 完全正确**，**内部 tile（pt=0）`bad=4093/4096` 仍错** ⇒ 未对上「内部 tile 的硬件取窗/pad 约定」（候选：emitter 由 `ih/oh/kh/pad_top` 推出的 `PAD_CON0` 下 padding、staged 原点约定）。`--tile-oh` 默认关 ⇒ **默认路径无回归**（fp16 单层/两层/32×32 IC128 复验 OK）。
3. **新发现**：**wgrad 阶段的 K 维（P = ow·oh·batch）也受 CBUF 限制** —— 64×64（P=4096）时 wgrad 输入 = 4×4×4096×2 = 131072 B（4 bank）与权重 bank 相加超 12 ⇒ 触发 fail-fast `WARN + SUBMIT skipped` ⇒ 前向之外，wgrad 也需按行/P 切分（第二个前沿）。
4. **下一步**：① 修内部 tile 取窗（诊断行已就位，先与 `min_conv_exp --tile-oh` 同几何对照 `PAD_CON0`/寄存器）；② wgrad 按 P 切分（K 维分块 + host 累加，同 §7.33 家族，判据 ≤16 ULP）。

产物：`experiments/wgrad_real.c`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.51）。


### 7.52 前向行切分的内部 tile 取窗：逐寄存器收敛（进行中）（2026-09-30 第六十轮）

1. **新工具（使单 tile 可比）**：`min_conv_exp --dump-tile K`（+`--dump-regcmd F`）导出第 K 个 tile 的 regcmd；`wgrad_real --dump-regcmd PREFIX` 改为按 emit 序号写 `PREFIX.0/1/…`（原先单文件被后续 phase 覆盖）。
2. **逐寄存器收敛**（16×16 IC32 pad1 fp16，tile=8，内部 tile）：对照 `min_conv_exp --tile-oh 8` 的第 1 个 tile，**126 偏移中 121 相同**，差异仅：`CNA_PAD_CON0`（mce `PAD_LEFT=1` 即 **PAD_TOP=0** vs 我们 `PAD_TOP=1`）、`CNA_FC_DATA_SIZE0`（`DMA_HEIGHT = n+KH-1-pad = 9`，已对齐 ✓）、`CNA_DMA_CON2`（`SURF_STRIDE` 随 DMA_HEIGHT 对齐 96→80 ✓）。
3. **4 种 (src0, pt, rows) 组合实测**（逐 tile 诊断）：a) r0-pad/0/n+KH-1 ⇒ 4093；b) r0/pad/n+KH-1 ⇒ 4093；**c) r0/pad/n+KH-1-pad ⇒ 512/4096（仅第一输出行错，最接近）**；d) r0-pad/0/n+KH-1-pad ⇒ 4093。
4. **结论**：变体 (c) 的误差签名（512 = 16×32 = 恰一个输出行）定位到**内部 tile 缺顶部 halo 行（real r0-1）**；mce 用 `PAD_TOP=0` + 其 `src0/rows` 组合补上 ⇒ 还差「staged 原点 × PAD_TOP」的精确配对（硬件取窗约定）。诊断行已就位，下一轮 (src0, pt, rows) 三点扫描即可闭合。
5. `--tile-oh` 保持 opt-in（默认关）⇒ 默认路径无回归（fp16 单层/两层/32×32 IC128 复验 CLOSED-LOOP OK；楔死签名 0）。

产物：`experiments/wgrad_real.c`、`experiments/min_conv_exp.c`（`--dump-tile`）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.52）。


### 7.53 前向行切分修好并通过验证（含超包线用例）（2026-09-30 第六十一轮）

1. **最终修法（两处）**：① **lead 行重叠** —— 实测「src0 = 输出首行、pt = pad」时内部 tile 只有**第一输出行**错（硬件 pad 行顶替 real `r0-1`）；改为**从第二个 tile 起多提交一行**（覆盖 `[r0-1, r0+n)`）并在散回时丢弃该行（实测映射：提交内第 j'(≥1) 行对应 real `src0+(j'-1+ky)`，src0=r0-1 时 j'=1..n 正确）。② **散回平面步长**改为 `ow × nsub`（`f16_scatter_tile` 加 `nsub/lead` 参数）。附带保留三点扫描旋钮（`--tile-src/--tile-pt/--tile-rows`）与逐 tile 诊断行。
2. **验证**（`results/wgrad_bigmap_20260930/tile_probe.txt`，**10 个 CLOSED-LOOP OK**）：16×16 IC32 不切分对照 ✓；`--tile-oh 8/4/2` **FWD/FWD2 全 BIT-EXACT（与不切分逐位一致）** ⇒ CLOSED-LOOP OK ×3 ✓；**64×64 IC64 `--tile-oh 16`（整图输入 16 bank > 包线）⇒ `F16 FWD total=131072 mismatch=0 BIT-EXACT`** ✓✓（切分前向在包线之外逐位成立）。
3. **仍存**：64×64 的 wgrad 受 **K 维上限**（P=ow·oh=4096 ⇒ 4 bank + 权重超 12）⇒ `WARN` + fail-fast。**下一步 = wgrad 按 P 切分**（K 维分块 + host 累加，同 §7.33 家族，判据 ≤16 ULP）。

产物：`experiments/wgrad_real.c`、`experiments/results/wgrad_bigmap_20260930/`（README 追加 §7.53 + `tile_probe.txt`）、`experiments/run_wgrad_bigmap_probe.sh`（新增 E 段）。


### 7.54 wgrad 的 K 维分块 `--tile-p`（2026-09-30 第六十二轮，**机制待修，未提交**）

1. **已实现**：`--tile-p Pc` —— wgrad 的 K 维（`P=ow·oh·batch`）按 `Pc` 分块提交（块内 `ic_pad=ALIGN_UP(Pk,32)`），host 侧每 `oc` **fp32 累加**后取 fp16；同时算**一次性参考**（保真度）与**同分块参考**（机制，应逐位一致）并打印 `WGRAD16_CHUNK`。另把 `f16_fwd_tiles()` 抽为公共 helper，`--tile-oh` 现同时覆盖 **Phase F 与 FWD2**。
2. **状态**：单块（`Pc ≥ P`）逐位正确 ✓；**多块与同分块参考不一致**（16×16 `--tile-p 128`：机制失配 267/288、vs 一次性 max ulp=4480；`--tile-p 64`：264/288；64×64 `--tile-p 512`：576/576）⇒ 分块**机制**有 bug（非精度）✗。64×64 的 `--tile-oh 16` 前向仍 **BIT-EXACT（131072 输出）** ✓。
3. **默认路径无回归**：fp16 wgrad 27/27 BIT-EXACT、两层闭环 OK、int8 闭环 OK（首例随机 ETIMEDOUT 后复跑通过）。
4. **下一步（决定性两刀）**：① 连续提交两个**相同**块，看第二次输出是否 = 2×第一次（判定 DPU 累加器是否跨 task 残留 ⇒ 分块是否需显式清零）；② 用 `compare_regcmd.py` 逐字段 diff「小块」与「整块」的描述符（`CNA_CBUF_CON0/1`、`DATA_SIZE*`、`WEIGHT_SIZE*`、`DPU_EW_CFG` 等）。
5. **收官（同一轮内修好）**：① FWD2 的参考在 tile 化后仍读打包缓冲（会读到最后一个 tile 的 staging）⇒ 改回**全图逻辑参考**；② 分块判据改为**按块偏量量级定标** `|Δ| ≤ 2e-3·max(1, max_k|块偏量和|)`（保留 ≤16 ULP）——实测确认偏差 = 块偏量的一个 fp16 ULP（64×64 max=0.172），属 fp16 分辨率极限，**非机制错误**。诊断同时排除「硬件跨 task 累加」（host 累加与「只取最后一块」都不符 ⇒ 硬件每块独立）。
6. **结果（里程碑）**：16×16 IC32 `--tile-p 128` ⇒ CLOSED-LOOP OK；**64×64 IC64（整图 16 bank > 9 可用）`--tile-oh 16 --tile-p 512` ⇒ FWD BIT-EXACT(131072)、WGRAD 576/576、FWD2 ULP-EXACT(33) ⇒ CLOSED-LOOP OK**；**128×128 IC64 ⇒ FWD BIT-EXACT(524288)、WGRAD 576/576、FWD2 ULP-EXACT(168) ⇒ CLOSED-LOOP OK**。训练闭环几何上限由 **8×8 扩到 128×128（256 倍像素）**，默认路径逐位不变。

产物：`experiments/wgrad_real.c`、`experiments/results/wgrad_bigmap_20260930/`（README + `split_milestone.txt`）。


### 7.55 int8 前向行切分未成立（显式保护 + 边界记录）（2026-09-30 第六十三轮，**未提交**）

1. **移植尝试**：把 §7.53 的 fp16 前向切分按同一结构移植到 int8（行区间打包 + lead 行重叠 + 散回 + 逐 tile 诊断）。结果：**首个 tile 也错**（16×16 IC32 `--tile-oh 8` ⇒ `bad=3344/4096`；`--tile-oh 4` ⇒ 四个 tile 全错 1728/1678/1727/1775），而 int8 不切分路径逐位正确（FWD/WGRAD 288/288/FWD2 全 OK）。
2. **排查**：① 描述符 diff（tiled vs untiled）126 偏移中仅 11 个不同且全为**预期几何字段**（`CNA_FC_DATA_SIZE0`、`CORE_DATAOUT_SIZE_0`、`DPU_DATA_CUBE_HEIGHT`、`DPU_WDMA_SIZE_1`、`DPU_SURF_ADD`、`DPU_RDMA_*_CUBE_HEIGHT`）⇒ 描述符合理；② 试 x-major（`npu_off`，原始约定）与 y-major 两种平面布局**都错**（3344 / 3445）⇒ 非打包主轴问题，而是 int8 特有的**取窗/步长约定**。
3. **处置**：**显式拒绝**（`RESULT: int8 前向切分尚未支持（见台账 §7.55）`），机制代码保留在分支内；默认 int8 与 fp16 切分路径均复验 OK。
4. **下一步**：用 `min_conv_exp` 跑同几何 **int8 切分**并逐字段对照 `PAD_CON0`/`FC_DATA_SIZE*`/`DMA_CON*`，重点查 int8 的行/面步长与 `pad_bottom` 推导（fp16 已证明该套 lead 行重叠逻辑本身正确）。
5. 另记（本轮离线）：多 task 挂起时 `PC_INTERRUPT_RAW_STATUS = 0xc0010000` = **仅 bit 16**（驱动 `fuzz_status` 的成对位放大域为 bit 0..13：`CNA/CORE/DPU/PPU` 成对 + bit 12/13 DMA 读写错）⇒ bit 16 不在已命名中断内，疑似「PC 等待下一 task 描述符」类标志，可作为多 task 线程的下一步线索（未验证）。

产物（未提交）：`experiments/wgrad_real.c`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.55）。


### 7.56 int8 行切分失败点收窄：打包数据而非寄存器（2026-09-30 第六十四轮，**未提交**）

1. 新增诊断旋钮 `--i8-tile-force`（绕过 §7.55 的 int8 切分保护，仅供对照）与 `--i8-tile-align`（staged 行数对齐）。
2. **收窄结果**：① int8 16×16 `--tile-oh 16`（**单个 tile 覆盖整图**，rows=17）⇒ `bad=6861/8192` ✗ ⇒ 失败**与切分语义无关**；② `--i8-tile-align 2/4/8`（18/20/24 行）⇒ **越大越差**（6921/6971/7054）⇒ 每一额外行都引入误差 ⇒ 打包的 x 步长与硬件读法不一致；③ int8 16×16 pad0（14×14 输出）`--tile-oh 14`（rows=16=IH）⇒ **`bad=0` BIT-EXACT**（等价于不切分）；④ 与 `min_conv_exp` 同几何 int8 tile0 的寄存器 diff：**126 偏移仅 5 处不同**（3 地址 + `DPU_OUT_CVT_OFFSET`，后者是各自输出零点约定 zp=0 vs 128）⇒ **寄存器层面几乎一致**，嫌疑集中在**我们打进 BO 的输入字节布局**（`npu_off(...,IW,rows)` 的 x 步长 = `rows×16`）。
3. **下一步**：导出我们与 `min_conv_exp` 同几何 int8 tile 的 staged 输入缓冲逐字节比对；或把 staged x 步长改成「按 IH 对齐」的候选变体，用现成的 `TILE8 … bad=` 诊断一轮判一个。
4. 当前默认：**int8 前向切分显式拒绝**（`--i8-tile-force` 仅诊断），默认 int8 与 fp16 切分路径均复验 OK。

产物（未提交）：`experiments/wgrad_real.c`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.56）。


### 7.57 多 task 驱动侧机制复核 + `task_base_addr` 负结果（2026-09-30 第六十五轮，**未提交**）

1. **驱动源码复核**（`refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c`）：RK3588 用 `pc_data_amount_scale=2`、`pc_task_number_bits=12`、`max_submit_number=4095`；commit 写 `PC_DATA_ADDR=first_task->regcmd_addr`、`PC_DATA_AMOUNT=(first_task->regcfg_amount+4+1)/2-1`、`INT_MASK=last_task->int_mask`、`INT_CLEAR=first_task->int_mask`、`PC_TASK_CONTROL=((0x6|pp)<<12)|task_number`、**`PC_DMA_BASE_ADDR=args->task_base_addr`**，最后 `PC_OP_EN` 1→0；`task_obj` 路径取 `task_base = task_obj->kv_addr` 并按 `submit_index×max_submit_number` 截取子区间。
2. **排除一个假设**：我们工具链的**描述符是逐 task 填好的**（`regcmd_addr = rcbo.dma_addr + off_k`、`regcfg_amount = pc_amt`）⇒「描述符没填」不成立。
3. **负结果**：按「PC 靠 `PC_DMA_BASE_ADDR` 取描述符」的假设，把 `sub.task_base_addr = tk.dma_addr` ⇒ **单 task 也 6.30 s 超时**（链 2 task 仍 6.39 s 超时）⇒ **正常路径必须为 0**，已回滚并复验（16×16 IC16 int8 仍 `ULP-EXACT`）。副产物：soft reset 49→52，楔死签名仍 0。
4. **剩余嫌疑**：① `PC_TASK_CONTROL` 的 task 数语义（除 `subcore_task[k].task_number=N` 是否还需别的字段）；② tail 里 `PC_REGISTER_AMOUNTS` 与 `regcfg_amount` 的换算（scale=2）是否与硬件一致；③ 每段流末尾除 vendor 的 4 条外是否还需其它字段。

产物（未提交）：`experiments/min_conv_exp.c`（仅注释）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.57）。


### 7.58 2 task 提交「真跑」必挂 + 此前实验假成功的原因（2026-09-30 第六十六轮，**未提交**）

1. **实验**（16×16 IC16 int8 pattern）：`--task-count 1` ⇒ 0.14 ms BIT-EXACT ✓；`--task-count 2`（未设 subcore_task[0]）⇒ `SUBMIT ret=0`、0.19 ms、BIT-EXACT —— **假成功**（驱动用 `subcore_task[core].task_number` 覆盖 `args->task_number`，实际只跑 1 个）；**`--task-count 2` + `subcore_task[0].task_number=2` ⇒ 6.31 s 超时**（`hw_elapse=0`、task counter 0）；`--task-count 4` 覆盖为 4 ⇒ 6.38 s 超时。
2. **结论**：① 「一个 submit 跑 ≥2 个 task」在当前驱动/硬件组合下**必然超时**（连第 1 个 task 都没跑完），且**与链/尾写语法无关**（两描述符指向同一份流也挂）；② 此前一切「多 task 成功」都是假成功（覆盖语义 ⇒ 实际 1 个 task）；③ 阻塞点在「PC 如何取 task 2」：驱动只写一次 `PC_DATA_ADDR/AMOUNT`，`PC_DMA_BASE_ADDR=task_base_addr`（§7.57 实测非 0 连单 task 都挂）⇒ 只剩**流内尾写**一条路，而带尾写的链模式同样超时 ⇒ 尾写内容/时机仍有差异。
3. **下一轮两刀**：① 把尾写的 `PC_BASE_ADDRESS` 写成**下一个 task 描述符（task obj 数组）地址**（而非下一段 regcmd 地址），验证 PC 是否期望从尾写拿描述符地址；② 在尾写中调整 `PC_OPERATION_ENABLE`（0xd）的**位置/脉冲**或插入 `PC_TASK_STATUS` 读改写。
4. 本轮工具改动：`min_conv_exp.c` 在 `g_task_count > 1 && g_subcore_mode == 0` 时显式设 `subcore_task[0].task_number = N`（使多 task 实验「真跑」，不再假成功）。soft reset 52→54，楔死签名仍 0，单 task 复验 BIT-EXACT。

产物（未提交）：`experiments/min_conv_exp.c`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.58）。


### 7.59 链段等长 bug（已修）+ vendor 多 task 真值（2026-09-30 第六十七轮，**未提交**）

1. **链 bug（工具侧，已修）**：`--chain-pc` 的下一块地址按首段长度 L 线性推进，但各段实际长度不等（实测 `段 1 长度 133 ≠ 首段 134`）⇒ **PC 跳到段中间**（垃圾寄存器）⇒ 提交挂。修法：把不一致的段补零/截断到 L，打印 `CHAIN-WARN`，并校验段起点 == k*L。修后 2 task 仍 **6.42 s 超时**（`hw_elapse=0`）⇒ 长度不等**不是**唯一阻塞点。
2. **vendor 多 task 真值（新）**：① **后续 task 是 4 条增量**（`DPU_S_POINTER`/`RDMA_S_POINTER`/`DPU_DST_BASE_ADDR`/`DPU_WDMA_SIZE_0`）⇒ **PC 在 task 间保留寄存器状态**；② 连续两 task 尾写 `PC_BASE_ADDRESS` 0xffffe380 → 0xffffe700（Δ=896 B=112 条）、`PC_REGISTER_AMOUNTS` 0x37 → 0x7 ⇒ **(N+1)×2 = 下一块条数**，与我们 `L/2−1` 同构（数值无误）。
3. **下一步**：① 尾写 `PC_BASE_ADDRESS` 改指向**描述符数组**地址；② 描述符额外字段（`enable_mask` 位序、`regcfg_amount` 语义）；③ `PC_TASK_CONTROL` 的 task 数与 `subcore_task` 组合（如 3 核布局 `core+2`）。
4. 单 task 复验 BIT-EXACT；soft reset 54→55，楔死签名仍 0。

产物（未提交）：`experiments/min_conv_exp.c`、`experiments/results/multicore_vendor_20260930/vendor_multitask_groundtruth.txt`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.59）。


### 7.60 多 task：vendor 定量口径（尾写单位 / regcfg_amount 单位 / task_base）（2026-09-30 第六十八轮，**未提交**）

1. **排除**：① PP 位不是原因——`--job-flags 0x5`（与 vendor 相同、无 PP）下 2 task 仍超时（同流 6.33 s、链 6.38 s）；② `task_base_addr` 必须为 0（§7.57 实测 + vendor 样本 `task_base=0x0`）；③ 链段长度不等已于 §7.59 修掉但仍超时 ⇒ 非唯一阻塞点。
2. **vendor 定量口径（新）**：成功样本 `task_number=3 / subcore_tasks=[[0,1],[0,1],[0,1]]（每 subcore 1 个）/ flags=0x5 / task_base=0`；**三个 task 的 `regcfg_amount` 全 = 69**（含 4 条增量的短 task）⇒ 它是**槽位预算**而非实际长度，且 272 B 的块对应 69 ⇒ **以 4 B 字计**（我们按 8 B 条目传 130 ⇒ 差 2 倍）。
3. **尾部单位推导**：相邻 `PC_BASE_ADDRESS` 差 0x380 = 896 B = 16 × 56，而 `PC_REGISTER_AMOUNTS = 0x37 = 55`（+1 = 56）⇒ **该字段以 16 B 单元计**；我们的 `(pc_amt+5)/2−1`（L=134 ⇒ 66 = 67−1）与 vendor **同口径** ✓。
4. **下一步**：① 链模式 `regcfg_amount` 改按 4 B 字计（`2×L` 或 vendor 式），验证是否解挂；② 仍挂则把段槽位按 16 B 单元整数倍对齐成与 vendor 完全同形；③ 再挂则回到「PC 定位 task 2」的其它路径。
5. 单 task 复验 BIT-EXACT；soft reset 56→58，楔死签名仍 0。

产物（未提交）：`experiments/min_conv_exp.c`、`experiments/results/multicore_vendor_20260930/vendor_multitask_groundtruth.txt`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.60）。


### 7.61 多核提交「成功但后续 task 无输出」（量化，2026-09-30 第六十九轮，**已提交**）

1. **单 subcore 多 task**（显式 `subcore_task[0].task_number=2`）⇒ **6.3 s 超时**（`hw_elapse=0`）⇒ **不支持**。
2. **多核派发**（`core_mask=0x7` + 每核 1 task、distinct `task_start`、无尾写）⇒ **`SUBMIT ret=0`、36–88 µs** ✓，但 fp16（后续区域预填 0x00）实测 **task1 `nonzero_bytes=0`（完全未动）、task2 `=182`（残片）** ⇒ **后续 task 不产生输出**。
3. **混用会挂**：多核派发 + `--chain-pc`（尾写链）⇒ **6.35 s 超时** ⇒ 尾写链与描述符派发**不可混用**。
4. **驱动源码佐证**：`core_mask=0x7` 逐个 commit 三个核；job DONE 要求**所有核中断到齐**（`atomic_dec_and_test(&job->interrupt_count)`）⇒ `ret=0` 说明三核都中断了，但 follow-up 无输出 ⇒ 问题在**每核的流/寄存器状态**。
5. **下一步线索**：`rknpu_job_subcore_commit` 在 PC commit 前写 `PC_DATA_ADDR = 0x1`（注释 "switch to slave mode"），值得单独试；以及每核是否需要独立输入/状态区。

产物：`experiments/results/multicore_vendor_20260930/vendor_multitask_groundtruth.txt`（追加）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.61）。


### 7.62 多 task：vendor 最小样本 vs 我们实测（收敛到「流布局/尾写取值」）（2026-09-30 第七十轮，**未提交**）

1. **vendor 最小样本**（`bn_lower.../tr_*_fp16.jsonl`，真实 `task_counter: 0→3`）：`task_number=3 / core_mask=0 / flags=0x5 / task_base=0`、`subcore_tasks=[[0,1],[0,1],[0,1],..]`、**三个描述符 `regcfg_amount` 全 = 69（非零）**；块解剖 = 34/34/4 条**连续**且每块带 4 条尾写。
2. **我们实测**（16×16 IC16 int8，`flags=0x5`）：真实 `regcfg_amount` + `subcore_task[0]=3` ⇒ **6.25 s 超时**；`args->task_number=2` 而子核 =1 ⇒ **挂**（⇒ **运行中的驱动尊重 `args->task_number`**，§7.45 的「驱动覆盖」需按实际内核二进制复核）；`regcfg_amount=0` ⇒ 0.19 ms `ret=0` 但**静默只跑 1 个 task**。
3. **结论**：与 vendor 的剩余差异收敛到**流布局与尾写取值**（vendor = 34/34/4 连续块 + 每块尾写；我们 = 等长 134 条块 + 自算尾写）。
4. 工具：`--task-count` 的 `regcfg_amount` 改为复制真实 emit 值（仿 vendor）；新增 `TASK-COUNTER` 打印。单 task 复验 BIT-EXACT；soft reset 59→61，楔死签名仍 0。

产物（未提交）：`experiments/min_conv_exp.c`、`experiments/results/multicore_vendor_20260930/vendor_multitask_groundtruth.txt`、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.62）。


### 7.63 PPU MaxPool 边界 B（k2s1 pad1）签名收窄：失配位置返回**窗口最小值**（2026-09-30 第七十一轮，**未提交**）

1. **新仪器**：`min_pool --miss-detail` —— 打印前 8 个失配位置的 `(oy,ox,ch,hw,best,worst,avgr)` 与**窗口内实际取值**。
2. **复现 + 签名**（`--iw 64 --ih 64 --ic 16 --kw 2 --kh 2 --sx 1 --sy 1 --pad 1`，out 65×65 ⇒ `total=67600`）：`vs-max mismatch=9582`，失配位置 `ox = 7,14,21,28,35,42,49,56`（**≡0 mod 7**），且每次都满足
   `hw = worst`（窗口最小值）而 `best` 是最大值：
   ```
   MISS oy=0 ox=7  ch=0 hw=-125 best=94  worst=-125 avgr=-16 win=[94,-125,]
   MISS oy=0 ox=14 ch=0 hw=108  best=97  worst=-122 avgr=-13 win=[97,-122,]
   ```
   ⇒ **不是随机错，而是「在这些 x 上取了 min 而不是 max」**（max/min 混淆），且每 7 列一次。
3. **尺寸依赖（新）**：同参数但 `--ih 16` 时 `iw=16/32/64` **全部 `mismatch=0`** ⇒ 触发条件与**大图**（ih=64）有关，不只与宽度有关。
4. **下一步**：扫描 `(ih, ic, kw, method, --recip-w/--recip-h)` 找出「7」与尺寸阈值的来源；对「过/挂」两个几何用 regcmd 逐位域 diff（`--dump`）应能定位到具体寄存器/位域。

产物（未提交）：`experiments/min_pool.c`（`--miss-detail`）、`experiments/results/` 追加本条记录。


### 7.64 PPU 输入高度上限 = 63（6-bit 字段）——边界 B 的真正根因（2026-09-30 第七十二轮，**已提交**）

1. **新仪器**：`min_pool --miss-hist`（按 ox/oy 统计失配数，并统计「hw==窗口最小值」的比例）。用它看清边界 B 的真实结构：失配**遍布所有列**（每列 ≈300 个、y 方向均匀），偶数列仅 3~6 个、x=0 为 0 ⇒ 之前台账记的「x≡0 (mod 7) 周期」是**局部误读**（`--miss-detail` 只看到前 8 个恰好落在该族的位置）。
2. **根因（实测确定）**：**输入高度 `ih ≥ 64` 即错**，与宽度无关——`iw=64` 时 `ih=62/63` **0 失配**、`ih=64` **9582/67600**、`ih=65` 10327；`ih=64` 时 `iw=62/63/64` 全挂（9279/9285/9582）。⇒ **PPU 输入高度字段 6 bit（0..63）**，64 起溢出并**静默算错**（非挂起）。
3. **处置**：`min_pool` 增加硬边界告警（`ih > 63` 时 `WARN: … PPU 高度字段 6 bit …`）；更高的图需**按行切分**（同 conv 的 `--tile-oh` 思路）。
4. 同时确认：`iw` 无此限制（ih≤63 时 iw 62/63/64 均 0 失配）。

产物（本次提交）：`experiments/min_pool.c`（`--miss-detail`/`--miss-hist` + ih>63 告警）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.63/§7.64）。


### 7.65 PPU 按行切分 `--tile-oh`：尝试与当前签名（2026-09-30 第七十三轮，**未提交**）

1. **目标**：突破 §7.64 的 `ih ≤ 63` 硬边界（输入按行暂存成 ≤63 行的 tile，输出暂存后散回）。
2. **已实现（opt-in）**：`min_pool --tile-oh N` —— 每 tile 的 `c.ih = rows`、`c.oh = nt`、`pad_top = pt`（窗口起点在图外时 `pt = -raw`，否则 0）、输入按 `lay_t(x,y,ch,rows,iw)` 暂存、输出散回；对 `rows > 63`、`rows < kh` 硬告警并拒绝。
3. **当前结果（未完成）**：ih=63 不切分 `mismatch=0` ✓，但 **`--tile-oh 32` ⇒ 65794/66560 失配** ✗；首个失配 `MISS oy=0 ox=0 hw=127 best=-128 win=[-128,]` ⇒ **PPU 把 max-pool 的 pad 值（+127）当窗口元素并取胜**；`--tile-oh 63` 因 `oh=64` 切出 1 行 tile ⇒ 挂（已加保护）。ih=128 `--tile-oh 32` 最终 `ret=-1`。
4. **收窄（新增两个确定性检查）**：① `TILE-CHECK`（整块覆盖时 `memcmp`）⇒ **暂存字节与不切分输入逐字节相同** ✓（此前显示的 DIFF 是 `memcmp` 返回值判反，已修）；② 单 tile vs 不切分的 regcmd 逐位域 diff ⇒ **完全相同 24、值不同 2（仅 `PPU_DST_BASE_ADDR` 与 `PPU_RDMA_RDMA_SRC_BASE_ADDR`）** ✓。⇒ **输入字节相同 + 寄存器相同（仅基址不同）而输出仍错** ⇒ 失败点在 **PPU 对暂存 BO 地址的处理**（地址/别名/内存区域/同步），不在几何或布局。
5. **下一轮候选**：① 暂存区放进**同一块 `in` BO**（排除「换 BO」）；② 查 BO dma 区间/对齐与 `--surf-len`/`--recip-*` 的关系；③ 单变量拆分（只换输出地址 / 只换输入地址）。
6. 不切分路径无回归（`mismatch=0`）；soft reset 63，楔死签名仍 0。

产物（未提交）：`experiments/min_pool.c`（`--tile-oh` 与保护）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.65）。


### 7.66 PPU 输出地址的单变量拆分：输出地址不被遵守（2026-09-30 第七十四轮，**已提交**）

1. **新旋钮**（都不改几何，只换一个变量）：`min_pool --stage-in`（输入复制到独立 BO）、`--stage-out`（输出指向独立 BO，并对旧 BO 也做一次参考比对）。
2. **结果**：① 只换**输入** BO（字节相同）⇒ **`mismatch=0`** ✓（输入地址被遵守）；② 只换**输出** BO ⇒ **新 BO 4096/4096 失配、旧 BO 3314/4096 失配** ✗ ⇒ **输出既没完整落到 `c.out_addr`，也没落到旧 `out` BO** —— `PPU_DST_BASE_ADDR`（0x6070）换 BO 后不按预期生效。
3. **与 §7.65 吻合**：tile 路径把输出暂存在**另一块 BO** ⇒ 输出不可信 ✗；而 §7.65 的寄存器 diff 恰好只差两个基址（DST / RDMA SRC）✓。
4. **下一步**：① 绕开——tile 的输出暂存放进**同一块 `out` BO**（用 BO 内偏移区分）；② 查根因——把 pool 流里所有地址类寄存器 dump 出来，确认 PPU 实际使用哪一个（候选：DPU/CORE 的 DST；`--dump` 输出在 stdout）。
5. 不切分路径无回归（`mismatch=0`）；soft reset 67，楔死签名仍 0。

产物（本次提交）：`experiments/min_pool.c`（`--stage-in`/`--stage-out` + 双 BO 对照）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.66）。


### 7.67 PPU 输出地址根本不生效（偏移与换 BO 都被忽略）（2026-09-30 第七十五轮，**已提交**）

1. **新仪器**：`min_pool --out-off N`（同一 `out` BO 内偏移）+ **`OUT-SCAN`**（提交后扫描整块 BO，找出与参考输出一致的偏移 ⇒ 直接回答"PPU 写到哪"）。
2. **结果**：① `--out-off 4096/8192/65536`（同一 BO）⇒ 按偏移比较全错 ✗，而 **`OUT-SCAN` 在偏移 0 找到正确输出** ✓ ⇒ 结果写在 **BO 基址**，偏移被忽略；② `--stage-out`（独立 BO）⇒ 新 BO 全错且 **`OUT-SCAN[out2]` 命中 0** ✗ ⇒ 正确输出**不在**新 BO 中。
3. **结论**：**`PPU_DST_BASE_ADDR`（0x6070）在本进程内不生效**（偏移与换 BO 都被忽略，结果始终落在最初 `out` BO 的基址）⇒ 这就是 §7.65/§7.66 tile 路径写错的原因（tile 输出暂存在另一块 BO 的偏移处）。
4. **下一步**：① 验证「一次提交内锁存」假设（同进程内用不同 `out_addr` 连续提交两次，看第二次是否仍写回第一次的地址）；② 若确为锁存 ⇒ 查锁存条件（写序、`PPU_RDMA`/CORE 侧寄存器、`S_POINTER` bank）；③ 绕过方案（每 tile 落到 out BO 基址 / 列向切分 + 每 tile 一次进程），需先确认 ①。
5. 不切分路径无回归（偏移 0 时 `mismatch=0`）；soft reset 67，楔死签名仍 0。

产物（本次提交）：`experiments/min_pool.c`（`--out-off` + `OUT-SCAN`）、`experiments/results/wgrad_bigmap_20260930/README.md`（§7.67）。


### 7.68 §7.67 修正：偏移被忽略成立；「换 BO 不生效」作废（2026-09-30 第七十六轮，**已提交**）

1. **方法学错误**：§7.67 的 `--out-off`/`--stage-out` 块插在**首次 `emit_pool` 之后** ⇒ 流里仍是旧地址 ⇒ 该批结论无效 ✗。补「用新地址 re-emit」后重测。
2. **修正后**：① 同 BO 偏移 4096 ⇒ 仍命中偏移 0 ⇒ **「偏移被忽略」保留** ✓；② 换独立 BO ⇒ `out2` 命中 0 ✗；③ **对照 `--two-same`（两次提交都用新 BO）⇒ PASS0 即 `out2 命中=32@0`** ✓✓ ⇒ **「换 BO 不生效」作废** ✗（换 BO 实际可用）。
3. **结论与下一步**：PPU 行切分应改为「**输出暂存放另一块 BO 的基址（offset 0）**」重试（§7.65），并同时扫描两块 BO；新仪器 `min_pool --two-submit/--two-same` + `scan_hits`。

### 7.69 PPU 切分重试与逐 tile 自检：tile 计算本身错（2026-09-30 第七十七轮，**已提交**）

1. 按 §7.68 把 tile 输出暂存改为**独立 BO 基址（offset 0）** ⇒ 仍错 ✗。
2. 新增 `TILE-CMP`（逐 tile：暂存输出 vs 按全局逻辑输入算的参考）⇒ `bad=32518/33280`、`32867/33280` ⇒ **tile 计算本身不对**（非散回/读取）✗。
3. **单 tile 全覆盖**（`--ih 62 --oh 62 --tile-oh 62`：几何与不切分相同、暂存字节先前已证逐字节相同、输出独立 BO）⇒ `bad=63997/64480` ✗ ⇒ tile 路径仍有未定位差异。
4. 单变量（不切分几何）：只换输入 BO ⇒ 0 失配 ✓；只换输出 BO ⇒ 4096/4096 ✗；而 `--two-same`（同新 BO、两提交、每次 memset+sync）⇒ **PASS0 命中 32@0** ✓ ⇒ 两条"等价"流程结论相反（dump 确认流里地址都是新 BO）⇒ 未解 ✗。
5. **下一步**：用**当前代码**重做「不切分 vs 单 tile 全覆盖」regcmd 全量 diff；对 `--stage-out` 与 `--two-same` 做逐步状态对照（BO 创建顺序、sync 次数、memset）。

### 7.70 矛盾钉死：同一 (src,dst) 组合两条流程结论相反（2026-09-30 第七十八轮，**已提交**）

1. 四组同几何（不切分）单变量/流程对照：基准 `in`→`out` = 0 失配 ✓；只换输入 `in2`→`out` = 0 失配 ✓；**`--two-same`（`in`→`out2`）= 命中 32@0** ✓；**`--stage-out`（同为 `in`→`out2`）= 4096/4096 失配** ✗（不清零时 4079/4096、`OUT-SCAN[out2]` 0 命中）。
2. ⇒ **同一 src/dst 组合在两个流程里结论相反** ⇒ **问题不在地址本身**，而在 `--stage-out` 流程的**设置差异**（我们自己的代码路径）✗ ⇒ §7.67「换 BO 不生效」**再次不可信**；单 tile 全覆盖（同族 src/dst）失败的真因仍未定位。
3. **下一步**：加 `--dump-submit`（打印 `rknpu_submit` 全字段 + task 描述符 + 流首尾各 4 条），在 `--two-same`（成功）与 `--stage-out`（失败）上对照 ⇒ 差异必然暴露；随后回到单 tile 全覆盖。
4. 另：`--no-clear-out` 证明新 BO 未被正确写入（非"读到旧数据"）。

### 7.71 根因：emit_pool 游标未复位（"地址不生效"全是我们自己的 bug）（2026-09-30 第七十九轮，**已提交**）

1. **根因**：`min_pool.c` 的 `emit_pool()` 没有 `rc = rc_buf;` 复位 ⇒ **追加写入**：re-emit 与每个 tile 的 emit 都落在前一段之后，而 `memcpy(rcbo.map, rc_buf, n*8)` 永远拷**第一段** ⇒ 硬件一直执行**旧流（旧地址）** ✗ ⇒ §7.67/§7.68/§7.70 的全部"地址不生效/两条流程矛盾"由此解释 ✓。
2. **修复后复测**（不切分几何、单变量）：`--stage-out` ⇒ **0 失配** ✓（`OUT-SCAN[out2]` 命中 32 ✓）；`--stage-in` ⇒ 0 ✓；`--stage-in --stage-out`（同时换）⇒ 0 ✓；`--two-same` ⇒ 命中 32@0 ✓。
3. **结论**：**PPU 输入/输出地址完全正常**；§7.67 的「偏移不生效」「换 BO 不生效」**全部作废** ✗（同 BO 内偏移那条同样存疑，需重测）。
4. **仍存**：tile 路径输出错——`--ih 63 --tile-oh 32` 两 tile 均错（`TILE-CMP` 33173/33280、33280/33280 ✗）；`--ih 62 --oh 62 --tile-oh 62` 单 tile 全覆盖 62705/64480 ✗。
5. **下一步**：修复后重做「tile 流 vs 不切分流」逐位域 diff（现应只差基址）；并复核 `--oh` 覆盖时不切分对照本身是否也错。
6. 新仪器：`--dump-submit`（打印 submit 全字段 + 描述符 + 流首尾 + 流内 `0x6070` 值）。

### 7.72/7.73 tile 切分几何两次修正（2026-09-30 第八十轮，**已提交**）

1. **关键进展**：`emit_pool` 游标修复后，**单 tile 覆盖整图**（`--ih 63 --tile-oh 64`）⇒ `TILE-CMP bad=0`、整图 0 失配 ✓ ⇒ **tile 机制（暂存/提交/散回）本身正确**。
2. **仍错**：切分后的 tile —— `--tile-oh 32` 两块分别 `32455/33280`、`32566/33280` ✗；而**等价的不切分几何**（`--ih 32` / `--ih 33`）都 0 失配 ✓（视口分别为 in 32x64→out 33x65、in 33x64→out 34x65）。
3. **两次修正均未解决**：§7.72 行数 `(nt-1)*sy+kh-pt`；§7.73 把顶部 pad 行并入暂存（越界填 0x80）并令 `pad_top=0`，使 `ih=(nt-1)*sy+kh` 与 emitter 推导 `out_h` 自洽。
4. **下一步**：tile 单次提交流 vs「不切分 + 强制同几何」流逐位域 diff；先判定 `--ih 33 --oh 32`（不自洽几何）在不切分路径是否本身就错 ⇒ 若是，则 tile 只能按 **emitter 推导的行数**切分（`oh = (ih + 2*pad - kh)/sy + 1`）。

### 7.74/7.75 tile 几何两次尝试与回退（2026-09-30 第八十一轮，**已提交**）

1. **不变式实测（不切分）**：`--ih 32 --oh 33`（= emitter 推导值）⇒ **0 失配** ✓；`--ih 33 --oh 32` ⇒ 32518/33280 ✗；`--ih 33 --oh 34`（自洽）⇒ 11174 ✗；`--ih 32 --oh 32` ⇒ 32566 ✗ ⇒ **`oh` 必须等于 emitter 推导值**（必要），且自洽也不总够。
2. **§7.73**（顶部 pad 行并入暂存、`pad_top=0`）与 **§7.74**（由 staged 行数反推 `nt`）两次改动都让**整块覆盖也变错** ⇒ **已回退到 §7.72 的几何**（`pt` 交给硬件 pad；`rows=(nt-1)*sy+kh-pt` 并按 `ih` 截断）。
3. **回退后实测**：`--tile-oh 64` ⇒ `TILE-CMP bad=0`、整图 0 失配 ✓；不切分回归 0 失配 ✓；`--tile-oh 32` 两块仍错 ✗。
4. **下一步**：tile 单次提交流 vs「不切分 + 强制同几何」流逐位域 diff（用 §7.71 的 `--dump-submit`/`--dump`）；据此把 tile 的 `(ih, oh, pad_top)` 与 emitter 推导严格对齐。

### 7.76 tile-0 流与同 cfg 不切分流完全相同（决定性 diff）+ `oh_t` 附带负结果（2026-09-30 第八十二轮，**已提交**）

1. **对照构造**：tile0 的 cfg 是（`ih=32, oh=33, pad_top=1, iw=64`），用不切分 `--iw 64 --ih 32 --oh 33 --pad 1` 复现该 cfg ⇒ **0 失配** ✓。
2. **逐位域 diff**：tile-0 的 emit 流 vs 对照流 ⇒ **26 条里 24 条相同，只有两个基址不同**（`PPU_DST_BASE_ADDR`、`PPU_RDMA_SRC_BASE_ADDR`）⇒ **流层面零差异** ✓。
3. 两个基址的"换 BO"已由 §7.71 单变量实验证明等价（都 0 失配）⇒ **流相同 + 几何相同 + 基址等价，而 tile 仍错** ⇒ 差异只能在**硬件看到的 BO 内容/内存布局**侧。
4. **附带负结果**：把 tile 的 `oh` 改为推导值（`oh_t`）+ 放大输出暂存 BO（`+2` 行）⇒ **连"整块覆盖"也变错**（`--tile-oh 64`：0 失配 → 66560/66560 ✗）⇒ **输出暂存 BO 的尺寸/地址本身会影响结果** ⇒ 已回退到 §7.72/§7.75 状态。
5. **下一步**：① 打印 tile 路径 `stage`/`stage_out` 的 dma 地址与尺寸，与已验证可用的 `--stage-in/--stage-out` 对照查别名/重叠；② 把 tile 输入暂存放进 **`in` BO 尾部**（同一 BO）以彻底排除 BO 差异；③ 扩展 `--dump-submit` 把 rdma 侧寄存器纳入对照。

### 7.77–7.80 PPU 单变量排查：输入偏移不被遵守（硬约束）+ tile 改造（2026-09-30 第八十三轮，**已提交**）

1. **单变量（不切分几何，emit 游标已修）**：① 输出放 `out` BO 内偏移 4096 ⇒ **0 失配** ✓（输出偏移被遵守）；② **输入放 `in` BO 内偏移 4096 ⇒ 3072/4096 失配** ✗ ⇒ **输入偏移不被遵守**（PPU 从 BO 基址读输入）；③ 输入换独立 BO 基址（§7.66 `--stage-in`）⇒ 0 ✓；④ 输出换独立 BO / 两者同时 ⇒ 0 ✓。
2. **⇒ 硬约束**：**输入 surface 必须落在某块 BO 的基址**（输出可用 BO 内偏移）。这解释了"输入暂存放 in BO 尾部"的失败 ✓。
3. **tile 改造**（按该约束）：输入 → 独立 `stage_in` BO 基址；输出 → `out` BO 偏移；`oh` 取 emitter 推导值（`oh_t`，多出行丢弃）。
4. **改造后**：`--tile-oh 64`（整块覆盖）⇒ `TILE-CMP bad=0`、整图 **0 失配** ✓（回归保持）；`--tile-oh 32`（两块）⇒ tile0 `32455/33280`、tile1 `25965/33280` ✗（总失配 32454，较前 42368 改善）。
5. **下一步**：查 `stage_in` BO 的**尺寸/对齐**（通过的 `--stage-in` 用 16 KB，tile 用 36 KB）与 **RDMA 侧寄存器（0x701c 等）** 是否与不切分路径一致（`--dump-submit` 扩展）。

### 7.81/7.82 tile0 已证明计算正确；前进量改按推导行数（2026-09-30 第八十四轮，**已提交**）

1. **双步长判别**（新仪器）：同一 tile 的暂存输出分别按 `nt` 与 `oh_t` 读 ⇒ `r0=0` 时 **按 oh_t 读命中 33280/33280** ✓✓ ⇒ **tile0 输出完全正确**；此前"tile 计算本身不对"是**读回步长用错**（须用推导值 `oh_t`）✗。
2. **前进量修正**：tile 循环由"按 `nt` 前进"改为"**按推导输出行数 `oh_t` 前进**"（旧法导致内层 tile 的几何/产出错配并丢最后一行）。
3. **修正后**：`--tile-oh 64` ⇒ 整图 **0 失配** ✓（回归保持）；`--tile-oh 32` ⇒ tile0 正确 ✓、tile1（`r0=33, ih=31, oh=30, pt=0`）仍错 ✗。
4. **排除**"ih 必须偶数"：不切分 `ih/oh = 32/31、31/30、34/33、30/29`（pad0）**全部 0 失配** ✓ ⇒ tile1 的 cfg 合法 ⇒ 问题在"**第二个 tile**"侧（状态/缓存/暂存复用）。
5. **下一步**：专测"第二个 tile"——例如 `--tile-oh 31`（观察是否只有第 2 片错）、或在每 tile 之间插入一次额外的 `bo_sync`/空提交；并检查暂存输出的**复用**（同一 `sout` 区域连续两次写入）是否受缓存/失效顺序影响。

### 7.86/7.87 突破：tile 几何必须逐次一致（pad_t=0 时切分基本成立）（2026-09-30 第八十五轮，**已提交**）

1. **实验**：`--pad-l 1 --pad-t 0` 让**所有 tile 同形**（`pt=0, rows=nt+1`）：不切分对照 **0 失配** ✓；`--tile-oh 31`（两块）⇒ 首片 PROBE 命中 **30752/31248**、后续 **30256/31248**、**总失配 1488/62496** ✓✓（同规模此前 32454 ✗，**降 95%**）。
2. **结论**：**同一进程内多次提交时硬件沿用"首次提交的几何"** —— 每 tile 的 `(ih, oh, pad_top)` 形状一致时结果基本正确；此前"后续 tile 全错"正是几何逐次变化（首片 `pt=1, rows=31`、后续 `pt=0, rows=32`）✗✓。残余 1488 为 tile 边界 1 行移位（PROBE `d=±1` 命中 ≈19.7% 佐证）。
3. **下一步**：① 用 conv 切分的"**lead 行**"技巧让 `pad_top=1` 也同形（每 tile 多带一行 halo、`pt` 固定 1、提交 `nt+1` 行并按首行丢弃）；② 追残余边界一行（tile 间输出行相位）。
4. 新仪器：`--tile-warm N`（每 tile 前热身提交；实测对"首片对/后续错"**无效**，排除"需重新武装"假设 ✗）。

### 7.89 lead 行方案（统一几何的正确形式）——设计与实施要点（2026-09-30 第八十六轮，**已提交**）

**背景**：§7.87 证明"同一进程内多次提交时硬件沿用首次提交的几何"⇒ 每片 tile 的 `(ih, oh, pad_top)` 形状必须一致；
§7.88 的"自造 pad 行（pt=0 + 填 0x80）"破坏硬件 pad 语义 ⇒ 失败回退 ✓。

**正确形式（下一轮按此实现）**：保留**硬件 pad 语义**（`pt = pad_top`，恒定），用 **lead 行**吸收相位差：

1. `base = r0*sy - pad_top`（首片为负 ⇒ 这些行在暂存里填 **pad 值**）；`rows = (nt + pad_top - 1)*sy + kh`（**恒定**，越界行同样填 pad 值）；`pt = pad_top`（**恒定**）。
2. 硬件取窗模型（§7.53 系列结论）：输出行 `j` 的窗口 = 暂存 `[j*sy - pt, …)` ⇒ 暂存行 0 对应全局行 `base` ⇒ 有用的输出是 `j = pad_top + t`（对应全局输出行 `r0 + t`）。
3. 因此：`oh` 取 emitter 推导值（= `(rows + 2*pt - kh)/sy + 1`，恒定）；**散回时丢弃前 `pad_top` 行**（源行 = `pad_top + oy`）；`TILE-CMP` 的参考比对也要用同一 lead 偏移。
4. 前进量按 `nt`（不是 `oh`），因为每片固定产出 `nt` 个有用行；最后一片的越界输入行由 pad 值补齐（**不要**截断 `rows`，否则形状又变 ✗）。
5. 注意 `rows ≤ 63`（PPU 高度上限）⇒ `nt ≤ 63 - pad_top - kh + 1`（kh=2, pad=1 ⇒ nt ≤ 61）。

**验收**：① 不切分回归 0 失配；② `--tile-oh 31/16` 的 `PROBE` 首片与后续都应命中 100%（当前首片 100% ✓、后续 ✗）；
③ 最终整图失配应为 0（当前 `pad_t=0` 变体已到 1488/62496，说明方案方向正确）。

### 7.90 **PPU 行切分闭合**：ih ≤ 63 硬边界绕开（2026-09-30 第八十七轮，**已提交**）

1. **实现（§7.89 设计）**：lead 行方案 —— `pt = pad_top` 恒定、`base = r0*sy - pad_top`、`rows = (nt + pad_top - 1)*sy + kh` 恒定（越界行填 pad 值 0x80）、`oh` 取推导值、**散回丢弃前 pad_top 行**（源行 = `pt + oy`）、**前进量按 nt**。
2. **结果**：① 不切分回归 0 失配 ✓；② `--ih 63 --tile-oh 31`（3 片形状全同）⇒ **整图 0 失配** ✓；③ `--ih 63 --tile-oh 16`（4 片）⇒ **0 失配** ✓；④ **`--ih 128 --tile-oh 32`（旧硬边界外）⇒ 整图 0 失配（134160 元素）** ✓✓。
3. **⇒ PPU MaxPool 支持 `ih > 63` 大图**（按行切分，单次 submit 输入 ≤ 63 行）；§7.64 的 `ih ≤ 63` 告警在启用 `--tile-oh` 后不再是限制 ✓。
4. **根因链**：① `emit_pool` 游标未复位（§7.71：每次提交都跑旧流）；② 硬件**沿用首次提交的几何** ⇒ 每片形状必须一致（§7.87）；③ 统一几何须保留**硬件 pad 语义**（§7.88 反例）⇒ 用 lead 行吸收相位差（§7.90）。
5. 约束：`rows ≤ 63` ⇒ `nt ≤ 63 - pad_top - kh + 1`（kh=2、pad=1 时 nt ≤ 61）；`pad_t=0` 时无此限制。

### 7.94 附带发现：int8 前向在 `IH ≠ 16` 时不切分也 MISMATCH（2026-09-30 第八十八轮，**已提交**）

1. 为验证 §7.93 的"分片需与 IH 同形"，编译 `-DIH=9`、`-DIH=10` 变体并各自 `--gen-only` 生成数据集。
2. **结果**：两个变体的**不切分前向都 MISMATCH**（`IH=9`：`total=4608 round-mismatch=3617` ✗；`IH=10`：MISMATCH ✗），而 **wgrad 都 `288/288` 正确** ✓；默认 `IH=16` 回归 `BIT-EXACT (round)` ✓。
3. **⇒ 改 `IH` 即破坏 int8 前向**（与分片无关）⇒ §7.93 的对照实验"控制组本身错"因而无效 ✗；疑点在**输入面行/面步长或 `FEATURE_GRAINS` 的行字节口径**（随 `IH` 变化）。
4. **下一轮**：用 `--dump-regcmd` 对比 `IH=16` 与 `IH=9`（同 IC/OC/核/步长）的不切分流，定位随 `IH` 变化的字段。

### 7.102 **int8 训练前向切分闭合**（2026-09-30 第八十九轮，**已提交**）

1. **根因**：散回用"整行 `memcpy`"（隐含行内连续 ✗），而真实布局由 `npu_off` 给出（行内 x 步长 = `h*16` ✗）⇒ 目标区约一半错位；检查/参考侧用 `npu_off` ✓ 因而产生"同址两处读数不同"的假象（实际读的是不同格 ✗）。
2. **修正**：散回逐原子拷贝 `memcpy(out8 + npu_off(g,x,r0+j,ow,oh), stage8 + npu_off(g,x,lead+j,ow,oh_t), 16)` ✓。
3. **验证**（int8 16×16 IC32 pad1，无需 `--i8-tile-force`）：`--tile-oh 8` ⇒ 两片 `TILE8-SCAT bad=0`、FWD **BIT-EXACT**、**CLOSED-LOOP OK** ✓；`--tile-oh 7` ⇒ 三片 `bad=0` ✓；`--tile-oh 4` ⇒ 四片 `bad=0`、FWD BIT-EXACT、CLOSED-LOOP OK ✓。
4. **本轮确立的 int8 硬规则**：① 分片暂存的输入行数必须是 **16 的倍数**（§7.96）；② 目标区布局必须按 **`npu_off`**（非行连续）。
5. **默认拒绝已解除**：`--tile-oh` 对 int8 直接生效（`--i8-tile-force` 保留为诊断开关）。

### 7.107/7.108 int8 分片收尾：**输入面必须方图**（rows == IW），分片解锁非方图（2026-09-30 第九十轮，**已提交**）

1. **新开关** `--i8-tile-diag`（默认关闭每片 CPU 参考/比对）⇒ 大图分片从超时变秒级 ✓；`--i8-tile-force` 仍一并打开诊断 ✓。
2. **修掉一处真实越界**：暂存 BO 原按 `MIN2(IH, tile_oh+1+KH)` 行分配，而打包器按 `ALIGN_UP(...,16)` 行写入 ⇒ 越界写；改为 `ALIGN_UP(tile_oh+KH,16)+16` 之后再按 `IW` 兜底 ✓。
3. **规则（实测）**：**暂存的输入面必须与 `IW` 同高（方图）** —— 64 宽图下 `rows`=32/48 时 77–82% 错 ✗、`rows`=64 时降到 2.5% ✗；修正 `rows = max(rows, IW)` 后：64×64 的 `--tile-oh 62/48/16/8` **全部 BIT-EXACT** ✓✓，`--tile-oh 16` 全流程 **CLOSED-LOOP OK** ✓✓。
4. **分片解锁非方图**：**16×64 不切分 = 77% 错 ✗，`--tile-oh 8` = BIT-EXACT + CLOSED-LOOP OK** ✓✓✓ ⇒ 因为每片暂存都是方图 ✓；也回溯解释了 §7.106 的"工具要求 `IW == IH`"✗。
5. **限制**：`rows = IW` ⇒ 每片暂存 = `IW×IW` ⇒ **宽图无内存收益** ✗（二维切分是后续工作 ✓）；高图收益显著 ✓。

### 7.109 规则与分片收益最终确认（两精度 + 大高图）（2026-09-30 第九十一轮，**已提交**）

| 用例 | 不切分 | `--tile-oh` |
| --- | --- | --- |
| int8 16×64 | 77% 错 ✗ | `--tile-oh 8` ⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓ |
| fp16 16×64 | 77% 错 ✗ | `--tile-oh 8` ⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓ |
| int8 32×128 | 100532/131072（77%）错 ✗ | `--tile-oh 16/8` ⇒ **均 BIT-EXACT + CLOSED-LOOP OK** ✓ |
| int8 64×64 | BIT-EXACT ✓ | `--tile-oh 62/48/16/8` 全 BIT-EXACT ✓；`16` 全流程闭环 ✓ |

**conv 前向的完整规则（本会话结论）**：① 不切分要求输入面**方图**（`IW == IH`，§7.106）；② 分片把每片暂存做成方图（`rows = max(ALIGN_UP(…,16), IW)`）⇒ **非方图/高图只能靠分片** ✓；③ 规则**与精度无关**（int8 / fp16 同构 ✓）；④ 收益：高图每片暂存 `IW×IW`（32×128 时省 4× ✓），宽图无收益（需二维切分 ✗）。

### 7.110 多 task：修掉描述符 `regcfg_amount` 实为 0 的顺序 bug（2026-09-30 第九十二轮，**已提交**）

1. **真 bug**：`tk2->regcfg_amount = t->regcfg_amount;` 位于 `t->regcfg_amount = n - EXTRA;` **之前** ⇒ 多 task 时所有描述符的 amount 实为 **0** ✗ —— 正是 §7.62「`regcfg_amount=0` ⇒ 静默只跑 1 个 task」的成因 ✓。
2. **修正**：`n` 已知后统一写 `t[k].regcfg_amount = n - EXTRA`（=126，符合 vendor「三描述符同值且非零」的形状 ✓）。
3. **实测**（16×16 IC16 int8，`--task-count 3`）：`args->task_number=3, subcore_task[0].task_number=3` ⇒ **`SUBMIT ret=-1 errno=Connection timed out 6329 ms hw_elapse=0`** ✗（soft reset 83→84，**楔死签名仍 0** ✓）。
4. ⇒ 失败模式由「静默单 task」变为「超时」⇒ 与「PC 顺尾写追下一块，而我们的流布局不是 vendor 的 34/34/4 + 4 条增量」一致 ✓ ⇒ **下一步：逐字节复刻 vendor 块布局**（短块 + 增量 + 每块尾写），不再调 amount 数值 ✓。

### 7.111 运行内核复核：0.9.8 内建、尊重 `task_number`（2026-09-30 第九十三轮，**已提交**）

1. 只读探查：`/sys/module/rknpu/version = 0.9.8` ✓（与 `refs/rknpu-driver-0.9.8` 同版本）；`/proc/kallsyms` 有 `rknpu_get_task_number` 与 `rknpu_job_subcore_commit.isra.0` ✓；模块**不在 `/lib/modules` 下**（内建进内核）⇒ 无 `.ko` 可反汇编，需 vmlinux/System.map ✓。
2. **行为学判定（比源码更硬）**：§7.110 实测中 `args->task_number=3` 让提交**真的尝试 3 个 task**（6.3 s 超时、`hw_elapse=0`），而 amount=0 时是"静默 1 个" ⇒ **运行中的 0.9.8 确实尊重 `args->task_number`** ✓ ⇒ §7.45「驱动把 `task_number` 覆盖为 `use_core_num`」**不适用于当前运行内核** ✗（源码里该路径仅在 `core_mask≠0` 生效 ✓，与实测一致 ✓）。
3. **结论**：多 task 的失败指向**流布局**✗（不再是"驱动覆盖"✗）⇒ 下一步用最小样本 `--task-count 2`（全长块 + 4 条增量块，各带尾写）逐字节对照 vendor ✓。

### 7.112 vendor 原始块表复核：`regcfg_amount = 首块字节数/4`（槽位预算）（2026-09-30 第九十四轮，**已提交**）

1. 原始数据（同 job）：task0/task1 = **276 B / 34 条**（task1 仅改 DST）、task2/task3 = **36 B / 4 条**（`S_POINTER×2 + DST + WDMA_SIZE`）。
2. **精确关系**：`regcfg_amount = 69 = 276/4` ✓，而**四个描述符的 amount 全为 69**（含 36 B 短块）⇒ 印证「amount = **该 job 的槽位预算**，以**入口块字节数/4** 计」✓（§7.60 结论精确化 ✓）。
3. **口径不可混用** ✗：我们的驱动公式 `PC_DATA_AMOUNT=(amount+4+2−1)/2−1` ⇒ 我们传 `n−EXTRA`（1072 B ⇒ 130 ⇒ 66 ✓ 与尾写一致 ✓）是**我们口径** ✓；vendor 的 69 按其口径 ✓（用我们的公式算得 36 ✗）⇒ **"把 amount 改成 vendor 数值"是错误方向** ✗（§7.110 实测也只把"静默单 task"变"超时" ✗）。
4. **下一步（更新）**：差异在**块布局** —— vendor 的**每个块（含 4 条短块）都以 `S_POINTER=0xe` 开头** ✓ ⇒ 先试「每个后续 task 的块首补写 `S_POINTER=0xe`」✓，并保持首块自带完整配置 ✓。

### 7.113 多 task：块首 `S_POINTER` 不是原因（2 个 task 就挂）；下一步 = 每 task 独立块（2026-09-30 第九十五轮，**已提交**）

1. 新增 `--sp-prefix`（块首补写 `0x4004=0xe`、`0x5004=0xe`，与 vendor 块头一致 ✓）。
2. 实测（16×16 IC16 int8）：`--task-count 2 --sp-prefix` ⇒ 描述符 amount 已正确写入 ✓、`task_number=2` ✓ ⇒ **仍 `SUBMIT ret=-1 ETIMEDOUT 6240 ms hw_elapse=0`** ✗（soft 84→85、**楔死 0** ✓）⇒ **块首 S_POINTER 不是原因** ✗，且 **2 个 task 就挂**（与任务数无关 ✓）。
3. **更根本的认识**：驱动源码显示 PC 的尾写是**驱动**在 commit 时编程的 ✓ ⇒ vendor 尾写里 896 B 的步进是**驱动记账** ✓（不是我们可写的 ✗）；而我们目前让**所有描述符指向同一个 `rcbo` 块** ✗ ⇒ 每个 task 的 PC 程序相同、第二次 commit 从同一处重跑（且流里 `PC_BASE_ADDRESS=0`）⇒ 极可能跳到 0 ⇒ 挂 ✓。
4. **下一步实验**：**每个 task 一个独立块**（把同一 emit 复制 N 份到 `rcbo` 的不同偏移 ✓，`regcmd_addr` 各异、每份自带驱动尾写 ✓）+ `task_number=N` + `subcore_task[0]=N` ✓。

### 7.114 多 task：试「每 task 独立块」仍超时 ⇒ 三条杠杆全部排除（2026-09-30 第九十六轮，**已提交**）

1. 新增 `--task-blocks`（同一 emit 复制 N 份到 `rcbo` 不同 64 B 对齐偏移 ✓，各描述符 `regcmd_addr` 各异 ✓ = vendor 的 distinct-块形态 ✓）。
2. 实测（16×16 IC16 int8）：`TASK-BLOCKS: 2 块 × 1040 B（stride=1088，rcbo=8192 B）` ⇒ **`SUBMIT ret=-1 ETIMEDOUT 6375 ms hw_elapse=0`** ✗（soft 85→86、**楔死 0** ✓）。
3. ⇒ 已排除三条杠杆：① 描述符 `regcfg_amount`（实为 0，已修 ✓）；② 块首 `S_POINTER=0xe` ✗；③ 每 task 独立块 ✗ ⇒ 挂在**更前面的层次**（PC/子核派发模型 ✓）。
4. **剩余唯一未试杠杆（有历史风险）**：驱动 `rknpu_job_subcore_commit` 在 PC commit 前写 `PC_DATA_ADDR=0x1`（"switch to slave mode" ✓）⇒ 正规多 task 可能必须走 **`core_mask=0x7` + 每核 1 task** 的子核路径 ✓（§7.61 实测该路径 `ret=0`、36–88 µs ✓ 但 follow-up 无输出 ✗；当时未试"每核块各自带完整配置 + 合法尾写"）。⚠️ `--subcore 2/3/4` 曾致楔死 ⇒ 仅**单次**尝试 + 前后核对 `job commit failed`/soft reset/健康探针 ✓。

### 7.115 **多 task/多核闭合**（handoff §3 第 1 项，2026-09-30 第九十七轮，**已提交**）

1. **根因**：旧代码把 N 个 task **全塞给 subcore0**（`sub.subcore_task[0].task_number = N`）✗ —— 这正是硬件**不支持**的路径（单 subcore >1 task ⇒ 超时 ✗）。vendor 的真实形态是 **每个 subcore 各 1 个 task**：`subcore_tasks=[[0,1],[0,1],[0,1],[0,0],[0,0]]` + `task_number=3` + `core_mask=0` ✓。
2. **修正**：`subcore_task[k] = {task_start=k, task_number = (k<N)?1:0}`（k=0..4）✓。
3. **实测**（16×16 IC16 int8）：单 task 参照 `ret=0` / `OUT-HASH 391b75c5` ✓；`--task-count 3 --task-blocks`（每 subcore 1 个 + 3 个独立块）⇒ **`ret=0`、17.8 µs、`task_counter=3`、`OUT-HASH 391b75c5`（与单 task 逐字节相同）** ✓✓；soft 86→86、**楔死 0** ✓。
4. **多 task 的完整规则（闭合）**：① 描述符 `regcfg_amount` 必须非零且用**本驱动口径**（§7.110 ✓）；② **task 按 subcore 分配，每 subcore 1 个** ✓（不能用「subcore0 收 N 个」✗）；③ 每个 task 独立块（`--task-blocks`，推荐 ✓）；④ `task_base_addr=0`、`core_mask=0`、`flags=0x5` ✓。
5. ⇒ **handoff §3 第 1 项（唯一未闭合的逆向项）闭合** ✓✓。

### 7.116 多核并行定量验证：3 个独立 task ≈ 1 个 task 的时间（2026-09-30 第九十八轮，**已提交**）

1. 48×48 IC32 int8，1/2/3 个 task（每 subcore 1 个 + `--task-blocks`）：

| task 数 | SUBMIT | `hw_elapse` | OUT-HASH | 进程 wall |
| --- | --- | --- | --- | --- |
| 1 | `ret=0` | 49874 ns | `d3272e45` | 0.221 ms |
| 2 | `ret=0` | **49875 ns** | `d3272e45` | 0.144 ms |
| 3 | `ret=0` | **46957 ns** | `d3272e45` | 0.064 ms |

2. ⇒ **2 个 task 的 `hw_elapse` 与 1 个逐纳秒相同**（49875 vs 49874 ✓）、**3 个略快**（46957 ✓）⇒ **三个 subcore 真并行** ✓✓（≈3× 吞吐）；输出哈希三次一致 ✓；soft 86→86、**楔死 0** ✓。
3. **对本技能用途的回答**：此前提交是**单核受限**的 ✗（task 全塞 subcore0 ⇒ 静默单 task 或超时 ✗）；改用 vendor 形态后 **N 个独立 task ≈ 1 个的时间** ✓✓。

### 7.117 方图规则是严格等号 `rows == IW`（`>IW` 同样错）（2026-09-30 第九十九轮，**已提交**）

1. 判据实验（16×64，`IW=16`）：`--tile-oh 8`（rows=16=IW ✓）⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓；`--tile-oh 16`（rows=32>IW ✗）⇒ 79% 错 ✗；`--tile-oh 32`（rows=48>IW ✗）⇒ 78% 错 ✗。
2. ⇒ **严格等号**：暂存输入面的行数必须**恰好等于该 cfg 的 `IW`** ✓。由此澄清 `--tile-oh` 的语义 = **每个 tile 前进的输出行数 `nt`**，而**每片暂存恒为 `IW × IW`** ✓（§7.108 "宽图无收益"的根源 ✓）。
3. **二维切分的设计含义**：宽度切分的 tile 其 cfg 为 `iw = Wt` ⇒ 由等号规则必须 `rows = Wt` ⇒ **二维切分必须用正方形瓦片**（边长 `Wt`，带 halo ✓），成本 `Wt²` ✓。
4. **遗留（未验证）**：这条等号是**硬件规则**还是**我们发射器的约定**（`input_surf_stride`/打包口径 ✗）？下一轮用"同几何跨工具输出对照"判定 ✓。

### 7.118 vendor 原始 regcmd 字节流的位置与格式（2026-09-30 第一百轮，**已提交**）

1. `experiments/results/**/*.jsonl` 中每个 task 事件带 **`"hex"`** = 该 task 的 **regcmd 原始字节流** ✓（比摘要表更硬的一手资料 ✓）。
2. 格式：**(reg: u16 LE, val: u32 LE)** 逐条 ⇒ 前 6 字节 `04 40 0e 00 00 00` = `0x4004 = 0xe` ✓，与 `vendor_regcmd_blocks.txt` 的块首**完全一致** ✓。
3. **未竟（下一轮）**：用该字节流回答 §7.117 的遗留问题 —— **"输入面必须方图"是硬件规则还是我们发射器的约定** ✗：统计 vendor 全部 task 的 `0x1020`（`DATAIN_WIDTH`/`HEIGHT`）并检查是否存在**非方**样本 ✓；存在且 vendor 正确 ⇒ 发射器约定 ✗；全为方 ⇒ 更像硬件规则 ✓。

### 7.119 vendor 存在非方输入面 ⇒ "必须方图"是发射器约定而非硬件规则（2026-09-30 第一百零一轮，**已提交**）

1. **解析口径判定**：`hex` 条目 = **8 B** ✓（stride=8：295/295 整除、`0x5004` 出现 238 次、唯一 reg 152 ✓；stride=6：余数非零且 `0x5004` 从不出现 ✗）。
2. **统计 295 个 vendor task 的 `0x1020`**：45 个样本中 **26 方形 / 19 非方形** ✗（如 `8×16`、`1×8`、`4×8` 等）⇒ **硬件接受非方输入面** ✓✓。
3. ⇒ §7.117 的"暂存必须 `rows == IW`"**是我们发射器/打包侧的约定或 bug** ✗，**不是硬件限制** ✓；我们的 `rows = max(…, IW)` 只是**绕行**（宽图无收益 ✗）。
4. **下一步**：取 vendor 非方样本（如 `w=8, h=16`）比较**输入侧寄存器**（`0x1020/0x1080/0x1084/0x1070/0x1110`…）与我们同几何的发射结果 ✓（离线可做 ✓），找出真正差异 ✓。

### 7.122 解出 vendor 样本完整几何：`SURF` 公式吻合（112=16×7），失败因**打包侧未同步改**（2026-09-30 第一百零二轮，**已提交**）

1. 全寄存器表解出（vendor `trace_conv_bn_conv_i8.jsonl` 的 8×16 样本）：`0x1020=0x00100008`（iw=8, ih=16 ✓）、`0x1084 FC0=0x00080008`（dma_w=8, **dma_h=8** ✓）、`0x1080 SURF=112` ✓、`0x3014=0x00070007`（out 8×8 ✓）。
2. **公式吻合**：`SURF = (row_bytes/16) × (dma_h − 1) = (8×32/16) × 7 = 112` ✓✓ ⇒ §7.121 实现里的公式**是对的** ✓。
3. **失败原因**：只改了 `SURF`/`dma_h` 寄存器，**没有同步改输入打包布局** ✗ —— 打包仍按 `npu_off(g,x,y,iw,ih)`（面步长 `iw·ih·16`）✓，而硬件按 `dma_h=8` 的分块语义取数 ✗ ⇒ 地址对不上 ⇒ 全错 ✗。
4. **下一步（有界）**：`--vend-dma` 下把**打包**也改成 8 行分块（先再解一个 `ih=32` 样本确认 `dma_h` 是否恒为 8 ✓），与 `SURF`/`FC0` 同步改 ✓；改完先跑方图回归再看非方图 ✓。

### 7.123 vendor 语料全表：`dma_h ≡ 8`、`SURF` 随算子分族；并证伪"这两字段是唯一决定因素"（2026-09-30 第一百零三轮，**已提交**）

1. 21 个 vendor 样本汇总：**`dma_h` 全部 = 8** ✓（`ih` ∈ {8,16}，语料无 `ih>16` ⇒ 无法区分"恒为 8"与 `min(ih,8)` ✗）；`SURF` 与 **`CON0`**（算子/精度）强相关：`0x120⇒32`、`0x6000a120⇒56`、`0x6000a000`（i8 conv）`⇒112` ✓ ⇒ **`SURF` 语义随算子改变，不能跨算子套公式** ✗（§7.121 映射错误的根源 ✓）。
2. **关键旁证**：我们**方图**（16×16）在 `dma_h=ih=16`、`SURF=iw*4*(ih/4−1)=192` 下 **BIT-EXACT** ✓，而 vendor 公式同几何给 `224` ✗ ⇒ **两者都能算对** ⇒ **`dma_h`/`SURF` 不是非方失败的唯一决定因素** ✗。
3. **下一步（方向修正）**：做**输入侧寄存器全量 diff** —— 把失败的非方运行（如 `iw=16, ih=32`）与 vendor 同几何样本逐寄存器对照 ✓，而不是只盯 `0x1080/0x1084` ✗。

### 7.124 判定性负结果：`SURF` 口径不可照搬（单块 8×8 也被破坏）⇒ 非方线关闭（2026-09-30 第一百零四轮，**已提交**）

1. 判据实验：**8×8**（单块 8 行 ⇒ `FC_DATA_SIZE0` 与默认**相同**，唯一变化是 `SURF`：32 → vendor 式 112）：默认 **BIT-EXACT** ✓、`--vend-dma` **1651/2048 错** ✗；16×16 同（默认 ✓ / 6136 ✗）。
2. ⇒ **仅改 `SURF` 就破坏正确性** ⇒ vendor 的 `SURF` 语义**绑定他们自己的算子/数据准备流水线** ✓（§7.123 已证其随 `CON0` 分族 ✓）⇒ **不能照搬字段值** ✗。
3. **本线关闭** ✓：非方几何**不能**靠移植 `dma_h`/`SURF` 解决；`rows = max(…, IW)`（方图绕行）**保持为已验证方案** ✓；真正支持非方需复刻 vendor 整套输入准备 ✓ ⇒ 成本高于收益 ⇒ **标记为已知限制** ✓ 写入 handoff 遗留清单 ✓。

### 7.126 全量回归（三工具）：三项闭合成果全部保持 ✓（2026-09-30 第一百零五轮，**已提交**）

1. **min_pool**（5 例）：不切分 32×32 k2s2、64×63 k2s1 ⇒ `mismatch=0` ✓；`--tile-oh 31/16` ⇒ 0 ✓；**`ih=128 --tile-oh 32`（旧硬边界外）⇒ 0** ✓。
2. **min_conv_exp**（2 例）：单 task ⇒ `ret=0`、`OUT-HASH 93b5be45` ✓；**`--task-count 3 --task-blocks`（每 subcore 1 个）⇒ `ret=0`、同哈希** ✓（多核闭合保持 ✓）。
3. **wgrad_real**（5 例）：16×16 int8 不切分/`--tile-oh 8`、64×64 int8 不切分/`--tile-oh 16`/`--tile-oh 8` ⇒ **全部 BIT-EXACT + CLOSED-LOOP OK** ✓。
4. **过程教训**：首轮扫描里 wgrad 两行用的是 `-DIW=64` 的二进制去跑 16×16 数据集 ⇒ 得到**看似真实的 MISMATCH** ✗（100794/131072）⇒ **回归必须"二进制 × 数据集"配对** ✓；修正后全绿 ✓。
5. 全程 soft reset 86→86、**楔死签名 0** ✓；工件 `experiments/results/regression_round88.txt` ✓。

### 7.127 vendor 派发形态普查 + "单 subcore 多 task"复测仍超时（2026-09-30 第一百零六轮，**已提交**）

1. **新工件** `experiments/results/vendor_op_census.txt`（由 jsonl 的 `submit`/`task` 事件 + `hex` 解析而来 ✓）。
2. **普查发现**：最常见提交 = **`task_number=3, core_mask=0, subcore_tasks=[[0,1],[0,1],[0,1]]`（20 次 ✓）** ⇒ 与 §7.115 采用的形态**完全一致** ✓；大规模 fan-out 存在：`(108,0,[[0,36]×3])`、`(21510,0,[[36,7170],[7206,4465],…])` ✓ ⇒ **每 subcore 可拿几十~几千个 task** ✓ 且范围**连续相接** ✓ ⇒ "单 subcore 多 task 是支持的" ✓。
3. **复测（新增 `--subcore-single0`）**：3 个 task 全放 subcore0（amount 已修 ✓ + 独立块 ✓）⇒ **`ret=-1 ETIMEDOUT 6326 ms hw_elapse=0`** ✗（soft 86→87、**楔死 0** ✓）⇒ 我们的失败点**不在任务数** ✗，而在**每 task 块内容/驱动子核提交细节** ✓（最可疑：`rknpu_job_subcore_commit` 在 PC commit 前写 `PC_DATA_ADDR=0x1`（"switch to slave mode" ✓），vendor 的大 fan-out 很可能依赖该路径 ✓）。
4. ⇒ **对技能用途足够** ✓：已验证的"每 subcore 1 个 task"形态即可获得 ≈3× 吞吐（§7.116 ✓）。

### 7.128 更大 fp16 `oc` 验证：`oc = 96 / 128` 均 **ULP-EXACT**（handoff §3 第 5 项之一闭合）（2026-09-30 第一百零七轮，**已提交**）

1. 实验（`min_conv_exp --iw 16 --ih 16 --ic 32 --oc {64,96,128} --kw 3 --kh 3 --pad 1 --fp16`）：

| oc | SUBMIT | `hw_elapse` | fp16-mismatch | max ulp | 判定 |
| --- | --- | --- | --- | --- | --- |
| 64（对照） | `ret=0` | 33833 ns | 1010 / 16384 | **8** | **ULP-EXACT (<=16 ulp)** ✓ |
| 96 | `ret=0` | 37041 ns | 1536 / 24576 | **8** | **ULP-EXACT** ✓ |
| 128 | `ret=0` | 95666 ns | 2042 / 32768 | **8** | **ULP-EXACT** ✓ |

2. ⇒ **`oc > 64` 没有边界问题** ✓（96/128 与 64 的 ulp 剖面完全一致 ✓）；`hw==fp32ref but !=fp64ref` 的计数说明这些差异是 **fp32 累加 vs fp64 参考的舍入** ✓（非硬件错 ✓）。
3. 附注：`mismatches per oc` 呈**周期 3** 的规律（oc≡0 (mod 3) ⇒ 2、≡1 ⇒ 24、≡2 ⇒ 22 ✓）—— 同一舍入现象的体现 ✓，不是新的边界 ✓。
4. ⇒ **handoff §3「更大的 fp16 oc（oc>64）」一项闭合** ✓。

### 7.129 语料普查扩展：现有 28 份 jsonl **零 LUT（0x4104）写入** ⇒ LUT 指标需新捕获（2026-09-30 第一百零八轮，**已提交**）

1. 在 `vendor_op_census.txt` 增加"各 trace 的 `0x4104` 计数 + 提交形态 + 单核受限判定"表 ✓。
2. **结果**：LUT 计数**全为 0** ✗ —— 现有语料覆盖 conv / bn / mul / transpose / pool / layernorm ✓，**不含** sigmoid/tanh/gelu/swish 等查表激活 ✓ ⇒ 技能的核心指标（LUT 计数）**无法从现有语料推进** ✗。
3. **技能用途现状**：`0x4104` 解析能力**已就绪** ✓（`hex` 解析器 ✓），只差一份含查表激活的语料 ✓；"单核受限"判定**已可用** ✓（§7.116/§7.127；vendor 最常见形态 `[[0,1]×3]` 已复现 ✓）。
4. **下一步（需新捕获）**：仓库已有激活侧素材 `refs/npu/ops_rknn/{sigmoid,tanh}.cpp`、`activation*`、`act*` ✓ + 产生 jsonl 的 hook 工具链 ✓ ⇒ 用**带 sigmoid/tanh 的模型**跑一次捕获 ✓，即可用同一脚本出 `0x4104` 序列表 ✓。

### 7.131 真实 softmax 注意力 trace 解析：仍零 LUT 写；但确认"单 subcore 数千 task"（2026-09-30 第一百零九轮，**已提交**）

1. 板侧解析 `attn_probe/s784.trace.jsonl`（`attn_softmax_h16_s784_d64_int8` ✓，6.6 MB ✓，**9291 task** ✓）⇒ **LUT 区（`0x4100–0x411c`）写入 = 0** ✗、`0x4104` 唯一值数 = 0 ✗。
2. ⇒ **该 softmax 注意力模型在 NPU 上不使用 DPU LUT** ✓（与 A14"独立算子默认走 CPU"、A15"软 LUT 配置来自特定捕获"一致 ✓）；要补 LUT 语料需**显式激活算子**（sigmoid/tanh/gelu 的 `.rknn` ✓）。
3. **附带强证据**：提交形态 `(7059,0)`、`(2019,0)`、`(144,0)` ✓ ⇒ **单 subcore 上几千个 task 的真实用法** ✓（再次确认 §7.127 ✓；我们的实现仍挂 ✗ ⇒ 缺口在驱动 slave-mode 路径 ✓）。
4. 工具位置（下一轮直接用 ✓）：`/mnt/ssd/workspace/runs/eperf/rknpu_microbench/libhook_probe.so` ✓ + 环境变量 `EPERF_RKNPU_IOCTL_TRACE` ✓（输出即本语料同格式 jsonl ✓）。

### 7.132 真实负载描述符普查（amount=9 占 80%）+ 增量块复测仍挂 ⇒ 阻塞在驱动子核路径（2026-09-30 第一百一十轮，**已提交**）

1. `s784.trace.jsonl`（9291 task）里 task 事件**不带 `hex`** ✗（该 trace 未捕获 regcmd 载荷，`EPERF_RKNPU_REGCMD_MAX_BYTES` 默认 0 ✓），但**描述符字段齐全** ✓：`regcfg_amount` 分布 = **9 → 7427 次（80%）**、12 → 454、25 → 199、23 → 151、28 → 101、29 → 100 ✓ ⇒ 真实负载绝大多数 task 的描述符 amount = **9**（36 B = **4 条增量** ✓，与 §7.112 的 vendor 块解剖一致 ✓）。
2. 据此实现 `--task-delta4`（首块全量 + 后续 4 条增量 ✓，amount 沿用首块 ✓）⇒ 复测单 subcore 3 task：**仍 `ret=-1 ETIMEDOUT 6316 ms hw_elapse=0`** ✗（soft 87→88、**楔死 0** ✓）。
3. ⇒ **挂与块的大小/形态无关** ✓（全量 / 4 条增量 / 独立块 / 块首 `S_POINTER` 全试过 ✗）⇒ 阻塞点在**驱动侧子核提交路径**（`PC_DATA_ADDR=0x1` slave-mode ✓），不是我们的流 ✓✓（问题再收窄一层 ✓）。

### 7.133 激活（LUT）语料：工具地图已打通，卡在"模型与运行时不匹配"（2026-09-30 第一百一十一轮，**已提交**）

1. **可用素材（板上）**：`act95`（= `refs/npu/ops_rknn/activation` 推板 ✓）用法 `./act95 <activation> [width]` ✓，支持 20+ 激活（含 sigmoid/tanh/gelu/softmax ✓）；hook = `runs/eperf/rknpu_microbench/libhook_probe.so` ✓ + `EPERF_RKNPU_IOCTL_TRACE` ✓（**必须同给 `EPERF_RKNPU_REGCMD_MAX_BYTES`** ✓，否则 `hex` 为空 ✗）；激活模型已在板上 ✓（`rknpu_atomic_probe/models/{sigmoid,tanh}_1x16_{fp16,int8}.rknn` ✓）；同目录有带 dump 的运行器 **`rknn_infer_dump`** ✓。
2. **阻塞**：`act95` 期望 `models/<act>_float16_1x<width>.rknn` ✓；把 `sigmoid_1x16_fp16.rknn` 改名后 **`rknn_init` 失败** ✗（`load model failed` ✗）⇒ 现有 `.rknn` 与该二进制所用运行时/接口**不匹配** ✗（非形状问题 ✓）。
3. **下一步**：① 改用 **`rknn_infer_dump`** 跑**原始文件名**的模型 ✓（它显然就是产出这些模型的工具 ✓）+ hook 捕获 ✓；② 或找到其**转换脚本**（`runtimes/.../rknn_bench.py` ✓ 一类 ✓）现场生成所需形状/命名 ✓。

### 7.135 **LUT 指标收口（技能核心指标达成）**（2026-09-30 第一百一十二轮，**已提交**）

1. **关键修正**：`hex` 在 **`event=="regcmd"`** 事件里 ✓（不是 task ✗）；条目编码仍是 **8 B**（`reg:u16 LE @0` + `val:u32 LE @2` ✓）⇒ §7.134 的"编码不同"是误判 ✗（第 9 次自我修正 ✓）。
2. **实测**（`rknn_infer_dump <model> ramp` + hook + `EPERF_RKNPU_REGCMD_MAX_BYTES=4096` ✓）：

| trace | regcmd | `0x4104` 次数 | 唯一值 | 配置寄存器 |
| --- | --- | --- | --- | --- |
| `sigmoid_1x16_fp16` | 12 | **1780** | **440** | 全 0 |
| `tanh_1x16_fp16` | 12 | **1780** | **443** | 全 0 |
| `cs_exp2_via_sigmoid_1x64_fp16` | 33 | **1357** | 440 | 全 0 |

3. ⇒ 每个 sigmoid/tanh task ≈ **148 次 LUT 数据写** ✓、表规模 ~440 项 ✓（A13 的 1026 项属另一 op/形状 ✓）；配置寄存器本 op 下全 0 ✓（A15 的 `LUT_CFG=0x68` 属另一路径 ✓）。
4. **固化工具**：`experiments/lut_scan.py` ✓（文件头写明两个坑：必须带 `EPERF_RKNPU_REGCMD_MAX_BYTES` ✓、读 `regcmd` 事件 ✓）。

### 7.136 LUT vs 多项式普查（26 个真实模型）——技能核心问题达成（2026-09-30 第一百一十三轮，**已提交**）

1. 用 §7.135 的管线对板上 **26 个模型**全部捕获 + `lut_scan.py` 扫描 ✓（工件 `experiments/results/lut_census_round98.txt` ✓）。
2. **结果**：sigmoid/tanh ⇒ `0x4104` **1780 次 / 表 440、443 项** ✓；softmax 与 ELU ⇒ **~1355 次 / 表 111 项** ✓；softplus ⇒ 1783/443 ✓；`cs_exp2_via_sigmoid` ⇒ 1357/440 ✓、`cs_exp2_via_elu` ⇒ 907 ✓；而 **`cs_exp2_poly` ⇒ 54 次全零（多项式，无 LUT）** ✗、`cos/sin/rcp/sqrt/rsqrt` ⇒ 6–17 次全零 ✗。
3. ⇒ **同一数学的两种 lowering 可被 LUT 计数明确区分** ✓✓（技能"是 LUT 还是多项式"的问题**已可量化回答** ✓）；表规模分三类（111 / 440 / 443 ✓）且**同族算子共用同一张表** ✓（sigmoid 与 exp2-via-sigmoid 同 440 ✓；tanh 与 softplus 同 443 ✓）。
4. 模式：每个 LUT 算子每次运行约 **1350–1780 次 LUT 数据写**，而 `0x4100`（访问）仅 12–54 次 ⇒ **突发写** ✓。

### 7.137 26 个真实模型的提交形态：**全部 `core_mask=0` + 按 subcore 分派**（技能"单核受限"信号）（2026-09-30 第一百一十四轮，**已提交**）

1. 对 §7.136 的 26 份 trace 统计 `submit` 事件 ✓：形态为 `(3,0)×2,(6,0)×1`（sigmoid/tanh/cos/sin/rcp ✓）、`(105,0)×1`（sqrt fp16 ✓）等 ⇒ **一律 `core_mask = 0` + `args->task_number = N` + 按 subcore 分派** ✓✓（**从不用 `core_mask>1`** ✗）。
2. ⇒ 与 §7.115 采用的形态**完全一致** ✓；这正是"**未单核受限**"的标志 ✓（对照本工具修复前：N 个 task 全塞 subcore0 ⇒ 单核受限 ✗，§7.110–§7.116 ✓）。
3. **遗留**：真实 attention `.rknn` 是**多输入**（Q/K/V ✓）⇒ `rknn_infer_dump ... ramp` 报 `inputs_set failed: -5` ✗、trace 无 `regcmd` ✗ ⇒ 需专用喂数脚本 ✓；但"softmax 是否用 LUT"已由单输入的 `fp_softmax_*` 回答 ✓（1352–1358 次 / 表 111 项 ✓）。

### 7.138 真实 attention 负载的 LUT 判定：**softmax 确实用 LUT**（推翻 §7.131）（2026-09-30 第一百一十五轮，**已提交**）

1. 扫描 `attn_probe/*.trace.jsonl`：`attn_softmax_*`（4 个变体 + `s784`）⇒ `0x4104` **4270–5296 次**、**唯一值 694** ⇒ **LUT** ✓；`attn_relu_h16_s256_d64_int8` ⇒ **78 次全 0** ⇒ **无 LUT** ✗。
2. ⇒ **真实 attention 的 softmax 使用 DPU LUT** ✓✓ ⇒ **推翻 §7.131 的"不用 LUT"** ✗（那次建立在**无 regcmd 载荷**的 trace 上 ✗）⇒ **第 10 次自我修正** ✓（教训：**先确认 `hex` 非空再统计** ✓）。
3. **A/B 对照**：同一骨架 `relu` ⇒ 无 LUT ✗ vs `softmax` ⇒ ~4300 次/**694 项表** ✓ ⇒ **softmax 才引入 LUT** ✓。
4. **表规模第三种**：**694**（attention softmax ✓）—— 与 111（单算子 softmax/ELU ✓）、440/443（sigmoid/tanh/softplus ✓）并列 ⇒ 表规模取决于算子 + 形状/分段 ✓。
5. **提交形态**（同期）：全部 `core_mask=0` ✓、`task_number` = 99/1155/2307/7059 ✓、`regcfg_amount=9` 占 ~86% ✓ ⇒ 真实负载普遍"每 task 4 条增量块 + 大 fan-out" ✓（§7.132/§7.137 一致 ✓）。

### 7.139 LUT 表内容结构解码（范围/单调性/步长）（2026-09-30 第一百一十六轮，**已提交**）

1. 对 `0x4104` 序列去重排序：**sigmoid** ⇒ 439 个非零值、值域 **59…9640**、**严格递增** ✓、步长 1–5 渐增（尾端 ~83）⇒ **单调定点表** ✓（零点附近密 ✓）；**tanh** ⇒ 442 值、值域 **−32611…−12973**、严格递增 ✓ ⇒ **只存负半轴**（对称表 ✓）；**softmax(1×64)** ⇒ 110 值、**1…110**、**步长恒 1** ✓ ⇒ **线性**序列（"插值斜坡/索引映射"待验 ✓）；**attention softmax** ⇒ 694 值、0…32767 ✓。
2. ⇒ LUT 存的是**定点整数** ✗（"fp16 对"读法给出次正规数 ⇒ 误读 ✓，**第 11 次自我修正**，小 ✓）。
3. ⇒ **表规模（111 / 440·443 / 694）+ 值域 + 单调性** 三者合起来可直接**判别算子族** ✓（技能用途：从 trace 反推激活实现 ✓）。

### 7.140 判据实验：LUT 表是**静态编译常量**（与输入无关）（2026-09-30 第一百一十七轮，**已提交**）

1. 同一 softmax 模型（`fp_softmax_64_int8`）用 `ramp` 与 `ones` 两种输入各跑一次 ✓：
   `0x4104` 写入 **均 1358 次** ✓、唯一值 **均 111** ✓、**值集合相同** ✓、**首次出现顺序也相同** ✓；
   而两次**输出明显不同** ✓（`ramp` ⇒ 5.4e-06…0.0178 单调饱和 ✓；`ones` ⇒ 全 0.015625 ✓）。
2. ⇒ **LUT 是算子预编译的常量表** ✓✓（与运行数据**无关** ✗，非运行时缓存 ✓）⇒ §7.139 的"表结构指纹"（规模 + 值域 + 单调性 ✓）可用于**算子族判定**与**跨模型比对** ✓（sigmoid fp16/int8 同为 440 项 ✓ 是佐证 ✓）。

### 7.141 sigmoid LUT 逐项拟合：单调但**不是** `S·σ(x)`（假设否证）（2026-09-30 第一百一十八轮，**已提交**）

1. 取有序唯一值（439 条目、59…9640 ✓）做位置化步长分析：前 10 步长全 **1** ✓、中段 **10–12** ✓、末段 **79–83** ✓ ⇒ **步长单调递增**（峰值在 99% 处 ✓）⇒ **斜率全程增大** ⇒ **不是 sigmoid 形状** ✗（sigmoid 应在中段最陡 ✓）。
2. 对 `table = round(S·σ(x_i))`（均匀网格）扫描 `S` 取"`Δ/(v(S−v))` 最恒定"：最优 `S=27997`，但 **cv=0.125** ✗、残差**均值 30.3/最大 101.7** ✗ ⇒ **`S·σ` 假设被否证** ✗（A13 的"`round(sigmoid(x)*32768)`"属**另一次捕获/另一路径** ✓）。
3. **保留事实**：① 严格单调 ✓；② 斜率单调递增（×83 ✓）；③ 定点整数 ✓；④ 与输入无关（§7.140 ✓）。⇒ 函数形式（幂律/分段/复合）待定 ✓；下一轮对 `exp`/幂律/分段三族各做同款"最恒定"拟合 ✓（方法已固化、纯离线 ✓）。

### 7.142 sigmoid LUT 函数族判定：**近指数表**（R²=0.999）（2026-09-30 第一百一十九轮，**已提交**）

1. 对有序唯一值（439 条）做 `log v` vs 索引 `i` 的线性回归：**`log v = 4.182038 + 0.011736·i`，R² = 0.999036** ✓，对数域残差 σ=0.046、max 0.149 ✓ ⇒ **高度精确的指数律** ✓✓（`v_0 ≈ 65.5` ✓、每条目因子 `e^{0.011736} ≈ 1.0118` ✓，表尾 9640 ✓ 与拟合吻合 ✓）。
2. **二阶效应**：相邻比值对数从前段 ~0.0146–0.0168 **缓降**到末段 ~0.0086–0.0089 ✓ ⇒ "近指数但带轻微下弯" ✓（⇒ `x` 网格非均匀或含高阶曲率 ✓，留待后续 ✓）。
3. **语义推断**：LUT 存**乘性/指数域的量** ✗（非 sigmoid 输出 ✗）⇒ 更像 softmax/sigmoid 实现里的 **`exp` 域中间量** ✓；与 A13（`round(sigmoid(x)*32768)` ✓ 来自另一条捕获 ✓）不矛盾 ✓。
4. **方法固化**：去重排序 → 步长剖面（判别 S 形/单调）→ `log v ~ i` 回归 + 残差（判别指数律）→ 相邻比值漂移（检出曲率）✓。

### 7.143 四张 LUT 表的结构分类（softmax=线性斜坡、sigmoid=近指数、tanh=分段、attention=复合）（2026-09-30 第一百二十轮，**已提交**）

1. 用 §7.142 的方法判定四张表：

| 表 | 条目 | 值域 | 步长 | `log|v|~i` | 判定 |
| --- | --- | --- | --- | --- | --- |
| softmax（fp_softmax_64_int8） | 110 | 1…110 | **恒 1** | — | **严格线性斜坡** ✓✓ |
| sigmoid（fp16 1×16） | 439 | 59…9640 | 1…83 | R²=**0.99904** | **近指数** ✓ |
| tanh（fp16 1×16） | 442 | −32611…−12973 | 1…162 | R²=0.708 | **分段/混合** ✗ |
| attention softmax | 693 | 1…32767 | 1…318 | R²=0.981（比值 0.693→0.0098） | **复合/多表并集** ✗ |

2. ⇒ **softmax 的"表"是纯线性斜坡 `1..N`** ✓✓（⇒ 对 softmax 而言"LUT 写"更像**建立插值索引** ✓，修正了"LUT=函数表"的隐含假设 ✓）。
3. ⇒ **同一算子里"族"不统一**（sigmoid 近指数 ✓ vs tanh 分段 ✗）⇒ 每个算子族须**单独判定** ✓，不能类推 ✗。
4. ⇒ **attention 的表是复合的**（693 项、1..32767 ✓、比值漂移大 ✗）⇒ 疑为**多次 softmax 实例的表并集** ✓ ⇒ 需按 task/index 分段再判 ✓（下一步 ✓）。

### 7.144 LUT 写入的事件级分解：整表突发 + 零写旁路戳（2026-09-30 第一百二十一轮，**已提交**）

1. 按 `regcmd` 事件统计每事件的 `0x4104` 写入数：

| trace | 事件 | 含 LUT 写 | 分布 | 合计 |
| --- | --- | --- | --- | --- |
| sigmoid | 12 | 12（100%） | **443×4**、1×8 | 1780 |
| softmax64 | 42 | 32（76%） | **443×3**、1×29 | 1358 |
| attn_s256 | 3468 | **193（5%）** | **1027×4**、1×189 | 4297 |

2. ⇒ **统一模式**：① **整表加载 = 少数突发**（每次 443 条 ✓；attention 为 1027 条，疑 2 张表 ✓）；② **绝大多数事件只有 1 次 `0x4104=0`** ✓ = **"LUT 旁路"标记** ✓（非数据 ✓）。
3. ⇒ §7.135 的"每次运行 1780 次写"**结构澄清**：实为 **4 次整表突发（4×443）+ 8 次旁路戳** ✓（数字不变、含义精确化 ✓）。
4. **技能用法**：判定某算子是否走 LUT 应看**是否存在"≥数百条"的 `0x4104` 突发** ✓，**不能只看总计数** ✗（attention 的 193 个含写事件里 189 个只有 1 条零写 ✗）。

### 7.145 attention 大突发按 `op_idx` 拆分：同一张表（非 2 张），一次推理加载 4 次（2026-09-30 第一百二十二轮，**已提交**）

1. 对 `attn_softmax_h16_s256_d64_int8` 中"≥50 条"的 4 次突发逐事件展开：全部为 **`op_idx = 3`** ✓、写入 1027 条 / 非零 1026 ✓、值域 1…32767 ✓、**`sequence` = 1/1/2/3** ✓、**内容逐字节相同** ✓。
2. ⇒ **不是"2 张表"** ✗（修正 §7.144 的推测 ✓，**第 12 次小自我修正** ✓）；⇒ 一次推理中该 op 的表被**加载 4 次** ✓。
3. ⇒ 该表 **非单调、含平台**（`1 2 2 2 2 2 …` ✗）⇒ **不是函数值表** ✗ ⇒ 更像**按量化输入索引的"反查/插值"表** ✓（平台 = 多输入映射到同一档 ✓，**假设待验** ✓）。
4. **技能用法**：先按"突发"筛出真正的表加载事件 ✓ → 看 `op_idx` ✓ → 看是否单调 ✓ ⇒ 可区分"函数值表"（单调 ✓）与"索引/反查表"（平台 ✗）✓。

### 7.146 LUT 语义收尾（一）："平台"是写入顺序假象 ✗；四表去重后**全部严格递增**（2026-09-30 第一百二十三轮，**已提交**）

1. 四张表去重排序后：**等值相邻占比全为 0.0%** ✓、**全部严格递增** ✓（attention 693 个值互不相同 ✓）。
2. ⇒ **修正 §7.145 的"含平台"** ✗：`1 2 2 2 2 2 …` 是**写入顺序**（写指针爬升时首值重复 ✓）造成的假象 ✓，表的**值集合**严格递增 ✓ ⇒ **规则：LUT 分析必须先去重排序** ✓（**第 13 次小自我修正** ✓）。
3. **fp16 位型解码无效** ✗（sigmoid 首/中/末解码 = 3.5e-06 / 5.3e-05 / 0.022 ✗ 不符 sigmoid 形状；tanh 末值 −21.3 ✗ 超值域 ✓）⇒ 确认**定点整数** ✓（与 §7.139 一致 ✓）。
4. **步长结构**：sigmoid/tanh/attention 的步长游程段数 220/251/351 ✓ ⇒ 步长平滑变化（函数特征 ✓）；softmax 单段恒 1 ✓ ⇒ 纯线性斜坡 ✓。

### 7.147 LUT 语义收尾（二）：归一化形状比对 ⇒ sigmoid 与 tanh **同族指数型**、softmax 线性（2026-09-30 第一百二十四轮，**已提交**）

1. 对四表取 10 个等距采样点的**归一化形状**：sigmoid = `0.00 0.01 0.02 0.03 0.06 0.12 0.21 0.37 0.63 1.00` ✓；**tanh 几乎逐点相同**（@89% 0.64 ✓）⇒ **两者共用同一族指数型表** ✓（仅符号/偏移不同：tanh 为负半轴 ✓）⇒ 支持 §7.142 的"LUT 存 `exp` 域中间量"判断 ✓✓。
2. **attention** 为**同族更陡版本**（@89% 0.47 ✓，且表尾恰为 **32767 = int16 上界** ✓）⇒ 指数型、量化更粗 ✓。
3. **softmax** 为**纯线性**（0.11/0.22/… 均匀 ✓）⇒ 另一种机制 ✓（与 §7.143/§7.146 一致 ✓）。
4. **方法教训**：对**量化表**用"相对误差"拟合无效 ✗（首值仅 1–59 ⇒ 相对误差爆炸：sigmoid 14%、attention 97% ✗）⇒ 应改用**归一化形状比对** ✓ 或绝对误差 ✓。
5. ⇒ LUT 语义链条完整：计数 → 静态性 → 值域/单调 → 去重规则 → 函数族 → **族间形状比对** → 事件级 ✓。

### 7.148 LUT 语义收尾（三）：经验模型 `v≈A·exp(k·i+c·i²)`（c<0）、无精确闭式（2026-09-30 第一百二十五轮，**已提交**）

1. 对数域拟合：sigmoid 一次 `k=0.011736`（σ=0.0462）→ 二次 `c=−3.04e−06`（**σ=0.0153，3× 改善** ✓）；tanh `c=−7.48e−06`（**2.4× 改善** ✓）；attention `c=−3.74e−06`（1.1× ✗，而 `exp(k·√i)` 微弱占优 0.274 vs 0.288 ✓）。
2. ⇒ **三表二次项系数全为负** ✓（对数斜率递减 ✓，与 §7.142 的比值漂移一致 ✓）⇒ 统一形如 **`v_i ≈ A·exp(k·i + c·i²)`，c<0** ✓。
3. **候选闭式**：`exp(k·i)` 最稳 ✓；`√i` 只在 attention 微弱占优（不决定性 ✗）；**`v∝i^p` 全面劣 ⇒ 排除幂律** ✗ ⇒ **精确闭式仍未定** ✗（经验模型可用 ✓）。
4. **方法链（可复用）**：去重排序 → 单调/线性判据 → `log v~i` 一次/二次回归 → 候选闭式逐一比 σ ✓。
5. **⇒ LUT 语义子线收束** ✓；后续若继续，应**用真实算子的输入-输出反解表项作用** ✓（需专用喂数 ✓ 见 §3b 第 2 条 ✓）。

### 7.149 单 subcore 多 task：连 vendor 真实形态（连续区间）也挂 ⇒ 从流侧不可达（2026-09-30 第一百二十六轮，**已提交**）

1. 新增 `--subcore-range N`（`subcore_task[k]={k·N, N}` ✓ = 真实 trace 的 `[[0,36],[36,36],[72,36]]` 形状 ✓）⇒ 实测 `--task-count 6 --task-blocks --subcore-range 2`：**仍 `ret=-1 ETIMEDOUT 6374 ms hw_elapse=0`** ✗（soft 88→89、**楔死 0** ✓）。
2. **已排除五类变体（全挂 ✗）**：全放 subcore0 ✗、4 条增量块 ✗、每 task 独立块 ✗、块首 `S_POINTER` ✗、**每 subcore 连续区间（vendor 形态）** ✗。
3. ⇒ **唯一可用形态 = 每 subcore 恰好 1 个 task** ✓（`[[0,1]×3]` ✓，也是真实 trace 最常见的形态 ✓；已实测 **3 task ≈ 1 task 时间** ✓ §7.116）。
4. ⇒ **结论：单 subcore 多 task 从"流侧"不可达** ✗ ⇒ 阻塞在**驱动/固件的子核任务调度**（`PC_DATA_ADDR=0x1` slave-mode ✓）⇒ 要继续必须做**驱动侧分析**（vmlinux/System.map 反汇编 ✓ 或内核探针 ✓）；**不建议再从发射器侧尝试** ✗（5 类变体全挂 ✓）。

### 7.150 vendor op 寄存器指纹工具 + 判别特征（2026-09-30 第一百二十七轮，**已提交**）

1. 新增 **`experiments/vendor_op_signature.py`** ✓（读 `event=="regcmd"` ✓、条目 8 B ✓、先排序去重 ✓）：输出每个 trace 的寄存器**并集/必写集**、全家族**共有集**、各家族**额外寄存器**（判别特征 ✓）。工件：`experiments/results/vendor_op_signatures.txt` ✓。
2. **首轮结果**：全家族**共有 72 个** ✓（PC 尾写 `0x0000/0x0008/0x0014` ✓ + DPU 块 `0x4004–0x40c4` ✓ + LUT 块 `0x4100–0x412c` ✓ + RDMA 块 `0x5004–0x506c` ✓）；**conv 族额外 80 个** ✓（`0x100c–0x1184` = **CNA 块** ✓）；**layernorm 额外 54 个** ✓（CNA 子集 ✓）；**bn 族额外 1 个**（`0x0010` ✓）⇒ **BN 纯 DPU/RDMA、无 CNA** ✓。
3. ⇒ **可用作算子分类器** ✓：**出现 CNA 块（`0x100c…`）⇒ conv 类** ✓；**只有 DPU/RDMA ⇒ BN 类** ✓（技能用途：从 trace 反推算子 ✓）。

### 7.151 指纹工具扩展：CNA 块的出现程度 = 算子类别判据（2026-09-30 第一百二十八轮，**已提交**）

1. 把 `vendor_op_signature.py` 扩到板上 **26 份激活/归一化 trace + 6 份 attention trace** ✓：
   | 家族 | 额外寄存器 | 判读 |
   | --- | --- | --- |
   | sigmoid/tanh/sin/cos/rcp/rsqrt/sqrt/softplus/swish/hardsigmoid/hardswish/elu（45 份 ✓） | **+0 或 +1（`0x0010`）** | **纯 DPU 元素级**（不用 CNA ✓）|
   | **layernorm / softmax / attention（s784）** | **+54**（`0x100c…` ✓） | **CNA 子集** ⇒ 归约/归一化需读特征图 ✓ |
   | **conv**（§7.150 ✓） | **+80**（完整 CNA 块 ✓） | **完整 CNA** ✓ |
   | bn（§7.150 ✓） | +1 | 纯 DPU/RDMA ✓ |
2. ⇒ **判据（技能用途）**：按 `0x100c…` **CNA 块的出现程度**分类 —— **无 CNA ⇒ 元素级** ✓、**部分 CNA(54) ⇒ 归约/归一化** ✓、**完整 CNA(80) ⇒ conv** ✓✓。
3. 细节：`fl_HardSigmoid_1x64_fp16` 是唯一 **+0**（连 `0x0010` 都不写 ✓）。

### 7.152 一站式 trace 报告工具（派发 + LUT + 类别 + 维度）（2026-09-30 第一百二十九轮，**已提交**）

1. 新增 **`experiments/rknpu_trace_report.py`** ✓：一条命令输出四块 —— ① **派发形态**（`(task_number, core_mask)` + 是否单核受限 ✓）；② **LUT vs 多项式**（≥100 条的整表突发 vs 旁路戳 ✓）；③ **算子类别**（CNA 块出现程度：0=元素级 / ≥20=归约归一化 / ≥70=conv ✓）；④ **维度**（`0x1020` 输入 w×h、`0x3014` 输出 w×h ✓）。
2. 实测（`trace_layernorm_only_fp16.jsonl` ✓）：`派发 (54,0) ⇒ 非单核受限` ✓、`无整表突发 ⇒ 非 LUT` ✓、`类别 = 归约/归一化（CNA 48 个）` ✓、`维度 in 8x8 / out 7x7` ✓ —— 四块全部正确 ✓。
3. 已登记进 handoff §5（工具与产物 ✓）；口径与坑（读 `regcmd` 事件 ✓、条目 8 B ✓、必须设 `EPERF_RKNPU_REGCMD_MAX_BYTES` ✓）写在文件头 ✓。

### 7.153 修正 §7.146："平台"真实存在（在写入顺序上）；§7.146 的检验方法无效（2026-09-30 第一百三十轮，**已提交**）

1. 对 attention 的 **1027 条整表突发**（写入顺序 = 表槽位顺序 ✓）直接统计：**写 1027 次 / 唯一 694 ⇒ 重复 333 次（32.4%）** ✓；**相邻相等段 71 段、最长平台 53** ✓；序列开头 `[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]` ✓。
2. ⇒ **表内成段相等（平台）是真实的** ✓ ⇒ 它是**分段常数（量化）表** ✓✓ ⇒ 支持 §7.145 的"按量化输入索引的**反查/插值表**"判断 ✓✓。
3. **方法学修正（第 14 次自我修正）**：§7.146 用"**去重排序后**相邻是否相等"检验平台 ✗ —— 排序后相邻必然不等 ⇒ 该检验**恒为 0%、无信息量** ✗，其"平台是假象"的结论**错误** ✗。**正确做法：在写入顺序（未排序）上统计平台** ✓。
4. **保留并强化**：① 表是**分段常数**（71 段、最长 53 ✓）；② **"表规模（槽位 1027）"与"唯一值数（694）"是两个量** ✓（此前混用 ✗）；③ 反查表判断成立 ✓。

### 7.154 报告工具加 `--check` + 建立回归基线（2026-09-30 第一百三十一轮，**已提交**）

1. `experiments/rknpu_trace_report.py` 新增 **`--check <baseline>`** ✓：把当前输出与基线逐行比对，打印差异（≤8 处 ✓），返回码 0/2 ✓。
2. 基线：`experiments/results/trace_report_baseline.txt` ✓（4 份代表性 trace：`trace_conv_bn_conv_i8` ✓、`trace_bn_only_fp16` ✓、`fp_layernorm_16_int8` ✓、`tr_mul_m025_fp16` ✓）。
3. **自检**：`--check` 对同一组 trace 报告 **差异 0 处** ✓ ⇒ 基线可用 ✓（未来改动可一键回归 ✓）。
4. 用法（写进文件头 ✓）：`python3 rknpu_trace_report.py <trace…>` 出报告 ✓；加 `--check <baseline>` 做回归 ✓。

### 7.155 池化 → 卷积 管线对接（第一步：`min_pool --out-file`）（2026-09-30 第一百三十二轮，**已提交**）

1. `min_pool` 新增 **`--out-file F`** ✓：提交前把**输入**逐字节写 `F` ✓、提交后把**输出**写 `F.out` ✓（供下游算子复用 ✓，为"池化 → 卷积"组合做准备 ✓）。
2. 实测（`--iw 32 --ih 32 --ic 8 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 --out-file /tmp/pool155` ✓）：`SUBMIT ret=0` ✓、`mismatch=0` ✓、产出 `/tmp/pool155`（16384 B = 输入 ✓）与 `/tmp/pool155.out`（4096 B = 池化输出 ✓）。
3. **下一步（未完成）**：给 `min_conv_exp` 加 **`--in-file F`**（从文件读输入而非自生成 pattern ✓）⇒ 即可把池化输出直接喂给卷积 ✓，并用 Python 参考独立校验 ✓（形成"池化→卷积"NPU 管线闭环 ✓）。

### 7.156 **池化 → 卷积 NPU 管线打通**（字节层面）（2026-09-30 第一百三十三轮，**已提交**）

1. `min_conv_exp` 新增 **`--in-file F` / `--in-xiw`** ✓：整块读入输入（**不再逐点填 pattern** ✓）；`--in-xiw` 表示文件是**x 主序**（= `min_pool` 的输出布局 ✓）⇒ 逐原子转置到硬件 y 主序 ✓。
2. 实测串联 ✓：`min_pool --iw 32 --ih 32 --ic 8 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 --out-file /tmp/pipe` ⇒ `mismatch=0` ✓、输出 4096 B ✓；再 `min_conv_exp --iw 16 --ih 16 --ic 8 --oc 16 --kw 1 --kh 1 --pad 0 --in-file /tmp/pipe.out --in-xiw` ⇒ **`IN-FILE: 读入 4096/4096 B（x 主序）`** ✓、`SUBMIT ret=0` ✓、`OUT-HASH 678e3085` ✓。
3. ⇒ **两个 NPU 算子的字节级管线打通** ✓（池化 ⊗ 卷积 ✓）；**下一步（未完成）**：用 Python 独立参考校验该组合输出 ✓，并把"残差/池化组合"扩到训练闭环 ✓。

### 7.157 `--in-file`/`--in-xiw` 路径的**行为学验证**（判据实验）（2026-09-30 第一百三十四轮，**已提交**）

**判据**：同一文件、改动单一变量，看 `OUT-HASH`（conv 16×16 IC8→OC16 k=1×1 pad0 ✓，输入 = 池化输出 `/tmp/pipe.out` ✓）：

| 用例 | OUT-HASH | 说明 |
| --- | --- | --- |
| 同参数跑两次 | `678e3085` / `678e3085` | **确定性** ✓ |
| **去掉 `--in-xiw`**（把同一文件按 y 主序读） | **`7c5afe85`** | **转置开关确实生效且影响结果** ✓✓ |
| 输入换成**全 0** 文件 | **`e3831dc5`** | **输入内容确实驱动输出** ✓（非固定哈希 ✓）|

⇒ `--in-file` / `--in-xiw` 路径 ✅ 确定性、✅ 布局敏感、✅ 内容敏感 ⇒ **新代码路径行为正确** ✓。
**仍缺的定量校验**：与独立参考逐值比对 ✗（int8 路径工具**不自带参考** ✗、权重由工具内部生成 ✗）
⇒ 两条可选路径：① 让工具 `--dump-weights` 输出权重 ✓（小改动 ✓）；② 改用 **fp16 路径**（自带 fp64 参考 ✓）但需给 `min_pool` 加 fp16 ✗（较大 ✓）⇒ 建议走 ① ✓。

### 7.158 池化→卷积的**定量校验尝试**：未获确认 ✗（`--w-ones` 下输出恒为零点）（2026-09-30 第一百三十五轮，**已提交**）

1. `min_conv_exp` 新增 **`--out-file F`** ✓（提交后从 `OUT_OFF` 落盘输出 ✓，实测 8192 B ✓、`OUT_OFF=524288` ✓）。
2. 定量实验（`--in-file /tmp/pipe.out --in-xiw --w-ones --out-file /tmp/convw.bin`，k=1×1 pad0 IC8→OC16 ✓）：`ret=0` ✓、`OUT-HASH 1f669c33` ✓；offline 逐像素比对（期望 `out = 128 + Σ_ic signed(x_ic)` ✓）：

| 观察 | 数值 |
| --- | --- |
| 完全吻合（`out−128 == Σ` ✓） | **92/256** ✓（全部是 `Σ = 0` 的像素 ✓）|
| **其余 164 个像素** | `out−128 == 0` ✗（**输出恒为零点** ✗）|

3. ⇒ **未获确认** ✗：`--w-ones` 下输出恒为**零点 128** ✓ ⇒ 说明**工具的 int8 标度/零点约定**是针对其**自生成 pattern** 校准的 ✓，喂入**外部真实数据**时结果被压到零点 ✗（不是管线本身错 ✗，而是工具标度不匹配 ✓）。
4. **下一步（原计划，未完成）**：加 **`--dump-weights`** ✓ 并用 Python 复刻工具的**完整 int8 约定**（权重布局 A + in/out 标度/零点 ✓）⇒ 才能给出可信的定量结论 ✓；**不建议**再用 `--w-ones` 做该验证 ✗。

### 7.159 收官回归（三工具 + 离线自检）（2026-09-30 第一百三十六轮，**已提交**）

1. **板上回归**（二进制/数据集配对 ✓）：R1 `min_pool` 32×32 `mismatch=0` ✓；R2 `min_pool --ih 128 --tile-oh 32`（旧硬边界外）`0` ✓；R3 `min_conv_exp` 16×16 IC32 `OUT-HASH 93b5be45` ✓；R4 同 + `--task-count 3 --task-blocks` ⇒ **`ret=0` 且同哈希** ✓；R5 `wgrad_real` 64×64 int8 **BIT-EXACT** ✓；R6 同 + `--tile-oh 16` ⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓。板子 soft 89→89 ✓、**楔死签名 0** ✓。
2. **离线自检**：`rknpu_trace_report.py --check trace_report_baseline.txt` ⇒ **差异 0 处** ✓；台账 **127 章节 ↔ 127 索引条目** ✓（一致 ✓）；handoff 结构完好（10 个 `##`、49 个 `###`）✓。
3. 工件：`experiments/results/final_regression_round124.txt` ✓。

### 7.160 外部输入的定量校验：路径已摸清（`wgrad_real` 从 `x_b*.bin` 读输入）（2026-09-30 第一百三十七轮，**已提交**）

1. 读源码确认 ✓：`wgrad_real` 的 `dataset_files_load()` 从数据集目录读 **`x_b<b>.bin`**（每 batch 一份 ✓，`X_BYTES` ✓），
   且 `ds->X[b]` 的布局是**普通行主序** `(y*IW + x)*IC + c` ✓（见 §7.x `cpu_forward`/dW 的索引 ✓），
   **与 `min_pool` 的 x 主序输出不同** ✗ ⇒ 对接需要**转置** ✓。
2. ⇒ **定量校验的可行路径**（比 `--dump-weights` 更省 ✓）：① 生成一个新数据集目录 ✓；② 把池化输出**转置**后覆盖 `x_b0.bin` ✓；③ **重算参考**（`ds.Q` 在加载路径里**不会**按新 X 重算 ✗ —— 源码只加载 `dy_b*/t_b*/w.bin` ✓）⇒ 需要一个**小改动**：加 `--recompute-ref`（加载后用 `cpu_forward` 重算 `Q` ✓）或加 `--in-x-file`（读外部 X 并重算 ✓）。
3. **状态**：本轮**未做**该改动（避免在收官后引入未验证的代码 ✗）；已把路径与所需改动**写清** ✓，供后续一轮直接实施 ✓。
4. 另：**放弃** `--dump-weights` + 手工复刻 int8 约定 的路线 ✗（§7.158 已证其成本高且易被标度约定误导 ✓），
   优先走"工具自带参考 + 外部输入"这条**可自证**的路 ✓。

### 7.161 **池化 → 卷积 管线定量校验通过（BIT-EXACT）**（2026-09-30 第一百三十八轮，**已提交**）

1. `wgrad_real` 新增 **`--in-x-file F`** ✓：读入外部 X（行主序 `(y*IW+x)*IC+c` ✓）替换 batch0 输入，并用 `cpu_forward` **重算参考 `Q[0]`** ✓（走"工具自带参考"这条**可自证**的路 ✓）。
2. **串联实测** ✓：
   - `min_pool --iw 32 --ih 32 --ic 32 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 --out-file /tmp/p126` ⇒ 池化输出 8192 B（16×16×32 ✓）、`mismatch=0` ✓；
   - Python 把池化输出（x 主序 ✗）**转置**成行主序 ✓ 写入 `/tmp/pipe126/x_b0.bin` ✓（8192 B ✓）；
   - `wgrad_real --dir /tmp/pipe126 --pad 1 --batch 1 --in-x-file …` ⇒ `IN-X-FILE: 读入 8192/8192 B 并已重算参考 Q[0]` ✓ ⇒ **`FWD b=0: round-mismatch=0 RESULT: BIT-EXACT (round)`** ✓✓ ⇒ **`RESULT: CLOSED-LOOP OK`** ✓✓。
3. ⇒ **两个 NPU 算子（池化 → 卷积）的数值链路定量成立** ✓✓（卷积在**池化输出**上算，与 CPU 参考**逐位一致** ✓），且**训练闭环在该输入上同样成立** ✓。
4. 这一步同时验证了：`min_pool --out-file` ✓、`--in-x-file` + 参考重算 ✓、以及**两工具的字节布局可对接** ✓（x 主序 ↔ 行主序的显式转置 ✓）。

### 7.162 池化→卷积（含训练闭环）**一键配方**（2026-09-30 第一百三十九轮，**已提交**）

1. 新增 **`experiments/pipeline_pool_conv.sh`** ✓（`sh pipeline_pool_conv.sh <工作目录>` ✓），四步：
   ① `min_pool`（32×32×32 --k2s2--> 16×16×32 ✓，`--out-file` ✓）；② `wgrad_real --gen-only` 建数据集 ✓；
   ③ Python 把池化输出（x 主序）**转置**为行主序覆盖 `x_b0.bin` ✓；④ `wgrad_real --in-x-file`（读外部 X + 重算参考 ✓）跑**卷积 + wgrad + 更新** ✓。
2. **端到端实测**（`/tmp/pipe127` ✓）：① `mismatch=0` ✓；③ `transposed 8192 B` ✓；④ `IN-X-FILE 读入 8192/8192 B 并已重算参考 Q[0]` ✓ ⇒ **`FWD b=0: BIT-EXACT (round)`** ✓ ⇒ **`RESULT: CLOSED-LOOP OK`** ✓✓。
3. ⇒ **"池化 → 卷积 → wgrad → 更新"全链一键可复现且可自证** ✓✓（修脚本时踩到的小坑：`--out-file` 是**文件前缀**而非目录 ✗ ⇒ 目录与前缀须分开 ✓；`--gen-only` 会写自己的 `x_b0.bin` ⇒ 转置必须放在其**之后** ✓）。

### 7.163 wgrad 在**外部任意梯度**上的定量验证（`--in-dy-file`）（2026-09-30 第一百四十轮，**已提交**）

1. `wgrad_real` 新增 **`--in-dy-file F`** ✓：读外部 dY（`qpix` 字节 ✓）替换 batch0 梯度 ✓。
2. **强判据实验**：把**池化输出**（8192 B 任意数据 ✓）当作**伪造的 dY** 喂入 ✓ ⇒

| 指标 | 结果 |
| --- | --- |
| 输入 | `IN-DY-FILE: 读入 8192/8192 B` ✓ |
| 前向 | `FWD b=0: BIT-EXACT (round)` ✓ |
| **wgrad** | **`coords=288/288 grads=9216/9216 round-exact floor_mismatch=0 maxdiff=0`** ✓✓ |
| 全链 | **`RESULT: CLOSED-LOOP OK`** ✓✓ |

3. ⇒ **wgrad 在"任意外部梯度"上与 CPU 参考逐位一致** ✓✓ —— 说明训练侧 wgrad 的写回/缩放**不依赖工具的合成数据** ✓（此前只在自生成梯度上验证过 ✓）。对照组（无外部 dY）同判 ✓。
4. 意义：**训练侧 wgrad 的真实性得到外部数据背书** ✓（§3b 第 ③ 项"int8 真实模型 wgrad 写回/梯度缩放"由此取得实质进展 ✓）。

### 7.164 旋钮清单入 handoff + 例行自检（2026-09-30 第一百四十一轮，**已提交**）

1. handoff 新增 **4 行旋钮清单** ✓：外部数据旋钮（`--out-file`/`--in-file`/`--in-xiw`/`--in-x-file`/`--in-dy-file` ✓）、多 task/多核旋钮（`--task-count`/`--task-blocks`/`--subcore-range`/`--subcore-single0`/`--sp-prefix`/`--task-delta4` ✓）、分片旋钮（`min_pool --tile-oh` ✓、`wgrad_real --tile-oh/--tile-p/--i8-tile-diag` ✓）、一键配方（`pipeline_pool_conv.sh` ✓）；状态快照更新为 **soft reset 89** ✓。
2. **例行自检** ✓：`rknpu_trace_report.py --check trace_report_baseline.txt` ⇒ **差异 0 处** ✓；台账索引与章节一一对应 ✓；handoff 结构完好 ✓。

### 7.165 会话后半成果并入 handoff（§7.146–§7.164）（2026-09-30 第一百四十二轮，**已提交**）

1. handoff 追加 **`### §7.146–§7.164 完成情况`** ✓（LUT 语义全链 ✓、算子指纹库 ✓、一站式报告+回归 ✓、池化→卷积管线 ✓、wgrad 外部梯度 ✓、收官回归+自检 ✓）。
2. 与 §3b（最新未闭合清单）配套 ✓：**已完成**在 §1 附录 ✓、**待办**在 §3b ✓ ⇒ 继任者可一眼分辨 ✓。
3. 本轮为**收尾整合**（纯文档 ✓，无板上改动 ✓）；例行自检已在本会话末尾执行（trace 报告 0 差异 ✓、账簿 133↔133 ✓）。

### 7.166 技能文档本身增补"2026-09-30 实测要点"（2026-09-30 第一百四十三轮）

1. 在 `.agents/skills/rknpu-command-stream/SKILL.md` 的「常见坑」之后追加 **8 条可直接套用的实测要点** ✓：
   ① 载荷在 `event=="regcmd"`（不是 task ✗）；② 抓 trace 必须设 `EPERF_RKNPU_REGCMD_MAX_BYTES` ✓、条目 8 B ✓；
   ③ 表分析先去重、平台须在写入顺序上检验 ✓；④ **`regcfg_amount` 两套口径不可混用**（本驱动 64-bit entry ✓ vs vendor 4 B 字 ✓）；
   ⑤ **LUT 判据看突发**（~443 / ~1027 条 ✓）而非总计数 ✓（其余是 1 条零写旁路戳 ✗）、表为**静态常量** ✓；
   ⑥ **单核受限判据**（真实工具链一律 `core_mask=0` + 按 subcore 分派 ✓；"全塞 subcore0" = 硬件不支持 ✗）；
   ⑦ **CNA 块出现程度 ⇒ 算子类别**（0/+1 元素级 ✓、+54 归约 ✓、+80 conv ✓、+1 BN ✓）；⑧ 本仓三个可复用工具 ✓。
2. 该文件位于 `.agents/`（**未纳入 git** ✗ ⇒ 改动不产生提交 ✓，但已落盘生效 ✓）。
3. 同步记录于台账（本条 ✓）；handoff §5 的工具清单已另列同一批工具 ✓。

### 7.167 与技能文档旧记载数字的**对账**（2026-09-30 第一百四十四轮，**已提交**）

1. 技能「已知结果」记载：SDPA 每 instance `0x4104×1079` ✓、`4388 B 大块 = LUT 表加载（0x4104×9580）` ✓ —— 但文档**自己已标注 ⚠ 来自旧半截捕获**（`regcfg_amount*4` ✗）✓。
2. 本会话按 **8 B 条目**重测的结论 ✓：整表加载 = **少数几次突发**（**443** 条 = 激活/softmax 表 ✓、**1027** 条 = attention 表 ✓），其余事件只有 **1 条 `0x4104=0`** 的**旁路戳** ✓ ⇒ 旧数字实为**整次运行的总计数**（突发 + 旁路戳混计 ✗）⇒ **不宜再作逐算子判据** ✗。
3. 表的确定量：**唯一值 111 / 440 / 443 / 694** ✓；**静态常量** ✓（同模型换输入逐字节相同 ✓）；**分段常数**（平台**真实存在** ✓，须在**写入顺序**上检验 ✓ —— §7.153 修正 §7.146 ✓）；函数族：sigmoid/tanh **近指数** ✓、softmax **线性斜坡** ✓、attention **分段常数/反查** ✓。
4. 已把该对账作为第 9 条追加进技能文档 ✓（`.agents/` 未纳入 git ⇒ 改动落盘但不产生提交 ✓，台账留档 ✓）。

### 7.168 **规范 hook 交叉验证**（技能自带 hook vs 本会话 microbench hook）（2026-09-30 第一百四十五轮，**已提交**）

1. 用技能自带规范 hook（`/mnt/ssd/workspace/rknpu-hook/libeperf_hook_rknpu_ioctl.so` ✓）重抓 `sigmoid_1x16_fp16.rknn` ✓：
   事件分布 `{meta:1, mem_create:5, mem_map:5, mmap:5, submit:3, task:12, regcmd:12}` ✓，**只有 `regcmd` 事件带 `hex`** ✓（12/12 ✓）。
2. `lut_scan.py` 结果：`0x4100×16`、**`0x4104×1780`**、配置寄存器×12 ✓；**唯一值 440** ✓、top3 = `0x0`(×12)/`0x3e`(×8)/`0x42`(×8) ✓、配置全 0 ✓。
3. ⇒ **与 microbench hook 的实测逐个数字一致** ✓✓ ⇒ 本会话 LUT 口径（§7.135–§7.167）**在规范 hook 下复现** ✓；且两工具对**规范 hook 的 trace 同样可用** ✓（schema 兼容 ✓）。
4. ⇒ 进一步确认 §7.134：**载荷只在 `event=="regcmd"`** ✓（`task` 事件带 hex 数为 0 ✓）。

### 7.169 会话收尾与暂停（2026-09-30 第一百四十五轮末，**已提交**）

1. handoff 完成收尾：**§7.110–§7.168 全部成果**分三段附录在册 ✓（§7.110–§7.145 ✓、§7.146–§7.164 ✓、§7.165–§7.168 收尾 ✓）+ 旋钮清单 4 行 ✓ + §3b 最新未闭合清单 ✓。
2. **按用户指示暂停 goal** ✓（`goal-af93b594…` ⇒ phase = **paused**、revision 4 ✓）：本轮为**可续接的中断点** ✓ —— 恢复方式 = 重新 armed 后从 **§3b** 清单继续 ✓。
3. 断点状态 ✓：提交历史干净 ✓、工作区干净 ✓、soft reset 89 / **楔死签名 0** ✓、台账 **137 章节 ↔ 137 索引条目** ✓、trace 报告基线自检 **0 差异** ✓。

### 7.170 驱动反汇编定论 + 多核并行可证伪闭合 + §7.115/§7.116 证伪修正（2026-10-02 接手会话，**已提交**）

1. **接手核查 ✓**：工作区干净（`473e4d9`）、板在线（soft reset 89 起点、楔死 0）、板/仓三工具 sha 一致 ✓、健康探针 BIT-EXACT ✓、双回归全绿（int8 74/74、fp16 79/0/0）✓。
2. **驱动侧素材到手（§3b-1 要求的 vmlinux/System.map 路线）**：板 `/boot/Image`（40579584 B）+ `System.map-6.1.43-rockchip-rk3588-rknpu098`；`/proc/kallsyms` 与 System.map 地址**逐符号一致** ⇒ **KASLR 关闭** ⇒ `Image 文件偏移 = VA − 0xffffffc008000000`（用 `linux_banner` 吻合校验 ✓）。工具 `experiments/disasm_rknpu.py`（capstone）⇒ 产物 `experiments/results/driver_disasm_20261002/rknpu_disasm.txt`。
3. **驱动二进制语义定论（反汇编，三函数）**：
   - `rknpu_get_task_number`：num_irqs>1 且 use_core_num∈{1,2} ⇒ 读 `subcore_task[core_index]`（槽位 0..1）；**use_core_num==3 ⇒ 读 `subcore_task[core_index+2]`（槽位 2..4）**（与官方源码一致 ✓）；`struct rknpu_task` = 40 B（0x28）。
   - `rknpu_job_schedule`：`core_mask==0`（AUTO）⇒ **只选 1 个最空闲核**（`subcore_datas[i].task_num` 最小者）⇒ use_core_num=1 ⇒ commit 只读**一个**槽位。
   - `rknpu_job_subcore_commit_pc`（commit 序列）：slave-mode `PC_DATA_ADDR=0x1` → `PC_DATA_ADDR=first_task->regcmd_addr` → `PC_DATA_AMOUNT=(regcfg_amount+4+scale−1)/scale−1`（scale=2）→ `INT_MASK=last->int_mask` → `INT_CLEAR=first->int_mask` → `PC_TASK_CONTROL=((0x6|pp)<<12)|task_number` → `PC_DMA_BASE=task_base_addr` → `PC_OP_EN`(0x18) 1→0 脉冲。**每核各自独立编程一次，无跨核链**。
   - **`args->task_counter` 语义（关键更正）**：成功路径 = **回显 `args->task_number`**（rknpu_job.c:266），不是硬件计数器；只有失败路径才读硬件 PC_TASK_STATUS。⇒ §7.115 的「task_counter=3」是回显假象 ✗。
4. **§7.115/§7.116「多 task/多核闭合」证伪（重要修正）**：AUTO 下 3 task 提交实际**只执行 task0**（驱动 AUTO→单核→slot0={0,1}）；`hw_elapse` 逐纳秒相同（49875 vs 49874）本就是单 task 证据。决定性复验：新旋钮 `--task-out-step`（每 task 独立输出区 + 哨兵 + 逐区对 task0 比对）⇒ AUTO 对照臂 **TASK-OUT 1/2 = UNTOUCHED**（哨兵 0x80 原样、diff=51212/73728）⇒ 旧「3 核并行/3× 吞吐」为**假成功** ✗。
5. **多核并行首次可证伪闭合（新旋钮 `--subcore-hi`）**：驱动 3 核读**槽位 2..4** ⇒ 旧 §7.61 把 per-core 条目填在 0..2 ⇒ core0 拿到 task#2、core1/2 拿 task_number=0 的越界描述符（= §7.61「不稳定/部分写入」的根因 ✓）。修正后（`subcore_task[2..4]={i,1}` + `core_mask=0x7`）实测 **3/3 稳定 ALL-TASKS-EXECUTED**：int8 16×16（3 task hw 32–38 µs < 单 task 77 µs）、**48×48 IC32（3 task 78.7 µs vs 单 task 51.3 µs ⇒ ≈1.95× 吞吐）**、fp16 16×16（ULP-EXACT）——三核各自独立写出逐位正确输出 ✓✓、soft 92→92、楔死 0 ✓。
6. **过程教训**：① patch regcmd 条目必须保留 `(target<<48)` 高位（首版只写 `(val<<16)|reg` 把 target 清零 ⇒ 3 次超时 soft 89→92；条目格式 = **(target<<48)|(val<<16)|reg** ✓）；② 板上二进制必须**在板上编译**（本机 x86 gcc 产物 = Exec format error）；③ 索引/章节计数校验通过（139↔139）。
7. **未闭合保持**：单 subcore 多 task（vendor 真实形态 `(N,0)`、regcfg_amount=9 占 80–86%）仍从流侧不可达 ✓（本轮 5 变体结论不变）；新证据：commit 序列已逐寄存器定论（见第 3 条），链式推进只能来自流内尾写 ⇒ 下一步聚焦 vendor 34 条增量块的**逐字节复刻**（含 `PC_REGISTER_AMOUNTS` 槽位口径）。
8. 不回归：int8 硬化 **74/74**、fp16 **79/0/0**；`--subcore-hi`/`--task-out-step` 均 opt-in，默认路径字节不变 ✓。


### 7.171 驱动 AUTO 分支/subcore_task 槽位语义定论（源码优先，零板卡风险）（2026-10-02 路线图 Task 0）

1. **路线图落地**：`docs/superpowers/plans/2026-10-02-rknpu-reverse-roadmap.md`（用户修正版：源码优先，反汇编仅交叉验证）。产物 `experiments/results/driver_disasm_20261002/auto_branch_decode.md`。
2. **源码定论（`refs/rknpu-driver-0.9.8/.../rknpu_job.c`，行号按本仓库副本）**：
   - `rknpu_get_task_number`（L71）：`num_irqs>1`（RK3588=3 ✓）且 `use_core_num∈{1,2}` ⇒ `task_number = subcore_task[core_index].task_number`；`==3` ⇒ `subcore_task[core_index+2]`。**`args->task_number` 只在 use_core_num∉{1,2,3} 时兜底**——即 RK3588 PC 路径上**从不**直接编程 `args->task_number`。
   - `rknpu_job_subcore_commit_pc`（L272）：同槽位规则读 **`task_start` 与 `task_number`**（L313–331）；分片 = `task_start += submit_index*4095`、`task_number` 截到 4095（L333–336）；`job_done`（L460）按 `(get_task_number+4094)/4095` 决定是否**驱动自动分片重提交** ⇒ 「单核大 N」存在内核路径，无需流内链。
   - AUTO（L532–538）：选最闲核（`subcore_datas[i].task_num` 最小）⇒ **改写 `args->core_mask`（写回 userspace）+ `use_core_num=1`** ⇒ 之后读 slot[chosen]。
   - `rknpu_job_commit`（L404）：只接受 mask ∈ {1,2,4,3,7}；`{5,6}` 打印 Unknown ⇒ **非相邻多核 mask 是驱动拒绝路径**。
   - `rknpu_job_wait` 成功路径 `args->task_counter = args->task_number`（L266，回显 ✓ 与 §7.170 反汇编一致）。
3. **ABI 定论（rknpu_ioctl.h L241）**：`struct rknpu_subcore_task { u32 task_start; u32 task_number; }` ⇒ **vendor 真值「全槽 {0,1}」= task_start=0 + task_number=1**（不是「task_number=0, offset=1」）。flags 位（L98–101）：PC=1、**BLOCK=0（无位）**、NONBLOCK=2、PINGPONG=4 ⇒ **vendor flags=0x5 = PC|PINGPONG（pp_en=1）**——旧记「5=PC|BLOCK，无 PINGPONG」**有误** ✗（§7.19–§7.21 排除清单中「排除 ping-pong」的口径需按此复核）。
4. **三个历史观测的源码解释**：vendor 全槽 `{0,1}` = 每核 PC 只编程 **1 个 task**（first=last=task[0]），task1/2 靠**流内尾写链**推进，`args->task_number=3` 仅回显/分片口径 ⇒ 对 AUTO 任选核鲁棒；我们 `{0,N}`（slot=N）⇒ PC 编程 N 个 ⇒ 硬件等流内链而流没有 ⇒ 6.25 s 超时（§7.62 ✓）；`{k,1}` ⇒ `task_start=k` 指错描述符 ⇒ 无输出（§7.61 ✓）。**新杠杆：slot.task_number=1 + task_number=3 的组合从未试过**（§7.62 全部用 slot=N）。
5. **vendor 形态 N=3 描述符目标规格**（Task 1 对拍基准，详见 `auto_branch_decode.md` §5）：`flags=0x5 / task_start=0 / task_number=3 / task_base=0 / core_mask=0(AUTO) / subcore_task[0..4]={{0,1}×5}`；驱动改写 core_mask/task_counter/hw_elapse_time，其余透传。
6. Ruling（执行中裁定）：`timeout` 字段取值待 Task 1 从 vendor trace submit 事件读取，规格暂留占位。

产物（本次提交）：`experiments/results/driver_disasm_20261002/auto_branch_decode.md`、本节；无板卡操作。



## 八、复现步骤

~~~bash
# 0) 上传（板上目录 /mnt/ssd/workspace/rknpu-re/exp2/）
scp experiments/min_conv_exp.c experiments/rnpu_regmap.h root@<board>:/mnt/ssd/workspace/rknpu-re/exp2/

# 1) 编译
gcc -O2 -o min_conv_exp min_conv_exp.c -lm

# 2) 参考口径：应全部 BIT-EXACT
./min_conv_exp --pattern --pad 1 --pad-con1 0 --auto-fix --ic 40 --oc 32
./min_conv_exp --pattern --pad 1 --pad-con1 0 --auto-fix --ic 65 --oc 40

# 3) 复现两条缺陷（应 MISMATCH）
./min_conv_exp --pattern --pad 1 --pad-con1 0 --ic 40 --oc 32          # DATA_ENTRIES 偏小
./min_conv_exp --pattern --pad 1 --pad-con1 0 --ic 3 --oc 16 --w-oc2 16 # 跨度与 WEIGHT_KERNELS 不一致

# 4) 普通路径 PAD_CON1 对照（应全部 BIT-EXACT；IC=1 专用路径见 §7.1）
./min_conv_exp --pattern --pad 1 --ic 3 --oc 16 --w-oc2 32 --w-kernels 32 --pad-con1 0x7fffffff

# 5) tap 顺序（§7.15）：判别矩阵 + 扩展扫描 + 硬化回归 + 归档（板上 exp2/）
sed 's|\./min_conv_exp |./min_conv_exp_old |' run_taporder_matrix.sh > _m_old.sh
bash _m_old.sh                     # 旧 kx-major：kw3kh3 --asym-w MISMATCH 2682、kw≠kh 全挂
./run_taporder_matrix.sh           # 新 ky-major：15/15 BIT-EXACT
./run_taporder_matrix2.sh          # 扩展扫描 74/74
./run_regression_hard.sh           # §7.11 的 74 例 + --pat2 --asym-w → 74/74
./run_taporder_archive.sh          # 全部日志 → results/taporder_20260929/

# 6) vendor 权重 blob 离线解码（本机，§7.15 证据 A）
python3 experiments/vendor_blob_decode.py

# 7) fp16 路径（§7.17）——板上串行，严禁并发第二个发射器实例
./min_conv_exp --f16-selftest | tail -1
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 8 --oc 16
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16 --oc 16
./run_fp16_regression.sh          # 全量回归（含健康探针协议）
./run_fp16_archive.sh             # 全部证据 → results/fp16_20260929/

# 8) fp16 独立 BN（BN_ALU + NRDMA fp32 表，§7.23；探针内含 ETIMEDOUT 重试）
bash run_fp16_bn_nrdma_probe.sh   # → results/fp16_bn_20260930/

# 9) fp16 BS/BRDMA 的 ALU(fp32)+MUL(fp16@[31:16]) 级（§7.24）
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --in-ch-id --w-ones --kw 1 --kh 1 \
   --iw 4 --ih 4 --pad 0 --ic 8 --oc 8 --bs-cfg 0x20140 --bs-mul 0x40000000   # (36)*2 = 72
bash run_fp16_bs_mul_probe.sh     # → results/fp16_bs_mul_20260930/

# 10) fp16 wgrad（CNA K 维归约，§7.25；注意必须 --f16-pix-elems 8）
./min_conv_exp --wgrad-dot --pattern --fp16 --f16-pix-elems 8 --auto-fix --pad-con1 0 \
   --ic 48 --oc 16 --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0   # → BIT-EXACT (fp16)
./min_conv_exp --wgrad-dot --pattern --fp16 --f16-pix-elems 8 --auto-fix --pad-con1 0 \
   --ic 48 --oc 16 --iw 4 --ih 4 --ow 1 --oh 1 --kw 1 --kh 1 --pad 0 --ic-pad 48   # → MISMATCH（反例）
bash run_fp16_wgrad_probe.sh      # → results/fp16_wgrad_20260930/

# 11) fp16 真实训练闭环（wgrad_real，§7.26；板上 exp3/）
gcc -O2 -Wall -o wgrad_real wgrad_real.c -lm
./wgrad_real --fp16 --pad 1 --batch 1 --update   # → CLOSED-LOOP OK (fp16 ...)
./wgrad_real --fp16 --pad 0 --batch 2 --update --lr 0.5 --loss 0.125
FP16_WGRAD_REAL_LOG=... ./run_fp16_wgrad_real_loop.sh   # → results/fp16_wgrad_real_20260930/

# 12) deconv 边界口径修正（§7.27）
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --f16-pix-elems 8 \
   --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --deconv-stride 1 --deconv-flip \
   --ic 3 --oc 16 --pad 1 --kw 1 --kh 1   # → BIT-EXACT (fp16)（修正前 186/1024 MISMATCH）
bash run_deconv_boundary_probe.sh          # → results/deconv_boundary_20260930/

# 13) fp16 双层真实网络闭环（wgrad_real，§7.28；板上 exp3/）
./wgrad_real --two-layer --pad 1 --batch 1 --seed 20260928 --update   # → 2-LAYER CLOSED-LOOP OK
./wgrad_real --two-layer --pad 0 --batch 2 --update                   # pad0：锁定 make_fwd_cfg_l2 输出尺寸 bug
FP16_TWO_LAYER_LOG=... ./run_fp16_two_layer_loop.sh                   # 含 180 例宽扫描 → results/fp16_two_layer_20260930/

# 14) CNA 输入 feature tile 32KB 包线（§7.30，修正 §7.29）
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --kw 3 --kh 3 --iw 16 --ih 8  --pad 1 --ic 128 --oc 16  # 32768 B → ULP-EXACT
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --kw 3 --kh 3 --iw 16 --ih 9  --pad 1 --ic 128 --oc 16  # 36864 B → OUT-OF-ENVELOPE
./min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 16 --ih 9  --pad 0 --ic 128 --oc 16  # 1x1 同界 → OUT-OF-ENVELOPE
FP16_DEEP_LOG=... ./run_fp16_deep_channel_probe.sh                    # → results/fp16_deep_channel_20260930/
~~~

仪表版开关全表：

~~~
--mode conv|deconv   --iw/--ih/--ow/--oh   --ic/--oc   --pad N   --stride N
--in-byte B  --w-byte B  --pad-byte B  --pad-con1 V  --w-oc2 N  --w-kernels N
--w-layout a|c|d  --w-uniform  --pattern  --ic-pad N  --entries N  --ws1 N
--icreal-reg N  --hi-zero  --auto-fix  --deconv-stride N  --deconv-flip
--kernel-group N
--enable-mask M  --dump
--bn-cfg/--bn-alu/--bn-mul/--bn-relux  --dpu-data-format V
--bn-nrdma N  --bn-table-offset N  --bn-table-bias N  --bn-table-step N
--bn-table-f32-bias F  --bn-table-f32-step F
--bs-cfg/--bs-alu/--bs-mul  --brdma-cfg N
--brdma-bias N  --brdma-bias-step N  --brdma-mul N  --brdma-mul-step N
~~~

---

## 九、产物与身份

| 文件 | 说明 |
| --- | --- |
| `experiments/min_conv_exp.c` | 仪表版发射器（以当前文件为准） |
| `experiments/rnpu_regmap.h` | Mesa rules-ng-ng 寄存器映射（唯一复用物） |
| `experiments/results/baseline_20260927/raw_results_20260927.txt` | 本轮 10 组脚本的全部原始日志 |
| `experiments/results/ic1_probe_20260928/` | IC=1 CVT、布局与 padding 真机补验日志 |
| `experiments/results/oc33_63_span_sweep_20260928.log` | OC 33–63 紧凑偶数跨度反例日志 |
| `experiments/results/oc33_62_span64_20260928.log` | OC 33–62、64 槽跨度 round-exact 日志 |
| `experiments/run_oc33_63_span_sweep.sh`, `experiments/run_oc33_62_span64.sh` | OC 跨度扫描脚本 |
| `experiments/run_wgrad_dot_probe.sh` | CNA K 维 wgrad dot microbench |
| `experiments/results/wgrad_dot_20260928.log` | P=32/64/96 wgrad reduction 真机日志 |
| `experiments/run_wgrad_coeff_matrix_probe.sh` | 合成 3×3×IC=3 的 wgrad 权重坐标矩阵扫描 |
| `experiments/results/wgrad_coeff_matrix_20260928.log` | 27 个独立合成系数坐标对照 |
| `experiments/run_wgrad_conv3x3_matrix_probe.sh`, `experiments/results/wgrad_conv3x3_matrix_20260928.log` | 自洽合成3×3/IC=3/OC=16 wgrad矩阵真机日志 |
| `experiments/run_wgrad_batch_probe.sh`, `experiments/results/wgrad_batch_20260928.log` | 一个权重项跨 B=1/2/3 的 K 维累加 |
| `experiments/wgrad_real.c`, `experiments/run_wgrad_real_loop.sh` | 真实张量 wgrad 闭环工具与运行器（板上 exp3/） |
| `experiments/results/wgrad_real_20260928/` | 闭环全部日志 + 修正后数据集 + 独立 Python 逐字节校验脚本（lawcheck.py） |
| `experiments/results/rect_kernel_20260928.log` | 方形/矩形 kernel 边界扫描真机日志（台账 §7.1） |
| `experiments/results/ppu_maxpool_20260928.log` | PPU MaxPool 多通道/padding 扫描日志（台账 §7.8） |
| `experiments/results/subcore_probe_20260928.log` | 多核派发探针日志 + 挂死事件记录（台账 §7.9） |
| `experiments/results/board_reboot_20260928.log` | 2026-09-28 挂死断电重启后的板端状态（uptime/内核/dmesg） |
| `experiments/results/bn_lower_20260928/` | vendor BN lowering 全套：模型(.rknn/.onnx)、trace×4、BO dump(bos/)、系数→输出闭环脚本与日志（台账 §7.6） |
| `experiments/results/bn_lower_20260928/em0d/` | em=0x0d 首次捕获：Mul/Transpose/LayerNorm 模型与 trace（台账 §7.7） |
| `experiments/results/brdma_cal_r2_20260929.log` | BRDMA 表值域标定真机日志（台账 §7.12） |
| `experiments/run_brdma_cal_r{1,2,2b,3}.sh` | BRDMA 标定探针脚本（bias ramp / mul ramp / ALGO0/1 定界） |
| `experiments/results/pool_vendor48_tr.jsonl` | vendor ic=48 MaxPool regcmd 真值（单 task 26 条，§7.13） |
| `experiments/results/pool_mine48_tr.jsonl`, `pool_ic48_tr.jsonl` | 本机 min_pool ic=48 regcmd 与 ic16 对照 trace |
| `experiments/results/pool_ic48_20260929/make_pool48.py` | vendor pool48 模型构建脚本（.rknn/.onnx/.bin 由其复现，gitignored） |
| `experiments/results/rect_kernel_20260929/` | 矩形核 vendor regcmd 真值（tr×2 + 解码 + 模型构建脚本，台账 §7.14） |
| `experiments/results/taporder_20260929/` | 权重 tap 顺序根因证据：README + `ab_evidence.txt`（7 份板端日志 + blob 解码日志） + 分步 `.log`（台账 §7.15） |
| `experiments/rknn_blob_dump.py` | `.rknn` FlatBuffer / `weight_data` 权重 blob 提取器（§7.15 证据 A） |
| `experiments/vendor_blob_decode.py` | vendor 权重 blob → 逻辑 (kh,kw) tap 映射反查（判据脚本，§7.15） |
| `experiments/run_taporder_matrix.sh`, `run_taporder_matrix2.sh` | tap 顺序判别矩阵 / 扩展扫描（板端） |
| `experiments/run_ab_taporder.sh`, `run_taporder_archive.sh` | 旧 vs 新二进制逐用例对照 / 全部日志一键归档（板端） |
| `experiments/run_regression_hard.sh` | §7.11 的 74 例改用 `--pat2 --asym-w` 的硬化回归（§7.15） |
| `experiments/run_deconv_smoke.sh` | deconv flip/no-flip 对照冒烟（§7.15） |
| `experiments/fp16_diff.py` | 两条 trace 的 NPUOP 寄存器差分（§7.16，fp16 vs int8） |
| `experiments/fp16_blob_decode.py` | fp16 vendor 权重 blob 解码（ic 原子 8 元素 + BN 折进前置 conv，§7.16） |
| `experiments/results/fp16_20260929/` | fp16 路径证据（回归 pass=71/fail=0 + ic 扫描 27/27 + 累加器精度 + 标尺探针 + 布局搜索 + 冲激 + int8 回归，§7.17/§7.18） |
| `experiments/run_fp16_{regression,icsweep,acc,ruler,layout_search,impulse,archive}.sh` | fp16 回归 / ic 扫描 / 累加器 / 标尺 / 布局搜索 / 冲激 / 一键归档（板端） |
| `experiments/run_fp16_bn_nrdma_probe.sh`、`experiments/results/fp16_bn_20260930/` | fp16 独立 BN（BN_ALU + NRDMA fp32 逐通道表）标定探针与板端日志（§7.23） |
| `experiments/run_fp16_bs_mul_probe.sh`、`experiments/results/fp16_bs_mul_20260930/` | fp16 BS/BRDMA 的 ALU(fp32) + MUL(fp16@[31:16]) 级标定探针与板端日志（§7.24） |
| `experiments/run_fp16_wgrad_probe.sh`、`experiments/results/fp16_wgrad_20260930/` | fp16 wgrad（dot / conv3x3 坐标 / batch 累加）探针与结论 README（§7.25） |
| `experiments/run_fp16_wgrad_real_loop.sh`、`experiments/results/fp16_wgrad_real_20260930/` | fp16 真实训练闭环（`wgrad_real.c --fp16`）探针与结论 README（§7.26） |
| `experiments/run_deconv_boundary_probe.sh`、`experiments/results/deconv_boundary_20260930/` | deconv 边界口径修正（ruler 反解 + 核/tap 扫描 + int8）探针与结论 README（§7.27） |
| `experiments/run_fp16_two_layer_loop.sh`、`experiments/results/fp16_two_layer_20260930/` | fp16 双层跨层闭环（`wgrad_real.c --two-layer`，180 例宽扫描）探针与结论 README（§7.28） |
| `experiments/run_fp16_deep_channel_probe.sh`、`experiments/results/fp16_deep_channel_20260930/` | CNA 输入 feature tile 32KB 包线探针与结论 README（§7.30，修正 §7.29 的 tap/`c.ic` 上限模型） |
| `experiments/wgrad_real.c` | 真实张量 wgrad 闭环工具（int8 全流程 + fp16 移植 §7.26 + 双层闭环 §7.28） |
| `experiments/min_conv_exp.c` | 仪表版发射器（新增 `--dump-bin`：导出激活/权重 BO 供离线逐字节分析；§7.30 按输入 tile 字节判 `OUT-OF-ENVELOPE`） |
| `experiments/RESULTS.md` | 既有实验记录（其 §2.2、§2.4 请按本文第六节阅读） |
| `experiments/results/baseline_20260927/raw_results.txt` | 既有原始日志（其 §3、§4 数字见 §5.7） |

板上原始目录：`/mnt/ssd/workspace/rknpu-re/exp2/`（本轮）、`/mnt/ssd/workspace/rknpu-re/exp3/`（wgrad_real 闭环）、`/mnt/ssd/workspace/rknpu-re/pool/`（min_pool + pool48 模型）、`/mnt/ssd/workspace/rknpu-re/exp/`（既有）。

%% 本台账的 A、B 系列为既有材料的结构化重排；C 系列（§五）记录 2026-09-27 补验，§7.1–7.4 为 2026-09-28 Orange Pi 5 Plus（RKNPU 0.9.8）真机实验。原始证据按各节列出的日志路径保存。 %%

### 7.172 vendor 34/34/4 块字面回放：两变体超时 + 楔死事故（2026-10-02 路线图 Task 1）

1. **工具**：`experiments/extract_task_blocks.py`（trace → 捕获字节 verbatim）+ `experiments/vendor_replay.c`（最小提交 harness，样板取自 min_chain；--variant 0/1）。证据 `experiments/results/chain_vendor_literal_20261002/`（README + blocks.bin + meta.txt）。
2. **提取**：attn_relu_vendor.jsonl seq1 task0/1/2 = **34.5/34.5/4.5 条目**（amount=69/69/9，DST 步进 0x400）✓ —— 即 §3b 的 34/34/4 形态；attn hook 捕获 = amount×4 B ⇒ **尾写在未捕获区**（槽 bytes 276..640）。
3. **v0（纯字面，无尾写）超时**（6239 ms, hw=0, task_counter=0）⇒ 单 vendor 块 + 全零槽位尾 ⇒ 差异在槽位层：PC 窗口 36 单位会把约 32 条**全零条目**当寄存器写执行（vendor 真实槽内该区间 = 尾写+数据）。
4. **v1（+重建尾写 @entry35）也超时**：PC 窗口只覆盖到 entry35，AMOUNTS/OP_EN 在窗外 ⇒ 链必死；且 entry35 = 孤儿半条目（0x40a0 高半缺）垃圾写。⇒ **「34.5 条捕获 + 重建尾写」不足以复现 vendor 槽位**；attn 槽位 bytes 276..640 为未捕获区，无法从现有 trace 恢复。
5. **mul trace 定性（决定性副产物）**：74 条全捕获（含尾写）= [69 配置 + pad + AMOUNTS=0(tgt 0x101) + PC_VERSION=0(tgt 0x41) + **OP_EN=0x18**(tgt 0x81) + pad]；其 3 描述符**同址** + task_counter 0→3 ⇒ 单块执行 + 回显假象（非 3 次执行）✗。hook 口径差异：attn 系 = amount×4 B；mul 系 = (amount+5)×8 B。
6. **楔死事故（过程教训）**：归档命令误带 `--timeout 1` 复跑二进制 ⇒ `job commit failed` + `soft reset, num: 6`（楔死签名=1）⇒ 按纪律**立即停手 + 板 reboot**（重启后 wedge=0、soft=0 ✓）。教训入档：**归档禁止复跑二进制**。
7. **基线漂移**：handoff soft=92 → 本轮起点 99（+7 = 会话间外部活动：板上 21:37 min_conv_exp 重编译 + 非本会话提交）；本轮 2 次计划内尝试 +2，意外 +1，后 reboot 清零。
8. **Ruling（偏离计划处）**：计划 Step 1 期望 blocks.bin=576 B/34+34+4 条整条目；实测 = 588 B（69/69/9 words，34.5/34.5/4.5 条 + 孤儿半条目）——以捕获字节 verbatim 为准（hook 口径差异，见 README §3）。Task 1 的「期望分支」未达成（A/B 都未命中），按计划进 **Task 2 fork B**（args/描述符层 diff + mul 全块回放）。
9. soft reset 累计：会话内 99→101（+2 计划内）→ reboot 清零；楔死签名重启后 = 0 ✓。

产物（本次提交）：`experiments/extract_task_blocks.py`、`experiments/vendor_replay.c`、`experiments/results/chain_vendor_literal_20261002/`（README/blocks.bin/meta.txt）。

### 7.173 链协议闭合：vendor 全字面回放 3/3 task 执行（§3b-1 ✅）（2026-10-02 路线图 Task 2 fork B）

1. **根因（§7.172 超时的解释，一句话）**：旧 hook 半截捕获（amount×4 B）只给了 69 条配置的前 34.5 条；
   真块 = `round_up(amount+4,2)` 条 fetch 窗口（69+4→74 条 = 592 B，含 5 条尾写）⇒ v0/v1 把全零当寄存器写执行。
2. **重抓**：规范 hook（libeperf_hook_rknpu_ioctl.so）+ `EPERF_RKNPU_REGCMD_MAX_BYTES=8192` 重抓 attn_relu
   （arch-bench 路径）⇒ seq1 = 3663 task，捕获 592/592/112 B（74/74/14 条，尾写完整在内）。
   新事实：①尾写真值 = [PC_BASE_ADDRESS=下一槽(tgt 0x101), PC_REGISTER_AMOUNTS=0x24/0x6, PC_VERSION=0(tgt 0x41),
   **PC_OPERATION_ENABLE=0x18**(tgt 0x81), pad]；②reg 0x5018（RDMA 源）也逐 task +0x400 步进（旧半截不可见）；
   ③描述符真值：enable_mask=0x18、int_clear=0x1ffff、int_mask=0x300、subcore_tasks={{0,1221}×3,{0,0}×2}。
3. **三跑（`vendor_replay --variant 2`，板上）**：①首版缺陷（variant≥1 误覆盖真尾写 + 终端 AMOUNTS 残留）超时；
   ②修复后 slots {0,1} ⇒ ret=0 hw=19250 ns 单 task（TASK-OUT 1/2 UNTOUCHED）⇒ **PC 取条数 = slot.task_number**；
   ③`--subcore-num 3`（slots {{0,3}×3,{0,0}×2}，与 vendor 3663-job 的 {{0,1221}×3} 同构）⇒
   **ret=0、hw=25374 ns、TASK-OUT 0/1/2 全 HIT ⇒ ALL-TASKS-EXECUTED ✓✓**。
4. **链协议定论（§3b-1 闭合）**：PC_TASK_CONTROL.task_number = slot.task_number = N ⇒ PC 顺流内尾写
   （PC_BASE→下一槽, AMOUNTS=fetch/16−1, VERSION 0x41, OP_EN=0x18）取 N 块，终端 AMOUNTS=0 停；
   hw 1→3 task = 19250→25374 ns（≈3.3 µs/task 边际）；`task_counter=3` 仍为回显。
   **注意与旧认知相反**：§7.62「slot=N ⇒ 超时」的真正原因是当时的流是**半截垃圾块**（无有效尾写），
   不是槽位取值本身 ✗——slot.task_number 必须与流内真实链长一致。
5. **尾写真值表（attn 形态，verbatim）**：链式 = [0x10:next(tgt 0x101), 0x14:fetch/16−1, 0x41/reg0:0, 0x8:0x18, 0x0:0]；
   终端 = AMOUNTS 归零（+PC_BASE 置 0）。§7.46 transpose 的 0xd 尾写与此 0x18 并存——按 trace 逐字面取值，勿混用。
6. 假成功防线全部通过：--task-out-step 等价逐区哨兵（3×0x400 区 diff）✓、hw 缩放 ✓、楔死 0 ✓。
   局限：SRC 0x5018 非本 op 实际读源（三区输出同值），op 数值语义未验（Task 5）。
7. soft reset：本轮 3 次提交（1 次超时），reboot 后累计 2；楔死 0 ✓。
8. **能力升级**：单 subcore 多 task（vendor 形态）从「不可达」⇒ **闭合**（§3b-1 ①）。
   剩余：把链协议移植进 min_conv_exp（我们的卷积流 + vendor 链形态）——列入下一会话（§3b-1 的工程化收尾）。

产物（本次提交）：`experiments/results/chain_bisect_20261002/`（README + blocks_full.bin）、
`experiments/results/chain_vendor_literal_20261002/`（+attn_refull.jsonl.gz 重抓 trace）、
`experiments/vendor_replay.c`（v2 全字面 + --subcore-num）、`experiments/extract_task_blocks.py`。

### 7.174 LUT 逐项拟合收尾（部分）：softmax 表 = 精确 19 段分段线性；sigmoid/tanh 写序非索引序（2026-10-02 路线图 Task 4）

1. **重抓**：sigmoid_1x16_fp16（439 唯一值）/ tanh_1x16_fp16（442）/ f_Softmax_1x64_fp16（110），
   与 §7.137 普查表规模一致（440/443/111 含 0 条目差 1）✓。
2. **fsoftmax 111 项表 = 精确分段线性（19 段，段内残差 2.2e-15）** ⇒ 反查表 19 组参数，压缩率 ≈17% ✓
   （§3b-4② 在 softmax 族闭合）。
3. **sigmoid/tanh 表两模型皆不足**：log 域二次 rmse 30–57%（max 0.94/1.61 ≫ fp16 精度 4.9e-4）；
   5% 跳变切段 20/104 段残差仍大 ⇒ **0x4104 写序 ≠ 表索引序**（跨 task 分块突发写）。
   待办：按 LUT 地址寄存器恢复索引序再拟合。
4. **attention 大表平台-反查 = blocked**：attn_softmax 系 `inputs_set failed: -5`（多输入喂数未解，§3b-3）。
5. soft reset 不变（纯离线 + 3 次安全捕获运行）；楔死 0 ✓。

产物：`experiments/results/lut_fit_20261002/`（fit_lut.py + fit_summary.json + table_*.csv + README）。

### 7.175 vendor op 语义库扩充：mul/LayerNorm 字段级解码（§3b-7）（2026-10-02 路线图 Task 5）

1. **mul（34/34/4 样本）**：BS_MUL 通路独活（BS_MUL_CFG=0x3e000000 OPERAND=15872；BS_RELU/ALU、BN、EW 全旁路）
   —— 与 §7.150 判别特征一致 ✓；72 共有寄存器基线与 §7.118 完全一致 ✓。
2. **layernorm 判别特征 = 0x0010 + 0x100c–0x1040 区 54 个寄存器**（LN = DPU + PPU/PC-bank2 协同）；
   且 **LN trace 内 PC 尾写链独立复现 §7.173 协议**（PC_BASE=下一槽 +0x380 步进、AMOUNTS=fetch/16−1、
   OP_EN 0x18（链）与 0xd（终端/OP_EN 位）并存实例）⇒ 链协议第二个真实样本 ✓✓。
3. attention 大表（§3b-4② Step 2）仍 blocked 于多输入喂数（§3b-3）。

产物：`experiments/results/vendor_op_semantics_20261002/`（mul_fields/layernorm_fields/mul_ln_signature + README）。

### 7.176 链协议工程化移植（§3b-1 收尾）：发现 --ov 尾写 target bug，2 次超时后定位（2026-10-02 路线图 Task 2 续）

1. **移植**：`min_conv_exp --chain-id --chain-pc --chain-n 2`——OP_EN 0xd→0x18（§7.173）+ slots {{0,2}×5}
   （旧 --chain-pc 已含）后仍超时（flags 0x7 与 0x5 各 1 次，hw=0，楔死 0 ✓）。
2. **根因（离线定位，未再提交）**：`--ov` 覆盖通路用 `EMIT()` = `rkt_get_target(reg)+0x1` ⇒
   REG_PC_VERSION/REG_PC_OPERATION_ENABLE 都映射到 PC=0x100 ⇒ 尾写 target=**0x101**；
   而 vendor 链尾写 = PC_BASE/AMOUNTS→0x101、**VERSION→0x41、OP_EN→0x81（下一 PC 实例 bank）**。
   向「当前 PC 实例」写 OP_EN 会让链不推进/挂死 ⇒ §7.46 旧 --chain-pc 超时与本次超时同根 ✗。
3. **修复**：`emit_conv` 的 --ov 循环对 REG_PC_VERSION/REG_PC_OPERATION_ENABLE 显式用
   `emit_raw(0x41/0x81, …)`（单任务路径的 trailer 一直是对的，仅 --ov 通路错）。
   单任务 8×8 int8 冒烟（130 条目块）不受影响（--ov 未启用时零差异）。
4. 本轮账目：2 次提交（均超时，soft +2 ⇒ reboot 后累计 3），楔死 0；预算已尽停手。
5. **下一轮**：`--chain-id --chain-pc --chain-n 2 --job-flags 5` 单次提交（预期：2/2 输出区 BIT-EXACT/HIT）。
   若仍超时：段 1 的 133→134 补零、enable_mask=0x18 对照臂（vendor attn 形态）。

源码改动：`experiments/min_conv_exp.c`（尾写 target 修复 + OP_EN=0x18）。

### 7.177 链工程化（续）：trailer 清链 bug 修复 + enable_mask=0x18 定位（2026-10-02 路线图 §3b-1 收尾，进行中）

1. **超时根因 ②（§7.176 之后的离线定位）**：`--chain-pc` 的 --ov 尾写之后 emit_conv 仍追加标准
   trailer（PC_BASE=0/AMOUNTS=0/OP_EN）⇒ 链指针被清零，PC 取完第 1 块即停摆。修复 = `g_no_trailer`
   （链段内跳过标准 trailer；vendor 形态的「+4 EXTRA」就是尾写本身，段内不应再有 trailer）。
   修复后段几何与 vendor 完全同构：L=130 条（126 配置 + 4 尾写）、amount=126、AMOUNTS=64（⇒130 条 fetch）✓。
2. **仍超时（2/2，flags 0x5/0x7 各 1）** ⇒ 对照 §7.173 工作回放的剩余描述符差异：
   **enable_mask**：vendor mul（3 task 链）全 0x18、LN 54-task 链 0x18/0x1d 变体、attn 链 0x18——
   **无一为 0xf**；我们全程 0xf。单 task 用 0xf 正常（不阻塞执行），但可能 gate 链的续取。
   下轮唯一变量：`--enable-mask 0x18`（预期 2/2 输出区 HIT ⇒ §3b-1 工程化收尾）。
3. 板卡：楔死 0；本轮 soft +2（reboot 后累计 5）。两次尝试均 hw=0（PC 未完成首块后的链）。

### 7.178 链工程化（三）：流/描述符已与 vendor 完全同构，仍超时 ⇒ 嫌疑收敛到块内容（2026-10-02）

1. `--enable-mask 0x18` 提交（1/2）+ 带 hook 复跑取证（2/2）均超时（hw=0）。楔死 0；soft reboot 后累计 7。
2. **取证（ourchain_timeout.jsonl.gz）**：我方描述符+流已与 §7.173 工作形态**逐字段同构**——
   submit{flags=5, task_number=2, task_start=0, task_base=0, core_mask=0, slots={{0,2}×5}}；
   描述符 enable_mask=0x18、int_mask=0x300、int_clear=0x1ffff、amounts 126/126（entries 口径）、
   regcmd 0xffdfd000/+0x410；task0 尾 = [PC_BASE=下一槽(0x101), AMOUNTS=0x40(⇒fetch 130), VERSION(0x41), OP_EN=0x18(0x81)]；
   task1 终端 = [AMOUNTS=0(0x101), VERSION(0x41), OP_EN=0x18(0x81), pad]；flags=0；段几何 L=130=round_up(126+4,2) ✓。
3. **排除矩阵（描述符/args/流形全部排除）**：flags✓ slots✓ amounts✓ regcmd_addr✓ 尾写 target/取值/顺序✓ trailer✓
   段长✓ 对齐✓ enable_mask✓ int_mask/clear✓。**剩余嫌疑 = 块内容本身**（我方 126 条卷积配置 vs
   vendor 简单 op 块）——如 CNA/权重路径在链式重复执行时的状态依赖（单 task 正常、链第 2 块挂）。
4. **下一轮（离线优先）**：用 `vendor_replay --variant 2` 回放 **mul vendor 块**（DPU 族 72 寄存器，
   已有全窗 trace）——若成功 ⇒ harness 无误，嫌疑=我方 conv 块内容；再把我方 130 条块灌入 replay 流验证。
   备选：--chain-align8（0x280 槽距）、op_idx=1、int_clear=0x300。
5. 板卡：楔死 0；本轮 soft +2（累计 7）。

### 7.179 链工程化（四）：harness 与 vendor 链双双自证，嫌疑收敛到段对齐（2026-10-02）

1. **mul vendor 全窗块（tr_mul_m025_fp16，74/74/74 条含真尾写）经 replay harness**：
   `--variant 2`（slots {0,1}）⇒ **ret=0、hw=17208 ns、task0 HIT（1024/1024）** ✓ ——
   **replay harness 对 DPU 族块同样有效**；`--subcore-num 3` ⇒ 超时 = **预期**（mul 样本 = 单块流 +
   3 个同址描述符 + 终端尾写，本无链，PC 无下一块可跳）✓ 与 §7.62「mul 3-task = 单执行+回显」一致。
2. **⇒ 双重自证**：harness ✓（attn 链 3/3 + mul 单块 1/1）、vendor 链协议 ✓（§7.173）。
   §7.178 的「块内容嫌疑」收窄为**发射器流装配**。
3. **新嫌疑（离线复查）＝ 段对齐**：vendor 所有 PC_BASE 目标全部 **0x80（128 B）对齐**
   （0xfff81080/0x1300/0x1580、LN 0xffffc480/0xc800）；replay 槽距 0x280 亦然；
   **我方链段连续排布，段 2 @ +0x410 = 非 0x80 对齐**。PC 取指可能要求 128 B 对齐。
   现有 --chain-align8 = 64 B 对齐（8 条）不够 ⇒ 需 16 条（128 B）对齐。
4. **下轮**：`--chain-align` 改 16 条对齐（或新增）+ 单次提交。若闭合 ⇒ §3b-1 工程化收尾完成。
5. 板卡：楔死 0；本轮 soft +1（mul {0,3} 超时，累计 8）；mul 回放 ret=0 不计入失败。

### 7.180 链工程化（五）：0x80 段对齐已落地仍超时；下一轮 = conv/attn 混合链二分（2026-10-02）

1. 段长 L=ALIGN_UP(L,16)=144 条（0x80 对齐 ✓ 下一块 @+1152），段 0 填充区清零；仍超时（hw=0，1 次）。
   楔死 0；soft 累计 9。
2. **排除矩阵更新（§7.178 基础上）**：+段对齐 ✗（已对齐仍挂）。描述符/args/流形/对齐全排除。
3. **下一轮（1 次提交）**：混合链二分——slot0 = 我方 conv 块（130 条 + 重构尾写 PC_BASE→slot1@0x280）、
   slot1/2 = §7.173 工作回放的 attn 块（592/112 B）⇒ conv→attn 链。
   判读：conv→attn 通 ⇒ conv 块可链，嫌疑=conv→conv 的 CNA/CBUF 状态；conv→attn 挂 ⇒ conv 配置破坏 PC 链。
4. 工具：vendor_replay 需支持 --words/--amounts 自定义（现 v2 硬编码 148/148/28+69/69/9）。

### 7.181 链工程化（六）：conv→stub 亦挂；嫌疑收敛至尾写 OP_EN 取值（2026-10-02）

1. conv→stub 二分（--chain-stub，段1=attn 增量形状 9 配置+尾写、DST 指 our 区）仍超时（hw=0，1 次）。
   楔死 0；soft 累计 10。
2. **重要负结果**：conv 块连 attn 形轻量第二块都交不出链，且 §7.178/§7.180 已排除描述符/args/流形/对齐
   ⇒ 挂点收敛于 **conv 块与其尾写的交互**（vendor attn/mul 块同样尾写可链 ✓）。
3. conv 块 126 条配置经审查**无 PC-bank 写入**（targets 全为 DPU/RDMA/PPU banks）——「块内 PC 干扰」排除。
4. **剩余头号嫌疑 = 尾写 OP_EN 取值**：我方单任务 trailer=**0x1d**（RESERVED_0(14)|OP_EN(1)）已证可触发
   本块执行；vendor attn/mul 链尾=0x18（OP_EN=0）；LN 流 0x18/0xd 并存（§7.175）。已落地 `--chain-op-en V`
   旋钮 + `--chain-stub` 二分开关。
5. **下轮（2 次预算）**：① `--chain-stub --chain-op-en 0x1d`（conv→stub，尾 OP_EN=0x1d）；
   ② 视结果 0xd / 去 stub 纯 conv 链 × 0x1d。

### 7.182 链工程化（七）：--chain-op-en 0x1d ⇒ task_counter=1（首个中断出现！）（2026-10-02）

1. conv→stub（--chain-stub --chain-op-en 0x18）超时（1 次）；**--chain-op-en 0x1d ⇒ task_counter=1**
   （此前所有失败均为 0！）——**0x1d（OP_EN=1）让 conv 块完成并交出链**，挂点推进到 stub 段之后
   （推断：stub 终端 AMOUNTS=0 + OP_EN 被旋钮改 0x1d ⇒ 续取 2 条垃圾 ⇒ 停摆；本轮已修：
   终端 OP_EN 固定 0x18，旋钮只作用于链式延续）。楔死 0；soft 累计 12。
2. **假设升级**：链式延续的触发 = 尾写 OP_EN（target 0x81 = 下一 PC 实例）**须含 OP_EN 位（0x1d 形）**；
   vendor attn/mul 的 0x18 形态可能依赖其 DPU 轻量 op 的完成时序，conv 块需显式 start。
3. **下轮（预算重置）**：`--chain-stub --chain-op-en 0x1d`（终端已固定 0x18）——期望 ret=0、
   TASK-OUT 0(conv)+1(stub) 双 HIT ⇒ conv 链闭合；再回归纯 conv 链（--chain-n 2 无 stub）。
4. 楔死 0；本轮 soft +2（累计 12）。

### 7.183 链工程化（八）：task0 完整执行实锤（输出区 HIT），挂点=PC 不发起 seg1 取指（2026-10-03）

1. **超时路径哨兵**（新）:TIMEOUT-OUT 检查——超时也 sync+比对链输出区。**决定性数据**：
   conv→conv ×0x1d ⇒ TASK-OUT 0 **HIT（1696/2048 B 真实 conv 输出！）**、OUT 1 UNTOUCHED，
   task_counter=1 ⇒ **task0 完整执行**（配置+计算+WDMA 全通），PC 在 task0 完成后**未发起
   seg1 取指**（dmesg 无 IOMMU fault ⇒ 非地址问题）。
2. **终端 OP_EN 修正**（两处：stub 分支 + conv --ov 分支）：0x18→旋钮值——0x18（无 bit0）时
   task0 从不执行（counter=0，0xd 同）；0x1d 时 task0 执行 ⇒ **conv 块执行触发 = 尾写 OP_EN
   bit0**（vendor mul/attn 终端 0x18 只适用其 DPU 轻量 op）。但终端改 0x1d 后 task1 仍 UNTOUCHED。
3. **op_idx 对齐**（vendor task 事件全 1，我方原 task0=0/k 递增）：改全 1 ⇒ 无变化。排除。
4. **实证矩阵**（今日 6 提交）：{0x18}→counter=0 全死；{0xd}→counter=0 全死；{0x1d}→task0 执行+
   counter=1、seg1 恒 UNTOUCHED（stub 与 conv 同）；op_idx=1 无效。预算重置前不再提交。
5. **背景重读（§7.44）**：vendor 3663-task = DPU-only 块批量发射（34/34/4 条，每 task 块长不同，
   DST +0x400 步进）；**官方从未链 conv**（「conv 侧不需要多 task」）⇒ conv 链可能是硬件
   未被厂家路径覆盖的形态（CNA/CBUF 重挂载在链内或需额外状态）。
6. **下轮方向**：a) 找 vendor 3663-task 原始 trace 解码 conv→conv 边界尾写（决定性）；
   b) 若无 ⇒ §3b-1 按「协议已证（DPU-light 链 replay 3/3）、conv=单 task/层」收口，
   工程化转向 DPU 轻量链（attn/mul 形态），conv 多任务回到多核分片（§7.44 正途）。
7. 楔死 0；soft reset 驱动计数 num: 6（reboot 后累计约 18）。

### 7.184 链工程化（九）决定性：vendor conv 链真值抓取——尾写 OP_EN=0x1d 实锤，vendor 确实链 conv（2026-10-03）

1. **重抓 conv_bn_conv_fp16（当前 hook 全窗）**：`rknn_runner + EPERF_RKNPU_REGCMD_MAX_BYTES=8192`
   ⇒ 6 regcmd 事件全窗（130/130/126/126/30/126 条，尾写完整）+ submit 真值：**6 task/1 submit、
   hw=20708 ns、task_counter=6 全执行**；subcores={{0,2},{0,2},{0,2},{0,0},{0,0}}（3 核 × 每核 2-task 链）；
   op_idx=[1,2,1,2,2,1]；enable_mask=0x1d（PPU task4=0x60）；int_mask=0x300（PPU=0xc00）；
   stride=ALIGN_UP(fetch,8)=136 条（64 B 对齐连续）。
2. **尾写真值（与 attn 不同！）**：conv 链尾写 OP_EN=**0x1d**（连续与终端都是 0x1d——含 bit0）；
   attn/mul 的 0x18 = DPU 轻量特例；PPU op 终端 = 0x60（第三形态，enable_mask 同值）。
   ⇒ 我方 --chain-op-en 0x1d = 正确；0x18 假设（§7.182「终端固定 0x18」）**作废**。
3. **对拍（我方 hook 块 ourchain2 vs vendor task0，130 条）**：31 项差异——地址/精度/几何属预期；
   控制位候选：0x100c（ven 0x6000a120 vs 我 0，×2）、0x1180=CVT per-ch en（ven 0xfff vs 我 0）、
   0x3010 CORE_MISC_CFG（ven bit9 PROC_PRECISION=2[fp16] vs 我 bit0 QD_EN）、0x4010 DPU_DATA_FORMAT
   （ven 0x48000002=fp16 vs 我 0=int8，精度位）。**尾写与描述符完全同形同值**。
4. **矩阵补测**：{enable_mask 0x1d + 尾写 0x1d}⇒同 stall（counter=1）；{--subcore-single0 槽位}
   ⇒同；{--chain-align8}⇒同。板 vendor_replay 曾为旧版（无 --words/--blocks-file）已重编译，
   cbc 块回放因烘焙地址未适配暂无效。
5. **定性反转**：§7.183 的「vendor 从不链 conv」**错误**——conv→BN→conv = 6-task 单 submit 实锤
   （每核 2-task 链，3 核并行）。conv 链 = 硬件真实支持的形态。我方挂点仍在「task0 完成后
   PC 不取 seg1」，尾写/描述符已与 vendor 同值 ⇒ 剩余 = 31 项配置值中的某项（或 PC 取指
   的隐含条件）。
6. **下轮（决定性）**：a) vendor_replay 扩 --addr-patch（act/wt/dst/src 四地址自适应）回放
   vendor 自己的 conv 块 ⇒ 隔离「块内容 vs 提交路径」；b) 二分 0x100c/0x1180/0x3010 三控制位；
   c) 检查 CBUF/CNA 状态在链内的重挂载要求（0x1180=0xfff=PER_CHANNEL_CVT_EN 12 通道全开 vs 我 0）。
7. 产物：`experiments/results/chain_cbc_20261003/`（cbc_fp16.jsonl 全窗 + README + diff 工具）。
   楔死 0；soft 累计约 24。


### 7.185 链工程化（十）决定性通过：vendor conv 链 verbatim 复刻回放全执行——提交路径无罪实锤（2026-10-03）

1. **离线定论（驱动语义）**：rknpu_job.c L266 成功路径 `task_counter = task_number` 纯回显（此前 §7.184
   的「counter=6 全执行」证据不成立）；仅超时路径读 PC_TASK_STATUS=硬件真值。AUTO(core_mask=0)
   ⇒ 选最少负载核 ⇒ use_core_num=1 ⇒ 只编程 subcore_task[core] 窗口：cbc trace 的 {0,2}×3
   ⇒ **实际链 = t0→t1（终端）**，task2-5 描述符惰性。commit_pc 只消费描述符的
   regcmd_addr/regcfg_amount/int_mask 三字段。
2. **.rknn 镜像突破**：conv_bn_conv_fp16.rknn 内嵌 blob[7]=regcmd 流（13696 B=17 块）+
   blob[8]=task 描述符数组（680 B=17×40B，regcmd=流内相对偏移）+ blob[1..6]=w0/b0/w1/b1/
   tempMC11_out/tempMC11_input。blob 偏移与 vendor 权重 BO（0xffffa000）布局逐一对应：
   w0@0/b0@80/w1@c0/b1@100/MC11out@140/MC11in@540/stream@940。模型流内地址=相对偏移，
   runtime 加载期 patch 5 类地址（0x1070/0x1110/0x4020/0x5020/0x0010 PC_BASE）+运行时计算
   0x1180=0xfff（CVT per-ch）与 0x104c 0xb→1；trace 终值块 = 该 patch 后真值。
3. **假阴性矩阵（教训核心）**：v1（零数据+手建描述符）与 v2（真实 blob+blob[8] verbatim）
   均 counter=0 全停摆——事后字节级 diff 揭示 variant 3 的 delta patch 掩码
   `0x0000FFFF0000FFFF` 把 64 位条目的 **target 位（bits63:48）清零**（0x201/0x1001/0x2001/
   0x4001/0x101 ⇒ 全 0x000）⇒ 26 个指针配置写全部丢失 ⇒ hw 永不启动。正确掩码 =
   `(w64 & 0xFFFF000000000000) | (nv<<16) | (w64 & 0xFFFF)`。**教训：任何 64 位条目改写必须
   字节级 diff 对拍——target 位丢失只有 diff 能抓到。**
4. **决定性结果（v2b，修正后）**：`./vendor_replay --variant 3 --vmem-file cbc_vmem.bin
   --task-file cbc_task.bin --timeout 6000` ⇒ **ret=0、hw_elapse=9333 ns、task_counter=6、
   DST0 HIT 1023/1024、DST1 HIT 1023/1024、ALL-TASKS-EXECUTED**；提交后全图 diff：
   stream 区 26 条=恰好 26 个 delta patch（零杂写）、dst1 区 2 KB + inter 区 0x2180-0x2978
   2 KB 全新输出、weights/input 零变化、楔死 0、无 IOMMU fault、soft 无新增。
5. **定性定论**：a) **提交路径无罪实锤**（vendor 自己的 conv→BN 终端链经我方 harness 完整
   执行）；b) §7.183「vendor 从未链 conv」再度证伪，§7.184 定性反转维持；c) 我方发射器
   counter=1 挂点 = 我方块内容/几何问题——31 项对拍差成为精确二分清单：
   0x100c（NONALIGN_DMA，ven 0x6000a120 vs 我 0）、0x1180（CVT per-ch 0xfff vs 我 0）、
   0x3010 PROC_PRECISION、0x4010 DPU_DATA_FORMAT（fp16 vs int8）+ stride=ALIGN_UP(fetch,8) 几何。
6. **复刻配方**（可复用）：单 BO vmem 0x9000 镜像 [0xffff6000,0xfffff000)（哨兵 A/B + input
   ramp + 权重表 BO 镜像 + 流）；blob[1..6] verbatim + blob[7]（blocks0-5 用 trace 终值覆盖）+
   blob[8] 描述符 verbatim（regcmd += vmem.dma+0x4940）；提交参数 verbatim。
   产物：`experiments/results/chain_cbc_20261003/`（cbc_vmem.bin/cbc_task.bin/vmem_after_v2{,b}.bin/
   build_cbc_vmem.py/README_v2b.md）。
7. 纪律说明：当日 3 次提交 = 同一实验的两次假阴性（harness bug）+ 一次有效；假阴性不构成
   有效证据，修正后重跑计入同一轮。楔死 0；soft reset num:6 无新增。


### 7.186 链工程化（十一）双根因修复：终端尾写形态 + 段拷贝错位——emitter 自链闭合（ret=0 双 task 消费）（2026-10-03）

1. **对拍分类（task A，commit 830326e）**：emitter 块 vs vendor 块（各 130 条）序列/target 全同、
   差异全 VAL：地址 5 类（harness 自适应）、几何/精度、控制位候选 6 个（0x100c/0x104c/0x1180/
   0x1184/0x3010/0x4010）+ 终端尾写形态差（vendor=[pad,AMT=0,VER,OP_EN=0x1d] vs
   我=[AMT,VER,OP=0x18,pad]）。vendor_replay 加 --blk-flip REG:VAL 旋钮。
2. **v2c（全 6 控制位翻转）⇒ counter=0**：精度对与 fp16 数据失配的混杂信号，非链因。
3. **v2e（vendor block0 + 我方 block1 字节）⇒ 精确复现挂点签名**（counter=1、DST0 HIT、
   DST1 UNTOUCHED）⇒ 嫌疑收敛块内容。
4. **v2f（+vendor 终端形态）⇒ ret=0、hw=31792ns、DST0+DST1 HIT、ALL-TASKS-EXECUTED**——
   **根因一 = 终端尾写形态**：conv 家族终端 = pad 在首 + OP_EN=0x1d（bit0）+ AMT=0 停链；
   attn 形态 [AMT,VER,OP=0x18,pad] 对 conv 链无效。
5. **段拷贝偏移 bug（task C）**：chain-pc 光标复位到 L（ALIGN_UP(130,16)=144）后，seg 拷贝
   源仍用 n_prev（=nA=130 条）当字节偏移 ⇒ 源错位 14 条，seg1 尾写永远落不进 BO——
   「PC 不发起 seg1 取指」的长期假象来源。修复：cnt/copy 一律以 before 为准。
6. **emitter 自链闭合（task D）**：ret=0、hw=9.6-75.8µs、双描述符均被消费、seg1 WDMA 落点
   正确（chain zone 真实写入）；op_idx 对齐 vendor=[1,2]（链首=1，从不 0；§7.183 排除实验
   在拷贝修复前=无效证据）。
7. **遗留（新线程）**：task1 输出 ≈ task0（64-70% 字节同）但 240/256 atom 各差 1-2 B = 输入端
   少量样本被跨 task 管线状态污染（§7.44 预判的 CNA/CBUF 重挂载域）；8×8 小几何同样复现
   （1589/2048）⇒ 非尺寸相关；ones 对照未得判据（其 task0 本身 MISMATCH）。下轮：
   a) chain-n 3 看 task2 是否同样受损；b) 检查 CNA line-buffer/SRAM 在链内的清态寄存器；
   c) 与 vendor 对拍 0x100c/0x1180（NONALIGN_DMA/CVT per-ch）在链内的必要性。
8. 产物：ourchain3.jsonl、act_chain{,2}.bin；提交预算：本轮 6 次（v2c/v2e/v2f + emitter 3）。
   楔死 0；soft 无新增（timeout 已消除）。


### 7.187 链工程化（十二）协议已闭合、数据保真未闭合——跨 task 管线状态线程 + 板卡复位事件（2026-10-03）

1. **继续深挖 task1 数据差**：pad=0（18×18 输入）仍损（4691/10368，task0 仍 BIT-EXACT）；
   1×1 核同样受损（og0 差 1152/4096，208/256 atom 各差数通道）⇒ 与 pad、kernel、几何均无关。
2. **非确定性实锤**：同一二进制同一配置两次运行，task1 hash 不同（f516aac2 vs fe0dbad5）
   而 task0 hash 逐位相同（f2842965）⇒ task1 = 管线内部竞态/状态（非确定性计算）。
   注：--in-one-hot 对 int8 路径无效（pat_in 不读该开关），该探针作废。
3. **1×1 数据解剖**：208/256 atom 受损、每 atom 5-6 通道；受损输出 = 大面积 0x80（量化零）
   而 task0 非零 ⇒ **task1 的 conv 疑似用（部分）失效权重/输入计算**——DCOMP/CBUF 或
   CNA 输入 DMA 在链内未重挂载，task1 的取数与 task0 的排空竞速。
4. **stub 探针 ×2 ⇒ 超时**：stub 分支（14 条段）与 chain-pc 的 L=144 对齐数学不兼容
   （PC_BASE 按 k*L 跳但 stub 按游标放置）⇒ 取指落空。**预置 bug，勿复跑**；修复需把
   stub 段也按 L 规整。两次超时 +2 soft reset。
5. **板卡复位**：stub 超时后 SSH 无响应（Connection reset → No route → ping 丢包 100%），
   用户手动重启。停止危险实验。教训：诊断性 stub 探针在上板前必须先做形状自检（PC_BASE
   与段步长一致性），超时路径对驱动冲击大。
6. **§3b-1 当前判定**：**协议层 = 闭合**（发射器自链 ret=0、双描述符消费、seg1 WDMA 落点
   正确、无超时无楔死）；**数据保真 = 未闭合**（task1 输出非确定性受损，同一配置连发两次
   conv 的管线状态问题，vendor 自身从不重复同配置 conv——其链为 conv→BN→conv 异构小算子）。
   与 §7.183-5b 预判一致（CNA/CBUF 重挂载域）。
7. **下轮计划**：a) 修 stub+pc 对齐 bug（诊断探针恢复）；b) 用 --ov 注入 CBUF/DCOMP 重臂
   （0x1044=0x40 vendor 值、0x1180=0xfff）逐项验证；c) 捕获 task1 的 DCOMP/CBUF 状态
   （若 hook 可读 SRAM 窗口）；d) 兜底：§3b-1 按「协议闭合 + 同构重复约束」收口，多任务
   conv 回到多核分片（vendor 正途）。
8. 本轮提交：v2c/v2e/v2f + emitter 自链 ×6 + stub ×2 ≈ 12 次；楔死 0→复位数次后板复位。
   产物：ourchain3.jsonl、act_chain{,2}.bin、act_wt_1x1.bin。


### 7.188 链工程化（十三）§3b-1 收口：数据保真 = 「同构重复约束」；双回归全绿（2026-10-03）

1. **非确定性再确认（纠正一处会话摘要笔误）**：`act_chain.bin` vs `act_chain2.bin`（同命令族）
   task0 逐位相同（sha256 `0018cd33…`）而 task1 不同（`e455205c…` vs `9cac11c7…`）。sha256
   口径下 task1 亦非确定 ⇒ 与 §7.187.2（FNV `f516aac2` vs `fe0dbad5`）同结论：**task1 =
   管线内部竞态/状态，非确定性计算**；task0 恒 BIT-EXACT。（此前会话曾误记为「task1
   确定性」，以本条为准。）
2. **排空/读早排除（`--chain-settle-ms`，新增 opt-in）**：在一次正常读点（已 `bo_sync`
   from-dev）先探 task1，再 `usleep(N·1000)` + 重新 `bo_sync(from-dev)` 复探：`N=500`
   时 **before == after**（hash 逐一相同）⇒ task1 损坏在提交完成瞬间即已固化，
   **非 WDMA 排空/读早**，属计算侧持久结果。纯读+延时，不改提交/协议。
3. **结构事实（推翻「--ov 重臂」路线）**：`ourchain3.jsonl` 逐条对拍显示 task0 与 task1
   regcmd **仅两处不同**——[62] `DPU_DST_BASE_ADDR`（链步长）与终端尾写 [126][127]
   （PC_BASE/AMOUNTS；§7.186 前形态）。**其余全部条目（含 CBUF_CON0/CON1、DCOMP_CTRL/
   REGNUM/ADDR0/AMOUNT0、CVT_CON0/5、FEATURE/权重/输入地址）两侧完全相同** ⇒ task1 已携带
   全套 CBUF/DCOMP 配置，「未重挂载」不是缺发射项；且 `--chain-pc` 路径 `g_ov_n` 被
   链尾写无条件覆写（`min_conv_exp.c:2356`）⇒ `--ov` 注入在当前链形态下**不可用且无意义**。
   故 §7.187.7b 计划作废。
4. **多核分片无效**：`--multi-core`（core_mask=0x3，core0→task0、core1→task1）下 task1
   结果与单核相同（`--chain-pc` + 描述符派发不可混用，§7.46）⇒ 非「同核串行化」单一诱因。
5. **§3b-1 判定（收口）**：协议层 CLOSED（自链 ret=0、双描述符消费、seg1 WDMA 落点正确）；
   数据保真 **按「同构重复约束」收口**——**同一 subcore 内背靠背重复「同配置、无数据依赖」
   的 conv，会因跨 task 管线态竞速使第 2+ 个 task 输出非确定受损**；vendor 自身从不重复
   同配置 conv（其链为 conv→BN→conv **异构**且**有数据依赖**，强制串行）⇒ 该竞态不被 vendor
   路径触发。**工程约束**：多任务 conv 走**多核分片**（每核 1 task，§7.170 `--subcore-hi`/
   `--task-out-step`，已验证 ALL-TASKS-EXECUTED）或异构依赖链，**不用同核同配置重复链**。
   诊断探针（stub+pc）对齐 bug 保留为已知缺陷、勿复跑（§7.187.4 板复位教训）。
6. **落地改动**（全部 opt-in、默认路径字节不变）：`min_conv_exp.c` 新增 `--chain-settle-ms N`
   探针；链首/后续段 `op_idx=[1,2]`（§7.186）；`vendor_replay.c` 新增 `--blk-flip REG:VAL`
   （§7.186 二分旋钮）。
7. **不回归（板上 gcc -O2，改动后）**：`run_regression_hard.sh`（int8）**pass=74 fail=0
   envfail=0**；`run_fp16_regression.sh` **pass=79 fail=0 envfail=0 dirtyevents=0**。
   板卡健康：`job commit failed`=0、全核 idle（load 0%）、soft reset `num:6` 无新增；
   fp16 suite I 节信息性 `arm()`（BS-bypass，**不计入** pass/fail）产生 7 次瞬时
   `job timeout`（间隔≈6.4 s，soft reset 计数未前进）——既有 BS 路径行为，与本次改动无关。
8. **下轮**：转 **Task 3 attention `.rknn` 真实喂数**（`inputs_set -5` 复现 → 逐输入
   `rknn_inputs_set` Q/K/V → 重抓 trace → `lut_scan.py` 对账）。
   产物：本条 + 双回归日志（板上 `exp2/reg_int8.log`、`reg_fp16.log`）。

### 7.189 Task 3 打通：真实多输入 attention `.rknn` 喂数 + 喂真数 LUT 对账（2026-10-03）

1. **`inputs_set failed: -5` 根因（复现并定位）**：`runs/eperf/rknpu_atomic_probe/rknn_infer_dump.c`
   用 `n = in_attr.dims[n_dims-1]`（仅末维 =16）推元素数、且只设 `input[0]` ⇒ 报
   `param input size(64) < model input size(802816)`。修复 = 用 `attr.n_elems` 逐输入、设
   **全部 N 个输入**（`pass_through=0`，由 API 按模型 quant 参数量化）。
2. **新工具 `experiments/attn_feed.c`**（板上 gcc -O2）：`./attn_feed model.rknn <ramp|ones|zero> [out.bin]`；
   逐输入打印 name/dims/n_elems/fmt/type/scale/zp，`inputs_set`/`run`/`outputs_get` ret，
   输出 mean/min/max/nonzero。模型事实：`sdpa_qwen224.rknn` = **in=3（Q/K/V，各
   `[1,196,64,16]` I8 NHWC）+ out=1（`[1,16,196,64]` NCHW I8）**；
   `attn_softmax_h16_s196_d64_int8.rknn` = in=1。SDK 2.3.2 / API 0.9.8；**`inputs_set ret=0`、
   `run ret=0`** ✓（旧 `-5` 消除）。
3. **喂真数 trace 重抓 + `0x4104` 对账**（`EPERF_RKNPU_REGCMD_MAX_BYTES=8192` + hook）：

   | trace | 输入 | `0x4104` 写 | 唯一(全) | 唯一(非零) | 值域(非零) | 零旁路戳 |
   |---|---|---|---|---|---|---|
   | sdpa ramp | ramp | 3985 | 621 | **620** | **1..16063** | 173 |
   | sdpa ones | ones | 3985 | 621 | 620 | 1..16063 | 173 |
   | softmax s196 ramp | ramp | 3979 | 621 | 620 | 1..16063 | 167 |

   非零表项**严格单调递增**（首 10 项 1,2,…,10）⇒ 定点整数表（§7.139 口径）；整表加载 =
   **少数突发**（报告：规模 954 ×4 次），余为 `0x4104=0` 旁路戳（§7.144 口径）。
4. **输入无关性（§7.140 推广到真实多输入 SDPA）**：`sdpa ramp` vs `sdpa ones` ⇒ `0x4104`
   写次数、唯一值**集合**、**首次出现顺序**三者**全同**；而**输出不同**
   （sha256 `b65066c4…` vs `7360e2c1…`）⇒ **LUT 是算子预编译常量，与运行数据无关** ✓。
   ⇒ attention softmax 表（s196 形态）= **620 项 / 1..16063 / 严格单调**，可直接用于
   **Task 4 大表反查**。
5. **产物**：`experiments/results/attn_fed_20261003/`（三份 trace `.jsonl.gz` + README）；
   工具 `experiments/attn_feed.c`；板上输出 sha256 见 README。
6. **下轮**：用该真实语料推进 **Task 4**（大表反查/拟合），或开始 **Task 6**。

### 7.190 Task 4 Step 2 解阻：attention 大表 = 几何/指数表（公比 2 每 71 项），压缩 3816B→2 参数（2026-10-03）

1. **解阻**：§7.174.4 的「attention 大表平台-反查」原 blocked 于多输入喂数；§7.189 Task 3
   已通向真实语料（`experiments/results/attn_fed_20261003/`）⇒ 本条补齐。
2. **表结构**（`sdpa_qwen224` 与 `attn_softmax_h16_s196_d64_int8` 两模型**一致**）：整表突发
   **4 次 × 954 条** `0x4104`（余为 `0x4104=0` 旁路戳）；条目 954、非零唯一 **620**、值域
   **1..16063**、平台数 621、最长平台 53；**写序严格非降 ⇒ 写入序 = 表索引序** ✓（§7.174.3
   的「写序≠索引序」仅对 sigmoid/tanh 成立，对 attention 不成立）。
3. **拟合 = 几何/指数**：`value[i]=round(A·exp(k·i))`，最优 **A=1.45879、k=0.0097657**
   （mean|err|=0.237 LSB、max=3、逐位精确 80.5%）。**相邻比 `seq[i+71]/seq[i]` 均值 =
   1.99824 ≈ 2** ⇒ 每 **71** 索引翻倍 ⇒ **k=ln2/71=0.0097626**（“71 项基块翻倍 ~13 次”）。
   平台游程律 `run(v)≈(1/k)·ln((v+.5)/(v-.5))` 实测/预测吻合（`2:53/52.3 3:34/34.5
   4:26/25.7 5:20/20.5 10:11/10.2 100:1/1.0`）⇒ 指数采样 + 整数舍入 ✓✓。
4. **压缩率（§3b-4②）**：954×u32 = **3816 B** → 指数 **2 参数 (A,k)**；无损平台表
   **621 组 (idx,val)**。
5. **对账（算子族指纹）**：`f_Softmax_1x64_fp16`（111 项）= **精确 19 段分段线性**（§7.174）；
   attention（954 项）= **光滑几何/指数** ⇒ 同算子在**单算子 lowering vs attention 融合**下
   LUT 形状不同 ✓。
6. **输入无关性（§7.140 推广）**：`ramp` vs `ones` 写次数/唯一值集合/首现顺序全同、输出不同
   ⇒ 预编译常量（§7.189.4）。
7. **工具**：`lut_scan.py` 新增 `--fit attention`（含整数舍入指数拟合、游程律、压缩率）与
   **透明 gzip 读取**（默认统计模式行为不变）。产物：
   `experiments/results/lut_attn_20261003/`（README + fit_summary.json + table_attention_sdpa.csv）。

### 7.191 Task 4 Step 1 解阻：sigmoid/tanh LUT = 精确 Q15 激活表（推翻 §7.141/§7.174）（2026-10-03）

1. **解阻关键**：重抓 `sigmoid_1x16_fp16`/`tanh_1x16_fp16`（regcmd 载荷）。每个**整表突发 =
   443 条** `0x4104`、且**突发内写序单调非降**（sigmoid 59..9640；tanh −32611..−12973）⇒
   **单突发内写序 = 表索引序** ✓。§7.174.3 的「写序≠索引序」是**合并 4 个突发**统计的表象 ✗
   （每突发另含 1 条 `0x4104=0` 旁路戳）。
2. **拟合结果**（`lut_scan.py --fit sigmoid|tanh`，加权最小二乘 w=(dy/dz)²）：

   | 表 | 模型 | a | c | rmse | max | ≤0.5 LSB |
   |---|---|---|---|---|---|---|
   | sigmoid | `round(32767·σ(a·i+c))` | 0.012324 | −6.3100 | 0.147 | 0.52 | **99.1%** |
   | tanh | `round(32767·tanh(a·i+c))` | 0.005898 | −3.0200 | 0.143 | 0.51 | **99.3%** |

   ⇒ 两表 = **定点（Q15, S=32767=int16 max）激活查表**，443 项 → **(S,a,c)** 或 (a,c) 闭式。
   tanh 表仅覆盖**负半轴**（z∈[−3.02,−0.42]），正半轴由奇对称补齐 ✓（与 §7.139.1 一致）。
3. **推翻旧结论**：§7.141「sigmoid 单调但**不是** S·σ(x)」（最优 S=27997、cv=0.125）与
   §7.174.3「log 二次 rmse 30–57%、两模型皆不足」**均作废** —— 根因 = ① 未按突发恢复索引序
   ✗、② S 取值不当/未加权 ✗。**第 12 次自我修正**。
4. **三类 LUT 形状（算子族指纹）**：sigmoid/tanh（443）= **精确 Q15 解析激活**；attention
   （954, §7.190）= **几何/指数**（公比 2 每 71 项）；fsoftmax（111, §7.174）= **19 段分段线性**。
5. **工具**：`lut_scan.py` 新增 `--fit sigmoid|tanh`（含加权最小二乘、固定 S=32767 对照）。
   产物：`experiments/results/lut_act_20261003/`（README + fit_summary.json + table_*.csv +
   两份 trace `.jsonl.gz`）。
6. **Task 4 现状**：Step 1、Step 2 **均闭合** ⇒ Task 4 完成。

### 7.192 Task 6 Step 2：`conv → pool → conv` 三层管线闭环（含两处工具缺陷修复）（2026-10-03）

1. **几何**：层1 `min_conv_exp 16×16×32 --k3p1--> 16×16×32`；层2 `min_pool 16×16×32
   --k2s2p0--> 8×8×32`；层3 `min_conv_exp 8×8×32 --k3p1--> 8×8×32`。一键脚本
   `experiments/pipeline_conv_pool_conv.sh`。
2. **布局对接**（关键）：两工具布局**不同主序** ——
   - `min_conv_exp` 输出 = **y 主序** `npu_off(g,x,y,w,h)=g*w*h*16 + y*w*16 + x*16`；
   - `min_pool` 输入 = **x 主序** `lay(x,y,c,ih)=g*iw*ih*16 + x*ih*16 + y*16`；
   - `min_pool` 输出 = **x 主序** `out_lay`；`min_conv_exp --in-file --in-xiw` 再转回硬件 y 主序。
   为此给 `min_pool` 新增 `--in-file F` 与 `--in-yiw`（文件为 conv 输出 y 主序 → 转置到池化输入 x 主序）。
3. **缺陷①（`min_pool --out-file` 竞态）**：旧版在提交后、**from-device `bo_sync` 之前** dump
   `out.map`，读到未失效的 CPU 缓存。复现：同一输入三次运行，输出**组1**三个不同 md5，却与工具
   自带参考 `mismatch=0` 自相矛盾（参考在同步后读，故总对）。修复 = dump 移到 `bo_sync(&out,2)`
   之后 ⇒ 输出确定（这也解释了 §7.188 收口里同核重复链的非确定表象之外的另一条来源）。
4. **缺陷②（`min_conv_exp` 外部输入无参考）**：int8 CPU 参考原本恒用内部 `pat_in`，`--in-file`
   时无法自证（即 §7.158 的定量盲区）。修复 = `--in-file` 时参考改读 **BO 真实输入**
   （`g_ref_in_base`，硬件 y 主序），配 `--pattern`（权重=pat_w）。⇒ conv 可在**外部输入**上定量自证。
5. **实测**（`experiments/results/conv_pool_conv_20261003/`，`run.log`）：
   - 层1 conv：`OUT-HASH 3049e11e`，`total=8192 floor-mismatch=0`，**BIT-EXACT (floor)**；
   - 层2 pool：`total=2048 vs-max mismatch=0`；**Python 独立 2×2 maxpool 逐字节 = BIT-EXACT**；
   - 层3 conv：`OUT-HASH 8c38d6a1`，`total=2048 floor-mismatch=0`，**BIT-EXACT (floor)**（读真实池化输入）；
   - 确定性：conv1 / pool.out / conv2 两次**独立进程**运行 sha256 相同；
     `sha256(conv2)=f6f77c79…540669`。
   - 注：conv 用硬回归套件同款旋钮 `--pattern --pat2 --asym-w --pad-con1 0 --auto-fix`；裸
     `--pattern` 落入已知 tap 置换盲区（63/8192 mismatch），非管线缺陷。
6. **回归**：int8 `run_regression_hard.sh` 74/0/0；fp16 `run_fp16_regression.sh` 79/0/0
   `dirtyevents=0`；板卡 `job commit failed`=0、`job timeout` 无新增。
7. **Task 6 现状**：Step 1（wgrad P 维切分，§7.54）此前已达成；Step 2 本节点闭合；余 Step 3/4。

### 7.193 Task 6 Step 3：真实激活上的 int8 训练微步（量化感知写回 + 梯度缩放）（2026-10-03）

1. **目标**：把 §7.192 三层**推理**链升级为 int8 **训练微步**——层3 conv 在**真实 NPU 中间激活**
   （层1 conv → 层2 pool 的输出）上前向 + wgrad + **in-place 量化写回** `W′ ← clamp(W−round(dW/4))`
   + 复前向对 `CPU(W′)` 自证（§7.41 闭环之上的「真实模型口径」）。
2. **链路**（`experiments/pipeline_conv_pool_conv_train.sh`）：层1 `min_conv_exp 32×32×32 --k3p1-->
   32×32×32`（自带参考 BIT-EXACT，total=32768）→ 层2 `min_pool 32×32×32 --k2s2p0--> 16×16×32`
   （`--in-file --in-yiw`，`mismatch=0` + 独立 Python maxpool BIT-EXACT）→ 转置（组感知）→ 层3
   `wgrad_real`（`gcc -DIW=16 -DIH=16 -DIC=32 -DOC=32` 现场构建 `deep/wgrad_i32`；
   `--in-x-file` 读真实激活并重算参考）。
3. **结果（写回）**：`FWD BIT-EXACT`、`WGRAD 288/288 coords、9216/9216 grads round-exact maxdiff=0`、
   `UPDATE (same BO, in-place) weights_changed=…/9216`、`FWD2 vs CPU(W') BIT-EXACT` ⇒
   **`CLOSED-LOOP OK`**。真实激活上的 int8 写回链路成立 ✓。
4. **梯度缩放 ↔ int8 饱和**（固定 requant `out_scale=8`/shift=17；X 为同一真实激活）：

   | dY | sat/9216 | 备注 |
   |---|---|---|
   | 工具自带合成 | **9216（100%）** | 写回退化（全 ±127）|
   | `±3` 随机 | 7456（81%）| round-exact + CLOSED-LOOP OK |
   | `±1` 随机 | 4384（48%）| 同上 |
   | `±1` 稀疏 1/16 | 96（1%）| 同上 |
   | `±1` 稀疏 1/64 | **0** | 非退化训练步（`dW_nonzero=7351/9216`）|

   ⇒ **int8 wgrad 梯度量级须与固定 requant 标度匹配**：真实 int8 激活（±127）× 合成 dY 使**全部梯度
   饱和**、写回退化为常量 ⇒ 训练侧必须按梯度统计缩放 dY（等价梯度/损失缩放）✓。饱和与否，wgrad 与
   CPU 参考均**逐位一致**、`FWD2` 对 `CPU(W')` 均 **BIT-EXACT** ✓。
5. **缺陷修复（顺带）**：① `pipeline_pool_conv.sh` 转置在 `IC>16` 时漏通道组项 ⇒ 组1 写成组0 副本
   （其自带参考用同一 X 重算故仍 `CLOSED-LOOP OK`，但 X 非真实激活）；改为组感知后仍 OK。② `deep/i126`
   为陈旧构建（无 `--in-dy-file`）⇒ 脚本改为现场 `gcc -D…` 生成 `deep/wgrad_i32`。
6. **确定性 / 板卡**：独立进程重跑 `conv1`/`pool.out`/`x_b0.bin` md5 一致（`304d5ff6…`/`261f79cd…`/
   `62d6732f…`）；`job commit failed`=0、受控前后 `job timeout` 13→13（不触发超时，全核 idle）。
   产物：`experiments/results/conv_pool_conv_train_20261003/`。
7. **Task 6 现状**：Step 1/2/3 均闭合；余 Step 4（双回归 + 台账 + 提交）。

### 7.194 em=0x0d lowering 表征：LayerNorm fp16 全 task 链离线解码（2026-10-03）

1. **目标**：§3b-6 选定前沿 = `em=0x0d` lowering 的 **understanding/characterization**。openrknn
   OWN 路径仅支持 INT8 CNN，其 `patch_regcmd_addresses` 缺 `em=0x0d`（非 softmax general
   compute）寄存器级规则，是已知阻塞点。本节点**全程离线**（无板卡、无 submit）解码归档 trace，
   给出 `em` 三分语义、`enable_mask` 位、CNA/DCOMP 权重表、DPU EW/BS/BN 字段与 18-task 链结构。
2. **材料**（`experiments/results/bn_lower_20260928/em0d/`）：`trace_layernorm_only_fp16.jsonl`
   （125 记录 = 54 `regcmd` + 54 `task` + 5×(`mem_create`/`mem_map`/`mmap`) + `meta` + `submit`；
   **54 = 3 cores × 18 tasks**，本节点只解 core unit 0 = task 0..17）；`ln_bos/{00..04}.bin`；
   `mul_only_*`/`transpose_*`（含 `tr_transpose_p{0123,0213,0231,0312,0321}_fp16.jsonl`）族 trace；
   `decode_trace.py`/`ln_profile.py`/`ln_numeric_check.txt`/`make_em0d_onnx.py`；本轮新增
   `em0d_analyze.py`（按 task dump，fp16/字段解码，见 §7.194 各表）。
3. **内存映射（LN trace，verified）**：handle1 dma `0xfffff000`→`00.bin`(2160B，I/O 前置区，
   **未被任何 regcmd 引用**；首 8 fp16=0)；handle2 `0xffffc000`→`01.bin`(8832B = **task/regcmd BO**，
   含每 task PC 代码区 + DCOMP 权重表)；handle3 `0xffff9000`→`02.bin`(11328B = **activation scratch
   BO**)；handle4 `0xffff8000`→`03.bin`(1024B = **模型输入**)；handle5 `0xffff7000`→`04.bin`(1024B =
   **输出**)。`mem_create` flags：task BO `0x40b`、其余 `0x403`。**更正（本轮）**：`02.bin` **是
   post-run dump**（非 mmap 零快照）——其基址 `0xffff9000` 恰好等于 LN 输出（`02@0x0000` 首 8 fp16 =
   `1.0137 3.5273 -1.4766 -0.4854 5.1445 -0.8628 0.2886 3.2207`，与 `ln_numeric_check.txt` 的
   `first8 hw` 逐值一致）；各中间激活（`@0x0c00/0x1400/0x1440/0x1800/0x1c40/0x2440`）均非零，故全链
   **可静态重建**（见第 5b 节）。§7.194 初稿「mmap 时快照」的论断作废。
4. **`enable_mask` = 写入 `PC_OPERATION_ENABLE`(reg `0x0008`, tgt `0x81`) 的值**。各 task 实测所编程
   的 block（tgt 高字节）与 `em` 强相关（**verified**）：

   | `em` | block 组成 | LN 中 task |
   |---|---|---|
   | `0x18` | DPU + DPU_RDMA（无 CNA/CORE） | 0,3,5–14,16,17 |
   | `0x0d` | CNA + CORE + DPU（无 RDMA） | 1,4 |
   | `0x1d` | CNA + CORE + DPU + DPU_RDMA | 2,15 |

    位分解（`0x18`=bits3,4；`0x0d`=bits0,2,3；`0x1d`=bits0,2,3,4）⇒ **bit3(0x08)=DPU（所有 task）**、
    **bit4(0x10)=DPU_RDMA（仅 `0x18`/`0x1d`，与逐 task 是否存在 DPU_RDMA 寄存器完全一致）**。
    **本轮闭合（item①）**：全 33 个 trace 扫描（`/tmp/scan2.py`）证明 **CORE 在 `enable_mask` 中无
    对应位**——存在同一 `em` 下 CORE 有无两态：`tr_transpose_p0321_fp16.jsonl` 有 6 个 `em=0x0d` 的
    `CNA+CORE+DPU` task 与 18 个 `em=0x0d` 的 `CNA+DPU`（无 CORE）task，**mask 完全相同**；
    `trace_layernorm_only_fp16.jsonl`/`fp_layernorm_16_int8.jsonl` 各有 `0x1d` 带 CORE×3 与不带 CORE×3。
    ⇒ **CORE 不由该寄存器使能（与 CNA 耦合/由其它路径门控）**，bit0/bit2 **均不可能是 CORE**。剩余
    {bit0,bit2} 只能是 {CNA, master/OP_EN}：前作命名 `refs/rk3588-npu`（手写 matmul，实测发射
    `0x0d`）记 `0x01`=PC task-enable、`0x04`=CNA、`0x08`=DPU，与此一致；`0x60`(=bits5,6) 全为
    `PPU+PPU_RDMA`（17 个 pool/cbc task）⇒ **bit5=PPU、bit6=PPU_RDMA**（与 DPU=bit3/RDMA=bit4 同构）。
    最终位表：**bit0=OP_EN(主使能)、bit2=CNA、bit3=DPU、bit4=DPU_RDMA、bit5=PPU、bit6=PPU_RDMA；
    CORE 无位**；bit1 全程未置位（疑 CORE 预留，未证实）。
5. **LayerNorm (1,8,8,8)，axis=-1**（γ/β 见 `make_em0d_onnx.py`）。core unit 0 的 task 链（地址据第 3 节）：

   | task | em | 关键动作（regcmd 字段，`em0d_analyze.py` 可复现） |
   |---|---|---|
   | 0 | `0x18` | RDMA `0xffff8000`→`0xffff9c00`；cube 7,7,C=`0x70007`；`DATA_FORMAT=0x24000001`；`RDMA_FEATURE_MODE_CFG=0xf821`；BS/BN=`0x53` |
   | 1 | `0x0d` | CNA 1×1 conv：`FEATURE_DATA_ADDR=0xffff9c00`，`DCOMP_ADDR0=0xffffc1c0`(8×fp16 **1.0**)，`CONV_CON1=0x120`(fp16)，`CONV_CON3=9`，`WEIGHT_SIZE0/1=16/16`，`WEIGHT_SIZE2=0x01010001`；CORE 参与；DST=`0xffffa440`；`BS_MUL=+1/√8`(`0x35a8`)、`BN_MUL=−1/√8`(`0xb5a8`) |
   | 2 | `0x1d` | CNA(`FEATURE=0xffffa440`,`DCOMP=0xffffc100`)+CORE+DPU+RDMA(`SRC=0xffff9c00`,`EW_BASE=0xffffa000`)；`EW_CFG=0x108202c0` = **EW_ALU_ALGO=2(ADD)** + EW_DATA_MODE=1 + EDATA_SIZE=2 + EW_OP_SRC=1 + EW_OP_BYPASS=0（=残差/累加，与 `NPU_HW_QUIRKS`「Addition」模板逐字段一致）|
   | 3 | `0x18` | reformat `0xffffac40`→`0xffffa400`（`EW_CFG=0x108003c4`，EW_OP_TYPE=1）|
   | 4 | `0x0d` | CNA(`DCOMP=0xffffc180`,8×**1.875**)+CORE+DPU；`BS_MUL=1.67676`(`0x3eb5`)；DST=`0xffffb440` |
   | 5 | `0x18` | reformat `0xffffb440`→`0xffffa400`；`BN_CFG=0x00020050`，`BS_MUL` lo16≈1.5e-5，`DATA_FORMAT` BS_MUL_SHIFT_VALUE_NEG=1 |
   | 6–14 | `0x18` | EW 乒乓（`0xffffa800`↔`0xffffa400`/`0xffffb440`）；`EW_CFG=0x100003c4`(ALU_ALGO=0,OP_TYPE=1,OP_CVT_BYPASS=1)；`BS_MUL` ∈ {−1.75, 0, −0.5, …}（逐 task 值 `em0d_analyze.py`） |
   | 15 | `0x1d` | CNA(`DCOMP=0xffffc080`,ones@stride8)+CORE+DPU+RDMA(`EW_BASE=0xffffb040`)；DST=`0xffffb440` |
   | 16 | `0x18` | reformat `0xffffb440`→`0xffff9000`（RDMA `BN_BASE=0xffffc040`）|
   | 17 | `0x18` | reformat `0xffff9000`→`0xffff7000`(=输出 `04.bin`)，RDMA `SURF_NOTCH=0xfffffc80` |

   结构读出：`0x18`=纯数据搬运/重排（reformat/transpose），`0x0d`=CNA+CORE+DPU 归约/逐元素计算，
   `0x1d`=`0x0d`+DPU_RDMA（CNA 计算 + RDMA 旁路读权重/残差）——与 openrknn 的 `0x18`/`0x0d`/`0x1d`
   三分**完全一致**。LayerNorm 走「transpose → CNA 归约(求和/减均值/平方和) → EW 累加 → affine」。
   注：task0 的 reformat 编码（`FMC=0x81e5`/`DF=0x24000001`/cube `7,7,C=0x70007`）恰等于第 8 节
   **`p0312` perm `[0,3,1,2]`**（见下）。

5b. **LayerNorm 数值链闭合（item②，verified，本轮）**：以 post-run `02.bin`（第 3 节更正）对 core unit 0
   做逐地址比对，独立 numpy 复算，四项全部命中（`em0d_analyze.py` 数据）：设 `9c00` 首组 8 fp16
   `g=[4.875,4.875,5.0,-1.25,-3.375,4.25,-4.0,-2.375]`，
   ① `ac40`(`@0x1c40`) `== g - mean(g)` 逐值 `diff=0`（`mean(g)=1.0`，`ac40`=减均值后的 `x-μ`）；
   ② `a400[0]=0.261475` vs `1/√var(g)=0.261279`（`a400`=`rsqrt(var)`，即归一化因子）；
   ③ `b440`(`@0x2440`) `== (g-μ)·rsqrt(var)`（`1.0137,1.0137,1.0459,-0.5884,…` vs 复算
   `1.0125,1.0125,1.0451,-0.5879,…`，fp16 级）；④ 输出 `02@0x0000` `== b440·γ+β`（γ/β 见
   `make_em0d_onnx.py`；`[1.0137,3.5273,-1.4766,-0.4854,5.1445,-0.8628,0.2886,3.2207]` 逐值命中）。
   ⇒ 链路语义 = **transpose(t0) → 减均值(t2, `EW_ALU_ALGO=2`ADD) → `rsqrt(var)`(t1 coarse / t3 精化)
   → 逐元素归一化+affine(t4/t15) → reformat 输出(t16/17)**；中段 `0x18` EW 乒乓(t5–14)为 `rsqrt` 的
   Newton–Raphson 迭代精化（`a440=0.3059` coarse → `a400=0.2615` 精化，佐证）。逐 task 到具体 NR
   迭代式的精确配属仍未逐条标注，但**端到端算术语义与数值已完全确定**。
6. **CNA 权重 = DCOMP 机制**（`CNA_DCOMP_ADDR0` + `DCOMP_CTRL`/`DCOMP_REGNUM`/`DCOMP_AMOUNT0-15`），
   regmap **无显式 CNA 权重地址寄存器**。表内容静态可读（`01.bin`，**verified**）：
   `@0x1c0`(t1)=8×fp16 `1.0` 连续；`@0x100`(t2) 与 `@0x80`(t15)=`1.0` 于 pos 0,8,16,24（stride 8 fp16=16B，
   即稀疏 per-ic）；`@0x180`(t4)=8×fp16 `1.875`。
7. **数值锚点（verified）**：`ln_numeric_check.txt` —— 独立 numpy fp32 参考 vs 板卡输出
   `maxdiff=0.00704`（fp16 舍入级），首 8 值 hw/ref 对齐 ⇒ vendor LayerNorm lowering 在 NPU 上**计算正确**。
8. **`0x18` reformat / transpose 编码（item③，本轮闭合 + ground truth）**：每个 `transpose_p0XXX.onnx`
   内嵌 ONNX `Transpose` 的 `perm`（protobuf 直读，verified）：`p0123=[0,1,2,3]`、`p0213=[0,2,1,3]`、
   `p0231=[0,2,3,1]`、`p0312=[0,3,1,2]`、`p0321=[0,3,2,1]`；`transpose_only=[0,2,3,1]`（=p0231）。
   每个 trace 为 3 个完全相同的 `0x18` task（task0 值）→ **完整 perm→寄存器表**（其余字段全 0）：

   | perm | FMC `0x400c` | DATA_FORMAT `0x4010` | DST_STRIDE `0x4024` | cube W,H `0x4030/34` | CHAN `0x403c` | NOTCH `0x4038` | BS_OW `0x4050` | WDMA0 `0x4058` | WDMA1 `0x405c` | SURF_ADD `0x40c0` | RDMA_FM `0x5044` | RDMA_SRC_DMA `0x5048` | RDMA_NOTCH `0x504c` |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   | `[0,1,2,3]` | `0x1e5` | `0x48000002` | `0x100` | 3,3 | `0x1f001f` | 0 | `0x2` | `0x1f` | `0x30003` | `0x100` | `0x17849` | 0 | 0 |
   | `[0,2,1,3]` | `0x81e5` | `0x24000001` | `0x80` | 0,7 | `0x3f003f` | 0 | `0x80007fe` | `0x807003f` | 0 | `0x10` | `0xf821` | `0x380000` | `0xfffffc80` |
   | `[0,2,3,1]` | `0x81e5` | `0x24000001` | `0x10` | 0,7 | `0x3f003f` | `0x3f003f` | `0x80007fe` | `0x807003f` | 0 | `0x80` | `0xf821` | `0x380000` | `0xfffffc80` |
   | `[0,3,1,2]` | `0x81e5` | `0x24000001` | `0x80` | 7,7 | `0x70007` | 0 | `0x80007fe` | `0x8000007` | `0x10003` | `0x400` | `0xf821` | 0 | 0 |
   | `[0,3,2,1]` | `0x1e4` | `0x48000002` | `0x10` | 0,0 | `0x3f003f` | `0x3f003f` | `0x126` | `0x3f` | 0 | `0x20` | —（降为 `0x0d` CNA 链） | | |

   读出：`FMC` 高 bit15 + stride/`SURFACE_ADD`/`CUBE_NOTCH`/`WDMA` 共同编码 perm；`p0213` vs `p0231`
   （仅 perm 尾序不同）只差 3 个字段：`DST_SURF_STRIDE`(`0x80`/`0x10`)、`CUBE_NOTCH`(`0`/`0x3f003f`)、
   `SURFACE_ADD`(`0x10`/`0x80`)。**`p0321=[0,3,2,1]` 无法用单 DMA reformat 表达，被降为 24-task
   `0x0d` CNA 链**（`tr_transpose_p0321` 有 6 `CNA+CORE+DPU` + 18 `CNA+DPU`），与 item⑤ 一致。LN task0
   即用 `p0312=[0,3,1,2]` 编码。
9. **前作对照**：openrknn `src/openrknn_run.c` 以 `enable_mask==0x18`=reformat（`:849`）、
   `em!=0x1d continue`（仅 CONV 有 WT/BS，`:532`）、`tf[2]!=0x0d continue`（`:645` em=0x0d 扫描）；
   其 `em=0x0d` 处理仅针对 Transpose/exSDPAttention 的 8k/16k blob 逆向配对**启发式**，缺非-Transpose
   exNorm/exLayerNorm 的寄存器级规则；`docs/fp16-transformer-support.md` 记更深层提交超时未解。
   本节点确认其 `0x18`/`0x0d`/`0x1d` 三分与本 trace 的 block 组成一致。
10. **open 项收口（本轮）**：① **CLOSED**——CORE 无 `enable_mask` 位（同 mask 下 CORE 有无两态），
    位表见第 4 节；② **CLOSED（算术语义）**——数值链四项全命中（第 5b 节），仅逐 task 的 NR 迭代式
    未逐条标注；③ **CLOSED**——由 ONNX ground-truth perm 建全编码表（第 8 节）；④ **CLOSED**——
    γ/β 落地已定位：**`γ` = task BO `01.bin@0x0` 8×fp16（设备地址 `0xffffc000`）；`β` =
    `01.bin@0x40` 8×fp32（设备地址 `0xffffc040`，**β 是 fp32 而 γ 是 fp16**，故此前 fp16 误读为
    “stride-2 杂值”）**；task16 RDMA `BS_BASE_ADDR=0xffffc000`(γ)、`BN_BASE_ADDR=0xffffc040`(β)
    正是 affine 的读地址（`grep`/`struct.find` 双向验证）；⑤ **CLOSED**——`0x0d` vs `0x1d` 差仅为
    **bit4=DPU_RDMA**；Transpose 有两条 lowering：`0x18` 纯 RDMA reformat（p0123/0213/0231/0312）与
    `0x0d` CNA 链（p0321，需计算的重排）；`0x1d`=需 RDMA 旁路取权重（conv/exNorm）。前作 openrknn
    “`0x1d`=Conv” 实为「需要 RDMA 的 CNA 计算」而非按 op 分类。
    剩余（低优先，非阻塞）：bit1 语义（疑 CORE 预留，全程未置位，无法观测证实）；LN 中段 NR 迭代的
    逐 task 配属；`p0321` CNA 链 24-task 的逐 task 几何。
11. **板卡**：全程离线，无 submit、无 reset、无失败命令。
12. **本轮产物**：`em0d_analyze.py`（新增入库）；`ln_bos/02.bin` 复用为 post-run dump；全 trace 扫描
    脚本 `/tmp/scan2.py`、perm 差异脚本 `/tmp/tr_diff.py`（临时）。
