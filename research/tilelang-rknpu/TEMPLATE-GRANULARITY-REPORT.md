# 硬件模板与执行粒度差距报告

> **Program-cache follow-up identity:** a later default-on creation-time
> `co_program` cache changed the adopted runtime to
> `cc4586eb4d51edf3b7cbab5d9fffd2fdc0d40e87b24ee9f052318108eb87cdc9`.
> See [PROGRAM-CACHE-REPORT.md](PROGRAM-CACHE-REPORT.md).

> **Prepared-validation follow-up identity:** a later default-off experiment
> (`ca6481b`) changed the runtime identity to
> `18dac5981ee05ade1d7d45108dc4bfcc8fbcc0d19187d9b471c56923c27fdc4d`.
> The `19b526...` hash and tables below are the historical template-stage
> artifact. The static-validation cache was not productized; see
> [RUNTIME-LAYER-PREPARED-VALIDATION.md](RUNTIME-LAYER-PREPARED-VALIDATION.md).

日期：2026-10-27。阶段目标是回答：当前差距中，矩阵模板本身、提交、布局和
Host/C 边界各占多少；然后只选一个证据最强的可实施改进，不回到 Python 编排
微优化，也不启动通用图级 BO 或整模型设备驻留平台。

机器可读证据、工具和重放命令见
[results/20261027T093259Z-template-gap](results/20261027T093259Z-template-gap/README.md)。

## 0. 结论

本阶段交付并采用两个**不改变数值政策、不新增硬件配方、不新增图级公共接口**
的 CUBE 布局桥接快路径，作为一个 layout-bridge 改动包：

1. **K-major 外部 CUBE 值只打包一次**：Host 上传完整的 256-row 打包值；
   各 256-column K slice 用稳定 view 读取。C validator 对 canonical offset
   做覆盖证明，不靠“看起来像 slice”。
2. **CUBE→row-major 下载按 8 元素 atom 复制**：每个物理 atom 一次 16-byte
   `memcpy`，不再对每个元素求 `co_index`；逻辑 rows/cols 和 padding 语义不变。

这两项直接减少了已测出的 upload/guard/download 边界成本：

- MLP b1/b8/b32 plan call：**1.065× / 1.062× / 1.041×**；
- MLP 最终交付端到端：**1.045× / 1.040× / 1.034×**；
- Tiny ViT b8/b1 最终交付端到端：**1.134× / 1.041×**；
- 全部全量 logits **逐位相同**，冻结 packed numeric spec **PASS / 0 problems**。

**本轮没有选择新的 GEMM register/tiling 模板。** 实测表明，对当前 M/N/K 和
固定 256 宽 K-chain，设备 submit+wait 只是 total/application 的一部分；在小
M、b8 ViT projection 和 QK/PV shared group 上，C 侧 validation、guard、
pack、download 以及固定任务/提交包络占比与模板本身相当或更大。没有证据支持
把“再换一个模板寄存器配方”作为下一项最高收益投资。

下一项需要决策的能力投资应是**workspace 级 validation/guard/publication
批处理契约**：对静态 binding/alias/lifetime 只证明一次，每次 run 只校验可变
指针和 finite 值，并降低 32 KiB slack 扫描及每 slot prep/fini 成本，同时保留
明确 failure isolation。该候选需要新的安全/公共契约和证明；本阶段只提交证据，
未实现。

## 1. 测量条件与案例选择

固定条件：

| 项 | 值 |
|---|---|
| board | RK3588 OPi 5 Plus |
| kernel / driver | `6.1.43-rockchip-rk3588-rknpu098` / `v0.9.8` |
| NPU | 1 core，governor `performance`，观测 1.0 GHz |
| DMC | governor `performance`，2.112 GHz |
| CPU | affinity 6，governor `performance`，观测 2.304 GHz |
| 数值 | 当前 packed FP16 政策；CPU FP32 / vendor 分列 |
| 应用计时 | prepared binding + reused workspace/output |
| 统计插桩 | 单独 attributed pass，不并入应用主值 |

选择案例不是全形状穷举，而是覆盖能区分瓶颈的粒度：

- MLP b1/b8/b32：完整 MLP 中间结果设备驻留，8 submits/11 tasks；
- ViT QKV/gate-up/down：`M=B×16`，K=64/128，N=192/256/64；
- QK/PV shared group：每 head `16×16×16`，b1 4 tasks / b8 32 tasks，均 1
  submit；
- classifier：M=B，K=64，N=16，代表极小 tile；
- 纯矩阵参考：CPU NumPy FP32，不同数值政策，不做同政策排名。

几何和 source tiling 逐项记录在
[`geometry.md`](results/20261027T093259Z-template-gap/geometry.md) 和同名 JSON。

## 2. 模板本身：tiling、任务展开、寄存器与布局

当前 `co.mm.f16.v1`/`co.mm_acc.f16.v1` 的固定事实：

- `gen_matmul_fp16` 一次发射一个 CNA/CORE/DPU regcmd task；
- 单任务 M/N 上限 256；K 以 256 为一 slice，K>256 形成有序
  `MM` + 多个 `MM_ACC` chain；
- CUBE physical rows 为 `max(M,12)` 向上取 4，physical cols 为 32 对齐；
- source TileLang 的 4×N64 输出 tile 会在当前 lowering 中合并成一个
  N256 设备任务；
- RKPlan 没有可选的 register budget、bank recipe、M=1/2/4 物理行模板或
  非 256 K-chain 的等价选择。

代表性几何：

| case | 逻辑 MMAC | 物理 MMAC | padding | tasks/submits |
|---|---:|---:|---:|---:|
| MLP b1 | 0.297 | 3.588 | 12.083× | 11/8 |
| MLP b8 | 2.376 | 3.588 | 1.510× | 11/8 |
| MLP b32 | 9.503 | 9.568 | 1.007× | 11/8 |
| QKV b8 | 1.573 | 1.573 | 1.000× | 1/1 |
| gate-up b8 | 2.097 | 2.097 | 1.000× | 1/1 |
| down b8 | 1.049 | 1.049 | 1.000× | 1/1 |
| QK group b8 | 0.131 | 0.524 | 4.000× | 32/1 |
| classifier b8 | 0.008 | 0.025 | 3.000× | 1/1 |

MLP b1 和 b8 的物理 MMAC、任务数和提交数完全相同：b1 的 1 行被物理 12 行
模板覆盖，因此它支付与 b8 几乎相同的设备任务成本。这是当前模板粒度的事实，
不是 Host 边界解释。

QK/PV 已经把 b1 的 4 个、b8 的 32 个独立 head task 合成一个 submit，任务
展开本身有效；但下面的 envelope 表明 32 个 16³ 小任务的设备 wait 只有约
98 µs。

## 3. 差距分解

完整表在
[`analysis.md`](results/20261027T093259Z-template-gap/analysis.md) 的
selected-case envelope；其中：

- `submit+wait` 是 C 的阻塞 submit 区间，**包含提交 API 和设备执行**，不是
  纯 kernel 计时；
- `C non-submit` 是 `total_us - submit_wait_us`；
- `binding delta` 是应用 wall 减去 C attributed total，包含 prepared
  binding/Python 外侧，可能因 clock 探针噪声出现小幅负数；
- physical MACs 来自编译后的 CUBE/WEIGHT 形状。

关键读数（最终 fast-download runtime）：

| case | app µs | submit+wait | share | guard | pack | download | validate |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 440.3 | 176.7 | 40.1% | 159.6 | 11.1 | 2.4 | 2.4 |
| MLP b8 | 491.9 | 178.7 | 36.3% | 159.2 | 42.8 | 2.7 | 17.3 |
| MLP b32 | 673.2 | 210.6 | 31.3% | 163.4 | 157.0 | 3.8 | 68.3 |
| QKV b8 | 234.6 | 87.3 | 37.2% | 26.3 | 43.6 | 35.2 | 17.2 |
| gate-up b8 | 254.9 | 99.6 | 39.1% | 27.3 | 43.6 | 40.6 | 17.3 |
| down b8 | 221.1 | 63.7 | 28.8% | 24.4 | 81.6 | 12.4 | 34.2 |
| classifier b8 | 99.7 | 49.2 | 49.3% | 21.6 | 6.1 | 2.7 | 1.3 |
| QK group b8 | 920.5 | 98.0 | 10.7% | 26.3 | 42.1 | 122.9 | 424.1 |
| PV group b8 | 723.9 | 96.6 | 13.4% | 25.9 | 46.8 | 122.6 | 237.8 |

这里最重要的不是某一个百分比，而是不同执行粒度给出不同事实：

- **MLP 对照**：没有层间 Host 边界，矩阵 submit+wait 占 b1 40.1%、b8 36.3%，
  但 guard 占 36.2%/32.4%，pack 在 b32 已达 23.3%。b1/b8 的物理工作量相同，
  说明小 M 固定模板和 8 submit/guard 包络比“更多矩阵任务”更相关。
- **ViT projection/FFN**：submit share 约 29–39%；download、pack、validate、
  guard 合计在同一量级。尤其 gate-up/down 的 output conversion 与 input
  packing 不是小项。
- **QK/PV shared group**：32 个 16³ task 的 submit+wait 仅约 98 µs，占
  app 10.7%；per-call C validation 424.1 µs、download 122.9 µs，才是该案例
  的主要成本。把 32 个 head 合成一个 submit 已经有效，继续只优化矩阵模板
  不会命中这个包络。

因此，当前差距**不能**全部归因于矩阵模板。更准确的阶段性结论是：

- 对 b32 MLP 这种 piece 较大、physical padding 低的 shape，模板吞吐是主要
  可观察部分之一（物理吞吐代理约 45–51 GMAC/s，仍含 submit）；
- 对 b1/b8、ViT projection 和 QK/PV group，固定任务/提交、C validation、
  guard、pack/download 的包络至少同等重要；
- 现有证据不足以把“新寄存器/tiling 模板”列为下一项最高收益投资；反而
  支持先处理 workspace validation/guard/publication 的公共契约。

## 4. 已实现改进一：K-major 外部 CUBE 单次打包

### 规则

原实现把一个 rank-2 外部 CUBE 值的每个 256-column tile 分别放进独立 BO、
独立 upload，再让 MM K-chain 读取不同 slice。新实现：

- 每个 256-row tile 分配一个 canonical CUBE buffer；
- 上传一次完整 K-major value；
- 为每个 256-column K slice 产生稳定的 offset view；
- C validator 增加有序覆盖证明：upload writer 必须具有相同 physical_rows、
  CUBE atom 对齐、canonical byte offset，并完整覆盖 reader 的物理范围；
- transpose/broadcast upload 不适用；device writer 仍要求 identical
  addressing。

没有改变 template、MM/MM_ACC 顺序、数值 narrowed 点或 task/submit 数。

### 结果

| MLP | plan old µs | plan new µs | plan gain | upload-only full gain | delivered full gain |
|---|---:|---:|---:|---:|---:|
| b1 | 509.2 | 478.0 | 1.065× | 1.033× | 1.045× |
| b8 | 573.7 | 540.5 | 1.062× | 1.037× | 1.040× |
| b32 | 806.2 | 774.4 | 1.041× | 1.024× | 1.034× |

资源：

- slots `22→19`；transfers `19→16`；upload actions `4→1`；
- reserved bytes `1,551,488→1,452,992`（b1/b8）；
- `1,660,288→1,561,792`（b32）；
- peak payload、upload bytes、constant bytes 不变；
- 全 10,000 张 MLP logits 逐位相同。

增益来自 upload prep/fini、clear/pack 调用和 publication-gate slot 扫描的
减少，不是矩阵指令加速。

## 5. 已实现改进二：CUBE→row-major 下载 fast path

原 `co_download` 对每个逻辑元素求一次 `co_index`。新路径利用 canonical
CUBE 的 8-FP16 atom：每个 atom 连续 16 byte，按行/atom 复制到 row-major
目标；padding 从不复制，逻辑 rows/cols 完全不变。该路径对 `CUBE` + FP16 +
atom8 生效，其他 layout/type 继续走原循环。

### 结果

| case | download before µs | download after µs | saved |
|---|---:|---:|---:|
| QKV b8 | 164.0 | 35.2 | 128.8 |
| gate-up b8 | 198.5 | 40.6 | 157.9 |
| down b8 | 52.8 | 12.4 | 40.4 |
| QK group b8 | 168.0 | 122.9 | 45.1 |
| PV group b8 | 153.6 | 122.6 | 30.9 |

整模型同 package、同固定频率 A/B：

| ViT | before µs | after µs | paired gain | bitwise |
|---|---:|---:|---:|---|
| b8 | 4512.9 | 4010.7 | **1.134×** | yes |
| b1 | 1593.4 | 1532.4 | **1.041×** | yes |

MLP 的最终输出只有很小 download，因此该改动对 MLP 基本中性；upload
coalescing 才是 MLP 的主增益。QK/PV 的 download 仍包含 32 个输出的 staging
和 C 包络，fast path 只消除了其中逐元素寻址部分，不能把剩余 download 时间
宣称为矩阵模板成本。

## 6. setup、常量预打包与端到端成本

`analysis.md` 的 setup 表同时给出：

- Host constant materialization；
- C plan/workspace creation；
- 一次性 prepared binding 构造；
- 复用 workspace 的 app；
- fresh output allocation；
- cold once 与按 1,000 次摊销值。

MLP 约 743–791 KiB 常数，Host materialization 约 1.78–1.81 ms，plan/workspace
setup 约 5.3–5.6 ms，cold once 约 7.4–7.9 ms；稳定调用为 0.44–0.67 ms。
QK/PV b8 的一次性 group binding 约 0.97 ms、workspace setup约 4.2 ms，steady
call约 0.72–0.92 ms。这些成本必须和 steady kernel 数字分开，不能拿 warm
call 代表冷启动。

## 7. 外部参考与边界

固定频率下：

| profile | final delivered µs | vendor µs | CPU FP32 µs | vendor max logit err |
|---|---:|---:|---:|---:|
| MLP b1 | 599.1 | 258.8 | 252.6 | 0.03049 |
| MLP b8 | 716.6 | 309.9 | 1748.9 | 0.03049 |
| MLP b32 | 1127.6 | 538.0 | 6830.5 | 0.03049 |
| ViT b8 | 4010.7 | 1582.1 | 5856.7 | 0.12277 |
| ViT b1 | 1532.4 | 589.4 | 869.5 | 0.10715 |

- vendor 是 non-quantized FP16 graph fusion，和 packed F16 不是同一数值政策；
  ViT 最大误差超过 packed 0.0625；
- CPU 是 NumPy FP32；
- 因此这三列不是同政策排名。vendor 的 `rknn_run` 是 Host runtime 调用，
  内部 device Submit 数、per-kernel 时间和 per-layer 计时均不可得；
- 旧 vendor-perf 探针只保留 placement/ranking：MLP b8 7 NPU + 3 CPU 边界层，
  ViT b8 50 NPU + 5 CPU 边界层；其 `COLLECT_PERF_MASK` 时间不作为 warm
  时延。

外部对比给出的不是“新模板已追平”的结论：差距仍在，尤其 b1 和 vendor；
但已证明本轮收益来自可解释的边界/布局路径，且没有引入新数值口径。

## 8. 验收、身份与决策

- Host：**348 passed, 3 skipped, 68 subtests**；
- CTest：**31/31**；
- 冻结 packed numeric spec：**PASS / 0 problems**；
- MLP b1/b8/b32 全 10k 最终交付与旧路径 logits **逐位相同**；
- ViT b8/b1 全 10k fast-download A/B logits **逐位相同**；
- 最终 package `models/compiled/manifest.json`：
  `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`；
- `models/numeric_acceptance_packed.json`：
  `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`；
- final board runtime `librknpu_plan.so`：
  `19b5263a137cda82eff215ca181d267923ca9645d4cca2378de290ab6dbcaa16`；
- Device FFN 保持显式可选，未因任何本轮数据自动切换独立数值政策。

**产品化选择**：保留并默认使用上述 CUBE layout bridge；不新增硬件模板或公开
数值政策。

**下一项能力投资决策候选**：workspace 级 validation/guard/publication
批处理。其证据是 QK/PV b8 的 per-call validation 约 424/238 µs、MLP b1/b8
guard 约 159 µs，均显著高于若干矩阵任务的边际收益；其缺口是静态 binding/
alias/lifetime 证明复用、guard 覆盖与 cache sync 的公共安全契约、以及
failure-isolation 回归。未获该契约批准前不实现，也不扩大默认支持域。

## 9. 非声明

- 不声称 `submit+wait` 是纯矩阵 kernel 时间；它包含提交 API 和设备执行。
- 不声称 vendor 内部 Submit 数量、per-kernel 时间或 compiler fusion 已知。
- 不声称 CPU FP32、vendor FP16 与 packed F16 可以同政策排名。
- 不声称 32 KiB guard scan、prep/fini 或 download staging 可以无契约关闭。
- 不声称 b1 的小 M 物理 padding 能靠当前 RKPlan 规划消除；它需要新硬件
  模板或新的批量执行语义。
- 不声称本轮的 layout fast path 能修复 viT Vendor gap 或 MLP b1 CPU gap。
