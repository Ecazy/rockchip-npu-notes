# Tiny ViT 优化总报告与最终交付验收

> **Package identity update (2026-10-27, template-gap stage):** the current
> compiled package is no longer the 2026-10-26 identity listed below. Current
> `models/compiled/manifest.json` SHA-256 is
> `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`;
> current `models/numeric_acceptance_packed.json` SHA-256 is
> `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`.
> The final board runtime for this stage is
> `19b5263a137cda82eff215ca181d267923ca9645d4cca2378de290ab6dbcaa16`.
> The hashes and tables below remain a historical record of the 10/26
> milestone. See [TEMPLATE-GRANULARITY-REPORT.md](TEMPLATE-GRANULARITY-REPORT.md)
> for the current artifact identity and the MLP upload-coalescing/campaign
> evidence.

日期：2026-10-26。本文汇总当前默认路径、已采用/已撤回方案、最终完整模型
验收结果和明确的非声明。各优化包的实现细节仍以
[OPTIMIZATION-RESULTS-1.md](archive/optimization-history.md#round-1) 至
[OPTIMIZATION-RESULTS-7.md](archive/optimization-history.md#round-7) 为准；本文不把历次局部
campaign 的加速比串接成累计数字。

## 0. 最终身份与入口

- 默认运行时源码：`60f4a82`（mapped QKV attention input）。
- 最终交付 harness：`8cb57d5`；最终 campaign 由 `96cb0dc` 的 harness 发起，
  之后只修正了本地分析聚合器。
- 优化前基线：`42f870f`，即首个优化提交 `d526ef3` 之前的源码树
  （tree `cf2b7f2d75940ddec51d283d318d90306ce49dac`）。
- 默认 Tiny ViT 清单：`models/compiled/tiny_vit_bN.rkhybrid.json`；
  默认布局为 `mapped_qkv`。
- 显式对照：
  - `tiny_vit_bN_splitheads.rkhybrid.json`：保留原 q/k/v 物化路径；
  - `tiny_vit_bN_deviceffn.rkhybrid.json`：独立设备 FFN 数值政策，非默认；
  - `tiny_vit_bN_hostffn.rkhybrid.json`、`tiny_vit_bN_rmsnorm.rkhybrid.json`：
    历史/策略对照，非默认。
- 编译、部署、推理和重放入口：
  - Host build/test：`scripts/run-verification.sh`（或 Host 最小命令见
    [REPLAY.md](REPLAY.md)）；
  - 板端完整验收：`RK_FINAL_LIMIT=10000 RK_FINAL_ROUNDS=3 ./scripts/run-final-delivery.sh`；
  - 单次模型推理：`tools/evaluate_models.py --backend board ...`，
    默认使用 `tiny_vit_bN`；
  - 冻结数值重放：`RK_FROZEN_SPEC=models/numeric_acceptance_packed.json
    RK_REOBSERVE_CALIBRATION=1 ./scripts/run-numeric-replay.sh`
    （或最终 campaign 内已执行的 calibration/acceptance observation）。
- 固定模型包 SHA-256：
  - `models/compiled/manifest.json`：
    `72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`
  - `models/mnist_mlp.npz`：
    `c7cdd120350625ddf39decfde71c94eaed58e60144230ff29dfe7e4d7690c98e`
  - `models/tiny_vit.npz`：
    `0190412619e2603853e52c308fc06caa90bf54f92c15caa0ad3ea4e27e37000f`
  - `numeric_acceptance_packed.json`：
    `f49d876b724b07ea417472ea1ac0c2b271c904e2cd6c7081e9a4c87cb3ddbe66`

## 1. 默认路径

Tiny ViT 默认：

```text
packed gate/up GEMM
  + exact Host FP32 SwiGLU
  + QK/PV static-repeat shared group / chain_shared
  + coalesced upload sync
  + fixed-layout 16x16 fast pack
  + Python ctypes binding descriptor reuse
  + mapped QKV attention input
```

普通执行不运行 `split_heads`；q/k/v 只在 `capture=True` 的诊断路径由
capture-only Host action 重新物化。B1/separate QK/PV 比较必须使用
`_splitheads` 清单，不能把布局映射收益混入提交策略归因。

| 路径 | 入口/清单 | 用途 |
|---|---|---|
| 默认 | `tiny_vit_bN`，不 capture | 部署与性能验收 |
| 诊断 | `capture=True` | 物化 q/k/v、layer trace、数值观察 |
| 布局对照 | `tiny_vit_bN_splitheads` | 原 q/k/v 物化路径；B1/separate 必须用它 |
| 独立数值政策 | `tiny_vit_bN_deviceffn` | 显式 opt-in，不是默认 |
| 历史策略对照 | `tiny_vit_bN_hostffn`、`tiny_vit_bN_rmsnorm` | 只用于回归/归因，不作为默认 |

## 2. 完整模型最终验收

`results/20261026T120625Z-final/`：同一 campaign、同一 payload、同一 core 1、
CPU affinity 6、同一 batch/tail 调度，三个逐轮交替 round；**五个静态 profile
全部使用完整 10,000 张 MNIST test split**。

| profile | accuracy | top-1 agreement | declared top-1 | packed task-logit |
|---|---:|---:|---:|---|
| mnist_mlp b32 | 0.98310 | 1.00000 | 1.00000 | PASS |
| mnist_mlp b8 | 0.98310 | 1.00000 | 1.00000 | PASS |
| mnist_mlp b1 | 0.98310 | 1.00000 | 1.00000 | PASS |
| tiny_vit b8 | 0.98210 | 1.00000 | 1.00000 | PASS |
| tiny_vit b1 | 0.98210 | 1.00000 | 1.00000 | PASS |

- 原模型阈值未改：MLP ≥ 0.97、Tiny ViT ≥ 0.95、accuracy drop ≤ 0.005、
  top-1 agreement ≥ 0.99。五个 profile 的 accuracy_delta 均为 **0.0**。
- 独立分析器从 campaign 权重/数据和原始 logits 重新计算 CPU/declared 参考，
  `analysis.json`：**PASS / 0 problems**。
- 当前 build 的 packed 冻结政策重新观察：
  `verify-packed-summary.json` **PASS / 0 problems**；33 个 layer gate 和
  5 个 profile logit gate 全部通过，未重新校准、未放宽阈值。
- baseline/current 三个 round 的全部原始 logits 逐位相同；max abs delta = 0。

## 3. 优化前基线 vs 当前默认

同 campaign、同一数据/权重/batch-tail/核/频率/计时边界；baseline =
`42f870f`，current = 当前默认。warm latency 是每轮 `model_call_seconds` 除以
调度 call 数后的三轮中位；吞吐由 warm latency 换算。

| profile | baseline warm | current warm | speedup | baseline img/s | current img/s |
|---|---:|---:|---:|---:|---:|
| mnist_mlp b32 | 2.881 ms | **2.246 ms** | **1.283×** | 11,056 | **14,181** |
| mnist_mlp b8 | 2.027 ms | **1.559 ms** | **1.300×** | 3,946 | **5,132** |
| mnist_mlp b1 | 2.031 ms | **1.373 ms** | **1.479×** | 492 | **728** |
| tiny_vit b8 | 34.631 ms | **10.603 ms** | **3.266×** | 231 | **755** |
| tiny_vit b1 | 7.138 ms | **3.329 ms** | **2.144×** | 140 | **300** |

完整五 profile × 10,000 张评估的进程墙钟（包含参考计算、trace 和所有 profile）：

- baseline：**224.214 s**
- current：**148.204 s**
- 比值：**1.513×**

创建成本（`runner_setup_seconds`，一次进程内 runner/plan workspace 创建）：

| profile | baseline | current |
|---|---:|---:|
| mnist_mlp b32 | 36.5 ms | 37.2 ms |
| mnist_mlp b8 | 17.5 ms | 18.0 ms |
| mnist_mlp b1 | 18.4 ms | 18.7 ms |
| tiny_vit b8 | 18.0 ms | **62.2 ms** |
| tiny_vit b1 | 18.2 ms | **25.1 ms** |

Tiny ViT 创建成本上升来自默认路径的更多 PlanCall/描述符预构造成本；它是
一次性成本，b8 在首次调用后即被每调用约 24 ms 的稳态收益摊平。该数字明确
保留，不把“warm 加速”表述成冷启动加速。

结构变化：

| profile | baseline task/submit | current task/submit | baseline up/down calls | current up/down calls |
|---|---:|---:|---:|---:|
| mnist_mlp b32 | 3,454 / 2,512 | 3,454 / 2,512 | 1,256 / 314 | 1,256 / 314 |
| mnist_mlp b8 | 13,750 / 10,000 | 13,750 / 10,000 | 5,000 / 1,250 | 5,000 / 1,250 |
| mnist_mlp b1 | 110,000 / 80,000 | 110,000 / 80,000 | 40,000 / 10,000 | 40,000 / 10,000 |
| tiny_vit b8 | 88,750 / **88,750** | 87,500 / **10,000** | 168,750 / 88,750 | 167,500 / 87,500 |
| tiny_vit b1 | 150,000 / **150,000** | 140,000 / **80,000** | 230,000 / 150,000 | 220,000 / 140,000 |

Tiny ViT b8 上传调用从 168,750 降至 167,500、上传字节 391.04 MB → 370.56 MB，
下载调用 88,750 → 87,500、下载字节保持 287.68 MB；b1 上传字节 404.48 MB →
384.00 MB，下载字节保持 294.40 MB。MLP 结构不变。

## 4. 已采用方案

1. C workspace 包络归因、静态解析/预分配、Host scratch 预算闭环；
2. 下载路径消除 per-call malloc/free、有限值扫描和拷贝特化；
3. guard slack 的精确 8-byte 扫描；
4. QK/PV 静态 repeat shared group / `chain_shared`；
5. packed gate/up GEMM + 原 FP32 Host SwiGLU；
6. shared BO 上传同步合并；
7. fixed-layout 16×16 CUBE/WEIGHT fast pack；
8. 上传策略创建时快照与 >128 上传安全回退；
9. `PlanCall` ctypes binding 描述符复用；
10. 非阻塞 descriptor lock，保持 `RK_E_BUSY`；
11. mapped QKV attention input + capture-only split_heads；
12. C/Python finite 检查同 buffer 去重。

## 5. 已撤回/未采用

- 全设备 FFN：独立数值政策（部分容差需从 0.0625 放宽到 0.125），只保留
  `_deviceffn` 显式清单；
- 复用 `co_row_ops` 的 RMSNorm：实测比 Host RMSNorm + 相邻上传慢约 2.8–3.5×；
- fused residual + RMSNorm Host action：无实测收益，默认不采用；
- 下载同步合并：端到端无稳定收益，已撤回；
- residual + reshape + mean 尾部融合：b8 累计 campaign 与单独 campaign 方向
  矛盾，已撤回；
- prepared ctypes binding arrays：b8 无收益，已撤回；
- 通用任意 stride 输入：未实现，mapped QKV 只接受显式 row/column slice 与
  transpose 关系。

## 6. 明确非声明

- 不宣称 mapped QKV 在 b8 上每轮稳定加速；其价值是实际消除 split_heads
  和 q/k/v Host 数组，b8 pooled 接近噪声边界。
- 不把本文的 3.266× / 2.144× 与历史局部 campaign 的加速比相加或串接；
  它们是同一 campaign 的 baseline/current 端到端比较。
- 不宣称冷启动更快；Tiny ViT runner 创建成本当前高于基线。
- 不宣称 device FFN、通用 Softmax/RMSNorm 下沉、任意 stride、多核并发、
  KV cache 或未知故障恢复已支持。
- 不宣称厂商 RKNN 路径与当前 packed per-logit 阈值等价；ViT 误差已超过
  0.0625，只能按分类保持性单独描述。
- 三层核心串行准入、数值冻结阈值和原模型里程碑条件均未被重新定义。

## 7. 厂商 RKNN 基线与三类对照（2026-10-27）

新增厂商基线 campaign：`results/20261027T015302Z-vendor/`。同一 board、
同一 10,000 张 test split、同一 weights、同一 batch/tail 调度、NPU core 1、
CPU affinity 6、单线程 BLAS。厂商路径使用 RKNN-Toolkit2 **2.3.2** 编译，
RKNN Runtime **2.3.2** 执行，非量化 FP16、`single_core_mode=True`、runtime
core 0；ONNX 图输入 raw `[B,784]`，归一化和 patchify 在厂商图内完成。
当前自研路径和 CPU 参考在同一 vendor campaign 中重跑两轮。

| profile | 优化前基线* | 当前自研 | 厂商 RKNN | CPU FP32 | 厂商 speedup | CPU speedup |
|---|---:|---:|---:|---:|---:|---:|
| mlp b32 | 2.881 ms | 1.968 ms | **0.819 ms** | 6.838 ms | **2.40×** | 0.29× |
| mlp b8 | 2.027 ms | 1.366 ms | **0.537 ms** | 1.752 ms | **2.54×** | 0.78× |
| mlp b1 | 2.031 ms | 1.235 ms | **0.432 ms** | **0.254 ms** | **2.86×** | **4.86×** |
| vit b8 | 34.631 ms | 10.421 ms | **4.120 ms** | 5.867 ms | **2.53×** | 1.78× |
| vit b1 | 7.138 ms | 3.065 ms | **1.018 ms** | **0.872 ms** | **3.01×** | **3.51×** |

speedup 定义为 `当前 warm ms / arm warm ms`；>1 表示该 arm 更快。
\* 优化前基线来自此前完整模型 campaign
`results/20261026T120625Z-final/`；其余三列来自本 vendor campaign，不能把
星号列与本 campaign 的绝对值逐项串接。

完整五 profile × 10,000 张评估进程墙钟（包含参考计算/trace，不是纯应用侧）：

| arm | wall |
|---|---:|
| current | 143.777 s |
| vendor | 106.519 s |
| CPU | 85.009 s |

厂商/当前 warm 差异明显大于此前的局部布局优化：厂商在每个 batch chunk 只有
一次 RKNN runtime 调用；当前默认仍要多次 submit 和多个 Host action。CPU 在
MLP b1 和 ViT b1 上比当前自研 NPU 快，说明“上 NPU”本身并不是 b1 的充分收益
条件。这里比较的是应用侧 raw→preprocess→inference→argmax；CPU arm 的
`setup` 不包含等价的 plan/compiler 创建。

精度：

- 厂商与 CPU 的 accuracy/top-1 在五个 profile 全部一致：MLP 0.98310、
  Tiny ViT 0.98210、top-1 agreement 1.00000。
- 厂商最大 logit 误差：MLP 0.03049；ViT b8 0.12277、b1 0.10715。
- packed task-logit 阈值是 0.0625：MLP 在阈值内，ViT **超出**。因此厂商路径
  是 classification-preserving，但不是 packed per-logit 等价路径；若部署
  RKNN，必须建立独立 numeric policy，不能把它当作当前冻结门限的替代。
- `results/20261027T015302Z-vendor/vendor-perf/` 的逐层 placement 探针显示：
  MLP 有 6–7 个 NPU 层 + 3 个 CPU 边界层；Tiny ViT b8 有 50 个 NPU 层 +
  5 个 CPU 层。Tiny ViT 的 attention 使用 `exSDPAttention`，FFN 使用
  `ConvExSwish`/`ConvMul`，RMSNorm 仅剩两个 CPU `Div`；也就是厂商把大量
  Host action 边界吸收进了设备图。这解释了 warm gap 主要不是 QKV 单段搬运。
  探针开启 `COLLECT_PERF_MASK`，其绝对时间只用于相对排序，不作为 benchmark。

## 8. 下一阶段最值得投入的一项

最终成本仍显示 per-call b8 plan workspace 约 **4.71 ms**、Host actions 约
**2.13 ms**、executor 约 **0.68 ms**；但厂商 baseline 已证明，仅做
QKV/QK/PV 的局部 device handoff 无法解释当前与厂商约 **2.4–3.0×** 的
warm gap。当前默认仍是多个 submit、多个 Host action；厂商路径是编译器生成的
整图 runtime 调用。因此**暂停把 device-resident QKV 作为独立下一项**；它是
未来整图驻留方案的一个子组件，不是足够的首要方案。

下一阶段最值得投入的是 **whole-model device-resident graph / Host-boundary
collapse**：

- 目标：把 QKV→attention→FFN 的中间结果保持在 runtime 管理的设备 BO 中，
  只有模型输入和最终 logits 跨 Host 边界；减少每次调用的 submit/Host action
  数量，而不是只优化某一段 QKV 搬运。
- 契约变化（先设计/探针，不直接实施）：
  - 版本化 graph-level plan/recipe，声明跨节点的 device BO 生命周期和
    producer/consumer 关系；
  - C 独立检查 owner、domain、容量、alias/重叠、offset/stride、整数溢出和
    quarantine；仍不开放裸 IOVA，不支持异步资源生命周期；
  - 若包含 device RMSNorm/Softmax/SwiGLU，必须新增独立数值契约、冻结阈值和
    calibration/acceptance evidence；不能继承当前 packed Host 阈值；
  - 保留当前 mapped QKV 和 Host 路径作为 control，便于逐位/精度归因。
- 不预设能追平厂商；厂商 ViT 的 per-logit 误差已超出 packed 阈值，说明其
  图融合也改变了数值语义，必须单独验收。

## 9. 证据索引

- 最终 campaign：`results/20261026T120625Z-final/`
  - `analysis.json`、`analysis.md`
  - `verify-packed-summary.json`
  - `README.md`（身份、哈希、重放命令）
- 完整 MNIST：五个 profile 各 10,000 张，3 个逐轮交替 round；
- 原始方案与第 1–7 轮过程：[优化历史](archive/optimization-history.md)；
- 厂商 baseline：`results/20261027T015302Z-vendor/`
  - `analysis.json`、`analysis.md`
  - `model-results/rN-{current,vendor,cpu}/`
  - `vendor/README.md`（RKNN 编译与板端重放命令、runtime/wheel SHA）
- 当前原始证据索引：`results/README.md`。
