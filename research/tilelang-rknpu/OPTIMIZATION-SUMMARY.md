# Tiny ViT 优化总报告与最终交付验收

日期：2026-10-26。本文汇总当前默认路径、已采用/已撤回方案、最终完整模型
验收结果和明确的非声明。各优化包的实现细节仍以
[OPTIMIZATION-RESULTS-1.md](OPTIMIZATION-RESULTS-1.md) 至
[OPTIMIZATION-RESULTS-7.md](OPTIMIZATION-RESULTS-7.md) 为准；本文不把历次局部
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
- 三层核心串行准入、数值冻结阈值和原模型里程碑条件均未被重新定义。

## 7. 下一阶段最值得投入的一项

按最终 b8 成本：per-call plan workspace 约 **4.71 ms**、Host actions 约
**2.13 ms**、executor 约 **0.68 ms**；其中最大单个 Host action 是 SwiGLU
约 **0.46 ms**，Softmax 约 0.35 ms。但单纯继续做 Host action 融合已经进入
低收益区。

下一阶段建议优先做 **device-resident QKV → QK/PV handoff**，而不是继续加
Host 映射分支：

- 目标：让 QKV 输出留在设备 BO，QK/PV 的上传直接消费设备侧 QKV 切片，
  消除 `qkv` 下载和 QK/PV 的 Host 重新上传边界；不改变现有 FP16 位模式。
- 契约变化：
  - plan/action schema 增加 `device_source` binding 与显式 device slice/
    transpose 描述，版本化；
  - C validator 独立检查设备 BO 所有权、容量、offset/stride、别名/重叠和
    整数溢出；
  - runtime 需要 graph-level BO lifetime/ownership，跨 QK/PV 计划保持
    producer BO，失败/关闭路径明确释放；仍保持同步、非异步资源生命周期；
  - manifest 的 `group_layout` 升版本，保留 host mapped QKV 作为对照；
  - 需要新的逐位一致性和 packed 数值重放，但不新增数值语义。
- 不先做 device SwiGLU/Softmax 下沉：那会引入新的设备数值政策和独立阈值，
  应在上述无算术变化的设备数据路径验证后再评估。

## 8. 证据索引

- 最终 campaign：`results/20261026T120625Z-final/`
  - `analysis.json`、`analysis.md`
  - `verify-packed-summary.json`
  - `README.md`（身份、哈希、重放命令）
- 完整 MNIST：五个 profile 各 10,000 张，3 个逐轮交替 round；
- 旧优化结果：`OPTIMIZATION-RESULTS-1.md` … `OPTIMIZATION-RESULTS-7.md`；
- 当前原始证据索引：`results/README.md`。
