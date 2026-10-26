# 第二个优化包：FFN 设备边界收敛（实施报告）

日期：2026-10-26。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](TRANSFORMER-OPTIMIZATION-PLAN.md) 第 6 节方案 C。
本轮只动 FFN 的桥接边界与静态权重布局，不改模型数学定义、数据、batch/tail 或
QK/PV 策略。

## 0. 结论摘要

默认 Tiny Transformer 现在把 FFN 的 gate/up 两个分支合并为**一次 N=2H 的
打包 GEMM**（`Wgu = concat(Wg, Wu)`，沿输出轴拼接），随后仍由注册 Host action
按原 FP32 路径计算精确 SwiGLU，再执行原 down GEMM 与 residual add。

RK3588 core 1、CPU 绑核、逐轮交替三臂（packed / device / host），4 轮 × 32 次
合并 128 样本：

| profile | packed（默认） | device（全设备 FFN） | host（原路径） | packed 相对 host |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | **11.772 ms** | 11.839 ms | 12.176 ms | **1.034×** |
| Tiny ViT b1 | **3.585 ms** | 3.776 ms | 3.827 ms | **1.067×** |

- b8 吞吐 657.0 → **679.6 img/s**；b1 261.3 → **278.9 img/s**
  （均由合并 p50 换算）。
- 每次 b8 调用结构：`task_count 71 → 70`、`submit_count 9 → 8`、
  `upload_calls 135 → 134`、`upload_bytes 312 832 → 296 448`（−16 384 B）；
  下载次数 `71 → 70`，下载字节不变（gate+up 的 64 KiB 只是合并成一次下载）。
- b8 C 侧停留（单次代表运行、第 1 轮）：`4.857 → 4.552 ms`；其中
  `submit_wait 1.432 → 1.229 ms`、`guard 0.291 → 0.258 ms`、
  `pack 1.196 → 1.153 ms`。`download 1.067 → 1.085 ms` 略升，未做归因外推。
- 默认 packed 路径对原 host 路径的 logits 与
  `gate/up/act/ffn/h2` **逐位相同**（Host mock 与板端数值门均验证）。
- 全设备 FFN（`device`）是另一条**独立数值政策**：它去掉 Host SwiGLU，但
  `act/ffn/h2/h2_tokens` 的冻结容差必须从 0.0625 放宽到 0.125；因此只保留为
  显式可选清单，不作为默认路径。

## 1. 变更

### 1.1 打包 gate/up GEMM（默认）

- `tools/compile_models.py` 为 b1/b8 编译 `tiny_vit_bN_gate_up`：
  真实 TileLang `make_matmul_prim(m, 64, 256)`，输入 `A=[M,64]`，
  权重 `B=[64,256]`，输出 `Y=[M,256]`。
- `tools/build_hybrid_manifests.py` 生成默认清单 `tiny_vit_bN.rkhybrid.json`：
  `gate_up -> swiglu.packed_gate_up.v1 -> down -> residual_ffn`。
- 新注册 Host action `swiglu.packed_gate_up.v1`：把 `[M,2H]` 结果切成
  `gate=[M,H]`、`up=[M,H]` 两个连续 view，再用
  `float32` 的 `gate * sigmoid(gate) * up` 只做一次 FP16 store。
- 新清单常量类型 `weight_concat`：从现有 `ffn_gate_w` / `ffn_up_w` 沿轴 1
  拼接，不新增权重文件、不改变权重内容。

### 1.2 全设备 FFN（显式可选）

- `models/tilelang_matmul.py:make_swiglu_ffn_f16` 是真实 TileLang 源，
  入口为 `X, H, Wg, Wu, Wd, Y`；`H` 是 residual 输入，避免把 `norm2` 错当
  residual。前端 `_extract_ffn_branch` 现允许 residual 使用与分支输入不同的
  外部参数。
- 该计划包含 `gate mm -> up mm -> sigmoid LUT -> gate*sigmoid -> *up ->
  down mm -> residual add` 七个设备任务、一次输入边界和一次输出边界。
- 为保留冻结层门，额外编译 `tiny_vit_bN_ffn_trace`：同一提取图、同样 primitive
  任务，但导出 `gate/up/act/ffn`。`HybridRunner` 新增 `capture_only` 节点语义：
  普通测量路径完全跳过，只有 `capture=True` 的数值观测才执行。

### 1.3 清单版本化

每个 batch 生成三份显式清单：

| 清单 | FFN 路径 | 数值政策 |
|---|---|---|
| `tiny_vit_bN.rkhybrid.json` | 默认 packed gate/up + Host FP32 SwiGLU | 与原路径逐位一致 |
| `tiny_vit_bN_deviceffn.rkhybrid.json` | 全设备 FFN + capture-only trace | 独立重标定 |
| `tiny_vit_bN_hostffn.rkhybrid.json` | 原 gate/up/down + Host SwiGLU | 原路径对照 |

回退只需换清单；不需要改模型、权重或 runner 代码。

## 2. 数值验收

### 2.1 默认 packed 路径

- 新 spec：`models/numeric_acceptance_packed.json`。
- 冻结来源：`results/20261026T084825Z-numeric/`（train 32 图校准，
  disjoint test 32 图验收；未读标签）。
- `layer_tolerances` 与 `task_logit_tolerances` 与上一轮
  `numeric_acceptance_train.json` **逐项相同**（33 层 + 5 profile，全部
  `atol/observed_max_abs` 无差异）。
- 验证结果：**PASS / 0 problems**。
- 冻结后复放：`results/20261026T085111Z-numeric-replay/`，**PASS / 0 problems**。
- Host mock 逐位检查：默认 packed 与 `_hostffn` 的 logits 及
  `gate/up/act/ffn/h2` 全部 `array_equal`。

### 2.2 全设备 FFN 路径

- 新 spec：`models/numeric_acceptance_device_ffn.json`。
- 冻结来源：`results/20261026T084901Z-numeric/`，**PASS / 0 problems**。
- 相对原路径必须独立放宽的行：
  `tiny_vit.act`、`tiny_vit.ffn`、`tiny_vit.h2`、`tiny_vit.h2_tokens`
  的 `atol` 由 0.0625 改为 0.125；`tiny_vit.pooled` 的观察误差由 0.00390625
  变为 0.0078125，但容差仍为 0.015625；5 个 profile logit 容差不变。
- 这不是默认路径，也不继承 packed/host 的 bit-exact 结论。

## 3. 测量方法

- 三臂共享同一源码、同一 C 构建、同一 plan/weights/data payload；臂只在运行时
  选择清单。
- 每轮顺序 device → packed → host 交替；`taskset -c 6`；NPU core 1；每臂每轮
  32 次 + 8 次 warm-up；权重/runner 构造在计时区间外。
- 计时字段与结构计数来自现有 `profile_bridge_envelope.py`；packed/device/host
  的结构差异是本次目的。
- 结果目录：`results/20261026T085009Z-ffn/`（`comparison.json` 为逐轮合并结果）。

## 4. 测试

- Host：259 pytest 全通过；30/30 ctest 全通过。
- 新增覆盖：
  - 默认 packed 清单 `task_count/submit_count/upload/download` 与
    `weight_concat`；
  - packed 对 host 的 logits 与全部 FFN trace 逐位一致；
  - `deviceffn` 清单为独立数值路径、且 trace 节点带 `capture_only`；
  - `capture_only` 在普通运行中跳过、在 capture 中执行；
  - `numeric_acceptance_packed.json` 与
    `numeric_acceptance_device_ffn.json` 的证据均可复放。

## 5. 未做与限制

- 没有把 gate/up 和 down 之间再融合；中间仍需要 Host FP32 SwiGLU 才能保持
  默认路径逐位一致。
- 没有扩展 C `independent_mm` chain 的几何支持域；尝试把两次 FFN branch MM
  放进一次 submit 会被 C 校验器按既有验证域拒绝，未绕过。
- 没有改 QK/PV shared group、Softmax、RMSNorm、量化、多核或数值政策之外的参数。
- b1/b8 的收益分别来自同一 campaign 的合并 p50；不同 campaign 的绝对延迟不可
  直接拼接成新的倍数结论。

## 6. capture-only 生产隔离与 trace 证据边界

本轮补齐了 `capture_only` 的**加载期**隔离，而不再只依赖执行期跳过：

- 普通执行节点读取 `capture_only` 节点的输出时，加载器直接拒绝；
- `outputs` 将 `capture_only` 节点的输出声明为普通模型输出时，加载器直接拒绝；
- 现有正例保留：`tiny_vit_bN_deviceffn` 的 `ffn_trace` 仍可在
  `capture=True` 时执行并导出 `gate/up/act/ffn`；
- `tiny_vit_b8_deviceffn` 在 `capture=False` 与 `capture=True` 下的最终
  `logits` 逐位相同，普通运行不会执行 trace 计划。

**证据边界**：trace 计划是同一提取图但单独执行的诊断 sibling，其
`gate/up/act/ffn` 是**重新执行得到的诊断证据**，不能表述为“被测生产计划内部值的
直接采样”。生产计划只发布其自身最终输出；冻结层门对 trace 的检查用于回归诊断，
不能替代对生产计划边界的证明。

## 7. QK/PV shared group 在 packed 默认 FFN 上的复验

保持 QK/PV 清单字段不变：

```json
{"repeat_submit": "shared_group", "group_plan": "attn_matmul_group"}
```

runner 提供三条显式策略臂：`separate`、`b1`、`chain_shared`
（`chain_shared` 强制使用清单声明的 shared group 计划；`manifest` 为默认等价路径）。
未扩大 C 侧已验证的独立 MM 几何域。

`results/20261026T085859Z-policy/`，packed FFN 默认，core 1、绑核、逐轮交替，
4 轮 × 32 次合并 128 样本：

| profile | B1 p50 | separate p50 | chain_shared p50 | chain_shared 相对 B1 |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | 14.584 ms | 22.344 ms | **11.692 ms** | **1.247×** |
| Tiny ViT b1 | 3.849 ms | 4.700 ms | **3.613 ms** | **1.065×** |

- b8 逐轮比值：`1.261 / 1.276 / 1.238 / 1.218`；
  b1：`1.060 / 1.105 / 1.060 / 0.923`。b1 第 4 轮反转，收益较小，不表述为稳定
  延迟改善；b8 四轮一致。
- 结构（b8）：B1/separate `task_count=70, submit_count=70`；
  chain_shared `task_count=70, submit_count=8`。上传/下载次数和字节三臂相同：
  `134 / 70`、`296 448 / 230 144 B`。
- 结构（b1）：B1/separate `task_count=14, submit_count=14`；
  chain_shared `task_count=14, submit_count=8`。上传/下载字节相同：
  `38 400 / 29 440 B`。
- 创建成本中位数（4 次采样）：b8 `runner_setup` B1 13.40 ms、separate 13.44 ms、
  chain_shared 13.67 ms；b1 分别为 7.71 / 7.75 / 7.84 ms。清单加载约 30 ms（b8）
  和 12.5 ms（b1）。chain_shared 创建成本略高 0.13–0.20 ms（b8）/
  0.09–0.13 ms（b1），未做显著性检验，不用于解释热路径收益。
- 默认策略：保留 `shared_group` / `chain_shared`。b8 收益明确；b1 pooled p50
  小幅领先但逐轮波动较大，按实测保留该策略而不承诺固定延迟收益。

## 8. 数值复验（packed 冻结政策，不重标定）

`results/20261026T085820Z-numeric-replay/`：使用
`models/numeric_acceptance_packed.json` 与 v1 冻结容差，重新观测 disjoint test
子集；**PASS / 0 problems**。实际模型输入下检查了 33 个层门和 5 个 profile
logit 门，例如 `tiny_vit.gate` max error 0.0078125、`tiny_vit.act` 0.015625、
`tiny_vit.ffn` 0.015625、`tiny_vit.h2` 0.015625、`tiny_vit.logits` 0.015625；
5 个 profile 均 `same_top1=True`。没有重新校准或放宽阈值。
