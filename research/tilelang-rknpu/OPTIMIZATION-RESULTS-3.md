# 第三个优化包：RMSNorm 连接边界评估与 Host 收口（实施报告）

日期：2026-10-26。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](TRANSFORMER-OPTIMIZATION-PLAN.md) 第 6/7 节方案 D
的 RMSNorm 部分。默认路径仍为 packed FFN + QK/PV `shared_group`/`chain_shared`；
本包不扩展独立 MM chain 几何域、不引入多核、不同时推进通用 Softmax。

## 0. 结论摘要

- **未发现可合法减少设备边界、且成本合理的 RMSNorm 连接。**
  当前 RKPlan 能力集没有行归约、general rsqrt、square 原语；已有 `co_row_ops`
  RMSNorm 是 C 侧显式 hybrid 算子，内部仍含 Host 全局 rsqrt 和逐 tile 统计下载，
  不能透明塞进 RKPlan 当作纯 NPU task。
- 复用 `co_row_ops` 的板端实测反而更慢：m=128,n=64（b8）约 **0.854 ms/次**
  （中位，1000 次重复；2 submits、3 tasks、2 uploads、2 downloads），
  b1 m=16,n=64 约 **0.266 ms/次**。对照当前 Host RMSNorm + 相邻上传：
  b8 `0.190 + 0.054 = 0.244 ms`（RMSNorm1）与
  `0.214 + 0.053 = 0.267 ms`（RMSNorm2）；b1 约 `0.093/0.095 ms`。
  因此该复用路径约为当前成本的 **2.8–3.5 倍**，且不消除 Host rsqrt。
- 作为保持数值路径的候选，实现了 `residual_attention + rmsnorm2` 的单 Host
  action（`tensor.add_rmsnorm.last_axis.v1`）并通过显式清单
  `tiny_vit_bN_rmsnorm` 选择。该路径与默认路径在 logits 及
  `h1/norm2/gate/up/act/ffn/h2` 上**逐位相同**。
- 板端 4 轮 × 32 次逐轮交替：fused 相对 control 的 b8 p50 为
  **0.997×**，b1 为 **0.989×**，结构计数与传输完全相同；没有实测收益。
  **默认路径保持不变**，fused 仅保留为显式对照清单。

## 1. 当前默认路径的实际成本（逐轮聚合）

数据：`results/20261026T091322Z-rmsnorm/` 的 control 臂（packed FFN +
QK/PV chain_shared），4 轮 × 32 次，core 1，CPU 绑核。
下表为每轮运行内 Host action / plan 分项在 128 个样本上的**合并中位**，不是
单次代表运行；它们用于说明成本位置，不与整体 p50 做加减解释。

| 区间 | b8 | b1 | 说明 |
|---|---:|---:|---|
| RMSNorm1 Host | 190.5 µs | 81.7 µs | Host FP32 mean/rsqrt/gamma，1 次 FP16 store |
| RMSNorm2 Host | 213.9 µs | 84.6 µs | 同上 |
| Softmax Host | 331.9 µs | 98.4 µs | stable max/exp/sum，Host FP32 |
| RMSNorm1 相邻上传（qkv pack） | 53.7 µs | 10.8 µs | 含 qkv 输入的 upload/pack |
| RMSNorm2 相邻上传（gate_up pack） | 53.1 µs | 10.8 µs | 含 packed gate/up 输入 upload |
| QK 相邻下载（qk download） | 189.6 µs | 23.9 µs | scores 出设备，供 Softmax |
| Softmax→PV 相邻上传（pv pack） | 412.7 µs | 40.7 µs | probs/V 入设备 |

Host 算术与桥接是两类开销：RMSNorm/Softmax 的 `elapsed_us` 是 Host 计算；
相邻 plan 的 `pack_us/download_us` 是设备边界前后的 staging/BO 成本。二者不能
简单相加成 p50，也不能把单次分项外推为整体加速比。

## 2. RMSNorm → QKV 与 RMSNorm → packed gate/up 的比较

| 连接 | 当前 Host 成本 | 后继 plan 输入桥接 | 可复用能力 | 结论 |
|---|---:|---:|---|---|
| RMSNorm1 → QKV | 190.5 µs (b8) | qkv pack 53.7 µs | 无 RKPlan 行归约/rsqrt；只能走 `co_row_ops` | 合法边界消除受阻 |
| RMSNorm2 → packed gate/up | 213.9 µs (b8) | gate_up pack 53.1 µs | 同上；packed gate_up 是单一 N=2H GEMM | 合法边界消除受阻；Host 成本更高，选作融合试验 |

`co_row_ops` 的端口是 `co_rowop_*` / `co_graph`，不是 RKPlan 的
`rk_action_desc`：要接入产品 hybrid 清单需要新增 plan action/template 语义与
C 校验，而不是现有能力。其内部流程是：device FP16 square、FP32 per-tile
sum、download 统计、Host 全局 rsqrt 和 gamma 系数、upload 系数、device scale。
因此即使把它包装成“RMSNorm 节点”，Host 步骤仍然存在，且新增 2 submits /
3 tasks / 2 uploads / 2 downloads。`co.lut.rsqrt_bounded` 的声明输入域仅为
`[0.25,4]`，也不能直接代表通用 RMSNorm 的均值域。

### 2.1 板端 `co_row_ops` 探针

使用已有 `row_graph`（不新增 NPU task），m/n 与模型一致，1000 次重复、
core 1，三次中位数：

```
m=128 n=64 chain    0.854 ms/run, 2 submits, 3 tasks, 2 uploads, 2 downloads
m=16  n=64 chain    0.266 ms/run, 2 submits, 3 tasks, 2 uploads, 2 downloads
device_bytes: 324032 B (b8), 252352 B (b1)
```

对照 Host RMSNorm + 相邻上传约 0.244/0.267 ms (b8) 与 0.093/0.095 ms (b1)，
该路径没有采用。

## 3. 保持原数值路径的 Host 融合尝试

新增 Host action：

`tensor.add_rmsnorm.last_axis.v1`：
- 输入 `a,b` FP16、`gamma` FP16；
- 先做 `a.float32 + b.float32 -> FP16`，与现有 `tensor.add.v1` 的 residual
  收窄点完全相同；
- 再以该 FP16 值做 last-axis FP32 RMS 均值、rsqrt、gamma 乘，最后一次 FP16
  store；
- 输出 `h`（residual FP16，保留 trace 与后续 FFN residual）和 `y`（norm）。

显式清单 `tiny_vit_bN_rmsnorm` 将
`residual_attention + rmsnorm2` 合并为一个 Host action，后续仍接
`gate_up -> swiglu -> down -> residual_ffn`。默认清单和 fused 清单共享同一
plan 包，不改变任何设备任务。

## 4. 端到端对照

`results/20261026T091322Z-rmsnorm/`，4 轮 × 32 次，逐轮交替，core 1：

| profile | control p50 | fused p50 | fused / control | 结构 |
|---|---:|---:|---:|---|
| Tiny ViT b8 | 11.885 ms | 11.921 ms | **0.997×** | 完全一致 |
| Tiny ViT b1 | 3.587 ms | 3.628 ms | **0.989×** | 完全一致 |

- b8 逐轮 `1.003 / 0.959 / 0.980 / 1.033`；b1 `0.973 / 0.990 / 1.009 / 0.976`。
- tasks/submits、upload/download 次数与字节三臂相同；b8 为 70/8、
  134/70、296448/230144 B，b1 为 14/8、22/14、38400/29440 B。
- Host action 合并后 median 从 `140.9 + 213.9 = 354.8 µs` 降到
  单 action `341.4 µs`（b8），b1 从 `129.6 µs` 降到 `120.5 µs`；
  节省量相对整体 p50 太小，墙钟无收益。
- 创建成本：b8 runner setup 14.14 → 14.02 ms，b1 7.73 → 7.70 ms，量级相同。

**默认策略：保留原路径。** fused 清单仅作为显式对照，不改变默认。

## 5. 数值复验

- 默认 packed：`results/20261026T091548Z-numeric-replay/`，使用
  `models/numeric_acceptance_packed.json` 的既有容忍限，**PASS / 0 problems**。
- fused RMSNorm 清单：`results/20261026T091612Z-numeric-replay/`，通过
  `--manifest-key tiny_vit_b8=tiny_vit_b8_rmsnorm` 等显式映射运行同一
  packed 容忍限，**PASS / 0 problems**。
- 两处都没有调用 `calibrate_numeric_train.py`，没有修改任何
  `atol/rtol/observed_max_abs`；只更新了 plan 包身份哈希，并重新观测
  deterministic train/test 子集以匹配新包。
- 全设备 FFN 的独立政策仍单独保留：`results/20261026T091954Z-numeric/`
  在显式 `_deviceffn` 映射下 **PASS / 0 problems**，但
  `act/ffn/h2/h2_tokens` 的 `atol=0.125` 属于新数值政策，不是原冻结政策。

## 6. 测试

- Host：260 pytest 全通过；30/30 ctest 全通过。
- 新增覆盖：
  - 合并 `residual + RMSNorm` action 与默认两 action 的逐位一致；
  - `_rmsnorm` 显式清单节点/action 形状与 trace；
  - `--manifest-key` 数值观测映射与两套 replay；
  - `profile_bridge_envelope.py` 的 pooled host/plan 统计与创建成本字段。

## 7. 未做

- 没有新增 device RMSNorm task/template，也没有把 `co_row_ops` 包装成
  RKPlan 伪节点。
- 没有改变 Host FP32 SwiGLU、Softmax、RMSNorm 的数学路径。
- 没有调整默认策略：fused RMSNorm 无实测收益。
- 没有扩展独立 MM chain、多核或通用 Softmax。
