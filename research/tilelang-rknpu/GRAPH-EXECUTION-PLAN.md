# 图级执行方案与最小验证原型

> **后续 package identity note:** this report's `72f898…` compiled-package
> hash is the historical graph-stage identity. The later 2026-10-27
> template/granularity stage recompiled the MLP plans for packed K-major uploads;
> the current `models/compiled/manifest.json` hash is
> `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`.
> The compiled graph executor itself is unchanged. See
> [TEMPLATE-GRANULARITY-REPORT.md](TEMPLATE-GRANULARITY-REPORT.md).

日期：2026-10-27。本阶段承接 prepared Host-residency lowering，但不再把
“整模型设备驻留、只剩输入输出边界”作为预设目标。产物是一个**已实现并实测**
的 same-policy 图级执行路径，以及一个需要独立数值政策决策的 device-FFN
候选。机器可读证据在
[results/20261027T080223Z-graph-plan](results/20261027T080223Z-graph-plan/README.md)。

## 0. 结论与选择

**值得产品化并已采用：prepared graph 的 compiled flat executor。** 它不改变
算子、权重、数值政策或硬件配方，把已证明可驻留的 Host 图一次性编译成平坦执行
序列，并把 metadata reshape 下沉为稳定 view、跳过生产路径中的 trace-only 写回。

同 campaign、全 10,000 张、CPU governor `performance`、NPU 单核：

| profile | generic ms | prepared loop ms | auto：compiled graph ms | auto/generic | auto/prepared |
|---|---:|---:|---:|---:|---:|
| Tiny ViT b8 | 9.219 | 6.371 | **6.072** | **1.518x** | **1.049x** |
| Tiny ViT b1 | 2.875 | 2.232 | **2.088** | **1.377x** | **1.069x** |

- 应用计时使用 `runner.run(..., stats=None)`；`RunStats` 另做独立 100-call
  采样，不进入上面的 application interval。
- auto 在 MLP 上仍选择 generic，行为不变；只有高 binding-fanout 的 ViT 使用
  compiled graph。
- 所有同政策 arms 的 10k logits 和 captured traces 逐位相同；冻结 packed
  numeric spec 重放 **PASS / 0 problems**。

**不采用：device FFN 作为统一默认路径。** 现有独立数值政策的 device FFN
候选在 b8 有收益、b1 回归：

| profile | same-policy auto ms | device FFN ms | 比值（device/auto） | max Δlogit | tasks/submits | upload/download B |
|---|---:|---:|---:|---:|---|---|
| Tiny ViT b8 | 5.962 | 5.709 | **0.958x（快 4.2%）** | 0.03125 | 70/8 → **75/12** | 296448/230144 → **280064/164608** |
| Tiny ViT b1 | 1.923 | 2.059 | **1.071x（慢 7.1%）** | 0.03125 | 14/8 → **19/12** | 38400/29440 → **36352/21248** |

它需要不同的 FFN layer 数值政策；本阶段不放宽默认门限，也不把 batch 相关
的政策切换偷偷藏进 auto。若未来要以 b8 延迟优先采用它，需要单独批准
数值政策及 per-profile 验收故事。

## 1. 当前 Host 边界为什么存在

对默认 Tiny ViT b8 执行图逐节点分类。详细表格见
[boundaries/tiny_vit_b8.md](results/20261027T080223Z-graph-plan/boundaries/tiny_vit_b8.md)；
`measured` 是 prepared control 的每 call Host dwell，不是可回收收益。

| node | 原因类别 | 当前实现做了什么 | measured ms | 当前 same-policy 可消除 | 必要新增成本 |
|---|---|---|---:|---:|---|
| patchify | numeric semantics + physical layout + implementation | FP32 normalize、patch raster、F16 store | 0.149 | 否 | 若下沉需 device preprocess 契约 |
| patch_pad | physical layout + implementation | 49→64 的 zero pad | 0.050 | 本原型未消 | 可做 upload zero-fill，但需要新的 binding layout 验证 |
| position_add | numeric semantics + missing device op | FP32 broadcast add 后一次 F16 写回 | 0.126 | 否 | device broadcast-add recipe |
| rmsnorm1 | numeric semantics + missing device op | FP32 row mean/rsqrt，一次 F16 写回 | 0.161 | 否 | device row reduction + rsqrt recipe |
| softmax | numeric semantics + missing device op | stable max/exp/sum，F16 写回 | 0.285 | 否 | device stable row softmax recipe，或独立数值政策 |
| merge_heads | physical layout + implementation | head-major→token-major 拷贝 | 0.056 | 本原型未消 | 输出 scatter/输入 head-concat layout |
| residual_attention | numeric semantics | FP32 add 后 F16 收窄 | 0.128 | 否 | 若下游全设备可在 FFN 内吸收 |
| rmsnorm2 | numeric semantics + missing device op | FP32 row mean/rsqrt | 0.181 | 否 | device row reduction recipe |
| swiglu | numeric semantics + missing device op | FP32 sigmoid/mul，F16 act；gate/up 仅 trace | 0.402 | 仅跳过 trace 写回 | 独立 policy 的 device LUT/图融合 |
| residual_ffn | numeric semantics | FP32 add 后 F16 收窄 | 0.128 | 否 | 可并入独立 device FFN |
| reshape_tokens | physical layout | 2-D→3-D contiguous reshape | 0.089 | **是：稳定 view** | 无 per-call 成本 |
| mean_pool | numeric semantics + missing device op | FP32 token mean 后 F16 | 0.083 | 否 | device FP32-preserving reduction |
| export_logits | numeric semantics + resource interface | F16 padded→F32 [B,10] | 0.039 | 否 | 公共输出契约需改，当前不划算 |

合计 measured Host dwell **1.876 ms**；直接、可证明消除的只有 reshape 的
**0.089 ms**。其余约 1.787 ms 不能当作可回收 Host 时间：其中大部分是
必须保留的数值工作，或者只有当新增 device 配方成立时才可转移。另有
`gate/up` trace-only 输出 64 KiB 写回，按 8 GB/s 估计约 **0.008 ms**；这只是
筛选估计，实际收益由测量决定。

## 2. 两条路线

### 路线 A：保留现有数值政策的图级编排／部分驻留

已实现内容：

- `passes/graph_execution.py`
  - 证明 `tensor.reshape.v1` 的输入输出元素数/dtype 匹配后可变成一个稳定 view；
  - 标记 production-needed 与 trace-only tensors；
  - 为每个 Host action 生成原因分类、device gap 和 current-impl 信息。
- `HybridRunner(execution_mode="graph")` 和 auto 的高 fanout 分支
  - 把 graph 拓扑编译成平坦的 `_GraphHostOp` / `_GraphNpuOp` / `_GraphViewOp`；
  - Host 输入字典和 resident output 指针在构造期准备；
  - NPU 继续使用已绑定的 prepared PlanCall；
  - production 不写回 trace-only 输出；
  - C 仍然在每个 RKPlan 入口做 binding、finite、alias 和 publication 检查；
  - public output 仍复制到调用者拥有的新数组。
- 没有新增 C ABI、硬件 recipe、算子、数值阈值或 graph-level 公共接口。
- `auto` 仍是 selector；低 fanout 继续 generic，高 fanout 选 prepared residency，
  并在该 prepared 路径内使用 compiled graph backend。显式 `prepared` 保留
  原 prepared loop 作为 control。

消除的成本：

- metadata-only reshape action（同时保留 trace view）；
- 每 call 重建 Host 输入字典、遍历原始 manifest、重复 Host output 校验与
  trace-only production copy；
- 一部分 prepared loop 的 Python/ctypes 外层 dwell。

不消除：

- device submits、device 传输、任何 Host 数值算子。

### 路线 B：使用独立数值政策的设备子图

本阶段只比较现有、已经验证过数值政策的 device FFN，不虚构尚未实现的
device Softmax/RMSNorm 收益。设备 FFN 将 gate/up、SiLU、gated multiply、
down 和 residual 放在一个设备计划里，和 mapped-QKV 组合成
`_deviceffn_qkv` 候选。

它能消除：

- Host SwiGLU 与 residual FFN（b8 Host 测量约 0.530 ms）；
- gate/up 到 down 的部分 Host bridge 字节：b8 upload 296,448→280,064 B，
  download 230,144→164,608 B。

它引入：

- 更多 device task 和 submit：b8 70/8→**75/12**，b1 14/8→**19/12**；
- 独立 FFN 数值政策：现有 device-FFN spec 对 `act/ffn/h2/h2_tokens`
  使用 **atol=0.125**，不能继承 packed 0.0625；
- 新增 per-profile 验收和回退边界。

结果不是“设备化必然更快”：b8 有 **4.2%** 整模型收益，b1 反而慢
**7.1%**。因此本阶段**不把 device FFN 合入默认 auto**。若未来目标明确转为
b8 延迟优先，这是一个需要单独批准的数值政策决定，而不是编译器自动选择。

### 未进入原型的 device attention

QK→Softmax→PV 的设备子图需要：

- stable max/exp/sum/normalize 的 device row reduction；
- 与 Host FP32 softmax 等价的数值契约，或独立校准的新政策；
- QKV CUBE/bank 到 QK/PV WEIGHT tile 的设备 reorder；
- 跨 plan 的 device BO 生命周期/alias/failure 契约。

这些都是新的硬件配方或公共契约，现有证据不足。按当前测量，QKV 传输-only
筛选估计 b8 约 0.377 ms；它没有覆盖 softmax 的 0.285 ms Host 计算，也没有
证明 device reorder 的成本。该子图在本阶段明确不实现优先。

## 3. 已实现的纵向路径

### 3.1 编译期

1. `analyze_graph_execution()` 检查 default graph：
   - `reshape_tokens` input/output dtype 和元素数一致；
   - 目标 shape 与属性一致；
   - production/trace-only 分类来自真实节点依赖。
2. `_prepare_resident_execution()` 分配稳定 slots 后，用
   `source.reshape(shape)` 建立 view，并检查 dtype、shape、C 连续性和
   `np.shares_memory`。
3. `_compile_graph_ops()` 生成 production 与 capture 两套平坦 op tuple。
   Host input template 复用 resident buffers；NPU binding key 按 node id 隔离，
   与现有 QK/PV 共 plan 的 prepared binding 规则一致。

### 3.2 运行期

- production 路径不执行 `reshape_tokens` Host action，不建立新的输出数组；
- `swiglu` 的 `gate/up` 仍由 action 计算，但 production 不把它们写入 resident
  slots；capture 路径仍写回，因此 layer/numeric evidence 不变；
- 每个 NPU op 仍走 `run_prepared`/`run_group_prepared`，C workspace 的每次
  run 校验不变；
- public output 逐 call 检查 finite 并复制到新数组；
- 记录 `graph_view_ops` 与 `graph_skipped_trace_output_writes` 结构计数。

### 3.3 公共契约状态

本次**没有引入新的 graph-level 公共接口**。compiled graph 是 `HybridRunner`
内部 lowering，C ABI、artifact、plan hash、权重的公共语义不变。

若未来引入跨 plan device BO，需要先审查并固定以下规则；本阶段只设计、不实现：

- **ownership**：graph plan 拥有内部 BO，caller 拥有 fd 和外部输入/输出；
  plan 本身不拥有设备资源；禁止裸 IOVA。
- **domain**：同一 fd、同一设备/核策略；不支持跨 fd 或跨设备 BO。
- **capacity**：静态 shape/byte 预算，整数溢出和物理列 padding 检查；容量与
  layout 必须同时匹配。
- **layout**：logical shape、physical rows/cols/atom、CUBE/WEIGHT/LINEAR、
  transpose/strides 全部显式；producer/consumer 不一致时必须插入单独计费的
  conversion，不能隐式重解释。
- **synchronization**：同步执行；一个图调用内不引入异步用户 lifetime；
  如允许分段，段边界、wait 和 Host sync 必须显式。
- **lifetime**：producer→last-consumer 区间；每个中间 BO 有 owner、domain、
  start/end；不能提前释放或在未完成 consume 时复用。
- **alias**：默认禁止输入/输出、输出/输出、跨节点读写重叠；仅显式
  accumulator/inout recipe 可 alias，并由 C 独立检查 offset/range。
- **failure isolation**：预提交错误不发布；提交后错误 poison graph context，
  保留 BO/handle 进入 quarantine/close 路径；不得返回部分结果。
- **output publication**：所有必需输出 finite/guard 通过后一次发布；失败时
  旧输出不被部分覆盖，调用者看到明确错误。
- **identity**：graph plan hash、layout id、primitive capability id 和 numeric
  policy id 必须共同匹配；不满足即 fail closed。
- **budget/compatibility**：device/host byte 预算、重复提交上限和静态 repeat
  几何；不把未测几何泛化。

这些规则只会在一个已验证的 device subgraph 有净收益后才成为公共接口；不以
“整图驻留应该更快”为依据先建平台。

## 4. 实测与数值验收

### 4.1 同政策全量

独立 campaign：3 rounds、每 round 以 batch 为粒度轮换 generic/prepared/auto
的执行顺序、每个 profile **10,000 张**、CPU governor performance 2.304 GHz、
NPU 1.0 GHz、CPU affinity 6。

| profile | generic | prepared | auto graph | auto/generic | auto/prepared |
|---|---:|---:|---:|---:|---:|
| Tiny ViT b8 | 9.219 ms | 6.371 ms | **6.072 ms** | **1.518x** | **1.049x** |
| Tiny ViT b1 | 2.875 ms | 2.232 ms | **2.088 ms** | **1.377x** | **1.069x** |

同一批 runner 的带统计采样（不作为 app 主值）：

| profile | generic | prepared | auto graph |
|---|---:|---:|---:|
| ViT b8 | 9.471 ms | 6.617 ms | **6.375 ms** |
| ViT b1 | 3.242 ms | 2.543 ms | **2.425 ms** |

结构与内存：

| profile | auto backend | tasks/submits | view/skip | setup ms | resident B |
|---|---|---|---:|---:|---:|
| ViT b8 | compiled_graph | 70/8 | 1/2 | 21.5 | 505,664 |
| ViT b1 | compiled_graph | 14/8 | 1/2 | 9.6 | 63,208 |

`setup` 是 runner 创建成本，单独计时、不摊入 warm。Auto 在 b8 比 prepared
多约 0.33 ms setup，b1 约多 0.32 ms；收益来自重复调用，不宣称冷启动更快。

### 4.2 外部参考（同一 governor/数据）

| profile | auto graph ms | vendor 2.3.2 ms | CPU FP32 ms | vendor/auto | CPU/auto |
|---|---:|---:|---:|---:|---:|
| ViT b8 | 6.072 | **2.907** | 5.847 | **2.089x** | **1.038x（CPU 更快）** |
| ViT b1 | 2.088 | **0.795** | 0.869 | **2.627x** | **2.402x（CPU 更快）** |

厂商 warm setup 约 62.9/61.1 ms；auto 为 21.5/9.6 ms。厂商 ViT max logit
error 为 0.12277/0.10715，超过 packed 0.0625；CPU 是 FP32 独立政策。
因此这两列只说明剩余系统差距，不是可由当前编译器直接回收的损失。厂商一次
Host `rknn_run` 不等于一次 device Submit，内部 submit 数仍为未知。

### 4.3 正确性与冻结政策

- 同政策 arms：所有 round 的 10k logits **逐位相同**；captured layer trace
  **逐位相同**。
- `verify-auto-summary.json`：冻结 packed spec **PASS / 0 problems**。
- 生产 compiled manifest 仍是
  `72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`。
- Runtime library SHA-256：
  `878ef3fa6c81cb651e02783db2b6bb72829cb2b0941444eaf3bc755784051d85`。
- Host regression：**346 passed, 3 skipped, 68 subtests**；ctest **30/30**。

## 5. 失败与淘汰记录

- metadata reshape 如果 input/output dtype、元素数或实际内存连续性不匹配，
  编译器 fail closed，不做“看起来像 reshape”的猜测。
- shared plan 的 QK/PV prepared bindings 按 node id 分 key；否则第二个节点会
  覆盖第一个节点的 ctypes 指针。已有测试覆盖。
- production trace-only 写回只跳过没有 production consumer、不是 public output
  的 tensor；capture=True 仍完整物化并对 layer trace 做逐位验收。
- device FFN 有 b8 正向、b1 反向，且需要独立数值政策；淘汰为默认候选。
- device-resident QKV 的 0.377 ms 只是筛选估计，不是严格上界；无 CUBE/bank
  reorder 模板，淘汰为本阶段实施候选。
- 没有把 `stats` 插桩成本算入 application latency；反过来，也没有用跳过
  `RunStats` 的数字与厂商插桩字段逐项拼表。

## 6. 重放

完整 campaign 的 runner/tool 入口：

```bash
cd research/tilelang-rknpu

# 1) 全 10k、same-policy generic/prepared/auto，应用计时 stats=None，
#    另做 100-call stats 采样；CPU governor 由外层脚本设为 performance。
PYTHONPATH=$PWD/python RKNPU_PLAN_LIB=/path/to/librknpu_plan.so \
taskset -c 6 python3 tools/benchmark_graph_execution.py \
  --backend board --lib /path/to/librknpu_plan.so \
  --plans-dir models/compiled --weights-dir models \
  --data-dir .data/mnist --limit 10000 --rounds 3 --stats-calls 100 \
  --batch 8 --batch 1 --out /tmp/graph-benchmark.json
```

外部分析按结果 README 中的 `r1/r2-{vendor,cpu}` 目录重放。图级结果分析：

```bash
latest=results/20261027T080223Z-graph-plan
PYTHONPATH=python:. python3 tools/analyze_graph_execution.py \
  --benchmark "$latest/benchmark.json" \
  --external "$latest/external" \
  --boundaries "$latest/boundaries" \
  --independent-benchmark "$latest/deviceffn/benchmark.json" \
  --out "$latest/analysis.json" --markdown "$latest/analysis.md"
```

boundary ledger：

```bash
PYTHONPATH=python:. python3 tools/analyze_graph_boundaries.py \
  --plans-dir models/compiled --manifest tiny_vit_b8 \
  --profile-key tiny_vit.b8 --benchmark "$latest/benchmark.json" \
  --out "$latest/boundaries/tiny_vit_b8.json" \
  --markdown "$latest/boundaries/tiny_vit_b8.md"
```

## 7. 非声明

- 不宣称当前自研路径追上厂商或 CPU；ViT b8/b1 仍慢于 vendor，CPU 仍更快。
- 不宣称 device-resident QKV 已实现，0.377 ms 不是严格上界。
- 不宣称 device FFN 的 per-logit 等价；它使用独立数值政策。
- 不宣称任意 graph/layout/stride、异步执行、多核、跨 fd 或 KV cache。
- 不把一次 vendor runtime API 调用推断为一次设备 Submit。
- 不宣称 32-slot fanout 是已证明的一般性能交叉点；它只是当前样本上的经验
  选择规则，后续数据可以推翻。
