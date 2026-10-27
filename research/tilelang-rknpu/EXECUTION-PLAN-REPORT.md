# 执行方案比较、驻留编译与验收报告

日期：2026-10-27。阶段目标不是继续追逐单个算子，而是先解释当前已测配置下
CPU、厂商 RKNN 与自研 RKPlan 的差距，再在**不改变冻结数值政策、不引入未验证
硬件配方、不改 C ABI**的前提下实现一个能通过同 campaign 验收的纵向候选。

本阶段结论：

- **采用 `auto` 编译选择**：静态 RKPlan binding fanout `< 32` 的图继续走
  generic 路径；`>= 32` 的图使用 prepared Host-residency lowering。
- Tiny ViT b8 从 **10.238 → 7.442 ms**（prepared/generic 配对中位数
  **1.367x**），b1 从 **3.075 → 2.499 ms**（**1.230x**）。`auto` 在 ViT
  上执行同一 prepared 路径，实跑中位数 b8 **7.303 ms**、b1 **2.358 ms**。
- MLP 只有 13 个 binding slot，prepared 的 Host 回写超过可摊销的 Python
  binding 成本；`auto` 保持 generic。这不是按模型名分支，而是按静态 layout/
  dependency fanout 选择。
- 全 campaign 的 custom arms 全量 10,000 张、原始 logits 与全部 captured
  layer trace **逐位相同**；冻结 packed numeric spec 重放 **PASS / 0
  problems**。
- Host residual+RMSNorm 融合实测无稳定叠加收益，撤回为 opt-in control；
  device-resident QKV 的传输-only 上界在 ViT b8 仅 **0.377 ms**，且缺少
  device CUBE/bank reorder、跨 workspace BO 生命周期与 layout 模板，不进入实现。
- 优化后仍有明显差距：厂商在 b8 仍快 **1.71x**、b1 快 **2.31x**；CPU
  FP32 在 ViT b8/b1 分别快 **1.24x/2.70x**（CPU 是独立数值政策，不直接排名）。
  剩余主因是每个 batch 仍有 **8 次 submit**、Host Softmax/RMSNorm/SwiGLU
  边界以及逐个 RKPlan 的 C workspace 包络，而不是 QKV 单段搬运。

完整机器可读结果、成本模型和重放命令在
[结果目录](results/20261027T025648Z-execution-candidates/README.md)。

## 1. 比较口径

固定模型、权重、原始 MNIST 输入、静态 batch/tail 调度和评估样本数：

| 项 | 值 |
|---|---|
| board | RK3588 OPi 5 Plus |
| kernel / driver | `6.1.43-rockchip-rk3588-rknpu098` / `v0.9.8` |
| NPU | 单核 `RKNPU_CORE_MASK=1`，观测频率 1.0 GHz；campaign 不修改频率 |
| CPU | affinity 6，单线程 BLAS（OMP/OpenBLAS/MKL=1） |
| CPU 频率 | 未锁定；评估时 `/proc/cpuinfo` 采样 min/mean/max 408/1578/2304 MHz |
| 全量数据 | 五个 profile 各 10,000 张 MNIST test |
| warm 边界 | raw 输入准备 + 模型执行 + 输出物化；argmax 单独；reference 不计入 |
| setup | 单独记录，不摊入 warm latency |
| 轮次 | custom arm 最多 5 轮；hostfused 3 轮；外部 vendor/CPU 2 轮 |

数值政策分组：

- **同政策 compiler arms**：generic、prepared、auto、hostfused 使用同一
  RKPlan/hybrid 图、FP16 storage 与既有 packed per-logit 政策；实测全部逐位相同。
- **厂商 RKNN**：RKNN-Toolkit2/Lite2 2.3.2，non-quantized FP16，
  `single_core_mode=True`，runtime core 0。ONNX 由相同 FP32 模型数学转换，
  归一化/patchify 在厂商图内。其 accuracy/top-1 一致，但 ViT per-logit 误差
  **0.12277/0.10715** 超过 packed atol 0.0625，只能按 classification-preserving
  单列。
- **CPU**：NumPy FP32 reference，单线程 BLAS；它是参考数值路径，不与
  packed FP16 同政策排名。

## 2. 当前差距与已测成本

同 campaign warm latency（每静态 batch call）：

| profile | generic | prepared | auto | hostfused | vendor | CPU FP32 |
|---|---:|---:|---:|---:|---:|---:|
| mlp b32 | 2.152 | 2.187 | 2.163 | 2.127 | **0.834** | 6.861 |
| mlp b8 | 1.271 | 1.368 | 1.304 | 1.330 | **0.527** | 1.751 |
| mlp b1 | 1.233 | 1.293 | 1.206 | 1.234 | **0.409** | **0.255** |
| vit b8 | 10.238 | 7.442 | **7.303** | 7.550 | **4.272** | 5.873 |
| vit b1 | 3.075 | 2.499 | **2.358** | 2.283 | **1.022** | **0.874** |

单位 ms。table 中 auto 是最佳合法默认；hostfused 是显式候选 control，不默认。

外部相对 auto 的差距：

| profile | vendor 比 auto 快 | CPU/auto 比 |
|---|---:|---:|
| mlp b32 | 2.594x | 0.315x（auto 快 3.17x） |
| mlp b8 | 2.473x | 0.745x（auto 快 1.34x） |
| mlp b1 | 2.949x | 4.722x（CPU 快 4.72x） |
| vit b8 | 1.709x | 1.244x（CPU 快 1.24x） |
| vit b1 | 2.308x | 2.697x（CPU 快 2.70x） |

结论不是“NPU 总是更快”：MLP b1、ViT b8/b1 上 CPU FP32 已比当前 NPU 路径快；
但这需要新的独立数值政策，不能直接拿来做 packed FP16 的实现。

ViT b8 的 prepared 成本账本（每 call）：

| 项 | generic | prepared |
|---|---:|---:|
| warm model call | 10.238 ms | 7.442 ms |
| Python plan-node dwell | 2.861 ms | **0.135 ms** |
| Host actions | 2.131 ms | 2.219 ms |
| C workspace | 4.511 ms | 4.333 ms |
| executor | 0.690 ms | 0.677 ms |
| tasks / submits | 70 / 8 | 70 / 8 |
| upload / download | 296,448 / 230,144 B | 296,448 / 230,144 B |

prepared 确实消除了 Python 重复 binding 解析/有限值扫描和 ctypes 描述符重建；
但它没有消除任何 submit、upload/download 或 Host action。C workspace 仍在
4 ms 量级，ViT b8 的 8 次 submit 与厂商每个 batch 一次 runtime call 仍是结构性差别。
厂商内部具体如何融合/放置只能由 `vendor-perf` 探针观察到（50 NPU + 5 CPU 层等），
其 compiler/driver 内部机制不得在无证据时猜测归因。

## 3. 候选执行方案比较

| 候选 | 子图/布局 | 设备驻留与 submit | 消除成本 | 引入成本 | 数值/模板条件 | 结论 |
|---|---|---|---|---|---|---|
| **C0 generic** | 原 22 节点 hybrid，QKV mapped group | 无跨 plan 驻留；8 submit | 无 | 每 plan Python binding/validation | 当前冻结政策 | 控制组 |
| **C1 prepared Host residency** | 同 C0；按图分配稳定 Host slot | 设备侧仍 8 submit；无跨 plan BO | ViT b8 Python plan-node 2.861→0.135 ms；b1 0.637→0.100 ms | 输出回写、resident 内存、setup；无 transfer 改变 | 需一 producer/slot、稳定 view、无 Host alias；C 每 run 仍验 finite/binding | **采用为 ViT 默认路径；auto 只在 fanout≥32 启用** |
| **C2 host boundary fusion** | residual.add + rmsnorm 合并为已注册动作；mapped QKV 控制 | 无改变；仍 8 submit | 一个 Host action 边界 | 同一遍读写与回写；无 submit/transfer 收益 | 已有 `tensor.add_rmsnorm.last_axis.v1`，经 mock/board 逐位验证 | **撤回为 control**：b8 7.550 vs prepared 7.442，无可复现叠加收益 |
| **C3 device-resident QKV** | QKV→QK/PV 直接消费；Softmax 仍 Host | 需跨 workspace/plan BO 生命周期；submit 仍 8 | b8 省 49,152 B qkv download + 163,840 B qkv-attributable upload，模型上界 0.377 ms | 未实现的 device reorder、BO owner/lifetime/alias 检查；若改变映射需独立数值验证 | 缺 CUBE bank→QK/PV WEIGHT/tile reorder 模板；缺 public BO 契约 | **不实现**：上界不足以解释差距，且不能减 submit/Host Softmax |
| **C4 CPU placement** | 整个模型或部分算子在 CPU | N/A | 可免 NPU submit/transfer | 改变内部精度与执行语义 | FP32 reference 与 packed FP16 不同政策 | 单列为外部基线，不排名 |

### C1 规则为何不是模型名分支

`analyze_residency` 只证明以下事实：每个 tensor 恰好一个 producer；槽位互不
alias；所有 view 绑定到稳定地址；public output 另拷贝。`auto` 选择使用静态
binding slot 数：

- MLP：13 slots → generic；
- ViT b1：42 slots → prepared；
- ViT b8：210 slots → prepared。

这个谓词由 repeat group/layout/dependency 图产生，不读取 `model == ...`。
阈值 32 来自 measured crossover：MLP 只有 0.065–0.156 ms Python plan-node
可省，但 resident 回写是净增；ViT b1/b8 有 0.637/2.861 ms 可省。

## 4. 成本模型

新增 `tools/execution_cost_model.py`，输入
`profile_bridge_envelope.py` 的 generic 实测账本和 graph manifest；输出分别
计入：

- Python plan-node/validation；
- Host action 计算与 resident write-back；
- pack/upload、download；
- submit、guard、executor；
- 必要布局转换（若未建模则明确写 `missing` 并排除在收益外）。

同 campaign 校验（profile 只提供组件系数，收益校验使用交错轮的 generic/
prepared 中位数）：

| profile | binding slots | auto | measured saving | modeled saving | error |
|---|---:|---|---:|---:|---:|
| mlp b32 | 13 | generic | -0.036 ms | +0.123 ms | +0.158 ms |
| mlp b8 | 13 | generic | -0.097 ms | +0.075 ms | +0.172 ms |
| mlp b1 | 13 | generic | -0.060 ms | +0.077 ms | +0.137 ms |
| vit b8 | 210 | prepared | **2.796 ms** | **2.795 ms** | **-0.001 ms** |
| vit b1 | 42 | prepared | **0.577 ms** | **0.601 ms** | **+0.024 ms** |

MLP 的小正预测在实测中反号，说明模型在低 fanout 区没有足够信噪比；因此不是
“模型预测正收益就 blanket 启用”，而是要求收益幅度高于阈值并保留 generic
control。ViT 的预测误差小于 25 us。

device-QKV 筛选把 213 KiB 的桥接字节按实测 download/pack 系数折算为 b8
**0.377 ms**，并明确不计任何 reorder；这低于 auto 与 vendor 在 b8 的约
2.9 ms 差距，因此不支持“先做 QKV 驻留”。

## 5. 编译器与运行时实现

### 5.1 `passes/residency.py`

- 独立于 manifest parser 再证明单 producer、无重复写、消费者先于 producer 失败；
- 为每个 graph tensor 建立一个稳定 slot，并记录 producer kind / consumer /
  capture-only / model-output 属性；
- 不把未初始化 slot 当 finite：finite 权威仍在 C RKPlan boundary，每次
  workspace run 仍做 binding、shape、alias、finite、publication check；
- public output 由 runner 拷贝到新数组，保持“返回后跨下一次 run 仍可读”的
  既有调用契约。

### 5.2 prebound RKPlan I/O

`PlanCall.bind_prepared_into` / `bind_group_prepared_into` 在构造后验证一次
结构形状，缓存 stable view 和 ctypes binding array；执行时不再重建 descriptor。
QK/PV 可能复用同一个 pre-expanded plan，因此 prepared binding 以 node id 为
key 分离，避免两个节点互相覆盖绑定（这是一个实际发现的 correctness gate）。

### 5.3 auto selection and controls

- `execution_mode="auto"` 为默认；
- `generic` 保留每 call 重新 binding 的原路径；
- `prepared` 强制 prepared，用于成本/正确性归因；
- `hostfused` 仅通过 `build_hybrid_manifests.py --include-hostfused` 生成，
  不改变冻结默认 package hash；
- C ABI、C 硬件 recipe、RKPlan 提交协议、权重和数值阈值均未改变。

## 6. 验收

- `analysis.json`: **PASS / 0 problems**；
- 五 profile × 10,000 张全部完成，custom arms accuracy：
  MLP **0.98310**、Tiny ViT **0.98210**、top-1 **1.00000**；
- prepared/auto/hostfused 与 generic 的 raw logits **全部逐位相同**，
  captured layer trace **全部逐位相同**；
- `verify-auto-summary.json`: 冻结 packed spec **PASS / 0 problems**；
- Host regression：**267 passed, 3 skipped, 40 subtests**；ctest **30/30**；
- runtime library SHA-256：
  `878ef3fa6c81cb651e02783db2b6bb72829cb2b0941444eaf3bc755784051d85`；
- 生产 compiled manifest 保持
  `72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`。

prepared setup / resident memory：

| profile | generic setup | auto setup | resident bytes |
|---|---:|---:|---:|
| vit b8 | 60.877 ms | 64.301 ms | 505,664 |
| vit b1 | 43.977 ms | 45.258 ms | 63,208 |

这是一次性成本，不摊入 warm latency，也不宣称冷启动更快。

## 7. 明确非声明与未知

- 不宣称自研路径追平厂商；b8/b1 仍分别慢 1.71x/2.31x。
- 不把 CPU FP32 的提升直接算作 same-policy 优化；CPU 是独立数值政策。
- 不宣称厂商图内部每个融合/调度的细节；探针只能观察层落点与包络。
- 不宣称 prepared 减少 submit 或设备传输；tasks 70、submits 8、bridge 字节均未变。
- 不宣称 device QKV 已实现；它只有有界成本筛选和缺口清单。
- 不宣称多核、异步 lifetime、任意 layout、任意 stride 或 KV cache 支持。
- CPU/NPU 频率没有硬件锁频；绝对时间使用同 campaign 中位数，收益优先使用
  交错轮配对比值，单跑离群值不进入宣称。

## 8. 下一步

最优下一步不是单独做 QKV，而是 **whole-model device-resident graph /
Host-boundary collapse**：

1. 让 QKV→attention→FFN 的中间 tensor 保持在一个 runtime-managed device
   BO 生命周期内，只保留输入与最终 logits 的 Host 边界；
2. 把每 batch 8 个 submit 收敛为可验证的图级提交/分段提交，并单独计费
   layout reorder；
3. 只有当 device RMSNorm/Softmax/SwiGLU 有自己的冻结数值契约、校准和验收时
   才下沉；不能继承当前 packed Host 阈值；
4. 版本化 graph plan/recipe，补齐 BO owner/domain/capacity/alias/offset/lifetime
   检查；不开放裸 IOVA，不引入异步资源生命周期；
5. 保留 mapped-QKV + Host 控制路径和 generic/prepared 选择器，使每个新收益可
   逐位归因。

这些工作涉及新的 graph-level 公共契约、可能的新硬件 recipe 和独立数值政策；
在实施前应先提交具体方案、证据和取舍，不把“更多算子放 NPU”当作收益前提。

## 9. 重放入口

```bash
cd research/tilelang-rknpu
RK_CANDIDATE_LIMIT=10000 RK_CANDIDATE_ROUNDS=3 \
RK_CANDIDATE_EXTERNAL_ROUNDS=2 ./scripts/run-execution-candidates.sh
```

该 harness 当前依次执行 generic / prepared / auto / hostfused、vendor 2.3.2
和 CPU，执行完整 10k × 5 profile，自动采集 cost-profile ledger，并调用
analyzer、cost-model 与 frozen numeric verifier。具体命令、哈希和已知
separate-policy 说明见
[结果 README](results/20261027T025648Z-execution-candidates/README.md)。
