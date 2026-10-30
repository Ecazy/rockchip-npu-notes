# 单 GEMM 显式 K 分块与 K-tail 交付报告

日期：2026-10-30  
结果目录：`results/20261030T083436Z-k-tail/`  
代码提交：`a3ac0ad7324c2da7b57c3269b5b0df8881031638`

本轮只扩展单 GEMM 的源表达、frontend 识别、静态 K 块记录和真实 SmolLM projection 搜索；没有扩展 multi-node、split-K、任意 BK、多核或新物理布局。

## 1. 已有能力与本轮补齐位置

已有能力：

- guarded rectangular-tail frontend 已能识别 A/B 每轮清零、M/N/K 边界 guard、输出 guard，以及 logical M/N/K 和 tail 数量。
- exact K 的 TileLang K loop 已能识别显式 `Acc16 -> Acc32 -> GEMM -> Acc32 -> Acc16`，但它要求 `K = trips * 256` 且 block grid 必须精确覆盖 M/N。
- planner 已有 256 宽升序 K slice、K tail、ping-pong MM_ACC、zero padding upload 和 C runtime 的 `co_k_tile_length`。
- C verifier 已能拒绝 primitive K>256 和相关布局错误。

本轮连接和补齐：

- 增加 guarded M/N/K tail 源的显式 per-K FP16 形式；guarded tail 不再只能表达最终一次 FP32->FP16 收窄。
- ragged extractor 根据实际 copy 结构区分 `explicit_per_k_fp16` 与 legacy `fp32_final_narrowing`，不再需要也不信任“tail supported”字样。
- M/N block 角色继续从 A/B/C 实际索引表达式推断，canonical 和 swapped grid 都通过；一维 extent-1 block 轴也允许带或不带 block 变量。
- schedule 契约现在要求显式舍入字段；缺失、`fp32_final_narrowing`、单 K tile 标记配多 K 块都会拒绝。
- 记录 logical K、`k_blocks` 起止/有效长度、tail 有效长度、物理 K padding、舍入点，以及 planner 每个任务的 `k0/kt/k_physical_extent/k_padding`。
- 现有 sealed package 的 default plan 与当前代码生成的 default candidate 在 bindings、publish 矩形和 K task 几何上完全一致；plan hash 不同只因当前 planner 增加了 `k_physical_extent/k_padding/source_k_rounding` 元数据。证据：`evidence/default-relationship.json`。

## 2. K=576、K=1536 的真实 TileLang 源与识别结果

真实源：

- K=576 尾块形式：`tests/fixtures/tilelang_ragged.py::ragged_tail_f16`
- 模型搜索使用：`models/tilelang_matmul.py::make_matmul_tiled_tail_prim`
- K=1536 完整块形式也由 exact K loop 覆盖：`models/tilelang_matmul.py::make_matmul_tiled_prim`

K=576 识别结果：

```json
{
  "source_k_rounding": "explicit_per_k_fp16",
  "source_accumulator_dtype": "float16",
  "logical_k": 576,
  "k_blocks": [
    {"index": 0, "k0": 0, "extent": 256},
    {"index": 1, "k0": 256, "extent": 256},
    {"index": 2, "k0": 512, "extent": 64}
  ],
  "tails": {"M": 13, "N": 33, "K": 64},
  "rounding_points": "after_each_k_block"
}
```

K=1536 识别结果：

```json
{
  "source_k_rounding": "explicit_per_k_fp16",
  "source_accumulator_dtype": "float16",
  "logical_k": 1536,
  "k_blocks": [
    {"index": 0, "k0": 0, "extent": 256},
    {"index": 1, "k0": 256, "extent": 256},
    {"index": 2, "k0": 512, "extent": 256},
    {"index": 3, "k0": 768, "extent": 256},
    {"index": 4, "k0": 1024, "extent": 256},
    {"index": 5, "k0": 1280, "extent": 256}
  ],
  "rounding_points": "after_each_k_block"
}
```

K=576 的 planner 任务：

```text
co.mm.f16.v1      k0=0    kt=256  physical=256  padding=0
co.mm_acc.f16.v1  k0=256  kt=256  physical=256  padding=0
co.mm_acc.f16.v1  k0=512  kt=64   physical=64   padding=0
```

K=257 的 tail 才产生物理 padding：

```text
co.mm_acc.f16.v1  k0=256  kt=1  physical=32  padding=31
```

完整证据：`evidence/k-tail-contract.json`，覆盖 K=1、255、256、257、511、512、576、1536。

## 3. 逻辑 K 块到物理任务/布局

- 源逻辑范围与实际物理分配范围分开记录。`source_k_loop.k_blocks` 是逻辑有效范围；planner 的 `k0/kt` 和 buffer/view 的 `physical_cols` 是物理分配范围。
- A upload 使用 `input_layout(rows, kt)`：physical cols = `round_up(kt, 32)`，upload 前对整块 `memset` 为零。
- B/weight upload 使用 `weight_layout(cols, kt)`：physical cols = `round_up(kt, 32)`，同样从零填充。
- K=576 最后 64 个 K 元素对应 physical cols 64；K=257 的最后 1 个元素对应 physical cols 32，其中 31 个是 padding，不参与数学结果。
- 每个 K slice 的任务保持升序 `k0`；MM_ACC 的 prior/output 是 ping-pong buffer，不能和输出别名。
- 输出 tail 仍由 M/N 逻辑矩形 guard 覆盖；publish 只发布有效 `rows/cols`。

K=257 tail view 证据：

```text
A tail: rows=13 cols=1 physical_rows=16 physical_cols=32 padding_state=zero
B tail: rows=33 cols=1 physical_rows=64 physical_cols=32 padding_state=zero
```

## 4. tail 和舍入正反例

正向：

- K=576 的 source 每个 K 迭代都执行 `Acc16 -> Acc32`、GEMM、`Acc32 -> Acc16`。
- K=576 最后 64 个元素参与计算：把 A[:,512:576] 置零后设备输出改变，且始终匹配独立 per-K oracle。
- K=257 的 tail 有效长度为 1，物理 padding 为 31。
- 交替输入两轮/三轮执行，输出随输入改变，不继承上一轮 scratch。
- canonical 和 swapped block grid 都能通过角色推断并生成相同 K 任务几何。
- B 作为 input 的 guarded tail 也能保持 dynamic-B 身份，不产生常量。

反例：

- 缺失 A/B tail zero fill：`RK_E_NUMERIC_CONTRACT`。
- 输出未 guard：`RK_E_UNSUPPORTED`。
- guard 差一或换轴：`RK_E_SHAPE`。
- A tail 索引偏移 1：拒绝 “canonical guarded tail indexing”。
- K>256 仅最终 FP32 收窄且带 schedule：`RK_E_NUMERIC_CONTRACT`。
- annotation 写 `fp32_final_narrowing` 但源实际 per-K：`RK_E_NUMERIC_CONTRACT`，annotation 与 proven source 不一致。
- 非 256 源 K block：拒绝。
- schedule 缺 rounding 字段、K blocks 不连续、单 K tile 标记描述多块：拒绝。

测试覆盖：

- `tests/test_frontend_ragged.py`：8 个代表 K、canonical/swapped、dynamic B、padding view、全部 tail 负例。
- `tests/test_ragged_execution.py`：K=257、576、1536、K=256/257 边界、最后 64 元素参与、交替输入。
- `tests/test_schedule_contract.py`：严格 schedule K 分区、scheduled default 与 legacy geometry 等价。
- `tests/test_rounding_semantics.py`：per-K FP16 与最终一次 FP16 可区分。
- host Python：474 collected，无失败，1 个 slow-test skip。
- C：33/33 ctest 通过（Debug host build；增加 K=576/1536 K-slice 覆盖）。

## 5. 真实 projection 搜索覆盖与 winner

固定：真实 SmolLM2-135M-Instruct 权重，捕获的 calibration activation。模型 prefill plan 绑定 M=256，因此 45 行真实捕获激活按模型 bucket 零填充到 256 行；decode QKV 为真实 M=1。

| case | 几何 | 候选 | 合法 | default | 最佳非 default | default/最佳 | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| prefill_qkv | M256 K576 N960 | 16 | 11 | 6.478 ms | m256_n128_n 7.526 ms | 0.861 | default |
| prefill_gate_up | M256 K576 N3072 | 16 | 5 | 18.435 ms | m256_n128_n 26.641 ms | 0.692 | default |
| prefill_down | M256 K1536 N576 | 16 | 11 | 9.816 ms | m44_n192_n 10.461 ms | 0.938 | default |
| decode_qkv | M1 K576 N960 | 6 | 6 | 0.800 ms | m1_n256_m 0.811 ms | 0.987 | default |
| lm_head_shard0 | M256 K576 N4096 | 16 | 5 | 30.254 ms | m256_n128_n 31.465 ms | 0.962 | default |
| attention_qk_prefill | M768 K64 N256 | 16 | 11 | 2.194 ms | m256_n128_n 2.368 ms | 0.927 | default |
| attention_qk_decode | M3 K64 N256 | 6 | 6 | 0.230 ms | m256_n128_m 0.304 ms | 0.757 | default |

总计：92 个候选，55 个通过 C preflight 和板端输出验证；37 个被保留原因拒绝，主要是 “too many buffers share one slot”，另有单 block 源索引形式在修复前被拒绝、现已覆盖。

关键结论：**default 在全部 7 个 case 获胜，本轮不宣称 M/N schedule 的模型性能收益。** 真实算子进入搜索空间这一点已成立：QKV、gate/up、down、lm_head 和 attention QK 都有实际执行不同任务几何/发布顺序的合法非 default 候选。

搜索身份：policy SHA-256 `4310d7f6949573859e0400dd034dab351c170576141dcdd416aba16622eac106`。

## 6. 独立确认、新进程重放和模型固定验收

- 独立确认：`confirm-all` 对 7 个 case 记录 default_wins，没有把 noisy 非 default 候选升级。
- 新进程 replay：7/7 case pass；每个 case 都在同一 workspace 中交替执行真实输入与派生输入，输出均匹配独立 oracle，且预期改变时输出确实改变。
- replay 身份：policy、package manifest、weights、library、plan SHA、schedule 和 cache key 全部校验。
- V5 默认路径：56/56 gate，complete PASS。
- V5 手动接入：用已验证的非 default `m256_n128_n` 对 QKV、gate/up、down、lm_head prefill 做受约束 override，102 个 plan key 的 base/selected plan hash、binding/constant role/shape、profile 和 task template 全部校验通过；V5 仍为 56/56 gate。该 override 是 identity/equivalence 测试，不是 autotuner winner。
- V5 spec、阈值、calibration 和 package 权重身份均未改动。

## 7. 身份与证据路径

代码：

- HEAD/code commit：`a3ac0ad7324c2da7b57c3269b5b0df8881031638`
- 远端 `librknpu_plan.so` SHA-256：`f9ed96ae2f1c38a941ac8a0e379026d8ef6e67ae9dcd62e4aafc11143d5d8419`
- package manifest SHA-256：`b923fc5b427cc2ee6f0f88e7d096444a7ab44babdfbd79cc64fe07f5771548be`
- weights SHA-256：`ff04cb6f25633772987ff8cfc66e65046004034e526cf1d6d6bb615333b945bf`

调优：

- `tuning/board/policy.json` SHA-256：`4310d7f6949573859e0400dd034dab351c170576141dcdd416aba16622eac106`
- `tuning/board/tune-run.json` SHA-256：`d248eb3253198c82d2ced3fa65b45d5d752beb479b7feb98f08eabdd8496cb9e`
- `tuning/board/tune-confirm.json` SHA-256：`09b3f94b0cba94230ec2b45b4b121a66a65169a8b86fb16a1c1a8c146f299632`
- `tuning/board/tune-replay.json` SHA-256：`72a659659e514a96f2225e461ef32002e36c6608c753700e489c2bcf29c9a537`

模型 override：

- manual policy SHA-256：`0afcbb1472ce4f80089ff4c16b614f2f8064d1509df19d2eac70c2f4c17ea502`
- default effective identity SHA-256：`0f6faa3e90611c94b6fcfce5708a54bf134db754c4c88a077fbf5a2d1a49a222`
- manual effective identity SHA-256：`f9f0f9c849e8374e22a0fe717c5ac3a9bcc63f3a2f398749eb3bf441ec2caad3`

V5：

- default verify summary SHA-256：`0348dc147c1095f4fc5c0dc01662b9ae05cc8f80798cb779c8e0669e7c2fb51a`
- default acceptance bundle SHA-256：`e4988a9a7ba8785290b66815f5dbf0e17b9cf760523002e3b2d44e66407a9fc1`
- manual verify summary SHA-256：`17ee07ee38abe6758fd4aad67cce556d872706d0ed1f6de1dbcd96c0731c4512`
- manual acceptance bundle SHA-256：`40abe4e1bfe36eaeb9778e4280c55156739c66579dbccf4d68b7b51ab90ba3fc`

配套文件：

- `evidence/k-tail-contract.json`：K 分区、任务、view、padding，以及 scheduled default 与 legacy geometry 等价。
- `evidence/tuning-summary.json`：逐 case 候选、拒绝原因、winner、确认/replay/history。
- `tuning/board/plans/manifest.json`、`tuning/board/plans/*/*.rkplan.json`：92 个候选计划。
- `model-acceptance-default/`、`model-acceptance-manual-override/`：两套 V5 固定验收。
- `model-override/`：default/manual effective identity 与 102-key 受约束 override policy。

## 8. 进入最小两节点调度前仍缺的具体能力

1. 中间 tensor 的 schedule-aware tile map：每个 producer tile 必须能证明被目标 consumer 的 M/N/K 矩形精确覆盖。
2. producer/consumer 的 K tail 对齐：上游最后 `kt` 如何映射到下游的 K tile 起点和有效长度。
3. 中间 buffer 的 scratch 生命周期和 ping-pong 复用证明：跨节点的 accumulator、tail padding 和 publish 顺序。
4. 跨节点依赖/提交分组：当前 schedule 只作用于单 GEMM，staged/multi-node 显式 schedule 仍被拒绝。
5. 两节点数值 oracle 和负例：中间 tail padding、错位 tile、缺失中间清零、不同 K 舍入点。
6. 多节点预算与 C verifier：需要证明中间 tile 数、buffer slot 和任务链上限，不只复用单 GEMM 的预算。
7. 模型级两节点策略选择和独立确认；本轮 manual override 只证明单 GEMM 计划能接入模型，不证明两节点收益。

仍未支持：split-K、任意 BK、swizzle、async pipeline、多核 schedule、device KV、RKNN 对照。
