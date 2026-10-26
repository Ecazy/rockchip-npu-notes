# 第五个优化包：上传策略契约修复与 QK/PV 布局打包快路径（实施报告）

日期：2026-10-26。默认组合保持 packed FFN + QK/PV `shared_group` /
`chain_shared`。本包不改变布局、padding、FP16 位模式、数值政策、独立 MM 链几何域、
多核或通用 Softmax。

## 0. 结论摘要

1. `run_upload_run()` 不再因内部 128 项数组上限拒绝合法大计划：超过上限时安全回退到
   原逐切片 prep/pack/fini 路径。新增 130 个连续上传 action 的合法 C 计划回归。
2. 上传策略改为 **workspace 创建时快照**：`RK_PLAN_UPLOAD_COALESCE` /
   `RK_PLAN_UPLOAD_FAST_PACK` 在创建后修改不影响已有 workspace；实际策略写入
   `rk_run_attribution.upload_coalesce/upload_fast_pack` 并由 Python `RunStats` 报告。
3. 合并优化只作用于已验证独立 MM shared group：单 submit、>1、≤32 个 plain MM task。
   其他计划一律走原逐切片上传路径。
4. 新增固定 QK/PV 16×16 CUBE/WEIGHT 布局打包快路径：
   - CUBE `[16,16] -> [16,32]` 按两个 8 列 bank 做连续 8 FP16 拷贝；
   - WEIGHT `[16,16] -> [32,32]`、`transpose_source` 直接转置写入；
   - 先 memset 整个目标 tile，padding 继续为零；
   - 其他输入走原通用循环。
5. 默认启用 fast pack。b8 布局耗时约 439 → 260 µs，端到端 b8 11.637 → 11.414 ms
   （1.0195×）；b1 3.706 → 3.663 ms（1.0117×）。仍按“小幅实测改善”表述：
   b8 四轮均为正，b1 有两轮基本持平。

## 1. 上传策略契约修复

### 1.1 超过 128 个上传的安全回退

`run_upload_run()` 先扫描连续 upload action：

- 若输入切片数超过内部 `RK_MAX_COALESCED_UPLOADS=128`，不再返回
  `RK_E_UNSUPPORTED`；
- 整段走原逐切片 `upload_action(manage_sync=1)`，保持原支持域；
- 新增 C 回归：一个 verifier 接受的合法计划包含 130 个连续 upload action，
  运行成功且 `upload_calls=130`、`task_count=1`。

### 1.2 创建时快照

- `rk_workspace_create` 读取并保存 `upload_coalesce` / `upload_fast_pack`；
- 运行路径只读 workspace 快照，不再读取环境变量；
- `rk_run_attribution` 追加实际策略字段，Python `RunStats` 暴露
  `plan_upload_coalesce/plan_upload_fast_pack`；
- 回归验证：先以 env=1/0 创建 runner，再修改 env，运行仍按创建时策略执行。

### 1.3 限定在已验证路径

合并仅当 `plan_is_independent_mm_group(plan)` 为真：

- 恰好一个 submit；
- 2..32 个 task；
- 全部为 `co.mm.f16.v1`。

其余计划保留原逐切片 prep/fini。该限制不缩小任何原有计划的支持域。

## 2. 布局打包快路径

固定几何来自已验收 QK/PV 的 16×16×16 probe：

- 输入 A：CUBE `rows=16, cols=16, physical_rows=16, physical_cols=32, atom=8`；
- 权重 B：WEIGHT `rows=16, cols=16, physical_rows=32, physical_cols=32, atom=32`，
  `transpose_source=True`；
- 仅在 `source_row_stride=16`、offset 为 0、无 broadcast、`elements=256` 时启用。

CUBE 按 `(c/8)` 两 bank、每行 8 个 FP16 连续拷贝；WEIGHT 保留转置语义，直接写
`base[r*32+c]=src[c*16+r]`。目标 tile 先完整清零，因此 padding 仍为零。

## 3. 板端结果

### 3.1 布局快路径（coalesced 均为开）

`results/20261026T094847Z-layout/`，4 轮 × 32 次，逐轮交替、core 1：

| profile | fast p50 | generic p50 | fast/generic |
|---|---:|---:|---:|
| Tiny ViT b8 | **11.414 ms** | 11.637 ms | **0.981×**（1.0195× speedup） |
| Tiny ViT b1 | **3.663 ms** | 3.706 ms | **0.988×**（1.0117× speedup） |

逐轮 fast/generic：

- b8：`0.985 / 0.965 / 0.972 / 0.969`，四轮一致改善；
- b1：`0.935 / 1.002 / 0.983 / 1.001`，两轮改善、两轮基本持平。

pack 分项（逐轮聚合中位，单次代表运行口径）：

| b8 | fast | generic |
|---|---:|---:|
| `pack_us` | 446.2 µs | 625.9 µs |
| `pack_layout_us` | 260.5 µs | 439.2 µs |
| qk `pack_layout_us` | 13.3 µs | 96.1 µs |
| pv `pack_layout_us` | 13.1 µs | 95.4 µs |

b1：`pack_us` 89.0 → 119.3 µs，`pack_layout_us` 32.1 → 56.9 µs（fast/generic）。

### 3.2 同步合并收益保持

前一轮 `results/20261026T093305Z-upload/` 的同步合并结论保留：
b8 11.969 → 11.721 ms（1.021×），b1 3.718 → 3.660 ms（1.016×）。
本轮 fast pack 与 coalescing 叠加，不重算或改写该 campaign 的比值。

## 4. 数值与失败回归

- packed 冻结政策复验：`results/20261026T094941Z-numeric-replay/`，
  **PASS / 0 problems**；未重新校准、未放宽阈值。
- 快路径与通用路径：mock 上 logits 及全部 trace 逐位相同，第二次不同输入运行
  也逐位相同。
- 覆盖：
  - fast/generic 打包字节与输出一致；
  - 输入变化后每轮重新打包；
  - coalesced vs per-slice 调用数变化但 `upload_calls/upload_bytes` 不变；
  - prep/fini 注入失败不提交消费任务且可重试；
  - >128 上传合法计划回退；
  - 环境变量在 workspace 创建后修改不改变策略；
  - ABI V1/V2 兼容与新字段尺寸。
- Host：263 pytest 全通过；30/30 ctest 全通过。

## 5. 采用结论

- 默认保留 sync coalescing 与 fixed-layout fast pack；
- 对照开关：`RK_PLAN_UPLOAD_COALESCE=0` 保留逐切片同步，
  `RK_PLAN_UPLOAD_FAST_PACK=0` 保留通用布局打包；
- 两者均在 workspace 创建时快照，修改环境不影响已创建 workspace；
- 结论限定为“小幅实测改善”，不宣称每轮、所有 profile 都必然更快。
