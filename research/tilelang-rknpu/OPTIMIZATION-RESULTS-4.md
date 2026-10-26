# 第四个优化包：QK/PV shared group 上传同步收口（实施报告）

日期：2026-10-26。默认路径保持 packed FFN + QK/PV `shared_group` /
`chain_shared`。本包只优化已验证独立 MM shared group 的上传准备阶段，不改变
布局、padding、上传区域、数值路径、独立 BO 约束、链几何域或多核策略。

## 0. 结论摘要

`upload_action()` 原先对每个切片分别执行 `rocket_bo_prep(write)` 和
`rocket_bo_fini()`。provider 契约确认：

- `rocket_bo_prep` 的 `dir` 只记录调用意图；实际行为是等待该 BO 的 WRITE
  usage fence，并做一次 whole-BO 的 `dma_sync_sgtable_for_cpu()`；
- `rocket_bo_fini` 做 whole-BO 的 device sync。

因此对**同一 BO 的连续、互不重叠切片上传**，可以安全地合并为：
一次 prep → 所有切片的清零/布局写入 → 一次 fini，且所有消费任务都在整个
上传段之后。实现只合并**连续 upload action 段**，不跨越任何 task/download。

默认开启合并，原逐切片路径通过 `RK_PLAN_UPLOAD_COALESCE=0` 显式保留为对照。

全量化 4 轮 × 32 次、逐轮交替、core 1：

| profile | coalesced p50 | per-slice control p50 | 加速比 |
|---|---:|---:|---:|
| Tiny ViT b8 | **11.721 ms** | 11.969 ms | **1.021×** |
| Tiny ViT b1 | **3.660 ms** | 3.718 ms | **1.016×** |

- b8 逐轮（coalesced/control）：`0.984 / 1.007 / 0.975 / 0.942`；
- b1：`0.986 / 0.967 / 0.984 / 0.988`；
- tasks/submits、upload/download 次数与字节完全一致；
- 默认采用 coalesced。

## 1. pack_us 拆分（逐轮聚合中位）

数据：`results/20261026T093305Z-upload/`。拆分为四个子区间，均属于
`pack_us` 内部，**不与 `pack_us` 相加**：

| b8 | control | coalesced |
|---|---:|---:|
| `pack_us` | 1122.9 µs | 597.6 µs |
| `pack_prep_us` | 300.4 µs | 41.1 µs |
| `pack_clear_us` | 87.5 µs | 54.5 µs |
| `pack_layout_us` | 410.4 µs | 408.9 µs |
| `pack_fini_us` | 273.0 µs | 56.3 µs |
| prep / fini calls | 134 / 134 | 10 / 10 |

| b1 | control | coalesced |
|---|---:|---:|
| `pack_us` | 159.5 µs | 117.5 µs |
| `pack_prep_us` | 48.7 µs | 25.7 µs |
| `pack_clear_us` | 15.8 µs | 11.4 µs |
| `pack_layout_us` | 56.3 µs | 57.8 µs |
| `pack_fini_us` | 30.9 µs | 16.6 µs |
| prep / fini calls | 22 / 22 | 10 / 10 |

QK/PV 两个 group plan 本身（b8）：

| plan | control pack | coalesced pack | control prep/fini calls | coalesced calls |
|---|---:|---:|---:|---:|
| qk | 425.2 µs | 158.2 µs | 64 / 64 | 2 / 2 |
| pv | 423.9 µs | 154.7 µs | 64 / 64 | 2 / 2 |

b1 的 qk/pv 从约 44.2/44.3 µs、8/8 calls 降到约 25.4/25.5 µs、2/2 calls。
layout 打包成本基本不变；收益主要来自消除重复的 BO prep/fini。

## 2. 实现边界与失败语义

- 只处理**连续 upload action 段**；段内按 destination BO 去重后一次 prep。
- 每个切片仍执行原有 `memset` 和 source→destination 布局打包，上传区域、
  padding、转置/广播标志与 bytes 计数不变。
- 所有切片写入完成后才 fini；随后才进入后续 task/download action。
- prep 失败：已 prep 的 BO 逐个 fini 清理，不提交任何消费任务。
- 布局/来源检查失败：清理已 prep 的 BO，不提交。
- fini 失败：不提交消费任务，返回错误；不清空或绕过现有 quarantine。
- 同一 workspace 在一次失败的 prep/fini 后可重试；mock 注入测试验证了
  `execs` 不前进且清理后可继续运行。
- 超过 128 个连续上传切片的异常大段退回/拒绝，不改变已验证产品几何
  （QK/PV 为 64 个切片）。

## 3. 数值与回归

- 使用 packed 冻结政策复验，**不重新校准、不放宽阈值**：
  `results/20261026T093427Z-numeric-replay/`，**PASS / 0 problems**；
  33 层门 + 5 个 profile logit 门全部通过。
- coalesced 与 per-slice 路径在 mock 上逐位相同；测试同时覆盖：
  - 重复运行输入变化后每个切片重新打包；
  - prep/fini 调用数下降而 `upload_calls/upload_bytes` 不变；
  - prep/fini 注入失败不提交消费任务、清理后可重试；
  - ABI 扩展字段的 C/Python 尺寸与旧 V1 兼容。
- Host：261 pytest 全通过；30/30 ctest 全通过。

## 4. 资源与创建成本

- b8 runner setup 中位约 13.76 → 12.82 ms；b1 约 7.79 → 8.21 ms
  （b1 反向波动，采样量有限，不作为收益声明）。
- manifest load 约 30 ms（b8）/12.5 ms（b1），两臂同 payload。
- 上传字节、下载字节、task/submit 数均无变化；无新增常驻 BO。

## 5. 采用结论

- 默认保留 coalesced 上传准备：b8/b1 的 pooled p50 均有正收益，且数值路径
  与结构计数不变。
- 回退开关：`RK_PLAN_UPLOAD_COALESCE=0` 使用原逐切片 prep/fini 路径。
- 不扩展独立 MM chain 几何域，不引入多核，不推进通用 Softmax，不改变数值政策。
