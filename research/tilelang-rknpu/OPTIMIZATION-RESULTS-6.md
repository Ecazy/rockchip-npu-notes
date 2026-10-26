# 第六个优化包：Python/ctypes binding 复用（实施报告）

日期：2026-10-26。默认路径保持 packed FFN + QK/PV `chain_shared` +
coalesced 上传同步 + fixed-layout fast pack。本轮仅在 Python/FFI 编排层复用
PlanCall 的 ctypes binding 描述符，不改变 C 计划、布局、传输或数值路径。

## 0. 结论摘要

- **采用**：`PlanCall` 在创建时缓存静态 binding 描述符；每次运行只更新
  `data`/`nbytes`，不再为每个输入/输出重新分配 `RKBinding` 与 shape 数组。
- 同一 campaign、同源码、同 payload、逐轮交替，只切换 binding 复用开关：
  - b8：11.565 → **10.081 ms**（**1.147×**）
  - b1：3.648 → **3.312 ms**（**1.101×**）
- b8 逐轮 reuse/control：`0.863 / 0.866 / 0.877 / 0.869`，四轮一致；
  b1：`0.794 / 0.908 / 0.919 / 0.919`，四轮一致。
- tasks/submits、上传/下载次数与字节、创建的计划结构完全不变。
- packed 冻结政策复验：`results/20261026T100356Z-numeric-replay/`
  **PASS / 0 problems**，未重新校准或放宽阈值。

## 1. 成本定位与被放弃的候选

### 1.1 下载同步合并（未采用，已撤回）

在 `results/20261026T095856Z-download/` 中比较 shared-group 下载侧
per-slice `prep/fini` 与同 BO 一次 prep/fini：

| profile | coalesced p50 | per-slice p50 | 结果 |
|---|---:|---:|---|
| b8 | 11.340 ms | 11.414 ms | 0.65% pooled，逐轮 `1.013/0.964/0.978/1.018` |
| b1 | 3.669 ms | 3.666 ms | 基本持平，逐轮混合 |

C 侧 `download_us` 有下降，但端到端没有稳定收益，且增加了额外运行时分支与
ABI 字段。因此**撤回实现**，默认继续使用逐切片下载路径。

### 1.2 Python/ctypes 编排

最新默认成本中，plan-node 包络之外仍有明显的 Python/FFI 停留。`PlanCall.run_into`
原先每次调用为每个 binding 执行：

- `make_binding()`；
- 分配新的 `RKBinding`；
- 分配新的 `shape` ctypes 数组；
- 重新填写 `abi_version/struct_size/id/dtype/rank/nbytes`。

QK/PV shared group 每次分别有 96 个 binding（64 输入 + 32 输出），两次
group 加上其他 plan，使这部分重复分配成为可独立消除的对象。

## 2. 实现

`PlanCall.__init__(..., reuse_bindings=True)`：

- 根据 immutable plan bindings 预构造输入/输出 `RKBinding` 描述符及 shape 数组；
- 每次 `run_into()` 仍执行原有 `_prepare_inputs/_prepare_outputs`：
  dtype、shape、contiguity、writable 和输入 finite 检查全部保留；
- 仅把验证后的数组 `data` 指针与 `nbytes` 写回缓存描述符；
- 复用描述符只在本次同步 C 调用期间持有指针，调用结束不保留数组引用；
- `reuse_bindings=False` 保留原 `make_binding()` 路径作为对照；
- `HybridRunner` 在创建时接收该选项并传给所有 PlanCall；对照实验使用显式
  `HybridRunner(reuse_bindings=False)` 或独立 runner。

未删除任何输入 validation 或 finite 检查；未修改 C workspace、plan、上传布局、
同步、guard、发布或 quarantine 语义。

## 3. 同 campaign 端到端结果

`results/20261026T100309Z-pybind/`，4 轮 × 32 次，逐轮交替、core 1：

| profile | reuse p50 | control p50 | reuse/control | 加速 |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | **10.081 ms** | 11.565 ms | 0.872 | **1.147×** |
| Tiny ViT b1 | **3.312 ms** | 3.648 ms | 0.908 | **1.101×** |

- b8 吞吐由约 691.7 → **793.6 img/s**
- b1 由约 274.1 → **302.0 img/s**
- 结构计数、上传/下载字节完全一致；该收益来自 Python/FFI 编排，不是设备
  计算或桥接字节变化。
- 创建成本：b8 runner setup 约 14.4 → 14.9 ms，b1 8.1 → 7.3 ms；
  不作为收益声明，只记录同量级。

## 4. 数值与回归

- `results/20261026T100356Z-numeric-replay/`：
  - 使用 `models/numeric_acceptance_packed.json` 既有冻结容忍限；
  - **PASS / 0 problems**；
  - 33 层门 + 5 个 profile logit 门全部通过；
  - 未调用重标定工具。
- 新增 binding 复用逐位回归：reuse on/off 的 logits 与
  `h1/norm2/gate/up/act/ffn/h2/logits` 全部 `array_equal`；
- 重复使用同一 runner 处理变化输入仍通过；
- Host：264 pytest 全通过；30/30 ctest 全通过。

## 5. 采用结论

- 默认开启 `reuse_bindings=True`；
- 对照开关：`HybridRunner(..., reuse_bindings=False)`；
- 这是创建时固定的 runner 配置，不在运行中读取环境或改变状态；
- 同 campaign 口径下 b8 1.147×、b1 1.101×，因此采用；
- 不同 campaign 的绝对延迟不作为累计加速结论。

## 6. 剩余主要瓶颈

当前 b8 默认（同 campaign 参考）的主要未消除项仍是：

- C workspace 的 `submit_wait` 与 `download`；
- Host Softmax / SwiGLU / RMSNorm 算术；
- plan-node 之外的 executor/Python 残留编排；
- guard、pack 与少量布局搬运。

下一项若要继续，需要在“更多 Python/FFI 编排收敛”与“Host 算术/相邻 Host
action 融合”之间选择；下载同步合并已因无可复现端到端收益撤回。
