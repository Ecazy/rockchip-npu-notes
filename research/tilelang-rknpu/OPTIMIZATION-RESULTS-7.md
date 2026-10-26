# 第七个优化包：QKV 输入布局映射（实施报告）

日期：2026-10-26。默认路径改为 packed FFN + QK/PV `chain_shared` +
coalesced 上传同步 + fixed-layout fast pack + binding descriptor 复用 +
**mapped QKV attention input**。物化 q/k/v 的路径保留为显式 `_splitheads`
对照清单。

## 0. 结论摘要

- 新增显式、版本化的 `group_layout` 映射契约，QK/PV 的 shared group 上传
  直接读取原始 QKV 行块：
  - QK A 读取 Q 列窗口，B 读取 K 列窗口；
  - PV B 读取 V 列窗口并保留转置；
  - PV A 仍读取 softmax 产出的连续 probs 切片。
- 普通执行不再运行 `split_heads` Host action，也不再分配 q/k/v 三个转置数组；
  q/k/v 仅在 `capture=True` 时由 capture-only Host action 重新物化，属于诊断
  证据，不计入生产计时边界。
- 同 campaign `results/20261026T113808Z-qkvmap/`、4 轮 × 32 次：

| profile | splitheads control | mapped qkv | pooled |
|---|---:|---:|---:|
| Tiny ViT b8 | 10.784 ms | **10.760 ms** | **1.0022×** |
| Tiny ViT b1 | 3.521 ms | **3.428 ms** | **1.0270×** |

b8 逐轮 `1.038 / 0.975 / 1.016 / 0.981`，接近噪声边界；b1 四轮均为正。
因此采用该路径的主要依据是**实际消除了 Host 数组分配、转置复制和
split_heads 调用**，同时 pooled 墙钟不退化；不宣称 b8 每轮稳定加速。

## 1. 映射契约

清单 QK/PV 节点新增 `group_layout`（版本 1）：

```json
{
  "version": 1,
  "inputs": {
    "A": {
      "tensor": "qkv", "shape": [16,192],
      "axes": ["batch_token","qkv"],
      "row_slice": {"batch_axis":"bi","row_stride":16,"row_extent":16},
      "slice": {"head_axis":"hi","column_base":0,
                "column_stride":16,"column_extent":16},
      "transpose": false
    }
  },
  "outputs": {
    "Y": {"tensor":"scores_heads","index":["bi","hi"],
          "shape":[16,16],"axes":["query","key"]}
  }
}
```

- `row_slice` 只表达一个受限的 batch 行窗口（`row_stride` / `row_extent`），
  不是任意 stride。
- `slice` 表达 head 列窗口及转置关系；`transpose` 与 plan upload action 的
  `transpose_source` 必须一致。
- 输出必须显式声明 target tensor、完整 repeat index、shape 和 axes。
- 映射版本固定为 1；不支持的版本 fail closed。
- 每个 group plan 的 binding 名、顺序、shape、role 与 repeat-major 展开严格
  对照；每个 upload action 的 `src_row_stride`、`src_col_offset`、
  `transpose_source` 与映射逐个核对。
- Python loader 检查索引范围、行窗口、列窗口、容量和 shape；
  C verifier 独立检查最终 plan descriptor 的 source capacity、索引算术和
  溢出，不信任 Python 预计算。

## 2. 执行与数值

- QK/PV group 的输入数组是 `qkv` 的连续 batch-block view 和 probs 的
  head-slice view；runner 不复制 q/k/v。
- QK B 需要 K 的“行块 + 列窗口 → 权重 tile 转置”关系；PV B 需要 V 的
  列窗口 → 权重 tile 转置关系。两者均按已验收的 16×16×16 shared-group
  几何执行，未扩大链几何域。
- 对同一底层输入 buffer 的重复 binding，C 和 Python 的 finite 检查均在
  **一次运行内按 `(data, nbytes)` 去重**；不跳过任何不同输入，也不改变
  finite 拒绝语义。
- `capture_only` HostNode 在普通运行中被跳过；`capture=True` 时重新物化
  q/k/v，供 trace 层门使用。其 Host 时间不计入普通执行边界。

## 3. 板端与结构

同 campaign 4 轮 × 32 次、逐轮交替、core 1，结构计数一致：
`task_count 70`、`submit_count 8`、上传/下载次数和字节不变。

b8 representative C-side：validate 约 0.246 → 0.301 ms（更大源 binding 的
检查），但生产 `split_heads` Host action 被消除；整体 pooled 墙钟基本持平。
b1 pooled 墙钟 3.521 → 3.428 ms。

普通运行实际消除：

- `split_heads` Host action 调用；
- q/k/v 三个 Host 转置数组及其分配；
- q/k/v 到 QK/PV 前的中间引用解析。

capture 路径仍会物化 q/k/v，这是诊断边界，不是生产路径。

## 4. 数值与回归

- packed 冻结政策复验：`results/20261026T113650Z-numeric-replay/`
  **PASS / 0 problems**，未重新校准、未放宽阈值。
- 映射路径与 `_splitheads` 对照：
  - b8 和 b1 的 logits 逐位相同；
  - `q_heads/k_t_heads/v_heads/scores_heads/probs_heads/attn_heads` trace
    逐位相同；
  - 不同输入批次重复运行逐位一致。
- 负例覆盖：
  - head 列窗口越界；
  - batch row slice 越界；
  - transpose 与 plan action 不一致；
  - C plan descriptor 的 source column offset 越界由 C 独立拒绝。
- Host：267 pytest 全通过；30/30 ctest 全通过。

## 5. 采用与后续

- 默认：`tiny_vit_bN` 使用 mapped QKV；`tiny_vit_bN_splitheads` 保留原
  materialized 路径作为对照。
- B1/separate QK/PV 策略对照仍使用 `_splitheads` 清单；mapped group 是
  chain/shared 执行形态，不提供 B1/separate 的替代语义。
- 剩余主要成本仍是 C `submit_wait`/`download`、Host Softmax/SwiGLU/RMSNorm
  与 executor 残留编排。此次映射收益在 b8 接近噪声边界，后续若要继续提升
  应优先处理这些主项，而不是再增加映射分支。
