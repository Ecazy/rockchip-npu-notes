# ic 维切分（跨 submit 累加）：非逐位一致（≤1 ULP），但可替代行切分（§7.33）

回答 handoff §8 的下一入口①：「真 CBUF 容量越界时 ic 切分能否替代输出行切分；**跨 submit
累加是否 BIT-EXACT**」。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp2 `bash run_ic_split_probe.sh`（日志 `ic_split_probe.txt`）。

## 机制（`min_conv_exp.c --ic-groups G`）

把 `ic_real` 按 **32 对齐**切成 G 组（`--ic-groups G` 是组数的下界：每组 ≥32 通道），每组
独立 emit + submit：

- **输入**：fp16 输入布局是 `[plane(8ch)][y][x][16 B]`，组 = 连续平面区间 ⇒ 组地址 =
  全局输入 + `ic0/8 · iw·ih·16 B`，**无需拷贝**（组内平面步长仍是 `iw·ih·16 B`）。
- **权重**：按组重新打包描述符（`fill_weights_f16` 用全局 `ic0` 偏移取逻辑权重），
  各组描述符在权重 BO 内独立放置。
- **输出**：每组写自己的 partial 区；最后在 **host 用 fp32** 逐元素相加、`fp32→fp16` 写回
  全局输出（同时与 fp64 累加对拍，报告累加本身引入的 ULP）。
- 仅 fp16（int8 的 partial 会被重量化，语义不对）；暂不与 `--tile-*`、BN/BRDMA 表组合。

## 结果（`ic_split_probe.txt`）

| 用例 | 不切分 hash | 切分 hash | 不切分 RESULT | 切分 RESULT |
| --- | --- | --- | --- | --- |
| `ic32 1×1` / groups 2 / 4 | `ae8c2925` | `ae8c2925` | BIT-EXACT | BIT-EXACT |
| `ic128 3×3p1` / groups 4 | `3606092d` | `3606092d` | ULP-EXACT(2) | ULP-EXACT(2) |
| `--in-ch-id --w-ones ic128` / groups 4 | `0096adc5` | `0096adc5` | BIT-EXACT | BIT-EXACT |
| `--in-ruler ic128 1×1` / groups 4 / 8 | `935d9e10` | `935d9e10` | BIT-EXACT | BIT-EXACT |
| `--in-ruler ic96 3×3p1` / groups 2 | `d15fb57a` | `abd09732` | **BIT-EXACT** | ULP-EXACT(1)，28/4096，max\|diff\|=16 |
| `--in-ruler ic96 3×3p1` / groups 3 | `d15fb57a` | `f53f0867` | **BIT-EXACT** | ULP-EXACT(1)，43/4096，max\|diff\|=16 |
| `ic256 ih40` 不切分（10+4 bank > 12） | `6ac4b9b2` | — | **OUT-OF-ENVELOPE**（max\|diff\| 623） | — |
| `ic256 ih40` / groups 4 且 groups 8 | — | `7cbe7291`（两者相同） | — | **ULP-EXACT(2)**，max\|diff\|=4.8e-07 |

- **不是逐位一致**：`--in-ruler ic96 3×3p1` 下切分与单次 submit 的 hash 不同，差异元素
  28–43/4096、`max|diff|=16`（该量级下 = **1 ULP**）；而同一用例的单次 arm 对 fp64 参考
  **BIT-EXACT** ⇒ 切分相对单次 submit 的偏离 **≤1 ULP**。根因：每组 partial 先被 fp16
  量化，再在 host 相加（单次 submit 的硬件是整段 fp32 累加）。
- **多数用例仍逐位一致**（默认 pattern、整数 pattern、`--in-ruler ic128 1×1`）。
- **host 累加本身不引入误差**：所有切分用例 `fp32 累加 vs fp64 累加: max ulp=0`。
- **真容量越界可救**：`ic256 iw16 ih40`（输入 10 bank + 权重 4 bank = 14 > 12）单发结构性
  算错；`--ic-groups 4`（或 8）后 **ULP-EXACT**，且组数 4 与 8 的 hash 相同。

## 与行切分的取舍

| 切分方式 | 是否需累加 | 精度 | 适用 |
| --- | --- | --- | --- |
| 输出行（`--tile-*`，§7.31） | **不需要**（输出面互不相交） | 逐 tile 与原结果同判据（ULP-EXACT/BIT-EXACT） | 首选；行可分时 |
| ic（`--ic-groups`，本节） | 需要（host fp32） | 通常逐位一致，最坏 **≤1 ULP** | 行不可分 / 宽行、或想保留整幅输出行 |

⇒ handoff 的「跨 submit 累加是否 BIT-EXACT」答案：**不是**（≤1 ULP，来自 partial 的 fp16
量化）；作为容量越界的替代路径**可用**（ULP-EXACT 级）。

## 产物

- `experiments/min_conv_exp.c`：`--ic-groups G`（+ `OUT-HASH` 输出摘要、`IC-SPLIT:` 累加统计）。
- `experiments/run_ic_split_probe.sh`、`ic_split_probe.txt`：板端探针与日志。
