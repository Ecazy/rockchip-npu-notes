# 二维切分：输出行 × ic（§7.34）

回答 handoff §8 的下一入口①：「ic 切分与行切分组合」——当**单个 ic 组的一整幅输入**仍超出
可用 data bank 时的形态。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp2 `bash run_tile_ic_split_probe.sh`（日志 `tile_ic_split_probe.txt`）。

## 机制

`--ic-groups G` 与 `--tile-*` 现在**可组合**（`min_conv_exp.c`）：

1. **组外层**（§7.33）：按 32 对齐切 ic；组输入 = 全局输入的连续平面区间（无需拷贝）；
   权重按组重打包描述符；每组写独立 partial 输出面。
2. **组内层**（§7.31）：按输出行切 tile；tile 的输入 stage **只含本组的平面**
   （源 = 组首平面 + tile 行区间），`pad_top` 相应下调；tile 输出 scatter 回**本组**的
   partial 面（互不相交，无需累加）。
3. **收尾**：host 用 fp32 累加各组 partial → fp32→fp16 写回全局输出。

预算：`max_rows_budget = (可用 data bank × 32768) / 组行字节`，其中组行字节 =
`iw × (组对齐通道/8) × 16`。顺带硬化：**激活 BO 现按几何自动放大**（全局输入可达数 MB 且
ic 切分需要 `(G+1)` 个输出面），旧默认 1 MB 在大 ic × 大 ih 下会静默踩内存（实测 segfault）。

## 结果（`tile_ic_split_probe.txt`）

**A) 同一几何三种策略**（`iw16 ic256 ih400` 1×1：全图 = 100 bank，单组 = 25 bank）

| 策略 | submits | OUT-HASH | RESULT |
| --- | --- | --- | --- |
| 不切分 | 1 | `d0fd2751` | OUT-OF-ENVELOPE（max\|diff\| 171） |
| 行切分 `--tile-auto` | 10 tile | `a452fec0` | **BIT-EXACT** |
| 纯 ic `--ic-groups 16`（实为 8×32，组内 13 bank > 11） | 8 | `f5ac557a` | MISMATCH（max\|diff\| 209） |
| **2D** `groups4 + tile-auto` | 3 tile × 4 组 = 12 | `a452fec0` | **BIT-EXACT** |
| **2D** `groups8 + tile-auto` | 2 tile × 8 组 = 16 | `a452fec0` | **BIT-EXACT** |
| **2D** `groups4 + tile-oh 100` | 4 tile × 4 组 = 16 | `a452fec0` | **BIT-EXACT** |

⇒ 行切分与三种 2D 切分 **hash 完全相同**（切分选择不影响结果）；纯 ic 在组内仍越界时结构性
算错，**必须二维**。所有切分 arm 的 `host fp32 累加 vs fp64 累加 = 0 ULP`。

**B) 组内 bank 够时纯 ic 即可**（`ic256 ih100`：全图 25 bank 越界，组 4 bank）
不切分 `b8ffce73` OUT-OF-ENVELOPE；`groups 4`/`groups 8` 均 `2762b03b` **BIT-EXACT**。

**C) 2D + `--in-ruler` 压力**（`ic96 ih400` 1×1）：2D（2 tile × 3 组）与行切分（4 tile）
**同 hash `dd33c35e` BIT-EXACT**（该几何下 partial 恰无量化损失；§7.33 已给出最坏 ≤1 ULP 的界）。

**D) 3×3 真卷积 + 深 ic**（`ic128 ih200` 3×3p1）：不切分 OUT-OF-ENVELOPE（545）；`groups4 +
tile-auto` 自动选 **1 tile × 4 组**（组 7 bank + 权重 2 ≤ 12）→ **ULP-EXACT**（4.8e-07）。

**E) 不回归**：`ic16 3x3p1 ih16` = `533d6b86`、`ic128 3x3p1 ih16 groups4` = `3606092d`
（与 §7.31/§7.33 记录一致）。

## 结论

- **能力闭包完成**：任意形状的 fp16 卷积现可按「行 × ic」二维切分，使每个 submit 的
  `输入 tile + 权重 bank ≤ 12`；结果与不切分/行切分一致（BIT-EXACT 或 ≤1 ULP）。
- **切分选择不影响正确性**（组数/行数不同的 2D 与纯行切分同 hash）；代价是 submit 数与
  host 累加（仅 ic 维需要累加）。
- 仍受「单行字节 > 可用 bank × 32768」限制（需按宽度/列切分）——已记录为后续方向。

## 产物

- `experiments/min_conv_exp.c`：`--ic-groups` 与 `--tile-*` 组合、激活 BO 几何自适应。
- `experiments/run_tile_ic_split_probe.sh`、`tile_ic_split_probe.txt`。
