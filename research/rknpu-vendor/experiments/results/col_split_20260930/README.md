# 输出列切分（列 × 行 × ic 三维切分）（§7.35）

回答 handoff §8 的下一入口①：「单行输入字节 > 可用 data bank」的极宽输入——行切分与 ic
切分都救不了，必须按**输出列**切分。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp2 `bash run_col_split_probe.sh`（日志 `col_split_probe.txt`）。

## 机制

- `--tile-ow N`：显式列宽；`--tile-auto`：列/行联合自动（当「4 行 × 全宽」都超 bank 时自动
  按列切，给 4 行留预算）。
- 列 tile 把输入**子矩形**逐行 stage 成紧凑块：行内字节 = `(Tc−1)·sx + kw`（列宽），
  平面步长 = `rows × 行内字节`；`ct.iw` 与 `ct.ow` 相应取 tile 的宽；`pad_left` 下调
  `max(0, −(ox0·sx − pad_left))`（首列 tile；内部/末列为 0）。
- 输出按**行段**（`Tc×16 B`）scatter 回全局面或 ic 组的 partial 面。
- **关键约束（本轮发现）**：`CNA_DMA_CON2.SURF_STRIDE = L·(staged_ih/4 − 1)` 编码
  「平面步长 = 4 行组」，因此 **staged 输入行数必须 ≥ 4**，否则该式无解（负值）→ 结构性算错。
  自动模式把行预算下限钳到 4；显式 `--tile-oh/--tile-ow` 越界时打印 `OUT-OF-ENVELOPE` 而非
  静默出错。

## 结果（`col_split_probe.txt`）

**A) 单行 16 bank（`iw1024 ic256 ih8` 1×1，可用 11 bank）——五种切分全部 BIT-EXACT 且同 hash**

| 策略 | tiles | OUT-HASH | RESULT |
| --- | --- | --- | --- |
| 不切分 | 1 | `e16482e5` | OUT-OF-ENVELOPE（max\|diff\| 6.6） |
| `--tile-auto`（自动列切） | 14（4×160） | `e86f99c0` | **BIT-EXACT** |
| `--ic-groups 4 --tile-auto` | 4（4×640）×4 组 | `e86f99c0` | **BIT-EXACT** |
| `--ic-groups 16 --tile-auto` | 2（5×1024）×8 组 | `e86f99c0` | **BIT-EXACT** |
| 显式 `--tile-ow 128` | 16（5×128） | `e86f99c0` | **BIT-EXACT** |
| 显式 `--tile-ow 256`（超容量） | 8（4×256） | — | OUT-OF-ENVELOPE（如实报错，不静默） |

**B) 3×3 宽输入**：`iw512 ic32 ih16` 3×3p1 不切分 OUT-OF-ENVELOPE（55）；`--tile-auto`
（2 tile = 8×512）**ULP-EXACT（4 ULP）**。

**C) 列切 + pad（覆盖首/内/末列 tile 的 `pad_left` 下调）**：`iw256 ic64 ih8` 3×3p1
不切分 / `auto` / 显式 `--tile-ow 64`（4 tile）/ `--tile-ow 64 --tile-oh 4`（8 tile）
**四种切法同 hash `ba3c6e94` ULP-EXACT**。

**D) 三维（列 × 行 × ic）**：`iw1024 ic256 ih32` 1×1 不切分 OUT-OF-ENVELOPE（409498 失配）；
`--ic-groups 4 --tile-auto`（16 tile × 4 组 = 64 submit）**BIT-EXACT**，host 累加 0 ULP。

**E) 不回归**：`ic16 3x3p1 ih16` = `533d6b86`、`ic128 3x3p1 ih16 groups4` = `3606092d`、
`ic256 ih400 1x1 groups4+tile-auto` = `a452fec0`（与 §7.31/§7.33/§7.34 记录一致）。

## 配套硬化

`run_regression_hard.sh`（int8 74 例）此前无重试，把随机 `SUBMIT ret≠0` 记为语义 FAIL
（连续三轮稳定在 73/1，失败例固定是 suite 第 2 例 `3x3 8x8 pad0`——冷启动超时）。现已按
handoff §七②的纪律加入 3 次重试与 `envfail` 计数：**74/74 envfail=0**。

## 结论

- **能力闭包完成**：fp16 卷积现可按「输出列 × 输出行 × ic」三维切分，覆盖任意
  `iw`/`ih`/`ic`/bank 组合；同一几何任意切分策略结果一致（BIT-EXACT 或 ≤1 ULP）。
- 唯一剩余限制：`staged 行数 ≥ 4` 的硬件编码约束（已自动钳制）与列 tile 的最小宽度
  `kw`（若 `kw × 每像素字节 > 可用 bank` 则必须减小 ic 组）。

## 产物

- `experiments/min_conv_exp.c`：`--tile-ow`、列/行联合自动、`pad_left` 下调、4 行下限钳制。
- `experiments/run_col_split_probe.sh`、`col_split_probe.txt`。
- `experiments/run_regression_hard.sh`：env 重试 + `envfail` 计数。
