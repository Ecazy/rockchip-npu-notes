# vendor `FEATURE_GRAINS` 的精确口径：`grains = ih + kh + pad_top`（§7.32）

> §7.31 只给出「grains 必须够大」的**充分**口径（tinygrad 的 `max(80, ⌈2·32768/row_bytes⌉+1)`），
> 并留下「vendor 实测只有 2–20，精确规则待反解」。本节用仓库内全部 hook trace 反解出
> vendor 的精确规则，并在板上验证。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：`python3 experiments/analyze_vendor_grains.py --all`（产出 `grains_tasks.csv`）、
板端 exp2 `bash run_vendor_grains_probe.sh`（产出 `grains_probe.txt`）。

## 结论

**`CNA_CONV_CON2.FEATURE_GRAINS = ih + kh + pad_top`**（`ih` = 该 task 的
`CNA_DATA_SIZE0.DATAIN_HEIGHT`，`kh` = `WEIGHT_SIZE2.WEIGHT_HEIGHT`，
`pad_top` = `PAD_CON0.PAD_TOP`）。

- **离线 85/85 task、32 个唯一几何全精确吻合（误差 0）**，见 `grains_tasks.csv`：
  1×1（`iw8 ih8 → 9`、`iw1 ih1 → 2`）、1×2/2×1 矩形（`18/17`）、3×3 pad1（`20/13/12/9`）、
  attention matmul 形态（`iw16 ih16 kw16 kh1 → 17`）、1 行 feature（`ih1 → 2`）全覆盖；
  **与 `kw`、`ic`、`oc`、stride、bank 数无关**。
- 与既有外部结论的关系：rk3568 RE 的 `FEATURE_GRAINS = kh + strY·(L−1)` 在本批
  （`strY=1`）下即 `L = ih + pad_top`；mesa/旧发射器的 `50 + stride_y + 1` 是**小图 magic**
  （在 `ih` 小的时候恰好 ≥ 需求，故 74/74 回归能过）。
- `pad_top` 与 `pad_bottom` 在本批 trace 中对称，无法区分；`kh` 项说明这是「一次抓取要覆盖
  `kh` 行的卷积窗口」的意思。

## 板端验证（`grains_probe.txt`）

只切 grains（bank 交给 `--auto-cfg`），旧值 52 vs vendor 口径 vs auto：

| 用例 | `grains=52`（旧） | `grains=ih+kh+pad_top`（vendor） | auto |
| --- | --- | --- | --- |
| fp16 3×3p1 `iw16 ih54` | MISMATCH 9.9 | **58 → ULP-EXACT** | 130 → ULP-EXACT |
| fp16 3×3p1 `iw16 ih64` | MISMATCH 15.0 | **68 → ULP-EXACT** | 130 → ULP-EXACT |
| fp16 3×3p1 `iw16 ih100` | MISMATCH 12.5 | **104 → ULP-EXACT** | 130 → ULP-EXACT |
| fp16 3×3p1 `iw8 ih100` | MISMATCH 16.5 | **104 → ULP-EXACT** | 258 → ULP-EXACT |
| fp16 1×1p0 `iw16 ih64` | MISMATCH 7.0 | **65 → BIT-EXACT (0)** | 130 → BIT-EXACT |
| fp16 5×5p2 `iw16 ih64` | MISMATCH 14.3 | **71 → ULP-EXACT** | 130 → ULP-EXACT |
| fp16 3×3p1 `ic64 ih100`（7 bank） | ULP-EXACT | 104 → ULP-EXACT | 104 → ULP-EXACT |
| fp16 3×3p1 `ic128 ih24`（3 bank） | ULP-EXACT | 28 → ULP-EXACT | 80 → ULP-EXACT |
| int8 3×3p1 `ic128 ih64` | BIT-EXACT | 68 → BIT-EXACT | 80 → BIT-EXACT |
| int8 3×3p1 `ic160 ih16` | BIT-EXACT | 20 → BIT-EXACT | 80 → BIT-EXACT |

⇒ **vendor 口径在全部 12 例（含 1×1/3×3/5×5、int8/fp16、1–7 bank）都是充分的**；
旧值 52 只在「`ih` 大到使需求超过 52」的窄行（row 256–512 B）形状上暴露。
注意大 tile（≥3 bank）即使 52 也过 ⇒ 硬件需求**不是**简单地随 `ih` 单调，
`ih + kh + pad_top` 是 vendor 选择的（偏保守的）工作点。

## 发射器口径（本轮落地）

`min_conv_exp.c` / `wgrad_real.c` 的 `--auto-cfg` 现取**两条已验证规则的上包络**：

```
grains = max( 50 + stride_y + 1,                                    # 旧默认（小图安全）
              ih + kh + pad_top,                                    # vendor 精确口径（§7.32）
              max(80, ⌈2·32768 / row_bytes⌉ + 1) )                  # tinygrad 充分口径（§7.31）
grains = min(grains, 1022)                                          # 字段 10 bit
```

`--grains N` 仍可显式覆盖（A/B 用）；`--no-auto-cfg` 回到 `50+stride+1`。

## 不回归

`run_fp16_regression.sh` **79/0/0**、`run_regression_hard.sh`（int8）**74/74**、
`run_deconv_smoke.sh` **12/0**。

## 产物

- `experiments/analyze_vendor_grains.py`：trace → `grains_tasks.csv`（每 task 几何 + grains）。
- `grains_tasks.csv`：85 行原始抽取（可离线复核 100% 吻合）。
- `experiments/run_vendor_grains_probe.sh`、`grains_probe.txt`：板端 A/B 日志。
- `experiments/min_conv_exp.c`、`experiments/wgrad_real.c`：auto 口径取上包络。
