# deconv 边界口径修正（§7.27，2026-09-30）

结论来源：`deconv_boundary.log`（板端 Orange Pi 5 Plus，RKNPU driver v0.9.8）。
复跑：板端 exp2 目录 `./run_deconv_boundary_probe.sh`。

## 结论

**「1×1 核 + S=2 deconv 失配」不是硬件边界，而是 CPU 参考工具的转置卷积口径写反了方向。**
修正参考公式后，**任意核形状（1×1/2×2/3×3/矩形/5×5）× 任意 stride（S=2/4/8）× int8/fp16
全部逐位一致**，与精度无关。

- 正确口径：`ox = ix·S + pad − kx`，即 gather 形式 `tx = ox − pad + kx`（须被 S 整除，
  `ix = tx/S`），权重 tap 直接用 `kx`（**无翻转**）。
- 旧口径：`tx = ox + pad − kx` 且 `--deconv-flip` 翻转权重 tap。
  两者仅当 **`2·pad = kw−1`** 时重合——`3×3 pad1` 恰好满足，故此前只有 3×3 通过，
  1×1/2×2/矩形核全 MISMATCH。这解释了「为什么只有 3×3 对」。

## 证据

### 1) 硬件映射反解（`--in-ruler` + `--w-ones`，fp16 ic=16）

输入按元素下标作标尺（每像素 8 个 fp16、平面步长 `iw·ih·16 B`），`w=1` 时输出 = 各通道
标尺之和 = `128·pix + 1080`（`pix = iy·4 + ix`）。对 1×1 核 S=2 pad1（仅 tap 0）反解：

| 输出 (ox,oy) | hw | 反解 (ix,iy) | 旧参考 | 新口径 `(ox−pad)/S` |
| --- | --- | --- | --- | --- |
| (1,1) | 1080 | (0,0) | (1,1) | (0,0) |
| (3,1) | 1208 | (1,0) | (2,1) | (1,0) |
| (1,3) | 1592 | (0,1) | (1,2) | (0,1) |
| (3,3) | 1720 | (1,1) | (2,2) | (1,1) |
| (7,7) | 3000 | (3,3) | 越界→0 | (3,3) |

硬件把外边界 `ox=7`（及 `oy` 行/列）视作有效输入，旧参考判越界丢成 0，差异即源于此。

### 2) 核形状扫描（S=2 pad1 ic3 oc16）

修正前：kw1/kh1=186、kw2/kh2=566、kw3×kh1=390、kw1×kh3=481 mismatch，仅 3×3=0。
修正后全部 `RESULT: BIT-EXACT (fp16)`：kw/kh ∈ {1×1, 2×2, 3×3, 3×1, 1×3, 5×5, 5×1}。

### 3) 单 tap 判定权重序（kw=3 S=2）

`--w-tap kx:ky` 遍历 9 个 tap 全 `BIT-EXACT` ⇒ 权重 tap = 原始 `kx`，**无翻转**；
`--deconv-flip` 在修正后无语义影响。

### 4) 多 stride 与 int8 对照

S=2/4/8 的 3×3 与 1×1 全 BIT-EXACT；int8（同一口径）kw ∈ {1×1, 2×2, 3×1, 5×5} 全
`BIT-EXACT (floor)`（修正前 kw1=172/1024）。

## 改动

- `experiments/min_conv_exp.c`：fp16 / int8 两处 deconv CPU 参考由 `+ pad − kx` 改为
  `− pad + kx`，权重 tap 由 `(flip ? kw−1−kx : kx)` 改为 `kx`；`--deconv-flip` 保留为
  兼容开关（no-op）。
- `experiments/run_deconv_boundary_probe.sh`：新增（映射反解 + 核扫描 + tap 扫描 + int8）。
- `experiments/run_deconv_smoke.sh`：加入非 3×3 核用例（12/12 pass）。
- `experiments/run_fp16_regression.sh`：J 段加入 1×1/2×2/矩形核用例。

## 无回归

- `run_fp16_regression.sh`：**pass=78 / fail=0 / envfail=1**（修正前 74/0；新增 4 例 deconv）。
- `run_regression_hard.sh`（int8）：**pass=74 / fail=0**。
- `run_deconv_smoke.sh`：**pass=12 / fail=0**。

## 说明

- 探针：`experiments/run_deconv_boundary_probe.sh`（26/26 pass，含 ruler/核/tap/int8）。
- 修正前已知「共享边界」记录（int8 172/1024、fp16 186/1024）作废，原因见上。
