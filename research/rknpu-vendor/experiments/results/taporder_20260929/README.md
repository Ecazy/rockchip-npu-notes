# §7.15 权重 tap 顺序：核内转置根因（2026-09-29 第二十三轮）

结论一句话：**CNA 权重槽的核内顺序是 ky-major（kx 变化最快，`slot = ky*kw + kx`）**；
`min_conv_exp` 的 layout A 曾写成 kx-major（`slot = kx*kh + ky`），方形核下两者互为转置
而被对称 pattern 抵消，只有 `kw ≠ kh` 且均 ≥2 时才暴露成 MISMATCH。

## 1. 证据 A：vendor `.rknn` 权重 blob 字节级解码（离线，与 pattern 无关）

`openrknn` 把 `.rknn` 的 `weight_data` blob **原样**拷进 NPU 权重 BO（64 B 对齐，
见 `refs/orangepi5plus-npu/openrknn/src/openrknn_model.c` 的 `wt_blob_offsets` 构造），
所以 blob 就是硬件布局。三个 probe 模型用固定 numpy seed 生成，逻辑权重完全已知。

解码步骤（`../vendor_blob_decode.py`）：

1. 拟合 rknn-toolkit2 的逐输出通道非对称 int8 量化
   `q(w) = round(w·s + zp)`，`s = 255/(wmax−wmin)`，`zp = −128 − s·wmin`；
2. 每个 16 B 行 = `[oc][ic0..ic2, pad×13]`，`pad = round(zp_oc)`；
3. 用已知 `(oc, ic, kh, kw)` 的权重反查每个 tap block 对应的逻辑 tap。

| 模型 | blob | 解出的 slot 顺序 | ky-major | kx-major |
| --- | --- | --- | --- | --- |
| conv 3×3, oc4, ic3（seed 7） | 576 B = 9 tap × 64 B | (0,0),(0,1),(0,2),(1,0)…(2,2) | 一致 | 一致（方形下两约定重合） |
| rect kw2 kh1（seed 11） | 512 B = 2 tap × 256 B | (0,0),(0,1) | **一致** | t=1 被否证 |
| rect kw1 kh2（seed 11） | 512 B = 2 tap × 256 B | (0,0),(1,0) | **一致** | t=1 被否证 |

另外确认 `blob 长度 = kw·kh·oc·16`，`ic` 原子 16 B，ic 越界槽填 `round(zp_oc)`
（不是 0）——这同时解释了 `.rknn` 里权重已按硬件布局打包，**无需再修 bo_dump 去抓
vendor 权重 BO**。

原始 `.rknn` 身份（本地 `experiments/vendor_models/`，gitignored；板上
`/mnt/ssd/workspace/bn_lower_probe/`）：

| 文件 | sha256 |
| --- | --- |
| `rect21_i8.rknn` | `3db3d67aa4ab2c1c568c1f0f2f27de4b1780fbfe9c4d92ca5d42eb7b9f419501` |
| `rect12_i8.rknn` | `79ce0e35c4a23b4d71a907d417e9d5cb8f167685b23adb63e8d7cac1e41a3f6c` |
| `conv16_i8.rknn` | `7b05b8c61e7a85a2e08a4731baa2ca51d8d55d3594050ba16ab2853b5510807f` |

## 2. 证据 B：真机 A/B（旧/新二进制同源，仅 `fill_weights` 布局 A 一行不同）

`16×16 ic3 oc16 valid --pat2 --auto-fix`；旧 = `min_conv_exp_old`（板上由 `sed` 反向
替换生成，`run_taporder_archive.sh` 内可复现）。

| 用例 | 旧（kx-major） | 新（ky-major） |
| --- | --- | --- |
| kw3 kh3（方形） | BIT-EXACT | BIT-EXACT |
| **kw3 kh3 `--asym-w`** | **MISMATCH 2682/4096（max 42）** | BIT-EXACT |
| kw3 kh3 `--w-tap 2:0` | MISMATCH 2624（max 3） | BIT-EXACT |
| kw3 kh3 `--w-tap 1:1` | BIT-EXACT（(1,1) 自映射） | BIT-EXACT |
| kw3 kh2 | MISMATCH 1357（max 8） | BIT-EXACT |
| kw2 kh3 | MISMATCH 1873（max 14） | BIT-EXACT |
| kw3 kh1 / kw1 kh3（1D） | BIT-EXACT | BIT-EXACT |

旧布局下方形核只在**对称**权重 pattern 下通过——这正是历史 "方形/矩形分界" 假象的来源。

## 3. 修复后的覆盖面

- 判别矩阵 15/15、扩展扫描 74/74（`04_extended_sweep.log`）：全部 20 个 `kw≠kh` 有序对
  （2..6）、方形 `--asym-w` 2..5、21 个 one-hot 单 tap（3×3 / 3×2 / 2×3）、`pad=1` 下
  25 种 `kw×kh` 组合、`stride=2` 下 4 种 `kw≠kh`——全部 BIT-EXACT。
- 硬化回归 74/74（`05_regression_hard.log`）：§7.11 的 74 例改用 `--pat2 --asym-w`
  重跑（旧 y 盲区 + 对称 pattern 只能证明输出面，不能证明 tap 语义）。
- deconv：`--deconv-flip` 6/6 BIT-EXACT（含 pat2+asym）；no-flip 是刻意的错误参考，
  在 y 盲区对称 pattern 下同样 MISMATCH，非回归。

## 4. 对既有结论的更正

- §7.14 item 3「方形（k×k）全 BIT-EXACT（含 `--asym-w` 双重验证）」**作废**：那只在旧
  y 盲区 pattern 下成立；`--pat2 --asym-w` 下旧布局 MISMATCH。方形核 tap 级语义此前
  从未被真正验证。
- §7.14 item 5 的「3×3 `--w-tap 1:1` 单 tap MISMATCH（±1~2）」实为 **kw2 kh3** 用例
  （板上 `exp2/tap23*/` 恰好 6 个 `w-tap` 文件 = 2×3）；旧布局复现 max abs diff 2，
  新布局 BIT-EXACT。3×3 的 tap 1:1 在两种布局下都过，不能作 tap 置换判据。
- §7.14 item 4 的「per-x 周期 4 / 成对全对列」不需要 CBUF 行取数假设解释，直接由核内
  转置解释。
- §7.1 的「方形 vs 矩形」硬件边界**作废**：矩形核无此限制，1D/方形/矩形全部 BIT-EXACT。

## 5. 文件

| 文件 | 说明 |
| --- | --- |
| `ab_evidence.txt` | 7 份板端日志 + blob 解码日志的拼接（可追踪） |
| `0*.log` | 同一批日志的分步原始件（gitignored，`run_taporder_archive.sh` 可复现） |
| `../../rknn_blob_dump.py` | `.rknn` FlatBuffer/权重 blob 提取器 |
| `../../vendor_blob_decode.py` | blob → 逻辑 tap 映射的反查（判据脚本） |
| `../../run_taporder_matrix.sh` / `matrix2.sh` | 判别矩阵 / 扩展扫描 |
| `../../run_ab_taporder.sh` | 旧 vs 新逐用例对照 |
| `../../run_regression_hard.sh` | §7.11 的 74 例 + `--pat2 --asym-w` 硬化回归 |
| `../../run_taporder_archive.sh` | 上述全部日志的一键归档脚本（板端） |
