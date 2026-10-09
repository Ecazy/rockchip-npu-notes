# fp16 `WEIGHT_SIZE1` 必须 ×2（根因）+ 权重侧非硬限 + oc 切分（§7.36）

起因：核查 handoff §3.5 未闭合项「**更大的 fp16 oc（oc>64）**」。小输入 + 多 ic/oc 块的
fp16 3×3 卷积会**静默算错后半个 oc 块**（mismatch 集中在 `oc ≥ 32`，且随几何/次数抖动）。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp2 `bash run_fp16_ws1_probe.sh`（日志 `ws1_probe.txt`）。

## 根因（决定性）：`CNA_WEIGHT_SIZE1` 是字节数，fp16 必须 ×2

§7.17 只把 `WEIGHT_SIZE0` ×2，**漏了 `WEIGHT_SIZE1`**。`WEIGHT_SIZE1 = kw·kh·ic` 描述
「每个 kernel/tap 的权重字节」，fp16 每元素 2 B ⇒ 应为 `kw·kh·ic·2`。
判别（A 段，`iw8 ih8 3x3 pad0`）：

| 用例 | 旧 `--ws1 kw·kh·ic` | 修复（默认 ×2） |
| --- | --- | --- |
| `ic64 oc64`（旧 ws1=576） | **MISMATCH**（max\|diff\| 12-17） | **ULP-EXACT**（ws1=1152） |
| `ic96 oc32`（864） | MISMATCH | ULP-EXACT |
| `ic32 oc128`（288） | MISMATCH | ULP-EXACT |
| `ic128 oc64`（1152） | MISMATCH（max\|diff\| 5.2e4） | ULP-EXACT（2304） |

⇒ 「大 oc 静默算错」不是硬件上限，而是**发射器漏乘**。修复已落在
`experiments/min_conv_exp.c` 与 `experiments/wgrad_real.c` 的 fp16 分支
（`kw·kh·ic·(g_fp16 ? 2 : 1)`）。

## 修复后：fp16 的 ic × oc 覆盖（B 段，3×3 `iw8 ih8` pad0）

`ic ∈ {32,64,96,128} × oc ∈ {32,64,96,128,256}` **全部 ULP-EXACT**（含多 ic 块与多 oc 块）；
`oc=256`（= 32 个输出平面）也一次通过 ⇒ handoff「更大的 fp16 oc」**闭合**（到 256 已验证）。

## 旁得（修正 §7.31 的措辞）：权重侧**不是**硬限

- `3x3 ic192 oc192 iw4 ih4`：权重描述符 663552 B = **21 bank**，`WEIGHT_BANK=11`，
  仍 **BIT-EXACT**（max\|diff\| = 0）⇒ 权重从 DRAM **流式**读取，描述符超过 `WEIGHT_BANK`
  也能算对；CBUF 的 weight bank 字段不是硬门限。
- 因此 **每个 submit 真正硬的约束只有输入侧**：`输入 tile ≤ DATA_BANK × 32768 B`
  （§7.31 的结论保留，但「输入+权重 ≤ 12」应读作**输入 bank 用满时权重让位**的规划口径，
  不是硬失败条件；§7.34 的 `ic256 iw16 ih40` 失败是**输入 10 bank > 实际分配 8** 所致）。

## 新能力：oc 维切分（`--oc-groups G`）

按 16 对齐切输出通道，各组写**互不相交的输出平面**（⇒ **无需累加**，与行/列切分同性质）；
权重描述符按组打包（打进权重 BO 里全局描述符之后的共享 scratch 槽）。D 段：
`ic192 oc192` 的不切分 / `--oc-groups 4` / `--oc-groups 8` **结果一致（ULP-EXACT）**。
与 `--ic-groups`/`--tile-*` 可组合（组描述符按 (ic,oc) 组合现场打包）。

## 不回归

| 套件 | 结果 |
| --- | --- |
| `run_fp16_regression.sh` | **79/0/0** |
| `run_regression_hard.sh`（int8） | **74/74 envfail=0** |
| `run_deconv_smoke.sh` | **12/0** |
| `wgrad_real.c`（fp16 单层 / `--two-layer` / int8 基线） | 全部 `CLOSED-LOOP OK` |
| §7.35 列切分归档臂 `iw1024 ic256 ih8 tile-auto` | hash `e86f99c0` **与归档一致** |

## 产物

- `experiments/min_conv_exp.c`：fp16 `WEIGHT_SIZE1` ×2；`--oc-groups G`；组描述符从
  「每组预先打包」改为「共享 scratch 槽现场打包」（支持 ic×oc 组合）。
- `experiments/wgrad_real.c`：fp16 `WEIGHT_SIZE1` ×2（板端 exp3 重新编译后复验：
  fp16 单层 / `--two-layer` / int8 三个闭环 `CLOSED-LOOP OK`；int8 首跑 1 次随机 ETIMEDOUT，
  重试 3/3 通过）。板/仓 sha 一致（`wgrad_real.c 1bd8cae3…`、`min_conv_exp.c a4d0a1eb…`）。
- `experiments/run_fp16_ws1_probe.sh`、`ws1_probe.txt`。
