# CNA 输入 feature tile 32KB 包线（§7.30，修正 §7.29；2026-09-30）

> ⚠ **本目录结论已由 §7.31 再修正（同日）**：这里的「32768 B 硬包线」是**旧发射器硬编码
> `DATA_BANK = 1`**（CBUF 每 bank 32 KB）的表现，「≤32 KB 内高输入失效」则是
> **`FEATURE_GRAINS` 写死 `50+stride+1`** 的表现——两者都是**配置**，不是硬件硬包线。
> 真包线只有 `ceil(in_bytes/32768) + weight_banks_need > 12`（CBUF 共 12 bank）。
> 结论与证据见 `../fp16_cbuf_bank_20260930/README.md`，探针 `run_fp16_cbuf_bank_probe.sh`。
> 本目录探针已改为**历史对照臂**（全部命令带 `--no-auto-cfg`，复现本节旧配置）；
> 默认（auto-cfg）下这些用例均已通过。

> **本目录结论已由 §7.30 修正。** 原 §7.29 认为包线是「tap 形状区分 `c.ic` 上限
> （多 tap≤64、1×1≤256）」，但其多 tap 探针固定 `iw=ih=16`、1×1 探针固定 `iw=ih=8`，
> tap 与 tile 尺寸未解耦。控制变量实验证明**真正变量是输入 feature tile 字节数**。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp2 目录 `./run_fp16_deep_channel_probe.sh`（可用 `FP16_DEEP_LOG=...` 覆盖日志名）。

## 结论

**单次 submit 的 CNA 输入 feature tile 字节数 `in_size = iw*ih*对齐通道*每元素字节`
必须 ≤ 32768（32 KB）；超出后硬件静默算错**（无报错、`SUBMIT ret=0`，仅结果大幅偏离，
max|diff| 可达数千）。fp16 每元素 2 B、int8 每元素 1 B，二者同受此限。

决定性对照（3×3p1 fp16，固定 `iw=16` 只扫 `ih`，输入字节 = `16*ih*128*2`；
已用大激活 BO 排除输入/输出区重叠，见「BO 重叠」小节）：

| 配置 | in_size | 结果 |
| --- | --- | --- |
| `ic128 ih8` | 32768 B | ULP-EXACT（max\|diff\|=4.8e-07） |
| `ic128 ih9` | 36864 B | MISMATCH（max\|diff\|≈240） |
| `ic128 ih12` | 49152 B | MISMATCH |
| `ic128 iw8 ih16` | 32768 B | ULP-EXACT |
| `ic128 iw9 ih16` | 36864 B | MISMATCH |
| `ic16 iw64 ih16` | 32768 B | ULP-EXACT |
| `ic16 iw65 ih16` | 33280 B | MISMATCH（max\|diff\|≈29） |

- **阈值精确为 32768 B**：`ic16 iw64 ih16`（恰 32768 B）净、`iw65`（33280 B）即坏。
- **`iw` 与 `ih` 对称**。
- **tap 形状无关**：同一 `iw/ih/ic` 下 1×1 与 3×3 同为 OK/BAD
  （`ic128 iw16 ih8`：1×1 BIT-EXACT、3×3 ULP-EXACT；`ih9`：两者皆 MISMATCH）。
- **小 ic 亦可越界**（证明由 tile 字节而非 `c.ic`）：`ic64 iw16 ih24`、`ic32 iw16 ih48`
  均 49152 B → MISMATCH。
- **`c.ic` 用对齐/pad 后的跨度**：`ic_real=64 --ic-pad 96`（c.ic=96→49152 B）坏；
  `ic_real=96 --ic-pad 64`（c.ic=64→32768 B）ULP-EXACT。
- **int8 同限**：`ic128 iw16 ih16`=32768 B BIT-EXACT；`ic160`=40960 B / `ih17`=34816 B MISMATCH。
- **权重字节无关**：`ic64 oc64`（权重描述符 73728 B）仍 ULP-EXACT。

## BO 重叠（已排除的混淆）

emitter 旧默认激活 BO=64 KB、输出偏移 `OUT_OFF=32768`；当 `in_size>32768` 时输入区与
输出区**重叠**，会额外污染结果。已把默认改为**激活 BO 1 MB / 输出偏移 512 KB**，新增
`--act-size`/`--out-off` 旋钮，权重 BO 由 64 KB 增至 **1 MB**（容纳 `ic64 oc64` 的
73728 B 描述符，此前该例的 BIT-EXACT 疑因越界）。用大 BO 复测确认**边界仍恰为 32768 B**
（`ih8` 净、`ih9` 坏），重叠只放大失配幅度（`ic128 ih16` max|diff| 3769→546），非根因。

## 次级约束（未闭合）

在 ≤32 KB 内，**极高的输入亦会失效**（独立于字节）：fp16 `ic16 iw8` 在 `ih53` ULP-EXACT、
`ih54` MISMATCH；边界随 `iw` 轻微浮动（`iw12`→52/53，`iw16`→53/54，`iw8` 的 `ih48` 好 /
`ih56` 坏）。疑为行/stride 或输出 tile 的独立上限（emitter 简化布局亦可能是诱因），待定域。

## 证据（关键行）

```
（以下为默认硬化 BO：激活 1MB / 输出 512KB / 权重 1MB 的净结果）
3x3 ic128 iw16 ih8  (32768B)   ULP-EXACT      max|diff|=4.77e-07
3x3 ic128 iw16 ih9  (36864B)   MISMATCH       max|diff|=239.941
3x3 ic128 iw16 ih16 (65536B)   MISMATCH       max|diff|=545.566
1x1 ic128 iw16 ih8  (32768B)   BIT-EXACT      max|diff|=0
1x1 ic128 iw16 ih9  (36864B)   MISMATCH       max|diff|=105.314
3x3 ic16  iw64 ih16 (32768B)   ULP-EXACT      max|diff|=9.54e-07
3x3 ic16  iw65 ih16 (33280B)   MISMATCH       max|diff|=29.3457
3x3 ic64  iw16 ih24 (49152B)   MISMATCH       max|diff|=311.333
3x3 ic32  iw16 ih48 (49152B)   MISMATCH       max|diff|=156.583
3x3 ic16  iw16 ih64 (32768B)   MISMATCH       max|diff|=11.3613（次级「高输入」约束）
ic64 --ic-pad 96    (49152B)   MISMATCH       max|diff|=311.332
ic96 --ic-pad 64    (32768B)   ULP-EXACT      max|diff|=9.54e-07
3x3 ic64  oc64      (w 73728B) ULP-EXACT      max|diff|=9.54e-07
int8 3x3 ic128 iw16 ih16 (32768B) BIT-EXACT (floor)
int8 3x3 ic128 iw16 ih17 (34816B) MISMATCH
int8 3x3 ic64  iw16 ih32 (32768B) BIT-EXACT (floor)
```

## 已排除的修复方向（§7.29 遗留，仍有效）

`--f16-watom 16/8`、`--f16-nest 1/2`、`--f16-ocatom 32`、`--entries 2/4/8/12/16`、
`--ws1 864/1728/3456/216`、`--kernel-group N`（`CNA_CONV_CON2.KERNEL_GROUP` bits16-23）
均无法越过包线。⇒ 非 `WEIGHT_SIZE1` 缩放、非打包布局、非 KERNEL_GROUP 分组迭代控制。

## 实现

- `experiments/min_conv_exp.c`：包线判别改为 `in_size > 32768`（不再按 tap 形状），
  打印 `OUT-OF-ENVELOPE (input feature tile > 32KB, §7.30)`；激活/权重 BO 硬化，
  新增 `--act-size`/`--out-off`。
- 探针 `experiments/run_fp16_deep_channel_probe.sh` 重写为边界/对称/tap-control/小-ic/
  对齐/int8 六组，并对无 `RESULT` 行（submit 超时）自动重试至多 3 次。

## 意义 / 后续

- 真实模型里 fp16（及 int8）深 `3×3`（如 ic=128/256）**必须切分**使单次 `in_size≤32768`：
  **按输出行（或 ic 组）皆可**。vendor runtime 深卷积正是**按输出行拆 task**
  （ic128 iw16 ih16 拆成 h=8 的两 task，每 tile 恰 32768 B），而非按 ic 拆。
- 待验：跨 submit 的输出行/ic 切分累加是否 BIT-EXACT；§7.30 次级「高输入」上限的精确定域。
