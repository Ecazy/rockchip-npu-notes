# vendor DPU/CNA 任务的**位域语义**（§7.47）

新工具 `experiments/analyze_vendor_dpu_fields.py`：把 trace 里 task 的每个寄存器值按
`rnpu_regmap.h` 的 `*__MASK/__SHIFT` **逐位域解成命名语义**（自动匹配寄存器名最长前缀）。

## 一、attn（`attn_relu_vendor.jsonl` op1，34 entries，纯 DPU）

| 寄存器 | 值 | 解码 |
| --- | --- | --- |
| `DPU_S_POINTER` | 0xe | `POINTER_PP_MODE=1 EXECUTER_PP_EN=1 POINTER_PP_EN=1`（ping-pong 开） |
| `DPU_RDMA_RDMA_S_POINTER` | 0xe | 同上 |
| `DPU_FEATURE_MODE_CFG` | 0x81e5 | `SURF_LEN=64 BURST_LEN=15 OUTPUT_MODE=2 FLYING_MODE=1` |
| `DPU_DATA_FORMAT` | 0x24000001 | `OUT_PRECISION=1 IN_PRECISION=1 PROC_PRECISION=1`（fp16 = 1） |
| `DPU_DATA_CUBE_*` | — | `WIDTH=0(→1) HEIGHT=63(→64) CHANNEL=7(→8) NOTCH=31` |
| `DPU_BS_CFG` / `DPU_BN_CFG` | 0x53 | **全 bypass**（`BS/BN_BYPASS=1 ALU=1 MUL=1 RELU=1`） |
| `DPU_BS_OW_CFG` | 0x80007fe | `TP_ORG_EN=1 SIZE_E_2/1/0=7 OD_BYPASS=1` |
| `DPU_WDMA_SIZE_0/1` | 0x30007 / 0x10000 | `SIZE_C_WDMA=3 CHANNEL_WDMA=7 / HEIGHT_WDMA=1` |
| `DPU_EW_CFG`、`*_CVT_SCALE` | 0x383 / 1 | 无转换缩放 |
| `DPU_LUT_LO_SLOPE_*` | 0 | **LUT 未使用** |

⇒ **attn 的批量 task = 纯数据搬运/布局变换**（DPU WDMA、8 通道 × 64 行、fp16、**不做任何 ALU/MUL/RELU/LUT**）。
这正是 skill 关心的那类「小算子批量发射」；对「是 LUT 还是多项式」的回答是：**两者都不是（无数学）**。

## 二、transpose（`em0d/tr_transpose_p0321_fp16.jsonl` 首 task，112 entries，CNA+DPU）

| 寄存器 | 值 | 解码 |
| --- | --- | --- |
| `CNA_CBUF_CON0` | 0x1b | **`WEIGHT_BANK=1 DATA_BANK=11`**（§7.38 的 vendor 口径 R4：权重按需、余量给输入） |
| `CNA_CONV_CON1` | 0x120 | `PROC_PRECISION=2 IN_PRECISION=2`（**fp16**） |
| `CNA_CONV_CON2` | 0x20 | `FEATURE_GRAINS=2` |
| `CNA_CONV_CON3` | 0x9 | `X/Y_STRIDE=1` |
| `CNA_DATA_SIZE0/1` | 0x80001 / 0x70008 | `DATAIN_WIDTH=8 HEIGHT=1` / `CHANNEL=8`（→ `DATA_SIZE3 ATOMIC=1`） |
| `CNA_WEIGHT_SIZE0/1/2` | 0x2000 / 0x80 / 0x810040 | `WEIGHT_BYTES=8192` / `PER_KERNEL=128` / `WIDTH=8 HEIGHT=1 KERNELS=64` |
| `CNA_CBUF_CON1` | 2 | `DATA_ENTRIES=2` |
| `CNA_CVT_CON0..4` | 0xb / 0x10000×4 | `CVT_BYPASS=1 DATA_SIGN=1`，三条 scale 均为 1（**无转换缩放**） |
| `PC_BASE_ADDRESS` / `PC_REGISTER_AMOUNTS` / `PC_OPERATION_ENABLE` | 见 §7.46 | 链式推进（写下一个 task 的地址/长度） |

⇒ **transpose = 1×1 fp16「卷积形」算子的布局变换**（CNA 8→8 通道、8×1 空间），且**逐 task 换
输入/输出地址**（§7.46）；bank 划分复现了 §7.38 推出的 vendor 规则。

## 三、对我们模型的字段级复核

- `PROC_PRECISION=2 / IN_PRECISION=2` = fp16 ✓（与 §7.17「fp16 四处 precision = 2」一致）；
- `DATA_BANK + WEIGHT_BANK = 12`、`WEIGHT_BANK = ceil(权重字节/32768)` ✓（§7.38 R3/R4，**在同一条 vendor task 上再次命中**）；
- `CVT_BYPASS + scale=1` ⇒ 我们「identity 量化」用例的写法与 vendor 同源 ✓；
- `DPU_LUT_*` 全 0 ⇒ vendor 在这些布局/激活 op 上不用 LUT ✓（LUT 只在我们探过的 BS/BN 表路径出现）。

## 产物

- `experiments/analyze_vendor_dpu_fields.py`（位域解码器）
- `attn_task_fields.txt`（35 行）、`transpose_task_fields.txt`（113 行）


---

# 追加 §7.48：**我们的流 vs vendor 真值**的逐位域 diff

新工具：`min_conv_exp --dump-regcmd FILE`（导出本 task 的 regcmd 条目为 `0xOFF 0xVAL` 文本）
+ `experiments/compare_regcmd.py`（两份条目表逐位域对比，自动命名）。
对照物：`vendor_op3_1x1_i8.txt`（attn op3 首个 CNA 任务，54 entries）vs
`our_int8_1x1_iw16.txt`（我们 int8 1×1 iw16 ih16 ic16 oc16，130 entries，BIT-EXACT）。
完整结果：`regcmd_diff_vs_vendor.txt`。

## 汇总（126 个偏移）

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 完全相同 | **41** | CNA 立方体尺寸、`DATA_SIZE0/1`、strides、`CVT_CON*`（bypass+scale 1）、`DMA_CON*`、`FC_DATA_SIZE*`、`PAD_CON*`、`WEIGHT_BANK/DATA_BANK` 之外的 CBUF 字段… |
| 值不同 | 9 | 见下表 |
| **仅我们有** | **76** | 我们无条件写全套 DPU/RDMA（BS/BN/EW/LUT/PPU 旁路寄存器）⇒ **我们的流是 vendor 的超集** |
| 仅 vendor 有 | **0** | ✅ 没有 vendor 写了而我们漏掉的寄存器 |

## 9 处值差异

| 寄存器 | vendor | 我们 | 判读 |
| --- | --- | --- | --- |
| `CNA_CBUF_CON0` | `WEIGHT_BANK=2 DATA_BANK=10` | `11 / 1` | §7.38 已定域：vendor「权重按需、余量给输入」，我们反之；两者都满足 R2/R3，均可用 |
| `CNA_CONV_CON2` | `FEATURE_GRAINS=17` | `258` | vendor 取**下界** `ih+kh+pad=16+1+0=17`（§7.32 ✓）；我们 auto 取**上包络**（§7.31/§7.32）——都对，只是策略不同 |
| `CNA_WEIGHT_SIZE0` | `65536` | `256` | **对照物不是同一逻辑 op**：vendor op3 权重 65536 B = 256×256（16×1 kernel × 256 "kernels" × 256 B/kernel）⇒ 是 attention 的**矩阵乘类** CNA op；我们的是 1×1 ic16→oc16（256 B） |
| `CNA_WEIGHT_SIZE1/2` | 256 / `W=16 H=1 K=256` | 16 / `W=1 H=1 K=16` | 同上（两种自洽编码：我们的「K=输出通道数、每 kernel=ic 字节」；vendor 的「K=ic×oc」） |
| `CNA_DATA_SIZE2/3` | `DATAOUT_WIDTH=1` / `ATOMICS=16` | `16` / `256` | 输出宽度字段语义随 op 形状（我们的 1×1 输出 16 宽、原子数 256 = 16×16） |
| `CNA_FEATURE_DATA_ADDR` / `CNA_DCOMP_ADDR0` | 各自 BO 地址 | 各自 BO 地址 | 地址本就不同 |

## 结论

1. **缺口方向是「我们有、vendor 没有」**：76 个旁路/默认寄存器我们全写、vendor 不写 ⇒ 不存在
   「vendor 需要而我们没设」的寄存器（对这条对照而言）。
2. **语义差异都可解释**：bank 策略（§7.38）、grains 策略（下界 vs 上包络）、以及**对照 op 不同类**
   （vendor 的是 attention 矩阵乘类 CNA op，权重 256× 于 1×1 conv）。
3. 工具链：`--dump-regcmd` + `compare_regcmd.py` 可用于今后任何「我们的流 vs vendor 流」核对
   （例如拿到 vendor 的真 1×1 conv / 深 ic 卷积 trace 时直接跑同一对比）。
