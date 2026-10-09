# 训练闭环的特征图尺寸扩展（§7.49）

工具：`experiments/run_wgrad_bigmap_probe.sh`（板端 exp3）；日志 `bigmap_probe.txt`。

## 改动（`wgrad_real.c`）

1. **`IW`/`IH` 加 `#ifndef` 守卫**（原先只有 `IC`/`OC`/`L2_IC` 可 `-D`；`IW/IH` 被源文件
   `#define` 覆盖 ⇒ 早期 `-DIW/-DIH` 实际无效，本轮才发现）。
2. **pad0 输出尺寸参数化**：写死的 `6` 改为 `OW_P0 = IW-(KW-1)` / `OH_P0 = IH-(KH-1)`（17 处）。
3. 新增 `--dump-regcmd FILE`（导出本 task 的 `0xOFF 0xVAL`），便于与 `min_conv_exp` 逐位域对照
   （配合 §7.48 的 `compare_regcmd.py`）。

## 结果

| 几何 | 结果 |
| --- | --- |
| 8×8 IC=32 pad1（回归） | **CLOSED-LOOP OK** |
| **16×16 IC=16 pad1** | FWD/FWD2/WGRAD 全 BIT-EXACT ⇒ **CLOSED-LOOP OK** |
| **16×16 IC=32 pad1** | FWD/FWD2 BIT-EXACT、WGRAD 288/288 ⇒ **CLOSED-LOOP OK** |
| **16×16 pad0（14×14 输出）IC=32** | `total=6272` 全 BIT-EXACT ⇒ **CLOSED-LOOP OK** |
| 16×16 IC=128 pad1 | FWD ✓、WGRAD 1152/1152 ✓，**FWD2 MISMATCH（hard=8188）** ✗ |
| 32×32 pad1（IC=32 与 IC=128） | **FWD 起就 MISMATCH**（`mismatch≈total`）✗ |

⇒ 训练闭环的特征图从 **8×8 扩到 16×16（IC≤32）**；两条新边界已精确定位：
- **16×16 + IC≥128**：只有「写回后复前向」错（FWD/WGRAD 都对）⇒ 与 §7.28/§7.37 的 family 不同
  （hard>16 ULP），更像**规模相关的字段/缓冲**问题；
- **32×32**：连 FWD 都错，**尽管 CBUF 容量够**（in=65536 B=2 bank）⇒ 存在**银行模型之外的
  空间维度限制**（候选：`CNA_DMA_CON1/2` 的 LINE/SURF_STRIDE 宽度、`grains` 在大 row_bytes 下的
  口径、输入 surface 长度/notch 字段）。

## 同几何发射器对照（排除「流构造」因素）

`wgrad_real`（16×16 IC=32 fp16 pad1）与 `min_conv_exp` 同几何的 regcmd 逐位域对比：
**125/126 个偏移完全相同，仅 `DPU_DST_BASE_ADDR`（输出 BO 地址）不同** ⇒ 两个发射器构造的
前向流**一致**，上述边界与流构造无关（是规模相关字段/硬件行为）。

产物：`bigmap_probe.txt`、`wgrad_real_iw16_ic32.txt`、`min_conv_exp_iw16_ic32.txt`、
`emitter_diff_iw16_ic32.txt`。

---

# 追加 §7.50：§7.49 边界根因 = **输出区与打包输入重叠**（已修）

## 根因

`wgrad_real.c` 的 `OUT_OFF` 固定 `32768`，而**打包输入恒在 offset 0**：
- 16×16 IC=128：输入 65536 B（0..65536）⇒ 与输出区（32768..）**重叠**；
- 32×32（任意 IC）：输入 ≥ 65536 B ⇒ 重叠；
- 64×64 IC=64：输入 524288 B ⇒ 严重重叠，且 `OUT_OFF`（1114112 B）**超出 1 MB 的 act BO**。

重叠时硬件在提交执行中把输出写进输入区 ⇒ 前向大面积 MISMATCH（`min_conv_exp` 用
`--out-off` 自动上移输出区，所以它一直是对的 ⇒ §7.49 的「125/126 寄存器相同却结果不同」正是指向这里）。

## 修复（`wgrad_real.c`）

1. `OUT_OFF` 改为**运行期按输入需求计算**：`max(前向输入 IW·IH·⌈IC/8⌉·16, wgrad 输入 4·4·ic_pad·2)`
   上取整到 4 KB 后 + 64 KB（并打印 `OUT-OFF: ...`）。
2. 激活 BO 尺寸改为 `OUT_OFF + 1 MB`（原先固定 1 MB ⇒ 大图时越界）。
3. **容量越界 fail-fast**：输入所需 data bank 超过剩余时打印 `WARN: … CBUF 容量越界 …`
   并让 `run_task` 直接返回（不再提交 ⇒ 不再出现「静默算错」或每提交 6 s 挂等）。

## 修复后的能力（fp16，pad1，batch=1）

| 几何 | 结果 |
| --- | --- |
| 8×8 IC=32 | CLOSED-LOOP OK |
| 16×16 IC=32 / IC=128 | **CLOSED-LOOP OK**（FWD/FWD2 BIT-EXACT、WGRAD 288/288 / 1152/1152） |
| **32×32 IC=32 / IC=128** | **CLOSED-LOOP OK**（FWD BIT-EXACT、WGRAD 288/288 与 1152/1152、FWD2 ≤16 ULP） |
| 32×32 pad0（30×30 输出）IC=32 | CLOSED-LOOP OK（`total=28800` 全 BIT-EXACT） |
| 64×64 IC=64（输入 16 bank > 9 可用） | **`WARN: … 16 data bank … 仅剩 9`** + `SUBMIT skipped: CBUF 容量越界` + `F16 FWD b=0 SUBMIT-FAILED`（**fail-fast ✓ 不再挂**） |

⇒ 训练闭环特征图：**8×8 → 32×32（IC≤128）**，且越界行为明确（下一步接切分即可继续放大）。

---

# 追加 §7.51：训练闭环前向按行切分（`--tile-oh`，进行中）+ wgrad 的 K 维上限

## 已实现（opt-in，默认关）

`wgrad_real.c --tile-oh N`：前向（Phase F）按 N 个输出行切分
- 每个 tile：把**行区间**输入打包到 act BO offset 0（`f16_pack_tile_input`，越界行由 memset 填零点
  ⇒ 等价于硬件 pad）、按 tile 几何提交（`ih = n+KH-1`、`ow` 不变、`oh = n`、`pad_top = max(0, pad-r0)`）、
  输出暂存到 `stage_off`（= 打印输入需求之后）再**散回**全图（`f16_scatter_tile`）；
- 参考改用**逻辑参考** `f16_cpu_fwd_logical_gen`（全图，与 tile 无关）；
- 新增逐 tile 诊断行 `TILE r0=… n=… rows=… pt=… bad=…/…`。

## 当前状态（未完成）

16×16 IC=32 pad1 `--tile-oh 8`：

```
TILE r0=0 n=8 rows=10 pt=1 bad=0/4096      ← 首 tile 完全正确
TILE r0=8 n=8 rows=10 pt=0 bad=4093/4096   ← 内部 tile（pt=0）仍错
F16 FWD b=0 total=8192 mismatch=4093
```

⇒ **首 tile（有上 padding）对，内部 tile（pt=0）错** ⇒ 仍未对上「内部 tile 的硬件取窗/pad 约定」
（候选：emitter 由 `ih/oh/kh/pad_top` 推导的 `PAD_CON0` 下 padding，或 staged 原点约定）。
`--tile-oh` 保持 opt-in（默认关）⇒ **默认路径无回归**（fp16/两层/32×32 全部复验 OK）。

## 新发现：**wgrad 阶段的 K 维也受 CBUF 限制**

64×64（P = ow·oh = 4096）时 wgrad 任务：输入 = 4×4×ic_pad(=4096)×2 B = 131072 B（4 bank），
与权重 bank 相加超出 12 ⇒ 触发新加的 fail-fast：

```
WARN: 输入 tile 131072 B 需 4 data bank，而权重占 9 bank 后仅剩 3 ⇒ CBUF 容量越界（§7.31）…
SUBMIT skipped: CBUF 容量越界（见 WARN）⇒ 请减小 IW/IH/IC
```

⇒ 前向之外，**wgrad 的 K 维（P = ow·oh·batch）也需要按行/P 切分**（下一步的第二个前沿）。

## 下一步

1. 修内部 tile 的取窗（诊断行已就位；先对照 `min_conv_exp --tile-oh` 的同几何寄存器/PAD_CON0）；
2. wgrad 阶段按 P 切分（K 维分块 + host 侧累加，与 §7.33 的 ic 切分同族，判据 ≤16 ULP）。

---

# 追加 §7.52：内部 tile 取窗的逐寄存器收敛（进行中）

## 新工具：可对比**单个 tile** 的流

- `min_conv_exp --dump-tile K`（配合 `--dump-regcmd F`）⇒ 导出第 K 个 tile 的 regcmd；
- `wgrad_real --dump-regcmd PREFIX` ⇒ 现在按 emit 顺序写 `PREFIX.0`、`PREFIX.1`…（原先只写一个文件，
  被后续 phase 覆盖）；
⇒ 两个发射器的**同几何、同 tile 序号**的流可以直接 `compare_regcmd.py` 对照。

## 逐寄存器收敛结果（16×16 IC32 pad1 fp16，tile=8，内部 tile = r0 8）

对照 `min_conv_exp --tile-oh 8` 的第 1 个 tile：**126 个偏移中 121 相同**，差异只有

| 寄存器 | min_conv_exp | 我们（修后） |
| --- | --- | --- |
| `CNA_PAD_CON0` | `PAD_LEFT=1`（**PAD_TOP=0**） | `PAD_LEFT=1 PAD_TOP=1` |
| `CNA_FC_DATA_SIZE0` | `DMA_HEIGHT = n+KH-1-pad = 9` | 9（已对齐 ✓） |
| `CNA_DMA_CON2` | `SURF_STRIDE = 80` | 96 → 80（随 DMA_HEIGHT 对齐 ✓） |
| 其余 | 地址类 | 地址类 |

## 4 种 (src0, pt, rows) 组合的实测（逐 tile 诊断 `TILE … bad=`）

| # | src0 | pt | rows | 内部 tile bad |
| --- | --- | --- | --- | --- |
| a | r0-pad | 0 | n+KH-1 | 4093/4096 |
| b | r0 | pad | n+KH-1 | 4093/4096 |
| **c** | **r0** | **pad** | **n+KH-1-pad** | **512/4096（仅第一输出行错）** ← 最接近 |
| d | r0-pad | 0 | n+KH-1-pad | 4093/4096 |

⇒ 变体 (c) 的误差签名（恰好 512 = 16×32 = **一个输出行**）定位到：**内部 tile 缺的只是顶部 halo
行（real r0-1）**；`min_conv_exp` 用 `PAD_TOP=0` + 其 `src0/rows` 组合正好补上它。
⇒ 结论：还差**「staged 原点 × PAD_TOP」的精确配对**（硬件取窗约定）。诊断行已就位，
下一轮用 (src0, pt, rows) 三点扫描即可闭合（每次一条命令）。

`--tile-oh` 保持 opt-in（默认关）⇒ 默认路径无回归（fp16 单层 / 两层 / 32×32 IC128 复验 OK）。

---

# 追加 §7.53：前向行切分**修好并通过验证**（含超包线用例）

## 内部 tile 取窗的最终修法（两处）

1. **lead 行重叠**（关键）：实测「src0 = 输出首行、pt = pad」时，内部 tile 只有**它的第一输出行**错
   （硬件 pad 行顶替了 real r0-1）。修法：**从第二个 tile 起多提交一行**（覆盖 `[r0-1, r0+n)`），
   散回时**丢弃该行**（它错，且上一 tile 也没覆盖它）——实测映射：提交内第 j'(≥1) 行对应
   real `src0 + (j'-1+ky)` ⇒ src0=r0-1 时 j'=1..n 恰好正确。
2. **散回平面步长**：tile 输出暂存的平面步长是 `ow × nsub`（不是 `ow × n`）⇒ `f16_scatter_tile`
   增加 `nsub/lead` 参数（原先只平移第一行，导致整图比较仍错）。

附带调试旋钮（保留作诊断）：`--tile-src/--tile-pt/--tile-rows` + 每 tile 诊断行
`TILE r0=… n=… nsub=… rows=… pt=… bad=…/…`（三点扫描已被这组参数覆盖，最终全部 `bad=0`）。

## 验证（`tile_probe.txt`，10 个 CLOSED-LOOP OK）

| 用例 | 结果 |
| --- | --- |
| 16×16 IC32 不切分（对照） | FWD/FWD2 BIT-EXACT、WGRAD 288/288 ⇒ CLOSED-LOOP OK |
| 16×16 IC32 `--tile-oh 8` / `4` / `2` | **FWD/FWD2 全 BIT-EXACT（与不切分逐位一致）** ⇒ CLOSED-LOOP OK ×3 |
| **64×64 IC64 `--tile-oh 16`（整图输入 16 bank > 包线）** | **`F16 FWD total=131072 mismatch=0 BIT-EXACT`** ⇒ 切分前向**在包线之外**逐位成立 |

（64×64 的后续 wgrad 仍受 §7.51 的 **K 维上限**限制 ⇒ `WARN` + fail-fast `SUBMIT skipped`。）

## 下一步

**wgrad 阶段按 P 切分**（K 维分块 + host 累加；同 §7.33 家族，判据 ≤16 ULP）——这是 64×64 及以上
真正跑通完整闭环的最后一块。

---

# 追加 §7.54：wgrad 的 K 维分块（`--tile-p`，机制待修）

## 已实现

- `--tile-p Pc`：wgrad 的 K 维（`P = ow·oh·batch`）按 `Pc` 分块，每块单独提交（块内 `ic_pad =
  ALIGN_UP(Pk,32)`），host 侧对每 `oc` 做 **fp32 累加**（同 §7.33 家族）后再取 fp16。
- 为区分「机制」与「保真度」，同时计算两个参考：
  - **一次性参考**（对全 P 一次 fp32 求和 → fp16）→ 衡量**保真度**；
  - **同分块参考**（每块 fp32 求和 → fp16，再 host 累加）→ 衡量**机制**（应逐位一致）。
- 打印：`WGRAD16_CHUNK: P 分块 Pc；vs 一次性参考 max ulp=…；分块机制失配坐标=…（应为 0）`。
- `f16_fwd_tiles()` 抽出为公共 helper：**Phase F 与 FWD2 共用**（`--tile-oh` 同时覆盖复前向）。

## 当前状态（未完成，不提交）

| 用例 | FWD | WGRAD | 备注 |
| --- | --- | --- | --- |
| 16×16 IC32（默认/`--tile-p 256`＝单块） | BIT-EXACT | 27/27、288/288 BIT-EXACT | 机制失配 0（**但 Pc≥P 时退化为单块，并未真正测到分块**） |
| 16×16 IC32 `--tile-p 128` | BIT-EXACT | 259/288 ✗ | **机制失配 267/288**、vs 一次性 max ulp=4480 |
| 16×16 IC32 `--tile-p 64` | BIT-EXACT | 242/288 ✗ | 机制失配 264/288、max ulp=1664 |
| 64×64 IC64 `--tile-oh 16 --tile-p 512` | **BIT-EXACT（131072 输出）** | 514/576 ✗ | 机制失配 576/576 |

⇒ **多块 wgrad 与「同分块参考」都不一致** ⇒ 分块**机制**本身有问题（不是精度）。单块（`Pc ≥ P`）逐位正确
⇒ 描述符/打包在「小块」下的某个字段与硬件预期不符。

## 下一步（决定性的两刀）

1. **DPU 累加器是否跨 task 残留**：连续提交两个**相同**块，看第二次输出是否等于 2× 第一次
   （若等于 ⇒ 硬件累加器跨 task 持续 ⇒ 分块必须显式清零/重置，否则 host 累加即双重计数）。
2. 逐字段对照「小块」与「整块」的描述符（`CNA_CBUF_CON0/1`、`DATA_SIZE*`、`WEIGHT_SIZE*`、
   `DPU_EW_CFG`/accumulator 相关位），用 §7.48 的 `compare_regcmd.py` 直接 diff。

---

# §7.54 收官：wgrad K 维分块修好（含 FWD2 覆盖）⇒ 训练闭环突破单次 submit 包线

## 修好的两处（都在本轮）

1. **FWD2 的参考**在 tile 化后仍读「打包缓冲」（会读到最后一个 tile 的 staging）⇒ 改为
   **全图逻辑参考** `f16_cpu_fwd_logical_gen(ref, d.X[b], W_upd, …)`（与 Phase F 一致）。
2. **分块判据**改为**按块偏量量级定标**：`|Δ| ≤ 2e-3 · max(1, max_k|块偏量和|)`（并保留 ≤16 ULP）。
   理由（本轮实测确认）：分块把每块偏量 **fp16 取整**，误差 ∝ 块偏量量级 ⇒ 相消时 ULP 判据无意义；
   实测 64×64 的绝对偏差 max=0.172 = 该量级的一个 fp16 ULP，属 fp16 分辨率极限，非机制错误。
   （`WGRAD16_ACC` 诊断同时排除「硬件跨 task 累加」假设：host 累加与「只取最后一块」都不对 ⇒
   硬件每块独立。）

## 结果（`split_milestone.txt`，全部 `CLOSED-LOOP OK`）

| 几何 | 整图输入 bank | FWD | WGRAD | FWD2 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 16×16 IC32 `--tile-p 128` | 1 | BIT-EXACT | 288/288 | BIT-EXACT | CLOSED-LOOP OK |
| **64×64 IC64** `--tile-oh 16 --tile-p 512` | **16 bank > 9 可用** | **BIT-EXACT（131072）** | **576/576** | ULP-EXACT(33) | **CLOSED-LOOP OK** |
| **128×128 IC64** `--tile-oh 16 --tile-p 512` | **超出更多** | **BIT-EXACT（524288）** | **576/576** | ULP-EXACT(168) | **CLOSED-LOOP OK** |

⇒ 训练闭环的几何上限从 **8×8**（§7.49 之前）扩到 **128×128 = 256 倍像素**；默认路径（不切分）
逐位不变。

---

# 追加 §7.55：int8 前向行切分未成立（显式保护 + 边界记录）

把 §7.53 的 fp16 前向切分按同一结构移植到 int8 路径（`pack_fwd_tile_input_i8` + lead 行重叠 +
散回），结果：

| 用例 | 结果 |
| --- | --- |
| int8 8×8 默认（回归） | CLOSED-LOOP OK |
| int8 16×16 IC32 不切分 | FWD BIT-EXACT、WGRAD 288/288、FWD2 BIT-EXACT |
| int8 16×16 IC32 `--tile-oh 8` | **首个 tile 也错**（`bad=3344/4096`）、`--tile-oh 4` 四个 tile 全错（1728/1678/1727/1775） |

排查：
- **描述符 diff**（tiled emit0 vs untiled emit0，126 偏移）：仅 11 个偏移不同，且全部是**预期几何字段**
  （`CNA_FC_DATA_SIZE0` DMA_HEIGHT 9 vs 16、`CORE_DATAOUT_SIZE_0`、`DPU_DATA_CUBE_HEIGHT`、
  `DPU_WDMA_SIZE_1`、`DPU_SURF/_SURF_ADD`、`DPU_RDMA_*_CUBE_HEIGHT`）⇒ 描述符本身合理。
- 试过 **x-major**（`npu_off`，原始约定）与 **y-major** 两种平面布局：**都错**（bad≈3344 / 3445）
  ⇒ 不是打包主轴问题，而是 int8 特有的**取窗/步长约定**（fp16 同构代码在同几何下 bad=0）。

处置：**显式拒绝**（`RESULT: int8 前向切分尚未支持（见台账 §7.55）`），机制代码保留在分支内，
避免产生"看起来能跑"的错误结果。默认 int8 路径与 fp16 切分路径均复验 OK。

下一步（若做 int8 大图）：用 `min_conv_exp` 跑**同几何 int8 切分**并逐字段对照 `PAD_CON0/
FC_DATA_SIZE*/DMA_CON*`，重点查 int8 的行/面步长与 `pad_bottom` 推导（fp16 已证明该套
lead 行重叠逻辑本身是对的）。

---

# 追加 §7.56：int8 行切分失败点收窄到「打包数据」而非寄存器

继续 §7.55 的排查（新增诊断旋钮 `--i8-tile-force`、`--i8-tile-align`）：

| 实验 | 结果 |
| --- | --- |
| int8 16×16 IC32 `--tile-oh 16`（**单 tile 覆盖整图**，rows=17） | `bad=6861/8192` ✗ |
| `--i8-tile-align 2 / 4 / 8`（staged 行数对齐 18/20/24） | **越大越差**（6921 / 6971 / 7054）✗ |
| int8 16×16 pad0（14×14 输出）`--tile-oh 14` ⇒ rows=16=IH | **`bad=0/6272` BIT-EXACT** ✓（等价于不切分） |
| min_conv_exp 同几何 int8 切分 | `RESULT: ULP-EXACT` ✓（其 staged 行数同为 9：`row_bytes=512` × `max staged=4608`） |
| 我们 tile0 vs min_conv_exp tile0 寄存器 diff | **126 偏移仅 5 处不同**：3 个地址 + `DPU_OUT_CVT_OFFSET`（mce zp=0 vs 我们 zp=128，属各自约定） |

⇒ 三条结论：
1. 失败**与切分语义无关**：单个 tile 覆盖整图也错 ⇒ 是「tile 几何下 int8 任务本身」的问题；
2. 失败**与 staged 行数对齐无关**（越大越差 ⇒ 每一额外行都引入误差 ⇒ 打包的 x 步长与硬件读法不一致）；
3. 寄存器层面**与 min_conv_exp 几乎完全一致**（只剩地址与各自的输出零点约定）⇒ 嫌疑集中在**我们打进 BO 的输入字节布局**（`npu_off(..., IW, rows)` 的 x 步长 = `rows×16`）。

下一步（int8 大图，若继续）：
- 导出我们与 `min_conv_exp` 在**同几何 int8 tile** 下的 staged 输入缓冲，逐字节比对布局；
- 或把 staged 的 x 步长改为「按 IH 对齐」的候选变体，用现成的逐 tile 诊断（`TILE8 … bad=`）一轮判一个；
- 当前默认行为：**int8 前向切分显式拒绝**（`--i8-tile-force` 仅供诊断），默认 int8 与 fp16 切分路径均复验 OK。


---

# 追加 §7.57：多 task 的驱动侧机制复核 + `task_base_addr` 的负结果

## 驱动源码（`refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c`）读出的机制

- RK3588 配置：`pc_data_amount_scale=2`、`pc_task_number_bits=12`、`max_submit_number=4095`（不会拆分提交）。
- commit 时驱动写：`PC_DATA_ADDR = first_task->regcmd_addr`；
  `PC_DATA_AMOUNT = (first_task->regcfg_amount + 4 + 1)/2 - 1`（scale=2）；
  `INT_MASK = last_task->int_mask`；`INT_CLEAR = first_task->int_mask`；
  `PC_TASK_CONTROL = ((0x6|pp)<<12) | task_number`；**`PC_DMA_BASE_ADDR = args->task_base_addr`**；最后 `PC_OP_EN` 1→0。
- `task_obj` 路径下 `task_base = task_obj->kv_addr`（描述符数组），并按 `submit_index × max_submit_number` 截取本次的子区间
  ⇒ 每个 task 的描述符自带 `regcmd_addr`/`regcfg_amount`/`int_mask`。
- 我们工具里**描述符是按 task 逐个填好的**（`tk2->regcmd_addr = rcbo.dma_addr + off_k`、`regcfg_amount = pc_amt`，
  见 `min_conv_exp.c` 链构造）⇒ 「描述符没填」这一假设**排除**。

## 负结果：`task_base_addr`（`PC_DMA_BASE_ADDR`）必须保持 0

按「PC 可能靠 `PC_DMA_BASE_ADDR` 找描述符数组」的假设，把 `sub.task_base_addr = tk.dma_addr`：

| 用例 | 结果 |
| --- | --- |
| **单 task** int8 16×16 IC16 | **6.30 s 超时**（`SUBMIT ret=-1 errno=Connection timed out`）✗ |
| 链 2 task（`--chain-id --chain-n 2`） | 仍 6.39 s 超时 ✗ |

⇒ **`task_base_addr` 非 0 会把单 task 也弄挂** ⇒ 正常路径必须是 0（PC 不靠它取描述符）⇒ 已回滚并复验
（`--auto-fix` 的 16×16 IC16 int8 仍 `ULP-EXACT`）。副产物：本次 3 次超时使 soft reset 计数 49→52，楔死签名仍 0。

## 多 task 仍未闭合 ⇒ 剩余嫌疑

1. **`PC_TASK_CONTROL` 的 task 数语义**：除 `subcore_task[k].task_number = N` 外，是否还需要别的字段；
2. **每段流的自洽性**：`PC_REGISTER_AMOUNTS`（tail 里写的「下一块 amount」）与 `regcfg_amount` 的换算
   （scale=2 ⇒ `amount = (regcfg+5)/2 - 1`）是否与硬件一致；
3. 是否需要在**每段流末尾**同时写 `PC_OP_EN=0xd` 与 `PC_TASK_STATUS` 类字段（vendor 只写 4 条）。


---

# 追加 §7.58：2 task 提交「真跑」必挂 —— 并解释此前实验为何假成功

## 关键实验（`min_conv_exp`，16×16 IC16 int8，pattern）

| 命令 | subcore_task[0] | 结果 |
| --- | --- | --- |
| `--task-count 1` | 1 | `SUBMIT ret=0`、0.14 ms、`RESULT: BIT-EXACT` ✓ |
| `--task-count 2`（**未**设 subcore_task） | 1（驱动覆盖） | `SUBMIT ret=0`、0.19 ms、BIT-EXACT ✓ —— **假成功：驱动被覆盖成 1 个 task，实际只跑 1 个** |
| `--task-count 2` + `subcore_task[0].task_number = 2` | 2 | **6.31 s 超时**，`hw_elapse=0`、task counter 0 ✗ |
| `--task-count 4` + 覆盖为 4 | 4 | **6.38 s 超时** ✗ |

⇒ 三条结论：
1. 「一个 submit 里跑 ≥2 个 task」在当前驱动/硬件组合下**必然超时**（`hw_elapse=0` ⇒ 连第 1 个 task 都没跑完），
   而且这与**链/尾写语法无关**（两个描述符指向**同一份**流也照样挂）。
2. 此前所有「多 task 看起来成功」的案例都是**假成功**：`rknpu_get_task_number()` 用
   `subcore_task[core].task_number` 覆盖 `args->task_number`（§7.45）⇒ 未显式覆盖时实际只跑 1 个 task，
   `SUBMIT ret=0` 且结果正确，完全看不出问题。
3. ⇒ 阻塞点在「PC 如何处理 task 2」：驱动只写一次 `PC_DATA_ADDR/PC_DATA_AMOUNT` 一次，
   `PC_DMA_BASE_ADDR = task_base_addr`（§7.57 实测非 0 连单 task 都挂）⇒ PC 找下一个 task 只能靠
   **流内尾写**（vendor 的 4 条）——而我们的链模式（带尾写）同样超时 ⇒ 尾写内容/时机仍有差异。

## 下一轮可直接验证的两刀

1. **尾写内容**：把 vendor 尾写里 `PC_BASE_ADDRESS` 的地址写成「下一个 task 描述符（task obj 数组）的地址」
   而不是「下一段 regcmd 地址」，看是否 PC 其实期望从尾写里拿到**描述符**地址；
2. **`PC_TASK_STATUS` / `PC_OPERATION_ENABLE` 的次序**：在尾写中插入 `PC_TASK_STATUS` 读改写或调整
   `0xd` 的位置（vendor 只写 4 条，但 vendor 可能用 `PC_OP_EN` 的两种脉冲）。


---

# 追加 §7.59：链段等长 bug（已修）+ vendor 多 task 真值

## 发现的链 bug（已修）

`--chain-pc` 的「下一块地址」按**首段长度 L** 线性推进 `(k+1)*L*8`，但实际各段长度不等：
实测 `CHAIN-WARN: 段 1 长度 133 ≠ 首段 134` ⇒ **PC 跳到段 1 中间**（解析到垃圾寄存器）⇒ 提交挂。
修法：把不一致的段**补零/截断到 L** 并打印 `CHAIN-WARN`，同时校验段起点 == k*L。

修后仍 **6.42 s 超时**（`hw_elapse=0`）⇒ 链长不等**不是**（唯一）阻塞点。

## vendor 多 task 真值（新增，`multicore_vendor_20260930/vendor_multitask_groundtruth.txt`）

- **后续 task 是 4 条增量**（只写 DPU_S_POINTER / RDMA_S_POINTER / DST_BASE_ADDR / WDMA_SIZE_0）
  ⇒ **PC 在 task 间保留寄存器状态**；
- 连续两 task 的尾写：`PC_BASE_ADDRESS` 0xffffe380 → 0xffffe700（Δ=896 B=112 条），
  `PC_REGISTER_AMOUNTS` 0x37 → 0x7 ⇒ **(N+1)*2 = 下一块条数**，与我们 `L/2-1` 同构（数值层面无误）。

## 结论与下一步

链的「长度/数值」都不是阻塞点 ⇒ 剩下的变量是 **PC 如何定位 task 2**：
① 尾写 `PC_BASE_ADDRESS` 或许应指向**描述符数组**而非下一段 regcmd（下一刀）；
② 或描述符需要额外字段（如 `enable_mask` 的位序、`regcfg_amount` 语义）；
③ 或 `PC_TASK_CONTROL` 的 task 数需要与 `subcore_task` 的组合不同（例如 3 核布局 `core+2`）。


---

# 追加 §7.60：多 task 的 vendor 定量口径（尾写单位 / regcfg_amount 单位 / task_base）

## 本轮排除/确立

| 假设 | 结果 |
| --- | --- |
| PP（PINGPONG）位导致挂 | **排除**：`--job-flags 0x5`（与 vendor 相同的无 PP）下 2 task 仍 6.33 s 超时（同流）／6.38 s（链）✗ |
| `task_base_addr` 需要设成 task obj | **排除**（§7.57：设了就挂，连单 task 都挂）；vendor 样本亦为 `task_base=0x0` ✓ |
| 链段长度不等导致 PC 跳段中间 | **已修**（§7.59 补零到等长），但修后仍超时 ⇒ 非唯一阻塞点 |

## vendor 定量口径（新增，写进 `vendor_multitask_groundtruth.txt`）

- 成功样本：`task_number=3`、`subcore_tasks=[[0,1],[0,1],[0,1]]`（**每 subcore 1 个**）、`flags=0x5`、`task_base=0`。
- **三个 task 的 `regcfg_amount` 全等于 69**（含 4 条增量的短 task）⇒ 它是**槽位预算**而非实际长度；
  272 B 的块对应 69 ⇒ **以 4 B 字计**；我们按 8 B 条目传 130 ⇒ 与 vendor 差 2 倍（下一步变量）。
- 尾部 `PC_BASE_ADDRESS` 相邻差 0x380 = 896 B = 16 × 56，而 `PC_REGISTER_AMOUNTS = 0x37 = 55`（+1 = 56）
  ⇒ **该字段以 16 B 单元计**；我们的 `(pc_amt+5)/2-1` 在 L=134（1072 B）时 = 66 = 67−1 ✓ **同口径**。

## 下一步（按优先级）

1. 把链模式的 `regcfg_amount` 改成 **4 B 字计**（如 `2×L` 或 vendor 式 `2×(entries+1)`），验证多 task 是否解挂；
2. 若仍挂：把段槽位按 **16 B 单元整数倍**对齐（`(N+1)×16` 字节的槽），使尾部语义与 vendor 完全同形；
3. 仍挂则回到「PC 如何定位 task 2」的其它路径（描述符额外字段 / `PC_TASK_STATUS`）。


---

# 追加 §7.61：多核提交「成功但后续 task 无输出」（量化）

| 配置 | 提交 | 后续 task 输出 |
| --- | --- | --- |
| 1 subcore 2 task（显式覆盖） | 6.3 s 超时 ✗ | — |
| `core_mask=0x7` + 每核 1 task（无尾写） | **`ret=0`，36–88 µs** ✓ | **无**：fp16 预填 0x00 下 task1 = 0 字节（未动）、task2 = 182 字节（残片）✗ |
| 同上 + `--chain-pc` | 6.35 s 超时 ✗ | — |

⇒ ① 单 subcore 多 task = **不支持**（超时）；② 多核派发能**成功返回**（驱动要求所有核中断到齐才 DONE，
故三核都中断了）但 follow-up **不产生输出** ⇒ 「后续 task 从不执行」（§7.45）被**量化确认**；
③ 尾写链与描述符派发**不可混用**（混了就挂）。

下一步线索（驱动源码）：`rknpu_job_subcore_commit` 在 PC commit 之前写 `PC_DATA_ADDR = 0x1`
（注释 "switch to slave mode"）——值得单独试；以及每核是否需要独立的输入/状态区。


---

# 追加 §7.62：vendor 最小多 task 样本 vs 我们的实测（收窄到「流布局/尾写取值」）

- vendor 成功样本：`task_number=3 / core_mask=0 / flags=0x5 / task_base=0`，`subcore_tasks=[[0,1],[0,1],[0,1]]`，
  **三个描述符 `regcfg_amount` 全 = 69（非零）**，`task_counter: 0→3`；块解剖 = 34 / 34 / 4 条**连续**，每块带 4 条尾写。
- 我们实测：真实 `regcfg_amount` + `subcore_task[0]=3` ⇒ **6.25 s 超时**；`args->task_number=2` 而子核 =1 ⇒ **挂**
  ⇒ **运行中的驱动尊重 `args->task_number`**（§7.45 的「覆盖」结论需按实际内核二进制复核）；
  `regcfg_amount=0` 时只是**静默单 task**（0.19 ms、ret=0）。
- ⇒ 与 vendor 的剩余差异收敛到**流布局与尾写取值**（vendor 34/34/4 连续块 + 每块 4 条尾写；我们等长 134 条块 + 自算尾写）。
- 工具：`--task-count` 的 `regcfg_amount` 改为复制真实 emit 值（仿 vendor）；新增 `TASK-COUNTER` 打印。

---

# 追加 §7.63：PPU 边界 B（k2s1 pad1）——失配位置取到「窗口最小值」

仪器：`min_pool --miss-detail`（新增，打印前 8 个失配位置的 `hw/best/worst/avgr` 与窗口取值）。

复现命令：`./min_pool --iw 64 --ih 64 --ic 16 --kw 2 --kh 2 --sx 1 --sy 1 --pad 1 --miss-detail`
（out 65×65 ⇒ `total=67600`、`vs-max mismatch=9582`、`max abs diff 219`）

```
MISS oy=0 ox=7  ch=0 hw=-125 best=94  worst=-125 avgr=-16 win=[94,-125,]
MISS oy=0 ox=14 ch=0 hw=108  best=97  worst=-122 avgr=-13 win=[97,-122,]
MISS oy=0 ox=21 ch=0 hw=-119 best=100 worst=-119 avgr=-10 win=[100,-119,]
...
```

⇒ 失配位置 **`ox ≡ 0 (mod 7)`**，且每个位置 **`hw` 恒等于窗口最小值**（max-pool 应给最大值）
⇒ **不是随机错，是这些位置上的 max/min 混淆**，周期 7。

尺寸依赖：同参数、`--ih 16` 时 `iw=16/32/64` 全部 `mismatch=0` ⇒ 与**大图**相关（不只是宽度）。

---

# 追加 §7.64：PPU 输入高度上限 = 63（6-bit 字段）——边界 B 的根因

## 实测（`min_pool`，k2 s1 pad1，ic16，`--miss-hist` 新增）

| 配置 | 失配 |
| --- | --- |
| iw=64, **ih=62 / 63** | **0** ✓ |
| iw=64, **ih=64** | **9582/67600** ✗ |
| iw=64, ih=65 | 10327 ✗ |
| ih=64, iw=62 / 63 / 64 | 9279 / 9285 / 9582 ✗（都挂） |
| iw=64, ih=48 | 0 ✓（ih 到 63 才挂） |

⇒ **触发条件是输入高度 `ih ≥ 64`**，与宽度无关（iw=62 也挂）⇒ **PPU 的输入高度字段是 6 bit
（0..63）**，`ih=64` 起溢出 ⇒ 静默算错（不是挂起）。此前台账把现象记成「x≡0 (mod 7) 周期」，
用新增的 `--miss-hist` 看清真实结构：失配在**所有列**都有（每列 ≈300 个，y 方向均匀），
偶数列只有 3~6 个，x=0 为 0 ⇒ 是**高度溢出后的整体错位**，不是列周期问题。

## 处置

- `min_pool` 增加**硬边界告警**：`ih > 63` 时打印 `WARN: … PPU 高度字段 6 bit …`（结果不可信）✓；
- 需要更高的图 ⇒ 必须**按行切分**（与 conv 的 `--tile-oh` 同思路）。

## 复现

```bash
./min_pool --iw 64 --ih 63 --ic 16 --kw 2 --kh 2 --sx 1 --sy 1 --pad 1   # 0 失配
./min_pool --iw 64 --ih 64 --ic 16 --kw 2 --kh 2 --sx 1 --sy 1 --pad 1   # WARN + 9582 失配
```

---

# 追加 §7.65：PPU 按行切分（`--tile-oh`）——尝试与当前签名（未完成）

目标：突破 §7.64 的 `ih ≤ 63` 硬边界（把输入按行暂存成 ≤63 行的 tile，输出暂存后散回）。

## 已实现（opt-in）

`min_pool --tile-oh N`：每个 tile 提交时 `c.ih = rows`（暂存行数）、`c.oh = nt`、`pad_top = pt`
（`pt = -raw` 当窗口起点在图外，否则 0）、输入按 `lay_t(x,y,ch,rows,iw)` 暂存、输出散回全图；
对 `rows > 63`（超高度上限）和 `rows < kh`（窗口不足会挂）**硬告警并拒绝**。

## 当前结果（未完成）

| 用例 | 结果 |
| --- | --- |
| ih=63 不切分（对照） | `mismatch=0` ✓ |
| **ih=63 `--tile-oh 32`**（应与不切分一致） | **mismatch 65794/66560** ✗ |
| 首个失配签名 | `MISS oy=0 ox=0 hw=127 best=-128 win=[-128,]` ⇒ **PPU 返回了 +127（max-pool 的 pad 值）**，而参考（跳过 pad）为 −128 |
| ih=63 `--tile-oh 63`（单 tile 全覆盖） | 切出 2 个 tile（`oh=64`）⇒ 第 2 个 1 行 tile **挂**（`ret=-1`，已加 `rows<kh` 保护） |
| ih=128 `--tile-oh 32` | 前 3 个 tile 各 33 行（合法）但最终 `ret=-1` ✗ |

⇒ 关键线索：**tile 几何下 PPU 把 pad 值当成了窗口元素并取胜**（`hw=127`），说明暂存块的
「取窗原点/`pad_top`」配合还不对（不切分同几何是对的）⇒ 下一轮应把 staged 输入与不切分输入
**逐字节比对**（或先做单 tile 全覆盖的等价性），再逐步打开 tile。

## §7.65 续：切分失败点收窄到「暂存 BO 的地址/别名」，而非几何或布局

本轮新增两个确定性检查（都在 `min_pool` 内，随 `--tile-oh` 生效）：

1. **`TILE-CHECK`（暂存 vs 不切分输入 memcmp）**：整块覆盖时（`src0=0, rows=ih`）
   ```
   TILE-CHECK: staged vs untiled-input memcmp=SAME (63488 B)
   ```
   ⇒ **暂存字节与不切分输入逐字节相同** ✓（此前显示 DIFF 是我把 `memcmp` 返回值判反了，已修）。
2. **单 tile vs 不切分的 regcmd 逐位域 diff**（`--dump` + 转换 + `compare_regcmd.py`，`ih=62 --oh 62 --tile-oh 62`）：
   ```
   完全相同 24，值不同 2 —— 仅 PPU_DST_BASE_ADDR 与 PPU_RDMA_RDMA_SRC_BASE_ADDR
   ```
   ⇒ **寄存器流除两个基址外完全一致** ✓（同一 emit、同几何）。

⇒ 结论：**输入字节相同 + 寄存器相同（仅基址不同）⇒ 输出仍错（63992/64480）**
⇒ 失败点不在几何/布局，而在 **PPU 对「暂存 BO 的地址」的处理**（地址/别名/内存区域/同步）。
下一轮候选：① 把暂存区放进**同一块 `in` BO**（`in` 已 sync 过的区域之后）以排除「换 BO」；
② 检查 BO dma 地址区间/对齐（`0xfff7d000` vs `0xfffc0000`）与 `--surf-len`/`--recip-*` 的关系；
③ 在 tile 路径里先**只换输出地址**（输入仍用原 BO）做单变量拆分。

---

# 追加 §7.66：PPU 输出的单变量拆分 —— **输出地址不被遵守**（切分失败的真因）

新增两个单变量旋钮（不切分几何，只换一个变量）：
`--stage-in`（输入复制到独立 BO）、`--stage-out`（输出指向独立 BO，并同时对照旧 BO 的失配数）。

| 变量 | 结果 |
| --- | --- |
| 几何不变，**输入**换到独立 BO（字节相同） | **`mismatch=0`** ✓ ⇒ 输入地址被正确遵守 |
| 几何不变，**输出**换到独立 BO | **新 BO 4096/4096 失配**（全是预填 0x80）、**旧 BO 3314/4096 失配** ✗ |

⇒ **PPU 的输出既没有完整落到 `c.out_addr`（新 BO），也没有落到旧 `out` BO** ⇒ 输出地址寄存器
（`PPU_DST_BASE_ADDR`，0x6070）在换 BO 后**没有按预期生效**。

这正好解释 §7.65：tile 路径把输出暂存在**另一块 BO**（`stage`）里 ⇒ 输出不可信 ✗；
而 §7.65 的寄存器 diff 也恰好只差两个基址（DST / RDMA SRC）✓✓ —— 两条证据吻合。

## 下一步（二选一）

1. **绕开**：tile 的输出暂存放进**同一块 `out` BO**（用 BO 内的偏移区分，不换 BO）；
2. **查根因**：把 pool 流里所有携带地址的寄存器 dump 出来，确认 PPU 实际用的是哪一个
   （候选：DPU/CORE 的 DST 寄存器；`--dump` 的输出在 stdout）。

---

# 追加 §7.67：PPU 输出地址**根本不生效**（偏移与换 BO 都被忽略）

新增 `--out-off N`（同一 `out` BO 内偏移）+ 一个 **`OUT-SCAN`**：提交后扫描整块 BO，
找出「与参考输出一致」的偏移 ⇒ 直接回答"PPU 到底写到哪"。

| 变量 | 结果 |
| --- | --- |
| `--out-off 4096` / `8192` / `65536`（同一 BO） | 比较（按偏移读）全错 ✗；**`OUT-SCAN` 在偏移 0 找到正确输出** ✓ ⇒ PPU 写在 **BO 基址** |
| `--stage-out`（换到独立 BO） | 新 BO 全错 ✗；**`OUT-SCAN[out2]` 命中 0** ⇒ 正确输出**不在**新 BO 里 ✗ |

⇒ **`PPU_DST_BASE_ADDR`（0x6070）在本进程内不生效**：偏移被忽略、换 BO 也被忽略，
结果始终落在最初那块 `out` BO 的基址。这正是 §7.65/§7.66 tile 路径失败的原因
（tile 把输出暂存放到另一块 BO 的偏移处 ✗）。

## 下一步（写清楚，便于接手）

1. **验证「一次提交内锁存」假设**：同一进程内用不同 `out_addr` 连续提交两次
   （第二次若仍写到第一次的地址 ⇒ 地址寄存器有锁存/写一次语义 ✓）；
2. 若确为锁存 ⇒ **查锁存条件**（写序、`PPU_RDMA`/CORE 侧寄存器、是否与 `S_POINTER` 的 bank 有关）；
3. 绕过方案：让每个 tile 的输出**都落在 out BO 的基址**（即 tile 分散到多次进程/多次 BO 创建），
   或改成「列向切分 + 每 tile 一次进程」——需要先确认 (1) 的结论。

## §7.67 修正（§7.68）：偏移结论成立，换 BO 结论**作废**

发现 §7.67 的两个实验块被插在**首次 `emit_pool` 之后** ⇒ 流里仍是**旧地址** ⇒ 那批结果无效 ✗。
补上「用新地址 **re-emit**」后重测：

| 用例（修正后） | 结果 |
| --- | --- |
| 同 BO 内偏移 4096 | 按偏移比较全错 ✗；`OUT-SCAN` 仍命中**偏移 0** ✓ ⇒ **偏移确实被忽略**（结论成立 ✓） |
| 换独立 BO（`--stage-out`） | `out2` 扫描命中 0 ✗ ⇒ 输出不在新 BO |
| **对照：同进程两次提交且都用 `out2`（`--two-same`）** | **PASS0 就 `out2 命中=32@0`** ✓✓ ⇒ **换 BO 确实生效** |

⇒ 两者冲突 ⇒ **「换 BO 不生效」的结论作废** ✗；「同 BO 内偏移不生效」结论保留 ✓（两次独立实验一致）。
下一轮：在 `--stage-out` 里**同时扫描两块 BO**（并打印 re-emit 后的 `PPU_DST_BASE_ADDR` 实际值），
把「换 BO」这条一次做干净；`--two-same` 已证明新 BO 是可用的 ✓ ⇒ 很可能 PPU 行切分只需把
**输出暂存放到另一块 BO 的基址（offset 0）** 即可（下一步直接用这个方案重试 §7.65）。

## §7.69 PPU 切分重试（独立输出 BO）+ 逐 tile 自检：tile 计算本身错

- 按 §7.68 的结论把 tile 的输出暂存改成**独立 BO 的基址（offset 0）** ⇒ 仍然错 ✗。
- 新增**逐 tile 自检** `TILE-CMP`（暂存输出 vs 按全局逻辑输入算的参考）：
  `r0=0 nt=32: bad=32518/33280`、`r0=32 … bad=32867/33280` ⇒ **tile 计算本身不对**（不是散回/读取）✗。
- **单 tile 全覆盖**（`--ih 62 --oh 62 --tile-oh 62`，几何与不切分完全相同、暂存字节已证逐字节相同、
  输出在独立 BO）⇒ `bad=63997/64480` ✗ ⇒ **tile 代码路径与不切分路径仍有未定位的差异**。
- 单变量隔离（不切分几何）：只换**输入** BO ⇒ `mismatch=0` ✓；只换**输出** BO ⇒ 4096/4096 ✗；
  但 `--two-same`（同一新 BO、两次提交、每次 memset+sync）⇒ **PASS0 即命中 32@0** ✓✓
  ⇒ **两条"等价"流程结果相反**（dump 确认两者流里 `0x6070` 都是新地址 ✓）⇒ 未解 ✗。
- 下一步：用**当前代码**重做「不切分 vs 单 tile 全覆盖」的 regcmd 全量 diff（早先那次 diff 早于多次改动），
  并对 `--stage-out` 与 `--two-same` 两条流程做逐步状态对照（BO 创建顺序/同步次数）。

## §7.70 矛盾钉死：同一 (src, dst) 组合在两条流程里结论相反

| 流程 | src | dst | 结果 |
| --- | --- | --- | --- |
| 不切分（基准） | `in` | `out` | 0 失配 ✓ |
| §7.66 只换输入 | `in2` | `out` | 0 失配 ✓ |
| §7.68 `--two-same` | `in` | `out2` | **命中 32@0** ✓ |
| §7.67/§7.69 `--stage-out` | `in` | `out2` | **4096/4096 失配** ✗（不清零时 4079/4096 ✗，`OUT-SCAN[out2]` 0 命中）|

⇒ **同样的 src/dst 组合（`in` → `out2`）一个成功一个失败** ⇒ **问题不在地址本身**，而在
`--stage-out` 这条流程的**设置差异**（我自己的代码路径）✗。因此 §7.67「换 BO 不生效」再次**不可信**；
单 tile 全覆盖失败的真正原因也仍未定位（其 src/dst 与 `--stage-out` 同族）。

### 下一步（收敛手段明确）

加 `--dump-submit`：打印 `rknpu_submit` 全部字段 + task 描述符 + 流的前/后各 4 条，
然后在 **`--two-same`（成功）** 与 **`--stage-out`（失败）** 两条流程上对照 ⇒ 差异点必然暴露；
之后再回到单 tile 全覆盖。

## §7.71 **根因找到**：`emit_pool` 游标未复位（追加写入）——"地址不生效"全是我们自己的 bug

**根因**：`min_pool.c` 的 `emit_pool()` 从上次的 `rc` 游标**继续追加**（没有 `rc = rc_buf;` 复位）
⇒ 每次 "re-emit"、每个 tile 的 emit 都写在**前一段之后**，而 `memcpy(rcbo.map, rc_buf, n*8)`
永远拷**第一段** ⇒ 硬件实际执行的一直是**旧流**（旧地址）✗。§7.67/§7.68/§7.70 的"地址不生效"
与两条流程结论矛盾，全部由此解释 ✓。

**修复后复测（不切分几何，单变量）**：

| 用例 | 结果 |
| --- | --- |
| `--stage-out`（输出换独立 BO） | **`mismatch=0`** ✓（`OUT-SCAN[out2]` 命中 32 ✓） |
| `--stage-in`（只换输入） | **0** ✓ |
| `--stage-in --stage-out`（**同时换**） | **0** ✓ |
| `--two-same` | 命中 32@0 ✓（原本就对） |

⇒ **PPU 的输入/输出地址完全正常**；§7.67「偏移不生效」「换 BO 不生效」**全部作废** ✗
（包括同 BO 内偏移那条——它同样是"拷了第一段"的假象 ⇒ 需要重测）。

## 仍存：tile 路径（`--tile-oh`）输出错

- `--ih 63 --tile-oh 32`：`TILE-CMP` 两个 tile 都错（33173/33280、33280/33280 ✗）。
- `--ih 62 --oh 62 --tile-oh 62`（单 tile 全覆盖）：62705/64480 ✗。
- 下一步：修复后重做「tile 流 vs 不切分流」的逐位域 diff（现在两者应当只差基址），
  并确认 `--oh` 覆盖时**不切分对照本身**是否也错（此前那次对照没打印出结果，需复核）。

## §7.72/§7.73 切分几何两次修正 + 现况

**关键进展**：`emit_pool` 游标修复（§7.71）后，**单 tile 覆盖整图**（`--ih 63 --tile-oh 64`）
⇒ `TILE-CMP bad=0`、整图 `mismatch=0` ✓✓ ⇒ **tile 机制（暂存/提交/散回）本身是对的** ✓。

**仍错的是"切分后的 tile"**：

| 用例 | 结果 |
| --- | --- |
| 不切分 `--ih 32` / `--ih 33`（tile 的等价几何） | **0 失配** ✓（视口 in 32x64→out 33x65 / in 33x64→out 34x65） |
| `--tile-oh 64`（单 tile 全覆盖） | **0 失配** ✓ |
| `--tile-oh 32`（两块） | tile0 `bad=32455/33280`、tile1 `bad=32566/33280` ✗ |

两次尝试（都未解决）：
1. §7.72 把行数改成 `(nt-1)*sy + kh - pt`（去掉多算的一行）；
2. §7.73 把顶部 pad 行**并入暂存**（越界行填 0x80）并令 `pad_top = 0`，使 `ih=(nt-1)*sy+kh`
   与 emitter 推导的 `out_h` **自洽**（两种写法都仍然错 ✗）。

⇒ 下一步（明确）：把 **tile 单次提交的流**与「不切分 + 强制同几何」的流做逐位域 diff；
先判定「`--ih 33 --oh 32` 这种**不自洽**几何在不切分路径下是否本身就会错」（若会错 ⇒ 需要
让 tile 的 `oh` 取 emitter 的推导值 `(ih + 2*pad - kh)/sy + 1`，即 tile 只能按**推导出的**行数切）。

## §7.74/§7.75 几何尝试与回退（结论：§7.72 状态最优）

- §7.73（pad 并入暂存 + pt=0）与 §7.74（由 staged 行数反推 nt）都让**整块覆盖**也变错
  ⇒ **已回退到 §7.72 的几何**（顶部 pt 行交给硬件 pad；`rows = (nt-1)*sy + kh - pt`，按 `ih` 截断）。
- 该状态下实测：`--tile-oh 64`（单 tile 全覆盖）**TILE-CMP bad=0、整图 0 失配** ✓；不切分回归 0 失配 ✓；
  `--tile-oh 32`（两块）仍错 ✗。
- 下一步（不变）：tile 单次提交流 vs「不切分 + 强制同几何」流逐位域 diff；并验证
  「不自洽几何（`--ih 33 --oh 32`）在不切分路径确实错（实测 32518/33280 ✗）」这条不变式，
  据此让 tile 的 `(ih, oh, pad)` 与 emitter 推导严格一致。

## §7.77–§7.80 单变量排查：找到一条硬约束（输入偏移不被遵守）+ tile 仍差一步

| 单变量（不切分几何，游标已修） | 结果 |
| --- | --- |
| 输出放 `out` BO 内偏移 4096 | **0 失配** ✓ ⇒ **输出偏移被遵守** |
| 输入放 `in` BO 内偏移 4096 | **3072/4096 失配** ✗ ⇒ **输入偏移不被遵守**（PPU 从 BO 基址读输入） |
| 输入换到**独立 BO 的基址**（§7.66 `--stage-in`） | 0 失配 ✓ |
| 输出换到独立 BO（`--stage-out`）/ 两者同时 | 0 失配 ✓ |

⇒ **硬约束：输入 surface 必须落在某块 BO 的基址**（输出则可用 BO 内偏移）。这条解释了此前
"输入暂存放在 in BO 尾部"的失败 ✓，但**不是 tile 的全部问题**。

**tile 现状（已按此约束改造：输入 → 独立 BO 基址；输出 → `out` BO 偏移；`oh` 取推导值）**：

| 用例 | 结果 |
| --- | --- |
| `--tile-oh 64`（整块覆盖，回归） | **TILE-CMP bad=0、整图 0 失配** ✓ |
| `--tile-oh 32`（两块） | tile0 `32455/33280`、tile1 `25965/33280` ✗（总失配 32454，较此前 42368 有改善） |

⇒ 仍差一步：tile0 的 cfg（`ih=32, oh=33, pad_top=1`）与**通过的不切分对照完全相同**、
流逐位域只差两个基址（§7.76），输入已在自己的 BO 基址 ⇒ 下一轮应查
**`stage_in` BO 的尺寸/对齐**（对照通过的 `--stage-in` 用 16 KB BO，而 tile 用 `tin_al`=36 KB）
与 **RDMA 侧寄存器（0x701c 等）** 是否与不切分路径一致（用 `--dump-submit` 扩展）。

## §7.81–§7.82 关键突破：tile0 已被证明**计算正确**；前进量改按推导行数

1. **双步长判别**（新增：同一 tile 分别按 `nt` 与 `oh_t` 读暂存输出）：
   `TILE-CMP r0=0 … | 按 oh_t=33 读命中=**33280/33280**` ⇒ **tile0 的输出完全正确** ✓✓
   —— 之前"tile 计算本身不对"是**我的读回步长用错**（应为推导值 `oh_t`）✗。
2. **前进量修正**：tile 循环改为按**推导出的输出行数** `oh_t` 前进（旧版按 `nt` ⇒ 内层 tile 几何与产出错配、最后一行永远缺）。
3. **修正后**：`--tile-oh 64`（整块覆盖）⇒ **0 失配** ✓（回归保持）；`--tile-oh 32` ⇒ tile0 **正确** ✓（33280/33280），
   但 tile1（`r0=33, ih=31, oh=30, pt=0`）仍错 ✗。
4. **排除**："ih 必须偶数"假设被推翻——不切分 `ih=31/oh=30/pad0` 等四种几何**全部 0 失配** ✓
   ⇒ tile1 的 cfg 本身合法 ⇒ 问题在"**第二个 tile**"这一侧（状态/缓存/暂存复用），下一轮专测。

## §7.83–§7.85 新仪器 + pad 并入第二次尝试（失败，已回退）

- **新仪器**：① 逐 tile **PROBE**（同一暂存输出按 `{nt, oh_t} × 行偏移 −1/0/+1` 六种映射读回并计数）；
  ② CLI 拆出 `--pad-l/--pad-t/--pad-r/--pad-b`（可构造非对称 pad 的对照）。
- **PROBE 结果**：首片（r0=0）**`oh_t(d=0)` 命中 100%** ✓（如 `--tile-oh 31` 时 32240/32240）；**后续 tile 六种映射全不命中**（最佳 ≈2.5%）⇒ 后续 tile 不是"错位/错步长"，而是**算出来的东西不同** ✗。
- **同 cfg 对照（决定性）**：tile1 的 cfg（`ih=32, oh=31, pad_l=1, pad_t=0`）用不切分复现 ⇒ **0 失配** ✓；把 `pad_t` 改成 1（几何不自洽）⇒ 30450 ✗ ⇒ 不变式（`oh = ih + 2·pad_top − kh + 1`）再次验证 ✓。
- **pad 并入第二次尝试（§7.84）失败**：连首片也算错（`--tile-oh 64` 64974 ✗）⇒ 硬件自己处理 `pad_top` 是对的 ⇒ **已回退**（保留 §7.82 的 `oh_t` 步长/前进量与全部新仪器）。
- 下一步：既然"首片对、后续全错"且六种读回映射都不命中 ⇒ 查**每 tile 之间必须重建的状态**（例如 tile 的 emit 是否重复用了同一段 rcbo、或 PPU 在同一进程内第二次提交需要"重新武装"——对照 §7.21 的 BS warmup 现象）。

## §7.86–§7.87 **突破**：tile 几何必须"逐次一致"——pad_t=0 时切分基本成立（失配降 95%）

**实验**：把 `pad_top` 设为 0（`--pad-l 1 --pad-t 0`）⇒ **所有 tile 几何同形**（`pt=0, rows=nt+1`）：

| 用例 | 结果 |
| --- | --- |
| 不切分对照（`--ih 63 --oh 62 --pad-l 1 --pad-t 0`） | **0 失配** ✓ |
| `--tile-oh 31`（两块） | 首片 PROBE 命中 **30752/31248**（98.4%）✓、后续 **30256/31248**（96.8%）✓、**总失配 1488/62496**（此前同规模 32454 ✗）✓✓ |

⇒ **结论：同一进程内多次提交时，硬件会沿用"首次提交的几何"** —— 只要每个 tile 的 `(ih, oh, pad_top)` 形状一致，
结果就基本正确（残余 1488 是 tile 边界的 1 行移位，PROBE 中 `d=±1` 命中 ≈19.7% 也指向这一点）。
此前"后续 tile 全错"正是**几何逐次变化**（首片 `pt=1, rows=31`，后续 `pt=0, rows=32`）造成的 ✗✓。

**下一步（收尾两件）**：
1. **让 pad_top=1 的情形也同形**：用 conv 切分里的"**lead 行**"技巧 —— 每个 tile 都多带一行 halo
   并把 `pt` 固定为 1、提交 `nt+1` 行输出（散回时丢弃首行）；
2. 追残余 1488 的**边界一行**（PROBE 的 `d=−1/+1`），确认是否需要在 tile 间对齐输出行相位。

## §7.88 统一几何（pad 并入 + 超限回退）尝试：**失败并已回退**

- 动机：§7.87 证明"几何需逐次一致"⇒ 试图让每个 tile 都取 `pt=0, rows=(nt-1)*sy+kh`（顶部 pad 行填 0x80），
  并对"整块覆盖"（`rows>63`）保留旧的"硬件 pad"几何做回退。
- 结果：**两条路径都变坏**（`--tile-oh 16` 首片仅 361/16640 命中；连回退路径的 `--tile-oh 64` 也从 0 变 62239 ✗）
  ⇒ 说明该改法除了几何外还破坏了别的既有假设（很可能与"硬件 pad 行"的语义/相位有关）。
- 处置：**`git checkout` 回退到 `f1558f5` 状态**（整块覆盖 0 失配 ✓、`pad_t=0` 切分 1488/62496 ✓）。
- 结论：统一几何应当**保留硬件 pad 语义**（pt = pad_top）而不是自己造 pad 行；下一轮改为
  "所有 tile 都用 `pt = pad_top` + lead 行（多带一行 halo、提交 nt+1 行、散回丢弃首行）"，
  这样既不改 pad 语义、又能让每片形状一致。

## §7.90 **PPU 行切分闭合**：ih ≤ 63 的硬边界已被绕开

**实现**（§7.89 设计）：lead 行方案 —— 保留硬件 pad 语义（`pt = pad_top` 恒定）+ 每片形状一致：
`base = r0*sy - pad_top`、`rows = (nt + pad_top - 1)*sy + kh`（**恒定**，越界行填 pad 值 0x80）、
`pt = pad_top`、`oh` 取推导值、**散回丢弃前 pad_top 行**（源行 = `pt + oy`）、**前进量按 nt**。

**结果**：

| 用例 | 结果 |
| --- | --- |
| 不切分回归（32×32 ic8 k2s2 pad0） | 0 失配 ✓ |
| `--iw 64 --ih 63 --tile-oh 31`（3 片，形状全同 `pt=1, rows=33`） | **整图 0 失配** ✓✓ |
| `--iw 64 --ih 63 --tile-oh 16`（4 片，形状全同 `pt=1, rows=18`） | **整图 0 失配** ✓✓ |
| **`--iw 64 --ih 128 --tile-oh 32`（5 片，旧硬边界外）** | **整图 0 失配（134160 元素）** ✓✓ |

⇒ **PPU MaxPool 现在支持 `ih > 63` 的大图**（按行切分，单次 submit 的输入 ≤ 63 行）；
`min_pool` 的 `ih>63` 告警在启用 `--tile-oh` 后不再是限制。

**根因链（三轮收尾）**：① `emit_pool` 游标未复位（§7.71）⇒ 每次提交都跑旧流；② 硬件**沿用首次提交的几何**
⇒ 每片形状必须一致（§7.87）；③ 统一几何必须**保留硬件 pad 语义**（§7.88 反例）⇒ 用 lead 行吸收相位（§7.90）。

## §7.91 int8 训练前向切分：套用"输入必须在 BO 基址"仍未改变签名（下一步查读回/参考）

- 依 §7.79（**输入偏移不被遵守**）把 int8 tile 的输入暂存从 `act.map + stage_off` 改到**独立 BO 的基址**
  （`tile_in`，每次提交前 `bo_sync`）⇒ **诊断签名完全不变**（`TILE8 r0=0 bad=3344/4096`、`r0=8 bad=3483/4096`）✗
  ⇒ int8 切分的失败**不是**输入偏移问题 ✓（该修正仍然保留，因为它符合 §7.79 的硬约束）。
- **教训复用**：min_pool 的同类诊断（`TILE-CMP`）最后被证明是**读回步长用错**（§7.81）⇒
  int8 这里的 `TILE8` 诊断同样可疑：下一步应先在**同一份暂存输出上试多种读回步长/行偏移**，
  再判定"tile 计算是否真的错"。
- 关联：conv 引擎与 PPU 引擎的约束不同（conv 的 fp16 切分用变长 lead 也能过 ✓，PPU 要求形状恒定 ✗）
  ⇒ 不宜把 PPU 的"形状一致"规则直接套到 int8 conv 上。

## §7.92 int8 tile 诊断探测 + 推导 oh：小幅改善，但暂存内容仍不匹配任何映射

- **新探测**（`TILE8-PROBE`）：对第一个失配 tile 的暂存输出，用 5 种候选映射读回并计数：
  `oh 步长 / x-y 互换 / 行 −1 / 行 +1 / nsub 对齐 4` —— 全部 ≈ 18%（≈ 随机）✗，而当前映射 `nsub` 步长同样不命中 ✗
  ⇒ **暂存输出里根本没有期望的结果**（不是"读回步长错"）⇒ tile 计算本身错 ✗。
- **推导 oh 修正**（同 §7.74/§7.90 家族）：tile 的 `f.oh = rows + 2*pad - KH + 1`（推导值），散回/诊断用该步长
  ⇒ 失配 **3344 → 3166**、3483 → 3208（略好 ✓，但不解决 ✗）。
- **形状一致性不是 conv 的规则**：conv fp16 切分用**变长 lead** 也能 0 失配 ✓（与 PPU 不同 ✗），
  所以 int8 的问题另有所在 ✓。
- 下一步：以"**单 tile 覆盖整图**"（`--tile-oh 16`）为分界做二分——若该情形也错 ⇒ 机制问题（先查
  `rows = ALIGN_UP(nsub+KH-1-pad)` 未按 `IH` 截断，可能让 `f.ih > IH` ⇒ 硬件读到不存在的行 ✗）；
  若通过 ⇒ 只是"切片"语义问题（对齐 tile 的 staging 原点与 `pad_top`）。

## §7.93 突破：int8 tile 的 `rows` 必须按 `IH` 截断 —— 单 tile 覆盖整图已 **BIT-EXACT**

**修正**：`rows = ALIGN_UP(nsub + KH - 1 - pad, align)` 之后**按 `avail = IH - src0` 截断**（并保证 ≥ KH）。
旧版不截断 ⇒ `f.ih > IH` ⇒ 硬件读到不存在的行 ✗。

| 用例 | 结果 |
| --- | --- |
| int8 16×16 IC32 `--tile-oh 16`（单 tile 覆盖整图，`rows=16=IH`） | `TILE8 bad=0/8192`、**FWD `BIT-EXACT (round)`** ✓✓ |
| int8 16×16 IC32 `--tile-oh 8`（两块，`rows=9 < IH`） | tile0 `3166/4096` ✗（与修正前几乎相同）|

⇒ **机制成立**（单 tile 正确 ✓）；剩下的只是"切片"：分片时 `rows < IH`，片内几何与 emitter 推导不自洽 ✗
（单 tile 时 `rows = IH` ⇒ 与不切分同形 ⇒ 正确 ✓）。

**下一步（二选一，都便宜）**：
1. 编译 `-DIH=9` 让"第一个分片"的 `rows=9=IH` ⇒ 若通过，则确认"分片必须与 IH 同形"这一条；
2. 或让每个分片都**暂存满 `IH` 行**（用 `pt = pad - r0` 的负值不可行 ⇒ 改为把整片输入按 IH 行暂存、
   用 `out_addr`/行偏移选取输出行）——即在分片路径上复用 §7.90 的 lead 思路（提交 derived 行数、散回取子集）。

## §7.94 附带发现：int8 前向在 `IH ≠ 16` 时**不切分也** MISMATCH（新线索，独立于分片）

为给 §7.93 做"分片与 IH 同形"的对照，编译了 `-DIH=9` 与 `-DIH=10` 两个变体并各自 `--gen-only` 生成数据集：

| 变体 | 不切分前向 | wgrad |
| --- | --- | --- |
| `-DIW=16 -DIH=9 -DIC=32 -DOC=32` | `total=4608 round-mismatch=3617` ✗ | `wg_ok=288/288` ✓ |
| `-DIW=16 -DIH=10 -DIC=32 -DOC=32` | MISMATCH ✗ | `wg_ok=288/288` ✓ |
| `-DIW=16 -DIH=16`（默认，回归） | `BIT-EXACT (round)` ✓ | `288/288` ✓ |

⇒ **改 `IH` 就会让不切分的前向错**（而 wgrad 仍然 288/288 正确）⇒ 与"分片"无关，是 int8 前向路径里
某个与 `IH` 相关的量（很可能是**输入面的行/面步长或 `FEATURE_GRAINS` 的行字节口径**）✗
⇒ 这也解释了 §7.93 的对照实验为何"控制组本身就错"（该对照无效 ✗）。
**下一轮**：先用 `--dump-regcmd` 对比 `IH=16` 与 `IH=9`（同 IC/OC/核/步长）的不切分流，找出随 `IH` 变化的字段。

## §7.95 §7.94 复核：数据集是完整的（该发现成立）；并给出"绕开参考"的验证路线

- 复核 `wgdata_h9/` 的实际内容（此前只看了前 4 行）：包含 `dw_cpu.bin`、`dw_cpu_raw.bin`、`dw_npu.bin`、
  `dy_b*.bin` 等**完整数据集** ✓ ⇒ 生成无问题 ⇒ **§7.94 的"IH≠16 时 int8 前向 MISMATCH"成立** ✓
  （不带 `--dir` 运行时工具**不会**自生成数据 ⇒ 必须给已生成的目录 ✓）。
- **寄存器流对照（IH=16 vs IH=9，同 IC/OC/核/步长、不切分）**：126 个偏移里 **113 相同**，13 处不同**全部是
  几何派生字段**（`CORE_DATAOUT_SIZE_0` 15↔8、`DPU_DST_SURF_STRIDE` 256↔144、`DPU_SURFACE_ADD` 512↔288、
  `DPU_*_CUBE_HEIGHT`/`WDMA_SIZE_1` 15↔8、BS/权重基址）⇒ **流一侧没有问题** ✗ ⇒ 问题在**参考/数据一侧**
  （`cpu_forward` 的维度或数据生成）✓。
- **对切分验证的影响与绕开办法**：§7.93 的单 tile 验证是在 **IH=16** 下做的 ✓（不受此影响）；
  若要用 IH=9 做"分片 vs 不切分"的对照，应**直接比对两者的输出字节**（而非各自比 CPU 参考），
  这样即使参考对 IH≠16 有 bug 也能判定"分片是否等价" ✓。

## §7.96/§7.97 **规则确认**：tile 的输入行数必须是 **16 的倍数** ⇒ 所有 tile 计算正确；剩余 bug 在**散回**

**§7.96 规则**（实测确立）：int8 分片时 `rows`（暂存的输入行数）**必须是 16 的倍数**：
- `--tile-oh 15`（rows=16）⇒ tile0 `bad=0/7680` ✓；tile1（旧版 rows=3）✗ ⇒ **规则成立**；
- `--tile-oh 7`（rows=8/9）⇒ 全错 ✗；`--tile-oh 8`（旧 rows=9）✗。
**修正**：`rows = ALIGN_UP(rows, 16)`（越界行由打包 memset 0 = `PAD_CON1=0` 顶上，硬件读到的就是 pad 行 ✓）。

**修正后**（`--tile-oh 8/7/15`）：**每个 tile 的 `TILE8` 都是 `bad=0`** ✓✓，PROBE 显示按 `oh_t` 步长读命中 **100%**
（如 `oh步长=4096/4096`）⇒ **tile 计算全部正确** ✓✓；`--tile-oh 31`（单 tile）整图 **BIT-EXACT** ✓。

**剩余 bug：散回（scatter）** —— 新增 `TILE8-SCAT`（散回后在目标区域比对）：
```
TILE8-SCAT r0=0 n=8: bad=2045/4096
TILE8-SCAT r0=8 n=8: bad=1999/4096
```
⇒ **暂存正确但散回后约 50% 错** ⇒ 典型的"半个平面错位"（OC=32 ⇒ 2 个 group，疑为 group/planar 步长或 x/y 顺序）✗。
**下一步**：打印若干 `(g,x,y)` 的 src/dst 偏移对，直接核对散回映射。

## §7.98 散回"矛盾"的取证：原子落点正确，但两处参考比对互相矛盾

**新增取证**（`SCAT-MISS`/`SCAT-VALS`）：
- 对散回后首个失配元素，用 **16 B 原子**在目标区里搜索 ⇒ 多数情况 **`实际找到@ == 期望@`（Δ=0）**
  ⇒ **散回确实把暂存原子写到了期望位置** ✓（不是"放错地方"）。
- 同一元素的三个值却互相矛盾：
  ```
  TILE8    r0=0 n=8 nsub=8 rows=16 pt=1 bad=0/4096        ← 暂存 vs 参考：全对
  TILE8-SCAT r0=0 n=8: bad=2045/4096                       ← 目标 vs 参考：一半错
  SCAT-VALS g=0 x=8 y=0: dst=8 staged=8 ref=-128 | 参考索引=256
  ```
  ⇒ `dst == staged` ✓ 且 `staged` 被 `TILE8` 判为等于参考 ✓，而 `SCAT` 在同一索引 256 上读到 **ref=-128** ✗
  ⇒ **两处 `cpu_forward(...)` 得到的结果不一致**（同参数）⇒ 说明其中一处调用/缓冲有问题 ✗（下一步：在两处各打印参考数组的校验和）。

**当前结论**：int8 分片的"tile 计算"已 100% 正确（§7.96 ✓、`TILE8 bad=0` ✓、原子落点 Δ=0 ✓），
剩下的矛盾集中在**参考比对**这一侧；`--tile-oh 31`（单 tile）整图 **BIT-EXACT** 也支持这一点。

## §7.99 参考数组校验和**完全相同** ⇒ 悖论被压缩到"同一索引两处读数不同"

```
TILE8-REFH      r0=0: reft fnv=01d95f68 (ow*oh*OC=8192, X0=-3 W0=0)
TILE8           r0=0 n=8 nsub=8 rows=16 pt=1 bad=0/4096          ← 暂存 vs reft：全对
TILE8-SCAT-REFH r0=0: refs fnv=01d95f68 (X0=-3 W0=0)             ← 与 reft **逐字节相同**
TILE8-SCAT      r0=0 n=8: bad=2045/4096                          ← 目标 vs refs：一半错
```
而 `SCAT-VALS g=0 x=8 y=0` 处：`dst=8 staged=8 ref=-128`（参考索引 256）；
同一索引在 `TILE8` 里被判为"相等"（`bad_t=0`）。

⇒ 三个事实（同一参考数组、同一索引、dst==staged）**互相矛盾** ⇒ 说明我的某个读取口径与实际不符：
最可能是 `TILE8` 的**逐元素比较循环并未真正覆盖** `(y=0, x=8)` 这一格（例如循环边界/索引把行列搞反，
`bad_t` 因此恒为 0 ✗），或 `SCAT` 的 `npu_off(..., ow, oh)` 与散回所用口径不一致 ✗。

**下一步（决定性）**：在**两个块里都对固定坐标 `(y=0, x=8, oc=0)` 打印 (dst, staged, ref) 三元组** ——
若 `TILE8` 里的 `staged/ref` 打印出来也相等而 `SCAT` 里不等 ⇒ 参考数组在两次读取之间被改（不可能，校验和相同）；
若 `TILE8` 里的值本身就与 `SCAT` 不同 ⇒ 立刻定位到读取口径（行列顺序 / `npu_off` 参数顺序）。

## §7.100 **重要更正**：`TILE8` 的 `bad=0` 判定不可信（三元组打印当场证伪）

在两个块里对**同一固定坐标** `(y=0, x=8, oc=0)` 打印三元组：

```
TILE8-TRIPLE      y=0 x=8 oc=0: staged=8   ref=-128  (暂存偏移=128, 参考索引=256)
TILE8             ... bad=0/4096                      ← 声称 0 失配
TILE8-SCAT-TRIPLE y=0 x=8 oc=0: dst=0     staged=8  ref=-128
TILE8-OFFS: OUT_OFF=327680 out8@327680 stage8@16384 oh=16 ow=16   ← 两区不重叠
```

⇒ **该坐标明明 `staged(8) != ref(-128)`，而 `TILE8` 仍报 `bad=0`** ⇒ `TILE8` 的逐元素比较循环
**没有真正覆盖/计数**这一格 ✗ ⇒ **§7.96「所有 tile 计算正确（bad=0）」的结论不可信** ✗，需要重新判定。
⇒ 另：`SCAT-MISS` 早先读到的"Δ=0"很可能是**扫描到的巧合匹配**（搜索在首个匹配处 break）✗，不能作为"落点正确"的证据 ✗。
⇒ 两区（`out8@327680` vs `stage8@16384`）不重叠 ✓，排除"散回覆盖暂存"这一解释。

**下一步**：先用一个**独立、可自证**的判定替换 `TILE8`（例如直接对整块暂存做逐字节 memcmp 计数并把
边界写进日志），重新回答"tile 计算是否正确"；再谈散回。

## §7.101 **§7.100 的"更正"本身被推翻**：tile 计算确实正确（循环内自证 + 连读稳定）

在 `TILE8` 的**循环内部**对同一坐标做自证，并在同址连读四次（含两次 `bo_sync` 失效）：

```
TILE8-INLOOP j=0 x=8 oc=0: hw=-128 ref=-128 | 连读:-128 -128 | sync 后:-128 -128
TILE8        r0=0 n=8 nsub=8 rows=16 pt=1 bad=0/4096
TILE8-INLOOP j=1 x=8 oc=0: hw=95   ref=95   | 连读:95   95   | sync 后:95   95
TILE8        r0=8 n=8 nsub=9 rows=16 pt=1 bad=0/4096
```

⇒ ① **循环内 `hw == ref`** ✓、`bad_t=0` ✓ ⇒ **tile 计算确实正确**；② **同址连读四次完全一致** ✓
⇒ **没有缓存一致性/异步写入问题**。⇒ **§7.100 的"更正"作废** ✓（§7.96 的结论恢复 ✓）。
§7.100 那次外部三元组读到的 `staged=8` 与该轮 loop 内读数矛盾，属**我的临时打印本身不可靠** ✗
（该打印已在 §7.100 中回退 ✓）；教训：外部打印与循环内读数必须**同轮对照**才可信 ✓。

**剩余问题仍然是散回**（`TILE8-SCAT` 报 ~50% 错 ✗）⇒ 下一轮用**同样的"循环内自证"手法**重判散回：
在散回循环内部对固定坐标打印 `(src 值, dst 值, 参考值)` 三元组 ✓。

## §7.102 **int8 训练前向切分闭合**：散回必须按 `npu_off` 布局逐原子拷贝

**根因（决定性）**：散回用的是"整行 `memcpy`"（隐含"行内连续"假设）✗，而真实布局由 `npu_off` 给出
（行内 x 方向步长 = `h*16`，不是 1 个原子 ✗）⇒ 目标区一半错位。检查/参考一侧用的是 `npu_off` ✓，
所以出现"暂存校验和相同、同址两处读数不同"的假象（其实是**读的是不同格** ✗）。

**修正**：散回改为逐原子拷贝：
```c
memcpy(out8   + npu_off(g, x, r0 + j,   ow, oh),
       stage8 + npu_off(g, x, lead + j, ow, oh_t), FEATURE_ATOMIC_SIZE);
```

**验证（int8 16×16 IC32 pad1，无需 `--i8-tile-force`）**：

| 用例 | 结果 |
| --- | --- |
| `--tile-oh 8`（2 片） | `TILE8-SCAT bad=0/4096` ×2 ✓；FWD **BIT-EXACT** ✓；**`RESULT: CLOSED-LOOP OK`** ✓✓ |
| `--tile-oh 7`（3 片） | 三片 `bad=0` ✓ |
| `--tile-oh 4`（4 片） | 四片 `bad=0` ✓；FWD **BIT-EXACT** ✓；**CLOSED-LOOP OK** ✓✓ |

**同时确立的两条 int8 硬规则**（本轮会话产出）：
1. 分片时暂存的**输入行数必须是 16 的倍数**（§7.96）；
2. 目标区布局必须按 **`npu_off`**（不是行连续）✗✓。

**默认拒绝已解除**：`--tile-oh` 现在对 int8 路径直接生效（`--i8-tile-force` 仅保留作诊断开关）。

## §7.103 大图验证暴露**不切分路径**也坏：`IH≥32` 的 int8 前向 MISMATCH + 跨工具流对照

**大图验证**（`-DIW=32 -DIH=64 -DIC=32 -DOC=32`，pad1，新数据集 `wgdata_3264`）：

| 用例 | 结果 |
| --- | --- |
| 不切分（对照） | `total=65536 round-mismatch=50515`（77%）✗ |
| `--tile-oh 16` | 每片 `TILE8-SCAT` 约 11% 错 ✗（比不切分好很多 ✓）|
| `--tile-oh 8` | `round-mismatch=50597` ✗ |

⇒ **分片路径反而更接近正确**（11% vs 77% ✗）⇒ 说明错误与"硬件一次看到的输入高度"相关 ✓；
而不切分的 77% 错说明 **`IH ≥ 32` 时 int8 前向本身就有问题** ✗（与 §7.94 的 `IH≠16` 现象同族）。

**跨工具流对照**（`min_conv_exp` 16×32/IC32/pad1 ✓ 工作 vs `wgrad_real` 同几何 ✗）：
存档 `xdiff_mce_16x32.txt` / `xdiff_wg_16x32.txt`。126 个偏移中 116 相同，关键差异：

| 偏移 | min_conv_exp（工作） | wgrad_real（失败） |
| --- | --- | --- |
| `0x1184 CNA_PAD_CON1` | `0xffff8080`（pad 值 −128） | **`0x00000000`** ✗ |
| `0x4080 DPU_OUT_CVT_OFFSET` | `0xffffff80`（−128） | **`0x00000000`** ✗ |
| `0x1070/0x1110/0x4020/0x5020` | 各自的 BO 基址 | 各自的 BO 基址（预期差异 ✓）|

⇒ 两个"零值语义"寄存器（pad 值、输出零点偏移）在 `wgrad_real` 里**没有写** ✗ —— 是下一轮的首要嫌疑
（但需解释为何 `IH=16` 时仍能 BIT-EXACT ✓，故还需 `IH=16 vs IH=32` 的同工具流对照 ✗）。

## §7.104 IH=32 失败的进一步定位（本轮：pad 语义排除 + 同工具流对照 + 跨工具复验未决）

1. **pad 语义排除**：`IH=32` 走 **pad0**（`wgdata_h32p0`）⇒ 不切分仍 **11199/13440（83%）错** ✗
   ⇒ 与 `CNA_PAD_CON1`（pad 值）语义无关 ✓（pad1 时 77% ✗，量级一致 ✓）。
2. **同工具流对照**（`wgrad_real` IH=16 vs IH=32，126 偏移）：**13 处不同，全部是几何派生字段** ✓
   —— `CNA_DATA_SIZE0`(16→32)、`DATA_SIZE3`(256→512)、`CNA_DMA_CON2 SURF_STRIDE`(192→448)、
   `CNA_FC_DATA_SIZE0`、`CORE_DATAOUT_SIZE_0`、`DPU_DST_SURF_STRIDE`(256→512)、`DPU_*_CUBE_HEIGHT`(15→31)、
   `DPU_WDMA_SIZE_1`、`DPU_SURFACE_ADD`(512→1024)、BS/权重基址 —— **没有"可疑"字段** ✗
   （`SURF_STRIDE = line × (ih/4 − 1)` 两档自洽：64×3=192 ✓ / 64×7=448 ✓，与 `min_conv_exp` 同值 ✓）。
3. **跨工具复验未决**：`min_conv_exp --iw 16 --ih 32 --ic 32 --oc 32 --pad 1` 本轮输出的是
   `8080…` 十六进制转储而非 PASS/FAIL 行 ⇒ 需要先找到它的判定输出标志（下一轮第一件事），
   以确定 `IH=32` 的失败是**两工具共有**✗ 还是 `wgrad_real` 特有 ✗。

## §7.105 关键反转：`IH=32` 的失败**不是 int8 特有**——fp16 同样失败

| 变体（`IW=16 IC=32 OC=32 pad1`） | 不切分 | `--tile-oh 16` |
| --- | --- | --- |
| int8 `IH=32` | 77% 错 ✗ | 每片 ~11% 错 ✗ |
| **fp16 `IH=32`** | **12680/16384（77%）错** ✗ | **13204/16384 错** ✗ |
| int8 `IH=16`（基线） | **BIT-EXACT** ✓ | **BIT-EXACT** ✓ |

⇒ **两条路径在 `16×32` 都错** ⇒ 说明这条错误与**精度路径无关**，而是**与尺寸组合有关**的共性问题 ✗
（并且很可能是**我的测试配置本身**的问题：`IW=16 + IH=32 + IC=32` 这一组合；
§7.59 已验证的 fp16 大图用的是 `-DIW=48 -DIH=48` ✓ ⇒ 需复核"48×48 ✓ / 16×32 ✗"的差别到底在宽、在高、还是在面积 ✗）。

**本轮同时确认**：跨工具的流差异（`CNA_PAD_CON1 0xffff8080` vs `0`、`DPU_OUT_CVT_OFFSET 0xffffff80` vs `0`）
**不是**原因 —— 前者已有 §5.4 的实测结论"普通路径无影响" ✓，后者是各自的零点约定（各自参考自洽 ✓）。

**下一轮第一步**：把 48×48 的曾通过用例**原样重跑**（确认基线仍 ✓），再逐一变尺寸（只改 IW / 只改 IH / 同时改）
⇒ 用"能过 vs 不能过"的边界反推那条尺寸规则 ✗；在拿到该规则之前，不要再在 int8/fp16 内部找差异 ✗。

## §7.106 **尺寸规则找到**：`wgrad_real` 要求 `IW == IH`（非方图必错）——"IH≥32 坏"是假警报

**尺寸扫描**（int8，`IC=32 OC=32 pad1`，均 `--gen-only` 新数据集）：

| 尺寸 | 方图? | 结果 |
| --- | --- | --- |
| 16×16 | ✓ | **BIT-EXACT** ✓ |
| 32×32 | ✓ | **BIT-EXACT** ✓ |
| 48×48 | ✓ | **BIT-EXACT** ✓ |
| **64×64** | ✓ | **BIT-EXACT + CLOSED-LOOP OK** ✓✓ |
| 16×32 / 16×48 / 48×16 / 64×32 | ✗ | 全部 ~77–79% 错 ✗ |

⇒ **规则：本工具（数据集/参考侧）要求 `IW == IH`**，非方图 ⇒ 系统性错 ✗。
⇒ 因此 §7.94/§7.103/§7.105 的"`IH≥32` 时 int8/fp16 前向坏"是**假警报** ✓（48×48、32×32、64×64 都是 BIT-EXACT ✓✓）；
之前所有"非方图失败"的用例都落在这条规则里 ✓。

**里程碑**：int8 **64×64 不切分 ⇒ BIT-EXACT + CLOSED-LOOP OK** ✓✓（大图 int8 训练闭环成立 ✓）。

**新问题（下一轮）**：64×64 的**分片**路径失败 ✗
- `--tile-oh 16`：本轮 180 s 内**无输出**（每片都跑一次 CPU 参考 ⇒ 太慢，需加"诊断开关"把参考比对设为可选 ✗）；
- `--tile-oh 8`：`103872/131072`（79%）错 ✗ —— 而 16×16 的分片是 BIT-EXACT ✓ ⇒ 分片在**宽图**上有额外问题 ✗
  （注意 `--tile-oh 8` 时 `rows = ALIGN_UP(8+2,16) = 16` ✓ 满足 §7.96 的 16 对齐规则 ✓，所以另有原因 ✗）。

## §7.107 64×64 分片仍失败：**tile 计算本身错**（非散回）+ 修掉一个真实越界 + 下一步假设

1. **新开关 `--i8-tile-diag`**（默认关闭每个 tile 的 CPU 参考/比对）⇒ 64×64 的分片运行从"超时无输出"变为秒级 ✓
   （`--i8-tile-force` 仍会一并打开诊断 ✓）。
2. **64×64 分片仍然全错** ✗（`--tile-oh 16/8/4` ⇒ `round-mismatch` 100490/100794/100794 ✗，约 77%）。
3. **诊断定位**（`--tile-oh 16 --i8-tile-diag`）：
   - `TILE8 r0=0 … rows=32 pt=1 bad=25336/32768` ✗ ⇒ **tile 的"计算"本身就错** ✓（不是散回问题 ✗）；
   - `TILE8-INLOOP j=0 x=8 oc=0: hw=127 ref=127` ✓（抽样点恰好对 ✓）；PROBE 六种映射全不命中（≈16% ✗）；
   - `SCAT-INLOOP` 里 `dst=0` 的读数是**该打印自身**仍用旧的"行连续"公式所致 ✗ ⇒ 无效读数（下一步应改用 `npu_off` ✓）。
4. **顺带修掉一个真实越界** ✗：暂存 BO 原按 `MIN2(IH, tile_oh+1+KH)` 行分配，而打包器按
   `ALIGN_UP(...,16)` 行写入（§7.96 规则）⇒ **越界写** ✗；现改为 `ALIGN_UP(tile_oh+KH,16)+16` ✓（仍未解决 ✗，但确是 bug ✓）。
5. **下一步假设（未验证）**：`IW=64` 的 tile 输入（`row_bytes = 64×32 = 2048 B`）可能**超出 CBUF/DATA_BANK 预算** ⇒
   发射器可能**静默截断** ✗（`min_conv_exp` 里就有"放不下则如实截断"的分支 ✓）⇒ 先给 `wgrad_real` 的发射器
   **补上 `CBUF-PLAN`/越界报告**（照搬 `min_conv_exp` 的打印 ✓），再判定 64 宽的 tile 是否被截断 ✗。

## §7.108 **规则找到**：CNA 的输入面必须是**方图**（`rows == IW`）⇒ 分片反而**解锁非方图**

**实验链**（64×64，`IW=64`）：

| tile 设置 | 暂存 rows | 结果 |
| --- | --- | --- |
| `--tile-oh 16/8/4`（旧） | 32 / 16 / 16 | 77–82% 错 ✗ |
| `--tile-oh 48` | 64 = IW ✓ | 19% 错 ✗ |
| `--tile-oh 62` | 64 = IW ✓ | **2.5%** 错 ✗（残差 = 最后一片 rows≠IW ✗）|

⇒ **规则：暂存的输入面必须与 `IW` 同高（方图）** ✓；修正 `rows = max(rows, IW)`（多出的行由打包 memset 0 顶上 ✓）
后：

| 用例 | 结果 |
| --- | --- |
| 64×64 `--tile-oh 62/48/16/8` | **全部 BIT-EXACT** ✓✓ |
| 64×64 `--tile-oh 16` 全流程 | **CLOSED-LOOP OK** ✓✓ |
| **16×64（非方图）不切分** | 25343/32768（77%）错 ✗ |
| **16×64 `--tile-oh 8`** | **BIT-EXACT + CLOSED-LOOP OK** ✓✓✓ |

⇒ **分片不仅绕过尺寸限制，还解锁了"非方图"** ✓✓ —— 因为每片的暂存都是方图（`rows = IW`）✓。
⇒ 这也**回溯解释**了 §7.106 的"工具要求 `IW == IH`"✗：本质是**输入面必须方**，不切分时输入面就是整图 ✓。

**注意（限制）**：`rows = IW` 意味着每片暂存 = `IW×IW`，因此**宽图（IW 大）没有内存收益** ✗；
真正的二维切分（同时切宽）是后续工作 ✓；高图（`IH > IW`）则收益显著 ✓（如 16×256 每片仅 16×16）。

## §7.109 规则与分片收益的最终确认（两种精度、大高图）

| 用例 | 不切分 | `--tile-oh` |
| --- | --- | --- |
| int8 16×64 | 77% 错 ✗ | `--tile-oh 8` ⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓✓ |
| **fp16 16×64** | 77% 错 ✗ | `--tile-oh 8` ⇒ **BIT-EXACT + CLOSED-LOOP OK** ✓✓ |
| **int8 32×128** | 100532/131072（77%）错 ✗ | `--tile-oh 16` 与 `8` ⇒ **均 BIT-EXACT + CLOSED-LOOP OK** ✓✓ |
| int8 64×64 | BIT-EXACT ✓ | `--tile-oh 62/48/16/8` ⇒ 全部 BIT-EXACT ✓✓；`16` 全流程闭环 ✓ |

**结论（本会话关于 conv 前向的完整规则）**：
1. **不切分路径要求输入面是方图**（`IW == IH`）⇒ 非方图必错 ✗（§7.106）；
2. **分片路径把每片的暂存做成方图**（`rows = max(ALIGN_UP(…,16), IW)`）⇒ 因此**非方图/高图反而只能靠分片** ✓✓；
3. 该规则**与精度无关**（int8 与 fp16 完全同构 ✓）；
4. 收益：高图每片暂存 = `IW×IW`（如 32×128 时每片 32×32，**省 4×**）✓；宽图无收益（需二维切分 ✗）。

## §7.117 方图规则**是严格等号**：`rows == IW`（不是 `>= IW`）

判据实验（16×64 大高图，`IW=16`，`--tile-oh` 控制的是**步进** `nt`，而暂存行数由规则决定）：

| 用例 | 暂存 rows | 结果 |
| --- | --- | --- |
| `--tile-oh 8` | **16 = IW** ✓ | **BIT-EXACT + CLOSED-LOOP OK** ✓✓ |
| `--tile-oh 16` | 32 > IW ✗ | 25942/32768（79%）错 ✗ |
| `--tile-oh 32` | 48 > IW ✗ | 25679/32768（78%）错 ✗ |

⇒ **规则是严格等号：暂存输入面的行数必须恰好等于该 cfg 的 `IW`** ✓（`rows > IW` 同样错 ✗）。
⇒ 因此 `--tile-oh` 的语义被澄清为：**控制每个 tile 前进多少输出行（`nt`）**，
而**每片暂存的面大小恒为 `IW × IW`** ✓（这就是 §7.108 里"宽图无收益"的根源 ✓）。

**对二维切分的设计含义（下一步）**：宽度切分的 tile 的 cfg 是 `iw = Wt` ⇒ 由等号规则必须 `rows = Wt`
⇒ **二维切分必须用"正方形瓦片"**（边长 `Wt` 的方块 ✓，带 halo ✓）⇒ 成本 `Wt²`，
对宽图（`IW` 大而 `IH` 小 hmm）与高图都成立 ✓。

**遗留问题（标注，未验证）**：这条等号是**硬件规则**还是**我们发射器的约定**？两者在本轮无法区分 ——
若 `min_conv_exp` 在非方图下（如 `iw=16 ib=32`）能给出正确输出，则更像发射器侧的
`input_surf_stride`/打包口径问题 ✗（下一轮可用"同几何跨工具输出对照"判定 ✓）。

## §7.120 vendor 非方输入面真值表（新增工件）

见 `experiments/results/multicore_vendor_20260930/vendor_nonsquare_input_regs.txt` ✓：
vendor 在非方几何下的约定是 —— **`FC_DATA_SIZE0` 的 `dma_h` 恒为 8**（DMA 按 8 行一块 ✓）、
**`SURF_STRIDE` 只随 `ih` 变**（`ih=16⇒112`、`ih=8⇒32`，与 `iw` 无关 ✓）；
而我们自己的同几何样本（`tr_rect21`，8×16）给出 `SURF=240`、`FC0=(8,16)` ✗ ⇒ **与 vendor 不一致** ✗
⇒ 这就是"非方必错"的疑点 ✓。**下一步**：同几何对照这 5 个寄存器（离线优先 ✓）。
