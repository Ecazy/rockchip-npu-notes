# 深通道 fp16 训练闭环（`wgrad_real.c` 参数化 IC/OC）（§7.37）

回答 handoff §3.3「训练侧剩余：**更深通道**、更多层与残差/池化组合」。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：板端 exp3 `bash run_wgrad_deep_probe.sh`（日志 `deep_probe.txt`）。

## 改动（`experiments/wgrad_real.c`）

1. `IC`/`OC`/`L2_IC` 加 `#ifndef` 守卫 ⇒ 可 `gcc -DIC=64 -DOC=64` 生成深通道变体
   （`L2_IC` 默认跟随 `OC`；默认 `IC=3/OC=16` 行为不变）。
2. fp16 输入通道对齐改由 `f16_ic_pad_for(ic)` 决定（`≤16→16`，否则 `ALIGN_UP(ic,32)`，
   与 `min_conv_exp` 同规则），替换原来写死的 `FEATURE_ATOMIC_SIZE=16`
   （在 `f16_pack_fwd_input` / `f16_pack_fwd_weights` / `f16_cpu_fwd` / `make_fwd_cfg_f16` /
   `make_fwd_cfg_l2` 五处）。
3. 激活/权重 BO 由 64 KB 放大到 **1 MB**（深 ic/oc 描述符）。
4. 表头打印 `ic/oc/iw/ih/k/pad`（日志自描述）。
5. 写回后复前向的判据改成 **§7.28 口径**（`≤ g_l2_ulp=16 ULP` 或 `|diff| ≤ g_l2_abs=1e-4`），
   并打印单个差异元素的 `ulp/abs`（区分「同一 W' 的两种 fp32 累加顺序」与真错）。

## 结果（`deep_probe.txt`）

**单层 fp16 闭环在 ic ≤ 128 / oc ≤ 128 全部 `CLOSED-LOOP OK`**：

| 变体（`-D`） | 用例 | 前向 | wgrad | 写回后复前向 |
| --- | --- | --- | --- | --- |
| 默认 `ic3 oc16` | pad1 B1 | BIT-EXACT | 27/27 BIT-EXACT | BIT-EXACT |
| `ic64 oc64` | pad1 B1 | BIT-EXACT | **576/576 BIT-EXACT** | BIT-EXACT |
| `ic64 oc64` | pad0 B1 | BIT-EXACT | 576/576 BIT-EXACT | **ULP-EXACT（1 元素 1 ULP，hard=0）** |
| `ic64 oc64` | pad0 B3 | 3×BIT-EXACT | 576/576 BIT-EXACT | 3×BIT-EXACT |
| `ic128 oc32` | pad1 B1 | BIT-EXACT | **1152/1152 BIT-EXACT** | BIT-EXACT |
| `ic128 oc32` | pad0 B2 | 2×BIT-EXACT | 1152/1152 BIT-EXACT | ULP-EXACT(1 元素) + BIT-EXACT |
| `ic64 oc128` | pad1 B1 | BIT-EXACT | 576/576 BIT-EXACT | BIT-EXACT |

- 深通道下 wgrad 的 27→576→1152 个坐标**全 BIT-EXACT**（CNA K 维归约 fp32 累加 ≥ CPU 参考）；
- 唯一非逐位一致的是两例「写回后复前向」的 **1 个元素 1 ULP**（如 `ic64 oc64 pad0`
  `(oy=2,ox=4,oc=45) hw=0xd5d3 ref=0xd5d4 ulp=1 abs=0.0625`）——同一 W' 的**两种 fp32 累加顺序**，
  与 §7.28 的近零/顺序残差同族，非语义错；
- 规模：`ic64 oc64` 578 次 submit（hw 21–30 ms）、`ic128 oc32` 1154 次（hw 10–40 ms）。

## 边界（本轮未支持，已记录）

| 场景 | 现象 | 原因 |
| --- | --- | --- |
| 深通道 **int8**（`-DIC=64 -DOC=64 --dir …`） | 前向 `round-mismatch=3979 maxdiff=255`、wgrad `SUBMIT-FAILED` | int8 打包器（`pack_*`/`pack_wgrad_weights`）按 **16 通道原子**写死，未按 ic 块参数化 |
| 深通道 **`--two-layer`**（`-DIC=64 -DOC=64`） | `L2 F1/F2 mismatch=4085/4089`（L2_WGRAD 仍 576/576 ≤1 ULP） | 层2 读回/缓冲仍假设 16 通道原子（`f16_read_act` C≤16、`L2_ACT` 尺寸等） |

## 产物

- `experiments/wgrad_real.c`：`#ifndef IC/OC/L2_IC`、`f16_ic_pad_for()`、1 MB BO、判据与明细。
- `experiments/run_wgrad_deep_probe.sh`、`deep_probe.txt`。


---

# 追加 §7.40：深通道 `--two-layer` 已修（原 §7.37 边界 b）

日志：`deep_probe2.txt`（104 行，`D2)` 段为两层深通道）。

## 两个根因

1. **层间偏移写死**：`L2_A2_OFF=36864`、`L2_W2_OFF=8192` 是按 `IC=3/OC=16` 选的常量。
   深通道时层1 输出面 = `ow·oh·OC·2` B（OC=64 → 8192 B，落在 `32768..40960`）与
   `L2_A2_OFF=36864` **重叠**；层1 fp16 权重 = `KW·KH·IC·oc_w·2`（IC=OC=64 → 73728 B）
   与 `L2_W2_OFF=8192` **重叠**。
   ⇒ 改为按几何计算 `l2_a2_off`/`l2_w2_off`（`ALIGN_UP(…,4096)`，并保留 oc16 旧布局下限），
   加 BO 容量检查，并打印 `L2_LAYOUT: a1@… a2@… w2@…`。
2. **层1 权重打包与参考的 `ic_pad` 写死 16**：`f16_pack_weights_gen(…, IC,
   FEATURE_ATOMIC_SIZE, …)` 与 `f16_cpu_fwd_gen(…, IC, FEATURE_ATOMIC_SIZE, …)`，
   而 cfg 用 `f16_ic_pad_for(IC)`（§7.37 已改）⇒ 描述符与硬件预期布局不一致，
   层1 前向 4093/4096 全错。层2 同理用 `L2_IC` 作 ic_pad（48 这类非 32 对齐时会错）。
   ⇒ 统一改为 `f16_ic_pad_for(IC)` / `f16_ic_pad_for(L2_IC)`。

## 结果（`deep_probe2.txt` D2 段）

| 变体 | L2 F1 | L2 F2 | L2_WGRAD（层2/层1） | 写回后复前向 | 结论 |
| --- | --- | --- | --- | --- | --- |
| `d3_16`（默认回归） | BIT-EXACT | BIT-EXACT | 144/144、27/27 全 BIT-EXACT | BIT-EXACT | OK |
| `d64_64` | BIT-EXACT | BIT-EXACT | 576/576、576/576（≤1 ULP） | ≤1 ULP | OK |
| `d128_32` | BIT-EXACT | BIT-EXACT | 288/288、1152/1152（≤1 ULP） | ≤1 ULP | OK |
| `d48_48`（非 32 对齐 ic） | BIT-EXACT | BIT-EXACT | 432/432、432/432（≤1 ULP） | ≤1 ULP | OK |

⇒ **深通道双层跨层闭环闭合**（`2-LAYER CLOSED-LOOP OK` ×4，与单层深通道 7 例合计
`deep_probe2.txt` 中 11 个 `CLOSED-LOOP OK`）。

## 仍存边界：深通道 int8（供下次接手）

失败形态：`FWD b=0: round-mismatch=3979 maxdiff=255` + `WGRAD SUBMIT-FAILED`。
写死站点（均需按 §7.40 同法参数化，int8 对齐规则 `ic ≤16→16 否则 32 对齐`、
`oc_w = oc≤32 ? ALIGN_UP(oc,2) : ALIGN_UP(oc,32)`）：

| 位置 | 问题 |
| --- | --- |
| `pack_fwd_input` | `memset(… IW*IH*FEATURE_ATOMIC_SIZE)` 与 `buf[npu_off(0,…)+c]` 只支持单 16 通道平面 |
| `pack_fwd_weights` | ic 维与布局 `((kx*KH+ky)*16+oc)*16+icc` 写死 16/16，未用布局 A 通用式 |
| `pack_wgrad_weights` | 槽位式 `(i1*OC+oc)*ic2+i2` 与 oc 跨度写死（默认几何下恰好等价） |
| `make_fwd_cfg` | `f.oc = 32` 写死（应 `oc ≤32 ? ALIGN_UP(oc,2) : ALIGN_UP(oc,32)`） |
| `cpu_fwd`/`cpu_wgrad`（int8） | 与上述打包同源的布局假设 |


---

# 追加 §7.41：深通道 int8 也已修（原 §7.37 边界 a）→ 训练侧两条边界全部闭合

日志：`deep_probe3.txt`（116 行，E 段为深 int8）。

## 改动（`wgrad_real.c`）

1. 新增 `i8_ic_align()`（`ic ≤16 → 16，否则 32 对齐`）、`i8_oc_w()`（每 tap oc 跨度 = 发射器
   `weights_kernels` 规则）、`i8_oc_span()`（**硬件侧 oc 跨度** = `ALIGN_UP(MAX(oc,32),32)`）
   与 `w_elem_index_i8()`（通用布局 A）。
2. `pack_fwd_input`：支持多 16 通道平面（原只写平面 0）。
3. `pack_fwd_weights` / `pack_wgrad_weights`：改用 `w_elem_index_i8()` 通用布局。
4. `make_fwd_cfg`/`make_wgrad_cfg`：`ic` 用 `i8_ic_align()`；**`oc` 用 `i8_oc_span()`**
   （原来写死 32，对 `OC>32` 是错的；`OC≤32` 时恰好等于旧值 ⇒ 默认路径不变）。
5. 输出回读（前向、复前向、wgrad）由 `out[oc]` 改成按 16 通道原子分组
   `out[npu_off(oc/16, ox, oy, ow, oh) + oc%16]`（`OC≤16` 时与原式等价）。

## 一个关键约定差异（写下来避免下次踩）

`wgrad_real.c` 的 int8 全链路用**空间转置约定**：输入 `npu_off()` 是 x-major、tap 嵌套是
`((…·kw + kx)·kh + ky)`（ky 内层）、输出回读也是 x-major。三者组合与硬件的
「y-major 输入 + kx 内层 tap」等价（方核下互为转置，整链自洽）。
而 `min_conv_exp.c` 的 int8 用 y-major + `ky*kw+kx`（kx 内层）。**两者都对**，但不能混用：
本轮若直接把发射器的 tap 嵌套搬过来，默认 int8 回归立刻错。

## 数据集陷阱（本轮踩到）

`--dir wgdata_pad1` 里的 `w.bin/x_b*/dy_b*` 是按**该构建的几何**落盘的
（默认 432 B / 192 B；深构建 `w.bin` 应为 36864 B）。用默认目录跑深构建会读到不完整数据
（现象：提交超时/结果离奇）。正确做法：`--gen-only --dir <新目录> --pad P --seed S`
（`dataset_prepare()` 在目录不存在时生成并落盘）→ 再在该目录跑闭环。

## 结果（`deep_probe3.txt`，14 个 `CLOSED-LOOP OK`）

| 段 | 内容 | 结果 |
| --- | --- | --- |
| A | 默认 int8 + 默认 fp16 | 2× CLOSED-LOOP OK |
| B/C/D | 单层 fp16 深通道（ic64/oc64、ic128/oc32、ic64/oc128，pad0/1 × batch） | 5× OK |
| D2 | 双层 fp16 深通道（d3_16/d64_64/d128_32/d48_48） | 4× 2-LAYER OK |
| E | **深通道 int8**（d3_16 `w.bin`=432 B、d64_64 =36864 B、d128_32 =36864 B） | **3× OK** |

深 int8 关键判据：`FWD round-mismatch=0`、`WGRAD_TOTAL coords=576/576 与 1152/1152
grads=36864/36864 round-exact`、`FWD2 BIT-EXACT`。
