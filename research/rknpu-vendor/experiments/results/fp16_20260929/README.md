# §7.17 FP16（PROC_PRECISION=2）路径打通（2026-09-29 第二十五轮）

结论一句话：**手写 regcmd 的 fp16 卷积在 RK3588 上跑通**——`min_conv_exp --fp16` 与 fp16
CPU 参考逐位一致（多元素求和时 ≤8 ULP），且**累加器 ≥ fp32（不是 fp16）**；§七-7 的
「INT16 输出置位会 timeout」**未复现**（fp16 输出走 `FP32TOFP16_EN` 直通）。

## 1. fp16 发射配置（相对 int8 的全部差异）

| 寄存器 | int8（同几何） | fp16 | 说明 |
| --- | --- | --- | --- |
| `CNA_CONV_CON1` 0x100c | 0x00000000 | 0x00000120 | `IN_PRECISION=2` + `PROC_PRECISION=2` |
| `CORE_MISC_CFG` 0x3010 | 0x00000001 | 0x00000200 | `QD_EN=0`、`PROC_PRECISION=2` |
| `DPU_DATA_FORMAT` 0x4010 | 0x00000000 | 0x48000002 | `IN/OUT/PROC_PRECISION=2` |
| `DPU_RDMA_RDMA_FEATURE_MODE_CFG` 0x5044 | 0x00007810 | 0x00017850 | `IN/OUT/PROC_PRECISION=2` |
| `DPU_OUT_CVT_SCALE` 0x4084 | 0x00004001 | 0x00010001 | `FP32TOFP16_EN=1`、`OUT_CVT_SCALE=1`（直通） |
| `DPU_OUT_CVT_SHIFT`/`OFFSET` | 0x0e / 0xffffff80 | 0 / 0 | 无重量化 |
| `DPU_BS_CFG` 0x4040 | 0x00020150 | 0x00000053 | fp16 整段 bypass BS（无 int8 域 bias 表） |
| `CNA_WEIGHT_SIZE0` 0x1030 | 0x00000a20 | 0x00001440 | 权重字节数 ×2（2 B/元素） |
| `CNA_CBUF_CON1` 0x1044 | 4 | 8 | `DATA_ENTRIES` 按 8 通道平面计 |
| `CNA_CONV_CON1` 之外 | — | — | `ic_real==1` 专用路径（ARGB/NONALIGN/CVT）在 fp16 下关闭 |

## 2. 布局规则（真机实测）

- **输入**：像素 = **16 B = 8 个 fp16**；ic 平面步长 = `iw·ih·16 B`，即 `[plane][y][x][8ch]`；
  行主序与 int8 相同。`DATA_ENTRIES` 按 8 通道平面计（`DIV_ROUND_UP(iw·2·ceil(ic/8), 8)`）。
- **权重**：与 int8 layout A 同构 `[oc1][i1][ky][kx][oc2][i2]`；ic 原子 = **32 元素**
  （`ic2=MIN(ic,32)`），每元素 2 B；ic 越界槽填 **0**（fp16 无 zero-point）。
- **通道对齐**：`ic ≤ 8 → 8，否则 16 对齐`（`c.ic=24` 的 3 平面组合失败；8/16/32/40/48/64 均过）；
  `oc` 按 16 对齐。
- **CBUF 权重 bank 估算**要按 2 B/元素 ×2。

## 3. 决定性证据

- **标尺探针**（`--in-ruler`，tan 目录 `04_ruler.txt`）：把输入填成"元素下标"，用单点权重读出
  硬件实际访问的输入元素下标。ic=16 时权重槽 j=0..7 → 元素 0..7，j=8..15 → 元素 128..135
  （iw=ih=4）⇒ 平面步长正好 `iw·ih·8` 元素 = `iw·ih·16 B`。布局矩阵（`05_layout_search.txt`）
  9 组合中只有 (pix-elems=8, watom≥16) 通过。
- **累加器精度**（`--in-cancel`，`03_accumulator.txt`）：输入 ch0=2048、ch1=1、ch2=−2048，
  权重全 1。fp16 逐步累加会把 2049 舍成 2048 → 结果 0；fp32 累加 → 1。实测
  **hw = 1.0（ic=3/8/16/32 全部）⇒ 累加器 ≥ fp32**。
- **ic 对齐**（同文件）：`--ic-pad 24`（3 平面）必挂；16/32/40/48/64 全过。

### 2bis. 通道对齐（§7.18 定论）

- **oc 原子 = 16**（`--f16-ocatom 16`，默认），且 **每 tap 的 oc 跨度必须按 oc 原子组总和向上
  对齐**：`w_oc2 = ALIGN_UP(oc_real, 16)`、`WEIGHT_KERNELS = w_oc2`。
- **权重 ic 行必须是完整的 16 元素（32 B）原子**：`c.ic = (ic_real ≤ 16) ? 16 : ALIGN_UP(ic_real, 16)`；
  否则「单 ic 平面（ic≤8）+ 多 oc 组（oc≥17）」会 submit **ETIMEDOUT**。
- 原实现的触发链：`--auto-fix` 的 int8 分支先把 `WEIGHT_KERNELS` 设成 `ALIGN_UP(oc_real,2)`
  （oc=17 → 18），fp16 分支只改 `w_oc2` 未同步 ⇒ 描述符 18 与实际打包跨度 32 不一致 ⇒ 第二个
  （不完整）oc 组的 per-tap 步进错位。

## 4. 覆盖（`01_fp16_regression.txt`：pass=74 fail=0 envfail=1；bias/1×1-deconv 段为信息性）

通过：1×1 ic∈{1..96} oc16；1×1 ic16 oc∈{4..64}（含 17/24/33）；**3×3p1 ic16
oc∈{4,8,16,17,24,32,33,40,48,64}（含全部不完整 oc 组）**；3×3p1 ic8/ic32 oc∈{17,24,40}；
3×3 pad0/1 六形状；5×5 pad2；矩形核 kw3kh2/kw2kh3/kw1kh5/kw5kh1/kw4kh6/kw6kh4；stride 2/3；
`1×1 ic≤8 且 oc≥24`（§7.18 修复前是 9 例 ETIMEDOUT，现全 BIT-EXACT）。
唯一非通过项：`3x3p1 ic16 oc65` 的**随机 ETIMEDOUT**（§7.12 纪律，单独跑通过）。
`02_ic_sweep.txt`：1×1 oc8 下 ic∈{1..96} 共 27 例 **27/27 BIT-EXACT**。

## 5. 未闭合（下轮入口）

1. ~~`kw·kh > 1` 且 `oc > 16`~~ —— §7.18 闭合（oc 原子 16 + 每 tap oc 跨度 16 对齐 +
   `WEIGHT_KERNELS` 同步）。
2. ~~`1×1` 且 `ic≤8` 且 `oc≥24` 的 ETIMEDOUT~~ —— §7.18 闭合（ic 行必须是完整 16 元素原子）。
3. ~~**fp16 deconv 未打通**~~ —— **§7.22 已闭合**：根因是 §7.17 新增的 fp16 CPU 参考**缺少 deconv
   scatter+flip 分支**（一直拿普通卷积当参考）。补上后 `--deconv-flip` S=2/4/8 **3/3 BIT-EXACT**。
   已知共享边界：**1×1 核 + S=2** 在 int8（172/1024）与 fp16（186/1024）**都 MISMATCH**（参考/边界
   口径问题，与精度无关）。
4. **fp16 的 BS/BRDMA bias（§7.20/§7.21）**：机制已定＝**fp32 表 + "仅当前一次 submit 的 BS 阶段为激活时才重载"**；配方（已验证 1×1 3/3、3×3 3/3 BIT-EXACT，长序列 4/5）＝**先跑一次同几何的 BS-bypass 提交**，再跑 BS 激活的用例（发射器在 BS 激活时自动预热 `--warmup-submits 3`）；在 BS 状态反复切换时仍脆弱（连续 BS 激活会 ETIMEDOUT 簇，且会累积设备 soft reset）。真实 workload（每个 conv 的 BS 常开）不受影响。详情：
   - **机制**：bias 槽是 **fp32**（决定性单点——槽 = `0x40a00000`(=5.0f) 时
     `hw = conv(36) + 5.0 = 41`；把参考改成"按 fp32 位型读表"后整例 BIT-EXACT，独立重复 3/3 成功）。
     可用配置：`--bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias <fp32 位型>`。
   - **不稳定**：同一进程连续 submit 时约 **50% `SUBMIT ETIMEDOUT`**（BRDMA 相关，奇数/偶数交替）；
     在长序列归档中即使 `ret=0`，也可能**表数据未送达 DPU**（`hw=36` vs `ref=41`，5/5 复现）。
     ⇒ 回归脚本把这一段列为**信息性、不计入 pass/fail**。
   - 下一轮建议：把 BS/BRDMA 表放进 **weight BO**（照 §7.12 与 vendor 的做法，`BS_BASE_ADDR` 指
     weight BO 偏移）而不是独立 bias BO，并加"表送达自检"（先跑一个已知 bias 的 1×1 用例确认生效）。
5. 仍未做：多核派发（`--subcore` 仍禁跑）、`ic>96`/`oc>64` 的更大组、fp16 的独立 BN（vendor 走 `BN_CFG=0x20150` + NRDMA 表，见 §7.21）。
4. **设备状态纪律**：板上**同一时刻只能有一个发射器实例**（CMA BO 固定大小；并发实例会互相
   踩缓冲，产生固定幅值的假 MISMATCH）。回归脚本采用"出现 MISMATCH 先跑已知 BIT-EXACT 的
   3×3 健康探针、探头脏则丢弃该结果"的协议（`dirtyevents` 计数）。核心 5 例已在 3 轮重复中
   完全一致。

## 6. 文件

| 文件 | 说明 |
| --- | --- |
| `01_fp16_regression.txt` | fp16 全量回归（健康探针协议、dirtyevents） |
| `02_ic_sweep.txt` | 1×1 oc8 的 ic 1..96 扫描（27/27） |
| `03_accumulator.txt` | 累加器精度（`--in-cancel`）与 ic 对齐（`--ic-pad`） |
| `04_ruler.txt` | 标尺探针：硬件实际访问的输入元素下标 |
| `05_layout_search.txt` | 像素原子 × 权重 ic 原子 9 组合搜索 |
| `06_impulse.txt` | ones/one-hot 冲激探针（分离权重与输入寻址） |
| `07_int8_regression.txt` | §7.11 的 74 例（`--pat2 --asym-w`）——本次 73/74，唯一的 FAIL 是随机 ETIMEDOUT（同一命令单独跑 5/5 BIT-EXACT） |
| `../../min_conv_exp.c` | `--fp16` / `--f16-selftest` / `--f16-pix-elems` / `--f16-watom` / `--f16-ocatom` / `--w-ic-tap` / `--in-one-hot` / `--in-ones` / `--w-ones` / `--in-ruler` / `--in-cancel` / `--din-ch` / `--dma-ch` / `--ws1` |
| `../../run_fp16_regression.sh` 等 | 板端回归与归档脚本 |
