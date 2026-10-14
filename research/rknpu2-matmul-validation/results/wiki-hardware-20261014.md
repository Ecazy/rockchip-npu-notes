# 第一部分：RKNPU2 各模块与寄存器的实际行为

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)


本文整理 RK3588 NPU 的寄存器配置、数据布局和已记录的板端行为。位域定义参考 TRM；
实现细节和适用边界结合公开源码说明。实验现象只适用于记录中的驱动、几何和配置，
不代表所有 RK3588 系统，也不代表本次编辑重新完成了板端验证。

## 0. 证据等级

标记区分描述的依据。**[板测]** 表示特定条件下的历史验证结果，扩展形状或更换环境后仍需复测；
**[代码]** 表示可检查的实现，不能据此推断所有硬件组合都可用。
本页保留历史实验摘要，链接指向公开实现或规范；未公开的原始实验记录不包含在 Wiki 中。

| 标记 | 含义 |
|---|---|
| **[板测]** | 特定环境下已有逐位或数值对照记录；本页为摘要，非本次复测 |
| **[代码]** | 当前库的实现方式；不单独证明硬件行为或性能上限 |
| **[推断]** | 由多组已验证配置归纳出的规律，未单独证明 |
| **[TRM]** | 只有 TRM 描述，本项目未验证 |
| **[历史]** | 早期结论，后来被修正或环境已变，保留以免重复遇到的问题 |

环境：RK3588，rknpu 驱动 0.9.8（早期 0.9.6），kernel 6.1.118-zt，NPU 1 GHz。
换驱动版本或 SoC 后，第 2 节的驱动行为要重新核对。性能数据是这些条件下的历史样本。

---

## 1. 整体模型

### 1.1 三个独立 core，每个 core 是一条固定流水线

```text
             ┌────────── CBUF 384 KiB（12 bank × 32 KiB）──────────┐
DDR ─AXI─> CNA（取 feature/weight、DCOMP 解压、输入 CVT）─> CORE（MAC 阵列、累加）
                                                               │ flying
DDR ─> DPU_RDMA（MRDMA/BRDMA/NRDMA/ERDMA 读第二操作数）───> DPU（BS→BN→EW→LUT→OUT_CVT）─WDMA─> DDR
                                                               └─> PPU（pooling，本项目未用）
```

- NPU 采用 **NVDLA 风格的固定卷积流水线**，本库通过配置流水线执行计算。已实现的运算表达成
  "卷积 + DPU 后处理" 或 "DPU 单独做逐元素/归约"。**[代码]**
- 矩阵乘 `A[M,K] @ B[K,N]` 按 1×1 卷积执行：feature 的空间维放 M，通道放 K；
  N 个 1×1×K kernel 就是 B。**[板测]**
- DPU 有两种主数据来源：
  - **flying**（`DPU_FEATURE_MODE_CFG.flying_mode=0`）：主数据来自同一 task 的 CORE 输出；
  - **offline**（`flying_mode=1`，同时 `RDMA_FEATURE_MODE_CFG` bit0=1）：主数据由 MRDMA
    从 DDR 读。此时 task 只开 DPU+DPU_RDMA，CNA/CORE 空闲。**[板测]**
- 同一 core 上 DPU-only task 与 GEMM task **串行**执行，DPU task 运行时 CORE 空转。
  同一核上的 NPU split-K 归约会占用后续 GEMM 的执行时间；CPU 归约可以与下一波 NPU 工作重叠。
  **[板测]** [INT8 DPU 归约实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_i8_dpu_chain.cpp)
- 三个 core 共享 DDR。本库多核按独立输出（通常按 N）分配任务。已验证实现中，
  `nn_mode` 保持 0；TRM 所述 co-work 模式未在本库启用。**[板测]**
  [多核提交实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_core.cpp)

### 1.2 算力与带宽的量级（1 GHz，表内分别注明核数）

| 项目 | 数值 | 等级 |
|---|---|---|
| INT8 MAC | TRM 标称 1024/cycle/core，约 2 TOPS/core | [TRM] |
| INT8 GEMM 实测单核最好 | 约 1.5–1.85 TOPS（native A/C，K≈1024–2048） | [板测] |
| 单核 B（weight）有效读带宽 | 约 8–10.8 GB/s | [板测] |
| 三核 GEMM 实测最好 | 约 4.18 TOPS NPU-only，反推约 23 GB/s | [板测] |
| FP16/BF16 | INT8 的一半；TF32 为 1/4；INT4 为 2 倍 | [TRM] |

这些样本用于第二部分的初步性能估算。小 M、较大 K 的 LLM 形状往往先受 DDR 流量和任务开销限制；
具体瓶颈需要结合形状、布局和完整调用测量确认。

---

## 2. 驱动接口与提交

### 2.1 ioctl 与结构体

默认通过 `/dev/dri/card1` 的 DRM ioctl（设备编号随系统变化）：`_IOWR('d', nr, sizeof(T))`，
ACTION=0x40、SUBMIT=0x41、MEM_CREATE=0x42、MEM_MAP=0x43、MEM_DESTROY=0x44、MEM_SYNC=0x45。
结构体必须 `#pragma pack(1)` 并与驱动字段顺序完全一致，
见 [内部 ABI 头文件](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_internal.h)。**[代码]**

task 描述符（`rknpu_task`，40 字节）中，**以下字段决定本文所述 PC 提交路径的启动和完成判定**：

| 字段 | 用途 |
|---|---|
| 第一个 task 的 `regcmd_addr` | 写入 `PC_DATA_ADDR (0x10)` |
| 第一个 task 的 `regcfg_amount` | 换算后写入 `PC_DATA_AMOUNT (0x14)` |
| 第一个 task 的 `int_mask` | 写入 `INT_CLEAR (0x24)`（注意：不是 `int_clear` 字段） |
| 最后一个 task 的 `int_mask` | 写入 `INT_MASK (0x20)`，并作为完成判据 |

`op_idx`、`enable_mask`、`int_clear`、`regcfg_offset` 在这个驱动里**完全不读**
（见 [rknpu_job.c](https://github.com/allbilly/rknpu_driver/blob/main/rknpu_job.c) 的 PC 提交函数）。
库里按各类 task 的模块契约填写（GEMM 为 0/0xd，
带 RDMA 为 1/0x1d，DPU-only 为 1/0x18）。**[代码]**

### 2.2 驱动怎么启动一条链

`rknpu_job_subcore_commit_pc()` 对每个 core 只做下面几件事：

```text
PC_DATA_ADDR   = first_task.regcmd_addr
PC_DATA_AMOUNT = (first.regcfg_amount + 4 + 1) / 2 - 1     # 单位 128 bit，+4 是 PC 尾
INT_MASK       = last_task.int_mask
INT_CLEAR      = first_task.int_mask
PC_TASK_CON    = ((0x6 | pingpong) << 12) | task_number      # task_number 12 bit
PC_DMA_BASE    = submit.task_base_addr（库里为 0）
PC_OP_EN       = 1，然后 0
```

**之后的 task 全靠硬件沿 regcmd 里的 PC 尾链下去**，驱动不再参与。
该 PC 路径不逐个读取后继描述符的 `regcmd_addr` 来启动 task；
仍应完整填写所有描述符，以便其他驱动路径和诊断使用。**[代码]**

- 单 core 一次最多 4095 个 task（`max_submit_number`），超出由驱动在中断里分批重提。**[代码]**
- 多核时驱动先给每个 core 写 `CNA/CORE S_POINTER (0x1004/0x3004) = 0xe + 0x10000000*core`。**[代码]**

### 2.3 regcmd 编码与 PC 尾

每条 64 bit：`(target << 48) | (value << 16) | reg_offset`。**[板测]**

| target | 模块 | 说明 |
|---|---|---|
| `0x0201` | CNA | bit57 |
| `0x0801` | CORE | bit59 |
| `0x1001` | DPU | bit60 |
| `0x2001` | DPU_RDMA | bit61 |
| `0x4001` / `0x8001` | PPU / PPU_RDMA | 按 TRM 位推断，本项目未用 [推断] |
| `0x0101` | PC 寄存器 | `PC_BASE_ADDRESS`、`PC_REGISTER_AMOUNTS` |
| `0x0081` | op_en 广播 | 写 `0x0008`，value 为各模块使能位 |
| `0x0041` | "VERSION" | TRM：写 op_en 之前必须有一条 `0x0041` 字 |
| `0x0001` | NOP | |

每个 task 的 body 后固定跟 4 个 qword 的尾（`write_chain()`）：

```text
非最后一个 task：
  E(0x0101, 0x0010, next_regcmd_dma & 0xfffffff0)   # PC_BASE_ADDRESS，16B 对齐，bit0=0 表示 PC 取数
  E(0x0101, 0x0014, ceil(next_body_qwords/2) + 1)   # PC_REGISTER_AMOUNTS，给"下一个"task 用
  E(0x0041, 0, 0)
  E(0x0081, 0x0008, op_en)                          # 本 task 的使能，必须放最后
最后一个 task：
  E(0x0001, 0, 0)
  E(0x0101, 0x0014, 0)                              # amount=0 结束链
  E(0x0041, 0, 0)
  E(0x0081, 0x0008, op_en)
```

- 每个 task 占 `align_up(body + 4, 2)` 个 qword，保证下一个 body 16 字节对齐。**[代码]**
- `op_en` 位与 `GLOBAL 0xf008` 相同：b0 CNA、b2 CORE、b3 DPU、b4 DPU_RDMA、b5 PPU、b6 PPU_RDMA。
  常用值：GEMM `0xd`；GEMM+第二操作数 `0x1d`；DPU-only `0x18`。**[板测]**
- **一条链里可以混合不同类型的 task**（GEMM `0xd` 与 DPU-only `0x18` 交替），
  每个 task 尾上写自己的 `op_en`。split-K NPU 归约和 KDA 都这样用。**[板测]**
- 多核时每个 core 有**独立封闭**的链：每个 core 最后一个 task 必须写结束尾，
  不能把 core0 的尾链到 core1 的 task。**[代码]**

### 2.4 中断与"完成"的判定

驱动 IRQ 里把 `INT_STATUS` 每两位合并（`rknpu_fuzz_status`：0x3、0xc、…、0x300、0xc00），
**要求结果恰好等于最后一个 task 的 `int_mask`**，否则打印 `invalid irq status` 并继续等待，
直到超时。

- 链的最后一个模块是 DPU 时用 `int_mask = 0x300`（DPU group 0/1）。库里所有 task 都写 0x300。**[代码]**
- 若最后是 PPU，按规则应为 `0xc00`。**[推断]**
- 超时日志 `require mask: 0x300 ... task counter: N` 里的 N 是驱动读到的 task counter
  （`PC_TASK_STATUS & 0xfff`），可以直接定位卡在第几个 task。**[代码]**

### 2.5 submit 参数

| 参数 | 实际要求 |
|---|---|
| `flags` | `PC(1) \| PINGPONG(4) = 5` 用于 GEMM 链、DPU 链以及 GEMM+DPU 混合链。历史 LUT 初始化与激活/逐元素混合链只在 PC-only (1) 下通过验证，使用 5 的样本超时。**flags 是 task 模板契约的一部分，不能全局固定。** [板测] |
| `core_mask` | 1/2/4/3/7 可用；**5、6 在板测矩阵中超时**，已禁止 [板测] |
| `subcore_task[]` | 单核 mask 用该核自己的槽位；mask=3 用槽 0/1；mask=7 用槽 **2/3/4** [代码] |
| `timeout` | ms。库默认 6000，多核路径 500，probe 常用 1500–2000 以加快失败 |
| `hw_elapse_time` | 驱动 commit→done 的 ktime 差，**单位是 ns**。只有整个 submit 的时间，没有 per-task 时间 [板测] |

早期 Python 底座上的 GEMM 链不带 PINGPONG 会每次挂死 **[历史]**；当时还在每次 submit
前调用 `ACTION RESET`，现在的库只在 ioctl 失败后才 reset。

### 2.6 超时与恢复

- 超时后驱动自己 soft reset，板子一般保持在线。**[板测]**
- 超时后的第一件事是跑一个**已知正确、完整初始化**的 canary（例如 W8 GEMM canary）。
  [allbilly 的 simple_add 示例](https://github.com/allbilly/rk3588/blob/main/examples/simple_add.py) 不完整初始化 BS/BN，
  在 LUT 实验后会误报（返回 5626 而不是 8），不是好 canary。**[板测]**
- **跨 regcmd allocation 的 PC 链会让 NPU 进入不可恢复状态**：当次输出正确、submit 成功，
  但之后显式 reset 也救不回，下一次 submit 超时。一条链的所有 body 必须放在同一个 regcmd buffer。
  **[板测]** [split-K 计划与流水线实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_kn_plan.cpp)
- **不要写 `/sys/.../npu/power/control=on`**：从 suspended 状态强制 resume 曾触发 IOMMU resume
  SError 和 kernel panic。**[板测]**

---

## 3. 内存

### 3.1 分配标志

| 用途 | flags | 原因 |
|---|---|---|
| CPU 读写的数据（A/B/C、系数） | `NON_CONTIGUOUS\|CACHEABLE\|IOMMU = 0x13` | page-backed，CPU 读约 16.9 GB/s；需要显式 sync |
| regcmd | `NON_CACHEABLE = 0` | CPU 只写少量字，免 sync |
| task 描述符 | `KERNEL_MAPPING = 0x8` | 驱动通过 `kv_addr` 读 task |

- **只写 `CACHEABLE (0x2)` 会走强制连续的 `dma_alloc_attrs`，CPU 顺序读只有约 1.84 GB/s。**
  **[板测]** [内存和 domain 实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_core.cpp)
- **不要让 CPU 在 non-cacheable NPU 内存上做逐元素累加**：同一归约 508.8 µs 对 78.8 µs。
  CPU 端的中间结果用 cacheable 内存。**[板测]**
- 所有寄存器里的地址都是 32 bit。每个 IOMMU domain 最多 4 GiB，驱动 0.9.8+ 有 16 个 domain。
  切换是全局的（所有 core 同时切），一次切换约 70–110 µs。
  **一次 submit 用到的 task、regcmd、所有数据必须在同一个 domain。** **[板测]**
- 地址对齐：`DPU_DST_BASE`、`DCOMP_ADDR0`、`PC_BASE_ADDRESS` 的低 4 bit 是保留位，
  所有数据地址至少 16 字节对齐；DCOMP 压缩流用 64 字节对齐。**[代码]**

### 3.2 cache 同步

- `MEM_SYNC` 对 non-cacheable 对象返回 `-EINVAL`。**[代码]**
- 典型开销（驱动 clean）：64 KiB 约 7.6 µs，1 MiB 约 58 µs，16 MiB 约 410 µs。
  用户态 `DC CVAC` 约快 20–25%，但端到端收益一般在 1% 以内。**[板测]**

- NPU runtime suspend（空闲约 3 s）之后，**第一次 sync/submit 会顺带唤醒 NPU**，约多出 220 µs。
  这个时间只会在 sync 和 submit 之间转移，不会消失。**[板测]**
- 多波次流水里"先提交下一波，再 invalidate 上一波输出"约快 3%。**[板测]**

### 3.3 越界读写

- 硬件可能读写超出"逻辑张量"的区域。INT4（INT16 空间卷积）会读取零权重填充的 C16 平面，
  输出后面要留至少 256 KiB 的已映射 guard。**[代码]** `rk_npu_i4_plan.cpp`
- 本文建议的 probe 在输出后填 `0xa5` guard，每次运行后检查。建议新 kernel 同样处理。**[代码]**
- DPU 的目的地址不能与仍在被读取的 EW/MRDMA 操作数重叠
  （ConvAdd 残差、in-place 均未证明安全；INT32 add 的 `dst == src_a` 已验证可用）。**[板测]**

---

## 4. 寄存器状态模型（最容易遇到的问题的一节）

### 4.1 寄存器状态会一直保留

- regcmd **只写不清**。上一个 task、上一次 submit，甚至前一进程写入的部分状态都可能保留。
  `ACTION RESET` 也不会清 LUT 和 BN 状态。**[板测]**

- 每个模块有两组 shadow（ping-pong）寄存器。`S_POINTER=0xe`
  （pointer_pp_en | executer_pp_en | pointer_pp_mode）打开 ping-pong，每个 body 都要写 DPU/RDMA 的 `S_POINTER`。**[代码]**
- 两组 shadow 需要分别初始化，之后的短 body 若只写变化字段，就依赖前驱留下的状态。
  本项目记录的短 body 实验未获得性能收益，库里每个 task 都写完整 body。
  **[板测]**

### 4.2 由此产生的规则

1. **每个 task 显式写出它依赖的所有非默认状态**，包括别的 kernel 可能改过的：
   - `CNA_DCOMP_CTRL / REGNUM / AMOUNT0..15`：INT8 压缩权重 task 会打开解压，后续 FP16 task 必须写回 0；
   - `DPU_OUT_CVT_OFFSET / SHIFT`：LUT/requant 类 task 会改，FP32 GEMM 配置曾因此出错；
   - `BS/BN/EW` 的 cfg 和 operand、`LUT_*`、`CNA_CVT_CON5`、`FC_CON*`、`PAD_CON*`、`0x40c4`。
   FP16 GEMM 的 108 寄存器 body（`make_f16_regs()` 的 plain 分支）就是一个"完整初始化"的参考。**[板测]**
2. **有状态依赖的 task 不能直接独立执行**，也不能在未检查依赖时删减 task：后面的短 body 会继承前面 task 的状态。
   Softmax 删掉 padding task 会超时，或得到错误布局。
   要删减或重排 task，必须把后继依赖的寄存器补进新的前驱（补齐 task 转换时的初始化状态）。**[板测]**

3. 链的安全性是**相邻 task 族之间的 A→B 性质**。`AddRMSNorm→AddRMSNorm` 验证通过，
   不代表 `GEMM→AddRMSNorm` 也可以。**[板测]**

### 4.3 LUT 的生命周期

- LUT SRAM 通过 `LUT_ACCESS_CFG (0x4100)` 选表（bit17 写、bit16 表号），
  再连续写 `LUT_ACCESS_DATA (0x4104)`，地址自动递增。每张表 513 项，两张（LE/LO）共 1026 次写，约 60 µs。**[板测]**
- **只写表不构成 task**：表写入后面必须跟一个真正计算的 DPU task，并保证 PC 尾和模块使能配置完整。**[板测]**
- 表内容**能跨 reset、跨进程保留**，**但 NPU runtime suspend 后会丢**（实测 4096 点中 940 点变化）。
  所以要么每次使用前重载，要么保证 NPU 不进 suspend；后者**不能**靠写 PM control 实现（见 2.6）。**[板测]**
- 同一个 core 上只有一套 LUT；SiLU、GELU、rsqrt 互相冲突。**[板测]**

---

## 5. 数据布局

### 5.1 基本单位：16 字节 atom

在 `DPU_DATA_FORMAT.mc_surf_out=0` 的已验证路径中，每个像素的每个 surface 占 16 字节。
一个 surface 含 `C2 = 16 / sizeof(elem)` 个通道。**[板测]**

| dtype | C2 | 本项目名称 |
|---|---:|---|
| INT4 | 32 | K32 |
| INT8 | 16 | K16 / "block-16" |
| INT16、FP16、BF16 | 8 | K8 / "block-8" |
| INT32、FP32、TF32 | 4 | N4 |

native（NC1HWC2）张量的地址：

```text
offset(c, pixel) = (c / C2) * SURF_STRIDE + pixel * 16 + (c % C2) * sizeof(elem)
紧凑 native：SURF_STRIDE = pixels * 16
```

矩阵的像素就是 M 行，所以"native A/C"就是 `[C/C2, M, C2]`。
**不同 dtype 的 native 不能混用**：FP16 block-8、INT8 block-16、INT32 block-4 即使字节数一样，也是不同布局。**[板测]**

### 5.2 各 dtype 的 GEMM 布局（已记录的验证路径）

| dtype | A normal | A native | B | C/D native | K 对齐 | N 对齐 |
|---|---|---|---|---|---:|---:|
| INT8→INT32/FP32 | `[M, align32(K)]` | `[K/16, M, 16]` | `[N/32][K/32][32n][32k]` | `[N/4, M, 4]` | 32 | 32 |
| INT8→INT8（requant） | 同上 | 同上 | 同上 | `[N/16, M, 16]` | 32 | 32 |
| FP16/BF16 | `[M, align32(K)]` | `[K/8, M, 8]` | `[N/16][K/32][16n][32k]` | `[N/8, M, 8]` | 32 | 32 |
| TF32→FP32 | `[M, align16(K)]` | `[K/4, M, 4]` | `[N/16][K/16][16n][16k]` | `[N/4, M, 4]` | 16 | 32 |
| INT4→INT16 | — | `[K/32, M, 32]` | `[N/64][K/32][64n][32k]` | `[N/8, M, 8]` | 32 | 64 |

- B 必须由 CPU 预先打包，硬件不会重排。CPU 打包函数和对应测试是检查实际布局的入口。表内 K/N 使用填充后的维度，
  INT4 行中的尺寸按逻辑元素计数，实际存储为每字节两个有符号 4 bit 值。
- normal C（row-major）写回对大 N 很慢：`M128,K1024,N8192` 为 0.55 TOPS，
  native C 为 1.5 TOPS。**输出布局是最大的性能开关。** **[板测]**
  [CPU 布局转换实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_cpu_kernels.cpp)
- INT8 requant 输出 `[N/16,M,16]` 就是下一个 INT8 GEMM 的 native A，可以零拷贝串接。
  但 INT32 native C `[N/4,M,4]` **不是**下一个 GEMM 的 A。**[板测]**
- FP16 native D `[N/8,M,8]` 与下一个 FP16 GEMM 的 native A、Add+RMSNorm 的输入输出相同。**[板测]**

### 5.3 用 cube 几何和 stride 表达 normal/native 输出

DPU 输出地址可以统一写成（所有已验证配置都符合）：**[推断]**

```text
addr(c, h, w) = DST_BASE + (c / C2) * DST_SURF_STRIDE
              + h * (W + NOTCH) * 16 + w * 16 + (c % C2) * bytes
W = DATA_CUBE_WIDTH + 1，NOTCH = DATA_CUBE_NOTCH 的低 13 bit
```

| 输出 | cube W×H | DST_SURF_STRIDE | NOTCH |
|---|---|---|---|
| row-major `[M, Npad]` | 1 × M | 16 | `Npad/C2 - 1`（行跨度 = Npad*bytes） |
| native `[N/C2, M, C2]` | M × 1 | `M*16` | 0 |
| panel（H×W 分组） | W × H | `W*16` | `(N/C2 - 1)*W` |

- NOTCH 只有 13 bit，panel 与超宽 row-major 会碰到上限（例如 N8192 时 W=8 无法表达）。**[板测]**
- 早期把 notch 截断到 `13*32=416` 列的写法，在多行 M tile 且 N>416 时出错。
  正确值就是行跨度减一，不需要截断。**[历史]**
- row-major 的一行就是"M=1 的 native"。DPU 复制 kernel 只改 `DST_SURF_STRIDE = rows*16`，
  就能在 row-major 与 native 之间切换目标布局。**[板测]** [行复制寄存器生成器](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_moe_regs.h)

### 5.4 CORE atomic、`SURFACE_ADD` 与 `BS_OW_CFG.size_e`

TRM 对 `SURFACE_ADD`（"how many surfaces in a row"）和 `size_e`（"how many 8 channels"）
的解释对不上实际行为。下面的规律覆盖了所有已验证配置：**[推断，逐项板测一致]**

```text
atomic_ch       = CORE 每次输出的通道数：INT4 64，INT8 32，FP16/BF16 16，TF32 16，DPU-only = 1 个 surface
surf_per_atomic = atomic_ch / C2_out
SURFACE_ADD     = surf_per_atomic * DST_SURF_STRIDE      # 相邻 atomic 之间的字节距离
size_e_0/1/2    = surf_per_atomic - 1                    # BS_OW_CFG[4:2]/[7:5]/[10:8]
```

| 组合 | surf/atomic | SURFACE_ADD（native） | BS_OW_CFG |
|---|---:|---|---|
| INT8 GEMM → INT32/FP32 | 8 | `M*128` | `0x7fc`/`0x7fe` |
| INT8 GEMM → INT8 | 2 | `M*32` | `0x126` |
| INT8 GEMM → FP16 | 4 | `M*64` | `0x36e` |
| FP16/BF16 GEMM → FP16/BF16 | 2 | `M*32`；row-major 时 `0x20` | `0x126`（`size_e=1`） |
| TF32 GEMM → FP32 | 4 | `M*64`；row-major 时 `0x40` | `size_e=3` |
| INT4 GEMM → INT16 | 8 | `M*128` | `0x7fe` |
| DPU-only（offline，flying_mode=1） | 1 | `rows*16` | `0x2` |

- 错误值的现象都很典型：INT8→INT8 用 `BS_OW_CFG=0x2` 只写出第一个 surface；
  `SURFACE_ADD=M*16` 让相邻 atomic 互相覆盖。**[板测]**

- 早期"INT8 GEMM 直接输出 FP16 会丢一半通道或挂死"的结论 **[历史]**，
  根因就是当时没按上表配置 4 个 surface。但 INT8 CACC 直出 FP16 还有第 7.4 节的零值异常，仍不推荐。

---

## 6. CNA（取数、CBUF、输入转换）

### 6.1 精度与模式

- `CNA_CONV_CON1 [9:7] proc / [6:4] in` 与 `CORE_MISC_CFG [10:8]` 必须一致：
  INT8=0、FP16=2、BF16=3、INT4=6、TF32=7。例如 FP16 配置为 `0x120`，BF16 为 `0x1b0`。**[板测]**
- `CONV_CON1 bit29 group_line_off`：normal A 置 1；native A 在 full-M 时为 0，split-M 时为 1。
  TRM 说它"只影响效率"，但它与下面 DMA stride 的组合是配套的，不能单独改。**[板测]**
- `CONV_CON3 = 0x9`（1×1 stride）。`nn_mode [30:28]` 保持 0。**[板测]**
- `CORE_MISC_CFG bit0 (qd_en)`：INT8/INT4 GEMM 置 1，FP16 plain 为 0，旧 FP16 融合路径为 1。**[代码]**

### 6.2 几何寄存器（矩阵乘映射）

| 寄存器 | normal A | native A（full M） | native A（split M，m<M） |
|---|---|---|---|
| `DATA_SIZE0` | `(1<<16) \| m`（W=1, H=m） | `(m<<16) \| 1` | 同左 |
| `DATA_SIZE2` | 1 | m | m |
| `DATA_SIZE3` | m（输出像素数） | m | m |
| `DMA_CON1`（line stride） | `row_bytes/16` | `0x200` | `M` |
| `DMA_CON2`（surf stride） | 0 | `0x10000000 - 3*M`（28 bit 编码的负偏移） | `M - m` |
| `CBUF_CON1`（entries） | `ceil(row_bytes/64)` | `ceil(m*row_bytes/64)` | 同左 |
| `feature_grains` | 见 6.4 | 2 | 2 |
| `CORE_DATAOUT_SIZE_0` | `(m-1)<<16` | `m-1` | `m-1` |

- native A split-M 若沿用 full-M 的 `0x200 / 0x10000000-3M`，结果是 WRONG 或 SUBMIT 失败。
  split-M 的已验证配置为 `DMA_CON1=M`、`DMA_CON2=M-m`。**[板测]** [INT8 寄存器生成实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_matmul_i8.cpp)
- `DATA_SIZE1 = ((span-1) << 16) | extent`，两半**含义不同**：高 16 位是 feature DMA 的实际通道跨度，
  低 16 位是计算用的对齐 K。FP16 PV 可以设成 `align8(S)-1 / align32(S)`，
  直接读 Softmax 的紧凑输出；**两半都写紧凑值会从第二个 N16 块起出错**。**[板测]**
- `DATA_SIZE0` 的 height 只有 11 bit，width 也有位宽上限；FP16 native M tile 上限取 1022。**[代码]**
- `WEIGHT_SIZE0 = row_bytes*Ntile`，`WEIGHT_SIZE1 = row_bytes`（19 bit），
  `WEIGHT_SIZE2 = (1<<24)|(1<<16)|Ntile`（kernels 14 bit）。已有单 task 样本覆盖 N=8128；这不表示所有 dtype、K 和布局都支持该 N。**[板测]**
- `FC_DATA_SIZE0 = DATA_SIZE0`，`FC_DATA_SIZE1 = align K`。**[代码]**
- `DMA_CON0 = 0x000f000f`（burst 16）在所测配置中表现最好；`0x7` 更慢，更大的值未观察到收益。**[板测]**

### 6.3 CBUF

- 每核 **12 bank × 32 KiB**；不能用某个位域示例的 bank 数替代实际预算。
  `CBUF_CON0 = (weight_banks << 4) | data_banks`，两者之和为 12。例如 `0x84` 表示 4 个 data bank、8 个 weight bank。**[板测]**
- 1 个 entry = 64 字节，1 bank = 512 entries。`CBUF_CON1.data_entries` 是一个 feature 行（native 时是整个 tile）占的 entry 数。**[推断]**
- weight 侧至少要放下一个输出通道组：INT8 为 `ceil(K*32/32K)` bank，FP16 为 `ceil(K*2*32/32K)` bank，
  窄 K 时最少 2 个。要做 ping-pong 需要两倍空间。**[板测]**
- 已验证的 A tile 预算：INT8 normal ≤ 8 bank（写 16 bank 会 WRONG/SUBMIT）；native A ≤ 4 bank；
  FP16 normal 用 `12 - weight_banks`。**[板测]**
- `weight_reuse / data_reuse` 需要整个 tile 的权重都放进 CBUF，这会迫使 N tile 变成 32，
  任务数暴增，整体约慢 2.4 倍。**[板测]**

### 6.4 `feature_grains`（`CONV_CON2[13:4]`）

TRM 定义是"卷积开始前需要缓冲的行数"。实测：**[板测]**

- normal A：`min(m+1, 两个 bank 能放下的行数)`。K>4096 且 m≥16 时用 30（当前库的大 K 配置）。
- native A 和 INT4：2。
- 太小会**静默算错**（全 N、normal A 时 2/4/8 都出错；FP16 在 K=3840 附近也出过错）；
  也不是越大越好（16 比 32/64 略快）。

### 6.5 输入转换 CVT（`CVT_CON0..5`）

- 是**整数定点**转换器：INT16→INT8 可以精确完成 scale/truncate，truncate 用 RNE。**[板测]**
- **它不会把 IEEE FP16 量化成 INT8**：FP16 位模式被当成整数，正负分别饱和到 ±127。**[板测]**
- `CVT_CON5` 是按通道的使能掩码；为 0 时转换关闭。GEMM 里一律写 `CVT_CON0 = 0xb`
  （有符号、bypass），`CVT_CON1..4 = 1<<16`。**[代码]**

### 6.6 DCOMP（权重解压）

- `DCOMP_ADDR0` **在非压缩模式下也是权重基址**（GEMM 的 B 地址就写在这里）。**[板测]**
- raw 模式：`DCOMP_CTRL=0, REGNUM=0, AMOUNT*=0`；压缩：`CTRL=1`，`REGNUM` 实际是 8 个 nibble 的码表，
  `AMOUNT0..15` 是 16 路数据量。实现见 [DCOMP 编解码](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_dcomp.cpp)。**[板测]**
- 压缩权重可由本库创建并用于已验证的多核 N 分片；压缩布局按分片准备，
  首次使用新分片时需计入准备开销。**[板测]**

### 6.7 未验证的功能

FC zero-skip（`FC_CON0/1/2`，TRM 要求 DPU `conv_mode=3`、BS bypass 等配合）、deconv、dilation、
pad、`surf_mode`（0–3 性能相同；在 native A 下 2/3 会静默出错）。**[TRM]/[板测负结果]**

---

## 7. DPU

### 7.1 流水顺序与数据源

```text
主数据（CORE 或 MRDMA）→ BS[ALU → MUL → ReLU] → BN[ALU → MUL → ReLU] → EW[CVT → ALU/MUL] → LUT
                        → OUT_CVT（offset/scale/shift）→ 精度转换 → WDMA
```

- BS 的操作数来自寄存器，或来自 BRDMA（`BS_BASE`，`BRDMA_CFG[4:1]` 选 ALU/CPEND/MUL/TRT）；
  BN 同理，来自 NRDMA；EW 来自 ERDMA（加 MRDMA，见第 8 节）。**[板测]**
- `*_CFG` 的 bypass 位：`0x53` = BS/BN 全 bypass；`0x42` = 只开 MUL；`0x383` = EW 全 bypass。**[板测]**
- FEATURE_MODE_CFG：GEMM 为 `0x1e4`（burst16、输出到内存、主数据来自 CORE），offline 为 `0x1e5`。**[板测]**
- `DATA_FORMAT = out<<29 | in<<26 | proc`，编码为 INT8 0、INT16 1、FP16 2、BF16 3、INT32 4、FP32 5、INT4 6。
  TF32 GEMM 的 DPU 一侧写 FP32 (5)，不能也写 7。**[板测]**
- INT8 GEMM 的 DPU 只改 `out`（`in/proc=0`），INT32 写 `4<<29`，FP32 写 `5<<29`。**[板测]**

### 7.2 各级运算的实际语义

| 功能 | 配置 | 实测语义 |
|---|---|---|
| BS/BN MUL（寄存器） | `*_MUL_CFG[31:16]` | 浮点路径为 **FP16 位模式**（例如 `half(scale)<<16`）；整数路径为 Q 定点，正移位 `[13:8]`，负移位在 `DATA_FORMAT[9:4]/[15:10]` [板测] |
| BS/BN MUL（外部，逐通道） | BRDMA/NRDMA，`BRDMA_CFG=0x8` / `NRDMA_CFG=8` | INT32/FP32 主数据：系数布局 `[N/4, 8]` 的 16 bit 值（每 16 B 的前 4 个有效）[板测] |
| BS ALU（外部，逐通道） | `BS_CFG` alu_src=1 | INT32 bias 沿 M 广播，只能做 per-channel，不能加整张量 [板测] |
| INT8 逐通道 bias/零点修正/scale | `BS_CFG=0x20140`, `BRDMA_CFG=0xe` | 每 8 个通道 64 B：`[0..31]` ALU int32 ×8、`[32..47]` CPEND int16 ×8、`[48..63]` MUL int16 ×8。ALU 是 MUL 之前加的 bias；CPEND = `-w_zp`，乘的是输入和（非对称零点修正） [板测] |
| EW ALU | `EW_CFG[19:16]` | 已验证：2 Add、3 Div（main/external）、4 Minus，以及 Max（`maxabs` 链）[板测]；0 max、1 min、5 abs、6 neg、7 floor、8 ceil 为 [TRM] |
| EW MUL 的第二操作数 | ERDMA，32 bit 槽 | **低 16 bit 放 FP16**（1.0 = `0x00003c00`）。直接放 IEEE FP32 会得到错误值甚至非有限值 [板测] |
| EW ADD/SUB 的第二操作数 | ERDMA，32 bit | IEEE FP32 可以直接用（FP32 main） [板测] |
| DPU min/max 归约 | `DATA_CUBE_HEIGHT[24:22] minmax_ctl` | 7 = 行最大，5 = 行最小；结果的前 M 个 FP16 是紧凑的行标量 [板测] |

**总结：DPU 的乘法系数实际是 16 bit 的。** "FP32 数据通路"不代表乘数是 FP32。
dequant 的 scale 只能是 FP16 精度，库里的 `RK_NPU_W8A8_NPU_DEQUANT` 因此是显式的精度折中。**[板测]**

### 7.3 EW operand converter（`EW_CFG bit8`）：看数据路径，没有统一规则

| 场景 | 正确设置 | 错误设置的现象 |
|---|---|---|
| FP16 GEMM flying + FP16 外部张量 ADD | cvt **开启**：`0x108202c0`，`EW_CVT_SCALE=1, OFFSET=0` | cvt bypass 时**加法静默忽略操作数**，D == A@B [板测] |
| FP16 flying MUL | `0x108003c4`（cvt bypass） | — |
| offline INT32 + INT32 ADD | cvt **bypass**：`0x10c203c0` | 打开 CVT 的 `0x10c202c0` 会**抽稀**操作数（间隔读取） [板测] |
| offline FP32 MUL（32 bit 槽） | `0x10c003c4` | 去掉 cvt bypass 会超时 [板测] |

早期"FP16 EW 只能乘不能加"的结论就是 cvt 位设错造成的 **[历史]**。

### 7.4 OUT_CVT

- `OUT_CVT_SCALE[15:0]` 是定点 scale，bit16 为 `fp32tofp16_en`（FP16/BF16 输出时置 1）。
  `OUT_CVT_SHIFT`：`[11:0]` shift、`[19:12] minus_exp`（输出 = 定点值 × 2^-minus_exp，可以得到真正的小数 FP16）。**[板测]**
- **零值异常**：INT8 CACC 直接经 OUT_CVT 输出 FP16 时，精确为 0 的累加值变成 `2^(17 - minus_exp)`。
  `offset=+1` 的绕法对 -1 失效，不安全。另一个已记录的 INT8→FP16 转换配置将 0 映射成 `2^-13`；不能把这类配置当作精确转换。**[板测]**
- 稳定的浮点输出路径：INT8 GEMM 先输出 FP32，再由 FP32 DPU 做 FP32→FP16；
  需要 requant 时应独立验证 INT8→浮点转换的零值、极值和舍入行为。**[板测]**
- `minus_exp` 要按累加值的范围选，过大会饱和。**[板测]**

### 7.5 LUT

- `LUT_CFG=0x68`、`LUT_INFO` 的 index select 字段、`LE/LO start/end`、slope 共同决定查表。
  已验证配置见 `rk_npu_moe_regs.h`（SiLU：两张表覆盖 [-8,8]，步长 1/64，输出 ×4096）
  ；rsqrt 还需单独验证误差与输入范围。**[板测]**
- **查表 index 用的是 BN MUL 之后的值，BN_ALU 的平移没有作用在 index 上**。要平移区间，
  用 LE/LO 的 start/end 描述。每张表最后两项留作 guard，避免上端点落进插值分数。**[板测]**
- LUT 输出是 int16 整数码。FP16 输出靠 `minus_exp` 缩回，但 FP16 路径同样有零值异常；
  FP32 输出路径保留正确的 0，再用一个 BS 标量 MUL 除以表的倍率。**[板测]**
- EW_CFG `0x302` = 只用 LUT（EW 运算 bypass）。**[板测]**

### 7.6 几何与大小限制

- cube W/H/C 都是 13 bit（-1 编码）。
- 库里 offline 任务每个 task 最多 1024 通道（`DPU_CHANNELS`）；
  一个 32768 值的 flat task 超时，4096 值的分块通过。**这些不一定是硬件上限，但超出就是未验证区。** **[板测]**

---

## 8. DPU_RDMA

### 8.1 四路读引擎

| 引擎 | 基址 | 读什么 |
|---|---|---|
| MRDMA | `SRC_BASE 0x5018` | offline 主数据；flying 时是 EW 操作数的"第一半 surface" |
| BRDMA | `BS_BASE 0x5020` | BS 逐通道系数 |
| NRDMA | `BN_BASE 0x502c` | BN 逐通道系数 |
| ERDMA | `EW_BASE 0x5038` | EW 操作数 |

`RDMA_FEATURE_MODE_CFG`：`[17:15] in`、`[14:11] burst`、`[10:8] comb_use`、`[7:5] proc`、
bit4 mrdma_disable、bit3 fp16→fp32、bit0 flying。常用值：FP16 flying 操作数 `0x17d40`，
FP32 offline `0x2f8a1`，INT32 offline `0x27881`，INT8 offline `0x7801`。**[板测]**

### 8.2 已验证 flying GEMM 的双 surface 操作数读取

GEMM 的 CORE 每个 atomic 输出 2 个（FP16）或更多 surface。EW 操作数要同时从两个引擎读：**[板测]**

```text
SRC_BASE (MRDMA) = operand_base
EW_BASE  (ERDMA) = operand_base + M * 16        # 下一个 surface
EW_SURF_STRIDE / SURF_NOTCH / EW_SURF_NOTCH 按 surface 距离设置
```

- **在本节已验证的 M≥4 几何中，只用 ERDMA 会超时**（DPU 等数据，永远不发 done）。**[板测]**
- 重绑定操作数时，MRDMA 与 ERDMA 地址必须一起更新；保留旧地址会导致错误或超时。**[板测]**
- **M=1 是特例**：已有测试配置使用 `ERDMA_CFG=0x4`（data_mode 0）、`MRDMA_SRC=0`、ERDMA 指向操作数基址；
  M≥4 用 `0x40000004` 加双地址。M=1 套用双地址时 submit 成功，但 254/256 个值错误。**[板测]**
- 所测 INT8 GEMM flying + INT32 操作数配置**仅覆盖一半通道**（32 通道 atomic 对 4 元素的 16 B 行），
  调任何 stride 都修不好。需要 INT32 加法时改用 offline DPU task。**[板测]**
- 旧的 FP16 融合路径用了一种"half-surface"操作数布局（每 16 通道 24*m 个元素），
  并要求 `SURF_NOTCH = 2*m`，见 `rk_npu_matmul_f16.cpp` 顶部注释。**[板测]**

### 8.3 ERDMA 读模式

- `ERDMA_CFG[31:30] data_mode`：0 per-channel（只读 N 个值，沿像素广播）、1 per-pixel（整张量）、2 per-channel-by-pixel。**[板测部分]**
  - per-channel 广播已验证：`EW_CFG=0x00c003c4, ERDMA_CFG=0xc, EW_SURF_STRIDE=16`。**[板测]**
  - **ERDMA 不能把一个标量广播到整行**：per-channel/per-pixel 仍按元素逐个读，
    stride=0 也不会重复第一个 atom，mode 2 的组合超时。
    本项目采用 CNA/CORE 卷积进行行广播（单位矩阵或全 1 权重）展开。**[板测]**
- `comb_use`：MRDMA/ERDMA 读同一份数据。在 FP16→INT8 转换里打开它，会把 ERDMA 让给第二个 surface，
  原来的行操作数就丢了。**[板测]**
- `EW_BASE` 只有 32 bit；`ERDMA_CFG bit0 = 1` 表示禁用 ERDMA（纯复制 task 这样写）。**[板测]**

### 8.4 用 `SURF_NOTCH` 做行 gather

offline 读 native `[N/C2, M, C2]` 中的第 r 行：

```text
SRC_BASE  = native_base + r * 16
cube W=H=1, C=N
SURF_NOTCH (0x504c) = (M - 1) * 16      # 跳过同一 surface 里的其他 M-1 行
```

配合 `DST_SURF_STRIDE=16` 就能得到紧凑行。逐 token dequant、MoE 行抽取都基于这个技巧。**[板测]**

---

## 9. PPU、DDMA/SDMA、GLOBAL

- PPU/PPU_RDMA（pooling）：本文讨论的 Softmax/RMSNorm 配置不使用这些模块，本库未实现相应路径。**[TRM]**
- DDMA/SDMA（`0x8000/0x9000`，outstanding、仲裁、QoS）：本库未验证通过 regcmd 配置这些寄存器的访问路径，
  不应直接套用其他模块的寄存器写法。**[TRM]**
- RW amount 计数 ioctl 在 RK3588 驱动里没有接硬件，所测驱动路径返回 0。测量带宽需要独立的 NoC/DDR 监测工具，并区分逻辑张量流量与实际总线流量。**[板测]**
- 本项目历史 SRAM 配置的实测带宽低于对应 DDR 配置；是否有收益需要结合驻留和数据搬运成本重新测量。**[板测]**

---

## 10. 数值行为

| 路径 | 行为 | 等级 |
|---|---|---|
| INT8×INT8 | INT32 精确累加；FP32 输出在 \|acc\|<2^24 时逐位等于 INT32 | [板测] |
| INT4×INT4 | 输出 **INT16** partial，会饱和，所以 K tile 要按权重证明上界（K>480 需要证明） | [板测] |
| FP16 | 高精度累加，输出经 fp32tofp16 | [板测] |
| BF16 | 输入使用 host RNE 转换；已有对照表明累加精度高于 BF16，少数输出在舍入分界处相差 1 ULP；不能要求任意浮点参考逐位一致 | [板测] |
| TF32 | **输入截断低 13 bit**（不是 RNE）；输出 FP32 | [板测] |
| DPU 乘数 | 16 bit（见 7.2） | [板测] |
| CNA CVT | 只做整数定点转换 | [板测] |
| INT8 CACC→FP16 OUT_CVT | 零值异常（见 7.4） | [板测] |
| exp 近似 | FP16 位技巧：`bits = round(z*1477 + 15300.624)` | [板测] |
| rsqrt 近似 | BF16 `0x5f37 - (bits>>1)` 种子 + 3 次 Newton（在已测 M=1 样本中，2 次与 3 次输出逐位相同） | [板测] |

---

## 11. 性能特性

### 11.1 GEMM：B 会被每个 M task 重读

一个 M tile 是一个 task。每个 task 都从 DDR 流一遍完整的 B（`K × Ntile`），因此：

```text
B 流量     ≈ ceil(M / Mtile) × K × N × sizeof(weight)
吞吐上限   ≈ B 带宽 × 2 × Mtile  (ops/s)
```

上式适用于未压缩权重；DCOMP 应使用实际压缩分片的字节数，不能直接套用逻辑 INT8 大小。
历史 INT8 样本中，`K=8192 → Mtile=16`：每多一个 task 增加约 100 µs（1 MiB B），8 个 task 对应约 256 GOPS。
所以大 K、小 N 的形状先是 **CBUF/Mtile 导致的带宽瓶颈**，再叠加固定任务开销。**[板测]**

对策是**沿 K 拆分**，让每一波都有 Mtile=128（INT8 normal A、M=128 时 Kt=2048 是结构拐点）。
`K8192/N512` 由此从 3.75 ms 降到 0.76–0.8 ms（含 CPU 归约，多核）。**[板测]**

### 11.2 固定开销

| 项目 | 量级 | 等级 |
|---|---|---|
| 一次 blocking submit（小任务，含 ioctl） | 约 20–35 µs | [板测] |
| 同样三个算子：三次 submit vs 一条 PC 链 | 212–252 µs vs 28–88 µs | [板测] |
| 链内每多一个 DPU-only task | 约 0.3–0.6 µs HW | [板测] |
| 链内每多一个小型 CNA/CORE 广播 task | 约 0.9 µs HW | [板测] |
| LUT 全表加载 | 约 60 µs | [板测] |
| IOMMU domain 切换 | 约 70–110 µs | [板测] |
| runtime suspend 后首次使用 | +约 220 µs | [板测] |

### 11.3 NPU 并不总是更快

| 工作 | NPU | CPU 基线（逐行注明） | 等级 |
|---|---|---|---|
| 动态 per-token 量化 M32/K2560 | 8 task 链，仅设备计时 203 µs | 22 µs（4 大核 NEON） | [板测] |
| KDA decode（16 head） | 297 µs/层 | 104 µs/层（4 大核 NEON） | [板测] |
| RMSNorm M1/D4096（历史配置） | 243 µs | 32 µs（NumPy） | [板测] |
| Softmax M32/S512 | 仅设备计时 1.07 ms | 10.4 ms（NumPy） | [板测] |

规律：小规模逐元素运算和 decode 的 M=1 通常输给 CPU；有 GEMM 复用、M 大、
或张量本来就留在 NPU 上时，NPU 才有意义。**CPU 与 NPU 共享 DDR**：CPU 归约单跑 1.20 ms，
和 NPU 重叠时变成 2.14 ms。**[板测]**

---

## 12. 问题清单（现象 → 原因 → 处理）

| 现象 | 原因 | 处理 |
|---|---|---|
| submit 超时，`require mask: 0x300` | 各种非法配置；也可能是 flags、双 surface、跨 allocation 链 | 看 task counter 定位第几个 task；逐项回退 |
| 融合 EW 挂死 | flying GEMM 的操作数只走了 ERDMA | MRDMA + ERDMA 双地址（M=1 例外） |
| 加法没有生效，D == A@B | FP16 flying ADD 时 EW cvt 被 bypass | 打开 cvt，SCALE=1 |
| INT32 操作数被抽稀 | offline INT32 ADD 时 cvt 开着 | cvt bypass `0x10c203c0` |
| 只覆盖一半通道 | INT8 CORE 的 32 通道 atomic 对 INT32 操作数 | 改用 offline DPU task |
| 输出只有第一个 surface / 块互相覆盖 | `size_e`、`SURFACE_ADD` 与 atomic 不匹配 | 按 5.4 表计算 |
| 乘法结果离谱、非有限 | EW 乘数写了 IEEE FP32 | 32 bit 槽低 16 位放 FP16 |
| 精确 0 变成 16384/4096/128 | INT8 CACC→FP16 OUT_CVT 零值异常 | 先输出 FP32，再做 FP32 DPU 转换 |
| 上一个 kernel 之后结果错误 | 状态残留：DCOMP、OUT_CVT、BS/BN/LUT | 每个 body 写完整状态 |
| simple_add 返回 5626 | LUT 实验残留 BN/BS 状态 | 用完整初始化的 canary |
| 删减有前驱依赖的 task 后超时/错误 | 短 body 依赖前驱状态 | 补齐前驱状态和寄存器初始化 |
| native A 在 M 分块后 WRONG/SUBMIT | split-M 仍用 full-M 的 DMA 配置 | `DMA_CON1=M, DMA_CON2=M-m, CON1=0x20000000` |
| N>416 的多行 M tile 出错 | DPU notch 被截断 | notch = 行跨度/16 - 1 |
| 大 K 算错 | M tile 超出 CBUF 预算或 grains 太小 | 按 6.3/6.4 的已验证预算 |
| 混合 LUT 链超时 | 用了 PC\|PINGPONG | 用 PC-only |
| 当次正确，下一次 submit 就挂，reset 也没用 | PC 链跨 regcmd allocation | 一条链放在一个 buffer |
| 第二次运行结果变了 | 内存规划复用了 graph 输入 | 输入默认保活，donate 必须显式 |
| mask 5/6 超时 | 该测试环境中未通过验证 | 只用 1/2/4/3/7 |
| 空闲一段时间后 LUT 结果错误 | runtime suspend 清掉了 LUT | 使用前重载；不要写 PM control |
| 写 power/control 后 kernel panic | IOMMU resume SError | 禁止 |
| CPU 处理 NPU 输出特别慢 | non-cacheable 或只写了 CACHEABLE(0x2) | 用 0x13 并 sync |
| 行广播读出后面全是 0 | ERDMA stride=0 不会广播 | 用 CNA/CORE 卷积广播 |
| 单个大 DPU task 超时 | 超出已验证的通道/元素数 | 分块（≤1024 通道、≤4096 值） |

---

## 13. 尚未解决的问题与矛盾

- **DPU 复制 task 与 GEMM 交错成一条链，在三核上超时**（core0 单核正常，改成 PC-only 也失败），
  当前实现使用两次 submit。而 split-K DPU 归约链和 KDA 的 GEMM+DPU 混合链在三核上工作正常。
  差别尚未定位。[MoE 任务调度实现](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_moe_w8.cpp)
- 为什么同为混合链，所测历史 LUT 混合链仅在 PC-only 下通过验证，而其他混合链用 PC|PINGPONG 正常，机制未知。
- offline INT32→INT8 直出、offline 两轴 scale 直接写 INT8，都超时，尚未获得通过验证的配置。
- flying FP32 EW 融合（TF32 GEMM + FP32 加法）超时，尚未获得通过验证的配置。
- native A full-M 时 `DMA_CON1=0x200`、`DMA_CON2=0x10000000-3M` 的含义未知，当前只能在已验证几何中使用，不能外推到其他布局。
- `feature_grains`、M tile 在大 K 边界的完整安全公式未建立。2026-07-10 曾发现 K8192 下
  15 行尾块出错；这些尾块需要独立验证。
- 旧记录（2026-08-27）称 raw 非 core0 和并行 submit 会超时/硬锁；当前库已有通过验证的三核 N 拆分实现。
  这类旧结论要按当时的驱动与代码来理解。

## 14. 公开实现与规范

- [本库源码](https://github.com/happyme531/rknpu2_matmul_open)及 [MM/BMM 接口说明](https://github.com/happyme531/rknpu2_matmul_open/blob/main/MM.md)。
- [Rockchip RK3588 TRM Part 1](https://dl.khadas.com/products/edge2/datasheet/rockchip-rk3588-trm-v1.0-part1.pdf)与 [Part 2](https://dl.khadas.com/products/edge2/datasheet/rockchip-rk3588-trm-v1.0-part2.pdf)，由 Khadas 托管。
- [RKNPU 驱动 PC 提交实现](https://github.com/allbilly/rknpu_driver/blob/main/rknpu_job.c)。本文的驱动细节以该实现为检查入口，其他版本应重新核对。
- [Mesa Rocket 后端](https://github.com/chaotic-cx/mesa-mirror/tree/main/src/gallium/drivers/rocket)及其寄存器定义。

---

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)
