# 第二部分：怎样给 RKNPU2 写一个新 kernel

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)


第一部分讲硬件"是什么样"，本文讲"怎么做"。这里的方法是从本项目几十个 kernel/probe 里总结出来的：
INT8/FP16/BF16/TF32/INT4 GEMM、split-K 归约、requant/dequant 融合、Add+RMSNorm、Softmax、
SiLU LUT、gather/复制、KDA 递推等。

核心方法：**把问题改写成硬件原本就会做的事（卷积 + 固定后处理），
从已经验证过的配置出发，一次只改一个变量，用能区分对错的数据逐位验证，
最后用完整调用时间而不是硬件时间判断值不值得。**

---

## 1. 心智模型

### 1.1 kernel 是什么

在这个 NPU 上，"kernel" 不是一段在 NPU 上执行的程序，而是：

```text
一组 task（每个 task = 一串寄存器写入 + 4 qword 的 PC 尾）
  + 已按硬件布局打包好的数据（A、B、系数、LUT）
  + 一次 submit
```

每个 task 就是"配置一次固定流水线并启动它"。所以写 kernel 要回答的问题是：

1. 这个计算能拆成哪几次**流水线调用**（GEMM/卷积、带后处理的 GEMM、DPU-only 逐元素/归约、复制）？
2. 每次调用的**数据布局**是什么？相邻两次调用能不能零拷贝衔接？
3. 哪些部分留给 CPU 更划算？

### 1.2 硬件能直接做的"原语"

| 原语 | 用什么实现 | 参考实现 |
|---|---|---|
| 矩阵乘 / 1×1 卷积 | CNA+CORE+DPU，op_en `0xd` | `rk_npu_matmul_i8.cpp`、`rk_npu_matmul_f16.cpp`、`rk_npu_i4_regs.cpp` |
| GEMM + 逐通道 scale/bias/requant | 同一 task 里 DPU 的 BS/BN/OUT_CVT | DPU BS/BN 与 OUT_CVT 配置 |
| GEMM + 逐元素张量运算 | flying EW + 双 surface RDMA，op_en `0x1d` | 本库 FP16 fused MUL/ADD |
| GEMM + 激活 | 同一 task 用 LUT（需要预先装表） | SiLU LUT，`rk_npu_moe_regs.h` |
| 逐元素 A∘B（张量） | offline DPU：MRDMA 主 + ERDMA 操作数，op_en `0x18` | INT32 add、FP32 BS/EW |
| 逐通道缩放/偏置 | BS/BN 外部系数（BRDMA/NRDMA） | dequant row task |
| 标量缩放/偏置 | BS/BN 寄存器操作数 | 同上 |
| 行最大/最小 | DPU `minmax_ctl` | Softmax ReduceMax |
| 行求和 / 均值 | CNA/CORE 全 1 权重卷积 | Softmax、RMSNorm |
| 标量→行广播 | CNA/CORE 单位/稀疏权重卷积 | Softmax、RMSNorm |
| 非线性函数 | LUT（513×2 项），或位运算技巧（exp、rsqrt） | SiLU LUT、exp/rsqrt 近似 |
| 复制、改布局、按 CPU 给的地址 gather | offline DPU 纯 bypass，每行一个 task | `copy_i8_body` |
| 行抽取 | MRDMA `SURF_NOTCH` 跳行 | dequant row task |

**本项目尚未获得通过验证配置的功能**：根据数据值计算地址的动态 gather/scatter、scatter-add、
ERDMA 标量广播、保留完整 FP32 精度的乘数、任意转置、跨 task 的 CACC 续加。
遇到这些要改写算法，或者留给 CPU。

---

## 2. 先判断值不值得上 NPU

先用第一部分 §11 的历史样本估算，再以实际板端测量修正模型。若预估成本高于 CPU 基线，
应先检查驻留、流水线或算子融合是否能改变数据搬运量。

### 2.1 GEMM 类

```text
t_npu ≈ Σ_task (B_tile_bytes / measured_B_bandwidth) + 流水线开销
       + submit (~20–30 µs) + CPU 打包/量化/sync/后处理
```

- M 很小（decode）时 B 流量决定一切：`M=1` 的 GEMM 基本等于"把权重读一遍"，
  应比较相同操作数、压缩格式、核数与频率下的完整调用；多核并行和释放 CPU 容量可能有价值。
- K 很大时先算 Mtile（CBUF 预算），再判断要不要沿 K 拆分。
- N 太小（≤32）时固定开销占主导。

### 2.2 非 GEMM 类

- 逐元素或归约（Softmax、Norm、量化）：每个 task 要读写一遍 DDR，
  链内的额外 task 虽然只有约 0.3–0.9 µs，但**每多一个中间张量就多一次 DDR 往返**。
- 对比基线必须是**优化过的 NEON/OpenMP**，NumPy 可以用于功能对照，但不能代表优化后的 CPU 性能上限。本项目多次出现"比 NumPy 快，比 NEON 慢"。
- NPU 的价值往往在于：张量本来就在 NPU 上（省掉一次 CPU 往返）、CPU 已经满载、或 M 足够大能摊薄开销。

### 2.3 必须算进来的隐藏成本

- CPU 打包/量化：动态量化加 native 打包，在 `K=8192` 时累计几百 µs（可以与 NPU 重叠）。
- cache sync：约 58 µs/MiB。
- DPU task 与 CORE 串行：放在 NPU 上的后处理会推迟下一个 GEMM。
- 共享 DDR：CPU 和 NPU 同时运行时双方都会变慢（CPU 归约 1.2 ms→2.1 ms）。

---

## 3. 开发流程

### 步骤 1：把算法改写成硬件形状

- 能做成 GEMM 的就做成 GEMM，而且**尽量做成大 GEMM**。KDA 把多个 head 放进块对角的 B，
  利用原本就要补零的 K/N 位置，任务数从 64 降到 21。
- 卷积的空间维就是矩阵的 M，通道维是 K/N。M 可以映射成 `H×W` 的二维形状，
  换一种排布就能改变访存模式（panel 布局在部分形状上快 1.2–1.6 倍）。
- 标量广播、行求和都用卷积做；"除以行和"只能写成 `main / external` 的顺序（没有操作数交换位）。
- 利用代数重排把运算塞进已有 task：RMSNorm 把 gamma 乘法并进广播 ConvMul；
  Softmax 把"减最大值"改写成 `BS×(-1)` 加 EW Add，从而和广播合并。

### 步骤 2：确定每条边的 dtype 和布局

- 画出数据流图，每条边标 dtype、native 块大小（K8/K16/N4…）、M/N 的 padding。
- 首选**生产者的输出布局正好是消费者的输入布局**（INT8 requant→native A、FP16 native D→native A、
  QK 输出→Softmax→PV）。这是 NPU 能赢 CPU 的主要来源之一。
- 注意 K 对齐：INT8/FP16/BF16/INT4 GEMM 的 K 按 32 对齐，TF32 GEMM 按 16 对齐；FP16 紧凑张量常以 8 通道为 atom。
  可以用 `CNA_DATA_SIZE1` 高/低 16 位分离"DMA 跨度"和"计算 K"来衔接（见第一部分 §6.2）。

### 步骤 3：选择最接近的已验证配置

优先从能检查到具体实现和适用范围的材料开始：

1. **本库寄存器生成器**：[INT8 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_matmul_i8.cpp)、
   [FP16/BF16/TF32 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_matmul_f16.cpp)、
   [INT4 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_i4_regs.cpp)、
   [MoE 行复制与 DPU 后处理](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_moe_regs.h)
   以及 [INT8 split-K DPU 链](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_i8_dpu_chain.cpp)。
   先确认 dtype、输入/输出布局、CBUF 预算、模块使能和 submit flags，再选择相近配置。
2. **Mesa Rocket**：[寄存器生成实现](https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/rocket/rkt_regcmd.c)
   和 [寄存器定义](https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/rocket/registers.xml)。
   与本库的布局和驱动契约逐项对照，不直接套用不同几何的参数。
3. **TRM**：用于查位域、编码和规定的配置顺序。没有板端记录的组合仍需从最小测试验证。

已有配置只保证原来验证的条件。改变几何、数据类型、前驱 task 或驱动后，重新检查状态依赖和数值行为。

### 步骤 4：做最小 canary，一次只改一个变量

- 从能工作的配置开始，形状取最小（M=4、K=N=32/64）。
  M<4 在部分路径上是特殊情况，M=1 常常需要单独的配置。
- **一次只改一个寄存器或字段**，每次都上板。已有配置中的关键参数通过单变量对照检查：
  EW cvt 位、32 bit 槽的低 16 位、双 surface、submit flags。
- 新配置先在 1 个 task、1 个 core 上跑通，再扩展到链、M/N tile、多核。

### 步骤 5：用能区分对错的数据验证

**恒等数据不能单独证明操作数生效**：乘 1、加 0、单位矩阵都无法区分"运算生效"和"操作数被忽略"。

- **判别性数据**：非平凡的 `A@B` 加非零常数操作数，或者让每个元素都不同（`buf[i]=i` 的 marker）。
- **marker 验证布局**：令 A=0（或 B=单位阵）、操作数 `buf[i]=i`，读回输出就能看到每个元素来自哪里。
  FP16 half-surface 布局、BRDMA 系数的 64 字节分块、INT32 半覆盖问题都是这样定位的。
- **参考计算要模拟硬件的舍入**：TF32 截断低 13 bit、FP16 系数、FP16 分阶段舍入。
  整数路径要求逐位一致。浮点路径先定义累加精度、操作顺序和舍入边界；
  可逐位复现时检查逐位一致，否则使用明确的误差容限，并另行报告相对理想 FP32 的误差。
- **边界覆盖**：M/N/K 尾块、M mod tile、极值（-128、±65504）、全零、正负抵消、重复索引。
- **重复运行**：同一个 prepared chain 连跑几百到上千次，还要 A/B/A 交替输入。
  有的配置第一次正确，重复后才出错（panel 布局、内存复用 bug）。
- **guard**：每个输出缓冲后面填 `0xa5`，每次运行后检查。
- **先跑 NPU，再跑 CPU 参考**，避免 CPU 先写出的正确值掩盖 NPU 没写的问题。

### 步骤 6：扩展到真实形状

- **M tile**：按 CBUF 预算计算（第一部分 §6.3），不要超出已验证的 bank 数。
- **N tile**：整块 N 可作为基线；多核拆分、notch 位宽或其他几何限制可能需要 N 分块。
  各候选应比较包含布局转换和同步的完整调用时间。
- **K 拆分**：`Kt` 取让 Mtile 达到上限的值（INT8 normal、M=128 时为 2048）。
  partial 用 CPU 精确归约，并与下一波 NPU 计算重叠。
- **多核**：按 N 均分成连续的 task 区间，每个 core 一条封闭的链；
  本库使用 mask 1/2/4/3/7。核数不能写死：历史样本中 `N=128` 单核较快，
  较大的 N 才可能摊薄多核开销；阈值还受 M/K、布局和 CPU 配置影响。
- **长链**：所有 body 放在同一个 regcmd buffer，所有数据放在同一个 IOMMU domain。

### 步骤 7：生产化

- **prepare 一次，运行时只 patch 地址**：建 plan 时 DMA 地址填 0，记下
  `CNA_FEATURE_DATA_ADDR / DCOMP_ADDR0 / DPU_DST_BASE / RDMA_SRC_BASE / RDMA_EW_BASE`
  所在的 qword 下标，run 时只改这些字（`find_patch_indices()`）。
  不要读 non-cacheable 的 regcmd 来做读-改-写，直接覆盖整个 qword。
- 尾部链接（PC 尾）在 prepare 时生成；多核重新分区时重建所有尾，避免残留旧的结束尾。
- 正确性测试可预填 marker 检查完整写回；若实现保证覆盖全部有效输出，性能测量避免在每次调用前无谓清零。
- 数据缓冲用 cacheable（0x13）并显式 sync；CPU 的中间结果不要写进 non-cacheable 内存。
- 模板的 relocation 只允许出现在白名单里的地址寄存器上。
  "寄存器值落在旧 BO 地址范围内就当成地址"会把恰好相等的常数也改掉。
- 内存复用：graph 输入默认保活到结束，只有显式 donate 才能复用。
- 需要 autotune 时，先用安全配置（anchor）算出参考结果，每个候选都与它逐位比较；
  任何 submit 失败都立即停止本次 tune。最终决胜用**包含打包/同步/后处理的完整调用时间**。

---

## 4. 代码骨架

### 4.1 C++：基于库内部接口

下面的辅助函数展示单核 GEMM/DPU 链的组织方式。调用者提供完整 body 和每个 task 的
模块使能值，并负责 A/B/C 缓冲的分配、地址绑定、缓存同步与生命周期。
它使用私有 `src/` 接口，应与同一版本的库一起编译。

```cpp
#include "rk_npu_internal.h"
#include <cstring>

using namespace rknpu2_matmul_open::detail;

int submit_bodies(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                  const std::vector<std::vector<uint64_t>>& bodies,
                  const std::vector<uint32_t>& enables) {
    if (!ctx || !domain || bodies.empty() || bodies.size() > 4095 ||
        bodies.size() != enables.size()) return RK_NPU_ERR_PARAM;
    if (domain->ctx != ctx) return RK_NPU_ERR_DOMAIN;

    std::vector<int> base;
    uint64_t qwords = 0;
    constexpr uint64_t max_qwords = UINT32_MAX / 8;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i].empty() || bodies[i].size() > max_qwords - 5 ||
            (enables[i] != 0xd && enables[i] != 0x1d && enables[i] != 0x18))
            return RK_NPU_ERR_PARAM;
        const uint64_t slot = (bodies[i].size() + 5) & ~uint64_t(1);
        if (qwords + slot > max_qwords) return RK_NPU_ERR_PARAM;
        base.push_back(static_cast<int>(qwords));
        qwords += slot;  // body + 四个 PC 尾字，向两个 qword 对齐
    }

    rk_npu_mem cmd{}, tsk{};
    int rc = rk_npu_mem_alloc(domain, qwords * 8, RK_NPU_MEM_NON_CACHEABLE, &cmd);
    if (rc != RK_NPU_OK) return rc;
    rc = rk_npu_mem_alloc(domain, bodies.size() * sizeof(rknpu_task),
                          RK_NPU_MEM_KERNEL_MAPPING, &tsk);
    if (rc != RK_NPU_OK) {
        rk_npu_mem_free(ctx, &cmd);
        return rc;
    }
    std::memset(tsk.vaddr, 0, static_cast<size_t>(tsk.size));
    auto* commands = static_cast<uint64_t*>(cmd.vaddr);
    auto* tasks = static_cast<rknpu_task*>(tsk.vaddr);
    write_chain(commands, tasks, cmd.dma_addr, bodies, base, {0xd, 0, 0xd});
    for (size_t i = 0; i < bodies.size(); ++i) {
        commands[base[i] + bodies[i].size() + 3] =
            E(T_PC, R_OPERATION_ENABLE, enables[i]);
        tasks[i].op_idx = enables[i] == 0xd ? 0 : 1;
        tasks[i].enable_mask = enables[i];
    }
    rc = do_submit(ctx->fd, tsk.obj_addr, static_cast<int>(bodies.size()),
                   domain->id, 2000);
    rk_npu_mem_free(ctx, &tsk);
    rk_npu_mem_free(ctx, &cmd);
    return rc;
}
```

`do_submit()` 的 flags 固定为 `PC|PINGPONG`，失败后会请求 reset。
它不适用于需要 PC-only 的初始化链。此类诊断应使用自己明确控制 flags 和恢复行为的封装，
失败后停止测试并保留日志。多核提交还需要在 `write_chain()` 生成时分别封闭各核的链，
不能把以上单核链直接当成多核描述符提交。

### 4.2 Python：检查编码和布局

Python 适合生成 marker、计算参考布局和检查 64 bit 寄存器命令。
下面的例子只验证编码；它不包含设备打开、DMA 分配、缓存同步或 NPU 提交。

```python
def regcmd(target, offset, value):
    if not (0 <= target < (1 << 16) and 0 <= offset < (1 << 16)
            and 0 <= value < (1 << 32)):
        raise ValueError("register command field outside encoded range")
    return (target << 48) | (value << 16) | offset

# DPU S_POINTER，指令格式见第一部分 §2.3。
word = regcmd(0x1001, 0x4004, 0xE)
assert word == 0x10010000000E4004
```

板端执行仍需遵循第 2、3 节的内存与链契约。已有可构建、可运行的入口见
[测试指南](https://github.com/happyme531/rknpu2_matmul_open/blob/main/tests/README.md)。

### 4.3 一个 offline DPU task 的完整模板

下例列出已测试的 FP32 native `[C/4, rows, 4]` 逐元素加配置。
修改数据类型、几何或前驱模块时，仍须检查完整初始化和共享状态；C 按 4 通道填充。

```text
DPU  0x4004=0xe  0x400c=0x1e5(offline)  0x4010=0xb4000005(FP32 in/proc/out)  0x4014=0
     0x4020=dst  0x4024=rows*16  0x4030=rows-1  0x4034=0  0x4038=0  0x403c=(C-1)<<16|(C-1)
     0x4040=0x53 0x4044=0 0x4048=0 0x404c=0  0x4050=0x2  0x4054=0  0x4058=C-1  0x405c=rows-1
     0x4060=0x53 0x4064=0 0x4068=0 0x406c=0
     0x4070=0x10c203c0(EW ADD, 32bit, cvt bypass) 0x4074=0 0x4078=1 0x407c=0
     0x4080=0 0x4084=1 0x4088=0  0x4090..0x40ac=0  0x40c0=rows*16  0x40c4=0
RDMA 0x5004=0xe  0x500c=rows-1  0x5010=0  0x5014=C-1  0x5018=src_a
     0x501c=0 0x5020=0 0x5028=0 0x502c=0
     0x5034=0x4000000c(per-pixel 32bit)  0x5038=src_b  0x5040=rows*16
     0x5044=0x2f8a1(FP32, offline)  0x5048=0 0x504c=0 0x5064=0 0x5068=0x01010101 0x506c=0
op_en = 0x18
```

改成乘法时，把 `0x4070` 改成 `0x10c003c4`，操作数改成 FP16-in-32bit 槽；
改成逐通道广播时，把 `0x5034` 改成 `0xc`、`0x4070` 的 data_mode 改成 0、`0x5040=16`。

---

## 5. 常用技巧

| 技巧 | 要点 | 对应实现或用途 |
|---|---|---|
| 沿 K 拆分 + CPU 归约 | 让每一波的 Mtile 达到上限，减少 B 重读；partial 与下一波重叠 | `rk_npu_kn_plan.cpp` |
| 输出 native C | 大 N 下是最大的单项性能收益 | CPU 布局转换与 GEMM 配置 |
| requant 串接 | INT8 GEMM 直出 `[N/16,M,16]`，下一个 GEMM 直接读 | DPU requant 与下一算子的布局契约 |
| 块对角 B 批处理小矩阵 | 把多个 head 塞进一个 GEMM 的补零区域 | KDA 多头批处理 |
| 卷积做广播和求和 | 稀疏单位权重广播，全 1 权重求和（BF16 求和可提高精度） | Softmax/RMSNorm |
| 每行一个 task + patch 寄存器 | 行尺度写进 BS_MUL 寄存器，每次运行只改 M 个字 | dequant row task |
| MRDMA notch 抽行 | `SURF_NOTCH=(M-1)*16` 从 native 布局抽一行 | dequant、MoE |
| DPU 当 DMA 用 | 纯 bypass 复制，目标 stride 决定 normal/native | `copy_i8_body` |
| 位运算近似 | FP16 exp、BF16 rsqrt 种子，用整数加乘操作位模式 | Softmax/RMSNorm 近似 |
| 把倍率留在数据流里 | SiLU LUT 的 ×1024 倍率合并进后续量化 scale，删掉恢复 task | MoE 中间阶段 |
| 代数重排合并 task | gamma 并进广播；`x - max` 改成 `BS(-1)` 加 EW Add | Add+RMSNorm、Softmax |
| 预生成模板 | 固定几何与完整寄存器配置 → 标记地址字段 → 重绑定缓冲 → 独立数值验证 | `rk_npu_add_rmsnorm_templates.inc` |

---

## 6. 安全规则（板子是共享资源）

- 使用自己的板端 SSH/执行工具，并用 `timeout` 限时；多个实验共用设备时，应约定同一个锁来串行化 NPU 提交。
- probe 的 submit timeout 设为 1–2 s，缩短失败循环。
- 每次超时之后：先看 dmesg，再跑一个**完整初始化**的已知 canary，然后才继续。
  不要在同一个进程里 reset 后接着重试失败的配置。
- 已知会导致超时的配置要在代码里**在提交前拒绝**，不要留成可以随手触发的 CLI 选项。
- **禁止写 NPU runtime-PM control**（曾导致 kernel panic）。改频率时保存原 governor，在 finally 里恢复并读回确认。
- SSH 失联或系统异常时暂停测试，先恢复设备并采集故障信息，避免自动重试继续提交任务。
- 模型和大型测试数据放在独立目录，记录来源和校验值，避免混入源码仓库。

---

## 7. 调试清单

**submit 超时**
1. 看 dmesg 的 task counter，定位是第几个 task。
2. 检查 flags（混合 LUT 链要用 PC-only）、core mask、`subcore_task` 槽位。
3. 检查链：每个 core 的最后一个 task 是否有结束尾；所有 body 是否在同一个 allocation。
4. flying EW：MRDMA 和 ERDMA 两个地址是否都改了；M=1 是否用了特殊配置。
5. 精度字段是否一致：CNA in/proc、CORE proc、DPU in/proc、RDMA in/proc。
6. 数值是否超出已验证范围：DPU 单 task 通道数、M tile bank 预算、notch 位宽。

**结果错但不超时**
1. 用 marker 数据找出错位规律：只有第一个 surface → `size_e`；块重叠 → `SURFACE_ADD`；
   一半通道 → atomic 不匹配；间隔读取 → EW cvt 位。
2. 换一个前驱 kernel 再跑，或者重启后先跑本 kernel：结果变化提示状态依赖；同时检查缓冲生命周期与同步，再补齐缺少的初始化。
3. 检查 relocation：是否有地址没改（仍指向旧 BO），是否把常数误当成地址改了。
4. 检查 native split 的 DMA 配置、`DATA_SIZE1` 高低位、feature_grains。
5. 检查 cache sync 方向和范围；检查 CPU 是否读了 NPU 还没写完的缓冲。

**性能不如预期**
1. 先比较 driver 的 `hw_elapse_time` 和 wall time：差距大说明问题在 CPU、sync 或 submit 次数。
2. 只有每个测试前缀都完整初始化且输出可验证时，才用前缀差分估算增量成本；
   重排 task、删除初始化或改变频率时，这种差值不能直接解释为单个 task 的成本。
3. 用 B 流量模型检查 GEMM：task 数 × B tile 字节 / 带宽。
4. 测量时固定频率，以设备支持的 OPP 和实测读回值为准。同一进程内交替运行各方案，
   奇偶轮反转顺序。重叠的阶段时间不能相加。

---

## 8. 案例：几个 kernel 是怎么做出来的

### 8.1 INT8 GEMM

INT8 GEMM 路径的来源署名保留在
[源码](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_matmul_i8.cpp)
和 [第三方说明](https://github.com/happyme531/rknpu2_matmul_open/blob/main/THIRD_PARTY_NOTICES.md) 中。
配置检查包括 INT8→INT32/FP32 输出精度、大 K 的 M tile/grains、native A split-M，
以及 native C 输出、split-K 和三核 N 分片。prepare 阶段生成 regcmd，运行时重绑定地址；
收益需要用包含 CPU 打包、同步和归约的完整调用时间确认。

### 8.2 融合 GEMM + 张量加法

在已测试的 flying GEMM 几何中，仅启用 ERDMA 会超时，双 surface 的 MRDMA/ERDMA 配置通过。
恒等操作数不能证明 FP16 加法真正生效：CVT 设置错误时，第二操作数可能被忽略。
INT8 flying 加法的另一个问题是仅覆盖部分通道，该路径改为 offline INT32 DPU 加法。
应区分超时、操作数未生效和通道覆盖不完整，并用非恒等输入分别定位。

### 8.3 NPU dequant（两轴 scale）
逐行 task 最初超时。拆成单变量对照后（FP32 copy → BS 标量乘 → BS+EW → native copy → notch 抽行），
发现是 EW 乘数的粒度问题：32 bit 槽的低 16 位要放 FP16。修正后每行一个 task，BS 乘行尺度、
EW 乘通道尺度，直接输出紧凑 FP32。

### 8.4 固定形状 Add+RMSNorm

本库的固定形状实现使用预先准备的寄存器模板。运行时生成 gamma、归约权重和广播权重，
按已知地址字段重绑定缓冲。算子顺序、近似 rsqrt 的迭代次数以及 gamma 的融合位置
都要按各自输入范围做数值验证，不能把一个形状下等价的变换推广到任意形状。

实现见 [Add+RMSNorm](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_add_rmsnorm_f16.cpp)
及 [固定形状模板](https://github.com/happyme531/rknpu2_matmul_open/blob/main/src/rk_npu_add_rmsnorm_templates.inc)。
当前公开接口仅覆盖 `M=1/128, D=4096, eps=1e-5`，不是通用 RMSNorm 内核。

### 8.5 KDA decode：已记录的性能限制
复用 TF32 GEMM 生成器和 FP32 DPU 配置，状态常驻 NPU，布局 `[Dk/4,Dv,4]`
同时满足 DPU 输出和下一次 GEMM 的 native A。通过块对角合并 head，任务数从 64 降到 21，
但在该历史样本中仍比四核 NEON 慢 2.9 倍：瓶颈是状态的反复读写，而不是任务数。
`Dk=Dv=128`、FP32 状态时，每个 head 占 64 KiB，16 个 head 合计 1 MiB；
每 token 的总流量还取决于这些状态在多少个 task 中被重复读取。
尝试 flying 融合超时后，记录未通过的配置，未继续扩大测试范围。
**教训：先用流量模型判断上限；任务数减少不等于更快。**

---

## 9. 文档与证据的写法

新 kernel 的记录应包括：

- 环境（驱动、runtime、编译器版本，频率）和形状范围；
- 最终寄存器配置中与已知配置**不同的字段**，以及每个字段为什么这样设；
- 失败的配置和现象（超时、错误值、半覆盖），并注明"未证明硬件不支持"；
- 正确性：参考计算定义、逐位还是容差、边界用例、重复次数；
- 性能：完整调用与硬件时间分开报告，写明对照基线（NEON 还是 NumPy）；
- 结果目录里保存源码 hash、原始样本和 governor 恢复记录。

可以按“环境与范围 → 数据布局 → 寄存器差异 → 正确性 → 完整调用性能 → 未解决问题”组织报告。
公开时使用可访问的源码或数据链接；历史样本、主机参考、板端结果和推断分别标明，
不能用源码实现代替板端验证，也不能用设备计时代替端到端延迟。

---

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)
