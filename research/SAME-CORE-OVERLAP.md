# 同核 MatMul 与 DPU-only LUT 能否重叠

2026-10-14，Orange Pi 5 Plus / RK3588，内核 `6.1.43-rockchip-rk3588-rknpu098`，厂商 `rknpu 0.9.8`，
读取 NPU 频率 1 GHz。全部提交固定在 **Core 0**（`mask=1`），只用一个 blocking Submit 路径。
正式证据：[overlap-20261014-145427](custom-ops/results/2026-10-14/overlap-20261014-145427/)，
汇总见 [analysis.json](custom-ops/results/2026-10-14/overlap-20261014-145427/analysis.json)。

后续 [同核 GEMM+GEMV 链式/并发对照](GEMM-GEMV-CONCURRENCY.md) 保留本组 MatMul＋LUT，另加 GEMV、
B8/B16 放大组、双进程同核/跨核正对照，并记录了同进程并发不可用的两个驱动约束。

## 结论

**同一 Core 0 上，已测 FP16 MatMul 与独立 DPU-only LUT 编排没有显示可分辨的额外重叠收益。** 证据是：

- 链式 Submit 比两次独立 Submit 少约 **52–53 µs**，但这个节省与两个 task 用哪些模块无关：
  异质对 `[MatMul, LUT]` 与同质对 `[LUT, LUT]` 的节省在两轮里分别只差 **146 ns 和 −146 ns**。
  它是**每个 Submit 一次的固定服务包络**被摊销，不是模块并发。
- 链的服务 ≈ 两个 task 独立服务之和减去一次包络；`S1 − (A+B)` 的两轮中位数是 **−583 ns 和 −146 ns**，
  即两次独立 Submit 就是加法，没有隐藏并发。
- 每个 Submit 只产生**一个 Core 0 完成中断**：链式 1 次、分开 2 次，两轮共 1024 个 Submit 全部一致。
- Setup 3 的位图消融不能解锁并发：清掉 MatMul 的 DPU 参与位后该 task **永不完成**（2 s 超时、驱动 soft reset）；
  在 `[LUT_dpu, MatMul_core]` 中 DPU-only 的 LUT 正常写回，随后链停在 MatMul_core。
  该 MatMul 的输出/完成通路依赖 DPU，已尝试的使能位消融未能解锁模块并发；尚未确认其他 PC 控制模式的可行性。

没有声称“硬件在物理上不可能”，只声称这条厂商提交通路上不可达；范围见文末。

## 对照 Ascend 的 FlashAttention tile 流水：复核补充

本次针对“Softmax(tile0) 与 MatMul(tile1) 能否重叠”的复核使用上述已有真机原始记录，
没有重新提交硬件任务。独立核验 `overlap_verify.py` 返回 PASS，24 个核验器测试通过。
源码清单中的 `tools/c/overlap_probe.c`、`src/co_row_ops.c`、`src/custom_ops.c` 与当前文件哈希一致；
当前核验器文件已不同于采集时版本，因此这里是当前核验器对历史记录的再次核验。

Ascend 场景按用户描述：AIC 执行下一块矩阵乘时，AIV 执行上一块 Softmax。
RK3588 当前路径不能把 CNA/CORE 与 DPU 直接视为同样可独立编程的两条执行队列：

- **Submit 调度粒度。** 厂商驱动 `rknpu_job.c` 的 `rknpu_job_next` 在
  `subcore_data->job` 非空时不启动下一 job；同核增加线程或异步提交只能增加待执行 job，
  不能由此获得两条独立的模块执行队列。这是源码依据，未新增双线程实测。
- **当前完整 Softmax 的资源需求。** `co_row_ops.c` 用 CPU 扫描行最大值、准备移位输入，
  用 DPU LUT 计算 exp；校正乘法和最终归一化调用 `co_emit_ew`，其实现通过 identity
  MatMul 调用 `finish_mm`；行求和直接调用 `co_emit_mm`，CPU 再合并分片统计。
  因而完整 Softmax 同时需要 Host、矩阵计算路径和 DPU，并非仅占 DPU 的向量程序。
- **已测链式编排。** 独立 LUT 是比完整 Softmax 更容易隔离的代理。即使输入/输出 BO
  互不依赖，`C_AB/C_BA` 在两轮正式记录中也没有显示超过同类链对照的明显收益。
  此结果支持“当前已测编排不能复现该流水”，不能单独证明所有寄存器编排在硬件上均不可行。

可用模型是 `T_single(A)=H+a`、`T_single(B)=H+b`。
若链串行，则 `T_chain(A,B)≈H+a+b`，节省约为一次固定包络 H；
若发生明显重叠，还应节省一部分 a/b。
这里异质链节省与 `B+B` 链节省之差只有 +146 / −146.5 ns，
并非把 B 在链中的约 50 µs 边际成本隐藏了。
H 包含哪些驱动开销或流水填充成分仍未分解。

**证据边界：** 链只有一个末尾完成 IRQ 本身不能证明内部串行；
清掉 MatMul 的 DPU 位后超时，也不能单独证明 PC 必须等待每个 task 的所有模块完全排空，
因为输出与完成通路同时被破坏。应把这些结果与服务时间对照、驱动单 job 调度一起解释。
未测完整 online Softmax 与多 tile FlashAttention 的交错链，也未获得模块级执行时间线；
原文中的“完整 Task 屏障”属于当前证据支持的工作解释，不能作为已经直接观测到的硬件事实。

后续可分别验证不同 NPU 核之间的分工，或 CPU Softmax 与下一次 NPU MatMul 的双缓冲流水；
这两种并发都需要重新测数值、同步、传输与端到端收益。
对于沿 KV 维切块的 online Softmax，必须维护运行最大值、归一化和及输出重缩放，
不能将每个 KV tile 单独归一化后简单拼接。

## 两个程序和提交形态

两个 program 由既有发射器生成，模块参与位来自 `PC_OPERATION_ENABLE` 的块位图：

| Task | 发射器 | 几何 | regcmd | `PC_OPERATION_ENABLE` | 运行块 |
|---|---|---|---:|---|---|
| A | `gen_matmul_fp16` | FP16 `M=N=K=256` | 126 word | `0x1D` | PC、CNA、DPU、DPU_RDMA |
| B | `gen_lut_activation_fp16` | HardSigmoid，240×256 = 61,440 元素 | 1,070 word | `0x18` | DPU、DPU_RDMA（**无 conv**） |

B 是真正的 DPU-only：MRDMA 以 flying 模式直接读入 fp16 cube，BN 缩放后进入 LUT，输出转换后写回，
不含 CNA/CORE。A 的主体是 CNA/CORE 的 MAC，DPU 只做输出转换。A、B 数据互不依赖，各有独立输入/输出 BO。

| Arm | 含义 |
|---|---|
| `A` / `B` | 单 program、单次 blocking Submit（单项服务基线） |
| `S1` | 先 A 后 B，两次独立 Submit（Setup 1） |
| `C_AB` / `C_BA` | A、B 打进同一个链式 Submit，两种顺序（Setup 2） |
| `C_AA` / `C_BB` | 两个独立同种 program 的对照链 |
| `S3_*` | 位图消融（Setup 3，`OL_EXPLORE=1`）：MatMul 的 `0x1D→0x15`（去 DPU 位） |

每轮 arm 顺序按 rep 轮换，抵消漂移；每个 arm 提交前把四个输出 BO 填 NaN 哨兵，提交后逐个实例独立评分
（MatMul tol 0.02，LUT tol 0.002），未参与的 BO 必须仍是 NaN。任何未写、写错、越界或非有限值都会拒绝该轮。

## 对照结果

两轮各 64 reps、每 rep 8 个 Submit，全部数值与 guard 通过（768 checks，0 失败）。
中位服务包络（`hw_elapse_time`，纳秒）：

| run | A | B | `C_AB` | `C_BA` | `C_AA` | `C_BB` | `S1` 配对和 |
|---|---:|---:|---:|---:|---:|---:|---:|
| core0-a | 334,829 | 102,665 | 384,849 | 384,412 | 625,033 | 152,831 | 436,911 |
| core0-b | 422,766 | 101,499 | 472,786 | 472,056 | 742,864 | 151,373 | 523,826 |

两轮之间存在整轮的环境差：core0-b 的 MatMul 服务比 core0-a 高约 26%（422.8 vs 334.8 µs，全程稳定偏高），
LUT 基本不变（101.5 vs 102.7 µs）。板子当时有其他登录会话，未做独占声明。
因此下面只用**同一轮内**的配对量结论，不跨轮比较绝对值。

一次 Submit 的固定包络用同质 `[LUT, LUT]` 标定（LUT 短、信噪比最好）：

| run | A+B 与链的差 `saving_AB` | `saving_BB` | 异质超出同质 |
|---|---:|---:|---:|
| core0-a | 52,645 | 52,499 | **+146 ns** |
| core0-b | 51,479 | 51,625 | **−146 ns** |

异质对并不比同质对显示更多节省，所以本轮没有观察到“MatMul 的 CNA/CORE 与 LUT 的 DPU 并发”带来的额外收益。
`C_AB − A` 的两轮中位数是 49,874 / 50,166 ns，等于 B 的独立服务减去一次包络（102,665−52,501=50,164；
101,499−51,770=49,729）。也就是说，链里 LUT 的边际成本就是它自己那一段，MatMul 那一段并没有被压掉。

## 逐提交分解与完成事件

定义 `saving = sum − chain`，per-rep 配对（n=64）：

| run | `AB` saved 中位数 [p10,p90] | `BA` | `BB` | `S1 − (A+B)` |
|---|---:|---:|---:|---:|
| core0-a | 52,646 [50,749, 54,249] | 52,792 | 52,501 | −583 |
| core0-b | 50,750 [10,208, 77,290] | 51,770 | 51,770 | −146 |

`C_AA` 的节省在两轮里分别约 44.6 µs 和 102.8 µs，不稳定（core0-b 的 A 单项偏高），
所以它不作为包络标定，也不支持任何“MatMul 之间会重叠”的说法。结论只用稳定的 `AB/BA/BB` 三者。

完成事件：每个 Submit 对应 `/proc/interrupts` 中三条 `fdab0000.npu` GIC 线的增量。两轮共 1024 个 Submit 中，
每个 Submit 的增量都恰好是 Core 0 的 `[1,0,0]`，包括 2-task 链。分开提交得到 2 个完成事件，
链式得到 1 个。`int_status` 末 task 完成位在自动乒乓下于 `0x100`/`0x200` 两组间切换，规律同
[Two-Task Chaining](TWO-TASK-CHAINING.md)；一轮里哪种臂落到哪一组取决于 arm 轮换的奇偶，
链式提交的完成位来自其最后一个 task。

## Setup 3：位图消融与启动/完成通路

尝试从寄存器层面让两个模块独立启动。本轮选择改写的相关寄存器是每个 task 尾部的
`PC_OPERATION_ENABLE`（`0x0008`，块位图），发射器允许用 `ROCKET_LUT_ENABLE` 覆盖 DPU-only 的值；
MatMul 的 `0x1D` 清掉 DPU 位得到 `0x15`（PC、CNA、DPU_RDMA）。结果：

| Arm | 提交 | 结果 | 服务 | 完成 IRQ | 输出 |
|---|---|---:|---:|---:|---|
| `S3_A_CORE` | 只有 MatMul，`0x15` | `-110` 超时 | 0 | `[0,0,0]` | MatMul 未写 |
| `S3_CORE_B` | `[MatMul_core, LUT_dpu]` | `-110` 超时 | 0 | `[0,0,0]` | 两者都未写 |
| `S3_B_CORE` | `[LUT_dpu, MatMul_core]` | `-110` 超时 | 0 | `[0,0,0]` | **LUT 正确写回**，MatMul 未写 |

判读：

- 去掉 DPU 位后，task 不再产生 DPU 完成事件，驱动的 `require mask: 0x300` 永不满足，
  2 s 后 `job timeout` 并 `soft reset`（每个 arm 各一次）。这说明 task 的完成/退役绑定其 DPU 块，
  本次使能位消融不能把它变为可正确输出并完成的 CORE-only task。
- `[LUT_dpu, MatMul_core]` 里 LUT 先按序完成并正确写回，链才推进到 MatMul_core 并停住。
  该顺序与 task 按序推进相容，但不能单凭输出结果证明前一个 task 的全部模块必须完全排空，
  也不能据此判定所有 PC 模式都存在完整 Task 屏障。
- 驱动侧同样没有并发入口：`PC_TASK_CONTROL`（`0x30`，mode 位 `0x6` 与 task 数由驱动写）和每核
  每核的单活动 `subcore_data->job` 调度状态均不由 userspace 控制；同一核的第二个 Submit 只能排队。

因此本轮使能位消融没有找到让两个模块独立启动并正确完成的做法；把此路径记录为已尝试且未成功，
而不是“硬件绝对不支持”。

## 可获得的模块性能计数器

RK3588 这条通路上**没有**可读的模块级 byte/event 计数器：

- 厂商 per-core amount 计数器（`0x2234/0x2434`）在 rk3588 配置里是 `NULL`，且其所在 `0x2000` 页未解码、
  读取会 hard-lock SoC（见 [硬件 DMA 字节计数器](../perf/hw-byte-counters.md)）。
- DPU/SDP 只有 stall/saturation 一类 flag，不计数，也未暴露到 userspace。

本轮实际可用并已记录的“模块侧”量只有：

| 量 | 来源 | 用途 |
|---|---|---|
| 驱动服务包络 | `Submit.hw_elapse_time` | 每次 Submit 的总设备服务时间 |
| 完成位 | 末 task `int_status` 的 `0x300` | 确认链确实执行完 |
| 完成中断 | `/proc/interrupts` 三条 `fdab0000.npu` | 每次 Submit 的完成事件数及核归属 |
| 块位图消融 | task 尾部 `PC_OPERATION_ENABLE` | 把“哪个块负责完成”做成差分实验 |

包络本身是 `register 配置 + 执行 + 到取时之间的中断延迟`，不是纯硬件计算时间；
约 52 µs 的固定项没有再拆成 driver/Host 与流水线填充，归因保留 unresolved。

## 限制与范围

- 两个 task 各是一个 primitive（单 task 的 256³ MatMul、单 task 的 LUT）；未测多 task 变体、
  K 分片链或 tensor 更长的编排。
- 只测 Core 0、厂商 `rknpu 0.9.8` blocking Submit 路径；不推广到 `rocket` 主线路、RK3576 或别的 datatype。
- 服务时间含驱动配置与 IRQ 延迟，不是纯硬件边界；两轮环境差说明绝对值依赖板子状态。
- 本轮没有测“同核两个 Submit 由两个线程并发下发”的驱动排队细节；后续 [GEMM/GEMV 并发](GEMM-GEMV-CONCURRENCY.md) 已补双进程同核/跨核对照，未验收同进程双线程配置。

## 复跑

```bash
# 板端：Core 0 两轮对照 + 一次位图消融探索（会触发受控超时与 soft reset）
bash research/custom-ops/run-overlap-board.sh

# 离线复核证据（无需设备）
python3 research/custom-ops/overlap_verify.py \
  research/custom-ops/results/2026-10-14/overlap-20261014-145427

# 本机负例（无需设备）
python3 research/custom-ops/overlap_verify_test.py
```

入口：`tools/c/overlap_probe.c`（对照、位图消融、host 布局测试）、
`tools/python/overlap_verify.py`（独立证据核验与配对统计）、
`tests/python/overlap_verify_test.py`（24 个正/负例）、
`scripts/run-overlap-board.sh` / `run-overlap-native.sh`。
板端 `ctest` 18 项、主机 `overlap_host` 与 24 个负例均通过。
`kernel-after.log` 只有三次受控 `job timeout`/`soft reset`（探索臂），无 IOMMU fault、无其他内核事件。
