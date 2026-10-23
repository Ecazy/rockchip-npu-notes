# HardSigmoid 组合算子的 K 维分片与加速

2026-10-09，本轮把 `Y = HardSigmoid(X[M,K] @ W[N,K].T + b[N])` 的逻辑 K 上限从 256 扩到 **4096**。
保持 RK3588、FP16、设备 Cube 交接以及每次 `CO_CHAIN` 调用 **一次 Submit**。
M/N 仍为 1..256。该实现不使用 CPU 累计、CPU 中间激活或每个 K 片的 CPU 输入重排。

## 实现顺序

1. **全输入 Cube 与 K 片。** 新接口 `co_tensor_input` 描述 `[Kphysical/8,Mphysical,8]`，单片宽度最多 256。
   第 k0 通道起的片段直接绑定设备地址 `input_dma + k0*Mphysical*2`。
   `co_tensor_input_slice` 保留行数/原子，最后一片补齐至 32 通道；输入只在 host 入口整体打包一次。
   原来的 `co_tensor_matrix` 和单 tile 权重接口仍保持 256 上限，不把大维度直接塞进寄存器。
2. **设备累计。** 第一个 CNA/CORE MatMul 写 ACC0；后续通过 `co_emit_mm_acc`，
   将当前卷积结果与上一片的 FP16 Cube 在 DPU EW 内相加，交替写 ACC0/ACC1。
   读写使用不同 BO，避免 ERDMA/WDMA 原位冲突。
3. **一次 epilogue。** 最后累计结果才加一次 bias，再执行 HardSigmoid。
   普通输出 task 数为 `ceil(K/256)+2`；大输出分两段 LUT，为 `ceil(K/256)+3`。
   例如 K=4096、M=N=64 共 18 个 task；M=N=256 共 19 个 task，均一个 Submit。
4. **融合加速。** 对照图将每片 MatMul 独立写临时 Cube，再用 identity-conv EW add 累计。
   两条图均使用已驻留的权重/bias、设备 Cube 和一个 Submit；融合图直接在当前 MatMul 的 DPU 中累计，减少 task 和中间流量。

每个 MatMul 仍使用完整寄存器配置，没有 CBUF reuse 或 delta-state 优化。
权重分片在创建上下文时打包一次，整个上下文拥有并持有各片 BO。
累计每片输出收窄到 FP16，不能视为 FP32 精确求和；融合图与未融合图多一次中间收窄的差异分别建模。

## 真机验证

最终记录：[ktiling-20261009-194732](../custom-ops/results/2026-10-09/ktiling-20261009-194732)，
[不可变清单](../custom-ops/results/2026-10-09/ktiling-20261009-194732/campaign.json) 保存全部证据 SHA256。
环境为 Orange Pi 5 Plus / RK3588，`6.1.43-rockchip-rk3588-rknpu098`，厂商 `rknpu 0.9.8`，读取 NPU 频率 1 GHz，GCC 11.4.0。
三个核分别固定 mask 1/2/4，各跑两个独立进程，串行执行；未修改时钟、驱动或 debug counter。
当前源码已与 [源码清单](../custom-ops/results/2026-10-09/ktiling-20261009-194732/source-manifest.json) 逐文件核对。

每个进程的在线检查统计为 264，六个进程均通过。
归档共 282 条被评分的功能/性能记录：66 条完整算子、96 条变化输入、96 条未激活累计、24 条性能图记录。
清单中的 submits/tasks 合计只针对这些归档记录，不代表包含 warmup、所有时序重复和前缀控制的总 ioctl 数。

| 验证项 | 覆盖与结果 |
|---|---|
| K 边界与尾片 | 255、256、257、511、512、513、769、1024、2049、4096 |
| 逻辑/物理维度 | M/N 含 1/1、16/32、29/45、64/64；M=N=256、K=4096 验证双 LUT 输出 |
| 两个入口 | host 与设备 Cube；外部 View 偏移 16/64 B，输入保持不变，View 外字节和 guard 完整 |
| 单次提交 | 所有生产 `CO_CHAIN` 调用核对实际 provider ioctl/task 计数；无 CPU 回退 |
| 未激活累计 | 最后累计值直接对照 FP64 总和和逐片 FP16 参考，防止激活饱和隐藏遗漏 |
| 通道脉冲 | 255、256、257、511、512 及不同 K 的最后通道，精确检查尾片贡献 |
| 跨片抵消 | K=513，第 17 与第 300 通道提供正/负贡献，跨越两个 K 片 |
| 累计前缀 | K=257、769、1024 的每个前缀，作为独立控制运行；最终生产图仍只提交一次 |
| 变化输入 | K=769 与 4096，一个权重上下文连续 8 次调用，交替 host/Cube 入口，无 sleep |
| 避免饱和掩盖 | 完整算子及性能图输出处于未饱和区；归档门要求相应比例至少 50% |

两种完整算子入口对 FP64 数学参考的最大绝对误差均为 `0.000320055069`，
变化输入最大误差 `0.000315931488`。
未激活累计对 FP64 最大误差 `0.0000610351562`，对分片 FP16 参考误差为 0。
所有结果使用事先指定的 `0.004` 门限；脉冲及抵消探针另要求精确值一致。

独立 Python 标准库 oracle 重新读取保存的输入、权重、bias 和硬件输出，复核相同结果。
主机与板端 7 个 CTest 门通过，主机 ASan/UBSan 覆盖的主机路径通过。
[kernel-after.log](../custom-ops/results/2026-10-09/ktiling-20261009-194732/kernel-after.log) 为 `-- No entries --`，测试区间未见 timeout、IOMMU fault 或 reset。

## 实际命令流

首次 K=513 烟测另外使用原始 Hook 捕获一个 Submit、五个有效 task，strict 和 strict-effective 均通过。
物理 M=32 时，三个输入片的 CNA feature 地址偏移为 0、16384、32768 B；
第二/第三个 MatMul 的 RDMA source 分别等于上一段的 destination，destination 在两个累计 BO 间交替。
这同时核对了完整输入 Cube 的通道切片与设备累计地址，而非仅依据 API 计数推断。
原始烟测在本机 `research/custom-ops/artifacts/ktiling-smoke/trace.jsonl` 保留；最终六进程记录单独归档。

## 加速结果与计时范围

每个形状、每个进程预热每条图四次，随后执行 24 对交替顺序的样本。
样本保留原始顺序；p10/median/p90 在副本上排序计算。
两条图同用设备端数据和一次 Submit，使用相同 guard 策略。
计时范围是 `co_execute` 的整条计算图，包括命令 BO 准备、submit 和输出同步，
不含 host 输入打包、权重打包、最终下载及 oracle；不是生产应用端到端加速声明。

| M/K/N | 融合/未融合 task | 融合 wall 中位数 | 未融合 wall 中位数 | 六次 wall 加速比中位数 | 设备区间加速比中位数 |
|---|---:|---:|---:|---:|---:|
| 64/4096/64 | 18 / 33 | 727.26 µs | 1043.57 µs | **1.43×** | 1.40× |
| 256/1024/256 | 7 / 10 | 2634.30 µs | 3949.69 µs | **1.43×** | 1.52× |

表中 wall 时间是六个进程各自中位数的中位数；加速比是各进程成对中位数比值的中位数。
前一形状各进程 wall 加速比范围 1.416–1.444×，后一形状为 1.300–1.603×。
全部原始样本、设备 `hw_elapse_time` 区间与分位数位于各进程的 `probe.jsonl`。
两条图 FP16 收窄顺序不同，分别使用其数值模型检查，不要求错误的逐位一致。

## 使用与复跑

应用继续使用 `co_linear_hsig_create/run/run_cube/destroy`。
大 K 的 Cube 请由新工厂 `co_tensor_input(&tensor,M,K)` 描述；结果依旧使用 `co_tensor_matrix(M,N,CO_F16)`。
CLI 的 K 上限也已改为 4096：

```bash
# 在板端的 custom-ops 构建目录对应项目内
RKNPU_CORE_MASK=1 ./build-ktiling/linear_hardsigmoid \
  29 513 45 x.bin weights.bin bias.bin y.bin chain 3

# 从当前仓库主机复跑完整验证与性能对照
bash research/custom-ops/run-ktiling-board.sh

# 仅复核已经保存的证据，不重新执行 NPU
python3 research/custom-ops/ktiling_campaign.py \
  research/custom-ops/results/2026-10-09/ktiling-20261009-194732
```

入口：[生产实现](../custom-ops/src/linear_hardsigmoid.c)、[原语/切片](../custom-ops/src/custom_ops.c)、
[主机边界测试](../custom-ops/tests/c/ktiling_host_test.c)、[真机测试与对照图](../custom-ops/tools/c/ktiling_probe.c)、
[独立 oracle](../custom-ops/tools/python/ktiling_gate.py)、[归档核验](../custom-ops/tools/python/ktiling_campaign.py)。
K>4096 或 M/N>256 明确拒绝，不静默截断。未实现 M/N 多 tile、任意 K 或其他精度。

初版实现和测试脚手架由按仓库 skill 配置的 CSU/DeepSeek 提供；Codex 独立检查代码和测试。
实现报告的 `complete` 状态不符合包装器合同，未当作成功回执；用户要求停止委派后，
Codex 结束会话并修正输入快照、跨片抵消、raw task 计数、不同精度 oracle、原始计时顺序及变化输入测试，完成真机验收。
