# RK3588 自写组合算子实验框架

这里把寄存器发射、数据布局和 BO 交接组成一个可复跑的最小算子框架。
运行时不使用 RKNN 模型、编译器或 Runtime；底层寄存器生成器和厂商驱动提交适配层复用本地获取的上游源码。
新增代码负责张量契约、打包、边界检查、LUT 内容、原语组合、执行顺序及数值验收。

当前目标为 Orange Pi 5 Plus / RK3588 / 厂商 `rknpu 0.9.8` / FP16。
最新 [K 维分片与加速](../kernels/FP16-K-TILING.md) 将组合算子的 K 上限扩至 4096，保持单次 Submit 和设备 Cube 交接。
新增 [静态图编译与 SwiGLU FFN](../graphs/STATIC-GRAPH-COMPILER.md) 提供 `co_linear_*`、`co_graph_plan_init/compile/run`，自动拓扑/形状分析、统一 tile 布局与 BO 生命周期规划；中间张量保持设备 Cube。
后续 [跨 tile 混合 RMSNorm/稳定 Softmax](../kernels/CROSS-TILE-ROW-OPS.md) 已加入图接口；Softmax明确读取logits到CPU，归约/rsqrt和系数展开为混合路径，不能归入仅图边界传输。
新增 [RoPE、KV cache 与 self-attention](../kernels/ATTENTION-KERNELS.md) 支持固定容量的MHA/GQA、prefill与增量decode、静态图RoPE/attention及受控错误回滚；QK/PV在NPU，KV/动态权重与Softmax仍为混合路径。
新增 [完整decoder block与单行decode](../graphs/TRANSFORMER-DECODER-BLOCK.md) 提供 `co_decoder_*`，共享RMSNorm的Q/K/V多输出图、post-FFN KV回滚，以及prefill/M=1两套执行几何。
完整算子入口为 `co_linear_hsig_create/run/run_cube/destroy`，文件程序为 `linear_hardsigmoid`。
最新结果见 [完整算子交付验收](results/2026-10-09/OPERATOR-DELIVERY.md)；此前原语验收见 [第一阶段记录](results/2026-10-09/README.md)。
这些是实际运行结果，与旧厂商实验分开。

后续 [Two-Task Chaining 四组对照](../scheduling/TWO-TASK-CHAINING.md) 仅使用 HardSigmoid 验证 A/B/C/D，
覆盖真实描述符、链地址、寄存器组完成位、BO 生命周期和连续提交压力测试。

[同核 MatMul/DPU 重叠实验](../scheduling/SAME-CORE-OVERLAP.md) 统一收录完整 task 链、regcmd 模块拆分、
工作量匹配及三组基线修正。32 tile 上，合并 Submit 节省 2.07%，重划分相对完整链额外节省 7.14%；
不是完整 attention 的端到端收益。旧记录的 `serial` 仅为同布局机制对照。
复跑入口与唯一主配置见该文；本目录保留 `scripts/run-matched-overlap-board.sh` 与
`tools/python/matched_overlap_configs/three-arm-baseline.json`。

[同核 GEMM+GEMV 链式/并发对照](../scheduling/GEMM-GEMV-CONCURRENCY.md) 在同一探针上加 GEMV（逻辑 M=1、物理 M=12）、
B8/B16 放大组、以及双进程 `P_same`/`P_cross` 正对照；已测链只省一次 Submit 包络，同核 job 串行，跨核服务包络重叠。
并发组用独立 fd 的进程加管道 barrier，因为驱动对 BO 映射设 `VM_DONTCOPY`，同进程并发也不可靠。

[多核分配与调度 Profiling](../scheduling/MULTICORE-SCHEDULING-PROFILING.md) 提供 `co_schedule_execute` / `co_schedule_pick`、五种真机场景和队列/服务时间区间核验。调度单位为一次 Submit，依赖 task 链整体留在一个核。

[单设备成本模型](../scheduling/DEVICE-COST-MODEL.md) 和 [Job并发与资源干扰](../scheduling/JOB-CONCURRENCY-INTERFERENCE.md) 已完成v2采集工具、945000次正式Submit和120次tracing诊断。离线预测API为 `tools/python/device_cost_analyze.py` 的 `DeviceCostModel`；入口为 `scripts/run-device-cost-board.sh`。模型区分validated、experimental及insufficient_geometry，不能将测量门禁PASS当作所有预测族均达标。

```bash
bash research/custom-ops/scripts/run-device-cost-board.sh smoke 1 12 2
bash research/custom-ops/scripts/run-device-cost-board.sh cost 5 200 20
bash research/custom-ops/scripts/run-device-cost-board.sh resident 5 600 20
```

以上命令从仓库根运行。每次上传实际源码并在新目录编译；v2 JSON配置在 `tools/python/device_cost_configs/`。DeepSeek的未完成v1草案已保存在日期结果目录中的 `handoff-v1-*`，当前入口不执行旧模板。

规范目录为 `include/`、`src/`、`tools/`、`tests/`、`scripts/`；根目录同名文件是兼容符号链接。详见 [项目结构](../../docs/PROJECT-STRUCTURE.md)。

## 已完成的顺序

1. **公共接口**：逻辑与物理 Tensor 描述、Buffer 所有权、带偏移 View、CPU 同步、完整 regcmd 发射及执行接口。
2. **首个组合算子**：`Y = HardSigmoid(X @ W.T + b)`，分别验收 MatMul、bias 和激活。
3. **BO 交接**：比较显式 CPU round trip 与中间 cube 直接交接；计算阶段不读取、重排或复制中间张量。
4. **task 链**：不同长度 regcmd 的链打包、下一段长度检查、终端封口，以及同构/异构几何的独立任务和依赖任务验收。
5. **原语扩展**：EW add/mul、HardSigmoid/sigmoid、SiLU/erf-GELU、有界 exp/rsqrt 探索、最后一维 sum/mean、PPU max/average/global pooling、固定 transpose。
6. **组合归一化与复现**：RMSNorm/LayerNorm 混合路径，三个核各两个独立进程，完整日志和 SHA256 清单。

## 文件入口

| 文件 | 用途 |
|---|---|
| [co_decoder.h](include/co_decoder.h) / [co_decoder.c](src/co_decoder.c) | 完整pre-RMSNorm decoder block、两套几何和block级cache提交/回滚 |
| [co_rope.h](include/co_rope.h) / [co_rope.c](src/co_rope.c) | 两种partial RoPE、跨tile配对，CPU重排/系数与NPU乘加 |
| [co_attention.h](include/co_attention.h) / [co_attention.c](src/co_attention.c) | 固定容量self-attention与事务式主机KV、MHA/GQA和prefill/decode |
| [co_row_ops.h](include/co_row_ops.h) / [co_row_ops.c](src/co_row_ops.c) | 跨tile混合RMSNorm/稳定Softmax，驻留scratch与容量/混合传输规划 |
| [row_cli.c](tools/c/row_cli.c) / [row_gate.py](tools/python/row_gate.py) | 图归约文件接口、数学参考、mask/负尾与饱和拒绝检查 |
| [co_graph.h](include/co_graph.h) / [co_graph.c](src/co_graph.c) | 静态 FP16 图 IR、拓扑/形状/tiling/生命周期规划与执行 |
| [graph_cli.c](tools/c/graph_cli.c) / [graph_gate.py](tools/python/graph_gate.py) | SwiGLU FFN 文件接口、独立数值参考与边界传输计数 |
| [custom_ops.h](include/custom_ops.h) | 支持范围与调用契约 |
| [custom_ops.c](src/custom_ops.c) | Tensor/Buffer/View、原语发射、链提交及混合归一化 |
| [linear_plan.c](src/linear_plan.c) | M/N/K 自动分解、输出块几何、BO 容量与提交数规划 |
| [linear_hardsigmoid.c](src/linear_hardsigmoid.c) | 完整组合算子上下文；一次打包 W/b，重复执行新输入或调用方 cube |
| [operator_cli.c](tools/c/operator_cli.c) | 读取用户 FP16 文件，执行完整算子，成功后写出结果 |
| [cli_gate.py](tools/python/cli_gate.py) | 独立 Python 数学参考与真实文件接口验收，无第三方包 |
| [probe.c](tools/c/probe.c) | 独立 CPU 参考、阶段对照、重复链及哨兵验收 |
| [reformat_template.h](src/reformat_template.h) | 从厂商 p0312 trace 提取的固定寄存器模板；源 SHA256 在文件头 |
| [run-native.sh](scripts/run-native.sh) | 板端串行验收，捕获内核 journal 区间 |
| [run-board.sh](scripts/run-board.sh) | 上传源码及两个依赖到新的板端目录，回收失败或成功证据 |
| [summarize.py](tools/python/summarize.py) | 源码/输入/输出哈希与完整性门禁 |
| [two_task_probe.c](tools/c/two_task_probe.c) / [two_task_trace.c](tools/c/two_task_trace.c) | 四组对照与实际 ABI/IRQ/lifetime 采集 |
| [two_task_verify.py](tools/python/two_task_verify.py) / [two_task_campaign.py](tools/python/two_task_campaign.py) | 独立过程核验及十二进程完整性门禁 |
| [run-two-task-board.sh](scripts/run-two-task-board.sh) | 单独复跑 Two-Task 实验 |
| [overlap_probe.c](tools/c/overlap_probe.c) / [overlap_verify.py](tools/python/overlap_verify.py) | 两个 suite：MatMul/LUT 与 GEMM/GEMV（B8/B16、双进程同核/跨核）、位图消融及独立证据核验 |
| [run-overlap-board.sh](scripts/run-overlap-board.sh) | 单独复跑同核重叠/并发实验（含受控超时探索） |

## 构建与运行

先按 [依赖获取说明](../upstream-components/README.md) 恢复 `rocket-userspace` 和 `rknpu-submit` 的清单版本。
本目录使用其中的可移植子集，主机编译不需要 ARM NEON、libdrm、厂商 SDK 或内核头文件。

```bash
cmake -S research/custom-ops -B research/custom-ops/build-host
cmake --build research/custom-ops/build-host -j4
ctest --test-dir research/custom-ops/build-host --output-on-failure
```

以上仅验证布局、边界、发射与链结构，不证明 NPU 数值正确。
主机 AddressSanitizer/UndefinedBehaviorSanitizer 验证可以另建目录：

```bash
cmake -S research/custom-ops -B research/custom-ops/build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build research/custom-ops/build-sanitize -j4
ctest --test-dir research/custom-ops/build-sanitize --output-on-failure
```

仓库 `.env` 提供 `ORANGE_PI_HOST`、`ORANGE_PI_USER`、`ORANGE_PI_PASSWORD`，可选 `ORANGE_PI_WORKSPACE`。
安装 `sshpass` 后从主机执行：

```bash
bash research/custom-ops/run-board.sh
```

脚本会上传本框架及两个依赖的实际源码，不复用旧板端二进制；结果存于新的时间戳目录，拒绝覆盖已有目录。
完整板端验收需要读取 `/sys/kernel/debug/rknpu/` 和 kernel journal 的权限。
直接在板端构建时，两份依赖与 `custom-ops` 的相对位置与本仓库一致即可；也可设置 `UPSTREAM_ROOT`。

```bash
bash research/custom-ops/run-native.sh
```

硬件测试使用锁串行运行，各个核的 mask 为 `1/2/4`，每次在新进程中读取配置。
单独排查可使用 `custom_ops_probe composite|chain|primitives|pool|reformat|norm|operator`。
没有 NPU 时退出 `77`；硬件失败退出非零，不回退 CPU 后报告成功。

## 直接使用完整算子

在板端完成构建后，可以直接输入自己的张量：

```bash
RKNPU_CORE_MASK=1 ./build/linear_hardsigmoid \
  29 37 45 x.bin weights.bin bias.bin y.bin chain 10
```

输入文件分别是 `X[29,37]`、`W[45,37]`、`b[45]`，输出为 `Y[29,45]`。
均采用 row-major、小端 IEEE FP16 原始字节，没有文件头。计算约定为 `Y = HardSigmoid(X @ W.T + b)`。
尾部的 `10` 表示同一权重上下文重复运行十次；`chain` 可换为 `separate` 作对照。
重复性检查应固定 core mask，以免把其他 FP16 输入上的每核舍入差异当成执行错误。
维度越界、文件长度错误或非有限输入会报错；成功之前不替换已有输出文件。

应用程序可直接链接 `custom_ops` 目标：

```c
co_linear_hsig *op = NULL;
int rc = co_linear_hsig_create(fd, M, K, N, weights, bias, CO_CHAIN, &op);
if (!rc) rc = co_linear_hsig_run(op, x, y);
/* 后续输入可继续调用 co_linear_hsig_run(op, next_x, next_y)。 */
co_linear_hsig_destroy(op);
/* 最后由调用方 rocket_close(fd)。 */
```

创建时复制并打包权重/bias；调用方随后可以释放原数组。上下文保留 BO 和完整寄存器模板，
运行时提交缓冲区仍由执行层创建。权重驻留在设备可访问 DDR，不宣称 CBUF 常驻或零分配。
`co_linear_hsig_run_cube(op, input_view, output_view)` 直接消费/生成调用方的设备 cube，
支持 16 B 对齐偏移，不做输入输出张量的 CPU 拆包/重包；调用方负责评分输出。
上下文同步执行且不支持并发调用；销毁上下文之后才能关闭 fd。

逻辑 M/N/K 均支持 `1..4096`。`co_linear_plan_init` 自动将 M/N 拆成不超过 256 的输出块，
K 拆成不超过 256 的归约块；`co_linear_plan_tile` 给出每个输出块的位置、尾块尺寸与 task 数。
规划器在分配 BO 前计算驻留 payload、DMA 余量和 guard 的容量，默认上限为 256 MiB；
该估计不包含驱动页对齐及提交时临时 BO。它是基于已有可靠 primitive 范围的确定性规划，尚无成本模型选 tile；静态 FP16 子图的自动规划由 `co_graph` 提供，详见上面的图编译说明。

Host `create/run` 和文件 CLI 自动分解与拼接完整张量；相同的 256 行 M 块复用 N 上下文及权重，
非满尾块最多增加一组上下文。Host 输出先暂存，全部成功后才写入调用方数组。
每个输出块独立提交：chain 模式每个块一次 Submit，separate 模式每个 task 一次 Submit。
K 的部分和与 bias/激活仍全部在设备上执行，CPU 仅处理输入输出边界的布局。

`co_linear_hsig_run_tiles(op, inputs, mtiles, outputs, output_tiles)` 接收设备 tile 集合：
输入按 M 分块，使用 `co_tensor_input(tile_rows,K)`；输出按 M-major/N-minor 排列，
使用 `co_tensor_matrix(tile_rows,tile_cols,CO_F16)`。所有 View 和输入/输出、输出/输出别名在首次提交前检查。
这一路径不读回/重排张量，但仍有输出区域标记、guard 检查和 cache 同步；设备错误可能留下已完成的前序输出块。
`run_cube` 保持 M/N≤256 的连续 Cube 契约；大 M/N 的单个全局 Cube 不支持自动行切片，返回 unsupported，需使用 tile 集合。

单个输出块的物理输出超过 32768 元素时，LUT 以两段线性 View 覆盖完整 cube，
避免 `256×256` 输出越过单 task 的 DPU 宽度字段；仍只需要一次链式 submit。
每个普通输出块为 `ceil(K/256)+2` 个 task，大输出块多一个 LUT task。每段累计值保存在两个设备 Cube 中交替交接。
K<=256 仍为 3/4 个 task。大 K 输入 Cube 使用 `co_tensor_input` 描述，每个 MatMul 的实际 K tile 仍不超过 256。

多 tile 验证可从仓库根运行：

```bash
bash research/custom-ops/scripts/run-multitile-board.sh
```

它上传新源码并在三个核上验证独立 Python 数值参考、chain/separate、设备 tile 集合与提交计数。
当前实测记录见 [多 tile 验收](results/2026-10-13/multitile-20261013-144659/README.md)。

## 布局与数值契约

- MatMul 约定 `X[M,K] × W[N,K]^T → Y[M,N]`。主机边界是 row-major；设备输入/FP16 输出是 C2=8 cube。
- 单 tile 的逻辑 `M/K/N` 为 `1..256`。物理行数补到至少 12 且为 4 的倍数；物理通道补到 32 的倍数。
  这是本接口同时适配 MatMul 和 EW 的保守范围，不能当作硅片的普适限制。
- 权重独立打包为 `(N/16,K/32,16,32)`。FP32 归约输出使用 C2=4，不直接作为 FP16 输入别名。
- 上传时清零 padding。非线性可能改变 padding 的值；下一阶段必须以逻辑范围为准，并使用正确的零权重或系数处理补齐通道。
- EW 只支持相同逻辑/物理 shape。Bias 在上传时显式展开到对应 cube；identity 权重在准备阶段生成。
- 每个 Buffer 额外保留 32 KiB DMA 读取余量和 64 B 尾部 guard。验收同时检查输出非有限值、整个余量区写越界以及尾部 guard。
- 单个 View 支持 16 B 对齐偏移，地址及最后一字节必须在 32 位设备地址范围内；CPU 映射地址不能作为 DMA 地址。
- 原语不使用重叠输入/输出 View。Buffer 的 fd 身份在提交时检查，View 不取得所有权，所有 Buffer 必须活到提交完成。
- 每个 task 发射完整配置；不使用 delta task、CBUF reuse 或跨 job 隐式状态。
- 一条链内既读又写的 BO 仅放入输出 handle 列表。链每段长度独立计算并校验，不能用统一长度处理 126-word MatMul 与 1070-word LUT。
- 当前禁止一个链混合 PPU 和 DPU 完成类别。PPU 独立提交并使用对应完成标志。

`co_execute(..., CO_SEPARATE)` 在厂商路径上逐 task 提交和等待；`CO_CHAIN` 使用单次提交。
计算期间的“直接交接”表示没有 CPU 张量复制或重排；仍有驱动完成等待和 cache 同步。
中间结果位于设备可访问 DDR BO，不能称为全程 CBUF 驻留，也未做性能收益声明。

## 原语支持范围

| 接口 | 范围与边界 |
|---|---|
| `co_emit_mm` | 有界 FP16 single-tile；可输出 FP16 或 FP32；归约使用第一输出通道 |
| `co_emit_ew` | 相同 shape 的 add/mul；CNA identity Conv + DPU EW；调用者提供正确 identity 权重 |
| `co_emit_lut` | HardSigmoid/sigmoid/GELU gate；有界 exp/rsqrt；完整 LE/LO 各 513 项，表项最小为 1；每次重新装载 |
| `co_gated_activation` | SiLU/erf-GELU；LUT gate + EW multiply，输入/中间/输出均为设备 cube |
| `co_linear_hsig_*` | 完整 `HardSigmoid(XW.T+b)`；权重/bias 一次打包；重复 host/cube 入口；大输出 LUT 分段 |
| `co_ones_upload` | 最后一维 sum/mean；mean 的 `1/K` 权重舍入到 FP16，存在系数量化误差 |
| `co_emit_pool` | FP16、无 pad、kernel=stride、各轴可整除；核尺寸 1..16，avg 各轴至少 2；当前空间面最多 256 点 |
| `co_emit_reformat_8x8x8` | 两个线性 512-element BO；实测 `out[c,a,b] = in[a,b,c]`，等价三维 perm `[2,0,1]` |
| `co_norm_hybrid` | FP16 input/output；NPU square/reduce/affine；CPU 每行统计/rsqrt及系数展开 |

固定 reformat 模板来自 ONNX p0312 的厂商命令，但本接口描述的是实际线性 BO 的字节排列。
它没有自动完成 Runtime 外部 NCHW/NHWC 与内部 cube 的边界转换，也不支持任意 shape/perm。

有界 exp 的输入域为 `[-4,4]`，有界 rsqrt 为 `[0.25,4]`，调用方保证逻辑元素落在该域。
输出分别归一化为 `exp(x)/64` 和 `rsqrt(x)/4` 写入 Q0.15 表，通过输出转换器的指数移位恢复。
发射器不能检查设备张量的值域。Rsqrt 表在允许域下方的 `0.125` 夹取，
使 `x=0.25` 两侧的插值样本仍落在光滑函数上；不能把这个配方推广到任意幅值的归一化统计。

归一化不等同于厂商 18-task LayerNorm 链的手写复刻。CPU 下载每行归约结果，计算 variance/rsqrt，
并展开上传 `M×N` affine 系数；输入激活 cube 不做 CPU round trip。
FP16 平方溢出、负方差和非有限系数会报错；尚未解决近常数行的消减误差，也没有任意幅值数值保证。

## 下一步扩展接口的方法

给定新表达式，先拆成已有原语，再逐段添加独立 CPU 参考与 BO 边界检查。
新增 shape 必须证明生产者输出字节与消费者输入字节匹配；线性组合算子现有 M/N/K 分解不能直接推广到 pooling、任意 transpose 或归一化。原语的 256 上限保持不变。

现有历史 Conv 发射器的后续 task 损坏根因仍未定位。
这里已提供数值可靠的新基线；后续可保持输入、权重、输出区域及核一致，比较两条路径的配置和尾写，逐字段二分。
本轮结果不能证明旧路径已修复，也不能把旧负结果解释为硬件不支持 task 链。

## 来源与许可

新增 C 代码标记为 GPL-3.0-or-later，并与同许可的上游寄存器生成器/提交适配层一起构建。
不将其他原实验文件整体重新声明许可。底层源码版本来自 [获取清单](../upstream-components/provenance.json)，
每次验收的实际源码哈希来自结果中的 `source-manifest.json`。

已有 `summary.json` 时，`summarize.py RESULTS` 只核对记录的文件集合与 SHA256，不重写清单。
新验收使用 `--create RESULTS`，拒绝覆盖旧清单；数值计算仍需板端复跑。

固定 reformat 的来源为本地
`rknpu-vendor/experiments/results/bn_lower_20260928/em0d/tr_transpose_p0312_fp16.jsonl` 的 task0 前 69 条配置。
模板清除了源/目标设备地址；程序运行时重新绑定 BO，并追加完整标准 task trailer。
