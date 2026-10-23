# GDN 逐 Token 执行与动态操作数对照

2026-10-20。目标是 RK3588 / 厂商 rknpu 路径上的有界 FP16 单 head 递推。
新增 [co_gdn.h](../custom-ops/include/co_gdn.h) / [co_gdn.c](../custom-ops/src/co_gdn.c)
和 [gdn_probe.c](../custom-ops/tools/c/gdn_probe.c)，不改变已有原语的别名或广播契约。

## 已实现的路线

| 路线 | 实际数据流 | 必须计入的成本 |
|---|---|---|
| GEMV-A | State 为 feature cube；动态 k 打包为 weight；输出 dv×1 | k 的清零/打包、BO prep/copy/fini、发射和执行、输出读回 |
| GEMV-A2 | 同一个旧 State，weight 两行分别为 k、q；输出 dv×2 | 两向量打包；与两次独立 GEMV 的 A_pair 对照 |
| GEMV-B | k 为 feature；State 为 weight | 从已更新的 feature State 下载/拆包，再打包上传 State weight；未实现设备端 weight State 更新 |
| GEMV-C | CPU AArch64 NEON，FP16 存储、FP32 FMA 累加 | CPU-resident GEMV；另报从设备 State 读回的 bridge 成本 |
| Outer-K1 | e[dv,1] × weight[k,1]^T | e、k 的不同布局打包和 K=32 的物理 padding |
| Outer-EW | 显式物化 E[r,c]=e[r]、K[r,c]=k[c]，再同 shape EW Mul | 两个完整矩阵的展开、打包、上传；EW 使用既有 identity Conv + DPU 路径 |

GEMV-B 的样本每次都改变 State，不把静态权重缓存收益当作动态 State 维护收益。
共同起点是 State 已在 feature cube；该状态准备在计时外。CPU-C 的主计时起点是 CPU row-major
State，不能把它的 `total_us` 与设备起点混为一谈；`read_or_bridge_us` 单列设备到 CPU 的边界。
CPU 每个样本执行 64 次 GEMV 后取均值，包含 FP16→FP32 转换，使用编译器屏障防止删除计算。
其他路线每个样本一次操作；两次 warmup 不纳入各路线的 20 个正式样本，路线顺序轮换。

`pack_us` 是预分配 staging 上的清零与布局 scatter；`upload_sync_copy_us` 包含共享内存 BO
的 prep/memcpy/fini，并不是独立 PCIe 上传或纯 DMA 时间。`execute_api_us` 包含提交、命令缓冲构造、
阻塞等待和执行器同步；`total_us` 还包含发射及输出读取，不含 oracle、guard 检查和一次性分配。
递推 API 的 `total_us` 则包含该 API 自身的系数分配、打包、guard 检查及输出下载。

## 六 Task 递推

以 S[dv,dk] 为约定：

```
D = alpha * S              # MatMul，alpha * I 的动态 weight
p = D * k                  # GEMV-A
r = v - p                  # p * (-I) + v：负单位权重 + DPU EW Add
e = beta * r               # K=1 MatMul，beta 为单元素逻辑 weight
S_next = D + e * k^T       # K=1 MatMul + DPU EW Add，prior 与输出 BO 分离
o = S_next * q             # GEMV-A，读取更新后的状态
```

`co_gdn_create/reset/step/export/destroy` 支持逻辑 dv/dk=1..256，单 head、固定形状；不含投影、
短卷积、k/q 归一化、gate 生成、输出门控或多 head 调度。alpha/beta 输入为有限 FP16，范围 [0,1]。
所有跨 task 中间结果为 FP16；这不是 FP32 状态或精确累加的实现。

每步六次动态上传：k 的两种 weight 布局、q、v、alpha 对角权重和 beta 权重。
负单位权重在创建时生成。仅下载最终输出向量，State 和中间预测/残差没有 CPU 数值计算或布局转换。
chain 模式每步一次 Submit、六 task；separate 每步六次 Submit。执行器仍有 CPU 控制、cache
同步与 guard 检查，不能称为无 Host 参与或 CBUF 常驻。

两个 State BO 交替；只有完整执行成功才切换 current。设备或下载错误将上下文标记为不可用，
必须 reset 才能继续。late-failure 测试在真实硬件已经执行后注入错误，检查输出未提交、后续调用
拒绝以及 reset 恢复。该策略不承诺在失败后可导出旧状态或直接重试。

## 标量、减法和广播的边界

- 已验证对角 MatMul 缩放和显式物化标量矩阵后的 EW Mul；独立覆盖 0、1、-0.5、0.9375。
- 已验证负单位 MatMul + EW Add 的减法。没有新启用未知的 DPU SUB 寄存器配置。
- alpha 的已验证等价实现会物化 dk×dk 对角权重并做额外乘加；不是免费标量广播。
- beta 使用逻辑 1×1 weight，仍按物理 32×32 weight 补齐。
- 已验证无广播的单乘子原生标量探针（LUT task 去 LUT、保留 BN multiplier），
  但未验证通用 DPU 标量直接广播或动态任意向量广播。Outer-EW 是显式物化的替代路线。

因此已补齐可执行公式，不代表获得最优缩放或广播原语。

## 合并 k/q 的语义

A2 得到同一旧状态的 `[S_old k, S_old q]`。最终输出必须修正：

```
e = beta * (v - alpha * S_old k)
o = alpha * S_old q + e * (k^T q)
```

独立 Python FP64 oracle 检查这一恒等式。现在已有实验实现把 `[Dk,Dq]`、残差、状态更新、
`k^T q` 主机点积和输出修正接入完整递推（`merged` 路线，7 task）。端到端结果见下文实验 4：
它没有比同框架复用的六 task 路线更快，因此没有替换默认配方。
FP16 中重新结合运算的舍入已单独验收，不能把 A2 单算子的省时称作完整 GDN 加速。

## 原位写回诊断

生产接口始终拒绝原位别名并使用 Ping-Pong。独立 `gdn_probe inplace` 从合法 out-of-place
rank-1 累加命令出发，仅重绑定 WDMA 地址及执行器输出 metadata，尝试读取和写入同一 State BO。
每次同时运行 out-of-place 正对照，检查结果和 guard。它是研究探针，不导出可复用的原位 API。

三核、五个形状、每形状八次独立更新中，当前配方未发现数值错误。每次都重新上传初始状态，
尚未覆盖长时递推、多 task 原位依赖、不同 stride/tile 或其他累加配方，因此不放开通用别名约束；
也不推翻已有其他配置下原位累加损坏的记录。超时/设备错误会终止后续原位诊断，避免继续提交。

## 验收与复跑

```bash
cmake -S research/custom-ops -B /tmp/gdn-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gdn-build -j4
ctest --test-dir /tmp/gdn-build --output-on-failure
bash research/custom-ops/scripts/run-gdn-board.sh
```

运行脚本读取忽略的 `.env`，上传实际源码，板端新目录编译；持有既有 NPU 实验锁。
覆盖三个核（mask 1/2/4）、chain/separate、dv×dk 为 1×1、19×37、32×32、128×128、256×256。
每条递推先检查连续 32 步的状态和输出，再重跑 32 步且不导出任何中间状态；两次结果逐位一致。
同一核的 chain/separate 输出和所有状态也必须逐位一致。

独立 [gdn_gate.py](../custom-ops/tools/python/gdn_gate.py) 从保存的 FP16 输入计算 FP64 递推，
不使用设备状态修正参考值。误差门限 `abs <= 0.0003 + 0.006*abs(reference)`；同时拒绝非有限值。
[验收器负例](../custom-ops/tests/python/gdn_gate_test.py) 覆盖丢失/重复样本、NaN 时间、非 SIMD
板端基线、错误 task 数、缺少连续状态验收和缺少最终 PASS。
[campaign](../custom-ops/tools/python/gdn_campaign.py) 重验回收数据、双模式逐位一致、内核错误，
并生成结果文件 SHA256；源清单和 binary SHA256 独立保存。

尚未覆盖真实模型精度、长序列漂移、多 head 批处理或并发上下文。
CPU-FP32 状态实现和完整 merged 递推已在 2026-10-20 补充实验中覆盖，但仍限于单 head、
batch 1、最大 256、每 block 32 步，不是长序列漂移结论。

## 2026-10-20 真机结果

正式证据：[gdn-20261020-091633](../custom-ops/results/2026-10-20/gdn-20261020-091633/)，
[完整 campaign.json](../custom-ops/results/2026-10-20/gdn-20261020-091633/campaign.json)。
原始大文件在忽略的本地 results 目录；此报告和复跑工具随源码保存。
此前 091237 是三形状初验，091457 加入边界/原位；最终结论使用 091633 的五形状、收紧门限、
64 次 CPU 计时和独立完整性验收，不混用前轮计时。

主机及板端 CTest 均 **22/22**。三核 × 双模式 × 五形状全部通过，960 个带逐步状态 oracle 的
Token，加上 960 个不中途导出状态的重复 Token；另外各上下文注入一次真实执行后的失败。
最大状态绝对误差 **7.761e-5**，最大输出绝对误差 **4.594e-5**（相对独立 FP64 递推）。
所有同核 chain/separate 保存输出和状态逐位一致，kernel interval 无日志事件。
原位诊断三核合计 120 次，当前五形状八次试验均无 mismatch，生产别名策略仍不变。

板端 NPU 记录为 **1 GHz**，CPU 绑定 A76 CPU4，CPU governor 为 **ondemand**，未锁频。
以下是各核各自 20 次样本中位数的范围，单位 µs；不是跨核并行执行，也不是受控频率下的硬件极限。
调频和执行 API 包络影响明显，数值不能用于宣称某个核更快。

| 路线 | 32×32 | 128×128 | 256×256 |
|---|---:|---:|---:|
| GEMV-A，一个 k | 102.08–104.12 | 99.17–154.29 | 161.29–226.77 |
| A_pair，分别 k/q 两次 | 201.83–206.79 | 197.89–309.31 | 328.41–458.49 |
| GEMV-A2，合并 k/q 一次 | 101.50–103.54 | 100.33–155.75 | 164.35–233.18 |
| GEMV-B，含 State 转换 | 126.29–130.37 | 337.16–412.70 | 1167.96–1213.46 |
| CPU-C，CPU-resident NEON | 0.52 | 8.19–8.99 | 32.72–32.78 |
| Outer-K1，含输入/输出边界 | 130.52–135.77 | 377.41–446.97 | 1096.07–1146.38 |
| Outer-EW，含展开/边界 | 142.62–150.79 | 580.99–585.37 | 1809.62–1878.74 |

CPU-C 的 bridge 不含在上表 CPU-resident 时间内，例如 core1/128×128 的 bridge 中位数为
202.56 µs。若已经选择设备 State，临时回 CPU 的比较应额外计入这个边界；CPU 常驻状态的
实现不需要该 bridge。GEMV-B 的总时间已经含 feature→Host→weight 的转换。

128×128 的完整六 task `co_gdn_step` 中位数：chain **1194.80–1198.44 µs/token/head**，
separate **1643.23–1724.31 µs/token/head**。这是输入动态打包、提交、guard 同步、输出下载的
完整 API 时间，不含测试用 State 导出和独立 oracle。没有整模型、CPU 完整递推或多 head 吞吐结论。

K1 外积的物理 K 从 1 补到 32；以 128×128 为例，逻辑 16384 MAC 对应 524288 个物理 Conv MAC，
输入加权重共 16384 B。EW 方案上传两个完整 32768 B Cube，共 65536 B，另有 identity Conv 的
2097152 MAC。这解释了本配方里 K1 优于显式 EW 展开，但不能推广为所有 DPU 广播实现的结论。

目前可以确认：GEMV-A、K1 外积和保守缩放/减法配方构成了可复用的设备 State 更新链。
动态向量仍由 Host 打包；GEMV-B 的全设备 weight-layout 维护、通用原生标量/向量广播和
通用原位更新仍需独立优化与验收。合并 GEMV 的完整输出修正已接入并做了端到端验收，
但当前配方没有收益。当前性能保留 CPU SIMD 为竞争基线。

## 2026-10-20 补充：CPU 基线、六 Task 成本、标量缩放与合并 k/q

正式证据目录：`../custom-ops/results/2026-10-20/gdn-experiments-20261020-110916/`
（本地忽略目录；含 `experiments.json`、`REPORT.md`、原始日志和二进制 SHA256）。
实验二进制 SHA256 为 `e198e9bdeb15827d54e87313fd3c95015ea34af67a1c01ff21f43ae8e5e2dc12`；
板端为 RK3588 / rknpu 0.9.8，NPU devfreq 固定在 1 GHz，CPU4 为 performance 2.304 GHz；
运行前记录原策略，结束后恢复。每个 Token 前后都记录 CPU/NPU 频率，实际观测集合分别为
`{2304000}` 和 `{1000000}` kHz。三核 mask 1/2/4 交错运行；每个形状每核 5 个 block × 32
Token，7 条路线均同时跑无插桩和带插桩版本。主机与板端 CTest 24/24 通过，独立 FP64 oracle
门禁 PASS；插桩前后保存的输出和状态逐位一致。

### 实验 1：完整递推的 CPU-FP16 / CPU-FP32 / NPU-FP16

CPU-FP16 的 State 常驻 CPU，按 NPU task 边界依次舍入 `D、p、r、e、S_next、o`，秩更新在
FP16 存储前使用 FP32 FMA；CPU-FP32 的 State、中间量和累加均为 FP32。两者都不含每步分配
或设备调用。NPU-FP16 为实际六 Task chain；`reuse` 只是同一配方的框架预分配版本。
表中是三个核各中位数的中位数，括号为三核范围；不是单核极限。

| n | CPU-FP16 | CPU-FP32 | NPU 六 Task 默认 | NPU 六 Task reuse | NPU 默认 / CPU-FP16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 8.75 us | 6.71 us | 667.91 us | 625.91 us | 76.3x |
| 128 | 31.21 us | 24.50 us | 1043.72 us | 967.15 us | 33.4x |
| 256 | 110.83 us | 104.27 us | 1762.08 us | 1483.25 us | 15.9x |

结论：CPU-resident 完整递推比当前 NPU API 端到端路径快约 15–76 倍；该比较对 CPU 有利，
因为 CPU 的 State 已在本地且无设备边界，而 NPU 时间包含打包、提交/等待、Guard 和下载。
CPU-FP16 与 CPU-FP32 在 256 仅相差约 6%，64/128 时 FP32 反而更快，说明小形状主要受
转换/控制而非状态带宽限制；不能仅凭 FP16 存储推断 CPU-FP16 一定更快。

### 实验 2：六 Task chain 的真实成本

`execute_api_us` 是主机侧包络：命令 BO 分配/构造、阻塞提交/等待和执行器输出同步，不是
纯硬件执行时间。Guard 是主机对每个驻留 BO 的完整 slack/integrity 扫描，保留并单独计时。
下表来自带插桩样本，用于分解；正式延迟用上表无插桩中位数。

| n | arm | total | host alloc | pack k/q/v | pack coeff | BO sync/copy | emit | execute API | guard | download |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | default | 677.0 | 0.3 | 5.9 | 16.0 | 19.8 | 7.6 | 182.6 | 437.5 | 4.1 |
| 64 | reuse | 628.0 | 0.0 | 2.7 | 1.2 | 19.0 | 7.9 | 152.7 | 438.7 | 4.1 |
| 128 | default | 1070.7 | 0.3 | 7.7 | 57.8 | 29.5 | 7.9 | 508.7 | 447.4 | 5.2 |
| 128 | reuse | 966.0 | 0.0 | 4.4 | 2.3 | 24.2 | 7.9 | 470.7 | 447.7 | 5.0 |
| 256 | default | 1822.6 | 0.7 | 10.1 | 230.7 | 47.8 | 9.9 | 1027.2 | 471.3 | 6.7 |
| 256 | reuse | 1483.4 | 0.0 | 6.8 | 4.7 | 37.6 | 10.2 | 939.4 | 473.1 | 7.0 |

可复用系数、命令 BO 和 Host staging 提前到 `co_gdn_create` 后有稳定收益：无插桩端到端
中位数从 667.91/1043.72/1762.08 降到 625.91/967.15/1483.25 us，即 **-6.3%/-7.3%/-15.8%**。
主要节省项是 alpha 对角权重打包 `16.0/57.8/230.7 us -> 1.2/2.3/4.7 us` 以及每 Token
命令 BO 分配 `约 39–70 us -> 0`；k/q/v 布局转换本身每项只有约 1–5 us。
代价项没有消失：Guard 仍占 64 形约 65–71%、128 形约 42–46%、256 形约 26–32% 的包络，
与 `execute_api_us` 一起构成主要成本。Guard 不能为了降低数字而删除。

单独提交诊断（只作定位，不是 chain 内单 task 硬件时间）显示：128 形六项分别约
171/99/86/84/163/97 us，alpha 缩放和 `S_next=D+ek^T` 最大；256 形约
515/159/111/87/394/171 us。每项都包含自己的 Host 提交、阻塞等待和同步包络。

### 实验 3：alpha/beta 标量缩放 A/B/C/D

A = 对角权重 MatMul；B = 同形状标量矩阵 + EW Mul（含矩阵构造和上传）；C = 已验证的
原生标量 BN multiplier 探针（该探针从现有 LUT task 生成器中去除 LUT 访问，仅保留
单乘子，不是通用 DPU 广播能力）；D = CPU 常驻直接缩放。NPU 三项均把输出下载和验证
排除在计时外；D 为 64 次重复的均值。表内为三个核合并样本中位数。

| geometry | n | A diag MatMul | B expanded EW | C native scale | D CPU | C vs A |
|---|---:|---:|---:|---:|---:|---:|
| alpha state | 64 | 273.14 | 279.70 | 252.73 | 1.85 | -7.5% |
| alpha state | 128 | 412.12 | 442.74 | 270.08 | 7.38 | -34.5% |
| alpha state | 256 | 706.99 | 865.66 | 349.70 | 29.60 | -50.5% |
| beta vector | 64 | 257.25 | 259.73 | 250.25 | 0.03 | -2.7% |
| beta vector | 128 | 281.16 | 285.25 | 254.04 | 0.06 | -9.7% |
| beta vector | 256 | 286.12 | 294.87 | 255.93 | 0.11 | -10.6% |

三条 NPU 路线的输出与逐元素 FP16 标量乘参考完全一致（所有样本 `max_abs=0`），包括
0、1、-1、0.9375、0.6、2、2^-14 和 -0.5。B 在所有测试形状都比 A 慢，说明显式矩阵
构造/上传没有抵消 EW 节省；C 的 execute 包络显著低于 A，但 Guard 约 157–190 us 仍在
small shape 占主导。C 只证明了已构造形状上的标量乘子，不可推广为任意广播或任意维度的
DPU 能力。

β 路径重写 `e = beta*v - D*(beta*k)` 也作为 `beta_fold` 路线进入完整递推：Host 先乘
`beta*k、beta*v`，去掉独立 β Task，五 Task 完成递推。相对同框架 reuse 六 Task，64/128 形
分别约 **-11.2%/-9.4%**，但 256 形为 **+3.3%**，没有一致收益；因此保留为实验路线，
不把它当作已证明的默认优化。FP64 oracle 已覆盖重结合后的状态和输出误差。

### 实验 4：合并 k/q 的完整递推

`merged` 路线用一次 `[Dk,Dq]=D[k,q]`、Host 点积 `k^Tq`、额外输出修正 Task 完成：

```
[p,u] = D[k,q]
e     = beta(v-p)
S'    = D + e k^T
o     = u + e (k^Tq)
```

它相对同框架 reuse 六 Task 的端到端中位数：64 形 625.91 -> 710.20 us（**+13.5%**），
128 形 967.15 -> 1054.94 us（**+9.1%**），256 形 1483.25 -> 1807.29 us（**+21.8%**）。
虽然旧状态 GEMV 合并且 `k^Tq` 在 Host 上很便宜，完整路径多出修正 Task、额外的输出 BO/
Guard 和更长的执行包络，**没有获得端到端加速**，因此不采用 A2 单算子收益作为递推优化结论。

数值上门禁通过，但 `merged` 在 256 形的输出最大绝对误差为 2.859e-4（六 Task 为 2.472e-4）；
相对六 Task 输出有约 49% 的 FP16 值不同，最大差 2.441e-4。State 路径与六 Task 相同，
其状态最大误差保持 2.903e-4。该差异来自 FP16 重结合，不是 oracle 失败；但它说明合并输出
必须作为独立配方验收，不能只复用旧 GEMV 误差结论。

### 精度与边界汇总

FP64 oracle 门限为 `abs <= 0.0003 + 0.006*abs(ref)`，所有 7 条路线、三形状、三核均通过。

| arm | 最大输出绝对误差 | 最大状态绝对误差 |
|---|---:|---:|
| CPU-FP16 | 2.472e-4 | 3.153e-4 |
| CPU-FP32 | 5.556e-8 | 3.823e-8 |
| NPU 六 Task | 2.472e-4 | 3.153e-4 |
| beta-fold | 2.408e-4 | 3.153e-4 |
| merged k/q | 2.859e-4 | 3.153e-4 |
| native alpha | 2.472e-4 | 3.153e-4 |

本轮只回答单 head、batch 1、`dk=dv∈{64,128,256}`、每 block 32 步的端到端行为；
仍不包含真实 GDN 层的投影、短卷积、归一化、多 head 调度或长序列漂移。NPU 时间固定
1 GHz 下测得，是算法对照而非能效/生产频率结论；`submit_wait_us` 和 separate 诊断都不能
冒充纯设备执行时间。复跑入口：

```bash
bash research/custom-ops/scripts/run-gdn-experiments-board.sh
python3 research/custom-ops/tools/python/gdn_experiment_report.py \
  research/custom-ops/results/2026-10-20/gdn-experiments-20261020-110916 \
  -o /tmp/gdn-report.md
```
