# RK3588 regcmd 时间可观测性实验设计

2026-10-21。状态：设计 + 部分软件实现，**未上板验证**。针对当前 vendor hook 和 RK3588；
不将 mainline rocket、RK3576 的现象直接推广到此路径。

### 当前进展（2026-10-21）

**P0 尚未验收。** 下列是用户态原型的软件实现与其宿主机自测状态，不是"可进入正式采集"的结论。

| 部件 | 状态 | 位置 |
|---|---|---|
| 读取预算统一记账、丢弃半成品样本、ready/armed 分离、截断自证 | 已实现，自测通过 | `rknpu_ioctl_hook.c` `bw_*()` |
| 时间原点是**提交入口**，不是硬件 kick | 已改名 `submit_enter_ns`，报告口径同步 | 同上 + 分析器 |
| 电源复查 | 用户态尽力而为，首读前必查；仍有检查后掉电竞态 | 同上 |
| 采样所有权 | 已绑定 owner ioctl，外来 ioctl 只计数不干扰；**内核侧仍缺** | 同上 |
| 回归测试：采样器 24 组、相位分析器 13 组、bank 判据器 8 组、槽位规划器 7 组、输出比对 8 组、延迟分析 4 组 | 通过 | [test_blockwatch_sampler.py](../rknpu-vendor/source/hook/tests/test_blockwatch_sampler.py) |
| 相位扫描采集脚本 | 已写，未上板 | [run_phase_sweep.sh](../rknpu-vendor/experiments/run_phase_sweep.sh) |
| **测量模型与能力边界汇总** | 每条结论带证据等级（实测/间接/模型/已撤回/未建立），含 §8 收尾判断与证据入口 | [REGCMD-TIMING-MEASUREMENT-MODEL.md](REGCMD-TIMING-MEASUREMENT-MODEL.md) |
| 采样延迟拆解 | **已上板**：主延迟项是首读前的电源 debugfs `open+pread`（21–43 µs），已归因并可复现 | [board_latency_20261021](../rknpu-vendor/experiments/results/board_latency_20261021/README.md) |
| **驱动事件 × 采样器对齐** | **已上板**：ftrace 用 `mono_raw` 与采样器同钟；实测**调度路径入口**（p50 15.9 µs，kick 的**下界**）、IRQ handler 入口（124.1 µs）；边沿差值以**区间**给出（上升沿−调度入口 2.8~15.2 µs）；trace 丢失 0/9；tracing 开销**无法与负载漂移分离** | [board_ftrace_20261021](../rknpu-vendor/experiments/results/board_ftrace_20261021/README.md) |
| 单 task 内多快照 | **已上板**：`CAP=32` 时 23 个快照/次、间隔 p50 **2.62 µs**、读取区间 1.17 µs、跨度 p50 84.6 µs；**8/12 次运行内完整解析 `clear→set→clear`**（窗口宽度下界 p50 7.9 µs）。绑核反而更差（3.21→7.29 µs），该优化关闭 | [board_intra_20261021](../rknpu-vendor/experiments/results/board_intra_20261021/README.md) |
| 已验证两 task 负载 + 多 task 组位 | **已上板**：`min_chain --independent` 用"两输出逐字节相同"作判据（含 `--tasks 1` 负对照），272/272 通过；多 task 下组位**不随 task 切换**（2 task 恒 DPU_1，0 次翻转） | [board_multitask_20261021](../rknpu-vendor/experiments/results/board_multitask_20261021/README.md) |
| submit→kick 偏移上界 | **已上板标定**：两个已验证负载独立给出 **≤ 5.5 / 5.1 µs**（区间 4.1–6.4 µs） | [board_kick_20261021](../rknpu-vendor/experiments/results/board_kick_20261021/README.md) |
| 中断状态置位窗口 + 同钟锚点 | **已上板**：412 次提交全 BIT-EXACT、dmesg 无新增错误；DPU 组位与提交奇偶 105/105 对应；下降沿跟随 `hw_elapse`（阈值可分性 0.992 vs 0.727）；ABBA 配对未检出扰动 | [board_irq_20261021](../rknpu-vendor/experiments/results/board_irq_20261021/README.md) |
| S_STATUS 模块活动采样 | **已上板首轮**：字可读且稳定（177+ 样本 4 种取值），但两个已验证负载分布相同、`DPU_RDMA` 恒 `0x0001000a`，**尚不能区分 idle/busy** | [board_status_20261021](../rknpu-vendor/experiments/results/board_status_20261021/README.md) |
| 相位扫描分析器（status / marker 两种语义分开） | 已写，反例自测通过 | [analyze_phase_sweep.py](../rknpu-vendor/experiments/analyze_phase_sweep.py) |
| 标记寄存器离线筛选 | 完成；**bank 结论已修订为未知** | [FINDINGS.md](../rknpu-vendor/experiments/results/marker_screen_20261021/FINDINGS.md) |
| 读回 bank 判据（只读） | 工具与判定器完成；**板端已观测到"CNA 0x1028 读回滞后一个 run"**（95% 任务处仍读旧值），bank 机理未定 | [run_bank_probe.sh](../rknpu-vendor/experiments/run_bank_probe.sh) / [analyze_bank_probe.py](../rknpu-vendor/experiments/analyze_bank_probe.py) |
| 标记注入（长度保持） | **已上板**：三臂 applied/sync/restore 全通过、输出逐字节一致；但 **PPU 标记写不可见**，路线 B 未走通 | [plan_marker_slots.py](../rknpu-vendor/experiments/plan_marker_slots.py) / `marker_*()` @ hook / [run_marker_arms.sh](../rknpu-vendor/experiments/run_marker_arms.sh) |

"宿主机自测"指用 `-DEPERF_BW_SELFTEST` 把 `/dev/mem` 换成 calloc 窗口后验证
状态机语义；它不能替代上板，也不证明任何 MMIO 行为。

目标是补充 task 内命令进度与模块活动的时间证据。优先通过重复执行、稀疏采样获得
统计时间区间；只有找到硬件事件记录能力，才承诺单次运行的逐命令时间戳。

## 1. 已有证据与边界

- [现有 hook](../rknpu-vendor/source/hook/rknpu/rknpu_ioctl_hook.c) 的**历史行为**（已在
  §2.1 修正，保留作为证据）：`bw_thread()` 的 edge 分支持续读取 `0x003c`，仅在低 12 位
  变化时记录并读模块状态；该分支提前 return，`spin_ns/nsleep` 的限速只作用于另一分支。
  当时“每 task 记录一次”不等于“每 task 只访问一次 MMIO”。
- **历史行为**（已修正）：`bw_start()` 曾在真正 submit 前启动并等待第一个样本，
  `bw_stop()` 在 ioctl 返回后停止；一次 debugfs 电源检查与“blocking ioctl 持电源引用”
  覆盖不了这两个窗口。现在改为 ready/armed 分离 + 可取消等待（§2.1）。
  当前全核 edge 触发仍以 core0 的低位计数变化为准，不适合推断 core1/2 的边界。
- [SDPA 记录](../rknpu-vendor/experiments/results/sdpa_timeline_20261020/README.md)：
  401 tasks，15304 个已捕获 entries，期望 15382；task13 的 1102 entries 仅捕获 1024。
  在补齐捕获之前，不能对该 task 做完整命令级映射。
  **截断根因已定位**：`write_regcmd_record()` 在 `EPERF_RKNPU_REGCMD_MAX_BYTES`
  或映射可用长度处截断，1024 条 × 8 B = 8192 B 正好落在 8 KiB 边界上，
  而 hook 默认上限是 16384 B（2048 条），本来装得下 1102 条。
  即截断发生在**用户态**。但具体是 `EPERF_RKNPU_REGCMD_MAX_BYTES` 还是
  `resolve_regcmd_pointer()` 给出的映射可用长度（`available`）所致，**尚未区分**：
  两者都会在 8 KiB 边界上截断。重抓前必须先确认是哪一种，否则"把上限提到 ≥ 16 KiB"
  可能根本不生效。
  旧记录只在 `bytes` 字段里体现短载荷、没有截断标记，因此容易被误当完整流；
  现已改为同时输出 `expected_bytes` 与 `truncated`。
- [寄存器表](../../ppu-rknn-capture/registers.xml)：`0x10` 是配置地址，`0x14`
  是配置长度；没有因此获得 live PC。`TASK_STATUS` 的整个 32-bit 原值值得保留，
  但现有证据不足以将其高位解释成 regcmd counter。
- [task 模型](../../encodings/regcmd-task-model.md)：寄存器组有持久状态和 ping-pong；
  `OP_ENABLE` 自清零；CNA/CORE 的指针含驱动管理的 per-core 位。
  [同核重叠实验](SAME-CORE-OVERLAP.md)证明部分 regcmd 重划分可工作，同时记录了
  提前启动、改位图、错误分段导致超时和静默错误的负例。
- [计数器调查](../../perf/hw-byte-counters.md)：RK3588 `0x2xxx` 有读操作硬锁记录。
  不扫描保留地址，不将其他芯片的 amount 寄存器当作本芯片可用能力。

需要区分三类事件：fetch（读到命令）、register-visible（配置写入可读回）、
module-complete（模块工作完成）。命令顺序不是这三者共用的一条完成时间线。
配置写之间的间隔也不是某条命令的独占执行成本。

## 2. 推荐架构：内核管采样生命周期，hook 管命令映射

hook 保留 BO/task 捕获、数学阶段映射与实验选择；增加完整捕获和原流/实验流哈希。
驱动实验接口负责 core/job 身份、runtime PM、采样时机和预分配记录环。

建议的 debug-only 接口是 arm/read/disarm：arm 指定 job cookie、目标 core、偏移白名单、
相位延迟、最大读取次数、最小读取间隔和记录容量；启动和停止由真实 job 生命周期驱动。
持有正确的设备电源及必要时钟引用，直到采样线程/回调完全退出，再释放引用。
job 完成、异常、reset、取消均走同一停止路径；避免跨入下一 job。
不在热路径打印，不在 IRQ 中忙等；实际采样延迟必须记录，不能把定时器计划时间当实测时间。

第一版只接受单个活动 job、固定 core0、blocking submit。拒绝 async/multicore 混用；
未来通过实际调度 core ID 和 per-job 状态扩展，不能仅相信 AUTO 请求值。
如暂不改内核，只能在已验证的持电源实验环境下做限额用户态原型，不宣称生命周期问题已解决。

修正现有 hook 时：edge 与普通模式使用同一个读取预算/限速器，即使 counter 不变也限速；
显式 sampler-ready 原子握手代替用首个样本作握手，防止 phase 延迟发生在 submit 前；
ready 状态、样本发布和停止条件使用一致的同步机制，禁止并发 submit 共用全局采样缓冲。

### 2.1 用户态原型已落地的部分（`rknpu_ioctl_hook.c`）

| 要求 | 实现 |
|---|---|
| 统一读取预算 | 所有 MMIO 走 `bw_read_u32()`，按 `EPERF_RKNPU_BLOCKWATCH_MAX_READS` 记账；预算耗尽的半成品样本被丢弃并记 `budget_hit` |
| 统一限速 | `bw_pace()` 同时服务 edge 与自由运行模式；edge 轮询另有 `POLL_NS` 下限，counter 不变也不再是无界 MMIO 循环。所有等待都是**可取消**的：`bw_wait_cancellable()` 分段检查停止标志，等待结束后、读 MMIO 之前重新检查停止与电源 |
| 相位对齐 | `bw_ready`（线程就绪）与 `bw_armed`/`submit_enter_ns`（进入 ioctl 前由 `bw_arm()` 写入）分离。**这是提交入口，不是硬件 kick**：系统调用、驱动准备和排队都在其后，报告一律称 submit-relative；CSV 列为 `submit_enter_ns` |
| 每次读取区间 | 每核 `c*_gb_ns/c*_ge_ns`（**一组**寄存器读共用一个区间），整样本 `t_ns/t_end_ns`，`t ≤ gb ≤ ge ≤ t_end` 由测试断言 |
| 电源与停止 | `POWER_CHECK_NS` 周期复查 debugfs 电源，**首读前必查一次**；`stop_reason ∈ {cap,budget,power,run_flag}` 显式区分"正常收满"与"被截断"。这是用户态尽力而为：检查与读取之间仍有掉电竞态，不能替代驱动侧持引用 |
| 单寄存器探针 | `RAW_OFF`/`RAW_CORE`：task status + 目标寄存器 + task status，给出 `task_before/task_after/sample_validity`。`RAW_OFF` 分三档：**vetted** 只有四个模块 S_STATUS（0x1000/0x3000/0x4000/0x5000）；**experimental** 需显式开关且报告 `raw_mode=experimental`；**forbidden** 永不接受（PC 页、0x2xxx、DDMA/SDMA、S_POINTER、OP_ENABLE、中断寄存器、DPU LUT 0x4100–0x412c） |
| 采样所有权 | 每次 ioctl 生成唯一 cookie；生命周期是 `IDLE→STARTING→RUNNING→STOPPING→IDLE`，**全部在 `pthread_mutex` 下迁移**，STARTING/STOPPING 期间新的 start 一律拒绝，只有资源完全释放后才回到 IDLE。外来调用只记 `owner_conflicts`，不能移动时间原点、停止或释放采样资源。用户态只能做隔离，真正的 job 归属仍需驱动 |
| 外来 ioctl 安全 | ioctl 尾段先 `bw_owns(cookie)` 判所有权，**之后**才把参数当 `rknpu_submit*` 读 `hw_elapse_time`；`dump_memory_ranges()` 也只在 request 确实是 submit 时调用。回归测试两条：直接调 helper（IDLE），以及**RUNNING 状态下走真实 `ioctl()` 包装器**送 PROT_NONE 保护页参数——后者用负向对照（把所有权检查挪到解引用之后）确认会 SIGSEGV，说明测试确实能挡住原缺陷 |

仍未解决、必须靠内核侧（或至少上板证据）的部分：真正的 job 生命周期归属、跨 core 的
采样所有权、以及"blocking ioctl 之外仍持有电源引用"的保证。

回归测试：`python3 research/rknpu-vendor/source/hook/tests/test_blockwatch_sampler.py`
（构建两种产物 + 24 组断言，含停止可取消、电源守卫、生命周期状态机与真并发交错、保护页上的外来 ioctl（helper 与真实包装器两条路径）、RAW 分档、shell 环境隔离、假 BO 驱动的 marker 改写/恢复路径，以及端到端 sweep→CSV→分析器，全部在宿主机跑，不需要板子）。

## 3. 路线 A：不改命令流，重复执行下的稀疏快照（先做）

每次运行在**提交入口**（用户态 ioctl 之前）后延迟 φ，只做一个或少量快照，然后退出采样。
提交入口不是硬件 kick：真实 kick 与它之间还隔着系统调用、驱动准备和排队，
因此所有 φ 都是 submit-relative：submit→kick 的延迟**未标定且可能逐次变化**，
它既平移分布也展宽分布，不能靠减一个常数换算成硬件时序（见 §2.1）。
跨运行扫描 φ。一个快照先只读 task status；验证后在不同运行分别加入一个模块状态，
避免一次读取所有模块造成大的观察时间跨度。目标是先补全 task 内模块活动分布。

初始实验可采用每次 1 个快照、相位粗步长 10–20 µs，再围绕转变区缩小；
这些是试验参数，不是已证明的安全频率或时间分辨率。禁止通过缩小步长提高单次运行的 MMIO 负载。
随机排列相位，穿插无采样基线，每个相位至少 30 个有效重复；根据分布宽度决定是否增加样本。
时钟、电源、输入、缓存预热、命令流及任务布局保持可比，同时记录实测频率。

每个读取使用 `[read_begin_ns, read_end_ns]` 包住，保留请求相位与实际相位。
当前实现给的是**每组读取**一个区间（每核 1–5 次寄存器读共用一个 `c*_gb_ns/c*_ge_ns`），
不是每个寄存器各自的区间——逐个寄存器取时钟会盖过 MMIO 本身的开销。
RAW_OFF 探针另有 `task_before/task_after/sample_validity`，跨 task 的样本必须丢弃。
需要 task 归属时可用 task-before / 单个目标寄存器 / task-after，但必须把三次读取计入预算；
若 task 不同、出现归零歧义或 reset，就标记跨边界/unknown，不强行归属。
不同模块在不同运行测得的 busy 概率不能当作同次运行的并发证据；联合状态需要
单独的短序列联合快照，并报告读间偏斜。

产物是 `P(module busy | 实际提交入口相对时间, task 条件)` 和转变时间分布。
它能改善原有 boundary-hold occupancy，但单凭它仍不能定位每条 regcmd。
对短于读延迟/运行间抖动的事件报告“未分辨”，不可用更细画图网格冒充精度。

### 3.1 采集与分析工具（已写，未上板）

- **先用长任务校准**：板端实测（`min_conv --iw 32`，5 次/变体）基线迟到 p50 32.1 µs、
  窗口覆盖率 50%；把电源 fd 预开（`PREFAULT` 以外的 `PREOPEN_POWER`）后降到迟到 8.6 µs、
  覆盖率 80%。预触页与单独绑核都无收益；三者叠加 11.8 µs/80%。剩余约 5 µs 是首读前那次
  `pread` 的固有成本，**没有**任何变体削弱电源保障。详见
  [board_latency_20261021](../rknpu-vendor/experiments/results/board_latency_20261021/README.md)。
  对约 7 µs 的 SDPA task，这个量级仍意味着多数样本落在任务之后。
- 采集：[run_phase_sweep.sh](../rknpu-vendor/experiments/run_phase_sweep.sh)。每次运行
  `CAP=1` + `PHASE_NS=<φ>` + `MAX_READS=8`，相位顺序随机化，每 N 次插一次无采样基线，
  输出 `manifest.csv`（`run_id,request_phase_ns,csv,condition`）。脚本会清掉继承来的
  `EPERF_RKNPU_*` 采样配置，只允许 `0x1000/0x3000/0x4000/0x5000` 四个 vetted 寄存器；
  探测其他候选必须显式 `--raw-experimental`，产出的 `BLOCKWATCH_CONFIG` 会标
  `raw_mode=experimental`。
- 分析：[analyze_phase_sweep.py](../rknpu-vendor/experiments/analyze_phase_sweep.py)。
  按 condition 分组，用**实测相位**（`t_ns − submit_enter_ns`）而非请求相位作图，给出
  Wilson 区间。两种语义严格分开：
  - **status（路线 A）**：`p̂` 是"该 φ 上 busy 的概率"，**允许非单调**，不做 PAVA、
    不输出分位数；`0,1,1,0` 就原样保留为 `0,1,1,0`。
  - **marker（路线 B）**：只有显式给出 token 且通过单调性检验时才做加权（按 n）PAVA
    并反解分位数。单调性检验看**所有配对的最大累计下降**（不是相邻点差），
    显著性用 Wilson 区间是否分离判定；每相位样本数低于门槛时判 `insufficient_samples`
    而不是默认单调。分位数逐项处理：已被首个样本越过 → `left_censored`（`≤ X µs` 上界）；
    从未达到 → `right_censored`（`≥ X µs` 下界）；内部穿越 → `estimated`，同时给出
    插值所在的 **bracket**，并注明方法是"相邻 CDF 点线性插值"——**没有 `measured` 这一档**。
    若整个转变只被判为 `bracketed_only`（区间内没有样本），分位数状态降为 `bracketed`：
    结果是区间，插值数只是辅助值。
  分辨率判据按 condition 独立计算，且只看**跨越转变的那一对相邻样本**：
  `resolvable_ns = max(读延迟中位数, 相位抖动)`；`rise_ns`（p≤0.1 到 p≥0.9 的相位跨度）
  `≤ resolvable` 判 `unresolved`；括住转变的两点之间没有中间样本则判 `bracketed_only`
  并报出该局部间隔。全程未跨越时按窗口边界删失：**信号从未出现 → `right_censored`
  （转变在窗口上方，给出下界）**，**最早相位就已出现 → `left_censored`（转变在窗口下方，
  给出上界）**，中途未完成过渡 → `not_observed`。
- 未记录 `submit_enter_ns` 的 run 会被直接丢弃并在报告中列出，不做插值；
  `sample_validity` 不是 `ok/unchecked` 的样本同样丢弃并计数。

宿主机自测覆盖：已知转变恢复（±5 µs）、非单调 busy 曲线被原样保留、不等间隔按
**局部**间隔判定（0/1/101 µs → 100 µs，`bracketed_only`）、删失方向、删失分位数输出上界
而非 0、跨多点缓慢下降被识别、样本过少判 `insufficient_samples`、缺 submit_enter 直接报错，
以及经 shell 入口的端到端 sweep→manifest→CSV→分析器。

## 4. 路线 B：命令位置的持久标记 + 稀疏快照（核心可行性实验）

关键想法：每次只选一个待测命令位置 j，使那里发生一个之后仍可观察的状态变化。
在重复运行中改变 φ，观察“标记尚未到达/已到达”。这样无需在同一次运行中高速追踪所有命令。

优先 B0：利用原命令流已有、不会立刻被覆盖的普通配置值变化作为自然标记。
先证明读回对应哪个 bank，变化能区分前后状态，并且持续到采样窗口结束。
若后续任务覆盖同组寄存器，只能缩小观察窗口或把该候选淘汰，不能把最终值当作历史记录。
重复写相同值、自动翻 bank 或只能读执行组的配置都可能无法作为自然标记。

B1 才考虑插入人工标记。候选必须来自已知映射的普通可写配置寄存器，位于整次实验
都不启动/不依赖的模块，并已证明可通过 regcmd 写入及无副作用读回。
例如“完全不用 PPU 的合成矩阵实验中的某个 PPU 普通配置字段”只是筛选方向，
并非已确定寄存器；含 pooling 的 SDPA 不能直接沿用。若找不到候选，停止此路线。
不要使用保留位、地址洞、LUT access、指针、启动、IRQ clear/mask 或链控制字段作为 scratch。

验证顺序：

1. 在独立小探针中证明候选的读写及 reset 行为、bank 选择、允许位掩码；初始化两个组，
   只接受能够明确解释的读回。每次回放重置标记，防止上一 run 残留假阳性。
2. 已知可运行的完整任务内加入一个 `marker := token`，位于待测位置 j 前或后；
   同次运行此后不能重写该标记或使读回切到另一个组。先验收数值、guard、完成状态。
3. 加入标记需重建流长度、对齐、链地址、后继长度与终端 seal；验证实际提交的修改后流，
   同步 BO 到设备。用独立副本，结束后恢复并同步，避免污染 runtime 缓存的命令 BO。
4. 每次至多读取一次标记，扫描 φ，统计 token 出现的概率。
   在回放稳定、标记单调且窗口有效的条件下，该概率近似标记可见时刻的 CDF。
   输出 p10/p50/p90、读延迟和样本置信区间，不能称为某一次运行的逐命令 timestamp。
5. 在同一位置插入等长 OP_NONE 作为命令长度对照；原流、NOP 流、marker 流随机配对。
   同时检查后续任务/普通基线，识别持久状态污染。OP_NONE 也需先在该布局验收。

### 4.1 离线筛选 + 板端结果

**中断状态窗口 + 时间锚点（2026-10-21，`min_conv_exp --fp16` 与 `--iw 32`）**：
- 安全：412 次提交全部 BIT-EXACT，dmesg 错误计数采集前后均为 84（无新增）⇒ 读 `INTERRUPT_STATUS` 未干扰 IRQ 路径；
- `INTERRUPT_STATUS` 只出现 `0x0100`（DPU_0）或 `0x0200`（DPU_1），从不同时出现，
  且在作业大部分时间内保持置位 ⇒ 是**DPU 对应中断位的置位窗口**，不是块完成脉冲；
  **未置位不等于空闲，置位也未证明等于持续计算**（措辞已按评审修正）；
- 组号与**提交序号奇偶**严格对应：105 个置位样本无一例外，跨进程成立；
- 下降沿跟随 `hw_elapse`：单一阈值可分性 0.992（φ − hw_elapse, 阈值 4.1/7.0 µs）
  vs 0.727/0.762（只用 φ），且 ioctl 返回后置位比例 0（43 个返回后样本）；
- 上升沿（14/18 µs）**语义未知**，不可当作启动或完成；
- ABBA 区组配对（10 组）采样−不采样差中位数 −0.9 µs、95% 区间 [−13.4, +12.0] µs ⇒ **未检出扰动**；
  初版两两配对 −10.8 µs 系奇偶混杂，已撤回；
- 仍未给出任何 regcmd 位置；`submit→kick` 偏移未标定。

详见 [board_irq_20261021](../rknpu-vendor/experiments/results/board_irq_20261021/README.md)。

**submit→kick 上界（2026-10-21，1 µs 步进细扫描，每负载 384 run）**：
由 `job_end − kick = hw_elapse` 与"IRQ 在作业结束时清位"推得
`x = φ − hw` 的过渡点 `x* = kick_rel + irq_latency`，故 `submit→kick ≤ x*`。
实测 `x*` = **5.54 µs**（fp16，区间 4.08–6.42）与 **5.10 µs**（iw32，区间 4.81–6.42），
两负载一致。**这是上界而非测量值**，IRQ 延迟与采样延迟未分离。详见
[board_kick_20261021](../rknpu-vendor/experiments/results/board_kick_20261021/README.md)。

**两 task 负载已修对（2026-10-21）**：`min_chain` 原 `MISMATCH` 的根因是**链语义错误**
（task1 用"输入布局、ic_real=3"去读 task0 按"输出布局、oc_real=16"写的中间缓冲），不是模型算术。
新增 `--independent`（同输入、同权重、各自输出）后，判据改为不依赖模型的两条：
两个输出**逐字节相同**、两 task 相对模型的偏差**完全一致**。
`--tasks 1` 作负对照（task1 缓冲未写 → 判失败）。默认尺寸下 272/272 + 20/20 通过，
`max abs diff` 由 255 降到 4，与单 task 的模型偏差相同。

**单 task 内多快照（2026-10-21，`CAP=8`，请求 1 µs 周期）**：
一次提交内取到 **8 个快照**，相邻间隔 p50 **3.21 µs**（p90 8.75）、单次读取区间 p50 **1.17 µs**、
首个快照 p50 11.7 µs（p10 9.0）。**边沿可在同一次运行内被夹住**：上升 4/20、下降 5/20、
两边沿 3/20；`i15` 一次运行里同时夹住两个边沿（上升 ∈ (8.8, 13.7] µs，下降 ∈ (23.3, 26.5] µs）。
加大 `CAP` 到 32 后：23 个快照/次、间隔 p50 2.62 µs、覆盖跨度 p50 84.6 µs，
**8/12 次运行内完整解析出 `clear→set→clear`**，同一次运行内测得的窗口宽度下界 p50 7.9 µs。
限制因素是**覆盖跨度而非间隔**（`CAP=8` 时下降沿仅 16/72，`CAP=32` 时 8/12）。
另：**绑核到 CPU7 反而更差**（间隔 3.21→7.29 µs、2.62→6.12 µs），该项优化关闭。
边沿仍只是**区间**，且任务最开头约 9 µs 不可观测。详见
[board_intra_20261021](../rknpu-vendor/experiments/results/board_intra_20261021/README.md)。

**驱动侧事实**：`rknpu_job_subcore_commit_pc()` 里 `task_pp_en = (flags & PINGPONG)`，
整个 job **只写一次** `PC_TASK_CONTROL = ((0x6|task_pp_en) << 12) | task_number`，
驱动**不按 task 写 `S_POINTER`**，task 列表由硬件自己走 ⇒ 组选择只能是 job 级。
"job 开始锁存一次、计数值按 `task_number` 递增"的模型可解释全部既有观测，但**未获证实**。

**判别实验失败并暴露链的稳定性短板**：用 `[2,1,2,1,...]` 任务数序列区分模型时，
10 个 2-task job **8 个超时**（`task counter: 1`、~503 ms、soft reset），判别作废；
而 272 次**连续** 2-task 运行 0 超时 ⇒ 该链**只在单独连续运行时稳定**，
与另一负载交替时大概率卡在 "task0 完成、task1 未完成"。此后已停止在该模式下变换配置。

**多 task 组位不可用于区分 task**：1 task/提交时组位严格随提交交替（12/12），
2 task/提交时**恒为 DPU_1、0 次翻转**（相位扫描 272 run 中 DPU_0 与"两者同时"均为 0 次）
⇒ 组位按 **task** 前进（每次提交 2 步），不暴露 task 间切换。单 task 的奇偶结论不能外推。

**`min_conv_exp --chain-id` 不可用**：触发 6.2 s job 超时 + soft reset（task1 输出未写），
该模式停用。另：`min_chain --iw 32` 会顶到 64 KB 缓冲上限，**触发 500 ms 级 job 超时 + soft reset**
（本轮因此产生 3 次，dmesg 错误计数 18 → 27，均由 soft reset 恢复）。

**其余多 task 链的负面结果（未用于任何结论）**：`min_chain --tasks 2` 提交成功但自身校验
12/12 `MISMATCH`（`max abs diff 255`，非舍入）；`min_conv_exp --chain-id` 触发 6.2 s
job 超时 + soft reset，task1 输出未写。**"已验证的两 task 负载"目前不存在**，
多 task 实验未开展；板子在该次超时后处于重启后状态（重启是否由其引起未证实），
此后 `--chain-id` 停用。


**S_STATUS 首轮（2026-10-21，`min_conv_exp --pattern --fp16` 两个已验证负载）**：
- 状态字可读、取值稳定、跨 run 重复性好（177+ 样本仅 4 种取值）；
- 但两个计算量不同的负载**取值分布几乎相同**，`DPU_RDMA` 恒为 `0x0001000a`
  ⇒ 目前**没有证据**表明它能区分负载阶段；
- 解码本身未验证：`registers.xml` 里只有 `STATUS_0=[1:0]`、`STATUS_1=[17:16]`，
  实际变化的是 reserved 位 2/3；`status_busy()` 是从 SDPA 沿用的启发式，按它会判 RDMA 永远 busy；
- 采样相位仍 ≥17 µs，模块大概率只在开头忙的那段窗口采不到；
- **离线查证：没有任何可及实现解码 `S_STATUS`**——内核驱动全文无引用（只用每块完成中断
  `INTERRUPT_STATUS` 与 `job->int_mask` 比对，以及超时路径上的 `PC_TASK_STATUS`），
  mesa rocket 只有命令构造，vendor librocketnpu 只有寄存器名。`STATUS_0/STATUS_1` 的取值
  含义无定义，观察到的变化落在 RESERVED 位 2/3 上 ⇒ `status_busy()` 无法离线校验。
  **建议改用 `INTERRUPT_STATUS (0x0028)`/`RAW_STATUS (0x002c)`**：每块完成位有驱动级证据，
  给出的是"块完成事件"而非含义不明的 busy 标志。
详见 [board_status_20261021](../rknpu-vendor/experiments/results/board_status_20261021/README.md)。


**负载数值校验（2026-10-21 板端）**：`min_conv --pattern` 在 6 种配置下全部 `MISMATCH`，
`min_chain --tasks 2` 也 12/12 `MISMATCH`；只有 `min_conv_exp --pattern --fp16` 连续 5 次
`BIT-EXACT (fp16)`（`hw_elapse` 17.5–96.5 µs，单 task）。**后续 S_STATUS 实验必须建立在
通过自身校验的负载上**，`min_chain` 的两 task 结论在此之前只能算原始观测。

**B0 自然标记（2026-10-21，`min_conv` 单 task，`CNA_DATA_SIZE2 0x1028`）**：
写入值随 `iw` 变化（0x10/0x20/0x40/0x60），随机配对 26 次。
- 该寄存器**确实可读**、值随配置变化、跨 run 持久；
- 但**所有"值真的变了"的 trial 都读到旧值**，最极端的一次样本落在任务 **95%** 处仍读旧值，
  而新值要到**下一个 run** 才出现 ⇒ 读回**滞后一个 run**；
- 因此它**不能**标记"本 run 内某条命令已执行"。
- **双 task 同 submit 的补充实验**（`min_chain --tasks 2`，两个 task 写不同的
  `CNA_FEATURE_DATA_ADDR`）：`--tasks 1/2` 的 hw_elapse 中位数 18.8 / 48.9 µs 证明
  两个 task 都执行；而 12 个落在窗口内（最高 95%）的样本**全部**读到 task0 的配置，
  0 个读到 task1 的 ⇒ 滞后是**task 级**的，不是进程级。机理仍未证明。
- **该配置下未建立可解释的组信号**：双寄存器采样（`RAW_OFF2`）在一次 2-task submit 内
  同时读 `0x1070` 与 `0x1004`。`0x1004` 19 次全部读回 0——但本负载的命令流**并没有**
  写 CNA `S_POINTER`（`min_chain.c:255` 只发 DPU/DPU_RDMA 指针），CNA 指针由驱动写。
  所以这只说明"本配置下没有可用信号"，不判其永久不可行。
- 同批 19 条样本里，计数器并非恒为 `0xf000`：16 条 `0xf000`、3 条 `0xa001`（低 12 位 = 1），
  有效性命中 `ok` 2 条、`task_crossed` 1 条、`task_ambiguous` 16 条。
  **仅凭 2 条 `ok` 且计数器=1 的样本**可以说"task 1 运行期间读到的是 task0 的配置"，
  且必须附上"该负载未通过自身数值校验"这一限定。
- **撤回**两处推理：①"任务结束后应追上最后一笔写"——按拍/按组暴露的机制在没有后续 task 时
  不一定刷新，不能用它排除按拍滞后；②`frac = 相位/hw_elapse` 不能证明"已在任务之后"，
  分子分母没有共同起点。
- `min_chain --tasks 4` 的 job 超时是**探针构造问题**：该发射器只构造两个描述符
  （`memset(t, 0, 2*sizeof(*t))`），`--tasks 4` 只是改了提交数量。源码已改为直接拒绝该参数，
  不再作为链能力的负面证据。
详见 [board_b0_20261021](../rknpu-vendor/experiments/results/board_b0_20261021/README.md)。


**板端首轮（2026-10-21，`min_conv --dup-config` 单 task）**：
- 三臂链路通过：`applied=1`、`orig` 与计划逐字一致、`sync_apply=0`、`RESTORE ok=1`；
- `nop` 与 `marker` 两臂输出都与基线**逐字节相同**（长度保持替换对输出无影响）；
- **标记可见性未获证据**：PPU `0x6010` 的唯一样本在提交后 44.625 µs 且标
  `task_ambiguous`；DPU `0x40c4` 的 marker 臂**没有任何样本**（只有表头）。
  因此本轮**既不能判成功也不能判失败**，只能说人工标记落点**未验证**；
  `CNA 0x1040` 读到 `0xb1` 证明该地址可读，但同样 `task_ambiguous`，不能排除持久状态；
- 附带发现两个必须先解决的测量问题：采样相位比请求滞后 **50–80 µs**（对 ~7 µs 的 SDPA task
  无法落进窗口），以及合成任务计时**双峰**（13–97 µs，早期"标记快 4 倍"是预热伪影）。
详见 [board_marker_20261021](../rknpu-vendor/experiments/results/board_marker_20261021/README.md)。

下面的离线筛选结论仍然有效；**PPU 作为标记落点本轮未验证成功，也未证伪**。


工具：[screen_marker_candidates.py](../rknpu-vendor/experiments/screen_marker_candidates.py)，
结果：[marker_screen_20261021/FINDINGS.md](../rknpu-vendor/experiments/results/marker_screen_20261021/FINDINGS.md)。

- **B1 的落点是 PPU**：1026 个 task 的 `enable_mask` 只有 `0xd` 与 `0x18`，`GLOBAL.OPERATION_ENABLE`
  的 `PPU_OP_EN`(bit5)/`PPU_RDMA_OP_EN`(bit6) **从未置位**；op6 也没有 page `0x6/0x7` 的 regcmd 写。
  这是目前唯一"既不被写、也从不使能"的模块对。
- **读回 bank 全部未知（结论已修订）**：DPU/DPU_RDMA 每 task 写 `S_POINTER=0x0e`
  （`POINTER_PP_EN` 置位）⇒ ping-pong 生效。CNA/CORE 虽然在本 op 的捕获流里没有
  `S_POINTER` 写，但**驱动在提交时会自己写** `0x1004/0x3004 = 0xE + 0x10000000*core_index`
  （[OPERATOR-DELIVERY.md](../custom-ops/results/2026-10-09/OPERATOR-DELIVERY.md)、
  [wiki-hardware-20261014.md](../rknpu2-matmul-validation/results/wiki-hardware-20261014.md)），
  同样是 `0xE`，同样是 ping-pong。因此**不能**用"流里没有 `S_POINTER` 写"推断单 bank；
  在解决"主机读回命中哪个组"之前，B0 任何候选都没有可解释的读回语义。
- B0 候选排名（衡量值变化/写次数/写后窗口，不含 bank 优势）：
  `CNA_DATA_SIZE1(0x1024)`、`CNA_WEIGHT_SIZE1(0x1034)`、`DPU_DST_BASE_ADDR(0x4020)`。
  `CNA_DATA_SIZE1` 每 task 写一次、跨 task 值必变、位于 task 约 12% 处、写后仍余约 52 条；
  它只有 4 个取值循环，只能做"到了/没到"的二值判据，不能反推 task 序号。
- **注入方式改为长度保持的槽位替换**：不插入字（那会改长度/对齐/后继指针/seal），
  而是替换一条**已被重复写消融证明冗余**的 8 字节命令，三臂 original/nop/marker 长度
  完全相同。op6 的 DPU 流里有 2835 个这样的槽位（`plan_marker_slots.py`）。
  改写前 hook 会校验：槽位是普通配置写、确实冗余、标记寄存器在 PPU/PPU_RDMA 且非
  S_POINTER/OP_ENABLE/S_STATUS/中断/LUT、值非 0；任一条不满足就整条放弃并报原因，
  不改动流。submitted 后立即恢复并同步。
  **改写路径本身在宿主机上验过**：`--marker-bo` 手工构造 task BO + regcmd BO 并注册成
  mmap 记录，直接驱动真实的 `marker_apply()/marker_restore()`，断言改的是目标字、
  其它字一个都不动、拒绝时缓冲区逐字节不变、恢复后完全还原。
- **注入标记的命令字编码**已按现有流验证：`word = (op<<48) | (value<<16) | reg`，
  `op = target|1`（PC/CNA/CORE/DPU/DPU_RDMA = `0x0101/0x0201/0x0801/0x1001/0x2001`），
  因此 PPU 用 `0x4001`、`PPU_RDMA` 用 `0x8001`，`reg` 取 `0x6xxx/0x7xxx`。
- **bank 判据做成只读实验**：捕获流给出"每个 task 写了什么"，相位扫描读出"主机看到什么"，
  两者对齐就得到**读回滞后**（lag）。`follows_write`(lag 0) / `one_write_behind`(lag 1) /
  `alternating_or_position_dependent` / `unreadable_or_unknown` 四类结论由
  [analyze_bank_probe.py](../rknpu-vendor/experiments/analyze_bank_probe.py) 给出，
  采集入口是 [run_bank_probe.sh](../rknpu-vendor/experiments/run_bank_probe.sh)。
  它**不写设备**，因此不需要先解决副作用问题；lag 为非零常数时 B0 仍成立，只是期望值要整体前移。
- **尚未证明**：任何寄存器可读、写入无副作用、任何模块的读回 bank、op 低字节只有 `0x01`
  合法。这些只能上板验证，清单见 FINDINGS.md §3——其中 **bank 判据是新的第一步**，
  排在读写与副作用检查之前。

单标记可见只能约束“标记写到这个目标的时刻”。即使 PC 按顺序 issue，跨模块的写完成
也可能不同步；未证明排序前，不把它当作前一模块已完成。相邻位置的 p50 差值可以画
统计进度趋势，但不能当作每条命令 latency，更不能把不同运行的分位数相减当严格上下界。

先定位 task 内配置段、LUT 装载段、显式模块启动、链尾等少量关键位置。
成功后再增加位置密度；为每一条命令插标记会显著改变原流，第一版不做。
对同一 j，改变下游计算量和前后配置长度以检查标记是否反映 dispatch、bank 切换或完成门控。

## 5. 路线 C：硬件进度/trace 证据（条件探索）

只审核 TRM、本芯片驱动、寄存器表和厂商 profiler 的实际数据来源，建立候选白名单。
对已验证可读的 PC 字段做单次快照；绝不从名字推断 `BASE_ADDRESS` 会逐字自增。
通过“相同 task 数、不同 regcmd 长度”“相同长度、不同计算量”“链地址改变”判别
候选是静态配置、fetch 地址、dispatch 进度还是完成状态。取指预取位置不能直接等同于执行位置。

只有确认存在能记录 NPU 命令事件的 trace 源、路由和时间戳机制，才实现单次逐命令 trace。
CPU trace/ftrace、普通 IRQ 时间戳、DDR 总量计数均不能直接提供 NPU 内部命令时间。
厂商 profiler 若只输出 layer/task 时间，也不能补足这一级能力。
若没有这些证据，交付路线 A/B 的统计区间并明确单次 per-regcmd trace 不可用。

## 6. 不作为主方案的干预

- 任意位置拆成更多 task：会改变 ping-pong、等待、预取和启动/完成条件。
  现有 overlap 成功案例支持“可做机制探针”，不支持“任意配置前缀可独立完成”。
  仅在自写合成探针验证配置-only 分段和完整输出后研究；结果属于修改后的程序。
- prefix 耗时差分：末尾 drain、IRQ 和链尾改变会影响差值，不等于原链里该段的耗时。
- 在命令间插入 PPU/DPU 小任务向 RAM 写 token：消耗模块、DMA 和额外依赖，
  RAM 可见时间含缓存一致性延迟；它不是 PC 的通用 store/timestamp 指令。
- 给每条命令发 IRQ：目前没有该能力的证据；模块完成 IRQ 不是配置写完成 IRQ。
- 拉低 NPU 时钟放大事件：仅作机制验证，需另记频率；不能线性外推回原始时间线。

## 7. 数据契约和实施顺序

建议统一输出：

```text
run_id, job_cookie, core, task_abs, original_npuop_index, instrumented_index,
stream_hash, mode, event_kind, register_offset, raw_value, bank_or_unknown,
submit_enter_ns, requested_phase_ns, read_begin_ns, read_end_ns,
task_before, task_after, sample_validity, dropped_samples,
frequency_metadata, output_valid, driver_result
```

`event_kind` 至少分为 task_sample / module_state / natural_marker / injected_marker /
fetch_candidate；估计结果另表，含方法、样本数、分位数、区间及扰动量。
零值、漏采、跨 task 和跨 run 不能通过插值变成 measured；记录环溢出必须显式报错。

| 阶段 | 具体交付 | 放行标准 | 软件侧状态 |
|---|---|---|---|
| P0 | 完整 capture；统一限速/预算；PM 与生命周期；每次读取时间区间 | 捕获量与实际长度一致；关闭采样无变化；异常清理/并发隔离通过 | **未验收**：预算/限速/时间区间/截断自证已实现并自测；PM 与 job 生命周期仍缺（用户态只能隔离，不能保证），电源复查有检查后掉电竞态，时间原点仍是提交入口而非硬件 kick |
| P1 | 合成 task + SDPA 单 core 的相位快照采集/分析 | 所有输出和 guard 通过；无新增 timeout/fault；时间扰动可量化 | **未验收**：仅建立"部分已验证单 task 负载中，单次运行内可采到 `clear→set→clear` 序列"（`CAP=32` 时 8/12，边沿按读取区间包围，宽度下界 p50 7.0 µs）；并非"精确测得作业结束"；短 task 仍不可采；submit→kick 已由 **ftrace 实测**（13.2–33.0 µs，含 wrapper 开销），旧的条件估计模型作废 |
| P2 | 自然标记可行性；一个候选人工标记的独立实验 | 两 bank 行为可解释；跨 run 无残留；NOP 对照、不同输入和后续任务通过 | **未验收**：自然标记在本 run 写的新值上读到的仍是旧值（变化 trial 5/5），多 task 内 task1 配置未见；人工标记三臂改写与恢复全部通过、输出逐字节一致，但**未取得有效窗口内的可见证据**；bank 机制未定 |
| P3 | 代表 task 的稀疏命令位置 CDF，与模块状态对齐 | 有稳定转变分布；分辨率不足明确标记；不声称跨 run 同步并发 | **未到达**：上升沿语义**未识别**（6 个形状点 + 3 个频率点不足以排除混合机制），不适合当相位信号；中断位只能到作业粒度 |
| P4 | 白名单 live-PC / trace 审核 | 有本芯片事件语义证据才扩展单次 trace，否则记录负结果 | **未到达**：唯一可读 PC 页项为 `0x0028`/`0x002c`（只读中断状态），live-PC 与 trace 均未开 |

**ftrace 追加（2026-10-21，按评审二次收紧）**：RKNPU 为内建（`CONFIG_ROCKCHIP_RKNPU=y`），改驱动需整内核
重编换镜像，风险不可接受；但 ftrace 的 `trace_clock` 支持 **`mono_raw`**（与采样器同钟），
且 `__rknpu_submit_ioctl` / `rknpu_job_next.part.0` / `rknpu_core*_irq_handler` /
`rknpu_power_*` 均在 `available_filter_functions` 中 —— **不改内核即可取到驱动时间点**。
实测 6 次：`rknpu_job_next` 的**调度路径入口** p50 15.9 µs、IRQ handler 入口 124.1 µs
（原点=采样器 submit_enter）。**源码核对后收紧**：`job_next` 有两条早退，均不设
`hw_commit_time`、不写 `PC_OP_EN`，真正的提交在不可追踪的 static inline 里 ⇒
调度入口只是 **kick 的下界**，kick 本身仍未测量。边沿关系只能给**区间**
（上升沿−调度入口 2.8~15.2 µs），事件语义保留为**未识别**。
"上升沿≠kick""上升沿=kick""kick≤5.5 µs"三条**均已撤回**。
详见 [board_ftrace_20261021](../rknpu-vendor/experiments/results/board_ftrace_20261021/README.md)。

**收口结论（2026-10-21，最终口径）**

在本板现有内核、可用追踪点和已验证访问方式下，**用户态 MMIO ＋ ftrace
尚不能可靠恢复 `PC_OP_EN` 写入时刻、块级完成、执行 bank 或 regcmd 位置**；
继续探索的预期收益不足，**停止新增板端探测**。

这三者不都只是"差一个不可内联观测点"：改驱动或 kprobe 能补软件事件，
但 **bank、块级完成与 regcmd 进度仍需要可解释的硬件信号**，不能承诺换内核就能获得。

- **交付**：采样与追踪工具、原始数据、回归测试（11 套件）；限定负载下的中断状态窗口、
  驱动函数入口与 PM 调用序列；已撤回假设、失败模式与停用配置。
- **验收**：可验收为**一轮完成归档的可观测性探索**；**不可验收为目标能力已实现**。
- **per-regcmd timing 未建立，P0/P1 未通过。**
- **重开条件**：只在观测条件改变时（可修改的内核、可用的指令级探针、
  或经过确认的硬件 trace／厂商接口）。**不稳定多 task 链继续停用。**

详见 [测量模型 §7/§8](REGCMD-TIMING-MEASUREMENT-MODEL.md)。

扰动门槛先约定而非事后挑选：例如 end-to-end 中位数增加不超过 2%、p95 不超过 5%，
但必须结合基线方差及置信区间；不足以分辨时增加配对重复，不能直接判通过。
这只是建议验收预算，不是已测结论。标记方案超预算仍可作机制实验，但不能声称代表原程序时序。

软件验证应覆盖：流长度/链布局、缺失 task、跨边界样本、归零/reset、bank 歧义、
重复运行初始化、记录溢出、异常停止和 BO 恢复。其中采样器状态机部分（相位对齐、
预算、cap、每读区间、单寄存器探针、端到端分析）已由宿主机测试覆盖；流长度/链布局、
bank 歧义与 BO 恢复仍需上板。硬件验收从独立小任务开始，逐步加入
长 LUT、不同计算量、两 task 链、最后才是完整 SDPA。出现错误立即停止当前配置，
保存日志并在恢复后重新验证普通基线。硬锁不能依靠用户态 timeout 解除，需要实验板外部恢复能力。

第一阶段成功定义：获得可信的 task 内模块活动分布和少量命令位置的统计可见时间，
同时量化分辨率与扰动。单次原始命令流的全部逐命令精确时间戳仍是条件目标。
