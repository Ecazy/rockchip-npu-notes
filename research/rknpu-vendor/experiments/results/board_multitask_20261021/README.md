# 多 task 链：先把负载修对，再看组位（2026-10-21，板端）

## 1. 修 `min_chain`：根因不是"模型算错"

原 `min_chain --tasks 2` 的校验恒为 `MISMATCH`（`mismatch=623/1024`，`max abs diff 255`）。
逐行核对后定位到**链本身的语义错误**，不是 CPU 模型的算术问题：

```c
c1 = c0;                                  /* 两个 task 配置完全相同 */
c1.in_addr = (uint32_t)mid.dma_addr;      /* task1 把 task0 的输出当输入 */
```

而 `emit_conv` 对两个 task 用的是同一套 `c`：**输入 ic_real=3**。task0 却按**输出布局**
（oc_real=16 通道、组内 tile 排布）写 `mid`。也就是说 task1 用"输入布局 + 3 通道"去读
一个"输出布局 + 16 通道"的缓冲——布局不匹配。改模型救不了，必须改流。

**改法（`--independent`）**：让两个 task 各自自洽——同输入、同权重、不同输出缓冲：

```c
if (g_independent) { c1.in_addr = c0.in_addr; c1.wt_addr = c0.wt_addr; }
c1.out_addr = act + OUT_OFF;              /* 与 task0 的 mid 不同 */
```

**判据也换成不依赖 CPU 模型的**：参考只算一次，要求
1. 两个输出缓冲**逐字节相同**（`PAIR IDENTICAL`）；
2. 两个 task 相对模型的偏差**完全一样**（`bad0 == bad1`）。

### 结果（含负对照）

| 调用 | task0(mid) | task1(act+32768) | PAIR | 判据 |
|---|---|---|---|---|
| `--tasks 1 --independent` | mismatch=455, max diff 4 | **mismatch=1024/1024**（缓冲未被写） | **DIFFER** | MISMATCH |
| `--tasks 2 --independent` | mismatch=455, max diff 4 | mismatch=455, max diff 4 | **IDENTICAL** | **PAIR BIT-EXACT** |

- `max abs diff` 从 255 降到 **4**，且单 task 与双 task 的模型偏差完全相同 ⇒ 链**没有引入**额外误差；
- **`--tasks 1` 是负对照**：task1 缓冲没被写，判据立刻失败 ⇒ 这条判据确实能检出"少执行了一个 task"；
- 稳定性：`--tasks 2 --independent` **20/20**、扫描中 **272/272** 全部 `PAIR BIT-EXACT`，dmesg 无新增错误。

**这就是"已验证的两 task 数值正确负载"。**

已知使用约束：`min_chain --iw 32` 会顶到 `act` 缓冲 64 KB 上限（输出 @+32768 + 32 KB + 输入），
**不只是返回 0**——它会让 job 挂起到驱动超时：dmesg 出现
`failed to wait job ... elapsed time: 520–530 ms` + `job timeout` + `soft reset`，
每次失败 3 行日志。本轮的测量里因此产生了 **3 次这样的超时**（dmesg 错误计数 18 → 27），
每次都由 soft reset 自行恢复，未影响后续实验。默认尺寸才是验证过的用法。

## 2. 两 task 提交下的组位：**不随 task 切换**

同样的 1 µs 问题用两种提交数量对比（相位 15 µs，各 12 次，同一台板子连续跑）：

| 提交方式 | 组位序列 | 相邻翻转 |
|---|---|---|
| 每次 1 个 task（`min_conv_exp --fp16`） | `D0 D1 D0 D1 D0 D1 D0 D1 D0 D1 D0 D1` | **11 次**（严格交替） |
| 每次 2 个 task（`min_chain --tasks 2 --independent`） | `none D1 D1 D1 D1 D1 D1 D1 none none none none` | **0 次** |

- 1 task/提交：组位**每次提交翻转**（12/12，与之前 105/105 的奇偶结论一致）；
- 2 task/提交：组位**恒定 DPU_1**，从不出现 DPU_0、也从不交替 ⇒ 每次提交组位前进**偶数**步
  （即**按 task** 前进，2 个 task 前进 2 步）；
- 序列里的 `none` 是该次提交在 15 µs 前就已结束（快模式 ~15–20 µs），无信息，不计入。

**结论（限定）**：在当前配置下，`INTERRUPT_STATUS` 的 DPU 组位**不能**用来区分
"一次提交里的第 0 个 task"和"第 1 个 task"——它不暴露 task 间的组切换。
单 task 时的"与提交奇偶严格对应"**不能**外推到多 task。

## 3. 相位扫描（272 run）的观察

两 task 提交的相位扫描里：`0x0200`（DPU_1）107 次、`0x0000` 131 次，
**DPU_0 一次也没有、两者同时出现也没有**，与上面的交替实验一致。
34 个 run 无样本（请求相位晚于 ioctl 返回）。

## 4. 没有证明的

- 组位为何在 2-task 提交里恒为 1（是"最后一个 task 的组"还是驱动只program 一次）未确定；
- 该行为在 3 个以上 task、或不同 `subcore_task` 形状下是否成立，未测；
- 仍然不涉及任何 regcmd 位置或 per-regcmd 时长。

## 5. 顺带记录：`--chain-id` 的路（不可用）

`min_conv_exp --chain-id` 会触发 **6.2 s job 超时 + soft reset**（task1 输出 `UNTOUCHED`），
之后板子处于重启后状态。该模式已停用，本目录的实验**全部**使用 `min_chain --independent`。

## 6. 驱动侧事实（离线查证）

`rknpu098/rknpu_job.c` 的 `rknpu_job_subcore_commit_pc()`：

```c
int task_pp_en = args->flags & RKNPU_JOB_PINGPONG ? 1 : 0;   /* 我们的提交都带 PINGPONG */
REG_WRITE(first_task->regcmd_addr, RKNPU_OFFSET_PC_DATA_ADDR);
REG_WRITE(((0x6 | task_pp_en) << pc_task_number_bits) | task_number,
          RKNPU_OFFSET_PC_TASK_CONTROL);      /* 整个 job 只写一次 */
REG_WRITE(0x1, RKNPU_OFFSET_PC_OP_EN);
REG_WRITE(0x0, RKNPU_OFFSET_PC_OP_EN);
```

- **驱动不按 task 写 `S_POINTER`**：整个 job 只写一次 `PC_TASK_CONTROL`，其中带
  `task_number`；task 列表由硬件自己走，`pc_task_number_bits = 12`（RK3588）。
- 因此"组位随提交翻转"不可能是驱动按 task 写指针造成的，只能来自硬件内部计数。

**由此得到一个能解释全部既有观测的模型（尚未被实验证实）**：组选择在**每个 job 开始时锁存一次**，
取自一个按 `task_number` 递增的全局 task 计数 ⇒
1 task/job 时每次 +1、锁存值逐 job 交替；2 task/job 时每次 +2、锁存值恒定；job 内部不变。
这同时解释了"1 task 严格交替（12/12）"和"2 task 恒为 DPU_1、0 次翻转"。

## 7. 该模型的判别实验**失败**，并暴露链的稳定性短板

用任务数序列 `[2,1,2,1,2,1]×3` 去区分"锁存模型"与"按提交交替模型"（两者预测几乎每个 job 都不同）。
结果：

- 10 个 2-task job 里 **8 个超时**（dmesg `task counter: 1`, `elapsed 503–505 ms`, `flags 0x5`，
  随后 `soft reset`），因此**没有任何 2-task 样本**；判别实验作废；
- 8 个 1-task job 都有样本，但全部显示组位 0——它们夹在超时 job 之间，硬件状态已被 soft reset 重置，
  所以这一序列不可解释。

**对比很关键**：

| 场景 | 2-task job 的超时率 |
|---|---|
| 272 次**连续** 2-task（本目录 §3 扫描） | **0 / 272** |
| 2-task 与 1-task **交替**（本节，10 次） | **8 / 10** |

⇒ 这个 2-task 链**只在单独连续运行时稳定**，与另一个工作负载交替时会大概率卡在
"task0 完成、task1 未完成"。

**后续更正（同日收尾时）**：即使是单独运行，这条链**也不是可靠的**——收尾复核时
`./min_chain --tasks 2 --independent` 单独跑一次就超时了
（dmesg `task counter: 1` / `elapsed 526 ms` / `soft reset`，错误计数 51 → 54），
随后 `min_conv_exp --fp16` 仍 `BIT-EXACT`、板子自行恢复。
因此 §1 的"272/272 + 20/20 通过"应理解为**那一次采集窗口内的结果**，
不能推广成"这条链稳定可用"。结论：
- 之前 §2 的组位结论（2 task 恒 DPU_1）只在那 272 次连续运行里成立；
- **任何依赖 2-task 链的新配置都不能视为安全**，包括"再跑一次确认"这种用法；
  本节实验后已停止使用该链。

## 8. 结论

- 组位在**单 task** 场景下可信（与提交奇偶 105/105）；
- 在**多 task** 场景下它不随 task 切换（272 次连续运行里 DPU_0 出现 0 次）；
- 驱动侧说明组选择是 **job 级、写一次**，锁存模型可解释全部观测但**未获证实**——
  判别实验被链的超时问题阻断。
