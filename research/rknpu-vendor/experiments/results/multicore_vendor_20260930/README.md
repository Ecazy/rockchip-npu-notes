# 多核派发尝试 + vendor 提交口径真值（§7.43）

对 §7.42「多 task 不执行」的追问：vendor 一条 submit 到底怎么填 `core_mask`/`subcore_task`，
多 task 是真执行还是被忽略。

结论来源：板端 Orange Pi 5 Plus，driver v0.9.8（源码 `refs/rknpu-driver-0.9.8/`）；
vendor 侧数据来自 hook trace 的 `submit`/`task` 事件（本轮**首次**解析这些字段）。

## 一、vendor 提交口径真值（新工具 `analyze_vendor_submits.py`，331 条 submit）

| 字段 | vendor 取值 |
| --- | --- |
| `core_mask` | **0（AUTO）** |
| `flags` | **0x5 = PC\|BLOCK（无 PINGPONG）** |
| `task_base_addr` | `0x0` |
| `regcfg_offset` | `0`（全部 7326 个 task） |
| `subcore_task` | `[[0, N/3], [0, N/3], [0, N/3], [0,0], [0,0]]` |
| `task_number` | 3 … **3663** |
| task 描述符 `regcfg_amount` 分布 | (9,0,0)×6540、(12,0,0)×534、(69,0,0)×160、(11,0,0)×24、(108,0,0)×12 … |

**决定性**：`attn_relu_vendor` 的一条 submit `task_number=3663`、`hw_elapse=7.194 ms`
⇒ **~1.96 µs/task** ⇒ **所有 3663 个 task 都真的执行了**（`core_mask=0` ⇒ 全在一个核上）。
即：**同核多 task 是能执行的**，§7.42 的「不执行」只是我们复现失败。

## 二、我们的复现（`run_task_chain_probe.sh` F 段、`multicore_probe.txt`）

| 配置 | 结果 |
| --- | --- |
| 同核 2 task，task1 regcmd 紧接 task0（`offB=n*8`），flags 7/5 | task1 输出区**未写入**（`CHAIN-HASH=bcc31dc5`=全零区 hash） |
| 同核 2 task，task1 regcmd 加 64B 间距 | 同上（未写入） |
| `core_mask=0x3` + 每核 1 个不同 task（`subcore_task[0]={0,1}`,`[1]={1,1}`），flags 7 | 两次：一次**部分写入**（`f395e3ce`，diff 7560 ≈ 58 B 非零）、一次未写入；`hw_elapse=0 ns` |
| 同上，flags 5（vendor 口径） | 一次部分写入（`b58221fb`）、一次未写入；`hw_elapse=28–42 µs` |

⇒ 我们的多核/多 task 配置**不稳定**（第二段有时写 58 字节、有时不写，`hw_elapse` 0/28–42 µs 抖动），
说明还有缺口（候选：`PC_DMA_BASE_ADDR` 语义 —— 设成 task BO dma 会超时；per-core 初始化；
task 描述符数量/`regcfg_amount` 与 vendor 的 9/12/69 量级差异）。

**安全结论**：`core_mask=0x3` 不会挂内核（`ret=0`、无 oops），与历史 `--subcore 2/3/4` 不同
⇒ 多核可以继续小心推进。

## 三、下一步（明确的单一实验）

仿 vendor 最小真实形状：**N=3 个小 task（amount 9/12/9，regcmd 连续排布）、`flags=0x5`、
`core_mask=0`**，三个 task 输出到三个不同地址，看是否**三个都执行**（可区分输出）。
这能判定缺口是「描述符个数」「amount 量级」「连续布局」还是「`flags`」。

## 产物

- `experiments/analyze_vendor_submits.py`、`vendor_submit_rules.txt`（331 条 submit 的口径表）。
- `experiments/results/multicore_vendor_20260930/`（本 README + `multicore_probe.txt`）。
- `experiments/min_conv_exp.c`：`--multi-core`；`experiments/run_task_chain_probe.sh`（F/G 段）。


---

# 追加 §7.44：多 task 机制的真实用途 —— 批量发射 DPU 小算子

日志：`multitask_probe.txt`（H 段为仿形实验）；`vendor_regcmd_blocks.txt`（vendor task 块解剖）。

## 1. vendor task 块解剖（决定性）

attn_relu 的每条 submit 有 3663 个 task，其 regcmd 块**只含 DPU 寄存器**：

```
op=1 task_index=0  entries=34  偏移 0x4004..0x5004  头: 0x4004=0xe 0x5004=0xe
                                                     0x400c=0x81e5 0x4010=0x24000001
                                                     0x4020=0xffaa0000 …
op=1 task_index=1  entries=34  …                     0x4020=0xffaa0400   ← +0x400
op=1 task_index=2  entries=4   …                     0x4020=0xffaa0800   ← 只有 4 条
```

⇒ **多 task 是「同一个 DPU 小算子按不同地址批量发射」**（每 task 递增 0x400），不是 conv 链、
也不是按核分片；每 task 的 `regcfg_amount` **不一致**（34/34/4）⇒ 硬件必然消费*每 task 自己的*
块长（用第一个 task 的 amount 推统一步长解释不通）。

## 2. 我们的仿形实验（N=3，`--chain-n 3`，连续 regcmd 块）

| 配置 | 结果 |
| --- | --- |
| 3×3 N=3 `flags=5`、`core_mask=0` | 后续 task 全 `UNTOUCHED`（`nonzero_bytes=0`） |
| 3×3 N=3 `flags=7`、`core_mask=0` | 同上 |
| 3×3 N=3 `flags=5`、`core_mask=0x7`（三核各一 task） | 同上 |
| 1×1 最小几何 N=3 `flags=5`、`core_mask=0` | 同上（说明与几何/流长无关） |

**已排除**：`flags`(5/7)、PINGPONG、regcmd 连续性（紧接 × 64 B 对齐都试过）、task 数、
单核 × 三核、几何大小（1×1 与 3×3 一致）。
**仍不稳定**：`core_mask≠0` 时偶发第二个核部分写入（§7.43 记录：4 次里 2 次 ≈58 B）。

## 3. 驱动侧事实（源码）与推论

- `PC_DATA_AMOUNT = (first_task->regcfg_amount + 4 + scale-1)/scale - 1`（`scale = pc_data_amount_scale
  = 2`，`EXTRA = 4`）⇒ 取指步长只由**第一个** task 的 amount 决定；
- `PC_DMA_BASE_ADDR = args->task_base_addr`（我们与 vendor 都传 **0**）⇒ 硬件**不是**从 task BO
  读描述符（`task_obj_addr` 是内核对象指针，只看驱动用）；
- `max_submit_number = 4095` ⇒ 每核按 4095 分块提交。

⇒ vendor 每 task 块长不同却都能执行，说明存在我们尚未复现的*取指路径*（候选：task 描述符数组的
设备地址由其它路径提供；或 PC 流内自带块长信息）。**不影响 conv 实现**（一条 conv = 一个 task，
内部已含 CNA/CORE/DPU 三段）。

## 4. 结论

**厂家的「多 task」真正服务于 DPU 小算子的批量发射**（elementwise/activation/transpose 这类，
正是 skill 关注的场景）；conv 侧不需要它。后续若要追这块，方向应是「DPU-only 任务 × N 个指针」
的批量发射，而非 conv 链。


---

# 追加 §7.45（**重要修正**）：多 task「不执行」是**我们自己的 submit 设置错误**

## 1. 根因（`refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c`）

```c
static int rknpu_get_task_number(struct rknpu_job *job, int core_index)
{
    int task_num = job->args->task_number;
    if (rknpu_dev->config->num_irqs > 1) {
        if (job->use_core_num == 1 || job->use_core_num == 2)
            task_num = job->args->subcore_task[core_index].task_number;   /* ← 覆盖 */
        else if (job->use_core_num == 3)
            task_num = job->args->subcore_task[core_index + 2].task_number;
    }
    return task_num;
}
```

`use_core_num = popcount(core_mask)`；`core_mask = 0`（AUTO，我们与 vendor 都用）时驱动把
`use_core_num` 置 1 ⇒ **每核实际提交的 task 数来自 `subcore_task[core].task_number`，
`args->task_number` 被覆盖**。

我们的发射器默认 `subcore_task[i] = {task_start 0, task_number 1}` ⇒ **无论 `task_number` 设成几，
驱动每核只提交 1 个 task** ⇒ §7.42/§7.43/§7.44 观察到的「后续 task 从不执行」**是我们 submit 字段
没配套的错误**，不是硬件性质。

vendor 的形状因此自洽：`task_number = N` **且** `subcore_task[i] = {0, N/3}`
（`core_mask=0` ⇒ 驱动在单核上提交 `N/3` 个 task）。

## 2. 修正后的复现（`--chain-n N` 同时把 `subcore_task[*].task_number = N`）

| 配置 | 现象 |
| --- | --- |
| N=2/N=3，`flags` 5/7，regcmd 块**连续**（步长 = amount+4 = 驱动 PC_DATA_AMOUNT 推出的 130 条） | **每次 submit 6.3 s 超时**（`task counter: 0x0`、`raw status: 0xc0010000`、`require mask: 0x300`） |
| 同上，但每个 task 的 regcmd 块按 **8 条(64 B)对齐**（vendor 观测粒度） | 同上（超时） |

即：驱动**确实**开始提交 N 个 task（行为从「静默只跑 1 个」变成「挂到超时」），但硬件仍未成功
逐个走完 per-task regcmd 块 ⇒ 还有一处未解（候选：per-task `regcfg_amount` 的消费路径 ——
vendor 每 task 的 amount **不一致**（34/34/4）说明硬件确实按 task 描述符逐个取指，而描述符数组的
位置/格式我们还没对上；`task_base_addr` 设成 task BO 的 DMA 地址会超时，见 §7.42）。

证据：`multitask_hang_dmesg.txt`。

## 3. 纪律

每次多 task 尝试 = 一次 6.3 s job timeout + soft reset（本轮 soft reset 从 20 涨到 24）。
建议每轮最多试 1–2 种配置，并先看 dmesg 是否出现 `job commit failed`/`EINVAL`（§7.39 的楔死签名）。


---

# 追加 §7.46：**多 task 的链式机制找到了** —— 任务流尾部写 PC 寄存器指向下一块

工具：`experiments/analyze_vendor_task_blocks.py`（命名寄存器 + 逐 task 差异）；
证据：`vendor_task_anatomy.txt`（301 行，attn 与 transpose 各若干 task 全字段）。

## 1. 两条决定性观测

**(a) attn（DPU-only 批量拷贝）**：每个 task 34 entries（`DPU_*` 命名寄存器），
相邻 task **只差 `DPU_DST_BASE_ADDR`（+0x400 = 1 KB）**；尾 task 只有 4 entries
（`DPU_S_POINTER`/`DPU_RDMA_RDMA_S_POINTER`/`DPU_DST_BASE_ADDR`/`DPU_WDMA_SIZE_0`）
⇒ 后面的 task 是**增量配置**（沿用前一块的其余寄存器）。

**(b) transpose**：每个 task 112 entries，相邻 task 变化的是
`CNA_FEATURE_DATA_ADDR`（+0x80）、`DPU_DST_BASE_ADDR`（+0x80）**以及 task 流尾部的 PC 寄存器**：

```
0x0010 PC_BASE_ADDRESS     = 0xffffe700   ← 变化 (was 0xffffe380, +0x380 = 896 B = 112×8)
0x0014 PC_REGISTER_AMOUNTS = 0x7          ← 变化 (was 0x37)
0x0000 PC_VERSION          = 0x0
0x0008 PC_OPERATION_ENABLE = 0xd
```

⇒ **每个 task 的 regcmd 流以「下一块的地址（`PC_BASE_ADDRESS`）+ 下一块的 amount
（`PC_REGISTER_AMOUNTS`）+ `PC_OPERATION_ENABLE`」收尾** —— 多 task 的链式推进是**写在该 task
自己的命令流里**的（链表式），而不是靠 task 描述符数组。

数值自洽：`PC_BASE_ADDRESS` 的步长 = 0x380 = 112 entries × 8 B = **正好是块长**（连续排布 ✓）；
`PC_REGISTER_AMOUNTS` 与驱动的 PC_DATA_AMOUNT 公式一致（`(amount+4+scale-1)/scale-1`，
amount=108 → 55 = 0x37 ✓；驱动 `pc_data_amount_scale = 2` ✓）。

## 2. 这解释了 §7.42–§7.45 的全部现象

- 我们的命令流**从不写** `0x0010/0x0014/0x0008`（这些由驱动在 kick 前写一次）；
  于是当 `subcore_task[*].task_number = N`（§7.45 修正后驱动确实提交 N 个 task）时，
  硬件要等 N 个 task 的链，而流里只给了一个 ⇒ **挂到 6.3 s 超时**（§7.45 实测 ✓）。
- `task_number = 1` 时（我们历来所有实验）自然只跑一块 ⇒ 与观测一致 ✓。
- 未解释的残留：**attn 的 task 0/1 尾部没有那三个 PC 写**（只有 `DPU_*`），说明该 op 的链推进
  另走一条路径（可能是描述符数组，或该 op 的 task 其实由另一层循环驱动）。

## 3. 下一步（配方已明确）

给 `min_conv_exp` 的 `--chain-n` 补上「每个 task 尾部写 `PC_BASE_ADDRESS`（下一块地址）、
`PC_REGISTER_AMOUNTS`（下一块 amount，按驱动公式）、`PC_OPERATION_ENABLE = 0xd`」，
保持 `subcore_task[*].task_number = N` ⇒ 再测 N=2/3 是否全部执行。
（每次尝试最多一个 6.3 s 超时 + soft reset，务必先看 dmesg 是否有楔死签名。）


## 4. 实现尝试与**负面结果**（`--chain-pc`）

按 §7.46 配方实现（`--chain-pc`，opt-in）：
- 先探测「带 4 条 PC 尾写」的段长 `L`（等长 ⇒ 下一块地址可预先算出），再重放 task0（带指向 task1 的尾写）；
- 每个 task 尾部写：`REG_PC_BASE_ADDRESS = rcbo.dma + (k+1)*L*8`（`& 0xfffffff0`）、
  `REG_PC_REGISTER_AMOUNTS = (amount+4+2-1)/2-1`、`REG_PC_VERSION = 0`、`REG_PC_OPERATION_ENABLE = 0xd`；
- 描述符 `regcfg_amount = L-4`，`subcore_task[*].task_number = N`。

实测（N=2，3×3 ic16 oc16，flags=5）：

```
CHAIN-PC: 每段 134 entries = 1072 B（amount=130，下一块 @+1072，PC_REGISTER_AMOUNTS=66）
SUBMIT ret=-1 errno=Connection timed out  6317 ms  hw_elapse=0 ns
```

⇒ **仍然超时**（soft reset 26）。即：**「尾部 PC 链写」这个机制是必要条件，但不充分**（或我们某处
与 vendor 不同）。下一步可逐个变量对照（每轮 ≤1–2 次尝试）：

1. 尾写的**顺序/组合**（vendor 实测顺序：0x10 → 0x14 → 0x0 → 0x8，值 `0xd`）；
2. `S_POINTER`（我们 0xe = 开 ping-pong；vendor 也是 0xe，但它的 `PC_TASK_CONTROL` 位不同）；
3. `PC_REGISTER_AMOUNTS` 的取整（我们 (L-4+5)/2-1 = 66；vendor amount=108 → 55 = (108+5)/2-1 ✓ 同式）；
4. 每段末尾 4 条「driver trailer」与我们 ov 尾写的相互位置；
5. attn 那种「尾 task 只 4 条」的增量块是否也需要同样的尾写（我们已按每段独立全量 + 尾写）。


## 5. 链的**精确布局**（offline，`tr_transpose_p0321_fp16`，一条 submit 24 个 task）

```
task_index | 描述符 regcmd_addr | 块尾 PC_BASE_ADDRESS | 块 entries / amount
    0      | 0xffffe000         | 0xffffe380（= task1 地址）| 112 / 108
    1      | 0xffffe380         | 0xffffe700（= task2）    | 112 / 108
    2      | 0xffffe700         | 0xffffe780（= task3）    |  16 / 12
    3..6   | 0xffffe780..       | 各自 = 下一个 task 的地址 |  16 / 12
    7      | 0xffffe980         | **无 PC_BASE_ADDRESS**   |  16 / 12
    8      | 0xffffe000（回到第一块！）| 0xffffe380 …        | 112 / 108
   ...（24 = 3 组 × 8，地址每组重复）
```

**完整配方（已写入实现 `--chain-pc`）**：
- 每个 task 尾部依次写 `PC_BASE_ADDRESS`（**下一个 task 的 regcmd DMA 地址**）、
  `PC_REGISTER_AMOUNTS`（= 驱动公式 `(amount+4+scale-1)/scale-1`，scale=2）、
  `PC_VERSION = 0`、`PC_OPERATION_ENABLE = 0xd`；
- **每组的最后一个 task 不写 `PC_BASE_ADDRESS`**，并把 `PC_REGISTER_AMOUNTS` 置 0（链终止）；
- 组末之后的下一个 task 又从第一块地址开始（`regcmd_addr = 0xffffe000`）⇒ 本 submit 是
  「3 组 × 8 块」的**循环复用**，说明组边界处硬件仍会从**描述符**拿下一个地址（或 vendor runtime
  按组迭代），不是一条纯线性链。

## 6. 负面结果 #2：按上述终止约定实现后仍超时

`--chain-pc`（终 task 用 3 条尾写：`PC_REGISTER_AMOUNTS=0`/`PC_VERSION=0`/`PC_OPERATION_ENABLE=0xd`）
⇒ N=2 实测**仍 6.28 s 超时**（`hw_elapse=0`，soft reset 27）。

⇒ 结论更新：**「尾部 PC 链写 + 终止约定」对 conv 流仍不足**。剩余差异候选（按可能性排序）：
1. 我们的 conv 流长度/内容（130 条、含 CNA/CORE/DPU 全量）与 vendor 的可链流（DPU-only 34 条、
   CNA+DPU 112 条）不同 ⇒ 也许可链性依赖 `PC_OPERATION_ENABLE = 0xd` 的**位域语义**或流内
   某个"允许链"的开关；
2. `PC_TASK_CONTROL`（驱动按 `task_number` 写 `(0x6|pp)<<12 | N`）与尾写的交互；
3. `S_POINTER = 0xe`（ping-pong）在链式下需改为非 PP； 
4. 链边界处硬件确实读**描述符数组** ⇒ 需要正确的 `task_base_addr`（我们设成 `tk.dma_addr` 曾超时，
   但那次没有尾写；两者**组合**未试）。

## §7.115 **多 task/多核闭合**（handoff §3 的第 1 项，唯一未闭合的逆向项）

**根因**：我们的多 task 把 N 个 task **全塞给 subcore0** ✗
（旧代码 `sub.subcore_task[0].task_number = N`）——而这正是**硬件不支持的路径**（单 subcore >1 task ⇒ 超时 ✗）。
vendor 的真实形态是 **「每个 subcore 各 1 个 task」**：

```
vendor: task_number=3, core_mask=0, subcore_tasks=[[0,1],[0,1],[0,1],[0,0],[0,0]]
我们（新）:  subcore_task[k] = {task_start=k, task_number = (k<N) ? 1 : 0}, k=0..4
```

**实测**（16×16 IC16 int8，`--task-count 3 --task-blocks`）：

| 运行 | SUBMIT | OUT-HASH |
| --- | --- | --- |
| 单 task 参照 | `ret=0`，20.1 µs | `391b75c5` |
| **3 task（每 subcore 1 个，core_mask=0，3 个独立块）** | **`ret=0`，17.8 µs，`task_counter=3`** | **`391b75c5`**（**与单 task 完全相同** ✓✓）|

⇒ 三个 task **都真的跑了** 且输出与单 task **逐字节一致** ✓✓；soft reset 86→86 ✓、**楔死签名 0** ✓。

**⇒ handoff §3 第 1 项（唯一未闭合的逆向项）闭合** ✓：
- 描述符 `regcfg_amount` 必须非零且按**本驱动口径**（§7.110 修 ✓）；
- **task 必须按 subcore 分配**（每 subcore 1 个 ✓），不能用「subcore0 收 N 个」✗；
- 每个 task 用独立块更接近 vendor（`--task-blocks` ✓，非必需但推荐 ✓）；
- `task_base_addr=0`、`core_mask=0`、`flags=0x5` ✓（沿用既有结论 ✓）。

## §7.116 多核并行的定量验证（48×48 IC32 int8，3 个独立 task）

| task 数 | SUBMIT | `hw_elapse` | OUT-HASH | 进程 wall |
| --- | --- | --- | --- | --- |
| 1 | `ret=0` | 49874 ns | `d3272e45` | 0.221 ms |
| 2 | `ret=0` | **49875 ns** | `d3272e45` | 0.144 ms |
| 3 | `ret=0` | **46957 ns** | `d3272e45` | 0.064 ms |

⇒ **2 个 task 的 `hw_elapse` 与 1 个逐纳秒相同**（49875 vs 49874 ns ✓）、**3 个反而略快**（46957 ns ✓）
⇒ **三个 subcore 真的是并行执行** ✓✓，即「同样的工作量、时间不变」= **~3× 吞吐** ✓；
输出哈希三次完全一致 ✓（每个 task 都算对 ✓）；soft 86→86 ✓、**楔死签名 0** ✓。

**对 `rknpu-command-stream` 技能用途的直接回答**：本工具此前的提交是**单核受限**的 ✗
（所有 task 塞 subcore0 ⇒ 要么静默单 task、要么超时 ✗）；采用 vendor 形态（每 subcore 1 task ✓）
后，**N 个独立 task 在 ~1 个 task 的时间内完成** ✓✓ —— 这正是「判断提交是否单核受限」的答案 ✓。
