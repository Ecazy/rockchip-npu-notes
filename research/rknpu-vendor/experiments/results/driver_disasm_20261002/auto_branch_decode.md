# 驱动 0.9.8 AUTO 分支 / subcore_task 槽位语义定论（源码优先，§7.171）

> 来源：`refs/rknpu-driver-0.9.8/drivers/rknpu/rknpu_job.c`（行号按本仓库副本）；
> 配置：`rknpu_drv.c` L187–196（RK3588）；ABI：`drivers/rknpu/include/rknpu_ioctl.h` L222–280。
> 反汇编（`rknpu_disasm.txt`）仅作交叉验证（上一会话已确认板端二进制与官方源码语义一致）。

## 1. RK3588 配置事实（rknpu_drv.c L187–196）

| config | 值 | 影响 |
| --- | --- | --- |
| `num_irqs` | 3 | `num_irqs>1` ⇒ 全部走 `subcore_task[]` 槽位路径 |
| `max_submit_number` | 4095（`(1<<12)-1`） | 单次 PC 提交的 task_number 上限，超出由驱动分片重提交 |
| `pc_data_amount_scale` | 2 | `PC_DATA_AMOUNT=(amount+4+scale−1)/scale−1` |
| `pc_task_number_bits` | 12 | `PC_TASK_CONTROL = ((0x6\|pp)<<12)\|task_number` |
| `pc_dma_ctrl` | 0 | 无 DMA controller 变体分支 |
| `pc_task_status_offset/mask` | 0x3c / 0xfff | 超时打印用 |

flags（`rknpu_ioctl.h` L98–101）：`PC=1<<0`、**`BLOCK=0<<1`（=0，无位）**、`NONBLOCK=1<<1`、`PINGPONG=1<<2`。
⇒ **vendor `flags=0x5` = `PC | PINGPONG`（pp_en=1）**。旧记「0x5 = PC|BLOCK」有误（BLOCK 恒为 0）——vendor 多 task **ping-pong 是开着的**；我们发射器默认 7 = PC|NONBLOCK|PINGPONG（pp 同为 1，仅 NONBLOCK 位差）。

## 2. 结构体字段序（rknpu_ioctl.h L222–280）

```c
struct rknpu_task      { u32 flags; u32 op_idx; u32 enable_mask; u32 int_mask;
                         u32 int_clear; u32 int_status; u32 regcfg_amount;
                         u32 regcfg_offset; u64 regcmd_addr; } __packed;  // 40 B
struct rknpu_subcore_task { u32 task_start; u32 task_number; };            // 8 B，顺序=先 start 后 number
struct rknpu_submit    { u32 flags; u32 timeout; u32 task_start; u32 task_number;
                         u32 task_counter; s32 priority; u64 task_obj_addr;
                         u32 iommu_domain_id; u32 reserved; u64 task_base_addr;
                         s64 hw_elapse_time; u32 core_mask; s32 fence_fd;
                         struct rknpu_subcore_task subcore_task[5]; };
```

**`subcore_task[i] = {task_start, task_number}`（vendor 真值「全 {0,1}」= start 0 / number 1，不是 {task_number=0, offset=1}）。**

## 3. 槽位语义（源码伪代码）

```c
// rknpu_job_alloc (L143): use_core_num = popcount(core_mask & 0x7)   // AUTO ⇒ 0
// rknpu_job_schedule (L525):
if (args->core_mask == RKNPU_CORE_AUTO_MASK /*0*/) {
    core_index = argmin_i(subcore_datas[i].task_num);   // 最闲核
    args->core_mask = 1 << core_index;                  // ★改写 userspace 可见 args
    job->use_core_num = 1;
}
// rknpu_get_task_number (L71) / subcore_commit_pc (L305-331)：
if (num_irqs > 1) {                                     // RK3588 恒真
    if (use_core_num == 1 || use_core_num == 2)
        { task_start = subcore_task[core].task_start;   task_number = subcore_task[core].task_number; }
    else if (use_core_num == 3)
        { task_start = subcore_task[core+2].task_start; task_number = subcore_task[core+2].task_number; }
}
// 即：num_irqs>1 时 args->task_number / args->task_start **从不参与 PC 编程**
//     （args->task_number 仅用于 task_counter 回显 L266、负载统计 L550/479、
//      job_done 分片判据 L468-473 —— 分片读的也是 get_task_number()=槽位值）。

// subcore_commit_pc (L272)：
task_start += submit_index * 4095;                       // L333：内核分片推进
task_number = min(slot.task_number - submit_index*4095, 4095);   // L334-336
first_task = task_obj[task_start]; last_task = task_obj[task_start+task_number-1];
PC_DATA_ADDR   = first_task->regcmd_addr;                // 只编程一次/核/片
PC_DATA_AMOUNT = (first_task->regcfg_amount + 4 + scale−1)/scale − 1;
INT_MASK = last_task->int_mask;  INT_CLEAR = first_task->int_mask;
PC_TASK_CONTROL = ((0x6 | (flags&PINGPONG?1:0)) << 12) | task_number;
PC_DMA_BASE  = args->task_base_addr;
PC_OP_EN(0x18): 1→0 脉冲
// job_done (L460)：submit_count[core]++ < ceil(get_task_number/4095) ⇒ 重提交下一片
```

## 4. 三个历史观测的源码解释

| 观测（台账） | 源码解释 |
| --- | --- |
| vendor 全槽 `{0,1}` ⇒ `task_counter 0→3` 全执行 | AUTO 选核后 `use_core_num=1` 读 **slot[chosen].task_start=0, task_number=1** ⇒ 每核 PC 只编程 **1 个 task**（first=last=task[0]）；task1/2 由**流内尾写链**推进；`args->task_number=3` 只用于回显/负载。**全 5 槽同值 = 对 AUTO 任选核鲁棒** |
| 我们 `{0,N}`（slot.task_number=N）⇒ 6.25 s 超时 | PC_TASK_CONTROL 编程 N ⇒ 硬件把 task0 块当 N-task 组首，**等流内链推进**；我们的流没有 vendor 尾写 ⇒ 挂（§7.46.4 同理） |
| 我们 `{k,1}`/槽位错 ⇒ 无输出/不稳 | `task_start=k` 指向描述符数组第 k 项 ⇒ 取错 regcmd_addr（越界/他块）；§7.61/§7.170 的槽位错位同源 |

## 5. vendor 形态 N=3 描述符目标规格（Task 1 对拍基准）

```c
struct rknpu_submit args = {
  .flags        = 0x5,        // PC | PINGPONG（pp_en=1 ⇒ PC_TASK_CONTROL=0x7<<12|N）
  .timeout      = <取 vendor trace submit 事件值>,
  .task_start   = 0,          // RK3588 PC 路径不读（slot 覆盖），保持 0
  .task_number  = 3,          // 回显/分片/负载口径
  .task_counter = 0,          // 驱动写回
  .task_obj_addr = <描述符 BO>,
  .task_base_addr = 0,        // PC_DMA_BASE 直写
  .core_mask    = 0,          // AUTO：内核改写为单核 mask + use_core_num=1
  .subcore_task[0..4] = {{0,1},{0,1},{0,1},{0,1},{0,1}},  // {task_start, task_number}
};
// 描述符数组：task[0] 被读（regcmd_addr/regcfg_amount/int_mask）；
// task[1..2] 存在但不被 PC 编程读取（流内尾写链负责推进）。
// 每描述符 regcfg_amount：vendor=69（4 B 字计，§7.62）vs 我们发射器 n−4 口径 —— Task 2 二分变量④。
```

**驱动改写 vs 透传**：改写 `core_mask`（AUTO→单核，且写回 userspace args）、`task_counter`（=task_number 回显）、`hw_elapse_time`；透传 `flags/timeout/task_base_addr/subcore_task[]/task_obj_addr`；`task_start`（args 顶层）在 PC 路径不使用。

## 6. 两条「单核 N task」路径（修正 Architecture 认知）

1. **驱动分片**（L333/468–473）：`slot.task_number > 4095` 时内核按 `submit_count` 自动分段重提交——**不需要流内链**，但每段仍是一次 PC 编程、段间依赖驱动中断推进；适合超大 N 的批量。
2. **流内尾写链**（vendor 常态，N 可为 3 等小值）：slot `{0,1}` + `task_number=3`（回显），块尾写 `PC_BASE_ADDRESS/PC_REGISTER_AMOUNTS/PC_OPERATION_ENABLE=0xd` 把 PC 指向下一块。**我们此前从未试过 slot.task_number=1 + task_number=3 的组合**（§7.62 用了 slot=N）⇒ Task 1 回放必须按 vendor 口径。
