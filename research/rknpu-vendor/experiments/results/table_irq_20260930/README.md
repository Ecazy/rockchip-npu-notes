# 带表 DPU 提交「逐次交替超时」的根因刻画（§7.39）

针对 handoff 未闭合项 #1「fp16 带表 DPU 路径的切换脆弱性」。本轮把它从「随机/奇偶
ETIMEDOUT，重试即消」细化为**可解释的机制**，并给出测量纪律。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8（源码在 `refs/rknpu-driver-0.9.8/`）。
复跑：板端 exp2 `bash run_table_irq_probe.sh`（日志 `table_irq_probe.txt`）。

## 现象（本轮新增的逐次诊断）

发射器新增 `WARMUP n/m ret=` 逐次打印（以及 `--task-count N`）。带表提交
（BS/BRDMA **或** BN/NRDMA）在同一进程内 3 次热身 + 1 次测量呈：

```
BS/BRDMA : 0, -1, 0, -1      （0=ok，-1=ETIMEDOUT 6 s）
BN/NRDMA : 0, -1, 0, -1
纯 conv  : 0,  0, 0,  0      （对照：无表 ⇒ 从不超时）
```

**不变量**（改了都一样交替）：`--job-flags 5`/`7`、`--pp-val 0`（关 ping-pong 指针）、
`--brdma-bias 0`（表值 0）、`--task-count 1/2/3`（单次 submit 里放几个 task）。

## 机制（驱动源码 + dmesg 证据）

1. **驱动只按中断完成 job，且是等值判定**（`rknpu_job.c:666`）：
   `if (rknpu_fuzz_status(status) != job->int_mask[core_index]) { 丢弃 }`，
   `fuzz_status` 把 6 对位（`0x3/0xc/0x30/0xc0/0x300/0xc00`）整对放大；CONV 完成是
   `0x300`、PPU 完成是 `0xc00`（见 `docs/RKNPU-构建算子-实测流程.md` §2.4）。
2. **失败提交的硬件其实做完了**：`hw_elapse` 有值（10–77 µs）；超时日志里
   `task counter: 0x0`（PC 已无待处理任务）。
3. **完成中断根本没进处理函数**：dmesg 中 `invalid irq status` **0 次**
   （480 次 job timeout 全部 `require mask: 0x300`）⇒ 不是"进来但掩码不匹配"，而是
   **IRQ 没来**（或走了 `if (!job)` 早退分支）。
4. **超时时刻的 `INT_RAW_STATUS` 带相位**：480 次里 `0xc000002a`(219) / `0xc0000015`(71) /
   `0xc000000a`(69) / `0xc0000005`(35) / `0xc0000001`(27) —— 低 6 位是 `0x2a` 与 `0x15`
   这样的**奇/偶位互补**（`0x2a = bits1,3,5`，`0x15 = bits0,2,4`）+ 位 30/31。
5. **相位随提交推进、并被 soft reset 重置**（`state_dependence.txt`）：同一条 BS 命令在
   本轮早段是 `0,-1,0`，末段（timeout=480、soft reset=480）变成 `0,0,0`×5；`--job-flags 5`
   在同一时刻变成 `-1,-1,-1` ⇒ **单次观测不足以判定某配置"修复了"**。

## 结论与纪律

- **性质**：不是寄存器配置错误，而是**带表路径的完成中断与驱动固定 `int_mask` 之间的相位
  不匹配**（硬件完成、中断不来 → 驱动 6 s 超时 → soft reset）。固定 `int_mask` 无法同时
  匹配两种相位 ⇒ `--job-flags`/`--pp-val`/表内容/task 数都改不动它。
- **测量纪律**：表格路径的提交**必须重试**；判定语义正确性时用重试后的成功提交，
  timeout 一律记为环境失败（与 §7.12 一致）；**性能测量**用「重复取 min」并同时记录
  每提交的 ret 序列与 soft reset 计数（否则会把相位差异记成性能差异）。
- **对真实 workload 的含义**：vendor 模型里 BS 常开、BN 折进前置 conv（§7.16/§7.21），
  且我们观察到的相位问题只在**反复单发带表提交**时稳定出现；真实链路不受影响。

## 追加：累计超时 → 内核 oops → NPU 楔死（本轮实际发生）

为刻画现象累计约 480 次带表超时（每次触发 soft reset）后：

- dmesg 出现 `Unable to handle kernel paging request … Internal error: Oops`（BO 释放路径，
  与 `docs/RKNPU-构建算子-实测流程.md` §2.4「坑二」记录的一致）；
- 此后所有 submit 返回 `SUBMIT ret=-1 errno=Invalid argument`，dmesg 为 `job commit failed`
  + `job abort … ret: -22` ⇒ **NPU 楔死，只能重启板卡**；
- `reboot` 后约 3 分钟恢复（`RKNPU driver: v0.9.8`、soft reset 归零、纯 conv 立即
  `ret=0` + ULP-EXACT、带表单次 `ret=0` ✓）。

**冷态复测**（重启后 soft reset=0 的第一次带表运行）：3 热身 + 1 测量 = `0,-1,0,-1`，
伴随 2 次 soft reset ⇒ **相位从冷态起就是完美交替**；soft reset/重启把相位 re-arm 到匹配位
（所以复位后的下一个提交总是好）。详见 `state_dependence.txt`。

**纪律**：带表路径探针必须限次（每次超时一次 soft reset，累计 ~480 次即楔死）；出现
`Invalid argument` + `job commit failed` 时立即停止并 `reboot`；纯 conv/切分路径不受影响。

## 产物

- `experiments/min_conv_exp.c`：`WARMUP n/m ret=` 逐次诊断、`--task-count N`。
- `experiments/run_table_irq_probe.sh`、`table_irq_probe.txt`、`state_dependence.txt`。
