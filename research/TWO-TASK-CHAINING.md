# Two-Task Chaining：HardSigmoid 四组对照

2026-10-09，Orange Pi 5 Plus / RK3588，内核 `6.1.43-rockchip-rk3588-rknpu098`，厂商 `rknpu 0.9.8`，读取 NPU 频率为 1 GHz。
本轮仅执行独立 DPU/DPU_RDMA HardSigmoid，不引入 MatMul、归一化或其他复杂算子。

**A/B/C/D 均通过。** 三个核、自动/固定寄存器组两种配置、各两次独立进程，共 12 个进程；
每个进程连续 128 轮，每轮依次执行四组实验。合计 1,536 轮、15,360 个在线数值对照、
7,680 次 submit、10,752 个 task。每轮均验证输出和内存边界。

最终记录：[two-task-20261009-174616](custom-ops/results/2026-10-09/two-task-20261009-174616/)，
汇总与文件 SHA256 见 [campaign.json](custom-ops/results/2026-10-09/two-task-20261009-174616/campaign.json)。
当前源码已与该轮 [源码清单](custom-ops/results/2026-10-09/two-task-20261009-174616/source-manifest.json) 逐文件核对。

## 对照与数值结果

令 `H(x)=clip(x/6+0.5,0,1)`。Xa/Xb 使用不同数据，各 task 具有独立输出区。
同一轮中各组使用相同 shape 和输入，保持算子配置一致。

| 实验 | 实际任务与提交 | 结果 | 每轮 IRQ 增量 |
|---|---|---|---:|
| A | `H(Xa)→Ya`，一个 task、一个 submit | CPU 参考通过，作为本轮基线 | 1 |
| B | `H(Xa)→Ya`、`H(Xb)→Yb`，两个独立 task、两个 submit | 两个输出均通过，task0 与 A 逐位相同 | 2 |
| C | 同 B，两个独立 task、一个链式 submit | 两个输出均通过，task0 与 A 逐位相同 | 1 |
| D | `H(Xa)→mid`、`H(mid)→Yd`，两个依赖 task、一个链式 submit | 中间与最终输出均通过，mid 与 A 逐位相同 | 1 |

第一阶段相对独立 CPU 数学参考的最大绝对误差为 `0.000518798828`；
D 第二阶段相对 `FP16(H(FP16(H(Xa))))` 参考的最大绝对误差为 `0.00048828125`。
两者门限均为 `0.002`。跨组 task0 比较检查实际 FP16 字节，而非仅比较浮点数值。
每个进程前 3 轮另保存输入及所有 task 输出，共 27 个张量文件；独立 Python 数学参考复核 21 个输出，
12 个进程合计 252 个额外离线输出核对。同核、同寄存器配置的两次进程张量文件逐字节一致。

## 压力测试

每个进程保持同一 fd，在 A/B/C/D 之间连续提交，无人为 sleep 或 reset。
shape 按 `16×32、29×45、64×64` 轮换，其中 `29×45` 需要物理 padding。
Xa 的符号/偏移及 Xb 的偏移逐轮交替；每 13 轮加入一次全零 Xa，用不同的 Xb 检查独立任务。
每轮结束后释放数据 BO，下一轮重新分配，覆盖地址复用与不同几何之间的状态继承。

所有输出提交前填 NaN 哨兵，未执行或漏写会导致数值失败；并验证：

- 输入 BO 的完整 payload 与提交前快照相同。
- 每个 View 前后各 64 B 留白不被写入。
- 不参与当前组的输出区保持 NaN 哨兵。
- 物理 cube 的补齐行/通道也写入正确结果：第一阶段 padding 为 `H(0)=0.5`，D 第二阶段为 `H(0.5)` 的近似。
- 每个 BO 的整个 32 KiB DMA 余量仍为零，64 B 尾部 guard 保持 `0xA5`。

在这些范围内没有观察到错误写回或跨任务污染。

## 描述符与 Regcmd 边界

[只读采集器](custom-ops/tools/c/two_task_trace.c) 读取实际 MEM_CREATE/MEM_MAP/mmap 和 submit 参数，
在 ioctl 前解析真实 task BO 与 regcmd BO，而不是记录发射器的预期参数。

每个 task 描述符为 40 B；二任务的描述符偏移为 `0/40`，使用区间为 80 B。
核对实际分配大小、mmap 长度、fd、domain、task object 身份和所选 subcore 的 start/count。
在 ioctl 返回后比较整个描述符映射，只允许最后一个 task 的 `int_status` 四字节发生驱动预期写入，
其余描述符、未使用槽位和页内剩余区域保持不变。

每个 HardSigmoid task 为 1,070 个 64 位命令；描述符 `regcfg_amount=1066`，另有 4-word trailer。
实际 fetch 为 8,560 B，16 B 对齐；C/D 的第二段相对第一段偏移为 `8560`（`0x2170`）。
第一段 PC 前向链接必须等于第二段实际 regcmd 地址，下一段 amount 为 `534`（`0x216`）。
最后一段前向链接封为 OP_NONE。所有 fetch 区间都必须落在仍存活的 regcmd BO 内。

所有 submit 均保存完整的紧凑字段记录，原始 trace 以 `trace.jsonl.gz` 保存。
每个进程前 15 次 submit 另保存完整 regcmd 二进制，覆盖三种 shape 的全部 A/B/C/D 组合；
离线解码逐项对照地址、尺寸、S_POINTER 和尾写字段。

## 寄存器组与完成事件

每个核分别运行以下两种配置，其他计算参数保持相同：

| 配置 | DPU/RDMA S_POINTER | 实际完成位统计 |
|---|---|---|
| 自动乒乓 | 两者均为 `0xE` | `DPU_0=0x100` 共 1,920 次，`DPU_1=0x200` 共 1,920 次 |
| 清除并固定组 0 | 两者均为 `0x30` | 全部 3,840 次为 `0x100` |

完成位取自驱动在 IRQ 路径写回的最后一个 task 的 `int_status`；没有用 `task_counter` 回显充当完成证明。
自动模式中，B 的两个单 task submit 完成位交替；每轮共有七个 task，因此相邻两轮 A 的完成位交替。
C/D 两个 task 对当前组的推进为偶数，最后完成组与本轮 A/B 第二次提交一致。
这些规律在所有核、重复进程和 128 轮中均核对。

实际 `/proc/interrupts` 增量与 A/B/C/D 的 `1/2/1/1` 对应，且只增加所选核的 IRQ 行。
每次 ioctl 成功返回，驱动硬件区间计时为正，所有引用对象仍存活。
本轮最长 submit ioctl 区间为自动模式 `0.218455 ms`、固定模式 `0.319954 ms`，远低于 1,500 ms 的慢提交拒绝门限。
该区间用于诊断超时，不作为性能对比结论。

执行组通过真实 DPU 完成位验证，RDMA 的切换配置和实际数据来源一起核对；没有直接读取 MMIO 的 S_POINTER。
CNA/CORE 不参与本轮计算，不以此实验声称它们的组切换也已经验收。

## 缓冲区生存期与 D 的依赖

采集器为每次分配生成唯一 ID，即使 handle、地址或 kernel object 被复用，也不会混同旧对象。
所有 task/command/input/output 引用在 submit 进入和返回时都对应仍存活、同 fd/domain 的对象。
活动 submit 内没有 sync、unmap 或 destroy；返回后才释放 regcmd，数据 BO 在本轮验证结束后释放。
十二个进程合计跟踪 16,908 次分配，结束时每个进程的重建 live 集合和采集器计数均为零。

D 的 task1.src 与 task0.dst 的设备地址和分配 ID 完全相同。
中间 cube 在同一 submit 内由 NPU 写入并读取；两 task 之间没有 CPU 读取、重排、复制或 cache sync。
提交结束后才下载 mid 与最终输出作独立评分。

## 验证器、复跑与范围

入口文件：

- [two_task_probe.c](custom-ops/tools/c/two_task_probe.c)：四组实验、IRQ、输出/输入/留白/padding 检查。
- [two_task_trace.c](custom-ops/tools/c/two_task_trace.c)：实际 ABI、完成回写和分配生命周期采集。
- [two_task_verify.py](custom-ops/tools/python/two_task_verify.py)：独立协议与张量证据核验。
- [two_task_verify_test.py](custom-ops/tests/python/two_task_verify_test.py)：15 个主机测试，包括错链、描述符溢出/杂写、提前释放、错误 IRQ/完成位、未切组、缺记录、坏张量及 JSON 归档一致性。
- [run-two-task-board.sh](custom-ops/scripts/run-two-task-board.sh)：从新目录上传源码、构建、串行跑完并回收证据。

```bash
bash research/custom-ops/run-two-task-board.sh
python3 research/custom-ops/two_task_campaign.py \
  research/custom-ops/results/2026-10-09/two-task-20261009-174616 128
```

第一条需板端配置与权限，执行硬件实验；第二条只复核已有证据。
已有 `campaign.json` 与 `validation.json` 不会被重新签入新哈希；文件变更或记录不一致会失败。
主机和板端五个 CTest 门均通过；主机 ASan/UBSan 覆盖的测试路径也通过。
[kernel-after.log](custom-ops/results/2026-10-09/two-task-20261009-174616/kernel-after.log) 为 `-- No entries --`，没有 NPU timeout、IOMMU fault 或 reset 日志。

首轮数值与硬件门通过，但归档复核因整数 bank 键在 JSON 中变成字符串而失败，
记录保留于 [171011 轮](custom-ops/results/2026-10-09/two-task-20261009-171011/)。
修复验证器的序列化稳定性后重新运行，最终记录另存于上述 174616 目录，没有覆盖失败记录。

实验工具和验证器最终由本助手完成。CSU/DeepSeek 两次调用返回 Internal Server Error，未生成代码；
用户随后要求停止委派，之后未再调用 opencode。

结论适用于本轮 RK3588 厂商 DRM 提交通路及 DPU HardSigmoid 程序。
它证明了当前发射器的 Two-Task 独立链和依赖链可靠性，不推广到任意算子，也不解释旧 Conv 链损坏根因。
