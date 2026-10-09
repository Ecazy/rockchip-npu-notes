# §7.173 chain_bisect（2026-10-02，路线图 Task 2 fork B）—— 链协议闭合

## 结论

**单 subcore 多 task 流内尾写链闭合**：vendor 34/34/4 形态块（完整 fetch 窗口）+ vendor 描述符形态，
`--variant 2 --subcore-num 3` 单次提交 ⇒ **ret=0、hw=25374 ns、TASK-OUT 0/1/2 全 HIT**
（逐区哨兵校验 ✓✓✓），楔死签名 0。假成功防线：task_counter=3 为回显不作证据；证据 = 3 区哨兵
 HIT + hw 随 task 数变化（1 task 19250 ns → 3 task 25374 ns，≈3.3 µs/task 边际）。

## 根因链（从 §7.172 超时到闭合的三步）

1. **旧 hook 半截捕获**（§7.172 的致命伤）：attn_relu_vendor.jsonl 的 regcmd = `regcfg_amount×4 B`
   （276 B）= 半截；真块 = `round_up(amount+4, 2)` 条目（69+4→74 条 = 592 B）。
   ⇒ v0/v1 的槽内 300 B 全零被当寄存器写执行 ⇒ 超时。
2. **规范 hook 重抓**（`attn_refull.jsonl`，EPERF_RKNPU_REGCMD_MAX_BYTES=8192）：
   task0/1 = **592 B（74 条 = 69 配置 + 5 条尾写）**、task2 = 112 B（14 条）—— 尾写真值到手：
   - 链式尾 = [PC_BASE_ADDRESS=下一槽(tgt 0x101), PC_REGISTER_AMOUNTS=0x24/0x6(tgt 0x101),
     PC_VERSION=0(tgt 0x41), **PC_OPERATION_ENABLE=0x18**(tgt 0x81), pad]
   - **PC_REGISTER_AMOUNTS = 下一槽 fetch 窗口/16 − 1**（0x24=36 ⇒ 74 条；0x6 ⇒ 14 条）✓ 与 §7.60 推导同构
   - **增补发现：reg 0x5018（RDMA 源地址）也逐 task 步进 +0x400**（旧半截捕获完全看不到）
3. **vendor_replay v2**：全字面回放（148/148/28 words），只 patch DST(0x4020)/SRC(0x5018)/尾写 PC_BASE(reg 0x10)；
   描述符 = vendor 真值（enable_mask=**0x18**、int_clear=0x1ffff、int_mask=0x300、amounts 69/69/9、
   slots={{0,1}×3,{0,0}×2}、timeout=36630）。
   - 首跑（slots {0,1}）：**ret=0、hw=19250 ns、TASK-OUT 0 HIT、1/2 UNTOUCHED** ⇒
     PC 按 **slot.task_number** 决定取指次数（1 = 只跑一块，链指针被忽略）✗ 半闭合。
   - 终跑 `--subcore-num 3`（slots {{0,3}×3,{0,0}×2}，= vendor 3663-job 的 {0,1221}×3 同构）：
     **ret=0、hw=25374 ns、TASK-OUT 0/1/2 全 HIT ⇒ ALL-TASKS-EXECUTED ✓✓✓**

## 链协议定论（可复用口径）

```
单核 N task（N ≤ 4095）：
  args = { flags=0x5, task_number=N, task_base=0, core_mask=0(AUTO),
           subcore_task[chosen] = {task_start=0, task_number=N} }   // 其余槽同构或 0
  描述符数组：task[k].regcmd_addr = 流内第 k 槽地址（物理连续不必，链指针决定），
             regcfg_amount = 配置条目数（vendor 口径：本例 69/9 = entries），
             enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff
  流：每槽 = round_up(amount+4,2) 条目 = 配置 + 尾写
      [PC_BASE_ADDRESS=下一槽(0x101), PC_REGISTER_AMOUNTS=下一槽fetch/16−1(0x101),
       PC_VERSION=0(0x41), PC_OPERATION_ENABLE=0x18(0x81), pad]
      终端块 = AMOUNTS=0 + PC_BASE 无效化
  PC_TASK_CONTROL.task_number = N ⇒ PC 取 N 次（链由流内尾写 PC_BASE 推进）
```

## 本轮提交账目（3 次，上限内 1 次浪费）

| # | 运行 | 结果 |
| --- | --- | --- |
| 1 | v2 首版（**缺陷**：variant≥1 误写重建尾写 @280 覆盖 vendor 真尾写 + 终端 AMOUNTS=0x6 未清） | 超时 6358 ms ✗（soft +1） |
| 2 | v2 修复（尾写 verbatim + PC_BASE/SRC/DST/终端 AMOUNTS patch）+ slots {0,1} | ret=0 hw=19250 单 task ✗（链未启动） |
| 3 | 同上 + `--subcore-num 3` | **ret=0 hw=25374 3/3 HIT ✓✓** |

soft reset：reboot 后 0 → 1（损坏首跑）→ 2；楔死签名 0 ✓。task_counter=3 = 回显（不作证据）。

## 备注

- 三区输出前 8 B 完全相同（2a 7a ca 1a...）：SRC(0x5018) 未必是本 op 的实际读源（或 op 读 elsewhere）；
  **链闭合判据 = 3 区哨兵全 HIT + hw 缩放**，op 数值语义归 Task 5。
- attn_refull.jsonl（重抓全窗口 trace，4950023 B）压缩存于 `../chain_vendor_literal_20261002/attn_refull.jsonl.gz`。
- 复跑配方：板上 `./vendor_replay --variant 2`（blocks.bin = blocks_full.bin 1296 B）。
