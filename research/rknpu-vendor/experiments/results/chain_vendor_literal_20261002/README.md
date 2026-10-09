# §7.172 chain_vendor_literal（2026-10-02，路线图 Task 1）

## 产物

- `blocks.bin`（588 B）= `extract_task_blocks.py` 从 `attn_relu_vendor.jsonl` seq1 提取的
  task0/1/2 捕获字节 verbatim（69/69/9 u32 words = 34.5/34.5/4.5 条目；DST = 0xffaa0000/0400/0800，步进 0x400）。
- `meta.txt`：每块字数/条目/DST/首尾寄存器。
- `vendor_replay.c` + `extract_task_blocks.py`（仓库 `experiments/`）。
- 本 README + 运行输出（下方 transcript）。

## 板上结果（2026-10-02 22:44–22:49，基线 soft=99※）

| 变体 | 内容 | 结果 |
| --- | --- | --- |
| v0 | 纯字面回放：捕获字节(276/276/36 B) + 0 填充至槽位(0x280/0x280/0x80)，**无尾写**；描述符 = vendor 规格（flags=5, task_number=3, slots {{0,1}×5}, amounts 69/69/9, int_mask=0x300, enable_mask=0xf, timeout=6000） | **超时** 6239 ms, hw=0, task_counter=0 ✗ |
| v1 | 同上 + 重建尾写（slot0/1@entry35: PC_BASE=下一槽, AMOUNTS=0x27/0x7, VERSION=0x41, OP_EN=0xd；slot2 终止: AMOUNTS=0, OP_EN=0x18） | **超时** 6286 ms ✗ |

※ 基线漂移：handoff 记 soft=92，本轮起点实测 99（+7 来自会话间外部活动，板上 21:37 有 min_conv_exp 重编译与非本会话提交）；本轮 2 次计划内尝试 99→101。

## 过程事故（Ruling 级）

第 3 次意外提交（归档命令误带 `--timeout 1` 重跑）触发 **`job commit failed` + `soft reset, num: 6`（楔死签名=1）** ⇒ 按 Global Constraints **立即停手 + 板 reboot**（22:49，重启后 wedge=0、soft=0）。
教训：**证据归档禁止复跑二进制**（哪怕改小 timeout）——每次运行都是一次真提交。

## 定性（本轮离线分析）

1. **纯字面单块也超时**（v0）⇒ 差异在**槽位/描述符层**，不是链协议本身：
   - v0 槽内 = 34.5 条捕获 + **300 B 全零**；驱动 `PC_DATA_AMOUNT=(69+4+1)/2−1=36` 单位（每单位疑 16 B = 36 条目 = 576 B）⇒ 硬件会执行**约 32 条全零条目**（target=0 写）——vendor 真实槽位的同区间是尾写+有效数据，不是零。
   - v1 的尾写放在 entry35（280 B），但 PC 窗口 36 条只到 entry35 ⇒ 只执行了第一条尾写（PC_BASE），AMOUNTS/OP_EN 在窗外 ⇒ 链必死；且 entry35 = 孤儿半条目（0x40a0 高半缺）垃圾写。
2. **mul trace（74 条全捕获）给出终端尾写真值**：[pad, PC_REGISTER_AMOUNTS=0(tgt 0x101), PC_VERSION=0(tgt 0x41), PC_OPERATION_ENABLE=0x18(tgt 0x81), pad] —— 其 3 描述符**同址**（0xffffe040）+ task_counter 0→3 = 单块执行 + 回显假象（非 3 次执行）。
3. **hook 口径差异（关键混淆源）**：attn 系 hook 捕获 = `regcfg_amount × 4 B`（276 B，含 1 孤儿半条目）；mul 系 hook = `(amount+5)×8 B`（592 B = 74 条含尾写）。⇒ **vendor 尾写在两种 trace 里都没被完整捕获（attn）或只对 mul 形态有效（mul）**；attn 槽位 bytes 276..640 = 未捕获区（尾写 + 未知）。
4. `job commit failed` 的直接触发 = 归档误跑 `--timeout 1`（driver abort ret=-22 路径）；与 v0/v1 的 6 s 超时同族但走了楔死分支。
5. 任务描述符（op_idx=1/enable_mask=0xf/int_mask=0x300）与 vendor mul trace 一致，但 attn trace 的 hook 未捕获 int_mask/enable_mask ⇒ 不排除这两个字段有出入。

## 下一步（Task 2 fork B）

1. **mul 全块回放**（1 次尝试）：`tr_mul_m025_fp16.jsonl` 的 74 条（含终端尾写）+ vendor mul 描述符形状（3 描述符同址 0xffffe040、amount=69、int_mask=0x300）⇒ 若成功 ⇒ harness 无误，问题 = attn 槽位未捕获区；若超时 ⇒ 差异在 harness（BO flags/enable_mask/int_clear 口径）。
2. `rknpu_submit` args 逐字段 diff（我方 vs mul trace submit 事件）：已知一致项 flags=5/task_start=0/task_number=3/task_base=0/core_mask=0/slots{{0,1}×5}；待核 = **enable_mask、int_clear、priority、iommu_domain_id**。
