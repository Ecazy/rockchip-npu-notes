# §7.185 决定性实验：vendor conv 链 verbatim 回放（完全忠实复刻 v2b）

日期：2026-10-03
模型：/mnt/ssd/workspace/bn_lower_probe/conv_bn_conv_fp16.rknn（conv3x3→BN→conv1x1，fp16）
trace：cbc_fp16_full.jsonl（6 regcmd 全窗 + submit 真值，EPERF_RKNPU_REGCMD_MAX_BYTES=8192）

## 实验矩阵（当日 3 次提交，前 2 次为 harness bug 假阴性）

| run | 配置 | 结果 |
|---|---|---|
| v1 | vendor 6 块 + 零数据 + 手建 6 描述符（uniform delta patch，**mask bug**） | ret=-1 ETIMEDOUT, counter=0, zones UNTOUCHED |
| v2 | vmem.bin 全图复刻（真实 blob）+ task.bin=blob[8] verbatim（**mask bug**） | 同上（假阴性） |
| **v2b** | 同 v2 + mask 修正（保留 target 位） | **ret=0, hw=9333ns, DST0 HIT 1023/1024, DST1 HIT 1023/1024, ALL-TASKS-EXECUTED** |

## Harness bug（§7.185 核心教训）
variant 3 的 delta patch 用了错误掩码：
```
nw = (w64 & 0x0000FFFF0000FFFF) | (nv << 16)   /* 错：bits[63:48]=target 被清零 */
nw = (w64 & 0xFFFF000000000000) | ((uint64_t)nv << 16) | (w64 & 0xFFFF)   /* 正确 */
```
后果：26 个指针条目的 target 位（CNA=0x201/DPU=0x1001/RDMA=0x2001/PPU=0x4001/PC=0x101）
全部变 0x000 ⇒ 配置写全部丢失 ⇒ hw 永不启动 ⇒ counter=0。
**教训：对拍 diff 显示 target 0x201→0x000 才抓到；字节级 diff 是唯一可靠的假阴性防线。**

## v2b 复刻配方（决定性成功）
- 单 BO vmem 0x9000 B，镜像 vendor IOVA [0xffff6000, 0xfffff000)：
  0x0000 DST1 哨兵A / 0x1000 input ramp / 0x2000 inter 哨兵B / 0x4000 权重+表 BO 镜像 / 0x4940 流
- 权重/表 = blob[1..6] verbatim（w0@+0x0000, b0@+0x080, w1@+0x00c0, b1@+0x0100, MC11_out@+0x140, MC11_in@+0x540）
- 流 = blob[7]（17 块 13696 B），blocks 0-5 用 trace 终值覆盖（runtime 加载期重定位后的真值）
- 描述符 = blob[8] 17×40B verbatim（regcmd 为流内相对偏移，运行时 += vmem.dma+0x4940）
- 提交参数 verbatim：flags=5, task_number=6, slots={{0,2}×3,{0,0}×2}, core_mask=0(AUTO), timeout=6000
- delta patch（修正版）作用于流区 [0x4940, 0x7ec0)：val≥0xffff0000 ⇒ val += (vmem.dma−0xffff6000)

## 驱动语义确认（rknpu_job.c）
- L266：成功路径 task_counter = task_number（**纯回显**，非硬件真值）；超时路径读 PC_TASK_STATUS（真值）
- AUTO(core_mask=0)：选最少负载核 ⇒ use_core_num=1 ⇒ 只编程 subcore_task[core] 窗口（{0,2} ⇒ task0-1）
- 描述符字段消费：仅 regcmd_addr / regcfg_amount / int_mask（flags/op_idx/int_clear/int_status/regcfg_offset 不用于 PC 模式）

## v2b 提交后全图 diff（36864 B）
- stream 区：26 条 = 恰好 26 个 delta patch（无任何 hw 杂写）
- dst1 区 0x0000-0x0800：2 KB 全新数据（task1 conv2+BN 输出）
- inter+sentB：0x2180-0x2978 2 KB 新数据（task0 conv 输出 = task1 FEAT）
- weights / input：0 变化
- 楔死 0；无 IOMMU fault；soft reset 计数无新增（num:6 不变）

## 结论
1. **vendor conv 链（conv→BN 终端 2-task 链）经我方 harness 完整执行**——提交路径无罪实锤
2. §7.183 的「我方发射器 counter=1 挂点」= 我方块内容/几何问题，与提交路径无关
3. §7.184 的 31 项对拍差成为下轮二分清单：0x100c/0x1180/0x3010/0x4010 + stride 几何
4. vendor 从未链 conv（§7.183 推断）再次确认**错误**——conv 链硬件真实支持（§7.184 定性反转维持）

## 产物
- cbc_vmem.bin（36864 B, sha256 14c179e1…33ea2）+ cbc_task.bin（680 B, sha256 b4c9fcf8…5850d5）
- vmem_after_v2.bin（mask bug 版 dump，counter=0 假阴性证据）/ vmem_after_v2b.bin（成功版）
- build_cbc_vmem.py（镜像构建+指针普查）、diff_blocks2.py、cbc_blocks.bin、cbc_fp16_full.jsonl

---

## §7.186 三连（flip 混杂 → hybrid 挂点复现 → hybrid2 决定性）

| 实验 | 流 | 结果 | 结论 |
|---|---|---|---|
| v2c | v2b + 6 控制位全翻转为 ours | counter=0, DST0 UNTOUCHED | 精度组(0x3010/0x4010/0x1180/0x1184/0x104c)与 fp16 数据冲突 ⇒ task0 直接死（混杂，非链信号） |
| v2e | vendor block0 + 我方 block1 字节（地址重定向） | counter=1, DST0 HIT, DST1 UNTOUCHED | **精确复现 emitter 挂点签名** ⇒ 嫌疑收敛到块内容 |
| v2f | v2e + 终端尾写换 vendor conv 形态 | **ret=0, hw=31792ns, DST0+DST1 HIT, ALL-TASKS-EXECUTED** | **根因 = 终端尾写形态** |

### 根因（§3b-1 挂点）
vendor conv 链终端 = `[pad(tgt0x000,reg0x0000,val0), AMT=0(tgt0x101,reg0x0014,val0), VER=0(tgt0x041,reg0x0000,val0), OP_EN=0x1d(tgt0x081,reg0x0008)]`
我方 emitter（§7.171 mul 真值沿用）= `[AMT=0, VER=0, OP_EN=0x18, pad]` —— pad 在尾 + OP_EN 无 bit0。
conv 家族终端 OP_EN=0x1d（含 bit0），attn/mul 终端=0x18；AMT=0 令 hw 停链。
v2f = vendor block0 + 我方 conv 配置 block1 + vendor 终端形态 ⇒ 全执行（DST1 首字节 0x7f/0x80=int8 量化输出，垃圾值但完整执行）。

vmem_after_v2f.bin sha256 3d1162a4…6a442；board soft reset 无新增、楔死 0、无 IOMMU fault。

---

## §7.186 emitter 自链闭合过程（task C/D）

1. 终端形态移植 + PAD 哨兵入 emitter；seg 拷贝源改 before（旧 n_prev 在 chain-pc 光标复位后
   错位 14 条——seg1 尾写从未进 BO，即「PC 不发起 seg1 取指」假象的真正来源）。
2. emitter 自链首跑：TIMEOUT（拷贝 bug 未修时）→ 修复后 ret=0、hw=9.6-75.8 µs、双描述符
   均被消费、seg1 WDMA 落点正确（chain zone 有真实写入）——**链协议闭合**。
3. op_idx 对齐 vendor（链首 1、逐段 +1；§7.183 的排除实验在拷贝修复前=无效证据）。
4. 遗留：task1 输出 ≈ task0 输出（64-70% 字节相同）但 240/256 atom 各差 1-2 B——输入端
   少量样本被管线跨 task 状态污染（§7.44 预判的 CNA/CBUF 重挂载域）。8×8 小几何同样
   复现（diff 1589/2048）⇒ 非尺寸相关。ones/ones 对照未得干净判据（task0 本身 MISMATCH）。
5. 证据：ourchain3.jsonl（op_idx=[0,1]→[1,2] 前的流）、act_chain{,2}.bin。

---

## §7.187 数据保真线程（未闭合）+ 板卡复位事故

- task1 数据探针矩阵：pad=0 / 1×1 / 8×8 / one-hot 全部受损；task1 非确定性（同配置两次
  运行 hash 不同）⇒ 管线内竞速/状态，非配置值。疑似 task1 的权重/输入取数未重挂载，
  与 task0 排空竞速（vendor 从不重复同配置 conv）。
- stub 探针 ×2 超时（stub 14 条段 vs chain-pc L=144 对齐不兼容，PC_BASE=k*L 与游标放置
  失配——预置 bug）⇒ 板卡 SSH 无响应，用户重启。实验按用户指令停止。
- 证据：ourchain3.jsonl、act_chain{,2}.bin、act_wt_1x1.bin。
- 下轮：修 stub 对齐；--ov 重臂 0x1044/0x1180；或按「协议闭合+同构重复约束」收口。
