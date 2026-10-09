# §7.186 bisect 基础：vendor v2b（工作链）vs 我方发射器 task0/task1 配置值分类

对拍基准：`cbc_blocks.bin`（vendor 运行时终值，v2b 复刻=全执行）vs
`ourchain_timeout.jsonl`（emitter EMIT 捕获，§7.177 时代；尾 OP_EN 已被 §7.183
的 0x1d 旋钮修正，故 [129] 0x18 差异作废）。寄存器序列与 target 完全一致，差异全为 VAL。

## 分类

### A. 地址（预期差异，harness 已自适应）
0x1070 FEAT / 0x1110 DCOMP / 0x4020 DST / 0x5020 BS / 0x0010 PC_BASE

### B. 几何（模型不同，预期差异；单 task 已证可执行）
0x1024 DATA_SIZE1、0x1030/0x1034/0x1038 WEIGHT_SIZE0/1/2、0x1044 CBUF_CON1、
0x107c/0x1080 DMA_CON、0x1088 FC_DATA_SIZE1、0x3018/0x403c/0x4058/0x5014 尺寸、
0x4050 BS_OW_CFG、0x5044 RDMA_FEATURE_MODE_CFG（ven bit16 置位！）

### C. 精度（int8 vs fp16，语义性差异）
0x4080/0x4084/0x4088 OUT_CVT（int8 反量化 vs fp16 直通）、0x3010 CORE_MISC_CFG（ven bit9 PROC_PRECISION=2 vs 我 bit0 QD_EN）

### D. 控制位候选（链敏感嫌疑，flip 实验对象）
| reg | 名称 | vendor | ours | 备注 |
|---|---|---|---|---|
| 0x100c ×2 | CNA_CONV_CON1 | 0x6000a120 (NONALIGN_DMA\|GROUP_LINE_OFF\|ARGB_IN\|PROC_PREC=2) | 0x0 | §7.1 专用路径同源 |
| 0x104c | CNA_CVT_CON0 | 0x1 | 0xb | runtime 计算值（model 0xb→runtime 0x1），emitter 沿用 model 值 |
| 0x1180 | CNA_CVT_CON5 | 0xfff | 0x0 | PER_CHANNEL_CVT_EN 12 通道 |
| 0x1184 | CNA_PAD_CON1 | 0x0 | 0xffff8080 | pad 值（int8 -1 约定） |
| 0x1068 | CNA_PAD_CON0 | 0x0 | 0x11 | pad 几何 |
| 0x1068 同 | | | | |

### E. 终端尾写次序（block1，emitter 待修候选）
vendor 终端 = [pad(0x000/0x0000), AMOUNTS=0(tgt 0x101), VERSION=0(tgt 0x041), OP_EN=0x1d(tgt 0x081)]
ours        = [AMOUNTS=0, VERSION=0, OP_EN(0x18→knob 0x1d), pad]  ← **pad 在尾，vendor 在首**
（注：block0 非终端尾写两侧同序同值，已证可执行；终端次序差异只影响 task1 结束，
不解释 seg1 不取指，但作为 emitter 修正项一并落地）

## 二分实验设计（v2b=已知良好基线，flip vendor→ours）
- v2c：全部 6 控制位翻转（0x100c→0、0x104c→0xb、0x1180→0、0x1184→0xffff8080、
  0x3010→1、0x4010→0）——若 ret=0 ⇒ 控制位全排除，嫌疑转向几何/数据/BO 层；
  若 counter=1+DST0 HIT ⇒ **精确复现 emitter 挂点签名** ⇒ 二分锁定单寄存器。
- 尾 OP_EN 不入列（§7.183 已证 0x18 全死、0x1d 已匹配）。
