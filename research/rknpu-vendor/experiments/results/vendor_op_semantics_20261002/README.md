# vendor_op_semantics_20261002（路线图 Task 5，§3b-7）

数据：`results/bn_lower_20260928/em0d/trace_mul_only_fp16.jsonl`（3 task，= §7.62 的 34/34/4 样本族）
与 `trace_layernorm_only_fp16.jsonl`（54 task），经 `analyze_vendor_dpu_fields.py` 字段级解码 +
`vendor_op_signature.py` 家族指纹对照（§7.150 判别特征表基线）。

## 产物

- `mul_fields.txt`：mul-only 逐字段（69 条配置 + 5 条尾写/块，DPU 侧语义已逐位注解）
- `layernorm_fields.txt`：layernorm-only（54 task）逐字段
- `mul_ln_signature.txt`：72 共有寄存器基线 + 家族判别集

## 关键语义（对照 §7.150 判别特征表）

1. **mul（逐元素乘，vendor 形态）**：72 寄存器全必写、无独有寄存器；语义 =
   `BS_CFG=0x42`（RELU/ALU bypass）+ `BS_MUL_CFG=0x3e000000`（**BS_MUL_OPERAND=15872**
   = 乘数定值，§7.150 已录）+ `BS_OW_CFG=0x2`（OD bypass）+ `BN_CFG=0x53`（全 bypass）+
   `EW_CFG=0x383`（EW 全 bypass）⇒ **mul 走 BS_MUL 通路，DPU 其余子单元全旁路** ✓（与 §7.150 一致）。
2. **layernorm 判别特征 = 0x0010/0x100c–0x1040 区 54 个寄存器**：
   - `PC_BASE_ADDRESS/PC_REGISTER_AMOUNTS/PC_OPERATION_ENABLE` 出现在 **块内**（PC 尾写链独立样本 ✓✓，
     与 §7.173 链协议一致；且同 trace 中 0x18 与 0xd 并存 = 链式/终端两种取值均有实例）；
   - `0x100c–0x1040` 区（PPU/PC 第二 bank）= **layernorm 独有** ⇒ **LN = DPU + PPU/PC-bank2 协同**，
     判别特征表新增 54 项。
3. 72 共有寄存器基线与 §7.118/§7.150 完全一致（不增不减）✓。

复跑：`python3 experiments/analyze_vendor_dpu_fields.py <trace> > mul_fields.txt`；
`python3 experiments/vendor_op_signature.py <mul-trace> <ln-trace> > mul_ln_signature.txt`。
