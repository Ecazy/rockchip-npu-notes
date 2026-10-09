# FP16 BS/BRDMA ALU(add) + MUL 级标定（§7.24，2026-09-30）

结论来源：`fp16_bs_mul_probe.log`（板端 Orange Pi 5 Plus，RKNPU driver v0.9.8）。

## 结论

fp16（`PROC_PRECISION=2`）下 BS/BRDMA 的语义为：

```
out_f16 = f16( (conv_fp32 + ALU) * MUL )
```

- **ALU addend = fp32**，**先施加**。
  - immediate（`BS_CFG.bit8=0`）= `BS_ALU_CFG` **整个 32 位按 IEEE-754 fp32** 读；
  - BRDMA src（`BS_CFG.bit8=1`）= bias 表槽内 **fp32**；
  - algo `0/1/2` = max/min/add（`BS_CFG[16:19]`）。
- **MUL factor = fp16**，**后施加**，**无 shift**（与 int8 的 s2.14 定点 + 移位不同）。
  - immediate（`BS_MUL_CFG.bit0=0`）= `BS_MUL_CFG[31:16]` 的 **fp16** 位型；
  - BRDMA src（`BS_MUL_CFG.bit0=1`）= mul 表槽内 **16 位按 fp16**。
- RELU：`BS_CFG.bit6` bypass；`bit7` 上界 = `BN_RELUX`。
- `BS_MUL_CFG` 的移位字段（`[13:8]`）在 fp16 下 **不参与**。

## 关键证据

| 用例 | 期望 | hw |
| --- | --- | --- |
| ALU 立即数 `0x40a00000`(fp32 5.0)，MUL bypass | conv+5 = 41 | 41 |
| ALU `0x3f000000`(fp32 0.5) | 36.5 | 36.5 |
| **ALU `0x44008000`(fp32 512.5) 判别** | 550（若截 bf16 则 548） | **550** |
| ALU `0xc0000000`(fp32 -2.0) | 34 | 34 |
| MUL 立即数 `0x40000000`([31:16]=fp16 2.0) | 36×2 = 72 | 72 |
| MUL `0x3f000000`([31:16]=fp16 1.75) | 36×1.75 = 63 | 63 |
| MUL `0x38000000`([31:16]=fp16 0.5) | 18 | 18 |
| MUL `0xc0000000`([31:16]=fp16 -2.0) | -72 | -72 |
| MUL 默认 `0` | **0**（fp16 +0.0） | **0** |
| ALU +2.0 then MUL ×2.0 | (36+2)×2 = 76 | 76 |
| 表 bias fp32 5.0 only（`0x20150` MUL bypass） | 41 | 41 |
| 表 mul fp16 0.5（`0x3800`=14336） | 36×0.5 = 18 | 18 |
| 表 bias 5.0 × 表 mul 2.0 | (36+5)×2 = 82 | 82 |
| 表 bias 常量 2.5 × 表 mul ramp(oc0=0.375) | 38.5×0.375 = 14.4375 | 14.4375 |
| 3×3 全 pattern：ALU +1.25 × MUL 1.5 | 163.875 | 163.875 |

全部 `RESULT: BIT-EXACT (fp16)`、`fp16-mismatch=0`、`max ulp=0`、`submit ret=0`。

## 谜底

`--bs-cfg 0x20140`（MUL 未 bypass）在 fp16 下**恒输出 0**：默认 `BS_MUL_CFG=0` ⇒
MUL factor = fp16 `+0.0` ⇒ 结果 0。并非寄存器/精度问题。

## 说明

- 探针：`experiments/run_fp16_bs_mul_probe.sh`（`--bs-cfg 0x20140` 是 ALU src=BRDMA + MUL active；
  `0x20150` 另含 MUL bypass；`0x20050` 是 ALU immediate + MUL bypass；`0x20040` 是 ALU immediate + MUL active）。
- fp16 带表 DPU 路径仍有偶发 ETIMEDOUT（环境），探针内置重试；本次 14 例全部一次通过。
- 参考实现：`apply_bs_fp16()`（`experiments/min_conv_exp.c`）。
