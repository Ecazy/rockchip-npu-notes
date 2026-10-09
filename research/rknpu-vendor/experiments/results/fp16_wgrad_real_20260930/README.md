# FP16 真实张量 wgrad 闭环打通（§7.26，2026-09-30）

结论来源：`fp16_wgrad_real.log`（板端 Orange Pi 5 Plus，RKNPU driver v0.9.8）。
复跑：板端 exp3 目录 `FP16_WGRAD_REAL_LOG=... ./run_fp16_wgrad_real_loop.sh`。

## 结论

- **fp16 训练侧闭环全链路逐位一致**：`wgrad_real.c` 新增 `--fp16` 路径，一次运行完成
  - **Phase F16**：正向 W0（fp16 输入/权重，fp32 累加），与 CPU 参考逐位一致；
  - **Phase W16**：27 个坐标 (kx,ky,ic) 的 dW 全部逐位一致；
  - **Phase U16**：`W' = fp16(W − lr·dW_npu)` 写回后重跑正向，与 `CPU(W')` 逐位一致。
- fp16 无表 DPU 路径**稳定**，多次运行零 ETIMEDOUT（唯一一次失败为板端偶发超时，
  见下「环境噪声」）。
- int8 闭环（`--dir wgdata_pad1 ...`）在改动后**不回归**，仍 `CLOSED-LOOP OK`。

## 覆盖（10/10 相位 `RESULT: CLOSED-LOOP OK (fp16 ...)`）

| 相位 | 参数 | 结果 |
| --- | --- | --- |
| F16-1 | `--pad 1 --batch 1 --update` | BIT-EXACT，27/27 coords |
| F16-0 | `--pad 0 --batch 1 --update` | BIT-EXACT，27/27 coords |
| F16-B | `--pad 1 --batch 2/3` | BIT-EXACT，27/27 coords |
| F16-lr | `--lr 0.5/0.125/1.0 --loss 0.125/0.5/1.0` | BIT-EXACT |
| F16-seed | `--seed 1/2/999` | BIT-EXACT |
| 基线 | int8 `--dir wgdata_pad1 --pad 1 --batch 1 --update` | CLOSED-LOOP OK |

正向判据：`total=1024 mismatch=0`（pad1）/ `total=576 mismatch=0`（pad0）；
wgrad 判据：`WGRAD16_TOTAL: coords=27/27 BIT-EXACT (fp16)`；
更新判据：`F16 FWD2 ... vs CPU(W') BIT-EXACT` 且 `outputs_changed=1024/1024`。

## 关键实现点

- `emit_conv()` 的 fp16 分支（由全局 `g_fp16` 门控）：`input_data_entries` 除以
  `FP16_ATOM_ELEMS=8`；`wbytes×2`；`weights_kernels=ALIGN_UP(MAX2(oc_real,16),16)`
  (=16)；`CNA_CONV_CON1 IN_PRECISION(2)|PROC_PRECISION(2)`；`WEIGHT_SIZE0×2`；
  `CORE_MISC_CFG PROC_PRECISION(2)`；`DPU_DATA_FORMAT` 全精度 2；`DPU_BS_CFG` 全 bypass；
  `OUT_CVT_OFFSET=0`；`OUT_CVT_SCALE=FP32TOFP16_EN(1)|SCALE(1)`；`OUT_CVT_SHIFT=0`；
  `RDMA_FEATURE_MODE_CFG IN_PRECISION(2)|PROC_PRECISION(2)`。
- `c.ic` 沿用 `ic_real<=16 ? 16 : ALIGN_UP(ic_real,32)`（§7.25 结论），正向 c.ic=16、
  wgrad c.ic=ALIGN_UP(P,32)；fp16 wgrad 的 `c.oc=16`（区别于 int8 的 32）。
- 正向发射寄存器序列与 `min_conv_exp --pattern --fp16 --auto-fix --pad-con1 0
  --kw 3 --kh 3 --iw 8 --ih 8 --pad 1 --ic 3 --oc 16` 的 126 条 EMIT **完全一致**
  （`--dump` 对比，diff 为空），是本次定位配置错误的决定性手段。

## 定位记录：`--fp16` 必须同时置 `g_fp16`

初次运行 3/3 `FWD SUBMIT FAILED (ETIMEDOUT)`。`--dump` 对比发现 `DPU_DATA_FORMAT`
仍为 int8 值 `0x0`（应为 `0x48000002`）等——原因是 `--fp16` 解析只置了 `fp16` 调度标志，
未置 `g_fp16` 发射门控，导致 fp16 数据按 int8 寄存器解释。修复后正向 regcmd 与参考完全一致，
闭环通过。

## 环境噪声

`fp16_wgrad_real.log` 中 int8 基线相位出现 1 次 `SUBMIT ret=-1 errno=Connection timed out`
（板端偶发，非语义失败）；单独重跑 3/3 `CLOSED-LOOP OK`。判据：`ret≠0` 一律重试后再记录。

## 说明

- 工具：`experiments/wgrad_real.c`（`--fp16 [--lr F] [--loss F] [--batch N] [--seed N]
  [--update]`）；探针：`experiments/run_fp16_wgrad_real_loop.sh`。
- fp16 路径无 requant/clamp/saturation（Phase S 停用）；更新用 `f16(W − lr·dW_npu)`
  直接舍入，不再做 int8 的 `round(dW/4)` 量化。
- 本轮只覆盖单层（3×3、IC=3、OC=16、8×8）；更深通道（ic>96/oc>64）与多层真实网络
  仍属后续方向。
