# FP16 独立 BN（BN_ALU + NRDMA fp32 表）证据 — 2026-09-30

台账 §7.23。板端日志 `fp16_bn_nrdma_20260930.log`（Orange Pi 5 Plus，RKNPU driver v0.9.8）。

- 发射器：`experiments/min_conv_exp.c`（新增 `--bn-table-f32-bias` / `--bn-table-f32-step`）
- 探针：`experiments/run_fp16_bn_nrdma_probe.sh`（对 ETIMEDOUT 自动重试）
- 结论：fp16 下 `BN_CFG=0x20150` + `NRDMA_CFG=0x2` + `BN_BASE_ADDR`(weight BO) 的逐通道表是
  **fp32**（`+1.5f` 重解释为 int32 会 +1.07e9，实测 BIT-EXACT 证伪），在 fp32 累加器上按输出
  通道相加后再 fp32→fp16 输出。1×1 / 3×3 / 5×5、单组与多组（oc=24）、int8 对照全部 BIT-EXACT。
- 现象：fp16 带表 DPU 提交 **奇偶交替 ETIMEDOUT**（`ret=0/-1/0/-1…`，日志末演示 6 连）；
  重试即消，判为环境脆弱（同 §7.20/§7.21），非语义错。
