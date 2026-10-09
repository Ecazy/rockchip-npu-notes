# lut_fit_20261002（路线图 Task 4，§3b-4②）

数据：2026-10-02 板上重抓（libhook_probe.so + EPERF_RKNPU_REGCMD_MAX_BYTES=4096）：
`sigmoid_1x16_fp16`（439 唯一值）、`tanh_1x16_fp16`（442）、`f_Softmax_1x64_fp16`（110）。
源 trace 52006/52009/79418 B 未归档（表值已入 CSV：`table_<name>.csv` = idx/raw_u32/value_abs）。

## 拟合结果（fit_summary.json；候选：近指数 log 域二次 vs 分段线性@5% 跳变）

| 表 | n | quad-log (A, b, c) | rmse_log | 分段@5% | AIC quad / piecewise | 胜者 |
| --- | --- | --- | --- | --- | --- | --- |
| sigmoid | 439 | A=e^-12.03, b=0.00212, c=3.31e-5 | 0.298（max 0.94） | 20 段 | -1057 / 5722 | quad-log（但 rmse 30% log ⇒ **两模型都不足**） |
| tanh | 442 | A=e^-10.41, b=-0.0146, c=9.41e-5 | 0.573（max 1.61） | 104 段 | -486 / 2584 | quad-log（同上，残差 ≫ fp16 精度） |
| fsoftmax | 110 | A=e^-15.06, b=0.0667, c=-3.73e-4 | 0.244（max 1.57） | **19 段** | -304 / -2246 | **分段线性，19 段 maxrel=2.2e-15（逐位精确）** ✓ |

## 结论

1. **fsoftmax（softmax 家族 111 项表）= 精确分段线性**：5% 相对跳变切出 **19 段**，段内线性残差
   2.2e-15（机器精度）⇒ **softmax LUT = 分段线性表（19 段）**，反查表 = 19 组 (起止 idx, 斜率, 截距)，
   压缩率 19/110 ≈ **17%** ✓（§3b-4 ②「平台-反查」在 softmax 族闭合）。
2. **sigmoid/tanh 表不是光滑指数**：log 域二次全局拟合 rmse 30–57%（max ≫ fp16 精度 4.9e-4），
   5% 跳变切段后 sigmoid=20 段、tanh=104 段仍残差大 ⇒ **表按写入序不构成单一光滑曲线**
   （0x4104 写序 = 跨 task 分块突发写，写序 ≠ 表索引序；§7.153 平台证据同样指向此）。
   **待办**：按 regcmd 的 LUT 地址寄存器（0x4100 区配置）恢复真实索引序后再拟合（下一会话）。
3. **attention 大表平台-反查（Step 2）＝ blocked**：attn_softmax 系模型 `inputs_set failed: -5`
   （多输入喂数，即 §3b-3/路线图 Task 3 的未解问题）；f_Softmax 单输入模型已给出方法论与基线。

复跑：`python3 fit_lut.py`（读 /tmp/opencode/tables.json，重抓配方见台账 §7.137 捕获要点）。
