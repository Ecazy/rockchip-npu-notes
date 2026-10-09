# lut_attn_20261003（路线图 Task 4 Step 2：attention 大表逐项拟合，§3b-4②）

数据：`../attn_fed_20261003/`（Task 3/§7.189 真实喂数语料，`EPERF_RKNPU_REGCMD_MAX_BYTES=8192`）。
工具：`python3 ../../lut_scan.py --fit attention [--csv out.csv] <trace.jsonl.gz>`。

## 表结构（`sdpa_qwen224` / `attn_softmax_h16_s196_d64_int8`，两模型一致）

| 量 | 值 |
|---|---|
| 整表突发 | 4 次 × **954** 条 `0x4104`（其余为 `0x4104=0` 旁路戳） |
| 条目 / 非零唯一 | 954 / **620**（621 个平台含首个 0） |
| 值域 | **1 .. 16063** |
| 写序 | **严格非降** ⇒ 写入序 = 表索引序（非分块乱序 ✓） |
| 最长平台 | 53（value=2 的游程） |

## 拟合：几何/指数表

`value[i] = round(A·exp(k·i))`（i=1..953，i=0 为 0 填充）：

* 网格最优 **A=1.45879, k=0.0097657** ⇒ mean|err|=**0.237 LSB**, max=3, 逐位精确 80.5%；
* **相邻比 `seq[i+71]/seq[i]` 均值 = 1.99824 ≈ 2** ⇒ 每 **71** 个索引翻倍 ⇒ **k = ln2/71 = 0.0097626**；
* **平台游程律** `run(v) ≈ (1/k)·ln((v+0.5)/(v-0.5))` 实测/预测（v: 实测/预测）：
  `2:53/52.3  3:34/34.5  4:26/25.7  5:20/20.5  10:11/10.2  100:1/1.0` ⇒ 指数采样 + 整数舍入 ✓✓。

⇒ attention softmax 大表 = **几何序列，公比 2 每 71 项**（“71 项基块翻倍 ~13 次”），
本质是 **exp 定点表**。

## 压缩率（§3b-4②）

* 原始：954 × u32 = **3816 B**。
* 指数模型：**2 参数 (A,k)**（+ 舍入规则）。
* 平台表：**621 组 (idx,val)**（无损，含全部 954 项）。

## 对账：与 fsoftmax 族不同

* `f_Softmax_1x64_fp16`（111 项，§7.174）= **精确 19 段分段线性**；
* attention（954 项，本条）= **光滑几何/指数**（无分段线性结构）。
⇒ “softmax” 在不同 lowering（单算子 vs attention 融合）下 LUT **形状不同**，可作算子族指纹。

## 输入无关性（§7.140 推广）

`sdpa ramp` vs `sdpa ones`：写次数、唯一值集合、首次出现顺序全同；输出不同
（`b65066c4…` vs `7360e2c1…`）⇒ 表为预编译常量。

## 复跑

```
python3 ../../lut_scan.py --fit attention \
  ../../../../experiments/results/attn_fed_20261003/attn_sdpa_fed.jsonl.gz
```

`table_attention_sdpa.csv` = idx,value（954 行，供离线拟合/绘图）。
`fit_summary.json` = 三份 trace 的完整拟合参数。
