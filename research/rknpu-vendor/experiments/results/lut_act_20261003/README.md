# lut_act_20261003（路线图 Task 4 Step 1：sigmoid/tanh 表逐项拟合，§3b-4②）

**结论：sigmoid/tanh LUT = 精确 Q15 激活表**，`y = round(32767·f(a·i+c))`，残差 ≤ 0.5 LSB
（99% 逐位精确）⇒ 推翻 §7.141 / §7.174.3 的「不是 S·σ / 落不到」结论（根因 = 当时未恢复
索引序、且 S 取值不当）。

## 数据
板上重抓（`rknn_infer_dump <model> ramp` + `libeperf_hook_rknpu_ioctl.so` +
`EPERF_RKNPU_REGCMD_MAX_BYTES=4096`），模型 `rknpu_atomic_probe/models/`：
`sigmoid_1x16_fp16`、`tanh_1x16_fp16`（各 12 个 regcmd 事件，4 个整表突发）。

## 关键观察：突发内写序 = 索引序
每个整表突发 443 条 `0x4104`，**单调非降**（sigmoid 值域 59..9640；tanh −32611..−12973）。
§7.174.3「写序≠索引序」是把 4 个突发合并统计的表象；**单突发内就是索引序** ✓。

## 拟合（`lut_scan.py --fit sigmoid|tanh`）

| 表 | 模型 | a | c | rmse | max | ≤0.5 LSB |
|---|---|---|---|---|---|---|
| sigmoid | `round(32767·σ(a·i+c))` | 0.012324 | −6.3100 | 0.147 | 0.52 | **99.1%** |
| tanh | `round(32767·tanh(a·i+c))` | 0.005898 | −3.0200 | 0.143 | 0.51 | **99.3%** |

* 尺度 **S=32767 = int16 max（Q15）**；加权最小二乘（w=(dy/dz)²）使自由 S 网格自动收敛到 32767。
* ⇒ 两表都是**定点激活查表**；443 项 → **3 参数 (S,a,c)** 或 443→(a,c)（S 常量）。
* 语义：sigmoid 表覆盖 z∈[−6.31, −0.87]（442 步 × 0.01232）；tanh 表覆盖 z∈[−3.02, −0.42]
  （仅**负半轴**，对称表 ⇒ 正半轴由奇对称补齐）。

## 与 attention 大表对比（§7.190）
* sigmoid/tanh（443）= **精确解析激活**（Q15，可闭式重建）；
* attention（954）= **几何/指数**（公比 2 每 71 项，非激活，属 softmax exp 表）；
* fsoftmax（111）= 19 段分段线性。
⇒ 三类表形状可作**算子族指纹**。

## 复跑
```
python3 ../../lut_scan.py --fit sigmoid --csv table_sigmoid.csv sigmoid_tr.jsonl.gz
python3 ../../lut_scan.py --fit tanh    --csv table_tanh.csv    tanh_tr.jsonl.gz
```
`fit_summary.json` = 两表完整参数；`table_*.csv` = idx,value。
