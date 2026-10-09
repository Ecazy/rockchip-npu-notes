# FP16 训练侧 wgrad（CNA K 维归约）打通（§7.25，2026-09-30）

结论来源：`fp16_wgrad_probe.log`（板端 Orange Pi 5 Plus，RKNPU driver v0.9.8）。
复跑：`bash experiments/run_fp16_wgrad_probe.sh`。

## 结论

- **fp16 的 CNA K 维归约（wgrad dot）与 int8 同构**：把归约点放到输入通道（K 维）、
  把 `dY[p,oc]` 作为每个输出通道的权重，一次 1×1 CONV 得到 16 个输出通道的梯度。
  fp16 下唯一额外要求是 **`c.ic` 必须上取整到 32 的倍数**（见下）。
- **累加器 ≥ fp32**（§7.18 已定）：fp16 输入的 K 维乘积和按 fp32 累加，输出再
  fp32→fp16，故 wgrad 的求和精度不会退化到 fp16。
- 合成向量 `wgrad_x(p)` / `wgrad_dy(p,oc)` 与 batch 累加（`sample=p/36`）均由
  CNA 内部 K 维归约完成，与 CPU 参考逐位一致。

## 决定性证据：fp16 `c.ic` 必须按 32 对齐

fp16 权重 ic 块是 32 元素（`ic2=32`），硬件不允许「半块」⇒ 当 `ic_real>16` 时
硬件把 `c.ic` 上取整到 32 的倍数。若发射器只按 16 对齐（旧行为），
`ic_real∈{33..48, 65..80, ...}` 会得到 `c.ic=48/80/...`（非 32 倍数）⇒ 第二个 ic 块
错位 ⇒ 全量 pattern MISMATCH。

| 用例（dot P=48） | `c.ic` | 结果 |
| --- | --- | --- |
| `--ic-pad 48`（非 32 倍数，旧行为） | 48 | **MISMATCH** |
| 默认 / `--ic-pad 64`（32 对齐，修复后） | 64 | **BIT-EXACT (fp16)** |

修复：`experiments/min_conv_exp.c` 中 fp16 分支 `c.ic = ic_real<=16 ? 16 :
ALIGN_UP(ic_real, 32)`（原为 16 对齐）。`ic_real≤16` 仍为 16（单个 16 原子块）。

## 覆盖（全部 `BIT-EXACT (fp16)`、`submit ret=0`）

- **A) dot 向量 P 扫描**：P=32/36/40/48/64/96（对应 `c.ic=32/64/64/64/64/96`）——
  32 与 64/96 是 32 倍数，36/40/48 走 32 上取整。
- **B) 3×3 conv wgrad 坐标矩阵**：`kx,ky∈{0,1,2} × ic∈{0,1,2}` 共 **27/27**
  （P=36，`--hi-zero`，仅 (0,0) 输入像素有效）。完整走了 kw=kh=3 的 tap 布局，
  验证 fp16 权重 tap 排列与 int8 §7.15 的 ky-major 一致。
- **C) batch 累加**：单坐标 (kx=1,ky=1,ic=1)，B=1/2/3（P=36/72/108，`c.ic=64/96/128`）。
- 全 42 例零 ETIMEDOUT（fp16 无表路径稳定）。

## 说明

- 探针：`experiments/run_fp16_wgrad_probe.sh`（`--wgrad-dot` / `--wgrad-conv3x3`
  / `--wgrad-batch`，统一 `--fp16 --f16-pix-elems 8`）。
- 输入打包需 **`--f16-pix-elems 8`**：否则 `fill_input` 按 `atom=c.ic` 打包退化
  （只前 8 通道生效），只有单平面用例「恰好」通过，不能作为真实多 ic 证据。
- 本 probe 仅覆盖 **单卷积的 K 维归约**；真实训练闭环（写回 + 梯度缩放，跨层累加）
  仍待 `wgrad_real.c` 的 fp16 移植（后续）。
- int8 对照（§7.4/§7.5）不回归：`run_wgrad_dot_probe.sh` / `run_wgrad_conv3x3_matrix_probe.sh`
  / `run_wgrad_batch_probe.sh` 结论不变。
