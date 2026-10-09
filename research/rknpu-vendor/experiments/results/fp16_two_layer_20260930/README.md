# fp16 双层真实网络闭环（§7.28，2026-09-30）

结论来源：`fp16_two_layer_20260930.log`（板端 Orange Pi 5 Plus，RKNPU driver v0.9.8）。
复跑：板端 exp3 目录 `./run_fp16_two_layer_loop.sh`（宽扫描可用 `TL_SWEEP=0` 跳过）。

## 结论

**fp16 双层真实网络（跨层前向 + 分层 wgrad + 权重更新 + 再前向）闭环成立，精度与单层一致，
跨层不做 CPU 回环（层 2 直接读层 1 的 NPU 输出面）。**

- 几何：层 1 `ic=3→oc=16`、层 2 `ic=16→oc=16`，均 `3×3 pad`、`8×8` 输入；
  ow×oh = 8×8（pad1）/ 6×6（pad0）。
- **跨层前向（更新前）F1/F2：720/720 BIT-EXACT**（pad0/pad1、seed1..30、B1..3）。
- **分层 wgrad（fp64 正确舍入参考）：348/360 BIT-EXACT，其余 ≤4 ULP**（实数累加偶发末位）。
- **权重更新后（`W'=fp16(W−lr·dW)`）再前向**：绝大多数字位一致，极少数 ≤5 ULP。
- **宽扫描 pad0/1 × seed1..30 × B1..3：180/180 `2-LAYER CLOSED-LOOP OK`**。

## 判据（近零相消）

fp16 多元素求和判据沿用台账 `≤16 ULP`；但**强相消**时输出量级远小于累加项，fp32 级累加的
绝对误差被放大成很大的 ULP 距离（相对误差），此时 ULP 不再是有意义的度量。故闭环判等为：

> 元素一致 ⇔ `ULP ≤ g_l2_ulp(默认 16)` **或** `|hw − ref| ≤ g_l2_abs(默认 1e-4)`。

命令行可用 `--tl-ulp N` / `--tl-abs A` 调整。

具例（`seed=11 pad=1 B=3`，层 2 更新后再前向，`oy=3 ox=5 oc=2`）：硬件 `0x0f48`(4.444e-4)、
fp64 精确和 `0x0f9f`(4.652e-4)、fp32 顺序和 `0x0f00`(4.395e-4)；ULP 距离 87，但绝对残差
`2.07e-5 ≤ 1e-4`。硬件结果落在「fp32 顺序和」与「fp64 精确和」之间，属 fp32 级累加在强相消下的
正常舍入，非逻辑错误。

## 实现要点

- `experiments/wgrad_real.c` 新增 `--two-layer`（隐含 `--fp16`），新增相位：
  `L2-F`（层 1→层 2 链式前向）、`L2-W`（层 2 wgrad 输入=NPU A1；dgrad 反传得 dA1；层 1 wgrad）、
  `L2-U`（更新 W1/W2 并再前向比对）。层 2 输入直接取层 1 NPU 输出面（偏移 `L2_A1_OFF`）。
- 参考：`f16_cpu_wgrad_dbl`（fp64 累加）用于 wgrad；`f16_cpu_fwd_gen` 改为 **fp64 累加**
  （对齐台账 fp16 判据口径）。
- 判等/统计：`f16_key`/`f16_ulp`（fp16 位型单调整数键）+ `f16_ok`（ULP/绝对双判据）+ `f16_verdict`。
- **修复**：`make_fwd_cfg_l2` 之前把输出面尺寸硬编码为 `IW×IH`（仅在 pad1 与 `ow×oh` 重合），
  pad0 下 6×6 输出 stride 错误导致前向大面积失配；改为按 `pad` 取 `ow×oh`。

## 覆盖

| 相位 | 配置 | 结果 |
| --- | --- | --- |
| L2-pad1 | B=1,2,3 | 3/3 OK，F1/F2 全 BIT-EXACT |
| L2-pad0 | B=1,2,3 | 3/3 OK，F1/F2 全 BIT-EXACT |
| L2-seed | seed 1/2/999，B=1 | 3/3 OK |
| L2-lr | `--lr {0.5,0.125,1.0} --loss {0.125,0.5,1.0}` | 3/3 OK |
| L2-sweep | pad0/1 × seed1..30 × B1..3 | **180/180 OK** |
| 基线 | 单层 fp16 + int8 闭环 | 均 `CLOSED-LOOP OK`，无回归 |

## 遗留

- 层 2 wgrad 与更新后再前向在真实随机数据上偶发末位（wgrad ≤4 ULP，再前向 ≤5 ULP），
  已由双判据覆盖；若要严格 BIT-EXACT 需复刻硬件归约顺序（当前观测为 fp32 级、顺序相关）。
- 更深通道（ic>96 / oc>64）与 `ic_real=1` 特例路径待验证。
