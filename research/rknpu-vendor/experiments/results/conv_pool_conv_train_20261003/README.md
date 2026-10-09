# conv → pool → conv 三层**量化感知训练微步**（§7.193，Task 6 Step 3）

日期：2026-10-03
环境：Orange Pi 5 Plus / RK3588 / kernel `6.1.43-rockchip-rk3588-rknpu098` / driver 0.9.8

## 目标
把 §7.192 的三层**推理**链升级为 int8 **训练微步**：在真实 NPU 中间激活（层1 conv → 层2 pool 的输出）
上跑层3 conv 的前向 + wgrad + `W′=W−round(dW/4)` **in-place 量化写回** + 复前向自证（`CLOSED-LOOP OK`），
并刻画 int8 梯度**饱和 / 梯度缩放**（§7.41 闭环之上的「真实模型口径」）。

## 链路（`pipeline_conv_pool_conv_train.sh`）
| 步 | 工具 | 变换 | 数据来源 | 判据 |
| --- | --- | --- | --- | --- |
| 1 conv | `min_conv_exp` | `32×32×32 --k3p1--> 32×32×32` | 内部 pattern | `BIT-EXACT (floor)`，total=32768 |
| 2 pool | `min_pool` | `32×32×32 --k2s2p0--> 16×16×32` | 层1 输出（`--in-file --in-yiw`）| `mismatch=0` + 独立 Python maxpool BIT-EXACT |
| 3 conv | `wgrad_real` | `16×16×32 --k3p1--> 16×16×32` | 层2 输出（x 主序 → 行主序，**组感知**）| 见下 |

## 实测（`run.log`）
```
层1 conv: total=32768 floor-mismatch=0 round-mismatch=0  RESULT: BIT-EXACT (floor)
层2 pool: total=8192 vs-max mismatch=0
独立参考: PY-REF total=8192 mismatch=0 max|diff|=0  BIT-EXACT
```

### 步 6 —— 工具自带 dY（**全饱和**基线）
```
FWD b=0:  BIT-EXACT (round)
WGRAD_TOTAL: coords=288/288 grads=9216/9216 round-exact maxdiff=0 sat=9216   ← 100% 饱和
UPDATE: dW_nonzero=9216/9216 grad_sat=9216 weights_changed=9216/9216 (same BO, in-place)
FWD2 b=0: total=8192 round-mismatch=0 vs CPU(W')  BIT-EXACT (round)
RESULT: CLOSED-LOOP OK (fwd exact, wgrad round-exact, update verified)
```

### 步 7 —— 稀疏 dY（密度 1/64，**消饱和**的量化感知训练步）
```
IN-DY-FILE: 读入 8192/8192 B
WGRAD_TOTAL: coords=288/288 grads=9216/9216 round-exact maxdiff=0 sat=0      ← 无饱和
UPDATE: dW_nonzero=7351/9216 grad_sat=0 weights_changed=7296/9216 (same BO, in-place)
FWD2 b=0: round-mismatch=0 vs CPU(W')  BIT-EXACT (round)
RESULT: CLOSED-LOOP OK (fwd exact, wgrad round-exact, update verified)
```

### 梯度缩放 → 饱和控制（同一真实激活 X，固定 requant `out_scale=8`/shift=17）
| dY | sat/9216 | 备注 |
| --- | --- | --- |
| 工具自带（合成）| **9216 (100%)** | 写回退化（全部 ±127）|
| `±3` 随机 | 7456 (81%) | round-exact + CLOSED-LOOP OK |
| `±1` 随机 | 4384 (48%) | 同上 |
| `±1` 稀疏 1/16 | 96 (1%) | 同上 |
| `±1` 稀疏 1/64 | **0** | 非退化训练步，`dW_nonzero=7351` |

⇒ **int8 wgrad 的梯度量级必须与固定 requant 标度匹配**；真实 int8 激活（±127）与合成 dY 组合会
让**全部梯度饱和**（写回退化为常量），须按梯度统计缩放 dY（等价于训练侧的梯度/损失缩放）。
无论饱和与否，wgrad 与 CPU 参考**逐位一致**、写回后复前向对 `CPU(W')` **BIT-EXACT**。

- 确定性：独立进程重跑，`conv1`/`pool.out`/`x_b0.bin` md5 完全一致
  （`304d5ff6…` / `261f79cd…` / `62d6732f…`）。
- 板卡：`job commit failed`=0；受控前后 `job timeout` 13→13（本链不触发超时，全核 idle）。

## 缺陷修复（顺带）
- `pipeline_pool_conv.sh` 的转置在 `IC>16` 时**漏掉通道组项** ⇒ 组1（通道 16–31）被写成组0 的副本
  （工具自带参考用同一 X 重算，故此前 `CLOSED-LOOP OK` 仍通过 —— 自洽但 X 非真实激活）。
  已改为组感知 `d[(c//16)*IW*IH*16 + x*IH*16 + y*16 + c%16]`；修复后仍 `CLOSED-LOOP OK`。
- `deep/i126` 为陈旧构建（无 `--in-dy-file`）：脚本改为现场 `gcc -O2 -DIW=16 -DIH=16
  -DIC=32 -DOC=32` 生成 `deep/wgrad_i32`，自包含且与几何一致。

## 意义
- 训练侧 int8 写回/缩放（§7.41）+ 外部梯度（§7.163）之外，**首次在真实 NPU 中间激活上闭合 int8
  训练微步**，并给出「梯度缩放 ↔ int8 饱和」的定量关系。
- 复现：`sh pipeline_conv_pool_conv_train.sh <工作目录>`。

复跑脚本统一维护于 [实验目录](../../pipeline_conv_pool_conv_train.sh)。请将脚本与所需程序放入板端工作目录后执行。
