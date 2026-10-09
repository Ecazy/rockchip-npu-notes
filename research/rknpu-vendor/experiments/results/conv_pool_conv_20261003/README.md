# conv → pool → conv 三层 NPU 管线（§7.192，Task 6 Step 2）

日期：2026-10-03
环境：Orange Pi 5 Plus / RK3588 / kernel `6.1.43-rockchip-rk3588-rknpu098` / driver 0.9.8

## 目标
补「conv → pool → conv」三层配方与闭环校验，与既有「pool → conv」（§7.156/§7.161/§7.162）互补。

## 几何与链路
| 层 | 工具 | 变换 | 布局 |
| --- | --- | --- | --- |
| 层1 conv | `min_conv_exp` | `16×16×32 --k3p1--> 16×16×32` | 输出 y 主序（`npu_off`）|
| 层2 pool | `min_pool` | `16×16×32 --k2s2p0--> 8×8×32` | 输入 x 主序；`--in-yiw` 把 conv 输出转置进池化输入 |
| 层3 conv | `min_conv_exp` | `8×8×32 --k3p1--> 8×8×32` | `--in-file --in-xiw` 把池化输出转置进硬件 y 主序 |

## 新增/修改（opt-in）
1. `min_pool --in-file F` / `--in-yiw`（§7.192）：从文件读输入；`--in-yiw` 表示文件为卷积输出 **y 主序**，转置到池化输入 **x 主序**。
2. `min_pool` 的 `--out-file` 输出 dump **移到 from-device `bo_sync` 之后**：旧版在同步前 dump，读到未失效的 CPU 缓存（本机复现：同输入三次运行，组1输出 3 个不同 md5，且与工具自带参考 `mismatch=0` 矛盾）。修复后确定。
3. `min_conv_exp` int8 CPU 参考在 `--in-file` 时改读 **BO 真实输入**（`g_ref_in_base`）而非内部 `pat_in`（§7.192）：使 conv 在**外部输入**上可定量自证（此前 §7.158 的定量盲区由此闭合）。需配 `--pattern`（权重=pat_w）。

## 实测结果（run.log）
```
层1 conv: OUT-HASH 3049e11e  total=8192 floor-mismatch=0  RESULT: BIT-EXACT (floor)
层2 pool: total=2048 vs-max mismatch=0
层3 conv: OUT-HASH 8c38d6a1  total=2048 floor-mismatch=0  RESULT: BIT-EXACT (floor)
独立参考(Python 2x2 maxpool of conv1): total=2048 mismatch=0  BIT-EXACT
确定性: conv1/pool.out/conv2 两次独立进程运行 sha256 相同
sha256(conv2) = f6f77c79a8162a008e0c65c60ee4bae2a18f7eb01491413537d2dbe61aeb4e58
```
- conv1 用硬回归套件同款 pattern 旋钮（`--pattern --pat2 --asym-w --pad-con1 0 --auto-fix`）；裸 `--pattern` 会落入已知 tap 置换盲区（63/8192 mismatch），非管线缺陷。
- 修正 `min_pool` out-file 竞态后，三层均可复现（前一轮 pool/conv2 非确定即该竞态所致）。

## 回归
- int8 `run_regression_hard.sh`：pass=74 fail=0 envfail=0
- fp16 `run_fp16_regression.sh`：pass=79 fail=0 envfail=0 dirtyevents=0
- 板卡：`job commit failed`=0，`job timeout` 无新增。

## 复现
```
sh pipeline_conv_pool_conv.sh <工作目录>
```

复跑脚本统一维护于 [实验目录](../../pipeline_conv_pool_conv.sh)。请将脚本与所需程序放入板端工作目录后执行。
