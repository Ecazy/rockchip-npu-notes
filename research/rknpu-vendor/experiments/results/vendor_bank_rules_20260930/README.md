# CBUF bank 分配：vendor 口径核对 + 时序 + `WEIGHT_BANK` 下限（§7.38）

回答 handoff §8 的下一入口「用 vendor 真值核对 bank/grains 规划」。全部离线判据 +
板端时序/下限探针。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8。
复跑：`python3 experiments/analyze_vendor_bank_rules.py --all`（→ `bank_rules.txt`）；
板端 exp2 `bash run_bank_split_probe.sh`（→ `bank_split_probe.txt`）。

## 1. 规则核对（85 个 regcmd task，覆盖 17 份 trace）

| 判据 | 结果 |
| --- | --- |
| **R1** `FEATURE_GRAINS == ih + kh + pad_top`（§7.32） | **85/85** |
| **R2** `DATA_BANK >= ceil(in_bytes/32768)`（输入必装得下，§7.31） | **85/85** |
| **R3** `DATA_BANK + WEIGHT_BANK == 12`（CBUF 共 12 bank，两字段互补） | **85/85** |
| **R4** `WEIGHT_BANK == ceil(WEIGHT_SIZE0/32768)`（**vendor 口径**） | **18/18 真 vendor task**；0/67 Mesa 派生 |

**R4 甄别出的真 vendor task（18 个）**：

| trace / op | 权重描述符 | 观测 `WEIGHT_BANK` | `DATA_BANK` | 输入字节 |
| --- | --- | --- | --- | --- |
| `attn_relu_vendor.jsonl` op3（1×1，iw16 ih16 ic16） | 65536 B = 2 bank | **2** | **10** | 4096 B（int8，≪10 bank） |
| `bn_lower/em0d/tr_transpose_p0321_fp16` op1 | 8192 B = 1 bank | **1** | **11** | 128 B |

而同文件的 `op4`/`op6` 与其余 67 个 task 全是 **`DATA_BANK=1 / WEIGHT_BANK=11`** 的固定写法
（Mesa `rkt_regcmd.c` 的 `input_banks=1`），它们满足 R2（输入 ≤1 bank）但不符合 R4
⇒ **R4 可作为「trace 是否来自真 vendor runtime」的判据**。

**结论**：vendor 的规划优先级是「**权重按需拿 bank，剩余全给输入**」；我们的发射器是
「**输入按需拿 bank，剩余给权重**」。两者在「输入+权重都放得下」时等价（R2 是硬约束，
R3 是硬件事实），差异只体现在谁拿余量。

## 2. 时序：bank 划分**不是**性能杠杆（本批规模）

同一 conv 只改 `--weight-bank`（8 次取 min `hw_elapse`）：

| 用例 | DB=4/8（输入按需） | DB=11（vendor 口径） | 差 |
| --- | --- | --- | --- |
| 3×3 `iw64 ih32 ic32 oc16`（输入 4 bank） | 52208 ns | 51624 ns | −1.1% |
| 3×3 `iw32 ih32 ic64 oc64`（输入 4 bank） | 117249 ns（DB=4） | 114916 ns（DB=9） | −2.0% |
| 1×1 `iw64 ih64 ic32`（输入 8 bank） | 80208 ns（DB=8） | 77875 ns（DB=11） | −2.9% |

差 ≤3%，远小于本板实测噪声（1.2–1.3×）⇒ **无系统性影响**；bank 划分应按「谁需要」分配，
不必为了性能调（对照：真正的性能杠杆是 §7.31–§7.36 的切分规划与 grains）。

## 3. 新定域：`WEIGHT_BANK` 过小 ⇒ submit **6 s 超时**（不是静默算错）

| 描述符 | `WB=1` | `WB=2` | `WB=3` | `WB≥4` |
| --- | --- | --- | --- | --- |
| 73728 B（3 bank，3×3 ic64 oc64 iw32 ih32） | **✗ 0/3** | ✓ 3/3 | ✓ 3/3 | ✓ |
| 147456 B（4.5 bank，3×3 ic64 oc128 iw8 ih8） | **✗ 0/3** | ✓ 3/3 | ✓ 3/3 | — |
| 663552 B（**21 bank**，3×3 ic192 oc192 iw4 ih4） | — | **✗ 0/3** | ✓ 3/3 | ✓ |

- 失败形态是 `SUBMIT ret=-1 ... Connection timed out`（6 s job timeout + soft reset），
  **不是**错误数值 —— 与我们发射器的静态判别不同，这是**驱动/硬件挂等**。
- 注意 21-bank 描述符配 `WEIGHT_BANK=11` 仍 **BIT-EXACT**（§7.36）⇒ 权重是流式的；
  但 **bank 给太少会挂**（下限随描述符增长，3/4.5 bank → ≥2，21 bank → ≥3）。
- 我们发射器的 auto 规则 `WEIGHT_BANK = ceil(wbytes/32768) + 1`（≥2）**恒安全** ✓
  —— 这也是历轮测试没撞上的原因。

## 产物

- `experiments/analyze_vendor_bank_rules.py`、`bank_rules.txt`（85 行逐 task 判据表）。
- `experiments/run_bank_split_probe.sh`、`bank_split_probe.txt`。
