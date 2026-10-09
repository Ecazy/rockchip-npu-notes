# CNA 输入 tile 的真包线：CBUF bank（DATA_BANK × 32 KB）+ FEATURE_GRAINS（§7.31，修正 §7.30）

> **本节修正 §7.30。** §7.30 的「单次 submit 的 CNA 输入 feature tile 硬上限 = 32768 B」
> 只是**旧发射器硬编码 `DATA_BANK = 1`** 的表现；「≤32 KB 内高输入亦失效」的次级约束则是
> **`FEATURE_GRAINS` 写死 `50+stride+1`** 的表现。两者都是**配置**，不是硬件硬包线。
> 真包线只有 CBUF 总容量：**输入 tile bank + 权重 bank ≤ 12**。

结论来源：板端 Orange Pi 5 Plus，RKNPU driver v0.9.8（`6.1.43-rockchip-rk3588-rknpu098`）。
复跑：板端 exp2 目录 `./run_fp16_cbuf_bank_probe.sh`（`CBUF_BANK_LOG=...` 覆盖日志名）。
判据：fp16 `BIT-EXACT` / `≤16 ULP`（fp64 参考）；int8 `BIT-EXACT (floor)`。

## vendor trace 旁证（离线，`analyze_vendor_cbuf_banks.py --all`）

仓库内 17 份 hook trace / 85 个 regcmd task 的 `CNA_CBUF_CON0` 分布：

| trace | task | DATA_BANK | WEIGHT_BANK | grains | 备注 |
| --- | --- | --- | --- | --- | --- |
| `attn_relu_vendor.jsonl`（vendor） | 12 | **10** | **2** | 17 | 1×1 matmul 形态（DATAIN 1×22×16、`WEIGHT_SIZE0`=65536 B = 2 bank） |
| `bn_lower_20260928/em0d/tr_transpose_p0321_fp16.jsonl` | 6 | **11** | **1** | 2 | transpose 形态（`WEIGHT_SIZE0`=8192 B = 1 bank） |
| 其余（含 vendor `tr_rect*`、`conv16*`、`conv_bn_conv_*`） | 67 | 1 | 11 | 5–20 | 小输入，默认分配 |

⇒ **`DATA_BANK`/`WEIGHT_BANK` 是 vendor runtime 真正在用的规划旋钮**：输入按字节取 bank
（可达 10/11），权重拿 `12 − DATA_BANK`（上表两例恰好分别容纳 2 bank / 1 bank 的权重描述符）；
旧发射器恒写 `DATA_BANK=1` 才是 §7.30 「32 KB 硬包线」的来源。

⚠ vendor 的 `FEATURE_GRAINS` 实测只有 **2–20**（随几何变化），远小于本文采用的
`max(80, …)`。⇒ `max(80, ⌈2·32768/row_bytes⌉+1)` 是**充分但保守**的可用口径（本板全形状验证）。
**vendor 的精确规则已在 §7.32 反解并验证：`grains = ih + kh + pad_top`（85/85 真实 trace 精确吻合）**，
见 `../vendor_grains_rule_20260930/`；发射器 auto 现取两者上包络。旧发射器的 `52` 只在小 feature 上恰好够用。

## 结论

**① 输入 feature tile 上限 = `DATA_BANK × 32768 B`。**
`CNA_CBUF_CON0` 的 `WEIGHT_BANK = 12 − DATA_BANK`，CBUF 共 12 个 32 KB bank。旧发射器
（及 mesa 的 `input_banks = 1`）恒写 `DATA_BANK=1` ⇒ 恰好 32768 B 的「硬」边界。

| 输入字节 | 需要 bank | `DATA_BANK=1` | `DATA_BANK=ceil(in/32768)` |
| --- | --- | --- | --- |
| 32768 B（`ic128 iw16 ih8`） | 1 | ULP-EXACT | ULP-EXACT |
| 36864 B（`ic128 iw16 ih9`） | 2 | MISMATCH（max\|diff\| 240） | ULP-EXACT |
| 65536 B（`ic128 iw16 ih16`） | 2 | MISMATCH（545.6） | ULP-EXACT |
| 98304 B（`ic128 iw16 ih24`） | 3 | MISMATCH（545.6） | ULP-EXACT |
| 131072 B（`ic128 iw16 ih32`） | 4 | MISMATCH（545.6） | ULP-EXACT |

**② `FEATURE_GRAINS`（`CNA_CONV_CON2[13:4]`，10 bit）必须随输入“行字节”放大。**
旧值 `50 + stride_y + 1`（mesa magic）在**高输入**下静默算错——这正是 §7.30 的「次级约束」
（fp16 `iw16 ic16`：53 行净、54 行错，且与 32 KB 无关：54 行仅 27648 B）。采用
[`refs/tinygrad`](../../../refs/tinygrad/tinygrad/renderer/rockchip.py) rockchip 后端的可用口径

```
grains = max(80, ceil(2·32768 / row_bytes) + 1)      # 取偶，上限 1022（字段 10 bit）
row_bytes = iw × (对齐通道 × 每元素字节)              # = 一行 feature 的全部通道字节
```

后全部转正（`--no-auto-cfg` 仅切 grains 的对照臂）：

| 配置 | `grains=52`（旧） | `grains=auto` |
| --- | --- | --- |
| fp16 `iw16 ic16 ih54`（row 512 B, auto=130） | MISMATCH（7.3） | ULP-EXACT |
| fp16 `iw16 ic16 ih64` | MISMATCH（15.4） | ULP-EXACT |
| fp16 `iw16 ic16 ih128` | MISMATCH（15.4） | ULP-EXACT |
| fp16 `iw16 ic16 ih200` | MISMATCH（16.5） | ULP-EXACT |
| fp16 `iw8 ic16 ih100`（row 256 B, auto=258） | MISMATCH（14.7） | ULP-EXACT |
| fp16 `iw8 ic16 ih200` | MISMATCH（15.8） | ULP-EXACT |

注：`grains ≥ 1024` 会被 10 bit 字段截断（`--grains 2048` → 写入 0 → MISMATCH），故 auto 上限 1022。

**③ 真包线 = CBUF 总容量（12 bank）须同时容纳输入 tile 与权重描述符。**
`ceil(in_bytes/32768) + weight_banks_need > 12` 时单次 submit 放不下，必须切分。

- `ic128 iw16 ih64`：输入 8 bank + 权重 3 bank = 11 ≤ 12 ⇒ 单次 submit 即 ULP-EXACT（无需切分）。
- `ic256 iw16 ih40`：输入 10 bank + 权重 4 bank = 14 > 12 ⇒ 单发 MISMATCH（max\|diff\| 623）；
  `--tile-auto` 切成 30+10 行（最大 tile 31 行 = 253952 B = 8 bank）⇒ **ULP-EXACT**。

**④ 按输出行切分是互不相交的输出面，无需跨 submit 累加**（§7.30 的判断成立）。
每个 tile 把自己需要的输入行块 `[g0, g0+rows)` 拷成紧凑 stage（`pad_top` 相应下调，
中间 tile 为 0），输出 scatter 回全局布局。`--tile-oh N` 显式指定，`--tile-auto` 按
「可用 data bank × 32768」自动选 T。

**⑤ 发射器默认 `--auto-cfg`**：按 ① 自动取 `DATA_BANK = min(ceil(in/32768), 12 − 权重 bank)`，
按 ② 自动取 grains（`--grains` 仍可覆盖）。`--no-auto-cfg` 复现 §7.30 的旧行为（对照臂）。

## 不回归

| 套件 | 结果 |
| --- | --- |
| `run_fp16_regression.sh` | **pass=79 / fail=0 / envfail=0**（原 78/0/1） |
| `run_regression_hard.sh`（int8 硬化） | **74/74**（首跑 1 例随机 ETIMEDOUT，复跑全过） |
| `run_deconv_smoke.sh` | **12/0** |

## 证据（关键行）

```
## A) A/B
fp16 ic128 iw16 ih16 OLD (65536B)   OUT-OF-ENVELOPE  total=4096 mismatch=2354  max|diff|=545.566
fp16 ic128 iw16 ih16 AUTO           ULP-EXACT        total=4096 mismatch=130   max|diff|=4.768e-07
    CBUF-CFG: in_size=65536 B row_bytes=4096 B DATA_BANK=2 WEIGHT_BANK=10 GRAINS=80
fp16 ic16  iw16 ih64 OLD (32768B)   MISMATCH         max|diff|=10.996
fp16 ic16  iw16 ih64 AUTO           ULP-EXACT        max|diff|=9.537e-07
    CBUF-CFG: in_size=32768 B row_bytes=512 B DATA_BANK=1 WEIGHT_BANK=11 GRAINS=130
int8 ic160 iw16 ih16 OLD (40960B)   MISMATCH
int8 ic160 iw16 ih16 AUTO           BIT-EXACT (floor)
    CBUF-CFG: in_size=40960 B row_bytes=2560 B DATA_BANK=2 WEIGHT_BANK=10 GRAINS=80

## D) 真·容量越界
ic256 ih40 AUTO（无切分）           OUT-OF-ENVELOPE  max|diff|=623.064   DATA_BANK=8 WEIGHT_BANK=4
ic256 ih40 --tile-auto（30+10 行）  ULP-EXACT        max|diff|=4.768e-07 最大 tile 253952 B
ic128 ih64 AUTO（11 ≤ 12 bank）     ULP-EXACT        max|diff|=4.768e-07
ic128 ih64 --tile-oh 32（强制切分） ULP-EXACT        max|diff|=4.768e-07
```

完整日志：`cbuf_bank_probe.txt`（60 行，含 B/C/E 段全表）。

## 产物

- `experiments/min_conv_exp.c`：`--auto-cfg`（默认）/`--no-auto-cfg`、`--tile-oh N`、
  `--tile-auto`、`--tile-rows-max N`；`CBUF-CFG:` / `CBUF-PLAN:` / `TILE:` 信息行。
- `experiments/wgrad_real.c`：同一口径（自动 bank/grains + `--no-auto-cfg`）已移植到训练闭环工具；
  验证 fp16 单层 10/10 相位、`--two-layer` 宽扫描 180/180、int8 基线不回归。
- `experiments/analyze_vendor_cbuf_banks.py`：离线扫 hook trace，统计 vendor 的
  `DATA_BANK`/`WEIGHT_BANK`/`FEATURE_GRAINS` 规划（上表）。
- `experiments/run_fp16_cbuf_bank_probe.sh`（本节探针）。
- `experiments/run_fp16_deep_channel_probe.sh`：改为**历史对照臂**（全部命令带
  `--no-auto-cfg`，复现 §7.30 的旧配置）。
