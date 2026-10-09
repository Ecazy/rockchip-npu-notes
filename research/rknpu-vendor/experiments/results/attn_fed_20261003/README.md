# attn_fed_20261003 — Task 3: attention `.rknn` 真实喂数（§7.189）

目标：打通真实多输入 attention `.rknn` 的喂数（旧 `rknn_infer_dump` 报
`inputs_set failed: -5`），喂入 **真实/非 ramp 输入** 后重抓 trace，对账
`0x4104`（DPU LUT）计数与表结构。

## 根因（-5）
`rknn_infer_dump.c` 用 `n = in_attr.dims[n_dims-1]`（仅末维=16）推元素数，且只设
`input[0]` ⇒ `param input size(64) < model input size(802816)`。
修复 = 用 `attr.n_elems` 逐输入、设全部 N 个输入（`pass_through=0`，API 量化）。

工具：`../../attn_feed.c`（新增）。board build：
```
gcc -O2 attn_feed.c -I<runtimes/mars-overhead/rknn/include> \
    -L<runtimes/mars-overhead/rknn/lib> -lrknnrt -o attn_feed
```
用法 `./attn_feed model.rknn <ramp|ones|zero> [out.bin]`。

## 模型（板卡 `/mnt/ssd/workspace/models/attention_probe/`）
| 模型 | in | out |
|---|---|---|
| `sdpa_qwen224.rknn` | 3（Q/K/V，各 `[1,196,64,16]` I8，NHWC） | 1（`[1,16,196,64]` NCHW I8） |
| `attn_softmax_h16_s196_d64_int8.rknn` | 1（`[3,196,64,16]` I8） | 1（`[1,16,196,64]` NCHW I8） |

SDK 2.3.2 / API 0.9.8；`inputs_set ret=0`、`run ret=0` ✓。

## 结果（`lut_scan.py`，条目 8 B，读 `event=="regcmd"`）
| trace | 输入 | regcmd 事件 | `0x4104` 写 | 唯一值(全) | 唯一(非零) | 值域(非零) | 零旁路戳 |
|---|---|---|---|---|---|---|---|
| attn_sdpa_fed | sdpa ramp | 3078 | 3985 | 621 | **620** | **1..16063** | 173 |
| sdpa_ones | sdpa ones | 3078 | 3985 | 621 | 620 | 1..16063 | 173 |
| asm196_fed | softmax ramp | 3216 | 3979 | 621 | 620 | 1..16063 | 167 |

* 非零表项 **严格单调递增**（首 10 项 = 1,2,…,10）⇒ 定点整数表（§7.139 口径）。
* 整表加载 = **少数突发**（`rknpu_trace_report.py`：规模 954 ×4 次），其余为
  `0x4104=0` 旁路戳（§7.144 口径）。
* **输入无关性（§7.140 推广到真实多输入 SDPA）**：`sdpa ramp` vs `sdpa ones`
  ⇒ 写次数、唯一值**集合**、**首次出现顺序**三者全同；而 **输出不同**
  （`b65066c4…` vs `7360e2c1…`）⇒ 表是算子预编译常量，与实际数据无关 ✓。

## 输出 sha256（板上 exp2/）
- `attn_sdpa_out.bin` = `attn_sdpa_fed_out.bin` = `b65066c4f6d7f68ed6468f028182838935ca5bd737ca3052d69a4838ee067429`
- `attn_softmax_out.bin` = `asm196_out.bin` = `98cea31db24573e6fed046739209645548907768da3dec5d56a39a12582af2ea`
- `sdpa_ones_out.bin` = `7360e2c1da91d1fc5a1830dbcf07c579818ad618ab93718f4cd87eff2d40a718`

## 捕获命令
```
export LD_LIBRARY_PATH=<runtimes/mars-overhead/rknn/lib>:$LD_LIBRARY_PATH
EPERF_RKNPU_IOCTL_TRACE=<out>.jsonl EPERF_RKNPU_REGCMD_MAX_BYTES=8192 \
LD_PRELOAD=<rknpu-hook>/libeperf_hook_rknpu_ioctl.so \
  ./attn_feed <model>.rknn <mode> <out>.bin
```

## 附带
命中 `attn_sdpa_fed.jsonl` 的 trace 可继续用于 **Task 4 大表反查**（真实
attention 表 620 项、1..16063 已就绪）。
