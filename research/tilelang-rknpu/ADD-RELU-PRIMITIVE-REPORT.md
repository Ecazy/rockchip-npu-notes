# MLP Add + ReLU primitive 融合报告

日期：2026-10-28。控制基线：`963fd0d` 的 Submit 分组融合、预算化 MLP plan
和缓存策略。新臂只改变 primitive 融合，**不重复计算**此前 8→3 Submit 的
收益；本包中控制臂和新臂都是 3 Submit。

## 1. 结论

采用独立 primitive 模板 **`co.ew_add_relu.f16.v1`**：在同一条 DPU EW
ADD 路径上清掉 `EW_RELU_BYPASS`（`DPU_EW_CFG` bit 9），
`DPU_EW_RELUX_CMP_VALUE` 保持 0，得到 `out = relu(x + operand)`。

- 仅当 Add 中间值只有 ReLU 一个消费者、未被导出、动作/布局条件成立时，
  由通用 lowering 规则替换；
- 保留原 `co.ew_add.f16.v1` + `co.ew_max.f16.v1` 两 task 路径，
  `fuse_add_relu=False` 可完全回退；
- 没有用 `max(x, -b)` 替代完整 add+relu；
- 没有改变 FP16 收窄、累加、输入域或数值门限；
- 没有新增模型名分支。

默认采用 `fuse_add_relu=True`。

## 2. 硬件路径与数值分析

原路径：

1. `EW_ADD`：identity-conv main 得到 `x`，ERDMA operand 为 bias，
   EW ALU ADD 得到 `fp16(x + b)`；`EW_RELU_BYPASS=1`。
2. `EW_MAX`：identity-conv main 重新读取该 fp16 值，EW ALU MAX 与 0
   比较，写出 ReLU 结果。

新路径：

- 同一个 identity-conv main、同一个 ERDMA operand、同一个 16-bit
  `EDATA_SIZE`；
- 只把 `EW_RELU_BYPASS` 清零，EW ALU 仍为 ADD；
- ReLU comparator 在 EW stage 内、OUT_CVT 之前生效；
- `DPU_EW_RELUX_CMP_VALUE=0`。

因此差异只可能来自“先写回 fp16 再 ReLU”与“EW stage 内先 ReLU 再
OUT_CVT”的边界行为。窄域真机探针没有使用 tolerance，直接比较输出 bit：

- 8192 个边界/随机/特殊输入：**0 mismatch**（finite 0，special 0）；
- 1,000,000 个元素：**0 mismatch**；
- 覆盖 signed zero、min/max subnormal、min normal、max finite、精确抵消、
  近抵消、随机 finite，以及单独 NaN/Inf；
- 所有 signed-zero 组合两路都输出 `+0x0000`；
- 非法组合（`ew_relu` 且无 accumulate / MAX / MUL）均由生成器拒绝。

探针使用真实 NC1HWC2 feature/operand cube、identity weight、`M>=12`、
`M%4`、CBUF guard padding；只读有效行。8192 元素两 task 控制 438.7 µs，
单 task 融合 324.3 µs；1M 元素 29,176 µs vs 26,140 µs。

## 3. 结构变化（MLP b32）

| 指标 | 控制（Add+Max） | 融合（Add+ReLU） |
|---|---:|---:|
| NPU tasks | 11 | **9** |
| Submits | 3 | 3 |
| data BOs | 19 | **15** |
| transfers | 16 | **14** |
| peak payload | 937,984 B | **888,832 B** |
| reserved | 1,561,792 B | **1,381,312 B** |
| command workspace | 98,304 B | **81,920 B** |
| device request | 1,660,096 B | **1,463,232 B** |
| Host estimate | 1,042,860 B | **1,001,660 B** |

b1/b8 同样为 11→9 task、19→15 data BO、16→14 transfer。减少的 2 个 task
是两处 ReLU；减少的 4 个 data BO 包含两个 Add 中间 BO 和两个 ReLU 零
常量 BO。Submits 保持 3，证明没有重复计算上一包的分组收益。

多提交缓存下：

| arm | cached | command request | command physical | Host delta | fallback command request/physical |
|---|---:|---:|---:|---:|---|
| control | 3/3 | 11,088 B | 16,384 B | 192,488 B | 98,304/98,304 B |
| fused | 3/3 | 9,072 B | 16,384 B | 159,240 B | 81,920/81,920 B |

## 4. 实测

### 孤立 MLP（battr）

| arm | total µs | submit_wait µs | output_sync µs | guard µs | pack µs |
|---|---:|---:|---:|---:|---:|
| control b32 | 602.6 | 253.2 | 21.0 | 157.5 | 156.3 |
| fused b32 | 548.0 | 232.2 | 15.5 | 129.8 | 156.3 |
| control b1 | 438.1 | 244.4 | 19.2 | 159.2 | 10.8 |
| fused b1 | 389.4 | 226.6 | 14.0 | 133.0 | 11.1 |

### 完整模型

两轮独立 interleaved，256 images / 7 rounds，同 fd、同 multi-submit-cache
策略；ratio 合并两轮 14 个 round。

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.0558×** | 1.0511× | 1.0595× | 1059.0→1002.4 | 1060.3→1006.7 | yes |
| mlp_b8 | **1.0783×** | 1.0714× | 1.0827× | 707.6→656.2 | 710.8→660.6 | yes |
| mlp_b1 | **1.0957×** | 1.0936× | 1.1260× | 607.5→555.0 | 506.5→452.7 | yes |
| vit_b8 | 1.0010× | 0.9985× | 1.0100× | 3366.1→3366.5 | 3081.0→3075.9 | yes |
| vit_b1 | 1.0073× | 1.0040× | 1.0990× | 1457.4→1445.6 | 1472.8→1462.1 | yes |

MLP 两轮均正收益。ViT 整体未见稳定变化；其反转轮原因尚未定位，不归因于
进程噪声。准确率一致，所有 logits 逐位相同。

## 5. 验收

- 新增 `test_epilogue_fusion` 两项：Submit 分组融合与 Add+ReLU primitive
  融合；后者断言 11→9 task、19→15 data BO、16→14 transfer、同 Submit 数，
  mock 执行逐位；
- CTest：**33/33 passed**；
- pytest：**358 passed, 71 subtests passed**；
- `rk_plan_host_test`：**875 checks / 0 failures**；
- `command_cache_test`：**270 checks / 0 failures**；
- fused plan package 冻结数值复验：**PASS / 0 problems**；
- 真机窄域探针：8192 和 1,000,000 元素均 0 mismatch；非法组合全部拒绝。

## 6. 身份与历史表述修正

- 结果目录：`results/20261028T075310Z-add-relu-primitive/`
- runtime SHA-256：
  `e5eeeb4e510051ca0d36e39f1443a6828f6597804e7ff3d4f74d27c80e79c61e`
- runtime probe SHA-256：
  `9bd3198ca5cd8c7c40e453a7641601ce33152d6987d9f498c7ef3d814615c009`
- primitive probe SHA-256：
  `a1026160102b10feeb2d35ca14c164defdab837baafa75e3488f9452890a8346`

同时修正上一包表述：上一阶段准确称为 **Submit 分组融合**，8→3 Submit、
仍是 11 task；未新增 primitive 寄存器模板，只新增了合法链组合规则。
ViT 的反转轮原因尚未定位，统一表述为整体未见稳定变化，不归因于进程噪声。
