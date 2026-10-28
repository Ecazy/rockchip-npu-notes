# MLP `Linear → bias → ReLU` 算子/编译优化报告

日期：2026-10-28。控制基线：`cad50a7`、预算化 MLP plan、冻结数值政策。
本轮只做算子/编译级优化，不扩展 runtime 缓存、预算、guard、同步或 Python
编排。

## 结论

采用通用 **Linear-epilogue Submit 分组融合**：在依赖直接、动作顺序相邻且
中间结果无其他消费者的条件下，把 `MM/MM_ACC` 链与后面的 `EW_ADD`、`EW_MAX`
放进同一个 Submit。MLP 从 **8 Submit / 11 task** 变为 **3 Submit / 11 task**，
task 模板、算术、buffers、tensor 布局、padding、传输和数值收窄全部不变。
本轮**没有新增 primitive 寄存器模板**；新增的是通用合法链组合规则及其
Python/C recipe 校验。

- 默认 `CompileOptions.fuse_epilogue=True`；
- `fuse_epilogue=False` 保留原 lowering，作为对照；
- 规则按 template/dependency/action-order 触发，没有模型名分支；
- 不新增硬件配方，不改变数值政策；
- 未采用任何会改变 FP16 收窄/累加顺序的融合。

## 候选比较与淘汰

| 候选 | 结构影响 | 数值影响 | 处置 |
|---|---|---|---|
| bias 融进 GEMM DPU BS stage | 去掉 bias EW task | bias 被移到最终收窄/累加之前，K-accum 顺序和 FP16 rounding 改变 | 淘汰；需新数值政策 |
| bias+ReLU 合成 `max(x, -b)` EW task | 去掉一个 task | `fp16(x+b)>=0` 与 `x>=-b` 在 rounding 边界不等价 | 淘汰；需近似政策 |
| 改 K tiling / GEMM template 减少 task | 减少 task/中间访问 | 改变每 K tile FP16 writeback 和累加语义 | 淘汰；当前 K=256 是控制基线 |
| 通用 Submit 分组融合 | 8→3 Submit，task 不变 | 完全不变 | **采用** |

硬件具备 DPU BS bias 和 EW max/min 能力，但“硬件可融合”不等于与现有
“MatMul FP16 写回 → FP16 bias add → FP16 ReLU”逐位等价。任何改变
收窄/累加方式的方案都必须单独提交并重新验收，本包不默认启用。

## 通用规则与安全边界

融合仅在所有条件满足时发生：

1. 后一个 Submit 只直接依赖前一个 Submit；
2. 后一组 task 的第一个动作直接依赖前一组最后一个 task；
3. 两组 task 在 plan action 顺序中连续，没有被 host action 隔开；
4. 前一组最后一个 task 没有其他消费者；
5. 合并后的模板序列属于已登记链：
   - `MM` / `MM_ACC*` → 可选 `EW_ADD` → 可选 `EW_MAX`；
   - 或独立的 `EW_ADD` → `EW_MAX`。

C validator 对上述链再做一次独立检查，并要求每个 epilogue task 直接依赖
前一个 task；不满足时保持原 Submit，不猜测、不重排。

## 结构变化（MLP）

| plan | tasks | Submits control→fused | command workspace | device request | Host bytes | peak/reserved | data BOs | transfers | 数据路径一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b1 | 11 | 8→3 | 65,536→98,304 B | 1,518,528→1,551,296 B | 895,532→928,780 B | 829,184/1,452,992 B | 19 | 16 | yes |
| b8 | 11 | 8→3 | 65,536→98,304 B | 1,518,528→1,551,296 B | 921,292→954,540 B | 829,184/1,452,992 B | 19 | 16 | yes |
| b32 | 11 | 8→3 | 65,536→98,304 B | 1,627,328→1,660,096 B | 1,009,612→1,042,860 B | 937,984/1,561,792 B | 19 | 16 | yes |

融合 plan 为更大的最坏 fallback command workspace（6 task vs 4）多付
+32,768 B device request 和 +33,248 B Host；但 `bindings/constants/buffers/
views/actions/publish` 全部逐项相同，19 个 data BO、peak payload、allocator
slack、tensor physical rows/cols/bytes、upload/download 和 transfer 数均不变。

多提交缓存开启时：

| plan | cached | command request | command physical | Host delta | fallback command request/physical |
|---|---:|---:|---:|---:|---|
| control | 8/8 | 11,088 B | 32,768 B | 208,528 B | 65,536/65,536 B |
| fused | 3/3 | 11,088 B | 16,384 B | 192,488 B | 98,304/98,304 B |

即 cache command BO 从 8 个/32 KiB 降到 3 个/16 KiB，部分抵消 fallback
workspace 的增长；没有扩大 runtime 预算或改变缓存准入。

## 实测

### 孤立 MLP（battr）

| plan | total µs | submit_wait µs | output_sync µs | pack µs |
|---|---:|---:|---:|---:|
| control b32 | 548.2 | 185.5 | 24.9 | 158.1 |
| fused b32 | 499.3 | 139.7 | 22.2 | 158.1 |
| control b1 | 394.0 | 175.0 | 25.7 | 11.7 |
| fused b1 | 346.5 | 130.7 | 22.8 | 11.7 |

Submit wait 下降约 45 µs；pack/layout 基本不变。

### 完整模型（两轮独立 interleaved，256 images / 7 rounds）

同一 fd、同一 multi-submit-cache 策略；ratio 合并两轮共 14 个 round ratio。

| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |
|---|---:|---:|---:|---|---|---|
| mlp_b32 | **1.1413×** | 1.0498× | 1.2348× | 999.5→949.9 | 1307.1→1061.7 | yes |
| mlp_b8 | **1.0845×** | 1.0663× | 1.3616× | 647.6→597.0 | 644.3→598.5 | yes |
| mlp_b1 | **1.4185×** | 1.1038× | 1.4205× | 863.3→608.4 | 861.0→606.7 | yes |
| vit_b8 | 1.0026× | 0.9991× | 1.0061× | 3323.1→3313.7 | 3060.0→3053.6 | yes |
| vit_b1 | 1.0039× | 0.9075× | 1.0072× | 1433.2→1419.4 | 1394.3→1437.0 | yes |

准确率一致，所有 logits 逐位相同。ViT 默认图没有 Linear-epilogue 链，整体
未见稳定变化；其反转轮原因尚未定位，不归因于进程噪声。b32/b8 的 round
ratio 有跨轮波动，因此报告 median/min/max，而不只取单轮最好值。

## 验收

- 新增 `test_epilogue_fusion`：control 8 Submit / fused 3 Submit、11 task
  模板序列相同、transfers 相同、accounting 交换明确、mock 输出逐位；
- CTest：**33/33 passed**；
- pytest：**357 passed, 71 subtests passed**；
- `rk_plan_host_test`：**875 checks / 0 failures**；
- `command_cache_test`：**270 checks / 0 failures**；
- fused plan package 的冻结数值复验：**PASS / 0 problems**；派生 spec 只更新
  compiled-manifest identity，阈值和 expected arrays 未变。

## 身份

- 结果目录：`results/20261028T071025Z-epilogue-fusion/`
- 最终 board runtime SHA-256：
  `677163cb54409831e22fe1c4f7d00c1f931b94a3af324b5b4cb6b8f2c014762b`
- probe SHA-256：
  `c345cc114e66532de92c17fe6aedbb78f2f6ff2046bf6cf5b5b6d49a7bd1ccb3`
