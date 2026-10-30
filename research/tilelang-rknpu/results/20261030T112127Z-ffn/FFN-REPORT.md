# 真实 SmolLM FFN 的 schedule-aware 设备驻留数据流

日期：2026-10-30  
结果目录：`results/20261030T112127Z-ffn/`  
实现提交：`2d7e04bf6296d06a030164cd2b1563feee682a84`

本轮只扩展受限 SmolLM-shaped FFN：packed gate/up → SwiGLU → down → residual。  
未实现通用多节点调度器、不同 applied M subview、device repack、多核或跨 stage 流水。

## 1. 已有设备 FFN 与新增能力边界

已有：

- 单 tile 设备 SwiGLU FFN（`build_swiglu`）已存在，但只适用于 K/M/N 单 primitive tile、且 gate/up 是两个分开的 matmul。
- 通用 multi-node dataflow 已能执行 gate/up、Sigmoid、Mul、down、Add，但使用固定 256 tiling，且显式 `source_schedule` 被拒绝。
- LUT/EW 模板、identity 常量、BO 生命周期、C verifier 和 submit 组织均可复用。
- GRAPH-FUSION 的 device FFN 是 TinyTransformer 的既有成果；本轮没有把它重新包装成新增能力。

新增：

- `rknpu.staged_schedule=staged_mn_tiles_v1` 的六 launch FFN 源表达。
- packed gate/up 输出上的两个零拷贝 `view` IR 节点（gate 半、up 半）。
- `build_scheduled_swiglu_ffn` planner：复用现有 MM/LUT/EW 模板、identity、C verifier 和 submit 组织。
- 每 stage 独立 schedule：gate/up、Sigmoid、Silu、gated mul、down、residual 不共享全局 tile/order。
- 没有建立绕过 RKPlan/verifier 的执行路径。

## 2. 三个逻辑 stage 与实际 IR/task

逻辑：

1. packed gate/up projection；
2. SwiGLU（Sigmoid → gate*sig → gated*up）；
3. down projection + residual。

实际 RKIR 是 8 个 op：

```text
matmul(X, Wgu) -> Zgu
view(Zgu, 0, 0, M, H) -> G
view(Zgu, 0, H, M, H) -> U
sigmoid(G) -> S
mul(G, S) -> Silu
mul(Silu, U) -> A
matmul(A, Wd) -> P
add(P, X) -> Y
```

实际硬件是 6 个 `T.Kernel` launch：

```text
stage0  gate/up matmul, K=576 tail 64
stage1  sigmoid LUT
stage2  gate * sigmoid
stage3  gated * up
stage4  down matmul, K=1536 = 6x256
stage5  residual add
```

M=256 的 fixed 192 计划：81 tasks / 36 submits / 81 BO / 3 publish。  
M=1 的 fixed 192 计划：75 tasks / 36 submits / 81 BO / 3 publish。  
task 数大于 3 是因为 SwiGLU 展开为 LUT/Mul，且每个 output tile/K slice 都是独立 task；不是伪装成一个融合 kernel。

## 3. gate/up 配对、activation→down 映射和生命周期

- packed gate/up 的 N 维按 256 分块；gate tile 为 `(m0, n0)`，up tile 为 `(m0, H+n0)`，两者来自同一个 `Zgu` 值。
- view 节点明确记录 `row_offset=0, col_offset=0/H, rows=M, cols=H`；frontend 负例拒绝 up 半使用 gate 偏移。
- Sigmoid、Silu、gated mul 都以相同 `(m0,n0)` activation tile 工作；stage3 的 `up_source` 直接引用 producer tile `(m0,H+n0)`。
- down 的每个 K slice 直接引用已生产的 activation tile `(m0,k0)`；不存在 Host 拼接或另一种布局。
- Zgu、S、Silu、A、P 都是 workspace buffer；没有 upload/download action。只有 X 和 weights 上传、Y 下载。
- 每个 Zgu/S/Silu/A buffer 的 lifetime 覆盖其最后一个 consumer；C verifier 保留 slot reuse、alias、producer dependency 检查。
- identity LUT/EW 常量仍由 lowering 生成；不假设 workspace 私有持久状态。

## 4. Host参考、设备 declared reference、硬件三层误差

实验数值政策 `ffn-experimental-v1`：

- A：当前 production Host FFN：NPU gate/up → Host FP32 SwiGLU → NPU down → Host residual。
- B：设备 declared reference：每 K 块 FP16、device Sigmoid LUT 的声明插值、S/Silu/A/P/Y 每步 FP16。
- C：实际 RK3588 硬件结果。

误差（max abs，三个真实层 0/15/29）：

| 组间 | M=256 layer0/15/29 | M=1 layer0/15/29 |
|---|---:|---:|
| A vs B（Host exact vs device declared） | 0.03125 / 0.0078125 / 4.0 | 0.0078125 / 0.000122 / 4.0 |
| B vs C（declared vs hardware） | 0.03125 / 0.0078125 / 0.5 | 0.0078125 / 0.000122 / 0.5 |
| A vs C（Host exact vs hardware） | 0.03125 / 0.0078125 / 4.0 | 0.0078125 / 0.000122 / 4.0 |

- A-B 主要是 device LUT Sigmoid 及每步 FP16 收窄；B-C 是实现符合性误差；A-C 是用户可见差异。
- layer 29 的激活/输出范围最大（输出范围约 5608），4.0 的绝对误差对应约 0.07% 相对量级；中间层误差更小。
- 该路径没有替换默认 Host FFN；V5 门槛未修改。

## 5. 支持域、负例和板端证据

支持域：

- hidden=576，intermediate=1536，down output=576；
- M=1 decode 与 M=256 prefill；
- stage0 tile_n=256；所有 stage 相同 applied M；
- SwiGLU elementwise tile_n=256；
- down/residual tile_n 64 或 192（必须整除 576）；
- 中间 H 256 对齐，tail 空白由显式 guard/zero fill 处理；
- 不支持不同 applied M subview、任意 tile 重组、非 32 对齐中间 tail、device repack、多核、跨 stage 流水。

负例：

| 负例 | 检查层 | 结果 |
|---|---|---|
| gate/up 两半错配（stage3 读 gate 偏移） | frontend | `RK_E_UNSUPPORTED` “up half at column H” |
| stage4 rounding annotation 与源冲突 | frontend | `RK_E_NUMERIC_CONTRACT` |
| down tile_n 不整除 576 | frontend/planner | `RK_E_SHAPE` |
| 缺 activation producer dependency | C verifier | `RK_E_SCHEMA` |
| 中间 buffer 提前复用/alias | C verifier | 现有 slot/alias 检查 |
| 不支持的 global schedule | frontend | `RK_E_UNSUPPORTED` |
| 非闭环 FFN 图 | legalize/planner | `RK_E_UNSUPPORTED` |

板端：每个 M 7 个 device 候选，各跑真实首/中/末层；加上 A 对照后全部通过当前实验数值政策，M=1/M=256 均执行成功。

## 6. A/B/C 完整调用性能、资源与执行签名

完整 FFN 调用中位数（三层分别）：

M=256：

```text
A  Host SwiGLU + two NPU plans         34.381 / 45.142 / 46.209 ms
B  fixed device FFN, down 192          41.567 / 38.091 / 39.015 ms   sum 118.674
C  scheduled C_ewn_n192                40.939 / 40.805 / 38.675 ms   sum 120.419
C  down 64 geometry-distinct           43.855 / 53.069 / 52.698 ms   sum 149.623
```

M=1：

```text
A  Host SwiGLU + two NPU plans          4.202 / 3.798 / 3.735 ms
B  fixed device FFN, down 192           6.329 / 5.714 / 4.621 ms      sum 16.664
C  scheduled C_ewn_n192                  5.813 / 5.696 / 5.659 ms      sum 17.168
C  down 64 geometry-distinct             7.539 / 7.582 / 7.967 ms      sum 23.088
```

解释：

- A 与 C 同时改变 Host boundary 和数值路径，A/C 差异不能全部归因于调度。
- B 与 C 使用相同设备数值政策、相同 device-resident 边界；B/C 差异只来自 stage task/submit 顺序或 down tile 几何。
- down=64 候选是真正的执行几何差异（123 tasks/48 submits），但更慢。
- metadata-only 候选不作为 winner；执行签名见 `analysis.json`。

## 7. 调优确认与新进程重放

- 7 个 device 候选（每个 M）先做 C verifier preflight；全部通过。
- 候选按完整调用中位数和真实执行签名筛选；B fixed 192 的 aggregate 最优。
- 独立交错 B vs C_ewn_n192：
  - M=256：unpaired median speedup 1.099，但 paired median 0.979，mixed；
  - M=1：unpaired median 1.080，paired median 1.069；
  - 结论：C_ewn_n192 没有稳定胜出，**最终保留 B_fixed_192**。
- 新进程 replay：
  - B_fixed_192：M=256/M=1 均 pass，逐层 max abs 0.03125/0.0078/0.5 和 0.0078/0.000122/0.5。
  - C_ewn_n192：也 pass，仅作为候选对照。
- 没有把 noisy 的 C 候选升级为最终策略。

## 8. opt-in 模型集成、固定验收与建议

本轮完成独立 FFN 模块验收，**没有接入整模型**：

- 现有 model override 只支持 `single_matmul_schedule_v1` 替换单个 plan；
- 本 FFN 是 gate_up + host-SwiGLU + down 三个既有模型边界的复合替换，需要新的 pair/multi-op policy；
- 强行接入会改变 SmolLM 计算组织，超出本轮独立模块验收范围；
- 因此没有运行 V5 固定验收；V5 spec、门槛、calibration 和默认 Host FFN 均未改变。

建议：只有当实现一个显式的“FFN pair override”策略、保留全部 residual/RMSNorm/SiLU 语义并冻结独立校准后，才考虑 opt-in 模型集成。当前不建议把实验 device FFN 作为默认。

## 9. 代码、库、模型、策略与证据身份

- 实现提交：`2d7e04bf6296d06a030164cd2b1563feee682a84`
- board lib：`f9ed96ae2f1c38a941ac8a0e379026d8ef6e67ae9dcd62e4aafc11143d5d8419`
- package manifest：`b923fc5b427cc2ee6f0f88e7d096444a7ab44babdfbd79cc64fe07f5771548be`
- weights：`ff04cb6f25633772987ff8cfc66e65046004034e526cf1d6d6bb615333b945bf`
- analysis：`7a7eb37559c08c55dfedc4e9bc6ad4baf1da04556a94a3aed329bc9e28a81a60`
- policy：`74a071df63d6a4db047ff2e1b615cda3bc2ca95e47ebf9605983e3d2de53af0c`
- execution distinctness：见 `analysis.json` 的 `execution_distinct_from_B`
- 回归：Python 498 collected，0 failed，1 slow-test skip；C 33/33 ctest pass。

证据：

- `m256/manifest.json`、`m256/board-run.json`、`m256/confirm.json`、`m256/replay.json`、`m256/replay-c.json`
- `m1/manifest.json`、`m1/board-run.json`、`m1/confirm.json`、`m1/replay.json`、`m1/replay-c.json`
- `analysis.json`、`policy.json`
- `m256/plans/`、`m1/plans/`

## 10. 下一步

下一项是显式 **FFN pair/multi-op model override**：

1. 允许一个 policy 同时替换 `layer.*.gate_up.*` 与 `layer.*.down.*`，并保留 residual 语义；
2. 记录每个 layer 的 FFN 计划、输入/输出 binding、数值政策和有效执行身份；
3. 在独立校准集上冻结 `ffn-experimental-v1`，再做 V5 opt-in；
4. 在不改变默认 Host FFN 的前提下，对真实模型质量/延迟做 A/B/C 对照。

在此之前，不应把本实验 FFN 称为模型收益或默认路径。
