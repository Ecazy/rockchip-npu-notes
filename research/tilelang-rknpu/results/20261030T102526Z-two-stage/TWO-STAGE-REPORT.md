# 受限两节点 schedule-aware 执行闭环

日期：2026-10-30  
结果目录：`results/20261030T102526Z-two-stage/`  
实现提交：`01996a1d26cb2b5d5d746015bbf76566cc1ebadb`

本轮交付边界：只支持真实 TileLang 两个独立 `T.Kernel` launch 的
`Z = FP16_tiled_matmul(X, W1)`、`Y = FP16_tiled_matmul(Z, W2)`。Z 是
设备驻留 FP16 中间值；两个 GEMM 都保持升序 256 K 块和每块显式 FP16 收窄。
没有实现通用多节点流水、split-K、任意 BK、多核或新的物理布局。

## 1. 已有能力与新增连接点

已有：

- 多个 `T.Kernel` 的 staged frontend 已能提取 launch 边界、全局中间参数、SSA 依赖和固定 256 数据流。
- planner 已有设备中间 tensor 的 tile/value/action 映射、ping-pong accumulator、BO 生命周期和 C verifier。
- 单 GEMM 已有 guarded K-tail、显式 per-K FP16 和 M/N schedule contract。
- 旧的固定 256 多节点路径不能表达 K=576/H=576 的 guard tail，因此也不能直接作为本轮 A 组；
  A 组是同一受限新路径上的固定 256 控制，不把旧固定分块重新包装成新成果。

新增：

- `staged_mn_tiles_v1` 版本化 per-stage annotation；全局 schedule 属性被拒绝，不会复制到所有节点。
- 两 launch guarded GEMM extractor：每 stage 独立解析源 M/N、K 块、tail、舍入点和 schedule。
- `build_scheduled_two_stage` planner：只接受 `X->W1->Z->W2->Y`，Z 不上传/不下载，consumer A view 直接指向 producer Z buffer。
- stage 0 全部 submit 在 stage 1 之前；stage 1 task 显式依赖对应 Z producer action，并记录 producer/consumer region。
- C verifier 未放宽；同一个 C 计划校验检查 producer 依赖、view 范围、slot 生命周期和 alias。新增前端/planner 层负例。

## 2. 每 stage 调度语法和支持域

```python
T.func_attr({
    "rknpu.staged_schedule": "staged_mn_tiles_v1",
    "rknpu.stage0.tile_m": 256,
    "rknpu.stage0.tile_n": 256,
    "rknpu.stage0.mn_order": "m_major",
    "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
    "rknpu.stage1.tile_m": 256,
    "rknpu.stage1.tile_n": 128,
    "rknpu.stage1.mn_order": "n_major",
    "rknpu.stage1.source_k_rounding": "explicit_per_k_fp16",
})
```

支持域：

- stage 0 必须有 `tile_n == 256`，使每个 consumer 256-K slice 恰好映射到一个 producer Z tile。
- stage 1 的 applied `tile_m` 必须等于 stage 0 的 applied `tile_m`；否则 CUBE 物理行几何不同，当前不能安全 subview。
- **不同 source M tile 是允许的**：source 可用 128 行 block，再由 backend retile 到 consumer-safe 的 applied 64 行；这验证了 source tile 与 backend policy 分离。
- 中间 H 的 tail 必须 32 元素对齐；H=576 支持，H=257 明确拒绝，因为当前 CUBE view 不能暴露 1 元素的物理 tail。
- 每 stage K 仍固定 256、升序、每块显式 FP16 收窄；缺少显式舍入或 annotation 与实际源不一致都会被拒绝。
- 未知 stage 属性、stage index 越界、global `rknpu.tile_m` 等一律拒绝。
- 不支持任意 producer/consumer tile 重组、不同 applied M 的 row subview、设备 repack、N-stage schedule 传播。

## 3. Z 的 producer/consumer 映射实例

固定实验：M=65、K=576、H=576、N=97。选中的 A/fixed 256 计划中：

```text
Z producer view 15, producer action 17
producer tile: [m0=0, n0=0, rows=65, cols=256]
consumer region: [m0=0, k0=0, rows=65, kt=256]
consumer first task deps: [17, 12]
W2 upload action: 12
submit groups: 3 tasks each, dependencies 0<-1<-2<-3
stage0/stage1 boundary: all stage0 submits precede first stage1 submit
```

Z 的审计属性记录在每个 stage 1 task 的 `z_source`：

- logical producer/consumer region；
- producer view/action；
- row offset；
- logical K；
- physical cols；
- padding 说明。

Z 不进入 `plan.bindings`，没有 upload 或 download action；其 buffer 只在 stage 0 producer 和 stage 1 consumer 之间存活。stage 1 读取的是 producer buffer 上的同 addressing view；不经过 Host 拼接。

## 4. 实际 task、submit、BO 与依赖差异

| 组 | 计划 | task | submit | BO | publish |
|---|---|---:|---:|---:|---:|
| A fixed | 256/256 -> 256/256 m/m | 12 | 4 | 20 | 1 |
| B manual | stage1 256x128 n | 12 | 4 | 20 | 1 |
| C geometry-distinct | stage1 256x64 n | 15 | 5 | 23 | 2 |
| D two plans | stage0 9 + stage1 3 | 12 | 4 | - | 2 |

- A 与 B 的 task/template/K extent/input-output layout/submit group length/publish 完全相同，B 只是计划 attrs 不同；因此 B 被标为 **metadata-only**，不参与最终选择。
- C 的 64-column stage 1 产生 2 个 N tile、15 tasks、5 submits、2 个 publish，是真正的执行几何差异。
- 所有 submit 都串行，stage 1 额外记录 `stage_barrier=all_previous_submits` 并依赖 Z producer action。
- D 是两次独立单 GEMM plan 调用：stage 0 9 tasks、stage 1 3 tasks；Z 先下载到 Host 再作为第二个 plan 的 A 上传，额外 Host 往返计入 D 的时间。

## 5. 正反例与板端数值证据

独立 oracle：每 stage 使用 float64 partial 求和并在每 256 K 块后显式转 FP16；先算 Z，再算 Y，不复用被测 frontend/planner。

正向：

- 26 个候选全部通过静态 C preflight 和板端逐次输出验证。
- M=65/N=97 非对齐 tail、K=576 尾块 64、H=576 尾块 64 均正确。
- canonical/swapped stage grid 都通过。
- 同一 workspace 交替输入，Z/Y 均匹配 oracle，输出随输入改变。
- 不同 source M tile（stage0 source 128，applied 64；stage1 source 64/applied 64）通过重放。
- hidden Z 没有 Host transfer；D 的 Z Host 往返作为显式对照。

负例：

| 负例 | 检查层 | 结果 |
|---|---|---|
| 缺 Z producer dependency | C verifier | `RK_E_SCHEMA` “producer ... not a declared dependency” |
| consumer view 超出 producer/buffer | plan/view 层 | `RK_E_SHAPE` |
| 缺中间 value / 两 stage 不形成链 | frontend | `RK_E_UNSUPPORTED` |
| stage rounding annotation 与源不一致 | frontend | `RK_E_NUMERIC_CONTRACT` |
| global schedule 复制到全部 stage | frontend | `RK_E_UNSUPPORTED` |
| stage0 tile_n != 256 | planner | `RK_E_UNSUPPORTED` |
| 不同 applied M row subview | planner | `RK_E_UNSUPPORTED`，明确要求显式 repack |
| H=257 非 32 对齐 tail | planner | `RK_E_UNSUPPORTED` |
| 提前复用 Z / alias | C verifier | 现有 slot/alias 检查，不放宽 |

板端完整调用中位数（同 shape、同数值契约）：

```text
A fixed 256                      1.414 ms
B manual stage1 256x128 n        1.366 ms  (metadata-only, disqualified)
C measured overall best 256x256 n 1.343 ms (metadata-only)
C execution-distinct best 256x64 n 1.494 ms
D two independent plans + Host Z 1.906 ms
```

## 6. 固定/人工/联合调优、确认与新进程重放

- A：固定 256 控制。
- B：手工 stage1 256x128 n；初始 1.366 ms，但执行签名与 A 相同，属于 metadata-only，不能作为 schedule 几何收益。
- C：26 个联合候选；实际 execution-distinct 的最佳是 256x64 n，1.494 ms，比 A 慢。
- D：两次独立 plan 调用 + Host Z 往返，1.906 ms；该差值不能归因于 M/N tile 调优本身。
- 独立交错确认：
  - B vs A：speedup 1.034，达到 2% 阈值，但 **metadata-only，不选中**；
  - C 256x256 n vs A：0.989，未确认；
  - C 256x64 n vs A：0.928，未确认。
- 最终保留 A：没有已确认的 execution-distinct 改进。
- 新进程 replay A：pass；B replay 也 pass，但仅作为执行等价/噪声对照。
- 所有候选保持相同 Z/Y 数学、显式 K 舍入和 8e-3 数值门槛。

## 7. 资源限制、模型回归、代码与证据身份

资源/兼容性：

- 旧 plan 和现有默认路径未改变。
- 没有关闭 C 检查、放宽 slot 或忽略依赖。
- 32-row 以下 consumer subview 受 CUBE 物理行几何限制；当前策略是要求 applied M 相同，source M 差异通过 backend retile 解决。
- 没有把 Z 下载/拼接/上传冒充设备侧衔接；D 明确计量了 Host 往返。
- 本轮没有强行接入 SmolLM；没有自然且语义一致的纯两节点子图，因此交付独立能力实验。

回归：

- Python：489 collected，0 failed，1 个 slow-test skip。
- C：33/33 ctest pass。
- 新 frontend/planner/lifecycle/负例：`tests/test_two_stage_schedule.py` 15 项。
- 现有 schedule contract、K-tail、staged、模型和 runtime 测试全量通过。

身份：

- 实现提交：`01996a1d26cb2b5d5d746015bbf76566cc1ebadb`
- board lib SHA-256：`f9ed96ae2f1c38a941ac8a0e379026d8ef6e67ae9dcd62e4aafc11143d5d8419`
- experiment manifest：`adce01a874a0f97672818e6acbccb17877ce34abcc737f1fe138fe7f68925808`
- board run：`134e090dfe5fadc2b989860d5c572150146af4d720b02548e227f5a2feef2c1f`
- policy：`262585e013e28a696bf55fb5eb052c9977a6b6b912a99febf88c8532fb9d3a4c`
- selected explain：`19e8ed524d3e8baed6691fafffe2afea8a21535ddbf83d15e35a2489286bb09e`
- execution-distinctness：`50794108e69734459f25049cfa3df137a836247b3ab9cf697533b9e57f00f1fb`
- selected A plan hash：`df53efff07c5edcb07ee5bdbefbcd35dc0074ab354f7b145ee783d77be616972`
- replay A：`dacdc64981439116ee252d8e87d555f06d88ddd2ca6e752ea7ed2fb8e22d0149`

证据：

- `experiment/manifest.json`
- `experiment/board-run.json`
- `experiment/policy.json`
- `experiment/selected-explain.json`
- `experiment/execution-distinctness.json`
- `experiment/confirm-b.json`
- `experiment/confirm-c64.json`
- `experiment/confirm.json`
- `experiment/replay-a.json`、`experiment/replay-b.json`
- `experiment/plans/*.rkplan.json`

## 8. 下一步真实模型子图与剩余缺口

SmolLM 中没有一个不经 elementwise 的自然纯两 GEMM 链适合直接接入本实验：

- attention 的 QK -> softmax -> PV 需要 score 中间值和 row-wise softmax/tail；
- FFN 的 gate/up -> SwiGLU -> down 需要三节点、分支和 elementwise 中间值；
- QKV/lm_head 后没有直接消费另一个 GEMM 的中间值。

因此下一步应选择 **FFN gate/up -> SwiGLU -> down**，扩展为受限三节点：
1. elementwise stage 的独立 schedule 和中间 layout；
2. producer/consumer N/K 维映射与 SwiGLU 的 FP16 收窄点；
3. 多中间值的分支、依赖、scratch 生命周期；
4. 允许不同 applied M/N tile 的安全 repack 或 row-subview；
5. 中间 tail 非 32 对齐时的设备侧 repack；
6. 三节点预算、C verifier 和 submit grouping；
7. 模型级真实策略搜索、确认和 V5 回归。

仍未支持：通用多节点流水、跨 stage tile 流水、device repack、split-K、任意 BK、swizzle、async pipeline、多核、device KV 和 RKNN 对照。
