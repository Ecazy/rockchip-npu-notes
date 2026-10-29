# TileLang→RKNPU 布局转换、打包与数据复用优化报告

日期：2026-10-28
主场景：SmolLM2-135M-Instruct（真实 269.1 MB FP16 权重，30 层，容量 256）
回归：MNIST MLP / Tiny Transformer 完整 10,000 图测试
设备：RK3588 OPi 5 Plus，driver v0.9.8，NPU 1 GHz，core mask 1
基线提交：`dca0fc7441c773460f445b4ef67e2eba060fac71`（工作区当时已有未提交修改，原样保留）

## 1. 结论

已确认并修复：

1. **P0 单次 GEMM 内重复 A 打包已确认并修复。** 旧 planner 为每个 N 输出 tile 重建同一份 A 的 input slice 与 upload action。现在按 `(binding, M row block, K slice, extent, source stride, destination layout)` 在一次 lowering 内共享 view/action。qkv decode A upload `12→3`，gate_up `36→3`，down `18→6`，lm_head 分片 `48→3`；全部 268 个 plan 的 A upload action 总数 `5660→980`，A packed bytes `309,852,672→56,689,152`（−81.7%）。动态 B 的相同 M-tile 重复上传也一并消除（attn prefill B `3→1`）。
2. **P1 固定权重复用已确认；跨 plan 的 packed BO 复用已实现。** 新增 runtime shared-constant cache，按真实源字节摘要 + 源区域/stride/transpose + dtype/layout/几何/padding + fd/context 身份共享已打包 BO；不再把 shape 当权重身份。当前 load 期间 5,652 个 immutable B upload action 中 2,826 个首次打包、2,826 个命中复用（50%）；唯一 packed payload 为 268,959,744 B，计入每 BO slack/guard 的 request bytes 约 361,742,976 B。runtime 累计的 723,485,952 B 是 decode/prefill 两次引用之和，不是实际常驻量。Host `lm_head` 分片也改为 decode/prefill 共用一个 Host packed 数组。
3. **P2 KV 全历史重复打包已确认并修复。** decode QK/PV 现在按 layer/KV-head 持有独立持久 workspace；首次/reset 做一次完整 native 初始化，稳态每 token 只写新增 stripe。每 token 的 K/V 逻辑更新量由全历史 `5,898,240 B` 降到 `23,040 B`（256×），且 reset、跨请求、层/head 隔离和 capacity 边界均通过。QK/PV 不再每步 Host 转置整份 K。
4. **P3 一处高频纯拷贝已消除。** Smol 只消费 SwiGLU 的 `act`，旧 action 还额外 contiguous 复制 gate/up 两个全宽数组。新增 `swiglu.packed_gate_up_act.v1`，保持完全相同的 FP16 输入、FP32 sigmoid/product、FP16 回写顺序；3 次 benchmark 的 SwiGLU Host 时间 `1.156s→0.828s`（−28.4%，1,530 次调用不变）。稳定 softmax 的数值实现未改。

仍存在：共享 packed weight 只在同一 live fd/context 内复用；持久化 native 权重产物未实现；P2 每 token 仍有 180 次小状态写，C API/BO sync 开销约 4 ms/token；publication guard 仍占约 0.75 s/次 benchmark；Host stable softmax、RMSNorm、RoPE、SwiGLU 计算仍是端到端主成本之一。

## 2. 基线与最终对照

固定请求：42-token prompt，3 次独立 greedy generation，每次 16 new tokens；baseline 与 current 交替运行 3 轮，同 core mask、同 dtype/数值政策。baseline = 旧 P0 plan + `--no-share-constants --no-persistent-kv`；current = 新 plan + shared constants + persistent KV。

| 指标（3 轮中位数） | baseline | current | 变化 |
|---|---:|---:|---:|
| 加载/初始化 | 8.018 s | 8.664 s | +8.1% |
| prefill（单次） | 2.691 s | 2.076 s | **−22.9%** |
| 稳态 decode（16 token 合计） | 6.183 s | 5.953 s | **−3.7%** |
| decode 单步中位数 | 386.8 ms | 357.2 ms | **−7.7%** |
| benchmark wall（3 次 generation） | 26.779 s | 24.209 s | **−9.6%** |
| load + benchmark wall | 34.797 s | 32.873 s | **−5.5%** |
| 峰值 RSS | 1,946.6 MiB | 1,205.8 MiB | **−38.1%** |
| NPU upload calls | 164,646 | 35,586 | **−78.4%** |
| NPU upload bytes | 1,891.6 MB | 455.3 MB | **−75.9%** |
| `plan_pack_us` | 3.776 s | 0.991 s | **−73.8%** |
| 其中 `pack_layout_us` | 2.869 s | 0.755 s | **−73.7%** |
| `plan_call_us` | 12.291 s | 9.305 s | **−24.3%** |
| `plan_guard_us` | 3.172 s | 2.265 s | **−28.6%** |
| Host action 时间 | 3.920 s | 3.849 s | −1.8% |
| download bytes | 644.1 MB | 644.1 MB | 0 |

三轮 paired wall ratio 分别为 `0.904 / 0.921 / 0.832`；中位 `0.914`。decode 有一轮受设备噪声影响（ratio 1.003 / 1.062 / 0.766），因此 decode 收益小于 prefill/pack，但三轮 wall 均下降。冷启动时间（load+wall）也下降 5.5%；加载单独略升，来源是 per-layer KV workspaces 和 shared cache 的逐 tile FFI/结构校验，而不是稳态 repack。

### 控制实验归因

| 控制组 | wall | upload calls | upload bytes | pack | 峰值 RSS |
|---|---:|---:|---:|---:|---:|
| baseline（旧 plan，无共享，无持久 KV） | 26.78 s | 164,646 | 1,891.6 MB | 3.776 s | 1,947 MiB |
| P0-only（新 plan，无共享，无持久 KV） | 25.27 s* | 44,226 | 738.4 MB | 1.889 s* | 1,623 MiB |
| P0+P1+P3（共享常量，无持久 KV） | 24.30 s | 44,226 | 738.4 MB | 1.713 s | 1,259 MiB |
| P0+P1+P2+P3（最终） | 24.21 s | 35,586 | 455.3 MB | 0.991 s | 1,206 MiB |

\* P0-only 只有一轮控制测量，wall/pack 仅作归因诊断；upload counts/bytes 是确定性计数。

P2 的三轮直接 off/on 配对中，wall `24.825→23.501 s`（−5.3%），decode 合计 `5.794→5.654 s`，pack `1.735→0.950 s`，upload bytes `738.4→455.3 MB`，KV append 状态更新 `0.199 s/3 runs`，峰值 RSS `+3.2%`（per-layer workspace）。P2 的代价是固定容量下仍需初始化/清空整份 5.9 MB native KV，但“计算覆盖容量”和“每 token 上传历史”已明确分离。

## 3. P0：A 切片共享

代码：`python/rknpu_backend/passes/dag.py::_MatmulLowering.build`。

共享键包含 binding、M 起始/行数、K slice 起点/extent、source stride 和 destination layout；B 切片用同一机制去重。缓存生命周期严格限于一次 lowering（一次静态执行的数据流），不跨 run 缓存内容，也不以 Host 指针判断内容未变；每次 run 的 A 仍重新 upload。各 plan 的 upload action/声明 packed bytes：

| plan | baseline uploads | current uploads | A uploads（前→后） |
|---|---:|---:|---:|
| layer.0.qkv.decode | 24 | 15 | 12→3 |
| layer.0.gate_up.decode | 72 | 39 | 36→3 |
| layer.0.down.decode | 36 | 24 | 18→6 |
| lm_head.decode.00 | 96 | 51 | 48→3 |
| attn.qk.prefill | 4 | 4 | B: 3→1 |

全部 268 plan：upload action `11,320→6,636`，声明 packed bytes `848,034,304→594,739,712`。

## 4. P1：不可变 packed 权重共享

代码：`custom-ops/include/rk_plan.h`、`custom-ops/src/rk_plan_runtime.c`、`python/rknpu_backend/models.py`、`runtime/ffi.py`。

规则：

- Python key 覆盖：实际 FP16 常量数组 SHA-256、源 offset/stride/transpose/broadcast、dtype、layout version、physical rows/cols/atom/bytes、buffer capacity/offset/alignment、padding state。
- C 再校验 action/source/destination 的 source region、stride、transpose、tensor 几何与 padding state；不一致拒绝。
- 只允许同一 live `fd`/provider context。handle 保存 fd；跨 fd 返回 `RK_E_DOMAIN`。不尝试跨 context 导入 BO。
- workspace 对每个唯一 handle 持有引用；confirmed destroy 后释放。最后一个引用释放时 cache entry 和 BO 一起销毁；后续请求重新打包，绝不复用已无所有者的 BO。
- shared destination 不允许再被非 shared upload 写入；shared/owned buffer 不能混在同一 slot。

Host packed 复用：`lm_head` shard 的 decode/prefill 现在共享同一个 Host 数组。BO 复用：`rk_workspace_create_shared` 跳过对应创建期 upload，并把该 buffer 的所有 view 指向 shared BO。未实现持久化 native 权重文件；P1 的收益只计入加载/内存，不计入稳态 decode。

## 5. P2：KV native 布局驻留与增量更新

代码：`custom-ops` 的 `rk_workspace_set_persistent_upload`、`rk_workspace_update_upload_region`、`rk_workspace_clear_upload_destination`；`smollm.py` 负责 layer/head/request 状态语义。

- decode QK/PV 每 `(layer, KV head)` 一个 PlanCall/workspace，避免不同层/head 串扰。
- Host K/V cache 保持 `[capacity, head_dim]`；native 初始化直接把行主序数据写入 packed destination，不再先构造 `K.T`。
- 首次 decode/reset 后完整写入一次；prefill 后首 decode 延迟做完整初始化。稳态 append：
  - QK 仅新增 position 对应的一个 destination row（64 个 FP16）；
  - PV 仅新增 position 对应的 destination column（64 个 FP16）。
- reset 清 Host cache，并对已初始化的 native destination 明确 `clear`；随后完整重写。不同请求通过 `generate()` 的 reset 隔离，A/B/A 输出逐 token 完全一致。
- 每 token K/V append 逻辑字节 `30×3×(64+64)×2 = 23,040 B`；全历史重传为 `30×3×2×256×64×2 = 5,898,240 B`，比例 256:1。每次请求首次初始化仍为 5,898,240 B。

## 6. P3：高频 Host 复制

代码：`python/rknpu_backend/hybrid.py::_swiglu_packed_act`、`smollm.py::_swiglu`。

`swiglu.packed_gate_up_act.v1` 去掉仅供旧调用者使用的 gate/up 两份 contiguous FP16 拷贝，只返回 `act`。它与旧 action 在随机形状 `(1/42/256, 3072)` 上逐 bit 相等（单测），保持原有 FP32 sigmoid/product 与 FP16 舍入。稳定 softmax、RMSNorm、RoPE、residual add 的数值路径未改。

## 7. 正确性与回归

- Host contract：`pytest -q tests` 全部通过（含 `tests/test_shared_constants.py`、slow Smol reset replay）；`ctest` 33/33 通过。
- Smol 冻结数值验收：`verify-final4-current-acceptance.json` PASS，56 gates 全过；incremental-vs-full `max_abs=0.0`、top1 全一致；teacher-forcing 首次分歧位置 6，仅作为质量统计，不是数值门限。`models/smollm2_numeric_acceptance.json` 的 `gates`、`calibration_metrics`、`quality` 与旧 spec 逐字段相等，只更新了 package manifest 身份和 spec SHA（见 `spec-policy-invariance.txt`）。
- 边界：prompt 256 prefill 通过；255 prompt + 1 decode 达到 `valid_len=256`；257 token 预检拒绝且随后请求健康；reset 后 A/B/A 输出完全一致；EOS 停止和短 prompt 通过。
- MLP/Tiny Transformer：`results/20261028T111741Z-models` PASS；MLP b32 全 10,000 图 accuracy `0.9831`，Tiny ViT b8 全 10,000 图 `0.9821`，top-1 agreement `1.0`。
- `test_shared_constants.py` 覆盖：同内容复用、同指针内容修改、不同权重、A/B/A 释放顺序、fd 隔离、不兼容 destination、持久 state 全量/单 stripe/clear、越界预提交失败后健康、同 A 指针内容每次 run 重新上传。

## 8. 原始证据与重放

证据根目录：`results/baseline-layout-reuse/`

| 内容 | 路径 |
|---|---|
| 最终 analysis | `analysis-final4.json` |
| baseline/current benchmark | `board-final4-baseline-r{1,2,3}-benchmark/`、`board-final4-current-r{1,2,3}-benchmark/` |
| P0/P1/P2/P3 控制组 | `board-contrib-p0-*`、`board-contrib-p0p1-*`、`board-p2off-*`、`board-p2on-*` |
| 冻结验收 | `board-final4-current-acceptance/`、`verify-final4-current-acceptance.json` |
| 边界 | `board-final4-current-boundary/`、`board-final4-current-capacity256/` |
| 模型回归 | `results/20261028T111741Z-models/`、`run-model-admission-final.log` |
| Host 测试 | `pytest-full-final4.log`、`ctest-final4.log`、`pytest-smollm-persistent-final4.log` |
| 源码身份 | `final-source-identity.txt`、`final-commit.txt`、`spec-policy-invariance.txt` |
| 旧 P0 plan 快照 | `package-before/` |

重放命令见 [REPLAY.md](REPLAY.md#layout-reuse--a-b-campaign-2026-10-28)。关键源码身份：

- `python/rknpu_backend/passes/dag.py` = `c875f978...`
- `python/rknpu_backend/models.py` = `dad45e04...`
- `python/rknpu_backend/smollm.py` = `c53bb288...`
- `custom-ops/src/rk_plan_runtime.c` = `d7a3f528...`
- current package manifest = `b923fc5b...`

## 9. 对 RKNN 差距的判断

本工作**没有可用的同边界 SmolLM2 RKNN/RKLLM 计时**，因此不能给出 Smol 的 RKNN 加速比，也不能把剩余差距全部归因于布局/打包。已有同设备 vendor 对照只覆盖 MLP/Tiny Transformer：那些模型本次没有改 plan，只有通用 runtime 变化，所以本报告不能声称解释了它们的 vendor 差距；其既有结论仍是“差距主要指向 graph-level scheduling/fusion 与 Host 边界”。

对 Smol 本后端可测得：

- 本轮消除的重复 layout/pack 使 `plan_pack_us` 降 73.8%、upload bytes 降 75.9%、`plan_call_us` 降 24.3%；这些是实测解释的部分。
- 但最终 current 的单次 generation 仍有约 3.0 s `plan_workspace`、约 1.28 s Host action（3 次合计约 3.85/3 = 1.28 s）、约 1.59 s `submit_wait`（3 次合计）量级成本；稳定 softmax、RMSNorm、RoPE、SwiGLU 和 180 次/token 的 KV stripe 状态写仍未下沉。
- 因而可以把 **pack/layout 重复成本从主要瓶颈降为已量化的小项**；剩余 RKNN 差距中，**Host 算子边界、逐层小 submit/命令开销、Guard/同步语义**仍未被本轮解释。要给出“解释百分比”，需要新增同输入、同 core mask、同数值政策、且明确计时边界的 Smol RKNN 对照；本轮不伪造该数字。
