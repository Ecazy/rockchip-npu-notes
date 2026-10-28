# Creation-time `co_program` cache

日期：2026-10-27。本阶段只缓存 `co_program`，不同时改命令 BO 打包、guard、
同步或数值政策。前一轮的 `f18c9f3` 报告作为比较入口；本阶段所有独立归因和
端到端数字都重新采集。

## 0. 结论

**采用**创建时预发射的 `co_program` 缓存，默认用于满足结构谓词的
single-submit workspace；保留 `RK_PLAN_PROGRAM_CACHE=0` 作为原路径控制。

- QK group b8：**1.1342×**（394.3→348.0 µs，约 46 µs）
- PV group b8：**1.1387×**（380.3→334.2 µs，约 46 µs）
- QKV b8：1.0087×，emit 只有约 1.5 µs，处于噪声范围
- MLP b32：0.9956×，结构谓词不满足（8 submits），fallback 原路径
- Tiny ViT b8 无插桩端到端：**1.0554×**（3700.8→3507.0 µs）
- Tiny ViT b1 无插桩端到端：**1.0282×**（1239.6→1202.7 µs）

缓存占用现有 `batch_scratch`，**Host 增量 0 B**；创建时间增加约
179–255 µs（QK/PV），但 steady call 收益在几 call 内覆盖。

## 1. 可缓存条件审计

`rk_emit_action(plan, views, action_id, out)` 的输出 `co_program` 包含：

- 由 template 和 shape 决定的 regcmd words；
- `reads[]`、`output` 的 `co_view`（BO pointer、offset、tensor metadata）；
- count/flags/nreads。

emitter 只读取不可变 action/template、`views` 的 BO 地址/tensor 描述和
固定 template 参数；不读取输入内容、不读取运行期配置。workspace 创建后：

- slots/views 固定，不重分配；
- BO 地址和 fd 固定；
- action/plan 固定；
- 输入内容通过 upload 写同一地址，不改变 program。

因此单个 submit 内完整 task set 是自然可缓存的。资格谓词为：

1. `plan->nsubmits == 1`；
2. `submit->ntask_ids > 0` 且不超过既有 `max_submit_tasks`；
3. 每个 task 都经 `rk_emit_action` 在创建时成功发射和校验；
4. 任何一项不满足（包括 unsupported 或 emitter error）→ 整个 workspace
   保留原 per-run emitter 路径，不拒绝原工作负载。

这是结构谓词，不按模型名或 batch 名匹配。

## 2. 实现

- `RK_PLAN_PROGRAM_CACHE` 在 workspace 创建时快照；默认 **on**。
- 创建时对 eligible single-submit workspace 的每个 task 调
  `rk_emit_action(..., &ws->batch_scratch[j], ...)`。
- `batch_scratch` 原本就按 `max_submit_tasks * sizeof(co_program)` 分配；
  缓存直接驻留在它里面，因此额外 Host 分配为 0。
- 运行 `run_repeat` 遇到该 submit 的 NPU task 时跳过 emit，直接执行缓存的
  `co_program`；其他 workspace（multi-submit、fallback、不满足谓词）仍走
  原路径。
- cache 不跨 workspace、不跨 fd、不跨 BO 分配；每个 workspace 的 cache
  随 workspace 销毁。
- 配置在 workspace 创建时快照；后续修改环境不会让旧 workspace 静默换程序。
- 创建失败：若 cache 构建失败，cleanup 走既有 workspace fail 路径；
  `batch_scratch` 由同一处释放。程序构建失败只触发原路径 fallback，不发布
  半成品结果。
- 运行时仍保留 binding、finite、alias、guard、sync、download、publication
  全部检查；本改动只移除重复 emit。

## 3. 独立归因

当前版本 `battr`（每 case 32 call 中位数）：

| case | emit cache-off µs | emit cache-on µs | total cache-off µs | total cache-on µs |
|---|---:|---:|---:|---:|
| QK group b8 | 46.4 | 0.0 | 434.3 | 346.8 |
| PV group b8 | 46.4 | 0.0 | 376.0 | 332.8 |
| QKV b8 | 1.5 | 0.0 | 163.6 | 163.9 |
| MLP b32 | 14.3 | 15.2 | 596.5 | 594.1 |

QK/PV emit 成本值得消除；QKV 不值得期待大幅收益；MLP b32 不被谓词覆盖，
保持原路径。`total` 是独立 sample，不作为 headline；主结论使用下面的
逐轮配对 median。

## 4. 隔离 A/B

同一库、同一 workspace、同一 pack/emitter/executor/guard/download/
publication，只切换 `RK_PLAN_PROGRAM_CACHE=0/1`。7 个 rotated rounds，
changed-input frames；主估计为逐轮 median 的 median/ratio，p10 辅助。

| case | cache off | cache on | per-round ratio median | range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 394.3 | 348.0 | **1.1342×** | 1.1305–1.1385 | 1.1305 |
| PV group b8 | 380.3 | 334.2 | **1.1387×** | 1.0220–1.1425 | 1.0220 |
| QKV b8 | 167.7 | 168.0 | 1.0087× | 0.9774–1.0425 | 0.9774 |
| MLP b32 | 589.7 | 592.7 | 0.9956× | 0.9916–1.0015 | 0.9916 |

所有 56 个 cache-off/on 输出逐位一致。

## 5. 创建时间、Host 内存、首调用

| case | setup off µs | setup on µs | cache bytes above existing scratch |
|---|---:|---:|---:|
| QK group b8 | 907.1 | 1162.3 | 0 |
| PV group b8 | 991.7 | 1170.4 | 0 |
| QKV b8 | 388.8 | 382.4 | 0 |
| MLP b32 | 3611.4 | 3680.5 | 0（不缓存；差异为噪声） |

`co_program` 每个 16,624 B，但缓存槽复用已存在的 `batch_scratch`；因此
Host 常驻增量是 **0 B**。QK/PV 创建时新增一次发射，约 179–255 µs；
steady 收益约 46 µs/call，超过约 4–6 calls 后净收益为正。

首调用开销包含创建后第一次运行，单独保存在日志中；未混入 warm 比较。

## 6. 完整模型（无插桩）

Tiny ViT b8/b1，256 张图，7 rounds；同一 compiled graph、同一进程，两个
runner，env 在各自 workspace 创建前快照；`stats=None`，无节点代理。
b1 在 b8 收益确认后才补。

| model | cache off median µs | cache on median µs | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3700.8 | 3507.0 | **1.0554×** | 1.0525 | yes |
| Tiny ViT b1 | 1239.6 | 1202.7 | **1.0282×** | 1.0231 | yes |

全模型输出 logits 逐位一致；b8 约 5.5%，b1 约 2.8%。

## 7. 生命周期/失败验证

- 三个独立 workspace（两个 cache-on、一个 cache-off）输入不同帧：
  - frame0 cache-on 与 cache-off 逐位一致；
  - frame1 另一 cache-on workspace 与 cache-off 对应帧逐位一致；
  - 同一 workspace 重复输入不变；
  - **PASS**。
- 错误静态 shape 和 non-finite 输入：拒绝，失败路径无部分输出；
  address change 完整性检查：PASS。
- 既有 CTest（含 allocation/creation failure、submit failure 等 mock
  scenario）在 cache 默认开启下 **32/32** 通过。
- 冻结数值复验：**PASS / 0 problems**。
- Python：**348 passed, 3 skipped, 68 subtests**。

## 8. 决策与范围

- 采用创建时单-submit `co_program` 缓存，默认 on；`RK_PLAN_PROGRAM_CACHE=0`
  可回退。
- 没有改 command BO 打包、guard、同步、下载、发布或数值政策。
- 程序 cache 已证明有效，因此可以另开实验验证命令打包复用；本阶段不做
  该第二项，避免失去归因。

## 9. 身份和证据

- 最终 runtime：`cc4586eb4d51edf3b7cbab5d9fffd2fdc0d40e87b24ee9f052318108eb87cdc9`
- probe：`c12153f14dc84ef68299e82fd56846a6cbff98858529f1122ff2f39d818d4b15`
- compiled package：`16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec：`2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
- 结果：`results/20261027T093259Z-template-gap/runtime-layers-program-cache/`
