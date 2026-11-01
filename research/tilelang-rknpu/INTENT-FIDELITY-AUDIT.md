# 意图保真审查：RKIR + RKPlan 能否承载上层合法控制意图

> **当前阅读入口：[当前能力、实验结论与重放](#current-status)。** 下文从最初审查开始按时间保留，早期的缺陷、测试数量和“只读”描述不代表当前状态；后续章节包含修复与更正。

- 日期：2026-10-31
- 基线：main @ de4b07a2456033db330c31f10a0df9d908a759c2（2026-10-31T10:25:42+08:00）
- 范围：research/tilelang-rknpu（frontend / RKIR / planner / RKPlan / C 执行链），设备目标 RK3588
- 本轮性质：**只读架构与契约审查**。未修改编译器、IR、planner、verifier、runtime、文档与历史证据；未放宽任何数值或安全检查；未新增 recipe；未改生产默认；未上板。

原始证据与最小复现位于 results/20261031T050000Z-intent-fidelity-audit/（expression/、execution/、cross-review/、lead-verification/）。本文件是**唯一主报告**。

---

## 0. 方法与证据分级

**三段式**：两个只读研究员并行（表达传递 / 执行约束）→ 队长独立复跑与地面真值核对 → 第三方独立交叉复核（对 11+1 条候选逐条给 CONFIRMED / PARTLY / REFUTED / UNVERIFIABLE）。交叉复核有权推翻作者结论：本轮实际推翻 1 条、降级 1 条、纠正 6 处数字与措辞。

**证据标签**（每条结论都带）：[源码推断] 只读代码未运行；[离线复现] 本机跑出且命令在案；[mock] 既有 host mock provider 的观测，非设备；[文档] 文档表述需与源码对照；[未验证] 本轮无证据并给出会回答它的命令。

**本轮板端证据为 0 条**。所有执行一致性结论来自 host mock provider 或 C 校验器的纯离线 ctypes 调用（../custom-ops/build-host-mvp/librknpu_plan.so，既有产物，未重建）。

**判定用语**（矩阵严格使用）：保留 / 等价改写 / 明确降级 / 拒绝 / 丢失 / 仅诊断 / 未验证，另有 **无入口**（该层结构上不存在通道）。

**两条读表前提**：

1. 调度**不是** descriptor 字段（codegen/descriptor.py:155-172 无 tile_m/tile_n/order）。调度意图的落地形态是 task 数量、view 几何（rows/cols/src_row_*/src_col_*/k0）与 publish 矩形，因此 descriptor 列对调度类意图恒为等价改写，判据在第 5/6 列的实际数字。
2. plan_hash 不参与判定。本轮同时出现“hash 相同、执行相同”与“hash 不同、执行相同”，均按字段级或位级证据判定。

---

## 1. 总体判断

### 能保证（有本轮证据）

1. **数据面完整**：几何、数据流、顺序、预算四类共 8 个执行字段（view/tensor、offset_bytes、slot_id、flags、action id、submit task_ids、publish 矩形、claimed/budget bytes）全部到达设备，并被 C 校验器独立复算 [离线复现]。
2. **提交边界与 K 累积顺序被真实遵循**：submit_mode=chain 声明 1 次、mock 实测 1 次；separate 声明 2 次、实测 2 次；两者与“每 256 宽 K 块 FP16 写回”的手算参考逐位相同（max_abs_err=0）[mock]。
3. **baseline 与图入口下，显式调度注解要么被消费、要么被明确拒绝**，不存在当作性能 hint 静默丢掉的路径 [离线复现]。
4. **非法意图仍被拒绝**：C 校验器 13 个单字段变异中，6 个必须仍拒绝的全部仍拒绝；单测基线（58 + 15 + 60）全绿 [离线复现]。
5. **计划持久化往返保语义**：compile→save→load→re-verify 的 exec 投影逐字段相同、canonical bytes 重存逐字节相同（38323 B）、篡改被拒 [离线复现]。

### 不能保证（本轮未证明或已证不成立）

1. **不能保证任意合法 TileLang 程序都能完整传递**。本轮只覆盖受支持域内的样本；且 L1 形状域在 TileLang 入口根本无法表达（见 F5），谈不上传递。
2. **不能保证“为什么这么排”到达设备**。控制面 8 条字段断链（F7）：设备知道怎么算，不知道为何这样排。
3. **不能保证 Python 的“已验证”等于 C 可接受**（F2）。两者不是同一规则集。
4. **不能保证 linear_task_order != baseline 时上层声明的调度被尊重**（F1）——它会被零痕迹丢弃，且同分支绕过数值契约门。
5. **不能保证板端数值**：本轮零板端证据，所有执行结论限于 mock 与离线。

---

## 2. 最重要的发现（按影响排序）

> **最终结论见 §23.2 汇总表**（2026-11-01 重核，只清点不改实现）：F1、F2≡F4、F3 已解决；F5 只剩文档一半；F6/F8/F9 仍成立但只影响说明与错误信息；F7 需重新分类；F10 的原判断已不成立。以下保留原始记录，逐条更正指向写在 §23.4。

### F1 · linear 策略分支零痕迹丢弃已声明的调度，并绕过数值契约门

- **优先级**：声明丢失属 2；数值门绕过属“接受/拒绝边界”维度（被执行数值本身未变，见下）。
- **分类**：G（信息传递/验证缺陷）+ 文档矛盾。
- **结论与影响**：linear_task_order != baseline 且单 matmul 时，分派到 build_scheduled_linear，该 builder **不读** source_schedule；已声明的调度在计划里**不留任何痕迹**；该分支也不调用 mn_schedule_from_attrs，使 tile.py:104-109 的 K-rounding 契约检查整段被跳过。
- **精确位置**：分派 passes/dag.py:2644-2646；builder passes/dag.py:1077-1243；mn_schedule_from_attrs 调用点仅 dag.py:1279/1280/1623/1624/2163；契约门 passes/tile.py:100-110（位于 mn_schedule_from_attrs 内，且在 tile.py:57-59 的 `if raw is None: return` 之后）；explain 回显来源 codegen/pipeline.py:129。
- **最小复现**：

```bash
cd research/tilelang-rknpu
export PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.pydeps-p0
python3 results/20261031T050000Z-intent-fidelity-audit/lead-verification/probe_plan_identity.py
python3 results/20261031T050000Z-intent-fidelity-audit/lead-verification/probe_numeric_gate.py
```

- **预期 vs 实际** [离线复现]：

| 策略 | 声明 128x128 | 不声明 | 是否同一计划 |
|---|---|---|---|
| baseline | 144 task / 48 submit / applied=144 | 36 / 12 / applied=36 | **否** |
| merged | 36 / 1 / **applied=0** | 36 / 1 / applied=0 | **是：plan_hash 与 to_dict 全等**（375fb6c4de） |
| scheduled_baseline | 36 / 12 / **applied=0** | 36 / 12 / applied=0 | **是**（0ae0ca28d8） |
| interleaved | 36 / 1 / **applied=0** | 36 / 1 / applied=0 | **是**（af1520ff3e） |

数值门（同一份 direct-RKIR，只改声明与选项）：

| 策略 | explicit_per_k_fp16 | fp32_final_narrowing | 缺省 |
|---|---|---|---|
| baseline | ACCEPT 144/48 | **REJECT -4 RK_E_NUMERIC_CONTRACT** | **REJECT -4** |
| merged | ACCEPT 36/1 | **ACCEPT** | **ACCEPT** |
| scheduled_baseline | ACCEPT 36/12 | **ACCEPT** | **ACCEPT** |

预期：声明被消费或被拒绝并诊断。实际：声明被丢弃且无诊断；同一份源的可采纳性取决于一个与之无关的 task-order 选项。

- **数值语义是否被改写**：**没有**。交叉复核用 mock 做位级比较：merged 与 baseline **逐位相同**（786432/786432 bits；与“整段收窄”相差 maxabs 1.95e-3）。linear 策略实际执行的是目标既定契约（逐 K 块 FP16 回写）。因此本条是**“接受/拒绝边界被移动 + 声明零痕迹”**，不是“合法程序的数值被换成第三种语义”。
- **是否已证明影响真实负载**：**否** [离线复现]。真实 decode merged 证据（tools/compile_decode_merged_policy.py → models/tilelang_matmul.py:make_matmul_prim）里 source_schedule=null、applied_schedule=0、linear_task_policy={merged}，即该负载从未发出可被丢弃的声明。触发条件是“per-K FP16 回写的源 + 显式 tile 声明 + linear 策略”三者同时在场。
- **文档直接矛盾** [文档]：TILELANG-SCHEDULE.md:70-72 称 planner consumes source_schedule for the supported single-GEMM path and records both source_schedule and applied_schedule on every emitted matmul task；同文 :93 称 No schedule annotation is silently treated as a performance hint and then dropped。两条在 linear 策略下均不成立。options.py:41-44 的 docstring 只承诺 task order / submit grouping / CBUF reuse，未声明会覆盖 M/N tile；SUPPORT-DOMAIN 无条目。
- **置信度**：高（队长独立复现 + 交叉复核 CONFIRMED）。**未验证**：scheduled 变体在板端的 task 发射顺序；n 更大时是否撞 plan.py:887 的 64-task/submit 上限。
- **最小修复方向（未实施）**：① 让 build_scheduled_linear 调用 mn_schedule_from_attrs 仅作契约校验（不动 geometry，先堵数值门）；② 在分派处对“声明将被覆盖”给显式诊断或拒绝；③ 同步修正 TILELANG-SCHEDULE.md:70-72 与 :93。**风险**：②会改变三种 linear 策略在“带声明源”上的接受面，需确认无生产路径依赖（本轮证据显示当前没有）。

### F2 · Python 的“已验证”与 C 可接受不等价

- **优先级** 2 · **分类** G · **置信度** 高（CONFIRMED，含交叉复核纠正）。
- **结论与影响**：passes/verify.py:9-24 只查计数与 host_compute；plan.py 的 _validate_plan 与 src/rk_plan_validate.c 不是同一规则集。planner **自己产出**的计划可以通过 Python 验证而在 workspace 创建时被 C 拒绝。
- **精确位置**：passes/verify.py:9-24；plan.py 的 _validate_plan；src/rk_plan_validate.c:533-584、747-774。
- **最小复现** [离线复现]：tile_m=1 的 planner 自产计划 → Python verify_plan ACCEPT，C rk_plan_create 返回 rc=-12。
- **纠正**：原以手工构造的“行窗口”计划为例；交叉复核判定该例为反事实构造（当前代码路径不可达），已降级为注记。真正有杀伤力的是 planner 自产 tile_m=1 一例。
- **影响真实负载**：未证明。**未验证**：真实模型是否存在触发组合。
- **最小修复方向**：把“Python 能放行但 C 必然拒绝”的稳定判据前移为 Python 侧显式拒绝，或在文档声明“Python 验证是必要非充分”。风险：前移判据易与 C 实现漂移，宜只前移已有 C 结论的稳定判据。

### F3 · “未声明调度”有两种编码，而该状态同时是采纳开关

- **优先级** 2 · **分类** F+G · **置信度** 高。
- **结论与影响**：source_schedule 是否被盖上，取决于源码是否 materialize 每 K 块 FP16 回写这一**偶然结构**，而不是用户的显式意图；而“有没有 source_schedule”又是 planner 的采纳开关。同一数学的源因此分裂成两类计划。
- **精确位置**：frontend/tilelang_adapter.py:3968-3969、:5511-5513（条件不同：前者 ann.present，后者 explicit_fp16_rounding OR ann.present）、:5897-5899；采纳开关 passes/dag.py:2623-2631；文案 dag.py:649-655。
- **最小复现** [离线复现]：同一 GEMM、同样零 rknpu.* 注解：每 K 块 FP16 回写 → source_schedule 存在，12 task（128x128 网格）；仅末端收窄 → 缺席，8 task（256x256）。数字随网格变化，不能当作该形状的定值（交叉复核纠正了原写的 24/8）。
- **预期 vs 实际**：预期“未声明”只有一种状态；实际两种，且决定下游采纳。
- **影响真实负载**：[源码推断] 现有 models/tilelang_matmul.py 各变体结构固定，同一模型内不混用；跨模型挑选不同源写法时会改变 task 分解。
- **最小修复方向**：把“是否有显式调度”改为显式三态（无 / 源码几何推断 / 显式注解），并在 explain 中区分。风险：触及 frontend 全部 stamp 点，需回归全部 schedule 契约测试。

### F4 · RK_MAX_BUFFERS_PER_SLOT=64 拒绝 planner 自产计划

- **优先级** 3 · **分类** F（校验器簿记复杂度界）· **置信度** 中高（PARTLY）。
- **精确位置**：../custom-ops/include/rk_plan.h:34；拒绝点 ../custom-ops/src/rk_plan_validate.c:768-774。
- **结论与影响**：细 M tile 把单 slot 的 buffer 数推过 64，planner 照常产出，C 在校验期以 RK_E_SCHEMA: too many buffers share one slot 拒绝，诊断不含 slot_id 也不给建议。
- **最小复现** [离线复现]：tile_m=4/n192 → 192 buffers/slot → REJECT；tile_m=16/n128 → 80 → REJECT；tile_m=64/n256 → 恰好 64 → ACCEPT；默认 256 网格 → 3/14 → ACCEPT。
- **真实负载画像** [离线复现]（execution/repro/probe_slot_pressure.py 全树 4792 个计划）：模型与战役产物 n=4670 → **max 16/slot，越界 0**；越界 42 个全部来自 tuning 战役（max 1408）。这些历史产物现已 stale（hash 不匹配），不能当作 C 接受/拒绝的证据，只能作为“当时生成过此类形状”的线索。
- **纠正**：原报告称 “>64 完全可执行”——交叉复核判定仅为 [源码推断]（执行侧按 slot 分配，runtime.c:911-913），而校验器是创建必经门，>64/slot 的计划今天执行不到。故按“实现限制 + 潜在阻断”记录。
- **最小修复方向**：或把上限提到与真实簿记同阶并给出 slot_id 诊断，或在 planner 侧对超界 tile 组合提前拒绝并给替代建议。风险：放宽上限触及 C 校验器的复杂度假设，需先证明无二次复杂度爆点。

### F5 · L1 形状域未在任何声明层落地

> **§23.4 更正**：拒绝诊断现在已明确给出 `linear_task_order=baseline` 替代路径（实测），所以本条只剩“SUPPORT-DOMAIN 未写这个实验域”一半。

- **优先级** 3 · **分类** E（有意实验域）+ G（未声明）· **置信度** 高。
- **精确位置**：passes/dag.py:1093-1096；可达条件 dag.py:2644。
- **结论与影响**：具名 linear 策略只接受 M in {1,256} 且 K=576 且 N=3072，拒绝信息只说 limited to，不指出同一形状换 baseline 可编译，也不指出该域属实验域。
- **最小复现** [离线复现]：M=128/K=576/N=3072 等 4 个邻近 shape 在 merged 下被拒；同一 shape 在 baseline 下分别得 36/36/24/33 task 并全部编译成功。C 侧没有同域检查（grep 576/3072 在 C 源 0 命中）。
- **纠正**：原称“TileLang 用户会撞上该门”——**错**。L1 域 K=576 不是 256 整倍，TileLang 入口在 K-block 化阶段就先拒绝（K-loop tiling does not tile the parameter），故该组合只能经 direct-RKIR 到达。
- **声明层核查** [源码推断]：SUPPORT-DOMAIN.md、capabilities/*.json、options.py 三处均无该域；全仓 md 亦无。
- **最小修复方向**：把该域写进 SUPPORT-DOMAIN（实验域 + opt-in），并在拒绝信息里给出 baseline 替代路径。风险：极低。

### F6—F10（压缩记录，字段同前）

| # | 结论 | 位置 | 分类 | 优先 | 证据 | 真实负载 | 修复方向 |
|---|---|---|---|---|---|---|---|
| F6 | options_to_dict 漏 fuse_task_chain：该选项把 submits 从 5 变 1，而两次 options 快照逐位相同；options 不入 plan_hash | codegen/pipeline.py:91-106 vs options.py:35；消费方 artifact.py:147-148 | G | 3 | [离线复现] | 已证（同族选项在用） | 补齐投影键；风险极低 |
| F7 | 控制面 8 条断链：source_k0（C 有字段 rk_plan.h:193、src/ 零读取）、applied_schedule/source_schedule/linear_task_policy（descriptor 无槽位）、Submit.core_policy（C 侧仅 3 处：字段+枚举+范围检查，co_schedule.c 0 命中）、Submit.recipe_id（无字段）、options.tile_limit（passes/ 0 引用）、options.linear_task_order（无 descriptor 槽位，但 planner 读它选分支 dag.py:1097,2644） | 见 execution/REPORT §2 全表 | G | 4 | [离线复现] | 不改变今天执行结果 | 先明确哪些是审计痕迹、哪些必须进 ABI |
| F8 | Action.attrs 是无 schema 开放字典：未知键四层静默通过；upload source_binding=x 在 **Plan 构造期**抛裸 ValueError（非 load 期）；k0=not-an-int 到 descriptor 编码才炸 | plan.py:463-503、871 | G | 4 | [离线复现] | 编译器自产计划不受影响 | 增键/类型白名单；需枚举现有全部键 |
| F9 | schema/rkplan.v1.json 与实现不同代（PLAN_SCHEMA_VERSION=2 且拒非 2）；v2 有测试消费者，v1 无（§23.5.5 重核：仍成立） | schema/rkplan.v1.json、plan.py | G | 4 | [离线复现] | 无 | 删除或标注 v1 为历史；风险极低 |
| F10 | ~~linear_task_policy 全仓 1 写 0 读~~ **§23.4 更正：explain 现在读取它**（实测 `['merged']`）；chunk_index 仍仅一个单测；k_physical_extent/k_padding 被 tools/tune_schedule.py 消费 | dag.py:1196 | F | 4 | [离线复现] | 无 | 并入 F7 统一处理 |

---

## 3. 意图传递矩阵

读表前提见 §0。每格 = 判定用语 + 证据锚点。

| # | 上层意图 | (1) 上层表达 | (2) frontend | (3) RKIR | (4) legalization | (5) planner | (6) RKPlan | (7) descriptor/verifier | (8) emitter/执行 |
|---|---|---|---|---|---|---|---|---|---|
| M1 | M/N tile 经 TileLang 注解 | 保留 | 保留→source_schedule adapter:124-142,5897-5910 | 保留 op.attrs | 保留（classic，graph_ops=None legalize.py:108-114） | 保留 dag:2163-2168（24 task @128x128） | 保留 | 等价改写（几何落 rows/cols） | 保留 [mock] |
| M2 | 同意图经 direct-RKIR | 保留 op.attrs | 无入口 | 保留 | 保留 | 保留 | 保留：与 M1 exec 投影完全相同 | 等价改写 | 保留 |
| M3 | 同意图经 CompileOptions | **无入口**（options.py:11-44 无字段） | — | — | — | — | — | — | — |
| M4 | M/N 遍历顺序 | 保留 rknpu.mn_order | 保留 adapter:109-115 | 保留 | 保留 | 保留 tile.py:96-99 | 保留（publish 矩形重排） | 等价改写 | 保留 |
| M5 | 源 tile != 应用 tile | 保留（源码几何） | 保留 source_mn_tiling 与 applied 两份 | 保留 | 保留 | 保留 dag:1509,1522,1535 | 保留（可审计） | 等价改写（按 applied） | 保留 |
| M6 | 未声明调度（由源码结构决定） | 无入口 | **两义** adapter:3968/5511/5897（F3） | **两义** | 保留 | **两义**（12 vs 8 task） | 两者皆留档 | 等价改写 | 保留（数学同） |
| M7 | K 分块 / 累积顺序（固定契约） | 无入口 | 拒绝越界 adapter:78-84 | 常量 k_tile=256 | 保留 | 固定升序 tile.py:81-84 | 保留 k0 序列 | 保留 desc:330 | 保留 [mock] 逐位 |
| M8 | 逐 K FP16 回写声明 vs 实际 | 保留 rknpu.source_k_rounding | 保留且对照校验 adapter:165-172 | 保留 | 保留 | baseline 拒绝不符 tile.py:100-110；linear 策略 **丢失**（F1） | linear 下 applied=0 | **丢失** | 保留（执行恒 per-K FP16）[mock] |
| M9 | 两阶段 per-stage M tile | 保留 rknpu.stageN.* | 保留（逐 op） | 保留 | 保留 legalize.py:401-410 | 相等→保留；applied 不等→拒绝 dag:1286-1295（C 类：CUBE 布局） | 相等时保留（含 z_source 映射） | 等价改写（字节同一视图） | 未验证（板端） |
| M10 | 两阶段不同 source M | 保留 | 保留 source_mn_tiling | 保留 | 保留 | **保留（可 retile）** dag:1289 文案 + tests/test_two_stage_schedule.py:145 实测通过 | 保留 | 保留 | 未验证（板端） |
| M11 | FFN per-stage elementwise tile | 保留 | 保留 | 保留 | 保留 | tile_m 必须等于 s0 否则拒绝 dag:1649-1653 | 保留 | 等价改写 | 保留 |
| M12 | linear task order 策略 | 保留 CompileOptions.linear_task_order | — | — | — | 分支选择 dag:2644；形状域拒绝 dag:1093（E）；**声明丢失**（F1） | 丢失（无 applied_schedule） | 丢失（无槽位） | 保留（分组维度）[mock] |
| M13 | submit 分组 fuse_task_chain | 保留 | — | — | — | 保留 schedule.py:325-374（5→1） | 保留（mode/task_ids） | 保留 | 保留 |
| M14 | core 分配策略 | **无入口** | — | — | — | 丢失（硬编码 schedule.py:359,366） | 保留字段 | 仅诊断（validate.c:934） | 丢失（co_schedule.c 0 引用） |
| M15 | recipe 身份 | 无用户入口 | — | — | — | 保留 schedule.py:348-355 | 保留 recipe_id | **丢失**（无字段，C 独立重证） | 等价改写 |
| M16 | buffer 生命周期 / slot 复用 | 隐式（值活跃区间） | 保留 | 保留 | 保留 | 保留 storage.py:34-66 | slot_id 权威、lifetime 派生 | lifetime 仅诊断；slot_id 被消费 | 保留（按 slot 分配） |
| M17 | padding 策略 | 无显式入口 | — | — | — | 保留 dag:2205,2258 | 保留 | 仅诊断（0 引用） | 保留（打包器零填，无独立证明） |
| M18 | tile_limit 旋钮 | 保留 | — | — | — | **丢失**（passes/ 0 引用） | — | — | — |
| M19 | 未知 action attr 键 | 无入口（property 面） | — | — | — | — | 保留（原样写入） | 静默忽略 | 无影响 |
| M20 | 计划持久化往返 | — | — | — | — | — | 保留（exec 投影逐字段同） | 保留（canonical bytes 稳定） | 保留 |
| M21 | L1 形状域（具名 linear 策略） | 保留（direct-RKIR）；TileLang 不可表达 | — | — | — | **拒绝** dag:1093（E，未声明 F5） | — | — | — |
| M22 | 多节点图上的显式调度 | 保留 | 拒绝（无该状态） | 保留 | 保留 | **拒绝** dag:2623-2631、文案 dag:649-655 | — | — | — |

**矩阵结论三条**：

1. 等价改写是常态而非缺陷（M1/M4/M5/M7/M9/M11/M15/M16）：效果保住了，只是不再以“名为 schedule 的字段”出现在 ABI 上。
2. **唯一的意图丢失**是 M8/M12（F1 同源），它同时是唯一“声明零痕迹”的点。
3. 其余“丢失/仅诊断”格子都在控制面（F7），不改变今天的执行结果。

---

## 4. 分类：硬件约束 / 实验域 / 实现限制 / 真缺陷

| 分类 | 条目 | 判定 | 是否该改 |
|---|---|---|---|
| **C 硬件 / ABI / 已验证 recipe** | applied M 必须相等（CUBE co_index 使行不是可切片轴，custom_ops.c:80-84）；原语 M/Kt/N <= 256（custom_ops.h:18-22）；device 写入被读时的覆盖规则（validate.c:688-726） | 真约束 | 不改；任何放宽都以正确性为代价 |
| **E 有意实验域** | L1 形状域 M in {1,256} 且 K=576 且 N=3072；默认 256 网格 | 设计选择 | 不改域，补声明（F5） |
| **F 实现硬编码 / 表示缺失** | RK_MAX_BUFFERS_PER_SLOT=64（校验器簿记界，planner 已改为满 64 换 slot）；options.tile_limit **显式固定值（原记“死旋钮”不准确，见 §23.4）**；CompileOptions 无 tile 通道；Action.attrs 无 schema | 实现限制 | 按优先级修，不必立刻 |
| **G 传递 / 验证 / 诊断缺陷** | **F1（首要）**、F2、F3、F6、F7、F8、F9 | 真缺陷 | F1 必须修；其余按需 |
| **A/B 上层不合法或与数值契约冲突** | 越界 tile（0/257）、未知键、未知 order、非 256 K 块、tile>256、注解与源码 rounding 不符、numeric_mode=strict、未 opt-in 的 LUT | 正当拒绝 | 不改（保留其精确诊断） |

**边界**：F 不自动意味着可以删除限制；E 不自动意味着设计错误；“硬件能做但项目尚无实现”与“已有安全实现被中间层错误阻断”（F1、F2）**分开记录**。

---

## 5. 已验证的正向能力

| # | 能力 | 证据 |
|---|---|---|
| P1 | 同一 M/N 意图经 TileLang 注解与 direct-RKIR 产生 exec 投影完全相同的计划（hash 不同、执行相同） | [离线复现] expression/MATRIX M1/M2 |
| P2 | 提交分组与 K 累积顺序被真实执行：chain 1/1、separate 2/2，且与手算逐位一致 | [mock] execution/repro/probe_verifier_boundary.py |
| P3 | 显式注解覆盖源码 tile，且 source/applied 两份几何同时留档（可审计） | [离线复现] expression M5 |
| P4 | 图入口对未知调度键、K!=256、越界 tile、dataflow 上的显式调度显式拒绝而非静默忽略 | [离线复现] dag:649-655,2623-2631 |
| P5 | save→load→re-verify 保语义；canonical bytes 重存逐字节相同；篡改被拒 | [离线复现] expression/e4 |
| P6 | C 校验器对应当拒绝的仍然拒绝（6/6 必拒变异）；计划级不变量（publish 精确覆盖、无双写、常量独占 slot、不与输入别名）被独立复算 | [离线复现] |
| P7 | 两阶段不同 source M 确实被 retile 到同一 applied tile（tests/test_two_stage_schedule.py:145，本轮 15 passed） | [离线复现]（推翻原 F-2(iii) 的“无 retile”断言） |
| P8 | 数值与安全检查未被削弱：numeric_mode=strict、LUT opt-in、注解与源码 rounding 对照均在位 | [离线复现] |

---

## 6. 下一轮只修一个：F1

**建议只修 F1，分两步落地**：① 让 build_scheduled_linear 调用 mn_schedule_from_attrs（仅作契约校验，不用其几何），堵住“数值契约门被策略选项绕过”；② 在分派处对“声明将被覆盖”给出显式诊断或拒绝，并修正 TILELANG-SCHEDULE.md:70-72 与 :93。

**理由**：它是本轮唯一同时满足三条的问题——声明零痕迹（hash 与 dict 与“未声明”全等）、数值契约门被跳过、文档明说相反。修法局部（一个 builder + 一个分派点 + 两行文档），不引入新 recipe、不改生产默认、不动数值政策、不影响 baseline 与图入口。风险可界定：只有“带显式声明的源 + linear 策略”这一组合会改变行为，而现有工具与模型都不在该组合上（F1 的真实负载栏），因此不会扰动已冻结的 decode 证据。其余问题（F2/F4/F7）都属“分层规则不一致或控制面缺口”，单修一条收益有限，宜在 F1 之后按“验证同源化”整体处理。

---

## 7. 团队与交叉复核结果

**分工**：队长（Lead）负责地面真值、复跑、集成与验收；expr-probe（表达传递：frontend/RKIR/持久化身份）；exec-probe（执行约束：planner/RKPlan/C 执行链）；cross-check（独立复核，不复用作者结论）。

| 项 | 原结论 | 复核后 | 依据 |
|---|---|---|---|
| expr P1 | source_schedule 永不被静默丢弃 | **推翻**，限定为“graph 入口 / baseline” | 硬门 dag:2637-2643 被 dag:2623 的 graph_ops is not None 包住，单 matmul 走 classic |
| exec F-2(iii) | 文档说 source 可不同、代码要求相同 → 文档超前 | **REFUTED** | 代码比的是 applied；不同 source 确实被 retile（tests/test_two_stage_schedule.py:145，15 passed，队长复跑） |
| C12（数值门绕过） | 候选优先级 1“静默改变数值语义” | **降级 PARTLY**：边界移动成立，数值未变（与 baseline 逐位相同） | [mock] 786432/786432 bits |
| exec F-5 | “>64/slot 完全可执行” | 降为 [源码推断] + 校验器簿记界 | 校验器是创建必经门 |
| 若干数字/措辞 | 24 task、artifact.py:145、schema 纯文档、C 更严所以对 | 已纠正（网格相关数字、artifact.py:147-148、v2 有测试消费者、规则集不同） | cross-review/OVERCLAIMS.md |

**作者脚本审计**：未发现预设结论型作弊；三类系统性问题记入 OVERCLAIMS：单入口观察写成全局断言、手工构造计划当现网语义、源码推断与实测混排。

**队长独立复跑清单**（不采信自述）：expression run_all.sh（10 探针 + 58 单测）、execution 四个探针（用绝对路径与两个既有 .so，4/4 rc=0）、tests/test_two_stage_schedule.py（15 passed）、两个自写探针（lead-verification/probe_plan_identity.py、probe_numeric_gate.py）、以及 mock 执行次数/数值抽查。

**流程教训（记录在案）**：expr-probe 两次因“只测默认通路”误判（_k_loop_module、build_scheduled_linear）。凡涉及分派的结论，须先枚举 build_draft 的全部分支（dag:2644-2664）再下判断。

---

## 8. 基线身份、复现入口、证据路径

**基线**：main @ de4b07a2456033db330c31f10a0df9d908a759c2；工作区含 6 项与本轮无关的既有改动（GRAPH-FUSION-REPORT.md、results/20261028T113459Z-graph-fusion/*、scripts/run-graph-fusion-eval.sh、custom-ops/tools/c/int16_tp_layout_probe.c、tools/analyze_tp_org_consumer.py），本轮未清理、未覆盖、未提交、未 stash。

**环境**（系统 python3 与 .venv 缺依赖，勿用）：

```bash
cd research/tilelang-rknpu
export PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.pydeps-p0
export RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan.so
export RKNPU_PLAN_MOCK_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan_mock.so
```

tilelang 0.1.15；两个 .so 为既有 host 产物，未重建。pyproject.toml 已带 -q，命令行再加 -q 会吞掉汇总行，用 -o addopts=""。

**一键复跑**：

```bash
sh results/20261031T050000Z-intent-fidelity-audit/expression/repro/run_all.sh
cd results/20261031T050000Z-intent-fidelity-audit/execution/repro
for p in probe_plan_trace probe_region_mapping probe_policy_scope probe_verifier_boundary; do python3 $p.py; done
```

**证据目录**：

| 路径 | 内容 |
|---|---|
| results/20261031T050000Z-intent-fidelity-audit/expression/ | REPORT.md（含 §7 P1-修订）、MATRIX.md、8+ 探针、raw |
| results/20261031T050000Z-intent-fidelity-audit/execution/ | REPORT.md（§1 十二条、§2 L6 断链表、§7 F-5 追补）、MATRIX.md、6 探针、raw |
| results/20261031T050000Z-intent-fidelity-audit/cross-review/ | VERDICT.md、OVERCLAIMS.md、CONFLICTS.md、INDEX.md、repro_x/、raw_x/ |
| results/20261031T050000Z-intent-fidelity-audit/lead-verification/ | 队长自写探针（probe_plan_identity.py、probe_numeric_gate.py）与原始输出 |

---

## 9. 本轮明确未验证

1. 一切板端数值与板端执行顺序（本轮零板端证据）。
2. F1 的 scheduled_* 变体在设备上的 task 发射顺序。
3. F2 的触发组合是否出现在真实模型中。
4. F4 的 >64/slot 可执行性仅为源码推断（执行到不了那一步）。**§23.4 更正：现已有 C 侧 64/65 边界负例与板端执行证据，可升级为“已证”。**
5. profile_hash / source_hash 只做非零检查，是否足以防止计划与目标或源码串用。
6. rk_plan.h 与 ctypes 结构体的逐字段偏移比对（仅依赖 rk_plan_create 实际接受）。
7. tilelang_adapter 在 L1 域 K=576 上的 K-block 拒绝是否应给出“如何改用可表达写法”的诊断。

---

## 10. 边界声明

- 本报告只做审查与分类，**不含任何实现改动**；所有“最小修复方向”均未实施。
- 未修改编译器、IR、planner、verifier、runtime、文档、schema、历史 results；未清理既有 dirty 改动。
- 未放宽任何数值或安全检查，未新增 recipe，未改生产默认，未做性能实验，未上板。
- 不声称“已证明全部合法 TileLang 程序都能完整传递”，也不声称“中间层约束问题已全部解决”；结论以 §2 的样本与 §3 的矩阵为界。

**后续**：本轮到此停止，实现方向由用户选择（本报告建议只修 F1）。
---

## 11. 勘误（2026-10-31，F1 修复轮）

本节修正本报告上一版的三处表述错误。原文保留，不删除；以本节为准。

### 勘误 1：L1 形状域在 TileLang 入口**可以**表达（原文 §F5 / §9 有误）

原文写“L1 形状域在 TileLang 入口根本无法表达，只能经 direct-RKIR 到达”。该表述错误，混淆了**一种源写法**与**整个入口**。

- `models/tilelang_matmul.py:make_matmul_prim_scheduled`（只注解、无 per-K 回写）在 K=576 时确实会在 K-block 化阶段被拒（`K-loop tiling does not tile the parameter shapes exactly`），这是**该源写法**的限制。
- `models/tilelang_matmul.py:make_matmul_tiled_tail_prim`（显式 K-tail、每 K 块 FP16 回写）可以表达同一工作负载：`m=256, k=576, n=3072, bm=128, bn=128` 实测编译成功（144 task / 48 submit，C 校验器 ACCEPT）。
- 正确表述：**“注解式但无 per-K 回写的源”无法在 K>256 时表达该调度意图；“显式 K-tail”源可以**。L1 形状域不是 direct-RKIR 专属。
- 证据：`results/20261031T120000Z-f1-verification/raw/probe_combo_exec.out`。

### 勘误 2：上一轮与本轮的执行证据都是 mock / 离线，未到达真实设备

- 上一轮报告中所有“执行一致”结论来自 host mock provider（`librknpu_plan_mock.so`）与 C 校验器 ctypes 调用（`librknpu_plan.so`），板端证据为 0。§5 正向能力表同样应读作“离线 / mock 层面成立”。
- 本轮 F1 修复验证同样如此：C ACCEPT、declared submits == observed mock executions、数值与 ordered per-K FP16 参考逐位一致（`max_abs_err=0`）均来自 mock。
- 板测在本工作区**不可执行**：无 `.env`（板测凭据入口），`scripts/run-board.sh` 是 fail-closed 桩（exit 2），唯一真实通道 `./scripts/run-board-admission.sh` 需要凭据与远端设备。标为 `未验证`，不是“通过”。

### 勘误 3：三类“限制”必须分开表述

| 类别 | 具体项 | 依据 | 中间层可否解决 |
|---|---|---|---|
| **布局 / ABI 约束** | applied M 必须相等（CUBE `co_index` 使物理行不是可切片轴，`custom_ops.c:80-84`）；行窗口不是连续区域 | 布局地址算术 + C 覆盖证明 | 否（属 ABI / recipe） |
| **既有 recipe 支持域** | 原语 `M/Kt/N <= 256`（`custom_ops.h:18-22`）；默认 256 网格；L1 具名策略的形状域 | 已冻结的 recipe 契约 | 否（属支持域声明） |
| **实现 / 校验器边界** | 单 submit `<= 64` task；`RK_MAX_BUFFERS_PER_SLOT=64`；`tile_limit` 无消费者 | 本校验器与 planner 的实现选择 | 是（需先证明复杂度与簿记假设） |

这三类都**不是** RK3588 的普遍硬件限制：硬件层面能做（更大 tile、更多 buffer、行窗口推进）与项目当前是否给出安全实现是两件事。上一版把它们合并叙述，容易被读成“硬件不行”，特此分开。

### 勘误对结论的影响

- §F1（linear 策略分支丢弃声明并绕过数值门）**结论不变**，已在同一个修复轮内修掉：声明的 M/N tile 现在参与组合，数值门在所有 single-matmul 分支共用。
- §F5 分类由“E + TileLang 不可达”改为“E + 仅**部分源写法**不可达”；缺失的仍是声明层（SUPPORT-DOMAIN / capabilities / options 未写该域）。
- §9“未验证清单”第 1 项（板端）依然成立，并已补充不可执行的具体原因。


---

## 12. F1 收尾：名义分块与尾块（2026-10-31）

### 12.1 原来的拒绝为什么不对

调度注解里的 `tile_m`/`tile_n` 说的是**名义分块大小**。真正要发射的任务按剩余范围裁剪：声明 256 行、实际只剩 1 行，就发射 1 行的任务；剩余 192 列就发射 192 列的任务。这条尾块规则是既有语义，`baseline` 一直在用（M=1、声明 256×256 → 36 task，publish 矩形 1×256）。

上一轮修复在 linear 策略分支加了“声明不得大于实际维度”的统一拒绝。结果是同一个声明：选 `baseline` 能执行，选 `merged` 或 `scheduled_baseline` 被拒。**分块上限是 256，但最后一块只有 1 个有效元素**——这不是错误输入，而是正常尾块；把它拒绝，等于让性能策略改变声明的含义与接受条件，正好是 F1 要禁止的事。

### 12.2 修改后哪些输入能执行

| 形状 | 声明 | 策略 | task / submit | 实际 publish 矩形 |
|---|---|---|---|---|
| M=1, K=576, N=3072 | 256×256 | merged | 36 / 1 | 1×256 |
| 同上 | 256×256 | scheduled_baseline | 36 / 12 | 1×256 |
| 同上 | 无声明 | baseline | 36 / 12 | 1×256 |
| M=256, K=576, N=3072 | 256×192 | merged / interleaved / interleaved_reuse | 48 / 1 | 256×192 |
| 同上 | 256×192 | scheduled_baseline | 48 / 16 | 256×192 |
| 同上（真实 TileLang 源 `make_matmul_tiled_tail_prim(bm=256, bn=192)`） | 256×192 | merged | 48 / 1 | 256×192 |

声明值不被改写：计划里每个任务记录的是声明本身（`applied_schedule.tile_m/tile_n`），真实几何另处报告（见 12.4）。

### 12.3 哪些检查继续保护正确性

- 块大小仍限 1..256：`tile_m=512` 仍然拒绝。
- `k_tile` 必须 256、累积顺序必须 ordered ascending FP16 writeback、rounding 必须是 explicit per-K FP16，K 分块必须完整升序：缺一即拒。
- 每个 tile 的读取范围、布局（CUBE / weight 物理行）、padding 状态、依赖顺序、预算：全部沿用原有检查。
- 单 submit 最多 64 个任务：`256×64` + `merged` 算出 144，仍然拒绝（诊断给出策略、几何、需求数与上限）。
- C 校验器继续独立复算几何与覆盖：本轮所有接受组合都过一次 C 校验（`rk_plan_create`），不是只看 Python 侧。

“裁剪”只发生在**最后一块**，不改变整块覆盖、不重叠、不越界——这由 `output_tiles` 的尾块计算与 C 校验器的覆盖证明共同约束。

### 12.4 结果分别来自哪里

| 结论 | 来源 |
|---|---|
| 上述 task/submit 数与 publish 矩形 | 离线编译（Python planner） |
| `C=ACCEPT` | C 校验器离线调用（`../custom-ops/build-host-mvp/librknpu_plan.so`），**未打开设备** |
| declared submits == observed execs、`max_abs_err=0` | mock provider（`librknpu_plan_mock.so`），不是硬件 |
| 板端行为 | **未验证**：本工作区无 `.env`；`scripts/run-board.sh` 是 fail-closed 桩；或走 `./scripts/run-board-admission.sh`（需凭据与远端设备） |

`explain` 的 `schedule.effective` 现在同时给出真实 publish 矩形与 `clipped_from_declared`，因此“声明 256×256、实际发射 1×256”是报告出来的事实，不是隐式改写。

### 12.5 代码与测试

- `passes/dag.py::build_scheduled_linear`：删除“声明不得大于实际维度”的拒绝；声明的 tile 直接作为名义块交给 `output_tiles`（尾块裁剪与 baseline 同一条规则）。
- `codegen/pipeline.py::_effective_geometry`：报告真实 publish 矩形与是否相对声明被裁剪。
- `tests/test_linear_schedule_contract.py`：原先专门期待“tile 超过 extent 就拒绝”的用例改为验证正确契约（5 个策略逐一验证 M=1 + 256×256），并新增 256×192 的正向组合用例与“仍应拒绝”的负例（超上限、超 64 任务）。
- 回归：全量单测 **522 passed, 1 skipped, 120 subtests**；F1 复现探针仍 PASS；无声明生产路径与冻结 decode 计划身份不变（`375fb6c4de…` / `48df00c8ab79a45a…`）。

### 12.6 尚未验证

- 板端执行（同上）。
- 名义块显著大于实际轴时的寄存器级行为（例如 M=1 声明 256×256 时 CUBE 物理行填充的硬件侧代价）。
- 其它组合的逐一枚举：本轮只覆盖 M=1 的 256×256、M=256 的 256×192 与既有 128×128/128×256；未穷举所有 (tile, policy, shape) 三元组。

---

## 13. F2：编译时说能跑，执行器却拒绝（2026-10-31）

### 13.1 准确的复现

形状 `M=64, K=512, N=384`，显式声明 `tile_m=1, tile_n=256`（`source_schedule` + `source_k_rounding=explicit_per_k_fp16`），选项 `numeric_mode=rk_f16_tiled`：

- Python 侧：编译通过，`verify_plan` 通过，得到 **256 个 task、128 个 submit、388 个 buffer、134 个 slot**。
- C 侧：`rk_plan_create()` 返回 **-12 `RK_E_SCHEMA`：too many buffers share one slot`**。
- 同一 slot 里最多塞了 **128** 个 buffer，上限是 **64**。

更细的邻域（同形状）：`1×64` → 单 slot 384 个 buffer，同样被拒；`2×256` → 128/slot 被拒；`4×256` → 32/slot 通过；`16×256` → 8/slot 通过。所以触发条件不是“tile_m=1”本身，而是**细 tile 让同一个 slot 里的 host 侧 buffer 数超过 64**。

拒绝发生在 **plan 创建阶段**（`rk_plan_create`），不是 workspace 创建。

### 13.2 根因

`research/custom-ops/include/rk_plan.h` 里 `RK_MAX_BUFFERS_PER_SLOT = 64u`，注释写明它属于“verifier complexity bounds（MVP）”，用来给成对的 alias/生命周期检查设上界，**不是设备限制**。执行侧按 **slot** 分配一个设备 buffer（`rk_plan_runtime.c` 的 per-slot 分配），一个 slot 里挂多少 host 侧 buffer 与设备能力无关。

所以问题是：**planner 生成了校验器自己的簿记上限管不住的计划**，而 Python 侧没有任何检查复现这条规则。这是“编译时说能跑、执行器拒绝”的直接原因，也是审查里 F2 与 F-5 的**同一个根因**（原报告把 F2 描述成“Python 校验太薄”，方向对但不够精确：真正被违反的是这一条具体规则）。

### 13.3 修复选择

改 **planner 的 slot 分配**，不改 C 校验器：

- `passes/storage.py`：slot 复用多了一个条件——该 slot 的成员数必须仍小于上限，满则开新 slot（其余逻辑不变：仍然是“生命周期不重叠 + 容量足够”才复用）。
- `plan.py`：新增 `MAX_BUFFERS_PER_SLOT = 64`，与 C 头文件同名同值，并有测试从 `rk_plan.h` 解析比对，防止两边漂移。
- `passes/verify.py`：`verify_plan` 复算每个 slot 的 buffer 数，超过上限就拒绝并给出 `slot_id`、实际数量与上限。**Python 现在与 C 对这条规则一致**，不会再交出注定被拒的计划。

为什么不是别的做法：

| 备选 | 为什么不选 |
|---|---|
| 放宽/删除 C 的 64 上限 | 那是在放宽校验器；而且没有证据表明放宽后 pairwise 检查的复杂度可接受。C 的规则本身没错，错的是 planner 越界。 |
| 一律拒绝 `tile_m=1` | 掩盖具体原因：`4×256`、`16×256` 这类细 tile 本来就能跑，问题只出在 slot 成员数。 |
| 改上层调度/自动放大 tile | 会改写用户的显式要求，正是本轮之前修掉的 F1 病症。 |
| 只在 Python 加“预计会被 C 拒绝”的预判并报错 | 只是把失败提前，用户仍拿不到能跑的计划；而 slot 拆分是语义等价的（同一 slot 内 buffer 本就时间不重叠）。 |

拆分是否安全：同一 slot 内的 buffer 在时间上互不重叠（既有 alias 证明），把它们放进不同的设备 buffer 不会引入别名冲突，也不会改变任务顺序、数值或依赖。代价是多几个设备 buffer 对象与每个 slot 的固定预留（32 KB + 64 B），本轮实测设备请求从 5.70 MB（136 slot）级别，仍在预算内。

### 13.4 现在哪些能跑、哪些仍不能

| 案例（64×512×384，声明） | 修复前 | 修复后 | 设备请求 |
|---|---|---|---|
| 1×256 | C 拒绝（128/slot） | **C 接受**，256 task / 128 submit，峰值 64/slot | 5.70 MB |
| 1×64 | C 拒绝（384/slot） | **C 接受**，768 task，峰值 64/slot | 6.22 MB |
| 2×256 | C 拒绝（128/slot） | **C 接受**，128 task，峰值 64/slot | 3.13 MB |
| 4×256 | 接受 | 接受，峰值 32/slot（**计划未变**：38 slot / 1.88 MB） | 1.88 MB |
| 16×256 | 接受 | 接受，峰值 8/slot（**计划未变**：14 slot / 0.97 MB） | 0.97 MB |

仍然拒绝的：超过 64 个成员且无法拆分的计划（例如把多个 persistent 常量塞进同一 slot——那本来就要求独占 slot）；设备/主机预算超限；以及其它与 slot 无关的既有规则（覆盖、依赖、别名重叠）。

**兼容性**：只有“某个 slot 会超过 64 个成员”的计划才会改变 slot 划分；生产包里 268 个计划全部 `verify_plan` 通过，最大成员数 16，冻结的 decode merged 计划身份仍是 `48df00c8ab79a45a…`，未声明的 F1 计划身份仍是 `375fb6c4de…`。

### 13.5 证据来源

| 结论 | 来源 |
|---|---|
| 修复前 C 拒绝、修复后 C 接受 | C 校验器离线调用（`../custom-ops/build-host-mvp/librknpu_plan.so`），未打开设备 |
| task/submit/slot/预算数字、保存加载后 `plan_hash` 与判定不变 | 离线编译与 `save_plan`/`load_plan` |
| C 侧兼容回归 | `ctest -R 'plan|verif|contract'`：`rk_plan_contract`、`rk_plan_host` 均 Passed |
| **C 的数量规则仍在原位、且边界正好在 64** | 描述符层把 64 个两两时间互斥的 buffer 塞进同一个新 slot → `rk_plan_create` 返回 0（接受）；塞 65 个 → **-12 "too many buffers share one slot"**。这条负面证据不经过任何 Python 检查，说明修复没有靠“把 C 放宽”（`results/20261031T160000Z-f2-slot-bound/raw/probe_c_negative.out`） |
| 板测四个计划 = 当前代码产出 | 用当前代码重新生成 B1–B4，`plan_hash` 与设备实跑的计划逐一相等：`41d8bb96…`、`af9d1bbc…`、`91c76b92…`、`dc5f8993…` |
| 生产包 268 个计划仍可加载并 `verify_plan` 通过 | 离线扫描 |
| Python 单测 | 全量 527 passed / 1 skipped / 123 subtests（含新增 `tests/test_slot_budget_contract.py`） |
| 设备执行 | 见 §14（F1/F2 板端最小验证） |

---

## 14. 板端最小验证（2026-10-31，RK3588）

### 14.1 连接与设备身份（实测）

- 凭据来自**仓库根** `.env`（`ORANGE_PI_HOST/USER/PASSWORD/WORKSPACE`），用与 `scripts/run-board-admission.sh` 相同的 askpass + `setsid -w ssh` 方式；本轮实测连接成功。
- 设备：RK3588 OPi 5 Plus，`Linux 6.1.43-rockchip-rk3588-rknpu098 aarch64`，RKNPU 驱动 `v0.9.8`，`/dev/dri` 有 `card0/card1/renderD128/renderD129`，`RKNPU_CORE_MASK=1`（固定单核）。
- C 库：**设备现场用当前源码重建**，`librknpu_plan.so` sha256 `c6a6dd1f…b64709`，源包 sha256 `30d96f1c…7e59ec2a`，cmake Release `-O3 -DNDEBUG`，gcc 11.4.0，`rk_target_check` rc=0，`abi_version=1`。没有复用远端历史库。
- 证据目录 `results/20261031T150000Z-f1-f2-board/`（`REPORT.md` + `raw/` 24 项 + `SHA256SUMS.txt`）；凭据自检：证据树内 `grep -F` 密码字面量无命中。

### 14.2 四案结果（全部 [实测]，非推导）

| 案例 | 计划（task/submit，publish） | 设备运行时自报 | 与声明数值契约参考的误差 | 结论 |
|---|---|---|---|---|
| B1 M=1 声明 256×256 + merged | 36 / 1，12×(1×256) | 一致 | r0/r1 max 1.95e-3（1/3072 元素，2 ULP），r2 0 | 通过 |
| B2 M=256 声明 256×192 + merged | 48 / 1，16×(256×192) | 一致 | max 3.12e-2（311/786432 元素，1–2 ULP） | 通过（见 14.3） |
| B3 同 shape 未声明 baseline（生产路径对照） | 36 / 12，12×(256×256) | 一致 | **与 B2 逐位相同**（三轮） | 通过（对照） |
| B4 F2 案例 64×512×384 声明 1×256 | 256 / 128，64×(1×256)+64×(1×128) | **128 次 chain submit，每次 2 task** | max 3.12e-2（14/24576），三轮 allclose 真 | 通过 |

- 输出全部 finite；同输入重复（同进程 r0/r1 与两个独立进程）输出逐位相同；换输入（r2）输出确实改变。
- B4 关键点：**没有任何 slot/BO 相关运行时错误**，无分配失败，跑前跑后 `dmesg` 无新增 RKNPU 行，跑完核占用回落到 Core0 3%。计划峰值正好压在 `RK_MAX_BUFFERS_PER_SLOT=64`（分布 {1,64}），设备按 136 个 slot 正常分配。

### 14.3 数值残差怎么看（重要，别读成“设备算错”）

两种参考不是一回事：

- **计划声明的数值契约**是“每 256 宽 K 块升序、每块 FP16 写回”。按这个契约算，四案都在容差内（B1/B4 三轮 allclose 全真；B2/B3 的 0.04% 元素差 1–2 个 FP16 ULP）。
- **纯 FP64 数学**（不插入 FP16 中间舍入）是另一个参考。用它对表，B2/B3 自然超差（max 7.3e-2，全元素不同），B1/B4 仍在容差内。这不是本轮改动引入的：**生产默认路径 B3 与声明路径 B2 三轮输出逐位相同**。
- 我离线复核了 311 个失配元素：它们是 FP16 舍入近并列点，差 1–2 ULP；设备与理想化“fp64 切片后舍入”的顺序差异足以跨过边界。把它判为“设备数值错误”证据不足，判为“参考模型与设备求和顺序不同”更准确；本轮不据此下任何正确性结论。

**F1 的板端结论只落在两件可验证的事上**：① 声明的几何与策略确实落到执行（task/submit/publish/`applied_schedule` 一致）；② 声明路径的数值与生产默认路径**逐位相同**（B2≡B3），即声明没有偷偷改变数值。

### 14.4 F2 的板端结论

F2 的正确结果是**计划被修好并真的跑起来**，不是提前拒绝：B4 在设备上完成 256 个 task、128 次 chain submit，与计划的 128 个 submit 一致，无数值或资源错误，输出与声明契约参考一致。原先 C 校验拒绝的是**簿记上限**，不是设备能力，这一点现在有实测支撑（同形状、同 64 上限的 slot 划分，设备侧无任何异常）。

### 14.5 计划身份与收尾（实测）

- 设备上实际执行的四个计划与本地 `campaign-v2/plans/` **逐字节相同**：B1 `8653cd77…f411427`、B2 `25e22a85…e4c1a03c`、B3 `a9792b36…bd12b3b4`、B4 `605eab95…b9b5df88`；对应 `plan_hash` 与当前代码重算一致（§13.5 末行）。
- **B4 跨进程重复已完成**：独立进程只跑 B4（同一计划、同一 seed 夹具），三轮输出与 run2 **逐位相同**（`bitwise_equal` 全 True；r0/r1 sha256 `2f81df76…`、r2 `0d8716a8…`），status PASS、256 task / 128 submit、`plan_hash` 不变。
- **远端临时资源已回收**：删除前校验绝对路径与目录名严格相符，再 `rm -rf` 该确切目录（239 MB，含上传的 source/campaign 包）；删除后 `/mnt/ssd/workspace` 下本轮条目为 0。本地证据已按 `SHA256SUMS`（57 条）复算，不依赖远端留存。
- **凭据卫生**：证据树按密码字面量全字符扫描无命中；顺带修掉审查侧临时辅助脚本用 `exec` 调 ssh 导致 EXIT trap 不触发的问题，`/tmp` 残留的 25 个 askpass 助手文件（仅两行脚本、引用环境变量，不含凭据）已删除，现残留 0。该脚本属审查侧助手，不在仓库生产代码里。
- 上一轮 `contract-probe` 的证据目录此前误落在仓库根 `results/`（未跟踪、未被忽略），已整体移入 `research/tilelang-rknpu/results/20261031T120000Z-f1-contract/`，与仓库证据布局一致并进入 `.gitignore` 覆盖范围。

### 14.6 仍未验证

- 只测 `core_mask=1`；2 核/4 核未测（超出本轮授权范围）。
- B4 已有跨进程重复（见 §14.5）；B1–B3 同样有跨进程重复。
- 没有在设备上做 C 校验器的负面用例（例如人为构造超上限 slot 的计划）——“C 拒绝”证据来自离线校验器。
- 未做性能测量；设备历史上（约 7–10 小时前）有多次 RKNPU job timeout/soft reset，与本轮无关，未追查。
- 无法给出“设备数值与精确参考逐位一致”的结论：FP16 中间舍入 + 设备内部求和顺序使其不成立，本轮也不要求。

---

## 15. F3：一个字段同时当三种信息用（2026-10-31）

> **勘误提示（见 §16）**：本节把闸门判据改成“排除 origin=derived”，这会改变分派与接受路径；
> 本节末尾“没有改任何既有接受/拒绝路径”“计划身份无变化”两句作废。以 §16 为准。

### 15.1 旧行为

RKIR 里的 `source_schedule` 同时被三种东西填进去：

1. 用户写的显式注解（`rknpu.schedule` / `rknpu.tile_m` / `rknpu.tile_n` / `rknpu.mn_order`）；
2. 源码已经证明的分块几何（内核里真实的 `bm/bn` 与精确网格）；
3. 一个结构副作用的连带结果——某些提取器只在“内核把每 K 块结果写回 FP16”时才把这份几何写出来。

而下游用**字段是否存在**当开关：`dag.py` 把带该字段的 op 当作“用户要求了 M/N 调度”，拒绝时会说 “explicit M/N scheduling …”；`tile.py` 的 K-rounding 数值门也在字段存在时才生效；`explain` 则把它一律叫作 `declared`（已声明）。

### 15.2 逐项回答

| 问题 | 答案 |
|---|---|
| 用户是否写了显式注解？ | 只有 `_schedule_annotation()` 为真时才算；但在此之前没有地方把这件事记录下来。 |
| 源码是否已经表达 tile 几何？ | 是。`_extract_*` 都有 `source_mn_tiling`（block_m/block_n/grid/coverage），与注解无关地存在。 |
| frontend 是从源码推导，还是产生新的执行要求？ | 产生的是**几何信息**，不是用户要求。但只有一个提取器在“内核写回 FP16”时才把这份几何写成 `source_schedule`（`tilelang_adapter.py:5517-5522`），于是“几何存在”和“用户要求”被混成一个字段。 |
| 没有这些信息时默认值在哪补入？ | `tile.py:57-59`（`MNTileSchedule()` 默认 256×256、m_major）；前端的 `narrowing_base` 明确写 “backend target policy: every 256 K elements”。 |
| 哪个字段/条件决定下游采纳？ | 旧行为：`source_schedule` 是否存在；新行为：字段 + `source_schedule_origin`。 |
| 这些状态是否被混淆？ | 是。表现有三处：`explain` 把推导几何叫 `declared`；`dag.py` 的闸门把推导几何当用户要求；没有任何记录说明该几何从哪来。 |

### 15.3 哪些差异合理，哪些确实是错的

**合理的（保留并解释，不改）**

- **plain 源码走后端目标策略**：内核用 FP32 累加器跨多个 K 块、只在最后窄化一次，前端会记录 `source_accumulator_dtype=float32`、`narrowing_base="backend target policy: every 256 K elements"`；guarded-tail 路径同样用 `guarded_rect_tail_v1` 说明这一点。这是文档化的降级，不是偷偷改数值。
- **数值不同就不该产生同一计划**：`K>256` 时“每 K 块回写”和“末端窄化”是两个数值程序，计划相同才是错的。
- **显式注解可以重新分块**：源码 128×256、注解 256×256 时，计划按 256×256 出 1 个任务（矩阵 C 行），这是既有契约允许的等价改写。

**确实错的（已修）**

- `explain` 把**推导**几何报成 `declared`，用户无法分辨“我要求的”和“编译器看出来的”。
- `dag.py` 的两处闸门（多节点 dataflow 与分派选择）用“字段存在”当“用户要求”，把推导几何也算作 explicit；一旦有提取器在图上盖了推导戳，用户会收到一条把要求算在自己头上的拒绝。
- 没有任何地方记录来源，所以“为什么这个计划用 128×256 而不是 256×256”在报告里查不到。

**仍存在但已可见的差异（保留，写进报告）**

同一个数值程序（`K=256`，只有一个 K 块，所以“每 K 块回写”与“末端窄化”是同一次舍入）用两种写法：

| 写法 | 源码块 | rounding | 来源 | 采用的网格 | task | publish 矩形 |
|---|---|---|---|---|---|---|
| plain FP32 累计 | 128×256 | `single_k_tile_fp16` | 无 | `default_grid` | 1 | 256×256 |
| 每 K 块 FP16 回写 | 128×256 | `explicit_per_k_fp16` | `derived` | `source_geometry` | 2 | 128×256 |

两者计划不同。**理由**：采纳源码分块的条件是“这份源码本身证明了每 K 块 FP16 回写”——也就是它确实是一个按 K 分块、写回契约与目标一致的核；plain 内核不做这个承诺，就按后端目标策略在默认网格上规划。这条规则现在**写在代码注释里、体现在 `grid_source` 字段里、可以逐例核对**，而不是靠“某个字段恰好有没有被盖出来”。不把它强行拉平，是因为拉平的唯一一致做法会让 plain 多 K 块源码全部被数值门拒绝（本轮试过，14 个 ragged/K-loop 测试与“plain 源按目标策略执行”这一文档化行为同时被打断），那等于用一致性换掉一个已声明的支持域。

### 15.4 控制来源与优先规则

| 来源 | 载体 | 作用 | 能否被改写 |
|---|---|---|---|
| 数学 / 索引 / 依赖 | 源程序 + RKIR op 图 | 不可协商 | 否 |
| 数值要求 | `numeric_contract` / `source_k_rounding` / `source_accumulator_dtype` | 决定哪些 K 分块合法 | 否（调度不能覆盖） |
| 源码已表达的分块几何 | `source_mn_tiling`（始终存在） | 采纳为 `source_geometry` 的依据 | 只有在内核证明每 K 块 FP16 回写时才被采纳 |
| 用户显式调度 | `source_schedule` + `source_schedule_origin="annotated"` | 必须落实或准确拒绝 | 可由注解重新分块（矩阵 C） |
| 后端默认 | `MNTileSchedule()` 默认值 + `backend target policy` | 没有前两者时补入 | 是（由策略） |
| 最终执行结构 | 任务几何 / submit 分组 / publish 矩形 | 计划实际内容 | 由上表决定 |

四条硬规则：① 数值与安全要求优先于任何调度字段；② 采纳源码分块必须由源码自身证明（每 K 块 FP16 回写）或由用户注解；③ 推导信息不得冒充用户要求（闸门按 provenance 判断）；④ 默认值只在没有前两者时补入，并在报告里标成 `default_grid`。

### 15.5 局部修改

| 位置 | 改动 | 为什么够 |
|---|---|---|
| `frontend/tilelang_adapter.py:5517-5522`、`:3983`、`:5920` | 三处 stamping 点都记录 `source_schedule_origin`（`annotated` / `derived`） | 把“谁要求的”变成显式数据，而不是从字段存在性推断 |
| `passes/dag.py:648-654`、`:2669-2671` | 闸门判据从“字段存在”改为“字段存在且 origin 不是 derived” | 推导几何不再冒充用户要求；direct-RKIR 手写 attrs（无 origin）仍按用户要求处理 |
| `codegen/pipeline.py::source_identity` | 报告新增 `source_schedule_origin` 与 `mn_grid_source`（`annotation` / `source_geometry` / `default_grid`） | explain 能回答“要求来自哪、最终用哪种网格” |

**没有改**：planner 采纳逻辑、数值门、默认值、`source_mn_tiling`、任何既有接受/拒绝路径。

### 15.6 验证矩阵（全部离线，实际数字）

| 案例 | 输入表达 | frontend 保留 | 来源 / 网格 | task / submit | publish | 结论 |
|---|---|---|---|---|---|---|
| A 无注解、内核每 K 块回写、块 128×256 | 源码块几何 | `source_mn_tiling` + 推导 schedule | derived / source_geometry | 2 / 2 | 128×256 | 接受 |
| B 同源 + 匹配注解 | 注解 128×256 | 同上，origin=annotated | annotated / annotation | 2 / 2 | 128×256 | 接受，**与 A 执行相同** |
| C 同源 + 注解 256×256 | 注解要求重新分块 | 注解覆盖源码块 | annotated / annotation | 1 / 1 | 256×256 | 接受（契约允许重新分块） |
| D 注解要求末端窄化 | 显式数值要求 | — | — | — | — | 拒绝 `RK_E_NUMERIC_CONTRACT`（frontend 对照证明点） |
| D2 注解但未写 rounding 字段 | 只有调度要求 | 用已证明的 rounding | annotated / annotation | 2 / 2 | 128×256 | 接受 |
| E 单块、无注解 | 无调度信息 | `source_mn_tiling` 全量 | derived / source_geometry | 1 / 1 | 256×256 | 接受（与默认网格重合） |
| F 同一计算、同一舍入、同一控制的两种写法（A/B/G） | 注解 / 推导 / 直接 attrs | 三份都到 RKIR | annotation / source_geometry | 2 / 2 | 128×256 | **执行相同，仅来源不同** |
| G direct-RKIR 手写 `source_schedule` | 无 origin 字段 | 原样保留 | 无 origin / source_geometry | 2 / 2 | 128×256 | 接受，闸门按“用户要求”处理 |
| H 编译→保存→加载→重验证 | 注解 128×256 | — | — | 2 / 2 | 128×256 | `plan_hash` 与几何保持（`hash_equal=True`） |
| I guarded-tail 无注解 | 源码 F32 累计 | `guarded_rect_tail_v1` + 目标策略 | 无 / default_grid | 2 / 1 | 13×33 | 接受（文档化降级） |
| J guarded-tail 每 K 块回写 + 注解 | 注解 | — | annotated / annotation | 2 / 1 | 13×33 | 接受，**与 I 执行相同** |
| K guarded-tail 注解 + 末端窄化 | 注解 | — | — | — | — | 拒绝 `RK_E_NUMERIC_CONTRACT` |
| L 推导记录 + linear 策略（M=1 名义 256×256 + merged） | 注解（F1 用例） | — | annotated / annotation | 36 / 1 | 1×256 | 接受（F1 回归保持） |

判据不只比 hash：每行都给 task/submit 数、publish 矩形、来源与网格，并核对 F1 的尾块、F2 的 slot 上限这些既有契约。

### 15.7 F1 / F2 回归与身份

- 全量 Python 单测：**536 passed / 1 skipped / 123 subtests**（含新增 `tests/test_schedule_provenance.py` 9 例；未改任何既有断言）。
- 冻结 decode merged 计划：`48df00c8ab79a45a…`（未变）。
- 板测用过的四个计划身份未变：`41d8bb96…`、`af9d1bbc…`、`91c76b92…`、`dc5f8993…`。
- F1 探针 `probe_f1_regression.py`、`probe_tail_contract.py` 仍 PASS；F2 探针 `probe_f2_slot_bound.py`、`probe_c_negative.py` 仍 PASS（本轮改动不触碰 storage/verify）。

### 15.8 实际验证了什么、还差什么

- **离线**：frontend 产物（origin、source_mn_tiling、rounding）、planner 几何、报告字段、A–L 全部案例。
- **C 校验器 / mock**：本轮改动不改变任何计划字节（`plan_hash` 与板测计划逐一相同），因此没有新的计划需要送 C 或 mock；既有 C/mock 结论继续有效。
- **板测**：本轮不重复上板。理由是可核对的：本轮只改 RKIR 元数据（`source_schedule_origin`）、报告字段与两个闸门判据，计划字节不变——冻结 decode 计划与四个板测计划哈希逐一复核相同，重跑只会得到同一批计划。
- **未验证**：① 推导几何在“多节点图”上的表现（当前没有任何提取器在图上盖推导戳，闸门改动属于前瞻性守卫，不是已触发缺陷的修复）；② “plain 单块源码是否也应采纳源码分块”这一策略问题未定，本轮按“源码承诺每 K 块回写才采纳”处理并记录；③ 板端未复测（理由如上）。

---

## 16. F3 勘误与修复：来源标记不能当闸门（2026-11-01）

### 16.1 审核复现（修复前，实测）

两个串联 matmul，单个节点 256×256、声明 128×256、`source_k_rounding=explicit_per_k_fp16`，只改 `source_schedule_origin`：

| 输入 | task / submit | publish 矩形 | 结论 |
|---|---|---|---|
| 无 origin | 4 / 4 | 128×256 | 接受（按调用者要求处理） |
| `origin=annotated` | 4 / 4 | 128×256 | 接受 |
| `origin=derived` | **2 / 2** | **256×256** | **被接受，且静默按固定 256 网格规划** |

再把声明改成当前不支持的 128×128：

| 输入 | 结果 |
|---|---|
| 无 origin | 拒绝：`stage0 tile_n must be 256 so each consumer K slice has one producer tile` |
| `origin=derived` | **接受 2 / 2，固定 256 网格**（绕过了上面的拒绝） |
| `origin=guess`（未知值） | 当时按“非 derived”处理并被接受 |

从**真实 TileLang 源**提取（`M=K=N=256`、源 tile 128×256、每 K 块 FP16 回写）会得到 `origin=derived`；把它组成上面的两节点 RKIR 后，行为与上表一致。

### 16.2 根因

上一轮我把“来源”写进了闸门判据：`dag.py` 的两处 admission 条件排除了 `origin=derived`，于是**标一个字段就能跳过调度与契约检查**，落到该路径的固定 256×256 网格上。来源本应只解释“这是谁提的要求”，不能决定“要不要检查”。

同时暴露两个次生问题：

- `source_schedule_origin` 没有取值校验，未知值被当作“非 derived”继续执行；
- 报告把 direct-RKIR 手写属性写成 `source_geometry`（或空），把调用者写的要求混进“源码推导”里。

顺带修掉一个真实的小 bug：`ir.py` 的“matmul 缺少 NumericContract”拒绝路径用了未导入的 `RK_E_NUMERIC_CONTRACT`，手写 RKIR 会得到 `NameError` 而不是诊断码。

### 16.3 最终来源与执行规则

| 来源标记 | 含义 | 对执行的要求 | 后端怎么处理 |
|---|---|---|---|
| `annotated` | 用户写的 `rknpu.schedule/tile_m/tile_n/mn_order` | 必须落实或准确拒绝；允许按既有契约重新分块 | 单节点：按声明规划；两节点：仅受限两阶段契约；图：拒绝 |
| `derived` | 提取器证明的源码块几何（内核每 K 块 FP16 回写） | **与 annotated 完全相同的检查**；区别只在解释文案 | 同上；做不到就拒绝，不退回 256 |
| 省略 | direct-RKIR 调用者手写 | 与 annotated 相同 | 同上；报告标成 `caller`，不称“frontend 推导” |
| 无 `source_schedule` | 没有调度要求 | 不涉及 | 走既有默认网格 / 目标数值策略，报告标 `default_grid` |

硬规则：① 来源只用于解释与文案，不参与“能否跳过检查”；② 数值与 schema 检查在任何 admission 结论之前跑；③ 未知 origin 直接拒绝；④ 不能落实的组合明确拒绝，不静默改用 256；⑤ 允许重新分块时必须在报告里写出“声明 → 采用 → 是否 reblocked”。

### 16.4 局部修改

| 位置 | 改动 |
|---|---|
| `passes/tile.py`：新增 `ORIGINS = {annotated, derived}`，`mn_schedule_from_attrs` 校验取值；缺失仍表示调用者手写 | 未知 origin 在任何 planner 路径上都会拒绝 |
| `passes/dag.py`：dataflow 闸门恢复“有调度就检查”，且先跑 `mn_schedule_from_attrs`（数值/schema），再按来源给不同文案 | derived 不再跳过；数值问题优先报成数值问题 |
| `passes/dag.py`：分派处的 `scheduled` 列表恢复为“字段存在”，文案按来源区分 | derived 与 annotated 走同一套 admission |
| `codegen/pipeline.py`：`source_schedule_origin` 增加 `caller` 语义；`mn_grid_source` 增加 `caller`；`schedule` 段新增 `reblocked` | 报告能回答“谁提的、采用什么、是否重新分块” |
| `python/rknpu_backend/ir.py` | 补 `RK_E_NUMERIC_CONTRACT` 导入 | 缺契约时给诊断码而不是 NameError |

### 16.5 修复后结果（实测）

| 输入 | task / submit | publish | 结论 |
|---|---|---|---|
| 两节点 + `origin=derived`（真实提取组成的 RKIR） | 4 / 4 | 128×256 | 接受，`plan_hash=dca70de827…` |
| 两节点 + `origin=annotated` | 4 / 4 | 128×256 | **同一计划**（hash 相同） |
| 两节点 + 无 origin | 4 / 4 | 128×256 | **同一计划** |
| 两节点 + 128×128（三种来源） | — | — | 全部拒绝 `RK_E_UNSUPPORTED`（`stage0 tile_n must be 256 …`） |
| 两节点 + `origin=guess` | — | — | 拒绝：`source_schedule_origin must be one of ['annotated','derived'] …` |
| 三节点 dataflow，无调度 | 3 task | — | 接受，`default_grid` |
| 三节点 dataflow，`derived` | — | — | 拒绝：“this source proves M/N blocks that no admitted multi-node contract can honour …” |
| 三节点 dataflow，`annotated` / 无 origin | — | — | 拒绝：“explicit M/N scheduling is admitted only for the restricted two-stage …” |
| 三种来源 + `fp32_final_narrowing` | — | — | 全部拒绝 `RK_E_NUMERIC_CONTRACT`（来源不能绕过数值契约） |

保存 → 加载 → 重验证：`plan_hash`、task/submit、publish 矩形保持（`tests/test_schedule_provenance.py::AdmissionTests`）。

### 16.6 C 验证与 mock（本轮新增的接受计划）

- 计划：`results/20261031T180000Z-f3-origin-gate/two-node-derived-128x256.rkplan.json`，`plan_hash=dca70de827029e37…`，4 task / 4 submit。
- C 校验器（离线 `librknpu_plan.so`）：**ACCEPT**。
- mock：声明 4 次 submit = 观测 4 次执行；输出 finite；与“Z=fp16(X@W1)、Y=fp16(Z@W2)”（K=256 单块，末端窄化一次，float64 计算后窄化）参考 **max_abs_err = 0**。

### 16.7 F1 / F2 与受影响回归

- 全量 Python 单测：**542 passed / 1 skipped / 129 subtests**（`tests/test_schedule_provenance.py` 扩到 15 例 6 subtests，含 A–H）。
- F1：显式声明不被 linear 策略丢弃、名义 tile 大于有效维度按尾块执行、256×192 + merged —— 两个 F1 探针重跑 PASS，四个板测计划身份 `41d8bb96…`/`af9d1bbc…`/`91c76b92…`/`dc5f8993…` 与冻结 decode `48df00c8ab79a45a…` 逐一复核未变。
- F2：slot 上限分配与 C 边界（64 接受 / 65 拒绝）探针重跑 PASS。
- 单节点路径：本轮只改 admission 判据与报告字段；单节点行为未变（`tests/test_schedule_provenance.py` 的 A/B/C/E 行复跑一致）。

### 16.8 对 §15 的勘误

| §15 原文 | 更正 |
|---|---|
| “没有改 planner 采纳逻辑、数值门、默认值…也没有改任何既有接受/拒绝路径” | **错**。§15 的闸门判据排除 `origin=derived`，改变了多节点图的分派与接受；本节的修复正是撤销这个排除并把检查恢复。 |
| “计划身份变化：无” | **只对四个冻结样本成立**。它们没有多节点 derived 图，所以哈希相同；不能据此推断“接受路径没变”。 |
| “本轮没有新的计划需要送 C / mock” | **错**。修复前 `origin=derived` 让两节点图落到固定 256 计划（2/2）；修复后同一输入产出 4/4、128×256 的新计划，已按 §16.6 送 C 与 mock。 |
| “未验证：推导几何在多节点图上的表现” | 已复现并修复（§16.1、§16.5）；剩余未验证项见 §16.10。 |

§15 的 A–L 矩阵与“来源 vs 网格”的解释仍然有效，但其闸门描述与“无行为变化”的结论作废。

### 16.9 未验证

- 只覆盖两节点受限两阶段与三节点 dataflow 两种图形态；其它图组合（含 FFN 分支）没有逐一枚举。
- 三节点 dataflow 的 derived 拒绝属“当前不支持”，本轮**没有**实现多节点调度；拒绝文案是结论，不是能力。
- 板端：见 §16.10（本轮修复改变了接受计划，做了最小必要板测）。

### 16.10 板端验证（2026-11-01，RK3588，实测）

**设备与库**：RK3588 OPi 5 Plus，RKNPU v0.9.8，`6.1.43-rockchip-rk3588-rknpu098`，core mask 1；设备现场用当前源码重建 `librknpu_plan.so`，sha256 `c6a6dd1f4c14508478740ca256431cd30f9aa07d6e4583727ec233c9a1b64709`（与上一轮同值，构建确定性），源包 `30d96f1c…7e59ec2a`，`cmake Release -O3`。

**计划**：`results/20261031T180000Z-f3-origin-gate/two-node-derived-128x256.rkplan.json`，`plan_hash=dca70de827029e37791cafd83b0d4f67ef9aa3f8f30afa0517eb647e74ee5678`，文件 sha256 `ba1060009feb2e7237e22f7df5d2d7cae43113b8fe13a28f9b4181a1898c058c`。板测侧用当前代码独立重生成，与本地**逐字节相同**；`derived` / `annotated` / 缺省三变体同一 `plan_hash`。

**计数与结构**：计划与运行时自报一致 —— **4 task / 4 submit**（每次 chain、各 1 task），**2 个 128×256 publish 矩形**。

**执行**：三轮输出全 finite；同输入 r0/r1 **逐位相同**；换输入后输出改变且对参考 allclose 通过；无任何运行时错误，`dmesg` 前后无新增 RKNPU 行，跑完 GPU load 回落 0%，无脏上下文。

**数值（未放宽，判定为未通过）**：对“Z=fp16(X@W1)、Y=fp16(Z@W2)（K=256 单块、末端窄化一次、float64 计算后窄化）”的参考，r0/r1 `max_abs=2.5e-1`、**167/65536 元素不同**、严格 `allclose(8e-3, 8e-3)` **未过**；r2 `max_abs=5.0e-1`、141 个不同元素但 allclose 通过。差异为 1–5 个 FP16 ULP，散布于全部行列（不按 128 行 tile 聚集），近舍入并列点富集；把参考换成“两级 FP32 部分和”反而更差（599/747）。

**离线敏感性分析**（`sensitivity.py`）：把中间量 Z 中 **1% 的元素移动 1 个 FP16 ULP**，Y 就出现 `max_abs≈1.0`、约 200 个元素超出 8e-3 容差、1.2 万个元素变化。也就是说，**Z 上极少量 1-ULP 差异就足以产生观测到的 Y 残差量级**；设备残差比这个实验更小（167 个元素），与该机制一致，而不是“计划把数学算错”。

**裁定**：执行链成立（身份、计数、publish、重复性、无错误全部符合）；**数值验收未通过**，我不放宽容差、也不把这一项写成通过。它属于“链式两级 GEMM 在 FP16 中间写回下的参考保真度/验收口径”问题：设备与“理想化 float64 参考”之间必然存在中间量 1-ULP 差异，而该差异会被第二级 GEMM 放大。要判定是否可接受，需要新的、与设备数值语义一致的参考模型（例如按设备求和顺序实现的中间量模型），这超出本轮范围。

**未做到**：只测 core mask 1；仅一个进程（无跨进程重复）；设备上只跑了 `derived` 这一份（`annotated` / 缺省的等价性目前只有编译期证据：同一 `plan_hash`）。

**回收与卫生**：远端隔离目录删除前做 `test -d` + `basename` + 绝对路径三重校验，删除后确认不存在（workspace 下本轮条目 0）；证据目录密码字面量扫描无命中；`/tmp` askpass 残留 0；`SHA256SUMS` 33 条。

---

## 17. 两级 GEMM 板测数值失败：分级隔离（2026-11-01）

### 17.1 原失败复现（逐位）

用**原始归档数组**（X/W1/W2 的 sha256 见 `results/20261101T000000Z-f3-stage-isolation/identity.json`）重跑原串联计划：

- 输出数组 sha256 `46a5be41caa2a32d…` 与上一轮归档的 `board_y_sha256` **完全一致**；167 个元素不同、`max_abs=0.25`，与上一轮数字一致。**FAIL 逐位复现**。
- 三轮全 finite、同输入 r0/r1 逐位相同、无任何运行时错误。
- 按计划 NumericContract（`atol=8e-3, rtol=8e-3`，未放宽）严格判定：**167 个元素不同，其中 2 个超过容差** —— `(135,33)` 4.410156 vs 4.367188（11 ULP，容差 4.29e-2，擦边）与 `(135,231)` −0.53076 vs −0.56543（该量级 71 ULP，容差 1.25e-2）。上一轮“167 个超标”的表述应更正为“167 个不同、2 个超标”。

### 17.2 三个计划的实际差异

| | 任务模板/几何 | k0/k_index | publish | task/submit |
|---|---|---|---|---|
| 串联计划 | 4×`co.mm.f16.v1` (128×256) | 0 / 无 | 2×(128×256) | 4 / 4 |
| S1（X@W1） | 2×`co.mm.f16.v1` (128×256) | 0 / 无 | 2×(128×256) | 2 / 2 |
| S2（Z@W2） | 2×`co.mm.f16.v1` (128×256) | 0 / 无 | 2×(128×256) | 2 / 2 |

离线核对：**串联 stage-0 的 task 列表与 S1 完全相同，stage-1 与 S2 完全相同**（模板、输出几何、k0、k_index 逐项相等）。差别只在绑定（S1/S2 各自只有 4 个 binding）与提交组织（独立执行各 2 submit）。S1 与 S2 的 `plan_hash` 相同（`bcc9e176…`），因为结构一致、只是运行期输入不同。

### 17.3 Z / Y 逐级比较（板端实测，每项 3 轮）

| 比较 | bitwise | 差异元素 | max_abs | RMSE | 超容差 |
|---|---|---|---|---|---|
| Z_ref vs Z_single（独立第一级） | 否 | 16 | 1.562e-2（= 1 ULP） | 7.84e-5 | 0 |
| Z_ref vs Z_chain（串联内部 Z） | 否 | 16 | 1.562e-2 | 7.84e-5 | 0 |
| **Z_single vs Z_chain** | **是** | 0 | 0 | 0 | 0 |
| Y_ref vs Y_chain（原 FAIL） | 否 | 167 | 2.5e-1 | 4.11e-3 | **2** |
| **Y_chain vs Y_split**（把设备 Z 交给独立第二级） | **是** | 0 | 0 | 0 | 0 |
| **Y_chain vs Y_chainZ**（把串联自己的 Z 交给独立第二级） | **是** | 0 | 0 | 0 | 0 |
| Y_refZ vs Y_ref（第二级单独、喂参考 Z） | 否 | 18 | 6.25e-2 | 2.94e-4 | 0 |
| Y_split vs Y_refZ（只是 Z 的来源不同） | 否 | 150 | 2.5e-1 | 4.10e-3 | 2 |

Z 的 16 个差异元素分散在 16 个不同行（17、22、32、43、44、62、104、108、114、135、136、141、156、169、189、255），每个都恰好差 1 个 FP16 ULP，例如 `Z(17,69)` device=3.324219 / ref=3.326172（ULP 1.95e-3）。

**串联内部 Z 的取法**（未改计划）：用同一个原串联计划、原 X/W1，只把运行期 `constant.W2` 换成单位阵。K=256 单块、FP32 累加乘 1 精确，所以 `Y = FP16(Z @ I) = Z` 逐位成立；计划的 task 几何、依赖、slot、生命周期一律未改。独立性由 C2 佐证：独立 S1 的输出与之**逐位相同**。

### 17.4 第一级生成了什么、第二级读到了什么、误差从哪开始

- **第一级**：设备生成的 Z 与“float64 累加后窄化”的参考差 16 个元素、每个 1 ULP，其余 65520 个元素逐位相同。
- **第二级**：用设备 Z 与用参考 Z 各跑一次，Y 分别有 167 与 18 个元素不同；差异是 150 个 Y 元素、`max_abs=0.25`，2 个超容差。也就是说**第一级的 16 个 1-ULP 差被第二级放大约 9 倍**，放大后落在近零输出上才越过 `atol`。
- **串联本身**：`Z_chain ≡ Z_single`、`Y_chain ≡ Y_split ≡ Y_chainZ`（全部逐位）。串联没有引入任何额外误差，也没有存储/消费/同步/生命周期问题——**把失败归给多节点调度、slot 复用或 stage-2 读错 Z，都被数据否定**。

### 17.5 那 1 ULP 是什么

计划的数值契约是“每 256 宽 K 块升序、每块 FP16 写回”，块内由设备用 FP32 累加；参考用 float64 累加后窄化。两者在**舍入边界**附近必然有极少数元素选边不同：

- 设备 Z 与 float64 参考差 **16** 个元素；
- 设备 Z 与 numpy 的 float32 累加结果差 **85** 个元素；numpy fp32 与 float64 之间差 79 个；
- 设备与 float64 的 16 个差异中只有 4 个与 numpy-fp32 的取值重合。

即设备比“朴素 FP32 顺序累加”更接近 float64（CUDA/CUBE 之类通常是块内多路部分和再合并），但**不等于** float64。这解释了为何差异恰好是 1 ULP、数量在十几个量级（256 维内积落在 FP16 舍入边界一个 FP32 相对误差内的概率量级正是 1e-4）。

### 17.6 结论：不改容差、不关闭 F3

- **没有发现实现缺陷**：串联执行链可逐位复现，第一级与独立第一级逐位相同，第二级与独立第二级逐位相同。
- **原数值检查仍然 FAIL**（2 个元素超容差），不放宽、不改写为通过。误差来源已经定位到：**第一级的 FP32-累加 vs float64-累加的舍入边界选边**，经第二级线性放大。
- 要让这类链式计划的数值验收可判定，需要把“参考模型”与计划声明的数值契约对齐（例如用与设备累加语义一致的参考，或对链式结构单独定义验收口径），**这属于数值政策/验收政策变更，超出本轮授权**，因此不实施、不提方案细节。

### 17.7 身份与重放

- 代码：`main @ de4b07a` + F1/F2/F3 改动（本轮未改任何生产代码；新增脚本都在证据目录）。
- 库：板端现场构建 `librknpu_plan.so` sha256 `c6a6dd1f…b64709`，源包 `30d96f1c…7e59ec2a`；离线用 `../custom-ops/build-host-mvp/librknpu_plan{,_mock}.so`。
- 计划：串联 `dca70de827…`（文件 `ba106000…`）、S1/S2 `bcc9e176…`（文件 `2cf0cce7…`）。
- 输入：X `ccf07a04…`、W1 `a4c50664…`、W2 `bd09d49a…`、单位阵 `identity.json` 记录。
- 证据：`results/20261101T000000Z-f3-stage-isolation/`（`build_campaign.py`、`analyze_stages.py`、`accumulation_probe.py`、`comparison.json`、`identity.json`、`board/` 含 6 案原始输出与 `SHA256SUMS`）。
- 重放：`python3 results/20261101T000000Z-f3-stage-isolation/analyze_stages.py`、`python3 results/20261101T000000Z-f3-stage-isolation/accumulation_probe.py`；板端按 `board/` 下 campaign 重跑。

### 17.8 未验证 / 过程记录

- 仅 core mask 1；未做性能实验。
- 未把第一级那 16 个元素继续细分成“哪几路 K 部分和”的差异；未建设备内部求和顺序模拟器（本轮明确排除）。
- 过程记录：`board-runner` 对证据目录做主机名/用户名脱敏时，把我目录里的 `analyze.py` 参数名 `board` 一并替换成 `<board-user>`，导致脚本语法错误；已重写为 `analyze_stages.py` 并核对 `build_campaign.py`、`identity.json` 未被改动。此事故不影响任何测量数据，仅记录在案。

---

## 18. NumericContract 承诺了什么，验收是否真的在检查它（2026-11-01）

### 18.1 合同逐项

`NumericContract`（`python/rknpu_backend/ir.py:166-227`）的字段与含义：

| 字段 | 取值 | 含义 | 代码依据 |
|---|---|---|---|
| `input_dtype` / `output_dtype` | 只允许 FP16 | 输入输出精度 | `ir.py:178-180` |
| `accumulator_dtype` | `float16` / `float32` | 累加器精度 | `ir.py:181-184` |
| `accumulation` | `strict_unspecified` / `ordered_k_tiles_with_f16_writeback` | 块间累积与写回策略 | `ir.py:185-189` |
| `narrowing_points` | 如 `("k_tile_boundary",)` | 舍入点（按 K 块边界） | `ir.py:196-200` |
| `approximate` | 布尔 | 是否允许近似算子（LUT 等） | `ir.py:174` |
| `tolerance` | `{atol, rtol}` | 只做非负校验 | `ir.py:190-194` |

类注释写明它的定位：**“Requested arithmetic semantics, not a promise made by the hardware.”** 取值来源是编译选项常量，与源码无关：`numeric_mode=rk_f16_tiled` → `(atol=8e-3, rtol=8e-3)`；`strict` → `(1e-3, 1e-3)`（`frontend/tilelang_adapter.py:5546-5556`；同样的常量也硬写在 `ir.py:666/672/731/737/785/791`）。

**合同里没有的东西**（这是后面所有问题的根源）：

- 没有 **作用域**（`scope`）：没说 `tolerance` 是“每个 GEMM 的输出”，还是“整条计算链的最终输出”；
- 没有 **组合规则**：多算子串起来时误差怎么累积、允许多少，没有定义；
- 没有 **块内累积精度**：`accumulator_dtype` 与 `ordered_k_tiles_with_f16_writeback` 只规定了“块边界写回 FP16”，块内（256 宽内积）用 FP32 还是其它精度、按什么顺序加，未规定；
- 没有 **参考模型定义**：验收拿什么做参考（float64 数学？块内 FP32？）未规定。

### 18.2 哪些规定真的被检查

| 环节 | 用到了合同的什么 | 证据 |
|---|---|---|
| frontend | 由 `numeric_mode` **生成**合同；把源舍入声明与证明出的舍入点对照 | `tilelang_adapter.py:165-172, 5546-5556` |
| legalize | 读 `accumulator_dtype` / `accumulation` / `narrowing_points` 决定是否允许 `rk_f16_tiled` 路径；`strict` 直接拒绝 | `legalize.py:123-135, 371-384, 527-528` |
| planner (`tile.py`) | 只用 `source_k_rounding`（数值契约的另一条表达）；`tolerance` 从不读取 | `tile.py:100-110` |
| **RKPlan** | **完全不携带**：计划 JSON 里没有任何 `numeric` / `toler` 字段，action attrs 只有 schedule/k/padding/stride/z_source | `results/20261101T050000Z-numeric-contract/plan_numeric_fields.py` 实测 |
| C 校验器 / runtime | **零引用**：`grep numeric_contract|tolerance|atol|rtol` 在 `custom-ops/include` 与 `src` 无命中 | 实测 |
| 验收工具 | `board_runner.py:252-263` 与 `verify_board_campaign.py:202-238` 从 **case manifest** 读 `atol`/`rtol`，对整个 case 的输出与 `declared` 参考做一次 `allclose` | 实测 |

结论：合同**只影响“能不能选这条路径”**（dtype / accumulation / narrowing），`tolerance` 与 `approximate` 从未参与执行，也从未被编译器写入计划。**RKPlan 丢掉整份合同**——这是本轮对主线的直接结论。

另外发现一处**工具自述错误**（不改归档脚本，仅记录）：F3 campaign 生成脚本把容差写成 `"source": "plan NumericContract; not loosened"`，但计划并不携带合同；实际来源是 frontend 常量 8e-3/8e-3。

### 18.3 参考与验收实际做了什么

验收脚本对每个 case 计算两个参考并**用同一个容差**比较：`declared.*`（FP16，块内 float64 累加后按块窄化）与 `fp64_math.*`（纯 float64，无中间窄化），状态只由 `declared` 的 `allclose` 决定（`board_runner.py:236-270`）。也就是说：

- **参考**：块内 float64 累加 → 这是合同**没有规定**的额外假设；
- **作用域**：一个容差作用于**整个 case 的输出**（两级串联链的最终 Y），而容差常量本来是为“一个 GEMM 的输出”定的；
- 项目里另有一套**端到端**数值制度：模型侧用 train 观测冻结的 per-tensor gate（`ACCEPTANCE.md:218-232`、`numeric_policy.py` 的 2-ULP 底板），与这个 8e-3 常量不是一回事。

### 18.4 单级容差被用在整链上：量化（沿用上轮板测数组）

| 作用域 | 参考 | max_abs | 超容差元素 | 需要多少倍容差才通过 |
|---|---|---|---|---|
| 第一级 Z（单独） | float64 累加 | 1.56e-2 | **0** | 0.10 |
| 第一级 Z（单独） | float32 累加 | 1.56e-2 | **0** | 0.10 |
| 第二级 Y（喂合规 Z） | 同 | 6.25e-2 | **0** | 0.12 |
| 整链 Y | float64 Z + float64 Y | 2.50e-1 | **2** | **2.77** |
| 整链 Y | float64 Z + float32 Y | 5.00e-1 | **2** | 2.77 |
| 整链 Y | float32 Z + float64 Y | 5.00e-1 | **2** | 2.77 |
| 整链 Y | 设备 Z + float64 第二级 | 6.25e-2 | **0** | 0.12 |

三条可直接引用的事实：

1. **每一级都满足合同容差**（余量 8–10 倍），包括用设备 Z 作为第二级输入的组合；
2. **只有“把两级串起来、拿最终 Y 与链参考比”时才超差**，需要约 **2.77 倍**的单级容差；
3. 换参考的块内精度（float64 / float32 / 混合）**不改变判定**（都是 2 个元素超差）——所以这里的失败不是“参考选错了”造成的，而是**作用域没有定义**。

### 18.5 原失败的最终分类

| 候选分类 | 判定 | 依据 |
|---|---|---|
| 实现错误 | **排除** | 串联 ≡ 独立设备执行（逐位）；第一级 ≡ 独立第一级；第二级 ≡ 独立第二级 |
| 硬件错误 | **排除** | 同上，且同输入重复逐位可复现 |
| 参考错误 | **不成立**（对本次判定） | float64/float32/混合参考都给出同样的 2 个超差 |
| **验证器使用错误 / 合同表述不完整** | **成立** | 单级容差被端到端使用；合同没有 scope、没有组合规则、没有块内精度定义 |
| 在既有要求下确实未通过 | **部分成立** | 按“现行判据就是这么跑的”它确实 FAIL（2/65536）；但那条判据对链式计划从未被定义过 |

一句话：**不是算错，也不是参考错，而是拿“单个 GEMM 的容差”去判“两级串联”，而合同从未说过这两件事能不能这么比。** 我不把它改写成 PASS。

### 18.6 对主线的影响

- **RKIR 保住了上层数值意图**：`numeric_contract` 在 op attrs 上，planner 用它决定路径（strict 拒绝、f16 tiled 允许）。
- **RKPlan 把它丢了**：计划不携带 dtype/accumulation/narrowing/tolerance/approximate 中的任何一项。于是一份计划被保存、分发、重放后，**无法从计划本身得知它承诺了什么数值语义**；验收只能靠外部常量或人手抄写的 manifest（本轮就看到一处抄写并错误归因的例子）。
- 执行链本身不检查数值承诺（C 与 runtime 零引用），这是设计选择，不是缺陷；缺陷是**计划层面没有留下这份承诺**，导致“编译时说 A、验收时按 B”这种漂移无法被发现。

### 18.7 建议（不实施，需用户批准）

| 方案 | 内容 | 影响 | 验证设计 |
|---|---|---|---|
| **A. 合同补作用域与组合规则** | 给 `NumericContract` 增 `scope`（`per_primitive` / `per_chain`）与可选 `chain_rule`；在 RKIR schema 与文档写明 | 只动 RKIR/schema/文档；不改执行；现有计划文件不受影响（合同不进计划） | 用本轮链例验证：`per_primitive` 下两级各判、`per_chain` 下按 chain_rule 判；两种口径都要能复现 18.4 的数字 |
| **B. 把合同写进 RKPlan（schema v3）** | 计划顶层增 `numeric_contract`（含 scope），验收从计划读而非手抄 | 计划 schema 升版，**所有 plan_hash 变化**，历史证据需按新 schema 重生成或明确标注旧 schema | 对比新旧计划在同一 DUT 上的 task/submit/publish 与执行结果（应逐位相同），只允许 hash 变 |
| **C. 复合计划改为“分级验收 + 端到端 gate”** | 每级用单级容差单独过；整链另立 gate（像模型侧那样用校准冻结的口径） | 验收流程变化；需要为链式结构定义并冻结新的 gate，不能用失败结果反推 | 用本轮 S1/S2/C3/C5 数据演示：分级全过 + 端到端 gate 的判定；保留原 FAIL 记录 |
| **D. 参考语义钉死** | 在合同或文档里写明“块内累加精度与求和顺序假设”，参考实现据此生成 | 文档 + 参考代码；不改变本轮判定（2 个超差点在任何参考下都超） | 用 float64/float32 两套参考复跑，确认判定不变 |

**推荐**：先做 **A**（定义清楚，代价最小、不动执行、不动历史计划），同时把 **B** 作为紧随其后的一步（因为“计划不携带数值承诺”是主线上真实的信息丢失，也是本轮唯一与 RKIR/RKPlan 保真直接相关的缺陷）；**C** 建议在 A/B 之后作为验收流程升级，且必须保留本次 FAIL 作为“旧口径”的证据；**D** 与 A 合并做，作为 A 的一部分。

**明确不建议**：放宽容差、按失败结果重新校准、把“每级都过”当成整链 PASS、或以“误差很小”替代合同判断。

### 18.8 未验证与身份

- 未验证：设备内部真实求和顺序（本轮不需要，且 18.4 已证明它不影响判定）；链式以外的图形态（FFN/分支）是否也超单级容差未测；未做 per-tensor 校准口径的对照实验（属方案 C 的验证工作）。
- 未修改任何生产代码或数值政策；本轮只新增诊断脚本与报告。
- 证据：`results/20261101T050000Z-numeric-contract/`（`plan_numeric_fields.py` 证明计划不携带合同、`scope_probe.py` 产出 18.4 的表与 `raw/scope.json`）；沿用 `results/20261101T000000Z-f3-stage-isolation/` 的板测数组（设备/库身份 `librknpu_plan.so c6a6dd1f…`、串联计划 `dca70de827…`、S1/S2 `bcc9e176…`、输入 sha256 见其 `identity.json`）。
- 重放：`python3 results/20261101T050000Z-numeric-contract/plan_numeric_fields.py`；`python3 results/20261101T050000Z-numeric-contract/scope_probe.py`。

---

## 19. 把 NumericContract 保存进 RKPlan（2026-11-01）

### 19.1 以前保存计划时丢了什么

编译出来的计划里**没有任何数值信息**：`grep numeric|toler` 在整个 plan JSON 里 0 命中（§18 已实测）。RKIR 的每个 matmul op 上都有 `numeric_contract`（dtype、accumulator、accumulation、narrowing_points、approximate、tolerance），但一旦落地成 RKPlan 就只剩任务几何、布局、K 链和预算。后果是：计划被保存、分发、重放之后，谁也无法从计划本身说出“这个算子当初要求什么数值语义”，验收只能靠外部常量或人手抄写的 manifest（§18 记录过一次抄写并把来源错误归因的例子）。

### 19.2 现在保存了什么

计划顶层新增**可选**字段 `numeric_requirements`，一条对应一个“带合同的逻辑 op”：

```json
{"op_index": 0, "opcode": "matmul",
 "contract": {"input_dtype": "float16", "output_dtype": "float16",
              "accumulator_dtype": "float16",
              "accumulation": "ordered_k_tiles_with_f16_writeback",
              "narrowing_points": ["k_tile_boundary"], "approximate": false,
              "tolerance": {"atol": 0.008, "rtol": 0.008}},
 "inputs": ["A", "B"], "outputs": ["Y"],
 "tasks": [4, 5],
 "execution": {"k_chain": [0, 256], "k_tile": 256,
               "templates": ["co.mm.f16.v1", "co.mm_acc.f16.v1"],
               "output_dtypes": ["float16"], "attributed": true}}
```

两侧含义分得很清楚：

- `contract` 是**上层要求**（原样拷贝，一个字段不改）；
- `execution` 是**实际执行事实**（K 链的 k0 序列、每片 K 宽度、用到的模板、输出 dtype、是否成功关联到任务），由计划里真实存在的 task 反推得到。
- 记录本身**不声称设备满足容差**；`explain` 里写明 `device_satisfaction: not asserted by the plan`。

### 19.3 怎样找到对应计算任务

关联来自 lowering 时写下的 origin，而不是名字猜测：

- `lowering:staged:stage{i}:mm…` → 逻辑 op `i`（两阶段/FFN 分阶段路径）；
- `lowering:matmul:*`、`lowering:matmul_acc:*`、`lowering:linear:*` → 该图唯一的 matmul op；
- 其它 origin（LUT、ew、上传/下载）不参与关联。

关联在**提交调度之后**执行（`codegen/pipeline.py:plan_verified` 里 `build_submits` 之后调用 `attach_numeric_requirements`），因为 `fuse_task_chain` 会把上传前移并重排 action id——早一步记录就会记到错的 id 上（本轮第一次实现正是栽在这里，测试 `test_device_residual_add_is_bitwise_host_equivalent` 抓到）。

实测两例：

- 单 GEMM（256×512×256，声明 128×256）：`tasks=[4,5]`、`k_chain=[0,256]`、模板 `co.mm.f16.v1`+`co.mm_acc.f16.v1`。
- 两级链（256×256×256×256）：两条记录，`op 0` → tasks `(4,5)`、`op 1` → tasks `(6,7)`，各带自己的 `input/output` 名字与 K 链，互不重叠。

### 19.4 校验与篡改负例

计划构造/加载时检查（`plan.py::_validate_plan`，只检查现有信息足以判断的东西）：

| 检查 | 结果 |
|---|---|
| 合同字段合法（复用 `NumericContract.from_dict`：dtype/accumulation 枚举、tolerance 非负有限） | 非法即拒 |
| 一个逻辑 op 只能有一条记录 | 重复即拒 |
| 记录引用的 task 必须存在、必须是 npu_task、不能被两条记录共用 | 违反即拒 |
| `execution.attributed` 必须与 tasks 是否为空一致 | 不一致即拒 |
| `execution.templates` / `output_dtypes` 必须与该记录的任务实际一致 | 不一致即拒 |
| `execution.k_chain` 必须是 0 起的 256 升序切片，且与该记录任务的 k0 集合一致 | 不一致即拒 |
| `approximate=false` 的合同不得由 `co.lut.*` 任务实现 | 违反即拒 |
| 多 K 切片且 `accumulation=ordered_k_tiles_with_f16_writeback` 时必须声明 `k_tile_boundary` 舍入点 | 缺失即拒 |

篡改负例（`tests/test_numeric_requirements.py::TamperTests`）：改 contract 的 tolerance、把 tasks 指向一个 upload、把 op_index 改成不存在的位置——三种都被拒绝（hash 不匹配或结构校验失败）。

### 19.5 旧计划怎么办

| 场景 | 行为 | 实测 |
|---|---|---|
| 读取旧计划（无该字段） | `numeric_requirements == ()`；`explain` 明确报 `stored: false`、`source: not stored` | `models/compiled/tiny_vit_b1_attn_out.rkplan.json`：hash `137e789a58b2` 不变，字段为空 |
| 旧计划读入再保存 | 不写该字段，字节与 hash 不变 | 重存后 `numeric_requirements` 键仍不存在，`plan_hash` 相同 |
| 旧计划重新编译 | 与新编译一致，会带上合同 | 两级链重编译后 `2` 条记录，执行字段（4 task / 4 submit / 2×(128×256)）与旧计划**完全相同**，只有 hash 变（`dca70de827…` → `428d462dfe32`） |

**没有补任何默认值**：旧计划不会被塞进 8e-3，也不会假装有合同。

### 19.6 哪些数值要求依然没有定义

保存合同不等于补齐语义，以下仍然未定义（本轮不自行补）：

- **块内求和顺序与精度**：`accumulation=ordered_k_tiles_with_f16_writeback` 只约束块边界写回，256 宽内积内部怎么加仍未写；
- **整链/整图容差**：合同是 per-op 的，计划里没有、也不臆造端到端判据（`explain` 写 `end_to_end_criterion: undefined`）；
- **容差的适用对象**：合同没说 tolerance 是“该 op 的输出”还是别的口径；本轮只保存原文，不做解释性扩写。

### 19.7 本轮没有改变的判定

- 容差、数值政策、生产默认、既有冻结 spec 全部未动；
- 两级链对旧 reference 仍然 **FAIL**（167 个元素不同、2 个超容差），本轮目标是保存与追踪，不是把 FAIL 变 PASS；
- 验收工具仍从 case manifest 取整链容差（旧命令行为保留），只是现在**计划自己能说清每个 op 要求什么、由哪些 task 实现**，两者可以对照。

### 19.8 已知缺口：融合 FFN/SwiGLU 路径的任务关联

`build_swiglu` / `build_scheduled_swiglu_ffn` 生成的 task 用 `lowering:swiglu:*`、`lowering:ffn:*` 作为 origin，其中不含逻辑 op 编号，所以本轮在**这条路径上只能记录合同（op_index、inputs、outputs、contract 全对），无法把 task 挂上去**，记录里如实写 `attributed: false`、`tasks: []`，不猜。

实测（`tilelang_ffn.swiglu_ffn_f16(64,256,128)`，allow_approx_lut）：6 条记录（matmul/silu/mul/matmul/add）全部 `attributed: false`。

验收上这样处理：单 GEMM、linear 策略、受限两阶段（本轮板测的两条路径）**关联完整**；融合路径**要求已保存、关联缺失且有明确标记**。要把融合路径补齐，需要在 build_swiglu/FFN 里给 task 打逻辑 op 编号（下一轮的小改动），本轮不做以免扩大改动面。

### 19.9 证据与重放

- 代码：`plan.py`（`NumericRequirement` 记录 + 可选字段 + 校验）、`passes/dag.py`（origin→op 关联与记录生成）、`passes/storage.py`（传入计划）、`codegen/pipeline.py`（在提交调度后挂接 + explain 的 `numeric` 段）、`schema/rkplan.v2.json`（新增可选定义，未升版）。
- 证据：`results/20261101T070000Z-plan-numeric-requirements/`（`make_plans.py`、`identity.json`、`plans/*.rkplan.json`、板测 `board/`）。
- 重放：`python3 -m pytest tests/test_numeric_requirements.py -o addopts="" -q`。

### 19.10 板端验证（RK3588，实测）

| 案例 | 计划 sha256 / plan_hash | task / submit / publish | 设备结果 |
|---|---|---|---|
| 新单 GEMM（带 1 条合同） | `3b0fc1e0…` / `5a518499…` | 2 / 2 / 2 | 输出 sha256 `d2a7df7e…`，与上一轮旧单级设备输出**逐位相同**（n_diff=0） |
| 新两级链（带 2 条合同） | `96c28239…` / `428d462d…` | 4 / 4 / 2 | 输出 sha256 `46a5be41…`，三轮 r0=r1=r2 |
| 旧两级链（无合同） | `ba106000…` / `dca70de827…` | 4 / 4 / 2 | 同上，**与新链逐位相同** |

- 设备自报 task/submit 与计划一致；三案三轮全 finite；无 error、无新增 kernel/dmesg 行；core mask 1；未做性能实验。
- 本机结构比对：新旧计划的 bindings/constants/views/buffers/actions/submits/publish/accounting/budget_policy/source_hash 逐项相同，**差别只有 `numeric_requirements`**，与设备输出逐位一致互相印证。
- **原 FAIL 未变**：两级链对旧 reference 仍 167 个元素不同、**2 个超容差**、`max_abs=0.25`、allclose 假；旧链同样如此，与本轮改动无关。

**一个必须记录的部署事实**：用上一轮的 Python 包加载新计划会被 fail-closed 拒绝（`RK_E_SCHEMA: Plan unknown fields: numeric_requirements`），换成当前工作区源码包后正常；重建出的 `librknpu_plan.so` 与旧库 **sha256 完全相同**（`c6a6dd1f…b64709`），说明这道 schema 闸门在 Python 侧、C 库未变。方向性结论：**旧计划 → 新工具：可读（本节 19.5）；新计划 → 旧工具：明确拒绝而不是误解**，符合“不伪造未保存过的要求”的原则。

**证据**：`results/20261101T070000Z-plan-numeric-requirements/board/`（三案原始 npz、计划、`SHA256SUMS` 56 条、`REPORT.md`）。

---

## 20. 收尾轮：容差有限性已修，任务关联未落地（2026-11-01）

### 20.1 修好的：NumericContract 的容差必须是有限非负数

`ir.py::NumericContract.__post_init__` 现在用 `math.isfinite` 检查每一个容差值：`NaN`、`+inf`、`-inf`、负数、`bool`、字符串全部以项目诊断 `RK_E_NUMERIC_CONTRACT (-4)` 拒绝，而不是让 `NaN` 一路走到比较里、也不是抛 `TypeError`/`NameError`。`from_dict` 走同一构造器，因此 RKIR 入口、`NumericRequirement` 构造、计划加载都会被同一条检查覆盖。

覆盖测试：`tests/test_numeric_requirements.py::ToleranceFinitenessTests`（有限值接受；`nan/inf/-inf/-1.0/True/"0.1"` 六种输入逐一拒绝；`from_dict` 同样拒绝）。这条修复**不改变任何被接受计划的字节**——它只是把非法输入更早挡住。

### 20.2 没有落地的：基于结构化信息的任务关联（本轮尝试后回退）

审核要求把 origin 字符串解析换掉。本轮做了实现并**回退**，原因如实记录：

| 尝试 | 结果 | 处置 |
|---|---|---|
| 在 `add_action` 给每个 npu_task 打 `source_op`（逻辑 op 编号），关联改读该字段 | 任务 attrs 变化 → 大批既有测试与计划身份受影响；实测全量 553 → **91 failed** | 回退 |
| 再加“模板族必须匹配 opcode”的守卫（防止残留指针把 ew 任务算到 matmul 上） | 实测 **105–186 failed**（融合图的合法任务被误判） | 回退 |
| 计划侧新增“task 的 `source_op` 必须等于记录 `op_index`”“opcode 与模板族一致”两条结构检查 | 与上面同源，无法在保持既有计划与测试的前提下判定 | 回退（仅保留 `inputs ∩ outputs` 非空这条独立检查） |

**因此关联机制与上一轮相同**：按 lowering 的 origin 解析（`lowering:staged:stage{i}…`→op i；`lowering:matmul*`/`lowering:linear*`→唯一 matmul op），FFN/SwiGLU 多算子路径中 origin 不含 op 编号的记录**仍为 `attributed: false`、`tasks: []`**（诚实标记，不猜）。融合路径的完整关联仍是**未完成项**，需要按下面的方案做：在 `build_swiglu`/`build_scheduled_swiglu_ffn` 里以“当前 op”为参数调用发射函数，把 op 编号写进 action attrs，并同步更新受影响的计划身份断言与测试基线（这一步是计划身份变更，必须单独一轮完成，不能在收尾轮顺手做）。

### 20.3 更正上一轮一处表述

§19.4 写“把 `op_index` 改成不存在的位置 → 被拒绝”。审核指出并复现：那次拒绝来自**旧的 plan_hash 校验**，不是结构检查。本轮复核确认：**重算 hash 之后，单独修改 `op_index`、`inputs/outputs` 或 `opcode` 仍然会被接受**——因为计划里没有 RKIR，无法把记录与逻辑图对照。仍然能被结构检查抓住的只有：

- 记录引用不存在的 action id、或引用的 action 不是 `npu_task`；
- 同一条记录被两条 requirements 共用；
- `execution.templates`/`output_dtypes`/`k_chain` 与它自己的任务不一致；
- `attributed` 与 tasks 是否为空不一致；
- `approximate=false` 却由 `co.lut.*` 实现；多 K 切片未声明 `k_tile_boundary`；
- `inputs` 与 `outputs` 出现同一个名字。

这正是本轮未完成的结构化关联要解决的问题：只有把“任务属于哪个逻辑 op”写进计划本身，`op_index`/`inputs`/`outputs`/`opcode` 才能被交叉校验。现状要如实读成：**文件完整性（hash）与“任务↔记录”的部分结构一致性有检查；“记录↔逻辑图”的一致性没有检查**。

### 20.4 执行与板测：不需要重跑，理由逐项核对

本轮唯一改动是容差有限性校验，它只拒绝非法输入，不参与任何被接受计划的生成：

- 计划字节不变：容差检查发生在 RKIR 构造期；合法合同（8e-3/8e-3、1e-3/1e-3 等）不受影响。
- 上一轮板测（`results/20261101T070000Z-plan-numeric-requirements/board/`）已证明：带 `numeric_requirements` 的新计划与旧计划在设备上 task/submit/publish 一致、**输出逐位相同**（`46a5be41…` / `d2a7df7e…`）。本轮没有改变任何计划内容，所以这些证据继续有效，不重复上板。
- 原两级 GEMM 的数值 FAIL 未变（167 个元素不同、2 个超容差），本轮不涉及、也不改写。

### 20.5 回归

- 全量 Python 单测：**556 passed / 1 skipped / 135 subtests**（新增 3 例容差有限性测试；未改任何既有断言）。
- F1/F2/F3 探针与冻结身份未受影响（本轮未触碰调度、slot、来源标记或数值政策）。

### 20.6 本轮没有改变的东西

- 容差数值、数值政策、生产默认、整链验收判据；
- 计算、舍入、布局、依赖、submit 分组；
- 原两级 GEMM 的 FAIL 记录与历史证据。

### 20.7 本轮最终状态（收尾核验）

**已交付并验证**：容差有限性检查（§20.1）。

**未交付**：结构化任务关联与“记录↔逻辑算子”一致性检查（本轮主要要求）。尝试过程中出现的失败，我按原因如实分类如下，未做任何批量改预期值：

| 类别 | 观察 | 结论 |
|---|---|---|
| 1. 真正的执行/依赖/布局/数值错误 | 未观察到：所有失败都在构造或断言层面，没有出现任务几何、依赖、布局、输出数值的差异 | 无 |
| 2. 来源/关联记录错误 | 未观察到错误关联（`source_op` 一旦打上，指向的都是正在 lowering 的那个 op）；但 elementwise 任务会继承上一个 matmul 的指针 → 属于**实现缺陷**，正是“模板族守卫”想解决的东西 | 未修完 |
| 3. 合法融合被过窄的模板规则误拒 | 观察到了：加“模板族必须匹配 opcode”守卫后出现 105–186 个失败，包含融合图的合法任务 | 守卫方案作废 |
| 4. 新计划增加记录导致 hash/快照断言变化 | 观察到了：在 `npu_task.attrs` 写入 `source_op` 后，全量 553 → 91 failed，绝大多数是计划身份类断言 | 未处理（需逐例证明执行未变后再改断言） |
| 5. 测试本身错误假定新增记录不能改变身份 | 少量；与类别 4 混在一起，未及分离 | 未处理 |

**过程事故（如实记录）**：在用脚本修复类别 4 的失败时，我用字符串替换方式补丁 `passes/dag.py`，引入了字面 `\n` 与缩进破坏；修复过程中文件进一步损坏。最后用 `git checkout -- python/rknpu_backend/passes/dag.py` 恢复到提交基线，并重新落地了此前各轮对 `dag.py` 的改动（F1 的 linear builder 声明/预算逻辑、F3 的闸门文案与 `attach_numeric_requirements`、§19 的记录生成），全量测试恢复 **556 passed / 1 skipped / 135 subtests**。教训：不该用大段字符串替换去改这个文件。

**因此本轮状态应读成**：容差有限性已修；结构化关联**未完成**——关联仍来自 origin 解析，FFN/SwiGLU 的 elementwise 记录仍为 `attributed: false`；“单独改 `op_index`/`inputs`/`opcode` 并重算 hash 后仍会被接受”的缺口**仍然存在**（§20.3 已如实说明）。原两级 GEMM 的数值 FAIL、容差与数值政策均未改变。

> **本节已被 §21 取代（2026-11-01，同日下一轮）**：origin 解析已删除，关联改为发射时登记；FFN/SwiGLU elementwise 记录已完整归属；重算 hash 后改 `op_index`/`opcode`/`inputs`/`outputs`/`tasks` 会被计划内部关系检查拒绝。§20.7 末尾“过程事故”的教训保留（本轮全程用小改+逐步语法/测试检查，未做大段替换）。

---

## 21. 结构化任务关联与逻辑算子表（2026-11-01，本轮，完全落地 §19.8 / §20.2 的三项）

### 21.1 一句话结论

关联不再从 origin 字符串里猜：每个设备任务在**发射的那一刻**写明它落实哪些逻辑算子，规划结束后写成计划顶层的一张表（`logical_ops`）。数值记录（`numeric_requirements`）现在要对着这张表逐项核对，所以“改错 op 编号 / opcode / 输入输出 / 任务归属后再重算 hash”会被**关系检查**拒绝，而不是被旧 hash 拒绝。FFN/SwiGLU 的 elementwise 记录不再是 `attributed: false`。

### 21.2 这三件事分别是什么状态

| 要求（§三） | 状态 | 证据 |
|---|---|---|
| 任务关联不再靠解析 origin 字符串 | 完成（解析函数已删除） | `passes/dag.py` 无 `_task_owner`；`attach_plan_records` 只读 `Action.source_ops` |
| SwiGLU/FFN elementwise 记录不再 `tasks: []` | 完成（6-op 与 8-op 两条路径都完整） | §21.9 A/B 表：`swiglu-ffn` 6/6、`scheduled-ffn` 6 条 elementwise 全部有任务 |
| 改 op_index/opcode/inputs/outputs 重算 hash 后能被拒 | 完成（新增关系检查 + 8 条负例） | `tests/test_logical_attribution.py::CrossCheckTests` |

### 21.3 关联怎么建立（不再猜，也不靠隐式状态）

```
lowering 发射任务时：  add_action("npu_task", ..., op_ids=(op.id,))
                          └─ Action.source_ops = (op.id,)   # 只是规划期标签
提交调度之后：            attach_plan_records(module, draft, options)
                          └─ 按 source_ops 汇总  op → 任务 id（此时 id 已是最终值）
                          └─ 写 Plan.logical_ops 与 numeric_requirements
```

四个要点：

1. **没有隐式“当前算子”状态**：`op_ids` 是每个 `add_action` 调用的显式参数。28 处 npu_task 发射点逐点传入“正在 lower 的那个 op”；四个共享发射器（`_emit_activation`、`_emit_binary`、`_emit_binary_views`）和两个闭包（`build_scheduled_two_stage::emit_chain`、`build_scheduled_swiglu_ffn::matmul_chain`）把 op 作为参数往下传，由调用点决定。
2. **不受重编号影响**：唯一会重排 action id 的地方是 `_hoist_host_uploads`（`fuse_task_chain` 时把上传前移到最前）。它用 `dataclasses.replace` 重建 action，字段随之复制，`source_ops` 跟着走；`attach_plan_records` 在 `build_submits` **之后**运行，写进计划的是最终 id。
3. **`Action.source_ops` 不进计划字节**：它 `compare=False`（不参与 dataclass 相等）、`to_dict` 不输出、`from_dict` 用默认值。所以 actions/submits/views/bindings 等执行字段与改动前**逐字节相同**（§21.9 实测）。计划要说的那句话只说一次，说在 `logical_ops` 里。
4. **origin 只做诊断**：字符串内容没有任何改动，只是不再参与归属判断。

### 21.4 计划里新增了什么

```json
"logical_ops": [
 {"index": 2, "opcode": "silu", "inputs": ["Zg"], "outputs": ["S"],
  "contract": {...},            // 该 op 的 NumericContract，没有就是 null
  "tasks": [8, 9],              // 落实它的设备任务（一个 op 可以有多个模板）
  "auxiliary": false}           // true = 用视图/布局实现，没有设备任务（如 packed FFN 的 view）
]
```

- 字段是**可选**的，`schema_version` 仍是 2；`schema/rkplan.v2.json` 里补了定义（它的 `additionalProperties` 是 false，不补就会被 schema 校验拒绝）。
- 旧计划（无该字段）读入、重存、重放**字节不变**：实测 `models/compiled/tiny_vit_b1_attn_out.rkplan.json` 键集合相同、`plan_hash` 相同（§21.11）。
- `numeric_requirements[i].execution` 新增 `approximated`（见 §21.7）。
- `explain` 新增 `logical` 段（`stored/source/ops/without_tasks`）。

### 21.5 怎样发现错配（这才是本轮的主要产出）

计划构造/加载时，只要带 `logical_ops` 就启用以下检查（`plan.py::_validate_plan`）：

| 检查 | 违反时 |
|---|---|
| 索引唯一且升序 | 拒绝 |
| 表里引用的任务必须存在、必须是 npu_task | 拒绝 |
| `auxiliary: true` 的 op 不得拥有任务 | 拒绝 |
| 表必须覆盖**全部** npu_task（不能有任务不属于任何 op） | 拒绝 |
| requirement 的 `op_index` 必须能在表里找到 | 拒绝 |
| requirement 与它点名的 op 在 `opcode`/`inputs`/`outputs`/`contract`/`tasks` 上逐项相等 | 拒绝 |
| 有合同的 op 必须有且只有一条 requirement；无合同的 op 不得有 requirement | 拒绝 |
| 有 requirements 却没有 `logical_ops`（只可能是被删掉的表） | 拒绝 |

负例全部**先重算 `plan_hash`**，所以拒绝只能来自上表（`CrossCheckTests`）：

| 篡改 | 拒绝理由（实测） |
|---|---|
| `op_index` 改成另一个真实存在的 op | `numeric requirement ... disagrees with the logical op it names` |
| 改 `opcode` | 同上 |
| 改 `inputs` | 同上 |
| 把任务挪到另一个 op（表与记录一起改） | `disagrees`（K 链/模板/归属随之不一致） |
| 表项少一个任务 | `disagrees` |
| 让表项指向一个 upload | `logical op references a non-NPU action` |
| 让某个设备任务不属于任何 op | `device tasks are not attributed to any logical op` |
| 删掉整张表但保留 requirements | `numeric requirements need the logical op table that makes them checkable` |
| **只在 hash 上“改回原值”**（内容不变） | 接受（`test_hash_rewrite_alone_is_accepted`） |
| 表和记录**一致地**改成另一个合法的 value 名 | 接受（`test_consistent_rewrite_is_accepted`，且 actions/submits 未变） |

这条边界要读清楚（与 §六 的要求一致）：

1. `plan_hash` 只说明**文件内容自洽**；
2. 本轮新增的是**计划内部关系**的一致性（记录 ↔ 逻辑表 ↔ 任务）；
3. **都没有**证明硬件满足容差。计划不会自称满足了误差，`explain` 仍写 `device_satisfaction: not asserted by the plan`、`end_to_end_criterion: undefined`。

也不声称能防“来源伪造”：一个从头到尾自洽的其他计划仍然可以被接受（这是设计选择，不是缺口）。

### 21.6 各 lowering 的覆盖（逐路径）

`op` 指逻辑算子；“任务”指该 op 名下实际登记的 npu_task 数（A/B 实测，§21.9 的同一批计划）。

| 路径 | 有合同的 op | 落实任务 | 无任务 / 辅助 |
|---|---|---|---|
| 普通单 GEMM（`build`，K 链） | matmul | `co.mm` × 1 + `co.mm_acc` × 1（视 K 切片数） | 无 |
| 分块 dataflow（`build_dataflow_tiled`） | 每个 matmul | 每 tile 一条 K 链 | 无 |
| 通用 dataflow（`build_dataflow`） | 每个 matmul | 一条 `co.mm` 覆盖整条 K | 无 |
| scheduled-linear（`build_scheduled_linear`） | matmul | 每 tile 一条 K 链（实例 36 条） | 无 |
| 受限两阶段（`build_scheduled_two_stage`） | 2 个 matmul | stage0 2 条、stage1 2 条（实例） | 无 |
| `build_swiglu`（6-op SwiGLU） | matmul×3、silu、mul、add（6/6） | gate/up/proj 各 1；**silu 2 条**（N×`co.lut.sigmoid` + 1×`co.ew_mul`）；mul 1；add 1 | 无 |
| `build_scheduled_swiglu_ffn`（8-op packed FFN） | 仅 2 个 matmul（前端不给 elementwise 合同） | matmul 36/18（实例）；sigmoid 12、mul 6、mul 6、add 3 | op1/op2 是 `view`：`auxiliary: true`、`tasks: []` |
| 融合 add+relu（dataflow/tiled） | add、relu 各自 | **同一条** `co.ew_add_relu` 任务 | 无 |

auxiliary 的语义是**声明的**（`AUXILIARY_OPCODES = {"view"}`），不是“找不到任务就当辅助”。identity / zero / 上传 / 下载本来就不是 npu_task，从来没有被算进任何 op 的任务里。

### 21.7 补齐关联后暴露的一个真实矛盾，以及处理方式

`_validate_plan` 里原有这条检查：“`approximate: false` 的合同不能由 `co.lut.*` 任务实现”。在关联为空的时候它**空转**（LUT 任务从未被登记到任何 op 上）；关联一补齐，它立刻在 SwiGLU/FFN 的 `silu`、以及 dataflow 的 `silu` 上触发，而这些路径**本来就已被接受并跑过**。

事实是两边都对：合同说的是“调用方要求的算术”，而 LUT 是否可用是**编译期准入**（`allow_approx_lut`，`legalize.py` 会先拒绝没有准入的 LUT）。合同里没有地方能表达后者。

处理（**没有改任何合同、没有放宽容差**）：规划时把“这条 op 的实现用了经准入的 LUT 近似”记进 `execution.approximated`，并把检查改成双向可判定：

- LUT 任务 + 非近似合同 ⇒ 必须有 `approximated: true`；
- `approximated: true` ⇒ 必须有 LUT 任务。

所以它仍是一条真实检查（两件事必须同时成立），而不是“加个字段就过”。

### 21.8 测试：新增什么、改了哪些既有断言、为什么

**新增** `tests/test_logical_attribution.py`（21 例）：覆盖单 GEMM、两阶段、融合 add+relu、无合同 elementwise、explain、SwiGLU 6-op、8-op packed FFN（含 `view` 辅助 op）、上传前移+链融合后的归属、保存/加载、旧计划重存、以及 11 条重算 hash 后的篡改负例/正例。

**修改**既有断言 3 处（逐条理由，没有改任何容差或数值判定）：

| 位置 | 原来 | 现在 | 为什么 |
|---|---|---|---|
| `test_dataflow.py::test_commutative_operand_order_is_equivalent` | 断言 `A*B` 与 `B*A` 两张图 `plan_hash` 相等 | 断言**执行字段全部相等**，且 `logical_ops` 只在唯一一项上不同（输入顺序互换），并断言 hash 不同 | 计划现在会如实复述源图的操作数顺序；“交换律不改变执行”仍被断言，而且更精确 |
| `test_numeric_requirements.py::LutGraphTests::test_lut_graph_keeps_the_matmul_contract_only` | LUT 任务必须 `contract.approximate == True` | 必须 `contract.approximate or execution.approximated` | 合同不承载编译期准入（§21.7）；规则本身保留 |
| `tests/fixtures/guard_after_download_worker.py` | 手工重排 action id 后直接重建计划 | 同时把 `logical_ops`/`numeric_requirements` 的 task id 按同一置换搬移 | 该 fixture 构造的是“重编号后的计划”；计划不允许留下过期归属（这正是新检查要抓的东西） |

**失败是怎么走过来的（逐类，不是整体回退）**：第一轮 36 failed → 定位出两类（① 我的两阶段 `emit_chain` 调用点用了不存在的局部名 `ops`，5 例；② §21.7 的 LUT/合同矛盾，25 例；③ 其余 6 例为真问题：见下）→ 修 ① 与 ② 后 6 failed（4 个真实原因 + 2 个 schema/夹具）→ 逐个修完后 1 failed。最后这 1 个与本轮无关：`test_replay_smollm2_v5` 需要未安装的 `transformers`（基线同样失败）。

全量：**基线 1 failed / 555 passed / 1 skipped**（同一环境失败）→ 本轮 **1 failed / 576 passed / 1 skipped / 135 subtests**（+21 = 新增测试文件）。

### 21.9 新旧执行内容逐项对照（离线 A/B，实测）

做法：把改动前的 5 个文件（`dag.py`/`plan.py`/`storage.py`/`codegen/pipeline.py`/`ir.py`，取自本轮快照）放回一份**独立包副本**，同一份输入分别用旧、新代码编译，逐键比较（`results/20261101T120000Z-structured-op-attribution/ab/`）。

| 案例 | 执行字段（bindings/constants/buffers/views/actions/submits/publish/accounting/budget_policy） | 记录 | 逻辑表 |
|---|---|---|---|
| single-gemm（256×512×256，声明 128×256） | **逐字节相同** | 1 条，改动前 0 条有归属 → 现在 1 条 | 1 个 op |
| two-stage（2 级） | **逐字节相同** | 2 条，2 条都有归属 | 2 个 op |
| linear-policy（M=1,K=576,N=3072） | **逐字节相同** | 1 条 | 1 个 op（36 任务） |
| dataflow-projection（matmul→silu→matmul→add） | **逐字节相同** | 2 条，改动前 0 归属 → 现在 2 归属 | 4 个 op，全部有任务 |
| swiglu-ffn（6-op） | **逐字节相同** | 6 条，改动前 **0** 归属 → 现在 **6/6** | 6 个 op |
| scheduled-ffn（8-op packed） | **逐字节相同** | 2 条（只有 matmul 有合同），2 归属 | 8 个 op：6 有任务，2 个 `view` auxiliary |

`plan_hash` 六个案例都变了——这是**允许的身份变化**：计划多了它自己要说的话（§三授权）。

一处需要如实说明的差异：两个 TileLang 案例的 `source_hash` 文本不同（旧树 `tilelang_ffn.py`，新树 `tests/fixtures/tilelang_ffn.py`），内嵌内容 sha256 相同（`7c6824a0…`）。已证明这是 A/B 复现环境的路径渲染差异、不是改动引起的：把**同一份新代码**复制到别的目录再编译，渲染同样变成 `tilelang_ffn.py`；并且生成该字段的 `frontend/tilelang_adapter.py` 与快照**逐字节相同**（sha256 `1a4ae031…`，本轮未改）。

**C 侧接受性（本地实测）**：用真实（非 mock）`custom-ops/build-host-mvp/librknpu_plan.so` 加载带 `logical_ops` 的新计划——single-gemm、two-stage、dataflow-projection、swiglu-ffn 四个全部 `create_plan` 成功。C 源码本轮未改，因此设备侧同一个库的行为与之相同。

**板测：逐项复用，不重复上板**（§三允许的两种方式之一，理由可核对）：

| 路径 | 已有设备证据 | 为什么继续有效 |
|---|---|---|
| 单 GEMM / 两级链 | `results/20261101T070000Z-plan-numeric-requirements/board/`（RK3588 实测，task/submit/publish 与计划一致、输出逐位相同） | 本轮的这两个计划与那次上板的计划在执行字段上**逐字节相同**（上表） |
| FFN / staged FFN | `results/20261023T144054Z-rk3588/board/`（`ffn_chain`、`ffn_separate`、`ffn_tilelang_staged_chain` 的设备残差记录） | 同上：执行字段逐字节相同，新增的只是说明性字段 |
| linear 策略 | mock/host 证据（`tests/test_linear_schedule_contract.py`、`results/*linear*`） | 同上 |

即：本轮**没有改变任何会被设备执行的东西**（任务、寄存器参数、视图、依赖、submit 分组、publish、预算全部相同），因此不存在“执行变了需要重新验证”的情形；如果将来这些字段变化，本节不能沿用。

### 21.10 本轮没有改变的东西

- 容差数值、数值政策、整链验收判据、生产默认；
- 计算、舍入、布局、依赖、submit 分组、发布、预算；
- **原两级 GEMM 对旧 reference 仍然 FAIL**（167 个元素不同、2 个超容差），本轮不涉及、也没有“修好”；
- `graph-fusion`/`int16-probe` 等无关工作区改动原样保留。

### 21.11 兼容与身份

| 场景 | 行为 | 实测 |
|---|---|---|
| 旧计划（无 `logical_ops`/`numeric_requirements`） | 读入、重存、重放**字节不变**，`explain` 报 `not stored` | `tiny_vit_b1_attn_out`：键集合相同、`plan_hash` 相同 |
| 新计划 | 带表；`plan_hash` 与改动前不同（新增说明信息） | §21.9 |
| 依赖旧 Python 包读新计划 | 明确拒绝（unknown field），不误解 | 与 §19.10 同类 |
| C 库 | 未改；真实库接受新计划 | §21.9 本地实测 |
| schema | 仍是 v2，`logical_ops` 为可选定义；`execution` 增加可选 `approximated` | `test_plan_verify`、`test_frontend_broadcast` 的 schema 校验通过 |

代码身份（本轮改动的文件与快照哈希）见 `results/20261101T120000Z-structured-op-attribution/final-hashes.txt` 与 `snapshot/original-hashes.txt`。

### 21.12 剩余事项（明确没做）

1. **重复展开后的计划不带记录**：`repeat_expand` 会把计划重建成一份新计划，`numeric_requirements`/`logical_ops` 都不搬（改动前 `numeric_requirements` 就已经是这样，不是本轮回归）。这类计划读起来是 `not stored`，不会假装有归属。**（已在 §22 修好：每份副本现在有自己的 op、自己的要求、自己的任务，并注明它来自哪个计划的哪个 op。）**
2. **Hybrid / 模型级重写** 同理，仅保留被重写前的那份计划的记录。
3. **contract 的“近似”语义未重写**：本轮选择记录编译期准入，而不是改写调用方合同（§21.7）。如果以后要让合同自己承载“可近似”，那是数值政策层面的决定，需要单独批准。
4. **板测未加重跑**，理由见 §21.9；若要求“每个路径都必须有新一次板测”，那是新的工作量，不是本轮缺口。

### 21.13 重放命令

```bash
cd research/tilelang-rknpu
# 全量回归（本机 .venv + 本地依赖目录；见 snapshot/ 说明）
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
# 本轮新增的结构化关联与篡改负例
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_logical_attribution.py -q -o addopts=""
# 新旧执行内容对照（旧树由 snapshot/original 的 5 个文件覆盖出独立包）
PYTHONPATH=$PWD/python .venv/bin/python results/20261101T120000Z-structured-op-attribution/ab/dump_plans.py <out-dir>
```

复现旧树的方法（本轮实测用过）：复制 `python/` 到独立目录，用 `snapshot/original/python/...` 的 5 个文件覆盖（`dag.py`、`plan.py`、`passes/storage.py`、`codegen/pipeline.py`、`ir.py`），再在包根旁放 `capabilities/`、`schema/` 两个链接。

证据目录：`results/20261101T120000Z-structured-op-attribution/`（`snapshot/`、`ab/new|old/`、`ab/compare.json`、`logs/`、`final-hashes.txt`）。

---

## 22. 计划转换不再丢记录：重复展开与输入改接（2026-11-01，本轮）

### 22.1 一句话结论

一个计划被复制成 N 份之后，**每份都还知道自己是谁**：它有自己编号的逻辑算子、自己那份数值要求、自己那条设备任务，并注明“我是从哪个计划的第几个算子复制来的”。输入改接（`remap_group_inputs`）不删记录、不改计算，只把每份实际读的窗口写进计划；说明和计算对不上时，计划在加载阶段就会被拒。设备上没有变：同一批转换前后**逐字节相同的执行**，板上输出也逐字节相同。

### 22.2 到底哪些入口会重建计划（清点，带证据）

| 入口 | 实际改变什么 | 记录 | 判定 |
|---|---|---|---|
| `passes/storage.py:131` | 正常编译时构造 Plan，两字段显式传入 | 保留 | 无需改（正确范本） |
| `passes/repeat_expand.py::expand_static_repeat` | 重建整个 Plan：绑定/常量/缓存/视图/动作/提交/发布/预算 | **修复前丢失**（回落默认空） | **已修** |
| `passes/repeat_expand.py::remap_group_inputs` | 逐个 upload 改 `src_*`/transpose，改输入 binding 的 shape，重算 accounting | 由 `replace` 保留 | 无需改（补了检查与 explain，见 §22.6） |
| `tools/compile_models.py:277/287/302` | 调用上面两个并把计划落盘 | 随上面 | **已修**，另外把 `logical_ops`/`numeric_requirements`/`attributed_requirements` 写进 manifest |
| `tools/benchmark_repeat_paths.py:179` | 内存里调用展开做基准 | 随上面 | 无需改（不落盘） |
| `tools/refactor_plan_projection.py:184` | `replace(plan, source_hash=…)` 后重算 hash | 保留 | 无需改 |
| `hybrid.py:137/1261/2688/2794` | `PlanInfo.plan` 是真 Plan（`load_plan` 读入）；`NpuNode.plan`、`group_plan`、`_NpuBindingPlan.plan` 都是 **str（计划名）** | 不重建 | 无需改（证据见下） |
| `artifact.py` / `models.py` / `smollm.py` / `passes/verify.py` | 持有引用、只读、包装 | 不涉及 | 无需改 |
| `passes/residency.py` / `passes/graph_execution.py` | `ResidencyPlan`/`GraphExecutionPlan` 里没有 Plan 字段 | 不适用 | 无需改 |

hybrid 不重建 RKPlan 的证据：`_NpuBindingPlan(plan=node.group_plan, …)`（`hybrid.py:2688-2692`）与 `plan=node.plan`（`:2794`），而两个字段声明为 `str`（`:192`、`:205`、`:263`）；文件里没有任何 `Plan(` 构造，真正的 Plan 只出现在 `PlanInfo.plan`（`:137`），由 `load_plan()`（`:1261`）从磁盘读。**它不丢字段，但它消费磁盘计划**，所以源头丢失会传导过去——这正是本轮要修源头的原因。

### 22.3 复制规则（复制后的每份计算对应谁）

```
父计划:  1 个逻辑 op（op[0]，1 条任务） + 1 条 numeric_requirement
展开 N:  op[i]: index=i, opcode=父opcode, inputs=(Q_i, K_i), outputs=(S_i,),
                 tasks=(first_task+i,), contract=父合同（原样）, derived_from={父plan_hash, 0}
          requirement[i]: op_index=i, 同一个 contract, tasks=(first_task+i,),
                 execution 由第 i 条**新任务自己**的 attrs 重新算出
```

- **编号**：副本从 0 连续编号；任务 id 是展开后真实的重编号结果（`first_task + i`，`shared_bo` 与否分别对应 `2N+i`）。父计划的那条任务 id 不会残留。
- **输入输出引用**：用展开后新绑定的真实名字（`Q_i`/`K_i`/`S_i`），不是父计划的名字。
- **数值要求**：合同逐字段原样复制（不放宽、不新增默认值）；`execution` 不照抄父计划的，而是从新任务的 `k0/kt/template` 与新输出视图的 dtype 重算，`attributed=true`、`approximated=false`。
- **溯源**：每份的 `derived_from = {"plan_hash": 父计划 hash, "op_index": 父计划里的算子编号}`。父计划没有 hash 时**拒绝转换**（否则这份记录无从追踪）。
- **不能直接复制整张表**：父表只有 1 条，复制 N 次而不改编号/任务/名字会与“每份一个 op 一条任务”的事实冲突，也会被 §22.5 的检查直接拒掉。

### 22.4 旧计划（无记录）与坏记录

| 输入 | 行为 |
|---|---|
| 父计划没有 `logical_ops`/`numeric_requirements`（例如归档的 `tiny_vit_b8_qk`） | 转换照旧；产出的组计划**也没有**记录，`explain` 报 `not stored`；不补任何默认合同 |
| 父计划有记录但 `op` 不指向那条唯一任务、或 requirement 与 op 不一致、或有 requirement 无合同、或父计划未计算 hash | **拒绝**，并说明原因；不会删掉记录继续跑 |
| 有 requirement 却没有 `logical_ops` | 在 `Plan` 构造阶段就拒绝（两个字段只能同留同丢） |

### 22.5 新增的三条检查（怎样发现“说明”和“计算”不一致）

只在计划自带 `logical_ops` 时启用，且全部只看计划内部关系：

1. **名字闭包**：逻辑算子提到的每个名字，必须是本计划里的 binding 名，或本计划某个算子的输出。转换后仍写父计划的 `Q`（而只剩 `Q_0..`）会被拒。
2. **数据来源一致**：一条任务的输入若来自 host upload，该 upload 的 `source_binding` 名字必须出现在“认领这条任务的那些算子”的 `inputs` 里；输出若被 download 取走，名字必须出现在它们的 `outputs` 里。`generator` 为 `identity`/`zero` 的常量是硬件管道（EW 的第三操作数），明确排除。**融合任务按认领它的所有算子的并集判定**——`co.ew_add_relu` 一条任务同时实现 add 与 relu，输出名属于 relu 而输入名属于 add，用并集才既严格又不误伤融合。
3. **`derived_from` 结构**：只允许 `plan_hash`（64 位小写十六进制）+ `op_index`（非负整数）两个键。

错配负例（**全部先重算 `plan_hash`**，所以拒绝只能来自上面的关系检查）：

| 篡改 | 结果 |
|---|---|
| 删掉一份副本的 requirement | 拒：`a logical op with a contract has no numeric requirement` |
| 把某份 requirement 指到另一份 op | 拒：`disagrees` |
| 把一份的任务挪给另一份 | 拒：`device tasks are not attributed to any logical op` |
| 保留父计划的输入名 `Q`/`K` | 拒：`logical op names a value this plan does not define` |
| 复用父计划的 action id | 拒：`logical op references a non-NPU action` |
| 伪造 `derived_from`（坏 hash / 负 op 号） | 拒：`derived_from.plan_hash` / `op_index` |
| 只改 hash、内容不变 | 接受 |
| 把某个名字在 binding、op、requirement 里**一起**改成另一份自洽请求 | 接受（执行字段不变） |

### 22.6 输入改接后实际读什么（可读且可检查）

`remap_group_inputs` 不复制记录、也不改记录里的名字——binding 名字没变，所以说明仍然成立；它改的是每个 upload 实际读取的窗口（`src_row_stride`、`src_col_offset`、`transpose_source`）与输入 binding 的 shape。为了让“实际读什么”不用人去翻 action，`explain` 的 `logical` 段现在把两者**连接**起来（不复制第二份数据，避免以后走样）：

```json
{"index": 1, "opcode": "matmul", "inputs": ["Q_1", "K_1"], "tasks": [5],
 "derived_from": {"plan_hash": "…", "op_index": 0},
 "reads": [{"task": 5, "binding": "Q_1", "col_offset": 16, "row_stride": 192, "transpose": false},
            {"task": 5, "binding": "K_1", "col_offset": 80, "row_stride": 192, "transpose": true}],
 "writes": [{"task": 5, "binding": "S_1"}]}
```

即：第 1 份读 `Q_1` 的第 16 列起的 16 列、`K_1` 的第 80 列起的 16 列（并转置），写 `S_1`。三个实例分别读 0/16/32 与 64/80/96（见 §22.9 板测）。

另外修掉一个潜在崩溃：`repeat_expand.py` 一直在错误分支里使用 `RK_E_SCHEMA`，却从未 import 它（既有代码里的隐藏 NameError，只在错误路径触发）。

### 22.7 修复前后执行内容对照（离线 A/B，实测）

把本轮改动前的 `passes/repeat_expand.py` 与 `plan.py`（取自本轮快照）放回一份独立包副本，用同一份请求分别编译、展开、改接，逐键比较：

| 案例 | bindings/constants/buffers/views/actions/submits/publish/accounting/预算 | 记录 |
|---|---|---|
| parent（不转换） | **完全相同**，`plan_hash` 都不变 | 1 → 1 |
| group_r2 / group_r2_shared | **逐字节相同** | 0 → 2 |
| group_r3 / group_r3_shared | **逐字节相同** | 0 → 3 |
| mapped（改接后） | **逐字节相同** | 0 → 3 |

结论：本轮只改“说明”和“身份”（`logical_ops`、`numeric_requirements`、`plan_hash`），**没有改任何会被设备执行的字段**。

### 22.8 C / mock 验证

- **mock 执行**（`librknpu_plan_mock.so`）：组计划一次跑 N 个任务的结果与“父计划跑 N 次”的结果**逐位相同**；改接后的组计划每个副本与“用该窗口数据单独跑父计划”的结果**逐位相同**（含窗口错位会立刻失败的断言）。
- **真实 C 验证器**（`custom-ops/build-host-mvp/librknpu_plan.so`）：带记录的 group / mapped 计划 `create_plan` 全部接受；由 `tools/compile_models.py` 重新生成的 6 个派生组计划（含 32 任务的 b8）**全部可加载且被 C 接受**。

### 22.9 板测（RK3588，实测，现有安全流程）

设备：RK3588 OPi 5 Plus，`6.1.43-rockchip-rk3588-rknpu098`，RKNPU v0.9.8，core mask 1；库为**当前源码在板上构建过的那一份**（sha256 `c6a6dd1f…b64709`，见 §19.10/§21 记录），本轮未重建；计划 sha256 逐条记在 `board/results/environment.log`。

| 案例 | 计划 | 任务/提交 | 设备状态 | 与 f16 参考的最大绝对差 | 不一致元素 |
|---|---|---|---|---|---|
| parent | `3ead2699a200` | 1 / 1 | PASS | 0.0 | **0 / 512** |
| group_new（带记录） | `49439ba659c6` | 3 / 1 | PASS | 0.0 | **0 / 1536** |
| group_old（修复前转换） | `09e11a9277a2` | 3 / 1 | PASS | 0.0 | **0 / 1536** |
| mapped_new（带记录+改接） | `5ee6468e3e64` | 3 / 1 | PASS | 0.0 | **0 / 1536** |
| mapped_old | `f6cf43401cc0` | 3 / 1 | PASS | 0.0 | **0 / 1536** |
| single_0/1/2（独立执行） | `3ead2699a200` | 1 / 1 | PASS | 0.0 | **0 / 512** |

三组逐位比较（板上原始 npz）：

1. `group_new` 与 `group_old` 的 6 个输出数组**逐位相同**；`mapped_new` 与 `mapped_old` 同样逐位相同 —— 记录的有无不影响设备结果。
2. 每个副本的输出与它**自己的单独执行**（single_i）逐位相同（3 副本 × 2 轮）。
3. 改接后的每份读的是**预期窗口**：参考值按 `Q[:, 16i:16i+16] @ K[:, 64+16i:64+16i+16]` 计算，设备结果与它**0 个元素不一致**；若窗口错位，这一列不可能为 0。

### 22.10 兼容、回归与身份

| 项 | 结果 |
|---|---|
| 全量回归 | **600 passed / 1 failed / 1 skipped / 139 subtests**；唯一失败是本轮之前就存在的 `test_replay_smollm2_v5`（环境缺 `transformers`，基线同样失败） |
| 新增测试 | `tests/test_plan_conversion_records.py`（23 例）：1 次/多次展开、shared_bo 开关、存盘后再展开、展开后改接、不同窗口、共享同一窗口、旧计划无记录、6 类错配负例、一致重写被接受、mock 逐位对照、真实 C 接受 |
| 既有计划 | 旧计划（无记录）读入/重存/重放字节不变；由旧计划展开的组计划仍是“未保存要求” |
| 重新生成的模型 | `tools/compile_models.py --out <证据目录>` 全部成功；6 个派生组计划带记录（b1：4 个 op/4 条要求；b8：32/32，全部 attributed） |
| schema | `derived_from` 为可选字段；未设置的算子不写该键，因此**普通新编译计划的字节与 §21 一致**（parent 的 `plan_hash` 在旧/新两个代码树里相同） |
| C 库 | 未改（`librknpu_plan.so` 源与二进制均未动） |

本轮改动的文件：`passes/repeat_expand.py`、`plan.py`、`codegen/pipeline.py`（explain 连接）、`tools/compile_models.py`（manifest 计数）、`tests/test_plan_conversion_records.py`（新）。哈希见 `results/20261101T160000Z-plan-conversion-records/final-hashes.txt`。

### 22.11 尚未完成（具体事项）

1. **嵌套展开不支持**：父计划必须恰好 1 个逻辑 op，所以“展开后再展开”会被拒（一条 `derived_from` 只记一跳）。目前没有这种调用。
2. **`repeat_expand` 的“op 不指向本计划唯一任务”分支不可达**：`Plan` 构造阶段就会拒掉指向非任务 id 的记录，admit 又只接受单任务计划，所以该分支只是防御代码，没有测试能真正走到（测试改成了断言“这种计划根本构造不出来”）。
3. **仓库里的 `models/compiled/*.rkplan.json` 仍是旧的（无记录）**：本轮只把重新生成的版本写进证据目录，没有覆盖归档（覆盖会改变既有 manifest 身份）。要让归档也带记录，重跑 `tools/compile_models.py` 即可——它现在会把记录计数写进 manifest。
4. **`remap_group_inputs` 仍用 origin 里的 `upload_a:`/`upload_w:` 标记判断“这个 upload 属于哪个角色/第几次重复”**：这是本模块自己写的硬件打包标记，不是逻辑归属猜测；改成结构化字段是另一处独立改动，本轮没做。
5. **`benchmark_repeat_paths.py` 未重跑**（本轮不做性能工作）；它只在内存里展开，不落盘。

### 22.12 重放命令

```bash
cd research/tilelang-rknpu
# 本轮新增的转换测试（含 mock 逐位对照与真实 C 接受性）
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_plan_conversion_records.py -q -o addopts=""
# 既有重复展开/分组测试
PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_repeat_expand.py tests/test_repeat_group.py -q -o addopts=""
# 重新生成模型（写进临时目录，不动归档）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python tools/compile_models.py --out <dir>
# 新旧执行对照
PYTHONPATH=$PWD/python .venv/bin/python results/20261101T160000Z-plan-conversion-records/ab/dump_conversions.py <out-dir>
```

证据目录：`results/20261101T160000Z-plan-conversion-records/`（`snapshot/`、`ab/new|old`、`ab/compare.json`、`compile-models/`、`board/`（计划、用例、`results/` 原始 npz 与 `device-metrics.json`）、`logs/`）。

---

## 23. 重核最初的审查清单：还剩什么（2026-11-01，只清点，不改实现）

### 23.1 一句话结论

最初 F1–F10 里，**会丢或改写上层合法要求的 5 条已经全部关闭**：F1（声明被丢弃）、F2≡F4（planner 自产计划被校验器簿记上限挡下）、F3（来源标记当闸门），以及后来追加的 §19.8/§20.3/§22（数值记录与逻辑说明的丢失、篡改不受检、转换丢记录）。

**仍然成立的只剩“说明层 / 错误信息”级别的 4 条**（F5 的文档缺口、F6 的审计快照漏项、F8 的属性字典无模式、F9 的残留 schema），外加 F7 需要重新分类。**没有发现任何仍会阻断合法输入的新缺陷**：本轮 28 个 planner 自产计划 + 85 个已存计划，在 Python 侧加载与真实 C 校验器上**判定完全一致**（85/85 接受，0 例“Python 放行、C 拒绝”）。

### 23.2 汇总表（最终结论以本表为准）

| 原编号 | 问题 | 当前结论 | 理由（一句话） | 关键证据 | 后续动作 |
|---|---|---|---|---|---|
| **F1** | linear 策略分支丢弃已声明调度、绕过数值契约门 | **已解决** | 三种 linear 策略都把声明落到 `applied_schedule`；同一份源在不同策略下的接受性一致 | §23.5.0 的 `f1_linear_policy_matrix` / `f1_numeric_gate` | 无（`TILELANG-SCHEDULE.md:70-72` 的说法现在成立） |
| **F2** | Python 说“已验证”，C 拒绝 | **已解决，且与 F4 是同一个问题** | 唯一已证实例是 slot 成员数超 64；根因在 planner 的 slot 划分，已修；`verify.py` 复算同一规则 | `slot_bound`（峰值正好 64，C 接受）+ 85 计划全一致 | 无 |
| **F3** | “未声明调度”有两种编码、且该状态是采纳开关 | **已解决（缺陷部分）**；现象保留为有意语义 | 来源不再参与准入；未知来源直接拒绝；`explain` 报 `mn_grid_source` 区分“注解/源码几何/默认网格” | 报告 §16.5；`source_identity()` | 无（可考虑把三态命名写进文档） |
| **F4** | `RK_MAX_BUFFERS_PER_SLOT=64` 拒绝 planner 自产计划 | **已解决（=F2）** | 满 64 就开新 slot；1×256 声明计划峰值正好 64 并被 C 接受 | `slot_bound`；§13.5 的 64/65 C 边界负例 | 无 |
| **F5** | L1 形状域未在任何声明层落地 | **部分仍成立**：诊断已修，文档仍缺 | 诊断现在明确给出 `linear_task_order=baseline` 替代路径；但 SUPPORT-DOMAIN/capabilities/options 仍无该域 | §23.5.1 | 需要用户决定是否补文档（低成本） |
| **F6** | `options_to_dict` 漏 `fuse_task_chain` | **仍然成立（说明层）** | 该选项把 SwiGLU 的 submits 从 5 变 1，而两份编译快照逐字段相同 | §23.5.3 | 低风险补齐 |
| **F7** | 控制面 8 条“断链” | **部分成立，需重新分类** | 逐条重核后：1 条确实无读取者、1 条根本没有 C 字段、1 条只有范围检查、1 条其实是显式固定值，3 条已有读者 | §23.5.2 | 先定 ABI 归属再动 |
| **F8** | `Action.attrs` 无 schema；未知键静默通过、坏类型抛裸 `ValueError` | **仍然成立（错误信息层）** | `unknown_key` 被接受；`source_binding="x"` 抛 `ValueError` 而不是项目诊断 | §23.5.4 | 低风险 |
| **F9** | `schema/rkplan.v1.json` 与实现不同代 | **仍然成立（残留文件）** | v1 无任何消费者；`PLAN_SCHEMA_VERSION=2` 且拒绝其它值 | §23.5.5 | 低风险 |
| **F10** | `linear_task_policy` 1 写 0 读 | **原判断已不成立** | `explain` 现在读取并输出它（实测 `['merged']`） | §23.5.6 | 无需动作 |
| **§17** | 两级 GEMM 板测有 2 个元素超旧容差 | **已定位；属“验收规则未定”，不是中间数据传错** | 单级容差被当作整链判据；生产默认路径与声明路径逐位相同 | §17.3/§18.4 | 需要用户决定整链判据 |
| **§18** | NumericContract 承诺与实际检查的差距 | **已被 §19/§21 覆盖** | 合同现在随计划保存，并与逻辑表交叉校验；**容差是否被硬件满足仍不由静态检查证明** | §21.5 | 无（除“整链判据”待定） |
| **§19.8 / §20.2** | 融合路径 elementwise 记录无归属 | **已解决** | 6-op SwiGLU 全部 6/6 有任务；8-op packed FFN 的 6 个 elementwise 全有任务 | §21.9 A/B | 无 |
| **§19.4 / §20.3** | 重算 hash 后改 `op_index`/`opcode`/输入输出仍被接受 | **已解决** | 计划自带逻辑表，记录与表逐项交叉校验 | `tests/test_logical_attribution.py::CrossCheckTests` | 无 |
| **§21.12 / §22.11** | 重复展开与输入改接丢记录 | **已解决** | 每份副本一个 op/一条要求/一条任务 + `derived_from`；改接不改记录、只改读取窗口 | §22（含板上逐位对照） | 无 |
| **§四.6** | 还有没有别的入口会丢新保存的记录 | **没有发现新的** | 全仓只有 2 处构造 `Plan`（正常编译、重复展开）；其余为 `replace`/`from_dict`，字段随之保留；没有手写计划 JSON 的写者 | §23.5.7 | 无 |
| *补充* | `fallback` / `tile_limit` / `core_policy` 三个“半死”字段 | **归类为显式固定值或无语义槽位，不算缺陷** | 见 §23.9.1 | 交叉核对 | 无（`core_policy` 需多核授权才有意义） |
| *补充* | “Python 比 C 松”的 3 类镜像缺失（physical_cols 上限、依赖/读覆盖、slot 容量） | **实测当前不可达，属防御纵深** | 公开入口被上游拒绝，手工改计划被 canonical/容量检查先挡下 | §23.9.2 / `probe_bounds.py` | 无 |
| *补充* | `explain_plan(plan)` 的 policy/k_tile 是常量 | **潜在说明问题，当前不可观察** | 唯一可编译的数值模式就是 `rk_f16_tiled`；`artifact.explain` 会用快照覆盖 | §23.9.3 | 无（若将来增加数值模式需一并修） |

### 23.3 哪些是同一个问题（不必重复追）

| 合并 | 说明 |
|---|---|
| **F2 ≡ F4** | 同一个根因：`RK_MAX_BUFFERS_PER_SLOT` 是校验器的簿记上限（`include/rk_plan.h:34` 注释写明“verifier complexity bounds”，不是设备限制），planner 越界产出。修的是 planner 的 slot 划分（`passes/storage.py:45`），不是 C。 |
| **F3 ≡ §16** | “来源标记当闸门”与“一个字段三种用途”是同一处代码的两个说法，§16 已修。 |
| **F1 ≡ §12** | F1 的“声明零痕迹丢弃”与 §12 的“名义分块与尾块/reblocked 报告”是同一条线的两端。 |
| **§19 → §21 → §22** | 同一条线：把数值要求存进计划 → 让记录可被逻辑表检查 → 让转换不丢记录。三步都已落地。 |
| **F6 / F7 / F9 / F10** | 同属“写进计划或快照，但读的人不全”。F7 是这一族的全表，F10 已被后来的 `explain` 补上，F6/F9 仍未补。 |

### 23.4 需要纠正的历史说法

1. **§4 表格里的 “options.tile_limit 死旋钮” 不准确**。`options.py:70-72` 对任何非 256 的值**明确拒绝**（`MVP tile_limit is fixed at 256`），它不是被静默忽略的旋钮，而是“只允许一个值、其它值报错”的参数。分类应改为“显式固定值（不是丢失控制）”。
2. **F10 的 “`linear_task_policy` 全仓 1 写 0 读” 已不成立**：`codegen/pipeline.py:358` 的 `explain` 会读取并输出它（实测 linear merged 计划输出 `['merged']`）。
3. **F2 的“Python 已验证 ≠ C 可接受”作为一般命题说得太宽**：已证实例只有 slot 一条；本轮 85 个计划（53 个归档 + 26 个重新生成 + 6 个历史证据计划）与 28 个自产计划全部两侧一致。“两套规则没有被证明同构”仍然成立，但**不能再当作一个已知缺陷**来讲。
4. **F5 的“拒绝信息不指出 baseline 可编译”已过期**：当前诊断直接写 `the M/N declaration path is linear_task_order=baseline`（§23.5.1 实测）。F5 只剩“文档没写这个实验域”这一半。
5. **§9 第 4 条（F4 的 >64/slot 可执行性仅是源码推断）可以升级**：现在既有 C 侧边界负例（64 接受 / 65 拒绝，§13.5），也有执行侧证据（136 slot、峰值 64 的计划在板上真跑过，§14.2 B4）。
6. **§4 关于 `>64/slot 完全可执行` 的更正应保留**（§13.2 已写）：它是校验器簿记界，不是设备限制；这一条本来就分类正确，无需再改。

### 23.5 仍然成立的问题（逐项复现）

#### 23.5.0 先给“已解决”的那一组实测数字（F1/F2/F4）

复现：`PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T180000Z-intent-fidelity-reinventory/repros/probe_matrix.py`

| 策略 | 声明 128×256 | 声明 256×256 | 未声明 | applied 里实际用的 tile |
|---|---|---|---|---|
| baseline | 36 task / 12 submit | 36 / 12 | 36 / 12 | 128×256 / 256×256 / 无 |
| scheduled_baseline | 36 / 12 | 36 / 12 | 36 / 12 | 同上 |
| merged | 36 / 1 | 36 / 1 | 36 / 1 | 同上 |
| interleaved | 36 / 1 | 36 / 1 | 36 / 1 | 同上 |
| interleaved_reuse | 36 / 1 | 36 / 1 | 36 / 1 | 同上 |

- **声明被采用**：每种策略下 `applied_schedule` 都等于声明的 tile，且任务里同时留了 `source_schedule`（可审计）。
- **数值门不再被策略绕过**：`source_k_rounding=fp32_final_narrowing` 在 baseline / merged / scheduled_baseline **三种策略下都被拒**（`-4: scheduled source has a different K-rounding contract`）。这正是 F1 当初“同一份源的可采纳性取决于一个无关选项”的病象，现在一致了。
- **slot 上限不再挡路**：`tile=1×256, 形状 64×512×384` → 388 个 buffer、136 个 slot、**峰值正好 64**，C 接受；`1×64` → 908 buffer、152 slot、峰值 64，C 接受。

#### 23.5.1 F5：L1 实验域的“最后一半”仍然成立（文档）

- **具体上层要求**：用 direct-RKIR 写 `M∈{1,256}, K=576, N=3072` 的单 GEMM，选择具名的 linear 任务顺序策略，希望被接受或被明确拒绝并指出出路。
- **复现**：`PYTHONPATH=$PWD:$PWD/python .venv/bin/python -c "…matmul_module(m=128,k=576,n=3072) + linear_task_order='merged'…"`（完整脚本见 `repros/probe_f5.py`）。
- **预期 vs 实际**：M=128 时 `merged` 拒绝（`-2 linear schedule policy is limited to M=1/256, K=576, N=3072; the M/N declaration path is linear_task_order=baseline`），同一 shape 换 `baseline` 接受（36 task / 12 submit）——**诊断已经指出替代路径**。
- **仍然缺的**：这个域在 `SUPPORT-DOMAIN.md`、`capabilities/*.json`、`options.py` 里都没有出现；读文档的人不知道“具名策略只在这三个形状上有意义”。
- **它挡住了合法用法吗**：**是**（§24 更正）。当时这里写“换 `baseline` 能编译，所以不挡合法用法”，这句话是错的：用户要的是 `merged` 的提交策略，`baseline` 只是另一个策略，能编译不等于满足要求。M=128 + `merged` 在 §24 已按真正的必要条件放开并上板验证。
- **证据级别**：离线编译（Python 侧读错误码与消息）；不需要上板。

#### 23.5.2 F7：控制面断链——重新分类后的四条事实

| 字段 | 当前事实 | 分类 | 是否影响执行 |
|---|---|---|---|
| `rk_plan_action.source_k0`（C 头文件 `include/rk_plan.h:193`） | C 源里**没有任何读取者**（`grep source_k0 ../custom-ops/src` 零命中）；Python 侧作为任务属性写入 | 死字段（ABI 有槽位、无人读） | 否 |
| `Submit.recipe_id` | **C 侧完全没有这个字段**；Python 侧用它选校验用的 recipe 签名表 | Python 侧标签，不是 ABI 字段 | 否（C 用结构证明代替 recipe 推断） |
| `Submit.core_policy` | C 有字段 + 枚举 + 范围检查（`rk_plan_validate.c:934`），但调度不使用（`co_schedule.c` 零命中）；Python 侧映射为 `RK_CORE_SINGLE`（`codegen/descriptor.py:347`） | 有槽位、只做范围校验 | 否（当前固定 `core_mask=1`） |
| `applied_schedule` / `source_schedule` / `linear_task_policy` | 不进 C 结构体，但 `explain` 读取并输出（审计痕迹） | 审计痕迹（有意不进 ABI） | 否 |
| `options.tile_limit` | 只能取 256，其它值显式拒绝（`options.py:70`） | 显式固定值 | 否 |
| `options.linear_task_order` | planner 真正读它选分支（`dag.py:1092/2644`） | 有效选项 | **是**（选 builder） |

**结论**：F7 描述的现象（若干控制字段“写得进、没人读”）大部分仍存在，但除 `core_policy` 外都不涉及“用户要求被执行”的问题——它们是**审计/标签字段**，而 `core_policy` 在单核固定配置下也没有可执行语义。真正需要用户决定的只有一件事：**哪些字段必须进 ABI 并被设备侧当作契约**。

#### 23.5.3 F6：编译快照漏了会改变提交分组的选项（仍然成立）

- **具体上层要求**：审计时能从编译快照看出“这次编译到底用了什么选项”。
- **复现**：`repros/probe_others.py`。
- **预期 vs 实际**：`fuse_task_chain` 把 SwiGLU FFN 的 submits 从 **5 变 1**（tasks 仍 7、actions 仍 14），但 `options_to_dict()` 的 13 个键里**没有它**——两份行为不同的编译，快照逐字段相同。
- **代码位置**：`codegen/pipeline.py:145-160` vs `options.py:35`。
- **影响**：只影响审计可读性；不影响任何执行结果。
- **证据级别**：离线实测（本次）。

#### 23.5.4 F8：`Action.attrs` 没有模式（仍然成立）

- **具体上层要求**：手改计划（或写工具）出错时，得到项目级诊断而不是 Python 内建异常。
- **复现**：`repros/probe_others.py`。
- **预期 vs 实际**：`source_binding="x"` → `ValueError: invalid literal for int() with base 10: 'x'`（不是 `BackendError`，没有 stage/action_id）；未知键 `unknown_key=1` → **静默接受**。
- **代码位置**：`plan.py` 的 `Action`（`attrs` 为自由字典）+ 各消费点直接 `int(...)`。
- **影响**：只影响错误路径的可诊断性；编译器自产计划不受影响。

#### 23.5.5 F9：`schema/rkplan.v1.json` 仍与实现不同代（仍然成立）

- **复现**：`grep -rn 'rkplan.v1' --include=*.py .` → 除报告本身外**零命中**；`PLAN_SCHEMA_VERSION = 2` 且 `Plan.__post_init__` 拒绝其它值。
- **影响**：读者可能误以为 v1 仍可用。删除或标注“历史”即可，风险极低。

#### 23.5.6 F10：原判断已被后续改动推翻

`explain` 的 `schedule.linear_task_policy` 会读出该字段（linear merged 计划实测 `['merged']`）。`chunk_index` 仍只有一个单测覆盖，但不是缺陷。

#### 23.5.7 §四.6：还有没有别的入口丢新记录

- 全仓构造 `Plan` 的地方只有两处：`passes/storage.py:131`（显式传入两字段）与 `passes/repeat_expand.py`（已修，§22）。
- 其余都是 `dataclasses.replace` 或 `Plan.from_dict`（字段随之保留）：`plan.py:989/991`、`tools/refactor_plan_projection.py:184`、`tools/make_schedule_override_policy.py:100/105`、`smollm.py:348/361/395`。
- 没有任何代码手写计划 JSON（`grep 'json.dump' … | grep -i plan` 只命中报告类文件）。

### 23.6 有意未支持的能力（与缺陷分开）

| 能力 | 为什么有意不做 | 是否挡住合法用法 |
|---|---|---|
| L1 具名策略只覆盖 `M∈{1,256}, K=576, N=3072` | 该域只有 decode 形状的板上证据（§4 E 类） | 不挡：同一输入换 `baseline` 可编译；缺的是文档 |
| 原语 tile `M/Kt/N ≤ 256` | CUBE 原语与已验证 recipe 的边界（`custom_ops.h:18-22`） | 挡的是超出原语的形状，与“中间层丢控制”无关 |
| `tile_limit` 只能是 256 | MVP 固定值 | 不挡：非 256 会明确报错，不是静默 |
| `core_policy` 不参与调度 | 当前固定 `core_mask=1`，多核无板上证据 | 不挡单核用法；多核需要新的板测授权 |
| 多核 / 2 核、4 核 | 超出既有授权 | 未测，不算缺陷 |

### 23.7 本轮仍未验证

1. **两套校验器是否“同构”**：本轮只证明“已测的 113 个计划两侧一致”，没有证明任意计划一致。
2. **多核**：`core_mask=2/4` 未测（用户未授权新硬件配置）。
3. **真实模型是否触发 F5 的具名策略域**：本轮未扫模型源码；历史上 decode merged 路径的声明为空（F1 真实负载栏）。
4. **整链容差判据**：§17 的 2 个超差元素已定位到“单级容差当整链用”，但**该用哪个判据属于验收政策，需要用户决定**。
5. **`RK_MAX_SUBMIT_TASKS=64` 是硬件上限还是命令打包簿记**：`include/rk_plan.h:13` 无注释，未判定（§23.9.4）。
6. **多任务 + `mode=separate` 的执行语义**：C 校验器不拒绝，runtime 只是跳过静态缓存；确认是否真按 per-task 逐次提交需要跑 `rk_plan_host_test`（本轮未做）。

### 23.8 优先建议（最多 3 项，推荐第 1 项）

| # | 建议 | 风险 | 为什么排这个位置 |
|---|---|---|---|
| **1（推荐）** | **说明层一致性包**：`options_to_dict` 补 `fuse_task_chain`（F6）＋ 删除或标注 `schema/rkplan.v1.json`（F9）＋ 把 L1 实验域写进 `SUPPORT-DOMAIN.md`（F5 的剩余一半） | 极低（不改任何执行路径、不改判定） | 这是**唯一“静默”性质的残留**：一个会改变提交分组的选项在审计快照里看不见；另外两项是纯文档。三项加起来≈一次小 PR，不需要新证据。 |
| 2 | `Action.attrs` 键/类型白名单，并把裸 `ValueError` 变成 `BackendError`（F8） | 低（只动错误路径），但需要先枚举现有全部键 | 只影响错误可诊断性，不影响执行；可在 1 之后单独做。 |
| 3 | 决定哪些控制字段必须进 ABI（F7 的 `core_policy` / `source_k0` / `recipe_id`） | 中（触及 C 结构体与调度语义，多核才有实际意义） | 当前单核配置下没有可执行语义，先定语义再动代码。 |

**不建议**现在改 `core_policy`/多核、也不建议为了消除“两套校验器未证明同构”去重写 C 校验。**如果用户不打算做说明层清理，那么本轮没有必须改的代码**——F1/F2/F3/F4 与三处记录丢失都已关闭。

### 23.9 交叉核对的补充（只读复核代理 + 本轮复核）

下面六条不在 F1–F10 的原文里，但属于同一把尺子（“上层要求能否被中间层准确处理/说明”），一并归档。

#### 23.9.1 三个“半死”选项/字段（归类，不是缺陷）

| 字段 | 事实 | 归类 |
|---|---|---|
| `options.fallback` | 值域只有 `error`（`options.py:50-57` 拒绝其它值）；除回显外无消费者 | **显式固定值**：只有一种失败策略时没有可表达的选择，不构成“丢失控制” |
| `options.tile_limit` | 唯一读点是 `options_to_dict` 回显（`pipeline.py:152`）；`passes/` 零引用；非 256 直接报错（`options.py:70`） | 同上（§23.4 第 1 条已更正“死旋钮”的措辞） |
| `Submit.core_policy` | planner 只产出 `single_core_per_submit`；`RK_CORE_AUTO` 合法但**没有生产者**；C 只做值域检查（`rk_plan_validate.c:934`），调度不使用 | **有槽位、无语义**：当前固定 `core_mask=1`，没有可观察的执行差异；要它有意义必须先有多核授权与板测 |

#### 23.9.2 C 独有 / Python 少镜像的检查：实测当前都不可达

| 差异 | 位置 | 能否用公开入口或手工改计划造出来 |
|---|---|---|
| cube `physical_cols` 只到 `CO_MAX_K=4096`（Python 的 canonical 检查只到 65520） | `rk_plan_validate.c:78` vs `plan.py:303-308` | **不能**：planner 上游限 K≤4096；手工把 view 放宽会先被“view 超出其逻辑 buffer”拒绝（`probe_bounds.py` 实测 4096/4288/8192 三种都拒） |
| 生产者必须声明为依赖、读区间必须被前序写覆盖 | `rk_plan_validate.c:577/702/710/734` | 不能：planner 逐任务写 deps；手工删 deps 会被“依赖必须前序”与读覆盖规则拒绝 |
| slot 容量溢出 / 缓冲活跃区间重叠 | `rk_plan_validate.c:783/1537-1560` | 不能：容量与 liveness 由 `storage.py` 统一分配，手工改会被别名/容量检查拒绝 |
| target 门（设备树 compatible + 驱动版本精确匹配） | `rk_plan_target.c:54-95` | 与计划无关，是**运行时设备匹配**，不是计划的传递问题 |
| submit 任务序列白名单 / multi-K 顺序 / CBUF reuse | `rk_plan_validate.c:1007-1407` | C 独有；Python 侧由 recipe 表与 planner 生成保证，不是双检 → 属**防御纵深** |

结论：这三类“Python 比 C 松”在**当前所有入口上都不可达**；它们不是用户能撞到的缺口。真正的含义只有一句：**“Python 已验证”不构成“C 必然接受”的证明**，而这句话现在有 113 个实测样本支持它在实践中成立。

#### 23.9.3 `explain_plan(plan)` 的 policy 段是常量（潜在说明问题）

`explain_plan()` 直接对计划调用时，`policy.numeric_mode` / `policy.fallback` / `schedule.k_tile` 是**代码里的常量**（`pipeline.py:432/437-441`），而计划本身根本不存 options 块（实测 `sorted(plan.to_dict())` 里没有）。`artifact.explain` 会用编译快照覆盖这三个值（`artifact.py:161-168`），所以只有“直接对已存计划调 explain”才会看到常量。

**当前不可观察**：唯一能编译成功的数值模式就是 `rk_f16_tiled`（`strict` 在任何模块上都被 legalize 拒绝，`probe_bounds.py` 实测），`fallback` 只有 `error`，`k_tile` 恒为 256。因此这是潜在问题，不是现在的错误陈述。

#### 23.9.4 一条仍未定的限制

`RK_MAX_SUBMIT_TASKS = 64`（`include/rk_plan.h:13`）没有注释说明它是硬件上限还是命令打包簿记；它与 `RK_MAX_PROGRAMS_PER_SUBMIT=64` 同值。本轮不判定，列入 §23.7。

### 23.10 基线、证据与重放

- 基线：`main @ de4b07a` + 工作区未提交改动（28 项，见 `results/20261101T180000Z-intent-fidelity-reinventory/snapshot/status.txt`）。本轮**未改任何生产实现**。
- 证据：`results/20261101T180000Z-intent-fidelity-reinventory/repros/`（`probe_matrix.py`、`probe_others.py`、`probe_f5.py`、`probe_bounds.py`、`reinventory-probes.json`、`reinventory-probes2.json`）。

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T180000Z-intent-fidelity-reinventory/repros/probe_matrix.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T180000Z-intent-fidelity-reinventory/repros/probe_others.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T180000Z-intent-fidelity-reinventory/repros/probe_f5.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T180000Z-intent-fidelity-reinventory/repros/probe_bounds.py
```

---

## 24. M=128 能不能跑 merged：把形状名单换成必要条件（2026-11-01）

### 24.1 一句话

`M=128, K=576, N=3072` + `linear_task_order="merged"` **能跑**，而且**只被那张形状名单挡着**。名单不是实现的需要，它是“第一次上板时用的是哪两个形状（decode 的 M=1、prefill 的 M=256，SmolLM lm_head 的 K=576/N=3072）”的记录。本轮把名单换成实现真正要求的条件，M=128 因此可用（36 个 task，**1 次 submit**，C verifier 接受），并且**M=1 与 M=256 的计划逐字节没有变化**（11 个旧计划 A/B 全等）。

### 24.2 原先为什么拒绝

`passes/dag.py` 的 `build_scheduled_linear` 在读完形状后有一句：

```python
if m not in (1, 256) or k != 576 or n != 3072:
    reject(RK_E_UNSUPPORTED, "linear schedule policy is limited to M=1/256, K=576, N=3072; ...")
```

它由提交 `237ff1f`（“add explicit GEMM task-order and CBUF reuse policy”）与策略本身一起引入。当时唯一的上板材料是 decode 形状（M=1）与 prefill 形状（M=256）的 lm_head：K=576（3 个 256 宽 K 切片，尾块 64）、N=3072（12 个 256 宽 N 块）。名单把“我们测过的形状”写成了准入条件。

### 24.3 这条理由是否必要（逐条查实现与 C 侧）

| 名单里的成分 | 实现/校验器真实要求 | 结论 |
|---|---|---|
| `M ∈ {1, 256}` | 原语 M 只要求 1..256；`output_tiles` 允许 M 不是 tile_m 的整数倍（尾块行数更少也合法），C 侧 `co.mm` 只比较 x/out 的行数与物理几何 | **不必要** |
| `K = 576` | K 必须按 256 升序切片；**但**分组策略用的 recipe `multi_k_acc_v1` 要求出现 `co.mm_acc` 任务（`plan.py:_recipe_matches`），即 K 必须大于一个切片（>256）。K=576 只是恰好 3 片 | **不必要**（真正的条件是 K>256） |
| `N = 3072` | N 只受 binding 维度上限 4096 与“一个 submit 最多 64 个任务”约束；N=3072 意味着 12 个 N 块 × 3 个 K 片 = 36 个任务 | **不必要**（真正的条件是 `tiles × k_slices ≤ 64`） |
| （名单外）B 必须是 `[K,N]` kn | 代码里本来就单独检查 | 必要，保留 |
| （名单外）原语 tile 1..256 | 声明的 tile 由 `mn_schedule_from_attrs` 检查；未声明时默认 256/1 | 必要，保留 |
| （名单外）预算 | `passes/storage.py` 检查 device/host 预算 | 必要，保留 |

反向证据（把名单在一份**只改这一处**的包副本里去掉后实测）：M=64/128/192/256 全部 36 task / 1 submit、C 接受；M=300 与 M=512 被“64 任务一个 submit”的既有检查拒绝（72 个任务），并给出可替代策略；K=256 被 recipe 检查拒绝（单切片没有累加步）。**没有任何一处因为 M=128 本身而出错。**

### 24.4 修改后依据什么允许

`build_scheduled_linear` 现在只检查实现真正要求的东西，并把理由写在代码里：

1. 一个 matmul，且 B 是 `[K,N]` kn（原有）；
2. **分组策略需要真正的 K 链**：`merged` / `interleaved` / `interleaved_reuse` 把每个 tile 的每个 K 片放进同一个 `multi_k_acc_v1` submit，而该 recipe 只有在存在 `co.mm_acc` 任务时才被计划校验器接受；单切片（K≤256）没有累加步 → 明确拒绝并指向 `baseline`；
3. **一个 submit 装得下所选策略要发的全部任务**（≤64）：用真正会生成的 tiles × k_slices 计算（原有检查，消息里列出能装下的策略）；
4. **原语 tile 在 1..256 之内**：来自声明（`mn_schedule_from_attrs`）或默认（M=1 时 tile_m=1，否则 256；tile_n=256）；
5. 预算、依赖、别名、publish 覆盖等既有检查不变。

名单已删除。没有把 128 加进名单，也没有放开到任意形状：**K≤256 的分组、超过 64 个任务的组合、声明与来源数值冲突，仍然按原样拒绝**。

### 24.5 三个必须保留的入口（实测）

| 入口 | 结果 |
|---|---|
| 生产 TileLang 源（`models/tilelang_matmul.make_matmul_prim(128,576,3072)`） | `merged` 接受：36 task / 1 submit；`baseline` 36/12；C 接受 |
| 显式逐 K 块 FP16 + M/N 声明的真实 TileLang 源（本轮新增 `tests/fixtures/tilelang_linear_scheduled.py`，K 循环每 256 宽回写 FP16、带 `rknpu.tile_m/tile_n/mn_order/source_k_rounding`） | `merged` 接受：`applied_schedule` = 256×256（或 128×256），`source_k_rounding=explicit_per_k_fp16`，`reblocked=false`，36 task / 1 submit；C 接受 |
| direct-RKIR（手写 `source_schedule` + `source_k_rounding`） | 同上：声明被采用，数值冲突（`fp32_final_narrowing`）在 baseline 与 merged 上**都被拒**（`K-rounding contract`） |

K 链顺序与写回没有变：任务 `k0` 依次 0/256/512，`k_count=3`，模板 `co.mm.f16.v1` + `co.mm_acc.f16.v1`；计划的 `logical_ops`/`numeric_requirements` 在保存-加载后与内存一致。

### 24.6 P（baseline）与 B（merged）对照（离线 + mock + 板端）

| 项 | P: baseline | B: merged |
|---|---|---|
| task 数 | 36 | 36 |
| submit 数 | **12** | **1** |
| recipe | `linear_k_acc_v1` | `multi_k_acc_v1` |
| 模板集合 | `co.mm` + `co.mm_acc` | 同 |
| 输出 tile 几何 | 128×256（12 个） | 同 |
| 每任务 `k0` / `k_count` | 0/256/512，3 | 同 |
| publish（几何+binding 名） | 12 个 (0, n0, 128, 256) | 同 |
| 预算 | device 4.97 MiB / host 5.81 MiB | device 6.52 MiB / host 6.34 MiB（同一 submit 内不能跨 submit 复用 slot） |
| mock 输出 | 与“每 256 宽切片 FP16 回写”参考**逐位相同** | 与 P **逐位相同** |

与同一个 builder 的 `scheduled_baseline` 相比更干净：views/buffers/publish **完全相等**，只有 36 条任务的 `submit_group` 从 `mm:0:<n0>` 变成 `mk:linear`，submit 从 12 变 1；命令工作区 48 KiB → 576 KiB。**减少 submit 是真的（字段与执行都变了），不是只改了标签。**

### 24.7 板测（RK3588，实测）

设备：RK3588 OPi 5 Plus，`6.1.43-rockchip-rk3588-rknpu098`，RKNPU v0.9.8，core mask 1；库为当前源码在板上构建过的那一份（sha256 `c6a6dd1f…b64709`）。计划以**文件**形式（`save_plan` → `load_plan`）上传并执行，逐条记在 `board/results/environment.log`。

| 用例 | task/submit（计划=设备自报） | 输出 finite | P vs B | 与声明 reference |
|---|---|---|---|---|
| m128 baseline | 36 / 12 | 是 | — | r0 通过；r1 有 **1/393216** 个元素超出 `atol=rtol=8e-3` |
| m128 merged | 36 / 1 | 是 | **与 baseline 逐位相同**（两轮） | 同上（同一元素、同一差值） |
| m192 baseline | 36 / 12 | 是 | — | 两轮通过 |
| m192 merged | 36 / 1 | 是 | **与 baseline 逐位相同** | 两轮通过 |

两个输入轮次（第二轮回换了一份新 A）都跑了；保存/加载后的计划与内存计划同 hash、同 actions/submits、同记录。

**那 1 个元素怎么读（重要，不掩盖）**：设备 `0.478516` vs 参考 `0.494141`，差 `1.56e-2`；该行输出经过的累加器量级达 128，此处一个 FP16 ULP 是 `0.125`——也就是说，差值是**累加器 ULP 的 1/8**，但对最终落在 0.49 的输出而言超过了 `atol=8e-3`。这是早就记录过的“单项容差被当作整链/任意量级判据”的问题（§17、§18.4），**不是 merged 引入的**：同一输入下 baseline 与 merged 的输出**逐位相同**，两者与参考的差别完全一样。本轮不改容差、不改判据，只如实记录。

### 24.8 仍然不能执行的输入（以及为什么）

| 输入 | 结果 | 原因（实现/校验器） |
|---|---|---|
| K ≤ 256 且选分组策略 | 拒绝（`this linear task policy groups a K-accumulation chain…`） | 分组用 `multi_k_acc_v1`，计划校验器要求存在 `co.mm_acc`；单切片没有累加步。同一形状选 `baseline` 可用 |
| 任一策略、tiles × k_slices > 64 | 拒绝（`…need more than one submit's 64-task bound`，并列出能装下的策略） | 一个 submit 最多 64 个任务（C 侧同样检查） |
| N > 4096 | 拒绝（binding 维度上限） | IR 层 binding 维度 1..4096 |
| 声明与来源数值冲突（如 `fp32_final_narrowing`） | 拒绝（`K-rounding contract`） | 数值契约（不放宽） |
| 设备/主机预算超限 | 拒绝（`RK_E_BUDGET`） | 预算门；merged 比 baseline 多占约 30%（同一 submit 内无法跨 submit 复用 slot），仍在默认 256 MiB 内 |

### 24.9 兼容与身份

- **旧计划身份不变**：M=1 与 M=256 的 5 种策略共 10 个计划，加上 M=128 的 baseline（原本就走 classic builder，不受名单影响），A/B 与改动前**逐字节相同**（11/11）。
- 新增测试 `tests/test_linear_merged_admission.py`（11 例 / 18 子测试）：M 带 1/64/128/192/256、分组策略 vs 每 tile 策略、K 单切片负例、64 任务负例、声明被采用、数值冲突负例、save/load、真实 TileLang 两个源、mock 逐位对照。
- 全量回归：见 §24.10。
- 生产默认、数值政策、容差、C recipe 均未改动；C 库二进制与源码未动。

### 24.10 回归结果

全量：**611 passed / 1 failed / 1 skipped / 157 subtests**（67 s）。唯一失败仍是 `test_replay_smollm2_v5`，原因是本机没装 `transformers`，与本次改动无关（此前每轮都是同一条）。本轮新增 11 例（`tests/test_linear_merged_admission.py`），基线 600 → 611。

受影响文件单跑：`test_linear_schedule_contract.py`、`test_schedule_contract.py`、`test_rounding_semantics.py`、`test_linear_merged_admission.py` 全绿；`test_storage.py`、`test_plan_budget_contract.py`、`test_repeat_expand.py`、`test_logical_attribution.py`、`test_plan_conversion_records.py` 在全量里同样通过。

计划身份对照（`ab/old` = 改动前 `dag.py` + 其余当前代码）：M=1 与 M=256 的 5 种策略 10 个计划、M=128 的 baseline 1 个计划，共 **11/11 逐字节相同**；M=128 的四种分组策略在旧树里被拒（`UNSUPPORTED`），在新树里被接受。

C 侧：所有新接受的计划都过真实 `librknpu_plan.so`（`create_plan` 接受）；C 源码与二进制本轮未改动。

### 24.11 对 §23 的更正

§23.5.1 曾写“它挡住了合法用法吗：不。同一输入换 baseline 能编译”。**这句话不准确**：用户要求的是 `merged` 的提交策略，`baseline` 是另一种策略；能编译不等于满足要求。现在 M=128 + `merged` 已可直接执行，`baseline` 只作为数值与执行组织的对照保留。§23.8 的第 1 项（把 L1 域写进 SUPPORT-DOMAIN）随之改变性质：不再是“补一个形状名单”，而是“说明分组策略的条件（K>256、一个 submit ≤64 任务）”——条件已写在代码注释与本节。

### 24.12 未解决 / 留给用户决定

1. **那 1 个超差元素**属于“容差判据该按什么量级/整链还是单项”的验收政策问题（§17/§18.4），本轮不改。
2. **merged 的资源代价**（device +31%、host +9%、命令工作区 ×12）没有做性能测量；本轮不涉及性能结论。
3. 分组策略在 N 很大、M 很小的一端（例如 N=4096、M=128 → 48 任务）已被同一条件允许，但本轮**只对 M∈{1,64,128,192,256} × K=576 × N=3072 做了板测**；其余被允许的组合有同样的结构性证明与 C 接受，但没有设备证据。

### 24.13 重放命令

```bash
cd research/tilelang-rknpu
# 准入与 P/B 结构、负例、保存加载（离线）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T200000Z-linear-merged-m128/repros/verify_linear_m128.py
# 新增测试
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_linear_merged_admission.py -q -o addopts=""
# 改动前后身份对照（旧树 = 当前 python + snapshot/passes/dag.py）
PYTHONPATH=$PWD/python .venv/bin/python results/20261101T200000Z-linear-merged-m128/ab/dump_plans.py <out-dir>
```

证据：`results/20261101T200000Z-linear-merged-m128/`（`snapshot/`、`ab/old|new`、`repros/verify-m128.json`、`board/`（plans、cases、results/ 原始 npz、environment.log）、`logs/`）。

---

## 25. 最终清点：还有没有第 1 类实质传递错误，以及这一阶段能否结束（2026-11-01）

### 25.1 一句话结论

**没有发现第 1 类实质传递错误**：没有任何“上层已经表达、执行能力也满足，却被中间层丢掉、改错或无必要拒绝”的路径能在公开入口上复现。全量 611 passed / 1 failed（唯一失败是本机缺 `transformers` 的旧环境问题）/ 1 skipped。**建议结束这一阶段**；剩下的都是说明层、未实现能力和数值政策决定，不属于“传递错误”。

> **本结论已被 §27 更正（2026-11-02）**：上面这句话是**基于正常计划**得出的，本轮漏查了一类**矛盾分组**——submit 的 task 列表与 runtime 实际执行顺序不一致（A/C 同组、B 另一组）。它当时同时通过 Python 与 C 校验，并在 mock 上给出与正常计划不同的数值。§27 给出根因、最小反例与两层修复；“可以结束”这个建议在此之前**不成立**。

### 25.2 当前结论表（每条只给一个类别）

类别：①实质传递错误 · ②正确落实但换了表示 · ③尚未实现并明确拒绝 · ④说明/错误提示不准确 · ⑤证据不足。

| 条目 | 类别 | 结论 | 依据 | 需要动作 |
|---|---|---|---|---|
| F1 显式调度被性能策略丢弃/绕过数值门 | ② | 已解决：声明被采用，数值门在三种策略下一致触发 | §23.5.0；`probe_matrix.py` | 无 |
| F2≡F4 planner 违反 slot 上限、Python 放行 C 拒绝 | ② | 已解决：满 64 换 slot，`verify.py` 复算同一规则 | §13.5、§23.5.0 | 无 |
| F3 来源标记当闸门 | ② | 已解决：来源不参与准入，未知来源拒绝，`explain` 报 `mn_grid_source` | §16.5 | 无 |
| M=128 + merged 被形状名单拒绝 | ② | 已解决：名单换成必要条件，36 task/1 submit，已上板 | §24 | 无 |
| 数值要求保存 / 逻辑算子关联 / 转换不丢记录 | ② | 已解决：编译→保存→加载→展开→改接→再保存链路实测记录与归属完整 | §21、§22、§25.3.2 | 无 |
| **F6 `options_to_dict` 漏 `fuse_task_chain`** | **④** | 仍然成立，但**只影响报告**：两份计划（5 vs 1 submit、hash 不同）的项目快照逐字段相同 | §25.3.1（本轮实测） | 可选的小维护 |
| F8 `Action.attrs` 未知键静默通过 / 坏类型抛裸 `ValueError` | ④ | 仍然成立，属错误信息质量；**不是**有效控制被静默忽略 | §25.3.4 | 可选 |
| F9 `schema/rkplan.v1.json` 残留 | ④ | 仍然成立（v1 无消费者） | §25.3.4 | 可选 |
| F7 `core_policy` / `recipe_id` / `source_k0` 等“断链” | ②/④ | 重新分类：K 偏移与 recipe 要求已落实为任务结构与 C 校验；`source_k0` 是冗余未读字段；`core_policy` 只有值域检查（当前固定单核，无语义差异） | §25.3.3 | 无 |
| 64-task/submit 上限 | ③ | 两侧都检查（`plan.py:1126/1147`、`schedule.py:346`、C `validate.c:935`），拒绝消息准确；“超 64 分批合并”是新能力，未获批准 | §25.3.5 | 未实现能力 |
| 分组策略要求 K>256 | ③ | 准确拒绝：`multi_k_acc_v1` 需要 `co.mm_acc`；单切片没有累加步 | §24.8 | 未实现能力 |
| 显式调度只被受限两阶段消费 | ③ | 准确拒绝并说明只能用于该契约 | §16.5 | 未实现能力 |
| 原两级 GEMM 的 2 个元素超差 | ⑤ | 已定位为“单项容差被当整链判据”，不是数据传错；本轮不改判据 | §17、§18.4 | 数值政策决定 |
| M=128 merged 板测中 1/393216 个元素超 `atol=8e-3` | ⑤ | 同上：差值是累加器 ULP 的 1/8，P 与 B 逐位相同 | §24.7 | 数值政策决定 |
| 两套校验器是否完全同构 | ⑤ | 113 个计划两侧一致，但没有证明任意计划一致 | §23.9.2 | 未验证（不阻碍结束） |

### 25.3 逐项复核（对应用户列出的 A–E）

#### 25.3.1 A · 编译选项是否完整传递

`options_to_dict()` 的消费者只有两处，都是**报告**：`CompileSnapshot.to_dict()`（`pipeline.py:134`）与 `artifact.explain()["options"]`（`artifact.py:145`）。它**不**参与缓存键、重编译判断或重放身份：

- `tune_schedule.cache_key()` 用自己构造的 identity（几何、schedule、`plan_sha256`、package/lib/profile 哈希），不含编译选项；
- 重放身份由 `plan_hash` 承载（本轮实测：同一图 `fuse_task_chain=False/True` → 计划 5 vs 1 submit、`plan_hash` 不同）；
- 没有代码把 options 投影拿来比较或缓存。

**本轮实测（`repros/options_snapshot_probe.py`，SwiGLU 图）**：

```
submits_off: 5   submits_on: 1
plan_hash_off: b5108df7…  plan_hash_on: 91329230…   plan_identical: False
options_projection_identical: True    snapshot_dict_identical: True
fuse_task_chain_in_projection: False
```

所以：**两份行为不同的编译，报告里逐字段相同**。这是说明层的问题（④），不是执行、缓存或重放的问题。**不预先称为“仅说明”是靠这条实测得出的，不是猜的。**

#### 25.3.2 B · 已保存记录是否还会丢

本轮的端到端链路（`repros/closure_probes.py`）：编译 → `save_plan` → `load_plan` → `expand_static_repeat(3)` → `save_plan` → `load_plan` → `remap_group_inputs` → `save_plan` → `load_plan`：

| 阶段 | task | logical_ops | requirements | 全部任务被认领 | `derived_from` |
|---|---|---|---|---|---|
| parent | 1 | 1 | 1 | 是 | — |
| group（3 副本） | 3 | 3 | 3 | 是 | 指向 parent 的 `plan_hash` + op 0 |
| mapped（改接后） | 3 | 3 | 3 | 是 | 同上（记录不变，只有读取窗口变） |

全仓只有两处构造 `Plan`（正常编译、重复展开），其余为 `replace`/`from_dict`；没有手写计划 JSON 的写者。**没有发现新的丢记录入口。**

#### 25.3.3 C · 执行控制的实际作用

| 字段 | 实际作用 | 归类 |
|---|---|---|
| `k0` | 进入 `descriptor.source_k0`；K 偏移真正落在 **host pack 偏移**（`src_row_offset`/`src_col_offset`）上；计划校验用它核对 `k_chain` | ② 已落实为结构（C 不读 `source_k0` 这个冗余字段，不等于信息丢失） |
| `k_index` / `k_count` | → `source_k_index`/`source_k_count`，C 用来证明每条 K 链的顺序 | ② 已落实并被 C 校验 |
| `recipe_id` | Python 侧标签（recipe 签名表）；C 没有该字段，改用“任务模板序列白名单 + 逐模板契约”独立证明 | ② 等价改写的另一侧，不是丢失 |
| `core_policy` | 只有值域检查；planner 只产出 `single_core_per_submit`，`RK_CORE_AUTO` 无生产者 | ③/④ 当前固定单核，无语义差异 |
| `submit_group` | 调度器据此分组（`schedule.py`） | ② 已落实 |
| `applied_schedule` 等 | `explain` 读取；不进 C 结构体 | ④ 审计痕迹（有意不进 ABI） |

#### 25.3.4 D · 错误处理

- **F8**：未知 `attrs` 键静默通过；`source_binding="x"` 抛 `ValueError` 而不是 `BackendError`。这只发生在**手写/改计划**时；编译器的自产计划不会出现。归类 ④。**没有发现哪一个“有效控制”被静默忽略**：descriptor 读取的键（`transpose_source`、`cache`、`data_reuse`、`weight_reuse`、`broadcast`、`source_binding`、`src_*`、`k0`、`k_index`、`k_count`）全部由 planner 一致写入；其余 attrs 键（`z_source`、`gate_source`、`stage_barrier`、`chunk_index`、`k_physical_extent` 等）是**审计痕迹**，既不声称控制执行，也不被当作控制使用。
- **F9**：`schema/rkplan.v1.json` 无消费者，`PLAN_SCHEMA_VERSION=2` 拒非 2。归类 ④（残留文件）。

#### 25.3.5 E · 剩余拒绝条件

| 拒绝 | 消息是否准确 | 是否存在“能执行却被拒”的合法输入 |
|---|---|---|
| 一个 submit 最多 64 任务 | 是：Python 与 C 都在同一位置检查（`plan.py:1126/1147`、`schedule.py:346`、`validate.c:935`） | 没有“无需新能力就能执行”的反例：超限必须分批提交，这正是**未获批准的新能力** |
| 分组策略需要 K>256 | 是：recipe `multi_k_acc_v1` 要求存在 `co.mm_acc` | 没有：单切片没有累加步；`baseline` 可用（但那是另一种策略，不等于满足 `merged`） |
| 显式调度只用于受限两阶段 | 是（消息指出只能用于该契约） | 没有：多节点消费显式调度的 lowering 尚未实现 |
| tile ≤256、K≤4096、N≤4096、对齐 | 是 | 没有：属原语/协议边界 |
| 数值冲突（`fp32_final_narrowing` 等） | 是 | 没有：这是数值契约，不属于传递问题 |

**没有把“baseline 能编译”当作“满足 merged 要求”**：§24.5 起已明确区分。

### 25.4 已经过期的旧判断（不要再列成待修问题）

| 旧判断 | 现状 |
|---|---|
| F1 声明被丢弃 / 数值门被绕过 | 已解决（§12、§23.5.0） |
| F2「Python 已验证 ≠ C 可接受」作为已知缺陷 | 已随 F4 的 slot 修复关闭；113 个计划两侧一致，无第二例 |
| F3「来源标记决定准入」 | 已解决（§16） |
| F4「>64/slot 完全可执行但被拒」 | 已由 slot 拆分 + `verify.py` 复算解决，且有 C 边界负例与板端证据 |
| F5「拒绝信息不给 baseline 出路」 | 已过期：诊断现在直接写 `linear_task_order=baseline` |
| F10「`linear_task_policy` 1 写 0 读」 | 不成立：`explain` 读它 |
| §4「`tile_limit` 死旋钮」 | 更正为“显式固定值，非 256 直接报错” |
| §23.5.1「换 baseline 能编译，所以不挡合法用法」 | 已更正：能编译 ≠ 满足指定提交策略；M=128 merged 已放开并上板（§24） |
| §24 的固定形状名单 | 已删除，改为四条必要条件 |
| §19.8 / §20.2 / §19.4 / §20.3 / §22.11 | 全部已解决（§21、§22） |

### 25.5 这一阶段能否结束

**能。** 理由：

1. 五类历史缺陷（声明丢失、slot 越界、来源当闸门、记录与关联、转换丢记录）都已有代码、测试与（部分）板端证据闭环；
2. 本轮在公开入口上找不到第 1 类错误：13 条重点拒绝全部给出准确理由，其中唯一可能“可执行却被拒”的 64 任务批量合并属于**未实现能力**，不是缺陷；
3. 剩余条目要么是说明/错误提示（④），要么是未实现能力（③），要么是需要用户决定的数值政策（⑤）；
4. 继续改代码的收益是“报告更好读”，不是“上层的合法要求能被满足”。

**不建议**为了保持工作继续而新增能力（分批提交、多核、量化、更多形状的设备证据）。

### 25.6 如果仍要做一件事（最多一项，且不是必需的）

**说明层小包（F6 + F9）**：把 `fuse_task_chain` 加进 `options_to_dict()`，并删除或标注 `schema/rkplan.v1.json`。</br>理由：这是唯一一处“报告会让读者误判”的地方（两份行为不同的编译快照逐字段相同，本轮已实测）；改动不触及任何执行路径、判定或数值政策，风险极低。**不做也不影响正确性。**

### 25.7 基线、证据与重放

- 基线：`main @ de4b07a` + 30 项未提交工作区改动（本轮**未改任何生产代码**，`git diff` 与本轮开工快照逐字节相同）。
- 全量：611 passed / 1 failed（本机缺 `transformers`）/ 1 skipped / 157 subtests。
- 证据：`results/20261101T220000Z-final-intent-closure/`（`snapshot/`、`repros/closure_probes.py`、`repros/options_snapshot_probe.py`、两份 JSON、`logs/baseline-pytest.log`）。

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T220000Z-final-intent-closure/repros/closure_probes.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261101T220000Z-final-intent-closure/repros/options_snapshot_probe.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
```

---

## 26. 适配层开销清点：gate/up projection 的实际构成（2026-11-02，审查+测量，未改实现）

### 26.1 结论先说

按当前 recipe 与安全契约，**这份计划已经处在“最少任务/最少搬运/最少字节”的位置**：36 个任务 = 12 个 N 块 × 3 个 K 片（N≤256、K≤256 是原语与 K 顺序契约），3 次输入上传 = 3 个 K 片，12 次下载 = 12 个输出块（每次读只能由一个生产者覆盖），搬运字节数恰好等于输入与输出张量本身。**本轮没有找到可以安全消除的中间层开销，因此没有修改实现**（§四允许：没有可安全消除的开销时，提交充分证据后停止）。

真正的时间花在两个 C 侧拷贝循环上（每元素下标计算 + 边界检查，约 10–19 ns/元素），以及每个 BO 一次的 guard 检查（约 25 µs/BO）；两者都不是计划层面的选择，前者要改 C 运行时拷贝实现（本轮授权范围之外），后者由共享常量缓存与“一次提交”的存活性决定。

### 26.2 工作负载与测量方法

- 计算：SmolLM gate/up projection，`K=576, N=3072`，`M=1`（decode，生产路径）与 `M=128`；`linear_task_order=merged`（一次提交）与 `baseline`（每 tile 一次提交）各一份。
- 为什么选它：来源与数值语义明确（`models/tilelang_matmul.make_matmul_prim`），§24 已证明两种策略都真实上板，且它是当前 decode 的关键路径。
- 方法：**生产缓存配置**（`PlanCall(share_constants=True)`，与 `smollm.py:167` 的默认一致），**每个计划只建一次 workspace**（生产形态：一个 workspace 多次调用），2 轮 warmup 丢弃，随后 **5 组交错轮询** 采集（顺序固定 m1_base → m1_merged → m128_base → m128_merged），每轮保留原始值。
- 设备：RK3588 OPi 5 Plus，`6.1.43-rockchip-rk3588-rknpu098`，RKNPU v0.9.8，core mask 1；库 sha256 `c6a6dd1f…b64709`（当前源码的板上构建）。计划 sha256 逐条记在证据里。
- 共享常量缓存实测：第一份计划 `created 36 / reused 0`、`shared_constant_bytes 4,720,896`；第二份（同权重）`created 0 / reused 36` —— 缓存真实生效，没有重复打包。

### 26.3 每次调用的实测分解（中位数，ms）

| 项 | m1 baseline | m1 merged | m128 baseline | m128 merged | 归类 |
|---|---|---|---|---|---|
| submit_wait（设备） | 1.508 | 1.023 | 9.572 | 8.991 | ① recipe 决定：36 任务 × 几何 |
| guard | 1.234 | **1.328** | 1.173 | **1.439** | ① 保护：52 个 BO 各一次 |
| download | 0.130 | 0.112 | **3.894** | **3.916** | ① 字节最少，C 侧逐元素 |
| pack | 0.056 | 0.048 | 1.506 | 1.517 | ① 字节最少，C 侧逐元素 |
| output_sync | 0.161 | 0.097 | 0.282 | 0.176 | ① 原子发布必需 |
| publish | 0.003 | 0.003 | 0.191 | 0.193 | ① 一次下载一个 tile |
| validate | 0.006 | 0.006 | 0.076 | 0.078 | ① |
| **total** | **3.230** | **2.657** | **16.775** | **16.428** | |

其它实测：计划 JSON **55–63 KiB**；`compile_rkir` 4.6 ms、`load_plan` 3.25 ms（没有“构建/加载明显重复工作”）；workspace 初始化 27.9–49.6 ms（首次含 36 个常量打包，之后常量命中缓存）。

### 26.4 计划的构成与逐项判断

| 项 | 数量（m1/m128 相同） | 判断 |
|---|---|---|
| npu_task | 36 = ⌈3072/256⌉ × ⌈576/256⌉ | ① 原语 N≤256、K 片 256 → 最少 |
| 权重上传 | 36 = 12 N 块 × 3 K 片，3.54 MiB | ① 每个 (N,K) 片是独立的 CBUF 图像；常量，只打包一次 |
| 输入上传 | 3 = 3 个 K 片，m128 144 KiB | ① 字节 = 输入张量；K 片布局不同，无法一次上传 |
| 下载 | 12 = 输出块数 | ① “一次读只能由一个生产者覆盖” ⇒ 不能合并成一次大拷贝 |
| 输出缓冲/slot | 12 个（merged）/1 个（baseline） | ② forced by 一次提交：所有输出要活到提交结束 |
| 累加器缓冲 | 每 tile 一对乒乓 | ① `mm_acc` 的 prior 与 out 不能同 BO |
| 设备预算 | m128 merged 6.52 MiB（baseline 4.97） | ② 合并提交的代价（多 352 KiB 的 per-slot slack） |
| 搬运字节 | 输入 144 KiB + 输出 768 KiB（m128） | ① 等于张量大小，无重复 |

**guard 的 52 个对象**：36 个共享常量 BO + 3 个输入 + 13 个累加/输出。36 个常量 BO 正是“共享常量缓存”换来的（否则每个 workspace 要重打包 4.5 MiB）；12 个输出 buffer 同时存活是**一次提交**的必然结果（baseline 分成 12 次提交才只需要 1 个输出 slot，但要多付 0.49–0.58 ms 的 submit_wait）。

### 26.5 两个候选优化：查过，都不成立（或超出本轮范围）

**(A) 把 36 个常量 tile 并进 1 个 BO，省掉 35 次 guard（≈0.9 ms/call，decode 的 35%）。**
不采用：共享常量缓存是**按 upload 动作**建立 BO 的（`rk_shared_constant_get_or_create` / `create_shared_constant_on`），一个 BO 里放 36 个区域会让生产配置（`share_constants=True`）无法工作；要落地就得关掉共享常量缓存，改为每 workspace 打包 4.5 MiB（≈47 ms 初始化 + 4.5 MiB 常驻），并且要改生产默认（`smollm.py:167`）。这违反“本轮不改生产默认”，而且是一个**权衡**，不是白得的收益。

**(B) 把 12 个输出并进 1 个 buffer / 1 次下载（省 11 次 host 拷贝）。**
不采用：① 校验器要求“一次读必须由一个生产者覆盖”，整块下载会跨 12 个写者，必须放宽契约——不允许；② 即便允许，实测表明下载成本是**元素级**的（393,216 元素 / 3.9 ms ≈ 10 ns/元素），合并只省每次调用的固定开销（m1 的 12 次下载共 0.112 ms ⇒ 每次约 9 µs），合计约 0.1 ms，而不是 3.8 ms。

### 26.6 真正的时间去哪了（为什么本轮不能在中层解决）

两个拷贝循环都是**逐元素**实现（`pack_upload_into` 的慢路径每元素做一次分支+边界检查；下载侧 `co_download` + `rk_publish_rows` 同样逐元素/逐行），实测：

- pack：m128 1.434 ms / 73,728 元素 = **19.4 ns/元素**；m1 0.013 ms / 576 元素 = **22.6 ns/元素**（与数据量成线性，说明是元素级成本，不是每次调用的固定开销）。
- download：m128 3.916 ms / 393,216 元素 = **10.0 ns/元素**；m1 的 12 次下载共 0.112 ms / 3,072 元素 ⇒ 每次约 9 µs 固定开销 + 元素成本。
- 上传快路径 `fixed_pack_16x16` 只覆盖 16×16 的 repeat-group 几何，本负载的 128×256 / 256×256 tile 走不到。

把这两个循环改成块结构拷贝，预计能把 m128 的 5.3 ms（32%）和 m1 的约 0.15 ms 压下去——但那是**C 运行时的拷贝实现**，不是“主要发生在中间层适配”的修复，超出本轮授权。本轮据实记录为目标，不自行实施。

### 26.7 保持了哪些不变量（本轮未改实现，故全部保持）

数学、数值政策、K 顺序（0/256/512 升序、每块 FP16 写回）、显式调度、输入输出与权重身份、依赖与所有权、padding 与设备读取余量、越界/alias/finite/失败不发布等保护，以及 §21/§22 的逻辑说明、数值要求、任务关联与转换记录：**均未触碰**。

### 26.8 证据与重放

- 证据目录：`results/20261102T000000Z-adaptation-overhead/`（`snapshot/`、`repros/describe_plan.py`、`repros/board_probe.py`、`repros/plan_*.json`、`repros/plan-composition.json`、`board/probe.json`（5 组逐轮原值）、`board/environment.log`）。
- 计划 sha256：m1_baseline `d783bb39…`、m1_merged `72e6ea76…`、m128_baseline `a8b92f9f…`、m128_merged `96f41a7f…`。

```bash
cd research/tilelang-rknpu
# 计划构成（离线）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T000000Z-adaptation-overhead/repros/describe_plan.py
# 板端分解（需要设备与 .env；脚本已随证据归档）
PYTHONPATH=python:/usr/lib/python3/dist-packages .venv/bin/python \
  results/20261102T000000Z-adaptation-overhead/board/upload_and_run.py
```

---

## 27. P0：提交顺序（submit 与 runtime 实际执行顺序不一致）（2026-11-02）

### 27.1 根因与实际执行顺序

`rk_plan_runtime.c` 的发射循环按 **action id 升序** 扫描：遇到 `npu_task` 时取 `sid = task_submit[a->id]`、`index = emit_index[a->id]`，把该任务写进 `st->batch[index]`；当 `index + 1 == submit->ntask_ids`（即到达**该 submit 列出的最后一个任务**）时，把整个 batch 提交执行。

所以计划**声明的分组**必须与“按 action 顺序扫描得到的运行段”一致。而在修复前：

- **C**（`rk_plan_validate.c` 的 `validate_execution_order`）只检查到：submit 的 task 列表升序、`[first, last]` 区间内**每个 action 都是 NPU task**；
- **Python**（`plan.py::_validate_plan`）检查到：task 存在/唯一/全覆盖、submit deps 前序；**完全没有区间检查**。

两层都缺一条：**区间 [first, last] 里的 NPU task 必须全部属于这个 submit**（以及 submit 的运行段必须按 submit id 顺序出现）。于是 `A → B → C` 的 action 顺序配上 `submit0=(A,C)`、`submit1=(B)` 会被接受，而 runtime 实际会：

1. 扫到 A：`index=0`，`0+1 ≠ 2`，只写 `batch[0]`，不提交；
2. 扫到 B：`index=0`，`0+1 == 1`（submit1 只有 1 个任务）→ **提交 submit1**，但它跑的 `batch[0]` 已经被 A 写入、又被 B 覆盖（共享 batch 数组）→ 实际执行的是 **B**；
3. 扫到 C：`index=1`，`1+1 == 2` → 提交 submit0 = `batch[0], batch[1]` = **B 的程序 + C 的程序**。

A 从未执行，B 执行了两次；声明的依赖 A→B 被彻底违反，而输出仍然 finite。

### 27.2 最小反例（修复前，已复现）

图形：`X → sigmoid → A → sigmoid → B → sigmoid → Y`（三个 task，连续、逐个依赖）。正常计划：3 个 task / 3 个 submit（`gate_ew_v1`）。

| 计划 | Python 加载 | 真实 C `rk_plan_create` | mock 输出（首元素） |
|---|---|---|---|
| 正常 | ACCEPT | ACCEPT | **0.65087890625**（finite） |
| 交错 `submit0=(t1,t3)`、`submit1=(t2)` | **ACCEPT** | **ACCEPT** | **0.62255859375**（finite，却错） |

计数/预算/hash 都已按交错分组重算（`command_workspace = max_tasks × 2048 × 8`、`host_bytes` 补 `CO_PROGRAM_BYTES`），所以拒绝原因不可能是预算或构造错误。证据：`repros/repro_interleaved_submit.py`、修复前的计划文件已归档。

### 27.3 最终顺序规则（两层都要，各自独立）

以 **runtime 当前行为**为准（扫描 action，到该 submit 末 task 处整批提交），一条计划只有在满足下面三条时“声明的顺序”才等于“实际执行顺序”：

1. **R1 升序**：一个 submit 的 task 列表必须按发射顺序严格升序；
2. **R2 连续段归属**：`[first_task, last_task]` 区间内每个 action 都必须**是该 submit 自己的 task**。由此推出：区间内不能夹 host action（原有规则）、不能夹别的 submit 的 task（新规则）；
3. **R3 段序**：各 submit 的运行段必须按 **submit id 顺序**出现（`first(s+1) > last(s)`），否则 runtime 先提交后面的组，声明里的 submit 依赖全部失效。

这三条只是把 runtime 已经在做的事写成可检查的契约；**没有**新增 id 格式、没有要求 id 连续（upload/download 仍可以停在段的**外面**，static-repeat 与 task-chain fusion 依赖这一点）。

实现位置：

| 层 | 文件 | 内容 |
|---|---|---|
| Python | `python/rknpu_backend/plan.py::_validate_plan` | R1/R2/R3，消息分别为 `submit tasks must be listed in ascending emission order`、`a host action is interleaved inside a task group` / `another submit's task is interleaved inside a task group; a submit must own a contiguous run of tasks`、`submit task runs must appear in submit order` |
| C | `custom-ops/src/rk_plan_validate.c::validate_execution_order` | 同一套 R1/R2/R3（C 独立计算 `[first,last]` 的游标匹配与 `s-1` 段序），不依赖 Python |

### 27.4 Python / C 独立拒绝证据

修复后（`tests/test_submit_order_contract.py`，13 例 + 5 子测试）：

| 反例 | Python | C（**绕过 Python 校验**，直接喂描述符） | mock 是否开始执行 |
|---|---|---|---|
| A/C 同组、B 另一组 | REJECT R2 | REJECT R2 | **否**（构造即失败） |
| submit 段序与 id 相反 | REJECT R3 | REJECT R3 | 否 |
| task 列表降序 | REJECT R1 | REJECT R1 | 否 |
| task 重复归属 | REJECT | REJECT | 否 |
| task 遗漏 | REJECT | REJECT | 否 |
| host action 夹在段内 | REJECT（本轮新增该 Python 检查） | 原本就有该检查（本轮未改） | 否 |

C 侧证据的取得方式：先构造正常的 `Plan`，再用 `object.__setattr__` 替换 `submits`/`accounting` 并用 `sha256_json(to_dict(include_hash=False))` 重算 `plan_hash`，从而**跳过 Python 的计划校验**，只让 C 的描述符校验器看到它。

### 27.5 正常计划兼容性

- 全量 Python：**624 passed / 1 failed / 1 skipped / 162 subtests**（+13 = 新测试文件；唯一失败仍是本机缺 `transformers` 的 `test_replay_smollm2_v5`）。
- C：重建 host 库后 `ctest` **33/33 passed**。
- 新测试里逐一覆盖规划器产出的形状：activation-chain、merged 多 K、baseline 多 K、`submit_mode=separate`、多输出、static repeat（4 副本），全部继续被接受，并逐一断言“每个 submit 的运行段连续且等于自己的 task 列表”。

### 27.6 构建、mock 与真实设备

库身份（源码 → 库）：

| 用途 | 路径 | sha256 |
|---|---|---|
| host 真实（修复前） | `custom-ops/build-host-mvp/librknpu_plan.so`（快照） | `a6522722…`（改动前） |
| host 真实（修复后） | 同上，重建 | `60f5dfb5f33e70983166e9d7973e97c1ceb84ac4925170d5a9e318a4143451a7` |
| host mock（修复后） | `librknpu_plan_mock.so` | `a3f36359e70bdc665b80aeae5e9284631355b97acef95b9ffe20337f2cf36c30` |
| RK3588 修复前 | `/mnt/ssd/workspace/smollm-work-20261030T085042Z/build/librknpu_plan.before-p0.so` | `c6a6dd1f…b64709`（与 §24 同一份） |
| RK3588 修复后 | 同目录 `librknpu_plan.so`（用本次改动的 `rk_plan_validate.c` 增量重建） | `cec3fa06f4739741c5983709858135fdd41e2ee2cce0610ad48995303cd430f8` |

板上（RK3588 OPi 5 Plus，RKNPU v0.9.8，core mask 1，`share_constants=True` 生产配置）对**同一批计划文件**分别用两个库执行，各 3 轮：

| 计划 | task/submit | before 输出 | after 输出 | 逐位相同 | 轮内稳定 |
|---|---|---|---|---|---|
| SwiGLU FFN（有依赖多阶段） | 7 / 5 | sha 见证据 | sha 见证据 | **是** | 是 / 是 |
| merged 多 K（36 任务 1 提交） | 36 / 1 | — | — | **是** | 是 / 是 |
| static repeat（4 副本 1 提交） | 4 / 1 | — | — | **是** | 是 / 是 |

计数（task/submit/upload/download）在两库间完全一致；`upload_calls`/`download_calls` 未变。**异常计划没有送到设备**（拒绝证据来自离线真实库与 mock）。

数值说明（如实）：merged 与 repeat 对声明参考 `allclose` 通过；SwiGLU 的“声明参考”**不通过**（`max_abs` 20，参考量级 8164 → 相对 2.4e-3），而在 **mock** 上同一参考是 `allclose=True`。失败元素仍是 §24.7/§18.4 记录的“绝对容差落在近零输出上”的情形（累加器量级处一个 FP16 ULP ≈ 8），且**两个库输出逐位相同**，因此与本轮修复无关；本轮不改容差、也不据此宣布任何通过。

### 27.7 对 §25 的更正

§25 的“没有发现第 1 类实质传递错误 / 可以结束这一阶段”是**只检查正常计划**得出的，漏掉了**矛盾分组**这一类：校验器接受的顺序与 runtime 真正执行的顺序不一致，属于“中间层没有正确落实上层要求”的第 1 类。更正后，该阶段的结束条件应至少包含本轮的三条顺序规则；§25.5 的“建议结束”在本轮修复之前**不成立**。

### 27.8 本轮登记但不实施的项

按用户指示，以下问题继续登记、本轮不动：MM_ACC / EW / LUT / View / RKIR 相关问题、预算溢出问题，以及 §26 记录的 C 侧逐元素拷贝循环（`pack_upload_into` 慢路径、`co_download`/`rk_publish_rows`）。

### 27.9 重放与证据

```bash
cd research/tilelang-rknpu
# Python + C 独立反例（C 绕过 Python 校验）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T040000Z-p0-submit-order/repros/repro_interleaved_submit.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_submit_order_contract.py -q -o addopts=""
# 全量
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
# C 侧
cd ../custom-ops/build-host-mvp && cmake --build . --target rknpu_plan rknpu_plan_mock -j4 && ctest -j2
```

证据目录：`results/20261102T040000Z-p0-submit-order/`（快照（`plan.py`、两个 `.c`、两个 host 库）、`repros/repro_interleaved_submit.py` 与 `result.json`、`board/`（3 个计划、probe、`before.json`/`after.json`）、`logs/full-pytest.log`、`rebuild_device_lib.py`）。

---

## 28. MM_ACC / EW / LUT：校验器与发码器对齐（2026-11-02）

### 28.1 一句话结论

三类任务都存在同一类不一致：**Python 与 C 都能接受、但发码器（或 workspace 建视图）根本不支持的张量几何**。本轮把发码器已经强制要求的条件补进了两层校验器，并在每类上构造了“除目标缺陷外其它关系全部自洽”的反例；修复后 Python 与 C 各自独立拒绝，mock 不再开始执行。P0 提交顺序修复保持不变（§27，`test_submit_order_contract.py` 13 例仍全绿）。

### 28.2 A · MM_ACC：单个任务的 K 只能是一个 256 宽 tile

**以前会被接受的错误配置**：把某个 `co.mm_acc.f16.v1` 任务的 x/weight 两张量扩成 K=512（逻辑与物理都是 512），其余（上传源区间、buffer 容量、预算、hash）全部自洽。

**真正执行为什么不支持**：`co_emit_mm_acc_impl` 要求 `x.physical_cols <= 256`——一个任务只描述一个 CUBE K tile，寄存器描述里没有第二个 K 块的位置。

**最早失败位置（按实测，不是猜）**：

| 阶段 | 结果 |
|---|---|
| Python `_validate_plan`（修复前） | **接受** |
| C `rk_plan_create`（修复前） | **接受** |
| workspace / view 创建（mock，修复前） | **拒绝**：`co_view_init rejected a view` |
| 发码器 | 未到达（建视图已挡） |

也就是说这一类**最早的真实失败在 workspace 的 view 创建**，而不是发码器；报告按此记录，不套用“emitter 才拒绝”的说法。

**补的检查（两层）**：`x.cols / w.cols / x.physical_cols / w.physical_cols` 任一超过 256 → 拒绝（与 `co.mm` 分支同界；K=64 之类的 tail 仍然合法）。另外把 prior/output 的比较升级为**整张量相等**（layout/atom/bytes，对齐发码器的 `same()`），并补上发码器实际编程的物理几何关系 `physical_rows(x)==physical_rows(o)`、`physical_cols(x)==physical_cols(w)`、`physical_rows(w)==physical_cols(o)`。

### 28.3 B · EW：X/operand/output 必须是 CUBE，identity 的物理几何也要对

**以前会被接受的错误配置**：把一个 Add 任务的 x/operand/output 三张视图换成 WEIGHT 布局（同字节、同逻辑形状），identity 换成匹配宽度的方形 weight。

**真正执行为什么不支持**：`co_emit_ew` 走矩阵通道（`finish_mm_op`），只接受 CUBE 的 x/operand/output；identity 还要求 `rows==cols==physical_rows==physical_cols==x.physical_cols`。

**最早失败位置**：Python 接受 → C 接受 → **运行开始后**发码器拒绝（mock 报 `-9`，即 custom-ops emitter rejected a validated action），说明计划被认可却永远跑不起来。

**补的检查（两层）**：x/operand/output 必须 `co.cube.f16.v1`；三者 atom/bytes 相等；identity 除 `rows/cols` 外还要求 `physical_rows/physical_cols == x.physical_cols`。（`co.ew_add/mul/max/add_relu` 共用同一分支，四个模板一起受约束。）

**一处必须说明的边界**：LINEAR 布局的 EW 操作数在**任何**自洽计划里都无法表达——identity 必须是边长等于 `x.physical_cols` 的方形 weight，而 `TensorLayout` 要求 weight 的 physical_rows ≤ 256，CUBE 的最小 tile 又是 12×32=384 元素，因此 linear 视图（physical_cols = 元素数 ≥ 384）没有可表达的 identity。所以这一类可触发的最小反例是 WEIGHT 布局（本轮用的就是它），linear 形态只作为“理论上也不合法”的说明。

### 28.4 C · LUT：同字节不算相同，输入输出必须是同一张 tile

**以前会被接受的错误配置**：输入 `13×33`（物理 `16×64`，2048 B）、输出 `32×17`（物理 `32×32`，2048 B）。字节数相同、都是 CUBE、元素数都在 65528 以内，两层都只看 `bytes`，于是放行。

**真正执行为什么不支持**：`co_emit_lut` 要求 `same(x,o)`——type/layout/rows/cols/physical_rows/physical_cols/atom/bytes 全等，因为它用同一份描述符索引两侧。

**最早失败位置**：Python 接受 → C 接受 → 运行开始后发码器拒绝（`-9`）。

**补的检查（两层）**：在保留原“字节相等”“CUBE 元素上限 65528”“linear 只对 linear”之外，新增**整张量相等**。合法场景（linear 分块 LUT、CUBE LUT、FFN 里的 `co.lut.sigmoid`）不受影响：规划器本来就给两侧同一几何。

**构造说明（诚实记录）**：要让 LUT 两侧几何不同而计划其它关系自洽，输出侧必须能过 download/publish 的寻址与形状检查，因此反例里同时把输出 binding 形状与 publish tile 改成 `32×17`（即“这个计划的输出本来就这么大”）。这比用户给的“输入13×33 / 输出32×17”多了一步绑定调整，几何数字一致。

### 28.5 必要条件清单（全部有发码器代码依据）

| 模板 | 发码器条件（依据） | 修复前缺失项 |
|---|---|---|
| `co.mm_acc.f16.v1` | CUBE/WEIGHT/CUBE/CUBE；`x.physical_cols<=256`；`x.cols==w.cols`；`physical_cols(x)==physical_cols(w)`；`physical_rows(x)==physical_rows(o)`；`physical_rows(w)==physical_cols(o)`；`same(prior,o)`；prior≠o buffer 且不重叠 | K 上界、物理几何、prior/o 的 atom/bytes |
| `co.ew_*` | CUBE×3；`same(x,operand)`；`same(x,o)`；identity WEIGHT 且 `rows==cols==physical_rows==physical_cols==x.physical_cols` | CUBE 布局、atom/bytes、identity 物理几何 |
| `co.lut.*` | `same(x,o)`；F16；CUBE 或 LINEAR；元素上限 65528 | 几何/atom 全等（原来只比 bytes） |

### 28.6 Python / C 独立拒绝

同一份反例工件（先构造合法计划，再绕过 Python 校验改造并重算 `plan_hash`）同时喂给两层：

| 反例 | Python | C（Python 校验已绕过） | mock |
|---|---|---|---|
| mm_acc K=512 | REJECT `K extent` | REJECT `mm_acc K extent exceeds one 256-wide tile` | 未开始（构造即失败） |
| EW 非 CUBE 操作数 | REJECT `CUBE tiles` | REJECT `EW operands and output must be CUBE tiles` | 未开始 |
| LUT 同字节不同几何 | REJECT `same tile` | REJECT `LUT input and output must be the same tile` | 未开始 |

修复前同一份工件的表现（`repros/before-fix.txt`）：三者 Python/C 全部 ACCEPT；mm_acc 在建视图时被挡，EW/LUT 在运行开始后被发码器拒绝。

### 28.7 回归、mock 与板测

- 全量 Python：**634 passed / 1 failed / 1 skipped / 169 subtests**；唯一失败仍是本机缺 `transformers` 的 `test_replay_smollm2_v5`（环境问题，未新增失败）。
- C：重建后 `ctest` **33/33 passed**；新增检查没有拒绝任何既有 C 测试计划。
- 新测试：`tests/test_task_contracts.py`（10 例 + 7 子测试）覆盖合法 K=256/K=320(tail)/K=576、add/mul/relu(max)/add_relu、linear 分块与 CUBE LUT，以及三类反例的两层拒绝和“拒绝后不执行”。
- 板测（RK3588 OPi 5 Plus，生产 `share_constants=True`，同一批计划文件、同一批输入，各 3 轮）：

| 计划 | task/submit | 库 before `cec3fa06…` | 库 after `088a8017…` | 输出逐位相同 |
|---|---|---|---|---|
| merged 多 K（mm + mm_acc 链） | 36 / 1 | 通过 | 通过 | **是** |
| SwiGLU FFN（LUT + EW + 多阶段） | 7 / 5 | 通过 | 通过 | **是** |
| relu 尾声（`co.ew_max`） | 4 / 4 | 通过 | 通过 | **是** |

三个计划在两库下的 task/submit/upload/download 计数完全一致、轮内稳定、输出 finite。声明参考：merged 与 relu 通过（`max_abs` 0.03125 / 0.0039）；SwiGLU 的参考 `max_abs` 24（量级 8164），**两库逐位相同**，属于既有“绝对容差落在近零输出”的记录（§24.7/§18.4），本轮不改容差、不据此宣布通过。

### 28.8 身份与重放

| 物件 | sha256 |
|---|---|
| `python/rknpu_backend/plan.py` | 见 `final-hashes.txt` |
| `research/custom-ops/src/rk_plan_validate.c` | `c36d14d4e1e6d1db2efb9e5d4885665577328450b3f2db186f33f7ba7f9ca820` |
| host 真实库 | 见 `final-hashes.txt` |
| host mock 库 | 见 `final-hashes.txt` |
| 设备 before（§27 产物） | `cec3fa06f4739741c5983709858135fdd41e2ee2cce0610ad48995303cd430f8` |
| 设备 after（本轮） | `088a801708f8770e8110c6d08b055ded7df8d268bb65970dda4073c98990064e` |

```bash
cd research/tilelang-rknpu
# 三类反例（修复后：两层拒绝、mock 不执行）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T080000Z-mmacc-ew-lut-contract/repros/repro_template_contracts.py
# 新增契约测试
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_task_contracts.py -q -o addopts=""
# 全量 + C
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
cd ../custom-ops/build-host-mvp && cmake --build . --target rknpu_plan rknpu_plan_mock -j4 && ctest -j2
```

证据目录：`results/20261102T080000Z-mmacc-ew-lut-contract/`（快照、`repros/before-fix.txt`、`repros/after-fix.txt`、`board/{plans,probe,before.json,after.json}`、`logs/full-pytest.log`、`board/upload_and_run.py`）。

### 28.9 尚未完成

- View / RKIR 轴与属性、producer、预算溢出等问题按用户指示**未动**。
- LINEAR 布局 EW 操作数在现有布局常量下不可表达，本轮只补了检查、没有构造可执行的线性反例（见 28.3 的边界说明）。
- SwiGLU 的声明参考偏差仍未定位（与两库差异无关），沿用既有记录。

---

## 29. RKIR 入口：属性、轴含义、producer/role 与函数签名（2026-11-02）

### 29.1 四项逐项确认

| 项 | 原判断 | 结论 |
|---|---|---|
| A MatMul 属性 | `transpose_a=True` 仍可编译 | **确认**：属性被**静默忽略**。pre-fix 下 `transpose_a` 与 baseline 出**同一份计划**（`plan_hash` 都是 `451a4cb6…`，`plans_identical: true`），mock 输出匹配 **A@B**（`max_abs` 3.1e-3）而**不匹配** A.T@B（`max_abs` 16.95），且这两个参考本身不同（`a_times_b_differs_from_aT_times_b: true`，A 取 16×16 非对称方阵） |
| B `TensorType.axes` | 名称还是轴顺序？ | **是标签，不是顺序**：全仓库没有任何 lowering/计划步骤读 `axes` 决定位置（`rows`/`cols` 一律取 shape 位置）。但有**两处不一致**：① `hybrid.py` 已要求 axes 唯一，`ir.py` 没有；② `axes=("col","row")`（把默认名换序）被接受，读者会误以为发生了转置 |
| C `Value.producer` / `role` / `Op.outputs` | 双向关系是否一致 | **有缺口**：`role=input` 且 `producer=0` 能被接受（只因“消费者早于生产者”这条更早触发，诊断还指向错的矛盾）；某值声明的 producer 若“恰好被别的值补上 op id”就**不会被发现**（消费者会读一个从未写过的值）；`op.outputs` 没有重复检查 |
| D `Module.inputs` 与 binding | `role=input` 未列入 inputs 是否仍成为外部输入 | **确认**：`dag._create_bindings` 对**所有** `role in {input, constant, output}` 建 binding，完全不看 `Module.inputs`；`Module` 只检查了反方向（列出的必须是 input）。于是一个未声明的 input 会静默变成函数参数（本轮反例里它因为没有上传动作才在计划阶段暴露） |

### 29.2 最小复现与最早失败位置

| 反例 | 修复前最早失败位置 | 修复后拒绝位置 |
|---|---|---|
| `transpose_a=True` | 无（一路编译，计划与 baseline 完全相同） | `Op.__post_init__`（RKIR 构造） |
| `k_pad` / 未知键 | 无（静默丢弃） | `Op.__post_init__` |
| `axes=("col","row")`（rank-2） | 无 | `TensorType.__post_init__` |
| `axes=("row","row")` | 无 | `TensorType.__post_init__` |
| rank-1 bias `axes=("row",)` | 无 | `TensorType.__post_init__` |
| `role=input` 且 `producer=0` | 误报 “op 0 consumes value 0 before its producer” | `Module.__post_init__`（准确的 producer/role 矛盾） |
| 伪造 producer（op 没列出该值） | 若别的值补齐 op id 则**通过** | `Module.__post_init__` |
| 未声明的 `role=input`（且被消费） | **通过**（多出一个外部参数） | `Module.__post_init__` |
| 声明但未使用的 input | 计划校验期 “has no upload action” | 同处，但诊断补上原因（见 29.3 的显式规则） |

复现脚本：`results/20261102T120000Z-rkir-entry-contract/repros/repro_rkir_entry.py`（`before-fix.txt` / `after-fix.txt`）、`repros/transpose_a_evidence.py`（pre-fix 包与修复后包各跑一次，输出 `transpose-a-evidence.json`）。

### 29.3 最终规则

**A · MatMul 属性**（`ir.MATMUL_ATTRS`）

| 类别 | 键 | 处理 |
|---|---|---|
| 影响计算 | `b_layout`, `numeric_contract` | 显式校验（`kn`/`nk`；contract 必填且按 `NumericContract` 解析） |
| 来源/几何/诊断（不改变发射的数学） | `source_accumulator_dtype`, `source_narrowing_points`, `source_k_rounding`, `source_k_loop`, `source_mn_tiling`, `source_tail_contract`, `source_schedule`, `source_schedule_origin`, `source_stage_index/count/boundary/stage_schedule_contract`, `source_k_index/count/k0`, `source_geometry`, `source_loop_annotations/span`, `source_rounding`, `source_ops`, `tile_m/n/k` | 允许（已登记；`tile_*` 只作来源提示，lowering 不读） |
| 其它 | 任何未知键 | **拒绝**，`actual` 列出键名；`transpose_a/b`、`a_layout`、`transposed_a/b` 给出专门的“不支持操作数转置”诊断 |

**B · `axes`**：是**标签**，不是轴顺序。规则：① 数量等于 rank；② 唯一；③ 一旦使用默认名，就必须是当前实现的顺序（rank-2 `("row","col")`、rank-1 `("col",)`、rank-0 `()`）；自定义名（如 `("token","depth")`）自由。换序/半改名会被拒并说明“rows/cols 只来自 shape 位置，换标签不会转置”。TileLang 的 canonical/swapped **源 block** 轴交换不受影响——那是源侧 schedule 的 block 轴映射，与 RKIR 张量轴含义是两件事，本轮未触碰。

**C · producer / role**：`input`/`constant` 不得声称 producer；`temporary`/`output` 必须有 producer，且该 op 的 `outputs` 必须列出它；producer 必须存在；一个 op 的 `outputs` 不得重复；消费者仍必须在生产者之后。`state` 继续不支持。

**D · 函数签名**：`role=input` 的值**必须**在 `Module.inputs` 里（缺失即拒绝：外部输入必须出现在签名里）；`constant` 是外部权重（不是参数），`temporary` 是设备内部值，`output` 是发布目标。**声明但未使用**的 input 明确保留为“合法 IR”，但在计划校验期被拒（`input binding 'X' has no upload action: a declared input has to be consumed by at least one upload to become a plan parameter`）——这是当前执行器只能绑定“被消费的输入”这一限制的**准确说明**，不是把两种矛盾混为一谈。

### 29.4 正反例与执行对照

- 合法：`b_layout` 两种、全部已登记来源元数据、canonical 轴、rank-1 `("col",)`、rank-0、`matmul_module`（classic）与 `swiglu_module`（dataflow）都继续编译，逻辑说明与数值要求照常存在。
- 执行对照（mock，非对称 16×16 A）：合法计划输出等于 A@B（`max_abs` 3.1e-3），不等于 A.T@B（`max_abs` 16.95）；`transpose_a` 的计划已不存在（构造期拒绝）。
- 新增测试 `tests/test_rkir_entry_contract.py`（19 例 + 11 子测试）覆盖属性、轴、producer/role、签名、执行对照与保存/加载/重验证；`tests/test_dataflow.py` 中一条旧负例的期望按新诊断（同一矛盾、更早位置、代码 `RK_E_SHAPE`）更新，已在该测试注释里写明原因。

### 29.5 回归、身份与重放

- 全量 Python：**655 passed / 1 failed / 1 skipped / 180 subtests**；唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`（环境问题，与本轮无关）。
- **C 层本轮未改**（没有 C 源改动，也没有重建库）。三份板测计划（§28 的 `merged-multi-k`、`swiglu-ffn`、`relu-ew`）在修复后重新编译仍**逐字节相同**（`to_dict()` 相等，hash `02010768…` / `b5108df7…` / `89460b59…`），因此 §28 的设备证据（库 `cec3fa06…`→`088a8017…`、输出逐位相同、计数一致）**逐项适用于本轮的正常路径**，本轮不再重复上板。
- 异常 RKIR 未送设备（全部在构造/编译期拒绝）。

```bash
cd research/tilelang-rknpu
# 四类反例（修复后全部在构造/编译期拒绝）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T120000Z-rkir-entry-contract/repros/repro_rkir_entry.py
# transpose_a：pre-fix 与修复后各跑一次
PYTHONPATH=/tmp/prefix_pkg:$PWD .venv/bin/python results/20261102T120000Z-rkir-entry-contract/repros/transpose_a_evidence.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T120000Z-rkir-entry-contract/repros/transpose_a_evidence.py
# 新增测试与全量
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_rkir_entry_contract.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
```

代码身份：`python/rknpu_backend/ir.py`（本轮唯一生产改动）、`tests/test_rkir_entry_contract.py`（新增）、`tests/test_dataflow.py`（一条负例期望更新）；C 源与库与 §28 相同（`rk_plan_validate.c` `c36d14d4…`，host 库 `ac0856e8…` / mock `9b307952…`）。

### 29.6 尚未完成

- 预算溢出、View 边界、其它新问题按指示**未动**。
- `ELEMENTWISE_SCHEDULE_META` 与 `MATMUL_SOURCE_META` 目前各自列举、部分重叠；后续可合并成一个“来源元数据”清单（本轮保持局部、避免大改）。
- `native_graph` 现在必须只带允许键：外部 JSON 若带未知属性会被拒绝（这是期望行为，但那些历史 JSON 未逐一回扫）。
- `source_schedule` 的**值** schema 仍在 legalize 校验，本轮只保证键被识别。
- 声明但未使用的 input 仍无法成为计划参数（执行器限制），只把它变成准确诊断。

### 29.7 更正：默认轴名与自定义轴名可以混用（2026-11-02 小修正）

**原规则过宽。** §29.3 写的是“一旦使用默认名就必须整组等于默认顺序”，于是 `("row","feature")`、`("token","col")` 被拒，而它们并没有声称交换方向，只是把**一个**标签改成了自定义名。

**现在的规则**（只改这一处）：

| 保留 | 位置约定 |
|---|---|
| 数量 = rank；非空字符串；同一张量内唯一；标签不改变 shape 位置的计算含义 | `row` 只允许 rank-2 位置 0；`col` 只允许 rank-2 位置 1 或 rank-1 位置 0；其它位置一律可以是自定义名（可与默认名混用）；rank-1 的 `row` 没有约定位置，继续拒绝；rank-0 必须空 |

没有通过交换 shape、修改 stride 或增加转置来解释名称，也没有大小写归一化等额外行为。

**实测表**（`results/20261102T140000Z-axes-label-mix/repros/axis-labels.txt`）：

| axes | shape | 结果 |
|---|---|---|
| `("row","col")` / `("token","feature")` | 16×8 | 接受 |
| `("row","feature")` / `("token","col")` | 16×8 | **接受（本轮新增）** |
| `("col","row")` / `("col","feature")` | 16×8 | 拒绝：col 不在 rank-2 位置 1 |
| `("feature","row")` | 16×8 | 拒绝：row 不在 rank-2 位置 0 |
| `("row","row")` | 16×8 | 拒绝：标签重复 |
| `("col",)` / `("feature",)` | 8 | 接受 |
| `("row",)` | 8 | 拒绝：row 没有 rank-1 约定 |
| `()`，以及 `("scalar",)` 对 rank-0 | () | 前者接受，后者拒绝 |
| `("row",)`、`("row","col","third")`、`("row","")` 对 rank-2 | 16×8 | 拒绝：数量/空名 |

**同一计算、不同标签 → 同一份计划**：同一 MatMul 分别用默认标签、全自定义标签、混合标签编译，`plan_hash` 与 `to_dict()`（bindings/views/actions/submits/publish/accounting）**完全相同**；rank-1 广播常量用 `("col",)` 与用 `("channel",)` 也完全相同（测试 `LabelInvarianceTests`）。

**一个如实登记的观察（不在本轮范围）**：`TensorType` 的相等比较包含 `axes`，所以“操作数/结果类型必须相同”的检查（例如 binary 广播的 `tensor_side.type != out.type`）会把**不同标签当成不同类型**。同一程序里操作数与结果标签一致时没有影响；本轮按要求只改名称检查，未改这个相等语义，已记为后续项。

**回归**：全量 **660 passed / 1 failed / 1 skipped / 194 subtests**（唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`）；P0（`test_submit_order_contract`）、模板（`test_task_contracts`）、RKIR 入口（`test_rkir_entry_contract`）全部通过；三份板测计划（§28）重新编译后**逐字节相同**（`02010768…` / `b5108df7…` / `89460b59…`），因此 §28 的设备证据继续适用，本轮未重跑板测。

---

## 30. 轴标签不再参与“能否一起算”的判断（2026-11-02）

### 30.1 以前会误拒什么

同一个 16×8 FP16 `add`／`mul`／`relu`，只要把**任意一个操作数或结果**的轴标签从 `("row","col")` 改成 `("token","feature")`，就会被拒绝：

| 情形 | 修复前 | 原因（代码） |
|---|---|---|
| add：一个输入改标签 | REJECT | `ir._infer_and_verify` 用 `a.type == b.type == out.type` 判断能否同形状计算 |
| add：另一个输入改标签 | REJECT | 同上 |
| add：仅结果改标签 | REJECT | 同上 |
| mul：仅结果改标签 | REJECT | 同上 |
| relu：输入或结果改标签 | REJECT | unary 分支 `x.type != out.type` |
| rank-1 广播：张量侧或结果改标签 | REJECT | 广播分支 `tensor_side.type != out.type` |
| 同形状但标签全不同（三者一致） | 接受 | 三者相等，比较通过 |

根因：`TensorType` 是 frozen dataclass，`==` 逐字段比较，**包含 `axes`**；而上表这些位置把 `==`／`!=` 当作“这两个张量能否一起计算”来用。标签只是 name，不改变 shape 位置、排列或计算方向。

### 30.2 现在怎样判断能否计算

新增局部助手 `ir.same_compute_type(a, b)`，只回答“能否喂同一个逐元素计算”：比较 **dtype、shape、有效 stride、index_map、finite**，**排除 axes**。有效 stride 把“省略 stride”规范成该 rank 的稠密形式（rank-2 `(cols,1)`、rank-0/1 全 1），所以省略与显式写出同一种存储不算差异，非稠密 stride 仍然不同。

改动位置（逐处判断，非机械替换）：

| 位置 | 用途 | 处理 |
|---|---|---|
| `ir.py` unary 分支 | 输入/输出必须同形同 dtype | 改用 `same_compute_type` |
| `ir.py` binary 同形状分支 | 两个操作数与结果必须同形 | 改用 `same_compute_type` |
| `ir.py` binary 广播分支 | 张量侧必须等于结果 | 改用 `same_compute_type` |
| `legalize.py:212` | add/mul 与“已产生值”同形检查 | 改用 `same_compute_type` |
| `legalize.py:438` | dataflow binary 同形状 | 改用 `same_compute_type` |
| `legalize.py:451` | dataflow 广播张量侧 | 改用 `same_compute_type` |
| `TensorType.__eq__`（dataclass 默认） | 身份、`plan_hash`、序列化、缓存键 | **保留**（精确描述就该含标签） |
| view / matmul 的几何检查 | rank、dtype、shape、region 边界 | **保留**（本来就是逐字段） |

`TensorType` 的完整相等没有被改，标签仍写进计划与序列化；没有新增转置、轴排列、动态广播或新算子。

### 30.3 正反例

正例（`tests/test_label_neutral_compat.py`，12 例 + 32 子测试）：

- add／mul：一个输入改标签、另一个输入改标签、仅结果改标签、三者混合自定义；
- unary：`relu`、`sigmoid` 的输入／结果／两者改标签；
- 广播：rank-1 常量（张量侧改、结果改、常量自身改 `("bias",)`）、rank-0 标量；
- 每一例都与默认标签版本编译出**同一份计划**（`plan_hash` 与 `to_dict()` 全等）。

负例仍拒绝：真正的 shape 不匹配、`finite` 要求不同、非法广播（列向量长度 ≠ N）、标签数量/重复/`row`/`col` 位置错误、数值要求（未知 accumulation 策略、负 tolerance）。dtype、index_map 与非稠密 stride 在现有 `TensorType` 里无法构造，用同类字段替身直接单测助手，确认这三项仍被比较。

### 30.4 实际计划与输出对照

| 对照 | 结果 |
|---|---|
| 默认 vs 一个输入改标签（add） | `plan_hash` 相同 `4a7f1b6d…` |
| 默认 vs 结果改标签（mul） | 相同 `e1f5790c…` |
| 默认 vs 改标签（relu） | 相同 `abb4a962…` |
| 默认 vs 改标签（rank-1 广播） | 相同 `ce46f5e5…` |
| 默认 vs 改标签（rank-0 广播） | 相同 `5ddc9fe2…` |
| mock 执行（同输入）：默认 vs 改标签 | 输出 **逐位相同**，task/submit 计数相同 |
| 保存 → 加载 → 重验证 | hash 不变、字典不变 |

### 30.5 C / mock / 设备证据

- **mock**：`LabelNeutralExecutionTests` 对 add／mul／relu／广播各跑默认与改标签两版，输出 sha256 相同、task/submit 相同、计划字典相同。
- **真实 C 校验器**：`test_real_c_verifier_accepts_relabelled_plans` 对改标签的 add／relu／广播计划调用 `rk_plan_create`，全部接受（C 本轮未改动）。
- **设备**：标签从不进入计划，§28 的三份板测计划（`merged-multi-k`、`swiglu-ffn`、`relu-ew`）重新编译后**逐字节相同**（`02010768…`／`b5108df7…`／`89460b59…`），所以 §28 的真机证据（库 `cec3fa06…`→`088a8017…`，输出逐位一致）逐项适用于本轮；本轮未重复上板，也未做性能测量。

### 30.6 回归与身份

- 全量 Python：**672 passed / 1 failed / 1 skipped / 226 subtests**；唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`（环境问题）。
- P0（`test_submit_order_contract`）、模板（`test_task_contracts`）、RKIR 入口（`test_rkir_entry_contract`）与 dataflow／广播／frontend／融合测试全部通过。
- 代码身份：`python/rknpu_backend/ir.py`、`python/rknpu_backend/passes/legalize.py`、新增 `tests/test_label_neutral_compat.py`；C 源与库不变（`rk_plan_validate.c` `c36d14d4…`，host `ac0856e8…`，mock `9b307952…`）。

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T160000Z-label-neutral-compat/repros/repro_label_compat.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_label_neutral_compat.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
```

### 30.7 尚未完成

- 预算溢出、View 边界等其它问题按指示未动。
- dtype 与非稠密 stride 目前在 IR 层不可表达（`SUPPORTED_DTYPES={float16}`，rank-2 stride 只允许稠密），助手对它们的比较靠字段替身单测覆盖；等这些能力真正开放时需要再确认调用点。
- `TensorType.__eq__` 仍含标签（有意保留）；若将来有新的“能否计算”比较，应按 `same_compute_type` 写，这一点已写进助手文档。

---

## 31. View 有效区域边界与设备预算累加（2026-11-02）

### 31.1 View：原来没有检查“起点＋长度”

`View.valid_region = (r0, nr, c0, nc)`（起始行、行数、起始列、列数）。修复前 Python 只检查：

```python
if min(r0, nr, c0, nc) < 0 or nr > self.tensor.rows or nc > self.tensor.cols:  # 少了 r0+nr / c0+nc
```

所以**长度各自合格、起点不为 0 时就会越界**。用户反例 `valid_region=(10,10,0,16)` 用在 16×16 逻辑 tensor 上：`nr=10 ≤ 16`、`r0=10 ≤ 16` 都过，但结束行 20 已经出界。

**错误值会走到哪一步（修复前实测）**：

| 阶段 | 结果 |
|---|---|
| Python `View.__post_init__` / `_validate_plan` | **通过** |
| 保存 → 加载 | **通过**（JSON 里就是那个越界区域） |
| C `rk_plan_create` | **拒绝**：`view valid_region invalid`（C 侧 `valid_region[0]+valid_region[1] > rows` 等本来就有） |
| 执行 | 到不了 |

也就是说：错误值能一路通过构造、序列化和加载，**最早的拒绝在 C 校验器**；Python 没有给出诊断。

**现在为什么能提前拒绝**（`plan.py::View.__post_init__`，三条依次）：

1. 四个分量非负（`view valid_region entries must be non-negative`）；
2. 起点本身不越界（`view valid_region starts outside the logical tensor`，`r0 > rows` 或 `c0 > cols`）；
3. 起点＋长度不越界（`view valid_region is outside the logical tensor`，`r0+nr > rows` 或 `c0+nc > cols`）。

逻辑 tensor 边界与 buffer 字节边界**分开**：字节边界仍是 `_validate_plan` 的 `view N exceeds its logical buffer`（本轮未改，测试里单独覆盖）。零长度区域的规定原样保留（`(16,0,0,0)`、`(0,0,0,0)` 仍合法），合法区域的读取、padding 与发布规则未动。

修复前后的实测（`repros/before-fix.txt` / `after-fix.txt`）：

| region（16×16） | 修复前 Python | 修复前 C | 修复后 |
|---|---|---|---|
| `(10,10,0,16)` 用户反例 | 接受 | 拒绝 | **Python 拒绝（第 3 条）** |
| `(10,7,0,16)` 仅行越界 | 接受 | 拒绝 | **Python 拒绝（第 3 条）** |
| `(0,16,10,8)` 仅列越界 | 接受 | 拒绝 | **Python 拒绝（第 3 条）** |
| `(17,1,0,8)` 起点越界 | 接受 | 拒绝 | **Python 拒绝（第 2 条）** |
| `(6,10,0,16)` 恰好到边界 | 接受 | 接受 | 仍接受 |
| `(2,4,3,5)` 合法内部 | 接受 | 接受 | 仍接受 |
| `(16,0,0,0)` 零长度 | 接受 | 接受 | 仍接受 |

### 31.2 设备预算：累加回绕不可靠

原来：

```c
reserved += slot_cap[i] + CO_SLACK + CO_GUARD;
if (reserved < slot_cap[i]) reject("reserved capacity overflow");   /* 事后比较，看不到回绕 */
```

回绕后的 `reserved` 只要**不小于刚加的那个 slot 容量**，比较就通过。构造：`slot_cap = [2^63, 2^63 - RESERVE/2 - 2·RESERVE, 4096]`（`RESERVE = CO_SLACK+CO_GUARD`），前两个 slot 把累计推到 `2^64 - RESERVE/2`，再加 4096 的小 slot 回绕成 ≈20 KiB，而它 ≥ 4096 → **旧检查放过**。实测修复前该描述符被 C **接受**（`wrapped=20512`）。

**现在为什么能拒绝**（`rk_plan_validate.c`，加法前检查，共三处）：

1. `b->capacity > UINT64_MAX - b->offset` → `buffer range overflow`（这是后面所有合计的加数）；
2. `slot_cap[i] > UINT64_MAX - CO_SLACK - CO_GUARD` → `slot capacity overflow`（原有，保留）；
3. `reserved > UINT64_MAX - amount` → `reserved capacity overflow`（**新增**，只在检查通过后才 `reserved += amount`）。

后续加法（`reserved > UINT64_MAX - command`）本来就是加法前检查，保留；没有降低最大预算、没有限制 buffer 容量、没有放宽任何校验。

| 描述符（slot 容量） | 修复前 C | 修复后 C |
|---|---|---|
| `[4096, 8192, 16384]` 正常 | 接受 | 接受 |
| `[UINT64_MAX-RESERVE, 8192, 16384]` 填到天花板 | **接受**（回绕） | `reserved capacity overflow` |
| `[UINT64_MAX-RESERVE+1, …]` 单个 amount 就溢出 | `slot capacity overflow` | 同（保留） |
| `[2^63, 2^63-RESERVE/2-2·RESERVE, 4096]` 回绕后仍大于末 slot | **接受** | `reserved capacity overflow` |
| `[2^63, 2^63, 4096]` 回绕发生在大 slot | `reserved capacity overflow`（侥幸命中） | 同 |

**可达性如实说明**：通过公开 API 生成的计划到不了这个算术——`storage` 阶段用**精确整数**算总需求并与 profile 预算比较，先报 `device budget exceeded`（测试 `test_the_python_layer_rejects_a_huge_request_even_earlier` 固定了这一点）。所以这些反例只在**描述符层**可达（例如外部生成器或未来 Python 回归），测试里显式绕过 Python 校验直接 `rk_plan_create`，且**只调用计划创建/验证**：不建 workspace、不分配对应设备内存、不送设备。

### 31.3 正常计划有没有变化

没有。判据不是“看起来一样”，而是：

- 三份代表计划（`merged-multi-k`、`swiglu-ffn`、`relu-ew`）重新编译后 `to_dict()` **逐字节相同**；
- 板测：用本轮源码重建设备库（`rk_plan_validate.c` `f202246a…` → 库 `3749e95d…`），对同一批计划文件、同一批输入各跑 3 轮，与修复前库（`088a8017…`）对照：

| 计划 | task/submit/upload/download | 输出 |
|---|---|---|
| merged-multi-k | 36 / 1 / 3 / 12 | **逐位相同** |
| swiglu-ffn | 7 / 5 / 1 / 1 | **逐位相同** |
| relu-ew | 4 / 4 / 1 / 2 | **逐位相同** |

（SwiGLU 的声明参考 `max_abs` 24 是既有记录，两库完全相同，§24.7/§18.4。）

### 31.4 回归与身份

- 全量 Python：**684 passed / 1 failed / 1 skipped / 243 subtests**；唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`。
- C：重建后 `ctest` **33/33 passed**。
- 新增测试 `tests/test_view_region_budget.py`（12 例 + 17 子测试）覆盖：用户反例、仅行/仅列越界、起点越界、负数分量、恰好边界、内部区域、零长度、保存/加载、字节边界仍独立、C 对同一越界区域仍拒绝（绕过 Python 校验）、预算的五个边界与 C 独立拒绝、以及“Python 更早拒绝大预算”的可达性说明。
- 代码身份：`python/rknpu_backend/plan.py`、`research/custom-ops/src/rk_plan_validate.c`、新增 `tests/test_view_region_budget.py`；host 库 `librknpu_plan.so` / `librknpu_plan_mock.so` 重建（哈希见 `final-hashes.txt`），设备库 `3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c`。

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T180000Z-view-region-budget-overflow/repros/repro_view_budget.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_view_region_budget.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
cd ../custom-ops/build-host-mvp && cmake --build . --target rknpu_plan rknpu_plan_mock -j4 && ctest -j2
```

### 31.5 尚未完成

- 预算溢出的**其它**算术（例如 `slot_cap` 分配大小、`slots` 计数与 `data_bo_count` 的交叉核对、host staging 的累加）本轮未逐一重审，只处理了任务指定的这条链。
- 零长度区域的语义仍按既有规定（未新增禁止或支持）；如果以后要区分“空区域”与“未使用区域”，需要另开一轮。
- mock 侧不做极大容量的执行测试（会真的分配内存），本轮只是计划创建级验证。

---

## 32. 计划动作种类：Python 允许写入、C 不能执行的那几步（2026-11-02）

### 32.1 线索核实结果

线索成立，但需要分清“谁产生、谁执行、什么时候合法”：

| 种类 | 谁产生 | 是否出现在最终计划 | C 侧 |
|---|---|---|---|
| `upload` / `npu_task` / `download` | 规划器（`dag.add_action`、`repeat_expand`） | 是 | 执行（`upload_action` / 提交 / download） |
| `host_pack` / `host_expand` / `host_check` | **没有产生者**；只在 upload 的 `origin` 字符串里作来源标签（`host_pack:dataflow:…`） | 否 | 校验器拒绝 `unsupported action kind` |
| `commit` | **没有产生者** | 否 | 同上 |
| `host_compute` | **没有产生者**；`passes/verify.py` 还专门有一条“禁止隐藏的 host 算术” | 否 | `hidden host compute is forbidden` |

补充两点：`hybrid.py` 里的 `HostAction` 是 **Hybrid Host 程序注册表**里的条目（名字 + 版本），不是计划动作种类，与本主题无关；C 头文件的 `enum rk_action_kind` 确实包含 8 个值，但校验器的分派只处理 `UPLOAD/DOWNLOAD/NPU_TASK`，`HOST_COMPUTE` 单独拒绝，其余落到 `unsupported action kind`；运行时末尾还有一条防御性拒绝（`unsupported action in the verified schedule`）。

所以真正的问题是：**`Plan` 的校验只要求“种类在枚举里”，没有要求“种类能被执行”**。临时步骤留在枚举里是对的（draft 需要表达），但一份交给运行时的成品计划不该包含它们。

### 32.2 复现（修复前实测）

在正常计划末尾追加一个该种类的空动作（`Action(id=N, kind=..., template_id=None, inputs=(), outputs=(), deps=(0,))`），重算 `plan_hash`，分别走四条路：

| 种类 | Python `_validate_plan` | 保存→加载 | C `rk_plan_create` |
|---|---|---|---|
| `host_pack` | **ACCEPT** | **ACCEPT** | REJECT `unsupported action kind` |
| `host_expand` | **ACCEPT** | **ACCEPT** | REJECT `unsupported action kind` |
| `host_check` | **ACCEPT** | **ACCEPT** | REJECT `unsupported action kind` |
| `host_compute` | **ACCEPT** | **ACCEPT** | REJECT `hidden host compute is forbidden` |
| `commit` | **ACCEPT** | **ACCEPT** | REJECT `unsupported action kind` |
| 正常计划（对照） | ACCEPT | ACCEPT | ACCEPT |

也就是说：Python 接受、能写盘、能再读回，**必然被 C 拒绝**；mock 也在 `create_plan` 就停下（没开始执行）。这是“Python 允许写入、C 不支持”的实例，不是枚举名字的差别。

### 32.3 修复：成品计划只允许可执行种类

- `plan.py` 新增 `EXECUTABLE_ACTION_KINDS = {"upload", "npu_task", "download"}`，紧挨着 `ACTION_KINDS` 放，方便对照；`ACTION_KINDS` **没有删任何值**，临时步骤仍可构造（`Action(kind="commit", …)` 仍然能建，测试里固定了这一点）。
- `_validate_plan`（构造与加载都会经过）在动作循环前新增一条：非可执行种类 → 拒绝。`host_compute` 保留原有专门措辞（`MVP forbids hidden host arithmetic actions`），其余给出“这是临时规划步骤、不是可执行动作”，并带上 `actual=<种类>` 与 `required=[download, npu_task, upload]`。
- 没有静默删除：任何携带这些步骤的计划都会**明确报错**，错误信息点名种类和允许集合。
- `passes/verify.py` 的 host_compute 检查保留（现已被更早的同义检查覆盖，作为纵深防御）。

修复后同一批复现：

| 种类 | Python | 加载 | C |
|---|---|---|---|
| `host_pack`/`host_expand`/`host_check`/`commit` | REJECT `plan action kind '…' is a transient planning step, not an executable action` | REJECT（同一诊断） | REJECT（不变） |
| `host_compute` | REJECT `MVP forbids hidden host arithmetic actions` | REJECT | REJECT（不变） |
| 正常计划 | ACCEPT | ACCEPT | ACCEPT |

### 32.4 合法路径没有受影响

- 规划器产出的种类集合实测只有三种（`{upload, npu_task, download}`），`matmul_module` 与 `swiglu_module` 都如此；新增测试对两者逐一断言。
- 三份代表计划（`merged-multi-k`、`swiglu-ffn`、`relu-ew`）重新编译后与 §28 存的计划文件 **`to_dict()` 逐字节相同**（`02010768…` / `b5108df7…` / `89460b59…`）。
- **C 执行代码本轮没有改动**（`rk_plan_validate.c` 仍是 §31 的 `f202246a…`，库仍是 `d527fef0…` / `c9e0d22e…`，设备库 `3749e95d…`）。因此按用户规则**不重复板测**：§31 的真机证据（同批计划、同批输入，输出逐位相同、计数一致）逐项适用——本轮只增加了“成品计划不得含临时步骤”这一条更早的拒绝。

### 32.5 与既有测试的一处冲突（已查明并修正）

P0 的 `test_submit_order_contract.py::test_a_host_action_cannot_be_hidden_inside_a_run` 原本用 `kind="commit"` 制造“提交段内夹 host 动作”。本轮修复后 `commit` 在计划校验就被拒绝，该测试会因**更早的拒绝**而失败——不是提交顺序规则失效。

处理：改用**真实的 `upload`** 充当段内 host 动作（复制计划原有 upload 的目标视图与 attrs），提交顺序检查照旧命中 `a host action is interleaved inside a task group`。测试注释写明了替换原因。两个文件合计 23 例 + 17 子测试全绿。

### 32.6 回归与身份

- 全量 Python：**694 passed / 1 failed / 1 skipped / 255 subtests**；唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`。
- C：`ctest` **33/33 passed**（C 源未改）。
- 新增 `tests/test_action_kind_contract.py`（10 例 + 12 子测试）：枚举集合关系、临时动作仍可构造、规划器只产出可执行种类、五种临时种类在计划/加载/C/mock 四条路径上的拒绝、诊断字段（`actual`/`required`）、正常计划保存加载身份不变。
- 代码身份：`python/rknpu_backend/plan.py`、`tests/test_action_kind_contract.py`（新增）、`tests/test_submit_order_contract.py`（一处测试适配）；C 源与库不变。

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T200000Z-action-kind-contract/repros/repro_action_kinds.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_action_kind_contract.py tests/test_submit_order_contract.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
cd ../custom-ops/build-host-mvp && ctest -j2
```

### 32.7 尚未完成

- `Action.__post_init__` 里有一处与 `ACTION_KINDS` 重复的种类检查（历史遗留、无副作用），本轮未删以免顺带改行为。
- 若将来真的想支持 `commit`（例如显式提交点），需要在运行时实现它，然后把它移进 `EXECUTABLE_ACTION_KINDS`；本轮不做。
- `passes/verify.py` 的 host_compute 检查现在与计划校验重复，保留作防御，未合并。

---

## 33. TileLang 的调度要求，实际落到 RKNPU 的有哪些（2026-11-02）

### 33.0 直接回答

**1）TileLang 目前能实际控制什么？** 四项，且都有设备证据：**M/N 分块**（`rknpu.tile_m/tile_n` → 任务网格）、**遍历顺序**（`rknpu.mn_order` → 发射的任务/提交顺序）、**K 分块**（源 K 循环的 256 宽块 → 每个 tile 的累积任务数）、**K 舍入点**（`rknpu.source_k_rounding: explicit_per_k_fp16` → 每 256 宽 K 块写回 FP16，数值可测地不同）。

**2）哪些只能通过后端编译选项？** 两个：**输入复用**（只有 `linear_task_order="interleaved_reuse"` 会打开 `data_reuse`；源里没有对应注解）和**提交分组**（`linear_task_order=baseline|merged` → 12 提交 vs 1 提交；`submit_mode` chain/separate）。

**3）哪些要求会被明确拒绝？** 声明与实现不符的 K 舍入（`fp32_final_narrowing`）、未定义的 `rknpu.*` 注解、非法取值（`mn_order` 非 m/n-major、tile 超出 1..256、tile_m/tile_n 只给一个），以及此前各轮已修的那些（mm_acc K>256、EW 非 CUBE、LUT 几何不符、临时动作种类、View 越界、提交顺序矛盾等）。

**4）还有“接受了要求、实际却没执行”的情况吗？** 五个被接受的注解逐个都有计划/寄存器/数值证据，**没有发现**。本轮修掉了一个真实的静默忽略：`rknpu.pipeline_stages`、`rknpu.double_buffer`、`rknpu.async_copy`、`rknpu.tile_num` 这类未定义键以前被无声丢弃（原来的前缀白名单只扫 schedule/tile_/mn_/k_/source_），现在一律拒绝并报出键名。**未实现**但也不假装支持的是：`weight_reuse`（C 与发射器支持，规划器恒为 False，没有入口）、流水/双缓冲（源的 `T.Pipelined(num_stages=2)` 不进入计划）、多核。

**5）本轮补通什么、还缺什么证据？** 补：`rknpu.*` 命名空间的准确拒绝；复用位在**寄存器字**上的读取；顺序/分块/复用/K 舍入的设备对照。缺：`weight_reuse` 无入口可测；流水/双缓冲/多核未实现；本轮不做性能结论（见 33.5）。

### 33.1 清点：从源到硬件的每一段

| 控制项 | 用户在哪里表达 | 后端承诺 | 改变什么 | 接受/拒绝 | 证据 |
|---|---|---|---|---|---|
| M/N 分块 | 源 `rknpu.tile_m/tile_n`（仅与 `rknpu.schedule="mn_tiles_v1"` 合用） | 任务网格 = ⌈M/tile_m⌉×⌈N/tile_n⌉，尾部 tile 带守卫 | 任务数、提交数、设备预算 | 1..256 接受；越界拒绝；只给一个拒绝 | 主机：bm=256 → 6 任务/2 提交，bm=128 → 12 任务/4 提交；设备两版输出**逐位相同** |
| 遍历顺序 | 源 `rknpu.mn_order=m_major|n_major` | 发射顺序按声明顺序访问 tile | 任务/提交顺序（不改任务集合） | 其它值拒绝 | 主机：m_major → `mm:0:0,mm:0:1,mm:1:0,mm:1:1`，n_major → `mm:0:0,mm:1:0,mm:0:1,mm:1:1`；设备两版输出**逐位相同** |
| K 分块 | 源 K 循环的 256 宽块（`T.Pipelined(T.ceildiv(k,256))`）+ `rknpu.source_k_rounding` | 每 256 宽 K 块一个累积任务，每块写回 FP16 | 累积任务数、舍入次数、数值 | K>256 的单任务被拒（mm_acc 契约） | 主机：k=576 → 6 任务（2 tile × 3 K），k=256 → 2 任务 |
| K 舍入点 | 源 `rknpu.source_k_rounding=explicit_per_k_fp16` | 每 K 块 FP16 写回，末尾不再有一次整体窄化 | 数值（不是任务数） | 声明 `fp32_final_narrowing` → **拒绝**（与证明的舍入点不符） | 设备 k=576：与 fp16-逐块参考仅差 45/131072 个元素，与单次 fp64 积累差 50028；k=256（单块）：与两者各差 43/131072 |
| 输入复用 | **仅**编译选项 `linear_task_order=interleaved_reuse` | 第二个及以后的 tile 复用同一输入 tile | 发射器 `data_reuse=1` → CBUF_CON0 置位 | 只与该策略一起出现；C 校验复用前驱几何/视图一致 | 主机：`interleaved` 0/6 个词带 0x1000，`interleaved_reuse` **3/6**；设备同一计划 `con0_reuse=3`，输出与不复用版**逐位相同** |
| 提交分组 | **仅**编译选项 `linear_task_order=baseline|merged`（+`submit_mode`） | 每次提交有多少任务 | 提交次数、命令工作区、设备预算 | 组合规则见 §24 | §24/§28 的设备证据 + 本轮计划计数复核（12 提交 vs 1 提交） |

未实现、也不声称支持：`weight_reuse`（`dag.py` 恒 False）、流水/双缓冲（`T.Pipelined` 的 stage 数不进入 RKIR/RKPlan）、多核（`core_mask` 由运行时设置）。这些在报告里列为“尚不支持”，不是“接受后忽略”。

### 33.2 寄存器级证据（不是描述符开关）

复用位最终写在 **CBUF_CON0 寄存器**里：vendor `npu_regcmd.c:214` 生成 `(weight_reuse?0x2000:0)|(data_reuse?0x1000:0)`，本仓库 `custom_ops.c` 把 `co_program.data_reuse` 交给同一个生成器，运行时 `rk_workspace_task_cbuf` 把真正发射出来的字返回（寄存器号 0x1040）。实测（mock，与设备同一发射器）：

| 策略 | 带 0x1000 的词 | 首个 CBUF_CON0 |
|---|---|---|
| `interleaved` | 0/6 | 0x84 |
| `interleaved_reuse` | **3/6** | 0x84（其余 3 个任务置位） |

设备侧同一计划用 `task_cbuf` 读回：`con0_reuse=3`，与主机一致。

### 33.3 设备对照（RK3588，库 `3749e95d…`，同一批输入）

| 成对实验 | 任务/提交 | 输出对比 |
|---|---|---|
| m_major vs n_major | 12/4 vs 12/4 | **逐位相同**（0/131072 不同） |
| tile 256 vs 128 | 6/2 vs 12/4 | **逐位相同** |
| 复用关 vs 开 | 6/1 vs 6/1（con0 复用词 0 vs 3） | **逐位相同** |
| K=576 vs K=256 | 6/2 vs 2/2 | **不同**：131041/131072 元素不同，max_abs 80.3 |

结论的正确读法：**分块、顺序、复用改变的是执行组织，不改数学**（所以逐位相同）；**K 分块改变任务数与累积次数，所以数字不同**。

> **本段的舍入结论已被 §34 更正（2026-11-02）**：K=576 与 K=256 是**不同的计算**，输出不同只能说明“不同 K 长度生成了相应任务”，不能用来证明舍入位置。固定 K=576 的舍入证据见 §34.3（同一计算、两种写回结构的参考）。

### 33.4 本轮修复：`rknpu.*` 命名空间不再静默忽略

修复前（HEAD 前的适配器）与修复后，同一组源注解：

| 注解 | 修复前 | 修复后 |
|---|---|---|
| `rknpu.pipeline_stages=3` | **ACCEPT（静默忽略）** | REJECT `unsupported TileLang schedule annotation 'rknpu.pipeline_stages'` |
| `rknpu.double_buffer=True` | **ACCEPT（静默忽略）** | REJECT（同上） |
| `rknpu.async_copy=1` | **ACCEPT（静默忽略）** | REJECT（同上） |
| `rknpu.tile_num=4` | REJECT（前缀命中） | REJECT（同一诊断） |
| 五个已声明注解 | ACCEPT | ACCEPT（不变） |

改动只在 `frontend/tilelang_adapter.py`：把“只扫描 schedule/tile_/mn_/k_/source_ 前缀”改为“整个 `rknpu.` 命名空间都属于后端调度命名空间”，命中未定义键就拒绝，并在诊断里带上键名与允许集合。没有新增能力，也没有放宽任何检查。

### 33.5 证据边界（不做过度结论）

- 本轮**不做性能结论**：没有交错计时，也不宣称复用/顺序/分块更快；它们“确实生效”由计划、寄存器字和计数证明。
- 顺序/分块/复用三组输出逐位相同，只说明**数值不变**，不能反推硬件时间顺序；`T.Pipelined(num_stages=2)` 这类静态结构本来就不承诺设备时间序，后端也未据此做任何声明。
- 复用只在 `interleaved_reuse` 下有定义；没有 TileLang 源入口，属于“只能由编译选项表达”的能力。
- `weight_reuse` 没有入口，本轮不构造“假计划改 JSON”的用例来假装验证。

### 33.6 回归与身份

- 全量 Python：**700 passed / 1 failed / 1 skipped / 261 subtests**；唯一失败仍是缺 `transformers` 的 `test_replay_smollm2_v5`。
- C：`ctest` **33/33 passed**（本轮 C 未改动）。
- 新增 `tests/test_schedule_namespace.py`（6 例 + 6 子测试）：五个注解仍接受、未定义 `rknpu.*` 拒绝、非法值拒绝、`mn_order` 改变发射的 tile 序列、tile 大小改变任务网格、复用位出现在 CBUF_CON0 且与计划里的声明数一致。
- 代码身份：`python/rknpu_backend/frontend/tilelang_adapter.py`、`tests/test_schedule_namespace.py`（新增）；C 源与库不变（`f202246a…`、`d527fef0…`/`c9e0d22e…`、设备 `3749e95d…`）。

```bash
cd research/tilelang-rknpu
# 主机：计划事实 + CBUF_CON0 寄存器字
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T220000Z-tilelang-control-fidelity/repros/experiments.py --stage host
# 命名空间前后对照
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T220000Z-tilelang-control-fidelity/repros/probe_unknown_schedule_attrs.py
# 设备（脚本见同目录 board_run.py，上传后用 --stage device）
# PYTHONPATH=... python3 experiments.py --stage device
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_schedule_namespace.py -q -o addopts=""
```

### 33.7 尚未完成

- `weight_reuse`：C 校验、发射器、寄存器位都在，但规划器没有入口；若要支持需要一条明确的源或选项路径（本轮不新增）。
- 流水/双缓冲与多核：RKIR/RKPlan 没有对应字段，源里的 `T.Pipelined` 结构不会进入计划；本轮按“尚不支持”记录。
- 性能：未评估复用/顺序/分块的实际收益；若将来要宣称加速，需要交错对照。
- 复用语义只在“相邻 K 切片之间”被 C 校验覆盖；跨 tile 的更强复用形态没有证据。

---

## 34. §33 复核：入口、舍入、顺序与复用证据分别证明了什么（2026-11-02）

### 34.0 直接回答

**哪些能力已经证实？** 四项：**M/N 分块**（任务网格随 `tile_m/tile_n` 变化）、**遍历顺序**（计划/命令/提交顺序随 `mn_order` 变化）、**K 循环的 256 宽块**（每 tile 的累积任务数随 K 长度变化）、**每 256 宽 K 块写回 FP16 的舍入结构**（固定同一计算后，与“块后写回”参考差 45–119/131072 个元素，与“末尾一次写回”参考差 50028/131072）。

**每项由用户从哪里控制？** 分块与遍历顺序来自 **TileLang 源里的 `rknpu.*` 注解**（必须与 `rknpu.schedule="mn_tiles_v1"` 合用）；K 分块来自 **源里的 K 循环结构**（不是注解）；写回结构由 **源注解 `rknpu.source_k_rounding`** 声明并被校验；**输入复用只能由后端编译选项 `linear_task_order="interleaved_reuse"` 打开**（源码无入口）。

**哪些说法需要收窄或撤回？** 三处：

| 原说法（§33） | 处理 |
|---|---|
| “K=576 vs K=256 的差异证明舍入点” | **撤回**。两者是不同计算，只能证明不同 K 长度生成了相应任务（§34.2）。 |
| “顺序被硬件落实” | **收窄**为“计划任务顺序 = 发射命令顺序 = 提交顺序”，设备内部执行顺序**没有观测**（§34.4）。 |
| “与每块写回参考几乎一致” | **收窄**为“写回结构属于每块写回一族”，残余 45–119 个元素来自块内求和顺序/舍入，参考未能建模，**不宣称逐项完全符合**（§34.3）。 |

另外明确：复用证据只证明**寄存器位被置起**，不推导带宽或性能（§34.5）。

**是否发现新的“接受了要求却没执行”？** 没有。复核中确认五条链都通；`weight_reuse`、流水/双缓冲、多核仍是“未实现、也未声称支持”。

### 34.1 入口核对：每项控制来自哪里、走了哪条路

| 实验 | 控制来源 | 具体入口 | 走的路 |
|---|---|---|---|
| order-m_major / n_major | TileLang 源 **注解** | `rknpu.schedule` + `tile_m/tile_n` + `mn_order` | 源 → 正常编译 → 保存 → 设备加载 → 执行 |
| tile-256 / tile-128 | TileLang 源 **注解** | `rknpu.tile_m`（配合 tile_n） | 同上 |
| k-576 / k-256 | TileLang 源 **结构**（K 循环 trip count） | 源码 `T.Pipelined(T.ceildiv(k, bk))`，`bk=256` | 同上 |
| rounding-per-k / fp32 / undeclared | TileLang 源 **注解** | `rknpu.source_k_rounding` | 同（`fp32_final_narrowing` 在编译期被拒，未产生计划） |
| reuse-off / reuse-on | **后端编译选项** | `CompileOptions(linear_task_order="interleaved_reuse")` | 同上 |

**没有手改计划、没有绕过编译入口**：设备侧加载的计划与主机侧编译并保存的计划逐一致（`plan_hash` 对照，全部 `match=True`）：

| 计划 | host hash | device hash |
|---|---|---|
| order-m-major / order-n-major | `47f5f88e…` / `8a9da12a…` | 同 |
| tile-256 / tile-128 | `29cbf140…` / `6447f9b1…` | 同 |
| reuse-off / reuse-on | `48941dc5…` / `d5e881a2…` | 同 |
| k-576 / k-256 | `29cbf140…` / `298444a4…` | 同 |

顺带一个诚实的观察：`rounding-per-k`、`rounding-undeclared` 与 `k-576`/`tile-256` **同一个计划哈希**（`29cbf140…`）——`source_k_rounding` 是**一致性声明**（要求如实记录已证明的舍入点），不是改变计划的旋钮。

### 34.2 K 实验的更正

§33 用 K=576 与 K=256 的输出差异去推断舍入位置，这是**错误归因**：两者 K 不同 ⇒ 数学不同，输出必然不同。保留原始数据，结论更正为：

| 数据（设备） | 更正后的含义 |
|---|---|
| k-576：6 任务 / 2 提交；k-256：2 任务 / 2 提交 | **源 K 循环的块数决定每个 tile 的累积任务数**（576 → 3 块，256 → 1 块） |
| k-576 vs k-256 输出 131041/131072 元素不同 | 不同计算的必然结果，**不能**作为舍入证据 |

### 34.3 同一计算下的舍入证据（补测）

固定 M=256、K=576、N=512、FP16 输入与同一批 A/B（seed 20261102），设备计划 `k-576`（`29cbf140…`），输出数组已归档（`repros/device-output-k576.npy`）。四个参考用同一组输入：

| 参考 | 差异元素 | max_abs |
|---|---|---|
| 每 256 块后写回 FP16（块内 fp64 求和） | **45 / 131072** | 0.03125 |
| 每 256 块后写回 FP16（块内 fp32 求和） | **119 / 131072** | 0.03125 |
| 全部 K 累加后一次写回（fp64） | **50028 / 131072** | 0.0625 |
| 全部 K 累加后一次写回（fp32） | **50028 / 131072** | 0.0625 |

参考之间本身可区分：两种写回结构相差 **50035** 个元素；块内 fp64 与 fp32 求和相差 **130** 个元素。

**这些数字支持什么判断**：设备输出与“块后写回”一族相差不到 0.1%，与“末尾一次写回”一族相差 38%，因此**写回结构确实是每 256 宽 K 块一次 FP16 写回**，与源注解 `explicit_per_k_fp16` 一致。

**不支持什么判断**：残余 45（对 fp64 块参考）/119（对 fp32 块参考）个元素说明**块内求和顺序与中间舍入**没有被我这两个模型复现；不能据此宣称逐元素完全符合，也没有把块内误差说成“已知的硬件求和顺序”。容差沿用既有标准（`atol/rtol=8e-3` 家族），本轮未改。

### 34.4 “顺序”到底验证到哪一层

| 层次 | 证据 | 结论 |
|---|---|---|
| 计划里的任务顺序 | `mn_order` 改变 `submit_group` 序列（m_major：`0:0,0:1,1:0,1:1`；n_major：`0:0,1:0,0:1,1:1`） | **成立** |
| 发射的命令顺序 | 运行时按 action 顺序发射，到该 submit 末任务整批提交；§27/§32 已把“submit 拥有连续任务段 + 段序按 submit id”变成强制规则，因此命令顺序 = 计划任务顺序 | **成立（代码+计划级）** |
| 提交顺序 | 同上；本轮设备计数复核为 12/4 与 6/2 两种分组 | **成立（计数级）** |
| 设备内部执行顺序（时间序） | **未观测**：没有抓到命令级时间戳，也没有 GPU/NPU 侧 trace | **不作结论** |

因此“输出逐位相同”只说明这些实验没有改变结果，**不能反推任何执行顺序**；§33 里“顺序被硬件落实”的措辞按此收窄。

### 34.5 复用证据复核

**规则（与 C 校验一致）**：`interleaved_reuse` 采用 K 外、N 内顺序；**同一 K 块内，第 2 个及以后的 N tile 复用同一输入（A）tile**，每个 K 块的第一个任务不复用。C 的规则是：`data_reuse` 必须满足“不是 j==0、前一个任务 `inputs[0]` 相同、`source_k_index` 相同”，且“多 K 提交的第一个任务不得复用”。

实测（6 任务计划，`d5e881a2…`）：

| 任务 | K 块 | inputs[0] | data_reuse | 说明 |
|---|---|---|---|---|
| 9 | k0 | 0 | 否 | 链首，按规则不得复用 |
| 10 | k0 | 0 | **是** | 与任务 9 同一输入 tile（换 N tile） |
| 11 | k1 | 1 | 否 | K 块切换，输入视图变了 |
| 12 | k1 | 1 | **是** | 与 11 同一输入 tile |
| 13 | k2 | 2 | 否 | K 块切换 |
| 14 | k2 | 2 | **是** | 与 13 同一输入 tile |

即 6 个任务里 3 个置位（每个 K 块一个），首个置位任务是**第二个任务**，置位位置正好是 `inputs[0]` 不变的边——与规则完全吻合；12×3 的 36 任务版本给出 33 个置位（每 K 块 11 个），同样吻合。

**寄存器字来自设备实际用的程序**：`task_cbuf` 调用 `rk_emit_action`（运行时提交时调用的同一函数、同一份计划、同一设备库 `3749e95d…`），返回的是 CBUF_CON0（寄存器 0x1040）的实发字。实测 `interleaved` 0/6、`interleaved_reuse` **3/6** 带 `0x1000` 位，与计划里的 3 个 `data_reuse` 任务一一对应。

**这证明什么**：复用开关确实写进了 CNA 的 `data_reuse` 位（vendor `npu_regcmd.c:214` 定义该位）。**不证明什么**：不推导带宽下降、不推导加速；本轮没有做性能测量，也没有验证复用对多核/大 tile 的影响。

### 34.6 回归、身份与重放

- 本轮**没有修改任何源码或库**（工作树里除报告外只有 §33 的适配器修复与新增测试，均已在上轮验证）；因此板测数据、C 库与设备库与本轮一致（`f202246a…`、`d527fef0…`/`c9e0d22e…`、设备 `3749e95d…`）。
- 上一轮回归仍为最新：全量 **700 passed / 1 failed / 1 skipped / 261 subtests**（唯一失败是缺 `transformers` 的环境问题），`ctest` **33/33**。
- 新增/更新的证据脚本：`repros/fetch_rounding_output.py`（设备侧导出输出数组）、`rounding_references.py`（四个参考与对比）、`repros/host-experiments.json`、`device-experiments.json`（含 `_pairs`）、`repros/device-output-k576.npy` 等。

```bash
cd research/tilelang-rknpu
# 同计算舍入参考（使用已归档设备输出）
PYTHONPATH=python .venv/bin/python results/20261102T220000Z-tilelang-control-fidelity/rounding_references.py
# 入口/计划事实与 CBUF_CON0（主机）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261102T220000Z-tilelang-control-fidelity/repros/experiments.py --stage host
# 设备侧重新导出输出（可选，需设备）
# PYTHONPATH=... python3 fetch_rounding_output.py   （见 fetch_rounding_run.py 的上传流程）
```

### 34.7 尚未完成

- 设备**内部执行顺序**没有任何观测手段（无 trace），本轮只验证到计划/命令/提交层。
- 块内求和顺序与中间舍入仍未建模，残余 45–119 个元素没有逐项解释；不宣称完全符合。
- `weight_reuse` 无入口；流水/双缓冲/多核未实现；性能未评估。

---

## 35. SmolLM2-135M 的 gate/up 投影：调度要求能不能落到 RKNPU，以及值不值得（2026-11-03）

### 35.0 大白话总结

拿真实模型权重、真实形状，把 gate/up 这一层投影从头到尾跑了一遍：源程序 → RKIR → RKPlan → 发射命令 → RK3588。结论是三句话：

1. **能真正落到硬件上的调度控制，目前是“提交怎么分组”和“输入要不要复用”这两个后端编译选项**。decode（M=1，每次一个 token）把 12 次提交合并成 1 次，单次调用从 **2.93 ms 降到 2.32 ms（约 −21%）**，五次交错测量的原始值互不重叠；打开输入复用反而略慢（2.44 ms）。
2. **prefill（一次 256 个 token）没有测出任何收益**：22.4 / 22.8 / 20.9 ms 三种配置的原始测量值互相重叠，不能据此说哪个更快；这一档的时间主要花在设备执行上（submit_wait 约 14–16 ms）。
3. **TileLang 源里的 M/N 分块与遍历顺序注解，用不到生产的这个源上**：后端明确拒绝“单次发射 + 宽 block grid”的组合（原因见 35.3），所以这一轮没有用源注解给生产计划做分块/顺序对照，也没有假装做过。

另外修掉两个“环境/记录层面的旧账”：`transformers` 缺失导致的固定红灯改成带原因的 skip；发现**磁盘上的生产包比当前源码旧**（缺逻辑算子/数值要求等记录，执行结构相同），这一点下面单列。

### 35.1 真实基线（不是近似计划）

| 项 | 事实 |
|---|---|
| 生产配方 | `tools/compile_smollm2.py`：`make_matmul_prim`，`CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)` |
| decode 形状 | gate/up = **M=1, K=576, N=3072**（每层一份，30 层） |
| prefill 形状 | gate/up = **M=CAPACITY=256, K=576, N=3072**；`prefill_bucket == capacity == 256`，主机强制两者相等 |
| prefill 行数真相 | 逻辑 token ≤ 256，**pad 到 256 行，计划实际处理 256 行**；真实长度靠 `valid_lengths` 掩码 |
| 生产计划结构 | 36 个 npu_task / 12 次提交 / 39 次上传 / 12 次下载；设备需求 4,960,320 B（decode）、5,491,264 B（prefill） |
| 数值契约（计划里记的） | `accumulation=ordered_k_tiles_with_f16_writeback`、`accumulator=float16`、`narrowing_points=[k_tile_boundary]`、`execution.k_chain=[0,256,512]`、`k_tile=256` |
| 共享常量 | 模型 host 默认 `share_constants=True`（`smollm.py:167`） |
| 设备库 | `librknpu_plan.so` `3749e95d…`（当前源码构建），RK3588 OPi 5 Plus，core mask 1 |

**一个必须说的发现：磁盘上的生产包是旧的**。`manifest.json` 记录的源是 `models/tilelang_matmul.py#sha256:5e14d798…`，其 decode 计划 `d8200f45…`、prefill `49733f6d…`；用当前代码按同一配方重建得到 `8daec9f1…` / `11f43cc4…`。逐字段 diff 的结论是：**执行结构完全一致**（36 任务/12 提交/39 上传/12 下载、同样绑定与常量），差异只有后来几轮审计新增的**记录字段**（`logical_ops` 0→1、`numeric_requirements` 0→1、每个任务的 `source_schedule/source_k_rounding/k_padding/k_physical_extent/applied_schedule`）。所以本轮把“当前代码重建”当作同路径基线，“旧包”当作生产对照，两者不可混用。

### 35.2 控制来源逐项标注

| 控制 | 来源 | 本轮是否用生产源测到 |
|---|---|---|
| 分块（M/N tile） | **TileLang 源注解** `rknpu.tile_m/tile_n` | **否**：生产源被拒（35.3）；§33/§34 在带 K 循环的 fixture 源上测过 |
| 遍历顺序 | **TileLang 源注解** `rknpu.mn_order` | 同上 |
| K 分块/写回结构 | 源结构（K 循环）+ 契约 `narrowing_points=[k_tile_boundary]` | 是（作为生产基线本身） |
| 提交分组 | **CompileOptions** `linear_task_order=merged` | **是**（decode/prefill 各一组） |
| 输入复用 | **CompileOptions** `linear_task_order=interleaved_reuse` | **是**（寄存器位 + 计时） |
| 分块尺寸上限、K 切片数、缓冲区/生命周期/上传下载 | 后端自动决定 | —（不作为 TileLang 能力） |

### 35.3 一处真实的能力边界（不是本轮新增限制）

把生产的 `make_matmul_prim` 原样加上 `rknpu.schedule/tile_m/tile_n/mn_order` 后编译，三个分块/顺序候选全部被拒：

```
prefill-annot-default  REJECT a wider block grid is only lowered together with a K loop
prefill-tile128        REJECT 同上
prefill-order-n        REJECT 同上
```

原因不是实现懒惰：生产源是一次发射 + fp32 累加 + 末尾一次窄化，而计划的 K 切片是 256 宽；如果允许这种源声明宽 block grid，规划器仍会把 K 切成 3 段并每段写回 FP16，那就**改变了源声明的舍入点**。所以“源侧 M/N 分块/顺序”目前只对**显式写了 256 宽 K 循环**的源开放。本轮按“不改调度语义”的原则保留该限制，并把结论写成：**生产源不能用源注解调分块/顺序**。

### 35.4 设备实测（真实权重 layer-0 gate/up，五次交错，warmup 已排除）

| 候选 | 控制来源 | 任务/提交 | 复用寄存器位 | 与生产输出 | 总时间中位数 | submit_wait | guard | 设备需求 | 主机需求 |
|---|---|---|---|---|---|---|---|---|---|
| decode-prod | 基线 | 36 / 12 | 0 | 基准 | 2.925 ms | 1.790 | 0.677 | 4.96 MB | 3.67 MB |
| **decode-merged** | CompileOptions | 36 / **1** | 0 | **逐位相同** | **2.320 ms** | 1.097 | 0.782 | 5.93 MB | 4.22 MB |
| decode-reuse | CompileOptions | 36 / 1 | **33** | 逐位相同 | 2.443 ms | 1.086 | 0.891 | — | — |
| prefill-prod | 基线 | 36 / 12 | 0 | 基准 | 22.430 ms | 15.699 | 1.824 | 5.49 MB | 8.60 MB |
| prefill-merged | CompileOptions | 36 / 1 | 0 | 逐位相同 | 22.807 ms | 13.916 | 2.324 | — | — |
| prefill-reuse | CompileOptions | 36 / 1 | 33 | 逐位相同 | 20.858 ms | 13.430 | 2.825 | — | — |

原始值（毫秒，按采集顺序）：

- decode-prod：2.921 / 2.925 / 2.940 / 2.911 / 3.009
- decode-merged：2.318 / 2.251 / 2.332 / 2.334 / 2.320（与 prod 不重叠）
- decode-reuse：2.443 / 2.471 / 2.410 / 2.430 / 2.445
- prefill-prod：22.37 / 22.43 / 22.72 / 20.43 / 22.59
- prefill-merged：22.81 / 25.07 / 19.60 / 24.90 / 19.65（与 prod 重叠）
- prefill-reuse：20.86 / 19.60 / 22.87 / 17.43 / 25.61（重叠，不作结论）

**正确性**：六个候选的输出与生产计划**逐位相同**（0 差异），重复同输入结果稳定，`finite`，换输入后输出改变（decode 的换输入检查在 mock 上做：3072/3072 个元素改变，无旧数据残留）。与“单次窄化”的 fp64 参考的差异（decode 1178/3072、prefill 299765/786432，max_abs 0.0078 / 0.0156）不是错误，而是**契约声明的每 K 块 FP16 写回**带来的既有数值差，与 §34 的结论一致。

**结论**：decode 的 21% 收益来自“12 次提交 → 1 次”（submit_wait 1.79→1.10 ms，per-call 固定开销被摊薄），代价是**设备需求 +0.97 MB**（命令工作区从 12×1 任务变 1×36 任务）与主机需求 +0.55 MB；输入复用把 33 条 CBUF_CON0 置位，但在本形状上**没有**带来额外收益（2.443 > 2.320），不宣称带宽或性能改善。

### 35.5 五个问题的答复

1. **TileLang 提出的调度设计，哪些确实改变了执行？** 对本生产形状：只有**提交分组**与**输入复用**（两者都是后端编译选项，不是源注解）改变了实际执行结构（提交数 12→1、33 条复用寄存器位），其中提交分组带来可测量的 decode 收益；源注解的分块/顺序在本形状上被拒（35.3）。
2. **RKIR/RKPlan 有没有丢失、误解或无必要限制？** 执行路径上没有发现丢失或误读：契约里声明的 `k_tile_boundary` 写回、`k_chain=[0,256,512]` 与执行一致，逻辑算子/任务/数值要求都在计划里（当前构建）。发现两件事：磁盘生产包**过期**（只有记录字段缺失，执行相同）；生产源**不能**用源注解调分块/顺序（有明确理由，保留）。
3. **哪些策略对 prefill / decode 分别有效？** decode：提交合并有效（−21%，证据是 submit_wait 减少与原始值不重叠）；prefill：三种配置无可分辨差异，不宣称任何收益；证据是 prefill 的 submit_wait 占 63–70%，被设备执行主导。
4. **模块收益是否转化成模型收益，付出什么成本？** **没有转成模型收益，也没有做模型级替换**。原因有二：磁盘生产包与当前源码不同源，直接替换需要重建 240 份计划的整包并重跑模型验收；而 gate/up decode 只占每层投影工作的一部分（30 层之一），模型级差异会低于噪声。成本侧数据已给出：+0.97 MB 设备、+0.55 MB 主机。
5. **哪些结论仍没有证据？** 模型级 TTFT/prefill/decode/内存；其它层的权重（只测了 layer 0）；真实捕获激活（本机 `.data/smollm2_v3_inputs.json` 为空，用的是确定性输入）；复用是否在别的大形状上有收益；prefill 的任何调度收益。

### 35.6 环境旧账

- `transformers` 缺失：网络可用，但完整依赖树（huggingface_hub/tokenizers/…）多次安装都在工具时限内没装完。该测试只借它的 `AutoTokenizer` 入口（且被 mock），属性本身与我们的代码无关。**处理**：改为“依赖缺失时带原因 skip”（`tests/test_replay_smollm2_v5.py`），现在该文件 **2 passed / 1 skipped**，不再是一盏常亮红灯。

### 35.7 身份与重放

| 物件 | 值 |
|---|---|
| 生产包（旧）decode/prefill | `d8200f45…` / `49733f6d…`（manifest `source_hash 5e14d798…`） |
| 当前代码重建 decode/prefill | `8daec9f1…` / `11f43cc4…` |
| decode 候选 | prod `8daec9f1…`、merged `86412918…`、reuse `c2b9d033…` |
| prefill 候选 | prod `11f43cc4…`、merged `1d748803…`、reuse `3409d380…` |
| 设备库 | `3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c` |
| 证据 | `results/20261103T000000Z-smollm-schedule-truth/`（`host-campaign.json`、`device-campaign.json`、`repros/production-identity.json`、`production-diff.txt`、`weights-layer0-gate_up.npy`、`plans/`、`board_campaign.py`、`logs/`） |

```bash
cd research/tilelang-rknpu
# 生产计划身份与“旧包 vs 当前构建”的差异
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T000000Z-smollm-schedule-truth/repros/production_identity.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T000000Z-smollm-schedule-truth/repros/production_diff.py
# 候选构建（主机）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T000000Z-smollm-schedule-truth/repros/campaign.py --stage host
# 设备：由 board_campaign.py 上传后在板上 --stage device
for PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_replay_smollm2_v5.py -q -o addopts=""
```

### 35.8 尚未完成

- 模型级替换与 TTFT/decode/内存测量（前置条件：重建整包 + 重新验收）。
- 其它层权重（layer 0 之外）与真实捕获激活。
- prefill 的可用调度收益（当前全部落在噪声里；要分开需要更多轮次或更大的形状变化）。
- 输入复用在其它形状上的收益（本轮只证明寄存器位与结构，不证明性能）。

---

## 36. 把已证实的调度候选装回 SmolLM2：模型级收益与代价（2026-11-03）

### 36.0 大白话总结

上一轮在 gate/up 这一个投影上量到：decode 时把 12 次提交并成 1 次，单次调用从 2.93 ms 降到 2.32 ms。本轮把这套候选通过**官方计划替换机制**（`single_matmul_schedule_v1`，不是改 JSON 造假）装进真实的 SmolLM2-135M 包里，在 RK3588 上跑整模型：

- **每个 token 从 453.5 ms 降到 411.9 ms，快 9.2%**（41.6 ms/token），五组交错测量里**每一对都是变快**（−31 ~ −640 ms/8步），顺序每组对调以排除“谁后跑谁占便宜”。
- **数值一模一样**：同样的 token 序列、同样的 prefill logits、同样的每步 logits（逐位相同）。
- **prefill 没有变化**：被替换的只是 decode 计划，prefill 计划原封不动；实测 prefill 差异（3742 vs 3785 ms）就是噪声，正好当作测量的对照。
- **代价是内存**：每个被替换的计划在设备上多占 0.97 MB（命令工作区从 12×49 KiB 变 1×576 KiB、BO 数 41→52），30 层合计 **+29.1 MB 设备、+16.5 MB 主机**。

所以这一轮的回答是：**上层调度设计能正确落地，并且确实让真实计算更快**——但只在 decode，且要付约 29 MB 设备内存。

### 36.1 装回模型的路径（没有手改计划）

| 步骤 | 事实 |
|---|---|
| 生产包 | `.data/models/smollm2_135m_instruct_package`（268 份计划 + 269 MB 权重），manifest 记录源 `models/tilelang_matmul.py#sha256:5e14d798…` |
| 候选 | `decode-merged.rkplan.json`（`86412918201c…`，36 任务 / 1 提交），由当前代码从生产源编译 |
| 替换机制 | `SmolLM2Runner(..., plan_overrides=...)`，每层 `layer.{i}.gate_up.decode` 一条，`replacement_kind="single_matmul_schedule_v1"`，`base_plan_hash` 取 manifest 里该计划自己的哈希 |
| 兼容性校验（先做） | `bind_sig` / `const_sig`（含 `pack_key`）/ `profile_hash` / 只含 mm+mm_acc 任务 / base 哈希：**decode 与 prefill 候选全部通过**（`override-compat.json`） |
| 加载确认 | 运行时报告 `loaded_plan_override_hashes = 30`（30 层全部生效），产线模式为 0 |

注意：**没有重建包**。磁盘包虽然是旧源构建的（缺 §21/§22 的记录字段，执行结构相同），但官方替换机制的契约只要求绑定/常量/配置一致，候选已满足，因此本轮把改动面缩到最小：只换 30 份 decode 计划。

### 36.2 模型级测量（RK3588，库 `3749e95d…`，5 组，顺序每组对调）

| 组 | 执行顺序 | production decode（8 步合计） | merged decode（8 步合计） | 差 |
|---|---|---|---|---|
| 0 | P→M | 3778.7 ms | 3257.6 ms | −521.1 |
| 1 | M→P | 3515.5 ms | 3304.5 ms | −211.0 |
| 2 | P→M | 3628.2 ms | 3597.4 ms | −30.8 |
| 3 | M→P | 3907.5 ms | 3267.8 ms | −639.7 |
| 4 | P→M | 3482.1 ms | 3295.3 ms | −186.7 |
| **中位数** | | **3628.2 ms** | **3295.3 ms** | **−332.9 ms** |

折算：**453.5 → 411.9 ms/token，−41.6 ms/token（−9.2%）**。

prefill（两种模式的计划完全相同）：production 中位数 **3742.3 ms**、merged **3785.4 ms**（+1.2%，落在 3703.6–3998.6 的散布内）→ 作为“未改动部分不漂移”的对照。

**正确性**：`same_tokens=true`、`same_prefill_logits=true`、`same_step_logits=true` —— token 序列与每一步 logits 逐位相同；两种模式各 9 个 token 完全一致，且组内重复稳定。

### 36.3 代价与解释

| 项 | production | merged（30 层 decode 全换） | 差 |
|---|---|---|---|
| 单计划设备需求 | 4,960,320 B | 5,929,728 B | +969,408 B |
| 命令工作区 | 49,152 B（12 提交） | 589,824 B（1 提交 × 36 任务） | +540,672 B |
| data BO 数 | 41 | 52 | +11 |
| 主机需求 | 3,667,804 B | 4,216,396 B | +548,592 B |
| 30 层合计 | — | — | **+29.1 MB 设备 / +16.5 MB 主机** |

为什么“并提交”能省：gate/up decode 一次调用原本要 12 次阻塞提交，每次都有固定开销（§24/§35 的 `submit_wait` 从中位数 1.79 ms 降到 1.10 ms 就是它）。30 层各调一次，模型里就摊成每 token 约 30×1.2 ms≈36 ms，与实测 41.6 ms 同量级（差额来自 guard/pack 的细微变化）。设备侧执行时间本身没变——这点必须说清楚：**省的是提交与门控的固定成本，不是算得快了**。

### 36.4 五个问题的答复

1. **哪些 TileLang 调度设计确实改变了执行？** 在本轮真实模型上，只有**提交分组**（后端编译选项）改变了执行结构并带来收益；源注解的分块/顺序在生产源上被拒（§35.3），输入复用有寄存器证据但本形状不更快（§35.4）。
2. **RKIR/RKPlan 有没有丢失、误解、无必要限制？修了什么？** 没有丢失或误解：契约里的 `k_tile_boundary` 写回、`k_chain=[0,256,512]`、逻辑算子与数值要求都能在计划里查到，替换后逐位一致。两件已登记的现状：磁盘生产包是旧源构建的（只有记录字段缺失，执行相同）；生产源不能声明 M/N 分块/顺序（有明确理由）。**本轮没有需要修的传递缺陷**，因此没有改任何源码。
3. **对 prefill / decode 分别有效？** decode：+9.2%（模型级，证据如上）；prefill：无变化（被替换的计划不参与 prefill，实测差异为噪声）。
4. **模块收益是否转化成模型收益，付出什么成本？** 转化了：模块级 −0.6 ms/调用 → 模型级 −41.6 ms/token（30 层）；代价 +29.1 MB 设备、+16.5 MB 主机，以及 1 次替换机制的配置（30 条 override，运行时确认全部生效）。
5. **哪些结论仍没有证据？** prefill 的任何调度收益；输入复用在模型级（未装回模型测）；其它角色（qkv/o/down）是否同样受益；不同 prompt 长度/批量下的稳定性；以及“重建整包 + 重新封存验收哈希”这一步没做（本轮刻意不重建）。

### 36.5 与既有证据的关系

- 模块级（§35.4）：decode 2.925 → 2.320 ms（−20.7%），逐位一致。
- 模型级（本轮）：453.5 → 411.9 ms/token（−9.2%），逐位一致。两者方向与量级一致（gate/up decode 约占每 token 时间的 19%）。
- 既有数值结论不变：与“单次窄化”的 fp64 参考的差异仍来自契约声明的每 K 块 FP16 写回，本轮未改容差、未改数值政策。

### 36.6 身份与重放

| 物件 | 值 |
|---|---|
| 生产包 manifest | `b923fc5b…`（268 份计划，capacity 256） |
| 生产 decode 计划（layer_00_gate_up_decode） | `d8200f45…`；prefill `49733f6d…` |
| 候选 merged decode | `86412918201c…`（36 任务 / 1 提交） |
| 替换校验 | `results/20261103T020000Z-smollm-model-level/override-compat.json`（decode/prefill 均 installable） |
| 设备库 | `3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c` |
| 原始时间 | `model-raw.txt`、`model-summary.json`（设备侧 JSON 因工作区清理未取回，控制台原始行已归档） |

```bash
cd research/tilelang-rknpu
# 替换兼容性（本地）
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T020000Z-smollm-model-level/override_compat.py
# 模型级 A/B（需设备；上传包后运行）
# python3 model_campaign.py --package <pkg> --candidate decode-merged.rkplan.json \
#         --out model-result.json --groups 5
```

### 36.7 尚未完成

- 未重建生产包、未重新封存模型验收哈希（本轮不需要：替换机制不动包）。
- prefill 侧没有候选（源注解被拒，提交合并对 256 行桶无收益）。
- 其它角色（qkv/o/down）与输入复用的模型级收益未测。
- 单 prompt、单 seed；不同上下文长度与温度下的稳定性未验证。
- 派出去做“重建包 + 验收封存前置条件”的只读侦察子代理在收尾时尚未回报，其范围已被本轮的实测（替换契约 + 包身份）覆盖。

---

## 37. 为什么生产的 gate/up 源码不能声明 M/N 分块与遍历顺序（2026-11-03）

### 37.0 大白话

**一句话：拒绝是必要的，但原因不是“分块做不到”，而是“这份源码没法证明自己要的舍入点”。**

生产源码 `make_matmul_prim` 是一次发射、一个 fp32 累加器、末尾只窄化一次。把它的 `T.Kernel(1,1)` 改成按 M/N 分块的网格并加上 `rknpu.tile_m/tile_n/mn_order`，会被前端拒绝：`a wider block grid is only lowered together with a K loop`。

为什么必要：设备原语一个任务最多累加 256 宽 K，而 K=576，所以**任何**真实计划都必须把 K 切成 3 段、每段写回 FP16。tiled 调度（`k_tile=256`、`accumulation=ordered_ascending_f16_writeback`）正是要求源码把这个写回点**声明出来并可被证明**；没有 K 循环的源码文本里没有任何东西能证明它，而它唯一能诚实声明的 `fp32_final_narrowing` 又和执行（3 次窄化）矛盾。所以只能拒绝。

**但有个重要的实测更正**：生产计划**实际执行的就是每 K 块写回**（下文：与“每块写回”参考只差 332/786432 个元素，与“末尾一次窄化”差 299760）。所以这不是“计算不同”导致的拒绝，而是“**源码文本无法自证**”导致的拒绝——是一条**证明义务**，不是能力缺口。

**修了什么**：只修诊断。原来那句 `a wider block grid is only lowered together with a K loop` 把原因说成了“网格”，现在明确说“宽网格需要显式的每 256 宽 K 的 FP16 写回（写 K 循环），或保持单 tile”，并带上 `required=[explicit_per_k_fp16, single_k_tile_fp16]`。**没有放宽任何检查，也没有把分块加进生产源**。

**能不能用**：换成能准确表达设备计算的源码（256 宽 K 循环 + `explicit_per_k_fp16`）之后，分块与遍历顺序都真的生效：`tile_m` 256→128 让任务从 36 变 72、提交从 12 变 24；`n_major` 改变发射的 tile 顺序。四条计划（生产 vs K 循环 vs 128 分块 vs n_major）**输出逐位相同**。

**值不值**：不值——128 分块慢 43%（28.4 ms vs 19.2 ms），`n_major` 与默认在噪声内（19.12 vs 19.15），decode 无差异（3.14 vs 3.18 ms）。所以生产策略保持不变。

### 37.1 拒绝复现（decode 与 prefill 都一样）

| 项 | 值 |
|---|---|
| 源码 | `models/tilelang_matmul.py:make_matmul_prim` + `T.func_attr(rknpu.schedule/tile_m/tile_n/mn_order/source_k_rounding)`，`T.Kernel(T.ceildiv(N,256), T.ceildiv(M,256))` |
| 形状 | decode `(1,576,3072)`、prefill `(256,576,3072)` |
| 选项 | `CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)`，profile `7ef710bb…` |
| 最早拒绝 | `frontend/tilelang_adapter.py:5654`（`extract()` 的 block-grid 校验），错误码 `RK_E_UNSUPPORTED` |
| 原因（修复前措辞） | `a wider block grid is only lowered together with a K loop` |
| 原因（现在措辞） | `a wider block grid needs an explicit per-256-K FP16 writeback (the tiled schedule narrows at every K block); write the K loop, or keep a single tile`，附 `required=[explicit_per_k_fp16, single_k_tile_fp16]` |
| 不加注解时采用的规则 | 36 任务 / 12 提交 / 39 上传 / 12 下载；计划记录 `source_k_rounding=fp32_final_narrowing`（描述**源码**），数值契约声明 `narrowing_points=[k_tile_boundary]`、`accumulation=ordered_k_tiles_with_f16_writeback`（描述**执行**） |

### 37.2 真正的原因（两道门，第二道才是语义）

1. **前端（先触发）** `tilelang_adapter.py:5641-5666`：block-grid 只允许 extent=1，或与可检测的 K 循环共存；`k_block = _detect_k_block(...)` 为 None 时直接拒绝。这条 gate 的存在理由写在 `_detect_k_block:5066-5070`：多 tile 的输出覆盖证明依赖 block 索引分析，而那套分析长在 K 循环提取器里。→ **这一半是“实现位置”问题**：分块机器本身不依赖 K 循环。
2. **调度器（语义）** `passes/tile.py:108-118`：scheduled 源码必须声明 `explicit_per_k_fp16` 或 `single_k_tile_fp16`，注释写明“absence, a final-only FP32 narrowing, and an annotation that disagrees with the proven source all mean the schedule would describe a different numerical program”。→ **这一半是必要的**：tiled 路径固定 `k_tile=256`、每块写回，`fp32_final_narrowing` 与它矛盾。

**为什么同一份无注解源码可以执行、加上不改计算的 M/N 要求反而被拒**：无注解时前端走单发射路径，不提任何舍入声明，规划器按默认把 K 切成 3 段（契约里已声明 `k_tile_boundary`），于是“源码说了什么”和“执行做了什么”不需要一致——事实上它们确实不一致（源码是 final-only，执行是 3 次窄化），只是没人声称要遵守。加上注解后，`rknpu.schedule` 触发“声明必须被证明”的义务，矛盾才暴露。这不是“注解让计算变难”，而是“注解要求自证”。

### 37.3 实测：生产计划执行的就是每 K 块写回

同一组真实权重（layer-0 gate/up，FP16）与同一输入，设备（库 `3749e95d…`）：

| 候选 | 任务/提交 | 与 fp64 单次窄化差 | 与 fp16 每块写回链差 |
|---|---|---|---|
| prefill-A 生产 | 36/12 | 299760/786432 | **332** |
| prefill-B K 循环 tile256 | 36/12 | 299760 | **332** |
| prefill-C K 循环 tile128 | **72/24** | 299760 | **332** |
| prefill-D K 循环 tile256 n_major | 36/12 | 299760 | **332** |
| decode-A 生产 | 36/12 | 1178/3072 | **1** |
| decode-B K 循环 | 36/12 | 1178 | **1** |

两两对照（设备输出，逐位）：A⟷B、B⟷C、B⟷D、decode A⟷B **全部 0 差异**。

读法：① 生产计划与“每块写回”源码**数值等价**，说明拒绝与计算含义无关；② C 让任务几何真的变了（36→72、提交 12→24），输出仍逐位相同，说明**分块控制确实生效且不改数学**；③ D 让发射的 tile 顺序改变（`mm:0:0,mm:0:1,mm:1:0,mm:1:1` vs `mm:0:0,mm:1:0,mm:0:1,mm:1:1`），输出仍相同——**顺序生效**只到“计划/命令/提交顺序”这一层，设备内部时间序未观测。

decode 只有 1 个 M tile，`m_major`/`n_major` 发射完全相同的任务序列（测试固定了这一点），所以 decode 不能作为“顺序改变执行”的证据。

### 37.4 是否值得采用（交错计时，5 轮中位数，含完整调用）

| 候选 | 任务/提交 | 总时间 | submit_wait | guard |
|---|---|---|---|---|
| prefill-A 生产 | 36/12 | 19.867 ms | 11.907 | 2.134 |
| prefill-B tile256 | 36/12 | 19.151 ms | 11.605 | 1.356 |
| prefill-C tile128 | 72/24 | **28.418 ms** | 19.914 | 2.204 |
| prefill-D n_major | 36/12 | 19.121 ms | 11.622 | 1.340 |
| decode-A 生产 | 36/12 | 3.135 ms | 1.735 | 0.925 |
| decode-B K 循环 | 36/12 | 3.177 ms | 1.747 | 0.934 |

结论：**没有值得采用的候选**。tile128 明显更慢（+43%，任务与提交翻倍），`n_major` 与默认在噪声内（−0.03 ms），decode 无差异。A vs B 的 0.7 ms 也不作为收益申报（两者计算相同、结构相同，差异落在 guard 项上，属噪声）。因此**保留生产策略**，本轮不做模型级替换（§36 的提交合并收益不在此重复申报）。

### 37.5 五个问题的答复

1. **原生产源码为什么被拒？** 前端 block-grid 校验要求“宽网格 + K 循环”，生产源码没有 K 循环；本质上是调度器要求 scheduled 源码声明并证明每 256 宽 K 的 FP16 写回，而该源码文本无法自证。
2. **拒绝是否必要？** 必要（在现有契约下）：K=576 时任何计划都要切 3 段并写回 3 次，`fp32_final_narrowing` 与执行矛盾；tiled 契约要求声明可证明的舍入点。**但**具体实现位置一半是历史原因（覆盖证明只在 K 循环提取器里），且原诊断措辞把原因归到“网格”，具有误导性。
3. **修了什么/为什么不修？** 修了诊断：把“网格需要 K 循环”改成“宽网格需要显式的每 256 宽 K FP16 写回，并给出两种合法舍入声明”。**没有**放宽 gate、没有把分块能力塞进单发射路径（那会让计划执行与记录不一致，正是本审计一直在消除的东西）。
4. **哪些 TileLang 源码设计已经实际改变执行？** 用能准确表达设备计算的源码（K 循环 + `explicit_per_k_fp16`）时：`tile_m` 改变任务/提交几何（36/12 → 72/24），`mn_order` 改变发射顺序（多 M tile 形状才能区分）。生产源本身无法表达这两项。
5. **是否更快、代价、未测？** 都不更快：128 分块 +43%，顺序在噪声内，decode 无差异；代价（128 分块）是任务/提交与设备预算翻倍。未测：其它 tile 尺寸（192/64）、其它角色（qkv/o/down）、模型级（无候选入选）、设备内部时间序。

### 37.6 代码与证据身份

| 物件 | 值 |
|---|---|
| 本轮代码改动 | `frontend/tilelang_adapter.py`（诊断措辞 + 注释，无行为变化）、新增 `tests/test_mn_schedule_source_entry.py` |
| 生产 decode/prefill 计划 | `8daec9f11046…` / `11f43cc45ee9…` |
| K 循环 tile256 / tile128 / n_major（prefill） | `f57a05661132…` / `141d9bc7f472…` / `5003b9efaf55…` |
| decode K 循环 | `7e9f18b897d0…` |
| 设备库 | `3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c` |
| 回归 | 全量 **705 passed / 2 skipped / 0 failed / 265 subtests** |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T040000Z-mn-schedule-source-entry/repros/source_entry.py --stage host
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_mn_schedule_source_entry.py -q -o addopts=""
# 设备：board_run.py 上传后在板上 --stage device（结果 device-sources.json）
```

证据：`results/20261103T040000Z-mn-schedule-source-entry/`（`host-sources.json`、`device-sources.json`、`repros/source_entry.py`、`repros/plans/*.rkplan.json`、`repros/weights-layer0-gate_up.npy`、`board_run.py`、`logs/full-pytest.log`、`snapshot/`）。

### 37.7 尚未完成

- 没有把“多 tile 覆盖证明”从 K 循环提取器抽出来给单发射路径用（那是一个真实的实现改进，但会引出“计划执行与源码声明不一致”的记录问题，超出本轮范围）。
- 只测了 tile 256/128 与两种顺序；未测 192/64、未测其它角色、未做模型级（无候选入选）。
- 设备内部执行顺序仍无法观测；本轮的“顺序生效”只到命令/提交层。

---

## 38. M/N 调度声明为什么要连带数值形式：设计、实测与一处文档/实现冲突（2026-11-03）

### 38.0 大白话

**这段不对称是设计，而且写在文档里；本轮修的是“拒绝理由报错了、和文档对不上”。**

- **不加调度声明**：走历史默认路径。后端按目标数值策略把 K 切成 256 宽的段、每段写回 FP16；计划里 `source_k_rounding=fp32_final_narrowing` 描述**源码**（一个累加器、末尾窄化），数值契约 `narrowing_points=[k_tile_boundary]` 描述**执行**。两者不一致是允许的，因为源码没有声称要控制执行。
- **加 M/N 声明**：源码声称控制几何，而几何（`k_tile=256`）与写回点耦合。所以要求源码本身写成“每 K 段写回”的形式；否则计划会真的多窄化两次，却被注解说成只窄化一次——那就是静默改写数值，必须拒绝。
- **用户不需要额外写舍入注解**：阶梯实验第 4 级证明，K 循环源码即使不写 `rknpu.source_k_rounding` 也被接受，适配器**按实际的 copy 结构推导**出 `explicit_per_k_fp16`。文档也明确写着“identifies them by their actual copies, never by an annotation”。所以“额外要求”其实是“源码要写成每块写回”，不是“多写一个注解”。
- **唯一仍被拒的组合**：声明了 M/N、但源码只窄化一次。`TILELANG-SCHEDULE.md:23-30` 明确承诺这种情形以 `RK_E_NUMERIC_CONTRACT` 拒绝，而实现原先报的是 `RK_E_UNSUPPORTED`（“wider block grid …”）——**文档/实现冲突**，本轮按文档对齐。
- 行为没有放宽：生产计划**逐字节不变**（decode `8daecf11…`、prefill `11f43cc4…`，与 §37 相同），因此设备证据沿用 §37，没有重复上板。

### 38.1 两类要求各自是什么意思

| 概念 | 含义 | 依据 |
|---|---|---|
| 源码表达的计算/舍入 | 循环体内 `Acc16→Cl`、`gemm`、`Cl→Acc16` = 每 K 段写回；单次发射 = 末尾一次窄化 | 适配器按实际 copy 识别（`_detect_k_block` / `_k_loop_module`），文档 `TILELANG-SCHEDULE.md:36-41` |
| `numeric_mode="rk_f16_tiled"` 允许什么 | 允许把 K 按 256 切开、每段 FP16 写回：契约 `accumulation=ordered_k_tiles_with_f16_writeback`、`narrowing_points=[k_tile_boundary]`、`k_tile=256` | 生产计划 `numeric_requirements` 实测 |
| `rknpu.source_k_rounding` | **描述源码**并与“已证明的舍入点”交叉校验；缺省时由结构推导 | 第 4 级 ACCEPT；第 6 级冲突 REJECT |
| M/N 分块与遍历顺序 | 本身不额外承诺数值；但要求源码把与几何配套的数值形式写清 | 第 2/3 级 REJECT；文档 `:23-30` |

**用户只提出 M/N 调度，是否就承担新的数值证明要求？** 承担，但只承担“把源码写成它本来就要执行的形式”这一条：适配器自己推导并交叉校验，用户不必再写注解。依据是文档的显式条款，而不是事后从某个拒绝条件倒推。

### 38.2 阶梯实验（真实 gate/up 源码，`rk_f16_tiled`，decode 与 prefill 结论相同）

| 级 | 源码形态 | decode | prefill | 最早决定位置 |
|---|---|---|---|---|
| 1 | 无注解（生产源码） | ACCEPT 36/12，stamp `fp32_final_narrowing` | 同 | — |
| 2 | + 网格声明（m_major，无 K 循环） | **REJECT -4** `scheduled source narrows only after the full K reduction…` | 同 | `frontend/tilelang_adapter.py`（原报 -2） |
| 3 | + 遍历顺序（n_major，无 K 循环） | **REJECT -4**（同上） | 同 | 同上 |
| 4 | 显式 K 循环 + 每段写回，**不写舍入注解** | **ACCEPT** 36/12，stamp `explicit_per_k_fp16`（推导） | 同 | — |
| 5 | K 循环 + 与源码一致的舍入声明 | ACCEPT 36/12 | 同 | — |
| 6 | K 循环 + 冲突声明（`fp32_final_narrowing`） | REJECT -4 `does not match the proven rounding points` | 同 | `frontend`（不是调度器） |

结论：**唯一“需要数值形式”的地方是第 2/3 级**（声明 M/N 但不写每段回写），而它不是“多写注解”的问题，而是“源码本身就只窄化一次”的问题。第 4 级说明注解是可选的。

### 38.3 修了什么（诊断与文档对齐，行为不变）

| 位置 | 之前 | 现在 |
|---|---|---|
| `frontend/tilelang_adapter.py`（网格拒绝） | `RK_E_UNSUPPORTED`（-2），`a wider block grid needs an explicit per-256-K FP16 writeback…` | `RK_E_NUMERIC_CONTRACT`（-4），`scheduled source narrows only after the full K reduction; the tiled schedule writes FP16 back at every 256-wide K block… (write the K loop, or drop the M/N declaration to stay on the legacy default path)`，`required=[explicit_per_k_fp16, single_k_tile_fp16]` |
| `tests/test_frontend.py::test_multi_block_grid_without_k_loop_is_rejected` | 断言 -2 与 “block grid”，注释说原因是“没有输出覆盖证明” | 断言 -4 与数值契约措辞，注释改为引用文档条款 |
| `tests/test_mn_schedule_source_entry.py` | 上一轮按 -2 断言 | 改为 -4，并新增“K 循环源码无需舍入注解”的断言 |

**没有放宽任何 gate**，也没有把分块加进单发射路径；生产计划的许多字段逐字节不变（下述）。

### 38.4 验证

| 检查 | 结果 |
|---|---|
| 生产计划（decode / prefill） | 与 §37 记录的计划 `to_dict()` **逐字节相同**（`8daecf11…` / `11f43cc4…`） |
| 合法 M/N 声明 | K 循环形态下仍改变任务几何（§37：`tile_m` 128 → 72 任务/24 提交）与发射顺序 |
| 冲突/未知/越界 | 第 6 级冲突声明、未知 `rknpu.*`、tile 越界仍被拒（§33/§37 测试） |
| 保存/加载 | 计划身份不变；`load_plan` 重验证通过 |
| 全量回归 | **706 passed / 2 skipped / 0 failed / 265 subtests** |
| 设备证据 | 本轮**没有新接受的组合**、计划与执行代码逐字节未变，故沿用 §37 的真机结果（A/B/C/D 逐位相同、计时 19.87/19.15/28.42/19.12 ms） |

### 38.5 对 §37 的更正

§37 把这条拒绝的原因写成“多次分块覆盖证明只长在 K 循环提取器里”，并据此把拒绝归为“一半是实现位置问题”。**更正**：真正的规则是数值契约（调度声明必须与每 256 K 写回配套），文档 `TILELANG-SCHEDULE.md:23-30` 明确写了这条拒绝；覆盖证明是伴生约束，不是理由。§37 的其余结论（拒绝必要、生产计划执行的就是每块写回、tiling/order 生效但不更快）不受影响。

### 38.6 五个问题的答复

1. **两种处理为何不同？** 不加注解走历史默认路径（后端按目标策略切 K 并记录两套字段）；加注解则声称控制几何，必须连同几何蕴含的数值形式一起成立。
2. **这种差异合理吗？** 合理，且是文档条款（不是从拒绝倒推）：几何与写回点通过 `k_tile=256` 耦合，允许“声明 M/N 但源码只窄化一次”等于让计划静默改写数值。用户的负担只是“源码写成它本来要执行的样子”，注解可省。
3. **修了什么？** 修了诊断：该拒绝从 `RK_E_UNSUPPORTED` 改为文档承诺的 `RK_E_NUMERIC_CONTRACT` 并明确措辞；两个测试同步。**没有**放宽校验。
4. **现在怎样表达 M/N 调度？** 写带 K 循环且每段写回的源码 + `rknpu.schedule/tile_m/tile_n/mn_order`；舍入注解可选。仍会被拒：单次发射源码 + M/N 声明、与源码冲突的舍入声明、未知 `rknpu.*`、tile 越界。
5. **哪些结论分别由代码/离线/设备验证？** 代码：两类要求与 gate 位置；离线：阶梯 6 级 + 生产计划身份 + 全量回归；设备：§37 的真实权重/输入逐位对照与计时（本轮计划未变，未重复上板）。

### 38.7 身份与重放

| 物件 | 值 |
|---|---|
| 本轮代码改动 | `frontend/tilelang_adapter.py`（拒绝码+措辞）、`tests/test_frontend.py`、`tests/test_mn_schedule_source_entry.py` |
| 生产 decode / prefill | `8daecf11046515c…` / `11f43cc45ee98…`（与 §37 相同） |
| 设备库 | `3749e95d…`（未变） |
| 证据 | `results/20261103T060000Z-mn-numeric-coupling/`（`repros/ladder.py`、`ladder.json`、`logs/full-pytest.log`、`snapshot/`） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T060000Z-mn-numeric-coupling/repros/ladder.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_mn_schedule_source_entry.py tests/test_frontend.py -q -o addopts=""
```

### 38.8 尚未完成

- 没有实现“单发射源码 + 多 tile 网格”的接受路径：文档与契约都要求这种源码先写成每段写回，本轮按文档保留拒绝。
- 只在 gate/up 形状上做了阶梯；qkv/o/down、lm_head 未逐项验证（同一代码路径，风险低但未测）。
- 环境里的 `transformers` 部分安装已清除，replay 测试按“依赖缺失”skip（原因写在测试里）。

---

## 39. gate/up 的跨 tile 数据复用审计：机会、传递与实测（2026-11-03）

### 39.0 大白话

**这段计算里真正值得复用的是权重，但现有调度没有任何一种顺序能让硬件复用权重；而已经实现的“输入复用”在 decode 上几乎没有可省的东西。**

- decode（M=1）每个任务读的输入 tile 只有 ~0.5–1 KB，权重 tile 是 128 KB：**CBUF 读量 3.70 MB 里 99.97% 是权重**。所以“复用输入”即便开了 33 个复用位，也省不到东西——这与 §35 的实测（复用没带来额外收益）一致，本轮给出了原因。
- prefill（M=256）输入与权重各约一半（3.46 MB / 3.46 MB），复用输入理论上能省下 11/12 的输入读量（约 3.17 MB 的 CBUF 读），但因为生产是 12 次提交、每个提交内 3 个 K 切片相邻，**相邻任务既不共享输入也不共享权重**（审计 `data_ok=0, weight_ok=0`）；只有改成 K 外、N 内的顺序（`interleaved_reuse`）才出现 33 个可复用邻接。
- **权重复用（weight_reuse）目前不可达**：C 校验要求“前一个任务与当前任务 `inputs[1]` 相同且 `source_k_index` 相同”。planner 能产生的所有顺序都把 K 放在最内层，所以同一 (N,K) 的两个 M tile 永远不相邻。要让它成立需要 (n,k) 固定、m 最内层的第三种嵌套——那是新的调度维度，本轮不做，也不置位未经验证的寄存器组合。

### 39.1 六层访问与复用（逐层计数）

| 层 | 内容 | decode (1,576,3072) | prefill (256,576,3072) | 结论 |
|---|---|---|---|---|
| 1 Host 原始数组 | A 按 K 切片读 3 次（每次逻辑 tile）；B 读 3 次 | A 1.1 KB / B 3.54 MB（一次） | A 3.46 MB / B 3.54 MB | 无重复读同一字节 |
| 2 Host packing | A 3 次上传；B 36 次常量上传（进程内一次） | A 1.1 KB | A 3.46 MB | 无重复打包 |
| 3 常量缓存 / 持久 BO | 36 个 shared constant BO（进程级，`share_constants=True`） | 4.72 MB 常驻 | 同 | 每调用**不重传**，但设备每次仍读 |
| 4 device → CBUF | 每任务读 A tile + B tile | **3.70 MB/调用**（A 1.1 KB，B 3.70 MB） | **7.08 MB/调用**（A 3.46，B 3.46+padding） | 这是真正可省的量级 |
| 5 CBUF operand reuse | `data_reuse` 已实现；`weight_reuse` 无路径 | 33 个邻接可复用（K 外/N 内） | 33 个 | 见 39.2 |
| 6 输出 / partial | 12 下载 + 12 publish；累加器乒乓 | 12 tiles，1×256 各 | 12 tiles，256×256 各 | merged 下 12 个输出同时存活（52 slots vs 41） |

数据来源：`repros/audit.py`（逐任务的 view/BO/offset/bytes 统计）+ 生产计划 accounting。**注意**：第 3 层的“缓存”不等于第 4 层“不再读”；本轮没有把二者混为一谈。

### 39.2 复用资格审计（C 校验器的定义）

C 规则（`rk_plan_validate.c`）：多 K 提交的第一个任务不得复用；`data_reuse` 要求前一任务 `inputs[0]` 相同且 `source_k_index` 相同；`weight_reuse` 要求前一任务 `inputs[1]` 相同且 `source_k_index` 相同。据此逐计划统计相邻对：

| 计划 | 任务/提交 | 不同的 A view / B view | 同一 A 且同一 k 的相邻对 | 同一 B 且同一 k 的相邻对 | 已置位 d/w |
|---|---|---|---|---|---|
| decode 生产 | 36/12 | 3 / 36 | **0** | **0** | 0 / 0 |
| decode + interleaved_reuse | 36/1 | 3 / 36 | **33** | 0 | **33** / 0 |
| prefill 生产 | 36/12 | 3 / 36 | **0** | **0** | 0 / 0 |
| prefill + interleaved_reuse | 36/1 | 3 / 36 | **33** | 0 | **33** / 0 |
| 512×576×512 m_major | 12/4 | 6 / 6 | 0 | **0** | 0 / 0 |
| 512×576×512 n_major | 12/4 | 6 / 6 | 0 | **0** | 0 / 0 |

结论：**`data_reuse` 只在“K 外、N 内”顺序下成立；`weight_reuse` 在 planner 能产生的任何顺序下都不成立**（每个 tile 内部 K 都在最内层，相邻任务的 K 一定变）。

### 39.3 发射命令里的实际置位（emitted-command 证据，非寄存器读回）

用运行时的真实发射函数（`rk_emit_action`）解码 CBUF_CON0（寄存器 0x1040），`interleaved_reuse` 计划：

| 计划 | 任务 | 声明复用 | 命令字带 0x1000 | 一致 | 未置位的 k_index | 每个 K 块的置位数 |
|---|---|---|---|---|---|---|
| decode | 36 | 33 | **33** | 是 | 0,1,2（各块链首） | 11/11/11 |
| prefill | 36 | 33 | **33** | 是 | 0,1,2 | 11/11/11 |

即：置位位置与资格审计逐一对齐（每个 K 块 12 个 N tile，除链首 1 个外 11 个置位），符合 C 规则；这**只**证明复用位被正确写入发射命令，不证明带宽下降。

### 39.4 候选与实测（不重复计算 §36 的提交合并收益）

| 候选 | 控制来源 | 改变了什么 | 实测 |
|---|---|---|---|
| `interleaved`（K 外/N 内，1 提交） | CompileOptions | 12 提交→1，无复用位 | decode 2.320 ms、prefill 22.8 ms（§35） |
| `interleaved_reuse`（同上 + 输入复用） | CompileOptions | 追加 33 个 `data_reuse` 位 | decode **2.443 ms**（比不复用慢 0.12 ms）、prefill 20.9 ms（噪声内）（§35） |
| 权重复用 | **无入口** | — | 未测；只有推导上界 |

**为什么输入复用没有收益（本轮给出原因）**：decode 的 A 读量只有 1.1 KB（占 0.03%），省它没有意义；prefill 的 A 占一半，但复用只省“同一 K 切片内重复读同一 A tile”的部分，而硬件本来也可能在 CBUF 内保持该 tile（`data_reuse` 表达的正是这一点），实测落在噪声内。

**权重复用如果实现，上界是多少（推导，不是实测）**：每个 (n,k) 的 B tile 读一次而不是 #M_tiles 次，节省 = B 字节 × (1 − 1/#M_tiles)。生产 tiling 下 #M_tiles = 1（decode 与 prefill 都是），**节省为 0**；只有把 `tile_m` 降到 128（prefill 2 个 M tile）才可能有 50% 的 B 读量节省，但 §37 已测 tile_m=128 反而慢 43%（任务/提交翻倍），所以即便加上权重复用也未必划算。

### 39.5 模型级

没有接入：唯一有稳定收益的是 §36 的提交合并（−9.2%/token，已接入并验证）；本轮没有发现新的、经实测更优的复用候选，因此**不改模型**。

### 39.6 六个问题的答复

1. **哪些数据存在可利用的重复访问、发生在哪一级？** 第 4 级（device→CBUF）：prefill 的 A tile 在 12 个 N tile 间被重复读（约 3.17 MB 可省），decode 的 A 只有 1.1 KB（无意义）；B tile 在 #M_tiles=1 时没有重复，只有把 `tile_m` 降到 128 才出现。第 3 级的常量缓存已经避免了重复上传，但那不等于硬件不读权重。
2. **RKIR/RKPlan 是否保留并利用了这些关系？** `data_reuse` 全链路保留并被利用（声明→C 校验→发射 33/33 置位，位置与资格一致）。`weight_reuse` 的管道（描述符位、C 规则、发射器）存在，但**没有规划路径能满足它的邻接条件**，因此未被利用，也没有计划能表达它。
3. **哪些候选真正改变了复用/任务/提交？** `interleaved_reuse`：12 提交→1、33 个复用位（其余候选不变结构）。
4. **哪些收益有实测支持，哪些只是推导？** 实测：模块时间（§35）、模型 −9.2%/token（§36，来自提交合并而非复用）。推导：CBUF 读量 3.70/7.08 MB、权重复用上界 B×(1−1/#M)、tile128 的 43% 代价（§37 实测的是时间，不是带宽）。
5. **模块收益是否转化为模型收益、代价多少？** 复用没有模块收益，故未接入；已接入的是提交合并，代价 +29.1 MB 设备 / +16.5 MB 主机（§36）。
6. **是否发现并修复语义丢失/错误拒绝/不安全放行？** 本轮**没有改代码**：审计确认 `data_reuse` 链路正确（声明与命令字一致、位置符合 C 规则），`weight_reuse` 属于“未实现且不可达”，不是不安全放行；也没有发现新的错误拒绝。`weight_reuse` 的最小实现条件（m 最内层顺序 + 邻接/几何/K 切片校验）已写清，但按“条件无法确认就不置位”的要求保留为未验证。

### 39.7 身份与重放

| 物件 | 值 |
|---|---|
| 计划 | decode 生产 `8daecf11046515c…`、decode interleaved_reuse `c2b9d0337273…`、prefill 生产 `11f43cc45ee98…`、prefill interleaved_reuse `3409d38081be…` |
| 设备库 | `3749e95d…`（本轮未上板：无新候选、计划未变） |
| 证据 | `results/20261103T080000Z-cross-tile-reuse-audit/`（`repros/audit.py`、`audit.json`、`repros/command_reuse.py`、`commands.json`、`logs/`） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T080000Z-cross-tile-reuse-audit/repros/audit.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T080000Z-cross-tile-reuse-audit/repros/command_reuse.py
```

### 39.8 尚未完成

- `weight_reuse` 未实现：需要 (n,k) 固定、m 最内层的顺序（新调度维度），并补齐邻接/几何/bank-split/K 切片/提交边界的资格反例；本轮只给出条件与上界，不置位。
- 没有硬件计数器，CBUF 读量是“每任务 view 字节”的推导上界，不是实测带宽。
- 未测其它角色（qkv/o/down/lm_head）与其它 tile 组合；未做模型级复用实验（无候选入选）。

---

## 40. 权重复用：缺的是任务顺序，已补通并实测（2026-11-03）

### 40.0 大白话

**§39 说“权重复用在任何顺序下都不可达”，方向对，但理由写错了**：不是“所有顺序都把 K 放最内层”，而是**没有任何一种顺序把 M tile 放在最内层**。interleaved / interleaved_reuse 是 K 外、N 内；merged / baseline 是 K 内。权重复用要求相邻任务是“同一 (N,K)、不同 M”，这一层嵌套 planner 从来没产生过。

**本轮补上了这一层**：新增两个显式策略 weight_stationary（只改顺序）与 weight_stationary_reuse（顺序 + 复用位）。它们不引入 SmolLM 形状名单，只按 operand/geometry/K 链/资源条件工作：固定 (N tile, K slice)，M tile 最内层；每个输出 tile 的 K 链仍严格升序、仍每段 FP16 写回。

**实测（真实 layer-0 gate/up 权重的 N=512 切片，RK3588）**：

| 候选 | 任务/提交 | 复用位（发射命令） | 与基线输出 | 中位总时间 | submit_wait |
|---|---|---|---|---|---|
| baseline-256（生产 geometry） | 6 / 2 | 0 | 基准 | **4.481 ms** | 1.959 |
| merged-128（同 geometry、旧顺序） | 12 / 1 | 0 | 逐位相同 | 5.391 ms | 3.039 |
| wstat-128（顺序改、复用关） | 12 / 1 | 0 | 逐位相同 | 5.638 ms | 3.059 |
| wstat-reuse-128（同顺序、复用开） | 12 / 1 | **6** | 逐位相同 | 5.633 ms | 2.988 |

即：**机制被真正触发（6 个任务在 CBUF_CON0 上置 0x2000）、数学不变（四者逐位相同）、但没有收益**（复用开关差 0.005 ms，落在噪声内；真正的时间差来自 tile_m=128 让任务数翻倍：4.48 → 5.4–5.6 ms）。

### 40.1 对 §39 的更正

| §39 的说法 | 更正 |
|---|---|
| “planner 所有顺序都把 K 放最内层” | **错**。baseline / scheduled_baseline / merged 是 tile 内 K 最内；interleaved / interleaved_reuse 是 **K 最外、N 内**（因此才有 33 个 A 复用邻接）。正确表述：**没有顺序把 M 放最内**，所以 B 复用邻接恒为 0。 |
| “decode 输入只有 1.1 KB” | **混用了逻辑字节与物理字节**。按物理 geometry，decode 的 A tile 是 12×256×2 = 6 KiB（rows 补到 12），K 尾块 12×64×2 = 1.5 KiB；3 个切片合计 **13.5 KiB**，不是 1.1 KB。B tile 每个 256×256×2 = 128 KiB（尾块 256×64×2 = 32 KiB），36 个合计 **4.22 MB**。A 占比 0.3%，结论（A 复用无意义）不变，但数字要按物理字节算。 |
| “输入复用减少量的分母” | 应写成：**可省字节 = Σ(可复用相邻对中当前任务的 A 物理字节)**，分母是 Σ 所有任务的 A+B 物理字节。interleaved_reuse 下 33 对、每对省 1 个 A tile。 |
| “CBUF 保持/固定开销吸收了收益” | 这是**假设**，本轮没有测量支撑；data_reuse 无收益的原因目前只能确定“A 占比极小（decode）”与“落在噪声内（prefill）”，不能断言是 CBUF 保持。 |
| “§37 的 tile128 慢 43% 可排除该方向” | **不能**。§37 测的是 tile-major + 每 tile 一个提交；本轮在 merged（1 提交）与 weight_stationary（1 提交）下重测，tile128 仍慢（+0.9~1.2 ms），但那是**新的分组方式下的独立结论**，不是沿用旧结论。 |

### 40.2 权重复用的完整执行条件（代码核实）

| 条件 | 依据 | 状态 |
|---|---|---|
| 语义：复用“上一个任务在该核 CBUF 里已加载的 operand” | vendor npu_regcmd.c:211-214（注释：reuse previous task on this core；位 0x2000=weight，0x1000=data） | 源码已确认 |
| 相邻性：必须是同一 submit 中**紧邻的前一个任务** | C 校验 rk_plan_validate.c（j==0 不得复用；比较 s->task_ids[j-1]） | 已有校验 |
| operand 身份：inputs[1] 视图相同 | 同上 | 已有校验 |
| K 索引：source_k_index 相同 | 同上 | 已有校验 |
| 几何 / bank split：由视图相同隐含（同一 view → 同一 physical geometry、atom、bank 划分）；emitter 另行校验 physical_cols 一致 | finish_mm_op 前置条件 + C | 已由前面两道保证 |
| 提交边界：不得跨 submit 复用 | 校验只在同一 submit 内比较 | 已有校验 |
| Python 侧资格检查 | **本轮新增**（此前 Python 完全不检查复用位） | 新增 |

所以缺的**不是硬件条件（已由 vendor 源码与现有校验界定），也不是 C/发射器支持（都在），而是 planner 从未产生“M 最内层”的任务顺序**。

### 40.3 本轮实现

| 位置 | 改动 |
|---|---|
| options.py | linear_task_order 增加 weight_stationary / weight_stationary_reuse（两者要求 submit_mode=chain） |
| passes/dag.py | 新增发射分支：按 (n0, m0) 排序后 **K 外、N 中、M 内** 发射；weight_stationary_reuse 在“前一任务 N 相同”处置 weight_reuse=True；仍属一个 mk:linear 提交，K 链与依赖保持不变 |
| plan.py | 新增 Python 复用资格检查：首任务不得复用；data_reuse / weight_reuse 要求同 submit 内前一任务同 operand 视图且同 k_index（与 C 规则一致，提前报错） |
| tests/test_weight_reuse.py | 8 例：顺序与置位数、merged 保持 K 最内、保存/加载保留标志、两层接受、首任务复用被拒（Python + C）、换权重视图被拒（Python + C）、复用位不改变数值 |

控制来源明确：**这两个策略是后端 CompileOptions，不是 TileLang 源码能力**；本轮没有新增源码注解。

### 40.4 结果与判断

1. **§39 哪些结论成立/被纠正？** 成立：“weight_reuse 此前不可达”“A 复用在 decode 无意义”“复用无实测收益”。被纠正：K 内/外的表述、逻辑 vs 物理字节、可省字节公式、“CBUF 保持吸收收益”的归因（降级为假设）、以及用 §37 的 tile128 结果排除该方向（本轮重新独立测量）。
2. **权重复用缺的是什么？** **任务顺序**。硬件条件（vendor 语义 + CBUF bank 划分）、C 校验、发射器、generator 都齐；planner 现在能产生所需顺序。
3. **本轮实现并验证了什么？** weight_stationary(_reuse) 策略 + Python 资格检查 + 8 个测试；C 校验器接受合法计划、拒绝三类负例；mock 与设备均逐位相同；发射命令里 6 个任务带 0x2000。
4. **复用开关的差异是否被独立隔离？** 是：wstat-128 与 wstat-reuse-128 任务数、提交数、顺序、geometry 完全相同，只差 6 个复用位；两者输出逐位相同、时间差 0.005 ms（噪声）。顺序的影响由 merged-128 vs wstat-128 隔离（同为 12 任务/1 提交，时间 5.391 vs 5.638 ms，顺序本身也没有收益）。
5. **是否值得采用？** 不采用：tile_m=128 的几何本身让这个小形状慢 20%（4.48→5.4 ms），复用位在此之上没有可测收益；生产默认不变，模型不接入。上界仍是 B 字节 ×(1−1/#M_tiles)，在 #M_tiles=2 时最多省一半的 B 读（本形状 12×~64 KiB 的一半 ≈ 0.38 MB），相对于每调用 ~5 ms 的固定开销太小。

### 40.5 身份与重放

| 物件 | 值 |
|---|---|
| 计划 | baseline-256 75daf9db…、merged-128 e4b88ad6…、wstat-128 92608b37…、wstat-reuse-128 d0463577… |
| 设备库 | 3749e95d… |
| 回归 | 全量 **714 passed / 2 skipped / 0 failed / 265 subtests** |
| 证据 | results/20261103T100000Z-weight-stationary/（repros/wstat_probe.py、wstat.json、plans/*.rkplan.json、device_wstat.py、device-wstat.json、logs/full-pytest.log） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T100000Z-weight-stationary/repros/wstat_probe.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_weight_reuse.py -q -o addopts=""
```

### 40.6 尚未完成

- 只在该 (256,576,512) 形状上验证；更大 N（12 个 N tile）会让任务数超过 64/submit，需要分批提交（本轮不实现）。
- 没有硬件计数器：B 读量的节省是推导，不是实测带宽。
- data_reuse 与 weight_reuse 同时可用的组合（固定 N、K，内层 M 交替时 A 会变）在本顺序下不成立，未研究。
- 未做模型级接入（无收益）。

---

## 41. 完整 gate/up（256×576×3072）上的 weight-stationary：分组、语义与实测（2026-11-03）

### 41.0 大白话

**上一轮之所以“整份计划被拒”，确实只是分组限制：不是形状不允许，而是所有任务被要求塞进一个 64 任务的 submit。** 本轮按“整 N tile 打包、每组不超过 64 任务”切分，完整 gate/up 就正常编译成 **2 个合法 submit（60 + 12 个任务）**，C 校验器接受，设备执行正确。

- **分组规则**：每个 N tile 有 #K×#M = 3×2 = 6 个任务，按 64//6 = 10 个 N tile 一组 → 组 0 = 60 任务、组 1 = 12 任务；组内 K 外、M 内（权重复用），每个输出 tile 的 K 链完整落在同一组内，每组首任务复用位关闭。
- **权重复用真的被置位**：72 个任务里 **36 个**在发射命令的 CBUF_CON0 上带 0x2000（每个 (N,K) 的第二个 M tile），与 C 的资格规则一致。
- **数值不变**：五份计划（生产 tile256 / merged-256 / wstat-256 / wstat-128 / wstat-reuse-128）设备输出**逐位相同**，与“每 K 块写回”参考只差 332/786432。
- **没有收益**：复用开/关在同一顺序与分组下是 29.54 vs 29.84 ms（逐轮 28.5–32.9 与 26.3–33.3 大幅重叠），**不能声称有效**；tile_m=128 本身比生产 geometry 慢约 12 ms。生产保持 17.70 ms 中位，为最快。

### 41.1 完整 gate/up 的分组与合法性

| 项 | 值 |
|---|---|
| 形状（现场核实） | M=256, K=576, N=3072；tile_m=128, tile_n=256 → 2×12=24 个输出 tile，K 切片 3 → **72 任务** |
| 生产对照 | tile_m=256：12 tile × 3 = 36 任务 / **12 提交**（每 tile 一个提交） |
| weight-stationary | 72 任务 / **2 提交**（60 + 12），组内 K 外、N 中、M 内 |
| submit 上限约束对象 | 每组任务数（不是整份计划）；未改动 64 的上限 |
| 跨组依赖 | 每个输出 tile 的 K 链在组内闭合（跨组只保留 submit 顺序依赖），测试断言 48 条 K 连续边全部同组 |
| 组首复用 | 每组第一个任务 weight_reuse=False（测试固定） |
| partial 生命周期 | acc 缓冲按 kidx 乒乓；组内最后一次消费后才被下一 tile 复用（沿用既有 buffer 规划，未改） |
| 提交配方 | 组名采用 mk: 前缀 → submit builder 判为 multi_k_acc_v1（其匹配器接受任意 mm/mm_acc 序列） |

**为什么安全**：① 组内任务连续且 ≤64；② 每条 K 链完整在组内（C 要求 prior 在**同 submit 内更早**出现）；③ 组首任务不复用；④ 组间只有顺序依赖，没有跨组的 partial 复用；⑤ 复用位仅在同 (N,K)、相邻 M tile 处出现，operand 视图与 K 索引都与前一任务相同。

### 41.2 一个仍然存在的拒绝（不是缺陷）

`merged`（以及 interleaved 系列）在 tile_m=128 上仍然被拒：它们的做法是**全部任务放一个 submit**，72 > 64。这是既有策略的形态，不是本轮的形状限制；`weight_stationary` 通过分组绕开它，而没有提高上限。因此在 tile_m=128 上**无法**给出“同 geometry、既有顺序”的对照——差异只能列出来，不能把组合变化归因于单一机制。

### 41.3 设备实测（真实 layer-0 gate/up 权重，RK3588，库 3749e95d…）

| 候选 | 任务/提交 | 复用位 | 与生产输出 | 中位总时间 | 逐轮原始值（ms） | submit_wait | guard |
|---|---|---|---|---|---|---|---|
| baseline-256（生产） | 36 / 12 | 0 | 基准 | **17.703 ms** | 18.07 / 17.50 / 17.41 / 17.70 / 18.86 | 11.271 | 1.353 |
| merged-256（同 geometry、单提交） | 36 / 1 | 0 | 逐位相同 | 17.655 ms | 17.43 / 17.66 / 23.19 / 17.63 / 18.87 | 11.050 | 1.326 |
| wstat-256（顺序改、无复用机会） | 36 / 1 | 0 | 逐位相同 | 21.797 ms | 18.01 / 25.14 / 21.80 / 21.76 / 22.99 | 11.076 | 3.299 |
| wstat-128（顺序+分组、复用关） | 72 / 2 | 0 | 逐位相同 | 29.835 ms | 32.92 / 29.84 / 29.28 / 26.32 / 33.29 | 18.111 | 4.347 |
| wstat-reuse-128（同顺序分组、复用开） | 72 / 2 | **36** | 逐位相同 | 29.543 ms | 32.30 / 29.54 / 32.90 / 28.52 / 29.44 | 17.776 | 4.012 |

正确性：五者输出全部逐位相同；重复稳定、换输入改变、finite；与 fp64 单次窄化参考差 299760/786432、与每 K 块写回链差 332（与 §37 一致，既有数值结论不变）。

**噪声说明**：逐轮散布 26–33 ms，复用开关的中位差（−0.30 ms，−1.0%）远小于散布，**不能判为有效**；wstat-256 与 merged-256 的中位差 4.1 ms 同样重叠（wstat 有一轮 18.01 ms），只能记为“未能分离”。本轮没有硬件计数器，operand 字节节省（每 (N,K) 省一个 256×128×2 = 64 KiB 的 B tile，共 12 组 × 3 K ≈ 2.25 MB 推导上界）**不是实测带宽**。

### 41.4 语义传递核对

| 环节 | 检查 | 结果 |
|---|---|---|
| TileLang → RKIR | 源码为显式每 K 块 FP16 写回的 K 循环；注解只有 schedule/tile_m/tile_n/mn_order | 通过（§37/§38 已固定） |
| RKIR → RKPlan | 任务覆盖 24 tile × 3 K = 72；每 tile K 升序；applied_schedule 记录 tile_m/tile_n/order | 通过（测试断言） |
| RKPlan → descriptor/C | C 校验器接受两份计划；首任务复用、错配 operand、错配 K 的负例被拒 | 通过（8 例测试，Python 与 C 各自验证） |
| 发射命令 | 真实发射路径解码 CBUF_CON0：36 个任务带 0x2000，位置与 C 规则一致 | 通过（emitted-command 证据，非寄存器读回） |
| 保存/加载 | 复用标志与计划身份在 save_plan/load_plan 后保持 | 通过 |

### 41.5 六个问题的答复

1. **原先的整计划拒绝是否由可处理的分组限制导致？** 是。tile_m=128 时 72 任务超过了“全部放一个 submit”的形态，而不是形状本身不被支持；按整 N tile 打包后编译成 2 个合法 submit。
2. **采用了什么分组规则，为什么安全？** 每组取 64//(#K×#M) 个整 N tile；组内 K 外、M 内；每输出 tile 的 K 链在组内闭合；组首不复用；组间只有顺序依赖。5 条不变量都有测试固定（含 48 条 K 连续边同组、组首复用关闭、越界拒绝）。
3. **RKIR/RKPlan 是否准确保留并落实调度与数值要求？** 是：任务覆盖、K 升序与每 K 块 FP16 写回结构、applied_schedule、复用位位置、逻辑算子与数值要求关联都在计划内并在加载后保持；C 与发射命令两侧一致。
4. **完整 gate/up 中顺序、分组、复用分别造成什么变化？** 顺序（merged-256 → wstat-256，同 36 任务/1 提交）中位 +4.1 ms 但逐轮重叠；分组（wstat-256 → wstat-128）任务 36→72、提交 1→2，中位 21.8→29.8 ms；复用（wstat-128 → wstat-reuse-128）只多 36 个复用位，中位 −0.30 ms（噪声内）。
5. **是否值得采用，代价多少？** 不值得。生产 geometry 中位 17.70 ms 最快；tile_m=128 明显更慢，复用位带来的差异不可分辨。资源上 tile_m=128 与 2 提交使 submit_wait 从 11.3 ms 升到 17.8 ms，guard 从 1.35 升到 4.01 ms；生产默认与模型均不变。
6. **哪些只是结构/命令证据，哪些有设备与性能证据？** 结构/命令：72 任务、2 提交、60/12 分组、36 个复用位位置、C 接受与负例、保存/加载一致。设备与性能：五份计划的逐位相同输出、重复/换输入/有限性、5 轮交错计时与 submit_wait/guard 分解。**没有**硬件带宽、也没有计数器级别的证据。

### 41.6 身份与重放

| 物件 | 值 |
|---|---|
| 计划 | baseline-256 f57a05661132、merged-256 d2dc49fb0f8e、wstat-256 b4b211df9439、wstat-128 c4c66316a379、wstat-reuse-128 4107a47b2d86 |
| 设备库 | 3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c |
| 回归 | 全量 **722 passed / 2 skipped / 0 failed / 270 subtests** |
| 证据 | results/20261103T120000Z-wstat-full-gate-up/（repros/build_plans.py、repros/add_plans.py、plans/*.rkplan.json、device_full.py、device-full.json、logs/full-pytest.log） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T120000Z-wstat-full-gate-up/repros/build_plans.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_weight_stationary_grouping.py tests/test_weight_reuse.py -q -o addopts=""
```

### 41.7 尚未完成

- 分组的“每组 N tile 数”目前按 64//(#K×#M) 取最大，没有比较更小分组的代价；未做组合搜索。
- 无法在 tile_m=128 上给出“同 geometry、既有顺序”的对照（merged 在该 geometry 仍被拒）。
- 未做模型级接入（无收益）；未测其它角色；无硬件计数器。

---

## 42. 权重复用方向的收尾：三层对照、分组与 recipe 可靠性（2026-11-03）

### 42.0 大白话

**结论：这个方向可以结束。** 完整 gate/up（256×576×3072）在三层真实权重上都能安全执行、输出逐位相同，但 **tile_m=128 的任何变体都比生产慢 6–9 ms**，而“复用开/关”“分组 10 vs 5 个 N tile”“组内顺序”这三个单机制对照在三层之间**符号翻转**，没有可复现的差异。

同时修了两处真实的脆弱点：① 分组上限原先只能“整份计划一个提交”，现在支持按整 N tile 的有界分组（未改 64 上限）；② submit recipe 原先只靠组名前缀（mm:/mk:）猜测，现在由任务上的显式 `submit_recipe` 字段决定，旧计划保留前缀回退。

### 42.1 三个隔离对照（同 geometry、同分组、同任务/提交数）

| 对照 | 保持一致 | 改变项 | 计划 |
|---|---|---|---|
| grouped-128 vs wstat-128 | geometry、group partition（60+12）、任务数 72、提交数 2、reuse 关闭 | **组内任务顺序**（K 内 vs K 外/M 内） | 9ba8d6ed8a79 / d67f4d73447f |
| wstat-128 vs wstat-reuse-128 | 同上 + 顺序 | **weight_reuse**（36 个置位） | d67f4d73447f / 6004178d50c1 |
| wstat-reuse-128 vs wstat-reuse-128-n5 | 计算、数值要求、组内顺序、reuse 政策 | **group partition**（60+12 vs 30+30+12） | 6004178d50c1 / 3f58c6d2b162 |

生产计划（baseline-256，36 任务/12 提交，f57a05661132）只作为最终采用价值对照。

### 42.2 三层实测（真实 layer-0/15/29 gate/up 权重，5 轮交错，中位总时间 ms）

| 候选 | 任务/提交 | 复用位 | layer 0 | layer 15 | layer 29 | 与生产输出 |
|---|---|---|---|---|---|---|
| baseline-256（生产） | 36 / 12 | 0 | **20.88** | **21.37** | **22.26** | 基准 |
| grouped-128（同分组、旧顺序） | 72 / 2 | 0 | 30.36 | 26.70 | 30.16 | 逐位相同 |
| wstat-128（顺序改） | 72 / 2 | 0 | 29.41 | 29.50 | 28.73 | 逐位相同 |
| wstat-reuse-128（复用开） | 72 / 2 | 36 | 31.02 | 29.26 | 28.34 | 逐位相同 |
| wstat-reuse-128-n5（分组 5） | 72 / 3 | 36 | 29.96 | 28.95 | 30.45 | 逐位相同 |

逐轮原始值见 device-layers.json。正确性：五个候选在三层都 `matches_production=True`（0 差异）、重复稳定、换输入改变、finite；与 fp64 单次窄化参考与每 K 块写回链的差异与 §37/§41 一致（既有数值结论不变，未改容差）。

**单机制结论（三层一致性的判据）**：

| 机制 | L0 | L15 | L29 | 判定 |
|---|---|---|---|---|
| 组内顺序（grouped→wstat） | −0.95 | +2.80 | −1.43 | 符号翻转 → 无可靠差异 |
| 权重复用（off→on） | +1.61 | −0.24 | −0.39 | 符号翻转 → 无可测收益 |
| 分组（10→5 N tile） | −1.06 | −0.31 | +2.11 | 符号翻转 → 无可靠差异 |
| geometry（生产 tile256 → tile128） | +8.5 | +7.9 | +6.5 | **三层一致更慢** → 可结束 |

### 42.3 recipe 选择方式审查与修复

| 问题 | 结论 |
|---|---|
| mk:/mm: 前缀是正式协议吗？ | 是既有内部约定（所有 linear 策略都用它），但**它同时承担了控制分派**：schedule.py 用前缀决定 linear_k_acc_v1 / multi_k_acc_v1 / swiglu_ffn_v1 / gate_ew_v1。 |
| recipe 是否独立按结构验证？ | 是：plan 级 `_recipe_matches` 按模板序列匹配（linear_k_acc_v1 只接受 mm+mm_acc 链、multi_k_acc_v1 接受任意 mm/mm_acc 且至少一个 mm_acc），C 校验器另按依赖/K 序/几何独立检查。 |
| 是否存在“改标签就改变准入、却不查结构”的路径？ | **有脆弱耦合**：前缀是唯一来源，改一个字符串就换 recipe（结构检查只在 plan 级兜底）。 |
| 修复 | 任务上新增显式 `submit_recipe` 字段（planner 直接声明），schedule.py 优先采用它、组内不一致则拒绝；没有该字段时保留前缀回退，旧计划不受影响。新增测试：声明字段生效、同一分区换组名不改变 recipe、错误 recipe 被 plan 级匹配器与 C 双重拒绝。 |

### 42.4 分组实现检查

| 检查项 | 结果 |
|---|---|
| 组大小合法值/非法值/尾组 | 60+12（auto）、30+30+12（5）、越界请求（11 个 N tile → 66 > 64）被拒 |
| 恰好达到上限/超过上限 | 5 个形状 subTest 断言每组 ≤64；merged 在 tile128 仍被拒（72 > 64），未改上限 |
| 不同 M/K/N 组合 | (256,576,3072)/(384,576,1024)/(256,576,512)/(256,512,768)/(512,1024,512) 均 ≤64 且 Python 校验通过 |
| submit 首任务禁用 reuse | 每段首任务断言为 False；组首/组间越界置位被 Python 与 C 分别拒绝 |
| operand/K/geometry 不匹配 | 换权重视图的负例被 Python 与 C 拒绝 |
| 缺失依赖、跨组 K 链 | 48 条 K 连续边断言同组；设计要求 K 链留在组内，越界情形直接拒绝 |
| 保存/加载/再验证 | 复用位与 recipe 在 save_plan/load_plan 后保持并再次通过校验 |

### 42.5 七问答复

1. **同 geometry、同分组的顺序对照结果？** grouped-128 vs wstat-128（同 72 任务、2 提交、同分区、reuse 关）在三层分别 −0.95 / +2.80 / −1.43 ms，符号翻转，没有可复现差异。
2. **单独启用 weight_reuse 的结果？** 31.02 / 29.26 / 28.34 对 29.41 / 29.50 / 28.73（+1.61 / −0.24 / −0.39 ms），符号翻转，**无可测收益**；复用位确实置了 36 个（发射命令），但这只证明机制触发。
3. **分组规模的影响与资源代价？** 10 vs 5 个 N tile：两层 −1.06 / −0.31 ms，一层 +2.11 ms；提交数 2→3、命令工作区与 submit_wait 随之上升（三层 submit_wait 与 guard 见 JSON）。无可靠收益。
4. **首、中、末层是否一致？** geometry 结论三层一致（tile128 慢 6.5–8.5 ms）；三个单机制对照三层符号翻转，即不一致——因此不作收益声明。
5. **recipe 与结构检查是否可靠，修了什么？** 结构检查可靠（plan 级匹配器 + C 独立校验），但 recipe 选择原先只靠组名前缀，属脆弱耦合；已改为显式 `submit_recipe` 字段优先、前缀回退兼容，并加负例。
6. **TileLang / RKIR / RKPlan 各自承担什么、传递证据是什么？** TileLang 源码结构决定 K 循环与每 K 块 FP16 写回（§37/§38 已固定）；source annotation 只决定 tile_m/tile_n/mn_order；CompileOptions 决定 linear_task_order 与 linear_group_n_tiles（本轮新增的 grouped_linear 与分组上限均属后端规划能力，不是源码控制）；RKIR/RKPlan 保留 applied_schedule、submit_recipe、weight_reuse 与 K 链；发射命令里 36 个 0x2000 位与 C 规则位置一致。
7. **是否采用、是否结束该方向？** 不采用（生产在三层都最快），**结束权重复用方向**：机制已打通、语义正确、复用位正确置位，但在完整 projection 上无可测收益，且 tile_m=128 的 geometry 代价稳定地更大。生产默认与模型均未改动。

### 42.6 身份、回归与证据

| 物件 | 值 |
|---|---|
| 计划 | baseline-256 f57a05661132、grouped-128 9ba8d6ed8a79、wstat-128 d67f4d73447f、wstat-reuse-128 6004178d50c1、wstat-reuse-128-n5 3f58c6d2b162 |
| 设备库 | 3749e95d0387e3c11d061a955b0acb02bcb430ad66504d5b5def4a66e1da590c |
| 回归 | 全量 **729 passed / 2 skipped / 0 failed / 272 subtests** |
| 新增测试 | tests/test_submit_recipe_and_grouping.py（7 例）、tests/test_weight_stationary_grouping.py（8 例）、tests/test_weight_reuse.py（8 例） |
| 证据 | results/20261103T140000Z-wstat-controls-and-layers/（plans/、repros/build_all.py、repros/export_layers.py、device_layers.py、device-layers.json、logs/） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T140000Z-wstat-controls-and-layers/repros/build_all.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_submit_recipe_and_grouping.py tests/test_weight_stationary_grouping.py -q -o addopts=""
```

### 42.7 与历史章节的关系

- §41 的“复用与分组无可测收益”在本轮得到三层独立确认；“tile128 更慢”也从单层扩展为三层一致。
- §40 的“缺的是任务顺序”仍然成立；本轮补上了同分组下的顺序对照，结论是顺序本身也没有收益。
- §39 的两处更正（K 内/外表述、物理字节）继续有效；本轮不重复。

---

## 43. TileLang 源结构能表达哪些 RKNPU 调度语义（2026-11-03）

### 43.0 大白话

**三个源结构里，只有“串行 K 循环的迭代顺序”是真的调度语义；输出 tile 的遍历顺序在 TileLang 里根本不存在（block grid 是 program-index 空间，没有顺序含义），而 `T.Pipelined(num_stages)` 是缓冲提示、不是 RKNPU 流水。**

本轮实测（同一数学、真实 gate/up 形状、正规编译入口）：

| 源变体 | 计划 | 说明 |
|---|---|---|
| canonical m_major | 36 任务 / 12 提交，2b36fedddecff127 | 基准 |
| 改循环变量名（rowtile/coltile） | **同一 hash** | 等价的索引写法被 canonicalize，不制造差异 |
| 交换 grid 轴（N×M vs M×N，索引同步交换） | **同一 hash** | grid 枚举顺序对计划没有任何影响 |
| 加 `T.Pipelined(..., num_stages=2)` | **同一 hash** | 流水提示不改变计划 |
| 声明 `mn_order=n_major` | 见 §37（fixture 路径） | 只有显式声明才改变发射顺序 |

所以：**输出 tile 的遍历顺序不可能从源结构推导**，它要么是显式声明（现有 `rknpu.mn_order`，属“源偏好”），要么是后端策略（CompileOptions）；而声明本身只是偏好，设备内部时间序仍未观测。

### 43.1 TileLang 各结构的真实语义（本地实现依据）

| 结构 | 承诺什么 | 依据 |
|---|---|---|
| `T.Kernel(gx, gy)` block grid | **program-index 空间**（blockIdx.x/y），不是执行顺序 | `.pydeps/tilelang/language/kernel.py:202/263/443`（dim=0 → blockIdx.x …） |
| `T.serial` K 循环 | 迭代按程序顺序发生 → 累积顺序是**依赖要求** | `_detect_k_block` 记录有序 `k_blocks`；计划 `k0/k_index/k_count` 按此顺序 |
| `T.Parallel` 块内循环 | 元素级并行，无调度含义 | 前端按 copy/fill 语义提取 |
| `T.Pipelined(start, stop, num_stages)` | “pipeline producers/consumers 之间最多用几个 buffer”；0 = 不启用流水，纯性能提示 | `.pydeps/tilelang/language/loop.py:112-133` |
| 多个 `T.Kernel` 作用域 | 显式分阶段（staged）程序 | `tilelang_adapter.py:5590` |

### 43.2 本轮选定的代表性设计：输出 tile 组织（Path C）

设计说明（先写结论、再验证）：

1. **源结构表达的是数学语义**：`T.Kernel` 的两轴只声明“有多少个独立输出 tile”，不声明它们的先后；`T.Parallel` 是元素级；因此 tile 的组织是数学的（覆盖关系），遍历不是。
2. **下游可做的等价改写**：重命名、网格轴互换后的索引重建、`Pipelined` 提示丢弃——这些都不改变计划（实测三者同 hash）。
3. **拟落实的 task order**：由 `rknpu.mn_order`（显式声明，Path B）或 `linear_task_order`（后端策略）决定；本轮不新增注解。
4. **仍应留在后端的控制**：submit 分组、CBUF reuse 资格、BO/bank 分配——源里没有对应语义（§40/§41 已确认 reuse 位只由“相邻任务同 operand”决定）。
5. **为什么没有误用语义**：block grid 从不被解释为设备时间顺序；`Pipelined` 从不被当作已实现的流水；`num_stages` 不进入计划。

### 43.3 发现的一处信息丢失（未修，如实记录）

`pipeline_hint_dropped` 记录在**两条提取路径上不一致**：

- `make_linear`（普通 K 循环路径）的计划里能看到 `pipeline_hint_dropped` 字段；
- 本轮 guarded-tail 形式的源码（`T.fill` + 守卫 copy + `T.Pipelined(num_stages=2)`）编译出的计划与不流水版本**逐字节同 hash**，且 `pipeline_hint_dropped: true` **出现 0 次**——即“源请求了 2 级缓冲、后端不做流水”这件事在该路径上没有被记录。

这属于 §5 要求区分的“源请求 vs 已应用决定”缺口，但**不影响执行或数值**（计划完全相同），本轮因预算未修；最小修法与验证方式已明确：在 tail 提取路径把 `num_stages`/`source_loop_annotations` 写进 hints（与普通路径同名字段），并加一条断言“带 Pipelined 与不带的计划只在记录字段上不同”。

### 43.4 传递链核对

| 环节 | 记录 | 结果 |
|---|---|---|
| 源 | 循环结构、索引、守卫、`T.Pipelined` 参数 | 已读（本地 TileLang 0.1.15 源码 + 前端提取点） |
| frontend | 有序 `k_blocks`、`source_k_loop`、`source_k_rounding`、`source_mn_tiling`；grid 只做 min/step 校验 | 与源一致 |
| RKIR | matmul 属性携带 schedule（`kind/tile_m/tile_n/order/k_tile/accumulation`） | 一致 |
| planner | 依 schedule 生成 task geometry/order；K 链按 `k_blocks` 顺序；submit 分组由策略决定 | 一致 |
| RKPlan | `applied_schedule`、`source_*` 记录、`k0/k_index/k_count`、`submit_group`、`submit_recipe` | 保存/加载后保持（既有测试） |
| 发射命令 | 复用位/顺序与计划一致 | §37/§41 既有设备证据（本轮计划与已测计划同 hash，未重复上板） |

### 43.5 六个问题的答复

1. **TileLang 哪些源结构具有可用于 RKNPU 调度的明确含义？** 只有串行 K 循环的迭代顺序（依赖要求，已被提取为有序 `k_blocks`）。block grid 是 program-index 空间、`T.Parallel` 是元素级、`T.Pipelined` 是缓冲提示——都不构成调度语义。
2. **本轮选了哪项，为什么？** 选“输出 tile 组织”作为研究对象，因为它是唯一看起来像调度、实际却没有语义的候选；用它来划定“源能表达 / 只能后端表达”的边界最有信息量。
3. **RKIR、RKPlan 如何承接，实际执行改变了什么？** 承接的是 K 链顺序与显式声明的 schedule；本轮四个源变体（改名、换网格轴、加流水提示）产生**同一计划**，即执行没有任何改变——这正是“不能靠源结构造出顺序”的证据。
4. **哪些控制仍需后端策略？** submit 分组（§41 新增 `linear_group_n_tiles`）、CBUF reuse 资格（§40/§41）、tile 遍历顺序的默认值、BO/bank/缓冲区规划。原因是源结构没有这些语义。
5. **是否发现并修复信息丢失、误解或错误拒绝？** 发现一处：guarded-tail 路径不记录被丢弃的流水提示（普通路径会记录）。未修（预算），已写明最小修法与验证方式；除此之外未发现新的误解或错误拒绝。
6. **哪些结论有源码、计划、命令、设备证据？** 源码级：kernel.py/loop.py 的语义定义；计划级：四个变体同 hash、`pipeline_hint_dropped` 的两路径差异；命令级与设备级：沿用 §37/§41（本轮计划与那些已测计划同 hash，只新增记录字段时不必重复上板）。

### 43.6 身份与证据

| 物件 | 值 |
|---|---|
| 计划 | canonical / renamed / swapped-grid / pipelined 四者同为 2b36fedddecff127；fixture 的 m_major 计划见 §37（f57a05661132） |
| 设备 | 本轮未上板（计划与已测计划同 hash，执行字段与库未变：3749e95d…） |
| 回归 | 未改执行代码，全量仍 729 passed / 2 skipped / 272 subtests（§42 末次记录） |
| 证据 | results/20261103T160000Z-source-structure-semantics/（repros/grid_semantics.py、repros/grid_semantics.json、snapshot/） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T160000Z-source-structure-semantics/repros/grid_semantics.py
```

### 43.7 尚未完成

- guarded-tail 路径的流水提示记录（最小修法已写明）。
- 未研究多 `T.Kernel` 的 staged 路径在 RKNPU 上的对应关系（属于独立方向）。
- 设备内部执行顺序仍不可观测；本轮所有“顺序”结论只到计划/命令/提交层。

---

## 44. 流水提示的记录补齐，以及对 §43 全局表述的收窄更正（2026-11-03）

### 44.0 大白话

§43 指出“guarded-tail 路径不记录被丢弃的流水提示”，本轮把它补齐了，并且顺手修了**两处**更严重的问题：普通 K-loop 路径其实也一直写着 `pipeline_hint_dropped: False`（哪怕源码明确要求 `num_stages=2`），而 block-annotation 路径用 `bool(annotations)` 判断，会把 `num_stages=0`（TileLang 语义是“不启用流水”）误报成“请求了缓冲却被丢弃”。

现在两条路径用同一个 helper：记录**实际请求值**（`source_pipeline_stages`）、**原始注解**（`source_loop_annotations`）、**是否被丢弃**（`pipeline_hint_dropped`）、**未采用原因**（`pipeline_not_applied_reason`），并且**不声称流水已落实**。生产源码（无 K 循环、无提示）计划哈希**逐字节不变**。

同时更正 §43 里两处超出证据的全局断言（见 44.4）。

### 44.1 遗漏发生在哪一步

| 路径 | 修复前 | 问题 |
|---|---|---|
| 普通 K-loop（`make_linear` 等） | `source_loop_annotations: {}`、`pipeline_hint_dropped: False` 硬编码 | 源码 `T.Pipelined(num_stages=2)` **完全没有被记录**（不是“记录为未采用”，而是“看不到请求”） |
| guarded-tail（`_extract_ragged_tail` 内） | 同上硬编码 | 同上 |
| block-annotation 路径 | `source_loop_annotations: block.annotations`、`pipeline_hint_dropped: bool(block.annotations)` | 只要出现注解就算“请求被丢弃”，**`num_stages=0` 被误报**；且没有原因字段 |
| staged 路径（两处） | 硬编码 `{}` / `False` | 字段缺失，不构成虚假声明 |

此外 `annotations` 里是 TIR 标量（`IntImm`），直接进计划会触发 `RK_E_SCHEMA: unsupported JSON value`——这也是之前“看起来没记录”的一部分原因。

### 44.2 修复后：请求、未采用决定、原因分别存在哪

计划里每个 matmul 任务的 `attrs.source_k_loop` 现在包含四个字段（同一 helper 生成，两条路径一致）：

| 字段 | 含义 | 例（fixture，`T.Pipelined(num_stages=2)`） |
|---|---|---|
| `source_loop_annotations` | 源码循环上的**原始注解**（TIR 标量已转成 JSON 标量） | `{"num_stages": 2}` |
| `source_pipeline_stages` | **实际请求**的缓冲级数（0 = 未请求） | `2` |
| `pipeline_hint_dropped` | 是否**未采用**（= 请求 > 0） | `true` |
| `pipeline_not_applied_reason` | 未采用原因（常量文本，说明本目标没有软件流水，K 链按 256 宽有序累加发射） | 见 `PIPELINE_NOT_APPLIED_REASON` |

`num_stages=0`：`stages=0`、`dropped=false`、`reason=null`（TileLang 语义：0 表示不启用流水，`loop.py:130-131`）。`software_pipeline_order/stage` 手写列表：折成 `stages≥1` 的请求。生产源码没有 K 循环，因此 `source_k_loop` 整体缺省——**不伪造请求**。

### 44.3 执行字段逐项未变

| 检查 | 结果 |
|---|---|
| 生产计划（decode/prefill） | 哈希仍是 `8daec9f11046…` / `11f43cc45ee9…`，**逐字节相同**（`test_a_source_without_a_loop_carries_no_request` 固定 `11f43cc45ee9`） |
| fixture 计划（带流水提示） | 哈希 `f57a0566…` → `d00d3b7c…`：**仅新增记录字段**；任务数 36、提交 12、设备需求 5,491,264 B、bindings/views/buffers/accounting/publish 全部不变（`verify_record.py` 去掉新字段后其余结构一致） |
| 数值要求、M/N 声明、既有拒绝 | 未改动（同一 contract 与 schedule 代码路径；全套回归无相关变化） |
| 执行库 | 未重建、未改动（`3749e95d…`），因此沿用既有设备证据（§37/§41） |

### 44.4 对 §43 的收窄更正

| §43 的原话 | 更正 |
|---|---|
| “只有串行 K 循环的迭代顺序是真的调度语义” | **改为**：在**已检查的**结构里（block-grid 轴交换、变量重命名、`T.Pipelined` 提示、串行 K 循环），只有 K 循环的迭代顺序被前端当作顺序要求提取；本轮**没有**穷尽串行 M/N 循环、其它合法控制结构或全部 TileLang 表达方式。 |
| “输出 tile 的遍历顺序在源里根本不存在” | **改为**：**已检查的**两种写法（grid 轴交换、变量重命名）canonicalize 成同一计划，这说明这些写法**不要求**不同的执行顺序；不能据此断言源语言里不存在任何表达 tile 顺序的方式。显式 `mn_order` 确实改变计划与命令顺序。 |
| “`T.Pipelined` 是纯性能提示（可安全忽略）” | **改为**：它是 TileLang 的**缓冲级数提示**（`num_stages=0` 表示不启用）；本目标没有实现软件流水，因此后端**记录**请求与未采用原因，而**不是无痕丢弃**。本轮不实现流水。 |

### 44.5 五问答复

1. **遗漏发生在哪一步？** 前端提取的提示记录：普通 K-loop 与 guarded-tail 两条路径硬编码 `{}`/`False`（丢失请求），block-annotation 路径用 `bool(annotations)`（把 `num_stages=0` 误报）；另外 TIR 标量未转换导致计划序列化失败。
2. **修复后请求、决定、原因分别存在哪？** 都在任务 `attrs.source_k_loop`：`source_loop_annotations`（原始请求）、`source_pipeline_stages`（请求级数）、`pipeline_hint_dropped`（未采用）、`pipeline_not_applied_reason`（原因文本）。保存/加载后保持（测试固定）。
3. **哪些记录/身份改变，哪些执行字段保持？** 改变：带提示的 fixture 计划哈希（`f57a0566…`→`d00d3b7c…`，仅记录差异）；保持：生产两份计划逐字节相同，所有计划的 bindings/constants/views/buffers/actions（除新记录键）/submits/publish/accounting 不变，执行库未改。
4. **§43 哪些结论被收窄？** 见 44.4：三条全局断言改为“已检查范围内”的表述；`T.Pipelined` 的说明改为“提示 + 已记录未采用”。
5. **测试结果与仍未验证事项？** 新增 `tests/test_pipeline_hint_record.py`（8 例：缺省/0/非零/手写列表/JSON 标量/计划记录/无循环源/保存加载），全量 **737 passed / 2 skipped / 0 failed / 272 subtests**。仍未验证：guarded-tail 源码中 `T.Pipelined` 的**通过计划**级实测（本轮用 fixture 的 K-loop 路径与 helper 单测覆盖；guarded-tail 的字段由同一 helper 生成，但未构造带提示的 tail 源码计划——因为 inline 生成的源码无法被 TileLang 取到源文本）、staged 路径的提示记录（字段缺省而非虚假声明）、以及流水的实际实现（不属于本轮）。

### 44.6 身份与证据

| 物件 | 值 |
|---|---|
| 代码 | `frontend/tilelang_adapter.py`（`_pipeline_record` + 两条路径接入）、`tests/test_pipeline_hint_record.py` |
| 计划 | 生产 decode `8daec9f11046…`、prefill `11f43cc45ee9…`（未变）；fixture 流水计划 `d00d3b7cb6c1…`（记录变更） |
| 回归 | **737 passed / 2 skipped / 0 failed / 272 subtests** |
| 设备 | 未上板：执行字段与库（`3749e95d…`）均未变，沿用 §37/§41 证据 |
| 证据 | `results/20261103T180000Z-tail-pipeline-record/`（`repros/verify_record.py`、`snapshot/`、`logs/`） |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261103T180000Z-tail-pipeline-record/repros/verify_record.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_pipeline_hint_record.py -q -o addopts=""
```

### 44.7 guarded-tail 源码的最终补测

44.5 的“guarded-tail 未做计划级实测”已由实际文件
`tests/test_tail_pipeline_sources.py` 的补测取代。固定 M=256/K=576/N=3072，
serial、Pipelined(0)、Pipelined(2) 均为 36 task / 12 submit。
前两者不请求流水；后者记录请求 2 级、未采用及原因。去掉提示记录与源码位置后，
执行字段、logical_ops、numeric_requirements 相同，保存/加载保持记录与 hash。
补测后全量为 743 passed / 2 skipped / 0 failed / 281 subtests。
执行库未改变，未重复板测。

<a id="current-status"></a>

## 45. 当前能力、实验结论与重放

这是当前阅读入口；§0—§44 保留调查过程，不应把早期发现当作仍未修复的问题。
本节整理累计代码与测试，不改变执行策略、数值标准或生产默认。

### 45.1 当前控制能力

| 能力 | 入口与落实 | 证据和限制 |
|---|---|---|
| M/N 分块 | TileLang `rknpu.schedule="mn_tiles_v1"`、`tile_m/tile_n`，以及已证明的源块几何；改变 task/view/publish | §37；名义块 1..256，尾块裁剪；显式调度源需证明支持的 K 舍入形式 |
| M/N 遍历 | TileLang `rknpu.mn_order` 改变计划和发射命令顺序 | §34、§37；只有一个 M tile 时可能无差异，未观测设备内部时间序 |
| K 链 | 源 K-loop、显式 FP16 copy/writeback → 升序 256 宽切片、tail、MM/MM_ACC | §34、§38；不支持任意 BK、split-K、归约重排；块内精确求和未建模 |
| 任务排列与提交 | CompileOptions 的 baseline / scheduled_baseline / merged / interleaved / interleaved_reuse / grouped_linear / weight_stationary / weight_stationary_reuse | 后端策略，不是 TileLang 原生循环自动映射 |
| 有界多提交 | grouped 策略与 `linear_group_n_tiles`：0 自动、正整数指定每组 N tile 数 | §41—§42；每 submit ≤64 task，完整 K 链留组内 |
| 输入/权重复用 | 后端 reuse 策略，资格检查与发射字 0x1000/0x2000 | §34、§40—§42；命令位不是带宽测量或寄存器读回 |
| 流水提示记录 | 普通 K-loop 和 guarded-tail 保存请求、未采用决定及原因 | §44；未实现软件流水，staged 提示记录未全面验证 |

block-grid 轴交换与变量重命名不要求不同硬件顺序，canonicalize 成同一计划合理。
不能据此断言所有 TileLang 源结构都无法表达 tile 顺序。未实现通用 GPU
thread/shared/fragment 到 RKNPU 的映射；本次不交付多核或双缓冲。

### 45.2 已修复的问题

- 调度声明不再被 linear 分支忽略；slot 成员上限与 C 对齐；来源标记不能绕过检查（§11—§16）。
- 数值要求、逻辑算子与任务关联随计划保存，重复展开及输入改接保留对应关系（§19—§22）；不代表设备达到容差。
- 提交必须拥有连续任务段，避免 runtime 延迟提交导致错误执行（§27）。
- MM_ACC/EW/LUT 约束、RKIR 未知计算属性、producer/role、签名和标签兼容性检查修复（§28—§30）。
- View 起点加长度、C 预算溢出、不可执行动作拒绝（§31—§32）。
- 未知 `rknpu.*` 拒绝，已识别但未实现的流水提示记录原因（§33、§44）。
- 新分组策略显式声明 `submit_recipe`，旧构造保留前缀回退，结构独立验证（§42）。

这些结论覆盖已复现问题，不是任意计划正确性的证明。

### 45.3 实验采用结论

| 实验 | 结论 | 适用限制 |
|---|---|---|
| decode gate/up 合并提交 | 模型五组对照中 453.5 → 411.9 ms/forward，耗时减少约 9.2%；token 和捕获 logits 相同 | §36；单 prompt、短 decode；设备 JSON 未取回，仅控制台原始记录归档；未重新封存完整 V5 验收 |
| M 分块 256 → 128 | 完整 gate/up 首/中/末层均慢于生产，未采用 | §37、§42；分块、任务数和资源共同变化，不能单独归因于分组 |
| 输入与权重复用 | 发射设置已验证，未得到稳定一致收益，不作为默认 | §35、§40—§42；跨层差值异号不能证明全是噪声，也不排除层相关效果 |
| 遍历与分组规模 | 没有足以采用的性能证据 | §37、§42；未测设备带宽或内部时间序 |

生产默认保持不变。两级 GEMM 的 2 个端到端超差元素及其它已登记 FAIL 保留；
“候选与生产逐位相同”不等于独立参考验收通过。

### 45.4 重放与证据

在 `research/tilelang-rknpu` 中执行（需本机已有依赖与 host 库）：

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/test_schedule_namespace.py tests/test_mn_schedule_source_entry.py \
  tests/test_pipeline_hint_record.py tests/test_tail_pipeline_sources.py \
  tests/test_weight_reuse.py tests/test_weight_stationary_grouping.py \
  tests/test_submit_recipe_and_grouping.py -q -o addopts=""
ctest --test-dir ../custom-ops/build-host-mvp --output-on-failure -j2
```

构建与依赖见 `REPLAY.md`。CTest 使用已有 host build，不等于新构建或板端验收。
历史设备库为 `3749e95d…`；本次整理未重建执行库或运行硬件实验。

证据保留原日期路径，以下均在本地 `results/` 下：

- `20261102T220000Z-tilelang-control-fidelity/`：入口、命令与舍入结构。
- `20261103T020000Z-smollm-model-level/`：模型提交合并对照。
- `20261103T040000Z-mn-schedule-source-entry/`、`20261103T060000Z-mn-numeric-coupling/`：源码准入。
- `20261103T100000Z-weight-stationary/`、`20261103T120000Z-wstat-full-gate-up/`、`20261103T140000Z-wstat-controls-and-layers/`：完整投影与权重复用。
- `20261103T160000Z-source-structure-semantics/`、`20261103T180000Z-tail-pipeline-record/`：源码结构与提示记录。

这些原始证据被 Git 忽略，克隆不会自动带回。版本控制中的测试可重跑；
历史板测复放还需原始输入、package、计划、设备和对应依赖。本次不搬动或改写旧证据。

### 45.5 尚未证明与未完成事项

- 未穷举合法 TileLang 程序、图形态和所有 shape，未证明两套 verifier 对任意输入一致。
- 通用跨 stage 流水、任意布局重排、多核未实现；提示记录不等于流水能力。
- 整链数值判据和块内精确求和模型仍未定义；本次不调整政策。
- `options_to_dict` 尚漏 `fuse_task_chain` 和新增的 `linear_group_n_tiles`，快照不是完整 options identity；本次不顺带修改。
- 缺少可用 transformers 时 tokenizer replay 测试跳过；测试绿不代表此重放已执行。

后续优化应另选具体计算和可验证的调度假设，不自动恢复已结束的权重复用搜索。

### 45.6 本次整理的验证与提交

整理基线为 `7d24032`（更早的意图保真修复已提交）。本次没有新增执行逻辑，
将此前工作区中的已完成实现、测试与历史报告纳入版本控制：

| 提交 | 内容 |
|---|---|
| `cfc29b6` | 源调度命名空间拒绝、数值错误诊断、流水请求记录及真实 guarded-tail 源码测试 |
| `897f419` | weight-stationary/grouped 调度、有界分组、显式 submit recipe 与 Python 复用检查 |
| `b0a1f44` | 缺少可用 transformers 时明确跳过 tokenizer replay 测试 |
| 本节所在文档提交 | §33—§44 原始调查、最终补测、当前汇总与导航；未移动原始证据 |

提交按主题分开，便于审查和回退；如撤销组合，应先撤销依赖相应实现的测试/文档，
并重跑回归，不应只删除实现中的单个字段。未推送远端。

本次在最终代码上重新运行：

- Python：**743 passed / 2 skipped / 281 subtests passed**，68.97 秒。
- 现有 host build 的 CTest：**33/33 passed**，5.73 秒；没有重新构建 C 库。
- `git diff --check`：通过。

未重新运行设备实验或完整模型验收；沿用各节明确列出的历史证据，
不将本次主机回归视为新增设备证明。
