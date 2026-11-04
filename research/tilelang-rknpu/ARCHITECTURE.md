# TileLang → RKNPU 后端：架构与实现契约

本文件是该后端**架构、模块边界与编译契约的唯一权威**。支持域与拒绝条件见
[SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)，验收标准与证据边界见 [VALIDATION.md](VALIDATION.md)，
命令见 [REPLAY.md](REPLAY.md)。逐轮开发、审查与实验记录不在本文件维护，历史见 Git。

文中所有状态描述以当前工作区代码为准；未实现的能力不写成现状。

## 1. 两个边界

后端只承诺两个边界：

1. **编译边界**：从 TileLang/TVM 源生成一份完整、可验证的目标执行方案，并封装为编译产物。
   编译阶段离线执行，不打开设备、不持有 fd、不分配设备内存。
2. **运行边界**：根据编译产物与调用者绑定创建资源，按已核验的方案同步执行。
   资源所有权、同步、并发拒绝与故障隔离全部属于 runtime，编译器不代其做假设。

规划、编码与实例化必须分开；把它们放进同一个包不等于可以互相越权。计算表示优先复用上游，
RKNPU 的精度、收窄点、能力与执行约束仍由目标专用逻辑负责。

## 2. 模块地图与唯一核心调用链

```text
python/rknpu_backend/
  graph.py                     # 计算图访问协议：planner 与载体之间唯一的耦合点
  codegen/
    pipeline.py                # 唯一核心链：normalize → legalize/plan → verify → explain
    descriptor.py              # 唯一 plan descriptor 编码（ctypes 结构与字段映射）
  artifact.py                  # CompiledArtifact / compile_artifact
  api.py                       # 旧入口薄门面（compile_rkir / compile_tilelang / load / save / explain）
  frontend/tilelang_adapter.py # 剩余 adapter 输入的结构识别
  frontend/graph_builder.py    # 剩余 adapter 使用的 RKIR 构造器
  frontend/full_logical_rewrite.py # QKV 源操作改写 → rk_* 两阶段转换（正式入口，见 §5）
  frontend/logical_schedule.py # full-logical 分派与任务链接口（见 §5）
  frontend/rknpu_pass.py       # guarded-tail pass 与任务链（默认策略已接正式入口，见 §5）
  frontend/tilelang_lowering.py # full-logical/guarded-tail 的共用分派器（见 §5.4）
  frontend/structured.py       # 旧结构化原型（未接入生产，见 §4）
  frontend/tilelang_backend.py # TileLang backend 注册与编译产物（opt-in，见 §5.4）
  passes/                      # 合法化、draft、submits、storage、verify（既有实现，未搬家）
  runtime/bindings.py          # Workspace 资源生命周期
  runtime/ffi.py               # ctypes ABI 桥 + codegen.descriptor 的薄重导出
```

**正式入口先做结构分派。** `compile_tilelang` 与 `compile_artifact`（默认
`carrier="rkir"`）对 PrimFunc 先判定 closed full-logical FP16 GEMM 输入范围；命中时直接进入
目标 TIR → 任务实例化 → 任务消费链，不调用 adapter 或 `compile_graph`。其他尚未迁移的
PrimFunc 形式仍经 `codegen.normalize_tilelang` → `frontend.tilelang_adapter.extract`
进入原有核心链：

```text
compile_artifact / compile_tilelang / compile_rkir
  → codegen.pipeline.compile_graph            (normalize → legalize → draft → submits → records → storage → verify)
  → Plan                                      (逻辑不可变；attrs 经 immutable.freeze)
  →（编码时）codegen.descriptor.build_plan_descriptor → C rk_plan_create
```

`tests/test_codegen_pipeline.py` 用打点断言这条链的可观察性质：编译入口只经过一次
`compile_graph`、descriptor 只编码一次、`passes/*` 不导入 runtime/ctypes、`codegen/*` 不导入 runtime。

正式 lowering 路径（§5）有两条明确分支：full-logical GEMM 在源 TIR 上证明数据流后生成
`rk_*` 目标 TIR，再进入任务实例化/任务消费；guarded-tail GEMM 经 localise/instantiate 后使用
同一任务消费链。两者复用同一 planner，但不互相 fallback。full-logical 已接入正式
`compile_tilelang`/`compile_artifact`；其他 adapter 形式仍保留原入口。

## 3. 计算表示

### 3.1 默认：薄 RKIR

RKIR 是**计算表示**，不是第二套后端：它只有薄图容器 + 类型/契约记录 + 严格 JSON schema，
没有独立的 legalizer/planner/encoder。`compile_rkir` 与 `compile_tilelang` 走同一条
`codegen.pipeline.compile_graph`。

- 依赖策略：`import rknpu_backend` 不导入 TVM/TileLang；`ir` 与 `plan` 无设备依赖；
  RKIR JSON 往返只在显式 `compile-rkir` / `load_module` 路径使用。
- 计划对外逻辑不可变：`plan.Action.attrs` 与 `ir.Op.attrs` 递归冻结，写操作抛 `TypeError`；
  `to_dict()` 输出普通容器，磁盘 JSON 与 canonical plan hash 与冻结前逐字节一致。

### 3.2 已退役：TVM 原生（Relax）载体

`compile_artifact(prim_func, carrier="native")` 已显式退役。该路径没有生产或模型调用方，
只存在于 opt-in 试验、工具与测试中。`native_graph.py`、native 专用工具、`RecordBuilder`
及 neutral graph records 已删除；`compile_artifact` 遇到 `carrier="native"` 会给出明确的
不支持诊断，不会静默回退到默认载体。`graph_builder.py` 的默认 RKIR 构造仍服务剩余
adapter 分支。

## 4. 结构化原型与正式 lowering 试点

| 项 | `frontend/structured.py`（原型） | `frontend/logical_schedule.py` + `frontend/rknpu_pass.py`（正式转换） |
|---|---|---|
| 性质 | 旧实验原型，**无生产调用方**；仅 `tests/test_structured_lowering.py`、`tests/test_structured_negatives.py` 引用 | full-logical 已接正式 compile 入口；guarded-tail 的默认策略已接正式入口，也可经注册 TileLang backend 进入 |
| 输入到计划 | 自建 stage 0–4 捕获与证明，产出 `StructuredModule`，再由 `plan_from_module` 组计划 | full-logical：源 TIR 证明→target `rk_*` TIR→任务消费；guarded-tail：IR→IR localise/instantiate→同一任务消费 |
| 与 planner 关系 | 直接给 planner 事实 | full-logical/guarded-tail → `codegen.compile_from_tasks`；复用 layout/pack/storage/template/submit/publish/verifier |
| 定位 | 不作为新路径的隐式 fallback | 两条显式转换，共用 `tilelang_lowering` 分派，不互相 fallback |

原型保留仅为回归与对照，不参与任何默认或注册路径；新链失败时不会退回原型或旧 extractor。

## 5. 正式 lowering 转换

### 5.1 两条链

```text
closed full-logical QKV / SiLU（正式入口）
  TIR（真实 A copy、B copy、fill、GEMM、可选 SiLU store、output copy 节点）
  → schedule_full_logical_source
      定位原语句树 → 以 GEMM 为锚点检查 producer 移动依赖
      → 改写原 region → A/B producer 移入 K 循环
      → 按原位置保留/重定界 SiLU store、fill、output copy → 插入 M/N 循环
  → lower_scheduled_full_logical
      逐项把已调度 tile op 降为 rk_fill/rk_copy/rk_gemm/rk_store
      显式插入 Acc16、padding fill、FP16 转换与写回
  → 复用 pass_instantiate → tasks_from_module
  → codegen.compile_from_tasks → dag.build_from_tasks
  → _Builder.build_instantiated_linear（复用 layout/pack/template/BO/submit/publish/verifier）
  → RKPlan

guarded-tail GEMM（baseline / scheduled_baseline 正式入口）
  TIR
  → BindTarget + MaterializeKernelLaunch(lower_thread_binding=False)   （可选 Simplify）
  → pass_localise     证明局部 tile op 并把它们改写成 rk_gemm / rk_copy / rk_fill extern 调用
  → pass_instantiate  grid 与 K 循环静态展开；guard 在迭代域上判定；
                      单 SeqStmt 的 rk_fill/rk_copy/rk_gemm/rk_store 执行序列
  → tasks_from_module 按执行顺序读回操作、operand、source/destination region、
                      tile/K 和 store rectangle
  → codegen.compile_from_tasks → dag.build_from_tasks
  → _Builder.build_instantiated_linear：消费有序目标操作，复用布局、pack、
                      模板、BO、submit、publish 与 verifier
  → RKPlan
```

直调入口分别是
`frontend.logical_schedule.compile_logical_gemm_tasks(prim_func, *, profile, options, b_is_input, simplify, name)`
和 `frontend.rknpu_pass.compile_guarded_tail(prim_func, *, profile, options, simplify, name)`，
均返回 `(plan, stages, facts, tasks)`。直调与注册 backend 共用
`frontend.tilelang_lowering.compile_tilelang_ir`：它只按“是否存在 serial tile loop”选择一条链，
选中后失败不回退。迁移前 comparator 已由 `tests/fixtures/frozen_plans/` 的冻结 plan 参考替代并删除。

### 5.2 证明规则

这些规则是 guarded-tail 链的语义契约；每条都有对应负例，拒绝即不产生计划。

1. **精确操作表**：`TILE_OPS = {tl.tileop.gemm, tl.tileop.fill, tl.tileop.copy}`。
   证明与改写共用 `_tile_op_role`；表外操作直接拒绝，没有 unknown-op 默认类别。
2. **真实迭代域**：所有被消费的 grid/K/copy/epilogue 循环只经 `_loop_domain` 取域，
   分析与展开共用同一函数；min 必须为 0、step 必须为 1、extent 必须已知。
3. **块坐标映射**：每个轴由 `_prove_tile_index` 求"块部分 = 某循环 × tile extent"并返回表达式；
   A 的行块、B 的列块与输出块必须在同一域上可证相等，K 轴必须等于 K 循环变量，
   GEMM 的三个 region 必须分别是两个共享 tile 与累加 fragment（按 buffer 身份，不按名字）。
4. **逐轴输出证明**：每个输出轴独立决定——guard 被消费为轴界（界必须等于逻辑长度），
   或在域上被证明冗余；没有消费 guard 的轴必须被 grid 精确覆盖；store 索引在域内必须落在输出范围内；
   折零只在单块时接受；两轴不得共用一个 grid 循环。
5. **局部写入覆盖**：逐元素 tile store（两个 staging copy 与 epilogue）的索引必须是本 nest 的
   一个循环变量，其 extent 等于该轴 tile extent，两轴不共用变量——即 iterate → tile 是双射。
6. **region 契约**：所有被消费的 fill/copy/GEMM 调用携带的每个 region 必须 offset 全 0、
   extent 等于整块 shape。
7. **效果闭合**：遍历整个 PrimFunc，每条有状态效果的语句（`BufferStore` / `Evaluate`）必须与
   证明阶段消费的语句按节点身份一致，否则拒绝；`pass_instantiate` 只消费
   `LocalFacts.drained` 记录过的 store。buffer 名字不是证明。
8. **数值与接口契约**：A/B/Y 与共享 tile 必须 FP16、参数行 stride 连续；累加链必须是
   `fill(running,0) → load → gemm → writeback → store` 一条 def/use 链；
   `clear_accum`、部分回写、转置 GEMM、非零 GEMM 累加坐标一律拒绝。
9. **分区**：K 切片必须从 0 连续升序覆盖 `logical_k`；输出 tile 矩形必须逐轴连续分割输出。

QKV 正式分支只接受单个 launch 内闭合的 full-logical GEMM：恰好一次
`tl.tileop.gemm`、一次 zero `fill`、三次 `copy`（A、B、accumulator），A/B 进入 GEMM tile、
accumulator 在 GEMM 后写回参数 C；A/B/C 必须是三个不同参数，region 必须覆盖完整逻辑
tensor 且 offset 为零；GEMM 必须非转置、非 clearing、accumulate 坐标为零；参数必须为静态
FP16 rank-2 且行连续，M/N/K 在 1..4096；可选的 accumulator 更新只能是逐元素
`acc = acc * sigmoid(acc)`。任何额外 call/store、serial tile loop、partial region 或不匹配
shape 都拒绝。该分支不选择 M/N/K 分块，只把逻辑 `matmul` 交给 planner。single launch/1×1
block grid 和“所有计算与 copy 无 enclosing TIR condition”是执行前提，必须检查；只有
launch-thread metadata 等不影响执行结构的字段记录为已忽略 hint。这份闭合 proof 由
`frontend/logical_schedule.py` 独占；adapter 的 legacy logical GEMM/SiLU 分支已删除。

### 5.3 产物与计划

- QKV 分支从真实 fill/copy/gemm/output-copy 节点出发，在源语句树上做合法移动和 region
  改写，再逐项降低为 `rk_*`；任务实例化、任务消费、layout/pack/BO/template/submit/publish
  与 verifier 全部复用 guarded-tail 的现有实现。合成 RKIR 只作为 binding、role 与
  NumericContract 载体，不生成任务。
- SiLU accumulator store 也进入同一源操作路径：调度阶段只重写该 store 的循环范围和访问，
  阶段二将该具体表达式局部降低为 `rk_silu`，再复用既有 sigmoid LUT 与 EW multiply 任务。
- QKV 与真实 SiLU 输入的旧执行描述和 mock 参考已冻结在
  `tests/fixtures/frozen_plans/`。新 QKV 计划并非只改 metadata：Host upload 的放置顺序、
  action/view 编号也会重排，因此完整 plan hash 不同。行为对照以实际读写关系为准：
  NPU 操作数及布局、任务顺序、submit 与 publish 一致，mock 输出一致。
- guarded-tail 分支证明后的操作进入 `rk_fill/rk_copy/rk_gemm/rk_store` extern 参数；
  `tasks_from_module` 保留执行顺序、source/destination region、tile/K、累加读写和 store
  rectangle。计划由 `codegen.compile_from_tasks` → `dag.build_from_tasks` →
  `_Builder.build_instantiated_linear` 从这些操作组装。
- 这里的合成 RKIR Module（matmul，或 matmul+SiLU）只是 binding、role 和 NumericContract
  的载体；planner 不从它的 shape 重建 tile、K 切片、任务顺序或 publish 矩形。
- 任务操作被证明后可以融合或消去，但必须有明确承载：
  A/B zero fill → target 输入 buffer 的 zero-padding 约束；
  A/B guarded staging → upload/pack 的 source region；
  FP16↔FP32 accumulator copy → `NumericContract` 的 per-K writeback 和累加链；
  epilogue `rk_store` → download/publish rectangle。
- 任务消费按目标 TIR 顺序单遍执行，不存在 GEMM 邻域搜索，也不再把操作收集成 `requests` 后重新
  生成任务骨架。每个 buffer 维护当前 zero/value 区域、value version、producer task/action 和实际 view；
  fill、staging copy、accumulator load/writeback、GEMM、SiLU 和 store 依次更新或消费这些状态。
  upload 与 MM/MM_ACC、activation、download/publish 的融合关系在消费实际操作时建立，并直接携带
  task ordinal、action/view 和依赖。无法用该状态表达的重叠、部分覆盖或不完整读取直接拒绝。
- initial fill 的零状态必须覆盖首次 accumulator load；operand fill 的零状态必须覆盖 GEMM 的完整 tile 读取区域，
  而不是只覆盖 tail copy；load → GEMM → writeback → store 的 producer 必须仍是消费点所见的生产者。
- GEMM 的 A/B 读取同时要求：完整读取区域已有状态；逻辑数据区域仍由对应 staging copy 的 value producer
  提供；存在 padding 时 zero state 覆盖完整读取区域。zero 状态不能替代已被后续清零覆盖的 staging value，
  value 消失时 producer 检查也不能跳过。
  原先散落的“附近有一次 fill”“位置序号比较”“禁止区间写入”等检查由该状态机统一取代。
- 上传与下载、pack 布局、模板、accumulator 复用、submit 分组与 accounting 仍由 planner
  决定；任务保持目标 TIR 顺序，upload 的字母序物化和 deferred download 只依据已生成的依赖与
  既有策略，不在去重/排序后假定语义不变。
- guarded-tail 路径的 `source_hash` 固定为 `rknpu-pass`；full-logical 路径保留 source span。

### 5.4 注册与编译产物

`frontend/tilelang_backend.py` 提供显式 opt-in 注册（import 不注册）：

- 本构建（TileLang 0.1.15 + 随附 TVM 0.25.dev0）**没有** rknpu target kind，且 kind 只能在 C++ 注册；
  注册落在 TVM 的 `ext_dev` kind + `rknpu` target key 上，并把 `rknpu` 注册为 target normalizer 名。
- `PassPipeline` 的 lowering 只调用一次 `tilelang_lowering.compile_tilelang_ir`，由它选择
  full-logical 或 guarded-tail 转换；无 try/except、无回退分支，不引用 adapter 或结构化原型。
- 编译产物**随 module 返回**：完整的 canonical plan JSON（内嵌 `plan_hash`）与哈希同时挂在
  返回 module 与其 PrimFunc 上；设备 codegen 重新挂上该产物、只拒绝"未携带产物"的模块；
  `tilelang.engine.lower(mod, target=<ext_dev+rknpu>)` 返回 `CompiledArtifact`，
  `kernel_source` 即该 RKPlan JSON。公开 API：`compile_to_plan` / `plan_from_artifact` /
  `plan_from_module` / `plan_from_json`（反序列化复用 `Plan.from_dict`，重算并校验 `plan_hash`）。
- 进程全局 lowering 日志只用于诊断；从产物还原计划不依赖它。
- `tvm.IRModule.inspect_source` 是本构建缺失方法的最小 shim（该版本 `engine/lower.py` 会调用它），
  仅在注册时安装、退注册时移除，只回答携带 RKPlan 的模块。它不构成对其它 backend 产物链的兼容性证明。

## 6. 产物边界

| 概念 | 归属 | 说明 |
|---|---|---|
| `Plan` | 编译产物 | 权威表示是**一份**已验证计划；attrs 冻结。计划说明由模块级函数 `explain_plan(plan)`（`codegen/pipeline.py`）生成 |
| descriptor | 编码投影 | 在 `instantiate` / `create_plan` 时编码，不跨 workspace 缓存，artifact 不持有 |
| `CompiledArtifact`（`artifact.py`） | 编译边界 | 设备无关；`explain()` 在 `explain_plan(self.plan)` 之上附加 compile/source 段；`instantiate(fd=…)` 才需要调用者持有的 fd；`export_plan(path)` 写 JSON |
| `Workspace` | 运行边界 | 每个 workspace 独立；共享常量校验、C plan 深拷贝、ownership/BUSY/poisoned 规则 |
| `rk_plan.h` / schema | 既有 ABI | Plan schema 2（kind `rkplan`）、RKIR schema 1 未变 |

正常使用是「编译产物 → 实例化 → 执行」：持久化 JSON 只服务导出、离线部署与重放，不是内存内调用的必经路径；任何「读回 JSON 再规划」的做法都不是本架构的一部分。

C 侧与 runtime 的语义边界不变：预提交的绑定/非有限值/上传失败必须安全返回，执行器错误进入
poison 并保留命令与数据 BO；支持的失败后行为是隔离/停止，恢复不在范围内。

## 7. 已采用与已撤回的默认变更

这些决定只说明"当前默认是什么、为什么不做某些方向"，不附测量数字；测量属于历史证据（见 VALIDATION §4）。

**已采用（默认路径包含）**：C workspace 包络归因与静态预分配、Host scratch 预算闭环；下载路径无
per-call malloc/free、有限值扫描与拷贝特化；guard slack 精确 8-byte 扫描；QK/PV 静态 repeat
shared group（`chain_shared`）；packed gate/up GEMM + 原 FP32 Host SwiGLU；shared BO 上传同步合并；
fixed-layout 16×16 CUBE/WEIGHT fast pack；上传策略创建时快照与 >128 上传安全回退；`PlanCall`
ctypes binding 描述符复用；非阻塞 descriptor lock（保持 `RK_E_BUSY`）；mapped QKV attention input +
capture-only split_heads；C/Python finite 检查同 buffer 去重。

**已撤回 / 未采用（不得当作当前能力）**：全设备 FFN 作为默认（独立数值政策，仅保留 `_deviceffn`
显式清单）；用 `co_row_ops` 复用 RMSNorm；fused residual + RMSNorm Host action；下载同步合并；
residual + reshape + mean 尾部融合；prepared ctypes binding arrays；通用任意 stride 输入
（mapped QKV 只接受显式 row/column slice 与 transpose 关系）。

**明确非声明**：mapped QKV 的端到端收益在 b8 接近噪声边界；不同 campaign 的加速比不可相加；
冷启动是否更快未声明；device FFN、通用 Softmax/RMSNorm 下沉、任意 stride、多核并发、KV cache
下沉与未知故障恢复均未声明；厂商 RKNN 路径与当前 packed per-logit 阈值不等价。

## 8. 支持域

见 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)。两域必须分开阅读：默认 adapter 路径较宽；
正式 lowering 覆盖 closed full-logical FP16 GEMM 与 guarded-tail FP16 GEMM。前者进行源操作
两阶段改写；后者叠加 §5.2 的九条规则并使用同一顺序任务消费链。两者都受 planner 的
M/N tile 不超过 256×256、K 分块大小固定为 256 的约束；逻辑 K 可包含多个分块及尾块。

## 9. 尚未完成的迁移

- closed full-logical FP16 GEMM 已从正式 `compile_tilelang`/`compile_artifact` 入口迁移到目标 TIR
  与任务消费链。guarded single-trip、三 buffer final-only、显式 round-trip 以及
  `baseline / scheduled_baseline / weight_stationary / weight_stationary_reuse /
  grouped_linear / merged` 策略也进入同一 guarded 源操作 lowering；无 guard 的
  final-only/round-trip K-loop 沿实际 loop/copy/GEMM 关系进入任务链。adapter 的
  `_extract_ragged_tail`、`_detect_k_block/_k_loop_module` 与对应分派均已删除。
  两个真实 SwiGLU FFN、投影链 `silu_proj`、双输出 `branch_outputs` 和 scalar/channel
  broadcast 输入已改为源端逐语句改写并进入同一有序任务 consumer：实际
  fill/copy/cast/sigmoid/GEMM/store 保留 token 和 source/destination region；cast/copy
  融合要求整区域、索引对应、dtype 一致，否则拒绝。显式 sigmoid 与后续乘法保持独立
  op 归因；多个 output copy 各自产生 `rk_store` 并驱动独立 download/publish。rank-0
  或 rank-1 broadcast 只从实际读取及其列循环关系承接，carrier 保留原始 rank/shape，
  task 记录每操作数广播映射；上传展开仍复用既有 broadcast pack/expanded view。
  adapter 的 `_extract_ffn_branch`、`_extract_static_dataflow`、
  `_extract_branch_outputs`、`_extract_broadcast_tail` 已删除。多 launch 源由
  `staged_launch` 按实际 `T.Kernel` 作用域分割；每个 launch 保留阶段契约、读写区域
  和提交边界。两 launch `make_two_stage` 已由实际 per-stage 任务在同一个 `_Builder`
  中消费；声明 M/N 调度先在真实 launch TIR 上落实，中间值作为设备全局 value/view
  登记，读取继承其 producer，不创建 Host 上传下载；MM/MM_ACC 选择由实际 Acc→Cl 与
  Cl→running copy 状态建立。projection 的 G 由四个 atom-aligned 256x64 设备子 view
  写入同一 CUBE 存储，stage3 的 256x256 K 读取依赖全部四个 producer；Python staged
  构建器与 C verifier 均做有界覆盖验证。carrier 从实际逐 launch 操作链建立，旧完整
  staged 图提取及 `extract_staged → compile_graph` 旁路已删除。基础 TIR、表达式和
  契约工具迁入 `tir_utils`，生产代码不再导入 `tilelang_adapter`；正式 compile/
  artifact/直调入口不再回退 adapter。
  elementwise 的 Add/Mul 只接受直接 load 操作数，sigmoid 必须直接读取一个值；SiLU
  必须是同一次读取与其 sigmoid 的乘积。未支持的嵌套表达式直接拒绝。staged 的
  elementwise 逐个承接原 copy/store 顺序及双方区域，读取消费时的值版本与 producer，
  不再由 opcode/参数/tile 摘要补建整段任务。fill 的实际零值、区域及 producer 与缺失值
  分开保存；copy 仅在区域核验成立时继承值/view/producer，零初始化融合到首个 MM，
  writeback copy 建立下一次 MM_ACC 的输入。launch 内外未承接的调用直接拒绝。
  CLI 和模型构建调用尚未整体切换。
- 已知迁移差异：无注解三 buffer final-only 多 grid 源现在保留源码实际 tile
  （例如 128×128 的 9 个输出 tile），不再被旧 adapter 按历史 256×256 默认网格重排。
  这是无声明输入的资源行为变化，单独报告。
- 显式 M/N 注解在 K-loop 源 TIR 调度阶段落实：注解 tile 与源物理 tile 不同时，先对
  实际 tile buffer、grid loop、copy/GEMM region 做局部重分块，再进入任务实例化。
  当前仅支持 M/N/K 均能被声明 tile 精确覆盖的形式；带 M/N/K tail 的显式重分块会返回
  具体 unsupported 诊断，不会把注解静默降为来源记录。
- 其他 PrimFunc 形式——包括 full-logical activation、多节点图、模型包、多 block stage、
  broadcast、hybrid——仍走 adapter；其退出需要逐类迁移，不能通过失败回退把新链作为旁路。
- `frontend/logical_gemm.py` 与 adapter 的 legacy logical GEMM/SiLU 分支已删除；正式 compile
  入口的 full-logical no-activation、SiLU、source-scheduled 单 GEMM 均进入 target TIR 任务链。
- native Relax 载体已退役；其专用模块、工具和 neutral builder 结构已删除（§3.2）。
- `frontend/structured.py` 未接入生产，仅保留为回归对照。
- 设备验证范围有明确身份限定：可引用的直证是
  `results/20261105T040000Z-pass-pilot/device-pass.json` 记录的 RK3588 运行，其执行计划为
  `43cd64c54b328f56aef9ed44e01a7478774cb59b9e3c1627e1222cec858376c6` 与
  `f3eb162fa79a56c07566ff34067e0df8c075b30107acbcbd49d653b32b75881d`（36 tasks / 12 submits）。
  当前 pass 链对等价输入生成的计划（例如 `52bf1a4b9ad981dcc4e25a94de768d4e409d612fa9f8aef9a9c86883168f7564`）
  与该历史计划只在执行字段上等价
  （`results/handoff-final-lead-audit-r4/device-equivalence-reviewed.json`，差异为记录属性），
  因此当前 lowering 的设备侧状态是 verification gap，而不是设备验收通过。

## 10. 验证与证据

验收标准、证据类别与各自能证明什么见 [VALIDATION.md](VALIDATION.md)；
命令见 [REPLAY.md](REPLAY.md)；原始证据的组织与查找方法见 [results/README.md](results/README.md)。
