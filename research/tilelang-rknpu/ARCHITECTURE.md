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
  frontend/tilelang_adapter.py # 默认路径与 native carrier 共用的结构识别
  frontend/graph_builder.py    # 载体缝：默认 RKIR 载体 / 试点记录载体
  frontend/native_graph.py     # 原生 Relax 试点载体（非默认，见 §3.2）
  frontend/rknpu_pass.py       # 正式 pass 试点链（opt-in，见 §5）
  frontend/structured.py       # 旧结构化原型（未接入生产，见 §4）
  frontend/tilelang_backend.py # TileLang backend 注册与编译产物（opt-in，见 §5.4）
  passes/                      # 合法化、draft、submits、storage、verify（既有实现，未搬家）
  runtime/bindings.py          # Workspace 资源生命周期
  runtime/ffi.py               # ctypes ABI 桥 + codegen.descriptor 的薄重导出
```

**默认入口仍走 adapter。** `compile_tilelang` 与 `compile_artifact`（默认 `carrier="rkir"`）
都经 `codegen.normalize_tilelang` → `frontend.tilelang_adapter.extract` 识别结构，
再进入同一条核心链：

```text
compile_artifact / compile_tilelang / compile_rkir
  → codegen.pipeline.compile_graph            (normalize → legalize → draft → submits → records → storage → verify)
  → Plan                                      (逻辑不可变；attrs 经 immutable.freeze)
  →（编码时）codegen.descriptor.build_plan_descriptor → C rk_plan_create
```

`tests/test_codegen_pipeline.py` 用打点断言这条链的可观察性质：编译入口只经过一次
`compile_graph`、descriptor 只编码一次、`passes/*` 不导入 runtime/ctypes、`codegen/*` 不导入 runtime。

正式 pass 试点是**并列的第二条编译路径**（§5），它复用同一 planner 的组装函数但不经
`compile_graph`；两条路径的对象、入口与支持域都不同，不能互相替代。

## 3. 计算表示

### 3.1 默认：薄 RKIR

RKIR 是**计算表示**，不是第二套后端：它只有薄图容器 + 类型/契约记录 + 严格 JSON schema，
没有独立的 legalizer/planner/encoder。`compile_rkir` 与 `compile_tilelang` 走同一条
`codegen.pipeline.compile_graph`。

- 依赖策略：`import rknpu_backend` 不导入 TVM/TileLang；`ir` 与 `plan` 无设备依赖；
  RKIR JSON 往返只在显式 `compile-rkir` / `load_module` 路径使用。
- 计划对外逻辑不可变：`plan.Action.attrs` 与 `ir.Op.attrs` 递归冻结，写操作抛 `TypeError`；
  `to_dict()` 输出普通容器，磁盘 JSON 与 canonical plan hash 与冻结前逐字节一致。

### 3.2 试点：TVM 原生（Relax）载体

`compile_artifact(prim_func, carrier="native")` 是显式选择，产出 `tvm.relax` 图代替 RKIR。

- **它仍依赖 adapter 做识别。** 结构识别只有 `tilelang_adapter.py` 一份；8 个建图点经
  `graph_builder.active_builder()` 抽象，`RecordBuilder` 产出无容器记录，再交给
  `native_graph.build` 生成 Relax 节点。原生载体不复制识别算法。
- 目标契约以函数级 `rknpu_contract`（`DictAttrs`）承载普通 Relax 不能表达的部分
  （per-value 角色/绑定名、per-op `numeric_contract`、累加器 dtype、收窄点、K 循环、M/N 分块、
  尾块契约、stage、`b_layout`、源 span）。契约由 `graph_signature()` 与图同构地关联，
  每次使用前重算并交叉校验。
- 允许的 pass 集合固定为 `Normalize` / `CanonicalizeBindings` / `DeadCodeElimination`，
  每个 pass 后立即重校验；集合之外或改变签名的变换一律拒绝。
- **未采用为默认**：既有 `compile_rkir`、`ir` 模块构造器、RKIR JSON 往返与 `compile-rkir` CLI 都必须继续
  可用，因此迁移不删除任何既有实现，薄 RKIR 仍是默认计算表示。
- **当前依赖关系**：载体不能脱离 adapter 单独存在——`tilelang_adapter.py:41` 直接导入
  `graph_builder.active_builder`，识别逻辑只有一份；`graph_builder.py`（载体缝）与
  `native_graph.py`（Relax 构建）在 opt-in 路径上都是必需依赖。它还依赖锁定 TVM 中的算子覆盖：
  例如 `hardsigmoid` 没有直接 Relax 算子，原生载体按拒绝处理而不是组合算子。
- 它与默认路径共享同一套合法化、规划、存储与编码算法，因此不构成"两套生产后端"。

## 4. 结构化原型与正式 pass 试点

| 项 | `frontend/structured.py`（原型） | `frontend/rknpu_pass.py`（正式试点） |
|---|---|---|
| 性质 | 旧实验原型，**无生产调用方**；仅 `tests/test_structured_lowering.py`、`tests/test_structured_negatives.py` 引用 | 正式 opt-in 编译路径；可直调，也可经注册的 TileLang backend 进入 |
| 输入到计划 | 自建 stage 0–4 捕获与证明，产出 `StructuredModule`，再由 `plan_from_module` 组计划 | 真实 IR→IR：`pass_localise` 改写 IR，`pass_instantiate` 静态展开成任务序列，读回实例后交给既有 planner |
| 与 planner 关系 | 直接给 planner 事实 | 任务序列 → `tasks_from_module` → `instances_from_tasks` → `codegen.compile_from_tasks` |
| 定位 | 不作为新路径的隐式 fallback | 唯一"新转换链"入口 |

原型保留仅为回归与对照，不参与任何默认或注册路径；新链失败时不会退回原型或旧 extractor。

## 5. 正式 pass 试点（opt-in）

### 5.1 链

```text
TIR
  → BindTarget + MaterializeKernelLaunch(lower_thread_binding=False)   （可选 Simplify）
  → pass_localise     证明局部 tile op 并把它们改写成 rk_gemm / rk_copy / rk_fill extern 调用
  → pass_instantiate  grid 与 K 循环静态展开；guard 在迭代域上判定；单 SeqStmt 任务序列
  → tasks_from_module 从 IR 读回任务记录
  → instances_from_tasks  归一为 m_major + 升序 K 的 tile / K 切片实例，并证明分区
  → codegen.compile_from_tasks → dag.build_from_tasks → _Builder.build_scheduled_linear(instances=…)
```

入口：`frontend.rknpu_pass.compile_guarded_tail(prim_func, *, profile, options, simplify, name)`
返回 `(plan, stages, facts, tasks)`。

### 5.2 证明规则

这些规则是这条链的语义契约；每条都有对应负例，拒绝即不产生计划。

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

### 5.3 产物与计划

- 证明后的块坐标进入 `rk_gemm` extern 参数；实例归一为 m_major + 升序 K，
  因此整体交换 grid 轴的程序给出逐字节相同的计划。
- 计划组装仍复用既有实现，但不是"逐操作直译"：本链先用已证几何构造一个**单 matmul 的 RKIR Module**
  （bindings、numeric contract、`source_schedule` / `source_k_loop` 记录），再由
  `codegen.compile_from_tasks` → `dag.build_from_tasks` → `_Builder.build_scheduled_linear(instances=…)`
  按 tile / K 切片实例组装计划。
- 因此任务序列不是计划的逐步来源：上传与下载、pack 布局、模板、accumulator 复用、submit 分组与
  accounting 仍由 planner 决定；任务 IR 只贡献实例几何、操作数名与已证事实。
- 该路径的 `source_hash` 固定为 `rknpu-pass`，与原 adapter 路径的计划身份分开。

### 5.4 注册与编译产物

`frontend/tilelang_backend.py` 提供显式 opt-in 注册（import 不注册）：

- 本构建（TileLang 0.1.15 + 随附 TVM 0.25.dev0）**没有** rknpu target kind，且 kind 只能在 C++ 注册；
  注册落在 TVM 的 `ext_dev` kind + `rknpu` target key 上，并把 `rknpu` 注册为 target normalizer 名。
- `PassPipeline` 的 lowering 只调用一次 `compile_guarded_tail`，无 try/except、无回退分支，
  不引用 adapter 或结构化原型。
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
正式 pass 试点只覆盖 guarded-tail FP16 GEMM，且叠加 §5.2 的九条规则与 planner 的
M/N tile 不超过 256×256、K 分块大小固定为 256 的约束；逻辑 K 可包含多个分块及尾块。

## 9. 尚未完成的迁移

- 正式 pass 试点仍是 opt-in：没有 CLI 子命令，也不在生产默认路径上；`compile_tilelang` 与
  `compile_rkir` 的行为与支持域不变。
- 试点支持域窄于默认路径：多节点图、模型包、多 block stage、broadcast、hybrid 等仍只走默认路径。
- 原生 Relax 载体是显式 opt-in，不是默认，也不在当前范围内替换薄 RKIR（§3.2）。
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
