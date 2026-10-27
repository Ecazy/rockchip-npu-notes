# TileLang → RKNPU 后端重构：执行规范与实施记录

日期：2026-10-27。状态：主线 R0–R3、R6 `passed`（含一次审核后的边界修复）；R4 试点 `passed`；R5 `not_adopted`（保留薄 RKIR，理由见第 15.3 节）。本文为执行智能体的任务规范兼唯一实施记录；目标架构见 [refactor-overview.md](refactor-overview.md)。第 1–14 节保留原始要求，第 15 节记录实际实现；两者冲突时以第 15 节的实际决定为准并已在此说明。

## 1. 范围、优先级与文档规则

必做主线：将现有规划及编码职责收敛到 codegen，提供直接可实例化的编译产物，保持运行时和兼容入口行为。

必做评估：建立不经过 RKIR 图构造的 TVM 直接路径，判断是否值得迁移。默认切换和删除独立 RKIR 是有条件结果；直接路径未能降低维护负担时，保留薄 RKIR 并记录结论，不阻塞必做主线交付。

本文覆盖的是架构重构，不实施新算子、新布局/提交配方、跨计划设备驻留、多核、异步执行、新数值政策或整模型联合规划器。已有或其他会话正在开发的这些能力不由本任务撤销，也不自动计入本次成果。

规则优先级：用户本次授权与约束 → 本重构 specific → 本重构 overview → 原有设计中的适用契约。原 [overview.md](overview.md)、[specific.md](specific.md) 是历史架构依据；其中必须使用独立 RKIR、公开 RKPlan 等组织要求可由本任务替换，数值、安全、能力及兼容要求不得因此放宽。

文档只维护本文件和配套 overview：

- 所有接口、文件安排、实现决定、进度、验证、未完成项与交接写入本文。
- 不新增重构 README、设计说明、状态文件、实验报告、ADR 或交接文档；也不将本任务细节分散写入现有这类文档。
- 代码、测试、fixture、脚本以及原始 `.log` / `.json` / 数组证据可以正常创建。证据目录不另生成 Markdown 报告，本文链接到实际文件。
- 模块内必要 docstring 和注释可保留，但不能代替本文的契约记录。
- overview 仅在架构目标或决策边界确有变化时更新，不写实施流水账。

## 2. 开工检查与工作区隔离

开始实现前执行并将结果摘要填入第 15 节：

```bash
git status --short
git rev-parse HEAD
```

识别当前 HEAD、已有 dirty diff 与其他会话新增文件。不能以本文日期或某个历史 commit 代替实际基线；不得 reset/clean、覆盖、批量纳入或提交不属于本任务的修改。

若主线程正在修改相同核心文件，优先使用独立 worktree/checkout 承载本次变更，并说明基于哪个已提交版本。不得在未确认所有权时复制或移走主线程未提交文件；必须依赖其变更时，明确记录依赖后协调，而非擅自重写。不要自动创建或使用子智能体。

只读核对以下入口，不要求逐文件重读全部历史代码：

| 现有位置（相对本目录） | 核对内容 |
|---|---|
| `python/rknpu_backend/api.py`、`__init__.py` | 编译入口、返回值、公开导出 |
| `python/rknpu_backend/frontend/tilelang_adapter.py`、`python/rknpu_backend/ir.py` | 语义识别与 RKIR 构造边界 |
| `python/rknpu_backend/passes/{infer,legalize,dag,schedule,storage,verify}.py` | 类型、能力、任务及存储规划 |
| `python/rknpu_backend/{plan,serialize}.py`、`schema/` | 内存计划与持久化耦合 |
| `python/rknpu_backend/runtime/{bindings,ffi}.py`、`python/rknpu_backend/{models,hybrid}.py` | descriptor 编码、资源与生产入口 |
| `python/rknpu_backend/cli.py`、`tools/compile_models.py` | 兼容调用方与模型编译粒度 |
| `../custom-ops/include/rk_plan.h` 及相应 C 实现 | ABI、独立验证、深拷贝和生命周期 |

复用当前锁定且可追溯的 TileLang/TVM 环境。不升级依赖、不从 main 获取新版来解决本次适配，不重新建立通用环境或基准平台。按当前有效命令执行 Host 基线，已有失败单列。

## 3. 已有隔离实验及其证据边界

2026-10-27 的隔离实验复用了现有 TileLang 提取结果：A 经 RKIR 规划，B 将提取结果转换为 TVM Relax 原生图，再由兼容视图调用同一规划器。

| 输入 | NPU task 数 | 完整计划比较 |
|---|---:|---|
| 单 MatMul | 1 | 一致，含 plan hash |
| MatMul → SiLU | 3 | 一致 |
| ragged K=257 | 2 | 一致 |
| staged projection | 22 | 一致 |
| staged branch / 多输出 | 6 | 一致 |
| staged FFN | 30 | 一致 |

六个样例 shape 推断与 TVM JSON 往返通过；Normalize、CanonicalizeBindings、DeadCodeElimination 的 18 个组合通过结构和属性保留检查。此结果不证明任意变换后数值属性仍语义有效。

TVM 原生推断拒绝错误 MatMul 维度，但接受一般广播和 K=4097；项目支持域限制仍须目标验证。strict 数值拒绝仍由现有 legalizer 产生。普通 Relax MatMul 的 FP16 输出本身没有表达有序 K 分片 FP16 写回契约。

本实验没有移除提取器对 RKIR 的依赖，没有独立实现直接编译，没有板端执行或维护成本对照。实验脚本原位于临时目录，不作为本次可重放交付的必要依赖；本轮按下面规范将必要验证收敛到正式测试或脚本，不额外建立实验框架。

## 4. 职责与依赖方向

| 阶段 | 输入 / 输出 | 允许的副作用 |
|---|---|---|
| 目标归一化 | 真实 PrimFunc → 受限计算图 + 目标契约 | 无设备副作用 |
| 合法化与规划 | 计算图 + profile/options → 完整内部计划 | 无设备副作用 |
| 验证与编码 | 内部计划 → 运行时 descriptor / 导出表示 | Host 内存操作；不创建设备资源 |
| 编译产物 | 包装已验证计划及编译身份 | 不拥有 fd、BO 或 native plan handle |
| 实例化 | artifact + 调用者 fd + 常量 → workspace | 按已有路径分配并绑定资源 |
| 执行 | workspace + runtime bindings → 结果 / 错误 | 按已有同步执行与发布契约 |

限制：

- planner 不导入 runtime 来获取设备信息或分配 BO；codegen 不通过实际提交选择合法候选。
- serializer 不参与图合法化、tiling 或调度；正常内存路径不得通过 JSON/dict round-trip 驱动阶段间调用。
- runtime 不从未合法化的 TVM 节点重新推断执行方案，也不执行新的隐式 fallback。
- Python 编译无需加载设备执行库；C descriptor 独立验证可作为显式离线验证步骤，但不能使基础 import/compile/explain 强依赖共享库。
- 保留现有 profile 来源、hash 与环境匹配检查，区分离线身份检查和实例化时设备检查，不顺手放宽。

## 5. 代码组织与单一权威表示

建议在 `python/rknpu_backend/` 内建立：

```text
artifact.py                 # CompiledArtifact 公共门面
codegen/
  __init__.py
  pipeline.py               # 组合归一化、规划、验证和编码
  descriptor.py             # 共享 descriptor 编码入口
frontend/                   # 现有语义接入，逐步支持原生图输出
passes/                     # 复用现有合法化及规划算法
runtime/                    # 资源绑定与执行
```

该目录是建议，不是必须制造的新层级。`Plan`、passes 不要求立即搬家；现有 `plan.py` 可继续承载内部执行计划及兼容类型。只有能减少职责混杂才移动实现，旧导入路径使用薄重导出。

内部执行计划至少继续表达 bindings/常量、buffer/view、布局与 padding、actions/deps、存储与独立 BO 约束、submits、publish、预算与来源身份。优先保留现有字段和验证器，不重新设计一套平行 RKPlan。

以一份已验证内部计划为编译产物的权威执行表示。JSON、解释结果和 descriptor 是投影；不得让它们各自参与修改调度。兼容层与新入口必须调用同一规划器和编码器。

计划对外逻辑不可变：特别核查现有 frozen dataclass 中的嵌套 attrs/dict。通过自有拷贝、只读访问或必要的边界复制防止 explain/report 修改执行语义，不要求为此重写全项目容器体系。

descriptor 编码可延迟到实例化；ctypes 数组、字符串及临时结构在 native 调用结束前必须保持存活。不把地址绑定 descriptor 长期缓存在可跨 workspace 复用的 artifact 中。

## 6. 编译产物与公共接口

采用新增入口而非静默改变现有返回类型，目标接口为：

```python
def compile_artifact(prim_func, *, profile, options,
                     b_is_input: bool = False) -> CompiledArtifact: ...

class CompiledArtifact:
    def explain(self) -> dict: ...
    def instantiate(self, *, fd: int,
                    constants: dict[str, "numpy.ndarray"]) -> Workspace: ...
    def export_plan(self, path: str) -> None: ...
```

要求：

1. `compile_artifact` 接受当前入口支持的真实 TileLang 对象，不执行 GPU pipeline，不解析源码/IR 文本。
2. artifact 保存 profile、options 的不可变编译快照及来源身份；不保存输入数组、TVM callable、设备地址或 fd。编译完成后执行不依赖原 PrimFunc 存活。
3. 若必须为诊断保留 TVM 图，作为可选、可释放的编译诊断对象，不成为实例化或导出的依赖，不默认保留源模块和全部中间图。
4. `explain()` 无设备访问，返回不会反向修改内部计划的结果；数值与 fallback 报告来自实际产物，不能用固定字符串伪造。
5. `instantiate()` 复用生产 workspace 创建和执行链；fd 继续由调用者持有。每次实例化使用独立资源，artifact 本身不持有或共享可变 descriptor。
6. 正常编译→实例化→执行不保存/加载 JSON，也不要求用户先调用 export。
7. `export_plan()` 输出既有 RKPlan 兼容格式，不额外引入 artifact 文件格式。原有 load_plan 后的运行继续可用。
8. artifact 不提供隐藏全局 workspace 或隐式设备打开；不得绕过既有 ownership/BUSY/poisoned 检查。

在不破坏上述契约的情况下可增加必要的私有工厂或只读属性；不要建立完整 artifact 插件/缓存平台。

## 7. 兼容入口与依赖隔离

| 现有入口 | 本轮处理 |
|---|---|
| `compile_tilelang(...) -> CompileResult` | 保持 `.plan/.diagnostics/.report/to_dict()` 使用方式，通过同一核心 pipeline 生成 |
| `compile_rkir(Module, ...)` | 保留 legacy 输入；不建立第二套 legalization/planner |
| `Plan`、`load_plan/save_plan/explain_plan` | 保留导出及离线重放语义；对接内部计划 |
| `Workspace.create(plan, ...)` | 继续有效，与 artifact.instantiate 共享资源创建实现 |
| CLI compile/validate/explain/run | 旧参数与退出码保持；内部可改用 artifact，无需新增并行 CLI 家族 |
| `PlanCall` / `HybridRunner` | 复用新的共享编码边界，保留已有生产热路径与归因，不无故替换成较慢入口 |

原有 RKIR 独立编译、plan 加载/执行及基础包 import 在未安装 TileLang/TVM 时继续工作。原生图迁移不得使 `import rknpu_backend` 顶层导入 TVM。TVM 相关模块按需加载；legacy 输入若需要适配，优先使用薄访问协议复用同一 planner，不能因此复制算法或强制安装 TVM。

兼容类型可暂时公开，避免本轮做破坏性删除。不得同时长期保留“旧生产后端”和“新生产后端”；差分 oracle 可存在于测试边界，但正式代码路径共享核心。

若无法在合理复杂度内同时实现默认原生图与无 TVM legacy 编译，记录成本并保留薄 RKIR；不要未经决策撤销已有依赖隔离契约。

## 8. TVM 原生图与目标契约

### 8.1 必须直接生成原生表示

先选择锁定版本中适合的 TVM 表示和归一化位置，说明为什么适合现有算子及分析。Relax 是已实验的候选，不要求强行留在原始 TIR，也不增加另一种自定义 dialect。

直接路径必须：

- 复用已有真实 AST 语义识别与上游 visitor/analysis；允许提取共同的识别 helper。
- 直接创建原生节点，不先实例化 RKIR Module/Value/Op 再转码。
- shape、dtype、算子和依赖由原生图承担；不将完整 RKIR 或完整图 JSON 放进一个 attribute。
- 访问、归约、初值、线程通信、predicate、mask 及 cast 的证明责任保持。
- 不按模型名、变量名、固定参数个数或整个 FFN 的固定外形定义新通用支持域。

### 8.2 仍需保留的目标语义

| 信息 | 表达与检查责任 |
|---|---|
| accumulator dtype、归约要求 | 源识别保存，目标 legalizer 检查 |
| 每个 FP16 收窄点及 K 写回边界 | 显式图结构或可验证的目标契约，不能仅用输出 dtype 代替 |
| LUT 近似、finite/值域责任 | 明确授权与检查方式，拒绝未声明近似或 clamp |
| 常量/运行时输入/导出值角色 | 边界元数据，和真实图引用一致 |
| 源位置与拒绝阶段 | 诊断元数据，转换后可追溯 |
| 合法物理布局、设备能力 | 目标分析与规划，不冒充通用 TensorStructInfo 已证明 |

普通 Relax MatMul 不能单靠属性附注就允许任意通用融合、重排或 cast 消除。选择明确的目标归一化阶段及允许 pass 集合；会改变关联节点或舍入语义的 pass，必须重建/验证契约，否则拒绝使用。

元数据不能仅以可变变量名或偶然遍历编号永久关联。对重命名、复制与合法变换使用稳定关联或显式传播；不能把本次实验中的名称兼容视图直接视为生产方案。

无须使所有 TVM 通用 pass 都适用。用最小必要 pass 完成迁移，不另建通用优化平台。

### 8.3 是否替代 RKIR 的判定

直接路径评估必须完成；默认迁移仅当以下条件同时成立：

1. 代表性正例与相邻负例成立，计划/descriptor 对照可解释。
2. 不丢失数值、访问、副作用、来源及支持域验证。
3. 后续 planner 不依赖重建完整 RKIR；兼容视图只提供必要访问，不积累另一份全图状态。
4. 删除/简化的代码、测试与契约维护负担大于新增适配负担，有具体对照；不以 LOC 单项决定。
5. 基础依赖隔离、legacy 输入和重放能力保持。

通过则采用原生图默认路径并将 RKIR 限定为 legacy 入口；未通过则保留薄 RKIR，将具体原因写入第 15 节。两种结论都可交付，不允许用伪原生包装声称迁移成功。

## 9. runtime 与性能不变量

- 复用既有 C verifier、emitter、executor 和 workspace；不新增寄存器生成器或提交器。
- C plan 深拷贝与 descriptor 生命周期保持；Python 不因为已验证就绕过 C 检查。
- 编译产物不保存 BO 身份；实例化再检查 fd/domain、容量、地址、布局和预算。
- 独立 workspace 不共享可变 binding cache；既有 PlanCall descriptor lock 覆盖指针修改至同步 native 调用结束，仍以非阻塞 BUSY 拒绝并发。
- 配置在编译/创建阶段快照，不在运行中改变上传策略、数值或提交安排。
- 保留 finite/alias/shape/dtype/contiguous/writable 校验、padding/guard 与失败不发布。
- 保留 shared-context quarantine 及 poisoned/BUSY destroy 下的所有权，不引入自动 reset/恢复继续。
- capture-only 不能喂给普通节点或普通模型输出；重构不能使诊断工作进入默认计时。
- 保持已有 packed FFN、shared group、上传同步、fast pack、描述符复用、mapped QKV 等有效优化，不以统一 API 为由退回逐任务或重复序列化路径。

## 10. 持久化、身份与差分

持久化格式维持现有严格解码、未知字段/非法值拒绝、canonical hash 与失败不发布规则。重构无需新增 schema 版本；确有字段语义变化时才版本化，不伪装成兼容加载。

每个差分分两层：

1. **身份差异**：参与构建源码、生成器/profile 身份或来源记录变化。
2. **执行差异**：binding、数值、layout、任务及依赖、slot/alias、生存期、Submit、publish、预算和传输变化。

纯组织重构优先要求完整计划和 descriptor 一致。实际代码哈希进入 profile 时允许身份随源码变化，不能固定旧哈希伪造一致；忽略字段必须逐项列入显式清单，仅限身份/来源，不得删除整个 header 或 attrs 来获得通过。

编号规范变化只有在必要时接受，需验证引用双射、依赖、顺序及物理计划一致；“拓扑上相同”不足以证明提交和存储方案等价。未解释的执行差异作为回归定位，不随重构自动接受。

descriptor 比较逐字段和有效数组内容，排除 Host 指针地址、padding 等非语义字节；不得直接 memcmp 包含指针的整个结构。shape、容量和参数字段不能被排除。

## 11. 连续工作包与关闭条件

工作包是交付边界，不要求每步汇报。可交叉推进独立事项，采用可运行的小提交；范围内失败自主修复。

| 包 | 工作 | 关闭条件 |
|---|---|---|
| R0 基线 | 核对工作区、依赖、消费者及 Host 基线；选取旧结果 oracle | 版本、已有修改/失败及代表性计划有记录 |
| R1 收敛核心 pipeline | 抽取规划/编码边界，旧入口共用核心，不改执行策略 | 代表性旧计划与 descriptor 无未解释差异 |
| R2 artifact | 新门面、直接实例化、explain/export、离线及依赖隔离 | 新路径无需持久化往返；生命周期和旧 API 通过 |
| R3 调用方 | 接入 CLI、模型和 legacy 入口的共享边界，消除重复编码 | 新旧入口共用算法，生产热路径与重放保留 |
| R4 原生图试点 | 真实 TileLang 直接生成 TVM 图，迁移目标契约与必要分析 | 完成正负例、差分和维护成本评估，作出采用决定 |
| R5 条件迁移 | 仅在第 8.3 节成立时切换默认，收缩 legacy RKIR | 无两套生产 planner；否则明确“未采用”及保留理由 |
| R6 累积验收 | 验证完整累积变更、必要设备检查、清理过渡代码 | 第 12–14 节成立，第 15 节完成交接 |

R1–R3 不因 R4 的依赖或可行性问题无限等待。R5 未采用不等于 R4 未做；只完成 RKIR→TVM 转码不能关闭 R4。

## 12. 最小验证矩阵

复用现有 fixtures 和工具，以下是具名风险，不要求为每行另造框架或测试文件。

| 场景 | 必须检查 |
|---|---|
| 真实单 MatMul、MatMul→SiLU | 新入口可编译，数值政策、task/Submit 和 descriptor 保持 |
| K=257、K 奇偶链、ragged M/N | 收窄、尾块、padding、乒乓 BO 与输出落点保持 |
| 分支、残差、多输出、代表性 FFN | 真实图依赖、所有消费者与导出值生存期保持 |
| 非零 fill、错误索引、predicate、barrier/atomic | 保持对应拒绝及可追溯 source，不能产生部分 artifact |
| strict FP32、缺失收窄、未许可 LUT | 对应数值拒绝；普通 TVM 类型合法不等于目标可编译 |
| 不支持广播、shape/dtype、超域尺寸 | 由目标检查拒绝，不因 TVM 一般能力而扩域 |
| 原生图 pass 与契约传播 | 至少验证实际采用 pass；破坏数值/关联的变换不得静默通过 |
| compile/explain 与无 TVM 环境 | 无设备访问；基础 import、legacy 编译、plan 重放保持约定依赖 |
| artifact 实例化与释放 | 原图释放后可实例化；独立 workspace 不串资源；销毁及错误所有权保持 |
| explain/report 修改 | 不影响权威计划和后续实例化 |
| descriptor 释放及恶意输入 | 深拷贝保持，C verifier 独立拒绝坏引用、容量、溢出和 alias |
| 并发与重复输入 | BUSY、不串 binding 指针、不额外提交、输入更新与缓存失效正确 |
| 编码/实例化/执行失败 | 无半有效结果，创建清理、publish 与 quarantine 保持 |
| 导出→加载→运行 | 旧文件与新导出可重放，正常内存路径不经序列化 |
| 模型默认和诊断路径 | 现有默认策略、trace 隔离、数值政策及结构计数保持 |

不为无影响的重导出/文件移动单独添加镜像测试。对原生图和 artifact 新边界的风险补充测试即可，测试数量不是完成度。

## 13. 实际命令、环境与设备验证

以下命令为已存在的常见入口，执行前核对当前有效环境和 build 目录，不把历史目录存在视为本轮构建成功：

```bash
cd research/tilelang-rknpu
PYTHONPATH=.pydeps:python:. python3 -m pytest -q
ctest --test-dir ../custom-ops/build-host-mvp --output-on-failure
```

锁定 source-built TileLang 路径优先按已有 provenance 和脚本使用，记录实际 Python、模块与库路径。不要运行会删除/重建依赖环境的脚本来完成纯重构；选定环境缺失时记录具体问题并继续不依赖它的主线。

修改 C 时按实际配置重新构建再测试。仅改 Python 时使用可信当前 C build，不重复建立所有构建目录。涉及 FFI 编码或资源生命周期变化时执行现有相关 sanitizer/ABI/失败回归；不因曾经有 sanitizer 结果就宣称本轮通过。

新 artifact 冒烟验证至少执行 compile_artifact→explain→instantiate→run，以及 export→load_plan→既有运行入口，两条路径结果一致。不要求新增 CLI 命令来验证 Python API。

设备证据按变化程度选择：

- 仅组织/门面变化且计划、descriptor 和 runtime 行为保持：复用既有硬件能力证据，完成可用板卡上的代表性成功/重复输入与冻结数值检查；不重新跑完整性能 campaign。
- 实际任务、布局、传输、提交或运行时行为改变：先定位是否为意外回归；确有必要且仍在范围内时按受影响路径执行冻结政策及模型验收，不借机重新校准。
- 无板卡：交付可执行入口与 Host 证据，设备项明确未运行，不将 mock 写成硬件验证；已证明相同的执行计划也不宣称本次真机重测。

只在实测出现退化或调用开销变化时做有界同环境性能对照。本任务不预设速度提升，也不通过几乎相同的性能实验拖延交付。

## 14. 证据目录与完成条件

在已有 `results/` 下使用一次独立的 `*-refactor/` 目录，至少保存实际需要的：

```text
manifest.json                 # 源码/依赖身份、命令、环境与退出状态
baseline/                     # 小型代表性计划/descriptor 投影
comparison.json               # 差分、明确忽略字段和未解释差异
native-ir-evaluation.json      # 原生路径覆盖、契约和采用判断
logs/                         # 实际构建/测试输出
```

允许按现有工具调整名称，避免为此开发独立证据系统。不新增 README/summary.md；摘要、决定和重放命令只写第 15 节。大型数组不提交，记录原始位置与必要哈希。前置失败、零测试、skip 与未测项不得写成通过。

必做主线完成要求：

1. 新 artifact 入口可以不经 JSON 直接实例化并运行，compile/explain 无设备副作用。
2. 规划与编码有明确边界；核心算法和 descriptor 编码没有双份维护。
3. 旧入口、必要依赖隔离、持久化及重放保持；确有未兼容项已获相应决策而非静默删除。
4. 数值、支持域、资源及故障行为没有放宽，计划差分可解释。
5. TVM 直接路径已完成真实试点与采用/保留判断；不以临时容器转码替代。
6. 相关测试及可用设备检查有实际证据，未测项可识别。
7. 过渡代码有明确用途，未留下第二套生产后端或孤立的伪新入口。
8. 本文实现记录与代码一致，无额外的本任务设计/状态/交接文档。

## 15. 实施记录与最终交接（执行中更新本节）

本节是本任务唯一实施记录。状态使用 `not_started / in_progress / passed / not_adopted / blocked`，其中 `not_adopted` 只用于有证据的条件候选。技术困难或缺设备不自动阻塞独立工作。

### 15.1 基线与工作区

- 实际基线 HEAD：`2996bb65d019d18bd118a35b03779dfba9c00f41`（`chore: checkpoint workspace before refactor`）。
  开工时 `git status --short` 为空，工作区干净，无其他会话的未提交改动；本次全部变更都在该提交之上。
- 隔离方式：用 `git worktree add --detach /tmp/rknpu-baseline 2996bb6` 承载旧代码 oracle，
  与工作树完全分离，未 reset/clean/覆盖任何文件，也没有复制未提交文件。证据脚本每次运行都自建
  临时 worktree（见 15.5），因此旧侧对照可重放而不是手工快照。
- 锁定依赖与实际导入路径（`--out` 结果的 `manifest.json` 记录机器可读版本）：
  - Python 3.12.3（`/usr/bin/python3`）；
  - `numpy`、`pytest`、`jsonschema` 与 `tilelang 0.1.15` 来自 `research/tilelang-rknpu/.pydeps`；
  - `tvm 0.25.dev0` 随 `tilelang` wheel 的 `3rdparty/tvm/python` 提供（`import tvm` 必须先 `import tilelang`）；
  - 共享库 `../custom-ops/build-host-mvp/librknpu_plan.so` 与 `librknpu_plan_mock.so` 沿用既有构建，未重建、未升级依赖。
- 基线命令与结果（与本次改动后一致，均 0 退出）：
  - `PYTHONPATH=.pydeps:python:. python3 -m pytest -q`：270 passed；
  - `ctest --test-dir ../custom-ops/build-host-mvp --output-on-failure`：30/30 passed。
  基线无既有失败项；未跳过、未 xfail、未因本次改动修改任何跳过条件。

### 15.2 工作包状态

| 包 | 状态 | 改动/决定 | 证据 |
|---|---|---|---|
| R0 | passed | 基线 HEAD、干净工作区、依赖路径与 Host 基线（270 pytest + 30 ctest）确认；选定旧代码 worktree 作 oracle | 本节 15.1；`results/20261027T064126Z-refactor/manifest.json` |
| R1 | passed | 规划/编码收敛到 `codegen/`：`pipeline.py` 是唯一 normalize→plan→verify 链，`descriptor.py` 是唯一 descriptor 编码；`api.py` 变薄门面，旧计划与 descriptor 逐字段一致 | `comparison.json`：19/19 用例 plan 与 descriptor 全等，7/7 负例错误码全等 |
| R2 | passed | `artifact.py`：`compile_artifact` / `CompiledArtifact.explain/instantiate/export_plan`，无 JSON 往返、无设备句柄、原 PrimFunc 释放后仍可实例化 | `tests/test_artifact.py`（18 项）、`tests/test_codegen_pipeline.py` |
| R3 | passed | CLI `compile-*` 改走 artifact 并只做导出；`PlanCall`/`HybridRunner`/`Workspace.create` 共用同一编码入口与资源创建；模型与 campaign 工具无重复 planner | `tests/test_codegen_pipeline.py::test_one_descriptor_encoder`、`test_all_compile_entries_use_the_one_pipeline`；`tests/test_models.py` 等原用例全绿 |
| R4 | passed | 真实 TileLang → `tvm.relax` 原生图（不经过 RKIR 容器），目标契约以函数级 `DictAttrs` 承载并在每次使用前校验；12 个代表性正例 plan/descriptor 与 RKIR 路径全等，8 个负例拒绝一致 | `native-ir-evaluation.json`；`tests/test_native_graph.py`（15 项） |
| R5 | not_adopted | 第 8.3 节条件 1/2/3/5 成立、条件 4 不成立：RKIR 容器无法因本次迁移删除，适配成本净增。保留薄 RKIR 为计算表示，原生载体降为非默认试点 | 15.3 的成本对照 |
| R6 | passed | 累积验收、证据目录、文档收敛；无新增设计/状态文档。审核后补齐四项边界与证据问题（计划只读、模型包比较器、证据脚本退出码、PrimFunc 释放测试），见 15.6；并补做 4 例真实 TileLang 输入的板端 smoke（单核串行），见 15.7 | `results/20261027T064126Z-refactor/`；`results/20261027T071045Z-artifact-smoke/`；15.5；15.6；15.7 |

### 15.3 原生图与 RKIR 决定

**选择的 TVM 表示、接入位置与依据**

- 表示：`tvm.relax`（`IRModule` + `relax.Function` + `BlockBuilder` dataflow）。shape、dtype、算子与
  数据流边全部由原生节点（`relax.Var` / `relax.Call` / `VarBinding` / `TensorStructInfo`）承担，
  未新增 dialect、未把整图 JSON 塞进 attribute。
- 接入位置：`frontend/graph_builder.py` 是识别器的载体缝。识别逻辑（`tilelang_adapter.py`，
  4405 行结构识别）只有一份，8 个建图点改为调用 `active_builder().value/op/finish`；
  默认载体 `RkirBuilder` 原样产出 `ir.Value/Op/Module`，试点载体 `RecordBuilder` 产出无容器记录，
  再由 `frontend/native_graph.py` 直接生成 Relax 节点。
- 依据：识别器本身与图容器无关；把容器替换成上游表示只可能改变"记录与校验"成本，不改变
  AST 语义识别，因此这是唯一能在不复制识别算法的前提下回答"是否值得迁移"的切法。

**直接路径、保留的目标契约及 pass 传播方式**

- 直接路径不经 RKIR：`tests/test_native_graph.py::test_no_rkir_object_is_created` 在构建期把
  `ir.Module/Value/Op` 换成记录器，断言原生编译期间一次都没有实例化。
- Relax 内建算子自带强类型 `Attrs`，无法附加自定义 per-node 契约；因此目标契约以函数级
  `rknpu_contract`（`DictAttrs`）承载，且只承载普通 Relax 不能表达的部分：per-value 角色与绑定名、
  per-op `numeric_contract`、`source_accumulator_dtype`、`source_narrowing_points`、
  `source_k_loop`、`source_mn_tiling`、`source_tail_contract`、stage 信息、`b_layout`、源 span。
- 契约与图的关联不用可变变量名，也不用遍历编号：`native_graph.graph_signature()` 生成
  "参数形状/dtype + 每个算子的原生算子名 + 操作数引用（`p<k>` 参数 / `v<j>` 第 j 个算子结果）+
  结果形状/dtype + 输出引用" 的规范签名，与图同构、与重命名和重编号无关。每次使用前
  `validate_contract()` 重算签名并与契约比对，`cross_validate_contract()` 再验证契约内部
  （opcode↔native_op、value_id↔producer、角色↔inputs/outputs、形状↔签名）自洽；任一不符即拒绝。
- 允许的 pass 集合固定为 `Normalize`、`CanonicalizeBindings`、`DeadCodeElimination`，每个 pass
  之后立即重校验；实测三者在单 MatMul 与 staged projection 上均 `preserved`，之后计划仍与
  旧路径全等。该集合之外、或使签名改变的变换一律拒绝而不是静默接受。

**删除/简化内容与新增适配/测试成本**

| 项目 | 数值（实测） |
|---|---|
| 主线（R1–R3）代码净变化 | 跟踪的代码文件 `+238 / −554` 行（净 −316）：`runtime/ffi.py` 移除 378 行、`codegen/descriptor.py` 承接 365 行，`api.py` 173→111 行；另有新增未跟踪模块 `codegen/{__init__,pipeline}.py` 278 行、`artifact.py` 256 行、`graph.py` 45 行 |
| 原生载体新增 | `frontend/graph_builder.py` 136 行 + `frontend/native_graph.py` 710 行 = **846 行** |
| 原生载体新增测试 | `tests/test_native_graph.py` 377 行（15 项 + 逐例 subtest） |
| 审核反馈修复 | `immutable.py`（135 行）冻结计划/图属性；比较器增加两侧文件集合与引用/摘要校验；证据脚本改为失败即非零退出；新增真实 PrimFunc 释放测试 |
| 识别器改动点 | 8 个建图点（`Module(` → `_finish(`）+ 59 处 `Value(`/`Op(` 构造改经载体缝（现共 61 处经缝调用）；识别语义零改动 |
| 需要重复的校验 | 原生侧需重做静态 shape/dtype 检查（`build_spec`）与角色合法性；RKIR 侧仍在 `ir.TensorType/Value/Op` |
| 契约重校验 | 每个被声明的 pass 之后一次签名重算；承载 10 类目标契约键 |
| 算子覆盖缺口 | `hardsigmoid` 在锁定 TVM 中无直接 Relax 算子，原生载体拒绝而非组合（组合会破坏一记录一节点关联） |
| 编译耗时（中位） | 原生/旧比 = 1.17（范围 1.10–1.71，12 例各 5 次重复），含 3 个 pass 与全部签名重校验 |
| 执行差异 | 0：12/12 计划与 descriptor 逐字段全等，8/8 负例拒绝码与 stage 全等 |

**是否采用默认原生路径及理由：不采用（`not_adopted`）**

对照第 8.3 节五个条件：

1. 成立——12 个代表性正例（核心、激活、K 分片、ragged 尾、标量/逐通道广播、多输出、MN 分块、
   staged projection/FFN/branch）计划与 descriptor 全等；8 个相邻负例（grid、K 分片、ragged 输出、
   缺失收窄、未授权 LUT、跨块读、非支持算子）拒绝码与 stage 与旧路径一致。
2. 成立——数值契约、收窄点、累加器 dtype、访问/副作用、来源 span 与支持域检查均未丢失：
   上述全等是以完整 plan + descriptor 为判据，不是仅比较任务数。
3. 成立——规划器未重建完整 RKIR；兼容视图只是从原生模块派生的只读投影（`values`/`ops`/`value(id)`），
   不写入、不派生第二张可变图。
4. **不成立**——本次无法删除或简化任何既有实现：`compile_rkir`、`ir` 的模块构造器
   （`matmul_module` 等，模型工具与 board campaign 直接使用）、RKIR JSON 往返与 `compile-rkir` CLI
   都必须继续可用，因此 RKIR 容器原样保留；迁移只会新增 913 行载体 + 重校验 + 算子覆盖缺口，
   并把编译耗时提高约 17%（中位）。净收益为负。
5. 成立——基础依赖隔离（无 TileLang/TVM 时 import/compile/explain/重放仍可用）、legacy 输入与
   导出重放均保持。

条件 4 不成立即不满足第 8.3 节"同时成立"的要求，故不切换默认，也不删除 RKIR。

**legacy RKIR 的最终职责和依赖策略**

- RKIR 仍是**计算表示**（薄图容器 + 类型/契约记录 + 严格 JSON schema），不是第二套后端：
  它没有独立的 legalizer/planner/encoder，`compile_rkir` 与 `compile_tilelang` 走同一
  `codegen.pipeline.compile_graph`。
- 依赖策略不变：`import rknpu_backend` 不导入 TVM/TileLang；`ir` 与 `plan` 无设备依赖；
  RKIR JSON 往返仅在显式 `compile-rkir`/`load_module` 路径使用。
- 原生载体保留为非默认、显式选择的试点（`compile_artifact(..., carrier="native")`），供评估与回归
  使用；它与默认路径共享 100% 的合法化、规划、存储与编码算法，因此不构成"两套生产后端"。
  若后续不推进迁移，删除 `frontend/graph_builder.py` 与 `frontend/native_graph.py`
  （连同 `tests/test_native_graph.py` 与 `tools/evaluate_native_graph.py`）即可，规划器无感知；
  推进迁移的触发条件是同时改造 `compile_rkir`/`ir` 构造器/board campaign 输入，使删除 RKIR 成为
  净简化——这属于新的授权范围，不在本次结论内。

### 15.4 最终接口与兼容范围

**实际模块安排与唯一核心调用链**

```text
python/rknpu_backend/
  graph.py                  # 计算图访问协议（planner 与载体之间唯一的耦合点）
  codegen/
    pipeline.py             # 唯一核心链：normalize → legalize/plan → verify → explain
    descriptor.py           # 唯一 plan descriptor 编码（ctypes 结构与字段映射）
  artifact.py               # CompiledArtifact / compile_artifact
  api.py                    # 旧入口薄门面（compile_rkir/compile_tilelang/load/save/explain）
  frontend/tilelang_adapter.py  # 结构识别（语义唯一副本）
  frontend/graph_builder.py     # 载体缝（默认 RKIR / 试点记录）
  frontend/native_graph.py      # 原生 Relax 试点载体（非默认）
  passes/                   # 既有合法化、draft、submits、storage、verify（未搬家）
  runtime/bindings.py       # Workspace 资源生命周期（未改语义）
  runtime/ffi.py            # ctypes ABI 桥 + codegen.descriptor 的薄重导出
```

唯一调用链：`compile_artifact` 与 `compile_tilelang`/`compile_rkir` → `codegen.pipeline.compile_graph`
→ `passes.*` → `Plan` →（编码时）`codegen.descriptor.build_plan_descriptor` → C `rk_plan_create`。
`tests/test_codegen_pipeline.py` 用打点断言：编译入口只经过一次 `compile_graph`，descriptor 只编码一次，
`runtime.ffi.build_plan_descriptor is codegen.descriptor.build_plan_descriptor`，
且 `passes/*` 不导入 runtime/ctypes、`codegen/*` 不导入 runtime。

**artifact API 与旧入口适配**

```python
def compile_artifact(prim_func, *, profile, options, b_is_input=False,
                     retain_graph=False, carrier="rkir") -> CompiledArtifact
class CompiledArtifact:
    plan, snapshot, profile, options, name, plan_hash, profile_hash, source_hash,
    source, diagnostics, has_retained_graph
    def explain(self) -> dict
    def instantiate(self, *, fd: int, constants: dict[str, numpy.ndarray]) -> Workspace
    def export_plan(self, path: str) -> None
    def with_retained_graph(graph) -> CompiledArtifact
    def release_diagnostics() -> CompiledArtifact
```

- `compile_artifact` 接受真实 `tvm.tirx.PrimFunc`（或其直接暴露 PrimFunc 的 JIT 包装），也接受
  已归一化的 `ir.Module`（`carrier="rkir"`）。不执行 GPU pipeline、不解析源码文本。
- 不可变快照：`CompileSnapshot(profile, options)`（两者本身是 frozen dataclass），外加来源身份
  `source`（module 名、source_hash、numeric/submit/approx 选项、收窄点、K 循环/MN 分块/尾块契约）。
  不保存输入数组、TVM callable、设备地址或 fd；`instantiate` 才需要 fd，且 fd 始终由调用者持有。
- TVM 图只在 `retain_graph=True` 时以可释放诊断对象存在，`explain()`/`instantiate()`/`export_plan()`
  都不依赖它；默认不保留。
- 旧入口：`compile_tilelang`/`compile_rkir` 返回 `CompileResult`，`.plan/.diagnostics/.report/to_dict()`
  语义不变，并新增 `.artifact`（同一 `Compilation` 的另一种视图，不重新规划）。
- `Plan`/`load_plan`/`save_plan`/`explain_plan`/`Workspace.create` 保持原语义；CLI 的
  `compile-tilelang`/`compile-rkir`/`validate`/`explain`/`run` 参数与退出码不变，`compile-*` 内部改为
  artifact 导出。顺带修正 `cli._cmd_run` 中 `RK_E_INVALID` 未导入的既有缺陷（只在 `--fd` 为负时触发；
  行为仍是退出码 2 的错误路径，无兼容性变化）。

**权威表示、descriptor 生命周期与导出路径**

- 权威表示是**一份已验证的内部 `Plan`**。`explain()` 每次从该计划重新生成报告并深拷贝，
  `export_plan()` 写出既有 RKPlan JSON，descriptor 是投影。
- 计划对外**逻辑上不可变**，且不只靠构造期拷贝：`plan.Action.attrs` 与 `ir.Op.attrs` 经
  `immutable.freeze()` 递归转成 `FrozenDict`/`FrozenList`（`dict`/`list` 的只读子类：仍是
  `isinstance(x, dict)`、仍与普通容器相等、仍可 JSON 序列化，但所有写方法抛 `TypeError`）。
  因此 `artifact.plan.actions[i].attrs["transpose_source"] = True` 直接失败，不会出现
  "descriptor 变了而 plan_hash 未变"的状态；`Action.to_dict()`/`Op.to_dict()` 用 `thaw()`
  输出普通容器，磁盘 JSON 与 canonical plan hash 与冻结前逐字节一致
  （`comparison.json` 的 19/19 仍全等即为此判据）。
  回归：`tests/test_codegen_pipeline.py` 的 `test_authoritative_plan_refuses_mutation`、
  `test_every_action_attribute_tree_is_read_only`、`test_recognized_graph_attributes_are_read_only`、
  `test_caller_mapping_edits_cannot_reach_the_plan`；
  `tests/test_artifact.py` 的 `test_descriptor_cannot_be_steered_by_editing_the_plan`
  （逐次尝试改写后重编码 descriptor 并要求 flags/counts 不变）与
  `test_attempted_plan_edits_do_not_change_execution`（改写尝试前后各跑一次真实 workspace，
  逐元素相等且仍等于数值参考）。
- descriptor 生命周期未变：编码在 `instantiate`/`create_plan` 时发生，ctypes 数组与字符串在 native
  调用结束前由 holder 保活，地址型 descriptor 不会被跨 workspace 缓存；artifact 不持有 descriptor。
- 每个 workspace 独立：`artifact.instantiate` 直接调用生产 `Workspace.create`，共享常量校验、
  C plan 深拷贝与 ownership/BUSY/poisoned 规则；未新增全局 workspace 或隐式开设备。

**C ABI / schema / API 变化**

- C 侧 `rk_plan.h` 及实现：**零改动**（本轮未编译、未改动 custom-ops；沿用既有 `build-host-mvp`）。
- Plan schema 版本：**未变**（仍为 schema 2 / kind `rkplan`），字段语义未变，故不新增版本号；
  RKIR schema 仍为 1，JSON 格式与严格解码规则未变。
- 公共 Python API：只新增（`compile_artifact`、`CompiledArtifact`、`CompileResult.artifact`、
  `rknpu_backend.codegen`、`rknpu_backend.graph`），未删除或改名任何既有导出；
  `runtime.ffi` 的 descriptor 类型名与常量保留为重导出。

### 15.5 验证与交接

**证据目录、实际命令与退出状态**

证据根：`results/20261027T064126Z-refactor/`（唯一 `*-refactor/` 目录，未新增 README/summary）：

```text
manifest.json                   # HEAD/base、环境与模块路径、各命令退出码、产物清单
baseline/projection.json        # base_ref worktree 的 19 例计划 + descriptor 投影
current/projection.json         # 工作树同一脚本的投影
comparison.json                 # 逐字段差分、显式忽略清单、未解释差异
native-ir-evaluation.json       # 原生载体覆盖、契约传播、成本与耗时
model-package-regeneration.json # 完整模型包重编译后的逐文件等价判定
logs/                           # 上述命令与 pytest/ctest 的原始输出
```

一条命令重放（自建 base worktree，不改工作树、不装依赖）：

```bash
cd research/tilelang-rknpu
RK_REFACTOR_BASE_REF=2996bb65d019d18bd118a35b03779dfba9c00f41 ./scripts/run-refactor-evidence.sh

# 追加完整模型包重编译对照（会临时改写 models/compiled 后自动还原）
RK_REFACTOR_WITH_MODELS=1 \
RK_REFACTOR_BASE_REF=2996bb65d019d18bd118a35b03779dfba9c00f41 ./scripts/run-refactor-evidence.sh
```

各步退出码均为 0：`worktree`、`baseline_projection`、`current_projection`、`comparison`、
`native_ir_evaluation`、`model_compile`、`model_package_regeneration`、`model_restore`、
`pytest`、`ctest`、`manifest`（见 `manifest.json.commands`）。`model_restore` 之后
`git status --short models/compiled` 为空，模型包文件未被本次证据采集改动。

**完整模型包重编译**（`tools/compile_models.py` 全量重跑后逐文件比较，
`model-package-regeneration.json`）：53 个文件里 52 个为 `identity-only`
（仅 `plan_hash`/`source_hash` 不同），`manifest.json` 为 `identity-cascade`
（其差异叶子仅 `plans.*.plan_hash`、`hybrid_manifests.*.sha256`），
`content-diff` 0 个、`failures` 0 个、`reference_problems` 0 个。

`identity-cascade` 不是默认放行：比较器先做三项独立检查（见 15.6 P2-1）——两侧 `*.json`
文件集合必须一致；`plans.<n>.path` / `hybrid_manifests.<n>.path` 必须存在且其 `plan_hash`、
`source_hash`、`sha256` 与所指文件的真实内容相符（`sha256` 按文件字节重算）；
每个被引用文件的自身判定必须是身份等价类。任一不满足即计入 `failures` 并返回非零。
即：重构后重编译整个模型包，执行内容（bindings、actions、submits、publish、预算、hybrid 图）
逐字段不变，只有那处既有身份不稳定反映到 hash 上。

**完整差分结果与允许的身份差异**

- 19/19 代表性用例的 `plan` 与 `descriptor` 逐字段全等；7/7 负例的错误码、stage 与消息全等。
  用例覆盖：RKIR 5 个构造器（matmul/projection/swiglu/branch/activation）+ TileLang 单 MatMul、
  MatMul→SiLU、K=512 分片、ragged K=257、标量/逐通道广播、多输出、MN 分块、silu_proj、
  staged projection/FFN/branch。
- 显式忽略清单（`comparison.json.allowed_differences`）只有三项，全部为环境位置类：
  `environment.backend_module`、`environment.executable`、`environment.platform`。
- descriptor 比较逐字段 + 逐有效数组元素，排除 host 指针地址与 padding；shape、容量、counts、
  顺序、flags、slot/offset、Submit 与 publish 均参与比较。
- 一处**先于本次重构就存在的**身份不稳定，按"不伪造一致"处理并记录：TileLang 派生计划的
  `source_hash` 取自 `repr(Span)`，其中 `SourceName(...)` 内含 TVM 对象地址，因而**同一份旧代码
  跨进程也会得到不同 `source_hash`**（在 base_ref worktree 里连跑两次即可复现，见下方命令）。
  因此投影只归一化该指针子串并**重算 canonical plan hash**，其余字段不做任何遮蔽；
  归一化前的原始串与原始 hash 一并保存在 `baseline/projection.json` 与 `current/projection.json`
  每例的 `identity.{source_hash_raw, plan_hash_as_compiled}` 字段中，可审查。
  复现（在 base_ref worktree 里连跑两次同一脚本，`case=tilelang_single_matmul`）：

  ```bash
  for i in 1 2; do
    ( cd /tmp/rknpu-baseline/research/tilelang-rknpu && \
      PYTHONPATH=$PWD/python:<tree>/.pydeps:$PWD \
      python3 <tree>/tools/refactor_plan_projection.py --out /tmp/base_run$i.json )
  done
  # 归一化前：两文件里 12 个 TileLang 用例的 source_hash / plan_hash 不同（SourceName 地址不同），
  #           7 个 RKIR 用例本来就相同；
  # 归一化后：19/19 用例的 plan_hash 全部相同（= comparison.json 的 identical 19/19）
  ```

  这正是"不能固定旧哈希伪造一致"的实例：本次不对计划语义做任何补偿，只把该字段作为身份差异显式记录。
- 结论：除上述身份字段外，没有未解释的执行差异（`comparison.json.summary
  .unexplained_execution_differences` 为空）。

**Host / ABI / sanitizer / 设备结果及未测项**

- Host：`pytest` **348 passed + 71 subtests passed**（基线 270 → 348；新增 78 项：
  `tests/test_artifact.py` 21、`tests/test_codegen_pipeline.py` 15、
  `tests/test_native_graph.py` 15、`tests/test_compare_compiled_models.py` 11、
  `tests/test_verify_artifact_smoke.py` 8、`tests/test_artifact_smoke_script.py` 8）。
  `ctest` 30/30 passed，含 `rk_plan_contract`、`rk_plan_host`、`provider_context`、`provider_seam`。
- 无 TileLang/TVM 环境：`tests/test_artifact.py::ArtifactIsolationTests` 在真实子进程中屏蔽
  `tilelang/tvm/tvm_ffi`，验证 `import rknpu_backend`、`compile_rkir`、`compile_artifact(Module)`、
  `export_plan` → `load_plan` → 运行链仍可用，且包 import 不触发 TVM。
- 编译/explain 不得加载设备库：`tests/test_codegen_pipeline.py`
  `test_compilation_never_constructs_the_runtime_library` 在 `LibRKPlan.__init__` 抛错的前提下完成
  compile + explain + export/load。
- ABI：C 侧零改动，未重跑 sanitizer/ABI 回归（本轮不涉及 C 代码或 FFI 编码语义变化；descriptor
  字节布局由 `tests/test_ffi.py` 的 ctypes/`offsetof` 对照用例保障，且仍全绿）。
- **板端真机**：本机无直连设备，先用根目录 `.env` 的板卡配置补做了 4 例代表性 smoke
  （单 MatMul、MatMul→SiLU、ragged K=257、分支多输出；全部由真实 TileLang PrimFunc 在 Host 编译），
  记录见 15.7 与 `results/20261027T071045Z-artifact-smoke/`。仍**未重跑**完整硬件 admission campaign
  与完整模型 campaign：本次变化是组织/门面级（12 个代表性用例的计划与 descriptor 逐字段全等），
  既有 `results/20261023T144054Z-rk3588/` 等硬件证据按当时快照解释，本节的 smoke 只补充
  "当前代码在生产运行库上确实能跑通 artifact 路径"，不构成新的性能或整模型数值结论。
- CLI 冒烟：`tests/test_cli.py` 覆盖 `compile-rkir → validate → explain → run`（mock 后端）与
  `compile-tilelang` 路径，全部通过；未新增 CLI 子命令。

**最终提交、关键文件、遗留项与重放命令**

- 关键新增：`python/rknpu_backend/{graph.py,artifact.py,immutable.py}`、`python/rknpu_backend/
  codegen/{__init__,pipeline,descriptor}.py`、`python/rknpu_backend/frontend/
  {graph_builder,native_graph}.py`、`tests/{test_artifact,test_codegen_pipeline,
  test_native_graph,test_compare_compiled_models}.py`、`tests/nodeps/sitecustomize.py`、
  `tools/{refactor_plan_projection,compare_plan_projections,evaluate_native_graph,
  compare_compiled_models}.py`、`scripts/run-refactor-evidence.sh`。
- 关键修改：`api.py`（薄门面）、`cli.py`（改用 artifact 导出）、`runtime/ffi.py`（descriptor 移出并重导出）、
  `frontend/tilelang_adapter.py`（载体缝，8 个建图点）、`passes/infer.py`（接受图协议）、
  `plan.py`/`ir.py`（attrs 边界深拷贝）、`__init__.py`（新导出）。
- 遗留项（均可识别、未伪装为完成）：
  1. 原生载体为 `not_adopted` 的非默认试点；删除条件见 15.3 最后一段。
  2. `hardsigmoid` 在原生载体无对应 Relax 算子，未组合、未扩域。
  3. 板端真机未复验（无板卡）。
  4. 未跑 sanitizer/ABI 重建（C 侧未改动）。
  5. TileLang 派生 `source_hash` 的跨进程不稳定性是既有性质，本次只在证据侧归一化，未改变计划语义；
     若日后要求跨进程可复现 hash，需单独决策（会改变既有产物身份）。
- 重放命令：

```bash
cd research/tilelang-rknpu
# 全量 Host 验收
PYTHONPATH=.pydeps:python:. python3 -m pytest -q
ctest --test-dir ../custom-ops/build-host-mvp --output-on-failure
# 证据包（含旧代码 worktree 差分、原生图评估）
RK_REFACTOR_BASE_REF=2996bb65d019d18bd118a35b03779dfba9c00f41 ./scripts/run-refactor-evidence.sh
# 单一用例：编译产物 → explain → 实例化 → 运行 → 导出重放
PYTHONPATH=.pydeps:python:. RKNPU_PLAN_LIB=../custom-ops/build-host-mvp/librknpu_plan_mock.so python3 -c "from rknpu_backend import *; import numpy as np; p=default_profile(); o=CompileOptions(numeric_mode='rk_f16_tiled', allow_approx_lut=True); a=compile_artifact(matmul_module(m=13,k=300,n=33,activation='silu'), profile=p, options=o); print(a.explain()['counts']); c={s.name: np.zeros(s.shape, np.float16) for s in a.plan.constants if s.generator is None}; w=a.instantiate(fd=0, constants=c); print(w.run({'A': np.zeros((13,300), np.float16)})['A1'].shape); w.close(); a.export_plan('/tmp/plan.json'); print(load_plan('/tmp/plan.json').plan_hash == a.plan_hash)"
```

### 15.6 审核反馈与修复记录

一次独立审核给出了四条意见，全部采纳并修复。本节记录根因、改动与验证方式；审核未要求重跑
硬件 campaign，设备项状态不变。

**P1　artifact 的权威计划仍可被外部修改（已修复）**

- 根因：`Plan`/`Action` 是 frozen dataclass，但 `Action.attrs` 是可变 `dict`；构造期
  `deepcopy` 只切断调用者原字典的别名，`artifact.plan.actions[i].attrs[...] = ...` 仍可写入，
  而 descriptor 的 flags 正是从该映射编码，于是出现"descriptor 变了、plan_hash 未变"。
- 修复：新增 `python/rknpu_backend/immutable.py`（135 行，`FrozenDict`/`FrozenList`/`freeze`/
  `thaw`/`is_frozen`）。`plan.Action.attrs` 与 `ir.Op.attrs` 在构造时 `freeze()` 成只读子类，
  写方法一律 `TypeError`；`to_dict()` 用 `thaw()` 输出普通容器，**磁盘 JSON 与 canonical
  plan hash 不变**（`comparison.json` 仍 19/19 全等、7/7 负例全等即为此判据）。
- 验证：`test_authoritative_plan_refuses_mutation`（改写、删除、`update`、`setdefault`）、
  `test_every_action_attribute_tree_is_read_only`（逐 action 断言 `is_frozen`）、
  `test_recognized_graph_attributes_are_read_only`（图级 `op.attrs["numeric_contract"]`）、
  `test_caller_mapping_edits_cannot_reach_the_plan`（含嵌套 dict/list）、
  `test_descriptor_cannot_be_steered_by_editing_the_plan`（改写尝试前后重编码 descriptor 并要求
  flags/source_binding/kinds/counts 完全一致）、
  `test_attempted_plan_edits_do_not_change_execution`（改写尝试前后各实例化并运行一次，输出逐元素相等
  且仍等于数值参考）。审核给出的复现（改 upload 的 `transpose_source`）现在抛 `TypeError`，
  descriptor flags 与 `plan_hash` 均不变。

**P2-1　模型包比较器可能把不完整包判为通过（已修复）**

- 根因：只遍历 candidate 文件，看不到 reference 侧被删除的文件；`identity-cascade` 只看差异叶子
  的名字，未逐项验证 manifest 引用与摘要是否真实。
- 修复：`tools/compare_compiled_models.py` 现在按两侧 `*.json` 的**并集**比较
  （`missing-candidate` / `missing-reference` 直接计入失败），并对每个 manifest 做引用一致性校验：
  `plans.<n>.path`、`hybrid_manifests.<n>.path` 必须存在；`plan_hash`/`source_hash` 必须等于所指
  计划文件的自身字段；`sha256` 必须等于所指文件**字节的实时 SHA-256**；
  `mlp_hidden_diagnostics.<n>.plan_hash` 必须与 `plans.<n>.plan_hash` 一致；
  `source_files.<path>` 的摘要必须与真实源码相符（可用 `--source-root` 指向源码树，
  不可达时列入 `unresolved_source_checks` 而不是静默跳过）。
  `identity-cascade` 仅在其引用的每个文件都通过校验且自身判定属于身份等价类时成立。
- 验证：新增 `tests/test_compare_compiled_models.py`（11 项，合成包对）：删除被引用 hybrid、
  删除未被引用的 plan、candidate 多出文件、篡改 hybrid 摘要、篡改 plan 内容、篡改 manifest 里的
  plan_hash、篡改 source 摘要、仅身份差异应通过、目录缺失应报错。审核的两个复现
  （删除被引用 hybrid 文件、修改 manifest 摘要）在真实 `models/compiled` 上均返回 1 并给出具体原因
  （`hybrid_manifests.mnist_mlp_b1.path missing` / `…sha256 does not match its file`）。

**P2-2　证据脚本不把失败汇总为非零退出码（已修复）**

- 修复：`scripts/run-refactor-evidence.sh` 仍先跑完全部步骤（一个失败也产出完整证据包），
  最后由状态表汇总：任一非零即 `exit 1` 并打印 `EVIDENCE FAILED`，全部成功才 `exit 0`
  并打印 `evidence complete`；`manifest.json` 缺失时也按失败处理而不是抛栈。
  被跳过的可选步骤记录独立的非零码，同样判定为失败（"没跑"不等于"通过"）。
- 验证：故意传 `RK_REFACTOR_BASE_REF=deadbeef` 运行，脚本跑完所有可运行步骤后
  打印 `worktree rc=128`、`baseline_projection rc=1` 等并以 1 退出；正常全量运行以 0 退出。

**P2-3　"原 PrimFunc 释放后仍可实例化"缺少测试证明（已修复）**

- 原测试只 `del` 局部名，而 fixture 模块仍持有该 `PrimFunc`，且之后没有实例化与运行。
- 修复：新测试 `tests/test_artifact.py::test_primfunc_released_before_instantiate_and_run`
  用 `tilelang_semantics._valid_kernel(...)`（普通工厂，返回无其他属主的全新 PrimFunc）编译，
  持有 `weakref`，删除引用并 `gc.collect()` 后断言弱引用已失效（即 artifact 不持有源对象），
  再用 `artifact.instantiate(fd=0, …)` 走生产 workspace 路径运行并对照 FP16 K 链数值参考。
  同时加入负向对照：仍被引用的对象弱引用必须存活，确保断言不是因为弱引用机制失效而"通过"。
  人工注入回归（让 artifact 持有 PrimFunc）可使该测试失败，说明它确实能捕获该缺陷。

**修复后的整体状态**

- 主线、原生图评估与 `not_adopted` 结论均未改变；`comparison.json` 仍 19/19 计划与 descriptor
  全等、7/7 负例全等，`native-ir-evaluation.json` 仍 12/12 全等。
- Host 结果更新为 `332 passed + 71 subtests`（新增 62 项）与 `ctest 30/30`。
- 证据根更新为 `results/20261027T064126Z-refactor/`，其 `manifest.json.commands` 全部为 0，
  脚本自身返回 0。

### 15.7 板端真机 smoke（重构后新增，2026-10-27）

审核后的补充验证：用仓库根 `.env` 的板卡配置，在当前代码 + 生产运行库上做一次代表性真机
smoke。这不是完整性能 campaign，也不重新校准数值政策。

**覆盖范围（4 例，单核串行）**

| 用例 | 来源 | 计划 task/submit | 板端检查 |
|---|---|---|---|
| `single_matmul` | 真实 TileLang `_valid_kernel(16,256,16)` | 1 / 1 | 输出 `Y[16,16]` |
| `matmul_silu` | 真实 TileLang `_silu_kernel(16,256,16)`（LUT 已授权） | 3 / 2 | 输出 `Y[16,16]` |
| `ragged_k257` | 真实 TileLang `ragged_tail(13,33,257,16,64,256)` | 2 / 1 | 尾块 K=257，输出 `C[13,33]` |
| `branch_outputs` | 真实 TileLang `branch_outputs(8,64,32,48)` | 2 / 2 | 多输出 `Y1[8,32]`、`Y2[8,48]` |

四例**全部**由真实 TileLang PrimFunc 在 Host 编译（板端无 TileLang/TVM，见未测项），
因此"真实 TileLang 输入编译后的板端执行"覆盖了全部用例而不仅一个。

**每例在板端验证的路径**

1. `Module.from_dict(host 归一化图)` → `compile_artifact()` → `explain()`：
   板端重新编译得到的 `plan_hash` 必须等于 Host artifact 的（跨机器身份一致，4/4 通过），
   `explain()` 的 actions/bindings/buffers/views/tasks/submits/data_bos 计数与计划一致。
2. `artifact.instantiate(fd=…)` → `run(A)` → `run(B)` → `run(A)`：**同一 workspace** 的输入更新；
   两次 A 输出逐字节相同，B 输出必须不同（否则"输入未更新"也会看起来正确）。
3. 两个**同时存活**的独立实例各自 `run(A)`：输出逐字节相同且等于首次实例化结果。
4. `artifact.export_plan()` → `load_plan()` → `Workspace.create()` → `run(A)`：离线重放路径，
   输出与 artifact 路径逐字节相同，`plan_hash` 与 Host 一致。
5. workspace 自报 `task_count`/`submit_count` 与计划会计一致（4/4 通过），`numeric_mode=rk_f16_tiled`。

**数值结果（Host 独立核对，容差取自计划自身声明的冻结契约，未放宽）**

| 用例 | 输出 | 最大绝对误差 | 允许上界 `atol+rtol·|ref|` | 结论 |
|---|---|---|---|---|
| `single_matmul` | `Y` | 0.000e+00（逐位一致） | 1.376e-02 | PASS |
| `matmul_silu` | `Y` | 2.441e-04 | 1.138e-02 | PASS |
| `ragged_k257` | `C` | 0.000e+00（逐位一致） | 1.297e-02 | PASS |
| `branch_outputs` | `Y1` / `Y2` | 0 / 0（逐位一致） | 1.163e-02 / 1.188e-02 | PASS |

- 参考值：Host 侧 float64 计算 + 每个声明收窄点显式舍入到 FP16（`ordered_k_tiles_with_f16_writeback`，
  每 256 宽 K 分片写回），与现有执行测试一致；容差 `atol=rtol=0.008` 直接从编译图的
  `numeric_contract.tolerance` 读取，脚本不提供任何调宽入口（同一 kernel 内出现不一致容差即报错）。
- 除 `matmul_silu` 的 LUT sigmoid 近似（2.4e-4，远小于冻结上界）外，其余用例与参考**逐位一致**。
- A/B 两组输入共用同一组常量，B 的参考值用同一权重重算；构建期即断言两组输入不相同。

**板端环境与安全边界（只读检查，未改设备状态）**

- `RK3588 OPi 5 Plus`，driver `RKNPU driver: v0.9.8`，`Python 3.10.12`，`librknpu_plan.so` 在板端
  现场构建（sha256 `878ef3fa6c81cb651e02783db2b6bb72829cb2b0941444eaf3bc755784051d85`）。
- 单核：`RKNPU_CORE_MASK=1`；串行；`npu_freq` 前后均为 `1000000000` Hz（只读采集，未调频），
  `npu_load` 前后均为 `Core0: 0%, Core1: 0%, Core2: 0%`；未执行 reset，未跑性能 campaign。
- 独立远端目录 `/mnt/ssd/workspace/tl-artifact-smoke-20261027T071045Z`（未复用既有目录）。
  该目录与上一次运行的 `/mnt/ssd/workspace/tl-artifact-smoke-20261027T065940Z` 均保留。
- 失败即停：runner 在第一个用例失败时立刻 `break`，不再提交后续用例；shell 侧一旦远端返回非零
  就跳过剩余 core mask。用"篡改首例 `plan_hash`"的设备无关演练验证了该守卫：
  退出码 1，后续 3 例均未被尝试（`logs/fail-closed-drill.log`）。
- 凭据：沿用既有 askpass 方式，密码只出现在进程环境变量中，不进入 argv/日志/证据；
  本次证据树对密码与板卡主机名的扫描均为 0 命中，脚本内扫描结果 `credential scan: PASS`。
- **顺带发现（早于本次任务，未修改）**：仓库既有证据
  `results/20261024T030615Z-models/model-results/core-1/cpu_benchmark.{log,json}` 与
  `results/20261026T014447Z-envelope/out-{baseline,candidate}/tiny_vit_b{1,8}.json`
  共 6 个文件中，凭据串作为**较长主机名字符串的子串**各出现 1 次（不是独立字段值）。
  这属于既有 campaign 采集主机名时带出的暴露，非本次改动造成，本次未改动这些文件（不在本任务范围）。
  建议对这些归档做脱敏或轮换凭据；本次新增证据不含该串，也不采集登录主机名。

**实际命令**

```bash
cd research/tilelang-rknpu
./scripts/run-artifact-smoke.sh          # 全流程：Host 编译 → 远端构建+执行 → Host 核对
```

脚本内部步骤（全部 0 退出，见证据 `manifest.json.commands`）：`build_campaign`（Host：真实
TileLang → artifact → `export_plan` + 参考值）、`bundle_source`/`bundle_campaign`、
`remote_mkdir`/`scp_*`/`remote_extract`、`remote_run`（板端 cmake 构建 `librknpu_plan` +
4 例单核执行）、`scp_results`、`verify`（Host 独立核对）、`credential_scan`。

**跨会话确定性（顺带得到）**：本 smoke 在同一板卡、同一构建上跑过两次独立会话
（`…065940Z` 与 `…071045Z`，各自新的远端目录）。两次的 30 个原始输出数组
（4 例 × {a1,a2,b,independent_1,independent_2,replay} × 各输出）**逐字节完全相同**，0 处不一致；
计划 hash 因 TileLang 派生 `source_hash` 含进程地址而按预期不同（见 15.5），但执行结果不变。

**脚本退出码修复（审核第 3 条非阻塞项）**：`scripts/run-artifact-smoke.sh` 原先记录了
`manifest_rc` 却未纳入最终判定，且失败计数依赖重新读取 `manifest.json`——而该文件正是可能失败的
那一步写出的，manifest 失败时可能误报成功。现改为：先从状态表（含 manifest 行）汇总失败步，
删除表之后再对 `manifest_rc`/`remote_rc`/`verify_rc` 逐项兜底（已记录的不重复计数），
状态表为空或不可写也算失败；失败时按步名列印 `artifact smoke: FAIL (step(rc) …)`。
回归：`tests/test_artifact_smoke_script.py`（8 项）直接抽取脚本尾部的聚合逻辑，覆盖
"仅 manifest 失败（其余全过）必须失败""manifest 行缺失但 `manifest_rc!=0` 仍失败"
"任意步失败""状态表不可写"等场景；另用无法连通的传输桩跑整脚本，确认退出 1 且逐个列出失败步。
真机重跑即为本次修复后的证据（`manifest_rc=0` 已参与判定）。

**过程说明（如实记录）**：首次真机 smoke 的证据目录在清理临时文件时被误删，因此本用例的证据是
**用同一脚本重跑一次**产生的（`…071045Z`）；被删目录对应的远端目录仍完整保留在板卡上
（`…065940Z`），本次用其原始输出做了上面的跨会话逐字节比对。重跑同时使修复后的退出聚合逻辑
在真机上得到验证（`manifest_rc=0` 参与判定）。板端 `librknpu_plan.so` 两次构建 sha256 相同
（`878ef3fa…`），说明重跑不是新的构建。

**证据**：`results/20261027T071045Z-artifact-smoke/`

```text
manifest.json                 # 覆盖范围、远端目录、显式未做项、各步骤退出码
campaign/                     # 每例 plan.json / rkir.json / 输入 A,B / 常量 / Host 参考值 / meta.json
board/core-1/summary.json     # 板端汇总：板端 plan_hash、explain 计数、workspace 会计、NPU 负载
board/core-1/<case>/board/*   # 原始输出：a1/A、b/B、a2/A、independent_1/2、replay（.npy，未加工）
board/{environment,configure,build}.log, library.sha256
verify.json                   # Host 独立核对：逐输出误差、允许上界、各项一致性检查
logs/fail-closed-drill.log    # 失败即停演练（设备无关，mock 库）
logs/{build-campaign,verify,remote-run,scp-*}.log
```

**未测项（明确未做）**

1. 未跑完整性能 campaign、未做频率/功耗标定、未做多核（仅 `RKNPU_CORE_MASK=1`）。
2. 板端无 TileLang/TVM，因此 **PrimFunc → artifact 的归一化在 Host 完成**；板端验证的是
   `归一化图 → artifact → explain → instantiate → run` 与 `export → load_plan → run` 两条路径，
   两者 `plan_hash` 与 Host artifact 完全一致。板端不支持 TileLang 是环境限制，不是本次改动造成。
3. 未覆盖全部算子/布局组合（只取 4 个代表性用例）、未覆盖并发与故障注入（这些由既有
   `test_workspace_busy`/`test_ws_poison`/board campaign 覆盖，本次未重跑）。
4. 未做 sanitizer 重建（C 侧零改动）；板端 `librknpu_plan.so` 为 Release 构建。

## 16. 自主执行与需要决策的边界

本规范已授权必要 helper、文件拆分、包组织、兼容适配、测试和证据脚本的有限改造。执行智能体自主推进 R0–R6、修复失败、选择实现和处理无收益候选；不因每包完成或某个局部方案未采用而停止等待确认。

只有以下情况提前提交具体方案：改变数值政策/支持域，新增未验证硬件配方，改变同步/资源生命周期，无法兼容的重要公共接口，破坏性操作，或缺少不可替代的外部条件。提问须说明实际冲突、已完成的可审查工作、可选方案及推荐取舍。

正常交付在完成整段工作后一次性汇报，并更新第 15 节。不要以增加文档、接口壳、mock 或测试数量替代真实编译到运行的闭环。
