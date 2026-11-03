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

## 结构化 lowering 试点（2026-11-04，opt-in，未切换默认）

本节记录 refactor 的第二条路径：不再把整份 PrimFunc 归类成模板再重建执行结构，
而是直接在 TileLang 产出的 TIR 上做分阶段转换，直到必须选择 RKNPU primitive 为止。
历史决策（native graph 试验等）见下文既有章节，此处不覆盖。

### 阶段划分与不变量

| 阶段 | 输入 | 输出 | 不变量 |
|---|---|---|---|
| 0 capture | PrimFunc | 骨架：1 个 root block、2 维 block grid（thread_binding）、1 个 serial K 循环、内核内的 tile 缓冲分配 | 拒绝非 SBlockRealize 根、拒绝有副作用调用（call_extern/packed） |
| 1 regions | 骨架 + K 循环子树 | 局部 tile 区域清单：fill、guarded copy、cast copy、gemm、FP16 回写；原始语句序列保留在 trace | 两个 shared tile 每轮清零、两个 guarded copy、每 K 块回写运行累加器 |
| 2 k-partition | K 循环 | 有序 K 切片（k0/extent/physical） | 切片有序且覆盖 logical K |
| 3 tiles | grid + 输出缓冲 | 每 tile 的 valid_rows/valid_cols 与物理 tile | grid 恰好覆盖输出一次 |
| 4 module | 以上 | StructuredModule（grid、tiles、slices、buffers、rounding、hints、stages） | 缓冲保留源 scope；数值舍入来自源而非 target；hint 只记录不应用 |

实现：`python/rknpu_backend/frontend/structured.py`；测试：`tests/test_structured_lowering.py`。
每个阶段返回 `Stage(name, invariants, ir_digest, notes)`，审查者可以逐阶段看到 IR 摘要，
而不是只看到最终字段。

### 与旧路径的关系

- 旧路径（`frontend/tilelang_adapter.py` + `passes/dag.py`）保持完全不动，仍是生产入口。
- 新路径是 opt-in 的库级 API，不改变任何默认；本轮**没有**把新模块接到计划生成，也**没有**替换 extractor。
- RKIR/RKPlan 的职责不变：仍是执行 IR；改动的是“谁来产生 tile 几何与 K 链”的推导位置。

### 已完成 / 未完成

已完成：阶段 0/1/3/4 在真实 guarded-tail 源码上可运行（grid (1,12)、tile (256,256)、12 个 tile、
三份源码共享同一几何、hint 记录、非法输入拒绝、模块可 JSON 化）。

未完成（阻塞点，已定位）：阶段 2 的 **K 尾块**推导。该 TVM fork 中 guard 比较节点既不是
`tir.LT`，serial 循环也没有 `loop_vars` 属性（探针输出：`loop vars: []`、`LT nodes: []`），
因此 `_logical_k_from_guards` 返回 None，第三切片被报成 extent=256（应为 64）。下一步：
用 `type(node).__name__`/`dir(node)` 枚举该 fork 的比较与循环变量表示，或改用 `prim_func.script()`
的规范化文本 + `tvm.arith` 分析；在修好之前，新路径**不得**用于生成计划。

---

## 46. 参考实现核实后的架构修正：走 TileLang 的 backend/pipeline 缝，而不是自建 stage 模块（2026-11-04）

### 46.0 大白话

§45 的方案把 RKNPU 的结构化 lowering 写成了一个自建的五阶段模块。核实本机 TileLang（0.1.15，commit da4eeab）之后，
结论要修正：**这套分阶段转换 TileLang 自己就有，并且已经有一个不用线程的后端（CPU）在同一条主链上跑**。
所以正确的接入点是注册一个 `rknpu` BackendModule + PassPipeline，复用通用主链，只在三个槽位插入 RKNPU 特有的东西；
我那个 `structured.py` 只应作为过渡期的结构事实提取器，不应成为最终架构。

### 46.1 已独立核实的关键事实

| 事实 | 证据（本地，本会话亲自 grep） |
|---|---|
| 无线程主链已存在 | `.pydeps/tilelang/cpu/pipeline.py:17` `MaterializeKernelLaunch(lower_thread_binding=False, unsupported_annotations=["cluster_dims"])`；`:81` 注释说明 CPU 因线程绑定未 lowering 而跳过 `LowerThreadAllreduce` |
| 后端是可注册的数据清单 | `.pydeps/tilelang/backend/module.py:29` `class BackendModule`、`:238` `register_backend`、`:112` `lower` |
| tile op 有按 target 解析的实现注册表 | `.pydeps/tilelang/tileop/gemm/registry.py:23` `register_gemm_impl`、`:38` `resolve_gemm_impl` |

（GPU 主链的逐 pass 清单、哪些 pass 带 GPU 假设，由本轮只读研究子代理给出并附 file:line；上述三条是我自己复核的部分，
其余引用其报告，未逐条复核。）

### 46.2 修正后的架构

```
TileLang TIR (block/loop/alloc/region 完整)
   |  rknpu BackendModule.register_backend()
   v
PassPipeline("rknpu")  =  通用主链（thread-free）
   BindTarget -> MaterializeKernelLaunch(lower_thread_binding=False)
   -> Simplify -> LayoutInference -> LowerTileOp  <-- 槽位 (ii)：RKNPU tile-op 实现
   -> LowerOpaqueBlock -> FlattenBuffer -> VectorizeLoop -> StorageRewrite
   -> SplitHostDevice / MakePackedAPI -> 设备侧 RKPlan 发射  <-- 槽位 (iii)
```

| 槽位 | 内容 | 为什么在这里 |
|---|---|---|
| (i) launch 物化 | `lower_thread_binding=False`；grid 只作为 program-index 空间 | 与 CPU 后端一致；不把 block grid 当时间序 |
| (ii) op lowering | 注册 RKNPU 的 gemm/copy/fill 实现（`register_*_impl`），在 `LowerTileOp` 的语句级匹配里被调用 | 匹配单元是单条 `Evaluate(Call)`，局部 region 自带边界；这正是 RKPlan task 该产生的位置 |
| (iii) tail | 由 RKPlan 发射 descriptor/命令；RKIR 仍是执行 IR | 复用既有 `Plan`/C 校验/runtime，不引入第二套执行器 |

**这解释了 §45 的阻塞点为什么不该用自建 walker 去解决**：`LowerOpaqueBlock` + `FlattenBuffer` 之后索引已经规范化，
尾块范围与 guard 变成可以直接读取的仿射表达式，比在本机 fork 的原始比较节点上猜类型可靠得多。
也就是说，阶段 2 的正确实现方式是**在通用主链之后**分析，而不是在原始 TIR 上手写 visitor。

### 46.3 与 §45 试点代码的关系（不删除，降级定位）

| §45 阶段 | 在全链架构里的对应 | 后续处置 |
|---|---|---|
| 0 capture | `BindTarget` + `MaterializeKernelLaunch` 之后的结构事实 | 保留为**审查/诊断用**的结构摘要，不承担 lowering |
| 1 regions | `LowerTileOp` 的语句级匹配（上游已实现） | 删除自建分类，改用上游匹配 + 我们的 impl |
| 2 k-partition | 上游 `FlattenBuffer`/规范化后的仿射分析 | 重写在前者之后；当前实现在尾块处仍不正确，已标记阻塞 |
| 3 tiles | SBlock 迭代 + 输出覆盖 | 保留为断言（coverage 检查） |
| 4 module | RKPlan 组装 | 保留，作为 slot (ii)/(iii) 的输出契约 |

### 46.4 修正后的迁移清单

1. 定义 `rknpu` BackendModule（target kind 先复用 `c`/自定义 kind 之一）+ `PassPipeline`，只放通用主链，`lower_thread_binding=False`。
2. 注册最小 tile-op 实现（gemm/copy/fill），先只做**记录**：把 region、索引、K 块、guard 写成 `StructuredModule`（复用 §45 的 dataclass）。
3. 在 slot (iii) 把 `StructuredModule` 变成 RKPlan（`plan_from_module`），与旧路径逐字段对照：task 几何、K 链、buffer、依赖、submits、publish、数值记录、输出。
4. mock + C 校验 → 板端对照；仅当新旧逐字段一致且设备输出一致，才把该入口设为 opt-in 默认。
5. 覆盖第二种源结构（单发射 GEMM）后再谈第三类；一条算子族全绿后才讨论删除旧 extractor。

**退出条件不变**：若某类结构在上游 TIR 层无法承载，必须定位到具体 pass 与限制并给出最小设计，不得以“保留旧 adapter”收尾。

### 46.5 尚未验证

- 本机 TileLang 的 `rknpu` 自定义 target kind 是否需要在 TVM 侧注册（CPU 复用 `c`/`llvm`，我们可能需要自己的 kind）。
- 通用主链中 `LayoutInference`/`LowerThreadAllreduce`/`ThreadSync` 在无线程目标上的行为：CPU 跳过 allreduce，我们是否也需要跳过并替换（待读 pass 体确认）。
- 设备侧 codegen 与 execution backend 的接口（`DeviceCodegen`/`ExecutionBackend`）能否只发 RKPlan 而不生成 C 源码。

---

## 47. 缝已跑通（2026-11-04）

实测：绑定 c target 后执行 BindTarget（tvm.tirx.transform）+ MaterializeKernelLaunch(lower_thread_binding=False) + Simplify（tilelang.transform），pilot 源码的 2 个 blockIdx 线程绑定循环变成普通 serial 循环，
For kind 序列由 [4,4,0,1,1,1,1,1] 变为 [0,0,0,1,1,1,1,1]，IR 中 blockIdx/thread_binding 出现 0 次（39 行 IR）。
证据：results/20261104T000000Z-structured-lowering/logs/thread-free-spine.txt 与 repros/run_spine.py。
含义：RKNPU 可以像 CPU 后端一样复用上游 thread-free 主链，§45 自建 walker 的定位收敛为 StructuredModule 契约与审查摘要；
§45 的 K 尾块阻塞点应改在 FlattenBuffer/Simplify 之后的规范化仿射式上解决。
未验证：rknpu 自己的 target kind / device codegen / execution backend，以及 FlattenBuffer 之后的尾块仿射式形态。

---

## 50. 结构化 lowering v2：语义靠证明，计划靠一致性闸门（2026-11-04）

### 50.0 大白话

上一版的结构化模块虽然能生成与旧路径逐字段相同的计划，但它的“语义判断”本身是从整体计数猜的（几个 shared、几个 fragment、
有没有含 FP16 的 copy），舍入还是调用者传进来的。本轮把它换成**逐语句证明**：每条 guard 的合取项、每次 fill 的值与覆盖、
回写的方向/类型/位置、操作数与参数的索引轴关系，全部从 TIR 节点推出来；舍入由计算推导，外部声明只做一致性检查。
`plan_from_module` 之后加了一道**一致性闸门**：planner 重建出的任务数、每个提交的 K 链、数值要求里的 K 链、tile 覆盖必须与
模块证明的一致，否则拒绝。

设备验证（真实 layer-0 gate/up 权重）：旧 22.755 ms、新 23.377 ms（5 轮中位），都是 36 任务 / 12 提交，**输出逐位相同**。

### 50.1 对 §1 所列问题的结论（逐项）

| 疑点 | 结论 | 现在的做法 |
|---|---|---|
| 用 shared BufferStore 数量判 guarded copy | 确认是缺陷 | 改为逐语句：在并行循环内找 `IfThenElse`，要求其合取项全为常量上界，记录 target/indexes/bounds/source；无 guard 的 shared store 直接拒绝 |
| 用“含 FP16 accumulator 的 copy”判回写 | 确认是缺陷 | 要求 `copy(fp32 fragment → fp16 fragment)`、**位置在 gemm 之后**、且存在 `copy(fp16 → fp32)` 在 gemm 之前；否则分别拒绝 |
| fill 数量不证明清零与覆盖 | 确认是缺陷 | 每个 shared tile 恰好 1 次 fill，值必须为 0，region extents 必须等于 buffer shape |
| `lower(source_rounding=...)` 接受调用者描述 | 确认是缺陷 | 签名改为 `declared_rounding`，仅用于交叉检查；舍入由“循环内有 FP16 回写”推出 |
| K 用参数形状启发式 | 确认是缺陷 | 由**索引关系**推导：哪个 operand 的哪个轴被 K 循环变量索引，那个轴就是 K，另一轴给 M 或 N |
| grid coverage 只比 extent | 确认不足 | 仍用 ceildiv 覆盖 + 每 tile 有效范围，但 M/N 来自上面的索引角色推导，输出参数由 `shape == [m, n]` 定位 |
| plan_from_module 重新构造 matmul graph | **属实** | 仍复用 `ir.matmul_module` + planner，但输出必须通过 `verify_plan` 一致性闸门；不一致即拒绝（新增 `PlanConformanceTests`） |

### 50.2 语义负例（离线，全部不送设备）

| 负例 | 结果 |
|---|---|
| 缺 A 的 guard | 拒绝：`a shared tile store inside the K loop needs a guarded copy` |
| 缺 B 的 zero fill | 拒绝：`each shared tile must be zero filled exactly once per K block` |
| 删除 FP32→FP16 回写 | 拒绝：`no FP32 to FP16 writeback into the running accumulator` |
| K 循环内出现未被消费的 guard store（写 Y） | 拒绝：`met a statement it does not consume` |
| 声明 `fp32_final_narrowing` 而计算是每块回写 | 拒绝：`declared source rounding disagrees with the proven rounding` |
| 变量重命名 | 接受，模块与未重命名一致 |
| 交换 grid 轴（索引同步交换） | 接受，grid/tile/slices/rounding 与未交换一致 |

### 50.3 结构转换的落点：谁消费了什么

| 结构 | 消费者 | 转换规则 | 转换后由谁承接语义 |
|---|---|---|---|
| 串行 K 循环 | `stage1_prove` + `stage2_k_partition` | 静态展开为有序切片（`k0/extent/physical`），因为 trip 数与 K 都是常量 | 计划的 `k0/k_index/k_count` 与数值要求的 `k_chain`（经一致性闸门核对） |
| 并行循环 + guard | `stage1_prove` | 作为“tile 级 guarded copy”的证明条件；循环本身不进入计划 | 任务的 `k_padding/k_physical_extent` 与 runtime 的零填充 |
| fill | `stage1_prove` | 证明“整块清零”后由 target 的零常量/零填充承接 | 计划的 padding 政策（既有） |
| copy（cast） | `stage1_prove` | 证明回写位置/方向后，由上/下载与 K 链承接 | `numeric_requirements` 的 `narrowing_points=k_tile_boundary` |
| buffer 分配 | `stage0_capture` 记录 scope/shape；规划器决定 RKNPU 存储 | 不机械映射 shared/fragment | 计划的 buffer/view/BO（证据不足的用法仍拒绝：见 50.5） |

### 50.4 计划一致性闸门（新增）

`verify_plan(module, plan)` 检查：任务数 = tiles × slices；每个提交的 `k_index` 集合 = 证明出的切片索引；
数值要求的 `k_chain` 长度与切片数一致；tile 有效面积之和 = M×N（恰好覆盖一次）。任何一条不满足即 `RK_E_*` 拒绝，
而不是“planner 怎么排就怎么算”。

### 50.5 真的换掉了什么、还依赖旧路径什么

| 项 | 状态 |
|---|---|
| guarded-tail GEMM 的**语义分析**（guard/清零/回写/舍入/操作数角色/K 链/tile 几何） | **已由新路径承担**（旧 extractor 不参与） |
| 计划的**组装**（buffer/view/BO/submit/publish/accounting） | 仍由现有 planner 承担，但受一致性闸门约束；本轮没有自己发射 buffer 与 submit |
| 其它形状（单发射 GEMM、staged FFN、含残差/激活） | 仍走旧 extractor |
| 未被消费的结构（例：K 循环内的其它 store、非矩形 guard、带副作用的调用） | 明确拒绝，不静默忽略 |
| RKIR / RKPlan 职责 | 保持区分：RKIR 是语义契约（值/算子/数值契约/来源），RKPlan 是执行契约（绑定/视图/动作/提交/publish/accounting）；planner 是决策层 |

### 50.6 验证与设备证据

| 检查 | 结果 |
|---|---|
| 结构化单测 | `tests/test_structured_lowering.py` 12 例全绿（证明、负例、合法变体、K=257、M 尾、计划一致性、C 校验） |
| 全量回归 | **755 passed / 2 skipped / 0 failed / 281 subtests** |
| 新旧计划签名 | 任务行、提交、publish、常量、设备/主机字节、BO 数逐项相同 |
| 独立 C 校验 | 新旧计划都 ACCEPT |
| 设备（真实权重，5 轮中位） | 旧 22.755 ms、新 23.377 ms；输出**逐位相同**；diff_ref 299760、diff_chain 332 两者一致 |

### 50.7 独立审查五问

1. **是否仍是第二套整程序模板提取器？** 不是形态上的模板匹配：判定依据是逐语句证明（guard 合取项、fill 值/覆盖、回写方向与位置、索引轴关系），
   且程序里出现未被消费的语句会拒绝。但**主要执行结构仍由 planner 组装**，所以只能称“语义分析已替换，计划组装未替换”。
2. **guard/索引/清零/回写的证明是否真实成立？** 成立且可复现：见 50.2 的五类拒绝与 50.1 的实现方式；
   guard 的合取项被逐条记录（expr/limit），fill 要求值 0 且 extents 等于 shape，回写要求 fp32→fp16 且在 gemm 之后。
3. **是否存在接受源码后忽略其计算的路径？** 不允许：K 循环体出现非 Fill/Copy/GEMM/guarded-store 的语句，或 guard store 落在非 shared buffer，都会拒绝。
4. **后续执行是否实际消费了新路径保留的信息？** 是，但**经 planner**：几何与 K 链来自证明，`verify_plan` 核对计划的 K 链/任务数/覆盖与证明一致；buffer/BO/submit 仍由 planner 决定。
5. **结构化转换在哪些地方已落地、哪些只是计划？** 已落地：guard/清零/回写/舍入/角色/K 链/tile 几何 + 计划一致性闸门；仍是计划：自建 buffer/BO/submit 发射、在 TileLang backend/pipeline 缝上注册、其它算子族。

### 50.8 证据与重放

| 物件 | 值 |
|---|---|
| 代码 | `python/rknpu_backend/frontend/structured.py`（v2，含 `verify_plan`）、`tests/test_structured_lowering.py`（12 例） |
| 计划 | 旧 `eefe49912249…`、新 `1064a6f6854d…` |
| 设备 | `device-new-vs-old.json`；库 `3749e95d…` |
| 脚本 | `repros/{probe_nodes,run_v2,save_plans}.py`、`device_run.py`、`board_run.py` |
| 日志 | `logs/module-v2.json` |

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_structured_lowering.py -q -o addopts=""
PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/20261104T020000Z-structured-semantics/repros/save_plans.py
```

### 50.9 尚未完成

- 自建 buffer/BO/submit 发射（当前 fork 到 planner，靠一致性闸门兜底）。
- 在 TileLang backend/pipeline 缝上注册（§47 迁移第 1 步）与 RKNPU device codegen/execution backend。
- 其它源结构（单发射 GEMM、staged FFN、带激活/残差）仍走旧 extractor。

## 正式 pass 编译试点（RKNPU，opt-in，2026-11-05）

### 入口

    from rknpu_backend.frontend import rknpu_pass as R
    plan, stages, facts, tasks = R.compile_guarded_tail(prim_func,
                                                        profile=..., options=...)

链：BindTarget -> MaterializeKernelLaunch(lower_thread_binding=False) [-> Simplify]
    -> pass_localise  (证明局部 tile op，并把它们改写成 RKNPU extern 调用)
    -> pass_instantiate (静态展开 grid 与 K 循环，按迭代域判定 guard)
    -> tasks_from_module (从 IR 读回任务序列)
    -> instances_from_tasks (归一为 m_major 顺序与升序 K 的 tile / K 切片实例)
    -> pipeline.compile_from_tasks -> dag.build_from_tasks -> _Builder.build_scheduled_linear(instances=...)

### 职责边界

- 上游 pass 负责规范化（保留 loop/If/buffer/index/cast/tile op/source location）；本地化分析不做 FlattenBuffer。
- `pass_localise` 是真正改写 IR 的阶段：每个 `tl.tileop.*` 调用被替换为 `rk_gemm/rk_copy/rk_fill` extern 调用，
  参数是已证明的事实（操作数缓冲名、grid 变量、tile 几何、K 变量与 K 宽度）。
- `pass_instantiate` 也是改写：循环被静态展开为单一 SeqStmt，guard 在迭代域上判定（真则删、假则丢、不确定则拒绝）。
- 计划由 `instances_from_tasks` 读回的 tile/K 实例驱动；planner 只做存储、submit、publish、accounting，并核对模块绑定与实例几何一致。

### 已知限制

- 仍只支持 guarded-tail GEMM 家族；K 循环必须静态有界；不支持的操作准确拒绝。
- 计划名/来源摘要沿用既有 RKIR 的绑定身份，不使用源码摘要作为语义证明。

## 第 65 轮：真实迭代域、完整路径映射、逐轴输出证明与注册（2026-11-06）

### 65.1 基线

HEAD `da4eeab`，其上为未提交重构。本轮开工时 `frontend/rknpu_pass.py` 的 SHA-256 为
`61f47489…19833b`（与交接单一致）；本轮结束时为 `da358490…b406493b`。
`git diff --check` 干净，未执行任何整文件回退、`git reset` 或未跟踪文件清理。

三条审核脚本在开工时的结果与交接单一致：前两份全部拒绝/通过如描述，
第三份 `tail_missing_row_guard` 与 `cross_grid_operands` 被接受且计划与对照完全相同。

### 65.2 修复的是信息接口，不是检查条数

修改集中在 `frontend/rknpu_pass.py`：

- `_prove_tile_index`（新增）：对一次访问的每个轴求 `index - local`，由索引本身（而非分析器能否证明为零）
  判定它命中的消费循环，并要求它恰等于 `loop × tile_extent`；返回变量与块表达式。
- 操作数副本：先用绑定域的 `_prove_under` 证明解引用安全，再求块映射；原来"按 extent==1 猜角色"的分支删除。
- 输出证明 `_prove_output_store`：在 epilogue 自身域上分类每个条件（消费/冗余/拒绝），
  逐轴证明"guard 界 = 逻辑长度"或"grid 精确覆盖"，并证明 store 索引在域内始终落在 `[0, logical)`；
  折零只在单块时接受，两个轴不得共用一个 grid 循环（否则同一块被写两次）。
- 交叉检查（新增）：A 的行块 = 输出的行块，B 的列块 = 输出的列块，A 的 K 块 = B 的 K 块；
  K 轴必须由 K 循环变量索引。这三条对应三个不同的源码位置，任一不符即拒绝。
- `_rewrite_tile_ops`：用证明得到的循环变量生成任务偏移，折零处写常量 0，不再回落到默认 `grid_m/grid_n`。
- `instances_from_tasks`：实例归一为 m_major + 升序 K，并新增 `_require_partition`：
  K 切片必须从 0 连续升序覆盖 `logical_k`，tile 矩形必须逐轴连续分割输出。
  这条修复了一个此前未发现的错误接受路径：K 循环少跑一块（如 576 只跑 2 块）会被静默接受。
- epilogue 之外的 `For` 中出现 tile op 或 allocation 现在直接拒绝（此前会被静默丢弃）。
- `pass_instantiate` 丢弃 `BufferStore` 前先核对目标：只允许两个共享 tile 与输出参数。
  其它目标说明有本路径从未证明过的状态改变，直接拒绝，而不是当作"已证明"丢掉。
- 任务序列：`rk_gemm` 参数来自证明结果；operand 副本与 accumulator 副本仍保留为任务记录，
  其合并依据是 65.3 的等式，不是"识别到 opcode"。

### 65.3 三条规则如何实际进入后续转换

| 源码事实 | 证明产出 | 计划侧消费 |
|---|---|---|
| grid 循环域（min/extent/step） | `_loop_domain` | `pass_instantiate` 用同一函数展开；域不符先拒绝 |
| K 循环域 | 同上 | K 切片数量与 extent |
| operand 读索引 | `(block_expr, var)` per axis | 与输出块表达式相等 = 该 operand 副本被 GEMM 上传吸收 |
| operand 读安全 | `0 <= idx < extent` 在域+条件下 | 上传区不越界 |
| epilogue guard | 消费/冗余/拒绝 | 消费则发布裁剪矩形；冗余则要求精确覆盖 |
| output store 坐标 | 块表达式 | `m0/n0` 直接取自该表达式 |
| 输出覆盖与唯一写 | 逐轴覆盖 + 轴不共用循环 | tile 实例与 `_require_partition` |
| Acc 零填充 / load / writeback / clear_accum | 位置与形状 | 单一有序累加链 + f16 回写 |
| 共享 tile 零填充 | 位置与形状 | K 尾块 padding 等价于短 K 切片 |

### 65.4 审核脚本复跑

    cd research/tilelang-rknpu
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-pass-review/reproduce.py --full
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-pass-recheck-64/reproduce.py --full
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-current-worktree-acceptance/reproduce.py --full

第三份结果：

    tail_base                 ACCEPT  f6e24e96…（与修复前相同计划）
    tail_missing_row_guard    REJECT  an unguarded epilogue axis needs a grid that covers it exactly
    grid_base                 ACCEPT  cee711d3…（与修复前相同计划）
    cross_grid_operands       REJECT  the A block coordinate and the output block coordinate are different computations

日志：`results/codex-pass-65/audit-scripts/`。

### 65.5 合法整体交换 grid 轴

`tests/test_rknpu_pass_pipeline.py::BlockMapTests` 用 M=N=512（2×2 grid）构造交换程序：
两个 grid 循环位置互换、全部索引用对应变量。证明按索引结构识别变量，
实例归一为 m_major，因此交换程序与普通程序给出**同一 plan_hash 与同一 plan dict**（cee711d3…），
mock 执行结果逐元素相同。

### 65.6 归一化后的实例顺序（更正上一节）

上一节写的 `instances_from_tasks (按程序顺序取 tile 与 K 切片实例)` 现在已改为
"归一为声明的 m_major 顺序与升序 K"。只有独立 tile 之间与 K 切片之间的次序变化；
`_require_partition` 同时证明每个输出元素恰好被写一次，所以重排不改变计算结果。

### 65.7 正式 opt-in 注册入口

`frontend/tilelang_backend.py`（新文件，341 行）：

- 本构建（tilelang 0.1.15 + 随附 TVM）**没有** rknpu target kind，且 kind 只能在 C++ 注册；
  `Target("rknpu")` 直接 ValueError。因此注册落在 TVM 的 `ext_dev` kind + target key `rknpu` 上，
  并把 `rknpu` 注册为 target normalizer 名称；裸 `ext_dev` 不会被本后端接管。
- 本构建的 lowering 缝只有一个整条管线 callable（`PassPipeline(name, LowerFunc)`，没有 PrimFuncPass）。
  该 callable 只做一件事：调用 `rknpu_pass.compile_guarded_tail` 一次，记录 `Lowering`，
  返回带 `rknpu_plan_hash/rknpu_stages/rknpu_tasks` 属性的模块。
- 设备 codegen 明确拒绝（本试点不产出 C 源码、不引入 runtime、不映射 GPU 线程语义）。
- `register_rknpu_backend()` / `registered()` / `unregister_rknpu_backend()` 显式；import 不注册。
- 无回退：lowering 只有一次调用、无 try/except、不引用 `tilelang_adapter`/`structured`；
  实测在一次注册 lowering 后两者都不在 `sys.modules` 里。

复跑：

    PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_tilelang_backend_registration.py -q -o addopts=""
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-pass-65/verify_registration.py

注册入口与直接调用对同一 PrimFunc 给出同一 plan_hash（`8741b4cb…`），阶段序列为
bind-target / materialize-kernel-launch / localise / instantiate / plan。

### 65.8 operand staging 为什么不是一个任务

源码里 `As[i, j] = A[im*256+i, ko*256+j]` 是手写的逐元素守卫循环，不是 `T.copy`。它被
`pass_localise` 证明、被 `pass_instantiate` 丢弃，理由不是"识别到了 copy"：

- 证明给出它的块表达式，并证明该表达式与输出块表达式、与 K 块表达式在同一个域上相等（65.2）；
- GEMM 任务的源切片（`m0/k0`、`k0/n0`）就是用同一批已证表达式生成的；
- 因此"逐元素把 A 的这一片搬进 As"与"计划上传 A 的这一片"是同一个映射，前者可以被后者吸收。

这条合并条件的有效性由等式承担；等式不成立时（交换块坐标、pinned 权重块）程序在
localise 就被拒绝，不会走到计划。`pass_instantiate` 现在还会核对它丢掉的 store 只落在
两个共享 tile 与输出参数上，避免"证明了一部分、丢掉了另一部分"。

accumulator 的 load/writeback 是 `T.copy` tile op，保留为 `rk_copy` 任务；
它们编码"每个 K 块累加、f16 回写"的语义，`clear_accum` 与部分回写都被拒绝。

### 65.9 新旧计划差异与正确性依据

`results/codex-pass-65/plan-diff.json`：

- 当前 pass 链计划（8741b4cb）与 2026-11-05 在板端实际执行过的计划（f3eb162f）
  **执行字段逐项相同**（任务行、submit、publish、字节、BO、上传下载）；
  差异只有记录用属性 `kt` 与 `source_hints`，它们不进执行。
- 与当前 `compile_tilelang(..., scheduled_baseline)` 参考构建器同样逐项相同。
- 因此本轮的设备结论可以按字段对应既有板端证据；没有新增执行结构。

## 第 68 轮：局部写入覆盖、参数类型契约与编译产物接口（2026-11-06）

审核对象是上一轮冻结文件（`rknpu_pass.py` `da358490…`）。旧反例与注册测试确认通过，
本轮修三条新问题；前两条在本节，第三条（编译产物接口）见 68.4。

### 68.1 局部写入覆盖：迭代次数不等于 tile 覆盖

反例（`results/codex-frozen-review/partial_staging.py`）：

    for i in T.Parallel(256):
        for j in T.serial(256):
            if im * 256 + i < 256 and ko * 256 < 576:
                As[i, 0] = A[im * 256 + i, ko * 256]

nest 仍是 256×256，read/store 的局部索引在两条轴上一致地都是 0，块表达式证明因此成立；
但 store 的列轴写的是常量 0，As 只有第 0 列有效，程序应得 3，计划给出 576。

缺的证明是 **store map 对 tile 的有效覆盖**。现在新增 `_prove_tile_coverage`，对每个逐元素
tile store（两个 staging copy 与 epilogue）要求：

1. 每条 tile 轴的索引必须是该 nest 自己的**一个**循环变量（可证相等），不是常量、不是重映射；
2. 该循环变量的 extent 必须等于这条轴的 tile extent；
3. 两条轴不得共用同一个循环变量（否则只写对角线/重复写一行）。

配合已有的"nest 必须迭代整块 tile"，这三条构成 iterate → tile element 的双射：
每个元素恰好写一次，所以循环 kind（parallel/serial）、是否重复写都不影响结果。
规则对 staging 与 epilogue 同一条代码路径生效，不是针对本例的常量列判断；
serial 也没有被加进拒绝白名单。

### 68.2 参数类型契约：拒绝而不是重写签名

反例（`results/codex-frozen-review/fp32_operands.py`）：A/B 与 As/Bs 为 float32，Cl 为 float32，
Acc/Y 为 float16。`localise` 与完整编译都接受，而 `_minimal_module` 硬编码
`ir.DTYPE_F16`，于是 plan binding 把 A/B 从 float32 改成 float16 —— 函数签名被悄悄改变。

现在 `_require_operand_contract` 在入口证明计划将要绑定的接口：A、B、Y 与两个共享 tile 的
dtype 必须是 float16，参数的行 stride 必须是连续的（显式 stride 时须等于 `(shape[1], 1)`）。
不满足就拒绝，错误里带上实际 dtype。`_minimal_module` 的 DTYPE_F16 因此不再是假设，
而是被证明过的接口；不再存在"有一个 FP16 accumulator 就当 FP16 源程序"的推断。

### 68.3 局部写入覆盖与循环域的区分

覆盖证明不改变循环域规则：grid/K/copy/epilogue 仍然只经 `_loop_domain` 取域。
新增的三条只作用于"索引写到哪里"，与"迭代多少次"分开证明，两者都需要，缺一不可。

### 68.4 编译产物接口：计划随模块返回，标准入口能完成

审核结论（§68.3）是：注册只接上了 lowering callback，标准编译入口仍然报错，产物要从进程全局侧取。
本轮补的正是这条接口。

**本构建的上游事实**（实测）：`tilelang/engine/lower.py:191` 在设备 codegen 之后调用
`codegen_mod.inspect_source()`，而这份 TVM（0.25.dev0，随 tilelang 0.1.15）的
`tvm.IRModule` **没有**这个属性（`hasattr` 为 False，import 了 `tilelang.backend` 之后仍然是 False）。
也就是说在本构建里，任何 backend 的 `engine.lower` 都会停在那一行。

**最小必要扩展**（不是把我们自己的函数套一层注册回调）：

1. `lower_module` 把完整计划（`canonical_json(plan.to_dict())`，15948 字节，内嵌 plan_hash）与哈希
   同时挂到返回的 **module 属性**与 **PrimFunc 属性**上。两处都必要且已实测：
   `Filter(is_device_call)` 只保留 module 属性、产出的 device module 没有函数；host module 保留函数。
2. 设备 codegen 不再一律拒绝：`build` 与 `build_without_compile` 都是
   `emit_plan_artifact`，把模块携带的产物重新挂上并返回；**只有携带不到产物时才拒绝**
   （那才是真正没被本链路 lowering 过的模块）。
3. `register_rknpu_backend()` 在缺失时安装 `tvm.IRModule.inspect_source`，并在
   `unregister` 时移除；它只回答携带 RKPlan 的模块，其它模块抛 `RK_E_UNSUPPORTED`，
   不编造源码。这是"上游接口不能承载时的最小扩展"，注释里写明了为什么需要它。
4. 公开、无全局的产物 API：`compile_to_plan` → `PlanCompilation`（plan / plan_json / plan_hash /
   stages / facts / tasks / module）、`plan_from_module`、`plan_from_artifact(CompiledArtifact)`、
   `plan_from_json`。反序列化复用 `serialize.canonical_json` + `Plan.from_dict`（会重算并校验 plan_hash）。
   `_LOWERINGS` 只继续服务 `last_lowering()` 诊断。

### 68.5 产物契约的实测

| 检查 | 结果 |
|---|---|
| 审核脚本 `verify_engine.py`（未修改） | `{"engine": {"status": "ACCEPT", "type": "<class 'tilelang.engine.param.CompiledArtifact'>"}}` |
| `artifact.kernel_source` 解析 | `kind=rkplan`，15948 字节，`plan_hash` 与 `compile_guarded_tail` 相同（ab873ba1…） |
| 清空进程全局后重建 | `plan_from_artifact` 的 hash 与 `to_dict()` 与原计划相同，`last_lowering() is None` |
| 篡改 plan_hash 的文档 | 拒绝：`plan_hash does not match canonical plan bytes` |
| 非本链路的模块 | 设备 codegen 拒绝；`inspect_source` 对空模块拒绝 |
| import 纯度 / 退注册 | import 后既无 shim 也未注册；离开 `registered()` 后 shim 移除、注册恢复 |
| 副作用 | `structural_equal(lowered, input)` 由 true 变 false：函数带了产物属性，**函数体未改动** |

### 68.6 仍未完成

- 板端复跑：本会话无设备与凭据（`adb devices` 空、SSH BatchMode 失败），未做设备实验。
- 支持域仍为 guarded-tail FP16 GEMM；tile 仍受 planner 的 256×256 / k=256 约束。
- `inspect_source` 是本构建的 shim；上游一旦补上该方法，`install_inspect_source()` 会返回 False 并不安装。
  范围说明：该 shim 只补`tvm.IRModule` 缺失的方法。本构建里其它 backend 在更早的 `Target context required` 处失败（注册前/中/后一致），因此本轮**没有**证明其它 backend 的完整编译兼容性，也不据此断言它们受到或未受到该 shim 影响。
- `unregister_rknpu_backend` 仍只能尽力恢复私有注册表与 shim（TileLang 无公开反注册 API）。

### 68.7 复现

    cd research/tilelang-rknpu
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-frozen-review/reproduce.py
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-frozen-review/verify_engine.py
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python results/codex-pass-66/lead_check_product.py
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/test_rknpu_pass_pipeline.py -q -o addopts=""
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""

证据目录：`results/codex-pass-66/`（审核三脚本复跑日志、pass 试点回归、全量回归、
`backend-product/` 的 API/产物记录与截取的前置状态、本文件的独立复核脚本）。

## 第 70 轮：累加链的 def/use 绑定（2026-11-06）

审核（§70）确认上一轮三项修复成立，并给出新反例：`T.gemm(As, Bs, Acc)`（写进 FP16 fragment），
随后 `T.copy(Cl, Acc)` 用 GEMM 之前的值覆盖结果，源码应输出 0，计划却按普通累加 GEMM 输出 576。

### 70.1 缺的是"同一条链"，不是方向或顺序

此前 `gemm_acc` 由分配 dtype 选出，copy 链只检查"存在 Acc→Cl 的 load、存在 Cl→Acc 的 writeback
及其相对位置"。tile GEMM 的第三个 region 只被记进 `rk_gemm`，从未与 `gemm_acc` 比较；
三个 region 的 offset/extent 也没有核实。于是"GEMM 写到别处、旧值覆盖"被当作正常链编译。

### 70.2 现在的连接检查

`_same_buffer`（同一分配：`same_as` 或 `data.same_as`，且 dtype/shape 一致；名字不作身份）
与 `_require_whole_region`（offset 全 0、extent 等整块）之后，`pass_localise` 一次证明完整链：

    fill(running, 0) -> load(running -> gemm_acc) -> gemm(tile_a, tile_b -> gemm_acc)
      -> writeback(gemm_acc -> running) -> store(Y <- running)
    staging: A -> tile_a, B -> tile_b（块坐标与输出块相等）

每条链接都用对象身份、dtype、region 与位置核实；GEMM 的 region[0..2] 必须分别就是 tile_a、
tile_b、`gemm_acc`，且为整块。§70 的反例因此在第三处被拒绝，错误信息带出两边的 buffer 与 dtype。

### 70.3 回归与证据

- 审核者的 `results/codex-pass-69-audit/reproduce.py`（未修改）现在给出 `compile: REJECT`；
  其目录在运行后已按 `SHA256SUMS.txt` 逐字节恢复，运行截图留在 `results/codex-pass-69-fix/`。
- 四份合法基线计划哈希不变（70d70281 / 8c2523aa / f6e24e96 / cee711d3），三份旧审核脚本的旧反例仍全部拒绝。
- 新增 4 条测试（GEMM 写入别的 buffer、operand 顺序颠倒、参考链、localise 层拒绝）；
  全量结果见 `results/codex-pass-69-fix/final-suite.log`。
- 无设备实验（无设备与凭据）；支持域未扩大，未切默认。

## 第 72 轮：精确操作表与 region 契约（2026-11-06）

审核（§72）给出两条：`T.cumsum` 这类本构建支持但试点不支持的 tile op 被忽略并改写成普通 copy；
accumulator copy 只比较身份与 extent，不检查 region 起点。

### 72.1 操作表：一个定义，两处使用，没有默认类别

`TILE_OPS` 精确列出本路径消费的三个操作（gemm / fill / copy）；`_tile_op_role` 对表外操作直接拒绝。
证明（`pass_localise`）与改写（`_rewrite_tile_ops`）共用它，原先的 `endswith` 归类删除。
"识别到了 opcode"因此不再等于"效果被落实"：认不出的操作不会被静默降级。

### 72.2 region：offset 与 extent 一起查，且真的被调用

`_require_whole_region` 现在覆盖 K 循环内每个 fill/copy 的全部 region、GEMM 的三个 region、
以及 K 循环外的 accumulator 清零 region。只比较 extent 或只比较 buffer 身份都不够：
起点偏移一行、extent 仍是整块、越出分配的 copy 现在被拒绝；显式写全切片仍被接受。

### 72.3 未支持的能力不实现

按审核要求，累计求和只拒绝、不实现；支持域仍是 guarded-tail FP16 GEMM。未切默认、未扩大算子族。

## W02 执行记录：控制、声明与 hint 的传递契约（2026-11-03）

本记录对应主审计 §74–§78。实现仍是 opt-in guarded-tail FP16 GEMM pass；没有切换生产默认、改变数值容差或扩大算子族。模块级 `rknpu.*` 属性在 registered backend 边界明确拒绝；请求需放在 PrimFunc 属性或有明确处理规则的 IR annotation scope。

### 执行和存储字段对应表

| 源字段/位置 | 处理方式与限制 |
|---|---|
| `SBlockRealize.predicate` | 只在无未消费调用、且对 enclosing loop/path domain 可证明恒真时省略；常假、选择性或不能证明的 predicate 拒绝。 |
| `IfThenElse.condition` | 对 guarded operand/output access，按完整路径条件证明或消费为对应 buffer 轴边界；实例化逐个静态域点判定可执行分支，无法决定时拒绝。Else 的路径条件是完整 `Not(condition)`，不拆坏合取。不能把所有选择性 If 都视为可忽略。 |
| `BufferStore.predicate` / `BufferLoad.predicate` | 无未消费调用且在所用域可证明恒真才接受；非平凡 mask 拒绝。Load mask 的证明采用完整程序 loop 域。 |
| `SBlockRealize.iter_values` / `SBlock.iter_vars` | 当前没有 block iteration binding lowering，非空时拒绝；两处分别检查。 |
| `SBlock.init` | 当前没有执行对应的 init lowering，非空时拒绝。 |
| `SBlock.match_buffers` | alias/view 当前不支持，非空时拒绝。 |
| `SBlock.alloc_buffers` 与 `buffer_map` | 每个参数及本地分配都要求 offset 为可静态证明的零、紧凑布局、无轴分隔；allocation 名称和 data 变量不歧义。实际 Load/Store view 必须与其声明在名称、dtype、shape、stride、scope 和 offset 上相符。本路径不实现通用 alias/view lowering。 |
| `For` 域和 launch binding | min/extent/step 受静态 0-based unit-stride 规则约束；原始 KernelLaunch 只接受唯一 `blockIdx.x/y` 绑定。launch-thread `Bind` 只接受 `tl.launch_thread_idx(0..2)` 且绑定变量未在表达式中使用；其它 binding 语义拒绝。 |
| statement/storage 容器 | `AllocBuffer`、`DeclBuffer`、`AttrStmt` 及没有已定义消费契约的 statement 拒绝。当前 vendored TVM TIRX 不提供 `Allocate`/`AllocateConst` 节点类。 |

### Call 位置与 GEMM 字段

调用只在确有消费者的位置允许：顶层 `Evaluate` 中的精确 `tl.tileop.gemm/fill/copy` 及其 region 描述、已验证且未使用的 launch-thread placeholder，以及实例化阶段消费的 `rk_gemm/rk_copy/rk_fill` 任务。其他 Call（包括会被算术折叠吞掉的 index/value/loop-domain Call）拒绝。tileop 实际参数数目、region 数目及嵌套 Call 位置都必须匹配消费路径。

GEMM 的 transpose、geometry、default policy、clear_accum、barrier、accumulate coordinates 都先解析为字面或可静态简化的值，再逐项校验。transpose 必须为 false，policy 为已支持的默认值，clear_accum 必须是已知 false，barrier 仅接受无 barrier 表示，coordinates 必须是零；常量表达式真 clear_accum 会准确拒绝，未知值也拒绝。tile copy/fill 只接受当前已消费的固定参数形式。

### 声明和纯 hint 的消费记录

支持的 `rknpu.*` key 是 `schedule`、`tile_m`、`tile_n`、`k_tile`。schedule 要求字符串 `mn_tiles_v1`；几何字段要求整数且与实际证明值相同。每个源 occurrence 保留 key、原表达式、scope、value、required value、消费阶段和结果。源 PrimFunc 与规范化 PrimFunc 的声明分别逐条校验并进入现有 facts hints；等值重复仍作为独立记录，任一 scope 的冲突值或未知 key/类型都会拒绝。规范化可以移动或简化节点表示，不以原始语法多重集相同作为接受条件；原请求仍会对最终证明几何验证并保存。

| annotation scope/key | 处理 |
|---|---|
| `SBlock.tl.launch_threads` | 要求三个正整数；记录原值并标记 `applied=false`，原因是 RKNPU plan 没有 GPU thread extent。 |
| `For.num_stages` | 要求正整数；每个 occurrence 连同 loop scope 记录为未采用，原因是 RKNPU 没有 software pipeline，K tasks 保持有序。 |
| `tl.tileop.copy` Call 的 `coalesced_width` | 要求正整数；记录为未采用，原因是 RKNPU copy 没有 GPU vector coalescing width。 |
| 其它非 `rknpu.*` annotation | 没有显式字段契约时按 scope 准确拒绝。 |

已记录的声明和未采用 hint 放入 `facts.hints`，随 RKPlan save/load 保留，并可通过 `explain_plan` 的 source hints 查看；它们不改变执行字段。PrimFunc 上的受支持声明在 pass 中逐条消费，IRModule attrs 中的 `rknpu.*` 当前明确拒绝，避免拆包时丢请求。

### W02 回归状态与证据

- 针对性声明测试：`tests/test_rknpu_pass_pipeline.py -k DeclarationContractTests`，4 passed / 79 deselected / 4 subtests，66.97 秒。此命令输出记录在执行会话；本轮 hint 增补前的早期版本曾有构造器类型失败，不能当作当前通过结果。
- 注册 engine 声明/hint/module 子集：`tests/test_tilelang_backend_registration.py -k 'registered_engine_consumes_duplicate_declarations or registered_engine_rejects_conflicting_and_invalid_declarations or module_level_rknpu_requests or registered_engine_records_copy_launch_and_pipeline_hints or registered_engine_rejects_unknown_copy_hint'`，5 passed / 29 deselected / 6 subtests，12.19 秒；原始日志 `results/handoff-w02-20261103/declaration-hints-engine.log`。
- module scope 正反例（含无请求普通 IRModule 正例）：`tests/test_tilelang_backend_registration.py -k module_level_rknpu_requests`，1 passed / 33 deselected / 2 subtests，6.61 秒；原始日志 `results/handoff-w02-20261103/module-scope.log`。
- 先前 W02 direct 控制/字段 targeted：20 passed / 59 deselected / 29 subtests，362.24 秒；针对 registered 控制/偏移/Call 字段 targeted：4 passed / 25 deselected / 15 subtests，22.28 秒。这两项早于声明/hint增补；最终结论以其后的完整回归为准。
- 主线程对首个四文件冻结候选完成两相关文件 pytest：10 failed / 111 passed / 66 subtests，1026.76 秒，exit 1，原日志 `results/handoff-w02-lead-audit-r3/pytest-related.log`。raw/materialised/simplified 的独立完整 C 执行 descriptor、常量与 contract 相等；其中 plan hash 仅差 source hint trace 的 `depth/ordinal` 与相应 hash。见 `normalisation-identity-summary.json`。按逐项证据完成获准的测试迁移后，主线程已在最终冻结版本运行完整 tests，包含这两文件，结果见下一项；首轮失败日志保留。
- 最终冻结候选由主线程运行 `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -rs -o addopts=''`：**902 passed / 2 skipped / 355 subtests passed，exit 0**。pytest 用时 1088.58 秒，driver 用时 1089.0038 秒；原始日志 `results/handoff-final-lead-audit-r4/pytest-full.log`（SHA-256 `3c07d8e7dacc34c2c0639be6d984cd2443919b3f64b823adb504b27d71145e16`），精确命令与退出码在 `pytest-full-command.json`。skip 1：transformers 未安装；skip 2：未设置 `RKNPU_SMOLLM_SLOW_TESTS=1`，未运行 package-sized mock replay。本全量包含两份相关正式测试文件。
- 本轮没有新建构，也没有真实设备测试；设备与历史执行字段等同性证据由主报告和后续独立审核说明，mock 不替代真机结论。

W02 的候选实现文件：`python/rknpu_backend/frontend/rknpu_pass.py` SHA-256 `e072e45e47d9913fe4913d7c4029a6b872bffb1478bb80aaab92ff24a6e927dd`；`python/rknpu_backend/frontend/tilelang_backend.py` SHA-256 `471c4e911ac0a25b8ce5f3f15b05cdece2ff9f17bd3d373384ef90fc45ca1463`。该记录仅描述本单实现与针对性检查，不替代主线程独立审核或 Goal 最终验收。

正式测试文件的当前冻结身份：`tests/test_rknpu_pass_pipeline.py` SHA-256 `5afd2796902035e91d5bf8c902bbfea44b222cfebbb22f48372a01da1e31b35d`；`tests/test_tilelang_backend_registration.py` SHA-256 `f564bfb33a1ec46e88beb9afe7b88011ad92ec4090da1b466a9098c371f9807b`。本单 targeted 命令/退出码/计数索引见 `results/handoff-w02-20261103/targeted-results.json`；注册测试原始日志随该目录保存。

## 第 81 轮：有状态效果的就地消费关系（2026-11-03）

审核（§81）给出反例：$`T.sblock("compute")$` 包住合法链之后 $`Y[0,0] = 0$` 被两个入口接受并丢弃，
计划 mock 输出 576 而应为 0。

### 81.1 两个缺口

1. $`_prove_output_store$` 只证明 $`_root_sequence$` 返回的那个局部 SeqStmt；祖先/兄弟序列从未被遍历。
2. $`pass_instantiate$` 按 buffer 名称放行并删除 $`BufferStore$`：同名 buffer 上的另一条写入也被删。
   "写到哪个 buffer"不等于"这次写入已被证明"。

### 81.2 规则

- $`_require_effect_closure$` 遍历**整个 PrimFunc**：每个 $`BufferStore$`/$`Evaluate$` 必须与证明阶段
  消费的语句按节点身份一致，否则拒绝（诊断给出语句类型、目标 buffer、下标）。
- $`LocalFacts.drained$` 记录证明过的 store 节点（不序列化、不改变计划字节）；$`pass_instantiate$`
  用 $`same_as$` 比对，名字不再放行任何删除。
- 合法的块封装仍接受，任务序列与未封装版本逐项相同（已入测试）。

### 81.3 证据

- 审核者四个探针（$`actual_source_probe$` / $`grid_sibling_store$` / $`outside_grid_store$` /
  $`outer_store$`）在两个入口全部 REJECT；其目录已按其 $`SHA256SUMS.txt$` 逐字节恢复。
- 计划不变：参考 hash $`12590578f133ecfa…$` 与 $`results/handoff-w02-lead-audit-r3/reference-anchor.json$`
  记录相同；三份旧审核脚本的四份合法基线 hash 与既有记录逐字相同，旧反例仍全部拒绝。
- 新增 6 条测试（块封装合法、块后/grid 内兄弟/grid 外兄弟/写输入参数四种额外写入、drained 恰为三条）。
- 全量回归见 $`results/codex-pass-80-fix/final-suite.log$`。

### 81.4 未做

无设备实验（无设备与凭据）；未实现额外赋值，按审核要求明确拒绝；支持域未扩大、默认未切。
