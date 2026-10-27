# TileLang → RKNPU 后端重构：执行规范与实施记录

日期：2026-10-27。初始状态：`not_started`。本文为执行智能体的任务规范；目标架构见 [refactor-overview.md](refactor-overview.md)。文中新增接口是目标要求，不代表当前代码已有实现。

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

- 实际基线 HEAD：待填写。
- 原有 dirty/untracked 归属及隔离方式：待填写。
- 锁定依赖与实际导入路径：待填写。
- 基线命令、结果及原有失败：待填写。

### 15.2 工作包状态

| 包 | 状态 | 改动/决定 | 证据 |
|---|---|---|---|
| R0 | not_started | — | — |
| R1 | not_started | — | — |
| R2 | not_started | — | — |
| R3 | not_started | — | — |
| R4 | not_started | — | — |
| R5 | not_started | — | — |
| R6 | not_started | — | — |

### 15.3 原生图与 RKIR 决定

- 选择的 TVM 表示、接入位置与依据：待填写。
- 直接路径、保留的目标契约及 pass 传播方式：待填写。
- 删除/简化内容与新增适配/测试成本：待填写。
- 是否采用默认原生路径及理由：待填写。
- legacy RKIR 的最终职责和依赖策略：待填写。

### 15.4 最终接口与兼容范围

- 实际模块安排和唯一核心调用链：待填写。
- artifact API 与旧入口适配：待填写。
- 权威表示、descriptor 生命周期与导出路径：待填写。
- C ABI/schema/API 变化：待填写；无变化也明确注明。

### 15.5 验证与交接

- 证据目录、实际命令与退出状态：待填写。
- 完整差分结果与允许的身份差异：待填写。
- Host / ABI / sanitizer / 设备结果及未测项：待填写。
- 最终提交、关键文件、遗留项与重放命令：待填写。

## 16. 自主执行与需要决策的边界

本规范已授权必要 helper、文件拆分、包组织、兼容适配、测试和证据脚本的有限改造。执行智能体自主推进 R0–R6、修复失败、选择实现和处理无收益候选；不因每包完成或某个局部方案未采用而停止等待确认。

只有以下情况提前提交具体方案：改变数值政策/支持域，新增未验证硬件配方，改变同步/资源生命周期，无法兼容的重要公共接口，破坏性操作，或缺少不可替代的外部条件。提问须说明实际冲突、已完成的可审查工作、可选方案及推荐取舍。

正常交付在完成整段工作后一次性汇报，并更新第 15 节。不要以增加文档、接口壳、mock 或测试数量替代真实编译到运行的闭环。
