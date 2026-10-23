# TileLang → RKNPU：Coding Agent 开发细则

初版日期：2026-10-20；范围修订：2026-10-23。本文是 coding agent 的目标规范：第1–9节规定行为，第10–16节规定文件落点、接口、工作包、测试命令和完成证据。当前已有部分实现；本文不逐项证明其已完成，实际状态以 `IMPLEMENTATION-STATUS.md`、代码及运行日志核对。现存 `co_*` 接口保持原有含义。架构背景见 [overview](overview.md)；开发决策以本文的明确契约为准。

执行范围：完成 P0–P8 的静态 FP16 MVP；P9 为后续扩展，不是 MVP 的完成条件。没有真实 TileLang 输入、只有手写 RKIR 的演示不算完成；只有 Host mock、没有真机证据时必须标记“主机阶段完成，板端未验收”。实现 agent 可以自行决定内部局部函数和数据结构优化，但不得放宽支持范围、数值语义、错误处理或验收条件。

### 任务边界与执行优先级

- 交付物是 TileLang 面向 RKNPU 的 IR 与后端。复用 TileLang 前端和 TVM PrimFunc/TileOp，不实现替代语言、源码解析器或通用前端。下文的 `frontend/tilelang_adapter.py` 与“提取”均指后端入口适配。
- 接入以已支持 primitive 的语义和静态数据依赖为单位，优先复用锁定版本的 visitor、分析与规范化 pass。函数参数个数、GEMM/copy 个数、变量名及某个 FFN 的固定写法不是能力契约；固定外形原型须如实列缺口，逐步用语义接入替代。
- 优先推进真实 TileLang → RKIR → RKPlan → custom-ops/runtime 的端到端闭环，再扩展范围。正确性、数值与资源安全要求保持；测试工具只围绕这些要求提供必要证据，不以无限补完 mock 框架替代后端交付。
- 已授权范围内的 helper、文件拆分、CMake 目标、诊断、测试组织和脚本依赖处理由执行 agent 自行决定，失败时保留最小复现并继续修复。只有需求或公共契约确需变更、破坏性操作或缺少必要外部信息时才请求用户决策，不为已确定路线重复请求确认。

## 1. 第一版范围与 TargetProfile

首版目标固定为 RK3588、厂商 RKNPU 提交路径、静态 FP16 rank-2 运算，M/N/K 为1..4096；沿用已有256上限的 primitive 配方。0维、负 stride、未证明的重叠、未知 dtype 和动态 shape 在规划前拒绝。此范围是实现政策，不是芯片极限。

`TargetProfile` 至少记录：

```text
schema_version, chip_id, driver_abi_id, runtime_abi_id
custom_ops_source_hash, generator_source_hash, capability_manifest_hash
tensor_layout_versions, template_versions
max_programs_per_submit=64, max_words_per_program=2048
primitive_extent_limit=256
execution_policy={synchronous, single_core_per_submit}
numeric_policy, fallback_policy, device_budget, host_budget
```

具体已验证内核/驱动范围引用 [能力总览](../../docs/CAPABILITIES.md)，不能只检查设备名称。首次实例化校验运行环境；不匹配应报 `RK_E_TARGET_MISMATCH`，不自动扩宽能力表。配置值在编译/创建时快照，不允许执行期间环境变量静默改变计划。

每个能力条目包含 `op/pattern`、输入输出 dtype/layout、shape 谓词、alias 谓词、数值契约、padding 规则、lowering 函数、emitter、completion class、可组合链配方、资源估计和证据引用。状态为 `enabled`、`experimental`、`unsupported`；只有 enabled 可默认选用。示意：

```yaml
id: co.mm_acc.f16.v1
op: rk.matmul_acc
emitter: co_emit_mm_acc
target_profile: rk3588_vendor_co_v1
inputs: [cube_f16, weight_f16, cube_f16]
output: cube_f16
numeric: ordered_k_tiles_with_f16_writeback
alias: prior_and_output_must_use_distinct_bos
chain_recipe: linear_k_acc_v1
evidence: research/kernels/FP16-K-TILING.md
```

这是需要实现的 manifest 格式。证据文件存在不代表所有可能 shape 均已验证；条目还须记录验证域、边界用例与未覆盖风险。

## 2. RKIR 数据模型

采用 Python frozen dataclass 表达 SSA 图，以版本化 JSON 持久化，不引入新的 MLIR dialect。JSON 由 Python 严格解码，通过明确的 C descriptor ABI 交给运行时；C 不解析 JSON，但必须独立验证 descriptor 的结构、容量和语义，不能信任 Python 内的预计算尺寸。具体 ABI 见第12节。

| 对象 | 必要字段 |
|---|---|
| Module | 版本、输入/输出、常量清单、节点列表、状态签名、源文件摘要 |
| TensorType | dtype、逻辑 shape、轴、逻辑 stride/index map、有限值/值域约束 |
| Value | ID、type、producer、输入/常量/临时/状态角色 |
| Op | opcode、operand/result IDs、attrs、source span、side effects |
| NumericContract | 输入/输出精度、请求的累加语义、允许收窄点、近似许可与验证容差 |
| RuntimeParam | 名称、类型、固定 shape、绑定类别、值域检查责任 |
| State | state_in/state_out、版本、提交组、并发限制 |

MatMul 统一写 `C=A[M,K] @ B[K,N]`。若源程序 `C += A@B`，记录初始 C 与更新顺序，不能把非零初值丢掉；第一版只接受明确的零初始化，非零初值必须通过独立合法化规则再放开。权重在边界转换为 custom-ops 的 `[N,K]` 逻辑约定并打包，不能把转置当作简单 stride 元数据消除。

显式 cast、mask、bias 和 activation 保留为语义节点。只在证明结果符合 NumericContract 后做融合；将独立 FP16 MatMul + Add 改为融合 DPU Add 可能改变收窄位置，不视为无条件等价。

静态递推循环可展开为有版本的 DAG，设置最大展开节点/任务预算。持续递推以每次调用的 `state_in → state_out` 表达；数据依赖 while 暂不支持。

## 3. RKPlan 数据模型与验证器

```text
RKPlan {
  header: schema/profile/source/capability hashes
  bindings: name, role, dtype, shape, allowed host layout
  constants: content hash or explicit version, pack_key
  buffers[]: id, role, capacity, alignment, layout_id, slot_id,
             lifetime, padding_state, ownership, must_not_share_bo[]
  views[]: buffer_id, offset_bytes, logical_extent, physical_extent,
           layout_id, valid_region
  actions[]: id, kind, template_id, inputs, outputs, attrs, deps, origin
  submits[]: ordered_task_ids, recipe_id, mode, core_policy, deps
  publish: outputs, state_versions, commit_dependencies
  accounting: peak payload/reserved/command/host bytes, transfers, tasks
}
```

`kind` 为 `host_pack`、`host_expand`、`host_check`、`upload`、`npu_task`、`download`、已注册 `host_compute` 或 `commit`。第一版 host packing 可与 upload 在实现中合并，但报告仍区分数据准备和算术回退。`co_upload_f16` 自带转换/同步时不要再重复 upload。

逻辑 buffer ID、allocation slot 和真实 BO 是三个不同概念。两个不同时存活的 buffer 可以共享 slot；两个必须独立的 buffer 不得共用 BO，即使 offset 区间不重叠。资源实例化必须保持该不变量，不能再把所有 slot 合成一个 arena。

验证器逐项检查：

1. 图引用、拓扑、定义先于使用、输出 shape 与每个模板签名。
2. offset/bytes/乘加无整数溢出，view 不越界，布局版本与对齐一致。
3. 每次读取均有数据生产和可见性保证，初值及 padding 已定义。
4. RAW、WAR、WAW 完整；存储复用引入的反依赖也进入执行顺序。
5. 生存期覆盖全部消费者、导出输出、Submit 完成与状态 commit；没有过早重用。
6. alias、独立 BO、workspace/数据 BO 隔离和计划资源预算。
7. Submit 配方、task 数、completion class 和 core policy；不能从“同 flags”推断任意链安全。
8. 所有 op 已消解，不存在未识别指令或隐式 Host fallback。

第一版顺序执行可用 `[first,last]` 区间；复用条件为 `old.last < new.first`。未来并发调度须以 happens-before 和完成事件证明复用，不能沿用一个任意拓扑序的位置判断。

## 4. 编译 pass 顺序

| Pass | 工作及产物 | 拒绝条件 |
|---|---|---|
| FrontendSubsetCheck | 从保留高层计算的 PrimFunc 提取源位置、循环、copy、GEMM/elementwise | 未注册操作、GPU 通信、不可证明索引 |
| NormalizeToRKIR | 统一 MatMul 方向、SSA、静态常量及状态边界 | 非零 GEMM 初值没有规则、动态控制流 |
| InferAndVerify | 类型、shape、索引、数值政策、值域责任 | dtype/shape/精度不匹配 |
| CapabilityLegalize | 选择模板族；显式广播、激活分解与布局动作 | 无 enabled 规则 |
| TileAndLayout | 选 M/N/K tile、pad、weight pack 与 feature tile 集合 | 物理几何/布局不合法 |
| BuildTaskDAG | 展开 primitives，插入 Host 动作和同步依赖 | 设备动态权重无合法布局转换 |
| ScheduleAndGroup | 保守拓扑序及白名单 Submit 分组 | 无可用链配方且无单任务路径 |
| PlanStorage | 按执行顺序求生命周期、slot、独立 BO 和峰值预算 | 预算不足、alias 冲突 |
| VerifyAndSerialize | 全量校验并输出 plan/diagnostics/解释报告 | 任意不变量失败 |

若调整 Submit 或调度导致生命周期变化，重新执行存储规划和验证。不可先定 slot，再任意重排任务。优化只比较合法候选，不通过试提交发现 shape 是否安全。

## 5. 物理布局、分片和 padding

沿用 [custom_ops.c](../custom-ops/src/custom_ops.c) 的构造函数作为权威布局规则：

- 单 feature tile：`Mp=round_up(max(Mt,12),4)`，`Cp=round_up(Ct,32)`，FP16 大小 `Mp*Cp*2`。
- Cube FP16 索引：`((c/8)*Mp+r)*8+c%8`。
- weight 逻辑为 `[Nt,Kt]`，两轴补到32；具体索引复用 `co_tensor_weight`/`co_index`，不再实现第二套公式。
- 每个 tile 的 M/K/N 逻辑及物理尺寸均要通过 emitter 谓词，不能仅验证原始逻辑尺寸。
- 跨图 Tensor 统一用 M-major/channel-minor tile 集合。已有 full-input Cube 的 K slice 可用 `co_tensor_input_slice`；不能用这一规则切任意 M 行或转置视图。

有效区域与 padding 必须分开。Host pack 首次补零；LUT sigmoid/HardSigmoid 会把零 padding 映射到非零值，因此不能给所有中间 Tensor 标注“padding 恒为零”。后续 MatMul 的 padded K 权重为零时，需要有限的 padding 才能避免 `0*NaN` 污染；按模板传播 `zero/finite/unknown` 状态，并证明无效行列不影响有效输出。需要恢复零 padding 时插入已实现的清理/重打包动作，否则拒绝该连接，不能假定硬件有任意 mask 写回。

slot 复用不保证旧数据为零。每次生产要说明覆盖范围，未覆盖的读取区域必须初始化；不能依赖 BO 首次 calloc 的效果。

## 6. 逐算子 lowering 规则

### 6.1 FP16 MatMul 与 K 累加

首版固定 tile 上限256，分块数为 `ceil(M/256)`、`ceil(N/256)`、`ceil(K/256)`。每个输出 tile 按 K 递增构建：

```text
p0 = MM(A[m,0], W[n,0])
p1 = MM_ACC(A[m,1], W[n,1], p0)
...
y  = last_partial
```

调用 `co_emit_mm` 和 `co_emit_mm_acc`；每次部分和写回 FP16。`prior` 与 `output` 的 type、布局、逻辑/物理尺寸必须一致，并且属于不同 BO；也不能与输入/权重重叠。分片尾部显式补零。没有累加数据时第一片不读取未初始化 scratch。

用两个 BO 交替，依据 K 片数的奇偶选择首次写入目标，使最终 partial 落在期望的 output slot；或令结果引用最终 BO，避免默认插入复制。后续 bias 只加一次，activation 只在完整 K 累加之后执行。非零初始累加值的融合仍需独立数值规则。

为控制第一版复杂度，K tile 重新划分只在 NumericContract 允许的集合内选择。K tile 大小会改变舍入路径，不能当作完全不影响结果的性能参数。

### 6.2 同形状 Add/Mul 与广播

`co_emit_ew` 经 CNA identity + DPU EW 执行，要求输入、另一个 operand、输出有相同 dtype、布局、逻辑和物理尺寸；输出不与输入重叠。identity weight 的两轴均为输入的 physical_cols。identity 按物理宽度缓存并计入权重、读取列表和预算。

广播政策：

| 来源 | lowering | 默认 |
|---|---|---|
| 完全同形状 | 直接 EW | 允许 |
| 编译期标量/每通道常量 | Host 常量展开至每 tile，打包一次后复用 | 允许，但需实现规则 |
| 运行时 Host 标量/向量 | 每次显式 `host_expand → upload → EW` | 需 `host_prepare` 策略 |
| 设备产生的标量/向量 | 下载、展开、上传的显式桥接，或独立广播模板 | MVP 拒绝 |
| 任意 NumPy 风格广播 | 无通用规则 | 拒绝 |

`co_emit_scale_native` 当前在源码中存在，证据局限于特定实验。登记为 experimental；其存在不改变上述默认政策。不能通过优化把受支持的 Host 展开偷偷替换为此路径。

### 6.3 LUT、SiLU 和 GELU

`co_emit_lut` 要求输入输出一致、FP16、Cube 或 Linear、互不重叠，且 `physical_elements/8 < 8191`。256×256 的完整物理 tile 不满足这个上限，不能直接一次发射。

复用现有图中的分段方法：对连续物理存储创建至多32768元素的 FP16 Linear view，覆盖全部物理字节，各段分别 LUT；保留原 Cube 解释供下一步 EW 使用。两段映射属于同一输出 tensor 的不同写区域，消费者依赖全部段完成。只能采用当前 `co_view_init` 能表示且已验收的分段，不发明新的跨 stride Linear view。

SiLU 为 `sigmoid(x)*x`，erf-GELU 为 `gelu_gate(x)*x`。gate scratch 与原 x 保持至 EW 完成；不能原位覆盖 x。HardSigmoid/sigmoid 可以直接返回 LUT 输出。

bounded exp 的输入域为 `[-4,4]`，exp-neg4 为 `[-4,0]`，rsqrt 为 `[0.25,4]`；当前接口不会执行值域扫描。规则必须指定以下一种责任：

- 编译期常量/范围分析证明；
- 已有 Host 输入校验；若数据在设备上，则在 hybrid 模式显式下载校验；
- 调用者明确接受前置条件，写入 plan 和运行报告，不能标作“已检查”。

默认为无法证明就拒绝；不能通过未声明 clamp 改变算子语义。使用值域检查也要区分有效元素和物理 padding，检查与模板评估策略均需覆盖后者。

### 6.4 状态更新与 hybrid 运算

第二阶段将已验证组合注册为白名单分解规则，不以“递推”作为任意计算的支持承诺。例如 GDN 路线的 feature-state、动态向量打包、六 task 顺序与具体数值契约都应显式进入 plan。

状态具有 committed 与 candidate 两个版本。执行只读 committed、写 candidate；全部 task、必要同步和验证成功后才交换状态句柄。如果一个时间步内部多次乒乓，不能覆盖仍需回滚的 committed BO，必须增加 scratch 或保留快照。跨多个时间步一次调用时，应在 ABI 中明确逐步提交还是整段事务；默认每次调用整体提交。

已有 RMSNorm/Softmax/Attention 等不能作为一个不透明“纯 NPU task”塞入计划。初期可用注册的复合 action 封装，但必须报告内部 Host 计算、传输、资源和同步，且不得跨该 action 擅自融合；随后逐步展开为统一 DAG。

## 7. 提交分组与资源规划

首版以已有 Linear 的每输出 tile K 链、激活 gate+EW 链作为配方，逐项匹配。独立输出 tile 默认分组执行；后续合并需要新的验证记录。没有已验证的组合链时，若每个单任务均已支持，则编译成显式顺序 `CO_SEPARATE`/单任务 Submit。

每组满足：

- 1..64个 program，每个 count 为1..2048；发射后再次检查实际 count。
- flags/completion class 一致，配方允许相应状态转换，依赖按拓扑排列。
- 组内没有 Host 动作；Host 检查/展开及设备到 Host 桥接构成硬边界。
- 所有 BO 属于同一 fd/domain，命令 BO 不与数据 BO 共用。
- 有依赖的链只在单核执行。固定核为1/2/4；允许 AUTO 时标明驱动选择不可预测。

选核调用已有 `co_schedule_pick`/`co_schedule_execute`；`co_execute` 不增加不存在的 mask 参数。两条执行入口封装在 adapter 中，不对同一组重复提交。后续允许跨核独立 job 时，必须新增完成依赖与线程安全设计。

超过64 task 的链可拆成多个同步组，部分和 BO 生存期跨组延续；若配方要求不可切分则拒绝。不因资源紧张启用未定义的 CPU fallback。

设备预算至少包括：

```text
sum(each allocated data BO payload + CO_SLACK + CO_GUARD)
+ command workspace capacity from rkt_chain_words(CO_MAX_WORDS)
+ other explicitly owned device resources
```

`CO_SLACK=32768`、`CO_GUARD=64` 按当前版本计算。命令 BO 按自身分配路径核算，不套数据 BO 公式。单独报告驱动页对齐可能增加的空间，严格预算模式使用分配器保证的上界；无法取得上界时不得称为实际物理内存上限。Host staging、常量、plan 和结果暂存也需单列预算。

## 8. 实例化与运行接口

建议生命周期：

```c
/* 以下均为待实现 API。compile/inspect 不需要 fd。 */
rk_compile(rkir, profile, options, &plan);
rk_plan_inspect(plan, &report);
rk_workspace_create(fd, plan, constants, &ws);
rk_run(ws, input_bindings, scalar_bindings, state, output_bindings);
rk_workspace_destroy(ws);
rk_plan_destroy(plan);
```

workspace 持有 plan 的不可变副本或引用、BO、views、packed constants、identity、程序数组、命令 workspace 与 Host 暂存。fd 由调用者持有并活到 workspace 销毁之后。第一版禁止外部任意 IOVA 导入；将来设备 binding 必须带 BO 身份、domain、容量、同步所有权，不能只给整数地址。

创建过程：

1. 验证 plan/schema/profile 和环境，计算预算；失败不产生半有效句柄。
2. 依独立 BO 约束分配 slot；通过 `co_view_init` 构造每个 view。
3. 上传/打包常量和 identity；按最大 Submit task 数创建执行 workspace。
4. 对绑定已经固定的任务发射 `co_program`。地址已绑定后可缓存，但不得跨 workspace/BO 重分配复用。
5. 校验实际发射 count、读写列表及模板预期；成功后发布 workspace。

当前 view offset 要求16字节对齐，实际 DMA 地址及 view 末地址必须落在32位地址范围。纯规划检查相对偏移与尺寸，实例化再检查分配结果；不能将64位 Host 指针截断后当作设备地址。

运行过程：

```text
lock workspace or return BUSY
validate bindings / alias / dtype / fixed shapes / scalar policy
prepare runtime inputs and candidate state
for action/group in verified order:
    perform declared host preparation and synchronization
    emit/re-emit only programs affected by changed bindings/parameters
    call selected existing execution adapter
    stop on first error
stage outputs; perform required finite/padding/guard checks
publish outputs and commit state
unlock
```

源码现有 `co_execute` 在提交后对各输出 BO 做同步；复用其行为，不能把“设备驻留”误说成完全没有 Host 同步。不要绕过 output handle 收集、链 pack/verify、终端封口和清理逻辑。

失败政策：提交前错误可安全清理；提交/等待超时等设备错误后，若无法确认 DMA 已停止，标记 workspace 为 poisoned，禁止立即重用/释放仍可能被设备访问的 BO，由驱动恢复/上下文销毁协议确认静止后回收。输出只在成功后从暂存发布；直接设备输出模式若未来提供，必须另定义部分写入语义。无状态运行的恢复和状态事务恢复分开测试。

常量缓存分两级：可跨设备的 packed Host bytes，以及绑定 fd/domain 的 device weight。缓存键包含内容哈希或显式版本、dtype、shape、transpose/stride、tile/packing 版本及数值相关策略；设备缓存再加 domain 身份。同一指针内容变化不会自动触发正确缓存失效，必须由绑定版本或哈希识别。

## 9. 可核对的完整计划示例

计算：

```text
X[13,257], B[257,33]
Z = X @ B
Y = SiLU(Z)
```

假设调用者明确允许按256划分 K、逐片 FP16 写回与 LUT 近似。仅一个 M/N 输出 tile，两个 K tile：

| View | 逻辑 shape / 物理 shape | payload |
|---|---|---|
| X0 | 13×256 / 16×256 | 8192 B |
| X1 | 13×1 / 16×32 | 1024 B |
| W0（B 的转置块） | 33×256 / 64×256 | 32768 B |
| W1 | 33×1 / 64×32 | 4096 B |
| P、Z、G、Y 各一个 | 13×33 / 16×64 | 各2048 B |
| identity | 64×64 / 64×64 | 8192 B |

```text
t0: MM(X0,W0)       -> P
t1: MM_ACC(X1,W1,P) -> Z       deps=[t0], BO(P) != BO(Z)
t2: LUT_SIGMOID(Z)  -> G       deps=[t1]
t3: EW_MUL(Z,G,I)   -> Y       deps=[t1,t2]
```

保守分组：S0=[t0,t1] 为 K 链，S1=[t2,t3] 为 gate+EW 链，S1 等 S0 完成；共4 task、2 Submit。全四任务合链作为单独候选，必须配方验收后启用。SEPARATE 对照为4次 Submit。

P 最后在 t1 使用，因此 Y 可以复用 P 的 slot；Z 与 G 到 t3 完成都存活。第一版 gate scratch 使用专用 slot，不参与普通 Tensor 的 first-fit 复用，输入 slot 也不复用，因此 G 不会先占用 P 的空闲 slot。scratch/output 三个 slot 的 payload 合计6144 B；输入两块9216 B、权重两块36864 B、identity8192 B，总 payload60416 B。若这些输入、权重与三个 scratch slot 各分配独立 BO，共8块，数据请求总量为 `60416 + 8*(32768+64) = 323072 B`，另加命令 workspace、页对齐和 Host staging。不同 BO 合并方案必须重新证明 alias 和余量约束，不能直接沿用这个数字。

输出只下载 Y 的13×33有效元素。输入/权重的 padding 在 pack 时清零；G padding 可变为 sigmoid(0)，不把它标记为零。此例的计划数值是规范推导，不是新增真机测量。

## 10. 开发环境与目录落点

### 10.1 开工检查

开始实现前执行以下只读检查，并把结果写入本模块 `IMPLEMENTATION-STATUS.md`。保留当前工作区已有修改；禁止用 reset/clean 恢复所谓干净基线。

```bash
git status --short
git rev-parse HEAD
cat research/upstream-components/provenance.json
cat research/custom-ops/CMakeLists.txt
```

必读源码：`custom_ops.h/.c`、`linear_plan.c`、`linear_hardsigmoid.c`、`co_graph.h/.c`、`co_tiled.h`、`co_schedule.h/.c`、`tests/c/graph_host_test.c`。具体路径均相对 `research/custom-ops/`。参考库读取 `docs/MM.md`、`src/rk_npu_mm.cpp`、`src/rk_npu_matmul_f16.cpp`；第一版不把参考库加入链接依赖。

基线构建和测试先执行第15节现有 CMake 命令。原有失败单独记录；不得通过删除测试或把失败标记为 skip 消除。发现不存在的兼容脚本、依赖缺失等基线问题，先定位再作最小修复，并与后端实现分别报告。

### 10.2 固定目录结构

除已有公共代码的有限提取外，新代码放下列位置。文档保持本目录，不把运行库放在文档目录。

```text
research/tilelang-rknpu/
  pyproject.toml
  IMPLEMENTATION-STATUS.md
  provenance.json
  python/rknpu_backend/
    __init__.py
    api.py                  # compile_rkir / compile_tilelang
    errors.py               # 稳定错误码、Diagnostic
    ir.py                   # RKIR frozen dataclass
    plan.py                 # RKPlan frozen dataclass
    serialize.py            # JSON严格校验与canonical编码
    capabilities.py
    frontend/tilelang_adapter.py
    passes/{infer,legalize,tile,dag,schedule,storage,verify}.py
    runtime/{ffi,bindings}.py
    cli.py
  capabilities/rk3588_vendor_co_v1.json
  schema/{rkir,rkplan,capability}.v1.json
  examples/{matmul,matmul_silu,swiglu}.py
  tests/{test_ir,test_capabilities,test_lowering,test_storage,
         test_plan_verify,test_frontend,test_ffi,test_cli}.py
  tests/fixtures/           # 小型IR/plan、恶意输入、预期诊断
  scripts/run-board.sh
  results/                 # 每次验收独立子目录

research/custom-ops/
  include/rk_plan.h         # 公开C ABI，前缀rk_
  src/rk_plan_internal.h
  src/rk_plan_validate.c
  src/rk_plan_runtime.c
  src/rk_plan_emit.c
  src/co_plan_contract.h    # 提取的内部纯shape/layout辅助契约
  tests/c/rk_plan_contract_test.c
  tests/c/rk_plan_host_test.c
  tools/c/rk_plan_probe.c
```

Python 使用 `numpy`、`pytest`、`jsonschema`；版本范围在 pyproject 声明，并保存实际验收环境锁文件。TileLang 作为独立可选依赖：不安装 TileLang 仍能 import 后端并编译 RKIR。`frontend` 延迟 import TileLang。禁止在模块 import、compile 或 explain 时打开设备。

新增静态库 target `rk_plan_core`（descriptor 校验，不访问设备），以及共享库 target `rknpu_plan`（供 ctypes 调用，链接 `rk_plan_core` 与 custom-ops）。需要共享链接的静态依赖显式启用 PIC；不要无理由改动所有已有 target。命名、符号导出、C11 编译选项加入现有 CMake。调度扩展接入后再链接 `co_schedule`，避免 custom_ops 与 co_schedule 形成循环。

### 10.3 必须复用与禁止复制的内容

| 当前代码 | 开发动作 |
|---|---|
| `co_tensor_*`、`co_index` | 作为实际布局权威；Python 的尺寸模型用 C contract 测试交叉核对 |
| `linear_plan.c` | 复用/提取范围检查和大小计算；不要调用会创建 BO 的函数做离线规划 |
| `co_graph.c` 生存期算法 | 复用其严格最后消费规则；新 planner 扩充到逐 task 与多输出 |
| `co_graph.c:emit_lut_chunks` | 提取共享 helper，或复用同一分段契约；不重复维护两个截断上限 |
| `co_tiled.h` | 提取公共纯布局辅助时保持旧调用行为；不要一次重写图编译器 |
| `co_execute_profiled` | 默认执行入口，复用 command workspace |
| `co_schedule_execute` | P9 显式逐组选核入口；MVP 通过独立进程固定核验收 |
| `graph_host_test.c` | 参考其链接器 wrap 方法，mock BO/提交；不把 CPU 解释器放入生产 runtime |

纯规划逻辑中禁止调用 `co_graph_compile`、`co_linear_create`、`co_buffer_alloc`。生产运行时禁止调用测试 mock 或悄悄调用 `co_graph_run` 代替 RKPlan 执行。旧 co_graph 可以作为数值和 task 统计对照。

## 11. 诊断、测试与验收

诊断必须包含 source span、op、实际条件、要求、目标 profile、失败阶段以及允许的替代政策。例如：

```text
RK_E_NUMERIC_CONTRACT at kernel.py:42
gemm requests fp32 accumulator; co.mm_acc.f16.v1 writes fp16 per K tile.
No enabled lowering satisfies the requested contract.
```

| 验证层 | 必须覆盖 |
|---|---|
| 前端与离线 | 不打开设备；未知 op、FP32 accumulator、非零初值、GPU barrier、动态 shape 拒绝；序列化往返 |
| 计划不变量 | 环、未初始化读取、尺寸溢出、BO独立约束、多输出/残差存活、复用反依赖、预算、64/65 task |
| 布局 | M=1/11/12/13/255/256/257；N/K=1/31/32/33/255/256/257及4096边界；转置打包和有效区域 |
| 激活 | 256×256 LUT 分段、值域边缘/越界、非有限值、padding、gate生存期 |
| 数值 | 数学 FP64 参考与声明的分片 FP16 路径参考同时比较；固定种子、多幅值、抵消与溢出 |
| 设备 | Core0/1/2 分别运行；chain/separate、重复执行、权重更新、guard、不同模板状态切换 |
| 故障 | 分配/上传/发射/Submit/同步失败，输出不发布，递推不提交，poisoned workspace 拒绝复用 |
| 性能 | 现有 custom-ops 与参考库同条件基线；cold/warm、预打包/动态权重、端到端和分阶段成本 |

硬件内部累加未建模时，分片参考只是声明写回路径的近似参考，不承诺逐位一致。阈值按模板和测试域登记，不能用一个全局 atol 覆盖全部形状。数值合法性和性能优势分别验收。

每轮验收保存实际参与构建的源码/依赖哈希、profile/plan、输入种子、编译选项、task/Submit 数、传输、误差与原始日志；实际核和频率在板端记录，无设备时标明未测。首版以 MatMul → SiLU 和 SwiGLU FFN 闭环作为准入，状态与 hybrid 在后续阶段单独准入。文档要求不构成已完成验收的证明。

## 12. 必须实现的接口与格式

### 12.1 Python 入口

实现以下签名；公共接口通过 `__init__.py` 导出，内部 pass 不直接访问设备。

```python
def compile_rkir(module: Module, *, profile: TargetProfile,
                 options: CompileOptions) -> CompileResult: ...
def compile_tilelang(prim_func, *, profile: TargetProfile,
                     options: CompileOptions) -> CompileResult: ...
def load_plan(path: str) -> Plan: ...
def save_plan(plan: Plan, path: str) -> None: ...
def explain_plan(plan: Plan) -> dict: ...

# CompileResult: plan, diagnostics, report
# plan 不保存 Python callable、TVM 对象、裸指针或fd。
class Workspace:
    @classmethod
    def create(cls, plan: Plan, *, fd: int,
               constants: dict[str, "numpy.ndarray"]) -> "Workspace": ...
    def run(self, inputs: dict[str, "numpy.ndarray"],
            *, outputs: dict[str, "numpy.ndarray"] | None = None
            ) -> dict[str, "numpy.ndarray"]: ...
    def close(self) -> None: ...
```

`CompileOptions` 冻结字段：

| 字段 | MVP 默认 / 校验 |
|---|---|
| `numeric_mode` | `strict`；合法化允许的另一值为 `rk_f16_tiled`，必须显式传入 |
| `allow_approx_lut` | false；激活示例显式设 true |
| `fallback` | `error`；`host_prepare` 仅允许第6.2节已有规则 |
| `submit_mode` | `chain`；另一值 `separate`，改变分组后重算计划 |
| `tile_limit` | 固定256；其他值 MVP 报 unsupported，不暴露无效调优选项 |
| `device_budget_bytes` | 256 MiB，按计划请求字节计；报告页对齐口径 |
| `host_budget_bytes` | 256 MiB，计入打包和输出暂存 |
| `max_actions` | 1,000,000；是总计划上限，不是单 Submit 上限 |
| `max_ir_nodes` | 4096；静态循环展开后也检查 |

strict 下无法证明契约兼容就失败，不承诺通用 GEMM 可用。`rk_f16_tiled` 表示调用者接受本 profile 的硬件计算与逐 K tile 写回契约；编译报告必须列出源数值请求和采用的路径，不能只保留一个布尔“fast math”。

NumPy binding 只接受 native little-endian、C-contiguous FP16、精确 shape；拒绝隐式 `astype` 和隐式 contiguous copy。输入有限值扫描发生在任何上传前。输出提供时检查 writable、size、与其他输出/输入不重叠；全部结果成功后发布。内部 pack/transposition 是计划明确声明的动作，不受此“禁止隐式 copy”限制。

### 12.2 JSON 与 descriptor 规范

JSON 根对象必含 `schema_version=1`、`kind`、`profile_hash`；IR/plan 的对象与 enum 按第2/3节完整定义进 schema。所有对象 `additionalProperties=false`，所有 ID 为从0开始的连续非负整数，所有尺寸/offset/count 为整数；禁止重复 key、NaN/Infinity、字符串数字和以 bool 冒充整数。跨引用和数值政策由 semantic verifier 检查。

canonical JSON 使用 UTF-8、排序 key、无多余空白；plan hash 对去除自身 hash 字段后的 canonical bytes 做 SHA-256。常量内容不内嵌大数组：引用名称、dtype、shape、byte length、SHA-256。plan 不带任意可执行导入路径。路径加载和数组内容由调用者绑定，C runtime 不根据 plan 打开文件。

layout ID 固定为 `co.cube.f16.v1`、`co.weight.f16.v1`、`co.linear.f16.v1`；MVP 不接受自定义 layout。模板 ID 固定为 `co.mm.f16.v1`、`co.mm_acc.f16.v1`、`co.ew_add.f16.v1`、`co.ew_mul.f16.v1` 及按激活种类分开的 LUT ID。禁止仅用一个“generic emitter”字符串动态分派。

C descriptor 使用自然对齐结构，公共字段选用 `uint32_t`/`uint64_t`，不使用 `#pragma pack`；所有结构首部含 `abi_version`、`struct_size`。数组以指针+count 传入，字符串经 ABI 传入时带长度，单次调用检查指针/count 配对。每种 action 对应固定参数结构或带 tag 的 union，不通过任意 JSON attrs 传到 C。

`rk_plan_desc` 必须能表示第3节的全部执行信息；C 复制后拥有独立存储，调用者可以释放 descriptor 数组。Python FFI 在 native 调用结束前保留所有数组及字符串引用。C verifier 重新计算容量和 action 依赖，不把 plan 中的 `verified=true` 当凭证。

### 12.3 C ABI

第8节是生命周期说明，落地函数以以下 ABI 为准：

```c
typedef struct rk_plan rk_plan;
typedef struct rk_workspace rk_workspace;
/* rk_plan_desc / rk_diag / rk_binding / rk_run_report
 * 的字段在rk_plan.h定义，对齐和sizeof须通过ABI测试。 */
int rk_plan_create(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag);
void rk_plan_destroy(rk_plan *plan);
int rk_workspace_create(int fd, const rk_plan *plan,
                        const rk_binding *constants, uint32_t nconstants,
                        rk_workspace **out, rk_diag *diag);
int rk_workspace_run(rk_workspace *ws,
                     const rk_binding *inputs, uint32_t ninputs,
                     const rk_binding *outputs, uint32_t noutputs,
                     rk_run_report *report, rk_diag *diag);
int rk_workspace_destroy(rk_workspace *ws, rk_diag *diag);
uint32_t rk_abi_version(void);
```

`rk_plan_create` 只访问 Host 内存，包含完整验证和深拷贝；workspace 再复制/持有 plan，创建成功后可释放 plan handle。`out` 在任何失败路径为 NULL；NULL destroy 为成功/无操作。workspace_destroy 可以返回 `RK_E_BUSY` 或 `RK_E_POISONED`，这些返回下对象仍归调用者持有，禁止悄悄释放。Python `close()` 同样传播错误，不把 GC finalizer 当作设备恢复方案。

`rk_binding` 含 binding ID、dtype、rank、shape[2]、data、nbytes；计算 Tensor 的 rank=2，标量/向量常量可为0/1，未使用 shape 项须为0。输入指针只在 run 期间借用，常量创建时复制或打包后不再借用。MVP 每次调用 synchronous；以 C11 原子 busy 标志拒绝同 workspace 并发 run，不能只在 Python 加锁。destroy 不允许与 run 竞争；API 要求调用者串行化销毁，并用文档与测试明确此边界。

运行报告至少有 task_count、submit_count、upload/download calls/bytes、pack/emit/submit_wait/output_sync 时间、数值模式与失败 action ID。没有测量的字段为 unavailable，不填0冒充实测。

错误码统一定义负值：`RK_E_INVALID=-1`、`RK_E_UNSUPPORTED=-2`、`RK_E_SHAPE=-3`、`RK_E_NUMERIC_CONTRACT=-4`、`RK_E_BUDGET=-5`、`RK_E_TARGET_MISMATCH=-6`、`RK_E_ALIAS=-7`、`RK_E_NOMEM=-8`、`RK_E_DEVICE=-9`、`RK_E_BUSY=-10`、`RK_E_POISONED=-11`、`RK_E_SCHEMA=-12`、`RK_E_DOMAIN=-13`。底层 co_/rocket 错误单独保留为 `diag.native_status`，不得直接混用同值错误码。Diagnostic 同时含阶段、op/action ID、源位置、违反条件与 profile。

### 12.4 TileLang/TVM IR 的后端接入契约

P1 锁定 upstream commit 后，从该版本的实际 PrimFunc/TileOp 节点提取，禁止解析 `str(prim_func)`、按变量名猜算子或仅运行独立 Python 示例而绕过 TileLang。

沿用上游的 AST、类型与遍历机制，选择保留 tile 语义的接入位置，并记录所复用的分析/pass 及必须新增的目标专用逻辑。无需重新实现 TileLang 语法。上游没有提供的访问/依赖证明可做受限实现，证明不了时精确拒绝，不以支持任意标量程序为目标。

必须识别以下语义模式，具体 AST 类名以锁定版本为准：

1. rank-2 FP16 buffer 参数；静态输出 tile 网格；无冲突的 store。
2. tile copy 的连续矩形索引，尾块条件严格等于逻辑 shape 边界；只允许补零读/忽略越界写，不接受任意数据相关 mask。
3. accumulator 的零 fill；有界 K 循环；GEMM 输入方向、转置标志、accumulator dtype 与 update 语义。
4. 无跨元素依赖的同形状加乘；注册激活的明确表达式或后端 intrinsic。
5. 多个 primitive 组合成的静态数据流，保留输入/输出/残差；出现未注册的指令立即报错。

参数角色由 buffer 的读写和数据流确定，不固定输入参数顺序或总数。每个受支持 GEMM、copy、cast、逐元素节点独立提取并构造 RKIR 的值与依赖；同一 buffer 跨阶段使用时保留版本和最后消费者。组合不要求整个函数命中单个 SwiGLU/FFN 模式。整图匹配只用于有独立数值依据的可选优化，不能绕过基础节点接入。未支持语义不得遗漏后生成部分计划。

MVP 后端采用256的 K 分片（最后一片可不足），这是当前配方与数值政策，不是 TileLang 的语法要求。接入保留源循环的归约顺序、累加类型和收窄点，再由合法化判断能否采用该分片；源 K 分块不兼容且没有明确许可时拒绝，不能在提取阶段随意合并。M/N 的前端 tiling 仅在证明同一输出覆盖一次且无通信时允许重排。前端 buffer scope 可以作临时值的逻辑分类，不能当成 GPU shared memory 已被映射。

若 TileLang 中 SiLU 写成 `x/(1+exp(-x))`，只能把整个已证明同形状的模式识别成 SiLU；单独出现的 exp 仍不支持。不能因字符串出现“silu”就匹配。通用 softmax 示例应明确失败。

激活表达式识别必须检查实际 dtype、cast 和写回位置。LUT gate 的 FP16 写回、SiLU/Mul 的写回、投影 MatMul 与 residual Add 之间的写回均属于数值契约；`allow_approx_lut` 只许可声明的 LUT 近似，不自动许可其他收窄变化。正例须与声明路径相容，缺失关键收窄的相邻程序须拒绝或经另有定义的显式许可处理。

每个允许模式提供一个真实 TileLang fixture，每个被拒绝模式提供相邻负例。未使用 RKNPU 时不改上游 GPU pipeline；对接优先用独立 adapter，确有上游修改则保存最小 patch、commit 和应用说明。

## 13. 工作包与闭环交付顺序

P0–P8 是交付清单，不是每包完全关闭后才允许进入下一包的串行关卡。实际依赖为：可用且可追溯的 TileLang/TVM 环境与 RKIR 契约支撑接入，RKIR/能力契约支撑 planner，RKPlan/C ABI 支撑 runtime，真实输入闭环支撑最终验收。先完成 MatMul 的最小纵向路径，再将同一套节点 lowering 扩展到激活、分支和残差；所需验证随实现补齐。P0 的额外源码构建复验或测试工具增强不阻塞已有可追溯环境上的独立开发；某次复验若未完成，其未完成状态必须保留，不能据此关闭相应验收。当前 P0 源码重建已完成，见 `P0-SOURCE-LOCK.md` 第 9 节。每包更新改动、实际命令、结果与剩余问题，不以接口空壳或 mock 全绿代替闭环。

### P0：基线与依赖锁

- 按第10.1节检查工作区并执行原有 Host 回归。
- 写 provenance：本仓库 HEAD+dirty 源文件哈希、portable generator/submit 版本、参考库版本、Python/C 编译环境。
- TileLang 使用官方仓库固定 commit 和其对应 TVM 依赖；不依赖可漂移的 main。能获取源码就继续，无需用户手工下载。网络/依赖阻塞写明具体失败，继续不依赖 TileLang 的工作。
- 产物：基线记录、provenance、独立虚拟环境与可安装的空包。
- 验收：`import rknpu_backend` 无 GPU/设备依赖；已有测试结果可审计。

### P1：TileLang/TVM 到 RKIR 的接入

- 在锁定 TileLang 上建立最小 MatMul PrimFunc，保存未经 GPU lowering 的 IR dump。
- 复用上游 IR 与分析，按第12.4节提取 GEMM/copy/fill/循环和数值语义，保留源位置；未支持节点默认失败。不为参数 arity、固定 copy 次数或整个示例形状新增特判。
- 写三个正例：单 tile、K=257、带尾块 M=13/N=33；负例含非零初值、动态 K、atomic/barrier 和未注册 exp。
- 定义并测试数值模式入口；不允许 FP32 accumulator 在 strict 模式被自动改为 FP16。
- 产物：接入位置/上游复用说明、薄适配器、fixtures、接入测试及受支持语义清单；该工作包不交付独立语言前端。
- 验收：真实 TileLang 对象生成 RKIR；修改一个索引或同步操作能触发预期拒绝。此阶段不要求访问 NPU。

实现补充（2026-10-23）：在原有单 tile/单 K-loop 正例之外，已接受“多个独立 `T.Kernel` 启动阶段 + 全局中间参数”的真实多 block projection/FFN/branch 源形式。每个阶段只写一个全局参数，后续阶段才可读取；同一并行 grid 内读取本阶段写出的全局参数会被拒绝，不做隐式同步假设。K-loop 必须每 256-wide K tile 显式写回 FP16 accumulator；FP32 跨 tile 累加后只收窄一次的源形式被拒绝。支持域和负例见 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)、`tests/test_staged_tilelang.py`。

### P2：RKIR/RKPlan/schema 与能力注册表

- 完成第2、3、12节的 dataclass、严格 serializer、schema、错误码与 manifest。
- 所有引用、类型、数值约束和未知字段有负例；缺字段错误与 unsupported 区分。
- 能力表只选择 enabled ID；未实现的声明不得注册 enabled。
- 生成 canonical plan hash 与 explain report。schema 变更同时改 fixture，不能悄悄兼容旧格式。
- 验收：IR/plan round-trip 稳定；重复编译 byte-identical；变更数值政策/packing 版本必须改变 hash。

### P3：MatMul planner

- 实现 shape 推导、B→W 方向规范化、固定256 M/N/K分片、尾块与任务展开。
- 按每输出 tile 的 K 顺序建链，显式初始化第一片，处理奇偶落点。
- 每条路径都生成 origin/source 与 NumericContract；不通过运行 co_graph 获取 plan。
- 完成纯 C contract helper 的有限提取，测试 Python 几何与 `co_tensor_*` 一致。
- 验收：第9节前两任务、K=1/256/257/512/513/4096的 task数与依赖正确；K>4096和非法 dtype 在分配前失败。

### P4：组合、存储与提交计划

- 实现同形状 Add/Mul、HardSigmoid/sigmoid、SiLU/erf-GELU；加常量标量/每通道广播规则。动态 Host 广播可在 P9 实现，MVP 遇到时明确拒绝。
- 实现 LUT 分段、gate 存活、padding 状态传播；残差与所有输出保活至 publish。
- 按 buffer 创建顺序逐个分配；候选空闲 slot 按容量递增、slot ID 递增排序，取首个足够大的 slot；约束相同 layout/alignment、旧 last < new first、非输入/常量/持久状态。gate scratch 单独预留一个容量为最大 gate tile 的 slot，在顺序执行中供各 gate 复用，不参与普通 Tensor slot 搜索；不复用输入 slot。
- 只合成白名单链；未知组合分成已支持的单任务组。为测试64上限构造独立 verifier fixture，不以 K>4096 绕过范围限制。
- 验收：第9节4 task/2 Submit、P/Y共享slot、P/Z不同BO、60416B payload 与323072B数据请求量全部断言；separate 为4 Submit。

### P5：C descriptor 校验与 FFI

- 实现 `rk_plan_create` 深拷贝、ID范围/尺寸溢出/alias/依赖/预算/模板校验。
- ctypes 字段定义与 C `sizeof/offsetof` 自动对照；不要手算结构偏移。
- JSON→dataclass→C descriptor→validate→destroy 整链可在无设备机器运行。
- 向 C 直接传入绕过 Python verifier 的坏 descriptor，确认仍拒绝；检查0/null/count错配、未知 enum 和过大数组计数。
- 验收：无 BO 分配和 Submit；ASan/UBSan 无错误；描述数组释放后 C plan 仍有效。

### P6：资源实例化、发射和 mock 执行

- 实现公共 ABI、常量打包、workspace创建与反向失败清理、emitter switch 和报告。
- 使用 `co_execute_profiled` 与预分配 workspace。实际 program 的读写 view、count、flags 与计划逐项核对。
- Host mock 以 BO/submit seam 为主要替换边界，生产 emitter 照常运行；必要的测试元数据不得污染生产 flags。根据调用记录检查分组、参数、地址域、缓存更新和生存期。mock 是有限的编排验证工具，不承诺硬件模拟精度或通用 program 支持；仅模拟本轮闭环所需模板，其余明确拒绝。
- 每个分配点注入失败，断言无泄漏；每组提交注入失败，断言不 publish。重复 run 输入变化应刷新，常量不应重复上传。
- 审查 submit 层每一种失败是否已保证设备停止。现有 `execute_impl` 的错误清理不是自动安全证明：必须追踪驱动返回及 command BO 生命周期。若不能保证，给已有 executor 增加最小、兼容的保留资源/错误分类接口，或把设备错误标记为需进程级恢复；禁止仅设置上层 poisoned 后仍提前释放命令 BO。
- 验收：mock 的成功/失败/重试契约成立；不确定设备错误不允许重试；原 co_graph/linear 回归保持通过。

mock 的签名表、标签追踪、容量和脚手架只需足以使所用测试可信。影响当前结论的缺陷须修复；不将通用复制支持、完整模拟器或独立测试框架完备性列为 P7 开工条件。bounded exp/rsqrt 等可选扩展未接入时可在能力匹配和 mock 中明确拒绝，无需为 MVP 另建模拟实现。已有工具与回归可继续复用，未完成增强单列，不删除失败证据或放宽生产契约换取绿灯。

### P7：CLI、Python run 与闭环示例

- 完成以下命令和 Python binding，添加 help/非法参数测试。
- `compile-tilelang` 使用显式 `--file/--symbol` 导入用户提供的模块；这是执行 Python 源文件的开发入口，`validate/explain` 不得导入模块。
- 示例全部经 TileLang→RKIR→RKPlan，不以内置 plan JSON 代替前端。同时提供 compile-rkir 作为调试入口。
- SwiGLU 示例表达两个 Linear 分支、SiLU、Mul、投影和 residual，经通用的受支持节点与数据流接入，不以单个整图特判替代该能力；可选融合另作验证。以已有 co_graph 对照，多个 output 的单独 fixture 检查生命周期。
- 验收：无设备 compile/validate/explain 可用；FFI mock 下绑定和发布正确；未编译 shared library 时 run 给出明确安装错误。

### P8：板端准入与最终交付

- 使用第15节板端入口，在同一板卡三个核分别跑 chain/separate；记录 profile 匹配与实际配置。
- 运行 MatMul、激活、MatMul→SiLU、SwiGLU、尾块及重复输入变更；进行独立数值校验和 guard 检查。
- 与当前 custom-ops 对照 task、Submit 和传输统计；不要求预设性能增益，但必须解释意外的上传下载及开销。
- 没有板卡时交付可运行 campaign、Host 全部结果和明确未验收清单；不能把 mock 报告写成硬件通过。
- 验收：第16节全部可用项达标，未完成板端项明确列出；无隐藏 fallback 或仅对示例 shape 特判的路径。

### P9：单独立项的扩展

按依赖分别实现动态 Host 广播、state/递推、hybrid、逐组调度、参考库模板和调优。每个扩展须增加 schema/能力条目、lowering、运行接口、负例及真机验收；不能只把 capability status 改为 enabled。状态 ABI 未实现前，对包含 State 的 IR 报 unsupported，不接受后静默忽略。跨核并发、原生标量探针、FP32严格累加、任意广播与参考库 split-K 均不得混入 MVP 的默认路径。

## 14. 测试清单与可执行断言

下表为必须落到测试文件的最小集合；不是用文档描述代替测试。用小型手工 fixture 检查语义，随机测试只作补充。

| 测试 ID / 文件 | 输入 | 必须断言 |
|---|---|---|
| F01 `test_frontend` | 真实TileLang GEMM，strict FP32 accumulator | `RK_E_NUMERIC_CONTRACT`与源位置；0次设备访问 |
| F02 `test_frontend` | 原子/数据相关mask/动态K/非零fill | 对应 unsupported，不能产生部分 plan |
| F03 `test_frontend` | M13/K257/N33，显式数值许可 | shape、方向、2个K片，dump能追到源GEMM |
| I01 `test_ir` | duplicate key、NaN、未知field、坏ID、shape=true | schema/invalid错误；不得转换后接受 |
| C01 `test_capabilities` | experimental scalar模板 | 默认拒绝；无 fallback |
| L01 `test_lowering` | K=256/257/512/513 | 每输出tile分别1/2/2/3个MM类task；奇偶结果正确 |
| L02 `test_lowering` | physical256×256激活 | LUT拆成2段，每段32768元素；SiLU共3task |
| L03 `test_lowering` | Add [13,33]+[33] | 普通tensor广播拒绝；显式常量广播生成展开动作 |
| S01 `test_storage` | 第9节示例 | 4task/2Submit/8数据BO，数值预算如第9节 |
| S02 `test_storage` | 分支残差、多输出 | 最终consumer/publish前不得覆盖残差或导出值 |
| V01 `test_plan_verify` | 同BO不重叠的prior/output | 仍报 `RK_E_ALIAS` |
| V02 `test_plan_verify` | 64/65task单组、PPU/DPU混合 | 64可通过其余约束；65原始组拒绝；合法planner拆组；混类拒绝 |
| V03 `test_plan_verify` | 删除producer、offset溢出、unknown padding | 拒绝；不能依赖calloc补救 |
| A01 `rk_plan_contract_test` | 布局边界集合 | Python计划与C `co_tensor_*`尺寸一致，16B偏移及32位地址检查 |
| A02 `test_ffi` | C结构/生命周期 | sizeof/offsetof一致；descriptor释放后plan有效；失败*out=NULL |
| R01 `rk_plan_host_test` | 第n次BO分配失败 | 所有此前分配回收，未开始Submit |
| R02 `rk_plan_host_test` | 第二组执行失败 | 输出哨兵不变；已分类安全失败可重建；未知设备状态poisoned |
| R03 `rk_plan_host_test` | 两次run输入不同 | 输入重复上传、常量仅创建时上传、无过期program绑定 |
| R04 `rk_plan_host_test` | 两线程同workspace run | 仅一个执行，另一个BUSY；无重复Submit |
| B01 板端campaign | 三核×chain/separate | 原始输出和独立参考、guard、重复执行及配置证据 |

数值测试分两层：小整数/二进制可精确表示输入用来发现方向、遗漏 K 片和布局错误；小幅随机 FP16（固定 seed）比较数学 FP64 与声明路径参考。P0 从现有验收引用每个算子的阈值来源，P3/P4 将其写入版本化 fixture；超出原证据形状/域的阈值作为待校准项，不以测试输出反向放宽容差。大幅值、抵消、Inf/NaN主要检验拒绝/失败策略，不能混入正常精度统计。

Host mock 不证明硬件数值、实际 IRQ、同核时序或真实 timeout 行为。硬件测试里不主动提交未经验证的危险配置来制造挂起；失败注入优先在 mock seam 完成。

按具名语义/故障场景报告覆盖，断言数量只作执行统计，不是完成度。测试基础设施的新增要求应对应具体后端风险；完成表内必要检查后继续实际闭环，不持续新增工具关卡。板端缺失、可选扩展和工具限制分别记录，不能相互冒充验收。

## 15. 构建、命令与结果目录

以下是实现必须提供并通过的命令接口；是否已可运行以对应版本的代码与实际日志为准。

### 15.1 Host 构建

```bash
# 已有基线；实施时同样用于最终全量回归。
cmake -S research/custom-ops -B research/custom-ops/build-host
cmake --build research/custom-ops/build-host -j4
ctest --test-dir research/custom-ops/build-host --output-on-failure

# 新增Python模块。
python3 -m venv research/tilelang-rknpu/.venv
research/tilelang-rknpu/.venv/bin/python -m pip install -e 'research/tilelang-rknpu[test]'
research/tilelang-rknpu/.venv/bin/python -m pytest research/tilelang-rknpu/tests
```

锁定 TileLang 安装方式写入 provenance 和模块 README，包含源码/TVM commit、实际安装命令，不提供一个会自动取最新版的隐藏安装脚本。pytest 中不依赖 TileLang 的测试始终执行；缺依赖时 frontend 测试可以标明 skip，但 P1/P7 不能据此标完成。

ASan/UBSan 使用独立 build 目录，测试包含 C descriptor 与失败清理；运行现有回归并记录基线差异，不把 sanitizer 参数永久加到发布库。涉及 FFI 内存风险时可用同一批插桩对象运行 Python 子进程；preload 与 worker 环境仅作用于该子进程。Python 兼容性排查有界，失败须记录未覆盖范围并继续独立开发，不能无限阻塞后端接入；真实内存错误仍须修复。poisoned 资源按契约保留的场景独立进程验证，不以此关闭其他 C 用例的泄漏检查。

验证脚本保存完整命令、实际参与构建文件的哈希和真实退出状态。环境白名单在覆盖/清理前记录继承值，构建与普通测试不继承 sanitizer worker 的设置。前置构建失败时，下游标记依赖失败并停止执行；独立分支与汇总继续，不把零测试或未运行写成通过。已有可靠手动证据可明确记为手动验收，自动化改进不应取代尚未交付的后端能力。

### 15.2 新 CLI 合同

```bash
python -m rknpu_backend.cli compile-tilelang \
  --file research/tilelang-rknpu/examples/matmul_silu.py \
  --symbol kernel --profile rk3588_vendor_co_v1 \
  --numeric-mode rk_f16_tiled --allow-approx-lut \
  --out /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli validate /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli explain /tmp/mm_silu.rkplan.json --format json
python -m rknpu_backend.cli run /tmp/mm_silu.rkplan.json \
  --bindings /tmp/mm_silu.bindings.json --output-dir /tmp/mm_silu-output
```

另提供 `compile-rkir <path> --profile ... --out ...`，参数政策同上。bindings 文件映射输入/常量名称到 `.npy` 路径；loader 用 `allow_pickle=False`，检查格式和大小；shape/dtype 由计划决定，不能从输入自动修改计划。输出 `.npy` 只在完整成功后以临时文件+替换方式发布。CLI run 自行打开受支持设备、按顺序销毁 workspace/关闭 fd；Python Workspace API 接受调用者 fd，两者所有权分开。

退出码：成功0，用户输入/编译拒绝2，资源或设备执行失败3。诊断写stderr；`explain --format json` stdout只能是单个JSON对象。CLI失败不留下看似有效的输出plan/结果文件。

实现补充（2026-10-23）：`run` 增加可选 `--fd N`。不提供时 CLI 自行 `rocket_open`；提供时使用调用者拥有的 fd 且不关闭。vendor `rocket_close(int)` 是 **void**，CLI 不读取或伪造关闭状态；只有 workspace destroy 确认静态后才关闭自有 fd。POISONED/BUSY destroy 后 fd/provider task BO 保留用于进程级恢复，CLI 报告 cleanup 并保留原始执行错误。板端 bundle 不携带 pinned TileLang，因此板端路径使用 `compile-rkir`，`compile-tilelang` 只在 Host 执行。RK3588 三个单核 mask 已逐例验证 compile/validate/explain/两次成功 run/缺少常量 binding 的 rc2 不发布路径（`board/core-*/cli/cli-report.json`）。

### 15.3 板端入口与证据

`scripts/run-board.sh` 以参数指定 `--host`、`--remote-dir`、`--results-dir`、`--core-mask`（1/2/4）、`--mode`（chain/separate），参考当前 run-graph-board.sh 的部署方式；不得硬编码个人SSH地址。每个核单独进程运行，环境设置在启动时完成，不在多线程中 setenv。

实现补充（2026-10-23）：实际准入入口为 `scripts/run-board-admission.sh` + `scripts/run-board-admission-native.sh`。连接信息只读仓库 `.env`；上传隔离源码/campaign bundle，远端重建 production `rknpu_plan` 后三个单核 mask 串行执行，并归档 source-bundle/library 哈希、per-round raw outputs、双参考、CLI report 和 fail-closed summary。

每轮结果目录至少包含：

```text
manifest.json             # 时间、版本/哈希、环境、完整命令、profile
plan.json
explain.json
bindings.json             # 相对路径，可重放
inputs/ outputs/
numeric-report.json       # 两种参考、阈值来源、max_abs/max_rel等
runtime-report.json       # 分组、task、传输、时间口径
stdout.log stderr.log
kernel-log.txt            # 能获取则保存；不能获取说明原因
summary.md                # 实际通过项、失败项、未测项
```

大矩阵只在结果目录保存，不提交巨型 fixture。CI提交小型可重放样例和汇总，原始数据位置/哈希留在 manifest。

## 16. 完成条件与 Agent 交接

每个阶段的状态只能为 `not_started/in_progress/host_passed/board_passed/blocked`；不需要板端的阶段可标 `complete`。状态文件必须关联实际测试日志，不以“代码已写”替代验证。

MVP 交付必须同时满足：

1. 真实 TileLang 三个示例从前端贯通；所有未支持语义有诊断，严格/近似政策可区分。
2. RKIR→RKPlan可离线重现；没有设备资源或进程地址进入持久化产物。
3. C verifier 不依赖 Python信任；binding、生命周期、BO独立及容量契约均受测试保护。
4. 每个 task 经已注册 emitter；每个 Submit 经现有执行基础设施，未复制新的寄存器生成器或提交器。
5. 第14节 Host测试完成，旧测试无新增回归；板端证据独立记录。
6. 文档中的接口、CLI、schema与最终代码一致；新增能力仍待板端时不改 CAPABILITIES 宣称本地真机支持。
7. 明确列出采用参考库的具体设计方法；未接入其代码/模板时不宣称已获得参考库性能。

coding agent 的最终交接按以下内容写：已完成工作包；改动文件；实际运行命令与结果；设备测试状态；数值/ABI/依赖变更；未完成项及可重放命令。若资源或网络阻塞某一项，完成其他独立项并报告具体阻塞；不得伪造版本、运行结果或验收数据，也不得通过悄悄删除困难需求宣布完成。
