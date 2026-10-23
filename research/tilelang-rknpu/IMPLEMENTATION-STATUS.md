# TileLang → RKNPU 实施状态

更新日期：2026-10-23（per-node/FFN/broadcast/ragged + RK3588 首次真机准入）。
当前结论：**核心 Host 闭环与 RK3588 小形状三核准入已通过；未达到 P0–P8 MVP
完成标准，仍禁止作为通用 release 版本**。

- 第一轮：[复核报告](REVIEW-2026-10-20.md) 阻断项 1–4 的原始三条反例。
- 第二轮：四个同源缺陷（未初始化内存、循环语义、常量 BO 生存期、覆盖性）。默认 `ctest`
  环境已开 `MALLOC_PERTURB_=128`。
- 第三轮：两处新缺陷（ABI 字符串复制越界、predicate/循环起点语义）。修复已静态核对，
  关闭证据本轮补齐：精确错误码、63 字符接受边界、predicate 四类负例、循环起点三类负例、
  可审计运行记录。
- RK3588 真机：27 cases × 3 个单核串行 mask = 81/81 PASS，包含多 tile projection/
  generic FFN/branch/activation、广播、双输出、K 奇偶分片和完整 CLI 路径；证据
  `results/20261023T121507Z-rk3588/`；仍属小形状准入，P8 未关闭。

四项证据语义复核项已关闭并留有回归：FP64 数学参考保持 float64、BO guard 位于最终
发布门禁、verifier 的 pair/字节/源包哈希语义收紧、target 使用完整 NUL 兼容项和精确
驱动字符串。逐项代码/测试/证据见 [REVIEW-CLOSURE.md](REVIEW-CLOSURE.md)；历史限制
（小形状、单 ragged 尾块、remote library hash 仅记录等）不隐藏。

MVP 收口矩阵见 [ACCEPTANCE-MATRIX.md](ACCEPTANCE-MATRIX.md)，支持域与明确拒绝项见
[SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)，运行期 BO/失败生命周期见
[RUNTIME-LIFECYCLE.md](RUNTIME-LIFECYCLE.md)，重放命令见 [REPLAY.md](REPLAY.md)。
P0 两次有界源码获取分别以 `exit 124` 和 `curl_rc=28` 失败，记录在
`provenance/p0-rebuild-attempt{,-2}.log`，仍为外部阻塞，未标完成。

## 最终 MVP 状态（2026-10-23）

- 实现域与有限代表矩阵：完成并真机通过。
- 原定多 tile 逐节点图：RKIR 与真实多 block TileLang staged projection/FFN/branch 均已贯通。
- 运行契约：target/binding/BO/budget/BUSY/pre-submit 安全/unknown→poison 保留、CLI cleanup/fd 均有代码、测试和日志。
- 全 MVP 尚未宣布关闭，仅剩两个具体条件：P0 源码构建外部阻塞；真实设备 submit/恢复故障未测。多板/并发/长压不列为前置，也不存在未定义的 “full release gate” 阻断。

## 1. 本轮修复对照

| 复核阻断项 | 本轮处理 | 证据 |
|---|---|---|
| 1. C 上传源内存边界未闭合 | `validate_upload` 要求 MVP 连续绑定 `source_row_stride == shape[1]`，重算最大源元素索引并检查乘加溢出与 `nbytes`；`upload_action` 在打包循环内再次检查索引 | `rk_plan_host_test` 的 `source_row_stride=0xffffffff` 负例；C 返回非 0 且 `*out==NULL` |
| 2. 执行顺序不受依赖约束 | 引入"发射序列 = action id 顺序"契约：每个 Submit 必须拥有**连续升序**的 task 区间且组内不得夹 Host 动作；新增单写者、写区间不重叠、读不早于最后写、slot 复用区间不相交检查 | `rk_plan_host_test` 的交换 K task 顺序负例、`w1`/输出同 slot 的 WAR 负例 |
| 3. M/N 多 tile 输出未实现 | `rk_publish_desc` 增加 `dst_row_offset/dst_col_offset/rows/cols`；publish 必须无重叠且精确覆盖输出 binding；runtime 按 tile 下载后拼装成完整输出，只有全部 tile 成功才发布 | C 侧 `multitile_publish_case`（M=257 两 tile 真实拼装并数值比对）；Python `MultitileTests`（257×33、13×257、257×257、513×600、4096×1） |
| 4. 前端语义检查不完整 | 提取器改为**顺序 + buffer 身份 + copy 数据流**驱动：参数角色由 copy 边（谁喂 GEMM 的 A/B、谁接收 accumulator）确定，方阵不再歧义；fill/copy/GEMM/激活/C 的顺序被证明；SiLU 要求循环层数、extent、store/load 索引与 loop var 完全一致 | `tests/fixtures/tilelang_negative.py` 十二个真实 TileLang 负例 + 方阵/SiLU 两个正例（均由真实 PrimFunc 对象构造） |
| 5. 完成范围不足 | **历史记录（后续已补）**：同 shape Add/Mul、K 循环、M/N 多 tile、SwiGLU 图 lowering、常量广播和逐节点多 tile 数据流均已补 | 见 P1/P3/P4/P7/P8 行与 [ACCEPTANCE-MATRIX.md](ACCEPTANCE-MATRIX.md) |
| 6. 设备准入与结果检查不足 | 下载后逐元素有限值检查，非有限即 `RK_E_NUMERIC_CONTRACT` 且不发布；上传/下载字节按**实际搬运的物理 tile 字节**统计；`rk_plan_desc.runtime_abi_version` 必须等于库 ABI；真实驱动/profile、选核与错误恢复仍未验证 | C 侧 NaN 注入负例；`rk_plan_host` 报告断言 |

第二轮复核四项（均已修复并各自固化反例）：

| 第二轮问题 | 修复 | 反例 |
|---|---|---|
| 有限值检查扫描物理 padding，读未初始化内存 | 只扫描 `co_download` 实际写入的逻辑 `rows*cols` 元素；padding 校验留给设备侧 guard campaign | CTest 默认设置 `MALLOC_PERTURB_=128`；修复前 10 项失败，修复后 0 项失败 |
| 前端静默丢弃包围 GEMM 的串行循环 | 任何 tileop 只要位于非线程循环内即拒绝；只有激活 store 允许在循环内（由精确循环校验处理） | 真实 TileLang `looped_gemm` fixture |
| 常量 BO 被后续输出复用 slot | C 校验器把"由带 `CACHE_CONSTANT` 标志或常量 binding 的上传所写"的 buffer 标记为持久，要求**独占 slot**；Python plan 校验同样规则 | C: 输出复用 w0 slot；Python: `test_constant_slot_is_exclusive` |
| 有 writer ≠ 读取区域已覆盖 | 物理几何必须等于 `co_tensor_*` 的规范值；每次读取必须被生产者**覆盖**：设备视图要求同 offset 同寻址，线性视图或线性写入的字节镜像允许按字节覆盖（LUT 分段契约） | C: 上传 12 行、MatMul 读 13 行被拒；改为 13 行后接受 |

第三轮复核两项（均已修复并各自固化反例）：

| 第三轮问题 | 修复 | 反例 |
|---|---|---|
| `origin[64]` 复制用 `strnlen(...,64)` 后写 `origin[64]=0`，未终止字段会单字节堆越界写 | 校验期要求每个 `origin` 在自己的 64 字节内以 NUL 结尾，否则 `RK_E_SCHEMA`；克隆期改为 `strnlen(..., sizeof-1)` 的**有界复制**，即使绕过校验也不会越界 | C: 把最后一个 action 的 origin 填满 64 字节；修复前 ASan 报 `heap-buffer-overflow WRITE of size 1`，修复后返回 `RK_E_SCHEMA` |
| 扫描器忽略 `SBlockRealize.predicate`；激活循环只查 extent 不查 `min` | predicate 必须可证明恒真（仅接受 `bool` 的 `IntImm` 真值），否则明确拒绝条件执行；激活循环要求 `min == 0`，否则拒绝"移位跳过/重复元素" | 真实 TileLang：`false_predicate`（守护条件恒假却仍生成 task）、`shifted_activation_loop`（循环从 1 开始） |

第三轮关闭证据补强（本轮）：

| 项目 | 补强内容 | 位置 |
|---|---|---|
| C 错误码 | 所有 descriptor 负例改用 `EXPECT_REJECT`，同时断言**精确返回码**、`*out == NULL`、`diag.code` 与诊断片段；不再只断言"非零" | `rk_plan_host_test.c` |
| origin 边界 | 未终止（64 字符）→ `RK_E_SCHEMA`；**63 字符 + NUL 必须被接受**，并断言克隆后 `strlen == 63`、`origin[63] == 0` | 同上 |
| predicate | 恒假、**自由布尔变量**、**未折叠的 `True and True`**、**嵌套 guarded block** 四类负例；恒真正例（既有 fixture）仍被接受 | `tests/fixtures/tilelang_negative.py` |
| 循环起点 | 外层 1、**内层 1**、**负起点 -1** 三种，分别断言诊断里的 `[1, 0]`/`[0, 1]`/`[-1, 0]`，确认拒绝原因各自成立 | 同上 |
| 可审计记录 | `scripts/run-verification.sh`：保存完整命令、退出码、逐步日志、环境与依赖身份、源码哈希、ctest 比例、pytest passed/skipped（含 skip 原因） | `results/<run-id>-host/` |

验证脚本可复现性收尾（本轮，由复核指出的三处缺口）：

| 缺口 | 修复 | 证据 |
|---|---|---|
| sanitizer 配置依赖旧缓存 | 每轮**删除并重建**两个 build 目录（`--keep-build` 才保留），configure 行显式传 `-DCMAKE_C_FLAGS`/`-DCMAKE_EXE_LINKER_FLAGS`/`-DCMAKE_SHARED_LINKER_FLAGS`，并在配置后断言 flags 已进入 `CMakeCache.txt` | `configure-sanitize.log`、`compile-flags/cmakecache-flags-sanitize.txt` |
| 归档实际编译/链接参数 | `CMAKE_EXPORT_COMPILE_COMMANDS=ON`；把真正的 compile commands 与 `link.txt` 归档到 `compile-flags/`，并断言其中含 `-fsanitize=address,undefined` | `compile-flags/compile-commands-{host,sanitize}.json`、`compile-flags/link-sanitize-*.txt` |
| 环境记录早于环境设置 | 先 export `PYTHONPATH/LD_LIBRARY_PATH/TILELANG_CACHE_DIR/RKNPU_PLAN_LIB/ASAN_OPTIONS/...`，再采集并测试；同时清理 `.pydeps` 中 `apache_tvm_ffi`/`z3_solver` 的重复 metadata，并把"重复发行版名"作为**失败条件** | `environment.log`（含 export 后的实际路径、`duplicate_distribution_names=[]`） |
| 证据生成不计入失败 | `source-hashes`、`archive-flags`、`summary` 都成为受检查步骤；缺文件/统计无法解析时写 `evidence_complete: false` + `evidence_problems` 并返回非 0；失败轮仍写出 `manifest.json`/`summary.md` | `results/20261020T085356Z-host/`（注入 cmake 失败的负对照，`overall_exit: 1`） |

扫描器"直接忽略的节点"审计（本轮逐项定性；**本轮审计范围内未新增已复现缺陷**）：

| 节点/字段 | 处理 | 理由 |
|---|---|---|
| `Bind` | **仅**接受 `value` 为 `tirx.launch_thread_idx(...)`；其它绑定明确拒绝 | 启动线程索引是元数据；其它绑定携带真实数据值 |
| `AssertStmt` | 条件必须可证明恒真，否则拒绝 | 运行时断言是前置条件，丢弃它等于让设备跑源程序拒绝的输入 |
| `LaunchThread` | 允许（仅线程身份元数据） | 不携带数据值；数据语义一律经 region 常量索引进入，已被限制 |
| 线程绑定循环 | 要求 extent == 1，多 block 明确拒绝 | 更宽的 grid 会在多 block 上并行同一 tile 空间，本后端未 lowering |
| block `iter_vars` / `iter_values` | 必须为空，否则拒绝 | 块迭代绑定会改变索引含义 |
| buffer `strides` / `elem_offset` | 参数与 GEMM tile buffer 必须是普通行主序且 `elem_offset == 0` | 非默认 stride/偏移会被当作平坦 [M,K] 使用，属静默语义丢失 |

P9 范围校准（按 [specific.md](specific.md) 第 13 节）：

- **运行时动态 Host 广播**、**逐组选核调度**属于 P9，不作为 P0–P8 完成阻断项。
- P0–P8 仍需完成的是：编译期常量展开（标量/每通道）、既定单核执行策略、以及三核分别
  验收。本文件此前把这些扩展项与 MVP 收尾混列，现已按指导文件分开。

第四轮复核两项（均已修复并固化反例）：

| 第四轮问题 | 修复 | 反例 |
|---|---|---|
| RKIR lowering 未核对真实数据流：把 `Z=X@B; S=SiLU(Z); Y=S*R` 的 Mul 输入改成 `Z`，两份 IR 都被接受且 **plan hash 相同**，实际仍算 `SiLU(Z)*R` | `legalize` 按 Value ID 校验整条链：activation 必须消费 matmul 结果、add/mul 必须消费前一步结果（交换律允许两侧任一位置，但 lowering 使用**解析出的操作数**）、module 输出必须等于链末端、不得导出链外 value；`dag._emit_binary` 改用 `legalized.binary_operand_id`，并在 lowering 起点再次复检边关系 | 新增 `tests/test_dataflow.py`（8 项）：连接链接受、交换操作数等价、`Z*R` 破坏链被拒、SiLU 换输入被拒、导出中间值被拒、不同合法图 hash 必须不同、自引用操作数被拒 |
| `--keep-build` 必然拒绝已有构建缓存：保留目录后仍无条件要求无 `CMakeCache.txt` | keep 模式改为**校验**已有配置：cache 存在且 `CMAKE_HOME_DIRECTORY` 等于本源码树、sanitizer 树带 `-fsanitize=address,undefined`；fresh 模式仍要求无缓存 | `results/20261022T084651Z-host`（keep 模式 rc=0）；`results/20261021T115302Z-host`（keep 模式下剥掉 sanitizer flags → rc=1 并给出明确错误） |

第五轮复核三项（K 循环路径）：

| 问题 | 修复 | 反例 |
|---|---|---|
| K 分支提前返回，绕过未注册 tileop 检查；还可能漏掉循环外多余 GEMM | 未注册 tileop 检查移到 K 分支**之前**；K 路径新增"节点必须全部被消费"：循环外只允许一个零 fill + 一个 accumulator→输出 copy，任何多余 tileop（GEMM/copy）明确拒绝 | `bad_unknown_tileop_in_loop`（`T.transpose` → `tl.tileop.transpose`）、`bad_extra_gemm_outside_loop`、`bad_extra_copy_outside_loop` |
| K 路径缺少内部 buffer stride/offset 检查 | 抽出 `_check_plain_tile_buffer`，单 tile 与 K 路径共用：GEMM 三个操作数 buffer 必须普通行主序且 `elem_offset == 0` | 两条路径共用同一诊断；负例见 frontend 既有 stride 检查 |
| 接受源 `block_k=128` 后合并为 256，与 specific.md 12.4 冲突 | 强制 **语义 K 分块 = 256**（`block_k != 256` 直接拒绝，附 `required=256` 与替代方案），不再在提取阶段重切分 | `bad_block_k_128`（K=512/BK=128）与 BK=64 均被拒 |

本轮收口（对应 specific 条款）：

### P5 · ABI 全量对照（specific §12.4 / P5）

- **实现**：`rk_plan_contract_test` 由 `include/rk_plan.h` 自动生成 `ABI_SIZE`（11 个结构总大小）
  与 `ABI_FIELD`（**138 个字段**的 offset/size）；`tests/test_ffi.py` 逐结构比对
  `ctypes.sizeof`、**字段集合必须完全相同**、每个字段的 `offset` 与 `size`。
  已删除"至少 50 项"的门槛，计数、行列偏移、预算、`runtime_abi_version` 等此前遗漏的字段
  全部覆盖。
- **仍有限制**：只比对当前头文件声明；C 端不校验 Python 侧 `_fields_` 的顺序语义（顺序由
  偏移比较间接保证）。
- **可支持结论**：字段交换、遗漏或结构总大小变化都会被测试发现。

### 前端与计划验证收口（specific §12.4 / P1、P3）

- **实现**：① Call 白名单——内核体中任何非 `tl.tileop.{fill,copy,gemm}` 的调用一律拒绝
  （`unhandled call`），K 分支也在其之前；② SiLU 内部的 `BufferLoad.predicate` 必须恒真或
  缺省；③ 激活循环 `step` 必须为 1；④ 计划校验先确认 task ID 是存在的 NPU task，再做配方
  匹配（不再可能抛 `KeyError`）。
- **仍有限制**：白名单以锁定的 TileLang 版本为准；非 tileop 的纯算术仅允许出现在已证明的
  SiLU 表达式中。
- **可支持结论**：四类非法输入各自命中目标规则并返回约定诊断。

### C 独立校验（specific §12.4 / P5）——范围已收窄

- **已实现并验证**：
  ⓪ **模板 arity 先于数组遍历**：`validate_template` 先用 `template_arity()` 表校验
  `ninputs/noutputs` 与模板一致，**之后**才遍历 `inputs[]/outputs[]`；另有全局上界
  （`RK_MAX_ACTION_ARGS`=8，`RK_MAX_ACTION_DEPS`=64）。伪造 `ninputs/noutputs = UINT32_MAX`
  的两条 C 负例在 ASan 下返回 `RK_E_SCHEMA`、`*out == NULL`、无越界访问。
  ① **task ID 先于模板读取**：每个 Submit 先做 `task_id == UINT32_MAX || task_id >= nactions`
  的范围检查与 "必须是 NPU task / 不得重复归属" 的类型检查，**之后**才读取 `template_id`
  做序列白名单；首位置与后续位置分别用 `nactions`、`UINT32_MAX` 覆盖，ASan 下断言
  `RK_E_SCHEMA`、`*out == NULL`、无越界。
  ② **预算零 claim 不再绕过**：`claimed_device_bytes` / `claimed_host_bytes` 为 0 直接
  `RK_E_BUDGET`；Host 暂存按「各 binding nbytes + 全部输出同时暂存的总量（下限 64 KiB）」
  独立重算。
  ③ **多输出预算边界**：`multi_output_budget_case` 用**两个独立输出 binding**（两个 MM 各
  发布一个 tile、各一个 download、两个 Submit）。Host 预算按「x + w + 两个输出」的
  binding 之和 + 暂存**两个输出同时存在**（下限 64 KiB）独立重算；claim 恰好等于边界 → 接受，
  **低一字节** → `RK_E_BUDGET`，0 → `RK_E_BUDGET`。早期三次失败均为用例构造错误（cube tile
  的 K 超过 256、weight 误加转置标志、value 用物理字节而非逻辑字节），已修正。
  ④ Submit 任务序列白名单（全 MM 链、MM→MM_ACC 延续、LUT 连续段±末尾 EW 乘、单个 EW）。
  ⑤ emit 布局 / primitive 上限 / alias 条件检查保留。
- **仍未做（不得声称已收口）**：
  · C 不解析 `recipe_id`（ABI 无该字段），只按任务模板序列校验形状；

  · **device 预算已逐项复算并有精确边界**：每 slot 取 `max(offset+capacity)`
    + `CO_SLACK` + `CO_GUARD`，再加 `max(len(submit.task_ids)) × CO_MAX_WORDS × 8` 命令区
    （**注意是最大单个 Submit 的任务数，不是任务总数**——上一轮脚本正是按总数计算，
    导致 C 得 205056、脚本得 221440 的假"口径不一致"；两侧常量其实都是
    `CO_MAX_WORDS = 2048`）。多输出用例上 claim 恰好等于边界 → 接受，**低一字节** →
    `RK_E_BUDGET`，1 → `RK_E_BUDGET`，均已断言。
- **可支持结论（收窄）**：越界 task id、伪造 arity、零 claim、非注册链、多输出预算越界
  五类坏 descriptor 在 `rk_plan_create` 被拒并返回 `RK_E_SCHEMA`/`RK_E_BUDGET`、
  `*out == NULL`；合法边界（含多输出恰好预算）保持接受（C 138 checks 全通过）。
- **不可支持结论**：不能称 "C 校验已与 Python 对等收口"。

### SwiGLU / FFN（specific §12.4 复合模式 / P7）

- **能力项**：`co.swiglu_ffn.f16.v1` 保持 **experimental**；`allow_experimental=True`
  只选择整图专用 lowering，不再是 canonical 图能否编译的前置条件。默认走通用
  per-node 路径。
- **已在 primitive 层取证**（`graph_host_test` 的 `execution_case`，本轮起其 check 数也写入
  manifest 的 `c_checks.graph_host`（2,400,033 checks），走 `co_graph` 真实
  emitter + mock NPU，与参考实现逐元素 `memcmp` 相等）：
  · 非方形权重两组形状 (m=13,d=257,h=289) 与 (m=257,d=17,h=33)；
  · 残差 add（参考实现含 `+x`）；
  · **失败不发布**：强制第 2 次执行失败 → 输出 buffer 全部字节保持 `0x23`；
  · **多次输入变化**：本轮新增——同一 plan 改输入后重跑，结果与重算参考一致且与首次不同，
    随后再插入一次强制失败仍不发布。
- **rk_plan 路径上的现状（本轮进展 + 明确缺口）**：
  · `rk_workspace_run` 已有 **MM 链数值对照**（2 task、改输入重跑、NaN 注入 →
    `RK_E_NUMERIC_CONTRACT` 且输出字节不变）；
  · **EW 行向量猜测已纠正（此前记录有误）**：查 `co_emit_ew → finish_mm` 后确认
    `co_emit_ew(p, x, id, o, &b, mul)` 的读取表是 `reads[0]=x`、`reads[1]=identity 权重`、
    `reads[2]=逐元素操作数`，identity 契约是 `[x.physical_cols, x.physical_cols]`；
    数学语义是 `y = x ⊙ b`（或 `x + b`），**没有行广播**。936 项失败正是 mock 误把
    `reads[1]`（identity）当作操作数所致。
  · 由此已补齐 **plan 级 LUT+EW 数值回归**（`lut_segment_case` 内，经
    `rk_plan_create → rk_workspace_create → rk_workspace_run`）：两轮不同输入，参考实现用
    独立逻辑行主序 + 声明 FP16 收窄点（LUT 输出、EW 输出各舍入一次），逐元素严格相等；
    NaN 注入 → `RK_E_NUMERIC_CONTRACT` 且输出哨兵 `0x23` 不变。
- **数值契约（B2 第 1 步，已完成）**：`tests/fixtures/tilelang_ffn.py` 现有两个真实 kernel——
  `swiglu_ffn_f16`（**目标正例**：显式写出 gate/up→FP16、SiLU→FP16、乘法→FP16 三个收窄点）
  与 `swiglu_ffn`（**数值契约负例**：全程 FP32 表达式）。两者当前都被 arity 拒绝；
  若将来支持 FP32 版本，必须另行定义覆盖这些收窄的许可与验证，**禁止静默改写**。
- **测试专用 mock 共享库（方案 b，已交付构建）**：`librknpu_plan_mock.so` 由
  `rk_plan_runtime_obj`（**同一批 runtime 对象**）+ `tests/mock/npu_mock.c` 链接而成，
  mock seam 抽到 `tests/mock/` 供 C 测试与 mock 库共用；通过链接期 `--wrap`
  （`rocket_bo_alloc32/free/prep/fini`、`rocket_batched_submit_supported`、
  `co_execute_profiled`、`co_emit_lut`、`co_emit_ew`）替换，**不新增 `rk_plan_load`、
  不让 C 解析 JSON**（specific §2）。生产 `rknpu_plan` 不链接 mock、不会自动回退。
  导出 `rk_mock_execs/live_bos/resident/allocs/fail_at/fail_alloc/inject_nan/
  tag_count/tag_overflow/reset` 供 ctypes 核验包装生效。**本轮 mock 正确性修复**：
  BO 结构先 `memset` 完整初始化、`handle` 改为唯一递增的模拟 id（`0x40000` 起、32 位内，
  不再截断 Host 指针）；标签表容量不足时置 `mock_tag_overflow` 并由执行器硬失败
  （`ROCKET_E_SHAPE`）；FFN 用例的 NumPy backing array 改由**每用例 holder** 持有，
  清理时释放（全局 `_KEEP_ALIVE` 已删除）。**仍缺**：program 复制/地址复用后的标签
  刷新、非对角权重方向自检、bounded exp/rsqrt 建模或拒绝、独立 Add/Mul 小用例、
  故障注入 API 的统一头文件与 ctypes 签名、独立 Submit 故障用例（当前 NaN 用例只证明
  数值检查后不发布）。

新增具名 mock 场景（最近两轮）与**实际结果**：

| 场景 | 结果 |
|---|---|
| 每个模拟 BO 的独立 32 位 DMA 区间（16B 对齐、不重叠、`MOCK_DMA_BASE`+递增） | **通过**；表/句柄/DMA 空间耗尽返回 `ROCKET_E_NOMEM` 且不分配 Host 内存 |
| BB 结构完整初始化（`memset`）+ 唯一句柄（`0x40000` 起，非指针截断） | **通过**（`rk_mock_bo_count`/`rk_mock_dma_of` 可查） |
| 全部 emitter（MM / MM_ACC / LUT / EW）建立标签 | **通过**；容量不足 → `mock_tag_overflow` → 执行器硬失败 |
| **未追踪程序硬失败** | **已达成**。此前"runtime 重新物化 program"的结论**是错的**：`rk_plan_runtime.c:423` 直接把 `&batch[index]` 交给 emitter，随后提交同一 `batch`，指针不变；当时 36 项失败的真实原因是我误删了 EW 分派分支。现按 `nreads` 猜测的回退已删除，未追踪 → `ROCKET_E_SHAPE` + `rk_mock_untracked` 计数 |
| **DMA 区间自检**（新增） | **通过**：对每个模拟 BO 断言 16B 对齐、落在 `[0x01000000, 0xF0000000)`、`rk_mock_dma_of(handle)` 与表内记录一致、句柄唯一、区间两两不重叠，workspace 销毁后 `rk_mock_bo_count() == 0` |
| **对齐溢出** | **已补**：`size > SIZE_MAX - 15` 直接 `ROCKET_E_NOMEM` |
| **EW weight index `(c,q)`** | **已修**：mock 原用 `ident[q][c]`（转置），WEIGHT layout 应为 `W[c][q]`，即 `pass[r,c] = Σ_q x[r,q]·W[c,q]`。因 identity 对称，旧测试无法发现；修为 `(c,q)` 后既有用例仍全绿 |
| **EW 方向回归（已通过）** | 契约：identity 上传带 `TRANSPOSE_SOURCE`（Host 按 `[K,N]`），device `W[c][q] = host[q][c]`。用例只设 `host[0][1] = 3` → device `W[1][0] = 3`。**直接实证**：`rk_mock_find_half(3.0)` 命中 identity BO 的 `flat_half = 32`（32 宽打包 atom 的 (row=1,col=0)），并**以断言而非打印**固定在测试里；输出侧断言第 1 列按 `pass[r][1] = acc[r][1] + 3·acc[r][0]` 变化、其余各列逐元素等于未扰动参考（含 sigmoid 的 FP16 写回）。**29 与 14 的差异（已核正）**：两次都同时存在对角翻倍（`identity[cols*cols-1]=2`）与非对角扰动。29 那次 EW 操作数期望写成 `x`（应为 LUT 的 `sigmoid(x)`）；14 那次已修正该期望，因此失败数下降，但主因仍是对角翻倍污染多列。删除对角翻倍后为 **0 失败**。对角内容检查若恢复，必须使用**独立初始化的 identity**，不得在同一数组上累积扰动。**结论更正**：此前"上传打包与读取顺序冲突"**无证据，撤回**；两处都调用同一个权威 `co_index`，实测方向与打包一致。**索引反转验证**：临时把 mock 权重读取改为 `(q,c)` → `rk_plan_host` 690 checks / **30 failures**；恢复 `(c,q)` → 690 checks / **0 failures**。该实验**只捕获了 stdout，未归档日志**，事后也未补造；补丁未提交（文件当即从备份恢复），复现步骤与局限记录在 `research/custom-ops/tests/mock/INDEX-REVERSAL-EVIDENCE.md` |
| **`bo->dma_address` real field** (new) | **pass**: the allocator writes the simulated DMA into `rocket_bo::dma_address` (the field the emitters patch into registers). New `bo_dma_field_case` allocates two BOs directly and asserts: field non-zero, 16-byte aligned, inside the window, `dma+size` in range, equal to the side table and to `rk_mock_bo_dma_address`, second BO non-overlapping, and after free both the table and the live counter are zero |
| FFN 运行期计数 | `untracked = 0`、`tags = 2`、`overflow = 0`、`live_bos 13 → 0`、`resident 503296 → 0` |

期间的三处自伤（EW 分支在补丁中被删、`own_workspace` 递归、快照广播）均已定位修复，
最终 `rk_plan_host` 138 checks、pytest 130 passed、ASan 干净。
  · 已补 `ffi.py` 的 `create_workspace / workspace_run / workspace_destroy /
    mock_control`（ctypes，复用 `rk_binding`/`rk_run_report`/`rk_diag`），
    并写了 `tests/wip_ffn_execution.py`（planner→FFI→mock runtime 的完整 FFN 对照，
    参考实现用 numpy + 显式 FP16 收窄）。
  · **未通过**：绑定构造仍失败——`constant binding count mismatch` 已解决（补齐
    `__identity_*`），但随后报 `non-finite FP16 input/constant is rejected before
    upload` 与二轮的 `missing binding`，疑点在我构造的 `rk_binding`（dtype/rank/shape
    字段与运行时期望不一致）。该用例**故意不命名为 test_***，默认不被收集，避免把
    WIP 计入通过数；运行方式 `RKNPU_FFN_EXEC_WIP=1 pytest tests/wip_ffn_execution.py`。
  · **本轮进展**：harness 已能真正跑通 `rk_plan_create → rk_workspace_create →
    rk_workspace_run`（mock 库），并修复了：硬编码输出 ID（改为从 `plan.bindings`
    取 role=output 的 id）、NumPy backing array 生命周期（`_KEEP_ALIVE` 持有）、
    参考实现的权重转置（RKIR 权重是 `[N,K]`，参考须 `wg.T`）、`workspace_destroy`
    仅在成功时清空 handle（BUSY/POISONED 保留所有权）、mock 控制函数逐个声明
    `argtypes/restype`（含 `size_t`）、`create_workspace` 接受调用者 `fd`。
  · **当前状态：3 项失败（数值不匹配，511/512 元素）**，1 项通过（失败不发布 + 资源回收）。
    该测试**已纳入默认收集**，不再用 skip 隐藏；`results/20261022T085839Z-host` 记录了
    `overall_exit=1`、`pytest 127 passed / 3 failed`。
  · **根因已确认并修复（本轮）**：mock 用 `p->flags` 的高位做
    模板标签，但 `co_schedule.c:38` 要求提交的 program `flags == 0`，且
    `custom_ops.c:344`（scale-native）把 `flags` 设为 `ROCKET_JOB_PPU_DONE`。
    **flags 是生产语义字段，不是自由标签**。修复：mock 改为**私有旁表**（按
    `co_program *` 地址记录模板与 `mul`），不再改写 `p->flags`。修复后 FFN 差异从
    511/512（量级错误）降到 204/512（仅 FP16 舍入顺序），再按"先 `sigmoid(gate)` →
    收窄 → 乘 `gate` → 收窄 SiLU"修正参考的 7 个收窄点后**逐元素严格相等**。
  · **已交付的 FFN 执行证据**（`tests/test_ffn_execution.py`，默认收集，4 项）：
    chain 与 separate 两种 Submit 模式、非方形权重 (m=6,d=96,h=48)、同一 workspace
    两轮不同输入、失败不发布（NaN 注入 → 抛错且输出不变）、BO 与 resident 回到基线。
    参考实现独立用 numpy，7 个 FP16 收窄点显式写出，**未使用任何容差**（逐元素相等）。
  · **结论（收窄）**：planner→FFI→runtime 的**单 tile FFN** 数值证据**已通过**；
    canonical 与 gate/up 交换图默认均走通用 per-node 路径并有 mock 数值对照；
    多输出生命周期已由 `tests/test_multi_output_execution.py` 覆盖。跨 tile
    仍未做，SwiGLU 专用能力项保持 experimental，P7 不关闭。
- **EW identity 内容校验（A 组补强，已完成）**：plan 级 EW mock 不再忽略 identity 内容，
  而是真实执行 `x @ identity` 再过逐元素运算；新增回归——把 identity 末列乘 2 后
  结果对应列必须改变、其余列必须不变。`rk_plan_host` 由 102 → **138 checks**。
- **B 组（真实 TileLang FFN 提取）**：
  · `tests/fixtures/tilelang_ffn.py` 提供 3 GEMM、SiLU、mul、残差的单 tile FFN，
    `X[m,d] · Wg[d,h] · Wu[d,h] · Wd[h,d] → Y[m,d]`；
  · **正例**：`swiglu_ffn_f16` 显式写 gate/up/SiLU/act_mul/projection/residual 六个
    FP16 写回点，前端提取为 6 节点 SSA 图，经通用 per-node 路径执行并与独立 numpy
    参考逐元素相等；
  · **负例**：`swiglu_ffn_fp32_residual` 与 all-FP32 `swiglu_ffn` 在 final T.copy 才
    收窄 residual，即使打开 LUT、experimental、chain/separate 也返回
    `RK_E_NUMERIC_CONTRACT`，不会被静默改写成“先收窄 projection 再 add”；
  · canonical 与 gate/up 交换的 RKIR 分支图均默认走通用 per-node 路径；整图
    `swiglu_ffn_v1` 仅作 opt-in 专用路径。
- **仍未完成（P7 不能关闭）**：
  · 多输出逐节点路径与真实 TileLang 双输出 fixture 均已接入并有 mock 数值证据；
  · FP32 projection+residual 融合路径未定义/未验证；若未来支持须新增独立数值契约与
    硬件证据，本次不放开；
  · 小形状 RK3588 三单核串行准入已通过；设备失败/恢复规范与完整 release gate 仍未
    关闭。多板、三核并发、性能和长时压力属于扩展验证，不是当前 MVP 阻断项。

复核补正（上一轮）：

| 项 | 处理 |
|---|---|
| SwiGLU 注册为 `enabled` 但无执行证据 | 能力项保持 **experimental**；2026-10-23 起 `allow_experimental=True` 只选择整图专用 lowering，canonical 图默认走通用 per-node 路径。`dag.py` 的 `TensorLayout` 已带 `atom`，权重按 `[N,K]` 直拷（`transpose_source=False`），非方形权重可编译 |
| ABI 对照未做字段偏移 | C 契约测试用 `offsetof`/`sizeof` 输出 60 条 `ABI_FIELD <struct> <field> <offset> <size>`；`test_ffi.test_ctypes_offsets_match_the_c_header` 逐字段与 ctypes 比对（同尺寸错位也会被抓到） |
| 验证脚本默认入口缺环境变量 | `run-verification.sh` 现在 **export `TILELANG_DEPS_DIR` 与 `TILELANG_LOCK_FILE`**，默认依赖树改为 P0 的 `.pydeps-p0`；`check_isolation.py` 在变量缺失时给出明确错误而非 `KeyError`。已用**不设置任何环境变量**的默认入口复跑通过 |

本轮新增（语义证据补齐 + SwiGLU）：

| 项 | 内容 |
|---|---|
| 谓词（load/store） | `tl.region` 的 BufferLoad 与 BufferStore 现在都要求 predicate 缺省或可证明恒真；`clear_accum=True` 时 TileLang 会把累加器降级为 store，因此**另有**一个 `gemm_accum_store_only` 反例单独验证访问掩码分支（不经过 clear_accum） |
| grid/K 几何 | 新增 `grid_min_one`、`grid_step_two`、`k_loop_min_one`、`k_loop_step_two` 四个真实 fixture（由 TIR surgery 改造 block/K 循环本身，不再是激活循环） |
| SwiGLU FFN | `legalize._match_swiglu` 按 Value ID 匹配 `Y=(SiLU(X@Wg^T)*(X@Wu^T))@Wd^T+X`；`dag.build_swiglu` 逐阶段 lowering（gate/up/SiLU/mul/projection/residual），每阶段独立 Submit、中间张量不出设备；新增能力项 `co.swiglu_ffn.f16.v1` 与配方 `swiglu_ffn_v1`；单 tile 范围（M/h/d ≤256、各 matmul 的 K ≤256） |

**SwiGLU 证据边界（2026-10-23 更新，能力项仍为 experimental）**：已有 plan 结构断言
（7 task / 6 submit / 中间张量不落盘）、chain/separate 两种执行、两轮输入、失败不发布、
canonical 与 gate/up 交换图的 mock 数值对照，以及 strict/未开 LUT/h>256/分支连错负例。
真实 TileLang 三分支 FFN 的显式 FP16 正例已贯通；FP32 residual 变体作为数值契约负例
拒绝。整图 `swiglu_ffn_v1` 只作为 opt-in 专用路径。

P1 语义收尾：补齐复核点名的语义检查

| 检查 | 规则 | 反例/正例 |
|---|---|---|
| `clear_accum` | 两条路径都读 gemm 第 9 个参数；为真即 `RK_E_NUMERIC_CONTRACT`（累加器只能被一次性零填充，不能在 GEMM 内清零） | `clear_accum_matmul`（单 tile）、`bad_clear_accum_in_loop`（K 循环） |
| load/store 访问掩码 | `tl.region` 的 mask 是访问语义：copy 源必须是 load(1)、目标必须是 store(2)；GEMM 的 A/B 是 load、累加器是 load+store(3)；fill 是 store。不符即拒绝（`clear_accum=True` 时 TileLang 会把累加器降级为 store，正是由此先被拦下） | 两路径共用 `_region(..., expect_mask=)`；负例由 `clear_accum` 触发该分支 |
| grid/K 循环 min 与 step | block 网格与 K 循环都必须 `min=0`、`step=1` | 解析期拒绝；`shifted_*` 负例覆盖 min |
| RKIR stride | 只接受 dense row-major `(cols, 1)` 或未设置；其它 stride 明确拒绝 | `test_strided_tensor_is_rejected` |
| submit 配方 | `recipe_id` 必须与真实任务序列匹配（`linear_k_acc_v1` = mm + mm_acc*；`gate_ew_v1` = LUT 分块 (+EW) 或单个 EW）；未知配方拒绝 | `test_recipe_must_match_the_task_sequence`、`test_unknown_recipe_is_rejected` |
| M-only 网格 | 1 维网格按输出索引推断它覆盖 M 还是 N；无法判定即拒绝 | `mn_tiled_m_only` 正例、`bad_single_grid_axis_ambiguous` 负例 |

显式 guarded 矩形尾块的 ragged K/M/N 已支持（2026-10-23）；仍显式拒绝：3 维网格、
无 K 循环的宽网格、无边界证明的全 tile copy。

P1 批次二（M/N 多 tile 提取）：

| 项 | 内容 |
|---|---|
| 提取 | `_scan` 记录 block 循环而不再直接拒绝；新增 `_affine_expr` 把索引解析为 `Σ coeff·role + const`，要求 A 行 = `by*BM`、A 列 = `ko*256`、B 行 = `ko*256`、B 列 = `bx*BN`、输出 = `C[by*BM, bx*BN]`，其余为常量 0 |
| 覆盖证明 | 要求 `grid_x*BN == N` 且 `grid_y*BM == M`（grid×tile 恰好铺满输出一次）；M/N 重排因此在数值上中性（每输出元素仍来自唯一一次点积，只有 K 顺序影响舍入，而 K 固定 256） |
| 留痕 | matmul op 记录 `source_mn_tiling = {block_m, block_n, grid_x, grid_y, coverage: exact_once}`，plan 首个 MM 任务与编译报告同步输出 |
| 正例 | 2 维网格 M=512/N=384/K=512（BM=256/BN=128）→ tasks 8、publish 4；1 维网格 M=256/N=512/BN=256 → tasks 4 |
| 相邻负例 | ragged 输出（grid×tile 越界）、A 用错 block 变量、输出 tile 用交换的索引、3 维网格 |

P1 批次三（显式 ragged 矩形尾块，2026-10-23）：

| 项 | 内容 |
|---|---|
| 源契约 | 仅接受真实 TIR 的规范化写法：每轮 K 先零填充 A/B 物理 tile，随后用 `T.Parallel` + 单一矩形 guard 条件加载有效元素；输出 store 也由 M/N 矩形 guard 保护。`ceildiv` 本身不作为内存安全证明 |
| 边界证明 | guard 必须是 `block*BT + local < logical_extent` 的纯仿射合取；A/B 读、输出写和所有中间 tile 都检查；A/B tile 必须每轮重新零定义，accumulator 仅在 K 循环前清零一次 |
| RKIR 留痕 | 逻辑 M/N/K 全量保留；matmul attrs 记录 `source_tail_contract`（tail 大小、zero-fill 语义、guarded reads/write、grid/trips）；planner 继续用自身 256 tiling 生成 M/N/K 尾块 |
| 数值正例 | M=13/N=33/K=257（BM=16/BN=64）与 M=N=257/K=300（BM=BN=128）；mock 逐元素匹配“每 256 K 段 FP16 写回 + 尾块零填充 + 多 tile 拼装” |
| 相邻负例 | 缺少 A/B 每轮零填充、边界偏一、错误轴 guard、未保护输出写 |
| 仍拒绝 | 任意条件控制流、if/else、非矩形 mask、没有零填充证明的全 tile copy、3-D grid |

P1 批次一（K 循环提取）与顺带发现的 C 校验缺陷：

| 项 | 内容 |
|---|---|
| K 循环提取 | 新增 `_detect_k_block`：唯一串行/Pipelined 循环、零起点、静态 trips、无嵌套；**源 K 分块固定 256**；两次 copy 必须早于 GEMM；循环变量只能以 `ko*256` 出现在 A(dim1)/B(dim0) 的 slice 源索引；累加器零填充在循环外且在前、输出 copy 在循环外且在后；`trips*256` 必须精确等于参数 K；循环内外任一多余 tileop 均拒绝；GEMM 操作数 buffer 需普通行主序 |
| 源信息留痕 | 首个 MM 任务的 attrs 记录 `source_k_block/source_k_trips/source_loop_annotations/source_loop_span/lowering_k_tile/pipeline_hint_dropped`，编译报告同步输出 |
| 相邻负例（8 条） | ragged K、循环变量走错轴、循环内 clear、循环内两个 GEMM、循环内激活、parallel 循环、tile→tile copy、循环内写输出 |
| **顺带修复的 C 缺陷** | 旧校验器要求"每个 buffer 只有一个写者/写入不得重叠"，导致 **≥3 个 K tile 的计划全部被误拒**（K=513/768/1024 实测 REJECT）。改为：host/device 写者不得混用、重复设备写必须同寻址、每次读取的**最近生产者**必须是该消费者声明的依赖（沿 deps 传递可达）；planner 相应补全依赖声明（首个 MM 任务依赖本 tile 全部 host 动作、EW 依赖 identity 上传、download 依赖产出 task） |
| 缺陷回归 | `tests/test_ffi.py::test_long_k_chain_is_accepted_by_the_c_verifier` 覆盖 K=256/257/512/513/768/1024/4096 |

另有两条新的 C 负例：生产者不是消费者声明的依赖、同一 buffer 被 host 与 device 混写。

顺带修复的缺陷：

- LUT 分段视图此前把**元素偏移**当**字节偏移**使用，第二个 32768 元素段与第一段重叠。
  新写区间检查直接暴露该问题，已改为 `offset*2`，并由 `test_lut_chunk_views_use_byte_offsets`
  锁定（offsets 必须为 `[0, 65536]`）。
- identity 常量上传动作此前被插在 Submit 组**中间**，违反"组内无 Host 动作"，已提前到
  组开始前。
- plan schema 因 publish 结构变化升到 **v2**（`schema/rkplan.v2.json`），旧 v1 保留为历史
  文件；缺少新字段的旧 plan 会被严格拒绝而不是静默兼容。

## 1.5 P0 交付（源码与环境锁）

详见 [P0-SOURCE-LOCK.md](P0-SOURCE-LOCK.md)。摘要：

| 交付项 | 状态 | 证据 |
|---|---|---|
| 源码锁（commit + 子模块） | 完成 | `provenance/source-lock.json`、`provenance/upstream-trees/*.json` |
| 最小补丁 | 无补丁（显式记录） | `source-lock.json: patches: []`、`patch_policy` |
| 干净安装 | 完成 | `scripts/lock_deps.sh install` → `.pydeps-p0`（28 制品，无重复 metadata） |
| 依赖锁（版本+制品哈希） | 完成 | `requirements.lock`（28 × SHA-256），`provenance.json: dependencies` |
| 隔离依赖环境重验 | 完成 | `.venv-p0`（不继承系统 site-packages）+ `isolation` 步骤断言；`results/20261020T112915Z-host/` |
| 完整 Python 文件集合校验 | 完成 | `provenance/wheel-vs-source.json`：1135/1135 一致，0 缺失，0 无法解释 |
| 从源码构建及产物重验 | **阻塞待闭合** | `P0-SOURCE-LOCK.md` 第 5 节（仅记录已尝试路径失败） |

## 2. 工作包状态

| 工作包 | 状态 | 实际完成 | 明确未完成 |
|---|---|---|---|
| P0 基线与依赖锁 | `in_progress` | 源码 commit + 子模块锁定完成；`requirements.lock` 28 制品 `--require-hashes`；`.venv-p0` 隔离解释器下重验通过（isolation 步骤断言第三方模块全部来自 `.pydeps-p0`）；6 个 scope 内 3644 文件（含完整 scope 的 1135 个 Python 文件）与锁定源码逐字节一致，0 缺失、0 无法解释的额外文件。详见 [P0-SOURCE-LOCK.md](P0-SOURCE-LOCK.md) | **源码构建及产物重验 blocked**：本机 `git clone` TLS 中断、`codeload` 超时、`raw.githubusercontent` 超时、PyPI 无 0.1.15 sdist（只记录已尝试路径） |
| P1 真实前端提取 | `in_progress` | 批次零/一/二/三已交付；2-GEMM projection、3-GEMM staged-FP16 SwiGLU、双输出 branch、标量/每通道常量广播、显式 guarded ragged K/M/N 与真实 K-loop/MN-tile fixture 已验；多 tile 逐节点 SSA 可由现有 RKIR builder/持久化 RKIR 接入 | 多 block TileLang TIR 到多 tile 逐节点 SSA 的提取仍未实现/未声明；3 维网格、任意条件控制流、FP32 residual + final-copy narrowing 显式拒绝 |
| P2 IR/Plan/schema/能力表 | `in_progress` | frozen dataclass、严格 JSON、canonical hash、rkir v1 + rkplan v2 + capability v1、enabled/experimental manifest | state/动态 shape/任意 broadcast 语义未实现 |
| P3 MatMul planner | `in_progress` | 单 GEMM M/N 多 tile、K 链/ping-pong/尾块；逐节点 RKIR 值 tile 集合（256x256 输出、256-wide K slice）已支持 projection/FFN/branch/activation 并真机通过；设备驻留中间值、残差/输出保活 | 全部 1..4096 形状组合、跨 tile completion 语义和真实设备失败类别未穷举 |
| P4 组合/存储/提交 | `in_progress` | SiLU/GELU/sigmoid/HardSigmoid；同 shape Add/Mul；编译期常量 rank-0 标量 / rank-1 `[N]` 每通道广播（显式 Host 展开 + 单次 constant upload）；LUT 分段；first-fit slot | 运行时 Tensor 广播、动态 Host 标量、任意 NumPy 广播、原生 scalar EW 模板未实现 |
| P5 C descriptor 校验/FFI | `in_progress` | 深拷贝；上传边界、单写者、发射顺序、slot 复用、publish 覆盖、ABI 版本、预算均独立校验；ctypes 尺寸对照；ASan/UBSan 通过 | 未覆盖全部旧 custom-ops 回归的 sanitizer |
| P6 资源实例化/执行 | `in_progress` | 线性 action 调度；多 tile 输出拼装；有限值闸门；发布前全数据 BO `co_guard_check`；常量仅创建时上传；C runtime 目标 chip/driver/ABI 自检；RK3588 真实驱动成功路径、三核、chain/separate 已验 | 真实设备失败类型/恢复策略仍需按规范补齐；不通过危险 timeout 注入来制造证据 |
| P7 CLI/Python run/示例 | `in_progress` | CLI 五条命令；`Workspace` caller-fd API；真实 TileLang matmul / matmul_silu / silu_proj / staged-FP16 FFN / 双输出 branch / ragged；**板端 `compile-rkir`→`validate`→`explain`→`run` 已在三 mask 通过**（成功发布、fd 平衡、bad-binding rc2 不发布）；`run --fd` 区分 caller-owned/自开 fd | FP32 residual 融合契约未定义；完整 release 文档仍随 P8 开放 |
| P8 板端准入 | `in_progress` | **代表性小形状、三个单核串行准入通过**：RK3588 OPi 5 Plus / driver v0.9.8 / 1 GHz/ABI 1，**27 cases × 3 masks = 81/81 PASS**；新增真实多 block TileLang staged projection/FFN/branch、tiled RKIR 图、广播、双输出、ragged 尾块、K 奇偶分片与完整 CLI；raw outputs、双参考、guard、target/profile/ABI、源包哈希重算均归档。证据 `results/20261023T121507Z-rk3588/` | **剩余仅两项具体条件**：P0 源码构建外部 blocked；真实设备 submit/恢复未测（Host/mock 已验 pre-submit 安全、BUSY、unknown→poison/保留）。有限代表形状矩阵已完成，不要求穷举 1..4096；多板/并发/长压不是前置。 |
| P9 扩展 | `not_started` | 无 | 动态 Host 广播、状态、hybrid、逐组选核、参考库模板 |

## 3. 实际测试结果

Host C（`research/custom-ops/build-host-mvp`）：

```text
26/26 tests passed            (rk_plan_host 默认带 MALLOC_PERTURB_=128)
rk_plan_contract: 26 checks, 0 failures   (ABI 尺寸: publish=32, plan=168)
rk_plan_host:     711 checks, 0 failures
```

Python（`.pydeps` 中 pinned TileLang wheel）：

```text
207 passed, 0 failed, 0 skipped, 38 subtests
                          (含 ABI 全量偏移、语义守卫、K 循环、M/N、SwiGLU、
                           显式 ragged、常量广播、多输出、campaign host preflight)
```

RK3588 real device（`results/20261023T121507Z-rk3588/`，小形状/三单核串行）：

```text
board: RK3588 OPi 5 Plus; kernel 6.1.43-rockchip-rk3588-rknpu098
driver: RKNPU driver: v0.9.8; npu_freq: 1000000000 Hz; runtime ABI: 1
core masks: 1 / 2 / 4 -> 27/27 cases each (81/81 PASS)
MatMul / ragged declared path: bit-exact
projection declared path: max_abs 4.8828125e-4; FP64 math max_abs 3.553947812e-4
FFN declared path:       max_abs 2.44140625e-4; FP64 math max_abs 2.377687311e-4
matmul FP64 math max_abs: 4.817205481e-4; ragged FP64 math max_abs: 4.865511364e-4
tiled projection/FFN/branch/activation declared max_abs:
  9.765625e-4 / 4.8828125e-4 / 2.44140625e-4 / 9.765625e-4
real multi-block TileLang staged projection/FFN/branch: all declared-pass
  (512x256x512; 22/30/6 tasks; separate T.Kernel stage boundaries)
broadcast scalar/channel Add/Mul, independent activation, 256x256 LUT split,
  dual output, M/N multi-tile and K=512/257 shard cases: all declared-pass
CLI on all masks: compile/validate/explain rc0; 2 successful run rounds with
  stable fd count and published Y; missing constant rc2 with no publication
CLI cleanup: destroy/close failures are reported, never masked as success;
  caller-owned fd is never closed by CLI
acceptance: np.allclose(abs(got-declared) <= 8e-3 + 8e-3*abs(declared))
FP64 math reference: float64, unquantized, stored/read unchanged, context only
chain/separate raw outputs: bit-equal for projection/FFN/ragged
raw output NPZ: per core/round under board/core-*/outputs/
source bundle + sha256: archived; source hash independently recomputed=True
loaded library sha256: remote record archived; not independently rehashed
runtime target check: exact NUL token rockchip,rk3588 + exact driver string +
  runtime ABI; rc=0; host negatives include rk3588-extra and v0.9.80
  (minimal target identity gate; not a full fleet/memory contract)
BO guard: co_guard_check at final publication gate; post-download later-task
          corruption test -> RK_E_DEVICE, both outputs unchanged, poisoned
numeric guards: NaN input and non-finite output keep output unchanged;
                NaN constant fails at workspace create before output binding
* hostname/full compatible string omitted; exact credential scan clean
```

ASan/UBSan（`research/custom-ops/build-sanitize-rk`，detect_leaks=1）：

```text
rk_plan_contract: 26 checks, 0 failures
rk_plan_host:     711 checks, 0 failures
sanitize FFN worker: 8 passed
poison worker: handle/live BOs retained, no sanitizer report
MALLOC_PERTURB_=128 与 detect_leaks=1 下同样通过
```

完整可重建记录（命令、退出码、日志、环境/依赖身份、源码哈希、skip 数、
真实编译与链接参数、sanitizer 显式启用证据）：

```bash
research/tilelang-rknpu/scripts/run-verification.sh          # 全新 build 目录
research/tilelang-rknpu/scripts/run-verification.sh --keep-build
# 产物: research/tilelang-rknpu/results/<run-id>-host/
# 通过记录: results/20261020T085458Z-host  (overall_exit 0, evidence_complete true)
# 负对照:   results/20261020T085356Z-host  (注入 cmake 失败, overall_exit 1)
```

关键断言：

- K=256/257/512/513/4096 的 MM 类 task 数分别为 1/2/2/3/16。
- 第 9 节 `X[13,257] @ B[257,33] -> SiLU`：4 task、chain 2 Submit、separate 4 Submit、
  8 数据 BO、payload `60416 B`、数据请求 `323072 B`。
- 多 tile：257×33 → 2 个 publish 矩形 `(0,0,256,33)`/`(256,0,1,33)`；257×257 → 4 个；
  C 侧跑通两个 tile 的真实拼装并逐字节比对参考。
- 复现的两条越权 descriptor 变体（越界 stride、交换 K task）都在 `rk_plan_create` 被拒绝，
  且失败时 `*out == NULL`。
- 多 tile 计划若 publish 重叠或覆盖不全、下载与 publish 不一一对应、slot 复用区间重叠，
  均被 C 或 Python 拒绝。
- 真实 TileLang：方阵 GEMM 接受；非零填充、额外 copy、交换激活下标、`z*exp(-z)`、
  `transpose_B=True`、双 GEMM、GEMM 外包串行循环、恒假 predicate、非零循环起点全部按
  预期拒绝并给出具体诊断。
- 常量所有权：任何由常量上传填充的 buffer 都不能与按调用存活的数据共享 slot。
- 覆盖性：生产者行的物理区域必须覆盖消费者；上传 12 行供 13 行 MatMul 读取被拒。
- 未初始化内存：`MALLOC_PERTURB_=128` 下全部 C 测试通过（修复前 10 项失败）。

复现入口：

```bash
research/tilelang-rknpu/scripts/run-host-tests.sh
```

## 4. 改动文件（本轮）

```text
research/custom-ops/include/rk_plan.h                    # publish tiling, ABI 字段, schema v2
research/custom-ops/src/rk_plan_internal.h               # task→submit 映射, runtime ABI
research/custom-ops/src/rk_plan_validate.c               # 上传边界/发射顺序/slot/publish 校验
research/custom-ops/src/rk_plan_runtime.c                # 线性调度、多 tile 拼装、有限值闸门、字节统计
research/custom-ops/tests/c/rk_plan_contract_test.c      # schema/尺寸断言
research/custom-ops/tests/c/rk_plan_host_test.c          # 复核负例 + 多 tile 执行用例
research/tilelang-rknpu/python/rknpu_backend/plan.py     # Publish 放置矩形 + 覆盖校验, schema v2
research/tilelang-rknpu/python/rknpu_backend/passes/dag.py     # 多 tile publish、LUT 字节偏移、binary lowering
research/tilelang-rknpu/python/rknpu_backend/passes/legalize.py# matmul(+act)(+binary) 路径
research/tilelang-rknpu/python/rknpu_backend/ir.py       # matmul_module 支持同形状 residual
research/tilelang-rknpu/python/rknpu_backend/runtime/ffi.py     # 新 publish 字段与 ABI 版本
research/tilelang-rknpu/python/rknpu_backend/frontend/tilelang_adapter.py  # 身份/顺序/数据流提取
research/tilelang-rknpu/schema/rkplan.v2.json            # 新 plan schema（v1 保留）
research/tilelang-rknpu/tests/fixtures/tilelang_negative.py    # 真实 TileLang 负例
research/tilelang-rknpu/tests/test_{frontend,lowering,plan_verify,storage,ffi}.py
research/tilelang-rknpu/provenance.json
```

## 5. 合规声明

- 未修改既有 `co_*` API 语义；未新增寄存器生成器或提交器。
- 未把 mock 结果写成硬件通过；未声称获得参考库性能。
- 未放宽任何数值/别名/预算规则；新增规则只会更严格。

## 6. 未完成项与下一步

1. **P0 收尾**（唯一剩余项）：在可直连 GitHub 的环境按 `P0-SOURCE-LOCK.md` 第 4 节从
   锁定 commit 构建 `libtvm`，把产物哈希补进 `provenance.json`，再以
   `TILELANG_DEPS_DIR` 指向源码构建产物重跑完整验证并与
   `results/20261020T093654Z-host` 逐项比对。
2. **P1 收尾**：SwiGLU 组合数值执行、多输出生存期、显式 guarded ragged K/M/N
   均已交付。剩余为 3-D grid 单独评估，以及无边界证明的任意控制流保持拒绝。
3. **P4**：编译期 rank-0/rank-1 常量广播已交付（显式 Host 展开、单次 constant
   upload、Host/device 预算与传输计数）。运行时动态/任意广播仍属 P9。
4. **P6/P8**：RK3588 小形状三单核串行、chain/separate、双参考/原始输出、BO guard、
   数值与三类 guard、runtime 目标自检已通过；剩余按规范关闭设备失败/恢复语义与完整
   release gate。不通过危险 timeout 注入制造证据；多板/并发/长压不算 P8 前置。
5. ASan/UBSan 全量回归（旧 custom-ops 目标）。
6. **RK1228**：按独立 target/profile、驱动/uAPI、memory-transfer 契约接入；具体型号
   与接口以实机识别为准，见 [RK1228-TARGET.md](RK1228-TARGET.md)。不得继承 RK3588
   寄存器配方、OBJ/BO 假设或验收结论。

P9（不在本轮阻断范围）：运行时动态 Host 广播、逐组选核调度、状态/递推、hybrid、参考库模板。

扩展验证（**不是 MVP/P8 阻断项**）：多板覆盖、三核并发吞吐、性能优势、长时间压力。
不在本轮通过危险 timeout 配置制造失败证据。

任何一项完成都必须新增独立结果目录与数值报告，不能只更新本文状态文字。

### 新增 mock 具名场景（本轮，已修复并转绿）

**发现的缺陷**：seam 在解析模板标签**之前**就取 `co_view` 并解引用 `buffer->bo.ptr`，
零 buffer 的 program 直接 SIGSEGV（exit 139，首个断言前崩溃）。

**修复**（`tests/mock/npu_mock.c`，`__wrap_co_execute_profiled`）——验证顺序显式化：
1. 调用本身：`p` 非空、`0 < count <= 64`、`mode ∈ {CO_CHAIN, CO_SEPARATE}` → `ROCKET_E_SHAPE`；
2. 每个 program：**先查标签**，未知 → `rk_mock_untracked++` + `ROCKET_E_SHAPE`，**不解引用**；
3. 再校验模板所需 view（`output`、`reads[0]`；EW 另需 `reads[1]`/`reads[2]`），要求 `buffer` 与 `bo.ptr` 均有效；
4. 全部通过后才进行任何数据读写。

**新增两个具名用例**（`rk_plan_host_test`）：

| 场景 | 断言 | 结果 |
|---|---|---|
| 零 buffer program | `ROCKET_E_SHAPE`、`untracked==1`、BO/resident 不变 | **通过** |
| 非法调用（NULL / count==0 / mode=99） | 均 `ROCKET_E_SHAPE` | **通过** |
| 结构合法但未追踪（`rk_mock_reset()` 清标签） | `ROCKET_E_SHAPE`、`untracked==1`、**存活 BO 与 resident 不受 reset 影响** | **通过** |
| 标签表容量边界（`rk_mock_force_tag_overflow`） | 容量 >= 64；置溢出后执行被拒 `ROCKET_E_SHAPE`；清除后 `overflow==0`、`untracked==0` | **通过**（仅守卫侧；真实填满 256 条的路径未跑） |

`rk_plan_host` 690 → **711 checks**；普通构建与 ASan/UBSan（`detect_leaks=1`）均 0 失败。
最小复现文档已更新为「已修复」并指向上述两处测试。

### Mock API 统一（本轮）

- 新增 ****：集中声明全部 （18 个），由  与  共同包含；C 测试里散落的手写声明已删除。该头文件**仅测试使用**，、 均不包含它，生产库不链接 mock。
- 新增 ****（目标 ）：把每个函数赋给**确切类型**的函数指针，签名漂移即编译失败；已覆盖  返回（）与指针参数（、）。构建通过、运行返回 0。
- Python 侧保留**显式 ctypes 签名表** ，并新增  只读视图；**不运行时解析 C 头文件**。

**仍未做**：合法未追踪用例改为"真实 BO/view + 生产 emitter + 清标签"、容量用例"真实登记到上限再登记第一个超限"，以及同址换模板 / 复制用例。

**C/Python 签名对照（本轮转绿）**： 打印 18 条 ； 把 C 的**类型名**映射为 ctypes 类型后**比较类型对象本身**（指针递归比较所指类型，保留数量/顺序/返回类型），因此本平台 、 自然通过，真错位仍失败。补齐了 // 三个导出，集合完全一致。新增负例证明比较器会拒绝错符号、错位宽、错参数数量。探针退出码改用  引用并固定 。**注释已更正**：打印字符串与函数指针声明是**两份手写信息**，字符串不能独立漂移的说法不成立，二者必须同步维护。

### rk_mock_scenarios 独立目标（本轮新增，已转绿）

- 新增 `tests/c/mock_scenarios.c` + CMake 目标 `rk_mock_scenarios`，加入常规验证步骤
  （`summary.md` 与 manifest 中都有 `mock_scenarios` 行）。
- **只构造与清理，不复制布局/emitter**：BO 用生产 `rocket_bo_alloc32`（由测试链接包装到 mock），
  view 用生产 `co_tensor_matrix` / `co_tensor_weight`，program 由生产 `co_emit_ew` 生成；
  helper 现改用生产 `co_buffer_alloc` -> `co_view_init`，释放走 `co_buffer_free`；payload 取 `32 + tensor.bytes` 以覆盖 32 字节放置偏移；任一步失败立即 `release()` 并退出该场景。
- 每次分配后立即 `release()` 清理；统一 `TEST_FD = 3`；每个场景前后检查 BO/resident 基线。

| 场景 | 断言 | 结果 |
|---|---|---|
| 独立 Mul / Add（`co_emit_ew` mul=0/1） | 全张量逐元素等于 `a*b` 或 `a+b`；`untracked==0`；基线回到 0 | **通过** |
| 清理回归 | 单 BO 分配→释放，`live_bos` 1→0、`resident` 归零 | **通过** |

`rk_mock_scenarios` 当前 **18 checks / 0 failures**。

**仍未做**：合法未追踪（真实 emitter 生成 program 后清标签）、真实容量边界（登记到上限再超限）、
同址换模板、program 复制、bounded 算子、Submit 故障场景、Python sanitizer、`proj16`、
多 GEMM 前端三批、板端 campaign、P0 有限重试。

**容量三阶段回归（本轮新增，已通过）**：登记恰好 capacity 个不同 program（无溢出，已登记者可执行）；登记第 capacity+1 个触发溢出（），随后旧程序与新程序**均被拒绝**、输出字节保持 0x33 未被改写、BO/resident 不变； 后重新发射可正常执行，而 reset 之前的旧 program **不再被视为已登记**（ +1 且被拒绝）。溢出检测发生在 **emitter wrapper（mock_tag）**， 本身仍返回 0——用例按此断言，未混称为"生产 emitter 拒绝"。 15 → **37 checks**。

**标签生命周期（本轮进展）**

- **删除**了"不完整结构键回退"（原先按  匹配）与其
  "runtime 可能复制 program"的错误注释；现有 runtime 无复制需求，测试专用 mock 直接拒绝未登记的副本。
- **执行期身份校验**：地址命中只用于定位槽位；随后必须与发射时保存的 ** 全文快照
   完全一致**才执行，内容被改动或陈旧即视为未追踪并拒绝（ + ）。
- **同址重新发射**按槽位刷新（不增条目、允许 EW→MM）， 后重新发射会新增登记。
- 容量用例的拒绝断言改为**精确错误码** ，并补最终  基线。
   37 → **38 checks**。

**仍未做**：同址 EW→MM 回归用例本身、复制回归用例、合法未追踪、bounded、Submit 故障、Python sanitizer。

**标签生命周期回归（本轮新增，已通过）** —  38 → **60 checks**

| 用例 | 断言 | 结果 |
|---|---|---|
| 同址 EW → MM | 同一  先发射 EW 并执行（输出等于 ），再重新发射 MM； **不增长**，执行输出等于  | **通过** |
| 复制拒绝 |  到不同地址、不重新发射：执行返回 ， +1，输出字节保持 ，BO/resident 不变； 仍可执行 | **通过** |
| 快照完整性 | 修改 （，须保全宽度）后执行被拒；恢复后执行通过 | **通过** |

场景结束调用  释放新增快照，并检查 、。
**限制（不夸大）**：全文快照**不复制 view 指向的 BO 状态**，因此它不证明资源存活或地址域，
那两项仍由 DMA/BO 自检与基线断言单独覆盖。修复过程中发现自己的一个真实缺陷：把
截断为  后恢复，导致"恢复后应通过"失败。

**Submit 故障 / poisoned 契约（本轮新增，已通过）** — `tests/test_ws_poison.py`

按 specific.md 的既定语义实测（**全程经 `rk_workspace_run`**，未直接调 mock executor）：

| 断言 | 实测 |
|---|---|
| 首次未分类提交失败 | `RK_E_DEVICE`，输出字节保持 `0x5A`（未发布） |
| 后续 `run` | `RK_E_POISONED`，且 **executor 计数不增加**（未再次提交） |
| `destroy` | `RK_E_POISONED`，**handle 仍在**，`live_bos` 与 `resident` 与失败后完全一致（资源保留） |
| 正常基线 | 故障前一次 `run` 成功，证明 workspace 本来健康 |

**子进程隔离（本轮完成）**：完整场景移入独立 worker 子进程，所有断言（健康基线、、
输出快照、、提交计数不增、destroy 后 handle 与 BO/resident 保持）在 worker 内执行；
**全部通过后**才输出单条  结构化结果并以 0 退出。父测试检查退出码、
结果字段完整性、，并断言父进程 BO/resident **未因该场景增加**（）。

**边界（不夸大）**：结果记录的是"**资源按契约保留**"与"**子进程退出**"，
**不是**正常销毁，也不证明设备已恢复；mock 子进程退出**不能**证明真实驱动的 DMA 恢复协议。
普通成功路径仍用既有 BO/resident 基线与 C 泄漏检查（）。
另外：已分类安全失败（清理与重建）未测；DMA/BO 自检只覆盖当前分配场景，**不能普遍证明**
每次执行的资源存活与地址域正确。

**Sanitizer 覆盖更正（本轮）**

- 上一轮 `rknpu_plan_mock_asan` 复用了**未插桩**的 `rk_plan_runtime_obj`，日志只证明子进程能加载执行，
  **不能**证明 runtime/validator/custom-ops 接受了 ASan/UBSan。该结论已撤回，旧记录保留不覆盖。
- 本轮改用**现有 sanitizer 构建目录**（`build-sanitize-rk`）构建 mock 库：该目录内
  `rk_plan_runtime.c`、`rk_plan_emit.c`、`rk_plan_validate.c`、`custom_ops.c`、`npu_mock.c`
  均带 `-fsanitize=address,undefined`（逐文件证据见结果目录 `sanitizer-python/compile-flags.txt`），
  产物导出 31 个 `__asan_*` 符号，实际加载库与其 sha256 一并归档。
- poisoned worker 在完整插桩库下重跑：**退出码 0**，单条 `POISON-RESULT`，无 ASan/UBSan 报告
  （日志 `sanitizer-python/poison-worker.log`）。
- **四个 FFN 成功/失败用例**也在同一插桩库下运行：**4 passed**（日志 `sanitizer-python/ffn-worker.log`）。
  poisoned worker 通过**不替代** FFN 路径覆盖，二者分开记录。

**当时仍未完成（历史记录，后续已更新）**：合法未追踪、bounded 算子、`proj16` 收窄、
收窄缺失负例、多 GEMM 前端批次、板端 campaign、P0 有限重试。当前状态以上文
“已完成/当前状态/P8”段落为准。

### 后端主线：MatMul -> SiLU -> 投影 + 残差（本轮起步）

**范围声明**（按指示）：停止扩展通用前端与独立 mock 框架；只做这一个真实程序的闭环，
SwiGLU 保留为后续组合验收。

**锁定接口 / 复用情况 / 真实缺口**

| 已由 TileLang/TVM 提供 | 适配器当前用法 | 真实缺口 |
|---|---|---|
| 、、、、/、 访问区域 | 结构性遍历（），按  判 buffer 身份 | 需要一个"定义-使用"数据流视图（当前按语句顺序 + 显式 region 推断） |
| 、、、 | 已用于 K 循环/块网格/布局检查 | 无 |
|  的 M/N/K、transpose 标志、、access mask | 已逐项校验 | 只支持**单 GEMM**；两 GEMM + 中间值传递是缺口 |
| 、 | 要求恒真 | 无 |

**当时已落地（历史记录，后续已扩展）**：真实 kernel
（M=13, D=64, H=32，非方形权重，**五个显式 FP16 收窄点**：GEMM1 acc→、
→、→、GEMM2 acc→、→），
以及 ：用  断言 2 个  与四个
 中间量（**按 TIR 内容而非变量名**），并钉住当前精确拒绝
（: 参数超过三个）。

**已完成（2026-10-23 闭环）**：两 GEMM 的 buffer-身份数据流提取、四节点 RKIR
（`matmul → silu → matmul → add`）、逐节点 legalize、`value_id → (cube view,
producer)` lowering、FFI mock 执行与独立 numpy 五收窄点参考、两轮输入变化、残差
保活断言，以及分支接错/收窄缺失/未支持算子的负例。证据：
`tests/test_frontend_silu_proj.py`、`tests/test_silu_proj_execution.py`、
`results/20261023T121654Z-host/`：207 pytest / 26 ctest，evidence_complete=true。

**已知待补的基础设施**：验证脚本的依赖短路（）与产物哈希；
 快照在  之后 —— 记录该限制，不另开整修阶段。

### 阶段性闭环方案（已定，实施中）

- **能力边界**：先支持「无 K 循环 + 每节点一个 primitive tile」的静态数据流；超范围输入**明确拒绝**，
  并标注「阶段性支持，非 MVP 完成」。
- **角色推导**：保留 `_param_buffers` 的静态类型/布局校验，**移除固定 arity**；角色按实际读写
  （GEMM 的 A/B region、被写的参数、copy/GEMM 数据流）推导，不按参数位置、也不按"是否参与最后 add"。
- **节点范围**：MatMul、sigmoid/SiLU、同形状 Mul/Add —— 逐节点匹配，**不匹配整个 silu_proj 函数**。
- **lowering**：`value_id -> (device view, producer action)` 映射，后续节点直接消费已有中间值；
  中间张量留在设备 BO，**不下载再上传**。权重继续复用 `b_layout="kn"` → 上传 `transpose_source`。
- **不改动**：`_detect_k_block` 与现有单 GEMM 多 tile 路径保持不变（K 循环回归必须继续通过），
  fixture 本身无 K 循环，首个闭环无需泛化它。
- **验收**：真实 PrimFunc → RKIR → RKPlan → FFI mock 数值对照（五个 FP16 收窄点、独立 numpy 参考）、
  两轮输入变化、残差保活、少量相邻负例（分支接错/收窄缺失/未支持算子）。

**当前状态**：该闭环已实现并通过 Host/mock 数值验收。前端已支持单 tile 复合图与
显式 guarded ragged K/M/N 尾块；ragged 路径仍保留逻辑尺寸和补零/有效区域语义，不把
全 tile copy 当作安全证明。
RKIR 级通用图形支持多输出；K-loop/M-N 多 tile 继续走原路径。canonical SwiGLU 默认走
per-node，`allow_experimental=True` 仅选择整图专用路径。FP32 projection+residual 融合
仍是明确的未定义数值契约。详见 [HANDOFF.md](HANDOFF.md)。
