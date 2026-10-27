# TileLang 后端开发与需求历史

整理日期：2026-10-27。

历史材料归档，按原写作时点解释，不作为当前任务清单。各节保留原文件内容和阶段限定；不同日期的状态、测试数量与性能不能合并为当前结论。当前入口见 [模块首页](../README.md)，验收见 [ACCEPTANCE.md](../ACCEPTANCE.md)，最终优化结论见 [OPTIMIZATION-SUMMARY.md](../OPTIMIZATION-SUMMARY.md)。命令仍按原文约定的工作目录执行，不能直接从 archive 目录运行。

## 内容索引

- [MVP 实施与修复记录](#implementation-status)（原 `IMPLEMENTATION-STATUS.md`）。
- [2026-10-20 原始复核](#review-2026-10-20)（原 `REVIEW-2026-10-20.md`）。
- [2026-10-23 复核关闭](#review-closure)（原 `REVIEW-CLOSURE.md`）。
- [2026-10-23 逐节点 lowering 交接](#handoff)（原 `HANDOFF.md`）。
- [源码、依赖锁与源码构建验收](#source-lock)（原 `P0-SOURCE-LOCK.md`）。
- [已完成的两模型里程碑需求](#model-requirements)（原 `NEXT-MILESTONE.md`）。

---

<a id="implementation-status"></a>

## MVP 实施与修复记录

原文件：`IMPLEMENTATION-STATUS.md`。以下为该文件历史内容。

<a id="implementation-status--tilelang--rknpu-实施状态"></a>
### TileLang → RKNPU 实施状态

> 本文记录后端 MVP 阶段的实施与修复历史；完整模型及后续优化分别验收，不在此累计更新。各阶段入口见 [README](../README.md)。文内“本轮”和测试数量均按对应历史时点解释。

更新日期：2026-10-23（source-built TileLang/TVM、多 block staged 图、provider-context 隔离）。
当前结论：**声明支持域的后端实现、代表性 RK3588 三单核串行准入和 P0 源码重建均已通过；
按已声明的 MVP 范围，交付闭环完成。真实未知故障后的恢复/继续计算明确不在范围内，未实现、
未测、不承诺；当前仍禁止作为通用 release 版本**。

- 第一轮：[复核报告](#review-2026-10-20) 阻断项 1–4 的原始三条反例。
- 第二轮：四个同源缺陷（未初始化内存、循环语义、常量 BO 生存期、覆盖性）。默认 `ctest`
  环境已开 `MALLOC_PERTURB_=128`。
- 第三轮：两处新缺陷（ABI 字符串复制越界、predicate/循环起点语义）。修复已静态核对，
  关闭证据本轮补齐：精确错误码、63 字符接受边界、predicate 四类负例、循环起点三类负例、
  可审计运行记录。
- RK3588 真机：使用 source-built TileLang/TVM 重新生成的同一 27-case campaign 在 3 个
  单核串行 mask 上 81/81 PASS，包含真实多 block staged projection/FFN/branch、多 tile
  RKIR、广播、双输出、K 奇偶分片、ragged 尾块和完整 CLI；证据
  `results/20261023T144054Z-rk3588/`。

四项证据语义复核项已关闭并留有回归：FP64 数学参考保持 float64、BO guard 位于最终
发布门禁、verifier 的 pair/字节/源包哈希语义收紧、target 使用完整 NUL 兼容项和精确
驱动字符串。逐项代码/测试/证据见 [REVIEW-CLOSURE.md](#review-closure)；历史限制
（小形状、单 ragged 尾块、remote library hash 仅记录等）不隐藏。

MVP 收口矩阵见 [ACCEPTANCE-MATRIX.md](../ACCEPTANCE.md#mvp)，支持域与明确拒绝项见
[SUPPORT-DOMAIN.md](../SUPPORT-DOMAIN.md)，运行期 BO/失败生命周期见
[RUNTIME-LIFECYCLE.md](../RUNTIME-LIFECYCLE.md)，重放命令见 [REPLAY.md](../REPLAY.md)。
P0 源码构建已完成：锁定 archive、submodule 与工具链清单、构建配置、库哈希、日志和源码
树清单见 `provenance/p0-source-build.json` 与 `provenance/p0-source-build/`；两次短超时
失败记录保留为历史。

<a id="implementation-status--运行契约缺口关闭2026-10-23"></a>
#### 运行契约缺口关闭（2026-10-23）

- `rocket_close` 按 vendor header 的真实 `void` ABI 配置和调用，不再读取伪造返回值。
- poisoned/BUSY destroy 后不调用 `rocket_close`；provider fd/context/task BO 保留，CLI 保留原始执行错误和 cleanup 诊断，caller fd 不关闭。
- `poisoned` 改为 `atomic_int`；run 先取得 atomic busy gate 再读 poison，失败 run 在持有 gate 时置 poison；destroy 也先取得 gate。并发失败 regress 验证 BUSY、无第二次提交、后续 POISONED。
- **provider 共享上下文隔离**：`rknpu-submit` 每 fd 增加 `atomic_int faulted`；`submit_kick()` 在释放 `submit_lock` 前置位；排队提交随后得到 `-EPIPE` 而不改写 task BO；同 fd 既有/新建 workspace 被拒绝；faulted context 的 `rocket_close` 保留 fd/task BO。
- 上游补丁记录：`provenance/upstream-patches.json`、`provenance/patches/rknpu-submit-context-fault.patch`（base `6b144b19...`）。
- 证据：`tests/test_runtime_abi.py`、`tests/test_workspace_busy.py`、`tests/test_cli.py`、`tests/test_ws_poison.py`、C `provider_context`，详见 [RUNTIME-LIFECYCLE.md](../RUNTIME-LIFECYCLE.md)。

<a id="implementation-status--最终-mvp-状态2026-10-23"></a>
#### 最终 MVP 状态（2026-10-23）

- 实现域与有限代表矩阵：完成并真机通过。
- 原定多 tile 逐节点图：RKIR 与真实多 block TileLang staged projection/FFN/branch 均已贯通。
- 运行契约：target/binding/BO/budget/BUSY/pre-submit 安全/unknown→poison 保留、CLI cleanup/fd 均有代码、测试和日志。
- **P0 源码重建：完成**。source-built TileLang/TVM/tvm-ffi/dlpack 的实际加载路径、完整 pytest、CTest、27 个 plan 语义等价和板端 81/81 已记录；`tvm_ffi` 的 `find_libtvm_ffi()` 与 `/proc/self/maps` 均验证为 source-built 库。
- **MVP 范围已明确为“成功路径 + 未知错误的隔离停止”**；真实设备 submit/恢复后继续计算不在范围，未实现、未测、不承诺。该限制不隐式追加为 MVP gate；多板/并发/长压也不列为前置。

<a id="implementation-status--1-本轮修复对照"></a>
#### 1. 本轮修复对照

| 复核阻断项 | 本轮处理 | 证据 |
|---|---|---|
| 1. C 上传源内存边界未闭合 | `validate_upload` 要求 MVP 连续绑定 `source_row_stride == shape[1]`，重算最大源元素索引并检查乘加溢出与 `nbytes`；`upload_action` 在打包循环内再次检查索引 | `rk_plan_host_test` 的 `source_row_stride=0xffffffff` 负例；C 返回非 0 且 `*out==NULL` |
| 2. 执行顺序不受依赖约束 | 引入"发射序列 = action id 顺序"契约：每个 Submit 必须拥有**连续升序**的 task 区间且组内不得夹 Host 动作；新增单写者、写区间不重叠、读不早于最后写、slot 复用区间不相交检查 | `rk_plan_host_test` 的交换 K task 顺序负例、`w1`/输出同 slot 的 WAR 负例 |
| 3. M/N 多 tile 输出未实现 | `rk_publish_desc` 增加 `dst_row_offset/dst_col_offset/rows/cols`；publish 必须无重叠且精确覆盖输出 binding；runtime 按 tile 下载后拼装成完整输出，只有全部 tile 成功才发布 | C 侧 `multitile_publish_case`（M=257 两 tile 真实拼装并数值比对）；Python `MultitileTests`（257×33、13×257、257×257、513×600、4096×1） |
| 4. 前端语义检查不完整 | 提取器改为**顺序 + buffer 身份 + copy 数据流**驱动：参数角色由 copy 边（谁喂 GEMM 的 A/B、谁接收 accumulator）确定，方阵不再歧义；fill/copy/GEMM/激活/C 的顺序被证明；SiLU 要求循环层数、extent、store/load 索引与 loop var 完全一致 | `tests/fixtures/tilelang_negative.py` 十二个真实 TileLang 负例 + 方阵/SiLU 两个正例（均由真实 PrimFunc 对象构造） |
| 5. 完成范围不足 | **历史记录（后续已补）**：同 shape Add/Mul、K 循环、M/N 多 tile、SwiGLU 图 lowering、常量广播和逐节点多 tile 数据流均已补 | 见 P1/P3/P4/P7/P8 行与 [ACCEPTANCE-MATRIX.md](../ACCEPTANCE.md#mvp) |
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

P9 范围校准（按 [specific.md](../specific.md) 第 13 节）：

- **运行时动态 Host 广播**、**逐组选核调度**属于 P9，不作为 P0–P8 完成阻断项。
- P0–P8 内已完成的有限项：编译期常量展开（标量/每通道）、既定单核执行策略，以及
  三个单核串行 mask 的真机验收。运行时动态广播和并发选核仍是 P9，不属于 P0–P8。

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

<a id="implementation-status--p5--abi-全量对照specific-124--p5"></a>
##### P5 · ABI 全量对照（specific §12.4 / P5）

- **实现**：`rk_plan_contract_test` 由 `include/rk_plan.h` 自动生成 `ABI_SIZE`（11 个结构总大小）
  与 `ABI_FIELD`（**138 个字段**的 offset/size）；`tests/test_ffi.py` 逐结构比对
  `ctypes.sizeof`、**字段集合必须完全相同**、每个字段的 `offset` 与 `size`。
  已删除"至少 50 项"的门槛，计数、行列偏移、预算、`runtime_abi_version` 等此前遗漏的字段
  全部覆盖。
- **仍有限制**：只比对当前头文件声明；C 端不校验 Python 侧 `_fields_` 的顺序语义（顺序由
  偏移比较间接保证）。
- **可支持结论**：字段交换、遗漏或结构总大小变化都会被测试发现。

<a id="implementation-status--前端与计划验证收口specific-124--p1p3"></a>
##### 前端与计划验证收口（specific §12.4 / P1、P3）

- **实现**：① Call 白名单——内核体中任何非 `tl.tileop.{fill,copy,gemm}` 的调用一律拒绝
  （`unhandled call`），K 分支也在其之前；② SiLU 内部的 `BufferLoad.predicate` 必须恒真或
  缺省；③ 激活循环 `step` 必须为 1；④ 计划校验先确认 task ID 是存在的 NPU task，再做配方
  匹配（不再可能抛 `KeyError`）。
- **仍有限制**：白名单以锁定的 TileLang 版本为准；非 tileop 的纯算术仅允许出现在已证明的
  SiLU 表达式中。
- **可支持结论**：四类非法输入各自命中目标规则并返回约定诊断。

<a id="implementation-status--c-独立校验specific-124--p5范围已收窄"></a>
##### C 独立校验（specific §12.4 / P5）——范围已收窄

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

<a id="implementation-status--swiglu--ffnspecific-124-复合模式--p7"></a>
##### SwiGLU / FFN（specific §12.4 复合模式 / P7）

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
- **声明 MVP 范围已完成**：
  · 多输出逐节点路径与真实 TileLang 双输出 fixture 均已接入并有 mock 数值证据；
  · FP32 projection+residual 融合路径明确不支持；若未来支持须新增独立数值契约与
    硬件证据，本次不放开；
  · 小形状 RK3588 三单核串行准入已通过；未知错误的支持行为是隔离并停止，真实恢复/
    继续计算明确 out of scope。多板、三核并发、性能和长时压力属于扩展验证，不是
    MVP 阻断项。

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

<a id="implementation-status--15-p0-交付源码与环境锁"></a>
#### 1.5 P0 交付（源码与环境锁）

详见 [P0-SOURCE-LOCK.md](#source-lock)。摘要：

| 交付项 | 状态 | 证据 |
|---|---|---|
| 源码锁（commit + 子模块） | 完成 | `provenance/source-lock.json`、`provenance/upstream-trees/*.json` |
| 最小补丁 | 无补丁（显式记录） | `source-lock.json: patches: []`、`patch_policy` |
| 干净安装 | 完成 | `scripts/lock_deps.sh install` → `.pydeps-p0`（28 制品，无重复 metadata） |
| 依赖锁（版本+制品哈希） | 完成 | `requirements.lock`（28 × SHA-256），`provenance.json: dependencies` |
| 隔离依赖环境重验 | 完成 | `.venv-p0`（不继承系统 site-packages）+ `isolation` 步骤断言；`results/20261020T112915Z-host/` |
| 完整 Python 文件集合校验 | 完成 | `provenance/wheel-vs-source.json`：1135/1135 一致，0 缺失，0 无法解释 |
| 从源码构建及产物重验 | **完成** | `provenance/p0-source-build.json`、`provenance/p0-source-build/`（archive/submodule 哈希、CMakeCache、编译命令、工具链、库哈希、pytest/CTest、源码树清单） |

<a id="implementation-status--16-下一阶段完整模型独立里程碑"></a>
#### 1.6 下一阶段完整模型（独立里程碑）

完整 MNIST MLP 与 Tiny Transformer 已按 [MODEL-MILESTONE-STATUS.md](../ACCEPTANCE.md#models)
完整关闭：真实学习权重、真实 TileLang staged 源、RKIR/RKPlan、生产 runtime；
完整 10,000 张测试集在 core 1 运行，MLP 98.31%、Tiny Transformer 98.21%，CPU/声明路径
top-1 一致率均为 100%。随后补关了计时、hybrid 计划和模型级清理缺口；数值边界最终
改为：32 张无标签 train 样本在板端校准、Host 冻结 33 个逐层和 5 个 profile 门禁，
再运行与 train 逐像素哈希不重叠的 32 张 test 样本。新增真实 TileLang MLP 诊断计划，
导出两层隐藏层的 linear+bias、ReLU 和最终 preactivation；生产计划仍只下载最终
logits。最终任务/计时证据为 `results/20261024T071144Z-models/`，train-calibrated
数值证据为 `results/20261024T091634Z-numeric/`，二者 verifier 均 PASS。此前查看过
test 结果的事实已记录，不宣称盲测；旧 test-envelope 仅保留为回归证据。该阶段独立于
已验收 MVP，不回写历史验收，也不宣称纯 NPU Transformer 或性能增益门槛。

<a id="implementation-status--2-工作包状态"></a>
#### 2. 工作包状态

| 工作包 | 状态 | 实际完成 | 明确未完成 |
|---|---|---|---|
| P0 基线与依赖锁 | `complete` | 源码/依赖锁、隔离环境、源码树 6 scope 一致性均完成；**锁定源码已独立构建** TileLang/TVM/tvm-ffi/dlpack/libbacktrace，source-built full pytest 213/40、CTest 27/27、27 个计划语义一致、同一 RK3588 campaign 81/81；详见 `P0-SOURCE-LOCK.md` 第 9 节与 `provenance/p0-source-build.json` | 无（真实设备故障恢复不属于 P0） |
| P1 真实前端提取 | `complete`（声明范围） | 真实多 block `T.Kernel` staged projection/FFN/branch 已接入逐节点 SSA；每阶段一个 launch 边界、全局中间参数、显式 FP16 K-tile 写回和精确 block 覆盖均验证；单 tile/K-loop/MN-tile/ragged/broadcast 正例保留 | 同 launch 内跨 block 依赖、FP32 跨 tile 累加后一次收窄、block 覆盖不精确均拒绝；3-D grid、任意条件控制流、FP32 residual + final-copy narrowing 显式拒绝 |
| P2 IR/Plan/schema/能力表 | `complete`（声明范围） | frozen dataclass、严格 JSON、canonical hash、rkir v1 + rkplan v2 + capability v1、enabled/experimental manifest | state/动态 shape/任意 broadcast 语义未实现 |
| P3 MatMul planner | `complete`（声明范围） | 单 GEMM M/N 多 tile、K 链/ping-pong/尾块；逐节点 RKIR 值 tile 集合（256x256 输出、256-wide K slice）已支持 projection/FFN/branch/activation 并真机通过；设备驻留中间值、残差/输出保活；source-built TileLang 生成的 27 个计划与 wheel 计划语义逐项一致 | 1..4096 全组合不要求穷举；真实设备故障后的恢复能力 out of scope |
| P4 组合/存储/提交 | `complete`（声明范围） | SiLU/GELU/sigmoid/HardSigmoid；同 shape Add/Mul；编译期常量 rank-0 标量 / rank-1 `[N]` 每通道广播（显式 Host 展开 + 单次 constant upload）；LUT 分段；first-fit slot | 运行时 Tensor 广播、动态 Host 标量、任意 NumPy 广播、原生 scalar EW 模板未实现 |
| P5 C descriptor 校验/FFI | `complete`（声明范围） | 深拷贝；上传边界、单写者、发射顺序、slot 复用、publish 覆盖、ABI 版本、预算均独立校验；ctypes 尺寸对照；ASan/UBSan 通过 | 未覆盖全部旧 custom-ops 回归的 sanitizer |
| P6 资源实例化/执行 | `complete`（声明安全契约） | 线性 action 调度；多 tile 输出拼装；有限值闸门；发布前全数据 BO `co_guard_check`；常量仅创建时上传；C runtime 目标 chip/driver/ABI 自检；workspace poison 原子化；**provider 共享 context 在 submit_lock 内故障隔离，共享 fd 排队/新 workspace/faulted close 均保留 task BO**；RK3588 成功路径、三核、chain/separate 已验 | 真实故障后的恢复/继续计算明确 out of scope；不通过危险 timeout 注入或宣称恢复成功 |
| P7 CLI/Python run/示例 | `complete`（声明范围） | CLI 五条命令；`Workspace` caller-fd API；真实 TileLang matmul / matmul_silu / silu_proj / staged-FP16 FFN / 双输出 branch / ragged；**板端 `compile-rkir`→`validate`→`explain`→`run` 已在三 mask 通过**（成功发布、fd 平衡、bad-binding rc2 不发布）；`run --fd` 区分 caller-owned/自开 fd | FP32 residual 融合等未支持形式明确拒绝；故障后恢复/继续计算 out of scope |
| P8 板端准入 | `complete`（声明 MVP 范围） | **代表性小形状、三个单核串行准入通过**：RK3588 OPi 5 Plus / driver v0.9.8 / 1 GHz/ABI 1，**27 cases × 3 masks = 81/81 PASS**；source-built TileLang/TVM 重新生成计划；真实多 block staged projection/FFN/branch、tiled RKIR、广播、双输出、ragged 尾块、K 奇偶分片、完整 CLI、源包/库哈希和凭据扫描均归档。证据 `results/20261023T144054Z-rk3588/` | 真实设备恢复/继续计算明确 out of scope；隔离停止契约已验证。 |
| P9 扩展 | `not_started`（有限模型 hybrid 已作为独立里程碑交付） | `rkhybrid` + 注册 Host action + RKPlan 已在 MLP/Tiny Transformer 完整验收 | 通用 hybrid/state、动态 Host 广播、逐组选核、参考库模板 |

<a id="implementation-status--3-实际测试结果"></a>
#### 3. 实际测试结果

Host C（`research/custom-ops/build-host-mvp`）：

```text
27/27 tests passed            (rk_plan_host 默认带 MALLOC_PERTURB_=128)
rk_plan_contract: 26 checks, 0 failures   (ABI 尺寸: publish=32, plan=168)
rk_plan_host:     711 checks, 0 failures
```

Python（`.pydeps` 中 pinned TileLang wheel，最终 Host 验证）：

```text
235 passed, 0 failed, 0 skipped, 41 subtests
                          (含 ABI 全量偏移、语义守卫、K 循环、M/N、SwiGLU、
                           显式 ragged、常量广播、多输出、campaign host preflight、
                           rkhybrid 图校验、逐层容差、清理不覆盖原异常，
                           以及 top-1 不变的 logits 放大拒绝负例)
```

RK3588 real device（`results/20261023T144054Z-rk3588/`，小形状/三单核串行）：

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
CLI cleanup: real void rocket_close ABI; destroy failures reported, never masked;
  poisoned/BUSY destroy retains provider fd/task BO (no close); caller fd owned
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

<a id="implementation-status--4-改动文件本轮"></a>
#### 4. 改动文件（本轮）

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

<a id="implementation-status--5-合规声明"></a>
#### 5. 合规声明

- 未修改既有 `co_*` API 语义；未新增寄存器生成器或提交器。
- 未把 mock 结果写成硬件通过；未声称获得参考库性能。
- 未放宽任何数值/别名/预算规则；新增规则只会更严格。

<a id="implementation-status--6-未完成项与下一步"></a>
#### 6. 未完成项与下一步

1. **P0 已完成**：锁定 commit 的源码已独立构建，完整 pytest/CTest、计划语义对照和
   同板 campaign 见 `provenance/p0-source-build.json`。无剩余 P0 项。
2. **P1 收尾**：SwiGLU 组合数值执行、多输出生存期、显式 guarded ragged K/M/N
   均已交付。剩余为 3-D grid 单独评估，以及无边界证明的任意控制流保持拒绝。
3. **P4**：编译期 rank-0/rank-1 常量广播已交付（显式 Host 展开、单次 constant
   upload、Host/device 预算与传输计数）。运行时动态/任意广播仍属 P9。
4. **P6/P8 已完成（声明 MVP 范围）**：RK3588 小形状三单核串行、chain/separate、双参考/
   原始输出、BO guard、数值与三类 guard、runtime 目标自检已通过。未知错误的隔离停止
   契约已验证；真实恢复/继续计算明确 out of scope。不通过危险 timeout 注入制造证据；
   多板/并发/长压不算 P8 前置。
5. ASan/UBSan 全量回归（旧 custom-ops 目标）。
6. **RK1228**：按独立 target/profile、驱动/uAPI、memory-transfer 契约接入；具体型号
   与接口以实机识别为准，见 [RK1228-TARGET.md](../RK1228-TARGET.md)。不得继承 RK3588
   寄存器配方、OBJ/BO 假设或验收结论。

P9（不在本轮阻断范围）：运行时动态 Host 广播、逐组选核调度、状态/递推、hybrid、参考库模板。

扩展验证（**不是 MVP/P8 阻断项**）：多板覆盖、三核并发吞吐、性能优势、长时间压力。
不在本轮通过危险 timeout 配置制造失败证据。

任何一项完成都必须新增独立结果目录与数值报告，不能只更新本文状态文字。

<a id="implementation-status--新增-mock-具名场景本轮已修复并转绿"></a>
##### 新增 mock 具名场景（本轮，已修复并转绿）

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

<a id="implementation-status--mock-api-统一本轮"></a>
##### Mock API 统一（本轮）

- 新增 ****：集中声明全部 （18 个），由  与  共同包含；C 测试里散落的手写声明已删除。该头文件**仅测试使用**，、 均不包含它，生产库不链接 mock。
- 新增 ****（目标 ）：把每个函数赋给**确切类型**的函数指针，签名漂移即编译失败；已覆盖  返回（）与指针参数（、）。构建通过、运行返回 0。
- Python 侧保留**显式 ctypes 签名表** ，并新增  只读视图；**不运行时解析 C 头文件**。

**仍未做**：合法未追踪用例改为"真实 BO/view + 生产 emitter + 清标签"、容量用例"真实登记到上限再登记第一个超限"，以及同址换模板 / 复制用例。

**C/Python 签名对照（本轮转绿）**： 打印 18 条 ； 把 C 的**类型名**映射为 ctypes 类型后**比较类型对象本身**（指针递归比较所指类型，保留数量/顺序/返回类型），因此本平台 、 自然通过，真错位仍失败。补齐了 // 三个导出，集合完全一致。新增负例证明比较器会拒绝错符号、错位宽、错参数数量。探针退出码改用  引用并固定 。**注释已更正**：打印字符串与函数指针声明是**两份手写信息**，字符串不能独立漂移的说法不成立，二者必须同步维护。

<a id="implementation-status--rk_mock_scenarios-独立目标本轮新增已转绿"></a>
##### rk_mock_scenarios 独立目标（本轮新增，已转绿）

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

<a id="implementation-status--后端主线matmul---silu---投影--残差本轮起步"></a>
##### 后端主线：MatMul -> SiLU -> 投影 + 残差（本轮起步）

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
历史 MVP Host 记录为 `results/20261024T032922Z-host/`（219 pytest / 27 ctest）；
完整模型最终 Host 重验为 `results/20261024T092339Z-host/`（235 pytest /
41 subtests、27/27 ctest、sanitizer，evidence_complete=true）。

**已知待补的基础设施**：验证脚本的依赖短路（）与产物哈希；
 快照在  之后 —— 记录该限制，不另开整修阶段。

<a id="implementation-status--阶段性闭环方案已定实施中"></a>
##### 阶段性闭环方案（已定，实施中）

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
仍是明确的未定义数值契约。详见 [HANDOFF.md](#handoff)。

---

<a id="review-2026-10-20"></a>

## 2026-10-20 原始复核

原文件：`REVIEW-2026-10-20.md`。以下为该文件历史内容。

<a id="review-2026-10-20--tilelangrkplan-实施复核"></a>
### TileLang/RKPlan 实施复核

日期：2026-10-20。结论：保留为未完成的实验原型，不按 specific.md 的 MVP 验收，不批准板端执行。此报告是源码审查与 Host 复现，不是硬件测试。

<a id="review-2026-10-20--实际检查"></a>
#### 实际检查

执行 `bash research/tilelang-rknpu/scripts/run-host-tests.sh`，C 26/26、Python 36/36 通过。测试总数包含此前 GDN 等工作区代码，不表示本提交单独新增26项。

随后用 `git checkout-index` 将暂存内容导出到独立临时目录，仅连接现有 upstream checkout 和 `.pydeps` 依赖，运行同一脚本：C **22/22**、Python **36/36** 通过。此副本不含未提交 GDN 代码，确认本次原型提交可独立构建。板端占位脚本通过 `bash -n`，调用返回2且没有执行 SSH/Submit。

另直接通过 ctypes 构造 `X[13,257] @ B[257,33] → SiLU` 的 C descriptor，只调用 `rk_plan_create/destroy`，不分配设备资源。结果：

| 变体 | 实际结果 | 应有结果 |
|---|---|---|
| 原始计划 | 返回0 | 接受 |
| 将首个 upload 的 `source_row_stride` 改为 `0xffffffff` | 返回0 | 拒绝源地址范围越界 |
| 交换首个 Submit 的两个 K task ID | 返回0 | 拒绝先读取未产生的 partial |
| 编译 M=257,K=64,N=33 | `duplicate publish entries are not allowed` | 按规范应支持多 M tile |
| 编译 M=13,K=64,N=257 | 同上 | 按规范应支持多 N tile |

<a id="review-2026-10-20--阻断项"></a>
#### 阻断项

1. **C 上传源内存边界未闭合**：`src/rk_plan_validate.c:validate_upload` 只检查逻辑行列范围和非零 stride；`src/rk_plan_runtime.c:upload_action` 用该 stride 计算真实源索引。应重算最大源元素索引与 `nbytes`，检查乘加溢出；MVP 连续绑定应要求 stride 与 shape 一致。修复前不能信任外部 descriptor。
2. **执行顺序不受依赖约束**：C 校验器检查 deps ID 拓扑与 task 归属，却没有确认实际 Submit 内/组间顺序满足生产消费关系。runtime 按 task_ids 发射，因此上述逆序计划会先读 partial。补上实际执行序列的依赖和初始化校验，并覆盖 slot 复用后的 WAR/WAW；不能只排序非法输入来掩盖错误。
3. **M/N 多 tile 输出未实现**：DAG 每个 tile 向同一 binding 添加 publish，plan 校验禁止重复；runtime 的 publish 也按单 view 下载整输出。需要明确 tile 拼接/有效区域描述和多 tile 输出暂存，再完成第14节边界测试。
4. **前端语义检查不完整**：源码按存在性匹配 fill/copy/GEMM，尚未使用收集的 order 证明先后；SiLU 匹配只比较 buffer 名称，未证明 load/store 索引一致和循环覆盖。还按 shape 猜参数角色，方阵会歧义拒绝。需基于实际 buffer identity、copy 数据流和顺序建立封闭模式，增加改序、错索引、额外调用等负例。这些是源码发现，尚未做完整畸形 PrimFunc 复现。
5. **完成范围不足**：P1 K循环/多tile、P4普通Add/Mul与常量广播、P7 SwiGLU均未完成；P0仍缺源码commit锁。不得用已有36项测试代替 specific.md 测试清单。
6. **设备准入与结果检查不足**：尚未验证真实驱动/profile；runtime 发布前没有完整有限值/guard检查，实际选核、错误恢复和硬件数值未验收。upload字节计数按整binding重复累加，分片时不是实际传输字节。以上需要在板端准入前修复并补测。

<a id="review-2026-10-20--本次清理与提交边界"></a>
#### 本次清理与提交边界

- 修正 IMPLEMENTATION-STATUS 的“主机阶段完成”和若干 `complete/host_passed` 状态，保留原有测试记录。
- 移除 run-board.sh 中未完成的 SSH/scp/执行流程：原流程只记录 core/mode、不施加配置，也未部署输入资产，并假定上传文件名。保留退出码2的明确占位入口，防止误当成验收脚本。
- 清理本模块可再生 `.pytest_cache`、`.cache`、`__pycache__`；保留1.1GB的 `.pydeps` 测试依赖，避免丢失复现环境。
- 保留原型源码、负例 fixture 和设计文档，作为后续修复起点；不因尚未完成就删除有用实现。
- 仅纳入 TileLang/RKPlan 及其必需的 command-workspace/profiled 执行接口。GDN/native scale、文档迁移、RK182X研究等原有工作区改动不混入本次功能提交，也不 discard。

下一步顺序：先修复1/2并新增绕过 Python 的 C 负例，再完成3/4；随后按 specific.md 补足 P0–P7，最后才启动 P8。

---

<a id="review-closure"></a>

## 2026-10-23 复核关闭

原文件：`REVIEW-CLOSURE.md`。以下为该文件历史内容。

<a id="review-closure--review-closure--2026-10-23"></a>
### Review closure — 2026-10-23

This closes the four review items whose evidence meanings were narrowed.  The
historical small-shape board result and its limitations are retained.

| # | Review item | Status | Code/evidence |
|---|---|---|---|
| 1 | `fp64_math` was rounded to FP16 on store/read, so it was not an unquantized math reference | **CLOSED** | Generator returns and stores `float64`; runner/verifier read `float64`; `tests/test_board_campaign.py` asserts the dtype is not representable unchanged in FP16; offline `tools/recompute_board_metrics.py` re-derived metrics from archived raw outputs in `results/20261023T103655Z-rk3588/recomputed/summary.md` |
| 2 | BO guard was checked at the first download, not at final publication after every submit | **CLOSED** | `rk_plan_runtime.c` checks every data BO with `co_guard_check` after all actions and before committing staged output; `tests/test_guard_failure.py` runs a legal `task A -> download A -> task B -> download B` mock plan where task B corrupts a guard and verifies `RK_E_DEVICE`, unchanged outputs and poisoning |
| 3 | Verifier over-claimed pair failures and hash verification | **CLOSED** | `tools/verify_board_campaign.py` makes byte-level chain/separate mismatches fatal (`dtype`, `shape`, `tobytes()`), recomputes `source-bundle.sha256`, and labels the remote loaded-library hash as `verified: false`; `tests/test_board_verifier.py` injects both a dtype-only pair mismatch and a tampered source bundle |
| 4 | Target parsing used substring matching and lost embedded NUL tokens | **CLOSED** | `src/rk_plan_target.c` matches one complete NUL-separated compatible item and the exact driver string; `tests/test_target_check.py` rejects `rk3588-extra`, `v0.9.80`, prefixed/suffixed versions and accepts the exact token |

<a id="review-closure--retained-limitations"></a>
#### Retained limitations

- Board evidence is small-shape and uses three **serial single-core** masks.
- The ragged real-device case covers one tail shape (`M=13,N=33,K=257`), not all
  1..4096 shapes.
- The loaded remote library hash is a record from the board, not a Host-side
  rehash of the `.so` bytes.
- The target gate checks an exact chip token, exact locked driver string and
  runtime ABI.  It is not a full fleet/memory-contract qualification.
- Multi-board, concurrent three-core throughput, performance and long-duration
  stress are extension validation, not MVP/P8 blockers.
- P8 remains in progress for the remaining normative release gaps; this table
  closes only the four evidence/semantic review items.

---

<a id="handoff"></a>

## 2026-10-23 逐节点 lowering 交接

原文件：`HANDOFF.md`。以下为该文件历史内容。

<a id="handoff--handoff--per-node-lowering-for-the-matmulsiluprojresidual-loop"></a>
### Handoff — per-node lowering for the MatMul→SiLU→proj+residual loop

> 历史交接快照：以下记录对应当时的实现与待办，不是当前任务清单。当前入口见 [README](../README.md)，支持范围见 [SUPPORT-DOMAIN](../SUPPORT-DOMAIN.md)。

_Updated 2026-10-23 after host closure, explicit ragged tails, and the first
RK3588 real-device admission campaign._

<a id="handoff--delivered-in-this-pass"></a>
#### Delivered in this pass

The no-K-loop path now returns a real multi-node SSA `Module` for the target
kernel; `_KBlock` and the K-loop path were not changed.

1. **Extraction (`tilelang_adapter._extract_static_dataflow`)**
   - Iterates every no-K-loop GEMM and runs the existing `_gemm_header`,
     full-region, offset, dense-layout and zero-fill checks per GEMM.
   - Associates fills, A/B copies and stores by **buffer identity + accessed
     region + statement-order def-use**.  It does not group by "one copy per
     GEMM" or by variable names.
   - Builds versioned tile definitions, so repeated stores to the same fragment
     (`p16 = acc2; p16 = p16 + X`) resolve to the correct earlier value.
   - Emits `matmul → silu → matmul → add` with the residual input edge
     `add(P, X)`, and records the five explicit source narrowing points:
     `gemm1_accumulator`, `sigmoid`, `silu_mul`, `gemm2_accumulator`,
     `residual_add`.
   - Refuses a two-GEMM branch, a wrong second-GEMM input, missing explicit FP16
     narrowing, extra stores/copies/fills and unsupported elementwise calls.

2. **Legalize (`passes/legalize._legalize_static_dataflow`)**
   - Multi-output graphs, graphs with more than one matmul, and graphs with more
     than three nodes go through per-node capability and `NumericContract`
     checks.  The experimental whole-graph SwiGLU matcher is only consulted when
     `allow_experimental=True`; otherwise canonical SwiGLU falls through to the
     same per-node legalizer.
   - Before any node is accepted it proves: at least one exported output, every
     op is reachable from an export, no non-exported temporary is unconsumed,
     role/producer combinations are consistent, and every output is produced.
     Exported intermediates may feed later nodes.
   - The old matmul/activation/binary path remains for the existing ≤3-op,
     K-loop and multi-tile cases.  The intermediate-state safety rule is now:
     **an extraction result is not planner input until legalize proves the whole
     graph connected**; do not add a bypass that sends a half-built SSA graph to
     `dag`.

3. **Lowering (`passes/dag._Builder.build_dataflow`)**
   - Maintains `value_id -> (device view, producer action)`.
   - Uploads each external binding once; weights with `b_layout="kn"` still use
     `transpose_source=True` into the custom-ops `[N,K]` weight layout.
   - Allocates one cube view per produced value and emits one submit group per
     primitive stage.  There is **no host round trip for Z/S/P**; the second GEMM
     consumes the activation's cube view directly.
   - Keeps the input X view alive through the final residual add.  The target
     graph has one download/publish; generic graphs may export several values,
     each with its own download/publish and extended lifetime.

4. **Additional graph-shape coverage (same pass)**
   - The generic static-dataflow lowerer now supports **multiple exported
     values**.  An exported intermediate may still feed a later node; each
     output gets its own download/publish rectangle and its buffer lifetime
     covers both the last device consumer and its own download.
   - `tests/test_multi_output_execution.py` covers two independent outputs and
     an exported pre-activation `Z` that is later consumed by SiLU while `P` is
     also exported.
   - The same per-node path lowers a reordered SwiGLU branch graph without using
     the whole-graph `co.swiglu_ffn.f16.v1` matcher.  `tests/test_ffn_execution.py`
     executes that graph through the mock and matches the independent numpy
     reference.  This proves the branch/shared-input/residual structure is not
     dependent on the experimental whole-graph optimisation.
   - Added `examples/silu_proj.py` and a CLI compile/validate/explain regression.
   - Added real TileLang multi-output branch extraction: two independent GEMMs
     share X and export two FP16 values.  The frontend emits a 2-node multi-output
     RKIR graph, and the plan produces two downloads/publishes
     (`tests/test_frontend_multi_output.py`, `tests/test_multi_output_execution.py`).
   - Added the explicit guarded rectangular-tail K loop (P1 batch 3): A/B tiles
     are zeroed on every K iteration, guarded affine loads cover only logical
     elements, and the output store is guarded to the logical M/N rectangle.
     The RKIR retains full logical shapes plus `source_tail_contract`; the
     existing planner generates M/N/K tails.  `ceildiv` alone is never accepted
     as a memory-safety proof.  Combined M=13/N=33/K=257 and multi-grid
     M=N=257/K=300 execute through the mock; missing zero-fill, off-by-one,
     wrong-axis and unguarded-output-store mutations are rejected.
   - Implemented the first P4 broadcast slice: explicit **constant** rank-0
     scalar and rank-1 `[N]` per-channel Add/Mul.  The source value is uploaded
     to an expanded `[M,N]` cube once at workspace creation; the existing EW
     emitter still sees two same-shape cube views.  Expansion bytes are charged
     to Host accounting, the constant upload counts once, and runtime tensor/
     dynamic/arbitrary broadcast remains rejected
     (`tests/test_frontend_broadcast.py`, `tests/test_broadcast_execution.py`).
   - Added the real three-GEMM TileLang FFN frontend for the explicit-FP16
     variant (`tests/fixtures/tilelang_ffn.py::swiglu_ffn_f16`).  It extracts the
     six-node branch graph and executes through the generic per-node path.
     `swiglu_ffn_fp32_residual` preserves the original FP32-accumulator residual
     and is a hard `RK_E_NUMERIC_CONTRACT` negative even when LUT/experimental/
     separate-submit options are enabled: this backend cannot fuse FP32
     projection+residual and then narrow once at the output.

5. **Closed-loop evidence**
   - `tests/test_silu_proj_execution.py`: real PrimFunc → RKIR → RKPlan → C ABI
     → mock NPU; independent numpy reference with all five FP16 roundings; same
     workspace and BOs run twice with different X; zero-weight case proves
     `Y == X` after both GEMMs, i.e. the residual input survived.
   - `tests/test_frontend_silu_proj.py`: SSA shape/edges and plan topology; the
     second GEMM's A view aliases the SiLU EW output buffer, and X's buffer
     lifetime extends through the add.
   - Adjacent negatives in `tests/fixtures/tilelang_silu_proj.py`:
     wrong branch (`z16` copied where `s16` is required), missing FP16 narrowing
     (FP32 SiLU fragment), unsupported `T.tanh` gate; plus dead-graph/contract
     negatives in `tests/test_dataflow.py`.
   - Both the canonical and reordered **RKIR** branch graphs use the generic
     per-node path.  The real three-GEMM TileLang FFN is also positive when the
     source materialises project16 before the residual add; the original
     FP32-residual variant stays a numeric-contract negative.

<a id="handoff--review-closure"></a>
#### Review closure

The four evidence/semantic review items from the prior round are closed in
[REVIEW-CLOSURE.md](#review-closure): unquantized float64 math reference,
final-publication BO guard, verifier byte/hash semantics, and exact NUL-token /
driver target parsing.  The table links each item to code, regressions and
archived evidence, and retains the historical limitations.

<a id="handoff--rk3588-real-device-admission-2026-10-23"></a>
#### RK3588 real-device admission (2026-10-23)

Scope: **small shapes, three single-core masks run serially**.  This is real RK3588
backend execution evidence, not a general P8/release qualification and not a
three-core concurrent-throughput result.

The board is reached through the ignored repository `.env` using the existing
host/bundle/remote-results flow.  Credentials are never printed or copied into
artifacts; the campaign performs an exact-value credential scan and fails closed.
Because this Host has no usable PTY, the SSH path uses
`SSH_ASKPASS_REQUIRE=force` instead of `sshpass`, without changing the deployment
shape.

Evidence: `results/20261023T144054Z-rk3588/`

- Device: `RK3588 OPi 5 Plus`; kernel `6.1.43-rockchip-rk3588-rknpu098`
- Driver: `RKNPU driver: v0.9.8`; NPU 1 GHz; runtime ABI 1
- `profile_hash = 954d0713...e62eb692d`; every plan matches the generated manifest
- C runtime independently validates an exact NUL-token match in the RK3588
  device-tree compatible list and the exact locked driver string before
  workspace creation (`rk_target_check`); the host regression rejects
  `rockchip,rk3588-extra`, `v0.9.80` and prefix/suffix version lookalikes.
  This remains narrower than a full fleet/driver-variation qualification.
- Source archive: `source-bundle.tgz` + `source-bundle.sha256`
- Loaded production library hash: `board/library.sha256`, also repeated in the
  runner environment record

Result: **27/27 cases passed on each of masks 1, 2, 4 (81/81)**:

- MatMul chain; projection chain / separate; staged-FP16 FFN chain / separate
- explicit guarded ragged K/M/N chain / separate
- scalar and per-channel constant add/mul (chain or separate)
- independent SiLU activation and a 256x256 physical LUT split into two chunks
- real two-output branch and M/N multi-tile MatMul
- even and odd K shard landing (`K=512`, `K=257`)
- tiled per-node projection, generic FFN, branch and activation (`M=N=K=300`,
  branch second output `300x200`)
- **real multi-block TileLang staged projection/FFN/branch** (`512x256x512`),
  each stage a separate `T.Kernel` launch with materialised global intermediates;
  same-grid cross-block dependencies and FP32-only K accumulation are rejected
- NaN input guard, NaN constant guard (workspace-create failure; no output
  binding is executed), finite-input FP16 overflow -> non-finite output guard
- full board CLI path on every core mask: `compile-rkir`, `validate`,
  `explain`, two successful `run` calls with stable fd count/output publication,
  and a missing-constant binding returning rc2 without publishing; Host
  regressions cover caller-owned fd, real void `rocket_close` ABI, and poisoned
  destroy retaining the provider fd/task BO

Numeric evidence:

- MatMul: bit-exact against the declared staged reference
- ragged: bit-exact against the declared staged reference; one real K/M/N tail
  shape only (`M=13,N=33,K=257`), not a claim over all 1..4096 shapes
- projection: declared-path `max_abs = 4.8828125e-4`
- FFN: declared-path `max_abs = 2.44140625e-4`
- multi-block TileLang staged projection / FFN / branch declared `max_abs`:
  `9.765625e-4` / `4.8828125e-4` / `2.44140625e-4`
- acceptance is `np.allclose(abs(got-declared) <= atol + rtol*abs(declared))`
  with `atol=rtol=8e-3`, sourced from the per-plan `rk_f16_tiled`
  `NumericContract.tolerance`; it is not an independent mathematical-FP64
  precision guarantee
- a second, **unquantized FP64 math** reference is retained as float64 per round
  for context; it is not narrowed to FP16 when stored/read and is not the
  acceptance predicate
- chain and separate hardware raw outputs are bit-identical for projection, FFN
  and ragged on all three masks

Guards:

- The runtime calls the existing `co_guard_check` helper for every allocated data
  BO at the **final publication gate**, after all submits/download staging and
  before any caller output bytes are committed.  A host mock regression includes
  a legal plan with task A -> download A -> task B, where task B corrupts a guard:
  publication is still refused, both outputs remain unchanged and the workspace
  is poisoned.
- NaN input and non-finite output rejections keep the pre-run output bytes
  unchanged.
- NaN constant is rejected during workspace creation before any output binding
  is executed; no "output unchanged" claim is made for that case.

Raw outputs, inputs and references are stored per round in the campaign.  The
verifier re-checks rounds, output sets/shapes/element counts, per-round task and
submit counts, declared-path numerics, raw chain/separate equality by
`(dtype, shape, bytes)`, driver, profile, runtime ABI, target check and credential
scan rather than trusting `status=PASS`.  It independently recomputes the source
bundle SHA-256; the remote loaded-library SHA-256 is labelled **recorded-only**
because the library bytes are not rehashed on the Host.

Offline recomputation of the archived raw outputs is recorded at
`results/20261023T144054Z-rk3588/recomputed/summary.md` (and for the prior archive
at `results/20261023T101551Z-rk3588/recomputed/summary.md`).  The recomputation
uses unquantized float64 math references; the latest declared-path and math-path
max errors match the campaign summary values above.

Multi-board coverage, concurrent three-core throughput, performance advantages
and long-duration stress are **extension validation**, not MVP/P8 blockers.

<a id="handoff--mvp-closure-artifacts"></a>
#### MVP closure artifacts

- [SUPPORT-DOMAIN.md](../SUPPORT-DOMAIN.md): accepted source forms, tiling,
  numerical policy and explicit non-support boundaries.
- [ACCEPTANCE-MATRIX.md](../ACCEPTANCE.md#mvp): one-row-per-necessary-condition
  map to code, tests and archived logs, with implementation/verification/
  unsupported/future/blocked classifications.
- [REPLAY.md](../REPLAY.md): Host, board, CLI, offline-recompute and source-build commands.
- [REVIEW-CLOSURE.md](#review-closure): the four evidence-semantic review items.
- [RUNTIME-LIFECYCLE.md](../RUNTIME-LIFECYCLE.md): command/task/data BO ownership,
  pre-submit safety, unknown-state poisoning/retention, BUSY and CLI cleanup.
- P0 source build: **COMPLETE** via resumable GitHub mirror archives and an
  independent CMake tree; see [provenance/p0-source-build.json](../provenance/p0-source-build.json)
  and `provenance/p0-source-build/`.  The corrected source env excludes the wheel
  `tvm_ffi` package/RECORD; `find_libtvm_ffi()` and `/proc/self/maps` bind the
  source-built `libtvm_ffi.so` (SHA-256 `3136acaf...59242594`).  Earlier timeout
  retries remain as history.

<a id="handoff--runtime-contract-closure-2026-10-23"></a>
#### Runtime contract closure (2026-10-23)

Three code-level contract issues are closed:

1. `rocket_close` is declared and called as the vendor `void rocket_close(int)`;
   no return value is inspected or invented.  The C signature probe compiles
   against `rocket_npu.h`, and the Python ctypes mirror is tested in
   `tests/test_runtime_abi.py`.
2. A poisoned/BUSY workspace destroy never leads to `rocket_close`: the provider
   fd/context (including `taskbo`) are retained for process-level recovery.  The
   CLI reports the cleanup failure, preserves the original execution error and
   does not close its owned fd after an unconfirmed destroy.  Caller-owned fd is
   never closed.
3. `poisoned` is `atomic_int`; runs acquire the atomic busy gate before observing
   poison, a failing run sets poison while holding busy, and destroy also acquires
   the busy gate before checking it.  A concurrent failing-run regression verifies
   the entrant gets `RK_E_BUSY`, no second submit occurs, and later calls get
   `RK_E_POISONED`; no output is published.
4. The provider context itself is quarantined per fd: an `atomic_int faulted` is
   set by `submit_kick()` **before releasing `submit_lock`** on any submit
   failure.  A queued submit acquires the lock afterwards and returns `-EPIPE`
   without rewriting the task BO.  Existing and newly created workspaces on the
   same fd are refused, outputs stay unpublished, and `rocket_close` on a
   faulted context retains the fd/task BO for process-level recovery.  The
   provider patch and base commit are recorded in
   `provenance/upstream-patches.json` and
   `provenance/patches/rknpu-submit-context-fault.patch`.

Details and exact test/log references: [RUNTIME-LIFECYCLE.md](../RUNTIME-LIFECYCLE.md).

<a id="handoff--final-mvp-status"></a>
#### Final MVP status

The implemented and board-verified domain is complete for the finite representative
matrix: real TileLang multi-block staged projection/FFN/branch, tiled RKIR graphs,
CLI, guards, bindings/BO/budget/BUSY/pre-submit/unknown-poison host regressions and
same-board execution.

The declared MVP scope is **complete** for the finite support domain and the
representative RK3588 campaign.  Failure handling is part of that scope only as
safe isolation and stop: after an unknown fault the provider context is
quarantined, retries are refused and resources are retained for process-level
disposal.  Recovery/continued computation, driver quiescence and physical release
are explicitly **out of scope, unmeasured and not claimed**, rather than an
implicit extra MVP gate.  P0 source rebuild is complete and the source-built
`libtvm_ffi.so` load path is verified.

<a id="handoff--verification-state"></a>
#### Verification state

* `pytest tests` → **219 passed, 41 subtests**, 0 skipped.
* `ctest --test-dir research/custom-ops/build-host-mvp` → **27/27 passed**.
* `scripts/run-verification.sh --keep-build` → exit 0;
  `results/20261024T032922Z-host/`:
  - pytest 219 passed / 0 failed / 0 skipped;
  - ASan+UBSan contract/host checks pass;
  - evidence complete.  This host record opens no device; the RK3588 evidence is
    the separate campaign above.

<a id="handoff--resolved-path-decision-2026-10-23"></a>
#### Resolved path decision (2026-10-23)

Approved behaviour: the canonical SwiGLU graph uses the **generic per-node path by
default**, just like any equivalent reordered graph.  `allow_experimental=True`
merely permits the optional whole-graph `swiglu_ffn_v1` specialisation; it is not
a gateway for compilation.  The specialisation still enforces its own capability,
numeric, layout and range checks, and `co.swiglu_ffn.f16.v1` remains
**experimental**.

Implemented and pinned by:

* `tests/test_swiglu.py`: canonical default selects `linear_k_acc_v1` /
  `gate_ew_v1` per-stage recipes, while `allow_experimental=True` selects
  `swiglu_ffn_v1`; strict / missing-LUT / >256-width rejections are checked on
  both paths; the capability still cannot be used through `profile.enabled()`.
* `tests/test_ffn_execution.py`: canonical default and gate/up-swapped graphs both
  execute via the generic path and match the independent numpy reference.
* The obsolete "experimental, no composed numeric evidence" rejection text was
  removed from `legalize`.

<a id="handoff--remaining-boundaries-do-not-overclaim"></a>
#### Remaining boundaries (do not overclaim)

* The no-K-loop frontend supports the 2-GEMM projection chain, the 3-GEMM
  staged-FP16 SwiGLU branch, two independent matmuls exporting two values, a
  single GEMM followed by additive/multiplicative rank-0 or rank-1 constant
  broadcast, and the explicit guarded ragged K/M/N form.  Composite graphs are
  limited to one primitive tile per node; the ragged K form retains logical
  M/N/K and delegates tail generation to the existing planner.  An FP32
  projection+residual narrowed only by the final copy is a deliberate
  numeric-contract rejection; no implicit reordering is allowed.
* Broadcast is deliberately narrow: only compile-time constants, only scalar or
  `[N]` per-channel, only Add/Mul, and only a full rank-2 peer operand.  Runtime
  tensor broadcast, dynamic Host scalars, arbitrary NumPy broadcasting and
  native scalar EW templates remain unsupported.
* The classic `_detect_k_block` path is untouched; the new ragged K form is a
  separate guarded-tail extractor.  K reduction, M/N multi-tile and the
  `co.mm_acc` ping-pong remain on the old planner path.
* The generic graph lowerer supports matmul, the registered unary activations,
  and same-shape add/mul.  It is not a general arbitrary-DAG compiler.
* The RK3588 campaign above is real-device evidence for the listed small shapes
  and policies, not a general performance or production qualification.  The
  broader release policy in `IMPLEMENTATION-STATUS.md` still applies.
* RK1228 is a separate target.  Its identification, profile, runtime/driver ABI
  and memory contract must be captured from the real device; see
  [RK1228-TARGET.md](../RK1228-TARGET.md).  No RK3588 register recipe or acceptance
  result transfers.

<a id="handoff--two-invariants-for-the-next-change"></a>
#### Two invariants for the next change

1. **Never hand a partially connected SSA graph to the planner.**  Extraction
   may produce several ops, but `legalize` must first prove reachability,
   consumer closure and the exported-root set.  Keep the current dead-node and
   unconsumed-temporary rejects when extending the op set.
2. **Keep intermediates in device BOs.**  The second GEMM must consume the
   previous stage's cube view; do not add a download/upload bridge and do not let
   storage reuse X before the residual add.  The zero-weight `Y == X` test and
   the X lifetime assertion are the regression guards for this.

---

<a id="source-lock"></a>

## 源码、依赖锁与源码构建验收

原文件：`P0-SOURCE-LOCK.md`。以下为该文件历史内容。

<a id="source-lock--p0源码与环境锁"></a>
### P0：源码与环境锁

日期：2026-10-20。范围：只做锁定与重验，不混入前端功能扩展。

**总状态：`complete`（源码构建已完成；真实设备故障恢复不属于 P0）。**

| 交付项 | 状态 | 证据 |
|---|---|---|
| 源码 commit + 子模块锁定 | 完成 | `provenance/source-lock.json`、`provenance/upstream-trees/*.json` |
| 制品哈希锁定（依赖） | 完成 | `requirements.lock`（28 个制品，SHA-256） |
| 当前 wheel Host 回归 | 通过 | `.pydeps` 基线记录（计数与第 4 节一致） |
| 隔离依赖环境重验 | 完成 | `.venv-p0` + `.pydeps-p0`，`isolation` 步骤断言通过 |
| 完整 Python 文件集合校验 | 完成 | `provenance/wheel-vs-source.json`：0 缺失、0 无法解释的额外文件 |
| 源码构建及产物重验 | **完成** | [provenance/p0-source-build.json](../provenance/p0-source-build.json)、`provenance/p0-source-build/` 构建日志与库哈希、源码树清单 |

<a id="source-lock--1-源码锁"></a>
#### 1. 源码锁

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/tile-ai/tilelang` |
| 标签 | `v0.1.15` |
| commit | `a35f8ddf45eba16c21211ec8822d56ce5363036f` |
| 解析方式 | GitHub `git/ref/tags` API（`git ls-remote` 在本机多次 TLS 中断/超时，仅作回退） |
| `3rdparty/tvm` | `4211874e9e4b8770fd20ad0d98ab80afd1df5028`（`apache/tvm`） |
| `3rdparty/cutlass` | `b2dd65dc864e09688245b316ac46c4a6cd07e15c`（`NVIDIA/cutlass`） |
| 根 tree 条目 | 2265，未截断 |

原始 GitHub tree 响应归档在 `provenance/upstream-trees/`，可离线复核上面的
commit/SHA，不必相信脚本的算术。

<a id="source-lock--最小补丁"></a>
##### 最小补丁

**无补丁。** 后端只读取上游公开 Python 对象（`PrimFunc`/`SBlock`/`Call`/`BufferLoad`/
`BufferStore`/`For` 等），不修改 TileLang/TVM 源码；`source-lock.json` 中
`patches: []` 与 `patch_policy` 显式记录该结论。

<a id="source-lock--2-安装内容--锁定源码的范围化校验"></a>
#### 2. 「安装内容 == 锁定源码」的范围化校验

按 6 个显式 scope 比较 git blob SHA（`sha1("blob <size>\0" + content)`）：

| scope | 覆盖 | 完整性口径 | 比较 | 一致 |
|---|---|---|---|---|
| `tilelang-package` | `tilelang/**`（除 3rdparty/src/lib） | 完整 | 397 | 397 |
| `tilelang-cpp-src` | `tilelang/src/**` ↔ 仓库根 `src/` | 完整 | 394 | 394 |
| `tvm-python` | `…/3rdparty/tvm/python/**` ↔ TVM `python/` | 完整 | 747 | 747 |
| `tvm-include` | `…/tvm/include/**` ↔ TVM `include/` | 完整 | 189 | 189 |
| `tvm-src` | `…/tvm/src/**` ↔ TVM `src/` | 完整 | 929 | 929 |
| `cutlass-subset` | `…/cutlass/**` ↔ cutlass tree | **仅子集**（无完整性声明） | 988 | 988 |

结果：

```text
参与比较: 3644 文件，全部逐字节一致，不一致 0
Python：完整 scope 内安装 1135 / 上游 1135
        缺失（上游有、安装无）= 0
        无法解释的额外 Python = 0
未归入任何 scope 的安装文件 = 0
```

仅有两类文件不是「源码可比较」的，均被逐条列出并说明：

- `tilelang/3rdparty/tvm/version.py` —— TVM 构建期生成，allowlist 记录原因；
- 12 个编译产物 `tilelang/lib/*.so` —— 记录各自 SHA-256，明确标注 provenance 是
  wheel 制品哈希而非源码匹配（`wheel-vs-source.json: binary_artifacts`）。

因此准确结论是：**上述 6 个 scope 内、3644 个文件（含完整 scope 的全部 1135 个 Python
文件）与锁定源码逐字节一致，无缺失、无无法解释的额外文件；编译产物不在该断言范围内。**

<a id="source-lock--3-依赖锁与隔离环境"></a>
#### 3. 依赖锁与隔离环境

`requirements.lock`：**28 个制品**，固定版本 + SHA-256，用
`pip --no-deps --require-hashes` 安装。此前遗漏的 `jsonschema`（`pyproject.toml` 声明的
依赖）及其闭包（`attrs`、`jsonschema-specifications`、`referencing`、`rpds-py`）已补入。

```bash
scripts/lock_deps.sh lock      # 生成 requirements.lock
scripts/lock_deps.sh install   # 装进全新的 .pydeps-p0
scripts/lock_deps.sh verify    # 解释器隔离 + 无重复 metadata + 导入路径
```

`pip --target` 只填充目录、不隔离系统包，因此：

- `python3 -m venv --without-pip .venv-p0`（本机无 ensurepip）——该 venv **不继承**
  `/usr/lib/python3/dist-packages`；
- 验证、pytest、CLI 子进程全部由 `.venv-p0/bin/python` 运行；
- `scripts/check_isolation.py` 断言：`sys.path` 无外部 dist/site-packages、每个第三方模块
  与每个发行版位置都在 `.pydeps-p0` 内、发行版集合与 `requirements.lock` 逐项相等；
  任一不满足即让整轮验证失败。

上一轮 `jsonschema` 来自 `/usr/lib/python3/dist-packages` 的问题已消除：本轮
`isolation.log` 记录其来源为 `.pydeps-p0/jsonschema/__init__.py`，`foreign_paths: []`。

<a id="source-lock--4-行为重验隔离环境"></a>
#### 4. 行为重验（隔离环境）

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 scripts/run-verification.sh
```

结果目录：`results/20261022T084627Z-host/`

| 项 | 结果 |
|---|---|
| 解释器 | `.venv-p0/bin/python`（`isolated_venv: true`） |
| isolation 步骤 | 通过（`ISOLATION OK`，0 problems） |
| ctest | 26/26 |
| `rk_plan_contract` / `rk_plan_host` | 26 / 67 checks，0 failures（`MALLOC_PERTURB_=128`） |
| Sanitizer（显式 flags + 全新 build 目录） | 26 / 67 checks，0 failures |
| pytest | 68 passed，**0 skipped** |
| evidence_complete | true |

与 `.pydeps` wheel 基线的计数逐项一致，无回退、无跳过，因此本次不存在需要解释的
AST/API 差异。

<a id="source-lock--5-当时的源码构建缺口已由第-9-节闭合"></a>
#### 5. 当时的源码构建缺口（已由第 9 节闭合）

以下**已尝试的获取路径**在本机失败，结论不外推到其他环境：

| 尝试路径 | 实际结果 |
|---|---|
| `git clone --depth 1 --branch v0.1.15` | `GnuTLS recv error (-110)`，重试仍超时 |
| `codeload.github.com` tarball（`-m 55`） | 收到 9,390,763 B 后超时（≈167 KB/s） |
| `raw.githubusercontent.com` 单文件 | 50 s 超时，0 字节 |
| PyPI sdist `--no-binary :all: tilelang==0.1.15` | 该版本没有 sdist（只有 0.1.7–0.1.14） |

55 秒超时与解压体积不足以证明任何环境都无法取得源码，只说明这几次尝试失败。

闭合方式（需直连 GitHub 或可用镜像）：

1. `git clone --recurse-submodules` 到 `v0.1.15`，校验 `3rdparty/tvm == 4211874e…`；
2. 按其构建脚本编译 `libtvm`，把产物哈希写入 `provenance.json`；
3. `TILELANG_DEPS_DIR` 指向源码构建产物重跑第 4 节，与
   `results/20261022T084627Z-host` 逐项比对；若出现 AST/API 差异，先修适配并解释，
   不放宽拒绝规则来维持通过数。

在此之前本模块是 **Host 原型、P0 部分完成、板端未验收**；P1 的 K 循环工作可基于当前
已记录的 wheel 基线推进，但不得写成源码构建验收通过。

<a id="source-lock--6-复现命令"></a>
#### 6. 复现命令

```bash
cd research/tilelang-rknpu

TILELANG_DEPS_DIR=$PWD/.pydeps-p0 python3 scripts/lock_source.py   # 源码一致性
./scripts/lock_deps.sh lock                                        # 依赖锁
./scripts/lock_deps.sh install                                     # 干净安装
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/lock_deps.sh verify     # 隔离校验
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh     # 全量重验
```

全部命令不打开设备。

<a id="source-lock--7-2026-10-23-有界源码重试"></a>
#### 7. 2026-10-23 有界源码重试

按“有界尝试、失败保存具体阻塞、继续其他任务”的规则，再次尝试从锁定 tag 获取源码：

```text
command: timeout 90s git clone --depth 1 --branch v0.1.15 \
         https://github.com/tile-ai/tilelang.git .src/tilelang-v0.1.15-attempt
exit:    124 (timeout)
log:     provenance/p0-rebuild-attempt.log
```

未保留半下载目录；没有源码或 `libtvm` 产物被标记为成功。**当时** P0 为
`in_progress / externally blocked`；后续第 9 节改用可续传镜像完成了源码构建。

同一尝试的机器可读记录：`provenance/p0-rebuild-attempt.json`。

<a id="source-lock--8-2026-10-23-第二次有界获取尝试"></a>
#### 8. 2026-10-23 第二次有界获取尝试

在继续独立开发前再次尝试 codeload tarball，仍未完成：

```text
command: timeout 60s curl -L --fail --connect-timeout 15 --max-time 55 \
         -o /tmp/tilelang-v0.1.15.tar.gz \
         https://codeload.github.com/tile-ai/tilelang/tar.gz/refs/tags/v0.1.15
exit:    28 (Operation timed out)
received: 2,673,599 bytes (partial archive discarded)
log:     provenance/p0-rebuild-attempt-2.log
record:  provenance/p0-rebuild-attempt-2.json
```

没有可用完整源码，也没有执行/伪造源码构建产物。**当时** P0 仍未完成；后续
第 9 节通过可续传镜像完成，且仍不把部分下载当作源码证据。


<a id="source-lock--9-2026-10-23-源码构建完成"></a>
#### 9. 2026-10-23 源码构建完成

使用可续传的 `gh-proxy.com` 镜像取得锁定 commit 的 GitHub archives；archive 名称包含
commit，下载后计算 SHA-256 并解压到独立目录：

| 组件 | 锁定 commit | archive SHA-256 | 用途 |
|---|---|---|---|
| TileLang | `a35f8ddf45eba16c21211ec8822d56ce5363036f` | `2e85edecef4cbeb795f4f8a3b2ac2ad4020d89f3d35687c624c3bbcf429b73bc` | compiler/frontend source |
| TVM | `4211874e9e4b8770fd20ad0d98ab80afd1df5028` | `ee73c88f82d949f451e541182f69973f8c4c452f658695442931c686faffadfa` | `libtvm_*` |
| tvm-ffi | `3c35034fd1026011736e19a4e0e1ed0f22058c42` | `b2c1947d91c0b7e8092d38d31c60f0ea5ad34bc45a05b315224774bdd0dd86a7` | TVM FFI |
| dlpack | `84d107bf416c6bab9ae68ad285876600d230490d` | `9978ddb7f533fcee7e0c0ab81a978257d3f8dc475eb83319a5e632738cafb271` | DLPack headers |
| libbacktrace | `793921876c981ce49759114d7bb89bb89b2d3a2d` | `858b1225351052234f7e71f0f36622fc9ad33aac947db7816d0b443ae0dd33ce` | TVM FFI optional backtrace, source supplied |

构建方式：

- 独立 build tree：`.src/p0-build-tilelang`；源树：`.src/p0-build-src/{tilelang,tvm}`。
- `TVM_ROOT` 指向锁定 TVM 源；TVM 的 `3rdparty/tvm-ffi`、`dlpack`、`libbacktrace`
  使用上述锁定源码。
- 本次目标为 CPU/编译器/RKNPU 后端，显式关闭 CUDA、ROCm、Metal、LLVM、Ascend、
  OpenCL；因此未下载/编译 cutlass（wheel 内 cutlass 子集不参与本次 native 构建）。
- 构建使用 source-built `libtilelang.so`、`libtvm_compiler.so`、`libtvm_runtime.so`、
  `libtvm_ffi.so` 和 Cython wrapper。
- `tvm_ffi` Python wrapper 必须使用独立 source env：排除 wheel 的 `tvm_ffi` 包和
  `apache_tvm_ffi-*.dist-info`，使用源 `tvm-ffi/python/tvm_ffi` wrapper 与独立构建的
  `core.abi3.so`。`find_libtvm_ffi()` 和 `/proc/self/maps` 均指向
  `.src/p0-build-tilelang/lib/libtvm_ffi.so`，SHA-256
  `3136acafd189050bf37338c78243a148ddc63e67352bdc550a7be25759242594`；
  见 `provenance/p0-source-build/tvmffi-load-path.log`。
- 第一次验证曾因 wheel RECORD 加载 `.pydeps-p0/tvm_ffi/lib/libtvm_ffi.so`，该证据
  保留为 `import-paths-pre-fix-wheel-tvmffi.log`，不再作为 P0 关闭依据。
- 仍来自预构建依赖的明确项：Z3 `.pydeps-p0/z3`、Cython 3.0.8（TileLang wrapper）/
  3.3.0（tvm-ffi core）、Ubuntu 3.12.3 dev headers；Python `tvm_ffi` wrapper 已改为
  源包，native `libtvm_ffi.so` 为源码构建。均记录在
  `provenance/p0-source-build.json`。

实际加载路径：

- TileLang：`.src/p0-build-src/tilelang/tilelang/__init__.py`
- TVM/tirx：`.src/p0-build-src/tilelang/3rdparty/tvm/python/tvm/...`
- native libs：`.src/p0-build-tilelang/lib`
- 日志：`provenance/p0-source-build/import-paths.log`

验证结果：

- source-built import + full pytest（修正 tvm_ffi 加载路径后重跑）：
  `213 passed, 40 subtests`
- CTest（同一 Host native build）：`27/27`
- 27 个 campaign plan 中，wheel 与 source-built 的
  `bindings/constants/buffers/views/actions/submits/publish/accounting` 全部逐项相同；
  只有 source span/hash 元数据差异（`plan-equivalence.json` 全部 `semantic_equal=true`）。
- `find_libtvm_ffi()`、`tvm_ffi.LIB._name` 和 `/proc/self/maps` 的实际 native 路径
  与哈希见 `tvmffi-load-path.log`；wheel tvm_ffi 包/其 RECORD 不在 source env 中。
- 用修正后的 source-built TileLang/TVM + source tvm_ffi 重新生成同一 27-case campaign，
  并在同一 RK3588 上执行：`results/20261023T144054Z-rk3588/`，**81/81 PASS**，
  三核 CLI PASS，source bundle hash 独立重算通过。

构建系统的一条诊断信息 `Found TVM_GIT_COMMIT_HASH=9baf...` 来自 TVM 源树位于
外层 git checkout 内这一事实；实际 TVM 版本以 archive commit `4211874e...` 和
源码树清单为准，不把该诊断字段当作 TVM commit。

---

<a id="model-requirements"></a>

## 已完成的两模型里程碑需求

原文件：`NEXT-MILESTONE.md`。以下为该文件历史内容。

<a id="model-requirements--完整模型推理已完成里程碑的需求"></a>
### 完整模型推理：已完成里程碑的需求

> 文件名保留供原有链接与脚本引用；本文不再代表“下一步”。验收见 [MODEL-MILESTONE-STATUS](../ACCEPTANCE.md#models)，后续优化从 [项目首页](../README.md) 进入。

日期：2026-10-24。本文是已验收静态 FP16 MVP 之后的独立里程碑，不追溯增加上一阶段的验收条件。现状见 [支持域](../SUPPORT-DOMAIN.md) 和 [验收矩阵](../ACCEPTANCE.md#mvp)。

**实施状态（2026-10-24）：两模型里程碑完整关闭。** 在首轮模型跑通之后，追加关闭
了三个缺口：schema-v2 计时边界、版本化 rkhybrid 计划化 Host 边界、以及冻结的逐层/
逐 profile 数值门禁和独立 verifier。冻结权重、真实 TileLang 源、RKPlan/rkhybrid
清单、完整测试集结果、板端成本与精确非声明见
[MODEL-MILESTONE-STATUS.md](../ACCEPTANCE.md#models)。

关闭证据摘要：板端最终任务 campaign `results/20261024T071144Z-models/`；独立
train-calibrated 数值 campaign `results/20261024T091634Z-numeric/` 的 verifier PASS。
当前数值政策在 `models/numeric_acceptance_train.json`：32 张无标签 train 校准样本，
32 张无标签且不重叠的 test 验收样本，33 个逐层门禁和 5 个 profile logits 门禁。
真实 TileLang MLP 诊断计划导出两层隐藏层 linear+bias、ReLU 和最终 preactivation；
生产计划仍只下载最终 logits。
`models/numeric_acceptance.json` / `numeric_golden.npz` 仅保留为历史
test-envelope 回归证据，不再作为独立数值验收依据。

<a id="model-requirements--1-总目标"></a>
#### 1. 总目标

在 RK3588 上，以真实学习所得的权重和标准数据集完成两个完整模型：核心计算全在 NPU 的 MNIST MLP，以及显式 hybrid 的小型 Transformer 分类器。交付权重/模型描述、真实 TileLang/TVM 输入、离线编译、部署、重复推理、任务准确率和成本报告的可重放流程。

按完整模型的需要补齐后端能力。复用 TileLang 前端、TVM IR、现有 RKIR/RKPlan 和 runtime；模型接入层只负责有限层类型、连接、常量和 shape，不另造通用语言前端或完整 ONNX/PyTorch 编译器。所有 NPU 子图仍经过真实 TileLang/TVM 接入，手写 RKIR 只作后端调试与测试。

<a id="model-requirements--2-固定模型与运行范围"></a>
#### 2. 固定模型与运行范围

| 模型 | 首轮结构 | 计算落点 |
|---|---|---|
| MNIST MLP | Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU → Linear(128,10)；batch独立编译为1、8、32 | Linear、bias、ReLU在NPU；图像归一化/输入整理为Host准备，argmax为应用后处理；中间激活设备驻留 |
| Tiny Transformer 图像分类器 | 同一MNIST；28×28图像分16个7×7 patch；16 tokens、D=64、4 heads、1层encoder、FFN=128、10类；batch独立编译为1、8 | patch projection、Q/K/V、QK/PV、FFN、分类头的矩阵计算在NPU；首轮RMSNorm、稳定Softmax、mean pooling可用已注册Host compute |

Transformer固定为patch projection/位置参数、pre-RMSNorm attention/residual、pre-RMSNorm SwiGLU FFN/residual、mean pooling和分类头。不包括causal mask、KV cache、增量decode或持续状态。训练/取得权重在Host进行，不扩展为NPU训练。

优先使用来源与许可明确的适配checkpoint；没有可用权重时提供固定seed的Host训练脚本并冻结产物。随机权重、只跑中间层、挑少量样本或只跑FFN，均不能替代完整模型验收。测试集不得参与训练或阈值调参。

<a id="model-requirements--3-必需的能力增量"></a>
#### 3. 必需的能力增量

| 优先级 | 能力 | 实现边界 |
|---|---|---|
| A | 模型包/权重绑定 | 记录结构版本、预处理、输入输出、常量、shape/dtype、数值政策、来源与哈希；先用明确模型描述和安全数组格式，不要求通用模型格式导入；禁止pickle |
| A | ReLU语义及lowering | 优先复用已有硬件配置；新增配方须单独验证，不能偷换成HardSigmoid/LUT近似或改变收窄点 |
| A | 多层Linear+bias+activation与尾块 | 复用既有广播/K链/tiling；不按模型名、层数或784/10尺寸做整图特判 |
| A | 模型级运行接口 | 一次创建/打包/上传常量，多次更新输入；workspace复用；保持BUSY、失败不发布和quarantine契约 |
| B | 静态3-D/4-D逻辑shape、轴含义、view/reshape | 用于batch/token/head表达，物理emitter仍可使用2-D矩阵；仅消除已证明保持元素对应关系的视图，不承诺任意rank或动态shape |
| B | 受限transpose、split/merge heads、patch整理、位置参数展开 | 区分元数据、搬运和算术；布局动作显式入计划；Cube不能直接作为weight布局 |
| B | 显式hybrid的RMSNorm、稳定Softmax、mean pooling | 复用custom-ops已有纯算法/helper，明确Host计算、传输、同步和预算；不能用不透明co_graph_run替代新后端执行 |
| B | 注意力运行时中间操作数与feature→weight桥接 | 首轮允许hybrid下显式download/Host pack/upload，记录每次调用的版本与流量；不要求未经验证的设备布局原生直连 |
| B | 模型所需明确广播规则 | 按固定轴、shape和来源实现位置参数/行缩放；不能直接宣布任意NumPy广播。可先在norm/softmax复合节点内定义所需行为 |

NumericContract继续约束逐K片FP16写回、激活与残差收窄。hybrid仅允许已注册Host action；不支持操作仍须拒绝，不能隐式CPU fallback。纯NPU与hybrid采用明确分开的profile/政策和报告。

需要扩展schema/C ABI时版本化修改，能复用现有结构就不增加第二套模型IR。持久化计划不保存任意Python callable。

<a id="model-requirements--4-连续工作包"></a>
#### 4. 连续工作包

<a id="model-requirements--n0冻结模型与参考"></a>
##### N0：冻结模型与参考

- 保留当前MVP回归基线，冻结两个模型结构、权重来源、预处理和输入输出。
- 取得CPU参考；完整准确率使用官方MNIST test split的10,000张图像。
- 同时提供原始模型FP32参考和声明硬件路径参考；后者保留FP16/LUT/Host计算边界。
- 形成简短能力差异清单后直接实现，不另立模型导入平台设计项目。

<a id="model-requirements--n1完整-npu-mlp"></a>
##### N1：完整 NPU MLP

- 接入全部权重、bias、ReLU，用真实TileLang staged程序编译执行。
- batch=1/8/32形成独立编译产物；未编译shape精确拒绝，不称为动态shape支持。
- 打通预处理→整个网络→logits→argmax→测试集评估，常量创建时上传一次，中间激活不出设备。
- 先正确执行，再选择有数值依据的融合/调度优化。

<a id="model-requirements--n2完整-hybrid-tiny-transformer"></a>
##### N2：完整 hybrid Tiny Transformer

- 实现必要的多轴view、head布局、节点和hybrid执行边界。
- 首轮允许有报告的Host RMSNorm/Softmax/mean pooling与布局桥接，尽快运行整个模型。
- QK/PV及全部Linear/FFN不能悄悄退回CPU；缺失lowering即列为未完成。
- 比较逐层参考与最终分类，记录所有Host action。完整模型通过后才选择一个主要桥接瓶颈优化，不新增首轮“纯NPU Transformer”门槛。

<a id="model-requirements--n3封装准入与交接"></a>
##### N3：封装、准入与交接

- 提供模型包、编译/推理/评估入口、两个独立可运行示例和准确支持域。
- 重用既有源码环境、campaign、sanitizer和故障回归；新增测试对应本阶段实际风险。
- 结果关联输入/权重/数据集哈希、编译产物、环境和完整命令；大文件沿用本地证据规则。
- 按模型包/参考、MLP、hybrid/Transformer、验收文档分批提交，中间提交不需要逐步审批。

<a id="model-requirements--5-有限验收条件"></a>
#### 5. 有限验收条件

1. **完整性**：两个模型全部层和学习所得权重参与执行；提供来源、预处理及输出语义。不能用随机权重或算子演示替代模型结果。
2. **编译路径**：NPU子图均经真实TileLang/TVM→RKIR→RKPlan→生产runtime；Host准备/计算明确记录。
3. **任务精度**：CPU FP32在10,000张测试图像上，MLP准确率至少97%，Tiny Transformer至少95%。RK3588相对各自CPU基线准确率下降不超过0.5个百分点，top-1一致率至少99%。未达到时定位权重/预处理/数值/lowering，不反向放宽门槛。
4. **数值分层**：按运算和shape登记声明路径容差与来源，在真机运行前冻结；分别报告对声明路径和FP32模型的logits误差。分类阈值不能替代算子检查，不沿用一个8e-3作为全部新能力保证。
5. **batch与尾部**：MLP batch=1/8/32、Transformer batch=1/8均验证；784维输入、10类输出等非对齐边界正确。测试集尾批采用已编译profile拆分/处理并记录策略。
6. **驻留复用**：MLP中间Tensor设备驻留；两个模型常量/workspace复用，输入更新无过期结果。Transformer桥接/Host算术可审计，失败不发布。
7. **板端范围**：完整测试集在一个固定核上完成；三个核分别运行同一确定性样本子集做准入。仍为单核串行，不新增并发、多板、长压或危险timeout实验。
8. **成本**：报告cold/warm、batch延迟/吞吐、准备、提交等待、传输、Host compute和端到端；CPU使用同一模型/数据/预处理，列线程数和频率。不预设NPU必然更快。
9. **依赖与故障**：源码/补丁/新增依赖可追溯；无隐藏fallback，quarantine与停止策略有效，故障恢复继续不承诺。

N1通过即可交付“完整MLP推理可用”的阶段成果；完成N2后才关闭本文定义的两模型里程碑。

**关闭记录（2026-10-24）：**

- 计时：`evaluate_models.py` 使用 schema v2，分离创建/常量准备、warm-up、
  预处理、执行、后处理与参考计算；参考秒数不在模型区间，CPU baseline 使用与
  NPU 相同的静态 batch/tail schedule，并在最终板端 campaign 中显式设置单线程
  BLAS 环境。旧 campaign 的性能字段已被标记失效。
- hybrid 计划：Transformer 与 MLP 均执行版本化 `rkhybrid` 清单；Host 节点只引用
  注册 action，NPU 节点引用不可变 RKPlan。清单显式记录 tensor shape/axes、输入
  输出、依赖、数值契约、bridge 记账和重复 QK/PV 轴；没有整模型不透明 Python
  节点或 fallback。
- 数值验收：真实 TileLang ReLU probe（EW ADD + EW MAX）、动态-B QK/PV、两模型
  关键层以及 MLP 两个隐藏层的 linear+bias/ReLU 均有可观测诊断计划。校准只使用固定
  train 子集；限值在板端 train 观测后冻结，随后才运行与 train 样本逐像素哈希不重叠
  的固定 test 子集。33 个逐层门禁和 5 个 profile logits 门禁用 `rtol=0`，并结合
  train 观测与 FP16 表示尺度；test 最大值/标签/top-1 不参与选参。verifier 从原始
  logits、独立参考、权重、计划和哈希重算，检查无重叠性，并拒绝“隐藏层超容差但最终
  top-1 不变”以及“top-1 不变但 logits 放大”的篡改。此前已查看 test 结果的事实
  明确披露，因此不宣称盲测。

<a id="model-requirements--6-范围控制"></a>
#### 6. 范围控制

动态shape通用编译、任意TileLang控制流/mask、通用ONNX导入、CNN全算子、NPU训练、INT8/INT4、完整LLM加载、KV cache/state、故障后继续计算、RK1228适配、多板/多核并发均不作为本轮完成条件。

RK1228到位后另行识别型号、驱动、内存与提交接口，再作为独立target复用模型/IR语义；不继承RK3588配方或验收结论。

执行agent自行决定本范围内的helper、接口细化、文件组织和实施顺序；卡点定位到具体语义/布局/数值条件，保留复现并继续独立工作，不反复请求相同路线审批。仅在公共契约需变更、破坏性操作或缺少必要外部条件时提问。

审核以完整模型、真实数据、后端能力和证据为主，不以mock接口数量或验证工具扩张代替交付。当前MVP保持已完成，不因本阶段新能力未实现而改写历史验收。
