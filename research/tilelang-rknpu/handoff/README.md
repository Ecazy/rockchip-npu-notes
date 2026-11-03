# Codex 长期接手包：TileLang → RKNPU 结构化编译试点

本包按用户明确要求新增，用于跨会话接手和主线程/执行代理协作。
它不替代 `refactor-overview.md`、`refactor-specific.md` 和唯一验收主报告
`INTENT-FIDELITY-AUDIT.md`，不再复制完整逐轮历史。

**接手时状态：尚未验收通过。当前唯一已确认的最新阻断项是主报告 §74：
`SBlockRealize.predicate` 被忽略，`T.where(False)` 仍执行 GEMM。**
“唯一”指本次已确认且尚未关闭的最新条目，不是对所有潜在问题的保证。

## 1. 使用顺序

1. 保留当前完整工作区，勿仅从 Git HEAD 新建一个空 worktree。
2. 按 [LEAD-START.md](LEAD-START.md) 启动主线程并粘贴角色 prompt。
3. 在同一线程启动该文件给出的 `/goal` 目标。
4. 主线程读取本文件、[STATE.md](STATE.md)、[BASELINE.json](BASELINE.json)，
   核实事实后创建执行代理，消息引用 [EXECUTOR-START.md](EXECUTOR-START.md)。
5. 之后由主线程连续派工、审核、退回修复，用户无需逐轮传达许可。

本次只编写接手材料；没有启动 Goal、执行代理或新的代码修复。

## 2. 固定角色和权限

| 项 | 主线程 | 执行代理 |
|---|---|---|
| 用户指定模型 | GPT sol 6.1，medium | GPT 6 Luna，low |
| 当前工具/客户端使用的标识 | `gpt-6.1-sol` / `medium` | `gpt-6-luna` / `low` |
| 职责 | 目标管理、任务拆分、独立审计、接受/退回、进展说明 | 调查、实现、正式测试、构建、设备实验、证据整理 |
| 实现/正式测试/配置/构建脚本 | 不改 | 仅改当前工作单授权内容 |
| 读取实现、运行测试 | 可以 | 可以 |
| 临时审计探针 | 可以，仅隔离证据目录或 /tmp，不变成实现补丁 | 可以；正式回归测试由执行者落地 |
| 审计结论 | 由主线程独立形成 | 可以自测与解释，不能替主线程宣布验收通过 |
| 可维护文档 | 本 STATE 和审计主报告 | 实施记录、证据索引、获分派的文档 |

模型来自用户指定，不是自动选型建议。启动时核对实际工具参数/返回信息；若不能
设置指定模型或 effort，不得默默继承主模型、换模型或提高 effort。

主线程不因执行代理遇到困难而自己打实现补丁。应缩小任务、澄清契约、重新派工，
必要时以干净上下文更换执行代理。无需为普通实现决策再次向用户请求许可。

## 3. 项目目标和本次 Goal 的终点

长期目标：让 TileLang 中受支持的计算与调度意图，经编译过程准确变成 RKNPU 执行，
再据此调优。工作不以专用化或通用化本身为目标。

当前阶段：参考 GPU 后端“在 IR 上逐步转换”的方法，替代旧前端的
“整程序模板识别 → 提取参数 → 重建普通 GEMM”。不复制 GPU 的线程、warp、
shared memory 或异步执行模型。

当前 Goal **仅完成 guarded-tail FP16 GEMM 新路径的修复及验收**：

1. 关闭 §74，建立支持的容器/控制节点字段契约。
2. 新路径 direct 与 registered engine 不接受后忽略计算；不支持的结构准确拒绝。
3. 所有历史反例仍正确处理，合法变体、规范化、尾块和产物接口回归通过。
4. 结构确实参与 lowering；能解释每项要求由何步骤消费或为何拒绝。
5. 最终冻结版本通过主线程独立审计，测试及必要设备证据充分，限制如实记录。

完成此 Goal **不等于整个 adapter 已替换**。不自动扩大到其它算子族、切生产默认、
恢复性能搜索或继续权重复用实验；后续方向另由用户确定。

## 4. 当前工作区，不能丢失

仓库：`/home/maoyu/Documents/code/rockchip-npu-notes`

项目：`research/tilelang-rknpu`；C runtime：`research/custom-ops`。

接手 HEAD：`da4eeabd1c022a6e077bddfef4c36f2c46e6b11f`。
其上是大量未提交重构，包含未跟踪实现和测试。

关键冻结身份：

| 文件（相对项目） | SHA-256 |
|---|---|
| `python/rknpu_backend/frontend/rknpu_pass.py` | `7041e39fae7bff6030f1bcc69a4537ce2cbb9390d00f70301909acbc7cadc982` |
| `python/rknpu_backend/frontend/tilelang_backend.py` | `20a5e5ecb5db05f2e8790eb57f4b8bc05ffe51781ef0fcfcdeb6a0e64739f050` |
| `tests/test_rknpu_pass_pipeline.py` | `d624a5d57312256a77d58eca79d208923cc9fbafa9dafce72637d13d20011275` |
| `tests/test_tilelang_backend_registration.py` | `0d2a99b93e0ee3b622d344b9adb6a770eb967bc5a6147c690f0e7d3e2133b8a4` |

完整接手清单见 BASELINE.json。它是不可改写的接手快照；实现开始后用 STATE 和
各次冻结清单记录新身份，不要求实现后仍保持本表哈希。

**禁止** `git reset --hard`、整文件 checkout 回退、`git clean`、清理未跟踪文件，
以及为使测试全绿而覆盖旧实现。禁止 `git add .`、`git add -A` 混入未知改动。
小块编辑后检查 diff，避免用大段字符串替换改坏函数或缩进。

若换机器或 worktree，Git clone 不能带走未提交文件及被忽略的 results/、模型、
依赖、host 库。必须另行可核对地迁移本任务内容，再核对身份；不得把缺文件的新
checkout 当作真实基线。当前应优先使用同一工作区。

## 5. 已实现的路径与现有限制

主要路径：

```text
TileLang PrimFunc
  → BindTarget + MaterializeKernelLaunch(lower_thread_binding=False)
  → pass_localise：证明并在 IR 内改写局部 tile op
  → pass_instantiate：静态展开为 rk_* 调用序列
  → tasks_from_module / instances_from_tasks
  → compile_from_tasks / build_from_tasks
  → 复用现有存储、提交、记录与 verifier
  → RKPlan → 既有 C verifier/runtime
```

- `rknpu_pass.py` 是新试点，不调用旧 extractor。
- `tilelang_backend.py` 显式注册 BackendModule/PassPipeline，使用 ext_dev kind
  加 rknpu key。当前正式 engine 能返回携带计划 JSON 的 CompiledArtifact。
- 产物随返回对象传递；全局 lowering log 只供诊断，不应作为计划读取依赖。
- 当前构建通过有记录的 IRModule.inspect_source shim 对接产物。适用版本有限，
  不应宣称其它 backend 已全面兼容。
- `structured.py` 是前一轮原型，保留但非新路径 fallback。
- 旧 `tilelang_adapter.py` 与生产入口保留。
- 仍复用 planner，负责 layout、BO、partial、submit、publish；复用它本身不是缺陷。
  但不得用其默认 GEMM 规则抹去尚未证明等价的源操作或控制条件。

支持的主实验：SmolLM2-135M gate/up，K=576、N=3072，prefill 实际 M=256、
decode M=1；同时有小型与 M/N/K tail 正负例。primitive tile ≤256，K slice=256
加 tail，逐 slice FP16 writeback。新路径不是通用 GPU kernel 编译器。

## 6. 首要阻断项：§74 的实际复现

相关源码：`rknpu_pass.py::pass_instantiate`，当前约 1549 行：

```python
if kind == "SBlockRealize":
    expand(node.block, env)
```

没有消费 predicate；前置分析也没有将其作为 path condition 或要求其恒真。

真实 TileLang 源 `results/codex-pass-73-audit/never_execute.py`：

```python
with T.Kernel(1, 1, threads=128) as (im, inn):
    T.where(False)
    # 正常 guarded-tail GEMM body
```

IR 中 tilelang_root 的 SBlockRealize.predicate=False。source 不应写外部 Y；
初始化 Y=-7 后应保持 -7。当前 direct 和标准 engine 都接受，生成正常 GEMM，
mock 将全部 65536 个元素变成 576。

修复不要求支持任意动态条件或空 Plan：可以准确拒绝无法承接的 block 条件，
也可正规化后真正消费条件并验证输出覆盖。不能只识别文本 `T.where(False)`。

同次清点以下容器字段的处理契约：

| 节点/字段 | 要核实的内容 |
|---|---|
| SBlockRealize.predicate | enclosing domain 上恒真、被消费，或拒绝 |
| SBlockRealize.iter_values / SBlock.iter_vars | 绑定是否影响索引；未支持则拒绝 |
| SBlock.init | 实际执行效果；不能只遍历 body |
| SBlock.alloc_buffers / match_buffers | 分配与 alias/view 的解释或限制 |
| Allocate/AttrStmt/Bind 等 | 控制或存储字段是否有意义；不能无依据透明穿透 |

除 predicate 外，其它字段是**待调查项，不是已确认缺陷**。本任务不要求全部
实现；准确拒绝未支持语义是合法结果。对提示和纯元数据可以记录未采用，不能
据此忽略有计算效果的字段。

## 7. 已修复但必须防止回归的类别

按最新审核结论，不按早期“通过”叙述判断：

- 显式 M/N 要求不能无痕丢失，未知 rknpu.* 准确拒绝。
- 真实 loop domain；输入/输出块坐标对应；完整输出与 K partition。
- 完整路径条件；copy 的局部写入覆盖；额外 fill/copy 不得消失。
- 源参数类型/stride，不得把 FP32 签名静默变成 FP16。
- GEMM 实际结果 buffer 与 accumulator load/writeback 的完整连接。
- tile op 精确表，未知操作不得降成 copy；全部 fill/copy/GEMM region 起点与范围。
- hint 的请求/未采用原因随计划保存、加载并可 explain。
- 历史 RKPlan 提交顺序、MM_ACC/EW/LUT 几何、View、预算等安全修复。

没有新反例不等于任意程序已被证明正确；但不能因此无限审计不结束。
终点是明确的支持契约、已知阻断项关闭、相应测试和独立冻结审计成立。

## 8. 命令与证据，先快后全

从项目目录运行；先 import tilelang 再 import tvm。已核实依赖为 TileLang 0.1.15、
TVM 0.25.dev0，实际文件均在 `.pydeps/tilelang` 及其 3rdparty/tvm 下。

```bash
cd /home/maoyu/Documents/code/rockchip-npu-notes/research/tilelang-rknpu
git status --short
git diff --check

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/test_rknpu_pass_pipeline.py \
  -k 'UnsupportedOperationTests or RegionContractTests' -q -o addopts=""

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/test_rknpu_pass_pipeline.py tests/test_tilelang_backend_registration.py \
  -q -o addopts=""

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests -q -o addopts=""
ctest --test-dir ../custom-ops/build-host-mvp --output-on-failure -j2
```

host 库：`../custom-ops/build-host-mvp/librknpu_plan.so` 与
`librknpu_plan_mock.so`。mock 不等于设备。CTest 使用已有 build 不等于重新构建。

完整回归约 10 分钟，不在每次小编辑后重跑；先针对性验证，候选冻结后完整运行，
新增改动/失败/未解决疑点才重复。编译 API 的正向测试也可能耗时较长。

最近执行者报告：874 passed / 2 skipped / 301 subtests；主线程独立复跑的最近
子集为 5 passed / 59 deselected。两者不能混称。
skip 分别关联 transformers 和 `RKNPU_SMOLLM_SLOW_TESTS=1`，不等于这些检查已执行。

关键历史审核目录：

```text
results/codex-pass-review/
results/codex-pass-recheck-64/
results/codex-current-worktree-acceptance/
results/codex-frozen-review/
results/codex-pass-69-audit/
results/codex-pass-71-audit/
results/codex-pass-73-audit/       # 最新已确认阻断项
```

脚本可能在自身目录写 JSON。**不要直接运行原路径后尝试恢复历史证据。**
有 --out 时指定新目录；没有时复制脚本及相对依赖的 fixture 到新的隔离目录，
从项目 cwd 用相同 PYTHONPATH 运行，确认复制字节一致。
例如复制最新 §74 的两个必要文件后执行：

```bash
audit_dir=$(mktemp -d "$PWD/results/handoff-audit-XXXXXX")
cp results/codex-pass-73-audit/reproduce.py "$audit_dir/"
cp results/codex-pass-73-audit/never_execute.py "$audit_dir/"
PYTHONPATH=$PWD:$PWD/python .venv/bin/python "$audit_dir/reproduce.py"
```

不要改脚本使错误结果变绿，不将 plan hash 当作语义证明，不改写旧 FAIL 或容差。
原证据和依赖被 Git 忽略；新机器不一定存在，应先验证，不伪称可重放。

## 9. 主线程的持续派工与定期审计

状态循环：

```text
核实基线 → 发一个具体工作单 → 执行 → 稳定冻结 → 主线程独立审核
                                      ↑                 ↓
                                      └── 带证据退回 ───┘
通过当前单元 → 更新 STATE → 下一个已授权单元 → 最终冻结验收 → 当前 Goal 完成
```

默认一个执行代理。可由执行代理按工作单申请/使用并行实现助手；不同文件可并行，
同一文件和设备必须有单一 owner。不得再委派最终审计。

审计频率是**每个稳定工作单结束时必审**，长工作约每 20–30 分钟要求一次稳定
检查点。它不是后台定时器：依赖当前 Goal 轮次和代理消息主动驱动，不能保证
客户端退出、暂停或预算耗尽后仍继续。执行者发现关键契约变化应提前报告。

主线程不得在执行者仍修改实现时审核“当前代码”。先要求进入 READY_FOR_AUDIT，
停止写源码及相关后台任务、记录 SHA；主线程核对审核前后身份，变动则本次审核无效。

每次审核至少：看实际 diff → 查与任务契约对应的处理路径 → 原反例 → 正例与相邻
反例 → 适当测试 → 明确结论。实质新问题要附最小复现；没有复现的推断单列，
不能把猜测当 blocker，也不能因执行者说全绿而直接通过。

审计也不能制造错误门槛：不要求原始节点永远保留；有依据的静态展开、条件消除、
buffer 重布局和等价合并合法。复用 planner、两份等价源生成同一计划、记录导致
hash 改变都不自动构成问题。真正要看源效果由什么承接。未观测设备内部时间序、
未实现多核等本阶段非目标，不作为无限追加任务的理由。

若同类问题连续两次退回，主线程应缩小并重写该工作单，要求语义字段清单和
消费对应表，而非继续“再补这个 fixture”。若连续三次没有实质进展，可在
保全工作区、结果和停止旧任务后，用新的 Luna low 执行代理接手，不更换模型。

用实际工具的 spawn/followup/message/wait 驱动。优先复用健康执行会话；更换时
不给整段历史，用本包、STATE、当前工作单和必要证据建立干净上下文。
等待使用事件/工具，不忙轮询；单次阻塞等待不超过 60 秒，以便响应用户。

## 10. 工作单与交付必须可审核

每张工作单包含：ID、前置身份、问题证据、允许改动、禁止项、预期输入/输出、
正反例、命令、验收标准、证据目录、何时冻结。模板见 EXECUTOR-START。

执行者返回：修改文件与 SHA、实际原因、实现规则、前后对照、测试退出码与原始
日志、未做事项、证据位置。禁止只返回“已完成”或“所有问题修好”。

主线程审核结果只取：PASS、CHANGES_REQUIRED、BLOCKED_EXTERNAL、NOT_REVIEWED。
明确标注“单元通过”与“整个 Goal 完成”的区别。失败时直接回派可操作工作单，
不把每轮失败转成用户许可问题。

若提供 get_goal/create_goal/update_goal：先读当前 Goal，已有则复用；只有用户
明确设置目标时才创建，不给执行代理再建共享 Goal。未指定 token budget 就不
擅自设置。工具对 blocked 的连续轮次、paused、预算等限制优先，不能用任务状态
文件绕过客户端生命周期。

STATE 是协调入口，不是证明。实际源码和可重跑证据优先；压缩上下文后先读
STATE、对应工作单/审核记录，再检查文件身份，不重读全部 74 节。

## 11. 设备、权限与 Goal 停止条件

普通源码检查、修复、测试、构建和合法板测已在当前任务授权内。不得每一步
请求许可。改变数值政策、生产默认、扩大独立方向或不可恢复操作才询问。

设备最近执行者称无可用凭据，不能假定永久不可用。只通过既有配置和通道核实，
不打印/复制凭据、不绕过认证、不反复请求凭据。设备只由一个执行者使用。

只增加拒绝/记录，且合法执行字段与库确实未变时，可以逐项引用旧设备证据。
新执行结构或新接受语义需要相应设备验证；mock 不可充当真机证明。
设备不可用时先完成全部独立主机工作；确有未满足的设备门槛则保持未完成并
准确记录阻碍，不伪称完成，也不无限重跑相同探测。

只有下面条件全部满足，主线程才结束当前 Goal：

- §74 和新确认的实质问题关闭；可检查的支持契约成立；
- 新旧合法变体、规范化、direct/engine、产物 roundtrip 与历史负例通过；
- 冻结代码有适当完整回归、C/mock 和必要设备证据/可核对的执行等同性；
- 主线程独立审核通过，文档与实现一致；
- 证据已取回，STATE 更新，无遗留写入任务；明确剩余未支持项。

不因写完报告、测试数字变大、暂未发现新反例或预算将耗尽而标完成。
使用 get_goal/update_goal 等工具时服从当前工具说明；不重复创建已有 Goal。
只有用户明确要求才 paused；blocked 的重复次数要求以实际工具说明为准。
本阶段完成后，不自动创建下一目标或恢复性能/其它算子工作。

## 12. 对用户的表达

主线程 brief 用朴素中文，只讲当前结论、具体影响、验证和下一步；不强调代理
安排，不使用“闭环/范围内/有边界”等代替具体含义。不向用户倾倒工具日志。
执行代理交付、工作单和技术报告可以使用精确术语，不必简化。
用户的 btw 问题不自动取消当前 Goal；先简短回答，继续已授权工作，除非明确改向。

## 13. OpenAI Docs 与本机核实说明

本机检查为 Codex CLI 0.160.0，features 中 goals、multi_agent 已启用；未启动
模型调用来验证账号可用性。模型选择是用户要求，不能据此保证其它客户端可用。

- [Using Goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)：
  Goal 应有可验证终点，可用 `/goal` 管理，不是无限后台运行承诺。
- [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)：
  执行模型与 effort 可单独指定；以当前工具暴露的参数为准。
- [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)：
  主线程 CLI 的 model 与 model_reasoning_effort 配置依据。

本包未修改用户级 Codex 配置、权限、模型提供方或自动任务设置。
