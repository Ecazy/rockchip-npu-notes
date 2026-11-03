# 验收标准、方法与证据边界

本文件是**当前验收要求与方法**的唯一权威：什么算证据、每类证据能证明什么、以及不能声明什么。
命令见 [REPLAY.md](REPLAY.md)，支持域见 [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md)，
架构与契约见 [ARCHITECTURE.md](ARCHITECTURE.md)，数值政策见
[models/NUMERIC-ACCEPTANCE.md](models/NUMERIC-ACCEPTANCE.md)。

本文件不记录历史 milestone 的 PASS、逐轮审计或历史计数；那些只作为 `results/` 下的原始证据存在，
其适用范围见 §4。

## 1. 状态术语

判断任一能力时只用这五个词，且必须带领域限定：

| 术语 | 含义 |
|---|---|
| **implemented + verified** | 当前代码实现了该行为，且当前测试/工具在当前代码上验证过 |
| **verification gap** | 已实现，但当前没有能证明它的验证（例如缺设备或缺口径） |
| **explicitly unsupported** | 入口明确拒绝，且拒绝有测试覆盖 |
| **out of scope** | 不属于本后端职责（例如故障后恢复、多板） |
| **future extension** | 有意留待扩展，当前不接受也不拒绝该形态 |

历史文档中的完成度断言（"已完成""PASS"）不构成当前判定；以当前代码与当前测试为准。

## 2. 证据类别与各自能证明什么

| 类别 | 能证明 | 不能证明 |
|---|---|---|
| **Python 单元/集成测试**（`pytest tests/`） | 当前代码在测试覆盖的输入上按断言行为执行；拒绝路径的拒绝码与 stage | 未覆盖输入；设备行为；数值包络 |
| **独立 C 检查**（`custom-ops` 的 `rk_plan_contract_test`、host plan 库 CTest） | C ABI/descriptor 契约与 host 侧计划校验对当前计划成立 | 编译器的语义正确性；NPU 执行 |
| **C mock 执行**（`librknpu_plan_mock.so` + `PlanCall`） | 生成的计划在**同一 C 执行路径**上算出的数值，可用于语义反例（错误接受 → 数值不符） | 真实 NPU 数值与性能；不覆盖硬件特化路径 |
| **板端 admission / campaign** | 那一次、那个代码快照、那些计划在 RK3588 上的数值与资源行为 | 之后任何代码或计划变更后的行为 |
| **独立数值 verifier** | 冻结模型包在冻结输入上的 per-layer/per-profile 门槛 | 其它几何、其它模型、其它数值政策 |
| **计划字段等价**（新旧计划逐字段比较） | 两个计划的执行字段是否相同（tile/K 切片、submit、publish、字节、BO、上传下载） | 计划的语义正确性；不检查的 attrs 差异 |

判定语义正确性时，按问题选择证据：是否被拒、拒得对不对用测试；生成计划的数值是否与源一致用
mock 或板端执行；两个计划是否同一条执行路径用字段对照。单一证据只回答它覆盖的那部分——
任务数、矩形面积或 buffer shape 不能单独证明索引或读写正确。

## 3. 验证方法与适用条件

这一节说明**怎么验证**，不新增验收门。任一能力是否算"已验证"，按 §1 的术语和 §2 的证据类别判定；
具体到某项改动需要哪类证据，取决于改动影响的范围：

- **拒绝路径**：改动拒绝条件时，在对应测试或夹具中检查拒绝的*含义*（诊断码/消息指向的原因），
  而不是只断言"抛了异常"。既有规范只要求拒绝准确、可诊断，没有要求每个条件必须单列一条负例。
- **接受路径**：合法输入至少要能生成计划并通过与改动相关的检查；只改动不受影响的部分时，
  用既有回归即可。是否追加板端验证取决于改动是否改变执行字段：
  - 执行字段（任务几何、操作数、submit、publish、字节、BO、上传下载）不变时，引用既有板端
    执行作为"执行路径等价"证据，并在引用处写明等价判据；
  - 改变执行字段或新增正式入口时，需要代表性板端验证；无法执行时按 §1 记为 verification gap，
    不得用 mock 或字段等价替代板端结论。
- **语义正确性**：拒绝/接受测试、数值证据（mock 或板端）与字段对照各自只回答一部分问题
  （见 §2）。判定时说明用的是哪一类、覆盖了哪个输入集合；不要求每次判定都同时具备三类，
  但不得用其中一类去回答它不能回答的问题。
- **接口保持**：参数 dtype、stride、region、别名与数值政策必须与源一致或被明确拒绝；
  这是既有实现与测试已经在执行的约束，不是本轮新增要求。

## 4. 历史设备结果的适用范围

- 板端结果只对**当时执行的计划**有效。引用时必须同时给出：日期、代码/计划身份（`plan_hash` 或
  commit）、板卡与核配置、数值政策；缺任一项就不能作为证据。
- 当前计划与历史计划在**执行字段**上逐字段相同（差异仅为记录属性）时，可以引用该历史执行作为
  "执行路径等价"的证据，但不得表述为"当前 lowering 已通过设备验收"。
- 设备直证的具体身份见 [ARCHITECTURE.md](ARCHITECTURE.md) §9；等价性判据记录在
  `results/handoff-final-lead-audit-r4/device-equivalence-reviewed.json`。
- 旧设备结果的性能数字属于当时快照，不是当前默认路径的性能结论。

### 冻结输入文本源的适用范围

`models/smollm2_v4_input_plan.json` 与 `models/smollm2_v5_input_plan.json` 用文档正文作为
calibration/acceptance 文本窗口的来源（`make_smollm2_ppl_windows.py` 取每个源文件**开头**
`windows × 256` 个 token）。被声明的源包括 `SUPPORT-DOMAIN.md`、`RUNTIME-LIFECYCLE.md`
以及两份顶层报告 `GRAPH-FUSION-REPORT.md`、`LAYOUT-REUSE-REPORT.md`（另有一个外部模型 README）。
`RUNTIME-LIFECYCLE.md` 同时是当前 runtime 契约；两份顶层报告只作为**冻结输入材料**保留原文，
**不是当前技术说明**，导航处已如此标注。

- `SUPPORT-DOMAIN.md` 的当前正文与其作为冻结输入源时的版本不同，因此**从该文件重新派生**
  窗口文本可能与既有冻结 spec 不同。作为输入源的那一版身份为
  commit `da4eeabd1c022a6e077bddfef4c36f2c46e6b11f` 中的 `research/tilelang-rknpu/SUPPORT-DOMAIN.md`，
  其内容 SHA-256 为 `8f2a20229eb2d6e0f533a66a42b4408918e652d78d15e34f30aed6e0ec971620`。
- 冻结的 v5 spec、sealed observation bundles 与已冻结的窗口 id 是权威输入；既有验收不因文本变化
  而失效，也不据此重算任何哈希。重新派生必须显式重跑并重新冻结。

## 5. 非声明

- 数值验收是**经验包络**，不是理论误差保证；门槛由固定 calibration 子集冻结后在 disjoint
  acceptance 子集上观察，且明确披露"测试结果曾被查看"，不是盲测。
- 不声明穷举组合覆盖；边界以列出的代表性形状与规则检查为准。
- 不声明：任意 importer、动态 shape、3-D 网格、state/recurrent 图、device-resident KV cache、
  量化、并发请求、多板、故障后恢复、RK1228 目标、以及任何性能优势。
- 未在冻结 manifest 之外的输入上声称模型级精度。

## 6. 数值政策

唯一的数值政策权威是 [models/NUMERIC-ACCEPTANCE.md](models/NUMERIC-ACCEPTANCE.md)：
声明 FP16 路径、per-op `NumericContract`、per-layer/per-profile 冻结门槛与 calibration 规则。
本文件不复制门槛数值。

## 7. 证据组织

原始证据（campaign 目录、日志、探针、manifest、哈希清单、重放脚本）的组织与查找方法见
[results/README.md](results/README.md)。证据按日期目录原地保存，不因文档清理改写其内容或哈希。
