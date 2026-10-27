# Transformer 优化方案与逐轮历史

整理日期：2026-10-27。

历史材料归档，按原写作时点解释，不作为当前任务清单。各节保留原文件内容和阶段限定；不同日期的状态、测试数量与性能不能合并为当前结论。当前入口见 [模块首页](../README.md)，验收见 [ACCEPTANCE.md](../ACCEPTANCE.md)，最终优化结论见 [OPTIMIZATION-SUMMARY.md](../OPTIMIZATION-SUMMARY.md)。命令仍按原文约定的工作目录执行，不能直接从 archive 目录运行。

## 内容索引

- [原始优化方案与范围](#plan)（原 `TRANSFORMER-OPTIMIZATION-PLAN.md`）。
- [第 1 轮优化结果](#round-1)（原 `OPTIMIZATION-RESULTS-1.md`）。
- [第 2 轮优化结果](#round-2)（原 `OPTIMIZATION-RESULTS-2.md`）。
- [第 3 轮优化结果](#round-3)（原 `OPTIMIZATION-RESULTS-3.md`）。
- [第 4 轮优化结果](#round-4)（原 `OPTIMIZATION-RESULTS-4.md`）。
- [第 5 轮优化结果](#round-5)（原 `OPTIMIZATION-RESULTS-5.md`）。
- [第 6 轮优化结果](#round-6)（原 `OPTIMIZATION-RESULTS-6.md`）。
- [第 7 轮优化结果](#round-7)（原 `OPTIMIZATION-RESULTS-7.md`）。

---

<a id="plan"></a>

## 原始优化方案与范围

原文件：`TRANSFORMER-OPTIMIZATION-PLAN.md`。以下为该文件历史内容。

<a id="plan--transformer-后端优化方案分析稿"></a>
### Transformer 后端优化方案（分析稿）

原分析日期：2026-10-24；状态导航整理：2026-10-26。

本文保留方案 A–D 的设计理由、依赖与验收方法；正文中的成本表和预测按原分析时点解释。实际完成状态与数字统一查以下结果报告，避免将旧基线与新测量混用。

| 阶段 | 结果入口 | 已交付范围 |
|---|---|---|
| A 与 B1 | [OPTIMIZATION-RESULTS-1.md](#round-1) | 运行包络归因、静态准备、成组调用与预算闭环 |
| C | [OPTIMIZATION-RESULTS-2.md](#round-2) | 默认 packed gate/up + Host FP32 SwiGLU；device FFN 为独立数值政策的可选路径 |
| B2 产品接线与复验 | [OPTIMIZATION-RESULTS-2.md](#round-2) | packed 默认 FFN 上的 QK/PV shared group，separate/B1/chain_shared 对照及冻结政策复放 |
| D（RMSNorm 部分） | [OPTIMIZATION-RESULTS-3.md](#round-3) | 现有 RKPlan 无合法行归约/rsqrt 能力；`co_row_ops` 复用更慢；residual+RMSNorm Host 融合逐位等价但无实测收益，默认不变 |
| QK/PV 上传同步收口 | [OPTIMIZATION-RESULTS-4.md](#round-4) | shared group 同 BO 连续切片合并为一次 prep/fini；prep/fini calls 134→10 (b8)，默认 coalesced，保留逐切片开关 |
| 上传策略契约与布局快路径 | [OPTIMIZATION-RESULTS-5.md](#round-5) | >128 上传安全回退；策略创建时快照；仅独立 MM shared group 合并；16×16 CUBE/WEIGHT 打包快路径，b8 1.0195× |
| Python/ctypes binding 复用 | [OPTIMIZATION-RESULTS-6.md](#round-6) | PlanCall 缓存静态 binding 描述符，运行仅更新数据指针；b8 1.147× / b1 1.101×；下载同步合并因无稳定收益撤回 |
| QKV 输入布局映射 | [OPTIMIZATION-RESULTS-7.md](#round-7) | 显式 group_layout 让 QK/PV 直接读 QKV 列窗口；生产移除 split_heads 及 q/k/v 数组；b1 pooled 1.027×，b8 接近噪声边界但无退化 |
| 最终交付验收 | [OPTIMIZATION-SUMMARY.md](../OPTIMIZATION-SUMMARY.md) | 五个静态 profile 全部 10,000 张、3 轮同 campaign baseline/current 对照、packed 冻结政策 PASS；完整评估墙钟 224.2→148.2 s；明确非声明与下一阶段契约建议 |
| 厂商 RKNN 基线 | [results/20261027T015302Z-vendor/analysis.md](../results/20261027T015302Z-vendor/analysis.md) | 当前自研 / RKNN 2.3.2 / CPU 三类对照；厂商 warm 较当前快 2.4–3.0×，ViT per-logit 误差超出 packed 阈值，需独立数值政策 |

完整模型里程碑已独立关闭，见 [MODEL-MILESTONE-STATUS.md](../ACCEPTANCE.md#models)。候选方案本身不构成实施授权、能力声明或验收通过。新一轮工作按具体任务推进，不由本文的历史“下一步”措辞自动启动。

<a id="plan--1-优化目标与适用范围"></a>
#### 1. 优化目标与适用范围

近期对象是当前 MNIST Tiny Transformer：16 tokens、D=64、4 heads、单层 encoder、SwiGLU FFN=128；重点分析 batch=8，batch=1作为独立延迟场景。保持权重、数据、预处理、模型数学定义及明确的数值政策。

目标是降低整模型端到端时间与设备往返成本，同时把方法实现为可复用的后端能力：静态 repeat 成组、计划与缓冲复用、已证明的布局传递、设备子图驻留。不得通过识别模型名或固定尺寸写整模型捷径。

原则可推广到其他Transformer，具体收益和优先级不能推广：长序列prefill可能受attention计算/中间矩阵容量限制；decode可能受权重带宽、KV访问与逐token调度限制。本文不设计KV cache、长序列attention、量化、多核并发或任意模型导入。

<a id="plan--2-当前证据与计时解释"></a>
#### 2. 当前证据与计时解释

分析依据：

- [MODEL-MILESTONE-STATUS.md](../ACCEPTANCE.md#models)。
- `results/20261024T071144Z-models/model-results/core-1/model_eval.json`。
- `models/compiled/tiny_vit_b8.rkhybrid.json`。
- `python/rknpu_backend/hybrid.py`、`models.py`，以及 `research/custom-ops/src/custom_ops.c` 的现有执行路径。

以下数据来自现有记录，本文没有重新进行性能测量。

| 区间（10,000张图像） | 累计时间 | 占端到端 | 含义 |
|---|---:|---:|---|
| 端到端 | 50.724 s | 100% | 模型调用和后处理；参考计算在区间外 |
| 全部RKPlan节点 | 45.440 s | 89.6% | 外层节点解析、binding与native调用等完整停留时间 |
| 其中C workspace | 35.183 s | 69.4% | `rk_workspace_run`；不是纯硬件计算时间 |
| 已注册Host action | 2.993 s | 5.9% | 含多个阶段，不能与RKPlan内嵌时间重复计算 |
| executor调度开销 | 2.228 s | 4.4% | 调度器记录的额外包络 |
| 单线程CPU FP32端到端 | 7.308 s | 比较项 | 相同batch/tail策略；不同数值执行路径 |

这些区间有嵌套，不能逐行相加。图内统计的未归属时间为0，不代表C workspace内各个成本都已分别解释，也不代表图外循环、logits拼接等开销不存在。

<a id="plan--21-主要瓶颈在逐head的细碎执行"></a>
##### 2.1 主要瓶颈在逐head的细碎执行

每个batch=8有32个 `(batch, head)` 对。当前每对分别调用一次QK和一次PV，另外还有7次常规矩阵节点调用：

```text
每batch：32 QK + 32 PV + 7 = 71次提交
完整数据集：1,250个batch × 71 = 88,750次提交
```

| 节点 | 调用数 | RKPlan节点时间 | C workspace时间 | driver submit/wait字段 |
|---|---:|---:|---:|---:|
| QK | 40,000 | 19.304 s | 14.585 s | 2.749 s |
| PV | 40,000 | 18.773 s | 14.248 s | 2.628 s |
| 合计 | 80,000 | 38.077 s | 28.833 s | 5.377 s |

QK/PV节点占端到端约75.1%。两节点的外层包络与C workspace相差约9.24秒；workspace扣除已有pack/emit/submit-wait/output-sync字段后仍有约21.87秒未单列。不能直接把这部分归因为NPU运算或驱动提交。

待辨别的来源包括：binding/有限值检查、Host staging分配与拼装、BO slack/guard扫描、cache prep/fini、Python/ctypes与view解析等。这里只列可疑来源，不认定哪一项已经是根因。

<a id="plan--22-搬运是高频边界问题"></a>
##### 2.2 搬运是高频边界问题

当前上传391.04 MB、下载287.68 MB；QK/PV各自的上传+下载约163.84 MB，合计占设备传输字节约48.3%。这些是运行期报告口径，不能当成物理DDR带宽测量。

大量小矩阵每次都经历准备、同步、提交、下载和再次调用。不能简单用总字节数除以带宽预测收益；Host tensor materialization字节也不能再次累加为设备DMA字节。

<a id="plan--23-两个约束性上界"></a>
##### 2.3 两个约束性上界

- 若只把QK/PV整个节点时间理想地降到0，其余部分保持不变，端到端仍约12.65秒，约791 img/s，仍慢于本次CPU参考。这个上界说明后续还需优化其他路径，不是可实现预测。
- 若只消除全部2.99秒Host action时间、数据流和其他成本不变，整体最多约1.063倍。因此“先写一个NPU Softmax就能解决问题”没有当前证据支持。消除Host边界若同时减少传输，收益才可能更大。

<a id="plan--3-推荐总体路线"></a>
#### 3. 推荐总体路线

```text
固定正确基线与成本归属
  → repeat解析/binding/分配的静态复用
  → QK/PV按batch-head成组执行
  → 去除可合法连接的设备子图边界
  → 再评估norm/Softmax下沉或其他placement
```

前两项直接针对调用次数与运行包络，优先级高于单独替换Softmax算术。每阶段使用同一个模型和冻结正确性政策，不引入新模型来替换基线。

<a id="plan--4-方案a减轻已知语义不变的运行包络"></a>
#### 4. 方案A：减轻已知语义不变的运行包络

<a id="plan--a1-细分c-workspace成本"></a>
##### A1. 细分C workspace成本

后续若获准实施，先在少量固定batch的隔离分析中区分binding/finite、upload prep/pack/fini、command处理、submit/wait、output sync、download/unpack、guard、Host分配/发布。可复用已有计时与调用计数，不另建长期profiling框架。

详细插桩可能扰动小任务，因此定位数据与无插桩端到端数据分开报告。停止条件是能够解释QK/PV的主要包络，并选出实际可减少的成本；不能以不断提高仪器精度取代优化交付。

<a id="plan--a2-编译静态执行描述预分配暂存"></a>
##### A2. 编译静态执行描述，预分配暂存

- 在HybridRunner创建时解析输入输出binding顺序、repeat索引、slice描述、dtype/shape约束和不可变plan身份。
- 预分配可复用的输入整理、head输出、C staging和结果缓冲；多次run仅更新内容与必要指针。
- 每次run的输入shape/dtype、有限值、alias、context fault及输出发布条件仍须检查；只有不可变结构验证才能移出热路径。
- 复用仍须服从workspace busy、Host数组保活和故障隔离，不把临时view悬空或输出暴露给下一次run覆盖。

<a id="plan--a3-缓存已绑定程序与合法chain准备"></a>
##### A3. 缓存已绑定程序与合法chain准备

若view、BO、fd/domain、模板、shape、数值政策和提交顺序不变，数据内容更新通常不改变寄存器地址。可研究workspace内缓存已发射program或预构造Host命令镜像。

缓存必须覆盖地址/布局/版本失效。当前command BO会被不同submit组复用，不能直接让多个组共用一份已覆盖的设备命令；若缓存完整设备命令，应独立规划区域与预算。继续复用已有pack/verify/终端封口逻辑，不复制提交器。

<a id="plan--a4-guard与同步不能以关掉方式优化"></a>
##### A4. guard与同步不能以“关掉”方式优化

当前每个小plan的最终发布门禁都可能检查多个BO。应先确定实际成本。如果将32个head纳入一个workspace，可在覆盖全部任务的发布边界统一检查，减少重复调用，同时保留全部必要检查。

不能在没有覆盖范围证明和失败回归时删除guard、跳过cache同步，或把设备故障转成成功。首轮保持检查强度，后续若区分诊断/生产模式应独立定义契约。

<a id="plan--5-方案bqkpv的batch-head成组主要结构优化"></a>
#### 5. 方案B：QK/PV的batch-head成组（主要结构优化）

<a id="plan--b1-以静态repeat为单元构建批量plan"></a>
##### B1. 以静态repeat为单元构建批量plan

以32个独立head矩阵任务组成一个QK组，完成后批量下载scores、执行现有Host Softmax，再以32个PV任务组成一组。一次Python/native入口处理一整组，不能只在Python中继续循环32次原单任务调用。

每个head仍有独立的正确输入、weight packing、输出区域和依赖。首轮保留现有physical布局、补齐与计算配方；不把不同行列的head直接拼成一次普通GEMM，以免算出跨head交叉项。block-diagonal GEMM会引入大量无效计算，不作为首选。

<a id="plan--b2-区分合并入口与合并硬件提交"></a>
##### B2. 区分“合并入口”与“合并硬件提交”

- 可以先用一个native组入口执行32次已支持的独立Submit，降低Python/ctypes/分配开销；明确报告Submit数没有减少。
- 把32个MM放进一次chain，需要新增独立MM组的合法配方与板端证据。任务无依赖、flags相同都不是充分证明；不能绕过白名单或复用K-acc链名义宣称安全。
- 当前每组最多64 tasks，batch=8/head=4的32任务组在数量上可容纳，但仍受BO独立、命令容量、内存预算、布局与completion约束。
- 新配方未验证时保持显式separate路径；验证失败精确拒绝/选择已有合法分组，不试投危险寄存器组合。

<a id="plan--b3-有条件的计数目标"></a>
##### B3. 有条件的计数目标

只有“每个QK组和PV组均合法地一次Submit”成立时，原有其他7个矩阵任务不变：

```text
每batch提交：71 → 9
全测试集提交：88,750 → 11,250（减少约87.3%）
primitive任务数初期仍为88,750，不是任务数减少87.3%。
```

这只是结构计数目标，不预测相同比例的延迟下降。一般静态B/H应按合法组容量分块，不能把batch=8/head=4写进通用后端规则。

<a id="plan--b4-避免为少数head增加不必要的驻留内存"></a>
##### B4. 避免为少数head增加不必要的驻留内存

首轮可使用每head独立BO以保持已知alias契约；组workspace会增加命令与数据预留，必须计入slack/guard和Host staging。后续只有证明view、独立BO及驱动要求均满足，才考虑合并分配。

组内任何失败都不发布部分head，context quarantine与排队提交隔离继续有效。Host Softmax必须等待全体scores可见，PV必须等待全部所需概率和V已准备。

<a id="plan--6-方案c去除设备子图之间的往返"></a>
#### 6. 方案C：去除设备子图之间的往返

按实际成本选择，推荐顺序如下。

| 候选 | 可复用的后端能力 | 主要限制 |
|---|---|---|
| gate/up共输入的打包GEMM | 把两个独立常量权重沿输出轴组合，产出后分成两个逻辑值 | 当前QKV已经合并，不重复优化；gate/up拼接需保持输出写回、合法N分片和split布局，不假设任意Cube切片可用 |
| FFN gate/up → SwiGLU → down → residual设备子图 | 复用已支持的MatMul/Mul/Add/LUT与存储规划 | 当前Host SwiGLU未必与LUT gate先FP16写回的数值路径等价；新增近似/收窄必须显式版本化并另行验收 |
| PV → head merge → output projection | 设备布局转换/消费布局融合 | 先证明head轴到模型通道的索引映射；不能只改shape/stride就认为转换已完成 |
| residual + RMSNorm → QKV或FFN | 行归约、运行时系数与后继输入驻留 | 几何、finite/padding、均方与rsqrt的精度和输入域需独立证明；不能把现有Host复合实现当不透明纯NPU节点 |
| position add、残差等EW进入已有设备阶段 | 减少一次materialize或边界 | 只在实际消除搬运/提交时有价值；为单个廉价Add新增Submit可能更慢 |

先选择能够明确减少一次upload/download或plan调用的连接，不要求“一次全部融合”。保留Hybrid清单解释优化前后节点、边界、传输和数值政策。

<a id="plan--7-方案dnormsoftmax下沉与其他placement"></a>
#### 7. 方案D：norm/Softmax下沉与其他placement

Host Softmax本轮约0.464秒；两次Host RMSNorm合计约0.584秒。单独加速这些算术的整体上限有限，优先理由应是能够打通设备驻留，消除更大的桥接包络。

1. 在B/C有结果后，重新判断scores/probs桥接是否成为主要成本。
2. 若研究NPU稳定Softmax，必须实现max平移、exp值域、sum/归一化、padding/finite与各head独立性；现有bounded exp LUT不能直接代表通用稳定Softmax。
3. 若研究RMSNorm，优先复用已有算法与证据，明确内部Host步骤是否仍存在；仅搬动节点名称不算减少Host参与。
4. 可另设显式placement对照：很小的QK/PV留CPU是否更快。它必须作为不同且可见的hybrid政策，经后续授权再研究；不能作为当前“全部矩阵上NPU”里程碑路径的静默fallback，也不能把CPU收益算成NPU优化收益。

<a id="plan--8-数值与安全不变量"></a>
#### 8. 数值与安全不变量

- 基线权重、数据、预处理、batch/tail与数值契约固定。独立校准集先确定政策，验收数据不参与反向扩大容差；不能以top-1不变取代逐层/logits检查。
- repeat成组、布局传递与缓存优先保持相同计算/收窄语义；仍须验证实际模板几何变化是否改变结果。
- 会改变FP16写回位置、LUT近似或归约顺序的融合，使用独立数值政策与结果列，不能继承旧路径的bit-exact结论。
- 布局必须证明logical→physical映射、有效区域与padding；输入权重/动态B更新不可使用过期内容。
- 不能削弱fd/domain一致性、alias、预算、busy、最终发布门禁或shared-context quarantine。失败组不能部分发布，未确认静止的资源不可释放/复用。
- 不以多核并发或危险timeout制造收益/证据；首轮继续单个固定核。

<a id="plan--9-后续若实施时的有限比较方法"></a>
#### 9. 后续若实施时的有限比较方法

本文不执行以下实验，它们是未来实施的比较设计。

| 版本 | 唯一主要变化 | 需要回答的问题 |
|---|---|---|
| Baseline | 当前23节点hybrid与冻结模型 | 在相同环境下原报告能否作为比较基准 |
| A | 静态解析/预分配/程序准备复用 | 在Submit数不变时，Python和C workspace包络分别减少多少 |
| B1 | repeat批量入口，保留单任务Submit | 入口合并与device batching的收益能否区分 |
| B2 | 已验收的独立MM组chain | Submit数是否真正降低，数值/guard/失败发布是否保持 |
| C | 选择一个设备子图连接 | 是否实际减少桥接、总时间和内存，而不只减少Python代码 |
| D | 仅在新瓶颈明确后选择norm/Softmax或placement | 算术下沉是否带来端到端收益，还是增加小任务成本 |

比较固定batch=8的吞吐和batch=1的延迟，分别报告，不用改变batch掩盖算法差异。CPU与NPU使用相同数据顺序、batch/tail、warm-up和计时边界；设置并记录CPU线程、亲和性、频率及NPU频率。CPU FP32与NPU FP16/hybrid的精度差异保持可见。

先用有限的确定性输入确认正确性和调用计数，再对候选重复测量，至少记录每次结果及中位数/波动；最终候选才复跑完整任务准确率。长时间压力、其他模型或其他板卡不是首轮优化关卡。

同时记录：完整end-to-end、各phase、外层plan节点、C workspace、guard/cache/staging等新增归属、Host action、tasks/submits、upload/download调用及字节、Host materialization、创建时间与峰值内存。时间分项注明嵌套和未测项，不把driver服务包络当成纯计算。

验收以正确性、安全不退化且端到端收益可复现为准。不设“必须快于CPU”或“必须提升固定倍数”的承诺。若只减少了静态计数而延迟不改善，记录无收益并回退候选；不要继续扩张框架来强行证明原假设。

<a id="plan--10-推荐的首个优化包与停止条件"></a>
#### 10. 推荐的首个优化包与停止条件

首个优化包建议限定为：**解释QK/PV运行包络 + 静态repeat批量入口 + 有边界的独立MM组成组验证**。它最直接覆盖当前75%的节点耗时，且不要求先实现新的Softmax算法或纯NPU Transformer。

首包结束时应能回答四个问题：

1. QK/PV约21.87秒的C workspace未细分包络，主要由哪些实际操作构成？
2. 不减少Submit，只减少重复解析/分配/入口调用，能得到多少真实收益？
3. 32个head任务能否通过已验证配方合法成组；若不能，明确阻断，而不是绕过白名单？
4. 形成的能力是否对静态B/H repeat的一类图生效，而不是只识别本模型？

满足首包问题后，再按新的成本排名选一个C/D候选。若正确性、安全或稳定性不能保持，停止该候选，不变更模型或扩大数值门槛换取速度。

本方案提出的是可供后续选择的路线，不包含已实现、已测得或已承诺的加速结论。

---

<a id="round-1"></a>

## 第 1 轮优化结果

原文件：`OPTIMIZATION-RESULTS-1.md`。以下为该文件历史内容。

<a id="round-1--第一个优化包桥接包络归属--静态-repeat-成组实施报告-v6"></a>
### 第一个优化包：桥接包络归属 + 静态 repeat 成组（实施报告 v6）

日期：2026-10-26（v6：完成 QK/PV 接线、数值复验与三臂端到端 campaign）。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](#plan) 第 10 节“首个优化包”。

范围严格限定在方案稿第 10 节的首包三项——解释 QK/PV 运行包络、静态 repeat 批量入口、
有边界的成组验证——以及由其成本排名直接导出的 Host 侧瘦身。不加 Softmax/RMSNorm 新算法、
不做量化、不改模型、不改权重、不改数值政策、不做多核并发、不新增 chain 配方。

**四轮复核意见已逐条处理**（§6）。v5 修复了 v4 探针的最后一处比较器错误（所有任务都与
最后一个任务的参考值比较），并补了无设备验证；修正后 **B2 的所有形状重新通过**，因此 v4 的
“设备侧阻塞/输入重定向不可用”结论同样**撤回**。同时完成 Host 预算闭环：planner 与 C 校验器
公式计入运行时常驻 scratch、创建时检查、重新生成 plan 包，并按**原有冻结数值政策**重新验收
（逐项容差与 v1 **完全相同**，仅哈希随包更新）。

<a id="round-1--0-结论摘要"></a>
#### 0. 结论摘要

RK3588 core 1，CPU 绑核，三臂**逐轮交替**测量 4 轮 × 32 次，按臂合并 128 个样本。同一
payload（plans/weights/data 同一 SHA-256）：`baseline` = HEAD 源树，`candidate` = 工作树，
`noprobe` = 工作树同一份源码但**编译期去掉计时探针**的对照库。

| profile | baseline（合并中位）| candidate | 比值 | 吞吐（由合并中位换算）| candidate / noprobe |
|---|---:|---:|---:|---:|---:|
| Tiny Transformer b8 | 33.844 ms | **14.768 ms** | **2.292×** | 236.4 → 541.7 img/s | 14.768 / 14.870（−0.7%）|
| Tiny Transformer b1 | 7.199 ms | **4.139 ms** | **1.739×** | 138.9 → 241.6 img/s | 4.139 / 4.171（−0.8%）|
| MNIST MLP b32 | 2.832 ms | **2.198 ms** | **1.289×** | 11 298 → 14 559 img/s | 2.198 / 2.285（−3.8%）|

- 吞吐一律由**合并中位延迟**换算（`batch / p50`），不取单轮字段。
- 逐轮比值（baseline / candidate）b8：`2.36, 2.36, 2.15, 2.32`；b1：`1.75, 2.37, 1.72, 1.64`；
  MLP：`1.33, 1.26, 1.13, 1.47`。
- **task / submit / 上传 / 下载数量与字节逐臂完全相同**（b8：71/71、135 上传、71 下载、
  312 832 B 上、230 144 B 下），由比较工具逐字段断言。
- b8 包络未归属残差 1.0%（0.089 ms / 8.616 ms，**单次代表运行**，口径见 §3.3）。
- 数值：本轮 plan 包（accounting 修正后重新生成）按**原有冻结政策**重新验收，
  `results/20261026T053818Z-numeric/` **PASS / 0 problems**，且逐项容差与 v1 相同。
- 探针代价：candidate 与 noprobe 的差在 ±4% 内（MLP 上 noprobe 反而更慢），不作为“零代价”宣称。

<a id="round-1--01-静态-repeat-组已接入-qkpvv6"></a>
##### 0.1 静态 repeat 组已接入 QK/PV（v6）

QK/PV 现在以**一个预展开 plan + 一次链式提交**执行，由清单显式声明
（`repeat_submit: shared_group` + `group_plan: attn_matmul_group`），`separate` 与 B1
两条路径保留为 runner 控制项，并有测试断言三者 logits 逐位相同。

三臂端到端 campaign（`results/20261026T081034Z-policy/`，逐轮交替、绑核、同输入/核/计时边界）：

| profile | b1 | separate | **manifest（shared group）** | 相对 b1 |
|---|---:|---:|---:|---:|
| Tiny Transformer b8 | 14.940 ms | 22.622 ms | **12.350 ms** | **1.210×** |
| Tiny Transformer b1 | 4.143 ms | 4.904 ms | **3.842 ms** | **1.078×** |

- b8 四轮比值 `1.20 / 1.26 / 1.20 / 1.22`；吞吐 535.5 → **647.7 img/s**。
- 设备结构：`submit_count` **71 → 9**，`task_count` 71 不变，上传/下载次数与字节不变
  （比较工具把 submit 数变化标为 `structure DIFFERS`，这是本次优化的目的，不是退化）。
- C 侧停留（单次代表运行，第 1 轮）：**8.522 → 5.207 ms**，其中 `submit_wait`
  **5.008 → 1.746 ms**；`guard` 基本不变（0.321 → 0.293 ms），因为共享布局把每组压到 3 个 BO。
- 相对优化前基线（33.556 ms，见 §3.2 的三臂 campaign）：**2.72×**。注意两个数字来自不同
  campaign（基线与 B1 对比在 §3.2，B1 与 shared 对比在 §0.1），各自同轮交替、同板同核。

**数值复验**（`results/20261026T081153Z-numeric/`）：plan 包发生变化后按**原有冻结政策**
重新验收，`layer_tolerances` 与 `task_logit_tolerances` 与 v1 **逐项相同**，验证
**PASS / 0 problems**；门禁包含 `tiny_vit.scores_heads`（QK 输出）与
`tiny_vit.attn_heads`（PV 输出），即 QK/PV 节点输出在真实模型输入下被直接比较。

**默认策略建议**：QK/PV 保持 `shared_group`。它对两条对照路径逐位一致、通过冻结数值门禁、
端到端比 B1 快 1.21×；如需回退，`HybridRunner(repeat_submit="b1"|"separate")` 一行即可，
无需改清单。

<a id="round-1--1-变更清单"></a>
#### 1. 变更清单

每项标注对应方案条款；没有任何一项依赖模型名或固定尺寸。

<a id="round-1--a1--c-侧包络归因方案-4-a1"></a>
##### A1 — C 侧包络归因（方案 §4 A1）

- `rk_run_report` **冻结为原来的 15 字段 / 88 字节**，`rk_workspace_run` 不读 `struct_size`，
  写入边界与扩展前逐字节相同。
- 扩展归因走**新结构 + 新入口**：`rk_run_attribution` + `rk_workspace_run_ex`；`struct_size`
  强制校验（不在 `[V1_SIZE, sizeof]` 内直接 `RK_E_SCHEMA`，不写一字节）。
- `total_us`（入口自身 C 侧停留）与 `clock_us`（探针自身开销）是自证用的见证量。
- `-DRK_PLAN_PROBE_OFF` 用同一份源码编译出探针被编译掉的库，作为端到端对照臂。

<a id="round-1--a2--静态解析预分配与-host-侧瘦身方案-4-a2"></a>
##### A2 — 静态解析、预分配与 Host 侧瘦身（方案 §4 A2）

- C：workspace 创建时按不可变 plan 预分配 `co_program` 发射数组、输出 tile 计数、staging
  arena 与偏移表；**下载 tile 改用 workspace 常驻 scratch**（此前每次 download 一次
  `malloc/free`）。
- Python：`HybridRunner` 创建时把每个 RKPlan 节点的 binding 顺序、repeat 索引、slice 目标
  形状、首写分配规则解析成 `_NpuBindingPlan`/`_RepeatRef`；热循环不再解析 ref / 查 dict /
  重算形状。
- 下载路径三处同语义替换（§3.3 显示 download 由 1.92 → 1.16 ms）：FP16 有限值扫描改为
  4 lane 一组的整数判定、publish 改为按行 `memcpy`、`co_download` 元素拷贝按宽度特化。
  FP16 判定用**穷举**证明：65 536 个 bit pattern × 每个 lane 位置与
  `isfinite((double)h)` 逐一比对（`tests/c/fp16_scan_test.c`）。

<a id="round-1--b1--静态-repeat-成组入口方案-5-b14-a4"></a>
##### B1 — 静态 repeat 成组入口（方案 §5 B1、§4 A4）

`rk_workspace_run_group(ws, bindings, nrepeat, rk_group_report *report, diag)`：一次 native
调用执行同一不可变 plan 的 N 个独立 binding 集。

- 每个 repeat 仍执行**全部**原有检查；submit 仍逐个阻塞提交，**数量不变**。
- 发布边界整组一次；任一 repeat 失败不发布任何字节，`fail_repeat` 定位失败下标。
- **Host 预算（v3 修正）**：扩容上界按**总常驻占用**核算，即
  `rk_workspace_host_bytes(ws, capacity) = arena + 下载 tile + 发射数组 + 输出簿记`，与
  `host_budget_bytes`（未声明则用校验器重算的 `claimed_host_bytes`）比较，天花板不低于单次
  运行已占用量；另做乘法溢出检查。超限 `RK_E_BUDGET`，不分配、不发布、不 poison。
- 锁作用域（v3 修正）：`busy` 的释放只在**成功取得锁**的路径执行，由
  `workspace_run_body` / group 入口的 `out:` 各自负责；被拒绝的调用不触碰他人所有权。

<a id="round-1--guard--同覆盖率的精确扫描方案-4-a4-允许范围内"></a>
##### Guard — 同覆盖率的精确扫描（方案 §4 A4 允许范围内）

`co_guard_check` 由逐字节遍历 32 KiB slack 区改为逐 8 字节精确比较；未删除检查、未跳过
cache 同步；`tests/c/guard_scan_test.c` 在 13 个偏移（含非对齐与两个子区两端）破坏字节并
断言仍报错。

<a id="round-1--测量工具"></a>
##### 测量工具

`profile_bridge_envelope.py`（设备 model、实际亲和掩码、NPU 频率、逐次样本、包络归因，
无归因时 `envelope_available=false`）、`compare_bridge_envelope.py`（多臂、跨轮合并、
**吞吐由合并中位换算**、单样本字段标注来源轮次、结构与字节逐字段比对）、
`run-bridge-envelope.sh`/`-native.sh`（先全建后逐轮交替、`taskset` 绑核、每轮记录
loadavg/freq/亲和）、`plan_envelope_probe.c`（原语成本地板）、
`independent_mm_chain_probe.c`（B2 有界验证）、`run-numeric-replay.sh`（不重标定的
post-freeze 复放）。

<a id="round-1--2-方案第一个包要求回答的四个问题"></a>
#### 2. 方案第一个包要求回答的四个问题

**Q1 QK/PV 的 C workspace 未细分包络主要由什么构成？**
在“旧算法 + 只加归因”的中间构建（`results/20261026T015339Z-envelope`，未交替、未绑核）上
测得 C 侧停留份额：guard 门 44.9%、阻塞 submit 32.2%、download 10.4%、pack 6.6%。guard 的
大头不是 ioctl（一次 prep+fini 3.5–3.8 µs），而是每次调用对每个 data BO 重扫 32 KiB slack
区的字节循环——这就是改成同覆盖率向量化扫描的依据。**口径说明**：该份额是中间构建上的
指示性测量，不是 baseline 二进制上的读数；优化后的直接测量见 §3.3。

**Q2 不减少 Submit，只减少重复解析/分配/入口调用能得多少收益？**
b8 33.844 → 14.768 ms（2.292×），四轮一致。已消除的成本类别：每次 1 MB 发射数组与
per-call staging 的分配释放（A2）、每次调用 3 个 data BO 各扫 32 KiB（Guard）、64 次
per-repeat Python/ctypes 过渡收敛为 2 次 group 入口（B1）、每次 download 的 malloc/free 与
逐字节扫描/拷贝（A2 续）。**不把总收益拆成这些项各自的精确贡献**——那需要逐项回退构建，
未做；可直接测量的部分见 §3.3 与记录。

**Q3 32 个 head 任务能否合法成组？**
**B2 的设备前提已用有界实验证明（§5）**：32 个各自持有独立输入/权重/输出 BO 的 MM 任务放进
一次 chain，输出与 32 次独立提交**逐位相等**，耗时 1546 → 389 µs。但**产品路径仍未接通**：
现有 group 入口每个 repeat 复用同一组 BO，无法用于 chain；要接通需要“静态 repeat 展开成多
任务 plan + 新配方 `independent_mm_v1` + 清单版本化”。本包没有绕过 chain 白名单，三次测量
中 `submit_count == task_count` 恒成立。

**Q4 形成的能力是否对静态 B/H repeat 的一类图生效？**
是。触发条件是 manifest 声明的 `repeat` 轴；C 侧按 `ninputs+noutputs` 由 plan 推出，不出现
8/4/16/32 之类常量。MNIST MLP（无 repeat 轴）走单次路径，仍获得 A2/Guard 的收益（1.275×）。

<a id="round-1--3-rk3588-测量"></a>
#### 3. RK3588 测量

<a id="round-1--31-方法"></a>
##### 3.1 方法

- 三臂同 payload；`baseline` 与 `candidate` 各用自己的 C runtime 与 Python 包，`noprobe`
  与 `candidate` 共用源码，只换编译开关与目标库。
- **先全部构建，再逐轮交替测量**（第 r 轮内依次测三臂的同一 profile），四轮后按臂合并。
- `taskset -c 6` 固定 CPU；每轮记录 `/proc/loadavg`、`rknpu/freq`、`rknpu/load` 与亲和掩码，
  profiler 再从 `sched_getaffinity` 记录实际掩码。
- core 1 固定；每臂每轮 32 次 + 8 次 warm-up；权重加载与 runner 构造在区间外。

<a id="round-1--32-结果全部由合并样本导出"></a>
##### 3.2 结果（全部由合并样本导出）

```
profile          arm        n    min     p10     p50     p90     max   img/s(batch/p50)
tiny_vit_b8      baseline  128  28.907  31.848  33.844  36.051  38.372    236.4
tiny_vit_b8      candidate 128  10.152  12.076  14.768  16.297  17.632    541.7
tiny_vit_b8      noprobe   128  10.404  12.220  14.870  16.219  17.455    538.0
tiny_vit_b1      baseline  128   6.442   6.754   7.199   7.712   8.810    138.9
tiny_vit_b1      candidate 128   3.591   3.700   4.139   4.512   5.164    241.6
tiny_vit_b1      noprobe   128   3.615   3.800   4.171   4.549   5.720    239.7
mnist_mlp_b32    baseline  128   2.371   2.520   2.832   3.106   3.665  11298.1
mnist_mlp_b32    candidate 128   1.606   1.800   2.198   2.480   4.010  14558.9
mnist_mlp_b32    noprobe   128   1.708   1.868   2.285   2.520   3.140  14006.3
```

（本轮为 `results/20261026T055654Z-envelope/`，plan 包已含 accounting 修正。）

逐轮比值（baseline / candidate）b8 `2.36, 2.36, 2.15, 2.32`；b1 `1.75, 2.37, 1.72, 1.64`；
MLP `1.33, 1.26, 1.13, 1.47`。单轮值只作波动参考，结论只用合并中位。

<a id="round-1--33-优化后-b8-的逐区间归因-33-优化后-b8-的逐区间归因单次代表运行第-1-轮"></a>
##### 3.3 优化后 b8 的逐区间归因### 3.3 优化后 b8 的逐区间归因（**单次代表运行，第 1 轮**）

C 侧停留 8.616 ms，探针 0.03 ms 量级，未归属残差 0.089 ms（1.0%）。下表是**一次运行的
分解**，不是 128 次样本的统计量；占比以该次运行的 C total 为分母：

| 区间 | 时间 | 占 C 侧 | 含义 |
|---|---:|---:|---|
| submit_wait | 5.242 ms | 60.8% | 71 次阻塞 submit |
| download | 1.135 ms | 13.2% | BO 同步 + 拷出 + 有限值扫描 + 落 staging |
| pack | 1.045 ms | 12.1% | 135 次上传（BO prep/memset/packing/fini）|
| guard | 0.300 ms | 3.5% | 9 个发布边界的 slack/guard 门 |
| validate | 0.243 ms | 2.8% | binding/有限值/alias 校验 |
| output_sync | 0.200 ms | 2.3% | submit 后的输出 BO 握手 |
| command | 0.195 ms | 2.3% | 命令 BO prep/pack/verify/fini |
| emit | 0.145 ms | 1.7% | co_program 发射 |
| publish | 0.016 ms | 0.2% | staging → 调用方 buffer |
| 残差 | 0.089 ms | 1.0% | 仍未归属 |

同一轮的原语地板（`out-candidate/primitives.jsonl`）：`bo_sync_read/write_pair` 3.79 µs、
`guard_check` 13.1 µs、`download_cube16` 6.7 µs、`upload_cube16` 5.8 µs、
`submit_one_task` **70.6 µs**。最后一项说明：本轮机器状态下**单任务 submit 的固定成本就有
70 µs**，而实测每次 QK/PV submit 是 73.8 µs（5.242 ms / 71），即提交开销已经贴在设备地板上，
不再有“单次提交太慢”的空间——只能减少提交次数（B2）。

<a id="round-1--34-与-v1v2-表述的差异纠正"></a>
##### 3.4 与 v1/v2 表述的差异（纠正）

- **吞吐列口径（v3 修复）**：v2 用四轮合并延迟却填单轮吞吐。现一律 `batch / p50`：b8 为
  238.4 → **530.5** img/s（v2 的 230.7 → 507.8 作废），b1 为 139.2 → **235.1**，MLP 为
  11 154 → **14 225**。
- **归因口径（v3 修复）**：v2 把单轮代表样本的分解称作“128 次中位代表”。现明确标注为
  “单次代表运行，第 1 轮”，占比分母是该次运行的 C total。
- v2 写“优化前包络 100% 未归属”→ 已改为“旧入口不提供归因，baseline 上无法直接读出区间”。
- v2 写“guard 占 42%”→ 现按中间构建实测 44.9%，并注明是指示性测量。
- v2 写“b8 基线 36.388 ms”→ 绑核后为 33.8–33.6 ms（不同 campaign）。
- v2 写“MLP 差异在噪声内”→ 交替+绑核后稳定 1.275×。
- v2 写“预算问题已关闭”→ **撤回**：v2 只核算 staging，未计发射数组等其他常驻暂存；
  v3 改为按总占用核算（§6 第 2b 行）。
- busy-poll 本轮实测 `ROCKET_BUSY_POLL=0/20/50/100` 的 submit 时间为 5.23 / 5.17 / 5.09 /
  5.33 ms，无统计意义差异，**不作为候选**（记录以免重复尝试）。

<a id="round-1--4-正确性证据"></a>
#### 4. 正确性证据

| 证据 | 结果 |
|---|---|
| `scripts/run-numeric-replay.sh`（RK3588，冻结 spec 复放，不重标定） | **PASS / 0 problems** |
| 冻结数值门：33 个层门 + 5 个 profile 原始 logit 门 | 全部在 train 冻结的 `atol` 内，`rtol=0` |
| 历史全量 campaign 原始 logits 复查（15 行） | 通过 |
| 本机 10 000 张 mock 全量评测 | 与里程碑记录逐位一致（ViT b8 acc 0.9821、declared max 0.03125、hardware−CPU 0.0384254）|
| `pytest` | 244 passed / 41 subtests |
| `ctest` | 30/30 passed |
| C 主机检查 | `rk_plan_host_test` 787 checks + `fp16_scan_test` 穷举 65 536 个 FP16 pattern |
| group 入口等价性 | 32 个 repeat 的 group 结果与 32 次独立调用**逐位相等** |
| group 入口原子性 | 末位 repeat 注入失败时，前 31 个 repeat 的输出一个字节都没被改写 |
| ABI 布局 | C/Python 逐字段 offset/size 比对覆盖 `rk_run_report`(88 B)、`rk_run_attribution`、`rk_group_report` |

回归有效性经过**反向验证**（把缺陷重新引入，对应断言立即失败）：越界写 → `tail_intact`；
staging-only 预算 → `rc == RK_E_BUDGET` 与 `untouched`；BUSY 路径无条件解锁 → `busy == 1`
断言与后续 `RK_E_BUSY` 断言。

<a id="round-1--5-b2-有界验证设备前提成立两轮撤回后的最终证据"></a>
#### 5. B2 有界验证：设备前提成立（两轮撤回后的最终证据）

这一节的结论两次被推翻，原因是探针本身：

- v4 的探针把**每个任务的参考值都下载到同一块 `ref`**，循环结束后只剩最后一个任务的结果，
  却拿它比较所有任务 ⇒ 不同输入“失败”、相同输入“通过”，与 provider 无关；
- v4 的负向控制又把 buffer 与自身比较，恒不成立 ⇒ “self_check_failed”。

现已修正并加了**无设备**验证：`src/rk_task_ref.h` 把“按任务切片”的比较单独实现，
`tests/c/task_ref_test.c` 断言（a）各任务对自己的切片通过、（b）交换两个任务必须被检出、
（c）沿用“全部比较最后一个参考值”的历史写法在 n 个不同任务上必然报 n−1 个失配、
（d）单字节破坏被检出。探针改为 `n × y_bytes` 参考数组 + 逐任务切片比较，负向控制改为
“复制正确结果 → 破坏参考切片 → 必须失配”。

修正后的实测（`results/20261026T033032Z-b2probe/probe_fixed.log`，RK3588 core 1，绑核）：

| 形状 | n | 失配 | 未写 | 20 ms 后 | chain 摊到每任务 | separate 摊到每任务 | 比值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 每任务独立 BO | 16 | 0 | 0 | 0 | 12.1 µs | 49.6 µs | 4.08× |
| 每任务独立 BO | 32 | 0 | 0 | 0 | 16.0 µs | 105.5 µs | 6.59× |
| 共享 BO（展开 plan 布局）| 8 | 0 | 0 | 0 | 23.1 µs | 235.8 µs | 10.20× |
| 共享 BO（展开 plan 布局）| 16 | 0 | 0 | 0 | 16.9 µs | 159.6 µs | 9.42× |
| 共享 BO（展开 plan 布局）| 32 | 0 | 0 | 0 | 15.8 µs | 140.7 µs | 8.91× |

因此：**链式提交对“每任务不同输入/输出 buffer”是数值正确的**，读、写句柄都按任务生效，
且固定提交成本被摊薄。表中的 `separate` 侧在不同轮次波动很大（同一 n 可见 129–236 µs/任务），
比值只能看数量级；chain 侧每任务 12–23 µs 稳定。**这不构成产品准入**：要接通仍需
plan 层“静态 repeat 展开为多任务 plan” + `independent_mm_v1` 配方（plan 级
`RECIPE_SIGNATURES` 与 C 校验器同时登记）+ 清单版本化，并对展开后的 plan 重新做数值验收。

<a id="round-1--6-复核意见关闭表"></a>
#### 6. 复核意见关闭表

| # | 复核意见 | 处理 | 证据 |
|---|---|---|---|
| 1 | 旧 ABI 调用越界写内存 | `rk_run_report` 冻结 88 字节且不读 `struct_size`；扩展只经 `rk_workspace_run_ex` + 强制 `struct_size` | 0x5a / 0x00 / 0xa5 三种初始化下 72 B 尾部哨兵不变（含 `ws=NULL` 路径）|
| 2a | 返回 BUSY 时释放了他人持有的锁 | 释放动作移入 `workspace_run_body`，只在成功取锁的路径执行；group 入口同样只在取锁后 `out:` 释放 | 预置 `busy=1` 后连续 3 轮调用 `run`/`run_ex`/`run_group`/`destroy` 全为 `RK_E_BUSY` 且 `busy` 恒为 1；释放后正常服务、完成后 `busy==0` |
| 2b | group Host 预算只核算 staging | 上界改为 `rk_workspace_host_bytes(ws, nrepeat)`（arena + 下载 tile + 发射数组 + 输出簿记），天花板不低于单次占用 | 用运行时自己的公式导出边界：`fit` 接受、`fit+1` 拒绝，且被拒用例满足“仅 staging 未超限、总占用超限”；输出未被改写、未 poison |
| 3 | 统计口径混用（合并延迟配单轮吞吐、归因称作中位代表） | 吞吐统一 `batch / 合并 p50`；归因标注“单次代表运行，第 r 轮”；`comparison.json` 增加 `images_per_s_source`、`single_sample_round` | §3.2 / §3.3 与 `comparison.json` 字段 |
| 4 | BUSY 修复后仍在解锁后访问 `ws->plan` | 任务数在持锁期间复制进 `rk_run_state.task_count`，两个入口只用局部状态填报告，解锁后不再解引用 workspace | `tests/test_runtime_abi.py::test_no_workspace_access_after_the_run_lock_is_released`（反向验证：改回 `ws->plan` 即失败）；ASan/UBSan 全量 host test 无告警 |
| 5 | B2 探针把“没写输出”判为正确 | 参考值与 chain 结果都从哨兵开始、参考值留在 Host、逐输出比较 + guard 检查 + 比较器反向控制 | `results/20261026T033032Z-b2probe/`：修正后 31/32 失配，结论由“可准入”改为“设备前提不成立”（§5）|
| 6 | 探针失败仍返回成功、清理未初始化数组 | 任务数组 `memset` 清零并记录 `built` 计数；每个失败 stage 打印 stage 并以非零码退出；设备故障只报告不重试 | 同上；`probe.log` 中 `rc=1` 与 `status=incomplete` |
| 7 | 创建阶段 Host 预算缺口（`ceiling=max(budget,resident)` 消化超额） | 仅完成"不抬高天花板"：扩容上界改为真实预算，单次占用已超预算的 plan 拒绝增长并给出"recompile the plan"诊断 | `rk_plan_host_test` 预算段（`fit`/`fit+1` 边界 + 只 staging 超限用例）|
| 7b | 上述缺口的根因（planner/C 校验器未把运行时常驻 scratch 计入 `host`） | **已闭环**：`passes/storage.py` 与 `rk_plan_validate.c` 同时加入 `max_tasks × sizeof(co_program)` + 下载 tile + 输出簿记；`rk_workspace_create` 在分配前用同一总量对比预算并 fail closed；`CO_PROGRAM_BYTES` 由 C ABI dump 与 pytest 双向钉住 | 重新生成 plan 包（仅 `accounting.host_bytes` 变化）+ 原有数值政策重新验收：`results/20261026T053818Z-numeric/` **PASS / 0 problems**，且 `layer_tolerances`/`task_logit_tolerances` 与 v1 **逐项相同**；mock 10k 全量评测逐位一致 |
| 8 | B2 探针把所有任务与最后一个参考值比较（v4 的“设备阻塞”结论由此而来） | 参考值改为 `n × y_bytes` 逐任务切片；比较逻辑抽到 `src/rk_task_ref.h` 并加无设备验证 | `tests/c/task_ref_test.c`（各任务通过 / 交换被检出 / 历史写法必报 n−1 / 单字节破坏被检出）；`results/20261026T033032Z-b2probe/probe_fixed.log` 全部形状 0 失配；§5 结论更新 |
| 9 | B2 探针负向控制把 buffer 与自身比较，恒失败 | 控制改为“复制正确结果 → 破坏参考切片 → 必须失配” | 同上；探针 `status=ok` 且 `rc=0` |
| 10 | 测量脚本未交替、归因不是中位运行的分解 | 工具改为四变体**逐轮交替**；每次运行把整组 native report 累加，报告“每轮聚合的中位”，并在 JSON/README/V6 表格中标注口径 | `results/20261026T073654Z-b2bench/`、`results/20261026T081034Z-policy/` |
| 11 | “B2 稳定输给 B1”结论过宽 | 撤回：PV 的 chain_shared 快于 B1（3.090 vs 3.872 ms）；每任务 BO 的 chain（4.044 ms）确实慢于 B1，但那是布局而不是 chain 本身 | §0.1、§5 与两份 campaign |
| 12 | 仅按结构独立准入（8×8×8 / 32×32×32 / 64 任务可通过） | Python 与 C 两侧同时限定几何与数量，并用 ABI dump 钉住；补两种“绕过 Python”的拒绝测试 | §6 第 7 行之后的 `repeat_expand` 测试；C 的 `RK_MAX_INDEPENDENT_MM_TASKS` 与 `RK_INDEPENDENT_MM_*` |
| 13 | QK/PV 尚未接入 + 数值未复验 | 清单显式策略字段接入，保留 separate/B1 对照；plan 包变更后按同一冻结政策重新验收 | `results/20261026T081034Z-policy/`、`results/20261026T081153Z-numeric/`（PASS）|

<a id="round-1--7-新的成本排名与下一步候选"></a>
#### 7. 新的成本排名与下一步候选

按优化之后的 b8 单次代表运行成本排序（§3.3）：

| 排名 | 成本 | 占比 | 性质 | 候选 |
|---|---|---:|---|---|
| 1 | ~~submit_wait 5.23 ms~~ → **1.75 ms** | C 侧 33.5% | 已由 QK/PV 链式组消除（71 → 9 submits） | 完成（§0.1）|
| 1' | pack 1.21 ms + download 1.07 ms | C 侧 23.2% + 20.6% | 边界搬运/校验 | **C（设备子图）**：FFN gate/up → SwiGLU → down 驻留，同时消掉这两类边界；需先解决 FP32 中间值的数值路径问题 |
| 2 | download 1.16 ms | 13% | gate/up/qkv 三个大输出占多数 | **C**：FFN gate/up → SwiGLU → down 设备子图（先证明写回位置与数值路径）|
| 3 | pack 1.06 ms | 12% | 135 次上传 | 与 C 同源：减少边界次数 |
| 4 | Python 节点停留 | 约 12% 墙钟 | per-repeat ref 解析与逐次校验 | **A2 续/A3**：per-repeat 地址计算下推到 C |
| 5 | 注册 Host action | 约 17% 墙钟 | swiglu / softmax / rmsnorm 为主 | **D**：先证明能打通设备驻留 |

<a id="round-1--8-复现"></a>
#### 8. 复现

```bash
cd research/tilelang-rknpu

# 本机：编译 + 全部测试（244 pytest / 41 subtests / 30 ctest / 787 C checks）
./scripts/run-host-tests.sh

# 本机：包络自洽性（mock，无设备）
PYTHONPATH=$PWD/python python3 tools/profile_bridge_envelope.py \
  --backend mock --model tiny_vit --batch 8 --runs 32 --warmup 4

# 板端：三臂（baseline / candidate / noprobe）先全建，再逐轮交替测量 + 原语地板
RK_ENVELOPE_ARMS=baseline,candidate,noprobe \
RK_ENVELOPE_PROFILES=tiny_vit:8,tiny_vit:1,mnist_mlp:32 \
RK_ENVELOPE_RUNS=32 RK_ENVELOPE_ROUNDS=4 RK_TIME_CPU=6 \
./scripts/run-bridge-envelope.sh

# 板端：post-freeze 数值验收复放（不重标定）
./scripts/run-numeric-replay.sh

# 板端：B2 有界验证（独立 MM 任务 chain vs 独立提交，逐位比较）
cmake --build <build> --target independent_mm_chain_probe && \
  taskset -c 6 <build>/independent_mm_chain_probe
```

<a id="round-1--9-明确未做的事"></a>
#### 9. 明确未做的事

- 没有把 32 个 MM 任务放进产品 chain；B2 的设备前提经修正后的探针判定**不成立**（§5），
  未新增配方、未改清单、未做 plan 展开。
<a id="round-1--未完成且已登记的事项"></a>
##### 未完成且已登记的事项
- **B2 实施**：plan 层“静态 repeat 展开为多任务 plan”、`independent_mm_v1` 配方登记
  （plan 级 `RECIPE_SIGNATURES` + C 校验器）、清单版本化，以及展开后 plan 的数值验收。
  设备前提已验证（§5），但这三项都未做。
- **plan 包 v1 的归档**：`models/compiled/` 下现在是计入运行时常驻 scratch 的新包；旧包的
  逐文件副本仍保存在 `results/20261024T071144Z-models/campaign/plans/` 与各 numeric
  campaign 的 `campaign/plans/` 中，可作基线复现。schema 未升版（字段结构与含义未变，
  只是 accounting 数值变化）。
- 没有实现 NPU Softmax/RMSNorm，没有改变任何 Host 算术的数值路径（D 未做）。
- 没有合并跨节点的设备子图，没有减少任何 upload/download 字节（C 未做）。
- 没有逐项回退构建来拆分各优化的贡献。
- 没有重新标定数值容差，没有改动模型、权重、预处理、batch/tail 策略。
- 没有引入多核并发、没有改动超时或 cache 同步策略；busy-poll 实测无收益，未采用。
- `mnist_mlp` 的 group 路径未被触发（无 repeat 轴），其单次调用行为与基线一致。

---

<a id="round-2"></a>

## 第 2 轮优化结果

原文件：`OPTIMIZATION-RESULTS-2.md`。以下为该文件历史内容。

<a id="round-2--第二个优化包ffn-设备边界收敛实施报告"></a>
### 第二个优化包：FFN 设备边界收敛（实施报告）

日期：2026-10-26。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](#plan) 第 6 节方案 C。
本轮只动 FFN 的桥接边界与静态权重布局，不改模型数学定义、数据、batch/tail 或
QK/PV 策略。

<a id="round-2--0-结论摘要"></a>
#### 0. 结论摘要

默认 Tiny Transformer 现在把 FFN 的 gate/up 两个分支合并为**一次 N=2H 的
打包 GEMM**（`Wgu = concat(Wg, Wu)`，沿输出轴拼接），随后仍由注册 Host action
按原 FP32 路径计算精确 SwiGLU，再执行原 down GEMM 与 residual add。

RK3588 core 1、CPU 绑核、逐轮交替三臂（packed / device / host），4 轮 × 32 次
合并 128 样本：

| profile | packed（默认） | device（全设备 FFN） | host（原路径） | packed 相对 host |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | **11.772 ms** | 11.839 ms | 12.176 ms | **1.034×** |
| Tiny ViT b1 | **3.585 ms** | 3.776 ms | 3.827 ms | **1.067×** |

- b8 吞吐 657.0 → **679.6 img/s**；b1 261.3 → **278.9 img/s**
  （均由合并 p50 换算）。
- 每次 b8 调用结构：`task_count 71 → 70`、`submit_count 9 → 8`、
  `upload_calls 135 → 134`、`upload_bytes 312 832 → 296 448`（−16 384 B）；
  下载次数 `71 → 70`，下载字节不变（gate+up 的 64 KiB 只是合并成一次下载）。
- b8 C 侧停留（单次代表运行、第 1 轮）：`4.857 → 4.552 ms`；其中
  `submit_wait 1.432 → 1.229 ms`、`guard 0.291 → 0.258 ms`、
  `pack 1.196 → 1.153 ms`。`download 1.067 → 1.085 ms` 略升，未做归因外推。
- 默认 packed 路径对原 host 路径的 logits 与
  `gate/up/act/ffn/h2` **逐位相同**（Host mock 与板端数值门均验证）。
- 全设备 FFN（`device`）是另一条**独立数值政策**：它去掉 Host SwiGLU，但
  `act/ffn/h2/h2_tokens` 的冻结容差必须从 0.0625 放宽到 0.125；因此只保留为
  显式可选清单，不作为默认路径。

<a id="round-2--1-变更"></a>
#### 1. 变更

<a id="round-2--11-打包-gateup-gemm默认"></a>
##### 1.1 打包 gate/up GEMM（默认）

- `tools/compile_models.py` 为 b1/b8 编译 `tiny_vit_bN_gate_up`：
  真实 TileLang `make_matmul_prim(m, 64, 256)`，输入 `A=[M,64]`，
  权重 `B=[64,256]`，输出 `Y=[M,256]`。
- `tools/build_hybrid_manifests.py` 生成默认清单 `tiny_vit_bN.rkhybrid.json`：
  `gate_up -> swiglu.packed_gate_up.v1 -> down -> residual_ffn`。
- 新注册 Host action `swiglu.packed_gate_up.v1`：把 `[M,2H]` 结果切成
  `gate=[M,H]`、`up=[M,H]` 两个连续 view，再用
  `float32` 的 `gate * sigmoid(gate) * up` 只做一次 FP16 store。
- 新清单常量类型 `weight_concat`：从现有 `ffn_gate_w` / `ffn_up_w` 沿轴 1
  拼接，不新增权重文件、不改变权重内容。

<a id="round-2--12-全设备-ffn显式可选"></a>
##### 1.2 全设备 FFN（显式可选）

- `models/tilelang_matmul.py:make_swiglu_ffn_f16` 是真实 TileLang 源，
  入口为 `X, H, Wg, Wu, Wd, Y`；`H` 是 residual 输入，避免把 `norm2` 错当
  residual。前端 `_extract_ffn_branch` 现允许 residual 使用与分支输入不同的
  外部参数。
- 该计划包含 `gate mm -> up mm -> sigmoid LUT -> gate*sigmoid -> *up ->
  down mm -> residual add` 七个设备任务、一次输入边界和一次输出边界。
- 为保留冻结层门，额外编译 `tiny_vit_bN_ffn_trace`：同一提取图、同样 primitive
  任务，但导出 `gate/up/act/ffn`。`HybridRunner` 新增 `capture_only` 节点语义：
  普通测量路径完全跳过，只有 `capture=True` 的数值观测才执行。

<a id="round-2--13-清单版本化"></a>
##### 1.3 清单版本化

每个 batch 生成三份显式清单：

| 清单 | FFN 路径 | 数值政策 |
|---|---|---|
| `tiny_vit_bN.rkhybrid.json` | 默认 packed gate/up + Host FP32 SwiGLU | 与原路径逐位一致 |
| `tiny_vit_bN_deviceffn.rkhybrid.json` | 全设备 FFN + capture-only trace | 独立重标定 |
| `tiny_vit_bN_hostffn.rkhybrid.json` | 原 gate/up/down + Host SwiGLU | 原路径对照 |

回退只需换清单；不需要改模型、权重或 runner 代码。

<a id="round-2--2-数值验收"></a>
#### 2. 数值验收

<a id="round-2--21-默认-packed-路径"></a>
##### 2.1 默认 packed 路径

- 新 spec：`models/numeric_acceptance_packed.json`。
- 冻结来源：`results/20261026T084825Z-numeric/`（train 32 图校准，
  disjoint test 32 图验收；未读标签）。
- `layer_tolerances` 与 `task_logit_tolerances` 与上一轮
  `numeric_acceptance_train.json` **逐项相同**（33 层 + 5 profile，全部
  `atol/observed_max_abs` 无差异）。
- 验证结果：**PASS / 0 problems**。
- 冻结后复放：`results/20261026T085111Z-numeric-replay/`，**PASS / 0 problems**。
- Host mock 逐位检查：默认 packed 与 `_hostffn` 的 logits 及
  `gate/up/act/ffn/h2` 全部 `array_equal`。

<a id="round-2--22-全设备-ffn-路径"></a>
##### 2.2 全设备 FFN 路径

- 新 spec：`models/numeric_acceptance_device_ffn.json`。
- 冻结来源：`results/20261026T084901Z-numeric/`，**PASS / 0 problems**。
- 相对原路径必须独立放宽的行：
  `tiny_vit.act`、`tiny_vit.ffn`、`tiny_vit.h2`、`tiny_vit.h2_tokens`
  的 `atol` 由 0.0625 改为 0.125；`tiny_vit.pooled` 的观察误差由 0.00390625
  变为 0.0078125，但容差仍为 0.015625；5 个 profile logit 容差不变。
- 这不是默认路径，也不继承 packed/host 的 bit-exact 结论。

<a id="round-2--3-测量方法"></a>
#### 3. 测量方法

- 三臂共享同一源码、同一 C 构建、同一 plan/weights/data payload；臂只在运行时
  选择清单。
- 每轮顺序 device → packed → host 交替；`taskset -c 6`；NPU core 1；每臂每轮
  32 次 + 8 次 warm-up；权重/runner 构造在计时区间外。
- 计时字段与结构计数来自现有 `profile_bridge_envelope.py`；packed/device/host
  的结构差异是本次目的。
- 结果目录：`results/20261026T085009Z-ffn/`（`comparison.json` 为逐轮合并结果）。

<a id="round-2--4-测试"></a>
#### 4. 测试

- Host：259 pytest 全通过；30/30 ctest 全通过。
- 新增覆盖：
  - 默认 packed 清单 `task_count/submit_count/upload/download` 与
    `weight_concat`；
  - packed 对 host 的 logits 与全部 FFN trace 逐位一致；
  - `deviceffn` 清单为独立数值路径、且 trace 节点带 `capture_only`；
  - `capture_only` 在普通运行中跳过、在 capture 中执行；
  - `numeric_acceptance_packed.json` 与
    `numeric_acceptance_device_ffn.json` 的证据均可复放。

<a id="round-2--5-未做与限制"></a>
#### 5. 未做与限制

- 没有把 gate/up 和 down 之间再融合；中间仍需要 Host FP32 SwiGLU 才能保持
  默认路径逐位一致。
- 没有扩展 C `independent_mm` chain 的几何支持域；尝试把两次 FFN branch MM
  放进一次 submit 会被 C 校验器按既有验证域拒绝，未绕过。
- 没有改 QK/PV shared group、Softmax、RMSNorm、量化、多核或数值政策之外的参数。
- b1/b8 的收益分别来自同一 campaign 的合并 p50；不同 campaign 的绝对延迟不可
  直接拼接成新的倍数结论。

<a id="round-2--6-capture-only-生产隔离与-trace-证据边界"></a>
#### 6. capture-only 生产隔离与 trace 证据边界

本轮补齐了 `capture_only` 的**加载期**隔离，而不再只依赖执行期跳过：

- 普通执行节点读取 `capture_only` 节点的输出时，加载器直接拒绝；
- `outputs` 将 `capture_only` 节点的输出声明为普通模型输出时，加载器直接拒绝；
- 现有正例保留：`tiny_vit_bN_deviceffn` 的 `ffn_trace` 仍可在
  `capture=True` 时执行并导出 `gate/up/act/ffn`；
- `tiny_vit_b8_deviceffn` 在 `capture=False` 与 `capture=True` 下的最终
  `logits` 逐位相同，普通运行不会执行 trace 计划。

**证据边界**：trace 计划是同一提取图但单独执行的诊断 sibling，其
`gate/up/act/ffn` 是**重新执行得到的诊断证据**，不能表述为“被测生产计划内部值的
直接采样”。生产计划只发布其自身最终输出；冻结层门对 trace 的检查用于回归诊断，
不能替代对生产计划边界的证明。

<a id="round-2--7-qkpv-shared-group-在-packed-默认-ffn-上的复验"></a>
#### 7. QK/PV shared group 在 packed 默认 FFN 上的复验

保持 QK/PV 清单字段不变：

```json
{"repeat_submit": "shared_group", "group_plan": "attn_matmul_group"}
```

runner 提供三条显式策略臂：`separate`、`b1`、`chain_shared`
（`chain_shared` 强制使用清单声明的 shared group 计划；`manifest` 为默认等价路径）。
未扩大 C 侧已验证的独立 MM 几何域。

`results/20261026T085859Z-policy/`，packed FFN 默认，core 1、绑核、逐轮交替，
4 轮 × 32 次合并 128 样本：

| profile | B1 p50 | separate p50 | chain_shared p50 | chain_shared 相对 B1 |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | 14.584 ms | 22.344 ms | **11.692 ms** | **1.247×** |
| Tiny ViT b1 | 3.849 ms | 4.700 ms | **3.613 ms** | **1.065×** |

- b8 逐轮比值：`1.261 / 1.276 / 1.238 / 1.218`；
  b1：`1.060 / 1.105 / 1.060 / 0.923`。b1 第 4 轮反转，收益较小，不表述为稳定
  延迟改善；b8 四轮一致。
- 结构（b8）：B1/separate `task_count=70, submit_count=70`；
  chain_shared `task_count=70, submit_count=8`。上传/下载次数和字节三臂相同：
  `134 / 70`、`296 448 / 230 144 B`。
- 结构（b1）：B1/separate `task_count=14, submit_count=14`；
  chain_shared `task_count=14, submit_count=8`。上传/下载字节相同：
  `38 400 / 29 440 B`。
- 创建成本中位数（4 次采样）：b8 `runner_setup` B1 13.40 ms、separate 13.44 ms、
  chain_shared 13.67 ms；b1 分别为 7.71 / 7.75 / 7.84 ms。清单加载约 30 ms（b8）
  和 12.5 ms（b1）。chain_shared 创建成本略高 0.13–0.20 ms（b8）/
  0.09–0.13 ms（b1），未做显著性检验，不用于解释热路径收益。
- 默认策略：保留 `shared_group` / `chain_shared`。b8 收益明确；b1 pooled p50
  小幅领先但逐轮波动较大，按实测保留该策略而不承诺固定延迟收益。

<a id="round-2--8-数值复验packed-冻结政策不重标定"></a>
#### 8. 数值复验（packed 冻结政策，不重标定）

`results/20261026T085820Z-numeric-replay/`：使用
`models/numeric_acceptance_packed.json` 与 v1 冻结容差，重新观测 disjoint test
子集；**PASS / 0 problems**。实际模型输入下检查了 33 个层门和 5 个 profile
logit 门，例如 `tiny_vit.gate` max error 0.0078125、`tiny_vit.act` 0.015625、
`tiny_vit.ffn` 0.015625、`tiny_vit.h2` 0.015625、`tiny_vit.logits` 0.015625；
5 个 profile 均 `same_top1=True`。没有重新校准或放宽阈值。

---

<a id="round-3"></a>

## 第 3 轮优化结果

原文件：`OPTIMIZATION-RESULTS-3.md`。以下为该文件历史内容。

<a id="round-3--第三个优化包rmsnorm-连接边界评估与-host-收口实施报告"></a>
### 第三个优化包：RMSNorm 连接边界评估与 Host 收口（实施报告）

日期：2026-10-26。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](#plan) 第 6/7 节方案 D
的 RMSNorm 部分。默认路径仍为 packed FFN + QK/PV `shared_group`/`chain_shared`；
本包不扩展独立 MM chain 几何域、不引入多核、不同时推进通用 Softmax。

<a id="round-3--0-结论摘要"></a>
#### 0. 结论摘要

- **未发现可合法减少设备边界、且成本合理的 RMSNorm 连接。**
  当前 RKPlan 能力集没有行归约、general rsqrt、square 原语；已有 `co_row_ops`
  RMSNorm 是 C 侧显式 hybrid 算子，内部仍含 Host 全局 rsqrt 和逐 tile 统计下载，
  不能透明塞进 RKPlan 当作纯 NPU task。
- 复用 `co_row_ops` 的板端实测反而更慢：m=128,n=64（b8）约 **0.854 ms/次**
  （中位，1000 次重复；2 submits、3 tasks、2 uploads、2 downloads），
  b1 m=16,n=64 约 **0.266 ms/次**。对照当前 Host RMSNorm + 相邻上传：
  b8 `0.190 + 0.054 = 0.244 ms`（RMSNorm1）与
  `0.214 + 0.053 = 0.267 ms`（RMSNorm2）；b1 约 `0.093/0.095 ms`。
  因此该复用路径约为当前成本的 **2.8–3.5 倍**，且不消除 Host rsqrt。
- 作为保持数值路径的候选，实现了 `residual_attention + rmsnorm2` 的单 Host
  action（`tensor.add_rmsnorm.last_axis.v1`）并通过显式清单
  `tiny_vit_bN_rmsnorm` 选择。该路径与默认路径在 logits 及
  `h1/norm2/gate/up/act/ffn/h2` 上**逐位相同**。
- 板端 4 轮 × 32 次逐轮交替：fused 相对 control 的 b8 p50 为
  **0.997×**，b1 为 **0.989×**，结构计数与传输完全相同；没有实测收益。
  **默认路径保持不变**，fused 仅保留为显式对照清单。

<a id="round-3--1-当前默认路径的实际成本逐轮聚合"></a>
#### 1. 当前默认路径的实际成本（逐轮聚合）

数据：`results/20261026T091322Z-rmsnorm/` 的 control 臂（packed FFN +
QK/PV chain_shared），4 轮 × 32 次，core 1，CPU 绑核。
下表为每轮运行内 Host action / plan 分项在 128 个样本上的**合并中位**，不是
单次代表运行；它们用于说明成本位置，不与整体 p50 做加减解释。

| 区间 | b8 | b1 | 说明 |
|---|---:|---:|---|
| RMSNorm1 Host | 190.5 µs | 81.7 µs | Host FP32 mean/rsqrt/gamma，1 次 FP16 store |
| RMSNorm2 Host | 213.9 µs | 84.6 µs | 同上 |
| Softmax Host | 331.9 µs | 98.4 µs | stable max/exp/sum，Host FP32 |
| RMSNorm1 相邻上传（qkv pack） | 53.7 µs | 10.8 µs | 含 qkv 输入的 upload/pack |
| RMSNorm2 相邻上传（gate_up pack） | 53.1 µs | 10.8 µs | 含 packed gate/up 输入 upload |
| QK 相邻下载（qk download） | 189.6 µs | 23.9 µs | scores 出设备，供 Softmax |
| Softmax→PV 相邻上传（pv pack） | 412.7 µs | 40.7 µs | probs/V 入设备 |

Host 算术与桥接是两类开销：RMSNorm/Softmax 的 `elapsed_us` 是 Host 计算；
相邻 plan 的 `pack_us/download_us` 是设备边界前后的 staging/BO 成本。二者不能
简单相加成 p50，也不能把单次分项外推为整体加速比。

<a id="round-3--2-rmsnorm--qkv-与-rmsnorm--packed-gateup-的比较"></a>
#### 2. RMSNorm → QKV 与 RMSNorm → packed gate/up 的比较

| 连接 | 当前 Host 成本 | 后继 plan 输入桥接 | 可复用能力 | 结论 |
|---|---:|---:|---|---|
| RMSNorm1 → QKV | 190.5 µs (b8) | qkv pack 53.7 µs | 无 RKPlan 行归约/rsqrt；只能走 `co_row_ops` | 合法边界消除受阻 |
| RMSNorm2 → packed gate/up | 213.9 µs (b8) | gate_up pack 53.1 µs | 同上；packed gate_up 是单一 N=2H GEMM | 合法边界消除受阻；Host 成本更高，选作融合试验 |

`co_row_ops` 的端口是 `co_rowop_*` / `co_graph`，不是 RKPlan 的
`rk_action_desc`：要接入产品 hybrid 清单需要新增 plan action/template 语义与
C 校验，而不是现有能力。其内部流程是：device FP16 square、FP32 per-tile
sum、download 统计、Host 全局 rsqrt 和 gamma 系数、upload 系数、device scale。
因此即使把它包装成“RMSNorm 节点”，Host 步骤仍然存在，且新增 2 submits /
3 tasks / 2 uploads / 2 downloads。`co.lut.rsqrt_bounded` 的声明输入域仅为
`[0.25,4]`，也不能直接代表通用 RMSNorm 的均值域。

<a id="round-3--21-板端-co_row_ops-探针"></a>
##### 2.1 板端 `co_row_ops` 探针

使用已有 `row_graph`（不新增 NPU task），m/n 与模型一致，1000 次重复、
core 1，三次中位数：

```
m=128 n=64 chain    0.854 ms/run, 2 submits, 3 tasks, 2 uploads, 2 downloads
m=16  n=64 chain    0.266 ms/run, 2 submits, 3 tasks, 2 uploads, 2 downloads
device_bytes: 324032 B (b8), 252352 B (b1)
```

对照 Host RMSNorm + 相邻上传约 0.244/0.267 ms (b8) 与 0.093/0.095 ms (b1)，
该路径没有采用。

<a id="round-3--3-保持原数值路径的-host-融合尝试"></a>
#### 3. 保持原数值路径的 Host 融合尝试

新增 Host action：

`tensor.add_rmsnorm.last_axis.v1`：
- 输入 `a,b` FP16、`gamma` FP16；
- 先做 `a.float32 + b.float32 -> FP16`，与现有 `tensor.add.v1` 的 residual
  收窄点完全相同；
- 再以该 FP16 值做 last-axis FP32 RMS 均值、rsqrt、gamma 乘，最后一次 FP16
  store；
- 输出 `h`（residual FP16，保留 trace 与后续 FFN residual）和 `y`（norm）。

显式清单 `tiny_vit_bN_rmsnorm` 将
`residual_attention + rmsnorm2` 合并为一个 Host action，后续仍接
`gate_up -> swiglu -> down -> residual_ffn`。默认清单和 fused 清单共享同一
plan 包，不改变任何设备任务。

<a id="round-3--4-端到端对照"></a>
#### 4. 端到端对照

`results/20261026T091322Z-rmsnorm/`，4 轮 × 32 次，逐轮交替，core 1：

| profile | control p50 | fused p50 | fused / control | 结构 |
|---|---:|---:|---:|---|
| Tiny ViT b8 | 11.885 ms | 11.921 ms | **0.997×** | 完全一致 |
| Tiny ViT b1 | 3.587 ms | 3.628 ms | **0.989×** | 完全一致 |

- b8 逐轮 `1.003 / 0.959 / 0.980 / 1.033`；b1 `0.973 / 0.990 / 1.009 / 0.976`。
- tasks/submits、upload/download 次数与字节三臂相同；b8 为 70/8、
  134/70、296448/230144 B，b1 为 14/8、22/14、38400/29440 B。
- Host action 合并后 median 从 `140.9 + 213.9 = 354.8 µs` 降到
  单 action `341.4 µs`（b8），b1 从 `129.6 µs` 降到 `120.5 µs`；
  节省量相对整体 p50 太小，墙钟无收益。
- 创建成本：b8 runner setup 14.14 → 14.02 ms，b1 7.73 → 7.70 ms，量级相同。

**默认策略：保留原路径。** fused 清单仅作为显式对照，不改变默认。

<a id="round-3--5-数值复验"></a>
#### 5. 数值复验

- 默认 packed：`results/20261026T091548Z-numeric-replay/`，使用
  `models/numeric_acceptance_packed.json` 的既有容忍限，**PASS / 0 problems**。
- fused RMSNorm 清单：`results/20261026T091612Z-numeric-replay/`，通过
  `--manifest-key tiny_vit_b8=tiny_vit_b8_rmsnorm` 等显式映射运行同一
  packed 容忍限，**PASS / 0 problems**。
- 两处都没有调用 `calibrate_numeric_train.py`，没有修改任何
  `atol/rtol/observed_max_abs`；只更新了 plan 包身份哈希，并重新观测
  deterministic train/test 子集以匹配新包。
- 全设备 FFN 的独立政策仍单独保留：`results/20261026T091954Z-numeric/`
  在显式 `_deviceffn` 映射下 **PASS / 0 problems**，但
  `act/ffn/h2/h2_tokens` 的 `atol=0.125` 属于新数值政策，不是原冻结政策。

<a id="round-3--6-测试"></a>
#### 6. 测试

- Host：260 pytest 全通过；30/30 ctest 全通过。
- 新增覆盖：
  - 合并 `residual + RMSNorm` action 与默认两 action 的逐位一致；
  - `_rmsnorm` 显式清单节点/action 形状与 trace；
  - `--manifest-key` 数值观测映射与两套 replay；
  - `profile_bridge_envelope.py` 的 pooled host/plan 统计与创建成本字段。

<a id="round-3--7-未做"></a>
#### 7. 未做

- 没有新增 device RMSNorm task/template，也没有把 `co_row_ops` 包装成
  RKPlan 伪节点。
- 没有改变 Host FP32 SwiGLU、Softmax、RMSNorm 的数学路径。
- 没有调整默认策略：fused RMSNorm 无实测收益。
- 没有扩展独立 MM chain、多核或通用 Softmax。

---

<a id="round-4"></a>

## 第 4 轮优化结果

原文件：`OPTIMIZATION-RESULTS-4.md`。以下为该文件历史内容。

<a id="round-4--第四个优化包qkpv-shared-group-上传同步收口实施报告"></a>
### 第四个优化包：QK/PV shared group 上传同步收口（实施报告）

日期：2026-10-26。默认路径保持 packed FFN + QK/PV `shared_group` /
`chain_shared`。本包只优化已验证独立 MM shared group 的上传准备阶段，不改变
布局、padding、上传区域、数值路径、独立 BO 约束、链几何域或多核策略。

<a id="round-4--0-结论摘要"></a>
#### 0. 结论摘要

`upload_action()` 原先对每个切片分别执行 `rocket_bo_prep(write)` 和
`rocket_bo_fini()`。provider 契约确认：

- `rocket_bo_prep` 的 `dir` 只记录调用意图；实际行为是等待该 BO 的 WRITE
  usage fence，并做一次 whole-BO 的 `dma_sync_sgtable_for_cpu()`；
- `rocket_bo_fini` 做 whole-BO 的 device sync。

因此对**同一 BO 的连续、互不重叠切片上传**，可以安全地合并为：
一次 prep → 所有切片的清零/布局写入 → 一次 fini，且所有消费任务都在整个
上传段之后。实现只合并**连续 upload action 段**，不跨越任何 task/download。

默认开启合并，原逐切片路径通过 `RK_PLAN_UPLOAD_COALESCE=0` 显式保留为对照。

全量化 4 轮 × 32 次、逐轮交替、core 1：

| profile | coalesced p50 | per-slice control p50 | 加速比 |
|---|---:|---:|---:|
| Tiny ViT b8 | **11.721 ms** | 11.969 ms | **1.021×** |
| Tiny ViT b1 | **3.660 ms** | 3.718 ms | **1.016×** |

- b8 逐轮（coalesced/control）：`0.984 / 1.007 / 0.975 / 0.942`；
- b1：`0.986 / 0.967 / 0.984 / 0.988`；
- tasks/submits、upload/download 次数与字节完全一致；
- 默认采用 coalesced。

<a id="round-4--1-pack_us-拆分逐轮聚合中位"></a>
#### 1. pack_us 拆分（逐轮聚合中位）

数据：`results/20261026T093305Z-upload/`。拆分为四个子区间，均属于
`pack_us` 内部，**不与 `pack_us` 相加**：

| b8 | control | coalesced |
|---|---:|---:|
| `pack_us` | 1122.9 µs | 597.6 µs |
| `pack_prep_us` | 300.4 µs | 41.1 µs |
| `pack_clear_us` | 87.5 µs | 54.5 µs |
| `pack_layout_us` | 410.4 µs | 408.9 µs |
| `pack_fini_us` | 273.0 µs | 56.3 µs |
| prep / fini calls | 134 / 134 | 10 / 10 |

| b1 | control | coalesced |
|---|---:|---:|
| `pack_us` | 159.5 µs | 117.5 µs |
| `pack_prep_us` | 48.7 µs | 25.7 µs |
| `pack_clear_us` | 15.8 µs | 11.4 µs |
| `pack_layout_us` | 56.3 µs | 57.8 µs |
| `pack_fini_us` | 30.9 µs | 16.6 µs |
| prep / fini calls | 22 / 22 | 10 / 10 |

QK/PV 两个 group plan 本身（b8）：

| plan | control pack | coalesced pack | control prep/fini calls | coalesced calls |
|---|---:|---:|---:|---:|
| qk | 425.2 µs | 158.2 µs | 64 / 64 | 2 / 2 |
| pv | 423.9 µs | 154.7 µs | 64 / 64 | 2 / 2 |

b1 的 qk/pv 从约 44.2/44.3 µs、8/8 calls 降到约 25.4/25.5 µs、2/2 calls。
layout 打包成本基本不变；收益主要来自消除重复的 BO prep/fini。

<a id="round-4--2-实现边界与失败语义"></a>
#### 2. 实现边界与失败语义

- 只处理**连续 upload action 段**；段内按 destination BO 去重后一次 prep。
- 每个切片仍执行原有 `memset` 和 source→destination 布局打包，上传区域、
  padding、转置/广播标志与 bytes 计数不变。
- 所有切片写入完成后才 fini；随后才进入后续 task/download action。
- prep 失败：已 prep 的 BO 逐个 fini 清理，不提交任何消费任务。
- 布局/来源检查失败：清理已 prep 的 BO，不提交。
- fini 失败：不提交消费任务，返回错误；不清空或绕过现有 quarantine。
- 同一 workspace 在一次失败的 prep/fini 后可重试；mock 注入测试验证了
  `execs` 不前进且清理后可继续运行。
- 超过 128 个连续上传切片的异常大段退回/拒绝，不改变已验证产品几何
  （QK/PV 为 64 个切片）。

<a id="round-4--3-数值与回归"></a>
#### 3. 数值与回归

- 使用 packed 冻结政策复验，**不重新校准、不放宽阈值**：
  `results/20261026T093427Z-numeric-replay/`，**PASS / 0 problems**；
  33 层门 + 5 个 profile logit 门全部通过。
- coalesced 与 per-slice 路径在 mock 上逐位相同；测试同时覆盖：
  - 重复运行输入变化后每个切片重新打包；
  - prep/fini 调用数下降而 `upload_calls/upload_bytes` 不变；
  - prep/fini 注入失败不提交消费任务、清理后可重试；
  - ABI 扩展字段的 C/Python 尺寸与旧 V1 兼容。
- Host：261 pytest 全通过；30/30 ctest 全通过。

<a id="round-4--4-资源与创建成本"></a>
#### 4. 资源与创建成本

- b8 runner setup 中位约 13.76 → 12.82 ms；b1 约 7.79 → 8.21 ms
  （b1 反向波动，采样量有限，不作为收益声明）。
- manifest load 约 30 ms（b8）/12.5 ms（b1），两臂同 payload。
- 上传字节、下载字节、task/submit 数均无变化；无新增常驻 BO。

<a id="round-4--5-采用结论"></a>
#### 5. 采用结论

- 默认保留 coalesced 上传准备：b8/b1 的 pooled p50 均有正收益，且数值路径
  与结构计数不变。
- 回退开关：`RK_PLAN_UPLOAD_COALESCE=0` 使用原逐切片 prep/fini 路径。
- 不扩展独立 MM chain 几何域，不引入多核，不推进通用 Softmax，不改变数值政策。

---

<a id="round-5"></a>

## 第 5 轮优化结果

原文件：`OPTIMIZATION-RESULTS-5.md`。以下为该文件历史内容。

<a id="round-5--第五个优化包上传策略契约修复与-qkpv-布局打包快路径实施报告"></a>
### 第五个优化包：上传策略契约修复与 QK/PV 布局打包快路径（实施报告）

日期：2026-10-26。默认组合保持 packed FFN + QK/PV `shared_group` /
`chain_shared`。本包不改变布局、padding、FP16 位模式、数值政策、独立 MM 链几何域、
多核或通用 Softmax。

<a id="round-5--0-结论摘要"></a>
#### 0. 结论摘要

1. `run_upload_run()` 不再因内部 128 项数组上限拒绝合法大计划：超过上限时安全回退到
   原逐切片 prep/pack/fini 路径。新增 130 个连续上传 action 的合法 C 计划回归。
2. 上传策略改为 **workspace 创建时快照**：`RK_PLAN_UPLOAD_COALESCE` /
   `RK_PLAN_UPLOAD_FAST_PACK` 在创建后修改不影响已有 workspace；实际策略写入
   `rk_run_attribution.upload_coalesce/upload_fast_pack` 并由 Python `RunStats` 报告。
3. 合并优化只作用于已验证独立 MM shared group：单 submit、>1、≤32 个 plain MM task。
   其他计划一律走原逐切片上传路径。
4. 新增固定 QK/PV 16×16 CUBE/WEIGHT 布局打包快路径：
   - CUBE `[16,16] -> [16,32]` 按两个 8 列 bank 做连续 8 FP16 拷贝；
   - WEIGHT `[16,16] -> [32,32]`、`transpose_source` 直接转置写入；
   - 先 memset 整个目标 tile，padding 继续为零；
   - 其他输入走原通用循环。
5. 默认启用 fast pack。b8 布局耗时约 439 → 260 µs，端到端 b8 11.637 → 11.414 ms
   （1.0195×）；b1 3.706 → 3.663 ms（1.0117×）。仍按“小幅实测改善”表述：
   b8 四轮均为正，b1 有两轮基本持平。

<a id="round-5--1-上传策略契约修复"></a>
#### 1. 上传策略契约修复

<a id="round-5--11-超过-128-个上传的安全回退"></a>
##### 1.1 超过 128 个上传的安全回退

`run_upload_run()` 先扫描连续 upload action：

- 若输入切片数超过内部 `RK_MAX_COALESCED_UPLOADS=128`，不再返回
  `RK_E_UNSUPPORTED`；
- 整段走原逐切片 `upload_action(manage_sync=1)`，保持原支持域；
- 新增 C 回归：一个 verifier 接受的合法计划包含 130 个连续 upload action，
  运行成功且 `upload_calls=130`、`task_count=1`。

<a id="round-5--12-创建时快照"></a>
##### 1.2 创建时快照

- `rk_workspace_create` 读取并保存 `upload_coalesce` / `upload_fast_pack`；
- 运行路径只读 workspace 快照，不再读取环境变量；
- `rk_run_attribution` 追加实际策略字段，Python `RunStats` 暴露
  `plan_upload_coalesce/plan_upload_fast_pack`；
- 回归验证：先以 env=1/0 创建 runner，再修改 env，运行仍按创建时策略执行。

<a id="round-5--13-限定在已验证路径"></a>
##### 1.3 限定在已验证路径

合并仅当 `plan_is_independent_mm_group(plan)` 为真：

- 恰好一个 submit；
- 2..32 个 task；
- 全部为 `co.mm.f16.v1`。

其余计划保留原逐切片 prep/fini。该限制不缩小任何原有计划的支持域。

<a id="round-5--2-布局打包快路径"></a>
#### 2. 布局打包快路径

固定几何来自已验收 QK/PV 的 16×16×16 probe：

- 输入 A：CUBE `rows=16, cols=16, physical_rows=16, physical_cols=32, atom=8`；
- 权重 B：WEIGHT `rows=16, cols=16, physical_rows=32, physical_cols=32, atom=32`，
  `transpose_source=True`；
- 仅在 `source_row_stride=16`、offset 为 0、无 broadcast、`elements=256` 时启用。

CUBE 按 `(c/8)` 两 bank、每行 8 个 FP16 连续拷贝；WEIGHT 保留转置语义，直接写
`base[r*32+c]=src[c*16+r]`。目标 tile 先完整清零，因此 padding 仍为零。

<a id="round-5--3-板端结果"></a>
#### 3. 板端结果

<a id="round-5--31-布局快路径coalesced-均为开"></a>
##### 3.1 布局快路径（coalesced 均为开）

`results/20261026T094847Z-layout/`，4 轮 × 32 次，逐轮交替、core 1：

| profile | fast p50 | generic p50 | fast/generic |
|---|---:|---:|---:|
| Tiny ViT b8 | **11.414 ms** | 11.637 ms | **0.981×**（1.0195× speedup） |
| Tiny ViT b1 | **3.663 ms** | 3.706 ms | **0.988×**（1.0117× speedup） |

逐轮 fast/generic：

- b8：`0.985 / 0.965 / 0.972 / 0.969`，四轮一致改善；
- b1：`0.935 / 1.002 / 0.983 / 1.001`，两轮改善、两轮基本持平。

pack 分项（逐轮聚合中位，单次代表运行口径）：

| b8 | fast | generic |
|---|---:|---:|
| `pack_us` | 446.2 µs | 625.9 µs |
| `pack_layout_us` | 260.5 µs | 439.2 µs |
| qk `pack_layout_us` | 13.3 µs | 96.1 µs |
| pv `pack_layout_us` | 13.1 µs | 95.4 µs |

b1：`pack_us` 89.0 → 119.3 µs，`pack_layout_us` 32.1 → 56.9 µs（fast/generic）。

<a id="round-5--32-同步合并收益保持"></a>
##### 3.2 同步合并收益保持

前一轮 `results/20261026T093305Z-upload/` 的同步合并结论保留：
b8 11.969 → 11.721 ms（1.021×），b1 3.718 → 3.660 ms（1.016×）。
本轮 fast pack 与 coalescing 叠加，不重算或改写该 campaign 的比值。

<a id="round-5--4-数值与失败回归"></a>
#### 4. 数值与失败回归

- packed 冻结政策复验：`results/20261026T094941Z-numeric-replay/`，
  **PASS / 0 problems**；未重新校准、未放宽阈值。
- 快路径与通用路径：mock 上 logits 及全部 trace 逐位相同，第二次不同输入运行
  也逐位相同。
- 覆盖：
  - fast/generic 打包字节与输出一致；
  - 输入变化后每轮重新打包；
  - coalesced vs per-slice 调用数变化但 `upload_calls/upload_bytes` 不变；
  - prep/fini 注入失败不提交消费任务且可重试；
  - >128 上传合法计划回退；
  - 环境变量在 workspace 创建后修改不改变策略；
  - ABI V1/V2 兼容与新字段尺寸。
- Host：263 pytest 全通过；30/30 ctest 全通过。

<a id="round-5--5-采用结论"></a>
#### 5. 采用结论

- 默认保留 sync coalescing 与 fixed-layout fast pack；
- 对照开关：`RK_PLAN_UPLOAD_COALESCE=0` 保留逐切片同步，
  `RK_PLAN_UPLOAD_FAST_PACK=0` 保留通用布局打包；
- 两者均在 workspace 创建时快照，修改环境不影响已创建 workspace；
- 结论限定为“小幅实测改善”，不宣称每轮、所有 profile 都必然更快。

---

<a id="round-6"></a>

## 第 6 轮优化结果

原文件：`OPTIMIZATION-RESULTS-6.md`。以下为该文件历史内容。

<a id="round-6--第六个优化包pythonctypes-binding-复用实施报告"></a>
### 第六个优化包：Python/ctypes binding 复用（实施报告）

日期：2026-10-26。默认路径保持 packed FFN + QK/PV `chain_shared` +
coalesced 上传同步 + fixed-layout fast pack。本轮仅在 Python/FFI 编排层复用
PlanCall 的 ctypes binding 描述符，不改变 C 计划、布局、传输或数值路径。

<a id="round-6--0-结论摘要"></a>
#### 0. 结论摘要

- **采用**：`PlanCall` 在创建时缓存静态 binding 描述符；每次运行只更新
  `data`/`nbytes`，不再为每个输入/输出重新分配 `RKBinding` 与 shape 数组。
- 同一 campaign、同源码、同 payload、逐轮交替，只切换 binding 复用开关：
  - b8：11.579 → **10.043 ms**（**1.153×**）
  - b1：3.785 → **3.356 ms**（**1.128×**）
- b8 逐轮 reuse/control：`0.890 / 0.852 / 0.891 / 0.854`，四轮一致；
  b1：`0.969 / 0.761 / 0.903 / 0.867`，四轮一致。
- tasks/submits、上传/下载次数与字节、创建的计划结构完全不变。
- packed 冻结政策复验：`results/20261026T105527Z-numeric-replay/`
  **PASS / 0 problems**，未重新校准或放宽阈值。

<a id="round-6--1-成本定位与被放弃的候选"></a>
#### 1. 成本定位与被放弃的候选

<a id="round-6--11-下载同步合并未采用已撤回"></a>
##### 1.1 下载同步合并（未采用，已撤回）

在 `results/20261026T095856Z-download/` 中比较 shared-group 下载侧
per-slice `prep/fini` 与同 BO 一次 prep/fini：

| profile | coalesced p50 | per-slice p50 | 结果 |
|---|---:|---:|---|
| b8 | 11.340 ms | 11.414 ms | 0.65% pooled，逐轮 `1.013/0.964/0.978/1.018` |
| b1 | 3.669 ms | 3.666 ms | 基本持平，逐轮混合 |

C 侧 `download_us` 有下降，但端到端没有稳定收益，且增加了额外运行时分支与
ABI 字段。因此**撤回实现**，默认继续使用逐切片下载路径。

<a id="round-6--12-pythonctypes-编排"></a>
##### 1.2 Python/ctypes 编排

最新默认成本中，plan-node 包络之外仍有明显的 Python/FFI 停留。`PlanCall.run_into`
原先每次调用为每个 binding 执行：

- `make_binding()`；
- 分配新的 `RKBinding`；
- 分配新的 `shape` ctypes 数组；
- 重新填写 `abi_version/struct_size/id/dtype/rank/nbytes`。

QK/PV shared group 每次分别有 96 个 binding（64 输入 + 32 输出），两次
group 加上其他 plan，使这部分重复分配成为可独立消除的对象。

<a id="round-6--13-residual--reshape--mean-host-尾部融合未采用已撤回"></a>
##### 1.3 residual + reshape + mean Host 尾部融合（未采用，已撤回）

新增候选将 `residual_ffn + reshape_tokens + mean_pool` 合并为
`tensor.add_reshape_mean_axis.v1`，保留 FP16 residual 收窄、reshape view 和
FP32 token mean。单独 campaign `results/20261026T110224Z-tail/`：

| profile | split | fused | pooled |
|---|---:|---:|---:|
| b8 | 9.970 ms | 9.793 ms | 1.018× |
| b1 | 3.390 ms | 3.297 ms | 1.028× |

但在三臂累计 campaign `results/20261026T110622Z-cumulative/` 中：

| profile | baseline | binding reuse | binding + tail fuse |
|---|---:|---:|---:|
| b8 | 11.679 ms | **9.932 ms** | 9.966 ms |
| b1 | 3.680 ms | 3.380 ms | **3.310 ms** |

b8 的尾部融合相对只做 binding reuse 反而慢约 0.3%，与单独 campaign 的方向
矛盾；因此收益不够稳健。实现已撤回，默认保留 `residual_ffn -> reshape ->
mean_pool` 三步 Host action，不保留额外 action/清单分支。

<a id="round-6--14-复用-ctypes-binding-数组未采用已撤回"></a>
##### 1.4 复用 ctypes binding 数组（未采用，已撤回）

在 descriptor 复用基础上进一步缓存 `RKBinding` ctypes 数组，避免每次把
binding 复制进新数组。三臂同 campaign `results/20261026T111038Z-pybind/`：

| profile | control | descriptors | prepared arrays |
|---|---:|---:|---:|
| b8 | 11.463 ms | **9.856 ms** | 9.884 ms |
| b1 | 3.519 ms | 3.292 ms | **3.205 ms** |

- descriptors 相对 control：b8 1.163×、b1 1.069×；
- prepared arrays 相对 control：b8 1.160×、b1 1.098×；
- prepared 相对 descriptors：b8 基本持平/略慢，b1 有约 2.7% 改善。

由于 b8 没有端到端收益，而新增 prepared-array 分支会扩大 Python/FFI
生命周期与缓存状态面，因此**撤回该层改动**，只保留已验证的 descriptor 复用；
默认仍使用原 binding array 构造/复制路径。

<a id="round-6--2-实现"></a>
#### 2. 实现

`PlanCall.__init__(..., reuse_bindings=True)`：

- 根据 immutable plan bindings 预构造输入/输出 `RKBinding` 描述符及 shape 数组；
- 每次 `run_into()` 仍执行原有 `_prepare_inputs/_prepare_outputs`：
  dtype、shape、contiguity、writable 和输入 finite 检查全部保留；
- 仅把验证后的数组 `data` 指针与 `nbytes` 写回缓存描述符；
- 复用描述符只在本次同步 C 调用期间持有指针，调用结束不保留数组引用；
- `reuse_bindings=False` 保留原 `make_binding()` 路径作为对照；
- `HybridRunner` 在创建时接收该选项并传给所有 PlanCall；对照实验使用显式
  `HybridRunner(reuse_bindings=False)` 或独立 runner。

未删除任何输入 validation 或 finite 检查；未修改 C workspace、plan、上传布局、
同步、guard、发布或 quarantine 语义。

<a id="round-6--21-并发描述符保护"></a>
##### 2.1 并发描述符保护

binding cache 是 `PlanCall` 内的共享可变 ctypes 对象，指针更新发生在 C 的
BUSY 门禁之前。为避免两个线程在 Python 层互相覆盖，新增：

- `_descriptor_lock`，覆盖“更新 `data/nbytes` → 同步 C 调用返回”的完整区间；
- 非阻塞获取；已有调用持锁时，第二个调用直接返回
  `RK_E_BUSY`，保持原 C workspace 的 BUSY 语义，而不是静默排队；
- `run_into()` 与 `run_group_into()`（包括 flat group cache）共用同一保护；
- 任何异常路径通过 `finally` 释放；调用期间的数组引用由局部列表保持；
- 测试用 Event 控制交错：两个不同输入/输出数组，一组调用暂停在 C 调用前，
  第二组必须得到 BUSY、缓存指针不变、输出字节不被改写；释放后第一组完成，
  第二组再正常重绑定和运行。

<a id="round-6--3-同-campaign-端到端结果"></a>
#### 3. 同 campaign 端到端结果

`results/20261026T105439Z-pybind/`，4 轮 × 32 次，逐轮交替、core 1：

| profile | reuse p50 | control p50 | reuse/control | 加速 |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | **10.043 ms** | 11.579 ms | 0.867 | **1.153×** |
| Tiny ViT b1 | **3.356 ms** | 3.785 ms | 0.887 | **1.128×** |

- b8 吞吐由约 690.9 → **796.5 img/s**
- b1 由约 264.2 → **297.9 img/s**
- 结构计数、上传/下载字节完全一致；该收益来自 Python/FFI 编排，不是设备
  计算或桥接字节变化。
- 创建成本：b8 runner setup 约 14.4 → 14.9 ms，b1 8.1 → 7.3 ms；
  不作为收益声明，只记录同量级。

<a id="round-6--4-数值与回归"></a>
#### 4. 数值与回归

- `results/20261026T105527Z-numeric-replay/`：
  - 使用 `models/numeric_acceptance_packed.json` 既有冻结容忍限；
  - **PASS / 0 problems**；
  - 33 层门 + 5 个 profile logit 门全部通过；
  - 未调用重标定工具。
- 新增 binding 复用逐位回归：reuse on/off 的 logits 与
  `h1/norm2/gate/up/act/ffn/h2/logits` 全部 `array_equal`；
- 重复使用同一 runner 处理变化输入仍通过；
- Host：265 pytest 全通过；30/30 ctest 全通过。

<a id="round-6--5-采用结论"></a>
#### 5. 采用结论

- 默认开启 `reuse_bindings=True`；
- 对照开关：`HybridRunner(..., reuse_bindings=False)`；
- 这是创建时固定的 runner 配置，不在运行中读取环境或改变状态；
- 同 campaign 口径下 b8 1.153×、b1 1.128×，因此采用；
- 不同 campaign 的绝对延迟不作为累计加速结论。

<a id="round-6--6-剩余主要瓶颈"></a>
#### 6. 剩余主要瓶颈

当前 b8 默认（同 campaign 参考）的主要未消除项仍是：

- C workspace 的 `submit_wait` 与 `download`；
- Host Softmax / SwiGLU / RMSNorm 算术；
- plan-node 之外的 executor/Python 残留编排；
- guard、pack 与少量布局搬运。

下一项若要继续，需要在“更多 Python/FFI 编排收敛”与“Host 算术/相邻 Host
action 融合”之间选择；下载同步合并已因无可复现端到端收益撤回。

---

<a id="round-7"></a>

## 第 7 轮优化结果

原文件：`OPTIMIZATION-RESULTS-7.md`。以下为该文件历史内容。

<a id="round-7--第七个优化包qkv-输入布局映射实施报告"></a>
### 第七个优化包：QKV 输入布局映射（实施报告）

日期：2026-10-26。默认路径改为 packed FFN + QK/PV `chain_shared` +
coalesced 上传同步 + fixed-layout fast pack + binding descriptor 复用 +
**mapped QKV attention input**。物化 q/k/v 的路径保留为显式 `_splitheads`
对照清单。

<a id="round-7--0-结论摘要"></a>
#### 0. 结论摘要

- 新增显式、版本化的 `group_layout` 映射契约，QK/PV 的 shared group 上传
  直接读取原始 QKV 行块：
  - QK A 读取 Q 列窗口，B 读取 K 列窗口；
  - PV B 读取 V 列窗口并保留转置；
  - PV A 仍读取 softmax 产出的连续 probs 切片。
- 普通执行不再运行 `split_heads` Host action，也不再分配 q/k/v 三个转置数组；
  q/k/v 仅在 `capture=True` 时由 capture-only Host action 重新物化，属于诊断
  证据，不计入生产计时边界。
- 同 campaign `results/20261026T113808Z-qkvmap/`、4 轮 × 32 次：

| profile | splitheads control | mapped qkv | pooled |
|---|---:|---:|---:|
| Tiny ViT b8 | 10.784 ms | **10.760 ms** | **1.0022×** |
| Tiny ViT b1 | 3.521 ms | **3.428 ms** | **1.0270×** |

b8 逐轮 `1.038 / 0.975 / 1.016 / 0.981`，接近噪声边界；b1 四轮均为正。
因此采用该路径的主要依据是**实际消除了 Host 数组分配、转置复制和
split_heads 调用**，同时 pooled 墙钟不退化；不宣称 b8 每轮稳定加速。

<a id="round-7--1-映射契约"></a>
#### 1. 映射契约

清单 QK/PV 节点新增 `group_layout`（版本 1）：

```json
{
  "version": 1,
  "inputs": {
    "A": {
      "tensor": "qkv", "shape": [16,192],
      "axes": ["batch_token","qkv"],
      "row_slice": {"batch_axis":"bi","row_stride":16,"row_extent":16},
      "slice": {"head_axis":"hi","column_base":0,
                "column_stride":16,"column_extent":16},
      "transpose": false
    }
  },
  "outputs": {
    "Y": {"tensor":"scores_heads","index":["bi","hi"],
          "shape":[16,16],"axes":["query","key"]}
  }
}
```

- `row_slice` 只表达一个受限的 batch 行窗口（`row_stride` / `row_extent`），
  不是任意 stride。
- `slice` 表达 head 列窗口及转置关系；`transpose` 与 plan upload action 的
  `transpose_source` 必须一致。
- 输出必须显式声明 target tensor、完整 repeat index、shape 和 axes。
- 映射版本固定为 1；不支持的版本 fail closed。
- 每个 group plan 的 binding 名、顺序、shape、role 与 repeat-major 展开严格
  对照；每个 upload action 的 `src_row_stride`、`src_col_offset`、
  `transpose_source` 与映射逐个核对。
- Python loader 检查索引范围、行窗口、列窗口、容量和 shape；
  C verifier 独立检查最终 plan descriptor 的 source capacity、索引算术和
  溢出，不信任 Python 预计算。

<a id="round-7--2-执行与数值"></a>
#### 2. 执行与数值

- QK/PV group 的输入数组是 `qkv` 的连续 batch-block view 和 probs 的
  head-slice view；runner 不复制 q/k/v。
- QK B 需要 K 的“行块 + 列窗口 → 权重 tile 转置”关系；PV B 需要 V 的
  列窗口 → 权重 tile 转置关系。两者均按已验收的 16×16×16 shared-group
  几何执行，未扩大链几何域。
- 对同一底层输入 buffer 的重复 binding，C 和 Python 的 finite 检查均在
  **一次运行内按 `(data, nbytes)` 去重**；不跳过任何不同输入，也不改变
  finite 拒绝语义。
- `capture_only` HostNode 在普通运行中被跳过；`capture=True` 时重新物化
  q/k/v，供 trace 层门使用。其 Host 时间不计入普通执行边界。

<a id="round-7--3-板端与结构"></a>
#### 3. 板端与结构

同 campaign 4 轮 × 32 次、逐轮交替、core 1，结构计数一致：
`task_count 70`、`submit_count 8`、上传/下载次数和字节不变。

b8 representative C-side：validate 约 0.246 → 0.301 ms（更大源 binding 的
检查），但生产 `split_heads` Host action 被消除；整体 pooled 墙钟基本持平。
b1 pooled 墙钟 3.521 → 3.428 ms。

普通运行实际消除：

- `split_heads` Host action 调用；
- q/k/v 三个 Host 转置数组及其分配；
- q/k/v 到 QK/PV 前的中间引用解析。

capture 路径仍会物化 q/k/v，这是诊断边界，不是生产路径。

<a id="round-7--4-数值与回归"></a>
#### 4. 数值与回归

- packed 冻结政策复验：`results/20261026T113650Z-numeric-replay/`
  **PASS / 0 problems**，未重新校准、未放宽阈值。
- 映射路径与 `_splitheads` 对照：
  - b8 和 b1 的 logits 逐位相同；
  - `q_heads/k_t_heads/v_heads/scores_heads/probs_heads/attn_heads` trace
    逐位相同；
  - 不同输入批次重复运行逐位一致。
- 负例覆盖：
  - head 列窗口越界；
  - batch row slice 越界；
  - transpose 与 plan action 不一致；
  - C plan descriptor 的 source column offset 越界由 C 独立拒绝。
- Host：267 pytest 全通过；30/30 ctest 全通过。

<a id="round-7--5-采用与后续"></a>
#### 5. 采用与后续

- 默认：`tiny_vit_bN` 使用 mapped QKV；`tiny_vit_bN_splitheads` 保留原
  materialized 路径作为对照。
- B1/separate QK/PV 策略对照仍使用 `_splitheads` 清单；mapped group 是
  chain/shared 执行形态，不提供 B1/separate 的替代语义。
- 剩余主要成本仍是 C `submit_wait`/`download`、Host Softmax/SwiGLU/RMSNorm
  与 executor 残留编排。此次映射收益在 b8 接近噪声边界，后续若要继续提升
  应优先处理这些主项，而不是再增加映射分支。
