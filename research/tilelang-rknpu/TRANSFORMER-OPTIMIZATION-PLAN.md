# Transformer 后端优化方案（分析稿）

原分析日期：2026-10-24；状态导航整理：2026-10-26。

本文保留方案 A–D 的设计理由、依赖与验收方法；正文中的成本表和预测按原分析时点解释。实际完成状态与数字统一查以下结果报告，避免将旧基线与新测量混用。

| 阶段 | 结果入口 | 已交付范围 |
|---|---|---|
| A 与 B1 | [OPTIMIZATION-RESULTS-1.md](OPTIMIZATION-RESULTS-1.md) | 运行包络归因、静态准备、成组调用与预算闭环 |
| C | [OPTIMIZATION-RESULTS-2.md](OPTIMIZATION-RESULTS-2.md) | 默认 packed gate/up + Host FP32 SwiGLU；device FFN 为独立数值政策的可选路径 |
| B2 产品接线与复验 | [OPTIMIZATION-RESULTS-2.md](OPTIMIZATION-RESULTS-2.md) | packed 默认 FFN 上的 QK/PV shared group，separate/B1/chain_shared 对照及冻结政策复放 |
| D（RMSNorm 部分） | [OPTIMIZATION-RESULTS-3.md](OPTIMIZATION-RESULTS-3.md) | 现有 RKPlan 无合法行归约/rsqrt 能力；`co_row_ops` 复用更慢；residual+RMSNorm Host 融合逐位等价但无实测收益，默认不变 |
| QK/PV 上传同步收口 | [OPTIMIZATION-RESULTS-4.md](OPTIMIZATION-RESULTS-4.md) | shared group 同 BO 连续切片合并为一次 prep/fini；prep/fini calls 134→10 (b8)，默认 coalesced，保留逐切片开关 |

完整模型里程碑已独立关闭，见 [MODEL-MILESTONE-STATUS.md](MODEL-MILESTONE-STATUS.md)。候选方案本身不构成实施授权、能力声明或验收通过。新一轮工作按具体任务推进，不由本文的历史“下一步”措辞自动启动。

## 1. 优化目标与适用范围

近期对象是当前 MNIST Tiny Transformer：16 tokens、D=64、4 heads、单层 encoder、SwiGLU FFN=128；重点分析 batch=8，batch=1作为独立延迟场景。保持权重、数据、预处理、模型数学定义及明确的数值政策。

目标是降低整模型端到端时间与设备往返成本，同时把方法实现为可复用的后端能力：静态 repeat 成组、计划与缓冲复用、已证明的布局传递、设备子图驻留。不得通过识别模型名或固定尺寸写整模型捷径。

原则可推广到其他Transformer，具体收益和优先级不能推广：长序列prefill可能受attention计算/中间矩阵容量限制；decode可能受权重带宽、KV访问与逐token调度限制。本文不设计KV cache、长序列attention、量化、多核并发或任意模型导入。

## 2. 当前证据与计时解释

分析依据：

- [MODEL-MILESTONE-STATUS.md](MODEL-MILESTONE-STATUS.md)。
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

### 2.1 主要瓶颈在逐head的细碎执行

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

### 2.2 搬运是高频边界问题

当前上传391.04 MB、下载287.68 MB；QK/PV各自的上传+下载约163.84 MB，合计占设备传输字节约48.3%。这些是运行期报告口径，不能当成物理DDR带宽测量。

大量小矩阵每次都经历准备、同步、提交、下载和再次调用。不能简单用总字节数除以带宽预测收益；Host tensor materialization字节也不能再次累加为设备DMA字节。

### 2.3 两个约束性上界

- 若只把QK/PV整个节点时间理想地降到0，其余部分保持不变，端到端仍约12.65秒，约791 img/s，仍慢于本次CPU参考。这个上界说明后续还需优化其他路径，不是可实现预测。
- 若只消除全部2.99秒Host action时间、数据流和其他成本不变，整体最多约1.063倍。因此“先写一个NPU Softmax就能解决问题”没有当前证据支持。消除Host边界若同时减少传输，收益才可能更大。

## 3. 推荐总体路线

```text
固定正确基线与成本归属
  → repeat解析/binding/分配的静态复用
  → QK/PV按batch-head成组执行
  → 去除可合法连接的设备子图边界
  → 再评估norm/Softmax下沉或其他placement
```

前两项直接针对调用次数与运行包络，优先级高于单独替换Softmax算术。每阶段使用同一个模型和冻结正确性政策，不引入新模型来替换基线。

## 4. 方案A：减轻已知语义不变的运行包络

### A1. 细分C workspace成本

后续若获准实施，先在少量固定batch的隔离分析中区分binding/finite、upload prep/pack/fini、command处理、submit/wait、output sync、download/unpack、guard、Host分配/发布。可复用已有计时与调用计数，不另建长期profiling框架。

详细插桩可能扰动小任务，因此定位数据与无插桩端到端数据分开报告。停止条件是能够解释QK/PV的主要包络，并选出实际可减少的成本；不能以不断提高仪器精度取代优化交付。

### A2. 编译静态执行描述，预分配暂存

- 在HybridRunner创建时解析输入输出binding顺序、repeat索引、slice描述、dtype/shape约束和不可变plan身份。
- 预分配可复用的输入整理、head输出、C staging和结果缓冲；多次run仅更新内容与必要指针。
- 每次run的输入shape/dtype、有限值、alias、context fault及输出发布条件仍须检查；只有不可变结构验证才能移出热路径。
- 复用仍须服从workspace busy、Host数组保活和故障隔离，不把临时view悬空或输出暴露给下一次run覆盖。

### A3. 缓存已绑定程序与合法chain准备

若view、BO、fd/domain、模板、shape、数值政策和提交顺序不变，数据内容更新通常不改变寄存器地址。可研究workspace内缓存已发射program或预构造Host命令镜像。

缓存必须覆盖地址/布局/版本失效。当前command BO会被不同submit组复用，不能直接让多个组共用一份已覆盖的设备命令；若缓存完整设备命令，应独立规划区域与预算。继续复用已有pack/verify/终端封口逻辑，不复制提交器。

### A4. guard与同步不能以“关掉”方式优化

当前每个小plan的最终发布门禁都可能检查多个BO。应先确定实际成本。如果将32个head纳入一个workspace，可在覆盖全部任务的发布边界统一检查，减少重复调用，同时保留全部必要检查。

不能在没有覆盖范围证明和失败回归时删除guard、跳过cache同步，或把设备故障转成成功。首轮保持检查强度，后续若区分诊断/生产模式应独立定义契约。

## 5. 方案B：QK/PV的batch-head成组（主要结构优化）

### B1. 以静态repeat为单元构建批量plan

以32个独立head矩阵任务组成一个QK组，完成后批量下载scores、执行现有Host Softmax，再以32个PV任务组成一组。一次Python/native入口处理一整组，不能只在Python中继续循环32次原单任务调用。

每个head仍有独立的正确输入、weight packing、输出区域和依赖。首轮保留现有physical布局、补齐与计算配方；不把不同行列的head直接拼成一次普通GEMM，以免算出跨head交叉项。block-diagonal GEMM会引入大量无效计算，不作为首选。

### B2. 区分“合并入口”与“合并硬件提交”

- 可以先用一个native组入口执行32次已支持的独立Submit，降低Python/ctypes/分配开销；明确报告Submit数没有减少。
- 把32个MM放进一次chain，需要新增独立MM组的合法配方与板端证据。任务无依赖、flags相同都不是充分证明；不能绕过白名单或复用K-acc链名义宣称安全。
- 当前每组最多64 tasks，batch=8/head=4的32任务组在数量上可容纳，但仍受BO独立、命令容量、内存预算、布局与completion约束。
- 新配方未验证时保持显式separate路径；验证失败精确拒绝/选择已有合法分组，不试投危险寄存器组合。

### B3. 有条件的计数目标

只有“每个QK组和PV组均合法地一次Submit”成立时，原有其他7个矩阵任务不变：

```text
每batch提交：71 → 9
全测试集提交：88,750 → 11,250（减少约87.3%）
primitive任务数初期仍为88,750，不是任务数减少87.3%。
```

这只是结构计数目标，不预测相同比例的延迟下降。一般静态B/H应按合法组容量分块，不能把batch=8/head=4写进通用后端规则。

### B4. 避免为少数head增加不必要的驻留内存

首轮可使用每head独立BO以保持已知alias契约；组workspace会增加命令与数据预留，必须计入slack/guard和Host staging。后续只有证明view、独立BO及驱动要求均满足，才考虑合并分配。

组内任何失败都不发布部分head，context quarantine与排队提交隔离继续有效。Host Softmax必须等待全体scores可见，PV必须等待全部所需概率和V已准备。

## 6. 方案C：去除设备子图之间的往返

按实际成本选择，推荐顺序如下。

| 候选 | 可复用的后端能力 | 主要限制 |
|---|---|---|
| gate/up共输入的打包GEMM | 把两个独立常量权重沿输出轴组合，产出后分成两个逻辑值 | 当前QKV已经合并，不重复优化；gate/up拼接需保持输出写回、合法N分片和split布局，不假设任意Cube切片可用 |
| FFN gate/up → SwiGLU → down → residual设备子图 | 复用已支持的MatMul/Mul/Add/LUT与存储规划 | 当前Host SwiGLU未必与LUT gate先FP16写回的数值路径等价；新增近似/收窄必须显式版本化并另行验收 |
| PV → head merge → output projection | 设备布局转换/消费布局融合 | 先证明head轴到模型通道的索引映射；不能只改shape/stride就认为转换已完成 |
| residual + RMSNorm → QKV或FFN | 行归约、运行时系数与后继输入驻留 | 几何、finite/padding、均方与rsqrt的精度和输入域需独立证明；不能把现有Host复合实现当不透明纯NPU节点 |
| position add、残差等EW进入已有设备阶段 | 减少一次materialize或边界 | 只在实际消除搬运/提交时有价值；为单个廉价Add新增Submit可能更慢 |

先选择能够明确减少一次upload/download或plan调用的连接，不要求“一次全部融合”。保留Hybrid清单解释优化前后节点、边界、传输和数值政策。

## 7. 方案D：norm/Softmax下沉与其他placement

Host Softmax本轮约0.464秒；两次Host RMSNorm合计约0.584秒。单独加速这些算术的整体上限有限，优先理由应是能够打通设备驻留，消除更大的桥接包络。

1. 在B/C有结果后，重新判断scores/probs桥接是否成为主要成本。
2. 若研究NPU稳定Softmax，必须实现max平移、exp值域、sum/归一化、padding/finite与各head独立性；现有bounded exp LUT不能直接代表通用稳定Softmax。
3. 若研究RMSNorm，优先复用已有算法与证据，明确内部Host步骤是否仍存在；仅搬动节点名称不算减少Host参与。
4. 可另设显式placement对照：很小的QK/PV留CPU是否更快。它必须作为不同且可见的hybrid政策，经后续授权再研究；不能作为当前“全部矩阵上NPU”里程碑路径的静默fallback，也不能把CPU收益算成NPU优化收益。

## 8. 数值与安全不变量

- 基线权重、数据、预处理、batch/tail与数值契约固定。独立校准集先确定政策，验收数据不参与反向扩大容差；不能以top-1不变取代逐层/logits检查。
- repeat成组、布局传递与缓存优先保持相同计算/收窄语义；仍须验证实际模板几何变化是否改变结果。
- 会改变FP16写回位置、LUT近似或归约顺序的融合，使用独立数值政策与结果列，不能继承旧路径的bit-exact结论。
- 布局必须证明logical→physical映射、有效区域与padding；输入权重/动态B更新不可使用过期内容。
- 不能削弱fd/domain一致性、alias、预算、busy、最终发布门禁或shared-context quarantine。失败组不能部分发布，未确认静止的资源不可释放/复用。
- 不以多核并发或危险timeout制造收益/证据；首轮继续单个固定核。

## 9. 后续若实施时的有限比较方法

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

## 10. 推荐的首个优化包与停止条件

首个优化包建议限定为：**解释QK/PV运行包络 + 静态repeat批量入口 + 有边界的独立MM组成组验证**。它最直接覆盖当前75%的节点耗时，且不要求先实现新的Softmax算法或纯NPU Transformer。

首包结束时应能回答四个问题：

1. QK/PV约21.87秒的C workspace未细分包络，主要由哪些实际操作构成？
2. 不减少Submit，只减少重复解析/分配/入口调用，能得到多少真实收益？
3. 32个head任务能否通过已验证配方合法成组；若不能，明确阻断，而不是绕过白名单？
4. 形成的能力是否对静态B/H repeat的一类图生效，而不是只识别本模型？

满足首包问题后，再按新的成本排名选一个C/D候选。若正确性、安全或稳定性不能保持，停止该候选，不变更模型或扩大数值门槛换取速度。

本方案提出的是可供后续选择的路线，不包含已实现、已测得或已承诺的加速结论。
