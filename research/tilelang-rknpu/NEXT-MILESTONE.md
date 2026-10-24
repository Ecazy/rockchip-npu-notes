# 下一阶段：完整模型推理

日期：2026-10-24。本文是已验收静态 FP16 MVP 之后的独立里程碑，不追溯增加上一阶段的验收条件。现状见 [支持域](SUPPORT-DOMAIN.md) 和 [验收矩阵](ACCEPTANCE-MATRIX.md)。

## 1. 总目标

在 RK3588 上，以真实学习所得的权重和标准数据集完成两个完整模型：核心计算全在 NPU 的 MNIST MLP，以及显式 hybrid 的小型 Transformer 分类器。交付权重/模型描述、真实 TileLang/TVM 输入、离线编译、部署、重复推理、任务准确率和成本报告的可重放流程。

按完整模型的需要补齐后端能力。复用 TileLang 前端、TVM IR、现有 RKIR/RKPlan 和 runtime；模型接入层只负责有限层类型、连接、常量和 shape，不另造通用语言前端或完整 ONNX/PyTorch 编译器。所有 NPU 子图仍经过真实 TileLang/TVM 接入，手写 RKIR 只作后端调试与测试。

## 2. 固定模型与运行范围

| 模型 | 首轮结构 | 计算落点 |
|---|---|---|
| MNIST MLP | Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU → Linear(128,10)；batch独立编译为1、8、32 | Linear、bias、ReLU在NPU；图像归一化/输入整理为Host准备，argmax为应用后处理；中间激活设备驻留 |
| Tiny Transformer 图像分类器 | 同一MNIST；28×28图像分16个7×7 patch；16 tokens、D=64、4 heads、1层encoder、FFN=128、10类；batch独立编译为1、8 | patch projection、Q/K/V、QK/PV、FFN、分类头的矩阵计算在NPU；首轮RMSNorm、稳定Softmax、mean pooling可用已注册Host compute |

Transformer固定为patch projection/位置参数、pre-RMSNorm attention/residual、pre-RMSNorm SwiGLU FFN/residual、mean pooling和分类头。不包括causal mask、KV cache、增量decode或持续状态。训练/取得权重在Host进行，不扩展为NPU训练。

优先使用来源与许可明确的适配checkpoint；没有可用权重时提供固定seed的Host训练脚本并冻结产物。随机权重、只跑中间层、挑少量样本或只跑FFN，均不能替代完整模型验收。测试集不得参与训练或阈值调参。

## 3. 必需的能力增量

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

## 4. 连续工作包

### N0：冻结模型与参考

- 保留当前MVP回归基线，冻结两个模型结构、权重来源、预处理和输入输出。
- 取得CPU参考；完整准确率使用官方MNIST test split的10,000张图像。
- 同时提供原始模型FP32参考和声明硬件路径参考；后者保留FP16/LUT/Host计算边界。
- 形成简短能力差异清单后直接实现，不另立模型导入平台设计项目。

### N1：完整 NPU MLP

- 接入全部权重、bias、ReLU，用真实TileLang staged程序编译执行。
- batch=1/8/32形成独立编译产物；未编译shape精确拒绝，不称为动态shape支持。
- 打通预处理→整个网络→logits→argmax→测试集评估，常量创建时上传一次，中间激活不出设备。
- 先正确执行，再选择有数值依据的融合/调度优化。

### N2：完整 hybrid Tiny Transformer

- 实现必要的多轴view、head布局、节点和hybrid执行边界。
- 首轮允许有报告的Host RMSNorm/Softmax/mean pooling与布局桥接，尽快运行整个模型。
- QK/PV及全部Linear/FFN不能悄悄退回CPU；缺失lowering即列为未完成。
- 比较逐层参考与最终分类，记录所有Host action。完整模型通过后才选择一个主要桥接瓶颈优化，不新增首轮“纯NPU Transformer”门槛。

### N3：封装、准入与交接

- 提供模型包、编译/推理/评估入口、两个独立可运行示例和准确支持域。
- 重用既有源码环境、campaign、sanitizer和故障回归；新增测试对应本阶段实际风险。
- 结果关联输入/权重/数据集哈希、编译产物、环境和完整命令；大文件沿用本地证据规则。
- 按模型包/参考、MLP、hybrid/Transformer、验收文档分批提交，中间提交不需要逐步审批。

## 5. 有限验收条件

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

## 6. 范围控制

动态shape通用编译、任意TileLang控制流/mask、通用ONNX导入、CNN全算子、NPU训练、INT8/INT4、完整LLM加载、KV cache/state、故障后继续计算、RK1228适配、多板/多核并发均不作为本轮完成条件。

RK1228到位后另行识别型号、驱动、内存与提交接口，再作为独立target复用模型/IR语义；不继承RK3588配方或验收结论。

执行agent自行决定本范围内的helper、接口细化、文件组织和实施顺序；卡点定位到具体语义/布局/数值条件，保留复现并继续独立工作，不反复请求相同路线审批。仅在公共契约需变更、破坏性操作或缺少必要外部条件时提问。

审核以完整模型、真实数据、后端能力和证据为主，不以mock接口数量或验证工具扩张代替交付。当前MVP保持已完成，不因本阶段新能力未实现而改写历史验收。
