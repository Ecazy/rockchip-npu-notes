# Runtime 分层隔离实验

> **同实现修正（2026-10-27，后续 bounded experiment）：** 本文旧的
> B/C、C−D 和完整模型 direct-adapter 数字是历史控制，不再作为因果结论。
> B/C 使用了不同的 upload pack 实现（以及不同 finite-scan helper）；C−D
> 包含实验路径差异，不能全部归为必要成本；direct adapter 每次重建 ctypes
> 指针数组并使用节点代理，其回退只说明那种接线方式无收益。新的同实现
> prepared-validation A/B 见
> [RUNTIME-LAYER-PREPARED-VALIDATION.md](RUNTIME-LAYER-PREPARED-VALIDATION.md)
> 和 `results/20261027T093259Z-template-gap/runtime-layers-prepared/`。

日期：2026-10-27。目标是判断当前应用路径中 **Python/FFI、RKPlan 通用层、
custom-ops/provider、当前执行粒度** 各自增加了多少成本；这是隔离实验，
不是新的默认优化，也不是 validation/guard/publication batching 的实现。

## 0. 结论

1. **prepared Python → native C 的差距很小**：相同 plan、相同输入输出下，
   A1（Python prepared binding + legacy C）与 B（原生 C RKPlan）的 p10
   差值在 MLP b1/b32、QKV b8 上为 **0–5.5 µs**，在 QK/PV expanded group 上为
   **54–63 µs**。这不是外部路径差距的主因。
2. **generic Python group 路径确有可消除的重复验证**：A0（`run_group_into`）
   到 A1 的 p10 差在 QK/PV 为 **1331/1147 µs**，主要来自每次 call 对 96 个
   bindings 的 Python shape/finite 检查和 descriptor 处理。但当前 compiled
   graph 的 production 路径是 prepared binding，所以这项不能直接算作当前
   default 的固定成本；它解释了为什么旧 generic group 实验会显得很重。
3. **RKPlan 通用层在 expanded group 上确有成本**：B 到 C（direct custom-ops）
   的 p10 差为 QK **+367.5 µs**、PV **+205.9 µs**；attribution 显示主要是
   per-run binding/alias validation、action interpretation、command/sync
   编排。C 仍保留 finite input/output、pack、必要 sync、完整 guard 和 staged
   publication。MLP b32 和 QKV b8 没有该收益：C 甚至分别慢 **37.3/7.6 µs**，
   因为 direct 实现的 pack 在非 16×16 shape 上比 runtime 慢。
4. **不能在完整模型上把局部收益直接兑现**：Tiny ViT b8 的 targeted
   qk/pv direct-C integration（72 个 plan/Host action 中只替换两个 group call）
   保持 96 组输出逐位相同，但 app p10 从 **3954.7 → 4352.5 µs**，反而
   慢 **397.8 µs**。被替换的 qk/pv NPU node p10 合计节省 **95.1 µs**；
   差值在替换点之外的 Host action/executor/integration residual。
   同程序双 runner 对照无 arm 偏置（paired median **0.9991×**）。
5. **当前数据不支持重设计 runtime**：局部 group 收益真实，但不能自动传导到
   完整模型；C 的 generic pack 还会在 MLP b32/QKV 上反向抵消。没有证据支持
   立即投入新的 runtime 架构或关闭 guard/sync。应保持默认安全契约。

## 1. 固定条件与案例

| 项 | 值 |
|---|---|
| board | RK3588 OPi 5 Plus |
| kernel / driver | `6.1.43-rockchip-rk3588-rknpu098` / `v0.9.8` |
| NPU | 1 core，governor `performance`，1.0 GHz |
| DMC | governor `performance`，2.112 GHz |
| CPU | affinity 6，governor `performance`，2.304 GHz |
| final runtime | `librknpu_plan.so` SHA-256 `19b5263a137cda82eff215ca181d267923ca9645d4cca2378de290ab6dbcaa16` |
| final package | compiled manifest `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869` |
| numeric spec | `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2` |

首组案例：MLP b1、MLP b32、ViT QKV b8、QK expanded group b8、PV expanded
group b8。MLP 覆盖多提交驻留图和小/大 M；QKV 覆盖投影矩阵；QK/PV 覆盖
32 个小任务合成一个 submit 的 shared group。

## 2. 五条路径的精确定义

| 路径 | 执行方式 | 计时包含 |
|---|---|---|
| **A0** | Python generic `run_into` / `run_group_into` | Python shape/finite/binding 检查、ctypes、C runtime |
| **A1** | Python `bind_*_prepared_into` + 直接 legacy `workspace_run`/`workspace_run_group` | Python ctypes/调用包装、C runtime；无 per-call Python shape/finite 扫描 |
| **B** | 原生 C harness 直接 `rk_workspace_run` | 同 descriptor、BO/view、C validation、pack、emit、submit、guard、download、publication |
| **C** | 原生 C direct custom-ops | 同 descriptor/emitter/executor；跳过 per-run generic shape/alias validation；保留 finite、pack、sync、emit、submit、完整 guard、输出 finite 检查和 staged publication |
| **D** | resident execution probe | 相同已发射 programs/submit 序列；input prep 与结果验证在区间外；不含 upload/download/guard/publication |

A0/A1 是同一 Python plan-call 层的两个控制；**请求表中的 A 对应 A1**。
A0 用来把 Python 层的重复 shape/finite 验证单独暴露出来。A1 是实验性
internal 用法，不是新增公共 API。

C 的明确差异：

- setup 时用 `rk_plan_clone` 做一次完整 descriptor 验证；
- 热路径不再对固定的 binding 指针重复做 generic shape/dtype/rank/alias
  检查；
- 保留输入 finite 扫描、输入 upload pack、BO prep/fini、`rk_emit_action`、
  相同 submit 分组、所有 slot 的 `co_guard_check`、每个 output tile 的
  finite 检查、staged publication；
- 16×16 CUBE/WEIGHT upload 使用与 runtime 相同的 fast-pack；其他 shape 走
  direct generic pack，因此 C 在 MLP b32/QKV 上不是优化后的 pack 实现。这个
  差异在结果中单独显示，不隐藏。

D 只回答“设备驻留提交与执行”的参考下界；它**不是纯 NPU kernel 时间**，
也不是应用端到端性能。

## 3. 主结果

协议：7 个 outer rounds；每轮旋转 case 和 path 顺序；每个 process 使用相同
的 changed-input 帧序列。下表是 warm samples 的 **p10**，用于避开设备
clock/thermal 的慢状态；中位数和范围在 `analysis.json` 中。

| case | A0 | A1 | B | C | D | A0−A1 Python | A1−B FFI | B−C runtime | C−D data/boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 477.5 | 427.6 | 427.6 | 420.9 | 228.1 | +49.9 | +0.0 | +6.7 | +192.8 |
| MLP b32 | 767.1 | 655.7 | 650.1 | 687.5 | 240.3 | +111.4 | +5.5 | −37.3 | +447.1 |
| QKV b8 | 246.7 | 186.1 | 182.0 | 189.6 | 54.0 | +60.7 | +4.1 | −7.6 | +135.6 |
| QK group b8 | 2228.3 | 897.2 | 834.2 | 466.7 | 179.4 | +1331.2 | +63.0 | +367.5 | +287.3 |
| PV group b8 | 1861.4 | 714.9 | 660.6 | 454.7 | 184.6 | +1146.5 | +54.2 | +205.9 | +270.1 |

`B−C` 为正表示 direct custom-ops 更快。A1−B 在 MLP b1 上的 0 是稳定值
附近；不是“FFI 免费”的一般结论。

各自解释：

- **MLP b1**：B/C 基本重合，说明 runtime 通用层不是主因；D 到 B 的约
  200 µs 是输入 upload/guard/download/publication 等仍必要的边界。
- **MLP b32**：B/C 的差异是 C **更慢** 37 µs；B profile 的 pack 156 µs，
  C upload 254.5 µs。小 M padding 和 guard/submit 仍占主要部分，不能从
  runtime 通用层解释。
- **QKV b8**：A1→B 仅 4.1 µs；B/C 的 C 慢 7.6 µs，同样来自 direct pack
  （C upload 67.5，B pack 43.5）。runtime 没有可兑现的通用层收益。
- **QK/PV group b8**：A1→B 为 63/54 µs，B→C 为 367/206 µs；这是唯一
  稳定指向 group runtime 通用层的信号。C 的收益来自避免 per-run expanded
  group validation/interpreter 编排，而不是设备执行加速。

## 4. setup、首次执行、warm

完整表在 `analysis.md`；典型值：

- MLP b1：B setup 4.21 ms，C 4.78 ms，D 4.93 ms；A1 setup 5.35 ms。
  B/C first 577.8/499.3 µs，warm 已到 428/421 µs。
- QK group b8：B setup 0.99 ms，C 0.79 ms，D 1.25 ms；A1 setup 4.84 ms。
  first 1.09/0.70/0.28 ms，warm p10 834/467/179 µs。
- QK/PV A1 binding preparation 约 1.0 ms，必须与 steady call 分开。
- 首调用普遍高于 warm；setup 不能拿来和外部 warm path 相比。

## 5. 独立 phase 采样

B 是 `rk_workspace_run_ex` 的 96-call attribution 中位数；C 是 direct C 在
32 个 changing frames 上除算的独立样本。**不靠不同分项中位数相加得到主结论。**

| case | B validate | B pack | B emit | B command | B submit/wait | B output-sync | B download | B guard | B publish | C finite | C upload | C emit | C execute+sync | C download | C guard | C publish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 2.3 | 11.4 | 14.3 | 30.3 | 175.3 | 27.7 | 2.6 | 163.3 | 0.0 | 0.4 | 14.5 | 14.2 | 237.7 | 2.5 | 163.8 | 0.1 |
| MLP b32 | 68.0 | 156.0 | 14.3 | 28.6 | 184.9 | 30.0 | 3.8 | 161.0 | 0.3 | 10.8 | 254.5 | 14.3 | 250.7 | 3.5 | 161.0 | 0.1 |
| QKV b8 | 17.2 | 43.5 | 1.5 | 2.0 | 48.7 | 5.0 | 35.6 | 26.5 | 1.8 | 2.7 | 67.5 | 1.4 | 56.7 | 25.6 | 26.2 | 1.8 |
| QK group b8 | 423.5 | 39.7 | 46.2 | 23.9 | 97.7 | 94.5 | 112.9 | 24.8 | 1.2 | 65.2 | 25.9 | 46.5 | 189.4 | 105.5 | 24.6 | 1.2 |
| PV group b8 | 236.8 | 45.2 | 45.6 | 24.8 | 60.1 | 98.6 | 116.4 | 25.4 | 1.2 | 35.3 | 34.2 | 45.4 | 190.1 | 109.9 | 25.2 | 1.1 |

QK/PV 的 B `validate` 为 423.5/236.8 µs；direct C 仍支付 finite 65/35 µs
和 guard 25 µs，但不再支付 generic shape/alias/解释编排。MLP b1 的
C guard 163.8 µs 与 B 163.3 µs 几乎相同，证明 C 没有靠删除 guard 获得收益。

## 6. 完整模型传导探针

Tiny ViT b8，256 张图，batch 8，3 rounds；所有 Host action 和其他 plan
保持当前 compiled graph，只把 qk/pv group call 替换成 native-C direct
context。该实验是完整模型里的 **targeted subgraph integration**，不是全 C
模型实现。

| arm | app median µs | app p10 µs |
|---|---:|---:|
| current compiled graph | 4002.6 | 3954.7 |
| qk/pv direct-C substitution | 4415.8 | 4352.5 |

paired current/variant：median **0.9050×**，p10 **0.8820×**；96 组输出逐位
相同。被替换的 NPU node p10：qk 555.9→527.9（省 28.0），pv 586.5→508.4
（省 78.2）；全部被测 NPU node 合计省 **95.1 µs**。但 app p10 慢 **397.8 µs**，
替换点之外的 measured residual 约 **+492.9 µs**（Host action/graph/output/
integration），不是由两个 group call 本身解释。

同程序双 runner 对照（无 adapter）：paired median **0.9991×**，p10
**0.9904×**，bitwise equal；因此不是简单的 A/B runner 位置偏置。

**判断**：局部 group 收益不能按“省 367+206 µs”直接加到模型；当前
integration shape 下完整模型反而更慢。一个完全 native C、单 context、
单一 Host-action/lifetime 实现的模型是另一个实验，现有数据不足以授权该
重设计。

## 7. 与外部路径差距的关系

固定频率下的外部参考仍是不同数值政策，不能做同政策排名：

| profile | current delivered µs | vendor µs | CPU FP32 µs |
|---|---:|---:|---:|
| MLP b1 | 599.1 | 258.8 | 252.6 |
| MLP b8 | 716.6 | 309.9 | 1748.9 |
| MLP b32 | 1127.6 | 538.0 | 6830.5 |
| ViT b8 | 4010.7 | 1582.1 | 5856.7 |
| ViT b1 | 1532.4 | 589.4 | 869.5 |

本轮能确认的是：

- prepared Python/FFI 在四个首组案例上的额外成本约 **0–63 µs**，不足以解释
  vendor 的数百 µs 到 ms 级差距；
- QK/PV expanded group 的 RKPlan generic cost 约 **206–368 µs**，但完整模型
  targeted integration 没有把它变成端到端收益；
- MLP b1/b8 的固定 12-row physical floor、MLP b32 的 guard/pack/submit
  envelope，以及 QK/PV 的 validation/download/guard 包络，属于执行粒度/
  安全边界，不是单纯 Python 或 RKPlan 解释器成本；
- vendor 内部 Submit 数、per-kernel 时间和 fusion body 仍未知，不能把剩余
  差距归因成某个已观测组件。

## 8. 可消除工作与契约成本

| 工作 | 结论 |
|---|---|
| Python generic group 的重复 shape/finite 检查 | A0→A1 可消除；但 production compiled graph 已走 prepared，不是当前固定主因 |
| expanded group 的 per-run generic shape/alias validation | B→C 显示可省；需要静态 proof 和新的安全契约，未实现 |
| finite input scan | 必须保留；C 仍支付 |
| input pack + BO prep/fini | 必须保留；C 的 generic pack 在 MLP b32/QKV 更慢，不能直接替换 runtime |
| guard（MLP b1/b32 约 160 µs） | 安全契约必须承担；不能关闭 |
| output finite + staged publication | 必须保留；C 保留并单独计时 |
| command + submit/wait + output sync | 设备/驱动边界；D 仍为 179–240 µs，不能算 runtime 通用层 |
| setup/first | 与 warm 分开；QK/PV A1 binding 约 1.0 ms，不能摊入稳态 |

## 9. 决策

**不重设计 runtime。** 不实施 validation/guard/publication batching，不默认
关闭 guard/sync，不改变设备 FFN opt-in 或任何数值政策。当前更重要的是执行
粒度/设备路径，而不是把 Python/FFI 或 RKPlan 通用层当成全部差距来源。

若后续仍要推进 runtime，只有一种值得重新立项的窄方向：**单 context、与当前
workspace/command path 共享的 direct-group executor + 静态 binding/lifetime
证明**；必须先做一个完整模型实验证明局部收益不被 Host action/executor
residual 抵消。本轮的 adapter 已经给出了反例，不能直接产品化。

## 10. 证据

- 原始 main samples：`results/20261027T093259Z-template-gap/runtime-layers/r1..r7/`
- Python/C profile：`runtime-layers/profile/`
- 完整模型：`runtime-layers/full-model-final/`
- 机器可读分析：`runtime-layers/analysis.json`
- 案例 descriptor/config：`runtime-layers/cases/`
- 可执行 harness：`tools/runtime_layers/runtime_layer_probe.c`，
  `tools/runtime_layers/benchmark_python_layers.py`，
  `tools/runtime_layers/full_model_direct_group.py`
- build 产物 hash：`runtime-layers/artifacts/` 与 `runtime-layers/campaign-manifest.json`
