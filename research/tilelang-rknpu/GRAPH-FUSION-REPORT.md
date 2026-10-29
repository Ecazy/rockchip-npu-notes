# TinyTransformer 图级调度、设备驻留与融合报告

日期：2026-10-28
主验收：TinyTransformer b1/b8，RK3588，core mask 1，NPU 1 GHz，CPU affinity 6
通用回归：MNIST MLP 完整 10,000 图；SmolLM2-135M-Instruct 冻结验收
基线源码：`dca0fc7441c773460f445b4ef67e2eba060fac71` + 工作区既有 P0–P3 布局复用修改
新包：

- `models/compiled_graph_fusion/`：默认 profile，含 `_deviceffn_fused`、`_addfused` 显式候选；默认 `tiny_vit_bN` 不改。
- `models/compiled_rmsnorm_lut/`：独立 profile + `_rmsnorm_approx` 近似全设备 RMSNorm 实验包。

## 1. 结论

### 已消除

1. **依赖连续的混合模板任务链可合并为一个 Submit。** 新增 planner 的 `fuse_task_chain`（含 Host upload 前移、依赖/action 顺序/写区间检查）和 C verifier 的 dependency-chain admission。它将 device FFN 的 7 个 task 从旧的 5 个 Submit 收敛为 **1 个 Submit**，task 数、buffer、数值和传输完全不变。
2. **旧 device FFN 的 b1 回归被消除。** 同数值政策下，b1 `0.963→0.875 s`（−9.1%），b8 `0.212→0.209 s`（−1.6%）；相对默认 Host FFN 路径，b1 `0.952→0.858 s`（−9.9%），b8 `0.239→0.210 s`（−12.1%），多轮 paired 中位 ratio 分别为 `0.904` 和 `0.875`。
3. **残差 Add 的设备路径已实现且逐 bit 等价。** `attn_out+residual`、`down+residual` 通过现有 `co.ew_add.f16.v1` 直接产生下一消费者输入；b1/b8 与对应 Host FP32 add→FP16 路径在 3 轮 × 512 图上逐 bit 相同。该候选本身因额外 residual upload 和 task 数增加，b1 慢约 4.9%、b8 慢约 1.4%，**不采用为默认**。
4. **SwiGLU 设备链已经验证为 layer 2/3。** gate/up MM、LUT SiLU、门控乘、down MM、residual Add 全在一份设备 plan 内；中间值有明确 buffer 和设备任务依赖。它不是“只把 LUT 放到 NPU”的混合原型。
5. **RMSNorm 全设备近似路径已实现为独立实验。** 新增通用 `co.lut.rsqrt.f16.v1` 模板和 `rmsnorm_approx_module`：平方 → 设备行归约 → 均值/eps → 缩放后的 bounded rsqrt LUT → gamma/行系数展开 → 乘。7 个 task 可成一个 Submit，Host 不读取激活或统计量。该路径数值降级且 b1/b8 都更慢，**不采用**。

### 仍保留 / 未完成

- **RoPE 仍是混合原型。** `co_rope.c` 当前会读取设备激活、在 CPU 构造 `rot/cos/sin` 再上传；它不是 layer 2。没有把它接入 RKIR 默认路径。具体最小实验和后继能力见第 7 节。
- **稳定 Softmax 仍必须 Host。** 当前 policy 是 FP32 max/subtract/exp/sum，再 F16 写回；没有合格设备稳定归约路径。
- **RMSNorm 的默认 Host FP32 路径未替换。** 新设备路径为独立近似 policy，精度和速度均不优于默认。
- **中间 DDR 写回没有因 Submit 合并而消失。** device FFN 的 gate/up/silu/mul/down/residual 仍写各自设备 buffer；合并只减少 Host transition、Submit/命令 envelope 和 Host 中间边界，不宣称片上驻留。
- **CORE_CTRL=0x000f 单 descriptor 拼接未采用、未测试。** 现有 task chain 已足以取得本轮收益，没有证据支持替换默认链协议。

## 2. 当前 TinyTransformer 默认路径边界表

默认 `tiny_vit_b1` / `tiny_vit_b8` 每 call 均为 8 个 NPU Submit；task 数 b1=14、b8=70。下表是当前默认路径，不是 fused-FFN 候选。

| 节点 | 执行位置 | producer → consumer / 布局 | 当前边界原因 |
|---|---|---|---|
| patchify_norm | Host | raw → patches F16 | 数值语义 + 物理 raster + 实现 |
| patch_pad | Host | 49→64 zero pad | 物理布局 + 实现 |
| patch projection | NPU plan | patches → projection | 1 Submit/1 task |
| position_add | Host | projection + pos → positioned | 缺少设备 broadcast add |
| rmsnorm1 | Host | positioned → norm1 | 缺少完整设备行归约/rsqrt |
| qkv projection | NPU plan | norm1 → qkv | 1 Submit/1 task |
| QKV split | Host/capture-only | qkv → q/head layout | 物理布局；生产路径 trace-only |
| QK group | NPU plan | q/k slices → scores | 宽度/batch 独立 task 已合 1 Submit |
| stable softmax | Host | scores → probs F16 | FP32 stable softmax 数值政策 |
| PV group | NPU plan | probs/V slices → head contexts | 独立 task 已合 1 Submit |
| merge_heads | Host | head-major → token-major | 物理布局 |
| attention O | NPU plan | merged → attn_projection | 1 Submit/1 task |
| residual_attention | Host | positioned + attn_projection → h1 | 当前默认仍 Host；设备 Add 已有候选 |
| rmsnorm2 | Host | h1 → norm2 | 缺少完整设备行归约/rsqrt |
| gate_up | NPU plan | norm2 → packed gate/up | 1 Submit/1 task |
| SwiGLU | Host | gate/up → act | 默认 Host FP32 sigmoid/product |
| down | NPU plan | act → ffn | 1 Submit/1 task |
| residual_ffn | Host | h1 + ffn → h2 | 当前默认仍 Host |
| reshape_tokens | metadata view | h2 → [B,T,D] | 已消除 per-call Host copy |
| mean_pool | Host | h2_tokens → pooled | 缺少设备 FP32-preserving token reduction |
| classifier | NPU plan | pooled → logits_padded | 1 Submit/1 task |
| export_logits | Host | F16 padded → F32 [B,10] | 公共接口/资源边界 |

这些原因不是同一种成本：Host 数值计算、物理布局 copy、缺失模板、资源接口和实现选择必须分开。Softmax 不是唯一硬边界；RMSNorm、position Add、SwiGLU、mean pool 和若干布局转换同样保留。

## 3. 代码与接口改动

### 3.1 通用依赖任务链

- `CompileOptions.fuse_task_chain`：显式 opt-in。
- `passes/schedule.py`：
  - `_hoist_host_uploads` 将所有 upload 移到第一个 device task 前，并重建 action id/deps/buffer live range；
  - `_fuse_task_chain_submits` 只合并连续 action、依赖拓扑、串行 Submit 且 task 数不超 ABI 上限的组。
- `plan.py` 新增 `dependency_chain_v1` recipe。
- `rk_plan_validate.c` 新增独立 admission：task id 连续、所有 NPU 依赖在更早 task 或更早 Submit、至少一个非 MM task、任务写区间两两不重叠；原有全局 read-after-write、alias 和 publish 检查仍运行。
- 不改变任何 arithmetic、buffer、upload/download、数值或同步；`rk_workspace` 仍执行同一 C `CO_CHAIN` seam。

### 3.2 device FFN candidate

`models/tilelang_matmul.py::make_swiglu_ffn_f16` 经 `compile_tilelang(..., fuse_task_chain=True)` 生成：

- 7 tasks：gate MM、up MM、sigmoid LUT、silu mul、gate×up、down MM、residual EW_ADD；
- 1 Submit、8 次 upload/download action（其中权重为 create-time 常量）；
- 与旧 device FFN control **逐 task、逐 buffer、逐传输相同**；只有 Submit envelope 不同。

manifest：`tiny_vit_bN_deviceffn_fused.rkhybrid.json`，显式 suffix `_deviceffn_fused`。

### 3.3 device residual Add

新增通用 `attn_out_add`、`down_add` plan：

- `matmul_module(..., residual="add")` 产生 MM + EW_ADD；
- `fuse_task_chain` 将两者合为 `linear_epilogue_v1` 一个 Submit；
- 残差是真实 device input view，不是把普通输出地址当 EW operand；消费者在 plan 内等待 producer task。

manifest：`tiny_vit_bN_addfused.rkhybrid.json`，显式 suffix `_addfused`。

### 3.4 全设备 RMSNorm 实验

新增 capability **只存在于独立 profile** `capabilities/rk3588_vendor_co_v1_rmsnorm_lut.json`，默认 profile 不含 rsqrt。

- `co.lut.rsqrt.f16.v1` 模板 → `co_emit_lut(..., CO_RSQRT_BOUNDED)`；
- `rmsnorm_approx_module(m,d)`：
  1. `Sq = X*X`（FP16 EW MUL）；
  2. `Stats = Sq @ ones[1,D]`（FP16 MM 单 tile 行和）；
  3. `Mean = Stats * (s/D)`；
  4. `MeanEps = Mean + s*eps`；
  5. `Scale = rsqrt_lut(MeanEps)`；
  6. `Coeff = Scale @ gamma_weight[D,1]`（gamma 和输出修正 `sqrt(s)` 折叠进常量）；
  7. `Y = X * Coeff`。
- 7 tasks 合 1 Submit；Host 只准备常量/启动 plan，不读取激活或统计量。
- 这是 **layer 2/3 for RMSNorm itself**，但是独立近似数值 policy；不沿用默认 FP32 RMSNorm 门槛。

## 4. 性能结果

### 4.1 默认 vs fused device FFN

固定 512 张 MNIST/ViT 输入、b1/b8 两个静态 profile、交错 5 轮；默认与 `_deviceffn_fused` 同 runtime、同权重、同 core mask/affinity。

| profile | 默认 median | fused median | 变化 | 默认 mean | fused mean | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny ViT b1 | 0.9517 s | 0.8576 s | **−9.9%** | 0.9247 s | 0.8329 s | −10.0% |
| Tiny ViT b8 | 0.2390 s | 0.2100 s | **−12.1%** | 0.2391 s | 0.2086 s | −12.7% |

5 轮 paired ratio：

- b1: `0.782 / 1.049 / 0.957 / 0.835 / 0.904`，中位 `0.904`；
- b8: `0.821 / 0.911 / 0.938 / 0.824 / 0.875`，中位 `0.875`。

b1 有一轮 fused 反而略慢，b8 五轮全部下降。
5 轮输出都与默认路径 top-1 全一致；最大 raw-logit 差 `0.03125`。这不是逐 bit 的默认路径替代，而是独立 LUT 数值 policy 候选。

### 4.2 旧 device FFN vs 合并 Submit

同 3 轮控制 campaign 中，task、upload/download 完全相同的两个 device FFN arm：

| profile | 旧 device FFN median | fused median | 变化 | task/call | submit/call |
|---|---:|---:|---:|---:|---:|
| b1 | 0.9627 s | 0.8751 s | **−9.1%** | 19 | 11 → **7** |
| b8 | 0.2125 s | 0.2092 s | **−1.6%** | 75 | 11 → **7** |

`fused` 与旧 device FFN 的 logits 在 3 轮 × b1/b8 全部 **逐 bit 相同**；因此该差值可单独归因于 task-chain envelope 消除，而不是数值或传输变化。

### 4.3 Add fused 与 RMSNorm approx

| arm | b1 median | b8 median | 结论 |
|---|---:|---:|---|
| Add 对照：默认 Host FFN | 0.9526 s | 0.2388 s | baseline |
| `_addfused`（同 policy） | 0.9996 s | 0.2421 s | 比各自对照组慢 4.9% / 1.4%；逐 bit 等价，额外 upload/task，不采用 |
| RMSNorm 对照：该实验包默认 Host FFN | 0.9359 s | 0.2275 s | baseline |
| `_rmsnorm_approx`（独立 policy） | 1.0888 s | 0.2683 s | 比各自对照组慢 16.3% / 17.9%；全设备但不采用 |

`_addfused` 在 3 轮 b1/b8 与默认路径逐 bit 相同。
`_rmsnorm_approx` 与默认最大 logit 差 `0.078125`、top-1 在 512 图仍为 `1.0`，但数值降级且无性能收益，不进入默认。

全 10,000 图单轮补充测量（该实验包默认 vs `_rmsnorm_approx`）：

| profile | 默认 e2e | approx e2e | 默认/approx accuracy | top-1 agreement | approx max logit error |
|---|---:|---:|---:|---:|---:|
| b8 | 4.602 s | 4.944 s（+7.4%） | 0.9821 / 0.9823 | 0.9998 | 0.4082 |
| b1 | 17.871 s | 19.106 s（+6.9%） | 0.9821 / 0.9823 | 0.9998 | 0.4082 |

因此 b1/b8 均无速度收益且最大 logit 误差显著增大；按“b1/b8 分别选择、新数值路径不静默替换默认”的原则，不推荐采用。

RMSNorm 还补了独立的 **operator-level 数值验收**（`models/rmsnorm_approx_acceptance.json`）：

> 旧记录说明：本节最初版本在同一次进程中先跑 calibration 和 acceptance，再写 spec；
> 它不能称为“先冻结、后验收”。该边界已在 §16 修正为两个独立进程，旧记录保留但本节
> 的“冻结/验收”措辞以 §16 为准。

- 校准与验收使用不同 seed、不同随机行集，并强制 mean-square 端点 `0.17`、`6.10`；
- 设备 plan 的输入域为 scaled LUT 域 `[0.125,6.32]`，对应原始 mean-square `[0.1667,8.43]`；
- 校准 `max_abs=0.01953`、`rmse=0.001247`；
- 验收 `max_abs=0.02344`、`rmse=0.001223`；
- 冻结 gate `atol=0.0390625`、`rmse_limit=0.009765625`，验收 PASS；
- 该 gate 只属于 `_rmsnorm_approx` 实验 policy，不用于默认 FP32 Host RMSNorm。

### 4.4 计数与内存

按 evaluate campaign 的实际汇总除以调用数：

| profile | arm | task/call | submit/call | upload B/call | download B/call | Host action µs/call | host residency B |
|---|---:|---:|---:|---:|---:|---:|---:|
| b1 | default | 14 | 8 | 38,400 | 29,440 | 530 | 63,208 |
| b1 | fused | 19 | **7** | 36,352 | 21,248 | 464 | 55,016 |
| b8 | default | 70 | 8 | 296,448 | 230,144 | 1,420 | 505,664 |
| b8 | fused | 75 | **7** | 280,064 | 164,608 | 1,037 | 440,128 |

- b1/b8 每 call 少 1 个 Submit，但多 5 个 task；device task 没有被隐藏或合并成“假 task”。
- upload 字节约降 5.3%/5.5%，download 约降 27.8%/28.5%；原因是 device FFN 少了 Host swiglu/residual 边界及其临时物化。
- host residency 降约 13%。
- `plan_guard_us` 在 b1/b8 反而略升（因为 task 数增加），不能把 guard 成本说成下降。
- 合并 Submit 没有减少 device FFN 内部的中间 DDR 写：gate/up/silu/mul/down 的 buffer 和 task 依赖仍存在。减少的是 Host transition、Submit/命令 envelope、中间 Host 物化和对应传输。

### 4.5 全 10,000 图

`compiled_graph_fusion` + `_deviceffn_fused`：

| profile | n | accuracy | declared top-1 agreement | max logit error vs reference |
|---|---:|---:|---:|---:|
| Tiny ViT b8 | 10,000 | **0.9821** | 1.0000 | 0.04180 |
| Tiny ViT b1 | 10,000 | **0.9821** | 1.0000 | 0.04180 |

默认完整模型回归同 campaign：MLP b32 全 10,000 图 `0.9831`，Tiny ViT b8 全 10,000 图 `0.9821`；`run-model-admission-final2.log` PASS。

## 5. 能力矩阵

| 算子 | 原实现 | 新实现 | 消除 Host 数值边界 | 设备直接消费 | 数值政策 | b1/b8 或 Smol 收益 | 状态 |
|---|---|---|---|---|---|---|---|
| 残差 Add | Host FP32 add → FP16 | `co.ew_add` + `linear_epilogue_v1`；device FFN 内 residual | 是（该 Add 本身） | attention Add 后接 Host RMSNorm；FFN residual 后设备 plan 结束 | 与 Host 路径逐 bit 相同（验证） | `_addfused` b1 −4.9%、b8 −1.4%（更慢） | A：可同 policy，未采用 |
| SiLU | Host FP32 sigmoid+product | device sigmoid LUT + EW MUL | 是（device FFN 子图） | 是，mul/down 同 plan | 独立 staged-F16 LUT policy | b1 −9.9%、b8 −12.1%（vs 默认） | B：显式候选 |
| GELU | 未用于 Tiny；LUT `0.5*(1+erf(v/√2))` | 既有 `co.lut.gelu_gate`，未新接模型 | 未在本模型验证 | 未接 | 既有 LUT，erf 语义 | 无本轮模型收益 | 未采用/未接入 |
| SwiGLU | Host gate/up split + FP32 silu×up | gate/up MM → LUT SiLU → EW MUL → down → Add | 是（device FFN 子图） | 是，全在同一 1-Submit plan | 独立 LUT policy；与旧 device FFN 逐 bit 相同 | 见 SiLU 行；旧 device FFN 的 b1 回归消失 | B：推荐显式采用 |
| RoPE | Host CPU reads activation, computes rotate + cos/sin, uploads 3 buffers (`co_rope.c`) | device P reorder；Smol 32-step opt-in 集成；prefill 常量驻留；decode 位置 one-hot gather 常驻表（runtime 与 indexed 两种模型路径都验收） | 是（prefill 模型路径；decode 设备 gather，Host 只传 one-hot） | 是（plan 内） | FP16 P/EW；连续 32-step、runtime/indexed 两套候选 policy | 生产 Host 对照：prefill +9.7~13.6%、32-step decode +18.9~19.2%，更慢 | B/D：连续集成/位置索引成立，性能不采用 |
| RMSNorm | Host FP32 mean/rsqrt；`co_norm_hybrid` 仍下载统计量 | 7-task full-device approximate chain + `co.lut.rsqrt` | 是（新实验路径） | 是（plan 内） | 独立 FP16 square/reduce + scaled bounded LUT；独立 operator acceptance PASS | 512 b1 +16.3%/b8 +17.9%；全 10k b1 +6.9%/b8 +7.4%，误差增大 | B/D：实现，未采用 |
| Softmax | Host FP32 stable max/exp/sum → F16 | 无 | 否 | 否 | 当前冻结 policy | 保持 Host | 未采用 |
| QKV→QK | qkv CUBE → Host split/K transpose → Q/K plans | 已测 CUBE/WEIGHT view + 固定 p0312 不匹配；PPU 无 transpose 寄存器、runtime 无 device copy；新命令契约已定义 | **否** | 否 | 当前 layout | b8 QK upload ~98 KB/call，未消除 | C：已测路径不匹配，不能称全部已有路径不可行 |
| PV→O | PV group → Host merge → O plan | 无 | 否 | 否 | 当前 layout | 保持 Host merge | 需要可变外部 BO/view 或跨 plan 驻留 |

## 6. 正确性与回归

- `tests/test_graph_fusion.py`：
  - 混合模板 FFN 链 1 Submit/7 task，旧 device FFN vs fused 逐 bit；
  - `attn_out+add`、`down+add` 与 Host FP32 add→FP16 逐 bit；
  - 默认 profile 不含 rsqrt，实验 profile 包含；
  - 全设备 RMSNorm 7 task/1 Submit，域内误差 `max<=0.01`、RMSE `<=0.003`。
- 全量 Host：`pytest -q tests` 通过；`ctest` 33/33 通过。
- TinyTransformer：3 轮 × 512 图 paired 输出 top-1 全一致；Add fused 逐 bit；fused device FFN 与旧 device FFN 逐 bit。
- 全 10,000 图：fused b1/b8 accuracy `0.9821`，declared top-1 agreement `1.0`。
- MLP/Tiny 默认完整模型 admission：PASS；MLP b32 full 10k `0.9831`，ViT b8 full 10k `0.9821`。
- SmolLM2：冻结 acceptance `pass=true`，56 gates 通过；incremental top-1 equality 保持；teacher-forcing 首次分歧位置 6 仅作质量统计。
- 失败/隔离：新链 admission 保留原 NPU alias、read-after-write、publish 和 lifecycle 检查；RMSNorm 实验在独立 profile，默认 profile 仍拒绝 rsqrt capability。没有关闭 guard 或删除同步。

## 7. RoPE 结论与最小后继实验

当前 `research/custom-ops/src/co_rope.c` 的路径明确是混合：

1. `rocket_bo_prep` 后从设备 buffer 读 `input`；
2. CPU 生成 `cos/sin`、构造 `rot`；
3. 用 `co_upload_f16` 上传 `rot/cos/sin`；
4. NPU 只执行 `x*cos + rot*sin` 的 3 个 EW task。

因此不能声称消除了 RoPE Host 数值边界。它仅存在于旧 graph/attention seam，不进入当前 RKIR 默认路径；TinyTransformer 也不使用 RoPE。

最小后继实验应当是：

- 预计算并驻留 `[capacity, heads*head_dim]` 的 cos/sin 表；
- 用 block-diagonal 符号置换矩阵 `P`（每 head 64×64，rotary_dim 内配对，非 rotary 行为单位）通过现有 `co.mm.f16.v1` 做 `rot = X @ P`；
- 用 3 个 EW task 做 `x*cos + rot*sin`；
- 增加一个 position-indexed device view/选择能力，或把 QKV projection 与 RoPE 编进同一 plan，避免每 layer 每 token 重新上传系数；
- 对 partial rotary、split-half/interleaved、位置边界、跨 head/tile 和 prefill/decode 分别验收。

所需缺口是“设备位置索引/常量行选择 + 多 head grouped plan”，不是 LUT。未完成集成前不把 RoPE 算作设备驻留或“仅剩 Softmax”。

## 8. 默认、实验与未采用

- **默认路径**：保持 `models/compiled/tiny_vit_bN.rkhybrid.json` 的 Host FFN 政策；本轮不静默替换数值政策。
- **推荐显式候选**：`_deviceffn_fused`。b1/b8 均有中位收益，b8 五轮全下降，b1 4/5 下降；若产品接受 LUT/staged-F16 policy，可按 batch profile 显式选择，MLP/其他模型 fallback 默认路径。
- **同 policy 但不值得采用**：`_addfused`（逐 bit 正确但更慢）。
- **数值实验、不采用**：`_rmsnorm_approx`（全设备但更慢且误差更大）。
- **未实现/阻塞**：RoPE 全设备路径；QKV→QK 设备 transpose/reorder；PV→O 跨 plan device residency；QKV→QK 仍需真实 weight-layout operand。
- **未执行**：CORE_CTRL=0x000f / 单 descriptor 拼接；没有充分理由替换已验证的 PC-chain。

## 9. 证据与重放

证据根目录：`results/20261028T113459Z-graph-fusion/`

| 内容 | 路径 |
|---|---|
| 5 轮 b1/b8 paired | `board-gf-pair-512/` |
| 3 轮 default/old deviceffn/fused/addfused | `board-gf-final-512/` |
| fused 全 10k | `board-gf-full2/` |
| RMSNorm 2 轮 512 | `board-rmsnorm-512-domain/` |
| RMSNorm 全 10k 单轮 | `board-rmsnorm-full-domain/` |
| 汇总 | `analysis-fusion.json` |
| 源码/库/包身份 | `final-source-identity.txt` |
| 包 manifest 身份 | `packages/SHA256SUMS` |
| MLP/Tiny 默认完整 admission | `run-model-admission-final2.log`，`results/20261028T123126Z-models/` |
| Smol acceptance | `smollm-acceptance/`、`smollm-verify.json` |
| Host 测试 | `pytest-stage2-final.log`、`ctest-stage2-final.log` |
| AB/BA paired | `abba-ab/`、`abba-ba/`、`abba-nab/`、`abba-nba/`、`abba-dab/`、`abba-dba/` |
| RMSNorm operator acceptance | `rmsnorm-operator-board-domain.json`、`models/rmsnorm_approx_acceptance.json` |
| RoPE device probe | `rope-device-interleaved-board.json`、`rope-device-split32-board.json` |
| PV→O device probe | `pv-o-device-board.json` |
| Stage-2 fused full 10k | `stage2-fused-full/` |
| Stage status | `STAGE-STATUS.json` |
| Continuous 32-step RoPE | `smollm-device-rope-cont-p0.json`、`smollm-device-rope-cont-p1-contpolicy.json`、`smollm-device-rope-indexed-cont-p0.json`、`smollm-device-rope-indexed-cont-p1-policy.json` |
| Decode position-indexed RoPE | `smollm-rope-indexed-board.json` |
| Continuous policy | `models/smollm2_device_rope_policy_cont32.json` |
| RoPE 源码证据 | `../../custom-ops/src/co_rope.c`、`../../custom-ops/include/co_rope.h` |

关键包身份：

- 默认冻结：`models/compiled/manifest.json` = `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- graph-fusion：`models/compiled_graph_fusion/manifest.json` = `657293883cd6d33dfde22be2f4f71c0709a83b6159aff7c1a5677b948082a554`
- RMSNorm 实验：`models/compiled_rmsnorm_lut/manifest.json` = `7a2ffaaf19d223c8f6030efee641952f12a7ec89b38eea27f5500b82e2e86c35`

重放命令见 [REPLAY.md](REPLAY.md#graph-fusion--device-residency-campaign-2026-10-28)。

## 10. RKNN 对照口径

本轮没有新增同口径 RKNN/RKLLM 对照。此前 MLP/Tiny 的 vendor 数字来自另一条执行路径；
`rknn_run` 调用次数不能换算成其内部 Submit 数，也不能与本报告的 RKPlan Submit/task
计数直接相减。因此本报告只声明自研路径内部、同输入同 core mask 的 paired 变化，
不把本轮结果表述为“缩小了多少 RKNN 差距”。若要扩展到 vendor 对照，必须重新固定
转换/加载、输入处理、同步和输出转换的计时边界，并确认双方完成相同计算和数值政策。

---

# 第二阶段补充（阶段性完成，不关闭整体任务）

本阶段保留上一轮 FFN 收益和原始证据，默认路径仍不调整。新增内容如下。

## 11. Mixed-template chain 准入收紧

Python `_fuse_task_chain_submits` 现在不是通用的 “依赖都在前面就合并”，而是：

- 只接受精确模板链前缀，最终必须命中 `ALLOWED_DEPENDENCY_CHAINS`；
- 候选 action id 必须连续；
- 每个 task 的 NPU dependency 必须来自候选内更早 task 或更早 Submit；
- 每个 device 输入 buffer 必须有更早 producer，且不能依赖同链后续写者；
- 候选输出 buffer 必须 SSA 唯一，写区间两两不重叠；
- 若只合并出合法链的前缀而非完整链，回退为原始 Submit，不发布半成品链。

C verifier 同步只接受当前三条精确序列：

1. staged SwiGLU FFN：`MM,MM,LUT_SIGMOID,EW_MUL,EW_MUL,MM,EW_ADD`；
2. approximate RMSNorm：`EW_MUL,MM,EW_MUL,EW_ADD,LUT_RSQRT,MM,EW_MUL`；
3. device RoPE reorder：`MM,EW_MUL,EW_MUL,EW_ADD`；
4. 以及 PV→O 最小实验：`MM×8,EW_ADD×3`（见第 13 节）。

负例已加入 `tests/test_graph_fusion.py`：

- 非连续 task id 拒绝；
- 模板序列不在 allowlist 拒绝；
- 写区间重叠拒绝；
- QKV 设备产物直接作为 QK 的 WEIGHT operand 拒绝，错误为
  `matmul weights must be runtime constants or inputs, not device-produced values`。

因此本轮不是“任意 mixed-template 链都能合并”。

## 12. AB/BA 配对：调度收益与新数值路径收益分开

固定 512 图、b1/b8、同 runtime/权重/core mask/CPU affinity。每个比较都执行 AB 5 轮 + BA 5 轮，共 10 个 paired observation。

| 比较 | b1 ratio 中位 | b1 mean | b8 ratio 中位 | b8 mean |
|---|---:|---:|---:|---:|
| **调度收益**：old device FFN → fused，同 task/数值/传输 | **0.901**（−9.9%） | 0.919（−8.1%） | **0.941**（−5.9%） | 0.940（−6.0%） |
| **新数值/驻留路径**：default Host FFN → old device FFN | 1.047（**+4.7% 回退**） | 1.057 | 0.904（−9.6%） | 0.907（−9.3%） |
| **合计**：default Host FFN → fused | **0.913**（−8.7%） | 0.913 | **0.874**（−12.6%） | 0.873 |

解释：

- **调度收益**是干净的单变量归因：两个 arm 的 task、buffer、上传下载和数值逐 bit 相同，只少 Submit envelope。b1 中位 −9.9%，b8 中位 −5.9%。
- **新数值路径收益**不能只看 fused 对 default：old device FFN 的 b1 在 AB/BA 中反而 **+4.7% 回退**，b8 **−9.6%**。这正好复现上一轮的 b1 回归；fused 的 Submit 合并把它拉回并通过 default。
- **合计收益** default→fused：b1 −8.7%，b8 −12.6%。这是“新数值路径 + 去掉 Host SwiGLU/residual + Submit 合并”的合成结果，不把它单独归因于调度。

原始文件：

- `abba-ab/`、`abba-ba/`：old device FFN vs fused；
- `abba-nab/`、`abba-nba/`：default vs old device FFN；
- `abba-dab/`、`abba-dba/`：default vs fused；
- 汇总：`analysis-fusion.json` 的 `abba` 字段。

## 13. RoPE 最小设备重排实验

新增通用 `rope_device_module`：

- `Rot = X @ P`，P 是 block-diagonal signed permutation 常量；
- `Y = X*cos + Rot*sin`，后三个 EW task；
- 4 task、1 Submit、Host activation reads = 0；
- Cos/Sin 是运行时输入，当前每次上传约 `6,144 B`，**尚未宣称系数驻留**；
- 支持 interleaved/split-half 和 partial rotary 的模板，但 Smol 集成尚未完成。

Board probe（Tiny 形状 m=16,width=64,head_dim=64，200 calls）：

| 变体 | task/submit | 设备 median | Host FP16 median | 结果 |
|---|---:|---:|---:|---|
| interleaved, rotary=64 | 4 / 1 | **236.8 µs** | 357.3 µs | 设备快 **33.7%**，逐 bit 一致 |
| split_half, rotary=32 | 4 / 1 | 236.5 µs | 210.9 µs | 设备慢 12.2%，逐 bit 一致 |

这说明 **RoPE 的旋转重排本身可以在设备上完成**，但当前只是 operator-level 实验；Cos/Sin 上传、位置索引和 Smol 多层/每 token 的调用组织仍未解决，不能把 Smol Host RoPE 边界标为已消除。

证据：`rope-device-interleaved-board.json`、`rope-device-split32-board.json`、对应 `.rkplan.json`。

## 14. QKV→QK 与 PV→O 真实设备数据流推进

### PV→O：已实现最小真实设备数据流

新增 `pv_o_device_module`：4 个 PV head MM 直接作为 4 个 O slice MM 的 A 操作数，随后 3 个 FP16 EW add；11 task 合成 1 Submit，Host 不下载/合并 `attn_heads`。

Board probe（b1，16 token，4 heads，head_dim=16，out=64）：

| 项 | 结果 |
|---|---|
| task/submit | 11 / 1 |
| upload/download | 8 upload / 1 download |
| vs Host 四头 FP16 参考 | **逐 bit 相同** |
| vs Host merged 单次 O MM | max diff `0.0625`，独立数值路径 |
| 设备 median | 432.8 µs |
| Host four-head median | 220.5 µs |
| Host merged median | 133.9 µs |

结论：**PV→O 的真实设备 producer→consumer 路径成立**，但当前小 shape 下明显更慢，且输入 probs/V 仍是 Host pack；不采用为默认。剩余缺口是 QKV→PV 的设备驻留和 b8 分组调度（b1 的 11-task 链已可单 Submit；b8 需要 >64 task 或分组）。

### QKV→QK：已定位到真实阻断点

当前默认 graph 仍把 `qkv` 下载后由 Host 按 head slice/transpose，再上传 Q/K：

- b8 每个 model call 的 QK group：4 task、8 upload、4 download，upload `98,304 B/call`；
- PV group 同样约 `98,304 B/call`；
- qkv 输出 download 约 `49,152 B/call`，attention O 输入 upload/download 各 `16,384 B/call`；
- Host `merge_heads` 约 `32,768 B/call`、`28.7 µs/call`（b8）。

现有 RKIR 尝试让 QKV 的设备产物直接作为 QK 的 B/WEIGHT operand 时，在 lowering 阶段拒绝：

```text
matmul weights must be runtime constants or inputs, not device-produced values
```

这不是数值误差，而是 **CUBE 输出 → WEIGHT operand 缺少设备端 slice/transpose/repack 能力**。因此：

- QKV→QK 的“设备数据流”尚未完成，不能声称消除 Host 分片/转置；
- 当前可用的是 Host pack 的 QK group（混合边界）；
- 后继所需能力是设备端 layout view/reorder（CUBE→WEIGHT）或位置/head-indexed 的可变 weight view，而不是再改 LUT 或数值政策。

进一步做了地址级可行性分析（`tools/analyze_cube_weight_mapping.py`，输出
`cube-weight-boundary.json`）：

- CUBE 16×16 物理索引：`((c//8)*16 + r)*8 + c%8`；
- QK B/WEIGHT 逻辑索引：`n*32 + k`（在 1024-element padded BO 内）；
- 所需映射：`dst_weight[n*32+k] = src_cube[(k//8)*128 + n*8 + k%8]`；
- 现有固定 p0312 `out[c,a,b]=in[a,b,c]` 对该 256 个逻辑元素有 **255/256 不匹配**；
- 例如 `K[1,0]` 的 CUBE 字节是 8，WEIGHT 需要目的字节 32，而 p0312 给目的 32 的源是 256。

因此这不是“删掉 legalize 限制”就能解决的软件问题；当前 custom-ops 没有满足该映射的设备 reorder/命令。
最小后继实验的接口已固定为：

```text
co.reorder.cube_to_weight.f16.v1
input:  CUBE [16,16]
output: WEIGHT [16,16]
dst_weight[n*32+k] = src_cube[(k//8)*128 + n*8 + k%8]
```

验收顺序必须是：单 head 16×16 逐 bit → 4 head 无重叠 → QKV producer→reorder→QK consumer；
在拿到对应 DMA/PPU 命令或可变 B view 前，不能提交“已设备化”结论。

证据：`tools/analyze_cube_weight_mapping.py`、`cube-weight-boundary.json`、
`tests/test_graph_fusion.py::test_cube_to_weight_requires_new_reorder_not_p0312`、
`tests/test_graph_fusion.py::test_qkv_to_qk_device_weight_operand_is_rejected`、
`pv-o-device-board.json`，以及默认 graph 的 `host_actions`/plan counters。

## 15. 第二阶段状态

- 保留：FFN fused 收益、原始 board 证据、默认路径冻结。
- 完成：chain 准入收紧 + 负例；RMSNorm 实际域统一 + 独立 operator 数值验收；RoPE 最小设备重排实验；PV→O 真实设备数据流最小实验；AB/BA 调度/数值归因。
- 未完成：Smol 的 RoPE 系数驻留与多层集成；QKV→QK 设备 CUBE→WEIGHT reorder；PV→O b8 调度/集成；QKV→PV 完全驻留。
- 阶段状态：**阶段性完成**。整体任务保持开启，等待上述设备 reorder/layout 能力或新的实现路径。

---

# 第三阶段修正与新增证据

以下内容修正前文相关结论边界；旧记录保留，但以本节为准。

## 16. RMSNorm：两步冻结/独立验收（修正）

前一版 `evaluate_rmsnorm_approx.py` 在同一次运行中先算 calibration 和 acceptance、
再写 spec，因此只能说明“公式没有使用 acceptance 调阈值”，不能称为“先冻结、后验收”。
旧记录保留，但该表述撤回。

现在拆成两个独立进程：

1. `--mode calibrate`：只跑 calibration split，写 spec、gate 和 `.sha256` sidecar；
2. `--mode verify`：另起进程，校验 spec 哈希、plan hash、profile hash、plan 文件哈希、
   库哈希，然后只跑 acceptance split，任何不匹配返回非零。

冻结 spec：

- `models/rmsnorm_approx_acceptance.json`
- `models/rmsnorm_approx_acceptance.json.sha256`
- spec sha256：`4812e0ca7427a1939b9e1722e30a6a8240a910c92cff84504d7df88e40d1d31b`
- plan hash：`736469e7a3b14bed166862929dfe854cfee62333dfd86ccc484460f910336173`
- profile hash：`c6b8d3e71acaf317a37a21d186cc7252c3ca7e0e61d201da12aba35480bddaf6`
- board library sha256：`f9ed96ae2f1c38a941ac8a0e379026d8ef6e67ae9dcd62e4aafc11143d5d8419`

独立 verify 结果：

- spec hash / plan / profile / library：全部匹配；
- acceptance `max_abs=0.0234375`，gate `0.0390625`；
- acceptance `rmse=0.0012227`，gate `0.009765625`；
- PASS。

证据：`rmsnorm-calibrate-frozen.json`、`rmsnorm-calibrate-frozen.json.spec.json`、
`rmsnorm-verify-frozen.json`、`rmsnorm-verify-frozen.log`。

## 17. Probe 退出码修正

`tools/evaluate_rope_device.py` 和 `tools/evaluate_pv_o_device.py` 此前无条件返回 0。
现在两者都检查：

- finite；
- 数值门槛；
- expected task/submit/transfer count；
- 不满足即返回非零。

RoPE 要求 `4 task / 1 submit / 3 upload / 1 download` 且与 Host FP16 逐 bit。
PV→O 要求 `11 task / 1 submit / 8 upload / 1 download`，并检查四头参考逐 bit 与
merged-O 误差门槛。后续自动化不会把未来不一致当作成功。

## 18. RoPE 性能结论降格与生产 Host 对照

前一版“设备快 33.7%”只相对于 **Python 逐维循环 Host 实现**，不能外推为生产 Host RoPE
收益。该结论现在降格为“小的单 head operator 参考”。

使用实际 SmolLM2 配置（9 q-head、3 kv-head、head_dim 64、theta=100000）：

| 场景 | 设备 plan | 生产 `smollm.rope_half.v1` Host action | 结果 |
|---|---:|---:|---|
| prefill q [256,9,64] | 13.36 ms | 2.54 ms | 设备慢 5.3× |
| prefill k [256,3,64] | 3.22 ms | 0.70 ms | 设备慢 4.6× |
| decode q [1,9,64] | 2.15 ms | 0.098 ms | 设备慢 22× |
| decode k [1,3,64] | 0.312 ms | 0.060 ms | 设备慢 5.2× |

原因不是算术，而是 576-wide plan 被 K/N tiling 为 18 task/12 Submit，且每层 q/k
都要跨 Host/device 边界。**当前 device RoPE 不具备生产采用性能。**

## 19. Smol RoPE 系数驻留与模型集成

已实现 opt-in `DeviceRopeRunner`（`tools/run_smollm_device_rope.py`）：

- prefill：Cos/Sin/P 作为 plan constants，在 PlanCall 创建时上传一次，30 层复用；
- decode：Cos/Sin 仍是 runtime inputs，因为当前 runtime 没有动态 position-indexed
  设备 view；
- 集成跑通完整 30 层 prefill + 一次 decode，不读取设备激活做 Host 旋转。

实际 board 结果：

| 场景 | baseline | device | 变化 | logits max diff | top1 |
|---|---:|---:|---:|---:|---:|
| prefill 42-token prompt | 3.741 s | 4.062 s | +8.6% | 0.0547 | equal |
| decode 1 token | 0.572 s | 0.643 s | +12.4% | 0.0664 | equal |
| prefill 36-token acceptance prompt | 3.799 s | 4.051 s | +6.6% | 0.0469 | equal |
| decode 1 token | 0.568 s | 0.620 s | +9.2% | 0.1016 | equal |

两 prompt 的 calibration/acceptance 候选 policy 在
`models/smollm2_device_rope_policy.json`：gate 为 prefill `0.109375`、decode
`0.1328125`、top1 必须一致；acceptance 由 `run_smollm_device_rope.py --policy`
在独立 board 进程执行，policy sha256
`1205d4753b9c2995e6179beb16d79fe3f1b2fa8d61c783d51bb86d4f68d6e7dd`，结果 PASS。
结论：**模型集成路径成立，数值政策可独立验收，但性能不值得采用；decode 系数驻留未完成。**

Residency probe（30 simulated layers）也显示：

- prefill q 设备 313.5 ms vs 生产 Host 67.1 ms；
- prefill k 设备 79.4 ms vs 生产 Host 29.7 ms；
- decode q 1.56 ms vs 0.102 ms；decode k 0.371 ms vs 0.046 ms。

这进一步确认：当前瓶颈是 Submit/传输组织，不是单个 RoPE 算术。

证据：`smollm-rope-device-board.json`、`smollm-rope-device-board-p1.json`、
`smollm-device-rope-board.json`、`smollm-rope-residency-board.json`、
`models/smollm2_device_rope_policy.json`。

## 20. QKV→QK：CUBE↔WEIGHT 地址级约束

已做地址级分析而不是只引用 legalize 拒绝：

- CUBE 16×16：`((c//8)*16+r)*8+c%8`；
- WEIGHT QK B：`n*32+k`（1024-element padded BO 内的逻辑项）；
- 所需映射：`dst_weight[n*32+k] = src_cube[(k//8)*128+n*8+k%8]`；
- 固定 p0312 `out[c,a,b]=in[a,b,c]` 对该 256 项有 **255/256 不匹配**；
- 示例 `K[1,0]`：CUBE 源 8，WEIGHT 目的 32，而 p0312 对目的 32 取源 256。

因此**当前所测的 CUBE/WEIGHT view + 固定 p0312 路径**无法实现该映射。这不是
“所有已有设备路径均已证明不可行”。已额外排查的现有机制：

- PPU（`npu_hw.h`）明确没有 reshape/transpose/permute 寄存器，只有 pooling；
- runtime 没有 device-to-device copy action，upload/download 都是 Host 边界；
- 固定版本源码里的 DPU int16 transposed writer（`tp_org_en`）只写 int16/low16，
  不是保持 FP16 原始位的通用 reorder；
- 单 MM 的 producer 输出组织无法单独消除 token 行与 W/K 行的映射冲突。

这些证据只支持“当前直接 p0312/view 方案不可用”；新的真实设备命令/组合仍可能。
下一步不是删 legalize，而是新增一个真实设备 reorder 命令/模板，接口必须是：

```text
co.reorder.cube_to_weight.f16.v1
input:  CUBE [16,16]
output: WEIGHT [16,16]
dst_weight[n*32+k] = src_cube[(k//8)*128+n*8+k%8]
```

最小实验顺序：单 head 16×16 逐 bit → 4 head 无重叠 → QKV producer→reorder→QK consumer。
当前实现状态：**未实现该硬件 reorder 命令；QKV→QK 仍为 Host pack 混合边界。**
这是具体能力缺口，不是“等待”理由；后续应直接实现该命令或给出厂商寄存器/寻址不可行证明。

证据：`tools/analyze_cube_weight_mapping.py`、`cube-weight-boundary.json`、
`tests/test_graph_fusion.py` 的两个负例。

## 21. PV→O b8 多 Submit 分组

新增 `pv_o_device_batches_module` 和 `tools/evaluate_pv_o_b8_device.py`：

- 8 个 batch，每个 11-task PV→O 链；
- **总 88 task / 8 Submit**，同一 device workspace，链间无 Host 合并；
- 不违反 64-task 单链限制。

Board b8 probe：

| 项 | 结果 |
|---|---|
| task/submit | 88 / 8 |
| upload/download | 64 / 8 |
| 与 Host 四头参考 | max abs `0.03125`、rmse `0.0009766` |
| 与 Host merged O | max abs `0.125`、rmse `0.01584` |
| 设备 median | 3.583 ms |
| Host four-head median | 1.761 ms |
| Host merged median | 1.065 ms |

结论：**组间设备驻留和合法多 Submit 成立，但端到端性能不值得采用**；作为能力实验保留，
不做模型级采用宣称。

证据：`pv-o-b8-device-board.json`、`pv-o-b8-device-board.json.rkplan.json`。

## 22. 第三阶段状态

- 默认冻结路径与 FFN 收益保持不变。
- 修正了 RMSNorm 冻结/验收和 probe 退出码。
- RoPE：operator-level、实际 Smol 形状生产 Host 对照、prefill 系数常驻/跨层复用、
  opt-in 模型 prefill/decode 集成均完成；性能不值得采用，decode 动态位置 view 未完成。
- QKV→QK：地址级可行性证明完成；需要新的设备 CUBE→WEIGHT reorder 命令，尚未实现。
- PV→O b8：合法多 Submit、组间设备驻留、数值边界和性能评估完成；未采用。
- 整体任务保持开启，下一步应直接实现 QKV CUBE→WEIGHT 设备 reorder 命令及
  Smol decode 位置索引设备 view。


## 23. Decode RoPE 位置索引设备 gather

新增 `rope_decode_indexed_module`：CosW/SinW/P 为常驻 plan constants；Host 每 token
只更新一个 one-hot 位置向量 `[1,capacity]`，设备 MM 从常驻表 gather 出 Cos/Sin，
再执行同一 P 重排和 3 个 EW task。

实际 Smol 形状 board probe：

| arm | task/submit | per-token upload B | device median | max diff vs production Host |
|---|---:|---:|---:|---:|
| decode q, 9 heads | 24 / 18 | 19,968 | 2.700 ms | 0.001953 |
| decode k, 3 heads | 6 / 6 | 10,752 | 0.887 ms | 0.001953 |

对照 runtime-input decode plan：

- q upload `41,472 → 19,968 B`，但 device `1.56 → 2.70 ms`；
- k upload `13,824 → 10,752 B`，但 device `0.371 → 0.887 ms`。

位置验证覆盖 `0,1,15,16,127,254,255`，跨表边界与最后合法位置均通过；
越界 `256` 在 Host one-hot 生成前拒绝；A/B/A 重放 bitwise 一致。
结论：**真实设备位置索引/gather 能力成立，常驻表不再每 token 重传；但 gather MM 的
Submit/task 成本使当前 decode 更慢，不能判定值得采用。**

## 24. RoPE 连续集成验收（32 步）

`run_smollm_device_rope.py` 已扩展为：

- baseline 与 device 同一 runner/package；
- baseline 产生固定 forced-token 前缀；
- device 用同一前缀连续 decode 32 步；
- 对比每步 logits/top1；
- 候选自身 incremental vs full-prefix 一致性；
- A/B/A reset 隔离。

两个 prompt 的 board 结果：

| prompt | prefill baseline→device | 32-step decode baseline→device | max decode diff | top1 |
|---|---:|---:|---:|---:|
| prompt-0 (43 tokens) | 3.945→4.218 s (+6.9%) | 17.277→19.730 s (+14.2%) | 0.11914 | 32/32 |
| prompt-1 (36 tokens) | 3.749→4.159 s (+10.9%) | 17.250→19.706 s (+14.2%) | 0.14453 | 32/32 |

额外结果：

- `incremental_vs_full_prefix_max_abs = 0.0`；
- `incremental_vs_full_prefix_rmse = 0.0`；
- `incremental_vs_full_prefix_top1_equal = true`；
- `aba_request_isolation_bitwise = true`；
- 两个 prompt 均 finite。

连续候选 policy：`models/smollm2_device_rope_policy_cont32.json`，由 prompt-0
校准门限、prompt-1 独立验收；policy sha256
`fc105a2c6a8f95208b4a1882f821d6ea8cf9808d3c714f57a89a347ab6f02101`，
独立 verify PASS。该 policy 是 32-step 候选 policy，不是默认 Smol acceptance。

结论：**连续 32 步 opt-in 集成验收通过；性能仍慢于生产 Host RoPE，保持 opt-in，
不进入默认。**

## 25. 连续 decode 的位置索引模型级集成

在 `DeviceRopeRunner` 增加 `--decode-indexed`：decode 不再传 Cos/Sin 向量，而是
从常驻 `CosW/SinW` 表用设备 one-hot gather；Host 每 token 只更新 one-hot 位置元数据。
同一 32-step 连续验收流程分别对 runtime-input 与 position-indexed 两种 decode 方式运行。

Board 结果（baseline 为生产 Host RoPE，device 为 opt-in 候选）：

| source / mode | prefill baseline→device | 32-step decode baseline→device | max decode diff | top1 |
|---|---:|---:|---:|---:|
| p0 runtime inputs (`smollm-device-rope-cont-p0.json`) | 3.945→4.218 s (+6.9%) | 17.277→19.730 s (+14.2%) | 0.11914 | 32/32 |
| p1 runtime inputs (`smollm-device-rope-cont-p1-contpolicy.json`) | 3.749→4.159 s (+10.9%) | 17.250→19.706 s (+14.2%) | 0.14453 | 32/32 |
| p0 indexed gather (`smollm-device-rope-indexed-cont-p0.json`) | 3.776→4.141 s (+9.7%) | 17.090→20.376 s (+19.2%) | 0.11914 | 32/32 |
| p1 indexed gather (`smollm-device-rope-indexed-cont-p1-policy.json`) | 3.690→4.192 s (+13.6%) | 17.071→20.303 s (+18.9%) | 0.14453 | 32/32 |
| p3 indexed gather, new unobserved acceptance (`smollm-device-rope-indexed-acc-p3-finalpolicy.json`) | 3.593→4.215 s (+17.3%) | 17.170→20.653 s (+20.3%) | 0.140625 | 32/32 |

两种方式都满足：

- `incremental_vs_full_prefix_max_abs = 0.0`；
- `incremental_vs_full_prefix_rmse = 0.0`；
- `incremental_vs_full_prefix_top1_equal = true`；
- `aba_request_isolation_bitwise = true`；
- 32/32 decode top-1 与 baseline 一致；
- finite。

旧 indexed policy 的 acceptance 结果已被观察，不能作为新的独立验收；其记录保留。
新的 `models/smollm2_device_rope_policy_cont32_final.json` 采用严格冻结顺序：

1. prompt-2 只用于 calibration 并冻结 gates；
2. prompt-3 为此前未观察输入，独立 verify；
3. policy 绑定 `decode_mode=position_indexed_device_gather`、`decode_steps=32`、
   `expected_decode_calls=3360`；
4. acceptance：prefill max `0.0703125`、decode max `0.140625`、top1 32/32、
   compare count `33/33`、A/B/A 8-step 完整序列 bitwise 一致、finite，PASS。

policy sha256：`2d1d7c485b5901b3936beab3dd62e99645b2df68a99c213e3dd5f825d144a5bd`。

它证明 **Host 不再每 token 重传 Cos/Sin 表**；但 gather MM 的 task/Submit 成本使总
decode 仍比生产 Host RoPE 慢约 19~20%，因此不采用。


## 26. 连续验收器补强

本轮修正了以下验证缺口：

- policy 现在强制绑定 `decode_mode`、`decode_steps`、`expected_decode_calls`；
  模式、步数、实际 decode 调用数任一不匹配都会拒绝；负例测试覆盖三者。
- incremental/full-prefix 现在比较 **全部 33 个位置**（1 prefill + 32 decode），
  并断言 `compare_count == decode_steps + 1`，不再用 `dev_pred[:-1]` 截短。
- A/B/A 每个请求现在都执行 **8 步 decode**，比较完整 prefill+decode 序列；
  同时检查 sequence shape 与 token 序列一致，记录 `aba_decode_steps=8`。
- final indexed policy 在全新 prompt-3 上完成独立 acceptance，校准使用 prompt-2；
  旧输入上的重跑只记录为回归，不作为新验收。
- 旧单步 policy 在连续输入上的失败记录保留；新的连续 policy 明确是独立范围/顺序。
