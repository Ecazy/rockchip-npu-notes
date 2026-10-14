# RoPE、KV cache 与 FP16 self-attention

2026-10-13，RK3588 / 厂商 RKNPU。新增 `co_rope_*`、`co_attention_*`、`co_linear_update_weights`，以及静态图的 `CO_GRAPH_ROPE` / `CO_GRAPH_ATTENTION` 节点。

当前实现是单序列、FP16、固定容量的混合 self-attention 基线。它接收已经投影的 Q/K/V，不包含 tokenizer、模型导入或完整 LLM generation runtime。

## 支持范围

| 能力 | 契约 |
|---|---|
| RoPE | interleaved / split-half；partial rotary；theta可配置；绝对起始位置为uint32 |
| MHA/GQA/MQA | Q heads必须是KV heads的整数倍；KV组按 `h/(Hq/Hkv)` 映射 |
| Head维度 | 1..256；`heads*head_dim≤4096`；启用RoPE时rotary_dim为正偶数且≤head_dim |
| Query / KV容量 | `1≤max_query≤capacity≤4096`；每次append tokens≤max_query且不超过剩余容量 |
| Mask | causal、可选causal滑动窗口（window≤capacity）、共享于所有head的FP16 additive mask；接受`-inf`，拒绝NaN/+inf；全masked行输出零 |
| KV cache | 主机持久数组；K为已经RoPE旋转后的FP16，V为原始FP16；连续append、reset、export |
| 图编译 | 静态self-attention，Q/K/V行数相同，形状/布局/生命周期和容量自动规划 |
| 数值路径 | NPU QK、NPU exp/reduce/scale、NPU PV；CPU参与RoPE配对/系数、KV与动态权重打包、score缩放/mask和Softmax统计 |

滑动窗口仅控制可见key，不驱逐缓存；容量满时返回错误。没有cross-attention、batch、多序列分页cache、FlashAttention、ALiBi、attention sinks、softcap、RoPE scaling变体或INT4/INT8支持。

## RoPE实现

对于第一 `rotary_dim` 个通道，频率为 `theta^(-2*i/rotary_dim)`，位置为 `position_base+row`。
interleaved按相邻对旋转；split-half将前后半区配对。超过rotary_dim的通道保持原值。

CPU收集完整逻辑行块，生成配对后的 `rotate(X)` 及FP16 cos/sin；因此配对可以跨256通道的tile边界。NPU执行三个EW task：

```
a = fp16(X * cos)
b = fp16(rotate(X) * sin)
Y = fp16(a + b)
```

每个tile一次chain Submit；identity权重与五个scratch BO在创建时分配并重复使用。逻辑CPU激活读取为M×N，三个展开数组上传为3×M×N；中间乘积留在设备。未启用旋转的通道在这一路径中使用cos=1、sin=0。

## Attention与缓存

固定容量在创建时确定，预分配QK/PV Linear上下文、Softmax上下文及输入/输出/score/probability tile集合。`co_linear_update_weights`更新既有BO的内容，形状和bias存在性不能改变，不重新分配weight BO。

每次step的执行顺序为：

1. 验证所有Q/K/V、可选mask、token数和剩余容量；尚不发布cache长度或用户输出。
2. 对新增K执行RoPE，下载旋转结果并暂存于未提交cache尾部；V复制到同一尾部。未启用RoPE时K直接复制。
3. 每个KV组从cache重排并上传K及转置V权重；组内多个query head复用同一权重。
4. 逐query head上传Q、执行RoPE，在NPU计算QK。
5. CPU读取score，应用 `1/sqrt(D)`（或指定scale）、causal/window和additive mask，写回FP16score。
6. [混合稳定Softmax](CROSS-TILE-ROW-OPS.md) 生成概率tile；概率直接作为PV的K tile，**不下载/拼接P**。
7. NPU计算PV，下载各head输出到内部暂存数组；所有head和guard检查成功后，再提交cache长度并发布用户输出。

失败会清除未提交的KV尾部；已提交prefix、长度和用户输出保持不变。scratch和weight BO可能已经改变，重试时会从已提交cache和本次输入重建。验收使用调用层 `co_execute` 的受控故障注入，并非硬件timeout/驱动reset恢复验收。

QK/PV使用既有256上限分解；PV每个K tile的部分和收窄至FP16。score缩放和mask之后也写入FP16，因此不保证任意大幅值输入/大mask偏移等价于FP32 SDPA。

**固定计算容量：**每次step都会计算 `max_query × capacity` 的score，包括未使用区域；这些区域由mask置零概率。单token decode也执行此固定工作量。这一版验证功能及缓存事务，不是decode吞吐优化，也不是O(N)显存的FlashAttention。

## API

[co_attention.h](custom-ops/include/co_attention.h) 的独立上下文用于跨调用持久KV：

```c
co_attention_config cfg = {
    .max_query=128, .capacity=1024,
    .heads=8, .kv_heads=2, .head_dim=128,
    .rotary_dim=128, .theta=10000,
    .style=CO_ROPE_SPLIT_HALF, .causal=1,
};
co_attention *attn = NULL;
int rc = co_attention_create(fd, &cfg, CO_CHAIN, &attn);
if (!rc) rc = co_attention_reset(attn, 0);
/* Q[T,8,128]，K/V[T,2,128]，Y[T,8,128]；row-major小端FP16。 */
if (!rc) rc = co_attention_step(attn, T, q, k, v, NULL, y);
/* 后续新token继续append；不重复提交旧K/V。 */
if (!rc) rc = co_attention_step(attn, 1, next_q, next_k, next_v, NULL, next_y);
unsigned committed = co_attention_cache_length(attn);
co_attention_destroy(attn); /* 调用方最后关闭fd */
```

可选mask形状为 `[本次tokens,旧长度+本次tokens]`。cache export复制 `[length,kv_heads,head_dim]` 的旋转K与原始V，调用方先查询length并提供不重叠数组。所有接口同步执行，不支持并发使用同一个上下文。

[co_rope.h](custom-ops/include/co_rope.h) 单独消费/生成规范的图tile集合；[co_graph.h](custom-ops/include/co_graph.h) 的静态节点示例：

```c
co_graph_node nodes[] = {
    {.op=CO_GRAPH_INPUT, .rows=T, .cols=Hq*D},
    {.op=CO_GRAPH_INPUT, .rows=T, .cols=Hkv*D},
    {.op=CO_GRAPH_INPUT, .rows=T, .cols=Hkv*D},
    {.op=CO_GRAPH_INPUT, .rows=T, .cols=T, .flags=CO_GRAPH_ALLOW_NEG_INF},
    {.op=CO_GRAPH_ATTENTION, .a=0, .b=1, .c=2,
     .has_mask=1, .mask_node=3,
     .heads=Hq, .kv_heads=Hkv, .head_dim=D,
     .rotary_dim=D, .theta=10000, .rope_style=CO_ROPE_SPLIT_HALF,
     .position_base=0, .causal=1},
};
co_graph_desc desc = {nodes, 5, 4, 0};
```

Q/K/V可来自上游Linear节点；输出可继续接Linear/Residual/FFN。这里的Q/K/V由tile集合下载到主机暂存，attention输出又上传到图BO，属于明确的CPU桥接；不能称为整个attention图设备驻留。

静态图每次运行会reset其attention cache，不保留跨run的decode状态。跨调用decode请用独立 `co_attention_step`。目前图节点不接不同行数的Q与K/V。

## 规划和可观测数据量

`co_attention_plan_init` 不需要设备，给出驻留BO请求字节、固定step task/Submit、上传/下载调用、CPU激活读取、score写回、统计下载和动态权重元素数。预算默认256 MiB，不包括驱动页对齐、临时提交BO或进程内所有Host内存；`host_bytes`仅估计主要数组，不包含下层Linear/Softmax的全部Host元数据和程序模板。

图规划新增 `rope_bytes` / `attention_bytes`、`activation_download_calls` / `activation_downloads`，明确区分统计下载与attention的Q/K/V、旋转K、输出下载。提交与传输计数均通过真实文件/图接口核对。

## 验证与复跑

```bash
cmake -S research/custom-ops -B research/custom-ops/build-host
cmake --build research/custom-ops/build-host -j4
ctest --test-dir research/custom-ops/build-host --output-on-failure
bash research/custom-ops/scripts/run-attention-board.sh
```

工具 [attention_probe.c](custom-ops/tools/c/attention_probe.c) 生成原始Q/K/V/mask、stream与graph输出、旋转KV、失败前后prefix和输出marker；每种执行重复三次。
[attention_gate.py](custom-ops/tools/python/attention_gate.py) 独立计算RoPE和double数学SDPA，不调用设备布局/图规划辅助函数。
[attention_campaign.py](custom-ops/tools/python/attention_campaign.py) 回收后重新核对数学结果、哈希、cache进度、计数和chain/separate字节一致性。

覆盖的配置为：

| tokens / capacity / max_query | Hq / Hkv / D / rotary | 样式与mask | step分解 |
|---|---|---|---|
| 7 / 7 / 3 | 4 / 2 / 6 / 4 | interleaved、causal、window=3 | 3,3,1 |
| 259 / 259 / 129 | 2 / 1 / 8 / 8 | split-half、causal | 129,129,1 |
| 257 / 257 / 257 | 1 / 1 / 19 / 18 | interleaved、causal、window=17 | 257 |
| 17 / 17 / 17 | 2 / 2 / 5 / 0 | 无RoPE、非causal | 17 |
| 8 / 16 / 8 | 4 / 1 / 32 / 16 | split-half、causal、window=4 | 8 |
| 13 / 13 / 13 | 32 / 16 / 19 / 18 | interleaved、causal、window=5；Q/K/V总通道跨256 | 13 |

另有13×400 split-half跨channel配对、257×304 interleaved跨M/channel tile配对。所有attention用例包含全masked行；末尾Q/K被设为强相关，并对最后key增加bias，使ragged K/PV尾块错误不能被1/N的小概率掩盖。

当前验收只覆盖所测形状与值分布。尚未测LLM perplexity、整模型吞吐或实际驱动故障后的恢复。

## 当前实测结果

[原始记录](custom-ops/results/2026-10-13/attention-20261013-174358/README.md) 保留全部输入/输出、失败前后cache、输出marker、源码/文件SHA256和真实task/Submit/传输计数。
三个核×两种提交模式，每个配置的流式与静态图各重复3次，共36组受控错误回滚；所有计数、缓存进度、字节重复性及chain/separate对照通过。

最大绝对误差：stream/graph SDPA均为 `0.000534888038`，旋转K cache为 `0.000639162092`，跨tile RoPE为 `0.000775070128`。
Host16/16、ASan/UBSan和原多tile路径回归通过；kernel journal区间无新增消息。

后续 [完整decoder block与M=1执行几何](TRANSFORMER-DECODER-BLOCK.md) 已加入七个投影/FFN矩阵和两个残差，并用cache clone/truncate在几何切换及post-FFN失败时保持逻辑历史。
