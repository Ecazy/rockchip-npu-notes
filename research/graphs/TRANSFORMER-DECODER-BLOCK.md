# 单层 Transformer decoder block 与单行 decode

2026-10-13，RK3588 / 厂商 RKNPU。新增 `co_decoder_*`，组合现有图编译器、RMSNorm、Linear、RoPE/GQA、Softmax与SwiGLU，执行完整的bias-free、pre-RMSNorm decoder block。

表达式为：

```
Z1 = RMSNorm(X, norm1)
Q,K,V = Linear(Z1, Wq/Wk/Wv)
A = self_attention_with_RoPE_and_persistent_KV(Q,K,V)
R = X + Linear(A, Wo)
Z2 = RMSNorm(R, norm2)
F = Linear(SiLU(Linear(Z2,Wgate)) * Linear(Z2,Wup), Wdown)
Y = R + F
```

这是单序列单层的FP16功能与API性能基线，不是模型loader、tokenizer、LM head、sampling或完整LLM runtime。当前维度上限仍为4096，head_dim≤256；不支持大于4096的常见FFN intermediate维度、量化或LayerNorm/bias/GELU架构变体。

## 图与缓存事务

pre graph一次执行RMSNorm，再共享同一个设备normalized tensor进行Q/K/V三个投影。新增 `co_graph_plan_many/compile_many/run_many`，以多个output根遍历DAG并去重共享节点；所有output的BO保留到末尾。Q/K/V全部下载并检查有限值后，才发布任何一个目标数组。

post graph执行Wo、第一残差、第二RMSNorm、SwiGLU/down投影和第二残差。中间Linear/逐元素tensor保持Cube，RMSNorm与attention仍是既有明确混合路径：统计/系数与Q/K/V/attention结果涉及CPU传输。

attention本身成功不再代表block已提交。若post graph失败，`co_attention_truncate`删除本步新增KV，恢复之前的长度；用户Y直到整个block成功才复制。
验收既包含调用层在post-FFN完成后返回受控错误，也包含真实的后置RMSNorm系数溢出拒绝。它不证明实际硬件timeout/reset后的恢复。

## 单 token 优化

默认在创建时编译两套固定几何：

- prefill/chunk：rows=max_query。
- tokens=1：rows=1，减少projection、norm、FFN及score的无效query行。

每套有自己的graph BO、attention上下文和权重副本，因此设备容量预算会增加。它们共享一个逻辑KV历史：只在执行几何切换时，用 `co_attention_cache_clone` 复制已提交的旋转K、原始V及position base。连续单token decode保持同一上下文，不在每步复制prefix；decode切回多token chunk同样保持历史。

cache clone检查KV/RoPE编码和容量，不复制scratch/weight BO；下一步从该prefix重新打包动态权重。失败的几何切换不更新canonical cache或成功切换计数。

`disable_one_row_decode=1`是对照路径，所有step仍使用max_query。KV列仍计算到固定capacity，没有有效长度bucketing、分页device KV或FlashAttention优化。逻辑score元素数减少不等于同倍率硅片MAC减少：physical M至少12，任务/同步开销依然存在。

## 接口

[co_decoder.h](../custom-ops/include/co_decoder.h) 提供计划、创建、reset、step、cache export与诊断getter。矩阵都是row-major FP16：

| 参数 | 形状 |
|---|---|
| norm1 / norm2 | [model_dim] |
| Wq | [heads*head_dim, model_dim] |
| Wk / Wv | [kv_heads*head_dim, model_dim] |
| Wo | [model_dim, heads*head_dim] |
| Wgate / Wup | [hidden_dim, model_dim] |
| Wdown | [model_dim, hidden_dim] |
| X / Y | [本次tokens, model_dim] |
| 可选mask | [本次tokens,旧KV长度+本次tokens] |

```c
co_decoder_config cfg = {
    .model_dim=512, .hidden_dim=1536, .epsilon=1e-5f,
    .attention={
        .max_query=128, .capacity=1024,
        .heads=8, .kv_heads=2, .head_dim=64, .rotary_dim=64,
        .theta=10000, .style=CO_ROPE_SPLIT_HALF, .causal=1,
    },
};
co_decoder_weights weights = {
    .norm1=gamma1, .norm2=gamma2,
    .wq=Wq, .wk=Wk, .wv=Wv, .wo=Wo,
    .wgate=Wgate, .wup=Wup, .wdown=Wdown,
};
co_decoder *block = NULL;
int rc = co_decoder_create(fd, &cfg, &weights, CO_CHAIN, &block);
/* 成功后可释放所有weights/gamma数组；两套几何都已经复制/打包。 */
if (!rc) rc = co_decoder_reset(block, 0);
if (!rc) rc = co_decoder_step(block, prefill_tokens, x, NULL, y);
if (!rc) rc = co_decoder_step(block, 1, next_x, NULL, next_y);
unsigned length = co_decoder_cache_length(block);
unsigned rows = co_decoder_last_query_rows(block); /* 上一次成功step的几何 */
co_decoder_destroy(block); /* 最后由调用方关闭fd */
```

创建时预分配全部operator scratch及模型weight BO。device_budget默认256 MiB，包含两套几何的驻留BO payload/DMA余量/guard；不含driver page rounding、Host程序模板或临时Submit BO。输入/输出可以原位使用，multi-output图的目标数组之间不能重叠。所有接口同步，不支持并发调用同一上下文。

RMSNorm仍有FP16平方与tile行能量门禁，attention/PV仍有FP16分片收窄，完整block不保证任意幅度数据等价于FP32。

## 实测

[原始记录](../custom-ops/results/2026-10-13/decoder-20261013-184941/README.md) 包含三个核×chain/separate、三种形状、两条流式路径与一条16-node静态完整图。
所有模式各重复3次，matrices/gammas在compile后清零并释放，以检查权重所有权。

| tokens / max_query / capacity | model / FFN | Hq / Hkv / D / rotary | step |
|---|---|---|---|
| 7 / 3 / 7 | 17 / 33 | 4 / 2 / 6 / 4 | 3,1,3 |
| 131 / 129 / 259 | 33 / 65 | 4 / 2 / 8 / 8 | 129,1,1 |
| 17 / 13 / 17 | 304 / 513 | 16 / 4 / 19 / 18 | 13,1,1,1,1 |

独立double数学参考覆盖两个RMSNorm、七个Linear、RoPE/causal/window/mask、Softmax、SwiGLU与两个残差。
完整block最大绝对误差 `0.000653523463`（门限0.008），投影/旋转KV最大误差 `0.001167819312`（门限0.006）。优化与固定MQ路径输出在本轮逐字节一致；chain/separate同一路径的output/cache也一致。

18组post-FFN受控错误回滚通过；缓存prefix和用户输出marker不变，重试成功。Host16/16及ASan/UBSan通过，原attention与multi-tile回归通过，kernel journal区间无新增消息。

## API计时与边界

计时为配对、交替执行的warm完整block API调用，每种schedule重复3次，排除constructor及受控错误调用，包含必要的cache切换。没有记录恒频控制/全生产负载；样本量不足以证明总体吞吐或尾延迟收益。

max_query=129、model=33、FFN=65、capacity=259的单token对照：

| Core mask，chain | 固定MQ API中位数 | M=1 API中位数 | 比值 |
|---|---:|---:|---:|
| 1 | 26.335 ms | 14.963 ms | 1.760× |
| 2 | 32.742 ms | 21.227 ms | 1.542× |
| 4 | 31.525 ms | 20.528 ms | 1.536× |

三个核、两种提交模式的此形状比值为1.491–1.760×。更小MQ的形状比值约0.89–1.25，部分慢于对照，不能宣称所有decode都加速。

此形状的单token逻辑score元素数由133644降到1036（129×），physical M由132降到12；chain Submit仍为49，task由88到84，上传/下载调用仍为61/20。主要减少每个调用的数据和计算量，没有消除launch/cache-sync/CPU打包开销。

```bash
bash research/custom-ops/scripts/run-decoder-board.sh
python3 research/custom-ops/tools/python/decoder_campaign.py \
  research/custom-ops/results/2026-10-13/decoder-20261013-184941
```

还没有真实checkpoint、整模型perplexity、LM head/sampling或多层cache事务验收。下一阶段的优化应以有效KV长度、设备cache布局和跨节点提交开销为对象，不能用本轮单层API比值代替整模型性能。
