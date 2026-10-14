# 跨 tile RMSNorm 与稳定 Softmax

2026-10-13，RK3588 / 厂商 RKNPU 路径。新增 `co_rowop_*` 和图节点 `CO_GRAPH_RMSNORM` / `CO_GRAPH_SOFTMAX`，逻辑 rows/cols 均为1..4096，使用与 Linear 相同的 M-major/channel-minor Cube tile 集合。

这是明确的 CPU/NPU 混合实现，没有在 NPU 失败后改用 CPU 报告成功。它为后续 attention 提供跨通道归约基础，其后 [RoPE、KV cache 与self-attention](ATTENTION-KERNELS.md) 已组合这些基础，另有独立验收。

## RMSNorm

```
squares_tile = fp16(X_tile * X_tile)      # NPU
partial_row_sum = fp32(sum(squares_tile)) # NPU，只计逻辑通道
row_sum = sum(partial_row_sum)           # CPU，跨所有 channel tile
coeff = fp16(gamma / sqrt(row_sum/N+eps)) # CPU，展开到每个 tile
Y_tile = fp16(X_tile * coeff_tile)       # NPU
```

输入激活不读回或在 CPU 上扫描。每个 channel tile 只下载每行一个 FP32 统计量；gamma 在创建/编译时复制，运行时上传系数。最后一个 channel tile 的 ones 权重只覆盖其逻辑通道，不能把32通道 padding 或非线性改变后的 padding 计入统计。

每个 tile 三个 task：square、FP32 reduce、scale。chain 模式分成两个 Submit，归约结束后 CPU 组合全行统计，再提交 scale。

数值契约是 FP16 square、FP32 tile sum、CPU double 跨 tile 求和、FP16 scale coefficient及输出，不能声称等价于纯FP32 RMSNorm。epsilon 必须有限且正，gamma 必须有限；系数不可表示时返回错误。

**能量门禁：**任一 tile 的任一逻辑行平方和达到65504时返回错误。真机 FP16 平方溢出会饱和为最大有限值，而非一定输出Inf，因此仅检查非有限统计会漏报。此门禁保守地拒绝部分平方和较高但没有元素溢出的输入；这不是硅片的通用 RMSNorm 上限。尚未实现自适应缩放的大幅值 RMSNorm。

## 稳定 Softmax

先从所有 channel tile 的逻辑元素中求全行最大值。对 `delta = x-row_max` 分成指数区间：

```
band = floor(-delta / 4)
remainder = delta + 4*band       # [-4,0]
e = exp(remainder) * exp(-4*band)
```

CPU 读取 logits，求 row max、分段 remainder 和 band coefficient；新的 `CO_EXP_NEG4` 在 NPU 上计算 remainder 的 exp，随后 NPU 乘系数得到FP16指数项并归约为FP32每行统计。CPU 组合跨 tile 分母和展开倒数系数，NPU 完成最后的归一化。

分段的必要性：LUT 的Q0.15非零下限约为 `1/32768`，直接把长负尾送入宽域exp会给每个尾部项增加一个概率质量下限。4096项时足以显著改变分母。当前 remainder 的最小 exp 为 `exp(-4)`，避免触及LUT下限；负尾通过FP16 band coefficient恢复。band≥5 的值小于最小FP16指数项的舍入区间，明确置零。

- Softmax只归约逻辑通道；padding的band coefficient和倒数系数为零。
- 支持 `-inf` mask，NaN和`+inf`返回错误。全masked行定义为全零输出，非masked行概率和接近1。
- Graph Input 接受 `-inf` 必须设置 `CO_GRAPH_ALLOW_NEG_INF`，且可达消费者只能是Softmax。不能把允许mask的Input接到Linear/RMSNorm等节点。
- CPU会扫描输入logits两次；这一路径有中间激活的CPU读取，不能沿用纯Linear/FFN“仅图边界传输”的说法。
- 归一化分母使用FP32 tile sum和CPU double求和；输出仍为FP16近似。

Softmax每个tile是 `ceil(physical_elements/32768)+3` 个task，chain模式两次Submit。若物理tile为256×256，exp LUT拆为两个task。

## 图规划与调用

接口为 [co_row_ops.h](custom-ops/include/co_row_ops.h) 和 [co_graph.h](custom-ops/include/co_graph.h)。Row算子的所有scratch BO、identity/ones权重在创建时分配；跨M块复用，不在每次执行时重新分配。图规划加入 `rowop_bytes` 和明确的混合处理计数：

| 每次节点运行 | RMSNorm | Softmax |
|---|---:|---:|
| CPU激活元素读取 | 0 | 2×M×N |
| FP32统计元素下载 | M×ceil(N/256) | M×ceil(N/256) |
| FP16展开数组元素上传 | M×N | 3×M×N |
| 上传调用次数 | tile数 | 3×tile数 |
| 统计下载调用次数 | tile数 | tile数 |

这些是逻辑数据量和调用数，不包含padding、cache同步、guard读取、输入输出边界或重复次数。Host输出仍在全图成功后才发布；设备集合接口出现失败时，前序输出tile可能已经完成。

```c
/* RMSNorm；shape由Input推导，gamma长度为N。 */
co_graph_node nodes[] = {
    {.op=CO_GRAPH_INPUT, .rows=M, .cols=N},
    {.op=CO_GRAPH_RMSNORM, .a=0, .gamma=gamma, .epsilon=1e-5f},
};

/* Masked Softmax；无需gamma。 */
co_graph_node softmax_nodes[] = {
    {.op=CO_GRAPH_INPUT, .rows=M, .cols=N, .flags=CO_GRAPH_ALLOW_NEG_INF},
    {.op=CO_GRAPH_SOFTMAX, .a=0},
};
```

`graph_ffn` 新增可选尾部参数 `rmsnorm gamma.bin`，表达式变为：

```
Z = RMSNorm(X)
Y = X + Linear(SiLU(Linear(Z,Wgate,b_gate))*Linear(Z,Wup),Wdown)
```

```bash
# 板端：gamma FP16[N]，输入/输出 FP16[M,N]。
./build/row_graph rms 257 513 x.bin gamma.bin y.bin chain 3
./build/row_graph softmax 13 4096 scores.bin - probabilities.bin chain 3
./build/graph_ffn 257 257 289 x.bin wg.bin wu.bin wd.bin bias.bin y.bin chain 3 rmsnorm gamma.bin

# 仓库根：上传新源码、三核串行验收、回收失败或成功的原始记录。
bash research/custom-ops/scripts/run-row-board.sh
```

数值验收包括独立数学参考、零行、大正偏移、全masked/部分masked、跨tile峰值、4096通道负尾、非线性后padding、RMSNorm前置FFN、提交/混合传输计数、重复与chain/separate对照。详细真机结果在本轮不可变记录中登记。

## 实测结果

[原始验收与复核说明](custom-ops/results/2026-10-13/row-20261013-154846/README.md) 保留三核逐组输入/输出、命令、混合传输计数、代码SHA256和kernel journal区间。

三个核的最大误差一致：RMSNorm `0.00188537460`、Softmax `0.000115289912`、前置RMSNorm FFN `0.00048828125`；Softmax行和误差最大 `0.000378668308`。
每核22个归约shape/mode检查及3个负例、16个归一化FFN检查及2个负例，重复3次；chain/separate字节一致，旧路径回归通过。Host、ASan/UBSan和artifact完整性负例通过，kernel区间无新增消息。

首轮发现了FP16平方饱和导致未拒绝溢出的情况，原始失败记录保留；当前能量门禁已通过真机负例。现场汇总文件名错误也已修正为独立 `row_campaign.py`，回收后重新核对SHA256与行数学结果。

| 每次chain运行 | 驻留BO请求字节 | Submit | task | 上传调用 | 下载调用 |
|---|---:|---:|---:|---:|---:|
| RMSNorm 257×513 | 1325120 | 12 | 18 | 12 | 12 |
| RMSNorm 13×4096 | 657856 | 32 | 48 | 32 | 32 |
| Softmax 257×513 | 1800384 | 12 | 26 | 24 | 12 |
| Softmax 13×4096 | 862784 | 32 | 64 | 64 | 32 |

调用数含图输入上传、图输出下载以及混合统计/系数传输；不含编译阶段的权重上传。它们不是吞吐指标，本轮没有做LLM perplexity或性能收益验收。
