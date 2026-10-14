# rknpu2_matmul_open 性能复核（2026-10-14）

**结论：不是所有形状都比官方快 50%。相同单核条件下，当前 INT8 split-K 接口在三个 M=128 的 LLM prefill 形状上，完整调用达到官方较快布局的 2.01×、2.05×、4.10×；M=1/4 的 decode 形状则略慢约 2%–8%。**

这支持“特定大矩阵有明显优化收益”，不能推导成所有算子、FP16、完整模型或 tokens/s 普遍提升 50%。本次未测 FP16、BF16、TF32、W4、融合 attention 或完整模型。

## 版本与环境

- 本地源码：`/home/maoyu/Documents/code/rknpu2_matmul_open`，commit `192f6ca25407fac7c0880123c0c394bd3e2f74f7`（2026-10-06）。原 checkout 没有修改。
- 板子：Orange Pi 5 Plus / RK3588，Linux `6.1.43-rockchip-rk3588-rknpu098`，内核版本串标记 0.9.8.1，debugfs 驱动版本报告 0.9.8。
- 官方运行库：匹配头文件的 `librknnrt 2.3.2 (429f97ae6b@2025-04-09T09:09:27)`；SHA256：`d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`。
- 最终受控测试：NPU 1 GHz、DDR 2.112 GHz、CPU 大核 2.304 GHz。CPU 固定到 4–7，OpenMP 4 线程、PASSIVE 等待。临时设置 performance governor，测试后恢复原来的 NPU/DDR ondemand 和 CPU schedutil，见恢复日志。
- 板端构建及结果：`/mnt/ssd/workspace/rknpu-open-validation-20261014`。构建 Release / `-O2`，与项目默认配置一致。

## 公平比较的方法

运算均为 `INT8 A[M,K] × INT8 B[K,N] → INT32 C[M,N]`，使用相同确定性输入。最新 typed 接口使用仓库 `bench_llm_i8` 的固定策略：Ktile≤2048、Ntile≤3072、CPU split-K 归约；没有额外 autotune、压缩权重或改变精度。

双方输入和输出均是紧凑 row-major 主机张量，权重可预处理并复用。计时包括每次请求的输入准备、阻塞 NPU 执行和输出回到 row-major；不包括初始化、权重打包、正确性参考计算。

官方测试分别使用 normal A/C 和 native A/C。normal 包括 A/C memcpy 和 `rknn_matmul_run` 的内部转换；native 包括 CPU A 打包、`rknn_matmul_run`、CPU C 解包。native CPU 转换复用本项目的 OpenMP/NEON 布局函数，给官方提供优化后的转换实现。最终每行选两种官方布局中较快者。

双方明确固定 NPU core 0。每轮预热 3 次、测量 20 次并取中位数，重复 3 轮，再取三个轮次中位数的中位数。`加速比 = 官方耗时 / 开源耗时`。1.5× 表示吞吐增加 50%，等价于耗时减少约 33.3%。

性能轮次按“开源 → 官方 native → 官方 normal”执行；未随机化顺序，没有统计置信区间。表格适用于这台板子及这些策略，不是全形状统计。原始逐次延迟没有被这两个 benchmark 保存，日志保留每轮中位数。

## 单核完整调用结果

单位 ms，官方列已取 normal/native 较快者。每行三个轮次，输出检查通过。

| 矩阵 | M | K | N | 开源 | 官方较快布局 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| hidden | 1 | 4096 | 4096 | 1.628 | 1.551 | 0.95× |
| hidden | 4 | 4096 | 4096 | 1.640 | 1.556 | 0.95× |
| hidden | 128 | 4096 | 4096 | 3.176 | 6.387 | **2.01×** |
| FFN up | 1 | 4096 | 11008 | 4.168 | 4.079 | 0.98× |
| FFN up | 4 | 4096 | 11008 | 4.187 | 4.093 | 0.98× |
| FFN up | 128 | 4096 | 11008 | 8.250 | 16.907 | **2.05×** |
| FFN down | 1 | 11008 | 4096 | 4.393 | 4.079 | 0.93× |
| FFN down | 4 | 11008 | 4096 | 4.413 | 4.112 | 0.93× |
| FFN down | 128 | 11008 | 4096 | 7.901 | 32.425 | **4.10×** |

FFN down 的 M=128 尤其说明基线布局的重要性：官方 normal 约 515 ms，native 约 32 ms。如果只选 normal 基线会得到非常夸张的加速比；上表使用 native 较快者。其余多数形状完整调用的 normal 略快，虽然仅 NPU 阶段不一定如此。

受控性能日志和生成表：[`results/controlled-summary.csv`](results/controlled-summary.csv)。运行 `python3 analyze.py` 可重新汇总。

## 三核接口

开源库的三核完整调用也通过检查。受控条件下另测一轮，20 次调用取中位数：

| 矩阵 | M=1 | M=4 | M=128 |
|---|---:|---:|---:|
| hidden | 0.666 ms | 0.602 ms | 1.678 ms |
| FFN up | 2.200 ms | 2.214 ms | 5.351 ms |
| FFN down | 1.575 ms | 1.601 ms | 4.393 ms |

M=128 对应约 2.16–2.63 TOPS，**包括 CPU packing/reduction/unpacking**，不是单独设备时间。

官方 `rknn_matmul_set_core_mask(..., RKNN_NPU_CORE_0_1_2)` 在本板与这个 runtime 上返回失败，并打印 `Not support core mask: 7`。所以没有给出“相同三核资源”下的官方加速比；上述单核对照才是资源匹配的结果。这里不排除多个官方 context 分核并发的进一步优化，未测试该方案。

## 正确性与原测试程序审查

- x86 主机 CPU-only 构建成功，CTest **15/15 通过**。该结果覆盖主机逻辑，不证明硬件执行。
- 性能轮次沿用 LLM benchmark 原本的每形状 16 个 INT32 CPU 参考检查点。
- 性能测量后另将参考检查扩展到每形状 **4096 个确定性点**，分别检查开源单核、开源三核、官方 normal、官方 native，9 个形状全部通过。参考按 INT32 位模式精确比较；这仍是抽样检查，未逐元素验证全部大型输出。补充正确性日志不用于计算性能表。
- 原 `test_perf --official --ac-native` 直接 memcpy row-major A 到 native A 缓冲区，并不校验官方 native C。这一路原始比值不能单独证明“相同数学输入且正确输出”。本次在独立测试副本中补上按 SDK 定义的 A 打包、C 解包及参考验证。
- 原测试没有固定官方 core mask，计时循环忽略运行返回值，CSV 也不能完整区分官方失败/未验证。本次明确绑定 core 0，并让计时失败或官方验证失败停止测试。
- 初步 prepared-path sweep 覆盖 27 个 M/K/N 组合，3 轮各 100 次；第一轮 6 个组合因原权重预算跳过，后两轮提高到 128 MiB 预算后全测，均通过原参考检查。这些日志使用原 ondemand/schedutil 策略，抖动明显，**不作为最终“快多少”的结论**。该路径不是上表的 typed split-K 接口。

本地 README 的“单核 parity”文字描述旧 prepared-path 对比，不能替代当前 typed 大 K split-K 的实测。仓库还引用了未随独立仓库发布的 `docs/int8_vs_official.md`，本次没有拿它作为原始证据。

## Wiki 核验

读取了用户链接的 [Wiki 首页](https://github.com/happyme531/rknpu2_matmul_open/wiki)、[Hardware Behavior](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior)、[Programming Guide](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)。当前文本没有找到普遍“比官方快 50%”的明确声明；Wiki 声明其数据是历史摘要、未公开全部原始记录，数字依赖驱动、布局、形状和频率。

其单核 1.5–1.85 TOPS、三核 4.18 TOPS 等数据标明 native 布局或 NPU-only，不能直接等同本次完整调用耗时。两个正文页面快照保存在 results。

## 复现材料

- [`benchmark-validation.patch`](benchmark-validation.patch)：相对原 commit 的独立验证修改，包括扩大 LLM 正确性抽样。不改变原 checkout。
- [`validate_official_llm.cpp`](validate_official_llm.cpp)：官方 normal/native 完整调用测试，重用示例的形状和确定性数据生成。
- [`run-controlled.sh`](run-controlled.sh)：板端受控测试脚本、资源锁、governor 保存与恢复。
- [`analyze.py`](analyze.py)：三个轮次中位数汇总并选择官方较快布局。
- [`results/`](results/)：主机检查、板端环境、探索性测试、受控测试、官方三核拒绝日志及频率策略恢复记录。

在原 commit 的独立副本应用 patch，把 `validate_official_llm.cpp` 放到副本 `tests/` 中。按本项目 README 配置 `RK_NPU_OFFICIAL=ON`，指定与 2.3.2 runtime 匹配的头文件和共享库，构建 `test_perf`、`bench_llm_i8`。官方测试用 `g++ -O2 -std=c++17 -fopenmp`，链接 `librk_npu_matmul` 和匹配的 `librknnrt`。`run-controlled.sh` 中的工作目录是本次板端路径；移到别的板子需调整工作目录和频率节点。

官方程序首行 `layout=native/normal` 表示实际布局；随后沿用早期 normal 基线的固定说明字符串，在 native 模式下该固定字符串不准确。实际 native 计时内容以源码和本说明为准。设置 `VALIDATE_NATIVE=1` 选择 native，`VALIDATE_CORRECTNESS=1` 则只测一次调用，用于补充正确性复核。

采用建议：若任务是 INT8、大 K、较大 M 的 prefill，值得继续评估这个实现；若任务是 M=1 decode，应关注三核调度收益，单核算子本身没有在这些形状上超过官方。当前工作区的 FP16 算子与 decoder 不能依据本表直接替换，需另测 FP16 完整调用及误差。
