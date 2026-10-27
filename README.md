# Rockchip NPU：逆向能力与算子实验工作区

当前开发目标为 Orange Pi 5 Plus / RK3588 / 厂商 RKNPU 0.9.8 / FP16。
仓库包含本地自写算子、调度研究工具、历史厂商实验，以及上游 Rocket 硬件知识与组件索引。

## 从这里开始

文档按“当前能力与使用 → 专题参考 → 历史与原始证据”阅读；完整路由见 [文档导航](docs/README.md)。

| 需求 | 入口 |
|---|---|
| 了解现在能做什么、哪些尚未验证 | [能力总览](docs/CAPABILITIES.md) |
| 理解目录、依赖、扩展与证据约定 | [项目结构](docs/PROJECT-STRUCTURE.md) |
| 开发组合算子或运行文件 CLI | [自写算子框架](research/custom-ops/README.md) |
| 将 TileLang 静态计算编译到 RK3588 | [TileLang RKNPU 后端](research/tilelang-rknpu/README.md) |
| 显式分配核、分离排队与服务时间 | [调度 Profiling](research/scheduling/MULTICORE-SCHEDULING-PROFILING.md) |
| 理解 K 分片与设备累计 | [FP16 K 分片](research/kernels/FP16-K-TILING.md) |
| 核对 task 链与寄存器组交接 | [Two-Task Chaining](research/scheduling/TWO-TASK-CHAINING.md) |
| 查历史实验与外部组件 | [研究目录索引](research/README.md) |
| 阅读完整上游知识目录 | [旧综合索引](docs/archive/LEGACY-README.md) / [上游指南](guide/README.md) |

## 构建与检查

先按 [组件清单](research/upstream-components/README.md) 恢复固定版本的 `rocket-userspace` 与 `rknpu-submit`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure

# 上传新源码、真机复跑并回收证据；连接配置在本机 .env。
bash research/custom-ops/scripts/run-schedule-board.sh
```

Host 测试不提交 NPU 任务；硬件验证必须在匹配的设备/驱动环境进行。
子项目仍可通过 `cmake -S research/custom-ops -B research/custom-ops/build-host` 独立构建。

## 当前可交付能力

- FP16 `HardSigmoid(XW.T+b)` 与通用 Linear：M/N/K 1..4096，自动分块；每个输出块一次链式 Submit，设备 Cube 交接与 DPU 分片累计。
- 有限静态 FP16 图、RoPE、主机 KV self-attention 与单层 decoder block；混合 CPU 阶段及验证范围见能力总览。
- 有界 MatMul、EW、LUT、末维归约、PPU、固定重排，以及明确包含 CPU 阶段的混合归一化。
- 逐 Submit 核分配、候选集合轮询和 AUTO 映射核验；队列时间区间、驱动服务时长与其他开销分离。

具体支持范围、证据等级和缺口以 [能力总览](docs/CAPABILITIES.md) 为准。主线 Rocket 的结论不能自动推广到厂商路径。

## 文件与证据

源码、公共接口、脚本与小型测试 fixture 纳入版本控制范围；构建、独立上游 checkout 和大型原始结果忽略。
历史证据保留原路径及原始校验清单。旧文档中的能力口径见 [历史进展快照](RKNPU-STATUS.md)。

本仓库包含 AI 辅助生成的研究笔记；结论须按对应来源与实验环境判断。
许可见 [LICENSE](LICENSE)，第三方来源及组合许可见 [SOURCES.md](SOURCES.md) 和各组件文件。
