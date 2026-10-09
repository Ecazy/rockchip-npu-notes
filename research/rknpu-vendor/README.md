# RKNPU 逆向实验工作区

本仓库保存 RK3588 NPU 命令流采集工具、手写算子发射器和真机实验依据。

## 文档入口

- [使用与能力指南](docs/guide.md)：能力、限制、采集分析及上游规格。
- [实验台账](docs/RKNPU-实验台账.md)：实验过程、结论修正和证据索引。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `source/hook/rknpu/` | ioctl Hook 和命令流分析器 |
| [experiments/](experiments/README.md) | 算子发射器、回归、专用探针和离线分析工具 |
| `experiments/results/` | 原始捕获、分析结果及实验说明 |
| `experiments/results/baseline_20260927/` | 早期基线日志，供历史结论复核 |
| `docs/` | 上述 2 份技术文档 |
| [refs/](refs/README.md) | 本地上游源码快照及索引 |
| `infra/` | 原工程构建配置，依赖原工程的目录环境 |
