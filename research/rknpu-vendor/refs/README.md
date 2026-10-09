---
title: RKNPU 参考仓库快照索引
tags:
  - rknpu
  - npu
  - reference
created: 2026-09-27
updated: 2026-09-27
---

# RKNPU 参考仓库快照

本目录是从 [[项目/RKNPU/README.md|RKNPU 项目索引]] 两份文档里提到的开源库拉下来的**浅克隆快照**，用于离线查阅源码与文档。

- 除注明外，全部为 `git clone --depth 1`（无历史、只看 tip）。
- 标 **固定 commit** 的按文档指定的 sha 精确检出。
- 标 **稀疏检出** 的用 `--filter=blob:none --sparse`，只取有知识价值的路径（预编译 wheel、演示模型等大文件已略过）。
- 复现脚本见 `clone_refs.sh`、`retry_clone.sh`、`retry_mesa.sh`、`fix_clone3.sh`。

## 快照清单

| 目录 | 上游 | commit | 上游日期 | 体积 | 文件数 |
| --- | --- | --- | --- | ---: | ---: |
| `ezrknpu` | Pelochus/ezrknpu | `1fd033c` | 2025-08-03 | 276K | 7 |
| `mesa` | mesa/mesa（gitlab.freedesktop.org） | `9315107` | 2026-09-27 | 2.7M | 12949 |
| `mesa-rk3576` | gahingwoo/mesa-rk3576 | `1f0de93` | 2026-08-31 | 382M | 11257 |
| `npu` | allbilly/npu | `e02707c` | 2026-05-06 | 23M | 148 |
| `npu-oc-tools` | sky-rk3588/npu-oc-tools | `b16bc2d` | 2026-09-02 | 256K | 6 |
| `onnc` | ONNC/onnc | `6628eb7` | 2021-06-18 | 28M | 1819 |
| `orangepi5plus-npu` | widgetii/orangepi5plus-npu | `63a1e99` | 2026-05-08 | 9.1M | 222 |
| `rk3568-npu` | iav/rk3568-npu | `cd7c0b0` | 2026-08-26 | 3.4M | 140 |
| `rk3588-npu` | mtx512/rk3588-npu | `4723947` | 2024-05-30 | 388K | 15 |
| `rkllama` | NotPunchnox/rkllama | `83891fb` | 2026-08-25 | 29M | 80 |
| `rknn-llm` | airockchip/rknn-llm | `878f936` | 2026-06-17 | 782M | 858 |
| `rknn-toolkit2` | airockchip/rknn-toolkit2 | `59a913d` | 2025-07-29 | 2.1G | 1456 |
| `rknpu` | rockchip-linux/rknpu | `e0cb42b` | 2023-08-03 | 621M | 256 |
| `rknpu-reverse-engineering` | phhusson/rknpu-reverse-engineering | `9727f94` | 2024-03-10 | 7.5M | 11 |
| `rknpu_ddk` | airockchip/rknpu_ddk | `838c6e7` | 2020-12-29 | 50M | 329 |
| `rknputop` | ramonbroox/rknputop | `fdb138c` | 2025-11-06 | 1.9M | 6 |
| `tinygrad` | allbilly/tinygrad | `4ccf0a87f` | 2026-09-11 | 80M | 1577 |

## 各仓库为什么在这里

| 目录 | 是什么 | 对哪条线索有用 |
| --- | --- | --- |
| `ezrknpu` | 部署侧封装（不涉及寄存器层） | 端到端使用视角 |
| `mesa` | 上游 Rocket Gallium 驱动与 teflon（**稀疏检出** `src/gallium/drivers/{rocket,teflon}`） | `rkt_regcmd.c` / `rkt_task.c` 是开源 regcmd 发射的上游；本项目 `rnpu_regmap.h` 就由它的 rules-ng-ng 生成 |
| `mesa-rk3576` | Mesa Rocket 驱动移植到 RK3576（含寄存器定义） | RK3576 的寄存器差异对照 |
| `npu` | `rknnops.h` / `rkt_registers.h` 等上游头文件（**固定 commit**） | 寄存器与 NPUOP 定义的一大来源 |
| `npu-oc-tools` | 输出通道寄存器探针（rocket + Teflon） | 寄存器探测手法参考 |
| `onnc` | Skymizer 的开源神经网络编译器框架，`librknnc.so` 被确认是它的私有 fork | 照着它的五阶段流水线理解厂商编译器 |
| `orangepi5plus-npu` | 四合一：`librocketnpu/`（TFLite/ONNX → regcmd 的编译器后端）、`openrknn/`（重写 librknn_api.so 的运行时）、`qemu/`（NPU MMIO 整机仿真）、`rknpu-driver/`（vendored vendor 驱动） | 算子构建主参考；CONV 发射、CBUF 切分、权重打包、逐通道量化都在这里 |
| `rk3568-npu` | 把 Mesa/开源栈移植到 RK3568 | 跨 SoC 兼容性对照 |
| `rk3588-npu` | JPL 的逆向，含 NPU 内部结构与**矩阵乘支持**，配 jas-hacks 博客系列 | 非 1×1 matmul 与 NPU 内部结构的唯一公开来源 |
| `rkllama` | Ollama 风格的 RKNN 部署封装 | 端到端使用视角 |
| `rknn-llm` | LLM 栈：`rkllm-toolkit`（Python 工具）、`rkllm-runtime`（板端库）、`doc`、`examples`、`rknpu-driver` | RKLLM 的 API 边界与示例；`examples` 占 552 MB 主要是演示模型 |
| `rknn-toolkit2` | 官方 SDK 全家桶（**稀疏检出**：doc / rknpu2 示例与头文件 / rknn-toolkit2 文档与示例 / lite2） | `rknpu2/runtime/Linux/librknn_api/include/` 下的 `rknn_api.h`、`rknn_custom_op.h`、`rknn_matmul_api.h`，以及 `doc/` 里的 RKNPU 用户指南、API 参考、**算子支持列表** |
| `rknpu` | 官方内核驱动（GPL）与 `rknn/` 用户态工具：`drivers/` 下按平台分目录，`rknn/` 含 doc/include/python/rknn_api | ioctl ABI、PC 前端编程、中断语义的权威来源 |
| `rknpu-reverse-engineering` | 最早的公开系统性逆向笔记（2023-02），含从 strings 到可运行 CONV 的完整推理链与手写最小寄存器程序 `instrs.h` | 理解「一个算子长什么样」的最短路径 |
| `rknpu_ddk` | 图级算子语义头文件与 doxygen：Conv2DAttr / FCAttr / PoolAttr / BatchNormAttr / ClipAttr 等 | 厂商侧的算子属性定义 |
| `rknputop` | top 风格的 NPU 占用监控 | 运行时可观测性工具 |
| `tinygrad` | tinygrad 的 Rockchip renderer（**固定 commit**） | 另一种寄存器发射实现的写法参考 |

## 使用提示

1. **先查索引再翻代码**：寄存器与 NPUOP 定义优先看 `npu`（`rknnops.h`/`rkt_registers.h`）与 `mesa`（`rkt_regcmd.c`）；ABI 与驱动行为看 `rknpu` 与 `orangepi5plus-npu/rknpu-driver`。
2. **算子语义**先看 `rknpu_ddk` 的头文件，再看 `orangepi5plus-npu/librocketnpu/src/rnpu_regcmd.c` 的实际发射。
3. **厂商文档**在 `rknn-toolkit2/doc/`（用户指南、API 参考、算子支持列表）与 `rknn-llm/doc/`。
4. 快照是**只读参考**：不要在这些仓库里改代码或提交；实验代码一律写在本项目的 `experiments/`。
5. 体积合计约 4 GB，其中 `rknn-llm/examples`（552 MB）与 `rknn-toolkit2/rknpu2`（883 MB）主要是演示模型与预编译库。

%% 全部快照于 2026-09-27 经 Mac（10.8.1.13:7897）代理拉取；直连 GitHub 时链路抖动会坏包，代理下正常。 %%
