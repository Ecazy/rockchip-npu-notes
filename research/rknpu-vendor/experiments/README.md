# 实验目录

当前能力见 [使用与能力指南](../docs/guide.md)，实验结论见 [台账](../docs/RKNPU-实验台账.md)。
脚本与对应源码保留在同一层，便于按原有板端布局复制和复跑。

## 工具与入口

| 用途 | 入口 |
| --- | --- |
| 卷积与反卷积发射 | `min_conv_exp.c` |
| 池化发射 | `min_pool.c` |
| 梯度计算与训练闭环 | `wgrad_real.c` |
| 厂商命令流重放 | `vendor_replay.c` |
| attention 输入实验 | `attn_feed.c` |
| 早期最小实现 | `min_conv.c`、`min_conv_relu.c`、`min_chain.c` |
| INT8 硬化回归 | [run_regression_hard.sh](run_regression_hard.sh) |
| FP16 回归 | [run_fp16_regression.sh](run_fp16_regression.sh) |
| tap 顺序扩展扫描 | [run_taporder_matrix2.sh](run_taporder_matrix2.sh) |
| 推理与训练管线 | `pipeline_pool_conv.sh`、`pipeline_conv_pool_conv.sh`、`pipeline_conv_pool_conv_train.sh` |
| 命令流报告与 LUT | `rknpu_trace_report.py`、`lut_scan.py`、`vendor_op_signature.py` |
| SDPA 模块/数学阶段时间线 | `sdpa_operation_map.py`、`analyze_sdpa_module_occupancy.py`、`render_sdpa_module_timeline_svg.py`、`validate_sdpa_math_mapping.py` |
| GDN 层 RKNN 结构与模块分布 | `build_gdn_layer_onnx.py`、`validate_gdn_layer_onnx.py`、`convert_gdn_layer_rknn.py`、`gdn_feed.c`、`analyze_gdn_trace.py`、`render_gdn_timeline_svg.py` |
| GDN lowering 探针与板端回归 | `probe_gdn_primitives.py`、`cut_gdn_probe.py`、`gdn_cycle.py`、`onnx_mapping_shim.py` |
| GDN CPU/NPU 拆分与阶段耗时 | `build_gdn_layer_onnx.py --front-end/--tail`、`validate_gdn_cuts.py`、`export_gdn_cpu_inputs.py`、`gdn_cpu_layer.c`、`analyze_gdn_fe_timeline.py`、`render_gdn_split_svgs.py` |
| SDPA regcmd issue order | `render_sdpa_regcmd_order_svg.py` |
| 寄存器与任务比较 | `compare_regcmd.py`、`extract_task_blocks.py`、`analyze_vendor_*.py` |
| 模型与权重分析 | `rknn_blob_dump.py`、`vendor_blob_decode.py`、`fp16_blob_decode.py`、`fp16_diff.py` |
| 专用边界和训练探针 | 其余 `run_*_probe.sh`、`run_*_loop.sh` 及标定脚本 |

## 板端复跑

在 RK3588 板端，将源码、寄存器头文件和所需脚本放入同一工作目录：

```bash
gcc -O2 -o min_conv_exp min_conv_exp.c -lm
bash run_regression_hard.sh
bash run_fp16_regression.sh
```

部分脚本固定使用板端 `/mnt/ssd/workspace/rknpu-re/` 下的 `exp2/`、`exp3/` 和 `pool/`。
复跑前查看脚本开头的路径和用法。
上述回归需要真实 NPU，离线语法检查仅能验证脚本可解析。

## 实验依据

[results/](results/) 按实验主题和日期保存日志、trace、拟合结果及说明。
早期根目录日志已归入 `results/baseline_20260927/`。
结果目录内的模型生成、转换和数值验证脚本用于复核对应实验，予以保留。
`vendor_models/` 保存本地模型，按 `.gitignore` 规则排除。

旧报告和一次性脚本的查看方法见 [清理说明](../README.md#清理说明)。
