# RK3588 单设备任务成本模型 v2

2026-10-10，Orange Pi 5 Plus / RK3588，厂商 RKNPU 0.9.8，FP16。
NPU 1 GHz，DMC performance，CPU performance；实际频率与温度采样随各轮证据保存。

正式隔离轮：102 个 cell × 5 个独立随机化 block × 200 次 operation，合计 102000 个 operation / 129000 次 Submit。

证据：[隔离成本轮](../custom-ops/results/2026-10-10/device-cost-cost-20261010-165249/analysis.json)，[模型 JSON](../custom-ops/results/2026-10-10/device-cost-cost-20261010-165249/model.json)。

## 测量与数据边界

- 一个 Job 是一次阻塞 Submit；separate 算子产生多个 Job，通过 operation_id 和 submit_index 关联，不能当成一个 Submit。
- service 是驱动 commit→IRQ 取时的包络，包括寄存器配置与部分中断延迟；不是纯硬件执行时间。
- response=ioctl_end-host_begin；api_response=host_done-host_begin；prepare 与 post_submit 分开记录。
- Host I/O 路径的 operation 窗口包含输入上传和输出下载；device_ready 路径输入先上传。参考计算、逐元素评分、padding/guard 检查在操作窗口外。
- closed-loop 完成吞吐的整个 makespan 包含请求之间的准备、评分、日志和 Host 排程间隔；不能视为纯设备吞吐。
- 所有原始 Host 字段存整数 ns；展示 us 不意味着时间点具有纳秒精度。无 tracing 轮的 queue=null，不能以 wall-service 代替 queue。

## 特征与模型

从实际 co_program 的 reads/output 推算访问量；从实际 count 获取命令字节数。独立 task 的工作量求和，组合链包括 identity Conv 的物理 MACs、展开 bias、中间 Cube 和大输出分片。
estimated_bytes 是物理张量访问量估计，未计缓存重用/实际 DDR 重读；allocated_tensor_bo_bytes 是实际持有张量 BO 的大小，包含 allocator slack/guard，不含每次 Submit 临时描述符/命令 BO。

经验模型采用 ridge=0.01 的线性回归，特征为截距、task_count、physical_MACs/1e6、estimated_bytes/1e4、command_bytes/1e3。每个 workload/program/core 分开拟合；系数不赋予未经识别的硬件含义。
以 cell/block 的中位数拟合。整组几何按固定 seed 留出至少20%，不随机拆重复请求；另用完整最后一个 block 做时间外验证。训练几何、留出几何、基线误差、特征边界和系数均在 model.json。

工程门限为 service 留出相对误差 p50≤15%、p90≤30%。未达到的族标 experimental，样本不足的族不给预测。Host API response 的误差独立报告，不以 service 达标替代。

| workload / program / mask | service p50误差 | service p90误差 | API response p50误差 | 状态 |
|---|---:|---:|---:|---|
| ('lut', 'chain', 1) | 1.9% | 3.6% | 1.9% | validated |
| ('lut', 'chain', 2) | 1.8% | 5.6% | 1.6% | validated |
| ('lut', 'chain', 4) | 2.1% | 5.7% | 1.7% | validated |
| ('matmul', 'chain', 1) | 23.0% | 38.1% | 2.1% | experimental |
| ('matmul', 'chain', 2) | 23.2% | 34.1% | 2.0% | experimental |
| ('matmul', 'chain', 4) | 22.9% | 34.2% | 2.1% | experimental |
| ('mm_bias_hsig', 'chain', 1) | 6.8% | 8.7% | 2.1% | validated |
| ('mm_bias_hsig', 'chain', 2) | 5.4% | 7.1% | 2.8% | validated |
| ('mm_bias_hsig', 'chain', 4) | 5.6% | 7.4% | 2.3% | validated |
| ('mm_bias_hsig', 'separate-0', 1) | 2.4% | 4.8% | 2.5% | validated |
| ('mm_bias_hsig', 'separate-0', 2) | 10.5% | 13.4% | 3.4% | validated |
| ('mm_bias_hsig', 'separate-0', 4) | 10.8% | 13.6% | 4.1% | validated |
| ('mm_bias_hsig', 'separate-1', 1) | 12.9% | 16.1% | 3.3% | validated |
| ('mm_bias_hsig', 'separate-1', 2) | 10.7% | 13.6% | 3.5% | validated |
| ('mm_bias_hsig', 'separate-1', 4) | 10.5% | 13.4% | 4.1% | validated |
| ('mm_bias_hsig', 'separate-2', 1) | 0.6% | 3.3% | 0.9% | validated |
| ('mm_bias_hsig', 'separate-2', 2) | 0.6% | 3.1% | 1.3% | validated |
| ('mm_bias_hsig', 'separate-2', 4) | 0.8% | 3.2% | 1.6% | validated |
| ('mm_bias_hsig', 'separate-3', 1) | — | — | — | insufficient_geometry |
| ('mm_bias_hsig', 'separate-3', 2) | — | — | — | insufficient_geometry |
| ('mm_bias_hsig', 'separate-3', 4) | — | — | — | insufficient_geometry |

## 使用

```python
import sys
sys.path.insert(0, 'research/custom-ops/tools/python')
from device_cost_analyze import DeviceCostModel
model = DeviceCostModel('research/custom-ops/results/2026-10-10/device-cost-cost-20261010-165249/model.json')
# 已发射的 64×64 HardSigmoid，4 个独立 program，单 Submit。
print(model.predict('lut', 1, task_count=4, physical_macs=0,
                    estimated_bytes=65536, command_bytes=34240))
```

返回 driver_service_us、api_response_us、模型状态和 queue_us=null。特征越界、未知族或不足几何的模型拒绝预测；特征范围内的新组合仍需独立校准，不能把包围盒当成全部形状验证。

本轮模型限定已测单 tile M/K/N≤256，LUT 独立 task/job≤64，FP16 与固定时钟/驱动环境。256×256 组合输出拆成两个 LUT task，因此为4个program。K>256累计、PPU、INT8和变频没有由本模型验收。

## 复跑

```bash
bash research/custom-ops/scripts/run-device-cost-board.sh cost 5 200 20
python3 research/custom-ops/tools/python/device_cost_analyze.py research/custom-ops/results/2026-10-10/device-cost-cost-20261010-165249
```

部署脚本保存实际源码包和哈希，在新板端目录重编；失败不覆盖旧轮次。派生分析允许用新解析器重算，并另存 review-source-manifest.json。原始 JSONL、trace、配置和源码包保留。
初次隔离轮 journalctl 时间格式错误，原失败日志保留；修正格式后对同一 UTC 时间窗只读回查，kernel-journal-reviewed.txt 为无事件、退出0。后续正式轮使用已修正采集格式。
