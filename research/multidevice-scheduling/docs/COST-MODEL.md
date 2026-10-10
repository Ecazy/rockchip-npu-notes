# 驻留与迁移成本模型

`cost_model.py` 将实测 service 查表、内存门禁和显式链路模型分开。驻留命中身份包括 device_id、model hash、layout、dtype、Runtime 版本和 state generation；任一变化使身份不同，不能复用物理地址/编译命令。RK3588 驻留是 DDR BO，未建模跨 Job 隐式 CBUF 状态。

`MemoryBudget` 计算 resident + live tensors + workspace + output + staging，使用实际对齐后的 BO 容量。迁移门禁同时检查 source 与 target；Host staging 是第三个单独资源。访问字节估计不能用来替代分配容量。

`HostStagedMigration.estimate` 的保守串行成本是：

`download(Bd) + host_copy(Bh) + layout + upload(Bu) + setup`。

各 link 使用 latency + 向上取整 bytes/bandwidth；copy 使用显式 Host 带宽。下载、Host 复制、上传字节分别指定，不能混用有效元素字节和 Runtime aligned_size。未知 layout 成本直接拒绝，只有明确 identical_layout 才能省略转换；此时下载和上传字节必须一致。链路标 configured 或 measured，不将本轮尚未测得的 PCIe/USB 带宽伪装成测量结果。

`migration_decision` 比较 source_queue+source_service 与 target_queue+transfer+target_service，要求两边内存都通过且严格缩短预测时间。它按串行成本保守比较，不预设与目标队列重叠、不赋予服务时间系数硬件含义，也未包括未测 model/KV 初始化。

`measured_cost` 要求完整数据条件，返回实测 service、置信度和硬件域。非驻留时必须提供 migration 估计；没有 cold_model_load 时 `cold_start_total_ns=None`，不能让未知冷加载时间默默成为零。ready_service_plus_transfer 只代表已准备好模型后的服务+数据移动，Host prepare/评分不是其中的隐含部分。

```python
import sys
sys.path.insert(0, 'research/multidevice-scheduling')
from cost_model import LinkCost, HostStagedMigration, MemoryBudget, migration_decision
path = HostStagedMigration(LinkCost(2_000_000_000, 5000),
                          LinkCost(2_000_000_000, 5000),
                          host_copy_bandwidth_bytes_per_s=8_000_000_000,
                          setup_ns=1000, identical_layout=True)
cost = path.estimate(download_bytes=65536)
print(cost)  # 配置模型，尚非 RK182X 带宽实测
```

```bash
python3 research/multidevice-scheduling/cost_model.py --example
```

此命令把实际 LUT cell 的 service、task_count 和 BO allocation 转换成模拟场景，生成 `examples/measured-rk3588.json` 与 `results/measured-baselines.json`。全 operation BO 作为一次 Job 的保守 workspace，未另计同一批张量 BO，避免重复容量核算。两台独立逻辑设备是配置拓扑，不等同于 RK3588 三核；RK182X 没有测量 service。

当前模型可用于条件内成本回放和配置敏感性实验。实际跨卡成本仍需在所选 SDK/固件/拓扑上独立测 download/upload/copy/layout/model load/KV 迁移，并补充原始 trace、硬件身份、block 置信区间。单设备事件核算通过没有提升为多卡反事实预测准确性验证。
