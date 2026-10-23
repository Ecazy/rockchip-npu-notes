# Device-Cost / Concurrency v2 实施记录

2026-10-10。当前执行器、schema、配置和分析均为 v2；原未完成 v1 源码保存于本地 `results/2026-10-10/handoff-v1-164818/`，旧 P0 证据不覆盖。

完整结果见 [成本模型](../scheduling/DEVICE-COST-MODEL.md) 和 [并发/干扰](../scheduling/JOB-CONCURRENCY-INTERFERENCE.md)。原始运行记录、源码摘要与失败轮次保留在本地结果目录；它们不随 Git 提交分发。

## 可执行组件

- `tools/c/cost_probe.c`：CNA/DPU原语与组合链，独立fd/BO、多worker、显式核映射/候选轮询/AUTO、连续/可中止burst、每Submit与operation计时。
- `tools/c/interference_load.c`：CPU compute与私有缓冲streaming，总缓冲256MiB，背景强度用软件流量记录。
- `tools/python/device_cost_campaign.py`：新目录、固定seed随机化block、一次外层锁、环境控制/恢复、子进程硬超时、源码与binary哈希、无trace性能及独立tracing。
- `tools/python/device_cost_schema.py`：真实配置与operation/Submit/task预算；没有待冻结模板。AUTO仅允许全部候选核；deferred池估计限制在总1GiB。
- `tools/python/device_cost_analyze.py`：缺失/重复请求、错核、单位、数值、guard、特征、索引、时序与恢复门禁；按完整几何留出和最后block验证；预测域外拒绝。
- `tools/python/device_cost_report.py`：由analysis/model JSON生成两份报告，报告中的CI按block计算。
- `scripts/run-device-cost-board.sh`：使用忽略的.env部署当前源码，在新目录重编、运行并回收失败证据。

## 正式轮

固定预算配置保存在 `tools/python/device_cost_configs/`；其中 `trace-v2.json` 对齐正式轮的 5 个 cell × 24 次 operation，共 120 次 Submit。部署脚本按当前执行器的 suite 和命令行预算生成实际 `config.json`，不会直接读取这些配置快照；复核以各结果目录保存的实际配置为准。

| suite | runs | measured Submit |
|---|---:|---:|
| cost | 510 | 129000 |
| concurrency（前台检查） | 435 | 261000 |
| resident（延后全量检查） | 435 | 261000 |
| saturation（256³长Job） | 90 | 54000 |
| interference | 150 | 90000 |
| tail（每条件25000请求） | 30 | 150000 |
| trace（独立诊断） | 5 | 120 |

合计945000次无trace正式Submit，另120次trace；不含warmup和smoke。每个请求全量数值、padding与BO guard验收，全部矩阵完整且环境恢复通过。

## 计时、验证与限制

一条submit记录对应一次阻塞ioctl；operation包含其全部program。separate模式产生多条submit，各自只记录该program的MACs/bytes/command bytes。大输出组合实际4program，不固定写3。

foreground计时之外逐请求检查，makespan包含请求间这些间隔；resident测量前准备输入和每请求独立输出，全部worker完成计时后再全量评分。仍有每次提交命令/描述符打包分配，没有宣称零分配或纯设备吞吐。

tracing保持审计内核/FIFO/IRQ核验。多个严格递增ktime_get候选用包围区间，不选一个冒充精确时间；重复候选、歧义、丢失、未知提交仍拒绝。原2026-10-09正式证据复算通过。

主机和最新板端CTest均11/11通过；v2证据负例20项，调度解析正反例18项。纯MatMul服务预测未达到预登记门限，标experimental；干扰校准仅限已测64×64条件。没有精确enqueue/kick、纯硬件时间、DDR计数器或跨设备结论。

两轮早期构建失败与一轮trace解析拒绝均保留。初次cost轮kernel journal格式错误，保留原错误日志，并对同一UTC时窗只读补查确认无事件。未通过轮次不计入正式样本。
