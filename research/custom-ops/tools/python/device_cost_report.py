#!/usr/bin/env python3
"""Generate reports from strict analysis outputs, without changing raw evidence."""
import argparse
import collections
import json
import pathlib
import statistics
from device_cost_analyze import bootstrap


def read(root, name='analysis.json'):
    return json.loads((root/name).read_text())


def link(root):
    return 'custom-ops/results/'+root.parent.name+'/'+root.name


def grouped(document):
    groups=collections.defaultdict(list)
    for run in document['runs']:
        c=run['config']
        groups[(c['workload'],c['workers'],c['strategy'],c['mask'],c['allowed'],c['start'])].append(run)
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cost','foreground','resident','saturation','interference','tail','trace'):
        p.add_argument('--'+name,type=pathlib.Path,required=True)
    p.add_argument('--research',type=pathlib.Path,default=pathlib.Path('research'))
    args=p.parse_args();roots={k:getattr(args,k) for k in ('cost','foreground','resident','saturation','interference','tail','trace')}
    docs={k:read(r) for k,r in roots.items()}
    assert all(d['status']=='PASS' for d in docs.values())
    model=read(args.cost,'model.json')
    lines=['# RK3588 单设备任务成本模型 v2','',
           '2026-10-10，Orange Pi 5 Plus / RK3588，厂商 RKNPU 0.9.8，FP16。',
           'NPU 1 GHz，DMC performance，CPU performance；实际频率与温度采样随各轮证据保存。',
           '',f"正式隔离轮：102 个 cell × 5 个独立随机化 block × 200 次 operation，合计 102000 个 operation / {docs['cost']['measured_submits']} 次 Submit。",'',
           f"证据：[隔离成本轮]({link(args.cost)}/analysis.json)，[模型 JSON]({link(args.cost)}/model.json)。",'',
           '## 测量与数据边界','',
           '- 一个 Job 是一次阻塞 Submit；separate 算子产生多个 Job，通过 operation_id 和 submit_index 关联，不能当成一个 Submit。',
           '- service 是驱动 commit→IRQ 取时的包络，包括寄存器配置与部分中断延迟；不是纯硬件执行时间。',
           '- response=ioctl_end-host_begin；api_response=host_done-host_begin；prepare 与 post_submit 分开记录。',
           '- Host I/O 路径的 operation 窗口包含输入上传和输出下载；device_ready 路径输入先上传。参考计算、逐元素评分、padding/guard 检查在操作窗口外。',
           '- closed-loop 完成吞吐的整个 makespan 包含请求之间的准备、评分、日志和 Host 排程间隔；不能视为纯设备吞吐。',
           '- 所有原始 Host 字段存整数 ns；展示 us 不意味着时间点具有纳秒精度。无 tracing 轮的 queue=null，不能以 wall-service 代替 queue。','',
           '## 特征与模型','',
           '从实际 co_program 的 reads/output 推算访问量；从实际 count 获取命令字节数。独立 task 的工作量求和，组合链包括 identity Conv 的物理 MACs、展开 bias、中间 Cube 和大输出分片。',
           'estimated_bytes 是物理张量访问量估计，未计缓存重用/实际 DDR 重读；allocated_tensor_bo_bytes 是实际持有张量 BO 的大小，包含 allocator slack/guard，不含每次 Submit 临时描述符/命令 BO。',
           '', '经验模型采用 ridge=0.01 的线性回归，特征为截距、task_count、physical_MACs/1e6、estimated_bytes/1e4、command_bytes/1e3。每个 workload/program/core 分开拟合；系数不赋予未经识别的硬件含义。',
           '以 cell/block 的中位数拟合。整组几何按固定 seed 留出至少20%，不随机拆重复请求；另用完整最后一个 block 做时间外验证。训练几何、留出几何、基线误差、特征边界和系数均在 model.json。',
           '', '工程门限为 service 留出相对误差 p50≤15%、p90≤30%。未达到的族标 experimental，样本不足的族不给预测。Host API response 的误差独立报告，不以 service 达标替代。','',
           '| workload / program / mask | service p50误差 | service p90误差 | API response p50误差 | 状态 |','|---|---:|---:|---:|---|']
    for key,value in sorted(model['models'].items()):
        if value['status']=='insufficient_geometry':
            lines.append(f"| {key} | — | — | — | insufficient_geometry |")
        else:
            lines.append(f"| {key} | {100*value['relative_error_p50']:.1f}% | {100*value['relative_error_p90']:.1f}% | {100*value['api_response_relative_p50']:.1f}% | {value['status']} |")
    lines += ['', '## 使用','', '```python', 'import sys', "sys.path.insert(0, 'research/custom-ops/tools/python')", 'from device_cost_analyze import DeviceCostModel',
              f"model = DeviceCostModel('research/{link(args.cost)}/model.json')",
              "# 已发射的 64×64 HardSigmoid，4 个独立 program，单 Submit。",
              "print(model.predict('lut', 1, task_count=4, physical_macs=0,",
              '                    estimated_bytes=65536, command_bytes=34240))', '```','',
              '返回 driver_service_us、api_response_us、模型状态和 queue_us=null。特征越界、未知族或不足几何的模型拒绝预测；特征范围内的新组合仍需独立校准，不能把包围盒当成全部形状验证。','',
              '本轮模型限定已测单 tile M/K/N≤256，LUT 独立 task/job≤64，FP16 与固定时钟/驱动环境。256×256 组合输出拆成两个 LUT task，因此为4个program。K>256累计、PPU、INT8和变频没有由本模型验收。','',
              '## 复跑','', '```bash','bash research/custom-ops/scripts/run-device-cost-board.sh cost 5 200 20',
              f"python3 research/custom-ops/tools/python/device_cost_analyze.py research/{link(args.cost)}", '```','',
              '部署脚本保存实际源码包和哈希，在新板端目录重编；失败不覆盖旧轮次。派生分析允许用新解析器重算，并另存 review-source-manifest.json。原始 JSONL、trace、配置和源码包保留。',
              '初次隔离轮 journalctl 时间格式错误，原失败日志保留；修正格式后对同一 UTC 时间窗只读回查，kernel-journal-reviewed.txt 为无事件、退出0。后续正式轮使用已修正采集格式。', '']
    (args.research/'DEVICE-COST-MODEL.md').write_text('\n'.join(lines))
    lines=['# RK3588 多 Job 并发与资源干扰 v2','',
           '2026-10-10，FP16 / 厂商驱动0.9.8 / NPU 1 GHz。Job=一次阻塞Submit；依赖链整体在单核。mask 1/2/4 为单核，0仅为厂商AUTO；候选集合由用户态产出单个位后提交。','',
           '## 正式证据','', '| 实验 | runs | 正式Submit | 证据 |','|---|---:|---:|---|']
    for name in ('foreground','resident','saturation','interference','tail','trace'):
        lines.append(f"| {name} | {len(docs[name]['runs'])} | {docs[name]['measured_submits']} | [analysis]({link(roots[name])}/analysis.json) |")
    lines += ['', '每轮请求预算固定，worker各分得确定配额；同策略对照保持总工作量、形状、program清单、CPU分配和运行路径一致。每个worker独立fd/BO，输入有两种符号变体，输出预置NaN；全部请求逐元素评分，包含padding与BO guard。',
              'foreground逐请求评分；resident预上传输入，每请求保留独立输出，全部worker完成测量后才逐请求验收，不是稀疏抽查。resident测量窗口仍包含现有适配器的每次命令/描述符分配、打包与同步，尚未做持久提交上下文优化。',
              'continuous为一次启动后持续closed-loop；burst为可中止的逐轮启动门，失败不会让其他worker永久卡在barrier。',
              'CPU按worker编号固定到0–7；1/2/3worker在A55，8worker还使用A76。与单worker比较包含Host并行和CPU类型影响，不能直接声称NPU三核加速；以下重点采用同worker数策略对照。','',
              '## 较长 Job 的同工作量对照','',
              'saturation使用256³ MatMul和四task组合算子。下面为同8worker、每轮600 operation、5个配对block：','',
              '| workload | 同核完成/s | 分核完成/s | 分核/同核 speedup [95%CI] | 分核/同核 service中位数比 |','|---|---:|---:|---:|---|']
    sat=grouped(docs['saturation'])
    for family in ('matmul','mm_bias_hsig'):
        fixed=sat[(family,8,'fixed',1,7,'continuous')];split=sat[(family,8,'split',1,7,'continuous')]
        pairs={r['block']:r for r in fixed};ratios=[pairs[r['block']]['makespan_ns']/r['makespan_ns'] for r in split]
        ci=bootstrap(ratios);services=[r['service_us']['p50']/pairs[r['block']]['service_us']['p50'] for r in split]
        lines.append(f"| {family} | {statistics.mean(r['operations_per_s'] for r in fixed):.0f} | {statistics.mean(r['operations_per_s'] for r in split):.0f} | {statistics.mean(ratios):.3f} [{ci[0]:.3f},{ci[1]:.3f}] | {statistics.mean(services):.3f} |")
    lines += ['', '分核减少同核排队并提高完成吞吐，同时服务包络膨胀。这个现象说明并发会改变服务成本；未据此确定DDR/CBUF/IRQ中的具体瓶颈。短Job foreground中同核与分核吞吐接近，Host检查和提交开销占比很高，原数据保留。','',
              '## 排队与服务的区间证据','',
              'tracing诊断与无trace性能轮分开；不能组合成同一请求的分解。内核代码经过既有audit，trace loss和未知提交会拒绝PASS。',
              'filtered-out调用/中断可能在同一显示层级贡献多个ktime_get；解析器保留候选的包围区间，再用驱动service时差收紧，不选择其中一个冒充精确时间点。重复/同量化时刻的候选、FIFO顺序歧义仍拒绝。',
              '', '| workload / workers / policy | IRQ核 | queue上下界中位数μs | service中位数μs | 最大queue区间宽μs |','|---|---|---:|---:|---:|']
    for r in docs['trace']['runs']:
        c,q=r['config'],r['queue']
        lines.append(f"| {c['workload']} / {c['workers']} / {c['strategy']} | {q['cores']} | {q['queue_lower_median_us']:.3f}–{q['queue_upper_median_us']:.3f} | {q['service_median_us']:.3f} | {q['queue_width_max_us']:.3f} |")
    lines += ['', '上下界分别取中位数，不是某一真实请求的区间。空闲核的微秒级enqueue→commit也包括派发时间，不必然是等待另一Job。现有观测仍没有精确enqueue/kick时间点或纯硬件执行边界。','',
              '## 尾延迟','', 'tail每条件5000 operation/block × 5block=25000请求；以下为block p99的均值及按block bootstrap的95%CI。数值与guard全量通过。','',
              '| workload / workers / policy | API operation p99 μs [95%CI] | 完成/s |','|---|---:|---:|']
    for key,runs in sorted(grouped(docs['tail']).items()):
        values=[r['operation_us']['p99'] for r in runs];ci=bootstrap(values)
        lines.append(f"| {key[0]} / {key[1]} / {key[2]} | {statistics.mean(values):.2f} [{ci[0]:.2f},{ci[1]:.2f}] | {statistics.mean(r['operations_per_s'] for r in runs):.0f} |")
    lines += ['', '公平性用共同active窗口内的完成进展计算Jain，异构负载不给未经归一化的公平性分数；最终相等配额不能证明公平性。request_overlap_fraction只表示Host请求窗口重叠，不能解释成硬件执行重叠。','',
              '## CPU与内存背景干扰','',
              'victim在CPU0提交；背景固定到A76的CPU4–7。背景私有缓冲总量256MiB，超过已盘点3MiB LLC；读写软件流量、checksum、线程/缓冲参数及生命周期均保留。软件GB/s不是DDR硬件计数器。DMC频率固定并回读，线程数仅为背景强度参数。','',
              '| victim | 背景 | service slowdown [95%CI] |','|---|---|---:|']
    configs={r['cell']:r['config'] for r in docs['interference']['runs']}
    for row in sorted(docs['interference']['comparisons'],key=lambda r:r['cell']):
        c=configs[row['cell']]
        if c['background']:
            ci=row['service_slowdown_ci95'];lines.append(f"| {c['workload']} | {c['background']} | {row['service_slowdown']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] |")
    calibration=read(args.interference,'interference-model.json')
    lines += ['', '不同核的同质/异质pair覆盖核集合0/1、0/2、1/2，另有三核同质Job；异质条件交换victim/competitor类型。service包含驱动配置与IRQ延迟，部分worker的CPU类型不同，具体资源归因保留unresolved。',
              f"[干扰校准表]({link(args.interference)}/interference-model.json) 用前4个block预测第5个block；service乘数留出误差p50={100*calibration['relative_error_p50']:.2f}%、p90={100*calibration['relative_error_p90']:.2f}%。",
              '另提供固定常数/线性基线的组合留出检查；这是事后探索性检验，未提升为普适预测能力。校准仅覆盖64×64所测条件，不支持未知形状、背景或由线程数推导反事实DDR带宽。','',
              '## 实现、复跑与限制','', '```bash',
              'bash research/custom-ops/scripts/run-device-cost-board.sh resident 5 600 20',
              'bash research/custom-ops/scripts/run-device-cost-board.sh saturation 5 600 20',
              'bash research/custom-ops/scripts/run-device-cost-board.sh interference 5 600 20',
              'bash research/custom-ops/scripts/run-device-cost-board.sh tail 5 5000 20',
              'bash research/custom-ops/scripts/run-device-cost-board.sh trace 1 24 2', '```','',
              '入口：tools/c/cost_probe.c、interference_load.c、tools/python/device_cost_campaign.py、device_cost_schema.py、device_cost_analyze.py。只使用C11/POSIX和Python标准库。',
              '外层持有一次NPU锁及trace锁；保存/恢复NPU min_freq、DMC/CPU governor和tracer设置，回读核验。全局旧trace内容保存但不能重新注入ring buffer；SIGKILL无法保证自动恢复，下一次须预检。',
              '失败保留构建日志、部分JSONL和原始trace；不自动重试。两个早期构建失败和首次trace解析拒绝已保留，不计入正式通过样本。修复后使用新源码包和新目录验收。',
              '无抢占、链内迁移或精确queue-depth观测；worker数不等于队列深度。没有采用M/M/1泊松假设或预测调度器。持久Submit上下文、精确job-ID tracepoint、PPU/大K成本校准为后续扩展。','']
    (args.research/'JOB-CONCURRENCY-INTERFERENCE.md').write_text('\n'.join(lines))
    print('Wrote DEVICE-COST-MODEL.md and JOB-CONCURRENCY-INTERFERENCE.md')


if __name__=='__main__':main()
