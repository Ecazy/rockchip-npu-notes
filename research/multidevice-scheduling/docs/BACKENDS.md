# Hardware Backend Interface

统一契约在 `backends/interface.py`：`submit`、`wait`、`transfer`、`query_memory`，以及显式 open/close、buffer/job/token 所有权管理。各设备 payload 与 native handle 都是独立类型和身份；同类型的不同适配实例也不能交换 handle。`identity` 是只读属性，capabilities 为 available/unavailable/unknown，后两者拒绝执行。

| 适配层 | 执行实现 | 当前证据 |
|---|---|---|
| RK3588 | ABI 2 C bridge→现有 `co_execute/co_schedule_execute`、BO 分配与 prep/fini cache sync、既有 LUT emitter | 主机构建及真实板端 smoke |
| RK182X | 独立 RKNN3 model/session payload 与版本/platform/capability 契约 | 契约验证；未绑定本地执行 SDK，无实机执行 |
| Mock | 内存字节数组、确定性时钟 | 测试替身，不代表硬件 |

## 操作与生命周期

RK3588 的 submit 是同步调用，返回 COMPLETE token；wait 只读取真实完成状态。不支持可取消 submit deadline，传入该 deadline 会在运行前拒绝。现有底层驱动超时与错误保留，失败不回退 CPU。真实异步完成目前仅由 Mock 契约检验；wait 超时保持 PENDING，随后可以继续等待，不能以超时推断设备已经停止。

pending token/job/被其引用的 buffer 不能释放，pending 工作不能 close。prepared job 固定其 tensor buffer，不允许提前 free 或改变 descriptor。close 成功后旧句柄全部失效，重新 open 不能复活。失败 submit 不留下调用方无法取回的 token。异常关闭仍需调用方处理错误，不声称设备资源已成功回收。

`transfer(buffer, direction='upload', data=bytes, offset=0)` 与 `direction='download', length=N` 分别操作 Host↔设备。每次最多 64MiB、禁止越过 payload；零字节明确成功但仍检查所有权。`transfer_between` 先在 Host 完成下载再上传，双方范围/能力提前检查，源数据保留；不共享 DMA 地址/fd/native handle，不声称目标部分写入失败时可以事务回滚。观测时间为 Host API wall time，包含 copy/sync，不替代设备服务时间。

`query_memory` 回报本适配器实际持有的 payload、allocated bytes 和 live buffers。RK3588 BO 容量包含 slack/guard；没有公开的设备/system 总量保持 None。不把 `/proc/meminfo` 空闲量当作本进程保证可分配的 NPU 容量。RK182X 未来应分别对接 `QUERY_DEVICE_MEM_INFO`、`CORE_MEM_SIZE` 和 allocation 信息；当前未绑定时查询不可用。

原生 bridge 用不复用的整数 ID 活跃表校验 handle，避免访问任意/已释放地址；job 引用阻止 buffer 被提前释放。Python ctypes 调用在进程内序列化，保护 C registry；直接 C ABI 调用也必须串行。当前适配器不承诺通过多个 Python 线程复现已有 C 多进程并发吞吐。

## 构建与真实 smoke

```bash
cmake -S research/multidevice-scheduling/backends/bridge \
  -B research/multidevice-scheduling/build-bridge -DCMAKE_BUILD_TYPE=Release
cmake --build research/multidevice-scheduling/build-bridge --target co_backend_bridge -j4
python3 -m unittest discover -s research/multidevice-scheduling/tests -p 'test_*.py'
# 在有 RK3588 的板端，显式执行真实操作：
CO_BACKEND_BRIDGE=/path/libco_backend_bridge.so RKNPU_CORE_MASK=1 \
  python3 research/multidevice-scheduling/backends/smoke.py
# 从本仓库主机，使用既有忽略的 .env 凭据，在新目录上传实际源码并编译：
bash research/multidevice-scheduling/scripts/run-backend-board.sh
```

不修改现有 `custom-ops`，CMake 将原静态目标以 PIC 编译并链接进 bridge；依赖仍采用现有 upstream 清单。smoke 通过既有 `co_emit_lut` 生成 RK3588 命令，提交 32×32 FP16 HardSigmoid，1024 个零输入输出逐字节应为精确 0.5，检查两个 BO 的整个 slack/guard。找不到硬件退出 77，数值/硬件错误非零，无 CPU fallback。

最新真实 ABI 2 证据在 [backend-20261010-191327](../results/backend-20261010-191327/smoke.json)，包括源码包/哈希、构建日志、uname、原始 stdout/stderr 与退出状态。1024 元素和 guard 通过，持有两个 BO 共 69760 字节；这是接口 smoke，不是新的服务时间/吞吐校准。此前 ABI 1 验证在独立目录保留。主机 Mock 或构建测试不会提升为 HARDWARE 证据。

## RK182X 独立边界

`RK182XPayload` 只描述 RKNN3 model/session/blob/input/output，不接受 RK3588 words，也没有继承 RK3588 ioctl/struct。固定官方 V1.1.0 头文件与 Runtime 导出检查见 [RKNN3 静态分析](RKNN3-STATIC-ANALYSIS.md)。公开 SDK 存在，与本地尚未绑定执行 SDK 是两件分别记录的事实。

当前 runtime 注入仅允许 `contract_only=True`，严格检查已知 major/minor、平台和 Boolean capability；metadata-only 对象不能声明真实执行能力。所有执行/内存/迁移操作 unavailable，P2P 未证明；`migration_plan` 仅说明将来 Host staging 路径。native RKNN3 的模型加载、分配对齐、mem_sync、run/session 及回调/query_state 完成绑定需要另行实现并在卡上验收。

这完成了独立适配层及拒绝错误 ABI 的接口验收，不宣称已实现 RK182X 真机 backend。后续无需改变上层调度接口，可在此独立层完善厂商 Runtime 绑定；不能用 RK3588 片内三核数据校准 RK182X 多卡。
