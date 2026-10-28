# 创建时 command BO 打包复用

日期：2026-10-28。本阶段只验证“命令准备”这一项：对
`program_cache_ready` 的 single-submit workspace，在创建时把命令 BO 内容、
task descriptors、链尾、flags 和句柄列表准备好，后续 submit 复用；原
per-submit 打包路径保留为 control 和 fallback。

程序缓存、上传、guard、输出同步、下载、发布和数值政策不变。没有新增公共
结构体、没有扩展 `rk_execute_workspace`/`rk_run_attribution` 的 ABI，也没有
另写提交器。

## 0. 结论

**采用** `RK_PLAN_COMMAND_CACHE=1` 作为符合资格 single-submit workspace 的
默认；`RK_PLAN_COMMAND_CACHE=0` 保留原路径。`RK_PLAN_COMMAND_CACHE_SYNC=1`
是审计臂：只恢复每次 submit 的命令 BO prep/fini，不恢复重复打包。

- QK group b8：**1.0866×**（347.7→320.0 µs）
- PV group b8：**1.0923×**（333.4→305.1 µs）
- QKV b8：1.0247×（168.6→165.1 µs，绝对收益小）
- MLP b32：1.0000×（8 submits，不满足结构谓词，原路径，逐位一致）
- Tiny ViT b8 无插桩端到端：**1.0365×**（3450.1→3327.0 µs）
- Tiny ViT b1 无插桩端到端：**1.0276×**（1505.3→1464.7 µs）

创建成本（直接测得的第一次 command build）为 QK/PV **19.3/21.9 µs**，
QKV 2.3 µs；steady 收益在约 1–2 call 内摊平。Host 增量为每个 eligible
workspace **3,184 B**（`sizeof(co_command_plan)`），命令 BO 增量 **0 B**。

## 1. 当前成本拆分

在同一 fd、同一组创建时 `co_program` 上重复 build 64 次，中位数（µs）：

| case | clear | pack | verify | handles | command sync | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.50 | 15.17 | 19.25 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 10.79 | 15.75 | 21.87 |
| QKV b8 | 0.29 | 0.00 | 0.00 | 0.00 | 0.88 | 1.46 | 2.04 |

可消除部分是整个 per-submit build（QK/PV 约 15 µs，QKV 约 1.5 µs）。
其中同步占 QK/PV 约 10–11 µs；清零约 1.2 µs，打包约 2.6 µs，链校验和
句柄列表各不到 1 µs。端到端配对收益（约 29 µs）大于 attribution 中的
command 区间，说明命令 BO 的反复 prep/fini 还会造成额外的缓存/同步扰动；
用同一实现的三臂 A/B 直接测量，不把分项相加当作预测。

## 2. 资格与实现

首轮只覆盖：

1. `ws->program_cache_ready` 已为同一个 single submit 置位；
2. `plan->nsubmits == 1`；
3. submit 为 `CHAIN`，或 `CO_SEPARATE` 且恰有一个 task；
4. 全部 task 已通过 `rk_emit_action` 的创建时校验并驻留在
   `batch_scratch`。

多 submit、`CO_SEPARATE` 多 task、emitter 失败和所有其他 workspace 继续
走原执行器，不扩展 multi-submit 缓存。

实现复用现有 seam：

- `custom_ops.c` 中原 `execute_impl` 的清零、`rkt_chain_pack_at`、链封口、
  `rkt_chain_verify`、句柄列表和 flags 逻辑被抽成唯一路径
  `co_command_plan_build()`；原 `co_execute_profiled()` 仍调用它，只是多了
  一个可选的 `co_command_times` 归因参数；
- 新内部头 `src/co_execute_internal.h` 定义 `co_command_plan` 和
  `co_execute_cached_profiled()`。该头不进入 `include/`，公开
  `co_execute_workspace`、`co_execute_profile` 和 `rk_plan.h` 的结构体布局
  未改；
- `rk_workspace` 增加一个仅内部可见的 `co_command_plan *`，只在 eligible
  workspace 上 `calloc`，随 workspace 销毁；
- 运行时继续调用既有 `rocket_submit_tasks_flags()`、输出同步去重、故障
  poisoning 和 staged publication；cached 路径只跳过命令 BO 的重复准备。

创建时 `build_command_cache()` 在 `build_program_cache()` 之后调用；命令
构建失败只清掉可选缓存的 diag 并释放该 `co_command_plan`，workspace 创建
继续走原 per-run 路径。只有后续 workspace 创建本身失败（BO 分配、常量上传
等）才进入统一 fail/cleanup；这与“缓存构建失败直接清理并终止创建”不同。

## 3. command BO 契约

**设备只读。** 命令 BO 是 PC 取指的 regcmd/chain 流，设备不会写回；唯一的
写者是创建时的 CPU：`memset`、regcmd 拷贝、`rkt_chain_pack_at` 写入链尾、
`rkt_chain_verify` 读取校验，最后一次 `rocket_bo_fini` 把 CPU 写入刷给
设备。创建完成后 CPU 不再修改该 BO。

**缓存不跨边界。** `co_command_plan` 记录所属 fd、program 数组地址、
command BO handle、task descriptors、flags 和句柄集合。cached entry 要求
fd、workspace、command BO handle、mode、task 数和 program 数组地址全部
一致；workspace 销毁时 plan 释放。没有跨 workspace、fd、IOMMU domain 或
BO 重分配的复用。

**同步单独论证，不顺手删除。** `rocket_bo_prep/fini` 是 CPU cache
maintenance：fini 刷脏行给设备，prep 在 CPU 读之前把设备写回。命令 BO
在创建后既不被 CPU 写、也不被 CPU 读，因此重复 prep+fini 没有需要维护的
方向。control 臂保留它们，`RK_PLAN_COMMAND_CACHE_SYNC=1` 可以单独恢复，
证明收益确实来自这一决定而不是打包。

板端 `bcmdverify` 在真实设备上连续执行 512（QK/PV）/1024（QKV）次后，
显式 `rocket_bo_prep`（FROM_DEVICE）再与创建时快照逐字节比较：
32,256 / 32,256 / 1,008 bytes，全部 unchanged。

## 4. 同实现对照

`command_cache_test`（254 checks）使用真实打包代码、只 stub 掉 submit
ioctl：

- 同 workspace 二次 build：命令字节、task 顺序、链尾、flags、输入/输出
  句柄集合逐项一致；
- 独立 workspace：task 相对偏移和句柄集合一致，绝对 regcmd 地址指向各自
  的 command BO，`rkt_chain_verify` 通过；
- 200 次 cached submit 后命令字节、task 数组和句柄记录不变；
- 同步臂与 no-sync 臂的 prep/fini 计数差恰好为每次 submit 一对；
- submit 失败时返回错误、命令字节不变、不执行输出同步，下一次 submit 正常；
- 错误 fd、错误 mode、错误 program 数组、另一 workspace 或重分配的 command
  BO 均被拒绝；
- 构建失败不泄漏私有 command BO。

生命周期集成由默认开启的 `rk_plan_host`（33/33 CTest）和 Python
`test_ws_poison` / failure-atomicity 覆盖：创建失败、submit 失败、guard
失败、non-finite 输入均不发布半成品；缓存 arm 与原路径逐位一致。

## 5. 隔离 A/B

同一库、同一 workspace 形状、同一输出同步、7 个 rotated rounds；主估计为
逐轮 median 的 median/ratio，p10 辅助。

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.7 | 336.3 | 320.0 | **1.0866×** | 1.0846× | 1.0846–1.0952 | yes |
| PV group b8 | 333.4 | 322.6 | 305.1 | **1.0923×** | 1.0917× | 1.0917–1.0988 | yes |
| QKV b8 | 168.6 | 166.2 | 165.1 | **1.0247×** | 0.8374× | 0.8374–1.0603 | yes |
| MLP b32 | 593.0 | 595.3 | 592.4 | 1.0000× | 0.9971× | 0.9971–1.0039 | yes |

QK/PV 的 p10 与中位数同向；QKV 的 p10 受一次进程级离群轮影响，但中位数
仍为正。MLP b32 不满足谓词，只作为“原路径逐位中性”的证据，不纳入收益估计。cache+sync 相对 cache 的增益为 QK 1.0510×、PV 1.0574×、QKV 1.0071×，
这就是单独保留同步的成本。

## 6. 创建成本、内存与 warm 收益

- 直接 first-build：QK 19.25 µs、PV 21.87 µs、QKV 2.04 µs。
- 进程级 `SETUP_US` 配对差：QK +39.1 µs、PV +47.0 µs、QKV −8.5 µs、
  MLP −31.2 µs（ineligible；这些进程级数字正负相抵，说明其噪声大于增量）；该数字包含进程噪声，仅作边界。
- 内存：`rk_workspace_host_bytes(on) - rk_workspace_host_bytes(off)` =
  **3,184 B/eligible workspace**；command BO 复用既有
  `co_execute_workspace.commands`，增量为 **0 B**。
- Tiny ViT b8：3450.1→3327.0 µs，**1.0365×**，p10 1.0352×。
- Tiny ViT b1：1505.3→1464.7 µs，**1.0276×**，p10 1.0250×。
- 两个 full-model 的所有 arm logits 逐位一致。

## 7. 验证与决策

- CTest **33/33**，新增 `command_cache` 254 checks、0 failures；
- pytest **351 passed, 71 subtests passed**；
- 定向 failure atomicity 1 passed + `test_ws_poison` 4 passed；
- 冻结数值复验：**PASS / 0 problems**；
- 默认 on 与显式 `RK_PLAN_COMMAND_CACHE=0` 输出逐位一致。

**决策：采用**。收益来自消除重复 CPU 打包（clear/pack/verify/handle）和
证明可省的命令 BO sync；输出同步、发布、故障隔离、数值政策均未改变。
`RK_PLAN_COMMAND_CACHE=0` 可回退，`RK_PLAN_COMMAND_CACHE_SYNC=1` 可单独
恢复旧同步行为。

## 8. 身份与证据

- 最终板端 runtime：`artifacts/librknpu_plan.so` SHA-256
  `581a898b94300be8ef3eb367137893fc6113cca402f415e79dd0ad076e6afa86`
- probe：`artifacts/probe-final` SHA-256 `4ebb413d6d7ea8408b694737f78f700742d2c52c4146cdf857f49ecf0a94cba3`
- compiled package：`16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec：`2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
- 结果：`results/20261028T023818Z-command-cache/`
