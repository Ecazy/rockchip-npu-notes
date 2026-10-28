# 创建时 command BO 打包复用（含验证与计时修订）

日期：2026-10-28。本阶段只验证“命令准备”这一项：对
`program_cache_ready` 的 single-submit workspace，在创建时把命令 BO 内容、
task descriptors、链尾、flags 和句柄列表准备好，后续 submit 复用；原
per-submit 打包路径保留为 control 和 fallback。

程序缓存、上传、guard、输出同步、下载、发布和数值政策不变。没有新增公共
结构体、没有扩展 `co_execute_workspace`/`rk_run_attribution` 的 ABI，也没有
另写提交器。

> **修订记录（2026-10-28）**：原结果
> `results/20261028T023818Z-command-cache/` 保留为历史证据，不覆盖、不重算。
> 本轮修正了 `CO_SEPARATE` 的提交前全量校验和命令准备计时口径；修正后的
> 短 A/B 与新证据在
> `results/20261028T030858Z-command-cache-fix/`。

## 0. 结论

**采用** `RK_PLAN_COMMAND_CACHE=1` 作为符合资格 single-submit workspace 的
默认；`RK_PLAN_COMMAND_CACHE=0` 保留原路径。`RK_PLAN_COMMAND_CACHE_SYNC=1`
是审计臂：只恢复每次 submit 的命令 BO prep/fini，不恢复重复打包。

修正后 short A/B（5 rotated rounds）：

- QK group b8：**1.0858×**（347.4→320.2 µs）
- PV group b8：**1.0937×**（334.2→305.7 µs）
- QKV b8：1.0127×（162.7→160.4 µs，绝对收益小）
- MLP b32：0.9995×（8 submits，不满足结构谓词，原路径，逐位一致）
- Tiny ViT b8 短端到端：**1.0392×**（3536.7→3409.3 µs）
- Tiny ViT b1 短端到端：**1.0268×**（1489.5→1453.1 µs）

原完整 campaign 的 uninstrumented 结果（Tiny ViT b8 1.0365×、b1 1.0276×）
保留为历史；修正后短 A/B 方向一致。创建成本约 QK/PV 24–32 µs（对第一次
build 更敏感），Host 增量每个 eligible workspace **3,184 B**，命令 BO 增量
**0 B**。

## 1. 本轮两处修正

### 1.1 恢复 `CO_SEPARATE` 提交前全量校验

重构后 `execute_impl` 曾先进入 `CO_SEPARATE` 递归，完整校验落在每个子调用
里；若第一个 program 合法、第二个非法，会先提交第一个再返回错误。现在：

- 提取 `validate_program_set()`，对 fd、workspace、mode、硬件门禁和整个
  program 数组做与重构前相同的校验；
- `execute_impl` 在 `CO_SEPARATE` 扇出之前调用它；只有全数组合法才开始逐项
  提交；
- `co_command_plan_build()` 也复用同一校验函数。

回归在 `command_cache_test` 中：第一个 program 合法、第二个 `count=0` 的
separate 调用断言 **0 次 submit**；合法的两 program separate 路径仍为
**2 次 submit** 且 flags 不含 `ROCKET_JOB_BATCHED`。

### 1.2 修正命令准备计时口径

- `times.sync_us` 现在同时包含 `rocket_bo_prep` 和 `rocket_bo_fini`；
- `times.total_us` 是一个连续的“PREP 前到句柄列表后”的外层区间，不再由
  不完整分项相加；
- `execute_impl` 的公开 `command_us` 直接使用 `times.total_us`，恢复重构前
  的覆盖范围；构建失败时不再把部分 command 区间写入归因；
- 细分时间戳只在独立探针 `bcmd` 中通过 `times.detail=1` 打开；生产 control
  路径只保留原有的 allocation 和 command 两个外层时间戳，不增加插桩。

因此，原报告里“端到端配对收益比 command 区间大约 29 µs，说明缓存/同步
扰动”的解释**撤回**。分项只用于独立归因；在遗漏区间未排除前，不把端到端
与 `command_us` 的差额归因于缓存或同步扰动。

## 2. 修正后的成本拆分

同一 fd、同一创建时 `co_program`，独立探针重复 build 64 次的中位数（µs）：

| case | clear | pack | verify | handles | sync (PREP+FINI) | total | first build |
|---|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 19.25 | 24.21 | 30.04 |
| PV group b8 | 1.17 | 2.62 | 0.29 | 0.58 | 20.12 | 25.37 | 32.67 |
| QKV b8 | 0.00 | 0.29 | 0.00 | 0.00 | 1.75 | 2.62 | 3.79 |

`total` 是连续外层区间；clear/pack/verify/handles/sync 是独立探针分项，
可能因探针边界和采样不与 total 严格相加。QK/PV 的重复准备约 24–25 µs，
其中 PREP+FINI 约 19–20 µs；QKV 约 2.6 µs。

## 3. 资格与实现

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
  `co_command_plan_build()`；原 `co_execute_profiled()` 仍调用它；
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
继续走原 per-run 路径。若可选的 3,184 B 会使 Host budget 超限，创建也会
丢弃 command cache 并继续原路径，而不是让可选 cache 变成创建失败。只有
后续 workspace 创建本身失败（BO 分配、常量上传等）才进入统一 fail/cleanup；
这与“缓存构建失败直接清理并终止创建”不同。

## 4. command BO 契约

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

## 5. 同实现对照

`command_cache_test`（259 checks）使用真实打包代码、只 stub 掉 submit
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
- **separate 全量校验回归：后续 program 非法时 0 次 submit；合法 separate
  路径 2 次 submit；**
- 构建失败不泄漏私有 command BO。

生命周期集成由默认开启的 `rk_plan_host`（33/33 CTest）和 Python
`test_ws_poison` / failure-atomicity 覆盖：创建失败、submit 失败、guard
失败、non-finite 输入均不发布半成品；缓存 arm 与原路径逐位一致。

## 6. 修正后的短 A/B

同一库、同一 workspace 形状、同一输出同步、7 个 rotated rounds；每次调用
的原始样本保留在 `pair/r*/<case>/<arm>.log` 的 `WARM_US` 行。

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| QK group b8 | 347.4 | 336.6 | 320.2 | **1.0858×** | 1.0792× | 1.0792–1.0973 | yes |
| PV group b8 | 334.2 | 325.8 | 305.7 | **1.0937×** | 0.9727× | 0.9727–1.0982 | yes |
| QKV b8 | 162.7 | 163.0 | 160.4 | **1.0127×** | 1.0035× | 1.0035–1.0401 | yes |
| MLP b32 | 590.3 | 591.5 | 591.5 | 0.9995× | 0.9946× | 0.9946–1.0034 | yes |

PV 保留一个离群轮；QK/PV 中位数仍为约 8.6–9.4% 收益。MLP b32 不满足谓词，
只作为原路径逐位中性证据。cache+sync 相对 cache 的增益为 QK 1.0510×、
PV 1.0658×、QKV 1.0164×；这是重复命令同步的成本。

短 full-model 复验（64 images、3 rounds）：

| model | control | cache+sync | cache | control/cache | p10 | bitwise |
|---|---:|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3536.7 | 3473.6 | 3409.3 | **1.0392×** | 1.0389× | yes |
| Tiny ViT b1 | 1489.5 | 1474.5 | 1453.1 | **1.0268×** | 1.0248× | yes |

原完整 campaign（256 images、7 rounds）的 1.0365× / 1.0276× 作为历史证据
保留；修正后短复验方向一致。

## 7. 创建成本、内存与 warm 收益

- 直接 first-build：QK 30.04 µs、PV 32.67 µs、QKV 3.79 µs；
  build 中位数 total 为 24.21 / 25.37 / 2.62 µs。
- 进程级 `SETUP_US` 配对差：QK +30.0 µs、PV +35.3 µs、QKV −5.8 µs、
  MLP −19.3 µs（ineligible）；该数字包含进程噪声，仅作边界。
- 内存：`rk_workspace_host_bytes(on) - rk_workspace_host_bytes(off)` =
  **3,184 B/eligible workspace**；command BO 复用既有
  `co_execute_workspace.commands`，增量为 **0 B**。
- Short full-model 收益见上表；所有 arm logits 逐位一致。

## 8. 验证与决策

- CTest **33/33**；
- `command_cache_test` **259 checks / 0 failures**，含 separate 全量校验回归；
- pytest **351 passed, 71 subtests passed**；
- 定向 failure atomicity 1 passed；`test_ws_poison` 4 passed；
- 冻结数值复验：**PASS / 0 problems**；
- 默认 on 与显式 `RK_PLAN_COMMAND_CACHE=0` 输出逐位一致。

**决策：继续采用**。收益来自消除重复 CPU 打包（clear/pack/verify/handle）
和已单独论证可省的命令 BO sync；输出同步、发布、故障隔离、数值政策均未
改变。`RK_PLAN_COMMAND_CACHE=0` 可回退，
`RK_PLAN_COMMAND_CACHE_SYNC=1` 可单独恢复旧同步行为。

## 9. 身份与历史

- 修正后板端 runtime：`artifacts/librknpu_plan.so` SHA-256
  `dfcba5bf112ca855ee37322bc5511f322d9853c612afd3f10bd5419ddfca046c`
- probe：`artifacts/probe-fix` SHA-256
  `de85183252d3449745dbafe272af382ae9dec14a6b80b403c0e0e6812d0b0cda`
- compiled package：`16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec：`2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
- 修正后结果：`results/20261028T030858Z-command-cache-fix/`
- 历史结果（不覆盖、不重算）：`results/20261028T023818Z-command-cache/`
