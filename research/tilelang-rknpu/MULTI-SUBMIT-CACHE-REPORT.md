# 受预算控制的多提交静态执行准备

日期：2026-10-28。本包把已有的 single-submit `co_program`/命令缓存扩展为按
submit 规划的多提交准备机制，但控制臂仍保持原 single-submit 策略；只比较
新增的多提交能力。

## 0. 结论

**采用** `RK_PLAN_MULTI_SUBMIT_CACHE=1` 作为默认，`=0` 为 control。
多提交准备只命中结构上合法的 submit；MLP 的 8 个 submit 全部命中。

- MLP b32 isolated：**1.0918×**（625.3→572.2 µs）
- MLP b1 isolated：**1.1341×**（446.5→393.5 µs）
- 五 profile 交错全模型（256 images，7 rounds）：
  - MLP b32 **1.0511×**（1059.2→1006.5 µs）
  - MLP b8 **1.0781×**（701.0→649.2 µs）
  - MLP b1 **1.0954×**（601.1→549.2 µs）
  - ViT b8 1.0030×、ViT b1 1.0021×（默认图没有多提交 workspace，中性）
- 创建成本：MLP b32 +200.7 µs、b1 +177.6 µs（同 A/B 的 `SETUP_US` 中位数）。
- 内存：每个 MLP workspace Host **208,528 B** + device command BO
  **11,088 B**；预算默认 4 MiB/workspace，超预算按 submit 前缀缓存，缓存
  构建失败/不足安全回退。
- 命令区域：8 个 submit 各自独立 command BO，handle 唯一；1,024 次
  cached execution 后 8 个 BO 字节全部 unchanged。
- 输出同步、guard、下载、发布、数值政策、submit 顺序和核策略不变。

## 1. 投资价值确认

当前 control 臂的 MLP 成本（µs，battr 中位数）：

| case | emit | command prep | total |
|---|---:|---:|---:|
| MLP b32 | 15.5 | 35.0 | 625.0 |
| MLP b1 | 15.9 | 34.7 | 447.7 |

另有实现前的独立快照 `baseline-cost/`，emit 15–16 µs、command 31–35 µs，
与 control 臂一致。可消除部分约为每次调用 50 µs（emit + command prep，
其中 command sync 约 8–9 µs 可通过只读 BO 论证一并省去）。因此预计收益
足以支撑受预算控制的实现。

## 2. 实现与资格

- `rk_workspace` 保存 `rk_submit_cache[]`；每项记录 submit id、task 数、
  mode、program 切片和 `co_command_plan`。
- 程序切片来自一个受预算限制的连续 `co_program` 数组；默认预算
  `RK_PLAN_MULTI_SUBMIT_BUDGET=4 MiB`，同时计入 Host 程序和 entry 块以及
  device command BO 字节。
- 资格是结构性的：`nsubmits > 1`；submit 有非零 task；全部为
  `RK_ACTION_NPU_TASK` 且 `rk_emit_action` 创建时成功；mode 为 `CHAIN`
  或单 task `CO_SEPARATE`。多 task `CO_SEPARATE` 和其他失败项继续使用原
  per-run 路径，没有模型名特判。
- 按 plan order 逐 submit 准备；预算不足时跳过放不下的 submit，继续尝试
  后续更小的 submit，因此支持“缓存前缀/部分缓存”。

## 3. command 区域独立性

每个成功 submit 由
`co_command_plan_build(fd, slice, tasks, mode, workspace=NULL, ...)`
分配一个 exact `words*8` 的独立 `rocket_bo`。没有修改 BO 基址、没有用
指针偏移伪装切片；`rkt_chain_pack_at` 使用该 BO 自己的 DMA base 写入链地址，
`rkt_chain_verify` 在同一对象内校验链尾。

默认开启的 MLP b32/b1 证据：

- `cached submits`：8/8；
- `device_bytes`：11,088；
- 8 个 submit 的 command handle：21..28，全部唯一；
- 1,024 次 cached execution 后，8 个 BO 经显式 FROM_DEVICE sync 与创建快照
  逐字节比较全部 unchanged；
- `command_cache_test` 独立 build 两个 owned plan，提交第二个后第一个字节
  不变。

## 4. 原执行语义保持

- `run_repeat` 仍按原 plan action 顺序处理；cached submit 只跳过创建时已
  完成的 `rk_emit_action` 和命令打包，submit 仍走既有
  `rocket_submit_tasks_flags`。
- 输出同步仍按原去重集合执行；guard、download、publish、host 边界和核策略
  不变。
- 多提交缓存命中时，`command_us` 只剩 `RK_PLAN_COMMAND_CACHE_SYNC=1`
  审计臂的 PREP/FINI；默认 no-sync 只依据已经证明的只读 command BO 契约。
- 缓存构建失败、内存不足或预算不足不会使 workspace 创建失败；部分创建在
  drop 路径释放全部 owned command BO、程序数组和 entry 数组。
- 提交失败仍按原路径 poison；poisoned workspace 的 destroy 继续拒绝释放
  可能被设备触碰的 BO，达到进程级恢复边界。

## 5. 隔离 A/B

7 rotated rounds；逐调用样本保留在 `pair/r*/<case>/<arm>.log` 的
`WARM_US` 行。

| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP b32 | 625.3 | 580.1 | 572.2 | **1.0918×** | 1.0886× | 1.0886–1.0970 | yes |
| MLP b1 | 446.5 | 401.9 | 393.5 | **1.1341×** | 1.1328× | 1.1328–1.1397 | yes |

归因（µs 中位数）显示 cache 臂 emit=0，command 从约 35 µs 降到 0.3–0.4 µs：

| case | arm | emit | command | output sync | total |
|---|---|---:|---:|---:|---:|
| MLP b32 | control | 15.5 | 35.0 | 29.9 | 625.0 |
| MLP b32 | cache+sync | 0.0 | 8.5 | 30.0 | 579.2 |
| MLP b32 | cache | 0.0 | 0.4 | 30.0 | 572.5 |
| MLP b1 | control | 15.9 | 34.7 | 25.5 | 447.7 |
| MLP b1 | cache+sync | 0.0 | 8.5 | 25.4 | 402.6 |
| MLP b1 | cache | 0.0 | 0.3 | 25.7 | 393.7 |

## 6. 创建成本与内存

同 A/B 的 `SETUP_US` 中位数：

| case | control µs | cache µs | delta |
|---|---:|---:|---:|
| MLP b32 | 3925.8 | 4126.5 | +200.7 |
| MLP b1 | 3663.3 | 3840.9 | +177.6 |

`bmulti` 直接测量（默认 on）：

| case | cached | device bytes | Host delta | unique BO | bytes unchanged |
|---|---:|---:|---:|---|---|
| MLP b32 | 8/8 | 11,088 | 208,528 B | yes | yes (8 BOs / 1024 runs) |
| MLP b1 | 8/8 | 11,088 | 208,528 B | yes | yes (8 BOs / 1024 runs) |

三个 MLP profile 的 plan 都是 8 submits / 11 tasks，缓存 footprint 相同。
4 MiB 是每 workspace 的硬上限；低预算测试只缓存 1/2 submit，未缓存 submit
仍逐次准备且输出逐位一致。

## 7. 五 profile 完整模型

同进程、同 fd、同 compiled graph/weights，两个 runner 逐 call 交替顺序，
256 images、7 rounds；`stats=None`，无节点代理。逐调用样本在
`full-model/result.json` 的 `samples`。

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b32 | 1059.2 | 1006.5 | **1.0511×** | 1.0303× | 30233.0 | 31612.6 | yes |
| mlp_b8 | 701.0 | 649.2 | **1.0781×** | 1.0397× | 10292.1 | 10974.2 | yes |
| mlp_b1 | 601.1 | 549.2 | **1.0954×** | 1.0472× | 1571.3 | 1703.8 | yes |
| vit_b8 | 3375.3 | 3362.4 | 1.0030× | 1.0014× | 2372.6 | 2381.5 | yes |
| vit_b1 | 1442.4 | 1438.9 | 1.0021× | 1.0010× | 707.2 | 710.2 | yes |

五个 profile 准确率相同、logits 逐位一致。ViT 默认图没有多提交 workspace，
其中性结果符合预期，也说明实现没有按模型名特判。

## 8. 回归与决策

- `command_cache_test`：**270 checks / 0 failures**；
- `rk_plan_host_test`：多提交缓存、低预算部分缓存和原路径输出均通过；
- CTest：**33/33**；
- pytest：**352 passed, 71 subtests passed**；
- 冻结数值复验：**PASS / 0 problems**；
- 默认 on 的 8 个独立 command BO 在 1,024 次执行后字节 unchanged。

**决策：采用默认 on（4 MiB 预算）**。没有扩展硬件配方、数值政策或异步
执行；single-submit 和 command-cache 的旧结果原样保留。

## 9. 身份与证据

- 最终 board runtime：`artifacts/librknpu_plan.so` SHA-256
  `ddf0fb836bf1e58eb653ecc9560c8fabe3473bccb6ca8eeb3888aedf0739a73f`
- probe：`artifacts/probe-multi-final` SHA-256
  `63568db3fa01ec3c4c86d8d2d52445472ed1a3136a0a0ac9c4e27a614b606c0a`
- compiled package：`16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec：`2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
- 结果：`results/20261028T035849Z-multi-submit-cache/`
