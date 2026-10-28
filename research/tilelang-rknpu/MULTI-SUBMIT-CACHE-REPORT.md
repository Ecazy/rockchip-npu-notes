# 受预算控制的多提交静态执行准备（预算闭环修订）

日期：2026-10-28。本包原目标是把 single-submit 程序/命令缓存扩展为按 submit
规划的多提交准备。独立 command BO、部分缓存和正收益方向已实现并有回归；
但预算闭环修订发现：当前 compiled MLP plan 的 device budget 已被数据槽和
原 fallback command workspace 恰好用满，额外 cache command BO 没有合法
设备余量。因此本包**不采用为默认**，只保留 opt-in 实验实现。

历史结果 `results/20261028T035849Z-multi-submit-cache/` 保持原样；其中
1.0918×/1.1341× 等性能数字是在未把新增 command BO 计入 device budget 的
实现上测得，现已撤回，不能作为产品收益声明。

## 0. 最终结论

- 默认：`RK_PLAN_MULTI_SUBMIT_CACHE=0`，保持原 single-submit 策略。
- opt-in：`RK_PLAN_MULTI_SUBMIT_CACHE=1` 仅对显式声明了 device 余量的
  workspace 生效；当前 compiled MLP 计划全部安全回退。
- 预算：`RK_PLAN_MULTI_SUBMIT_BUDGET` 默认 4 MiB/workspace；非法、负数、
  溢出和显式 0 都 fail closed 为 0（禁用 cache），不会静默放开限制。
- 单 submit 的 command BO cache 和此前所有结果不变。

## 1. 分配前同时约束三类预算

`build_multi_submit_cache()` 在每次分配前检查：

1. **缓存自身上限**：Host entry/program 字节 + command BO 的页对齐物理上界
   ≤ `RK_PLAN_MULTI_SUBMIT_BUDGET`；
2. **workspace Host 余量**：
   `plan Host budget - rk_workspace_host_bytes(ws,1)` 仍能容纳该 submit 的
   program slice 和 entry；
3. **设备预算余量**：
   `plan device budget - 当前数据槽 request - 原 fallback command workspace
   request` 仍能容纳该 submit 的 command request。

原 fallback command workspace 没有被替换或忽略：MLP 中它的 request/physical
均为 **65,536 B**，始终计入 base device usage。若数据槽 + fallback 已用满
计划 device budget，则 command request 余量为 0，任何 cache BO 都不会分配。

不满足任一预算时跳过该 submit；部分缓存按 plan order 保留前缀，全部不满足
则回退原 per-run 路径，workspace 仍可运行。

## 2. 设备字节口径

- **request bytes**：计划声明和 `plan->device_budget_bytes` 校验使用的口径。
- **physical upper bound**：provider 对每个 BO 做 4,096 B 页对齐，因此严格
  物理上限是每个 BO `page_up(size)` 之和。
- 独立 command BO 的页对齐开销按 BO 逐个计算，不能把多个小 BO 的 request
  相加后只 page-up 一次。
- 历史 MLP 的 11,088 B 只是 cache command **request**；若按当前独立 BO
  分配，物理上界是 **32,768 B**（8 个 BO × 4 KiB）。本轮修正后该 cache
  因 device budget 无余量而不分配，实际增量 request/physical 都是 **0 B**。

## 3. 预算解析加固

`env_bytes()` 现在的行为：

- 未设置或空串：使用默认值；
- 首字符不是数字（包括负数、`+`、空白、单词）：返回 0；
- `strtoull` 溢出（`ERANGE`）或尾随非法字符：返回 0；
- 显式 `0`：返回 0，即禁用可选 cache；
- 不会把负数/溢出解释成极大预算，也不会在解析失败时退回默认值。

## 4. 针对性回归

`rk_plan_host_test` 新增一个 4-submit synthetic plan，直接断言 mock 的 BO
分配次数，防止“先超额分配再释放”：

- device budget 恰好等于基础数据槽 + fallback command：**0 个 cache BO**，
  分配次数仅 base；
- Host 余量不足：只缓存 **3/4** submit，分配次数 base+3，
  `rk_workspace_host_bytes(ws,1) ≤ host_budget`，输出逐位一致；
- cache 自身上限只够一个 submit：缓存 **1/4**，分配次数 base+1，输出逐位一致；
- `-1`、非数字、`0`、超长溢出值：全部禁用 cache，分配次数仅 base，输出逐位一致。

`command_cache_test` 另覆盖独立 command BO 不互相覆盖、合法/非法
`CO_SEPARATE`、后续 program 非法零提交、submit 失败和构建失败清理。

## 5. 修正后的短性能确认

当前 compiled plan 下，opt-in 与 control 都走原 per-submit 路径。同进程、
同 fd、逐 call 交替的五 profile 交错结果（256 images、3 rounds）：

| profile | control µs | opt-in µs | on/off | bitwise |
|---|---:|---:|---:|---|
| mlp_b32 | 1355.9 | 1353.3 | 1.0016× | yes |
| mlp_b8 | 1001.9 | 1000.1 | 1.0022× | yes |
| mlp_b1 | 902.1 | 900.1 | 1.0029× | yes |
| vit_b8 | 3112.5 | 3093.1 | 1.0055× | yes |
| vit_b1 | 1198.4 | 1179.9 | 1.0154× | yes |

这是**安全回退的中性确认**，不是收益声明。`bmulti` 同时显示 MLP：
`ready=0, cached=0/8, host_delta=0, cache request/physical=0/0`，fallback
command request/physical=65,536/65,536 B。

## 6. 验证

- `command_cache_test`：**270 checks / 0 failures**
- `rk_plan_host_test`：**875 checks / 0 failures**（含上述预算边界）
- CTest：**33/33 passed**
- pytest：**352 passed, 71 subtests passed**
- 冻结数值复验：**PASS / 0 problems**
- 未重跑未变化的外部基线；未改硬件配方、数值政策或异步执行

## 7. 决策

**不采用默认开启。** 独立 command BO 与部分缓存实现保留为 opt-in
实验；只有未来 plan 明确声明了 device budget 余量时才可能生效。当前
compiled MLP 计划 device budget 已满，因此保持原执行路径和中性结果。

## 8. 身份与证据

- 修正后 board runtime：`artifacts/librknpu_plan.so` SHA-256
  `b5741177637882433d363d629a284ea93373ea3c4e0919a2aafa1e231a7aec49`
- probe：`artifacts/probe-budget-fix` SHA-256
  `9bae73e14e5d63bfebcb558ec79160acf21ef6b72c32708e85d6f2f6a6d9afa3`
- 修正后结果：`results/20261028T055700Z-multi-submit-budget-fix/`
  （以实际生成的目录名为准）
- 历史结果（未覆盖）：`results/20261028T035849Z-multi-submit-cache/`
