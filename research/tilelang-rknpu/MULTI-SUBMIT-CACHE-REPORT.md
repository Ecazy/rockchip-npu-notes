# 预算契约贯通与多提交缓存采用结论

日期：2026-10-28。本包完成 Plan→descriptor 的预算契约贯通，并在显式预算
余量下重新评估多提交静态准备。最终结论：**采用**
`RK_PLAN_MULTI_SUBMIT_CACHE=1` 为默认；旧计划缺少 `budget_policy` 时保持
strict 语义并安全回退，新计划只有在显式声明的 Host/device 余量内才会缓存。

此前两个结果保持历史原样，不作为产品收益声明：

- `results/20261028T035849Z-multi-submit-cache/`：未把 cache command BO 计入
  device budget，收益声明已撤回；
- `results/20261028T055700Z-multi-submit-budget-fix/`：保守严格回退版本，
  device budget 被 base 资源用满，收益中性。

新收益只引用本包 budgeted plan package 的合法配置实测。

## 1. 预算契约

`CompileOptions.device_budget_bytes` / `host_budget_bytes` 现在写入计划的
版本化 `budget_policy`：

```json
"budget_policy": {
  "schema_version": 1,
  "device_budget_bytes": 268435456,
  "host_budget_bytes": 268435456
}
```

- 该块进入 canonical plan hash，序列化/加载往返保留；
- `build_plan_descriptor()` 把它投影到既有 C 字段
  `device_budget_bytes` / `host_budget_bytes`；
- `claimed_device_bytes` / `claimed_host_bytes` 继续表示编译器 accounting
  估算；
- 没有新增或重排公共 C ABI 字段；
- 旧 JSON 缺少 `budget_policy` 时 `budget=None`，descriptor 的 budget 仍等于
  accounting 估算，保持原先严格限制，不自动补成 256 MiB 或扩大权限；
- `budget_policy.schema_version` 独立版本化；不认识的版本直接拒绝。

`build_cases.py` 的 probe bundle 也同步区分 budget 与 claimed，避免测试
工具再次把两者混为一谈。

## 2. 分配前检查

`build_multi_submit_cache()` 在每次持久分配前检查三类预算：

1. **缓存自身上限**：Host entries/programs + command BO 页对齐物理上界 ≤
   `RK_PLAN_MULTI_SUBMIT_BUDGET`（默认 4 MiB/workspace）；
2. **workspace Host 余量**：`plan Host budget - rk_workspace_host_bytes(ws,1)`
   必须容纳 entry 和 program slice；
3. **device 预算余量**：`plan device budget - 数据槽 request - fallback
   command workspace request` 必须容纳新 command request。

fallback command workspace 始终存在并计入，不能视为被替换。`entries` 在
Host 余量不足时不会先分配；每个 submit 先把 program 发射到现有
`batch_scratch` 作为成本探针，三项预算都通过后才分配持久 program slice、
重新发射到该 slice，并分配独立 command BO。因此临时和持久分配都不突破
声明边界。

`RK_PLAN_MULTI_SUBMIT_BUDGET` 解析 fail closed：未设置/空串使用默认值；
负数、带符号、空白/单词、尾随垃圾、溢出和显式 0 都返回 0（禁用 cache），
不会静默变成默认值或无界值。

## 3. 设备字节口径

- plan 的 `device_budget_bytes` 按**request bytes** 校验，与已声明
  accounting 口径一致；
- cache 自身上限按**每个 command BO 的页对齐 physical upper bound** 计算，
  因为 provider 对每个 BO 做 4 KiB 对齐；
- fallback command workspace 的 request/physical 均单独报告并计入 base
  device usage。

MLP b32 的实测：

| variant | cache state | cached | skipped budget/static | Host delta | cache request | cache physical | fallback command request/physical |
|---|---|---:|---|---:|---:|---:|---|
| exact | fallback | 0/8 | 8/0 | 0 B | 0 B | 0 B | 65,536/65,536 B |
| partial | prepared | 1/8 | 7/0 | 92,160 B | 4,032 B | 4,096 B | 65,536/65,536 B |
| full | prepared | 8/8 | 0/0 | 208,528 B | 11,088 B | 32,768 B | 65,536/65,536 B |

exact 预算恰好用完数据槽 + fallback command，零 cache BO；partial 只剩
4,032 request，第一个 4-task submit 命中后其余 7 个因 device 余量跳过；
full 使用 `CompileOptions` 文档默认 256 MiB 上限，8 个 submit 全缓存，未为
恢复收益反向提高预算。三者输出逐位一致。

## 4. 合法余量下的性能

预算化 MLP plan（b1/b8/b32，256 MiB 显式上限）加未改 ViT plan；同进程、
同 fd、逐 call 交替，256 images、7 rounds，逐调用样本在
`full-model/result.json`。

| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp_b32 | 1073.8 | 1037.3 | **1.0538×** | 1.0247× | 28172.9 | 28377.8 | yes |
| mlp_b8 | 1005.9 | 969.0 | **1.0386×** | 1.0357× | 7974.1 | 8268.5 | yes |
| mlp_b1 | 898.3 | 859.2 | **1.0459×** | 1.0449× | 1227.4 | 1298.7 | yes |
| vit_b8 | 3391.6 | 3395.7 | 0.9999× | 0.9975× | 2373.4 | 2371.5 | yes |
| vit_b1 | 1477.1 | 1470.0 | 1.0059× | 1.0053× | 694.7 | 698.9 | yes |

五个 profile 准确率相同、logits 逐位一致。MLP 收益 1.034–1.054×；ViT
默认图没有多提交 workspace，保持中性。

## 5. 验收

- `plan-budget-contract`：4 tests，覆盖 compile/persist/hash/load/hybrid
  manifest/C descriptor separation、旧计划 strict 语义、非法 budget 拒绝；
- `rk_plan_host_test`：**875 checks / 0 failures**，含 device 恰满零分配、
  Host 余量 partial、cache 余量 partial、非法配置、mock 分配次数和输出逐位；
- `command_cache_test`：**270 checks / 0 failures**；
- CTest：**33/33 passed**；
- pytest：**356 passed, 71 subtests passed**；
- budgeted plan package 的冻结数值复验：**PASS / 0 problems**；派生 spec 只
  更新 compiled-manifest identity，阈值和 expected arrays 未变；
- exact/partial/full 三种预算的 MLP output SHA-256 完全一致。

## 6. 历史与身份

- 最终 board runtime：`artifacts/librknpu_plan.so` SHA-256
  `b94454bdbfd15f6fd3f6af29089e81f1a1e902d5f448fef19cbf1f4fb232f0e9`
- probe：`artifacts/probe-contract-final` SHA-256
  `f443e5e4887b12f167f51f399496e54f6966fa16cf461cd11f71c1bc87189b65`
- 结果：`results/20261028T063532Z-budget-contract/`
- 历史结果（未覆盖、未重算）：
  - `results/20261028T035849Z-multi-submit-cache/`
  - `results/20261028T055700Z-multi-submit-budget-fix/`
