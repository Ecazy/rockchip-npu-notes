# 第一个优化包：桥接包络归属 + 静态 repeat 成组（实施报告 v5）

日期：2026-10-26（v5 同日修订，回应第四轮只读复核）。对应
[TRANSFORMER-OPTIMIZATION-PLAN.md](TRANSFORMER-OPTIMIZATION-PLAN.md) 第 10 节“首个优化包”。

范围严格限定在方案稿第 10 节的首包三项——解释 QK/PV 运行包络、静态 repeat 批量入口、
有边界的成组验证——以及由其成本排名直接导出的 Host 侧瘦身。不加 Softmax/RMSNorm 新算法、
不做量化、不改模型、不改权重、不改数值政策、不做多核并发、不新增 chain 配方。

**四轮复核意见已逐条处理**（§6）。v5 修复了 v4 探针的最后一处比较器错误（所有任务都与
最后一个任务的参考值比较），并补了无设备验证；修正后 **B2 的所有形状重新通过**，因此 v4 的
“设备侧阻塞/输入重定向不可用”结论同样**撤回**。同时完成 Host 预算闭环：planner 与 C 校验器
公式计入运行时常驻 scratch、创建时检查、重新生成 plan 包，并按**原有冻结数值政策**重新验收
（逐项容差与 v1 **完全相同**，仅哈希随包更新）。

## 0. 结论摘要

RK3588 core 1，CPU 绑核，三臂**逐轮交替**测量 4 轮 × 32 次，按臂合并 128 个样本。同一
payload（plans/weights/data 同一 SHA-256）：`baseline` = HEAD 源树，`candidate` = 工作树，
`noprobe` = 工作树同一份源码但**编译期去掉计时探针**的对照库。

| profile | baseline（合并中位）| candidate | 比值 | 吞吐（由合并中位换算）| candidate / noprobe |
|---|---:|---:|---:|---:|---:|
| Tiny Transformer b8 | 33.844 ms | **14.768 ms** | **2.292×** | 236.4 → 541.7 img/s | 14.768 / 14.870（−0.7%）|
| Tiny Transformer b1 | 7.199 ms | **4.139 ms** | **1.739×** | 138.9 → 241.6 img/s | 4.139 / 4.171（−0.8%）|
| MNIST MLP b32 | 2.832 ms | **2.198 ms** | **1.289×** | 11 298 → 14 559 img/s | 2.198 / 2.285（−3.8%）|

- 吞吐一律由**合并中位延迟**换算（`batch / p50`），不取单轮字段。
- 逐轮比值（baseline / candidate）b8：`2.36, 2.36, 2.15, 2.32`；b1：`1.75, 2.37, 1.72, 1.64`；
  MLP：`1.33, 1.26, 1.13, 1.47`。
- **task / submit / 上传 / 下载数量与字节逐臂完全相同**（b8：71/71、135 上传、71 下载、
  312 832 B 上、230 144 B 下），由比较工具逐字段断言。
- b8 包络未归属残差 1.0%（0.089 ms / 8.616 ms，**单次代表运行**，口径见 §3.3）。
- 数值：本轮 plan 包（accounting 修正后重新生成）按**原有冻结政策**重新验收，
  `results/20261026T053818Z-numeric/` **PASS / 0 problems**，且逐项容差与 v1 相同。
- 探针代价：candidate 与 noprobe 的差在 ±4% 内（MLP 上 noprobe 反而更慢），不作为“零代价”宣称。

## 1. 变更清单

每项标注对应方案条款；没有任何一项依赖模型名或固定尺寸。

### A1 — C 侧包络归因（方案 §4 A1）

- `rk_run_report` **冻结为原来的 15 字段 / 88 字节**，`rk_workspace_run` 不读 `struct_size`，
  写入边界与扩展前逐字节相同。
- 扩展归因走**新结构 + 新入口**：`rk_run_attribution` + `rk_workspace_run_ex`；`struct_size`
  强制校验（不在 `[V1_SIZE, sizeof]` 内直接 `RK_E_SCHEMA`，不写一字节）。
- `total_us`（入口自身 C 侧停留）与 `clock_us`（探针自身开销）是自证用的见证量。
- `-DRK_PLAN_PROBE_OFF` 用同一份源码编译出探针被编译掉的库，作为端到端对照臂。

### A2 — 静态解析、预分配与 Host 侧瘦身（方案 §4 A2）

- C：workspace 创建时按不可变 plan 预分配 `co_program` 发射数组、输出 tile 计数、staging
  arena 与偏移表；**下载 tile 改用 workspace 常驻 scratch**（此前每次 download 一次
  `malloc/free`）。
- Python：`HybridRunner` 创建时把每个 RKPlan 节点的 binding 顺序、repeat 索引、slice 目标
  形状、首写分配规则解析成 `_NpuBindingPlan`/`_RepeatRef`；热循环不再解析 ref / 查 dict /
  重算形状。
- 下载路径三处同语义替换（§3.3 显示 download 由 1.92 → 1.16 ms）：FP16 有限值扫描改为
  4 lane 一组的整数判定、publish 改为按行 `memcpy`、`co_download` 元素拷贝按宽度特化。
  FP16 判定用**穷举**证明：65 536 个 bit pattern × 每个 lane 位置与
  `isfinite((double)h)` 逐一比对（`tests/c/fp16_scan_test.c`）。

### B1 — 静态 repeat 成组入口（方案 §5 B1、§4 A4）

`rk_workspace_run_group(ws, bindings, nrepeat, rk_group_report *report, diag)`：一次 native
调用执行同一不可变 plan 的 N 个独立 binding 集。

- 每个 repeat 仍执行**全部**原有检查；submit 仍逐个阻塞提交，**数量不变**。
- 发布边界整组一次；任一 repeat 失败不发布任何字节，`fail_repeat` 定位失败下标。
- **Host 预算（v3 修正）**：扩容上界按**总常驻占用**核算，即
  `rk_workspace_host_bytes(ws, capacity) = arena + 下载 tile + 发射数组 + 输出簿记`，与
  `host_budget_bytes`（未声明则用校验器重算的 `claimed_host_bytes`）比较，天花板不低于单次
  运行已占用量；另做乘法溢出检查。超限 `RK_E_BUDGET`，不分配、不发布、不 poison。
- 锁作用域（v3 修正）：`busy` 的释放只在**成功取得锁**的路径执行，由
  `workspace_run_body` / group 入口的 `out:` 各自负责；被拒绝的调用不触碰他人所有权。

### Guard — 同覆盖率的精确扫描（方案 §4 A4 允许范围内）

`co_guard_check` 由逐字节遍历 32 KiB slack 区改为逐 8 字节精确比较；未删除检查、未跳过
cache 同步；`tests/c/guard_scan_test.c` 在 13 个偏移（含非对齐与两个子区两端）破坏字节并
断言仍报错。

### 测量工具

`profile_bridge_envelope.py`（设备 model、实际亲和掩码、NPU 频率、逐次样本、包络归因，
无归因时 `envelope_available=false`）、`compare_bridge_envelope.py`（多臂、跨轮合并、
**吞吐由合并中位换算**、单样本字段标注来源轮次、结构与字节逐字段比对）、
`run-bridge-envelope.sh`/`-native.sh`（先全建后逐轮交替、`taskset` 绑核、每轮记录
loadavg/freq/亲和）、`plan_envelope_probe.c`（原语成本地板）、
`independent_mm_chain_probe.c`（B2 有界验证）、`run-numeric-replay.sh`（不重标定的
post-freeze 复放）。

## 2. 方案第一个包要求回答的四个问题

**Q1 QK/PV 的 C workspace 未细分包络主要由什么构成？**
在“旧算法 + 只加归因”的中间构建（`results/20261026T015339Z-envelope`，未交替、未绑核）上
测得 C 侧停留份额：guard 门 44.9%、阻塞 submit 32.2%、download 10.4%、pack 6.6%。guard 的
大头不是 ioctl（一次 prep+fini 3.5–3.8 µs），而是每次调用对每个 data BO 重扫 32 KiB slack
区的字节循环——这就是改成同覆盖率向量化扫描的依据。**口径说明**：该份额是中间构建上的
指示性测量，不是 baseline 二进制上的读数；优化后的直接测量见 §3.3。

**Q2 不减少 Submit，只减少重复解析/分配/入口调用能得多少收益？**
b8 33.844 → 14.768 ms（2.292×），四轮一致。已消除的成本类别：每次 1 MB 发射数组与
per-call staging 的分配释放（A2）、每次调用 3 个 data BO 各扫 32 KiB（Guard）、64 次
per-repeat Python/ctypes 过渡收敛为 2 次 group 入口（B1）、每次 download 的 malloc/free 与
逐字节扫描/拷贝（A2 续）。**不把总收益拆成这些项各自的精确贡献**——那需要逐项回退构建，
未做；可直接测量的部分见 §3.3 与记录。

**Q3 32 个 head 任务能否合法成组？**
**B2 的设备前提已用有界实验证明（§5）**：32 个各自持有独立输入/权重/输出 BO 的 MM 任务放进
一次 chain，输出与 32 次独立提交**逐位相等**，耗时 1546 → 389 µs。但**产品路径仍未接通**：
现有 group 入口每个 repeat 复用同一组 BO，无法用于 chain；要接通需要“静态 repeat 展开成多
任务 plan + 新配方 `independent_mm_v1` + 清单版本化”。本包没有绕过 chain 白名单，三次测量
中 `submit_count == task_count` 恒成立。

**Q4 形成的能力是否对静态 B/H repeat 的一类图生效？**
是。触发条件是 manifest 声明的 `repeat` 轴；C 侧按 `ninputs+noutputs` 由 plan 推出，不出现
8/4/16/32 之类常量。MNIST MLP（无 repeat 轴）走单次路径，仍获得 A2/Guard 的收益（1.275×）。

## 3. RK3588 测量

### 3.1 方法

- 三臂同 payload；`baseline` 与 `candidate` 各用自己的 C runtime 与 Python 包，`noprobe`
  与 `candidate` 共用源码，只换编译开关与目标库。
- **先全部构建，再逐轮交替测量**（第 r 轮内依次测三臂的同一 profile），四轮后按臂合并。
- `taskset -c 6` 固定 CPU；每轮记录 `/proc/loadavg`、`rknpu/freq`、`rknpu/load` 与亲和掩码，
  profiler 再从 `sched_getaffinity` 记录实际掩码。
- core 1 固定；每臂每轮 32 次 + 8 次 warm-up；权重加载与 runner 构造在区间外。

### 3.2 结果（全部由合并样本导出）

```
profile          arm        n    min     p10     p50     p90     max   img/s(batch/p50)
tiny_vit_b8      baseline  128  28.907  31.848  33.844  36.051  38.372    236.4
tiny_vit_b8      candidate 128  10.152  12.076  14.768  16.297  17.632    541.7
tiny_vit_b8      noprobe   128  10.404  12.220  14.870  16.219  17.455    538.0
tiny_vit_b1      baseline  128   6.442   6.754   7.199   7.712   8.810    138.9
tiny_vit_b1      candidate 128   3.591   3.700   4.139   4.512   5.164    241.6
tiny_vit_b1      noprobe   128   3.615   3.800   4.171   4.549   5.720    239.7
mnist_mlp_b32    baseline  128   2.371   2.520   2.832   3.106   3.665  11298.1
mnist_mlp_b32    candidate 128   1.606   1.800   2.198   2.480   4.010  14558.9
mnist_mlp_b32    noprobe   128   1.708   1.868   2.285   2.520   3.140  14006.3
```

（本轮为 `results/20261026T055654Z-envelope/`，plan 包已含 accounting 修正。）

逐轮比值（baseline / candidate）b8 `2.36, 2.36, 2.15, 2.32`；b1 `1.75, 2.37, 1.72, 1.64`；
MLP `1.33, 1.26, 1.13, 1.47`。单轮值只作波动参考，结论只用合并中位。

### 3.3 优化后 b8 的逐区间归因### 3.3 优化后 b8 的逐区间归因（**单次代表运行，第 1 轮**）

C 侧停留 8.616 ms，探针 0.03 ms 量级，未归属残差 0.089 ms（1.0%）。下表是**一次运行的
分解**，不是 128 次样本的统计量；占比以该次运行的 C total 为分母：

| 区间 | 时间 | 占 C 侧 | 含义 |
|---|---:|---:|---|
| submit_wait | 5.242 ms | 60.8% | 71 次阻塞 submit |
| download | 1.135 ms | 13.2% | BO 同步 + 拷出 + 有限值扫描 + 落 staging |
| pack | 1.045 ms | 12.1% | 135 次上传（BO prep/memset/packing/fini）|
| guard | 0.300 ms | 3.5% | 9 个发布边界的 slack/guard 门 |
| validate | 0.243 ms | 2.8% | binding/有限值/alias 校验 |
| output_sync | 0.200 ms | 2.3% | submit 后的输出 BO 握手 |
| command | 0.195 ms | 2.3% | 命令 BO prep/pack/verify/fini |
| emit | 0.145 ms | 1.7% | co_program 发射 |
| publish | 0.016 ms | 0.2% | staging → 调用方 buffer |
| 残差 | 0.089 ms | 1.0% | 仍未归属 |

同一轮的原语地板（`out-candidate/primitives.jsonl`）：`bo_sync_read/write_pair` 3.79 µs、
`guard_check` 13.1 µs、`download_cube16` 6.7 µs、`upload_cube16` 5.8 µs、
`submit_one_task` **70.6 µs**。最后一项说明：本轮机器状态下**单任务 submit 的固定成本就有
70 µs**，而实测每次 QK/PV submit 是 73.8 µs（5.242 ms / 71），即提交开销已经贴在设备地板上，
不再有“单次提交太慢”的空间——只能减少提交次数（B2）。

### 3.4 与 v1/v2 表述的差异（纠正）

- **吞吐列口径（v3 修复）**：v2 用四轮合并延迟却填单轮吞吐。现一律 `batch / p50`：b8 为
  238.4 → **530.5** img/s（v2 的 230.7 → 507.8 作废），b1 为 139.2 → **235.1**，MLP 为
  11 154 → **14 225**。
- **归因口径（v3 修复）**：v2 把单轮代表样本的分解称作“128 次中位代表”。现明确标注为
  “单次代表运行，第 1 轮”，占比分母是该次运行的 C total。
- v2 写“优化前包络 100% 未归属”→ 已改为“旧入口不提供归因，baseline 上无法直接读出区间”。
- v2 写“guard 占 42%”→ 现按中间构建实测 44.9%，并注明是指示性测量。
- v2 写“b8 基线 36.388 ms”→ 绑核后为 33.8–33.6 ms（不同 campaign）。
- v2 写“MLP 差异在噪声内”→ 交替+绑核后稳定 1.275×。
- v2 写“预算问题已关闭”→ **撤回**：v2 只核算 staging，未计发射数组等其他常驻暂存；
  v3 改为按总占用核算（§6 第 2b 行）。
- busy-poll 本轮实测 `ROCKET_BUSY_POLL=0/20/50/100` 的 submit 时间为 5.23 / 5.17 / 5.09 /
  5.33 ms，无统计意义差异，**不作为候选**（记录以免重复尝试）。

## 4. 正确性证据

| 证据 | 结果 |
|---|---|
| `scripts/run-numeric-replay.sh`（RK3588，冻结 spec 复放，不重标定） | **PASS / 0 problems** |
| 冻结数值门：33 个层门 + 5 个 profile 原始 logit 门 | 全部在 train 冻结的 `atol` 内，`rtol=0` |
| 历史全量 campaign 原始 logits 复查（15 行） | 通过 |
| 本机 10 000 张 mock 全量评测 | 与里程碑记录逐位一致（ViT b8 acc 0.9821、declared max 0.03125、hardware−CPU 0.0384254）|
| `pytest` | 244 passed / 41 subtests |
| `ctest` | 30/30 passed |
| C 主机检查 | `rk_plan_host_test` 787 checks + `fp16_scan_test` 穷举 65 536 个 FP16 pattern |
| group 入口等价性 | 32 个 repeat 的 group 结果与 32 次独立调用**逐位相等** |
| group 入口原子性 | 末位 repeat 注入失败时，前 31 个 repeat 的输出一个字节都没被改写 |
| ABI 布局 | C/Python 逐字段 offset/size 比对覆盖 `rk_run_report`(88 B)、`rk_run_attribution`、`rk_group_report` |

回归有效性经过**反向验证**（把缺陷重新引入，对应断言立即失败）：越界写 → `tail_intact`；
staging-only 预算 → `rc == RK_E_BUDGET` 与 `untouched`；BUSY 路径无条件解锁 → `busy == 1`
断言与后续 `RK_E_BUSY` 断言。

## 5. B2 有界验证：设备前提成立（两轮撤回后的最终证据）

这一节的结论两次被推翻，原因是探针本身：

- v4 的探针把**每个任务的参考值都下载到同一块 `ref`**，循环结束后只剩最后一个任务的结果，
  却拿它比较所有任务 ⇒ 不同输入“失败”、相同输入“通过”，与 provider 无关；
- v4 的负向控制又把 buffer 与自身比较，恒不成立 ⇒ “self_check_failed”。

现已修正并加了**无设备**验证：`src/rk_task_ref.h` 把“按任务切片”的比较单独实现，
`tests/c/task_ref_test.c` 断言（a）各任务对自己的切片通过、（b）交换两个任务必须被检出、
（c）沿用“全部比较最后一个参考值”的历史写法在 n 个不同任务上必然报 n−1 个失配、
（d）单字节破坏被检出。探针改为 `n × y_bytes` 参考数组 + 逐任务切片比较，负向控制改为
“复制正确结果 → 破坏参考切片 → 必须失配”。

修正后的实测（`results/20261026T033032Z-b2probe/probe_fixed.log`，RK3588 core 1，绑核）：

| 形状 | n | 失配 | 未写 | 20 ms 后 | chain 摊到每任务 | separate 摊到每任务 | 比值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 每任务独立 BO | 16 | 0 | 0 | 0 | 12.1 µs | 49.6 µs | 4.08× |
| 每任务独立 BO | 32 | 0 | 0 | 0 | 16.0 µs | 105.5 µs | 6.59× |
| 共享 BO（展开 plan 布局）| 8 | 0 | 0 | 0 | 23.1 µs | 235.8 µs | 10.20× |
| 共享 BO（展开 plan 布局）| 16 | 0 | 0 | 0 | 16.9 µs | 159.6 µs | 9.42× |
| 共享 BO（展开 plan 布局）| 32 | 0 | 0 | 0 | 15.8 µs | 140.7 µs | 8.91× |

因此：**链式提交对“每任务不同输入/输出 buffer”是数值正确的**，读、写句柄都按任务生效，
且固定提交成本被摊薄。表中的 `separate` 侧在不同轮次波动很大（同一 n 可见 129–236 µs/任务），
比值只能看数量级；chain 侧每任务 12–23 µs 稳定。**这不构成产品准入**：要接通仍需
plan 层“静态 repeat 展开为多任务 plan” + `independent_mm_v1` 配方（plan 级
`RECIPE_SIGNATURES` 与 C 校验器同时登记）+ 清单版本化，并对展开后的 plan 重新做数值验收。

## 6. 复核意见关闭表

| # | 复核意见 | 处理 | 证据 |
|---|---|---|---|
| 1 | 旧 ABI 调用越界写内存 | `rk_run_report` 冻结 88 字节且不读 `struct_size`；扩展只经 `rk_workspace_run_ex` + 强制 `struct_size` | 0x5a / 0x00 / 0xa5 三种初始化下 72 B 尾部哨兵不变（含 `ws=NULL` 路径）|
| 2a | 返回 BUSY 时释放了他人持有的锁 | 释放动作移入 `workspace_run_body`，只在成功取锁的路径执行；group 入口同样只在取锁后 `out:` 释放 | 预置 `busy=1` 后连续 3 轮调用 `run`/`run_ex`/`run_group`/`destroy` 全为 `RK_E_BUSY` 且 `busy` 恒为 1；释放后正常服务、完成后 `busy==0` |
| 2b | group Host 预算只核算 staging | 上界改为 `rk_workspace_host_bytes(ws, nrepeat)`（arena + 下载 tile + 发射数组 + 输出簿记），天花板不低于单次占用 | 用运行时自己的公式导出边界：`fit` 接受、`fit+1` 拒绝，且被拒用例满足“仅 staging 未超限、总占用超限”；输出未被改写、未 poison |
| 3 | 统计口径混用（合并延迟配单轮吞吐、归因称作中位代表） | 吞吐统一 `batch / 合并 p50`；归因标注“单次代表运行，第 r 轮”；`comparison.json` 增加 `images_per_s_source`、`single_sample_round` | §3.2 / §3.3 与 `comparison.json` 字段 |
| 4 | BUSY 修复后仍在解锁后访问 `ws->plan` | 任务数在持锁期间复制进 `rk_run_state.task_count`，两个入口只用局部状态填报告，解锁后不再解引用 workspace | `tests/test_runtime_abi.py::test_no_workspace_access_after_the_run_lock_is_released`（反向验证：改回 `ws->plan` 即失败）；ASan/UBSan 全量 host test 无告警 |
| 5 | B2 探针把“没写输出”判为正确 | 参考值与 chain 结果都从哨兵开始、参考值留在 Host、逐输出比较 + guard 检查 + 比较器反向控制 | `results/20261026T033032Z-b2probe/`：修正后 31/32 失配，结论由“可准入”改为“设备前提不成立”（§5）|
| 6 | 探针失败仍返回成功、清理未初始化数组 | 任务数组 `memset` 清零并记录 `built` 计数；每个失败 stage 打印 stage 并以非零码退出；设备故障只报告不重试 | 同上；`probe.log` 中 `rc=1` 与 `status=incomplete` |
| 7 | 创建阶段 Host 预算缺口（`ceiling=max(budget,resident)` 消化超额） | 仅完成"不抬高天花板"：扩容上界改为真实预算，单次占用已超预算的 plan 拒绝增长并给出"recompile the plan"诊断 | `rk_plan_host_test` 预算段（`fit`/`fit+1` 边界 + 只 staging 超限用例）|
| 7b | 上述缺口的根因（planner/C 校验器未把运行时常驻 scratch 计入 `host`） | **已闭环**：`passes/storage.py` 与 `rk_plan_validate.c` 同时加入 `max_tasks × sizeof(co_program)` + 下载 tile + 输出簿记；`rk_workspace_create` 在分配前用同一总量对比预算并 fail closed；`CO_PROGRAM_BYTES` 由 C ABI dump 与 pytest 双向钉住 | 重新生成 plan 包（仅 `accounting.host_bytes` 变化）+ 原有数值政策重新验收：`results/20261026T053818Z-numeric/` **PASS / 0 problems**，且 `layer_tolerances`/`task_logit_tolerances` 与 v1 **逐项相同**；mock 10k 全量评测逐位一致 |
| 8 | B2 探针把所有任务与最后一个参考值比较（v4 的“设备阻塞”结论由此而来） | 参考值改为 `n × y_bytes` 逐任务切片；比较逻辑抽到 `src/rk_task_ref.h` 并加无设备验证 | `tests/c/task_ref_test.c`（各任务通过 / 交换被检出 / 历史写法必报 n−1 / 单字节破坏被检出）；`results/20261026T033032Z-b2probe/probe_fixed.log` 全部形状 0 失配；§5 结论更新 |
| 9 | B2 探针负向控制把 buffer 与自身比较，恒失败 | 控制改为“复制正确结果 → 破坏参考切片 → 必须失配” | 同上；探针 `status=ok` 且 `rc=0` |

## 7. 新的成本排名与下一步候选

按优化之后的 b8 单次代表运行成本排序（§3.3）：

| 排名 | 成本 | 占比 | 性质 | 候选 |
|---|---|---:|---|---|
| 1 | submit_wait 5.23 ms | C 侧 60% / 墙钟约 35% | 71 次阻塞 submit；当前**测量支持**优先减少提交次数（原语数据不足以证明绝对下界）| **B2 可做**：链式提交对独立输入/输出已验证数值正确（§5），下一步是 plan 展开 + `independent_mm_v1` 配方 + 清单版本化 |
| 2 | download 1.16 ms | 13% | gate/up/qkv 三个大输出占多数 | **C**：FFN gate/up → SwiGLU → down 设备子图（先证明写回位置与数值路径）|
| 3 | pack 1.06 ms | 12% | 135 次上传 | 与 C 同源：减少边界次数 |
| 4 | Python 节点停留 | 约 12% 墙钟 | per-repeat ref 解析与逐次校验 | **A2 续/A3**：per-repeat 地址计算下推到 C |
| 5 | 注册 Host action | 约 17% 墙钟 | swiglu / softmax / rmsnorm 为主 | **D**：先证明能打通设备驻留 |

## 8. 复现

```bash
cd research/tilelang-rknpu

# 本机：编译 + 全部测试（244 pytest / 41 subtests / 30 ctest / 787 C checks）
./scripts/run-host-tests.sh

# 本机：包络自洽性（mock，无设备）
PYTHONPATH=$PWD/python python3 tools/profile_bridge_envelope.py \
  --backend mock --model tiny_vit --batch 8 --runs 32 --warmup 4

# 板端：三臂（baseline / candidate / noprobe）先全建，再逐轮交替测量 + 原语地板
RK_ENVELOPE_ARMS=baseline,candidate,noprobe \
RK_ENVELOPE_PROFILES=tiny_vit:8,tiny_vit:1,mnist_mlp:32 \
RK_ENVELOPE_RUNS=32 RK_ENVELOPE_ROUNDS=4 RK_TIME_CPU=6 \
./scripts/run-bridge-envelope.sh

# 板端：post-freeze 数值验收复放（不重标定）
./scripts/run-numeric-replay.sh

# 板端：B2 有界验证（独立 MM 任务 chain vs 独立提交，逐位比较）
cmake --build <build> --target independent_mm_chain_probe && \
  taskset -c 6 <build>/independent_mm_chain_probe
```

## 9. 明确未做的事

- 没有把 32 个 MM 任务放进产品 chain；B2 的设备前提经修正后的探针判定**不成立**（§5），
  未新增配方、未改清单、未做 plan 展开。
### 未完成且已登记的事项
- **B2 实施**：plan 层“静态 repeat 展开为多任务 plan”、`independent_mm_v1` 配方登记
  （plan 级 `RECIPE_SIGNATURES` + C 校验器）、清单版本化，以及展开后 plan 的数值验收。
  设备前提已验证（§5），但这三项都未做。
- **plan 包 v1 的归档**：`models/compiled/` 下现在是计入运行时常驻 scratch 的新包；旧包的
  逐文件副本仍保存在 `results/20261024T071144Z-models/campaign/plans/` 与各 numeric
  campaign 的 `campaign/plans/` 中，可作基线复现。schema 未升版（字段结构与含义未变，
  只是 accounting 数值变化）。
- 没有实现 NPU Softmax/RMSNorm，没有改变任何 Host 算术的数值路径（D 未做）。
- 没有合并跨节点的设备子图，没有减少任何 upload/download 字节（C 未做）。
- 没有逐项回退构建来拆分各优化的贡献。
- 没有重新标定数值容差，没有改动模型、权重、预处理、batch/tail 策略。
- 没有引入多核并发、没有改动超时或 cache 同步策略；busy-poll 实测无收益，未采用。
- `mnist_mlp` 的 group 路径未被触发（无 repeat 轴），其单次调用行为与基线一致。
