# TileLang → RKNPU 实施状态

更新日期：2026-10-20。当前结论：**部分主机原型；未达到 P0–P8 MVP 完成标准，禁止作为板端准入版本**。已有测试通过不等于开发细则已满足。2026-10-20 复核发现 C descriptor 校验可接受越界 stride 和逆序 K task；详见 [复核报告](REVIEW-2026-10-20.md)。

## 1. 开工基线（P0）

执行的只读检查：

```text
git status --short          # 工作区已有大量修改/删除/未跟踪文件，原样保留
git rev-parse HEAD          # 716914a8a875ac2b0556879c6b19c9ae8365f255
cat research/upstream-components/provenance.json
cat research/custom-ops/CMakeLists.txt
```

基线构建与现有 Host 回归：

```bash
cmake -S research/custom-ops -B research/custom-ops/build-host-mvp
cmake --build research/custom-ops/build-host-mvp -j4
ctest --test-dir research/custom-ops/build-host-mvp --output-on-failure
```

结果：基线 24/24 通过。新增 RKPlan 两个 C 测试后为 26/26 通过，无原测试回归。
没有 reset/clean 工作区，没有删除或 skip 既有测试。

复核补充：上述为含其他未提交工作的工作区口径。仅导出本次暂存提交的独立副本，C 22/22、Python 36/36 通过；差异是未纳入本次提交的4项 GDN 测试。两组结果都不构成完整 MVP 验收。

## 2. 工作包状态

| 工作包 | 状态 | 实际完成 | 明确未完成 |
|---|---|---|---|
| P0 基线与依赖锁 | `in_progress` | 基线构建/ctest；`provenance.json` 记录 HEAD、源哈希、Python/TileLang wheel 版本 | TileLang 使用不可变 wheel 0.1.15，不是 git commit；P8 前仍需源码 commit lock |
| P1 真实前端提取 | `in_progress` | 用真实 `tvm.tirx.PrimFunc` 提取 rank-2 FP16、zero fill、单 GEMM、`z*sigmoid(z)`、源 span；matmul/matmul_silu 正例；多 GEMM 负例 | 只支持单 primitive tile（M/N/K≤256）；K 循环/多 tile/SwiGLU 多 GEMM 未提取；其余 activation 未提取 |
| P2 IR/Plan/schema/能力表 | `in_progress` | frozen dataclass、严格 JSON、canonical hash、三个 schema、enabled/experimental manifest、稳定错误码 | 没有实现 state/动态 shape/任意 broadcast 的 schema 语义 |
| P3 MatMul planner | `in_progress` | RKIR MatMul 的 B→W 方向规范、M/N/K 256 分片、K 链、ping-pong、尾块、任务 origin、NumericContract | 前端只产生单 tile RKIR；K 分片已有测试，M/N>256 的计划因重复 publish 被拒，不能称 M/N 分片已完成 |
| P4 组合/存储/提交 | `in_progress` | SiLU/GELU/sigmoid/HardSigmoid plan lowering；LUT 32768 元素分段；每输出 tile K 链/激活链；first-fit slot；第 9 节 S01 数值完全复现 | 同 shape Add/Mul 与常量广播没有进入 minimal 前端/planner；SwiGLU 只是负例 |
| P5 C descriptor 校验/FFI | `in_progress` | `rk_plan_create` 深拷贝；ID/几何/alias/依赖/submit/budget 独立校验；ctypes ABI size 对照；Host test 验证 descriptor 释放后 plan 可用；新增 C 测试 ASan/UBSan 通过 | C 验证器尚未拒绝 source_row_stride 越界和 K task 逆序；ASan/UBSan 未覆盖所有旧 custom-ops 回归 |
| P6 资源实例化/mock 执行 | `in_progress` | 6 slot + command workspace 分配；常量一次上传、输入每次上传；真实 `co_emit_*`；`co_execute_profiled` mock seam；输出 staging 后发布；repeated run/alias 测试 | 真机驱动成功/失败、poisoned 后的进程级恢复协议未验收 |
| P7 CLI/Python run/示例 | `in_progress` | compile-tilelang/compile-rkir/validate/explain/run CLI；`Workspace` caller-fd API；真实 TileLang matmul、matmul_silu 示例 | SwiGLU 未实现；`run` 未在设备上执行；输出文件按 `.npy` 单独发布 |
| P8 板端准入 | `blocked` | 保留拒绝执行的 `scripts/run-board.sh` 占位入口、结果目录规范 | 无 RK3588 设备访问；三核、chain/separate、guard、数值、timeout 均未执行；原脚本不强制 core/mode，复核后已移除其部署执行逻辑 |
| P9 扩展 | `not_started` | 无 | 动态 Host 广播、状态、hybrid、逐组选核、参考库模板均未接入 |

## 3. 实际测试结果

Host C：

```text
26/26 tests passed
rk_plan_contract: 25 checks, 0 failures
rk_plan_host:     27 checks, 0 failures
```

Python（`.pydeps` 中安装 pinned TileLang wheel，`TILELANG_CACHE_DIR` 指向模块目录）：

```text
36 passed in 1.30s
```

ASan/UBSan（独立目录 `research/custom-ops/build-sanitize-rk`，只跑新增两个 C 测试）：

```text
rk_plan_contract: 25 checks, 0 failures
rk_plan_host:     27 checks, 0 failures
detect_leaks=1, halt_on_error=1: exit 0
```

关键断言：

- K=256/257/512/513/4096 的 MM 类 task 数分别为 1/2/2/3/16。
- 第 9 节 `X[13,257] @ B[257,33] -> SiLU`：4 task、chain 2 Submit、separate 4 Submit。
- 8 个数据 BO；payload `60416 B`；数据请求量 `323072 B`。
- P 和 Y 复用 slot；Z 与 P 使用不同 BO；`mm_acc` prior/output 共用 BO 被拒绝。
- 256×256 输出 tile 的 LUT 拆成 2 个 32768 元素段，SiLU 激活共 3 个 task。
- `rk_plan_create` 深拷贝后 descriptor 数组可改写/释放；65-task submit 被拒；
  C verifier 不读取 Python 的 `verified` 字段。
- 真实 TileLang MatMul/SiLU fixture 结构提取成功；strict FP32 accumulator 被拒；
  SwiGLU 双 GEMM fixture 明确拒绝。

复现入口：

```bash
research/tilelang-rknpu/scripts/run-host-tests.sh
```

## 4. 改动文件

新增/修改的公共 C 代码：

```text
research/custom-ops/include/rk_plan.h
research/custom-ops/src/rk_plan_internal.h
research/custom-ops/src/rk_plan_validate.c
research/custom-ops/src/rk_plan_runtime.c
research/custom-ops/src/rk_plan_emit.c
research/custom-ops/tests/c/rk_plan_contract_test.c
research/custom-ops/tests/c/rk_plan_host_test.c
research/custom-ops/CMakeLists.txt
```

新增 Python/资料：

```text
research/tilelang-rknpu/pyproject.toml
research/tilelang-rknpu/provenance.json
research/tilelang-rknpu/README.md
research/tilelang-rknpu/IMPLEMENTATION-STATUS.md
research/tilelang-rknpu/python/rknpu_backend/{__init__,api,capabilities,cli,errors,ir,options,plan,serialize}.py
research/tilelang-rknpu/python/rknpu_backend/frontend/tilelang_adapter.py
research/tilelang-rknpu/python/rknpu_backend/passes/{infer,legalize,tile,dag,schedule,storage,verify}.py
research/tilelang-rknpu/python/rknpu_backend/runtime/{ffi,bindings}.py
research/tilelang-rknpu/capabilities/rk3588_vendor_co_v1.json
research/tilelang-rknpu/schema/{rkir,rkplan,capability}.v1.json
research/tilelang-rknpu/examples/{matmul,matmul_silu,swiglu}.py
research/tilelang-rknpu/tests/*.py
research/tilelang-rknpu/scripts/{setup-tilelang-env,run-host-tests,run-board}.sh
research/tilelang-rknpu/results/README.md
```

## 5. 数值/ABI/依赖变更

- 没有修改既有 `co_*` API 的语义；新增 C ABI 以 `rk_` 为前缀。
- 新增静态库 `rk_plan_core`、共享库 `rknpu_plan`；`custom_ops` 仅为共享链接启用 PIC。
- 所有新 C 结构自然对齐、无 `#pragma pack`，首部含 `abi_version/struct_size`。
- `co_execute_profiled` 是唯一 NPU 执行入口；`co_emit_mm/mm_acc/ew/lut` 是唯一
  寄存器生成入口；没有新增寄存器生成器或提交器。
- 参考库 `rknpu2_matmul_open` 未链接、未修改，也未声称获得其性能。
- `TILELANG_CACHE_DIR` 指向模块内 `.cache`，执行期间环境变量不会被 planner 静默读取。

## 6. 未完成项与可重放命令

1. 板端三核 × chain/separate、真实数值/guard/IRQ：`blocked`，无板卡。
2. 真实 TileLang K=257/多 M/N tile、SwiGLU、Add/Mul、常量广播的提取：
   需要扩展 adapter 与 negative fixtures，不能把 `swiglu.py` 的拒绝当作已支持。
3. ASan/UBSan 已覆盖新增 descriptor/runtime 两个 C 测试，尚未覆盖全部旧 custom-ops 回归。
4. 逐提交选核未接入 C ABI；`scripts/run-board.sh` 现在以退出码2拒绝执行，待补齐准入条件。
5. firmware/driver 版本首次实例化校验目前只有 `RKNPU_CHIP`/`ROCKET_CHIP` 提示，
   未读取真实 RKNPU 驱动 ABI 版本。

任何上述项完成后，应新增结果目录和独立数值报告，再把对应状态改为
`board_passed`；不得仅修改本文状态文字。
