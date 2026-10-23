# P0：源码与环境锁

日期：2026-10-20。范围：只做锁定与重验，不混入前端功能扩展。

**总状态：`complete`（源码构建已完成；真实设备故障恢复不属于 P0）。**

| 交付项 | 状态 | 证据 |
|---|---|---|
| 源码 commit + 子模块锁定 | 完成 | `provenance/source-lock.json`、`provenance/upstream-trees/*.json` |
| 制品哈希锁定（依赖） | 完成 | `requirements.lock`（28 个制品，SHA-256） |
| 当前 wheel Host 回归 | 通过 | `.pydeps` 基线记录（计数与第 4 节一致） |
| 隔离依赖环境重验 | 完成 | `.venv-p0` + `.pydeps-p0`，`isolation` 步骤断言通过 |
| 完整 Python 文件集合校验 | 完成 | `provenance/wheel-vs-source.json`：0 缺失、0 无法解释的额外文件 |
| 源码构建及产物重验 | **完成** | [provenance/p0-source-build.json](provenance/p0-source-build.json)、`provenance/p0-source-build/` 构建日志与库哈希、源码树清单 |

## 1. 源码锁

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/tile-ai/tilelang` |
| 标签 | `v0.1.15` |
| commit | `a35f8ddf45eba16c21211ec8822d56ce5363036f` |
| 解析方式 | GitHub `git/ref/tags` API（`git ls-remote` 在本机多次 TLS 中断/超时，仅作回退） |
| `3rdparty/tvm` | `4211874e9e4b8770fd20ad0d98ab80afd1df5028`（`apache/tvm`） |
| `3rdparty/cutlass` | `b2dd65dc864e09688245b316ac46c4a6cd07e15c`（`NVIDIA/cutlass`） |
| 根 tree 条目 | 2265，未截断 |

原始 GitHub tree 响应归档在 `provenance/upstream-trees/`，可离线复核上面的
commit/SHA，不必相信脚本的算术。

### 最小补丁

**无补丁。** 后端只读取上游公开 Python 对象（`PrimFunc`/`SBlock`/`Call`/`BufferLoad`/
`BufferStore`/`For` 等），不修改 TileLang/TVM 源码；`source-lock.json` 中
`patches: []` 与 `patch_policy` 显式记录该结论。

## 2. 「安装内容 == 锁定源码」的范围化校验

按 6 个显式 scope 比较 git blob SHA（`sha1("blob <size>\0" + content)`）：

| scope | 覆盖 | 完整性口径 | 比较 | 一致 |
|---|---|---|---|---|
| `tilelang-package` | `tilelang/**`（除 3rdparty/src/lib） | 完整 | 397 | 397 |
| `tilelang-cpp-src` | `tilelang/src/**` ↔ 仓库根 `src/` | 完整 | 394 | 394 |
| `tvm-python` | `…/3rdparty/tvm/python/**` ↔ TVM `python/` | 完整 | 747 | 747 |
| `tvm-include` | `…/tvm/include/**` ↔ TVM `include/` | 完整 | 189 | 189 |
| `tvm-src` | `…/tvm/src/**` ↔ TVM `src/` | 完整 | 929 | 929 |
| `cutlass-subset` | `…/cutlass/**` ↔ cutlass tree | **仅子集**（无完整性声明） | 988 | 988 |

结果：

```text
参与比较: 3644 文件，全部逐字节一致，不一致 0
Python：完整 scope 内安装 1135 / 上游 1135
        缺失（上游有、安装无）= 0
        无法解释的额外 Python = 0
未归入任何 scope 的安装文件 = 0
```

仅有两类文件不是「源码可比较」的，均被逐条列出并说明：

- `tilelang/3rdparty/tvm/version.py` —— TVM 构建期生成，allowlist 记录原因；
- 12 个编译产物 `tilelang/lib/*.so` —— 记录各自 SHA-256，明确标注 provenance 是
  wheel 制品哈希而非源码匹配（`wheel-vs-source.json: binary_artifacts`）。

因此准确结论是：**上述 6 个 scope 内、3644 个文件（含完整 scope 的全部 1135 个 Python
文件）与锁定源码逐字节一致，无缺失、无无法解释的额外文件；编译产物不在该断言范围内。**

## 3. 依赖锁与隔离环境

`requirements.lock`：**28 个制品**，固定版本 + SHA-256，用
`pip --no-deps --require-hashes` 安装。此前遗漏的 `jsonschema`（`pyproject.toml` 声明的
依赖）及其闭包（`attrs`、`jsonschema-specifications`、`referencing`、`rpds-py`）已补入。

```bash
scripts/lock_deps.sh lock      # 生成 requirements.lock
scripts/lock_deps.sh install   # 装进全新的 .pydeps-p0
scripts/lock_deps.sh verify    # 解释器隔离 + 无重复 metadata + 导入路径
```

`pip --target` 只填充目录、不隔离系统包，因此：

- `python3 -m venv --without-pip .venv-p0`（本机无 ensurepip）——该 venv **不继承**
  `/usr/lib/python3/dist-packages`；
- 验证、pytest、CLI 子进程全部由 `.venv-p0/bin/python` 运行；
- `scripts/check_isolation.py` 断言：`sys.path` 无外部 dist/site-packages、每个第三方模块
  与每个发行版位置都在 `.pydeps-p0` 内、发行版集合与 `requirements.lock` 逐项相等；
  任一不满足即让整轮验证失败。

上一轮 `jsonschema` 来自 `/usr/lib/python3/dist-packages` 的问题已消除：本轮
`isolation.log` 记录其来源为 `.pydeps-p0/jsonschema/__init__.py`，`foreign_paths: []`。

## 4. 行为重验（隔离环境）

```bash
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 scripts/run-verification.sh
```

结果目录：`results/20261022T084627Z-host/`

| 项 | 结果 |
|---|---|
| 解释器 | `.venv-p0/bin/python`（`isolated_venv: true`） |
| isolation 步骤 | 通过（`ISOLATION OK`，0 problems） |
| ctest | 26/26 |
| `rk_plan_contract` / `rk_plan_host` | 26 / 67 checks，0 failures（`MALLOC_PERTURB_=128`） |
| Sanitizer（显式 flags + 全新 build 目录） | 26 / 67 checks，0 failures |
| pytest | 68 passed，**0 skipped** |
| evidence_complete | true |

与 `.pydeps` wheel 基线的计数逐项一致，无回退、无跳过，因此本次不存在需要解释的
AST/API 差异。

## 5. 未闭合：源码构建与产物重验

以下**已尝试的获取路径**在本机失败，结论不外推到其他环境：

| 尝试路径 | 实际结果 |
|---|---|
| `git clone --depth 1 --branch v0.1.15` | `GnuTLS recv error (-110)`，重试仍超时 |
| `codeload.github.com` tarball（`-m 55`） | 收到 9,390,763 B 后超时（≈167 KB/s） |
| `raw.githubusercontent.com` 单文件 | 50 s 超时，0 字节 |
| PyPI sdist `--no-binary :all: tilelang==0.1.15` | 该版本没有 sdist（只有 0.1.7–0.1.14） |

55 秒超时与解压体积不足以证明任何环境都无法取得源码，只说明这几次尝试失败。

闭合方式（需直连 GitHub 或可用镜像）：

1. `git clone --recurse-submodules` 到 `v0.1.15`，校验 `3rdparty/tvm == 4211874e…`；
2. 按其构建脚本编译 `libtvm`，把产物哈希写入 `provenance.json`；
3. `TILELANG_DEPS_DIR` 指向源码构建产物重跑第 4 节，与
   `results/20261022T084627Z-host` 逐项比对；若出现 AST/API 差异，先修适配并解释，
   不放宽拒绝规则来维持通过数。

在此之前本模块是 **Host 原型、P0 部分完成、板端未验收**；P1 的 K 循环工作可基于当前
已记录的 wheel 基线推进，但不得写成源码构建验收通过。

## 6. 复现命令

```bash
cd research/tilelang-rknpu

TILELANG_DEPS_DIR=$PWD/.pydeps-p0 python3 scripts/lock_source.py   # 源码一致性
./scripts/lock_deps.sh lock                                        # 依赖锁
./scripts/lock_deps.sh install                                     # 干净安装
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/lock_deps.sh verify     # 隔离校验
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 ./scripts/run-verification.sh     # 全量重验
```

全部命令不打开设备。

## 7. 2026-10-23 有界源码重试

按“有界尝试、失败保存具体阻塞、继续其他任务”的规则，再次尝试从锁定 tag 获取源码：

```text
command: timeout 90s git clone --depth 1 --branch v0.1.15 \
         https://github.com/tile-ai/tilelang.git .src/tilelang-v0.1.15-attempt
exit:    124 (timeout)
log:     provenance/p0-rebuild-attempt.log
```

未保留半下载目录；没有源码或 `libtvm` 产物被标记为成功。**当时** P0 为
`in_progress / externally blocked`；后续第 9 节改用可续传镜像完成了源码构建。

同一尝试的机器可读记录：`provenance/p0-rebuild-attempt.json`。

## 8. 2026-10-23 第二次有界获取尝试

在继续独立开发前再次尝试 codeload tarball，仍未完成：

```text
command: timeout 60s curl -L --fail --connect-timeout 15 --max-time 55 \
         -o /tmp/tilelang-v0.1.15.tar.gz \
         https://codeload.github.com/tile-ai/tilelang/tar.gz/refs/tags/v0.1.15
exit:    28 (Operation timed out)
received: 2,673,599 bytes (partial archive discarded)
log:     provenance/p0-rebuild-attempt-2.log
record:  provenance/p0-rebuild-attempt-2.json
```

没有可用完整源码，也没有执行/伪造源码构建产物。**当时** P0 仍未完成；后续
第 9 节通过可续传镜像完成，且仍不把部分下载当作源码证据。


## 9. 2026-10-23 源码构建完成

使用可续传的 `gh-proxy.com` 镜像取得锁定 commit 的 GitHub archives；archive 名称包含
commit，下载后计算 SHA-256 并解压到独立目录：

| 组件 | 锁定 commit | archive SHA-256 | 用途 |
|---|---|---|---|
| TileLang | `a35f8ddf45eba16c21211ec8822d56ce5363036f` | `2e85edecef4cbeb795f4f8a3b2ac2ad4020d89f3d35687c624c3bbcf429b73bc` | compiler/frontend source |
| TVM | `4211874e9e4b8770fd20ad0d98ab80afd1df5028` | `ee73c88f82d949f451e541182f69973f8c4c452f658695442931c686faffadfa` | `libtvm_*` |
| tvm-ffi | `3c35034fd1026011736e19a4e0e1ed0f22058c42` | `b2c1947d91c0b7e8092d38d31c60f0ea5ad34bc45a05b315224774bdd0dd86a7` | TVM FFI |
| dlpack | `84d107bf416c6bab9ae68ad285876600d230490d` | `9978ddb7f533fcee7e0c0ab81a978257d3f8dc475eb83319a5e632738cafb271` | DLPack headers |
| libbacktrace | `793921876c981ce49759114d7bb89bb89b2d3a2d` | `858b1225351052234f7e71f0f36622fc9ad33aac947db7816d0b443ae0dd33ce` | TVM FFI optional backtrace, source supplied |

构建方式：

- 独立 build tree：`.src/p0-build-tilelang`；源树：`.src/p0-build-src/{tilelang,tvm}`。
- `TVM_ROOT` 指向锁定 TVM 源；TVM 的 `3rdparty/tvm-ffi`、`dlpack`、`libbacktrace`
  使用上述锁定源码。
- 本次目标为 CPU/编译器/RKNPU 后端，显式关闭 CUDA、ROCm、Metal、LLVM、Ascend、
  OpenCL；因此未下载/编译 cutlass（wheel 内 cutlass 子集不参与本次 native 构建）。
- 构建使用 source-built `libtilelang.so`、`libtvm_compiler.so`、`libtvm_runtime.so`、
  `libtvm_ffi.so` 和 Cython wrapper；不加载 wheel 内旧 `.so`。
- 仍来自预构建依赖的明确项：Z3 `.pydeps-p0/z3`、Python `tvm_ffi` wrapper、
  Cython 3.0.8 wheel、Ubuntu 3.12.3 dev headers；均记录在
  `provenance/p0-source-build.json`。

实际加载路径：

- TileLang：`.src/p0-build-src/tilelang/tilelang/__init__.py`
- TVM/tirx：`.src/p0-build-src/tilelang/3rdparty/tvm/python/tvm/...`
- native libs：`.src/p0-build-tilelang/lib`
- 日志：`provenance/p0-source-build/import-paths.log`

验证结果：

- source-built import + full pytest：`213 passed, 40 subtests`
- CTest（同一 Host native build）：`27/27`
- 27 个 campaign plan 中，wheel 与 source-built 的
  `bindings/constants/buffers/views/actions/submits/publish/accounting` 全部逐项相同；
  只有 source span/hash 元数据差异（`plan-equivalence.json` 全部 `semantic_equal=true`）。
- 用 source-built TileLang/TVM 重新生成同一 27-case campaign，并在同一 RK3588 上执行：
  `results/20261023T141328Z-rk3588/`，**81/81 PASS**，三核 CLI PASS，source bundle hash
  独立重算通过。

构建系统的一条诊断信息 `Found TVM_GIT_COMMIT_HASH=9baf...` 来自 TVM 源树位于
外层 git checkout 内这一事实；实际 TVM 版本以 archive commit `4211874e...` 和
源码树清单为准，不把该诊断字段当作 TVM commit。
