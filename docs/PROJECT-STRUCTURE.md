# 项目结构与扩展约定

2026-10-14。当前实现按职责组织，历史证据和上游主题笔记保持原位。

## 目录

```text
rockchip-npu-notes/
├── README.md                    # 中文首页与当前能力入口
├── CMakeLists.txt               # 整个工作区的构建/CTest入口
├── docs/
│   ├── CAPABILITIES.md          # 支持范围、证据等级、缺口
│   ├── PROJECT-STRUCTURE.md     # 本文
│   └── archive/LEGACY-README.md # 原长首页快照与完整主题索引
├── scripts/                    # 文档精选打包、链接核验等仓库工具
├── research/
│   ├── README.md               # 当前实现/历史实验/外部组件路由
│   ├── custom-ops/
│   │   ├── include/            # 公共Tensor、算子、调度接口
│   │   ├── src/                # 实现与私有寄存器模板
│   │   ├── tools/c/            # 文件CLI、真机probe与Hook
│   │   ├── tools/python/       # 独立oracle、捕获、证据门禁
│   │   ├── tests/c/            # Host边界测试
│   │   ├── tests/python/       # 数值/协议/调度反例测试
│   │   ├── tests/scheduling/   # 小型固定trace fixture
│   │   ├── scripts/            # 上传/板端验收/结果回收
│   │   ├── compatibility.json  # 原平面文件名→规范路径
│   │   ├── results/            # 大型原始证据（忽略Git）
│   │   └── build*/artifacts/   # 生成物（忽略Git）
│   ├── rknpu-vendor/           # 历史逆向与旧发射器，保持原位
│   ├── upstream-components/   # 版本清单 + 独立外部checkout
│   └── *-*.md                 # 当前专项实验报告
├── chips/encodings/perf/guide/ # 上游按主题组织的知识与历史数据
└── RKNPU-STATUS.md             # 早期本地进展快照
```

公共库 `custom_ops` 与 `co_schedule` 是代码复用入口；CMake 传播 `include/`，私有模板只在 `src/`。
真机测试、CLI 和 Hook 不进入公共 API 实现目录。
上游生成器通过固定版本 `rocket-userspace` 引入，厂商 provider 通过 `rknpu-submit` 引入，没有复制改写它们。

## 构建和兼容性

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure

# 也可以独立构建算子子项目。
cmake -S research/custom-ops -B research/custom-ops/build-host
cmake --build research/custom-ops/build-host -j4
ctest --test-dir research/custom-ops/build-host --output-on-failure
```

旧的平面文件名保留为**相对符号链接**，不放第二份实现。
例如 `custom-ops/run-schedule-board.sh`、`schedule_analyze.py` 及旧 `.c/.h` 链接仍可使用。
这保证历史文档链接、旧命令和外部调用方继续有效；新开发应使用规范目录。
脚本用真实文件路径找到子项目根，Python 测试显式引用工具目录与 fixture。

上传源码包保留相对链接及其目标；不要只复制兼容链接而漏掉规范目录。
原始结果中的源码快照、manifest、trace、二进制和既有哈希不改写。
本轮更新了新运行的源码清单生成逻辑，使其覆盖规范子目录；旧清单仍按当时结构解释。

## 新增能力的最小步骤

1. 公共接口在 `include/` 写明精度、几何、布局、别名、BO所有权与CPU阶段。
2. 原语/组合实现放 `src/`；增加对应CMake源文件，保留上游提交 seam。
3. 必须的Host边界/反例放 `tests/`；真机probe放 `tools/c/`，独立数学参考和证据门禁放 `tools/python/`。
4. 驱动/时钟/形状、输入/输出和源码摘要写入新的结果目录，不覆盖旧轮次。
5. 用专项报告说明验证范围，再更新 `docs/CAPABILITIES.md`。

“函数已实现”“协议完成”“数值通过”“多次稳定”“性能收益”分别登记，不相互替代。
CPU fallback或混合阶段要在API和总览同时明示。

## 版本控制与证据分发

源码、接口、工具、脚本、小型 fixture、固定实验配置、兼容链接及专项报告纳入 Git。
`.env`、独立上游 checkout、构建、`results/`、`artifacts/` 和导出包保持忽略。
`compatibility.json` 登记保留的平面入口；新实现只有规范目录中的一份源码。

专项报告中的原始结果链接指向开发机本地材料。仅克隆 Git 仓库不会取得这些材料，
独立复核历史测量需另行获取相应结果目录；硬件复跑使用报告列出的脚本和固定依赖。
提交前应完成干净构建、Host 测试、兼容链接与文档链接核验，避免用现存构建掩盖漏加的新文件。

## 本轮重构验证

根目录新构建及子项目Host测试通过；Python语法、Shell语法、旧入口链接和递归源码清单检查通过。
板端用重构后的完整源码包重新构建，CTest 9/9通过，五种调度场景的96个请求/5766个task与同规模无tracing对照全部通过。
新证据：[scheduling-20261009-220520](../research/custom-ops/results/2026-10-09/scheduling-20261009-220520/summary.json)。
本轮未改算法、精度或原始历史证据；新的调度数值不替换上一轮专项报告的测量结果。
