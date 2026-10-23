# RK1828：硬件任务命令在哪里生成？

2026-10-16，静态分析；未连接 RK182X、未加载驱动、未执行厂商程序。

**结论：模型专用 NPU 控制程序在 PC 离线转换阶段由 RKNN3 Toolkit 编译生成，已经装进 `.rknn`；推理时，Host API / Proxy / PCIe 驱动传输模型、数据和执行请求，设备固件装载控制程序、更新地址和任务参数、排队提交，NPU 侧的 RISC-V 执行控制程序并写硬件寄存器。**

因此“Host 每次推理生成底层硬件命令”和“设备固件收到模型图后重新编译全部硬件命令”都不是本轮样本所支持的描述。若必须在运行时 Host 与设备之间二选一，**最终任务准备与硬件提交在设备端**；若问模型专用控制代码的初始来源，答案是 **PC 离线编译器**。必须区分编译机器、推理 Host、设备 Runtime、NPU 控制处理器四个角色。

## 证据覆盖范围

本地原来只有完整的 `rknn3-toolkit` checkout。依据其官方固件链接，本轮补齐 **官方 V1.1.0 板级发行包和配套 Host 驱动源码**；没有取得文档所描述的整个板级 SDK 源码工程，尤其没有设备 Runtime / NPU 驱动的可重建源码。不能将安装包称为完整开源 SDK。

| 材料 | 固定版本与实际可分析程度 |
|---|---|
| Toolkit checkout | `airockchip/rknn3-toolkit@2136cc54fa79ad289c43e56ed9cf7e63ef56439b`，V1.1.0；文档、API、二进制与 wheel 均可本地检查 |
| 板级 ARM64 TGZ | `rknn3_rk182x_sodimm_v1.1.0_installer_arm64.tgz`，31,591,729 B；静态解包得到固件、Proxy、启动脚本等 |
| 板级 x86 DEB | `rknn3-rk182x-sodimm_v1.1.0_x86.deb`，6,127,664 B；只用 `dpkg-deb -x` 解包，得到完整 `pcie-rkep-3.3.1` 源码及另一份固件 |
| 官方预转换模型 | `v1.1.0/cnn/mobilenet_v2/i8/mobilenet_v2.rknn`，186,848 B；可直接定位并反汇编内含的 NPU 控制程序 |
| 较旧交叉样本 | Khadas `android_hardware_rockchip_rk1820@b489afee824b349c0c88dc05759b6f973da6af56`；固件 Runtime 为 0.3.0b0，未用于替代 V1.1.0 证据 |

下载来源、公共链接、版本路径及下载件哈希见 [source-manifest.json](evidence/source-manifest.json)，逐项输入 SHA256 见 [input-manifest.json](evidence/input-manifest.json)。`refs/` 下保留本机下载与解包材料，但通过本目录 `.gitignore` 排除，避免将上游 checkout 和大体积 SDK 混入笔记仓库；上游许可不变。

## 直接证据：`.rknn` 已经包含会写 NPU 寄存器的程序

模型 SHA256：`e51d7a626a0ca638ad5f9cb8fd09c67731dfd491e55de57de810a18f0c4f33ba`。模型内字符串标识 Toolkit 1.1.0、Compiler 1.1.0；该文件下载自官方 RKNN3 模型目录，无须在本机执行转换器。

从二进制实际读到：

| 文件偏移 | 实际内容 | 依据 |
|---|---|---|
| `0x0000` | `RKNX` | RKNN3 模型容器标识 |
| `0x09e0` | `0x56524e4e`，ASCII `NNRV` | 与编译器内嵌 `RKNRV_BOOT_MAGIC` 一致 |
| `0x09e4` | boot version = 2 | 与 `boot_params.c` 一致 |
| `0x09e8` | entry_point = `0xa100` | 相对 command 起点，文件入口为 `0xaae0` |
| `0x09f4` | exec_type = 1 | `RKNRV_EXEC_TYPE_MODEL` |
| `0x09f8` | exec_params = `0x60` | 文件偏移 `0x0a40` |
| `0x0a40` | `0x45584d44` | 与 `RKNRV_EXEC_MODEL_MAGIC` 一致 |

按 RV32+C 静态解码，入口即包含对 `0xfff02400` 附近的寄存器写入。更容易核对的初始化段在 command 相对地址 `0xa2b0`（文件 `0xac90`）开始，包含：

```text
lui  a3, 0xfff01          # CNA base = 0xfff01000
lui  a1, 0xfff05          # DPU_RDMA base = 0xfff05000
...
sw   a4, 0x10(a3)         # CNA burst configuration, a4 = 0x7007
sw   a0, 0x10(a1)         # DPU_RDMA burst configuration, a0 = 0x700
...
lui  a0, 0xfff03          # CORE base
lui  a2, 0xfff04          # DPU base
sw   a4, 4(a3)            # pingpong enable, a4 = 0x3e
sw   a4, 4(a0)
sw   a4, 4(a2)
sw   a4, 4(a1)
```

这些寄存器基址、偏移和值，与编译器内嵌模板的 **P1 分支**逐项相符。这里的型号分支匹配来自实际模型指令，而非从 Host 库里出现一个 `REGCMD` 字符串推测。完整字节与解码见 [model-command.dis.txt](evidence/model-command.dis.txt)，启动头和候选 MMIO `lui` 位置见 [model-command.json](evidence/model-command.json)。候选位置计数不是硬件 task 数或寄存器写次数。

**这条证据直接排除了“离线模型没有低层控制代码，必须由固件从图临时生成全部控制程序”的解释。** 当前只解码了启动和部分初始化，未恢复整个 MobileNet 的逐算子寄存器配置。

## 离线编译器提供了什么

Linux cp310 wheel 内 `rknn/api/lib/linux-x86_64/rknx/librknnc.so` 为 29,495,201 B，SHA256 `36cde26bcaaa26d1115894988f53c19a3dff1bd56fff57ca40208c221df12b74`。该 ELF 虽 stripped，但包含可以原样提取的完整、NUL 结尾文本资源，**不是反编译后猜出的源码**：

- [boot_params.h](evidence/boot_params.h.txt) / [boot_params.c](evidence/boot_params.c.txt)：定义 boot magic、入口、执行参数位置、stack、kernel library、cache、LoRA、profile 等字段。
- [exec_params_model.h](evidence/exec_params_model.h.txt)：定义 `const_addr`、`internal_addr[8]`、`core_mask`、`task_id`、输入/输出数量与 `io_addr[]`；解释了为什么离线有程序，运行时仍必须更新实际地址。
- [rknrv_reg_base.h](evidence/rknrv_reg_base.h.txt) / [rknrv_reg_commit.h](evidence/rknrv_reg_commit.h.txt)：包含 NPU 基址、CNA/CORE/DPU/RDMA 寄存器提交、pingpong、burst 和中断配置代码。
- [model-build.mk](evidence/model-build.mk.txt)：调用随 wheel 携带的 `rknrv_toolchain`，使用 RV32 `rv32imfv_zicsr_zvfh` / `ilp32f`，输出 `generate_model.elf`，再 `objcopy -O binary` 得到 `generate_model.bin`。

每个资源在 `librknnc.so` 中的文件偏移、长度和 SHA256 见 [compiler-templates.json](evidence/compiler-templates.json)。

构建规则默认 `NRV_USE_ENCRYPTION=1`，源文件扩展名为 `.nrvsrc/.nrvhdr`，并使用 `-fdecrypt-source`。这表明部分生成源码存在受保护路径；不代表所有镜像、模型 command 或所有内嵌模板都不可分析。本轮仅提取已经明文存在的资源，没有解密受保护内容。

不要把这一 RV32 NPU 控制程序，与设备 RT-Thread 的 RV64 控制 CPU 程序，或示例中 C908 自定义 CPU 算子插件混为一谈。

## `rknn3_rk1820.img`：可以拆，且不是一个普通 Linux rootfs

以下偏移只属于 **ARM64 TGZ 内固件**，大小 3,598,865 B，SHA256 `4a8560b21da81e9b997f6e45f305790169523a799c01b1414addbd4a3ec11ba5`。

镜像以 `RKFW` 开头，内层 `RKAF` 在文件 `0x219ed`。按 RKAF 条目解析得到：

| 分区 | 镜像文件偏移 | 字节数 | 内容 |
|---|---:|---:|---|
| parameter | `0x221ed` | 504 | 参数文本封装 |
| subsoc_os | `0x229ed` | 2,716,192 | `subsoc_rtt.img`，含 server Runtime |
| node2_os | `0x2ba1ed` | 610,336 | `node2_rtt.img`，含 VRV Runtime 和 NPU task 提交逻辑 |
| node4_os | `0x34f9ed` | 124,960 | `node4_rtt.img`；本轮未细化其职责 |

后三项有 `RESC` 头，payload 从各分区 `0x800` 开始，`CSKYCSKY` 在 `0x804`。尽管存在历史 `CSKY` 标记，`0x834` 处可按 **RISC-V RV64+C** 解码出 `csrw mie, zero`、`csrw mip, zero`、`csrci mstatus, 8` 等启动指令。不能把标记当作 C-SKY ISA 结论。结构与短反汇编见 [firmware-partitions.json](evidence/firmware-partitions.json)、[firmware-startup.txt](evidence/firmware-startup.txt)。

固件内实际可定位的记录包括：

| ARM 固件文件偏移 | 原始文本及作用线索 |
|---|---|
| `0x1c0455` | `librknn3rt_srv version: 1.1.0 (fcad0502@2026-08-13T19:39:14)` |
| `0x1c265d` | `Loading command for core %d, src=%p, dst=%p, size=%ld` |
| `0x1c28a5` | `Loading command for core %d, dst=%p, size=%zu, encrypted=%d` |
| `0x32c8cd` | `librknn3rt_vrv version: 1.1.0 (fcad0502@2026-08-13T19:39:30)` |
| `0x329905` | Worker 接收 `RKNN3_SUB_MSG_LOAD_NETWORK` |
| `0x329bcd` | Worker 接收 `RKNN3_SUB_MSG_RUN` |
| `0x3305f5` | `Invalid boot_params for task %s` |
| `0x330d65` | `Failed to update task params for graph %d` |
| `0x330a35` | `RKNPU submit failed: core=%d, task_id=%d` |
| `0x335215` | `command_addr` 对齐检查 |

更多任务队列、输入/输出地址和 KV 地址更新记录见 [firmware-strings.txt](evidence/firmware-strings.txt)。这些日志**证明相关路径与错误处理存在**；单独一条日志不能证明一次实际执行一定经过该路径。与 `.rknn` 的 boot/exec 结构及官方架构图交叉核对后，才支持“设备端更新参数并提交预编译控制程序”的分工判断。

可分析性判断：**高于只有 opaque blob 的状态**。可以拆分、识别版本/模块/请求名、定位控制代码并进一步做字符串交叉引用；还没有设备符号 ELF、完整函数边界、经过验证的全镜像地址映射或设备端完整源码，因此本轮没有宣称恢复了全部 task ABI / 队列布局 / NPU ISA。

同名 x86 DEB 固件也是 3,598,865 B，但 SHA256 为 `c4b038a6f396f957d197f160f1763c5988d470ba06d92c715c383e5acb44d671`，与 ARM TGZ **不相同**。二者内嵌 srv/vrv 版本标识一致，仍不能混用文件偏移，见 [firmware-variants.json](evidence/firmware-variants.json)、[firmware-x86-versions.txt](evidence/firmware-x86-versions.txt)。

## Host PCIe Driver 与 Proxy：可看到通信任务，不能当成 NPU 任务

DEB 内 `usr/src/pcie-rkep-3.3.1/` 包含 `pcie-rkep.c`、`rk-pcie-ep.h`、DMA 支持文件、Makefile 与 DKMS 配置，可直接审查 C 源码。

`pcie-rkep.c` 的 `DRV_VERSION=0x00030301`；`pcie_rkep_ioctl` 从 847 行开始，处理 DMA cache sync、DMA block xfer、virtual ID、IRQ/ELBI 通知、BAR/用户内存映射、连续内存分配、控制器/slot reset。PCI ID 表含 Rockchip `0x182a`。源码定位见 [host-driver-index.txt](evidence/host-driver-index.txt)。这些是 **Host 与 EP 之间的传输操作**，没有在该驱动中发现模型图 lowering 或 CNA/DPU 计算寄存器发射器。

这个“没有发现”限定于实际检查的驱动实现，不能据此宣称 PCIe BAR 不可能用于其他访问。关键是当前 Runtime/Proxy/firmware 组合实际采用的职责划分。

Toolkit Linux ARM64 Proxy 是 stripped AArch64 ELF，但保留动态符号和大量错误日志。其 `TransferClientProxy::{dispatcher,sendProcess,recvProcess,listener}`、`cc_cmp_pcie_start_dma` 可定位；`dispatcher=0x41a2b8`，`cc_cmp_pcie_start_dma=0x44b21c`。保留 [符号](evidence/sdk-proxy-arm64-symbols.txt)、[通信字符串](evidence/sdk-proxy-arm64-strings.txt)和 [dispatcher 反汇编](evidence/proxy-dispatcher.dis.txt)。可继续恢复传输消息、分块和 buffer 生命周期；本轮未恢复完整线协议。

`cc_cmp_pcie_task_create/send_buff/start_dma` 里的 task 是 PCIe 通信上下文/通道，固件里的 `rknpu_tasks_submit` 才是 NPU 执行提交；不能看到一个 `task_create` 就认定 Host 正在创建 NPU 寄存器任务。

配套 x86 `librknn3_api_rkcp.so` 的 `rknn3_run` 在 `0x45bc0`，静态反汇编可见其组织张量 buffer 信息，并在 `0x46cd5` 调用 `RK::TransferClient::send`。见 [host-rknn3_run.dis.txt](evidence/host-rknn3_run.dis.txt)。这确认该已检查路径是请求/数据发送路径；没有逐条恢复其私有消息字段，不为未知字段编造名称。

还有一个易误判来源：Toolkit ARM64 `librknn3_api_rkcp.so` 混入了 `/dev/rknpu`、旧 `RKNPUEmitterV2/F2/T4` 和 RK3588 后端字符串。不能将这些直接归到 RK1828。`librknn3_api_native.so` 的 `/dev/rknpu3` 也不能自动当成 RK182X PCIe Host 路径：官方文档区分 native 与 rkcp 后端。

## 证据强度与仍未解决的边界

官方 Development Guide 的协处理器架构图（PDF 第 8 页）把 Runtime、Task Scheduler、NPU Driver、register ops 放在 RK182X 端；C API 文档明确 rkcp 经 localsocket 到 Proxy；板级 Quick Start 给出 Host 驱动结构、固件下载与设备控制方法。摘录见 [official-doc-excerpts.txt](evidence/official-doc-excerpts.txt)。这些描述与上述实际文件相互吻合。

当前可以确认：

1. **控制程序初始来源**：官方离线模型里已经有可解码的寄存器控制程序；编译器中有对应源模板、链接布局和生成 binary 的规则。
2. **运行时责任边界**：Host 发请求和数据；设备 Runtime 有 command 装载、boot/exec 参数更新、任务排队和 NPU 提交路径。最终 MMIO 写由设备/NPU 侧执行已下载控制程序产生。
3. **不能简单照搬 RK3588 Regcmd 提交**：这里至少增加了模型内 RV32 控制程序与设备端 Runtime；公开 Host PCIe ioctl 是传输接口，不能当成直接 NPU submit ABI。

仍未完成：每个算子的完整 command 格式、task descriptor 的精确二进制布局、运行时可能进行的全部重定位/动态派生、设备内部全部处理器映射、实际链路抓包和真机命令执行轨迹。部分算子可在设备执行时继续计算动态寄存器值，所以“离线生成”不等于每个最终寄存器值都在导出时固定。

最有效的后续验证是同一版本/同一模型上抓取设备端“装载后、run 前、submit 前”的 command 与 exec_params 快照，对比 Host 输入变化后的差异；配合设备控制处理器入口与 MMIO trace，可把当前静态分工结论细化为完整任务协议。此步骤需要设备访问，本轮未做。

## 复核

下载件与解包树保留在本机 `refs/`。分析脚本不导入 wheel、不 dlopen 厂商库、不运行 installer/postinst：

```sh
python3 research/rk182x-re/analyze_sdk.py
```

需要 Python 3、系统 `file/readelf/nm/objdump/llvm-objdump-16/dpkg-deb`；Capstone 可选，用于上述 RV32/RV64 短反汇编。本次环境已有这些工具。脚本提取原始明文资源、验证启动头/入口、解析分区、输出哈希与带偏移证据。
