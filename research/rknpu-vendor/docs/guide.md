# RK3588 RKNPU 使用与能力指南

本指南合并原能力矩阵、逆向操作手册和上游对照。
实验依据见 [台账](RKNPU-实验台账.md)，程序入口见 [实验目录](../experiments/README.md)。

结论对应 Orange Pi 5 Plus、内核 `6.1.43-rockchip-rk3588-rknpu098`、RKNPU 驱动 `0.9.8`。
SDK 相关观察对应 RKNN Toolkit/Runtime `2.3.2`。
能力状态截至 2026-10-03，覆盖台账 §7.194；本文于 2026-10-08 合并整理。
下文的 §7.x 编号均指台账条目，其他章节编号指本指南。

## 1. 已验证能力

手写 regcmd 已覆盖卷积、反卷积、池化、affine、训练原语及多核、切分和 task 链。
当前工具按算子族发射寄存器程序，新算子需要单独表征。
下表区分真机验证、机制已知和语义已解但尚未实现的路径。

| 判据 | 含义 |
| --- | --- |
| BIT-EXACT | 整数逐值或 FP16 位模式一致 |
| ≤ N ULP | 与参考相差不超过 N 个最低有效位 |
| round-exact | 量化往返后与定点参考逐值一致 |
| CLOSED-LOOP OK | 前向、反向、权重写回和重算闭环通过 |
| 回归门 | 历史验收：INT8 74/74、FP16 79/0/0；复跑方法见 §7 |

### 1.1 前向推理算子

| 算子 | 状态 | 覆盖域 | 证据 |
| --- | --- | --- | --- |
| **int8 Conv** | **BIT-EXACT** | 矩形 / 1D / 方形核、pad、stride、多 ic/oc 组 | 硬化回归 74/74 |
| **fp16 Conv** | **BIT-EXACT / ≤16 ULP** | ic 1..128、oc 4..256、矩形核、pad、stride、5×5、深通道 | 回归 74/74；§7.36、§7.128 |
| **int8 Deconv** | **BIT-EXACT** | 任意核形状 × S∈{2,4,8} | §7.27 |
| **fp16 Deconv** | **BIT-EXACT** | 任意核形状 × S∈{2,4,8} | §7.27 |
| **PPU MaxPool** | **多组 BIT-EXACT** | 输入高 `ih ≤ 63`；按行切分后 `ih=128` 亦可（整图 0 失配） | §7.13、§7.64、§7.90 |
| **int8 DPU BN / BS / BRDMA affine** | **round-exact** | 逐通道 `γ·x+β`、per-channel ALU add | §7.3、§7.12 |
| **fp16 Bias（BS+BRDMA）** | **BIT-EXACT** | fp32 ALU 表 + fp16 MUL；语义已定式 | §7.24 |
| **fp16 独立 BN（NRDMA 路径）** | **BIT-EXACT** | fp32 表 | §7.23 |
| **逐元素乘（Mul）** | **已表征** | 走 `BS_MUL` 通路 | §7.175 |
| **Depthwise Conv** | 机制已知 | `CONV_MODE=3` + `CORE_MISC_CFG_DW_EN`；**未纳入当前硬化回归** | 本指南 §6 |
| **FC / 1×1 matmul** | 机制已知 | 作为 1×1 conv 特例（DMA 打包切段） | 本指南 §6 |
| **Transpose / Reformat** | **可离线解码，发射器未构建** | `em=0x18`（纯 RDMA）/ `em=0x0d`（CNA 链）；perm→编码映射已解 | §7.194 |
| **LayerNorm** | **数值链已解，发射器未构建** | transpose→减均值→rsqrt(var)→归一化+affine 全链逐值命中 | §7.194 |
| **通用归约（sum/mean/max 任意轴）** | **不支持** | PPU 仅局部池化归约；需 host 侧或拆成多 task | 见 §3 |
| **Loss（CE/MSE）** | **不支持（可分解）** | 需跨通道归约 + log/exp | 见 §3 |
| **Sigmoid / Tanh / Softmax** | **表语义已解，发射器未构建** | LUT 表内容已定量解码（见 §1.4） | §7.146–§7.153 |

### 1.2 反向 / 训练原语

| 原语 | 状态 | 说明 | 证据 |
| --- | --- | --- | --- |
| **int8 wgrad**（dot / 系数矩阵 / 真实张量闭环） | 已完成 | 单层 conv 的 `∂L/∂W` 全 27 坐标 round-exact | §7.4/§7.5 |
| **fp16 wgrad**（CNA K 维归约 dot / conv3×3 / batch） | **BIT-EXACT** | 42/42 | §7.25 |
| **fp16 真实训练闭环**（单层 3×3/IC=3/OC=16/8×8） | **BIT-EXACT** | 正向 + dW + 写回重算，10/10 相位 | §7.26 |
| **深通道 fp16 训练闭环**（ic≤128 / oc≤128） | **BIT-EXACT / ≤1 ULP** | `-DIC`/`-DOC` 变体 | §7.37 |
| **深通道 fp16 双层跨层闭环** | **2-LAYER CLOSED-LOOP OK** | `d3_16`/`d64_64`/`d128_32`/`d48_48` | §7.40 |
| **深通道 int8 训练闭环**（ic≤128 / oc≤128） | **CLOSED-LOOP OK** | FWD=0 mismatch、WGRAD round-exact、FWD2 BIT-EXACT | §7.41 |
| **fp16 双层跨层闭环**（层2 直读层1 输出面） | **≤16 ULP / 近零绝对残差** | 宽扫描 180/180 | §7.28 |
| **conv→pool→conv 三层推理** | **BIT-EXACT** | 管线配方 | §7.192 |
| **int8 训练微步闭环**（在 conv→pool 激活上 in-place 写回） | **CLOSED-LOOP OK** | dY 缩放↔饱和率 100%→0% 定量 | §7.193 |
| **dgrad**（`∂L/∂x`，转置卷积） | 已验证 | `DECONV=1` + `DECONV_X/Y_STRIDE=S−1`，S∈{2,4,8}，max_diff=0 | §7.27 |
| **梯度外部数据注入**（`--in-x-file`/`--in-dy-file`） | **BIT-EXACT** | 任意数据当 dY，grads 9216/9216 round-exact | §7.155–§7.163 |

### 1.3 调度、切分与同步

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| **多核并行** | **闭环**：`--subcore-hi`（槽位 2..4 + `core_mask=0x7`），≈1.95× 吞吐 | §7.170 |
| **单 subcore 多 task 链** | **闭环**：流内尾写链 PC_BASE/AMOUNTS/OP_EN=0x18 + `slot.task_number=N`；3/3 哨兵命中 | §7.173、§7.188 |
| **一次 submit 多 task** | 全部闭合 | §7.110–§7.188 |
| **输出行切分**（多 submit，各 tile 独立输出面） | 已落地：`--tile-oh` / `--tile-auto` | §7.31、§7.96–§7.102 |
| **ic 维切分**（多 submit + host fp32 累加，仅 fp16） | 已落地：`--ic-groups G`，最坏 ≤1 ULP | §7.33 |
| **二维切分（行 × ic）** | 已落地 | §7.34 |
| **三维切分（列 × 行 × ic）** | 已落地：`--tile-ow`，五种切分同 hash | §7.35 |
| **oc 维切分** | 已落地：`--oc-groups G`，各组写不相交输出平面 | §7.36 |
| **conv 前向分片 + 非方图绕行** | 已落地：`rows = max(ALIGN_UP(…,16), IW)`，散回按 `npu_off` 逐原子拷贝 | §7.96–§7.117 |
| **PPU 按行切分** | 已闭合：lead 行方案，`ih=63/128` 整图 0 失配 | §7.90 |

### 1.4 观测 / 分析能力（非算子支持，但已掌握）

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| **`.rknn` 离线取证**：权重 blob（硬件布局，64 B 对齐直拷）、tensor 元数据、ONNX 内嵌 perm | 可用 | `rknn_blob_dump.py`、§7.194 |
| **regcmd 命令流抓取与解码**（规范 hook，`0x4104` LUT 计数等） | 可用 | `lut_scan.py`、§7.168 |
| **LUT 语义**：sigmoid/tanh 精确 Q15 激活表、softmax 线性斜坡、attention 反查表 | 已定量解码 | §7.146–§7.153 |
| **派发指纹**：`f0` param、CNA 块出现程度判别算子类别 | 可用 | `vendor_op_signature.py`、§7.150 |
| **厂商负载派发口径**：26 个真实模型一律 `core_mask=0` + 按 subcore 分派 | 已定量 | §7.137 |
| **em 分类**：`0x18` 纯 RDMA reformat / `0x0d` CNA 链 / `0x1d` 需 RDMA 的 CNA 计算 | 已解 | §7.194 |
| **单元使能位表**：bit0=OP_EN、bit2=CNA、bit3=DPU、bit4=DPU_RDMA、bit5=PPU、bit6=PPU_RDMA；**CORE 无位** | 已解 | §7.194 |
| **回归与自检**：三工具 R1–R6、trace 报告 `--check`、台账索引校验 | 已有验收记录 | §7.159、§7.164 |

---

## 2. 已知限制

| 限制 | 影响 | 绕行 / 现状 |
| --- | --- | --- |
| **PPU 输入高硬上限 `ih ≤ 63`** | 大特征图不能一次池化 | `--tile-oh` 行切分（§7.90）；`min_pool` 已加告警 |
| **PPU `k≥3 + stride=1 + 大图` ETIMEDOUT** | 部分池化配置不可用 | 仍挂账，未找到绕行 |
| **conv 暂存输入面必须与 `IW` 同高（方图，严格等号）** | 真非方输入不能直接支持 | `rows = max(ALIGN_UP(…,16), IW)` 方图绕行（已验证）；真非方需复刻 vendor 整套输入准备（成本高于收益，暂不做） |
| **单行字节超 CBUF bank** | 一行装不下 | 列切分（`--tile-ow`，§7.35） |
| **输入 tile 容量** | `ceil(in_bytes/32768) + 权重 bank > 12` 时静默算错 | 发射器 `--auto-cfg` 自动规划；越界打 `OUT-OF-ENVELOPE` 并建议 `--tile-auto` |
| **`WEIGHT_BANK` 过小 ⇒ submit 6 s 超时** | 提交挂死 | auto `ceil(wbytes/32768)+1` 安全（§7.38） |
| **fp16 BS 状态切换脆弱** | 连续切换 BS/BRDMA 时偶发错 | §7.19–§7.21；机制已定式 |
| **fp16 独立 BN 交替 ETIMEDOUT** | 偶发超时 | 重试即消（§7.23） |
| **int8 外部数据 + `--w-ones` 输出恒为零点 128** | 探针陷阱 | 设计用例时避免（§7.158） |
| **`--ov` 与 `--chain-pc` 不兼容** | 链路径不能覆写输出 | 用 `--blk-flip` 等替代（§7.187.7b） |
| **`--subcore 2/3/4` 裸用挂内核** | 填错槽位 | 必须用 `--subcore-hi`（§7.170） |
| **INT16 输出会 timeout** | 老文档记载 | **本次装置未复现**；若用于训练需单独确认（旧结论存疑） |

---

## 3. 待实现与未表征路径

| 项目 | 当前边界或可行方向 | 依据 |
| --- | --- | --- |
| 通用归约与 Loss | 现有发射器未提供任意轴归约；可由 CPU 处理或分解为 task | 本指南 §1 |
| 优化器与自动微分 | 训练闭环已有手工更新；SGD/Adam 和自动反向图尚未形成通用接口 | §7.193 |
| LUT 激活发射器 | 表内容已解，预加载 task 的发射尚未工程化 | §7.146–§7.153 |
| LayerNorm 与 Transpose | 可离线解码，尚无手写发射器 | §7.194 |
| `em=0x0d` 残余 | bit1 语义、LN 迭代逐 task 配属、p0321 的 24-task 几何 | §7.194 |
| `ic_real=1` 扩展 | `ih<8` 的 bank 边界仍需表征 | §7.1 |
| SDMA、DDMA、Global | 尚未系统取证 | 本指南 §4.1 |
| 完整输入准备与 RK3576 | 复杂布局准备未复刻，跨 SoC 兼容性未验证 | §7.119～§7.124 |

FP16 累加器至少具有 FP32 精度，证据是 `2048 + 1 − 2048` 探针输出 1（§7.17）。
这与是否支持 FP32 输入、输出或独立算子是不同问题。
旧文档“无 FP32 累加”的表述与该实测冲突，引用精度结论时以探针为准。

## 4. 命令格式与布局

- **字格式**：`[63:48]=tgt [47:16]=val [15:0]=reg`（lod 条目 8 B = `reg(u16 LE)@0` + `val(u32 LE)@2`）。
- **int8 布局**：输入/输出行主序 `atom = g·w·h·FA + y·w·FA + x·FA`（`FA=16`）；IC=1 行步长对齐 16；权重 layout A = `[oc1][i1(32)][ky][kx][oc2][i2]`；`oc` 打包跨度 ≤32 取偶对齐、32<x≤64 必须取 64（否则跨 32 槽组 mismatch）。
- **fp16 布局**：`PROC_PRECISION=2` 四处；`WEIGHT_SIZE0` 与 `WEIGHT_SIZE1` 均为字节数都要 ×2；输入像素 16 B=8 fp16、平面步长 `iw·ih·16 B`；`c.ic = ic≤16?16:ALIGN_UP(ic,32)`；多 ic 输入打包须 `--f16-pix-elems 8`。
- **CBUF**：共 12 个 32 KB bank；`DATA_BANK + WEIGHT_BANK == 12`；`FEATURE_GRAINS = max(80, ⌈2·32768/row_bytes⌉+1)`（取偶，截到 1022）；发射器 `--auto-cfg` 自动规划。
- **fp16 BS/BRDMA**：`out_f16 = f16((conv_fp32 + ALU_fp32) × MUL_fp16)`，无 shift。
- **单元使能位**见 §1.4。**`enable_mask` 在 0.9.8 上不参与执行**（驱动不读该字段）。
- **数值精度**：fp16 累加器 ≥ fp32（决定性）；INT8 累加 16 对齐；FP32 数据路径需另行表征。

---

### 4.1 ABI 与寄存器

legacy 字符设备使用 ioctl magic `r`；本装置 DRM ABI 使用 `d` 和 `0x40 + command_nr`。
ACTION、SUBMIT、MEM_CREATE/MAP/DESTROY/SYNC 的编号分别为 0、1、2～5。
`rknpu_task` 为 packed 40 B，`regcfg_amount` 的单位是 64 位命令条目。
驱动读取 `round_up(regcfg_amount + 4, 2)` 条，包含 trailer。
结构体布局以 [Hook 源码](../source/hook/rknpu/rknpu_ioctl_hook.c)和 v0.9.8 驱动为准。

| 单元 | block code | 寄存器窗口 |
| --- | --- | --- |
| PC | `0x0100` | `0x0000` |
| CNA | `0x0200` | `0x1000` |
| CORE | `0x0800` | `0x3000` |
| DPU | `0x1000` | `0x4000` |
| DPU_RDMA | `0x2000` | 独立读通道 |
| PPU | `0x4000` | `0x6000` |
| PPU_RDMA | `0x8000` | 独立读通道 |

常见写命令的 op 为 block code 加 1。DDMA、SDMA 窗口分别为 `0x8000`、`0x9000`。
block code 和寄存器窗口是两个字段。

| 单元 | 名称 | offset | 用途/已知字段 |
|---|---|---:|---|
| PC | `PC_OPERATION_ENABLE` | `0x0008` | block 启用 |
| PC | `PC_DATA_ADDR` | `0x0010` | PC 数据基址 |
| PC | `PC_DATA_AMOUNT` / `PC_REGISTER_AMOUNTS` | `0x0014` | task 的寄存器 entry 数 |
| PC | `INT_MASK/CLEAR/STATUS/RAW_STATUS` | `0x0020/24/28/2c` | 中断 |
| PC | `PC_TASK_CONTROL` | `0x0030` | task control |
| PC | `PC_DMA_BASE_ADDR` | `0x0034` | DMA 基址 |
| PC | `PC_TASK_STATUS` | `0x003c` | task status |
| CNA | `CNA_CONV_CON1` | `0x100c` | precision、conv mode |
| CNA | `CNA_CONV_CON2/3` | `0x1010/0x1014` | kernel groups、stride |
| CNA | `CNA_DATA_SIZE0/1` | `0x1020/0x1024` | 输入 width/height/channel |
| CNA | `CNA_CBUF_CON0` | `0x1040` | weight/data bank |
| CNA | `CNA_FEATURE_DATA_ADDR` | `0x1070` | feature 基址 |
| CNA | `CNA_DCOMP_ADDR0` | `0x1110` | 解压权重基址 |
| CORE | `CORE_MISC_CFG` | `0x3010` | precision、`qd_en` |
| CORE | `CORE_DATAOUT_SIZE_0/1` | `0x3014/0x3018` | 输出尺寸 |
| DPU | `DPU_S_POINTER` | `0x4004` | 寄存器组指针 |
| DPU | `DPU_FEATURE_MODE_CFG` | `0x400c` | feature mode |
| DPU | `DPU_EW_CFG` | `0x4070` | elementwise、EW LUT bypass 等 |
| DPU | `DPU_EW_OP_VALUE_0..7` | `0x4090..0x40ac` | EW operands |
| DPU | `DPU_LUT_ACCESS_CFG` | `0x4100` | LUT bank/address/type |
| DPU | `DPU_LUT_ACCESS_DATA` | `0x4104` | 写入 LUT 表项 |
| DPU | `DPU_LUT_CFG/INFO` | `0x4108/0x410c` | LUT 配置/索引信息 |
| DPU | `DPU_LUT_LE/LO_START..END` | `0x4110..0x411c` | LE/LO 分段端点 |
| DPU | `DPU_LUT_*_SLOPE_SCALE/SHIFT` | `0x4120..0x412c` | 分段斜率 scale/shift |
| PC/RW | `CLR_ALL_RW_AMOUNT` | `0x8010` | 清读写统计 |
| PC/RW | `DT_WR/DT_RD/WT_RD_AMOUNT` | `0x8034/0x8038/0x803c` | data/weight 流量计数器 |
| PC | `ENABLE_MASK` | `0xf008` | block enable mask |

## 5. 命令流采集与分析

### 5.1 采集

将 Hook、模型及输出放在板端 `/mnt/ssd/workspace/`。
记录 Hook、runtime、模型、输入的 SHA256，以及驱动版本和完整命令。
当前仓库的 Hook 源码位于 `source/hook/rknpu/`。
原手册的 Compose 入口依赖原工程，当前工作区未包含完整构建环境。

```bash
LD_PRELOAD=/mnt/ssd/workspace/rknpu-hook/libeperf_hook_rknpu_ioctl.so EPERF_RKNPU_IOCTL_TRACE=/mnt/ssd/workspace/rknpu-hook/trace.jsonl EPERF_RKNPU_REGCMD_MAX_BYTES=16777216   <workload>
```

当前源码默认每条 regcmd 最多捕获 16,384 B。
上例将上限提高到 16 MiB；应核对实际命令长度，确保捕获覆盖完整命令和 trailer。
旧手册“关闭限幅”的要求在此版本应通过足够大的正值实现，设为 0 不会取消上限。

Plain RKNN 可用设备上的 `native_rknn_arch_bench` 加载目标 `.rknn`：

```bash
./native_rknn_arch_bench target.rknn valid-handoff.rknn mode no-replay 1 1
```

RKLLM 路径使用目标 case 对应的 `rkllm_bench`、runtime 和模型。
两种路径的 submit 切分不同，统计命令数量时分别报告。

### 5.2 解码与闭合

在仓库根目录运行：

```bash
python3 source/hook/rknpu/analyze_rknpu_ioctl.py --strict trace.jsonl
python3 source/hook/rknpu/analyze_rknpu_regops.py trace.jsonl
python3 experiments/rknpu_trace_report.py trace.jsonl
```

`analyze_rknpu_ops.py` 可提取 `op_idx` 与节点名称，但需要同次 vendor profile 校验映射。
`analyze_rknpu_regdecode.py` 可解位域；其默认 CSV 路径沿用原工程，当前仓库未带这些表。
当前寄存器定义可查 `experiments/rnpu_regmap.h` 及 §6 的上游 XML。

| 事件 | 用途与关联字段 |
| --- | --- |
| `meta` | Hook、进程和 schema 身份 |
| `mem_create`、`mem_map`、`mmap` | BO、CPU 映射、IOVA 和 IOMMU domain |
| `submit` | task 数、core_mask、subcore_tasks、捕获完整性和硬件耗时 |
| `task` | `sequence`、`task_index`、`op_idx`、regcfg 和地址 |
| `regcmd` | 同一 task 的原始 `hex` 和 `bytes` |

用 `sequence` 连接 submit、task 和 regcmd，以独立 task 事件作为 `op_idx` 来源。
严格检查要求 task 索引连续、无重复、捕获数等于声明数，且 submit/task 序列闭合。
subcore 归属读取每个 submit 自带的 5 组槽位，不能均分或累计成全局 task 序号。

### 5.3 故障排查与测量边界

| 现象 | 排查重点 |
| --- | --- |
| arch-bench 可解码，RKLLM 首字异常 | IOVA 查找是否按 `iommu_domain_id` 过滤 |
| language 地址为 `0xffffffffXXXXXXXX` | 核对低 32 位地址回退匹配 |
| task 明细少于 submit 声明 | 旧 Hook 的 256-task cap，或固定 submit JSON 缓冲丢记录 |
| LUT 数量约减半 | 旧 Hook 将 `regcfg_amount` 按 4 B 而非 8 B 读取 |
| 回归超时或池化无输出 | 检查返回值、完成中断、bank 和形状约束；见 §2 |

`hw_elapse_time` 是整个 submit 的耗时，混合 submit 不能直接拆成算子或 task 时间。
寄存器写计数用于结构分析；LUT 写数量、MAC 理论下限和耗时残差均不能直接代表某硬件块耗时。
性能判断需要 vendor profile、受控对照或独立 microbench。

### 5.4 LUT 与 SDK lowering

`0x4100` 控制 LUT 访问，`0x4104` 写表项，`0x4108/0x410c` 配置计算和索引。
完整 sigmoid loader 有 1,102 个命令条目、1,027 次 `0x4104` 写。
其中有效表为 1,026 项，按 513 + 513 分成两个 bank，拼接成同一 Q15 曲线。
表近似 `round(sigmoid(x) * 32768)`，步长约 0.012300，起点约 −6.304853。
这些参数属于该捕获，其他函数需分别拟合。

表装载与执行是两个阶段，运行 task 可能复用已加载表。
手写 LUT 探针可以实现某函数，仍需 trace 确认 SDK 是否将对应独立算子放到 NPU。
RKNN 2.3.2 的独立 Exp、Exp2、Reciprocal、Sin、Cos 默认 CPU 路径已有观察记录。
LayerNorm 捕获表现为 DPU EW 与 CNA 归约，数值链解码见台账 §7.194。
最新 LUT 拟合和 attention 输入结果见 §7.189～§7.191 及对应结果目录。

## 6. 上游来源与 SDK 边界

### 6.1 源码溯源

完整快照清单和 commit 见 [refs 索引](../refs/README.md)。行号对应 2026-09-27 的快照。

| 来源 | 用途与定位 |
| --- | --- |
| Mesa Rocket | `src/gallium/drivers/rocket/registers.xml`、`gen_header.py`、`rkt_regcmd.c`：寄存器与发射 |
| `refs/npu/` | `include/rknnops.h`、`ops_rknn/`：独立算子实现、PPU 和 LUT 对照 |
| `refs/rk3588-npu/` | `src/npu_matmul.c` 及 `include/npu_*.h`：矩阵乘和硬件定义 |
| `refs/rknpu-driver-0.9.8/` | `drivers/rknpu/rknpu_job.c`、ioctl 头文件：板端 ABI 和调度 |
| `refs/orangepi5plus-npu/` | `librocketnpu/src/rnpu_regcmd.c`：编译器后端与权重打包 |
| `refs/rknn-toolkit2/` | SDK 算子规格及运行时 API 头文件 |

寄存器头文件的复核命令如下，历史核对结果与本仓库 `rnpu_regmap.h` 逐字节一致：

```bash
python3 refs/mesa/src/gallium/drivers/rocket/gen_header.py   --rnn refs/mesa/src/gallium/drivers/rocket/rules-ng.xsd   --xml refs/mesa/src/gallium/drivers/rocket/registers.xml   c-defines > /tmp/rkt_registers.h
cmp /tmp/rkt_registers.h experiments/rnpu_regmap.h
```

官方 v0.9.8 源码来自 `rknn-llm/rknpu-driver/rknpu_driver_0.9.8_20241009.tar.bz2`。
vendored 驱动存在少量差异；ABI 引用需明确版本。
`rknpu_job.c:666` 对完成中断作等值比较，PPU 的 `int_mask=0xc00` 与此相关。
链协议、权重布局和使能位的实测修正以台账 §7.15、§7.173、§7.188、§7.194 为准。

可扩展路径包括平均池化的 RECIP/精度配置、纯 DPU LUT task 和非 1×1 矩阵乘。
上游示例是表征起点，当前发射器支持状态仍以 §1 为准。

### 6.2 厂商算子规格

下表摘自本地 `05_RKNN_Compiler_Support_Operator_List_V2.3.2.pdf` §3.2（第 50～101 页）。
它描述 SDK 支持规格，不能替代手写发射器的验收。

| 算子 | 类别 | 关键约束 |
| --- | --- | --- |
| Add / AddRelu | 逐元素 | 任意维度广播（同 Expand）；多核联合**支持** |
| BatchNormalization | 归一化 | epsilon ∈ (0, ∞] |
| Concat | 形状变换 | axis 仅 {0,1,2,3} |
| **Convolution** | 卷积类 | kernel ≤ 31、stride ≤ 7、pad ≤ 15、dilation ≤ 31；weight 支持 per-channel；首层输入宽另有约束 |
| Depthwise Convolution | 卷积类 | kernel ≤ 8、dilation ≤ 32，stride/pad 同 Convolution |
| Convolution + {Relu, Clip, PRelu/LeakyRelu, Add, AddRelu} | 卷积融合 | 规格同 Convolution；多核联合**支持** |
| Convolution + {Sigmoid, Tanh, Softplus, HardSigmoid, HardSwish, Elu, Swish, Mish} | 卷积融合 | 规格同 Convolution；多核联合**暂不支持** |
| **ConvTranspose** | 卷积类 | **stride 仅 1/2/4/8**；group 仅 1；dilation ≠ (1,1) 时 kernel 仅 (1,1)；pad 有 4 条「不支持」不等式 |
| ConvTranspose + 12 个融合激活 | 卷积融合 | 规格同 ConvTranspose；多核联合暂不支持 |
| Div | 逐元素 | **仅 float16**；多核联合不支持 |
| Expand | 形状变换 | 5 类广播模式，h、w ∈ [1, 8192] |
| GRU / LSTM | 循环 | GRU 仅 float16；batch > 1 时须为 4 的倍数；LSTM 的 projection/cifg/has_projection 限定 0 |
| Gather / Split / Slice | 形状变换 | axis 仅 {0,1,2,3}；Slice 的 steps 有对齐条件 |
| Hardmax | 归一化 | axis 仅 1、3；axis=1 时 h×w < 16384 |
| Max / Min / Mul / MulRelu / Sub / Where / Pow | 逐元素 | Mul 的 T2 为 float 常量；Sub 广播仅 [b,c,1,1] 与标量；Pow 指数仅 0/1/2/3/±0.5 |
| Pad | 形状变换 | 仅 constant 与 reflect；reflect 限 width ∈ (0, 8176] |
| **MaxPool / AveragePool** | 池化 | kernel ∈ (0, 7]、stride ∈ (0, 8]、pad ∈ [0, 7]；`ceil_mode` 不支持；`count_include_pad` 固定 1 |
| GlobalMaxPool / GlobalAveragePool | 池化 | 输入 height、width ∈ (0, 7] |
| Relu / Clip(ReLU6) / PRelu / LeakyRelu | 激活 | PRelu 的 slope 仅标量或 C 维系数 |
| Reshape / Resize / Tile | 形状变换 | Resize 仅 nearest/linear 且整数倍 2–8；Tile 不支持广播 |
| Softmax / exSoftmax13 / exSoftmaxMask | 归一化 | axis 仅 1、3；exSoftmaxMask 的 mask 为 [1,c,h,w]，mask_value 仅 0、1 |
| Sigmoid / Tanh / Softplus / HardSigmoid / HardSwish / Elu / exSwish / exMish / exGelu / exSwooshR / exSwooshL | 激活 | 均 per-layer、int8/float16；多核联合尚不支持 |
| Transpose | 形状变换 | 列为 NPU 算子；24 种 perm 逐条约束，多处要求 h % 16 == 0 |
| exMatMul | 矩阵乘 | A [b,k,1,n] × B [b,k,1,m] → C [b,n,1,m]，k ∈ (0, 8192]；c_type 支持 add / mul |
| **exSDPAttention** | 矩阵乘 | query/key/value 均为 [b,c,h,w]；query/value 的 channel ∈ (0, 8192] |
| exNorm | 归一化 | **仅 float16**；mode 支持 layer / rms / mean_var；axes 仅 [1] 或 [1,2,3] |
| exGlu / exWindow | 其他 | exGlu 的 axis 仅 1，输出 channel 减半 |

同名算子也可能出现在 SDK 的 CPU 列表，实际落点取决于规格和转换结果。
厂商“多核联合”支持范围是 SDK 的逐算子限制，需与手写 submit 的多核验证区分。
官方 `rknn_custom_op.h` 的 target 仅含 CPU、GPU；该版本 API 未提供 NPU 自定义算子 target。

## 7. 复跑与维护

将源码、寄存器头文件和脚本放到板端对应目录，入口见 [实验目录](../experiments/README.md)。

```bash
# 板端 exp2 目录
bash run_regression_hard.sh
bash run_fp16_regression.sh
```

管线使用 `pipeline_pool_conv.sh`、`pipeline_conv_pool_conv.sh` 和 `pipeline_conv_pool_conv_train.sh`。
每次新增能力或修正限制，先记录台账证据并更新索引，再同步本指南。
历史实验中的文件名、捕获计数和旧结论保留用于审计，引用时确认后续修正。
