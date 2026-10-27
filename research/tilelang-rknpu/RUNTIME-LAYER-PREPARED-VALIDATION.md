# Same-implementation prepared-validation A/B

日期：2026-10-27。目标：纠正上一轮完整模型 adapter 的混杂，判断 **仅复用
固定 binding 元数据／alias 检查** 是否能在同一 C 实现、同一 pack/executor
下产生可传导收益。本阶段不重设计 runtime，不修改 guard、同步、数值政策或
默认安全策略。

## 0. 对上一轮结论的修正

上一轮以下证据现在只能作为历史控制，不能用作因果结论：

1. **B/C 直接相减**：C direct 的 upload pack 与 runtime pack 实现不同，并且
   C 使用了更快的 finite-scan helper。MLP b32/QKV 的 C 更慢、QK/PV 的 C 更快
   都包含这些实现差异，不能全部归到“RKPlan 通用层”。
2. **C−D**：D 是驻留执行参考，不含 upload、guard、download 和 publication；
   C−D 包含实验路径差异，不是全部“不可消除必要成本”。
3. **完整模型 direct adapter**：`DirectGroupCall._run()` 每次重建输入/输出
   ctypes 指针数组，且模型探针使用节点代理；其回退不能证明局部 runtime 收益
   无法传导。它只能说明那种接线方式没有收益。

本轮改成 **同一实现、同一 workspace、同一 pack/executor、只切换一个 env
snapshot 的 prepared-validation 标志**。

## 1. 实验实现

在 `rk_workspace` 内增加实验性、默认关闭的 prepared-validation cache：

- `RK_PLAN_PREPARED_VALIDATION=1` 时，workspace 在第一次 run 完成全量 binding
  验证后缓存输入的 **id/dtype/rank/shape/nbytes/data pointer** 及对应的输出
  binding 元数据；
- 后续 call 仅在**完整签名逐项相同**时跳过重复的 generic shape/dtype/alias
  检查；
- 每次 call 仍扫描所有输入 finite 值；
- 任何形状、地址、nbytes 或输出 binding 变化都会走完整 validation，成功后
  刷新 cache；失败则本次不发布；cache 不会被错误复用；
- 没有新增公共 ABI、结构体字段或 Python API；flag 在 workspace 创建时快照，
  默认关闭；guard、sync、pack、emitter、command、executor、download 和
  publication 都保持原实现。

这个实验刻意**不**优化 finite 扫描，也不把 validation、guard、publication
三项一起改。

## 2. 协议

- 固定 RK3588 条件：CPU6 `performance` 2.304 GHz，NPU `performance` 1.0 GHz，
  DMC `performance` 2.112 GHz。
- 先做 QK/PV b8；再做完整 Tiny ViT b8；b8 有收益后才补完整 ViT b1。
- 隔离案例：7 个 outer rounds；每个 round 用同一 changed-input 帧序列，
  `RK_PLAN_PREPARED_VALIDATION=0/1` 各一个进程；路径顺序按 round 轮换。
- 完整模型：同一个进程、同一 compiled graph，两个 runner；第一个 workspace
  创建前 env=0，第二个创建前 env=1，之后恢复 env。无 adapter、无指针数组
  重建、无节点代理。
- 主估计：**每轮 median 的 median/ratio 分布**；p10 仅辅助。
- 输出逐位比较；输入变化；错误 shape、non-finite、地址变化和失败不发布
  另做检查。

## 3. QK/PV 和同实现回归案例

µs，7 轮；ratio = env=0 median / env=1 median，>1 表示实验路径更快。

| case | env=0 | env=1 | per-round ratio median | ratio range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 884.9 | 871.5 | **1.0154×** | 0.9776–1.0178 | 0.9776× |
| PV group b8 | 721.9 | 709.0 | **1.0185×** | 0.9402–1.0206 | 0.9402× |
| MLP b1 | 436.0 | 436.3 | 0.9993× | 0.9960–1.0040 | 0.9960× |
| MLP b32 | 664.4 | 663.2 | 1.0000× | 0.9987–1.0071 | 0.9987× |
| QKV b8 | 184.0 | 184.0 | 0.9984× | 0.9920–1.0032 | 0.9920× |

QK/PV 有约 13–14 µs 的小幅、方向一致的收益；MLP/QKV 没有。

独立 `battr` sample：

| case | arm | validate µs | pack µs | command µs | submit/wait µs | output-sync µs | download µs | guard µs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QK | env=0 | 422.9 | 40.0 | 25.4 | 61.8 | 98.6 | 117.2 | 25.4 |
| QK | env=1 | 409.6 | 39.4 | 25.1 | 98.0 | 98.3 | 116.7 | 25.4 |
| PV | env=0 | 236.5 | 45.2 | 25.1 | 61.0 | 98.6 | 116.7 | 25.1 |
| PV | env=1 | 223.1 | 46.1 | 25.4 | 97.3 | 106.7 | 124.7 | 26.2 |

关键发现：静态 metadata/alias 复用只省 **13–14 µs**；QK/PV 的
`validate` 仍有约 409/223 µs，因为 **per-call finite scan 保留**。
上一轮 B/C 的数百 µs 差值主要来自不同 finite-scan/pack 实现，而不是
binding metadata 层。

## 4. 完整模型

Tiny ViT b8/b1，256 张图，batch 8/1，7 轮；两个 runner 同一 compiled graph，
只差 workspace 创建时的 prepared-validation flag。

| model | current median µs | experimental median µs | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 4126.3 | 4077.9 | **1.0168×** | 1.0058× | yes |
| Tiny ViT b1 | 1552.5 | 1551.1 | **1.0010×** | 0.9999× | yes |

- b8 的 7 轮 ratio：约 1.0058–1.0188，median 1.0168；约 **1.7%** 端到端收益。
- b1 基本中性：median 1.0010×，p10 0.9999×。
- b1 是在 b8 收益成立后才补的；没有反过来扩展更多形状。
- b8/b1 输出 logits 全部逐位一致。

## 5. 安全检查

对 QK/PV 各执行：

- 错误静态 shape：拒绝，输出不变；
- 同 pointer 下 non-finite 输入：拒绝，输出不变；
- 输入地址变化：完整重验证，cache 刷新后继续；
- 恢复正常 binding：输出恢复并和 baseline 逐位一致。

QK/PV 检查均为 **PASS**。这证明 cache 不会跳过可变 finite 检查，也不会在
失败路径发布部分输出。

## 6. 结论

1. **静态 validation 复用确实存在，但很小**：QK/PV 约 1.5–1.8%，完整 ViT
   b8 约 1.7%，b1 约 0.1%。
2. **速度瓶颈不是 metadata/alias 检查**：runtime `validate` 的主要部分是
   per-call finite scan；本实验保留了它。
3. 上一轮“QK/PV RKPlan 通用层 367/206 µs”不能作为静态验证复用收益；
   那部分是不同 pack/finite-scan 实现和 C direct 路径差异。
4. 完整模型 b8 有小的真实收益，但不满足当前产品化门槛：
   - b8 约 1.7%，b1 约 0.1%，形状依赖明显；
   - 需要长期 cache/lifetime/失效证明和新的安全契约；
   - 本轮不允许顺手改 guard/sync/publication；
   - 现有默认路径保持“每 call 全量检查”的语义。

**决定：不将静态 validation 复用作为默认产品路径。**
实验 flag 保留为默认关闭的隔离设施/证据，不新增公共 ABI，不改变默认策略。
如果未来还要继续，先做 finite-scan 等价实现和完整模型收益的独立实验，不应
把 guard、sync、publication 和 static validation 混在一起。

## 7. 身份

- final experimental runtime：`librknpu_plan.so` SHA-256
  `18dac5981ee05ade1d7d45108dc4bfcc8fbcc0d19187d9b471c56923c27fdc4d`
- default-off behavior：env 未设置时 `prepared_validation=0`
- experimental probe：`runtime_layer_probe_prepared` SHA-256
  `90ac208b6b37aca72ff2c11cccebe0ae80293d3c5d00a1c3b6a5b996e990c58a`
- compiled package still `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec still `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`
- evidence：`results/20261027T093259Z-template-gap/runtime-layers-prepared/`
