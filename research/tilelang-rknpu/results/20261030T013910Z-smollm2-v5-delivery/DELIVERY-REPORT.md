# SmolLM2 V5 交付收口与 Host 开销优化报告

本报告记录 V5 功能验收通过后的交付收口。V5 spec、输入、请求、门限和原始观测
均未改写；性能候选与 V5 基线分开记录。

## 1. 文档修正与自动重放入口

已修正的数值口径：

- `teacher forcing vs declared reference`：**76/76**，这是 V5 数值 gate；
- 对归档 official continuation 的质量诊断：**31/32，first divergence index 0**，
  仅作为历史报告，不参与数值 gate；
- incremental 的 17 个 logits 位置来自 **1 个初始 prefill 的最后一个 logits +
  16 次 decode**，不是 17 次 decode；
- 实际校准为 **6 条 prompt**，不是 V5 spec 冻结文本里遗留的 “three calibration
  prompts” 元数据笔误。V5 spec 文件/规范哈希保持不变，勘误记录在
  `POLICY-NOTE-ERRATUM.json`；未来 spec 生成器已改为使用实际 prompt 数。

可执行入口：

```bash
./scripts/replay-smollm2-v5.sh reuse                # 复用已封存观测
./scripts/replay-smollm2-v5.sh rerun-fixed-acceptance \
  --evidence-root ... --results ... \
  --calibration-root .data/evidence/smollm2-20261029T120416Z/calibration
./scripts/replay-smollm2-v5.sh campaign \
  --evidence-root ... --results ...
```

`reuse` 从 `models/specs/smollm2_spec_hashes.json` 独立记录读取预期 spec 哈希，
不再把现场计算出的同一文件哈希当作外部预期身份。`rerun-fixed-acceptance`
严格复用 V5 校准和冻结 spec，重跑 acceptance board/PPL、重建 declared/official
参考、封存并验证。`campaign` 的顺序是：输入准备与冻结 → 官方 continuation →
calibration board/reference/PPL → seal calibration → freeze spec → acceptance
board/reference/PPL → seal acceptance → verify；任一阶段失败立即非零退出。

## 2. 首次分叉诊断

诊断探针为 8 条 instruction prompt 各选一个相同前缀位置，记录 backend board、
declared reference 和 official FP32 的 top5/top1/top2/margin：

- 8/8 prompt 的同一前缀 top1 三者一致；
- 归档 official generation 与 backend 在 8 条 prompt 上合计匹配 102/256 token；
- 修正 official generation 的显式 all-ones attention mask 后，backend 对
  corrected official generation **256/256 token** 一致（每条 32/32）；
- 该修正不改变 V5 输入、门槛、封存观测或数值 gate；修正参考保存在独立交付目录。

根因：tokenizer 的 pad id 与 chat 模板分隔符/EOS 同为 id 2，归档工具调用
`model.generate()` 未显式传 attention mask，transformers 从
`input_ids != pad_token_id` 推导 mask，并用 `cumsum(mask)-1` 推导 position_ids，
从而把真实 prompt 中的 id 2 当成 padding 并重置 RoPE position。这是 official
generation 工具口径缺陷，不是 backend 矩阵实现偏差。

top1/top2 margin 也记录了原始数值；没有把所有分叉自动称为 near-tie。例如
acc-2 的 backend margin 为 `0.03125`，而 acc-1 为 `3.171875`，数值性质不同。

## 3. Decode 时间分层分解

profiled 3 次 32-token generate，每次实际 31 个 decode forwards；数值为
median ms/forward。

| 层级 | baseline | candidate |
|---|---:|---:|
| decode wall | 619.099 | 468.947 |
| runner embed | 125.407 | 0.172 |
| 30 layers total | 436.013 | 411.954 |
| final RMSNorm | 0.871 | 0.156 |
| lm_head plans total | 55.701 | 55.877 |
| logits concat | 0.272 | 0.305 |
| top-level measured residual | 0.137 | 0.121 |
| PlanCall wrapper total | 378.103 | 387.364 |
| C runtime reported total | 316.254 | 321.635 |
| Host actions (cross-cut) | 88.427 | 89.159 |

candidate 的层内分解：attention `141.973`，gate/up `100.847`，down `49.326`，
qkv `43.072`，o `32.974`，KV append `12.020`，RoPE `8.971`，norm 合计 `9.447`，
swiglu `4.567`，layer residual `2.427` ms/fwd。

PlanCall seam（candidate）：input preparation `32.779`，binding `6.127`，
lock `1.349`，output preparation `2.317`，inner unattributed `5.906`，
wrapper unattributed `9.381`，PlanCall wrapper 总计 `387.364`，C runtime wrapper `332.750` ms/fwd。C runtime 内部
命名区间中 submit wait `197.416`、guard `74.226`、pack `17.805`、download
`13.619`、output sync `10.196`、validate `2.674` ms/fwd；C children 合计
`317.968`，相对 C reported total 的残差为 `3.660` ms/fwd，Python/ctypes seam
为 `11.115` ms/fwd。这些 C 子区间只是父区间的细分，不能与 top-level 再相加。

## 4. 实际优化、内存代价与 A/B

实际采用的优化：runner 初始化时缓存不可变的 FP16 embedding 表和按 key 懒加载
的 norm gamma，避免每步从 NPZ 重新读取/构造 embedding 和重复读取 gamma。
布局、dtype、数学语义、finite/shape/identity/所有权检查全部保留。逻辑新增
常驻 `56,693,376 B`（embedding `56,623,104 B` + 61 个 gamma `70,272 B`）。

最终 A/B（profiler 关闭，同 request/package/board/core mask 1/FP16，3 组交错）：

| 指标 | baseline | candidate |
|---|---:|---:|
| TTFT median | 3,870.855 ms | 3,865.049 ms |
| generate wall median | 22,711.064 ms | 17,854.156 ms |
| decode forward | 608.435 ms/fwd | 455.652 ms/fwd |
| decode forward/s | 1.644 | 2.195 |
| full-generate token/s | 1.409 | 1.792 |
| peak RSS median | 1,237,348 kB | 1,243,836 kB |
| final vmRSS median | 1,182,176 kB | 1,233,236 kB |

generate-wall speedup `1.272x`，decode speedup `1.335x`，peak RSS 增加
`6,488 kB`，最终 vmRSS 增加 `51,060 kB`。逐元素等价：44/44 arrays 相等；
V5 acceptance board 重跑与归档 board 44/44 arrays 相等、19/19 request token
lists 相等；PPL NLL 5/5 arrays 相等。

## 5. V5 复验、测试与回归

- 固定 V5 验收重跑：`complete_acceptance=true`，`numeric_pass=true`，
  `quality_pass=true`，56/56 gate，`gate_failures=[]`，`failures=[]`，exit 0；
- `reuse` 重验：同样 56/56 gate PASS，且 spec 身份来自独立哈希记录；
- host：432 tests collected，431 passed、1 skipped、71 subtests passed；
- C：33/33 ctest PASS；
- board 回归：MLP b1/b8/b32、TinyViT b1/b8 各 8 图 smoke，accuracy 1.0，
  top1 agreement 1.0；
- 新增针对性测试：Host 权重缓存/计时 4/4、分叉诊断 2/2、自动重放身份/顺序 3/3。

## 6. 身份、证据与尚存限制

身份：

- 任务代码提交：`4c0112233dd5671caef49a87eceda3917b0195da`；交付报告/身份记录提交：`d5cfc3d8d6ac0866a49567ac9ab654b5ce9afb96`；最终本地元数据提交只新增 `DELIVERY-MANIFEST.json`。均未推送；无关 graph-fusion/int16-probe 工作树改动未进入这些提交；
- model：`HuggingFaceTB/SmolLM2-135M-Instruct`，revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`；
- package manifest SHA-256：`b923fc5b427cc2ee6f0f88e7d096444a7ab44babdfbd79cc64fe07f5771548be`；
- package weights SHA-256：`ff04cb6f25633772987ff8cfc66e65046004034e526cf1d6d6bb615333b945bf`；
- V5 spec file/canonical：`04c4011c7eb7daec4f4f607d65d234f8ec94ddba112a931fc2fd24056f70e38f` /
  `1a8c68d2da51eb9bc6885a4d540950f74011e8bcbf150a5d975d75a7575f368b`；
- plan library SHA-256：`f9ed96ae2f1c38a941ac8a0e379026d8ef6e67ae9dcd62e4aafc11143d5d8419`；
- 固定 V5 验收重跑 verify summary：`2c4165d8c48f339bb2a715670eec34eedad745fba38967a2cb1ec8d8f2f0672e`；
- 固定 V5 验收重跑 acceptance bundle：`37160a1920c35e841b863370b84f3319c3d3ae1688dfef85a760a168ece22111`。

证据路径：

- 本目录：`results/20261030T013910Z-smollm2-v5-delivery/`；
- `DELIVERY-MANIFEST.json`：最终代码/库/包/输入/证据文件哈希；
- `LATENCY-REPORT.md`、`latency.json`：分层计时和 A/B；
- `divergence/DIVERGENCE-REPORT.md`、`divergence/diagnostic.json`：分叉诊断；
- `equivalence.json`、`acceptance-equivalence.json`、
  `acceptance-ppl-equivalence.json`：逐元素等价和 V5 重跑等价；
- `rerun-fixed-acceptance/`：独立固定验收重跑 board/reference/PPL/verify；
- `.data/evidence/smollm2-20261029T120416Z/`：原始 V5 观测。

尚存限制：

- 单板、core mask 1、batch=1、capacity 256、FP16；Host KV 编排，不是通用
  device-resident KV API；
- QKV→QK 设备侧转换仍不支持，QK/PV B operand 由 runtime Host 提供；
- 未声明量化、多请求、多核、多板、长上下文、故障后恢复或 RK1228；
- corrected official generation 仅是本轮诊断参考，不作为 V5 gate，也不宣告
  任意 prompt/任意实现上的逐 token 等价。
