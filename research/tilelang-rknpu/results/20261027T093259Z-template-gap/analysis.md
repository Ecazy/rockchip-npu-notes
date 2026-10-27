# Hardware-template and execution-granularity findings

Fixed-frequency campaign: CPU governor `performance`, CPU6 2.304 GHz;
NPU governor `performance`, 1.0 GHz; DMC governor `performance`,
2.112 GHz; one NPU core. Application timing uses the prepared
(`run_prepared`/`run_group_prepared`) path with reused workspace and
output; constant materialization and workspace setup are separate.

## Selected-case envelope

`submit+wait` is the C blocking submit interval, not a pure kernel
timer. `C non-submit` is `total_us - submit_wait_us`; `binding delta`
is application wall minus attributed C total (signed; a small negative
value is clock/probe noise). Physical MACs use the compiled
CUBE/WEIGHT padding.

| case | tasks/submits | app µs | submit+wait µs | share | C non-submit µs | validate µs | pack µs | command µs | output-sync µs | guard µs | download µs | binding delta µs | physical MMAC | physical GMAC/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp/b1 | 11/8 | 440.3 | 176.7 | 40.1% | 252.8 | 2.4 | 11.1 | 28.6 | 25.9 | 159.6 | 2.4 | +10.7 | 3.588 | 20.31 |
| mlp/b8 | 11/8 | 491.9 | 178.7 | 36.3% | 298.8 | 17.3 | 42.8 | 28.6 | 26.1 | 159.2 | 2.7 | +14.3 | 3.588 | 20.08 |
| mlp/b32 | 11/8 | 673.2 | 210.6 | 31.3% | 472.8 | 68.3 | 157.0 | 28.0 | 30.1 | 163.4 | 3.8 | -10.2 | 9.568 | 45.44 |
| qkv/b1 | 1/1 | 112.3 | 54.0 | 48.1% | 46.5 | 2.3 | 8.4 | 2.0 | 2.9 | 22.6 | 5.5 | +11.7 | 0.197 | 3.64 |
| qkv/b8 | 1/1 | 234.6 | 87.3 | 37.2% | 133.7 | 17.2 | 43.6 | 2.0 | 4.9 | 26.3 | 35.2 | +13.6 | 1.573 | 18.01 |
| gate_up/b1 | 1/1 | 96.8 | 55.8 | 57.7% | 47.4 | 2.3 | 8.4 | 2.0 | 3.1 | 23.3 | 6.4 | -6.4 | 0.262 | 4.69 |
| gate_up/b8 | 1/1 | 254.9 | 99.6 | 39.1% | 142.4 | 17.3 | 43.6 | 2.0 | 5.3 | 27.3 | 40.6 | +13.0 | 2.097 | 21.06 |
| down/b1 | 1/1 | 95.2 | 51.2 | 53.7% | 51.3 | 4.5 | 13.5 | 2.0 | 2.8 | 22.5 | 3.5 | -7.2 | 0.131 | 2.56 |
| down/b8 | 1/1 | 221.1 | 63.7 | 28.8% | 161.0 | 34.2 | 81.6 | 2.0 | 3.5 | 24.4 | 12.4 | -3.7 | 1.049 | 16.46 |
| classifier/b1 | 1/1 | 58.3 | 31.4 | 53.9% | 34.7 | 0.3 | 3.9 | 1.9 | 2.7 | 21.6 | 2.5 | -7.8 | 0.025 | 0.78 |
| classifier/b8 | 1/1 | 99.7 | 49.2 | 49.3% | 38.1 | 1.3 | 6.1 | 2.0 | 2.8 | 21.6 | 2.7 | +12.5 | 0.025 | 0.50 |
| qk_qkv_group/b1 | 4/1 | 164.6 | 53.1 | 32.3% | 115.9 | 51.7 | 10.0 | 4.5 | 9.4 | 21.3 | 11.7 | -4.3 | 0.066 | 1.23 |
| qk_qkv_group/b8 | 32/1 | 920.5 | 98.0 | 10.7% | 804.2 | 424.1 | 42.1 | 27.3 | 104.1 | 26.3 | 122.9 | +18.3 | 0.524 | 5.35 |
| pv_qkv_group/b1 | 4/1 | 157.5 | 54.0 | 34.3% | 93.2 | 28.3 | 10.8 | 4.5 | 9.6 | 21.1 | 11.9 | +10.2 | 0.066 | 1.21 |
| pv_qkv_group/b8 | 32/1 | 723.9 | 96.6 | 13.4% | 623.8 | 237.8 | 46.8 | 26.5 | 104.0 | 25.9 | 122.6 | +3.5 | 0.524 | 5.42 |

Run-to-run submit/wait variation is retained in `analysis.json` under
`envelope_ranges`; b1-only cases are most sensitive because their
device interval is close to the scheduler noise floor. The full-model
paired measurements, not single isolated points, are used for the
product decision.

## MLP device-resident control

| profile | tasks/submits | logical MMAC | physical MMAC | pad | submit+wait µs | submit share | guard µs | pack µs | app µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b1 | 11/8 | 0.297 | 3.588 | 12.083× | 176.7 | 40.1% | 159.6 | 11.1 | 440.3 |
| b8 | 11/8 | 2.376 | 3.588 | 1.510× | 178.7 | 36.3% | 159.2 | 42.8 | 491.9 |
| b32 | 11/8 | 9.503 | 9.568 | 1.007× | 210.6 | 31.3% | 163.4 | 157.0 | 673.2 |

MLP b1 and b8 have the same physical CUBE task size (12 padded rows)
and the same 11 tasks / 8 submits, so b1's 1-row GEMM pays the b8
physical task cost. b32 removes most padding and raises modeled
physical throughput, but its larger upload/output conversion also
raises the boundary share. This is an execution-granularity finding,
not evidence that a different GEMM register recipe wins.

## Implemented same-policy improvement: one packed K-major input upload

The compiler now packs an external rank-2 CUBE value once per 256-row
tile and exposes its 256-column slices as stable views. The C verifier
proves the upload's canonical column offsets cover each reader; no
device template, arithmetic, numeric policy, or submit/task count
changes.

| profile | old plan µs | new plan µs | upload-only plan gain | upload-only full gain | old delivered µs | new delivered µs | delivered gain | bitwise | slots | transfers |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| b1 | 509.2 | 478.0 | 1.065× | 1.033× | 627.1 | 599.1 | 1.045× | yes | 22→19 | 19→16 |
| b8 | 573.7 | 540.5 | 1.062× | 1.037× | 746.8 | 716.6 | 1.040× | yes | 22→19 | 19→16 |
| b32 | 806.2 | 774.4 | 1.041× | 1.024× | 1168.5 | 1127.6 | 1.034× | yes | 22→19 | 19→16 |

Resource effect: upload actions `4→1`, slots `22→19`, transfers
`19→16`, reserved bytes `1,551,488→1,452,992` for b1/b8
(`1,660,288→1,561,792` for b32); peak payload and upload bytes are
unchanged. Full-test logits are bitwise equal. The steady-state gain
is from fewer upload prep/fini calls and publication-gate slots, not
from a matrix instruction change.

## Setup, constant prepacking and output refresh

`app` reuses prepared bindings and output arrays. `fresh` allocates
row-major outputs per call. `cold once` includes Host constant
materialization, plan/workspace setup, one-time binding preparation
and one application call; it is not the steady-state latency.

| case | const KB | const mat µs | plan setup ms | bind prep µs | app µs | fresh µs | cold once ms | amortized/1000 µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp/b1 | 743.5 | 1810.1 | 5.306 | 83.4 | 440.3 | 783.0 | 7.408 | 447.2 |
| mlp/b8 | 754.2 | 1806.0 | 5.361 | 66.4 | 491.9 | 551.8 | 7.552 | 498.8 |
| mlp/b32 | 791.0 | 1781.9 | 5.559 | 65.8 | 673.2 | 1081.0 | 7.903 | 680.4 |
| qkv/b1 | 24.0 | 137.7 | 0.523 | 38.4 | 112.3 | 154.1 | 0.768 | 113.0 |
| qkv/b8 | 24.0 | 145.8 | 0.587 | 39.4 | 234.6 | 292.0 | 0.945 | 235.4 |
| gate_up/b1 | 32.0 | 196.0 | 0.575 | 39.4 | 96.8 | 160.1 | 0.856 | 97.6 |
| gate_up/b8 | 32.0 | 207.2 | 0.628 | 40.4 | 254.9 | 313.6 | 1.072 | 255.7 |
| down/b1 | 16.0 | 117.8 | 0.494 | 38.5 | 95.2 | 159.8 | 0.699 | 95.8 |
| down/b8 | 16.0 | 127.5 | 0.531 | 39.4 | 221.1 | 312.5 | 0.860 | 221.7 |
| classifier/b1 | 2.0 | 144.8 | 0.429 | 37.5 | 58.3 | 137.2 | 0.635 | 58.9 |
| classifier/b8 | 2.0 | 162.5 | 0.473 | 40.7 | 99.7 | 141.3 | 0.707 | 100.4 |
| qk_qkv_group/b1 | 0.0 | 2.0 | 0.731 | 188.0 | 164.6 | 381.2 | 1.031 | 165.5 |
| qk_qkv_group/b8 | 0.0 | 2.3 | 4.227 | 975.2 | 920.5 | 2287.7 | 6.003 | 925.6 |
| pv_qkv_group/b1 | 0.0 | 2.0 | 0.768 | 194.4 | 157.5 | 336.9 | 1.055 | 158.4 |
| pv_qkv_group/b8 | 0.0 | 2.3 | 4.268 | 968.6 | 723.9 | 1918.0 | 5.810 | 729.0 |

The QK/PV expanded group has a large one-time binding/DMA preparation
charge; it is amortized only when the same workspace is reused. The
steady-state group call is still dominated by per-call validation and
download staging, not by the 32 tiny MM tasks.

## Runtime CUBE→row-major download fast path

The C download path now copies each canonical eight-FP16 atom with one
16-byte memcpy instead of evaluating `co_index` per element. Logical
rows/cols are unchanged; padding is never copied. The full ViT model
pair uses the same final package and differs only in this runtime
library change.

| profile | before µs | after µs | paired gain | bitwise |
|---|---:|---:|---:|---|
| tiny_vit b8 | 4512.9 | 4010.7 | 1.134× | yes |
| tiny_vit b1 | 1593.4 | 1532.4 | 1.041× | yes |

| case | download before µs | download after µs | saved µs |
|---|---:|---:|---:|
| mlp/b1 | 2.5 | 2.4 | 0.0 |
| mlp/b8 | 3.3 | 2.7 | 0.7 |
| mlp/b32 | 6.6 | 3.8 | 2.8 |
| qkv/b1 | 18.1 | 5.5 | 12.6 |
| qkv/b8 | 164.0 | 35.2 | 128.8 |
| gate_up/b1 | 22.7 | 6.4 | 16.3 |
| gate_up/b8 | 198.5 | 40.6 | 157.9 |
| down/b1 | 8.3 | 3.5 | 4.8 |
| down/b8 | 52.8 | 12.4 | 40.4 |
| classifier/b1 | 2.6 | 2.5 | 0.2 |
| classifier/b8 | 3.4 | 2.7 | 0.7 |
| qk_qkv_group/b1 | 18.0 | 11.7 | 6.4 |
| qk_qkv_group/b8 | 168.0 | 122.9 | 45.1 |
| pv_qkv_group/b1 | 18.0 | 11.9 | 6.1 |
| pv_qkv_group/b8 | 153.6 | 122.6 | 30.9 |

MLP has only one tiny final download, so its fast-download delta is
noise; the upload-coalescing gain is the relevant MLP result. The ViT
projection/FFN plans and the QK/PV groups carry large output
conversions and benefit materially.

## External references under the same fixed frequencies

| profile | final delivered µs | vendor µs | CPU FP32 µs | vendor/final | final/CPU | vendor max logit error |
|---|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b1 | 599.1 | 258.8 | 252.6 | 0.432× | 2.372× | 0.03049 |
| mnist_mlp.b8 | 716.6 | 309.9 | 1748.9 | 0.432× | 0.410× | 0.03049 |
| mnist_mlp.b32 | 1127.6 | 538.0 | 6830.5 | 0.477× | 0.165× | 0.03049 |
| tiny_vit.b8 | 4010.7 | 1582.1 | 5856.7 | 0.394× | 0.685× | 0.12277 |
| tiny_vit.b1 | 1532.4 | 589.4 | 869.5 | 0.385× | 1.762× | 0.10715 |

Vendor is non-quantized FP16 with graph fusion; ViT fails the packed
per-logit tolerance and MLP is a different numeric policy. CPU is
FP32. Neither is a same-policy ranking. In `vendor/final`, a value <1
means vendor is faster; in `final/CPU`, <1 means the final path is
faster. The vendor one-call runtime API
does not expose its internal device Submit count or per-kernel time;
its layer table is placement/ranking evidence only.

## CPU FP32 same-GEMM reference (different numeric policy)

| case | custom prepared app µs | NumPy FP32 ms | max abs delta |
|---|---:|---:|---:|
| qkv/b1 | 112.3 | 0.1762 | 0.00013 |
| qkv/b8 | 234.6 | 1.3942 | 0.00019 |
| gate_up/b1 | 96.8 | 0.2335 | 0.00024 |
| gate_up/b8 | 254.9 | 1.8538 | 0.00024 |
| down/b1 | 95.2 | 0.1210 | 0.00023 |
| down/b8 | 221.1 | 0.9539 | 0.00024 |
| classifier/b1 | 58.3 | 0.0029 | 0.00006 |
| classifier/b8 | 99.7 | 0.0102 | 0.00012 |

The CPU reference is FP32 and the custom path is packed FP16; the two
columns are not a same-policy ranking. For the smallest classifier
tile the plan envelope makes the NPU path slower than the CPU
reference; for the larger projection/FFN tiles the custom path is
faster despite its host envelope.

## Decision and next capability investment

The selected implementable improvement is the **exact CUBE layout
bridge packet**: one canonical K-major external upload with slice
views, plus atom-wise CUBE→row-major download. Both changes preserve
the arithmetic and numeric policy and do not add a hardware recipe or
a new graph-level interface. They are now part of the default path.

The residual split shows where not to spend the next cycle. For the
MLP device-resident control, small-M physical padding and the 8-submit
envelope dominate b1/b8; for ViT projections, pack/guard/validation/
download boundaries are comparable to or larger than submit+wait; for
the QK/PV groups, per-call binding validation and download staging
dominate the 32 tiny MM tasks. A new GEMM register recipe alone is not
supported by this evidence as the next highest-value investment.

The next capability to evaluate is a narrow **workspace-level
validation/guard and publication batching contract**: prove static
binding/alias/lifetime facts once per workspace, validate only mutable
pointers/finite values per call, and reduce the 32 KiB slack scan /
per-slot prep-fini cost while preserving explicit failure isolation.
That requires a public safety-contract decision and a new validation
proof; it is not implemented in this stage.

## Attribution limits

- `submit+wait` includes both the submit API and device execution.
  No current probe exposes per-task hardware issue latency.
- Kernel register content is fixed by `gen_matmul_fp16`; RKPlan has no
  alternative register budget, M/N/K tile choice beyond the fixed 256
  chain, or bank-reorder recipe.
- The earlier vendor-perf probe observed MLP b8 as 7 NPU + 3 CPU
  boundary layers and ViT b8 as 50 NPU + 5 CPU boundary layers. Its
  per-layer times use `RKNN_FLAG_COLLECT_PERF_MASK`, so they are
  placement/ranking evidence only; vendor internal Submit count and
  per-kernel time remain unknown.
- No numeric-policy or hardware-template change is claimed by the
  implemented upload coalescing or download fast path.
