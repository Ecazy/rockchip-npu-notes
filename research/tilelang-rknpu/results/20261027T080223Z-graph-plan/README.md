# Tiny ViT graph-execution campaign

Date: 2026-10-27 (UTC).  This is the focused full-test campaign for the compiled
flat graph executor and the independent-policy device-FFN comparison.

Implementation source: `a37ca6e`; C ABI/custom-ops source remains at its last
change `60f4a82`.  Exact tool, library, data and policy hashes are in
`campaign-manifest.json`.

## Measurement

- board: RK3588 OPi 5 Plus
- kernel/driver: `6.1.43-rockchip-rk3588-rknpu098` / `v0.9.8`
- NPU: one core, observed 1.0 GHz; not re-clocked by the campaign
- CPU: affinity 6, one application thread, OMP/OpenBLAS/MKL=1
- CPU governor: **performance**, observed 2.304 GHz; restored after runs
- images: **10,000** MNIST test images for every profile
- batches: `8` and `1`
- rounds: 3 for same-policy arms and the independent device-FFN candidate;
  2 for vendor/CPU references
- application interval: `runner.run(..., stats=None)` —
  raw input preparation + model execution + output materialisation
- separate instrumentation: 100 calls through `RunStats`, excluded from app time
- argmax/evaluation: outside the same-policy application interval; identical
  post-processing for all custom arms
- arm order: round-robin per static batch with rotation each batch/round

## Same-policy result

| profile | generic ms | prepared loop ms | auto compiled graph ms | auto/generic | auto/prepared |
|---|---:|---:|---:|---:|---:|
| Tiny ViT b8 | 9.219 | 6.371 | **6.072** | **1.518x** | **1.049x** |
| Tiny ViT b1 | 2.875 | 2.232 | **2.088** | **1.377x** | **1.069x** |

Selected `auto` behavior:

| profile | selected residency | backend | tasks/submits | view/skip | setup ms | resident B |
|---|---|---|---:|---:|---:|---:|
| Tiny ViT b8 | prepared | compiled_graph | 70/8 | 1/2 | 21.5 | 505,664 |
| Tiny ViT b1 | prepared | compiled_graph | 14/8 | 1/2 | 9.6 | 63,208 |

`view/skip` means one metadata reshape lowered to a stable view and two
trace-only `gate/up` output writes skipped in the production path.  Submits,
tasks, upload/download bytes and NPU operations are otherwise unchanged.

Instrumented samples (not application timing):

| profile | generic | prepared loop | auto graph |
|---|---:|---:|---:|
| Tiny ViT b8 | 9.471 ms | 6.617 ms | **6.375 ms** |
| Tiny ViT b1 | 3.242 ms | 2.543 ms | **2.425 ms** |

All same-policy arms produce bit-identical 10k logits and captured traces.
`verify-auto-summary.json` is **PASS / 0 problems** against the unchanged frozen
packed numeric spec.

## External references (separate numeric policies)

| profile | auto graph ms | vendor 2.3.2 ms | CPU FP32 ms | vendor/auto | CPU/auto | vendor max logit err |
|---|---:|---:|---:|---:|---:|---:|
| Tiny ViT b8 | 6.072 | **2.907** | 5.847 | **2.089x** | **1.038x** | 0.12277 |
| Tiny ViT b1 | 2.088 | **0.795** | 0.869 | **2.627x** | **2.402x** | 0.10715 |

Vendor non-quantized FP16 does not satisfy the packed 0.0625 per-logit tolerance
on ViT; CPU is FP32 reference math.  These are references, not a same-policy
ranking.  Vendor setup was about 62.9/61.1 ms, CPU setup 0; auto setup is above.
The vendor Host `rknn_run` count does not expose internal device Submit count.

## Independent numeric-policy candidate (not default)

`deviceffn/benchmark.json` compares same-policy auto against the opt-in
`_deviceffn_qkv` graph, which uses the existing independent device-FFN numeric
policy (`act/ffn/h2/h2_tokens` atol=0.125).

| profile | same-policy auto ms | device FFN ms | device/auto | max Δlogit | tasks/submits | upload/download B |
|---|---:|---:|---:|---:|---|---|
| ViT b8 | 5.962 | 5.709 | **0.958x (4.2% faster)** | 0.03125 | 70/8 → 75/12 | 296448/230144 → 280064/164608 |
| ViT b1 | 1.923 | 2.059 | **1.071x (7.1% slower)** | 0.03125 | 14/8 → 19/12 | 38400/29440 → 36352/21248 |

It removes Host SwiGLU/residual work and bridge bytes but adds four submits in
both profiles; b8 improves and b1 regresses.  It is **not adopted** because a
uniform default would require an independent numeric policy decision.

## Host-boundary ledger

`boundaries/*.json` and `.md` classify every Host action by:

- `numeric_semantics`
- `missing_device_operator`
- `physical_layout`
- `resource_interface`
- `current_implementation`

and record measured Host dwell, only provably removable work, required
new-cost caveats, trace-only bytes and missing device recipes.  The b8 ledger
has 1.876 ms total Host dwell; only the 0.089 ms reshape is directly removed by
the current same-policy prototype.  The remainder is not claimed as recoverable.

## File index

- `benchmark.json` / `benchmark.log`: same-policy full-test campaign
- `analysis.json` / `analysis.md`: combined same-policy, external, boundary and
  device-FFN analysis (`PASS / 0 problems`)
- `benchmark.vit_b*.r*.*.logits.npy`: selected same-policy raw logits
- `deviceffn/benchmark.json`: full-test independent-policy comparison
- `deviceffn/manifests/`: exact opt-in hybrid manifests and hashes
- `external/r{1,2}-{vendor,cpu}/`: raw external reports, logs and wall times
- `boundaries/`: per-profile Host-boundary ledgers. Tiny ViT timings use this
  campaign; the unchanged MLP boundary timings come from the preceding
  same-policy campaign and are identified by its `analysis.json`.
- `numeric-auto*/`: fresh frozen-policy observation files
- `verify-auto-summary.json` / `verify-auto.log`: frozen numeric verifier PASS
- `campaign-manifest.json`: board, source, tool, data, policy and external IDs

## Replay

Set and restore the CPU governor around the measurement:

```bash
old=$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor)
echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
trap 'echo "$old" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor' EXIT
```

Same-policy arms (`plans-dir` and `manifest-dir` are the frozen compiled package;
the board build is the C library under test):

```bash
cd research/tilelang-rknpu
PYTHONPATH=$PWD/python RKNPU_PLAN_LIB=/path/to/librknpu_plan.so \
taskset -c 6 python3 tools/benchmark_graph_execution.py \
  --backend board --lib /path/to/librknpu_plan.so \
  --plans-dir models/compiled --manifest-dir models/compiled \
  --weights-dir models --data-dir .data/mnist \
  --limit 10000 --rounds 3 --stats-calls 100 \
  --batch 8 --batch 1 \
  --arm generic=generic --arm prepared=prepared --arm auto=auto \
  --out /tmp/graph-benchmark.json
```

The result analyzer:

```bash
PYTHONPATH=python:. python3 tools/analyze_graph_execution.py \
  --benchmark /tmp/graph-benchmark.json \
  --external results/20261027T080223Z-graph-plan/external \
  --boundaries results/20261027T080223Z-graph-plan/boundaries \
  --independent-benchmark results/20261027T080223Z-graph-plan/deviceffn/benchmark.json \
  --out /tmp/graph-analysis.json --markdown /tmp/graph-analysis.md
```

The device-FFN candidate manifests were generated with the opt-in builder flag:

```bash
PYTHONPATH=python python3 tools/build_hybrid_manifests.py \
  --plans-dir /tmp/devicefn-plans --weights-dir models \
  --include-deviceffn-qkv
```

Full external compile/runtime details and hashes are in `campaign-manifest.json`
and the existing `vendor/README.md`.

## Recorded hashes

- campaign manifest: `86f161c144a933720d302996136c4dec5c28a3cff6e3d0688889835f9c73052d`
- same-policy benchmark: `60b88d38136c553a62beca7e95db5a9b4b12fbad7bec0a7a28d0cf5805e33c80`
- combined analysis: `91477363874ecef65bca63d9e48141f1e8e50c6cf0df2dcc5de55a1523fad92b`
- analysis Markdown: `843f734ad4e41550b6b7ef00ddd8a5bd956518216663e1a532898ef82a272df7`
- frozen numeric verifier: `9cd83510b859570aaf2c3001f70fafb7a7981e26a8305b822e7f3c26fc5edfb4`
- independent device-FFN benchmark: `0f1166a2a429fca928b21992ec2c5872603f6f6bc8b35d588fd5ca9bcdf38b3f`
