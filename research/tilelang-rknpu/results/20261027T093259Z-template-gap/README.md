# Hardware-template and execution-granularity campaign

Date: 2026-10-27 (UTC). This campaign measures representative GEMM/attention
shapes from the frozen MLP and Tiny ViT plans, separates the device submit+wait
interval from C/Host layout and publication costs, and tests two exact CUBE
layout fast paths.

## Identity and fixed conditions

Implementation source: `16d49ad`.

| item | value |
|---|---|
| board | RK3588 OPi 5 Plus |
| kernel / driver | `6.1.43-rockchip-rk3588-rknpu098` / `v0.9.8` |
| NPU | one core, governor `performance`, 1.0 GHz |
| DMC | governor `performance`, 2.112 GHz |
| CPU | affinity 6, governor `performance`, 2.304 GHz |
| final runtime | `runtime/librknpu_plan.so`, SHA-256 `19b5263a137cda82eff215ca181d267923ca9645d4cca2378de290ab6dbcaa16` |
| compiled package | `models/compiled/manifest.json`, SHA-256 `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869` |
| numeric spec | `models/numeric_acceptance_packed.json`, SHA-256 `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2` |
| weights | `mnist_mlp.npz` / `tiny_vit.npz` from `models/` |

All timing arms use the prepared single/group binding path, a reused workspace
and reused outputs. Constant materialization, C workspace setup, one-time
binding preparation and fresh-output allocation are separate fields.
`submit+wait` is a C blocking submit interval: it includes submission and device
execution, not a pure kernel timer. `RunStats` is not used in the application
medians; attributed envelope collection is a separate pass.

## Main decomposition

Selected-case values are medians over six fixed-frequency runs (three forward
and three reverse case orders); `analysis.json` retains min/max ranges.

| case | tasks/submits | app µs | submit+wait share | guard µs | pack µs | download µs | validate µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 11/8 | 440.3 | 40.1% | 159.6 | 11.1 | 2.4 | 2.4 |
| MLP b8 | 11/8 | 491.9 | 36.3% | 159.2 | 42.8 | 2.7 | 17.3 |
| MLP b32 | 11/8 | 673.2 | 31.3% | 163.4 | 157.0 | 3.8 | 68.3 |
| QKV b8 | 1/1 | 234.6 | 37.2% | 26.3 | 43.6 | 35.2 | 17.2 |
| gate-up b8 | 1/1 | 254.9 | 39.1% | 27.3 | 43.6 | 40.6 | 17.3 |
| down b8 | 1/1 | 221.1 | 28.8% | 24.4 | 81.6 | 12.4 | 34.2 |
| QK group b8 | 32/1 | 920.5 | 10.7% | 26.3 | 42.1 | 122.9 | 424.1 |
| PV group b8 | 32/1 | 723.9 | 13.4% | 25.9 | 46.8 | 122.6 | 237.8 |

MLP b1 and b8 have the same physical task size: logical 0.297/2.376 MMAC but
**3.588 MMAC physical** and the same 11 tasks / 8 submits. The b1 path pays the
12-row physical CUBE floor. MLP b1/b8 guard alone is ~159 µs. QK/PV shared-group
case has only ~98 µs submit+wait for 32 tiny 16³ tasks; per-call validation and
download staging dominate that case. This is why the next investment is not a
blind new GEMM register recipe.

## Implemented layout bridge

Two exact changes, no numeric-policy, hardware-recipe or graph-interface change:

1. **One K-major external CUBE upload.** The compiler allocates one canonical
   packed buffer per 256-row tile and exposes 256-column K slices as stable
   views. The C validator proves canonical offset/layout coverage. For MLP:
   upload actions `4→1`, slots `22→19`, transfers `19→16`, reserved bytes
   `1,551,488→1,452,992` (b32 `1,660,288→1,561,792`); peak/upload/constant
   bytes unchanged.
2. **CUBE→row-major atom download.** The runtime copies each eight-FP16 CUBE
   atom in one 16-byte `memcpy`, preserving logical rows/columns and never
   copying padding. It reduces e.g. QKV b8 download 164.0→35.2 µs and gate-up b8
   198.5→40.6 µs.

Same-policy results:

| profile | old plan µs | new plan µs | plan gain | final delivered old µs | final delivered new µs | delivered gain |
|---|---:|---:|---:|---:|---:|---:|
| MLP b1 | 509.2 | 478.0 | 1.065× | 627.1 | 599.1 | 1.045× |
| MLP b8 | 573.7 | 540.5 | 1.062× | 746.8 | 716.6 | 1.040× |
| MLP b32 | 806.2 | 774.4 | 1.041× | 1168.5 | 1127.6 | 1.034× |
| Tiny ViT b8 | — | — | — | 4512.9 | 4010.7 | 1.134× |
| Tiny ViT b1 | — | — | — | 1593.4 | 1532.4 | 1.041× |

MLP `plan` values use alternating same-process old/new plan calls. MLP `final
delivered` uses old package + old runtime vs new package + final runtime. Tiny
ViT uses the same package with old/new runtime only. All listed comparisons
produce bitwise-equal full-test logits.

## Setup / prepacking / conversion

`analysis.md` records Host constant materialization, C plan/workspace creation,
prepared-binding setup, reused-workspace application, fresh output allocation,
cold-once and 1,000-call amortized cost. MLP constants are 743–791 KiB:
materialization ~1.78–1.81 ms, workspace setup ~5.3–5.6 ms, cold once
~7.4–7.9 ms, steady calls 0.44–0.67 ms. QK/PV b8 has ~0.97 ms one-time binding
preparation and ~4.2 ms workspace setup; steady group calls are 0.72–0.92 ms.
Do not compare cold setup against another implementation's warm call.

## CPU and external references

The isolated CPU column is NumPy FP32 on the same GEMM shapes; it is a different
numeric policy and is not ranked with packed FP16. The full external run used
the same fixed CPU/DMC/NPU frequencies:

| profile | final delivered µs | vendor µs | CPU FP32 µs | vendor max logit error |
|---|---:|---:|---:|---:|
| MLP b1 | 599.1 | 258.8 | 252.6 | 0.03049 |
| MLP b8 | 716.6 | 309.9 | 1748.9 | 0.03049 |
| MLP b32 | 1127.6 | 538.0 | 6830.5 | 0.03049 |
| ViT b8 | 4010.7 | 1582.1 | 5856.7 | 0.12277 |
| ViT b1 | 1532.4 | 589.4 | 869.5 | 0.10715 |

Vendor is non-quantized FP16 graph fusion; Tiny ViT exceeds the packed 0.0625
per-logit tolerance. CPU is FP32. The vendor one-call runtime API does not expose
internal device Submit count or per-kernel time. The earlier vendor-perf probe
only provides placement/ranking (MLP b8: 7 NPU + 3 CPU boundary layers; ViT b8:
50 NPU + 5 CPU), not a warm per-layer timer.

## Numeric and regressions

- Frozen packed spec: **PASS / 0 problems** (`verify-numeric-summary.json`).
- MLP b1/b8/b32 full-test delivered comparison: bitwise equal.
- Tiny ViT b8/b1 full-test fast-download comparison: bitwise equal.
- Host: **348 passed, 3 skipped, 68 subtests**; CTest: **31/31**.
- Final C library includes the new `download_fastpath_test` exactness regression.

## File index

- `geometry.md` / `geometry.json`: representative geometry, source tiling,
  physical padding, tasks/submits.
- `geometry-baseline.*`: previous MLP plan resources before upload coalescing.
- `analysis.md` / `analysis.json`: envelope decomposition, setup/cold costs,
  CPU reference, fast-download deltas, external context and decision.
- `timing/benchmark-*.json`: same-policy and fast-download template campaign
  arms (three forward + three reverse where applicable).
- `timing/plan-pair-b*.txt`: same-process old/new MLP plan pairs.
- `timing/model-pair-b*.txt`: upload-coalescing-only full-model pairs.
- `timing/mlp-delivery-b*.txt`: final package/runtime full-model pairs.
- `timing/vit-fastdl-b*.txt`: same-package ViT runtime before/after pairs.
- `numeric/`: frozen calibration/acceptance observations, spec copy and
  verifier output.
- `external/`: fixed-frequency vendor/CPU full-test JSON and logs (raw logits
  intentionally not committed).
- `runtime/librknpu_plan.so`: exact final board runtime binary.
- `campaign-manifest.json`: hashes, board and replay identity.

## Replay

Set the fixed-frequency conditions around all arms:

```bash
for f in /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor \
         /sys/class/devfreq/fdab0000.npu/governor \
         /sys/class/devfreq/dmc/governor; do
  printf '%s\n' "$(cat "$f")" > "/tmp/$(basename "$f").old"
done
echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
echo performance > /sys/class/devfreq/fdab0000.npu/governor
echo performance > /sys/class/devfreq/dmc/governor
```

Representative plan benchmark:

```bash
PYTHONPATH=$PWD/python RKNPU_CORE_MASK=1 taskset -c 6 \
python3 tools/benchmark_gemm_templates.py \
  --plans-dir models/compiled --weights-dir models \
  --lib runtime/librknpu_plan.so --backend board \
  --runs 500 --warmup 20 --out /tmp/template.json
```

Geometry inventory:

```bash
PYTHONPATH=$PWD/python python3 tools/extract_gemm_geometry.py \
  --plans-dir models/compiled --out /tmp/geometry.json --markdown /tmp/geometry.md
```

Full campaign analysis:

```bash
PYTHONPATH=$PWD/python:$PWD python3 tools/analyze_template_gap.py \
  --result-dir results/20261027T093259Z-template-gap \
  --out /tmp/analysis.json --markdown /tmp/analysis.md
```

Frozen numeric replay:

```bash
PYTHONPATH=$PWD/python RKNPU_CORE_MASK=1 taskset -c 6 \
python3 tools/run_numeric_observation.py --mode calibration \
  --plans-dir models/compiled --weights-dir models --data-dir .data/mnist \
  --lib runtime/librknpu_plan.so --execution-mode auto \
  --out /tmp/numeric/calibration/observation.npz
PYTHONPATH=$PWD/python RKNPU_CORE_MASK=1 taskset -c 6 \
python3 tools/run_numeric_observation.py --mode acceptance \
  --plans-dir models/compiled --weights-dir models --data-dir .data/mnist \
  --spec models/numeric_acceptance_packed.json \
  --lib runtime/librknpu_plan.so --execution-mode auto \
  --out /tmp/numeric/acceptance/observation.npz
```

## Nonclaims

- No claim that `submit_wait_us` is pure matrix kernel time.
- No claim about vendor internal Submit count, per-kernel time, or fusion body.
- No claim that the small-M physical padding can be removed by current RKPlan.
- No claim that guard/cache sync can be removed without a new safety contract.
- The device-FFN path remains explicit/opt-in and was not auto-selected here.
