# Prepared-validation same-implementation A/B

Date: 2026-10-27. Source/tool commit: `ca6481b`.

This result corrects the previous causal interpretation. The earlier B/C
comparison used different upload-pack/finite-scan implementations, C−D
included experimental path differences, and the full-model direct adapter
rebuilt ctypes pointer arrays per call. None of those is used as a causal
estimate here.

Both arms now use exactly the same RKPlan workspace implementation, descriptors,
pack, emitter, command workspace, executor, guard, download and publication.
Only the workspace-snapshotted **experimental** flag
`RK_PLAN_PREPARED_VALIDATION` differs. The flag defaults off and adds no public
ABI or default policy change.

## What the flag does

- First run performs full binding validation and caches the exact input/output
  binding signatures (id, dtype, rank, shape, nbytes, data pointer).
- Later calls skip the repeated generic shape/dtype/alias checks only when every
  cached signature matches.
- Every call still scans all input finite values.
- Any address/shape/nbytes/output-binding change takes the full validation path
  and refreshes the cache only after success.
- A failed validation publishes nothing.
- Guard, synchronization, pack, emitter, command, download and publication are
  unchanged.

## Protocol

- RK3588 OPi 5 Plus; CPU6 `performance` 2.304 GHz; NPU `performance` 1.0 GHz;
  DMC `performance` 2.112 GHz.
- Isolated QK/PV and MLP/QKV: 7 rotated outer rounds, changed-input frames,
  separate `static=0` and `static=1` processes.
- Primary estimate: median of **per-round paired medians**; p10 is auxiliary.
- Full Tiny ViT b8/b1: same process, two compiled-graph runners; env is
  snapshotted before each runner's workspaces are created. No adapter, no
  pointer rebuilding, no node proxies.
- Output logits and raw case outputs are bitwise compared.

## Isolated results

µs; ratio = static=0 / static=1.

| case | static=0 | static=1 | per-round ratio median | ratio range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 884.9 | 871.5 | **1.0154×** | 0.9776–1.0178 | 0.9776× |
| PV group b8 | 721.9 | 709.0 | **1.0185×** | 0.9402–1.0206 | 0.9402× |
| MLP b1 | 436.0 | 436.3 | 0.9993× | 0.9960–1.0040 | 0.9960× |
| MLP b32 | 664.4 | 663.2 | 1.0000× | 0.9987–1.0071 | 0.9987× |
| QKV b8 | 184.0 | 184.0 | 0.9984× | 0.9920–1.0032 | 0.9920× |

QK/PV gain about 13–14 µs; MLP/QKV is neutral.

## Validation phase split

Independent `battr` samples (µs), same runtime path:

| case | arm | validate | pack | command | submit/wait | output-sync | download | guard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QK | static=0 | 422.9 | 40.0 | 25.4 | 61.8 | 98.6 | 117.2 | 25.4 |
| QK | static=1 | 409.6 | 39.4 | 25.1 | 98.0 | 98.3 | 116.7 | 25.4 |
| PV | static=0 | 236.5 | 45.2 | 25.1 | 61.0 | 98.6 | 116.7 | 25.1 |
| PV | static=1 | 223.1 | 46.1 | 25.4 | 97.3 | 106.7 | 124.7 | 26.2 |

The static cache removes only **13–14 µs** of metadata/alias work. QK/PV still
spend about 409/223 µs in `validate` because the per-call finite scan remains.
The earlier B/C delta of hundreds of µs was therefore not a metadata/alias
layer measurement; it mixed C's faster finite-scan helper and pack implementation
with runtime generic work.

## Full-model results

Tiny ViT, 256 images, 7 rounds; one compiled graph per arm.

| model | current median µs | experimental median µs | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 4126.3 | 4077.9 | **1.0168×** | 1.0058× | yes |
| Tiny ViT b1 | 1552.5 | 1551.1 | **1.0010×** | 0.9999× | yes |

b8 has a small repeatable gain (~1.7%); b1 is essentially neutral (~0.1%). b1
was run only after the b8 gain was observed. All b8/b1 logits are bitwise equal.

## Safety checks

QK/PV both report `PASS` for:

- wrong static shape rejected;
- non-finite input rejected with no output change;
- input address change triggers full revalidation and cache refresh;
- restored bindings reproduce the baseline output bitwise.

Failed shape/finite runs publish no partial output.

## Decision

The static-only cache gives a **real but small** b8 gain and no meaningful b1
gain. It is **not enabled by default and not productized in this stage**:

- b8 ~1.7%, b1 ~0.1%, with shape-dependent behavior;
- it would require a long-lived cache/lifetime/invalidation safety contract;
- this stage explicitly withheld guard/sync/publication changes;
- default behavior remains full per-call validation.

The remaining runtime `validate` cost is dominated by finite scanning, not
static metadata. A future experiment should isolate an equivalent fast finite
scan separately, not bundle it with guard/publication changes.

## Identity

- experimental runtime: `artifacts/librknpu_plan.so`, SHA-256
  `18dac5981ee05ade1d7d45108dc4bfcc8fbcc0d19187d9b471c56923c27fdc4d`
- probe: `artifacts/runtime_layer_probe_prepared`, SHA-256
  `90ac208b6b37aca72ff2c11cccebe0ae80293d3c5d00a1c3b6a5b996e990c58a`
- compiled package: `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec: `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`

## File index

- `analysis.md` / `analysis.json`: generated paired summary.
- `r1..r7/`: primary QK/PV paired samples, outputs and env.
- `other/r1..r7/`: same-implementation MLP/QKV regression samples.
- `profile-static/`: separate `battr` attribution samples.
- `full-model/`, `full-model-b1/`: full Tiny ViT A/B result, logits and logs.
- `checks/`: wrong-shape/non-finite/address-change/republication checks.
- `cases/`: descriptor bundles, constants and case definitions; frames are
  deterministic and regenerated by `tools/runtime_layers/build_cases.py`.
- `artifacts/`: exact experimental library, probe and build logs.
- `campaign-manifest.json`: hashes and protocol identity.
