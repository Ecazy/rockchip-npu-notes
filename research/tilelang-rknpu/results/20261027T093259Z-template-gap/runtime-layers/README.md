# Runtime layer isolation checkpoint

Date: 2026-10-27. Implementation/tool source: `136e2ba`.

This result answers one question: when the same RKPlan descriptor, program
emission, submit grouping and device execution are held fixed, how much time is
added by the Python/FFI layer, the RKPlan generic layer, the custom-ops/provider
path, and the remaining execution boundary?

It is an isolation experiment. It does **not** change the default safety
contract, does not enable validation/guard/publication batching, and does not
change device-FFN policy.

## Protocol and fixed conditions

- RK3588 OPi 5 Plus, kernel `6.1.43-rockchip-rk3588-rknpu098`, driver `v0.9.8`.
- CPU affinity 6, governor `performance`, 2.304 GHz.
- NPU one core, governor `performance`, 1.0 GHz.
- DMC governor `performance`, 2.112 GHz.
- A0/A1/B/C main timings use the legacy, non-attributed entry. A and B share
  the same C runtime; C uses direct custom-ops calls. Attribution is a separate
  `battr`/`cprof` sample.
- 7 outer rounds; case and path order rotated each round.
- Every process consumes the same changed-input frame sequence; 63/31 warm
  samples per process for the 64/32-frame cases.
- Main table values are p10 of warm samples (stable floor). Medians and full
  ranges are in `analysis.json`.
- A0/A1/B/C/D outputs are compared bitwise.

Paths:

| path | meaning |
|---|---|
| A0 | Python generic `run_into` / `run_group_into` (per-call Python validation) |
| A1 | Python prepared binding arrays + direct legacy C entry (the A used for A/B/C comparison) |
| B | native C harness, direct `rk_workspace_run` on the same RKPlan |
| C | native C direct custom-ops; same descriptor/emitter/executor, no per-run generic shape/alias validation |
| D | resident execute-only probe; input updates and result validation outside timing |

C still performs finite input scans, exact upload packing, BO prep/fini, every
submit in the same grouping, `co_guard_check` on all slots, finite output-tile
checks and staged publication. Its explicit difference from B is the absence of
the generic per-run binding shape/alias interpreter path. C's generic upload
pack is not optimized for every non-16×16 shape; those costs are shown, not
hidden.

## Main result

| case | A0 | A1 | B | C | D | A0−A1 Python | A1−B FFI | B−C runtime | C−D data/boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 477.5 | 427.6 | 427.6 | 420.9 | 228.1 | +49.9 | +0.0 | +6.7 | +192.8 |
| MLP b32 | 767.1 | 655.7 | 650.1 | 687.5 | 240.3 | +111.4 | +5.5 | −37.3 | +447.1 |
| QKV b8 | 246.7 | 186.1 | 182.0 | 189.6 | 54.0 | +60.7 | +4.1 | −7.6 | +135.6 |
| QK group b8 | 2228.3 | 897.2 | 834.2 | 466.7 | 179.4 | +1331.2 | +63.0 | +367.5 | +287.3 |
| PV group b8 | 1861.4 | 714.9 | 660.6 | 454.7 | 184.6 | +1146.5 | +54.2 | +205.9 | +270.1 |

All values are µs. Positive `B−C` means direct custom-ops was faster.

Key readings:

- prepared Python/FFI is 0–5.5 µs on MLP/QKV and 54–63 µs on the two expanded
  groups; it is not the external vendor gap.
- generic Python group validation is large (A0→A1: 1331/1147 µs on QK/PV), but
  the production compiled graph already uses prepared bindings.
- RKPlan generic cost is visible only on QK/PV expanded groups (B−C:
  +367.5/+205.9 µs). MLP b32 and QKV are **not** helped: C is 37.3/7.6 µs
  slower, because the direct generic pack is slower than the runtime pack on
  those shapes.
- D is a reference only, not end-to-end execution: it excludes input prep,
  upload, guard, download and publication.

## Separate phase sample

B is `rk_workspace_run_ex` attribution over 96 calls; C is direct C over 32
changing frames, divided per call. These are separate samples, not components
summed to manufacture the headline result.

| case | B validate | B pack | B emit | B command | B submit/wait | B output-sync | B download | B guard | B publish | C finite | C upload | C emit | C execute+sync | C download | C guard | C publish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 2.3 | 11.4 | 14.3 | 30.3 | 175.3 | 27.7 | 2.6 | 163.3 | 0.0 | 0.4 | 14.5 | 14.2 | 237.7 | 2.5 | 163.8 | 0.1 |
| MLP b32 | 68.0 | 156.0 | 14.3 | 28.6 | 184.9 | 30.0 | 3.8 | 161.0 | 0.3 | 10.8 | 254.5 | 14.3 | 250.7 | 3.5 | 161.0 | 0.1 |
| QKV b8 | 17.2 | 43.5 | 1.5 | 2.0 | 48.7 | 5.0 | 35.6 | 26.5 | 1.8 | 2.7 | 67.5 | 1.4 | 56.7 | 25.6 | 26.2 | 1.8 |
| QK group b8 | 423.5 | 39.7 | 46.2 | 23.9 | 97.7 | 94.5 | 112.9 | 24.8 | 1.2 | 65.2 | 25.9 | 46.5 | 189.4 | 105.5 | 24.6 | 1.2 |
| PV group b8 | 236.8 | 45.2 | 45.6 | 24.8 | 60.1 | 98.6 | 116.4 | 25.4 | 1.2 | 35.3 | 34.2 | 45.4 | 190.1 | 109.9 | 25.2 | 1.1 |

QK/PV C still pays finite + guard + download/publication; the gain is the
removed generic validation/interpreter path, not disabling safety. MLP b1
guard is ~164 µs in both B and C.

## Setup, first execution, warm

`analysis.md` contains the full 25-arm table. Examples:

- MLP b1: B/C/D warm p10 427.6/420.9/228.1 µs; setup 4.21/4.78/4.93 ms.
- MLP b32: B/C/D warm p10 650.1/687.5/240.3 µs; setup 4.26/4.92/5.18 ms.
- QK group b8: B/C/D warm p10 834.2/466.7/179.4 µs; setup
  0.99/0.79/1.25 ms. A1 prepared-binding setup adds ~1.0 ms.
- QK/PV first execution is materially above warm; setup/binding time is not
  mixed into steady-state call time.

## Complete-model transmission probe

Tiny ViT b8, 256 images, batch 8, 3 rounds. Only qk/pv group PlanCalls were
replaced by the experimental native-C direct context; all Host actions and
other plans remained the current compiled graph. All 96 paired batches are
bitwise equal.

| arm | app median µs | app p10 µs |
|---|---:|---:|
| current compiled graph | 4002.6 | 3954.7 |
| qk/pv direct-C substitution | 4415.8 | 4352.5 |

Paired current/variant: median **0.9050×**, p10 **0.8820×**. The two replaced
NPU nodes save 28.0 µs (qk) and 78.2 µs (pv) at p10; all measured NPU nodes
save 95.1 µs, but the app is 397.8 µs slower. The ~492.9 µs residual is outside
the replaced group calls and comes from integration/Host/executor behavior. A
no-adapter two-runner control has paired median **0.9991×**, so runner placement
does not explain it.

This is a targeted subgraph integration, not a fully native C model. It is
enough to reject the current direct-group integration shape as a product path;
it does not justify a runtime redesign. A fully native C model is a separate
experiment and would need a single-context Host-action/lifetime implementation.

## External gap

The fixed-frequency external table remains a different numeric policy:

| profile | current delivered µs | vendor µs | CPU FP32 µs |
|---|---:|---:|---:|
| MLP b1 | 599.1 | 258.8 | 252.6 |
| MLP b8 | 716.6 | 309.9 | 1748.9 |
| MLP b32 | 1127.6 | 538.0 | 6830.5 |
| ViT b8 | 4010.7 | 1582.1 | 5856.7 |
| ViT b1 | 1532.4 | 589.4 | 869.5 |

Prepared Python/FFI is too small to explain these gaps. The QK/PV runtime
generic cost is real but does not transmit in the full-model direct adapter.
Vendor internal Submit count/per-kernel time is unknown; the remaining gap
cannot be assigned to Python/FFI or RKPlan alone from this experiment. The
MLP b1/b8 physical floor, guard/pack envelope and vendor boundary/fusion
behavior remain active hypotheses.

## What is removable vs contractual

| work | status |
|---|---|
| per-call generic Python group validation | avoidable via prepared bindings; already production path |
| per-run generic shape/alias/interpreter work for fixed expanded group | B−C gain; requires static proof and a new safety contract; not implemented |
| finite input/output checks | retained in C; must remain |
| upload pack + BO prep/fini | retained; C direct generic pack is slower on some shapes, not a drop-in |
| all-slot guard | retained; MLP ~160 µs, cannot be silently removed |
| download staging + publication | retained; required for all-or-nothing output |
| command/submit/wait/output sync | device/driver floor; D ~179–240 µs still includes it |

## Artifact identity

- final runtime: `artifacts/librknpu_plan.so`,
  SHA-256 `19b5263a137cda82eff215ca181d267923ca9645d4cca2378de290ab6dbcaa16`
- direct shared adapter: `artifacts/librl_direct_group.so`,
  SHA-256 `959f078ae4879647e69e74b890002ddbda3021c9f5b3c070859280dc97eafdaf`
- C probe: `artifacts/runtime_layer_probe`,
  SHA-256 `ca8121edf4d3fbf09fe6d3b580423423b55ecddd314d22a02b464519f50c19f0`
- package: `models/compiled/manifest.json`
  `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec: `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`

## File index

- `analysis.md` / `analysis.json` / `analysis.log`: generated summary.
- `r1..r7/`: per-round A0/A1/B/C/D logs, raw outputs and stderr.
- `profile/`: separate `battr` and `cprof` samples.
- `full-model-final/`: targeted Tiny ViT b8 integration result, raw logits and
  two-runner control.
- `cases/`: descriptor bundles, constants and case definitions; deterministic
  `frames.bin` is regenerated by `tools/runtime_layers/build_cases.py`.
- `artifacts/`: exact board binaries and build logs.
- `campaign-manifest.json`: hashes, counts and protocol identity.

## Replay

See `tools/runtime_layers/README.md` for case generation, native C build and
all five paths. The full experiment is reproducible from:

```bash
PYTHONPATH=$PWD/python:$PWD \
python3 tools/runtime_layers/build_cases.py --out /tmp/runtime-cases
```

The only omitted inputs are the deterministic frame files (~30 MB) and raw model
weights (already in `models/`); case descriptors, constants and frame counts
are committed here.
