# Creation-time `co_program` cache checkpoint

Date: 2026-10-27. Source/tool commit: `ec965b5` (baseline `f18c9f3` report
entry; this candidate is the next source delta).

This result implements and evaluates one bounded candidate:

> At workspace creation, emit and validate `co_program` for eligible tasks;
> reuse those immutable programs across runs. Do not change command-BO packing,
> guard, synchronization, publication or numeric policy.

## Eligibility predicate

A workspace is eligible only when:

- it has exactly one submit;
- that submit has a non-zero whole task set that fits the existing
  `batch_scratch`;
- every task is an NPU task accepted and validated by `rk_emit_action` at
  creation;
- all reads/outputs are workspace-owned views with stable BO addresses.

This is structural, not model-name based. Multi-submit or failed/unsupported
programs keep the original per-run emitter path. The cache uses the existing
`batch_scratch`, so it adds **0 bytes** of Host `co_program` allocation.

## Implementation and lifecycle

- env flag: `RK_PLAN_PROGRAM_CACHE`, snapshotted at workspace creation;
- default: **on** for eligible single-submit workspaces;
- control/opt-out: `RK_PLAN_PROGRAM_CACHE=0`;
- cache is per workspace, per fd/domain; no cross-workspace or global state;
- no BO is reallocated after workspace creation;
- cache is built after BO/view creation and before the workspace is returned;
- a cache-build failure falls back to the original emitter path;
- workspace creation failure cleanup frees the same `batch_scratch` and all
  other allocations through the existing fail path;
- runtime binding/finite/alias validation, guard, sync, download and staged
  publication are unchanged.

## Independent emitter attribution (current version)

`battr` medians over 32 calls, µs:

| case | cache off emit | cache on emit | cache off total | cache on total |
|---|---:|---:|---:|---:|
| QK group b8 | 46.4 | 0.0 | 434.3 | 346.8 |
| PV group b8 | 46.4 | 0.0 | 376.0 | 332.8 |
| QKV b8 | 1.5 | 0.0 | 163.6 | 163.9 |
| MLP b32 | 14.3 | 15.2 | 596.5 | 594.1 |

QK/PV emit is worth removing; QKV emit is too small; MLP b32 is not eligible
(8 submits) and correctly falls back to the original path.

## Isolated A/B

Same library, same workspace implementation, same pack/emitter/executor/guard/
download/publication; only the program-cache env differs. Seven rotated rounds,
changed-input frames. Primary estimate: median of per-round medians; p10
auxiliary.

| case | cache off | cache on | per-round ratio median | range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 394.3 | 348.0 | **1.1342×** | 1.1305–1.1385 | 1.1305× |
| PV group b8 | 380.3 | 334.2 | **1.1387×** | 1.0220–1.1425 | 1.0220× |
| QKV b8 | 167.7 | 168.0 | 1.0087× | 0.9774–1.0425 | 0.9774× |
| MLP b32 | 589.7 | 592.7 | 0.9956× | 0.9916–1.0015 | 0.9916× |

All **56** cache-off/on output comparisons are bitwise equal.

## Creation time and Host memory

| case | setup off µs | setup on µs | extra Host `co_program` bytes |
|---|---:|---:|---:|
| QK group b8 | 907.1 | 1162.3 | 0 |
| PV group b8 | 991.7 | 1170.4 | 0 |
| QKV b8 | 388.8 | 382.4 | 0 |
| MLP b32 | 3611.4 | 3680.5 | 0 (not eligible) |

Each `co_program` is 16,624 B, but the cache uses the already allocated
`batch_scratch`. QK/PV pay about 179–255 µs once at creation; the steady
~46 µs/call gain repays this after roughly 4–6 calls.

## Full Vit model (uninstrumented, `stats=None`)

256 images, 7 rounds; same compiled graph, two runners in one process, env
snapshotted before each runner's workspaces; no node proxies.

| model | cache off median | cache on median | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3700.8 µs | 3507.0 µs | **1.0554×** | 1.0525× | yes |
| Tiny ViT b1 | 1239.6 µs | 1202.7 µs | **1.0282×** | 1.0231× | yes |

b1 was run only after the b8 gain was observed. Full output logits are saved
in `full-model-b8/` and `full-model-b1/` and are bitwise equal.

## Lifecycle / failure checks

- Three-workspace isolation: two cache-on workspaces with different input
  frames and one cache-off reference; both frame outputs and the repeat output
  are bitwise equal. **PASS**.
- Wrong static shape, non-finite input, address change, restored binding, and
  failed-publication checks: **PASS** for QK/PV.
- Existing CTest failure-injection scenarios (including allocation/creation
  and submit failure) pass with the cache default on: **32/32**.
- Frozen numeric acceptance replay: **PASS / 0 problems**.
- Host test suite: **348 passed, 3 skipped, 68 subtests**.

## Decision

**Adopt the creation-time single-submit `co_program` cache as the default** for
eligible workspaces. `RK_PLAN_PROGRAM_CACHE=0` retains the original path for
A/B and rollback.

This is only the program cache. Command-BO packing reuse, guard, sync,
publication and numeric policy remain unchanged. Command packing is a separate
candidate for a later experiment; it is not implemented here.

## Identity

- adopted runtime: `artifacts/librknpu_plan.so`
  `cc4586eb4d51edf3b7cbab5d9fffd2fdc0d40e87b24ee9f052318108eb87cdc9`
- probe: `artifacts/runtime_layer_probe_final`
  `c12153f14dc84ef68299e82fd56846a6cbff98858529f1122ff2f39d818d4b15`
- compiled package:
  `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`
- numeric spec:
  `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`

Note: the isolated A/B and full-model runs used the same program-cache code
with `RK_PLAN_PROGRAM_CACHE=1`; the numeric replay and final artifact use the
adopted default-on build `cc4586...`.

## File index

- `analysis.md` / `analysis.json`: generated summary.
- `pair/r1..r7/`: cache-off/on isolated logs, outputs and environment.
- `profile-emit/`: independent `battr` samples.
- `full-model-b8/`, `full-model-b1/`: uninstrumented full-model A/B, logits and
  logs.
- `checks/`: isolation and failure checks.
- `numeric/`: frozen replay summary and acceptance observation.
- `cases/`: descriptor/config/constants; frames are deterministic and
  regenerated by `tools/runtime_layers/build_cases.py`.
- `artifacts/`: adopted library, probe and build logs.
- `campaign-manifest.json`: hashes and protocol identity.
