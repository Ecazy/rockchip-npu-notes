# Creation-time co_program cache A/B

This experiment caches `co_program` only. Command-BO packing, guard,
synchronization and numeric policy are unchanged. The cache is
workspace-local, built after BO/view creation, and reused only for
the same workspace. The current default is on for a structurally
eligible single-submit workspace; `RK_PLAN_PROGRAM_CACHE=0` is the
control.

## Eligibility

- all NPU task programs are emitted through `rk_emit_action` during
  workspace creation and must pass its existing validation;
- the complete task set is one submit and fits the existing
  `batch_scratch`;
- all reads/outputs point to workspace-owned BOs with stable
  addresses; no plan/view/config input changes after creation;
- multi-submit or failed/program-ineligible workspaces keep the
  original per-run emitter path.

The existing `batch_scratch` holds the cached `co_program` array, so
the incremental Host allocation is **0 bytes**; no new `co_program`
array is allocated.

## Isolated cases

µs; ratio = cache-off / cache-on; primary estimate is the median of
per-round paired medians.

| case | cache off | cache on | per-round ratio median | ratio range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 394.3 | 348.0 | **1.1342×** | 1.1305–1.1385 | 1.1305× |
| PV group b8 | 380.3 | 334.2 | **1.1387×** | 1.0220–1.1425 | 1.0220× |
| QKV b8 | 167.7 | 168.0 | **1.0087×** | 0.9774–1.0425 | 0.9774× |
| MLP b32 | 589.7 | 592.7 | **0.9956×** | 0.9916–1.0015 | 0.9916× |

## Creation and emitter cost

| case | setup off µs | setup on µs | emit off µs | emit on µs |
|---|---:|---:|---:|---:|
| QK group b8 | 907.1 | 1162.3 | 46.4 | 0.0 |
| PV group b8 | 991.7 | 1170.4 | 46.4 | 0.0 |
| QKV b8 | 388.8 | 382.4 | 1.5 | 0.0 |
| MLP b32 | 3611.4 | 3680.5 | 14.3 | 15.2 |

The cache eliminates the runtime `emit` interval for QK/PV (about
46–47 µs in the current attribution), but QKV emit is only about
1.5 µs and MLP b32 is not eligible (8 submits), so its fallback is
bitwise equal and neutral. Creation time rises by roughly 179–255 µs
for QK/PV; the steady gains dominate over the first few calls.

## Full model (uninstrumented)

| model | cache off median µs | cache on median µs | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 3700.8 | 3507.0 | **1.0554×** | 1.0525× | yes |
| Tiny ViT b1 | 1239.6 | 1202.7 | **1.0282×** | 1.0231× | yes |

b8 gains about 5.5%; b1 gains about 2.8%. The b1 run was done after
the b8 gain was observed. Both use `stats=None` with no node proxies.

## Lifecycle and failure checks

- isolation: **PASS** ({"frame0_cache_matches_reference": true, "frame1_other_workspace_matches_reference": true, "repeat_same_workspace_unchanged": true, "status": "PASS", "workspace_count": 3})
- pv_b8: **PASS** ({"address_revalidated": true, "finite_output_unchanged": true, "rejected_finite": true, "rejected_shape": true, "restored_output_unchanged": true, "shape_output_unchanged": true, "status": "PASS"})
- qk_b8: **PASS** ({"address_revalidated": true, "finite_output_unchanged": true, "rejected_finite": true, "rejected_shape": true, "restored_output_unchanged": true, "shape_output_unchanged": true, "status": "PASS"})
- pair comparison rows: 56, differences: 0
- frozen numeric replay: **PASS**, problems: []
- existing CTest suite (including allocation/creation failures) and
  Python workspace tests pass with the cache default enabled; commands
  change only cache control, and a failed program build falls back.

## Decision

Adopt the creation-time co_program cache as the default for eligible
single-submit workspaces. Keep `RK_PLAN_PROGRAM_CACHE=0` as the
control/opt-out. This is not command-BO packing reuse; that remains a
separate future experiment.
