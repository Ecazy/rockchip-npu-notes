# Same-implementation prepared-validation A/B

This is the corrected causal experiment. Both arms use exactly the
same C implementation, pack, emitter, command workspace, executor,
guard, download and publication. Only the workspace-snapshotted
`RK_PLAN_PREPARED_VALIDATION` flag differs. The flag is experimental
and defaults off; no public ABI or default policy changed.

Primary estimate is the median of **per-round paired medians**; p10
is auxiliary. Finite input scans remain per call, and changed
bindings fall back to full validation before cache refresh.

## Isolated cases

| case | static=0 µs | static=1 µs | per-round ratio median | ratio range | p10 ratio |
|---|---:|---:|---:|---:|---:|
| QK group b8 | 884.9 | 871.5 | 1.0154× | 0.9776–1.0178 | 0.9776× |
| PV group b8 | 721.9 | 709.0 | 1.0185× | 0.9402–1.0206 | 0.9402× |
| MLP b1 | 436.0 | 436.3 | 0.9993× | 0.9960–1.0040 | 0.9960× |
| MLP b32 | 664.4 | 663.2 | 1.0000× | 0.9987–1.0071 | 0.9987× |
| QKV b8 | 184.0 | 184.0 | 0.9984× | 0.9920–1.0032 | 0.9920× |

## Validation phase split

Same `battr` path, one sample per arm; fields are µs.

| case | arm | validate | pack | command | submit/wait | output-sync | download | guard | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QK group b8 | static=0 | 422.9 | 40.0 | 25.4 | 61.8 | 98.6 | 117.2 | 25.4 | 850.3 |
| QK group b8 | static=1 | 409.6 | 39.4 | 25.1 | 98.0 | 98.3 | 116.7 | 25.4 | 870.6 |
| PV group b8 | static=0 | 236.5 | 45.2 | 25.1 | 61.0 | 98.6 | 116.7 | 25.1 | 674.3 |
| PV group b8 | static=1 | 223.1 | 46.1 | 25.4 | 97.3 | 106.7 | 124.7 | 26.2 | 704.5 |

The static cache removes only the generic metadata/alias portion;
the finite input scan remains and is the dominant `validate` cost.
For QK/PV the measured validate delta is about 13–14 µs, not the
hundreds of µs previously inferred from a different finite-scan/pack
implementation. The earlier B/C delta therefore cannot be assigned
to the RKPlan metadata layer.

## Full model

| model | current median µs | experimental median µs | per-round ratio median | p10 ratio | bitwise |
|---|---:|---:|---:|---:|---|
| Tiny ViT b8 | 4126.3 | 4077.9 | 1.0168× | 1.0058× | yes |
| Tiny ViT b1 | 1552.5 | 1551.1 | 1.0010× | 0.9999× | yes |

The b8 arm shows a small but repeatable gain; b1 does not. The b1
result is included only after the b8 gain was observed.

## Safety checks

- QK group b8: status **PASS**; wrong shape rejected True; non-finite rejected True; output unchanged on both failures True/True; address change revalidated True; restored output unchanged True.
- PV group b8: status **PASS**; wrong shape rejected True; non-finite rejected True; output unchanged on both failures True/True; address change revalidated True; restored output unchanged True.

## Decision

The static-only cache gives only a small b8 gain and essentially no
b1 gain. It is not enabled by default and is not productized in this
stage. Guard, synchronization, finite checks and publication remain
unchanged. A future product decision would need a separate safety/
lifetime contract; the observed gain alone is not sufficient.
