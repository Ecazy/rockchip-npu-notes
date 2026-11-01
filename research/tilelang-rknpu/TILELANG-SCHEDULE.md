# TileLang → RKIR → RKPlan schedule contract (first milestone)

This milestone makes a **small, explicit scheduling surface** control the RK
execution plan.  It is not a general autotuner and does not change the numeric
contract.

## Source mapping and rounding boundary

TileLang's block grid is a launch structure; it does not by itself define a
unique hardware time order.  The frontend proves the M/N block-variable roles
from the actual A/B/C index expressions.  Both the canonical 2-D grid
(x=N, y=M) and the equivalent swapped grid (x=M, y=N) are admitted when their
index maps cover each output element exactly once.  A block-axis mapping that
cannot be proved from the expressions is rejected.

The backend schedule (rknpu.tile_m, rknpu.tile_n, rknpu.mn_order) is a separate
policy layer.  If it differs from the source tile extents, the planner performs
an explicit retile and records both source_mn_tiling and applied_schedule; this
is not a silent overwrite.  The launch-grid order is never presented as a
hardware execution order.

K rounding is part of numerical semantics, not a tunable hint.  The admitted
explicit source form performs FP16 accumulator -> FP32 GEMM -> FP16 accumulator
inside every K block.  A scheduled source that only narrows once after a full
FP32 K reduction is rejected with RK_E_NUMERIC_CONTRACT; legacy sources without
an explicit schedule remain on the historical default path and report their
rounding as fp32_final_narrowing/single_k_tile_fp16.

The guarded rectangular-tail extractor supports two distinct source forms and
identifies them by their actual copies, never by an annotation:

- `explicit_per_k_fp16`: `Acc16` is cleared once, each K iteration copies
  `Acc16 -> Acc32`, runs the GEMM, and copies `Acc32 -> Acc16`;
- `fp32_final_narrowing`: one FP32 accumulator is cleared once and narrowed only
  by the final output store.  This remains a legacy, unscheduled default form.

Every accepted tail source zero-fills the full A/B physical tiles on every K
iteration, guards only the logical reads, and guards the output rectangle.  The
frontend records logical K, the ordered `k_blocks` partition, each block's
effective extent, the tail extent, the physical K padding and the source
rounding points.  The planner records `k0`, `kt`, `k_physical_extent` and
`k_padding` on every emitted task.

## Supported contract

TileLang PrimFuncs may declare:

```python
T.func_attr({
    "rknpu.schedule": "mn_tiles_v1",
    "rknpu.tile_m": 256,
    "rknpu.tile_n": 128,
    "rknpu.mn_order": "m_major",   # or "n_major"
})
```

The frontend validates these annotations and writes an RKIR matmul attribute:

```json
{
  "kind": "mn_tiles_v1",
  "tile_m": 256,
  "tile_n": 128,
  "order": "m_major",
  "k_tile": 256,
  "accumulation": "ordered_ascending_f16_writeback"
}
```

A declared `source_schedule` is consumed by every single-GEMM path.  When one
is present the plan records both `source_schedule` and `applied_schedule` on
every emitted matmul task, and a source whose declaration cannot be applied is
rejected instead of being planned at the default geometry.  When nothing is
declared the historical 256x256 M-major geometry and the target numeric policy
the frontend already stamped are used, and no schedule field is written, so the
plan identity of an undeclared production source does not move.  The source M/N block extents in a real TileLang K-loop source are also
carried into the same contract; changing `bm`/`bn` in the TileLang kernel changes
the plan.

## Fixed in this milestone

- K is always sliced in ascending order by `256`; a final K tail, if present,
  remains the last ordered slice with its exact effective extent.
- Accumulation is always ordered FP16 writeback at each K tile.  A scheduled
  source must prove explicit per-K FP16 rounding; an absent rounding field is
  rejected rather than treated as compatible.
- Tile extents are `1..256`.
- Unknown schedule keys, unknown traversal orders, non-256 K blocks,
  non-contiguous K partitions and explicit schedules on multi-node dataflow are
  rejected with a diagnostic.
- No schedule annotation is silently treated as a performance hint and then
  dropped.
- Where the M/N block decision came from is recorded, not inferred:
  `source_schedule_origin` is `annotated` for a user annotation and `derived`
  when the source itself proved the blocks (a kernel that writes FP16 back after
  every K block), and the report states the adopted grid
  (`mn_grid_source`: `annotation` / `source_geometry` / `default_grid`).  Plan
  admission gates only treat `annotated` (or a hand-written direct-RKIR
  attribute) as a user requirement; a derived record is never reported back to
  the user as their own explicit scheduling request.
- A declared M/N tile is nominal: when it is wider than the remaining
  axis the last tile is clipped to the exact tail, on every policy.  The plan
  keeps the declared block and the report exposes the published rectangles and
  whether they were clipped.
- A declared M/N tile composes with a linear task policy.  Combinations whose
  submit grouping would need more than one submit's 64 tasks are rejected with
  the policy, the tile geometry, the required task count and the bound, rather
  than falling back to the default grid or surfacing a generic submit error.

## Information audit

| source information | status |
|---|---|
| M/N block extents | retained and applied |
| M/N traversal order | retained and applied where more than one tile exists in the slower axis |
| K block | retained only as validation; fixed at 256 |
| K accumulation order | fixed by numeric contract |
| TileLang `num_stages` / pipeline annotation | retained in `source_k_loop.pipeline_hint_dropped` as dropped metadata; not executed |
| logical GEMM shape | retained in bindings and task `k0/kt`; output tiling is re-planned |
| submit grouping / BO slots | local planner decision, not a TileLang promise |
| shared-memory allocation sizes | source execution detail; not a hardware-plan identity |

## Direct RKIR

A direct-RKIR matmul can express the same schedule by putting the identical
`source_schedule` object on the `Op` attrs.  The planner path is shared with the
TileLang carrier; tests and board experiments verify the same task/buffer counts
and bit-identical outputs.

## Tuner, replay and model override

`tools/tune_schedule.py prepare` compiles finite M/N candidates for real SmolLM
shapes and captured activations.  `run` performs static C-verifier preflight,
board correctness checks and full-call timing, selects a winner (retaining the
default when it is within 2% of the measured best), and writes `policy.json`
with a cache identity.  `replay` is invoked in a fresh SSH process and rejects
changed package/weights/library/plan/schedule identities.

The selected QK-prefill plan can be applied to the full model through
`--schedule-policy` (`smollm2-plan-overrides-v1`), without changing the accepted
package manifest/weights identity.  The V5 fixed acceptance under that override
passed 56/56 gates with bit-identical observation arrays; the end-to-end
benchmark did not show a stable speedup, so the operator gain is not misreported
as a model gain.

## Restricted two-stage schedule

Two real `T.Kernel` launches may express `Z = X @ W1`, `Y = Z @ W2` with Z kept
on device.  The versioned annotation is:

```python
T.func_attr({
    "rknpu.staged_schedule": "staged_mn_tiles_v1",
    "rknpu.stage0.tile_m": 256, "rknpu.stage0.tile_n": 256,
    "rknpu.stage0.mn_order": "m_major",
    "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
    "rknpu.stage1.tile_m": 256, "rknpu.stage1.tile_n": 128,
    "rknpu.stage1.mn_order": "n_major",
    "rknpu.stage1.source_k_rounding": "explicit_per_k_fp16",
})
```

A global schedule attribute is rejected instead of being copied to every stage.
The safe support domain is deliberately narrow: stage 0 `tile_n` is 256 so each
consumer K slice maps to one producer Z tile; the applied stage-0 and stage-1 M
tiles must match; the intermediate H tail must be 32-element aligned.  Different
source M tiles are allowed only through an explicit backend retile to the same
consumer-safe applied M tile.  Arbitrary producer/consumer tile recombination,
row subviews with different CUBE physical-row geometry, and a device repack are
rejected.

Z is a temporary value with per-tile device buffers, no host upload or download.
Every consumer task records the producer view/action, producer tile, consumer
region, row offset and logical K; submit groups remain serial, with all stage-0
submits before stage-1.  A missing producer dependency is rejected by the C
verifier; out-of-buffer or mismatched views are rejected by the plan/view layer.

## SmolLM-shaped scheduled FFN

The closed FFN graph is:

```text
Zgu = X @ Wgu                         (packed gate/up, K=576 tail)
G = view(Zgu[:, :H]); U = view(Zgu[:, H:])
S = fp16(sigmoid_lut(G)); Silu = fp16(G*S); A = fp16(Silu*U)
P = A @ Wd                           (down, K=H)
Y = fp16(P + X)                      (residual)
```

It is six real `T.Kernel` launches: gate/up matmul, sigmoid, silu multiply,
gated multiply, down matmul, residual add.  The packed gate/up output is split
by two zero-copy `view` IR nodes, so no Host split or repack is used.  Every
stage has its own `rknpu.stageN.*` annotations; a global schedule is rejected.

Support domain: SmolLM D=576, H=1536; stage0 tile_n=256; all stages use the same
applied M; SwiGLU elementwise tile_n=256; down/residual tile_n must divide 576
(64 or 192 in the experiment); intermediate H is 256-aligned.  Unsupported
shapes/tile combinations are rejected.  The device numeric policy is an explicit
experimental path: FP16 after sigmoid LUT, after gate multiply, after gated
multiply, after down K blocks, and after residual add.  It is not the default
Host FP32 SwiGLU path and did not change V5 gates.

## Unsupported in this milestone

- General staged/multi-node schedule propagation is rejected.  Only the
  restricted two-stage contract above is admitted.
- K tile sizes or K accumulation order other than the fixed contract are
  rejected.
- Sources with K>256 that narrow only at the final store are rejected when a
  schedule is requested; the legacy default path remains available without a
  schedule.
- K=1/255/256/257/511/512/576/1536 are covered.  An arbitrary source K block,
  split-K, swizzle, async pipeline, multi-core and multi-node schedule
  propagation are not included.
- A general search/autotuner, core mask selection, and multi-request scheduling
  are not included.

## Replay

```bash
python3 tools/run_schedule_experiment.py compile \
  --source tiled --out /tmp/small-plans --m 512 --k 512 --n 384 \
  --schedule base_256x128,256,128,m_major \
  --schedule tile_128x128,128,128,m_major

./scripts/run-smollm2.sh schedule-bench /tmp/small-plans /tmp/small-board.json
```

The first milestone experiment and measurements are in
`results/20261030T030803Z-tilelang-schedule/SCHEDULE-REPORT.md`.

The restricted two-stage experiment is in
`results/20261030T102526Z-two-stage/TWO-STAGE-REPORT.md`.

The SmolLM FFN experiment is in
`results/20261030T112127Z-ffn/FFN-REPORT.md`.

The K-tail round is in
`results/20261030T083436Z-k-tail/K-TAIL-REPORT.md`: it adds explicit per-K
FP16 rounding for K=576 tails, exact K=1536 blocks, strict rounding/schedule
validation, and model-binding-compatible M/N candidates for QKV, gate/up,
down and lm_head.  The board autotuner retained the default in all seven cases;
the scheduled-source coverage is reported separately from performance.
