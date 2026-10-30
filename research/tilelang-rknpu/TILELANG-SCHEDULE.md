# TileLang → RKIR → RKPlan schedule contract (first milestone)

This milestone makes a **small, explicit scheduling surface** control the RK
execution plan.  It is not a general autotuner and does not change the numeric
contract.

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

The planner consumes `source_schedule` for the supported single-GEMM path and
records both `source_schedule` and `applied_schedule` on every emitted matmul
task.  The source M/N block extents in a real TileLang K-loop source are also
carried into the same contract; changing `bm`/`bn` in the TileLang kernel changes
the plan.

## Fixed in this milestone

- K is always sliced in ascending order by `256`; a final K tail, if present,
  remains the last ordered slice.
- Accumulation is always ordered FP16 writeback at each K tile.
- Tile extents are `1..256`.
- Unknown schedule keys, unknown traversal orders, non-256 K blocks and
  explicit schedules on multi-node dataflow are rejected with a diagnostic.
- No schedule annotation is silently treated as a performance hint and then
  dropped.

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

## Unsupported in this milestone

- Explicit schedules on staged/multi-node dataflow are rejected until that
  lowerer has a schedule-aware tiling proof.
- K tile sizes or K accumulation order other than the fixed contract are
  rejected.
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

The delivered experiment and measurements are in
`results/20261030T030803Z-tilelang-schedule/SCHEDULE-REPORT.md`.
