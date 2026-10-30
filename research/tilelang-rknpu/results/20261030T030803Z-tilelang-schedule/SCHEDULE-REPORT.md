# TileLang schedule experiment report

Scope: verify that a user-visible TileLang M/N schedule changes the RKPlan, then
run two SmolLM-shaped modules on the RK3588 board.  K blocking and accumulation
remain the existing fixed contract (`256`-wide ascending slices, ordered FP16
writeback).

## 1. Small same-GEMM experiment (real TileLang block grid)

Geometry `M=512, K=512, N=384`, real tiled TileLang sources with K-loop and
different M/N block sizes/order.  Compile-only and mock/board execution all
passed.

| source schedule | tasks | submits | buffers | data BOs | publishes | plan hash |
|---|---:|---:|---:|---:|---:|---|
| block 256x128, m_major | 12 | 6 | 22 | 12 | 6 | `3662f2580758...` |
| block 128x128, m_major | 24 | 12 | 38 | 16 | 12 | `48a0e1f0c02a...` |
| block 128x128, n_major | 24 | 12 | 38 | 16 | 12 | `f360bea8f362...` |

The third schedule has the same task/buffer counts as the second but a different
publish traversal order and plan hash.  The source M/N block extents are retained
in `source_mn_tiling`; the chosen tile/order is retained in `source_schedule` and
`applied_schedule`.  TileLang pipeline metadata is recorded as dropped rather
than silently executed.  Submit grouping, BO slots and exact output rectangles
are re-planned by the RK backend.

Board output for all three plans was bit-identical; CPU FP16-K-tiled reference
max abs error was `0.00048828125`.

## 2. SmolLM-shaped board experiments

All runs used the RK3588 board, one process, core mask 1, fixed 256-wide K
contract, and the same deterministic A/B tensors.

### Prefill QKV projection: `M=256, K=576, N=960`

| schedule | tasks | submits | buffers | BOs | median |
|---|---:|---:|---:|---:|---:|
| default (256x256, m_major) | 12 | 4 | 23 | 17 | 8.060 ms |
| tile M=128, N=256 | 24 | 8 | 34 | 20 | 9.280 ms |
| tile M=256, N=128 | 24 | 8 | 43 | 29 | 8.820 ms |
| tile M=256, N=128, n_major | 24 | 8 | 43 | 29 | 8.579 ms |

The default was best for this prefill projection.  Smaller M tiles increased
tasks and buffers without a timing gain.

### Prefill attention QK: `M=768, K=64, N=256`

| schedule | tasks | submits | buffers | BOs | median |
|---|---:|---:|---:|---:|---:|
| default (256x256, m_major) | 3 | 3 | 7 | 5 | 2.377 ms |
| tile M=256, N=128 | 6 | 6 | 11 | 6 | 1.965 ms |
| tile M=256, N=128, n_major | 6 | 6 | 11 | 6 | 1.973 ms |
| tile M=128, N=256 | 6 | 6 | 13 | 8 | 2.705 ms |

Here the explicit N=128 schedule reduced median latency by about **17.3%**
(`2.377 → 1.965 ms`) with bit-identical output.  The traversal-order change did
not add a measurable gain for this shape.

### Decode attention QK: `M=3, K=64, N=256`

| schedule | tasks | submits | buffers | BOs | median |
|---|---:|---:|---:|---:|---:|
| default (256x256, m_major) | 1 | 1 | 3 | 3 | 0.226 ms |
| tile M=256, N=128 | 2 | 2 | 5 | 4 | 1.100 ms |
| tile M=1, N=256 | 3 | 3 | 7 | 5 | 0.420 ms |
| tile M=1, N=128, n_major | 6 | 6 | 11 | 6 | 0.648 ms |

The best prefill strategy (N=128) was much worse for decode; the default single
task remained best.  This is the intended evidence that prefill and decode must
not be assumed to share a schedule.

## 3. Direct RKIR parity

For `M=256, K=576, N=960`, all schedule variants were also emitted as direct
RKIR modules with the same `source_schedule` objects.  TileLang and direct-RKIR
plans had identical task/submit/buffer/BO/publish counts and bit-identical
outputs.  Board medians (same process/order) were in the same range, for example
default TileLang `7.119 ms` vs direct-RKIR `7.742 ms`; this difference is not
interpreted as a TileLang speedup because the plan structure is identical and
the measurement is short/noisy.  The important result is that the schedule
information is representable and executable on both carriers.

## 4. Correctness and limits

Every schedule experiment reports:

- `schedule_equal_all=true`: all variants are bit-identical to the baseline plan
  output for the same deterministic tensors;
- `reference_pass_all=true`: all outputs agree with an independent FP16-K-tiled
  NumPy reference under the declared tolerance;
- max abs error `4.88e-4` (prefill) / `1.22e-4` (QK prefill) / `9.54e-7`
  (decode QK).

Current limits:

- only the supported single-GEMM path consumes explicit `source_schedule`;
- explicit schedules on multi-node/staged dataflow are rejected, not ignored;
- K tile size/order is intentionally fixed at the existing numeric contract;
- whole-model default/optimized and RKNN comparison, plus multi-core search, are
  the next milestone, not claimed here.

## 5. Test and integration status

- targeted schedule tests: `tests/test_schedule_contract.py` 6/6;
- frontend/ragged/K-loop/multitile/FFN targeted suite: 123 passed, 2 subtests;
- full host regression: 441 collected, 440 passed, 1 skipped, 71 subtests;
- C regression: 33/33 ctest passed;
- direct-RKIR board parity: same task/buffer/BO/publish geometry and
  bit-identical output for all tested schedules.

The current milestone stops at the **single-GEMM schedule path**.  Whole-model
default-vs-scheduled package compilation, RKNN comparison and a device-level
multi-core search are deliberately not claimed yet.  The next model-level
experiment should apply the validated QK-prefill N=128 schedule to the SmolLM
attention QK plan, then re-run the complete V5 acceptance against the fixed
V5 spec and compare default/scheduled/direct-RKIR variants.  RKNN requires a
separately exported model and must not be represented by a fake path.
