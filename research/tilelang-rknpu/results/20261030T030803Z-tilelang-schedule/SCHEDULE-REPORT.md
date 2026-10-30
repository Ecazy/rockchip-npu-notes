> IMPORTANT: Sections 1-7 below are historical exploration.  They predate the
> explicit per-K rounding source and the confirmation gate.  The former 14.3%
> QK claim is not a current result; see the authoritative section 8.

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

## 6. Autotuner closed loop (real board, new-process replay)

The tuner used real SmolLM2 package weights and captured backend activations from
`acceptance/board/run-00/captured-00.npz`.  Cases cover QKV, gate/up, down, one
lm_head shard, one M=1 decode QKV, and attention QK prefill/decode.  K remains
256-wide ordered FP16 writeback.  Every candidate is a real TileLang PrimFunc;
C verifier preflight rejected candidates that exceeded the runtime's per-slot
buffer bound before they reached the board.

| case | default median | selected | selected median | measured best | reason |
|---|---:|---|---:|---|---|
| prefill QKV `44x576x960` | 1.239 ms | default | 1.239 ms | default | measured best |
| prefill gate/up `44x576x3072` | 3.620 ms | default | 3.620 ms | default | measured best |
| prefill down `44x1536x576` | 1.722 ms | default | 1.722 ms | default | measured best |
| decode QKV `1x576x960` | 0.812 ms | default | 0.812 ms | default | measured best |
| lm_head shard0 `44x576x4096` | 5.191 ms | default | 5.191 ms | default | measured best |
| attention QK prefill `768x64x256` | 2.428 ms | `m256_n128_n` | **2.080 ms** | `m256_n128_n` | measured best |
| attention QK decode `3x64x256` | 0.216 ms | default | 0.216 ms | default | measured best |

The only non-default winner is attention QK prefill; it reduces the measured
operator call from `2.428` to `2.080 ms` (about **14.3%**) while preserving
bit-identical output for the actual captured tensors.  The prefill QKV, gate/up,
down, lm_head and decode QKV shapes favor the existing default.  This is an
important negative result: the schedule search is not rigged to force a
non-default winner.

Every candidate record includes plan hash, task/submit/buffer/BO/publish counts,
schedule, per-round raw times, workspace initialization time, upload/download
counts, output hash and reference error.  `policy.json` stores the selected plan,
cache identity and cache key.  `tune-replay.json` is produced by a **new SSH
process** and verifies the policy cache key, plan hashes, outputs and timing again;
all seven replay cases passed.

Cache identity includes geometry, dtype, fixed K numeric contract, schedule,
plan SHA, package manifest/weights SHA, plan-library SHA and profile hash.
Corrupt or mismatched policy/plan identity is rejected rather than silently
replayed.

Raw evidence:

- `tuning/board/tune-run.json`
- `tuning/board/policy.json`
- `tuning/board/policy-summary.json`
- `tuning/board/tune-replay.json`
- `tuning/plans/manifest.json`
- `tuning/prepare.log`

## 7. Model integration probe

The QK-prefill winner is applied as an external runner plan override for
`attn.qk.prefill`; the accepted package manifest/weights identity is intentionally
left unchanged.  This allows the scheduled plan to be exercised without
rewriting V5 calibration or claiming a new package policy.  Model-level
benchmark/replay results and fixed V5 acceptance are recorded separately under
`model-benchmark/` and `model-acceptance-scheduled/`.

### Model-level benchmark and fixed acceptance

The selected QK-prefill schedule was applied through an external runner plan
override (`attn.qk.prefill`); the accepted SmolLM package manifest and weights
identity were left unchanged.  Policy SHA-256:
`8d3bf552b46611d3c776fcd7cd15ce045d1c1086000074ec09fdf687d4c99a7c`.

Interleaved full-generation A/B (same package, board, core mask 1, one
`generate()` per process, three baseline/scheduled pairs; all six runs produced
the same 32-token output ids):

| metric | default median | scheduled median |
|---|---:|---:|
| TTFT | 3,939.70 ms | 3,834.10 ms |
| prefill | 3,937.84 ms | 3,832.28 ms |
| generate wall | 17,834.77 ms | 18,243.33 ms |
| decode | 13,870.27 ms | 14,331.20 ms |
| full-generate token/s | 1.794 | 1.754 |

The model-level result is **not a stable end-to-end speedup**; the QK-prefill
operator improvement is real, but its share of the full model is small and the
three-pair measurement is noisy.  This is distinct from the schedule-mapping
verdict, which is positive.

Fixed V5 acceptance with the same override:

- `complete_acceptance=true`, `numeric_pass=true`, `quality_pass=true`;
- `gate_count_expected=56`, `gate_count_executed=56`;
- `gate_failures=[]`, `failures=[]`;
- all 44 baseline/scheduled acceptance arrays are **bit-identical**, max_abs
  difference `0.0`;
- generated token ids are identical;
- verify summary SHA-256:
  `3f48ae374a0a6cedcb99cce3742bbe043baecf877fb24145189d6e3d362c1878`;
- acceptance bundle SHA-256:
  `653d342290ae99834faee28a6bb648ca5b89a9aba4bc5e617fe8e9341bb8fa99`.

Evidence: `model-benchmark/`, `model-ab/`,
`model-acceptance-scheduled/`.

### MLP / TinyTransformer regression

The complete-model admission campaign was rerun on the board after the planner
changes.  It passed (`credential scan` also passed), with MLP and TinyViT batch
1/8/32 profiles and all layer traces matching their existing declared contracts.
The concise summary is `model-admission/summary.json`; the full verifier output
is `model-admission/verify.log` (source campaign:
`results/20261030T034726Z-models`).

This regression exercises precompiled existing plans and therefore checks that
the new schedule metadata/planner path did not disturb the legacy plan/runtime
contract.


## 8. Authoritative correction after source-semantics and confirmation audit

### Source mapping

The frontend now proves the M/N block-variable roles from the actual A/B/C index
expressions.  Both the canonical 2-D grid (x=N, y=M) and the equivalent swapped
grid (x=M, y=N) are accepted when the index map covers the output exactly once.
A 2-D mapping that cannot be proven is rejected.  The launch-grid order is not
presented as a hardware time order; planner order is an explicit policy.

Tests:

- canonical and swapped source mapping produce the same task/publish geometry;
- source M/N tile extraction remains in source_mn_tiling;
- explicit backend retile is reported through source_schedule/applied_schedule;
- tests/test_frontend_mntile.py covers swapped axes, traversal order, and
  unprovable mappings.

### K rounding semantics

The tiled fixture now performs an explicit FP16 accumulator round-trip inside
every K block:

    FP16 accumulator -> FP32 GEMM accumulator -> FP16 accumulator

The frontend records source_k_rounding=explicit_per_k_fp16.  A source that keeps
FP32 across K and narrows only at the final store is recorded as
fp32_final_narrowing.  If such a legacy source also carries a schedule,
planner rejects it with RK_E_NUMERIC_CONTRACT rather than silently applying the
backend 256-wide FP16 contract.  Legacy sources without a schedule remain on
the historical default path.

The explicit source was recompiled and board-run:

- M=512, K=512, N=384;
- base 256x128: 12 tasks, 6 submits, 22 buffers, 12 BOs, 6 publishes, median 6.372 ms;
- tile 128x128 m_major: 24/12/38/16/12, median 7.806 ms;
- tile 128x128 n_major: same counts, different publish order/hash, median 8.180 ms;
- all schedules bit-identical to one another, all pass the independent FP16
  K-tiled reference (max abs 4.88e-4).

The old rows in sections 1-2 that used full-source FP32-final narrowing for
K>256 are historical target-policy experiments, not evidence of source
rounding equivalence.

### Tuning result and confirmation gate

Current final tuner state (see tuning/board):

- 7 real-activation cases: QKV, gate/up, down, decode QKV, lm-head shard,
  attention QK prefill/decode;
- 92 candidates, with 4 C-verifier per-slot rejections and scheduled
  K=576/1536 legacy-source rejections recorded rather than executed;
- every candidate record has bitwise_equal, numeric_pass and candidate_valid;
  warmup and every timed execution are checked outside the timer;
- final winner is the fixed default for every case;
- attention QK prefill default median 2.217 ms; the closest candidate
  m256_n128_n was 2.218 ms, effectively a tie;
- because the autotuner retained default, independent A/B was explicitly
  skipped (tune-confirm.json: skipped=default_wins), not falsely reported as a
  gain;
- the prior 14.3% observation is therefore not upgraded.

Cache/replay:

- policy.json contains shape, dtype, K contract, schedule, plan SHA, package
  manifest/weights SHA, plan-library SHA, profile and cache key;
- tune-replay.json is a fresh SSH process; all 7 default cases pass and
  candidate_valid/numeric_pass are true; bitwise equality is reported
  separately (false for most projection cases, true for decode QK);
- negative tests reject empty/duplicate policy, changed plan/loaded hashes,
  changed package/library/numeric identity, and unexpected overrides.

### Override effective identity

A manual legal schedule plan was used to exercise the override path (not as an
autotuner winner): attn.qk.prefill m256_n128_n.

The override policy includes:

- base package manifest and weights identity;
- logical plan key attn.qk.prefill;
- base plan hash and selected plan hash/plan SHA;
- replacement_kind=single_matmul_schedule_v1;
- policy content SHA and selected-plan cache key.

Runner compatibility checks binding roles/shapes/dtypes, constants, target
profile, and single-matmul task family before loading.  The verifier accepts an
expected execution identity and checks the board-recorded policy SHA, plan keys,
plan hashes, loaded plan hashes, package/weights/library/profile/numeric mode.

Manual override V5 result:

- 56/56 gates, complete_acceptance=true, numeric_pass=true, quality_pass=true;
- verify summary SHA-256 eec1e9f029bf0188674859919a2bffc1e743b06f44bdbc1023bdf8c0ed7eb766;
- acceptance bundle SHA-256 c9888f668b4459642f95a4f90d7772c0308179a974ddfc8cbd505e81a4ec65cf;
- manual override policy SHA-256 6ac67f80556467f83afe62d21d1328c6f5644a37ac7498cfc2f6bd4a6ef5ef10;
- expected execution identity SHA-256 b9279b57caddb17970bc3964179a453904c2fdbb713c9c68ddc45d7d845e75ec.

Final default V5 rerun:

- 56/56 gates, complete_acceptance=true, numeric_pass=true, quality_pass=true;
- verify summary SHA-256 33bbb85662578f20647016a3c8b8a468e9e29f559ce716d4e05c2e24a6b2e0fa;
- acceptance bundle SHA-256 7587ca14e408d4711520d7aaefd4313450916ada72928cf3a293baf3ec15c812.

### Regression and remaining boundary

- full host: 454 collected, 453 passed, 1 skipped, 71 subtests;
- C: 33/33 ctest passed;
- MLP/TinyViT admission: PASS (model-admission/summary.json and verify.log);
- no re-calibration and no V5 threshold changes were made.

Still not supported:

- K=576/1536 sources using only final FP32 narrowing cannot enter the scheduled
  path; they remain legacy default-only until a tail-aware explicit-rounding
  source is implemented;
- multi-node/staged explicit scheduling, split-K, swizzle, async pipeline,
  multi-core, device KV and RKNN comparison remain out of scope;
- no end-to-end model speedup is claimed; the only current winner is the fixed
  default.
