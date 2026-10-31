# Real SmolLM gate/up GEMM: task order, submit boundary, CBUF input reuse

Workload: real SmolLM2-135M gate/up weight and captured post-norm input;
M=1 and M=256, K=576, N=3072; K=256/256/64 ascending; FP16 writeback after
each K block; same physical M/N tiling and IO/weight/guard contracts.
RK3588, core mask 1, one board, caches OFF for all four arms.

## 1. Team and review

- Lead: implementation, integration, board runs, final conclusion.
- Researcher (read-only): traced TileLang -> RKIR -> planner -> Submit ->
  emitter -> generator -> runtime; verified data_reuse exists in the vendor
  generator but was dead in the shipping emitter; enumerated same-core,
  same-submit, non-first, same operand/view/K and same bank-split preconditions.
- Independent reviewer: found and forced fixes for the global reuse guard,
  A using a different lowering path, policy identity, geometry equality, and
  the single-submit program/command cache confound.

## 2. Control chain and gaps

The real chain is:

    TileLang source (make_matmul_tiled_tail_prim)
     -> source schedule attrs (mn_tiles_v1, K fixed 256, order m/n major)
     -> RKIR matmul attrs
     -> planner (one task per output tile/K slice, ping-pong accumulators)
     -> Submit builder (group per submit_group; chain/separate modes)
     -> RKPlan descriptor -> emitter -> co_emit_mm/co_emit_mm_acc
     -> gen_matmul_fp16 -> CNA CBUF_CON0 -> RKPlan runtime submit

Baseline gap: output tiles were serialized by per-tile submit groups and
downloads, so the N loop could not enter one submit. The C verifier also did not
admit multiple independent K-accumulation chains in one submit. The generator
had data_reuse, but co_program/emitter did not carry it, so it was always 0.

Minimal extension: explicit linear_task_order policy in CompileOptions, one
scheduled-linear builder for A/B/C/D, a multi_k_acc_v1 recipe plus explicit
source_k0/index/count descriptor fields, and a C verifier rule for multi-chain
K order and reuse eligibility. No older verifier rule was relaxed.

## 3. A/B/C/D task, submit, BO and reuse identity

All arms have the same 36 hardware tasks (12 N tiles x 3 K slices).

| arm | task order | submits | data BOs | reuse tasks |
|---|---|---:|---:|---:|
| A | tile-outer, K-inner, per-tile group | 12 | 52 | 0 |
| B | tile-outer, K-inner, one group | 1 | 52 | 0 |
| C | K-outer, tile-inner, one group | 1 | 63 | 0 |
| D | same as C | 1 | 63 | 33 |

B keeps A's ping-pong footprint because only one tile's two partials are live at
a time in tile-outer order. C/D make all 24 partials live, adding 11 data BOs.
D reuses the input feature tile for N tiles 2..12 of every K block; it never
reuses across a K boundary or the first N tile.

## 4. K order, lifetime, and reuse eligibility

- Every output tile has one ascending K chain 0->1->2, same ping-pong buffers,
  same per-K FP16 writeback in all arms. The order policy only changes emission
  order, not arithmetic.
- The C verifier now proves: a multi-K submit starts at K index 0; each MM_ACC
  continues the named prior's index+1 and declares that dependency; every
  non-final K task has a later continuation; task ids are contiguous (no Host
  action inside the chain); writes are disjoint from live reads.
- Global reuse rule: reuse requires chained submit, non-first task, the same
  input/weight view as the immediately preceding task, the same K index, and
  identical physical/logical geometry (same bank split). Python rejects
  interleaved_reuse with submit_mode=separate.
- Negative coverage: separate-mode reuse rejected; missing prior dependency,
  wrong K continuation, first-task reuse, K-boundary reuse, and geometry/view
  mismatch rejected by planner or C verifier.
- Limitation: plan-level reuse bits verified (0/0/0/33); this round did not add
  a standalone register-word decoder that prints CNA_CBUF_CON0 bit12.

## 5. Correctness

Every warmup and timed call was validated outside the timer against an
independent per-K FP16 reference. All A/B/C/D calls were finite and
elementwise bit-identical to A on every layer/M. The independent reference is
not bit-identical to hardware because the generated kernel tail/accumulator
rounding is one ULP from the scalar reference; A is the cross-arm bitwise
control. No stale-CBUF mismatch appeared in D.

## 6. Performance

Medians over 20 rotated rounds, caches off, warmup 5 excluded. Each row is a
separate layer measurement; no sum-of-layer medians is called one call.

M=256 (ms):

| layer | A | B | C | D |
|---|---:|---:|---:|---:|
| 0 | 26.534 | 22.990 | 24.783 | 22.835 |
| 15 | 21.375 | 20.204 | 22.060 | 20.413 |
| 29 | 26.176 | 23.533 | 25.058 | 21.283 |

M=1 (ms):

| layer | A | B | C | D |
|---|---:|---:|---:|---:|
| 0 | 3.736 | 2.526 | 2.693 | 2.704 |
| 15 | 3.699 | 2.546 | 2.694 | 2.624 |
| 29 | 3.745 | 2.575 | 2.757 | 2.676 |

Paired per-layer medians:

| comparison | M=256 L0/L15/L29 | M=1 L0/L15/L29 |
|---|---|---|
| A/B | 1.106 / 1.066 / 1.104 | 1.460 / 1.418 / 1.444 |
| B/C | 0.928 / 0.909 / 0.954 | 0.935 / 0.942 / 0.930 |
| C/D | 1.084 / 1.062 / 1.078 | 1.007 / 1.039 / 1.037 |
| B/D | 1.004 / 0.966 / 1.093 | 0.934 / 0.993 / 0.970 |

Interpretation:

- Merging 12 submits into 1 (A->B) is a clear win: 6-10% M=256 and 30-46% M=1.
  This is submit/IRQ envelope reduction, not output-sync reduction (12 distinct
  output BOs still need 12 syncs).
- K-outer interleave without reuse (B->C) is slower by 6-9% at both M. C/D pay
  for 11 extra live partial BOs; the order itself is not free.
- Input reuse (C->D) is positive: 6-8% M=256 and 0-4% M=1. The effect is input
  traffic only; weights are not reused and M=1 is weight-pack dominated.
- B vs D is mixed: B wins M=1 and layer 15, D wins M=256 layers 0/29. No single
  arm wins every layer/M. B is the robust default; D is the demonstrated reuse
  path.

## 7. Independent confirmation and replay

The 20-round A/B/C/D rotation is the interleaved measurement. New-process
replays of B and D were run from the frozen plans; B and D replayed valid on
both M and all layers. No result is attributed to program/command cache because
all three caches were disabled for every arm.

## 8. What is useful and what is not

Useful: one legal 36-task submit; explicit K-order/reuse admission; input reuse
for N tiles at M=256; bitwise cross-arm correctness.

Not useful: K-outer interleave alone (C slower than B); reuse on M=1 (near
null, weight-pack dominated); treating A->B as output-sync savings.

## 9. Identity and replay

- result directory: results/20261031T013000Z-linear/
- library: remote build sha recorded by the run JSON
- policies: A scheduled_baseline, B merged, C interleaved, D interleaved_reuse
- replay command:
  python3 tools/run_linear_schedule_experiment.py run --plans <dir> --warmup 5 --rounds 20 --only B --out replay-B.json
- cache control: RK_PLAN_PROGRAM_CACHE=0 RK_PLAN_COMMAND_CACHE=0 RK_PLAN_MULTI_SUBMIT_CACHE=0
- code identity: see MANIFEST.json in this directory.

## 10. Open limitation

A register-word decoder proving CNA_CBUF_CON0 bit12 is not included. The
evidence proves the descriptor reuse flags and cross-arm bitwise correctness,
not the raw register word. This is the next verification item if D is promoted.
