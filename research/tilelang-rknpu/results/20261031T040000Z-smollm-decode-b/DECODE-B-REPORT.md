# SmolLM decode gate/up merged-B candidate

Scope: only layer.N.gate_up.decode for all 30 layers. Source is the real
models/tilelang_matmul.py:make_matmul_prim. The policy is produced by
tools/compile_decode_merged_policy.py through CompileOptions
linear_task_order=merged; no hand-edited plan JSON, planner/verifier bypass,
prefill override, or other operator is used.

## 1. Compile-entry control evidence

Compile chain:
make_matmul_prim (one real TileLang Kernel, no schedule attrs)
 -> compile_tilelang(options.linear_task_order=merged)
 -> RKIR single matmul with the production numeric contract
 -> build_scheduled_linear planner branch
 -> submit group mk:linear / recipe multi_k_acc_v1
 -> descriptor 36 tasks / 1 submit
 -> C verifier / runtime -> actual workspace.

Planner hard gate is M in {1,256}, K=576, N=3072, so qkv/o/down are structurally
ineligible. Merged tasks keep identical per-task K slices 0/256/512 and extents
256/256/64, identical per-K FP16 writeback, identical templates
(co.mm.f16.v1 + 2x co.mm_acc.f16.v1), identical bindings/constants/profile.
No parallel TileLang block grid is claimed to define hardware order; the order
is a target-specific CompileOptions decision.

Plan identity (real package decode plan vs B):

| arm | plan hash | tasks | submits | data BOs |
|---|---|---:|---:|---:|
| P | d8200f450a1218cbe370bccb5a37d74f070c0afd9c9359b305ce83d10ec4dfba | 36 | 12 | 41 |
| B | 48df00c8ab79a45a367776c7ec20c177ca9b9f860e5e866d24f02ec573ce3a0b | 36 | 1 | 52 |

The policy has exactly the 30 decode gate_up keys. Prefill keys are untouched:
layer.0.gate_up.prefill remains 36 tasks / 12 submits. A 1-token prompt runs the
decode path inside prefill(), so the override also affects that path; multi-token
prefill does not.

## 2. Actual loading, not metadata only

Runner recorded loaded_plan_override_hashes for all 30
layer.N.gate_up.decode keys, each equal to
48df00c8ab79a45a367776c7ec20c177ca9b9f860e5e866d24f02ec573ce3a0b.
Unoverridden prefill plan remains 12 submits.

Decode per-forward counts, measured from run stats:
P: 1032 submits / 3006 tasks.
B: 702 submits / 3006 tasks.
Delta: exactly -330 submits/forward = 30 layers x (12 - 1); task count unchanged.
This is actual execution accounting, not a theoretical multiplication.

## 3. Correctness

P and B were run on the same fixed 5 x 32-token generate requests. All
generated token IDs are identical. Every timed call was validated outside the
timer; outputs are finite.

V5 numeric verification (B policy, frozen V5 spec and references):
- 56/56 numeric gates executed, 0 gate failures, numeric_pass true.
- teacher forcing 76/76; incremental 17/17; incremental top1 equal.
- complete_acceptance is false only because the prior acceptance bundle was
  not rebuilt for the new evidence tree; the acceptance bundle hash/missing-file
  failures are external bundle identity, not numeric or quality gate failures.
Quality is therefore reported as unchanged by token equality and numeric gates,
not as a newly regenerated complete V5 bundle.

## 4. Model performance (5 paired processes, P/B/B/P alternating)

Fixed prompt, 32 new tokens; warmup processes discarded; per-process median:

| metric | P | B | P/B ratio | change |
|---|---:|---:|---:|---:|
| prefill (ms) | 3804.5 | 3774.5 | 1.0076 | -0.8% |
| TTFT (ms) | 3811.7 | 3781.3 | 1.0096 | -0.8% |
| decode (ms) | 14060.5 | 12939.8 | 1.0676 | -6.8% |
| generate wall (ms) | 17896.7 | 16740.9 | 1.0530 | -5.3% |
| decode forward/s | 2.205 | 2.396 | -- | +8.7% |
| output token/s | 1.788 | 1.911 | -- | +6.9% |

Ratios are time(P)/time(B); percent change is reported separately. Per-pair
wall ratios range 1.038-1.106, decode 1.046-1.139. Prefill/TTFT stay within
about 1%, consistent with the decode-only scope.

## 5. Initialization and memory

Load time: P 10.521 s, B 10.673 s (+1.5%).
Peak RSS: P 1,246,080 kB, B 1,276,480 kB (+30.4 MB, +2.4%).
Per overridden plan device bytes +969,408 B; 30 layers ~+27.7 MB device.
Data BOs per plan 41 -> 52 (+11); total model plan-declared data BOs 7168 ->
7498. Host bytes per plan -85,184 B.
These are real costs and are reported separately from online decode gain.

## 6. Cost attribution

Decode submit_wait falls from 163.9 ms/forward (P) to the B measured profile,
consistent with 330 fewer submit envelopes; task count is unchanged. submit_wait
contains device execution and cannot be split into device vs IRQ/Host without a
hardware counter. Pack and download counts are unchanged (3 uploads / 12
downloads per gate_up call). No cost is claimed as pure IRQ.

## 7. Team review and open issues

- Researcher confirmed the compile chain, decode/prefill selection, and that the
  policy generator is a real compile entry, not metadata.
- Reviewer required decode-only scope, 30-key completeness, measured prefill
  invariance, V5, and actual loaded-plan evidence. These are satisfied except:
  - policy generator is a new tool rather than the older tuner generator;
  - B was compiled from the current repository revision whose make_matmul_prim
    function body is unchanged from package, but whole-file source_hash differs;
  - full complete V5 bundle was not regenerated (numeric 56/56 passed);
  - only layers 0/15/29 were captured at op level, while model generate covers
    all 30 layers.
No production default was changed.

## 8. Recommendation

Merged-B is a credible **decode-only, opt-in** candidate: 5.3% lower generate
wall, 6.8% lower decode, prefill/TTFT unchanged, tokens identical, numeric V5
56/56. It costs ~30 MB RSS and ~1.5% load time. Do not change production default;
keep P. Adopt B only if the user accepts the decode-only scope and the pending
complete-V5-bundle/source-revision evidence.

## 9. Identity

- code: tools/compile_decode_merged_policy.py + existing policy path
- policy: merged-decode-gate-up-policy.json
  sha256 e4b1c22151d6e1898e211f6b5b8b8e0eba31ad42588c411b0546b2151e3cc8b7
- B plan: merged-decode-plans/gate_up_decode_merged.rkplan.json
  sha256 c99cbd24646c10b822218cea562beb750d03e98b6b0e7e1124a905b4bb316818
- result dir: results/20261031T040000Z-smollm-decode-b/
- replay:
  RK_SMOLLM_SCHEDULE_POLICY=<policy> scripts/run-smollm2.sh board-run <request> <out>
  and P without the env var.
