# Production-condition confirmation: P / B / D

Workload: real SmolLM2-135M gate/up, M=1/256, K=576, N=3072, first/mid/last
layer captures, ascending K=256/256/64, same weights/dtype/core mask/output
validation. RK3588, core mask 1, production cache defaults on.

P = actual packaged production gate_up plan (layer_00_gate_up_prefill/decode).
B = same production source (make_matmul_prim) with merged single submit.
D = same production source with K-interleaved order and data_reuse.
P is not scheduled_baseline; P is the shipped package plan.

## 1. Production identity and cache state

Measured plan identity:

| arm | source | tasks | submits | data BOs |
|---|---|---:|---:|---:|
| P | packaged layer.0.gate_up prefill/decode | 36 | 12 | 41 |
| B | make_matmul_prim + merged | 36 | 1 | 52 |
| D | make_matmul_prim + interleaved_reuse | 36 | 1 | 63 |

Production cache defaults are on. Actual create-time cache state:

| arm | nsubmits | program ready | command ready | multi-submit ready/cached | host bytes | device bytes |
|---|---:|---:|---:|---:|---:|---:|
| P | 12 | 0 | 0 | 1 / 12 | 2,390,780 | 5,491,264 |
| B | 1 | 1 | 1 | 0 / 0 | 2,305,596 | 7,834,880 |
| D | 1 | 1 | 1 | 0 / 0 | 2,305,596 | 9,637,824 |

P is structurally ineligible for program/command cache and hits the multi-submit
command cache 12/12. B/D are structurally ineligible for multi-submit cache and
hit program+command cache. This is production behavior, not a miss to equalize.
M=1 P host/device = 699,132 / 4,960,320; B = 613,948 / 5,929,728; D =
613,948 / 6,358,464.

Weights are not packed online: every arm reports 3 upload calls and only the
input slices (294,912 B at M=256; 13,824 B at M=1). Weight pack is creation-time.
The earlier claim that M=1 is online weight-pack dominated is withdrawn.

## 2. Register evidence

Decoder reads the real emitted co_program words, register 0x1040
CNA_CBUF_CON0, bit12=data_reuse, bit13=weight_reuse.

- P: all decoded CNA words 132/177, bit12/13 clear, 0 reuse tasks.
- B: all decoded CNA words 132/177, bit12/13 clear, 0 reuse tasks.
- D: exactly 33 bit12 tasks; raw 4228 (0x1084) for K blocks 0/1 and 4273
  (0x10B1) for the K=64 tail; 3 tasks clear.
- D task ordinals 0, 12, 24 (submit first task / K-block first task) are clear;
  reuse starts at ordinals 1..11, 13..23, 25..35.
- bit13 is clear for all P/B/D tasks.
- Descriptor flag count and decoded register count match: 0/0/33.
Diagnostic decode is outside the timing loop.

## 3. Correctness

Every warmup and timed call was validated outside the timer. All P/B/D outputs
are finite and elementwise bit-identical to P on every layer and both M.

## 4. Performance with production caches

10 rotated rounds, warmup 5 excluded, per-layer medians. Ratios are time(base)
/ time(candidate); >1 means candidate faster. Percent changes are separately
computed and not conflated with ratios.

M=256 (ms):

| layer | P | B | D | P/B | B/D | P/D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 10.346 | 10.540 | 10.494 | 0.989 | 0.987 | 0.991 |
| 15 | 19.822 | 22.851 | 22.506 | 0.889 | 1.016 | 0.859 |
| 29 | 10.196 | 10.970 | 9.879 | 0.932 | 1.110 | 1.038 |

M=1 (ms):

| layer | P | B | D | P/B | B/D | P/D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.322 | 3.487 | 3.729 | 1.243 | 0.958 | 1.145 |
| 15 | 3.271 | 2.295 | 2.425 | 1.477 | 0.948 | 1.400 |
| 29 | 2.606 | 2.161 | 2.333 | 1.249 | 0.933 | 1.162 |

Percent change vs P:
- M=256 B: -1.1% / -12.5% / -7.3% (all slower than P).
- M=256 D: -0.9% / -16.4% / +3.8% (slower L0/L15, faster L29).
- M=1 B: +19.5% / +29.8% / +17.1% faster.
- M=1 D: +13.7% / +25.9% / +10.5% faster, but B is faster than D at M=1.

## 5. Attribution

Representative M=256 layer15:
- P: submit_wait 13.75 ms, pack 0.78 ms, output_sync 0.37 ms, wall 19.82 ms.
- B: submit_wait 13.93 ms, pack 0.79 ms, output_sync 0.36 ms, wall 22.85 ms.
- D: submit_wait 13.53 ms, pack 0.78 ms, output_sync 0.26 ms, wall 22.51 ms.

M=1 layer15:
- P: submit_wait 2.67 ms, pack 0.025 ms, output_sync 0.13 ms, wall 3.27 ms.
- B: submit_wait 1.10 ms, pack 0.026 ms, output_sync 0.058 ms, wall 2.29 ms.
- D: submit_wait 1.16 ms, pack 0.022 ms, output_sync 0.102 ms, wall 2.43 ms.

submit_wait includes device execution; it is not all IRQ/Host. pack is online
input pack only. Downloads remain 12 per call for all arms. No hardware counter
was available to split submit_wait into device execution vs submit/IRQ. The
legacy report does not expose every host bucket; full attribution remains
partial.

## 6. Conclusions

1. Reuse does enter the expected register: P/B zero, D exactly 33 bit12 tasks,
   first/K-boundary zero, bit13 zero, descriptor/register counts match.
2. B is not a production win at M=256 under production caches: all three
   measured layers are slower than P. B is a clear M=1 win (17-30% faster).
3. D is not a production win at M=256 except layer29 (+3.8%); it is slower at
   L0/L15. D is faster than P at M=1 but slower than B there.
4. The prior cache-off A->B M=256 gain does not transfer to production P:
   P's 12-submit multi-submit cache is effective, and B's 1-submit
   program/command cache does not compensate.
5. Costs: B/D add 11 live partial BOs (device 7.83/9.64 MB vs P 5.49 MB).
   Initialization cost was not separately instrumented. D's extra device memory
   is the largest cost; no online weight-pack cost exists.
6. Measured gains/losses come from submit_wait and cache applicability; pack and
   output-sync count are unchanged. The remaining host envelope is not fully
   attributed.

Recommendation: do not change the production default. B is a candidate only for
M=1; D is not a general candidate and only showed a small M=256 layer29 gain.
Keep P as production. This is a recommendation only; no default was changed.

## 7. Identity and replay

- result dir: results/20261031T030000Z-linear-prod2/
- board lib: c6a6dd1f4c14508478740ca256431cd30f9aa07d6e4583727ec233c9a1b64709
- P plan: packaged layer_00_gate_up prefill/decode (41 BOs)
- B/D source: models/tilelang_matmul.py make_matmul_prim
- caches: production defaults; actual state recorded above
- register decoder: rk_workspace_task_cbuf, CNA_CBUF_CON0 0x1040 bit12/13
- replay: python3 tools/run_linear_schedule_experiment.py run --plans <dir> --warmup 5 --rounds 10 --only P,B,D --production-cache --out replay.json
- code/results manifest: MANIFEST.json
