# SmolLM2 V5 delivery latency and host-weight-cache report

All values are medians across independent 32-token generations. Per-forward values divide by the actual 31 decode forwards.

## Layered decode profile (profiled runs)

| Component | Baseline ms/forward | Candidate ms/forward |
|---|---:|---:|
| decode wall | 619.099 | 468.947 |
| embed (runner top) | 125.407 | 0.172 |
| 30 layers total | 436.013 | 411.954 |
| final RMSNorm | 0.871 | 0.156 |
| lm_head plans total | 55.701 | 55.877 |
| logits concat | 0.272 | 0.305 |
| top-level measured residual | 0.137 | 0.121 |
| PlanCall wrapper total | 327.085 | 332.750 |
| C runtime reported total | 316.254 | 321.635 |
| Host actions (cross-cut) | 88.427 | 89.159 |

## Layer subdivision (candidate, ms/forward)

| Operation | ms/forward |
|---|---:|
| attention_total | 141.973 |
| down_plan | 49.326 |
| gate_up_plan | 100.847 |
| kv_append | 12.020 |
| norm_input | 4.789 |
| norm_post | 4.658 |
| o_plan | 32.974 |
| qkv_plan | 43.072 |
| qkv_split | 2.127 |
| residual_attn | 2.231 |
| residual_ffn | 2.225 |
| rope | 8.971 |
| swiglu | 4.567 |
| measured layer residual | 2.427 |
| attention residual | 2.709 |

## PlanCall / C runtime seam (candidate, ms/forward)

| Segment | ms/forward |
|---|---:|
| plan_call.binding | 6.127 |
| plan_call.c_runtime | 332.750 |
| plan_call.inner_unattributed | 5.906 |
| plan_call.input_prepare | 32.779 |
| plan_call.lock_wait | 1.349 |
| plan_call.output_prepare | 2.317 |
| plan_call.wrapper_unattributed | 9.381 |
| C runtime reported total | 321.635 |
| C children sum | 317.968 |
| C measured residual | 3.660 |
| Python/ctypes seam (wrapper minus C total) | 11.115 |

C children:

| C interval | ms/forward |
|---|---:|
| alloc | 0.000 |
| command | 0.081 |
| download | 13.619 |
| emit | 0.000 |
| guard | 74.226 |
| output_sync | 10.196 |
| pack | 17.805 |
| publish | 0.966 |
| submit_alloc | 0.000 |
| submit_wait | 197.416 |
| validate | 2.674 |

## Interleaved final A/B (no profiler)

| Metric | Baseline | Candidate |
|---|---:|---:|
| TTFT median ms | 3870.855 | 3865.049 |
| generate wall median ms | 22711.064 | 17854.156 |
| decode median ms | 18861.493 | 14125.202 |
| decode ms/forward | 608.435 | 455.652 |
| decode forward/s | 1.644 | 2.195 |
| full-generate token/s | 1.409 | 1.792 |
| peak RSS median kB | 1237348 | 1243836 |
| final vmRSS median kB | 1182176 | 1233236 |

Generate-wall speedup: `1.272x`; decode speedup: `1.335x`; added peak RSS: `6488 kB`; added final vmRSS: `51060 kB`.

The profiler is a diagnostic run. The A/B table above is measured with profiling disabled; C-runtime child intervals are not added to the runner top-level totals.
