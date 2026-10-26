# Repeat-path benchmark: separate vs B1 group vs B2 chain (per-task BOs)

Tool: `tools/benchmark_repeat_paths.py` (this run's `campaign.tgz` holds the exact
sources).  RK3588 core 1, `taskset -c 6`, 30 runs + 5 warm-up per variant, QK and
PV group of 32 tasks, real runtime, all paths checked bit-identical.

| plan | variant | p50 | per task | submits | guard | submit_wait | create |
|---|---|---:|---:|---:|---:|---:|---:|
| qk | separate | 7.210 ms | 225.1 us | 32 | 0.032 ms | 0.069 ms | 1.06 ms |
| qk | group (B1) | **2.240 ms** | 70.0 us | 32 | 0.030 ms | 0.459 ms | 1.03 ms |
| qk | chain (B2) | 3.379 ms | 105.6 us | **1** | **0.853 ms** | 0.243 ms | 11.78 ms |
| pv | separate | 5.498 ms | 171.0 us | 32 | 0.028 ms | 0.016 ms | 0.95 ms |
| pv | group (B1) | 4.136 ms | 129.1 us | 32 | 0.032 ms | 2.943 ms | 0.93 ms |
| pv | chain (B2) | **3.320 ms** | 103.6 us | **1** | **0.853 ms** | 0.246 ms | 12.23 ms |

`guard`/`submit_wait`/`create` are measured, not modelled.  `separate` reports one
call's attribution; `group`/`chain` report the whole group.

## What this settles

1. **B1 (one native group entry, N submits) is the best executable variant today**
   and it is already in the model: qk 7.21 -> 2.24 ms, pv 5.50 -> 4.14 ms.
2. **B2 with one BO per task loses to B1.**  It does remove the submits (32 -> 1,
   submit_wait 0.459 -> 0.243 ms on qk) and its wall time is much more stable
   (qk max 3.82 ms vs 4.79/8.36), but the 96-BO guard gate costs 0.853 ms per
   group and workspace creation costs 11.8-12.2 ms (one-off, 96 BO allocations).
   The guard alone is ~1.7 ms per batch-8 for the QK+PV pair, i.e. ~11% of the
   current 14.8 ms wall.
3. **A shared-BO chain would change the verdict on paper**: with three BOs per
   group the guard falls to ~0.03 ms, i.e. attributed C work of roughly
   0.68 ms against 1.27 ms for B1.  That is a projection, not a measurement, and
   it needs the verifier's writer/upload rules to become region-granular (today a
   buffer may only be uploaded by one host action, so N tiles in one BO are
   refused).
4. All three paths are bit-identical on both plans, so the measurement is of
   execution cost only.

Decision input: the per-task-BO form of B2 should not be wired into the model.
Either invest in region-granular validation plus shared allocation (projected
~1.2 ms per batch-8), or stop at B1 and move to the device-subgraph candidate.
