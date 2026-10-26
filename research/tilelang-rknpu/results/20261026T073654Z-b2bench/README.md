# Repeat-path benchmark v2: interleaved, with shared-BO chain

Tool: `tools/benchmark_repeat_paths.py` (sources in `campaign.tgz`).
RK3588 core 1, `taskset -c 6`, **4 variants alternated round by round**, 30 rounds
+ 5 warm-up, QK and PV group of 32 tasks.  Attribution is the **median of
per-round aggregates over the whole group** (`separate` sums its 32 calls), not a
single sampled run.  All four paths are bit-identical on both plans.

| plan | variant | p50 | per task | submits | guard | submit_wait | create |
|---|---|---:|---:|---:|---:|---:|---:|
| qk | separate | 7.569 ms | 236.5 us | 32 | 0.976 ms | 2.093 ms | 1.10 ms |
| qk | group (B1) | 3.955 ms | 123.6 us | 32 | 0.032 ms | 1.995 ms | 0.65 ms |
| qk | chain, per-task BO | 3.918 ms | 122.4 us | 1 | 0.971 ms | 0.204 ms | 10.30 ms |
| qk | **chain, shared BO** | **2.930 ms** | **91.6 us** | 1 | 0.034 ms | 0.213 ms | 5.22 ms |
| pv | separate | 7.402 ms | 231.3 us | 32 | 0.945 ms | 2.124 ms | 0.94 ms |
| pv | group (B1) | 3.872 ms | 121.0 us | 32 | 0.030 ms | 2.009 ms | 0.47 ms |
| pv | chain, per-task BO | 4.044 ms | 126.4 us | 1 | 0.997 ms | 0.224 ms | 10.93 ms |
| pv | **chain, shared BO** | **3.090 ms** | **96.6 us** | 1 | 0.034 ms | 0.220 ms | 5.23 ms |

## Conclusions

1. **Both effects are real and separable**: the chained submit removes ~1.8 ms of
   submit cost per group (submit_wait 2.0 -> 0.21 ms) and the shared-BO layout
   removes ~0.94 ms of guard cost (0.97 -> 0.03 ms).  Either one alone cancels the
   other: the per-task-BO chain (3.92 ms) is no better than B1 (3.96 ms).
2. **`chain_shared` beats B1 by 1.03 ms (qk) and 0.78 ms (pv)**, i.e. about
   **1.8 ms per batch-8 for the QK+PV pair**, ~12% of the current 14.8 ms wall.
   Device memory for a group plan is 754 KB instead of 3.8 MB, and creation costs
   5.2 ms once per workspace instead of 10.3 ms.
3. Device domain per group: 3 BOs, 32 chained tasks, 64 uploads, 32 downloads;
   all outputs bit-identical to the separate path, so the frozen numeric policy
   applies unchanged to the shared plan (no arithmetic, layout or narrowing
   change).

Next: wire the shared group plan into the hybrid manifest for the QK/PV nodes
(keeping the separate and B1 paths as controls), then re-run the numeric replay
and the envelope campaign to confirm the end-to-end gain.
