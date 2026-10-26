# B2 chain validity probe (final evidence)

Probe: `custom-ops/tools/c/independent_mm_chain_probe.c`, per-task reference
comparison: `custom-ops/src/rk_task_ref.h`, host check of that comparison:
`custom-ops/tests/c/task_ref_test.c` (ctest `task_ref`).
Source hashes: `probe_source.sha256`. Commands: `commands.txt`.

## Evidence history (kept, not overwritten)

| file | state | why it is not authoritative |
|---|---|---|
| `probe_early_broken_comparator.log` | first run | every reference downloaded into one shared buffer, so only the last task's reference survived; all tasks were compared against it |
| `probe_intermediate_broken_control.log` | second run | per-task references fixed, but the negative control compared a buffer with itself so it always failed (`self_check_failed`) |
| `probe_final.log` | **authoritative** | per-task references, working negative control, full n = 2/8/16/32 matrix in both buffer layouts |

## Final result (RK3588 core 1, `taskset -c 6`, 10 repeats, sentinel-checked)

Per-task BOs (one input/weight/output BO per task) and shared BOs (three BOs
total, one tile per task by offset) both pass at n = 2, 8, 16, 32: 0 mismatched,
0 untouched, 0 non-finite, guard clean, and 0 mismatches when re-read after 20 ms.

Per-task cost (chain / separate): 24.2/40.9 us (n=2), 13.2/44.0 (n=8),
11.0/48.8 (n=16), 11.2/44.4 (n=32) for per-task BOs; 55.4/120.4, 18.1/70.0,
13.8/116.1, 10.8/80.6 for shared BOs. The separate side varies a lot between
rounds (44-236 us per task across runs), so only the order of magnitude is
claimed; the chained side is stable at 11-24 us per task.

## Scope of this evidence

It covers **one geometry** (16x16x16 matmul, `co.mm.f16.v1`, K within a single
tile) and n <= 32, with independent tasks that write distinct outputs and read
distinct (or shared) inputs. It does not license arbitrary MM chains: any other
geometry, task count or task mix needs its own admission, which is why the
expanded plan path must stay explicitly limited and keep a separate-submit
control.
