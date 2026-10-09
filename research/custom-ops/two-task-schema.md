# Two-Task experiment evidence schema (v1)

Four arms use ONLY standalone FP16 HardSigmoid DPU/DPU_RDMA programs:
A: one program/one submit; B: two independent programs/two submits;
C: two independent programs/one chained submit;
D: two dependent programs/one chained submit (task1 reads task0's output cube).
Each round executes A, B, C, D in that order: 5 submits, 7 tasks.
Shapes rotate (16,32), (29,45), (64,64); inputs alternate and include special zero inputs.
Processes are pinned to core masks 1,2,4. Two S_POINTER modes:
auto=0xE (ping-pong), fixed=0x30 (both pointers clear, ping-pong off, group0).

## Files per process

- `probe.jsonl`: one `round` record per round and a final `summary`.
  round: `round`, `rows`, `cols`, `pattern`, `pass`, `checks`, `arms` list.
  each arm: `arm` (A/B/C/D), `submits`, `tasks`, `irq_delta` (3 ints),
  `checks`, `max_abs`, `pass`; only selected core's IRQ increments, with
  A/B/C/D expected counts 1/2/1/1. Core order follows the three NPU IRQ rows.
  summary: `rounds`, `checks`, `failures`, `core_mask`, `sptr`, `status`.
- `trace.jsonl`: LD_PRELOAD observer records with ordered monotonically increasing `event_id`.
  ALL addresses/object IDs are integer numbers, not hex strings. `fd` present on ioctl records.
  `create`: `id` unique allocation generation, `handle`, `obj`, `dma`, `size`, `domain`.
  `map`: `id`, `offset` (MEM_MAP returned offset).
  `mmap`: `id`, `address`, `length`.
  `unmap`: `id`, `address`, `length`.
  `destroy`: `id`, `obj`, `result` (0 means success).
  `sync`: `id`, `offset`, `size`, `flags`, `result`.
  `stage`: `round`, `arm` (set BEFORE each arm; B has two submits in same stage).
  `submit_begin`: `submit_id`, `round`, `arm`, `flags`, `count`, `task_obj`,
  `task_id` (descriptor allocation generation), `task_offset`, `task_bytes`,
  `task_size` (must be 40), `task_start`, `core_mask`, `domain`, `base` (PC DMA base, 0).
  `program`: `submit_id`, `index`, `desc_offset` (offset inside task BO), `desc_bytes` (40),
  `regcmd`, `amount` (descriptor configuration words, excludes 4 trailer words),
  `words` (amount+4), `rc_id`, `rc_offset`, `fetch_bytes` (even-rounded words * 8),
  `int_mask` (0x300), `sptr_dpu`, `sptr_rdma`, `src`, `dst`, `src_id`, `dst_id`,
  `data_bytes`, `next` (0 on last), `next_amount`, `enable` (0x18), `valid`.
  These fields are derived by reading ACTUAL mapped descriptor/regcmd memory at submit.
  `submit_end`: `submit_id`, `result`, `errno`, `wall_ns`, `hw_ns`,
  `last_status` (ACTUAL driver-written last-task int_status), `live` (all referenced
  descriptor/regcmd/input/output allocation generations remain live through return).
  Also `descriptor_guard`: the entire mapped task BO is unchanged except the
  last task's int_status four bytes, verified against a pre-submit CPU snapshot.
  `subcore`: `submit_id`, `index`, `start`, `count` records the actual selected
  core's effective task-array range and must agree with the submit descriptor range.
  `end`: `live_allocations` (must be 0 after BO frees and rocket_close).
- `program-*.bin`: raw regcmd witnesses for the first 15 submits, for independent decoding.
- `probe.log`: readable numerical detail and failure messages.
- Each arm also records `padding_guard=true` after scoring every padded physical
  cube lane (one H(0), or two H(H(0)) for D's final output).
- First three rounds include `round-NNNN-input-a/b.bin` and all
  `round-NNNN-arm-A/B/C/D-task-0/1.bin` logical FP16 tensor witnesses; independently
  evaluate the mathematical reference, including the intermediate FP16 rounding.
- Campaign archives trace.jsonl as trace.jsonl.gz after execution, losslessly.

## Verifier obligations

Use stdlib only. A standalone function `verify_process(path)` should reject malformed,
missing or contradictory data, not trust the final PASS string or task_counter echo.
Check each round all 4 arms, counts above, IRQ deltas, numerical pass/finiteness,
all round checks sum to summary, patterns alternate, >1 round, shapes vary.
Check actual descriptor 40-byte stride, boundaries within known live task BO;
regcmd address = known live rc DMA + offset, 16-byte alignment; even-rounded fetch
window within rc BO. For chained C/D the programs are contiguous at rounded word
stride, forward links point to actual next descriptor regcmd, next_amount encodes
NEXT program length ((next_words+1)//2-1). Last link zero. No extra program fetched.
S_POINTER fields match declared process mode; source/destination spans fit live
domain-matched BOs. A/B/C have distinct input/output allocations; C independent
task sources differ; D task1.src equals task0.dst (same allocation AND exact address).
Check every submit began and returned successfully, hw_ns>0, wall_ns<1500ms,
last_status has DPU completion (0x100/0x200 bits only), lifetimes and sync range bounds.
Collect ACTUAL IRQ last_status bank bits: fixed must use group0 (0x100); auto B's
two one-task submits must alternate 0x100/0x200, and A must alternate across rounds
(7 auto tasks per round). Chained last_status may contain both completion bits;
do not invent hardware group reads. All references survive submit_end; destroys/unmaps
must not occur during active submit. End allocation count and reconstructed live set zero.
Verify captured raw words against the program metadata where `.bin` witnesses exist.
Generate a process summary with evidence SHA256, diagnostics and errors; nonzero exit on failure.
Implement meaningful unit tests for corrupted next link, descriptor overflow,
early destroy, bad IRQ/completion, group mismatch and truncated/missing round evidence.

For campaign: 12 processes (3 masks x 2 sptr modes x 2 repeated runs), serial,
128 rounds/process; kernel journal cursor spans all runs and must show no NPU faults.
No host writes or reads of D's intermediate between the two tasks.
