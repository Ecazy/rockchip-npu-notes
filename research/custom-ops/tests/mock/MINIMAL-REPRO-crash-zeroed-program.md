# Zeroed `co_program` crash — FIXED

Status: **fixed** (2026-10-22). Regression lives in
`research/custom-ops/tests/c/rk_plan_host_test.c`:
`mock_zero_program_case()` and `mock_untracked_program_case()`.

Original repro:

```c
co_program fake;
memset(&fake, 0, sizeof(fake));
fake.count = 1;
fake.nreads = 2;
co_execute_profiled(0, &fake, 1, CO_CHAIN, NULL, &profile);
```
→ `SIGSEGV`, exit 139, before any assertion.

Root cause: the seam resolved the template tag *after* it had already taken
`co_view` copies and dereferenced `buffer->bo.ptr` (the zeroed program has no
buffers at all).

Fix (in `tests/mock/npu_mock.c`, `__wrap_co_execute_profiled`), validation order
is now explicit:

1. the call itself: `p` non-null, `0 < count <= 64`, `mode` is CO_CHAIN/CO_SEPARATE
   → `ROCKET_E_SHAPE`;
2. per program: tag lookup first (unknown → `rk_mock_untracked++` and
   `ROCKET_E_SHAPE`, with **no** dereference);
3. then the views the template needs (`output`, `reads[0]`, and for EW
   `reads[1]`/`reads[2]`, each requiring a live `buffer` and `bo.ptr`);
4. only after all of that does any data read/write happen.

Covered by:

| scenario | expectation |
| --- | --- |
| zeroed program | `ROCKET_E_SHAPE`, `untracked == 1`, BO/resident unchanged |
| NULL / count==0 / bad mode | `ROCKET_E_SHAPE` |
| structurally valid but untracked (tags cleared by `rk_mock_reset()`) | `ROCKET_E_SHAPE`, `untracked == 1`, live BOs and resident bytes survive the reset |

Both run in the normal build (705 checks) and under ASan/UBSan with leak
detection on.
