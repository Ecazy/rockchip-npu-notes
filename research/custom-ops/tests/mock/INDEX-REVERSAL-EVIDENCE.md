# EW weight-index reversal check (evidence)

Purpose: prove the non-diagonal orientation regression is sensitive to the weight
read order, i.e. it is not a tautology.

Method (performed 2026-10-22, in `research/custom-ops`):

1. copy `tests/mock/npu_mock.c` to a scratch copy;
2. replace `ident[co_index(&id.tensor, c, q)]` with `ident[co_index(&id.tensor, q, c)]`
   (a single-line reversal, marked `TEMP: reversed` in the working tree);
3. `cmake --build build-host-mvp -j4 --target rk_plan_host_test`;
4. `MALLOC_PERTURB_=128 ./build-host-mvp/rk_plan_host_test`;
5. restore the file from the scratch copy, rebuild everything, rerun.

Observed (actual stdout of those runs):

```
--- with reversed (q,c) index ---
RK_PLAN host checks=690 failures=30
--- restored (c,q) index ---
RK_PLAN host checks=690 failures=0
```

Limitations of this evidence, stated explicitly:

* the two runs above were captured from **stdout only**; no log file was archived
  at the time, and none has been reconstructed afterwards;
* the reversal patch is not committed: the file was restored from the scratch copy
  immediately after step 5, so `git diff` shows the correct `(c, q)` form.

Reproducing it is mechanical: apply the one-line change described above, rebuild
`rk_plan_host_test`, run it, and expect ~30 failures; revert and expect 0.
