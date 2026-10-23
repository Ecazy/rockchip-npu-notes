# Review closure — 2026-10-23

This closes the four review items whose evidence meanings were narrowed.  The
historical small-shape board result and its limitations are retained.

| # | Review item | Status | Code/evidence |
|---|---|---|---|
| 1 | `fp64_math` was rounded to FP16 on store/read, so it was not an unquantized math reference | **CLOSED** | Generator returns and stores `float64`; runner/verifier read `float64`; `tests/test_board_campaign.py` asserts the dtype is not representable unchanged in FP16; offline `tools/recompute_board_metrics.py` re-derived metrics from archived raw outputs in `results/20261023T103655Z-rk3588/recomputed/summary.md` |
| 2 | BO guard was checked at the first download, not at final publication after every submit | **CLOSED** | `rk_plan_runtime.c` checks every data BO with `co_guard_check` after all actions and before committing staged output; `tests/test_guard_failure.py` runs a legal `task A -> download A -> task B -> download B` mock plan where task B corrupts a guard and verifies `RK_E_DEVICE`, unchanged outputs and poisoning |
| 3 | Verifier over-claimed pair failures and hash verification | **CLOSED** | `tools/verify_board_campaign.py` makes byte-level chain/separate mismatches fatal (`dtype`, `shape`, `tobytes()`), recomputes `source-bundle.sha256`, and labels the remote loaded-library hash as `verified: false`; `tests/test_board_verifier.py` injects both a dtype-only pair mismatch and a tampered source bundle |
| 4 | Target parsing used substring matching and lost embedded NUL tokens | **CLOSED** | `src/rk_plan_target.c` matches one complete NUL-separated compatible item and the exact driver string; `tests/test_target_check.py` rejects `rk3588-extra`, `v0.9.80`, prefixed/suffixed versions and accepts the exact token |

## Retained limitations

- Board evidence is small-shape and uses three **serial single-core** masks.
- The ragged real-device case covers one tail shape (`M=13,N=33,K=257`), not all
  1..4096 shapes.
- The loaded remote library hash is a record from the board, not a Host-side
  rehash of the `.so` bytes.
- The target gate checks an exact chip token, exact locked driver string and
  runtime ABI.  It is not a full fleet/memory-contract qualification.
- Multi-board, concurrent three-core throughput, performance and long-duration
  stress are extension validation, not MVP/P8 blockers.
- P8 remains in progress for the remaining normative release gaps; this table
  closes only the four evidence/semantic review items.
