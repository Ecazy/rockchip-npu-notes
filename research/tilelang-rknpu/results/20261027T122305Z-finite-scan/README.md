# Finite-scan A/B artifacts

Primary report: [`RUNTIME-FINITE-SCAN.md`](../../RUNTIME-FINITE-SCAN.md).

This directory contains the reproducible board evidence for the binary16
finite-scan change:

- `campaign/` contains the frozen compiled plans, weights, deterministic QK/PV
  cases, run manifest and package checksums. `cases/` retains the input frame
  bundles; `rounds/` contains all 7 × 2 arm logs, exit codes, output bytes and
  raw `WARM_US` samples.
- `full-model/b8/` and `full-model/b1/` contain each arm's timing JSON, run
  stdout/stderr/exit code and bitwise-compared logits.
- `full-model-raw/b8/` and `full-model-raw/b1/` are the subsequent raw-sample
  captures used by the report: 224 paired calls for b8 and 1,792 for b1, with
  per-call arm latencies, paired ratios, per-round medians and logits. The
  failed wrapper-only attempt is retained under `full-model-raw/failed-attempt-1/`.
- `artifacts/` contains both libraries' hashes, CMake caches, compile command
  lists, runtime target flags, native probe and the original and independent
  compile-flag audit outputs. The original comparison's `MISMATCH` and the
  first offline whitespace-normalization attempt are retained;
  `compile-flag-comparison-rechecked.json` independently verifies that the
  reference macro is the only compile difference across all 63 compiler commands.
- `host/` contains the exact relinked Host build, pytest and CTest logs and
  commands, plus the focused ASan/UBSan `fp16_scan` and `rk_plan_host` result.
  `board-environment.log`, `commands.log`, `build-logs/` and
  `source-snapshot/board-worker.sh` record the board conditions, execution and
  deployed worker source.
- `analysis.json` is the machine-readable isolated-case summary. The primary
  full-model results are in each batch's `result.json`.

The board run used a staged source snapshot whose identities are in
`campaign/run-manifest.json`. The acceptance replay is separately preserved in
[`../20261027T122757Z-numeric-replay/`](../20261027T122757Z-numeric-replay/).
The initial frequency-path check failure and the first numeric replay filename
failure are preserved in
[`../20261027T122026Z-finite-scan/`](../20261027T122026Z-finite-scan/) and
[`../20261027T122531Z-numeric-replay/`](../20261027T122531Z-numeric-replay/);
neither contains a claimed performance or numeric pass.

`SHA256SUMS.self-including` preserves the first generated index. The corrected
`campaign/SHA256SUMS` excludes itself, and the result-root `SHA256SUMS` covers
the collected artifact tree while excluding itself. Large libraries, weights,
input frames and archives remain local and are identified by the inventories.
Small output byte dumps, logits and the numeric acceptance observation are
retained in Git with the timing and verifier evidence.
