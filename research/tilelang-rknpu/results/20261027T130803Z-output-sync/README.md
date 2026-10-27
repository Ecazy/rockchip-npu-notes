# Output-sync deduplication A/B artifacts

Report: [`RUNTIME-OUTPUT-SYNC.md`](../../RUNTIME-OUTPUT-SYNC.md).

- `campaign/`: exact compiled package identity, input cases, source hashes and
  a checksum index that excludes itself.
- `rounds/`: seven rotated QK/PV rounds, all per-arm `WARM_US` lines, exit
  codes and raw output bytes.
- `profile/` and `profile-summary.json`: four independent `battr` runs with
  32 `ATTR_US` samples each; `profile-analysis.py` reproduces the summary.
- `full-model/`: b8/b1 timings, 224/1,792 samples per arm, paired per-call
  ratios, paired per-round medians, all per-call bitwise checks, logits and
  stdout/stderr/exit records.
- `host/`: full relinked build, CTest 32/32, pytest 351 + 71 subtests and the
  real `execute_impl` output-sync counter test (40 checks). It also retains the
  targeted ASan/UBSan 1/1 log and finite-scan CLI compatibility checks.
- `artifacts/`, `build-logs/`, `board-environment.log`, `commands.log` and
  `host/source-hashes.sha256`: compile flags/commands, both library identities,
  board state, exact invocations and source identities.
- `SHA256SUMS` at this result root covers the collected artifact tree and
  excludes itself.

The frozen packed replay is
[`../20261027T131044Z-numeric-replay/`](../20261027T131044Z-numeric-replay/).
