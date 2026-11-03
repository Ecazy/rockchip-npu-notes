# Results: raw evidence layout

This directory holds raw experiment evidence: campaign directories, logs, probes, plan and manifest
files, hash lists and their replay scripts. It is not a status report and not an acceptance record.
What evidence may be cited, and how, is in [VALIDATION.md](../VALIDATION.md); supported domain is in
[SUPPORT-DOMAIN.md](../SUPPORT-DOMAIN.md).

## What is actually here

Naming and internal layout are **not** uniform across directories. Three shapes exist in practice:

| shape | example | typical content |
|---|---|---|
| campaign directory `<UTC timestamp>-<campaign>` | `20261023T144054Z-rk3588/`, `20261029T120416Z-smollm2-v5/` | per-stage subdirectories (`campaign/`, `board/`, `calibration/`, `acceptance/`, `recomputed/`), remote and local logs, sometimes a `manifest.json`, a `summary.md` and `SHA256SUMS` |
| review directory `codex-*` / `handoff-*` | `codex-frozen-review/`, `handoff-w02-lead-audit-r3/` | an independent review's probe scripts, `snapshot/` of the reviewed files, generated JSON/TIR/plan dumps and its logs |
| task/experiment directory | `20261105T040000Z-pass-pilot/`, `baseline-layout-reuse/` | focused probe scripts, plans, logs, `SHA256SUMS.txt` |

108 directories exist; 14 carry a `manifest.json`, 47 carry a `SHA256SUMS*` file and 26 carry a
`README.md`. Do not assume a file exists in a directory because a sibling directory has it.

## How to find evidence

1. **By date and campaign**: directory names start with a UTC timestamp (`YYYYMMDDTHHMMSSZ`) and a
   short campaign slug. The date is the run date, not the date of the code it exercised.
2. **By plan identity**: plan JSON files embed `plan_hash`; `final-hashes.txt`, `SHA256SUMS*` and
   `manifest.json` record file hashes. Grep for a hash to find the run that produced it.
3. **By source snapshot**: review directories keep a `snapshot/` of the code they reviewed, so a
   hash in the directory identifies the exact implementation it applies to.
4. **By replay script**: campaign directories usually keep the script that produced them
   (`reproduce.py`, `*_probe.py`, `make-*.log` names the pipeline). Re-running it regenerates that
   directory's outputs; do not edit the recorded results in place.

## Rules for citing this evidence

- Every claim taken from here must state the **directory**, the **date**, and the identity
  (code hash, `plan_hash`, or file hash). A result without identity is not evidence.
- Board results apply only to the plan that was executed on that run and to the code snapshot that
  produced it. They are not acceptance of later code.
- Files recorded in a `SHA256SUMS*` or `manifest.json` are frozen: do not rewrite their content or
  recompute their hashes to fit documentation changes. If a document is referenced by such a record,
  the document stays as part of the evidence even when it is also an old report.
- Some directories are ignored by Git and exist only in this working copy
  (`.data/`, large arrays, board bundles). Absence in a fresh clone is expected; the committed
  records that list them (`manifest.json`, `SHA256SUMS*`, `*-sha256` files) are the reference.
