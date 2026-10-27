#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare a compiled model package against a reference copy of the same package.

The model campaign tools rewrite ``models/compiled/`` in place, so any refactor
that claims "same plans" has to show what a full recompilation changed *and* that
the package it is comparing is still complete and self-consistent.

Three independent checks run over the union of both directories:

1. **file set** -- a file present on one side only is a failure.  A comparison that
   only walks the candidate cannot see deletions.
2. **content** -- every file is classified:

   ``identical``
       byte-identical JSON content.
   ``identity-only``
       differs only in ``plan_hash``/``source_hash``, which embed the address of the
       TVM ``SourceName`` object and are process volatile even before the refactor.
   ``identical-after-normalisation``
       after masking that address and dropping the derived ``plan_hash`` the whole
       content (bindings, actions, submits, publish, budgets, hybrid graph) is equal.
   ``identity-cascade``
       a package manifest whose only differing leaves are ``plan_hash`` /
       ``source_hash`` / ``hybrid_manifests.*.sha256`` **and** whose referenced
       files pass the consistency check below with identity-equivalent verdicts.
   ``content-diff``
       anything else: a real change that must be explained individually.

3. **reference consistency** (manifests only) -- every ``plans.<n>.path`` and
   ``hybrid_manifests.<n>.path`` must exist on both sides, every recorded
   ``plan_hash``/``source_hash`` must equal the referenced plan file's own value, and
   every ``hybrid_manifests.<n>.sha256`` must equal the SHA-256 of the referenced
   file bytes on both sides.  A stale or edited digest therefore fails instead of
   being waved through as an identity difference.

Exit status is non-zero if the file sets differ, any reference check fails, or any
file is classified ``content-diff``.

Usage:
    python3 tools/compare_compiled_models.py \
        --reference /tmp/regen/compiled --candidate models/compiled \
        --out results/<stamp>-refactor/model-package-regeneration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VOLATILE = re.compile(r"SourceName\((.*?), 0x[0-9a-fA-F]+\)")
DERIVED_LEAVES = {"plan_hash", "source_hash"}
CASCADE_DIGEST_PREFIX = "hybrid_manifests."
IDENTITY_VERDICTS = {"identical", "identity-only",
                     "identical-after-normalisation"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(value: Any) -> Any:
    """Mask the volatile SourceName address and drop derived plan hashes."""
    if isinstance(value, str):
        return VOLATILE.sub(lambda match: f"SourceName({match.group(1)})", value)
    if isinstance(value, dict):
        return {key: normalise(child) for key, child in value.items()
                if key not in DERIVED_LEAVES}
    if isinstance(value, list):
        return [normalise(child) for child in value]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            flat.update(_flatten(value[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flat.update(_flatten(child, f"{prefix}[{index}]"))
    else:
        flat[prefix] = VOLATILE.sub(
            lambda match: f"SourceName({match.group(1)})",
            value) if isinstance(value, str) else value
    return flat


def classify(reference: Any, candidate: Any) -> tuple[str, list[str]]:
    if reference == candidate:
        return "identical", []
    before, after = _flatten(reference), _flatten(candidate)
    differ = sorted(key for key in set(before) | set(after)
                    if before.get(key) != after.get(key))
    if all(path.split(".")[-1] in DERIVED_LEAVES for path in differ):
        return "identity-only", differ
    if normalise(reference) == normalise(candidate):
        return "identical-after-normalisation", differ
    return "content-diff", differ


def _is_cascade(differ: list[str]) -> bool:
    for path in differ:
        leaf = path.split(".")[-1]
        if leaf in DERIVED_LEAVES:
            continue
        if path.startswith(CASCADE_DIGEST_PREFIX) and leaf == "sha256":
            continue
        return False
    return bool(differ)


def verify_references(manifest: Any, package_dir: Path,
                      source_root: Path | None = None
                      ) -> tuple[list[str], list[str]]:
    """Check one manifest's references and digests against its own package.

    A manifest is only self-consistent against the directory it was written into;
    comparing it with the other side would just restate the identity difference.

    Returns ``(problems, skipped)``.  ``source_files`` entries are checked against
    ``source_root`` when it is given (or the package's repo-root parent).  A
    relocated snapshot cannot reach the original tree, so an unreachable source is
    reported as skipped rather than silently ignored -- pass ``--source-root`` to
    check both sides against the real tree.
    """
    problems: list[str] = []
    skipped: list[str] = []
    if not isinstance(manifest, dict):
        return problems
    plans = manifest.get("plans")
    if isinstance(plans, dict):
        for name, entry in plans.items():
            if not isinstance(entry, dict):
                continue
            relative = entry.get("path")
            if not relative:
                problems.append(f"plans.{name} has no path")
                continue
            path = package_dir / relative
            if not path.is_file():
                problems.append(f"plans.{name}.path missing: {relative}")
                continue
            try:
                plan = json.loads(path.read_text())
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"plans.{name}.path unreadable: {exc}")
                continue
            if plan.get("plan_hash") != entry.get("plan_hash"):
                problems.append(f"plans.{name}.plan_hash does not match its file")
            if "source_hash" in entry and \
                    plan.get("source_hash") != entry["source_hash"]:
                problems.append(f"plans.{name}.source_hash does not match its file")
    hybrids = manifest.get("hybrid_manifests")
    if isinstance(hybrids, dict):
        for name, entry in hybrids.items():
            if not isinstance(entry, dict):
                continue
            relative = entry.get("path")
            if not relative:
                problems.append(f"hybrid_manifests.{name} has no path")
                continue
            path = package_dir / relative
            if not path.is_file():
                problems.append(f"hybrid_manifests.{name}.path missing: {relative}")
                continue
            actual = sha256_file(path)
            if actual != entry.get("sha256"):
                problems.append(
                    f"hybrid_manifests.{name}.sha256 does not match its file "
                    f"({actual[:16]}...)")
    diagnostics = manifest.get("mlp_hidden_diagnostics")
    if isinstance(diagnostics, dict) and isinstance(plans, dict):
        for name, entry in diagnostics.items():
            if not isinstance(entry, dict):
                continue
            plan_entry = plans.get(name)
            if not isinstance(plan_entry, dict):
                problems.append(f"mlp_hidden_diagnostics.{name} has no plans entry")
                continue
            if entry.get("plan_hash") != plan_entry.get("plan_hash"):
                problems.append(
                    f"mlp_hidden_diagnostics.{name}.plan_hash disagrees with "
                    f"plans.{name}.plan_hash")
    sources = manifest.get("source_files")
    if isinstance(sources, dict):
        root = source_root if source_root is not None else package_dir.parents[1]
        for relative, digest in sources.items():
            path = root / relative
            if not path.is_file():
                skipped.append(f"source_files.{relative} not reachable under {root}")
            elif sha256_file(path) != digest:
                problems.append(f"source_files.{relative} digest does not match")
    return problems, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True,
                        help="reference package directory (e.g. a recompiled copy)")
    parser.add_argument("--candidate", required=True,
                        help="package directory to check")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", default=None,
                        help="tree that source_files paths are relative to; "
                             "defaults to each package's repo-root parent")
    args = parser.parse_args(argv)
    source_root = Path(args.source_root).resolve() if args.source_root else None

    reference_dir = Path(args.reference)
    candidate_dir = Path(args.candidate)
    if not reference_dir.is_dir():
        raise SystemExit(f"reference directory does not exist: {reference_dir}")
    if not candidate_dir.is_dir():
        raise SystemExit(f"candidate directory does not exist: {candidate_dir}")

    reference_files = {path.name for path in reference_dir.glob("*.json")}
    candidate_files = {path.name for path in candidate_dir.glob("*.json")}

    files: dict[str, Any] = {}
    counts: dict[str, int] = {}
    reference_problems: dict[str, list[str]] = {}
    unresolved_sources: dict[str, list[str]] = {}

    def record(name: str, verdict: str, differ: list[str] | None = None) -> None:
        files[name] = {"verdict": verdict, "differing_paths": differ or []}
        counts[verdict] = counts.get(verdict, 0) + 1

    for name in sorted(reference_files - candidate_files):
        record(name, "missing-candidate")
    for name in sorted(candidate_files - reference_files):
        record(name, "missing-reference")

    for name in sorted(reference_files & candidate_files):
        reference = json.loads((reference_dir / name).read_text())
        candidate = json.loads((candidate_dir / name).read_text())
        verdict, differ = classify(reference, candidate)
        # Each manifest is validated against the package it belongs to, so a
        # stale/edited digest fails while a pure identity change does not.
        candidate_problems, candidate_skipped = verify_references(
            candidate, candidate_dir, source_root)
        baseline_problems, baseline_skipped = verify_references(
            reference, reference_dir, source_root)
        problems = sorted(set(candidate_problems + baseline_problems))
        skipped = sorted(set(candidate_skipped + baseline_skipped))
        if skipped:
            unresolved_sources[name] = skipped
        if problems:
            reference_problems[name] = problems
        if verdict == "content-diff" and _is_cascade(differ) and not problems:
            referenced = set()
            for section in ("plans", "hybrid_manifests"):
                for entry in (candidate.get(section) or {}).values():
                    if isinstance(entry, dict) and entry.get("path"):
                        referenced.add(entry["path"])
            referenced &= set(files)
            unresolved = sorted(path for path in referenced
                                if files[path]["verdict"] not in IDENTITY_VERDICTS)
            if not unresolved:
                verdict = "identity-cascade"
            else:
                differ = differ + [f"unresolved-reference:{path}"
                                   for path in unresolved]
        record(name, verdict, differ)

    failures = sorted(
        name for name, entry in files.items()
        if entry["verdict"] in {"content-diff", "missing-candidate",
                                "missing-reference"} or name in reference_problems)

    report = {
        "kind": "model-package-regeneration",
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "counts": counts,
        "failures": failures,
        "reference_problems": reference_problems,
        "unresolved_source_checks": unresolved_sources,
        "reference_checks": [
            "both directories contain the same set of *.json files",
            "plans.<n>.path exists on both sides and its plan_hash/source_hash match "
            "the referenced plan file",
            "hybrid_manifests.<n>.path exists on both sides and its sha256 matches "
            "the reference and candidate file bytes",
            "mlp_hidden_diagnostics.<n>.plan_hash agrees with plans.<n>.plan_hash",
            "source_files.<path> exists (under --source-root when given) and its "
            "recorded digest matches; unreachable sources are listed in "
            "unresolved_source_checks instead of being silently ignored",
            "an identity-cascade manifest may only reference files whose own "
            "verdict is identity-equivalent",
        ],
        "normalised_fields": [
            "source_hash: the TVM SourceName object address is masked",
            "plan_hash: derived from source_hash, dropped before comparing content",
            "hybrid_manifests.<n>.sha256: accepted only when the referenced file's "
            "bytes really hash to the recorded value and that file is identity-only",
        ],
        "files": files,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"files={len(files)} {counts} failures={len(failures)} -> {out}")
    if failures:
        for name in failures[:10]:
            detail = reference_problems.get(name) or files[name]["verdict"]
            print(f"  FAIL {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
