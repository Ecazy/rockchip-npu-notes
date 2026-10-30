#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the delivery manifest for the SmolLM2 V5 closure round."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def tree(root: Path, exclude: set[Path]) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve() in exclude:
            continue
        out[p.relative_to(root).as_posix()] = {
            "sha256": sha(p), "bytes": p.stat().st_size}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--hash-record", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    r = Path(args.results).resolve()
    raw = Path(args.raw).resolve()
    package = Path(args.package).resolve()
    spec = Path(args.spec).resolve()
    record = read(Path(args.hash_record))
    rel = spec.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    spec_id = record[rel]
    pm = read(package / "manifest.json")
    reuse = read(r / "reuse/replay-reuse.json")
    fixed_path = r / "rerun-fixed-acceptance/results/replay-rerun-fixed.json"
    fixed = read(fixed_path) if fixed_path.is_file() else None
    latency = read(r / "latency.json")
    equiv = read(r / "equivalence.json")
    div = read(r / "divergence/diagnostic.json")
    archived_manifest = read(
        Path(__file__).resolve().parents[1] /
        "results/20261029T120416Z-smollm2-v5/manifest.json")
    code_paths = [
        "python/rknpu_backend/models.py",
        "python/rknpu_backend/smollm.py",
        "tools/run_smollm2.py",
        "tools/smollm2_official.py",
        "tools/diagnose_smollm2_divergence.py",
        "tools/analyze_smollm2_delivery.py",
        "tools/verify_smollm2_host_cache_equivalence.py",
        "tools/replay_smollm2_v5.py",
        "scripts/replay-smollm2-v5.sh",
        "scripts/run-smollm2.sh",
        "tests/test_smollm2_host_weight_cache.py",
        "tests/test_diagnose_smollm2_divergence.py",
        "tests/test_replay_smollm2_v5.py",
    ]
    board_env = read(r / "board-profile-candidate/run.json")["environment"]
    out_path = Path(args.out) if args.out else r / "DELIVERY-MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "kind": "smollm2-v5-delivery-manifest",
        "git": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).decode().strip(),
            "dirty_paths": subprocess.check_output(
                ["git", "status", "--porcelain"]).decode().splitlines(),
        },
        "code_sha256": {p: sha(Path(p)) for p in code_paths},
        "model": {
            "id": pm["model_id"],
            "revision": pm["revision"],
            "package_manifest_sha256": sha(package / "manifest.json"),
            "weights_file_sha256": pm["weights"]["file_sha256"],
            "model_files_sha256": pm["model_files"],
        },
        "spec": {
            "path": str(spec),
            "file_sha256": spec_id["file_sha256"],
            "canonical_content_sha256": spec_id["canonical_content_sha256"],
            "independent_record": str(Path(args.hash_record).resolve()),
        },
        "inputs": {
            "manifest": str(Path(args.inputs).resolve()),
            "manifest_sha256": sha(Path(args.inputs)),
            "request_dir": str(Path(args.requests).resolve()),
            "request_sha256": {
                p.name: sha(p) for p in sorted(Path(args.requests).glob("*.json"))
            },
        },
        "environment": board_env,
        "raw_evidence_root": str(raw),
        "reuse_verification": reuse,
        "fixed_acceptance_rerun": fixed,
        "host_weight_cache": {
            "equivalence": {k: equiv[k] for k in
                            ("pass", "numeric_arrays_compared",
                             "numeric_arrays_equal", "failures")},
            "latency": latency,
        },
        "divergence_diagnostic": {
            "pass": div["pass"],
            "prompts_checked": div["prompts_checked"],
            "divergent_prompts": div["divergent_prompts"],
            "archived_official_matching_positions":
                div["archived_official_matching_positions"],
            "corrected_official_matching_positions":
                div["corrected_official_matching_positions"],
            "classifications": div["classifications"],
            "official_auto_mask_is_pad_token_mask":
                div["official_auto_mask_is_pad_token_mask"],
            "failures": div["failures"],
        },
        "historical_failure_preservation": [
            {k: row.get(k) for k in
             ("policy", "role", "pass", "gate_failures", "spec_path")}
            for row in archived_manifest.get("policy_history", {}).get(
                "entries", [])
        ],
        "explicit_non_claims": [
            "single board, core mask 1, batch 1, capacity 256, FP16",
            "Host KV orchestration, not a generic device-resident KV API",
            "no quantization, multi-request, multi-core or long-context claim",
            "no post-device-fault recovery or RK1228 qualification",
            "corrected official generation is a diagnostic reference, not a "
            "change to the V5 gate or archived observation",
        ],
        "files": tree(r, {out_path.resolve()}),
    }
    out_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "out_sha256": sha(out_path),
        "files": len(manifest["files"]),
        "reuse_pass": reuse.get("pass"),
        "fixed_rerun_pass": None if fixed is None else fixed.get("pass"),
        "equivalence_pass": equiv.get("pass"),
        "divergence_pass": div.get("pass"),
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
