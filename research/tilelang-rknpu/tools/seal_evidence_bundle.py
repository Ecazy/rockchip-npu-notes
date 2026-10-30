#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Seal an arbitrary SmolLM2 evidence directory tree with relative hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--role", choices=("calibration", "acceptance"),
                    required=True)
    ap.add_argument("--board-run", required=True)
    ap.add_argument("--spec", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    board_run = json.loads(Path(args.board_run).read_text())
    env = board_run.get("environment", {})
    files: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == out:
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
    if not files:
        raise SystemExit(f"no files under evidence root {root}")
    manifest = {
        "schema_version": 1,
        "kind": "smollm2-evidence-bundle",
        "role": args.role,
        "root_basename": root.name,
        "board_run_sha256": sha(Path(args.board_run)),
        "board_request_sha256": board_run.get("request_sha256"),
        "package_manifest_sha256": env.get("package_manifest_sha256"),
        "package_weights_sha256": env.get("package_weights_sha256"),
        "files": files,
    }
    if args.spec:
        spec_path = Path(args.spec).resolve()
        spec = json.loads(spec_path.read_text())
        if "canonical_content_sha256" in spec:
            manifest["spec_canonical_content_sha256"] = \
                spec["canonical_content_sha256"]
        elif "spec_sha256" in spec:
            manifest["spec_legacy_sha256"] = spec["spec_sha256"]
        manifest["spec_file_sha256"] = sha(spec_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "role": args.role, "root": str(root), "files": len(files),
        "manifest": str(out), "manifest_sha256": sha(out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
