#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Seal a SmolLM2 board/reference observation bundle with explicit hashes."""
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


def add(files: dict, label: str, path: Path) -> None:
    if path.is_file():
        files[label] = {"sha256": sha(path), "bytes": path.stat().st_size}
    else:
        raise SystemExit(f"missing observation file: {path}")


def add_tree(files: dict, label: str, root: Path, pattern: str) -> None:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise SystemExit(f"no files matching {root / pattern}")
    for path in matches:
        rel = path.relative_to(root)
        add(files, f"{label}/{rel.as_posix()}", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("calibration", "acceptance"),
                    required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-tf", required=True)
    ap.add_argument("--official-logits", required=True)
    ap.add_argument("--ppl-board", default="")
    ap.add_argument("--ppl-official", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    board = Path(args.board).resolve()
    ref = Path(args.reference).resolve()
    reftf = Path(args.reference_tf).resolve()
    official = Path(args.official_logits).resolve()
    files: dict = {}
    add_tree(files, "board", board, "run-*/result.json")
    add(files, "board/run.json", board / "run.json")
    add(files, "board/stats.json", board / "stats.json")
    add(files, "board/run-00/captured-00.npz",
        board / "run-00" / "captured-00.npz")
    add(files, "board/run-00/all-logits-00.npy",
        board / "run-00" / "all-logits-00.npy")
    add(files, "board/run-01/logits-00.npy",
        board / "run-01" / "logits-00.npy")
    add(files, "board/run-02/incremental-logits-00.npy",
        board / "run-02" / "incremental-logits-00.npy")
    add(files, "reference/captured.npz", ref / "captured.npz")
    add(files, "reference/logits.npy", ref / "logits.npy")
    add(files, "reference/meta.json", ref / "meta.json")
    add(files, "reference_tf/logits.npy", reftf / "logits.npy")
    add(files, "reference_tf/meta.json", reftf / "meta.json")
    add(files, "official/logits.npy", official)
    official_result = official.parent / "result.json"
    add(files, "official/result.json", official_result)
    if args.ppl_board:
        ppl = Path(args.ppl_board).resolve()
        add_tree(files, "ppl_board", ppl, "run-*/result.json")
        add_tree(files, "ppl_board", ppl, "run-*/nll-*.npy")
        add(files, "ppl_board/stats.json", ppl / "stats.json")
        add(files, "ppl_board/run.json", ppl / "run.json")
    if args.ppl_official:
        op = Path(args.ppl_official).resolve()
        add(files, "ppl_official/result.json", op)
        add(files, "ppl_official/nll.npy", op.parent / "nll.npy")
    board_meta = json.loads((board / "run.json").read_text())
    env = board_meta.get("environment", {})
    manifest = {
        "schema_version": 1,
        "kind": "smollm2-observation-bundle",
        "role": args.role,
        "board_request_sha256": board_meta.get("request_sha256"),
        "package_manifest_sha256": env.get("package_manifest_sha256"),
        "package_weights_sha256": env.get("package_weights_sha256"),
        "files": files,
    }
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        if "canonical_content_sha256" in spec:
            manifest["spec_canonical_content_sha256"] = \
                spec["canonical_content_sha256"]
        elif "spec_sha256" in spec:
            manifest["spec_legacy_sha256"] = spec["spec_sha256"]
        manifest["spec_file_sha256"] = sha(Path(args.spec))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"role": args.role, "files": len(files),
                      "out": str(out),
                      "file_sha256": sha(out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
