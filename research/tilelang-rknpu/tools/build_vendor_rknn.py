#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the frozen ONNX comparison graphs with vendor RKNN-Toolkit2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rknn.api import RKNN

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(ROOT / "vendor" / "onnx"))
    ap.add_argument("--out-dir", default=str(ROOT / "vendor" / "rknn"))
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--single-core", action="store_true", default=True)
    args = ap.parse_args()
    onnx_dir = Path(args.onnx_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((onnx_dir / "export-manifest.json").read_text())
    rows = {}
    for key, entry in sorted(source["profiles"].items()):
        onnx_path = onnx_dir / entry["path"]
        rknn_path = out_dir / f"{key}.rknn"
        rknn = RKNN(verbose=False)
        try:
            rc = rknn.config(
                target_platform=args.target,
                float_dtype="float16",
                optimization_level=3,
                single_core_mode=bool(args.single_core),
                compress_weight=False,
            )
            if rc:
                raise RuntimeError(f"config failed {key}: {rc}")
            rc = rknn.load_onnx(model=str(onnx_path))
            if rc:
                raise RuntimeError(f"load_onnx failed {key}: {rc}")
            rc = rknn.build(do_quantization=False)
            if rc:
                raise RuntimeError(f"build failed {key}: {rc}")
            rc = rknn.export_rknn(str(rknn_path))
            if rc:
                raise RuntimeError(f"export failed {key}: {rc}")
        finally:
            rknn.release()
        rows[key] = {
            "model": entry["model"],
            "batch": entry["batch"],
            "onnx": entry["path"],
            "onnx_sha256": entry["onnx_sha256"],
            "rknn": rknn_path.name,
            "rknn_sha256": sha256(rknn_path),
            "rknn_bytes": rknn_path.stat().st_size,
            "target": args.target,
            "float_dtype": "float16",
            "quantization": "none",
            "single_core_mode": bool(args.single_core),
        }
        print(rknn_path, rows[key]["rknn_bytes"], flush=True)
    (out_dir / "rknn-manifest.json").write_text(
        json.dumps({"schema_version": 1, "profiles": rows},
                   indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
