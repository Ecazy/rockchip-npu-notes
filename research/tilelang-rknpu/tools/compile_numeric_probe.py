#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the real-TileLang exact-ReLU numeric probe into the frozen package."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from rknpu_backend import (CompileOptions, compile_tilelang, default_profile,
                           save_plan)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default=str(ROOT / "models" / "compiled"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    args = ap.parse_args()
    out = Path(args.plans_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    from models.tilelang_mlp import make_relu_probe_prim
    profile = default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled",
                             allow_approx_lut=False)
    result = compile_tilelang(make_relu_probe_prim(), profile=profile,
                              options=options)
    plan = result.plan
    templates = [a.template_id for a in plan.actions if a.kind == "npu_task"]
    if templates != ["co.ew_add.f16.v1", "co.ew_max.f16.v1"]:
        raise SystemExit(f"unexpected ReLU probe template sequence: {templates}")
    path = out / "mnist_relu_probe.rkplan.json"
    save_plan(plan, str(path))
    manifest["plans"]["mnist_relu_probe"] = {
        "path": path.name, "plan_hash": plan.plan_hash,
        "source_hash": plan.source_hash, "batch": 2,
        "role": "numeric-acceptance-relu-probe",
        "templates": templates,
        "constant_inputs": ["X", "B"],
    }
    manifest["source_files"]["models/tilelang_mlp.py"] = sha256(
        ROOT / "models" / "tilelang_mlp.py")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "tools" / "build_hybrid_manifests.py"),
        "--plans-dir", str(out), "--weights-dir", str(Path(args.weights_dir).resolve()),
    ], check=True)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
