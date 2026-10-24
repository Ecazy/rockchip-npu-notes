#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the two frozen model graphs from real TileLang source to RKPlan."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from rknpu_backend import CompileOptions, compile_tilelang, default_profile, save_plan

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transformer_shapes(batch: int):
    m = batch * 16
    return {
        "patch": (m, 64, 64, False),
        "qkv": (m, 64, 192, False),
        "attn_out": (m, 64, 64, False),
        "gate": (m, 64, 128, False),
        "up": (m, 64, 128, False),
        "down": (m, 128, 64, False),
        "classifier": (batch, 64, 16, False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "compiled"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)

    import models.tilelang_matmul as tmm
    import models.tilelang_mlp as tmlp

    manifest = {
        "schema_version": 1,
        "profile_hash": profile.profile_hash,
        "compiler": "real TileLang PrimFunc -> rknpu_backend.compile_tilelang",
        "numeric_mode": options.numeric_mode,
        "allow_approx_lut": options.allow_approx_lut,
        "plans": {},
        "source_files": {
            "models/tilelang_mlp.py": sha256(ROOT / "models/tilelang_mlp.py"),
            "models/tilelang_matmul.py": sha256(ROOT / "models/tilelang_matmul.py"),
        },
    }

    def write(plan, name):
        path = out / f"{name}.rkplan.json"
        save_plan(plan, str(path))
        manifest["plans"][name] = {
            "path": path.name, "plan_hash": plan.plan_hash,
            "source_hash": plan.source_hash, "batch": plan.bindings[0].shape[0],
        }
        return path

    for batch in (1, 8, 32):
        plan = compile_tilelang(tmlp.make_mlp_prim(batch), profile=profile,
                                options=options).plan
        write(plan, f"mnist_mlp_b{batch}")
        print("wrote", f"mnist_mlp_b{batch}", plan.plan_hash, flush=True)

    probe = compile_tilelang(tmlp.make_relu_probe_prim(), profile=profile,
                             options=options).plan
    templates = [a.template_id for a in probe.actions
                 if a.kind == "npu_task"]
    if templates != ["co.ew_add.f16.v1", "co.ew_max.f16.v1"]:
        raise SystemExit(f"unexpected ReLU probe templates: {templates}")
    write(probe, "mnist_relu_probe")
    manifest["plans"]["mnist_relu_probe"].update({
        "role": "numeric-acceptance-relu-probe",
        "templates": templates,
        "constant_inputs": ["X", "B"],
    })
    print("wrote mnist_relu_probe", probe.plan_hash, flush=True)

    for batch in (1, 8):
        plans = {}
        for name, (m, k, n, binp) in transformer_shapes(batch).items():
            plans[name] = compile_tilelang(
                tmm.make_matmul_prim(m, k, n), profile=profile,
                options=options, b_is_input=binp).plan
        # QK and PV use the same 16x16 dynamic-B primitive; one immutable plan is
        # intentionally reused by the model runner for both operations.
        plans["qk"] = compile_tilelang(tmm.make_matmul_prim(16, 16, 16),
                                       profile=profile, options=options,
                                       b_is_input=True).plan
        plans["pv"] = plans["qk"]
        for name, plan in plans.items():
            write(plan, f"tiny_vit_b{batch}_{name}")
        print("wrote", f"tiny_vit_b{batch}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "tools" / "build_hybrid_manifests.py"),
        "--plans-dir", str(out),
    ], check=True)
    print(out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
