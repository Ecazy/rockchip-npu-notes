#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build budgeted MLP plan campaigns for legal multi-submit cache evaluation.

Each variant starts from the frozen compiled package and replaces only the MLP
b1/b8/b32 plan + hybrid manifest plan hash.  Budgets are explicit CompileOptions
values persisted in the plan; old plans without budget_policy remain strict.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.serialize import save_plan
from models.tilelang_mlp import make_mlp_prim


def compile_mlp(batch: int, device_budget: int, host_budget: int,
                fuse_epilogue: bool = False, fuse_add_relu: bool = False):
    options = CompileOptions(
        numeric_mode="rk_f16_tiled", allow_approx_lut=False,
        device_budget_bytes=device_budget, host_budget_bytes=host_budget,
        fuse_epilogue=fuse_epilogue, fuse_add_relu=fuse_add_relu)
    return compile_tilelang(make_mlp_prim(batch), profile=default_profile(),
                            options=options).plan


def patch_manifest(compiled: Path, out: Path,
                   budgets: dict[int, tuple[int, int]],
                   fuse_epilogue: bool = False,
                   fuse_add_relu: bool = False):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src in compiled.iterdir():
        if src.is_file():
            shutil.copy2(src, out / src.name)

    # First compile loose plans to learn the base estimates, then recompile with
    # the requested explicit budgets.
    loose = {b: compile_mlp(b, 256 * 1024 * 1024, 256 * 1024 * 1024,
                            fuse_epilogue, fuse_add_relu)
             for b in budgets}
    package_manifest = json.loads((compiled / "manifest.json").read_text())
    for batch, (device_budget, host_budget) in budgets.items():
        if device_budget is None:
            device_budget = loose[batch].accounting.device_request_bytes
        if host_budget is None:
            host_budget = loose[batch].accounting.host_bytes
        plan = compile_mlp(batch, device_budget, host_budget, fuse_epilogue,
                           fuse_add_relu)
        # Accounting must not move when only the authorized budget changes.
        if (plan.accounting.device_request_bytes !=
                loose[batch].accounting.device_request_bytes or
            plan.accounting.host_bytes != loose[batch].accounting.host_bytes):
            raise SystemExit(f"b{batch}: budget changed accounting estimates")
        name = f"mnist_mlp_b{batch}"
        save_plan(plan, str(out / f"{name}.rkplan.json"))
        hybrid_path = out / f"{name}.rkhybrid.json"
        hybrid = json.loads(hybrid_path.read_text())
        hybrid["plans"]["mlp"]["path"] = f"{name}.rkplan.json"
        hybrid["plans"]["mlp"]["plan_hash"] = plan.plan_hash
        hybrid["profile_hash"] = plan.profile_hash
        # A primitive fusion can remove lowering-internal constants (for
        # example the zero tensor of a fused ReLU).  Keep the node constant
        # declaration exactly in sync with the new plan.
        node = next((n for n in hybrid["nodes"]
                     if getattr(n, "get", lambda *_: None)("plan") == "mlp"),
                    None)
        if node is not None:
            names = {spec.name for spec in plan.constants}
            declared = node.get("constants", {})
            node["constants"] = {k: v for k, v in declared.items()
                                 if k in names}
        hybrid_path.write_text(json.dumps(hybrid, indent=2, sort_keys=True) + "\n")
        if name in package_manifest.get("plans", {}):
            package_manifest["plans"][name]["plan_hash"] = plan.plan_hash
        print(name, "device", device_budget, "host", host_budget,
              "plan", plan.plan_hash[:16])
    (out / "manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", type=Path, default=ROOT / "models" / "compiled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fuse-epilogue", action="store_true")
    ap.add_argument("--fuse-add-relu", action="store_true")
    args = ap.parse_args()
    # Conservative exact budget; partial: 4 KiB request headroom (first submit
    # command request is 4,032 B); full: the documented CompileOptions default.
    variants = {
        "exact": {b: (None, None) for b in (1, 8, 32)},
        "partial": {b: (None, None) for b in (1, 8, 32)},
        "full": {b: (256 * 1024 * 1024, 256 * 1024 * 1024) for b in (1, 8, 32)},
    }
    # Fill exact/partial from loose accounting below; use a second pass.
    for name, budgets in variants.items():
        out = args.out / name
        if name == "exact":
            levels = {b: (None, None) for b in (1, 8, 32)}
        elif name == "partial":
            levels = {b: (None, 256 * 1024 * 1024) for b in (1, 8, 32)}
        else:
            levels = budgets
        patch_manifest(args.compiled, out, levels, args.fuse_epilogue,
                       args.fuse_add_relu)
        if name == "partial":
            # Rebuild partial with +4 KiB device request headroom.
            for batch in (1, 8, 32):
                plan_path = out / f"mnist_mlp_b{batch}.rkplan.json"
                plan = json.loads(plan_path.read_text())
                base = plan["accounting"]["device_request_bytes"]
                host = 256 * 1024 * 1024
                plan2 = compile_mlp(batch, base + 4096, host,
                                    args.fuse_epilogue, args.fuse_add_relu)
                save_plan(plan2, str(plan_path))
                hybrid_path = out / f"mnist_mlp_b{batch}.rkhybrid.json"
                hybrid = json.loads(hybrid_path.read_text())
                hybrid["plans"]["mlp"]["plan_hash"] = plan2.plan_hash
                hybrid_path.write_text(json.dumps(hybrid, indent=2, sort_keys=True) + "\n")
                manifest_path = out / "manifest.json"
                package_manifest = json.loads(manifest_path.read_text())
                package_manifest["plans"][f"mnist_mlp_b{batch}"]["plan_hash"] = plan2.plan_hash
                manifest_path.write_text(
                    json.dumps(package_manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
