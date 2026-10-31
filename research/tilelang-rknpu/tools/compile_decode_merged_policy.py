#!/usr/bin/env python3
"""Compile merged B plans for decode gate/up only and emit a model policy."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--layers", default=",".join(str(i) for i in range(30)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from rknpu_backend import (CompileOptions, compile_tilelang,
                               default_profile, load_plan, save_plan)
    from models.tilelang_matmul import make_matmul_prim
    pkg = Path(args.package).resolve()
    pm = json.loads((pkg / "manifest.json").read_text())
    prof = default_profile()
    opt = CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                         linear_task_order="merged")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plans_dir = out.parent / "merged-decode-plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    merged = compile_tilelang(make_matmul_prim(1, 576, 3072),
                              profile=prof, options=opt).plan
    if merged.accounting.task_count != 36 or merged.accounting.submit_count != 1:
        raise SystemExit("merged decode plan is not 36 tasks / 1 submit")
    bpath = plans_dir / "gate_up_decode_merged.rkplan.json"
    save_plan(merged, str(bpath))
    overrides = {}
    for layer in [int(x) for x in args.layers.split(",") if x]:
        key = f"layer.{layer}.gate_up.decode"
        base = load_plan(str(pkg / pm["plans"][key]["path"]))
        if base.accounting.task_count != 36 or base.accounting.submit_count != 12:
            raise SystemExit(f"{key}: base is not 36 tasks / 12 submits")
        overrides[key] = {
            "replacement_kind": "single_matmul_schedule_v1",
            "effective_policy": "merged",
            "base_plan_hash": base.plan_hash,
            "plan_hash": merged.plan_hash,
            "plan_sha256": sha(bpath),
            "task_count": merged.accounting.task_count,
            "submit_count": merged.accounting.submit_count,
            "data_bo_count": merged.accounting.data_bo_count,
            "cache_key": "decode_gate_up_merged_v1",
            "plan": merged.to_dict(),
        }
    policy = {
        "schema_version": 1,
        "kind": "smollm2-plan-overrides-v1",
        "source": "models/tilelang_matmul.py:make_matmul_prim + CompileOptions(linear_task_order=merged)",
        "scope": {"phase": "decode", "role": "gate_up", "layers": sorted(int(x) for x in args.layers.split(",") if x)},
        "compiler": {
            "numeric_mode": "rk_f16_tiled", "allow_approx_lut": False,
            "linear_task_order": "merged",
            "profile_hash": prof.profile_hash,
            "source_sha256": sha(ROOT / "models/tilelang_matmul.py"),
        },
        "plan": {"path": str(bpath.relative_to(out.parent)),
                 "sha256": sha(bpath), "plan_hash": merged.plan_hash,
                 "task_count": merged.accounting.task_count,
                 "submit_count": merged.accounting.submit_count,
                 "data_bo_count": merged.accounting.data_bo_count},
        "overrides": overrides,
    }
    out.write_text(json.dumps(policy, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"policy": str(out), "sha256": sha(out),
                      "override_count": len(overrides),
                      "plan": policy["plan"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
