#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn a board tuning policy into constrained SmolLM2 plan overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rknpu_backend.plan import Plan


CASE_KEYS = {
    "prefill_qkv": lambda key: key.endswith(".qkv.prefill"),
    "decode_qkv": lambda key: key.endswith(".qkv.decode"),
    "prefill_gate_up": lambda key: key.endswith(".gate_up.prefill"),
    "prefill_down": lambda key: key.endswith(".down.prefill"),
    "prefill_lm_head": lambda key: key.startswith("lm_head.prefill."),
    "decode_lm_head": lambda key: key.startswith("lm_head.decode."),
    "attention_qk_prefill": lambda key: key == "attn.qk.prefill",
    "attention_qk_decode": lambda key: key == "attn.qk.decode",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binding_signature(plan: Plan) -> tuple:
    return tuple(sorted(
        (b.name, b.role, b.dtype, tuple(b.shape)) for b in plan.bindings))


def constant_signature(plan: Plan) -> tuple:
    return tuple(sorted(
        (c.name, c.dtype, tuple(c.shape), c.pack_key) for c in plan.constants))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-policy", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="autotuner")
    ap.add_argument("--select", action="append", default=[],
                    help="case:candidate override for an explicit identity test")
    args = ap.parse_args(argv)

    plans = Path(args.plans).resolve()
    package = Path(args.package).resolve()
    policy = load_json(Path(args.tuning_policy).resolve())
    manifest = load_json(plans / "manifest.json")
    package_manifest = load_json(package / "manifest.json")
    cases = {c["id"]: c for c in policy["cases"]}
    plan_cases = {c["id"]: c for c in manifest["cases"]}
    forced = {}
    for item in args.select:
        case_id, sep, candidate = item.partition(":")
        if not sep or not case_id or not candidate:
            raise SystemExit("--select must be case:candidate")
        forced[case_id] = candidate

    overrides: dict[str, dict] = {}
    provenance = []
    for case_id, pcase in cases.items():
        selected_name = forced.get(case_id, pcase["selected_candidate"])
        if forced and case_id not in forced:
            continue
        if not forced and selected_name == "default":
            continue
        case = plan_cases.get(case_id)
        if case is None:
            raise SystemExit(f"manifest lacks tuning case {case_id!r}")
        candidate = next((c for c in case["candidates"]
                          if c["name"] == selected_name), None)
        if candidate is None or not candidate.get("c_verifier_pass"):
            raise SystemExit(f"{case_id}: candidate {selected_name!r} is absent "
                             "or not C-verifier clean")
        source_case = case_id
        model_selector = CASE_KEYS.get(source_case)
        if model_selector is None and source_case == "lm_head_shard0":
            model_selector = CASE_KEYS["prefill_lm_head"]
        if model_selector is None:
            raise SystemExit(f"no model-plan mapping for tuning case {case_id!r}")
        keys = [key for key in package_manifest["plans"] if model_selector(key)]
        if not keys:
            raise SystemExit(f"no package plans match tuning case {case_id!r}")
        plan_path = plans / candidate["plan"]
        plan_dict = load_json(plan_path)
        replacement = Plan.from_dict(plan_dict)
        if replacement.plan_hash != candidate["plan_hash"]:
            raise SystemExit(f"{case_id}: selected plan hash changed")
        for key in keys:
            info = package_manifest["plans"][key]
            base = Plan.from_dict(load_json(package / info["path"]))
            if base.plan_hash != info["plan_hash"]:
                raise SystemExit(f"{key}: package base plan hash mismatch")
            if binding_signature(base) != binding_signature(replacement):
                raise SystemExit(f"{key}: replacement changes bindings")
            if constant_signature(base) != constant_signature(replacement):
                raise SystemExit(f"{key}: replacement changes constants")
            if base.profile_hash != replacement.profile_hash:
                raise SystemExit(f"{key}: replacement changes target profile")
            if any(a.kind == "npu_task" and a.template_id not in
                   {"co.mm.f16.v1", "co.mm_acc.f16.v1"}
                   for a in replacement.actions):
                raise SystemExit(f"{key}: replacement is not single-matmul")
            if key in overrides:
                raise SystemExit(f"duplicate model override key {key!r}")
            overrides[key] = {
                "replacement_kind": "single_matmul_schedule_v1",
                "base_plan_hash": base.plan_hash,
                "plan_hash": replacement.plan_hash,
                "plan_sha256": sha(plan_path),
                "task_count": int(replacement.accounting.task_count),
                "cache_key": pcase["cache_key"],
                "source_case": case_id,
                "source_candidate": selected_name,
                "source_plan": candidate["plan"],
                "source_schedule": candidate.get("source_schedule"),
                "applied_schedule": candidate.get("applied_schedule"),
                "source_k_rounding": candidate.get("source_k_rounding"),
                "plan": plan_dict,
            }
        provenance.append({
            "tuning_case": case_id,
            "candidate": selected_name,
            "plan": candidate["plan"],
            "plan_sha256": candidate["plan_sha256"],
            "plan_hash": candidate["plan_hash"],
            "task_count": candidate["task_count"],
            "source_schedule": candidate.get("source_schedule"),
            "applied_schedule": candidate.get("applied_schedule"),
            "source_k_rounding": candidate.get("source_k_rounding"),
            "model_plan_keys": keys,
        })

    result = {
        "schema_version": 1,
        "kind": "smollm2-plan-overrides-v1",
        "source": args.source,
        "package_manifest_sha256": sha(package / "manifest.json"),
        "package_weights_sha256":
            package_manifest["weights"]["file_sha256"],
        "tuning_policy": str(Path(args.tuning_policy).resolve()),
        "tuning_policy_sha256":
            sha(Path(args.tuning_policy).resolve()),
        "provenance": provenance,
        "overrides": overrides,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out),
        "sha256": sha(out),
        "override_count": len(overrides),
        "cases": [p["tuning_case"] for p in provenance],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
