#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract representative GEMM/attention geometries from frozen hybrid plans."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rknpu_backend.hybrid import load_hybrid_program

CASES = (
    ("mnist_mlp_b1", "mlp"),
    ("mnist_mlp_b8", "mlp"),
    ("mnist_mlp_b32", "mlp"),
    ("tiny_vit_b1", "qkv"),
    ("tiny_vit_b8", "qkv"),
    ("tiny_vit_b1", "gate_up"),
    ("tiny_vit_b8", "gate_up"),
    ("tiny_vit_b1", "down"),
    ("tiny_vit_b8", "down"),
    ("tiny_vit_b1", "classifier"),
    ("tiny_vit_b8", "classifier"),
    ("tiny_vit_b1", "attn_qk_qkv_group"),
    ("tiny_vit_b8", "attn_qk_qkv_group"),
    ("tiny_vit_b1", "attn_pv_qkv_group"),
    ("tiny_vit_b8", "attn_pv_qkv_group"),
)


def _task_geometry(plan, action):
    a = plan.views[action.inputs[0]].tensor
    # A matmul has two inputs; both are present for MM/MM_ACC.
    w = plan.views[action.inputs[1]].tensor
    y = plan.views[action.outputs[0]].tensor
    return {
        "m": a.rows, "k": a.cols, "n": w.rows,
        "physical_m": a.physical_rows, "physical_k": a.physical_cols,
        "physical_n": w.physical_rows,
        "a_layout": a.layout, "w_layout": w.layout,
        "y_layout": y.layout, "template": action.template_id,
        "source_k_loop": action.attrs.get("source_k_loop"),
        "source_mn_tiling": action.attrs.get("source_mn_tiling"),
    }


def _logical_macs(rows):
    total = 0
    physical = 0
    for row in rows:
        total += row["m"] * row["k"] * row["n"] * row["count"]
        physical += (row["physical_m"] * row["physical_k"] *
                     row["physical_n"] * row["count"])
    return total, physical


def collect(plans_dir: Path, manifest_key: str, plan_key: str) -> dict:
    program = load_hybrid_program(plans_dir / f"{manifest_key}.rkhybrid.json",
                                  plans_dir=plans_dir)
    plan = program.plans[plan_key].plan
    action_by_id = {a.id: a for a in plan.actions}
    submits = []
    for submit in plan.submits:
        tasks = [action_by_id[i] for i in submit.task_ids]
        mm_tasks = [a for a in tasks if a.template_id in {
            "co.mm.f16.v1", "co.mm_acc.f16.v1"}]
        geometry = Counter()
        examples = {}
        for action in mm_tasks:
            row = _task_geometry(plan, action)
            key = (row["m"], row["k"], row["n"], row["template"],
                   row["physical_m"], row["physical_k"], row["physical_n"])
            geometry[key] += 1
            examples[key] = row
        rows = []
        for key, count in sorted(geometry.items(), key=lambda x: (x[0][0],
                                                                  x[0][2],
                                                                  x[0][1])):
            item = dict(examples[key])
            item["count"] = count
            rows.append(item)
        logical, physical = _logical_macs(rows)
        # For the K-chain recipes, sum the K slices (the same M/N tile is
        # written repeatedly); for independent groups, each task stands alone.
        if rows and submit.recipe_id == "linear_k_acc_v1":
            logical_k = sum(row["k"] * row["count"] for row in rows)
            runs = 1
        elif rows and submit.recipe_id == "independent_mm_v1":
            logical_k = rows[0]["k"] if len(rows) == 1 else None
            runs = len(mm_tasks)
        else:
            logical_k = None
            runs = 1
        submits.append({
            "submit": submit.id, "mode": submit.mode,
            "recipe": submit.recipe_id, "tasks": len(tasks),
            "mm_tasks": len(mm_tasks), "logical_k_if_k_chain": logical_k,
            "independent_runs": runs, "logical_macs": logical,
            "physical_macs": physical,
            "padding_ratio": (physical / logical) if logical else None,
            "geometry": rows,
        })
    return {
        "manifest": manifest_key,
        "model": program.model,
        "batch": program.batch,
        "plan": plan_key,
        "plan_hash": plan.plan_hash,
        "task_count": plan.accounting.task_count,
        "submit_count": plan.accounting.submit_count,
        "data_bo_count": plan.accounting.data_bo_count,
        "peak_payload_bytes": plan.accounting.peak_payload_bytes,
        "reserved_bytes": plan.accounting.reserved_bytes,
        "transfers": plan.accounting.transfers,
        "submits": submits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] /
                    "models" / "compiled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    rows = [collect(args.plans_dir, manifest, plan)
            for manifest, plan in CASES]
    result = {
        "schema_version": 1,
        "kind": "gemm-geometry-inventory",
        "register_recipe": {
            "matmul": "gen_matmul_fp16: one CNA/CORE/DPU regcmd task per "
                      "compiled task; no user-visible register budget",
            "k_limit": 256,
            "mn_limit": 256,
            "physical_rows": "max(m,12) rounded up to 4",
            "physical_cols": "round_up(k/n,32)",
            "note": "K>256 is an ordered MM/MM_ACC chain; M/N>256 is output "
                    "tiling. The plan stores no register schedule or "
                    "alternative tiling recipe.",
        },
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Representative GEMM geometry",
        "",
        "| manifest | plan | sums | tasks/submits | logical MACs | "
        "physical MACs | pad ratio | peak B |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        logical = sum(s["logical_macs"] for s in row["submits"])
        physical = sum(s["physical_macs"] for s in row["submits"])
        lines.append(
            f"| {row['manifest']} | {row['plan']} | {len(row['submits'])} | "
            f"{row['task_count']}/{row['submit_count']} | {logical:,} | "
            f"{physical:,} | {physical/logical:.3f} | "
            f"{row['peak_payload_bytes']:,} |")
    lines += [
        "",
        "## Submit-level detail",
        "",
        "| manifest | plan | submit | mode/recipe | tasks | logical K | "
        "task geometry (M×K×N × count) | source tiling |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        for submit in row["submits"]:
            if not submit["geometry"]:
                continue
            geom = "; ".join(
                f"{g['m']}×{g['k']}×{g['n']} × {g['count']}"
                for g in submit["geometry"])
            tiling = "; ".join(
                sorted({json.dumps(g["source_mn_tiling"], sort_keys=True)
                        for g in submit["geometry"]}))
            lines.append(
                f"| {row['manifest']} | {row['plan']} | {submit['submit']} | "
                f"{submit['mode']}/{submit['recipe']} | {submit['tasks']} | "
                f"{submit['logical_k_if_k_chain'] or '-'} | {geom} | "
                f"{tiling} |")
    lines += [
        "",
        "`logical MACs` and `physical MACs` use the compiled task views; the",
        "physical count includes the CUBE/WEIGHT padding. Register content is",
        "not represented in RKPlan: the template is the fixed `gen_matmul_fp16`",
        "recipe with one task per K/M/N tile. No alternative register budget is",
        "currently selectable from the compiler.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "ok", "out": str(args.out),
                      "cases": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
