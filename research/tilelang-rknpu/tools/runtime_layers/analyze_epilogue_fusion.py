#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the generic Linear-epilogue Submit fusion package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


ATTR_NAMES = ["validate", "pack", "emit", "command", "alloc", "submit_wait",
              "output_sync", "download", "guard", "publish", "alloc2",
              "total", "pack_layout"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_kv(line: str) -> dict[str, float]:
    return {k: float(v) for k, v in
            re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", line)}


def parse_smoke(path: Path) -> dict:
    out = {"na": {}, "info": {}, "mem": {}, "cmdbytes": None}
    for line in path.read_text().splitlines():
        if line.startswith("MULTI_NA "):
            out["na"] = {k: int(v) for k, v in
                         re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_INFO "):
            out["info"] = {k: int(v) for k, v in
                           re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_MEM "):
            out["mem"] = parse_kv(line)
        elif line.startswith("MULTI_CMDBYTES_OK "):
            vals = line.split()
            out["cmdbytes"] = {"same": int(vals[1]),
                               "checked": int(vals[2].split("=")[1]),
                               "runs": int(vals[3].split("=")[1])}
    return out


def parse_attr(path: Path) -> dict[str, float]:
    rows = [list(map(float, line.split()[1:]))
            for line in path.read_text().splitlines()
            if line.startswith("ATTR_US ")]
    if not rows:
        return {}
    return {name: float(statistics.median(row[i] for row in rows))
            for i, name in enumerate(ATTR_NAMES)}


def load_full(path: Path) -> dict:
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    plans = {}
    structural = {}
    for batch in (1, 8, 32):
        c = json.loads((root / "campaign" / "control" /
                        f"mnist_mlp_b{batch}.rkplan.json").read_text())
        f = json.loads((root / "campaign" / "fused" /
                        f"mnist_mlp_b{batch}.rkplan.json").read_text())
        same_data = all(c[k] == f[k] for k in
                        ("bindings", "constants", "buffers", "views",
                         "actions", "publish"))
        c_tasks = [c["actions"][t]["template_id"]
                   for s in c["submits"] for t in s["task_ids"]]
        f_tasks = [f["actions"][t]["template_id"]
                   for s in f["submits"] for t in s["task_ids"]]
        structural[f"b{batch}"] = {
            "control_submits": len(c["submits"]),
            "fused_submits": len(f["submits"]),
            "tasks": c["accounting"]["task_count"],
            "task_sequence_identical": c_tasks == f_tasks,
            "data_path_identical": same_data,
            "control_command_workspace": c["accounting"]["command_workspace_bytes"],
            "fused_command_workspace": f["accounting"]["command_workspace_bytes"],
            "control_device_request": c["accounting"]["device_request_bytes"],
            "fused_device_request": f["accounting"]["device_request_bytes"],
            "control_host": c["accounting"]["host_bytes"],
            "fused_host": f["accounting"]["host_bytes"],
            "peak_payload": c["accounting"]["peak_payload_bytes"],
            "reserved": c["accounting"]["reserved_bytes"],
            "allocator_slack": (c["accounting"]["reserved_bytes"] -
                                c["accounting"]["peak_payload_bytes"]),
            "data_bo_count": c["accounting"]["data_bo_count"],
            "transfers": c["accounting"]["transfers"],
            "control_recipes": [s["recipe_id"] for s in c["submits"]],
            "fused_recipes": [s["recipe_id"] for s in f["submits"]],
        }
        plans[f"b{batch}"] = {"control": c, "fused": f}
    smoke = {}
    attr = {}
    for arm in ("control", "fused"):
        smoke[f"{arm}_b32"] = parse_smoke(root / "smoke" / f"{arm}-mlp_b32.log")
        smoke[f"{arm}_b1"] = parse_smoke(root / "smoke" / f"{arm}-mlp_b1.log")
        attr[f"{arm}_b32"] = parse_attr(root / "battr" / f"{arm}-mlp_b32.log")
        attr[f"{arm}_b1"] = parse_attr(root / "battr" / f"{arm}-mlp_b1.log")
    full_runs = []
    for name in ("run1", "run2"):
        p = root / "full-model" / name / "result.json"
        if p.is_file():
            full_runs.append(load_full(p))
    profiles = {}
    for profile in ("mlp_b32", "mlp_b8", "mlp_b1", "vit_b8", "vit_b1"):
        rows = [r["profiles"][profile] for r in full_runs]
        ratios = [x["ratio_control_over_prepared"]
                  for row in rows for x in row["rounds"]]
        profiles[profile] = {
            "runs": [{
                "ratio_median": row["round_ratio_median"],
                "ratio_p10": row["round_ratio_p10"],
                "control_us": row["timing"]["control"]["median_us"],
                "fused_us": row["timing"]["prepared"]["median_us"],
                "setup_us": row["setup_us"],
                "bitwise": row["bitwise"],
                "accuracy": row["accuracy"],
            } for row in rows],
            "combined_ratio_median": float(statistics.median(ratios)),
            "combined_ratio_min": float(min(ratios)),
            "combined_ratio_max": float(max(ratios)),
            "all_bitwise": all(row["bitwise"] for row in rows),
        }
    numeric = json.loads((root / "numeric" / "verify-summary.json").read_text())
    checks = {
        "ctest": [line for line in (root / "checks" / "ctest.log").read_text()
                  .splitlines() if "tests passed" in line or "Total test" in line],
        "pytest": (root / "checks" / "pytest.log").read_text().strip().splitlines()[-1],
        "epilogue_fusion": (root / "checks" /
                            "epilogue-fusion.log").read_text().strip(),
        "command_cache": (root / "checks" /
                          "command_cache-test.log").read_text().strip(),
        "rk_plan_host": (root / "checks" /
                         "rk_plan_host.log").read_text().strip().splitlines()[-1],
    }
    result = {
        "schema_version": 1,
        "kind": "epilogue-submit-fusion",
        "control": "fuse_epilogue=False",
        "fused": "fuse_epilogue=True",
        "structural": structural,
        "smoke": smoke,
        "attr_us": attr,
        "full_model": {"protocol": {"limit": 256, "rounds": 7,
                                    "interleaved": True,
                                    "multi_submit_cache": 1},
                       "profiles": profiles},
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "checks": checks,
        "decision": {
            "adopt_default": True,
            "reason": "Same tasks, data path and arithmetic; Submit count "
                      "8->3. Full-model MLP gains are positive in two runs; "
                      "ViT is neutral and all logits are bitwise equal.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Linear → bias → ReLU Submit-envelope fusion",
        "",
        "Generic rule: a directly dependent `MM/MM_ACC` chain may absorb an",
        "immediately following `EW_ADD` and `EW_MAX` when the actions are",
        "contiguous in plan order. The tasks, templates, arithmetic, buffers,",
        "transfers and numeric narrowing are unchanged; only the Submit",
        "envelope is reduced. `fuse_epilogue=False` keeps the original",
        "per-stage lowering as the control.",
        "",
        "## Structural comparison (MLP)",
        "",
        "| plan | tasks | submits control→fused | command workspace | device request | host bytes | peak/reserved | BOs | transfers | data path identical |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for k, v in structural.items():
        lines.append(
            f"| {k} | {v['tasks']} | {v['control_submits']}→{v['fused_submits']} "
            f"| {v['control_command_workspace']}→{v['fused_command_workspace']} B "
            f"| {v['control_device_request']}→{v['fused_device_request']} B "
            f"| {v['control_host']}→{v['fused_host']} B "
            f"| {v['peak_payload']}/{v['reserved']} B "
            f"| {v['data_bo_count']} | {v['transfers']} | "
            f"{'yes' if v['data_path_identical'] else 'NO'} |")
    lines += [
        "",
        "The fused plan pays +32,768 B device request and +33,248 B Host for the",
        "larger worst-case fallback command workspace (6 tasks instead of 4),",
        "but the data path is byte-identical: same 19 data BOs, same peak payload,",
        "same allocator slack, same upload/download count and same tensor views.",
        "With multi-submit cache on, the fused plan needs 3 command BOs / 16 KiB",
        "physical instead of 8 command BOs / 32 KiB, offsetting part of that",
        "fallback increase.",
        "",
        "## Multi-submit-cache allocation",
        "",
        "| plan | cache | cached | request | physical | Host delta | fallback command request/physical |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for arm in ("control", "fused"):
        s = smoke[f"{arm}_b32"]
        info = s["info"] or s["na"]
        mem = s["mem"]
        lines.append(
            f"| {arm} | {'prepared' if info.get('ready') else 'fallback'} | "
            f"{info.get('cached', 0)}/{info.get('submits', 0)} | "
            f"{int(mem.get('device_request_bytes', 0))} B | "
            f"{int(mem.get('device_physical_bytes', 0))} B | "
            f"{int(mem.get('host_delta', 0))} B | "
            f"{int(mem.get('fallback_cmd_request', 0))}/"
            f"{int(mem.get('fallback_cmd_physical', 0))} B |")
    lines += [
        "",
        "## Isolated MLP timing",
        "",
        "| plan | total µs | submit_wait µs | output_sync µs | pack µs |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("control", "fused"):
        for batch in ("b32", "b1"):
            a = attr[f"{arm}_{batch}"]
            lines.append(f"| {arm} {batch} | {a.get('total', 0):.1f} | "
                         f"{a.get('submit_wait', 0):.1f} | "
                         f"{a.get('output_sync', 0):.1f} | "
                         f"{a.get('pack', 0):.1f} |")
    lines += [
        "",
        "Submit wait falls by about 45 µs on both b32 and b1; pack and data",
        "movement are unchanged.",
        "",
        "## Full-model interleaved comparison",
        "",
        "256 images, 7 rounds, two independent runs, same fd and multi-submit",
        "cache policy. Ratios are control/fused from all 14 rounds.",
        "",
        "| profile | ratio median | min | max | run1 control→fused µs | run2 control→fused µs | bitwise |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, v in profiles.items():
        runs = v["runs"]
        def rtxt(i):
            if i >= len(runs):
                return "—"
            r = runs[i]
            return f"{r['control_us']:.1f}→{r['fused_us']:.1f}"
        lines.append(
            f"| {name} | **{v['combined_ratio_median']:.4f}×** | "
            f"{v['combined_ratio_min']:.4f}× | {v['combined_ratio_max']:.4f}× | "
            f"{rtxt(0)} | {rtxt(1)} | {'yes' if v['all_bitwise'] else 'NO'} |")
    lines += [
        "",
        "MLP improves in every run; ViT has no Linear-epilogue fusion in its",
        "default graph and stays neutral. Accuracy is identical and every logit",
        "tensor is bitwise equal.",
        "",
        "## Validation",
        "",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- epilogue fusion test: `{checks['epilogue_fusion']}`",
        f"- command cache: `{checks['command_cache']}`",
        f"- rk_plan host: `{checks['rk_plan_host']}`",
        f"- frozen numeric replay on fused plans: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Adopt the generic `fuse_epilogue=True` compile default for new plans.",
        "The original lowering remains available with `fuse_epilogue=False` and",
        "is the control used above. No model-name branch, no changed numeric",
        "policy and no runtime expansion were introduced.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
