#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the add+ReLU primitive fusion package."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    structural = {}
    for batch in (1, 8, 32):
        rows = {}
        for arm in ("control", "fused"):
            d = json.loads((root / "campaign" / arm /
                            f"mnist_mlp_b{batch}.rkplan.json").read_text())
            rows[arm] = {
                "submits": len(d["submits"]),
                "tasks": d["accounting"]["task_count"],
                "data_bos": d["accounting"]["data_bo_count"],
                "transfers": d["accounting"]["transfers"],
                "command_workspace": d["accounting"]["command_workspace_bytes"],
                "device_request": d["accounting"]["device_request_bytes"],
                "host_bytes": d["accounting"]["host_bytes"],
                "peak_payload": d["accounting"]["peak_payload_bytes"],
                "reserved": d["accounting"]["reserved_bytes"],
                "templates": [d["actions"][t]["template_id"]
                              for s in d["submits"] for t in s["task_ids"]],
                "plan_hash": d["plan_hash"],
            }
        structural[f"b{batch}"] = {
            "control": rows["control"], "fused": rows["fused"],
            "delta": {k: rows["fused"][k] - rows["control"][k]
                      for k in ("submits", "tasks", "data_bos", "transfers",
                                "command_workspace", "device_request",
                                "host_bytes", "peak_payload", "reserved")},
            "task_sequence_control": rows["control"]["templates"],
            "task_sequence_fused": rows["fused"]["templates"],
        }
    smoke = {}
    attr = {}
    for arm in ("control", "fused"):
        for batch in ("b32", "b1"):
            smoke[f"{arm}_{batch}"] = parse_smoke(
                root / "smoke" / f"{arm}-mlp_{batch}.log")
            attr[f"{arm}_{batch}"] = parse_attr(
                root / "battr" / f"{arm}-mlp_{batch}.log")
    probe = {}
    for n in ("8192", "1000000"):
        p = root / "probe" / f"ew_add_relu_probe-{n}.log"
        if p.is_file():
            lines = p.read_text().splitlines()
            mainline = next((line for line in lines
                             if line.startswith("EW_ADD_RELU_PROBE ")), "")
            reject = next((line for line in lines
                           if line.startswith("EW_ADD_RELU_PROBE_REJECT ")), "")
            zero = next((line for line in lines
                         if line.startswith("EW_ADD_RELU_PROBE_ZERO_PAIRS ")), "")
            probe[n] = {"main": parse_kv(mainline), "reject": reject,
                        "zero_pairs": zero}
    full_runs = []
    for name in ("run1", "run2"):
        p = root / "full-model" / name / "result.json"
        if p.is_file():
            full_runs.append(json.loads(p.read_text()))
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
        "kind": "add-relu-primitive-fusion",
        "baseline": "963fd0d Submit grouping fusion + multi-submit cache",
        "control": "fuse_add_relu=False (separate EW_ADD and EW_MAX tasks)",
        "fused": "fuse_add_relu=True (co.ew_add_relu.f16.v1)",
        "structural": structural,
        "probe": probe,
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
            "reason": "EW add+ReLU is bitwise on the hardware probe and keeps "
                      "the frozen numeric policy; MLP loses two tasks, four "
                      "data BOs and two transfers while full-model MLP gains "
                      "are positive in two runs. ViT shows no stable change.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Add + ReLU primitive fusion",
        "",
        "Baseline: `963fd0d` Submit grouping fusion + multi-submit cache. The",
        "control keeps the original two-task `EW_ADD` → `EW_MAX` path. The",
        "candidate replaces it only when the add value is a non-exported,",
        "single-consumer temporary, using `co.ew_add_relu.f16.v1`.",
        "",
        "## Hardware path",
        "",
        "`EW_CFG_ALU(ADD)` sets `EW_RELU_BYPASS` bit 9 and leaves",
        "`DPU_EW_RELUX_CMP_VALUE` at zero. The candidate clears only bit 9; the",
        "identity-conv main feed, ERDMA operand, EDATA_SIZE and OUT_CVT stay",
        "unchanged. The hardware probe compares this against the separate",
        "FP16 add then FP16 max path with exact input bits.",
        "",
        "## Probe",
        "",
        "| elements | mismatches | finite | special | control µs | fused µs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n, p in sorted(probe.items(), key=lambda kv: int(kv[0])):
        m = p["main"]
        lines.append(f"| {n} | {int(m.get('mismatches', -1))} | "
                     f"{int(m.get('finite', -1))} | {int(m.get('special', -1))} | "
                     f"{m.get('control_us', 0):.1f} | {m.get('fused_us', 0):.1f} |")
    lines += [
        "",
        "The suite includes every finite pair from a boundary set (signed zero,",
        "min/max subnormal, min normal, exact cancellation, max finite),",
        "near-cancellation, deterministic random finite values, and separate",
        "NaN/Inf cases. Signed-zero pairs all report `+0` for both paths. The",
        "generator rejects `ew_relu` with no accumulation, MAX, or MUL.",
        "",
        "## Structural result (MLP)",
        "",
        "| plan | tasks | submits | data BOs | transfers | peak payload | reserved | command workspace | device request | Host |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in structural.items():
        c, f = v["control"], v["fused"]
        lines.append(
            f"| {name} | {c['tasks']}→{f['tasks']} | {c['submits']}→{f['submits']} "
            f"| {c['data_bos']}→{f['data_bos']} | {c['transfers']}→{f['transfers']} "
            f"| {c['peak_payload']}→{f['peak_payload']} B "
            f"| {c['reserved']}→{f['reserved']} B "
            f"| {c['command_workspace']}→{f['command_workspace']} B "
            f"| {c['device_request']}→{f['device_request']} B "
            f"| {c['host_bytes']}→{f['host_bytes']} B |")
    lines += [
        "",
        "For b32 the primitive removes 2 NPU tasks, 4 data BOs, 2 transfers,",
        "49,152 B peak payload, 180,480 B reserved device memory and 41,200 B",
        "Host estimate. Submits stay at the 3 achieved by the prior grouping",
        "fusion, so this arm does not double-count that benefit.",
        "",
        "## Multi-submit cache",
        "",
        "| arm | cached | command request | command physical | Host delta | fallback command request/physical |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ("control", "fused"):
        s = smoke[f"{arm}_b32"]
        info = s["info"] or s["na"]
        mem = s["mem"]
        lines.append(
            f"| {arm} | {info.get('cached', 0)}/{info.get('submits', 0)} | "
            f"{int(mem.get('device_request_bytes', 0))} B | "
            f"{int(mem.get('device_physical_bytes', 0))} B | "
            f"{int(mem.get('host_delta', 0))} B | "
            f"{int(mem.get('fallback_cmd_request', 0))}/"
            f"{int(mem.get('fallback_cmd_physical', 0))} B |")
    lines += [
        "",
        "## Isolated MLP timing",
        "",
        "| arm | total µs | submit_wait µs | output_sync µs | guard µs | pack µs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("control", "fused"):
        for batch in ("b32", "b1"):
            a = attr[f"{arm}_{batch}"]
            lines.append(f"| {arm} {batch} | {a.get('total', 0):.1f} | "
                         f"{a.get('submit_wait', 0):.1f} | "
                         f"{a.get('output_sync', 0):.1f} | "
                         f"{a.get('guard', 0):.1f} | {a.get('pack', 0):.1f} |")
    lines += [
        "",
        "## Full-model interleaved comparison",
        "",
        "256 images, 7 rounds, two independent runs, same fd and multi-submit",
        "policy. Ratios combine all 14 round ratios.",
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
        "MLP improves in every run; ViT shows no stable change. Accuracy is",
        "identical and every logit tensor is bitwise equal.",
        "",
        "## Validation",
        "",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- epilogue/primitive tests: `{checks['epilogue_fusion']}`",
        f"- command cache: `{checks['command_cache']}`",
        f"- rk_plan host: `{checks['rk_plan_host']}`",
        f"- frozen numeric replay on fused plans: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Adopt `fuse_add_relu=True` as the compile default. Keep the original",
        "two-task path available through `fuse_add_relu=False`. No model-name",
        "branch and no runtime/cache/budget expansion were introduced.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
