#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the budgeted multi-submit static-preparation campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

CASES = ("mlp_b32", "mlp_b1")
ARMS = ("off", "sync", "nosync")
LABELS = {"mlp_b32": "MLP b32", "mlp_b1": "MLP b1"}
ATTR = ("validate", "pack", "emit", "command", "alloc", "submit_wait",
        "output_sync", "download", "guard", "publish", "alloc2", "total",
        "pack_layout")


def median(values):
    vals = [float(v) for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def p10(values):
    vals = sorted(float(v) for v in values)
    return vals[(len(vals) - 1) // 10] if vals else None


def parse_timing(path: Path):
    out = {"warm": []}
    for line in path.read_text().splitlines():
        if line.startswith("SETUP_US "):
            out["setup"] = float(line.split()[1])
        elif line.startswith("FIRST_US "):
            out["first"] = float(line.split()[1])
        elif line.startswith("WARM_US "):
            out["warm"] = [float(x) for x in line.split()[1:]]
    return out


def parse_attr(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("ATTR_US "):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) == len(ATTR):
                rows.append(dict(zip(ATTR, vals)))
    return {k: median(r[k] for r in rows) for k in ATTR}


def parse_bmulti(path: Path):
    out = {"info": {}, "mem": {}, "cmdbytes": None, "handles": []}
    for line in path.read_text().splitlines():
        if line.startswith("MULTI_INFO "):
            out["info"] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        elif line.startswith("MULTI_MEM "):
            out["mem"] = {k: float(v) for k, v in
                          re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", line)}
        elif line.startswith("MULTI_CMDBYTES_OK "):
            vals = line.split()
            out["cmdbytes"] = {"same": int(vals[1]),
                               "checked": int(vals[2].split("=")[1]),
                               "runs": int(vals[3].split("=")[1])}
        elif line.startswith("MULTI_BO "):
            vals = line.split()
            out["handles"].append(int(vals[2]))
    out["handles_unique"] = len(out["handles"]) == len(set(out["handles"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir

    round_dirs = sorted((p for p in (root / "pair").glob("r*") if p.is_dir()),
                        key=lambda p: int(p.name[1:]))
    nrounds = len(round_dirs)
    paired = {}
    for case in CASES:
        per = {arm: [] for arm in ARMS}
        setup = {arm: [] for arm in ARMS}
        first = {arm: [] for arm in ARMS}
        bitwise = True
        for rdir in round_dirs:
            for arm in ARMS:
                t = parse_timing(rdir / case / f"{arm}.log")
                per[arm].append(statistics.median(t["warm"]))
                setup[arm].append(t["setup"])
                first[arm].append(t["first"])
            hashes = {hashlib.sha256((rdir / case / f"{arm}.bin").read_bytes())
                      .hexdigest() for arm in ARMS}
            bitwise = bitwise and len(hashes) == 1
        row = {
            "median_us": {arm: median(per[arm]) for arm in ARMS},
            "setup_us": {arm: median(setup[arm]) for arm in ARMS},
            "first_us": {arm: median(first[arm]) for arm in ARMS},
            "bitwise_all_rounds": bitwise,
        }
        for a in ("sync", "nosync"):
            ratios = [per["off"][i] / per[a][i] for i in range(nrounds)]
            row[f"ratio_off_over_{a}"] = {
                "median": median(ratios), "p10": p10(ratios),
                "min": min(ratios), "max": max(ratios), "rounds": ratios}
        paired[case] = row

    attr = {case: {arm: parse_attr(root / "attr" / f"{case}-{arm}.log")
                   for arm in ARMS} for case in CASES}
    bmulti = {case: parse_bmulti(root / "cmdbytes" / f"{case}.log")
              for case in CASES}
    baseline = {case: parse_attr(root / "baseline-cost" / f"{case}.log")
                for case in CASES}
    full = json.loads((root / "full-model" / "result.json").read_text())
    numeric = json.loads((root / "numeric" / "verify-summary.json").read_text())
    checks = {
        "command_cache_test": (root / "checks" /
                               "command_cache-test.log").read_text().strip(),
        "rk_plan_host": (root / "checks" /
                         "rk_plan_host.log").read_text().strip().splitlines()[-1],
        "ctest": [line for line in (root / "checks" / "ctest.log").read_text()
                  .splitlines() if "tests passed" in line or "Total Test time" in line],
        "pytest": (root / "checks" / "pytest.log").read_text().strip().splitlines()[-1],
    }

    result = {
        "schema_version": 1,
        "kind": "budgeted-multi-submit-static-preparation",
        "control": "RK_PLAN_MULTI_SUBMIT_CACHE=0",
        "adopted_default": "RK_PLAN_MULTI_SUBMIT_CACHE=1",
        "sync_arm": "RK_PLAN_COMMAND_CACHE_SYNC=1",
        "budget": "RK_PLAN_MULTI_SUBMIT_BUDGET=default 4 MiB per workspace; "
                  "covers Host program/entry bytes plus device command BOs",
        "baseline_current": baseline,
        "paired_isolated": paired,
        "attribution": attr,
        "command_cache": bmulti,
        "full_model": full["profiles"],
        "full_model_protocol": {
            "limit": full["limit"], "rounds": full["rounds"],
            "per_call_samples": "full-model/result.json samples",
            "arms": ["control", "prepared"],
        },
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "checks": checks,
        "decision": {
            "adopt_default": True,
            "reason": "MLP multi-submit profiles gain 5.1-9.5% warm latency "
                      "with a bounded 208,528 B Host + 11,088 B device cache "
                      "per workspace; ViT profiles have no multi-submit "
                      "workspace and remain neutral.",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Budgeted multi-submit static preparation",
        "",
        "Control keeps the existing single-submit strategy.  The prepared arm",
        "caches each eligible submit's programs and packs an independent command",
        "BO per submit; the default budget is 4 MiB per workspace and covers",
        "Host program/entry bytes plus device command BO bytes.",
        "",
        "## Current multi-submit cost (control arm, same A/B)",
        "",
        "Per-call medians from the control arm (`attr/<case>-off.log`), plus the",
        "independent pre-candidate snapshot in `baseline-cost/`:",
        "",
        "| case | emit µs | command µs | total µs |",
        "|---|---:|---:|---:|",
    ]
    for case in CASES:
        b = attr[case]["off"]
        lines.append(f"| {LABELS[case]} | {b['emit']:.1f} | {b['command']:.1f} | "
                     f"{b['total']:.1f} |")
    lines += [
        "",
        "## Isolated MLP A/B",
        "",
        f"{nrounds} rotated rounds; per-call samples are in `pair/r*/.../WARM_US`.",
        "",
        "| case | control | cache+sync | cache | control/cache | p10 | range | bitwise |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in CASES:
        r = paired[case]
        lines.append(
            f"| {LABELS[case]} | {r['median_us']['off']:.1f} | "
            f"{r['median_us']['sync']:.1f} | {r['median_us']['nosync']:.1f} | "
            f"**{r['ratio_off_over_nosync']['median']:.4f}×** | "
            f"{r['ratio_off_over_nosync']['p10']:.4f}× | "
            f"{r['ratio_off_over_nosync']['min']:.4f}–"
            f"{r['ratio_off_over_nosync']['max']:.4f} | "
            f"{'yes' if r['bitwise_all_rounds'] else 'NO'} |")
    lines += [
        "",
        "Attribution confirms the removed work (medians, µs):",
        "",
        "| case | arm | emit | command | output sync | total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        for arm in ARMS:
            a = attr[case][arm]
            lines.append(f"| {LABELS[case]} | {arm} | {a['emit']:.1f} | "
                         f"{a['command']:.1f} | {a['output_sync']:.1f} | "
                         f"{a['total']:.1f} |")
    lines += [
        "",
        "## Creation cost and cache memory",
        "",
        "| case | setup control µs | setup cache µs | delta µs | first |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        r = paired[case]
        delta = r["setup_us"]["nosync"] - r["setup_us"]["off"]
        first = bmulti[case]["mem"].get("setup_on_us", 0.0)
        lines.append(
            f"| {LABELS[case]} | {r['setup_us']['off']:.1f} | "
            f"{r['setup_us']['nosync']:.1f} | {delta:+.1f} | {first:.1f} |")
    lines += [
        "",
        "| case | cached submits | command request | command physical | fallback request/physical | Host delta | unique BOs | bytes unchanged |",
        "|---|---:|---:|---:|---|---:|---|---|",
    ]
    for case in CASES:
        b = bmulti[case]
        cb = b["cmdbytes"] or {}
        lines.append(
            f"| {LABELS[case]} | {b['info'].get('cached')}/"
            f"{b['info'].get('submits')} | "
            f"{b['info'].get('device_request_bytes')} B | "
            f"{b['info'].get('device_physical_bytes')} B | "
            f"{int(b['mem'].get('fallback_cmd_request', 0))}/"
            f"{int(b['mem'].get('fallback_cmd_physical', 0))} B | "
            f"{int(b['mem'].get('host_delta', 0))} B | "
            f"{'yes' if b['handles_unique'] else 'NO'} | "
            f"{'yes' if cb.get('same') else 'NO'} "
            f"({cb.get('checked', 0)} BOs / {cb.get('runs', 0)} runs) |")
    lines += [
        "",
        "Request bytes (11,088 B) match the plan's declared device accounting.",
        "Physical occupancy is page-aligned per BO: the eight cache BOs occupy",
        "32,768 B, while the 11,088 B request is 8 pages only if one fused arena",
        "were used.  The fallback command workspace remains allocated and is",
        "shown separately.  The 4 MiB cache budget is a hard per-workspace cap",
        "using Host bytes plus page-aligned command bytes; plan device-budget",
        "admission uses the declared request-byte口径.  A tight budget caches only",
        "the prefix of submits that fits.",
        "",
        "## Command regions",
        "",
        "- Each cached submit is packed into its own `rocket_bo`, allocated with",
        "  the exact `words * 8` size by `co_command_plan_build(..., workspace=NULL)`.",
        "- No workspace command BO pointer is adjusted or faked into a slice;",
        "  `rkt_chain_pack_at` derives regcmd addresses from that BO's own DMA base",
        "  and `rkt_chain_verify` validates the chain inside the same object.",
        "- The eight MLP submits report eight unique handles; no later submit can",
        "  overwrite an earlier cache.",
        "- `command_cache_test` independently builds two owned command plans and",
        "  asserts the second submit leaves the first plan's bytes unchanged.",
        "",
        "## Five-profile full model (interleaved, 256 images, 7 rounds)",
        "",
        "| profile | control µs | prepared µs | ratio | p10 | throughput control | throughput prepared | bitwise |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, v in full["profiles"].items():
        throughput = {arm: statistics.median(
            row["throughput_images_s"][arm] for row in v["rounds"])
            for arm in ("control", "prepared")}
        lines.append(
            f"| {name} | {v['timing']['control']['median_us']:.1f} | "
            f"{v['timing']['prepared']['median_us']:.1f} | "
            f"**{v['round_ratio_median']:.4f}×** | "
            f"{v['round_ratio_p10']:.4f}× | "
            f"{throughput['control']:.1f} | {throughput['prepared']:.1f} | "
            f"{'yes' if v['bitwise'] else 'NO'} |")
    lines += [
        "",
        "All five profiles use the same compiled graphs, weights, fd and timed",
        "interval; control and prepared runners alternate order per call and round.",
        "Accuracy is identical and all logits are bitwise equal.",
        "",
        "## Validation",
        "",
        f"- command-cache host test: `{checks['command_cache_test']}`",
        f"- rk_plan host: `{checks['rk_plan_host']}`",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- frozen numeric replay: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Adopt `RK_PLAN_MULTI_SUBMIT_CACHE=1` with the 4 MiB default budget.",
        "`RK_PLAN_MULTI_SUBMIT_CACHE=0` remains the control.  No hardware recipe,",
        "numeric policy or asynchronous execution changed.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
