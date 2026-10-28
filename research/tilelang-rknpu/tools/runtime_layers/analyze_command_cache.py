#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the bounded command-BO packing-reuse campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

CASES = ("qk_b8", "pv_b8", "qkv_b8", "mlp_b32")
ARMS = ("off", "sync", "nosync")
LABELS = {"qk_b8": "QK group b8", "pv_b8": "PV group b8",
          "qkv_b8": "QKV b8", "mlp_b32": "MLP b32"}


def median(values):
    rows = [float(x) for x in values if x is not None]
    return float(statistics.median(rows)) if rows else None


def p10(values):
    rows = sorted(float(x) for x in values)
    return rows[(len(rows) - 1) // 10] if rows else None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def parse_bcmd(path: Path):
    out = {"build": [], "run": {0: [], 1: []}, "mem": None, "na": None}
    for line in path.read_text().splitlines():
        if line.startswith("CMDBUILD_US "):
            out["build"].append([float(x) for x in line.split()[1:]])
        elif line.startswith("CMDRUN_US "):
            vals = line.split()
            out["run"][int(vals[1])].append([float(x) for x in vals[2:]])
        elif line.startswith("CMDMEM "):
            out["mem"] = {k: int(v) for k, v in
                          re.findall(r"(\w+)=(-?\d+)", line)}
        elif line.startswith("CMDBUILD_NA"):
            out["na"] = line
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir

    round_dirs = sorted(
        (p for p in (root / "pair").glob("r*") if p.is_dir()),
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
            hashes = {sha(rdir / case / f"{arm}.bin") for arm in ARMS}
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
                "median": median(ratios),
                "p10": p10(ratios),
                "min": min(ratios),
                "max": max(ratios),
                "rounds": ratios,
            }
        paired[case] = row

    bcmd = {case: parse_bcmd(root / "bcmd" / f"{case}.log")
            for case in ("qk_b8", "pv_b8", "qkv_b8")}
    for case in ("mlp_b32",):
        p = root / "bcmd" / f"{case}.log"
        bcmd[case] = parse_bcmd(p) if p.exists() else {
            "build": [], "run": {0: [], 1: []}, "mem": None, "na": None}
    for case, row in bcmd.items():
        if row["build"]:
            names = ("allocation", "clear", "pack", "verify", "sync",
                     "handles", "total")
            row["median_us"] = {
                name: median(v[i] for v in row["build"])
                for i, name in enumerate(names)}
            row["first_us"] = dict(
                zip(names, row["build"][0]))
        if row["run"][0]:
            row["run_median_us"] = {
                str(arm): {
                    "command": median(v[0] for v in rows),
                    "submit_wait": median(v[1] for v in rows),
                    "output_sync": median(v[2] for v in rows),
                }
                for arm, rows in row["run"].items() if rows}

    full = {}
    for name, label in (("full-model-b8", "Tiny ViT b8"),
                        ("full-model-b1", "Tiny ViT b1")):
        d = json.loads((root / name / "result.json").read_text())
        full[name] = {
            "label": label,
            "bitwise": d["bitwise"],
            "setup_us": d["setup_us"],
            "timing": d["timing"],
            "round_ratio_median": d["round_ratio_median"],
            "round_ratio_p10": d["round_ratio_p10"],
        }

    cmdbytes = {}
    for p in sorted((root / "cmdbytes").glob("*.log")):
        text = p.read_text()
        m = re.search(r"CMDBYTES_OK (-?\d+) bytes=(\d+) runs=(\d+)", text)
        if m:
            cmdbytes[p.stem] = {"same": int(m.group(1)),
                                "bytes": int(m.group(2)),
                                "runs": int(m.group(3))}

    defaultcheck = {}
    for case in ("qk_b8", "pv_b8", "qkv_b8"):
        defaultcheck[case] = {
            "default_sha256": sha(root / "defaultcheck" / f"{case}-default.bin"),
            "cache_off_sha256": sha(root / "defaultcheck" / f"{case}-off.bin"),
        }
        defaultcheck[case]["bitwise"] = (
            defaultcheck[case]["default_sha256"] ==
            defaultcheck[case]["cache_off_sha256"])

    numeric = json.loads((root / "numeric" / "verify-summary.json").read_text())
    checks = {
        "command_cache_test": (root / "checks" / "command_cache-test.log").read_text().strip(),
        "ctest": [line for line in (root / "checks" / "ctest.log").read_text().splitlines()
                  if "tests passed" in line or "Total Test time" in line],
        "pytest": (root / "checks" / "pytest.log").read_text().strip().splitlines()[-1],
        "failure_atomicity": (root / "checks" / "failure-atomicity.log").read_text().strip().splitlines()[-1],
    }

    result = {
        "schema_version": 1,
        "kind": "command-bo-packing-reuse-ab",
        "paired_isolated": paired,
        "command_breakdown": bcmd,
        "full_model": full,
        "command_bytes": cmdbytes,
        "default_check": defaultcheck,
        "numeric": {"status": numeric.get("status"),
                    "problems": numeric.get("problems")},
        "checks": checks,
        "decision": {
            "adopt_default": True,
            "flag": "RK_PLAN_COMMAND_CACHE",
            "control": "RK_PLAN_COMMAND_CACHE=0",
            "sync_arm": "RK_PLAN_COMMAND_CACHE_SYNC=1",
            "scope": "single-submit workspaces with program_cache_ready and a "
                     "stable CHAIN or one-task submit",
            "host_memory_increment_bytes_per_eligible_workspace": bcmd["qk_b8"]["mem"]["delta"],
            "command_bo_increment_bytes": 0,
            "reason": "Packing is immutable after creation; the command BO is "
                      "device-read-only, so the repeated clear/pack/verify and "
                      "CPU cache-maintenance pair can be removed while output "
                      "sync, publishing and failure isolation stay unchanged.",
        },
        "protocol": "Same library, same workspace shape and same output sync. "
                    "Three arms: command cache off, cache on with per-submit "
                    "command sync, cache on without it.  Seven rotated rounds; "
                    "primary estimate is the median of per-round paired "
                    "medians, p10 auxiliary.  Full model uses three runners in "
                    "one process with env snapshotted at creation; stats=None.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Command-BO packing reuse A/B",
        "",
        "This experiment reuses the command BO packed at workspace creation for",
        "eligible single-submit workspaces.  The control packs every submit; the",
        "sync arm reuses the pack but keeps the historical command-BO prep/fini;",
        "the adopted arm also removes that repeated CPU cache maintenance.",
        "Output sync, guard, download, publication and failure isolation are",
        "identical in all arms.",
        "",
        "## Isolated cases",
        "",
        "µs; ratio = control / arm; primary is the median of per-round paired",
        f"medians over {nrounds} rotated rounds.",
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
        "The cache-on-with-sync arm isolates the command-sync component:",
        "",
        "| case | control/cache+sync | cache+sync/cache |",
        "|---|---:|---:|",
    ]
    for case in CASES:
        r = paired[case]
        sync_ratio = r["ratio_off_over_sync"]["median"]
        delta = r["median_us"]["sync"] / r["median_us"]["nosync"]
        lines.append(f"| {LABELS[case]} | {sync_ratio:.4f}× | {delta:.4f}× |")
    lines += [
        "",
        "",
        f"Creation cost (process setup medians from the same {nrounds} paired runs;",
        "the direct first-build interval is the cleaner attribution):",
        "",
        "| case | setup control | setup cache | setup delta | first command build |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        r = paired[case]
        delta = r["setup_us"]["nosync"] - r["setup_us"]["off"]
        first = (f"{bcmd[case]['first_us']['total']:.2f}"
                 if bcmd[case].get("first_us") else "n/a (ineligible)")
        lines.append(
            f"| {LABELS[case]} | {r['setup_us']['off']:.1f} | "
            f"{r['setup_us']['nosync']:.1f} | {delta:+.1f} | {first} |")
    lines += [
        "",
        "## Command preparation breakdown",
        "",
        "Medians of 64 builds into a separate command workspace on the same fd,",
        "using the creation-time `co_program` array.  `sync` is the command-BO",
        "prep+fini pair; clear, pack, verify and handles are the rest of the",
        "removed build.  `total` is one continuous outer interval; sub-parts are",
        "probe-only and may not sum to it.  MLP b32 is ineligible (8 submits).",
        "",
        "| case | clear | pack | verify | handles | sync | total | first build |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in ("qk_b8", "pv_b8", "qkv_b8"):
        r = bcmd[case]
        m = r["median_us"]
        lines.append(
            f"| {LABELS[case]} | {m['clear']:.2f} | {m['pack']:.2f} | "
            f"{m['verify']:.2f} | {m['handles']:.2f} | {m['sync']:.2f} | "
            f"{m['total']:.2f} | {r['first_us']['total']:.2f} |")
    lines += [
        "",
        "Host accounting measured with `rk_workspace_host_bytes`: the cache adds",
        f"**{bcmd['qk_b8']['mem']['delta']} B** per eligible workspace "
        "(`sizeof(co_command_plan)`), and **0 B** of command BO because the",
        "existing `co_execute_workspace` command BO is reused.",
        "",
        "## Full model (uninstrumented)",
        "",
        "| model | control | cache+sync | cache | control/cache | p10 | bitwise |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("full-model-b8", "full-model-b1"):
        f = full[name]
        t = f["timing"]
        lines.append(
            f"| {f['label']} | {t['control']['median_us']:.1f} | "
            f"{t['cache_sync']['median_us']:.1f} | "
            f"{t['cache_nosync']['median_us']:.1f} | "
            f"**{f['round_ratio_median']['ratio_control_over_cache_nosync']:.4f}×** | "
            f"{f['round_ratio_p10']['ratio_control_over_cache_nosync']:.4f}× | "
            f"{'yes' if f['bitwise'] else 'NO'} |")
    lines += [
        "",
        "The sync-only contribution is "
        f"{full['full-model-b8']['round_ratio_median']['ratio_cache_sync_over_cache_nosync']:.4f}× "
        "for b8 and "
        f"{full['full-model-b1']['round_ratio_median']['ratio_cache_sync_over_cache_nosync']:.4f}× "
        "for b1; the rest is the removed CPU clear/pack/verify work.",
        "",
        "",
        "Default-on check: the isolated outputs with no environment override and",
        "with `RK_PLAN_COMMAND_CACHE=0` are bitwise equal for "
        + ", ".join(
            f"{case} ({'yes' if c['bitwise'] else 'NO'})"
            for case, c in defaultcheck.items()) + ".",
        "",
        "## Contract evidence",
        "",
        "- The cached plan borrows the workspace's existing command BO; no new BO",
        "  is allocated, no handle crosses a workspace/fd/domain, and the plan is",
        "  freed with the workspace.",
        "- The runtime only reaches the cached path when `program_cache_ready` is",
        "  set for the same single submit; multi-submit, `CO_SEPARATE` with more",
        "  than one task, failed builds and all other workspaces keep the original",
        "  per-submit packing path.",
        "- `command_cache_test` compares cached and freshly built plans with the",
        "  same implementation: command bytes, task order, chain tail, flags and",
        "  handle lists match; a failed submit leaves the plan and bytes unchanged",
        "  and performs no output sync; wrong fd/workspace/reallocation is refused.",
    ]
    for case, c in cmdbytes.items():
        lines.append(
            f"- Board `bcmdverify` {case}: {c['runs']} cached executions, "
            f"{c['bytes']} command bytes, unchanged={bool(c['same'])} after an "
            "explicit FROM_DEVICE sync.")
    lines += [
        "",
        "## Validation",
        "",
        f"- command-cache host test: `{checks['command_cache_test']}`",
        f"- CTest: `{checks['ctest'][0] if checks['ctest'] else ''}`",
        f"- pytest: `{checks['pytest']}`",
        f"- failure atomicity: `{checks['failure_atomicity']}`",
        f"- frozen numeric replay: **{numeric.get('status')}**, "
        f"problems: {numeric.get('problems')}",
        "",
        "## Decision",
        "",
        "Adopt `RK_PLAN_COMMAND_CACHE=1` as the default for eligible",
        "single-submit workspaces and keep `RK_PLAN_COMMAND_CACHE=0` as the",
        "control.  `RK_PLAN_COMMAND_CACHE_SYNC=1` remains the audit arm that",
        "restores the repeated command-BO sync without restoring packing.",
        "",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
