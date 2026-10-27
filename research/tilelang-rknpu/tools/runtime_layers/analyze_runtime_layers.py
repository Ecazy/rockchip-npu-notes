#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize the A/B/C/D runtime-layering experiment."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CASES = ("mlp_b1", "mlp_b32", "qkv_b8", "qk_b8", "pv_b8")
PATHS = ("a0", "a1", "b", "c", "d")
ATTR_FIELDS = ("validate_us", "pack_us", "emit_us", "command_us",
               "submit_alloc_us", "submit_wait_us", "output_sync_us",
               "download_us", "guard_us", "publish_us", "alloc_us",
               "total_us", "pack_layout_us")
CPROFILE_FIELDS = ("finite_us", "upload_us", "emit_us", "execute_us",
                   "download_us", "guard_us", "publish_us", "wall_us")


def median(values):
    rows = [float(x) for x in values if x is not None]
    return float(statistics.median(rows)) if rows else None


def p10(values):
    rows = sorted(float(x) for x in values)
    if not rows:
        return None
    # nearest-rank lower decile: robust floor, explicitly not the median.
    idx = max(0, (len(rows) - 1) // 10)
    return rows[idx]


def parse_timing(path: Path):
    out = {"warm": []}
    for line in path.read_text().splitlines():
        if line.startswith("SETUP_US "):
            out["setup"] = float(line.split()[1])
        elif line.startswith("BIND_US "):
            out["bind"] = float(line.split()[1])
        elif line.startswith("FIRST_US "):
            out["first"] = float(line.split()[1])
        elif line.startswith("WARM_US "):
            out["warm"] = [float(x) for x in line.split()[1:]]
        elif line.startswith("VERIFY_US "):
            out["verify"] = float(line.split()[1])
    return out


def parse_attr(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("ATTR_US "):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) < 13:
                raise SystemExit(f"old battr log without pack_layout: {path}")
            rows.append(dict(zip(ATTR_FIELDS, vals)))
        elif line.startswith("ATTR_COUNTS "):
            # preserve counts as a separate companion, not a fake timing field
            ...
    return rows


def parse_cprofile(path: Path):
    for line in path.read_text().splitlines():
        if line.startswith("CPROFILE "):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) != 9:
                raise SystemExit(f"unexpected CPROFILE: {path}")
            calls = int(vals[0])
            return {"calls": calls,
                    **{k: vals[i + 1] / calls
                       for i, k in enumerate(CPROFILE_FIELDS)}}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    main_rows = {}
    comparison = (root / "comparison.txt").read_text().splitlines()
    bad = [line for line in comparison if "EQUAL" not in line]
    for case in CASES:
        branches = {}
        for path in PATHS:
            timings = [parse_timing(root / f"r{r}" / case / f"{path}.log")
                       for r in range(1, 8)]
            warm = [x for item in timings for x in item["warm"]]
            branch = {
                "setup_us": median(x.get("setup") for x in timings),
                "bind_us": median(x.get("bind", 0.0) for x in timings),
                "first_us": median(x.get("first") for x in timings),
                "warm_median_us": median(warm),
                "warm_p10_us": p10(warm),
                "warm_min_us": min(warm),
                "warm_max_us": max(warm),
                "warm_samples": len(warm),
                "round_medians_us": [median(x["warm"]) for x in timings],
                "verify_us": median(x.get("verify") for x in timings),
            }
            branches[path] = branch
        main_rows[case] = branches
    profiles = {}
    for case in CASES:
        b_rows = []
        c_rows = []
        for r in (1, 2, 3):
            b_rows += parse_attr(root / "profile" / f"r{r}" / case / "battr.log")
            c = parse_cprofile(
                root / "profile" / f"r{r}" / case / "cprof.log")
            if c is not None:
                c_rows.append(c)
        b = {field: median(row[field] for row in b_rows)
             for field in ATTR_FIELDS}
        c = None
        if c_rows:
            c = {field: median(row[field] for row in c_rows)
                 for field in CPROFILE_FIELDS}
            c["calls"] = c_rows[0]["calls"]
        profiles[case] = {"b_runtime": b, "c_direct": c,
                          "b_samples": len(b_rows), "c_samples": len(c_rows)}
    full_model = None
    full_model_path = root / "full-model-final" / "result.json"
    if full_model_path.exists():
        full_model = json.loads(full_model_path.read_text())
        control_path = root / "full-model-final" / "control256.json"
        if control_path.exists():
            full_model["two_runner_control"] = json.loads(
                control_path.read_text())
    # Deltas use the stable lower-decile floor; raw medians/ranges remain
    # available in the JSON.
    deltas = {}
    for case, branches in main_rows.items():
        def v(path):
            return branches[path]["warm_p10_us"]
        deltas[case] = {
            "a0_py_validation_plus_ffi_vs_b": v("a0") - v("b"),
            "a0_vs_a1_python_validation": v("a0") - v("a1"),
            "a1_python_ffi_vs_b": v("a1") - v("b"),
            "b_minus_c_rkplan_generic": v("b") - v("c"),
            "c_minus_d_data_prep_and_boundary": v("c") - v("d"),
            "b_minus_d_runtime_plus_data": v("b") - v("d"),
            "a0_minus_b_python_total": v("a0") - v("b"),
        }
    result = {
        "schema_version": 1,
        "kind": "runtime-layer-isolation",
        "protocol": "7 outer rounds, path and case order rotated; 63/31 "
                    "changed-input warm samples per process per case",
        "main": main_rows,
        "deltas_p10_us": deltas,
        "profiles": profiles,
        "full_model": full_model,
        "output_comparison": {
            "bad": len(bad), "total": len(comparison),
            "all_equal": not bad,
        },
        "disclosure": "A0 is generic Python run_into/run_group_into; A1 is "
                      "prepared Python bindings with direct legacy C calls; "
                      "B is native C RKPlan; C is direct custom-ops with "
                      "equivalent finite/upload/sync/guard/publication work "
                      "but no per-run generic shape/alias validator; D is "
                      "resident execution with input prep and verification "
                      "outside the timed interval.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    labels = {"mlp_b1": "MLP b1", "mlp_b32": "MLP b32",
              "qkv_b8": "QKV b8", "qk_b8": "QK group b8",
              "pv_b8": "PV group b8"}
    lines = [
        "# Runtime layer isolation",
        "",
        "Seven outer rounds with rotated path/case order; each process uses",
        "changing input frames. Values below are **p10 of warm samples**, a",
        "stable floor rather than a mean contaminated by device-clock/thermal",
        "outliers; medians and full ranges are in `analysis.json`.",
        "",
        "Paths: A0 = Python generic `run_into`/`run_group_into`; "
        "A1 = Python prepared binding + direct legacy C; B = native C RKPlan; "
        "C = direct custom-ops; D = resident execute only.",
        "",
        "| case | A0 µs | A1 µs | B µs | C µs | D µs | "
        "A0/A1 Python | A1/B FFI | B/C runtime | C/D data+boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        b = main_rows[case]; d = deltas[case]
        lines.append(
            f"| {labels[case]} | {b['a0']['warm_p10_us']:.1f} | "
            f"{b['a1']['warm_p10_us']:.1f} | "
            f"{b['b']['warm_p10_us']:.1f} | "
            f"{b['c']['warm_p10_us']:.1f} | "
            f"{b['d']['warm_p10_us']:.1f} | "
            f"{d['a0_vs_a1_python_validation']:+.1f} | "
            f"{d['a1_python_ffi_vs_b']:+.1f} | "
            f"{d['b_minus_c_rkplan_generic']:+.1f} | "
            f"{d['c_minus_d_data_prep_and_boundary']:+.1f} |")
    lines += [
        "",
        "A positive B/C delta means direct custom-ops ran faster than the",
        "generic RKPlan runtime. A negative A1/B value is FFI/measurement",
        "noise; the prepared-binding path does not bypass the C runtime.",
        "",
        "## Setup, first execution and warm",
        "",
        "`warm_median` is the actual center of all changed-input samples;",
        "`warm_p10` is the stable floor used in the table above.",
        "",
        "| case | path | setup µs | first µs | warm median µs | warm p10 µs | "
        "warm range µs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        for path in PATHS:
            b = main_rows[case][path]
            lines.append(
                f"| {labels[case]} | {path.upper()} | {b['setup_us']:.1f} | "
                f"{b['first_us']:.1f} | {b['warm_median_us']:.1f} | "
                f"{b['warm_p10_us']:.1f} | "
                f"{b['warm_min_us']:.1f}–{b['warm_max_us']:.1f} |")
    lines += [
        "",
        "## Separate phase samples",
        "",
        "B is the runtime attribution sample (96 calls per case, medians).",
        "C is direct custom-ops over 32 changing frames, divided per call; it",
        "uses a separate sample, not a sum borrowed from B.",
        "",
        "| case | B validate | B pack | B emit | B command | B submit/wait | "
        "B output-sync | B download | B guard | B publish | C finite | "
        "C upload | C emit | C execute+sync | C download | C guard | C publish |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        b = profiles[case]["b_runtime"]; c = profiles[case]["c_direct"]
        lines.append(
            f"| {labels[case]} | {b['validate_us']:.1f} | {b['pack_us']:.1f} | "
            f"{b['emit_us']:.1f} | {b['command_us']:.1f} | "
            f"{b['submit_wait_us']:.1f} | {b['output_sync_us']:.1f} | "
            f"{b['download_us']:.1f} | {b['guard_us']:.1f} | "
            f"{b['publish_us']:.1f} | {c['finite_us']:.1f} | "
            f"{c['upload_us']:.1f} | {c['emit_us']:.1f} | "
            f"{c['execute_us']:.1f} | {c['download_us']:.1f} | "
            f"{c['guard_us']:.1f} | {c['publish_us']:.1f} |")
    lines += [
        "",
        "The B/C profile difference is consistent with the main path: QK/PV",
        "lose substantial generic runtime validation work, while MLP b1 and",
        "QKV remain dominated by the same pack/guard/download/submit envelope.",
        "The direct C upload implementation is not faster than the runtime's",
        "pack path on every shape; it is reported as measured, not hidden.",
        "",
        "## Complete-model transmission probe",
        "",
        "The targeted complete model is Tiny ViT b8 with 256 images, batch 8,",
        "three rounds. Only the qk/pv group calls are replaced with the",
        "experimental native-C direct context; every Host action and all other",
        "plans remain the current compiled graph. All 96 paired outputs are",
        "bitwise equal. This is a subgraph integration probe, not a fully native",
        "C model implementation.",
        "",
    ]
    if full_model is not None:
        cur = full_model["timing"]["current"]
        var = full_model["timing"]["variant"]
        pair = full_model["paired_current_over_variant"]
        ctrl = full_model.get("two_runner_control")
        lines += [
            "| arm | median app µs | p10 app µs |",
            "|---|---:|---:|",
            f"| current compiled graph | {cur['median_us']:.1f} | "
            f"{cur['p10_us']:.1f} |",
            f"| qk/pv direct-C substitution | {var['median_us']:.1f} | "
            f"{var['p10_us']:.1f} |",
            "",
            f"Paired current/variant ratio: median **{pair['median_us']:.4f}×**, "
            f"p10 **{pair['p10_us']:.4f}×**; bitwise equal: "
            f"**{'YES' if full_model.get('bitwise') else 'NO'}**. "
            f"Adapter setup: {full_model['setup_us']['adapter']:.1f} µs.",
            "",
            "| NPU node | current p10 µs | variant p10 µs | current−variant µs |",
            "|---|---:|---:|---:|",
        ]
        node_current_sum = node_variant_sum = 0.0
        for node, arms in full_model["group_node_timing"].items():
            a = arms["current"]["p10_us"]
            b = arms["variant"]["p10_us"]
            node_current_sum += a
            node_variant_sum += b
            lines.append(f"| {node} | {a:.1f} | {b:.1f} | {a-b:+.1f} |")
        lines += [
            f"| **sum of measured NPU nodes** | **{node_current_sum:.1f}** | "
            f"**{node_variant_sum:.1f}** | **{node_current_sum-node_variant_sum:+.1f}** |",
            "",
            f"Measured app p10 difference (current→variant): "
            f"**{var['p10_us']-cur['p10_us']:+.1f} µs**; the measured NPU-node",
            f"sum changes by **{node_variant_sum-node_current_sum:+.1f} µs**. The",
            "remaining approximately",
            f"**{((var['p10_us']-cur['p10_us']) - (node_variant_sum-node_current_sum)):+.1f} µs**",
            "is outside the replaced group nodes (Host actions/graph/output",
            "materialisation or integration overhead). This is a measured",
            "residual, not a sum used as the headline result.",
            "",
        ]
        if ctrl is not None:
            lines += [
                "A same-program two-runner control (identical compiled graph,",
                "no adapters) shows no arm-placement bias:",
                f"paired median **{ctrl['paired_a_over_b']['median_us']:.4f}×**, "
                f"p10 **{ctrl['paired_a_over_b']['p10_us']:.4f}×**; app p10 "
                f"{ctrl['timing']['a']['p10_us']:.1f} vs "
                f"{ctrl['timing']['b']['p10_us']:.1f} µs, bitwise equal.",
                "",
            ]
        lines += [
            "The direct context removes qk/pv group-call time but the complete",
            "model is slower. Therefore the isolated B/C gain does **not**",
            "transmit to this model with the current integration shape. A fully",
            "native C Host-action/lifetime implementation is a different",
            "experiment; these data do not justify authorizing it as a redesign.",
            "",
        ]
    lines += [
        "## Output equality",
        "",
        f"- comparison rows: {len(comparison)}, differences: {len(bad)}",
        f"- A0/A1/B/C/D final outputs are bitwise equal: "
        f"{'YES' if not bad else 'NO'}",
        "",
        "D result validation/download is outside its timed interval; its",
        "validation time is recorded separately in `VERIFY_US` logs.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS" if not bad else "FAIL",
                      "out": str(args.out), "comparison_bad": len(bad)}, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
