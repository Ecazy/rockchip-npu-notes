#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Aggregate the fixed-frequency template/granularity campaign."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROFILES = ("mlp_b1", "mlp_b8", "mlp_b32")


def med(values):
    rows = [float(x) for x in values if x is not None]
    return float(statistics.median(rows)) if rows else None


def load_text_pair(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip()]
    bitwise = next(line.split(" ", 1)[1] == "True" for line in lines
                   if line.startswith("BITWISE "))
    data = json.loads(next(line for line in lines if line.startswith("{")))
    return bitwise, data


def aggregate_bench(paths: list[Path]) -> dict:
    by_case: dict[str, list[dict]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            by_case.setdefault(case["case"], []).append(case)
    out = {}
    envelope_fields = ("validate_us", "pack_us", "emit_us", "command_us",
                       "submit_alloc_us", "submit_wait_us",
                       "output_sync_us", "download_us", "guard_us",
                       "publish_us", "alloc_us", "total_us", "clock_us",
                       "pack_prep_us", "pack_clear_us", "pack_layout_us",
                       "pack_fini_us")
    for name, rows in by_case.items():
        env = {}
        for field in envelope_fields:
            env[field] = med(row["attributed"]["per_call"].get(field, 0.0)
                             for row in rows)
        cpu_rows = [row["cpu_fp32_reference"] for row in rows
                    if row.get("cpu_fp32_reference")]
        cpu = None
        if cpu_rows:
            cpu = {
                "median_ms": med(row.get("median_ms") for row in cpu_rows),
                "min_ms": med(row.get("min_ms") for row in cpu_rows),
                "max_abs_error": med(row.get("max_abs_error")
                                     for row in cpu_rows),
                "mean_abs_error": med(row.get("mean_abs_error")
                                      for row in cpu_rows),
            }
        out[name] = {
            "runs": len(rows),
            "manifest": rows[0]["manifest"],
            "plan": rows[0]["plan"],
            "kind": rows[0]["kind"],
            "task_count": med(row["attributed"]["task_count"] for row in rows),
            "submit_count": med(row["attributed"]["submit_count"]
                                for row in rows),
            "upload_calls": med(row["attributed"]["upload_calls"]
                                for row in rows),
            "download_calls": med(row["attributed"]["download_calls"]
                                  for row in rows),
            "upload_bytes": med(row["attributed"]["upload_bytes"]
                                for row in rows),
            "download_bytes": med(row["attributed"]["download_bytes"]
                                  for row in rows),
            "application_us": med(row["application_reuse_us"]["median"]
                                   for row in rows),
            "application_min_us": min(
                row["application_reuse_us"]["median"] for row in rows),
            "application_max_us": max(
                row["application_reuse_us"]["median"] for row in rows),
            "fresh_output_us": med(
                row["application_fresh_output_us"]["median"] for row in rows),
            "setup_us": med(row["setup_us"] for row in rows),
            "workspace_setup_us": med(row["workspace_setup_us"]
                                      for row in rows),
            "bind_prepared_us": med(row["bind_prepared_us"] for row in rows),
            "constant_materialize_us": med(
                row["constant_materialize_us"] for row in rows),
            "constant_bytes": med(row["constant_bytes"] for row in rows),
            "cold_once_us": med(row["cold_once_us"] for row in rows),
            "amortized_1000_us": med(row["amortized_1000_us"] for row in rows),
            "envelope": env,
            "envelope_ranges": {
                field: {
                    "min": min(float(row["attributed"]["per_call"].get(field, 0.0))
                               for row in rows),
                    "max": max(float(row["attributed"]["per_call"].get(field, 0.0))
                               for row in rows),
                } for field in envelope_fields
            },
            "cpu_fp32": cpu,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    root = args.result_dir
    geom = json.loads((root / "geometry.json").read_text())
    base_geom = json.loads((root / "geometry-baseline.json").read_text())
    current = aggregate_bench(sorted((root / "timing").glob(
        "benchmark-fastdownload-*.json")))
    before_download = aggregate_bench(sorted((root / "timing").glob(
        "benchmark-current-*.json")))
    baseline = aggregate_bench(sorted((root / "timing").glob(
        "benchmark-baseline-r*.json")))
    geometry = {(x["manifest"], x["plan"]): x for x in geom["cases"]}
    base_geometry = {(x["manifest"], x["plan"]): x for x in
                     base_geom["cases"]}
    pairs = {}
    for b in (1, 8, 32):
        bit_plan, plan_pair = load_text_pair(
            root / "timing" / f"plan-pair-b{b}.txt")
        bit_model, model_pair = load_text_pair(
            root / "timing" / f"model-pair-b{b}.txt")
        pairs[f"b{b}"] = {
            "bitwise": bool(bit_plan and bit_model),
            "plan": plan_pair, "model": model_pair,
        }
    delivery = {}
    for b in (1, 8, 32):
        bit, payload = load_text_pair(
            root / "timing" / f"mlp-delivery-b{b}.txt")
        delivery[f"b{b}"] = {"bitwise": bool(bit), **payload}
    fast_download = {}
    for b in (8, 1):
        bit, payload = load_text_pair(
            root / "timing" / f"vit-fastdl-b{b}.txt")
        fast_download[f"b{b}"] = {"bitwise": bool(bit), **payload}
    external = {}
    for arm in ("vendor", "cpu"):
        raw = json.loads((root / "external" / f"{arm}.json").read_text())
        external[arm] = {}
        for row in raw["profiles"]:
            key = f"{row['model']}.b{row['batch']}"
            calls = int(row["batch_schedule"]["calls"])
            external[arm][key] = {
                "warm_ms": 1000.0 * float(row["model_call_seconds"]) / calls,
                "setup_ms": 1000.0 * float(row["setup_seconds"]),
                "accuracy": float(row["metrics"]["accuracy"]),
                "max_abs_logit_error": float(
                    row["metrics"]["max_abs_logit_error"]),
            }
    fast_download_cases = {}
    for name in current:
        if name not in before_download:
            continue
        old = before_download[name]
        new = current[name]
        fast_download_cases[name] = {
            "app_before_us": old["application_us"],
            "app_after_us": new["application_us"],
            "download_before_us": old["envelope"]["download_us"],
            "download_after_us": new["envelope"]["download_us"],
            "download_saved_us": (old["envelope"]["download_us"] -
                                  new["envelope"]["download_us"]),
        }
    # Add geometry and derived shares to every current case.
    cases = {}
    for name, row in current.items():
        g = geometry.get((row["manifest"], row["plan"]))
        if g is None:
            raise SystemExit(f"missing geometry for {name}")
        logical = sum(s["logical_macs"] for s in g["submits"])
        physical = sum(s["physical_macs"] for s in g["submits"])
        env = row["envelope"]
        app = row["application_us"]
        submit = env["submit_wait_us"]
        non_submit = env["total_us"] - submit
        binding = app - env["total_us"]
        row.update({
            "logical_macs": logical,
            "physical_macs": physical,
            "padding_ratio": physical / logical if logical else None,
            "device_submit_us": submit,
            "device_submit_share": submit / app if app else None,
            "non_submit_c_us": non_submit,
            "non_submit_c_share": non_submit / app if app else None,
            "binding_delta_us": binding,
            "binding_delta_share": binding / app if app else None,
            "pack_share": env["pack_us"] / app if app else None,
            "guard_share": env["guard_us"] / app if app else None,
            "download_share": env["download_us"] / app if app else None,
            "physical_gmac_s": (physical / submit / 1000.0
                                if submit else None),
        })
        cases[name] = row
    mlp = {}
    for b in (1, 8, 32):
        key = f"b{b}"
        old = baseline.get(f"mlp_mnist_mlp_{key}")
        new = cases.get(f"mlp_mnist_mlp_{key}")
        go = base_geometry.get((f"mnist_mlp_{key}", "mlp"))
        gn = geometry.get((f"mnist_mlp_{key}", "mlp"))
        mlp[key] = {
            "old_app_us": old["application_us"] if old else None,
            "new_app_us": new["application_us"] if new else None,
            "old_setup_us": old["setup_us"] if old else None,
            "new_setup_us": new["setup_us"] if new else None,
            "old_upload_calls": old["upload_calls"] if old else None,
            "new_upload_calls": new["upload_calls"] if new else None,
            "old_slots": go["data_bo_count"] if go else None,
            "new_slots": gn["data_bo_count"] if gn else None,
            "old_transfers": go["transfers"] if go else None,
            "new_transfers": gn["transfers"] if gn else None,
            "old_reserved_bytes": go["reserved_bytes"] if go else None,
            "new_reserved_bytes": gn["reserved_bytes"] if gn else None,
            "new_peak_bytes": gn["peak_payload_bytes"] if gn else None,
            "new_physical_macs": next((row["physical_macs"] for row in
                                       cases.values()
                                       if row["manifest"] ==
                                       f"mnist_mlp_{key}"), None),
            "new_logical_macs": next((row["logical_macs"] for row in
                                      cases.values()
                                      if row["manifest"] ==
                                      f"mnist_mlp_{key}"), None),
            **pairs[key],
        }
    result = {
        "schema_version": 1,
        "kind": "hardware-template-and-granularity-analysis",
        "register_recipe": geom["register_recipe"],
        "cases": cases,
        "baseline_cases": baseline,
        "mlp_plan_productization": mlp,
        "mlp_delivery": delivery,
        "vit_fast_download": fast_download,
        "fast_download_cases": fast_download_cases,
        "external": external,
        "pair_protocol": "same process, alternating old/new plan calls; full "
                         "10k model runner uses per-batch alternating calls",
        "disclosure": "submit_wait_us is the C blocking submit+wait interval. "
                      "It includes device execution and submission, and is not "
                      "a pure matrix-kernel timer. Vendor internal submit and "
                      "operator time are unavailable.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # Markdown table.
    order = ["mlp_mnist_mlp_b1", "mlp_mnist_mlp_b8",
             "mlp_mnist_mlp_b32", "qkv_tiny_vit_b1",
             "qkv_tiny_vit_b8", "gate_up_tiny_vit_b1",
             "gate_up_tiny_vit_b8", "down_tiny_vit_b1",
             "down_tiny_vit_b8", "classifier_tiny_vit_b1",
             "classifier_tiny_vit_b8", "attn_qk_qkv_group_tiny_vit_b1",
             "attn_qk_qkv_group_tiny_vit_b8",
             "attn_pv_qkv_group_tiny_vit_b1",
             "attn_pv_qkv_group_tiny_vit_b8"]
    labels = {name: name.replace("_tiny_vit_", "/").replace(
        "attn_", "").replace("_mnist_mlp_", "/") for name in order}
    lines = [
        "# Hardware-template and execution-granularity findings",
        "",
        "Fixed-frequency campaign: CPU governor `performance`, CPU6 2.304 GHz;",
        "NPU governor `performance`, 1.0 GHz; DMC governor `performance`,",
        "2.112 GHz; one NPU core. Application timing uses the prepared",
        "(`run_prepared`/`run_group_prepared`) path with reused workspace and",
        "output; constant materialization and workspace setup are separate.",
        "",
        "## Selected-case envelope",
        "",
        "`submit+wait` is the C blocking submit interval, not a pure kernel",
        "timer. `C non-submit` is `total_us - submit_wait_us`; `binding delta`",
        "is application wall minus attributed C total (signed; a small negative",
        "value is clock/probe noise). Physical MACs use the compiled",
        "CUBE/WEIGHT padding.",
        "",
        "| case | tasks/submits | app µs | submit+wait µs | share | "
        "C non-submit µs | validate µs | pack µs | command µs | "
        "output-sync µs | guard µs | download µs | binding delta µs | "
        "physical MMAC | physical GMAC/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|",
    ]
    for name in order:
        r = cases[name]
        env = r["envelope"]
        lines.append(
            f"| {labels[name]} | {r['task_count']:.0f}/{r['submit_count']:.0f} | "
            f"{r['application_us']:.1f} | {r['device_submit_us']:.1f} | "
            f"{100*r['device_submit_share']:.1f}% | "
            f"{r['non_submit_c_us']:.1f} | {env['validate_us']:.1f} | "
            f"{env['pack_us']:.1f} | {env['command_us']:.1f} | "
            f"{env['output_sync_us']:.1f} | {env['guard_us']:.1f} | "
            f"{env['download_us']:.1f} | {r['binding_delta_us']:+.1f} | "
            f"{r['physical_macs']/1e6:.3f} | {r['physical_gmac_s']:.2f} |")
    lines += [
        "",
        "Run-to-run submit/wait variation is retained in `analysis.json` under",
        "`envelope_ranges`; b1-only cases are most sensitive because their",
        "device interval is close to the scheduler noise floor. The full-model",
        "paired measurements, not single isolated points, are used for the",
        "product decision.",
        "",
        "## MLP device-resident control",
        "",
        "| profile | tasks/submits | logical MMAC | physical MMAC | pad | "
        "submit+wait µs | submit share | guard µs | pack µs | app µs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in (1, 8, 32):
        r = cases[f"mlp_mnist_mlp_b{b}"]
        lines.append(
            f"| b{b} | {r['task_count']:.0f}/{r['submit_count']:.0f} | "
            f"{r['logical_macs']/1e6:.3f} | {r['physical_macs']/1e6:.3f} | "
            f"{r['padding_ratio']:.3f}× | {r['device_submit_us']:.1f} | "
            f"{100*r['device_submit_share']:.1f}% | "
            f"{r['envelope']['guard_us']:.1f} | "
            f"{r['envelope']['pack_us']:.1f} | {r['application_us']:.1f} |")
    lines += [
        "",
        "MLP b1 and b8 have the same physical CUBE task size (12 padded rows)",
        "and the same 11 tasks / 8 submits, so b1's 1-row GEMM pays the b8",
        "physical task cost. b32 removes most padding and raises modeled",
        "physical throughput, but its larger upload/output conversion also",
        "raises the boundary share. This is an execution-granularity finding,",
        "not evidence that a different GEMM register recipe wins.",
        "",
        "## Implemented same-policy improvement: one packed K-major input upload",
        "",
        "The compiler now packs an external rank-2 CUBE value once per 256-row",
        "tile and exposes its 256-column slices as stable views. The C verifier",
        "proves the upload's canonical column offsets cover each reader; no",
        "device template, arithmetic, numeric policy, or submit/task count",
        "changes.",
        "",
        "| profile | old plan µs | new plan µs | upload-only plan gain | "
        "upload-only full gain | old delivered µs | new delivered µs | "
        "delivered gain | bitwise | slots | transfers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for b in (1, 8, 32):
        m = mlp[f"b{b}"]
        oldp = m["plan"]["old"]["median"]
        newp = m["plan"]["new"]["median"]
        d = delivery[f"b{b}"]
        lines.append(
            f"| b{b} | {oldp:.1f} | {newp:.1f} | {oldp/newp:.3f}× | "
            f"{m['model']['paired_ratio_median']:.3f}× | "
            f"{d['old_median_us']:.1f} | {d['new_median_us']:.1f} | "
            f"{d['paired_ratio_median']:.3f}× | "
            f"{'yes' if (m['bitwise'] and d['bitwise']) else 'NO'} | "
            f"{m['old_slots']}→{m['new_slots']} | "
            f"{m['old_transfers']}→{m['new_transfers']} |")
    lines += [
        "",
        "Resource effect: upload actions `4→1`, slots `22→19`, transfers",
        "`19→16`, reserved bytes `1,551,488→1,452,992` for b1/b8",
        "(`1,660,288→1,561,792` for b32); peak payload and upload bytes are",
        "unchanged. Full-test logits are bitwise equal. The steady-state gain",
        "is from fewer upload prep/fini calls and publication-gate slots, not",
        "from a matrix instruction change.",
        "",
        "## Setup, constant prepacking and output refresh",
        "",
        "`app` reuses prepared bindings and output arrays. `fresh` allocates",
        "row-major outputs per call. `cold once` includes Host constant",
        "materialization, plan/workspace setup, one-time binding preparation",
        "and one application call; it is not the steady-state latency.",
        "",
        "| case | const KB | const mat µs | plan setup ms | bind prep µs | "
        "app µs | fresh µs | cold once ms | amortized/1000 µs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        r = cases[name]
        lines.append(
            f"| {labels[name]} | {r['constant_bytes']/1024:.1f} | "
            f"{r['constant_materialize_us']:.1f} | "
            f"{r['setup_us']/1e3:.3f} | {r['bind_prepared_us']:.1f} | "
            f"{r['application_us']:.1f} | {r['fresh_output_us']:.1f} | "
            f"{r['cold_once_us']/1e3:.3f} | "
            f"{r['amortized_1000_us']:.1f} |")
    lines += [
        "",
        "The QK/PV expanded group has a large one-time binding/DMA preparation",
        "charge; it is amortized only when the same workspace is reused. The",
        "steady-state group call is still dominated by per-call validation and",
        "download staging, not by the 32 tiny MM tasks.",
        "",
        "## Runtime CUBE→row-major download fast path",
        "",
        "The C download path now copies each canonical eight-FP16 atom with one",
        "16-byte memcpy instead of evaluating `co_index` per element. Logical",
        "rows/cols are unchanged; padding is never copied. The full ViT model",
        "pair uses the same final package and differs only in this runtime",
        "library change.",
        "",
        "| profile | before µs | after µs | paired gain | bitwise |",
        "|---|---:|---:|---:|---|",
    ]
    for b in (8, 1):
        d = fast_download[f"b{b}"]
        lines.append(
            f"| tiny_vit b{b} | {d['lib1_median_us']:.1f} | "
            f"{d['lib2_median_us']:.1f} | "
            f"{d['paired_ratio_median']:.3f}× | "
            f"{'yes' if d['bitwise'] else 'NO'} |")
    lines += [
        "",
        "| case | download before µs | download after µs | saved µs |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        d = fast_download_cases.get(name)
        if d is None:
            continue
        lines.append(
            f"| {labels[name]} | {d['download_before_us']:.1f} | "
            f"{d['download_after_us']:.1f} | {d['download_saved_us']:.1f} |")
    lines += [
        "",
        "MLP has only one tiny final download, so its fast-download delta is",
        "noise; the upload-coalescing gain is the relevant MLP result. The ViT",
        "projection/FFN plans and the QK/PV groups carry large output",
        "conversions and benefit materially.",
        "",
        "## External references under the same fixed frequencies",
        "",
        "| profile | final delivered µs | vendor µs | CPU FP32 µs | "
        "vendor/final | final/CPU | vendor max logit error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    external_rows = [
        ("mnist_mlp.b1", delivery["b1"]["new_median_us"]),
        ("mnist_mlp.b8", delivery["b8"]["new_median_us"]),
        ("mnist_mlp.b32", delivery["b32"]["new_median_us"]),
        ("tiny_vit.b8", fast_download["b8"]["lib2_median_us"]),
        ("tiny_vit.b1", fast_download["b1"]["lib2_median_us"]),
    ]
    for profile, final_us in external_rows:
        v = external["vendor"][profile]
        c = external["cpu"][profile]
        vendor_us, cpu_us = v["warm_ms"] * 1000.0, c["warm_ms"] * 1000.0
        lines.append(
            f"| {profile} | {final_us:.1f} | {vendor_us:.1f} | "
            f"{cpu_us:.1f} | {vendor_us/final_us:.3f}× | "
            f"{final_us/cpu_us:.3f}× | {v['max_abs_logit_error']:.5f} |")
    lines += [
        "",
        "Vendor is non-quantized FP16 with graph fusion; ViT fails the packed",
        "per-logit tolerance and MLP is a different numeric policy. CPU is",
        "FP32. Neither is a same-policy ranking. In `vendor/final`, a value <1",
        "means vendor is faster; in `final/CPU`, <1 means the final path is",
        "faster. The vendor one-call runtime API",
        "does not expose its internal device Submit count or per-kernel time;",
        "its layer table is placement/ranking evidence only.",
        "",
        "## CPU FP32 same-GEMM reference (different numeric policy)",
        "",
        "| case | custom prepared app µs | NumPy FP32 ms | max abs delta |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        r = cases[name]
        if not r.get("cpu_fp32"):
            continue
        cpu = r["cpu_fp32"]
        lines.append(
            f"| {labels[name]} | {r['application_us']:.1f} | "
            f"{cpu['median_ms']:.4f} | {cpu['max_abs_error']:.5f} |")
    lines += [
        "",
        "The CPU reference is FP32 and the custom path is packed FP16; the two",
        "columns are not a same-policy ranking. For the smallest classifier",
        "tile the plan envelope makes the NPU path slower than the CPU",
        "reference; for the larger projection/FFN tiles the custom path is",
        "faster despite its host envelope.",
        "",
        "## Decision and next capability investment",
        "",
        "The selected implementable improvement is the **exact CUBE layout",
        "bridge packet**: one canonical K-major external upload with slice",
        "views, plus atom-wise CUBE→row-major download. Both changes preserve",
        "the arithmetic and numeric policy and do not add a hardware recipe or",
        "a new graph-level interface. They are now part of the default path.",
        "",
        "The residual split shows where not to spend the next cycle. For the",
        "MLP device-resident control, small-M physical padding and the 8-submit",
        "envelope dominate b1/b8; for ViT projections, pack/guard/validation/",
        "download boundaries are comparable to or larger than submit+wait; for",
        "the QK/PV groups, per-call binding validation and download staging",
        "dominate the 32 tiny MM tasks. A new GEMM register recipe alone is not",
        "supported by this evidence as the next highest-value investment.",
        "",
        "The next capability to evaluate is a narrow **workspace-level",
        "validation/guard and publication batching contract**: prove static",
        "binding/alias/lifetime facts once per workspace, validate only mutable",
        "pointers/finite values per call, and reduce the 32 KiB slack scan /",
        "per-slot prep-fini cost while preserving explicit failure isolation.",
        "That requires a public safety-contract decision and a new validation",
        "proof; it is not implemented in this stage.",
        "",
        "## Attribution limits",
        "",
        "- `submit+wait` includes both the submit API and device execution.",
          "  No current probe exposes per-task hardware issue latency.",
        "- Kernel register content is fixed by `gen_matmul_fp16`; RKPlan has no",
        "  alternative register budget, M/N/K tile choice beyond the fixed 256",
        "  chain, or bank-reorder recipe.",
        "- The earlier vendor-perf probe observed MLP b8 as 7 NPU + 3 CPU",
        "  boundary layers and ViT b8 as 50 NPU + 5 CPU boundary layers. Its",
        "  per-layer times use `RKNN_FLAG_COLLECT_PERF_MASK`, so they are",
        "  placement/ranking evidence only; vendor internal Submit count and",
        "  per-kernel time remain unknown.",
        "- No numeric-policy or hardware-template change is claimed by the",
        "  implemented upload coalescing or download fast path.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "ok", "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
