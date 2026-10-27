#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the model-vs-campaign table for execution cost-model JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROFILE_ORDER = (
    ("mnist_mlp", 32), ("mnist_mlp", 8), ("mnist_mlp", 1),
    ("tiny_vit", 8), ("tiny_vit", 1),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for model, batch in PROFILE_ORDER:
        path = args.results / f"cost-model-{model}_b{batch}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        data = json.loads(path.read_text())
        validation = data["same_campaign_validation"]
        candidate = data["candidates"]
        device = candidate["device_resident_qkv"]["candidates"]
        fusion = candidate["host_boundary_fusion"]["candidates"]
        rows.append({
            "key": f"{model}.b{batch}",
            "slots": data["structure"]["npu_binding_slots"],
            "selected": candidate["auto_selection"]["selected"],
            "reference_ms": validation["reference_ms"],
            "prepared_ms": validation["prepared_ms"],
            "measured_saving_ms": validation["measured_saving_ms"],
            "modeled_saving_ms": validation["modeled_saving_ms"],
            "error_ms": validation["prediction_error_saving_ms"],
            "fusion_ms": sum(item["modeled_saving_us"] for item in fusion) / 1e3,
            "device_qkv_ms": sum(item["modeled_saving_us"] for item in device) / 1e3,
            "device_download_bytes": sum(
                item["removed_producer_download_bytes"] for item in device),
            "device_upload_bytes": sum(
                item["removed_consumer_upload_bytes"] for item in device),
        })
    lines = [
        "# Execution cost model",
        "",
        "The model uses the generic profile ledger: C workspace, pack/download, "
        "submit, guard, Host actions and executor. Prepared residency removes the "
        "measured Python plan-node surface (5% residual), adds one 8 GB/s resident "
        "write-back per Host output byte, and leaves C transfers/submits unchanged.",
        "",
        "| profile | binding slots | auto choice | generic ms | prepared ms | "
        "measured saving ms | modeled saving ms | error ms | host-fusion model ms | "
        "device-QKV bound ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {row['slots']} | {row['selected']} | "
            f"{row['reference_ms']:.3f} | {row['prepared_ms']:.3f} | "
            f"{row['measured_saving_ms']:.3f} | {row['modeled_saving_ms']:.3f} | "
            f"{row['error_ms']:+.3f} | {row['fusion_ms']:.3f} | "
            f"{row['device_qkv_ms']:.3f} |")
    lines += [
        "",
        "Notes:",
        "",
        "- Tiny ViT binding surfaces (42 and 210 slots) are above the 32-slot "
        "threshold; the model follows the prepared candidate there. MLP has 13 "
        "slots, where the modeled prepared gain is at or below the observed noise "
        "and the measured campaign reverses its sign; auto keeps it on the generic "
        "path.",
        "- The Host-fusion control removes only a boundary, not a submit or "
        "transfer. Its modeled saving is a few microseconds and the campaign shows "
        "no stable additive gain.",
        "- The device-QKV screen is a transfer-only upper bound. It excludes device "
        "reorder cost, keeps all eight submits and still leaves Host Softmax "
        "between QK and PV. Missing binding/layout templates are listed in each "
        "per-profile JSON.",
        "- For Tiny ViT b8 the bound removes 49,152 bytes of QKV download and "
        "163,840 bytes of QKV-attributable QK/PV upload, yet models only about "
        "0.38 ms out of the multi-millisecond vendor gap.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
