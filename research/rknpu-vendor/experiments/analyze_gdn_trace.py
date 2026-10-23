#!/usr/bin/env python3
"""Analyse the RKNN trace + op table of the Qwen3.5 GDN layer probe.

Two evidence sources are joined here:

* ``EPERF_RKNPU_IOCTL_TRACE`` jsonl from the ioctl hook: every submit with its
  wall/hw time and every captured task with ``sequence`` (submit) and
  ``op_idx``;
* the ``RKNN_LOG_LEVEL=5`` op table printed by the runtime on the board: one
  line per RKNN op with its index, type, NPU/CPU target and the name of the
  ONNX node it came from.

The ONNX node name is what makes the math-stage attribution possible: the GDN
builder names every node after the phase it implements (``conv1d``,
``nilpotent_acc2``, ``state_next3`` ...), so the rknn op table maps 1:1 onto the
Qwen3.5 math stages.

Outputs
-------
``gdn_op_map.csv``       one row per RKNN op: index, type, target, node, stage
``gdn_submit_summary.csv`` one row per submit: tasks, ops, hw/wall time, stage mix
``gdn_trace_summary.json`` totals + per-submit detail for the renderer
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
from pathlib import Path

# Stage order matters: it is the legend order in the figure.
STAGES = [
    "Input RMSNorm",
    "QKV/Z/B/A projections",
    "Causal conv1d + SiLU",
    "L2 norm (q,k)",
    "Gates (beta, g)",
    "QK + UT system",
    "Triangular inverse",
    "Delta-rule scan",
    "Gated RMSNorm",
    "Output projection",
    "Layout / staging",
    "Host (CPU op)",
]

# (prefix list, stage) -- first match wins, longer/more specific prefixes first.
STAGE_PATTERNS = [
    (("rms_",), "Input RMSNorm"),
    (("proj_",), "QKV/Z/B/A projections"),
    (("conv_", "split_q", "split_k", "split_v"), "Causal conv1d + SiLU"),
    (("l2_",), "L2 norm (q,k)"),
    (("beta_", "gate_", "z_gate"), "Gates (beta, g)"),
    (("pair_", "ut_system", "ut_matmul", "intra_matmul", "intra_chunk_attn", "k_transpose", "k_beta"), "QK + UT system"),
    (("nilpotent_",), "Triangular inverse"),
    (
        (
            "v_new",
            "inter_chunk_attn",
            "intra_out",
            "state_",
            "chunk_",
            "ut_solve",
            "q_decayed",
            "k_decayed",
            "k_decay",
            "cum_",
            "core_",
        ),
        "Delta-rule scan",
    ),
    (("gated_", "z_rows", "norm_w"), "Gated RMSNorm"),
    (("out_proj",), "Output projection"),
]

MODULE_BY_OP = {
    # CNA/CORE carry the convolution and matmul work, DPU the layout/copy work.
    "Conv": ("CNA", "CORE"),
    "ConvSigmoid": ("CNA", "CORE"),
    "exMatMul": ("CNA", "CORE"),
    "Mul": ("DPU",),
    "Add": ("DPU",),
    "Sub": ("DPU",),
    "Max": ("DPU",),
    "Softplus": ("DPU",),
    "LeakyRelu": ("DPU",),
    "Reshape": ("DPU",),
    "Transpose": ("DPU",),
    "Pad": ("DPU",),
    "Slice": ("DPU",),
    "Concat": ("DPU",),
    "Gather": ("DPU",),
    "Tile": ("DPU",),
    "Expand": ("DPU",),
    "ReduceMean": ("DPU",),
    "ReduceSum": ("DPU",),
    "Sqrt": ("DPU",),
    "Div": ("DPU",),
    "Exp": ("DPU",),
    "Sigmoid": ("DPU",),
    "CumSum": ("DPU",),
}


def classify(stage_name: str, target: str) -> str:
    if target != "NPU":
        return "Host (CPU op)"
    for prefixes, stage in STAGE_PATTERNS:
        if any(stage_name.startswith(p) for p in prefixes):
            return stage
    return "Layout / staging"


def parse_rknn_log(path: Path) -> dict[int, dict]:
    """`<idx> <op> <dtype> <target> <inputs> <outputs> ... <node>` op table."""
    ops: dict[int, dict] = {}
    pattern = re.compile(r"^(\d+)\s+(\S+)\s+(\w+)\s+(NPU|CPU|GPU)\s")
    for line in path.read_text(errors="replace").splitlines():
        body = line.split("] ")[-1].rstrip()
        match = pattern.match(body)
        if not match:
            continue
        index = int(match.group(1))
        op_type = match.group(2)
        target = match.group(4)
        node_match = re.search(r"(\w+):(\S+)\s*$", body)
        node = node_match.group(2) if node_match else ""
        rknn_name = re.sub(r"_(rs|tp|mm|node|rs_tp|rs_mm)(#\d+)?$", "", node)
        stage = classify(rknn_name, target)
        ops[index] = {
            "op_idx": index,
            "op_type": op_type,
            "target": target,
            "node": node,
            "stage": stage,
            "modules": MODULE_BY_OP.get(op_type, ("DPU",)) if target == "NPU" else ("HOST",),
        }
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--rknn-log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="qwen35_gdn_layer0_probe.rknn")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ops = parse_rknn_log(args.rknn_log)

    submits: dict[int, dict] = {}
    task_ops: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt", errors="replace") as trace_fh:
        for line in trace_fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "submit":
                submits[record["sequence"]] = record
            elif event == "task":
                task_ops[record["sequence"]][record["op_idx"]] += 1

    rows = []
    detail = []
    for seq in sorted(task_ops):
            submit = submits.get(seq, {})
            stage_tasks = collections.Counter()
            stage_ops = collections.Counter()
            module_tasks = collections.Counter()
            for op_idx, count in task_ops[seq].items():
                info = ops.get(op_idx)
                stage = info["stage"] if info else "Layout / staging"
                stage_tasks[stage] += count
                stage_ops[stage] += 1
                for module in (info["modules"] if info else ("DPU",)):
                    module_tasks[module] += count
            tasks = sum(task_ops[seq].values())
            hw_ns = submit.get("hw_elapse_time_ns") or 0
            wall_ns = submit.get("wall_elapsed_ns") or 0
            dominant = stage_tasks.most_common(1)[0][0] if stage_tasks else ""
            rows.append(
                {
                    "submit_seq": seq,
                    "tasks": tasks,
                    "ops": len(task_ops[seq]),
                    "hw_us": round(hw_ns / 1000.0, 3),
                    "wall_us": round(wall_ns / 1000.0, 3),
                    "wall_start_ns": submit.get("wall_start_ns") or 0,
                    "dominant_stage": dominant,
                    "stage_task_mix": ";".join(f"{k}={v}" for k, v in stage_tasks.most_common()),
                    "module_task_mix": ";".join(f"{k}={v}" for k, v in module_tasks.most_common()),
                }
            )
            detail.append(
                {
                    "submit_seq": seq,
                    "tasks": tasks,
                    "ops": len(task_ops[seq]),
                    "hw_ns": hw_ns,
                    "wall_ns": wall_ns,
                    "wall_start_ns": submit.get("wall_start_ns") or 0,
                    "stage_tasks": dict(stage_tasks),
                    "stage_ops": dict(stage_ops),
                    "module_tasks": dict(module_tasks),
                    "core_mask": submit.get("core_mask"),
                    "subcore_tasks": submit.get("subcore_tasks"),
                }
            )

    with (args.out_dir / "gdn_submit_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with (args.out_dir / "gdn_op_map.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["op_idx", "op_type", "target", "node", "stage", "modules"])
            for index in sorted(ops):
                info = ops[index]
                writer.writerow([index, info["op_type"], info["target"], info["node"], info["stage"], "+".join(info["modules"])])

    totals = {
            "model": args.model,
            "submits": len(detail),
            "tasks": sum(d["tasks"] for d in detail),
            "rknn_ops": len(ops),
            "npu_ops": sum(1 for o in ops.values() if o["target"] == "NPU"),
            "cpu_ops": sum(1 for o in ops.values() if o["target"] != "NPU"),
            "hw_us_total": round(sum(d["hw_ns"] for d in detail) / 1000.0, 3),
            "wall_us_total": round(sum(d["wall_ns"] for d in detail) / 1000.0, 3),
            "stages": STAGES,
            "stage_legend": {
                "Input RMSNorm": "decoder pre-norm folded into the graph (rms_*)",
                "QKV/Z/B/A projections": "in_proj_qkv / in_proj_z / in_proj_b / in_proj_a (exMatMul)",
                "Causal conv1d + SiLU": "depthwise k=4 causal conv as shifted 2-D MACs + SiLU",
                "L2 norm (q,k)": "q/k L2 normalisation (l2_*)",
                "Gates (beta, g)": "beta = sigmoid(b), g = -exp(A_log)*softplus(a+dt_bias)",
                "QK + UT system": "q@k^T, k_beta@k^T with intra-chunk decay, intra-chunk attention",
                "Triangular inverse": "(I+L)^-1 by nilpotent doubling (10 batched 64x64 matmuls)",
                "Delta-rule scan": "4-step chunk scan: v_new, inter/intra chunk attention, state update",
                "Gated RMSNorm": "RMSNormGated(core, z) * norm.weight",
                "Output projection": "out_proj matmul back to hidden 4096",
                "Layout / staging": "reshape/transpose/pad/slice/concat/gather and staging ops",
                "Host (CPU op)": "ops the RKNN runtime keeps on the host CPU",
            },
            "submit_detail": detail,
    }
    (args.out_dir / "gdn_trace_summary.json").write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in totals.items() if k != "submit_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
