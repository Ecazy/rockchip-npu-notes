#!/usr/bin/env python3
"""Analyse the NPU front-end cut: per-submit structure + S_STATUS samples.

Inputs
  fe_trace.jsonl   EPERF_RKNPU_IOCTL_TRACE of the front-end model (submits, tasks, wall times)
  fe_l5.log        RKNN level-5 op table (op idx -> type/target/onnx node)
  fes_*.csv        edge blockwatch samples, one file per submit (EPERF_RKNPU_BLOCKWATCH_NTH=k)

Output: gdn_fe_timeline.json — per submit: task/op composition, traced wall span,
sampled module busy timeline (t_rel_ms + per-module busy flags), sample coverage.

The S_STATUS module state is *measured*.  The math-stage colour of a submit is
taken from its op mix (interpolated at submit granularity only): the PC task
counter of this graph does not tick once per task, so a per-task stage
attribution would be an assumption, not a measurement.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
from pathlib import Path

MODULES = [("CNA", "c0_cna"), ("CORE", "c0_core"), ("DPU", "c0_dpu"), ("RDMA", "c0_rdma")]

STAGE_PATTERNS = [
    (("rms_",), "Input RMSNorm"),
    (("proj_",), "QKV/Z/B/A projections"),
    (("conv_", "split_"), "Causal conv1d + SiLU"),
    (("l2_",), "L2 norm (q,k)"),
    (("beta_", "gate_", "z_gate"), "Gates (beta, g)"),
    (("pair_", "ut_", "intra_", "k_"), "QK + UT system"),
    (("nilpotent_",), "Triangular inverse"),
    (("v_new", "inter_", "state_", "chunk_", "cum_", "core_", "q_dec", "k_dec"), "Delta-rule scan"),
    (("gated_", "z_rows", "norm_w"), "Gated RMSNorm"),
    (("out_proj",), "Output projection"),
    (("fe_out", "fe_gather", "fe_reshape"), "Layout / staging"),
]


def stage_of(node: str, target: str) -> str:
    if target != "NPU":
        return "Host (CPU op)"
    name = re.sub(r"_(rs|tp|mm|gather|node|rs_tp|rs_mm|rs_gather|tp_mm|gather_rs)(#\d+)?$", "", node)
    for prefixes, stage in STAGE_PATTERNS:
        if any(name.startswith(p) for p in prefixes):
            return stage
    return "Layout / staging"


def parse_l5(path: Path) -> dict[int, dict]:
    ops = {}
    pattern = re.compile(r"^(\d+)\s+(\S+)\s+(\w+)\s+(NPU|CPU|GPU)\s")
    for line in path.read_text(errors="replace").splitlines():
        body = line.split("] ")[-1].rstrip()
        m = pattern.match(body)
        if not m:
            continue
        node = re.search(r"(\w+):(\S+)\s*$", body)
        node_name = node.group(2) if node else ""
        ops[int(m.group(1))] = {
            "type": m.group(2),
            "target": m.group(4),
            "node": node_name,
            "stage": stage_of(node_name, m.group(4)),
        }
    return ops


def busy(value: str) -> bool:
    v = int(value, 16)
    s0 = v & 0x3
    s1 = (v >> 16) & 0x3
    return s0 in (1, 2) or s1 in (1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--rknn-log", type=Path, required=True)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ops = parse_l5(args.rknn_log)

    submits: dict[int, dict] = {}
    tasks: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "submit":
                submits[rec["sequence"]] = rec
            elif rec.get("event") == "task":
                tasks[rec["sequence"]][rec["op_idx"]] += 1

    detail = []
    for seq in sorted(tasks):
        sub = submits.get(seq, {})
        stage_tasks = collections.Counter()
        for op_idx, count in tasks[seq].items():
            info = ops.get(op_idx)
            stage_tasks[info["stage"] if info else "Layout / staging"] += count
        sample_path = args.samples_dir / f"fes_{seq}.csv"
        samples = []
        if sample_path.exists():
            rows = list(csv.DictReader(sample_path.open()))
            if rows:
                t0 = int(rows[0]["t_ns"])
                for row in rows:
                    samples.append(
                        {
                            "t_ms": (int(row["t_ns"]) - t0) / 1e6,
                            "counter": int(row["c0_ts"], 16) & 0xFFF,
                            "modules": {name: busy(row[col]) for name, col in MODULES},
                        }
                    )
        detail.append(
            {
                "submit_seq": seq,
                "tasks": sum(tasks[seq].values()),
                "ops": len(tasks[seq]),
                "wall_us": (sub.get("wall_elapsed_ns") or 0) / 1000.0,
                "hw_us": (sub.get("hw_elapse_time_ns") or 0) / 1000.0,
                "stage_tasks": dict(stage_tasks),
                "dominant_stage": stage_tasks.most_common(1)[0][0] if stage_tasks else "",
                "samples": samples,
                "sample_span_ms": (samples[-1]["t_ms"] - samples[0]["t_ms"]) if samples else 0.0,
                "counter_min": min(s["counter"] for s in samples) if samples else 0,
                "counter_max": max(s["counter"] for s in samples) if samples else 0,
                "counter_distinct": len({s["counter"] for s in samples}),
            }
        )

    summary = {
        "submits": len(detail),
        "tasks": sum(d["tasks"] for d in detail),
        "wall_us_total": sum(d["wall_us"] for d in detail),
        "hw_us_total": sum(d["hw_us"] for d in detail),
        "sampled_submits": sum(1 for d in detail if d["samples"]),
        "stages": sorted({d["dominant_stage"] for d in detail}) if detail else [],
        "submit_detail": detail,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "submit_detail"}, indent=2))
    for d in detail:
        print(f"S{d['submit_seq']}: tasks={d['tasks']:5d} ops={d['ops']:3d} wall_us={d['wall_us']:8.1f} "
              f"samples={len(d['samples']):4d} span_ms={d['sample_span_ms']:.2f} "
              f"counter={d['counter_min']}..{d['counter_max']}({d['counter_distinct']}) "
              f"dominant={d['dominant_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
