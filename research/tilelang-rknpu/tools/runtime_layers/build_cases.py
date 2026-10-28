#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build self-contained runtime-layer case bundles from frozen RKPlans.

The C probe consumes only these files: case.def, bundle.bin, frames.bin and
const_<value-id>.bin.  Python A arms consume the same files, so A/B/C/D share
one descriptor, one constant set, one frame sequence and one output order.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from rknpu_backend.codegen.descriptor import (BUFFER_ROLE_MAP, KIND_MAP,
                                              LAYOUT_MAP, TEMPLATE_MAP, _hash64)
from rknpu_backend.hybrid import load_hybrid_program, _materialize_constant
from rknpu_backend.models import load_weight_npz

CASES = (
    # name, manifest, plan key, kind, frame count
    ("mlp_b1", "mnist_mlp_b1", "mlp", "single", 64),
    ("mlp_b32", "mnist_mlp_b32", "mlp", "single", 64),
    ("qkv_b8", "tiny_vit_b8", "qkv", "single", 64),
    ("qk_b8", "tiny_vit_b8", "attn_qk_qkv_group", "group", 32),
    ("pv_b8", "tiny_vit_b8", "attn_pv_qkv_group", "group", 32),
)
MAGIC = b"RKPB1\0\0\0"
VERSION = 1


def _u32(v):
    return struct.pack("<I", int(v))


def _u64(v):
    return struct.pack("<Q", int(v))


def _bytes(v):
    v = bytes(v)
    return _u32(len(v)) + v


def _value_records(plan):
    role = {"input": 0, "constant": 1, "output": 2}
    return [{"id": b.id, "name": b.name, "role": role[b.role],
             "shape": list(b.shape), "nbytes": b.nbytes}
            for b in plan.bindings]


def _write_bundle(path: Path, plan: dict, values, config):
    out = bytearray()
    out += MAGIC + _u32(VERSION)
    out += _u32(len(values)) + _u32(len(plan["buffers"]))
    out += _u32(len(plan["views"])) + _u32(len(plan["actions"]))
    out += _u32(len(plan["submits"])) + _u32(len(plan["publish"]))
    out += _u32(config["frames"])
    out += _u64(_hash64(plan["profile_hash"]))
    out += _u64(_hash64(plan["source_hash"]))
    budget = plan.get("budget_policy") or {}
    out += _u64(budget.get("device_budget_bytes",
                           plan["accounting"]["device_request_bytes"]))
    out += _u64(budget.get("host_budget_bytes",
                           plan["accounting"]["host_bytes"]))
    out += _u64(plan["accounting"]["device_request_bytes"])
    out += _u64(plan["accounting"]["host_bytes"])
    for v in values:
        out += _u32(v["id"]) + _u32(v["role"]) + _u32(0) + _u32(2)
        shape = list(v["shape"]) + [0, 0]
        out += _u32(shape[0]) + _u32(shape[1]) + _u64(v["nbytes"])
        out += _bytes(v["name"].encode())
    for b in plan["buffers"]:
        out += _u32(b["id"]) + _u32(BUFFER_ROLE_MAP[b["role"]])
        out += _u32(b["slot_id"]) + _u32(b["alignment"])
        out += _u32(LAYOUT_MAP[b["layout_id"]])
        out += _u32({"zero": 0, "finite": 1, "unknown": 2,
                     "valid_only": 3}[b["padding_state"]])
        out += _u32(b["lifetime"][0]) + _u32(b["lifetime"][1])
        out += _u64(b["offset"]) + _u64(b["capacity"])
    for v in plan["views"]:
        t = v["tensor"]
        out += _u32(v["id"]) + _u32(v["buffer_id"])
        out += _u64(v["offset_bytes"])
        out += b"".join(_u32(x) for x in v["valid_region"])
        out += _u32(0) + _u32(LAYOUT_MAP[t["layout"]])
        out += _u32(t["rows"]) + _u32(t["cols"])
        out += _u32(t["physical_rows"]) + _u32(t["physical_cols"])
        out += _u32(t["atom"]) + _u32(0) + _u64(t["bytes"])
    for a in plan["actions"]:
        attrs = a["attrs"]
        flags = 0
        if attrs.get("transpose_source"):
            flags |= 1
        if attrs.get("cache") == "constant":
            flags |= 2
        if attrs.get("broadcast") == "scalar":
            flags |= 4
        elif attrs.get("broadcast") == "channel":
            flags |= 8
        out += _u32(a["id"]) + _u32(KIND_MAP[a["kind"]])
        out += _u32(TEMPLATE_MAP[a["template_id"]]
                    if a["template_id"] else 0)
        out += _u32(len(a["inputs"])) + _u32(len(a["outputs"]))
        out += _u32(len(a["deps"]))
        out += _u32(int(attrs.get("source_binding",
                                  attrs.get("binding_id", 0))))
        out += _u32(int(attrs.get("src_row_offset", 0)))
        out += _u32(int(attrs.get("src_col_offset", 0)))
        out += _u32(int(attrs.get("src_row_stride", 0)))
        out += _u32(flags) + _u32(0)
        out += b"".join(_u32(x) for x in a["inputs"])
        out += b"".join(_u32(x) for x in a["outputs"])
        out += b"".join(_u32(x) for x in a["deps"])
        out += _bytes(a["origin"].encode()[:63])
    for s in plan["submits"]:
        out += _u32(s["id"])
        out += _u32(0 if s["mode"] == "chain" else 1)
        out += _u32(0 if s["core_policy"] == "single_core_per_submit" else 1)
        out += _u32(len(s["task_ids"])) + _u32(len(s["deps"]))
        out += b"".join(_u32(x) for x in s["task_ids"])
        out += b"".join(_u32(x) for x in s["deps"])
    for p in plan["publish"]:
        out += _u32(p["binding_id"]) + _u32(p["view_id"])
        out += _u32(p["dst_row"]) + _u32(p["dst_col"])
        out += _u32(p["rows"]) + _u32(p["cols"])
    path.write_bytes(out)


def build_case(root: Path, plans_dir: Path, weights_dir: Path,
               name: str, manifest_key: str, plan_key: str, kind: str,
               frame_count: int):
    program = load_hybrid_program(plans_dir / f"{manifest_key}.rkhybrid.json",
                                  plans_dir=plans_dir)
    plan = program.plans[plan_key].plan
    model = "mnist_mlp" if manifest_key.startswith("mnist_mlp") else "tiny_vit"
    weights = load_weight_npz(weights_dir / f"{model}.npz")
    node = next((n for n in program.nodes
                 if getattr(n, "plan", None) == plan_key), None)
    constants = {}
    if plan.constants:
        if node is None:
            raise SystemExit(f"missing plan node for {manifest_key}/{plan_key}")
        lookup = {s.name: s for s in plan.constants}
        constants = {s.name: _materialize_constant(
            node.constants[s.name], lookup[s.name], weights, program.weights)
            for s in plan.constants}
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    values = _value_records(plan)
    _write_bundle(d / "bundle.bin", plan.to_dict(include_hash=False),
                  values, {"frames": frame_count})
    by_id = {v["id"]: v for v in values}
    const_meta = []
    for spec in plan.constants:
        arr = np.ascontiguousarray(constants[spec.name], dtype=np.float16)
        (d / f"const_{spec.binding_id}.bin").write_bytes(arr.tobytes())
        const_meta.append({"id": spec.binding_id, "name": spec.name,
                           "shape": list(spec.shape),
                           "nbytes": int(spec.nbytes)})
    inputs = [v for v in values if v["role"] == 0]
    outputs = [v for v in values if v["role"] == 2]
    rng = np.random.default_rng(20261027 + sum(map(ord, name)))
    frames = []
    for _ in range(frame_count):
        row = []
        for item in inputs:
            a = (rng.standard_normal(item["shape"]) * 0.1).astype(
                np.float16)
            row.append(a.tobytes(order="C"))
        frames.append(b"".join(row))
    (d / "frames.bin").write_bytes(b"".join(frames))
    lines = [f"frames {frame_count}", f"inputs {len(inputs)}"]
    for item in inputs:
        sh = list(item["shape"]) + [0, 0]
        lines.append(f"input {item['id']} 2 {sh[0]} {sh[1]} {item['nbytes']}")
    lines.append(f"constants {len(const_meta)}")
    for item in const_meta:
        lines.append(f"constant {item['id']} {item['nbytes']}")
    lines.append(f"outputs {len(outputs)}")
    for item in outputs:
        sh = list(item["shape"]) + [0, 0]
        lines.append(f"output {item['id']} 2 {sh[0]} {sh[1]} {item['nbytes']}")
    (d / "case.def").write_text("\n".join(lines) + "\n")
    (d / "case.json").write_text(json.dumps({
        "name": name, "manifest_key": manifest_key, "plan_key": plan_key,
        "kind": kind, "frames": frame_count,
        "inputs": inputs, "outputs": outputs, "constants": const_meta,
        "bundle": "bundle.bin", "frame_file": "frames.bin",
        "def_file": "case.def", "plans_dir": str(plans_dir.resolve()),
        "weights_dir": str(weights_dir.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(name, manifest_key, plan_key, "inputs", len(inputs),
          "constants", len(const_meta), "outputs", len(outputs),
          "frames", frame_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] /
                    "models" / "compiled")
    ap.add_argument("--weights-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "models")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, manifest, plan, kind, frames in CASES:
        build_case(args.out, args.plans_dir, args.weights_dir,
                   name, manifest, plan, kind, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
