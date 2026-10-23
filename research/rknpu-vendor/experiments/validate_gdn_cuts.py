#!/usr/bin/env python3
"""Validate the GDN cut models (front-end / tail) segment by segment.

The full GDN graph is first validated against the numpy oracle
(``validate_gdn_layer_onnx.py``).  A cut model is accepted only if every tensor
on the cut boundary matches the same tensor inside the full graph, so the CPU
side of the split layer receives bit-comparable inputs.

  # front-end: single concatenated [q | k | v | beta | g | z]
  python3 validate_gdn_cuts.py --cut fe --full /tmp/gdn_full/gdn_layer.onnx \
      --cut-onnx /tmp/gdn_fe/gdn_layer.onnx --inputs qwen35_gdn_inputs.npz \
      --out-json fe_boundaries.json

  # tail: gated RMSNorm + out_proj, inputs (core, z)
  python3 validate_gdn_cuts.py --cut tail ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEGMENTS = ("gqa_q", "gqa_k", "reshape_v", "beta_sigmoid", "gate_decay", "proj_z")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else float("nan")


def full_reference(full_onnx: Path, x: np.ndarray, want: tuple[str, ...]) -> dict[str, np.ndarray]:
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(full_onnx))
    existing = {o.name for o in model.graph.output}
    for name in want:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    tmp = full_onnx.parent / "with_boundaries.onnx"
    onnx.save(model, str(tmp))
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    feeds = {"x": x}
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", choices=["fe", "tail"], required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--cut-onnx", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--board-output", type=Path, help="optional device dump of the cut model output (fp32)")
    ap.add_argument("--core", type=Path, help="tail: npy of the core tensor produced by the CPU recurrence")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    import onnxruntime as ort

    x = np.load(args.inputs)["x"][0:1].astype(np.float32)
    report: dict[str, object] = {"cut": args.cut, "cut_onnx": str(args.cut_onnx)}

    if args.cut == "fe":
        ref = full_reference(args.full, x, SEGMENTS)
        sess = ort.InferenceSession(str(args.cut_onnx), providers=["CPUExecutionProvider"])
        fe = sess.run(None, {"x": x})[0].ravel()
        sizes = [ref[name].size for name in SEGMENTS]
        if fe.size != sum(sizes):
            raise SystemExit(f"front-end size {fe.size} != {sum(sizes)}")
        offset = 0
        segments = {}
        for name, size in zip(SEGMENTS, sizes):
            segments[name] = fe[offset : offset + size].reshape(ref[name].shape)
            offset += size
        report["boundaries"] = {
            name: {"corr": corr(ref[name], segments[name]), "shape": list(ref[name].shape)}
            for name in SEGMENTS
        }
        if args.board_output:
            board = np.fromfile(args.board_output, dtype=np.float32).ravel()
            offset = 0
            device = {}
            for name, size in zip(SEGMENTS, sizes):
                device[name] = board[offset : offset + size].reshape(ref[name].shape)
                offset += size
            report["device_boundaries"] = {
                name: {
                    "corr": corr(ref[name], device[name]),
                    "max_abs": float(np.abs(ref[name] - device[name]).max()),
                    "rms_ref": float(np.sqrt((ref[name] ** 2).mean())),
                    "rms_dev": float(np.sqrt((device[name] ** 2).mean())),
                }
                for name in SEGMENTS
            }
    else:
        if not args.core:
            raise SystemExit("tail validation needs --core")
        core = np.load(args.core).astype(np.float32)
        ref = full_reference(args.full, x, ("gated_silu", "out_proj"))
        sess = ort.InferenceSession(str(args.cut_onnx), providers=["CPUExecutionProvider"])
        z = full_reference(args.full, x, ("proj_z",))["proj_z"]
        y = sess.run(None, {"core": core.reshape(1, -1), "z": z.reshape(1, -1)})[0]
        report["tail"] = {
            "corr_vs_full": corr(ref["y"] if "y" in ref else ref["out_proj"], y),
            "shape": list(y.shape),
        }
        if args.board_output:
            board = np.fromfile(args.board_output, dtype=np.float32)
            report["tail"]["corr_device_vs_host"] = corr(y, board)
            report["tail"]["corr_device_vs_full"] = corr(ref["out_proj"], board.reshape(y.shape))

    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
