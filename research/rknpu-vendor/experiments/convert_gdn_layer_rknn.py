#!/usr/bin/env python3
"""Convert the GDN layer ONNX graph to RKNN for RK3588.

Quantisation follows the SDPA probe convention (``w8a8`` INT8) so the GDN
timeline can be read against the Full-Attention timeline; pass ``--fp16`` to
build a non-quantised model when the INT8 accuracy gate fails.

Calibration uses real layer-0 inputs: token-embedding lookups of eight 196-token
windows taken from this repository's own documentation (see
``make_gdn_layer_inputs.py``), not random noise.

Usage:
  python3 convert_gdn_layer_rknn.py --onnx gdn_layer.onnx --inputs gdn_inputs.npz \
      --out gdn_layer_int8.rknn [--fp16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="npz with x [N, 1, L, H] or [N, L, H]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-samples", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--optimization-level", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import onnx_mapping_shim

    onnx_mapping_shim.install()
    from rknn.api import RKNN

    x = np.load(args.inputs)["x"].astype(np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    n = min(args.calib_samples, x.shape[0])

    work = args.out.parent / (args.out.stem + "_calib")
    work.mkdir(parents=True, exist_ok=True)
    dataset = work / "dataset.txt"
    lines = []
    for i in range(n):
        path = work / f"sample_{i:02d}.npy"
        np.save(path, x[i])
        lines.append(str(path))
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform="rk3588",
        # mean/std stay unset: rknn applies them per *last-dim* channel and the
        # GDN input is a [1, 196, 4096] activation, not an image
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        optimization_level=args.optimization_level,
        # one input: [1, 196, 4096]
        quant_img_RGB2BGR=False,
    )
    ret = rknn.load_onnx(model=str(args.onnx))
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")
    ret = rknn.build(do_quantization=not args.fp16, dataset=str(dataset))
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")
    ret = rknn.export_rknn(str(args.out))
    if ret != 0:
        raise SystemExit(f"export_rknn failed: {ret}")
    rknn.release()

    summary = {
        "onnx": str(args.onnx),
        "rknn": str(args.out),
        "quantized": not args.fp16,
        "optimization_level": args.optimization_level,
        "calib_samples": n,
        "calib_input_rms": float(np.sqrt((x[:n] ** 2).mean())),
        "rknn_size_bytes": args.out.stat().st_size,
    }
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
