#!/usr/bin/env python3
"""Decode the FP16 (PROC_PRECISION=2) vendor weight layout from a .rknn blob.

Same idea as vendor_blob_decode.py, but for fp16 models the weights are exact
fp16 (no quantization), so the blob can be compared to the ONNX weights
directly.  Source model: experiments/results/bn_lower_20260928/conv_bn_conv_fp16.rknn
built by make_onnx.py (rng = np.random.default_rng(7),
w0 = (8,3,1,1), w1 = (4,8,1,1), BN(8) folded into conv0).

Usage: fp16_blob_decode.py
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rknn_blob_dump import parse  # noqa: E402

MODEL = "results/bn_lower_20260928/conv_bn_conv_fp16.rknn"
gamma = np.array([1.0, 2.0, 0.5, 1.25, -1.0, 0.75, 1.5, -0.25])
var = np.array([1.0, 2.0, 4.0, 0.25, 1.0, 2.25, 0.5, 1.0])
eps = 1e-5


def main():
    rng = np.random.default_rng(7)
    w0 = rng.integers(-3, 4, (8, 3, 1, 1)).astype(np.float32)
    rng.integers(-2, 3, (8,))          # b0 (consumed, folded)
    w1 = rng.integers(-3, 4, (4, 8, 1, 1)).astype(np.float32)

    m = parse(MODEL)
    blobs = {b["idx"]: b["bytes"] for b in m["blobs"]}
    t = {x["name"]: x for x in m["tensors"] if x["name"] in ("w0", "w1")}
    for name, W in (("w0", w0), ("w1", w1)):
        oc, ic = W.shape[0], W.shape[1]
        raw = blobs[t[name]["weight_blob"]]
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        rows = a.reshape(len(a) // 8, 8)
        print(f"===== {name}: ONNX shape {W.shape}; blob {len(raw)} B = "
              f"{len(a)} fp16 = {len(rows)} rows x 8 fp16 (16 B atom)")
        print(f"  rows expected = oc*kh*kw = {oc} (1x1 kernel)")
        got = rows[:oc, :ic]
        print("  blob rows [:, :ic]:");  print(got)
        print("  ONNX weights:");        print(W.reshape(oc, ic))
        if name == "w0":
            fold = gamma / np.sqrt(var + eps)
            print("  BN fold factor per oc (gamma/sqrt(var+eps)):")
            print("   ", np.round(fold, 6))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(W.reshape(oc, ic) != 0,
                                 got / W.reshape(oc, ic), np.nan)
            print("  blob / ONNX row ratio (nan = ONNX 0/0):")
            print("   ", np.round(ratio, 6))
            print("  blob == ONNX * fold ? (fp16 tolerance)",
                  bool(np.allclose(got, fold[:, None] * W.reshape(oc, ic),
                                   rtol=2e-3, atol=2e-3)))
        else:
            print("  blob == ONNX ?", bool(np.array_equal(got, W.reshape(oc, ic))))
        print("  ic padding slots (ic..8) all zero:",
              bool(np.all(rows[:oc, ic:] == 0)))
        print()


if __name__ == "__main__":
    main()
