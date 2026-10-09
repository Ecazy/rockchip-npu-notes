#!/usr/bin/env python3
"""Recover the hardware CNA weight-slot order from a vendor .rknn weight blob.

openrknn copies the .rknn `weight_data` blob verbatim into the NPU weight BO
(64 B aligned), so that blob *is* the hardware layout.  The probe models built
by make_conv16.py (3x3/oc4/ic3) and make_rect_kernel.py (kw/kh in {1,2}) use a
known numpy seed, so every logical weight is known.  This script fits the
per-output-channel affine int8 quantization used by rknn-toolkit2

    q_oc(w) = round(w * s_oc + zp_oc),  s_oc = 255/(wmax_oc - wmin_oc),
    zp_oc  = -128 - s_oc * wmin_oc

and then reports, for every 16-byte row of the blob, which logical (kh, kw)
tap it holds.  The recovered map is the ground truth for tap nesting.

Blob layout recovered here (ic atom 16 B, oc span = oc count):
    blob = [tap][oc][ic0..ic2, pad]        pad value = round(zp_oc)
    tap  = kh_index * kw + kw_index        (kx varies fastest)

Usage: vendor_blob_decode.py
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rknn_blob_dump import parse  # noqa: E402


def qfun(W, oc):
    """Per-output-channel asymmetric int8 quantization of W[oc]."""
    w = W[oc].astype(np.int64)
    wmin, wmax = int(w.min()), int(w.max())
    s = 255.0 / (wmax - wmin)
    zp = -128.0 - s * wmin
    return s, zp


def qval(W, oc, a, b):
    s, zp = qfun(W, oc)
    v = np.rint(W[oc, :, a, b].astype(np.float64) * s + zp)
    return np.clip(v, -128, 127).astype(np.int64)


def decode(path, W, label):
    m = parse(path)
    blobs = {b["idx"]: b["bytes"] for b in m["blobs"]}
    wt = next(t for t in m["tensors"] if t["name"] == "w")
    blob = np.frombuffer(blobs[wt["weight_blob"]], dtype=np.int8)
    oc, ic, kh, kw = W.shape
    rowsz = 16
    rows = len(blob) // rowsz
    print(f"===== {label}  {path}")
    print(f"  ONNX W shape (oc,ic,kh,kw) = {W.shape}; blob {len(blob)} B "
          f"= {rows} rows x {rowsz} B; rows/(oc) = {rows//oc} tap blocks")
    assert rows == oc * kh * kw, f"row count {rows} != {oc*kh*kw}"
    nblk = rows // oc
    best, ties = [], []
    for t in range(nblk):
        score = {}
        for a in range(kh):
            for b in range(kw):
                s = 0
                for o in range(oc):
                    row = blob[(t * oc + o) * rowsz:(t * oc + o) * rowsz + rowsz]
                    s += int(np.abs(row[:ic].astype(np.int64) - qval(W, o, a, b)).sum())
                score[(a, b)] = s
        m = min(score.values())
        top = sorted(k for k, v in score.items() if v == m)
        best.append(top[0])
        ties.append(top)
        print(f"  slot t={t:2d} (byte {t*oc*rowsz:5d}) -> logical {top}"
              f"{'  (exact)' if m == 0 else f'  L1={m}'}")
    print("  --- linear rule check ---")
    for A, name in ((kw, "tap = kh*w + kw  (kx fastest, ky-major)"),
                    (kh, "tap = kw*h + kh  (ky fastest, kx-major)")):
        bad = [t for t in range(nblk) if (t // A, t % A) not in ties[t]]
        tag = "CONSISTENT" if not bad else f"CONTRADICTED at t={bad}"
        extra = "  [square: conventions coincide]" if kh == kw else ""
        print(f"  {name:44s} {tag}{extra}")
    pads = sorted({int(blob[r * rowsz + ic]) for r in range(rows)})
    print(f"  ic>=ic_real padding slot values (unique) = {pads}")
    zps = sorted({round(qfun(W, o)[1]) for o in range(oc)})
    print(f"  round(zp_oc) over oc = {zps}  -> padding == round(zp_oc)? "
          f"{set(pads) == set(zps)}")
    print()


def main():
    np.random.seed(7)
    W16 = np.random.randint(-8, 8, (4, 3, 3, 3))
    decode("results/spatial_ohfix_20260928/conv16_i8.rknn", W16,
           "vendor conv 3x3, oc=4, ic=3 (seed 7)")
    np.random.seed(11)
    decode("vendor_models/rect21_i8.rknn",
           np.random.randint(-8, 8, (16, 3, 1, 2)),
           "vendor conv kw=2 kh=1 (seed 11)")
    np.random.seed(11)
    decode("vendor_models/rect12_i8.rknn",
           np.random.randint(-8, 8, (16, 3, 2, 1)),
           "vendor conv kw=1 kh=2 (seed 11)")


if __name__ == "__main__":
    main()
