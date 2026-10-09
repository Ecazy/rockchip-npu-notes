#!/usr/bin/env python3
"""Minimal .rknn (FlatBuffer) inspector: tensors + weight_data blobs.

Schema field numbers mirror openrknn/src/openrknn_model.c (which itself was
recovered from librknnrt).  openrknn copies weight_data blobs verbatim into the
weight BO (64 B aligned), so a blob here *is* the hardware weight layout.

Usage:
  rknn_blob_dump.py model.rknn [--blob N] [--dump-dir DIR] [--hex N]
"""
import argparse
import os
import struct
import sys

import numpy as np


def u16(b, p): return struct.unpack_from("<H", b, p)[0]
def u32(b, p): return struct.unpack_from("<I", b, p)[0]
def i32(b, p): return struct.unpack_from("<i", b, p)[0]


def fb_follow(b, p): return p + u32(b, p)


def fb_field(b, table, field):
    vt = table - i32(b, table)
    vt_size = u16(b, vt)
    off = 4 + field * 2
    if off >= vt_size:
        return 0
    fo = u16(b, vt + off)
    return table + fo if fo else 0


def fb_vec_len(b, fpos): return u32(b, fb_follow(b, fpos))


def fb_vec_at(b, fpos, idx):
    vec = fb_follow(b, fpos)
    if idx >= u32(b, vec):
        return 0
    return fb_follow(b, vec + 4 + idx * 4)


def fb_string(b, fpos, maxlen=512):
    try:
        s = fb_follow(b, fpos)
        n = u32(b, s)
        if n > maxlen:
            return None
        return bytes(b[s + 4:s + 4 + n]).decode("utf-8", errors="replace")
    except Exception:
        return None


def fb_bytes(b, fpos):
    s = fb_follow(b, fpos)
    n = u32(b, s)
    return bytes(b[s + 4:s + 4 + n])


def find_fb_base(b):
    """The .rknn file starts with a small binary header; the FlatBuffer root
    offset follows it. Locate the base by validating the vtable of the
    candidate root table."""
    for base in range(0, min(4096, len(b) - 8), 4):
        if b[base:base + 4] not in (b"RKNN",):
            pass
        rel = u32(b, base)
        root = base + rel
        if not (0 < rel < len(b) - 8) or root + 4 > len(b):
            continue
        vt = root - i32(b, root)
        if vt < base or vt + 4 > len(b):
            continue
        vt_size = u16(b, vt)
        if vt_size < 4 or vt_size > 256 or vt_size % 2:
            continue
        if vt + vt_size > len(b):
            continue
        nf = (vt_size - 4) // 2
        ok = True
        for i in range(nf):
            fo = u16(b, vt + 4 + 2 * i)
            if fo and (root + fo + 4 > len(b) or fo > 4096):
                ok = False
                break
        if ok:
            return base
    raise SystemExit("no FlatBuffer root found")


def parse(path):
    b = open(path, "rb").read()
    base = find_fb_base(b)
    root = base + u32(b, base)
    out = {"blobs": [], "tensors": [], "version": None, "base": base}
    # version is an int field on the root table; find it by scanning low slots
    for f in (0, 1, 2, 3, 4, 5, 6):
        p = fb_field(b, root, f)
        if p and p + 4 <= len(b):
            v = u32(b, p)
            if v in (3, 4, 5, 6, 7, 8):
                out["version"] = v
                break
    wt_field = 20 if (out["version"] or 0) > 5 else 4
    wt_fpos = fb_field(b, root, wt_field)
    if not wt_fpos:
        wt_field, wt_fpos = (4, fb_field(b, root, 4))
    if wt_fpos:
        n = fb_vec_len(b, wt_fpos)
        for i in range(n):
            e = fb_vec_at(b, wt_fpos, i)
            if not e:
                continue
            f0 = fb_field(b, e, 0)
            data = fb_bytes(b, f0) if f0 else b""
            out["blobs"].append({"idx": i, "bytes": data})
    f2 = fb_field(b, root, 2)
    if f2:
        sg = fb_vec_at(b, f2, 0)
        if sg:
            tf = fb_field(b, sg, 0)
            if tf:
                for i in range(fb_vec_len(b, tf)):
                    t = fb_vec_at(b, tf, i)
                    if not t:
                        continue
                    rec = {"idx": i, "weight_blob": None, "name": None,
                           "dtype": None, "size": None, "shape": None,
                           "act_off": None}
                    for f, k in ((5, "name"), (7, "dtype")):
                        v = fb_field(b, t, f)
                        if v:
                            rec[k] = fb_string(b, v)
                    v = fb_field(b, t, 12)
                    if v:
                        rec["size"] = u32(b, v)
                    v = fb_field(b, t, 18)
                    if v:
                        rec["weight_blob"] = u32(b, v)
                    v = fb_field(b, t, 13)
                    if v:
                        rec["act_off"] = u32(b, v)
                    v = fb_field(b, t, 4)
                    if v:
                        vec = fb_follow(b, v)
                        nd = u32(b, vec)
                        rec["shape"] = [i32(b, vec + 4 + 4 * j) for j in range(nd)]
                    out["tensors"].append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--blob", type=int, default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--hex", type=int, default=64)
    ap.add_argument("--npy", default=None, help="write blob as int8 .npy")
    a = ap.parse_args()

    m = parse(a.model)
    print(f"model={a.model} version={m['version']} blobs={len(m['blobs'])} "
          f"tensors={len(m['tensors'])}")
    for t in m["tensors"]:
        if t["weight_blob"]:
            print(f"  tensor[{t['idx']}] name={t['name']} dtype={t['dtype']} "
                  f"shape={t['shape']} size={t['size']} blob={t['weight_blob']}")
    for bl in m["blobs"]:
        print(f"  blob[{bl['idx']}] len={len(bl['bytes'])}")

    sel = [bl for bl in m["blobs"] if a.blob is None or bl["idx"] == a.blob]
    for bl in sel:
        data = bl["bytes"]
        if not data:
            continue
        print(f"--- blob[{bl['idx']}] len={len(data)}")
        h = min(a.hex, len(data))
        for o in range(0, h, 16):
            print(f"  {o:04x}: " + " ".join(f"{x:02x}" for x in data[o:o + 16]))
        if a.dump_dir:
            os.makedirs(a.dump_dir, exist_ok=True)
            p = os.path.join(a.dump_dir, f"blob_{bl['idx']:03d}.bin")
            open(p, "wb").write(data)
            print("  wrote", p)
        if a.npy:
            np.frombuffer(data, dtype=np.int8).tofile(a.npy)
            print("  wrote", a.npy)


if __name__ == "__main__":
    main()
