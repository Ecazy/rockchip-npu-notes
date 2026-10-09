#!/usr/bin/env python3
"""Dump 每个 regcmd task 的 CNA 几何 + FEATURE_GRAINS + CBUF bank（CSV），用于反解 grains 口径。

用法：analyze_vendor_grains.py [--all | <trace.jsonl> ...]
输出列：trace,op,seq,in_prec,iw,ih,ic_align,ic_real,ow,oh_atoms,atoms,kw,kh,kernels,stride_x,stride_y,
        row_bytes,rows,data_bank,weight_bank,grains,grains_need_bank(rows that fit in 1 bank)
"""
import glob
import json
import struct
import sys

R = {
    "conv_con1": 0x100c, "conv_con2": 0x1010, "conv_con3": 0x1014,
    "data_size0": 0x1020, "data_size1": 0x1024, "data_size2": 0x1028, "data_size3": 0x102c,
    "weight_size0": 0x1030, "weight_size1": 0x1034, "weight_size2": 0x1038,
    "cbuf_con0": 0x1040, "cbuf_con1": 0x1044,
    "dma_con1": 0x1074, "dma_con2": 0x1078,
    "pad_con0": 0x1068, "pad_con1": 0x106c,
}


def i(v, mask, shift):
    return ((v & mask) >> shift) if v is not None else -1


def decode(blob):
    regs, raw = {}, bytes.fromhex(blob)
    for k in range(len(raw) // 8):
        w = struct.unpack_from("<Q", raw, k * 8)[0]
        regs[w & 0xFFFF] = (w >> 16) & 0xFFFFFFFF
    return {name: regs.get(addr) for name, addr in R.items()}


def rows_of(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"regcmd"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "regcmd" or "hex" not in ev:
            continue
        r = decode(ev["hex"])
        c1, c2, c3 = r["conv_con1"], r["conv_con2"], r["conv_con3"]
        if c2 is None:
            continue
        prec = i(c1, 0x70, 4) if c1 is not None else -1
        iw = i(r["data_size0"], 0x7FF0000, 16)
        ih = i(r["data_size0"], 0x7FF, 0)
        ic = i(r["data_size1"], 0xFFFF, 0)
        ic_real = i(r["data_size1"], 0x3FFF0000, 16) + 1
        ow = i(r["data_size2"], 0x7FF, 0)
        atoms = i(r["data_size3"], 0x3FFFFF, 0)
        kw = i(r["weight_size2"], 0x1F000000, 24)
        kh = i(r["weight_size2"], 0x1F0000, 16)
        kernels = i(r["weight_size2"], 0x3FFF, 0)
        sx = i(c3, 0x7, 0)
        sy = i(c3, 0x38, 3)
        pb = 2 if prec == 2 else 1            # fp16 每元素 2 B
        atom_elems = 8 if prec == 2 else 16   # fp16 原子 8 元素 / int8 16
        plane_bytes = max(1, ic) * pb if prec == 2 else max(16, ((ic + 15) // 16) * 16)
        # row_bytes = 一行（全通道）字节：fp16 ic 个元素 ×2；int8 对齐到 16 元素原子
        planes = (ic + atom_elems - 1) // atom_elems
        row_bytes = iw * planes * 16 if prec == 2 else iw * plane_bytes
        rows = ih
        out.append(dict(
            op=ev.get("op_idx"), seq=ev.get("sequence"), prec=prec,
            iw=iw, ih=ih, ic_align=ic, ic_real=ic_real, ow=ow, atoms=atoms,
            kw=kw, kh=kh, kernels=kernels, sx=sx, sy=sy,
            row_bytes=row_bytes, rows=rows,
            data_bank=i(r["cbuf_con0"], 0xF, 0), weight_bank=i(r["cbuf_con0"], 0xF0, 4),
            grains=i(c2, 0x3FF0, 4), kg=i(c2, 0xFF0000, 16),
            wbytes=r["weight_size0"], line=i(r["dma_con1"], 0xFFFFFFF, 0),
            surf=i(r["dma_con2"], 0xFFFFFFF, 0),
            pad_left=i(r["pad_con0"], 0xF0, 4), pad_top=i(r["pad_con0"], 0xF, 0),
        ))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv or not args:
        args = sorted(set(glob.glob("experiments/**/*.jsonl", recursive=True)))
    hdr = ("trace,op,seq,prec,iw,ih,ic_align,ic_real,ow,atoms,kw,kh,kernels,sx,sy,"
           "row_bytes,rows,data_bank,weight_bank,grains,kg,wbytes,pad_left,pad_top")
    print(hdr)
    for p in args:
        try:
            rs = rows_of(p)
        except FileNotFoundError:
            continue
        for r in rs:
            print(f"{p},{r['op']},{r['seq']},{r['prec']},{r['iw']},{r['ih']},{r['ic_align']},"
                  f"{r['ic_real']},{r['ow']},{r['atoms']},{r['kw']},{r['kh']},{r['kernels']},"
                  f"{r['sx']},{r['sy']},{r['row_bytes']},{r['rows']},{r['data_bank']},"
                  f"{r['weight_bank']},{r['grains']},{r['kg']},{r['wbytes']},{r['pad_left']},{r['pad_top']}")


if __name__ == "__main__":
    main()
