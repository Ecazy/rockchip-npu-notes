#!/usr/bin/env python3
"""用 hook trace 核对 §7.31–§7.32 推出的 CNA 规划规则是否与 vendor runtime 一致。

规则（本仓库实测推出的模型）：
  R1 `FEATURE_GRAINS = ih + kh + pad_top`（§7.32）
  R2 输入 tile 必须放得进分配的 data bank：`ceil(in_bytes/32768) <= DATA_BANK`（§7.31）
  R3 `WEIGHT_BANK = 12 - DATA_BANK`（CBUF_CON0 的两个字段互补）
  R4 vendor 的 bank 分配口径（待核对）：`WEIGHT_BANK = ceil(WEIGHT_SIZE0/32768)`（权重按需），
     `DATA_BANK = 12 - WEIGHT_BANK`（剩余全给输入）

用法：analyze_vendor_bank_rules.py [--all | <trace.jsonl> ...]
输出逐个 task 的判据 + 汇总通过率；同时标注 trace 来源（Mesa 的 rkt_regcmd.c 恒写
DATA_BANK=1/WEIGHT_BANK=11，可作为对照）。
"""
import glob
import json
import struct
import sys

R = {"conv2": 0x1010, "conv3": 0x1014, "size0": 0x1020, "size1": 0x1024,
     "wsize0": 0x1030, "wsize2": 0x1038, "cbuf0": 0x1040, "pad0": 0x1068}


def dec(blob):
    regs, raw = {}, bytes.fromhex(blob)
    for k in range(len(raw) // 8):
        w = struct.unpack_from("<Q", raw, k * 8)[0]
        regs[w & 0xFFFF] = (w >> 16) & 0xFFFFFFFF
    return {n: regs.get(a) for n, a in R.items()}


def fld(v, mask, shift):
    return ((v & mask) >> shift) if v is not None else 0


def tasks(path):
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
        r = dec(ev["hex"])
        if r["conv2"] is None or r["cbuf0"] is None:
            continue
        iw = fld(r["size0"], 0x7FF0000, 16)
        ih = fld(r["size0"], 0x7FF, 0)
        ic = fld(r["size1"], 0xFFFF, 0)
        kh = fld(r["wsize2"], 0x1F0000, 16)
        pt = fld(r["pad0"], 0xF, 0)
        grains = fld(r["conv2"], 0x3FF0, 4)
        db = fld(r["cbuf0"], 0xF, 0)
        wb = fld(r["cbuf0"], 0xF0, 4)
        wsize0 = r["wsize0"] or 0
        # 输入元素大小未知（无精度寄存器）；按 int8=1B、fp16=2B 两种算
        def inb(elem):
            atom = 16 if elem == 1 else 8
            return iw * ih * ((ic + atom - 1) // atom) * atom * elem
        out.append(dict(op=ev.get("op_idx"), iw=iw, ih=ih, ic=ic, kh=kh, pt=pt,
                        grains=grains, db=db, wb=wb, wsize0=wsize0,
                        in8=inb(1), in16=inb(2)))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv or not args:
        args = sorted(set(glob.glob("experiments/**/*.jsonl", recursive=True)))
    tot = ok_grains = ok_fit = ok_comp = ok_vendor = 0
    print(f"{'trace':<52}{'op':>4}{'iw':>5}{'ih':>4}{'ic':>5}{'kh':>3}{'pt':>3}"
          f"{'grain':>6}{'ih+kh+pt':>9}{'DB':>4}{'WB':>4}{'in8':>8}{'in16':>8}"
          f"{'wsize0':>8}{'WB=ceil':>8}")
    for p in args:
        try:
            ts = tasks(p)
        except FileNotFoundError:
            continue
        for t in ts:
            tot += 1
            g_pred = t["ih"] + t["kh"] + t["pt"]
            if t["grains"] == g_pred:
                ok_grains += 1
            fit8 = (t["in8"] + 32767) // 32768
            fit16 = (t["in16"] + 32767) // 32768
            if t["db"] >= min(fit8, fit16):
                ok_fit += 1
            if t["db"] + t["wb"] == 12:
                ok_comp += 1
            wb_pred = (t["wsize0"] + 32767) // 32768
            if t["wb"] == wb_pred and wb_pred >= 1:
                ok_vendor += 1
            print(f"{p.replace('experiments/','')[:52]:<52}{t['op']:>4}{t['iw']:>5}{t['ih']:>4}"
                  f"{t['ic']:>5}{t['kh']:>3}{t['pt']:>3}{t['grains']:>6}{g_pred:>9}"
                  f"{t['db']:>4}{t['wb']:>4}{t['in8']:>8}{t['in16']:>8}{t['wsize0']:>8}{wb_pred:>8}")
    print(f"\ntasks={tot}")
    print(f"  R1 grains == ih+kh+pad_top          : {ok_grains}/{tot}")
    print(f"  R2 DATA_BANK >= ceil(in/32768)      : {ok_fit}/{tot}")
    print(f"  R3 DATA_BANK + WEIGHT_BANK == 12    : {ok_comp}/{tot}")
    print(f"  R4 WEIGHT_BANK == ceil(WEIGHT_SIZE0/32768): {ok_vendor}/{tot}")


if __name__ == "__main__":
    main()
