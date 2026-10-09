#!/usr/bin/env python3
"""Diff the vendor NPUOP register stream between two capture traces.

Used for the FP16-vs-INT8 differential on the same ONNX graph
(bn_lower_20260928/trace_conv_bn_conv_{fp16,i8}.jsonl): groups regcmd events by
op_idx (or by task index) and prints, per op, every register whose value differs.

Register names come from experiments/rnpu_regmap.h (`#define REG_<NAME> 0x...`),
so this script has no dependency on the observation-line repo layout.

Usage: fp16_diff.py A.jsonl B.jsonl [--all]
"""
import json
import re
import struct
import sys
from collections import defaultdict

BLOCKS = {
    0x0100: "PC", 0x0200: "CNA", 0x0800: "CORE", 0x1000: "DPU",
    0x2000: "DPU_RDMA", 0x4000: "PPU", 0x8000: "PPU_RDMA",
}
REGMAP = "experiments/rnpu_regmap.h"


def load_names():
    names = {}
    pat = re.compile(r"^#define REG_([A-Z0-9_]+)\s+0x([0-9a-fA-F]+)")
    for line in open(REGMAP, encoding="utf-8", errors="replace"):
        m = pat.match(line)
        if m:
            names[int(m.group(2), 16)] = m.group(1)
    return names


def decode(blob):
    out = []
    for i in range(len(blob) // 8):
        w = struct.unpack_from("<Q", blob, i * 8)[0]
        out.append(((w >> 48) & 0xffff, (w >> 16) & 0xffffffff, w & 0xffff))
    return out


def load(path):
    """op_idx -> list of tasks; each task = ordered list of (reg, value)."""
    per_op = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("event") != "regcmd":
            continue
        entries = decode(bytes.fromhex(rec["hex"]))
        per_op[rec.get("op_idx")].append([(reg, val) for _blk, val, reg in entries])
    return per_op


def show(v):
    return "0x%08x" % v


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    names = load_names()
    a, b = load(args[0]), load(args[1])
    print(f"A={args[0]}  ops={sorted(a)}")
    print(f"B={args[1]}  ops={sorted(b)}")
    for op in sorted(set(a) | set(b)):
        ta, tb = a.get(op, []), b.get(op, [])
        print(f"\n===== op_idx={op}  tasks A={len(ta)} B={len(tb)}")
        for ti in range(max(len(ta), len(tb))):
            da = dict(ta[ti]) if ti < len(ta) else {}
            db = dict(tb[ti]) if ti < len(tb) else {}
            regs = sorted(set(da) | set(db))
            diffs = [r for r in regs if da.get(r) != db.get(r)]
            order_a = [r for r, _ in ta[ti]] if ti < len(ta) else []
            order_b = [r for r, _ in tb[ti]] if ti < len(tb) else []
            if order_a != order_b:
                only_a = [r for r in order_a if r not in order_b]
                only_b = [r for r in order_b if r not in order_a]
                print(f"  task{ti} write-ORDER differs; only-A="
                      f"{[hex(r) for r in only_a]} only-B={[hex(r) for r in only_b]}")
            if not diffs and not show_all:
                print(f"  task{ti}: {len(regs)} regs identical")
                continue
            for r in (regs if show_all else diffs):
                nm = names.get(r, "?")
                blk = BLOCKS.get(r & 0xff00, "?")
                if r in da and r in db:
                    print(f"  task{ti} {blk:8s} 0x{r:04x} {nm:34s} "
                          f"A={show(da[r])} B={show(db[r])}")
                elif r in da:
                    print(f"  task{ti} {blk:8s} 0x{r:04x} {nm:34s} "
                          f"A={show(da[r])} B=<absent>")
                else:
                    print(f"  task{ti} {blk:8s} 0x{r:04x} {nm:34s} "
                          f"A=<absent> B={show(db[r])}")


if __name__ == "__main__":
    main()
