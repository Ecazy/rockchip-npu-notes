#!/usr/bin/env python3
"""解剖 hook trace 里每条 task 的 regcmd 块（命名寄存器 + 逐 task 差异）（§7.46）。

用途：回答「vendor 一条 submit 里成百上千个 task 到底在做什么、彼此差什么」。
输出：每个 task 的 (偏移, 寄存器名, 值) 列表 + 与上一个 task 的差异行。

用法：analyze_vendor_task_blocks.py <trace.jsonl> [sequence] [最多打印几个 task]
"""
import json
import re
import sys

REGMAP = "experiments/rnpu_regmap.h"


def load_names():
    names = {}
    try:
        txt = open(REGMAP).read()
    except FileNotFoundError:
        return names
    for m in re.finditer(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)", txt):
        names.setdefault(int(m.group(2), 16), m.group(1))
    return names


def decode(blob):
    raw = bytes.fromhex(blob)
    out = []
    for k in range(len(raw) // 8):
        w = int.from_bytes(raw[k * 8:k * 8 + 8], "little")
        out.append((w & 0xFFFF, (w >> 16) & 0xFFFFFFFF))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/attn_relu_vendor.jsonl"
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    names = load_names()
    prev = None
    shown = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"event":"regcmd"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("sequence") != seq or "hex" not in ev:
            continue
        regs = decode(ev["hex"])
        print("--- task_index=%s op_idx=%s amount=%s entries=%d ---"
              % (ev.get("task_index"), ev.get("op_idx"), ev.get("regcfg_amount"), len(regs)))
        for off, val in regs:
            mark = ""
            if prev is not None:
                old = dict(prev).get(off)
                if old is not None and old != val:
                    mark = "   ← 变化 (was %#x)" % old
                elif old is None:
                    mark = "   ← 新增"
            print("    0x%04x %-34s = %#x%s" % (off, names.get(off, "?"), val, mark))
        prev = regs
        shown += 1
        if shown >= limit:
            break


if __name__ == "__main__":
    main()
