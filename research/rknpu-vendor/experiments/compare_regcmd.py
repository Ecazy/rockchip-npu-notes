#!/usr/bin/env python3
"""逐位域对比两份 regcmd 条目表（"0xOFF 0xVAL" 文本），用于「我们的流 vs vendor 的流」（§7.48）。

用法：compare_regcmd.py <A.txt> <B.txt> [A名] [B名]
"""
import re
import sys

REGMAP = "experiments/rnpu_regmap.h"


def load_map():
    txt = open(REGMAP).read()
    regs, fields = {}, {}
    for m in re.finditer(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)", txt):
        regs.setdefault(int(m.group(2), 16), m.group(1))
    regnames = sorted(set(regs.values()), key=len, reverse=True)
    for m in re.finditer(r"#define\s+([A-Za-z0-9_]+)__MASK\s+0x([0-9a-fA-F]+)", txt):
        full, mask = m.group(1), int(m.group(2), 16)
        reg = next((r for r in regnames if full.startswith(r + "_")), None)
        if not reg:
            continue
        sm = re.search(r"#define\s+%s__SHIFT\s+(\d+)" % re.escape(full), txt)
        if sm:
            fields.setdefault(reg, []).append((full[len(reg) + 1:], mask, int(sm.group(1))))
    return regs, fields


def load_pairs(path):
    d = {}
    for ln in open(path):
        p = ln.split()
        if len(p) == 2 and p[0].startswith("0x"):
            d[int(p[0], 16)] = int(p[1], 16)
    return d


def fmt(regs, fields, off, val):
    if val is None:
        return "(缺)"
    name = regs.get(off, "0x%04x" % off)
    parts = ["%s=%d" % (f, (val & mk) >> sh)
             for f, mk, sh in fields.get(name, []) if (val & mk) >> sh]
    return "%s = %#010x%s" % (name, val, ("  [" + " ".join(parts) + "]") if parts else "")


def main():
    a, b = sys.argv[1], sys.argv[2]
    na = sys.argv[3] if len(sys.argv) > 3 else "A"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B"
    regs, fields = load_map()
    A, B = load_pairs(a), load_pairs(b)
    both = sorted(set(A) | set(B))
    same = diff = onlya = onlyb = 0
    for off in both:
        va, vb = A.get(off), B.get(off)
        if va is not None and vb is not None:
            if va == vb:
                same += 1
                continue
            diff += 1
            tag = "≠"
        elif va is not None:
            onlya += 1
            tag = "仅 " + na
        else:
            onlyb += 1
            tag = "仅 " + nb
        print("%s 0x%04x | %-64s | %s" % (tag, off, fmt(regs, fields, off, va),
                                          fmt(regs, fields, off, vb)))
    print("\n汇总: 完全相同 %d，值不同 %d，仅 %s 有 %d，仅 %s 有 %d（共 %d 个偏移）"
          % (same, diff, na, onlya, nb, onlyb, len(both)))


if __name__ == "__main__":
    main()
