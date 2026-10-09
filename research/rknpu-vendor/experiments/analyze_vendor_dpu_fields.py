#!/usr/bin/env python3
"""把 hook trace 里 task 的寄存器值**按位域**解成命名语义（§7.47）。

数据源：`experiments/rnpu_regmap.h` 里的
    #define <REG>_<FIELD>__MASK  0x...
    #define <REG>_<FIELD>__SHIFT n
用法：
    analyze_vendor_dpu_fields.py <trace.jsonl> [sequence] [最多 task 数]
输出：每个 task 的非零寄存器 → 逐位域解码（寄存器名/位域名/值）。
"""
import json
import re
import sys

REGMAP = "experiments/rnpu_regmap.h"


def load_regmap():
    txt = open(REGMAP).read()
    regs = {}       # offset -> name
    fields = {}     # reg_name -> [(field, mask, shift)]
    for m in re.finditer(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)", txt):
        regs.setdefault(int(m.group(2), 16), m.group(1))
    regnames = sorted(set(regs.values()), key=len, reverse=True)
    for m in re.finditer(r"#define\s+([A-Za-z0-9_]+)__MASK\s+0x([0-9a-fA-F]+)", txt):
        full, mask = m.group(1), int(m.group(2), 16)
        reg = next((r for r in regnames if full.startswith(r + "_")), None)
        if not reg:
            continue
        fld = full[len(reg) + 1:]
        sm = re.search(r"#define\s+%s__SHIFT\s+(\d+)" % re.escape(full), txt)
        if sm:
            fields.setdefault(reg, []).append((fld, mask, int(sm.group(1))))
    return regs, fields


def decode(blob):
    raw = bytes.fromhex(blob)
    return [(int.from_bytes(raw[k * 8:k * 8 + 8], "little") & 0xFFFF,
             (int.from_bytes(raw[k * 8:k * 8 + 8], "little") >> 16) & 0xFFFFFFFF)
            for k in range(len(raw) // 8)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/attn_relu_vendor.jsonl"
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    regs, fields = load_regmap()
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
        print("=== task_index=%s op_idx=%s entries=%d ==="
              % (ev.get("task_index"), ev.get("op_idx"), ev.get("regcfg_amount")))
        for off, val in decode(ev["hex"]):
            name = regs.get(off, "0x%04x" % off)
            flds = fields.get(name, [])
            parts = ["%s=%d" % (f, (val & mk) >> sh)
                     for f, mk, sh in flds if (val & mk) >> sh]
            if parts:
                print("  %-34s = %#010x  " % (name, val) + " | ".join(parts))
            else:
                print("  %-34s = %#010x" % (name, val))
        shown += 1
        if shown >= limit:
            break


if __name__ == "__main__":
    main()
