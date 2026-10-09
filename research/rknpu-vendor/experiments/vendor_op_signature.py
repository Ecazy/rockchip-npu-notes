#!/usr/bin/env python3
"""从 RKNPU ioctl trace 生成 **算子寄存器指纹**（op 语义库）。

用法: python3 vendor_op_signature.py <trace.jsonl> [...]

口径（见台账 §7.118/§7.134/§7.146）：
  * regcmd 载荷在 **event == "regcmd"**（不是 task ✗）；条目 8 B（reg:u16 LE @0 + val:u32 LE @2）；
  * 分析前**必须先去重排序**（写入顺序会把首值重复 ⇒ 假"平台" ✗）；
  * 统计每个寄存器在多少比例的 task 中出现（**必写寄存器**），以及家族内**独有**的寄存器（判别特征）。
"""
import sys, json, struct, collections, os

def load(f):
    evs = []
    for ln in open(f, errors="ignore"):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("event") != "regcmd":
            continue
        b = bytes.fromhex(ev.get("hex") or "")
        regs = {}
        for i in range(0, len(b) - 7, 8):
            r = struct.unpack_from("<H", b, i)[0]
            v = struct.unpack_from("<I", b, i + 2)[0]
            regs.setdefault(r, v)
        if regs:
            evs.append(regs)
    return evs

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    fams = {}
    for p in sys.argv[1:]:
        name = os.path.basename(p).replace(".trace.jsonl", "").replace(".jsonl", "")
        evs = load(p)
        if not evs:
            continue
        c = collections.Counter()
        for e in evs:
            for r in e:
                c[r] += 1
        n = len(evs)
        mandatory = sorted(r for r, k in c.items() if k == n)
        fams[name] = (n, mandatory, c)
    all_mand = collections.Counter()
    for name, (n, mand, c) in fams.items():
        for r in mand:
            all_mand[r] += 1
    fam_regs = {name: set(c) for name, (n, mand, c) in fams.items()}
    print("%-34s %5s %7s %7s  %s" % ("op/模型", "task", "并集", "必写", "判别寄存器（该家族独有）"))
    for name in sorted(fams):
        n, mand, c = fams[name]
        mine = fam_regs[name]
        others = set().union(*[v for k, v in fam_regs.items() if k != name]) if len(fam_regs) > 1 else set()
        uniq = sorted(mine - others)
        print("%-34s %5d %7d %7d  %s" % (name, n, len(mine), len(mand),
              " ".join("0x%04x" % r for r in uniq[:8]) or "(无)"))
        globals().setdefault('_fam', {})[name] = mine
    common = set.intersection(*fam_regs.values()) if fam_regs else set()
    print("\n全部家族**共有**寄存器（%d 个）：%s" % (len(common),
          " ".join("0x%04x" % r for r in sorted(common))))
    print("\n**各家族相对共有集的额外寄存器**（= 判别特征）：")
    for name in sorted(fam_regs):
        extra = sorted(fam_regs[name] - common)
        print("  %-32s +%3d 个: %s" % (name, len(extra),
              " ".join("0x%04x" % r for r in extra[:12]) + (" …" if len(extra) > 12 else "")))
    return 0

if __name__ == "__main__":
    sys.exit(main())
