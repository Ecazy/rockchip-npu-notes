#!/usr/bin/env python3
"""RKNPU ioctl trace 一站式报告：派发形态 + LUT + 算子类别。

用法: python3 rknpu_trace_report.py <trace.jsonl> [...]

输出四块（对应 skill 的三类问题）：
  1) 提交形态   ⇒ **是否单核受限**（core_mask / task_number / subcore 分布）
  2) LUT        ⇒ **是 LUT 还是多项式**（整表突发次数与规模、旁路戳）
  3) 算子类别   ⇒ CNA 块出现程度（无 CNA=元素级 / +54=归约归一化 / +80=conv / +1=BN）
  4) op 维度    ⇒ 0x1020(DATAIN w,h) / 0x3014(DATAOUT w,h) / 0x100c(CON0)

口径（台账 §7.118/§7.134/§7.146/§7.151）：
  * 载荷在 **event=="regcmd"**（不是 task ✗）；条目 **8 B**（reg:u16 LE@0, val:u32 LE@2）；
  * 捕获**必须**设 `EPERF_RKNPU_REGCMD_MAX_BYTES`（否则 hex 为空 ⇒ 全部统计无意义 ✗）；
  * 表分析**先排序去重**（写入顺序会重复首值 ⇒ 假"平台" ✗）。
"""
import sys, json, struct, collections, os

CNA_LO, CNA_HI = 0x100c, 0x1184
LUT_DATA, LUT_LO, LUT_HI = 0x4104, 0x4100, 0x411c

def load(path):
    evs = []
    for ln in open(path, errors="ignore"):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        evs.append(ev)
    return evs

def analyze(path):
    r = dict(submits=collections.Counter(), tasks=0, regcmd=0,
             lut_area=collections.Counter(), lut_vals=collections.Counter(),
             bursts=[], regs=set(), dims=collections.Counter(), opidx=collections.Counter())
    for ev in load(path):
        et = ev.get("event")
        if et == "submit":
            r["submits"][(ev.get("task_number"), ev.get("core_mask"))] += 1
        elif et == "task":
            r["tasks"] += 1
        elif et == "regcmd":
            b = bytes.fromhex(ev.get("hex") or "")
            if not b:
                continue
            r["regcmd"] += 1
            n4104 = 0
            vals = []
            for i in range(0, len(b) - 7, 8):
                reg = struct.unpack_from("<H", b, i)[0]
                val = struct.unpack_from("<I", b, i + 2)[0]
                r["regs"].add(reg)
                if LUT_LO <= reg <= LUT_HI:
                    r["lut_area"][reg] += 1
                if reg == LUT_DATA:
                    n4104 += 1
                    vals.append(val)
                if CNA_LO <= reg <= CNA_HI:
                    pass
                if reg == 0x1020:
                    r["dims"][("in", val & 0xffff, (val >> 16) & 0xffff)] += 1
                if reg == 0x3014:
                    r["dims"][("out", val & 0xffff, (val >> 16) & 0xffff)] += 1
            if n4104 >= 100:
                r["bursts"].append((ev.get("op_idx"), n4104))
            r["opidx"][ev.get("op_idx")] += 1
    return r

def main():
    args = sys.argv[1:]
    check = None
    if args and args[0] == "--check":
        check = args[1]; args = args[2:]
    if not args:
        print(__doc__); return 1
    buf = []
    orig = print
    def cap(*a, **k):
        buf.append(" ".join(str(x) for x in a))
    globals()['print'] = cap
    for p in args:
        r = analyze(p)
        name = os.path.basename(p)
        print("=" * 78)
        print("TRACE:", name)
        # 1) 派发形态
        if r["submits"]:
            top = ", ".join("%s x%d" % (k, v) for k, v in r["submits"].most_common(3))
            masks = {k[1] for k in r["submits"]}
            verdict = "单核受限? 否（core_mask=0 且 task 按 subcore 分派）" if masks <= {0} else \
                      ("core_mask>1 ⇒ 走 use_core_num 路径" if max(masks) > 1 else "core_mask=1")
            print("  1) 派发: %s  ⇒ %s" % (top, verdict))
        # 2) LUT
        d = r["lut_area"].get(LUT_DATA, 0)
        bursts = r["bursts"]
        if bursts:
            tables = collections.Counter(n for _, n in bursts)
            print("  2) LUT : 整表突发 %d 次（规模 %s）⇒ **LUT 路径** ✓（表规模≈最大突发）" %
                  (len(bursts), dict(tables)))
        else:
            print("  2) LUT : 无 ≥100 条的突发（0x4104 共 %d 次，多为旁路戳）⇒ **非 LUT（多项式/旁路）**" % d)
        # 3) 算子类别
        cna = sorted(x for x in r["regs"] if CNA_LO <= x <= CNA_HI)
        if len(cna) >= 70:
            cls = "conv（完整 CNA 块 %d 个）" % len(cna)
        elif len(cna) >= 20:
            cls = "归约/归一化（softmax/layernorm/attention，CNA 子集 %d 个）" % len(cna)
        elif len(cna) == 0:
            cls = "元素级（无 CNA：激活/rcp/sqrt/sin…）"
        else:
            cls = "混合/其它（CNA %d 个）" % len(cna)
        print("  3) 类别: %s" % cls)
        # 4) 维度
        if r["dims"]:
            items = [("%s %dx%d" % (k[0], k[1], k[2]), n) for k, n in r["dims"].most_common(4)]
            print("  4) 维度: %s" % ", ".join("%s x%d" % (t, n) for t, n in items))
    globals()['print'] = orig
    out = "\n".join(buf)
    if check:
        exp = open(check).read().rstrip("\n").split("\n")
        got = out.split("\n")
        bad = [(i, e, g) for i, (e, g) in enumerate(zip(exp, got)) if e != g]
        if len(exp) != len(got):
            bad.append(("长度", str(len(exp)), str(len(got))))
        print("CHECK: %s（差异 %d 处）" % (check, len(bad)))
        for b in bad[:8]:
            print("   ≠ 行%s: 期望 %r 实际 %r" % b)
        return 0 if not bad else 2
    print(out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
