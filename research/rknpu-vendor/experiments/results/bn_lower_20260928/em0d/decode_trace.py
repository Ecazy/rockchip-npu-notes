import json, struct, sys, re

def load_regmap():
    names = {}
    pat = re.compile(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)")
    txt = open("/home/maoyu/Documents/code/RKNPU/experiments/rnpu_regmap.h").read()
    for m in pat.finditer(txt):
        reg, val = m.group(1), int(m.group(2), 16)
        base = re.sub(r"__.*$", "", reg)
        if val not in names:
            names[val] = base
    return names

TARGETS = {0x0100:"PC", 0x0200:"CNA", 0x0800:"CORE", 0x1000:"DPU", 0x2000:"DPU_RDMA",
           0x4000:"PPU", 0x8000:"PPU_RDMA", 0x10000:"DDMA", 0x20000:"SDMA", 0x40000:"GLOBAL"}

def target_str(t):
    base = t & 0xFFF00
    sub = t & 0xF000F  # extra bits
    name = TARGETS.get(base, hex(base))
    extra = ""
    if t & 0xFF and base in TARGETS:
        extra = f"|{t & 0xFF:x}"
    if t & ~0xFFF00 & 0xFF00:
        extra += f"+hi{t>>8 & 0xff:x}"
    return name + extra

def decode(path, task_index=None):
    names = load_regmap()
    recs = [json.loads(l) for l in open(path)]
    for r in recs:
        if r.get("event") != "regcmd":
            continue
        if task_index is not None and r["task_index"] != task_index:
            continue
        h = r["hex"]
        data = bytes.fromhex(h)
        n = len(data) // 8
        print(f"== task {r['task_index']} op_idx={r['op_idx']} entries={r['regcfg_amount']} bytes={len(data)}")
        for i in range(n):
            v = struct.unpack_from("<Q", data, i * 8)[0]
            if v == 0:
                continue
            tgt = (v >> 48) & 0xFFFF
            val = (v >> 16) & 0xFFFFFFFF
            reg = v & 0xFFFF
            nm = names.get(reg, "?")
            print(f"  [{i:3d}] {target_str(tgt):14s} reg=0x{reg:04x} val=0x{val:08x} ({val}) {nm}")

if __name__ == "__main__":
    decode(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
