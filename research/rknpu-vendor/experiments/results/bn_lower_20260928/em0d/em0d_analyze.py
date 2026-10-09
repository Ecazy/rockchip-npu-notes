import json, struct, re, sys
RG = {}
pat = re.compile(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)")
for m in pat.finditer(open("/home/maoyu/Documents/code/RKNPU/experiments/rnpu_regmap.h").read()):
    reg, val = m.group(1), int(m.group(2),16)
    RG.setdefault(val, re.sub(r"__.*$","",reg))
TGT={0x0100:"PC",0x0200:"CNA",0x0800:"CORE",0x1000:"DPU",0x2000:"DPU_RDMA",0x4000:"PPU",0x8000:"PPU_RDMA",0x10000:"DDMA",0x20000:"SDMA",0x40000:"GLOBAL"}
def tstr(t):
    return TGT.get(t&0xFFF00, hex(t&0xFFF00)) + (("|%x"%(t&0xff)) if (t&0xff) else "")
def f16(u): return struct.unpack('<e', struct.pack('<H', u&0xffff))[0]
recs=[json.loads(l) for l in open(sys.argv[1])]
te={r['task_index']:r for r in recs if r.get('event')=='task'}
reg={r['task_index']:r for r in recs if r.get('event')=='regcmd'}
lo=int(sys.argv[2]) if len(sys.argv)>2 else 0
hi=int(sys.argv[3]) if len(sys.argv)>3 else lo+1
for ti in range(lo,hi):
    t=te[ti]; r=reg[ti]
    print(f"\n===== task {ti} op={t['op_idx']} em=0x{t['enable_mask']:02x} amount={t['regcfg_amount']} =====")
    data=bytes.fromhex(r['hex'])
    for i in range(len(data)//8):
        v=struct.unpack_from('<Q',data,i*8)[0]
        tgt=(v>>48)&0xffff; val=(v>>16)&0xffffffff; rr=v&0xffff
        if rr==0 and val==0: continue
        nm=RG.get(rr,'?')
        extra=""
        if 0x4030<=rr<=0x403c or rr in (0x1050,0x1054,0x1058,0x105c):
            pass
        if rr==0x4048 or rr==0x4068:
            extra=f"  hi16={f16(val>>16):.6g} lo16={f16(val&0xffff):.6g}"
        print(f"  {tstr(tgt):10s} 0x{rr:04x}={val:#010x} {nm}{extra}")
