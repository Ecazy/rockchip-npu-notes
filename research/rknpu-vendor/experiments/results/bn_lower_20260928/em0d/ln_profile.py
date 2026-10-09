import json, struct

names = {}
import re
txt = open("/home/maoyu/Documents/code/RKNPU/experiments/rnpu_regmap.h").read()
for m in re.finditer(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)", txt):
    reg, val = m.group(1), int(m.group(2), 16)
    if val not in names: names[val] = re.sub(r"__.*$", "", reg)
BLK = {0x0100:"PC",0x0200:"CNA",0x0800:"CORE",0x1000:"DPU",0x2000:"RDMA",0x4000:"PPU",0x8000:"PRDMA"}

recs = [json.loads(l) for l in open("trace_layernorm_only_fp16.jsonl")]
by_task = {}
for r in recs:
    if r.get("event") == "regcmd":
        data = bytes.fromhex(r["hex"])
        blocks = {}
        cube = {}
        for i in range(len(data)//8):
            v = struct.unpack_from("<Q", data, i*8)[0]
            if v == 0: continue
            tgt = (v >> 48) & 0xFFFF; val = (v >> 16) & 0xFFFFFFFF; reg = v & 0xFFFF
            b = tgt & 0xFFF00
            nm = BLK.get(b, hex(b))
            blocks[nm] = blocks.get(nm, 0) + 1
            if reg in (0x4020,0x4030,0x4034,0x403c,0x400c,0x4010,0x4080,0x4084):
                names.setdefault(f"DPU_{reg:04x}", reg)
                cube[f"reg{reg:04x}"] = val
        print(f"task{r['task_index']:2d} op={r['op_idx']} em={r.get('enable_mask',0):#04x} blocks={sorted(blocks.items())} dst={cube.get('reg4020',0):#x} cube={cube.get('reg4030','')},{cube.get('reg4034','')},C={cube.get('reg403c',''):#x} fmt={cube.get('reg4010',0):#x} fmc={cube.get('reg400c',0):#x}")
