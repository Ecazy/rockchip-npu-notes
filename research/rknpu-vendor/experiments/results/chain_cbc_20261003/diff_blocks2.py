import json, re

def entries(b):
    out = []
    for i in range(len(b) // 8):
        w = int.from_bytes(b[i*8:(i+1)*8], 'little')
        out.append(((w >> 48) & 0xFFFF, w & 0xFFFF, (w >> 16) & 0xFFFFFFFF))
    return out

ven = {}
for ln in open('/mnt/ssd/workspace/cbc_fp16.jsonl', errors='ignore'):
    if '"event":"regcmd"' not in ln:
        continue
    e = json.loads(ln)
    if e['task_index'] < 2:
        ven[e['task_index']] = bytes.fromhex(e['hex'])

our = []
for ln in open('/tmp/ourblk.jsonl', errors='ignore'):
    m = re.match(r'0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)', ln)
    if m:
        our.append((int(m.group(1), 16), int(m.group(2), 16)))

ve = entries(ven[0])
print(f"vendor task0: {len(ve)} entries; ours {len(our)} entries")

# target 推导（与 emitter 同规则）
def tgt_of(reg):
    if 0x1000 <= reg < 0x2000: return 0x201
    if 0x4000 <= reg < 0x5000: return 0x1001
    if 0x5000 <= reg < 0x6000: return 0x2001
    if 0x3000 <= reg < 0x4000: return 0x3001
    return None

n = min(len(ve), len(our))
nd = 0
for i in range(min(n, 126)):
    vt, vr, vv = ve[i]
    rr, ov = our[i]
    if vr != rr or vv != ov:
        if nd < 20:
            print(f"  [{i:3d}] vendor reg=0x{vr:04x} val=0x{vv:08x}   ours reg=0x{rr:04x} val=0x{ov:08x}"
                  f"   {'REG' if vr != rr else 'VAL'}")
        nd += 1
print(f"config diffs (first 126): {nd}")
# 尾部对比
print("vendor tail:", [(hex(t), hex(r), hex(v)) for t, r, v in ve[126:130]])
print("our tail   :", [(hex(r), hex(v)) for r, v in our[-4:]] if len(our) >= 4 else "n/a")
