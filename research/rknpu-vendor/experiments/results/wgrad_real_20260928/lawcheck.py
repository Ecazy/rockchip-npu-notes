#!/usr/bin/env python3
import struct

IW = IH = 8; IC = 3; OC = 16; KW = KH = 3
X = open('/tmp/opencode/wgdata_valid/x_b0.bin','rb').read()
W = open('/tmp/opencode/wgdata_valid/w.bin','rb').read()
QF = open('/tmp/opencode/wgdata_valid/q_cpu_b0.bin','rb').read()   # dataset's stored Q (576B)
NP = open('/tmp/opencode/wgdata_valid/q_npu_pre_b0.bin','rb').read()  # hw fwd (576B)

xs = struct.unpack('<%db' % (IW*IH*IC), X)      # [y*IW+x][ic]
ws = struct.unpack('<%db' % (KW*KH*IC*OC), W)   # [(ky*KW+kx)*IC+ic][oc]

def requant(acc):
    v = acc * 16385
    sh = 14
    if sh == 0:
        r = v
    else:
        sign = 1 if v < 0 else 0
        guide = (abs(v) >> (sh-1)) & 1 if False else ((v >> (sh-1)) & 1)
        sticky = 1 if (sh > 1 and (v & ((1 << (sh-1)) - 1)) != 0) else 0
        round_up = guide & ((1 - sign) | sticky)
        r = (v >> sh) + round_up
    return max(-128, min(127, r))

def make_conv(pad, ow, oh):
    q = {}
    for oc in range(OC):
        for y in range(oh):
            for x in range(ow):
                acc = 0
                for ky in range(KH):
                    for kx in range(KW):
                        iy = y + ky - pad; ix = x + kx - pad
                        if iy < 0 or iy >= IH or ix < 0 or ix >= IW: continue
                        for ic in range(IC):
                            acc += xs[(iy*IW+ix)*IC+ic] * ws[((ky*KW+kx)*IC+ic)*OC+oc]
                q[(x,y,oc)] = requant(acc)
    return q

q6 = make_conv(0, 6, 6)
q8 = make_conv(0, 8, 8)

def rd6(buf, x, y, oc):  # 6-wide read
    return struct.unpack('<b', buf[(y*6+x)*OC+oc:(y*6+x)*OC+oc+1])[0]
def rd6x(buf, x, y, oc):  # X-major 6-wide read (npu_off)
    return struct.unpack('<b', buf[(x*6+y)*OC+oc:(x*6+y)*OC+oc+1])[0]
def rd8(buf, x, y, oc):
    off = (y*8+x)*OC+oc
    if off+1 > len(buf): return None
    return struct.unpack('<b', buf[off:off+1])[0]

# Compare QF (stored cpu ref) against q6/q8 under different rasters
def cmp(name, f):
    bad = 0
    for oc in range(OC):
        for y in range(6):
            for x in range(6):
                if f(oc, x, y) != q6[(x,y,oc)]: bad += 1
    print('%-40s vs conv6(x-major-hw-read): mis=%d/576' % (name, bad))

print('--- hw buffer (NP) analysis ---')
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd6x(NP, x, y, oc) != q6[(x,y,oc)])
print('hw(X-major6) vs conv6: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd6(NP, x, y, oc) != q6[(x,y,oc)])
print('hw(row-major6) vs conv6: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd8(NP, x, y, oc) is not None and rd8(NP, x, y, oc) != q6[(x,y,oc)])
print('hw(row-major8) vs conv6: mis=%d/576' % bad)

print('--- stored cpu ref (QF) analysis ---')
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd6(QF, x, y, oc) != q6[(x,y,oc)])
print('QF(row-major6) vs conv6: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd8(QF, x, y, oc) != q6[(x,y,oc)])
print('QF(row-major8) vs conv6: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd6(QF, x, y, oc) != q8[(x,y,oc)])
print('QF(row-major6) vs conv8: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd8(QF, x, y, oc) != q8[(x,y,oc)])
print('QF(row-major8) vs conv8: mis=%d/576' % bad)
bad = sum(1 for oc in range(OC) for y in range(6) for x in range(6) if rd6x(NP, x, y, oc) != rd6(QF, x, y, oc))
print('hw(X-major6) vs QF(row-major6) [the C-side comparison]: mis=%d/576' % bad)
