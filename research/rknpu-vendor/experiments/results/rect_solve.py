#!/usr/bin/env python3
import struct

act = open('/tmp/opencode/mc_act_bo.bin','rb').read()
wt  = open('/tmp/opencode/mc_wt_bo.bin','rb').read()
IW, IH, OW, OH = 3, 2, 2, 2
IC, OC = 3, 16
OUT_OFF = 32768

def s8(b): return b - 256 if b >= 128 else b

# input cube (normal path): npu_off(0,x,y,iw,ih) + ch = (x*IH + y)*16 + ch
def X(x, y, icc):
    return s8(act[(x*IH + y)*16 + icc])

# weight cube layout A: idx = ((x*kh + y)*ic1*... -> ((tap)*16 + oc)*16 + icc
# (ic1=1, oc1n=1, oc2n=16 for oc_w=16)
def W(tap, oc, icc):
    return s8(wt[(tap*OC + oc)*16 + icc])

# hw output: npu_off(0,ox,oy,ow,oh) + oc = (ox*OH + oy)*16 + oc
def HW(ox, oy, oc):
    return s8(act[OUT_OFF + (ox*OH + oy)*16 + oc])

# true taps for kw=2 kh=1: s=0 -> (kx=0,ky=0) X(ox,oy); s=1 -> (kx=1,ky=0) X(ox+1,oy)
# candidate models: tap s uses X(ox+dx, oy+dy) * W(cube_tap)
offset = -128
def req(acc):  # requant approx: acc*16385>>14 - 128, clamp
    v = acc * 16385
    sign = 1 if v < 0 else 0
    guide = (v >> 13) & 1
    sticky = 1 if (v & ((1 << 13) - 1)) != 0 else 0
    r = (v >> 14) + (guide & ((1 - sign) | sticky)) + offset
    return max(-128, min(127, r))

print('== verify packing: X and W as parsed ==')
print('X(0,0)=', [X(0,0,c) for c in range(3)], ' X(1,0)=', [X(1,0,c) for c in range(3)])
print('W(0,0,icc) for oc0 =', [W(0,0,c) for c in range(3)], ' W(1,0,..) =', [W(1,0,c) for c in range(3)])

print('\n== observed hw acc (byte+128, only valid range) vs true acc ==')
def true_acc(ox, oy, oc, model=None):
    acc = 0
    for kx in range(2):
        ix = ox + kx; iy = oy
        if ix >= IW: continue
        for icc in range(IC):
            acc += X(ix, iy, icc) * W(kx, oc, icc)
    return acc

data = []
for oc in range(OC):
    for oy in range(OH):
        for ox in range(OW):
            hb = HW(ox, oy, oc)
            hw_acc = hb - offset
            ta = true_acc(ox, oy, oc)
            data.append((ox, oy, oc, hb, hw_acc, ta, req(ta)))

# show first 8
for d in data[:8]:
    print('  (%d,%d) oc=%d: hw_byte=%4d hw_acc=%4d true_acc=%4d req(true)=%4d %s'
          % (d[0], d[1], d[2], d[3], d[4], d[5], d[6], 'OK' if d[3]==d[6] else '<<'))

print('\n== brute-force tap model search ==')
# model: tap s = (dx, dy, cube_tap); hw_acc = sum_s sum_icc X(ox+dx, oy+dy, icc)*W(cube_tap, oc, icc)
import itertools
positions = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)]
best = []
for p0 in positions:
    for t0 in (0,1):
        for p1 in positions:
            for t1 in (0,1):
                ok = 0; tot = 0
                for (ox,oy,oc,hb,hw_acc,ta,_) in data:
                    pred = 0
                    for (dx,dy),ct in ((p0,t0),(p1,t1)):
                        ix, iy = ox+dx, oy+dy
                        if 0 <= ix < IW and 0 <= iy < IH:
                            for icc in range(IC):
                                pred += X(ix, iy, icc) * W(ct, oc, icc)
                    tot += 1
                    if req(pred) == hb: ok += 1
                best.append((ok, tot, p0, t0, p1, t1))
best.sort(reverse=True)
for b in best[:5]:
    print('  model ok=%d/%d  tap0: X+(*,%s)*W%d  tap1: X+(*,%s)*W%d' % (b[0], b[1], b[2], b[3], b[4], b[5]))
