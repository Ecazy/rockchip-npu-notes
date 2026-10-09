#!/usr/bin/env python3
import struct, sys

OC = 16
OW = OH = 6
QPX = OW * OH * OC  # 576

def load(p, n=QPX):
    d = open(p, 'rb').read()
    assert len(d) >= n, (p, len(d))
    return list(struct.unpack('<%db' % n, d[:n]))

# hw buffer: our_off(oc,x,y) = (x*OH + y)*OC + oc   [X-MAJOR npu_off]
def our_off(oc, x, y): return (x * OH + y) * OC + oc
# cpu ref: cpu_off(oc,x,y) = (y*OW + x)*OC + oc
def cpu_off(oc, x, y): return (y_ := 0) or (y * OW + x) * OC + oc if False else (y * OW + x) * OC + oc

def law_cpu_m(n):  # buffer atom n holds cpu linear pixel 8*(n mod 6) + n//6
    return 8 * (n % 6) + (n // 6)

def check(name, npu, cpu):
    id_mis = law_mis = 0
    law_ok_dropped = 0
    for oc in range(OC):
        for y in range(OH):
            for x in range(OH):
                hv = npu[our_off(oc, x, y)]
                # identity expectation
                cv = cpu[(y * OW + x) * OC + oc]
                if hv != cv: id_mis += 1
                # permutation law: our-read (x,y) shows cpu pixel m=8y+x (if valid)
                m = 8 * y + x
                if m < 36:
                    cy, cx = divmod(m, OW)
                    cv2 = cpu[(cy * OW + cx) * OC + oc]
                    if hv == cv2: law_ok_dropped += 1
        # forward direction over atoms
        for n in range(36):
            got = npu[n * OC + oc]
            m = law_cpu_m(n)
            if m >= 36: continue
            exp = cpu[m * OC + oc]
            if got == exp: law_mis += 1  # count law MATCHES
    # note: law_mis here counts matches; print clearly
    print('%s: identity-mis=%d/576' % (name, id_mis))
    print('   our-read matches law cpu[8y+x]: %d/576 (540 possible)' % law_ok_dropped)
    print('   forward-law atom matches: %d/576 (max 528=16oc*33valid-atoms)' % law_mis)

npu_post = load('/tmp/opencode/wgdata_valid/q_npu_post_b0.bin')
cpu_post = load('/tmp/opencode/wgdata_valid/q_cpu_post_b0.bin')
npu_pre  = load('/tmp/opencode/wgdata_valid/q_npu_pre_b0.bin')
cpu_q0   = load('/tmp/opencode/wgdata_valid/q_cpu_b0.bin')

print('=== FWD2 (npu_post vs cpu_post, correct layouts) ===')
check('FWD2', npu_post, cpu_post)
print('=== FWD1 (npu_pre vs q_cpu_b0) ===')
check('FWD1', npu_pre, cpu_q0)
