#!/usr/bin/env python3
"""Build cbc v2 full-fidelity replay images (§7.185).

cbc_vmem.bin : 0x9000 B composite mirror of vendor IOVA [0xffff6000, 0xfffff000)
               sentinel zones + real weight/BS blobs + regcmd stream (blocks 0-5
               overwritten with runtime-final trace bytes)
cbc_task.bin : blob[8] verbatim (17 x 40 B descriptors, regcmd = stream-relative)
"""
import hashlib
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments"))
from rknn_blob_dump import parse  # noqa: E402

WBO = 0x4000            # composite offset of vendor 0xffffa000 BO
STREAM_OFF = WBO + 0x940

m = parse("/tmp/opencode/cbc/conv_bn_conv_fp16.rknn")
B = {bl["idx"]: bl["bytes"] for bl in m["blobs"]}
w0, b0, w1, b1, mc_o, mc_i, stream, task = (B[i] for i in range(1, 9))
assert (len(w0), len(b0), len(w1), len(b1), len(mc_o), len(mc_i),
        len(stream), len(task)) == (128, 64, 64, 64, 1024, 1024, 13696, 680)

VM = bytearray(0x9000)
for i in range(0x0000, 0x1000):
    VM[i] = (i * 37 + 0x11) & 0xFF          # DST1 bn-out  (t1 DST, sentinel A)
for i in range(0x1000, 0x2000):
    VM[i] = i & 0xFF                        # t0 CNA input ramp
for i in range(0x2000, 0x3000):
    VM[i] = (i * 53 + 0x2A) & 0xFF          # t0 DST + inter sentinel B

def put(off, data):
    VM[off:off + len(data)] = data

put(0x4000 + 0x0000, w0)   # DCOMP base 0xffffa000
put(0x4000 + 0x0080, b0)   # RDMA_BS for t0 (0xffffa080)
put(WBO + 0x00c0, w1)
put(WBO + 0x0100, b1)
put(WBO + 0x0140, mc_o)    # tempMC11_output
put(WBO + 0x0540, mc_i)    # tempMC11_input  (PPU_RDMA 0xffffa540)
put(WBO + 0x0940, stream)

blocks = open(os.path.join(HERE, "cbc_blocks.bin"), "rb").read()
p = 0
for off, nent in zip([0x0000, 0x0440, 0x0880, 0x0C80, 0x1080, 0x1180],
                     [130, 130, 126, 126, 30, 126]):
    n = nent * 8
    VM[STREAM_OFF + off:STREAM_OFF + off + n] = blocks[p:p + n]
    p += n
assert p == len(blocks), (p, len(blocks))

open(os.path.join(HERE, "cbc_vmem.bin"), "wb").write(VM)
open(os.path.join(HERE, "cbc_task.bin"), "wb").write(task)

print("== descriptors (blob[8], regcmd = stream-blob-relative) ==")
for k in (0, 1, 4, 5):
    fl, op, em, im, ic, ist, amt, roff = struct.unpack_from("<8I", task, k * 40)
    rc = struct.unpack_from("<Q", task, k * 40 + 32)[0]
    print("desc[%2d] flags=%d op=%d em=0x%x im=0x%x ic=0x%x ist=%d amt=%d roff=%d regcmd=0x%x"
          % (k, fl, op, em, im, ic, ist, amt, roff, rc))

print("== pointer entries in stream region ==")
nptr = 0
for o in range(STREAM_OFF, STREAM_OFF + len(stream), 8):
    w = struct.unpack_from("<Q", VM, o)[0]
    val = (w >> 16) & 0xFFFFFFFF
    if val >= 0xFFFF0000:
        nptr += 1
        reg, tgt = w & 0xFFFF, w >> 48
        print("  ptr reg=0x%04x tgt=0x%03x val=0x%08x" % (reg, tgt, val))
        assert 0xFFFF6000 <= val < 0xFFFFF000, "out of mirror: 0x%08x" % val
print("total pointer entries:", nptr)

for name in ("cbc_vmem.bin", "cbc_task.bin"):
    d = open(os.path.join(HERE, name), "rb").read()
    print("%s: %d B sha256=%s" % (name, len(d), hashlib.sha256(d).hexdigest()))
