#!/bin/sh
# conv -> pool -> conv 三层 NPU 管线（含闭环校验）—— §7.192
#   层1: min_conv_exp 16x16x32 --k3p1--> 16x16x32 （自带 int8 CPU 参考）
#   层2: min_pool     16x16x32 --k2s2p0--> 8x8x32   （--in-file --in-yiw：卷积输出 y 主序 → 池化输入 x 主序）
#   层3: min_conv_exp  8x8x32  --k3p1--> 8x8x32    （--in-file --in-xiw：池化输出 x 主序 → 硬件 y 主序）
#   校验: 各 conv 用工具自带参考；pool 用 Python 独立 maxpool 参考逐字节比对 + 两次运行确定性
# 用法: sh pipeline_conv_pool_conv.sh <工作目录>
set -e
D=${1:-/tmp/cpc}
CONV=/mnt/ssd/workspace/rknpu-re/exp2
POOL=/mnt/ssd/workspace/rknpu-re/pool
mkdir -p "$D"

echo "== 1) 层1 conv（16x16x32 --k3p1--> 16x16x32, int8, 自带参考）=="
(cd "$CONV" && ./min_conv_exp --pattern --pat2 --asym-w --pad-con1 0 --auto-fix --iw 16 --ih 16 --ic 32 --oc 32 --kw 3 --kh 3 --pad 1 \
   --out-file "$D/conv1") | grep -E "OUT-FILE|total=|RESULT|OUT-HASH"

echo "== 2) 层2 pool（16x16x32 --k2s2p0--> 8x8x32, 读 conv1 输出）=="
(cd "$POOL" && ./min_pool --iw 16 --ih 16 --ic 32 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 \
   --in-file "$D/conv1" --in-yiw --out-file "$D/pool") | grep -E "IN-FILE|OUT-FILE|mismatch="

echo "== 3) 层3 conv（8x8x32 --k3p1--> 8x8x32, 读 pool 输出, int8, 自带参考）=="
(cd "$CONV" && ./min_conv_exp --pattern --pat2 --asym-w --pad-con1 0 --auto-fix --iw 8 --ih 8 --ic 32 --oc 32 --kw 3 --kh 3 --pad 1 \
   --in-file "$D/pool.out" --in-xiw --out-file "$D/conv2") | grep -E "IN-FILE|total=|RESULT|OUT-HASH"

echo "== 4) 独立参考：Python 对 conv1 输出做 2x2 maxpool，逐字节比对 pool.out =="
python3 - "$D" <<'PY'
import sys
D = sys.argv[1]
IW = IH = 16; IC = 32; OW = OH = 8
ATOM = 16
conv1 = open(D + "/conv1", "rb").read()
pool  = open(D + "/pool.out", "rb").read()
groups = IC // ATOM
assert len(conv1) == groups * IW * IH * ATOM, (len(conv1), groups*IW*IH*ATOM)
assert len(pool)  == groups * OW * OH * ATOM, (len(pool),  groups*OW*OH*ATOM)

def s8(b): return b - 256 if b >= 128 else b   # 硬件按 int8 有符号比较

def conv_y(g, x, y, c):           # 卷积输出：y 主序 g*w*h*16 + y*w*16 + x*16
    return s8(conv1[g*IW*IH*ATOM + y*IW*ATOM + x*ATOM + c])

def pool_x(g, ox, oy, c):         # 池化输出：x 主序 g*ow*oh*16 + ox*oh*16 + oy*16
    return s8(pool[g*OW*OH*ATOM + ox*OH*ATOM + oy*ATOM + c])

bad = 0; total = 0; maxd = 0
for g in range(groups):
    for ox in range(OW):
        for oy in range(OH):
            for c in range(ATOM):
                best = -129
                for ky in range(2):
                    for kx in range(2):
                        v = conv_y(g, ox*2 + kx, oy*2 + ky, c)
                        if v > best: best = v
                hw = pool_x(g, ox, oy, c)
                total += 1
                d = abs(hw - best)
                if d > maxd: maxd = d
                if hw != best: bad += 1
print(f"  PY-REF: total={total} mismatch={bad} max|diff|={maxd}")
print("  PY-REF RESULT: " + ("BIT-EXACT (independent maxpool)" if bad == 0 else "MISMATCH"))
PY

echo "== 5) 确定性：重跑层1+层2+层3，比对哈希 =="
(cd "$CONV" && ./min_conv_exp --pattern --pat2 --asym-w --pad-con1 0 --auto-fix --iw 16 --ih 16 --ic 32 --oc 32 --kw 3 --kh 3 --pad 1 \
   --out-file "$D/conv1b" >/dev/null 2>&1)
(cd "$POOL" && ./min_pool --iw 16 --ih 16 --ic 32 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 \
   --in-file "$D/conv1b" --in-yiw --out-file "$D/poolb" >/dev/null 2>&1)
(cd "$CONV" && ./min_conv_exp --pattern --pat2 --asym-w --pad-con1 0 --auto-fix --iw 8 --ih 8 --ic 32 --oc 32 --kw 3 --kh 3 --pad 1 \
   --in-file "$D/poolb.out" --in-xiw --out-file "$D/conv2b" >/dev/null 2>&1)
python3 - "$D" <<'PY'
import sys, hashlib
D = sys.argv[1]
def h(p): return hashlib.sha256(open(D + "/" + p, "rb").read()).hexdigest()
for a, b in [("conv1","conv1b"), ("pool.out","poolb.out"), ("conv2","conv2b")]:
    ha, hb = h(a), h(b)
    print(f"  {a:10s} {ha[:16]}  vs  {b:10s} {hb[:16]}  {'OK' if ha==hb else 'DIFF'}")
print("  sha256 conv2 =", h("conv2"))
PY
echo "== DONE =="
