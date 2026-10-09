#!/bin/sh
# conv -> pool -> conv（训练闭环）三层微训练步 —— §7.193
#   层1 conv: min_conv_exp 32x32x32 --k3p1--> 32x32x32 （NPU，自带参考）
#   层2 pool: min_pool     32x32x32 --k2s2p0--> 16x16x32（--in-file --in-yiw，自带参考 + 独立 maxpool）
#   层3 conv: wgrad_real   16x16x32 --k3p1--> 16x16x32（外部 X=真实池化激活；前向 + wgrad + int8
#             W'=W-lr*dW 写回 + 复前向自证 = CLOSED-LOOP OK）
# 用法: sh pipeline_conv_pool_conv_train.sh <工作目录>
set -e
D=${1:-/tmp/cpc_train}
CONV=/mnt/ssd/workspace/rknpu-re/exp2
POOL=/mnt/ssd/workspace/rknpu-re/pool
EXP3=/mnt/ssd/workspace/rknpu-re/exp3
mkdir -p "$D"

echo "== 1) 层1 conv（32x32x32 --k3p1--> 32x32x32, int8, 自带参考）=="
(cd "$CONV" && ./min_conv_exp --pattern --pat2 --asym-w --pad-con1 0 --auto-fix \
   --iw 32 --ih 32 --ic 32 --oc 32 --kw 3 --kh 3 --pad 1 \
   --out-file "$D/conv1") | grep -E "OUT-FILE|total=|RESULT"

echo "== 2) 层2 pool（32x32x32 --k2s2p0--> 16x16x32, 读 conv1 输出）=="
(cd "$POOL" && ./min_pool --iw 32 --ih 32 --ic 32 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 \
   --in-file "$D/conv1" --in-yiw --out-file "$D/pool") | grep -E "IN-FILE|OUT-FILE|mismatch="

echo "== 3) 独立参考：Python 对 conv1 输出做 2x2 maxpool，逐字节比对 pool.out =="
python3 - "$D" <<'PY'
import sys
D=sys.argv[1]; IW=IH=32; IC=32; OW=OH=16; ATOM=16
conv1=open(D+"/conv1","rb").read(); pool=open(D+"/pool.out","rb").read()
groups=IC//ATOM
assert len(conv1)==groups*IW*IH*ATOM and len(pool)==groups*OW*OH*ATOM
s8=lambda b:(b-256 if b>=128 else b)
c1=lambda g,x,y,c: s8(conv1[g*IW*IH*ATOM + y*IW*ATOM + x*ATOM + c])
po=lambda g,x,y,c: s8(pool[g*OW*OH*ATOM + x*OH*ATOM + y*ATOM + c])
bad=maxd=0; tot=0
for g in range(groups):
  for ox in range(OW):
    for oy in range(OH):
      for c in range(ATOM):
        best=max(c1(g,ox*2+kx,oy*2+ky,c) for ky in range(2) for kx in range(2))
        hw=po(g,ox,oy,c); tot+=1; bad+=(hw!=best); maxd=max(maxd,abs(hw-best))
print(f"  PY-REF: total={tot} mismatch={bad} max|diff|={maxd}")
print("  PY-REF RESULT: " + ("BIT-EXACT (independent maxpool)" if bad==0 else "MISMATCH"))
PY

echo "== 4) 构建层3训练器（-DIW=16 -DIH=16 -DIC=32 -DOC=32）并生成数据集 =="
(cd "$EXP3" && gcc -O2 -o deep/wgrad_i32 -DIW=16 -DIH=16 -DIC=32 -DOC=32 wgrad_real.c -lm \
   && echo "  BUILD_OK deep/wgrad_i32" \
   && ./deep/wgrad_i32 --gen-only --dir "$D" --pad 1 --seed 20260928 >/dev/null 2>&1)

echo "== 5) 真实池化激活（x 主序 → 行主序，含通道组）覆盖 x_b0.bin =="
python3 - "$D" <<'PY'
import sys
D=sys.argv[1]; IW=IH=16; IC=32; ATOM=16
d=open(D+"/pool.out","rb").read()
out=bytearray(IW*IH*IC)
for g in range(IC//ATOM):
    for x in range(IW):
        for y in range(IH):
            for c in range(ATOM):
                out[(y*IW+x)*IC + g*ATOM + c] = d[g*IW*IH*ATOM + x*IH*ATOM + y*ATOM + c]
open(D+"/x_b0.bin","wb").write(bytes(out))
print("  transposed", len(out), "B ->", D+"/x_b0.bin (group-aware)")
PY

echo "== 6) 层3 conv 训练闭环（外部 X=真实激活；工具自带 dY）=="
(cd "$EXP3" && ./deep/wgrad_i32 --dir "$D" --pad 1 --batch 1 --in-x-file "$D/x_b0.bin" --update) \
  | grep -E "IN-X-FILE|FWD b=0|WGRAD_TOTAL|UPDATE:|FWD2|RESULT"

echo "== 7) 梯度缩放：稀疏 dY（密度 1/64）→ 消饱和的量化感知训练步 =="
python3 - "$D" <<'PY'
import sys, random
D=sys.argv[1]; IW=IH=16; IC=32; n=IW*IH*IC
random.seed(11); b=bytearray(n)
for i in range(n):
    if random.random() < 1/64:
        b[i] = (1 if random.random() < 0.5 else -1) & 0xff
open(D+"/dy_sparse.bin","wb").write(bytes(b))
print("  wrote dy_sparse.bin (density 1/64)")
PY
(cd "$EXP3" && ./deep/wgrad_i32 --dir "$D" --pad 1 --batch 1 \
   --in-x-file "$D/x_b0.bin" --in-dy-file "$D/dy_sparse.bin" --update) \
  | grep -E "IN-DY-FILE|FWD b=0|WGRAD_TOTAL|UPDATE:|FWD2|RESULT"
echo "== DONE =="
