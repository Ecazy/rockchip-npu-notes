#!/bin/sh
# 池化 → 卷积（含训练闭环）——可自证的定量管线（§7.156/§7.161）
# 用法: sh pipeline_pool_conv.sh <工作目录>
set -e
D=${1:-/tmp/pipe}
POOL=/mnt/ssd/workspace/rknpu-re/pool
EXP3=/mnt/ssd/workspace/rknpu-re/exp3
mkdir -p "$D"
echo "== 1) 池化（NPU，32x32x32 --k2s2--> 16x16x32）=="
(cd "$POOL" && ./min_pool --iw 32 --ih 32 --ic 32 --kw 2 --kh 2 --sx 2 --sy 2 --pad 0 \
   --out-file "$D/pool") | grep -E "OUT-FILE|mismatch="
echo "== 2) 生成数据集（会写出自己的 x_b0.bin）=="
(cd "$EXP3" && ./deep/i126 --gen-only --dir "$D" --pad 1 --seed 20260928 >/dev/null 2>&1) || true
echo "== 3) 用池化输出（转置为行主序）覆盖 x_b0.bin =="
python3 - "$D" <<'PY'
import sys
D=sys.argv[1]; IW=IH=16; IC=32
d=open(D+"/pool.out","rb").read()
out=bytearray(IW*IH*IC)
for x in range(IW):
    for y in range(IH):
        for c in range(IC):
            out[(y*IW+x)*IC+c]=d[(c//16)*IW*IH*16 + x*IH*16 + y*16 + (c%16)]
open(D+"/x_b0.bin","wb").write(bytes(out))
print("  transposed", len(out), "B ->", D+"/x_b0.bin")
PY
echo "== 4) 卷积 + wgrad + 更新（外部 X，自带参考）=="
(cd "$EXP3" && ./deep/i126 --dir "$D" --pad 1 --batch 1 --in-x-file "$D/x_b0.bin") \
  | grep -E "IN-X-FILE|FWD b=0|RESULT"
