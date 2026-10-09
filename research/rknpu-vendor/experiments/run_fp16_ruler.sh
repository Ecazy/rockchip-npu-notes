#!/usr/bin/env bash
# 标尺输入 + 单点权重：输出值 = 硬件读到的元素下标
set -u
cd "$(dirname "$0")"
P00() { grep "P\[oc=0 x=0 y=0\]" | head -1 | sed 's/.*hw=//;s/ ref=.*//'; }
B="--fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --oc 8 --ic 16 --in-ruler"
for layout in "8 8" "8 16" "8 32" "16 16" "16 32" "32 32"; do set -- $layout
  echo "=== pix=$1 watom=$2 ==="
  echo -n "  hw(j) j=0..15:"
  for j in $(seq 0 15); do
    v=$(./min_conv_exp $B --f16-pix-elems $1 --f16-watom $2 --w-ic-tap $j 2>&1 | P00)
    echo -n " $v"
  done; echo
  echo -n "  ref(j) j=0..15:"
  for j in $(seq 0 15); do
    v=$(./min_conv_exp $B --f16-pix-elems $1 --f16-watom $2 --w-ic-tap $j 2>&1 | grep "P\[oc=0 x=0 y=0\]" | head -1 | sed 's/.*ref=//')
    echo -n " $v"
  done; echo
done
