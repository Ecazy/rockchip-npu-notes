#!/usr/bin/env bash
# 冲激探针：分离 fp16 权重寻址（Exp W）与输入寻址（Exp I）
set -u
cd "$(dirname "$0")"
P00() { grep "P\[oc=0 x=0 y=0\]" | head -1 | sed 's/.*hw=//;s/ ref=.*//'; }
B="--fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --oc 8"
for pn in "8 8" "8 16" "16 16" "16 32"; do
  set -- $pn
  echo "=== pix=$1 watom=$2 ==="
  echo -n "  ExpW hw(j) j=0..N:"
  for j in $(seq 0 15); do
    v=$(./min_conv_exp $B --f16-pix-elems $1 --f16-watom $2 --ic 16 --in-ones --w-ic-tap $j 2>&1 | P00)
    echo -n " $v"
  done; echo
  echo -n "  ExpI hw(c0) c0=0..15:"
  for c0 in $(seq 0 15); do
    v=$(./min_conv_exp $B --f16-pix-elems $1 --f16-watom $2 --ic 16 --w-ones --in-one-hot $c0 2>&1 | P00)
    echo -n " $v"
  done; echo
done
