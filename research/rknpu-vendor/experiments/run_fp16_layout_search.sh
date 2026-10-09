#!/usr/bin/env bash
# fp16 ic>8 布局搜索：输入像素原子 × 权重 ic 原子
set -u
cd "$(dirname "$0")"
B="--pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 8 --ih 8 --pad 0"
for pix in 8 16 32; do for wa in 8 16 32; do
  out=$(./min_conv_exp $B --f16-pix-elems $pix --f16-watom $wa --ic 16 --oc 8 2>&1)
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9() -]*" | head -1)
  sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
  m=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+" | head -1)
  printf "pix=%-3s watom=%-3s %-22s %-10s %s\n" "$pix" "$wa" "$r" "$sub" "$m"
done; done
echo "--- same, ic=32 ---"
for pix in 8 16 32; do for wa in 8 16 32; do
  out=$(./min_conv_exp $B --f16-pix-elems $pix --f16-watom $wa --ic 32 --oc 8 2>&1)
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9() -]*" | head -1)
  sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
  m=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+" | head -1)
  printf "pix=%-3s watom=%-3s %-22s %-10s %s\n" "$pix" "$wa" "$r" "$sub" "$m"
done; done
