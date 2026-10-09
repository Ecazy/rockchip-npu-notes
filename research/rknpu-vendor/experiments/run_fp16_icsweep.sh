#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
B="--pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --oc 8"
pass=0; fail=0
for ic in 1 2 3 4 7 8 9 10 15 16 17 20 23 24 25 31 32 33 40 47 48 49 56 64 65 80 96; do
  out=$(./min_conv_exp $B --ic $ic 2>&1)
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9() -]*" | head -1)
  m=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
  sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
  case "$r" in *BIT-EXACT*) pass=$((pass+1));; *) fail=$((fail+1));; esac
  printf "ic=%-3s %-22s %-22s %s\n" "$ic" "$r" "$sub" "$m"
done
echo "TOTAL pass=$pass fail=$fail"
