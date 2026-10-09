#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
P() { grep "P\[oc=0 x=0 y=0\]" | head -1; }
B="--fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 4 --ih 4 --pad 0 --oc 8 --w-ones --in-cancel"
echo "=== 累加器精度：期望 fp32 累加 = 1.0，fp16 逐步累加 = 0.0 ==="
for ic in 3 8 16 32; do
  echo -n "  ic=$ic: "; ./min_conv_exp $B --ic $ic 2>&1 | P
done
echo "=== ic 对齐规则（--ic-pad 覆盖 c.ic）==="
BB="--pattern --fp16 --auto-fix --pad-con1 0 --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --oc 8"
for ic in 17 33; do for pad in 0 24 32 40 48 64; do
  if [ "$pad" = 0 ]; then out=$(./min_conv_exp $BB --ic $ic 2>&1); else out=$(./min_conv_exp $BB --ic $ic --ic-pad $pad 2>&1); fi
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9() -]*" | head -1)
  m=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+" | head -1)
  printf "  ic=%-3s ic-pad=%-4s %-22s %s\n" "$ic" "$pad" "$r" "$m"
done; done
