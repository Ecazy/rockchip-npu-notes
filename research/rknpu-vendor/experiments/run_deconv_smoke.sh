#!/usr/bin/env bash
# deconv 冒烟：验证 tap 顺序修正未破坏转置卷积
set -u
cd "$(dirname "$0")"
pass=0; fail=0
run() {
    desc="$1"; shift
    out=$(./min_conv_exp "$@" 2>&1)
    r=$(echo "$out" | grep -o "RESULT: [A-Z-]*" | head -1)
    if [ "$r" = "RESULT: BIT-EXACT" ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
    printf "%-40s %s\n" "$desc" "$r"
}
for pat in "--pattern" "--pattern --pat2 --asym-w"; do
  for s in 1 3 7; do
    run "deconv 4x4->8x8 S=$((s+1)) ${pat##*--}" $pat --pad-con1 0 --auto-fix \
        --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --deconv-stride $s --deconv-flip \
        --ic 3 --oc 16 --pad 1 --w-oc2 32
  done
done
run "deconv 4x4->8x8 S=2 pat2 noflip" --pattern --pat2 --asym-w --pad-con1 0 --auto-fix \
    --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --deconv-stride 1 --ic 3 --oc 16 --pad 1 --w-oc2 32
# §7.27：非 3x3 核（修正前 1x1/2x2/矩形核全 MISMATCH；3x3 因 2·pad=kw-1 恰好重合而通过）
for k in "1 1" "2 2" "3 1" "1 3" "3 3"; do set -- $k
  run "deconv ${1}x${2} S=2" --pattern --pat2 --asym-w --pad-con1 0 --auto-fix \
      --iw 4 --ih 4 --mode deconv --deconv-stride 1 --deconv-flip \
      --ic 3 --oc 16 --iw 4 --ih 4 --ow 8 --oh 8 --pad 1 --kw $1 --kh $2 --w-oc2 32
done
echo "TOTAL: pass=$pass fail=$fail"
