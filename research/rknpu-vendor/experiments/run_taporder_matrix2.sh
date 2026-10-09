#!/usr/bin/env bash
# §7.15 修复后扩展验证：全 kw!=kh 有序对 + one-hot tap + asym 方形 + 回归
set -u
cd "$(dirname "$0")"
BASE="--pattern --pat2 --auto-fix --pad 0 --pad-con1 0 --iw 16 --ih 16 --ic 3 --oc 16"
pass=0; fail=0
run() {
    desc="$1"; shift
    out=$(./min_conv_exp "$@" 2>&1)
    r=$(echo "$out" | grep -o "RESULT: [A-Z-]*" | head -1)
    n=$(echo "$out" | grep -oE "floor-mismatch=[0-9]+ \(max abs diff [0-9]+\)" | head -1)
    if [ "$r" = "RESULT: BIT-EXACT" ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
    printf "%-30s %-22s %s\n" "$desc" "$r" "$n"
}
echo "=== E) all ordered kw!=kh in 2..6 (pat2) ==="
for kw in 2 3 4 5 6; do for kh in 2 3 4 5 6; do
  [ "$kw" = "$kh" ] && continue
  run "kw${kw} kh${kh}" $BASE --kw $kw --kh $kh
done; done

echo "=== F) asym-w square 2..5 ==="
for k in 2 3 4 5; do run "kw${k} kh${k} asym" $BASE --kw $k --kh $k --asym-w; done

echo "=== G) one-hot single tap (pat2) ==="
for ky in 0 1 2; do for kx in 0 1 2; do run "kw3kh3 tap ${kx}:${ky}" $BASE --kw 3 --kh 3 --w-tap ${kx}:${ky}; done; done
for ky in 0 1; do for kx in 0 1 2; do run "kw3kh2 tap ${kx}:${ky}" $BASE --kw 3 --kh 2 --w-tap ${kx}:${ky}; done; done
for ky in 0 1 2; do for kx in 0 1; do run "kw2kh3 tap ${kx}:${ky}" $BASE --kw 2 --kh 3 --w-tap ${kx}:${ky}; done; done

echo "=== H) pad=1 sweep kw/kh (pat2) ==="
for kw in 1 2 3 4 5; do for kh in 1 2 3 4 5; do
  run "pad1 kw${kw} kh${kh}" --pattern --pat2 --auto-fix --pad 1 --pad-con1 0 --iw 16 --ih 16 --ic 3 --oc 16 --kw $kw --kh $kh
done; done

echo "=== I) stride=2 with kw!=kh ==="
for s in "3 2" "2 3" "4 3" "3 4"; do set -- $s
  run "stride2 kw${1} kh${2}" --pattern --pat2 --auto-fix --pad 1 --pad-con1 0 --iw 16 --ih 16 --ic 3 --oc 16 --stride 2 --kw $1 --kh $2
done

echo "TOTAL: pass=$pass fail=$fail"
