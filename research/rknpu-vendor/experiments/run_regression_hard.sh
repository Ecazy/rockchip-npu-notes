#!/usr/bin/env bash
# 硬化回归：§7.11 的 74 例 + pat2/--asym-w（消除 tap 置换抵消与 y 盲区）
# 覆盖：3×3 17形状×pad0/1 + IC=1 10形状×pad0/1 + 1×1 4形状×pad0/1 + ic/oc/stride/kw/kh 变体
# 全部默认布局 flag（行主序 in/out + 宽度 DATA_ENTRIES 已为默认）
set -u
cd "$(dirname "$0")"

BASE="--pattern --pat2 --asym-w --pad-con1 0 --auto-fix"
pass=0; fail=0; env=0

# §7.35 硬化：submit ret≠0 是已知环境失败（随机 ~1/3，见 handoff §七②），
# 按 §7.12 重试，不计为语义 FAIL（与 run_fp16_regression.sh 同纪律）。
run() {
    desc="$1"; shift
    local try out sub
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
        [ "$sub" = "SUBMIT ret=0" ] && break
        echo "  (env retry $try: $sub)" >&2
        sleep 1
    done
    if [ "$sub" != "SUBMIT ret=0" ]; then
        env=$((env+1)); echo "ENV   $desc (${sub:-no-submit})"; return
    fi
    if echo "$out" | grep -q "RESULT: BIT-EXACT"; then
        pass=$((pass+1)); echo "PASS  $desc"
    else
        fail=$((fail+1)); echo "FAIL  $desc"
        echo "$out" | grep -E "RESULT|mismatch|MISMATCH" | head -5
    fi
}

echo "=== A) 3x3 pad0/pad1, 17 shapes ==="
for s in "8 8" "16 16" "32 32" "16 8" "8 16" "12 8" "8 12" "16 12" "12 16" \
         "20 8" "8 20" "11 7" "7 11" "9 5" "5 9" "16 24" "24 16"; do
    set -- $s
    for p in 0 1; do
        run "3x3 ${1}x${2} pad$p" $BASE --iw $1 --ih $2 --pad $p --pad-con1 0
    done
done

echo "=== B) IC=1 pad0/pad1, 10 shapes ==="
for s in "8 8" "16 8" "8 16" "12 8" "8 12" "16 16" "24 16" "16 24" "7 11" "11 7"; do
    set -- $s
    for p in 0 1; do
        run "IC=1 3x3 ${1}x${2} pad$p" $BASE --ic 1 --iw $1 --ih $2 --pad $p --pad-con1 0
    done
done

echo "=== C) 1x1 pad0/pad1, 4 shapes ==="
for s in "16 8" "8 16" "12 16" "8 12"; do
    set -- $s
    for p in 0 1; do
        run "1x1 ${1}x${2} pad$p" $BASE --kw 1 --kh 1 --iw $1 --ih $2 --pad $p --pad-con1 0
    done
done

echo "=== D) variants (default 8x8 unless noted) ==="
for ic in 8 16 32 40 65; do
    run "3x3 ic=$ic pad1" $BASE --ic $ic --oc 40 --pad 1 --pad-con1 0
done
run "oc=40 pad1"        $BASE --oc 40 --pad 1 --pad-con1 0
run "stride2 pad1"      $BASE --stride 2 --pad 1 --pad-con1 0
run "stride3 pad1"      $BASE --stride 3 --pad 1 --pad-con1 0
run "kw3xkh1 pad1"      $BASE --kw 3 --kh 1 --pad 1 --pad-con1 0
run "kw1xkh3 pad1"      $BASE --kw 1 --kh 3 --pad 1 --pad-con1 0
run "pad2"              $BASE --pad 2 --pad-con1 0
run "kw5xkh5 pad1"      $BASE --kw 5 --kh 5 --pad 1 --pad-con1 0

echo "=================================="
echo "TOTAL: pass=$pass fail=$fail envfail=$env"
