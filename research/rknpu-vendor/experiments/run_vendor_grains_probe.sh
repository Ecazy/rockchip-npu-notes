#!/usr/bin/env bash
# run_vendor_grains_probe.sh — §7.32：FEATURE_GRAINS 的 vendor 精确口径
#   `grains = ih + kh + pad_top`（离线 85/85 真实 trace 吻合，见 grains_tasks.csv）
# 在板端 exp2 目录运行；用法：./run_vendor_grains_probe.sh [VENDOR_GRAINS_LOG=...]
#
# 板端只切 grains（`--auto-cfg` 负责 bank；`--grains N` 显式覆盖），比较：
#   旧值 52（50+stride+1，mesa magic） / vendor 口径 / auto 口径
# 期望：凡是 vendor 口径 > 52 的用例，52 结构性算错而 vendor/auto 准确。
#
# 判据：fp16 `BIT-EXACT` / `≤16 ULP`（fp64 参考）；int8 `BIT-EXACT (floor)`。
set -u
cd "$(dirname "$0")"
log=${VENDOR_GRAINS_LOG:-fp16_vendor_grains_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"
I="--pattern --auto-fix --pad-con1 0"

verdict() {
    local desc="$1"; shift
    local out res stat cfg try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    cfg=$(echo "$out" | grep -oE "GRAINS=[0-9]+" | head -1)
    printf "%-40s %-44s %-10s %s\n" "$desc" "$res" "$cfg" "$stat"
}

# 对每个几何：先打 vendor 口径（ih+kh+pad_top），再跑旧值 52 与 auto
trio() {  # desc iw ih ic kw kh pad mode_base...
    local desc="$1"; iw="$2"; ih="$3"; ic="$4"; kw="$5"; kh="$6"; pad="$7"; shift 7
    local vg=$((ih + kh + pad))
    verdict "$desc grains=52 (旧)"    "$@" --kw "$kw" --kh "$kh" --iw "$iw" --ih "$ih" --pad "$pad" --ic "$ic" --oc 16 --grains 52
    verdict "$desc grains=$vg (vendor)" "$@" --kw "$kw" --kh "$kh" --iw "$iw" --ih "$ih" --pad "$pad" --ic "$ic" --oc 16 --grains $vg
    verdict "$desc grains=auto"        "$@" --kw "$kw" --kh "$kh" --iw "$iw" --ih "$ih" --pad "$pad" --ic "$ic" --oc 16
    sleep 0.3
}

{
    echo "# §7.32 vendor FEATURE_GRAINS 口径 = ih + kh + pad_top  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) fp16 3x3p1（vendor = ih+4）：旧 52 在 ih≥54 起结构性算错"
    trio "fp16 3x3 iw16 ih54" 16 54 16 3 3 1 $F
    trio "fp16 3x3 iw16 ih64" 16 64 16 3 3 1 $F
    trio "fp16 3x3 iw16 ih100" 16 100 16 3 3 1 $F
    trio "fp16 3x3 iw8  ih100" 8 100 16 3 3 1 $F

    echo
    echo "## B) 其它 kernel/pad（vendor 口径含 pad_top）"
    trio "fp16 1x1p0 iw16 ih64" 16 64 16 1 1 0 $F
    trio "fp16 5x5p2 iw16 ih64" 16 64 16 5 5 2 $F

    echo
    echo "## C) 大 tile（>2 bank，bank 由 auto-cfg 负责）"
    trio "fp16 3x3 iw16 ic64 ih100" 16 100 64 3 3 1 $F
    trio "fp16 3x3 iw16 ic128 ih24" 16 24 128 3 3 1 $F

    echo
    echo "## D) int8 对照"
    trio "int8 3x3 iw16 ic128 ih64" 16 64 128 3 3 1 $I
    trio "int8 3x3 iw16 ic160 ih16" 16 16 160 3 3 1 $I
} 2>&1 | tee "$log"

echo "SAVED: $log"
