#!/usr/bin/env bash
# run_fp16_cbuf_bank_probe.sh — §7.31：CNA 输入 tile 的 CBUF bank 包线 + FEATURE_GRAINS
# 口径 + 输出行切分兜底（修正 §7.30 的「32 KB 硬包线」结论）
# 在板端 exp2 目录运行；用法：./run_fp16_cbuf_bank_probe.sh
#   [CBUF_BANK_LOG=... 覆盖日志名]
#
# 结论（实测，RKNPU driver v0.9.8，Orange Pi 5 Plus / RK3588）：
#   ① 单次 submit 的 CNA 输入 feature tile 上限 = **DATA_BANK × 32768 B**
#      （CNA_CBUF_CON0：WEIGHT_BANK = 12 − DATA_BANK）。§7.30 观测到的「阈值恰为
#      32768 B」是旧发射器硬编码 DATA_BANK=1 的结果，**不是硬件硬包线**。
#      A 段：同一几何 65536/131072 B 在 DATA_BANK=2/4 下 ULP-EXACT。
#   ② FEATURE_GRAINS（CNA_CONV_CON2[13:4]，10 bit）必须随输入**行字节**放大：
#      旧值 `50 + stride + 1`（mesa magic）在「高输入」下静默算错（fp16 iw16 ic16：
#      53 行净、54 行错）——即 §7.30 的「次级约束」。采用 tinygrad rockchip 后端口径
#      `grains = max(80, ceil(2·32768 / row_bytes) + 1)`（取偶、≤1022）后全部转正。
#   ③ 真·包线只有 CBUF 容量：`ceil(in_bytes/32768) + weight_banks_need > 12`（12 个
#      32 KB bank 须同时容纳输入 tile 与权重描述符）；此时必须切分。**按输出行切分是
#      互不相交的输出面，无需跨 submit 累加**（D 段：--tile-auto 通过）。
#   ④ 发射器默认 `--auto-cfg` 已按 ① ② 自动取 bank 与 grains；`--no-auto-cfg` 复现
#      §7.30 的旧行为（A 段对照臂）。
set -u
cd "$(dirname "$0")"
log=${CBUF_BANK_LOG:-fp16_cbuf_bank_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"
I="--pattern --auto-fix --pad-con1 0"

# tinygrad rockchip 口径：grains = max(80, ceil(2*32768/row_bytes)+1) 取偶，≤1022
grains_auto() {
    python3 -c 'import sys; rb=int(sys.argv[1]); g=-(-2*32768//rb)+1; g=(g+1)//2*2; print(max(80,min(g,1022)))' "$1"
}

verdict() {  # 打印：描述 | RESULT | 失配统计 | CBUF-CFG（banks/grains）
    local desc="$1"; shift
    local out res stat cfg try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT, 疑似 submit 超时)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "total=[0-9]+  (fp16-)?mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    cfg=$(echo "$out" | grep -oE "CBUF-CFG: .*" | head -1)
    printf "%-40s %-46s %s\n" "$desc" "$res" "$stat"
    [ -n "$cfg" ] && printf "    %s\n" "$cfg"
}

{
    echo "# §7.31 CBUF bank envelope + FEATURE_GRAINS + row tiling  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) A/B：同一几何「旧默认（--no-auto-cfg, DATA_BANK=1/grains=52）」vs「auto-cfg」"
    verdict "fp16 ic128 iw16 ih16 OLD (65536B)"  $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --no-auto-cfg
    verdict "fp16 ic128 iw16 ih16 AUTO"          $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16
    verdict "fp16 ic16  iw16 ih64 OLD (32768B)"  $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 16  --oc 16 --no-auto-cfg
    verdict "fp16 ic16  iw16 ih64 AUTO"          $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 16  --oc 16
    verdict "int8 ic160 iw16 ih16 OLD (40960B)"  $I --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 160 --oc 16 --no-auto-cfg
    verdict "int8 ic160 iw16 ih16 AUTO"          $I --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 160 --oc 16

    echo
    echo "## B) 输入字节 vs DATA_BANK（--no-auto-cfg 下用 --weight-bank 显式设 DB=12-WB）"
    echo "   # 期望：DB-1 坏、DB=ceil(in/32768) 好 ⇒ 阈值 = DB×32768 B"
    for spec in "9 36864 11 10" "16 65536 11 10" "24 98304 10 9" "32 131072 9 8"; do
        set -- $spec; ih=$1; bytes=$2; wb_bad=$3; wb_ok=$4
        verdict "ic128 ih$ih (${bytes}B) DB=1  [bad]" $F --kw 3 --kh 3 --iw 16 --ih $ih --pad 1 --ic 128 --oc 16 --no-auto-cfg --weight-bank $wb_bad
        verdict "ic128 ih$ih (${bytes}B) DB=$((12-wb_ok))  [ok]" $F --kw 3 --kh 3 --iw 16 --ih $ih --pad 1 --ic 128 --oc 16 --no-auto-cfg --weight-bank $wb_ok
    done
    verdict "ic128 ih8 (32768B) DB=1 [边界净]" $F --kw 3 --kh 3 --iw 16 --ih 8 --pad 1 --ic 128 --oc 16 --no-auto-cfg

    echo
    echo "## C) FEATURE_GRAINS：固定 bank 充足，只切 grains（旧 52 vs tinygrad auto）"
    for spec in "16 16 54 1" "16 16 64 1" "16 16 128 2" "16 16 200 4" "8 16 100 1" "8 16 200 2"; do
        set -- $spec; iw=$1; ic=$2; ih=$3; db=$4; wb=$((12-db)); rb=$((iw*2*ic)); g=$(grains_auto $rb)
        verdict "iw$iw ic$ic ih$ih grains=52  [old]"  $F --kw 3 --kh 3 --iw $iw --ih $ih --pad 1 --ic $ic --oc 16 --no-auto-cfg --weight-bank $wb --grains 52
        verdict "iw$iw ic$ic ih$ih grains=$g [auto]" $F --kw 3 --kh 3 --iw $iw --ih $ih --pad 1 --ic $ic --oc 16 --no-auto-cfg --weight-bank $wb --grains $g
    done

    echo
    echo "## D) 真·容量越界（in + weight bank > 12）→ 按输出行切分兜底"
    echo "   # ic256 3x3 oc16：weight 73728 B → 4 bank；ih40 输入 327680 B → 10 bank（合计 14 > 12）"
    verdict "ic256 ih40 AUTO（越界，无切分）"  $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 1 --ic 256 --oc 16
    verdict "ic256 ih40 --tile-auto（切分）"   $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 1 --ic 256 --oc 16 --tile-auto
    echo "   # ic128 3x3 oc16：weight 36864 B → 3 bank；ih64 输入 262144 B → 8 bank（11 ≤ 12，可单发）"
    verdict "ic128 ih64 AUTO（bank 够，无需切分）" $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 128 --oc 16
    verdict "ic128 ih64 --tile-oh 32（强制切分）"  $F --kw 3 --kh 3 --iw 16 --ih 64 --pad 1 --ic 128 --oc 16 --tile-oh 32

    echo
    echo "## E) 切分不改结论（同一用例 tiled vs 非 tiled 均应为准确结果）"
    verdict "fp16 ic128 iw16 ih16 非切分"      $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16
    verdict "fp16 ic128 iw16 ih16 --tile-oh 6" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --tile-oh 6
    verdict "int8 ic128 iw16 ih16 --tile-auto" $I --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --tile-auto
} 2>&1 | tee "$log"

echo "SAVED: $log"
