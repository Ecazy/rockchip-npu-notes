#!/usr/bin/env bash
# run_ic_split_probe.sh — §7.33 ic 维切分（跨 submit，host fp32 累加）
# 在板端 exp2 目录运行；用法：./run_ic_split_probe.sh [IC_SPLIT_LOG=...]
#
# 问题（handoff §8 下一入口①）：真 CBUF 容量越界时，按 **ic** 切分（而非输出行）能否替代？
# 跨 submit 累加是否 BIT-EXACT？
#
# 答案（实测，RKNPU driver v0.9.8）：
#   · 机制：`--ic-groups G` 把输入通道按 32 对齐切成 G 组，每组独立 emit+submit 到独立
#     partial 输出区（组输入 = 全局输入的连续平面，无需拷贝；权重描述符按组打包），
#     最后在 **host 用 fp32** 逐元素相加并 fp32→fp16 写回。
#   · **不是逐位一致**：每组 partial 先被 fp16 量化，再相加 ⇒ 与单次 submit 相比可差
#     **1 ULP**（`--in-ruler` ic96 3×3p1 3 组：43/4096 元素、max|diff|=16，而单次 arm
#     对 fp64 参考 BIT-EXACT ⇒ 切分偏离 ≤1 ULP）。多数用例下仍逐位一致。
#   · **容量场景可替代行切分**：`ic256 iw16 ih40`（输入 10 + 权重 4 > 12 bank）单发结构性
#     算错，`--ic-groups 4` 后 ULP-EXACT（max|diff|=4.8e-07）。
#   · host 的 fp32 累加本身与 fp64 累加对拍 max ulp=0（不引入额外误差）。
set -u
cd "$(dirname "$0")"
log=${IC_SPLIT_LOG:-fp16_ic_split_$(date +%Y%m%d_%H%M%S).log}

gcc -O2 -Wall -o min_conv_exp min_conv_exp.c -lm || { echo "BUILD FAILED"; exit 1; }

F="--pattern --fp16 --auto-fix --pad-con1 0"

row() {  # desc args...
    local desc="$1"; shift
    local out res stat hash ic try
    for try in 1 2 3; do
        out=$(./min_conv_exp "$@" 2>&1)
        res=$(echo "$out" | grep -oE "RESULT: .*" | head -1)
        [ -n "$res" ] && break
        echo "  (retry $try: no RESULT)" >&2
        sleep 1
    done
    stat=$(echo "$out" | grep -oE "total=[0-9]+  fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
    hash=$(echo "$out" | grep -oE "OUT-HASH: [0-9a-f]+" | sed 's/OUT-HASH: //')
    ic=$(echo "$out" | grep -oE "IC-SPLIT: .*" | head -1)
    printf "%-32s hash=%s  %-34s %s\n" "$desc" "$hash" "$res" "$stat"
    [ -n "$ic" ] && printf "    %s\n" "$ic"
}

{
    echo "# §7.33 ic 维切分（跨 submit，host fp32 累加）  $(date -u)"
    uname -a
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || true

    echo
    echo "## A) 默认 pattern：切分 vs 不切分（hash 相同 = 逐位一致）"
    row "ic32  1x1 不切分"    $F --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 32  --oc 16
    row "ic32  1x1 groups2"   $F --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 32  --oc 16 --ic-groups 2
    row "ic32  1x1 groups4"   $F --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 32  --oc 16 --ic-groups 4
    row "ic128 3x3p1 不切分"  $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16
    row "ic128 3x3p1 groups4" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 128 --oc 16 --ic-groups 4
    row "ic128 ih32 groups4"  $F --kw 3 --kh 3 --iw 16 --ih 32 --pad 1 --ic 128 --oc 16 --ic-groups 4
    sleep 0.5

    echo
    echo "## B) 压力 pattern --in-ruler（值域大、partial 量化粗）→ 暴露 ≤1 ULP 偏差"
    row "ruler ic128 1x1 不切分"   $F --in-ruler --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 128 --oc 16
    row "ruler ic128 1x1 groups4"  $F --in-ruler --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 128 --oc 16 --ic-groups 4
    row "ruler ic128 1x1 groups8"  $F --in-ruler --kw 1 --kh 1 --iw 16 --ih 16 --pad 0 --ic 128 --oc 16 --ic-groups 8
    row "ruler ic96 3x3p1 不切分"  $F --in-ruler --f16-pix-elems 8 --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 96 --oc 16
    row "ruler ic96 3x3p1 groups2" $F --in-ruler --f16-pix-elems 8 --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 96 --oc 16 --ic-groups 2
    row "ruler ic96 3x3p1 groups3" $F --in-ruler --f16-pix-elems 8 --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 96 --oc 16 --ic-groups 3
    sleep 0.5

    echo
    echo "## C) 真容量越界（输入+权重 > 12 bank）：ic 切分替代行切分"
    row "ic256 ih40 不切分（越界）"  $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 1 --ic 256 --oc 16
    row "ic256 ih40 groups4（救）"   $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 1 --ic 256 --oc 16 --ic-groups 4
    row "ic256 ih40 groups8（救）"   $F --kw 3 --kh 3 --iw 16 --ih 40 --pad 1 --ic 256 --oc 16 --ic-groups 8
    sleep 0.5

    echo
    echo "## D) 整数 pattern（--in-ch-id --w-ones）：应逐位一致"
    row "ch-id ic128 不切分"  $F --in-ch-id --w-ones --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 128 --oc 16
    row "ch-id ic128 groups4" $F --in-ch-id --w-ones --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 128 --oc 16 --ic-groups 4

    echo
    echo "## E) 不回归：无切分的既有形状"
    row "ic16 3x3p1 不切分" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16 --oc 16
    row "ic64 3x3p1 不切分" $F --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 64 --oc 16
} 2>&1 | tee "$log"

echo "SAVED: $log"
