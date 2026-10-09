#!/usr/bin/env bash
# §7.27 deconv 边界口径修正探针。
#
# 背景：旧 CPU 参考用 gather `tx = ox + pad - kx`（--deconv-flip 再翻转权重 tap），
# 仅当 2·pad = kw-1（如 3x3 pad1）时与硬件巧合重合，因此 3x3 全 BIT-EXACT 而
# 1x1/2x2/矩形核全 MISMATCH（int8 172/1024、fp16 186/1024）。
#
# 本探针用 --in-ruler + --w-ones（ic=16、fp16）反解硬件映射，证明硬件为
#   ox = ix*S + pad - kx   ⇒   tx = ox - pad + kx，权重 tap = kx（无翻转），
# 然后覆盖非 3x3 核 / 矩形核 / 多 S / 单 tap 全量验证。
#
# 纪律：submit ret≠0 视为环境失败，重试。
set -u
cd "$(dirname "$0")"
BASE="--pattern --fp16 --auto-fix --pad-con1 0 --f16-pix-elems 8"
GEOM="--iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --ic 16 --oc 16 --pad 1"
pass=0; fail=0; env=0

run() {
  local desc="$1"; shift
  local out sub r try
  for try in 1 2 3; do
    out=$(./min_conv_exp "$@" 2>&1)
    sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
    [ "$sub" = "SUBMIT ret=0" ] && break
    sleep 1
  done
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9()<>= -]*" | head -1)
  if [ "$sub" != "SUBMIT ret=0" ]; then env=$((env+1)); printf "%-34s %-12s (env)\n" "$desc" "$sub"; return; fi
  case "$r" in *BIT-EXACT*) pass=$((pass+1));; *) fail=$((fail+1));; esac
  printf "%-34s %s\n" "$desc" "$r"
}

echo "=== A) 硬件映射反解（ruler + w-ones）：hw 值 = 128·pix + 1080 ==="
echo "    (输出 hw=1080*k+... ; 用 P[oc x y] hw 字段人工核对 ix=(ox-1)/2, iy=(oy-1)/2)"
./min_conv_exp $BASE --in-ruler --w-ones --deconv-stride 1 $GEOM --kw 1 --kh 1 2>&1 \
  | grep -E "P\[oc=0 x=1 y=1\]|P\[oc=0 x=3 y=1\]|P\[oc=0 x=7 y=7\]|RESULT"

echo
echo "=== B) 核形状扫描（修正前除 3x3 外全挂）S=2 pad1 ic3 oc16 ==="
for k in "1 1" "2 2" "3 3" "3 1" "1 3" "5 5" "5 1"; do set -- $k
  run "kw=${1} kh=${2} S=2" $BASE --deconv-flip --deconv-stride 1 --ic 3 --oc 16 \
      --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --pad 1 --kw $1 --kh $2
done

echo
echo "=== C) 多 stride：S=2/4/8 ==="
for s in 1 3 7; do
  run "3x3 S=$((s+1))" $BASE --deconv-flip --deconv-stride $s --ic 3 --oc 16 \
      --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --pad 1 --kw 3 --kh 3
  run "1x1 S=$((s+1))" $BASE --deconv-flip --deconv-stride $s --ic 3 --oc 16 \
      --iw 4 --ih 4 --mode deconv --pad 1 --kw 1 --kh 1
done

echo
echo "=== D) 单 tap 判定权重序（kw=3 S=2，9 个 (kx,ky) 应全过 = 无翻转）==="
for kx in 0 1 2; do for ky in 0 1 2; do
  run "w-tap ${kx}:${ky}" $BASE --deconv-flip --deconv-stride 1 --ic 3 --oc 16 \
      --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --pad 1 --kw 3 --kh 3 --w-tap $kx:$ky
done; done

echo
echo "=== E) int8 对照（同一口径）==="
for k in "1 1" "2 2" "3 1" "5 5"; do set -- $k
  run "int8 kw=${1} kh=${2} S=2" --pattern --auto-fix --pad-con1 0 --deconv-flip \
      --deconv-stride 1 --ic 3 --oc 16 --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv \
      --pad 1 --kw $1 --kh $2
done

echo "TOTAL: pass=$pass fail=$fail env=$env"
