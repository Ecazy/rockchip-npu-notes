#!/usr/bin/env bash
# fp16（PROC_PRECISION=2）回归。
# 纪律：① 出现 submit ret≠0 后先跑健康探针（已知 BIT-EXACT 的 3x3），探针不干净
#       就 sleep 5 重试，避免把超时后的脏状态记成语义 FAIL；② 已知 1x1 且 ic≤8 且
#       oc≥24 必超时，单独放到最后一节（已知限制，不混入主表）。
set -u
cd "$(dirname "$0")"
BASE="--pattern --fp16 --auto-fix --pad-con1 0"
HEALTH="$BASE --kw 3 --kh 3 --iw 8 --ih 8 --pad 0 --ic 8 --oc 16"
pass=0; fail=0; env=0; dirty=0

health() {  # 返回 0 = 干净
  local out
  out=$(./min_conv_exp $HEALTH 2>&1)
  echo "$out" | grep -q "RESULT: BIT-EXACT\|RESULT: ULP-EXACT"
}
run() {
  desc="$1"; shift
  local try out sub r m
  for try in 1 2 3; do
    out=$(./min_conv_exp "$@" 2>&1)
    sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
    [ "$sub" = "SUBMIT ret=0" ] && break
    sleep 1
    if ! health; then dirty=$((dirty+1)); sleep 5; health || { sleep 5; health; }; fi
  done
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9()<>= -]*" | head -1)
  m=$(echo "$out" | grep -oE "fp16-mismatch=[0-9]+  max\|diff\|=[0-9.e+-]+  max ulp=[0-9]+" | head -1)
  if [ "$sub" != "SUBMIT ret=0" ]; then env=$((env+1)); printf "%-30s %-12s (env fail)\n" "$desc" "$sub"; return; fi
  case "$r" in *BIT-EXACT*|*ULP-EXACT*) pass=$((pass+1));;
    *)  # MISMATCH：先确认设备是否干净，脏结果不得记成语义 FAIL
        if ! health; then
          dirty=$((dirty+1)); sleep 5
          if ! health; then env=$((env+1)); printf "%-30s %-24s (DISCARDED: device dirty)\n" "$desc" "$r"; return; fi
        fi
        fail=$((fail+1));;
  esac
  printf "%-30s %-24s %s\n" "$desc" "$r" "$m"
}
echo "=== A) 1x1 ic sweep (oc<=16) ==="
health || echo "WARN: device dirty at section A start"
for ic in 1 2 3 4 7 8 9 15 16 17 24 32 33 48 64 96; do
  run "1x1 ic${ic} oc16" $BASE --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic $ic --oc 16
done
echo "=== B) 1x1 oc sweep (ic=16) ==="
for oc in 4 8 16 24 32 40 48 64; do
  run "1x1 ic16 oc${oc}" $BASE --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 16 --oc $oc
done
echo "=== B2) 3x3p1 ic16 oc sweep（含不完整 oc 组）==="
for oc in 4 8 16 17 24 32 33 40 48 64 65; do
  run "3x3p1 ic16 oc${oc}" $BASE --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 16 --oc $oc
done
echo "=== C) 3x3 pad0/pad1 多形状 ==="
for s in "8 8" "16 16" "16 8" "8 16" "32 32" "12 20"; do set -- $s; for p in 0 1; do
  run "3x3 ${1}x${2} pad$p" $BASE --kw 3 --kh 3 --iw $1 --ih $2 --pad $p --ic 8 --oc 16
done; done
echo "=== D) 5x5 pad2 / 1x1 大 ic ==="
run "5x5 pad2 16x16" $BASE --kw 5 --kh 5 --iw 16 --ih 16 --pad 2 --ic 8 --oc 16
run "1x1 ic96 oc16" $BASE --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic 96 --oc 16
echo "=== E) 矩形核 ==="
for k in "3 2" "2 3" "1 5" "5 1" "4 6" "6 4"; do set -- $k
  run "kw${1} kh${2} pad0" $BASE --kw $1 --kh $2 --iw 16 --ih 16 --pad 0 --ic 8 --oc 8
done
echo "=== F) stride ==="
for st in 2 3; do
  run "3x3 s${st} pad1" $BASE --kw 3 --kh 3 --stride $st --iw 16 --ih 16 --pad 1 --ic 8 --oc 8
done
echo "=== G) 3x3p1 oc sweep 其他 ic ==="
for ic in 8 32; do for oc in 17 24 40; do
  run "3x3p1 ic${ic} oc${oc}" $BASE --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic $ic --oc $oc
done; done
echo "=== H) 1x1 ic<=8 且 oc>=24（§7.18 修复后应全过）==="
for ic in 1 3 8; do for oc in 24 32 40; do
  run "1x1 ic${ic} oc${oc}" $BASE --kw 1 --kh 1 --iw 8 --ih 8 --pad 0 --ic $ic --oc $oc
done; done
echo "=== I) [信息性] fp16 BS/BRDMA bias（配方：先一次 BS-bypass 提交，再 BS 激活跑（内部自动预热 3））==="
# §7.21：BS/BRDMA 表只在"上一次 submit 的 BS 阶段是激活的"时才重载；且必须由一个
# BS-bypass 提交把状态打断。故每个 bias 用例前先跑一次同几何的无 bias 提交。
BIASNP="$BASE --in-ch-id --w-ones --kw 1 --kh 1 --iw 4 --ih 4 --pad 0"
info() {
  local desc="$1"; shift; local out sub r hw
  out=$(./min_conv_exp "$@" 2>&1)
  sub=$(echo "$out" | grep -oE "SUBMIT ret=[0-9-]+" | head -1)
  r=$(echo "$out" | grep -o "RESULT: [A-Za-z0-9()<>= -]*" | head -1)
  hw=$(echo "$out" | grep -o "P\[oc=0 x=0 y=0\] hw=[0-9.e+-]* ref=[0-9.e+-]*" | head -1)
  printf "%-28s %-12s %-24s %s\n" "$desc" "$sub" "$r" "$hw"
}
arm() { ./min_conv_exp "$@" >/dev/null 2>&1; }
arm $BIASNP --ic 8 --oc 8
info "bias 5.0f ic8 oc8"      $BIASNP --ic 8 --oc 8 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000
arm $BIASNP --ic 8 --oc 8
info "bias -3.0f ic8 oc8"     $BIASNP --ic 8 --oc 8 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0xc0400000
arm $BIASNP --ic 8 --oc 8
info "bias ramp ic8 oc8"      $BIASNP --ic 8 --oc 8 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000 --brdma-bias-step 0x200000
arm $BIASNP --ic 8 --oc 24
info "bias 5.0f ic8 oc24"     $BIASNP --ic 8 --oc 24 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000
arm $BASE --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 8 --oc 16
info "bias 5.0f 3x3 ic8 oc16" $BASE --kw 3 --kh 3 --iw 16 --ih 16 --pad 1 --ic 8 --oc 16 --bs-cfg 0x20150 --brdma-cfg 0xe --brdma-bias 0x40a00000
echo "=== J) fp16 deconv（§7.22；§7.27 口径修正后 1x1/2x2 亦闭合）==="
for s in 1 3 7; do
  run "deconv 4x4->8x8 S=$((s+1))" $BASE --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --deconv-stride $s --deconv-flip --ic 3 --oc 16 --pad 1
done
echo "--- §7.27 共享边界修正：非 3x3 核（原 1x1/2x2/矩形均 MISMATCH）---"
for k in "1 1" "2 2" "3 1" "1 3"; do set -- $k
  run "deconv ${1}x${2} S=2" $BASE --iw 4 --ih 4 --ow 8 --oh 8 --mode deconv --deconv-stride 1 --deconv-flip --ic 3 --oc 16 --pad 1 --kw $1 --kh $2
done
echo "TOTAL: pass=$pass fail=$fail envfail=$env dirtyevents=$dirty"
