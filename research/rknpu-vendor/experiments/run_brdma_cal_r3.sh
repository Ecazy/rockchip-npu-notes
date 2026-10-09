#!/usr/bin/env bash
# BRDMA 标定 第 3 轮：mul 表槽位/值域决定性 A/B
# 假设 H1: mul 槽 = int16 s2.14 @ chunk+48（当前发射器）；H2: fp16（0x3C00=1.0）。
# 斜坡表可同时暴露 stride 错位与格式。
set -u
cd "$(dirname "$0")"
OUT=brdma_r3; mkdir -p $OUT
S="--iw 8 --ih 2 --ic 4 --oc 16 --kw 1 --kh 1 --pad 0"
BASE="--pattern $S --pad-con1 0 --auto-fix"
# ALGO2 + SRC1 + MUL bypass: 纯 bias 通路（复核）
# BS_CFG: 0x20140 = ALGO2|SRC1|MUL_BYPASS|RELU_BYPASS

echo "### M0 sanity: bias ramp + MUL bypass BIT-EXACT (表 bias 槽复核)"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-bias -24 --brdma-bias-step 3 --brdma-mul 16384 \
  > $OUT/m0.txt 2>&1

echo "### M1: ALGO2 add-0 + BS_MUL from table, const mul=16384 shift14 (×1.0) → identity"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-bias 0 --brdma-mul 16384 --brdma-mul-step 0 \
  > $OUT/m1.txt 2>&1

echo "### M2: mul ramp 16384+1024*ch (s2.14: ×1.0,1.0625,...) 逐通道斜率"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-mul 16384 --brdma-mul-step 1024 \
  > $OUT/m2.txt 2>&1

echo "### M3: mul fp16 语义判别: 表值 15360 (=fp16 1.0 = s2.14 0.9375)"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-bias 0 --brdma-mul 15360 --brdma-mul-step 0 \
  > $OUT/m3.txt 2>&1

echo "### M4: mul=-16384 (s2.14 -1.0) 符号"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-bias 0 --brdma-mul -16384 --brdma-mul-step 0 \
  > $OUT/m4.txt 2>&1

echo "### M5: vendor 配置复现: BS_CFG=0x20140 MUL_CFG=0x00000e01 表 mul=16384 bias ramp 全链"
./min_conv_exp $BASE --bs-cfg 0x20140 --bs-mul 0x00000e01 --dpu-data-format 0xe0 \
  --brdma-cfg 0xe --brdma-bias 7 --brdma-bias-step 2 --brdma-mul 16384 --brdma-mul-step 0 \
  > $OUT/m5.txt 2>&1

echo "### M6: BRDMA_DATA_USE=0x2 仅 use1（mesa 风格）同 M2"
./min_conv_exp $BASE --bs-cfg 0x20140 --dpu-data-format 0xe0 \
  --brdma-cfg 0x2 --brdma-bias 0 --brdma-bias-step 1024 --brdma-mul 16384 --brdma-mul-step 0 \
  > $OUT/m6.txt 2>&1

grep -H "total=\|RESULT" $OUT/*.txt | sed "s|$OUT/||"
echo done
