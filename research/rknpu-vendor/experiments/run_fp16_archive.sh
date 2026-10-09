#!/usr/bin/env bash
# §7.17 证据归档（板端串行执行，必须独占 NPU）
set -u
cd /mnt/ssd/workspace/rknpu-re/exp2
OUT=results/fp16_20260929
mkdir -p "$OUT"
hdr() {
  echo "# §7.17 FP16 证据归档"
  echo "# 日期: $(date -Iseconds)"
  echo "# 板: $(hostname)  $(uname -r)  $(cat /sys/kernel/debug/rknpu/version 2>/dev/null)"
  echo "# min_conv_exp.c sha256: $(sha256sum min_conv_exp.c | cut -d' ' -f1)"
  echo
}
{ hdr; echo "########## 1) fp16 回归（--fp16，健康探针协议）"; echo; } > "$OUT/01_fp16_regression.txt"
./run_fp16_regression.sh >> "$OUT/01_fp16_regression.txt" 2>&1
{ hdr; echo "########## 2) fp16 ic 扫描（1x1 oc8）"; echo; } > "$OUT/02_ic_sweep.txt"
./run_fp16_icsweep.sh >> "$OUT/02_ic_sweep.txt" 2>&1
{ hdr; echo "########## 3) 累加器精度 + ic 对齐（--in-cancel / --ic-pad）"; echo; } > "$OUT/03_accumulator.txt"
./run_fp16_acc.sh >> "$OUT/03_accumulator.txt" 2>&1
{ hdr; echo "########## 4) 标尺探针（硬件实际访问的输入元素下标）"; echo; } > "$OUT/04_ruler.txt"
./run_fp16_ruler.sh >> "$OUT/04_ruler.txt" 2>&1
{ hdr; echo "########## 5) 布局搜索（pix-elems x watom）"; echo; } > "$OUT/05_layout_search.txt"
./run_fp16_layout_search.sh >> "$OUT/05_layout_search.txt" 2>&1
{ hdr; echo "########## 6) 冲激探针（ones / one-hot 分离权重与输入寻址）"; echo; } > "$OUT/06_impulse.txt"
./run_fp16_impulse.sh >> "$OUT/06_impulse.txt" 2>&1
{ hdr; echo "########## 7) int8 硬化回归 74 例（确认 fp16 改动未回归 int8）"; echo; } > "$OUT/07_int8_regression.txt"
./run_regression_hard.sh >> "$OUT/07_int8_regression.txt" 2>&1
tar czf "$OUT.tgz" "$OUT"
ls -la "$OUT.tgz"
