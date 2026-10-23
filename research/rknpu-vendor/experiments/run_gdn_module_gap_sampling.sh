#!/bin/sh
cd /mnt/ssd/workspace/gdn-probe || exit 1
export LD_LIBRARY_PATH=/mnt/ssd/workspace/runtimes/mars-overhead/rknn/lib:$LD_LIBRARY_PATH
export EPERF_RKNPU_BLOCKWATCH=1
export EPERF_RKNPU_BLOCKWATCH_SELECT=5
export EPERF_RKNPU_BLOCKWATCH_MIN_EFF=1
export EPERF_RKNPU_BLOCKWATCH_CAP=20000
export EPERF_RKNPU_BLOCKWATCH_NSLEEP=50000
rm -f fw_*.csv fw_*.err fw_progress.txt
for k in $(seq 1 18); do
  EPERF_RKNPU_BLOCKWATCH_NTH=$k EPERF_RKNPU_BLOCKWATCH_FILE=fw_$k.csv \
    LD_PRELOAD=/mnt/ssd/workspace/gdn-probe/hook/libeperf_hook_rknpu.so \
    timeout 90 ./gdn_feed c3.rknn fw$k c3_in0.bin > /dev/null 2> fw_$k.err
  rc=$?
  printf "%s rc=%s lines=%s fails=%s %s\n" "$k" "$rc" "$(wc -l < fw_$k.csv 2>/dev/null)" \
    "$(grep -c 'failed to submit' fw_$k.err)" \
    "$(grep -o 'hw_elapse_ns=[0-9]*' fw_$k.err | head -1)" >> fw_progress.txt
done
echo done >> fw_progress.txt
