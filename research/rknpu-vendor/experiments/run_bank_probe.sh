#!/usr/bin/env bash
# run_bank_probe.sh — 寄存器读回 bank 判据（路线 B 的第一道门槛，板端运行）
#
# 问题：驱动提交时写 CNA/CORE `S_POINTER = 0xE + core位`、命令流写 DPU/RDMA
# `S_POINTER = 0x0E`，两者都置了 POINTER_PP_EN ⇒ 每个 block 都在 ping-pong。
# 那么主机经 /dev/mem 读一个普通配置寄存器时，看到的是**当前正在写的那一组**，
# 还是**另一组**？
#
# 做法：只读实验，不往设备写任何东西。
#   1. 用已捕获的 regcmd 流当"期望值表"（每个 task 写了什么值）；
#   2. 用相位扫描在 task 内取样，读回该寄存器；
#   3. 读回值等于本 task 写的 ⇒ lag 0；等于上一个写作 task 的 ⇒ lag 1。
# lag 就是 bank 判据，由 analyze_bank_probe.py 给出结论。
#
# 用法（板端）：
#   ./run_bank_probe.sh --offset 0x1024 --rows results/sdpa_timeline_20261020/sdpa_regcmd_order_rows.csv \
#       -- ./rknn_runner /path/sdpa_qwen224.rknn
#   离线自检（不需要板子）：./run_bank_probe.sh --dry-run --out /tmp/bp
#
# 注意：0x1024 属于 experimental 档（vetted 只有 0x1000/0x3000/0x4000/0x5000），
# 脚本会自动带上 --raw-experimental，并在结果里标明这是实验探针。
set -u

MODULE="CNA"
OFFSET="0x1024"
ROWS="results/sdpa_timeline_20261020/sdpa_regcmd_order_rows.csv"
PHASES="1000,2000,3000,4000,6000,8000"
REPS="${REPS:-20}"
OP_IDX="${OP_IDX:-6}"
OUT=""
HOOK="${HOOK:-/mnt/ssd/workspace/rknpu-re/exp2/libeperf_hook_rknpu_sel.so}"
SEED=""
DRY_RUN=0
EXPERIMENTAL=1

HERE=$(cd "$(dirname "$0")" && pwd)

while [ $# -gt 0 ]; do
  case "$1" in
    --module) MODULE="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --phases) PHASES="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --op-idx) OP_IDX="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --hook) HOOK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --vetted) EXPERIMENTAL=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

OUT=${OUT:-results/bank_probe_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd)
SWEEP_OUT="$OUT_ABS/sweep"

SWEEP_ARGS=(--phases "$PHASES" --reps "$REPS" --raw-off "$OFFSET"
            --op-idx "$OP_IDX" --out "$SWEEP_OUT")
[ "$EXPERIMENTAL" -eq 1 ] && SWEEP_ARGS+=(--raw-experimental)
[ -n "$SEED" ] && SWEEP_ARGS+=(--seed "$SEED")
[ "$DRY_RUN" -eq 1 ] && SWEEP_ARGS+=(--dry-run)
[ -n "$HOOK" ] && SWEEP_ARGS+=(--hook "$HOOK")

if [ "$DRY_RUN" -eq 0 ] && [ $# -eq 0 ]; then
  echo "need the workload command after --" >&2
  exit 2
fi

echo "== sweep: $OFFSET on $MODULE (experimental=$EXPERIMENTAL) =="
if [ $# -gt 0 ]; then
  "$HERE/run_phase_sweep.sh" "${SWEEP_ARGS[@]}" -- "$@"
else
  "$HERE/run_phase_sweep.sh" "${SWEEP_ARGS[@]}"
fi
sweep_rc=$?
if [ $sweep_rc -ne 0 ]; then
  echo "sweep failed (rc=$sweep_rc)" >&2
  exit $sweep_rc
fi

MANIFEST="$SWEEP_OUT/manifest.csv"
if [ ! -s "$MANIFEST" ]; then
  echo "no manifest produced at $MANIFEST" >&2
  exit 2
fi

# dry-run 只验证参数拼装，没有真实样本
if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "dry run ok; the analysis step needs real samples:"
  echo "  python3 $HERE/analyze_bank_probe.py --rows $ROWS --manifest <manifest> \\"
  echo "      --module $MODULE --offset $OFFSET --out-dir $OUT_ABS/analysis"
  exit 0
fi

echo
echo "== analysis =="
python3 "$HERE/analyze_bank_probe.py" --rows "$ROWS" --manifest "$MANIFEST" \
  --module "$MODULE" --offset "$OFFSET" --out-dir "$OUT_ABS/analysis"
