#!/usr/bin/env bash
# run_marker_arms.sh — 标记实验三臂对照（板端运行）
#
# 三臂共用**完全相同**的命令流长度，只差被替换的那个 8 字节槽位：
#   original  不动            （基线）
#   nop       槽位 = OP_NONE   （长度对照）
#   marker    槽位 = PPU 写     （标记本身）
#
# 用法（先用 plan_marker_slots.py 选出槽位）：
#   ./run_marker_arms.sh --task 1 --slot 2 --value 0x00c0ffee \
#       --expect-orig 0x1001000001e5400c -- ./rknn_runner model.rknn
#
# 脚本会：
#   1. 三臂各跑一次，保存完整日志；
#   2. 抽取 hook 的 MARKER / MARKER_RESTORE 行，核对 applied 与 orig 是否与计划一致；
#   3. 抽取 hw_elapse_time（有的话）打印对照表。
# 输出的逐位一致性比较需要外部工具（例如 rknn_infer_dump），本脚本不臆造结论。
set -u

HERE=$(cd "$(dirname "$0")" && pwd)

TASK=""
SLOT=""
REG="${REG:-0x6010}"
VALUE="${VALUE:-0x00c0ffee}"
NTH="${NTH:-1}"
EXPECT_ORIG=""
PLAN_FILE=""
PLAN_SLOT_REG=""
REG_EXPERIMENTAL="${REG_EXPERIMENTAL:-0}"
OUT=""
HOOK="${HOOK:-/mnt/ssd/workspace/rknpu-re/exp2/libeperf_hook_rknpu_sel.so}"
DRY_RUN=0
RUNS="${RUNS:-1}"
DUMPS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --slot) SLOT="$2"; shift 2 ;;
    --reg) REG="$2"; shift 2 ;;
    --value) VALUE="$2"; shift 2 ;;
    --nth) NTH="$2"; shift 2 ;;
    --expect-orig) EXPECT_ORIG="$2"; shift 2 ;;
    --plan) PLAN_FILE="$2"; shift 2 ;;
    --reg-experimental) REG_EXPERIMENTAL=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --hook) HOOK="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --dumps) DUMPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --plan：直接从工作负载打印的 MARKER_PLAN 行取值，避免手抄 slot/原词出错
if [ -n "$PLAN_FILE" ]; then
  [ -f "$PLAN_FILE" ] || { echo "plan file not found: $PLAN_FILE" >&2; exit 2; }
  line=$(grep -o 'MARKER_PLAN .*' "$PLAN_FILE" | tail -1)
  [ -n "$line" ] || { echo "no MARKER_PLAN line in $PLAN_FILE" >&2; exit 2; }
  # NOTE: the plan's `reg=` is the register of the ORIGINAL word being replaced,
  # not the marker register.  Keep REG from --reg/env (default PPU 0x6010);
  # taking the plan's value here would aim the marker at a live CNA/DPU register.
  for kv in $line; do
    case "$kv" in
      task=*) TASK="${kv#task=}" ;;
      slot=*) SLOT="${kv#slot=}" ;;
      reg=*)  PLAN_SLOT_REG="${kv#reg=}" ;;
      orig=*) EXPECT_ORIG="${kv#orig=}" ;;
    esac
  done
  echo "plan: task=$TASK slot=$SLOT slot_reg=${PLAN_SLOT_REG:-?} marker_reg=$REG expect_orig=$EXPECT_ORIG"
fi

if [ -z "$TASK" ] || [ -z "$SLOT" ]; then
  echo "--task and --slot are required (see plan_marker_slots.py)" >&2
  exit 2
fi
case "$RUNS" in ''|*[!0-9]*) echo "RUNS must be a positive integer" >&2; exit 2 ;; esac
[ "$RUNS" -gt 0 ] || { echo "RUNS must be > 0" >&2; exit 2; }

OUT=${OUT:-results/marker_arms_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd)

if [ $DRY_RUN -eq 0 ]; then
  [ $# -gt 0 ] || { echo "need the workload command after --" >&2; exit 2; }
  [ -f "$HOOK" ] || { echo "hook not found: $HOOK (set HOOK=...)" >&2; exit 2; }
fi

summary="$OUT_ABS/summary.txt"
: > "$summary"
printf "%-9s %-4s %-8s %-18s %-14s %s\n" arm run applied orig sync_apply restore | tee -a "$summary"

fail=0
for arm in original nop marker; do
  for run in $(seq 1 "$RUNS"); do
    log="$OUT_ABS/${arm}_r${run}.log"
    if [ $DRY_RUN -eq 1 ]; then
      echo "[dry-run] EPERF_RKNPU_MARKER_ARM=$arm TASK=$TASK SLOT=$SLOT REG=$REG VALUE=$VALUE" \
           "EPERF_RKNPU_MARKER_NTH=$NTH LD_PRELOAD=$HOOK $*"
      continue
    fi
    # EPERF_RKNPU_MARKER_EXPECT_ORIG 让 hook 在**写入之前**核对槽位内容，
    # 计划索引错了就直接拒绝改写，而不是事后才发现改错了地方。
    env -u EPERF_RKNPU_MARKER_EXPECT_ORIG \
      EPERF_RKNPU_MARKER_ARM="$arm" \
      EPERF_RKNPU_MARKER_TASK="$TASK" \
      EPERF_RKNPU_MARKER_SLOT="$SLOT" \
      EPERF_RKNPU_MARKER_REG="$REG" \
      EPERF_RKNPU_MARKER_VALUE="$VALUE" \
      EPERF_RKNPU_MARKER_NTH="$NTH" \
      ${EXPECT_ORIG:+EPERF_RKNPU_MARKER_EXPECT_ORIG=$EXPECT_ORIG} \
      ${REG_EXPERIMENTAL:+EPERF_RKNPU_MARKER_REG_EXPERIMENTAL=$REG_EXPERIMENTAL} \
      EPERF_MARKER_ARM_NAME="$arm" \
      LD_PRELOAD="$HOOK" "$@" > /dev/null 2> "$log"
    rc=$?

    applied=$(grep -o 'applied=[01]' "$log" | tail -1 | cut -d= -f2)
    orig=$(grep -o 'orig=0x[0-9a-f]*' "$log" | tail -1 | cut -d= -f2)
    marker=$(grep -o 'marker=0x[0-9a-f]*' "$log" | tail -1 | cut -d= -f2)
    reason=$(grep -o 'reason=[a-z_]*' "$log" | tail -1 | cut -d= -f2)
    hw=$(grep -o 'hw_elapse_time_ns=[0-9-]*\|hw_elapse=[0-9-]*' "$log" | tail -1 | cut -d= -f2)
    restored=$(grep -c 'MARKER_RESTORE' "$log")

    sync_apply=$(grep -o 'sync_apply=[-0-9]*' "$log" | tail -1 | cut -d= -f2)
    restore_ok=$(grep -o 'MARKER_RESTORE ok=[01]' "$log" | tail -1 | cut -d= -f2)
    sync_rc=$(grep -o 'MARKER_RESTORE ok=[01] sync_rc=[-0-9]*' "$log" | tail -1 | sed 's/.*sync_rc=//')
    printf "%-9s %-4s %-8s %-18s sync_apply=%-4s restore=%s\n" "$arm" "$run" \
      "${applied:-?}" "${orig:-?}" "${sync_apply:-n/a}" "${restore_ok:-n/a}" | tee -a "$summary"

    if [ $rc -ne 0 ]; then
      echo "  run failed (rc=$rc), log: $log" >&2
      fail=1
      continue
    fi
    if [ "$arm" = "original" ]; then
      # the baseline must not rewrite anything
      if [ "$applied" = "1" ]; then
        echo "  ERROR: original arm modified the stream" >&2
        fail=1
      fi
      continue
    fi
    if [ "${applied:-0}" != "1" ]; then
      reason=$(grep -o 'reason=[a-z_]*' "$log" | tail -1 | cut -d= -f2)
      echo "  ERROR: $arm arm did not apply (reason=${reason:-unknown}), log: $log" >&2
      fail=1
      continue
    fi
    if [ -n "$EXPECT_ORIG" ] && [ "${orig:-}" != "$EXPECT_ORIG" ]; then
      echo "  ERROR: rewrote slot with orig=$orig, plan expected $EXPECT_ORIG" >&2
      echo "         (the flat task index or slot index does not match the plan)" >&2
      fail=1
    fi
    if [ "${sync_apply:-1}" != "0" ]; then
      echo "  ERROR: BO sync to device failed (sync_apply=${sync_apply:-none});" \
           "applied=1 does not prove the device saw the marker ($log)" >&2
      fail=1
    fi
    if [ "${restore_ok:-0}" != "1" ]; then
      echo "  ERROR: stream restore did not confirm (restore=${restore_ok:-none}" \
           "sync_rc=${sync_rc:-none}); log: $log" >&2
      fail=1
    fi
  done
done

if [ $DRY_RUN -eq 1 ] && [ -n "$DUMPS" ]; then
  echo
  echo "[dry-run] would compare: baseline=$DUMPS/original.* arms=$DUMPS/{nop,marker}.*"
fi
echo
echo "logs: $OUT_ABS"
echo "summary: $summary"
if [ $DRY_RUN -eq 1 ]; then
  exit 0
fi
echo
if [ -n "$DUMPS" ]; then
  # 逐位比对：三臂长度与算术都相同，只有 bit-identical 才算标记无副作用
  base=$(ls "$DUMPS"/original.* 2>/dev/null | head -1)
  if [ -z "$base" ]; then
    echo "no baseline dump found in $DUMPS (expected original.*)"; exit $((fail | 1))
  fi
  arms=()
  for arm in nop marker; do
    f=$(ls "$DUMPS/$arm".* 2>/dev/null | head -1)
    [ -n "$f" ] && arms+=(--arm "$arm=$f")
  done
  if [ ${#arms[@]} -eq 0 ]; then
    echo "no arm dumps found in $DUMPS (expected nop.* / marker.*)"; exit $((fail | 1))
  fi
  echo "== output comparison =="
  python3 "$HERE/compare_arm_outputs.py" --baseline "$base" "${arms[@]}" \
    --out-dir "$OUT_ABS/compare"
  cmp_rc=$?
  [ $cmp_rc -ne 0 ] && fail=1
else
  echo "no --dumps directory given: output bit-equality was NOT checked."
  echo "a marker arm that changes outputs is not a marker; rerun with --dumps <dir>."
fi
echo
echo "still not concluded by this script: timing perturbation of the marker arm."
exit $fail
