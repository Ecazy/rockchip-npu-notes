#!/usr/bin/env bash
# run_phase_sweep.sh — route A/B 相位扫描采集器（板端运行）
#
# 设计见 research/scheduling/REGCMD-TIMING-OBSERVABILITY.md §3（模块活动分布）
# 与 §4 第 4 步（标记可见时间 CDF）。每次运行**只取一个快照**：
#   EPERF_RKNPU_BLOCKWATCH_CAP=1  +  每次不同的 PHASE_NS  +  硬读取预算 MAX_READS。
#
# 相位从 hook 的 arm 时刻起算，那是**提交入口**（用户态 ioctl 之前），不是硬件
# kick；报告里一律叫 submit-relative。相位顺序随机化，并穿插无采样基线。
#
# 用法（板端）：
#   ./run_phase_sweep.sh --phases 0,10000,20000,30000 --reps 30 -- \
#       ./rknn_runner /path/sdpa_qwen224.rknn
#   ./run_phase_sweep.sh --raw-off 0x1000 --phases 5000,15000 --reps 20 -- <cmd>
#   # 只读中断字（INTERRUPT_STATUS / RAW_STATUS），无需实验开关：
#   ./run_phase_sweep.sh --raw-off 0x0028 --raw-off2 0x002c --phases ... -- <cmd>
#   # 非 vetted 候选（如 0x1024）必须显式开实验模式，默认拒绝：
#   ./run_phase_sweep.sh --raw-off 0x1024 --raw-experimental --phases ... -- <cmd>
#   离线自检（不需要板子）：./run_phase_sweep.sh --dry-run --phases 0,10 --out /tmp/x
#
# 可选环境变量：
#   HOOK=/path/libeperf_hook_rknpu_sel.so   预加载的 hook（默认取 exp2 目录）
#   OUT=<dir>                               输出目录（默认 results/phase_sweep_<ts>）
#   OP_IDX=<n>                              只对某个 op 采（默认 6）
#   BASELINE_EVERY=<n>                      每 n 次运行插一次无采样基线（默认 10，0=关闭）
#   POWER_CHECK_NS=<ns>                     采样期电源复查周期（默认 200000，**不要设 0**）
set -u

PHASES="0"
REPS=10
RAW_OFF=""
RAW_EXPERIMENTAL=0
RAW_OFF2=""
RAW_CORE=0
SIGNAL=""
OUT=""
OP_IDX="${OP_IDX:-6}"
BASELINE_EVERY="${BASELINE_EVERY:-10}"
POWER_CHECK_NS="${POWER_CHECK_NS:-200000}"
HOOK="${HOOK:-/mnt/ssd/workspace/rknpu-re/exp2/libeperf_hook_rknpu_sel.so}"
SEED=""
DRY_RUN=0
PREOPEN_POWER=0

while [ $# -gt 0 ]; do
  case "$1" in
    --phases) PHASES="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --raw-off) RAW_OFF="$2"; shift 2 ;;
    --raw-experimental) RAW_EXPERIMENTAL=1; shift ;;
    --raw-off2) RAW_OFF2="$2"; shift 2 ;;
    --raw-core) RAW_CORE="$2"; shift 2 ;;
    --signal) SIGNAL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --op-idx) OP_IDX="$2"; shift 2 ;;
    --baseline-every) BASELINE_EVERY="$2"; shift 2 ;;
    --power-check-ns) POWER_CHECK_NS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --hook) HOOK="$2"; shift 2 ;;
    --preopen-power) PREOPEN_POWER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# 信号列的默认值必须和实际读的寄存器一致，否则分析器会去读一个不存在的列
if [ -z "$SIGNAL" ]; then
  if [ -n "$RAW_OFF" ]; then SIGNAL="c0_raw"; else SIGNAL="c0_cna"; fi
fi

# 严格正整数：'00'、'-1'、'abc' 都不能拿来关掉电源守卫
is_positive_int() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$1" -gt 0 ]
}
if ! is_positive_int "$POWER_CHECK_NS"; then
  echo "POWER_CHECK_NS must be a positive integer (got '$POWER_CHECK_NS');" \
       "the sampler must re-check the power domain" >&2
  exit 2
fi
if ! is_positive_int "$REPS"; then
  echo "REPS must be a positive integer (got '$REPS')" >&2
  exit 2
fi

if [ $DRY_RUN -eq 0 ]; then
  if [ $# -eq 0 ]; then
    echo "need the workload command after --" >&2
    exit 2
  fi
  if [ ! -f "$HOOK" ]; then
    echo "hook not found: $HOOK (set HOOK=...)" >&2
    exit 2
  fi
fi

if [ "$RAW_EXPERIMENTAL" -eq 1 ] && [ -z "$RAW_OFF" ]; then
  echo "--raw-experimental needs --raw-off" >&2
  exit 2
fi
# 快速失败：vetted 只有四个模块 S_STATUS（权威规则在 rknpu_ioctl_hook.c 的
# bw_raw_offset_forbidden/bw_raw_offset_vetted，这里只是提前给出可读的报错）。
if [ -n "$RAW_OFF" ]; then
  case "$RAW_OFF" in
    0x1000|0x3000|0x4000|0x5000|0X1000|0X3000|0X4000|0X5000) ;;
    0x0028|0x002c|0X0028|0X002C) ;;   # read-only interrupt words: no experimental flag
    *)
      if [ "$RAW_EXPERIMENTAL" -ne 1 ]; then
        echo "RAW_OFF=$RAW_OFF is not a vetted register (vetted: 0x1000 0x3000 0x4000 0x5000);" \
             "pass --raw-experimental to probe it anyway" >&2
        exit 2
      fi
      ;;
  esac
fi
if [ "$RAW_EXPERIMENTAL" -eq 1 ]; then
  echo "note: RAW_OFF=$RAW_OFF is an EXPERIMENTAL probe (not a vetted register);" \
       "results are mechanism-only until the register is verified on the board" >&2
fi

if [ -n "$RAW_OFF2" ]; then
  if [ -z "$RAW_OFF" ]; then
    echo "--raw-off2 needs --raw-off (the second read only exists in raw mode)" >&2
    exit 2
  fi
  case "$RAW_OFF2" in
    0x1000|0x3000|0x4000|0x5000|0x0028|0x002c|0X1000|0X3000|0X4000|0X5000|0X0028|0X002C) ;;
    *)
      if [ "$RAW_EXPERIMENTAL" -ne 1 ]; then
        echo "RAW_OFF2=$RAW_OFF2 is neither vetted nor a read-only interrupt word;" \
             "pass --raw-experimental to probe it anyway" >&2
        exit 2
      fi
      ;;
  esac
fi

OUT=${OUT:-results/phase_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/runs" "$OUT/meta" "$OUT/baseline" "$OUT/out"
# 绝对路径：分析器相对 manifest 目录解析，写成相对路径会重复拼接
OUT_ABS=$(cd "$OUT" && pwd)
MANIFEST="$OUT_ABS/manifest.csv"

echo "run_id,request_phase_ns,csv,condition" > "$MANIFEST"

IFS=',' read -r -a PHASE_LIST <<< "$PHASES"
ORDER=$(python3 - "$SEED" "$REPS" "${PHASE_LIST[@]}" <<'PY'
import random, sys
seed = sys.argv[1]
seed = int(seed) if seed else None
reps = int(sys.argv[2])
phases = [int(x) for x in sys.argv[3:]]
rnd = random.Random(seed)
plan = [p for p in phases for _ in range(reps)]
rnd.shuffle(plan)
print(" ".join(str(p) for p in plan))
PY
) || { echo "python3 needed for phase randomisation" >&2; exit 2; }

n=0
for phase in $ORDER; do
  n=$((n + 1))
  run_id=$(printf "p%08d_r%03d" "$phase" "$n")
  csv="$OUT_ABS/runs/$run_id.csv"
  meta="$OUT_ABS/meta/$run_id.txt"

  # 只在真正给出 RAW_OFF 时才导出它：空字符串会让 hook 走到偏移 0（PC_VERSION）
  ENV_OPEN=()
  [ "$PREOPEN_POWER" -eq 1 ] && ENV_OPEN+=("EPERF_RKNPU_BLOCKWATCH_PREOPEN_POWER=1")
  ENV_RAW=()
  [ -n "$RAW_OFF" ] && ENV_RAW+=("EPERF_RKNPU_BLOCKWATCH_RAW_OFF=$RAW_OFF")
  [ "$RAW_EXPERIMENTAL" -eq 1 ] && \
    ENV_RAW+=("EPERF_RKNPU_BLOCKWATCH_RAW_OFF_EXPERIMENTAL=1")
  [ -n "$RAW_OFF2" ] && ENV_RAW+=("EPERF_RKNPU_BLOCKWATCH_RAW_OFF2=$RAW_OFF2")

  if [ $DRY_RUN -eq 1 ]; then
    echo "$run_id,$phase,$csv," >> "$MANIFEST"
    continue
  fi

  # -u 清掉从父环境继承的采样配置：没显式要求的开关一律不生效，
  # 否则父 shell 里的 RAW_OFF/EDGE/MINIMAL/TASK_CHECK 会悄悄改变采样语义。
  env \
    -u EPERF_RKNPU_BLOCKWATCH_FILE -u EPERF_RKNPU_BLOCKWATCH_CAP \
    -u EPERF_RKNPU_BLOCKWATCH_PHASE_NS -u EPERF_RKNPU_BLOCKWATCH_MAX_READS \
    -u EPERF_RKNPU_BLOCKWATCH_SPIN_NS -u EPERF_RKNPU_BLOCKWATCH_NSLEEP \
    -u EPERF_RKNPU_BLOCKWATCH_POLL_NS -u EPERF_RKNPU_BLOCKWATCH_POWER_FILE \
    -u EPERF_RKNPU_BLOCKWATCH_SELECT -u EPERF_RKNPU_BLOCKWATCH_MIN_EFF \
    -u EPERF_RKNPU_BLOCKWATCH_NTH -u EPERF_RKNPU_BLOCKWATCH_ALL_CORES \
    -u EPERF_RKNPU_BLOCKWATCH_MINIMAL -u EPERF_RKNPU_BLOCKWATCH_EDGE \
    -u EPERF_RKNPU_BLOCKWATCH_RAW_OFF -u EPERF_RKNPU_BLOCKWATCH_RAW_OFF_EXPERIMENTAL \
    -u EPERF_RKNPU_BLOCKWATCH_RAW_OFF2 \
    -u EPERF_RKNPU_BLOCKWATCH_RAW_CORE -u EPERF_RKNPU_BLOCKWATCH_TASK_CHECK \
    -u EPERF_RKNPU_BLOCKWATCH_REQUIRE_ALL_TASKS_SAME_OP \
    -u EPERF_RKNPU_IOCTL_TRACE -u EPERF_RKNPU_REGCMD_MAX_BYTES \
    -u EPERF_RKNPU_PROBE -u EPERF_RKNPU_DEDUP \
    EPERF_RKNPU_BLOCKWATCH=1 \
    EPERF_RKNPU_BLOCKWATCH_FILE="$csv" \
    EPERF_RKNPU_BLOCKWATCH_CAP=1 \
    EPERF_RKNPU_BLOCKWATCH_PHASE_NS="$phase" \
    EPERF_RKNPU_BLOCKWATCH_MAX_READS=8 \
    EPERF_RKNPU_BLOCKWATCH_SPIN_NS=0 \
    EPERF_RKNPU_BLOCKWATCH_NSLEEP=0 \
    EPERF_RKNPU_BLOCKWATCH_POWER_CHECK_NS="$POWER_CHECK_NS" \
    EPERF_RKNPU_BLOCKWATCH_POWER_FILE="${POWER_FILE:-}" \
    EPERF_RKNPU_BLOCKWATCH_SELECT=5 \
    EPERF_RKNPU_BLOCKWATCH_MIN_EFF=1 \
    EPERF_RKNPU_BLOCKWATCH_OP_IDX="$OP_IDX" \
    EPERF_RKNPU_BLOCKWATCH_RAW_CORE="$RAW_CORE" \
    "${ENV_OPEN[@]}" \
    "${ENV_RAW[@]}" \
    LD_PRELOAD="$HOOK" \
    "$@" > "$OUT_ABS/out/$run_id.txt" 2> "$meta"
  rc=$?

  if [ $rc -ne 0 ]; then
    echo "run $run_id failed (rc=$rc), keeping $meta" >&2
    continue
  fi
  if grep -q "BLOCKWATCH_RAW_REJECTED" "$meta"; then
    echo "run $run_id rejected RAW_OFF, fix the argument before analysing" >&2
    continue
  fi
  if [ ! -s "$csv" ]; then
    echo "run $run_id produced no snapshot (see $meta)" >&2
    continue
  fi
  if ! grep -q "budget_hit=0" "$meta"; then
    echo "run $run_id hit the read budget; snapshot may be incomplete" >&2
  fi
  if grep -q "power_lost=1" "$meta"; then
    echo "run $run_id lost the power domain during sampling" >&2
    continue
  fi
  echo "$run_id,$phase,$csv," >> "$MANIFEST"

  if [ "$BASELINE_EVERY" -gt 0 ] && [ $((n % BASELINE_EVERY)) -eq 0 ]; then
    # 无采样基线：同样的负载，不预加载 hook。用来量化采样本身的扰动。
    ( time "$@" > /dev/null 2>&1 ) 2> "$OUT_ABS/baseline/${run_id}_baseline.txt"
    echo "  baseline $run_id"
  fi
done

echo
echo "manifest: $MANIFEST  ($(($(wc -l < "$MANIFEST") - 1)) runs)"
echo
if [ -n "$RAW_OFF" ]; then
  echo "next (offline), register probe${RAW_EXPERIMENTAL:+ (experimental)}:"
  echo "  python3 analyze_phase_sweep.py --manifest $MANIFEST --raw --out-dir $OUT_ABS/analysis"
  echo "  # marker CDF: add --marker-value 0x<token> once the token is validated on the board"
else
  echo "next (offline), module status:"
  echo "  python3 analyze_phase_sweep.py --manifest $MANIFEST --signal $SIGNAL --out-dir $OUT_ABS/analysis"
fi
