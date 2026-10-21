#!/usr/bin/env python3
"""Read-only register-bank probe for route B (the gate before any marker work).

Why this exists
---------------
Route B needs one register whose value can be read back after a chosen command
position.  Before that can mean anything, we need to know which *register group*
a host read through /dev/mem actually returns:

  the driver writes CNA/CORE `S_POINTER` = 0xE + 0x10000000*core at submit time
  and the command stream writes DPU/DPU_RDMA `S_POINTER` = 0x0E, and 0x0E has
  POINTER_PP_EN set, i.e. every block in the SDPA path runs with ping-pong.

So the host may be looking at the group that is currently being written, at the
other one, or at a fixed group.  FINDINGS.md §3 step 1 asks for a probe; this
script implements it **without writing anything to the device**:

  1. the captured command stream tells us, for every task t, exactly which value
     was written to the register (and in which order);
  2. the phase sweep tells us, for a sample taken inside task t, which value the
     host read back;
  3. if the read equals the value written by task t          -> lag 0
     if it equals the value written by task t-1              -> lag 1
     if it alternates with task parity                       -> alternating
     if it matches nothing that was ever written             -> unreadable

That "lag" is the bank verdict, and it is obtained by reads only.  A non-zero
lag does not kill route B, it changes the interpretation: you compare against
the value expected at t-lag instead of t.

Usage
-----
    python3 analyze_bank_probe.py --rows results/sdpa_timeline_20261020/sdpa_regcmd_order_rows.csv \
        --manifest results/bank_probe_20261022/manifest.csv \
        --module CNA --offset 0x1024 --out-dir results/bank_probe_20261022/analysis
    python3 analyze_bank_probe.py --selftest

The manifest is the one produced by run_phase_sweep.sh (absolute csv paths).
Samples are read from the `c0_raw` column; `task_before` must be present.
"""

import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys

VALID_VALIDITY = {"ok", "unchecked", ""}


# ---------------------------------------------------------------------------
# expected-value timeline from the captured command stream
# ---------------------------------------------------------------------------

def load_write_timeline(path, module, offset):
    """task_index -> list of (position_in_task, value) for one register."""
    timeline = collections.defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        need = {"task_index", "task_local_regcmd_index", "module", "reg_offset", "value"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            sys.exit("rows CSV missing: %s" % ", ".join(sorted(missing)))
        for row in reader:
            if row["module"] != module:
                continue
            if int(row["reg_offset"], 16) != offset:
                continue
            timeline[int(row["task_index"])].append(
                (int(row["task_local_regcmd_index"]), int(row["value"], 16)))
    for task in timeline:
        timeline[task].sort()
    if not timeline:
        sys.exit("register %s@0x%04x never written in %s" % (module, offset, path))
    return timeline


def value_series(timeline):
    """Ordered tasks that write this register, with the value each leaves behind."""
    tasks = sorted(timeline)
    return [(task, timeline[task][-1][1]) for task in tasks]


def value_at(series, index):
    """Value left by series[index] (index counts writing tasks, not task ids)."""
    if 0 <= index < len(series):
        return series[index][1]
    return None


def distinctness(series):
    """Fraction of consecutive writing tasks whose value differs.

    A register that repeats the same value cannot be used for lag inference:
    every hypothesis matches equally well.
    """
    if len(series) < 2:
        return 0.0
    changes = sum(1 for (_, a), (_, b) in zip(series, series[1:]) if a != b)
    return changes / (len(series) - 1)


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------

def load_samples(manifest_path, signal):
    samples = []
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        base = os.path.dirname(os.path.abspath(manifest_path))
        for run in reader:
            path = run["csv"]
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            if not os.path.exists(path):
                continue
            with open(path, newline="") as fh2:
                for row in csv.DictReader(fh2):
                    validity = (row.get("sample_validity") or "unchecked").strip()
                    if validity not in VALID_VALIDITY:
                        continue
                    submit_enter = row.get("submit_enter_ns") or row.get("kick_ns")
                    if not submit_enter:
                        continue
                    if signal not in row:
                        sys.exit("column %s missing in %s" % (signal, path))
                    samples.append({
                        "run_id": run["run_id"],
                        "request_ns": int(run["request_phase_ns"]),
                        "phase_ns": int(row["t_ns"]) - int(submit_enter),
                        "task": int(row.get("task_before") or -1),
                        "value": int(row[signal], 16),
                    })
    if not samples:
        sys.exit("no usable samples")
    return samples


# ---------------------------------------------------------------------------
# lag inference
# ---------------------------------------------------------------------------

def match_lags(series, task, value, lag_max):
    """Which lags k make `value` equal to the value left by the task k writes back.

    Returns the list of candidate lags.  A lag is counted in *writing tasks*, so
    lag 0 means "the value this task writes", lag 1 "the one written by the
    previous writing task", and so on.
    """
    task_ids = [t for t, _ in series]
    if task not in task_ids:
        # The sample landed in a task that does not write this register; the
        # newest write belongs to an earlier task, so start from that one.
        earlier = [i for i, t in enumerate(task_ids) if t <= task]
        if not earlier:
            return []
        base = earlier[-1]
    else:
        base = task_ids.index(task)
    lags = []
    for k in range(lag_max + 1):
        if value_at(series, base - k) == value:
            lags.append(k)
    return lags


def classify(rows, lag_max, min_samples=20, min_match_rate=0.8, min_tasks=5, max_dominant=0.9):
    """Turn per-sample lag matches into an OBSERVATION, not a bank verdict.

    A lag of 0 or 1 is compatible with several hardware behaviours at once, so
    the result is reported as `observed_lag_k` and carries `hypothesis_only`.
    In particular a lag of 1 can mean "the host reads the other bank" OR simply
    "the sample landed before this task wrote the register"; only a sweep that
    spans the in-task write position can separate those.

    Evidence thresholds are enforced before any label is produced: a single
    matching sample among many misses must not be reported as a finding.
    """
    matched = [r for r in rows if r["lags"]]
    unmatched = [r for r in rows if not r["lags"]]
    unambiguous = [r for r in matched if len(r["lags"]) == 1]
    ambiguous = [r for r in matched if len(r["lags"]) > 1]
    histogram = collections.Counter(r["lags"][0] for r in unambiguous)
    total = sum(histogram.values())
    tasks = len({r["task"] for r in rows})
    match_rate = (len(matched) / len(rows)) if rows else 0.0

    base = {
        "hypothesis_only": True,
        "lag_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "samples": len(rows),
        "samples_matched": len(matched),
        "samples_ambiguous": len(ambiguous),
        "samples_unmatched": len(unmatched),
        "match_rate": round(match_rate, 3),
        "distinct_tasks": tasks,
        "thresholds": {"min_samples": min_samples, "min_match_rate": min_match_rate,
                       "min_tasks": min_tasks, "max_dominant": max_dominant},
    }

    if (not matched and len(rows) >= min_samples and tasks >= min_tasks):
        base["verdict"] = "no_value_matched"
        base["note"] = ("no sample matched any value this register ever had written, "
                        "over %d samples and %d tasks: the register is not readable, the "
                        "read returns a group that was never captured, or the replay is "
                        "not reproducing the stream" % (len(rows), tasks))
        return base

    missing = []
    if len(rows) < min_samples:
        missing.append("%d samples < %d" % (len(rows), min_samples))
    if tasks < min_tasks:
        missing.append("%d distinct tasks < %d" % (tasks, min_tasks))
    if match_rate < min_match_rate:
        missing.append("match rate %.2f < %.2f" % (match_rate, min_match_rate))
    if total and total < len(rows) * min_match_rate:
        missing.append("%d unambiguous matches < %.0f%% of samples"
                       % (total, min_match_rate * 100))
    if missing:
        base["verdict"] = "insufficient_evidence"
        base["note"] = ("not enough evidence for any label: " + "; ".join(missing) +
                        ". Increase repeats and task coverage before reading anything "
                        "into the lag.")
        return base

    dominance = (max(histogram.values()) / total) if total else 0.0
    if total == 0:
        base["verdict"] = "ambiguous"
        base["note"] = ("every matching sample fits several lags: this register repeats "
                        "values too often to separate the groups; pick another one")
    elif dominance < max_dominant:
        base["verdict"] = "mixed_lag"
        base["note"] = ("no single lag dominates (%s); inspect the per-phase breakdown "
                        "before concluding anything" % dict(sorted(histogram.items())))
    else:
        lag = max(histogram, key=histogram.get)
        base["verdict"] = "observed_lag_%d" % lag
        base["dominant_lag"] = lag
        base["dominance"] = round(dominance, 3)
        if lag == 0:
            base["note"] = ("the read-back matched what the current task writes in "
                            "%.0f%% of usable samples. This is consistent with a single "
                            "bank or with a pointer that tracks the write, but it does "
                            "not prove either: a read that returns a constant group "
                            "would look the same for a register whose value never "
                            "changes at that position." % (dominance * 100))
        elif lag == 1:
            base["note"] = ("the read-back matched the previous writing task in %.0f%% "
                            "of usable samples. That is consistent with ping-pong where "
                            "the host reads the other group, but ALSO with samples taken "
                            "before this task wrote the register. Separate the two by "
                            "sweeping phases that straddle the write position."
                            % (dominance * 100))
        else:
            base["note"] = ("read-back lagged %d writing tasks in %.0f%% of usable "
                            "samples; treat as an observation only" % (lag, dominance * 100))
    return base


def analyse(rows_path, manifest_path, module, offset, lag_max=3, signal="c0_raw",
            min_samples=20, min_match_rate=0.8, min_tasks=5):
    timeline = load_write_timeline(rows_path, module, offset)
    series = value_series(timeline)
    samples = load_samples(manifest_path, signal)

    per_sample = []
    for sample in samples:
        lags = match_lags(series, sample["task"], sample["value"], lag_max)
        per_sample.append({
            "run_id": sample["run_id"],
            "request_ns": sample["request_ns"],
            "phase_ns": sample["phase_ns"],
            "task": sample["task"],
            "value": sample["value"],
            "lags": lags,
        })

    verdict = classify(per_sample, lag_max, min_samples, min_match_rate, min_tasks)

    # per-phase breakdown: a lag that flips with the requested phase is a
    # position effect, not a bank effect
    by_phase = collections.defaultdict(list)
    for row in per_sample:
        if len(row["lags"]) == 1:
            by_phase[row["request_ns"]].append(row["lags"][0])
    phase_table = []
    for request_ns in sorted(by_phase):
        lags = by_phase[request_ns]
        phase_table.append({
            "request_ns": request_ns,
            "n": len(lags),
            "lag_histogram": {str(k): v for k, v in
                              sorted(collections.Counter(lags).items())},
        })

    return {
        "module": module,
        "offset": "0x%04x" % offset,
        "signal": signal,
        "register_tasks": len(series),
        "register_distinct_values": len({v for _, v in series}),
        "register_value_change_rate": round(distinctness(series), 3),
        "lag_max": lag_max,
        "samples": len(per_sample),
        "verdict": verdict,
        "per_phase": phase_table,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "request_ns", "phase_ns", "task", "value", "lags"])
        for row in result["per_sample"]:
            w.writerow([row["run_id"], row["request_ns"], row["phase_ns"], row["task"],
                        "0x%08x" % row["value"], "|".join(str(k) for k in row["lags"])])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    verdict = result["verdict"]
    lines = ["# 寄存器读回 bank 判据（只读实验）\n"]
    lines.append("寄存器：`%s` @ `%s`；样本 %d 个；该寄存器在 %d 个 task 里被写、"
                 "%d 个不同取值、跨 task 变化率 %.2f。\n" % (
                     result["module"], result["offset"], result["samples"],
                     result["register_tasks"], result["register_distinct_values"],
                     result["register_value_change_rate"]))
    lines.append("\n## 判定：**%s**\n\n%s\n" % (verdict["verdict"], verdict["note"]))
    if verdict["lag_histogram"]:
        lines.append("\n| 读回滞后（写作 task 数） | 样本数 |")
        lines.append("|---|---|")
        for lag, count in verdict["lag_histogram"].items():
            meaning = {0: "本 task 刚写的值", 1: "上一个写作 task 的值"}.get(
                int(lag), "更早")
            lines.append("| %s（%s） | %d |" % (lag, meaning, count))
    if verdict.get("samples_ambiguous"):
        lines.append("\n另有 %d 个样本同时匹配多个滞后（寄存器取值重复），未计入上表。"
                     % verdict["samples_ambiguous"])
    if verdict.get("samples_unmatched"):
        lines.append("\n另有 %d 个样本不匹配任何已知写入值。" % verdict["samples_unmatched"])

    if result["per_phase"]:
        lines.append("\n## 按请求相位分解\n")
        lines.append("| 请求 φ (µs) | n | 滞后分布 |")
        lines.append("|---|---|---|")
        for row in result["per_phase"]:
            lines.append("| %.1f | %d | %s |" % (
                row["request_ns"] / 1000.0, row["n"],
                ", ".join("%s×%d" % (k, v) for k, v in row["lag_histogram"].items())))
    lines.append("\n## 这份判据能说明什么\n")
    lines.append("- `follows_write`：主机读到当前 task 写入的那组 ⇒ 标记可直接按"
                 "“本 task 写了什么”比对。")
    lines.append("- `one_write_behind`：主机读到上一 task 填的那组 ⇒ ping-pong 生效且"
                 "读回错过最新写入；路线 B 仍可做，但期望值要整体滞后一个写作 task。")
    lines.append("- `unreadable_or_unknown`：读回值对不上任何已捕获写入 ⇒ 先查读回是否"
                 "返回别的组、模型是否确定性重放，再谈标记。")
    lines.append("\n**未覆盖**：本判据只回答“读到哪一组”，不回答写入是否有副作用，"
                 "也不给出标记可见时间与命令执行时间的对应关系。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# self test: simulate each plausible hardware behaviour
# ---------------------------------------------------------------------------

def _synth(tmpdir, hypothesis, n_tasks=40, reps=5):
    """Build a rows CSV plus a sweep manifest for one hardware hypothesis."""
    rnd = random.Random(11)
    os.makedirs(os.path.join(tmpdir, "runs"), exist_ok=True)
    rows_path = os.path.join(tmpdir, "rows.csv")
    values = [0x1000 + i for i in range(n_tasks)]  # every task writes a new value
    with open(rows_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_index", "task_local_regcmd_index", "module", "reg_offset",
                    "reg_name", "value"])
        for task, value in enumerate(values):
            w.writerow([task, 3, "CNA", "0x1024", "CNA_DATA_SIZE1", "0x%08x" % value])

    manifest = os.path.join(tmpdir, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "request_phase_ns", "csv", "condition"])
        for rep in range(reps):
            for task in range(1, n_tasks - 1):
                if hypothesis == "follows":
                    observed = values[task]
                elif hypothesis == "behind":
                    observed = values[task - 1]
                elif hypothesis == "alternating":
                    observed = values[task] if task % 2 == 0 else values[task - 1]
                elif hypothesis == "unreadable":
                    observed = 0xDEAD0000
                else:
                    raise AssertionError(hypothesis)
                phase = 10_000 + rep * 1_000
                path = os.path.join(tmpdir, "runs", "t%d_r%d.csv" % (task, rep))
                submit_enter = 1_000_000_000
                with open(path, "w", newline="") as fh2:
                    w2 = csv.writer(fh2)
                    w2.writerow(["t_ns", "t_end_ns", "c0_ts", "c0_raw", "c0_gb_ns",
                                 "c0_ge_ns", "submit_enter_ns", "task_before",
                                 "task_after", "sample_validity"])
                    w2.writerow([submit_enter + phase, submit_enter + phase + 400,
                                 "0x%08x" % task, "0x%08x" % observed,
                                 submit_enter + phase, submit_enter + phase + 400,
                                 submit_enter, task, task, "ok"])
                w.writerow(["t%d_r%d" % (task, rep), phase, path, ""])
    return rows_path, manifest


def selftest():
    import tempfile
    failures = 0

    def group(name, fn):
        nonlocal failures
        print(name)
        try:
            fn()
        except AssertionError as exc:
            print("  FAIL " + str(exc))
            failures += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        def check_hypothesis(hypothesis, expected):
            rows, manifest = _synth(os.path.join(tmpdir, hypothesis), hypothesis)
            result = analyse(rows, manifest, "CNA", 0x1024, 3, "c0_raw", 10, 0.8, 5)
            got = result["verdict"]["verdict"]
            assert got == expected, "expected %s, got %s (%s)" % (
                expected, got, result["verdict"]["note"])
            print("  ok  %-12s -> %s" % (hypothesis, got))

        group("readback that follows the current write",
              lambda: check_hypothesis("follows", "observed_lag_0"))
        group("readback one writing task behind",
              lambda: check_hypothesis("behind", "observed_lag_1"))
        group("readback alternating with task parity",
              lambda: check_hypothesis("alternating", "mixed_lag"))
        group("readback that matches nothing",
              lambda: check_hypothesis("unreadable", "no_value_matched"))

        def one_match_is_not_evidence():
            # Reviewer counterexample: 1 matching sample among 99 misses must not
            # be reported as a finding.
            root = os.path.join(tmpdir, "one_match")
            rows, manifest = _synth(root, "unreadable", n_tasks=20, reps=5)
            # patch exactly one sample so it matches the current task's value
            import glob
            victim = sorted(glob.glob(os.path.join(root, "runs", "t*_r0.csv")))[0]
            with open(victim, newline="") as fh:
                recs = list(csv.DictReader(fh))
            with open(victim, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
                w.writeheader()
                for rec in recs:
                    rec["c0_raw"] = "0x%08x" % (0x1000 + int(rec["task_before"]))
                    w.writerow(rec)
            result = analyse(rows, manifest, "CNA", 0x1024, 3, "c0_raw", 20, 0.8, 5)
            got = result["verdict"]["verdict"]
            assert got == "insufficient_evidence", got
            assert result["verdict"]["match_rate"] < 0.8
            print("  ok  1 match out of 100 -> %s" % got)

        group("a single matching sample is not a verdict", one_match_is_not_evidence)

        def too_few_tasks():
            rows, manifest = _synth(os.path.join(tmpdir, "fewtasks"), "follows",
                                    n_tasks=3, reps=10)
            result = analyse(rows, manifest, "CNA", 0x1024, 3, "c0_raw", 5, 0.8, 5)
            got = result["verdict"]["verdict"]
            assert got == "insufficient_evidence", got
            assert result["verdict"]["distinct_tasks"] < 5
            print("  ok  two distinct tasks -> %s" % got)

        group("task coverage is part of the evidence", too_few_tasks)

        def repeated_values_are_ambiguous():
            # a register whose value repeats every task cannot separate the groups
            root = os.path.join(tmpdir, "repeat")
            os.makedirs(os.path.join(root, "runs"), exist_ok=True)
            rows = os.path.join(root, "rows.csv")
            with open(rows, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["task_index", "task_local_regcmd_index", "module",
                            "reg_offset", "reg_name", "value"])
                for task in range(30):
                    w.writerow([task, 3, "CNA", "0x1024", "CNA_DATA_SIZE1", "0x00000007"])
            manifest = os.path.join(root, "manifest.csv")
            with open(manifest, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["run_id", "request_phase_ns", "csv", "condition"])
                for task in range(1, 29):
                    path = os.path.join(root, "runs", "t%d.csv" % task)
                    with open(path, "w", newline="") as fh2:
                        w2 = csv.writer(fh2)
                        w2.writerow(["t_ns", "t_end_ns", "c0_ts", "c0_raw", "c0_gb_ns",
                                     "c0_ge_ns", "submit_enter_ns", "task_before",
                                     "task_after", "sample_validity"])
                        w2.writerow([1_000_000_000 + 10_000, 1_000_000_000 + 10_400,
                                     "0x%08x" % task, "0x00000007",
                                     1_000_000_000 + 10_000, 1_000_000_000 + 10_400,
                                     1_000_000_000, task, task, "ok"])
                    w.writerow(["t%d" % task, 10_000, path, ""])
            result = analyse(rows, manifest, "CNA", 0x1024, 3, "c0_raw", 10, 0.8, 5)
            assert result["verdict"]["verdict"] == "ambiguous", result["verdict"]
            print("  ok  a repeating register is reported ambiguous, not guessed")

        group("a register that repeats values cannot separate groups",
              repeated_values_are_ambiguous)

        def unreadable_register_is_explicit():
            rows, manifest = _synth(os.path.join(tmpdir, "missing"), "follows")
            try:
                analyse(rows, manifest, "PPU", 0x6008)
            except SystemExit:
                print("  ok  a register never written in the stream is a hard error")
                return
            raise AssertionError("expected a hard failure")

        group("a register absent from the stream is rejected",
              unreadable_register_is_explicit)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all bank-probe selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", help="regcmd rows CSV (captured stream)")
    ap.add_argument("--manifest", help="sweep manifest produced by run_phase_sweep.sh")
    ap.add_argument("--module", default="CNA")
    ap.add_argument("--offset", default="0x1024")
    ap.add_argument("--lag-max", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--min-match-rate", type=float, default=0.8)
    ap.add_argument("--min-tasks", type=int, default=5)
    ap.add_argument("--signal", default="c0_raw")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.rows or not args.manifest or not args.out_dir:
        ap.error("--rows, --manifest and --out-dir are required unless --selftest")
    result = analyse(args.rows, args.manifest, args.module, int(args.offset, 0),
                     args.lag_max, args.signal, args.min_samples,
                     args.min_match_rate, args.min_tasks)
    write_outputs(result, args.out_dir)
    verdict = result["verdict"]
    print("wrote %s" % args.out_dir)
    print("verdict: %s" % verdict["verdict"])
    print("note: %s" % verdict["note"])
    if verdict["lag_histogram"]:
        print("lag histogram: %s" % verdict["lag_histogram"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
