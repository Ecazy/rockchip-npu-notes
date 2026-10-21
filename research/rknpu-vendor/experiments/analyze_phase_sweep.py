#!/usr/bin/env python3
"""Analyse the route-A phase sweep: one snapshot per run, many runs, swept phi.

Design: research/scheduling/REGCMD-TIMING-OBSERVABILITY.md, section 3 (route A)
and section 4 step 4 (marker CDF).  Each run takes ONE snapshot after a delay
phi measured from the recorded submit entry, so a run costs a couple of MMIO
reads rather than a high-frequency poll.  Repeating the run at different phi
rebuilds the in-task distribution.

Two signal kinds are kept strictly apart:

  status  (route A)  the snapshot is a module S_STATUS word -> the result is a
          *probability of being busy at phi*.  It may rise and fall again and is
          NOT a CDF.  No monotone fit, no quantiles.
  marker  (route B)  the snapshot is one register compared against a token ->
          the result may be read as "probability the marker has been written",
          and only then (and only if the observed curve is monotone within
          noise) is a CDF fitted and inverted for p10/p50/p90.

Guarantees this script makes (they are asserted in --selftest):

  * non-monotone busy curves are preserved verbatim (0,1,1,0 stays 0,1,1,0);
  * resolution is judged per condition and from the LOCAL pair of samples that
    brackets the transition, never from a global minimum step;
  * an unobserved transition is reported as left/right censored or
    not_observed, never as a point estimate and never as an exception;
  * the x axis is submit-relative (t_ns - submit_enter_ns).  That is NOT the
    hardware kick: the syscall, driver preparation and queueing all happen
    after it, so a constant offset remains and is reported as such;
  * samples marked task_crossed / task_ambiguous are dropped and counted.

Manifest (CSV, one row per run):

    run_id,request_phase_ns,csv[,condition]

Usage:

    python3 analyze_phase_sweep.py --manifest sweep.csv --signal c0_cna \
        --out-dir results/phase_sweep_20261021
    python3 analyze_phase_sweep.py --manifest sweep.csv --raw \
        --marker-value 0xdeadbeef --out-dir results/marker_cdf
    python3 analyze_phase_sweep.py --selftest
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys

SUBMIT_COLUMNS = ("submit_enter_ns", "kick_ns")  # second is the legacy name
VALID_VALIDITY = {"ok", "unchecked", ""}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def status_busy(value):
    """S_STATUS busy decoding, identical to analyze_sdpa_module_occupancy.py."""
    reg = int(value, 16) if isinstance(value, str) else int(value)
    s0 = reg & 0x3
    s1 = (reg >> 16) & 0x3
    return s0 in (1, 2) or s1 in (1, 2)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def load_manifest(path):
    runs = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        need = {"run_id", "request_phase_ns", "csv"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            sys.exit("manifest missing columns: %s" % ", ".join(sorted(missing)))
        base = os.path.dirname(os.path.abspath(path))
        for raw in reader:
            csv_path = raw["csv"]
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(base, csv_path)
            runs.append({
                "run_id": raw["run_id"],
                "request_ns": int(raw["request_phase_ns"]),
                "path": csv_path,
                "condition": (raw.get("condition") or "").strip(),
            })
    if not runs:
        sys.exit("empty manifest")
    return runs


def submit_enter_of(row):
    for column in SUBMIT_COLUMNS:
        value = row.get(column)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def load_run(run, signal_column, marker_value=None):
    """Return the run's samples, or None when the submit entry is missing."""
    samples = []
    with open(run["path"], newline="") as fh:
        for row in csv.DictReader(fh):
            submit_enter = submit_enter_of(row)
            if submit_enter <= 0:
                return None
            if signal_column not in row:
                sys.exit("column %s missing in %s" % (signal_column, run["path"]))
            raw = row[signal_column]
            if marker_value is not None:
                seen = (int(raw, 16) == marker_value
                        if raw.lower().startswith("0x") else int(raw) == marker_value)
                value = 1.0 if seen else 0.0
            else:
                if signal_column == "c0_ts" or signal_column.startswith("c0_gb"):
                    sys.exit("%s is not a signal column" % signal_column)
                value = 1.0 if status_busy(raw) else 0.0
            t = int(row["t_ns"])
            samples.append({
                "run_id": run["run_id"],
                "request_ns": run["request_ns"],
                "actual_ns": t - submit_enter,
                "latency_ns": int(row.get("t_end_ns") or t) - t,
                "value": value,
                "validity": (row.get("sample_validity") or "unchecked").strip(),
                "task_before": row.get("task_before", ""),
                "condition": run["condition"],
            })
    return samples


def group_by_request(samples):
    groups = {}
    for s in samples:
        groups.setdefault((s["condition"], s["request_ns"]), []).append(s)
    out = []
    for (condition, request_ns) in sorted(groups, key=lambda key: (key[1], key[0])):
        rows = groups[(condition, request_ns)]
        values = [r["value"] for r in rows]
        k = sum(1 for v in values if v >= 0.5)
        n = len(values)
        p, lo, hi = wilson(k, n)
        phases = sorted(r["actual_ns"] for r in rows)
        latencies = sorted(r["latency_ns"] for r in rows)

        def pct(seq, q):
            if not seq:
                return 0
            return seq[min(len(seq) - 1, max(0, int(round(q * (len(seq) - 1)))))]

        out.append({
            "condition": condition,
            "request_ns": request_ns,
            "n": n,
            "k": k,
            "p": p,
            "ci_lo": lo,
            "ci_hi": hi,
            "actual_p10_ns": pct(phases, 0.10),
            "actual_p50_ns": pct(phases, 0.50),
            "actual_p90_ns": pct(phases, 0.90),
            "latency_p50_ns": pct(latencies, 0.50),
            "latency_max_ns": max(latencies) if latencies else 0,
        })
    return out


def isotonic(points):
    """Weighted PAVA over (x, p, n), sorted by x.  Returns [(x, monotone p)]."""
    blocks = []
    for x, p, n in sorted(points):
        blocks.append([p * n, float(n), [x]])
        while len(blocks) >= 2:
            a, b = blocks[-2], blocks[-1]
            if a[0] / a[1] <= b[0] / b[1]:
                break
            blocks[-2:] = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]]
    out = []
    for total, weight, xs in blocks:
        for x in xs:
            out.append((x, total / weight))
    return out


def invert(curve, level):
    """First x where the monotone curve reaches `level`, or None."""
    prev = None
    for x, y in curve:
        if y >= level:
            if prev is None:
                return x
            x0, y0 = prev
            if y == y0:
                return x
            return x0 + (x - x0) * (level - y0) / (y - y0)
        prev = (x, y)
    return None


def quantile(curve, level, coarse=False):
    """Locate one quantile, respecting both the window edges and the sampling.

    Three outcomes, none of which claims more than the data supports:

      left_censored   the first sample already exceeds `level` -> upper bound;
      right_censored  the curve never reaches `level`           -> lower bound;
      estimated       the level is crossed between two adjacent CDF points, so
                      the number is a LINEAR INTERPOLATION, never a measurement.
                      The bracket it was interpolated inside is always reported.
                      When the sweep is too coarse to place any sample inside
                      that bracket (`coarse=True`), the status is `bracketed`:
                      the bracket is the result, the interpolated number is kept
                      only as a helper and is labelled as such.
    """
    if not curve:
        return {"status": "no_data"}
    if curve[0][1] >= level:
        return {"status": "left_censored", "upper_ns": curve[0][0]}
    if curve[-1][1] < level:
        return {"status": "right_censored", "lower_ns": curve[-1][0]}
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if y0 < level <= y1:
            estimate = x1 if y1 <= y0 else x0 + (x1 - x0) * (level - y0) / (y1 - y0)
            return {
                "status": "bracketed" if coarse else "estimated",
                "bracket_ns": [x0, x1],
                "estimate_ns": estimate,
                "method": "linear interpolation between adjacent CDF points",
            }
    return {"status": "not_reached"}


def transition_report(points, marker_mode, level_lo=0.10, level_hi=0.90):
    """Per-condition verdict: resolved / bracketed_only / unresolved / censored.

    `points` must belong to ONE condition.  The bracket is the local pair of
    sample groups that straddles the rise, never a global minimum step.
    """
    pts = sorted(points, key=lambda pt: pt["actual_p50_ns"])
    n_total = sum(pt["n"] for pt in pts)
    if not pts:
        return {"status": "no_data", "n": 0, "note": "no samples for this condition"}

    # Censoring direction: the observation window has a lower and an upper edge.
    # Never seeing the signal means the transition lies ABOVE the window
    # (right-censored); seeing it from the very first phase means it lies BELOW
    # the window (left-censored).
    if all(pt["p"] <= level_lo for pt in pts):
        return {"status": "right_censored", "n": n_total,
                "bound": {"lower_ns": pts[-1]["actual_p50_ns"]},
                "note": "signal never appears inside the swept window; the transition is "
                        "later than %.1f us or does not happen in this task"
                        % (pts[-1]["actual_p50_ns"] / 1000.0)}
    if all(pt["p"] >= level_hi for pt in pts):
        return {"status": "left_censored", "n": n_total,
                "bound": {"upper_ns": pts[0]["actual_p50_ns"]},
                "note": "signal is already present at the earliest phase (%.1f us); the "
                        "transition is earlier than the swept window"
                        % (pts[0]["actual_p50_ns"] / 1000.0)}

    lo_pt = None
    hi_pt = None
    for pt in pts:
        if pt["p"] <= level_lo:
            lo_pt = pt
        elif lo_pt is not None and pt["p"] >= level_hi:
            hi_pt = pt
            break
    if lo_pt is None or hi_pt is None:
        return {"status": "not_observed", "n": n_total,
                "note": "probability rises but never reaches %.0f%% inside the window; "
                        "extend the sweep" % (level_hi * 100)}

    rise_ns = hi_pt["actual_p50_ns"] - lo_pt["actual_p50_ns"]
    latency = statistics.median([pt["latency_p50_ns"] for pt in pts])
    jitter = max(pt["actual_p90_ns"] - pt["actual_p10_ns"] for pt in pts)
    resolvable_ns = max(latency, jitter)
    between = [pt for pt in pts
               if lo_pt["actual_p50_ns"] < pt["actual_p50_ns"] < hi_pt["actual_p50_ns"]]

    if rise_ns <= resolvable_ns:
        status = "unresolved"
        note = ("the rise (%.1f us) is not wider than the measurement resolution "
                "(%.1f us: read latency / phase jitter): only a bracket is measured"
                % (rise_ns / 1000.0, resolvable_ns / 1000.0))
    elif not between:
        status = "bracketed_only"
        note = ("no sample lies between %.1f us and %.1f us: the transition is bracketed "
                "by that %.1f us gap, its position inside the gap is not measured"
                % (lo_pt["actual_p50_ns"] / 1000.0, hi_pt["actual_p50_ns"] / 1000.0,
                   rise_ns / 1000.0))
    else:
        status = "resolved"
        note = ("rise spans %.1f us with %d intermediate sample group(s)"
                % (rise_ns / 1000.0, len(between)))
    return {
        "status": status,
        "n": n_total,
        "bracket_ns": [lo_pt["actual_p50_ns"], hi_pt["actual_p50_ns"]],
        "rise_ns": rise_ns,
        "local_gap_ns": rise_ns,
        "resolvable_ns": resolvable_ns,
        "intermediate_groups": len(between),
        "marker_mode": marker_mode,
        "note": note,
    }


def monotonicity_test(points):
    """Can a marker curve be treated as non-decreasing?

    Two things are reported:
      * max_cumulative_drop -- the largest p_i - p_j over ALL pairs i < j, so a
        slow decline spread over many steps (1.0, 0.9, 0.8, ... 0.0) is not
        missed by an adjacent-point check;
      * significant -- whether some pair is separated by its Wilson intervals
        (ci_lo_i > ci_hi_j).  A fixed drop threshold would both miss small but
        certain declines and reject small-sample noise, so significance is
        decided by the intervals instead.

    With too few samples per phase nothing can be significant; that case is
    reported separately by the caller (`insufficient_samples`) rather than being
    silently accepted as monotone.
    """
    pts = sorted(points, key=lambda pt: pt["actual_p50_ns"])
    worst_drop = 0.0
    worst_pair = None
    significant_pair = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            drop = pts[i]["p"] - pts[j]["p"]
            if drop > worst_drop:
                worst_drop = drop
                worst_pair = (pts[i], pts[j])
            if pts[i]["ci_lo"] > pts[j]["ci_hi"] and significant_pair is None:
                significant_pair = (pts[i], pts[j], drop)
    return {
        "max_cumulative_drop": round(worst_drop, 4),
        "worst_pair_ns": ([worst_pair[0]["actual_p50_ns"], worst_pair[1]["actual_p50_ns"]]
                          if worst_pair else None),
        "significant": significant_pair is not None,
        "significant_pair_ns": ([significant_pair[0]["actual_p50_ns"],
                                 significant_pair[1]["actual_p50_ns"]]
                                if significant_pair else None),
        "significant_drop": round(significant_pair[2], 4) if significant_pair else 0.0,
    }


def analyse(runs, signal_column, marker_value=None, min_samples=10):
    samples = []
    skipped = []
    dropped = []
    for run in runs:
        got = load_run(run, signal_column, marker_value)
        if got is None:
            skipped.append(run["run_id"])
            continue
        for sample in got:
            if sample["validity"] not in VALID_VALIDITY:
                dropped.append({"run_id": sample["run_id"],
                                "validity": sample["validity"],
                                "task_before": sample["task_before"]})
                continue
            samples.append(sample)
    if not samples:
        sys.exit("no usable samples (missing submit_enter_ns, or all dropped)")

    points = group_by_request(samples)
    conditions = {}
    for condition in sorted({pt["condition"] for pt in points}):
        pts = [pt for pt in points if pt["condition"] == condition]
        curve = sorted((pt["actual_p50_ns"], pt["p"], pt["n"]) for pt in pts)
        block = {
            "status_curve": [{"phase_ns": x, "p": p, "n": n} for x, p, n in curve],
            "transition": transition_report(pts, marker_value is not None),
            # status mode never has a CDF: p(busy|phi) is not a distribution over
            # a monotone event, so no quantiles are produced for it.
            "cdf": None,
        }
        if marker_value is not None:
            test = monotonicity_test(pts)
            block["monotonicity"] = test
            thin = [pt for pt in pts if pt["n"] < min_samples]
            if thin:
                block["cdf"] = None
                block["transition"]["status"] = "insufficient_samples"
                block["transition"]["note"] = (
                    "at least one phase has %d sample(s) < %d: monotonicity cannot be "
                    "judged, so no CDF is fitted"
                    % (min(pt["n"] for pt in thin), min_samples))
            elif test["significant"]:
                block["cdf"] = None
                block["transition"]["status"] = "non_monotone_marker"
                block["transition"]["note"] = (
                    "marker probability drops by %.2f between %.1f us and %.1f us with "
                    "non-overlapping intervals (max cumulative drop %.2f): the register "
                    "is not a persistent marker (overwritten or bank-switched); "
                    "refusing to invert a CDF"
                    % (test["significant_drop"], test["significant_pair_ns"][0] / 1000.0,
                       test["significant_pair_ns"][1] / 1000.0,
                       test["max_cumulative_drop"]))
            else:
                mono = isotonic(curve)
                coarse = block["transition"]["status"] == "bracketed_only"
                block["cdf"] = {
                    "p10": quantile(mono, 0.10, coarse),
                    "p50": quantile(mono, 0.50, coarse),
                    "p90": quantile(mono, 0.90, coarse),
                }
                block["cdf_method"] = ("linear interpolation between adjacent CDF "
                                       "points; never a measured instant")
        conditions[condition] = block

    return {
        "signal": signal_column,
        "marker_value": marker_value,
        "mode": "marker_cdf" if marker_value is not None else "status_probability",
        "time_origin": "submit_enter (user-space ioctl entry), NOT the hardware kick",
        "runs": len(runs),
        "runs_skipped": skipped,
        "samples": len(samples),
        "samples_dropped": dropped,
        "points": points,
        "conditions": conditions,
    }


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["condition", "request_ns", "actual_p10_ns", "actual_p50_ns", "actual_p90_ns",
            "n", "k", "p", "ci_lo", "ci_hi", "latency_p50_ns", "latency_max_ns"]
    with open(os.path.join(out_dir, "phase_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for pt in result["points"]:
            w.writerow([pt[c] for c in cols])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    marker = result["marker_value"] is not None
    lines = ["# 相位扫描结果\n"]
    lines.append("signal = `%s`%s；模式 = **%s**；%d 个 run，%d 个有效样本。\n" % (
        result["signal"],
        "" if result["marker_value"] is None else "（标记值 0x%x）" % result["marker_value"],
        "marker CDF" if marker else "module status probability",
        result["runs"], result["samples"]))
    lines.append("时间轴 = **提交入口相对时间**（`t_ns − submit_enter_ns`），不是硬件 kick。"
                 "真实 kick 与提交入口之间隔着系统调用、驱动准备和排队，这段延迟"
                 "**未标定，且可能逐次变化**：它既会平移分布，也会把它展宽。"
                 "因此不能靠减去一个常数把这里的结果换算成硬件时序。\n")
    if not marker:
        lines.append("status 模式下 p̂ 是“该 φ 上模块处于 busy 的概率”，**允许非单调**；"
                     "本文件不对它做单调拟合，也不输出分位数。\n")
    if result["runs_skipped"]:
        lines.append("\n**丢弃的 run**（未记录 submit_enter）：%s\n"
                     % ", ".join(result["runs_skipped"]))
    if result["samples_dropped"]:
        lines.append("\n**丢弃的样本**（跨 task / task 歧义）：%d 条 —— %s\n" % (
            len(result["samples_dropped"]),
            ", ".join("%s(%s)" % (d["run_id"], d["validity"])
                      for d in result["samples_dropped"][:8])))

    lines.append("\n| condition | 请求 φ (µs) | 实测 φ p50 (µs) | 抖动 p10–p90 (µs) | n | p̂ | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for pt in result["points"]:
        lines.append("| %s | %.1f | %.1f | %.1f–%.1f | %d | %.2f | %.2f–%.2f |" % (
            pt["condition"] or "-", pt["request_ns"] / 1000.0, pt["actual_p50_ns"] / 1000.0,
            pt["actual_p10_ns"] / 1000.0, pt["actual_p90_ns"] / 1000.0,
            pt["n"], pt["p"], pt["ci_lo"], pt["ci_hi"]))

    lines.append("\n## 每个 condition 的判定\n")
    for condition, block in result["conditions"].items():
        tr = block["transition"]
        lines.append("### `%s`\n" % (condition or "(default)"))
        lines.append("- 判定：**%s** —— %s" % (tr["status"], tr["note"]))
        if "bracket_ns" in tr:
            lines.append("- 括住的区间：%.1f–%.1f µs（局部间隔 %.1f µs）" % (
                tr["bracket_ns"][0] / 1000.0, tr["bracket_ns"][1] / 1000.0,
                tr["local_gap_ns"] / 1000.0))
            lines.append("- 测量分辨率：%.2f µs（读延迟/相位抖动）"
                         % (tr["resolvable_ns"] / 1000.0))
        if marker:
            def fmt_quantile(entry):
                if entry is None:
                    return "n/a"
                if entry["status"] in ("estimated", "bracketed"):
                    text = "≈%.1f µs（区间 %.1f–%.1f µs，线性插值）" % (
                        entry["estimate_ns"] / 1000.0,
                        entry["bracket_ns"][0] / 1000.0,
                        entry["bracket_ns"][1] / 1000.0)
                    if entry["status"] == "bracketed":
                        text = "仅括住 " + text.replace("≈", "插值 ")
                    return text
                if entry["status"] == "left_censored":
                    return "≤ %.1f µs（早于观察窗口）" % (entry["upper_ns"] / 1000.0)
                if entry["status"] == "right_censored":
                    return "≥ %.1f µs（晚于观察窗口）" % (entry["lower_ns"] / 1000.0)
                return entry["status"]

            if block.get("cdf"):
                cdf = block["cdf"]
                lines.append("- 标记可见时间（提交入口相对）：p10=%s, p50=%s, p90=%s" % (
                    fmt_quantile(cdf["p10"]), fmt_quantile(cdf["p50"]),
                    fmt_quantile(cdf["p90"])))
            test = block.get("monotonicity") or {}
            if test:
                lines.append("- 单调性：最大累计下降 %.2f（区间不重叠的下降：%s）"
                             % (test.get("max_cumulative_drop", 0.0),
                                "有" if test.get("significant") else "无"))
        lines.append("")
    lines.append("\n估计只回答“在此 φ 上观察到 signal 的概率”，不等于某次运行的时间戳，"
                 "也不构成跨模块同步证据。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _synth_runs(tmpdir, phases, reps=1, values=None, transition_ns=30_000,
                jitter_ns=1_000, seed=7, truncate_payload=False):
    """Synthetic sweep: `reps` runs per requested phase.

    `values` (optional) gives the per-phase on/off truth as a probability; each
    run of that phase is on with that probability, which lets a test build a
    non-monotone or gradually declining curve on purpose.
    """
    rnd = random.Random(seed)
    runs = []
    submit_enter = 1_000_000_000
    index = 0
    for phase_index, request in enumerate(phases):
        for _rep in range(reps):
            path = os.path.join(tmpdir, "run%04d.csv" % index)
            lateness = int(rnd.gauss(0, jitter_ns / 2)) if jitter_ns else 0
            actual = max(0, request + lateness)
            if values is not None:
                on = rnd.random() < values[phase_index]
            else:
                on = actual >= transition_ns + rnd.gauss(0, jitter_ns)
            latency = 400
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_ns", "t_end_ns", "c0_gb_ns", "c0_ge_ns", "c0_ts",
                            "c0_cna", "c0_core", "c0_dpu", "c0_rdma", "submit_enter_ns",
                            "task_before", "task_after", "sample_validity"])
                w.writerow([submit_enter + actual, submit_enter + actual + latency,
                            submit_enter + actual, submit_enter + actual + latency,
                            "0x00000001", "0x00000001" if on else "0x00000000",
                            "0x0", "0x0", "0x0",
                            "" if truncate_payload else submit_enter, 7, 7, "ok"])
            runs.append({"run_id": "r%04d" % index, "request_ns": request,
                         "path": path, "condition": ""})
            index += 1
    return runs


def _as_marker(runs, token=0x00C0FFEE):
    for run in runs:
        with open(run["path"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(run["path"], "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                row["c0_cna"] = ("0x%08x" % token if row["c0_cna"] == "0x00000001"
                                 else "0x00000000")
                w.writerow(row)


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

    sweep_phases = list(range(0, 40_000, 2_000))

    with tempfile.TemporaryDirectory() as tmpdir:
        def status_sweep():
            runs = _synth_runs(tmpdir, sweep_phases, reps=5, jitter_ns=2_000)
            block = analyse(runs, "c0_cna")["conditions"][""]
            assert block["cdf"] is None
            assert block["transition"]["status"] in ("bracketed_only", "resolved",
                                                     "unresolved"), block["transition"]
            print("  ok  status mode reports %s and never a CDF"
                  % block["transition"]["status"])

        group("status mode reports a transition but no CDF", status_sweep)

        def marker_sweep():
            runs = _synth_runs(tmpdir, sweep_phases, reps=15, jitter_ns=1_000)
            _as_marker(runs)
            result = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)
            cdf = result["conditions"][""]["cdf"]
            assert cdf is not None, result["conditions"][""]["transition"]
            p50 = cdf["p50"]
            assert p50["status"] == "estimated", p50
            assert p50["bracket_ns"][0] <= p50["estimate_ns"] <= p50["bracket_ns"][1], p50
            assert "linear interpolation" in p50["method"], p50
            value = p50["estimate_ns"]
            assert abs(value - 30_000) <= 5_000, value
            print("  ok  marker p50 ≈ %.1f us inside [%.1f, %.1f] (truth 30.0, "
                  "status=estimated)" % (value / 1000.0, p50["bracket_ns"][0] / 1000.0,
                                         p50["bracket_ns"][1] / 1000.0))

        group("marker mode inverts a persistent marker curve", marker_sweep)

        def non_monotone_preserved():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000, 30_000], reps=5,
                               values=[0.0, 1.0, 1.0, 0.0], jitter_ns=0)
            result = analyse(runs, "c0_cna")
            curve = [pt["p"] for pt in result["points"]]
            assert curve == [0.0, 1.0, 1.0, 0.0], curve
            assert result["conditions"][""]["cdf"] is None
            print("  ok  busy curve 0,1,1,0 stays 0,1,1,0 (no PAVA in status mode)")

        group("status mode preserves a falling edge", non_monotone_preserved)

        def non_persistent_marker():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000, 30_000], reps=20,
                               values=[0.0, 1.0, 1.0, 0.0], jitter_ns=0)
            _as_marker(runs)
            block = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)["conditions"][""]
            assert block["cdf"] is None, "a falling marker curve must not be inverted"
            assert block["transition"]["status"] == "non_monotone_marker"
            assert block["monotonicity"]["significant"] is True
            print("  ok  falling marker refused (drop %.2f, cumulative %.2f)"
                  % (block["monotonicity"]["significant_drop"],
                     block["monotonicity"]["max_cumulative_drop"]))

        group("marker mode refuses a non-persistent marker", non_persistent_marker)

        def gradual_decline():
            # Reviewer's case: 0, 1, 0.9, 0.8, ... 0.1 -- every adjacent step is
            # small, the cumulative drop is 1.0.
            phases = [i * 5_000 for i in range(11)]
            values = [0.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            runs = _synth_runs(tmpdir, phases, reps=30, values=values, jitter_ns=0, seed=3)
            _as_marker(runs)
            block = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)["conditions"][""]
            test = block["monotonicity"]
            assert test["max_cumulative_drop"] >= 0.8, test
            assert test["significant"] is True, test
            assert block["cdf"] is None
            assert block["transition"]["status"] == "non_monotone_marker"
            print("  ok  gradual decline detected (cumulative %.2f, significant)"
                  % test["max_cumulative_drop"])

        group("a decline spread over many steps is still detected", gradual_decline)

        def insufficient():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000, 30_000], reps=3,
                               values=[0.0, 1.0, 1.0, 1.0], jitter_ns=0)
            _as_marker(runs)
            block = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)["conditions"][""]
            assert block["cdf"] is None, "3 samples per phase cannot justify a CDF"
            assert block["transition"]["status"] == "insufficient_samples"
            print("  ok  thin sweeps are refused as %s" % block["transition"]["status"])

        group("too few samples per phase cannot be called monotone", insufficient)

        def local_gap():
            # Reviewer counterexample: 0, 1, 101 us with p = 0, 0, 1 -> the
            # transition is bracketed by a 100 us gap.
            runs = _synth_runs(tmpdir, [0, 1_000, 101_000], reps=5, jitter_ns=0,
                               values=[0.0, 0.0, 1.0])
            tr = analyse(runs, "c0_cna")["conditions"][""]["transition"]
            assert tr["status"] == "bracketed_only", tr
            assert abs(tr["local_gap_ns"] - 100_000) < 2_000, tr["local_gap_ns"]
            print("  ok  uneven sweep judged on the local gap (%.1f us), status=%s"
                  % (tr["local_gap_ns"] / 1000.0, tr["status"]))

        group("resolution uses the local bracket, not the global min step", local_gap)

        def censoring():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000], reps=5, jitter_ns=0,
                               values=[0.0, 0.0, 0.0])
            tr = analyse(runs, "c0_cna")["conditions"][""]["transition"]
            assert tr["status"] == "right_censored", tr
            assert tr["bound"]["lower_ns"] >= 19_000, tr
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000], reps=5, jitter_ns=0,
                               values=[1.0, 1.0, 1.0])
            tr = analyse(runs, "c0_cna")["conditions"][""]["transition"]
            assert tr["status"] == "left_censored", tr
            assert tr["bound"]["upper_ns"] <= 1_000, tr
            print("  ok  never-seen -> right_censored, always-seen -> left_censored")

        group("censoring direction follows the window edges", censoring)

        def censored_quantiles():
            runs = _synth_runs(tmpdir, [5_000, 10_000, 15_000], reps=12, jitter_ns=0,
                               values=[1.0, 1.0, 1.0])
            _as_marker(runs)
            block = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)["conditions"][""]
            cdf = block["cdf"]
            assert cdf is not None, block["transition"]
            for key in ("p10", "p50", "p90"):
                assert cdf[key]["status"] == "left_censored", cdf[key]
                assert cdf[key]["upper_ns"] <= 5_000, cdf[key]
                assert "value_ns" not in cdf[key], cdf[key]
            print("  ok  always-visible marker yields bounded quantiles (<= 5.0 us), "
                  "not 0")

        group("a censored quantile is a bound, not zero", censored_quantiles)

        def coarse_quantile():
            # Only two phases, far apart: the sweep cannot place anything inside
            # the bracket, so the quantile must be reported as a bracket.
            runs = _synth_runs(tmpdir, [2_000, 40_000], reps=15, jitter_ns=0,
                               values=[0.0, 1.0])
            _as_marker(runs)
            block = analyse(runs, "c0_cna", marker_value=0x00C0FFEE)["conditions"][""]
            assert block["transition"]["status"] == "bracketed_only", \
                block["transition"]
            for key in ("p10", "p50", "p90"):
                entry = block["cdf"][key]
                assert entry["status"] == "bracketed", entry
                assert entry["bracket_ns"] == [2_000, 40_000] or \
                    entry["bracket_ns"][0] < entry["bracket_ns"][1], entry
            print("  ok  coarse sweep reports a bracket, not a measured quantile")

        group("an interpolated quantile is never called measured", coarse_quantile)

        def dropped():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000], reps=1, jitter_ns=0,
                               values=[0.0, 1.0, 0.0])
            path = runs[1]["path"]
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                row["sample_validity"] = "task_crossed"
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            result = analyse(runs, "c0_cna")
            assert len(result["samples_dropped"]) == 1, result["samples_dropped"]
            assert result["samples"] == 2
            print("  ok  task_crossed sample dropped and counted")

        group("task-crossed samples are dropped", dropped)

        def missing_submit():
            runs = _synth_runs(tmpdir, [0, 10_000], reps=1, truncate_payload=True)
            try:
                analyse(runs, "c0_cna")
            except SystemExit:
                print("  ok  missing submit_enter is a hard error, not an interpolation")
                return
            raise AssertionError("expected a hard failure")

        group("runs without a submit entry are rejected", missing_submit)

        def conditions():
            runs = _synth_runs(tmpdir, [0, 10_000, 20_000, 30_000], reps=1,
                               values=[0.0, 0.0, 1.0, 1.0])
            for run in runs[2:]:
                run["condition"] = "other"
            result = analyse(runs, "c0_cna")
            assert set(result["conditions"]) == {"", "other"}, list(result["conditions"])
            print("  ok  conditions are analysed separately")

        group("conditions are not pooled", conditions)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all phase-sweep selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--signal", default="c0_cna",
                    help="blockwatch column to read: c0_cna/c0_core/c0_dpu/c0_rdma, "
                         "or c0_raw in --raw mode")
    ap.add_argument("--raw", action="store_true",
                    help="the sweep used EPERF_RKNPU_BLOCKWATCH_RAW_OFF (single register)")
    ap.add_argument("--marker-value", default=None,
                    help="hex token meaning 'marker visible'; switches to marker/CDF mode")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="minimum samples per phase before a marker CDF is fitted")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.manifest or not args.out_dir:
        ap.error("--manifest and --out-dir are required unless --selftest is given")
    marker = int(args.marker_value, 0) if args.marker_value else None
    signal = args.signal
    if args.raw and signal == "c0_cna":
        signal = "c0_raw"
    if marker is not None and not args.raw:
        ap.error("--marker-value requires --raw (the marker must be read as c0_raw)")
    result = analyse(load_manifest(args.manifest), signal, marker, args.min_samples)
    write_outputs(result, args.out_dir)
    print("wrote %s (%s mode)" % (args.out_dir, result["mode"]))
    for pt in result["points"]:
        print("  phi=%8.1f us  n=%3d  p=%.2f [%.2f,%.2f]" % (
            pt["actual_p50_ns"] / 1000.0, pt["n"], pt["p"], pt["ci_lo"], pt["ci_hi"]))
    for condition, block in result["conditions"].items():
        print("  [%s] %s: %s" % (condition or "-", block["transition"]["status"],
                                 block["transition"]["note"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
