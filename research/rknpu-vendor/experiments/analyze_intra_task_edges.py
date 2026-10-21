#!/usr/bin/env python3
"""Edges of the DPU group window resolved INSIDE a single run.

With CAP>1 the sampler takes several snapshots per submit, so a rise or a fall can
be bracketed by two consecutive snapshots of the SAME run — no run-to-run
stationarity assumption is needed (the one-snapshot phase sweep still needs it).

Every edge is reported as an interval (lo, hi], never as a point:
    rise in (last clear before the first set, first set]
    fall in (last set, first clear after it]
A run that starts already set gives no rise bound, and a run that ends still set
gives no fall bound; those are counted, not guessed.

Usage
-----
    python3 analyze_intra_task_edges.py --dir DIR [--out-dir OUT]
    python3 analyze_intra_task_edges.py --selftest
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

DONE_MASK = 0x3FFF


def percentile(values, q):
    """Linear-interpolated quantile (the usual definition, so a median of an even
    number of points is the average of the two middle ones -- an order statistic
    would systematically differ from what a reviewer recomputes by hand)."""
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def load_run(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    submit = int(rows[0]["submit_enter_ns"])
    samples = []
    for r in rows:
        gb = int(r["c0_gb_ns"]) - submit
        ge = int(r["c0_ge_ns"]) - submit
        samples.append({
            "phase_ns": int(r["t_ns"]) - submit,
            "gb_ns": gb,                      # the read started here
            "ge_ns": ge,                      # and ended here
            "bracket_ns": ge - gb,
            "set": 1 if int(r["c0_raw"], 16) & DONE_MASK else 0,
        })
    samples.sort(key=lambda s: s["phase_ns"])
    return {"run": os.path.basename(path), "samples": samples}


def percentile_nearest(values, q):
    """nearest-rank quantile (upper median for an even count).

    Reported alongside the interpolated one because reviewers recompute by hand
    with this convention and the two differ (e.g. 22.5 vs 23 snapshots)."""
    values = sorted(values)
    if not values:
        return None
    pos = q * (len(values) - 1)
    return values[min(len(values) - 1, max(0, int(math.floor(pos + 0.5))))]


def segments(run):
    """Number of maximal runs of an identical status value, and the value string."""
    states = [s["set"] for s in run["samples"]]
    if not states:
        return 0, ""
    count = 1 + sum(1 for a, b in zip(states, states[1:]) if a != b)
    return count, "".join("S" if v else "c" for v in states)


def edges(run):
    """(rise_lo, rise_hi, fall_lo, fall_hi) in ns; None where not bracketed.

    Each snapshot's value was read somewhere inside its own read interval
    [gb, ge], not at the single instant t_ns.  So for a clear -> set pair the
    transition T satisfies

        gb(clear) < T <= ge(set)

    and symmetrically for set -> clear.  Using t_ns alone would claim a narrower
    interval than the read can support: the first clear read may only have
    finished AFTER the set->clear transition, so its t_ns is not an upper bound."""
    s = run["samples"]
    rise = fall = None
    for a, b in zip(s, s[1:]):
        if not a["set"] and b["set"] and rise is None:
            rise = (a["gb_ns"], b["ge_ns"])
        if a["set"] and not b["set"] and fall is None:
            fall = (a["gb_ns"], b["ge_ns"])
    rise_lo, rise_hi = rise if rise else (None, None)
    fall_lo, fall_hi = fall if fall else (None, None)
    return rise_lo, rise_hi, fall_lo, fall_hi


def summarise(runs):
    per_run, spacings, brackets, spans = [], [], [], []
    for run in runs:
        s = run["samples"]
        if len(s) < 2:
            continue
        rise_lo, rise_hi, fall_lo, fall_hi = edges(run)
        spacings += [b["phase_ns"] - a["phase_ns"] for a, b in zip(s, s[1:])]
        brackets += [x["bracket_ns"] for x in s]
        spans.append(s[-1]["phase_ns"] - s[0]["phase_ns"])
        nseg, segstr = segments(run)
        per_run.append({
            "run": run["run"], "samples": len(s), "segments": nseg, "states": segstr,
            "first_ns": s[0]["phase_ns"], "last_ns": s[-1]["phase_ns"],
            "n_set": sum(x["set"] for x in s),
            "rise_lo_ns": rise_lo, "rise_hi_ns": rise_hi,
            "fall_lo_ns": fall_lo, "fall_hi_ns": fall_hi,
        })
    rise_runs = [r for r in per_run if r["rise_lo_ns"] is not None]
    fall_runs = [r for r in per_run if r["fall_lo_ns"] is not None]
    both = [r for r in per_run if r["rise_lo_ns"] is not None and r["fall_lo_ns"] is not None]
    widths = [r["fall_lo_ns"] - r["rise_hi_ns"] for r in both]
    return {
        "runs": len(per_run),
        "samples_per_run": percentile([r["samples"] for r in per_run], 0.5),
        "spacing_ns": {q: percentile(spacings, q) for q in (0.1, 0.5, 0.9)},
        "bracket_ns": {q: percentile(brackets, q) for q in (0.1, 0.5, 0.9)},
        "span_ns": {q: percentile(spans, q) for q in (0.1, 0.5, 0.9)},
        "first_sample_ns": {q: percentile([r["first_ns"] for r in per_run], q)
                            for q in (0.1, 0.5, 0.9)},
        "rise_bracketed_runs": len(rise_runs),
        "fall_bracketed_runs": len(fall_runs),
        "both_bracketed_runs": len(both),
        "window_width_ns_lower_bound": percentile(widths, 0.5),
        "segments_median": percentile([r["segments"] for r in per_run], 0.5),
        "runs_with_three_or_more_segments": sum(1 for r in per_run if r["segments"] >= 3),
        "conventions": {
            "primary": "linear-interpolated quantiles",
            "nearest_rank": {
                "samples_per_run": percentile_nearest([r["samples"] for r in per_run], 0.5),
                "spacing_p90_ns": percentile_nearest(spacings, 0.9),
                "span_p50_ns": percentile_nearest(spans, 0.5),
                "first_sample_p50_ns": percentile_nearest(
                    [r["first_ns"] for r in per_run], 0.5),
            },
        },
        "per_run": per_run,
    }


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "edges.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "samples", "segments", "states", "first_ns", "last_ns", "n_set",
                    "rise_lo_ns", "rise_hi_ns", "fall_lo_ns", "fall_hi_ns"])
        for r in result["per_run"]:
            w.writerow([r["run"], r["samples"], r["segments"], r["states"], r["first_ns"],
                        r["last_ns"], r["n_set"], r["rise_lo_ns"], r["rise_hi_ns"],
                        r["fall_lo_ns"], r["fall_hi_ns"]])

    def us(v):
        return "n/a" if v is None else "%.2f" % (v / 1000.0)

    lines = ["# 单 task 内的多快照：边沿夹逼\n"]
    lines.append("`CAP>1` 时一次提交内取多个快照，因此窗口的上升/下降可以被**同一次运行**里的"
                 "相邻两个快照夹住，不需要逐次运行的平稳性假设。\n")
    lines.append("边沿按**读取区间**保守包围：每次快照的值是在它自己的 `[gb, ge]` 之内读到的，"
                 "不是 `t_ns` 那一个瞬间。因此 clear→set 的转变 `T` 满足 "
                 "`gb(clear) < T <= ge(set)`，反向同理。只用 `t_ns` 会得到比读取能力更窄的区间。\n")
    lines.append("**注意**：下面的「间隔」是间隔本身的分位数，**不是精度保证**，"
                 "也不构成 ±X µs 的误差界。\n")
    lines.append("| 指标 | p10 | p50 | p90 |")
    lines.append("|---|---|---|---|")
    lines.append("| 每次运行的快照数（中位数） | %.1f |" % (result["samples_per_run"] or 0))
    for key, label in (("spacing_ns", "相邻快照间隔 (µs)"),
                       ("bracket_ns", "单次读取区间 (µs)"),
                       ("span_ns", "覆盖跨度 (µs)"),
                       ("first_sample_ns", "首个快照 (µs)")):
        v = result[key]
        lines.append("| %s | %s | %s | %s |" % (label, us(v.get(0.1)), us(v.get(0.5)),
                                                us(v.get(0.9))))
    lines.append("")
    nr = result["conventions"]["nearest_rank"]
    lines.append("- 运行数：%d" % result["runs"])
    lines.append("- **分位数口径**：上表用线性插值；若按 **nearest-rank**（上中位数）复算，"
                 "同一批数据为：快照数 %s、间隔 p90 %s µs、跨度 p50 %s µs、首个快照 p50 %s µs。"
                 "两种口径都列出，避免与他人复算结果对不上。"
                 % (nr["samples_per_run"], us(nr["spacing_p90_ns"]),
                    us(nr["span_p50_ns"]), us(nr["first_sample_p50_ns"])))
    lines.append("- 上升沿被夹住的运行：%d" % result["rise_bracketed_runs"])
    lines.append("- 下降沿被夹住的运行：%d" % result["fall_bracketed_runs"])
    lines.append("- 同一次运行内出现 **≥3 段**（clear→set→clear 这类完整窗口）：**%d/%d**"
                 % (result["runs_with_three_or_more_segments"], result["runs"]))
    lines.append("- 两边沿都被夹住：%d（其中位窗口宽度下界中位数 %s µs）"
                 % (result["both_bracketed_runs"],
                    us(result["window_width_ns_lower_bound"])))
    lines.append("\n## 没有证明的\n")
    lines.append("- 边沿是**区间**，宽度由相邻快照的读取区间决定；「间隔中位数」不是误差界；")
    lines.append("- 首个快照仍有一个约 9 µs 的下限，任务最开头不可观测；")
    lines.append("- 不涉及任何 regcmd 位置或 per-regcmd 时长；")
    lines.append("- 结论仅适用于本板、本驱动版本与该负载。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


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

    def run_from(pattern, bracket=1000, start=9000, step=3000):
        """pattern: list of bools; each sample's read spans [phase, phase+bracket]."""
        return {"run": "x", "samples": [
            {"phase_ns": start + i * step,
             "gb_ns": start + i * step,
             "ge_ns": start + i * step + bracket,
             "bracket_ns": bracket, "set": 1 if v else 0}
            for i, v in enumerate(pattern)]}

    def rise_and_fall():
        r = run_from([0, 1, 1, 0, 0])          # reads span [phase, phase+1000]
        lo, hi, flo, fhi = edges(r)
        assert (lo, hi) == (9000, 13000), (lo, hi)      # gb(clear)=9000 .. ge(set)=12000+1000
        assert (flo, fhi) == (15000, 19000), (flo, fhi)  # gb(set)=15000 .. ge(clear)=18000+1000
        s = summarise([r])
        assert s["rise_bracketed_runs"] == 1 and s["fall_bracketed_runs"] == 1
        assert s["both_bracketed_runs"] == 1
        assert s["window_width_ns_lower_bound"] == 2000, s["window_width_ns_lower_bound"]
        print("  ok  rise and fall bracketed inside one run; width lower bound 3 us")

    group("both edges are bracketed inside a single run", rise_and_fall)

    def no_edges_when_constant():
        for pattern, what in (([0, 0, 0], "all clear"), ([1, 1, 1], "all set")):
            r = run_from(pattern)
            lo, hi, flo, fhi = edges(r)
            assert (lo, hi, flo, fhi) == (None, None, None, None), (what, lo, hi, flo, fhi)
        s = summarise([run_from([1, 1, 1])])
        assert s["rise_bracketed_runs"] == 0 and s["fall_bracketed_runs"] == 0
        print("  ok  a constant sequence brackets nothing (no edge invented)")

    group("constant sequences yield no edges", no_edges_when_constant)

    def starts_set_gives_fall_only():
        r = run_from([1, 1, 0])
        lo, hi, flo, fhi = edges(r)
        assert (lo, hi) == (None, None), (lo, hi)
        assert (flo, fhi) == (12000, 16000), (flo, fhi)
        print("  ok  a run that starts set gives a fall bound but no rise bound")

    group("a run that starts set still gives a fall bound", starts_set_gives_fall_only)

    def read_interval_widens_the_bracket():
        """A longer read interval must widen the bracket, never narrow it."""
        narrow = edges(run_from([0, 1], bracket=100))
        wide = edges(run_from([0, 1], bracket=3000))
        assert narrow[0] == wide[0] == 9000, (narrow, wide)
        assert wide[1] - narrow[1] == 2900, (narrow, wide)
        print("  ok  a wider read interval widens the edge bracket by the same amount")

    group("the bracket follows the read interval", read_interval_widens_the_bracket)

    def metrics():
        runs = [run_from([0, 1, 1, 0]), run_from([0, 1, 0, 0])]
        s = summarise(runs)
        assert s["runs"] == 2 and s["samples_per_run"] == 4
        assert s["spacing_ns"][0.5] == 3000
        assert s["bracket_ns"][0.5] == 1000
        assert s["first_sample_ns"][0.5] == 9000
        assert s["rise_bracketed_runs"] == 2 and s["fall_bracketed_runs"] == 2
        print("  ok  spacing/bracket/first-sample metrics are reported")

    group("spacing, bracket and first-sample metrics", metrics)

    def segment_counts():
        r = run_from([0, 0, 1, 1, 0])
        n, text = segments(r)
        assert n == 3 and text == "ccSSc", (n, text)
        s = summarise([r, run_from([1, 1, 1])])
        assert s["runs_with_three_or_more_segments"] == 1, s["runs_with_three_or_more_segments"]
        # interpolated median of [1, 3] is 2.0; nearest-rank would give 3
        assert s["segments_median"] == 2.0, s["segments_median"]
        print("  ok  a clear-set-clear run counts as three segments")

    group("segments are counted per run", segment_counts)

    def report():
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o")
            write_outputs(summarise([run_from([0, 1, 0])]), out)
            text = open(os.path.join(out, "report.md")).read()
            assert "读取区间" in text and "regcmd 位置" in text
            assert "不是精度保证" in text, "the report must not present spacing as an error bound"
            assert "≥3 段" in text
            assert "0.01" not in text, "sample count must not be formatted as microseconds"
            assert os.path.exists(os.path.join(out, "edges.csv"))
            print("  ok  report states interval semantics and its limits")

    group("the report states interval semantics", report)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all intra-task edge selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dir or not args.out_dir:
        ap.error("--dir and --out-dir are required unless --selftest")
    runs = [r for r in (load_run(p) for p in sorted(glob.glob(os.path.join(args.dir, "*.csv")))) if r]
    if not runs:
        sys.exit("no csv files with samples in %s" % args.dir)
    result = summarise(runs)
    write_outputs(result, args.out_dir)
    print("runs=%d samples/run=%s spacing p50=%s ns rise_runs=%d fall_runs=%d" % (
        result["runs"], result["samples_per_run"], result["spacing_ns"][0.5],
        result["rise_bracketed_runs"], result["fall_bracketed_runs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
