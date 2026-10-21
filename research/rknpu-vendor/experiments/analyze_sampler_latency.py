#!/usr/bin/env python3
"""Judge whether the one-shot sampler can actually land inside a task window.

A phase sweep is only meaningful when the sample lands inside the task it claims
to describe.  The acceptance metrics are therefore NOT "the sample happened at
phi"; they are:

    window coverage   fraction of samples whose read completed before the task's
                      own hw_elapse_time (i.e. inside the execution window)
    lateness          actual_phase - requested_phase, as a distribution
    bracket width     c0_ge_ns - c0_gb_ns, the read's own uncertainty

The tool also breaks the latency into stages from BLOCKWATCH_META:
    arm_observed -> phase_end -> power_begin -> power_end -> first_read
so an optimisation can be attributed to the stage it actually fixed instead of
being applied blindly.

Usage
-----
    python3 analyze_sampler_latency.py --dir results/board_latency_20261021/lat \
        --out-dir results/board_latency_20261021/analysis
    # the variant is taken from the file name prefix before the first "_"
    python3 analyze_sampler_latency.py --selftest
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

META_RE = re.compile(r"BLOCKWATCH_META (?P<body>.*)")
CONFIG_RE = re.compile(r"BLOCKWATCH_CONFIG (?P<body>.*)")
HW_RE = re.compile(r"hw_elapse=(\d+)")
LAT_KEYS = ["lat_arm_observed_ns", "lat_phase_end_ns", "lat_power_begin_ns",
            "lat_power_end_ns", "lat_first_read_ns"]


def parse_kv(text):
    out = {}
    for item in text.split():
        key, _, value = item.partition("=")
        if value:
            out[key] = value
    return out


def load_run(csv_path, log_path, out_path):
    """One run -> its sample metrics and stage timings (None when missing)."""
    run = {"csv": csv_path, "samples": [], "hw_elapse_ns": None, "stage": {}}
    if out_path and os.path.exists(out_path):
        text = open(out_path, errors="replace").read()
        match = HW_RE.search(text)
        if match:
            run["hw_elapse_ns"] = int(match.group(1))
    if log_path and os.path.exists(log_path):
        text = open(log_path, errors="replace").read()
        meta = META_RE.search(text)
        if meta:
            kv = parse_kv(meta.group("body"))
            run["stage"] = {k: int(kv[k]) for k in LAT_KEYS if k in kv}
            run["stop_reason"] = kv.get("stop_reason")
            run["power_checks"] = int(kv.get("power_checks", 0))
        config = CONFIG_RE.search(text)
        if config:
            kv = parse_kv(config.group("body"))
            run["phase_ns"] = int(kv.get("phase_ns", 0))
            run["variant_flags"] = {k: kv.get(k) for k in
                                    ("prefault", "preopen_power", "pin_cpu")}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                submit = int(row.get("submit_enter_ns") or row.get("kick_ns") or 0)
                if submit <= 0:
                    continue
                run["samples"].append({
                    "phase_ns": int(row["t_ns"]) - submit,
                    "bracket_ns": int(row["c0_ge_ns"]) - int(row["c0_gb_ns"]),
                    "validity": (row.get("sample_validity") or "unchecked").strip(),
                    "value": row.get("c0_raw") or row.get("c0_cna"),
                })
    return run


def variant_of(path):
    base = os.path.basename(path)
    return base.split("_")[0]


def summarise(runs):
    """Per-variant aggregate."""
    by_variant = collections.defaultdict(list)
    for run in runs:
        by_variant[run["variant"]].append(run)
    out = {}
    for variant, group in sorted(by_variant.items()):
        phases, lateness, brackets, covered, total = [], [], [], 0, 0
        stage_deltas = collections.defaultdict(list)
        for run in group:
            hw = run["hw_elapse_ns"]
            for sample in run["samples"]:
                total += 1
                phases.append(sample["phase_ns"])
                brackets.append(sample["bracket_ns"])
                if run.get("phase_ns"):
                    lateness.append(sample["phase_ns"] - run["phase_ns"])
                if hw is not None and sample["phase_ns"] <= hw:
                    covered += 1
            stage = run.get("stage") or {}
            if stage.get("lat_arm_observed_ns", -1) >= 0 and \
                    stage.get("lat_phase_end_ns", -1) >= 0:
                stage_deltas["arm_to_phase_end"].append(
                    stage["lat_phase_end_ns"] - stage["lat_arm_observed_ns"])
            if stage.get("lat_power_begin_ns", -1) >= 0 and \
                    stage.get("lat_power_end_ns", -1) >= 0:
                stage_deltas["power_check"].append(
                    stage["lat_power_end_ns"] - stage["lat_power_begin_ns"])
            if stage.get("lat_power_end_ns", -1) >= 0 and \
                    stage.get("lat_first_read_ns", -1) >= 0:
                stage_deltas["power_end_to_read"].append(
                    stage["lat_first_read_ns"] - stage["lat_power_end_ns"])
        entry = {
            "runs": len(group),
            "samples": total,
            "coverage": round(covered / total, 3) if total else 0.0,
            "phase_ns": _stats(phases),
            "lateness_ns": _stats(lateness),
            "bracket_ns": _stats(brackets),
            "stages_ns": {k: _stats(v) for k, v in sorted(stage_deltas.items())},
        }
        out[variant] = entry
    return out


def _stats(values):
    if not values:
        return None
    values = sorted(values)
    return {
        "n": len(values),
        "min": values[0],
        "p50": statistics.median(values),
        "p90": values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))],
        "max": values[-1],
    }


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    lines = ["# 采样时序拆解\n"]
    lines.append("判据不是“样本发生在 φ”，而是：**目标窗口覆盖率**、迟到分布、读取区间宽度。\n")
    lines.append("| 变体 | runs | 样本 | 窗口覆盖率 | 迟到 p50 (µs) | 迟到 p90 (µs) | 区间宽度 p50 (µs) |")
    lines.append("|---|---|---|---|---|---|---|")
    for variant, entry in result["variants"].items():
        late = entry["lateness_ns"]
        bracket = entry["bracket_ns"]
        lines.append("| %s | %d | %d | %.0f%% | %s | %s | %s |" % (
            variant, entry["runs"], entry["samples"], entry["coverage"] * 100,
            "%.1f" % (late["p50"] / 1000) if late else "n/a",
            "%.1f" % (late["p90"] / 1000) if late else "n/a",
            "%.2f" % (bracket["p50"] / 1000) if bracket else "n/a"))
    lines.append("\n## 阶段耗时（µs）\n")
    lines.append("| 变体 | arm→相位结束 p50 | 电源检查 p50 / max | 电源后→首读 p50 |")
    lines.append("|---|---|---|---|")
    for variant, entry in result["variants"].items():
        st = entry["stages_ns"]
        def cell(key, scale=1000.0, field="p50"):
            v = st.get(key)
            return "%.2f" % (v[field] / scale) if v else "n/a"
        lines.append("| %s | %s | %s / %s | %s |" % (
            variant, cell("arm_to_phase_end"),
            cell("power_check"), cell("power_check", field="max"),
            cell("power_end_to_read")))
    lines.append("\n结论按阶段归因：哪一项把延迟降下来了，就记在哪一项上；"
                 "没有降低电源保障的变体才允许采用。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

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
        def write(name, phase_ns, actual_ns, hw, stage, bracket=1000):
            csv_path = os.path.join(tmpdir, name + ".csv")
            with open(csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_ns", "t_end_ns", "c0_ts", "c0_raw", "c0_gb_ns",
                            "c0_ge_ns", "submit_enter_ns", "task_before",
                            "task_after", "sample_validity"])
                base = 1_000_000_000
                w.writerow([base + actual_ns, base + actual_ns + bracket, "0x0", "0x1",
                            base + actual_ns, base + actual_ns + bracket, base, 0, 0, "ok"])
            out_path = os.path.join(tmpdir, name + ".out")
            open(out_path, "w").write("SUBMIT ret=0 hw_elapse=%d ns\n" % hw)
            log_path = os.path.join(tmpdir, name + ".log")
            meta = "BLOCKWATCH_META result=0 samples=1 stop_reason=cap " + " ".join(
                "%s=%d" % (k, v) for k, v in stage.items())
            cfg = "BLOCKWATCH_CONFIG select=5 phase_ns=%d prefault=0 preopen_power=1 pin_cpu=-1\n" % phase_ns
            open(log_path, "w").write(meta + "\n" + cfg)
            return csv_path, log_path, out_path

        def coverage():
            runs = []
            for i in range(4):
                c, l, o = write("base_%d" % i, 1000, 30000, 20000,
                                {"lat_arm_observed_ns": 1000, "lat_phase_end_ns": 2000,
                                 "lat_power_begin_ns": 2100, "lat_power_end_ns": 30000,
                                 "lat_first_read_ns": 30100})
                run = load_run(c, l, o)
                run["variant"] = "base"
                runs.append(run)
            result = summarise(runs)
            assert result["base"]["coverage"] == 0.0, result["base"]
            assert result["base"]["stages_ns"]["power_check"]["p50"] == 27900, result
            print("  ok  a late sample is reported as 0%% coverage")

        group("coverage counts only samples inside the task window", coverage)

        def inside_window():
            runs = []
            for i in range(4):
                c, l, o = write("fast_%d" % i, 1000, 5000, 20000,
                                {"lat_arm_observed_ns": 1000, "lat_phase_end_ns": 2000,
                                 "lat_power_begin_ns": 2100, "lat_power_end_ns": 3000,
                                 "lat_first_read_ns": 3100})
                run = load_run(c, l, o)
                run["variant"] = "fast"
                runs.append(run)
            result = summarise(runs)
            assert result["fast"]["coverage"] == 1.0, result["fast"]
            assert result["fast"]["lateness_ns"]["p50"] == 4000, result["fast"]
            print("  ok  in-window samples give 100%% coverage and a usable lateness")

        group("a fast variant shows up as coverage and lateness", inside_window)

        def stage_attribution():
            runs = []
            for i in range(2):
                c, l, o = write("slow_%d" % i, 1000, 40000, 50000,
                                {"lat_arm_observed_ns": 1000, "lat_phase_end_ns": 2000,
                                 "lat_power_begin_ns": 2100, "lat_power_end_ns": 40000,
                                 "lat_first_read_ns": 40500})
                run = load_run(c, l, o)
                run["variant"] = "slow"
                runs.append(run)
            result = summarise(runs)
            stage = result["slow"]["stages_ns"]
            assert stage["power_check"]["p50"] == 37900, stage
            assert stage["power_end_to_read"]["p50"] == 500, stage
            print("  ok  the dominant stage is attributed to the power check")

        group("stages are attributed to the right phase", stage_attribution)

        def missing_sample():
            csv_path = os.path.join(tmpdir, "empty.csv")
            with open(csv_path, "w") as fh:
                fh.write("t_ns,t_end_ns,c0_ts,c0_raw,c0_gb_ns,c0_ge_ns,"
                         "submit_enter_ns,task_before,task_after,sample_validity\n")
            run = load_run(csv_path, None, None)
            run["variant"] = "empty"
            result = summarise([run])
            assert result["empty"]["samples"] == 0, result
            assert result["empty"]["coverage"] == 0.0
            print("  ok  a run with no samples is counted, not ignored")

        group("runs without samples are visible", missing_sample)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all sampler-latency selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="directory with <variant>_<n>.csv/.log/.out triples")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.dir or not args.out_dir:
        ap.error("--dir and --out-dir are required unless --selftest")

    runs = []
    for csv_path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        stem = csv_path[:-4]
        run = load_run(csv_path, stem + ".log", stem + ".out")
        run["variant"] = variant_of(csv_path)
        runs.append(run)
    if not runs:
        sys.exit("no csv files in %s" % args.dir)
    result = {"dir": args.dir, "runs": len(runs), "variants": summarise(runs)}
    write_outputs(result, args.out_dir)
    print("wrote %s" % args.out_dir)
    for variant, entry in result["variants"].items():
        late = entry["lateness_ns"]
        print("  %-10s runs=%-3d samples=%-3d coverage=%.0f%%  lateness p50=%s"
              % (variant, entry["runs"], entry["samples"], entry["coverage"] * 100,
                 ("%.1f us" % (late["p50"] / 1000)) if late else "n/a"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
