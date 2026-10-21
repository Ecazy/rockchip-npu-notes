#!/usr/bin/env python3
"""DPU group window in INTERRUPT_STATUS, measured with one snapshot per run.

What the bits are, as far as this board shows (2026-10-21, single-task min_conv_exp)
-----------------------------------------------------------------------------------
* INTERRUPT_STATUS bits 8/9 (DPU_0 / DPU_1) follow the ping-pong group of the job:
  over the set samples DPU_0 occurs on even submit indices and DPU_1 on odd ones,
  with no exceptions, across process boundaries.
* The bits are NOT a one-off completion pulse.  They are set for most of the job and
  cleared by the driver's IRQ path (INT_CLEAR = 0x1ffff), so they behave like an
  in-flight flag for the group.
* The FALLING edge follows the driver's hw_elapse_time: with the phase measured from
  the submit entry, one threshold on (phase - hw_elapse) separates set from unset far
  better than one threshold on phase alone.  That is the signature of a clear at the
  end of the job, not of a fixed-time event.  The threshold is fitted in-sample.
* The RISING edge (roughly 11-18 us after the submit entry) is not explained.

Anchors, all CLOCK_MONOTONIC_RAW (the same clock as the sample)
---------------------------------------------------------------
    submit_enter   entry to the submit path; the phase origin
    ioctl_return   return of the real ioctl (recorded by the hook)

Scope limits (repeated in every report)
---------------------------------------
* one snapshot per run: p(set | phase) is an estimate, not a per-run event time;
* a run is dropped unless the workload printed BIT-EXACT; nothing is repaired;
* task_crossed samples are dropped; task_ambiguous samples are kept because a
  one-task job keeps the task counter at 0 both while running and when idle;
* a paired timing comparison is only read if it is balanced against the ping-pong
  parity (ABBA blocks).  Unbalanced pairs are not a perturbation estimate.

Usage
-----
    python3 analyze_irq_completion.py --sweep fp16=DIR_A --sweep iw32=DIR_B \\
        --pairs abba.csv --out-dir OUT
    python3 analyze_irq_completion.py --selftest
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import re
import statistics
import sys

DONE_MASK = 0x3FFF                       # INTERRUPT_STATUS bits 0..13
BIT_NAMES = {0: "CNA_FEATURE_0", 1: "CNA_FEATURE_1", 2: "CNA_WEIGHT_0",
             3: "CNA_WEIGHT_1", 4: "CNA_CSC_0", 5: "CNA_CSC_1", 6: "CORE_0",
             7: "CORE_1", 8: "DPU_0", 9: "DPU_1", 10: "PPU_0", 11: "PPU_1",
             12: "DMA_READ_ERROR", 13: "DMA_WRITE_ERROR"}
KEPT_VALIDITY = {"ok", "unchecked", "task_ambiguous"}
META_RE = re.compile(r"BLOCKWATCH_META (.*)")
HW_RE = re.compile(r"hw_elapse=(\d+)")
RESULT_RE = re.compile(r"RESULT: (.+)")


def first_group(regex, text):
    match = regex.search(text)
    return match.group(1).strip() if match else ""


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as fh:
        return fh.read()


def parse_kv(text):
    out = {}
    for item in text.split():
        key, sep, value = item.partition("=")
        if sep:
            out[key] = value
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def load_sweep(label, sweep_dir, drops):
    """Records for one load.  Every dropped run is counted under its reason."""
    manifest = os.path.join(sweep_dir, "manifest.csv")
    if not os.path.exists(manifest):
        sys.exit("no manifest in %s" % sweep_dir)
    records = []
    with open(manifest, newline="") as fh:
        for index, run in enumerate(csv.DictReader(fh)):
            stem = os.path.splitext(os.path.basename(run["csv"]))[0]
            csv_path = os.path.join(sweep_dir, "runs", stem + ".csv")
            rows = []
            if os.path.exists(csv_path):
                with open(csv_path, newline="") as fh2:
                    rows = list(csv.DictReader(fh2))
            if not rows:
                drops[label]["no_sample (phase after the return)"] += 1
                continue
            row = rows[0]
            meta = parse_kv(first_group(META_RE,
                                        read_text(os.path.join(sweep_dir, "meta", stem + ".txt"))))
            out_text = read_text(os.path.join(sweep_dir, "out", stem + ".txt"))
            validity = (row.get("sample_validity") or "").strip()
            submit = int(row.get("submit_enter_ns") or 0)
            if submit <= 0:
                drops[label]["no_submit_anchor"] += 1
                continue
            if validity not in KEPT_VALIDITY:
                drops[label]["validity:%s" % (validity or "none")] += 1
                continue
            if not first_group(RESULT_RE, out_text).startswith("BIT-EXACT"):
                drops[label]["not_bit_exact"] += 1
                continue
            ret_rel = int(meta.get("lat_ioctl_return_ns", "-1"))
            if ret_rel <= 0:
                drops[label]["no_return_anchor"] += 1
                continue
            hw = HW_RE.search(out_text)
            if not hw:
                drops[label]["no_hw_elapse"] += 1
                continue
            records.append({
                "run": stem,
                "label": label,
                "index": index,
                "request_ns": int(run["request_phase_ns"]),
                "phase_ns": int(row["t_ns"]) - submit,
                "status": int(row["c0_raw"], 16),
                "raw": int(row["c0_raw2"], 16) if row.get("c0_raw2") else None,
                "counter": int(row["c0_ts"], 16) & 0xFFF,
                "validity": validity,
                "ret_rel": ret_rel,
                "hw_ns": int(hw.group(1)),
            })
            drops[label]["kept"] += 1
    return records


def edge_model(records, min_phase_ns=20000):
    """Which single threshold separates set from unset best?

    Only samples after the rising edge are used, so the question is about the
    falling edge.  Two keys are compared: the phase itself, and phase - hw_elapse.
    Accuracy is in-sample: the threshold is fitted on the same data, so it is an
    upper bound on what a fixed threshold would achieve."""
    pts = [(r["phase_ns"], r["hw_ns"], 1 if r["status"] & DONE_MASK else 0)
           for r in records if r["phase_ns"] >= min_phase_ns]
    if len(pts) < 10:
        return {"n": len(pts), "status": "insufficient_samples"}

    def best(key):
        values = sorted(set(key(p) for p in pts))
        top = (0.0, None, None)
        for t in values:
            for rule, pred in (("set iff key < t", lambda k, t=t: k < t),
                               ("set iff key >= t", lambda k, t=t: k >= t)):
                acc = sum(1 for p in pts if pred(key(p)) == bool(p[2])) / len(pts)
                if acc > top[0]:
                    top = (acc, rule, t)
        return {"accuracy": round(top[0], 3), "rule": top[1], "threshold_ns": top[2]}

    return {"n": len(pts), "status": "ok",
            "phase_key": best(lambda p: p[0]),
            "phase_minus_hw_key": best(lambda p: p[0] - p[1])}


def summarise(records, bin_ns):
    if not records:
        return {"n": 0, "classification": "no_data"}
    set_recs = [r for r in records if r["status"] & DONE_MASK]
    before = [r for r in records if r["phase_ns"] < r["ret_rel"]]
    after = [r for r in records if r["phase_ns"] >= r["ret_rel"]]

    def frac_set(group):
        if not group:
            return None
        return sum(1 for r in group if r["status"] & DONE_MASK) / len(group)

    p_before, p_after = frac_set(before), frac_set(after)
    if not set_recs:
        cls = "no_window_signal"
    elif p_after is None:
        cls = "no_samples_after_return"
    elif p_after == 0 and p_before:
        cls = "cleared_by_return"
    elif p_after >= 0.5:
        cls = "latched_past_return"
    else:
        cls = "mixed_after_return"

    bins = collections.defaultdict(list)
    for r in records:
        bins[r["phase_ns"] // bin_ns].append(r)
    table, rising = [], None
    for b in sorted(bins):
        grp = bins[b]
        k = sum(1 for r in grp if r["status"] & DONE_MASK)
        p, lo, hi = wilson(k, len(grp))
        centre = b * bin_ns + bin_ns // 2
        table.append({"bin_centre_ns": centre, "n": len(grp), "k": k,
                      "p": round(p, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                      "after_return": sum(1 for r in grp if r["phase_ns"] >= r["ret_rel"])})
        if rising is None and p >= 0.5:
            rising = centre

    bits = collections.Counter()
    values = collections.Counter()
    for r in set_recs:
        v = r["status"] & DONE_MASK
        values["0x%04x" % v] += 1
        for bit, name in BIT_NAMES.items():
            if (v >> bit) & 1:
                bits[name] += 1
    raws = collections.Counter("0x%08x" % r["raw"] for r in records if r["raw"] is not None)

    parity = collections.Counter()
    for r in set_recs:
        if r["status"] & 0x100:
            parity[("DPU_0", r["index"] % 2)] += 1
        if r["status"] & 0x200:
            parity[("DPU_1", r["index"] % 2)] += 1

    returns = [r["ret_rel"] for r in records]
    hws = [r["hw_ns"] for r in records]
    return {
        "n": len(records),
        "classification": cls,
        "p_set_before_return": None if p_before is None else round(p_before, 3),
        "p_set_after_return": None if p_after is None else round(p_after, 3),
        "n_before_return": len(before),
        "n_after_return": len(after),
        "rising_edge_bin_centre_ns": rising,
        "ioctl_return_ns": {"p10": percentile(returns, 0.1), "p50": percentile(returns, 0.5),
                            "p90": percentile(returns, 0.9)},
        "hw_elapse_ns": {"p10": percentile(hws, 0.1), "p50": percentile(hws, 0.5),
                         "p90": percentile(hws, 0.9)},
        "bit_occurrence": dict(bits.most_common()),
        "status_values": dict(values.most_common(8)),
        "raw_values": dict(raws.most_common(8)),
        "dpu_bit_by_submit_parity": {"%s/parity%d" % k: v for k, v in sorted(parity.items())},
        "edge_model": edge_model(records),
        "bins": table,
    }


def paired_blocks(rows):
    """rows: (block, arm, hw_ns) from ABBA blocks.  Each block gives
    mean(sampled) - mean(plain); the ping-pong parity cancels inside a block because
    each arm holds one even and one odd submit.  The interval is a bootstrap over
    blocks.  Fewer than 5 blocks is not a result."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for block, arm, hw in rows:
        by[block][arm].append(hw)
    diffs = [statistics.mean(by[b]["sampled"]) - statistics.mean(by[b]["plain"])
             for b in sorted(by) if by[b]["sampled"] and by[b]["plain"]]
    if len(diffs) < 5:
        return {"design": "ABBA blocks", "n_blocks": len(diffs),
                "status": "insufficient_blocks"}
    rnd = random.Random(20261022)
    boots = sorted(statistics.median(rnd.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"design": "ABBA blocks", "n_blocks": len(diffs), "status": "ok",
            "median_diff_ns": statistics.median(diffs),
            "mean_diff_ns": statistics.mean(diffs),
            "ci95_median_ns": [boots[int(0.025 * (len(boots) - 1))],
                               boots[int(0.975 * (len(boots) - 1))]]}


def paired_pairs(rows):
    """rows: (pair_id, arm, hw_ns) from two-by-two pairs.  Only valid when the arms
    are balanced against the ping-pong parity; otherwise the result is confounded."""
    by = collections.defaultdict(dict)
    for pid, arm, hw in rows:
        by[pid][arm] = hw
    diffs = [by[p]["sampled"] - by[p]["plain"] for p in sorted(by)
             if "sampled" in by[p] and "plain" in by[p]]
    if len(diffs) < 5:
        return {"design": "two-by-two pairs", "n_pairs": len(diffs),
                "status": "insufficient_pairs"}
    rnd = random.Random(20261021)
    boots = sorted(statistics.median(rnd.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"design": "two-by-two pairs", "n_pairs": len(diffs), "status": "ok",
            "median_diff_ns": statistics.median(diffs),
            "mean_diff_ns": statistics.mean(diffs),
            "ci95_median_ns": [boots[int(0.025 * (len(boots) - 1))],
                               boots[int(0.975 * (len(boots) - 1))]]}


def load_pair_rows(path):
    """ABBA files carry a 'block' column; plain pair files carry 'pair_id'."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    if "block" in header:
        kept = [(r["block"], r["arm"], int(r["hw_ns"])) for r in rows
                if r.get("ok", "1") == "1" and int(r["hw_ns"]) > 0]
        return "abba", kept
    return "pairs", [(r["pair_id"], r["arm"], int(r["hw_ns"])) for r in rows]


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "bins.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["load", "bin_centre_ns", "n", "k", "p", "ci_lo", "ci_hi", "after_return"])
        for label, block in result["loads"].items():
            for row in block["summary"]["bins"]:
                w.writerow([label, row["bin_centre_ns"], row["n"], row["k"], row["p"],
                            row["ci_lo"], row["ci_hi"], row["after_return"]])

    def us(value):
        return "n/a" if value is None else "%.1f" % (value / 1000.0)

    lines = ["# DPU 组置位窗口与驱动时间锚点（板端，单快照）\n"]
    lines.append("INTERRUPT_STATUS(0x0028) 与 RAW_STATUS(0x002c) 每个 run 各采一次，"
                 "相位从提交入口起算；ioctl 返回与采样同一时钟（CLOCK_MONOTONIC_RAW）。\n")
    lines.append("**读法（已修正）**：置位位 DPU_0/DPU_1 不是单次“块完成”脉冲，而是在作业大部分"
                 "时间内置位、由 IRQ 路径清除的组级标记。下降沿与驱动的 `hw_elapse_time` 对齐；"
                 "上升沿（约 11–18 µs）的机理尚未解释。\n")
    lines.append("**限定**：每个 run 只有一个快照，p(置位 | φ) 是估计；只统计 workload 打印 "
                 "BIT-EXACT 的 run；task_crossed 丢弃，task_ambiguous 保留。\n")
    for label, block in result["loads"].items():
        s = block["summary"]
        lines.append("\n## 负载 `%s`\n" % label)
        drops = ", ".join("%s×%d" % kv for kv in sorted(block["drops"].items())) or "无"
        lines.append("- 保留 %d 个 run；丢弃：%s" % (s["n"], drops))
        if not s["n"]:
            lines.append("- 无可用数据")
            continue
        lines.append("- hw_elapse 中位数 %s µs（p10–p90 %s–%s）" % (
            us(s["hw_elapse_ns"]["p50"]), us(s["hw_elapse_ns"]["p10"]),
            us(s["hw_elapse_ns"]["p90"])))
        lines.append("- ioctl 返回（提交入口相对）中位数 %s µs（p10–p90 %s–%s）" % (
            us(s["ioctl_return_ns"]["p50"]), us(s["ioctl_return_ns"]["p10"]),
            us(s["ioctl_return_ns"]["p90"])))
        lines.append("- 分类：**%s**；返回前置位比例 %s，返回后 %s（返回后样本 %d 个）" % (
            s["classification"], s["p_set_before_return"], s["p_set_after_return"],
            s["n_after_return"]))
        lines.append("- 上升沿（p ≥ 0.5 的相位区间中心，机理未解释）：%s µs"
                     % us(s["rising_edge_bin_centre_ns"]))
        lines.append("- 置位位名计数：%s" % (s["bit_occurrence"] or "无"))
        lines.append("- DPU 位与提交序号奇偶：%s" % (s["dpu_bit_by_submit_parity"] or "无"))
        lines.append("- 状态字取值（仅置位样本）：%s" % (s["status_values"] or "无"))
        lines.append("- RAW_STATUS 取值：%s" % (s["raw_values"] or "无"))
        em = s["edge_model"]
        if em.get("status") == "ok":
            lines.append("- 下降沿模型（φ ≥ 20 µs，样本内阈值）：只用 φ 的最佳准确率 %.3f"
                         "（%s）；只用 φ − hw_elapse 的最佳准确率 %.3f（%s，阈值 %s µs）" % (
                             em["phase_key"]["accuracy"], em["phase_key"]["rule"],
                             em["phase_minus_hw_key"]["accuracy"],
                             em["phase_minus_hw_key"]["rule"],
                             us(em["phase_minus_hw_key"]["threshold_ns"])))
        lines.append("\n| 相位中心 (µs) | n | 置位 | p | 95% CI | 返回后样本 |")
        lines.append("|---|---|---|---|---|---|")
        for row in s["bins"]:
            lines.append("| %.1f | %d | %d | %.2f | %.2f–%.2f | %d |" % (
                row["bin_centre_ns"] / 1000.0, row["n"], row["k"], row["p"],
                row["ci_lo"], row["ci_hi"], row["after_return"]))
    if result.get("paired"):
        p = result["paired"]
        lines.append("\n## 配对计时（%s，采样 − 不采样）\n" % p.get("design"))
        if p["status"] != "ok":
            lines.append("区组数不足以给出区间。")
        else:
            lines.append("区组数 %d；区组内均值差的中位数 %.1f µs，95%% 区间 [%.1f, %.1f] µs；"
                         "均值 %.1f µs。ABBA 设计使乒乓组奇偶在两臂中相互抵消。" % (
                             p["n_blocks"], p["median_diff_ns"] / 1000.0,
                             p["ci95_median_ns"][0] / 1000.0,
                             p["ci95_median_ns"][1] / 1000.0,
                             p["mean_diff_ns"] / 1000.0))
    lines.append("\n## 本报告没有证明的\n")
    lines.append("- 置位窗口不是单个块的完成时刻，也不给出任何 regcmd 位置或 task 内时刻；")
    lines.append("- 上升沿的机理未知，不能当作“启动”或“完成”；")
    lines.append("- 下降沿与 hw_elapse 对齐，是 IRQ 清除的间接证据；IRQ 时刻本身未直接观测；")
    lines.append("- submit→kick 偏移未标定；由下降沿反推只能给出估计，且阈值是样本内拟合；")
    lines.append("- 初版两两配对（未做奇偶平衡）的结果已撤回，不作为扰动估计；")
    lines.append("- 数字仅适用于本板、本驱动版本与本负载。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


def analyse(sweeps, pairs_path, bin_ns):
    drops = collections.defaultdict(collections.Counter)
    loads = {}
    for label, sweep_dir in sweeps:
        records = load_sweep(label, sweep_dir, drops)
        loads[label] = {"drops": dict(drops[label]), "summary": summarise(records, bin_ns)}
    result = {"bin_ns": bin_ns, "loads": loads,
              "scope": "one snapshot per run; p(set | phase) is an estimate"}
    if pairs_path:
        kind, rows = load_pair_rows(pairs_path)
        result["paired"] = paired_blocks(rows) if kind == "abba" else paired_pairs(rows)
    return result


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _make_sweep(root, specs):
    """specs: dicts with request, actual, status, ret, hw, validity, result, raw."""
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    os.makedirs(os.path.join(root, "out"), exist_ok=True)
    submit = 1_000_000_000
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["run_id", "request_phase_ns", "csv", "condition"])
        for i, spec in enumerate(specs):
            stem = "r%04d" % i
            w.writerow([stem, spec["request"], os.path.join("/elsewhere", stem + ".csv"), ""])
            header = ("t_ns,t_end_ns,c0_ts,c0_raw,c0_raw2,c0_gb_ns,c0_ge_ns,"
                      "submit_enter_ns,task_before,task_after,sample_validity\n")
            if spec.get("no_sample"):
                open(os.path.join(root, "runs", stem + ".csv"), "w").write(header)
            else:
                with open(os.path.join(root, "runs", stem + ".csv"), "w", newline="") as fh:
                    fh.write(header)
                    cw = csv.writer(fh)
                    t = submit + spec["actual"]
                    cw.writerow([t, t + 400, "0x00000000", "0x%08x" % spec["status"],
                                 "0x%08x" % spec.get("raw", 0), t, t + 400, submit, 0, 0,
                                 spec["validity"]])
            with open(os.path.join(root, "meta", stem + ".txt"), "w") as fh:
                fh.write("BLOCKWATCH_META result=0 samples=%d stop_reason=cap "
                         "lat_ioctl_start_ns=2000 lat_ioctl_return_ns=%d\n"
                         % (0 if spec.get("no_sample") else 1, spec["ret"]))
            with open(os.path.join(root, "out", stem + ".txt"), "w") as fh:
                fh.write("SUBMIT ret=0 hw_elapse=%d ns\n%s\n" % (spec["hw"], spec["result"]))


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

    grid = [4000 * i for i in range(31)]        # 0..120 us on a 4 us grid

    def spec(request, status, ret=60000, validity="ok",
             result="RESULT: BIT-EXACT (fp16)", hw=50000, raw=0, no_sample=False):
        return {"request": request, "actual": request, "status": status, "ret": ret,
                "hw": hw, "validity": validity, "result": result, "raw": raw,
                "no_sample": no_sample}

    def window_specs(start, end_offset, ret=110000):
        """set iff start <= phase < hw + end_offset; hw varies per run."""
        out = []
        for i, p in enumerate(grid):
            hw = 30000 + (i * 7000) % 70000
            on = start <= p < hw + end_offset
            out.append(spec(p, 0x300 if on else 0, ret=ret, hw=hw))
        return out

    with tempfile.TemporaryDirectory() as tmp:
        def window_closes_at_hw():
            _make_sweep(os.path.join(tmp, "w"), window_specs(12000, 4000))
            s = analyse([("w", os.path.join(tmp, "w"))], None, 4000)["loads"]["w"]["summary"]
            assert s["classification"] == "cleared_by_return", s["classification"]
            em = s["edge_model"]
            assert em["status"] == "ok", em
            assert em["phase_minus_hw_key"]["accuracy"] == 1.0, em
            assert em["phase_key"]["accuracy"] < 0.95, em
            assert 3000 <= em["phase_minus_hw_key"]["threshold_ns"] <= 5000, em
            print("  ok  falling edge follows hw (acc %.2f) not phase (acc %.2f)"
                  % (em["phase_minus_hw_key"]["accuracy"], em["phase_key"]["accuracy"]))

        group("the falling edge is anchored to hw_elapse, not to the phase",
              window_closes_at_hw)

        def latched():
            _make_sweep(os.path.join(tmp, "l"),
                        [spec(p, 0x300 if p >= 12000 else 0) for p in grid])
            s = analyse([("l", os.path.join(tmp, "l"))], None, 4000)["loads"]["l"]["summary"]
            assert s["classification"] == "latched_past_return", s["classification"]
            print("  ok  still set after the return -> latched_past_return")

        group("a bit still set after the return is latched", latched)

        def none():
            _make_sweep(os.path.join(tmp, "n"), [spec(p, 0) for p in grid])
            s = analyse([("n", os.path.join(tmp, "n"))], None, 4000)["loads"]["n"]["summary"]
            assert s["classification"] == "no_window_signal", s["classification"]
            assert s["rising_edge_bin_centre_ns"] is None
            print("  ok  never set -> no_window_signal, no rising edge invented")

        group("no window signal is reported as such", none)

        def filters():
            specs = [spec(p, 0x300 if p >= 12000 else 0) for p in grid]
            specs.append(spec(10000, 0x300, result="RESULT: MISMATCH"))
            specs.append(spec(10000, 0x300, validity="task_crossed"))
            specs.append(spec(10000, 0x300, ret=-1))
            specs.append(spec(10000, 0x300, validity="task_ambiguous"))
            specs.append(spec(10000, 0, no_sample=True))
            _make_sweep(os.path.join(tmp, "f"), specs)
            block = analyse([("f", os.path.join(tmp, "f"))], None, 4000)["loads"]["f"]
            drops = block["drops"]
            assert drops.get("not_bit_exact") == 1, drops
            assert drops.get("validity:task_crossed") == 1, drops
            assert drops.get("no_return_anchor") == 1, drops
            assert any(k.startswith("no_sample") for k in drops), drops
            assert block["summary"]["n"] == len(grid) + 1, block["summary"]["n"]
            print("  ok  MISMATCH, task_crossed, missing anchor and no-sample runs dropped")

        group("runs that do not qualify are dropped, not repaired", filters)

        def two_loads():
            _make_sweep(os.path.join(tmp, "a"),
                        [spec(p, 0x300 if p >= 40000 else 0) for p in grid])
            _make_sweep(os.path.join(tmp, "b"),
                        [spec(p, 0x300 if p >= 10000 else 0) for p in grid])
            r = analyse([("a", os.path.join(tmp, "a")), ("b", os.path.join(tmp, "b"))],
                        None, 4000)
            assert r["loads"]["a"]["summary"]["rising_edge_bin_centre_ns"] == 42000
            assert r["loads"]["b"]["summary"]["rising_edge_bin_centre_ns"] == 14000
            print("  ok  two loads keep their own rising edges (42 us vs 14 us)")

        group("loads are analysed separately", two_loads)

        def parity_table():
            specs = []
            for i, p in enumerate(grid):
                if p < 12000:
                    specs.append(spec(p, 0))
                else:
                    specs.append(spec(p, 0x100 if i % 2 == 0 else 0x200))
            _make_sweep(os.path.join(tmp, "p"), specs)
            s = analyse([("p", os.path.join(tmp, "p"))], None, 4000)["loads"]["p"]["summary"]
            table = s["dpu_bit_by_submit_parity"]
            assert "DPU_0/parity0" in table and "DPU_1/parity1" in table, table
            assert "DPU_0/parity1" not in table and "DPU_1/parity0" not in table, table
            print("  ok  the DPU bit is tabulated against the submit parity")

        group("the DPU bit is tabulated against the submit parity", parity_table)

        def abba_blocks():
            rows = []
            for b in range(1, 7):
                rows += [(str(b), "sampled", 63000 + b), (str(b), "plain", 60000 + b),
                         (str(b), "plain", 60001 + b), (str(b), "sampled", 63001 + b)]
            p = paired_blocks(rows)
            assert p["status"] == "ok" and p["n_blocks"] == 6, p
            assert p["median_diff_ns"] == 3000.0, p
            few = paired_blocks(rows[:8])
            assert few["status"] == "insufficient_blocks", few
            print("  ok  ABBA block difference 3 us (median), fewer than 5 blocks refused")

        group("ABBA blocks give a median difference and an interval", abba_blocks)

        def pairs_file_kind():
            path = os.path.join(tmp, "pairs.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["pair_id", "arm", "hw_ns"])
                for i in range(6):
                    w.writerow(["p%d" % i, "sampled", 5000 + i])
                    w.writerow(["p%d" % i, "plain", i])
            kind, rows = load_pair_rows(path)
            assert kind == "pairs" and len(rows) == 12, (kind, len(rows))
            res = paired_pairs(rows)
            assert res["design"] == "two-by-two pairs" and res["status"] == "ok", res
            print("  ok  plain pair files are read as two-by-two pairs and labelled as such")

        group("the two designs are labelled, not mixed", pairs_file_kind)

        def report_written():
            _make_sweep(os.path.join(tmp, "rep"), window_specs(12000, 4000))
            out = os.path.join(tmp, "out")
            write_outputs(analyse([("rep", os.path.join(tmp, "rep"))], None, 4000), out)
            text = open(os.path.join(out, "report.md")).read()
            assert "只统计 workload 打印" in text and "regcmd 位置" in text, text[:200]
            assert "不是单个块的完成时刻" in text and "上升沿的机理未知" in text
            assert os.path.exists(os.path.join(out, "bins.csv"))
            print("  ok  report states that the bit is a window, not a block completion")

        group("the report states its limits", report_written)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all irq-completion selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--pairs", default=None,
                    help="ABBA block file (block,pos,arm,index,hw_ns,ok) or a pairs file")
    ap.add_argument("--bin-us", type=float, default=4.0)
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.sweep or not args.out_dir:
        ap.error("--sweep (repeatable) and --out-dir are required unless --selftest")
    sweeps = []
    for spec in args.sweep:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            ap.error("--sweep expects LABEL=DIR, got %r" % spec)
        sweeps.append((label, path))
    result = analyse(sweeps, args.pairs, int(args.bin_us * 1000))
    write_outputs(result, args.out_dir)
    for label, block in result["loads"].items():
        s = block["summary"]
        rising = s.get("rising_edge_bin_centre_ns")
        print("%-8s n=%-4d class=%-22s rising=%s" % (
            label, s["n"], s["classification"],
            "n/a" if rising is None else "%.1f us" % (rising / 1000.0)))
    if result.get("paired"):
        p = result["paired"]
        print("paired   %s status=%s" % (p.get("design"), p.get("status")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
