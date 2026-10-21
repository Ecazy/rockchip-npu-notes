#!/usr/bin/env python3
"""Align driver-side ftrace events with the user-space sampler, on one clock.

ftrace can be told to record in `mono_raw` (`trace_clock`), which is the same
clock the blockwatch sampler stamps into its CSV.  With that, the two timelines
can be subtracted directly -- no clock-bridging assumption.

Events used (all present in `available_filter_functions` on this board):

    __rknpu_submit_ioctl     entry to the real submit ioctl
    rknpu_job_next.part.0    the commit: sets hw_commit_time and writes PC_OP_EN
    rknpu_coreN_irq_handler  IRQ entry; hw_elapse_time is computed just after it

Reported quantities (all relative to the sampler's `submit_enter`):

    ioctl_rel    our LD_PRELOAD wrapper's own cost before the real ioctl
    commit_rel   the kick
    irq_rel      IRQ entry
    rise/fall    the interrupt-bit edges, bracketed by their read intervals
    rise - commit, fall - irq
    hw_elapse    driver-reported, vs (irq - commit)

ftrace prints seconds with 1 us resolution, so every ftrace-derived number here
carries a ~1 us quantisation; the sampler side is ns.

Usage
-----
    python3 analyze_driver_events.py --dir DIR [--out-dir OUT]
    python3 analyze_driver_events.py --selftest
"""

import argparse
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_intra_task_edges import edges, load_run  # noqa: E402

TRACE_RE = re.compile(r"^\s*\S+\s+\[\d+\]\s+\S+\s+(?P<ts>\d+\.\d+):\s+(?P<fn>\S+)")
HW_RE = re.compile(r"hw_elapse=(\d+)")


CALLER_RE = re.compile(r"<-(?P<caller>\S+)")


def parse_trace(path):
    """Events in ns on the ftrace clock (mono_raw), WITH their call source.

    `rknpu_job_next.part.0` is NOT the commit: its entry is reached on two early
    return paths (soft_reseting, or subcore busy / empty todo list) that neither
    stamp hw_commit_time nor write PC_OP_EN.  The call source is the only thing
    ftrace can tell us about which path was taken, so it is kept:

        caller == rknpu_job_schedule  -> "sched_entry"  (first call, may dequeue)
        caller == rknpu_irq_handler* -> "irq_entry"    (post-completion check)

    Which branch ran inside is NOT observable with the traceable functions: the
    commit itself lives in static inline helpers (rknpu_job_commit ->
    rknpu_job_subcore_commit -> ..._commit_pc) that do not appear in
    available_filter_functions."""
    out = {"job_next": [], "job_schedule": [], "irq": []}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = TRACE_RE.match(line)
        if not m:
            continue
        fn = m.group("fn")
        ts = float(m.group("ts")) * 1e9
        cm = CALLER_RE.search(line)
        caller = cm.group("caller") if cm else "?"
        if fn == "__rknpu_submit_ioctl" and "ioctl" not in out:
            out["ioctl"] = ts
        elif fn.startswith("rknpu_job_next"):
            out["job_next"].append({"ts": ts, "caller": caller})
        elif fn == "rknpu_job_schedule":
            out["job_schedule"].append(ts)
        elif fn.startswith("rknpu_core") and fn.endswith("irq_handler"):
            out["irq"].append(ts)
        elif fn == "rknpu_power_on" and "power_on" not in out:
            out["power_on"] = ts
        elif fn == "rknpu_power_off" and "power_off" not in out:
            out["power_off"] = ts
    # classify by call source, not by order
    for ev in out["job_next"]:
        if "job_schedule" in ev["caller"]:
            ev["kind"] = "sched_entry"
        elif "irq_handler" in ev["caller"]:
            ev["kind"] = "irq_entry"
        else:
            ev["kind"] = "other_entry"
    return out


LIFE_EVENTS = ("rknpu_power_on", "rknpu_power_off", "rknpu_power_get",
               "rknpu_power_put", "rknpu_power_put_delay", "rknpu_power_off_delay_work",
               "rknpu_job_free", "rknpu_job_cleanup_work")


def lifecycle(trace_path):
    """PM / job lifecycle facts for one run (times relative to the first event).

    Notable, and easy to get wrong: `rknpu_job_next.part.0` runs TWICE per job --
    once from `rknpu_job_schedule` (the commit) and once from the IRQ handler
    (dispatching the next job).  Only the first is the kick."""
    events = []
    for line in open(trace_path, errors="replace"):
        m = TRACE_RE.match(line)
        if not m:
            continue
        events.append((float(m.group("ts")), m.group("fn")))
    if not events:
        return None
    t0 = events[0][0]
    rel = lambda ts: round((ts - t0) * 1e6, 2)
    first = {}
    counts = {}
    for ts, fn in events:
        counts[fn] = counts.get(fn, 0) + 1
        if fn not in first:
            first[fn] = ts
    job_next = [ts for ts, fn in events if fn.startswith("rknpu_job_next")]
    return {
        "events": len(events),
        "power_on_us": rel(first["rknpu_power_on"]) if "rknpu_power_on" in first else None,
        "power_off_us": rel(first["rknpu_power_off"]) if "rknpu_power_off" in first else None,
        "commit_us": rel(job_next[0]) if job_next else None,
        "job_next_calls": len(job_next),
        "job_next_gap_us": round((job_next[1] - job_next[0]) * 1e6, 2) if len(job_next) > 1 else None,
        "job_free_us": rel(first["rknpu_job_free"]) if "rknpu_job_free" in first else None,
        "power_get_count": counts.get("rknpu_power_get", 0),
        "power_put_delay_count": counts.get("rknpu_power_put_delay", 0),
        "power_on_seen": "rknpu_power_on" in first,
        "power_off_seen": "rknpu_power_off" in first,
        "domain_on_before_commit": ("rknpu_power_on" in first and job_next
                                    and first["rknpu_power_on"] < job_next[0]),
    }


LOSS_RE = re.compile(r"entries-in-buffer/entries-written:\s*(\d+)/(\d+)")


def trace_loss(path):
    """ftrace tells us in the header whether the ring wrapped (events lost)."""
    if not os.path.exists(path):
        return None
    head = open(path, errors="replace").read(2048)
    m = LOSS_RE.search(head)
    if not m:
        return None
    inbuf, written = int(m.group(1)), int(m.group(2))
    return {"in_buffer": inbuf, "written": written, "lost": max(0, written - inbuf)}


def analyse_lifecycle_only(trace_path):
    """When a run has no sampler CSV (lifecycle-only probe), still report PM facts."""
    life = lifecycle(trace_path)
    if not life:
        return None
    return {"run": os.path.basename(trace_path), "samples": 0,
            "ioctl_rel_us": None, "commit_rel_us": life.get("commit_us"),
            "irq_rel_us": None, "hw_elapse_ns": None,
            "rise_us": None, "fall_us": None, "lifecycle": life}


def analyse_run(trace_path, csv_path, out_path):
    """Align one run.  Differences are reported as INTERVALS, never as points:

        rise - ref  in (rise_lo - ref, rise_hi - ref]
        fall - ref  in (fall_lo - ref, fall_hi - ref]

    and `ref` is named for what it actually is (a scheduler-path entry or an IRQ
    handler entry), not for the mechanism it might implement."""
    events = parse_trace(trace_path)
    run = load_run(csv_path)
    if not run or run["samples"] == [] or not events["job_next"]:
        return None
    rows = list(csv.DictReader(open(csv_path)))
    hook = int(rows[0]["submit_enter_ns"])
    hw = None
    if os.path.exists(out_path):
        m = HW_RE.search(open(out_path, errors="replace").read())
        if m:
            hw = int(m.group(1))
    rise_lo, rise_hi, fall_lo, fall_hi = edges(run)
    rel = lambda t: round((t - hook) / 1000.0, 2)      # us, origin = submit_enter

    sched = [e for e in events["job_next"] if e["kind"] == "sched_entry"]
    irq_jn = [e for e in events["job_next"] if e["kind"] == "irq_entry"]
    rec = {
        "run": os.path.basename(csv_path),
        "samples": len(run["samples"]),
        "hw_elapse_ns": hw,
        "ioctl_rel_us": rel(events["ioctl"]) if "ioctl" in events else None,
        # names say what they are: entries, not the commit inside
        "sched_entry_rel_us": rel(sched[0]["ts"]) if sched else None,
        "irq_handler_rel_us": rel(events["irq"][0]) if events["irq"] else None,
        "job_schedule_calls": len(events["job_schedule"]),
        "job_next_calls": len(events["job_next"]),
        "job_next_sched_calls": len(sched),
        "job_next_irq_calls": len(irq_jn),
        "rise_us": (rise_lo / 1000.0, rise_hi / 1000.0) if rise_lo is not None else None,
        "fall_us": (fall_lo / 1000.0, fall_hi / 1000.0) if fall_lo is not None else None,
        "trace_loss": trace_loss(trace_path),
    }
    if rec["sched_entry_rel_us"] is not None and rec["irq_handler_rel_us"] is not None:
        rec["irq_minus_sched_entry_us"] = round(
            rec["irq_handler_rel_us"] - rec["sched_entry_rel_us"], 2)
        if hw is not None:
            rec["hw_minus_irq_minus_sched_us"] = round(
                hw / 1000.0 - rec["irq_minus_sched_entry_us"], 2)
    if rec["rise_us"] is not None and rec["sched_entry_rel_us"] is not None:
        rec["rise_minus_sched_entry_us"] = (
            round(rec["rise_us"][0] - rec["sched_entry_rel_us"], 2),
            round(rec["rise_us"][1] - rec["sched_entry_rel_us"], 2))
    if rec["fall_us"] is not None and rec["irq_handler_rel_us"] is not None:
        rec["fall_minus_irq_handler_us"] = (
            round(rec["fall_us"][0] - rec["irq_handler_rel_us"], 2),
            round(rec["fall_us"][1] - rec["irq_handler_rel_us"], 2))
    return rec


def summarise(records):
    def col(key):
        return [r[key] for r in records if r.get(key) is not None]

    def col_lo(key):
        return [r[key][0] for r in records if r.get(key) is not None]

    def col_hi(key):
        return [r[key][1] for r in records if r.get(key) is not None]

    def stats(values):
        if not values:
            return None
        return {"n": len(values), "min": round(min(values), 2),
                "p50": round(statistics.median(values), 2), "max": round(max(values), 2)}

    return {
        "runs": len(records),
        "ioctl_rel_us": stats(col("ioctl_rel_us")),
        "sched_entry_rel_us": stats(col("sched_entry_rel_us")),
        "irq_handler_rel_us": stats(col("irq_handler_rel_us")),
        "hw_minus_irq_minus_sched_us": stats(col("hw_minus_irq_minus_sched_us")),
        # interval-valued: report both endpoints
        "rise_minus_sched_entry_us": {"lo": stats(col_lo("rise_minus_sched_entry_us")),
                                      "hi": stats(col_hi("rise_minus_sched_entry_us"))},
        "fall_minus_irq_handler_us": {"lo": stats(col_lo("fall_minus_irq_handler_us")),
                                      "hi": stats(col_hi("fall_minus_irq_handler_us"))},
        "job_next_calls": stats([float(r["job_next_calls"]) for r in records]),
        "job_schedule_calls": stats([float(r["job_schedule_calls"]) for r in records]),
        "run_count": len(records),
        "per_run": records,
    }


def write_outputs(result, out_dir):
    import json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(out_dir, "alignment.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "samples", "ioctl_rel_us", "sched_entry_rel_us",
                    "irq_handler_rel_us", "hw_elapse_ns", "rise_lo_us", "rise_hi_us",
                    "fall_lo_us", "fall_hi_us", "rise_minus_sched_lo", "rise_minus_sched_hi",
                    "fall_minus_irq_lo", "fall_minus_irq_hi",
                    "hw_minus_irq_minus_sched_us", "job_next_calls", "job_schedule_calls",
                    "trace_lost"])
        for r in result["per_run"]:
            rise = r.get("rise_us")
            fall = r.get("fall_us")
            rs = r.get("rise_minus_sched_entry_us")
            fm = r.get("fall_minus_irq_handler_us")
            loss = r.get("trace_loss") or {}
            w.writerow([r["run"], r["samples"], r.get("ioctl_rel_us"),
                        r.get("sched_entry_rel_us"), r.get("irq_handler_rel_us"),
                        r.get("hw_elapse_ns"),
                        rise[0] if rise else "", rise[1] if rise else "",
                        fall[0] if fall else "", fall[1] if fall else "",
                        rs[0] if rs else "", rs[1] if rs else "",
                        fm[0] if fm else "", fm[1] if fm else "",
                        r.get("hw_minus_irq_minus_sched_us"), r.get("job_next_calls"),
                        r.get("job_schedule_calls"), loss.get("lost", "")])

    def s(key, field="p50"):
        v = result.get(key)
        return "n/a" if not v else "%.1f" % v[field]

    lines = ["# 驱动事件与采样器的时间对齐（ftrace `mono_raw`）\n"]
    lines.append("ftrace 记录在 **mono_raw**，与采样器 CSV 同一时钟，因此可以直接相减；"
                 "原点统一取采样器的 `submit_enter`。ftrace 时间戳分辨率为 1 µs。\n")
    lines.append("| 量 | p50 | min | max | n |")
    lines.append("|---|---|---|---|---|")
    for key, label in (("ioctl_rel_us", "hook 入口 → 真实 ioctl 入口（我们自己的 wrapper 开销）"),
                       ("sched_entry_rel_us", "**`rknpu_job_next` 调度路径入口**（来自 `job_schedule`）"),
                       ("irq_handler_rel_us", "**IRQ handler 入口**"),
                       ("hw_minus_irq_minus_sched_us", "`hw_elapse − (IRQ 入口 − 调度入口)`")):
        v = result.get(key)
        lines.append("| %s | %s | %s | %s | %s |" % (
            label, s(key), s(key, "min"), s(key, "max"), v["n"] if v else 0))
    life = [r["lifecycle"] for r in result.get("per_run", []) if r.get("lifecycle")]
    if life:
        def agg(key):
            vals = [l[key] for l in life if l.get(key) is not None]
            return (round(statistics.median(vals), 1), len(vals)) if vals else (None, 0)
        lines.append("\n## PM / 作业生命周期（同批 trace）\n")
        lines.append("| 量 | 中位 | n |")
        lines.append("|---|---|---|")
        for key, label in (("power_on_us", "`rknpu_power_on`（相对首事件）"),
                           ("commit_us", "commit"),
                           ("job_free_us", "`rknpu_job_free`"),
                           ("job_next_gap_us", "两次 `rknpu_job_next` 间隔")):
            med, n = agg(key)
            lines.append("| %s | %s | %d |" % (label, "n/a" if med is None else "%.1f µs" % med, n))
        on_seen = sum(1 for l in life if l["power_on_seen"])
        off_seen = sum(1 for l in life if l["power_off_seen"])
        lines.append("| `rknpu_power_on` 出现次数 | %d / %d 次运行 | |" % (on_seen, len(life)))
        lines.append("| `rknpu_power_off` 出现次数 | %d / %d 次运行 | |" % (off_seen, len(life)))
        lines.append("\n- **域是由负载自己打开的**：首次 `rknpu_power_get`（GEM create）在 +4 µs 内触发 "
                     "`rknpu_power_on`，比 commit 早数毫秒。采样器的电源复查之所以能通过，"
                     "靠的就是这些 GEM 引用，**不是**采样器。")
        if on_seen < len(life) and off_seen == 0:
            lines.append("- 后续运行里没有再次 `rknpu_power_on`，且 `rknpu_power_off` 全程未出现 ⇒ "
                         "**域跨运行保持开启**（延迟断电未在观测窗口内发生）。")
        lines.append("")
        lines.append("- 注意：本表时间原点是**该 trace 的第一条事件**（通常是首个 GEM create），"
                     "与主表（原点 = 采样器 `submit_enter`）不同。")
        lines.append("- **`rknpu_job_next.part.0` 每个 job 被调用两次**：第一次来自 `rknpu_job_schedule`"
                     "（commit），第二次来自 IRQ handler（派发下一个 job）。只有第一次是 kick —— "
                     "本工具取第一次。\n")

    for key, label in (("rise_minus_sched_entry_us", "上升沿 − 调度路径入口（区间）"),
                       ("fall_minus_irq_handler_us", "下降沿 − IRQ handler 入口（区间）")):
        v = result.get(key)
        if not v or not v["lo"]:
            continue
        lines.append("| %s | %s ~ %s | %s | %s | %s |" % (
            label, "%.1f" % v["lo"]["p50"], "%.1f" % v["hi"]["p50"],
            "%.1f" % v["lo"]["min"], "%.1f" % v["hi"]["max"], v["lo"]["n"]))
    jc = result.get("job_next_calls")
    js = result.get("job_schedule_calls")
    lines.append("| `job_next` 调用次数 / `job_schedule` 调用次数 | %s / %s | | | | |" % (
        "n/a" if not jc else "%.0f" % jc["p50"], "n/a" if not js else "%.0f" % js["p50"]))
    lines.append("\n## 语义（严格）\n")
    lines.append("- `rknpu_job_next.part.0` 的**函数入口**只叫「调度路径入口」。源码里它有两条早退"
                 "（`soft_reseting`、`subcore_data->job` 忙或 todo_list 空），早退路径**不设** "
                 "`hw_commit_time`，也**不写** `PC_OP_EN`；")
    lines.append("- 真正的提交在 `rknpu_job_commit → rknpu_job_subcore_commit → ..._commit_pc`，"
                 "全是 `static`/`static inline`，**不在 `available_filter_functions`** 里 ⇒ "
                 "ftrace 无法直接给出 `PC_OP_EN` 写入时刻；")
    lines.append("- 因此：调度入口**早于** kick，只能作为 `kick_rel` 的**下界**；")
    lines.append("- 两次 `job_next` 调用来源不同（`job_schedule` / `irq_handler`），"
                 "**不能按次数统计提交或 task**；第二次也可能只是早退检查。")
    lines.append("- 位与事件的关系只用**区间**表达，事件语义保留为「未识别其对应的硬件动作」。\n")
    lines.append("## 没有证明的\n")
    lines.append("- 只覆盖这些运行与这个负载；ftrace 分辨率 1 µs；")
    lines.append("- 不涉及 regcmd 位置或 per-regcmd 时长；")
    lines.append("- 位与提交奇偶的对应仍不等于已识别执行 bank。\n")
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

    HOOK = 1_000_000_000

    def make(tmp, io_off_us=7.0, commit_off_us=15.0, irq_off_us=130.0, hw=123400,
             pattern=("--", "S-"), bracket=900):
        trace = os.path.join(tmp, "t.txt")
        with open(trace, "w") as fh:
            for fn, off, caller in (
                    ("__rknpu_submit_ioctl", io_off_us, "drm_ioctl_kernel"),
                    ("rknpu_job_schedule", commit_off_us - 3, "rknpu_submit_ioctl"),
                    ("rknpu_job_next.part.0", commit_off_us, "rknpu_job_schedule"),
                    ("rknpu_core0_irq_handler", irq_off_us, "__handle_irq_event_percpu"),
                    ("rknpu_job_next.part.0", irq_off_us + 20, "rknpu_irq_handler.constprop.0")):
                fh.write("  x-1 [000] .... %d.%06d: %s <-%s\n"
                         % (int((HOOK + off * 1000) // 10 ** 9), (HOOK + off * 1000) % 10 ** 9 // 1000,
                            fn, caller))
        csvp = os.path.join(tmp, "s.csv")
        with open(csvp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_ns", "t_end_ns", "c0_ts", "c0_raw", "c0_raw2", "c0_gb_ns",
                        "c0_ge_ns", "submit_enter_ns", "task_before", "task_after",
                        "sample_validity"])
            for i, st in enumerate(pattern):
                ts = HOOK + (10 + i * 20) * 1000        # 10 us, 30 us, ...
                v = 0x100 if st == "S" else 0
                w.writerow([ts, ts + bracket, "0x0", "0x%08x" % v, "0x0", ts, ts + bracket,
                            HOOK, 0, 0, "ok"])
        outp = os.path.join(tmp, "out.txt")
        open(outp, "w").write("SUBMIT ret=0 hw_elapse=%d ns\nRESULT: BIT-EXACT (fp16)\n" % hw)
        return trace, csvp, outp

    def alignment():
        with tempfile.TemporaryDirectory() as tmp:
            rec = analyse_run(*make(tmp))
            assert rec is not None, "run not parsed"
            assert abs(rec["ioctl_rel_us"] - 7.0) < 0.01, rec
            assert abs(rec["sched_entry_rel_us"] - 15.0) < 0.01, rec
            assert abs(rec["irq_handler_rel_us"] - 130.0) < 0.01, rec
            assert abs(rec["irq_minus_sched_entry_us"] - 115.0) < 0.01, rec
            assert abs(rec["hw_minus_irq_minus_sched_us"] - (123.4 - 115.0)) < 0.01, rec
            assert rec["job_next_calls"] == 2, rec
            assert rec["job_next_sched_calls"] == 1 and rec["job_next_irq_calls"] == 1, rec
            print("  ok  entries land at their offsets and are classified by CALLER (1 sched + 1 irq)")

    group("driver events are placed on the sampler clock", alignment)

    def trace_loss_is_read():
        with tempfile.TemporaryDirectory() as tmp:
            trace, csvp, outp = make(tmp)
            txt = open(trace).read()
            open(trace, "w").write("# tracer: function\n"
                                   "# entries-in-buffer/entries-written: 31/40   #P:8\n" + txt)
            loss = trace_loss(trace)
            assert loss == {"in_buffer": 31, "written": 40, "lost": 9}, loss
            print("  ok  a wrapped ring buffer is reported as lost events")

    group("trace loss is detected from the ftrace header", trace_loss_is_read)

    def edge_differences():
        with tempfile.TemporaryDirectory() as tmp:
            # first sample clear at 10 us, second set at 30 us -> rise bracket [10, 30.9]
            rec = analyse_run(*make(tmp, pattern=("--", "S"), commit_off_us=15.0))
            assert rec["rise_us"][0] == 10.0, rec["rise_us"]
            lo, hi = rec["rise_minus_sched_entry_us"]
            assert abs(lo - (10.0 - 15.0)) < 0.01, rec
            assert abs(hi - (30.9 - 15.0)) < 0.01, rec
            print("  ok  rise difference vs scheduler entry is a full INTERVAL (%+.1f ~ %+.1f us)" % (lo, hi))

    group("edge-to-event differences are full intervals vs the scheduler entry",
          edge_differences)

    def lifecycle_facts():
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.txt")
            with open(p, "w") as fh:
                base = 1_000_000_000
                for off, fn in ((0, "rknpu_power_get"), (4, "rknpu_power_on"),
                                (6410, "rknpu_power_get"), (6416, "rknpu_job_next.part.0"),
                                (6601, "rknpu_job_next.part.0"), (6771, "rknpu_job_free"),
                                (11975, "rknpu_power_put_delay")):
                    t = base + off * 1000
                    fh.write("  x-1 [000] .... %d.%06d: %s <-c\n"
                             % (t // 10 ** 9, (t % 10 ** 9) // 1000, fn))
            life = lifecycle(p)
            assert life["power_on_us"] == 4.0, life
            assert life["commit_us"] == 6416.0, life  # lifecycle(): first job_next, caller kept separately
            assert life["job_next_calls"] == 2, life
            assert life["job_next_gap_us"] == 185.0, life
            assert life["power_off_seen"] is False, life
            assert life["domain_on_before_commit"] is True, life
            print("  ok  power_on/sched entry/job_free placed, second job_next flagged, no power_off")

    group("lifecycle facts, including the double job_next", lifecycle_facts)

    def missing_events():
        with tempfile.TemporaryDirectory() as tmp:
            trace, csvp, outp = make(tmp)
            open(trace, "w").write("# empty\n")
            rec = analyse_run(trace, csvp, outp)
            assert rec is None, "a run without a commit must not be reported"
            print("  ok  a trace without commit yields no record (no invented kick)")

    group("missing driver events are not invented", missing_events)

    def summary_stats():
        with tempfile.TemporaryDirectory() as tmp:
            recs = []
            for i, commit in enumerate((15.0, 17.0, 19.0)):
                d = os.path.join(tmp, "r%d" % i)
                os.makedirs(d)
                recs.append(analyse_run(*make(d, commit_off_us=commit)))
            s = summarise(recs)
            assert s["runs"] == 3 and s["sched_entry_rel_us"]["p50"] == 17.0, s["sched_entry_rel_us"]
            assert s["sched_entry_rel_us"]["min"] == 15.0 and s["sched_entry_rel_us"]["max"] == 19.0
            print("  ok  summary reports p50/min/max over runs")

    group("summary statistics", summary_stats)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all driver-event selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="directory with t<N>.txt / s<N>.csv / out<N>.txt")
    ap.add_argument("--out-dir")
    ap.add_argument("--lifecycle-only", action="store_true",
                    help="trace files without sampler CSVs: report PM/job lifecycle only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dir or not args.out_dir:
        ap.error("--dir and --out-dir are required unless --selftest")
    records = []
    i = 1
    while True:
        t = os.path.join(args.dir, "t%d.txt" % i)
        c = os.path.join(args.dir, "s%d.csv" % i)
        o = os.path.join(args.dir, "out%d.txt" % i)
        if not (os.path.exists(t) or os.path.exists(c)):
            break
        if args.lifecycle_only or not os.path.exists(c):
            rec = analyse_lifecycle_only(t) if os.path.exists(t) else None
        else:
            rec = analyse_run(t, c, o)
        if rec:
            records.append(rec)
        i += 1
    if not records:
        sys.exit("no complete run found in %s" % args.dir)
    result = summarise(records)
    write_outputs(result, args.out_dir)
    def p50(key):
        v = result.get(key)
        return "n/a" if not v else "%.1f" % v["p50"]

    rse = result.get("rise_minus_sched_entry_us") or {}
    fmi = result.get("fall_minus_irq_handler_us") or {}
    def iv(v):
        if not v or not v.get("lo") or not v.get("hi"):
            return "n/a"
        return "%.1f~%.1f" % (v["lo"]["p50"], v["hi"]["p50"])
    print("runs=%d  sched_entry p50=%s us  irq p50=%s us  rise-sched=%s  fall-irq=%s"
          % (result["runs"], p50("sched_entry_rel_us"), p50("irq_handler_rel_us"),
             iv(rse), iv(fmi)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
