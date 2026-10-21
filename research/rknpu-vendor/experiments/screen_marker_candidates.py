#!/usr/bin/env python3
"""Screen regcmd registers for use as persistent position markers (route B).

Purpose
-------
The regcmd timing plan (research/scheduling/REGCMD-TIMING-OBSERVABILITY.md, route B)
needs a register whose value can be read back AFTER a chosen command position and
still distinguish "marker write has happened" from "has not happened yet".
This script ranks every register that appears in a captured regcmd stream with
that criterion in mind.  It is an OFFLINE screen: it reads the already-captured
8-byte regcmd payloads, so it answers only "which registers are worth a board
test", never "this register is safe".

Input
-----
A decoded rows CSV, one row per 8-byte regcmd entry.  Required columns:
    task_index, task_local_regcmd_index, module, reg_offset, reg_name, value
`sdpa_regcmd_order_rows.csv` (produced by export_sdpa_timeline.py) has them.
Optional `--xml` gives register names/bits for offsets the CSV left unnamed.

Exclusion classes (never proposed, per plan section 4)
------------------------------------------------------
    chain control   PC_BASE_ADDRESS / PC_REGISTER_AMOUNTS / PC_OPERATION_ENABLE
    banking         *_S_POINTER (also tells us whether ping-pong is live)
    side effects    *_OPERATION_ENABLE, LUT_ACCESS_*, *_INTERRUPT*, *_STATUS
    unknown         empty reg_name or offset absent from the XML

Candidate score (higher = better natural marker)
------------------------------------------------
    single write per task        \
    high across-task value change >  natural "before/after" discrimination
    early position in the task   /
    long quiet remainder after the write
    no ping-pong on the owning module (readback bank unambiguous)
    written in many tasks (measurement can be repeated at many positions)

Usage
-----
    python3 screen_marker_candidates.py \
        --rows results/sdpa_timeline_20261020/sdpa_regcmd_order_rows.csv \
        --xml ../../../ppu-rknn-capture/registers.xml \
        --out-dir results/marker_screen_20261021
"""

import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys

# ---------------------------------------------------------------------------
# exclusion rules
# ---------------------------------------------------------------------------

# Registers that are not plain configuration: writing or reading them has a
# side effect, selects a bank, or controls the command chain itself.
EXCLUDE_PATTERNS = [
    ("chain_control", re.compile(r"^PC_(BASE_ADDRESS|REGISTER_AMOUNTS|OPERATION_ENABLE)$")),
    ("chain_control", re.compile(r"TASK_CON|CHAIN|SEAL|JUMP")),
    ("banking", re.compile(r"S_POINTER|PP_CLEAR|PP_EN|PP_MODE")),
    ("side_effect", re.compile(r"OPERATION_ENABLE|OP_ENABLE")),
    ("side_effect", re.compile(r"LUT_ACCESS")),
    ("side_effect", re.compile(r"INTERRUPT|IRQ")),
    ("side_effect", re.compile(r"STATUS|AMOUNTS|_PC_")),
    ("readonly_id", re.compile(r"VERSION")),
]

# Modules that are pure write targets in the captured stream: their registers are
# not a general purpose store.  Kept for reporting, not used to exclude.
READ_WINDOW_BASE = {
    # base offsets inside one core window (0x10000 apart), as used by
    # bw_thread() /dev/mem window in rknpu_ioctl_hook.c
    "CNA": 0x1000,
    "CORE": 0x3000,
    "DPU": 0x4000,
    "DPU_RDMA": 0x5000,
    "PPU": 0x6000,
    "PPU_RDMA": 0x7000,
}


def classify(module, offset, name):
    """Return (kind, reason).  kind in {config, excluded}."""
    if not name:
        return "excluded", "unknown_register_name"
    for reason, pat in EXCLUDE_PATTERNS:
        if pat.search(name):
            return "excluded", reason
    if module not in READ_WINDOW_BASE:
        return "excluded", "module_not_in_mmio_window"
    return "config", "plain_config_candidate"


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"task_index", "task_local_regcmd_index", "module", "reg_offset",
                "reg_name", "value"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            sys.exit("rows CSV missing columns: %s" % ", ".join(sorted(missing)))
        rows = []
        for raw in reader:
            rows.append({
                "task": int(raw["task_index"]),
                "pos": int(raw["task_local_regcmd_index"]),
                "module": raw["module"],
                "offset": int(raw["reg_offset"], 16),
                "name": raw["reg_name"].strip(),
                "value": int(raw["value"], 16),
                "task_len": int(raw.get("task_regcmd_count_captured") or 0),
            })
    if not rows:
        sys.exit("no rows in %s" % path)
    return rows


def load_xml_names(path):
    """offset-only fallback naming for modules whose CSV name was empty."""
    if not path:
        return {}
    text = open(path).read()
    names = {}
    for m in re.finditer(r'<reg32 offset="(0x[0-9a-f]+)" name="([A-Z0-9_]+)">', text):
        names[int(m.group(1), 16)] = m.group(2)
    return names


def module_summary(rows):
    """Which modules are actually driven, and does anyone enable ping-pong?"""
    per_module = collections.defaultdict(lambda: {
        "entries": 0, "tasks": set(), "writes_to_s_pointer": 0,
        "pp_enabled_values": collections.Counter(),
        "operation_enable_writes": 0, "distinct_regs": set(),
    })
    for r in rows:
        s = per_module[r["module"]]
        s["entries"] += 1
        s["tasks"].add(r["task"])
        s["distinct_regs"].add(r["offset"])
        if "S_POINTER" in r["name"]:
            s["writes_to_s_pointer"] += 1
            # POINTER_PP_EN is bit1, POINTER_PP_MODE bit3, EXECUTER_PP_EN bit2
            if r["value"] & 0b1110:
                s["pp_enabled_values"][hex(r["value"])] += 1
        if "OPERATION_ENABLE" in r["name"]:
            s["operation_enable_writes"] += 1
    out = {}
    for mod, s in per_module.items():
        # A block's register bank cannot be inferred from the captured command
        # stream alone: the driver writes CNA/CORE S_POINTER (0x1004/0x3004) with
        # 0xE + 0x10000000*core at submit time, so a stream with no S_POINTER
        # write does NOT mean "single bank".  Every module therefore starts at
        # "unknown"; only an in-stream ping-pong write makes it explicit.
        if s["writes_to_s_pointer"] == 0:
            bank_state = "unknown_driver_writes_it"
        elif bool(s["pp_enabled_values"]):
            bank_state = "ping_pong_live"
        else:
            bank_state = "pointer_selected"
        out[mod] = {
            "entries": s["entries"],
            "tasks_written": len(s["tasks"]),
            "distinct_registers": len(s["distinct_regs"]),
            "s_pointer_writes": s["writes_to_s_pointer"],
            "bank_state": bank_state,
        }
    return out


def analyse(rows, xml_names):
    tasks = collections.defaultdict(list)
    for r in rows:
        if not r["name"]:
            r["name"] = xml_names.get(r["offset"], "")
        tasks[r["task"]].append(r)
    for t in tasks.values():
        t.sort(key=lambda r: r["pos"])

    # per (module, offset): every write, in global order
    writes = collections.defaultdict(list)
    for task_index in sorted(tasks):
        for r in tasks[task_index]:
            writes[(r["module"], r["offset"])].append(r)

    modules = module_summary(rows)
    results = []
    for (module, offset), ws in writes.items():
        name = ws[0]["name"]
        kind, reason = classify(module, offset, name)
        by_task = collections.defaultdict(list)
        for w in ws:
            by_task[w["task"]].append(w)
        per_task_counts = [len(v) for v in by_task.values()]
        # last value written per task, in task order -> across-task signal
        last_vals = []
        for task_index in sorted(by_task):
            last_vals.append(by_task[task_index][-1]["value"])
        changes = sum(1 for a, b in zip(last_vals, last_vals[1:]) if a != b)
        p_change = changes / (len(last_vals) - 1) if len(last_vals) > 1 else 0.0
        # position of the first write inside the task, as a fraction of task len
        pos_frac = []
        quiet_after = []
        for task_index, lst in by_task.items():
            tlen = max(r["task_len"] for r in lst) or (lst[-1]["pos"] + 1)
            pos_frac.append(lst[0]["pos"] / tlen if tlen else 0.0)
            quiet_after.append(tlen - lst[-1]["pos"])
        # late rewrites of the same register inside one task: a marker that is
        # rewritten later cannot be read as "position j has passed"
        tasks_with_rewrite = sum(1 for v in per_task_counts if v > 1)
        bank_state = modules.get(module, {}).get("bank_state", "unknown_driver_writes_it")

        score = 0.0
        if kind == "config":
            score = 1.0
            score *= 0.4 + 0.6 * p_change                      # discriminative
            score *= 1.0 / (1.0 + (statistics.mean(per_task_counts) - 1) * 2)  # single write
            score *= 1.0 - min(0.9, statistics.mean(pos_frac))  # early = longer window
            # No offline evidence can establish a unique readback bank, so every
            # candidate carries the same penalty instead of a false bonus.
            score *= 0.5 if bank_state != "single_bank" else 1.0
            score *= min(1.0, len(by_task) / 40.0)             # repeatability

        results.append({
            "module": module,
            "offset": "0x%04x" % offset,
            "name": name or "(unnamed)",
            "kind": kind,
            "reason": reason,
            "writes": len(ws),
            "tasks_written": len(by_task),
            "writes_per_task_mean": round(statistics.mean(per_task_counts), 3),
            "tasks_with_rewrite": tasks_with_rewrite,
            "distinct_values": len(set(last_vals)),
            "p_value_changes_across_tasks": round(p_change, 3),
            "first_write_pos_frac_mean": round(statistics.mean(pos_frac), 3),
            "quiet_entries_after_last_write_mean": round(statistics.mean(quiet_after), 1),
            "module_bank_state": bank_state,
            "score": round(score, 4),
        })
    results.sort(key=lambda r: (-r["score"], r["module"], r["offset"]))
    return results, modules


def write_csv(path, results):
    cols = ["rank", "module", "offset", "name", "kind", "reason", "score", "writes",
            "tasks_written", "writes_per_task_mean", "tasks_with_rewrite",
            "distinct_values", "p_value_changes_across_tasks",
            "first_write_pos_frac_mean", "quiet_entries_after_last_write_mean",
            "module_bank_state"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(results, 1):
            w.writerow([i] + [r[c] for c in cols[1:]])


def write_summary(path, results, modules, rows):
    tasks = {r["task"] for r in rows}
    lines = []
    lines.append("# 标记寄存器离线筛选（route B 前置）\n")
    lines.append("由 `screen_marker_candidates.py` 生成。输入是**已捕获**的 regcmd 流，"
                 "结论只回答“哪些寄存器值得上板试”，不回答“这个寄存器安全”。\n")
    lines.append("## 流概况\n")
    lines.append(f"- tasks: {len(tasks)}, regcmd entries: {len(rows)}\n")
    lines.append("\n| module | entries | tasks | distinct regs | S_POINTER writes | bank state |")
    lines.append("|---|---|---|---|---|---|")
    for mod in sorted(modules):
        s = modules[mod]
        lines.append("| %s | %d | %d | %d | %d | %s |" % (
            mod, s["entries"], s["tasks_written"], s["distinct_registers"],
            s["s_pointer_writes"], s["bank_state"]))
    lines.append("")

    idle = [m for m in READ_WINDOW_BASE if m not in modules]
    lines.append("\n本流中**完全没有被写入**的模块：%s。"
                 "注入式标记（B1）必须落在这类模块里；若为空，B1 只能换一个合成负载。\n"
                 % (", ".join(idle) if idle else "无"))

    top = [r for r in results if r["kind"] == "config"][:12]
    lines.append("\n## 自然标记（B0）候选 top %d\n" % len(top))
    lines.append("| # | module | offset | name | p(value 变化) | 单 task 写次数 | 首次写位置 | 写后剩余条目 | bank state | score |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(top, 1):
        lines.append("| %d | %s | %s | %s | %.2f | %.2f | %.2f | %.1f | %s | %.3f |" % (
            i, r["module"], r["offset"], r["name"], r["p_value_changes_across_tasks"],
            r["writes_per_task_mean"], r["first_write_pos_frac_mean"],
            r["quiet_entries_after_last_write_mean"],
            r["module_bank_state"], r["score"]))
    lines.append("")
    lines.append("\n### 排除了什么，为什么\n")
    reasons = collections.Counter(r["reason"] for r in results if r["kind"] == "excluded")
    for reason, n in reasons.most_common():
        lines.append(f"- `{reason}`: {n} 个寄存器")
    lines.append("")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--xml", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = load_rows(args.rows)
    xml_names = load_xml_names(args.xml)
    results, modules = analyse(rows, xml_names)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "candidates.csv"), results)
    write_summary(os.path.join(args.out_dir, "README.md"), results, modules, rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "source_rows": args.rows,
            "tasks": len({r["task"] for r in rows}),
            "regcmd_entries": len(rows),
            "modules": modules,
            "idle_modules": [m for m in READ_WINDOW_BASE if m not in modules],
            "top_config_candidates": [r for r in results if r["kind"] == "config"][:12],
        }, f, indent=2)
    print("wrote %s (%d registers classified)" % (args.out_dir, len(results)))


if __name__ == "__main__":
    main()
