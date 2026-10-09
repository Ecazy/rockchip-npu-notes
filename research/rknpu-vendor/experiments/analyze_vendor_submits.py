#!/usr/bin/env python3
"""从 hook trace 的 `submit`/`task` 事件提取 **vendor 侧提交口径**（§7.43）。

回答的问题：vendor 一条 submit 放多少 task、`core_mask`/`subcore_task` 怎么填、
task 描述符字段（flags/`regcfg_amount`/`regcfg_offset`/`regcmd_addr`）怎么排、
以及整条 submit 的 `hw_elapse` 意味着每个 task 多贵。

用法：analyze_vendor_submits.py [--all | <trace.jsonl> ...]
"""
import collections
import glob
import json
import sys


def load(path):
    subs, tasks = [], collections.Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"submit"' not in line and '"task"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") == "submit":
            subs.append(ev)
        elif ev.get("event") == "task":
            tasks[(ev.get("regcfg_amount"), ev.get("regcfg_offset"), ev.get("flags"))] += 1
    return subs, tasks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv or not args:
        args = sorted(set(glob.glob("experiments/**/*.jsonl", recursive=True)))
    for p in args:
        try:
            subs, tasks = load(p)
        except FileNotFoundError:
            continue
        if not subs:
            continue
        print(f"== {p} ==")
        for ev in subs:
            n = ev.get("task_number") or 0
            hw = ev.get("hw_elapse_time_ns")
            per = (hw / n) if (hw and n) else None
            print("   submit: core_mask=%s task_start=%s task_number=%s flags=%#x "
                  "task_base=%s task_counter=%s→%s hw=%s ns%s"
                  % (ev.get("core_mask"), ev.get("task_start"), n, ev.get("flags") or 0,
                     ev.get("task_base_addr"), ev.get("task_counter_before"),
                     ev.get("task_counter_after"), hw,
                     f"  ⇒ 每 task ≈ {per:.1f} ns" if per else ""))
            print("           subcore_tasks=%s" % (ev.get("subcore_tasks"),))
        if tasks:
            print("   task 描述符字段分布 (regcfg_amount, regcfg_offset, flags) → 条数:")
            for k, v in tasks.most_common(6):
                print("      %s → %d" % (k, v))
        print()


if __name__ == "__main__":
    main()
