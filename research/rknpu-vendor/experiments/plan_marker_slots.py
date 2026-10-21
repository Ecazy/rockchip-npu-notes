#!/usr/bin/env python3
"""Plan length-preserving marker slots for route B.

Why replacement instead of insertion
------------------------------------
Inserting an 8-byte word forces a rebuild of stream length, alignment, the
successor pointer and the terminal seal.  SAME-CORE-OVERLAP.md records what that
class of change does on this board (timeouts, changed bank/start conditions), so
the first marker experiment does not do it.

Instead we replace a slot that the duplicate-write ablation already proved
redundant: the same (block, register) was written with the same value earlier in
the same submit, so dropping it cannot change the computation.  All three arms
have identical length:

    original   untouched
    nop        that slot becomes OP_NONE       (length control)
    marker     that slot becomes a PPU write   (the marker)

The hook enforces this at run time (see `marker_check()` in
rknpu_ioctl_hook.c); this script only proposes slots and emits the exact
environment for the three arms.

Usage
-----
    python3 plan_marker_slots.py --rows results/sdpa_timeline_20261020/sdpa_regcmd_order_rows.csv \
        --module DPU --out-dir results/marker_plan_20261022
    python3 plan_marker_slots.py --selftest

Caveat that must be checked on the board
----------------------------------------
The hook counts tasks in *submit traversal order*, while the rows CSV is ordered
by task index.  For a single-core submit they coincide; the hook prints
`orig=0x...` for the slot it actually rewrote, so the plan can be verified
against that line before trusting any timing result.
"""

import argparse
import collections
import csv
import json
import os
import sys

TARGET = {"PC": 0x0100, "CNA": 0x0200, "CORE": 0x0800, "DPU": 0x1000,
          "DPU_RDMA": 0x2000, "PPU": 0x4000, "PPU_RDMA": 0x8000}

# marker register policy, identical to marker_reg_forbidden() in the hook
MARKER_CANDIDATES = [
    (0x6010, "PPU DATA_CUBE_IN_HEIGHT"),
    (0x6014, "PPU DATA_CUBE_IN_CHANNEL"),
    (0x6018, "PPU DATA_CUBE_OUT_WIDTH"),
    (0x6084, "PPU DATA_FORMAT"),
    (0x7010, "PPU_RDMA DATA_CUBE_IN_HEIGHT"),
]


def side_effect(op, reg):
    """Mirror of ded_side_effect() in the hook."""
    if ((op >> 8) & 0xFF) == 0x01:
        return True
    if reg == 0x0008 or 0x0020 <= reg <= 0x002C:
        return True
    if reg in (0x1004, 0x3004, 0x4004, 0x5004, 0x6004, 0x7004):
        return True
    if reg in (0x1008, 0x3008, 0x4008, 0x5008, 0x6008, 0x7008):
        return True
    if reg in (0x4100, 0x4104):
        return True
    return False


def marker_reg_ok(reg):
    """Mirror of marker_reg_forbidden() == 0."""
    if (reg >> 12) not in (0x6, 0x7):
        return False
    if reg & 3:
        return False
    if (reg & 0xFFF) in (0x000, 0x004, 0x008):
        return False
    return not side_effect(0x4001, reg)


def load_rows(path):
    tasks = collections.defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        need = {"task_index", "task_local_regcmd_index", "module", "reg_offset",
                "reg_name", "op", "value"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            sys.exit("rows CSV missing: %s" % ", ".join(sorted(missing)))
        for row in reader:
            tasks[int(row["task_index"])].append({
                "task": int(row["task_index"]),
                "pos": int(row["task_local_regcmd_index"]),
                "op": int(row["op"], 16),
                "reg": int(row["reg_offset"], 16),
                "module": row["module"],
                "name": row["reg_name"],
                "value": int(row["value"], 16),
            })
    for task in tasks:
        tasks[task].sort(key=lambda r: r["pos"])
    return [tasks[t] for t in sorted(tasks)]


def encode(op, value, reg):
    return (op << 48) | (value << 16) | reg


def bank_affecting(reg):
    """Mirror of ded_bank_affecting(): these writes may move the register group."""
    sub = reg & 0xFFF
    return sub in (0x004, 0x008) or reg in (0x4100, 0x4104)


def find_slots(task_list, module=None):
    """Slots whose write is redundant in the ONLY sense that is safe to assume:
    the same (block, register) was written with the same value earlier, with no
    bank/start/chain write and no task boundary in between.

    A cross-task match is not enough: with ping-pong the second write may be the
    first initialization of the other register group, and deleting it would
    remove a needed configuration.  The hook enforces the same rule at run time
    (ded_is_redundant + generation bumps), so a slot listed here is one the hook
    will also accept.
    """
    generation = 0
    last = {}
    out = []
    for flat, ops in enumerate(task_list):
        # a task boundary ends every observation window
        last = {}
        generation += 1
        for slot, op in enumerate(ops):
            if op["op"] == 0:
                continue
            block = op["op"] >> 8
            if side_effect(op["op"], op["reg"]):
                if bank_affecting(op["reg"]) or block == 0x01:
                    last = {}
                    generation += 1
                continue
            key = (generation, block, op["reg"])
            redundant = last.get(key) == op["value"]
            last[key] = op["value"]
            if redundant and (module is None or op["module"] == module):
                out.append({
                    "flat_task": flat,
                    "slot": slot,
                    "task": op["task"],
                    "module": op["module"],
                    "reg": "0x%04x" % op["reg"],
                    "name": op["name"],
                    "value": "0x%08x" % op["value"],
                    "orig_word": "0x%016x" % encode(op["op"], op["value"], op["reg"]),
                })
    return out


def plan(task_list, module, marker_reg, marker_value, limit):
    slots = find_slots(task_list, module)
    candidates = [{"reg": "0x%04x" % reg, "name": name,
                   "word": "0x%016x" % (((0x4001 if (reg >> 12) == 0x6 else 0x8001) << 48)
                                        | (marker_value << 16) | reg),
                   "allowed": marker_reg_ok(reg)}
                  for reg, name in MARKER_CANDIDATES]
    return {
        "module_filter": module,
        "marker_reg": "0x%04x" % marker_reg,
        "marker_reg_ok": marker_reg_ok(marker_reg),
        "marker_value": "0x%08x" % marker_value,
        "slots": slots[:limit],
        "slot_count": len(slots),
        "candidate_marker_registers": candidates,
    }


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "slots.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flat_task", "slot", "task", "module", "reg", "name",
                    "value", "orig_word"])
        for s in result["slots"]:
            w.writerow([s["flat_task"], s["slot"], s["task"], s["module"], s["reg"],
                        s["name"], s["value"], s["orig_word"]])
    with open(os.path.join(out_dir, "plan.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    lines = ["# 标记槽位规划（长度保持）\n"]
    lines.append("候选槽位 = **同一 task 内、同一 bank 世代**的重复写：同一 (block, 寄存器)"
                 "写过相同的值，且两次之间没有 S_POINTER / OPERATION_ENABLE / LUT / PC 写，"
                 "也没有 task 边界。只有这种重复写，替换成 OP_NONE 或标记写才**可能**不改变"
                 "计算；跨 task 的同值写不满足条件，因为另一个 bank 可能需要自己的初始化。\n")
    lines.append("- 过滤模块：`%s`；可用槽位 **%d** 个（下表列前 %d 个）" % (
        result["module_filter"] or "全部", result["slot_count"], len(result["slots"])))
    lines.append("- 建议标记寄存器：`%s`（%s）" % (
        result["marker_reg"], "通过策略检查" if result["marker_reg_ok"] else "**被策略拒绝**"))
    lines.append("- 标记值：`%s`\n" % result["marker_value"])
    if result["slots"]:
        lines.append("| flat_task | slot | 原命令 | 期望词 |")
        lines.append("|---|---|---|---|")
        for s in result["slots"]:
            lines.append("| %d | %d | %s %s = %s | `%s` |" % (
                s["flat_task"], s["slot"], s["module"], s["name"], s["value"],
                s["orig_word"]))
        first = result["slots"][0]
        lines.append("\n## 三臂环境变量（以上表第一行为例）\n")
        lines.append("```sh")
        for arm in ("original", "nop", "marker"):
            lines.append(
                "EPERF_RKNPU_MARKER_ARM=%s EPERF_RKNPU_MARKER_TASK=%d "
                "EPERF_RKNPU_MARKER_SLOT=%d EPERF_RKNPU_MARKER_REG=%s "
                "EPERF_RKNPU_MARKER_VALUE=%s LD_PRELOAD=<hook> <workload>"
                % (arm, first["flat_task"], first["slot"], result["marker_reg"],
                   result["marker_value"]))
        lines.append("```")
        lines.append("\n上板后先核对 hook 打印的 `MARKER ... orig=<期望词>` 与上表一致，"
                     "再采信任何时间结论。\n")
    else:
        lines.append("\n**这条流里没有满足条件的槽位。** 逐 task 的 S_POINTER 写会关闭每个 "
                     "config 写之前的观察窗口，因此 task 内不存在可安全删除的重复写；"
                     "实测 SDPA 的 DPU/DPU_RDMA/CNA/CORE 四个模块都是 0 个。\n")
        lines.append("\n可选下一步：\n")
        lines.append("1. 用**自写合成负载**故意在同一 task 内写两遍同一配置寄存器，"
                     "造出一个由构造保证的冗余槽位——这也是设计里\"先合成探针\"的顺序；")
        lines.append("2. 或改回真正的**插入**（需要重建流长度、后继指针与终端 seal），"
                     "风险更高，暂缓。\n")
        lines.append("\n注意：即便如此，也只能证明\"删掉这条写不改变输出\"，"
                     "不能证明任何寄存器可读或写无副作用。\n")
    lines.append("\n## 还没证明的\n")
    lines.append("- 槽位“冗余”只来自命令流层面（同值重复写），没有证明硬件对该写无副作用。")
    lines.append("- 标记寄存器是否可读、写入是否无副作用、读回命中哪个 bank，都没验证。")
    lines.append("- 标记可见时间不等于命令执行时间。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def _synth(tmpdir, n_tasks=6):
    """A stream with one genuine within-task duplicate per task, plus a value
    that repeats across tasks but must NOT count as redundant."""
    path = os.path.join(tmpdir, "rows.csv")
    rows = []
    op = 0x1001  # DPU
    for task in range(n_tasks):
        rows.append([task, 0, "DPU", "0x4020", "DPU_DST_BASE_ADDR", "0x%08x" % (0x1111 + task)])
        rows.append([task, 1, "DPU", "0x4024", "DPU_DST_SURF_STRIDE", "0x00002222"])
        # same register, same value, same task -> redundant
        rows.append([task, 2, "DPU", "0x4020", "DPU_DST_BASE_ADDR", "0x%08x" % (0x1111 + task)])
        # identical in every task, but the task boundary forbids a cross match
        rows.append([task, 3, "DPU", "0x4028", "DPU_DATA_CUBE_WIDTH", "0x00003333"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_index", "task_local_regcmd_index", "module", "reg_offset",
                    "reg_name", "op", "value"])
        for task, slot, module, reg, name, value in rows:
            w.writerow([task, slot, module, reg, name, "0x%04x" % op, value])
    return path


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
        def finds_duplicates():
            task_list = load_rows(_synth(tmpdir))
            slots = find_slots(task_list, "DPU")
            assert slots, "no redundant slots found"
            assert len(slots) == len(task_list), (len(slots), len(task_list))
            assert all(s["slot"] == 2 for s in slots), "wrong slot index"
            assert all(s["flat_task"] == i for i, s in enumerate(slots)), \
                [s["flat_task"] for s in slots]
            print("  ok  %d redundant slots, one per task, at slot 2" % len(slots))

        group("redundant writes are found", finds_duplicates)

        def rejects_side_effects():
            # an S_POINTER write must never be proposed, even when it repeats
            path = os.path.join(tmpdir, "ptr.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["task_index", "task_local_regcmd_index", "module",
                            "reg_offset", "reg_name", "op", "value"])
                for i in range(4):
                    w.writerow([0, i, "DPU", "0x4004", "DPU_S_POINTER",
                                "0x%04x" % 0x1001, "0x0000000e"])
            assert find_slots(load_rows(path), "DPU") == []
            print("  ok  S_POINTER repeats are never proposed")

        group("side-effect registers are excluded", rejects_side_effects)

        def rejects_cross_task_matches():
            # Reviewer counterexample: the same value in every task is NOT a
            # redundant write, because the other bank may need its own init.
            task_list = load_rows(_synth(tmpdir))
            slots = find_slots(task_list, "DPU")
            assert all(s["reg"] != "0x4028" for s in slots), \
                [s for s in slots if s["reg"] == "0x4028"]
            print("  ok  a value repeated across tasks is not treated as redundant")

        group("cross-task identical values are not redundant", rejects_cross_task_matches)

        def pointer_write_breaks_the_window():
            path = os.path.join(tmpdir, "ptr_window.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["task_index", "task_local_regcmd_index", "module",
                            "reg_offset", "reg_name", "op", "value"])
                # S_POINTER then config, then the same config again
                w.writerow([0, 0, "DPU", "0x4004", "DPU_S_POINTER", "0x00001001", "0x0000000e"])
                w.writerow([0, 1, "DPU", "0x4020", "DPU_DST_BASE_ADDR", "0x00001001", "0x00000011"])
                # a bank write in between invalidates the earlier observation
                w.writerow([0, 2, "DPU", "0x4004", "DPU_S_POINTER", "0x00001001", "0x0000000e"])
                w.writerow([0, 3, "DPU", "0x4020", "DPU_DST_BASE_ADDR", "0x00001001", "0x00000011"])
            slots = find_slots(load_rows(path), "DPU")
            assert slots == [], slots
            print("  ok  an intervening S_POINTER write invalidates the match")

        group("a pointer write closes the redundancy window",
              pointer_write_breaks_the_window)

        def word_encoding():
            task_list = load_rows(_synth(tmpdir))
            result = plan(task_list, "DPU", 0x6010, 0x00C0FFEE, 5)
            slot = result["slots"][0]
            # the expected original word must match what the hook will print
            op, value, reg = 0x1001, 0x1111, 0x4020
            assert slot["orig_word"] == "0x%016x" % encode(op, value, reg), slot
            assert result["marker_reg_ok"] is True
            print("  ok  original word encoding matches the hook's layout")

        group("the planned original word is the one the hook reports", word_encoding)

        def marker_policy():
            assert marker_reg_ok(0x6010) and marker_reg_ok(0x7014)
            for bad in (0x6000, 0x6004, 0x6008, 0x6012, 0x1024, 0x4104):
                assert not marker_reg_ok(bad), hex(bad)
            print("  ok  PPU plain config accepted, status/pointer/enable/LUT refused")

        group("the marker register policy mirrors the hook", marker_policy)

        def empty_module():
            task_list = load_rows(_synth(tmpdir))
            result = plan(task_list, "CNA", 0x6010, 0x1234, 5)
            assert result["slot_count"] == 0
            print("  ok  a module with no redundant writes yields no slots")

        group("a module without candidates is reported empty", empty_module)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all marker-slot planner checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows")
    ap.add_argument("--module", default="DPU",
                    help="restrict slots to one module (empty string = any)")
    ap.add_argument("--marker-reg", default="0x6010",
                    help="PPU/PPU_RDMA plain config register (0x6000+ / 0x7000+)")
    ap.add_argument("--marker-value", default="0x00c0ffee")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.rows or not args.out_dir:
        ap.error("--rows and --out-dir are required unless --selftest")
    task_list = load_rows(args.rows)
    result = plan(task_list, args.module or None, int(args.marker_reg, 0),
                  int(args.marker_value, 0), args.limit)
    write_outputs(result, args.out_dir)
    print("wrote %s: %d redundant slots in %s" % (
        args.out_dir, result["slot_count"], result["module_filter"] or "any module"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
