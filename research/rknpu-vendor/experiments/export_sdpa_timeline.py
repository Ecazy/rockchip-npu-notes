#!/usr/bin/env python3
"""Export an ordered task/regcmd/NPUOP timeline for one captured SDPA run.

Input is the rknpu ioctl JSONL created by rknpu_ioctl_hook.c.  The hook records
submit/task/regcmd events.  A submit may advertise task_number = 3 * N while
subcore_task[0..2] all describe the same [start, N] range; with core_mask=0
(AUTO) the driver consumes only one slot.  This tool therefore selects the
first effective subcore range for each submit before building the timeline.

The trace has no per-task timestamp.  Each task is assigned the wall-clock and
driver hw_elapse window of its containing SUBMIT, never a fabricated per-task
time.  Module attribution comes from the actual register offsets written by the
regcmd block, not from the task enable_mask (which the driver does not consume).

Outputs
-------
summary.md
summary.json
sdpa_task_timeline.csv
sdpa_npuop_timeline.csv
sdpa_task_sequence.txt

Usage
-----
python3 export_sdpa_timeline.py \
  --trace results/attn_fed_20261003/attn_sdpa_fed.jsonl.gz \
  --out results/sdpa_timeline_20261020
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, Iterator


MODULE_BY_PAGE = {
    0x0: "PC",
    0x1: "CNA",
    0x2: "PC2",
    0x3: "CORE",
    0x4: "DPU",
    0x5: "DPU_RDMA",
    0x6: "PPU",
    0x7: "PPU_RDMA",
    0x8: "DDMA",
    0x9: "SDMA",
}

MODULE_ORDER = [
    "PC",
    "PC2",
    "CNA",
    "CORE",
    "DPU",
    "DPU_RDMA",
    "PPU",
    "PPU_RDMA",
    "DDMA",
    "SDMA",
]

# Registers useful for a compact, human-auditable task row.  The full write
# stream is in sdpa_npuop_timeline.csv.
KEY_REGISTERS = OrderedDict(
    [
        (0x0010, "PC_BASE_ADDRESS"),
        (0x0014, "PC_REGISTER_AMOUNTS"),
        (0x0008, "PC_OPERATION_ENABLE"),
        (0x100C, "CNA_CONV_CON1"),
        (0x1010, "CNA_CONV_CON2"),
        (0x1014, "CNA_CONV_CON3"),
        (0x1020, "CNA_DATA_SIZE0"),
        (0x1024, "CNA_DATA_SIZE1"),
        (0x1030, "CNA_WEIGHT_SIZE0"),
        (0x1034, "CNA_WEIGHT_SIZE1"),
        (0x1038, "CNA_WEIGHT_SIZE2"),
        (0x1070, "CNA_FEATURE_DATA_ADDR"),
        (0x1110, "CNA_DCOMP_ADDR0"),
        (0x3010, "CORE_MISC_CFG"),
        (0x3014, "CORE_DATAOUT_SIZE_0"),
        (0x3018, "CORE_DATAOUT_SIZE_1"),
        (0x400C, "DPU_FEATURE_MODE_CFG"),
        (0x4010, "DPU_DATA_FORMAT"),
        (0x4020, "DPU_DST_BASE_ADDR"),
        (0x4024, "DPU_DST_SURF_STRIDE"),
        (0x4030, "DPU_DATA_CUBE_WIDTH"),
        (0x4034, "DPU_DATA_CUBE_HEIGHT"),
        (0x403C, "DPU_DATA_CUBE_CHANNEL"),
        (0x4058, "DPU_WDMA_SIZE_0"),
        (0x405C, "DPU_WDMA_SIZE_1"),
        (0x4070, "DPU_EW_CFG"),
        (0x4108, "DPU_LUT_CFG"),
        (0x4110, "DPU_LUT_LE_START"),
        (0x500C, "DPU_RDMA_DATA_CUBE_WIDTH"),
        (0x5010, "DPU_RDMA_DATA_CUBE_HEIGHT"),
        (0x5014, "DPU_RDMA_DATA_CUBE_CHANNEL"),
        (0x5018, "DPU_RDMA_SRC_BASE_ADDR"),
        (0x5044, "DPU_RDMA_FEATURE_MODE_CFG"),
    ]
)

KEY_REG_NAMES = set(KEY_REGISTERS.values())


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_register_names(regmap: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not regmap.exists():
        return names
    pattern = re.compile(r"#define\s+REG_([A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)")
    for match in pattern.finditer(regmap.read_text(encoding="utf-8", errors="ignore")):
        names.setdefault(int(match.group(2), 16), match.group(1))
    return names


def decode_regcmd(hex_blob: str) -> list[dict]:
    raw = bytes.fromhex(hex_blob or "")
    ops: list[dict] = []
    for index in range(len(raw) // 8):
        word = int.from_bytes(raw[index * 8 : index * 8 + 8], "little")
        op = (word >> 48) & 0xFFFF
        value = (word >> 16) & 0xFFFFFFFF
        reg = word & 0xFFFF
        page = reg >> 12
        ops.append(
            {
                "npuop_index": index,
                "op": op,
                "value": value,
                "reg": reg,
                "module": MODULE_BY_PAGE.get(page, f"PAGE_0x{page:x}"),
            }
        )
    return ops


def fetch_entries(amount: int) -> int:
    # Driver: (amount + 4 + scale - 1) / scale - 1 with scale=2, written as a
    # fetch window.  Expressed in entries this is round_up(amount + 4, 2).
    return ((amount + 4 + 1) // 2) * 2


def module_counts(ops: Iterable[dict]) -> dict[str, int]:
    return dict(Counter(op["module"] for op in ops))


def ordered_modules(ops: Iterable[dict]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for op in ops:
        seen.setdefault(op["module"], None)
    return list(seen)


def module_list(counts: dict[str, int]) -> str:
    return ",".join(name for name in MODULE_ORDER if counts.get(name, 0))


def task_class(counts: dict[str, int], n_lut: int) -> str:
    if n_lut >= 100:
        return "lut_table_load"
    if counts.get("CNA", 0):
        return "cna_core_dpu"
    if counts.get("DPU", 0) or counts.get("DPU_RDMA", 0):
        return "dpu_dma"
    if counts.get("PC", 0):
        return "pc_only"
    return "other"


def key_regs_text(ops: Iterable[dict]) -> str:
    values: dict[str, int] = {}
    for op in ops:
        name = KEY_REGISTERS.get(op["reg"])
        if name:
            # Last write wins: this matches the final value in the regcmd block.
            values[name] = op["value"]
    return ";".join(
        f"{name}=0x{values[name]:08x}" for name in KEY_REG_NAMES if name in values
    )


def op6_annotation(op_idx: int, local_task: int) -> tuple[str, str, str]:
    """Return (stage, head_group, head_task_order) for the 401-task op6 chain."""
    if op_idx != 6:
        if op_idx in (3, 4, 5):
            return f"pre_reformat_op{op_idx}", "", ""
        if op_idx == 7:
            return "post_reformat_op7", "", ""
        return f"op{op_idx}", "", ""
    if local_task == 0:
        return "sdpa_prologue", "prologue", "0"
    zero_based = local_task - 1
    head = zero_based // 25
    rel = zero_based % 25
    if 0 <= head <= 15:
        return "sdpa_head_group", f"head{head}", str(rel)
    return "sdpa_tail", "tail", str(local_task)


def build_actual_chain(events: list[dict]):
    submits = {e["sequence"]: e for e in events if e.get("event") == "submit"}
    tasks = {
        (e["sequence"], e["task_index"]): e
        for e in events
        if e.get("event") == "task"
    }
    regcmds = {
        (e["sequence"], e["task_index"]): e
        for e in events
        if e.get("event") == "regcmd"
    }

    actual: list[dict] = []
    seen_descriptor: dict[int, dict] = {}
    for sequence in sorted(submits):
        submit = submits[sequence]
        slots = submit.get("subcore_tasks") or []
        if not slots or not slots[0][1]:
            raise SystemExit(f"submit {sequence}: missing effective subcore slot")
        slot_start, slot_count = int(slots[0][0]), int(slots[0][1])
        capture_start = int(submit.get("task_start", 0))

        # The hook captures a local array beginning at submit.task_start.  The
        # effective slot is an index into the global descriptor array.
        local_start = slot_start - capture_start
        if local_start < 0:
            raise SystemExit(
                f"submit {sequence}: slot start {slot_start} precedes"
                f" capture start {capture_start}"
            )

        for offset in range(slot_count):
            local_index = local_start + offset
            key = (sequence, local_index)
            if key not in tasks or key not in regcmds:
                raise SystemExit(
                    f"submit {sequence}: missing task/regcmd at local index"
                    f" {local_index} (offset {offset})"
                )
            descriptor_index = slot_start + offset
            if descriptor_index in seen_descriptor:
                raise SystemExit(
                    f"duplicate executed descriptor {descriptor_index}:"
                    f" {seen_descriptor[descriptor_index]['sequence']}/{seen_descriptor[descriptor_index]['task_index']}"
                    f" and {sequence}/{local_index}"
                )
            entry = {
                "sequence": sequence,
                "submit": submit,
                "task": tasks[key],
                "regcmd": regcmds[key],
                "local_task": local_index,
                "submit_task_order": offset,
                "descriptor_index": descriptor_index,
            }
            seen_descriptor[descriptor_index] = entry
            actual.append(entry)

    actual.sort(key=lambda e: e["descriptor_index"])
    return submits, actual


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_hex(value: int | None) -> str:
    return "" if value is None else f"0x{value:08x}"



def load_probe_csv(path: Path) -> dict[int, list[int]]:
    """Load raw prefix-probe rows and return L -> successful hw_elapse list."""
    by_l: dict[int, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            fields = [field.strip() for field in line.strip().split(",")]
            if len(fields) < 4 or fields[0] in ("rep", "#rep"):
                continue
            try:
                _rep, length, rc, hw = fields[:4]
                if int(rc) != 0 or hw in ("", "NA"):
                    continue
                by_l[int(length)].append(int(hw))
            except ValueError:
                continue
    return by_l


def build_head_timing(task_rows: list[dict], probe_by_l: dict[int, list[int]]) -> list[dict]:
    """Turn prefix-probe cumulative T(L) into per-head segment estimates."""
    if not probe_by_l:
        return []

    def cumulative_us(length: int) -> tuple[float, int, int, int]:
        values = probe_by_l.get(length, [])
        if not values:
            return (float("nan"), 0, 0, 0)
        return (
            statistics.median(values) / 1000.0,
            len(values),
            min(values) / 1000.0,
            max(values) / 1000.0,
        )

    rows: list[dict] = []
    prologue_us, prologue_n, prologue_min, prologue_max = cumulative_us(1)
    if prologue_n:
        rows.append(
            {
                "segment": "prologue_task0",
                "head": "",
                "start_L_exclusive": 0,
                "end_L_inclusive": 1,
                "start_us": 0.0,
                "end_us": round(prologue_us, 3),
                "segment_us": round(prologue_us, 3),
                "n_probe_end": prologue_n,
                "end_min_us": round(prologue_min, 3),
                "end_max_us": round(prologue_max, 3),
                "task_count": 1,
                "cna_tasks": 0,
                "n_cna": 0,
                "n_core": 0,
                "n_dpu": 0,
                "n_dpu_rdma": 0,
                "n_lut_0x4104": 0,
                "notes": "op6 task 0; one initial DPU/DPU_RDMA configuration block",
            }
        )

    previous_us = prologue_us
    previous_n = prologue_n
    for head in range(16):
        start_L = 1 if head == 0 else 26 + 25 * (head - 1)
        end_L = 26 + 25 * head
        start_us = previous_us
        end_us, end_n, end_min, end_max = cumulative_us(end_L)
        if not end_n or not previous_n:
            previous_us = end_us
            previous_n = end_n
            continue
        head_rows = [
            row
            for row in task_rows
            if row["op_idx"] == 6 and row["head_group"] == f"head{head}"
        ]
        counts = {
            key: sum(int(row[key]) for row in head_rows)
            for key in ("n_cna", "n_core", "n_dpu", "n_dpu_rdma", "n_lut_0x4104")
        }
        rows.append(
            {
                "segment": f"head{head}",
                "head": head,
                "start_L_exclusive": start_L,
                "end_L_inclusive": end_L,
                "start_us": round(start_us, 3),
                "end_us": round(end_us, 3),
                "segment_us": round(end_us - start_us, 3),
                "n_probe_end": min(previous_n, end_n),
                "end_min_us": round(end_min, 3),
                "end_max_us": round(end_max, 3),
                "task_count": len(head_rows),
                "cna_tasks": sum(1 for row in head_rows if int(row["n_cna"]) > 0),
                **counts,
                "notes": (
                    "head0 also contains the 954-entry shared softmax LUT load"
                    if head == 0
                    else ""
                ),
            }
        )
        previous_us, previous_n = end_us, end_n
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--regmap",
        type=Path,
        default=Path(__file__).resolve().parent / "rnpu_regmap.h",
        help="register name table (default: sibling rnpu_regmap.h)",
    )
    parser.add_argument(
        "--probe-csv",
        type=Path,
        help="optional CSV: rep,L,rc,hw_elapse_time_ns from EPERF_RKNPU_PROBE",
    )
    args = parser.parse_args()

    events = load_events(args.trace)
    register_names = load_register_names(args.regmap)
    submits, actual = build_actual_chain(events)
    if not actual:
        raise SystemExit("no effective tasks selected")

    args.out.mkdir(parents=True, exist_ok=True)
    base_wall = min(int(e["submit"].get("wall_start_ns", 0)) for e in actual)
    total_hw = sum(int(e["submit"].get("hw_elapse_time_ns", 0)) for e in actual)

    task_rows: list[dict] = []
    npuop_rows: list[dict] = []
    per_submit_counts: dict[int, Counter] = {}
    task_class_counts: Counter = Counter()
    stage_counts: Counter = Counter()

    for order, entry in enumerate(actual):
        submit = entry["submit"]
        task = entry["task"]
        regcmd = entry["regcmd"]
        ops = decode_regcmd(regcmd.get("hex", ""))
        counts = module_counts(ops)
        first_touch = ordered_modules(ops)
        n_lut = sum(1 for op in ops if op["reg"] == 0x4104)
        fetch_n = fetch_entries(int(task.get("regcfg_amount", 0)))
        expected_bytes = fetch_n * 8
        captured_bytes = int(regcmd.get("bytes", 0))
        truncated = captured_bytes < expected_bytes
        stage, head_group, head_task_order = op6_annotation(
            int(task.get("op_idx", -1)), int(entry["local_task"])
        )
        cls = task_class(counts, n_lut)
        task_class_counts[cls] += 1
        stage_counts[stage] += 1
        per_submit_counts.setdefault(entry["sequence"], Counter()).update(counts)

        task_row = {
            "exec_order": order,
            "descriptor_index": entry["descriptor_index"],
            "submit_seq": entry["sequence"],
            "submit_task_order": entry["submit_task_order"],
            "local_task_index": entry["local_task"],
            "stage": stage,
            "head_group": head_group,
            "head_task_order": head_task_order,
            "op_idx": task.get("op_idx", ""),
            "enable_mask": fmt_hex(int(task.get("enable_mask", 0))),
            "regcfg_amount": task.get("regcfg_amount", ""),
            "fetch_entries": fetch_n,
            "fetch_bytes": expected_bytes,
            "captured_bytes": captured_bytes,
            "capture_truncated": int(truncated),
            "regcmd_addr": task.get("regcmd_addr", ""),
            "modules_touched": module_list(counts),
            "modules_first_touch_order": ",".join(first_touch),
            "n_pc": counts.get("PC", 0),
            "n_cna": counts.get("CNA", 0),
            "n_core": counts.get("CORE", 0),
            "n_dpu": counts.get("DPU", 0),
            "n_dpu_rdma": counts.get("DPU_RDMA", 0),
            "n_ppu": counts.get("PPU", 0),
            "n_ppu_rdma": counts.get("PPU_RDMA", 0),
            "n_lut_0x4104": n_lut,
            "task_class": cls,
            "key_registers": key_regs_text(ops),
            "submit_wall_start_rel_ns": int(submit.get("wall_start_ns", 0)) - base_wall,
            "submit_wall_end_rel_ns": int(submit.get("wall_end_ns", 0)) - base_wall,
            "submit_wall_elapsed_ns": submit.get("wall_elapsed_ns", ""),
            "submit_hw_elapse_ns": submit.get("hw_elapse_time_ns", ""),
            "time_scope": "submit_only",
        }
        task_rows.append(task_row)

        for op in ops:
            reg = op["reg"]
            npuop_rows.append(
                {
                    "exec_order": order,
                    "descriptor_index": entry["descriptor_index"],
                    "submit_seq": entry["sequence"],
                    "submit_task_order": entry["submit_task_order"],
                    "stage": stage,
                    "head_group": head_group,
                    "head_task_order": head_task_order,
                    "op_idx": task.get("op_idx", ""),
                    "npuop_index": op["npuop_index"],
                    "op": f"0x{op['op']:04x}",
                    "module": op["module"],
                    "reg_offset": f"0x{reg:04x}",
                    "reg_name": register_names.get(reg, KEY_REGISTERS.get(reg, "")),
                    "value": f"0x{op['value']:08x}",
                    "value_u32": op["value"],
                    "submit_wall_start_rel_ns": task_row["submit_wall_start_rel_ns"],
                    "submit_wall_end_rel_ns": task_row["submit_wall_end_rel_ns"],
                    "submit_hw_elapse_ns": task_row["submit_hw_elapse_ns"],
                    "time_scope": "submit_only",
                }
            )

    probe_by_l: dict[int, list[int]] = {}
    head_timing: list[dict] = []
    if args.probe_csv:
        probe_by_l = load_probe_csv(args.probe_csv)
        head_timing = build_head_timing(task_rows, probe_by_l)
        if probe_by_l:
            with (args.out / "probe_prefix_raw.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow(["L", "n", "median_hw_ns", "min_hw_ns", "max_hw_ns"])
                for length in sorted(probe_by_l):
                    values = probe_by_l[length]
                    writer.writerow(
                        [
                            length,
                            len(values),
                            int(statistics.median(values)),
                            min(values),
                            max(values),
                        ]
                    )

    write_csv(
        args.out / "sdpa_task_timeline.csv",
        list(task_rows[0].keys()),
        task_rows,
    )
    write_csv(
        args.out / "sdpa_npuop_timeline.csv",
        list(npuop_rows[0].keys()),
        npuop_rows,
    )
    if head_timing:
        write_csv(
            args.out / "sdpa_head_timing.csv",
            list(head_timing[0].keys()),
            head_timing,
        )

    # head1 is the first fully repeated 25-task pattern; head0 carries extra
    # setup and the shared LUT load.  Keep a compact representative pattern.
    head1_rows = [
        row
        for row in task_rows
        if row["op_idx"] == 6 and row["head_group"] == "head1"
    ]
    if head1_rows:
        with (args.out / "sdpa_head1_pattern.txt").open(
            "w", encoding="utf-8"
        ) as fh:
            fh.write(
                "# representative op6 head pattern (head1): task order as emitted;\n"
                "# modules are derived from actual register offsets.\n"
            )
            for row in head1_rows:
                fh.write(
                    f"rel={row['head_task_order']:>2} "
                    f"task={row['submit_task_order']:>3} "
                    f"op={row['op_idx']} en={row['enable_mask']} "
                    f"amt={row['regcfg_amount']:>3} fetch={row['fetch_bytes']:>3}B "
                    f"mods={row['modules_touched']} "
                    f"CNA={row['n_cna']} CORE={row['n_core']} "
                    f"DPU={row['n_dpu']} DPU_RDMA={row['n_dpu_rdma']} "
                    f"LUT={row['n_lut_0x4104']} class={row['task_class']}\n"
                )

    # Compact human-readable order.  The time prefix is the containing submit's
    # wall window, not a per-task timestamp.
    with (args.out / "sdpa_task_sequence.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            "# time=submit wall window [start,end) relative to first submit; "
            "hw=whole-submit driver hw_elapse; no per-task timestamp exists\n"
        )
        for row in task_rows:
            fh.write(
                f"#{int(row['exec_order']):04d} "
                f"S{row['submit_seq']} "
                f"t={row['submit_task_order']:04d} "
                f"{row['stage']}"
                f"{'/' + str(row['head_group']) if row['head_group'] else ''}"
                f"{'/rel' + str(row['head_task_order']) if row['head_task_order'] else ''} "
                f"op={row['op_idx']} en={row['enable_mask']} "
                f"amt={row['regcfg_amount']:>4} fetch={row['fetch_bytes']:>4}B "
                f"mods={row['modules_touched'] or '-'} cls={row['task_class']} "
                f"wall=[+{row['submit_wall_start_rel_ns']},"
                f"+{row['submit_wall_end_rel_ns']})ns "
                f"hw={row['submit_hw_elapse_ns']}ns"
                f"{' truncated' if row['capture_truncated'] else ''}\n"
            )

    submit_summary: list[dict] = []
    for sequence in sorted(submits):
        submit = submits[sequence]
        selected = [row for row in task_rows if row["submit_seq"] == sequence]
        counts = per_submit_counts.get(sequence, Counter())
        submit_summary.append(
            {
                "submit_seq": sequence,
                "flags": submit.get("flags"),
                "core_mask": submit.get("core_mask"),
                "core_mask_after": submit.get("core_mask_after"),
                "task_start": submit.get("task_start"),
                "task_number_reported": submit.get("task_number"),
                "subcore_tasks": submit.get("subcore_tasks"),
                "effective_task_count": len(selected),
                "first_descriptor_index": selected[0]["descriptor_index"] if selected else None,
                "last_descriptor_index": selected[-1]["descriptor_index"] if selected else None,
                "wall_start_rel_ns": int(submit.get("wall_start_ns", 0)) - base_wall,
                "wall_end_rel_ns": int(submit.get("wall_end_ns", 0)) - base_wall,
                "wall_elapsed_ns": submit.get("wall_elapsed_ns"),
                "hw_elapse_ns": submit.get("hw_elapse_time_ns"),
                "n_cna": counts.get("CNA", 0),
                "n_core": counts.get("CORE", 0),
                "n_dpu": counts.get("DPU", 0),
                "n_dpu_rdma": counts.get("DPU_RDMA", 0),
                "n_ppu": counts.get("PPU", 0),
                "n_pc": counts.get("PC", 0),
                "n_lut_0x4104": sum(
                    int(row["n_lut_0x4104"]) for row in selected
                ),
                "task_classes": dict(
                    Counter(row["task_class"] for row in selected)
                ),
            }
        )

    summary = {
        "trace": str(args.trace),
        "timing_limitation": (
            "Trace has submit wall windows and per-submit hw_elapse only. "
            "Tasks are ordered but have no individual timestamp."
        ),
        "raw_task_events": sum(1 for e in events if e.get("event") == "task"),
        "raw_regcmd_events": sum(1 for e in events if e.get("event") == "regcmd"),
        "effective_task_count": len(actual),
        "effective_npuop_count": len(npuop_rows),
        "effective_fetch_bytes": sum(int(row["fetch_bytes"]) for row in task_rows),
        "captured_regcmd_bytes": sum(int(row["captured_bytes"]) for row in task_rows),
        "truncated_tasks": [
            {
                "exec_order": row["exec_order"],
                "submit_seq": row["submit_seq"],
                "submit_task_order": row["submit_task_order"],
                "expected_bytes": row["fetch_bytes"],
                "captured_bytes": row["captured_bytes"],
            }
            for row in task_rows
            if row["capture_truncated"]
        ],
        "sum_submit_hw_elapse_ns": total_hw,
        "task_classes": dict(task_class_counts),
        "stages": dict(stage_counts),
        "submits": submit_summary,
        "prefix_probe": {
            "csv": str(args.probe_csv) if args.probe_csv else None,
            "L_values": sorted(probe_by_l) if probe_by_l else [],
            "head_timing": head_timing,
        },
        "csv_columns": {
            "task": list(task_rows[0].keys()),
            "npuop": list(npuop_rows[0].keys()),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Markdown summary with a few representative rows.
    def task_line(row: dict) -> str:
        head = (
            f" / {row['head_group']}/rel{row['head_task_order']}"
            if row["head_group"]
            else ""
        )
        trunc = " **truncated**" if row["capture_truncated"] else ""
        return (
            f"| {row['exec_order']} | {row['submit_seq']} | {row['submit_task_order']} | "
            f"{row['stage']}{head} | {row['op_idx']} | {row['enable_mask']} | "
            f"{row['regcfg_amount']} | {row['fetch_bytes']} | "
            f"{row['modules_touched'] or '-'} | {row['task_class']}{trunc} |"
        )

    with (args.out / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# 单个 SDPA 的 regcmd 顺序与模块时间线\n\n")
        fh.write(f"- trace: `{args.trace}`\n")
        fh.write(f"- 原始 task event: **{summary['raw_task_events']}**\n")
        fh.write(f"- 原始 regcmd event: **{summary['raw_regcmd_events']}**\n")
        fh.write(
            f"- 按 `subcore_tasks[0]` 归一后的实际执行 task: "
            f"**{summary['effective_task_count']}**\n"
        )
        fh.write(f"- 实际 NPUOP 写: **{summary['effective_npuop_count']}**\n")
        fh.write(
            f"- 实际 fetch regcmd: **{summary['effective_fetch_bytes']} B**；"
            f"实际捕获: **{summary['captured_regcmd_bytes']} B**\n\n"
        )
        fh.write("## 时间口径\n\n")
        fh.write(
            "现有 trace **没有 per-task / per-NPUOP 时间戳**。"
            "驱动只报告每个 SUBMIT 的 `hw_elapse_time_ns`，"
            "hook 只给 SUBMIT 的 `wall_start_ns/wall_end_ns`。"
            "因此本文按 task 顺序给出模块，时间只精确到所在 SUBMIT 的时间窗；"
            "不能把同一 submit 内的 401 个 task 说成 401 个已知时刻。\n\n"
        )
        fh.write(
            "`submit_hw_elapse_ns` 是驱动 `commit -> IRQ` 的整笔 submit 时长，"
            "包含 PC 按链取指、寄存器配置、CNA/CORE/DPU/DPU_RDMA 的流水与 stall，"
            "不是各 task 时间之和。\n\n"
        )
        fh.write("## Submit 时间窗\n\n")
        fh.write(
            "| submit | flags | core_mask(before→after) | slot[0] | "
            "reported task_number | effective task | descriptor range | "
            "wall window (rel) | hw_elapse |\n"
            "|---:|---:|---|---:|---:|---:|---|---:|---:|\n"
        )
        for s in submit_summary:
            slot = (s["subcore_tasks"] or [[None, None]])[0]
            fh.write(
                f"| {s['submit_seq']} | 0x{s['flags']:x} | "
                f"{s['core_mask']} → {s['core_mask_after']} | "
                f"[{slot[0]},{slot[1]}] | {s['task_number_reported']} | "
                f"{s['effective_task_count']} | "
                f"{s['first_descriptor_index']}..{s['last_descriptor_index']} | "
                f"[+{s['wall_start_rel_ns']}, +{s['wall_end_rel_ns']}) ns | "
                f"{s['hw_elapse_ns']} ns |\n"
            )
        fh.write("\n## 每个 submit 触达的模块\n\n")
        fh.write(
            "| submit | CNA | CORE | DPU | DPU_RDMA | PPU | PC | "
            "`0x4104` LUT 写 | task class |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for s in submit_summary:
            fh.write(
                f"| {s['submit_seq']} | {s['n_cna']} | {s['n_core']} | "
                f"{s['n_dpu']} | {s['n_dpu_rdma']} | {s['n_ppu']} | "
                f"{s['n_pc']} | {s['n_lut_0x4104']} | "
                f"{s['task_classes']} |\n"
            )
        if head_timing:
            fh.write("\n## Prefix probe 实测时间（单 submit 内累计）\n\n")
            fh.write(
                "以下不是逐 task 时间戳。做法是每次 fresh process 把 op6 链截到前 L 个 task，"
                "读驱动 `hw_elapse_time_ns`，得到累计 `T(L)`；再对相邻 head 边界做差分。"
                "每组尾部点用多次运行的 median，差分是 median 的差。\n\n"
            )
            fh.write(
                "| segment | L span | time window (us) | segment (us) | tasks | CNA tasks | "
                "CNA writes | CORE | DPU | DPU_RDMA | LUT writes |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for row in head_timing:
                fh.write(
                    f"| {row['segment']} | ({row['start_L_exclusive']},{row['end_L_inclusive']}] | "
                    f"{row['start_us']:.3f}..{row['end_us']:.3f} | {row['segment_us']:.3f} | "
                    f"{row['task_count']} | {row['cna_tasks']} | {row['n_cna']} | "
                    f"{row['n_core']} | {row['n_dpu']} | {row['n_dpu_rdma']} | "
                    f"{row['n_lut_0x4104']} |\n"
                )
            fh.write(
                "\nhead0 的 385 us 中包含共享 LUT 整表加载；因此 head0 明显长于后续约 "
                "290–380 us 的 head。head 之间的差异仍含温度/频率/调度噪声。\n"
            )
            fh.write(
                "`probe_prefix_raw.csv` 保存每个 L 的 median/min/max；"
                "原始重复次数和值由 `--probe-csv` 输入保留在该文件的上游。\n"
            )
        fh.write("\n## 关键顺序片段\n\n")
        fh.write(
            "| exec# | submit | task# | stage/head | op | enable | amount | "
            "fetch B | modules touched | class |\n"
            "|---:|---:|---:|---|---:|---:|---:|---:|---|---|\n"
        )
        sample_indices = list(range(min(18, len(task_rows))))
        # Include the start of each head and the LUT load so the table is useful
        # without dumping all 401 rows into Markdown.
        for head in range(16):
            local = 1 + head * 25
            match = next(
                (i for i, r in enumerate(task_rows) if r["submit_seq"] == 2 and r["local_task_index"] == local),
                None,
            )
            if match is not None:
                sample_indices.append(match)
        lut_index = next(
            (
                i
                for i, r in enumerate(task_rows)
                if r["submit_seq"] == 2 and r["n_lut_0x4104"] >= 100
            ),
            None,
        )
        if lut_index is not None:
            sample_indices.append(lut_index)
        sample_indices = sorted(set(sample_indices))
        for index in sample_indices:
            fh.write(task_line(task_rows[index]) + "\n")
        fh.write("\n## 输出文件\n\n")
        fh.write(
            "- `sdpa_task_timeline.csv`：每个实际执行 task 一行；"
            "含模块计数、关键寄存器、所在 submit 时间窗。\n"
            "- `sdpa_npuop_timeline.csv`：每个实际 NPUOP 写入一行；"
            "严格保留 task 内与 task 间顺序。\n"
            "- `sdpa_task_sequence.txt`：紧凑的人读顺序。\n"
            "- `sdpa_head_timing.csv`：有 prefix probe 时的 head 级实测时间。\n"
            "- `sdpa_head1_pattern.txt`：一个可代表 head1–15 的 25-task 顺序模板。\n"
            "- `probe_prefix_raw.csv`：每个 L 的 median/min/max。\n"
            "- `summary.json`：机器可读汇总。\n\n"
        )
        fh.write("## 注意事项\n\n")
        fh.write(
            "- `enable_mask` 是 toolkit/runtime 的记账字段，驱动不消费；"
            "本表的 `modules_touched` 由实际 regcmd 寄存器 offset 推导。\n"
            "- `task_number=3N` 与 `subcore_tasks` 的三个相同槽位会放大原始抓取；"
            "本表只取实际 AUTO 会消费的 `slot[0] = [start,N]`。\n"
            "- 截断的 LUT task 在表中标记 `capture_truncated`；其末尾 trailer "
            "不在 8192 B 捕获窗口内。\n"
        )

    print(f"wrote {args.out}")
    print(f"effective tasks: {len(actual)}")
    print(f"effective NPUOPs: {len(npuop_rows)}")
    for s in submit_summary:
        print(
            f"submit {s['submit_seq']}: effective={s['effective_task_count']} "
            f"reported={s['task_number_reported']} slot={s['subcore_tasks'][0]} "
            f"hw={s['hw_elapse_ns']}ns"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
