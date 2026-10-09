#!/usr/bin/env python3
"""Summarize an `EPERF_RKNPU_IOCTL_TRACE` JSONL capture.

Answers whether the NPU command stream is observable from userspace:

  * how many `submit` records and how many `task` records were captured;
  * what fraction of task `regcmd_addr` values could be translated to a mapped
    host buffer (`resolved`, and via which range);
  * `regcmd` size distribution and first-word (opcode-ish) histogram;
  * per-submit task `op_idx` sequence joined from canonical task events;
  * capture-integrity checks for orphan/missing/partial submit/task records.

This is a static parse of the hook output; it does not decode the RKNPU ISA.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    records = []
    stream_context = (
        gzip.open(path, "rt", encoding="utf-8", errors="strict")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8", errors="strict")
    )
    with stream_context as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON record at {path}:{line_number}"
                ) from exc
    return records


def first_word(hex_text: str) -> int | None:
    if len(hex_text) < 8:
        return None
    return int(hex_text[0:8], 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top-opcodes", type=int, default=12)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when submit/task sequence closure is incomplete",
    )
    parser.add_argument(
        "--strict-effective",
        action="store_true",
        help="fail when captured tasks are not the effective subcore range",
    )
    args = parser.parse_args()

    records = load(args.trace)
    kinds = collections.Counter(record.get("event") for record in records)
    tasks = [r for r in records if r.get("event") == "task"]
    regcmds = [r for r in records if r.get("event") == "regcmd"]
    submits = [r for r in records if r.get("event") == "submit"]
    submit_by_sequence = {int(r["sequence"]): r for r in submits}
    tasks_by_sequence: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    duplicate_task_keys = []
    for task in tasks:
        sequence = int(task["sequence"])
        task_index = int(task["task_index"])
        if task_index in tasks_by_sequence[sequence]:
            duplicate_task_keys.append((sequence, task_index))
        tasks_by_sequence[sequence][task_index] = task

    orphan_task_sequences = sorted(set(tasks_by_sequence) - set(submit_by_sequence))
    count_mismatches = []
    incomplete_submit_sequences = []
    inline_mismatches = []
    joined_op_indices: dict[int, list[int]] = {}
    for sequence, submit_record in submit_by_sequence.items():
        sequence_tasks = tasks_by_sequence.get(sequence, {})
        actual_indices = sorted(sequence_tasks)
        captured_count = int(submit_record.get("captured_task_count", 0))
        if actual_indices != list(range(captured_count)):
            count_mismatches.append(
                (sequence, captured_count, len(actual_indices))
            )
        joined = [
            int(sequence_tasks[index]["op_idx"])
            for index in actual_indices
        ]
        joined_op_indices[sequence] = joined
        inline = submit_record.get("task_op_idx")
        if isinstance(inline, list) and [int(value) for value in inline] != joined:
            inline_mismatches.append(sequence)
        if submit_record.get("task_capture_complete") is False or (
            captured_count != int(submit_record.get("task_number", 0))
        ):
            incomplete_submit_sequences.append(sequence)

    integrity_ok = not (
        orphan_task_sequences
        or duplicate_task_keys
        or count_mismatches
        or inline_mismatches
        or incomplete_submit_sequences
    )

    print(f"records: {len(records)}")
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count}")

    resolved = [t for t in tasks if t.get("resolved")]
    via = collections.Counter(t.get("resolved_via") for t in resolved)
    print(f"\ntask regcmd resolution: {len(resolved)}/{len(tasks)}")
    for name, count in via.most_common():
        print(f"  via {name}: {count}")
    unresolved = collections.Counter(
        (t.get("regcfg_amount"), t.get("regcmd_addr")) for t in tasks
        if not t.get("resolved"))
    if unresolved:
        print(f"  unresolved samples: {list(unresolved)[:5]}")

    if regcmds:
        sizes = sorted(r.get("bytes", 0) for r in regcmds)
        opcodes = collections.Counter(first_word(r.get("hex", "")) for r in regcmds)
        print(f"\nregcmd dumps: {len(regcmds)}")
        print(f"  bytes min/median/max: {sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
        print(f"  top first-words:")
        for word, count in opcodes.most_common(args.top_opcodes):
            print(f"    0x{word:08x}: {count}" if word is not None else f"    n/a: {count}")

    if submits:
        print(f"\nsubmits: {len(submits)}")
        seq = [joined_op_indices.get(int(s["sequence"]), []) for s in submits]
        seq_summary = [
            {
                "count": len(values),
                "head": values[:12],
                "top": collections.Counter(values).most_common(4),
            }
            for values in seq[:3]
        ]
        print(f"  first task_op_idx sequences: {seq_summary}")
        elapse = [s.get("hw_elapse_time_ns", 0) for s in submits]
        print(f"  hw_elapse_time_ns min/max: {min(elapse)}/{max(elapse)}")
    print("\ncapture integrity:")
    print(f"  status: {'passed' if integrity_ok else 'failed'}")
    print(f"  orphan task sequences: {len(orphan_task_sequences)}")
    print(f"  duplicate task keys: {len(duplicate_task_keys)}")
    print(f"  submit/task count mismatches: {len(count_mismatches)}")
    print(f"  inline/task op_idx mismatches: {len(inline_mismatches)}")
    print(f"  incomplete submit captures: {len(incomplete_submit_sequences)}")
    if orphan_task_sequences:
        print(f"  orphan samples: {orphan_task_sequences[:8]}")
    if count_mismatches:
        print(f"  count mismatch samples: {count_mismatches[:8]}")
    if incomplete_submit_sequences:
        print(f"  incomplete samples: {incomplete_submit_sequences[:8]}")

    # Effective subcore range.  COORDINATES: subcore_task[].task_start is an
    # ABSOLUTE index into the task array, while the trace `task_index` is
    # RELATIVE to submit.task_start.  A submit may capture candidate tasks
    # (task_number) that the driver never executes (only the per-core subcore
    # slot runs; core_mask=0 -> driver auto-selects a core).  Correct gate:
    #   effective range unambiguous AND effective_tasks subset-of captured AND
    #   under_capture == 0.  Over-capture (candidates not executed) is allowed.
    eff_ok = True
    eff_ambiguous = []
    over_capture = []
    under_capture = []
    cand_total = 0
    eff_total = 0
    for sequence, submit_record in submit_by_sequence.items():
        tn = int(submit_record.get("task_number", 0))
        ts = int(submit_record.get("task_start", 0))
        cm = int(submit_record.get("core_mask", 0))
        cma_raw = submit_record.get("core_mask_after")
        cma = int(cma_raw) if cma_raw is not None else None
        sc = submit_record.get("subcore_tasks") or []

        def slot_range(i):
            try:
                start, num = int(sc[i][0]), int(sc[i][1])
            except (TypeError, ValueError, IndexError):
                return None
            return (start, start + num) if num > 0 else None

        mask = cma if cma is not None else cm
        ranges = []
        if mask:
            bits = [i for i in range(5) if (mask >> i) & 1]
            # driver indexing: 1/2 cores -> subcore_task[core]; 3 cores -> [2..4]
            slots = [2, 3, 4] if (mask & 0x7) == 0x7 else bits
            for i in slots:
                rr = slot_range(i)
                if rr is not None:
                    ranges.append(rr)
        if not ranges:
            for i in range(len(sc)):
                rr = slot_range(i)
                if rr is not None:
                    ranges.append(rr)
        if not ranges:
            eff_ambiguous.append((sequence, cm, tn))
            continue
        eff = set()
        for start, end in ranges:
            eff.update(range(start, end))
        eff &= set(range(ts, ts + tn))
        captured_abs = {ts + idx for idx in tasks_by_sequence.get(sequence, {})}
        cand_total += len(captured_abs)
        eff_total += len(eff)
        if captured_abs - eff:
            over_capture.append((sequence, len(captured_abs), len(eff)))
        if eff - captured_abs:
            under_capture.append((sequence, len(captured_abs), len(eff)))
    if under_capture or eff_ambiguous:
        eff_ok = False
    print("\neffective subcore range:")
    print(f"  status: {'passed' if eff_ok else 'failed'}")
    print(f"  captured candidates total: {cand_total}")
    print(f"  effective (executed) total: {eff_total}")
    print(f"  over-capture (candidates not executed): {len(over_capture)} submits")
    print(f"  under-capture (effective missing from capture): {len(under_capture)} submits")
    print(f"  ambiguous without subcore info: {len(eff_ambiguous)}")
    if under_capture:
        print(f"  under-capture samples: {under_capture[:8]}")
    if eff_ambiguous:
        print(f"  ambiguous samples: {eff_ambiguous[:8]}")
    if args.strict_effective and not eff_ok:
        return 1
    return 1 if args.strict and not integrity_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
