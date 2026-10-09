#!/usr/bin/env python3
"""Per-operator analysis of an EPERF_RKNPU_IOCTL_TRACE capture.

Joins the captured `task.op_idx` with the compiled RKNN node table and reports,
per operator: task count, regcmd total/median bytes, number of distinct regcmd
payloads (templates), and the dominant first word.

The node table is extracted from the .rknn file by reading the ordered
`Type:/path` strings the compiler emits; this was validated by matching the
trace `op_idx` range exactly (2..21 for the step.rknn reference model).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def extract_ops(rknn: Path) -> dict[int, str]:
    data = rknn.read_bytes()
    runs = [r.decode("ascii", "ignore") for r in re.findall(rb"[\x20-\x7e]{3,}", data)]
    ordered: list[str] = []
    seen: set[str] = set()
    for text in runs:
        if re.match(r"^[A-Z][A-Za-z0-9_]*:.+", text) and "##" not in text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return {index: name for index, name in enumerate(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--rknn", type=Path, required=True)
    args = parser.parse_args()

    ops = extract_ops(args.rknn)
    tasks: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for line in args.trace.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "regcmd":
            tasks[record["op_idx"]].append((record["bytes"], record["hex"]))

    print(f"{'op':>3} {'tasks':>5} {'totB':>7} {'medB':>5} {'uniq':>5}  {'top first-word':>14}  op")
    for op in sorted(tasks):
        entries = tasks[op]
        sizes = sorted(entry[0] for entry in entries)
        unique = len({hashlib.sha256(bytes.fromhex(entry[1])).hexdigest() for entry in entries})
        first_words = collections.Counter(
            int(entry[1][0:8], 16) for entry in entries if len(entry[1]) >= 8)
        top = f"0x{first_words.most_common(1)[0][0]:08x}" if first_words else "-"
        print(f"{op:>3} {len(entries):>5} {sum(sizes):>7} {sizes[len(sizes)//2]:>5} "
              f"{unique:>5}  {top:>14}  {ops.get(op, '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
