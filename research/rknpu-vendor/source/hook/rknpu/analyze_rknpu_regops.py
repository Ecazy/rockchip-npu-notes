#!/usr/bin/env python3
"""Decode captured RKNPU regcmd as NPUOP register writes.

The RK3588 NPU command stream ("regcmd") is a list of 64-bit NPUOP words:

    NPUOP(op, value, reg) = (op & 0xffff) << 48 | (value & 0xffffffff) << 16 | (reg & 0xffff)

with op = block | 0x01 (PC=0x0101, CNA=0x0201, CORE=0x0801, DPU=0x1001,
DPU_RDMA=0x2001, PPU=0x4001, PPU_RDMA=0x8001) and reg a register offset such as
CNA_CONV_CON1=0x100c or DPU_S_POINTER=0x4004.

Register/op definitions come from the community reverse engineering:
  https://github.com/mtx512/rk3588-npu  (include/npu_hw.h, npu_cna.h, npu_dpu.h)
  https://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html

Output is per-operator block usage (how many NPUOP entries per CNA/DPU/CORE/PPU)
and the most-written registers.  Alignment/entry-size is not yet fully proven:
most captures are 4 + 8*n bytes, so a 4-byte header/tail is suspected.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

BLOCKS = {
    0x0100: "PC", 0x0200: "CNA", 0x0800: "CORE", 0x1000: "DPU",
    0x2000: "DPU_RDMA", 0x4000: "PPU", 0x8000: "PPU_RDMA",
}


def decode(blob: bytes):
    count = len(blob) // 8
    entries = []
    for index in range(count):
        word = struct.unpack_from("<Q", blob, index * 8)[0]
        entries.append(((word >> 48) & 0xffff, (word >> 16) & 0xffffffff, word & 0xffff))
    return entries, len(blob) % 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top-registers", type=int, default=6)
    args = parser.parse_args()

    blocks: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    registers: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    tails: collections.Counter = collections.Counter()
    for line in args.trace.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "regcmd":
            continue
        entries, tail = decode(bytes.fromhex(record["hex"]))
        tails[tail] += 1
        for operation, _value, register in entries:
            block = operation & 0xff00
            blocks[record["op_idx"]][BLOCKS.get(block, f"0x{block:04x}")] += 1
            registers[record["op_idx"]][f"0x{register:04x}"] += 1

    print("tail remainder histogram:", dict(tails))
    for op in sorted(blocks):
        summary = ", ".join(f"{name}={count}" for name, count in blocks[op].most_common())
        top = ", ".join(f"{reg}x{n}" for reg, n in registers[op].most_common(args.top_registers))
        print(f"op {op:2d}: {summary}")
        print(f"        regs: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
