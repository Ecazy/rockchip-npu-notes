#!/usr/bin/env python3
"""Field-level decode of captured RKNPU regcmd.

Decodes the 64-bit NPUOP register writes and splits each written value into the
named bitfields from the Mesa rules-ng-ng register map
(`docs/references/rknpu/rknpu-register-fields.csv`).

Usage:
  analyze_rknpu_regdecode.py <trace.jsonl> [--op IDX] [--template BYTES]
                             [--fields] [--csv OUT]

Without --op/--template it prints one representative regcmd per distinct byte
length (template).  --fields expands every register to its named fields.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTERS_CSV = ROOT / "docs" / "references" / "rknpu" / "rknpu-registers.csv"
FIELDS_CSV = ROOT / "docs" / "references" / "rknpu" / "rknpu-register-fields.csv"

BLOCKS = {
    0x0100: "PC", 0x0200: "CNA", 0x0800: "CORE", 0x1000: "DPU",
    0x2000: "DPU_RDMA", 0x4000: "PPU", 0x8000: "PPU_RDMA",
}

# Registers worth surfacing with decoded fields.
INTERESTING = {
    "CNA_CONV_CON1", "CNA_DATA_SIZE0", "CNA_DATA_SIZE1", "CNA_WEIGHT_SIZE2",
    "CNA_CBUF_CON0", "CNA_FEATURE_DATA_ADDR", "CNA_DCOMP_ADDR0",
    "CORE_MISC_CFG", "CORE_DATAOUT_SIZE_0", "CORE_DATAOUT_SIZE_1",
    "DPU_DATA_FORMAT", "DPU_EW_CFG", "DPU_DATA_CUBE_WIDTH", "DPU_DATA_CUBE_HEIGHT",
    "DPU_DATA_CUBE_CHANNEL", "DPU_DST_BASE_ADD", "DPU_DST_SURF_STRIDE",
    "DPU_WDMA_SIZE_0", "DPU_WDMA_SIZE_1", "DPU_LUT_CFG", "DPU_LUT_INFO",
    "DPU_LUT_LE_START", "DPU_LUT_LE_END", "DPU_LUT_LO_START", "DPU_LUT_LO_END",
    "PC_OPERATION_ENABLE",
}


def load_register_names() -> dict[int, str]:
    with REGISTERS_CSV.open(encoding="utf-8", newline="") as stream:
        return {int(row["offset"], 16): row["register"] for row in csv.DictReader(stream)}


def load_fields() -> dict[str, list[tuple[str, int, int]]]:
    result: dict[str, list[tuple[str, int, int]]] = collections.defaultdict(list)
    with FIELDS_CSV.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["shift"]:
                continue
            result[row["register"]].append((row["field"], int(row["mask"], 16), int(row["shift"])))
    return result


def decode_entries(hex_text: str):
    blob = bytes.fromhex(hex_text)
    for index in range(len(blob) // 8):
        word = struct.unpack_from("<Q", blob, index * 8)[0]
        yield (word >> 48) & 0xffff, (word >> 16) & 0xffffffff, word & 0xffff


def field_values(register: str, value: int, fields) -> list[str]:
    out = []
    for name, mask, shift in fields.get(register, []):
        if name.startswith("RESERVED"):
            continue
        out.append(f"{name}={(value & mask) >> shift}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--op", type=int, default=None)
    parser.add_argument("--template", type=int, default=None)
    parser.add_argument("--fields", action="store_true", help="expand every register")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    reg_names = load_register_names()
    fields = load_fields()
    records = [json.loads(line) for line in
               args.trace.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]

    selected: list[dict] = []
    if args.op is not None:
        selected = [r for r in records if r.get("event") == "regcmd" and r["op_idx"] == args.op][:1]
    elif args.template is not None:
        selected = [r for r in records if r.get("event") == "regcmd" and r["bytes"] == args.template][:1]
    else:
        seen: set[int] = set()
        for r in records:
            if r.get("event") == "regcmd" and r["bytes"] not in seen:
                seen.add(r["bytes"])
                selected.append(r)

    csv_rows = []
    for record in selected:
        print(f"=== op_idx={record['op_idx']} bytes={record['bytes']} "
              f"regcfg_amount={record['regcfg_amount']} ===")
        for operation, value, register in decode_entries(record["hex"]):
            name = reg_names.get(register, f"0x{register:04x}")
            block = BLOCKS.get(operation & 0xff00, f"0x{operation & 0xff00:04x}")
            fields_text = field_values(name, value, fields)
            if not args.fields and name not in INTERESTING and not fields_text:
                continue
            suffix = "  " + " ".join(fields_text) if fields_text else ""
            print(f"  {block:7s} {name:26s} = 0x{value:08x}{suffix}")
            for field in (fields_text or [""]):
                csv_rows.append({"op_idx": record["op_idx"], "bytes": record["bytes"],
                                 "block": block, "register": name, "value": f"0x{value:08x}",
                                 "field": field.split("=")[0] if field else "",
                                 "field_value": field.split("=")[1] if field else ""})

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]) if csv_rows else [])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
