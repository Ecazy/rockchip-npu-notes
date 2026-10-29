#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyze the CUBE->WEIGHT boundary for a 16x16 QK head.

This is a proof/constraint tool, not a fake implementation.  It computes the
actual C++ co_index() formulas, the existing fixed p0312 reformat permutation,
and the required CUBE->WEIGHT reorder.  If no existing view/permutation matches,
the output lists the exact constraints a new device reorder primitive must meet.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cube16(r: int, c: int) -> int:
    physical_rows = 16
    return (c // 8) * physical_rows * 8 + r * 8 + c % 8


def weight16(n: int, k: int) -> int:
    # co_tensor_weight(16,16): physical_rows=32, physical_cols=32
    return (((n // 16) * 1 + k // 32) * 16 + n % 16) * 32 + k % 32


def reformat_p0312(out_index: int) -> int:
    # out[c,a,b] = in[a,b,c]
    c, a, b = out_index // 64, (out_index % 64) // 8, out_index % 8
    return (a * 8 + b) * 8 + c


def main() -> int:
    required = {}
    for n in range(16):
        for k in range(16):
            required[weight16(n, k)] = cube16(n, k)
    mismatches = []
    for dest, src in required.items():
        got = reformat_p0312(dest)
        if got != src:
            mismatches.append({"dest": dest, "required_src": src,
                               "p0312_src": got})
    # A concrete example: K[0,1] must land at W[0,1].
    examples = [
        {"logical": "K[0,1]",
         "cube_src": cube16(0, 1), "weight_dest": weight16(0, 1),
         "p0312_src_for_dest": reformat_p0312(weight16(0, 1))},
        {"logical": "K[1,0]",
         "cube_src": cube16(1, 0), "weight_dest": weight16(1, 0),
         "p0312_src_for_dest": reformat_p0312(weight16(1, 0))},
    ]
    report = {
        "schema_version": 1,
        "kind": "cube-to-weight-boundary-analysis",
        "tile": [16, 16],
        "cube_index": "((c//8)*16+r)*8+c%8",
        "weight_index": "n*32+k for n,k<16 (within the 1024-element padded BO)",
        "required_mapping_pairs": len(required),
        "p0312_mismatches": len(mismatches),
        "p0312_mismatch_examples": mismatches[:8],
        "required_examples": examples,
        "existing_view_verdict": (
            "No existing CUBE/WEIGHT view or p0312 permutation maps the full "
            "16x16 logical K tile to the QK B/WEIGHT physical layout."
        ),
        "new_primitive_contract": {
            "id": "co.reorder.cube_to_weight.f16.v1",
            "input": "CUBE [16,16]",
            "output": "WEIGHT [16,16]",
            "element_mapping": "dst_weight[n*32+k] = src_cube[(k//8)*128+n*8+k%8]",
            "sync": "one PPU/DMA task or a copy command with explicit source/dest BO and 16-byte alignment",
            "grouping": "one 16x16 head per task; a grouped emit may cover 4/12 heads in one submit",
            "acceptance": "bitwise equal to host K^T/layout reference for one head; then 4 heads; then QKV->reorder->QK producer chain",
        },
        "minimal_experiment": [
            "construct one 16x16 CUBE K tile and its host weight-layout reference",
            "run the proposed reorder task three times with same output and compare bitwise",
            "extend to four heads and verify no cross-head overlap",
            "only after bitwise pass, wire QKV output and QK consumer in one graph",
        ],
        "not_done": "No hardware reorder command exists in the current custom-ops set; this tool does not claim a device implementation.",
    }
    out = ROOT / "results" / "20261028T113459Z-graph-fusion" / "cube-weight-boundary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
