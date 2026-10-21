#!/usr/bin/env python3
"""Math-phase taxonomy for the SDPA op6 timeline.

The figure's y rows already encode hardware modules (CNA/CORE/DPU/DPU_RDMA).
Colors therefore encode *math phases* that are orthogonal to those rows: the
same phase can light several module rows, and QK/PV use different colors even
though both are CNA+CORE tasks.

Evidence levels
---------------
validated
    The phase boundary is tied to a dumped intermediate tensor and a CPU
    reference correlation (see sdpa_intermediate_validation.csv).  For
    softmax exp/LUT, H1 task 38 writes the FP16 row-max-subtracted score and
    task 39 writes its FP16 exponential.
conservative
    Data movement / layout / unclassified DPU work.  This label makes no claim
    beyond the regcmd task boundary.

Per-head task templates, confirmed on all 16 heads:

  H0     : QK @ 4, reduce @ 10, exp/LUT @ 11 & 13, normalize @ 14-17, PV @ 24
  H1-H15 : QK @ 5, reduce @ 11, exp/LUT @ 12-13, normalize @ 14-17, PV @ 24

H0 is shifted one slot earlier between QK and the exp phase because the LUT is
loaded once as a dedicated `lut_table_load` task inside H0.  H1-H15 reuse it.
"""

from __future__ import annotations

import re

# (display label, stable key, colour).  Colours are taken from the reference
# layer0_pipeline.svg palette, but the grouping is math, not register family.
OPERATIONS = [
    ("Prologue / init", "prologue", "#b2b2b2"),
    ("LUT table load", "lut_load", "#1f5076"),
    ("QK matmul", "qk_matmul", "#7c4c9f"),
    ("Softmax reduce", "softmax_reduce", "#cf515b"),
    ("Softmax exp/LUT", "softmax_exp_lut", "#db8225"),
    ("Softmax normalize", "softmax_normalize", "#d0a6cc"),
    ("PV matmul", "pv_matmul", "#2097a5"),
    ("Staging / layout", "staging", "#9a674e"),
]

OP_COLOR = {key: color for _, key, color in OPERATIONS}
OP_LABEL = {key: label for label, key, _ in OPERATIONS}
INFERRED_OPERATIONS: set[str] = set()

PHASE_CONFIDENCE = {
    "prologue": "conservative",
    "lut_load": "explicit",
    "qk_matmul": "validated",
    "softmax_reduce": "validated",
    "softmax_exp_lut": "validated",
    "softmax_normalize": "validated",
    "pv_matmul": "validated",
    "staging": "conservative",
}


def _head_ordinal(row: dict) -> int | None:
    match = re.fullmatch(r"head(\d+)", row.get("head_group") or row.get("head") or "")
    return int(match.group(1)) if match else None


def _relative_task(row: dict) -> int | None:
    value = row.get("head_task_order")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_operation(row: dict) -> str:
    """Map one emitted op6 task to a validated/conservative math phase."""
    if row.get("stage") == "sdpa_prologue":
        return "prologue"

    # The single explicit LUT-table loading task is kept separate from the
    # exp/LUT arithmetic task that consumes the loaded table.
    if row.get("task_class") == "lut_table_load":
        return "lut_load"

    head = _head_ordinal(row)
    rel = _relative_task(row)
    if head is None or rel is None:
        return "staging"

    if head == 0:
        by_position = {
            4: "qk_matmul",
            10: "softmax_reduce",
            11: "softmax_exp_lut",
            13: "softmax_exp_lut",
            24: "pv_matmul",
        }
    else:
        by_position = {
            5: "qk_matmul",
            11: "softmax_reduce",
            12: "softmax_exp_lut",
            13: "softmax_exp_lut",
            24: "pv_matmul",
        }

    if rel in by_position:
        return by_position[rel]

    # Row-sum, reciprocal/normalisation scale, probability multiply and its
    # final layout are one phase.  The row-sum and probability endpoints are
    # both directly validated by workspace dumps.
    if rel in (14, 15, 16, 17):
        return "softmax_normalize"

    # All remaining tasks are staging/layout or unclassified DPU work.  The
    # classifier deliberately does not invent finer math labels here.
    return "staging"


def operation_color(row: dict) -> str:
    return OP_COLOR[classify_operation(row)]
