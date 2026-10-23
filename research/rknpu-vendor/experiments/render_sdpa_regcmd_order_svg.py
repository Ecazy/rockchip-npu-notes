#!/usr/bin/env python3
"""Render regcmd-issue-order views without replacing the task/S_STATUS figures.

The input is the captured 8-byte regcmd stream already exported to
`sdpa_npuop_timeline.csv`.  Within each task the stream is ordered by
`npuop_index`; across tasks it is ordered by `submit_task_order`.

This is an issue/program-order view, NOT a per-regcmd elapsed-time trace:
the hardware exposes a task counter, not a regcmd-level timestamp.  Each
marker is one captured regcmd word, placed at its target block row and
colored with the validated math phase of its enclosing task.

Outputs:
  sdpa_regcmd_order.csv
  sdpa_regcmd_order_summary.json
  sdpa_regcmd_order_a.svg          # all op6 tasks
  sdpa_regcmd_order_b.svg          # H1, 25 consecutive tasks
  sdpa_regcmd_order.svg            # stacked compatibility view

Usage:
  python3 render_sdpa_regcmd_order_svg.py \
    --npuop results/sdpa_timeline_20261020/sdpa_npuop_timeline.csv \
    --occupancy results/sdpa_timeline_20261020/sdpa_module_occupancy.csv \
    --tasks results/sdpa_timeline_20261020/sdpa_task_timeline.csv \
    --out results/sdpa_timeline_20261020/sdpa_regcmd_order.svg
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from sdpa_operation_map import OP_COLOR, OPERATIONS

FONT = "Times New Roman, Noto Serif CJK SC, serif"
W, H = 504, 173.88
LEFT, RIGHT = 58.0, 498.24
STRIP_TOP, STRIP_BOTTOM = 48.0, 57.0
PLOT_TOP, PLOT_BOTTOM = 60.0, 151.0
ROWS = ["PC", "CNA", "CORE", "DPU", "DPU_RDMA"]
ROW_COLOR = {
    "PC": "#d9d9d9",
    "CNA": "#7c4c9f",
    "CORE": "#ab72ad",
    "DPU": "#db8225",
    "DPU_RDMA": "#9a674e",
}
FRAME = "#d6dbe1"
MUTED = "#555555"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def text(x: float, y: float, value: str, *, size: float = 7.2,
         fill: str = "#222222", anchor: str = "middle",
         weight: str = "400") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" '
        f'font-size="{size:.2f}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, *, fill: str,
         stroke: str | None = None, width: float = 0.3,
         opacity: float | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.01):.2f}" '
        f'height="{max(h, 0.01):.2f}" fill="{fill}"'
        f"{stroke_attr}{opacity_attr}/>"
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str,
         width: float = 0.35, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def text_on(fill: str) -> str:
    """Readable label color for a phase-colored task strip."""
    value = fill.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.52 else "#222222"


def load_data(npuop: Path, occupancy: Path, tasks: Path):
    ops = [
        r for r in load_csv(npuop)
        if r.get("op_idx") == "6"
    ]
    occ = {
        int(r["task_index"]): r for r in load_csv(occupancy)
    }
    task_meta = {
        int(r["submit_task_order"]): r
        for r in load_csv(tasks)
        if r.get("op_idx") == "6"
    }
    ops.sort(key=lambda r: (int(r["submit_task_order"]), int(r["npuop_index"])))
    by_task: dict[int, list[dict]] = collections.defaultdict(list)
    for r in ops:
        by_task[int(r["submit_task_order"])].append(r)
    for task_idx in by_task:
        by_task[task_idx].sort(key=lambda r: int(r["npuop_index"]))
    return ops, occ, task_meta, by_task


def build_regcmd_rows(ops: list[dict], occ: dict[int, dict],
                      task_meta: dict[int, dict]) -> list[dict]:
    counts = collections.Counter(int(r["submit_task_order"]) for r in ops)
    rows: list[dict] = []
    for global_index, r in enumerate(ops):
        task_idx = int(r["submit_task_order"])
        task = occ[task_idx]
        meta = task_meta[task_idx]
        rows.append({
            "global_regcmd_index": global_index,
            "task_index": task_idx,
            "head": task.get("head", ""),
            "head_task_order": task.get("head_task_order", ""),
            "math_phase": task.get("math_phase", ""),
            "phase_confidence": task.get("phase_confidence", ""),
            "task_local_regcmd_index": int(r["npuop_index"]),
            "task_regcmd_count_captured": counts[task_idx],
            "task_regcmd_count_expected": int(meta.get("fetch_entries") or 0),
            "regcmd_capture_truncated": int(meta.get("capture_truncated") or 0),
            "op": r.get("op", ""),
            "module": r.get("module", ""),
            "reg_offset": r.get("reg_offset", ""),
            "reg_name": r.get("reg_name", ""),
            "value": r.get("value", ""),
            "task_start_us": task.get("start_us", ""),
            "task_end_us": task.get("end_us", ""),
            "task_duration_us": task.get("duration_us", ""),
        })
    return rows


def write_outputs(rows: list[dict], out_svg: Path) -> dict:
    base = out_svg.with_suffix("")
    csv_path = base.with_name(base.name + "_rows.csv")
    json_path = base.with_name(base.name + "_summary.json")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    phase_counts = collections.Counter(r["math_phase"] for r in rows)
    module_counts = collections.Counter(r["module"] for r in rows)
    head_counts = collections.Counter(r["head"] for r in rows)
    task_counts = collections.Counter(r["task_index"] for r in rows)
    truncated = sorted({
        int(r["task_index"])
        for r in rows if r["regcmd_capture_truncated"]
    })
    h1_rows = [r for r in rows if r["head"] == "head1"]
    summary = {
        "source": "captured 8-byte regcmd stream from rknpu ioctl hook",
        "semantics": {
            "order": (
                "captured regcmd issue order: submit_task_order then npuop_index; "
                "not a per-regcmd elapsed-time measurement"
            ),
            "module": "target register page of each 8-byte regcmd word",
            "completion": (
                "module blocks can execute concurrently after dispatch; one task's "
                "regcmd issue order is not a serial completion order"
            ),
            "colors": "same validated math-phase palette as the S_STATUS timeline",
        },
        "task_count": len({r["task_index"] for r in rows}),
        "captured_regcmd_entries": len(rows),
        "expected_regcmd_entries": sum(
            max(int(r["task_regcmd_count_expected"]) for r in rows if r["task_index"] == t)
            for t in task_counts
        ),
        "truncated_tasks": truncated,
        "phase_entry_counts": dict(phase_counts),
        "module_entry_counts": dict(module_counts),
        "head_entry_counts": dict(head_counts),
        "zoom_head1": {
            "task_indices": sorted({int(r["task_index"]) for r in h1_rows}),
            "captured_entries": len(h1_rows),
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def draw_legend(svg: list[str]) -> None:
    items = [(label, color) for label, _key, color in OPERATIONS]
    cols = 4
    col_w = (RIGHT - LEFT) / cols
    for i, (label, color) in enumerate(items):
        row, col = divmod(i, cols)
        lx = LEFT + col * col_w
        ly = 12.39 + row * 9.834
        svg.append(rect(lx, ly - 5.04, 7.2, 5.04, fill=color))
        svg.append(text(lx + 9.72, ly, label, size=6.8,
                        fill="#333333", anchor="start"))
    svg.append(text(
        LEFT, 30.8,
        "Order = captured 8-byte regcmd entries; x is issue index, not elapsed time.",
        size=5.8, fill=MUTED, anchor="start",
    ))
    svg.append(text(
        LEFT, 36.2,
        "Module rows show target block; after dispatch, blocks may complete concurrently.",
        size=5.8, fill=MUTED, anchor="start",
    ))


def draw_panel(svg: list[str], rows: list[dict], *, total: int,
               strip_segments: list[dict], panel_tag: str, panel_desc: str,
               tick_step: int, zoom: bool, truncated_note: str | None) -> None:
    def x(index: float) -> float:
        return LEFT + (index + 0.5) / max(total, 1) * (RIGHT - LEFT)

    def x_band(start: float) -> float:
        return LEFT + start / max(total, 1) * (RIGHT - LEFT)

    svg.append(text((LEFT + RIGHT) / 2, STRIP_TOP - 4.3,
                    f"{panel_tag} {panel_desc}", size=8.0))
    svg.append(rect(LEFT, STRIP_TOP, RIGHT - LEFT,
                    STRIP_BOTTOM - STRIP_TOP, fill="#ffffff",
                    stroke=FRAME, width=0.35))

    cursor = 0
    for seg in strip_segments:
        w = seg["count"]
        x0 = x_band(cursor)
        x1 = x_band(cursor + w)
        if seg.get("phase"):
            fill = OP_COLOR.get(seg["phase"], "#e5e5e5")
            fg = text_on(fill)
        else:
            fill = seg.get("fill", "#e5e5e5")
            fg = "#444444"
        svg.append(rect(x0, STRIP_TOP, x1 - x0,
                        STRIP_BOTTOM - STRIP_TOP, fill=fill,
                        stroke="#ffffff", width=0.25))
        if seg.get("label") and x1 - x0 > (7.0 if zoom else 10.0):
            svg.append(text((x0 + x1) / 2, STRIP_BOTTOM - 2.2,
                            seg["label"], size=4.8 if zoom else 6.6,
                            fill=fg))
        if seg.get("star"):
            svg.append(text(x1 - 2.0, STRIP_TOP + 4.1, "*",
                            size=6.0, fill="#222222", anchor="end"))
        cursor += w

    row_h = (PLOT_BOTTOM - PLOT_TOP) / len(ROWS)
    for ri, module in enumerate(ROWS):
        top = PLOT_TOP + ri * row_h
        cy = top + row_h / 2
        svg.append(rect(LEFT, top + 0.9, RIGHT - LEFT, row_h - 1.8,
                        fill="#ffffff", stroke=FRAME, width=0.5))
        svg.append(text(LEFT - 4.0, cy + 2.9, module, size=8.4, anchor="end"))

    tick = 0
    while tick <= total + 0.01:
        gx = x_band(tick)
        svg.append(line(gx, PLOT_TOP, gx, PLOT_BOTTOM,
                        stroke="#eef1f4", width=0.45))
        svg.append(line(gx, PLOT_BOTTOM, gx, PLOT_BOTTOM + 2.7,
                        stroke="#000000", width=0.7))
        svg.append(text(gx, PLOT_BOTTOM + 10.2, f"{tick:d}",
                        size=7.3, fill="#444444"))
        tick += tick_step

    # One thin vertical marker per captured 8-byte regcmd entry.
    for r in rows:
        module = r["module"]
        if module not in ROWS:
            continue
        ri = ROWS.index(module)
        cy = PLOT_TOP + (ri + 0.5) * row_h
        gx = x(r["global_regcmd_index"] if not zoom else r["_zoom_index"])
        color = OP_COLOR.get(r["math_phase"], "#b2b2b2")
        svg.append(line(gx, cy - 4.1, gx, cy + 4.1,
                        stroke=color, width=0.46 if zoom else 0.24))

    svg.append(line(LEFT, PLOT_BOTTOM, RIGHT, PLOT_BOTTOM,
                    stroke="#000000", width=0.5))
    svg.append(text((LEFT + RIGHT) / 2, PLOT_BOTTOM + 19.2,
                    "Captured regcmd issue index", size=8.0))
    if truncated_note:
        svg.append(text(RIGHT, PLOT_BOTTOM + 19.2, truncated_note,
                        size=5.4, fill="#666666", anchor="end"))


def render(rows: list[dict], out_svg: Path) -> None:
    base = out_svg.with_suffix("")
    full = [r for r in rows]
    zoom = [dict(r) for r in rows if r["head"] == "head1"]
    for index, r in enumerate(zoom):
        r["_zoom_index"] = index

    head_runs: list[dict] = []
    for head in ["prologue"] + [f"head{h}" for h in range(16)]:
        part = [r for r in rows if r["head"] == head]
        if not part:
            continue
        label = "P" if head == "prologue" else head.replace("head", "H")
        if head == "head0":
            label += "*"
        head_runs.append({
            "count": len(part),
            "label": label,
            "fill": "#eeeeee" if head == "prologue" else "#f7f8fa",
        })

    zoom_runs: list[dict] = []
    # Preserve the actual task order from the CSV rows.
    seen: list[int] = []
    for r in zoom:
        t = int(r["task_index"])
        if t not in seen:
            seen.append(t)
    for t in seen:
        part = [r for r in zoom if int(r["task_index"]) == t]
        zoom_runs.append({
            "count": len(part),
            "label": str(part[0]["head_task_order"]),
            "phase": part[0]["math_phase"],
        })

    def figure(panel_rows: list[dict], segments: list[dict], total: int,
               panel_tag: str, panel_desc: str, tick_step: int,
               is_zoom: bool, note: str | None) -> str:
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" '
            f'height="{H}pt" viewBox="0 0 {W} {H}" version="1.1">',
            f'<title>SDPA regcmd issue order: {panel_desc}</title>',
            rect(0, 0, W, H, fill="#ffffff"),
        ]
        draw_legend(svg)
        draw_panel(svg, panel_rows, total=total,
                   strip_segments=segments, panel_tag=panel_tag,
                   panel_desc=panel_desc, tick_step=tick_step,
                   zoom=is_zoom, truncated_note=note)
        svg.append("</svg>")
        return "\n".join(svg) + "\n"

    truncated = next((r for r in rows if r["regcmd_capture_truncated"]), None)
    full_note = None
    if truncated is not None:
        full_note = (
            f"* {truncated['head']} task{truncated['task_index']} "
            f"regcmd capture: {truncated['task_regcmd_count_captured']}/"
            f"{truncated['task_regcmd_count_expected']} entries."
        )
    fig_a = figure(full, head_runs, len(full), "(a)", "all op6 tasks",
                   5000, False, full_note)
    fig_b = figure(zoom, zoom_runs, len(zoom), "(b)", "H1: 25 tasks",
                   200, True, None)

    for suffix, fig in (("a", fig_a), ("b", fig_b)):
        path = base.with_name(base.name + f"_{suffix}.svg")
        path.write_text(fig, encoding="utf-8")
        print(f"wrote {path}")

    combined = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" '
                f'height="{2 * H}pt" viewBox="0 0 {W} {2 * H}">']
    for i, fig in enumerate((fig_a, fig_b)):
        fig = fig.replace(f'width="{W}pt" height="{H}pt"',
                          f'width="{W}" height="{H}"', 1)
        combined.append(fig.replace('<svg ', f'<svg x="0" y="{i * H}" ', 1))
    combined.append("</svg>")
    out_svg.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"wrote {out_svg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npuop", required=True, type=Path)
    ap.add_argument("--occupancy", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ops, occ, task_meta, _by_task = load_data(
        args.npuop, args.occupancy, args.tasks
    )
    rows = build_regcmd_rows(ops, occ, task_meta)
    out_svg = args.out
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    summary = write_outputs(rows, out_svg)
    render(rows, out_svg)
    print(
        f"captured regcmd entries={summary['captured_regcmd_entries']}, "
        f"expected={summary['expected_regcmd_entries']}, "
        f"truncated_tasks={summary['truncated_tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
