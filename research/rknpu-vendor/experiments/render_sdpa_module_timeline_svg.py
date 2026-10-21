#!/usr/bin/env python3
"""Render a publication-style SDPA module timeline.

Two standalone figures each carry a compact math-phase legend:
  (a) all 16 heads over the full measured op6 window;
  (b) a zoom of four consecutive heads (H0-H3) for task-scale detail.

Rows are CNA/CORE/DPU/DPU_RDMA, x is PC-counter time, bars are math-phase-colored
S_STATUS operating snapshots at PC-task-counter transitions.  The figure keeps
only the information needed in a paper figure; the complete measurement notes
live in sdpa_module_occupancy.md.

Usage:
  python3 render_sdpa_module_timeline_svg.py \
    --occupancy results/sdpa_timeline_20261020/sdpa_module_occupancy.csv \
    --out results/sdpa_timeline_20261020/sdpa_module_timeline.svg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

from sdpa_operation_map import OP_COLOR, OPERATIONS

FONT = "Times New Roman, Noto Serif CJK SC, serif"
W, H = 504, 173.88
LEFT, RIGHT = 58.0, 498.24
ROWS = [
    ("CNA", "cna_state"),
    ("CORE", "core_state"),
    ("DPU", "dpu_state"),
    ("DPU_RDMA", "dpu_rdma_state"),
]
MUTED = "#555555"
FRAME = "#d6dbe1"


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
         opacity: float | None = None, rx: float = 0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.15):.2f}" '
        f'height="{max(h, 0.15):.2f}" rx="{rx}" fill="{fill}"'
        f"{stroke_attr}{opacity_attr}/>"
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str,
         width: float = 0.35, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def head_spans(rows: list[dict]) -> list[dict]:
    spans = []
    for h in range(16):
        hr = [r for r in rows if r.get("head") == f"head{h}"]
        if hr:
            spans.append({
                "head": h,
                "start": min(float(r["start_us"]) for r in hr),
                "end": max(float(r["end_us"]) for r in hr),
            })
    return spans


def draw_legend(svg: list[str]) -> None:
    items = [(label, color) for label, _key, color in OPERATIONS]
    items.append(("mixed", "url(#mixedHatch)"))
    cols = 5
    col_w = (RIGHT - LEFT) / cols
    for i, (label, color) in enumerate(items):
        row, col = divmod(i, cols)
        lx = LEFT + col * col_w
        ly = 12.39 + row * 9.834
        if color == "url(#mixedHatch)":
            svg.append(rect(lx, ly - 5.04, 7.2, 5.04, fill="#d9d9d9"))
            svg.append(rect(lx, ly - 5.04, 7.2, 5.04, fill=color))
        else:
            svg.append(rect(lx, ly - 5.04, 7.2, 5.04, fill=color))
        svg.append(text(lx + 9.72, ly, label, size=7.2, fill="#333333", anchor="start"))
    svg.append(text(LEFT, 31.2,
                    "Bars: S_STATUS at PC-task boundaries, zero-order hold (not continuous).",
                    size=5.8, fill="#666666", anchor="start"))
    svg.append(text(LEFT, 36.8,
                    "Staging/layout is conservative; SDPA math phases are tensor-validated.",
                    size=5.8, fill="#666666", anchor="start"))


def draw_panel(
    svg: list[str],
    rows: list[dict],
    spans: list[dict],
    *,
    total: float,
    strip_top: float,
    strip_bottom: float,
    plot_top: float,
    plot_bottom: float,
    tick_step: float,
    panel_tag: str,
    panel_desc: str,
    bar_h: float,
    head_font: float,
    minor_task_ticks: bool = False,
    x_label: bool = False,
) -> None:
    def x(us: float) -> float:
        return LEFT + us / total * (RIGHT - LEFT)

    # Match the reference: a small centered caption above the timeline.
    svg.append(text((LEFT + RIGHT) / 2, strip_top - 4.3,
                    f"{panel_tag} {panel_desc}", size=8.0))

    # Head strip.
    svg.append(rect(LEFT, strip_top, RIGHT - LEFT, strip_bottom - strip_top,
                    fill="#ffffff", stroke=FRAME, width=0.35))
    prologue_end = spans[0]["start"] if spans else 0.0
    if prologue_end > 0:
        pe = x(prologue_end)
        svg.append(rect(LEFT, strip_top, max(pe - LEFT, 0.4),
                        strip_bottom - strip_top, fill="#e5e5e5",
                        stroke="#ffffff", width=0.3))
        if pe - LEFT > 9:
            svg.append(text(LEFT + 2.5, strip_bottom - 2.2, "P", size=5.2,
                            fill="#555555", anchor="start"))
    for i, sp in enumerate(spans):
        sx0, sx1 = x(sp["start"]), x(sp["end"])
        fill = "#f7f8fa" if i % 2 == 0 else "#ffffff"
        svg.append(rect(sx0, strip_top, sx1 - sx0, strip_bottom - strip_top,
                        fill=fill, stroke="#ffffff", width=0.25))
        svg.append(text((sx0 + sx1) / 2, strip_bottom - 2.2,
                        f"H{sp['head']}", size=head_font, fill="#444444"))
        if i:
            svg.append(line(sx0, strip_top, sx0, strip_bottom,
                            stroke="#c8c8c8", width=0.3))
    if minor_task_ticks:
        for r in rows:
            if r.get("head") not in {f"head{h}" for h in range(4)}:
                continue
            tx = x(float(r["start_us"]))
            svg.append(line(tx, strip_bottom - 3.0, tx, strip_bottom,
                            stroke="#9e9e9e", width=0.25))

    # Separate, lightly outlined module lanes as in layer0_pipeline.svg.
    row_h = (plot_bottom - plot_top) / len(ROWS)
    for ri, (module, _state_key) in enumerate(ROWS):
        top = plot_top + ri * row_h
        cy = top + row_h / 2
        svg.append(rect(LEFT, top + 1.0, RIGHT - LEFT, row_h - 2.0,
                        fill="#ffffff", stroke=FRAME, width=0.5))
        svg.append(text(LEFT - 4, cy + 3.1, module, size=9.0,
                        anchor="end"))

    tick = 0.0
    while tick <= total + 0.01:
        gx = x(tick)
        svg.append(line(gx, plot_top, gx, plot_bottom,
                        stroke="#eceff2", width=0.5))
        tick += tick_step

    # Operation-colored S_STATUS bars.
    for ri, (_module, state_key) in enumerate(ROWS):
        cy = plot_top + (ri + 0.5) * row_h
        for r in rows:
            state = r[state_key]
            if state not in ("busy", "mixed"):
                continue
            bx = x(float(r["start_us"]))
            bw = x(float(r["end_us"])) - bx
            color = OP_COLOR.get(r.get("math_phase", ""), "#b2b2b2")
            if state == "busy":
                svg.append(rect(bx, cy - bar_h / 2, bw, bar_h, fill=color))
            else:
                svg.append(rect(bx, cy - bar_h / 2, bw, bar_h, fill=color,
                                opacity=0.25))
                svg.append(rect(bx, cy - bar_h / 2, bw, bar_h,
                                fill="url(#mixedHatch)"))

    # Thin black baseline, ticks and serif labels match the reference.
    svg.append(line(LEFT, plot_bottom, RIGHT, plot_bottom,
                    stroke="#000000", width=0.5))
    tick = 0.0
    while tick <= total + 0.01:
        gx = x(tick)
        svg.append(line(gx, plot_bottom, gx, plot_bottom + 3.0,
                        stroke="#000000", width=0.8))
        svg.append(text(gx, plot_bottom + 10.55, f"{tick:.0f}", size=8.0,
                        fill="#444444"))
        tick += tick_step
    if x_label:
        svg.append(text((LEFT + RIGHT) / 2, plot_bottom + 19.83,
                        "PC-counter time (µs)", size=8.0, fill="#222222"))


def render_figure(rows: list[dict], spans: list[dict], *, total: float,
                  zoom: bool) -> str:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
        f'viewBox="0 0 {W} {H}" version="1.1">',
        '<title>SDPA module timeline: ' + ('H0–H3' if zoom else '16 heads') + '</title>',
        '<defs><pattern id="mixedHatch" width="3" height="3" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="3" stroke="#cccccc" '
        'stroke-width="0.45"/></pattern></defs>',
        rect(0, 0, W, H, fill="#ffffff"),
    ]
    draw_legend(svg)
    draw_panel(
        svg, rows, spans, total=total,
        strip_top=46.8, strip_bottom=56.8,
        plot_top=59.8, plot_bottom=148.68,
        tick_step=100 if zoom else 500,
        panel_tag="(b)" if zoom else "(a)",
        panel_desc="H0–H3" if zoom else "16 heads",
        bar_h=10.0, head_font=7.0,
        minor_task_ticks=zoom, x_label=True,
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def render(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r["task_index"]))
    total = max(float(r["end_us"]) for r in rows)
    spans = head_spans(rows)
    zheads = [s for s in spans if s["head"] <= 3]
    zoom_end = max(s["end"] for s in zheads)
    zrows = [r for r in rows if r.get("head") == "prologue" or r.get("head") in
             {f"head{h}" for h in range(4)}]
    figures = [
        render_figure(rows, spans, total=total, zoom=False),
        render_figure(zrows, zheads, total=zoom_end, zoom=True),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, figure in zip(("a", "b"), figures):
        path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")
        path.write_text(figure, encoding="utf-8")
        print(f"wrote {path}")

    # Keep the existing combined filename usable for links and comparisons.
    combined = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" '
                f'height="{2 * H}pt" viewBox="0 0 {W} {2 * H}">']
    for i, figure in enumerate(figures):
        # Isolate pattern IDs so both nested figures remain valid together.
        figure = figure.replace("mixedHatch", f"mixedHatch{i}")
        figure = figure.replace(f'width="{W}pt" height="{H}pt"',
                                 f'width="{W}" height="{H}"', 1)
        combined.append(figure.replace('<svg ', f'<svg x="0" y="{i * H}" ', 1))
    combined.append("</svg>")
    out_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occupancy", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    render(load_csv(args.occupancy), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
