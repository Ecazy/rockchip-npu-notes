#!/usr/bin/env python3
"""Render the Qwen3.5 GDN-layer RK3588 structure timeline.

The figure is the GDN counterpart of ``render_sdpa_module_timeline_svg.py`` and
keeps the same typographic conventions (Times New Roman 7.2 pt labels, 5.8 pt
notes, 504 pt wide panels, stage-coloured bars, hatched uncertainty).

What is plotted
---------------
(a) one stacked horizontal bar per RKNN submit: how many hardware tasks of each
    Qwen3.5 GDN math stage that submit contains;
(b) a module x submit structure map: how many tasks of each submit touch CNA /
    CORE / DPU / host CPU, derived from the RKNN op table (op type -> block).
    This is *op-derived engagement*, not an S_STATUS sample: the runtime
    dispatches this graph as three subcore batches and the PC task counter does
    not advance per task (measured: one counter transition per submit), so the
    SDPA-style per-task S_STATUS timeline cannot be sampled for this workload.

Usage:
  python3 render_gdn_timeline_svg.py --summary gdn_trace_summary.json --out gdn_module_timeline.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTH = 504.0
FONT = "Times New Roman, Noto Serif CJK SC, serif"

STAGE_COLORS = {
    "Input RMSNorm": "#b2b2b2",
    "QKV/Z/B/A projections": "#7c4c9f",
    "Causal conv1d + SiLU": "#1f5076",
    "L2 norm (q,k)": "#cf515b",
    "Gates (beta, g)": "#db8225",
    "QK + UT system": "#2097a5",
    "Triangular inverse": "#d0a6cc",
    "Delta-rule scan": "#9a674e",
    "Gated RMSNorm": "#6b8e23",
    "Output projection": "#4682b4",
    "Layout / staging": "#d9d9d9",
    "Host (CPU op)": "#f2f2f2",
}
STAGE_ORDER = list(STAGE_COLORS)
# shorter labels for the legend row; the JSON keeps the full stage names
STAGE_LABELS = {
    "Input RMSNorm": "Input RMSNorm",
    "QKV/Z/B/A projections": "QKV/Z/B/A proj.",
    "Causal conv1d + SiLU": "Conv1d + SiLU",
    "L2 norm (q,k)": "L2 norm (q,k)",
    "Gates (beta, g)": "Gates (beta, g)",
    "QK + UT system": "QK + UT system",
    "Triangular inverse": "Triangular inverse",
    "Delta-rule scan": "Delta-rule scan",
    "Gated RMSNorm": "Gated RMSNorm",
    "Output projection": "Output proj.",
    "Layout / staging": "Layout / staging",
    "Host (CPU op)": "Host CPU",
}
MODULES = ["CNA", "CORE", "DPU", "HOST"]
MODULE_LABELS = {"CNA": "CNA", "CORE": "CORE", "DPU": "DPU", "HOST": "Host CPU"}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def rect(self, x, y, w, h, fill, hatch=False, stroke="none", sw=0.0, rx=0.0) -> None:
        extra = ' stroke-dasharray="1.2,0.9" stroke-width="0.35"' if hatch else ""
        if stroke != "none":
            extra = f' stroke="{stroke}" stroke-width="{sw}"'
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
            f'fill="{fill}"{extra}/>'
        )

    def text(self, x, y, body, size=7.2, fill="#333333", anchor="start", weight="400") -> None:
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size:.2f}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(body)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke="#888888", sw=0.35, dash=None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )


def legend(svg: Svg, y0: float, stages: list[str]) -> float:
    """Two-row legend like the SDPA figure; returns the y after the legend."""
    col_w = (WIDTH - 2 * 58.0) / 4.0
    for i, stage in enumerate(stages):
        row, col = divmod(i, 4)
        x = 58.0 + col * col_w
        y = y0 + row * 9.83
        svg.rect(x, y, 7.2, 5.04, STAGE_COLORS[stage])
        if stage == "Host (CPU op)":
            svg.rect(x, y, 7.2, 5.04, "none", stroke="#999999", sw=0.35, hatch=True)
        svg.text(x + 9.72, y + 5.04, STAGE_LABELS.get(stage, stage))
    rows = (len(stages) + 3) // 4
    return y0 + rows * 9.83


def render_panel_a(summary: dict, submits: list[dict]) -> str:
    top = 6.0
    legend_bottom = 0.0
    svg = Svg()
    # measure legend height first
    legend_bottom = top + ((len(STAGE_ORDER) + 3) // 4) * 9.83
    plot_top = legend_bottom + 12.0
    row_h = 8.6
    plot_h = row_h * len(submits)
    height = plot_top + plot_h + 36.0
    svg.body_top = plot_top
    svg.height = height

    legend(svg, top, STAGE_ORDER)
    svg.text(58.0, legend_bottom + 5.0, "Bars: hardware tasks per RKNN submit, stacked by Qwen3.5 GDN math stage "
             f"({summary['tasks']:,} tasks / {summary['submits']} submits / {summary['rknn_ops']} rknn ops)",
             size=5.8, fill="#666666")

    max_tasks = max(s["tasks"] for s in submits)
    left, right = 58.0, WIDTH - 12.0
    usable = right - left
    axis_y = plot_top + plot_h + 3.0
    for s in submits:
        y = plot_top + (s["submit_seq"] - 1) * row_h
        svg.text(left - 3.0, y + row_h - 2.0, f"S{s['submit_seq']}", size=5.8, fill="#333333", anchor="end")
        svg.line(left, y + row_h - 1.6, right, y + row_h - 1.6, stroke="#eeeeee", sw=0.3)
        x = left
        for stage in STAGE_ORDER:
            count = s["stage_tasks"].get(stage, 0)
            if not count:
                continue
            w = usable * count / max_tasks
            svg.rect(x, y + 0.8, w, row_h - 2.4, STAGE_COLORS[stage],
                     hatch=(stage == "Host (CPU op)"))
            x += w
        label = f"{s['tasks']:,} tasks  ({s['ops']} ops)"
        label_w = 4.2 * len(label) * 0.52
        svg.rect(right - label_w - 1.0, y + 0.9, label_w, row_h - 2.6, "#ffffff")
        svg.text(right - 0.5, y + row_h - 2.2, label, size=5.6, fill="#444444", anchor="end")

    max_k = max(s["tasks"] for s in submits) / 1000.0
    ticks = 5
    for i in range(ticks + 1):
        value = max_k * i / ticks
        x = left + usable * i / ticks
        svg.line(x, axis_y, x, axis_y + 2.0, stroke="#666666", sw=0.35)
        svg.text(x, axis_y + 7.0, f"{value:.0f}k" if max_k >= 2 else f"{value:.1f}k",
                 size=5.2, fill="#555555", anchor="middle")
    svg.text(left, axis_y + 16.5, "hardware tasks per submit", size=5.6, fill="#555555")
    svg.text(58.0, height - 4.0,
             "Source: EPERF_RKNPU_IOCTL_TRACE task records joined with the RKNN level-5 op table. "
             "Task counts are exact; the trace run's hw_elapse is inflated by tracing and is not plotted.",
             size=5.2, fill="#777777")
    return wrap(svg, f"GDN layer structure per RKNN submit ({summary['model']})")


def render_panel_b(summary: dict, submits: list[dict]) -> str:
    top = 6.0
    legend_bottom = top + ((len(STAGE_ORDER) + 3) // 4) * 9.83
    plot_top = legend_bottom + 12.0
    row_h = 13.0
    plot_h = row_h * len(MODULES)
    height = plot_top + plot_h + 30.0
    svg = Svg()
    svg.height = height
    legend(svg, top, STAGE_ORDER)
    svg.text(58.0, legend_bottom + 5.0,
             "Cells: share of a submit's hardware tasks that run on the module (op-derived from the RKNN op table, "
             "not an S_STATUS sample)",
             size=5.8, fill="#666666")

    left, right = 58.0, WIDTH - 12.0
    cell_w = (right - left) / len(submits)
    for m_i, module in enumerate(MODULES):
        y = plot_top + m_i * row_h
        svg.text(left - 3.0, y + row_h - 4.5, MODULE_LABELS[module], size=6.4, anchor="end")
        for s_i, s in enumerate(submits):
            total = sum(s["module_tasks"].values()) or 1
            share = s["module_tasks"].get(module, 0) / total
            x = left + s_i * cell_w
            shade = share ** 0.45
            fill = mix("#ffffff", "#1f5076", shade)
            svg.rect(x + 0.25, y + 0.8, cell_w - 0.5, row_h - 1.9, fill,
                     stroke="#e2e2e2", sw=0.25)
            if share >= 0.30:
                svg.text(x + cell_w / 2.0, y + row_h - 4.6, f"{share * 100:.0f}%",
                         size=5.0, fill="#ffffff" if shade > 0.55 else "#333333", anchor="middle")
    for m_i in range(len(MODULES) + 1):
        y = plot_top + m_i * row_h
        svg.line(left, y, right, y, stroke="#bbbbbb", sw=0.3)

    axis_y = plot_top + plot_h + 2.0
    for s_i, s in enumerate(submits):
        svg.text(left + s_i * cell_w + cell_w / 2.0, axis_y + 6.5, f"S{s['submit_seq']}",
                 size=5.0, fill="#555555", anchor="middle")
    svg.text(left, axis_y + 13.5, "submit index (chronological)", size=5.6, fill="#555555")
    svg.text(58.0, height - 4.0,
             f"{summary['npu_ops']} of {summary['rknn_ops']} rknn ops stay on the NPU, {summary['cpu_ops']} fall back to the "
             f"host CPU; the graph is dispatched as {summary['submits']} submits of three subcore task groups.",
             size=5.2, fill="#777777")
    return wrap(svg, f"GDN layer module engagement per submit ({summary['model']})")


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two #rrggbb colours."""
    va = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    vb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(va, vb))


def wrap(svg: Svg, title: str) -> str:
    body = "\n".join(svg.parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{svg.height:.2f}pt" '
        f'viewBox="0 0 {WIDTH} {svg.height:.2f}" version="1.1">\n'
        f"<title>{esc(title)}</title>\n"
        f'<rect x="0" y="0" width="{WIDTH}" height="{svg.height:.2f}" fill="#ffffff"/>\n'
        f"{body}\n</svg>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    submits = sorted(summary["submit_detail"], key=lambda s: s["submit_seq"])

    a_svg = render_panel_a(summary, submits)
    b_svg = render_panel_b(summary, submits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    (stem.parent / f"{stem.name}_a.svg").write_text(a_svg, encoding="utf-8")
    (stem.parent / f"{stem.name}_b.svg").write_text(b_svg, encoding="utf-8")

    # merged: (a) on top, (b) below, sharing one legend
    s_a = json.loads(a_svg.split("\n", 1)[1].split("\n", 1)[1].split(">", 1)[1].rsplit("</svg>", 1)[0] and "{}")
    merged_height = json.loads(json.dumps(_svg_height(a_svg))) if False else None
    h_a = _svg_height(a_svg)
    h_b = _svg_height(b_svg)
    body_a = a_svg.split("\n", 3)[3].rsplit("</svg>", 1)[0]
    body_b = b_svg.split("\n", 3)[3].rsplit("</svg>", 1)[0]
    merged = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{h_a + h_b:.2f}pt" '
        f'viewBox="0 0 {WIDTH} {h_a + h_b:.2f}" version="1.1">\n'
        f"<title>GDN layer RKNN structure timeline ({esc(summary['model'])})</title>\n"
        f'<rect x="0" y="0" width="{WIDTH}" height="{h_a + h_b:.2f}" fill="#ffffff"/>\n'
        f'<g transform="translate(0,0)">\n{body_a}\n</g>\n'
        f'<g transform="translate(0,{h_a:.2f})">\n{body_b}\n</g>\n</svg>\n'
    )
    args.out.write_text(merged, encoding="utf-8")

    if args.preview:
        try:
            import cairosvg
        except ImportError:
            print("cairosvg not available; skipped PNG previews")
            return 0
        for svg_text, path in (
            (a_svg, stem.parent / f"{stem.name}_a_preview.png"),
            (b_svg, stem.parent / f"{stem.name}_b_preview.png"),
            (merged, stem.parent / f"{stem.name}_preview.png"),
        ):
            cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(path), scale=2.0)
    print(f"wrote {args.out} (+ _a/_b)")
    return 0


def _svg_height(svg_text: str) -> float:
    marker = 'height="'
    start = svg_text.index(marker) + len(marker)
    end = svg_text.index("pt", start)
    return float(svg_text[start:end])


if __name__ == "__main__":
    raise SystemExit(main())
