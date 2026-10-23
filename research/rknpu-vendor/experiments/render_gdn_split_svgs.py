#!/usr/bin/env python3
"""Render the two deliverables of the GDN CPU/NPU split measurement.

Figure 1  gdn_stage_cost.svg
    Measured per-stage cost of the Qwen3.5-9B GDN layer-0 prefill (S=196, real
    weights, real embedding input) for three executions:
      * all CPU, single Cortex-A76 thread (C reference, -O3)
      * hybrid: NPU front-end -> CPU delta-rule recurrence -> NPU tail
      * the monolithic NPU graph, which is numerically broken (shown hatched,
        excluded from the comparison)

Figure 2  gdn_fe_module_timeline.svg
    Edge-triggered S_STATUS samples of the NPU front-end cut: per submit, four
    module rows (CNA/CORE/DPU/DPU_RDMA) over sampled time, coloured by the math
    stage the PC counter points at.  The counter advances once per task in each
    of the three identical subcore chains (measured ratio 3.000 in all nine
    submits), so counter -> task index -> op index is exact inside a submit.

Usage:
  python3 render_gdn_split_svgs.py --cost gdn_stage_cost.json \
      --timeline gdn_fe_timeline.json --out-dir <results dir>
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

# stage keys used by the cost figure
COST_STAGES = [
    ("fe", "Front-end (RMSNorm/proj/conv/l2norm/gates)", "QKV/Z/B/A projections"),
    ("rec", "Delta-rule recurrence", "Delta-rule scan"),
    ("gnorm", "Gated RMSNorm", "Gated RMSNorm"),
    ("oproj", "Output projection", "Output projection"),
]
HANDOFF_COLOR = "#555555"



def wrap_note(text: str, width: int = 112) -> list[str]:
    """Split a footnote into at most two lines so it never runs off the canvas."""
    if len(text) <= width:
        return [text]
    cut = text.rfind(" ", 0, width)
    if cut <= 0:
        cut = width
    rest = text[cut + 1 :]
    while len(rest) > width:
        cut2 = rest.rfind(" ", 0, width)
        if cut2 <= 0:
            break
        rest = rest[cut2 + 1 :]
    return [text[:cut], rest]


def draw_note(svg: "Svg", text: str) -> None:
    lines = wrap_note(text)
    base = svg.height - 3.0 - (len(lines) - 1) * 3.6
    for i, line in enumerate(lines):
        svg.text(58.0, base + i * 3.6, line, size=5.0, fill="#777777")


class Svg:
    def __init__(self, height: float, title: str) -> None:
        self.height = height
        self.parts: list[str] = []
        self.title = title

    def rect(self, x, y, w, h, fill, stroke="none", sw=0.0, hatch=False) -> None:
        extra = ""
        if hatch:
            extra = ' stroke="#666666" stroke-width="0.45" stroke-dasharray="1.4,1.0"'
        elif stroke != "none":
            extra = f' stroke="{stroke}" stroke-width="{sw}"'
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.0):.2f}" height="{h:.2f}" fill="{fill}"{extra}/>'
        )

    def text(self, x, y, body, size=7.2, fill="#333333", anchor="start", weight="400") -> None:
        body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size:.2f}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{body}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke="#888888", sw=0.35) -> None:
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def render(self) -> str:
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{self.height:.2f}pt" '
            f'viewBox="0 0 {WIDTH} {self.height:.2f}" version="1.1">\n'
            f"<title>{self.title}</title>\n"
            f'<rect x="0" y="0" width="{WIDTH}" height="{self.height:.2f}" fill="#ffffff"/>\n'
            f"{body}\n</svg>\n"
        )


# ----------------------------------------------------------------- figure 1
LEGEND = [
    ("Front-end", STAGE_COLORS["QKV/Z/B/A projections"]),
    ("Delta-rule", STAGE_COLORS["Delta-rule scan"]),
    ("Gated norm", STAGE_COLORS["Gated RMSNorm"]),
    ("Out proj", STAGE_COLORS["Output projection"]),
    ("transfer", HANDOFF_COLOR),
    ("monolithic (wrong)", STAGE_COLORS["Triangular inverse"]),
]


def render_cost(cost: dict) -> str:
    rows = cost["rows"]
    top = 6.0
    legend_rows = 2
    legend_bottom = top + legend_rows * 9.83
    height = legend_bottom + 12.0 + (len(rows) * 13.0 + 30.0) + (13.0 + 30.0) + 26.0
    svg = Svg(height, f"GDN layer stage cost ({cost['model']})")

    col_w = (WIDTH - 2 * 58.0) / 3.0
    for i, (label, color) in enumerate(LEGEND):
        row, col = divmod(i, 3)
        x = 58.0 + col * col_w
        y = top + row * 9.83
        svg.rect(x, y, 7.2, 5.04, color, hatch=(label.startswith("monolithic")))
        svg.text(x + 9.72, y + 5.04, label, size=6.8)
    svg.text(58.0, legend_bottom + 5.5,
             "Measured wall clock per stage: CPU = 1x Cortex-A76 (C, -O3); NPU = RKNN FP16, warm, min of 10 runs",
             size=5.8, fill="#666666")

    left, right = 58.0, WIDTH - 66.0
    usable = right - left
    y = legend_bottom + 16.0

    def bars(rows_subset, max_total, title, note_size=4.8):
        nonlocal y
        svg.text(left, y, title, size=6.0, fill="#333333", weight="600")
        y += 4.0
        for row in rows_subset:
            x = left
            for stage in row["stages"]:
                w = usable * stage["ms"] / max_total
                svg.rect(x, y, w, 8.0, stage.get("color") or STAGE_COLORS[stage["stage"]],
                         hatch=stage.get("estimated", False))
                x += w
            svg.text(left - 3.0, y + 6.0, row["short"], size=6.0, anchor="end")
            svg.text(right + 3.0, y + 6.0, f"{row['total_ms']:.0f} ms", size=6.0, fill="#333333")
            y += 9.0
            if row.get("note"):
                svg.text(left, y + 2.6, row["note"], size=note_size, fill="#777777")
                y += 4.0
        # axis
        svg.line(left, y + 1.5, right, y + 1.5, stroke="#666666", sw=0.4)
        for i in range(5):
            value = max_total * i / 4.0
            xx = left + usable * i / 4.0
            svg.line(xx, y + 1.5, xx, y + 3.5, stroke="#666666", sw=0.35)
            svg.text(xx, y + 8.0, f"{value:.0f}", size=5.0, fill="#555555", anchor="middle")
        svg.text(left, y + 15.0, "milliseconds", size=5.4, fill="#555555")
        y += 24.0

    bars([r for r in rows if not r.get("detail_only")], max(r["total_ms"] for r in rows if not r.get("detail_only")),
         "(a) end-to-end wall clock, shared scale")
    hybrid = next(r for r in rows if r.get("detail_only"))
    bars([hybrid], hybrid["total_ms"], "(b) hybrid breakdown, own scale (7x magnification)")

    svg.height = y + 10.0 + (len(wrap_note(cost["footnote"])) - 1) * 3.6
    draw_note(svg, cost["footnote"])
    return svg.render()


# ----------------------------------------------------------------- figure 2
UNRELIABLE = {5}  # submits whose own submit aborted while being sampled


def render_timeline(timeline: dict) -> str:
    submits = [d for d in timeline["submit_detail"] if d["samples"]]
    modules = ("CNA", "CORE", "DPU", "RDMA")
    top = 6.0
    a_top = top + 26.0
    row_h = 10.0
    heat_h = row_h * len(modules)
    detailed = max(submits, key=lambda d: len(d["samples"]))
    b_top = a_top + heat_h + 40.0
    b_row_h = 10.0
    height = b_top + b_row_h * len(modules) + 34.0
    svg = Svg(height, f"GDN NPU front-end measured module activity ({timeline['model']})")

    svg.text(58.0, top + 5.0, "(a) measured S_STATUS busy share per submit x module", size=7.2, weight="600")
    # intensity key
    for i in range(5):
        share = i / 4.0
        x = 58.0 + i * 22.0
        svg.rect(x, top + 9.0, 20.0, 6.0, mix("#ffffff", "#1f5076", share ** 0.6), stroke="#dddddd", sw=0.25)
        svg.text(x + 10.0, top + 19.5, f"{share * 100:.0f}%", size=5.0, fill="#666666", anchor="middle")
    svg.text(58.0 + 5 * 22.0 + 6.0, top + 15.0,
             "share of edge samples in which the module reports operating (white = 0%, dark = 100%)",
             size=5.8, fill="#666666")

    left, right = 58.0, WIDTH - 12.0
    cell_w = (right - left) / len(submits)
    for m_idx, module in enumerate(modules):
        y = a_top + m_idx * row_h
        svg.text(left - 3.0, y + row_h - 3.0, module, size=6.0, anchor="end")
        for s_idx, sub in enumerate(submits):
            share = sum(1 for smp in sub["samples"] if smp["modules"].get(module)) / len(sub["samples"])
            x = left + s_idx * cell_w
            svg.rect(x + 0.3, y + 0.5, cell_w - 0.6, row_h - 1.4,
                     mix("#ffffff", "#1f5076", share ** 0.6), stroke="#e2e2e2", sw=0.25,
                     hatch=sub["submit_seq"] in UNRELIABLE)
            label = f"{share * 100:.0f}%" if share >= 0.25 else ""
            if label:
                svg.text(x + cell_w / 2.0, y + row_h - 3.2, label, size=5.0,
                         fill="#ffffff" if share > 0.55 else "#333333", anchor="middle")
    for s_idx, sub in enumerate(submits):
        x = left + s_idx * cell_w + cell_w / 2.0
        svg.text(x, a_top + heat_h + 7.0, f"S{sub['submit_seq']}", size=5.6, fill="#555555", anchor="middle")
        svg.text(x, a_top + heat_h + 13.0, f"{sub['tasks']}", size=4.6, fill="#999999", anchor="middle")
    svg.text(left, a_top + heat_h + 20.0,
             "submit index (chronological) / hardware tasks; hatched column S5 aborted its own submit under sampling",
             size=5.4, fill="#555555")

    svg.text(58.0, b_top - 6.0,
             f"(b) raw samples of S{detailed['submit_seq']}: {len(detailed['samples'])} edge samples, "
             f"{detailed['tasks']} tasks, {detailed['sample_span_ms']:.1f} ms sampled span",
             size=7.0, weight="600")
    span = max(detailed["sample_span_ms"], 1e-3)
    for m_idx, module in enumerate(modules):
        y = b_top + m_idx * b_row_h
        svg.text(left - 3.0, y + b_row_h - 3.0, module, size=6.0, anchor="end")
        svg.rect(left, y + 0.6, right - left, b_row_h - 2.2, "#f4f4f4")
        for smp in detailed["samples"]:
            if not smp["modules"].get(module):
                continue
            x = left + (right - left) * min(smp["t_ms"] / span, 1.0)
            svg.rect(x, y + 0.6, max((right - left) / len(detailed["samples"]) * 0.9, 0.4),
                     b_row_h - 2.2, "#333333")
    axis_y = b_top + b_row_h * len(modules) + 2.0
    svg.line(left, axis_y, right, axis_y, stroke="#666666", sw=0.4)
    for i in range(5):
        x = left + (right - left) * i / 4.0
        svg.line(x, axis_y, x, axis_y + 2.0, stroke="#666666", sw=0.35)
        svg.text(x, axis_y + 7.0, f"{span * i / 4.0:.1f}", size=5.2, fill="#555555", anchor="middle")
    svg.text(left, axis_y + 14.5,
             "sampled time inside the submit (ms); dark = S_STATUS reports operating", size=5.4, fill="#555555")
    svg.height = axis_y + 27.0 + (len(wrap_note(timeline["footnote"])) - 1) * 3.6
    draw_note(svg, timeline["footnote"])
    return svg.render()


def mix(a: str, b: str, t: float) -> str:
    va = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    vb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(va, vb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    cost = json.loads(args.cost.read_text(encoding="utf-8"))
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    timeline.setdefault("model", "qwen35_gdn_layer0 front-end")
    timeline.setdefault(
        "footnote",
        "S_STATUS sampled with EPERF_RKNPU_BLOCKWATCH_EDGE=1 / SELECT=5 / POLL_NS=20000; "
        "one submit (S5) failed its submit under sampling and is omitted.",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cost_svg = render_cost(cost)
    tl_svg = render_timeline(timeline)
    (args.out_dir / "gdn_stage_cost.svg").write_text(cost_svg, encoding="utf-8")
    (args.out_dir / "gdn_fe_module_timeline.svg").write_text(tl_svg, encoding="utf-8")
    if args.preview:
        try:
            import cairosvg
        except ImportError:
            print("cairosvg missing; no previews")
            return 0
        for svg_text, name in ((cost_svg, "gdn_stage_cost_preview.png"),
                               (tl_svg, "gdn_fe_module_timeline_preview.png")):
            cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(args.out_dir / name), scale=2.0)
    print("wrote gdn_stage_cost.svg and gdn_fe_module_timeline.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
