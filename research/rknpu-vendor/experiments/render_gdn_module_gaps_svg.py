#!/usr/bin/env python3
"""GDN busy/idle timelines using the SDPA reference's exact panel geometry.

(a) S14 full sampled window; (b) a 4 ms transition-rich S14 detail.
(c)/(d) projection and tail details, and an all-submit atlas, are companions.
All times are local to the separately sampled submit, never spliced across runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse the reference's typography, strokes and canvas constants directly.
from render_sdpa_module_timeline_svg import FONT, FRAME, H, LEFT, RIGHT, W, line, text

MODULES = ("CNA", "CORE", "DPU", "RDMA")
LABELS = {"CNA": "CNA", "CORE": "CORE", "DPU": "DPU", "RDMA": "DPU_RDMA"}
BUSY = "#1f5076"
GAP = "#fff0d8"


def rect(x, y, w, h, fill, stroke=None, width=0.3):
    # Unlike the reference helper, do not artificially widen narrow intervals.
    if w <= 0 or h <= 0:
        return ""
    edge = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ''
    return (f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" '
            f'fill="{fill}"{edge}/>')


def intervals(sub):
    """Zero-order hold within adjacent observations; never extend the last sample.

    Long missing intervals are unknown after one median interval. An isolated
    sample has no measured interval and is shown as a point, not a busy bar.
    """
    samples = sub['timeline']
    median = sub['interval_p50_us'] / 1000
    for a, b in zip(samples, samples[1:]):
        start, stop = a['t_ms'], b['t_ms']
        if stop <= start:
            raise ValueError('Sample times must increase')
        if median > 0 and stop - start > 3 * median:
            stop = start + median
        yield start, stop, a['modules']


def compute_gaps(sub):
    """Runs of held observations with both CNA and CORE reporting idle."""
    runs = []
    for a, b, modules in intervals(sub):
        if modules['CNA'] or modules['CORE']:
            continue
        if runs and abs(a - runs[-1][1]) < 1e-8:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def zoom_window(sub, width=4.0):
    """A reproducible transition-rich window, excluding submit start/finish."""
    samples = sub['timeline']
    span = sub['sampled_span_ms']
    width = min(width, span)
    candidates = [s['t_ms'] for s in samples if s['t_ms'] >= 0.5 and s['t_ms'] + width <= span - 0.5]
    if not candidates:
        return 0.0, span
    def score(start):
        window = [s for s in samples if start <= s['t_ms'] <= start + width]
        return sum(sum(a['modules'][m] != b['modules'][m] for m in MODULES)
                   for a, b in zip(window, window[1:]))
    start = max(candidates, key=score)
    return start, start + width


def figure(sub, start, end, tag, desc, zoom=None):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
        f'viewBox="0 0 {W} {H}" version="1.1">',
        f'<title>GDN module activity: S{sub["submit_seq"]}, {desc}</title>',
        '<defs><pattern id="unknown" width="3" height="3" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="3" height="3" fill="#f4f5f7"/>'
        '<line x1="0" y1="0" x2="0" y2="3" stroke="#c5cbd3" stroke-width="0.45"/>'
        '</pattern></defs>',
        rect(0, 0, W, H, '#ffffff'),
    ]
    # Reference legend anchors, text size, sample swatch size and serif font.
    for lx, label, color in [
        (LEFT, 'Operating', BUSY), (LEFT + 94, 'Idle / no bar', '#ffffff'),
        (LEFT + 199, 'CNA + CORE idle', GAP),
        (LEFT + 329, 'Unobserved', 'url(#unknown)'),
    ]:
        parts.append(rect(lx, 7.35, 7.2, 5.04, color, FRAME if color == '#ffffff' else None))
        parts.append(text(lx + 9.72, 12.39, label, size=7.2, fill='#333333', anchor='start'))
    parts.append(text(LEFT, 25.5,
        'Bars: fixed-interval S_STATUS, zero-order hold (not continuous); core 0 only.',
        size=5.8, fill='#666666', anchor='start'))
    parts.append(text(LEFT, 32.5,
        f'Poll median {sub["interval_p50_us"]:.1f} µs; sequential register reads. Idle is not proof of a stall.',
        size=5.8, fill='#666666', anchor='start'))

    strip_top, strip_bottom = 46.8, 56.8
    plot_top, plot_bottom = 59.8, 148.68
    row_h = (plot_bottom - plot_top) / 4
    tx = lambda t: LEFT + (t - start) / (end - start) * (RIGHT - LEFT)
    parts.append(text((LEFT + RIGHT) / 2, strip_top - 4.3, f'({tag}) {desc}', size=8))
    parts.append(rect(LEFT, strip_top, RIGHT - LEFT, strip_bottom - strip_top, '#f7f8fa', FRAME, 0.35))
    parts.append(text((LEFT + RIGHT) / 2, strip_bottom - 2.2,
        f'S{sub["submit_seq"]}  |  {sub["tasks"]:,} hardware tasks  |  {sub["samples"]} samples  |  independent capture',
        size=7, fill='#444444'))

    observed = list(intervals(sub))
    for j, module in enumerate(MODULES):
        top = plot_top + j * row_h
        cy = top + row_h / 2
        parts.append(rect(LEFT, top + 1, RIGHT - LEFT, row_h - 2, '#ffffff', FRAME, 0.5))
        parts.append(text(LEFT - 4, cy + 3.1, LABELS[module], size=9, anchor='end'))
        # Hatched areas are genuinely unobserved, including after the final poll.
        cursor = start
        for a, b, states in observed:
            lo, hi = max(start, a), min(end, b)
            if hi <= lo:
                continue
            if lo > cursor:
                parts.append(rect(tx(cursor), top + 1.4, tx(lo)-tx(cursor), row_h-2.8, 'url(#unknown)'))
            cursor = max(cursor, hi)
        if cursor < end:
            parts.append(rect(tx(cursor), top + 1.4, tx(end)-tx(cursor), row_h-2.8, 'url(#unknown)'))
        if j < 2:
            for a, b in compute_gaps(sub):
                lo, hi = max(start, a), min(end, b)
                if hi > lo:
                    parts.append(rect(tx(lo), top + 1.4, tx(hi)-tx(lo), row_h-2.8, GAP))

    # Same light grid and black baseline as the SDPA reference.
    for i in range(5):
        gx = LEFT + (RIGHT - LEFT) * i / 4
        parts.append(line(gx, plot_top, gx, plot_bottom, stroke='#eceff2', width=0.5))
    for j, module in enumerate(MODULES):
        cy = plot_top + (j + 0.5) * row_h
        busy_runs = []
        for a, b, states in observed:
            lo, hi = max(start, a), min(end, b)
            if hi <= lo or not states[module]:
                continue
            if busy_runs and abs(lo - busy_runs[-1][1]) < 1e-8:
                busy_runs[-1] = (busy_runs[-1][0], hi)
            else:
                busy_runs.append((lo, hi))
        for lo, hi in busy_runs:
            parts.append(rect(tx(lo), cy - 5, tx(hi)-tx(lo), 10, BUSY))
        # A lone observation cannot define an interval; show a point rather than a bar.
        if len(sub['timeline']) == 1:
            smp = sub['timeline'][0]
            if start <= smp['t_ms'] <= end:
                parts.append(line(tx(smp['t_ms']), cy-4, tx(smp['t_ms']), cy+4,
                                  stroke=BUSY if smp['modules'][module] else '#999999', width=0.5))
    if len(sub['timeline']) == 1:
        parts.append(text((LEFT + RIGHT)/2, 100, 'Single sample: gap / overlap unresolved', size=7))
    if zoom:
        # Bracket in the strip only: no overlay across measured module bars.
        lo, hi = zoom
        parts.append(line(tx(lo), strip_bottom-0.6, tx(hi), strip_bottom-0.6, stroke='#1f5076', width=1.2))
        for t in (lo, hi):
            parts.append(line(tx(t), strip_bottom-3, tx(t), strip_bottom, stroke='#1f5076', width=0.5))
    parts.append(line(LEFT, plot_bottom, RIGHT, plot_bottom, stroke='#000000', width=0.5))
    for i in range(5):
        gx = LEFT + (RIGHT - LEFT) * i / 4
        t = start + (end - start) * i / 4
        parts.append(line(gx, plot_bottom, gx, plot_bottom+3, stroke='#000000', width=0.8))
        parts.append(text(gx, plot_bottom+10.55, f'{t:.2f}', size=8, fill='#444444',
                          anchor='start' if i == 0 else 'end' if i == 4 else 'middle'))
    parts.append(text((LEFT+RIGHT)/2, plot_bottom+19.83, 'Time since submit entry (ms)', size=8))
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def combine(figures):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" '
             f'height="{len(figures)*H}pt" viewBox="0 0 {W} {len(figures)*H}">']
    for i, fig in enumerate(figures):
        fig = fig.replace('unknown', f'unknown{i}')
        fig = fig.replace(f'width="{W}pt" height="{H}pt"', f'width="{W}" height="{H}"', 1)
        parts.append(fig.replace('<svg ', f'<svg x="0" y="{i*H}" ', 1))
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def write_figure(svg, path, preview):
    path.write_text(svg, encoding='utf-8')
    if preview:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(),
                         write_to=str(path.with_name(path.stem + '_preview.png')), scale=2)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    submits = {s['submit_seq']: s for s in data['submit_detail']}
    sub = submits[14]
    zoom = zoom_window(sub)
    figures = [figure(sub, 0, sub['sampled_span_ms'], 'a', 'GDN delta-rule: full S14', zoom=zoom),
               figure(sub, *zoom, 'b', 'GDN delta-rule: 4 ms detail')]
    for tag, seq, desc in [('c', 1, 'Projection / conv'), ('d', 18, 'Tail')]:
        s = submits[seq]
        figures.append(figure(s, *zoom_window(s), tag, f'{desc}: 4 ms detail'))
    for tag, fig in zip('abcd', figures):
        write_figure(fig, args.out.with_name(f'{args.out.stem}_{tag}.svg'), args.preview)
    write_figure(combine(figures[:2]), args.out, args.preview)
    atlas = [figure(s, 0, max(s['sampled_span_ms'], 0.001), str(s['submit_seq']),
                    f'GDN submit S{s["submit_seq"]}') for s in data['submit_detail']]
    write_figure(combine(atlas), args.out.with_name(args.out.stem + '_all_submits.svg'), args.preview)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
